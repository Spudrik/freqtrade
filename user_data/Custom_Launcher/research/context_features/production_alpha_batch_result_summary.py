from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_DIR = ROOT / "user_data" / "research_news_data" / "context_features" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one or more strategies from a Freqtrade backtest ZIP.")
    parser.add_argument("--backtest-zip", type=Path, required=True)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    csv_path = args.output_dir / f"production_alpha_batch_summary_{tag}.csv"
    md_path = args.output_dir / f"production_alpha_batch_summary_{tag}.md"

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "backtest_zip": str(args.backtest_zip),
        "baseline_zip": str(args.baseline_zip) if args.baseline_zip else None,
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    baseline = load_first_strategy(args.baseline_zip) if args.baseline_zip else None
    rows = [summary_row(name, result, args.backtest_zip, baseline) for name, result in load_strategies(args.backtest_zip).items()]
    frame = pd.DataFrame(rows).sort_values(["verdict_rank", "return_pct"], ascending=[True, False])
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_report(frame, args.backtest_zip, baseline), encoding="utf-8")

    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(frame)}, indent=2))
    return 0


def load_strategies(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json") and not name.endswith("_config.json")]
        result_names = [name for name in names if "meta" not in Path(name).name]
        if len(result_names) != 1:
            raise ValueError(f"Expected one result JSON in {path}, found {result_names}")
        payload = json.loads(archive.read(result_names[0]))
    return payload.get("strategy", {})


def load_first_strategy(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    strategies = load_strategies(path)
    if not strategies:
        return None
    name, result = next(iter(strategies.items()))
    return {"name": name, **result}


def summary_row(name: str, result: dict[str, Any], source_zip: Path, baseline: dict[str, Any] | None) -> dict[str, Any]:
    trades = result.get("trades", [])
    long_trades = sum(1 for trade in trades if not trade.get("is_short"))
    short_trades = sum(1 for trade in trades if trade.get("is_short"))
    total_trades = int(result.get("total_trades") or len(trades))
    min_side_ratio = min(long_trades, short_trades) / total_trades if total_trades else 0.0
    return_pct = float(result.get("profit_total") or 0.0) * 100.0
    market_pct = float(result.get("market_change") or 0.0) * 100.0
    drawdown_pct = float(result.get("max_drawdown_account") or 0.0) * 100.0
    profit_factor = float(result.get("profit_factor") or 0.0)
    baseline_return = float(baseline.get("profit_total") or 0.0) * 100.0 if baseline else None
    baseline_drawdown = float(baseline.get("max_drawdown_account") or 0.0) * 100.0 if baseline else None
    verdict = verdict_text(
        return_pct=return_pct,
        market_pct=market_pct,
        drawdown_pct=drawdown_pct,
        profit_factor=profit_factor,
        total_trades=total_trades,
        min_side_ratio=min_side_ratio,
        baseline_return=baseline_return,
        baseline_drawdown=baseline_drawdown,
    )
    return {
        "strategy": name,
        "source_zip": str(source_zip),
        "start": result.get("backtest_start"),
        "end": result.get("backtest_end"),
        "trades": total_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "min_side_ratio_pct": min_side_ratio * 100.0,
        "win_rate_pct": float(result.get("winrate") or 0.0) * 100.0,
        "return_pct": return_pct,
        "market_change_pct": market_pct,
        "return_minus_market_pct": return_pct - market_pct,
        "max_drawdown_pct": drawdown_pct,
        "profit_factor": profit_factor,
        "baseline_return_pct": baseline_return,
        "baseline_drawdown_pct": baseline_drawdown,
        "return_delta_pct": return_pct - baseline_return if baseline_return is not None else None,
        "drawdown_delta_pct": drawdown_pct - baseline_drawdown if baseline_drawdown is not None else None,
        "verdict": verdict,
        "verdict_rank": 0 if verdict.startswith("Promising") else 1 if verdict.startswith("Mixed") else 2,
    }


def verdict_text(
    *,
    return_pct: float,
    market_pct: float,
    drawdown_pct: float,
    profit_factor: float,
    total_trades: int,
    min_side_ratio: float,
    baseline_return: float | None,
    baseline_drawdown: float | None,
) -> str:
    failures: list[str] = []
    if market_pct and return_pct <= market_pct:
        failures.append("does not beat buy-and-hold")
    if baseline_return is not None and return_pct < baseline_return:
        failures.append("return trails baseline")
    if baseline_drawdown is not None and drawdown_pct > baseline_drawdown:
        failures.append("drawdown worsens")
    if total_trades < 50:
        failures.append("too few trades")
    if min_side_ratio < 0.20:
        failures.append("long/short balance below 80/20 gate")
    if profit_factor < 1.5:
        failures.append("profit factor weak")
    if not failures:
        return "Promising: passes current headline gates."
    if len(failures) <= 2:
        return "Mixed: " + "; ".join(failures) + "."
    return "Weak: " + "; ".join(failures) + "."


def markdown_report(frame: pd.DataFrame, source_zip: Path, baseline: dict[str, Any] | None) -> str:
    lines = [
        "# Production Alpha Batch Summary",
        "",
        f"- Source ZIP: `{source_zip}`",
    ]
    if baseline:
        lines.append(f"- Baseline: `{baseline['name']}`")
    lines.extend([
        "",
        "## Acceptance Gates",
        "",
        "- Beat same-window buy-and-hold where practical.",
        "- Do not worsen drawdown unless return/risk clearly justifies it.",
        "- Keep enough trades to be meaningful.",
        "- Keep at least an 80/20 long/short balance either way.",
        "- Prefer family-specific exits over generic exits unless crash/reversal confluence is broad.",
        "",
        "## Results",
        "",
        "| # | Strategy | Trades | Long/Short | Return | Market | Drawdown | PF | Verdict |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for index, row in enumerate(frame.to_dict("records"), start=1):
        lines.append(
            "| {idx} | `{strategy}` | {trades} | {long}/{short} | {ret:.2f}% | {market:.2f}% | {dd:.2f}% | {pf:.2f} | {verdict} |".format(
                idx=index,
                strategy=row["strategy"],
                trades=int(row["trades"]),
                long=int(row["long_trades"]),
                short=int(row["short_trades"]),
                ret=float(row["return_pct"]),
                market=float(row["market_change_pct"]),
                dd=float(row["max_drawdown_pct"]),
                pf=float(row["profit_factor"]),
                verdict=row["verdict"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
