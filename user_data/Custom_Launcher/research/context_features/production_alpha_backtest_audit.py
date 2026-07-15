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
    parser = argparse.ArgumentParser(description="Create promotion/failure audit files for a Freqtrade backtest ZIP.")
    parser.add_argument("--backtest-zip", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "backtest_zip": str(args.backtest_zip),
        "baseline_zip": str(args.baseline_zip) if args.baseline_zip else None,
        "outputs": str(args.output_dir),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    audited = load_backtest(args.backtest_zip)
    baseline = load_backtest(args.baseline_zip) if args.baseline_zip else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)

    summary = pd.DataFrame([summary_row(audited, "audited")])
    if baseline is not None:
        summary = pd.concat([summary, pd.DataFrame([summary_row(baseline, "baseline")])], ignore_index=True)

    trades = pd.DataFrame(audited["trades"])
    trades = normalize_trades(trades)
    years = periodic_frame(audited, "year")
    months = periodic_frame(audited, "month")
    worst_trades = trades.sort_values("profit_abs").head(30)
    family = family_audit(trades)

    summary_path = args.output_dir / f"production_alpha_{tag}_promotion_gate.csv"
    years_path = args.output_dir / f"production_alpha_{tag}_years.csv"
    months_path = args.output_dir / f"production_alpha_{tag}_worst_months.csv"
    worst_trades_path = args.output_dir / f"production_alpha_{tag}_worst_trades.csv"
    family_path = args.output_dir / f"production_alpha_{tag}_worst_families.csv"
    md_path = args.output_dir / f"production_alpha_{tag}_promotion_gate.md"

    summary.to_csv(summary_path, index=False)
    years.to_csv(years_path, index=False)
    months.sort_values("profit_abs").head(30).to_csv(months_path, index=False)
    worst_trades.to_csv(worst_trades_path, index=False)
    family.to_csv(family_path, index=False)
    md_path.write_text(markdown_report(audited, baseline, years, months, family, worst_trades), encoding="utf-8")

    print(json.dumps({
        "summary": str(summary_path),
        "years": str(years_path),
        "worst_months": str(months_path),
        "worst_trades": str(worst_trades_path),
        "worst_families": str(family_path),
        "markdown": str(md_path),
    }, indent=2))
    return 0


def load_backtest(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("path is required")
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        json_names = [name for name in archive.namelist() if name.endswith(".json") and not name.endswith("_config.json")]
        if len(json_names) != 1:
            raise ValueError(f"Expected one result JSON in {path}, found {json_names}")
        payload = json.loads(archive.read(json_names[0]))
    strategies = payload.get("strategy", {})
    if len(strategies) != 1:
        raise ValueError(f"Expected one strategy in {path}, found {list(strategies)}")
    strategy_name, result = next(iter(strategies.items()))
    result["strategy_name_from_key"] = strategy_name
    result["source_zip"] = str(path)
    return result


def normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    for col in ("open_date", "close_date"):
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")
    trades["direction"] = trades["is_short"].map({True: "short", False: "long"}).fillna("unknown")
    trades["open_year"] = trades["open_date"].dt.year
    trades["open_month"] = trades["open_date"].dt.to_period("M").astype(str)
    trades["win"] = trades["profit_abs"] > 0
    return trades


def summary_row(result: dict[str, Any], role: str) -> dict[str, Any]:
    market_change = float(result.get("market_change") or 0.0)
    profit_total = float(result.get("profit_total") or 0.0)
    return {
        "role": role,
        "strategy": result.get("strategy_name") or result.get("strategy_name_from_key"),
        "source_zip": result.get("source_zip"),
        "trades": int(result.get("total_trades") or 0),
        "return_pct": profit_total * 100.0,
        "profit_abs": float(result.get("profit_total_abs") or 0.0),
        "max_drawdown_pct": float(result.get("max_drawdown_account") or 0.0) * 100.0,
        "max_drawdown_abs": float(result.get("max_drawdown_abs") or 0.0),
        "profit_factor": float(result.get("profit_factor") or 0.0),
        "win_rate_pct": float(result.get("winrate") or 0.0) * 100.0,
        "wins": int(result.get("wins") or 0),
        "losses": int(result.get("losses") or 0),
        "market_change_pct": market_change * 100.0,
        "return_vs_market_ratio": profit_total / market_change if market_change else None,
        "start": result.get("backtest_start"),
        "end": result.get("backtest_end"),
    }


def periodic_frame(result: dict[str, Any], period: str) -> pd.DataFrame:
    rows = result.get("periodic_breakdown", {}).get(period, [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["win_rate_pct"] = (frame["wins"] / frame["trades"].replace(0, pd.NA) * 100.0).fillna(0.0)
    return frame


def family_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    rows: list[dict[str, Any]] = []
    for tag, group in trades.groupby("enter_tag", dropna=False):
        wins = group.loc[group["profit_abs"] > 0, "profit_abs"].sum()
        losses = -group.loc[group["profit_abs"] < 0, "profit_abs"].sum()
        rows.append({
            "enter_tag": tag,
            "trades": int(len(group)),
            "profit_abs": float(group["profit_abs"].sum()),
            "mean_profit_ratio_pct": float(group["profit_ratio"].mean() * 100.0),
            "win_rate_pct": float(group["win"].mean() * 100.0),
            "profit_factor": float(wins / losses) if losses else None,
            "worst_trade_abs": float(group["profit_abs"].min()),
            "best_trade_abs": float(group["profit_abs"].max()),
            "long_trades": int((group["direction"] == "long").sum()),
            "short_trades": int((group["direction"] == "short").sum()),
        })
    return pd.DataFrame(rows).sort_values(["profit_abs", "trades"], ascending=[True, False])


def markdown_report(
    audited: dict[str, Any],
    baseline: dict[str, Any] | None,
    years: pd.DataFrame,
    months: pd.DataFrame,
    family: pd.DataFrame,
    worst_trades: pd.DataFrame,
) -> str:
    audited_row = summary_row(audited, "audited")
    baseline_row = summary_row(baseline, "baseline") if baseline is not None else None
    full_years = full_year_frame(years, audited)
    positive_years = int((full_years["profit_abs"] > 0).sum()) if not full_years.empty else 0
    total_years = int(len(full_years))
    partial_years = int(len(years) - len(full_years)) if not years.empty else 0
    worst_month = months.sort_values("profit_abs").head(1).to_dict("records")[0] if not months.empty else {}
    worst_family = family.head(1).to_dict("records")[0] if not family.empty else {}
    worst_trade = worst_trades.head(1).to_dict("records")[0] if not worst_trades.empty else {}

    lines = [
        "# Production Alpha Promotion Gate",
        "",
        "## Trader Question",
        "",
        "Does this branch deserve to become the current production-alpha research baseline before adding more entry families, exits, or risk overlays?",
        "",
        "## Audited Branch",
        "",
        f"- Strategy: `{audited_row['strategy']}`",
        f"- Trades: `{audited_row['trades']}`",
        f"- Return: `{audited_row['return_pct']:.2f}%`",
        f"- Max drawdown: `{audited_row['max_drawdown_pct']:.2f}%`",
        f"- Profit factor: `{audited_row['profit_factor']:.2f}`",
        f"- Win rate: `{audited_row['win_rate_pct']:.1f}%`",
        f"- Market change: `{audited_row['market_change_pct']:.2f}%`",
        f"- Return / market-change ratio: `{audited_row['return_vs_market_ratio']:.2f}x`",
        f"- Positive completed years: `{positive_years}/{total_years}`",
        f"- Partial edge-year buckets excluded from full-year gate: `{partial_years}`",
        "",
    ]
    if baseline_row is not None:
        lines.extend([
            "## Baseline Comparison",
            "",
            f"- Baseline strategy: `{baseline_row['strategy']}`",
            f"- Baseline return: `{baseline_row['return_pct']:.2f}%`",
            f"- Baseline drawdown: `{baseline_row['max_drawdown_pct']:.2f}%`",
            f"- Baseline profit factor: `{baseline_row['profit_factor']:.2f}`",
            f"- Return delta: `{audited_row['return_pct'] - baseline_row['return_pct']:.2f}` percentage points",
            f"- Drawdown delta: `{audited_row['max_drawdown_pct'] - baseline_row['max_drawdown_pct']:.2f}` percentage points",
            f"- Profit-factor delta: `{audited_row['profit_factor'] - baseline_row['profit_factor']:.2f}`",
            "",
        ])
    lines.extend([
        "## Failure Notes",
        "",
        f"- Worst month by absolute profit: `{worst_month.get('date')}` with `{float(worst_month.get('profit_abs', 0.0)):.2f}` profit.",
        f"- Worst family by absolute profit: `{worst_family.get('enter_tag')}` with `{float(worst_family.get('profit_abs', 0.0)):.2f}` profit across `{int(worst_family.get('trades', 0))}` trades.",
        f"- Worst trade: `{worst_trade.get('enter_tag')}` opened `{worst_trade.get('open_date')}` with `{float(worst_trade.get('profit_abs', 0.0)):.2f}` profit.",
        "",
        "## Verdict",
        "",
        verdict(audited_row, baseline_row, positive_years, total_years),
        "",
    ])
    return "\n".join(lines)


def verdict(audited: dict[str, Any], baseline: dict[str, Any] | None, positive_years: int, total_years: int) -> str:
    reasons: list[str] = []
    if baseline is not None and audited["return_pct"] < baseline["return_pct"]:
        secondary: list[str] = []
        if audited["max_drawdown_pct"] <= baseline["max_drawdown_pct"]:
            secondary.append("drawdown improved versus baseline")
        if audited["profit_factor"] >= baseline["profit_factor"]:
            secondary.append("profit factor improved versus baseline")
        if total_years and positive_years == total_years:
            secondary.append("all completed yearly buckets were positive")
        if audited["return_vs_market_ratio"] and audited["return_vs_market_ratio"] > 1.0:
            secondary.append("return beat same-window market change")
        suffix = f" Supporting reasons found: {'; '.join(secondary)}." if secondary else ""
        return "Do not replace the current return baseline because return underperformed the baseline." + suffix
    if baseline is not None and audited["return_pct"] > baseline["return_pct"]:
        reasons.append("return improved versus baseline")
    if baseline is not None and audited["max_drawdown_pct"] <= baseline["max_drawdown_pct"]:
        reasons.append("drawdown did not worsen versus baseline")
    if baseline is not None and audited["profit_factor"] >= baseline["profit_factor"]:
        reasons.append("profit factor improved versus baseline")
    if total_years and positive_years == total_years:
        reasons.append("all completed yearly buckets were positive")
    if audited["return_vs_market_ratio"] and audited["return_vs_market_ratio"] > 1.0:
        reasons.append("return beat same-window market change")
    if len(reasons) >= 4:
        return "Promote as the current research baseline, while keeping failure-mode work open: " + "; ".join(reasons) + "."
    return "Do not promote without more work. Supporting reasons found: " + ("; ".join(reasons) if reasons else "none") + "."


def full_year_frame(years: pd.DataFrame, result: dict[str, Any]) -> pd.DataFrame:
    if years.empty:
        return years
    start = pd.to_datetime(result.get("backtest_start"), utc=True, errors="coerce")
    end = pd.to_datetime(result.get("backtest_end"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return years
    frame = years.copy()
    frame["period_end"] = pd.to_datetime(frame["date"], dayfirst=True, utc=True, errors="coerce")
    frame["period_start"] = frame["period_end"].dt.to_period("Y").dt.start_time.dt.tz_localize("UTC")
    return frame.loc[(frame["period_start"] >= start.floor("D")) & (frame["period_end"] <= end.floor("D"))].drop(columns=["period_start", "period_end"])


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
