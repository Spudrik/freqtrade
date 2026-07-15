from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = USER_DATA_DIR.parent
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
SIEVE_RESULTS_DIR = USER_DATA_DIR / "Custom_Launcher" / "launcher_v2" / "runtime" / "entry_sieve" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract promising entry-sieve runs into trader-readable lead candidates.")
    parser.add_argument("--results-dir", type=Path, default=SIEVE_RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-return", type=float, default=0.0)
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--pair-required", action="store_true")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    candidates = collect_candidates(
        results_dir=args.results_dir,
        min_trades=args.min_trades,
        min_profit_factor=args.min_profit_factor,
        min_return=args.min_return,
        pair=args.pair,
        pair_required=args.pair_required,
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_dir / "trader_sieve_runtime_candidate_leads.csv"
    tagged_csv_path = args.reports_dir / f"trader_sieve_runtime_candidate_leads_{safe_name(args.tag)}.csv"
    md_path = args.reports_dir / f"trader_sieve_runtime_candidate_leads_{safe_name(args.tag)}.md"
    meta_path = args.reports_dir / f"trader_sieve_runtime_candidate_leads_{safe_name(args.tag)}_meta.json"

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        frame = frame.sort_values(["candidate_score", "strategy"], ascending=[False, True]).drop_duplicates("strategy", keep="first")
    frame.to_csv(csv_path, index=False)
    frame.to_csv(tagged_csv_path, index=False)
    md_path.write_text(markdown_report(frame, args), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(args.results_dir),
        "candidate_count": int(len(frame)),
        "min_trades": args.min_trades,
        "min_profit_factor": args.min_profit_factor,
        "min_return": args.min_return,
        "pair": args.pair,
        "pair_required": bool(args.pair_required),
        "outputs": {
            "csv": str(csv_path),
            "tagged_csv": str(tagged_csv_path),
            "markdown": str(md_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def collect_candidates(
    results_dir: Path,
    min_trades: int,
    min_profit_factor: float,
    min_return: float,
    pair: str,
    pair_required: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for record in iter_jsonl(path):
            if not is_candidate(record, min_trades=min_trades, min_profit_factor=min_profit_factor, min_return=min_return):
                continue
            row = candidate_row(record, path, pair=pair)
            if pair_required:
                if intish(row.get("pair_trade_count")) < min_trades:
                    continue
                if floatish(row.get("pair_profit_total")) <= min_return:
                    continue
            rows.append(row)
    return rows


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def is_candidate(record: dict[str, Any], min_trades: int, min_profit_factor: float, min_return: float) -> bool:
    if text(record.get("status")).lower() != "ok":
        return False
    trades = intish(record.get("trade_count"))
    if trades < min_trades:
        return False
    total_return = floatish(record.get("profit_total"))
    profit_factor = floatish(record.get("profit_factor"))
    winrate = floatish(record.get("winrate"))
    if total_return <= min_return:
        return False
    return profit_factor >= min_profit_factor or winrate >= 0.55


def candidate_row(record: dict[str, Any], evidence_path: Path, pair: str) -> dict[str, Any]:
    strategy = text(record.get("strategy"))
    direction = text(record.get("side")) or infer_direction(strategy)
    pair_stats = pair_trade_stats(Path(text(record.get("backtest_file"))), pair)
    trades = intish(pair_stats.get("trade_count")) or intish(record.get("trade_count"))
    winrate = floatish(pair_stats.get("winrate")) or floatish(record.get("winrate"))
    total_return = floatish(pair_stats.get("profit_total")) or floatish(record.get("profit_total"))
    profit_factor = floatish(pair_stats.get("profit_factor")) or floatish(record.get("profit_factor"))
    drawdown = floatish(pair_stats.get("max_drawdown_pct")) or floatish(record.get("max_drawdown_pct"))
    candidate_score = score_candidate(trades, winrate, total_return, profit_factor, drawdown)
    question, market_state = describe_strategy(strategy, direction)
    return {
        "strategy": strategy,
        "strategy_class": text(record.get("strategy_class")),
        "direction": direction,
        "branch": "sieve_borrowed",
        "trader_question": question,
        "visible_market_state": market_state,
        "source_detail_blocks": source_detail_blocks(strategy),
        "feature_columns": strategy,
        "target_or_outcome": "Entry-sieve validation return, win rate, drawdown, and profit factor.",
        "trade_count": trades,
        "winrate": winrate,
        "profit_total": total_return,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown,
        "pair": pair,
        "pair_trade_count": intish(pair_stats.get("trade_count")),
        "pair_winrate": floatish(pair_stats.get("winrate")),
        "pair_profit_total": floatish(pair_stats.get("profit_total")),
        "pair_profit_factor": floatish(pair_stats.get("profit_factor")),
        "pair_max_drawdown_pct": floatish(pair_stats.get("max_drawdown_pct")),
        "training_timerange": text(record.get("training_timerange")),
        "validation_timerange": text(record.get("validation_timerange")),
        "take_profit_pct": text(record.get("take_profit_pct")),
        "stoploss_pct": text(record.get("stoploss_pct")),
        "strategy_batch": text(record.get("strategy_batch")),
        "core_behavior": text(record.get("core_behavior")),
        "backtest_file": text(record.get("backtest_file")),
        "params_file": text(record.get("params_file")),
        "hyperopt_file": text(record.get("hyperopt_file")),
        "candidate_score": candidate_score,
        "evidence_paths": str(evidence_path),
        "next_action": "Borrow as a lead seed only: retest as a direct trader rule, then apply confluence filters and tailored exits before strategy-block merging.",
    }


def score_candidate(trades: int, winrate: float, total_return: float, profit_factor: float, drawdown: float) -> float:
    score = 80.0
    score += min(trades, 80) * 0.45
    score += max(winrate - 0.50, 0.0) * 80.0
    score += max(total_return, 0.0) * 650.0
    score += min(max(profit_factor - 1.0, 0.0), 4.0) * 12.0
    score -= max(drawdown, 0.0) * 140.0
    return round(score, 3)


def pair_trade_stats(backtest_file: Path, pair: str) -> dict[str, float | int]:
    if not backtest_file.exists():
        return {}
    try:
        with zipfile.ZipFile(backtest_file) as archive:
            result_name = first_result_json_name(archive)
            if not result_name:
                return {}
            data = json.loads(archive.read(result_name))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return {}

    strategies = data.get("strategy", {})
    if not strategies:
        return {}
    strategy_result = next(iter(strategies.values()))
    trades = [trade for trade in strategy_result.get("trades", []) if text(trade.get("pair")) == pair]
    if not trades:
        return {"trade_count": 0, "winrate": 0.0, "profit_total": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}

    returns = [floatish(trade.get("profit_ratio")) for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    compounded = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        compounded *= 1.0 + value
        peak = max(peak, compounded)
        if peak > 0:
            max_drawdown = min(max_drawdown, compounded / peak - 1.0)
    return {
        "trade_count": len(trades),
        "winrate": len(wins) / len(trades),
        "profit_total": compounded - 1.0,
        "profit_factor": gross_win / gross_loss if gross_loss else (999.0 if gross_win > 0 else 0.0),
        "max_drawdown_pct": abs(max_drawdown),
    }


def first_result_json_name(archive: zipfile.ZipFile) -> str:
    names = []
    for name in archive.namelist():
        if not name.endswith(".json"):
            continue
        if "_config" in name:
            continue
        stem = Path(name).stem
        if re.search(r"_[A-Z][A-Za-z0-9]+$", stem):
            continue
        names.append(name)
    if names:
        return sorted(names)[0]
    fallback = [name for name in archive.namelist() if name.endswith(".json") and "_config" not in name]
    return sorted(fallback)[0] if fallback else ""


def describe_strategy(strategy: str, direction: str) -> tuple[str, str]:
    words = strategy.replace("sieve3_novel_mtf_", "").replace("sieve2_", "").replace("_", " ")
    timeframe = "daily" if "_d1_" in strategy else "4h" if "_h4_" in strategy else "multi-timeframe"
    side = "long" if direction == "long" else "short" if direction == "short" else "trade"
    if "prior_high_break" in strategy:
        return (
            f"When price breaks a prior high on {timeframe} context, does the {side} follow through?",
            f"Price is breaking above a prior high with {timeframe} structure context.",
        )
    if "prior_low_break" in strategy:
        return (
            f"When price breaks a prior low on {timeframe} context, does the {side} follow through?",
            f"Price is breaking below a prior low with {timeframe} structure context.",
        )
    if "range_expansion" in strategy:
        return (
            f"When a {timeframe} range expands and 1h structure confirms, does the {side} continue?",
            f"Market leaves a range and lower-timeframe structure is expected to confirm continuation.",
        )
    if "compression" in strategy:
        return (
            f"When compression releases on {timeframe} context, does the {side} continue?",
            f"Price was compressed, then breaks out or down with lower-timeframe confirmation.",
        )
    if "failed_low_reclaim" in strategy or "failed_high_reject" in strategy:
        return (
            f"When a failed break/reclaim pattern appears on {timeframe} context, does the {side} work?",
            f"Price briefly fails a level and then reclaims/rejects it, suggesting a trap or reversal.",
        )
    if "bos" in strategy or "choch" in strategy or "hhhl" in strategy or "lhll" in strategy:
        return (
            f"When market structure shifts on {timeframe} context, does the {side} continue?",
            f"Market structure is changing or trending and the rule asks whether continuation follows.",
        )
    return (
        f"Does the sieve rule '{words}' produce a repeatable {side} entry?",
        f"Sieve-derived market state: {words}.",
    )


def source_detail_blocks(strategy: str) -> str:
    blocks = ["OHLCV", "Sieve-derived custom indicator rule"]
    lower = strategy.lower()
    if "vp" in lower or "vah" in lower or "val" in lower:
        blocks.append("custom VP/value-area")
    if "tlv2" in lower or "support" in lower or "resistance" in lower:
        blocks.append("custom TLV2/support-resistance")
    if any(token in lower for token in ["rectangle", "triangle", "wedge", "compression", "pattern"]):
        blocks.append("custom pattern/compression geometry")
    if any(token in lower for token in ["bos", "choch", "hhhl", "lhll"]):
        blocks.append("custom market structure")
    if "volume" in lower or "pressure" in lower:
        blocks.append("volume/pressure")
    return "; ".join(dict.fromkeys(blocks))


def markdown_report(frame: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# Sieve Runtime Candidate Leads",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Purpose: borrow plausible good entry-sieve trades as lead seeds for the trader-confluence lifecycle.",
        "",
        "These are not final trading rules. Each candidate must still be retested in the lead/confluence/exit/risk pipeline.",
        "",
        "## Filters",
        "",
        f"- Minimum trades: {args.min_trades}",
        f"- Minimum profit factor or win-rate gate: profit factor >= {args.min_profit_factor}, or win rate >= 55%",
        f"- Minimum return: > {args.min_return}",
        f"- Pair scored: {args.pair}",
        f"- Pair required: {bool(args.pair_required)}",
        "",
        f"Candidates: {len(frame)}",
        "",
    ]
    if frame.empty:
        return "\n".join(lines) + "\n"
    cols = [
        "strategy",
        "direction",
        "trade_count",
        "winrate",
        "profit_total",
        "profit_factor",
        "max_drawdown_pct",
        "pair_trade_count",
        "pair_profit_total",
        "candidate_score",
        "trader_question",
    ]
    lines.append(frame[cols].head(60).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:180] or "unknown"


def infer_direction(value: str) -> str:
    lower = value.lower()
    if "short" in lower or "breakdown" in lower or "reject" in lower or "bear" in lower:
        return "short"
    if "long" in lower or "breakout" in lower or "reclaim" in lower or "bull" in lower:
        return "long"
    return "both"


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def floatish(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def intish(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
