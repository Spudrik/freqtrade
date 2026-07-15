from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from trader_rule_backtests import fmt_pct, num, resolve_exit, safe_name, summarize_trades


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"


@dataclass(frozen=True)
class LeadSpec:
    lead_id: str
    source_file: str
    priority: int
    trader_question: str
    visible_market_state: str
    stop_loss: float
    take_profit: float
    hold_hours: int


LEADS = (
    LeadSpec(
        "lead__validated__val_vacuum_no_support_reclaim_warning_short",
        "trading_lead_signals_20260605_val_vacuum_no_reclaim.parquet",
        1,
        "Does the cleanest no-reclaim VAL/vacuum quick short work?",
        "Support was removed, price broke below value, downside is thin, and broken support is not being reclaimed.",
        0.014,
        0.025,
        6,
    ),
    LeadSpec(
        "lead__validated__support_break_support_cleared_short",
        "trading_lead_signals_support_removal_support_cleared_20260605.parquet",
        2,
        "Does support-cleared breakdown continue lower?",
        "Support or a recent low broke and the orderbook shows support has cleared.",
        0.025,
        0.050,
        18,
    ),
    LeadSpec(
        "lead__validated__val_vacuum_bearish_pressure_short",
        "trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_pressure_gte_20.parquet",
        3,
        "Does VAL/vacuum short improve when sell pressure remains active?",
        "Price broke below value after support removal, downside is thin, and bearish pressure persists.",
        0.010,
        0.035,
        8,
    ),
    LeadSpec(
        "lead__validated__val_vacuum_loose_no_reclaim_short",
        "trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_no_reclaim_lte_46.parquet",
        4,
        "Can the strict no-reclaim short be widened slightly?",
        "Same no-reclaim VAL/vacuum story, but with a looser support-reclaim warning threshold.",
        0.010,
        0.035,
        8,
    ),
    LeadSpec(
        "lead__validated__val_break_downside_vacuum_after_support_removed_short",
        "trading_lead_signals_support_removal_val_vacuum_20260605.parquet",
        5,
        "Does the broad VAL/vacuum support-removal short work?",
        "Price broke below value after support removal and downside liquidity is thin.",
        0.014,
        0.025,
        6,
    ),
)

CORE_LEAD_IDS = {
    "lead__validated__val_vacuum_no_support_reclaim_warning_short",
    "lead__validated__support_break_support_cleared_short",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check overlap and combined behaviour for validated support-removal leads.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_validated_overlap"))
    parser.add_argument("--near-hours", type=int, default=8)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    features = load_features(args.features)
    signals = load_all_signals(args.reports_dir)
    overlap = overlap_matrix(signals, int(args.near_hours))
    individual_summary, individual_trades = simulate_individual(features, signals)
    combined_trades = simulate_combined(features, signals)
    core_signals = signals[signals["lead_id"].isin(CORE_LEAD_IDS)].copy()
    core_trades = simulate_combined(features, core_signals)
    combined_summary = summarize_trades(
        combined_trades,
        "validated_orderbook_support_removal_combined",
        "orderbook_state",
        "short",
        "Can the validated support-removal shorts be combined without duplicating the same entries?",
        "Take the highest-priority active validated lead, then stay flat until that trade exits.",
        features,
    )
    core_summary = summarize_trades(
        core_trades,
        "validated_orderbook_core_support_cleared_plus_no_reclaim",
        "orderbook_state",
        "short",
        "Does the genuinely separate support-cleared plus strict no-reclaim block work better than adding all nested VAL/vacuum variants?",
        "Combine support-cleared breakdowns with strict no-reclaim VAL/vacuum shorts; skip nested broad VAL/vacuum leftovers.",
        features,
    )
    combined_signals = exportable_combined_signals(combined_trades)
    core_export_signals = exportable_combined_signals(core_trades)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    overlap_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_overlap.csv"
    individual_summary_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_individual_summary.csv"
    individual_trades_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_individual_trades.csv"
    combined_trades_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_combined_trades.csv"
    core_trades_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_core_trades.csv"
    combined_signals_path = args.reports_dir / f"trading_lead_signals_{tag}_combined_validated_orderbook.parquet"
    core_signals_path = args.reports_dir / f"trading_lead_signals_{tag}_core_validated_orderbook.parquet"
    markdown_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}.md"
    meta_path = args.reports_dir / f"trading_lead_validated_overlap_{tag}_meta.json"

    overlap.to_csv(overlap_path, index=False)
    individual_summary.to_csv(individual_summary_path, index=False)
    individual_trades.to_csv(individual_trades_path, index=False)
    combined_trades.to_csv(combined_trades_path, index=False)
    core_trades.to_csv(core_trades_path, index=False)
    combined_signals.to_parquet(combined_signals_path, index=False)
    core_export_signals.to_parquet(core_signals_path, index=False)
    markdown_path.write_text(markdown_report(overlap, individual_summary, combined_summary, core_summary, combined_signals_path, core_signals_path), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_count": len(LEADS),
        "raw_signal_rows": int(len(signals)),
        "combined_trades": int(combined_summary.get("trades", 0)),
        "combined_return": combined_summary.get("total_return"),
        "combined_profit_factor": combined_summary.get("profit_factor"),
        "core_trades": int(core_summary.get("trades", 0)),
        "core_return": core_summary.get("total_return"),
        "core_profit_factor": core_summary.get("profit_factor"),
        "outputs": {
            "overlap": str(overlap_path),
            "individual_summary": str(individual_summary_path),
            "individual_trades": str(individual_trades_path),
            "combined_trades": str(combined_trades_path),
            "core_trades": str(core_trades_path),
            "combined_signals": str(combined_signals_path),
            "core_signals": str(core_signals_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            raise KeyError(f"Missing OHLCV column: {col}")
    return frame


def load_all_signals(reports_dir: Path) -> DataFrame:
    frames: list[DataFrame] = []
    for spec in LEADS:
        path = reports_dir / spec.source_file
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first")
        frame["lead_id"] = spec.lead_id
        frame["priority"] = spec.priority
        frame["direction"] = "short"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "priority"]).reset_index(drop=True)


def overlap_matrix(signals: DataFrame, near_hours: int) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for left in LEADS:
        left_dates = lead_dates(signals, left.lead_id)
        for right in LEADS:
            right_dates = lead_dates(signals, right.lead_id)
            exact = len(set(left_dates) & set(right_dates))
            near = near_overlap_count(left_dates, right_dates, near_hours)
            rows.append(
                {
                    "left_lead_id": left.lead_id,
                    "right_lead_id": right.lead_id,
                    "left_signals": len(left_dates),
                    "right_signals": len(right_dates),
                    "exact_overlap": exact,
                    "near_overlap": near,
                    "exact_overlap_share_of_left": exact / len(left_dates) if left_dates else 0.0,
                    "near_overlap_share_of_left": near / len(left_dates) if left_dates else 0.0,
                }
            )
    return pd.DataFrame(rows)


def lead_dates(signals: DataFrame, lead_id: str) -> list[pd.Timestamp]:
    return list(pd.to_datetime(signals.loc[signals["lead_id"].eq(lead_id), "date"], utc=True).sort_values())


def near_overlap_count(left_dates: list[pd.Timestamp], right_dates: list[pd.Timestamp], near_hours: int) -> int:
    if not left_dates or not right_dates:
        return 0
    right = pd.Series(right_dates)
    delta = pd.Timedelta(hours=near_hours)
    count = 0
    for date in left_dates:
        if ((right - date).abs() <= delta).any():
            count += 1
    return count


def simulate_individual(features: DataFrame, signals: DataFrame) -> tuple[DataFrame, DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for spec in LEADS:
        lead_signals = signals[signals["lead_id"].eq(spec.lead_id)].copy()
        trades = simulate_signal_rows(features, lead_signals, spec)
        if not trades.empty:
            trade_frames.append(trades)
        summary = summarize_trades(
            trades,
            spec.lead_id,
            "orderbook_state",
            "short",
            spec.trader_question,
            spec.visible_market_state,
            features,
        )
        summary.update(
            {
                "lead_id": spec.lead_id,
                "raw_signals": int(len(lead_signals)),
                "priority": spec.priority,
                "stop_loss": spec.stop_loss,
                "take_profit": spec.take_profit,
                "hold_hours": spec.hold_hours,
            }
        )
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.concat(trade_frames, ignore_index=True) if trade_frames else DataFrame()


def simulate_combined(features: DataFrame, signals: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    flat_after = pd.Timestamp.min.tz_localize("UTC")
    specs = {spec.lead_id: spec for spec in LEADS}
    for _, signal in signals.sort_values(["date", "priority"]).iterrows():
        signal_date = pd.to_datetime(signal["date"], utc=True)
        if signal_date <= flat_after:
            continue
        spec = specs[str(signal["lead_id"])]
        trades = simulate_signal_rows(features, pd.DataFrame([signal]), spec)
        if trades.empty:
            continue
        trade = trades.iloc[0].to_dict()
        rows.append(trade)
        flat_after = pd.to_datetime(trade["exit_date"], utc=True)
    return pd.DataFrame(rows)


def simulate_signal_rows(features: DataFrame, signals: DataFrame, spec: LeadSpec) -> DataFrame:
    by_date = {date: idx for idx, date in enumerate(features["date"])}
    open_ = num(features, "open")
    high = num(features, "high")
    low = num(features, "low")
    close = num(features, "close")
    rows: list[dict[str, Any]] = []
    flat_after = -1
    for _, signal in signals.sort_values("date").iterrows():
        idx = by_date.get(pd.to_datetime(signal["date"], utc=True))
        if idx is None or idx <= flat_after or idx > len(features) - spec.hold_hours - 2:
            continue
        entry_idx = idx + 1
        exit_idx = idx + spec.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit(
            "short",
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            spec.stop_loss,
            spec.take_profit,
        )
        if not np.isfinite(exit_price) or exit_price <= 0.0:
            continue
        gross_return = entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": spec.lead_id,
                "rule_id": spec.lead_id,
                "concept_id": "orderbook_support_removal_short",
                "direction": "short",
                "signal_date": features["date"].iloc[idx],
                "entry_date": features["date"].iloc[entry_idx],
                "exit_date": features["date"].iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - 0.0012,
                "exit_reason": exit_reason,
                "signal_score": signal.get("signal_score", np.nan),
                "priority": spec.priority,
                "hold_hours": spec.hold_hours,
                "stop_loss": spec.stop_loss,
                "take_profit": spec.take_profit,
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def exportable_combined_signals(combined_trades: DataFrame) -> DataFrame:
    columns = ["date", "enter_long", "enter_short", "enter_tag", "rule_id", "concept_id", "threshold", "signal_score"]
    if combined_trades.empty:
        return pd.DataFrame(columns=columns)
    out = combined_trades[["signal_date", "rule_id", "concept_id", "signal_score"]].copy()
    out["date"] = pd.to_datetime(out["signal_date"], utc=True, errors="coerce")
    out["enter_long"] = 0
    out["enter_short"] = 1
    out["enter_tag"] = out["rule_id"].astype(str)
    out["threshold"] = pd.NA
    return out[columns].dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)


def markdown_report(overlap: DataFrame, individual: DataFrame, combined: dict[str, Any], core: dict[str, Any], signal_path: Path, core_signal_path: Path) -> str:
    lead_cols = ["lead_id", "raw_signals", "trades", "win_rate", "total_return", "max_drawdown", "profit_factor"]
    pair_cols = [
        "left_lead_id",
        "right_lead_id",
        "exact_overlap",
        "near_overlap",
        "exact_overlap_share_of_left",
        "near_overlap_share_of_left",
    ]
    non_self = overlap[~overlap["left_lead_id"].eq(overlap["right_lead_id"])].copy()
    non_self = non_self.sort_values(["near_overlap_share_of_left", "exact_overlap_share_of_left"], ascending=[False, False])
    lines = [
        "# Validated Orderbook Lead Overlap",
        "",
        "Trader question: are the validated support-removal shorts separate entry opportunities, or mostly duplicated signals?",
        "",
        "## Combined Non-Overlapped Result",
        "",
        f"1. Trades: {combined.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(combined.get('max_drawdown'))}",
        f"5. Profit factor: {combined.get('profit_factor', 0):.2f}",
        f"6. Exported signal file: `{signal_path}`",
        "",
        "## Core Non-Overlapped Result",
        "",
        f"1. Trades: {core.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(core.get('win_rate'))}",
        f"3. Total return: {fmt_pct(core.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(core.get('max_drawdown'))}",
        f"5. Profit factor: {core.get('profit_factor', 0):.2f}",
        f"6. Exported signal file: `{core_signal_path}`",
        "",
        "## Individual Leads",
        "",
        individual[lead_cols].to_markdown(index=False),
        "",
        "## Largest Cross-Lead Overlaps",
        "",
        non_self.head(15)[pair_cols].to_markdown(index=False),
        "",
        "## Plain-English Read",
        "",
        "1. Exact overlap shows the same signal hour.",
        "2. Near overlap shows another lead fired within the configured nearby window, so the ideas may be describing the same market event.",
        "3. If combined trades are much lower than the sum of individual trades, the leads compete for the same opportunities.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
