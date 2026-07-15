from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from trading_lead_exit_research import (
    ExitSpec,
    REPORTS_DIR,
    DEFAULT_FEATURES,
    load_frame,
    load_signals,
    num,
    safe_name,
    summarize_trades,
)


DEFAULT_SIGNALS = REPORTS_DIR / "trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet"
DEFAULT_SELECTED_EXITS = REPORTS_DIR / "trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_selected.csv"


@dataclass(frozen=True)
class ManagementSpec:
    management_id: str
    trader_question: str
    visible_market_state: str
    management_type: str
    skip_builder: Callable[[DataFrame], Series] | None = None
    exit_builder: Callable[[DataFrame], Series] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Test trade-management exits/skips on selected BTC lead signals.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--selected-exits", type=Path, default=DEFAULT_SELECTED_EXITS)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    args = parser.parse_args()

    for path in (args.features, args.signals, args.selected_exits):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = load_frame(args.features)
    signals = load_signals(args.signals)
    selected = load_selected_exits(args.selected_exits)
    specs = build_management_specs()
    summaries, trades = run_management_tests(
        frame,
        signals,
        selected,
        specs,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trading_lead_trade_management_{tag}_summary.csv"
    trades_path = args.output_dir / f"trading_lead_trade_management_{tag}_trades.csv"
    markdown_path = args.output_dir / f"trading_lead_trade_management_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_trade_management_{tag}_meta.json"
    summaries.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    markdown_path.write_text(markdown_report(summaries), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": str(args.features),
        "signals": str(args.signals),
        "selected_exits": str(args.selected_exits),
        "entry_signals": int(len(signals)),
        "management_specs": [s.management_id for s in specs],
        "outputs": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_selected_exits(path: Path) -> dict[str, ExitSpec]:
    frame = pd.read_csv(path)
    required = {"rule_id", "exit_id", "hold_hours", "stop_loss", "take_profit", "visible_market_state"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Selected exit file missing columns: {missing}")
    selected: dict[str, ExitSpec] = {}
    for row in frame.itertuples(index=False):
        selected[str(getattr(row, "rule_id"))] = ExitSpec(
            exit_id=str(getattr(row, "exit_id")),
            hold_hours=int(getattr(row, "hold_hours")),
            stop_loss=float(getattr(row, "stop_loss")),
            take_profit=float(getattr(row, "take_profit")),
            exit_story=str(getattr(row, "visible_market_state")),
        )
    return selected


def build_management_specs() -> tuple[ManagementSpec, ...]:
    return (
        ManagementSpec(
            "baseline_selected_exits",
            "What happens if we use the current selected entries and tailored exits with no extra trade-management layer?",
            "Entry and selected per-rule exit only.",
            "baseline",
        ),
        ManagementSpec(
            "orderbook_invalidation_exit",
            "Once in a trade, does the orderbook show the original idea is failing?",
            "Exit longs when support disappears, sell pressure appears, or resistance rejects; exit shorts when support rebuilds or bullish pressure appears.",
            "early_exit",
            exit_builder=orderbook_invalidation_exit,
        ),
        ManagementSpec(
            "orderbook_crash_downturn_exit",
            "For longs, can orderbook crash/downturn escalation get us out before a larger loss?",
            "Exit longs when bearish pressure, downside vacuum, support removal, spread fragility, or range breakdown cluster together.",
            "early_exit",
            exit_builder=orderbook_crash_downturn_exit,
        ),
        ManagementSpec(
            "volume_failure_exit",
            "If the move does not get volume confirmation after entry, should we exit instead of waiting for the stop?",
            "Exit longs when bullish volume fades or bearish volume takes over; mirror for shorts.",
            "early_exit",
            exit_builder=volume_failure_exit,
        ),
        ManagementSpec(
            "structure_level_failure_exit",
            "If price loses the structure/VP level that justified the trade, should we exit?",
            "Exit longs on breakdown or bearish structure; exit shorts on breakout or bullish structure.",
            "early_exit",
            exit_builder=structure_level_failure_exit,
        ),
        ManagementSpec(
            "volatility_shock_risk_off_exit",
            "If volatility expands hard against the trade, should we exit instead of waiting?",
            "Exit when price moves against the trade during high range/volume expansion.",
            "early_exit",
            exit_builder=volatility_shock_exit,
        ),
        ManagementSpec(
            "time_to_confirm_exit_6h",
            "If a trade has not started working after six hours, should we leave?",
            "Exit stale trades after six hours if price is not positive and confirmation is absent.",
            "early_exit",
            exit_builder=time_to_confirm_exit,
        ),
        ManagementSpec(
            "mtf_disagreement_skip",
            "Should we skip entries where higher timeframe structure strongly disagrees?",
            "Skip longs when 4h/1d structure is bearish; skip shorts when 4h/1d structure is bullish.",
            "skip",
            skip_builder=mtf_disagreement_skip,
        ),
        ManagementSpec(
            "range_chop_avoidance_skip",
            "Should we skip breakout-style entries when the market is compressed but has not actually broken range?",
            "Skip entries where compression/range chop is active but range break confirmation is missing.",
            "skip",
            skip_builder=range_chop_avoidance_skip,
        ),
        ManagementSpec(
            "post_break_acceptance_skip",
            "Should we require the break to show acceptance before taking the signal?",
            "Keep longs only if breakout/above-value behaviour is already visible; keep shorts only if breakdown/below-value behaviour is visible.",
            "skip",
            skip_builder=post_break_acceptance_missing_skip,
        ),
        ManagementSpec(
            "sr_proximity_skip",
            "Should we skip trades where the next opposing support/resistance zone is too close?",
            "Skip longs very close to resistance and shorts very close to support.",
            "skip",
            skip_builder=sr_proximity_skip,
        ),
    )


def run_management_tests(
    frame: DataFrame,
    signals: DataFrame,
    selected: dict[str, ExitSpec],
    specs: tuple[ManagementSpec, ...],
    *,
    round_trip_fee: float,
    slippage: float,
) -> tuple[DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    baseline_trades = simulate_with_management(frame, signals, selected, None, round_trip_fee, slippage)
    baseline_summary = summarize_management(frame, baseline_trades, specs[0], baseline_trades)
    rows.append(baseline_summary)
    trade_frames.append(baseline_trades.assign(management_id="baseline_selected_exits"))
    for spec in specs[1:]:
        managed = simulate_with_management(frame, signals, selected, spec, round_trip_fee, slippage)
        rows.append(summarize_management(frame, managed, spec, baseline_trades))
        trade_frames.append(managed.assign(management_id=spec.management_id))
    summary = pd.DataFrame(rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else DataFrame()
    return summary.sort_values(["verdict_rank", "drawdown_delta", "total_return"], ascending=[True, False, False]), trades


def simulate_with_management(
    frame: DataFrame,
    signals: DataFrame,
    selected: dict[str, ExitSpec],
    spec: ManagementSpec | None,
    round_trip_fee: float,
    slippage: float,
) -> DataFrame:
    indexed = frame.set_index("date", drop=False)
    frame_index = {date: idx for idx, date in enumerate(frame["date"])}
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    signals = signals.sort_values("date").reset_index(drop=True)
    flat_until = pd.Timestamp.min.tz_localize("UTC")
    rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        signal_date = pd.to_datetime(signal["date"], utc=True)
        if signal_date <= flat_until:
            continue
        rule_id = str(signal["rule_id"])
        exit_spec = selected.get(rule_id)
        if exit_spec is None:
            continue
        signal_row = indexed.loc[signal_date] if signal_date in indexed.index else None
        if spec is not None and spec.skip_builder is not None and signal_row is not None:
            skip_mask = spec.skip_builder(pd.DataFrame([signal_row]).assign(direction=str(signal["direction"]), rule_id=rule_id))
            if bool(skip_mask.iloc[0]):
                continue
        i = frame_index.get(signal_date)
        if i is None or i > len(frame) - exit_spec.hold_hours - 2:
            continue
        entry_idx = i + 1
        base_exit_idx = i + exit_spec.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        direction = str(signal["direction"])
        exit_idx = base_exit_idx
        exit_price = float(close.iloc[base_exit_idx])
        exit_reason = "time_exit"
        stop_price, take_price = stop_take_prices(direction, entry_price, exit_spec.stop_loss, exit_spec.take_profit)
        for idx in range(entry_idx, base_exit_idx + 1):
            if direction == "long":
                if float(low.iloc[idx]) <= stop_price:
                    exit_idx = idx
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                    break
                if float(high.iloc[idx]) >= take_price:
                    exit_idx = idx
                    exit_price = take_price
                    exit_reason = "take_profit"
                    break
            else:
                if float(high.iloc[idx]) >= stop_price:
                    exit_idx = idx
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                    break
                if float(low.iloc[idx]) <= take_price:
                    exit_idx = idx
                    exit_price = take_price
                    exit_reason = "take_profit"
                    break
            if spec is not None and spec.exit_builder is not None and idx > entry_idx:
                elapsed = idx - entry_idx
                current_profit = close_return(direction, entry_price, float(close.iloc[idx]))
                test_row = frame.iloc[[idx]].copy()
                test_row["direction"] = direction
                test_row["rule_id"] = rule_id
                test_row["elapsed_hours"] = elapsed
                test_row["current_profit"] = current_profit
                if bool(spec.exit_builder(test_row).iloc[0]):
                    exit_idx = idx
                    exit_price = float(close.iloc[idx])
                    exit_reason = spec.management_id
                    break
        gross_return = close_return(direction, entry_price, exit_price)
        rows.append(
            {
                "rule_id": rule_id,
                "concept_id": str(signal["concept_id"]),
                "direction": direction,
                "signal_date": signal_date,
                "entry_date": frame["date"].iloc[entry_idx],
                "exit_date": frame["date"].iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
                "base_exit_id": exit_spec.exit_id,
                "hold_hours": exit_spec.hold_hours,
                "stop_loss": exit_spec.stop_loss,
                "take_profit": exit_spec.take_profit,
                "signal_score": float(signal["signal_score"]) if np.isfinite(float(signal["signal_score"])) else np.nan,
            }
        )
        flat_until = pd.to_datetime(frame["date"].iloc[exit_idx], utc=True)
    return DataFrame(rows)


def summarize_management(frame: DataFrame, trades: DataFrame, spec: ManagementSpec, baseline: DataFrame) -> dict[str, Any]:
    summary = summarize_trades(
        trades,
        spec.management_id,
        "btc_sieve_quality_family",
        "both",
        spec.trader_question,
        spec.visible_market_state,
        frame,
    )
    baseline_summary = summarize_trades(
        baseline,
        "baseline",
        "btc_sieve_quality_family",
        "both",
        "baseline",
        "baseline",
        frame,
    )
    summary["management_id"] = spec.management_id
    summary["management_type"] = spec.management_type
    summary["trader_question"] = spec.trader_question
    summary["visible_market_state"] = spec.visible_market_state
    summary["trade_delta"] = int(summary["trades"]) - int(baseline_summary["trades"])
    summary["return_delta"] = float(summary["total_return"]) - float(baseline_summary["total_return"])
    summary["drawdown_delta"] = float(summary["max_drawdown"]) - float(baseline_summary["max_drawdown"])
    summary["profit_factor_delta"] = clean_float(summary["profit_factor"]) - clean_float(baseline_summary["profit_factor"])
    summary["win_rate_delta"] = clean_float(summary["win_rate"]) - clean_float(baseline_summary["win_rate"])
    summary["early_exit_count"] = int(trades["exit_reason"].eq(spec.management_id).sum()) if not trades.empty else 0
    summary["baseline_return"] = float(baseline_summary["total_return"])
    summary["baseline_drawdown"] = float(baseline_summary["max_drawdown"])
    summary["baseline_profit_factor"] = clean_float(baseline_summary["profit_factor"])
    verdict, rank = verdict_for(summary)
    summary["verdict"] = verdict
    summary["verdict_rank"] = rank
    return summary


def verdict_for(row: dict[str, Any]) -> tuple[str, int]:
    if row["management_type"] == "baseline":
        return "baseline", 2
    drawdown_better = float(row["drawdown_delta"]) > 0.002
    baseline_return = max(clean_float(row.get("baseline_return")), 1e-9)
    return_ratio = clean_float(row.get("total_return")) / baseline_return
    return_close = return_ratio >= 0.80
    return_acceptable_for_rework = return_ratio >= 0.50
    return_better = float(row["return_delta"]) > 0.0
    pf_better = float(row["profit_factor_delta"]) > 0.05
    enough_trades = int(row["trades"]) >= 150
    if drawdown_better and return_close and (return_better or pf_better) and enough_trades:
        return "promote_for_freqtrade_validation", 0
    if drawdown_better and return_close and enough_trades:
        return "worth_rework", 1
    if drawdown_better and return_acceptable_for_rework and enough_trades:
        return "drawdown_helped_but_profit_cost_high", 1
    if return_better and enough_trades:
        return "return_improved_but_check_drawdown", 1
    return "reject_for_now", 3


def orderbook_invalidation_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    long_bad = (
        max_existing(frame, ["conf_ob_bearish_pressure_agreement", "conf_ob_support_removed_strength", "conf_ob_downside_vacuum_after_support_removed"]).ge(0.45)
        | max_existing(frame, ["conf_ob_resistance_rejection", "conf_ob_ask_absorption", "conf_ob_breakout_failure"]).ge(0.60)
    )
    short_bad = (
        max_existing(frame, ["conf_ob_bullish_pressure_agreement", "conf_ob_resistance_removed_strength", "conf_ob_upside_vacuum_after_resistance_removed"]).ge(0.45)
        | max_existing(frame, ["conf_ob_support_bounce", "conf_ob_bid_absorption", "conf_ob_breakdown_failure"]).ge(0.60)
    )
    return (direction.eq("long") & long_bad) | (direction.eq("short") & short_bad)


def orderbook_crash_downturn_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    crash_score = (
        max_existing(frame, ["conf_ob_bearish_pressure_agreement"]).ge(0.35).astype(int)
        + max_existing(frame, ["conf_ob_support_removed_strength", "conf_ob_support_cleared"]).ge(0.35).astype(int)
        + max_existing(frame, ["conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum"]).ge(0.35).astype(int)
        + max_existing(frame, ["conf_ob_spread_fragility", "conf_ob_single_venue_extreme"]).ge(0.50).astype(int)
        + max_existing(frame, ["px_close_breakdown_6h", "px_close_breakdown_24h"]).ge(0.50).astype(int)
    )
    return direction.eq("long") & crash_score.ge(2)


def volume_failure_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    elapsed = pd.to_numeric(frame["elapsed_hours"], errors="coerce").fillna(0)
    profit = pd.to_numeric(frame["current_profit"], errors="coerce").fillna(0)
    bull = max_existing(frame, ["conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"])
    bear = max_existing(frame, ["conf_volume_bearish_confirmation", "-px_volume_pressure_6h", "-px_volume_pressure_24h"])
    long_fail = direction.eq("long") & elapsed.ge(3) & profit.le(0.003) & (bull.lt(0.20) | bear.ge(0.45))
    short_fail = direction.eq("short") & elapsed.ge(3) & profit.le(0.003) & (bear.lt(0.20) | bull.ge(0.45))
    return long_fail | short_fail


def structure_level_failure_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    long_fail = direction.eq("long") & (
        max_existing(frame, ["conf_structure_breakdown_trigger_score", "conf_structure_bearish_state_score", "st_1h_ms_bos_to_bear", "st_4h_ms_bos_to_bear"]).ge(0.45)
        | max_existing(frame, ["px_close_breakdown_6h", "st_1h_vp_below_value_area", "st_4h_vp_below_value_area"]).ge(0.60)
    )
    short_fail = direction.eq("short") & (
        max_existing(frame, ["conf_structure_breakout_trigger_score", "conf_structure_bullish_state_score", "st_1h_ms_bos_to_bull", "st_4h_ms_bos_to_bull"]).ge(0.45)
        | max_existing(frame, ["px_close_breakout_6h", "st_1h_vp_above_value_area", "st_4h_vp_above_value_area"]).ge(0.60)
    )
    return long_fail | short_fail


def volatility_shock_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    ret = max_existing(frame, ["px_return_1h"])
    range_hot = max_existing(frame, ["px_range_1h", "px_volume_z_6h", "px_volume_z_24h"]).ge(0.75)
    long_bad = direction.eq("long") & range_hot & ret.lt(-0.008)
    short_bad = direction.eq("short") & range_hot & ret.gt(0.008)
    return long_bad | short_bad


def time_to_confirm_exit(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    elapsed = pd.to_numeric(frame["elapsed_hours"], errors="coerce").fillna(0)
    profit = pd.to_numeric(frame["current_profit"], errors="coerce").fillna(0)
    long_confirm = max_existing(frame, ["conf_volume_bullish_confirmation", "conf_structure_bullish_state_score", "px_close_breakout_6h"]).ge(0.45)
    short_confirm = max_existing(frame, ["conf_volume_bearish_confirmation", "conf_structure_bearish_state_score", "px_close_breakdown_6h"]).ge(0.45)
    return (elapsed.ge(6) & profit.le(0.002) & ((direction.eq("long") & ~long_confirm) | (direction.eq("short") & ~short_confirm)))


def mtf_disagreement_skip(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    higher_bear = max_existing(frame, ["st_4h_ms_bos_to_bear", "st_1d_ms_bos_to_bear", "conf_structure_bearish_state_score"]).ge(0.55)
    higher_bull = max_existing(frame, ["st_4h_ms_bos_to_bull", "st_1d_ms_bos_to_bull", "conf_structure_bullish_state_score"]).ge(0.55)
    return (direction.eq("long") & higher_bear) | (direction.eq("short") & higher_bull)


def range_chop_avoidance_skip(frame: DataFrame) -> Series:
    compression = max_existing(frame, ["conf_price_compression_24h", "st_1h_pg2_compression_squeeze_active", "st_4h_pg2_compression_squeeze_active"]).ge(0.45)
    range_break = max_existing(frame, ["px_close_breakout_24h", "px_close_breakdown_24h", "conf_structure_breakout_trigger_score", "conf_structure_breakdown_trigger_score"]).ge(0.40)
    return compression & ~range_break


def post_break_acceptance_missing_skip(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    long_accept = max_existing(frame, ["px_close_breakout_6h", "px_close_breakout_24h", "st_1h_vp_above_value_area", "st_4h_vp_above_value_area"]).ge(0.40)
    short_accept = max_existing(frame, ["px_close_breakdown_6h", "px_close_breakdown_24h", "st_1h_vp_below_value_area", "st_4h_vp_below_value_area"]).ge(0.40)
    return (direction.eq("long") & ~long_accept) | (direction.eq("short") & ~short_accept)


def sr_proximity_skip(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str)
    res = max_existing(frame, ["conf_ob_resistance_distance_min_bps"])
    sup = max_existing(frame, ["conf_ob_support_distance_min_bps"])
    near_res = res.gt(0.0) & res.le(35.0)
    near_sup = sup.gt(0.0) & sup.le(35.0)
    return (direction.eq("long") & near_res) | (direction.eq("short") & near_sup)


def stop_take_prices(direction: str, entry_price: float, stop_loss: float, take_profit: float) -> tuple[float, float]:
    if direction == "long":
        return entry_price * (1.0 - stop_loss), entry_price * (1.0 + take_profit)
    return entry_price * (1.0 + stop_loss), entry_price * (1.0 - take_profit)


def close_return(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return exit_price / entry_price - 1.0
    return entry_price / exit_price - 1.0


def max_existing(frame: DataFrame, cols: list[str]) -> Series:
    parts: list[Series] = []
    for col in cols:
        sign = -1.0 if col.startswith("-") else 1.0
        clean_col = col[1:] if col.startswith("-") else col
        if clean_col in frame.columns:
            parts.append(sign * pd.to_numeric(frame[clean_col], errors="coerce"))
    if not parts:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def clean_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def markdown_report(summary: DataFrame) -> str:
    lines = [
        "# Trading Lead Trade Management Research",
        "",
        "This keeps the selected BTC lead block fixed and tests skip/early-exit ideas intended to reduce drawdown.",
        "",
        summary[
            [
                "management_id",
                "management_type",
                "trades",
                "win_rate",
                "total_return",
                "max_drawdown",
                "profit_factor",
                "trade_delta",
                "return_delta",
                "drawdown_delta",
                "early_exit_count",
                "verdict",
            ]
        ].to_markdown(index=False),
        "",
        "## Plain-English Notes",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"1. `{row['management_id']}`")
        lines.append(f"   - Trader question: {row['trader_question']}")
        lines.append(f"   - What was looked at: {row['visible_market_state']}")
        lines.append(
            f"   - Result: {int(row['trades'])} trades, return {fmt_pct(row['total_return'])}, drawdown {fmt_pct(row['max_drawdown'])}, profit factor {fmt(row['profit_factor'])}."
        )
        lines.append(f"   - Verdict: {row['verdict']}")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: Any) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
