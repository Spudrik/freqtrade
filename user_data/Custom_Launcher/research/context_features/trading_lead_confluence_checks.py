from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_TRADES = REPORTS_DIR / "trader_rule_backtests_sieve_exact_min10_20260605_combined_trades.csv"


@dataclass(frozen=True)
class FilterSpec:
    filter_id: str
    trader_question: str
    data_family: str
    mask_builder: Callable[[DataFrame], Series]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test confluence filters against selected trader-rule entries.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--min-trades", type=int, default=8)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.trades.exists():
        raise FileNotFoundError(args.trades)
    features = load_features(args.features)
    trades = pd.read_csv(args.trades)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["signal_date"]).sort_values("signal_date").reset_index(drop=True)
    merged = trades.merge(features, left_on="signal_date", right_on="date", how="left", suffixes=("", "_feature"))
    filters = build_filters(merged)
    summary = run_checks(merged, filters, min_trades=args.min_trades)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = safe_name(args.tag)
    csv_path = args.output_dir / f"trading_lead_confluence_checks_{safe_tag}.csv"
    md_path = args.output_dir / f"trading_lead_confluence_checks_{safe_tag}.md"
    meta_path = args.output_dir / f"trading_lead_confluence_checks_{safe_tag}_meta.json"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(markdown_report(summary, trades, filters), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trades_path": str(args.trades),
        "features_path": str(args.features),
        "input_trades": int(len(trades)),
        "filters": [f.filter_id for f in filters],
        "result_rows": int(len(summary)),
        "outputs": {"summary": str(csv_path), "markdown": str(md_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    keep_patterns = (
        "date",
        "conf_",
        "px_",
        "st_volume",
        "st_range",
        "st_1h_vp_",
        "st_4h_vp_",
        "st_1d_vp_",
        "st_1h_ms_",
        "st_4h_ms_",
        "st_1d_ms_",
        "ob_",
        "context_",
        "ctx_",
    )
    cols = [c for c in frame.columns if c == "date" or c.startswith(keep_patterns[1:])]
    return frame[cols]


def build_filters(frame: DataFrame) -> tuple[FilterSpec, ...]:
    return (
        FilterSpec(
            "strong_volume_agrees_with_direction",
            "Did volume/pressure strongly agree with the entry direction?",
            "OHLCV_volume_pressure",
            lambda f: directional_score(f, long_cols=["conf_volume_bullish_confirmation", "px_volume_pressure_24h", "st_volume_pressure_24h"], short_cols=["conf_volume_bearish_confirmation", "px_volume_pressure_24h", "st_volume_pressure_24h"], threshold=0.60),
        ),
        FilterSpec(
            "structure_trigger_agrees",
            "Did custom structure agree with the entry direction?",
            "custom_structure",
            lambda f: directional_score(f, long_cols=["conf_structure_breakout_trigger_score", "st_1h_ms_bos_to_bull", "st_4h_ms_bos_to_bull"], short_cols=["conf_structure_breakdown_trigger_score", "st_1h_ms_bos_to_bear", "st_4h_ms_bos_to_bear"], threshold=0.34),
        ),
        FilterSpec(
            "vp_context_agrees",
            "Was VP/value-area context supportive instead of fighting the entry?",
            "custom_structure_vp",
            lambda f: vp_context_agrees(f),
        ),
        FilterSpec(
            "range_break_agrees",
            "Was price breaking the relevant recent range in the same direction?",
            "OHLCV_range_position",
            lambda f: directional_score(f, long_cols=["px_close_breakout_24h", "px_close_breakout_72h", "conf_range_position_high_stack"], short_cols=["px_close_breakdown_24h", "px_close_breakdown_72h", "conf_range_position_low_stack"], threshold=0.50),
        ),
        FilterSpec(
            "compression_release_context",
            "Was the market compressed enough that a break could release into movement?",
            "price_structure_compression",
            lambda f: max_existing(f, ["conf_price_compression_24h", "conf_delta_compression_breakout_release_score", "conf_delta_compression_breakdown_release_score"]).ge(0.45),
        ),
        FilterSpec(
            "orderbook_present",
            "Was real orderbook data available at this decision hour?",
            "orderbook",
            lambda f: orderbook_present(f),
        ),
        FilterSpec(
            "orderbook_direction_agrees",
            "When orderbook exists, did it agree with the intended direction?",
            "orderbook",
            lambda f: orderbook_present(f) & orderbook_direction_agrees(f),
        ),
        FilterSpec(
            "regime_agrees",
            "Did the broader price regime suit this entry type?",
            "price_regime",
            lambda f: regime_agrees(f),
        ),
        FilterSpec(
            "context_safe_or_missing",
            "Did context data avoid obvious timestamp-risk or stale-data warnings?",
            "news_context_readiness",
            lambda f: context_safe_or_missing(f),
        ),
        FilterSpec(
            "no_support_reclaim_warning",
            "After a downside break, was there no strong warning that broken support was being reclaimed?",
            "orderbook_downside_failure_warning",
            lambda f: no_support_reclaim_warning(f),
        ),
        FilterSpec(
            "no_support_rebuild_warning",
            "After a downside break, was there no strong sign that buy-side support was rebuilding?",
            "orderbook_downside_failure_warning",
            lambda f: no_support_rebuild_warning(f),
        ),
        FilterSpec(
            "bearish_pressure_persists",
            "After the short signal, did sell pressure remain present instead of fading?",
            "orderbook_volume_pressure",
            lambda f: bearish_pressure_persists(f),
        ),
        FilterSpec(
            "downside_vacuum_still_open",
            "After support removal, was there still thin liquidity below price?",
            "orderbook_liquidity_vacuum",
            lambda f: downside_vacuum_still_open(f),
        ),
        FilterSpec(
            "breakdown_failure_warning_absent",
            "Was there no strong warning that the breakdown was failing?",
            "orderbook_structure_failure_warning",
            lambda f: breakdown_failure_warning_absent(f),
        ),
        FilterSpec(
            "panic_after_break_state",
            "Did the orderbook look like a downside panic or acceleration state after the first break?",
            "orderbook_downside_momentum",
            lambda f: panic_after_break_state(f),
        ),
    )


def run_checks(trades: DataFrame, filters: tuple[FilterSpec, ...], min_trades: int) -> DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all_selected_entries", trades)]
    groups.extend((str(rule_id), group) for rule_id, group in trades.groupby("rule_id", dropna=False))
    for group_id, group in groups:
        base = summarize(group)
        rows.append(result_row(group_id, "base_no_extra_filter", "Base selected entry set before extra confluence filters.", "base", base, base, min_trades))
        masks: dict[str, Series] = {}
        for spec in filters:
            mask = spec.mask_builder(group).fillna(False).astype(bool)
            masks[spec.filter_id] = mask
            filtered = group[mask]
            rows.append(result_row(group_id, spec.filter_id, spec.trader_question, spec.data_family, summarize(filtered), base, min_trades))
        for combo_id, parts, question in confluence_combos():
            if all(part in masks for part in parts):
                combo_mask = pd.Series(True, index=group.index)
                for part in parts:
                    combo_mask &= masks[part]
                rows.append(
                    result_row(
                        group_id,
                        combo_id,
                        question,
                        "combined_confluence",
                        summarize(group[combo_mask]),
                        base,
                        min_trades,
                    )
                )
    result = pd.DataFrame(rows)
    result = result.sort_values(["group_id", "quality_delta", "filtered_trades"], ascending=[True, False, False]).reset_index(drop=True)
    return result


def confluence_combos() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    return (
        (
            "strong_volume_plus_structure",
            ("strong_volume_agrees_with_direction", "structure_trigger_agrees"),
            "Did strong volume and custom structure agree at the same decision hour?",
        ),
        (
            "strong_volume_plus_range_break",
            ("strong_volume_agrees_with_direction", "range_break_agrees"),
            "Did strong volume appear while price was breaking the relevant range?",
        ),
        (
            "vp_plus_range_break",
            ("vp_context_agrees", "range_break_agrees"),
            "Did VP/value-area context agree while price broke the relevant range?",
        ),
        (
            "compression_plus_range_break",
            ("compression_release_context", "range_break_agrees"),
            "Was price breaking range after compression?",
        ),
        (
            "compression_plus_volume_plus_range",
            ("compression_release_context", "strong_volume_agrees_with_direction", "range_break_agrees"),
            "Was this a compressed market breaking range with strong directional volume?",
        ),
        (
            "orderbook_plus_structure",
            ("orderbook_direction_agrees", "structure_trigger_agrees"),
            "Did orderbook direction and custom structure agree together?",
        ),
        (
            "orderbook_plus_strong_volume",
            ("orderbook_direction_agrees", "strong_volume_agrees_with_direction"),
            "Did orderbook direction and strong volume agree together?",
        ),
        (
            "regime_plus_range_break",
            ("regime_agrees", "range_break_agrees"),
            "Did broader regime agree while price broke the relevant range?",
        ),
        (
            "support_break_no_reclaim",
            ("no_support_reclaim_warning", "no_support_rebuild_warning"),
            "Did the support break avoid the two main failure warnings: reclaim and support rebuild?",
        ),
        (
            "support_break_pressure_persists",
            ("no_support_reclaim_warning", "bearish_pressure_persists"),
            "Did the support break avoid reclaim pressure while sell pressure stayed present?",
        ),
        (
            "vacuum_no_failure_warning",
            ("downside_vacuum_still_open", "breakdown_failure_warning_absent"),
            "Did the downside vacuum remain open without a breakdown-failure warning?",
        ),
        (
            "panic_no_rebuild",
            ("panic_after_break_state", "no_support_rebuild_warning"),
            "Did downside panic appear without support rebuilding underneath price?",
        ),
    )


def result_row(group_id: str, filter_id: str, question: str, family: str, stats: dict[str, float], base: dict[str, float], min_trades: int) -> dict[str, object]:
    trade_delta = stats["trades"] - base["trades"]
    quality_delta = (stats["profit_factor"] - base["profit_factor"]) + ((stats["win_rate"] - base["win_rate"]) * 1.5) + ((stats["avg_return"] - base["avg_return"]) * 40.0)
    enough = stats["trades"] >= min_trades
    if filter_id == "base_no_extra_filter":
        verdict = "baseline"
    elif not enough:
        verdict = "too_few_trades"
    elif quality_delta > 0.15 and stats["profit_factor"] > base["profit_factor"]:
        verdict = "worth_more_testing"
    elif quality_delta > 0.0:
        verdict = "watchlist"
    else:
        verdict = "reject_for_now"
    return {
        "group_id": group_id,
        "filter_id": filter_id,
        "trader_question": question,
        "data_family": family,
        "filtered_trades": int(stats["trades"]),
        "base_trades": int(base["trades"]),
        "trade_delta": int(trade_delta),
        "win_rate": round(stats["win_rate"], 4),
        "base_win_rate": round(base["win_rate"], 4),
        "avg_return": round(stats["avg_return"], 5),
        "base_avg_return": round(base["avg_return"], 5),
        "total_return_sum": round(stats["total_return_sum"], 5),
        "base_total_return_sum": round(base["total_return_sum"], 5),
        "profit_factor": round(stats["profit_factor"], 4),
        "base_profit_factor": round(base["profit_factor"], 4),
        "max_loss": round(stats["max_loss"], 5),
        "base_max_loss": round(base["max_loss"], 5),
        "quality_delta": round(quality_delta, 4),
        "verdict": verdict,
    }


def summarize(frame: DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return_sum": 0.0,
            "profit_factor": 0.0,
            "max_loss": 0.0,
        }
    ret = pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0)
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": float(len(frame)),
        "win_rate": float((ret > 0).mean()),
        "avg_return": float(ret.mean()),
        "total_return_sum": float(ret.sum()),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
        "max_loss": float(ret.min()),
    }


def directional_score(frame: DataFrame, long_cols: list[str], short_cols: list[str], threshold: float) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    long_score = max_existing(frame, long_cols)
    short_score = max_existing(frame, short_cols)
    signed_short_score = short_score.copy()
    for col in short_cols:
        if col in frame.columns and "pressure" in col:
            signed_short_score = pd.concat([signed_short_score, -num(frame, col)], axis=1).max(axis=1)
    long_mask = long_score.ge(threshold)
    short_mask = signed_short_score.ge(threshold)
    return (direction.eq("long") & long_mask) | (direction.eq("short") & short_mask)


def vp_context_agrees(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    long_score = max_existing(frame, ["st_1h_vp_score_long", "st_4h_vp_score_long", "st_1d_vp_score_long"])
    short_score = max_existing(frame, ["st_1h_vp_score_short", "st_4h_vp_score_short", "st_1d_vp_score_short"])
    long_mask = long_score.ge(short_score)
    short_mask = short_score.ge(long_score)
    return (direction.eq("long") & long_mask) | (direction.eq("short") & short_mask)


def orderbook_present(frame: DataFrame) -> Series:
    cols = [c for c in frame.columns if c.endswith("_feature_present") and c.startswith("ob_")]
    if not cols:
        cols = [c for c in frame.columns if c.endswith("_coverage_ratio") and c.startswith("ob_")]
    if not cols:
        return pd.Series(False, index=frame.index)
    block = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return block.gt(0).any(axis=1)


def orderbook_direction_agrees(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    long_cols = [
        "conf_orderbook_resistance_evaporation_breakout_cmp_score_resistance_removed",
        "conf_orderbook_resistance_evaporation_breakout_cmp_score_upside_vacuum",
        "conf_orderbook_resistance_evaporation_breakout_trigger_component_score",
    ]
    short_cols = [
        "conf_orderbook_support_removed_drawdown_cmp_score_support_removed",
        "conf_orderbook_support_removed_drawdown_cmp_score_downside_vacuum",
        "conf_orderbook_support_removed_drawdown_trigger_component_score",
    ]
    long_mask = max_existing(frame, long_cols).ge(0.40)
    short_mask = max_existing(frame, short_cols).ge(0.40)
    return (direction.eq("long") & long_mask) | (direction.eq("short") & short_mask)


def regime_agrees(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    bull = max_existing(frame, ["conf_price_bull_trend_regime", "px_return_24h", "px_return_72h"])
    bear = pd.concat([-num(frame, "px_return_24h"), -num(frame, "px_return_72h"), max_existing(frame, ["conf_price_bear_trend_regime"])], axis=1).max(axis=1)
    return (direction.eq("long") & bull.gt(0)) | (direction.eq("short") & bear.gt(0))


def context_safe_or_missing(frame: DataFrame) -> Series:
    if "context_source_future_violation" not in frame.columns and "context_present" not in frame.columns:
        return pd.Series(True, index=frame.index)
    violation = num(frame, "context_source_future_violation").gt(0)
    stale = num(frame, "context_source_age_hours").gt(168)
    return ~(violation | stale)


def no_support_reclaim_warning(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    warning = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "conf_epsilon_downside_first_break_exhaustion_cmp_score_reclaim_pressure",
            "conf_failed_breakdown_exhaustion_score",
        ],
    )
    return ~direction.eq("short") | warning.le(0.42)


def no_support_rebuild_warning(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    rebuild = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_epsilon_downside_first_break_exhaustion_cmp_score_support_rebuild",
        ],
    )
    return ~direction.eq("short") | rebuild.le(0.25)


def bearish_pressure_persists(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    pressure = pd.concat(
        [
            max_existing(
                frame,
                [
                    "conf_ob_bearish_pressure_agreement",
                    "conf_beta_ob_support_removed_breakdown_cmp_score_bearish_pressure",
                    "conf_epsilon_orderbook_panic_after_break_cmp_score_bearish_pressure",
                ],
            ),
            -num(frame, "px_volume_pressure_6h"),
            -num(frame, "px_volume_pressure_24h"),
            -num(frame, "st_volume_pressure_6h"),
            -num(frame, "st_volume_pressure_24h"),
        ],
        axis=1,
    ).max(axis=1)
    return ~direction.eq("short") | pressure.ge(0.33)


def downside_vacuum_still_open(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    vacuum = max_existing(
        frame,
        [
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_downside_vacuum",
            "conf_beta_ob_support_removed_breakdown_cmp_score_downside_vacuum",
            "conf_delta_orderbook_support_weak_drawdown_cmp_score_downside_vacuum",
        ],
    )
    return ~direction.eq("short") | vacuum.ge(0.20)


def breakdown_failure_warning_absent(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    warning = max_existing(
        frame,
        [
            "conf_ob_breakdown_failure",
            "conf_failed_breakdown_exhaustion_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "ob_spot_obts_breakdown_failure_score",
            "ob_linear_obts_breakdown_failure_score",
            "ob_inverse_obts_breakdown_failure_score",
        ],
    )
    return ~direction.eq("short") | warning.le(0.20)


def panic_after_break_state(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    panic = max_existing(
        frame,
        [
            "conf_epsilon_orderbook_panic_after_break_score",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_support_removed",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_downside_vacuum",
            "conf_ob_spread_fragility",
        ],
    )
    return ~direction.eq("short") | panic.ge(0.30)


def max_existing(frame: DataFrame, cols: list[str]) -> Series:
    existing = [num(frame, col) for col in cols if col in frame.columns]
    if not existing:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(existing, axis=1).max(axis=1).fillna(0.0)


def num(frame: DataFrame, col: str) -> Series:
    if col not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def markdown_report(summary: DataFrame, trades: DataFrame, filters: tuple[FilterSpec, ...]) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    top = summary[(summary["group_id"].eq("all_selected_entries")) & (summary["filter_id"].ne("base_no_extra_filter"))]
    top = top.sort_values(["verdict", "quality_delta"], ascending=[True, False])
    watch = summary[summary["verdict"].isin(["worth_more_testing", "watchlist"])].sort_values("quality_delta", ascending=False).head(40)
    lines = [
        "# Trading Lead Confluence Checks",
        "",
        f"Generated: {generated}",
        "",
        "Trader question: if we keep the current selected entries but require another source family to agree, do entries get cleaner?",
        "",
        f"Input selected trades: {len(trades)}",
        f"Filters tested: {len(filters)}",
        "",
        "## All Selected Entries",
        "",
        top[["filter_id", "trader_question", "filtered_trades", "base_trades", "win_rate", "base_win_rate", "avg_return", "base_avg_return", "profit_factor", "base_profit_factor", "quality_delta", "verdict"]].to_markdown(index=False),
        "",
        "## Watchlist Across Individual Rules",
        "",
        watch[["group_id", "filter_id", "filtered_trades", "win_rate", "base_win_rate", "avg_return", "base_avg_return", "profit_factor", "base_profit_factor", "quality_delta", "verdict"]].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Good filter: keeps enough trades and improves profit factor, average return, or win rate.",
        "- Bad filter: removes too many trades or makes quality worse.",
        "- This is an entry-quality screen only; exit tuning and risk sizing are separate stages.",
    ]
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
