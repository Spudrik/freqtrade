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
DEFAULT_TRADES = REPORTS_DIR / "trading_lead_exit_research_20260605_combined_trades.csv"


@dataclass(frozen=True)
class OverlaySpec:
    overlay_id: str
    trader_question: str
    data_family: str
    multiplier_builder: Callable[[DataFrame], Series]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate simple risk/leverage overlays on selected BTC trading leads.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
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
    overlays = build_overlays()
    summary = run_overlays(merged, overlays)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    csv_path = args.output_dir / f"trading_lead_risk_overlay_{tag}.csv"
    md_path = args.output_dir / f"trading_lead_risk_overlay_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_risk_overlay_{tag}_meta.json"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trades": str(args.trades),
        "features": str(args.features),
        "input_trades": int(len(trades)),
        "overlays": [o.overlay_id for o in overlays],
        "outputs": {"summary": str(csv_path), "markdown": str(md_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    keep = ["date", "px_", "st_", "conf_", "ob_", "context_", "ctx_"]
    cols = [c for c in frame.columns if c == "date" or any(c.startswith(prefix) for prefix in keep[1:])]
    return frame[cols]


def build_overlays() -> tuple[OverlaySpec, ...]:
    return (
        OverlaySpec(
            "score_confidence_0p5_to_1p5",
            "Should stronger entry scores get more size and weaker scores get less?",
            "entry_score",
            lambda f: 0.5 + pd.to_numeric(f["signal_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0),
        ),
        OverlaySpec(
            "directional_volume_size",
            "Should we size up when volume pressure agrees and size down when it does not?",
            "OHLCV_volume_pressure",
            lambda f: directional_volume_multiplier(f),
        ),
        OverlaySpec(
            "structure_agreement_size",
            "Should custom structure agreement increase size and disagreement reduce size?",
            "custom_structure",
            lambda f: structure_multiplier(f),
        ),
        OverlaySpec(
            "orderbook_warning_size",
            "Should orderbook agreement increase size and reclaim/rebuild warnings reduce size?",
            "orderbook",
            lambda f: orderbook_multiplier(f),
        ),
        OverlaySpec(
            "core_orderbook_quality_size",
            "For support-removal shorts, should stronger support-cleared/downside-vacuum evidence get more size?",
            "orderbook_core",
            lambda f: core_orderbook_quality_multiplier(f),
        ),
        OverlaySpec(
            "volatility_risk_off_size",
            "Should unusually large current range reduce position size?",
            "OHLCV_volatility",
            lambda f: volatility_multiplier(f),
        ),
        OverlaySpec(
            "combined_conservative_size",
            "Does a conservative mix of score, volume, structure, orderbook, and volatility improve risk-adjusted returns?",
            "combined",
            lambda f: combined_multiplier(f),
        ),
        OverlaySpec(
            "story_specific_orderbook_size",
            "Should orderbook size rules depend on the specific entry story instead of one generic warning?",
            "story_specific_orderbook",
            lambda f: story_specific_orderbook_multiplier(f),
        ),
        OverlaySpec(
            "story_specific_structure_volume_size",
            "Should structure/volume sizing depend on whether this is a breakout long, squeeze long, or breakdown short?",
            "story_specific_structure_volume",
            lambda f: story_specific_structure_volume_multiplier(f),
        ),
        OverlaySpec(
            "story_specific_conservative_size",
            "Does a conservative story-specific blend improve sizing without over-levering one broad metric?",
            "story_specific_combined",
            lambda f: story_specific_conservative_multiplier(f),
        ),
        OverlaySpec(
            "extreme_orderbook_invalidation_guard",
            "Should orderbook only reduce size when the original trade idea is clearly being invalidated?",
            "extreme_orderbook_invalidation",
            lambda f: extreme_orderbook_invalidation_multiplier(f),
        ),
        OverlaySpec(
            "structure_volume_with_extreme_ob_guard",
            "Can structure/volume sizing improve returns while extreme orderbook invalidation controls the worst thesis failures?",
            "structure_volume_plus_extreme_orderbook_guard",
            lambda f: (story_specific_structure_volume_multiplier(f) * extreme_orderbook_invalidation_multiplier(f)).clip(0.40, 1.35),
        ),
        OverlaySpec(
            "btc_sieve_quality_story_size",
            "For the current BTC sieve quality block, should size follow the specific rule story: breakout, reclaim, continuation, LVN travel, or rejection?",
            "btc_sieve_quality_story",
            lambda f: btc_sieve_quality_story_multiplier(f),
        ),
        OverlaySpec(
            "btc_sieve_quality_story_with_ob_guard",
            "For the current BTC sieve quality block, can story-specific sizing help while orderbook invalidation cuts risk when the book argues against the idea?",
            "btc_sieve_quality_story_plus_orderbook_guard",
            lambda f: (btc_sieve_quality_story_multiplier(f) * btc_sieve_quality_orderbook_guard(f)).clip(0.40, 1.45),
        ),
    )


def run_overlays(frame: DataFrame, overlays: tuple[OverlaySpec, ...]) -> DataFrame:
    rows: list[dict[str, object]] = []
    base = summarize_scaled(frame, pd.Series(1.0, index=frame.index), "base_1x", "No risk overlay; every selected trade uses the same size.", "base")
    rows.append(base)
    for spec in overlays:
        mult = spec.multiplier_builder(frame).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 1.75)
        rows.append(summarize_scaled(frame, mult, spec.overlay_id, spec.trader_question, spec.data_family, base))
    result = pd.DataFrame(rows)
    return result.sort_values(["verdict", "risk_adjusted_delta", "total_return"], ascending=[True, False, False]).reset_index(drop=True)


def summarize_scaled(frame: DataFrame, multiplier: Series, overlay_id: str, trader_question: str, data_family: str, base: dict[str, object] | None = None) -> dict[str, object]:
    returns = pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0)
    scaled = returns * multiplier
    equity = (1.0 + scaled).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = scaled[scaled.gt(0)]
    losses = scaled[scaled.lt(0)]
    pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
    row = {
        "overlay_id": overlay_id,
        "trader_question": trader_question,
        "data_family": data_family,
        "trades": int(len(scaled)),
        "avg_multiplier": float(multiplier.mean()),
        "min_multiplier": float(multiplier.min()),
        "max_multiplier": float(multiplier.max()),
        "win_rate": float(scaled.gt(0).mean()),
        "avg_trade_return": float(scaled.mean()),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "profit_factor": pf,
    }
    if base is None:
        row.update({"risk_adjusted_delta": 0.0, "return_delta": 0.0, "drawdown_delta": 0.0, "verdict": "baseline"})
        return row
    return_delta = float(row["total_return"]) - float(base["total_return"])
    drawdown_delta = float(row["max_drawdown"]) - float(base["max_drawdown"])
    risk_adjusted_delta = return_delta + drawdown_delta + (float(row["profit_factor"]) - float(base["profit_factor"])) / 5.0
    verdict = "watchlist" if risk_adjusted_delta > 0.03 and float(row["total_return"]) >= float(base["total_return"]) else "reject_for_now"
    row.update({"risk_adjusted_delta": risk_adjusted_delta, "return_delta": return_delta, "drawdown_delta": drawdown_delta, "verdict": verdict})
    return row


def directional_volume_multiplier(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    bull = max_existing(frame, ["conf_volume_bullish_confirmation", "px_volume_pressure_24h", "st_volume_pressure_24h"])
    bear = max_existing(frame, ["conf_volume_bearish_confirmation", "-px_volume_pressure_24h", "-st_volume_pressure_24h"])
    agrees = (direction.eq("long") & bull.ge(0.60)) | (direction.eq("short") & bear.ge(0.60))
    disagrees = (direction.eq("long") & bear.ge(0.60)) | (direction.eq("short") & bull.ge(0.60))
    return pd.Series(np.where(agrees, 1.25, np.where(disagrees, 0.60, 1.0)), index=frame.index)


def structure_multiplier(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    long_score = max_existing(frame, ["conf_structure_breakout_trigger_score", "st_1h_ms_bos_to_bull", "st_4h_ms_bos_to_bull"])
    short_score = max_existing(frame, ["conf_structure_breakdown_trigger_score", "st_1h_ms_bos_to_bear", "st_4h_ms_bos_to_bear"])
    agrees = (direction.eq("long") & long_score.ge(0.34)) | (direction.eq("short") & short_score.ge(0.34))
    disagrees = (direction.eq("long") & short_score.ge(0.67)) | (direction.eq("short") & long_score.ge(0.67))
    return pd.Series(np.where(agrees, 1.20, np.where(disagrees, 0.70, 1.0)), index=frame.index)


def orderbook_multiplier(frame: DataFrame) -> Series:
    present = orderbook_present(frame)
    direction = frame["direction"].astype(str).str.lower()
    long_agree = max_existing(
        frame,
        [
            "conf_ob_resistance_removed_strength",
            "conf_ob_upside_vacuum_after_resistance_removed",
            "conf_ob_bullish_pressure_agreement",
        ],
    ).ge(0.40)
    short_agree = max_existing(
        frame,
        [
            "conf_ob_support_removed_strength",
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_ob_support_cleared",
            "conf_ob_bearish_pressure_agreement",
        ],
    ).ge(0.40)
    warning = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_ob_support_bounce",
            "conf_ob_bid_absorption",
        ],
    ).ge(0.50)
    agrees = present & ((direction.eq("long") & long_agree) | (direction.eq("short") & short_agree))
    present_disagree = present & ~agrees & ~warning
    return pd.Series(np.where(warning, 0.60, np.where(agrees, 1.15, np.where(present_disagree, 0.80, 1.0))), index=frame.index)


def core_orderbook_quality_multiplier(frame: DataFrame) -> Series:
    direction = frame["direction"].astype(str).str.lower()
    support_removed = max_existing(
        frame,
        [
            "conf_ob_support_removed_strength",
            "conf_ob_support_cleared",
            "conf_beta_ob_support_removed_breakdown_cmp_score_support_removed",
        ],
    )
    downside_vacuum = max_existing(
        frame,
        [
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_beta_ob_downside_vacuum_breakdown_cmp_score_downside_vacuum",
        ],
    )
    bearish_pressure = max_existing(
        frame,
        [
            "conf_ob_bearish_pressure_agreement",
            "conf_beta_ob_support_removed_breakdown_cmp_score_bearish_pressure",
        ],
    )
    reclaim_warning = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_ob_support_bounce",
            "conf_ob_bid_absorption",
        ],
    )
    quality = (support_removed + downside_vacuum + bearish_pressure) / 3.0
    short = direction.eq("short")
    return pd.Series(
        np.where(
            short & reclaim_warning.ge(0.50),
            0.60,
            np.where(short & quality.ge(0.38), 1.30, np.where(short & quality.ge(0.25), 1.10, 0.85)),
        ),
        index=frame.index,
    )


def volatility_multiplier(frame: DataFrame) -> Series:
    range_rank = max_existing(frame, ["px_range_1h", "px_volume_z_24h"]).rolling(720, min_periods=24).rank(pct=True)
    return pd.Series(np.where(range_rank.ge(0.90), 0.65, np.where(range_rank.le(0.40), 1.10, 1.0)), index=frame.index)


def combined_multiplier(frame: DataFrame) -> Series:
    parts = [
        0.35 * (0.5 + pd.to_numeric(frame["signal_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)),
        0.20 * directional_volume_multiplier(frame),
        0.20 * structure_multiplier(frame),
        0.15 * orderbook_multiplier(frame),
        0.10 * volatility_multiplier(frame),
    ]
    return sum(parts).clip(0.35, 1.50)


def story_specific_orderbook_multiplier(frame: DataFrame) -> Series:
    rule = frame["rule_id"].fillna("").astype(str)
    mult = pd.Series(1.0, index=frame.index)

    support_removed = max_existing(frame, ["conf_ob_support_removed_strength", "conf_ob_support_cleared"])
    downside_vacuum = max_existing(frame, ["conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum"])
    bearish_pressure = max_existing(frame, ["conf_ob_bearish_pressure_agreement"])
    support_reclaim = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_ob_support_bounce",
            "conf_ob_bid_absorption",
            "conf_ob_breakdown_failure",
        ],
    )
    resistance_removed = max_existing(frame, ["conf_ob_resistance_removed_strength", "conf_ob_resistance_cleared"])
    upside_vacuum = max_existing(frame, ["conf_ob_upside_vacuum_after_resistance_removed", "conf_ob_upside_vacuum"])
    bullish_pressure = max_existing(frame, ["conf_ob_bullish_pressure_agreement"])
    resistance_warning = max_existing(
        frame,
        [
            "conf_ob_resistance_rejection",
            "conf_ob_ask_absorption",
            "conf_ob_breakout_failure",
            "conf_ob_resistance_added_strength",
        ],
    )

    support_break_short = rule.str.contains("support_break|support_cleared|val_vacuum|prior_day_low|rectangle_lower|tlv2_sup", case=False, regex=True)
    breakout_long = rule.str.contains("break_orderbook_accept|squeeze_upper|tlv2_res|triangle_upper|break_above|breakout", case=False, regex=True)
    val_accept_short = rule.str.contains("vp_val_accept_short", case=False, regex=False)

    short_quality = (support_removed + downside_vacuum + bearish_pressure) / 3.0
    long_quality = (resistance_removed + upside_vacuum + bullish_pressure) / 3.0

    mult.loc[support_break_short & short_quality.ge(0.45) & support_reclaim.lt(0.35)] = 1.25
    mult.loc[support_break_short & support_reclaim.ge(0.45)] = 0.55
    mult.loc[val_accept_short & support_reclaim.ge(0.30)] = 0.45
    mult.loc[val_accept_short & short_quality.ge(0.50) & support_reclaim.lt(0.25)] = 1.15

    mult.loc[breakout_long & long_quality.ge(0.45) & resistance_warning.lt(0.35)] = 1.20
    mult.loc[breakout_long & resistance_warning.ge(0.45)] = 0.60
    return mult


def story_specific_structure_volume_multiplier(frame: DataFrame) -> Series:
    rule = frame["rule_id"].fillna("").astype(str)
    mult = pd.Series(1.0, index=frame.index)

    bull_structure = max_existing(frame, ["conf_structure_breakout_trigger_score", "conf_structure_bullish_state_score", "st_4h_ms_bos_to_bull", "st_1h_ms_bos_to_bull"])
    bear_structure = max_existing(frame, ["conf_structure_breakdown_trigger_score", "conf_structure_bearish_state_score", "st_4h_ms_bos_to_bear", "st_1h_ms_bos_to_bear"])
    bull_volume = max_existing(frame, ["conf_volume_bullish_confirmation", "px_volume_pressure_24h", "px_volume_z_24h"])
    bear_volume = max_existing(frame, ["conf_volume_bearish_confirmation", "-px_volume_pressure_24h", "px_volume_z_24h"])
    compression_release = max_existing(frame, ["st_4h_pg2_compression_squeeze_active", "st_4h_pg2_triangle_squeeze_active", "st_1h_pg2_compression_squeeze_active"])
    value_accept_short = max_existing(frame, ["st_4h_vp_below_value_area", "st_1h_vp_below_value_area", "conf_structure_breakdown_trigger_score"])

    breakout_long = rule.str.contains("break_orderbook_accept|squeeze_upper|tlv2_res|triangle_upper|break_above|breakout", case=False, regex=True)
    squeeze_long = rule.str.contains("squeeze|triangle_upper", case=False, regex=True)
    support_break_short = rule.str.contains("support_break|support_cleared|val_vacuum|prior_day_low|rectangle_lower|tlv2_sup|vp_val_accept_short", case=False, regex=True)

    mult.loc[breakout_long & bull_structure.ge(0.45) & bull_volume.ge(0.40)] = 1.25
    mult.loc[breakout_long & bear_structure.ge(0.55)] = 0.65
    mult.loc[squeeze_long & compression_release.ge(0.50) & bull_volume.ge(0.35)] = 1.20
    mult.loc[support_break_short & bear_structure.ge(0.45) & bear_volume.ge(0.35)] = 1.20
    mult.loc[support_break_short & bull_structure.ge(0.55)] = 0.65
    mult.loc[rule.str.contains("vp_val_accept_short", case=False, regex=False) & value_accept_short.lt(0.25)] = 0.60
    return mult


def story_specific_conservative_multiplier(frame: DataFrame) -> Series:
    score = 0.5 + pd.to_numeric(frame["signal_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    parts = [
        0.30 * score,
        0.25 * story_specific_structure_volume_multiplier(frame),
        0.25 * story_specific_orderbook_multiplier(frame),
        0.20 * volatility_multiplier(frame),
    ]
    return sum(parts).clip(0.40, 1.45)


def btc_sieve_quality_story_multiplier(frame: DataFrame) -> Series:
    rule = frame["rule_id"].fillna("").astype(str)
    mult = pd.Series(1.0, index=frame.index)

    bull_structure = max_existing(
        frame,
        [
            "conf_structure_breakout_trigger_score",
            "conf_structure_bullish_state_score",
            "st_1h_ms_bos_to_bull",
            "st_4h_ms_bos_to_bull",
            "st_8h_ms_bos_to_bull",
            "st_1d_ms_bos_to_bull",
        ],
    )
    bear_structure = max_existing(
        frame,
        [
            "conf_structure_breakdown_trigger_score",
            "conf_structure_bearish_state_score",
            "st_1h_ms_bos_to_bear",
            "st_4h_ms_bos_to_bear",
            "st_8h_ms_bos_to_bear",
            "st_1d_ms_bos_to_bear",
        ],
    )
    bull_volume = max_existing(frame, ["conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h", "px_volume_z_24h"])
    bear_volume = max_existing(frame, ["conf_volume_bearish_confirmation", "-px_volume_pressure_6h", "-px_volume_pressure_24h", "px_volume_z_24h"])
    compression_release = max_existing(frame, ["st_1h_pg2_compression_squeeze_active", "st_4h_pg2_compression_squeeze_active", "st_8h_pg2_compression_squeeze_active"])
    vp_above_value = max_existing(frame, ["st_1h_vp_above_value_area", "st_4h_vp_above_value_area", "st_8h_vp_above_value_area", "st_1d_vp_above_value_area"])
    vp_below_value = max_existing(frame, ["st_1h_vp_below_value_area", "st_4h_vp_below_value_area", "st_8h_vp_below_value_area", "st_1d_vp_below_value_area"])
    lvn_thin_up = max_existing(frame, ["st_1h_vp_lvn_above_thinness", "st_4h_vp_lvn_above_thinness", "st_8h_vp_lvn_above_thinness"])
    resistance_stack = max_existing(frame, ["conf_structure_resistance_stack_score", "st_1h_tlv2_resistance_stack_score", "st_4h_tlv2_resistance_stack_score"])
    support_stack = max_existing(frame, ["conf_structure_support_stack_score", "st_1h_tlv2_support_stack_score", "st_4h_tlv2_support_stack_score"])

    resistance_break_long = rule.str.contains("res_break|resistance_breakout|prior_month_high_break", case=False, regex=True)
    reclaim_long = rule.str.contains("sup_reclaim", case=False, regex=False)
    bos_continuation_long = rule.str.contains("bos_bull_continuation", case=False, regex=False)
    lvn_fast_long = rule.str.contains("lvn_fast_traverse", case=False, regex=False)
    poc_reject_short = rule.str.contains("poc_reject_short", case=False, regex=False)
    double_top_short = rule.str.contains("double_top_present_short", case=False, regex=False)

    long_confirm = bull_structure.ge(0.40) & bull_volume.ge(0.35)
    long_conflict = bear_structure.ge(0.55)
    short_confirm = bear_structure.ge(0.35) & bear_volume.ge(0.30)
    short_conflict = bull_structure.ge(0.55)

    mult.loc[resistance_break_long & long_confirm & vp_above_value.ge(0.25)] = 1.30
    mult.loc[resistance_break_long & long_confirm & compression_release.ge(0.40)] = 1.25
    mult.loc[resistance_break_long & long_conflict] = 0.65
    mult.loc[reclaim_long & support_stack.ge(0.25) & bull_volume.ge(0.35)] = 1.20
    mult.loc[reclaim_long & bear_structure.ge(0.55)] = 0.65
    mult.loc[bos_continuation_long & long_confirm] = 1.20
    mult.loc[bos_continuation_long & long_conflict] = 0.70
    mult.loc[lvn_fast_long & lvn_thin_up.ge(0.25) & bull_volume.ge(0.35)] = 1.25
    mult.loc[lvn_fast_long & bull_volume.lt(0.15)] = 0.75
    mult.loc[poc_reject_short & short_confirm & vp_below_value.ge(0.20)] = 1.20
    mult.loc[poc_reject_short & short_conflict] = 0.65
    mult.loc[double_top_short & resistance_stack.ge(0.25) & bear_volume.ge(0.25)] = 1.15
    mult.loc[double_top_short & short_conflict] = 0.70
    return mult


def btc_sieve_quality_orderbook_guard(frame: DataFrame) -> Series:
    rule = frame["rule_id"].fillna("").astype(str)
    mult = pd.Series(1.0, index=frame.index)
    present = orderbook_present(frame)

    resistance_removed = max_existing(frame, ["conf_ob_resistance_removed_strength", "conf_ob_resistance_cleared", "conf_ob_upside_vacuum_after_resistance_removed"])
    resistance_warning = max_existing(frame, ["conf_ob_resistance_rejection", "conf_ob_ask_absorption", "conf_ob_breakout_failure", "conf_ob_resistance_added_strength"])
    support_removed = max_existing(frame, ["conf_ob_support_removed_strength", "conf_ob_support_cleared", "conf_ob_downside_vacuum_after_support_removed"])
    support_warning = max_existing(frame, ["conf_ob_support_bounce", "conf_ob_bid_absorption", "conf_ob_breakdown_failure", "conf_ob_support_added_strength"])
    bullish_pressure = max_existing(frame, ["conf_ob_bullish_pressure_agreement"])
    bearish_pressure = max_existing(frame, ["conf_ob_bearish_pressure_agreement"])

    long_rule = rule.str.contains("long", case=False, regex=False)
    short_rule = rule.str.contains("short", case=False, regex=False)
    mult.loc[present & long_rule & resistance_removed.ge(0.45) & bullish_pressure.ge(0.35) & resistance_warning.lt(0.45)] = 1.15
    mult.loc[present & long_rule & resistance_warning.ge(0.65) & resistance_removed.lt(0.30)] = 0.55
    mult.loc[present & short_rule & support_removed.ge(0.45) & bearish_pressure.ge(0.35) & support_warning.lt(0.45)] = 1.15
    mult.loc[present & short_rule & support_warning.ge(0.65) & support_removed.lt(0.30)] = 0.55
    return mult


def extreme_orderbook_invalidation_multiplier(frame: DataFrame) -> Series:
    rule = frame["rule_id"].fillna("").astype(str)
    mult = pd.Series(1.0, index=frame.index)

    support_reclaim = max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "conf_ob_support_bounce",
            "conf_ob_bid_absorption",
            "conf_ob_breakdown_failure",
        ],
    )
    resistance_reject = max_existing(
        frame,
        [
            "conf_ob_resistance_rejection",
            "conf_ob_ask_absorption",
            "conf_ob_breakout_failure",
            "conf_ob_resistance_added_strength",
        ],
    )
    support_removed = max_existing(frame, ["conf_ob_support_removed_strength", "conf_ob_support_cleared", "conf_ob_downside_vacuum_after_support_removed"])
    resistance_removed = max_existing(frame, ["conf_ob_resistance_removed_strength", "conf_ob_resistance_cleared", "conf_ob_upside_vacuum_after_resistance_removed"])

    support_break_short = rule.str.contains("support_break|support_cleared|val_vacuum|prior_day_low|rectangle_lower|tlv2_sup|vp_val_accept_short", case=False, regex=True)
    breakout_long = rule.str.contains("break_orderbook_accept|squeeze_upper|tlv2_res|triangle_upper|break_above|breakout", case=False, regex=True)

    mult.loc[support_break_short & support_reclaim.ge(0.70) & support_removed.lt(0.25)] = 0.35
    mult.loc[support_break_short & support_reclaim.ge(0.60)] = 0.60
    mult.loc[breakout_long & resistance_reject.ge(0.70) & resistance_removed.lt(0.25)] = 0.35
    mult.loc[breakout_long & resistance_reject.ge(0.60)] = 0.60
    return mult


def orderbook_present(frame: DataFrame) -> Series:
    cols = [c for c in frame.columns if c.startswith("ob_") and c.endswith("_feature_present")]
    if not cols:
        return pd.Series(False, index=frame.index)
    return frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0.0).any(axis=1)


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


def markdown_report(summary: DataFrame) -> str:
    lines = [
        "# Trading Lead Risk Overlay",
        "",
        "This keeps entries and exits fixed, then tests whether simple position-size multipliers improve the trade block.",
        "",
        summary[["overlay_id", "trades", "avg_multiplier", "win_rate", "avg_trade_return", "total_return", "max_drawdown", "profit_factor", "risk_adjusted_delta", "verdict"]].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `watchlist` means the overlay improved return and risk-adjusted score versus equal 1x sizing.",
        "- This is not final leverage logic. It is a first screen for whether extra metrics should size trades up/down.",
        "- News/context is not used here as a directional sizing input because source-specific context quality is still parked.",
    ]
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
