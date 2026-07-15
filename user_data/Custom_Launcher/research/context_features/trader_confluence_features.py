from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


ORDERBOOK_PREFIXES = ("ob_spot_", "ob_linear_", "ob_inverse_")
STRUCTURE_TIMEFRAMES = ("1h", "4h", "1d")


def append_price_features(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    close = num(out, "close")
    high = num(out, "high")
    low = num(out, "low")
    volume = num(out, "volume").clip(lower=0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    body_pressure = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)
    signed_volume = body_pressure * volume.fillna(0.0)
    out["px_return_1h"] = close.pct_change(1, fill_method=None)
    out["px_return_3h"] = close.pct_change(3, fill_method=None)
    out["px_return_6h"] = close.pct_change(6, fill_method=None)
    out["px_return_24h"] = close.pct_change(24, fill_method=None)
    out["px_return_72h"] = close.pct_change(72, fill_method=None)
    out["px_range_1h"] = (high - low) / close.replace(0.0, np.nan)
    for window in (6, 24, 72):
        min_periods = max(3, window // 2)
        prior_high = high.shift(1).rolling(window, min_periods=min_periods).max()
        prior_low = low.shift(1).rolling(window, min_periods=min_periods).min()
        prior_range = (prior_high - prior_low).replace(0.0, np.nan)
        out[f"px_range_position_{window}h"] = ((close - prior_low) / prior_range).clip(0.0, 1.0)
        out[f"px_close_breakout_{window}h"] = close.gt(prior_high).astype(float).where(prior_high.notna())
        out[f"px_close_breakdown_{window}h"] = close.lt(prior_low).astype(float).where(prior_low.notna())
        out[f"px_volume_z_{window}h"] = zscore(volume, window, min_periods=min_periods)
        out[f"px_volume_pressure_{window}h"] = signed_volume.rolling(window, min_periods=min_periods).sum() / volume.rolling(window, min_periods=min_periods).sum().replace(0.0, np.nan)
    for horizon in (1, 3, 6, 24, 72):
        valid = continuous_past_mask(out, horizon)
        column = f"px_return_{horizon}h"
        if column in out:
            out.loc[~valid, column] = np.nan
    for window in (6, 24, 72):
        valid = continuous_past_mask(out, window)
        for column in (
            f"px_range_position_{window}h",
            f"px_close_breakout_{window}h",
            f"px_close_breakdown_{window}h",
            f"px_volume_z_{window}h",
            f"px_volume_pressure_{window}h",
        ):
            if column in out:
                out.loc[~valid, column] = np.nan
    return out


def append_future_labels(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    close = num(out, "close")
    high = num(out, "high")
    low = num(out, "low")
    for horizon in (1, 3, 6, 24, 72):
        out[f"future_return_{horizon}h"] = close.shift(-horizon) / close - 1.0
        future_high = future_extreme(high, horizon, "max")
        future_low = future_extreme(low, horizon, "min")
        out[f"future_max_upside_{horizon}h"] = future_high / close - 1.0
        out[f"future_max_drawdown_{horizon}h"] = future_low / close - 1.0
    prior_high_24h = high.shift(1).rolling(24, min_periods=12).max()
    prior_low_24h = low.shift(1).rolling(24, min_periods=12).min()
    future_high_6h = future_extreme(high, 6, "max")
    future_low_6h = future_extreme(low, 6, "min")
    future_low_3h = future_extreme(low, 3, "min")
    future_high_24h = future_extreme(high, 24, "max")
    future_low_24h = future_extreme(low, 24, "min")
    future_close_3h = close.shift(-3)
    future_close_6h = close.shift(-6)
    future_close_24h = close.shift(-24)
    breakout_attempt_6h = future_high_6h.ge(prior_high_24h * 1.001)
    breakdown_attempt_6h = future_low_6h.le(prior_low_24h * 0.999)
    breakout_attempt_24h = future_high_24h.ge(prior_high_24h * 1.001)
    breakdown_attempt_24h = future_low_24h.le(prior_low_24h * 0.999)
    out["breakout_attempt_next_6h"] = breakout_attempt_6h.astype(float).where(prior_high_24h.notna())
    out["breakout_success_next_6h"] = (breakout_attempt_6h & future_close_6h.gt(prior_high_24h * 1.001)).astype(float).where(prior_high_24h.notna())
    out["breakout_failure_next_6h"] = (breakout_attempt_6h & future_close_6h.lt(prior_high_24h * 0.999)).astype(float).where(prior_high_24h.notna())
    out["breakdown_attempt_next_6h"] = breakdown_attempt_6h.astype(float).where(prior_low_24h.notna())
    out["breakdown_success_next_6h"] = (breakdown_attempt_6h & future_close_6h.lt(prior_low_24h * 0.999)).astype(float).where(prior_low_24h.notna())
    out["breakdown_failure_next_6h"] = (breakdown_attempt_6h & future_close_6h.gt(prior_low_24h * 1.001)).astype(float).where(prior_low_24h.notna())
    out["failed_breakout_next_24h"] = (breakout_attempt_24h & future_close_24h.lt(prior_high_24h)).astype(float).where(prior_high_24h.notna())
    out["failed_breakdown_next_24h"] = (breakdown_attempt_24h & future_close_24h.gt(prior_low_24h)).astype(float).where(prior_low_24h.notna())
    out["large_drawdown_next_6h"] = out["future_max_drawdown_6h"].le(-0.03).astype(float).where(out["future_max_drawdown_6h"].notna())
    out["large_drawdown_next_24h"] = out["future_max_drawdown_24h"].le(-0.04).astype(float).where(out["future_max_drawdown_24h"].notna())
    out["downside_continuation_next_3h"] = (
        future_low_3h.le(close * 0.985) & future_close_3h.lt(close * 0.995)
    ).astype(float).where(future_low_3h.notna() & future_close_3h.notna() & close.notna())
    out["downside_continuation_next_6h"] = (
        future_low_6h.le(close * 0.980) & future_close_6h.lt(close * 0.995)
    ).astype(float).where(future_low_6h.notna() & future_close_6h.notna() & close.notna())
    out["downside_exhaustion_next_6h"] = (
        future_low_6h.gt(close * 0.990) | future_close_6h.gt(close * 1.003)
    ).astype(float).where(future_low_6h.notna() & future_close_6h.notna() & close.notna())
    out["support_reclaim_next_6h"] = (
        close.lt(prior_low_24h * 1.001) & future_close_6h.gt(prior_low_24h * 1.001)
    ).astype(float).where(prior_low_24h.notna() & future_close_6h.notna() & close.notna())
    out["fakeout_next_24h"] = (
        out["failed_breakout_next_24h"].eq(1.0) | out["failed_breakdown_next_24h"].eq(1.0)
    ).astype(float).where(out["future_return_24h"].notna())
    out["time_to_plus_2pct"] = time_to_threshold(out, up_pct=0.02, down_pct=None, horizon=24)
    out["time_to_minus_2pct"] = time_to_threshold(out, up_pct=None, down_pct=-0.02, horizon=24)
    out["hit_plus_3pct_before_minus_2pct"] = hit_before(out, up_pct=0.03, down_pct=-0.02, horizon=24, direction="up")
    out["hit_minus_3pct_before_plus_2pct"] = hit_before(out, up_pct=0.03, down_pct=-0.02, horizon=24, direction="down")
    for horizon in (1, 3, 6, 24, 72):
        valid = continuous_future_mask(out, horizon)
        horizon_columns = [
            f"future_return_{horizon}h",
            f"future_max_upside_{horizon}h",
            f"future_max_drawdown_{horizon}h",
        ]
        if horizon == 3:
            horizon_columns.append("downside_continuation_next_3h")
        if horizon == 6:
            horizon_columns.extend(
                [
                    "breakout_attempt_next_6h",
                    "breakout_success_next_6h",
                    "breakout_failure_next_6h",
                    "breakdown_attempt_next_6h",
                    "breakdown_success_next_6h",
                    "breakdown_failure_next_6h",
                    "large_drawdown_next_6h",
                    "downside_continuation_next_6h",
                    "downside_exhaustion_next_6h",
                    "support_reclaim_next_6h",
                ]
            )
        if horizon == 24:
            horizon_columns.extend(
                [
                    "failed_breakout_next_24h",
                    "failed_breakdown_next_24h",
                    "large_drawdown_next_24h",
                    "fakeout_next_24h",
                    "time_to_plus_2pct",
                    "time_to_minus_2pct",
                    "hit_plus_3pct_before_minus_2pct",
                    "hit_minus_3pct_before_plus_2pct",
                ]
            )
        for column in horizon_columns:
            if column in out:
                out.loc[~valid, column] = np.nan
    return out


def append_confluence_features(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    out = append_structure_state(out)
    out = append_orderbook_state(out)
    out = append_context_state(out)
    out = append_hypothesis_state(out)
    numeric = out.select_dtypes(include=["number", "bool"]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def append_structure_state(out: DataFrame) -> DataFrame:
    resistance_bits = []
    support_bits = []
    breakout_bits = []
    breakdown_bits = []
    bullish_structure_bits = []
    bearish_structure_bits = []
    for tf in STRUCTURE_TIMEFRAMES:
        resistance_bits.extend(
            [
                num(out, f"st_{tf}_tlv2_resistance_score_rank0").ge(0.50) & num(out, f"st_{tf}_tlv2_resistance_distance_atr_rank0").abs().le(1.25),
                flag(out, f"st_{tf}_vp_above_value_area"),
                num(out, f"st_{tf}_vp_value_area_position").ge(0.80),
                flag(out, f"st_{tf}_vp_upper_rejection_with_pressure"),
                flag(out, f"st_{tf}_vp_hvn_above_reject"),
            ]
        )
        support_bits.extend(
            [
                num(out, f"st_{tf}_tlv2_support_score_rank0").ge(0.50) & num(out, f"st_{tf}_tlv2_support_distance_atr_rank0").abs().le(1.25),
                flag(out, f"st_{tf}_vp_below_value_area"),
                num(out, f"st_{tf}_vp_value_area_position").le(0.20),
                flag(out, f"st_{tf}_vp_lower_rejection_with_pressure"),
                flag(out, f"st_{tf}_vp_hvn_below_reclaim"),
            ]
        )
        breakout_bits.extend([flag(out, f"st_{tf}_ms_bos_to_bull"), flag(out, f"st_{tf}_ms_choch_to_bull"), flag(out, f"st_{tf}_vp_vah_breakout_with_pressure")])
        breakdown_bits.extend([flag(out, f"st_{tf}_ms_bos_to_bear"), flag(out, f"st_{tf}_ms_choch_to_bear"), flag(out, f"st_{tf}_vp_val_breakdown_with_pressure")])
        bullish_structure_bits.extend(
            [
                flag(out, f"st_{tf}_ms_bos_to_bull"),
                flag(out, f"st_{tf}_ms_choch_to_bull"),
                flag(out, f"st_{tf}_ms_higher_high"),
                flag(out, f"st_{tf}_ms_higher_low"),
                num(out, f"st_{tf}_vp_score_long").gt(0.0),
            ]
        )
        bearish_structure_bits.extend(
            [
                flag(out, f"st_{tf}_ms_bos_to_bear"),
                flag(out, f"st_{tf}_ms_choch_to_bear"),
                flag(out, f"st_{tf}_ms_lower_high"),
                flag(out, f"st_{tf}_ms_lower_low"),
                num(out, f"st_{tf}_vp_score_short").gt(0.0),
            ]
        )
    out["conf_structure_resistance_stack_count"] = sum_bool(resistance_bits)
    out["conf_structure_support_stack_count"] = sum_bool(support_bits)
    out["conf_structure_breakout_trigger_count"] = sum_bool(breakout_bits) + flag(out, "px_close_breakout_24h").astype(float)
    out["conf_structure_breakdown_trigger_count"] = sum_bool(breakdown_bits) + flag(out, "px_close_breakdown_24h").astype(float)
    out["conf_structure_resistance_stack_score"] = clamp01(out["conf_structure_resistance_stack_count"] / 5.0)
    out["conf_structure_support_stack_score"] = clamp01(out["conf_structure_support_stack_count"] / 5.0)
    out["conf_structure_breakout_trigger_score"] = clamp01(out["conf_structure_breakout_trigger_count"] / 3.0)
    out["conf_structure_breakdown_trigger_score"] = clamp01(out["conf_structure_breakdown_trigger_count"] / 3.0)
    out["conf_structure_bullish_state_score"] = clamp01(sum_bool(bullish_structure_bits) / 6.0)
    out["conf_structure_bearish_state_score"] = clamp01(sum_bool(bearish_structure_bits) / 6.0)
    out["conf_price_bull_trend_regime"] = clamp01(
        num(out, "px_return_24h").fillna(0.0).clip(lower=0.0) * 10.0
        + num(out, "px_return_72h").fillna(0.0).clip(lower=0.0) * 5.0
        + num(out, "px_range_position_72h").fillna(0.5).sub(0.50).clip(lower=0.0) * 1.5
        + out["conf_structure_bullish_state_score"].fillna(0.0) * 0.5
    )
    range_mean_24h = num(out, "px_range_1h").rolling(24, min_periods=12).mean()
    range_percentile_30d = rolling_percentile_current(range_mean_24h, 720, min_periods=168).fillna(0.5)
    out["conf_price_compression_24h"] = clamp01(1.0 - range_percentile_30d)
    out["conf_price_range_expansion_24h"] = clamp01(range_percentile_30d)
    out["conf_volume_bullish_confirmation"] = clamp01((num(out, "px_volume_z_24h").fillna(0.0).clip(lower=0.0) / 3.0) + (num(out, "px_volume_pressure_24h").fillna(0.0).clip(lower=0.0)))
    out["conf_volume_bearish_confirmation"] = clamp01((num(out, "px_volume_z_24h").fillna(0.0).clip(lower=0.0) / 3.0) + (-num(out, "px_volume_pressure_24h").fillna(0.0).clip(upper=0.0)))
    out["conf_volume_bullish_impulse_short"] = clamp01((num(out, "px_volume_z_6h").fillna(0.0).clip(lower=0.0) / 2.5) + (num(out, "px_volume_pressure_6h").fillna(0.0).clip(lower=0.0)))
    out["conf_volume_bearish_impulse_short"] = clamp01((num(out, "px_volume_z_6h").fillna(0.0).clip(lower=0.0) / 2.5) + (-num(out, "px_volume_pressure_6h").fillna(0.0).clip(upper=0.0)))
    out["conf_range_position_high_24h"] = clamp01((num(out, "px_range_position_24h").fillna(0.5) - 0.65) / 0.35)
    out["conf_range_position_low_24h"] = clamp01((0.35 - num(out, "px_range_position_24h").fillna(0.5)) / 0.35)
    out["conf_range_position_high_72h"] = clamp01((num(out, "px_range_position_72h").fillna(0.5) - 0.65) / 0.35)
    out["conf_range_position_low_72h"] = clamp01((0.35 - num(out, "px_range_position_72h").fillna(0.5)) / 0.35)
    out["conf_range_position_high_stack"] = pd.concat(
        [out["conf_range_position_high_24h"], out["conf_range_position_high_72h"]],
        axis=1,
    ).max(axis=1, skipna=True)
    out["conf_range_position_low_stack"] = pd.concat(
        [out["conf_range_position_low_24h"], out["conf_range_position_low_72h"]],
        axis=1,
    ).max(axis=1, skipna=True)
    out["conf_near_range_high"] = num(out, "px_range_position_72h").ge(0.75).astype(float)
    out["conf_near_range_low"] = num(out, "px_range_position_72h").le(0.25).astype(float)
    out["conf_inside_range_mid"] = num(out, "px_range_position_72h").between(0.25, 0.75).astype(float)
    out["conf_lvn_up_thinness"] = max_existing(out, [f"st_{tf}_vp_lvn_above_thinness" for tf in STRUCTURE_TIMEFRAMES])
    out["conf_lvn_down_thinness"] = max_existing(out, [f"st_{tf}_vp_lvn_below_thinness" for tf in STRUCTURE_TIMEFRAMES])
    return out


def append_orderbook_state(out: DataFrame) -> DataFrame:
    present_cols = [f"{prefix}present" for prefix in ORDERBOOK_PREFIXES if f"{prefix}present" in out]
    if present_cols:
        out["conf_ob_venue_count_present"] = sum(num(out, col).fillna(0.0).gt(0.0).astype(float) for col in present_cols)
    else:
        out["conf_ob_venue_count_present"] = 0.0
    out["conf_ob_resistance_removed_strength"] = mean_existing(out, [f"{p}obts_resistance_zone_removed_strength_1h" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_support_removed_strength"] = mean_existing(out, [f"{p}obts_support_zone_removed_strength_1h" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_resistance_added_strength"] = mean_existing(out, [f"{p}obts_resistance_zone_added_strength_1h" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_support_added_strength"] = mean_existing(out, [f"{p}obts_support_zone_added_strength_1h" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_upside_vacuum"] = mean_existing(out, [f"{p}obts_upside_liquidity_vacuum_near" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_downside_vacuum"] = mean_existing(out, [f"{p}obts_downside_liquidity_vacuum_near" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_upside_vacuum_after_resistance_removed"] = mean_existing(out, [f"{p}obts_upside_vacuum_after_resistance_removed" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_downside_vacuum_after_support_removed"] = mean_existing(out, [f"{p}obts_downside_vacuum_after_support_removed" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_pressure_direction_mean"] = mean_existing(out, [f"{p}obts_pressure_direction" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_pressure_direction_std"] = std_existing(out, [f"{p}obts_pressure_direction" for p in ORDERBOOK_PREFIXES])
    pressure_columns = [f"{p}obts_pressure_direction" for p in ORDERBOOK_PREFIXES if f"{p}obts_pressure_direction" in out]
    if pressure_columns:
        pressure_frame = pd.concat([num(out, column) for column in pressure_columns], axis=1)
        out["conf_ob_bullish_venue_count"] = pressure_frame.gt(0.20).sum(axis=1)
        out["conf_ob_bearish_venue_count"] = pressure_frame.lt(-0.20).sum(axis=1)
        out["conf_ob_pressure_agreement_ratio"] = (
            pd.concat([out["conf_ob_bullish_venue_count"], out["conf_ob_bearish_venue_count"]], axis=1).max(axis=1)
            / out["conf_ob_venue_count_present"].replace(0.0, np.nan)
        ).fillna(0.0)
        out["conf_ob_pressure_disagreement"] = (
            out["conf_ob_bullish_venue_count"].gt(0) & out["conf_ob_bearish_venue_count"].gt(0)
        ).astype(float)
        out["conf_ob_single_venue_extreme"] = (
            pressure_frame.abs().gt(0.55).sum(axis=1).eq(1)
            & pressure_frame.abs().gt(0.20).sum(axis=1).eq(1)
        ).astype(float)
    else:
        out["conf_ob_bullish_venue_count"] = 0.0
        out["conf_ob_bearish_venue_count"] = 0.0
        out["conf_ob_pressure_agreement_ratio"] = 0.0
        out["conf_ob_pressure_disagreement"] = 0.0
        out["conf_ob_single_venue_extreme"] = 0.0
    out["conf_ob_bullish_pressure_agreement"] = clamp01(out["conf_ob_pressure_direction_mean"].clip(lower=0.0))
    out["conf_ob_bearish_pressure_agreement"] = clamp01((-out["conf_ob_pressure_direction_mean"]).clip(lower=0.0))
    out["conf_ob_pressure_divergence"] = mean_existing(out, [f"{p}obts_pressure_divergence" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_breakout_failure"] = mean_existing(out, [f"{p}obts_breakout_failure_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_breakdown_failure"] = mean_existing(out, [f"{p}obts_breakdown_failure_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_ask_absorption"] = mean_existing(out, [f"{p}obts_ask_absorption_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_bid_absorption"] = mean_existing(out, [f"{p}obts_bid_absorption_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_resistance_cleared"] = mean_existing(out, [f"{p}obts_resistance_cleared_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_support_cleared"] = mean_existing(out, [f"{p}obts_support_cleared_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_resistance_rejection"] = mean_existing(out, [f"{p}obts_resistance_rejection_resolved_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_support_bounce"] = mean_existing(out, [f"{p}obts_support_bounce_resolved_score" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_resistance_distance_min_bps"] = min_abs_existing(out, [f"{p}obts_resistance_zone_distance_bps" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_support_distance_min_bps"] = min_abs_existing(out, [f"{p}obts_support_zone_distance_bps" for p in ORDERBOOK_PREFIXES])
    resistance_persistence = mean_existing(out, [f"{p}obts_resistance_zone_persistence_24h" for p in ORDERBOOK_PREFIXES])
    support_persistence = mean_existing(out, [f"{p}obts_support_zone_persistence_24h" for p in ORDERBOOK_PREFIXES])
    out["conf_ob_resistance_persistence_24h"] = clamp01(resistance_persistence.fillna(0.0))
    out["conf_ob_support_persistence_24h"] = clamp01(support_persistence.fillna(0.0))
    out["conf_ob_resistance_persistence_weak"] = clamp01(1.0 - out["conf_ob_resistance_persistence_24h"])
    out["conf_ob_support_persistence_weak"] = clamp01(1.0 - out["conf_ob_support_persistence_24h"])
    out["conf_ob_spread_fragility"] = clamp01(
        mean_existing(out, [f"{p}obts_spread_bps_p95" for p in ORDERBOOK_PREFIXES]) / 8.0
    )
    out["conf_ob_wall_distance_stretched"] = clamp01(
        mean_existing(
            out,
            [
                "conf_ob_resistance_distance_min_bps",
                "conf_ob_support_distance_min_bps",
            ],
        ).fillna(0.0)
        / 300.0
    )
    out["conf_ob_multi_venue_agreement"] = (
        out["conf_ob_venue_count_present"].ge(2).astype(float)
        * (1.0 - clamp01(out["conf_ob_pressure_direction_std"].fillna(0.0)))
        * clamp01(out["conf_ob_pressure_direction_mean"].abs())
    )
    out["conf_ob_balanced_book"] = (
        out["conf_ob_venue_count_present"].gt(0).astype(float)
        * out["conf_ob_pressure_direction_mean"].abs().le(0.20).astype(float)
        * out["conf_ob_pressure_divergence"].fillna(0.0).le(0.30).astype(float)
    )
    return out


def append_context_state(out: DataFrame) -> DataFrame:
    out["conf_context_activity_24h"] = clamp01(num(out, "ctx_weighted_context_total_intensity_24h") / 10.0)
    out["conf_context_activity_acceleration"] = clamp01(num(out, "ctx_weighted_context_total_intensity_6h").fillna(0.0) / 5.0)
    out["conf_context_topic_confluence_24h"] = max_existing(out, [col for col in out.columns if col.startswith("ctx_") and col.endswith("_confluence_24h")])
    out["conf_context_topic_persistence_24h"] = clamp01(max_existing(out, [col for col in out.columns if col.startswith("ctx_") and col.endswith("_persistence_hours_24h")]) / 24.0)
    out["conf_context_topic_severity_24h"] = max_existing(out, [col for col in out.columns if col.startswith("ctx_") and col.endswith("_severity_max_24h")])
    risk_columns = [
        "ctx_war_geopolitical_intensity_24h",
        "ctx_banking_liquidity_intensity_24h",
        "ctx_china_property_credit_intensity_24h",
        "ctx_inflation_rates_intensity_24h",
        "ctx_official_macro_release_intensity_24h",
        "ctx_oil_energy_shock_intensity_24h",
        "ctx_security_exploit_intensity_24h",
        "ctx_stablecoin_liquidity_intensity_24h",
        "ctx_dollar_risk_off_intensity_24h",
        "ctx_ai_bubble_risk_intensity_24h",
        "ctx_pandemic_health_shock_intensity_24h",
    ]
    out["conf_context_risk_event_pressure"] = clamp01(mean_existing(out, risk_columns) / 3.0)
    event_components = [
        out["conf_context_activity_24h"],
        out["conf_context_topic_confluence_24h"],
        out["conf_context_topic_persistence_24h"],
        out["conf_context_topic_severity_24h"],
        out["conf_context_risk_event_pressure"],
    ]
    out["conf_context_event_regime_score"] = clamp01(pd.concat(event_components, axis=1).mean(axis=1, skipna=True))
    article_z = num(out, "ctx_article_count_z_7d")
    weighted = num(out, "ctx_weighted_context_total_intensity_24h")
    present = num(out, "context_present").fillna(0.0).gt(0.0)
    out["conf_context_quiet_regime"] = (present & article_z.fillna(0.0).le(0.25) & weighted.fillna(0.0).le(weighted.rolling(720, min_periods=72).quantile(0.50).fillna(weighted.median()))).astype(float)
    out["conf_context_missing_or_stale"] = (1.0 - num(out, "context_present").fillna(0.0)).clip(lower=0.0, upper=1.0)
    return out


def append_hypothesis_state(out: DataFrame) -> DataFrame:
    resistance = num(out, "conf_structure_resistance_stack_score").fillna(0.0)
    support = num(out, "conf_structure_support_stack_score").fillna(0.0)
    breakout = num(out, "conf_structure_breakout_trigger_score").fillna(0.0)
    breakdown = num(out, "conf_structure_breakdown_trigger_score").fillna(0.0)
    vol_bull = num(out, "conf_volume_bullish_confirmation").fillna(0.0)
    vol_bear = num(out, "conf_volume_bearish_confirmation").fillna(0.0)
    event = num(out, "conf_context_event_regime_score").fillna(0.0)
    quiet = num(out, "conf_context_quiet_regime").fillna(0.0)
    ob_bull = mean_scores(out, ["conf_ob_bullish_pressure_agreement", "conf_ob_resistance_removed_strength", "conf_ob_upside_vacuum_after_resistance_removed", "conf_ob_resistance_cleared"])
    ob_bear = mean_scores(out, ["conf_ob_bearish_pressure_agreement", "conf_ob_support_removed_strength", "conf_ob_downside_vacuum_after_support_removed", "conf_ob_support_cleared"])
    ob_balanced = num(out, "conf_ob_balanced_book").fillna(0.0)
    range_high = num(out, "conf_near_range_high").fillna(0.0)
    range_low = num(out, "conf_near_range_low").fillna(0.0)
    context_rise = mean_scores(out, ["conf_context_activity_acceleration", "conf_context_event_regime_score", "conf_context_topic_persistence_24h"])
    support_removed = mean_scores(out, ["conf_ob_support_removed_strength", "conf_ob_support_cleared", "conf_ob_downside_vacuum_after_support_removed"])
    resistance_removed = mean_scores(out, ["conf_ob_resistance_removed_strength", "conf_ob_resistance_cleared", "conf_ob_upside_vacuum_after_resistance_removed"])
    bearish_pressure = mean_scores(out, ["conf_ob_bearish_pressure_agreement", "conf_ob_pressure_agreement_ratio"])
    bullish_pressure = mean_scores(out, ["conf_ob_bullish_pressure_agreement", "conf_ob_pressure_agreement_ratio"])

    add_named_hypothesis(
        out,
        "bearish_event_breakdown_confluence",
        {"resistance_stack": resistance, "context_rise": context_rise, "support_removed_or_thinning": support_removed},
        {"lower_low_break": breakdown, "bearish_volume": vol_bear, "book_bearish_pressure": bearish_pressure},
        {"resistance_stack": resistance, "context_event_pressure": event, "support_removed": support_removed, "downside_vacuum": num(out, "conf_ob_downside_vacuum_after_support_removed"), "breakdown": breakdown, "bearish_volume": vol_bear},
        min_setup_components=2,
        min_trigger_components=3,
    )
    add_named_hypothesis(
        out,
        "bullish_event_breakout_confluence",
        {"resistance_stack": resistance, "context_rise": context_rise, "resistance_removed_or_thinning": resistance_removed},
        {"higher_high_break": breakout, "bullish_volume": vol_bull, "book_bullish_pressure": bullish_pressure},
        {"resistance_stack": resistance, "context_event_pressure": event, "resistance_removed": resistance_removed, "upside_vacuum": num(out, "conf_ob_upside_vacuum_after_resistance_removed"), "breakout": breakout, "bullish_volume": vol_bull},
        min_setup_components=2,
        min_trigger_components=3,
    )
    add_named_hypothesis(
        out,
        "quiet_technical_breakout_confluence",
        {"quiet_context": quiet, "balanced_book": ob_balanced, "resistance_stack": resistance},
        {"breakout": breakout, "bullish_volume": vol_bull, "book_not_bearish": 1.0 - bearish_pressure.clip(upper=1.0)},
        {"quiet_context": quiet, "balanced_book": ob_balanced, "resistance_stack": resistance, "breakout": breakout, "bullish_volume": vol_bull},
        min_setup_components=3,
        min_trigger_components=3,
    )
    add_named_hypothesis(
        out,
        "quiet_technical_breakdown_confluence",
        {"quiet_context": quiet, "balanced_book": ob_balanced, "support_stack": support},
        {"breakdown": breakdown, "bearish_volume": vol_bear, "book_not_bullish": 1.0 - bullish_pressure.clip(upper=1.0)},
        {"quiet_context": quiet, "balanced_book": ob_balanced, "support_stack": support, "breakdown": breakdown, "bearish_volume": vol_bear},
        min_setup_components=3,
        min_trigger_components=3,
    )
    add_named_hypothesis(out, "multi_tf_resistance_rejection", {"resistance_stack": resistance, "range_high": range_high}, {"rejection_absorption": mean_scores(out, ["conf_ob_resistance_rejection", "conf_ob_ask_absorption", "conf_ob_breakout_failure"]), "bearish_volume": vol_bear}, {"resistance_stack": resistance, "range_high": range_high, "rejection": num(out, "conf_ob_resistance_rejection"), "ask_absorption": num(out, "conf_ob_ask_absorption"), "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "multi_tf_support_bounce", {"support_stack": support, "range_low": range_low}, {"support_bounce_absorption": mean_scores(out, ["conf_ob_support_bounce", "conf_ob_bid_absorption", "conf_ob_breakdown_failure"]), "bullish_volume": vol_bull}, {"support_stack": support, "range_low": range_low, "support_bounce": num(out, "conf_ob_support_bounce"), "bid_absorption": num(out, "conf_ob_bid_absorption"), "bullish_volume": vol_bull}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "failed_breakout_exhaustion", {"resistance_stack": resistance, "range_high": range_high}, {"breakout_failure_or_divergence": mean_scores(out, ["conf_ob_breakout_failure", "conf_ob_pressure_divergence"]), "volume_not_bullish": 1.0 - vol_bull.clip(upper=1.0)}, {"resistance_stack": resistance, "range_high": range_high, "breakout_failure": num(out, "conf_ob_breakout_failure"), "pressure_divergence": num(out, "conf_ob_pressure_divergence"), "volume_not_bullish": 1.0 - vol_bull.clip(upper=1.0)}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "failed_breakdown_exhaustion", {"support_stack": support, "range_low": range_low}, {"breakdown_failure_or_divergence": mean_scores(out, ["conf_ob_breakdown_failure", "conf_ob_pressure_divergence"]), "volume_not_bearish": 1.0 - vol_bear.clip(upper=1.0)}, {"support_stack": support, "range_low": range_low, "breakdown_failure": num(out, "conf_ob_breakdown_failure"), "pressure_divergence": num(out, "conf_ob_pressure_divergence"), "volume_not_bearish": 1.0 - vol_bear.clip(upper=1.0)}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "lvn_fast_travel_up", {"lvn_thin_above": num(out, "conf_lvn_up_thinness"), "upside_vacuum": num(out, "conf_ob_upside_vacuum")}, {"breakout": breakout, "bullish_volume": vol_bull}, {"lvn_thin_above": num(out, "conf_lvn_up_thinness"), "upside_vacuum": num(out, "conf_ob_upside_vacuum"), "book_bullish": ob_bull, "breakout": breakout, "bullish_volume": vol_bull}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "lvn_fast_travel_down", {"lvn_thin_below": num(out, "conf_lvn_down_thinness"), "downside_vacuum": num(out, "conf_ob_downside_vacuum")}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"lvn_thin_below": num(out, "conf_lvn_down_thinness"), "downside_vacuum": num(out, "conf_ob_downside_vacuum"), "book_bearish": ob_bear, "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "orderbook_resistance_evaporation_breakout", {"resistance_stack": resistance, "resistance_removed": resistance_removed}, {"breakout": breakout, "bullish_volume": vol_bull}, {"resistance_stack": resistance, "resistance_removed": resistance_removed, "upside_vacuum": num(out, "conf_ob_upside_vacuum_after_resistance_removed"), "breakout": breakout, "bullish_volume": vol_bull}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "orderbook_support_removal_breakdown", {"support_stack": support, "support_removed": support_removed}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"support_stack": support, "support_removed": support_removed, "downside_vacuum": num(out, "conf_ob_downside_vacuum_after_support_removed"), "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "context_pressure_plus_structure_break", {"context_event_pressure": event, "near_key_level": (resistance + support).clip(upper=1.0)}, {"structure_break": (breakout * vol_bull + breakdown * vol_bear).clip(upper=1.0)}, {"context_event_pressure": event, "resistance_stack": resistance, "support_stack": support, "breakout": breakout, "breakdown": breakdown, "bullish_volume": vol_bull, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=1)
    add_named_hypothesis(out, "macro_context_liquidity_stress_breakdown", {"macro_risk_pressure": num(out, "conf_context_risk_event_pressure"), "support_stack": support}, {"breakdown": breakdown, "book_bearish": ob_bear, "bearish_volume": vol_bear}, {"macro_risk_pressure": num(out, "conf_context_risk_event_pressure"), "support_stack": support, "breakdown": breakdown, "book_bearish": ob_bear, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "multi_venue_orderbook_confluence", {"multi_venue_present": num(out, "conf_ob_venue_count_present").ge(2).astype(float)}, {"venue_pressure_agreement": num(out, "conf_ob_multi_venue_agreement")}, {"venue_pressure_agreement": num(out, "conf_ob_multi_venue_agreement"), "pressure_agreement_ratio": num(out, "conf_ob_pressure_agreement_ratio"), "single_venue_extreme_absence": 1.0 - num(out, "conf_ob_single_venue_extreme").clip(upper=1.0), "breakout": breakout, "breakdown": breakdown})
    append_hypothesis_pack_beta(out, resistance, support, breakout, breakdown, vol_bull, vol_bear, event, quiet, ob_bull, ob_bear, ob_balanced, range_high, range_low, context_rise, support_removed, resistance_removed, bearish_pressure, bullish_pressure)
    append_hypothesis_pack_gamma_bull(out, resistance, support, breakout, vol_bull, event, quiet, ob_bull, ob_balanced, range_high, range_low, context_rise, resistance_removed, bullish_pressure)
    append_hypothesis_pack_delta_discovery(out, resistance, support, breakout, breakdown, vol_bull, vol_bear, ob_bull, ob_bear, bearish_pressure, bullish_pressure)
    append_hypothesis_pack_epsilon_downside_after_break(out, support, breakdown, vol_bull, vol_bear, ob_bear, bearish_pressure, support_removed)
    return out


def append_hypothesis_pack_epsilon_downside_after_break(
    out: DataFrame,
    support: Series,
    breakdown: Series,
    vol_bull: Series,
    vol_bear: Series,
    ob_bear: Series,
    bearish_pressure: Series,
    support_removed: Series,
) -> None:
    initial_break = mean_scores(
        out,
        [
            "px_close_breakdown_6h",
            "px_close_breakdown_24h",
            "conf_structure_breakdown_trigger_score",
        ],
    )
    fast_drop = clamp01((num(out, "px_return_1h").fillna(0.0).mul(-100.0) - 0.25) / 1.25)
    three_hour_drop = clamp01((num(out, "px_return_3h").fillna(0.0).mul(-100.0) - 0.60) / 2.40)
    early_break_visible = clamp01(
        pd.concat([initial_break, fast_drop, three_hour_drop, breakdown], axis=1).mean(axis=1, skipna=True).fillna(0.0)
    )
    volume_staying_bearish = clamp01(
        pd.concat(
            [
                vol_bear,
                clamp01((num(out, "px_volume_z_6h").fillna(0.0) - 0.50) / 2.00),
                clamp01(num(out, "px_volume_pressure_6h").fillna(0.0).mul(-1.0)),
            ],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )
    support_weak_or_removed = clamp01(
        pd.concat(
            [
                support_removed,
                num(out, "conf_ob_support_persistence_weak"),
                num(out, "conf_ob_downside_vacuum_after_support_removed"),
                num(out, "conf_ob_support_cleared"),
            ],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )
    orderbook_panic = clamp01(
        pd.concat(
            [
                ob_bear,
                bearish_pressure,
                support_weak_or_removed,
                num(out, "conf_ob_downside_vacuum"),
                num(out, "conf_ob_spread_fragility"),
            ],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )
    support_rebuild = clamp01(
        pd.concat(
            [
                num(out, "conf_ob_support_added_strength"),
                num(out, "conf_ob_support_bounce"),
                num(out, "conf_ob_bid_absorption"),
            ],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )
    pressure_fading = clamp01(
        pd.concat(
            [
                1.0 - vol_bear.clip(upper=1.0),
                1.0 - bearish_pressure.clip(upper=1.0),
                num(out, "conf_ob_pressure_divergence"),
            ],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )
    reclaim_pressure = clamp01(
        pd.concat(
            [support_rebuild, vol_bull, num(out, "conf_ob_breakdown_failure")],
            axis=1,
        ).mean(axis=1, skipna=True).fillna(0.0)
    )

    add_named_hypothesis(
        out,
        "epsilon_downside_first_break_continuation",
        {"early_downside_break_visible": early_break_visible, "support_near_or_weak": clamp01(pd.concat([support, support_weak_or_removed], axis=1).mean(axis=1, skipna=True).fillna(0.0))},
        {"volume_stays_bearish": volume_staying_bearish, "orderbook_panic": orderbook_panic},
        {"early_break_visible": early_break_visible, "volume_stays_bearish": volume_staying_bearish, "support_removed": support_weak_or_removed, "orderbook_panic": orderbook_panic},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "epsilon_downside_first_break_exhaustion",
        {"early_downside_break_visible": early_break_visible, "support_or_range_low": support},
        {"pressure_fading": pressure_fading, "support_rebuild": support_rebuild},
        {"early_break_visible": early_break_visible, "pressure_fading": pressure_fading, "support_rebuild": support_rebuild, "reclaim_pressure": reclaim_pressure},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "epsilon_support_reclaim_after_break",
        {"early_downside_break_visible": early_break_visible, "support_stack": support},
        {"reclaim_pressure": reclaim_pressure, "volume_not_bearish": 1.0 - vol_bear.clip(upper=1.0)},
        {"early_break_visible": early_break_visible, "support_rebuild": support_rebuild, "breakdown_failure": num(out, "conf_ob_breakdown_failure"), "bullish_volume": vol_bull},
        min_setup_components=1,
        min_trigger_components=1,
    )
    add_named_hypothesis(
        out,
        "epsilon_orderbook_panic_after_break",
        {"early_downside_break_visible": early_break_visible, "support_removed": support_weak_or_removed},
        {"bearish_book_pressure": bearish_pressure, "downside_vacuum": num(out, "conf_ob_downside_vacuum")},
        {"early_break_visible": early_break_visible, "support_removed": support_weak_or_removed, "bearish_book_pressure": bearish_pressure, "downside_vacuum": num(out, "conf_ob_downside_vacuum"), "volume_stays_bearish": volume_staying_bearish},
        min_setup_components=1,
        min_trigger_components=2,
    )


def append_hypothesis_pack_beta(
    out: DataFrame,
    resistance: Series,
    support: Series,
    breakout: Series,
    breakdown: Series,
    vol_bull: Series,
    vol_bear: Series,
    event: Series,
    quiet: Series,
    ob_bull: Series,
    ob_bear: Series,
    ob_balanced: Series,
    range_high: Series,
    range_low: Series,
    context_rise: Series,
    support_removed: Series,
    resistance_removed: Series,
    bearish_pressure: Series,
    bullish_pressure: Series,
) -> None:
    near_key_level = (resistance + support).clip(upper=1.0)
    lvn_up = num(out, "conf_lvn_up_thinness").fillna(0.0)
    lvn_down = num(out, "conf_lvn_down_thinness").fillna(0.0)
    inside_range = num(out, "conf_inside_range_mid").fillna(0.0)
    ob_disagreement = num(out, "conf_ob_pressure_disagreement").fillna(0.0)
    ob_divergence = num(out, "conf_ob_pressure_divergence").fillna(0.0)
    ob_multi = num(out, "conf_ob_multi_venue_agreement").fillna(0.0)
    ob_ask_absorb = num(out, "conf_ob_ask_absorption").fillna(0.0)
    ob_bid_absorb = num(out, "conf_ob_bid_absorption").fillna(0.0)
    ob_breakout_fail = num(out, "conf_ob_breakout_failure").fillna(0.0)
    ob_breakdown_fail = num(out, "conf_ob_breakdown_failure").fillna(0.0)
    upside_vacuum = num(out, "conf_ob_upside_vacuum").fillna(0.0)
    downside_vacuum = num(out, "conf_ob_downside_vacuum").fillna(0.0)
    upside_after_removed = num(out, "conf_ob_upside_vacuum_after_resistance_removed").fillna(0.0)
    downside_after_removed = num(out, "conf_ob_downside_vacuum_after_support_removed").fillna(0.0)
    macro_risk = num(out, "conf_context_risk_event_pressure").fillna(0.0)
    topic_severity = num(out, "conf_context_topic_severity_24h").fillna(0.0)
    topic_confluence = num(out, "conf_context_topic_confluence_24h").fillna(0.0)
    topic_persistence = num(out, "conf_context_topic_persistence_24h").fillna(0.0)
    activity_accel = num(out, "conf_context_activity_acceleration").fillna(0.0)

    add_named_hypothesis(out, "beta_structure_resistance_breakout_continuation", {"resistance_stack": resistance, "range_high": range_high}, {"breakout": breakout, "bullish_volume": vol_bull}, {"resistance_stack": resistance, "range_high": range_high, "breakout": breakout, "bullish_volume": vol_bull}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_support_breakdown_continuation", {"support_stack": support, "range_low": range_low}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"support_stack": support, "range_low": range_low, "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_resistance_rejection_failure", {"resistance_stack": resistance, "range_high": range_high}, {"bearish_volume": vol_bear, "breakout_absent": 1.0 - breakout.clip(upper=1.0)}, {"resistance_stack": resistance, "range_high": range_high, "bearish_volume": vol_bear, "breakout_absent": 1.0 - breakout.clip(upper=1.0)}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_support_bounce_failure", {"support_stack": support, "range_low": range_low}, {"bullish_volume": vol_bull, "breakdown_absent": 1.0 - breakdown.clip(upper=1.0)}, {"support_stack": support, "range_low": range_low, "bullish_volume": vol_bull, "breakdown_absent": 1.0 - breakdown.clip(upper=1.0)}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_lvn_up_continuation", {"lvn_thin_above": lvn_up, "inside_range": inside_range}, {"breakout": breakout, "bullish_volume": vol_bull}, {"lvn_thin_above": lvn_up, "inside_range": inside_range, "breakout": breakout, "bullish_volume": vol_bull}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_lvn_down_continuation", {"lvn_thin_below": lvn_down, "inside_range": inside_range}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"lvn_thin_below": lvn_down, "inside_range": inside_range, "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_range_high_fakeout", {"resistance_stack": resistance, "range_high": range_high}, {"breakout": breakout, "volume_not_confirming": 1.0 - vol_bull.clip(upper=1.0)}, {"resistance_stack": resistance, "range_high": range_high, "breakout": breakout, "volume_not_confirming": 1.0 - vol_bull.clip(upper=1.0)}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_structure_range_low_fakeout", {"support_stack": support, "range_low": range_low}, {"breakdown": breakdown, "volume_not_confirming": 1.0 - vol_bear.clip(upper=1.0)}, {"support_stack": support, "range_low": range_low, "breakdown": breakdown, "volume_not_confirming": 1.0 - vol_bear.clip(upper=1.0)}, min_setup_components=1, min_trigger_components=2)

    add_named_hypothesis(out, "beta_ob_resistance_removed_breakout", {"resistance_stack": resistance, "resistance_removed": resistance_removed}, {"breakout": breakout, "bullish_pressure": bullish_pressure}, {"resistance_removed": resistance_removed, "upside_vacuum": upside_after_removed, "breakout": breakout, "bullish_pressure": bullish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_ob_support_removed_breakdown", {"support_stack": support, "support_removed": support_removed}, {"breakdown": breakdown, "bearish_pressure": bearish_pressure}, {"support_removed": support_removed, "downside_vacuum": downside_after_removed, "breakdown": breakdown, "bearish_pressure": bearish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_ob_ask_absorption_rejection", {"resistance_stack": resistance, "ask_absorption": ob_ask_absorb}, {"bearish_volume": vol_bear, "breakout_failure": ob_breakout_fail}, {"resistance_stack": resistance, "ask_absorption": ob_ask_absorb, "breakout_failure": ob_breakout_fail, "bearish_volume": vol_bear}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "beta_ob_bid_absorption_bounce", {"support_stack": support, "bid_absorption": ob_bid_absorb}, {"bullish_volume": vol_bull, "breakdown_failure": ob_breakdown_fail}, {"support_stack": support, "bid_absorption": ob_bid_absorb, "breakdown_failure": ob_breakdown_fail, "bullish_volume": vol_bull}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "beta_ob_upside_vacuum_breakout", {"upside_vacuum": upside_vacuum, "resistance_stack": resistance}, {"breakout": breakout, "bullish_volume": vol_bull}, {"upside_vacuum": upside_vacuum, "breakout": breakout, "bullish_volume": vol_bull, "bullish_pressure": bullish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_ob_downside_vacuum_breakdown", {"downside_vacuum": downside_vacuum, "support_stack": support}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"downside_vacuum": downside_vacuum, "breakdown": breakdown, "bearish_volume": vol_bear, "bearish_pressure": bearish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_ob_pressure_disagreement_fakeout", {"near_key_level": near_key_level, "pressure_disagreement": ob_disagreement}, {"breakout_or_breakdown": (breakout + breakdown).clip(upper=1.0), "volume_confirming": (vol_bull + vol_bear).clip(upper=1.0)}, {"near_key_level": near_key_level, "pressure_disagreement": ob_disagreement, "pressure_divergence": ob_divergence, "breakout": breakout, "breakdown": breakdown}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "beta_ob_multi_venue_pressure_continuation", {"multi_venue_pressure": ob_multi}, {"structure_break": (breakout + breakdown).clip(upper=1.0), "volume_confirming": (vol_bull + vol_bear).clip(upper=1.0)}, {"multi_venue_pressure": ob_multi, "breakout": breakout, "breakdown": breakdown, "bullish_pressure": bullish_pressure, "bearish_pressure": bearish_pressure}, min_setup_components=1, min_trigger_components=1)

    add_named_hypothesis(out, "beta_context_risk_support_breakdown", {"macro_risk": macro_risk, "support_stack": support}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"macro_risk": macro_risk, "support_stack": support, "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "beta_context_risk_resistance_rejection", {"macro_risk": macro_risk, "resistance_stack": resistance}, {"bearish_volume": vol_bear, "breakout_failure": ob_breakout_fail}, {"macro_risk": macro_risk, "resistance_stack": resistance, "breakout_failure": ob_breakout_fail, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=1)
    add_named_hypothesis(out, "beta_context_attention_breakout", {"context_rise": context_rise, "resistance_stack": resistance}, {"breakout": breakout, "bullish_volume": vol_bull}, {"context_rise": context_rise, "topic_severity": topic_severity, "resistance_stack": resistance, "breakout": breakout, "bullish_volume": vol_bull}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "beta_context_attention_breakdown", {"context_rise": context_rise, "support_stack": support}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"context_rise": context_rise, "topic_severity": topic_severity, "support_stack": support, "breakdown": breakdown, "bearish_volume": vol_bear}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "beta_context_persistent_topic_trend", {"topic_persistence": topic_persistence, "topic_confluence": topic_confluence}, {"structure_break": (breakout + breakdown).clip(upper=1.0)}, {"topic_persistence": topic_persistence, "topic_confluence": topic_confluence, "topic_severity": topic_severity, "breakout": breakout, "breakdown": breakdown}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "beta_quiet_context_structure_breakout", {"quiet_context": quiet, "resistance_stack": resistance}, {"breakout": breakout, "bullish_volume": vol_bull}, {"quiet_context": quiet, "resistance_stack": resistance, "breakout": breakout, "bullish_volume": vol_bull, "balanced_book": ob_balanced}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_quiet_context_structure_breakdown", {"quiet_context": quiet, "support_stack": support}, {"breakdown": breakdown, "bearish_volume": vol_bear}, {"quiet_context": quiet, "support_stack": support, "breakdown": breakdown, "bearish_volume": vol_bear, "balanced_book": ob_balanced}, min_setup_components=1, min_trigger_components=2)

    add_named_hypothesis(out, "beta_macro_ob_support_evaporation_crash", {"macro_risk": macro_risk, "support_removed": support_removed}, {"breakdown": breakdown, "bearish_pressure": bearish_pressure}, {"macro_risk": macro_risk, "support_removed": support_removed, "downside_vacuum": downside_after_removed, "breakdown": breakdown, "bearish_pressure": bearish_pressure}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "beta_macro_ob_resistance_evaporation_squeeze", {"topic_severity": topic_severity, "resistance_removed": resistance_removed}, {"breakout": breakout, "bullish_pressure": bullish_pressure}, {"topic_severity": topic_severity, "resistance_removed": resistance_removed, "upside_vacuum": upside_after_removed, "breakout": breakout, "bullish_pressure": bullish_pressure}, min_setup_components=2, min_trigger_components=2)
    add_named_hypothesis(out, "beta_macro_ob_pressure_flip_down", {"macro_risk": macro_risk, "near_key_level": near_key_level}, {"bearish_pressure": bearish_pressure, "support_removed": support_removed}, {"macro_risk": macro_risk, "near_key_level": near_key_level, "bearish_pressure": bearish_pressure, "support_removed": support_removed, "breakdown": breakdown}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_macro_ob_pressure_flip_up", {"topic_severity": topic_severity, "near_key_level": near_key_level}, {"bullish_pressure": bullish_pressure, "resistance_removed": resistance_removed}, {"topic_severity": topic_severity, "near_key_level": near_key_level, "bullish_pressure": bullish_pressure, "resistance_removed": resistance_removed, "breakout": breakout}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_event_volume_expansion_down", {"event_pressure": event, "activity_accel": activity_accel}, {"bearish_volume": vol_bear, "breakdown": breakdown}, {"event_pressure": event, "activity_accel": activity_accel, "bearish_volume": vol_bear, "breakdown": breakdown, "downside_vacuum": downside_vacuum}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_event_volume_expansion_up", {"event_pressure": event, "activity_accel": activity_accel}, {"bullish_volume": vol_bull, "breakout": breakout}, {"event_pressure": event, "activity_accel": activity_accel, "bullish_volume": vol_bull, "breakout": breakout, "upside_vacuum": upside_vacuum}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "beta_failed_breakdown_reclaim_context", {"macro_risk": macro_risk, "support_stack": support}, {"breakdown_failure": ob_breakdown_fail, "bullish_volume": vol_bull}, {"macro_risk": macro_risk, "support_stack": support, "breakdown_failure": ob_breakdown_fail, "bullish_volume": vol_bull, "bid_absorption": ob_bid_absorb}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "beta_failed_breakout_reject_context", {"topic_severity": topic_severity, "resistance_stack": resistance}, {"breakout_failure": ob_breakout_fail, "bearish_volume": vol_bear}, {"topic_severity": topic_severity, "resistance_stack": resistance, "breakout_failure": ob_breakout_fail, "bearish_volume": vol_bear, "ask_absorption": ob_ask_absorb}, min_setup_components=1, min_trigger_components=1)


def append_hypothesis_pack_gamma_bull(
    out: DataFrame,
    resistance: Series,
    support: Series,
    breakout: Series,
    vol_bull: Series,
    event: Series,
    quiet: Series,
    ob_bull: Series,
    ob_balanced: Series,
    range_high: Series,
    range_low: Series,
    context_rise: Series,
    resistance_removed: Series,
    bullish_pressure: Series,
) -> None:
    bull_regime = num(out, "conf_price_bull_trend_regime").fillna(0.0)
    bull_structure = num(out, "conf_structure_bullish_state_score").fillna(0.0)
    compression = num(out, "conf_price_compression_24h").fillna(0.0)
    vol_impulse = mean_scores(out, ["conf_volume_bullish_confirmation", "conf_volume_bullish_impulse_short"])
    lvn_up = num(out, "conf_lvn_up_thinness").fillna(0.0)
    upside_vacuum = num(out, "conf_ob_upside_vacuum").fillna(0.0)
    upside_after_removed = num(out, "conf_ob_upside_vacuum_after_resistance_removed").fillna(0.0)
    resistance_cleared = num(out, "conf_ob_resistance_cleared").fillna(0.0)
    support_bounce = num(out, "conf_ob_support_bounce").fillna(0.0)
    bid_absorption = num(out, "conf_ob_bid_absorption").fillna(0.0)
    topic_severity = num(out, "conf_context_topic_severity_24h").fillna(0.0)
    topic_persistence = num(out, "conf_context_topic_persistence_24h").fillna(0.0)
    inside_range = num(out, "conf_inside_range_mid").fillna(0.0)
    mtf_bull = clamp01(pd.concat([bull_regime, bull_structure], axis=1).mean(axis=1, skipna=True).fillna(0.0))

    add_named_hypothesis(out, "gamma_bull_structure_breakout_regime", {"bull_regime": bull_regime, "resistance_stack": resistance}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"bull_regime": bull_regime, "resistance_stack": resistance, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_mtf_structure_alignment", {"bull_structure": bull_structure, "bull_regime": bull_regime}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"bull_structure": bull_structure, "bull_regime": bull_regime, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_vah_acceptance", {"bull_regime": bull_regime, "range_high": range_high}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"bull_regime": bull_regime, "range_high": range_high, "breakout": breakout, "bullish_volume_impulse": vol_impulse, "lvn_thin_above": lvn_up}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_lvn_fast_travel", {"lvn_thin_above": lvn_up, "bull_regime": bull_regime}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"lvn_thin_above": lvn_up, "bull_regime": bull_regime, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_compression_breakout", {"compression": compression, "inside_range": inside_range}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"compression": compression, "inside_range": inside_range, "breakout": breakout, "bullish_volume_impulse": vol_impulse, "bull_regime": bull_regime}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_pullback_higher_low_resume", {"bull_regime": bull_regime, "support_stack": support}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"bull_regime": bull_regime, "support_stack": support, "range_low": range_low, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_tlv2_resistance_break", {"resistance_stack": resistance, "bull_structure": bull_structure}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"resistance_stack": resistance, "bull_structure": bull_structure, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_range_expansion_volume", {"range_high": range_high, "compression": compression}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"range_high": range_high, "compression": compression, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_structure_only_no_context", {"mtf_bull": mtf_bull, "resistance_stack": resistance}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"mtf_bull": mtf_bull, "resistance_stack": resistance, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_quiet_technical_breakout", {"quiet_context": quiet, "ob_balanced": ob_balanced, "bull_structure": bull_structure}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"quiet_context": quiet, "ob_balanced": ob_balanced, "bull_structure": bull_structure, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_ob_resistance_removed", {"bull_regime": bull_regime, "resistance_removed": resistance_removed}, {"breakout": breakout, "bullish_pressure": bullish_pressure}, {"bull_regime": bull_regime, "resistance_removed": resistance_removed, "resistance_cleared": resistance_cleared, "breakout": breakout, "bullish_pressure": bullish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_ob_upside_vacuum", {"bull_regime": bull_regime, "upside_vacuum": upside_vacuum}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"bull_regime": bull_regime, "upside_vacuum": upside_vacuum, "upside_after_removed": upside_after_removed, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_ob_pressure_confirmation", {"bull_regime": bull_regime, "ob_bullish": ob_bull}, {"breakout": breakout, "bullish_pressure": bullish_pressure}, {"bull_regime": bull_regime, "ob_bullish": ob_bull, "breakout": breakout, "bullish_pressure": bullish_pressure}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_reclaim_after_support_defense", {"bull_regime": bull_regime, "support_stack": support}, {"support_bounce": support_bounce, "bullish_volume_impulse": vol_impulse}, {"bull_regime": bull_regime, "support_stack": support, "support_bounce": support_bounce, "bid_absorption": bid_absorption, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=1)
    add_named_hypothesis(out, "gamma_bull_context_attention_breakout", {"context_rise": context_rise, "bull_structure": bull_structure}, {"breakout": breakout, "bullish_volume_impulse": vol_impulse}, {"context_rise": context_rise, "topic_severity": topic_severity, "topic_persistence": topic_persistence, "breakout": breakout, "bullish_volume_impulse": vol_impulse}, min_setup_components=1, min_trigger_components=2)
    add_named_hypothesis(out, "gamma_bull_event_squeeze_breakout", {"event_pressure": event, "resistance_removed": resistance_removed}, {"breakout": breakout, "bullish_pressure": bullish_pressure}, {"event_pressure": event, "resistance_removed": resistance_removed, "upside_after_removed": upside_after_removed, "breakout": breakout, "bullish_pressure": bullish_pressure}, min_setup_components=1, min_trigger_components=2)


def append_hypothesis_pack_delta_discovery(
    out: DataFrame,
    resistance: Series,
    support: Series,
    breakout: Series,
    breakdown: Series,
    vol_bull: Series,
    vol_bear: Series,
    ob_bull: Series,
    ob_bear: Series,
    bearish_pressure: Series,
    bullish_pressure: Series,
) -> None:
    high_stack = num(out, "conf_range_position_high_stack").fillna(0.0)
    low_stack = num(out, "conf_range_position_low_stack").fillna(0.0)
    high_24 = num(out, "conf_range_position_high_24h").fillna(0.0)
    low_24 = num(out, "conf_range_position_low_24h").fillna(0.0)
    expansion = num(out, "conf_price_range_expansion_24h").fillna(0.0)
    compression = num(out, "conf_price_compression_24h").fillna(0.0)
    vol_bull_short = num(out, "conf_volume_bullish_impulse_short").fillna(0.0)
    vol_bear_short = num(out, "conf_volume_bearish_impulse_short").fillna(0.0)
    lvn_up = num(out, "conf_lvn_up_thinness").fillna(0.0)
    lvn_down = num(out, "conf_lvn_down_thinness").fillna(0.0)
    support_weak = num(out, "conf_ob_support_persistence_weak").fillna(0.0)
    resistance_weak = num(out, "conf_ob_resistance_persistence_weak").fillna(0.0)
    wall_distance = num(out, "conf_ob_wall_distance_stretched").fillna(0.0)
    downside_vacuum = num(out, "conf_ob_downside_vacuum").fillna(0.0)
    upside_vacuum = num(out, "conf_ob_upside_vacuum").fillna(0.0)
    downside_after_removed = num(out, "conf_ob_downside_vacuum_after_support_removed").fillna(0.0)
    upside_after_removed = num(out, "conf_ob_upside_vacuum_after_resistance_removed").fillna(0.0)

    add_named_hypothesis(
        out,
        "delta_range_high_breakout_success",
        {"range_high_24h": high_24, "range_high_stack": high_stack},
        {"breakout": breakout, "bullish_volume_short": vol_bull_short},
        {"range_high_stack": high_stack, "breakout": breakout, "bullish_volume": vol_bull, "bullish_volume_short": vol_bull_short},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "delta_range_low_breakdown_success",
        {"range_low_24h": low_24, "range_low_stack": low_stack},
        {"breakdown": breakdown, "bearish_volume_short": vol_bear_short},
        {"range_low_stack": low_stack, "breakdown": breakdown, "bearish_volume": vol_bear, "bearish_volume_short": vol_bear_short},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "delta_expansion_drawdown_risk",
        {"range_expansion": expansion, "range_high_or_low": (high_stack + low_stack).clip(upper=1.0)},
        {"bearish_volume_short": vol_bear_short, "breakdown_or_weak_support": (breakdown + support_weak).clip(upper=1.0)},
        {"range_expansion": expansion, "bearish_volume_short": vol_bear_short, "support_weak": support_weak, "breakdown": breakdown},
        min_setup_components=1,
        min_trigger_components=1,
    )
    add_named_hypothesis(
        out,
        "delta_compression_breakout_release",
        {"compression": compression, "range_high_stack": high_stack},
        {"breakout": breakout, "bullish_volume_short": vol_bull_short},
        {"compression": compression, "range_high_stack": high_stack, "breakout": breakout, "bullish_volume_short": vol_bull_short, "lvn_up": lvn_up},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "delta_compression_breakdown_release",
        {"compression": compression, "range_low_stack": low_stack},
        {"breakdown": breakdown, "bearish_volume_short": vol_bear_short},
        {"compression": compression, "range_low_stack": low_stack, "breakdown": breakdown, "bearish_volume_short": vol_bear_short, "lvn_down": lvn_down},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "delta_orderbook_support_weak_drawdown",
        {"support_weak": support_weak, "wall_distance_stretched": wall_distance},
        {"bearish_pressure": bearish_pressure, "downside_vacuum": downside_vacuum},
        {"support_weak": support_weak, "wall_distance_stretched": wall_distance, "bearish_pressure": bearish_pressure, "downside_vacuum": downside_vacuum, "downside_after_removed": downside_after_removed},
        min_setup_components=1,
        min_trigger_components=1,
    )
    add_named_hypothesis(
        out,
        "delta_orderbook_resistance_weak_breakout",
        {"resistance_weak": resistance_weak, "wall_distance_stretched": wall_distance},
        {"bullish_pressure": bullish_pressure, "upside_vacuum": upside_vacuum},
        {"resistance_weak": resistance_weak, "wall_distance_stretched": wall_distance, "bullish_pressure": bullish_pressure, "upside_vacuum": upside_vacuum, "upside_after_removed": upside_after_removed},
        min_setup_components=1,
        min_trigger_components=1,
    )
    add_named_hypothesis(
        out,
        "delta_structure_orderbook_drawdown_confluence",
        {"range_low_stack": low_stack, "support_or_book_weak": (support_weak + downside_vacuum).clip(upper=1.0)},
        {"breakdown": breakdown, "bearish_volume_short": vol_bear_short},
        {"range_low_stack": low_stack, "support_stack": support, "support_weak": support_weak, "downside_vacuum": downside_vacuum, "breakdown": breakdown, "bearish_volume_short": vol_bear_short, "book_bearish": ob_bear},
        min_setup_components=1,
        min_trigger_components=2,
    )
    add_named_hypothesis(
        out,
        "delta_structure_orderbook_breakout_confluence",
        {"range_high_stack": high_stack, "resistance_or_book_weak": (resistance_weak + upside_vacuum).clip(upper=1.0)},
        {"breakout": breakout, "bullish_volume_short": vol_bull_short},
        {"range_high_stack": high_stack, "resistance_stack": resistance, "resistance_weak": resistance_weak, "upside_vacuum": upside_vacuum, "breakout": breakout, "bullish_volume_short": vol_bull_short, "book_bullish": ob_bull},
        min_setup_components=1,
        min_trigger_components=2,
    )


def add_named_hypothesis(
    out: DataFrame,
    name: str,
    setup_components: dict[str, Series | float],
    trigger_components: dict[str, Series | float],
    score_components: dict[str, Series | float],
    *,
    setup_threshold: float = 0.35,
    trigger_threshold: float = 0.35,
    component_threshold: float = 0.35,
    min_setup_components: int | None = None,
    min_trigger_components: int | None = None,
) -> None:
    setup_frame = component_frame(out, name, setup_components, "setup")
    trigger_frame = component_frame(out, name, trigger_components, "trigger")
    score_frame = component_frame(out, name, score_components, "score")
    setup_score = setup_frame.mean(axis=1, skipna=True).fillna(0.0)
    trigger_score = trigger_frame.mean(axis=1, skipna=True).fillna(0.0)
    evidence_score = score_frame.mean(axis=1, skipna=True).fillna(0.0)
    setup_required = min_setup_components if min_setup_components is not None else min(2, len(setup_frame.columns))
    trigger_required = min_trigger_components if min_trigger_components is not None else min(2, len(trigger_frame.columns))
    setup_count = setup_frame.ge(float(component_threshold)).sum(axis=1) if not setup_frame.empty else pd.Series(0, index=out.index)
    trigger_count = trigger_frame.ge(float(component_threshold)).sum(axis=1) if not trigger_frame.empty else pd.Series(0, index=out.index)
    out[f"conf_{name}_setup_component_score"] = clamp01(setup_score)
    out[f"conf_{name}_trigger_component_score"] = clamp01(trigger_score)
    out[f"conf_{name}_setup_component_count"] = setup_count.astype(float)
    out[f"conf_{name}_trigger_component_count"] = trigger_count.astype(float)
    out[f"conf_{name}_setup"] = (setup_score.ge(float(setup_threshold)) & setup_count.ge(int(setup_required))).astype(float)
    out[f"conf_{name}_trigger"] = (out[f"conf_{name}_setup"].gt(0.0) & trigger_score.ge(float(trigger_threshold)) & trigger_count.ge(int(trigger_required))).astype(float)
    out[f"conf_{name}_score"] = clamp01(evidence_score * 0.60 + setup_score * 0.20 + trigger_score * 0.20)


def component_frame(out: DataFrame, hypothesis_name: str, components: dict[str, Series | float], component_role: str) -> DataFrame:
    values = []
    for component_name, component in components.items():
        column = f"conf_{hypothesis_name}_cmp_{component_role}_{component_name}"
        series = clamp01(as_series(component, out.index).fillna(0.0))
        out[column] = series
        values.append(series)
    if not values:
        return DataFrame(index=out.index)
    return pd.concat(values, axis=1)


def add_hypothesis(out: DataFrame, name: str, setup: Series, trigger: Series, components: list[Series | float]) -> None:
    setup_score = clamp01(as_series(setup, out.index))
    trigger_score = clamp01(as_series(trigger, out.index))
    component_frame = pd.concat([as_series(component, out.index).fillna(0.0).clip(lower=0.0, upper=1.0) for component in components], axis=1)
    out[f"conf_{name}_setup"] = setup_score.ge(0.20).astype(float)
    out[f"conf_{name}_trigger"] = trigger_score.ge(0.20).astype(float)
    out[f"conf_{name}_score"] = clamp01(component_frame.mean(axis=1, skipna=True) * 0.70 + setup_score * 0.15 + trigger_score * 0.15)


def num(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def flag(frame: DataFrame, column: str) -> Series:
    return num(frame, column).fillna(0.0).gt(0.0)


def zscore(series: Series, window: int, *, min_periods: int) -> Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (series - mean) / std


def rolling_percentile_current(series: Series, window: int, *, min_periods: int) -> Series:
    def percentile(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[np.isfinite(values)]
        if not np.isfinite(current) or len(valid) == 0:
            return np.nan
        return float(np.mean(valid <= current))

    return series.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def sum_bool(items: list[Series]) -> Series:
    if not items:
        return pd.Series(dtype=float)
    total = pd.Series(0.0, index=items[0].index)
    for item in items:
        total = total + item.fillna(False).astype(float)
    return total


def mean_existing(frame: DataFrame, columns: list[str]) -> Series:
    existing = [pd.to_numeric(frame[column], errors="coerce") for column in columns if column in frame]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(existing, axis=1).mean(axis=1, skipna=True)


def max_existing(frame: DataFrame, columns: list[str]) -> Series:
    existing = [pd.to_numeric(frame[column], errors="coerce") for column in columns if column in frame]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(existing, axis=1).max(axis=1, skipna=True)


def std_existing(frame: DataFrame, columns: list[str]) -> Series:
    existing = [pd.to_numeric(frame[column], errors="coerce") for column in columns if column in frame]
    if len(existing) < 2:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.concat(existing, axis=1).std(axis=1, skipna=True).fillna(0.0)


def min_abs_existing(frame: DataFrame, columns: list[str]) -> Series:
    existing = [pd.to_numeric(frame[column], errors="coerce").abs() for column in columns if column in frame]
    if not existing:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(existing, axis=1).min(axis=1, skipna=True)


def mean_scores(frame: DataFrame, columns: list[str]) -> Series:
    return clamp01(mean_existing(frame, columns).fillna(0.0))


def clamp01(series: Series | float) -> Series:
    if isinstance(series, Series):
        return series.clip(lower=0.0, upper=1.0)
    return pd.Series(float(series)).clip(lower=0.0, upper=1.0)


def as_series(value: Series | float, index: pd.Index) -> Series:
    if isinstance(value, Series):
        return value.reindex(index)
    return pd.Series(float(value), index=index)


def future_extreme(series: Series, horizon: int, method: str) -> Series:
    future = series.shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    result = rolling.max() if method == "max" else rolling.min()
    return result.iloc[::-1]


def continuous_future_mask(frame: DataFrame, horizon: int) -> Series:
    if "date" not in frame:
        return pd.Series(True, index=frame.index)
    date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return date.shift(-horizon).eq(date + pd.Timedelta(hours=horizon)).fillna(False)


def continuous_past_mask(frame: DataFrame, horizon: int) -> Series:
    if "date" not in frame:
        return pd.Series(True, index=frame.index)
    date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return date.shift(horizon).eq(date - pd.Timedelta(hours=horizon)).fillna(False)


def time_to_threshold(frame: DataFrame, *, up_pct: float | None, down_pct: float | None, horizon: int) -> Series:
    close = num(frame, "close").to_numpy(dtype=float)
    high = num(frame, "high").to_numpy(dtype=float)
    low = num(frame, "low").to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").to_numpy() if "date" in frame else None
    out = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if not np.isfinite(close[i]) or close[i] == 0.0:
            continue
        hit = False
        valid_horizon = True
        for step in range(1, horizon + 1):
            j = i + step
            if j >= len(frame):
                valid_horizon = False
                break
            if dates is not None and pd.Timestamp(dates[j]) != pd.Timestamp(dates[i]) + pd.Timedelta(hours=step):
                valid_horizon = False
                break
            if up_pct is not None and high[j] >= close[i] * (1.0 + up_pct):
                out[i] = float(step)
                hit = True
                break
            if down_pct is not None and low[j] <= close[i] * (1.0 + down_pct):
                out[i] = float(step)
                hit = True
                break
        if not hit and valid_horizon:
            out[i] = float(horizon + 1)
    return pd.Series(out, index=frame.index)


def hit_before(frame: DataFrame, *, up_pct: float, down_pct: float, horizon: int, direction: str) -> Series:
    close = num(frame, "close").to_numpy(dtype=float)
    high = num(frame, "high").to_numpy(dtype=float)
    low = num(frame, "low").to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").to_numpy() if "date" in frame else None
    out = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if not np.isfinite(close[i]) or close[i] == 0.0:
            continue
        valid_horizon = True
        for step in range(1, horizon + 1):
            j = i + step
            if j >= len(frame):
                valid_horizon = False
                break
            if dates is not None and pd.Timestamp(dates[j]) != pd.Timestamp(dates[i]) + pd.Timedelta(hours=step):
                valid_horizon = False
                break
            hit_up = high[j] >= close[i] * (1.0 + up_pct)
            hit_down = low[j] <= close[i] * (1.0 + down_pct)
            if hit_up and hit_down:
                out[i] = 0.5
                break
            if hit_up:
                out[i] = 1.0 if direction == "up" else 0.0
                break
            if hit_down:
                out[i] = 1.0 if direction == "down" else 0.0
                break
        else:
            if valid_horizon:
                out[i] = 0.0
    return pd.Series(out, index=frame.index)
