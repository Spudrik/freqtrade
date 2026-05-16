# Indicator Strategy Reference

This file is reference material for transferring active indicators into strategies. It is not an agent instruction file.

Strategy agents should treat every active indicator listed here as an eligible strategy input. This file intentionally avoids reliability rankings, return summaries, and quality opinions. Use it for output meanings, integration shape, and initial hyperopt ranges.

## Active Indicator Scope

Active strategy-facing indicators:
- `pattern_bos_choch.py`
- `complex_trendline_projection_v2.py`
- `pattern_geometry_v2.py`
- `complex_volume_profile.py`
- `complex_relative_strength.py`
- `pattern_reversal.py`
- `pattern_continuation.py`
- `pattern_multi_peak.py`
- `pattern_wolfe_waves.py`

Helper/foundation files:
- `pivot_foundation.py` is the canonical cleaned-pivot source.
- `pattern_common.py` and `__init__.py` are helper files.

Parked external context:
- `work_in_progress/external_orderbook_context_features.py`
- `work_in_progress/external_global_context_features.py`
- `work_in_progress/external_news_web_sentiment_features.py`

The parked external context files are not active strategy inputs.

## Shared Usage Notes

Use indicator outputs as dataframe evidence. Strategies own entries, exits, stop logic, target logic, stake sizing, and cross-indicator confluence.

Start hyperopt with one indicator family or one compact family group at a time. Keep non-strategy columns off in normal strategy runs.

## BOS/CHoCH

Main purpose:
- Confirmed swing break and change-of-character event context from cleaned pivots.

Primary outputs:
- `ms_bos_to_bull`
- `ms_bos_to_bear`
- `ms_choch_to_bull`
- `ms_choch_to_bear`
- `ms_structure_event`
- `ms_state`

Initial hyperopt surface:
- `strength`: `2-5`
- `min_prominence_atr`: `0.20-1.20`
- `min_pivot_spacing_bars`: `1-8`
- `max_pivot_age_bars`: `40-320`
- `breakout_buffer_atr`: `0.00-0.40`

## Trendline V2

Main purpose:
- Ranked support and resistance trendlines from cleaned pivots.

Primary outputs:
- `tlv2_support_line_rank0`
- `tlv2_support_score_rank0`
- `tlv2_support_distance_atr_rank0`
- `tlv2_resistance_line_rank0`
- `tlv2_resistance_score_rank0`
- `tlv2_resistance_distance_atr_rank0`

Initial hyperopt surface:
- `timeframe`: strategy-selected active timeframe values, commonly `1h`, `4h`, `8h`, `1d`
- `min_output_line_score`: `0.35-0.80`
- `min_output_active_bars`: `2-24`
- `max_active_line_distance_atr_mult`: `3.0-10.0`
- `proximity_rank_weight`: `0.00-0.20`
- `proximity_rank_distance_cap_atr`: `3.0-10.0`

Second-pass construction surface:
- `pivot_strength`: `2-4`
- `min_anchor_bars`: `6-24`
- `max_anchor_bars`: `40-140`
- `max_slope_atr_per_bar`: `0.15-0.70`
- `max_projection_bars`: `40-160`
- `raw_line_output_count`: `1-3`

## Geometry V2

Main purpose:
- Triangle, wedge, compression, rectangle, ascending-channel, and descending-channel evidence.

Family-level outputs:
- `pg2_<family>_pattern_present`
- `pg2_<family>_indicator_score`
- `pg2_<family>_direction`
- `pg2_<family>_width_atr`
- `pg2_<family>_squeeze_active`
- `pg2_<family>_upper`
- `pg2_<family>_lower`

Family names:
- `triangle`
- `wedge`
- `compression`
- `rectangle`
- `ascending_channel`
- `descending_channel`

Initial family gates:
- `include_triangle_patterns`: `true/false`
- `include_wedge_patterns`: `true/false`
- `include_compression_patterns`: `true/false`
- `include_rectangle_patterns`: `true/false`
- `include_ascending_channel_patterns`: `true/false`
- `include_descending_channel_patterns`: `true/false`

Initial compression/triangle/wedge surface:
- `timeframe`: strategy-selected active timeframe values
- `min_pattern_bars`: `6-24`
- `max_pattern_bars`: `36-120`
- `compression_max_width_atr`: `1.20-3.50`
- `local_narrowing_lookback_bars`: `4-12`
- `squeeze_active_width_atr`: `1.00-3.50`
- `local_narrowing_min_ratio`: `0.03-0.25`
- `min_line_score`: `0.35-0.75`
- `min_containment`: `0.65-0.95`
- `max_recent_touch_age_bars`: `6-36`

Initial channel/rectangle surface:
- `channel_min_pattern_bars`: `8-48`
- `channel_max_pattern_bars`: `60-200`
- `channel_min_quality`: `0.65-0.95`
- `channel_min_containment`: `0.55-0.90`
- `channel_near_boundary_atr_mult`: `0.30-1.20`
- `channel_breakout_atr_mult`: `0.15-0.80`
- `channel_lifecycle_confirm_break_bars`: `1-4`

Slot columns:
- `pg2_slot_*` columns describe concrete overlapping candidates.
- Start strategy logic from family-level columns, then add slot columns only when a strategy needs multiple simultaneous structures.

## Volume Profile

Main purpose:
- Rolling volume profile levels, value-area context, HVN/LVN interaction, score columns, and tactical entry/hold/exit evidence.

Primary outputs:
- `vp_entry_trigger_long`
- `vp_entry_trigger_short`
- `vp_node_entry_long`
- `vp_node_entry_short`
- `vp_node_hold_long`
- `vp_node_hold_short`
- `vp_node_exit_long`
- `vp_node_exit_short`
- `vp_market_context`
- `vp_score_long`
- `vp_score_short`
- `vp_score_abs`
- `vp_state`
- `vp_prior_poc`
- `vp_prior_vah`
- `vp_prior_val`

Initial tactical surface:
- `window`: `48-192`
- `entry_score_margin`: `0.00-0.10`
- `node_near_pct`: `0.003-0.025`
- `pressure_delta_min`: `0.00-0.20`
- `volume_percentile_min`: `0.30-0.80`
- `node_hvn_strength_min`: `0.45-0.90`
- `node_lvn_thinness_min`: `0.35-0.85`
- `fast_traverse_atr_mult`: `0.60-2.50`
- `score_window`: `24-96`

Second-pass profile construction surface:
- `bins`: `24-96`
- `value_area_pct`: `0.60-0.80`
- `price_source`: `close`, `hl2`, `hlc3`, `ohlc4`
- `smooth_bins`: `1-5`

Context surface:
- `context_full_min`: `0.30-0.70`
- `context_full_margin`: `0.02-0.20`
- `context_soft_min`: `0.15-0.50`
- `context_soft_margin`: `0.01-0.12`
- `context_balance_min`: `0.25-0.65`

## Relative Strength

Main purpose:
- Pair performance relative to a supplied benchmark plus regime-aware advice/context columns.

Primary outputs:
- `rs_go_long`
- `rs_go_short`
- `rs_hold_long`
- `rs_hold_short`
- `rs_exit_long`
- `rs_exit_short`
- `rs_long_caution`
- `rs_market_context`
- `rs_score_long`
- `rs_score_short`
- `rs_score_abs`
- `rs_state`
- `rs_reference_trend_z`
- `rs_target_trend_z`

Initial entry surface:
- benchmark pair/source: strategy-defined categorical choices
- `entry_score_min`: `0.30-0.75`
- `min_outperformance`: `-0.02-0.03`
- `short_window`: `6-24`
- `medium_window`: `24-96`
- `long_window`: `96-288`
- `percentile_window`: `120-480`
- `entry_cooldown_bars`: `0-24`

Regime gate surface:
- `long_reference_min_z`: `-0.50-1.00`
- `long_target_min_z`: `-0.50-1.00`
- `short_reference_max_z`: `-1.00-0.50`
- `short_target_max_z`: `-1.00-0.50`
- `short_relative_max_z`: `-1.00-0.50`
- `caution_reference_max_z`: `-1.00-0.50`

## Reversal Patterns

Main purpose:
- Double top, double bottom, head-and-shoulders, and inverse head-and-shoulders structure plus neckline/target levels.

Family outputs:
- `pat_double_top_*`
- `pat_double_bottom_*`
- `pat_head_shoulders_*`
- `pat_inverse_head_shoulders_*`

Common output suffixes:
- `_pattern_present`
- `_pattern_confirmed`
- `_indicator_score`
- `_confirmation_level`
- `_target_level`
- `_p1_index`, `_p2_index`, and `_p3_index` where applicable

Initial double-pattern surface:
- `double_pattern_window`: `48-140`
- `min_double_pattern_bars`: `6-24`
- `max_double_pattern_bars`: `36-100`
- `double_reaction_max_bars`: `8-48`
- `min_double_quality`: `0.50-0.90`
- `min_double_reaction_score`: `0.20-0.90`
- `min_double_between_cleanliness_score`: `0.20-0.90`

Initial head-and-shoulders surface:
- `head_shoulders_window`: `60-160`
- `min_head_shoulders_bars`: `8-24`
- `max_head_shoulders_bars`: `60-140`
- `min_head_shoulders_quality`: `0.50-0.90`
- `min_head_shoulders_shoulder_score`: `0.25-0.80`
- `min_head_shoulders_time_balance_score`: `0.25-0.80`
- `min_head_shoulders_neckline_score`: `0.20-0.80`
- `min_head_shoulders_neckline_body_respect_ratio`: `0.50-0.95`
- `head_shoulders_max_neckline_slope_atr_per_bar`: `0.05-0.35`

Shared second-pass surface:
- `pivot_strength`: `2-4`
- `pattern_pivot_strength`: `1-3`
- `pivot_min_prominence_atr`: `0.10-0.80`
- `peak_prior_impulse_min_efficiency`: `0.20-0.80`

## Continuation Patterns

Main purpose:
- Flag and pennant setup/confirmation evidence after impulse moves.

Family outputs:
- `pat_flag_*`
- `pat_pennant_*`

Common output suffixes:
- `_setup_present`
- `_pattern_present`
- `_pattern_confirmed`
- `_direction`
- `_indicator_score`
- `_upper`
- `_lower`
- `_confirmation_level`
- `_setup_invalidated`
- `_invalidation_level`

Initial pattern surface:
- `impulse_window`: `24-72`
- `scale_window`: `24-96`
- `min_pattern_bars`: `4-16`
- `max_impulse_extreme_age_bars`: `16-48`
- `min_impulse_efficiency`: `0.20-0.70`
- `min_impulse_dominance_score`: `0.20-0.90`
- `min_setup_retrace_ratio`: `0.02-0.20`
- `max_setup_retrace_ratio`: `0.35-0.85`
- `min_continuation_pole_score`: `0.20-0.80`
- `min_continuation_containment_score`: `0.35-0.90`
- `min_continuation_boundary_touch_score`: `0.25-0.90`
- `min_continuation_boundary_span_score`: `0.20-0.80`
- `min_flag_quality`: `0.50-0.90`
- `min_pennant_quality`: `0.50-0.90`

Shape and impulse surface:
- `min_flag_shape_score`: `0.00-0.70`
- `min_pennant_shape_score`: `0.00-0.70`
- `min_pennant_boundary_span_score`: `0.20-0.80`
- `max_pennant_boundary_start_gap_ratio`: `0.20-0.70`
- `min_impulse_body_mult`: `2.0-8.0`
- `min_impulse_atr_mult`: `0.80-4.00`
- `min_impulse_volume_ratio`: `0.80-1.50`
- `pivot_strength`: `2-4`
- `pattern_pivot_strength`: `1-3`

## Multi-Peak Patterns

Main purpose:
- Triple top and triple bottom structure plus confirmation/target levels.

Family outputs:
- `pat_triple_top_*`
- `pat_triple_bottom_*`

Common output suffixes:
- `_pattern_present`
- `_pattern_confirmed`
- `_indicator_score`
- `_confirmation_level`
- `_target_level`
- `_p1_index`
- `_p2_index`
- `_p3_index`

Initial pattern surface:
- `triple_pattern_window`: `60-150`
- `max_triple_pattern_bars`: `60-130`
- `triple_max_candidate_pivots`: `6-12`
- `min_triple_spacing_bars`: `3-15`
- `min_triple_touch_similarity_score`: `0.35-0.80`
- `min_triple_touch_turn_score`: `0.40-1.00`
- `triple_level_breach_tolerance_mult`: `0.10-0.80`
- `triple_level_breach_pivot_grace_bars`: `0-4`
- `min_triple_quality`: `0.50-0.90`

Entry hypotheses a strategy can model separately:
- Triple-top neckline confirmation: close below `pat_triple_top_confirmation_level`.
- Triple-bottom neckline confirmation: close above `pat_triple_bottom_confirmation_level`.
- Failed-top breakout: close above the max price at `pat_triple_top_p1_index`, `pat_triple_top_p2_index`, and `pat_triple_top_p3_index`.
- Failed-bottom breakdown: close below the min price at `pat_triple_bottom_p1_index`, `pat_triple_bottom_p2_index`, and `pat_triple_bottom_p3_index`.

Second-pass dynamic scale surface:
- `pivot_strength`: `2-4`
- `pattern_pivot_strength`: `1-3`
- `peak_premove_body_mult`: `2.0-8.0`
- `peak_premove_atr_mult`: `0.50-3.00`
- `peak_level_tolerance_body_mult`: `0.50-3.00`
- `peak_level_tolerance_atr_mult`: `0.10-1.50`
- `peak_level_tolerance_prominence_mult`: `0.20-2.00`
- `peak_depth_body_mult`: `1.0-6.0`
- `peak_depth_atr_mult`: `0.25-2.50`
- `peak_reaction_body_mult`: `0.50-4.00`
- `peak_base_return_buffer_body_mult`: `0.00-2.00`
- `peak_prior_impulse_min_efficiency`: `0.20-0.80`

## Wolfe Wave

Main purpose:
- Bullish and bearish five-pivot Wolfe wave structure with setup/confirmation lifecycle.

Primary outputs:
- `pww_bearish_pattern_present`
- `pww_bullish_pattern_present`
- `pww_bearish_pattern_confirmed`
- `pww_bullish_pattern_confirmed`
- `pww_bearish_indicator_score`
- `pww_bullish_indicator_score`
- `pww_bearish_confirmation_level`
- `pww_bullish_confirmation_level`
- `pww_bearish_target_level`
- `pww_bullish_target_level`

Initial hyperopt surface:
- `min_pattern_bars`: `12-30`
- `max_pattern_bars`: `60-160`
- `min_leg_spacing_bars`: `2-8`
- `prior_window`: `32-96`
- `min_internal_reaction_pct`: `0.000-0.030`
- `min_internal_reaction_body_mult`: `0.50-2.50`
- `min_internal_reaction_atr_mult`: `0.10-0.80`
- `min_wave_quality`: `0.50-0.90`

## Naming Migration Target

Advice-only columns should use:
- `go_long`
- `go_short`
- `exit_long`
- `exit_short`
- `hold_long`
- `hold_short`

Advice plus reason should use:
- `go_long__bounce_at_support`
- `go_short__rejection_at_resistance`
- `exit_long__support_lost`
- `exit_short__resistance_reclaimed`

Neutral evidence should use:
- `bounce_at_support`
- `rejection_at_resistance`
- `breakout_above_resistance`
- `breakdown_below_support`
- `near_support`
- `near_resistance`

Existing pattern columns that do not follow this style should be migrated before the relevant pattern file is called production-ready.
