# Indicator Strategy Reference

This file is reference material for transferring active indicators into strategies. It is not an agent instruction file.

## Active Outputs Summary

Volume Profile:
- Provides market-profile levels and directional context.
- Context convention: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, `-2` bear.
- Strategies may use profile levels, value migration, HVN/LVN behaviour, entry triggers, hold evidence, and exit evidence as inputs.
- The indicator must not own final trade, stop, target, stake, or risk decisions.

Relative Strength:
- Provides benchmark-relative pair strength, regime-aware long/short advice, hold/exit evidence, and caution context.
- Strategies should treat it as pair selection, regime filtering, and position-management evidence rather than a complete setup.

Pivot Foundation:
- Canonical cleaned body pivots.
- Shared pivot behaviour belongs here first.
- Other indicators should consume this source instead of defining private pivots.

BOS/CHoCH Market Structure:
- Consumes Pivot Foundation.
- Provides compact BOS/CHoCH event flags and structure state.
- It does not emit structural zones, range position, invalidation levels, active swing levels, or raw pivot-score outputs in normal mode.
- It does not detect channels.

Trendline V2:
- Consumes Pivot Foundation.
- Provides ranked support/resistance trendlines.
- It is lines only; shapes and patterns belong to pattern indicators.

Orderbook Context:
- Provides real order-book/liquidity context where collected data exists.
- Prefer this over OHLCV proxy orderflow when both are available.
- Treat orderbook columns as context, filter, risk, and timing evidence until strategy tests prove otherwise.

Global Context:
- First-pass external risk-appetite/regime context.
- Not production-ready; the recent sample was fast but mostly neutral.
- Do not treat it as a standalone entry signal until forward-return and drawdown buckets prove stable usefulness.

News/Web Sentiment:
- First-pass article/event sentiment context from launcher News/Web SQLite data.
- Not production-ready; recent data coverage and signal checks were weak.
- Treat missing or sparse event history as unavailable context, not as neutral conviction.

Orderbook Context Strategy / Hyperopt Guidance:
- Treat Orderbook Context as real liquidity/order-flow context, not as an OHLCV pattern replacement. The recent review had low ready-row coverage, high risk-block activity, and roughly 2.2-2.45 second runtime for about 100 rows, so it is not ready for broad production-style strategy use.
- First-pass strategy logic should hard-gate on `obctx_ready == 1` and `obctx_risk_block == 0` before trusting `obctx_score_long`, `obctx_score_short`, `obctx_state`, `obctx_book_bias_score`, `obctx_pressure_score`, `obctx_wall_score`, or wall/liquidity-zone levels.
- First-pass hyperopt should focus on data quality and headline sensitivity: `min_coverage_ratio`, `pressure_threshold`, `wall_near_threshold_bps`, `microprice_scale_bps`, `spread_penalty_bps`, `short_window`, `medium_window`, and `persistence_window`.
- Keep `include_diagnostics=False` and `include_market_context=False` in normal strategy tests. Turn diagnostics on only for root-cause plots of spread, imbalance, wall, comparison, and missing-coverage behavior.
- Do not optimize market lists, comparison pairs, source timeframe, readiness, wall thresholds, and strategy entries in one pass. First prove that ready rows are frequent enough and that risk blocks behave sensibly.
- `allow_missing=True` is useful for resilient research runs, but it should not silently convert missing order-book history into a valid neutral signal. Missing rows should remain blocked by readiness/risk gates.
- Before treating this as a serious hyperopt surface, fix or account for the known query/runtime issue around `orderbook_metric_bars` scans and decide the compact strategy-facing output profile.

Global Context Strategy / Hyperopt Guidance:
- Treat Global Context as a broad regime/risk-appetite filter. The recent sample was fast but mostly neutral, so it should not be used as a standalone entry signal.
- First-pass strategy logic should consume `gctx_score_latest`, `gctx_score_roll_medium`, `gctx_score_roll_long`, `gctx_score_delta_short`, `gctx_score_delta_medium`, `gctx_greed_persistence`, `gctx_fear_persistence`, `gctx_score_long`, `gctx_score_short`, `gctx_score_abs`, and `gctx_state`.
- First-pass hyperopt should tune only regime timing and persistence: `availability_lag_candles`, `max_age_periods`, `short_window`, `medium_window`, `long_window`, `persistence_window`, `fear_threshold`, `greed_threshold`, and `persistence_min`.
- Source-group selection is a second-pass research lever. Use `source_groups` only after checking which groups actually have data across the backtest window.
- Use Global Context as a guard, stake modifier, or risk-off overlay first. Entry strategies should already work without it, then test whether global context improves drawdown, exposure, or tail-risk buckets.
- Keep `allow_missing=False` for validation unless the strategy explicitly handles missing external context as unavailable rather than neutral.

News/Web Sentiment Strategy / Hyperopt Guidance:
- Treat News/Web Sentiment as event context and risk/regime evidence, not as a trade-pattern indicator. The current implementation is rule-weighted keyword/source scoring over collected articles, so it needs data coverage and forward-return validation before strategy reliance.
- First-pass strategy logic should consume `nwctx_article_count_roll_medium`, `nwctx_weighted_count_roll_medium`, `nwctx_source_breadth_roll_medium`, `nwctx_positive_score_roll_medium`, `nwctx_negative_score_roll_medium`, `nwctx_net_score_roll_medium`, `nwctx_severity_score_roll_medium`, `nwctx_attention_score`, `nwctx_positive_persistence`, `nwctx_negative_persistence`, `nwctx_severity_persistence`, `nwctx_score_long`, `nwctx_score_short`, `nwctx_score_abs`, and `nwctx_state`.
- First-pass hyperopt should focus on event timing and persistence: `availability_lag_candles`, `short_window`, `medium_window`, `long_window`, `persistence_window`, `positive_threshold`, `negative_threshold`, `severity_threshold`, `persistence_min`, and `activity_scale`.
- Dataset choice and weighting dictionaries are second-pass research levers. Tune `datasets`, relevance weights, source-group weights, source-type weights, and topic weights only after checking that the selected sources have enough history across the backtest window.
- Use this first as a guard, exposure modifier, or risk-off/risk-on overlay. Entry logic should already work without it, then test whether sentiment improves drawdown, tail losses, or event-window selectivity.
- Keep `allow_missing=False` for validation unless the strategy explicitly treats missing News/Web rows as unavailable. Do not let missing article history become a valid neutral score.

Pattern Indicators:
- Pattern outputs are split by family rather than aggregated through the old Pattern Structure layer.
- Geometry V2 emits triangle, wedge, compression, rectangle, ascending channel, and descending channel evidence through `pg2_*` slot columns plus row-level compression/channel context.
- Reversal, continuation, multi-peak, and Wolfe wave logic should be consumed from their own files.
- Multi-peak triple top/bottom identity uses dynamic level breach checks. `triple_level_breach_tolerance_mult` and `triple_level_breach_pivot_grace_bars` are strategy/hyperopt levers: tolerance controls body-level invalidation after reversal, while grace only forgives near-pivot fuzz before the interval has armed on a real opposing reaction.
- Wolfe wave identity now requires meaningful internal reactions between P1-P2 and P3-P4. The `min_internal_reaction_*` levers are intended to reject shallow trend-drift structures that happen to form five alternating pivots.
- Strategies decide whether to trade, wait for breakout, use `1h` as trigger data, treat channel rails as avoid/context evidence, or ignore lower-timeframe patterns.

Coverage Scope:
- Dedicated strategy/hyperopt sections cover the active strategy-facing indicators: Orderbook Context, Global Context, News/Web Sentiment, BOS/CHoCH, Trendline V2, Geometry V2, Volume Profile, Relative Strength, Reversal, Continuation, Multi-Peak, and Wolfe Wave.
- `pivot_foundation.py`, `pattern_common.py`, and `__init__.py` are helper/foundation files. They should not get separate strategy hyperopt surfaces unless they start emitting standalone strategy columns.

BOS/CHoCH Strategy / Hyperopt Guidance:
- Treat BOS/CHoCH as market-structure event context, not as a complete setup. It says that a confirmed swing level broke; strategy logic must still decide whether that break is continuation, reversal, liquidity sweep, or late exhaustion.
- First-pass strategy logic should consume `ms_bos_to_bull`, `ms_bos_to_bear`, `ms_choch_to_bull`, `ms_choch_to_bear`, `ms_structure_event`, and `ms_state`.
- The focused first-pass hyperopt surface is small: `strength`, `min_prominence_atr`, `min_pivot_spacing_bars`, `max_pivot_age_bars`, and `breakout_buffer_atr`.
- `strength` and `min_prominence_atr` control swing definition. Raise them to avoid noisy 1h/4h breaks; lower them only when plots show missed obvious structure.
- `breakout_buffer_atr` controls how decisive a body break must be. This should be optimized with timeframe and fee/slippage assumptions, because weak buffers can turn every wick-adjacent close into a structure event.
- Keep `include_sequence` and `include_diagnostics` off in normal strategy runs. Use them for plotting HH/HL/LH/LL and break levels during review, then return to the compact event/state output.
- Do not optimize BOS, CHoCH, sequence labels, and separate pattern entries in the same pass. First decide whether structure events are regime filters, entry confirmations, or exit warnings.

Trendline V2 Strategy / Hyperopt Guidance:
- Treat TLV2 as ranked support/resistance evidence. It exports local lines, scores, distances, slopes, and pivot counts; it does not decide entry direction by itself.
- First-pass strategy logic should consume `tlv2_support_line_rank0`, `tlv2_support_score_rank0`, `tlv2_support_distance_atr_rank0`, `tlv2_resistance_line_rank0`, `tlv2_resistance_score_rank0`, and `tlv2_resistance_distance_atr_rank0`. Lower ranks are secondary context unless plots prove rank0 is too unstable.
- A practical first strategy test is high score plus low distance: nearby high-score support can be a long guard or invalidation reference; nearby high-score resistance can be a long avoid/take-profit context. The mirrored short logic should be validated separately.
- First-pass hyperopt should focus on output quality and active relevance: `timeframe`, `min_output_line_score`, `min_output_active_bars`, `max_active_line_distance_atr_mult`, `proximity_rank_weight`, and `proximity_rank_distance_cap_atr`.
- Only move into line-construction levers after plots show structural problems: `pivot_strength`, `min_anchor_bars`, `max_anchor_bars`, `max_slope_atr_per_bar`, `max_projection_bars`, and the `anchor_break_*` settings.
- Keep duplicate/join/absorption levers out of broad strategy hyperopt at first. They change the identity of the exported line set and can hide whether the strategy actually likes support/resistance behavior.
- Runtime p90 has been high in review, so start with rank0/rank1 and the existing timeframe profiles before increasing `raw_line_output_count` or diagnostics.

Geometry V2 Strategy / Hyperopt Guidance:
- Treat Geometry V2 as line-pair structure, not as a direct buy/sell engine. The broad review showed mixed aggregate directionality: triangle/compression pockets were useful, channel outputs were better as context, and raw wedge/ascending/descending directional trades were inconsistent across windows.
- Keep the first-pass surface family-scoped. Use the family wrappers or explicit `include_*_patterns` gates so a triangle test is not simultaneously optimizing rectangles, wedges, and channels.
- First-pass compression/triangle/wedge hyperopt should focus on identity and timing: `timeframe`, `min_pattern_bars`, `max_pattern_bars`, `compression_max_width_atr`, `squeeze_active_width_atr`, `local_narrowing_min_ratio`, `min_line_score`, `min_containment`, and `max_recent_touch_age_bars`.
- First-pass channel hyperopt should use channels mainly as guard/context evidence. Tune `channel_min_quality`, `channel_min_containment`, `channel_near_boundary_atr_mult`, `channel_breakout_atr_mult`, and `channel_lifecycle_confirm_break_bars` before considering channel breakouts as entries.
- Strategy logic should consume compact family columns first: `*_pattern_present`, `*_indicator_score`, `*_direction`, `*_width_atr`, `*_squeeze_active`, `*_upper`, and `*_lower`. Slot columns are mainly for plotting, overlap review, and debugging.
- Avoid broad optimization of the pivot, envelope, merge, lifecycle, and slope families together. They interact heavily and can make hyperopt select a visually poor line object that only happens to fit one sample.
- Runtime is acceptable but not free. Use one family or a small family set during focused strategy tests, then broaden only after plots show the chosen family is identifying the intended structures.

Volume Profile Strategy / Hyperopt Guidance:
- Treat Volume Profile as market-profile context and tactical entry evidence. The broad review favored `vp_entry_trigger_long` over short-side equivalents; shorts should stay guarded by separate market-regime and trend filters until they prove stable.
- First-pass strategy logic should prioritize `vp_entry_trigger_long`, `vp_node_entry_long`, `vp_node_hold_long`, `vp_node_exit_long`, `vp_market_context`, `vp_score_long`, `vp_score_short`, and the prior value levels `vp_prior_poc`, `vp_prior_vah`, and `vp_prior_val`.
- Use prior value-area columns for entry decisions. Current-row profile levels are useful context, but prior levels avoid making the same completed candle define and trigger its own setup.
- First-pass hyperopt should tune tactical thresholds, not rebuild the whole profile: `entry_score_margin`, `node_near_pct`, `pressure_delta_min`, `volume_percentile_min`, `node_hvn_strength_min`, `node_lvn_thinness_min`, `fast_traverse_atr_mult`, and `score_window`.
- Keep profile-construction knobs narrow unless plots show structural profile errors: `window`, `bins`, `value_area_pct`, `price_source`, and `smooth_bins`. These can change the whole market map and are easy to overfit.
- Context thresholds should be tested as guards after entry evidence is stable: `context_full_min`, `context_full_margin`, `context_soft_min`, `context_soft_margin`, and `context_balance_min`.
- Leave `include_diagnostics=False` in normal strategies. Turn diagnostics on for plot review or explaining a trigger, then turn it back off before production-style backtests.

Relative Strength Strategy / Hyperopt Guidance:
- Treat Relative Strength as pair selection, regime filter, and position-management evidence. The broad review favored suggested/go long behavior; raw score/state columns fired often and did not add enough edge by themselves.
- First-pass strategy logic should prefer `rs_go_long`, `rs_hold_long`, `rs_exit_long`, `rs_long_caution`, `rs_market_context`, `rs_score_long`, `rs_score_short`, `rs_reference_trend_z`, and `rs_target_trend_z`.
- `rs_long_caution` is a warning label, not a hard long veto. Strong relative assets can keep outperforming while the benchmark falls, so strategies should test it as stake reduction, stricter confirmation, or profit-protection context.
- First-pass hyperopt should stay around benchmark choice and entry selectivity: benchmark pair/source, `entry_score_min`, `min_outperformance`, `short_window`, `medium_window`, `long_window`, `percentile_window`, and `entry_cooldown_bars`.
- Tune regime gates separately from raw strength: `long_reference_min_z`, `long_target_min_z`, `short_reference_max_z`, `short_target_max_z`, `short_relative_max_z`, and `caution_reference_max_z`.
- Be conservative with short-side optimization. `rs_go_short` should usually require bearish benchmark context, falling target context, and strategy-level confirmation; do not treat relative weakness alone as permission to short.
- Avoid optimizing score, regime, benchmark, and cooldown surfaces all at once. A clean sequence is benchmark selection, long entry quality, long caution/hold/exit behavior, then short-side research if the strategy actually trades shorts.

Reversal Pattern Strategy / Hyperopt Guidance:
- Treat Reversal output as pattern identity plus neckline confirmation. Current cleanup-window evidence was positive for double tops, double bottoms, and normal head-and-shoulders; inverse head-and-shoulders was inconsistent and should be lower confidence until retuned.
- First-pass strategy logic should keep the four families separate: `pat_double_top_*`, `pat_double_bottom_*`, `pat_head_shoulders_*`, and `pat_inverse_head_shoulders_*`. Do not let a good family hide a weak one in aggregate backtests.
- Long-side logic should prioritize `pat_double_bottom_pattern_present`, `pat_double_bottom_pattern_confirmed`, `pat_double_bottom_indicator_score`, `pat_double_bottom_confirmation_level`, and `pat_double_bottom_target_level`. Treat inverse head-and-shoulders as research-only or lower weight until plots and forward returns improve.
- Short-side logic can research `pat_double_top_*` and `pat_head_shoulders_*`, but should still require trend/regime confirmation rather than trading pattern-present alone.
- First-pass double-pattern hyperopt should focus on identity and confirmation: `min_double_pattern_bars`, `max_double_pattern_bars`, `double_reaction_max_bars`, `min_double_quality`, `min_double_reaction_score`, and `min_double_between_cleanliness_score`.
- First-pass head-and-shoulders hyperopt should focus on shape validity: `min_head_shoulders_quality`, `min_head_shoulders_shoulder_score`, `min_head_shoulders_time_balance_score`, `min_head_shoulders_neckline_score`, `min_head_shoulders_neckline_body_respect_ratio`, and `head_shoulders_max_neckline_slope_atr_per_bar`.
- Shared pivot/scale controls such as `pivot_strength`, `pattern_pivot_strength`, `pivot_min_prominence_atr`, `peak_prior_impulse_min_efficiency`, and the dynamic body/ATR multipliers should be second-pass levers after plots show identity errors.
- Keep `include_pattern_diagnostics=False` in normal runs. Enable diagnostics only when plotting proof lines, neckline scores, reaction scores, and pivot choices.

Continuation Pattern Strategy / Hyperopt Guidance:
- Treat Continuation output as flag/pennant identity and confirmation evidence only. The broad review showed useful pockets but weak asymmetry: flag shorts and pennant longs were better than flag longs and pennant shorts, and runtime was the highest among technical indicators.
- First-pass strategy logic should consume `pat_flag_setup_present`, `pat_flag_pattern_present`, `pat_flag_pattern_confirmed`, `pat_flag_direction`, `pat_flag_indicator_score`, `pat_flag_confirmation_level`, `pat_flag_setup_invalidated`, plus the equivalent `pat_pennant_*` columns.
- Confirmation is a boundary break, not a trade command. Strategy logic still owns trend context, entry timing, stops, target, volume confirmation, and higher-timeframe agreement.
- First-pass hyperopt should focus on pattern identity and confirmation: `min_impulse_efficiency`, `min_impulse_dominance_score`, `min_pattern_bars`, `min_setup_retrace_ratio`, `max_setup_retrace_ratio`, `min_continuation_pole_score`, `min_continuation_containment_score`, `min_continuation_boundary_touch_score`, `min_continuation_boundary_span_score`, `min_flag_quality`, and `min_pennant_quality`.
- Use the shape gates deliberately: `min_flag_shape_score`, `min_pennant_shape_score`, `min_pennant_boundary_span_score`, and `max_pennant_boundary_start_gap_ratio` are the main levers for rejecting noisy shapes that technically fit the rails.
- Keep impulse scale multipliers and pivot controls as second-pass levers unless plots show missing obvious impulses or accepting weak impulses. These include `min_impulse_body_mult`, `min_impulse_atr_mult`, `min_impulse_volume_ratio`, `pivot_strength`, and `pattern_pivot_strength`.
- Because runtime is expensive, test one side/family/timeframe at a time before broadening. Use diagnostics only for plotting proof lines and then turn `include_pattern_diagnostics` back off.

Multi-Peak Strategy / Hyperopt Guidance:
- Treat multi-peak output as pattern identity evidence only. A valid triple top/bottom is not a trade by itself; strategy logic still owns confirmation, direction, stop, target, timeframe weighting, and context filters.
- First-pass hyperopt surface should stay small and identity-focused: `min_triple_spacing_bars`, `min_triple_touch_similarity_score`, `min_triple_touch_turn_score`, `triple_level_breach_tolerance_mult`, `triple_level_breach_pivot_grace_bars`, and `min_triple_quality`.
- Second-pass surface may add pivot, impulse, and dynamic scale controls if plots show identity problems: `pivot_strength`, `pattern_pivot_strength`, `peak_premove_body_mult`, `peak_premove_atr_mult`, `peak_depth_body_mult`, `peak_depth_atr_mult`, and `peak_prior_impulse_min_efficiency`.
- Avoid broad hyperopt over every internal scale knob initially. The dynamic body/ATR/prominence multipliers can interact heavily and may overfit samples before the pattern identity surface is stable.
- Sensible breach-grace search should keep `triple_level_breach_pivot_grace_bars` small, for example 0-4 bars. Larger values can forgive real breaks and turn failed structures back into accepted patterns.
- `min_triple_touch_turn_score` is a clipped score, so useful values should stay at or below 1.0.

Wolfe Wave Strategy / Hyperopt Guidance:
- Treat Wolfe output as experimental until the plotted identity set is larger. The broad review showed useful forward-return pockets, but the event count is still small.
- First-pass hyperopt should prefer existing identity levers before broad tuning: `min_pattern_bars`, `min_leg_spacing_bars`, `min_internal_reaction_pct`, `min_internal_reaction_body_mult`, `min_internal_reaction_atr_mult`, and `min_wave_quality`.
- `min_internal_reaction_*` is the specific lever family for shallow five-pivot drift failures. Raising it should remove weak internal pullbacks; lowering it allows smaller Wolfe channels to pass.
- Avoid using Wolfe present/confirmed as an isolated entry trigger. Strategy logic should still own trend context, confirmation timing, target/risk, and higher-timeframe agreement.

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
