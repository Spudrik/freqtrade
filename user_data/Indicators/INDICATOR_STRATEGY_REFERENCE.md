# Indicator Strategy Reference

This file is reference material for transferring active indicators into strategies. It is not an agent instruction file.

## Active Outputs Summary

Volume Profile:
- Provides market-profile levels and directional context.
- Context convention: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, `-2` bear.
- Strategies may use profile levels, value migration, HVN/LVN behaviour, entry triggers, hold evidence, and exit evidence as inputs.
- The indicator must not own final trade, stop, target, stake, or risk decisions.

Pivot Foundation:
- Canonical cleaned body pivots.
- Shared pivot behaviour belongs here first.
- Other indicators should consume this source instead of defining private pivots.

Pivot-Based Market Structure:
- Consumes Pivot Foundation.
- Provides strategy-facing pivot, structural-zone, market-structure, and context evidence.
- It does not detect channels.

Trendline V2:
- Consumes Pivot Foundation.
- Provides ranked support/resistance trendlines.
- It is lines only; shapes and patterns belong to pattern indicators.

Orderbook Context:
- Provides real order-book/liquidity context where collected data exists.
- Prefer this over OHLCV proxy orderflow when both are available.
- Treat orderbook columns as context, filter, risk, and timing evidence until strategy tests prove otherwise.

Global Context And News/Web Sentiment:
- First-pass external context/filter inputs.
- Ready for guarded research tests.
- Do not treat them as standalone entry signals until forward-return and drawdown buckets prove stable usefulness.

Pattern Indicators:
- Pattern outputs are split by family rather than aggregated through the old Pattern Structure layer.
- Geometry V2 emits triangle, wedge, compression, rectangle, ascending channel, and descending channel evidence through `pg2_*` slot columns plus row-level compression/channel context.
- Reversal, continuation, range, multi-peak, and Wolfe wave logic should be consumed from their own files.
- Multi-peak triple top/bottom identity uses dynamic level breach checks. `triple_level_breach_tolerance_mult` and `triple_level_breach_pivot_grace_bars` are strategy/hyperopt levers: tolerance controls body-level invalidation after reversal, while grace only forgives near-pivot fuzz before the interval has armed on a real opposing reaction.
- Wolfe wave identity now requires meaningful internal reactions between P1-P2 and P3-P4. The `min_internal_reaction_*` levers are intended to reject shallow trend-drift structures that happen to form five alternating pivots.
- Strategies decide whether to trade, wait for breakout, use `1h` as trigger data, treat channel rails as avoid/context evidence, or ignore lower-timeframe patterns.

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

Multi-Peak Strategy / Hyperopt Guidance:
- Treat multi-peak output as pattern identity evidence only. A valid triple top/bottom is not a trade by itself; strategy logic still owns confirmation, direction, stop, target, timeframe weighting, and context filters.
- First-pass hyperopt surface should stay small and identity-focused: `min_triple_spacing_bars`, `min_triple_touch_similarity_score`, `min_triple_touch_turn_score`, `triple_level_breach_tolerance_mult`, `triple_level_breach_pivot_grace_bars`, and `min_triple_quality`.
- Second-pass surface may add pivot and impulse controls if plots show identity problems: `pivot_strength`, `pattern_pivot_strength`, `min_triple_neckline_depth_pct`, and `peak_prior_impulse_min_efficiency`.
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
