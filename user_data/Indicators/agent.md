# Indicator Objective

Build a multilayer Freqtrade indicator stack that finds real market edge instead of relying on lagging confirmation alone.

Core principles:
- Prefer no-lookahead, confirmed structure over repainting signals.
- Use vectorized NumPy and pandas methods wherever possible.
- Avoid row-by-row dataframe loops.
- Non-vectorized indicator logic is not acceptable unless the user gives explicit written permission first.
- If a non-vectorized exception is approved, add a code comment at the exception site explaining why vectorization was not practical.
- Bounded loops over strengths, horizons, ranks, or candidate slots are acceptable only when the per-candle work remains vectorized over Series or arrays.
- Expose tuning knobs on every indicator so Freqtrade hyperopt can optimize the feature logic and thresholds.
- Indicator modules are for Freqtrade strategy consumption only; they should produce dataframes, evidence columns, and confidence scores, not final trading decisions.
- Indicators can expose market intent, contextual state, entry triggers, hold evidence, and exit evidence, but the strategy owns the final entry, exit, sizing, and risk decision.
- Volume Profile market context should be directional, not a standalone chop detector: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, and `-2` bear.
- Score market state, not just single events.
- Combine signals across timeframes into a confluence score.
- Favor structure, liquidity, and regime context over basic lagging indicators alone.

Priority indicator families:
- Confirmed pivot structure and market structure state
- Trendline and channel projection from pivots
- Volume profile, HVN / LVN, VAH / VAL, POC migration
- Advanced volume behavior, effort-vs-result, anchored VWAP, liquidity sweeps
- Pattern detection such as flags, pennants, wedges, triangles, and HH / HL sequence scoring
- Market regime classification and transition scoring

Target outcome:
- Each module emits evidence columns plus a normalized confidence score.
- All key thresholds and windows should be configurable for hyperopt.
- Scores can be summed across multiple timeframes.
- The final system should produce trade setups that are logical, explainable, and testable against buy and hold.

## Indicator Validation Standard

The primary validation question is:

Does the indicator do what its name implies with enough accuracy and clarity to be worth keeping?

Numeric forward-return validation is secondary. The Indicator External Validator can be useful for smoke checks, contract checks, and rough edge screening after visual review, but it must not be treated as authoritative evidence that an indicator is semantically correct.

Use the numeric validator for:
- Column presence and naming checks.
- Finite values, non-empty outputs, bounded normalized scores, and state-code sanity.
- Rough post-visual edge checks against forward returns.

Do not use the numeric validator as proof that:
- Trendlines touch meaningful pivots.
- Support/resistance levels are structurally useful.
- Volume-profile events identify sensible price/volume behaviour.
- Regime, pattern, volatility, or relative-strength labels mean what their names imply.

## Plot-First Review Method

Before keeping, tuning, or using an indicator in strategies:
- Define the expected visual behaviour in plain language.
- Plot the indicator over multiple market regimes, not only recent data.
- Use long windows for structural or bigger-picture indicators such as regime, volume profile, pivots, trendlines, structural trendlines, and relative strength.
- Use shorter windows for local/hourly indicators such as liquidity sweeps, anchored VWAP, volatility events, and tactical trendline interactions.
- Validate on at least three coins when possible: BTC as the anchor, one liquid major such as ETH or SOL, and one noisier/high-beta coin.
- Split cluttered plots into focused views: levels, events, scores, states, and projected targets should be separated when a combined plot becomes unreadable.
- Treat plots as failing evidence if the output is too cluttered to distinguish the intended behaviour.

Suggested plotting windows:
- 1h local review: 250-600 candles.
- 4h tactical/medium review: 300-900 candles.
- 12h/1d structural review: 500-1200 candles.
- 3d bigger-picture review: as much history as available, usually 500-1000 candles.

Indicator disposition after plotting:
- keep: visually does what the name implies and has coherent score/event columns.
- repair: concept is useful, but implementation is noisy, late, mislabeled, or visually misleading.
- archive/remove: output is incoherent, too cluttered to inspect, or does not represent the named market behaviour.

## Volume Profile Strategy Outputs

Volume Profile should expose profile evidence, context, triggers, hold evidence, and exit evidence. It must not own final trade decisions or emit pre-packaged stop/target/risk-reference decisions.

Strategy-facing VP columns:
- `vp_entry_trigger_long` / `vp_entry_trigger_short`: de-duplicated event trigger evidence. Strategies may use these as entry triggers only after applying their own guards, timeframe context, stake, and risk rules.
- `vp_market_context`: directional context, where `+2` is bull, `+1` is bullish chop, `0` is undefined, `-1` is bearish chop, and `-2` is bear.
- `vp_node_entry_long` / `vp_node_entry_short`: refined HVN/LVN node-specific trigger evidence.
- `vp_node_hold_long` / `vp_node_hold_short`: if already in that direction, node evidence suggests the position still has support from profile structure.
- `vp_node_exit_long` / `vp_node_exit_short`: if already in that direction, node evidence suggests exit/reduce should be considered.
- `vp_value_direction_pct`: value-area midpoint direction over the configured migration window.

VP tuning levers that strategies may expose to hyperopt:
- Profile shape: `window`, `bins`, `value_area_pct`, `price_source`, `smooth_bins`.
- Node quality/proximity: `hvn_threshold`, `lvn_threshold`, `node_near_pct`, `node_hvn_strength_min`, `node_lvn_thinness_min`, `node_hold_near_mult`.
- Pressure/participation: `pressure_delta_min`, `volume_percentile_min`, `fast_traverse_atr_mult`.
- Context/trigger scoring: `poc_migration_window`, `score_window`, `entry_score_margin`, `context_full_min`, `context_full_margin`, `context_soft_min`, `context_soft_margin`, `context_balance_min`.

## Complex Volume Strategy Outputs

Complex Volume (`vol`) should infer volume intent from OHLCV only. It does not have true order-flow delta, so CVD, pressure, absorption, and sweep outputs are proxy evidence derived from candle body, close location, range, and relative volume.

Expected behaviour:
- High-quality bullish readings should cluster around springs, low sweeps with reclaim, bullish absorption, anchored VWAP reclaim, and volume-confirmed upside breaks.
- High-quality bearish readings should cluster around upthrusts, high sweeps with rejection, bearish absorption, anchored VWAP rejection, and volume-confirmed downside breaks.
- Diagnostic entry columns should be sparse fresh events, not every candle in an ongoing pressure state.

Strategy-facing VOL concepts:
- Pressure and CVD proxy: `vol_delta_pressure`, `vol_delta_zscore`, `vol_cvd`, `vol_cvd_trend`, and `vol_cvd_trend_confirm_long/short` describe inferred participation direction.
- Relative volume state: `vol_rvol`, `vol_volume_zscore`, `vol_regime_dry`, `vol_regime_expansion`, `vol_regime_climax`, `vol_regime_capitulation`, and `vol_regime_accumulation` describe participation quality.
- Effort-vs-result: `vol_evr_bull_absorption`, `vol_evr_bear_absorption`, `vol_evr_spring`, `vol_evr_upthrust`, and exhaustion columns identify high-volume candles where the price result may reveal absorption, stop-runs, or exhaustion.
- Liquidity sweep and breakout evidence: `vol_liq_stoprun_long/short`, `vol_vol_breakout_confirm_long/short`, and fakeout-risk columns are raw evidence. Strategies should combine them with structure or regime guards.
- Anchored VWAP context: `vol_avwap`, `vol_avwap_upper/lower`, `vol_avwap_reclaim_long`, `vol_avwap_reject_short`, and mean-reversion columns describe interaction with the current volume anchor.
- Directional context: `vol_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; full states require stronger score separation and recent/active volume pressure.
- Entry diagnostics: `vol_entry_stoprun_long/short`, `vol_entry_absorption_long/short`, `vol_entry_avwap_reclaim_long`, `vol_entry_avwap_reject_short`, `vol_entry_volume_breakout_long`, and `vol_entry_volume_breakdown_short` are reason-specific sparse trigger evidence.
- Composite triggers: `vol_suggested_entry_long/short` are de-duplicated composites of the reason-specific entry diagnostics. They are useful for plotting and smoke tests, but strategies should usually prefer the reason-specific columns.
- Position evidence: `vol_hold_long`, `vol_hold_short`, `vol_exit_long`, and `vol_exit_short` expose whether current volume evidence supports holding or warns that opposite pressure is appearing.

VOL tuning levers that strategies may expose to hyperopt:
- Windows: `short_window`, `medium_window`, `long_window`, `divergence_window`, `sweep_window`, `vwap_window`, and `score_window`.
- Volume thresholds: `dry_rvol`, `expansion_rvol`, `climax_rvol`, `anchor_volume_zscore`, and `vwap_band_mult`.
- Effort/result thresholds: `absorption_zscore`, `low_result_atr`, `wide_result_atr`, `accumulation_close_location`, `absorption_close_location`, and `exhaustion_close_location`.
- Pressure/sweep thresholds: `pressure_zscore_min`, `spring_close_location_min`, `upthrust_close_location_max`, `sweep_reclaim_close_location`, `absorption_delta_tolerance`, and `stoprun_delta_tolerance`.

## Volatility Cycle Strategy Outputs

Volatility Cycles (`vc`) should describe transitions between compression, expansion, and exhaustion. It should not mark ordinary candles as expansion; expansion means ATR/range and participation are above their recent baseline.

Expected behaviour:
- `vc_compression_score` should rise during tight ranges, low ATR ratio, and dry participation.
- `vc_expansion_score` should rise only when short-term ATR/range and relative volume push meaningfully above baseline.
- `vc_exhaustion_score` should rise around large stretched candles with high participation and extreme close location.
- Entry diagnostics should be sparse release/exhaustion events, not every candle in a volatile trend.

Strategy-facing VC concepts:
- Core cycle evidence: `vc_compression_score`, `vc_expansion_score`, `vc_exhaustion_score`, `vc_cycle_state`, `vc_compression_phase`, `vc_expansion_phase`, `vc_exhaustion_phase`, `vc_atr_ratio`, `vc_range_ratio`, `vc_range_atr`, and `vc_rvol`.
- Compression release evidence: `vc_recent_compression`, `vc_expansion_long_setup`, and `vc_expansion_short_setup` show when expansion follows a recent squeeze.
- Directional context: `vc_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; it is volatility-direction context, not a full trend regime.
- Entry diagnostics: `vc_entry_breakout_long`, `vc_entry_breakdown_short`, `vc_entry_exhaustion_reversal_long`, and `vc_entry_exhaustion_reversal_short` are reason-specific sparse events.
- Composite triggers: `vc_suggested_entry_long/short` combine the reason-specific diagnostics for plotting and broad smoke tests.
- Position evidence: `vc_hold_long`, `vc_hold_short`, `vc_exit_long`, and `vc_exit_short` expose whether volatility behaviour supports the direction or warns of exhaustion/opposite expansion.

VC tuning levers that strategies may expose to hyperopt:
- Windows: `atr_period`, `short_window`, `long_window`, `volume_window`, `compression_release_lookback`, `entry_cooldown_bars`, and `context_window`.
- Compression thresholds: `compression_atr_ratio` and `dry_volume_rvol`.
- Expansion thresholds: `expansion_atr_ratio` and `expansion_volume_rvol`.
- Exhaustion thresholds: `exhaustion_range_atr`, `exhaustion_volume_rvol`, and `close_location_extreme`.

## Relative Strength Strategy Outputs

Relative Strength (`rs`) should compare the traded asset against a benchmark such as BTC, ETH, or a market index. It is cross-asset context only; it does not decide trades alone.

Expected behaviour:
- Strong bullish RS should persist while the asset is outperforming the benchmark across short/medium/long windows and the RS line is high in its own range.
- Strong bearish RS should persist while the asset is underperforming the benchmark across those windows and the RS line is low in its own range.
- Rotation diagnostics should be fresh events around transitions, not every candle in an existing outperformance state.

Strategy-facing RS concepts:
- Core relative evidence: `rs_line`, `rs_slope`, `rs_ret_short`, `rs_ret_medium`, `rs_ret_long`, `rs_percentile`, and `rs_benchmark_close`.
- Ongoing state: `rs_outperforming` and `rs_underperforming` are raw ongoing benchmark-relative states. They are guards/context, not entry triggers.
- Directional context: `rs_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; full states require stronger smoothed score and margin separation.
- Entry diagnostics: `rs_entry_rotation_long`, `rs_entry_persistent_strength_long`, `rs_entry_rotation_short`, and `rs_entry_persistent_weakness_short` are sparse benchmark-relative trigger evidence.
- Composite triggers: `rs_suggested_entry_long/short` combine the reason-specific diagnostics for plotting and broad tests.
- Position evidence: `rs_hold_long`, `rs_hold_short`, `rs_exit_long`, and `rs_exit_short` expose whether benchmark-relative behaviour supports holding or warns that the edge has flipped.

RS tuning levers that strategies may expose to hyperopt:
- Windows: `short_window`, `medium_window`, `long_window`, `percentile_window`, `context_window`, and `entry_cooldown_bars`.
- Score scaling: `min_outperformance`, `slope_scale`, and `entry_score_min`.
- Context thresholds: `context_full_min`, `context_full_margin`, `context_soft_min`, and `context_soft_margin`.
- Benchmark selection: `benchmark_close` and the informative benchmark dataframe/series supplied by the strategy.

## Market Regime Strategy Outputs

Market Regime (`regime`) should identify broad risk-on/risk-off conditions from OHLCV behaviour. Its hardest and most important job is early bear/drawdown warning. It should be treated as a higher-timeframe context layer, not an entry system.

Expected behaviour:
- Bull regime should persist when price is above a rising EMA stack with supportive DMI/trend pressure.
- Bear or crash regime should appear when downside trend pressure, drawdown from recent highs, negative slope, and volatility expansion combine.
- Neutral/chop should be the default valid state when no directional regime has sufficient evidence.
- `REGIME_CODE_UNKNOWN` should mostly be warmup/invalid data, not normal live-market uncertainty.

Strategy-facing regime concepts:
- Core inputs: `regime_atr`, `regime_atr_pct`, `regime_atr_ratio`, `regime_adx`, `regime_plus_di`, `regime_minus_di`, EMA columns, `regime_trend_slope_pct`, `regime_price_location_atr`, `regime_volume_ratio`, `regime_drawdown_from_recent_high`, and `regime_range_compression_score`.
- Scores/pressure: `regime_bull_score`, `regime_bear_score`, `regime_chop_score`, `regime_crash_score`, `regime_bull_pressure`, `regime_bear_pressure`, `regime_crash_pressure`, and `regime_directional_pressure`.
- Legacy state: `regime_code` uses `-2` crash, `-1` bear, `0` neutral/chop, `1` bull, and `9` unknown/warmup.
- Shared context: `regime_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; this is the preferred strategy-facing directional context.
- Early warnings: `regime_bear_warning` and `regime_crash_warning` are risk-off context flags intended to appear before or during larger drawdown conditions.
- Recovery context: `regime_bull_recovery` marks improving trend pressure after risk-off or neutral conditions.
- Entry diagnostics: `regime_entry_risk_on_long` and `regime_entry_risk_off_short` are sparse regime-transition diagnostics, not direct strategy entries.
- Composite triggers: `regime_suggested_entry_long/short` mirror the risk-on/risk-off diagnostics for plotting and broad tests.
- Position evidence: `regime_hold_long`, `regime_hold_short`, `regime_exit_long`, and `regime_exit_short` expose whether the broad regime supports or warns against a direction.
- Legacy response columns from `add_regime_response_columns` are compatibility-only. New strategies should avoid treating regime as stake/risk/exit authority; strategies own final stake, risk, exit, add, and peel decisions.

Regime tuning levers that strategies may expose to hyperopt:
- Windows: `regime_atr_period`, `regime_atr_ema_window`, `regime_adx_period`, `regime_ema_fast`, `regime_ema_slow`, `regime_ema_long`, `regime_volume_window`, `regime_range_window`, `regime_slope_window`, `regime_drawdown_window`, `regime_confirm_bars`, `regime_confirm_choices`, and `regime_event_cooldown_bars`.
- Trend/risk thresholds: `regime_adx_trend_min`, `regime_slope_scale`, `regime_volume_confirm_min`, `regime_bear_atr_spike`, `regime_early_bear_min`, `regime_early_crash_min`, and `regime_bull_recovery_min`.

## Pivot Structure Strategy Outputs

Pivot Structure should separate tactical local swings from larger structural market memory.

Strategy-facing pivot concepts:
- Local pivots: `pa_pivot_high_*` / `pa_pivot_low_*` are confirmed local swing points for each configured strength. They are useful for tactical pullback, local break, and local trendline logic, but they can be noisy.
- Structural pivots: `pa_structural_pivot_high` / `pa_structural_pivot_low` use stricter confirmation, prominence, spacing, and distance filters. They should be treated as higher-importance market structure.
- Structural zones: `pa_structural_resistance`, `pa_structural_support`, and their zone upper/lower columns are horizontal support/resistance memory from structural pivots. Strategies should prefer these over the local projected support/resistance lines for broad market structure.
- Structural events: `pa_structural_resistance_break`, `pa_structural_support_break`, `pa_structural_resistance_reject`, and `pa_structural_support_reclaim` are de-duplicated evidence flags around structural zones. They are triggers/guards, not final trade decisions.
- Local projected lines: `pa_resistance_line` / `pa_support_line` remain available but are less authoritative. They are based on recent local pivots and should be treated as tactical geometry, not durable structure.
- Channel compression: `pa_channel_width_ratio` is current channel width divided by recent median width. `pa_channel_compression` is a bounded `0..1` score, where higher means the local pivot channel is unusually tight versus its own recent history.
- Structural state: `pa_structural_state` is `1` for confirmed higher-high/higher-low structural bias, `-1` for lower-high/lower-low structural bias, and `0` when mixed or undefined.
- Pivot market context: `pa_market_context` uses the shared directional context convention: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, and `-2` bear. Full `+2/-2` states should require stronger structural evidence than `+1/-1`; the lower-intensity states mean directional bias exists but conditions are still messy.
- Entry diagnostics: `pa_entry_*_long` and `pa_entry_*_short` columns are separated by plain-English reasons, such as structural breakouts, support reclaims, continuation breaks, reversal breaks, compression breaks, and range support/resistance reactions. They are diagnostic triggers for testing, not final strategy entries.
- Compatibility aliases: older `*_bos_*` and `*_choch_*` columns may exist for backwards compatibility only. Prefer `*_continuation_break_*` and `*_reversal_break_*` in new code and plots.

Pivot tuning levers that strategies may expose to hyperopt:
- Local pivot sensitivity: `strength`, `strengths`, `min_prominence_atr`, `min_prominence_pct`, `min_pivot_spacing_bars`, `min_pivot_distance_atr`, `min_pivot_distance_pct`.
- Local zones/channels: `zone_atr_mult`, `zone_pct`, `min_channel_width_pct`, `max_pivot_age_bars`.
- Structural pivot sensitivity: `structural_strength`, `structural_min_prominence_atr`, `structural_min_prominence_pct`, `structural_min_pivot_spacing_bars`, `structural_min_pivot_distance_atr`, `structural_min_pivot_distance_pct`.
- Structural zones: `structural_max_age_bars`, `structural_zone_atr_mult`, `structural_zone_pct`.
- Compression: `compression_window`, `compression_min_periods`, `compression_full_at_ratio`, `compression_none_at_ratio`.

## Structural Trendline Strategy Outputs

Structural Trendlines (`stl`) should represent sparse, higher-importance support/resistance geometry from confirmed major pivots. They are not intended to draw every local line.

Strategy-facing STL concepts:
- Core levels: `stl_resistance_line`, `stl_support_line`, and ranked variants describe the best structural lines currently known from confirmed pivots.
- Line quality: `stl_resistance_score`, `stl_support_score`, touch counts, span, age, and respect ratios explain why a line was selected.
- Compression: `stl_triangle_score`, `stl_wedge_compression`, and `stl_channel_compression` describe converging support/resistance structure.
- Directional context: `stl_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; full states require stronger line score, trend bias, or recent structural line events.
- Entry diagnostics: `stl_entry_resistance_breakout_long`, `stl_entry_support_reclaim_long`, `stl_entry_triangle_breakout_long`, `stl_entry_support_breakdown_short`, `stl_entry_resistance_reject_short`, and `stl_entry_triangle_breakdown_short` are sparse trigger evidence, not final trade decisions.
- Position evidence: `stl_hold_long`, `stl_hold_short`, `stl_exit_long`, and `stl_exit_short` expose whether structural lines support holding or suggest caution.

STL tuning levers that strategies may expose to hyperopt:
- Pivot source: `strength`, `strengths`, `pivot_prefix`, `missing_pivot_mode`, and optional `pivot_config`.
- Line selection: `ranked_line_count`, `candidate_pivot_count`, `min_touch_count`, `min_anchor_span_bars`, `min_line_span_bars`, `max_line_age_bars`, `min_pivot_prominence_atr`.
- Respect/proximity: `touch_tolerance_atr_mult`, `touch_tolerance_pct`, `min_respect_ratio`, `min_span_age_ratio`, `max_slope_pct_per_bar`, `max_projection_distance_pct`.
- Context/events: `breakout_buffer_pct`, `compression_window`, `compression_min_periods`, `score_window`.

## Tactical Trendline Projection Strategy Outputs

Trendline Projection (`tl`) should represent shorter-lived tactical support/resistance lines from confirmed local pivots. It is expected to be noisier and denser than `stl`.

Strategy-facing TL concepts:
- Core levels: `tl_resistance_line`, `tl_support_line`, ranked line columns, projections, and near/break columns describe current local channel geometry.
- Quality: `tl_support_quality`, `tl_resistance_quality`, touch counts, violation counts, respect ratios, and score columns show whether a local line has been respected enough to matter.
- Directional context: `tl_market_context` uses `+2`, `+1`, `0`, `-1`, `-2` and should be treated as tactical context, not higher-timeframe regime.
- Entry diagnostics: `tl_entry_support_reclaim_long`, `tl_entry_resistance_breakout_long`, `tl_entry_compression_breakout_long`, `tl_entry_resistance_reject_short`, `tl_entry_support_breakdown_short`, and `tl_entry_compression_breakdown_short` are reason-specific trigger evidence.
- Position evidence: `tl_hold_long`, `tl_hold_short`, `tl_exit_long`, and `tl_exit_short` expose tactical line support or warning evidence.

TL tuning levers that strategies may expose to hyperopt:
- Pivot source: `strength`, `strengths`, `pivot_prefix`, `missing_pivot_mode`, and optional `pivot_config`.
- Line quality: `candidate_pivot_count`, `ranked_line_count`, `min_anchor_span_bars`, `max_anchor_age_bars`, `min_anchor_prominence_atr`, `min_fit_touch_count`, `min_respect_ratio`.
- Channel/projection: `near_zone_atr_mult`, `near_zone_pct`, `breakout_buffer_pct`, `max_line_slope_pct_per_bar`, `min_channel_width_pct`, `max_channel_width_pct`, `max_projection_distance_pct`, `max_active_line_distance_pct`.
- Context/events: `compression_window`, `compression_min_periods`, `score_window`, `event_windows`.

## Pattern Structure Strategy Outputs

Pattern Structure (`pat`) should identify impulse-then-consolidation behaviour such as flags and pennants. Ongoing setup state is separate from fresh trigger events so plots and strategies do not treat every setup candle as a new signal.

Strategy-facing PAT concepts:
- Setup state: `pat_flag_state_long/short` and `pat_pennant_state_long/short` describe ongoing flag/pennant conditions.
- Fresh setup events: `pat_flag_long/short` and `pat_pennant_long/short` are de-duplicated setup starts for visual review and trigger testing.
- Breakout context: `pat_raw_breakout_long/short` is the raw consolidation break; `pat_breakout_long/short` requires a recent flag/pennant setup and is de-duplicated.
- Directional context: `pat_market_context` uses `+2`, `+1`, `0`, `-1`, `-2` based on score direction plus recent setup/breakout evidence.
- Entry diagnostics: `pat_entry_flag_breakout_long`, `pat_entry_pennant_breakout_long`, `pat_entry_flag_breakdown_short`, and `pat_entry_pennant_breakdown_short` are reason-specific trigger evidence.
- Position evidence: `pat_hold_long`, `pat_hold_short`, `pat_exit_long`, and `pat_exit_short` expose whether the pattern still supports holding or has produced opposite-pattern/breakout warning evidence.

PAT tuning levers that strategies may expose to hyperopt:
- Shape: `impulse_window`, `consolidation_window`, `impulse_atr_min`, `impulse_pct_min`, `max_retrace_pct`, `min_range_contraction`.
- Participation/compression: `dry_volume_rvol_max`, `breakout_buffer_pct`, `pivot_prefix`, `trendline_prefix`.
