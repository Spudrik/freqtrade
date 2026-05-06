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
- Relative strength is a risk, stake, filter, and ranking input first. Entry suggestions from it are optional diagnostics only and must have a clear market rationale.
- Avoid redesigning standard technical indicators unless the proposed output is materially different from what TA-Lib, qtpylib, ATR/range, relative volume, or simple rolling statistics already provide.
- If a proposed indicator mainly repackages standard TA into new names, stop and tell the user before spending refinement time. Prefer structure, liquidity, market profile, order-book, and pattern evidence where the output is genuinely different.
- OHLCV-only volume and volatility proxies are lower priority when real order-book or richer market microstructure data is available.
- Simple confluence may generate entry triggers, but only when it combines indicator families that logically belong together. Do not treat raw signal count as a trade setup.
- Volume Profile market context should be directional, not a standalone chop detector: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, and `-2` bear.
- Score market state, not just single events.
- Combine signals across timeframes into a confluence score.
- Favor structure, liquidity, and regime context over basic lagging indicators alone.

Priority indicator families:
- Confirmed pivot structure and market structure state
- Trendline and channel projection from pivots
- Volume profile, HVN / LVN, VAH / VAL, POC migration
- Real order-book / liquidity behaviour when available; avoid OHLCV proxy orderflow unless explicitly requested
- Pattern detection such as flags, pennants, wedges, triangles, and HH / HL sequence scoring

Target outcome:
- Each module emits evidence columns plus a normalized confidence score.
- All key thresholds and windows should be configurable for hyperopt.
- Scores can be summed across multiple timeframes.
- The final system should produce trade setups that are logical, explainable, and testable against buy and hold.

## Active Indicator Stack

Current foundation order:
- `pivot_foundation.py` is the canonical cleaned-pivot source. Shared pivot behaviour belongs here first.
- `complex_pivot_structure.py` consumes the foundation for strategy-facing pivot, structural-zone, market-structure, and context evidence. It does not detect channels.
- `complex_trendline_projection_v2.py` consumes the foundation and is the canonical ranked trendline generator.
- `complex_pattern_structure.py` consumes confirmed pivots and trendline context where explicitly useful. Channel-like behaviour belongs in pattern detection, not a separate active channel indicator.

Current disposition:
- Done/ready for strategy tests: Volume Profile, Pivot Foundation, Pivot Structure, and Trendline V2.
- Work in progress: Pattern Structure.
- First-pass external context scaffold: `external_global_context_features.py`.
- Removed: Market Regime. It was mostly standard EMA/ADX/ATR/range/volume state repackaging and should not be rebuilt unless a genuinely different regime premise is defined.
- Archived/lower-value: Trendline Channels, Simple Confluence, Volatility Cycles, and OHLCV Complex Volume.

Column naming standard:
- Advice/action columns must say the advice plainly: use `go_long`, `go_short`, `exit_long`, `exit_short`, `hold_long`, or `hold_short` when the column is meant to be acted on directly.
- Cause/evidence columns must not look like advice. Prefer names such as `long_bounce_support`, `short_reject_resistance`, `breakout_up`, `breakdown_down`, `near_support`, or `near_resistance`.
- Avoid ambiguous names such as `resistance_reject_short` when the reader could mistake the final `_short` as exit advice. If the column is a cause, put the side first or use a neutral event name; if it is advice, use `go_short`.
- Indicators remain evidence providers. Even a `go_long` or `go_short` column means "this indicator sees a directional opportunity here", not "the strategy must enter".

Archived prototype modules:
- Legacy trendline, structural trendline, line-engine, old pattern, simple confluence, volatility cycles, and OHLCV complex-volume files live under `archive_legacy_20260504`.
- Do not import archived modules into new strategies or indicators.
- If an old strategy still imports archived modules, treat that strategy as prototype code requiring retuning against the active stack.
- Structural trendlines are not a separate active foundation. Use the same local trendline logic on higher timeframes when structural context is needed.
- Trendline channels were archived because channel-like evidence is now part of pattern development. Do not maintain a separate channel indicator unless pattern work proves it needs a shared standalone foundation.
- Market regime was deleted from the active stack. The previous version was mostly standard technical-indicator state; future regime work should start from a clean premise, ideally using structure/profile/order-book evidence rather than EMA-style labels.

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

## Archived OHLCV Volume And Volatility Notes

`complex_volume_indicators.py` and `complex_volatility_cycles.py` are archived under `archive_legacy_20260504`.

Reason:
- Volatility Cycles mainly repackaged ATR/range/relative-volume concepts that can be derived directly with standard TA primitives.
- OHLCV Complex Volume used candle-body and close-location proxies for delta, CVD, absorption, and stop-runs. That is lower-value if real order-book or richer market microstructure data is available.
- Future volume/liquidity work should prioritize real order-book features over OHLCV proxy orderflow.

If resurrecting either module:
- First explain why the proposed output is not just a renamed standard TA indicator.
- Require plot-first proof that it provides strategy-useful information that Volume Profile, Pattern Structure, Trendlines, Pivots, or real order-book features do not already provide.
- Keep any resurrection behind a new filename or explicit review branch so archived proxy logic does not silently re-enter active strategies.

## External Data Feature Direction

External data indicators should convert existing SQLite datasets into candle-aligned dataframe columns. They must not collect data, scrape APIs, run background processes, or make final trade decisions.

Current external datasets:
- Global Context SQLite: `user_data/research_news_data/global_context/global_context.sqlite`.
- Global Context source config: `user_data/Custom_Launcher/research/config/global_context_sources.json`.
- Global Context table: `global_context_ticks`.
- Global Context columns of interest: `ts`, `source_ts`, `source_id`, `source_group`, `source_type`, `metric_key`, `score`, `source_score`, `calc_score`, `signal`, `value`, `unit`, `notes`, `raw_json`, `created_at`.
- Effective score convention: `score` is 0..100, where lower values are Fear / weaker risk appetite, higher values are Greed / stronger risk appetite, and 50 is neutral.
- Current source groups include `sentiment`, `market`, `defi`, `equity_indices`, and `rates_fx`.

First-pass module:
- `external_global_context_features.py` is a scaffold for another agent to finish and validate.
- It reads the Global Context SQLite rows, aligns them to candle timestamps, applies a one-candle availability lag by default, and emits rolling score/persistence features only.
- It intentionally does not emit entry, exit, hold, stake, risk, or final regime authority columns.
- It defaults to strict missing-data behavior. If the DB path is wrong or empty, fix the collector/path/state issue rather than silently treating missing context as neutral.

First-pass Global Context outputs to validate:
- `gctx_score_latest`: latest aligned effective score normalized to 0..1.
- `gctx_score_roll_short`, `gctx_score_roll_medium`, `gctx_score_roll_long`: rolling score means.
- `gctx_score_delta_short`, `gctx_score_delta_medium`: score change over rolling windows.
- `gctx_greed_persistence`, `gctx_fear_persistence`: fraction of recent candles above/below the configured thresholds.
- `gctx_score_long`, `gctx_score_short`, `gctx_score_abs`, `gctx_state`: shared score-contract columns derived only from the medium rolling score and persistence.
- `gctx_<source_group>_score_roll_medium`, `gctx_<source_group>_greed_persistence`, and `gctx_<source_group>_fear_persistence`: group-level context summaries where the source group exists.

Validation priorities for the next agent:
- Confirm no lookahead by checking that a collector tick cannot influence the same or earlier strategy candle.
- Validate on 1h BTC, ETH, SOL, and at least one higher-beta alt before strategy integration.
- Bucket forward returns and drawdowns by `gctx_score_roll_medium`, `gctx_fear_persistence`, and `gctx_greed_persistence`.
- Test whether Global Context works better as a long/short filter, stake/risk filter, exit pressure input, or short-expansion guard before adding it to `DailyStructureLadderStrategy.py`.
- Keep Global Context separate from News/Web sentiment until News/Web have their own rolling scored features.

## Relative Strength Strategy Outputs

Relative Strength (`rs`) should compare the traded asset against a benchmark such as BTC, ETH, a market index, or a selected basket. It is cross-asset risk, filter, ranking, stake, and exit context first; it does not decide trades alone.

Expected behaviour:
- Strong bullish RS should persist while the asset is outperforming the benchmark across short/medium/long windows and the RS line is high in its own range.
- Strong bearish RS should persist while the asset is underperforming the benchmark across those windows and the RS line is low in its own range.
- Rotation diagnostics should be fresh events around transitions, not every candle in an existing outperformance state. They are useful for investigation, but RS entries need an additional structure/trigger rationale before they should be strategy entries.

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

Market Regime (`regime`) should identify broad risk-on/risk-off conditions from OHLCV behaviour. Its hardest and most important job is early bear/drawdown warning. It should be treated as a higher-timeframe risk, exit, stake, and filter layer, not an entry system.

Expected behaviour:
- Bull regime should persist when the tested evidence reliably captures durable risk-on conditions. EMA stacks, DMI, slope, drawdown, volatility, and volume can be inputs, but none of them are mandatory definitions.
- Bear or crash regime should appear as early as practical before or during larger drawdown conditions, using whichever evidence proves useful without becoming curve-fit noise.
- Neutral/chop should be the default valid state when no directional regime has sufficient evidence.
- `REGIME_CODE_UNKNOWN` should mostly be warmup/invalid data, not normal live-market uncertainty.

Strategy-facing regime concepts:
- Core inputs: `regime_atr`, `regime_atr_pct`, `regime_atr_ratio`, `regime_adx`, `regime_plus_di`, `regime_minus_di`, EMA columns, `regime_trend_slope_pct`, `regime_price_location_atr`, `regime_volume_ratio`, `regime_drawdown_from_recent_high`, and `regime_range_compression_score`.
- Scores/pressure: `regime_bull_score`, `regime_bear_score`, `regime_chop_score`, `regime_crash_score`, `regime_bull_pressure`, `regime_bear_pressure`, `regime_crash_pressure`, and `regime_directional_pressure`.
- Legacy state: `regime_code` uses `-2` crash, `-1` bear, `0` neutral/chop, `1` bull, and `9` unknown/warmup.
- Shared context: `regime_market_context` uses `+2`, `+1`, `0`, `-1`, `-2`; this is the preferred strategy-facing directional context.
- Early warnings: `regime_bear_warning` and `regime_crash_warning` are risk-off context flags intended to appear before or during larger drawdown conditions.
- Recovery context: `regime_bull_recovery` marks improving trend pressure after risk-off or neutral conditions.
- Entry diagnostics: `regime_entry_risk_on_long` and `regime_entry_risk_off_short` are sparse regime-transition diagnostics only. They should not be treated as primary entries unless a later strategy proves a rational use case.
- Composite triggers: `regime_suggested_entry_long/short` mirror the risk-on/risk-off diagnostics for plotting and broad tests only.
- Position evidence: `regime_hold_long`, `regime_hold_short`, `regime_exit_long`, and `regime_exit_short` expose whether the broad regime supports or warns against a direction.
- Legacy response columns from `add_regime_response_columns` are compatibility-only. New strategies should avoid treating regime as stake/risk/exit authority; strategies own final stake, risk, exit, add, and peel decisions.

Regime tuning levers that strategies may expose to hyperopt:
- Windows: `regime_atr_period`, `regime_atr_ema_window`, `regime_adx_period`, `regime_ema_fast`, `regime_ema_slow`, `regime_ema_long`, `regime_volume_window`, `regime_range_window`, `regime_slope_window`, `regime_drawdown_window`, `regime_confirm_bars`, `regime_confirm_choices`, and `regime_event_cooldown_bars`.
- Trend/risk thresholds: `regime_adx_trend_min`, `regime_slope_scale`, `regime_volume_confirm_min`, `regime_bear_atr_spike`, `regime_early_bear_min`, `regime_early_crash_min`, and `regime_bull_recovery_min`.

## Simple Confluence Archive Note

Simple Confluence (`sci`) failed testing and has been archived. Do not add it to new strategies unless explicitly resurrecting and redesigning it from scratch.

## Pivot Structure Strategy Outputs

Pivot Structure should separate tactical local swings from larger structural market memory.

Strategy-facing pivot concepts:
- Local pivots: `pa_pivot_high_*` / `pa_pivot_low_*` are confirmed local swing points for each configured strength. They are useful for tactical pullback, local break, and local trendline logic, but they can be noisy.
- Structural pivots: `pa_structural_pivot_high` / `pa_structural_pivot_low` use stricter confirmation, prominence, spacing, and distance filters. They should be treated as higher-importance market structure.
- Structural zones: `pa_structural_resistance`, `pa_structural_support`, and their zone upper/lower columns are horizontal support/resistance memory from structural pivots. Strategies should prefer these over the local projected support/resistance lines for broad market structure.
- Structural events: `pa_structural_resistance_break`, `pa_structural_support_break`, `pa_structural_resistance_reject`, and `pa_structural_support_reclaim` are de-duplicated evidence flags around structural zones. They are triggers/guards, not final trade decisions.
- Local projected lines: `pa_resistance_line` / `pa_support_line` remain available but are less authoritative. They are based on recent local pivots and should be treated as tactical geometry, not durable structure. Pivot Structure does not build local channels; channel evidence belongs in the dedicated trendline-channel indicator.
- Structural state: `pa_structural_state` is `1` for confirmed higher-high/higher-low structural bias, `-1` for lower-high/lower-low structural bias, and `0` when mixed or undefined.
- Pivot market context: `pa_market_context` uses the shared directional context convention: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, and `-2` bear. Full `+2/-2` states should require stronger structural evidence than `+1/-1`; the lower-intensity states mean directional bias exists but conditions are still messy.
- Trend-aligned breakouts: `pa_ms_bullish_trend_aligned_breakout` and `pa_ms_bearish_trend_aligned_breakout` mean price has crossed the active swing level in the same direction as the prior structure. A `0` value means the event is absent, not that the opposite event happened.
- Trend-flip breakouts: `pa_ms_bullish_trend_flip_breakout` and `pa_ms_bearish_trend_flip_breakout` mean price has crossed the active swing level against the prior structure, suggesting a possible structural direction change. A `0` value means the event is absent.
- Entry diagnostics: `pa_entry_*_long` and `pa_entry_*_short` columns are separated by plain-English reasons, such as structural breakouts, support reclaims, trend-aligned breakouts, trend-flip breakouts, and range support/resistance reactions. They are diagnostic triggers for testing, not final strategy entries.
- Compatibility aliases: older `*_break_*`, `*_continuation_break_*`, `*_reversal_break_*`, `*_bos_*`, and `*_choch_*` columns may exist for backwards compatibility only. Prefer `*_breakout_*`, `*_trend_aligned_breakout_*`, and `*_trend_flip_breakout_*` in new code and plots.

Pivot tuning levers that strategies may expose to hyperopt:
- Local pivot sensitivity: `strength`, `strengths`, `min_prominence_atr`, `min_prominence_pct`, `min_pivot_spacing_bars`, `min_pivot_distance_atr`, `min_pivot_distance_pct`.
- Local line freshness: `max_pivot_age_bars`.
- Structural pivot sensitivity: `structural_strength`, `structural_min_prominence_atr`, `structural_min_prominence_pct`, `structural_min_pivot_spacing_bars`, `structural_min_pivot_distance_atr`, `structural_min_pivot_distance_pct`.
- Structural zones: `structural_max_age_bars`, `structural_zone_atr_mult`, `structural_zone_pct`.

## Trendline V2 Strategy Outputs

Trendline V2 (`tlv2`) is the active trendline foundation. It generates cleaned body-pivot trendlines, filters broken seed lines, joins compatible endpoint continuations, removes contained duplicates, absorbs nearby weaker lines into stronger lines, and exports compact ranked support/resistance slots.

Strategy-facing TLV2 concepts:
- Pivots: `tlv2_pivot_high`, `tlv2_pivot_low`, pivot index, availability index, prominence, and score columns expose the cleaned pivot source used by trendlines and patterns.
- Ranked lines: `tlv2_resistance_line_rankN` and `tlv2_support_line_rankN` are the active line slots. Slots are ranked by current relevance and score, not fixed identities over the whole dataframe.
- Line quality: score, slope, anchor indexes, projection end index, pivot count, and line id columns explain why each ranked line is present.
- Trendline V2 does not emit channels, exits, hold decisions, or final trade entries. Those belong in derived layers.

TLV2 tuning levers that strategies may expose to hyperopt:
- Pivot source: `pivot_strength`, `pivot_method`, `zigzag_atr_mult`, `pivot_score_touch_atr_mult`, and `pivot_score_touch_pct`.
- Candidate limits: `candidate_pivot_count` and `raw_line_output_count`.
- Line validity: `min_anchor_bars`, `max_anchor_bars`, `min_pivot_prominence_atr`, `max_slope_pct_per_bar`, `max_projection_bars`, `max_active_line_distance_pct`, and `max_active_line_distance_atr_mult`.
- Merge/join behaviour: `join_angle_tolerance_pct`, `join_midpoint_tolerance_pct`, `max_joined_pivots`, `shared_pivot_merge_min`, `nearby_merge_proximity_pct`, `absorb_touch_tolerance_pct`, `absorb_angle_tolerance_pct`, and duplicate thresholds.
- Seed-break handling: `anchor_break_edge_bars`, `anchor_break_max_run`, `anchor_break_max_pct`, and `anchor_break_max_atr_mult`.

## Trendline Channel Strategy Outputs

Trendline Channels consume TLV2 ranked support/resistance lines. They do not generate pivots or trendlines.

Strategy-facing channel concepts:
- Ranked channels: `tlv2_local_channel_*_rankN` columns pair resistance/support rails into active channel evidence.
- Rail relevance: `near_data`, `upper_near_data`, `lower_near_data`, and recent rail touch counts suppress geometrically valid channels that are too far from real candle behaviour.
- Shape: `shape` is `0` for parallel, `1` for converging, and `-1` for diverging/broadening. Shape describes geometry only, not a trade decision.
- Events: `breakout_up`, `breakdown_down`, `near_upper`, `near_lower`, and `inside` are pattern/strategy evidence.

Channel tuning levers that strategies may expose to hyperopt:
- Source/slots: `source_prefix`, `output_prefix`, `output_label`, `source_line_count`, and `channel_output_count`.
- Geometry: `channel_min_overlap_bars`, `channel_slope_tolerance_pct`, `channel_parallel_width_change_pct`, `channel_min_convergence_pct`, `channel_min_width_pct`, `channel_max_width_pct`, and `channel_preferred_width_pct`.
- Candle relevance: `channel_data_proximity_pct`, `channel_data_lookback_bars`, `channel_min_recent_rail_touches`, `channel_touch_tolerance_pct`, and `channel_touch_atr_mult`.
- Duplicate/break handling: `channel_duplicate_overlap_pct`, `channel_duplicate_mid_width_mult`, `channel_duplicate_width_tolerance_pct`, and `channel_breakout_grace_bars`.

## Pattern Structure Strategy Outputs

Pattern Structure (`pat`) should identify geometry-backed chart patterns from the active TLV2 pivot and channel foundations. Strategy-facing pivots, trendlines, and channels should still come from the foundation indicators. Pattern Structure may use an internal no-lookahead micro-pivot source for local consolidation scoring because cleaned TLV2 pivots are intentionally too sparse inside short flags/pennants; those micro pivots are diagnostic internals, not exported as a replacement pivot foundation.

Pattern deletion watchlist:
- `pat_triangle_*`, `pat_wedge_*`, and `pat_range_contraction_score` remain review-gated. The old rolling-slope detector was replaced by pivot-envelope geometry. Do not use TLV2 line pairs as a second pattern detector; TLV2 is a standalone trendline foundation and should not redefine pattern labels. Keep this family out of production composite setup/entry columns unless explicitly enabled for review or a strategy experiment.
- `include_channel_sequence_context` is marked `TODO_DELETE_CHECK`. It is too broad for named-pattern detection because sequence+channel evidence can describe ordinary structure rather than a specific chart pattern. Keep it only for explicit review runs.
- TODO: add a production/review output profile. Pattern Structure is still review-wide and can emit 200+ columns; production strategy runs should eventually expose only compact summary, family/type setup, score, go/exit, and management columns.

Strategy-facing PAT concepts:
- Pivot sequence: `pat_pivot_higher_high`, `pat_pivot_lower_high`, `pat_pivot_higher_low`, `pat_pivot_lower_low`, and sequence score columns describe HH/LH/HL/LL behaviour over the configured window.
- Channel context: `pat_channel_*` columns mirror the selected active TLV2 channel rank into a pattern-facing namespace.
- Flag/pennant prototypes: `pat_flag_setup_long/short`, `pat_pennant_setup_long/short`, quality columns, impulse start/end indexes, impulse efficiency, retrace, and contraction diagnostics are WIP visual-review outputs. Current logic is `impulse/pole first -> controlled post-impulse consolidation`, not breakout detection.
- Pattern proof lines: diagnostics can expose geometry columns named like `pat_<pattern>_<side>_lineN_x1/y1/x2/y2` when `include_pattern_diagnostics=True`. For continuation patterns, line1 is the impulse/pole, line2 is the upper consolidation boundary, and line3 is the lower consolidation boundary. For double tops/bottoms, line1 connects the twin pivots and line2 is the neckline. For head-and-shoulders patterns, line1/line2 connect shoulder-head-shoulder and line3 is the neckline. These are visual proof/evidence columns, not trading orders, and are hidden from normal strategy runs.
- Pivot plotting: pivot marker plots must use the source pivot index columns such as `tlv2_pivot_high_index` and `tlv2_pivot_low_index`, not the dataframe row where the pivot was confirmed. Confirmation rows are valid for signal availability; source indexes are required for visual sanity checks so pivots do not appear to float in space.
- Attention columns: `pat_attention_long/short`, `pat_attention_flag_long/short`, and `pat_attention_pennant_long/short` are the preferred strategy-facing “pay attention, this pattern may be appearing” diagnostics. They are aliases of the de-duplicated setup events and are not trade commands. Older `pat_suggested_entry_long/short` columns remain compatibility aliases only.
- Strongest-pattern summary: `pat_strongest_pattern_code`, `pat_strongest_pattern_bias`, and `pat_strongest_pattern_score` are always-on compact summary columns. They let a strategy or review table see the highest-scoring currently visible pattern without reading every subtype column. Code `0` means no active pattern. Positive bias favours long, negative bias favours short.
- Strongest-pattern code map: `101` flag long, `-101` flag short, `102` pennant long, `-102` pennant short, `201` ascending triangle long, `-202` descending triangle short, `203` symmetric triangle long, `-203` symmetric triangle short, `301` falling wedge long, `-302` rising wedge short, `401` converging pattern long, `-401` converging pattern short, `700` neutral rectangle/range, `701` rectangle support bounce long, `-701` rectangle resistance rejection short, `501` double bottom long, `-501` double top short, `601` inverse head-and-shoulders long, and `-601` head-and-shoulders short.
- Setup aliases: `pat_long_setup` and `pat_short_setup` are de-duplicated composite setup flags intended to mark the candle where price action may start to matter. For head-and-shoulders patterns, raw shape detection is separated from setup: `pat_head_shoulders_structure_short` / `pat_inverse_head_shoulders_structure_long` mark confirmed right-shoulder structure, while `pat_head_shoulders_setup_short` / `pat_inverse_head_shoulders_setup_long` mark the first neckline approach/reject/reclaim window after that structure.
- High-quality flag/pennant evidence should require several independent checks to agree: sharp pole, pole efficiency, pole volume, pole break through prior range, directional dominance versus recent opposite moves, controlled retrace, local micro-pivot geometry, and consolidation drift/containment. A real-looking pattern can still fail as a trade; strategies must add higher-timeframe context, breakout confirmation, volume confirmation, risk, and exits.
- Pattern validation should judge whether the structure looked like the named pattern when the signal fired. Do not mark a signal bad only because the later market outcome failed; mark it bad when the signal is noisy, random, or not visually defensible as that pattern.
- Triangle/wedge labels use non-overlapping slope definitions. Ascending triangle means flat resistance plus rising support. Descending triangle means falling resistance plus flat support. Symmetric triangle means falling resistance plus rising support. Falling wedge means both rails fall, with resistance falling faster than support. Rising wedge means both rails rise, with support rising faster than resistance. Ambiguous compression should stay as `pat_converging_pattern_*`, not force a named triangle/wedge label.
- Triangle/wedge outputs are currently unvalidated diagnostics. They are excluded from composite setup and suggested-entry columns unless `include_unvalidated_geometric_patterns=True` is explicitly set for a review run.
- Geometry boundary packet: `pat_geometry_active`, `pat_geometry_bias`, `pat_geometry_upper`, `pat_geometry_lower`, and `pat_geometry_width_pct` describe the currently active pivot-envelope pattern rails. Bias is `1` for long-favoured geometry, `-1` for short-favoured geometry, and `0` when no useful geometry is active.
- Geometry management references: `pat_geometry_stop_ref_long/short` and `pat_geometry_target_ref_long/short` are pattern-derived reference prices. They are not final stoploss/take-profit orders; strategies can use them as context for risk placement or discard them.
- Geometry cause columns: `pat_geometry_long_bounce_support` means long-biased geometry touched or slightly pierced the lower rail and closed back within/above the support area. `pat_geometry_long_breakout` means long-biased geometry closed above the upper rail. `pat_geometry_short_reject_resistance` means short-biased geometry touched or slightly pierced the upper rail and closed back within/below the resistance area. `pat_geometry_short_breakdown` means short-biased geometry closed below the lower rail. These columns explain why the indicator has directional interest.
- Geometry advice columns: `pat_geometry_go_long` means long-biased geometry has produced either support-bounce or upside-breakout evidence. `pat_geometry_go_short` means short-biased geometry has produced either resistance-rejection or downside-breakdown evidence. Strategies may use these as entry triggers, but should normally require volume, regime, higher-timeframe context, or risk filters before entering.
- Geometry invalidation/exit columns: `pat_geometry_invalid_long` and `pat_geometry_exit_long` mean long-biased geometry lost the lower support rail. `pat_geometry_invalid_short` and `pat_geometry_exit_short` mean short-biased geometry lost the upper resistance rail. Strategies can use these as immediate exit evidence or as stop/reference context.
- Geometry boundary-break columns: `pat_geometry_breakout_up` and `pat_geometry_breakdown_down` are neutral rail-break events. They describe what happened to price, not whether the strategy should go long or short. Prefer `pat_geometry_go_long/go_short` for advice.
- Geometry diagnostics: proof-line and internal geometry columns are hidden by default behind `include_pattern_diagnostics=True`. Keep that flag off for normal strategy runs unless visual review or debugging needs the extra columns.
- Rectangle/range pattern: `pat_rectangle_setup` means price appears boxed between repeated horizontal support and resistance. `pat_rectangle_go_long` means the current candle reacts at rectangle support. `pat_rectangle_go_short` means the current candle rejects rectangle resistance. `pat_rectangle_breakout_up` and `pat_rectangle_breakdown_down` are neutral boundary breaks. `pat_rectangle_exit_long` fires on rectangle support loss; `pat_rectangle_exit_short` fires on rectangle resistance loss. Rectangle work is newly started and must be visually reviewed before strategy use.
- Generic sequence-plus-channel context is excluded from `pat_setup_long/short` by default. Set `include_channel_sequence_context=True` only for explicit review runs where broad channel/sequence scaffolding is being tested.
- Setup scaffolding: `pat_setup_long`, `pat_setup_short`, and `pat_setup_contraction` combine sequence, channel, and accepted prototype evidence for review. They are not final named-pattern proof.
- Entry diagnostics: `pat_entry_channel_*`, `pat_entry_flag_*`, and `pat_entry_pennant_*` columns are diagnostic events only.
- Directional context: `pat_market_context` uses `+2`, `+1`, `0`, `-1`, `-2` based on validated setup evidence plus sequence/channel context.
- Composite suggestions: `pat_suggested_entry_long/short` are compatibility diagnostics only. Prefer `pat_attention_long/short` in new strategy work.

PAT tuning levers that strategies may expose to hyperopt:
- Inputs: `output_prefix`, `pivot_prefix`, `trendline_prefix`, `channel_prefix`, `channel_label`, and `channel_rank`.
- Sequence/context: `sequence_window`, `min_sequence_score`, `min_channel_score`, and `entry_cooldown_bars`.
- Flag/pennant prototype: `impulse_window`, `min_impulse_bars`, `min_impulse_pct`, `min_impulse_efficiency`, `min_impulse_volume_ratio`, `impulse_extreme_tolerance_pct`, `impulse_dominance_mult`, `min_impulse_dominance_score`, `min_impulse_break_score`, `max_consolidation_drift_pct_per_bar`, `max_impulse_extreme_age_bars`, `pattern_pivot_strength`, `min_pattern_bars`, `min_setup_retrace_pct`, `max_setup_retrace_pct`, `max_setup_range_pct`, `max_setup_to_impulse_range_mult`, `max_pattern_breakout_pct`, `max_pattern_boundary_excursion_pct`, `min_pattern_side_pivots`, `min_boundary_slope_pct_per_bar`, `max_flag_counter_slope_pct_per_bar`, `parallel_slope_tolerance_pct_per_bar`, `min_flag_quality`, `min_pennant_quality`, and `pattern_side_dominance_mult`.
- Review gates: `include_unvalidated_geometric_patterns` and `include_channel_sequence_context` should stay `False` by default until their plots prove useful.
- Geometry management: `geometry_management_buffer_pct`, `geometry_management_max_age_bars`, and `include_pattern_diagnostics`.
- Rectangle/range: `rectangle_window`, `min_rectangle_bars`, `rectangle_min_width_pct`, `rectangle_max_width_pct`, `rectangle_boundary_tolerance_pct`, `rectangle_management_buffer_pct`, `rectangle_min_side_touches`, `min_rectangle_containment_ratio`, `min_rectangle_quality`, and `include_rectangle_patterns`.
- Future named-pattern work should add specific tuning levers only when the plotted pattern detector proves useful.

Pattern Structure update 2026-05-05:
- Keep Pattern Structure modular. Each pattern family should live in its own small helper block with its own config levers and outputs, even while it remains in `complex_pattern_structure.py`. If the file becomes too large, split helpers into separate pattern modules without changing the strategy-facing column names.
- `pat_triangle_*`, `pat_wedge_*`, and `pat_range_contraction_score` are still WIP. The current detector is stricter than the first rolling-slope version because it now requires shape, containment, and alternating pivot touches, but plots still showed sparse/misclassified target windows. Keep these outputs excluded from composite setup/entry columns by default until further review proves them.
- Double reversal diagnostics are present as promising WIP outputs: `pat_double_top_structure_short`, `pat_double_bottom_structure_long`, de-duplicated `pat_double_top_setup_short`, de-duplicated `pat_double_bottom_setup_long`, setup quality columns, raw structure quality columns, first/second pivot index/price columns, neckline index/price columns, and neckline score columns. Structure columns mark every qualifying second-pivot pattern; setup columns are sparse "pay attention now" events after cooldown and anchor-aware duplicate suppression. They identify plausible local double-top or double-bottom structures, not final trade commands.
- Head-and-shoulders diagnostics are now present as WIP outputs: `pat_head_shoulders_setup_short`, `pat_inverse_head_shoulders_setup_long`, quality columns, left/head/right pivot index/price columns, and neckline columns. Initial plots showed useful sparse detections on higher timeframes and controls, but noisy lower-timeframe examples still need more review.
- Head-and-shoulders review levers are `head_shoulders_window`, `min_head_shoulders_bars`, `max_head_shoulders_bars`, `shoulder_tolerance_pct`, `min_head_prominence_pct`, `min_head_shoulders_neckline_depth_pct`, `min_head_shoulders_prior_move_pct`, `min_head_shoulders_quality`, `head_shoulders_setup_monitor_bars`, and `head_shoulders_neckline_proximity_pct`.
- `pat_attention_double_top_short` and `pat_attention_double_bottom_long` are de-duplicated "pay attention" diagnostics built from the sparse setup columns. Set `include_double_reversal_patterns=True` for review runs or experimental strategies that intentionally test these signals.
- `pat_attention_head_shoulders_short` and `pat_attention_inverse_head_shoulders_long` are de-duplicated review diagnostics. Set `include_head_shoulders_patterns=True` only for review runs or intentional experiments until this family is accepted.
- Triangle/wedge review gates are `include_triangle_wedge_patterns=True` or `include_unvalidated_geometric_patterns=True`. Double reversal review gate is `include_double_reversal_patterns=True`. Keep these gates `False` by default until visual review accepts the relevant pattern family.
- Head-and-shoulders review gate is `include_head_shoulders_patterns=True`.

Pattern Structure update 2026-05-06:
- `complex_pattern_structure.py` is now the public entrypoint/config/context layer only. Pattern-family internals are split into `pattern_continuation.py` for flags/pennants, `pattern_reversal.py` for double tops/bottoms and head-and-shoulders, `pattern_geometry.py` for triangle/wedge review logic, `pattern_range.py` for rectangle/range review logic, and `pattern_common.py` for shared proof-line, pivot, clipping, and geometry helpers.
- The split is intended to preserve strategy-facing column names and behaviour while making each pattern family easier to refine or extract later.
- Post-split smoke validation compiled all pattern modules and passed a prefix no-lookahead check on sampled ETH 4h rows across all `pat_*` columns with review gates enabled.
- Current visual-review status: flags/pennants, double tops/bottoms, and head-and-shoulders/inverse head-and-shoulders are promising review candidates; triangle/wedge remains the weakest family and must stay excluded from production composite setup columns unless explicitly enabled for review.
- Triangle/wedge refinement update: `pattern_geometry.py` now uses bounded pivot-envelope search instead of fitting one rolling cloud. Candidate patterns must be anchored to real high/low pivot boundary pairs, pass containment/contraction/alternation checks, and emit fresh de-duplicated setup events. Aggregate review scores are exposed as `pat_triangle_quality_long/short` and `pat_wedge_quality_long/short`.
- Triangle/wedge defaults are intentionally strict (`min_triangle_quality=0.80`, `min_wedge_quality=0.80`). They remain review-gated by default, but current plots are materially cleaner than the old rolling-fit detector.
- Double reversal refinement update: `pattern_reversal.py` now separates raw double-top/bottom structure rows from de-duplicated setup rows, uses only neckline pivots confirmed by the current row, adds neckline-position quality plus second-pivot rejection/reclaim behaviour into quality scoring, and defaults to `min_double_quality=0.80`. Duplicate suppression compares anchor overlap and neckline proximity, keeping the first no-lookahead attention row while suppressing later repeated rows that describe the same broad structure. Review plots on BTC/ETH/SOL across `1h`, `4h`, `1d`, and `3d` showed cleaner sparse attention events, while weaker examples still remain possible because a pattern can be visually plausible without later playing out.
- Double reversal tuning levers are `double_pattern_window`, `min_double_pattern_bars`, `max_double_pattern_bars`, `double_peak_tolerance_pct`, `min_double_neckline_depth_pct`, `double_min_neckline_position`, `min_double_prior_move_pct`, `min_double_quality`, `double_duplicate_overlap_pct`, `double_duplicate_neckline_tolerance_pct`, and `entry_cooldown_bars`.
- Pattern weakness pass 2026-05-06:
- Flags/pennants now expose component evidence columns: `pat_continuation_pole_score_long/short`, `pat_continuation_retrace_score_long/short`, `pat_continuation_containment_score_long/short`, `pat_flag_shape_score_long/short`, and `pat_pennant_shape_score_long/short`. The detector should only emit continuation setup evidence when pole, retrace, containment, local boundary shape, and quality all agree. Tunable gates are `min_continuation_pole_score`, `min_continuation_retrace_score`, and `min_continuation_containment_score`.
- Broad converging pattern is split from named triangle/wedge claims. `pat_converging_pattern_setup_long/short` and `pat_converging_pattern_quality_long/short` mean "price appears to be compressing between real pivot boundaries"; they do not claim a specific triangle/wedge subtype. Include them in composites only when `include_converging_patterns=True`. Older `pat_converging_structure_*` columns are compatibility aliases only.
- Double reversal setup rows are subdivided into `pat_double_bottom_clean_long`, `pat_double_top_clean_short`, `pat_double_bottom_range_retest_long`, and `pat_double_top_range_retest_short`. Clean means neckline placement and quality are high enough to treat the structure as a stronger double reversal candidate. Range/retest means the structure is plausible but less decisive; strategies should usually demand extra context before using it.
- Head-and-shoulders now exposes component diagnostics: `pat_head_shoulders_structure_quality`, `pat_inverse_head_shoulders_structure_quality`, shoulder/head/neckline score columns, and right-shoulder reaction score columns. Setup rows should only appear after the right-shoulder structure is confirmed and price produces a first neckline approach/reject/reclaim style reaction without using future pivots.
- Pattern-facing channel context now mirrors selected channel relevance columns: `pat_channel_near_data`, `pat_channel_upper_recent_touch_count`, and `pat_channel_lower_recent_touch_count`. Use these as guards for channel-derived pattern evidence; avoid treating channels far from candle data as active pattern context.
- Validation caveat: `pat_channel_*` columns are copied from the WIP channel foundation and can change if the underlying ranked TLV2/channel slot changes in a prefix-run comparison. Treat channel mirrors as review-only until the channel foundation passes no-lookahead ranking checks. Named pattern-family outputs excluding `pat_channel_*` passed the 2026-05-06 prefix-run sample check.
- Raw `pat_*_setup_*` columns can remain active across several candles while the pattern is present. New strategy work should prefer de-duplicated `pat_attention_*` or `pat_entry_*` columns when it needs single-point "pay attention now" marks.
- Three-loop refinement pass 2026-05-06:
- Loop 1 converted visible setup/proof outputs for continuation and head-and-shoulders into de-duplicated event marks. Quality diagnostics still describe the pattern evidence, but setup/proof rows should now behave more like single "pay attention here" points.
- Loop 2 added stricter shape/usefulness gates: continuation events now expose and gate on `pat_continuation_terminal_score_long/short`; geometry exposes and gates on `pat_geometry_recent_touch_score`; double reversals expose and gate on `pat_double_top_reaction_score` and `pat_double_bottom_reaction_score`; head-and-shoulders exposes and gates on `pat_head_shoulders_head_position_score` and `pat_inverse_head_shoulders_head_position_score`.
- Loop 3 resolved label collisions: a row should not claim both flag and pennant for the same side unless the higher-quality label wins; named triangle/wedge rows suppress broad converging fallback for the same candidate; head-and-shoulders now applies anchor-aware duplicate suppression across similar shoulder/head/neckline structures.
- New pattern tuning levers from this pass are `min_continuation_terminal_score`, `min_geometry_recent_touch_score`, `min_double_reaction_score`, and `min_head_shoulders_head_position_score`.
- Current visual status after the three-loop pass: flags/pennants, double clean reversals, and head-and-shoulders/inverse are the most defensible. Double range/retest remains useful but should require more strategy context. Triangle/wedge/converging geometry is improved but still the weakest family because broad projected boundaries can look plausible without being a clean human-named pattern; keep the geometry family review-gated until more target-window validation accepts it.
- False-positive guard pass 2026-05-06:
- Continuation patterns now expose and gate on `pat_continuation_boundary_touch_score_long/short` and `pat_continuation_boundary_span_score_long/short`. These reduce flag/pennant false positives where the pole is real but the consolidation rails are based on sparse or clustered pivots instead of a visible local structure. New tuning levers are `min_continuation_boundary_touch_score` and `min_continuation_boundary_span_score`.
- Triangle/wedge/converging geometry now exposes and gates on `pat_geometry_anchor_balance_score` and `pat_geometry_touch_balance_score`. These require the upper and lower boundaries to start and finish in roughly comparable parts of the pattern window, reducing false positives where one rail is stale and the other rail is current. New tuning levers are `min_geometry_anchor_balance_score` and `min_geometry_touch_balance_score`.
- Double top/bottom now exposes and gates on `pat_double_top_between_cleanliness_score` and `pat_double_bottom_between_cleanliness_score`. This penalizes candidate twins where a more extreme same-side pivot appears between the first and second anchor, which usually means the detector is forcing a double pattern onto a messier range. New tuning lever is `min_double_between_cleanliness_score`.
- Head-and-shoulders / inverse head-and-shoulders now exposes and gates on `pat_head_shoulders_neckline_clearance_score` and `pat_inverse_head_shoulders_neckline_clearance_score`. This requires shoulders and head to be clearly separated from the neckline, reducing noisy three-pivot sequences that do not have a meaningful neckline. New tuning lever is `min_head_shoulders_neckline_clearance_score`.
- Visual status after the false-positive pass: best and borderline plots on BTC/ETH/SOL across `1h`, `4h`, `1d`, and `3d` showed mostly plausible pattern evidence rather than random noise. Failed later market outcomes are acceptable when the pattern was visually rational at the signal candle. Geometry remains the weakest family but is now cleaner; keep it review-gated until target-window validation accepts it for strategy tests.
- No-lookahead status after this pass: `complex_pattern_structure.py` and split pattern modules compiled, and a prefix-run comparison on ETH 4h with review gates enabled passed sampled rows across named `pat_*` pattern-family columns excluding WIP `pat_channel_*` mirrors.
- Retired TLV2 static geometry pass 2026-05-06:
- The optional TLV2 line-pair source for triangle/wedge/converging patterns was removed after review. TLV2 works well as its own trendline indicator, but mixing TLV2 rails into pattern naming made pattern geometry harder to reason about.
- Legacy diagnostics such as `pat_geometry_tlv2_source`, `pat_geometry_source_count`, `pat_geometry_experimental_confluence_score`, and `pat_geometry_experimental_confluence_bonus` may remain temporarily for column compatibility, but they should not be used by strategies and should be removed in a later cleanup if plots confirm no value.
- Proof-line columns remain diagnostic evidence for visual review/no-lookahead checks. They are now gated by `include_pattern_diagnostics=False` by default across pattern families so normal strategy runs do not carry plotting-only columns.
- Geometry boundary packet pass 2026-05-06:
- `pattern_geometry.py` now emits a compact strategy-facing boundary packet for active triangle/wedge/converging review geometry. The packet gives strategies rails, bias, stop/target references, cause flags, explicit `go_long/go_short` advice flags, invalidation flags, and exit evidence without requiring proof-line columns.
- First visual packet plots on BTC 1h/4h, ETH 4h, and SOL 4h showed that the packet is mechanically useful: early false shorts were invalidated quickly, while later short evidence captured a partial move. The toy BTC 1h window was still slightly negative using naive setup-entry and 5% target rules, so this remains evidence for strategy filtering rather than a standalone entry system.
