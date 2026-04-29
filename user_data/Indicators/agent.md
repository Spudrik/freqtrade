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
