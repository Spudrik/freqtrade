# Indicator Review TODO

This file is the working cleanup list from the broad indicator review. Remove
items as they are resolved, retested, and either committed or explicitly parked.

## 1. Strategy-Ready / Highest Value

1. `pattern_geometry_v2.py`
   - Keep as the main direct structure indicator.
   - Best use: triangle, wedge, rectangle, channel, and compression context on
     `1h` and `4h`.
   - Verify updated squeeze semantics:
     - Shared `pg2_compression_flag` and `pg2_best_width_atr` were removed.
     - Each family now has `*_width_atr` and `*_squeeze_active`.
     - Retest strategy-facing columns and plots before removing this TODO.
   - Keep monitoring runtime. Median was acceptable, but p90 was high.

2. `complex_volume_profile.py`
   - Useful as context and entry support.
   - `vp_entry_trigger_long` looked useful.
   - `vp_entry_trigger_short` was weaker and should be treated cautiously.
   - Default output was reduced to strategy-facing columns.
   - Raw pressure, distance, migration-component, event-component, entropy,
     skew, and kurtosis columns are diagnostics-only through
     `include_diagnostics=True`.
   - Retest strategy files that directly referenced hidden diagnostic columns.

3. `complex_relative_strength.py`
   - Useful as pair-ranking and market-relative guard context.
   - Advice now separates regime-aware outputs:
     - `rs_go_long = 1` for relative strength with stable/rising reference and
       target regimes.
     - `rs_go_short = 1` for falling reference, falling target, and relative
       weakness.
     - `rs_long_caution = 1` for relative strength while the reference is
       falling.
   - Future pandas warning from `pct_change` default fill behavior was fixed.
   - Forward-return retest summary:
     - `rs_go_long`: positive overall edge, weak only on `8h`.
     - `rs_go_short`: positive overall edge, weak on `4h`/`8h`.
     - `rs_long_caution`: useful as a warning label only; do not treat it as a
       hard long veto because strong relative assets often keep outperforming
       while the reference falls.
   - Optional next step: visual spot-check a few `go_short` and caution windows,
     then decide whether to keep, retune, or timeframe-gate `4h`/`8h` shorts.

4. `pattern_bos_choch.py`
   - Replaces the broad `pivot_based_market_structure.py` module.
   - Focused on BOS/CHoCH events and structure state only.
   - Duplicate `_5` strength columns, generic score, range position, break
     level, invalidation level, active swing levels, and raw pivot-score outputs
     were removed from normal output.
   - HH/HL/LH/LL sequence labels are optional through `include_sequence=True`.
   - Raw pivot/index columns are diagnostics-only through
     `include_diagnostics=True`.
   - Retest strategy imports and any old `pa_*` references separately because
     this intentionally breaks the old broad market-structure API.

## 2. Partial / Needs Retune

1. `pattern_reversal.py`
   - Active fixed price-percent gates were replaced with rolling body/ATR/pivot
     prominence scaling.
   - Current moderate defaults had positive weighted forward-return edge for
     double tops, double bottoms, and normal H&S in the cleanup window pack.
   - Inverse H&S remains inconsistent in the cleanup window pack and should be
     retuned or marked lower confidence before strategies rely on it.

2. `pattern_wolfe_waves.py`
   - Sparse but interesting.
   - Some forward-return checks looked strong, but sample size was too small.
   - Keep as experimental until visual review and larger sample validation.

3. `pattern_multi_peak.py`
   - Weak current usefulness.
   - Triple tops/bottoms overlap with reversal logic but performed worse.
   - Decide whether to retune, merge conceptually with reversal, or archive.

4. `pattern_continuation.py`
   - Weakest technical indicator in the review.
   - Slowest file by median runtime.
   - Final flag/pennant outputs were sparse relative to computation cost.
   - Decide whether to archive, rewrite leaner, or keep only if visual review
     proves it catches structures other files cannot.

## 3. External Context Indicators

1. `external_orderbook_context_features.py`
   - Not ready for strategy use.
   - Runtime was about 2.2-2.45 seconds for roughly 100 rows in the recent test.
   - Ready rows were low, and risk-block rows were high.
   - Exports 124 columns by default in the tested path.
   - Query performance issue:
     - Current query can full-scan `orderbook_metric_bars`.
     - Consider an index on `canonical_pair`, `timeframe_seconds`,
       `market_key`, and `ts_end`.
   - Decide compact strategy-facing output profile before using in strategies.

2. `external_global_context_features.py`
   - Not ready.
   - Fast, but state did not fire in the recent sample.
   - Current output appears neutral rather than actionable.
   - Needs more history and validation before strategy use.

3. `external_news_web_sentiment_features.py`
   - Not ready.
   - Recent-only data and weak signal checks.
   - Needs more history and validation before strategy use.

## 4. Cross-Indicator Cleanup

1. Naming and semantics
   - Keep strategy-facing naming consistent where meanings match.
   - Avoid forcing identical names where indicators genuinely differ.
   - Distinguish pattern structure, context guard, and tactical advice outputs.

2. Runtime
   - Prioritize `pattern_continuation.py`, `pattern_geometry_v2.py`, TLV2 p90,
     and `external_orderbook_context_features.py`.
   - Avoid runtime changes that alter downstream semantics unless explicitly
     tested before/after.

3. Dead code and redundant levers
   - Remove dormant levels, unused optional paths, and abandoned experiments.
   - Do not keep temporary code if it is not being used.
   - List deletions for approval before editing.

4. Documentation
   - Document which indicators are trade-pattern indicators and which are
     market-status or guard indicators.
   - Document best-known timeframe focus.
   - Document known weak outputs so strategies do not over-trust them.

## 5. Review Evidence

Generated review files:

1. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\technical_indicator_runtime_summary.csv`
2. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\technical_indicator_signal_summary.csv`
3. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\technical_pattern_signal_summary.csv`
4. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\technical_indicator_strategy_activity_by_indicator.csv`
5. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\external_context_runtime_activity.csv`
6. `C:\FreqTradeStuff\user_data\plot\indicator_usefulness_review\external_context_signal_summary.csv`
