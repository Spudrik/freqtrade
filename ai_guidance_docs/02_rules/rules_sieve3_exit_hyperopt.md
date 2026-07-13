---
doc_status: active
default_read: routed
owner: user+agent
purpose: Sieve3 exit-stage Hyperopt design rules for branching Sieve2 entry foundations into exit-system tests.
do_not_use_for: Entry discovery, non-Sieve Hyperopt packages, or detailed run history.
last_rebuilt: 2026-06-25
---

# Rules - Sieve3 Exit Hyperopt

## Core Objective

Sieve3 is the exit-development stage. It should take proven or interesting Sieve2 entry foundations and test exit architecture against a fixed baseline.

Active folder layout:

1. Active Sieve3 exit files live directly under `C:\FreqTradeStuff\user_data\strategies` as top-level `sieve3*.py` files.
2. `sieve3/`, `sieve3_candidates/`, and `sieve2_exit_generated/` are not active working folders for the current campaign.
3. Non-active Sieve material belongs under `user_data/strategies/Archive/`.
4. Rejected Sieve3 files move to `Archive/` only after the user decides they are rejected.
5. Passed Sieve3 files move to a future successful-strategy folder only after the user decides they passed.
6. Sieve3 pass/fail has no fixed thresholds yet; user discretion controls reject/pass decisions.
7. The current campaign goal is to Hyperopt all active top-level Sieve3 files across multiple windows and multiple random seeds.
8. Temporary backlog-clearing folder: during the current Sieve3 exit campaign only, files with fewer than `10` validation trades across multiple windows should be moved out of the active top-level set into `user_data/strategies/sieve3_low_trade_long_window_retest/` for later longer-window testing. Treat these as deferred sparse-sample candidates, not rejected files and not passed files.
Baseline to beat in most cases:

`TP 3% / SL 3%`, no trailing, no partial exits, no dynamic stop movement.

Core question:

Given the same entry signal, which exit system improves expectancy, drawdown, trade duration, tail risk, and robustness without curve-fitting one lucky target?

Launcher boundary:

For Sieve3 exit Hyperopt, disable Entry Sieve exit control. The Sieve entry-test TP/SL override is for entry-quality diagnostics only. Exit-stage files must let the strategy's own fixed-exit, trailing, partial, stop-movement, and target-provider parameters control exits. Baseline `TP 3% / SL 3%` should be encoded as an exit-test branch or control file, not injected by the entry-run override.

Loss-function default:

For Sieve3 exit Hyperopt, use Freqtrade's built-in `MultiMetricHyperOptLoss` by default. It is preferred for exit research because it scores profit, drawdown, profit factor, expectancy ratio, winrate, and minimum trade-count confidence together. Do not default Sieve3 exit runs to entry-oriented winrate-heavy losses such as `CustomHyper`; use a different loss only when the user explicitly asks or the run is an intentional comparison.

Entry-parameter and sample-surface preflight:

For quick Sieve3-versus-Sieve2 baseline comparisons, read `../04_results/sieve3_sieve2_baseline_lookup.md` first. Treat it as compact source context for winrate, return, trade count, drawdown, pair/window scope, and source batch; verify high-stakes claims against current result files before making promotion or rejection decisions.

1. Before any Sieve3 exit batch, confirm the promoted entry surface is locked. The Sieve3 strategy defaults or loaded params must match the selected Sieve2 `buy` params that justified promotion.
2. Do not use BTC-only speed runs for sparse, MTF, structure, or pattern-derived entries unless the source evidence shows enough BTC-only trades. BTC-only can starve Hyperopt and make every exit epoch score as zero-trade noise.
3. For ordinary Sieve3 exit batches, use the standard five core speed pairs unless the user explicitly requests a different pair set: `BTC/USDT:USDT`, `ETH/USDT:USDT`, `ADA/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`.
4. Rare pattern files need larger pair/window coverage. If a training window produces zero or only a few trades, do not judge the exit family from that run; broaden the pair set, choose a larger routed window, or defer the branch to a long-window retest.
5. Treat pair scope as part of the baseline. When comparing Sieve3 exit results to promoted Sieve2 entry evidence, note whether the pair set changed. If a campaign is resumed after changing pair count or pair list, preserve existing rows with their original `speed_pair_count` and resume future rows under the new pair scope rather than rewriting historical rows.
6. Treat very small training samples as a run-setup failure for exit Hyperopt. If the first live Hyperopt for a sparse/pattern branch reports fewer than roughly `10` training trades, stop the batch before it burns epochs and restart with broader pair/window coverage.
7. For sparse or pattern-heavy Sieve3 exit batches, prefer long mixed-cycle training windows such as `long_cycle_mixed_2020_2022` and `long_cycle_mixed_2023_2025` over narrow auto-selected pattern windows. Auto-window mode remains useful for entry discovery, but can under-sample rare exit branches.
8. If five speed pairs plus long mixed-cycle windows still produce fewer than roughly `10` training trades on the first branch, broaden to a larger liquid pair set before judging the exit family. For sparse pattern batches, more pair coverage is preferable to optimizing exits on a handful of trades.
9. Do not treat the validation label or job metadata as proof that a run is valid. Confirm the actual Freqtrade command/log shows the intended pairs, spaces, loss class, random state, and that the first hyperopt has a non-trivial trade count.

Implementation authority:

`rules_sieve3_exit_regeneration_agent.md` is the controlling implementation spec for regenerating Sieve3 exit strategy files. This document defines the campaign shape and exit families; the regeneration-agent rules define how to split files, keep HyperOpt noise controlled, avoid lookahead, keep strategies standalone, and encode trigger/guard/PnL-specific logic.

## File Branching Standard

For each input Sieve2 strategy selected for Sieve3, create top-level `sieve3*.py` exit files unless the source strategy lacks enough trades or target data to support useful tests.

Each Sieve3 file should contain multiple related Hyperopt paths and theories. Split files by exit-system family, not by every parameter combination.

Each branch file should normally stay near `5-10` active HyperOpt parameters in ordinary modes. Use named categorical plans and compact null/off/ignore states instead of wide decimal ranges or enable flags hiding inactive parameters. A branch file is too broad when most modes make several parameters irrelevant, when it combines unrelated target/invalidation providers, or when the winning result cannot be summarized as one trader-readable exit theory.

Use separate files when exit behaviour changes meaningfully:

1. `sieve3_exit_fixed_tp_sl_from_<source>.py`
2. `sieve3_exit_partial_fixed_from_<source>.py`
3. `sieve3_exit_breakeven_from_<source>.py`
4. `sieve3_exit_trailing_from_<source>.py`
5. `sieve3_exit_indicator_target_from_<source>.py`
6. `sieve3_exit_pivot_target_from_<source>.py`
7. `sieve3_exit_vp_target_from_<source>.py`
8. `sieve3_exit_tlv2_target_from_<source>.py`
9. `sieve3_exit_partial_be_trail_from_<source>.py`
10. `sieve3_exit_indicator_partial_trail_from_<source>.py`
11. `sieve3_exit_level_zone_reversal_from_<source>.py`

Good internal Hyperopt mode parameters:

1. `exit_target_mode`
2. `partial_exit_mode`
3. `stop_movement_mode`
4. `trailing_mode`
5. `target_timeframe_mode`
6. `target_level_type`
7. `target_action_mode`
8. `trigger_guard_response_mode`
9. `pnl_overlay_mode`
10. `target_touch_response_mode`
11. `invalidation_action_mode`
12. `level_source_mode`
13. `level_confirmation_mode`
14. `level_action_mode`

Avoid one giant file that mixes fixed exits, VP exits, pivot exits, TLV2 exits, time exits, trailing exits, and partial exits together. Results must remain interpretable in trader terms.

Avoid the opposite failure too: do not split into one tiny file per single condition where a dozen ordinary backtests would answer the question. Each file should still use HyperOpt to compare coherent named mode families.

### Level-Zone Reversal Exits

Use `sieve3_exit_level_zone_reversal_from_<source>.py` for non-pattern sources where the exit question is: price approaches a meaningful target zone, then management depends on touch/rejection/reversal evidence.

The branch should compare compact categorical plans for:

1. target source: prior swing high/low, VP level, source indicator target, or nearest available target,
2. target lookback: broad prior high/low windows such as `24 / 48 / 96 / 168 / 336`,
3. target band: zone tolerance up to roughly `3%`,
4. touch offset: act before exact level, for example `0 / 1 / 2 / 3%`,
5. confirmation: touch, 1/2/3 candle reversal, 2/3 opposite closes, or close rejection,
6. trigger timeframe: trade timeframe, `1h`, `4h`, `1d`, or source context timeframe when available,
7. action: full exit, partial then breakeven, partial then trail, tighten stop, or full after partial.

Do not apply this generic branch to rare pattern sources by default. Pattern branches should prefer their own indicator-defined projections, boundaries, necklines, and family-specific invalidation contracts.

## Required Metadata

Each Sieve3 exit file must preserve traceability:

1. `SIEVE_STAGE = "sieve3"`
2. `SOURCE_STRATEGY = "<input_sieve2_strategy>"`
3. `SOURCE_RESULT_BATCH = "<source_result_or_review_batch>"`
4. `RESEARCH_PATH = "sieve3_exit_<exit_family>"`
5. `ENTRY_SOURCE_STAGE = "sieve2"`
6. `EXIT_HYPOTHESIS = "<plain-English exit theory>"`

The entry logic should remain materially unchanged from the input Sieve2 file unless the specific Sieve3 test explicitly requires entry-family tagging or target-source capture.

Generated Sieve3 files must remain standalone HyperOpt strategy modules. Do not use parent strategy classes, strategy mixins, or external strategy helper files for generated strategy logic. Project indicator modules under `user_data/Indicators/` remain allowed dependencies when the source entry already uses those outputs.

Each generated file should include a short module-level comment or metadata block naming:

1. source strategy,
2. exit family,
3. primary trigger,
4. primary guard,
5. target provider,
6. invalidation provider,
7. active HyperOpt parameters,
8. branch-local or potentially inactive parameters,
9. why the file was not split further.

## Exit Families To Test In Isolation

### Fixed Arbitrary Exits

These are pure price-percent exits and should be the control group.

Test:

1. Symmetric TP/SL: `2/2`, `3/3`, `4/4`, `5/5`.
2. Reward-skewed TP/SL: `3/2`, `4/2`, `5/2`, `6/3`, `8/4`.
3. Defensive TP/SL: `2/3`, `3/4`, `4/5`.
4. Wide reversal/crash TP/SL: `5/3`, `8/4`, `10/5`, `12/6`.

Purpose:

Find whether the entry has raw edge under fixed risk/reward before adding structural exit logic.

### Partial Fixed Exits

Take some profit at fixed levels and leave the remainder for a larger move.

Useful named-plan ingredients. These are not independent HyperOpt dimensions to multiply together; combine them into named categorical plans before generation.

1. First partial level: `1.5%`, `2%`, `2.5%`, `3%`, `4%`.
2. Partial size: `25%`, `33%`, `50%`, `66%`.
3. Final target: `4%`, `5%`, `6%`, `8%`, `10%`, `12%`.
4. Stop after partial: unchanged, breakeven, breakeven plus fees, previous candle low/high.

Isolation tests:

1. Partial only, no trailing.
2. Partial plus breakeven.
3. Partial plus wider final TP.
4. Partial plus trailing remainder.

### Breakeven Stop Movement

Move stop once the trade reaches a favorable trigger.

Useful named-plan ingredients. These are not independent HyperOpt dimensions to multiply together; combine them into named categorical plans before generation.

1. Breakeven trigger: `1%`, `1.5%`, `2%`, `2.5%`, `3%`, `4%`.
2. Breakeven offset: `0%`, `0.1%`, `0.25%`, `0.5%`.
3. Stop move target: true entry, entry plus fees, entry plus small lock.
4. Optional delay after trigger: `0` to `6` candles.

Tests:

1. Breakeven only.
2. Breakeven plus fixed TP.
3. Breakeven plus partial exit.
4. Breakeven plus trailing.
5. Breakeven after indicator target touch.

This answers whether entries often move favorably before failing.

### Trailing Stop

Test both immediate trailing and delayed trailing.

Useful named-plan ingredients. These are not independent HyperOpt dimensions to multiply together; combine them into named categorical plans before generation.

1. Trailing activation: `1.5%`, `2%`, `3%`, `4%`, `5%`, `8%`.
2. Trailing distance: `0.5%`, `0.75%`, `1%`, `1.5%`, `2%`, `3%`, `4%`, `5%`.
3. ATR trail multiple: `1.0`, `1.5`, `2.0`, `3.0`, `4.0`, `5.0`.
4. Structure lookback: `3`, `6`, `12`, `24`, `48` candles.

Isolation tests:

1. Trailing only after activation.
2. Trailing after first partial.
3. Trailing after indicator target.
4. Trailing with no fixed TP.
5. Trailing with capped final TP.

### Indicator Target Exits

Use project indicator outputs as target levels where available.

Long exit targets may include:

1. Prior pivot high.
2. Confirmed pivot high.
3. 1d pivot range high.
4. 4h pivot range high.
5. TLV2 resistance.
6. Volume Profile POC.
7. Volume Profile VAH.
8. HVN.
9. Prior day high.
10. Prior week high.
11. Pattern upper boundary.

Short exit targets mirror the long side:

1. Prior pivot low.
2. Confirmed pivot low.
3. 1d pivot range low.
4. 4h pivot range low.
5. TLV2 support.
6. Volume Profile POC.
7. Volume Profile VAL.
8. LVN/HVN.
9. Prior day low.
10. Prior week low.
11. Pattern lower boundary.

Isolation tests:

1. Full exit at first target.
2. Partial exit at first target, hold remainder.
3. Target touch plus breakeven stop.
4. Target touch plus trailing remainder.
5. Target ignored unless minimum profit is reached.

## Head-To-Head Challenges

### Fixed Versus Indicator Target

Question:

Is `TP 3%` better than exiting at nearby real market structure?

Compare:

1. Fixed `3/3`.
2. Nearest pivot high/low exit.
3. Nearest 4h target.
4. Nearest 1d target.
5. Nearest VP target.
6. Nearest TLV2 target.

### Resistance Versus Prior Pivot High

For long entries, compare:

1. Exit at resistance level.
2. Exit at previous pivot high.
3. Partial at resistance, full at pivot high.
4. Partial at pivot high, trail remainder.
5. Ignore both unless target is at least `1.5R`.

For short entries, mirror with support and previous pivot low.

### Lower-Timeframe Entry Versus Higher-Timeframe Target

Example:

Entered on 1h. Price hits a 1d pivot range.

Test:

1. Full exit at 1d pivot range.
2. Partial exit at 1d pivot range.
3. Move stop to breakeven at 1d pivot range.
4. Tighten stop to 1h/4h structure at 1d pivot range.
5. Ignore 1d range and keep fixed target.

### Take Profit Versus Stop Management

Compare:

1. Fixed TP only.
2. No TP, trailing only.
3. Partial plus trailing.
4. TP plus breakeven.
5. Target touch plus breakeven.
6. Target touch plus partial plus trailing.

### Arbitrary Partials Versus Structural Partials

Compare:

1. `50%` at `2%`.
2. `50%` at pivot target.
3. `50%` at VP POC/VAH/VAL.
4. `50%` at TLV2 level.
5. `50%` at first meaningful resistance/support.

## Combination Matrix

Do not generate every permutation blindly. Use staged expansion.

Stage A - single-mechanism tests:

1. Fixed TP/SL.
2. Fixed partials.
3. Breakeven movement.
4. Trailing.
5. Indicator full exit.
6. Indicator partial exit.

Stage B - two-mechanism tests:

1. Partial plus breakeven.
2. Partial plus trailing.
3. Indicator target plus breakeven.
4. Indicator partial plus trailing.
5. Fixed TP plus breakeven.
6. Fixed TP plus trailing.

Stage C - three-mechanism tests:

1. Partial plus breakeven plus trailing.
2. Indicator partial plus breakeven plus trailing.
3. Fixed partial plus indicator full target plus breakeven.
4. Indicator target plus structure stop plus trailing.
5. Arbitrary first partial plus structural second partial plus trailing.

Stage D - four-mechanism tests, only for survivors:

1. Partial plus breakeven plus trailing plus final fixed TP.
2. Indicator partial plus breakeven plus ATR trailing plus final structure target.
3. Arbitrary partial plus indicator partial plus breakeven plus trailing.
4. Target touch plus stop ratchet plus trailing remainder plus time stop.

## Categorical Values Worth Hyperopting

Do not use continuous decimal ranges for Sieve3 generation. Use named categorical plans or small categorical numeric sets that answer trader-readable questions. If fine tuning is ever justified, it belongs in a later narrow follow-up branch after a broad categorical theory proves useful.

Fixed target plans:

1. `baseline_3_3`
2. `sym_2_2`
3. `sym_4_4`
4. `sym_5_5`
5. `reward_3_2`
6. `reward_4_2`
7. `reward_5_2`
8. `reward_6_3`
9. `reward_8_4`
10. `reward_10_5`
11. `defensive_2_3`
12. `defensive_3_4`
13. `wide_12_6`

Partial-exit plans:

1. `none`
2. `p25_at_1_5pct`
3. `p33_at_2pct`
4. `p50_at_2pct`
5. `p33_at_3pct`
6. `p50_at_3pct`
7. `p66_at_4pct`
8. `ladder_25_2pct_25_5pct`
9. `ladder_33_2pct_33_6pct`
10. `ladder_50_3pct_25_8pct`

Breakeven and stop-shift plans:

1. `off`
2. `entry_after_1pct`
3. `entry_after_1_5pct`
4. `entry_after_2pct`
5. `fee_after_1_5pct`
6. `lock_0_25_after_2pct`
7. `lock_0_5_after_3pct`
8. `after_first_partial`
9. `after_indicator_touch`
10. `after_htf_target_touch`

Trailing plans:

1. `off`
2. `after_2pct_trail_0_75`
3. `after_3pct_trail_1`
4. `after_4pct_trail_1_5`
5. `after_5pct_trail_2`
6. `after_first_partial_trail_1`
7. `after_indicator_touch_trail_1`
8. `after_htf_touch_trail_1_5`
9. `structure_lookback_6`
10. `structure_lookback_12`
11. `structure_lookback_24`
12. `atr_1_5`
13. `atr_2`
14. `atr_3`

Indicator target filters:

1. `ignore_distance`
2. `skip_if_less_0_5pct`
3. `skip_if_less_1pct`
4. `skip_if_less_1R`
5. `skip_if_less_1_5R`
6. `skip_if_more_8pct`
7. `skip_if_more_12pct`
8. `nearest_only`
9. `next_beyond_nearest`
10. `htf_only`

Trigger/guard and PnL response modes should use compact named plans such as `guard_flip_tighten_only`, `guard_flip_reduce_if_profit`, `trigger_fail_lock_if_profit`, `target_touch_then_guard_decides`, `pnl_overlay_off`, `pnl_overlay_profit_lock`, and `pnl_overlay_asymmetric_loss_cut_profit_trail`.

## Full Sieve3 Exit Campaign Standard

Sieve3 exit work should be broad and aggressive. Do not reduce the campaign to a small starter matrix unless the user explicitly asks for a quick smoke batch.

For every selected Sieve2 source entry, generate a full exit-logic branch set covering fixed targets, arbitrary partials, breakeven movement, trailing, structure targets, indicator targets, trigger/guard responses, PnL overlays, and layered combinations. Source-column availability, trade count, target availability, and split/noise checks are hard gates before branch count. Ten files per source is the target for sufficiently rich sources, not permission to generate irrelevant or noisy files. A strong source may justify dozens of files when it has enough trades and enough target-source columns to support the search.

Every source entry should normally receive these Sieve3 branch families:

1. Fixed TP/SL full sweep.
2. Fixed partial-exit sweep.
3. Breakeven and stop-shift sweep.
4. Trailing stop sweep.
5. Fixed TP/SL plus breakeven combinations.
6. Fixed partial plus breakeven combinations.
7. Fixed partial plus trailing combinations.
8. Breakeven plus trailing combinations.
9. Fixed partial plus breakeven plus trailing combinations.
10. Pivot target full-exit sweep.
11. Pivot target partial-exit sweep.
12. Pivot target plus breakeven and trailing sweep.
13. Volume Profile target full-exit sweep where VP columns exist.
14. Volume Profile target partial plus stop-management sweep where VP columns exist.
15. TLV2/support-resistance target full-exit sweep where TLV2 columns exist.
16. TLV2/support-resistance partial plus stop-management sweep where TLV2 columns exist.
17. Prior high/low and prior day/week level exit sweep.
18. Higher-timeframe target interaction sweep, especially 1h entries hitting 4h/1d/3d target zones.
19. Pattern projection full-versus-partial sweep where pattern target columns exist.
20. Pattern boundary/retest management sweep where pattern boundary, confirmation, or invalidation columns exist.
21. Opposite-pattern reduce/tighten/full-exit sweep where opposing pattern columns exist.
22. Geometry V2 branch family where `pg2_*` columns exist.
23. Reversal-pattern branch family where double-top/bottom or head-and-shoulders columns exist.
24. Continuation-pattern branch family where flag/pennant columns exist.
25. Multi-peak branch family where triple-top/bottom columns exist.
26. Wolfe branch family where `pww_*` columns exist.
27. Arbitrary target versus structural target head-to-head sweep.
28. Structural partial plus arbitrary final target sweep.
29. Arbitrary partial plus structural final target sweep.
30. Multi-partial ladder sweep.
31. Structure-based stop tightening sweep.
32. Time stop and stagnation exit sweep, only as a named exit family.

Each branch file should still be interpretable. A file may contain multiple Hyperopt paths when the paths answer the same trading question. It should not mix unrelated theories just to reduce file count. Split files when the normal active parameter set would exceed roughly `5-10`, when more than about one-third of parameters are inactive in most modes, when target and invalidation providers differ across unrelated concepts, or when a broad Cartesian product of independent choices would swamp the optimizer.

## Full But Controlled Test Coverage

Each selected source entry should test all applicable ideas below unless the required columns are absent, the source has too few trades to support the test, or the test would violate the split/noise-control rules. Full coverage means every useful theory family is represented across the branch set; it does not mean every permutation must exist in one file.

### Fixed TP/SL Sweep

Test useful fixed target shapes as named categorical plans:

1. Symmetric: `1.5/1.5`, `2/2`, `2.5/2.5`, `3/3`, `4/4`, `5/5`, `6/6`.
2. Reward-skewed: `3/1.5`, `3/2`, `4/2`, `5/2`, `5/3`, `6/3`, `8/3`, `8/4`, `10/5`, `12/6`.
3. Defensive: `1.5/2`, `2/3`, `2.5/3`, `3/4`, `4/5`.
4. Crash/reversal wide: `6/4`, `8/4`, `10/5`, `12/6`, `15/8`.

### Arbitrary Partial Sweep

Test single, double, and triple partial structures as named categorical plans. These are ingredients for combined plans, not separate parameters to cross product:

1. First partial trigger: `1%`, `1.5%`, `2%`, `2.5%`, `3%`, `4%`, `5%`.
2. First partial size: `20%`, `25%`, `33%`, `50%`, `66%`, `75%`.
3. Second partial trigger: `3%`, `4%`, `5%`, `6%`, `8%`, `10%`.
4. Second partial size: `20%`, `25%`, `33%`, `50%`.
5. Third partial trigger: `6%`, `8%`, `10%`, `12%`, `15%`.
6. Remainder action: fixed final TP, trailing, indicator target, time stop, or hold until stop.

### Breakeven And Stop-Shift Sweep

Test stop movement independently and in combinations through named stop-shift plans. These are ingredients for combined plans, not separate parameters to cross product:

1. Move to entry after favorable move.
2. Move to entry plus fees.
3. Move to entry plus `0.1%`, `0.25%`, `0.5%`, `0.75%`.
4. Move to prior candle low/high.
5. Move to local swing low/high.
6. Move to 1h pivot invalidation.
7. Move to 4h structural invalidation.
8. Delay stop movement by `0`, `1`, `2`, `3`, `6`, `12` candles after trigger.

### Trailing Sweep

Test trailing as a primary exit and as a remainder-management tool through named trail plans. These are ingredients for combined plans, not separate parameters to cross product:

1. Immediate trailing.
2. Delayed trailing after profit activation.
3. Trailing after first partial.
4. Trailing after second partial.
5. Trailing after indicator target touch.
6. Percent trail.
7. ATR trail.
8. Swing-structure trail.
9. Candle-low/high trail.
10. Hybrid ATR plus structure trail.

### Structural And Indicator Target Sweep

For each available target source, test:

1. Full exit at first target.
2. Partial exit at first target.
3. Stop to breakeven at first target.
4. Stop to structure at first target.
5. Trail activation at first target.
6. Ignore target unless profit is above a minimum threshold.
7. Ignore target unless target distance is at least `1R`, `1.5R`, or `2R`.
8. Skip target when too close.
9. Skip target when too far.
10. Use nearest target.
11. Use next target beyond nearest.
12. Use first higher-timeframe target only.

Target sources include pivots, TLV2, VP, prior highs/lows, daily/weekly levels, and pattern boundaries where available.

Indicator-target branches must use explicit source-family mappings. Do not keyword-scan columns for likely targets. Do not use future-confirmed pivots, retroactive pattern targets, future candle scans, or ambiguous target availability. If a target level becomes available after entry, treat it as a later trade-management update rather than the original entry target.

Pattern target branches must be split by the exposed indicator contract when possible. Do not collapse Geometry V2, reversal, continuation, multi-peak, and Wolfe patterns into one generic pattern rule when their source columns exist. Use the controlling regeneration spec for exact pattern-family columns and branch behaviour.

### Trigger / Guard / PnL Sweeps

For each source where guard and trigger roles can be reconstructed, test:

1. Trigger intact plus guard aligned: hold, trail, target manage, or partial at target.
2. Trigger intact plus guard weak: ignore, tighten, or reduce only after profit.
3. Trigger intact plus guard opposite: reduce or tighten first; full exit normally needs profit, target touch, multiple opposing guards, or source-specific invalidation.
4. Trigger invalid plus guard aligned: structural stop, tightened stop, or one retest when HTF context still supports.
5. Trigger invalid plus guard weak/opposite: full exit, partial plus lock, or tight structure stop based on PnL bucket.
6. Target touched plus same-direction evidence: partial then trail or next target.
7. Target touched plus opposing evidence: partial, lock, or full exit based on PnL and partial state.

Use compact categorical overlays such as `off`, `loss_protect`, `profit_lock`, `profit_scale`, `asymmetric_loss_hold_profit_reduce`, and `asymmetric_loss_cut_profit_trail`.

### Head-To-Head Sweeps

Every mature Sieve3 source should include direct head-to-head files when the comparison is broad enough to justify HyperOpt. A head-to-head file must include at least one action or state layer such as target action, profit gate, stop response, guard response, or remainder mode. If the comparison is only a handful of static choices, combine it with a related state/action layer or downgrade it to a later validation backtest.

1. Fixed target versus pivot target.
2. Fixed target versus VP target.
3. Fixed target versus TLV2 target.
4. Fixed target versus higher-timeframe prior high/low target.
5. Arbitrary partial versus structural partial.
6. Full structural exit versus structural partial.
7. Structural exit versus breakeven shift at structure.
8. Trailing-only versus fixed final target.
9. Partial plus trailing versus partial plus fixed final target.
10. Breakeven-only versus partial-only.
11. Breakeven plus trailing versus partial plus breakeven.
12. 1h target management versus 4h/1d target management.

## Implementation Safety Requirements

Sieve3 exit files must be testable by Freqtrade and Sieve, not only trader-readable on paper.

1. Full exits, partial exits, stop movement, and trailing may require different Freqtrade methods. Use existing local method signatures and patterns before generating code.
2. A file claiming to test partial exits must use a valid position-adjustment path; a sell signal that exits the whole trade is not a partial exit.
3. A file claiming to test breakeven or stop movement must actually affect stop behaviour used by Freqtrade.
4. A file claiming to test trailing must name whether it is percent trailing, structure trailing, indicator-activated trailing, ATR trailing, or stoploss-based trailing.
5. Every HyperOpt parameter must affect at least one reachable branch mode. Rare branch parameters should be combined into named plans or split into another file.
6. Missing required columns should be a generation-time reason to omit that branch, not a runtime reason to silently fall back to generic fixed exits.
7. Do not add broad exception handlers around exit logic. Fix the target, method, or state mapping instead.
8. Long and short comparisons must be side-aware for target touch, invalidation, current profit, stop movement, and trailing.
9. Source entry logic should remain materially unchanged. Sieve3 is exit development, not entry rescue.

## Run Budget And Scripted Spot Checks

1. Branch-rich Sieve3 files need materially higher epochs than narrow files.
2. Use multiple random states/seeds, typically at least `3`, and more when branch mode count is high.
3. If the runner exposes an initial random/startup candidate setting, set it above default. Do not add unsupported flags.
4. Review top several candidates, not only the best candidate.
5. Do not judge branch quality from a tiny smoke run or one low-epoch run.
6. After batches run, use approved scripts to spot-check mechanism behaviour. Do not have agents manually read raw HyperOpt logs.
7. Scripted spot checks should confirm whether partial-exit, stop-shift, trailing, trigger/guard, and PnL-overlay branches actually appear in parsed result summaries or generated diagnostic outputs.
8. If a mechanism has zero events in a batch, report it as a scripted diagnostic finding and decide whether the branch is too rare, broken, or simply absent in that sample.

## Result Interpretation

Do not promote a Sieve3 exit system by raw profit alone.

### Analysis Shape For Existing Results

When the user asks for analysis of existing Sieve3 exit results, do not return a raw top-N table as the main answer. Raw row rankings over-represent repeated variants of the same entry source, exit family, seed, or window.

Default analysis shape:

1. Group first by entry/source condition.
2. Within each entry/source, group by exit family.
3. Compare selected exit Hyperopt params before comparing seed performance.
4. Pick the best representative row for each `entry source + exit family`.
5. Show alternates only when they add a meaningful trade-off, such as lower drawdown, higher winrate, higher trade count, or clearly different selected exit params.
6. Suppress dominated variants.
7. Then show any cross-entry exit-family themes.

A dominated variant is usually a row with the same entry/source and exit family that has similar or lower profit, worse drawdown, worse winrate, and no meaningful trade-count or parameter-diversity advantage.

Suspicion checks:

1. Same strategy, seed, and training window with different metrics: investigate before interpretation.
2. Same strategy, seed, training window, and selected params with different metrics: treat as a serious result-integrity bug.
3. Same strategy and seed but different training windows: valid, but label the windows clearly.
4. Different selected params with the same or similar result: possible inactive/equivalent parameters; not proof of robustness until params are compared.
5. Same or near-equivalent params across seeds with strong results: stronger robustness evidence.
6. One entry/source flooding the top rows: collapse to one best representative plus useful alternates.
7. Zero validation trades on a promoted source are not an automatic strategy failure. First check entry-signal coverage, pair/window scope, source baseline window, promoted entry params, and whether the source is sparse or pattern-like. Classify as `coverage_starved` until those checks are resolved.
8. Low winrate plus high profit is not automatically suspicious. It can be a valid asymmetric exit shape. It becomes suspicious only when profit is dominated by terminal `force_exit`, very few trades, or one oversized open-window winner.
9. Terminal `force_exit` means the trade was still open at validation end. Treat force-exit-heavy rows as `terminal_dependent_promising`, not fake and not cleanly proven. Do not subtract force-exit profit and present the remainder as a real alternate result; use that only as a diagnostic to show dependency on unresolved open trades.
10. For terminal-dependent rows, inspect exit reasons, trade durations, open/close dates, selected params, and source baseline before judging. Prefer longer validation windows or closed-trade sanity checks before calling the exit logic robust.

Useful secondary aggregates:

1. Entry robustness across exits: for each entry/source, report median result, top-quartile result, hit-rate of profitable rows, hit-rate of low-drawdown rows, and count of distinct profitable exit families. This can highlight entries that stay useful across many exit styles.
2. Exit-family robustness across entries: for each exit family, report how often it appears in the top tier of its entry/source group, its profitable-row rate, low-drawdown-row rate, and median/percentile drawdown and profit rank. This can highlight broadly useful exit concepts.
3. Parameter-region robustness: count repeated or near-equivalent selected exit params across seeds/windows/sources before claiming convergence.
4. Risk quality: treat lower drawdown percentiles and smoother profit factor as separate strengths from raw return.

Do not average seeds or collapse rows into a single verdict. Aggregates are screening diagnostics only; specific candidate rows remain the evidence for tradeable exit paths.

### Seed Repeat Interpretation

For Sieve3 exit research, each result row is its own exit-search candidate:

`fixed entry source + training window + random seed + selected exit hyperparams + validation result`

The entry logic is fixed. Different seeds are not retesting the entry; they are exploring different optimiser paths through the exit parameter space.

Do not average seeds or treat seed agreement/disagreement as a strategy vote. A bad result from one seed does not invalidate a good result from another seed unless both resolved to the same or near-equivalent exit hyperparams. If the selected exit params differ, the rows are different exit solutions.

Interpret seed repeats like this:

1. Good result with unique params: valid standalone exit candidate.
2. Multiple good results converging on same or equivalent params: strong evidence that exit region is robust.
3. Bad results converging on same or equivalent params: strong evidence that exit region is poor.
4. Mixed results with different params: do not average or call this contradiction; it shows separate exit paths.
5. Multiple different profitable param sets: evidence that the fixed entry may support several viable exit styles.

Before judging seed consistency, compare selected Hyperopt params first, then compare performance. The question is not "did every seed win?" It is "which exit parameter sets worked, and did any independently converge?"

Track:

1. Profit versus `3/3` baseline.
2. Winrate.
3. Trade count.
4. Max drawdown.
5. Average loss.
6. Average win.
7. Profit factor.
8. Expectancy.
9. Average duration.
10. Percent of trades reaching first favorable excursion.
11. How often breakeven saves a loser.
12. How often partials cap a later winner.
13. Whether indicator targets are hit before fixed TP.
14. Which trigger/guard/PnL response mode won.
15. Whether the winning mode repeated across random states/seeds.
16. Whether the winning parameters were active in the actual winning branch.

The key diagnostic is not only whether the exit system won. It is what failure mode the exit system solved.

## Stop Conditions

Park a Sieve3 exit branch when:

1. It loses to the `3/3` baseline across the relevant windows with no clear failure-mode improvement.
2. It improves return only by increasing drawdown or trade duration unacceptably.
3. It works only through a hidden inactive parameter or uninterpretable gate.
4. It depends on a target source that is unavailable or unreliable for the source entry family.
5. It requires changing the source entry logic beyond traceable tagging or target capture.
