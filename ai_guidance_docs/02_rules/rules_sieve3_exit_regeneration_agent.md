---
doc_status: active
default_read: routed
owner: user+agent
purpose: Agent instructions for regenerating Sieve3 exit strategy files with nuanced, branch-rich, HyperOpt-efficient exit logic.
do_not_use_for: Non-Sieve exit packages, live trading settings, or adding new indicator systems.
last_rebuilt: 2026-06-12
---

# Rules - Sieve3 Exit Regeneration Agent

## Core Correction

The current broad generated Sieve3 exit files are not acceptable as final work.

The failure pattern to correct:

1. Wide decimal sweeps such as `0.010` to `0.250`.
2. Same universal sell-parameter block pasted into every branch.
3. Too many parameters inactive behind branch modes.
4. Keyword-based target discovery standing in for real entry-family-specific exit logic.
5. No explicit trade-state model connecting current profit, target source, guard flip, trigger invalidation, and action choice.

The regeneration objective is not to reduce each file to one tiny question. That would collapse Sieve back into manual backtesting. The objective is to create branch-rich Sieve3 files that use HyperOpt properly:

1. Multiple related exit concepts per file.
2. Broad categorical theory choices, not fine decimal tuning.
3. Null/weak states where possible so parameters can become low-impact without hidden gates.
4. Branch-local parameters only where a branch genuinely needs them.
5. Enough epochs and random states/seeds for HyperOpt to explore the branch modes.

## Non-Negotiable Implementation Rules

2. Strategy files must remain standalone HyperOpt modules. Do not use parent strategy classes, mixin strategy bases, or external strategy helper files for generated Sieve3 strategy logic. Project indicator modules under `user_data/Indicators/` remain allowed dependencies when the source entry already uses those outputs.
3. Do not add new generic TA systems, RSI, MACD, EMA, Bollinger, stochastic, or generic oscillator logic as new exit evidence.
4. Use only the source entry logic, existing project indicator outputs, OHLCV-derived trade state, and existing guard columns already present in the source strategy.
5. Preserve source entry logic materially unchanged.
6. Do not use wide decimal ranges where categorical values or multipliers express the same trader question.
7. Each branch file should contain several related concepts, not a single backtest-like condition and not an unbounded mega-matrix.
8. HyperOpt results must be interpretable: a winning branch mode should explain a trader-readable exit theory.

## File Shape

Each source entry should receive a branch set. A branch file may contain several related internal modes.

Current placement rule:

1. Regenerated or promoted active Sieve3 files belong directly in `C:\FreqTradeStuff\user_data\strategies` as top-level `sieve3*.py` files.
2. Do not put active Sieve3 work in `sieve3/`, `sieve3_candidates/`, or `sieve2_exit_generated/`.
3. Archive rejected or inactive Sieve material under `user_data/strategies/Archive/` when the user decides it is rejected or inactive.
4. Move passed Sieve3 files to a future successful-strategy folder only when the user decides they passed.
5. Sieve3 pass/fail is not rule-thresholded yet; the current work is broad multi-window, multi-seed Hyperopt evidence collection for user discretion.

Good branch file examples:

1. `fixed_rr_and_baseline_from_<source>.py`
2. `partial_ladder_fixed_from_<source>.py`
3. `breakeven_stop_shift_from_<source>.py`
4. `trailing_remainder_from_<source>.py`
5. `indicator_target_full_vs_partial_from_<source>.py`
6. `indicator_target_stop_shift_from_<source>.py`
7. `guard_flip_reduce_or_tighten_from_<source>.py`
8. `trigger_invalidation_from_<source>.py`
9. `structural_target_vs_fixed_from_<source>.py`
10. `mixed_target_partial_stop_from_<source>.py`
11. `mixed_guard_trigger_target_from_<source>.py`
12. `time_stagnation_from_<source>.py`
13. `level_zone_reversal_from_<source>.py`

For strong, high-trade, or target-rich sources, add more files:

1. `vp_target_management_from_<source>.py`
2. `tlv2_target_management_from_<source>.py`
3. `pivot_target_management_from_<source>.py`
4. `pattern_projection_management_from_<source>.py`
5. `htf_target_interaction_from_<source>.py`
6. `multi_partial_structure_ladder_from_<source>.py`
7. `level_zone_reversal_from_<source>.py`

Do not blindly generate every branch for a source that lacks the required columns. If VP, TLV2, pivot, pattern, or MTF target data is absent, omit that target-specific branch or replace it with a relevant source-family branch.

`level_zone_reversal_from_<source>` is the preferred generic non-pattern branch when the source has enough OHLCV, VP, or indicator-target context to test target areas rather than exact prices. It should sweep prior high/low lookbacks, VP or source-indicator target choice, target-zone bands up to roughly `3%`, early-touch offsets, reversal/opposite-candle confirmation, trigger timeframe, and target-zone actions. Keep these as compact categorical modes; do not turn them into wide decimal ranges or independent Cartesian products.

## Split And Noise Control Rules

A medium-logic agent should use these rules before generating each file. The goal is enough breadth for HyperOpt to discover interactions, without burying the result in inactive parameters or unrelated branch noise.

### Keep Concepts Together When

Keep modes in one file when they share most of these:

1. Same source entry foundation.
2. Same primary trigger family.
3. Same target provider family.
4. Same invalidation provider family.
5. Same management sequence, for example `target touch -> partial -> stop shift -> runner`.
6. Same implementation surface, for example all logic lives in custom exit, all logic lives in custom stoploss, or all logic uses the same partial-adjustment path.
7. Same trader-readable question.

Examples that can stay together:

1. `VP target touch: full exit vs partial vs partial_then_trail`.
2. `TLV2 retest failure: tighten vs breakeven vs partial_if_profit`.
3. `Pattern projection: fixed final target vs projection target vs projection partial`.
4. `Guard flip in profit: ignore vs reduce vs tighten vs full exit after target`.

### Split Files When

Split into separate branch files when any of these are true:

1. The file would exceed roughly `5-10` active HyperOpt parameters in normal modes.
2. More than about one-third of parameters are inactive in most branch modes.
3. A branch mode makes several other parameters irrelevant.
4. Two concepts use different target providers and different invalidation providers.
5. The file mixes unrelated questions, such as VP target management plus time stagnation plus trailing-only.
6. One concept needs partial-exit position adjustment while another only needs a custom exit signal.
7. Long and short logic need materially different source columns that cannot be cleanly mirrored.
8. A source-specific branch and a generic baseline branch are only connected by the source filename.
9. The winning result could not be summarized in one trader-readable sentence.
10. Adding another branch would require new enable flags with hidden thresholds underneath.

### Cross-Product Containment

Do not create noisy Cartesian products of independent choices.

Use this rule of thumb:

1. One large categorical mode with `20-50` named coherent plans can be acceptable.
2. Several independent `10-20` value categories multiplied together is usually too noisy.
3. If choices only make sense together, combine them into a named plan such as `vp_poc_partial33_then_be` instead of separate target, size, stop, and action toggles.
4. If the branch requires target source, action, stop, partial, trail, guard response, and PnL overlay all at once, split it unless most are compact combined plans.
5. Do not add a parameter merely because it is easy to expose. Add it only if it changes a meaningful exit theory.

### Minimum Useful Breadth

Do not split so aggressively that Sieve becomes manual backtesting again.

Each generated branch file should normally have:

1. At least one branch-mode parameter.
2. At least one action-mode parameter.
3. At least one target/stop/profit-state parameter when relevant.
4. Enough named modes that the search is materially broader than `10-15` simple backtests.
5. A compact null/off/ignore state where a concept may reasonably have no value.

### Per-File Preflight Comment

Each generated strategy file should include a short module-level comment or metadata block listing:

1. Source strategy.
2. Exit family.
3. Primary trigger.
4. Primary guard.
5. Target provider.
6. Invalidation provider.
7. Active HyperOpt parameters.
8. Branch-local or potentially inactive parameters.
9. Why this file was not split further.

This is for reviewability, not a separate manifest. Do not create new metadata files for this.

## Parameter Design Standard

Use categorical parameters first. Use small categorical numeric sets second. Use multipliers only when a base value has a clear meaning. Avoid continuous decimal sweeps.

### Profit Gates

Use categories such as:

1. `any_profit`
2. `profit_0_5pct`
3. `profit_1pct`
4. `profit_1_5pct`
5. `profit_2pct`
6. `profit_3pct`
7. `profit_1R`
8. `profit_1_5R`
9. `profit_2R`

The `any_profit` or `ignore` state is the null-like state. It lets HyperOpt make the profit gate low-impact without hiding it behind an enable.

### Fixed TP/SL

Use categorical target pairs, not wide decimals:

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

If later fine-tuning is needed, create a second-stage branch with small local categories, not a wide range.

### Partial Exits

Use categorical partial plans:

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

Use `none` only when the branch file is explicitly testing partial versus no partial. Do not put `none` in a branch where partial exit is the whole point unless the comparison is intentional.

### Breakeven And Stop Shift

Use mode categories:

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

Use stop target categories:

1. `entry`
2. `entry_plus_fee`
3. `lock_0_25pct`
4. `lock_0_5pct`
5. `prior_candle_low_high`
6. `recent_swing`
7. `entry_level_retest`
8. `htf_structure`

Avoid separate `enable_breakeven` plus `breakeven_value` unless the branch requires it.

### Trailing

Use named trail plans:

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

Do not combine every trail plan with every branch. Put percent trails, structure trails, and indicator-activated trails into interpretable branch files.

### Target Distance Gates

Use categories:

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

Distance gates are especially important for indicator targets. A target that is too close may only justify stop tightening; a target that is far away may need partials or trailing before final target.

### Action Modes

Use a single action-mode parameter where possible:

1. `hold`
2. `full_exit`
3. `partial_25`
4. `partial_33`
5. `partial_50`
6. `partial_66`
7. `breakeven`
8. `lock_profit`
9. `tighten_to_structure`
10. `activate_trail`
11. `partial_then_be`
12. `partial_then_trail`
13. `partial_then_next_target`

The `hold` or `ignore` state is the null state when the branch compares acting versus not acting.

## Trade-State Model Required In Code

Every generated branch must reconstruct enough state to decide action:

1. `entry_side`: long or short.
2. `current_profit`: current trade profit.
3. `profit_bucket`: loss, flat, small profit, material profit, strong profit.
4. `entry_family`: VP, TLV2, pivot, BOS/CHoCH, pattern, crash, MTF, prior level, liquidity.
5. `target_source`: fixed, VP, TLV2, pivot, prior high/low, pattern projection, HTF level.
6. `nearest_target_distance`: whether the nearest target is too close, useful, or far.
7. `target_touched`: whether price has reached the chosen level.
8. `guard_state`: aligned, neutral, weakening, opposite.
9. `trigger_state`: intact, retest failed, invalidated, opposite structure.
10. `partial_state`: no partial, first partial done, second partial done.
11. `time_state`: early, normal, stale, very stale.

The action should be a function of state, not a raw signal flip.

Examples:

1. Guard flips while trade is losing: usually hold until structural stop, tighten to invalidation, or fixed stop; do not automatically full exit.
2. Guard flips while trade is up `1%`: test partial or breakeven.
3. Guard flips while trade is up `2%+`: test partial plus stop lock, tight structure stop, or full exit.
4. Indicator target touched while trade is up but momentum still aligned: partial or trail, not always full exit.
5. Indicator target touched and guard flips opposite: partial plus breakeven, tighten, or full exit depending on branch mode.
6. Trigger invalidates before profit: use structural stop or fixed stop.
7. Trigger invalidates after first partial: tighten or trail remainder.

## Trigger, Guard, And Position-State Interaction Model

Every regenerated file must know which evidence opened the trade and which evidence merely supported it. This is not cosmetic. Trigger failure and guard weakness should not receive the same exit action.

Use these roles:

1. `primary_trigger`: the indicator or rule family that actually fired the entry.
2. `primary_guard`: the strongest non-trigger condition that made the entry acceptable.
3. `secondary_guard`: optional supporting evidence that can weaken, align, or oppose after entry.
4. `target_provider`: the indicator family that supplies the next obstacle or profit objective.
5. `invalidation_provider`: the indicator family that defines where the entry idea is broken.

For each branch, encode the role explicitly in metadata, entry tags, module comments, or named mode values. Do not let a later agent infer it from filename fragments alone.

### Trigger Versus Guard Response

Use a trigger/guard response parameter with named categorical modes such as:

1. `ignore_guard_unless_trigger_fails`
2. `guard_flip_tighten_only`
3. `guard_flip_reduce_if_profit`
4. `guard_flip_full_exit_after_target`
5. `trigger_fail_structure_stop`
6. `trigger_fail_lock_if_profit`
7. `trigger_fail_reduce_then_trail`
8. `trigger_fail_full_exit_if_guard_opposes`
9. `target_touch_then_guard_decides`
10. `target_touch_then_trigger_decides`

Core interaction rules:

1. Trigger intact and guard aligned: hold, trail, or target-manage. Do not force an exit.
2. Trigger intact and guard weak: test tighten, partial, or ignore based on profit bucket.
3. Trigger intact and guard opposite: test reduce or tighten first; full exit should normally require profit, target touch, multiple opposing guards, or source-specific invalidation.
4. Trigger invalid and guard aligned: test structural stop or tightened stop; the higher-timeframe guard may justify giving the trade one retest.
5. Trigger invalid and guard weak/opposite: this is the strongest de-risk state. Test full exit, partial plus lock, or tight structure stop based on profit bucket.
6. Target touched and trigger intact: test partial, stop shift, trail to next target, or full exit.
7. Target touched and guard flips opposite: test partial/full exit/lock-profit modes, not blind hold.

### PnL Overlay

Exit behaviour must account for current open-trade profit or loss. The same opposing signal can mean different things when the trade is down `1%`, flat, up `1%`, or up `4%`.

Add an optional categorical overlay where useful:

1. `pnl_overlay_mode = off`
2. `pnl_overlay_mode = loss_protect`
3. `pnl_overlay_mode = profit_lock`
4. `pnl_overlay_mode = profit_scale`
5. `pnl_overlay_mode = asymmetric_loss_hold_profit_reduce`
6. `pnl_overlay_mode = asymmetric_loss_cut_profit_trail`

Use profit buckets such as:

1. `loss_beyond_half_stop`
2. `small_loss`
3. `flat`
4. `profit_0_5`
5. `profit_1`
6. `profit_2`
7. `profit_3_plus`
8. `profit_after_first_partial`
9. `profit_after_target_touch`

Valid overlay actions:

1. `hold_to_original_stop`
2. `tighten_to_invalidation`
3. `tighten_to_entry_level`
4. `move_to_breakeven`
5. `lock_0_25`
6. `lock_0_5`
7. `partial_25`
8. `partial_33`
9. `partial_50`
10. `activate_trail`
11. `full_exit`

The overlay must be optional and compact. Prefer one named `pnl_overlay_mode` plus a few categorical bucket/action plans over many hidden enable flags. If the branch does not need PnL logic, use `off` as a real null state.

### Long/Short Mirroring

Every indicator-specific branch must mirror long and short logic explicitly.

Long-side rules:

1. Targets usually sit above entry: prior pivot high, TLV2 resistance, VP POC/VAH, pattern upper boundary, bullish pattern target.
2. Invalidation usually sits below entry: failed reclaim, broken support, failed retest, close back inside pattern, bearish BOS/CHoCH.
3. Opposing evidence is bearish structure, bearish pattern confirmation, VP short context, loss of support, or rejection from target.

Short-side rules:

1. Targets usually sit below entry: prior pivot low, TLV2 support, VP POC/VAL, pattern lower boundary, bearish pattern target.
2. Invalidation usually sits above entry: failed breakdown, reclaimed resistance, close back inside pattern, bullish BOS/CHoCH.
3. Opposing evidence is bullish structure, bullish pattern confirmation, VP long context, reclaim of support/resistance, or bounce from target.

Do not reuse long comparisons for shorts with only the tag changed. Directional profit, target touch, stop movement, and invalidation comparisons must all be side-aware.

## Critical Implementation Guardrails

These guardrails are mandatory because exit systems can look profitable when they accidentally use unavailable levels, future-confirmed pivots, or branch parameters that never affected trades.

### No Lookahead Or Retroactive Levels

1. Use only values available at the candle where the exit decision is made.
2. For pivots, prefer confirmed/available columns such as `pivot_high_confirmed`, `pivot_low_confirmed`, `pivot_high_available_index`, and `pivot_low_available_index` when deciding whether a pivot level is usable.
3. Do not use a future-confirmed pivot, pattern target, or line projection as if it was known at trade entry unless the indicator contract makes that availability explicit.
4. If a target level is discovered after entry, the branch must treat it as a later target-management update, not as the original entry target.
5. If target availability is ambiguous, do not generate that target-specific branch. Use fixed or already-known structural targets instead.
6. Do not scan future candles to decide whether a target was touched. The code must act only from current analyzed dataframe state, current rate, current profit, and trade state available to Freqtrade at that moment.

### Strategy API Reality Check

1. Full exits, partial exits, stop movement, and trailing may require different Freqtrade methods. Use the existing local strategy patterns and method signatures before generating code.
2. If a branch claims to test partial exits, it must actually use a valid position-adjustment path, not just set a sell signal that exits the whole trade.
3. If a branch claims to test stop movement or breakeven, it must actually update stop behaviour in a way Freqtrade will use, not just compute an unused value.
4. If a branch claims to test trailing, verify whether it is fixed trailing, structure trailing, indicator-activated trailing, or stoploss-based trailing. Name the mode accordingly.
5. Do not use mutable global dictionaries or cross-pair caches unless an existing accepted local pattern already does so safely. Prefer reconstructing state from trade fields, entry tags, current dataframe row, and deterministic columns.
6. Partial-exit branches must be idempotent. Use a concrete local pattern such as `trade.nr_of_successful_exits`, `trade.has_open_orders`, filled-order/order-tag checks, or an equivalent accepted guard so the same partial action cannot fire repeatedly on every `adjust_trade_position()` call.
7. Full-exit branches may use `custom_exit()` or exit signals, but they must be tagged as full-position exits and must not be described as partial exits.
8. Stop-shift and breakeven branches must use a valid custom-stoploss path when they claim to move the stop. Built-in trailing and custom-stoploss trailing should not both be active unless the branch explicitly tests and documents that interaction.

### Branch Result Integrity

1. Every HyperOpt parameter in the file must affect at least one reachable branch mode.
2. If a parameter is only relevant after a rare branch, combine it into that branch's named plan or split the file.
3. A branch mode should not silently fall back to fixed `3/3` because an indicator column is missing. Missing required columns should be a generation-time reason to omit the branch.
4. Do not catch broad exceptions around exit logic and continue with a default hold/exit behaviour. Fix the column/method/state mapping instead.
5. Do not add generic defensive fallback targets such as "nearest column containing high" or keyword-discovered levels. Source-family target mapping must be explicit.

## Project Indicator Exit Playbooks

Use these playbooks when generating indicator-specific and mixed indicator branches. The column names below are current strategy-facing indicator contracts; use exact columns when present in the source strategy or its informative timeframe merges. Do not create new indicator columns, reshape indicator modules, or use parked work-in-progress indicators.

### Volume Profile

Key columns include `vp_entry_trigger_long`, `vp_entry_trigger_short`, `vp_node_entry_long`, `vp_node_entry_short`, `vp_node_hold_long`, `vp_node_hold_short`, `vp_node_exit_long`, `vp_node_exit_short`, `vp_market_context`, `vp_score_long`, `vp_score_short`, `vp_score_abs`, `vp_state`, `vp_prior_poc`, `vp_prior_vah`, and `vp_prior_val`.

Long scenarios:

1. VP long trigger from value reclaim: entry expects movement from VAL/POC toward POC/VAH. Test partial at POC, stop shift at POC acceptance, and full/partial at VAH.
2. VP long trigger from VAH breakout: entry expects acceptance above value. Losing VAH is trigger damage; losing POC is stronger invalidation. Test hold/trail if price accepts above VAH.
3. VP node long trigger: if `vp_node_hold_long` remains true, do not exit only because target is touched. Test partial plus trail to next node or fixed runner.
4. VP long trade reaches a bearish node or `vp_node_exit_long`: if profitable, test partial/full exit/lock. If losing, compare structural stop versus fixed stop.

Short scenarios:

1. VP short trigger from value rejection: entry expects movement from VAH/POC toward POC/VAL. Test partial at POC and full/partial at VAL.
2. VP short trigger from VAL breakdown: entry expects acceptance below value. Reclaiming VAL is trigger damage; reclaiming POC is stronger invalidation.
3. VP node short trigger: if `vp_node_hold_short` remains true, test trail/hold through the node rather than immediate full exit.
4. VP short trade reaches a bullish node or `vp_node_exit_short`: if profitable, test partial/full exit/lock. If losing, compare structural stop versus fixed stop.

Trigger/guard combinations:

1. VP trigger plus BOS/CHoCH guard: if VP target is hit and structure remains aligned, test partial then trail. If structure flips opposite at target, test partial/full exit/lock.
2. BOS/CHoCH trigger plus VP guard: if structure remains intact but VP context weakens, test guard-flip tighten or reduce only after profit. If VP flips opposite and trigger also invalidates, test full exit.
3. Pattern trigger plus VP guard: if a pattern projection has not hit but VP target is close, test VP as first partial and pattern as final target.
4. TLV2 trigger plus VP guard: if price breaks a trendline into VAH/VAL/POC, test first obstacle partial before final trendline/pattern target.

### Trendlines / TLV2

Key columns include `tlv2_support_line_rank0`, `tlv2_support_score_rank0`, `tlv2_support_distance_atr_rank0`, `tlv2_resistance_line_rank0`, `tlv2_resistance_score_rank0`, and `tlv2_resistance_distance_atr_rank0`.

Long scenarios:

1. Support-bounce trigger: TLV2 support is the invalidation provider. Test stop to support, stop to recent swing below support, and partial at resistance.
2. Resistance-break trigger: broken resistance becomes the retest/invalidation line. Test retest-failure tighten, breakeven after successful retest, and next resistance target.
3. Far resistance target: use target-distance categories. If too far, test fixed first partial then trail to TLV2 resistance.
4. Close resistance target: test partial/stop shift, not blind full exit.

Short scenarios:

1. Resistance-rejection trigger: TLV2 resistance is the invalidation provider. Test stop to resistance or recent swing above resistance, and partial at support.
2. Support-break trigger: broken support becomes the retest/invalidation line. Test retest-failure tighten, breakeven after successful retest, and next support target.
3. Far support target: test fixed first partial then trail to TLV2 support.
4. Close support target: test partial/stop shift.

Trigger/guard combinations:

1. TLV2 trigger plus VP guard: if TLV2 break enters VP congestion, VP should often control first partial while TLV2 controls invalidation.
2. TLV2 trigger plus pattern guard: if pattern confirms same direction after the TLV2 entry, test hold/trail rather than early fixed TP.
3. Pattern trigger plus TLV2 guard: TLV2 level can be target/invalidation even when pattern supplied the entry.
4. BOS/CHoCH trigger plus TLV2 guard: structure trigger can stay intact while TLV2 retest fails; test tighten/partial based on PnL before full exit.

### Pivot Foundation

Key columns include `pivot_high`, `pivot_low`, `pivot_high_index`, `pivot_low_index`, `pivot_high_available_index`, `pivot_low_available_index`, `pivot_high_prominence`, `pivot_low_prominence`, `pivot_high_prominence_pct`, `pivot_low_prominence_pct`, `pivot_high_score`, `pivot_low_score`, `pivot_high_confirmed`, and `pivot_low_confirmed`.

Long scenarios:

1. Next confirmed pivot high is a natural first target or partial level.
2. Prior pivot low is a natural invalidation or stop reference.
3. High prominence pivot targets can justify full exit or larger partial; low prominence targets should often be partial/ignore.
4. If price breaks above a pivot high and accepts, test trail to next pivot rather than forced full exit.

Short scenarios:

1. Next confirmed pivot low is a natural first target or partial level.
2. Prior pivot high is a natural invalidation or stop reference.
3. High prominence pivot targets can justify full exit or larger partial; low prominence targets should often be partial/ignore.
4. If price breaks below a pivot low and accepts, test trail to next pivot.

Trigger/guard combinations:

1. Pivot target overlay can be used with any trigger family as first obstacle, but it must respect target distance and prominence.
2. VP trigger plus pivot target: if POC/VAH/VAL and pivot target cluster, test stronger partial/full actions.
3. Pattern trigger plus pivot target: if projection and pivot align, test projection full exit versus pivot partial then trail.
4. BOS/CHoCH trigger plus pivot invalidation: opposite structure through a pivot should be treated as stronger trigger damage.

### BOS / CHoCH Market Structure

Key columns include `ms_bos_to_bull`, `ms_bos_to_bear`, `ms_choch_to_bull`, `ms_choch_to_bear`, `ms_structure_event`, and `ms_state`.

Long scenarios:

1. Bullish BOS trigger: target prior/next pivot high, VP/TLV2 obstacle, or HTF level. Bearish CHoCH is warning; bearish BOS is stronger invalidation.
2. Bullish CHoCH trigger: entry is earlier and less confirmed than BOS. Test tighter stop shift and faster partials.
3. Long structure remains bullish after first target: test trail by swing/pivot instead of fixed full exit.
4. Structure flips bearish while trade is profitable: test partial plus lock, structure trail, or full exit after target.

Short scenarios:

1. Bearish BOS trigger: target prior/next pivot low, VP/TLV2 obstacle, or HTF level. Bullish CHoCH is warning; bullish BOS is stronger invalidation.
2. Bearish CHoCH trigger: test tighter stop shift and faster partials.
3. Short structure remains bearish after first target: test trail by swing/pivot.
4. Structure flips bullish while trade is profitable: test partial plus lock, structure trail, or full exit after target.

Trigger/guard combinations:

1. BOS/CHoCH as trigger plus VP/TLV2 as target provider: target touch should usually test partial/trail before full exit.
2. VP/TLV2/pattern as trigger plus BOS/CHoCH as guard: opposite CHoCH is guard weakness; opposite BOS is stronger and can unlock full-exit modes.
3. BOS/CHoCH as guard with pattern trigger: if the pattern fails but structure remains aligned, test tighten/one-retest modes.
4. BOS/CHoCH as trigger with pattern guard: opposite rare pattern while profitable should test reduce/tighten before full exit.

### Geometry V2 Patterns

Key columns include `pg2_<family>_pattern_present`, `pg2_<family>_indicator_score`, `pg2_<family>_direction`, `pg2_<family>_width_atr`, `pg2_<family>_squeeze_active`, `pg2_<family>_upper`, and `pg2_<family>_lower`. Families include `triangle`, `wedge`, `compression`, `rectangle`, `ascending_channel`, and `descending_channel`.

Long scenarios:

1. Breakout above `upper`: the upper line becomes retest/invalidation; target can be projection, next pivot, TLV2 resistance, or VP target.
2. Compression/squeeze release upward: test early partial at first obstacle and trail if expansion continues.
3. Channel continuation long: lower/channel support can be invalidation; upper/channel resistance can be partial target.
4. Failed breakout back inside the pattern is trigger damage and should test stop shift or exit based on profit.

Short scenarios:

1. Breakdown below `lower`: the lower line becomes retest/invalidation; target can be projection, next pivot, TLV2 support, or VP target.
2. Compression/squeeze release downward: test early partial at first obstacle and trail if expansion continues.
3. Channel continuation short: upper/channel resistance can be invalidation; lower/channel support can be partial target.
4. Failed breakdown back inside the pattern is trigger damage.

Trigger/guard combinations:

1. Geometry trigger plus VP guard: VP obstacle controls first partial; pattern boundary controls invalidation.
2. Geometry trigger plus BOS/CHoCH guard: same-direction structure after breakout supports trail/hold. Opposite structure after failed breakout supports full exit or lock.
3. VP/TLV2 trigger plus geometry guard: a forming squeeze can justify waiting for expansion, but opposite breakout should trigger de-risk.
4. Pattern score weakens without boundary failure: test guard-flip tighten/partial, not automatic full exit.

### Reversal Patterns

Key columns include `pat_double_top_*`, `pat_double_bottom_*`, `pat_head_shoulders_*`, and `pat_inverse_head_shoulders_*` suffixes such as `_pattern_present`, `_pattern_confirmed`, `_indicator_score`, `_confirmation_level`, `_target_level`, and pivot point indexes.

Long scenarios:

1. Double bottom or inverse head-and-shoulders trigger: confirmation/neckline is the retest line; target level is the natural structural target.
2. Long trade sees double top or head-and-shoulders confirmed against it: treat as opposing rare-pattern evidence. Use PnL overlay to choose reduce/tighten/full exit.
3. Long reaches bullish reversal target while BOS/VP still aligns: test partial then trail, not mandatory full exit.
4. Bullish reversal fails back below confirmation: trigger damage; test breakeven/lock if profitable or structural stop if losing.

Short scenarios:

1. Double top or head-and-shoulders trigger: confirmation/neckline is the retest line; target level is the natural structural target.
2. Short trade sees double bottom or inverse head-and-shoulders confirmed against it: use PnL overlay to choose reduce/tighten/full exit.
3. Short reaches bearish reversal target while BOS/VP still aligns: test partial then trail.
4. Bearish reversal fails back above confirmation: trigger damage.

Trigger/guard combinations:

1. Reversal trigger plus BOS/CHoCH guard: if structure confirms after pattern, test slower exit and trail. If structure refuses confirmation, test earlier partials.
2. Reversal trigger plus VP target: VP POC/VAH/VAL can be first partial before pattern target.
3. VP/TLV2 trigger plus reversal guard: opposite reversal pattern near target is stronger than generic guard weakness and should unlock de-risk modes.
4. Pattern target hit plus opposite guard flip: compare full exit versus partial plus stop lock.

### Continuation Patterns

Key columns include `pat_flag_*` and `pat_pennant_*` suffixes such as `_setup_present`, `_pattern_present`, `_pattern_confirmed`, `_direction`, `_indicator_score`, `_upper`, `_lower`, `_confirmation_level`, `_setup_invalidated`, and `_invalidation_level`.

Long scenarios:

1. Bull flag or bullish pennant confirmation: confirmation level is the retest line; invalidation level is the stop reference.
2. Breakout reaches first obstacle: test partial and stop to confirmation/invalidation rather than full exit only.
3. Setup invalidates while profitable: test partial plus lock or trail. While losing: compare invalidation stop versus fixed stop.
4. Same-direction continuation remains active after first target: test runner/trail.

Short scenarios:

1. Bear flag or bearish pennant confirmation: confirmation level is the retest line; invalidation level is the stop reference.
2. Breakdown reaches first obstacle: test partial and stop to confirmation/invalidation.
3. Setup invalidates while profitable: test partial plus lock or trail. While losing: compare invalidation stop versus fixed stop.
4. Same-direction continuation remains active after first target: test runner/trail.

Trigger/guard combinations:

1. Continuation trigger plus VP/TLV2 target: structural obstacle is first partial; pattern continuation controls runner.
2. Continuation trigger plus BOS/CHoCH guard: same-direction BOS after breakout supports holding remainder.
3. BOS/VP trigger plus continuation guard: a confirmed continuation can delay fixed TP and move to trail/target management.
4. Opposite continuation pattern while profitable should test reduce/tighten; full exit usually requires target touch, trigger failure, or multiple opposing guards.

### Multi-Peak Patterns

Key columns include `pat_triple_top_*` and `pat_triple_bottom_*` suffixes such as `_pattern_present`, `_pattern_confirmed`, `_indicator_score`, `_confirmation_level`, `_target_level`, and peak indexes.

Long scenarios:

1. Triple bottom trigger: confirmation is retest line; target level is structural target.
2. Triple top confirmed against a long: treat as strong opposing pattern, especially near VP/TLV2/pivot resistance.
3. Failed triple top breakout that resolves upward can become hold/trail evidence if the long trigger remains intact.
4. Target hit with same-direction guards aligned: test partial/trail rather than full exit only.

Short scenarios:

1. Triple top trigger: confirmation is retest line; target level is structural target.
2. Triple bottom confirmed against a short: treat as strong opposing pattern, especially near VP/TLV2/pivot support.
3. Failed triple bottom breakdown that resolves downward can become hold/trail evidence if the short trigger remains intact.
4. Target hit with same-direction guards aligned: test partial/trail.

Trigger/guard combinations:

1. Multi-peak trigger plus pivot target: peak structure and pivot target often overlap; test cluster-strength actions.
2. VP/TLV2 trigger plus multi-peak guard: a confirmed opposing triple top/bottom near a target should unlock profit-protect modes.
3. BOS/CHoCH trigger plus multi-peak guard: opposite multi-peak without structure flip is warning; with opposite BOS it is stronger invalidation.
4. Multi-peak trigger plus VP guard: value-area acceptance through confirmation can justify holding for pattern target.

### Wolfe Waves

Key columns include `pww_bearish_pattern_present`, `pww_bullish_pattern_present`, `pww_bearish_pattern_confirmed`, `pww_bullish_pattern_confirmed`, `pww_bearish_indicator_score`, `pww_bullish_indicator_score`, `pww_bearish_confirmation_level`, `pww_bullish_confirmation_level`, `pww_bearish_target_level`, and `pww_bullish_target_level`.

Long scenarios:

1. Bullish Wolfe trigger: confirmation level is retest/invalidation; bullish target level is the structural objective.
2. Bearish Wolfe appears against a long: treat as opposing rare-pattern evidence. Use PnL overlay and target proximity.
3. Bullish Wolfe target hit while VP/BOS remains aligned: test partial plus trail.
4. Bullish Wolfe fails below confirmation: trigger damage; test lock if profitable or structure stop if losing.

Short scenarios:

1. Bearish Wolfe trigger: confirmation level is retest/invalidation; bearish target level is the structural objective.
2. Bullish Wolfe appears against a short: use PnL overlay and target proximity.
3. Bearish Wolfe target hit while VP/BOS remains aligned: test partial plus trail.
4. Bearish Wolfe fails above confirmation: trigger damage.

Trigger/guard combinations:

1. Wolfe trigger plus VP/TLV2 target: first obstacle partial, Wolfe target final, or Wolfe target plus trail.
2. VP/TLV2 trigger plus Wolfe guard: same-direction Wolfe can justify holding through fixed TP; opposite Wolfe can justify de-risk only after profit or target proximity.
3. BOS/CHoCH trigger plus Wolfe guard: opposite Wolfe plus opposite CHoCH/BOS is stronger than either alone.
4. Pattern target hit plus guard still aligned: test runner rather than only full exit.

## Source Family Plans

### Volume Profile Entries

Applies to source names or entry logic involving VP, POC, VAH, VAL, HVN, LVN, value area, node, or fast traverse.

Target sources:

1. Nearest POC.
2. VAH for long resistance / VAL for short support.
3. VAL reclaim for long entries after value-area failure.
4. HVN as congestion target.
5. LVN as traverse continuation target.
6. Prior VP node from entry trigger.

Invalidation sources:

1. Price falls back inside value after breakout long.
2. Price reclaims value after breakdown short.
3. VP score flips against trade.
4. Context score flips from bull to bear or bear to bull.
5. Fast traverse stalls before next node.

Branch contents:

1. `vp_full_vs_fixed`: target mode `fixed_3_3`, `poc_full`, `vah_val_full`, `hvn_lvn_full`, `next_node_full`.
2. `vp_partial_stop`: action mode `partial_33_then_be`, `partial_50_then_be`, `partial_33_then_trail`, `target_touch_tighten`.
3. `vp_acceptance_failure`: if breakout loses value acceptance, action mode `tighten`, `partial_if_profit_1pct`, `full_exit_if_profit_2pct`, `hold_to_fixed_stop`.
4. `vp_context_flip`: if VP context flips opposite, action mode depends on profit gate and partial state.

Important interactions:

1. A close target should not force a full exit; test partial/tighten.
2. A far LVN/next-node target should allow early partial plus trail.
3. If entry was a VP breakout, losing the breakout level is stronger invalidation than generic context score drift.
4. If price reaches POC then rejects, test stop lock; if price accepts through POC, hold/trail to VAH/VAL.

### TLV2 / Support-Resistance Entries

Applies to TLV2, trendline, support, resistance, res break, sup hold, res reject, sup break.

Target sources:

1. Next TLV2 resistance/support.
2. Entry line retest.
3. Prior local swing.
4. Higher-timeframe TLV2 level when available.

Invalidation sources:

1. Long breakout closes back below broken resistance.
2. Short breakdown closes back above broken support.
3. Retest fails after breakout.
4. TLV2 slope/level flips against trade if column exists.

Branch contents:

1. `tlv2_next_level`: fixed versus next TLV2 target.
2. `tlv2_retest_failure`: action at failed retest with profit gates.
3. `tlv2_partial_remainder`: partial at next level, remainder fixed/trail/next level.
4. `tlv2_stop_shift`: stop to entry line, recent swing, or HTF structure after level touch.

Important interactions:

1. If long is above broken resistance and retest holds, do not exit just because a guard weakens; test trail/hold.
2. If price loses the broken level while in profit, test partial plus lock.
3. If price loses the broken level while not in profit, test fixed stop versus structure stop.
4. If next TLV2 level is very close, test breakeven or partial rather than full exit.

### BOS / CHoCH Market Structure Entries

Applies to BOS, CHoCH, HH/HL/LH/LL, market-structure state, pivot-structure entries.

Target sources:

1. Prior pivot high for longs.
2. Prior pivot low for shorts.
3. Confirmed swing high/low.
4. Next structure level after BOS.
5. Higher-timeframe structure if available.

Invalidation sources:

1. Opposite BOS.
2. Opposite CHoCH.
3. Loss of HH/HL for long or LH/LL for short.
4. Close through invalidating swing.
5. Market-structure state flips.

Branch contents:

1. `structure_target`: fixed versus pivot target versus next structure.
2. `opposite_structure_response`: full exit, partial, breakeven, or tighten based on profit gate.
3. `swing_stop_shift`: stop to latest swing after favorable move.
4. `structure_trail`: trail by swing lookback after partial or target touch.

Important interactions:

1. Opposite CHoCH in profit is often reduce/tighten first, not automatic full exit.
2. Opposite BOS after profit may justify full exit if target already touched or guard flips also confirm.
3. Structure signal while losing should usually test structural stop, not panic exit.
4. Pivot target close to entry should be compared with fixed `3/3`, not treated as always superior.

### Prior High / Low And Liquidity Entries

Applies to prior day/week/month highs/lows, equal highs/lows, liquidity sweep, failed high, failed low.

Target sources:

1. Prior high/low extension.
2. Next prior period level.
3. Sweep origin level.
4. Failed-break reclaim level.
5. VP/TLV2 nearby obstacle if available.

Invalidation sources:

1. Breakout fails back below prior high for long.
2. Breakdown fails back above prior low for short.
3. Liquidity sweep reverses through the sweep level.
4. Equal-high/low signal flips.

Branch contents:

1. `prior_level_fixed_vs_next`: fixed target versus next prior level.
2. `sweep_snapback_management`: partial after snapback, stop lock, trail.
3. `failed_break_response`: hold, tighten, partial, or full exit depending on profit.
4. `prior_level_htf_interaction`: 1h execution hitting 4h/1d prior level.

Important interactions:

1. A sweep entry that snaps back quickly should test early partials.
2. If price re-sweeps the level against the trade while profitable, tighten or partial.
3. If price accepts beyond the level, trail rather than fixed early exit may be better.
4. If target is a higher-timeframe prior level, test partial/BE before full exit.

### Pattern Entries

Fallback only. Apply this section only when a pattern source cannot be classified into Geometry V2, reversal, continuation, multi-peak, or Wolfe using the detailed project indicator playbooks above. Known pattern families must use their family-specific columns, confirmation logic, invalidation logic, and target logic rather than this generic fallback.

Applies to unclassified wedge, triangle, rectangle, channel, flag, pennant, head-and-shoulders, inverse head-and-shoulders, double/triple top/bottom style sources only when no more specific project indicator contract is available.

Target sources:

1. Pattern projection.
2. Pattern upper/lower boundary.
3. Breakout/retest line.
4. Neckline for H&S / inverse H&S / double/triple structures.
5. Prior pivot beyond pattern.

Invalidation sources:

1. Breakout falls back inside pattern.
2. Breakdown reclaims pattern.
3. Retest fails.
4. Opposite pattern state appears.
5. Pattern score/confirmation disappears if the indicator exposes it.

Branch contents:

1. `pattern_projection_vs_fixed`: fixed target versus projection.
2. `pattern_boundary_retest`: hold if retest confirms, tighten/exit if retest fails.
3. `pattern_partial_remainder`: partial at boundary/projection, trail or next pivot.
4. `opposite_pattern_reduce`: opposite pattern while in profit triggers reduce/tighten/full exit comparison.

Important interactions:

1. Pattern projection should be ignored if less than `1R` unless the branch is testing quick partials.
2. Failed breakout back inside pattern is stronger invalidation than generic guard weakness.
3. Same-direction pattern continuation after entry may support hold/trail, not exit.
4. Opposite rare pattern in profit should test partial/reduce before full exit.

### Crash / Capitulation Entries

Applies to crash, flush, capitulation, reclaim, snapback, forced liquidation style entries. Do not add new RSI logic unless the source strategy already uses it and the user accepts that source.

Target sources:

1. Snapback percentage levels.
2. Reclaim level.
3. Prior breakdown level.
4. Local pivot before the crash impulse.
5. VP/TLV2 obstacle if available.

Invalidation sources:

1. New low after long reclaim.
2. Failed reclaim.
3. Volume/range pressure resumes downward.
4. No snapback after N candles.

Branch contents:

1. `crash_snapback_partials`: partial at `2%`, `3%`, `4%`, then trail.
2. `crash_reclaim_stop_shift`: move stop to reclaim line or entry after confirmation.
3. `crash_failed_reclaim`: full exit versus tighten versus hold to fixed stop.
4. `crash_stagnation`: time stop if bounce does not start.

Important interactions:

1. Crash entries often need earlier partials than normal continuation entries.
2. If price reclaims and holds, breakeven/lock should activate before far target.
3. If flush resumes while profitable, tighten or partial may beat full exit.
4. If flush resumes while losing, structural stop should be compared with fixed stop.

### MTF Entries

Applies to files with explicit 1h/4h/1d/3d hierarchy.

Target sources:

1. Higher-timeframe pivot range.
2. Higher-timeframe VP level.
3. Higher-timeframe TLV2 level.
4. Higher-timeframe pattern boundary.
5. Lower-timeframe execution trigger level.

Invalidation sources:

1. HTF context flips against trade.
2. LTF trigger invalidates while HTF still supports.
3. HTF target rejects.
4. HTF level accepts through in favor of trade.

Branch contents:

1. `htf_target_touch`: full exit, partial, BE, tighten, or trail at HTF target.
2. `ltf_trigger_failure`: tighten/exit if LTF trigger fails, with different action if HTF still supports.
3. `htf_reject_vs_accept`: if HTF target rejects, reduce/exit; if accepts, trail/hold.
4. `timeframe_head_to_head`: 1h target management versus 4h/1d target management.

Important interactions:

1. A 1h entry reaching a 1d level should rarely be a blind full exit only. Test partial, BE, structure stop, and trail.
2. If LTF invalidates but HTF remains aligned and trade is not deeply negative, test tighten rather than full exit.
3. If HTF flips against trade while profitable, reduce or lock profit should be tested.
4. If HTF accepts through target, fixed TP may cap winners; test trail remainder.

## Mixed Branch Design

Mixed branches must test ordered action sequences, not unrelated toggles.

### Target Touch Then Action

When target is touched:

1. If profit gate fails: ignore or tighten only.
2. If target is close and profit is small: partial or breakeven.
3. If target is useful and guard aligned: partial then trail or next target.
4. If target is useful and guard weakens: partial then breakeven or tighten.
5. If target is useful and guard flips strongly: partial/full exit based on profit gate.

Parameters:

1. `target_source_mode`
2. `target_distance_mode`
3. `target_action_mode`
4. `remainder_mode`
5. `profit_gate_mode`

### Guard Flip While In Trade

Guard state categories:

1. `aligned`
2. `neutral`
3. `weakening`
4. `opposite_one_source`
5. `opposite_multi_source`

Action categories:

1. `ignore_until_profit`
2. `tighten_to_entry`
3. `lock_0_25`
4. `partial_33`
5. `partial_50`
6. `full_exit_after_partial`
7. `full_exit_if_profit_2pct`
8. `trail_remainder`

Profit buckets:

1. `loss`
2. `flat`
3. `profit_0_5`
4. `profit_1`
5. `profit_2`
6. `profit_3_plus`

The branch should not say "guard bad, exit." It should compare what to do based on profit, partial state, and whether the original trigger is still intact.

### Trigger Invalidation

Trigger invalidation means the entry reason is damaged. It should be stronger than a generic guard flip.

Action logic:

1. If invalidated before any favorable move: fixed stop versus structural stop.
2. If invalidated after `1%` favorable move: breakeven or lock.
3. If invalidated after first partial: tighten/trail remainder.
4. If invalidated after target touch: full exit or trail stop.
5. If invalidated but HTF context still supports: tighten, not automatic full exit.

Parameters:

1. `invalidation_severity_mode`
2. `profit_gate_mode`
3. `invalidation_action_mode`
4. `stop_reference_mode`
5. `remainder_mode`

### Partial Plus Stop Plus Target

This branch tests layered management:

1. First target or profit gate triggers partial.
2. Stop shifts to entry, fee, lock, or structure.
3. Remainder exits at fixed target, next structure, or trail.
4. Guard flip after partial can tighten or exit.

Parameters:

1. `first_partial_plan`
2. `stop_after_partial_mode`
3. `remainder_target_mode`
4. `post_partial_guard_action`
5. `time_stagnation_mode`

## Null-State And Hidden-Parameter Discipline

Hidden branch parameters are sometimes unavoidable. The agent must minimize them.

Use these patterns:

1. Replace `enable_x` plus `x_threshold` with `x_threshold_mode = ignore / weak / normal / strict / impossible`.
2. Replace broad decimal ranges with named categorical values.
3. Put branch-specific thresholds only in files where that branch is a central concept.
4. Do not include partial-size parameters in a branch that never takes partials.
5. Do not include indicator-target params in a branch that never looks at indicator targets.
6. If one branch mode makes several parameters irrelevant, split the file or redesign the parameter as a combined categorical plan.

Acceptable combined plans:

1. `partial_33_at_2_then_be`
2. `partial_50_at_vp_then_trail`
3. `target_touch_lock_0_25`
4. `guard_flip_profit_2_full_exit`
5. `trigger_fail_after_partial_trail`

Combined plans reduce hidden parameter noise because the mode itself represents the coherent action package.

## HyperOpt Run Guidance For These Files

Use Sieve.

For branch-rich files:

1. Use materially higher epochs than narrow files.
2. Use multiple random states/seeds, typically at least `3`, and more when branch mode count is high.
3. If the runner exposes an initial random/startup candidate setting, set it above default. Do not add unsupported flags.
4. Review top several candidates, not only the best.
5. Treat one-state winners with caution unless repeated across seeds.
6. Do not infer failure from one tiny smoke or one low-epoch run.
7. Summarize winners by mode families: target source, action mode, stop mode, partial plan, profit gate.

Epoch sizing heuristic:

1. Narrow file with 4-6 categorical choices: small smoke is fine only for load verification.
2. Branch-rich file with 4 internal branch modes and 5-8 active params: use enough epochs to sample each mode repeatedly.
3. Files with target source plus action plus stop plus partial plus guard-state modes need higher epochs and multiple seeds.

## Regeneration Workflow

1. Inventory the 96 source foundations.
2. Extract source metadata: class, side, timeframe, entry tag, indicators imported, entry condition columns, guard columns, target-like columns.
3. Classify each source into one or more families: VP, TLV2, BOS/CHoCH, pivot/prior level, pattern, crash, liquidity, MTF, standard-MTF source already present.
4. Build a source-specific exit profile:
   - primary trigger family,
   - primary guard family,
   - secondary guard families,
   - target provider family,
   - invalidation provider family,
   - target sources,
   - invalidation sources,
   - guard flip sources,
   - trigger invalidation,
   - long and short mirror logic,
   - PnL overlay modes worth testing,
   - HTF/LTF distinction,
   - likely favorable path,
   - likely failure mode.
5. Map source columns to exact project indicator contracts. If a branch needs VP, TLV2, pivot, BOS/CHoCH, pattern, Wolfe, or continuation columns, prove the source or its informative merge has the required columns before generating that branch.
6. Check lookahead safety for target/invalidation columns. Confirm the branch uses only columns available at the decision candle.
7. Decide file splits with the split/noise rules before writing code. Record the split rationale in a module-level comment.
8. Generate only relevant branch files for that source.
9. Use categorical/multiplier parameters only.
10. Avoid universal parameter blocks.
11. For every source-specific branch, include at least one mode where the source indicator supplies target/invalidation and at least one mode where it interacts with a different trigger or guard family when those columns exist.
12. For every mixed branch, include the trigger/guard/PnL state model in code rather than a raw signal-flip exit.
13. Verify that each claimed full exit, partial exit, stop movement, or trailing mechanism is implemented through a valid local Freqtrade method pattern.
14. Compile all files.
15. Import all files with the controller venv.
16. Run a small isolated Sieve smoke with at least one broad branch and one mixed/source-specific branch.
17. After real batches, use approved scripts to spot-check mechanism behaviour in parsed summaries or diagnostics. Do not assign agents to manually read raw HyperOpt logs.
18. Report counts by source family and branch family.

## Acceptance Checklist

A regenerated branch set is acceptable only if:

1. No wide decimal sweeps remain for TP, SL, partial triggers, trailing distances, target gates, or stop shifts.
2. Branch files do not carry irrelevant universal sell params.
3. Indicator-specific files name actual target and invalidation sources.
4. Mixed files model profit bucket, target state, guard state, trigger state, partial state, and action state.
5. VP sources use VP-specific target/invalidation logic.
6. TLV2 sources use support/resistance/retest logic.
7. BOS/CHoCH sources use structure-specific logic.
8. Pattern sources use pattern boundary/projection/invalidation logic.
9. Crash sources use snapback/reclaim/failed-reclaim logic.
10. MTF sources use higher-timeframe target interaction logic.
11. Pivot sources use confirmed pivot, prominence, and side-aware high/low target logic.
12. Geometry V2 sources use exact family boundary, squeeze, direction, and score columns where present.
13. Reversal, continuation, multi-peak, and Wolfe branches use their own confirmation, invalidation, and target columns instead of one generic "pattern" rule.
14. Long and short comparisons are side-aware for targets, invalidations, profit, and stop movement.
15. Trigger, guard, target provider, and invalidation provider roles are explicit for each mixed branch.
16. Guard flips do not cause blind full exits without considering trigger state, target touch, PnL bucket, and partial state.
17. PnL overlay logic is optional, categorical, and compact, with a real `off` or null state.
18. Split decisions follow the split/noise rules and each file documents why it was not split further.
19. Normal branch modes stay near the `5-10` active HyperOpt parameter target unless the file uses compact combined plans.
20. No branch relies on broad Cartesian products of independent mode parameters.
21. No branch uses future-confirmed pivots, retroactive pattern targets, future candle scans, or ambiguous target availability.
22. Claimed partial exits, stop shifts, and trailing modes use valid local Freqtrade method patterns.
23. Partial-exit branches include an idempotency guard so repeated callback calls cannot repeatedly execute the same partial.
24. No broad exception fallback masks missing columns or broken state mapping.
25. HyperOpt params are mostly categorical named modes or small categorical numeric sets.
26. Multiple random states/seeds and larger epoch budgets are planned for real Sieve runs.
27. Compile/import checks pass.
28. A Sieve smoke run completes before broad testing.
29. Post-batch mechanism checks are performed with approved scripts against parsed result summaries or diagnostics, never by agents manually reading raw HyperOpt logs.
