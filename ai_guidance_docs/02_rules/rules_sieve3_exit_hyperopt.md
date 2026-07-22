---
doc_status: active
default_read: routed
owner: user+agent
purpose: Current Sieve3 V2 exit-family, Hyperopt, smoke, queue, and result rules.
do_not_use_for: Entry discovery, historical V1/V2 reconstruction, or non-Sieve Hyperopt packages.
last_rebuilt: 2026-07-22
---

# Rules - Sieve3 V2 Exit Hyperopt

## Objective

Sieve3 V2 keeps each promoted entry fixed and tests focused, trader-readable exit systems. Sieve owns Hyperopt/discovery. Strategy files must be standalone Freqtrade modules under `user_data/strategies` and use the `sieve3_V2_` prefix.

## 22 July 2026 Checkpoint

Sieve3 V2 replaced the broad original Sieve3 exit sweep because that surface had several material research-quality problems:

1. Some promoted entry parameters were not reliably carried into later branches, so copied files could silently fall back to strategy defaults instead of the Sieve2-selected entry.
2. Filenames did not always describe the exit logic that actually executed.
3. Generic shared exit templates were applied to entries without enough attention to entry-specific targets and invalidations.
4. Numeric ideas were sometimes encoded as unrelated categorical choices, while some searches tested narrow predefined plans rather than letting Hyperopt resolve ordered trading values.
5. AI-generated trailing-only, fixed-RR-control, generic breakeven-only, and broad mixed families were judged weak research questions by the user and removed from the active V2 surface.
6. Generated helpers, inheritance, and support scaffolding made individual strategies harder to inspect and less representative of ordinary standalone Freqtrade strategies.
7. Pattern and non-pattern strategies were mixed despite materially different trade frequency and window requirements.
8. Historical queue construction allowed duplication and made source coverage harder to verify.

The checkpoint contains `599` active standalone strategies derived from `134` fixed entry sources:

- `134` `profit_level_full_or_ladder` files;
- `134` `profit_ladder_three_stage_no_ratchet` files;
- `134` `profit_ladder_three_stage_ratchet` files;
- `100` applicable `entry_target_full_or_zone_reversal` files;
- `97` applicable `target_partial_invalidation_remainder` files.

The queue contains every active file exactly once across `47` standard batches and `19` pattern batches. The exact checkpoint manifest is `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/v2/sieve3_v2_batches.json`.

For exit work:

1. Use `spaces=sell`.
2. Set `control_entry_exits=false`; the strategy owns exits, partials, and stop movement.
3. Use `MultiMetricHyperOptLoss` unless the user explicitly requests a comparison.
4. Default research pairs are `BTC/USDT:USDT`, `ETH/USDT:USDT`, and `SOL/USDT:USDT`.
5. Keep rare-pattern sources in pattern-only batches and give them broader windows when the ordinary surface starves them of trades.

## Current Exit Families

Every eligible entry source receives these three general families:

1. `profit_level_full_or_ladder`: full exit at an arbitrary profit target or a compact two/three-stage partial ladder.
2. `profit_ladder_three_stage_no_ratchet`: staged exits while the original protective stop remains unchanged.
3. `profit_ladder_three_stage_ratchet`: staged exits with stop movement after completed stages. Moving to breakeven means moving the stop to the entry price, not producing zero total trade profit after an earlier partial.

Applicability-gated families:

4. `entry_target_full_or_zone_reversal`: only when the entry/indicator exposes an explicit target known without lookahead. Compare full target exit with target-zone reversal confirmation.
5. `target_partial_invalidation_remainder`: only when the source exposes both an explicit target and a coherent entry-specific invalidation. Take a partial at/around the target and manage the remainder against the invalidation/ratchet contract.

Do not create a target family from keyword guesses, arbitrary OHLC columns, or a generic fallback target. Missing target/invalidation coverage is a generation problem to resolve, not a runtime fallback to fixed percentages.

Trailing-only, fixed-RR control, generic breakeven-only, and broad mixed exit families are not part of the current V2 set. Add them only if the user explicitly reopens those theories.

## Coverage Invariants

1. Every current entry source must appear once in each of the three general families.
2. Every source with a verified target provider must appear in `entry_target_full_or_zone_reversal`.
3. Every source with verified target and invalidation providers must appear in `target_partial_invalidation_remainder`.
4. Count coverage by parsed source stem after `_from_`; do not infer completeness from total file count.
5. Archive old strategies only after the replacement coverage matrix has been checked.
6. Queue generation must include every active V2 file exactly once and no archived file.

## Parameter Design

Use the parameter type that preserves useful search structure:

1. Use `IntParameter` for small contiguous integer choices.
2. Use `DecimalParameter` when genuinely ordered decimal resolution matters.
3. For coarse regular numeric steps, Hyperopt an integer multiplier and apply the step at point of use. Example: `1..8 * 0.05` represents `5%..40%` partial sizes.
4. Use `CategoricalParameter` for genuinely unordered modes such as `full`, `two_stage`, `three_stage`, `touch`, or `zone_reversal`.
5. Do not convert a small seven- or ten-value integer sweep into categorical values merely to avoid integers.
6. Avoid wide high-resolution sweeps that add no trading meaning.
7. Prefer search spaces below roughly `10,000` combinations. This is a judgement guide, not a hard cutoff; split only when doing so creates two coherent trading questions.
8. Avoid hidden parameters behind disabled gates where a clean sentinel or separate file is clearer.

Partial sizes should normally use coarse integer multipliers. The final exit always closes the remaining position; earlier partial percentages do not need to sum to 100%.

## Exit/Stop Collaboration

Stop movement is an optional management response, not a standalone exit theory.

Useful coherent sequences include:

1. First partial completes, then stop moves to entry.
2. Second target completes, then stop moves to the first completed target.
3. Final target or confirmed reversal closes the remainder.
4. Entry-defined target is reached, then a lower-timeframe reversal confirmation chooses full exit versus partial/remainder management.
5. Source invalidation remains the initial protective reference when the entry has a real invalidation level.

Do not combine unrelated target providers or unrelated invalidation theories in one file merely to reduce file count.

## Standalone Strategy Contract

1. Preserve the selected entry logic and static entry values.
2. Do not reactivate entry Hyperopt parameters during Sieve3 exit work.
3. Use one local `IStrategy` class per file.
4. Do not inherit from another strategy or import another strategy file.
5. Indicator modules are allowed dependencies; strategy helpers and generated strategy scaffolding are not.
6. Use native Freqtrade callbacks for full exits, partial position adjustment, and custom stoploss behaviour.
7. Partial actions must be idempotent and must not fire repeatedly for the same stage.
8. Long/short target, reversal, invalidation, and stop comparisons must be direction-aware.

## Planned Behaviour-Preserving Performance Refactor

The 22 July strategy checkpoint intentionally precedes a focused callback-performance change. The current exit decisions are valid, but the generic callback implementation repeats avoidable dataframe work:

1. `custom_exit`, `custom_stoploss`, and partial-position callbacks can independently rebuild the same trade context on the same candle.
2. Pure percentage ladders unnecessarily fetch and rescan analyzed data even though their decisions can be made from current profit/rate, timestamps, and stored stage state.
3. Source-target exits repeat dataframe date conversion, filtering, copying, sorting, and duplicate removal even though Freqtrade already provides a bounded time-sliced analyzed dataframe during Hyperopt.

The next implementation change must remain functionally equivalent:

1. Preserve every fixed entry, active exit parameter, target/invalidation binding, partial size, stop rule, timeout, and exit decision.
2. Calculate context once per pair/trade/candle and reuse it across callbacks.
3. Remove dataframe processing from percentage-only ladders where no candle/indicator evidence is consumed.
4. For source-target/reversal families, read only the minimum already-closed rows needed for the decision while preserving causal evaluation.
5. Treat any changed trade, selected parameter, or exit event as a behavioural regression, not a performance improvement.

## Smoke Test

Smoke testing is technical only:

1. BTC only.
2. One recent seven-day window.
3. Five Hyperopt epochs, one seed, one window, `spaces=sell`.
4. `control_entry_exits=false` and `MultiMetricHyperOptLoss`.
5. No derived backtest is required.
6. Compile/import, invalid parameter spaces, missing columns, callback exceptions, and failed commands are smoke failures.
7. Zero trades, weak performance, and rare-pattern starvation are not smoke failures.

After a complete smoke pass, do not keep describing smoke as pending. Move directly to real Sieve batches.

## Queue Rules

1. Use the `sieve3_v2_refined_standard_batch_*` namespace for ordinary entries.
2. Use `sieve3_v2_refined_pattern_batch_*` for pattern entries.
3. Keep all files for one entry source in one batch where the configured batch limit permits it.
4. Standard batches precede pattern batches.
5. Rebuild the queue whenever active strategy coverage changes.
6. Historical queue IDs and result files retain their original meaning and must not be silently reused for different strategy sets.

## Result Interpretation

Each row is a self-contained candidate: fixed entry + window + seed + selected exit parameters + validation result.

### TLV2 behavioral boundary

`complex_trendline_projection_v2` was replaced with a causal, prefix-stable implementation on 22 July 2026. This can materially change TLV2 line identity, confirmation timing, scores, targets, invalidations, entries, and exits. Treat results produced before this boundary as historical exploratory evidence only for any strategy that consumes TLV2 outputs; do not compare them as if they used the same entry/exit surface. New TLV2-dependent promotion or exit conclusions require results generated with the repaired indicator.

1. Compare selected exit parameters before comparing seeds.
2. Different seeds with different selected parameters are different exit solutions, not votes for or against the entry.
3. Similar parameters converging across seeds/windows are stronger robustness evidence.
4. Collapse dominated rows from the same entry and exit family.
5. Analyse entry families and exit families across individual rows; batch averages are not evidence.
6. High drawdown is the clearest weakness. Low winrate with strong asymmetric profit is not automatically invalid.
7. Pattern entries are naturally sparse and require pattern-specific interpretation.
