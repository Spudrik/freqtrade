---
doc_status: active
default_read: routed
owner: user+agent
purpose: Luna Max execution handoff for consolidating legacy Sieve3 exits into coherent source-specific Sieve3 V2 strategies.
do_not_use_for: Entry redesign, automatic strategy promotion, deleting tested legacy files, or deleting historical result evidence.
last_rebuilt: 2026-07-15
---

# Work Package - Luna Max Sieve3 V2 Exit Regeneration

## Decision

The refined comparison justifies replacing the broad generic exit-generation method with a consolidated Sieve3 V2 architecture.

Do not clone legacy exit files one-to-one. Recover the canonical entry foundation and the useful exit hypotheses, then generate one higher-quality integrated V2 strategy per entry foundation by default.

Preserve every strategy file that has a successful completed result and preserve all historical result artifacts. Replace untested legacy branch files with verified V2 coverage, then remove those untested originals so the strategy folder becomes smaller rather than doubling.

Preserve strong legacy rows as controls because some legacy exits still beat the refined pilot.

## Inventory Baseline

Snapshot at package creation:

1. `1,692` top-level `sieve3*.py` files.
2. `15` completed refined-pilot files.
3. `1,677` other legacy files.
4. `510` legacy files with at least one successful `status=ok` result row.
5. `1,167` legacy files with no recorded successful result.
6. `0` existing `sieve3_V2_*.py` files.
7. Approximately `150` canonical entry-source groups from filename and source metadata; Luna Max must resolve the exact count before generation.

With one V2 strategy per source, the indicative final top-level count is about `675`: `525` preserved tested files plus about `150` V2 files. A justified second V2 file for some sources gives an upper planning bound near `825`, still materially below the current `1,692`.

Define `tested original` strictly as a current strategy file referenced by at least one successful Sieve result row. Failed setup/code attempts are not successful tests. Where a result archive contains an exact source snapshot, use it to verify the historical file; record mismatches rather than creating extra active copies.

## Required Reading

1. `../00_project_control/objective_current.md`
2. `../02_rules/rules_codebase_workflow.md`
3. `../02_rules/rules_runtime_environment.md`
4. `../02_rules/rules_hyperopt_general.md`
5. `../02_rules/rules_sieve3_exit_hyperopt.md`
6. `../02_rules/rules_sieve3_exit_regeneration_agent.md`
7. `../02_rules/rules_result_reporting.md`
8. `../04_results/sieve3_sieve2_baseline_lookup.md`

## Pilot Evidence

Canonical comparison job:

`20260711T230600_entry_sieve3_exit_rework_comparison_001`

The JSONL has `85` physical rows because `25` completed keys were rerun. Analyze `60` unique keys by grouping on strategy, training window, seed, and validation window, then retaining the latest `finished_at` row.

| Exit architecture | Profitable | Drawdown <=5% | Drawdown >10% | Interpretation |
|---|---:|---:|---:|---|
| Layered | 17/20 | 14/20 | 1/20 | Most consistent pilot architecture. |
| Invalidation | 16/20 | 11/20 | 4/20 | Strong when invalidation directly represented the entry thesis; dangerous when it was generic or lagging. |
| Target zone | 11/20 | 10/20 | 7/20 | Highest upside in some target-rich sources, but least reliable when the source lacked a stable structural target. |

Selected same-scope legacy versus refined evidence:

| Entry foundation | Strong legacy row: return / DD | Strong refined row: return / DD | Decision signal |
|---|---:|---:|---|
| BOS bear continuation short | 8.50% / 1.31% | 7.69% / 0.70% target zone | Comparable return with lower refined DD; preserve both. |
| MTF daily MACD-volume breakout long | 11.14% / 14.47% | 5.58% / 6.34% layered | Refined reduced risk materially but did not preserve return. |
| H4 TLV2 support-break short | 16.30% / 0.92% | 10.24% / 2.64% target zone | Legacy control is stronger; do not discard it. |
| Multi2 TLV2/VP resistance-break long | 4.51% / 1.31% | 14.05% / 3.15% target zone | Clear refined improvement with still-controlled DD. |
| Prior-month-high breakout long | 7.64% / 3.19% | 11.89% / 4.30% layered | Strong refined improvement for modest additional DD. |

These five controls show mixed but sufficient evidence: two clear improvements, one substantial risk improvement, one comparable trade-off, and one legacy win.

## Objective

Consolidate the current legacy branch matrix into filename-accurate, entry-specific `sieve3_V2` strategies while keeping each promoted Sieve2 entry surface locked.

The normal unit of work is one canonical entry foundation, not one legacy exit file. Each V2 strategy should let Hyperopt compare coherent combinations of three complementary concepts inside one strategy:

1. `balanced_layered`: target interaction, partial realization, explicit remainder management, and separate invalidation.
2. `defensive_invalidation`: exit or reduce only when the original entry thesis materially fails.
3. `asymmetric_target_runner`: exploit a coherent structural target while retaining a controlled runner.

Include pure target and pure invalidation plans as ablation controls, then emphasize integrated target-plus-invalidation and layered plans. Create a second V2 strategy only when one file would mix unrelated target ecosystems, exceed the active-parameter/noise limits, or require a materially different Freqtrade implementation surface.

The intended result is fewer files that each test a richer but coherent set of human-readable exit policies.

## Scope

In scope:

1. All canonical Sieve3 entry foundations with verified Sieve2 lineage.
2. Existing source indicators and OHLCV-derived trade state.
3. VP, TLV2, pivots, BOS/CHoCH, prior levels, and source-native MTF context already available to the entry.
4. Pattern-native projections, boundaries, necklines, and failure states for pattern sources.
5. Standalone top-level `sieve3_V2_*.py` strategy files.
6. Consolidation and removal of untested legacy branch files after verified replacement.
7. Sieve Hyperopt and validation through existing launcher services.

Out of scope:

1. Editing entry logic to rescue weak exit results.
2. Applying generic non-pattern target logic to pattern sources without pattern-native review.
3. Adding generic RSI, MACD, EMA, Bollinger, stochastic, or oscillator exits unless that indicator already defines the source entry thesis.
4. Editing protected `freqtrade/**` code.
5. Deleting tested legacy files, historical results, params, backtest archives, or Hyperopt evidence.
6. Automatically promoting or rejecting regenerated strategies.

## V2 Naming And Traceability

Default filename:

`sieve3_V2_integrated_from_<canonical_entry_source>.py`

Default class prefix:

`Sieve3V2IntegratedFrom...`

Pattern filename:

`sieve3_V2_pattern_integrated_from_<canonical_entry_source>.py`

If a justified split is required, use a trader-readable role after `V2`, for example:

1. `sieve3_V2_integrated_from_<source>.py`
2. `sieve3_V2_asymmetric_runner_from_<source>.py`

Do not include a legacy exit-family name such as `breakeven`, `trailing`, or `fixed_tp_sl` unless the V2 file is intentionally a narrow control. The V2 filename should identify the canonical entry foundation and actual integrated role.

Each V2 module preflight block must include:

1. Canonical Sieve2 source and selected entry params source.
2. Locked entry signature.
3. Tested legacy control files and result IDs.
4. Untested legacy ideas consolidated or discarded as duplicates.
5. Target provider hierarchy and fixed/adaptive semantics.
6. Invalidation contract.
7. Named exit-policy groups and active Hyperopt parameters.
8. Why the source remains one V2 file or why a second file was required.

## Luna Max Coordinator Contract

Luna Max owns the migration map, source grouping, architecture, delegation, review gates, and final deletion pass. Delegate focused implementation work by entry archetype, not by arbitrary filename ranges:

1. Structure: BOS, CHoCH, swing break, support/resistance failure.
2. TLV2/VP: level break, value-area transition, POC/value-edge interaction.
3. Periodic/prior-level breakout: prior day/week/month and similar reference levels.
4. MTF trend/momentum: only source-native MTF conditions, with explicit lag and failure semantics.
5. Liquidity, crash, mean-reversion, and rejection entries.
6. Pattern entries: geometry, reversal, continuation, projection, neckline, and boundary families.
7. Other non-pattern OHLCV/level entries.

Use separate evidence, implementation, validation, and deduplication agents. An implementation agent must not self-certify its own filenames, locked entry parameters, executed exit signature, or legacy-deletion eligibility.

Luna Max must keep all agents working from the same canonical source-group map. Do not let separate agents create multiple V2 files for the same entry under slightly different normalized names.

## Per-Source Procedure

1. Resolve all legacy files belonging to the canonical entry foundation.
2. Resolve the exact promoted Sieve2 source and the result that justified promotion.
3. Verify every locked `buy` parameter against the selected params JSON or embedded defaults.
4. Mark each legacy file `tested_successfully` or `untested` from structured Sieve result rows.
5. For tested files, extract the executed best params and useful exit hypothesis. Preserve the file and its result artifacts unchanged.
6. For untested files, extract only any distinct trader idea worth retaining. Do not copy the universal legacy exit block.
7. Describe the entry premise in one sentence: why the trade exists, what objective it expects, and what observation disproves it.
8. Identify available target and invalidation columns from actual code. Do not infer capabilities from filenames.
9. Identify the strongest completed legacy rows on comparable pair/window surfaces and retain them as controls.
10. Design one integrated V2 policy surface using compact categorical plans.
11. Generate one standalone `sieve3_V2_integrated_from_<canonical_source>.py` file by default.
12. Independently verify filename, metadata, entry lock, active sell parameters, no-lookahead behavior, and executed exit signature.
13. Mark V2 coverage complete only when every distinct retained legacy hypothesis maps to a named V2 policy or is explicitly rejected as duplicate/incoherent.
14. After V2 load/preflight validation, delete only the untested legacy siblings listed in the approved migration map.
15. Run the comparison wave and return one representative result per entry and policy role, plus only materially distinct alternates.

Do not clone legacy files one-to-one. Clone the entry foundation and general useful ideas; redesign the exit implementation using the V2 contracts below.

## V2 Integrated Policy Architecture

Each V2 file should normally expose one primary categorical `exit_policy_plan`. Its values are complete, coherent state machines rather than arbitrary independent toggles.

Required plan groups where the source supports them:

1. `baseline_3_3`: fixed 3% TP / 3% SL control.
2. `target_full`: source-relevant target plus confirmation, then full exit.
3. `invalidation_full`: confirmed source-thesis failure, then full exit.
4. `target_partial_be`: target confirmation, partial realization, then breakeven remainder.
5. `target_partial_trail`: target confirmation, partial realization, then trailing remainder.
6. `target_then_invalidation`: manage profit at the target, then use thesis invalidation for the remainder.
7. `invalidation_reduce_then_target`: reduce on soft invalidation but retain a smaller position for the valid target.
8. `target_tighten_invalidation_exit`: tighten at the target and exit fully only on later invalidation.
9. `dual_target_runner`: partial at the first target and manage toward a second coherent target.
10. `time_or_progress_failure`: only where the entry premise implies timely follow-through.

Build approximately `20-50` named source-specific policy values by combining only sensible target, confirmation, invalidation, partial, and remainder choices. This gives Hyperopt a broad theory surface without allowing nonsensical Cartesian combinations.

Use no more than a few shared categorical modifiers where they remain meaningful across nearly all plans:

1. `risk_envelope`: compact hard-stop/fallback bundles.
2. `confirmation_strictness`: fast, normal, or strict closed-candle confirmation.
3. `target_distance_class`: near, normal, or extended, expressed through source-aware distance or R requirements.
4. `remainder_aggression`: breakeven, tight trail, normal trail, or structural trail.

Do not expose ten separate parameters when a named policy can encode their coherent relationship. Do not make Hyperopt waste epochs varying values that are inactive under the selected policy.

### Trade-State Sequence

Every integrated policy must be explainable as a state sequence:

1. `entry`: capture or identify the source target and invalidation contract using information available at that time.
2. `pre_target`: monitor hard invalidation and whether expected progress is occurring.
3. `target_zone`: apply the configured band, timeframe, and touch/reversal/rejection confirmation.
4. `realization`: full exit, partial, stop tightening, or no action according to the selected named plan.
5. `remainder`: manage with breakeven, trail, second target, or source invalidation.
6. `terminal`: full exit on the plan's final target, hard invalidation, risk envelope, or explicit fallback.

Partial-exit and state transitions must be idempotent. The same partial or state change must not fire repeatedly.

## Source-Relevance Hierarchy

Prefer targets in this order when available and coherent:

1. Source-native target captured by the entry indicator.
2. Next structural obstacle in the trade direction.
3. Pattern projection, neckline, or boundary target for pattern entries.
4. TLV2 or pivot level.
5. VP POC, value edge, HVN, or LVN interaction.
6. Prior swing or prior period high/low.
7. Fixed-R target only as a control or fallback.

Require a categorical minimum target distance or minimum R. Reject targets on the wrong side of entry or trivial nearby levels.

Targets must explicitly declare one of these semantics:

1. `frozen_at_entry`
2. `adaptive_closed_candle`
3. `frozen_then_nearer_risk_reduction_only`

An adaptive target may become more conservative but must not silently move farther away merely to avoid an exit.

Prefer invalidations in this order:

1. Failure of the exact reclaimed/broken/rejected source level.
2. Opposing source-native BOS/CHoCH or pattern failure.
3. Loss of required HTF context.
4. Agreement of two weaker failure sources.
5. Time/progress failure where the entry thesis is explicitly time-sensitive.

Do not use a generic one-bar EMA or oscillator flip as the sole invalidation of an unrelated structural thesis.

## Exit Role Contracts

### Balanced Layered

Required sequence:

`coherent target area -> confirmation -> partial realization -> breakeven or trail remainder`

Monitor source invalidation separately. Reaching a target and invalidating an entry are different events and must not be collapsed into one Boolean.

Use named plans such as:

1. `touch_partial33_then_be`
2. `reversal_partial50_then_trail`
3. `two_of_three_partial33_then_trail`
4. `close_reject_partial50_then_be`

### Defensive Invalidation

The invalidation must answer: "What observable event means the original entry thesis is no longer true?"

Valid examples include losing a reclaimed level, a confirmed opposing BOS/CHoCH, losing required HTF context, or closing back through the source breakout boundary.

Do not use a generic moving-average cross as invalidation for an unrelated structural entry. Use confirmation bars and a small categorical level band to avoid one-candle noise.

Actions may be full exit, reduce-and-tighten, losing-trade exit with profit lock, or tighten-only, but each must be an explicit named plan.

### Asymmetric Target Runner

The target must be causally related to the entry where possible: source level, next structural obstacle, pivot, VP POC/value edge, TLV2 level, or prior swing.

Explicitly encode whether a target is:

1. frozen from information available at entry, or
2. dynamically updated from closed candles.

Do not silently move a nominal target every candle. Treat fixed and adaptive targets as different modes.

Use compact categorical choices for:

1. target provider,
2. lookback,
3. target band up to roughly 3%,
4. confirmation timeframe,
5. touch/reversal/rejection confirmation,
6. full/partial/tighten action,
7. remainder behavior.

Add a categorical minimum-distance or minimum-R requirement so a technically valid but trivial nearby level cannot dominate the search.

## Entry-Archetype Exit Guidance

| Entry archetype | Preferred targets | Preferred invalidation | Layered emphasis |
|---|---|---|---|
| BOS/CHoCH and structural breaks | Next swing, pivot, TLV2 level, VP obstacle | Re-cross of broken level, opposing CHoCH/BOS, loss of HTF structure | Partial at first obstacle, trail toward next structure, full exit on opposing structure |
| TLV2/VP breaks and rejections | Next TLV2 level, POC, value edge, HVN/LVN transition | Failed reclaim/break plus adverse VP context | Realize at first level interaction, retain runner only while level context holds |
| Prior day/week/month and pivot breakouts | Next period level, swing, VP edge | Confirmed close back through breakout boundary, failed retest, time/progress failure | Partial at next reference area, breakeven remainder, exit on boundary failure |
| MTF trend/momentum | Next structure or VP obstacle; fixed-R only as control | Confirmed combination of LTF trigger failure and HTF context loss | Avoid one-indicator invalidation; partial on objective, require multi-source failure for remainder exit |
| Liquidity sweep, crash, and mean reversion | Mean/AVWAP where source-native, opposing liquidity, pivot, VP POC | Failure to reclaim promptly, new adverse sweep, lost reversal structure | Faster realization and tighter failure handling; do not force long runners without evidence |
| Geometric/reversal/continuation patterns | Measured move, neckline, channel/wedge boundary, pattern reaction level | Pattern boundary failure, opposite break, failed neckline/retest | Pattern target partial plus structural runner; use longer pattern windows |

Luna Max must refine these contracts from each source's actual columns. The table is a routing guide, not permission to manufacture unavailable signals.

## Hyperopt Design Rules

1. Use broad categorical theories, not wide integer or decimal ranges.
2. Prefer combined named plans over noisy Cartesian products.
3. Avoid hidden gates. Use null/sentinel states where they remain meaningful.
4. Keep normal modes near `5-10` active Hyperopt parameters.
5. Declare branch-local or potentially inactive parameters in the module preflight comment.
6. Use only closed candles and preserve informative-dataframe alignment.
7. Keep long and short semantics mirrored only where the underlying source logic truly mirrors.
8. Every filename must describe behavior that is actually reachable and executed.

### Search Emphasis From The Pilot

Lean into the evidence without assuming one architecture is universally best:

1. Make roughly half of the distinct policy plans integrated or layered target-plus-invalidation designs.
2. Retain smaller pure target and pure invalidation groups as ablation controls.
3. Retain the fixed 3/3 baseline and a small number of coherent fallback controls.
4. Do not duplicate equivalent plans merely to weight the search.
5. Give target-rich entries more meaningful target-provider variation.
6. Give MTF or lagging entries stronger multi-source invalidation and fewer one-signal full-exit plans.
7. Give high-asymmetry entries partial-plus-runner choices rather than forcing full target exits.
8. Give sparse pattern entries fewer, more source-native plans and broader training coverage.

Hyperopt should explore these trader questions:

1. Which source-relevant target is useful?
2. Should that target be frozen or adapt conservatively?
3. How wide is the target zone and how early may interaction begin?
4. Which timeframe and confirmation pattern makes the interaction actionable?
5. Should target interaction cause full exit, partial realization, or stop tightening?
6. What manages the remainder: breakeven, trail, second target, or invalidation?
7. Which event genuinely invalidates the source entry, and how much confirmation does it require?
8. Does invalidation behavior differ when the trade is losing, near breakeven, or already profitable?
9. What compact risk envelope prevents target/invalidation misses from becoming large drawdowns?

Use two search stages:

1. `theory discovery`: broad named policy plans, multiple seeds, and multiple routed windows.
2. `local refinement`: only after a policy family proves useful, tune a small categorical neighborhood around its target distance, band, confirmation, partial, and trail choices.

Do not fine-tune a weak theory. Do not use local refinement to turn an arbitrary value such as EMA 226 into a false precision result.

## Full V2 Migration Sequence

### Phase 0 - Canonical Inventory

1. Parse all top-level legacy Sieve3 files and all structured Sieve result JSONL files.
2. Normalize source lineage using module metadata first and filename suffixes second.
3. Produce one approved migration ledger with: legacy file, source group, tested status, successful result IDs, retained hypothesis, V2 replacement, checksum, validation status, and deletion status.
4. Detect ambiguous source groups, missing lineage, duplicate entry logic, and files whose current content may differ from the archived tested snapshot.
5. Do not generate or delete files until the source-group map passes independent review.

### Phase 1 - V2 Architecture Regression Wave

Use the five foundations from the completed refined pilot. Generate one integrated V2 file per source and confirm that the consolidated policy surface can reproduce or improve the useful target, invalidation, and layered behaviors without relying on the pilot helper/mixin.

Pass conditions:

1. Entry trades match the locked source under a neutral exit control.
2. Every exit plan is reachable and its signature matches its name.
3. Partial/state behavior is idempotent.
4. At least one target-rich and one layered source retain useful behavior.
5. The known weak MTF one-bar invalidation is not reproduced as a sole thesis failure.

Fix architecture/root-cause errors before proceeding.

### Phase 2 - Cross-Archetype Wave

Select `12-18` canonical sources spanning every entry archetype, both sides, sparse and high-trade entries, and at least three pattern sources. Generate, preflight, and run their V2 strategies.

Use this wave to verify that one integrated file per source remains interpretable and searchable. Split a source only when the documented split rules require it.

### Phase 3 - Full Generation

Proceed source family by source family. For each canonical source:

1. Build the V2 source contract.
2. Generate the standalone V2 file.
3. Run static, import, strategy-load, entry-lock, parameter-activity, and signature checks.
4. Record coverage of useful tested legacy hypotheses.
5. Mark untested legacy siblings eligible for deletion only after all checks pass.

Do not wait until every source is generated before reviewing quality. Complete and audit one family at a time.

### Phase 4 - Controlled Cleanup

1. Preserve the `510` tested legacy files and `15` tested pilot files unchanged.
2. Preserve all historical result, params, Hyperopt, and backtest artifacts.
3. Delete an untested legacy file only when the ledger shows no successful result and a validated V2 replacement covers its entry source and any distinct useful hypothesis.
4. Never use wildcard or broad recursive deletion. Delete exact ledger-approved paths only.
5. Recount top-level strategies after each family cleanup and verify no tested file disappeared.

### Phase 5 - V2 Batch Preparation

Create V2-only batches using the `sieve3_V2_*.py` filter. Separate pattern batches so their longer windows and rarity handling cannot be diluted by ordinary sources.

Keep legacy controls out of ordinary V2 discovery batches. Rerun a preserved legacy control only when a direct same-runtime comparison is required.

## Run Contract

Unless the user changes the campaign settings:

1. Sieve3 exit work only.
2. `spaces=sell`.
3. `control_entry_exits=false`.
4. `MultiMetricHyperOptLoss`.
5. Seeds `42` and `1337`.
6. Training windows `long_cycle_mixed_2020_2022` and `long_cycle_mixed_2023_2025`.
7. Validation timerange `20240401-20260401`.
8. Pairs: BTC, ETH, ADA, SOL, and BNB USDT perpetual pairs.
9. Use the user-approved current core schedule; do not invent or silently change resource limits.
10. Route pattern sources to separate V2 pattern batches with the longer windows required to produce a useful training sample.

Treat zero or tiny trade samples as a setup/coverage problem before treating them as strategy evidence.

## Analysis Contract

1. Deduplicate restarted rows by strategy, training window, seed, and validation window; retain the latest completed row.
2. Treat each seed result as a self-contained exit solution unless two seeds converge to the same effective parameters.
3. Same parameters plus similar good performance is strong confirmation.
4. Same parameters plus similar poor performance is strong rejection evidence for that exit contract.
5. Different parameters are different exit solutions; a weak seed does not invalidate a strong seed.
6. Group by entry foundation first, then executed policy role. Do not lead with raw top-N rows dominated by one source.
7. Compare return, drawdown, profit factor, trade count, win rate, expectancy where available, and exit signature.
8. Large drawdown is weak evidence unless compensated by a clearly distinct and useful role. There is no automatic promotion threshold.
9. Prefer a role when it improves return without disproportionate DD, materially reduces DD while preserving useful return, or provides a distinct asymmetric behavior worth further validation.
10. Preserve a stronger legacy control when the refined variant does not beat it.

Allowed verdicts:

1. `retain_refined`
2. `retain_legacy_control`
3. `retain_both_distinct_roles`
4. `revise_contract`
5. `park_source_for_later`

## Required Verification

Before any wave launch, independently confirm:

1. Exact Sieve2 source and promotion result.
2. Locked entry defaults or explicit loaded params.
3. Only sell/exit parameters are optimized.
4. No Sieve TP/SL entry override is active.
5. Strategy is standalone and loads through Freqtrade.
6. Target and invalidation columns exist on the executed timeframe.
7. Partial-exit state cannot repeat unintentionally.
8. Stop, fallback, and force-exit behavior are explicit.
9. Filename begins `sieve3_V2_`, class begins `Sieve3V2`, and module metadata matches actual behavior.
10. Named exit-policy signatures are materially distinct and reachable.
11. The V2 file contains no copied universal legacy exit block unrelated to its source.
12. Every planned deletion path is untested and has verified V2 replacement coverage.

## Preservation And Stop Rules

1. Preserve every tested legacy file and all historical results unchanged.
2. Preserve the helper-based 15-file pilot as comparison evidence, but do not use it as final standalone production architecture.
3. Replace and remove only untested legacy files after exact ledger-approved V2 coverage and independent verification.
4. Do not promote or reject strategies automatically.
5. On code/config/runtime failure, fix the root cause and rerun the same wave; do not skip it.
6. If the regression or cross-archetype wave fails to improve coherence or provide useful trade-offs versus controls, stop and revise the architecture before full generation.
7. If a source lacks a real target or invalidation contract, omit that policy group rather than manufacturing generic logic.
8. If one integrated file becomes noisy or uninterpretable, split it once along a genuine implementation or target-ecosystem boundary. Do not recreate the old many-file matrix.
9. Do not edit tested originals to add V2 metadata or naming. Their unchanged state is part of their control value.

## Handoff Output

Maintain the single approved migration ledger from Phase 0. Return a concise table with one row per entry foundation and representative executed policy:

`source | V2 file | policy | executed exit signature | seed/window | trades | return % | DD % | PF | WR % | legacy control | verdict | reason`

Final handoff must also report:

1. Tested originals preserved.
2. Untested originals replaced and removed.
3. Canonical source groups covered.
4. V2 files created.
5. Sources requiring a justified second V2 file.
6. Ambiguous or blocked sources left untouched.
7. Result and archive preservation verification.

Do not create extra history logs or duplicate metadata files. Put strategy-specific traceability in the module preflight block and use existing Sieve result artifacts.
