---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for Sieve entry, exit, position-adjustment, and candidate-selection research.
do_not_use_for: Per-run reporting, final promotion decisions, raw seed rows, or replacing Sieve rules and result evidence.
last_rebuilt: 2026-07-22
---

# Track - Sieve Entry and Exit Research

## 1. Boundary

This track owns Sieve generation history, entry/exit research stages, position-adjustment experiments, and programme-level Sieve conclusions. It does not own indicator implementation, FreqAI modelling, data ingestion, or detailed result storage.

## 2. Current state

- Sieve is the only approved Hyperopt/discovery system.
- The broad legacy Sieve3 campaign is preserved as exploratory evidence but is no longer the active strategy surface. A snapshot found about `1,645` regular files and results for about `505` distinct strategies across `12` tested families.
- Sieve3 V2 is the active exit programme. It contains `599` standalone files built from `134` fixed entry sources and tests five focused exit families rather than the legacy broad family sweep.
- Three general families cover all `134` sources: arbitrary profit full/ladder, three-stage ladder without stop ratchet, and three-stage ladder with stop ratchet.
- Verified source targets allow `100` target/full-or-zone-reversal files. Verified target plus invalidation contracts allow `97` target-partial/invalidation-remainder files.
- The exact active queue has `66` batches (`47` standard and `19` pattern) and includes every active V2 strategy exactly once.
- Refined standard batches `001` through `003` are complete with `10/10` clean rows each. Batch `004` was active when this checkpoint was written.
- A callback-performance refactor is next. It must preserve all trading behaviour while removing repeated dataframe reconstruction within the same trade/candle.
- Pattern sources require their separate track and wider coverage rules.
- Same-direction signal adds were tested; systematic structural price-zone staggering remains unproven.

### Why V2 replaced the original Sieve3 surface

- Promoted entries must carry the selected entry values into every exit branch; copied files must not silently revert to unrelated defaults.
- Exit filenames must match the executed exit behaviour.
- Targets and invalidations must come from the actual entry/indicator contract rather than generic fallback levels or keyword guesses.
- Strategies must be standalone ordinary Freqtrade modules rather than generated inheritance/helper frameworks.
- Ordered numeric searches use `IntParameter`, `DecimalParameter`, or coarse integer multipliers. Categorical parameters are reserved for genuinely unordered decisions.
- Trailing-only, fixed-RR-control, generic breakeven-only, and broad mixed AI-generated families were removed because the user considers them weak research questions.
- Pattern sources are separated because rarity requires different windows and interpretation.

## 3. Stage definitions

| Stage | Definition | Durable distinction |
|---|---|---|
| Sieve1 | Breadth-first disposable entry probes under coarse entry-only tuning and fixed TP/SL diagnostics. | Asked whether an isolated idea showed enough edge for deeper work. |
| Sieve2 | Next-generation entry refinement/validation, usually from promising or salvageable Sieve1 foundations. | Added HTF context, cleaner confluence, overtrade guards, sparse-trigger loosening, reframing, and optional fixed-target validation. |
| Sieve3 | Current exit/risk development around locked selected entry surfaces, with explicitly marked novel ideas allowed. | Tests source-specific targets, invalidations, partials, trails, guards, and position management. |

The names are workflow generations, not permanent indicator categories. A new idea introduced during a later generation should use that active generation's naming.

## 4. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| 15-21 May 2026 | Sieve1 broad entry discovery | Established entry-only diagnostics across indicator and confluence families; results had no automatic pass/fail gate. | `../../04_results/results_recent_summary.md`; runtime Entry Sieve results. |
| 21 May-12 June | Sieve2 refinement/rescue | Promising entries received context/guards; noisy concepts were guarded, sparse concepts loosened, and weak concepts reframed or parked. | `../../04_results/sieve3_sieve2_baseline_lookup.md`; archived Entry Sieve results. |
| 5-7 June | Entry overlap and reserve/add tests | Same-direction overlap and selective later-signal adds were strong subsets, but reserving stake reduced whole-strategy return. | `../../99_archive/original_uploaded_docs/history_logs/objectives_progress.md`. |
| 8-9 June | Broader position management | Adds/partials/reductions often lowered drawdown but cut runners too aggressively; the best runner branch was near its baseline. | Same historical progress log. |
| 11 June | Non-Sieve discovery retired | Sieve became the sole discovery path; older production-alpha results remain architecture lessons only. | Repo-root `AGENTS.md` and Sieve rules. |
| 13 June-10 July | Broad Sieve3 exit campaign | Generic exit-family performance varied strongly by entry; only part of the generated backlog was tested. | Entry Sieve speed JSONL and `../../02_rules/rules_sieve3_exit_hyperopt.md`. |
| 11 July onward | Refined exit comparison | Began testing whether explicit target/invalidation/action contracts are more coherent than legacy naming/execution. | `../../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_exit_rework/results/`. |
| 20-22 July | Sieve3 V2 focused surface | Replaced the unfinished broad sweep with `599` standalone strategies, five focused exit families, exact source coverage, and separate standard/pattern queues. | `../../02_rules/rules_sieve3_exit_hyperopt.md`; committed V2 batch manifest and result JSONL files. |
| 21-22 July | Indicator causality boundary | Pattern geometry and TLV2 calculations were made causal; TLV2 was then rebuilt as a bounded, prefix-stable upcoming-zone indicator. This materially changes affected entries, targets, invalidations, and exits. | Indicator commits `4eb7e7962`, `2d9a3facc`, and `37a6f6957`. |

## 5. Current conclusions

### Demonstrated

- Sieve infrastructure can collect broad entry-only and exit-stage evidence reproducibly.
- Entry quality and exit quality must be tested separately.
- Exit behaviour is entry-family dependent.
- Later same-direction signal stacking was implemented and tested.
- Standalone V2 strategies can complete focused sell-space Hyperopt and validation with strategy-owned exits.
- The repaired TLV2 implementation is prefix-stable in targeted checks and reduced its indicator-stage runtime from hours in the broken run to roughly a minute in the first repaired batch.

### Promising

- Guarded TLV2/VP/prior/structure entries.
- Source-specific target zones and invalidations.
- Selective adds when the entry thesis remains intact.
- Multiple exit roles only when their executed behaviour is genuinely distinct.

### Weak or harmful in tested forms

- Raw pivot/score-style triggers that overtrade without structural filtering.
- Broad shared defensive reductions and warning exits that remove profitable runners.
- Treating an exit filename as proof of executed behaviour.
- Reserving excessive stake for adds without measuring the whole-strategy opportunity cost.
- Treating trailing-only, fixed-RR-control, or generic breakeven-only plans as sufficiently rich exit research.
- Repeating full analyzed-dataframe reconstruction independently in every exit/stop/partial callback.

### Insufficient evidence

- Rare pattern entries with only a handful of trades.
- Early refined-exit rows before the comparison completes.
- Early multi-partial structural-ladder results from a very small tested sample.

## 6. Outstanding work

### Active

- Complete all ordered V2 standard and pattern batches, stopping on runtime failures and rerunning the same batch after root-cause fixes.
- Compare individual exit rows across matched entries rather than treating batch averages as evidence.
- Perform the behaviour-preserving callback optimization documented in `rules_sieve3_exit_hyperopt.md` before committing to the full long-duration queue.

### Next

- Decide whether untested legacy exits should continue unchanged or be regenerated in bounded source-family batches.
- After separate approval, build one canonical aggregation view from raw JSONL/ZIP evidence, the Sieve2 lookup, and one compact Entry/Exit Decision Register.
- Derive `exit_signature` from target source, invalidation source, partial behaviour, stop movement, trailing, trigger timeframe, and fallback—not filename.
- Screen all observations automatically, review only survivors in depth, and retain primary/defensive/trend roles only when materially distinct.

### Planned research

- Test structural top/middle/bottom entry-zone tranches against one-shot entry and later-signal stacking under the same stake budget.
- Consolidate retained entries across windows/pairs/seeds before assigning master-strategy weights.
- Measure entry overlap so correlated variants do not multiply the same evidence.

## 7. Canonical evidence

- Active task routing: repo-root `AGENTS.md` plus the user's latest explicit request.
- Entry/exit/position rules: `../../02_rules/rules_entry_exit_position.md`
- Sieve3 exit rules: `../../02_rules/rules_sieve3_exit_hyperopt.md`
- Refined regeneration logic: `../../02_rules/rules_sieve3_exit_regeneration_agent.md`
- Sieve2 baseline lookup: `../../04_results/sieve3_sieve2_baseline_lookup.md`
- Recent V2 result checkpoint: `../../04_results/results_recent_summary.md`
- Active V2 batch manifest: `../../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/v2/sieve3_v2_batches.json`
- Entry Sieve manual: `../../../user_data/Custom_Launcher/docs/EXPLORER_ENTRY_SIEVE_MANUAL.md`
- Broad results: `../../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_speed/results/`
- Refined results: `../../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_exit_rework/results/`
