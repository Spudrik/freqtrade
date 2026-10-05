---
doc_status: temporary-active
default_read: no
owner: user+agent
purpose: Preserve the approved Sieve3 V2 target-quality investigation through context compression.
delete_when: Permanent generic-level Sieve3 replacement files have been created, their evidence is routed elsewhere, and the user approves deletion.
created: 2026-07-30
last_rebuilt: 2026-08-07
---

# Standalone Target Quality Investigation - Stage One

The strategies retain `sieve3_V2` in their filenames only to preserve their
entry lineage. This investigation is not a Sieve3 result stage. Its jobs,
status, logs, and results live under the dedicated
`entry_sieve_target_quality_investigation_stage1` runtime.

## Stage One Status

Stage One completed cleanly on `2026-08-04`:

1. `312` temporary strategies covered all `39` identified entry sources and all eight target families.
2. All `40/40` manifest batches completed.
3. The audited latest-result set contains `344` successful operational rows: `280` standard rows and `64` rare-pattern rows across two routed windows.
4. One superseded Batch 001 result is excluded by selecting the newest completed result file for every manifest batch.
5. Four runtime fixes are preserved in `fixes.jsonl`.
6. Stage One is closed. Stage Two is a separate target-context investigation, not an extension of Stage One exit management.

## Why This Investigation Exists

Some current target-aware V2 strategies build one pooled VP/TLV2 target before entry by selecting the nearest level relative to that candle's close. At the actual fill:

1. the selected level can be behind or too close to the fill;
2. a valid second or third level is not tried;
3. POC, VAH/VAL, HVN, LVN, and TLV2 lines are treated as equivalent;
4. available quality metadata is mostly ignored;
5. base- and higher-timeframe levels are not ranked systematically.

Stage One fixes only the first four questions. Multi-timeframe priority is deliberately deferred.

## Known Scope

The current active V2 surface contains:

1. `76` affected target files across `39` entry sources.
2. `28` files across `14` sources with the exact generic pooled VP/TLV2 selector.
3. `47` files across `24` sources with similar nearest-forward or entry-time target selection.
4. One additional BOS-continuation source uses the same nearest-level behavior through an indirect provider constant; it was found by the residual read-only audit.
5. All `39` identified entry sources remain in one investigation surface. A source's existing named target is not assumed to be better merely because an earlier AI selected it.
6. After source-specific review, 41 valid or minimally repaired strategies were restored. Nine generic-nearest-target strategies from five sources remain parked under `user_data/strategies/isolated_possible_errors`; its `README.md` records the exact rationale.
7. Other non-target files are not part of this target-quality investigation.

## Stage One Entry Cohort

Use all `39` identified entry sources. Do not split them into generic-target and named-target cohorts.

The exact source stems are the source component of each generated filename:

`complete_pattern_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h`,
`complete_pattern_multi2_tlv2_boschoch_res_prox_reject_choch_bear_short_4h`,
`complete_pattern_multi2_tlv2_vp_res_break_vp_node_long_4h`,
`complete_pattern_multi2_vp_prior_equal_highs_reject_vp_vah_short`,
`mtf_confluence_d1_vp_bos_4h_retest_short`,
`multi2_tlv2_vp_res_break_vp_bullctx_long_1h`,
`multi2_tlv2_vp_sup_break_vp_bearctx_short_1h`,
the six prior-month/prior-week VP-high-break sources,
`multi3_prior_avwap_vp_breakout_long`,
`multi4_prior_range_avwap_vp_breakout_long`,
`overtrade_bos_bull_continuation_long_1h_vp_market_guard`,
the five overtrade demand/supply/pivot/H4-structure sources,
the four overtrade TLV2/VP node sources,
the four overtrade prior-day VP sources,
the four overtrade TLV2 breakout/breakdown/reclaim sources,
and the six reframed capitulation/TLV2/VP sources.

## Temporary Target Families

Create one focused file per target family and entry source:

1. `pooled`: corrected pooled-nearest control selected against the actual fill, with fallback to the next valid candidate.
2. `poc_age`: current POC with a Hyperopted minimum persistence/age requirement.
3. `prior_poc`: prior POC as the directional target.
4. `va_edge`: VAH for long trades or VAL for short trades.
5. `hvn_strength`: directional HVN gated by its existing strength metadata.
6. `lvn_thinness`: directional LVN gated by its existing thinness metadata.
7. `tlv2_quality`: directional TLV2 levels filtered using score, explicit pivot count, and causal line age.
8. `confluence`: the nearest directional zone where VP/TLV2 features overlap; Hyperopt the required feature count from `1` through `4`.

Total temporary files: `39 entries * 8 families = 312`.

Operational batching: `40` batches in total, comprising `32` standard
three-pair batches and `8` rare-pattern five-pair batches.

## Target Selection Contract

1. Select targets from the last closed candle available at the actual entry fill.
2. Compare every candidate to the actual fill rate, not a pre-entry close.
3. Ignore candidates on the wrong side or inside the existing minimum-distance boundary.
4. If the nearest candidate fails, inspect the next candidate rather than returning no target immediately.
5. Freeze the chosen target and its quality metadata into trade state.
6. Keep target providers explicit; do not keyword-scan dataframe columns.
7. Use only causal indicator outputs already available at the decision candle.
8. Do not edit VP or TLV2 indicator modules for Stage One.

## Exit Isolation

Stage One measures target quality, not partial-exit or stop-ratchet architecture:

1. Use full exits only.
2. Allow the existing ordered target-zone width and a compact touch-versus-close full-exit choice.
3. Keep the same protective invalidation/hard-stop behavior across every variant of one source.
4. Do not add partial exits, stop ratchets, timeout resets, target queues, or target-to-target progression.
5. Keep strategy files standalone and in normal Freqtrade format.

## Quality Parameters

1. Use ordered integer parameters or integer multipliers where the metadata is ordered.
2. Keep numeric resolution coarse and meaningful.
3. POC quality means causal persistence at approximately the same level.
4. HVN quality uses the indicator's strength output.
5. LVN quality uses the indicator's thinness output.
6. TLV2 quality uses score, explicit pivot count, and causal line age.
7. Confluence count is an ordered integer `1..4`; `1` is the fallback/control.
8. Keep each file below roughly `10,000` combinations where practical.

## Run Contract

Use the isolated runtime and manifest below. Do not add these temporary files
or their results to the Sieve3/V2 queues or result directories.

- Runtime:
  `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1`
- Manifest:
  `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1/target_quality_stage1_batches.json`

Source-equivalent settings:

1. Hyperopt space: `sell`.
2. Sieve exit control: `control_entry_exits=false`.
3. Loss: `MultiMetricHyperOptLoss`.
4. Epochs are family-scaled instead of using one arbitrary count:
   - `pooled`, `prior_poc`, and `va_edge`: `60`;
   - `hvn_strength` and `lvn_thinness`: `180`;
   - `poc_age`: `250`;
   - `confluence`: `180`;
   - `tlv2_quality`: `800`.
5. Seed: `42`.
6. Standard-source training: `long_cycle_mixed_2020_2022`.
7. Rare-pattern training: two routed auto-windows, preserving the broader
   coverage required for sparse pattern entries.
8. Validation: `20240401-20260401`.
9. Standard pairs: `BTC/USDT:USDT`, `ETH/USDT:USDT`,
   `SOL/USDT:USDT`.
10. Rare-pattern pairs: `BTC/USDT:USDT`, `ETH/USDT:USDT`,
    `ADA/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`.
11. Split-venv backtest pipeline enabled.

## Review Gates

Before launch verify:

1. Entry behavior and locked entry values match the source file.
2. Every file is standalone and imports no strategy parent/helper.
3. Filename, class name, target family, and executed selector agree.
4. Target selection is actual-fill-relative and direction-aware.
5. Quality columns exist before use and are causal.
6. Only intended sell parameters are active.
7. Files compile, import, and instantiate.
8. The isolated manifest contains every temporary file exactly once.

## Result Interpretation

1. Analyse individual strategy rows, not batch averages.
2. Compare the same entry across target families first.
3. Then look for target families or quality thresholds that recur across different entries.
4. Expect multiple useful candidates rather than one universal winner.
5. High drawdown is the clearest weakness; do not reject low-win-rate asymmetric results automatically.
6. Treat confluence results as exploratory rather than conclusive.
7. Preserve consistently strong or entry-specific target ideas for the Stage Two discussion.

## Stage One Evidence Snapshot

The machine-readable summary preserves exact individual rows, selected sell parameters, run metadata, statistical definitions, raw-result links, and the final filesystem decision:

- Summary: `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1/stage1_result_summary.json`
- Raw results: `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1/results`
- Manifest: `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1/target_quality_stage1_batches.json`
- Fix ledger: `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage1/fixes.jsonl`

Standard-source headline evidence (`n=35` per family):

| Family | Positive | Median profit | Profit Q25..Q75 | Median DD | Within 1pp of source best |
|---|---:|---:|---:|---:|---:|
| `va_edge` | 25 | 1.91% | -0.07%..3.48% | 3.07% | 26 |
| `confluence` | 24 | 1.51% | -0.49%..4.10% | 3.44% | 21 |
| `hvn_strength` | 21 | 0.48% | -0.41%..2.56% | 3.72% | 17 |
| `prior_poc` | 20 | 0.46% | -0.92%..2.97% | 3.85% | 14 |

The paired eight-family Friedman test produced `p=6.642789811151292e-07`. This shows that the observed family profit distributions differ; it does not prove that a named level caused the difference.

Final standalone decision:

1. Retain `va_edge`, `confluence`, `hvn_strength`, and `prior_poc`.
2. Archive `pooled`, `poc_age`, `lvn_thinness`, and standalone `tlv2_quality`.
3. TLV2 remains available as a contributing member in confluence tests, not as a retained standalone family.
4. The retained `156` temporary files (`39` per retained family) were archived on `2026-08-07` under `user_data/strategies/Archive/sieve3_v2_target_quality_stage1_20260807` after their evidence had been captured. The rejected archive contains `39` files for each rejected family.

## Stage Two Superseded By FreqAI

The isolated Stage Two Sieve run completed all `8/8` batches, but it did not become the final decision source. Its limited entry-conditioned comparisons could not cleanly distinguish recurring level relevance from chance alignment with a particular entry sample. The later FreqAI level-reaction and exit-readiness work superseded Stage Two by testing named levels, cross-type clusters, cross-timeframe alignment, quality metadata, weakening, BTC context, chronological holdouts, and stale-level placebos directly.

Durable FreqAI evidence:

- Results ledger: `user_data/research_news_data/context_features/reports/general_exit_freqai_results_ledger.csv`
- Reaction and placebo reports: `user_data/research_news_data/context_features/reports/general_exit_level_reaction`
- Stage Two operational evidence remains at `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_target_quality_investigation_stage2`, but is exploratory rather than the final architecture decision.
- Its `48` temporary top-level strategies were archived on `2026-08-07` under `user_data/strategies/Archive/sieve3_v2_target_quality_stage2_superseded_20260807`; the runtime evidence remains in place.

The permanent interpretation is:

1. Retain directional VA edge, strength-qualified HVN, prior POC, and explicit confluence as generic level candidates.
2. TLV2 may strengthen a confluence zone but is not retained as a standalone scored generic target family. The later user-approved `tlv2_quality_reaction_base/higher_tf` files isolate TLV2-plus-weakening as a focused diagnostic rather than silently promoting it into the scored set.
3. Pooled-nearest, standalone POC-age, and LVN-thinness variants are not retained.
4. A level touch by itself is not sufficient exit evidence. Proximity, level relevance, cross-type or cross-timeframe alignment, and closed-candle weakening/rejection provide the useful readiness context.
5. Higher-timeframe and multi-timeframe alignment can raise relevance, but no timeframe or level type is universally dominant.
6. Broad BTC context is not a mandatory exit gate. Same-side BTC level/reaction context may contribute a small optional score only when it is itself relevant.
7. These findings define compact deterministic Sieve3 level scoring. Permanent Sieve3 strategies do not run FreqAI inference.

## Permanent Sieve3 Return Scope

The current permanent top-level Sieve3 surface contains `990` files from `134` locked entry sources. The temporary target-quality files are archived and excluded, as are the nine historical references under `isolated_possible_errors`, all other archived files, deleted files, and stale manifest references.

The three general profit/ladder families cover all `134` sources. Active source-target coverage is now `94` full/reaction files and `94` partial/invalidation files. The generic-level cohort contains the original `39` uncovered sources plus `mtf_confluence_d1_vp_bos_4h_retest_short`, whose active generic-nearest target branch was archived and replaced. The resulting `40` sources contain `33` one-hour and seven four-hour entries, split evenly between `20` long and `20` short.

The completed build created ten corrected permanent generic-level files for each of those `40` sources:

1. `sieve3_V2_generic_scored_level_full_or_reaction_from_<source>.py`
2. `sieve3_V2_generic_scored_level_partial_progression_from_<source>.py`
3. Base and higher-timeframe `va_edge_reaction`.
4. Base and higher-timeframe `prior_poc_reaction`.
5. Base and higher-timeframe `hvn_strength_reaction`.
6. Base and higher-timeframe `tlv2_quality_reaction`.

One missing coherent `target_partial_invalidation_remainder` counterpart was also added for `mtfx_d1_vp_vah_reject_short_4h_retest`, producing `401` final queued tests. Rare-pattern sources remain in a pattern-only batch with broader routed windows.

### Shared Fixed Contract

1. Copy the active standalone locked-entry file and change only the exit section plus required strategy-local level columns.
2. Use explicit VA-edge, strength-qualified HVN, prior-POC, and confluence providers. Do not keyword-scan dataframe columns or restore an isolated selector.
3. Use only levels available from closed candles. Compare candidates to the actual fill, reject wrong-side or too-close candidates, and inspect the next valid candidate.
4. Merge nearby compatible levels into a direction-aware zone and preserve its type, timeframe, quality, and confluence metadata in trade state.
5. A clean pass advances to the next currently known qualifying zone. A reaction is evaluated only after the zone becomes relevant.
6. Preserve a coherent source invalidation where one exists; otherwise preserve the source file's existing hard-stop contract. Do not present a generic percentage as source invalidation.
7. Keep every resulting strategy standalone and use native Freqtrade callbacks and idempotent partial state.

### Full Or Reaction Hyperopt Surface

1. `level_order_mode`: `nearest_qualified`, `highest_score`, or `higher_tf_priority`.
2. `minimum_level_score`: ordered integer `3..7`.
3. `zone_width_quarter_percent`: ordered integer `1..12`, applied as `0.25%` steps.
4. `reaction_profile`: rejection close, `2-of-2` weakening, `2-of-3` weakening, or `3-of-3` weakening.
5. `mtf_alignment_bonus`: ordered integer `0..2`.
6. `btc_same_side_bonus`: ordered integer `0..2`.

The nominal surface is approximately `6,480` combinations. The discrete reaction profiles are categorical because they are different behaviours; ordered distances and weights remain integer searches.

### Partial Progression Hyperopt Surface

1. `zone_width_quarter_percent`: ordered integer `1..12`, applied as `0.25%` steps.
2. `partial_1_five_percent_units`: integer `1..8`, applied as `5%..40%`.
3. `partial_2_five_percent_units`: integer `0..8`; zero means no second partial and nonzero values represent `5%..40%`.
4. `ratchet_lag_levels`: `0` for no ratchet, `1` for the most recently protected level, or `2` for two protected levels behind. At the first protected stage, the ratchet reference is entry.
5. `reaction_profile`: `2-of-2`, `2-of-3`, or `3-of-3` closed-candle weakening.

The nominal surface is approximately `7,776` combinations. The final action always closes the remaining position. A cleanly crossed zone advances without forcing a partial.

### Focused Named-Level Reaction Surfaces

The eight simple branches isolate one named provider at a time and require directional weakening/rejection around its zone:

1. Base VA edge, prior POC, and qualified TLV2: `192` combinations each.
2. Higher-timeframe VA edge, prior POC, and qualified TLV2: `576-768` combinations each.
3. Base strength-qualified HVN: `960` combinations.
4. Higher-timeframe strength-qualified HVN: `2,880-3,840` combinations.

These files compare simple level reactions against the scored families. They do not use BTC context or cross-family scoring, and a clean pass advances to the next level rather than forcing an exit.

### Active Final Queue

The checked manifest is `user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_final_generic_levels/sieve3_v2_final_generic_level_batches.json`. It contains all `401` files exactly once across ten standard logic-family batches and one routed rare-pattern batch. The first batch started on `2026-08-07`.

## Completion Meaning

Generating these files restores research coverage; it does not complete an entry. Sieve3 completion means obtaining several statistically relevant positive and materially distinct exit solutions for each locked entry. Isolated, archived, deleted, failed, incomplete, temporary, or superseded strategies and results contribute zero completion credit.

The permanent files and evidence routing are now built. This document remains only because deletion requires explicit user approval.
