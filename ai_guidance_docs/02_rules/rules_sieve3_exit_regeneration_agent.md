---
doc_status: active
default_read: routed
owner: user+agent
purpose: Implementation contract for creating or repairing standalone Sieve3 V2 exit strategies.
do_not_use_for: Historical V2 architecture, broad strategy generation, or result promotion.
last_rebuilt: 2026-08-07
---

# Rules - Sieve3 V2 Strategy Regeneration

## Required Process

For each entry source:

1. Start from the current standalone strategy that contains the locked entry logic.
2. Preserve entry conditions, informative data, static entry values, direction, timeframe, and entry tags.
3. Replace only the exit-family implementation and its sell/risk parameters.
4. Keep the resulting file standalone in normal Freqtrade strategy format.
5. Compile/import the file and verify filename, class name, `EXIT_FAMILY`, active sell parameters, and locked entry surface.

Do not use parent strategies, mixins, external strategy helpers, mechanical support frameworks, or runtime fallback target discovery.

## Required Family Coverage

Every source:

1. `profit_level_full_or_ladder`
2. `profit_ladder_three_stage_no_ratchet`
3. `profit_ladder_three_stage_ratchet`

When an explicit source target is available without lookahead:

4. `entry_target_full_or_zone_reversal`

When both target and entry-specific invalidation are available:

5. `target_partial_invalidation_remainder`

When no trusted active source-target branch exists, add both FreqAI-informed generic-level families:

6. `generic_scored_level_full_or_reaction`
7. `generic_scored_level_partial_progression`

For the approved final generic-level diagnostic cohort, also add the focused base/higher-timeframe reactions for VA edge, prior POC, strength-qualified HVN, and quality-qualified TLV2. Each file must isolate exactly the named level provider and require directional weakening or rejection around that level. Do not silently merge these focused hypotheses into the scored files.

Do not omit applicable target families merely because a previous generator failed to classify the source. Review the source indicator contract and preserved historical source-specific strategy evidence.

Files under `Archive`, `isolated_possible_errors`, any other non-active directory, or a temporary `sieve3_V2_tq_*` investigation prefix do not count as coverage. Never restore an isolated strategy mechanically. It may return only after a source-specific review disproves or corrects the recorded concern and compile, import, and locked-entry parity checks pass; otherwise build a corrected new standalone file from the active locked-entry source and the approved current exit contract.

## Target And Invalidation Mapping

Target and invalidation providers must be explicit and source-specific.

Acceptable evidence:

1. A strategy-facing indicator column already used by the source entry.
2. A level frozen or available at entry without future candles.
3. A documented prior V2 source-specific target/invalidation contract that can be verified against current indicator outputs.
4. A prior high/low, VP, TLV2, pivot, pattern projection, neckline, channel boundary, or similar level only when its availability and direction are explicit.

Unacceptable shortcuts:

1. Keyword-scanning dataframe columns.
2. Selecting any column containing `high`, `low`, `target`, or `level`.
3. Future-confirmed pivots or retroactive patterns treated as entry-time knowledge.
4. Generic fixed percentages presented as indicator-defined targets.
5. Silent fallback when the required target column is absent.

If two materially different invalidation levels are both coherent, create separate focused files rather than hiding a structural choice behind a deep branch.

## Family Behaviour

### Profit Level Full Or Ladder

Compare full exit against compact two/three-stage arbitrary profit ladders. Partial sizes use coarse integer multipliers. The last stage closes the remainder.

### Three-Stage No Ratchet

Use three ordered profit stages. The original stop remains unchanged after partial exits so this file isolates the value of staged profit-taking.

### Three-Stage Ratchet

Use the same staged structure but move the stop after completed stages. First movement normally means stop to entry; later movement may use the previous completed target.

### Entry Target Full Or Zone Reversal

Use the source-defined target. Compare immediate/touch or close-based full exit with waiting inside a tunable zone for coherent reversal evidence. Zone width and confirmation depth may be Hyperopted with ordered numeric parameters.

### Target Partial Invalidation Remainder

Take a Hyperopt-selected partial around the source target, then manage the remainder using the source invalidation and optional stage-linked stop movement. Do not substitute a generic fixed stop for a claimed source invalidation.

### Generic Scored Level Full Or Reaction

Use only levels available from closed candles and compare candidates to the actual fill. Merge nearby VA-edge, strength-qualified HVN, prior-POC, and qualified confluence candidates into direction-aware zones. A clean pass advances to the next available zone; a level touch alone is not treated as proof of an exit. Full exit requires the selected close/rejection/weakening reaction profile.

### Generic Scored Level Partial Progression

Use the same deterministic scored-zone provider. A confirmed reaction may take one or two coarse Hyperopt-sized partials; the final action closes the remainder. A clean pass advances to the next zone. Optional stop movement means stop to entry after the first protected stage, then to an earlier passed zone after later progress. Keep each partial idempotent.

### Focused Named-Level Reaction

Use only the filename's named level family. Base variants use the strategy timeframe; higher-timeframe variants Hyperopt the valid higher timeframe. Select the next direction-aware level relative to the actual fill, advance after a clean pass, and exit only after the selected weakening mode and confirmation profile trigger around the zone. HVN variants must qualify node strength; TLV2 variants must qualify line score, confirmations, and age. These are simple comparison branches, not generic fallbacks or scored-confluence aliases.

## Parameter Rules

1. Use ordered `IntParameter`/`DecimalParameter` ranges when ordering matters.
2. Use integer multipliers for coarse regular percentages or distances.
3. Use categorical values only for unordered action/mode choices.
4. Keep combinations generally below about `10,000`, splitting only along coherent theory boundaries.
5. Every active Hyperopt parameter must affect a reachable branch.
6. Keep entry parameters static and out of the active Hyperopt surface.

## Validation

Before queueing regenerated files, verify:

1. Exact source coverage by parsing the stem after `_from_`.
2. No duplicate filenames or strategy class names.
3. Exactly one local `IStrategy` class per file.
4. No imports from another Sieve strategy.
5. Module compiles, imports, and instantiates.
6. `EXIT_FAMILY` matches the filename.
7. Required indicator target/invalidation columns are populated before callbacks use them.
8. The queue contains every active file exactly once and excludes archives.
9. The source has valid completed evidence before it is described as Sieve3-complete; generated file count alone is not completion.
