# Reviewer R1 Objective Alignment Review

Date: 2026-05-29

Scope: review of the worker outputs against `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md` and `C:\FreqTradeStuff\ai_guidance_docs\objective_examples.md`.

Reviewed files:

1. `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md`
2. `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md`
3. `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md`
4. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md`
5. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md`

## 1. Summary Verdict

1. Overall alignment: partial but directionally strong.
2. The worker outputs correctly move away from broad opaque FreqAI and toward source-detail coverage, clean windows, named trader hypotheses, controls, ablations, and promotion gating.
3. The outputs do not yet fully satisfy the revised objectives because the coverage/window artifacts are still too coarse to be used as final gates, the FreqAI pilot claim is under-evidenced inside the reviewed outputs, and multi-timeframe confluence remains mostly a placeholder.
4. No objective should be marked 100% complete.

## 2. P1 Findings

### 2.1 P1 - Coverage windows can be misread as usable confluence because coverage is not separated from active, nonzero, and source-usable evidence

1. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:6` defines coverage as non-null/present rows and active range as at least one nonzero numeric signal.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:16` marks `context_gkg_documents` as covered with 54,579 coverage rows but only 215 active nonzero rows, ending on 2020-01-09.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:15` through `:19` marks multiple `gdelt_only` candidate windows in 2021-2026 even though the report notes those windows are based on coverage, not active/nonzero signal.
2. Why it matters:
   - The current objectives require exact active date range, nonzero coverage range, missing-data gaps, and clean windows before interpreting tests.
   - A window with non-null cached columns but no active source signal can be mistaken for a valid context window or a quiet-context regime.
3. Fix recommendation:
   - Split every source-detail status into at least `present_coverage`, `active_nonzero`, and `usable_for_hypothesis`.
   - For each clean window, include required blocks, present rows, active rows, inactive rows, missing rows, and whether inactive rows mean true quietness or unavailable/no-signal data.
   - Downgrade or relabel GKG-dependent windows unless the hypothesis explicitly allows inactive GKG as a tested quiet/missing-data condition.

### 2.2 P1 - The FreqAI pilot candidate is overclaimed without an attached preflight record in the reviewed outputs

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:106` through `:108` says `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h` is the only current pilot-level FreqAI candidate.
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:216` through `:222` repeats the pilot/pilot-watch status and cites strong direct-test evidence or threshold sweep promise.
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:57` through `:58` requires a queued experiment preflight record with date range, source-detail groups, active rows, nonzero rows, missing rows, low-coverage rows, future-violation rows, and max source age.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:10` says full confluence has no overlap because `context_btc_etf_flows` and `context_google_trends` are too sparse.
2. Why it matters:
   - The plan's own gate says promotion requires source-detail coverage, exact threshold settings, rows, positives, controls, baselines, and clean-window class. Those details are not included with the pilot claim in the reviewed outputs.
   - This risks treating a narrow macro/context result as queue-ready before proving which context source-detail blocks were actually usable.
3. Fix recommendation:
   - Change the status to `conditional pilot candidate pending preflight` until a queue preflight artifact names exact thresholds, clean-window class, source-detail blocks, rows, positives, controls, baseline deltas, and source ages.
   - Attach or reference the exact direct-test and threshold-sweep rows used as evidence.
   - If Google Trends or ETF flows are not required for the pilot, explicitly state that the run is `context-source-specific`, not full confluence.

### 2.3 P1 - Clean-window report does not list the exact source-detail block set per candidate window

1. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:5` says windows are rows where every listed source-detail block has coverage, but the table at `:10` through `:22` only gives window type labels.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:27` and `:28` describe `structure_orderbook` windows with price, Bybit linear, and Bybit inverse notes, but do not list whether `orderbook_spot`, VP, TLV2, BOS/CHoCH, and pattern geometry are required and satisfied.
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:43` through `:58` requires full confluence, structure+orderbook, context, and orderbook windows to be source-detail explicit.
2. Why it matters:
   - The revised objective is not just to find any overlap; it is to prevent invalid or partial coverage from being silently tested as confluence.
   - Without the exact block list, a downstream test can accidentally use a source family not guaranteed by the window.
3. Fix recommendation:
   - Add columns to the clean-window report: `required_source_detail_blocks`, `optional_blocks_present`, `excluded_blocks`, `active_nonzero_required_rows`, `low_coverage_rows`, `future_violation_rows`, and `max_source_age`.
   - Split `structure_orderbook` into explicit variants, for example `structure_orderbook_spot`, `structure_orderbook_linear_inverse`, and `structure_orderbook_all_venues`.

## 3. P2 Findings

### 3.1 P2 - Objective audit is not reconciled with the new worker B coverage/window artifacts

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md:38` through `:43` still lists source coverage map and clean windows as missing work.
   - `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md:138` says the dedicated source-detail coverage map and clean-window report are still the top gap.
   - Worker B created `source_coverage_audit_20260529_worker_b.md` and `clean_testing_windows_20260529_worker_b.md`, but those are not referenced in the audit.
2. Why it matters:
   - The audit is supposed to map objectives to current evidence. If it omits the new evidence, orchestrators may repeat work or incorrectly treat the coverage reports as nonexistent.
3. Fix recommendation:
   - Update the audit in a future worker pass to cite both Worker B reports as partial evidence.
   - Keep the gap wording, but change it from "produce a dedicated report" to "harden the report with active/nonzero/source-type/window-specific usability fields."

### 3.2 P2 - Source coverage report lacks source type and source-detail usability classifications required by the objectives

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md:28` through `:33` requires exact active date range, nonzero coverage range, missing gaps, timestamp safety, and whether each source is live-collected, historical backfill, or derived/cache data.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:10` through `:26` reports status, row counts, date ranges, active rows, and segments, but not source type or final usability by hypothesis class.
2. Why it matters:
   - `context_global_market_macro`, `context_topic_severity`, and `context_article_source_activity` can look broadly covered while representing different data-generation paths and availability assumptions.
3. Fix recommendation:
   - Add `source_type`, `artifact_origin`, `availability_rule`, `freshness_rule`, and `usable_window_classes`.
   - For derived/cache blocks, state which upstream source-detail blocks they depend on.

### 3.3 P2 - Multi-timeframe confluence is mostly named but not operationalized

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md:8` through `:9` makes multi-source, multi-timeframe trader-readable behaviour the end goal.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:3` and `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:3` scope the artifacts to the 1h snapshot.
   - `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md:57` and `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:122` mention multi-timeframe confirmation only as an ablation concept.
2. Why it matters:
   - The examples require higher timeframe resistance/support, path-to-next-level, and multi-timeframe agreement. Current plans could still pass a nominal confluence gate while only testing 1h behaviour.
3. Fix recommendation:
   - Add explicit source-detail blocks for higher-timeframe structure, such as 4h/1d VP, TLV2 levels, BOS/CHoCH state, range boundaries, and acceptance/rejection state.
   - Add clean-window checks for those higher-timeframe derived columns and require a `minus_multi_timeframe` ablation only when those columns are actually present.

### 3.4 P2 - Hypothesis redesign is good as a design document, but it is not yet a registry-quality hypothesis artifact

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md:1` states the file is design plan only and does not claim implementation.
   - The hypothesis packs use candidate components and candidate labels, for example `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md:81` through `:96` and `:124` through `:139`.
   - `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md:42` through `:50` requires exact source-detail blocks, columns, setup logic, trigger logic, target label, pass/fail threshold, controls, and ablations.
2. Why it matters:
   - The plan preserves trader nuance, but it cannot yet satisfy the objective that each hypothesis must be testable and promotable without ambiguity.
3. Fix recommendation:
   - Convert each pack into registry entries with exact column names, thresholds, labels, setup mask, trigger mask, required clean-window class, controls, and ablation list.
   - Keep the prose theory, but bind it to concrete feature and label definitions before any new direct tests.

### 3.5 P2 - Context-heavy hypotheses still need source-specific variants to avoid collapsing unavailable context into broad context labels

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md:117` through `:118` requires GDELT, GKG, live news/web, macro/ETF/global for breakdown/crash detection.
   - `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md:330` through `:336` requires multiple context source-detail blocks for macro/news stress.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md:10` shows no full-confluence overlap because ETF flows and Google Trends are too sparse.
2. Why it matters:
   - If the registry keeps a single "context stress" hypothesis, the system may either block all context research or accidentally run on partial context while using full-context language.
3. Fix recommendation:
   - Define separate variants such as `gdelt_gkg_stress_breakdown`, `live_news_stress_breakdown`, `macro_only_liquidity_stress`, and `etf_trends_diagnostic_only`.
   - Each variant should have its own clean-window class, required blocks, freshness rules, and verdict language.

## 4. P3 Findings

### 4.1 P3 - The reports should state "not complete" outcomes more explicitly

1. Evidence:
   - `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md:8` through `:12` gives a clear partial summary, but the worker B reports do not include an objective-level verdict.
2. Fix recommendation:
   - Add a short footer to future worker reports: `objective_status = partial`, `safe_to_test = yes/no/conditional`, and `safe_to_promote = no/conditional`.

### 4.2 P3 - The `covered` status label is too broad for sparse or derived blocks

1. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:12` marks `context_article_source_activity` as covered despite only 112 active nonzero rows.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md:18` marks `context_topic_severity` as covered and active across all rows, which may be valid but needs upstream source interpretation.
2. Fix recommendation:
   - Use clearer statuses such as `covered_active`, `covered_mostly_inactive`, `too_sparse`, `derived_requires_upstream_check`, and `metadata_only`.

## 5. What The Worker Outputs Do Satisfy

1. The audit file exists separately from the canonical objective list and broadly maps objective status, evidence, missing work, risk, and readiness.
2. The hypothesis redesign plan preserves the trader-confluence examples better than a single-indicator or single-source design. The packs include structure, orderbook, volume, context, controls, ablations, and failure modes.
3. The FreqAI readiness plan correctly blocks broad all-feature modelling and defines strong promotion gates.
4. The source coverage and clean-window reports are valuable first-pass artifacts and correctly show that full confluence is not currently available when all required context blocks are included.

## 6. Recommended Fix Order

1. First, harden the coverage and clean-window reports so they distinguish present, active, nonzero, source-usable, and quiet/missing conditions.
2. Second, reconcile `objective_audit.md` with the new Worker B reports and downgrade the "missing coverage report" gap to "coverage report needs hardening."
3. Third, require a preflight artifact before any FreqAI pilot queue, especially for `macro_context_liquidity_stress_breakdown`.
4. Fourth, convert the hypothesis redesign packs into concrete registry entries with exact columns, thresholds, windows, controls, and ablations.
5. Fifth, add real multi-timeframe structure blocks or explicitly mark multi-timeframe confirmation as not yet implemented.

## 7. Remediation Re-check - 2026-05-29

1. Scope:
   - Re-reviewed only `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md`, `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md`, `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_review_fixed.md`, and `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.md`.

2. P1.1 status - resolved for the reviewed artifacts:
   - The fixed source report now defines usable rows separately from coverage and active rows, and states sparse/zero-filled context blocks require usable source activity unless a source-specific availability flag exists: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_review_fixed.md:8`.
   - `context_gkg_documents` is no longer treated as generally covered for clean windows; it is marked `too_sparse` with only 215 usable rows: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_review_fixed.md:17`.
   - The fixed clean-window report removes GKG from `gdelt_only` windows and isolates GKG as a `conditional_sparse_source_candidate`: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.md:15` through `:20`.

3. P1.2 status - resolved as an overclaim; implementation gate still open:
   - The FreqAI plan now calls `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h` a conditional pilot candidate only and blocks queueing until a preflight artifact proves thresholds, clean-window class, source-detail blocks, active/usable rows, positives, controls, baseline deltas, and timestamp-safety counts: `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:106` through `:107`.
   - The current promotion section repeats that the candidate is pending preflight and blocked from queue execution: `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md:217` through `:221`.
   - Remaining work is correctly downgraded to creating the preflight artifact, not a claim that the candidate is already queue-ready.

4. P1.3 status - partially resolved:
   - The fixed clean-window table now includes `required_blocks` and `required_block_usable_rows`, including explicit full-confluence requirements and sparse blockers: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.md:8` through `:10`.
   - The `structure_orderbook` windows now explicitly require price, VP, TLV2, BOS/CHoCH, pattern geometry, spot orderbook, Bybit linear, and Bybit inverse: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.md:11` through `:12`.
   - It remains partial because candidate rows still do not include per-window low-coverage rows, future-violation rows, max source age, optional blocks, or excluded blocks. The objective audit correctly keeps this as work to enforce in direct-test/FreqAI preflight paths: `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md:41` through `:43`.

5. Overall remediation verdict:
   - P1.1: resolved.
   - P1.2: resolved as overclaim; preflight artifact still required before any queue.
   - P1.3: partially resolved; exact block sets are present, but full preflight-grade window diagnostics are still open.
