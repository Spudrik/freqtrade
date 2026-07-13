# Reviewer R2 Code/Data Correctness Review - 2026-05-29

1. Scope

- Reviewed code:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_column_dictionary.py`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py`
- Reviewed reports:
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.csv`
- Ownership respected: only this review file was written.

2. Top Findings

2.1. P1 - Clean GDELT windows can be false-clean because zero-filled/non-null GKG rows are counted as source coverage

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:262`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:268`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:271`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv:7`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.csv:7`
- Evidence:
  - `context_gkg_documents` is marked `covered` with `54579` coverage rows, but only `215` active nonzero rows.
  - All active GKG rows are `2020-01-01T01:00:00+00:00` through `2020-01-09T23:00:00+00:00`.
  - The top `gdelt_only` clean window is `2023-11-02T12:00:00+00:00` through `2025-03-27T09:00:00+00:00` with `12262` rows.
  - Read-only snapshot check found `0` GKG nonzero rows inside that candidate window.
- Why this matters:
  - The audit treats `context_present & any(required block column is not null)` as coverage. For blocks that are zero-filled after source absence, non-null zeros are being treated as valid source coverage.
  - This can certify a `gdelt_only` window as clean even when one required block has no real activity/source rows in that period.
- Concrete fix recommendation:
  - Root-cause fix the context builder/audit contract by adding or using per-source availability flags such as `ctx_gkg_present`, `ctx_gdelt_present`, `ctx_article_present`, `ctx_google_trends_present`, and `ctx_btc_etf_present`.
  - For source-count/event-count families, require a source-specific availability/completeness flag or raw source row/file count, not merely non-null feature values.
  - Recompute clean windows using source-specific coverage masks. Do not use active nonzero alone as the coverage definition because zero can be a valid event count when source ingestion is known to be present.

2.2. P1 - Window timestamp-safety notes claim checks that the masks do not enforce

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:260`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:261`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:262`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:263`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:481`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:486`
- Evidence:
  - `coverage_mask_for_block()` uses `ob_*_present > 0` for orderbook blocks and `context_present > 0` plus non-null fields for context blocks.
  - It does not explicitly reject rows where `*_source_future_violation > 0`, `*_low_coverage > 0`, or `max_source_available_at > date`.
  - `timestamp_note_for_window()` states that orderbook windows exclude low-coverage/future rows and context windows require availability timestamps not later than candle date.
- Why this matters:
  - The current worker-b data shows zero present-and-future rows and zero present-and-low-coverage rows in the snapshot, so this does not currently invalidate those specific orderbook windows.
  - The code is still unsafe as an audit: if a future snapshot violates these conditions, the window can remain `candidate` while the note says it was filtered.
- Concrete fix recommendation:
  - Make timestamp rules part of the actual masks, not just report text.
  - For orderbook masks: `present & ~low_coverage & ~source_future_violation`, and if source max timestamp columns exist, assert `source_max_ts <= date`.
  - For context masks: require source-specific present/availability flags and explicitly reject `context_source_future_violation`; if `max_source_available_at` exists, require `<= date`.
  - Add audit columns for `excluded_future_rows`, `excluded_low_coverage_rows`, and `excluded_late_available_at_rows`.

2.3. P2 - Source-detail taxonomy misclassifies real activity/source-count columns as availability metadata

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py:91`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py:92`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py:107`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv:3`
- Evidence:
  - Columns such as `ctx_unique_source_count_6h`, `ctx_source_group_count_24h`, `ctx_war_geopolitics_source_count_24h`, `ctx_rates_source_count_24h`, and `ctx_gkg_source_count_24h` are classified as `context_availability_metadata`.
  - The cause is the broad `"source_" in lower_column` check before the article/source activity rules.
- Why this matters:
  - Source-count features are source activity signals, not metadata. Misclassification makes source-detail coverage and column dictionary summaries misleading and can remove important source-detail columns from the intended required block.
- Concrete fix recommendation:
  - Restrict availability metadata to actual metadata names: `context_present`, `context_row_present`, `*_available_at`, `*_source_min_ts`, `*_source_max_ts`, `*_source_future_violation`, and similar diagnostics.
  - Classify `unique_source_count`, `source_group_count`, and topic-specific `*_source_count_*` as `context_article_source_activity` or the relevant topic/source family.
  - Add a small taxonomy regression test with representative context columns.

2.4. P2 - Direct tests use broad family eligibility instead of source-detail clean windows

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:169`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:173`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:175`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:179`
- Evidence:
  - Context eligibility only requires `context_present`.
  - Orderbook eligibility requires any venue via `conf_ob_venue_count_present` or OR across venue present flags.
  - It does not intersect tests with the source-detail blocks actually used by a hypothesis or with the clean-window audit output.
- Why this matters:
  - A context hypothesis can be scored on rows where the specific context source family is absent/zero-filled.
  - An orderbook hypothesis can be scored with partial venue coverage even if the hypothesis/readout implies multi-venue evidence.
  - This can inflate row counts and make weak evidence look stable.
- Concrete fix recommendation:
  - Derive per-hypothesis required source-detail groups from component columns and require those masks during direct tests.
  - Add `eligible_source_blocks`, `eligible_window_start`, `eligible_window_end`, and per-block coverage counts to the direct-test output.
  - For orderbook hypotheses, decide explicitly whether "any venue", "all venues", or "minimum N venues" is the intended test condition and encode that in the hypothesis metadata.

2.5. P2 - Model ablation holdouts are not forward-time-safe

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:330`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:335`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py:336`
- Evidence:
  - The model ablation loop holds out each month, but trains on all other months, including future months relative to early holdouts.
- Why this matters:
  - This is acceptable only as a same-dataset ablation diagnostic. It is not a timestamp-safe validation design and can overstate model evidence if reported like a backtest-style validation.
- Concrete fix recommendation:
  - Use walk-forward/month-forward splits for any performance claims.
  - If leave-one-month-out remains, label it explicitly as non-causal ablation only and exclude it from promotion/watchlist decisions.

2.6. P3 - Active nonzero ranges are useful diagnostics but easy to overread as source availability

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:274`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:278`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv:2`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv:7`
- Evidence:
  - `context_article_source_activity` is `covered` for `54579` rows but active only `112` rows.
  - `context_gkg_documents` is `covered` for `54579` rows but active only `215` rows.
- Why this matters:
  - Nonzero activity is not equivalent to source coverage, but a very low active ratio beside very high coverage is a red flag for zero-filled missing data or a sparse signal family.
- Concrete fix recommendation:
  - Add status/caution levels for `active_nonzero_ratio` far below `coverage_ratio`, especially when active range does not overlap candidate windows.
  - For every candidate window, report per-required-block active row count inside that window.

2.7. P3 - `structure_present` is reported as `non_signal_metadata` with no coverage, which is confusing

- References:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py:48`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:268`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py:269`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.csv:18`
- Evidence:
  - `structure_present` is classified as `structure_availability`.
  - `is_signal_column()` excludes `_present`, leaving no useful columns, so the audit reports `structure_availability` as no coverage.
- Why this matters:
  - Required structure sub-blocks still use `structure_present` as their base mask, so this is not corrupting the required windows directly.
  - It does make the audit table look like structure availability is missing.
- Concrete fix recommendation:
  - Treat availability groups separately from signal groups in the audit and report their present-row counts, or suppress non-required availability groups from coverage status rows.

3. Read-Only Checks Run

- Loaded AGENTS.md and all requested code/report files.
- Parsed the worker-b source coverage CSV and clean testing window CSV.
- Parsed the frozen parquet snapshot read-only to compare non-null rows, nonzero rows, and candidate-window overlap for GKG/article/Google Trends/ETF source families.
- Checked orderbook present rows against low-coverage and future-violation flags in the current snapshot; no present rows overlapped those flags in this snapshot, but the audit code still does not enforce the exclusion itself.

4. Overall Assessment

- The generated full-confluence report correctly refuses a full-confluence window due to sparse Google Trends and ETF coverage.
- The largest correctness risk is still false confidence in narrower clean windows, especially `gdelt_only`, because the coverage logic treats zero-filled context feature rows as valid source coverage without source-specific availability.
- The direct-test code should not be used for promotion decisions until eligibility is intersected with source-detail-specific clean masks and model ablations are clearly separated from causal walk-forward validation.

5. Remediation Re-Check - 2026-05-29

5.1. Scope

- Re-checked only:
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py`
  - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_feature_taxonomy.py`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_review_fixed.csv`
  - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.csv`
- Did not re-check direct-test/model-ablation code because it was outside the requested file scope.

5.2. Status By Prior Finding

- P1 `2.1` false-clean GDELT/GKG windows: Resolved for the reviewed artifacts.
  - `source_coverage_audit.py:305` adds a `usable_mask_for_block()` and `source_coverage_audit.py:374` now builds windows from `combined_usable_mask()`.
  - `context_gkg_documents` is now `too_sparse` with `usable_rows=215`, and the old broad GDELT+GKG clean candidate is replaced by separate `gdelt_only` and `gdelt_gkg_documents` definitions at `source_coverage_audit.py:64` and `source_coverage_audit.py:68`.
  - `clean_testing_windows_20260529_review_fixed.csv` marks `gdelt_gkg_documents` as `conditional_sparse_source_candidate` for only `2020-01-01T01:00:00+00:00` to `2020-01-09T23:00:00+00:00`, so the previous 2023-2025 false-clean GKG overlap is no longer present.

- P1 `2.2` timestamp-safety notes not enforced: Resolved for source coverage/window masks in reviewed files.
  - Orderbook coverage now filters present rows by `low_coverage`, `source_future_violation`, and `source_max_ts <= date` at `source_coverage_audit.py:282` through `source_coverage_audit.py:288`.
  - Context coverage now rejects `context_source_future_violation` and checks `ctx_max_source_available_at <= date` at `source_coverage_audit.py:289` through `source_coverage_audit.py:296`.
  - Remaining caveat: this re-check did not validate upstream construction of those timestamp columns, only that the audit now enforces them when present.

- P2 `2.3` source-detail taxonomy misclassifies source-count activity as availability metadata: Partially resolved.
  - The broad `"source_" in lower_column` metadata rule was removed/restricted at `trader_confluence_feature_taxonomy.py:91` through `trader_confluence_feature_taxonomy.py:100`.
  - Confirmed examples now classify correctly: `ctx_unique_source_count_6h -> context_article_source_activity`, `ctx_source_group_count_24h -> context_article_source_activity`, `ctx_war_geopolitics_source_count_24h -> context_topic_severity`, and `ctx_gkg_source_count_24h -> context_gkg_documents`.
  - Still open edge case: `ctx_rates_source_count_24h` and `ctx_inflation_source_count_24h` still classify as `context_other`, as reflected by `context_other` sample columns in `source_coverage_audit_20260529_review_fixed.csv`. If these are intended topic/source activity columns, add aliases such as `rates`, `inflation`, and similar legacy topic names to the taxonomy or normalize column names upstream.

- P2 `2.4` direct tests use broad family eligibility instead of source-detail clean windows: Not re-checked; still open pending review of `trader_confluence_direct_tests.py`, which was not in this remediation scope.

- P2 `2.5` model ablation holdouts are not forward-time-safe: Not re-checked; still open pending review of `trader_confluence_direct_tests.py`, which was not in this remediation scope.

5.3. Current Artifact Assessment

- `full_confluence` remains invalid with sparse/missing usable context blocks, which is correct for the reviewed coverage state.
- `recent_live_news_context` is now too short and explicitly marks `context_article_source_activity` as sparse, which avoids false confidence.
- The reviewed remediation materially improves the audit/report side. The remaining actionable item in-scope is the taxonomy alias gap for rates/inflation source-count columns.
