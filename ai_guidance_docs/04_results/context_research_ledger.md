---
doc_status: active
default_read: routed
owner: agent
purpose: Compact durable ledger for context, orderbook, and FreqAI research handoff notes.
do_not_use_for: Full historical run logs or generated report contents.
last_rebuilt: 2026-06-11
---

# Context / Orderbook / FreqAI Research Ledger

## Use When

Update this file only when a task materially changes or validates:

1. context/news/GDELT/GKG/web/global source data,
2. orderbook source data or feature conditioning,
3. timestamp/lookahead alignment,
4. FreqAI research inputs or result interpretation,
5. source coverage/readiness status,
6. resolved/open data gaps.

Do not copy full generated reports here. Reference exact generated report paths and keep the summary trader-readable.

## Current High-Level State

1. News/GDELT/GKG/web/global/context is parked by default because coverage, formatting, and timestamp-ready normalized features are incomplete.
2. Isolated source windows may be usable only when source-specific readiness proves coverage, availability timing, and completeness.
3. Orderbook is partially usable for validated windows and is most useful as targeted confluence, risk, invalidation, target-zone, and failure-state evidence, not as a broad generic filter.
4. Custom indicators, sieve-derived entries, structure/VP/TLV2/BOS/CHoCH/pattern geometry, price action, and volume pressure remain the strongest usable inputs.
5. Sieve is the active Hyperopt/discovery path. FreqAI/orderbook/context work should stay parked until the user explicitly routes it into a new objective or into Sieve as a specific proven input source.

## Update Template

Append compact entries in this shape:

```text
## YYYY-MM-DD - <short topic>

- Scope:
- Source/data touched:
- Snapshot/window:
- Timestamp/lookahead checks:
- Result in trader language:
- Key metrics, if any:
- Artifact paths:
- Verdict: accepted / parked / rejected / blocked / needs rework
- Next action:
```

## Open Issues Carried Forward

1. Do not treat missing news/context source rows as quiet/no-news unless source availability proves the feed was operating.
2. Do not treat aggregate GDELT/GKG rows as final trader-readable normalized facts.
3. Do not use downloaded/imported time as market availability unless it is the conservative `available_at` for the source.
4. Do not carry orderbook book state, zone state, rolling features, or persistence through raw archive gaps.
5. Do not judge full confluence until each included source-detail block has usable timestamp-safe coverage.

## 2026-06-11 - Non-Sieve Exit/Risk Branch Retired

- Scope: Recent non-Sieve exit/risk and overlay-ablation branch.
- Result in trader language: The branch added only narrow value beyond Sieve and did not justify the token, context, and maintenance cost.
- Verdict: retired as an active objective. Use Sieve as the primary system for entries, exits, position adjustment, and risk Hyperopt batches. Preserve FreqAI/confluence/orderbook feature builders and discoveries only as separate later overlay candidates after solid Sieve strategies exist.

## 2026-06-15 - Historical News/Context Overlay Restart

- Scope: Reopened parked historical news/context work as a Sieve-adjacent overlay research path, not a non-Sieve strategy lane.
- Source/data touched: GDELT/GKG SQLite audit, context feature DB audit, Bybit historical orderbook feature coverage, OHLCV coverage check, quality-gate scripts.
- Snapshot/window: Main usable windows are `2020-01` to `2020-10`, `2021-03` to `2022-07`, GDELT-only `2022-08` to `2022-12`, and GDELT plus Bybit orderbook `2023-01-18` to `2023-06-30`.
- Timestamp/lookahead checks: Existing `context_features_1h` has no detected future-source violations in audited windows, but missing/source-age flags must be used. Silver GDELT/GKG tables and story clusters are empty.
- Result in trader language: First-pass research can test whether bad news, relief, quiet-news regimes, and orderbook-confirmed stress improve Sieve entry confidence, fakeout risk, crash continuation, exit urgency, and stake/stop decisions. Article-level story nuance is blocked until raw/silver GKG is repaired.
- Key metrics, if any: GDELT aggregate coverage is broad from 2020 onward; GKG aggregate is best in `2021-03` to `2022-07`; Bybit orderbook overlap begins `2023-01-18`; live RSS/web/global is not historical 2020-2023.
- Artifact paths: `ai_guidance_docs/04_results/news_context_overlay_restart_plan_20260615.md`; quality manifests under `user_data/research_news_data/context_features/reports/gdelt_gkg_quality_*news_context*.json`.
- Verdict: accepted for source-masked aggregate direct tests; article/story-level GKG nuance parked; final trading promotion blocked until direct tests and clean-window validation.
- Next action: Build source-readiness masks and direct-test the listed trader concepts before any FreqAI validation.

## 2026-06-15 - Historical News Overlay Exploratory Leads

- Scope: Source-masked aggregate GDELT/GKG theory tests through end-2022.
- Source/data touched: `gdelt_hourly_features`, `gdelt_gkg_file_features`, BTC 1h futures OHLCV, generated frozen analysis frames under the historical news overlay reports folder.
- Snapshot/window: `w1_2020_jan_oct`, `w2_2021_mar_2022_jul`, and `w3_2022_aug_dec_gdelt_only`.
- Timestamp/lookahead checks: Tail future labels fixed to remain unknown instead of false negatives; source coverage separated from source activity; model feature screen excludes parse/source/debug columns and requires improvement over shuffled and price-only controls.
- Result in trader language: Several useful overlay behaviours appeared: relief bounce, bad-news breakdown continuation, attention/compression expansion, high-attention breakout follow-through, and bad-news absorption/strength.
- Key metrics, if any: Latest batch produced `56` direct-test lead rows across all three windows. Examples: W2 bad-news breakdown continuation lift `+0.108`; W2 attention-compression downside lift `+0.179`; W1 attention breakout not overextended lift `+0.141`; W3 bad-news absorbed support lift `+0.183`.
- Artifact paths: `ai_guidance_docs/04_results/historical_news_overlay_results_20260615.md`; `user_data/research_news_data/context_features/reports/historical_news_overlay/historical_news_overlay_theory_results_batch3_refined_w2_w1_w3.csv`.
- Verdict: exploratory_signal, not promoted.
- Next action: Join these overlay states to Sieve entry/export rows and score by entry family before any FreqAI promotion.

## 2026-06-26 - Live Formatted Snapshot FreqAI Hypothesis Smoke Batch

- Scope: User-approved non-Sieve FreqAI feature-discovery run on the strict live aligned snapshot.
- Source/data touched: `live_context_orderbook_aligned_1h_latest.parquet` with BTC 1h futures OHLCV, safe context/news/global features, Binance recent orderbook features, market breadth, and derived confluence/state columns.
- Snapshot/window: Main context/market batch `20260508-20260605`; orderbook/cross-source batch `20260515-20260522` because strict orderbook-ready rows are sparse.
- Timestamp/lookahead checks: Used the v2 aligned snapshot validation with duplicate pair/date `0`, source future violations `0`, unsafe context/orderbook numeric cells `0`; GKG unavailable rows remain masked.
- Result in trader language: FreqAI found exploratory risk-overlay signal in context/news and market breadth for next-24h drawdown risk, plus market-breadth support for short-term downside/breakdown ranking. Orderbook/cross-source ran only on a small strict-ready slice and is too sparse for a strong verdict.
- Key metrics, if any: Context/news large-drawdown AUC `0.670` versus price-control `0.332`; market-breadth large-drawdown AUC `0.620` versus `0.332`; market-breadth breakdown-success AUC `0.761` versus `0.641`; orderbook breakout-success AUC `0.518` versus `0.406` on `168` rows.
- Artifact paths: `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_032201/formatted_freqai_hypothesis_metrics.csv`; `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_032247/formatted_freqai_hypothesis_metrics.csv`.
- Verdict: exploratory_signal for context/news and market-breadth risk overlays; orderbook/cross-source needs more strict-ready data before interpretation.
- Next action: If continuing this branch, test the promising context/news and market-breadth drawdown/breakdown overlays on a longer clean window or against approved Sieve exports; do not promote from this smoke batch alone.

## 2026-06-26 - Refined Live Snapshot FreqAI Best-Leads Pass

- Scope: Refined the best smoke-batch leads into source-family profiles: compact context, topic-risk context, crypto-native context, context activity, compact market breadth, raw market breadth, and market volatility.
- Source/data touched: Same strict live aligned BTC 1h snapshot and existing FreqAI bridge/runner; no strategy promotion.
- Snapshot/window: Main refinement `20260508-20260605`; validation checks `20260410-20260508`, `20260108-20260205`, and `20251016-20251113`.
- Timestamp/lookahead checks: Used the already validated strict formatted snapshot. One attempted parallel validation collision at `formatted_freqai_hypothesis_20260626_033339` is excluded from evidence because both runs shared the same timestamped directory; runner run IDs now include microseconds to prevent recurrence.
- Result in trader language: The big context/news drawdown-risk result appears real for the recent stress slice, but the older comparison windows mostly had GDELT-style archive context rather than live news/web/global coverage. Do not treat those older checks as valid live-media validation. The more repeatable non-live-media leads from that pass were market-volatility as a short-risk/reward overlay and raw market breadth as a downside/breakdown warning.
- Key metrics, if any: Recent-window large-drawdown AUC: crypto-native context `0.690`, topic-risk context `0.677`, price-only `0.332`. Across eligible windows, market-volatility short-reward-before-danger 24h was positive in `3/4` AUC windows with average top-minus-bottom event separation `0.202`; raw-market-breadth breakdown-success was positive in `2/4` windows with average top-minus-bottom separation `0.131`.
- Artifact paths: `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_032908/formatted_freqai_hypothesis_metrics.csv`; `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_033151/formatted_freqai_hypothesis_metrics.csv`; `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_033520_526159/formatted_freqai_hypothesis_metrics.csv`; `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_033624_096196/formatted_freqai_hypothesis_metrics.csv`; compact cross-window file `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/refined_best_results_cross_window_summary_20260626.csv`.
- Verdict: accepted as exploratory refined leads for market-volatility short-risk and raw-breadth breakdown overlays; live-media context drawdown verdict superseded by the later live-window-only test.
- Next action: For live media/news/web/global, use only windows where the live media block is actually present and include short `1h`/`2h`/`4h` targets.

## 2026-06-26 - Live Media Short-Horizon FreqAI Rerun

- Scope: Reran the live media/news/web/global investigation using only the actual live media window and fast targets.
- Source/data touched: Strict live aligned BTC 1h snapshot; live media means news + web + global only. GDELT/GKG were excluded from live-media profiles. Live Binance orderbook was tested separately only on the narrow overlap.
- Snapshot/window: Main live-media FreqAI run `20260516-20260611` with `7` train days and `2` backtest days; live-media source rows scored on `624` ready BTC hours. Live-media-plus-Binance-orderbook check `20260511-20260516`, scored on only `93` orderbook-ready rows.
- Timestamp/lookahead checks: Used existing strict aligned snapshot and live-media-ready scoring scope. The earlier failed attempt `formatted_freqai_hypothesis_20260626_082631_106859` is excluded because its first training window predated the live media feed, causing all live-media targets to be masked.
- Result in trader language: Short-horizon tests look more useful than only `6h`/`24h`. Live media improved ranking for 1h long follow-through, 1h drawdown risk, 1h short reward/risk, 2h short reward/risk, and 4h upside bursts versus price-only. Orderbook overlap is too small for a real conclusion.
- Key metrics, if any: Main live-media rows `624`. Examples: `live_media` long-reward-before-danger 1h AUC `0.673` vs price `0.547`, AP `0.163` vs `0.065`; topic-risk large-drawdown 1h AUC `0.673` vs price `0.587`; activity large-upside 2h AUC `0.697` vs price `0.643`; topic-risk large-upside 4h AUC `0.530` vs price `0.375`.
- Artifact paths: `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_082732_953751/formatted_freqai_hypothesis_metrics.csv`; `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_082920_530798/formatted_freqai_hypothesis_metrics.csv`; compact summary `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/live_media_short_horizon_summary_20260626.csv`.
- Verdict: exploratory_signal for live-media fast overlays; live orderbook overlap insufficient.
- Next action: Keep live-media tests scoped to the live-media-ready window, add source-family ablations only inside that window, and do not promote until another live-data month confirms the short-horizon lead.

## 2026-06-26 - Live Media Tactical Shapes And VP Add-On

- Scope: Added tactical `1h`/`2h`/`4h` live-media trader-shape features and tested whether VP improves those short-horizon live-media predictions.
- Source/data touched: Strict live aligned BTC 1h snapshot; live news + web + global only for media; GDELT/GKG excluded. VP uses the existing project `complex_volume_profile` helper on OHLCV inside the FreqAI bridge.
- Snapshot/window: FreqAI comparison `20260516-20260611`, `7` train days, `2` backtest days, `624` live-media-ready scored rows.
- Timestamp/lookahead checks: Used existing strict aligned snapshot and live-media-ready scope. VP features are rolling OHLCV-derived columns from the same candle stream; no orderbook data was mixed into this run.
- Result in trader language: The tactical live-media layer is most useful as a short/risk overlay, especially when source bursts combine with volume expansion or risk/financial-stress topics accelerate. VP helps some long follow-through targets but weakens several of the strongest downside/risk rankings, so VP should be tested as a conditional support layer rather than assumed globally helpful.
- Key metrics, if any: Tactical shapes improved `short_reward_before_danger_4h` to AUC `0.640` vs price `0.539` and `large_drawdown_next_4h` to AUC `0.651` vs price `0.591`. Direct threshold checks found `source burst plus volume expansion over 2h` enriched `short_reward_before_danger_4h`/`large_drawdown_next_4h` about `3.16x` over baseline on `22` trigger rows. VP add-on improved `long_reward_before_danger_1h` AUC to `0.638` vs price `0.547`, but reduced `large_drawdown_next_4h` versus tactical no-VP.
- Artifact paths: Tactical rerun metrics `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_103509_783670/formatted_freqai_hypothesis_metrics.csv`; FreqAI importance run `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_103907_154173/live_media_trader_shapes_feature_importance_overall.csv`; threshold/confluence sweep `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_103907_154173/live_media_threshold_confluence_sweep_20260626.csv`; VP comparison `user_data/research_news_data/context_features/reports/formatted_freqai_hypothesis/formatted_freqai_hypothesis_20260626_105806_066500/formatted_freqai_hypothesis_metrics.csv`.
- Verdict: exploratory_signal for tactical live-media risk overlays; VP conditional/needs refinement.
- Next action: Refine only the strongest tactical candidates into named overlay concepts: source burst + volume expansion, financial-stress acceleration, risk-off 4h acceleration, crypto-attention high, and macro-policy shock. Do not promote until more live-media history is collected.

## Archive Reference

For old detailed history, open only when verifying evidence:

- `ai_guidance_docs/99_archive/original_uploaded_docs/history_logs/context_research_detailed_findings.md`
- `ai_guidance_docs/99_archive/original_uploaded_docs/reviews/review_code_data_correctness_20260529.md`
- `ai_guidance_docs/99_archive/original_uploaded_docs/reviews/review_objective_alignment_20260529.md`
