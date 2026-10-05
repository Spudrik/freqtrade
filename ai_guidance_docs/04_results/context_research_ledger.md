---
doc_status: active
default_read: routed
owner: agent
purpose: Compact durable ledger for context, orderbook, and FreqAI research handoff notes.
do_not_use_for: Full historical run logs or generated report contents.
last_rebuilt: 2026-09-12
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
4. Causal custom-indicator levels, structure/VP/TLV2/BOS/CHoCH/pattern geometry, OHLCV, price action, and volume pressure are the primary current research inputs.
5. Objective 02b now preserves reaction-zone discovery while testing bounded event-scoped direction through slow background, major events, BTC/ETH/broad-market leadership, coin-group response, local modification, and post-event ranges. Unrestricted every-candle direction, trade actions, and strategy promotion remain parked.
6. Timestamped US prediction-market expectations are now source-ready for headline/core CPI and payroll/unemployment. Fed decision expectations are reconstructable, but actual decision surprises are too rare for a historical direction test. These are market-implied distributions, not economist surveys.
7. The first retained expectation result is agreement between headline and core CPI surprises. It remains a historical lead requiring future whole-release confirmation and must later be combined with market readiness, Bitcoin confirmation, and reaction locations rather than treated as a complete explanation.
8. A whole-episode recheck of seven older aggregate-news interactions retained one limited directional lead: unusually high news attention during a high-volume non-extended Bitcoin breakout. It improved historical direction but did not reliably identify reaction size, has one small contradictory component comparison, and cannot be confirmed until the missing source pipeline is rebuilt or replaced.
9. The 2026-09-19 ownership reassessment below reviews all 65 existing checklist records at inventory/report level, reconciles later level results, and adds five missing handoff records. This is not a full raw-data reproduction. Read that entry before treating early retained labels or the Stage-1 coverage preflight as current performance evidence.

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

## 2026-08-08 - Sieve3 Event-Reaction And Interaction Round One

- Scope: Frozen post-Sieve3 cohort analysis using exact raw entry triggers, resolved exit backtests, direct controls, and a four-layer FreqAI ablation (`price`, named levels, event identities, event-plus-level interactions).
- Source/data touched: `63` handover entries, `189` resolved exit solutions, BTC/ETH/SOL futures OHLCV, existing five-timeframe generic level caches, and exact handover parameter surfaces. Three legacy entry sources were parked because their own informative `merge_asof` code mixes `datetime64[ms]` and `datetime64[us]`.
- Snapshot/window: Raw event cache `2022-12-01` through `2026-06-28`; FreqAI development `2024-04-01` through `2025-04-01`; validation `2025-04-01` through `2026-04-01`; late `2026-04-01` through `2026-06-28` opened once only for the selected co-event lead.
- Timestamp/lookahead checks: Entry events use candle-close decision time, targets use future-only OHLCV, event caches have zero duplicate/missing event timestamps, and rolling FreqAI training is chronological. Direct controls are same-pair/month/ATR-regime ordinary candles plus `168h` shifted placebos.
- Result in trader language: Hyperopt profitability did not translate into a broadly repeatable directional entry reaction. No entry passed the robust signed-return gate. Some triggers consistently identify changed path shape or volume/activity. Whole-profile FreqAI layers were not promotion-ready: event identity was near-neutral overall, level context had narrow return-ranking value but failed full pair/window portability, and the large event-plus-level model generally hurt performance.
- Key metrics, if any: `60/63` entry sources analysed; `57,316` active trigger rows and `20,867` onsets; `10,823` executed exits; `1,258` controlled entry comparisons; `1,048` exit-reaction rows; `4` FreqAI profiles and `41,800` side-aware score rows. The strongest conditional lead was `mtf_std_daily_prior_high_breakout_long_1h`: inside those events, event context produced median 1-ATR up-before-down AUC about `0.62-0.63`, roughly `+0.028` to `+0.036` versus price control where sample support existed. Its co-occurrence with the H4 supply-breakout/local-break entry had strong validation uplift but only three contradictory late-holdout co-events.
- Artifact paths: `user_data/research_news_data/context_features/sieve3_event_reaction/round1/catalogue_run_summary.json`; `.../direct_entry_reaction_results.csv`; `.../direct_exit_reaction_results.csv`; `.../freqai/freqai_reaction_scores.csv`; `.../freqai/freqai_portability_summary.csv`; `.../freqai/prior_high_coevent_followup_summary.csv`; detailed chronological notes in `.../round1_research_log.md`.
- Verdict: needs rework / exploratory leads only. No entry, exit, FreqAI profile, or interaction is promoted or ready to merge.
- Next action: Retest the prior-high/H4-supply co-event on more independent data; run exact co-event ablations; simplify the level-context hypothesis and recheck BTC validation; test side-specific exit timing against earlier/later/no-exit counterfactuals; repair the three legacy datetime merges only with explicit strategy-file approval.

## 2026-08-12 - Market-First Reaction-Zone Objective And Generation 0A Audit

- Scope: Replaced the Sieve3-dependent event-validation objective with market-first, non-directional reaction-zone discovery and began the read-only Generation 0A infrastructure/data audit.
- Source/data touched: Existing Binance futures OHLCV inventory; current VP, pivot, BOS/CHoCH, TLV2, and geometry contracts; existing structural caches; the general level-reaction direct study; current FreqAI strategy/profile/queue paths; routed source-readiness summaries.
- Snapshot/window: The frozen project top-ten set has continuous `1h`, `4h`, `8h`, and `1d` files. The common `1h` surface can start after AVAX warm-up and runs through `2026-07-19`; `4h`/`8h`/`1d` extend to early August. Most `3d` files contain at least one multi-candle gap and are not accepted into the initial common surface without a separate gap decision.
- Timestamp/lookahead checks: Raw candle timestamps require close-time conversion. Clean pivots and TLV2 expose causal availability. Existing reaction caches shift aligned level state before testing the contact candle, but the old study's use of already-shifted VP `prior_poc`/`prior_vah`/`prior_val` creates a deliberately older level than the latest settled profile; the new atlas must keep those representations distinct. Full per-family causality remains a Generation 0A gate.
- Result in trader language: There is enough OHLCV across ten coins and four common timeframes to run the new programme without Sieve events. The existing three-coin study is reusable evidence about implementation shape, not a sufficient baseline: it covers only VA edge, prior POC, HVN, and one TLV2 rank; uses fixed percentage touch/cluster bands; observes only one to four hours; pools important summaries; and lacks the complete random, matched-state, near-miss, price-shift, and same-density controls. Its FreqAI path primarily scores a down-versus-up target and therefore violates the current direction boundary.
- Key metrics, if any: Existing cache coverage is BTC/ETH/SOL, five source timeframes, `27,049` one-hour rows per coin from `2023-03-01` to `2026-04-01`. The old direct run contains `63,050` event rows, `26,488` first-touch rows, and `48,868` stale-placebo rows. These counts describe available infrastructure only and are not promoted findings.
- Artifact paths: `ai_guidance_docs/01_objectives/objective_02b_market_reaction_zone_discovery.md`; `user_data/Custom_Launcher/research/context_features/structural_feature_cache.py`; `user_data/Custom_Launcher/research/context_features/general_exit_level_reaction_study.py`; `user_data/research_news_data/context_features/reports/general_exit_level_reaction/level_reaction_meta.json`.
- Verdict: accepted stage change; existing pipeline components require a dedicated non-directional reaction-zone implementation rather than relabelling the direction/exit study.
- Next action: Finish the causal output matrix, freeze the top-ten/four-timeframe/horizon/control/chronological surface, and implement the smallest dedicated direct-atlas pipeline. Keep `user_data/Indicators/**` unchanged; it is clean at baseline commit `3dd331803a739f442c9ec98615c7ca45ee37801a`.

## 2026-08-13 - Evidence-Triggered One-Minute Directional Replay Lane

- Scope: Added a bounded later-stage microscope lane inside Objective 02b. Repeated direction-neutral price/volume reactions at causal levels, clusters, or time-knowable significant price references may be queued and occasionally replayed at `1m` resolution to investigate immediate direction and path.
- Source/data touched: Guidance only. The planned lane may use source-timeframe-scaled targeted `1m` OHLCV slices, full level-construction history at coarser timeframes, synchronized crypto-wide state, causal lower-timeframe indicators, and source-ready orderbook/trade-flow/external context where genuine historical overlap exists.
- Snapshot/window: No market run launched by this documentation update. Initial batches normally use `6-12` independent episodes. Dense replay defaults are `24h/12h` pre/post for a `1h` anchor, `72h/48h` for `4h`, `7d/4d` for `8h`, and `30d/14d` for `1d`, plus the level's full source-timeframe construction history. One whole-pattern doubling is allowed only when a direction-neutral boundary audit shows truncation.
- Timestamp/lookahead checks: Candidate selection must be frozen from direction-neutral parent evidence before signed `1m` paths are inspected. Every qualifying episode is retained, repeated paths are de-duplicated, retrospective-only price references cannot be treated as causal predictors, and post-contact pressure/flow may not be used for a first-contact prediction.
- Result in trader language: The new lane allows lower-timeframe investigation of why a known reaction area broke, rejected, stalled, or continued without allowing attractive chart examples or a broad direction search to consume the main reaction-zone programme.
- Key metrics, if any: This is a method/scope decision, not market evidence. One direct aligned-path pass precedes any FreqAI ladder; one replay batch and normally one worker may run while a main generation is unfinished.
- Artifact paths: `ai_guidance_docs/01_objectives/objective_02b_market_reaction_zone_discovery.md`; `ai_guidance_docs/02_rules/reference_freqai_event_reaction_research_method.md`; `ai_guidance_docs/02_rules/theory_freqai_regime_direction_level_reaction.md`; `ai_guidance_docs/02_rules/rules_goal_mode_iteration_control.md`.
- Verdict: accepted bounded scope addition; no trading promotion.
- Next action: Continue the main repeatable reaction-zone review. Populate the replay queue only from frozen direction-neutral patterns, audit existing `1m` coverage, download only missing merged episode ranges through approved Freqtrade launcher tools, and return every microstudy result to the joint generation review before any descendant test.

## 2026-08-13 - Breadth-First Branch Queue And Joint Success Targets

- Scope: Refined Objective 02b so useful ideas inspired by interim or final results are preserved without interrupting the frozen main batches. Added a rolling provenance-linked candidate queue, a minimum five-route investigation portfolio, an exact meme-cohort contract, and explicit joint reaction-and-direction targets.
- Source/data touched: Guidance and ledger only. No market data, model, indicator, strategy, or run was changed or launched.
- Snapshot/window: Not applicable to market evidence. Future meme batches use the `10` most traded eligible meme coins on the frozen venue/market type, ranked before outcomes by median daily quote turnover over the preceding `30` completed UTC days.
- Timestamp/lookahead checks: Interim ideas remain provisional; outcome-inspired coins or groups cannot be counted as confirmation; every later batch freezes membership, data, hypotheses, controls, and criteria before execution.
- Result in trader language: New indicators, mathematics, timeframes, convergence patterns, level definitions, coins, and contextual sources may all become later tests, but none may hijack an unfinished generation. At least five genuinely different routes must be tested. A directional call succeeds jointly only when the declared abnormal volume reaction occurs and the predicted direction/path is also correct on the same call.
- Key metrics, if any: `55%` joint success on untouched chronological data is the minimum acceptable lead floor; `65%` is the main target. Both require adequate independent support, meaningful issued-call coverage, and improvement over relevant simple comparators. The end-state wording uses at least three materially independent retained indications from at least two route families; five or more useful mechanisms remains preferred.
- Artifact paths: `ai_guidance_docs/01_objectives/objective_02b_market_reaction_zone_discovery.md`; `ai_guidance_docs/02_rules/reference_freqai_event_reaction_research_method.md`; `ai_guidance_docs/02_rules/theory_freqai_regime_direction_level_reaction.md`; `ai_guidance_docs/02_rules/rules_goal_mode_iteration_control.md`; `ai_guidance_docs/00_project_control/objectives_master.md`.
- Verdict: accepted guidance refinement; not market evidence and not trading promotion.
- Next action: Finish the frozen initial targets, record valid result-inspired ideas in the queue, review them together at the generation gate, and freeze balanced later batches without exceeding the five pre-authorized branch layers.

## 2026-08-21 - Broad Reaction-Source Portfolio And Pairwise Freeze

- Scope: Completed the seven-sibling broad market-reaction portfolio, including direct controls and matching FreqAI source ladders for ten established markets and the separately frozen top-ten meme cohort.
- Source/data touched: Causal 1h event cache; 1h/4h/8h/1d calculated levels; recent OHLCV/standard indicators; BTC/ETH/cohort crypto state; historical BTC order-book context; source-ready aggregate GDELT; frozen BTC, established-alt, smart-contract, and meme views.
- Snapshot/window: Normal development from 2021-06 through 2023-12 with 2024-01 to 2025-03 and 2025-04 to 2026-03 validation; meme development/validation used its shorter frozen chronology through July 2026. Contacts were separated by more than the four-hour maximum reaction horizon.
- Timestamp/lookahead checks: Both cohorts completed 84/84 profiles and 72/72 comparisons with equal prediction keys, zero duplicate prediction rows, causal 72-hour-old placebos, terminal commands, and no profit or signed future-direction targets.
- Result in trader language: Calculated areas repeatedly helped estimate next-hour volume/range and four-hour absolute movement/dwell. Recent OHLCV state, especially local volume/pressure, added a smaller increment; BTC state added another small volume/range increment. Many direct timeframe/news/order-book relationships became redundant when level-only and source-only models were compared fairly. Pressure-change magnitude did not retain broad support.
- Key metrics, if any: Adding calculated levels reduced equal-coin volume error about 9-11%, range about 4-6%, four-hour excursion about 5-6%, and dwell about 4-5% across both cohorts/periods. Local volume/pressure added roughly 4-5% volume improvement; BTC state added roughly 1-3% volume improvement. The strict cross-cohort review retained eight target rows but interprets them as a few related mechanisms, not eight independent edges.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/generation6_review/g6_joint_review_20260821a/g6_joint_review.json`; readable detail in `user_data/research_news_data/context_features/market_reaction_zones/relationship_investigation_register.md`; frozen next batch at `user_data/research_news_data/context_features/market_reaction_zones/generation6_review/g7_frozen_pairwise_batch.json`.
- Verdict: accepted direction-neutral reaction components; no trading promotion. Timeframe clusters need exact width/dependency attribution. Average news/order-book additions are parked, with only two bounded conditional challenges retained.
- Next action: Run the complete ten-sibling Generation 7 portfolio. Each full interaction must beat level-only, each one-component model, the two sources without level, and stale controls before any Generation 8 refinement is frozen.

## 2026-08-21 - Generation 7 Pairwise Reaction Interactions

- Scope: Completed the frozen pairwise interaction portfolio for normal and meme cohorts. Each complete model combined a current calculated level with two named conditions and was challenged by level-only, both one-condition models, both conditions without level, stale level, stale first condition, and stale second condition on identical rows.
- Source/data touched: Reused terminal Generation 6 causal level/OHLCV/BTC/ETH/order-book/GDELT caches; added research-only prior-range geometry, prior-4h-Volume-Profile geometry, and true two-family cluster geometry. A cluster required overlapping zones or a gap no larger than `0.10 ATR`.
- Snapshot/window: Normal and meme development plus the same two frozen chronological validation periods used by Generation 6. Exact support was frozen before targets were opened; all `18` branch/cohort cells passed.
- Timestamp/lookahead checks: `144/144` profiles and `126/126` comparisons completed; prediction keys were identical, duplicate predictions were zero, specialty stale inputs were at least `72h` old, and profit/signed direction remained absent.
- Result in trader language: The one strict complete interaction was current level + local participation + volatility/compression for normal-coin next-hour volume. It improved roughly `0.8%` over the strongest level-plus-participation model, `4-6%` over level-only, and about `10%` over both context blocks without a current level or with a stale level. Local participation + volatility/compression for next-hour range and participation + trend/momentum for next-hour volume were provisional in both cohort views. BTC context, cross-timeframe agreement, and order-book/BTC produced narrower normal-group provisional leads.
- Negative result: No meme target was strict. Specific prior-range interaction, prior-4h-Volume-Profile dampening interaction, explicit two-family clusters, meme ETH context, and aggregate GDELT failed the complete ladder. This parks their descendants without erasing earlier standalone-level direct evidence.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/generation7_review/g7_joint_review_20260821a/g7_joint_review.json`; plain detailed account in `user_data/research_news_data/context_features/market_reaction_zones/relationship_investigation_register.md`; frozen next batch at `user_data/research_news_data/context_features/market_reaction_zones/generation7_review/g8_frozen_attribution_batch.json`.
- Verdict: one small control-resistant direction-neutral reaction-estimation lead, several provisional attribution leads, and explicit parked routes; no trading promotion.
- Next action: Run outcome-blind causal representation/support preflight for all eight frozen Generation 8 attribution siblings, then finish the complete sibling portfolio before any Generation 9 multi-source or `1m` descendant.

## 2026-08-21 - Generation 8 Attribution And Three-Seed Confirmation

- Scope: Completed all eight frozen Generation 8 attribution siblings, then froze every seed-42 route/target/group that beat all point controls and repeated only that exact set with deterministic seeds 17 and 73.
- Source/data touched: Existing causal Generation 8 normal/meme caches; 1h/4h/8h/1d calculated-level lineage; current OHLCV participation, volatility/compression, trend/momentum, BTC context, room geometry, historical BTC order-book, and contact-state blocks. Canonical indicators and Freqtrade core were unchanged.
- Snapshot/window: Reused the two frozen chronological validation periods for each cohort. The initial runs completed 404 profiles/385 comparisons; confirmations completed 200 profiles/174 comparisons. Total evidence was 604 profiles and 559 comparisons.
- Timestamp/lookahead checks: All 19 cells passed outcome-blind support. Stable feature bands came from development predictors. Causal stale copies were 72 hours old; shuffled copies were deterministic within-period nonself matches. All prediction keys were equal, duplicate rows were zero, and profit/future signed direction were absent.
- Result in trader language: Near calculated price areas, current relative volume and the duration of local participation states repeatedly improved prediction of next-hour trading volume. For normal coins both were strict across established-alt, full-normal, and smart-contract groups. The same two relationships were provisional across the frozen top-ten memes. First-contact state also added provisional volume information in both cohorts. Smaller normal-only range leads remained for compression, trend duration, oscillator displacement, and selected level properties.
- Key metrics, if any: Normal relative volume and participation duration each passed 18/18 control-period-seed checks in three groups. Relative volume reduced error by at least about 2.7-2.9% in every comparison, with roughly 3.0-3.3% median reduction; duration reduced it by at least 1.3-1.4%, with roughly 1.6-1.8% median reduction. Meme relative volume was point-positive 18/18 and strict 17/18. Of 61 frozen group candidates, 6 were strict, 15 provisional, and 40 failed seed replication.
- Negative result: BTC horizon, eight-hour incremental location, historical BTC order-book attribution, absolute volatility, meme width/history, and most occupancy/retest variants did not remain seed-stable. They are not repaired by choosing favourable thresholds or model seeds.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/generation8_review/g8_three_seed_joint_review_20260821a/g8_three_seed_joint_review.json`; plain detail in `user_data/research_news_data/context_features/market_reaction_zones/relationship_investigation_register.md`; frozen confirmation at `user_data/research_news_data/context_features/market_reaction_zones/generation8_review/g8_frozen_seed_confirmation_20260821a.json`.
- Verdict: two strict attribution relationships within one local-participation route family, three cross-cohort provisional corroborations, several smaller normal-only leads, and three branches without seed-stable support. No direction, profit, entry, exit, or trading promotion claim.
- Next action: Freeze the complete Generation 9 limited three-source and evidence-triggered one-minute sibling sets together. Test refined participation combinations plus selected independent provisional mechanisms, and select every one-minute episode by a causal direction-neutral rule before inspecting signed paths.

## 2026-08-21 - Generation 9 Limited Multi-Source And One-Minute Review

- Scope: Completed and jointly reviewed both outcome-blind frozen Generation 9 sibling families: seven limited three-source FreqAI questions and twelve evidence-selected `1m` reaction/path replays.
- Source/data touched: Existing causal Generation 8 normal/meme caches; exact calculated-level lineage; standard multi-timeframe OHLCV indicators; usable historical BTC order-book fields; twelve exact `1m` intervals across twelve pairs. Missing TRX, XRP, and WIF ranges were repaired through isolated exact-range downloads and atomic merges on `D:`.
- Snapshot/window: The FreqAI family reused both frozen chronological validation periods and seeds `42`, `17`, and `73`. The `1m` family sampled one independent episode from every normal/meme x development/early-validation/late-validation x `1h`/`4h` cell, with source-timeframe-scaled context windows.
- Timestamp/lookahead checks: Both families were frozen before outcomes. All model prediction keys matched and duplicates were zero. The `1m` selection used actual causal level contact, current relative-volume state, and unsigned next-hour volume only; future direction did not select episodes. Final `1m` coverage was `74,160/74,160` expected rows with zero gaps, duplicates, invalid rows, or overlap conflicts.
- Result in trader language: No complete three-source model survived every leave-one-out, stale-input, shuffled-input, period, and seed check. Three small normal-only routes remained provisional, but meme results did not reproduce them. In the selected `1m` sample, six of twelve contacts met the stricter price-and-volume reaction rule; paths split into five breakouts, six rejections, and one tie, so the level itself supplied no direction. Nine pre-contact direction methods all stayed below the `55%` joint floor.
- Key metrics, if any: Models completed `210/210` profiles and `189/189` comparisons; group routes were `0` strict, `3` provisional, and `16` failed. The best `1m` joint score was `4/12 = 33.3%`; normal reaction was `4/6`, meme reaction `2/6`. State-balanced controls were only `2` matched no-level, `1` stale-level, and `1` shifted-level example, so no causal level claim is allowed.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/generation9_review/g9_joint_review_20260821a/g9_joint_review.json`; detailed plain-language account in `user_data/research_news_data/context_features/market_reaction_zones/relationship_investigation_register.md`; frozen queue at `.../generation9_review/g9_joint_review_20260821a/g9_result_inspired_queue.csv`.
- Verdict: limited three-source family rejected as a stable combined mechanism; three normal-only observations parked; `1m` lane diagnostic and insufficient; no direction, profit, entry, exit, or trading promotion.
- Next action: Freeze Generation 10 untouched confirmation for the two strict Generation 8 single-source participation mechanisms—current relative volume and participation-state duration—keeping them separate. Do not expand the `1m` lane from this result.

## 2026-08-21 - Generation 10 Untouched Participation Confirmation

- Scope: Completed the final authorized Objective 02b layer. Retested the two strict
  Generation 8 single-source mechanisms separately: current relative volume and the
  duration of the current participation state, both for unsigned next-hour volume near
  calculated areas.
- Source/data touched: Existing causal calculated-level and local OHLCV caches for the
  frozen ten normal markets. The cache schema was repaired before evidence generation
  to expose required standard target columns; configurations still trained only the
  frozen next-hour-volume target. Canonical indicators and Freqtrade core were unchanged.
- Snapshot/window: `2026-07-20` to `2026-08-05` and `2026-08-05` to `2026-08-20`,
  across established-altcoin, full-normal, and smart-contract-platform groups and seeds
  `42`, `17`, and `73`. These are target-and-mechanism-unopened dates, not globally
  pristine history, because an unrelated absolute-excursion target used the same dates.
- Timestamp/lookahead checks: Questions, groups, controls, periods, and seeds were frozen
  before volume outcomes. All `24/24` repaired-run profiles and `18/18` comparisons
  completed with equal prediction keys, zero duplicate rows, and no profit or future
  signed-direction target. The first technical run failed four profiles before training
  and was excluded.
- Result in trader language: Neither current relative volume nor participation duration
  remained a broad timing-specific predictor. All six declared group routes failed.
  Relative volume often beat the plain and `72h`-old models but rarely beat shuffled
  timing with strict uncertainty; participation duration generally lost to its shuffled
  version.
- Conditional observations: Against the plain model only, both mechanisms were positive
  in every active-market slice (`18/18` combined), but only `4/18` combined ordinary-
  activity slices were positive. BNB alone was point-positive for both mechanisms against
  all controls, periods, and seeds. Both findings are outcome-inspected questions needing
  a new untouched controlled test, not promoted edges.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation10_review/g10_terminal_review_20260821a/g10_terminal_review.json`; detailed
  account in `user_data/research_news_data/context_features/market_reaction_zones/`
  `relationship_investigation_register.md`; frozen future choices in the terminal
  review's `g10_result_inspired_queue.csv`.
- Verdict: Generation 10 terminal; Objective 02b unresolved at the authorized horizon.
  No direction, profit, entry, exit, or strategy promotion. Broad versions of both
  participation claims are parked rather than retuned.
- Next action: Request new user authorization before any further branch. The preserved
  options are a full-control conditional-activity replication, a BNB integrity and
  predeclared-peer replication, or a separately scoped reaction-selected direction
  objective.

## 2026-08-22 - Generation 11 Multi-Horizon Reaction Attribution

- Scope: Reframed the FreqAI market-reaction question around a combined unsigned
  price-and-volume reaction and future relative volume at `1h`, `2h`, `4h`, and `8h`.
  Tested level identity/timeframe, cluster geometry, local activity/volatility, local
  trend/momentum, wider crypto state, usable historical BTC order-book context, aggregate
  news context, and a restrained combination.
- Source/data touched: Causal Generation 6 event caches for ten established markets and
  the frozen top-ten meme cohort. The news route had zero eligible rows and was parked as
  not tested. Canonical indicators and Freqtrade core were unchanged.
- Snapshot/window: Established-market validation ran across 2024-01 to 2026-03; meme
  validation used its shorter 2026 chronology through 2026-07-13. These were reused
  frozen periods, so the result was initial attribution rather than later confirmation.
- Timestamp/lookahead checks: Predictor readiness was frozen before targets. Labels were
  purged for the maximum `8h` horizon at every rolling training boundary. All `62/62`
  profiles completed with identical comparison keys, zero duplicate prediction rows,
  causal `72h` placebos, and deterministic within-period nonself shuffles.
- Result in trader language: Local activity/volatility was the strongest cross-cohort
  lead for deciding whether a calculated-level contact became a material price-and-volume
  reaction. Trend/momentum, level identity, and wider-market state were smaller leads.
  Historical BTC order-book added nothing broadly, news could not be tested, and the
  large combination mostly failed its component-removal controls.
- Key metrics, if any: `10/56` route/target cells were strict in both cohorts and `24/56`
  more were point-positive in both. Strong activity-family balanced reaction accuracy
  was roughly `58-65%`, but this measured reaction only, not direction.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g11_freqai_initial_attribution/`
  `g11_initial_joint_review_20260822a/g11_joint_freqai_review.json`; detailed plain
  account in `relationship_investigation_register.md`.
- Verdict: useful initial reaction-attribution leads; no profit, direction, entry, exit,
  or trading promotion. The conflict with Generation 10 required a later chronology.
- Next action: Complete the frozen Generation 12 later-date stress test, component
  attribution, and restrained combinations before opening any descendant.

## 2026-08-22 - Generation 12 Later Chronology And Component Review

- Scope: Completed the later April-August 2026 normal-coin stress test and the separate
  normal/meme decompositions of activity, volatility, trend, oscillators, BTC/ETH leader
  state, wider-cohort state, pair-relative state, and four predeclared combinations.
- Source/data touched: Existing causal Generation 11 caches and exact-column research
  variants under `user_data/**`. No live databases, canonical indicators, upstream
  Freqtrade/FreqAI code, profit targets, or signed future-direction targets were used.
- Snapshot/window: The later lane used `2026-04-01` to `2026-07-01` and `2026-07-01` to
  `2026-08-21` as half-open normal-coin periods. Component attribution reused the frozen
  normal and meme validation chronologies and is therefore explanatory, not new temporal
  confirmation.
- Timestamp/lookahead checks: The full plan was frozen before outcomes; all `104/104`
  profiles and `135/135` comparisons completed. Prediction keys matched, duplicate rows
  were zero, all controls were terminal, and target purging covered the maximum `8h`
  forecast horizon.
- Result in trader language: Current local activity/volatility survived later chronology
  for the combined price-and-volume reaction at `1h`, `2h`, `4h`, and `8h`. It did not
  retain later future-volume strength beyond `1h`. Wider BTC/crypto state also retained
  smaller reaction information. Level identity and trend were provisional; the current
  geometry representation added nothing on later dates.
- Key metrics, if any: Activity/volatility reduced equal-coin reaction error by at least
  about `1.3-1.8%` against every simpler/stale/shuffled comparator, with positive gains
  in `54-57/60` coin/period/control checks per horizon. Balanced reaction accuracy was
  about `60-62%` on base rates of about `38-48%`. Reused-period decomposition found `17`
  route/target cells strict in both cohorts, led by participation/pressure, volatility/
  compression, BTC/ETH leader state, and RSI/MACD momentum. No combination was strict in
  both cohorts.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g12_joint_review_20260822a/g12_joint_review.json`; detailed plain account in
  `relationship_investigation_register.md`.
- Verdict: a credible reaction-only lead, not a directional or trading edge. The joint
  `55%` reaction-and-direction floor remains untested and unmet.
- Next action: Complete three new siblings before branching: matched ordinary/no-level
  and near-miss controls, timestamp-safe `4h`/`8h`/`1d` indicator context, and
  `12h`/`24h`/`48h` reaction paths. Keep order-book/news parked until timestamp-ready
  coverage exists.

## 2026-08-22 - Generation 13 Broad Controls And Time-Horizon Review

- Scope: Completed all three frozen siblings: genuine calculated-level contacts versus
  matched ordinary and near-miss controls; completed `4h`/`8h`/`1d` state beyond `1h`;
  and `12h`/`24h`/`48h` reaction and volume outcomes at structural levels.
- Source/data touched: Causal cached data for ten established coins, the later normal
  chronology, and the frozen top-ten meme cohort. No canonical indicators or upstream
  code changed; no profit or signed future-direction targets were used.
- Timestamp/lookahead checks: All `20` pair caches supported all `58` frozen profiles.
  All `87/87` FreqAI models completed, prediction keys matched, duplicate predictions
  were zero, and the maximum target horizon was purged at training boundaries. Direct
  controls were selected without future outcomes and every control timestamp coinciding
  with any tracked real contact was excluded.
- Result in trader language: Real calculated-level contacts reacted more often than
  equally busy ordinary timestamps and genuine near misses over `1h` through `4h`.
  Completed `8h` activity/volatility also improved `8h` reaction estimates beyond `1h`,
  stale, and shuffled controls in both normal chronologies and memes. Local activity at
  `4h` source levels had a provisional `48h` reaction lead, but the newest late-normal
  slice was too sparse and uncertain for strict confirmation.
- Key metrics, if any: Direct all-timeframe real-contact differences versus matched
  ordinary time were about `3.7-8.4` percentage points at `1h`-`4h`. The `8h` activity
  route's balanced reaction accuracy was about `61-63%` across all three cells on base
  rates around `44-48%`; its minimum error gain was `0.0043` with a positive `0.00033`
  weekly-block lower bound. The `4h`-level/`48h` lead had minimum point gain `0.0382`,
  but its joint lower bound was negative because of the sparse later-normal late slice.
- Negative evidence: Current `4h` and `1d` higher-timeframe state, higher-timeframe
  trend/momentum, most long source/target combinations, and the long volume targets did
  not survive the complete cross-window ladder.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/freqai/g13_joint_review_20260822a/g13_joint_review.json`; direct
  controls are under the sibling `direct_level_controls/g13_direct_controls_20260822a/`.
- Verdict: retained reaction-only evidence with meaningful alternatives rejected. No
  direction or trading claim; the unseen `55%` joint reaction-and-direction floor is
  still untested and unmet.
- Next action: Freeze one broad sibling batch covering label sensitivity, `8h` activity
  attribution, single-level/cluster interaction, wider-market conditioning, restrained
  interaction with the provisional `4h`-level/`48h` lead, and coin/time stability before
  any descendant is launched.

## 2026-08-22 - Generation 14 Reaction Robustness And Combination Review

- Scope: Completed one frozen broad sibling batch covering nine reaction definitions,
  completed-`8h` feature attribution, calculated-level identity, cluster/wider-market/
  oscillator interactions, `12h-48h` local-plus-`8h` combinations, coin/time stability,
  and external-source readiness.
- Source/data touched: Reused the causal Generation 13 normal and frozen top-ten meme
  event caches; materialized one outcome-blind combined context cache on `D:`. No
  canonical indicator, trading strategy, Freqtrade core, live process, or source data
  was changed.
- Snapshot/window: Original normal validation, later April-August 2026 normal chronology,
  and the frozen meme validation periods. The long meme cell was parked because all ten
  pairs had fewer than `30` pre-window training contacts.
- Timestamp/lookahead checks: All supported `89/89` FreqAI model commands completed on
  equal prediction keys with zero duplicate prediction rows. Stale controls were causal
  `72h`-old copies, shuffled controls stayed within period, and the coverage gate did not
  inspect future outcomes. Profit and future signed direction were absent.
- Result in trader language: Real calculated-level contacts remained more likely than
  matched ordinary times and near misses to produce joint unsigned price-and-volume
  activity over `1h-4h` under all nine nearby label definitions. Most added feature
  combinations were redundant or harmful. Current level family/timeframe identity gave
  only a very small `1h` improvement beyond local and completed `8h` activity; its
  uncertainty still crossed zero. The prior local-plus-`8h` `48h` suggestion failed.
- Key metrics, if any: Label robustness passed `11/12` point rows and `9/12` strict rows;
  the dependable cells were `1h`, `2h`, and `4h` across both cohorts and normal windows.
  The sole cross-cell FreqAI point lead had a minimum equal-coin probability-error gain
  of `0.00073` and minimum bootstrap lower bound of `-0.00313`, so it is not strict.
  Original-normal leave-one-coin-out checks were `60/60` positive; later normal and meme
  were each `58/60`, but no predeclared normal subgroup was uniformly stable.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combinations/freqai/g14_joint_review_20260822a/`
  `g14_joint_review.json`; direct label detail is in sibling folder
  `label_sensitivity/g14_label_sensitivity_20260822a/`.
- Verdict: Retain the short-window calculated-location effect; keep the `1h` identity
  increment only as a weak attribution lead; reject the tested broad bundles and park
  the failed long-horizon combination. Historical news remains unavailable and
  historical order book lacks the newest confirmation period.
- Next action: Freeze one broad batch that decomposes reaction behaviour, attributes
  level family/timeframe, tests approach and contact lifecycle, repairs explicit cluster
  composition, checks predeclared coin groups, and uses order book only on supported
  identical rows. No direction or trading claim is authorized by this result.

## 2026-08-22 - Generation 15 Reaction-Path And Attribution Review

- Scope: Completed and jointly reviewed seven outcome-blind frozen siblings across the
  ten established coins and frozen ten-meme cohort: path decomposition, level family,
  source timeframe, contact activity combinations, lifecycle/arrival, explicit cluster
  composition, and timestamp-safe historical BTC order book.
- Source/data touched: Reused the frozen Generation 6 event cache and causal Generation 8
  history features. Pair-level bulky matches were written on `D:`. No canonical indicator,
  strategy, Freqtrade core, live collector, or source data was changed.
- Snapshot/window: Original normal validation, later April-August 2026 normal chronology,
  frozen meme validation, and `1h`, `2h`, `4h`, `8h` future paths.
- Timestamp/lookahead checks: Matching used only current/past state, exact family/timeframe/
  approach where applicable, a `0.10 ATR` starting-distance caliper, and controls away
  from all real contact times. Contact-candle inputs were used only after that candle
  closed. Profit and future signed direction were absent.
- Result in trader language: Calculated areas repeatedly located more future volume,
  wider range, and more crossings. Strict general evidence was strongest over `1h-4h`.
  Absolute excursion, hit timing, dwell, and pressure did not repeat broadly, so this is
  an activity-location result rather than a bounce, breakout, or direction result.
- Level detail: Prior ranges, round numbers, and several volume-profile families retained
  short activity effects. `1h` and `4h` source levels were clearest; `8h` retained only
  short volume and `1d` did not complete the ladder. These were control comparisons, not
  head-to-head rankings and not evidence for automatic higher-timeframe precedence.
- Context result: No volume/range/pressure combination, lifecycle/arrival state, clean
  exclusive cluster composition, or supported order-book condition survived original
  normal, later normal, and meme cells. Overlapping cluster-any-presence differed in old
  normal and meme cells but failed later chronology and is not attribution-eligible.
- Key metrics: `383,679` matched location rows and `94,308` contextual contrast rows
  before horizon purges. General point leads were `12/32`; eight remained strict after
  weekly uncertainty and a maximum `50%` one-coin contribution rule. Future-volume
  differences were about `+0.17` to `+0.44` at `1h`, weakening to `+0.07` to `+0.19` at
  `8h`. No contextual row completed all three broad cells.
- Artifact path: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combinations/g15_broad_direct/joint_review/`
  `g15_joint_review_20260822a/g15_joint_review.json`.
- Verdict: Retain calculated areas as repeatable short-window activity locators. Do not
  claim direction, causation, profit, a universal confluence bundle, or the user's `55%`
  joint target. The target remains untested and unmet.
- Next action: Freeze one broad direction-neutral batch covering completed contact-candle
  matching, activity persistence, level density/coverage, head-to-head family/timeframe
  attribution, and local/wider-market regime modulation. A separate bounded,
  evidence-triggered `1m` replay may inspect direction on a frozen representative queue;
  the current objective does not authorize broad hourly direction modelling.

## 2026-08-22 - Generation 16 Completed-Contact, Density, And Bounded Direction Review

- Scope: Completed all seven frozen direction-neutral routes plus the separately frozen
  `1m` replay before joint review. Direct routes covered completed contact-candle
  matching, stale/shift controls, pre-to-post change, head-to-head family/timeframe,
  density/overlap, and local/wider/completed-`8h` context. FreqAI ran eight incremental
  profiles in original normal, later normal, and top-ten meme cells.
- Source/data touched: Reused causal Generation 6 and Generation 13 caches. Built an
  outcome-blind Generation 16 FreqAI support cache on `D:` before joining targets and
  downloaded only the five missing exact `1m` intervals for the frozen replay. No
  canonical indicator, upstream Freqtrade/FreqAI code, live collector, trading strategy,
  profit target, or source database was changed.
- Snapshot/window: Ten established coins across original and April-August 2026 normal
  chronologies, the frozen ten most-traded meme coins, retained `1h`/`4h`/`8h` source
  levels, and `1h`/`2h`/`4h` direction-neutral outcomes. The replay used `12` distinct-
  pair episodes and `15m`/`60m`/`240m`/`720m` horizons.
- Timestamp/lookahead checks: Direct matching used pre-contact state plus the completed
  contact candle only, with a `0.10 ATR` starting-distance caliper, horizon overlap
  purges, and maximum control reuse of three. FreqAI features/support were frozen before
  target materialization; all `24/24` full model commands completed with identical keys
  and zero duplicate predictions. The `1m` sample was selected without future reaction
  or direction outcomes.
- Result in trader language: After completed contact volume, range, and pressure were
  matched, real calculated areas still acted like busier price junctions. Broad normal-
  coin contacts had strictly more crossings through the area at `1h`, `2h`, and `4h`,
  more direction-neutral reaction at `1h`, and more future volume at `2h` than matched
  ordinary times and genuine near misses.
- FreqAI result: Continuous level density plus cluster geometry strictly improved future
  crossing-count estimates at `1h`, `2h`, and `4h` in original normal, later normal, and
  meme cells. Average relative error gains were about `5.2-6.9%` at `1h`, `5.0-6.5%` at
  `2h`, and `2.6-3.8%` at `4h`; positive coin-period rows were `60/60`, `59/60`, and
  `54/60`. Completed `8h` activity gave a point-level `4h` reaction lead with reaction-
  only accuracy around `61.9-68.4%`, but it was not a strict incremental result.
- Negative evidence: No family or source timeframe won head to head across both normal
  chronologies. No pre-to-post test proved that contact started the activity. No direct
  context combination or direct whole-question meme result survived. The full level-
  plus-wider FreqAI model did not beat every simpler component.
- Direction result: Best joint reaction-and-direction performance in the `1m` lane was
  `4/12 = 33.3%`; zero of `14` causal methods reached `55%`. Reaction-only accuracy and
  crossing prediction do not satisfy the user's joint target.
- Artifact path: `user_data/research_news_data/context_features/market_reaction_zones/`
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/`
  `joint_review/g16_joint_review_20260822a/g16_joint_review.json`.
- Verdict: Retain calculated areas as short-window traffic locators and retain continuous
  density/geometry as the first robust cross-market FreqAI attribution lead. Do not claim
  bounce, breakout, direction, causation, profit, or promotion. The `55%` joint floor and
  `65%` target remain unmet.
- Next action: Complete the jointly queued six-batch branch layer before any descendant:
  density/geometry decomposition, crossing semantics, reaction calibration, broader
  causal level sources, source-ready external-context regimes, and a bounded larger `1m`
  direction replay.

## 2026-08-23 - Generation 17 Broad Branch-Layer Review

- Scope: Completed and jointly reviewed the six frozen Generation 17 siblings before
  allowing any descendant. The batch decomposed density/geometry, separated post-contact
  path forms, calibrated reaction probability, tested a broad rational level-source
  atlas, audited external-source regimes, and ran a larger bounded `1m` direction replay.
- Source/data touched: Reused causal normal and meme event caches, generated a frozen
  `113`-definition level registry, stored approximately `10.35 million` source/control
  event rows on `D:`, and downloaded only the missing intervals needed to complete the
  frozen `1m` sample. No upstream Freqtrade/FreqAI code, canonical indicator, strategy,
  live collector, or trading configuration was changed.
- FreqAI integrity: All `54` full profiles completed with identical prediction keys and
  no duplicate keys. Four strict all-three-cell questions survived. Current density
  beat shuffled and causal-stale density for `2h` crossing count; proximity/width added
  to contact-only information for `1h` and `2h` crossing count. These remain
  direction-neutral traffic estimates.
- Direct path result: Actual level contacts showed more any-recross behaviour at
  `1h`/`2h` across normal subgroups and memes, broad support at `4h`, and repeated
  recrossing at `2h` in all-normal and meme scopes. Meme contacts separately showed
  more one-sided breakthrough. Controls matched causal contact state and included
  ordinary times and genuine near misses.
- Level-source result: Adaptive Volume Profile nodes gave the clearest portable crossing
  result. Rolling prior high/low boundaries gave the clearest portable unsigned-reaction
  result. A neighbouring group of `72h` VP HVN/LVN definitions survived rather than one
  isolated setting. The `303` strict result rows are related parameter, horizon, metric,
  and market-scope cells; they must be deduplicated and confirmed, not counted as
  independent discoveries. BTC contributed a disproportionate number of isolated rows.
- External-source status: source-ready orderbook rows were tested but did not support a
  broad retained claim. Historical news had zero ready rows for this surface, and no
  suitable non-crypto global-market feature block was available, so both were parked
  rather than filled with zeroes.
- Direction result: `34` causal methods were checked on `40` frozen episodes over
  `15m`/`60m`/`240m`/`720m`. The best joint reaction-and-direction result was
  `14/40 = 35%`; no method reached `55%`. A `66.7%` conditional-direction subset is not
  the target because it excludes failed/no-call cases.
- Verdict: Retain calculated areas as probabilistic activity-location evidence. Retain
  density/proximity, recross behaviour, adaptive VP nodes, and rolling high/low
  boundaries for a balanced confirmation batch. Do not claim direction, causation,
  profit, entry/exit value, or trading promotion.
- Next action: Treat density confirmation, reaction-form confirmation, rational
  level-source confirmation, market/context regimes, and coin/timeframe portability as
  one next sibling layer. Keep the direction descendant parked unless a later reaction
  gate and newly frozen data justify reopening it.
- Evidence:
  `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/`
  `generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/`
  `g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/`
  `joint_review/g17_joint_review_20260823a/g17_joint_review.json`.

## 2026-08-23 - Generation 18 Later-Block, Context, And Timeframe Confirmation

- Scope: Completed five active siblings and honestly parked the direction sibling before
  joint review. Direct work reused the frozen Generation 17 rational-level events on two
  later blocks per cohort. A separate source registry recalculated eight compact level
  definitions on completed `1h`, `4h`, and `8h` candles for every normal and meme pair.
- Integrity: All `20/20` direct pair cases and `20/20` multi-timeframe pair cases
  completed. The multi-timeframe calculation made a source value available only after
  its source candle closed. Results used equal-coin effects, both later blocks, control
  ladders, bootstrap uncertainty, minimum member counts, and one-coin dominance checks.
- Broad physical result: Adaptive VP and the combined VP/rolling-boundary surface showed
  more any-recross at `1h`/`2h`/`4h` and more repeated-recross at `2h`/`4h` than ordinary
  times and genuine near misses in both normal and meme groups. This is repeatable
  traffic at a location, not a directional call.
- Specificity limit: Requiring current coordinates to beat ordinary, near-miss, causal
  stale, and price-shifted versions narrowed the result to BTC VP crossings, meme
  rolling-boundary reaction, and one meme LVN crossing setting. The prior broad VP
  parameter plateau did not confirm across both cohorts.
- Context result: High cross-coin dispersion increased the level-versus-control extra
  future-volume response over `2h` across normal coins. High completed local
  volume/range activity increased the extra `1h` volume response inside the frozen
  smart-contract-platform group. Other strict context cells were mostly BTC-specific.
- Timeframe/cluster result: `15` higher-versus-`1h` and `15` cluster-versus-isolated rows
  passed their narrow head-to-head comparison, but none of those candidates also passed
  the complete artificial-location controls. Valid higher-timeframe precedence rows:
  `0`. Valid cluster-superiority rows: `0`.
- Source readiness: Orderbook had no common timestamp-ready rows in both later blocks.
  News and non-crypto global markets remained parked. Missing sources were not encoded as
  quiet, neutral, or zero.
- Verdict: Retain recross as the strongest direction-neutral market behaviour. Retain
  dispersion/local-activity volume modulation as small conditional leads. Keep exact
  level families market-labelled, and do not claim direction, higher-timeframe
  precedence, cluster superiority, causation, profit, or promotion.
- Next action: One balanced later layer is queued for exact-coordinate recross
  specificity, regime-plus-level ablations, market-specific source mechanisms, recross
  timing/onset, and one compact rational acceptance/anchored-zone extension. External
  sources and direction remain parked under current coverage/evidence.
- Evidence:
  `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/`
  `generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/`
  `g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/`
  `g18_broad_confirmation/joint_review/g18_joint_review_20260823a/g18_joint_review.json`.

## 2026-08-29 - Generation 25 One-Batch Joint Review

- Scope: Ran one frozen sibling batch and stopped before any descendant. Five routes were
  assessed together: three executable routes plus the daily anchored-VWAP and external-
  source routes that failed their outcome-blind coverage gates. No canonical indicator,
  upstream Freqtrade/FreqAI file, collector, strategy, or trading configuration changed.
- Convergence representation: Pooled all `60` predeclared round-number x rolling-price-
  distribution definitions before opening outcomes, removed duplicated timestamps, and
  equalized actual plus eight control samples inside each coin/period. `16/20` coins had
  at least ten rows per control in both periods. `80/100` market-scope questions had a
  complete ladder; none passed the full two-period ladder. The closest lead was meme-coin
  `2h` crossing count: `13/16` period/control checks passed (`12/16` strict), and every
  mean actual-minus-control effect was positive. It remains an incomplete fresh-
  confirmation lead, not a retained reaction result.
- Connected Volume Profile: Built a read-only research representation from the preceding
  `96` completed `1h` candles using `48` bins. Current-candle data was excluded. Connected
  HVN areas and LVN corridors were compared with equal-density one-bin nodes, matched
  random times, genuine near misses, `72h` stale zones, and zones shifted by `2 ATR`.
  About `85.3%` of actual events used multi-bin regions, so this was materially different
  from a point node. `140/200` questions had complete ladders; zero passed. Park the
  current connected-zone definition.
- Market-state FreqAI: Re-scored the already-completed market-state-only models against a
  training-median constant, within-pair shuffled training labels, a simple causal current-
  activity percentile, and the same activity measure delayed by `24h`. Four fixed model/
  seed cells and two chronological periods were required. `15/40` scope/target questions
  survived every cell: all eight unsigned `1h/2h/4h/8h` volume/range targets for the
  top-10 meme cohort and seven normal-coin scope/target rows. On retained cells, rank
  correlations were about `0.16-0.31`, top-versus-bottom activity separation about
  `0.12-0.25`, and the highest predicted quartile was actually above its training median
  about `58.9-69.2%` of the time.
- Interpretation boundary: The FreqAI result is a provisional answer to *when is the
  market likely to be unusually active?* It does not predict direction, attribute the
  activity to a price level, satisfy the joint reaction-and-direction target, or justify
  a trade. It reused the same chronology that motivated the follow-up and therefore needs
  genuinely later confirmation.
- Coverage findings: The fixed daily current-session VWAP centre question still needs two
  later `15-16` day OHLCV blocks. Historical/live external sources still do not supply two
  complete timestamp-safe overlapping blocks. Missing coverage was not filled with
  neutral values.
- Next questions, not launched: confirm the frozen market-state activity model on later
  normal and meme data; confirm the fixed pooled meme convergence question on later data;
  and revisit the already-selected daily VWAP question only after its coverage gate
  passes. No Generation 26 batch was started.
- Evidence:
  `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/`
  `generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/`
  `g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/`
  `g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings/g21_broad_siblings/`
  `g22_broad_siblings/g23_broad_siblings/g25_broad_siblings/joint_review/`
  `g25_joint_review_20260829a/g25_joint_review.json`.

## 2026-09-01 - Generation 26 Market-State Plus Exact-Level Interaction

- Scope: Ran one frozen, same-holdout exploratory batch asking whether exact current
  level contacts become more informative when the retained Generation 25 model expects
  unusually high future activity. This was a reaction-only test: it did not use signed
  direction, profit, entries, exits, or trading outcomes.
- Coverage gate: Updated the frozen `19` distinct-pair normal/meme surface at `1h`, `4h`,
  `8h`, and `1d`. All `80` required files were present with no duplicate timestamps and
  common `1h` coverage through `2026-09-01 21:00 UTC`. The two genuinely later blocks are
  still incomplete and cannot provide the planned fresh confirmation before
  `2026-09-20`.
- Outcome-blind level gate: Required adequate event/control support before opening
  reaction outcomes. Adaptive Volume Profile nodes and rolling VWAP deviation bands
  passed. Donchian boundaries, weekly pivots, and generic MA/Bollinger references did
  not pass the frozen support gate and were excluded rather than rescued by relaxing it.
- Test: Evaluated `59,584` real/control event rows across two level families, four
  unsigned reaction measures, `1h`/`2h`/`4h` horizons, and five market scopes. Exact
  contacts had to beat matched ordinary times, genuine near misses, `72h`-stale levels,
  and levels shifted by `2 ATR` in both fixed periods. The high-activity-versus-low-
  activity difference also had to beat the same complete control ladder.
- Result: Zero of `120` combined questions passed even the complete point-level gate;
  zero passed the strict gate. One location-only row passed (`1h` crossing count at BTC
  Volume Profile nodes), and two interaction-only rows passed (BTC Volume Profile volume
  response over `1h` and `4h`), but no question passed both halves. No multi-coin normal,
  smart-contract, established-alt, or top-10-meme scope retained the interaction.
- Verdict: Park this exact state-plus-location formulation. The existing activity model
  and location/re-cross evidence remain separate provisional ideas, but the current test
  does not show that a high activity forecast makes these exact levels more specific or
  more useful. Do not tune against this reused holdout. Confirm the frozen Generation 25
  activity model on genuinely later data first; reopen a combination only if that model
  confirms and a new trader-readable reason justifies it.
- Evidence:
  `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/`
  `generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/`
  `g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/`
  `g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings/g21_broad_siblings/`
  `g22_broad_siblings/g23_broad_siblings/g26_broad_siblings/state_location_interaction/`
  `g26_state_location_interaction_20260901a/g26_state_location_result.json`.

## 2026-09-03 - Untouched Three-Route Confirmation Batch Frozen

- Scope: Froze the three evidence-backed questions left by the Generation 25 review as
  one later-data batch: the OHLCV-only busy-market forecast, the pooled round-number x
  rolling-distribution meme crossing question, and the daily current-session VWAP
  crossing question. This executes the existing queue; it is not a new descendant of
  the failed Generation 26 state-plus-level interaction.
- Exact state-model surface: Preserved the `15` previously retained cohort/scope/target
  questions and `16` fixed candidate/shuffled-label profiles formed from two profile
  roles, four model/seed cells, and two cohorts. Constant, simple-current-activity, and
  causal `24h`-stale activity controls remain required.
- Exact direct surfaces: Preserved all `60` convergence definitions pooled without
  winner selection for top-10-meme `2h` crossing traffic, and the daily current-session
  VWAP centre calculated only through the previous completed hour for all-normal `8h`
  crossing traffic. Both retain their full component, density, ordinary-time,
  near-miss, stale, displaced, and historical-analogue controls as applicable.
- Fresh chronology: Both cohorts use `2026-08-20` to `2026-09-05` and `2026-09-05` to
  `2026-09-20` as half-open UTC blocks. Because the longest outcome is eight hours, the
  last required hourly candle opens at `2026-09-20 07:00 UTC`; checking only the block
  end would truncate the last responses.
- Outcome-blind gate on 3 September: All `20` cohort/pair cells (`19` distinct pairs)
  had their required training history, no missing files, no duplicate timestamps, and
  no internal hourly gaps. The common latest candle was only `2026-09-01 21:00 UTC`, so
  `0/20` cells passed the ending-data gate. No future reaction, signed direction, or
  profit values were opened.
- External-source status: Live web and global-market status files were still reporting
  without an error, while live news continued to report an SSL certificate failure and
  the orderbook status reported a lost connection. Status files alone do not prove
  usable historical overlap, so external inputs remain excluded from this confirmation.
- Next action: Keep the questions unchanged. After the required hourly data exists,
  build and freeze all three event/model support surfaces together, verify event and
  control counts, and only then open the two complete fresh outcome blocks.
- Support tooling prepared: Added one coverage-gated, outcome-blind support command for
  all three routes. It preserves the full frozen market-state/random-label feature
  contract, the one selected daily VWAP centre plus five controls, and all `60` pooled
  meme convergence definitions plus eight controls. The command checks training and
  prediction row minimums, same-period control counts, minimum eligible coins, causal
  source timestamps, and shuffled-label source availability before jointly freezing
  support. It does not calculate reaction outcomes, direction, or profit.
- Gate verification on 3 September: The real command refreshed timestamp coverage and
  correctly returned `waiting_for_complete_fresh_ohlcv`: `0/20` cells yet reached the
  required `2026-09-20 07:00 UTC` candle. It created the small waiting record on `C:` and
  no bulky `D:` support directory. Historical feature parity, causal-contact, period,
  matched-control, future-source rejection, and waiting-gate checks all passed
  (`12` focused tests including the existing freeze tests; Ruff also passed).
- Evidence:
  `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/`
  `generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/`
  `g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/`
  `g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings/g21_broad_siblings/`
  `g22_broad_siblings/g23_broad_siblings/g25_broad_siblings/fresh_confirmation/`
  `coverage/fresh_confirmation_coverage_20260903a/fresh_confirmation_coverage_result.json`
  and `support/fresh_confirmation_support_20260903a/`
  `fresh_confirmation_support_result.json`.

## 2026-09-03 - Frozen Confirmations Coverage-Parked; Event Hierarchy Layer 1 Started

- Scope: Closed the old waiting batch without changing or opening its outcomes, then
  began the authorized outcome-blind source and event-definition layer.
- Frozen confirmation disposition: The OHLCV busy-market forecast, pooled meme
  convergence crossing question, and daily current-session VWAP crossing question are
  now `coverage_parked_frozen_confirmation`. Their definitions and two half-open blocks
  remain unchanged and may be reopened only after the required candle at
  `2026-09-20 07:00 UTC` exists. They no longer block the new investigation.
- Source/data touched: Read only the existing frozen 1h trader-confluence parquet;
  no live collector database was queried or paused.
- Coverage result: The snapshot contained `55,971` rows and `27` source blocks.
  Aggregate GDELT activity and some global/crypto-market columns have long usable
  stretches, but article/source activity, GKG documents, Google Trends, and ETF-flow
  history are too sparse for broad confluence. No full-source overlap exists.
- Semantic limit: The `D:` GDELT export archive currently contains `1,871` files from
  March-July 2026 and the GKG raw folder has no files. Historical aggregate GDELT may
  support event activity/intensity but cannot by itself reconstruct old stories,
  expectations, or directional surprise. Those claims require official timestamped
  sources or must be parked.
- Verdict: Old confirmations coverage-parked intact. Layer 1 proceeds with official
  scheduled events, independently sourced unexpected events, causal timestamps,
  whole-event partitions, and explicit unsupported-source states before any market
  outcome is opened.
- Evidence:
  `user_data/research_news_data/context_features/reports/`
  `source_coverage_audit_event_hierarchy_layer1_20260903a.*` and
  `clean_testing_windows_event_hierarchy_layer1_20260903a.*`.
- Next action: Audit the existing remembered-event list for hindsight leakage, freeze
  inclusion and timing rules, and construct the first outcome-blind event catalogue
  plus matched controls.

## 2026-09-04 - Event Hierarchy Layers 1-5 Completed Or Coverage-Parked

- Question: Can scheduled events or causal aggregate-news activity identify unusually
  active crypto periods, and do slow market background, broad first-hour movement,
  calculated levels, or prior-range position add repeatable information without
  pretending unsigned news predicts direction?
- Layer 1 definitions: Parsed the official FOMC calendar into `45` already occurred
  releases and `11` future untouched releases. All `22` remembered major-move examples
  were restricted to discovery because their original list was selected with hindsight.
  Aggregate GDELT counts were retained as news intensity only; story meaning,
  expectation, surprise, and signed direction remained unavailable.
- Layer 2 complete individual tests: Tested `80` whole events (`45` FOMC and `35`
  non-colliding GDELT activity spikes), `19` coins, `739` event/control timestamps, and
  `1h`, `2h`, `4h`, `8h`, and `24h` responses. No profit or trade-return target was used.
  FOMC releases repeatedly produced more absolute movement, wider ranges, and more
  volume than matched ordinary hours across BTC, ETH, and established alts. The GDELT
  detector produced only two weak activity cells and should be treated as a readiness
  hint, not proof of a major story or direction.
- Layer 2 nulls: BTC/ETH/broad movement was usually simultaneous or absent rather than
  a repeatable advance leader. Leader-to-alt and leader-to-meme timing, full top-ten
  meme amplification, and stable reusable post-event ranges did not pass. Historical
  orderbook overlap contained only four FOMC and two selected GDELT events, so the
  orderbook pair was coverage-parked.
- Slow-background result: After a positive first `4h` BTC move, at least half the rise
  was lost by hour `24` in `63.6%` of negative-background FOMC cases versus `23.1%` of
  positive-background cases. The comparable GDELT rates were `60.0%` versus `20.0%`.
  Both relationships repeated in the earlier and later periods and were materially
  weaker at matched ordinary timestamps.
- Layer 3 pairwise controls: Froze and tested `65` pair cells. The slow-background fade
  pair survived both event sources, beat event-only and background-only controls by
  about `14.8-32.8` percentage points, and beat time-rotated background labels. This is
  evidence about *fade after an observed positive move*, not advance event direction.
- Local calculated-level result: Two adjacent-horizon patterns survived event-only,
  level-only/far, chronological, and time-rotated controls. Around GDELT activity spikes,
  BTC near one calculated level showed extra unsigned activity at `1h` and `2h` on
  `68.8%` of `16` qualifying events. Around FOMC releases, ETH near one calculated level
  did so at `2h` and `4h` on all `15` qualifying events. The latter is striking but still
  development evidence selected from this history, not an untouched 100% expectation.
- Level attribution: No single indicator supplied the retained local state. The GDELT/
  BTC cases were mainly Bollinger midline (`5`), round price (`5`), EMA50 (`3`), and
  other Bollinger edges (`3`). The FOMC/ETH cases were round price (`6`), EMA200 (`4`),
  Bollinger midline (`3`), and EMA50 (`2`). Counts are too small to rank indicators.
- Layer 3 parked combinations: First-hour broad direction produced three isolated cells
  only (FOMC BTC `8h`, meme `2h`, meme `24h`). Prior-30-day range edges produced three
  isolated cells only. Neither family repeated at adjacent horizons or across event
  sources, so both were parked rather than combined.
- Layer 4 chain gate: Combining event confirmation, negative slow background, and the
  retained local single-level state left only `6` GDELT/BTC and `3` FOMC/ETH complete
  events. Both fell below the ten-event floor, so the new `24h` chain outcomes were not
  opened and no three-block model or FreqAI run was justified.
- Layer 5 untouched freeze: Preserved six future routes: scheduled-event activity,
  GDELT-spike activity, the two source-specific negative-background fade checks, GDELT/
  BTC single-level activity at `1h-2h`, and FOMC/ETH single-level activity at `2h-4h`.
  Each waits for the first ten qualifying unseen events in arrival order. The `11`
  official releases from `2026-09-16` through `2027-12-08` remain unopened. Missing or
  late sources create abstention/coverage rows rather than guessed values.
- Verdict: Retain scheduled events as strong activity/readiness warnings. Retain slow
  background as a conditional capped-rally/fade lead after market confirmation. Retain
  the two local single-level patterns as promising unsigned reaction leads. Do not call
  any of them a trade rule, signed news prediction, named-leader forecast, meme
  amplification result, or untouched confirmation.
- Evidence: `user_data/research_news_data/context_features/event_hierarchy/`
  `layer1_honest_history_20260903a/`, `layer2_individual_links_20260903a/`,
  `layer3_pairwise_20260904a/`, `layer4_three_block_20260904a/`, and
  `layer5_untouched_confirmation_20260904a/`. Bulky Layer 2/3 detail tables are under
  `D:\FreqTradeStuffLargeData\research_outputs\event_hierarchy\`.

## 2026-09-04 - Live Context Audit And Bounded Semantic Pilot Frozen

- Question: Is the accumulated live news, web, global-market, and orderbook history
  ready to add story meaning to the current event hierarchy, and can a small pilot
  prove incremental value before any broad historical upgrade?
- Collection state: The point-in-time audit found `77,274` news articles, `4,238` web
  items, `56,322` global-context ticks, `451,196,711` orderbook messages, and
  `34,118,160` orderbook metric rows. The inspected collectors were active. Two ECB
  feeds repeatedly fail certificate verification; several Google Trends macro/stress
  groups and Stooq risk groups are disabled, stopped, or historically short.
- Readiness gap: News/web category, tag, relevance, urgency, and duplicate enrichment
  exists, but all audited direction values were `unknown`, severity/confidence were
  unset, and forward-return tables were empty. This supports count/activity baselines,
  not signed historical story claims.
- Frozen pilot: First export immutable snapshots and freeze exact coverage. Then label
  at most `300` duplicate-clustered stories across two chronological blocks without
  looking at later prices. Preserve direction, severity, novelty, escalation/relief,
  source confluence, evidence, uncertainty, and full version metadata. Validate the
  schema deterministically and review up to `40` clusters before market testing.
- Market comparison: On identical BTC/ETH rows, compare simple counts/event metadata
  with each semantic sibling separately, including one major story and accumulated
  same-direction minor stories. Test activity, volume, volatility/range, and pressure
  before signed direction. Use whole-story chronological validation and
  event-appropriate horizons; profit is not a target.
- Decision rule: Expand only if semantics add repeatable information beyond the simple
  baseline in both chronological blocks without one event or source dominating.
  Otherwise revise once for a clear engineering defect or park the enrichment lane.
- Sequencing: The already-written confluence freeze remains unopened and paused. It
  resumes only after the pilot decision and may use only source-ready features. This
  bounded exception does not globally resume Objective 02d.
- Governing documents: `objective_02b_market_reaction_zone_discovery.md` Section 7.7,
  `objective_02d_news_context_integration.md`, `status_source_readiness.md`, and
  `source_readiness_matrix.csv`.

## 2026-09-04 - Reaction Leads And Directional Context Goal Reopened

- Authority: The user replaced the temporary direction-neutral goal with a new active
  goal that preserves positive reaction-point findings and adds event-scoped direction,
  with greater emphasis on news, web, Google Trends, major events, wider markets, and
  confirmation. It also requires parked promising leads and earlier untested requests
  to remain visible and receive breadth-first consideration.
- Governing/checklist files: Objective 02b remains the governing plan;
  `hypothesis_ledger.csv` is the investigation checklist; this file remains the single
  durable outcome ledger. No competing tracker was created.
- Preserved positive evidence: scheduled FOMC and five additional macro-release
  families repeatedly marked unusual BTC/ETH activity; VIX and Nasdaq shocks also
  marked unusual BTC/ETH activity; negative prior-month background remained a lead for
  fading an already observed positive event move; and source-specific calculated
  single levels remained promising activity locations. None alone is a complete
  advance directional rule.
- Semantic freeze executed: The bounded tool froze `120` outcome-blind stories, `60`
  in each of two chronological blocks, representing `149` source articles. All freeze
  integrity checks and artifact hashes passed. `25` clusters contained exact-title
  duplicates and `95` were singletons under the conservative pilot rule. No OHLCV,
  price, return, or market outcome was read.
- Data-quality result: `5,174` news records were identified as stale startup/backfill
  material and one as implausibly future-dated relative to collection. The pilot uses
  observed `collected_at` as its safe causal time and excludes those rows. Every
  eligible news/web row still lacks usable semantic direction, severity, and
  confidence, so market outcomes remain closed pending labeling and review.
- Checklist repair: Added explicit source siblings for broader scheduled macro events,
  VIX/Nasdaq, Google Trends, live web announcements, global flows/risk, event-aligned
  orderbook, and unscheduled major stories. Added later combinations for accumulated
  minor-story balance, event/context plus reaction-area direction, BTC/alt/meme
  transmission, and evidence-triggered one-minute replay.
- Sequencing: Complete the independent source siblings as one breadth batch before any
  combination. Report reaction/activity and direction separately. Story combinations
  wait for the semantic gate; orderbook work waits for exact overlap coverage; disabled
  Trends sources receive isolated collection tests rather than guessed history. The
  previously written confluence freeze remains outcome-unopened until this batch is
  jointly reviewed.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/semantic_data_pilot_20260904a/`,
  `broad_event_relevance_20260904a/`, and `cross_asset_relevance_20260904a/`.

## 2026-09-04 - Global Context Source Preflight

- Question: Can Google Trends, ETF flows, fear/greed, wider-market data, stablecoins,
  and broad crypto measurements be used causally in the reopened directional goal?
- Method: Audited the live SQLite store read-only without loading raw payloads, OHLCV,
  returns, or labels. Counted genuinely distinct source timestamps, collection cadence,
  unchanged repeat pulls, revisions, and any value timestamped before collection.
- Result: `56,388` rows covered `43` metrics from `15` sources, but frequent polling
  substantially inflated the apparent history. Only one metric was directly safe as
  stored; all `42` others are recoverable through collection-time extraction, and `10`
  contain preserved revisions that must remain separate vintages rather than being
  treated as values known from the start.
- Google Trends: The enabled crypto-attention family has `182` distinct observations
  per metric, roughly twelve hours apart, with no same-time revisions. The values were
  stored about `20.6` minutes before actual receipt at the median and up to `62.9`
  minutes early, so `created_at` must replace `ts` as the test availability time.
- Other sources: Daily ETF, fear/greed, and FRED values were repeatedly recorded by the
  half-hour collector; unchanged pulls are not extra evidence. ETF/FRED values can
  revise, so preserve their first-seen/changed sequence. CoinGecko snapshots retain
  much denser changing observations. The inspected DeFi-chain metric never changed and
  is parked unless fresh variation appears.
- Verdict: Source preparation passed with repair required; no directional claim has
  been tested. Next freeze small source-specific causal extracts and run every source
  sibling separately before combinations.
- Frozen extract: The outcome-blind causal parquet retained `22,038` first-seen or
  changed observation vintages and removed `34,350` unchanged repeat polls. It keeps
  first/last observed time, revision identity, and collection-time availability.
- Evidence: `user_data/research_news_data/context_features/source_preflight/`
  `global_context_source_preflight_20260904a.json`.

## 2026-09-04 - Recent Orderbook Event And Reaction Coverage Preflight

- Question: Is the recent four-venue orderbook history complete enough to test early
  confirmation around major events and already identified reaction areas?
- Method: Queried only indexed one-minute coverage and quality fields from the live
  database for a fixed `2026-08-23 20:00` to `2026-09-04 19:00 UTC` interval. No
  price outcomes, directional labels, or orderbook feature values were opened.
- Result: All `40` streams passed across ten coins, with no duplicate minute keys or
  premature timestamps. Each stream exceeded `95%` clock coverage and `80%` valid
  samples both overall and around the one frozen scheduled event in the interval.
- Limits: One scheduled event is not enough to judge event direction, and DOGE is the
  only meme coin in the collector. The data can support a later reaction-point
  confirmation pilot for the ten covered coins once exact contacts are frozen, but not
  a broad event or top-ten-meme conclusion.
- Verdict: Retain for reaction confirmation; accumulate events prospectively and queue
  meme-coverage improvement separately. Do not change the running collector merely to
  manufacture historical overlap.
- Evidence: `user_data/research_news_data/context_features/source_preflight/`
  `orderbook_event_coverage_preflight_20260904a.json`.

## 2026-09-04 - External Live-Media And CPI Evidence Added To The Queue

- Live-media lead: Direct artifact review found that the exact FreqAI target asked
  whether price reached `+1.5%` before `-1%` within the next three hourly closes. The
  live-media profile beat price-only AUC in eight of nine non-BTC coins and average
  precision in seven of nine from `2026-05-16` to `2026-06-27`. TRX failed and had only
  three positive labels. Contrary to the imported summary, those nine-coin runs did not
  include same-coin market-only, volatility-only, or shifted-time controls. The labels
  also overlap. This remains one secondary lead with a control gap, not nine separate
  discoveries, proof beyond market state, or a trading rule. Its Bitcoin `24h` and
  loose-orderbook findings stay parked because they were sparse or inconsistent.
- CPI lead: The completed direct and robustness reviews retained unusually high BTC and
  ETH activity during the first `5-60` minutes after CPI and a historical signed
  relationship with month-on-month core-CPI change. All four tested short windows
  survived the recorded pre-release, matched-week, and whole-release shuffle checks;
  the reported directional rates were about `67-80%`. The relationship weakened at
  `2h-4h`, and the 2026 slice was too small for confirmation.
- Interpretation limit: The CPI direction test used change from the previous release,
  not actual CPI minus the market's timestamped forecast. That missing expectation is
  now an explicit source-reconstruction/prospective-capture task before the short lead
  can be described as a true surprise response.
- New independent questions queued: CPI first-`5m/15m` BTC/ETH/breadth transmission to
  established coins; recent/prospective CPI response in the top-ten meme cohort after
  ordinary beta and volatility adjustment; non-FOMC central-bank decisions; bond,
  sovereign-credit, banking, and liquidity shocks; major trade, tariff, sanction, war,
  and credible relief events; and narrowly defined systemic corporate/financial shocks.
  Producer prices, personal-consumption inflation, employment, retail sales, and GDP
  remain covered by the existing scheduled-macro activity family and must not be
  silently counted as new untested families.
- Sequencing: These are checklist additions, not permission to branch immediately. The
  current independent-source breadth batch finishes first; descendants are frozen only
  after the joint sibling review. Levels and clusters enter event combinations only
  after event and leader links have been tested independently.
- Evidence:
  `user_data/research_news_data/context_features/reports/live_media_deep_gapfill_20260628/`
  and `user_data/research_news_data/context_features/event_hierarchy/`
  `cpi_release_family_20260904a/direct_review_20260904a/`.

## 2026-09-04 - Independent Context Sources Completed Before Combinations

- Question: Which available live or longer-history context sources independently mark
  unusual crypto activity or supply bounded direction before they are combined with
  events, levels, or one another?
- Method: Froze ten live source families without opening outcomes, reduced `56,388`
  repeated polls to causal observation changes, and selected `359` source-defined
  events; `354` had at least six prior ordinary controls. The direct review then used
  nineteen markets and two whole recent time blocks (`2026-06-01` to `07-15` and
  `2026-07-16` to `08-30`). Seven longer-history cross-market families used separate
  `2021-2023` development and `2024-2025` validation blocks. Activity meant that the
  combined absolute move, full range, and volume was at least `1.20x` its frozen
  ordinary-time baseline; direction had to reach `55%` in both blocks and beat recent
  trend, development majority, and a rotated source-sign control by at least three
  percentage points.
- Strong unsigned activity clocks: Live-news bursts repeated for BTC and ETH at `1h`,
  `4h`, and `8h`, and for the established-coin group at `4h`. Live-web bursts repeated
  for all four market groups at `1h`, for BTC, ETH, and established coins at `4h`, and
  for BTC at `8h` and `24h`. These sources indicate that a more active period is likely;
  they do not contain a trustworthy up/down sign.
- Longer-history activity context: Large US market-fear changes repeated BTC and ETH
  activity at `4h`; large technology-share changes repeated ETH `4h` and BTC `8h`
  activity; and large US-dollar changes repeated ETH `8h` activity. None produced a
  standalone absolute direction that survived both periods and all simple controls.
- One relative direction lead: A sharp Bitcoin-dominance change predicted whether BTC
  would outperform the frozen top-ten meme group over `8h`: `8/11` development events
  and `15/23` validation events, about nine percentage points above the strongest
  simple control in each block. This is relative rotation between groups, not a claim
  that either BTC or memes rise or fall.
- Parked siblings: Causally retimestamped Google attention and live FRED equity risk
  were weak or inconsistent. ETF flow and broad crypto market state lacked ten events
  in the development block. Fear/greed, live rate/dollar risk, and stablecoin supply
  reached `55%` in a validation cell but failed a development or comparator rule.
  Crude oil and the yield curve were weak; financial conditions and ten-year yields
  lacked two-partition counts.
- Dependence check: Only four live-news and live-web event hours matched exactly and
  nine matched within one hour, so they are not identical clocks. Overlap rises to
  `28` events within four hours, so future confluence tests must compare news alone,
  web alone, their union, and their overlap rather than count them as independent
  confirmations.
- Verdict: The breadth-first independent-source batch is terminal. Retain live news,
  live web, historical fear/technology/dollar shocks as activity context and Bitcoin
  dominance as a relative-rotation lead. Do not combine Google attention or the parked
  daily sources merely to enlarge a model. The next frozen combination batch must keep
  component-only controls, test whether the source clock arrives before rather than
  after the move, and use early BTC/ETH or broad-market response for absolute direction.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `independent_context_sources_20260904a/direct_review_20260904a/`.

## 2026-09-04 - Historical Event-Family Confluence Sibling Batch

- Question: Do several distinct event families occurring within the prior `24h` mark
  more market activity than ordinary same-clock periods and isolated event families?
- Frozen siblings: At least two event families, at least three event families, at least
  two different source types, two agreeing positive wider-market signs, two agreeing
  negative wider-market signs, one isolated measurable shock at least twice its source
  threshold, and one isolated event-family reference. Sources were previously frozen
  scheduled releases, archive-news activity episodes, and wider-market shocks. Each
  route used a `72h` episode cooldown and all routes were frozen before outcomes.
- Coverage: The two-family route had `74` development and `53` validation episodes;
  the three-family route had `22` and `19`; and the two-source-type route had `61` and
  `40`. The signed positive/negative agreement and isolated-extreme routes had fewer
  than ten events in at least one period and were parked before a directional verdict.
- Result: Three or more recent families repeated unusually high ETH activity over the
  next `1h`. Two or more families repeated unusually high ETH activity over `4h`, and
  two different source types did the same over `4h`. Each result cleared the frozen
  ordinary-time threshold in both `2021-2023` and `2024-2025` and exceeded the isolated
  event-family reference in both periods. Single isolated event families were weak.
- Dependence limit: These are nested descriptions of substantially overlapping event
  episodes. Every three-family episode was within `72h` of a two-family episode, and
  `95%` was within `24h`; all two-source-type episodes were within `72h` of a two-family
  episode. Count this as one confluence finding, not three independent discoveries.
- Interpretation: The result supports the user's theory that several individually
  ordinary influences can jointly identify a busier market. It does not show which
  influence caused the move, does not yet predict up or down, and does not justify
  choosing component families after seeing their outcomes. The useful next direction
  question is whether a causally earlier BTC/ETH or broad-market response supplies the
  sign after a confluence clock has fired.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_confluence_20260904a/direct_review_20260904a/`.

## 2026-09-04 - Official Major-Event Source Feasibility

- Question: Can the queued non-FOMC central-bank, bond/liquidity, banking, and
  systemic-company event families be reconstructed from primary sources without
  selecting dates because crypto moved?
- Central banks: ECB, Bank of England, and Bank of Japan official archives are
  sufficient for a bounded `2020-2026` decision catalogue. ECB and BoE mostly use
  known scheduled times with documented exceptions; BoJ must use the actual time on
  each decision page. A small outcome-blind catalogue pilot is running before any
  market data are opened.
- Financial stress: Federal Reserve emergency/liquidity release pages are the best
  first intraday source because many record an exact official release time. Treasury
  auctions offer good structured fields but no consistently recorded results-release
  time; FDIC failures are mainly date-level; quarterly refunding is a slower document
  parsing lane.
- Systemic companies: SEC EDGAR supplies exact filing-acceptance timestamps and can
  support a fixed issuer/form catalogue. Acceptance is disclosure arrival, not proof
  that the filing first caused the story, so Nasdaq/VIX and ordinary filing-day
  controls remain required.
- Direction limit: None of these official sources provides a consistent historical
  field for what traders expected. Activity can be tested once catalogue QA passes;
  surprise direction must use a separately timestamped expectation source or an
  explicitly frozen post-release market-confirmation route.
- Outcome boundary: These were source-only probes. No crypto return or reaction
  outcome was inspected, and no finding has been promoted.

## 2026-09-05 - Central-Bank And Fed Source Pilots

- Central-bank pilot: Nine fixed representative decisions were captured from official
  ECB, Bank of England, and Bank of Japan pages. All eleven supporting URLs returned
  successfully, unique IDs and timezone conversions passed, and consensus remained
  null. Only the three BoJ pages supplied exact release times; the six ECB/BoE rows are
  date-only sentinels and are barred from intraday tests until official clock evidence
  is added.
- Fed liquidity pilot: The official-page parser recovered exact release times for the
  March 2020 support measures, March 2023 BTFP creation, and January 2024 termination
  notice. EDT/EST conversion, missing-time failure, source-domain restriction, hashes,
  and unique IDs passed ten focused tests.
- Interpretation: These pilots validate data collection and timing safeguards only.
  They contain nine and three hand-bounded examples respectively, so neither is market
  evidence. The next source step is a complete outcome-blind catalogue selected by
  fixed archive and vocabulary rules before any crypto outcome is opened.

## 2026-09-05 - Broad Event Clock Plus First-Hour Confirmation Batch

- Question: After one of ten previously frozen news, web, multi-event, US-dollar,
  market-fear, or technology-share clocks, does an unusually active first hour in BTC,
  ETH, both together, or the broad market improve later same-direction movement?
- Breadth: The frozen batch tested all `544` allowed route, leader, response-group, and
  `2h/4h/8h/24h` combinations. Historical routes excluded the present meme cohort;
  recent routes included it. Each route event had twelve prior same-weekday/hour
  ordinary comparisons. Direction after the first hour was compared with the event
  alone, first-hour confirmation at ordinary times, both time partitions, and rotated
  signs. No source was assigned a positive or negative meaning.
- Initial screen: `27/544` cells met the original cell rule. They were heavily related:
  news and news-or-web shared about nine tenths of their event anchors, BTC-plus-ETH
  confirmation was nested inside the BTC and ETH sets, and repeated horizons reused
  the same events. No meme cell passed. Explicit news-plus-web overlap and technology
  shares produced no passing directional cell.
- Family-wide check: The complete matched-set search was repeated `2,000` times, with
  each real event exchanged for one of its twelve ordinary comparisons and the same
  exchange shared across all dependent cells. Zero of the `27` apparent passes remained
  unusual after its route-family search; zero survived the full `544`-cell search. The
  smallest family-wide probability was `0.556`, meaning an equal or stronger best result
  appeared in over half of the randomized family searches. The smallest global value
  was `0.905`.
- Decision: Park generic first-hour continuation as directional evidence. Retain the
  already demonstrated unsigned news, web, confluence, and wider-market activity clocks,
  but do not refine the passing cells or describe them as edges. A first-hour link may
  be retested only when a distinct, causally signed event family or genuinely future
  events supply a new fixed question.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_confirmation_breadth_20260905a/direct_review_20260905a/`.

## 2026-09-05 - Semantic Story Labels Independently Reviewed

- Scope: `120` outcome-blind story records retained all frozen IDs and evidence links,
  split equally between development and validation. No price, OHLCV, orderbook, or market
  outcome was opened during labelling or review.
- Review finding: The schema was complete, but duplicate copies had been confused with
  later story updates and several company, product, or exchange-specific signs had been
  written as if they described the whole crypto market. Thirty stories also had little
  more than a title or scraped-URL summary.
- Repair: A separate reviewed label artifact conservatively changed `50` affected
  records. Duplicate coverage remains measurable through duplicate and source counts;
  uncertain or locally scoped market direction now abstains. All `120` reviewed records
  passed the existing schema and evidence validator.
- Coverage after repair: Before the shared time cooldown there are `15` signed stories
  in development and `16` in validation. No story has two independent source groups, so
  this pilot cannot honestly test independent-source narrative confluence. The next
  frozen siblings are all-story activity, moderate-or-major activity, and clearly signed
  story direction; a major-only or source-confluence route is excluded.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `semantic_data_pilot_20260904a/story_labels_reviewed.jsonl`.

## 2026-09-05 - Reviewed Semantic-Story Market Test

- Question: Do causally observed stories identify unusual market activity; do
  moderate-or-major stories improve on story counts and lower-severity stories; and
  does a conservatively reviewed positive or negative story sign predict market
  direction better than simple controls?
- Frozen method: The `120` labels were completed without price outcomes. A shared
  six-hour story cooldown left `86` all-story episodes, `39` moderate-or-major
  episodes, and `24` signed episodes split between development and validation. Each
  story began at the first complete hourly candle at or after it was first seen.
  Gapped windows were excluded. BTC, ETH, established coins, and the frozen top-ten
  meme cohort were measured at `1h`, `4h`, `8h`, and `24h` against twelve prior
  same-weekday/hour periods. Direction was compared with the completed prior-30-day
  trend, development-period majority, rotated story signs, and matched ordinary
  periods. Profit was not used.
- Activity result: Zero of `32` activity cells passed. The strongest repeated row was
  moderate-or-major stories for memes at `8h`, but it produced an unusual-activity
  response in only `40%` of development episodes and `33.3%` of validation episodes,
  below the frozen `55%` requirement. Its median activity score was `0.925`, so the
  typical episode was less active, not at least `1.20` times more active, than its
  ordinary controls.
- Direction result: Zero of `16` signed cells passed. The best overall story-sign
  agreement was `45.8%` for BTC and ETH at `4h`; the strongest simple comparator was
  `70.8%` for BTC and `62.5%` for ETH. No market group or horizon reached the frozen
  `55%` overall requirement while repeating at `50%` or better in both periods.
- Decision: Park semantic meaning for this current pilot. Preserve the independently
  demonstrated unsigned news and web activity clocks, but do not infer that generic
  story severity or broad-crypto story sign adds value. Do not expand this sample or
  tune its labels from the outcome. Reopen only when prospective collection supplies
  materially more major events, genuinely independent source groups, or another
  price-blind semantic basis. The user's several-small-stories hypothesis therefore
  remains a later data-dependent question rather than a failed test of confluence.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `semantic_outcome_20260905a/direct_review_20260905a/`.

## 2026-09-05 - Complete Federal Reserve Liquidity-Action Source Catalogue

- Question: Can Federal Reserve emergency facility, liquidity, temporary credit, and
  material facility-lifecycle announcements be reconstructed systematically with exact
  first-public timestamps, without choosing dates from later crypto movement?
- Method: A fixed action vocabulary was applied to every official annual Federal
  Reserve press-release index from `2020` through `2026`. Regular FOMC releases,
  speeches, reports, consultations, routine disclosures, and minor implementation
  notices were excluded. Each retained page had to be linked from its annual official
  index and contain an exact `For release at ... EST/EDT` clock. Missing clocks fail
  rather than becoming guessed midnight events. No market outcomes or profit were read.
- Coverage: All `40` frozen pages passed source and timing validation. Two pairs of
  simultaneous pages reduce the set to `38` distinct public release clocks. The yearly
  distribution is `32` in 2020, `4` in 2021, `3` in 2023, and `1` in 2024; there were
  no qualifying pages under the frozen rule in 2022, 2025, or 2026 year-to-date.
- Independence limit: The 40 pages are not 40 independent financial shocks. A
  conservative grouping leaves four broad episodes: the 2020 pandemic support
  programme, its 2021 lifecycle, the March 2023 banking-stress response, and the 2024
  BTFP termination notice. Historical consensus is unavailable and announcement times
  are not assumed to have been preannounced.
- Decision: Retain the source-complete event clock, but do not run or report a
  40-independent-event direction test. Continue the wider financial-stress breadth
  family through separately selected Treasury, sovereign-credit, banking, and
  systemic-company sources. A later market batch may use the Fed clocks only with
  simultaneous releases collapsed and whole broad episodes kept together.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `fed_liquidity_catalogue_20260905a/`.

## 2026-09-05 - Five Scheduled US Release Families: Short Direction

- Question: Does acceleration versus the previous published figure provide repeatable
  `5`, `15`, `30`, or `60` minute Bitcoin and Ethereum direction after producer-price,
  employment, consumer-spending/price, retail-sales, or GDP releases?
- Frozen rule: Acceleration meant a stronger latest change than the preceding published
  change. The deliberately simple rate-pressure interpretation predicted crypto down
  after acceleration and up after deceleration. This was frozen before signed market
  returns were opened. It is not actual news minus the market forecast.
- Coverage and controls: `255` releases at `242` distinct clocks were frozen across
  `2021-2023` development and `2024-2025` validation. Only complete one-minute windows
  were scored. Each cell had to reach `55%` overall, at least `50%` in both periods,
  and beat the strongest of development majority, preceding price direction, previous
  release sign, rotated event signs, and clean matched ordinary weeks by three points.
  Missing anchors, minute gaps, zero returns, and inadequate matched controls abstained.
- Paired-control correction: The first review compared ordinary-week performance from
  only control-eligible releases with event performance from all available releases.
  It was withdrawn and rerun so each event and ordinary-week comparison used identical
  eligible releases. The corrected review left employment candidates for BTC and ETH at
  `15`, `30`, and `60` minutes; both `5` minute cells were removed. Overall agreement
  on the retained raw cells was `62.5-66.7%` for Bitcoin and `63.0-69.6%` for Ethereum.
  Retail sales produced one BTC-only `60` minute candidate. Producer prices produced
  none. Consumer spending/prices and GDP lacked enough clean matched-week controls for
  a final route decision; their raw rows must not be called passes.
- Search adjustment: The full five-family, two-coin, four-window search was repeated
  with `2,000` whole-release sign reshuffles. A sign remained joined across BTC, ETH,
  and every window. Seven total candidates occurred with probability about `5.65%`;
  six candidates in one selected family occurred with probability about `5.45%`, while
  three shared BTC/ETH horizons occurred with probability about `4.20%`. The frozen
  rule required both family checks to be no more than `5%`, so employment narrowly
  failed and was parked. The isolated retail-sales cell was also common under
  reshuffling (`67.8%` by the selected-family cell-count check) and was parked.
- Decision: Retain all five scheduled families as previously demonstrated activity
  clocks, but park every previous-publication direction rule. Employment remains a
  promising near-lead for a later actual-versus-market-expectation test; it is not a
  retained finding and the current sample must not be tuned. Do not count the two coins
  or three overlapping windows as separate discoveries.
- Boundary: This is historical direction correlation, not causal proof, profit evidence,
  or a trading rule. A stronger or weaker employment change is not necessarily a surprise
  to traders; historical consensus expectations remain the central missing input.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `scheduled_macro_direction_20260905a/direct_review_20260905a/validation_20260905a/`.

## 2026-09-05 - Non-FOMC Central-Bank Decision Activity

- Question: Do scheduled ECB, Bank of England, or Bank of Japan decisions repeatedly
  make Bitcoin or Ethereum unusually active during the following `5`, `15`, `30`, or
  `60` minutes?
- Source foundation: Official archives supplied `145` decisions from `2020-2025`.
  The activity test excluded one unscheduled and one same-day-rescheduled BoJ meeting,
  leaving `143` ordinary scheduled decisions. ECB used its documented July 2022 time
  change; BoJ used the actual time on every decision page. BoE used the official noon
  standard plus known August/November 2020 and May 2025 clock exceptions and the
  September 2022 postponement; its historical exception inventory is not claimed to
  be exhaustive.
- Frozen method: BTC and ETH `1m` windows were compared with clean prior
  same-weekday/UTC-clock weeks. Gapped windows failed. A route needed at least `10`
  events, median activity of at least `1.20x`, and activity above the ordinary-period
  median on at least `55%` of events in both `2021-2023` and `2024-2025`. It also had
  to survive removal of central-bank releases within four hours of frozen CPI, FOMC,
  other US scheduled-macro, or another central-bank clock.
- Result: One of `24` bank/coin/window routes passed those direct gates: BoJ decisions
  followed by ETH activity over `60` minutes. Development had `14` events, median
  activity `1.201x`, and `64.3%` above ordinary weeks. Validation had `16` events,
  median activity `1.262x`, and `56.3%` above ordinary weeks. No nearby-event removals
  changed those counts.
- Other banks: BoE did not produce a route that repeated across both periods. ECB had
  many active all-release cells, but most clean cells fell to only five events because
  ECB releases often occurred close to US releases; the activity cannot yet be
  attributed to ECB alone.
- Decision: Preserve BoJ-to-ETH `60m` as one narrow unsigned lead pending a later
  whole-event familywide search adjustment. Do not refine it now, count overlapping
  windows as independent evidence, infer direction, or construct a trading rule.
  Move to another event family to preserve breadth.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `central_bank_catalogue_20260905a/direct_review_20260905a/`.

## 2026-09-06 - Treasury Quarterly Refunding Activity

- Question: Do all quarterly US Treasury refunding statements from `2020-2025`
  repeatedly make Bitcoin or Ethereum unusually active during the following `5`,
  `15`, `30`, or `60` minutes?
- Source foundation: The official Treasury archive supplied exactly four statements
  per year and `24` total. Each page provided a timezone-aware official publication
  time. The catalogue was frozen before crypto outcomes were opened. Historical market
  expectations were unavailable and were not guessed.
- Frozen method: BTC and ETH one-minute windows were compared with clean prior
  same-weekday/clock weeks. A route needed median activity of at least `1.20x` and to
  beat ordinary weeks on at least `55%` of events, with at least `10` development and
  `6` validation events. It also had to survive removal of statements within four hours
  of CPI, FOMC, scheduled US macro, ECB, BoE, or BoJ clocks.
- Coverage repair: Existing BTC and ETH files began too late, so the approved
  Freqtrade downloader prepended continuous Binance futures one-minute data from
  `2020-08-01`. This raised usable development coverage from seven to all twelve
  statements. Three of eight later statements collided with another major scheduled
  clock, leaving only five for that stricter variant.
- Result: None of the eight market/window routes repeated across both periods, even
  before nearby-event removal. ETH at `5m` and `15m` passed only during `2024-2025`;
  the corresponding `2021-2023` activity medians were only `0.69x` and `0.98x` and
  therefore contradicted a stable standalone effect. Bitcoin produced no passing
  period pair.
- Decision: Park quarterly Treasury refunding statements as a standalone short-window
  crypto activity clock. Do not tune the later-only ETH cells, infer direction, or run
  FreqAI on this family. Continue the wider financial-stress breadth queue through
  distinct sovereign-credit, banking, and systemic-company sources.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `treasury_refunding_catalogue_20260905a/direct_review_20260906a/`.

## 2026-09-06 - ESMA Sovereign-Rating Source Audit

- Question: Can the official European Rating Platform supply enough distinct,
  exactly timed national-sovereign rating publications for a later activity-first
  Bitcoin and Ethereum test?
- Source foundation: The source-only audit selected fixed national-government issuer
  aliases for the US, UK, Japan, China, and 20 current euro-area countries. It kept
  state issuer ratings and rejected similarly coded public bodies such as central
  banks. The official action-validity time is UTC publication or subscription-
  distribution time under the reporting standard.
- Coverage: `46,036` raw rating-action rows collapsed into `4,406` country/agency/time
  publications. Of those, `436` contained a source-labelled upgrade, downgrade,
  default/suspension change, or outlook/watch change, representing `429` exact market
  clocks. Simultaneous agency/country records share a market-window group so later
  tests cannot count them as independent evidence.
- Geographic limit: European signed-action coverage repeats across `2021-2023` and
  `2024-2025`. The US, UK, Japan, and China together have only `8` signed-action
  records in the later period, so their separate confirmation route is currently
  coverage-limited rather than disproved.
- Decision: Keep the ESMA European route as source-ready potential and queue a small
  activity-first test after the current breadth-first source inventory. Do not infer
  crypto direction from the direction of a rating action, and do not call this causal
  or profitable evidence. Historical market expectations remain unavailable.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `esma_sovereign_rating_catalogue_20260906a/`.

## 2026-09-06 - SEC Hyperscaler Filing Source Catalogue

- Question: Can official SEC history provide a complete, price-blind set of exact
  filing-acceptance clocks for major US technology-company disclosures?
- Fixed scope: Every `8-K` and `8-K/A` from Microsoft, Alphabet, Amazon, and Meta from
  `2020-2025`, using each company's main submissions JSON and every referenced
  historical JSON file that overlaps the fixed dates.
- Coverage: The catalogue contains `249` unique filings: `48` context rows in 2020,
  `124` development rows in 2021-2023, and `77` validation rows in 2024-2025. `97`
  contain Item `2.02` and are labelled earnings-like candidates. That item label does
  not say whether results were surprising, positive, negative, or important.
- Timing boundary: The SEC API provides exact UTC acceptance times. SEC acceptance is
  not proof of the first public-news time, and SEC states that public availability can
  lag acceptance. A later test may ask whether activity remains unusual around EDGAR
  acceptance; it must not claim the filing caused the move without an independently
  verified earlier release clock.
- Decision: Keep this as a source-ready systemic-company input. Queue one conservative
  activity screen and a separate outcome-blind text/first-public-time audit after the
  wider source batch. Direction requires expectations or independently frozen market
  confirmation, and later tests require Nasdaq/VIX controls.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `sec_hyperscaler_8k_catalogue_20260906a/`.

## 2026-09-06 - Non-US Economic-Release Source Inventory

- Question: Which non-US official sources can support complete minute-safe release
  catalogues without choosing events because crypto moved?
- First route: UK ONS release pages provide actual local times for CPI, labour, retail,
  and the same-clock monthly GDP/trade bundle. Release-specific pages and artifacts
  must be used because current time-series values may include later revisions.
- Second route: Japan has fixed official clocks and release archives for national CPI,
  Tankan, and the first preliminary GDP estimate. Later GDP estimates and revisions
  must remain attached to the first event rather than counted independently.
- Later routes: China NBS has timestamped CPI, official PMI, and national-economy
  releases. Eurostat has broad scheduled coverage but needs calendar-history and
  postponement QA before minute testing. South Korea has strong clocks but is a later
  regional extension after release-artifact QA.
- Direction limit: None of these primary sources supplies a historical timestamped
  market-consensus series. Initial tests may examine activity and post-release market
  confirmation only; prior-period change must not be called market surprise.
- Decision: Queue source-only UK and Japan pilots first, China next, and keep Eurostat
  and South Korea behind their stated QA gates. Do not pool regions or simultaneous
  release components as independent confirmations.

## 2026-09-06 - FDIC Failed-Bank Source Audit

- Question: Can the official US failed-bank history provide a complete, defensible
  clock set for testing banking-stress effects on crypto?
- Source foundation: The fixed official FDIC API query selected every failed bank
  dated from `2020-01-01` through `2025-12-31`, before opening any crypto outcome.
  It returned `13` institutions. Failures no more than three calendar days apart were
  grouped by a general pre-set rule, leaving `12` independent date-level episodes.
- Coverage: There were four failures in 2020, none in 2021-2022, five in 2023, two in
  2024, and two in 2025. Silicon Valley Bank and Signature Bank form one March 2023
  episode rather than two independent market shocks.
- Timing limit: The source provides an official failure date but no official release
  time. The catalogue therefore leaves the UTC time empty and marks every row
  ineligible for intraday testing; it does not invent midnight, market-open, or
  market-close timestamps.
- Decision: Retain FDIC failure episodes only for conservative daily or multi-day
  banking-stress background and possible case-study context. Do not use this source
  for minute-level reaction timing, first-public attribution, directional claims, or
  FreqAI training labels. No crypto outcome or profit was read during this audit.
- Verification: The frozen artifacts revalidated without overwrite, and all five
  focused source-catalogue tests passed.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `fdic_failed_bank_catalogue_20260906a/`.

## 2026-09-06 - UK ONS CPI Source Catalogue

- Question: Can the official UK statistics archive provide a complete, exactly timed,
  outcome-blind CPI release set for a later short-window Bitcoin and Ethereum activity
  test?
- Source foundation: The fixed catalogue used every release-specific ONS CPI page for
  publication years `2020-2025`. Each page had to return directly from the official
  ONS domain, contain the expected release title, and expose exactly one `Released` or
  `Release date` minute clock. Page content and output artifacts were hashed before any
  crypto outcome was opened.
- Coverage: All `72` monthly releases passed. The first three 2020 publications used
  `09:30` local time and the remaining `69` used `07:00`. Europe/London daylight-saving
  conversion was applied per release, producing `72` unique UTC clocks. Thirty-four
  pages retain a schedule-change notice, which is recorded rather than ignored.
- Evidence split: `12` releases are context-only 2020 rows, `36` are the frozen
  `2021-2023` development period, and `24` are the `2024-2025` internal-validation
  period. Whole releases, not their overlapping minute windows, are the independent
  units.
- Direction limit: ONS does not supply timestamped historical market consensus. This
  source can support an activity-first test and later post-release market-confirmation
  work, but not a claim that CPI was hotter or cooler than traders expected.
- Decision: Retain UK CPI as source-ready potential. Do not open its market outcomes
  until the current non-US source siblings have been completed or coverage-parked and
  reviewed together. Keep UK CPI separate from US CPI and from other same-clock UK
  releases in the first test.
- Verification: Five focused tests, Ruff, compilation, live source validation, and a
  no-overwrite hash revalidation all passed. No market outcome or profit was read.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `uk_ons_cpi_catalogue_20260906a/`.

## 2026-09-06 - Japan CPI, Tankan, And GDP Source Audit

- Question: Which major Japanese economic releases can be reconstructed with a clock
  strong enough for causal minute-level testing, without selecting dates from crypto
  movement?
- Coverage: Official archives support a fixed `2020-2025` publication inventory of
  `72` national CPI releases, `24` Tankan Summary releases, and `24` first-preliminary
  GDP releases. Later Tankan comprehensive data and later GDP estimates or revisions
  belong to the original event and must not be counted as new independent releases.
- Tankan clock: The Bank of Japan archive supplies all `24` Summary dates and their
  contemporaneous PDFs. Its official FAQ says the release time is always `08:50 JST`,
  and the PDFs use an `8:50 a.m.` embargo. This is a defensible official event clock,
  although it is not a server-posting timestamp.
- CPI clock: The Statistics Bureau states that national CPI is normally released at
  `08:30 JST`, but the archive does not provide a complete per-event actual posting-time
  history or exhaustive postponement log. The 2021 base change also means later
  backcasts must not replace the contemporaneous 2015-base first-release artifacts for
  January 2020 through June 2021.
- GDP clock: Cabinet Office archives identify all `24` first-preliminary releases and
  distinguish them from later estimates. The official schedule uses `08:50 JST`, but
  also warns that web publication can occur later; individual historical artifacts do
  not prove an actual public-availability minute.
- Direction limit: None of the three official routes supplies timestamped historical
  market consensus. Tankan respondent forecasts are business-survey answers, not the
  market's expectation for the release. Every surprise field therefore remains
  unavailable rather than being replaced with the prior published value.
- Decision: Retain Tankan as source-ready potential under its explicit official
  `08:50 JST` convention. Coverage-park Japanese CPI and GDP from strict minute-level
  attribution unless stronger actual-time evidence is found or a later batch explicitly
  accepts nominal clocks with uncertainty. They remain usable as date-level background.
  Move to China rather than deep-searching unofficial timestamp substitutes now.
- Source anchors: Statistics Bureau CPI FAQ and archive; Bank of Japan Tankan FAQ and
  Summary archives; Cabinet Office first-preliminary GDP archive and release schedule.
  No crypto outcome or profit was opened during this audit.

## 2026-09-06 - China NBS CPI, PMI, And National-Economy Source Audit

- Question: Can official Chinese statistics sources supply a complete, exactly timed,
  outcome-blind event set for later Bitcoin and Ethereum activity tests?
- Calendar coverage: The NBS annual release calendars enumerate `175` scheduled
  events across `2021-2025`: `60` CPI releases, `60` purchasing-manager releases,
  and `55` national-economy releases. The calendars provide exact Beijing dates and
  times, including non-standard cases rather than assuming one fixed clock.
- Clock limitation: NBS explicitly labels calendar dates preliminary and subject to
  adjustment. Chinese release pages sampled from the official data-release archive do
  preserve a displayed publication minute, including migrated 2022 pages that retain
  their original 2022 timestamp. However, a complete automated calendar-to-page audit
  could not be frozen reproducibly because the archive began returning its JavaScript
  access challenge during the bounded crawl.
- Direction limit: The official calendar and release pages do not provide timestamped
  historical market consensus. A previous publication must not be substituted for what
  traders expected, so no directional surprise label is available from this source.
- Decision: Retain China as nominal-clock, source-feasible potential, but do not call it
  exact-page-complete. Park strict minute-level attribution until the actual-page archive
  can be retrieved reproducibly. A later sensitivity test may use the published nominal
  clocks with an explicit uncertainty flag, separately for CPI, purchasing managers,
  and national-economy releases. No crypto outcome or profit was opened.
- Source anchors: Official NBS annual release-calendar index and the Chinese NBS
  `Data > Latest Releases` archive. The English translations are not event clocks
  because translated pages can appear after the Chinese release.

## 2026-09-07 - Trade, Sanctions, Escalation, And Relief Source Audit

- Question: Can major trade-policy, sanctions, war-escalation, and credible-relief
  events be reconstructed without selecting memorable dates because crypto moved?
- European sanctions: Council of the EU archive listings and release pages preserve
  publication minutes across the `2020-2025` period. This is the strongest historical
  official route found. A complete catalogue still needs fixed archive filters,
  duplicate handling, exclusion of routine renewals or administrative notices, and
  explicit confirmation of how the displayed local clock maps through Brussels
  daylight-saving time before market outcomes are opened.
- US tariff and sanctions sources: USTR, Treasury/OFAC, and White House pages provide
  official text and dates, and legal instruments may provide an implementation time.
  Those implementation times are not the public-announcement clock, while the public
  pages generally do not provide a dependable first-public minute. They are suitable
  as slow or date-level background unless a separate contemporaneous source proves the
  earlier public clock.
- War and relief: NATO and other official pages commonly preserve a scheduled event
  time or a later `last updated` time rather than the first moment an unexpected event
  became public. No complete official historical first-public archive was identified
  for war escalation, ceasefire progress, collapse, or credible relief. These events
  require prospective first-seen capture from more than one independent source, with
  rumours and unconfirmed political statements kept separate and allowed to abstain.
- Decision: Retain an EU-sanctions historical catalogue as source-feasible potential;
  keep US date/effective-time records as background; and route unscheduled escalation
  and relief to prospective multi-source evidence. Do not infer direction from words
  such as `sanction`, `escalation`, or `ceasefire`; first test whether the confirmed
  event changes Bitcoin or Ethereum activity. No crypto outcome or profit was opened.
- Relationship to earlier work: Weak aggregate GDELT trade/sanctions and oil-topic
  counts did not test discrete major announcements, so they neither verify nor rule out
  this event-defined route.

## 2026-09-07 - Remaining Bond, Auction, And Liquidity Source Audit

- Question: Do the remaining official financial-stress sources close a material gap
  after the Fed-liquidity, Treasury-refunding, sovereign-rating, bank-failure, and SEC
  catalogues?
- Treasury auctions: The official Fiscal Data history contains auction dates,
  competitive closing times, bid-to-cover, accepted bidder groups, yields, and result
  document names. It does not contain the recorded public result-release timestamp.
  Treasury documentation says the official clock is delivery of the result XML and
  records a standard award-notice delay after the competitive close, but the historical
  row does not preserve that actual delivery clock. A close-plus-standard-delay value
  would therefore be nominal and must be tested with timing sensitivity, not presented
  as an exact publication minute.
- Auction direction limit: A high yield or weak bid-to-cover ratio relative to the prior
  auction is not the market surprise. A proper directional label needs the contemporaneous
  when-issued yield or another expectation available before results, which is absent from
  the official auction dataset.
- Other interventions: The Bank of England's September 2022 gilt intervention has an
  official retrospective `11:00` announcement clock, but it is one short crisis sequence.
  ECB emergency pages commonly preserve dates rather than exact first-public minutes,
  while Japan's Ministry of Finance publishes foreign-exchange intervention totals and
  daily details after the market operations. These can support context or selected case
  studies, not a broad independent repeated-event sample.
- Decision: The financial-stress source breadth pass is complete. Retain ESMA sovereign
  ratings for a separate activity-first test. Keep Treasury refunding parked, FDIC
  failures as daily background, Treasury auctions as an optional nominal-clock
  sensitivity route, and rare non-Fed emergency interventions as case studies. Do not
  manufacture exact times or expectations. No new crypto outcome or profit was opened.

## 2026-09-07 - Eurostat Historical Release-Clock Audit

- Question: Can EU-wide economic releases be treated as exactly timed historical events
  rather than silently using the UK or ECB as a substitute for Europe?
- Official clock: Eurostat's release calendar declares `Europe/Luxembourg` time, and its
  HICP and short-term-statistics manuals state that releases are issued at `11:00`
  CET/CEST on predeclared dates. The current JSON calendar also returns `11:00` for every
  one of its `222` Euro-indicator rows in 2025, and the official iCalendar carries a
  full `Europe/Luxembourg` timezone definition.
- Historical limit: The current JSON calendar returned complete 2025 and current/future
  rows but no events when queried for 2020-2024. The iCalendar likewise begins in 2025.
  Historical Euro-indicator articles preserve publication dates, but the present source
  cannot mechanically prove every earlier reschedule, postponement, or actual posting
  minute. Article dates plus the standard 11:00 rule are therefore nominal clocks, not
  a complete actual-time archive.
- Duplication limit: Inflation has both a flash estimate and a later final HICP release;
  GDP has preliminary, flash, and later aggregate updates. These are separate information
  vintages but must not be pooled as independent confirmations of one underlying period.
- Direction limit: Eurostat does not supply timestamped historical market consensus.
  Prior releases and later revised series must not be called expectations or surprises.
- Decision: The regional source breadth pass is complete. Retain Eurostat as a nominal-
  clock sensitivity source, keeping each release family and vintage separate. UK ONS CPI
  remains the stronger exact-page historical route, and Japan Tankan remains the other
  defensible official-clock route. No crypto outcome or profit was opened.

## 2026-09-07 - Layer 2 Event Breadth Joint Review

- Question: Which independently prepared event families repeatedly coincide with an
  unusual short Bitcoin or Ethereum reaction, and does the first CPI move contain any
  useful information about the following move in established coins or memes?
- Frozen test: The batch opened UK CPI, Japan Tankan, European sovereign-rating
  publications, all fixed-hyperscaler SEC 8-K acceptances, the Item 2.02 earnings-like
  subset, CPI-to-established-coin transmission, and CPI-conditioned top-ten meme
  response together. It used whole-event development/validation periods, same-clock
  ordinary-week controls, collision-clean variants, and no profit measure.
- Source-activity results: UK CPI, Tankan, European sovereign ratings, and all
  hyperscaler 8-K acceptances were weak or inconsistent across the two historical
  periods. They do not justify a branch from this batch. The Item 2.02 earnings-like
  subset was the only repeated activity lead, at 15 minutes. Bitcoin's median combined
  movement/range/volume score was `1.37x` ordinary weeks in development and `1.33x` in
  validation; Ethereum's was `1.42x` and `1.57x`. The result also survived removing
  events within four hours of another frozen event. Bitcoin and Ethereum are one
  connected market response, not two discoveries. SEC acceptance may occur after the
  company's first press release, so this identifies a useful event family and time
  area but does not yet prove that the SEC filing caused the move.
- CPI transmission result: When the first 15-minute Bitcoin and Ethereum moves agreed
  and were larger than their ordinary-week moves, the median direction of BNB, ADA,
  and TRX from minute 15 through minute 60 agreed on `64.0%` of `25` issued event
  calls, versus `38.6%` for matched ordinary times. The established coins' own first
  15-minute direction reached `60.0%`, so the extra value from naming Bitcoin/Ethereum
  as leaders was small. ADA produced larger rates (`72.2%` to `77.4%` across the three
  linked leader definitions), but its own first move already reached `66.7%` to
  `74.2%`. Treat all overlapping leaders, coins, and windows as one provisional family
  until a whole-family shuffle tests whether candidate searching explains it.
- Meme result after data repair: The frozen top-ten cohort supplied only one complete
  development event and six complete 2026 events because several members had not
  accumulated enough pre-event trading history for earlier releases. Across the five
  frozen windows, reaction success was `25%` to `50%`, conditional direction was `0%`
  to `60%`, and joint success was `0%` to `33.3%`; none beat the `55%` joint floor and
  its ordinary-time control. This is not a positive CPI meme lead. Keep it prospective
  rather than replacing the frozen cohort after seeing outcomes.
- Data-integrity correction: A full adjacent-candle audit disproved the earlier
  top-thirty meme manifest's no-gap claim. Seven one-minute files and one DOGE
  three-day file contained `39` internal gaps totalling `2,610,303` missing candles.
  Exact isolated Freqtrade downloads repaired every gap. The final audit covers `30`
  pairs by `15` timeframes, `450` files and `59,953,783` rows, with zero gaps,
  duplicates, invalid rows, or missing intervals. Separately, `3,330` BTC one-minute
  candles were restored for the frozen CPI event/control hours. Results produced
  before these repairs must not be used for meme coverage claims without rerunning.
- Joint decision: Retain three separate validation leads: Item 2.02 earnings-like
  15-minute activity, CPI first-move-to-established-coin transmission, and the earlier
  narrow Bank of Japan-to-Ethereum 60-minute activity result. Park the failed source
  families and the current CPI meme route. The next frozen batch must complete all of
  these siblings before any one result creates another branch:
  1. a full-family whole-event shuffle for the SEC activity search;
  2. SEC first-public-clock and Nasdaq/VIX alternative-explanation controls;
  3. a full-family whole-event shuffle for all CPI transmission candidates;
  4. CPI common-event coverage and follower-own-momentum controls; and
  5. the already queued full-family adjustment for the Bank of Japan activity search.
- No entry, exit, profit, or trading rule was tested or promoted.

## 2026-09-08 - Layer 2 Five-Sibling Validation Joint Review

- Scope discipline: All five frozen siblings were completed before any result was
  interpreted as a branch. The batch tested the full searched families rather than
  selecting the most attractive coin, window, or event source after seeing outcomes.
- SEC activity: The connected Bitcoin/Ethereum 15-minute activity association around
  earnings-like Item 2.02 SEC acceptance times survived the full Layer 2 source-family
  randomization. Only `7` of `2,000` shuffled full-family maxima were at least as
  strong, giving a familywide probability of about `0.4%`. Retain this as one short
  event-time activity association, not as separate Bitcoin and Ethereum discoveries.
- SEC attribution limit: The `61` frozen earnings clusters have `0` independently
  verified first-public press-release minutes in the validated local archive, and the
  historical Nasdaq and VIX files are daily rather than intraday. The recent market
  collector begins in 2026 and does not overlap the 2021-2025 events. The test must
  therefore remain an association: it cannot distinguish the SEC filing from an
  earlier press release or a simultaneous technology-equity move.
- CPI transmission: The established-group median still reproduced the descriptive
  `64.0%` direction agreement after a large agreeing Bitcoin/Ethereum first 15 minutes,
  but `1,454` of `2,000` full-family shuffles produced an equally strong or stronger
  best route. The familywide probability was about `72.7%`, so the separate leader
  claim is not retained.
- CPI self-movement control: On the same `25` calls, the established followers' own
  first movement reached `60.0%`. Bitcoin/Ethereum agreement added only four percentage
  points, below the frozen five-point incremental-value floor. This does not weaken the
  separately retained finding that CPI causes an unusually large short reaction; it
  removes only the claim that Bitcoin/Ethereum independently lead the following move.
- Bank of Japan: The narrow ETH 60-minute activity route failed its original gates on
  repaired data and `1,498` of `2,000` full-family null maxima were at least as strong,
  a probability of about `74.9%`. Remove this clock-only route from the positive queue.
- Joint decision: Carry forward the CPI short-reaction lead and the SEC earnings-time
  activity association, both with their stated limits. Remove CPI leader transmission
  and BoJ/ETH activity from the positive queue. Defer SEC causal attribution and true
  CPI surprise direction until the missing timestamped expectation or intraday market
  data exist. No profit, entry, exit, or trading rule was tested or promoted.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `layer2_breadth_batch_20260907a/layer2_validation_batch_20260907a/`
  `layer2_validation_joint_review_20260908a/`.

## 2026-09-08 - Broad Event/Market/Level FreqAI Review And Signal-Portfolio Handoff

- Question: Across a broad price-blind catalogue, which separate information families
  help forecast market activity or signed movement around event times, and which
  limited combinations genuinely add more than either component alone?
- Frozen surface: Actual FreqAI compared `14` profiles on BTC, ETH, BNB, ADA, and TRX.
  Profiles kept recent market behaviour, slow background, cross-market movement,
  event identity, signed source values, event-family overlap, isolated levels, and
  level clusters separate before testing six event-plus-one-component combinations.
  The targets were movement range, volume, close direction, and excursion balance at
  `1h`, `4h`, and `8h`; no profit target was used.
- Support correction: The catalogue contained `707` source rows grouped into `567`
  episodes, but the original FreqAI training and evaluation surface gave some episodes
  two or three votes because they had several nearby catalogue anchors. A 10 September
  audit collapsed evaluation to one vote per whole episode: `380/151/133` anchor rows
  became `331/124/112` independent episodes in development/2024/2025. Prediction keys
  remained identical, but the historical models themselves were trained on the
  anchor-weighted rows and must not be promoted or reused.
- Corrected strongest result: Event identity plus recent market behaviour still had a
  historical all-five incremental volume result at `1h`. The previous all-five `8h`
  incremental claim disappeared after whole-episode weighting; `4h` was not a retained
  complete-comparison result. Recent market behaviour alone remains the readable
  activity baseline, and the later 2026 test did not show a useful generic event
  increment beyond that baseline.
- Independent secondary result: Event identity plus prior BTC/ETH/broad-market
  movement forecast `1h` volume at `66.9-71.0%` across the two held-out years and beat
  both isolated components by at least `3.3` points with better error and ranking.
  This is evidence that the event clock and already-visible market participation carry
  complementary activity information; it is not proof of advance BTC leadership.
- Events without context: Event identity alone repeatedly described later range and
  volume, especially `1h` and `8h`, but did not supply broad signed direction. Signed
  source fields also did not produce broad direction. Many signs are activity-only or
  previous-release proxies rather than genuine actual-minus-expectation surprises.
- Levels and clusters: Event-plus-level models often described activity, but broad
  combinations usually failed to beat event and level components separately. After
  whole-episode weighting, only BTC event plus isolated-level state for `1h` range
  retained the narrow complete-comparison result; the former ETH event-plus-cluster
  `4h` volume result disappeared. This does not erase separately established general
  reaction-zone evidence because this batch asked only whether levels added information
  at these event timestamps.
- Direction: No tested profile produced a broad five-coin direction lead. Whole-episode
  weighting removed the former BTC recent-behaviour `1h` close-direction cell. The raw
  historical direct survivors are BNB cluster state for `4h` close direction at about
  `55.6%` minimum correctness and BTC recent behaviour for `4h` excursion balance at
  about `58.9%`. They remain parked by the later family-wide chance screen and are not
  trading instructions.
- Joint-call limitation: No volume-plus-direction route reached a valid decision. The
  frozen confidence gate compared deliberately conservative, shrunken regression
  predictions with raw target-distribution cutoffs and issued no more than eight
  independent calls in a held-out year. The required minimum was twenty. Classify this
  as a score-calibration representation failure, not a negative market result. A later
  frozen batch may calibrate or rank prediction scores causally and must retain the
  original gate as a control.
- Handoff: The user authorized an additive evidence-to-usable-signal stage organized
  as five independent families: major-event information; background and market
  leadership; calculated single levels and clusters; local participation and
  pressure; and cross-asset transmission/amplification. Every earlier retained,
  conditional, specialist, unresolved, parked, and negative finding remains in this
  ledger. The five-family limit constrains active implementation; it does not delete
  evidence or force one universal ruleset.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_freqai_breadth_20260908a/freqai/event_freqai_breadth_20260908a/`.

## 2026-09-09 - Fresh 2026 Signal And Meme-Transmission Review

- Fresh five-family check: Five real FreqAI profiles tested one-hour volume across BTC,
  ETH, BNB, ADA, and TRX in two later 2026 periods. Their selected rows were often busy,
  but matched controls were also busy and a same-size ranking by already-visible relative
  volume matched or beat almost every model result. The only exception was a negligible
  `1.3` percentage-point cell. Retain the simple statement that high recent volume often
  persists; do not retain five separate FreqAI edges.
- Gate correction: The formal result reported zero confirmations because its frozen
  support gate required twenty called episodes in each half even though only `30` and
  `24` independent episodes existed before calls. This makes the formal gate
  coverage-limited, not evidence that market activity was absent.
- Overlap and direction controls: The old local-activity plus cross-market overlap did
  not beat its strongest same-size component in both 2026 halves and had only `10` then
  `5` independent episodes. Seven narrow historical direction cells were also no more
  than expected from the `462` related cells searched: `98.4%` of `2,000` fixed-support
  chance simulations produced at least seven survivors. Park those direction cells.
- Simple Bitcoin trigger: Prior-hour Bitcoin volume at or above its causal seven-day
  median preceded above-median next-hour volume in `24/28` event episodes (`85.7%`),
  versus `46.2%` without the trigger. Matched control times reached `87.2%`, so this is
  useful general volume persistence rather than event-specific information. It says
  that an already-busy market often remains busy; it does not identify whether news,
  anticipation, or another upstream condition originally created that activity.
- Top-ten meme response: Among `30` later-2026 episodes with an observed Bitcoin
  reaction, at least six of ten memes had elevated same-hour volume and range in
  `26/30` (`86.7%`). Established coins were almost identical. This verifies broad
  same-hour market participation but is not an advance forecast.
- Direction: The meme-group median followed Bitcoin's initial reaction during the next
  hour in `19/30` episodes (`63.3%`) and both Bitcoin directions were represented evenly.
  It weakened from `78.6%` in January-April to `50.0%` in May-August; the one-sided
  single-cell chance probability was `10.0%`, and the whole-event best-of-four direction
  probability was `24.5%`. Eight of ten individual memes exceeded `55%` overall but only
  four did so in the later half. Preserve this only as an exploratory prospective lead.
- Transmission and amplification: Bitcoin itself continued its initial direction only
  `53.3%` of the time. Memes matched Bitcoin's concurrent next-hour direction `70.0%`,
  followed the initial direction `81.3%` when Bitcoin continued, and only `42.9%` when
  Bitcoin reversed. Raw meme movement exceeded the established group in `90%` of cases,
  but after removing each coin's rolling Bitcoin sensitivity and ordinary volatility the
  result fell to `43-47%`. Memes normally move more; no special event amplification was
  demonstrated. Bitcoin continuation is a response observed during the following hour,
  so it is a useful confirmation or outcome description, not information available at
  the original event time.
- Joint result: The pre-event Bitcoin activity signal identified both a confirmed
  Bitcoin reaction and broad meme activity in `20/28` episodes (`71.4%`) versus `51.1%`
  at matched controls. It did not predict the subsequent meme direction: joint activity
  plus correct direction was `50.0%` at one hour and `39.3%` at three hours.
- Decision: Keep simple recent-volume persistence and same-hour group breadth as
  activity/readiness information. Park three-hour follower direction, adjusted meme
  amplification, the old FreqAI narrow direction cells, and the unconfirmed model
  complexity. No profit, entry, exit, or trading rule was tested or promoted.
- Five-family readable sweep: Six simple prototypes covered all five approved families
  across BTC, ETH, BNB, ADA, and TRX at one, four, and eight hours. Prior-hour relative
  volume was the only broad survivor: its minimum period rate was about `72-78%` by
  scope and horizon, and it beat no-signal rows by roughly `18-40` percentage points.
  These nine scope/horizon cells are one persistence mechanism, not nine discoveries.
- Activity-role negatives: A negative thirty-day background, generic single-area or
  cluster presence, and four-hour broad directional agreement did not improve future
  volume over their own simple no-signal rows. The generic event catalogue produced
  one BTC-only four-hour candidate but not an established-alt result. These findings
  reject only a generic volume-warning role; they do not erase separately demonstrated
  fade, level-reaction, or relative-rotation questions.
- Bounded branch review: At high-recent-volume BTC event times, four-hour above-normal
  volume occurred at `79.1%`, `78.5%`, and `92.9%` across 2024, 2025, and 2026, versus
  `75.7%`, `74.8%`, and `85.1%` at high-volume matched clocks. Direct event-to-matched-
  control comparisons beat the control median on `60.0%`, `64.2%`, and `56.0%`; the
  mean reaction advantage fell to only `1.3` points in 2026. Do not use the pooled
  event label as a separate generic four-hour warning. Keep prior volume as market
  readiness and judge each event family's driver evidence separately; this comparison
  did not test what caused the prior volume.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_signal_fresh_2026_20260909a/freqai/event_signal_fresh_2026_20260909a/` and
  `event_signal_fresh_2026_20260909a/event_meme_transmission_20260909a/` and
  `event_signal_fresh_2026_20260909a/event_simple_signal_families_20260909a/`.

## 2026-09-09 - Whole-Episode Interpretation Reanalysis

- Why this was required: The previous pooled event-plus-volume wording could be read as
  making prior volume compete with news for the role of cause. The frozen outcomes were
  re-read without fitting a new model. Each observation is now labelled as background
  or readiness, possible driver, confirmation, local modifier, transmission, or later
  outcome. A useful downstream response is not treated as evidence against an upstream
  event.
- Event and prior-volume result: High prior volume remained a strong indicator that
  later volume would also be high at both event and matched non-event clocks. After
  matching the prior-volume state, the pooled actual-event advantage was inconsistent
  at one hour. At four hours it was about `+12.5`, `+14.9`, and `+1.3` percentage points
  for high-volume episodes in 2024, 2025, and 2026; at eight hours it was about `+16.9`,
  `+15.6`, and `+12.7` points. This is a conditional event-plus-readiness lead for the
  next broad batch, not proof that every catalogue item caused the activity.
- Event-family context: In the later-2026 sample, all eight usable CPI and all five
  usable FOMC one-hour episodes had high prior volume and high following volume. Their
  separately completed matched-release tests remain the relevant evidence that these
  announcements repeatedly drive short activity. The new pooled comparison neither
  proves nor rejects that cause; it shows that readiness and event identity can coexist.
- Meme conditional leads: After a confirmed Bitcoin reaction, meme one-hour direction
  agreed with Bitcoin's first move in `15/21` (`71.4%`) episodes with high prior Bitcoin
  activity versus `4/9` (`44.4%`) without it. Agreement was `13/18` (`72.2%`) outside a
  negative prior-month Bitcoin background versus `6/12` (`50.0%`) inside one. These
  conditions overlap and the cells are small, so they explain plausible suppression or
  reinforcement to test later; they are not rules. Bitcoin's own later continuation
  remains a downstream confirmation, not an advance input.
- Level representation limit: Every one of the 30 qualifying later-2026 episodes had
  the same generic single-level/cluster classification, so that binary representation
  could not explain differing meme paths. A later broad test must use continuous,
  coin-local level distance, source, timeframe, role, and cluster composition rather
  than concluding that levels do not matter.
- Existing-result decisions: Retain CPI/FOMC short activity as event-driver evidence;
  retain the negative-background positive-move fade as a conditional modifier; retain
  levels and clusters for reaction and modifier roles; retain volume persistence as
  readiness/confirmation; preserve one-hour meme direction as an unresolved conditional
  lead. The generic event clock is not promoted as a universal forecast, and clock-only
  failures still reject only the exact tested representation rather than every possible
  influence from that event family.
- Next breadth-first batch: Complete five siblings before branching: event family plus
  background/narrative state; event surprise where timestamped expectations exist;
  initial BTC/ETH response and established-coin transmission; event response modified
  by continuous single-level/cluster context; and meme transmission only after a
  verified Bitcoin reaction. Review all five together before developing descendants.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_signal_fresh_2026_20260909a/event_whole_episode_reanalysis_20260909a/`
  `whole_episode_reanalysis_result.json`.

## 2026-09-09 - Conditional Whole-Episode Breadth Batch And Joint Review

- Batch integrity: Six siblings were frozen together after the interpretation
  reanalysis. Five used existing data and the true expectation-surprise sibling was
  coverage-parked before outcomes because no timestamped consensus expectation exists.
  Nearby catalogue anchors were collapsed to one independent whole-episode decision
  point: earliest for pre-event background/level questions and latest for the explicit
  within-day accumulation question.
- Screen discipline: The direct screen produced `3,540` overlapping decision views.
  Only `82` had a repeated effect direction without a material opposite-period result;
  another `103` materially changed direction between periods. After collapsing related
  coins, horizons, scopes, and outcomes, only `15` cross-pair or multi-horizon patterns
  remained, representing three possible mechanisms rather than independent edges.
- Cross-market overlap lead: During daily cross-market-state episodes, two or more
  distinct frozen global-market families within 24 hours coincided with extra one-hour
  volume across all five tested coins in the older development rows; BTC and TRX also
  repeated across longer horizons. The 2026 true-state support was only eight episodes
  and its median effect was negative, so this is not current confirmation. It is queued
  for a source-deduplicated test of genuinely distinct dollar, yield, equity-fear, oil,
  or other global drivers. Do not call it several aligned news stories yet.
- Signed-accumulation anomaly: Two or more same-signed catalogue items coincided with
  less extra activity across BTC, BNB, and TRX in older rows, with BTC repeating at one,
  four, and eight hours. Only one qualifying 2026 episode remained. This does not show
  that positive or negative stories suppress markets; it queues a price-blind audit of
  source identity, sign meaning, duplication, and availability before any market model.
- Outer-range discrete-event lead: Older discrete media, financial, or corporate events
  near the outer fifth of the prior 30-day range produced more one-hour activity in BTC,
  BNB, and ADA than comparable inner-range episodes and parent controls. It did not yet
  repeat across longer horizons and had no usable 2026 confirmation cell, so it is a
  narrow location-modifier branch only.
- No new branches from other siblings: Continuous level/cluster conditions produced no
  coherent event-specific pattern across enough coins and horizons; this does not weaken
  their separately established general reaction-location role. Established-coin and
  meme follow-through lacked both condition states plus confirmed-Bitcoin matched
  controls in enough periods, so those questions remain coverage-parked, not rejected.
  True expectation surprise remains untested rather than approximated.
- FreqAI decision: Do not run another broad grid from the raw cells. First complete the
  three named direct/source-integrity branches together. FreqAI becomes useful only if a
  readable relationship survives and needs continuous magnitude ranking or abstention
  calibration against the simpler rule.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_signal_fresh_2026_20260909a/event_conditional_episode_batch_20260909a/`
  `direct_review_20260909a/joint_review_20260909a/`
  `conditional_episode_joint_review.json`.

## 2026-09-10 - Active-Test Guardrail Audit And Branch Closure

- Audit scope: Rechecked the active historical event/FreqAI grid, its five-family signal
  portfolio, the fresh 2026 FreqAI check, the simple signal sweep, meme transmission,
  and the conditional whole-episode batch. The audit tested independent-event weighting,
  control reuse, information availability, source-sign meaning, and separation of
  upstream information from already-observed market response. It did not fit a new
  model or use profit.
- Independent-event correction: Some older event episodes had two or three catalogue
  anchors and therefore received repeated votes. Collapsing to one evaluation vote per
  episode reduced historical direct pass cells from `253` to `209` and complete
  event-plus-component cells from `17` to `12`; joint reaction-plus-direction remained
  `0`. These are correlated grid cells, not counts of independent discoveries. The old
  models were also trained on anchor-weighted rows, so none may be promoted or reused;
  any future event model must collapse or weight the training surface by whole episode.
- Material headline corrections: The historical all-five event-plus-recent-volume
  incremental result now remains only at `1h`, not `8h`. The ETH event-plus-cluster
  `4h` volume modifier disappeared; the narrow BTC event-plus-single-level `1h` range
  modifier remains historical. The BTC recent-market `1h` direction cell disappeared;
  only BNB cluster `4h` close direction and BTC recent-market `4h` excursion balance
  remain as raw direct cells, and the existing multiple-testing screen still parks all
  narrow direction findings.
- Role correction: The generic event-identity input contained upstream releases,
  downstream global-market movements, composite market measurements, and aggregate
  media attention. It must be described as an event-or-state prototype, not a pure
  news model. Aggregate GDELT activity is an attention/information-arrival proxy unless
  story semantics establish a distinct driver. A downstream market response can
  confirm or modify transmission; it does not disprove an upstream news cause.
- Sign correction: The former cross-family same-signed accumulation question was a
  representation failure. Of `565` rows with raw source arithmetic, only `236` had an
  explicit crypto-relation sign and `167` of those signs pointed the opposite way from
  the raw source number. Unlike CPI values, dollar moves, fear measures, yields, oil,
  and equity moves cannot be added by raw positive/negative arithmetic. This branch is
  closed without treating its old market outcome as evidence for or against narrative
  accumulation.
- Family-matched branch result: The apparent eight cross-market overlap findings fell
  to two after every comparison was first matched inside the same current event family.
  One was ADA-only at `1h`; the other was BTC-only at `4h`. Both described less extra
  activity when an upstream event had a known response context, survived only the older
  development and 2024 blocks, weakened or reversed in 2025, and had only one combined
  2026 case. Neither repeated across a related coin group or has enough later support;
  both are parked and `0` cells advance to FreqAI.
- Other branch results: Separating upper and lower prior-range edges within the same
  discrete event family retained `0/10` decisions. Signed response continuation lacked
  enough whole events. These exact representations are parked; this does not reject
  the separately established general reaction-area role or future true event-surprise
  data.
- Positive-result review: No withdrawal is required for causal calculated-area contact
  evidence, CPI/FOMC short activity, SEC earnings-time activity, simple recent-volume
  persistence, or broad same-hour meme participation. The area work already uses
  causal levels and near-miss/artificial-location controls and claims busy locations,
  not direction. CPI/FOMC use whole releases and pre-release or matched ordinary-time
  controls. SEC remains an association because acceptance may follow an earlier press
  release. Volume is readiness/confirmation depending on when it is measured; it is
  not interpreted as a rival root cause to news.
- Evidence:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `event_signal_fresh_2026_20260909a/event_conditional_episode_batch_20260909a/`
  `direct_review_20260909a/joint_review_20260909a/`
  `event_interaction_branch_batch_20260909a/active_test_guardrail_audit.json` and
  `event_interaction_branch_batch_20260909a/direct_review_20260910a/`.
- Next action: Return to breadth-first source and event-family work. Preserve calculated
  reaction areas, scheduled-event activity, simple market readiness, and market-group
  transmission as separate layers. Do not build a new FreqAI descendant from this
  closed branch batch; only a later source-ready family with a readable incremental
  question should justify another model.

## 2026-09-10 - Fresh Source Readiness And Market-Implied CPI Expectation Pilot

- Fresh collection check: A read-only rerun found `60,255` global-context rows across
  the same `43` metrics and `15` sources. Since the 4 September audit, the live stores
  added about `2,686` news rows, `194` web rows, `11` observations per enabled Google
  crypto-attention series, and only `3-6` distinct observations in the slower daily
  sources. Collection is healthy, but this is not a new independent validation period,
  so the existing market tests were not repeated on six extra days.
- Free survey-consensus result: No dependable free historical economist-consensus API
  was found. Trading Economics documents the required forecast field, but its former
  guest account now returns a discontinued-account response. Prior-release change must
  therefore remain labelled as a proxy rather than renamed as market surprise.
- Prediction-market alternative: Kalshi's public historical API preserves the separate
  threshold contracts and minute candles for monthly headline CPI. A price-blind source
  pilot froze ten releases, two per year where available across `2022-2026`, and read
  `84` threshold markets. The common information cutoff was the earliest contract close
  within each release; no crypto price, return, direction, or profit was opened.
- Source-quality result: All `10/10` releases had at least three two-sided threshold
  quotes within two hours of the cutoff, probabilities on both sides of `50%`, median
  bid/ask spread no wider than `0.25`, and a reconstructable market-implied median. Six
  were in the `2022-2024` source block and four in `2025-2026`, so the frozen source gate
  passed in both periods.
- Interpretation boundary: This provides a market-implied distribution, not the
  Bloomberg or Reuters economist consensus. It may contain risk premia and crowd bias.
  The old short-direction lead specifically used core month-on-month CPI, while this
  pilot checked headline month-on-month CPI. Core-CPI source coverage must therefore be
  preflighted separately rather than inferred from the headline pass.
- Decision: Retain and queue a later full catalogue plus core-CPI sibling. Do not open
  crypto outcomes immediately or turn CPI into the only research path. The later direct
  batch must compare actual-minus-market-implied expectation with the old previous-
  release proxy, use whole releases and chronological blocks, and abstain on stale,
  wide, non-bracketing, or internally inconsistent contract curves.
- EU sanctions source status: The official archive showed `18` topic pages and its main
  and official mirror versions matched in bounded browser comparisons. The current RSS
  supports reading the displayed clock as UTC, but historical pages lack an embedded
  offset. The automated official mirror then returned HTTP `503` twice. Catalogue code
  and source-only tests are ready, but the route is parked until the official source is
  stable; no bypass, guessed clock, crypto outcome, or result was created.
- Evidence:
  `user_data/research_news_data/context_features/source_preflight/`
  `global_context_source_preflight_20260910a.json` and
  `kalshi_cpi_expectation_20260910a/kalshi_cpi_expectation_preflight.json`.

## 2026-09-10 - Multi-Family Market-Implied Expectation Source Preflight

- Objective and boundary: Test whether the free prediction-market archive can provide
  causal pre-release expectations across several macro families before any crypto
  outcome is opened. Four dates per component were frozen in advance across older and
  newer source periods. This was a source-quality test, not a direction or profit test.
- Coverage found: Core CPI has `49` historical events, core PCE `22`, payrolls `40`,
  unemployment `60`, Fed decisions `27`, and real GDP `20`. Payroll and unemployment
  are two components of the same jobs-report episode and must never be counted as two
  independent events.
- Frozen gate: A component needed at least `3/4` usable samples, including at least one
  in both `2022-2024` and `2025-2026`. Threshold sources required three fresh two-sided
  quotes within two hours, probabilities bracketing `50%`, no ordering contradiction
  proven by bid/ask bounds, and median spread at most `0.25`. Fed categories had to
  bracket total probability one.
- Result: Core CPI passed `3/4`, payrolls `4/4`, unemployment `3/4`, and Fed decisions
  `3/4`. Core PCE and real GDP each passed only `2/4`; PCE had no usable older-period
  sample, while GDP had one incoherent old curve and one event whose contract close
  times differed by four minutes. Those two sources are partial, not evidence that the
  underlying economic releases do not matter.
- Source defects handled explicitly: One legacy PCE contract exposed a positive numeric
  strike even though its ticker, title, and settlement rule all said `-0.1`; the signed
  value was accepted only because all three semantic fields agreed. A Fed ticker's `>`
  character required normal URL encoding. The mixed-close GDP event was rejected rather
  than allowing later information into some thresholds.
- Interpretation: A passing source can reconstruct what prediction-market participants
  expected. It is not an economist survey, may contain crowd bias or risk premia, and
  does not yet say whether any release moves cryptocurrency in the expected direction.
- Decision: Queue one balanced full-catalogue batch for headline/core CPI, payroll and
  unemployment as one jobs-report family, and Fed decisions. First join official
  initially published outcomes and preserve common pre-release cutoffs; only after that
  outcome-blind freeze may a later batch inspect BTC/ETH `5-60m` response. Keep PCE and
  GDP as optional coverage and do not force incomplete historical tests.
- Evidence:
  `user_data/research_news_data/context_features/source_preflight/`
  `kalshi_macro_expectation_breadth_20260910a/`
  `kalshi_macro_expectation_breadth_preflight.json`.

## 2026-09-11 - Full Macro Expectations And Short-Direction Review

- Scope: Completed the queued balanced expectation batch for headline CPI, core CPI,
  payroll, unemployment, and Federal Reserve decisions. The source catalogue and
  official actual values were frozen without crypto outcomes first; only the eligible
  routes then opened Bitcoin and Ethereum direction over `5`, `15`, `30`, and `60`
  minutes. Profit was never used.
- Source/data touched: Kalshi historical threshold/category contracts, official
  initially published macro observations reconstructed through FRED/ALFRED vintages,
  official event clocks, and BTC/ETH `1m` futures OHLCV. Missing BTC/ETH event windows
  were downloaded into isolated folders, validated, and atomically merged before the
  final review.
- Source coverage: The full source-only catalogue made headline CPI, core CPI,
  payroll, unemployment, and Fed decisions source-ready. It produced `181` eligible
  component rows across `117` whole macro episodes. Core PCE was later-period only and
  real GDP lacked enough later events, so neither was forced into the direction batch.
- Timestamp/lookahead checks: All expectations use a common market-information cutoff
  before each announcement. Later official revisions were excluded. Payroll and
  unemployment were collapsed into one jobs-report episode, and headline/core CPI
  were also kept as components of one release. Outcome rows had zero duplicate
  route/event/pair/horizon keys, zero missing event responses, and zero remaining
  abstentions after the targeted OHLCV repair.
- Main result in trader language: When headline inflation and underlying inflation
  were both above their market-implied expectations, Bitcoin and Ethereum usually
  fell during the short reaction; when both were below expectation, they usually rose.
  This agreement route retained all eight BTC/ETH and `5-60m` candidate cells across
  `28` whole releases (`19` older and `9` later). Only `12/2000` whole-event
  reshuffles produced an equally broad best route, corresponding to a guarded
  familywide probability of about `0.65%`.
- Most useful view: At five minutes, Bitcoin direction was correct for `82.1%` of all
  agreement releases and `77.8%` of the later releases. Ethereum was `71.4%` overall
  and `66.7%` later. Longer windows remained historically strong overall but some
  later cells fell to `55.6%`, so this does not justify claiming equally dependable
  direction for the entire hour.
- Relationship to the old proxy: On the `13` agreement releases where the old
  previous-release-change formulation was also usable, both formulations made the
  same calls. The new market-expectation route therefore has not yet proved extra
  direction on that overlap. Its current added value is `15` additional valid events,
  where direction remained approximately `66.7%-80.0%` correct depending on asset and
  horizon.
- Other family results: Headline-only and core-only inflation each produced some good
  cells but did not survive the complete whole-batch adjustment. The combined jobs
  route was descriptively promising: across `21` older and `6` later reports, overall
  accuracy was about `70.4%-77.8%`, while the six later reports were `83.3%-100%`.
  However, only four later reports had the required ten ordinary-period comparisons,
  so jobs remains coverage-limited rather than retained or rejected. Fed direction
  was source-limited because only one eligible historical decision differed from the
  reconstructed dominant expectation; the separate FOMC activity finding remains.
- Interaction boundary: This batch isolates one possible event-pressure input. It does
  not claim that the announcement acts alone. Existing sentiment, liquidity, ongoing
  stories, volume readiness, wider-market state, Bitcoin confirmation, and nearby
  levels or clusters may strengthen, suppress, delay, or reverse the response and are
  reserved for later controlled combinations.
- Corrections and validation: Before accepting the result, payroll units were converted
  from the official thousands scale, delayed combined jobs releases were matched by
  reference period, duplicate component identifiers were replaced by canonical whole
  event IDs, and every control was repaired to compare exactly the same announcements.
  The earlier unpaired-control output was overwritten and is not independent evidence.
  The final scripts passed `37` targeted tests plus Ruff, used `2000` whole-event
  permutations, and retained the frozen `55%` minimum, `65%` strong target, and source
  coverage requirements without lowering them after seeing results.
- Artifact paths:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `macro_expectation_catalogue_20260910a/kalshi_macro_expectation_catalogue_result.json`;
  `macro_expectation_catalogue_20260910a/official_actual_freeze_20260911a/`
  `macro_expectation_actual_freeze_result.json`; and
  `macro_expectation_catalogue_20260910a/official_actual_freeze_20260911a/`
  `direction_batch_20260911a/direct_review_20260911a/`
  `macro_expectation_direction_plain_review.md`.
- Verdict: CPI headline-plus-core agreement is a retained historical direction lead
  requiring future confirmation. Jobs is a promising coverage-limited lead. CPI
  components alone were not retained after the whole-family search check. Fed decision
  surprise is untestable with the current historical variation. No trading rule or
  strategy is promoted.
- Next action: Freeze CPI refinement and return to the breadth-first checklist. Preserve
  a future whole-release confirmation queue for CPI and jobs; only after independent
  event families are reviewed should event pressure be combined with market readiness,
  Bitcoin confirmation, reaction levels/clusters, and coin-group transmission.

## 2026-09-12 - Seven Historical News And Market Interactions Rechecked

- Scope: Rechecked all seven saved historical aggregate-news interactions as one
  sibling batch before permitting any result-specific branch. This was a direct
  behaviour test of reaction size and direction; it did not use profit, entries,
  exits, FreqAI, or a trading rule.
- Source/data touched: The three frozen hourly historical-news overlay frames covering
  January-October 2020, March 2021-July 2022, and August-December 2022. The original
  frame-building script is missing, so this is a robustness review of frozen data and
  not reproducible independent confirmation from raw sources.
- Freeze and timestamp checks: The outcome-blind phase read only source and market
  columns. A frame timestamp is the opening time of a completed one-hour candle, the
  usable decision time is one hour later, and outcomes start with the next candle.
  All future-path columns were independently reproduced from the following 24 hourly
  highs and lows with zero discrepancy.
- Whole-episode and control design: Repeated trigger hours were collapsed into chains
  separated by gaps above 24 hours. The final surface contained `745` full episodes,
  `478` week-old-source comparisons, and `1,084` same-month component-control pairs.
  Controls separately represented the same market setup without the exact source
  condition and the source condition without the exact market setup. Controls within
  24 hours of any raw full trigger were excluded and were not reused inside a route.
- Outcome meaning: Reaction means the following 24-hour maximum excursion exceeded
  the median for that historical window. Direction means the larger excursion was on
  the predeclared side. The frozen joint gate required both in the same episode. Four
  control paths ended too close to a source-frame boundary and remained missing rather
  than being replaced after outcomes were visible; all full episode outcomes existed.
- Strongest partial result: High news attention during a high-volume Bitcoin breakout
  that was not already at the top extreme produced the expected upward side in `65.5%`
  of `87` market-matched whole episodes. That was `26.4` percentage points above
  similar breakouts without the exact attention condition, `20.2` points above
  attention episodes without the exact breakout setup, and `44.7` points above the
  week-old source-timing comparison. Batch-wide probabilities were about `0.15%`
  versus market-only and `3.25%` versus news-only. Two historical windows were strong;
  one small third-window news-only comparison was `5.6` points opposite. Reaction
  itself was only `54.0%` and did not pass the family-wide reaction check. Verdict:
  strong limited historical direction modifier, not a complete signal or rule.
- Other useful but incomplete evidence: Bad-news pressure after a high-volume lower
  break improved direction by `21.0` points over breakdown-only controls with a
  family-wide probability of about `1.1%`, but absolute direction was only `51.4%`,
  the news-only comparison missed the guarded probability threshold, and current
  source timing did not beat the week-old comparison. Preserve it as a later
  story-specific direction question below the `55%` target.
- Background-dependent evidence: Broad macro stress ignored while Bitcoin was rising
  improved the joint outcome by `11.1` points over market-only and `17.1` points over
  news-only controls. Those pooled differences survived the batch-wide check, but
  direction alone was only `48.1%`, one historical window reversed, and current timing
  was `5.1` points worse than the stale-source comparison. This supports a missing-
  background interpretation, not a bullish broad-stress rule.
- Parked exact representations: High attention after downside compression had roughly
  `57.1%` reaction and `55.4%` direction, but added only `6.4` and `3.6` points over
  comparable compression states and did not survive the chance check. Macro relief
  near the range low had `71.4%` reaction but only `50.0%` direction and its joint
  result was worse than matched range-low states. Broad crypto stress near a range
  high was worse than matched range-high states for both reaction and direction. Bad-
  news absorption near support remained unjudgeable because only `0`, `2`, and `2`
  same-month market-only matches existed across the three windows.
- Statistical audit: Each of reaction, direction, and joint behaviour received `2,000`
  linked whole-episode reshuffles against both component controls, producing `12,000`
  null rows. The final family-wide comparison uses a paired studentized statistic so
  a small high-variance family cannot dominate the best-of-seven correction. The
  scripts passed `10` targeted tests and Ruff; duplicate episode/control keys,
  timestamp-offset errors, control-exclusion violations, and source-path arithmetic
  discrepancies were all zero.
- Interpretation boundary: These results do not say news is irrelevant when an exact
  aggregate route fails. The broad source measures can miss story meaning and severity.
  The retained partial result means attention added directional information in this
  historical breakout context; market background, liquidity, Bitcoin confirmation,
  calculated levels/clusters, and other simultaneous stories can still strengthen,
  suppress, delay, or reverse the response.
- Artifact paths:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `historical_news_interaction_retest_20260911a/` and its
  `direct_review_20260911a/historical_news_interaction_plain_review.md`.
- Verdict: Preserve the high-attention breakout result as a strong historical
  direction lead needing a reproducible new-period test. Preserve bad-news breakdown
  as a weaker below-target lead and broad macro-stress resilience as an unstable
  background-dependent lead. Park the other exact representations without rejecting
  their source, level, compression, or event concepts in different roles.
- Next action: Return to the remaining independent breadth-first event/source checklist.
  Queue rebuilding or replacing the missing historical attention pipeline and a
  preregistered fresh-period breakout confirmation for a later batch; do not branch it
  immediately or send it to FreqAI before the wider batch review is complete.

## 2026-09-12 - China NBS Nominal-Clock Activity And Eurostat Coverage Review

- Source-only freeze: The official China NBS calendars supplied `175` whole events
  across 2021-2025: `60` shared CPI-plus-PPI releases, `60` official PMI bundles, and
  `55` national-economy bundles. Complete exact historical first-publication times
  were unavailable, so the later market test treated the official planned times as
  nominal clocks and tested fixed `-30`, `0`, and `+30` minute sensitivity views.
- Eurostat coverage: Its current official calendar API returned zero historical rows
  for 2021-2024 and only the available 2025 block. That cannot support the frozen
  2021-2023 development versus 2024-2025 validation design, so Eurostat remained
  coverage-parked and no Eurostat market outcomes were opened.
- Frozen test: BTC and ETH `1m` candles measured unsigned `60m` and `240m` activity
  through absolute movement, full range, and volume. Each event used `12` prior
  same-weekday/same-UTC-clock controls. Both chronological periods had to pass in the
  all-event and collision-clean views, followed by `2,000` linked whole-event
  reshuffles across the complete search. No profit, direction, causation, entry, or
  exit was tested.
- Complete QA: All `175` events were tested, `155` remained collision-clean, all
  `2,100` frozen controls were present, and all `27,300/27,300` required market
  windows were contiguous and usable. Control weekday/clock mismatches, collision
  violations, independently recalculated raw-metric differences, and independently
  recalculated observed-score differences were all zero.
- National-economy result: BTC `240m` activity in 2024-2025 reached `1.202x` across
  all events and `1.370x` collision-clean, but its corresponding development values
  were only about `1.08x`. The stronger later-period behaviour therefore did not
  repeat across both chronological periods.
- Other family results: The CPI-plus-PPI BTC `60m` all-event median was `1.174x` in
  development but only `0.866x` in validation. PMI's best nominal all-event cell was
  ETH `60m` in development at `1.025x`, and its later-period result was weaker.
- Result: Zero of the `12` family-asset-horizon routes passed both chronological
  periods and both collision views, so zero routes were retained. A few shifted-clock
  cells met only the softer adjacent-clock requirements; none rescued the required
  nominal route or qualified as a strict timing-mismatch result.
- Interpretation boundary: UK CPI and Japan Tankan were weak or inconsistent. Across
  `175` China events, none of `12` BTC/ETH one-hour/four-hour routes passed both
  chronological periods and both collision views; the later BTC four-hour national-
  economy activity did not repeat in development. This rejects only generic clock-
  only warnings, not effects conditional on surprise, importance, background, or
  cross-market transmission. Eurostat was not market-tested.
- Decision: Do not branch the generic UK CPI, Japan Tankan, or China NBS nominal-clock
  activity routes. Eurostat remains coverage-parked because the official feed lacks
  the required 2021-2024 release rows. Preserve regional releases only as possible
  conditional modifiers and reopen direction when true surprises, stronger exact
  timestamps, or prospective evidence exist; continue to an independent event family.
- Evidence paths:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `regional_nominal_clock_batch_20260912a/regional_nominal_clock_freeze_result.json`;
  `regional_nominal_clock_batch_20260912a/`
  `regional_nominal_clock_market_analysis_contract.json`;
  `regional_nominal_clock_batch_20260912a/direct_review_20260912a/`
  `regional_activity_result.json`;
  `regional_nominal_clock_batch_20260912a/direct_review_20260912a/`
  `regional_activity_route_decisions.csv`; and
  `regional_nominal_clock_batch_20260912a/direct_review_20260912a/`
  `regional_activity_plain_review.md`.

## 2026-09-13 - EU Council Sanctions Natural-Background Activity Review

- Source freeze: The official topic archive returned `357` releases, of which `98`
  selected rows collapsed into `69` whole-date episodes; `67` episodes from 2023-2025
  were scored. The archive returned only three total rows before 2023, so it is source-
  complete for the filtered archive response but not a complete historical universe of
  all EU sanctions before 2023.
- Frozen design: BTC and ETH `1m` candles measured unsigned `60m`, `240m`, and `1440m`
  activity at fixed `-120`, `-60`, `0`, `+60`, and `+120` minute clock offsets. Each
  episode used `12` same-year, same-weekday weekly controls across separate 2023, 2024,
  and 2025 partitions, followed by `2,000` linked whole-episode randomizations over the
  full family search. The primary view preserved the natural mixed event background.
- Complete QA: All `67` scored episodes and `804` controls were present, producing
  `37,440` member-metric rows, `26,130` episode-metric rows, `90` route cells, `6`
  route decisions, and `2,000` finite null rows. Exactly eight outputs and their hashes
  validated, and all `17` focused tests passed.
- Result: Zero of six BTC/ETH horizon routes repeated across all three yearly
  partitions, so zero routes were retained. Illustrative strongest individual cells
  moved across years and offsets: BTC `60m` in 2023 at `-120` was `1.300x` with `63.6%`
  above controls; BTC `240m` in 2024 at `+120` was `1.242x` with `68.0%`; ETH `240m`
  in 2024 at `+120` was `1.263x` with `56.0%`. None repeated across the required
  periods and offset pair.
- Interaction boundary: `40/67` events and `505/804` controls were within 26 hours of
  other catalogued anchors. A descriptive nominal-clock split did not show stronger
  activity in overlapped episodes, but this was post-hoc, omitted unscheduled stories,
  and cannot imply that news was not a driver. Episodes with already-high prior volume
  often remained more active, consistent with readiness or persistence; that neither
  replaces nor disproves event effects.
- Decision: Do not branch/tune the generic Council publication-clock result. Preserve
  substantive tightening/easing as a possible conditional news layer. After remaining
  independent broad families are reviewed, a separate frozen batch may test price-
  blind importance/severity, unexpectedness, independent-source confirmation, other-
  story confluence, and pre-event market state. Keep genuinely unscheduled escalation/
  credible relief prospective.
- Evidence paths:
  `user_data/research_news_data/context_features/event_hierarchy/`
  `eu_council_sanctions_catalogue_20260913a/eu_sanctions_activity_analysis_contract_v2.json`;
  `eu_council_sanctions_catalogue_20260913a/eu_council_sanctions_source_freeze.json`;
  `eu_council_sanctions_catalogue_20260913a/eu_council_sanctions_catalogue_result.json`;
  `eu_council_sanctions_catalogue_20260913a/direct_review_20260913a/`
  `eu_sanctions_activity_analysis_result.json`;
  `eu_council_sanctions_catalogue_20260913a/direct_review_20260913a/`
  `eu_sanctions_activity_route_decisions.csv`; and
  `eu_council_sanctions_catalogue_20260913a/direct_review_20260913a/`
  `eu_sanctions_activity_plain_review.md`.

## 2026-09-13 - Anticipated Crypto Structural Event Source Freeze

- Frozen question and scope: Source-only review of whether scheduled Bitcoin halvings
  and consensus activations, Ethereum mainnet upgrades, and SEC spot BTC/ETH product
  decisions can later support honest activation or anticipation studies. The three
  families remain separate, with related stages and regulatory processes explicitly
  linked rather than counted as fully independent evidence.
- Exact counts: Bitcoin supplied `3` activation clocks. Ethereum supplied `18` named
  components at `14` clocks linked into `9` programmes. The SEC supplied `22` final
  orders covering `42` proposal members collapsed to `19` date episodes.
- Source and provenance QA: All `48` frozen sources verified. Official project and
  regulator sources establish the named events and anchors; the three Bitcoin
  Blockstream observations and six pre-merge Ethereum Etherscan block pages remain
  explicitly labelled secondary chain evidence. Twelve Ethereum component rows use
  exact clocks mechanically derived from the official epoch anchor as Beacon genesis
  plus `epoch x 384` seconds, not as official publication timestamps.
- Information-time corrections: Paris block `15,537,394` is frozen at the verified
  `2022-09-15T06:42:59Z` block timestamp. Taproot's `2021-05-01` release preceded
  miner lock-in, so final schedule knowability remains unavailable until a documented
  lock-in source exists. Eventual activation clocks must never be backfilled as facts
  that were knowable in earlier anticipation windows.
- Feasibility: Bitcoin remains three case studies. Ethereum's `14` heterogeneous,
  programme-linked clocks split `9` in 2020-2022 versus `5` in 2023-2025. SEC dates
  split `11` versus `8`, but all `19` lack an exact first-public minute and therefore
  remain date-scale only. No family is pooled with another to manufacture support.
- Decision: Do not pool the three families or run FreqAI. Preserve BTC/ETH activation
  clocks for honest case-study replay and prospective capture, using only schedules or
  block-arrival estimates knowable then. Keep SEC decisions date-scale only until exact
  first-public timestamps and prior decision deadlines are independently frozen.
  Reopen inference only after materially more independent events or a separately frozen
  established-chain cohort.
- Outcome boundary: No market data, FreqAI, direction, profit, or inferred surprise was
  opened or tested in this source freeze.
- Evidence paths:
  `user_data/Custom_Launcher/research/context_features/market_event_crypto_structural_catalogue.py`;
  `user_data/tests/research/test_market_event_crypto_structural_catalogue.py`;
  `user_data/research_news_data/context_features/event_hierarchy/`
  `crypto_structural_events_catalogue_20260913a/crypto_structural_events_catalogue.csv`;
  `crypto_structural_events_catalogue_20260913a/official_source_manifest.json`;
  `crypto_structural_events_catalogue_20260913a/crypto_structural_events_source_freeze.json`;
  `crypto_structural_events_catalogue_20260913a/crypto_structural_events_catalogue_result.json`;
  and `crypto_structural_events_catalogue_20260913a/crypto_structural_events_catalogue_report.md`.

## 2026-09-19 - Ownership Reassessment, Evidence Reuse, And Bounded Development Handoff

### Scope and review strength

- User authorized reassessment of existing results and cheaper delegated development,
  with the parent agent owning questions, decisions, interpretation and review.
- Reviewed the 65 existing hypothesis records, their evidence references and relevant
  compact family reports. Reconciled early level entries with the relationship register
  and later joint reviews through Generation 26. Added five missing records for the
  later level findings, one-minute integrity repair and existing portfolio preflight.
- This is a complete checklist inventory review, NOT a claim that every historical
  experiment, raw observation or implementation has been independently reproduced.
  Existing selected method spot checks inform it; remaining source/timing doubts are
  explicitly retained. No new market outcomes or fresh confirmation blocks were opened.
- A referenced artifact can be available without its underlying source pipeline being
  reproducible. Historical test periods already used to select ideas are development
  evidence for any new combination, not a newly untouched test.
- No trade, cost-adjusted profitability, live order or production strategy was tested
  or approved by this reassessment.

### What is worth carrying forward

| Possible use | Evidence worth keeping | Important limit |
|---|---|---|
| Anticipate busy periods | Recent completed volume; CPI/FOMC release activity; some news/web activity and wider-market shocks | Activity does not supply direction or explain the original cause. Several inputs may measure the same episode. |
| Locate interaction areas | VP-related recrossing, thin-LVN context, some group-specific local-activity effects | Persistent traffic is not necessarily a new reaction caused by contact. Exact special coordinates and universal cluster superiority are unproven. |
| Short event direction | Headline/core CPI expectation agreement | 28 releases, only nine in the later block; strongest at five minutes. Historical association, not executable profit or universal news logic. |
| Conditional reversal | Negative prior-month background plus an already-observed event rise | Only 11 FOMC and ten GDELT cases in the retained comparisons; decision is four hours after the event. |
| Choose market exposure | BTC dominance versus meme relative performance; raw larger meme moves | Relative performance is not absolute direction. Ordinary greater BTC sensitivity may still be useful, but increases risk too. |
| Recover a promising older idea | High-attention, high-volume, non-extended breakouts | Historical source builder is missing; the label measures the larger excursion side, not closing profit. Needs reproducible inputs and later confirmation. |

These are reusable roles, not six independent proven trading edges. Keep overlapping
CPI, news, volume and BTC-response findings connected rather than counting each
description as a separate discovery.

### All 65 original checklist records: reassessment dispositions

The identifiers below match the checklist. A parked tested version does not reject
every rational conditional version; equally, a plausible explanation does not rescue
a version that failed a fair comparison.

| Hypothesis | Current interpretation and next use |
|---|---|
| `rare_custom_pattern_entries` | User-reported historical potential; exact patterns and evidence still need mapping. Not internally confirmed. |
| `gamma_bull_compression_breakout` | Historical candidate. Audit the selected threshold, overlapping triggers and decision-time features before rebuilding it. |
| `macro_context_liquidity_stress_breakdown` | Source-readiness limited; cannot turn the old score into a current macro signal. |
| `vah_rejection_orderbook_breakout_failure` | Initial orderbook improvement did not repeat in the broader tested model. Preserve the setup, not a claim that all orderbook information fails. |
| `support_touch_fakeout_orderbook` | Small historical improvement remains a watchlist, requiring cleaner sequences and source coverage. |
| `historical_news_relief_bounce` | Exact broad hourly combination did not repeat against comparable range-low setups. |
| `historical_news_breakdown_continuation` | Direction improved over breakdown-only controls, but stale timing and other comparisons limit attribution. Conditional lead, not ready. |
| `historical_news_attention_compression` | Small inconsistent improvement; park this encoding rather than tune its thresholds. |
| `historical_news_attention_breakout` | Preserve the strongest old news/setup direction lead; source reconstruction and fresh confirmation are prerequisites. |
| `historical_news_bad_news_absorption` | Fair support-state comparisons were too scarce. Unanswered, not rejected. |
| `historical_news_crypto_stress_range_high` | Tested broad stress encoding did not help; specific adverse stories near resistance remain a different question. |
| `historical_news_macro_stress_ignored_rising` | Some combined-outcome improvement, but direction and current-versus-stale timing were unstable. Background watchlist only. |
| `sieve3_prior_high_coevent_filter` | Small model-ranking improvement with sparse companions; not a validated entry filter. |
| `sieve3_prior_high_h4_supply_overlap` | Three late co-events cannot establish the relationship. Preserve pending independent examples. |
| `sieve3_side_specific_exit_timing` | Post-exit behaviour is selected by the existing trade/exit process. Earlier/later/no-exit comparisons are still needed. |
| `mrz_8h_round_cluster_location` | Early association survived some controls, but later tests did not isolate a special round-number price. Preserve broader geometry. |
| `mrz_vp_attribute_roles` | Split the early bundle: LVN activity is conditional; HVN/POC did not broadly confirm in memes. Avoid one blanket retained label. |
| `mrz_mirrored_extreme_bollinger_cluster` | Later component comparisons were sparse and period results mixed. Original 12/13-path screen is not current confirmation. |
| `mrz_market_regime_modulation` | Original batch completed but lacked adequately populated comparisons. Market conditions remain a valid modifier question. |
| `mrz_g25_market_state_activity` | Promising unsigned activity ranking; keep its exact fresh-confirmation contract unchanged. |
| `mrz_g25_meme_convergence_crossing` | Incomplete crossing-location lead, not a passed result. Wait for its already-frozen later test. |
| `mrz_g26_state_location_interaction` | No complete combined result in this formulation. Do not tune the reused holdout or reject all context/level interactions. |
| `mrz_daily_session_vwap_center` | Incomplete crossing lead with a frozen future test; not a direction rule. |
| `event_fomc_market_activity` | Keep activity warning. Too few actual decision surprises to judge that particular direction route. |
| `event_gdelt_market_activity` | Weak activity/readiness input, not signed story meaning. |
| `event_negative_background_positive_move_fade` | Keep conditional historical reversal lead, with small sample and four-hour decision delay explicit. |
| `event_local_single_level_activity` | Preserve narrow event/coin/horizon activity modification; no level-family ranking or direction follows. |
| `event_first_hour_broad_continuation` | Broad screen did not survive its search-size check. No general continuation signal established. |
| `event_prior_month_range_edge_activity` | Isolated activity cells did not form a repeated pattern; continuous or event-specific location questions remain distinct. |
| `event_story_semantic_incremental_value` | Tested hourly labels failed. Only one major story and no independent-source confluence: major drivers and accumulation were not settled. |
| `event_scheduled_macro_families` | Previous-release-change direction is parked; real expectations have different source readiness by release family. |
| `event_vix_nasdaq_risk_activity` | Keep activity context; standalone signed direction was not demonstrated. |
| `context_google_trends_attention` | Tested attention representation was weak/inconsistent. No defensible sign was available; it did not test signed narrative effects. |
| `context_live_web_announcements` | Keep unsigned activity association; generic first-hour direction did not confirm. Semantic pilot is completed, not pending. |
| `context_live_news_activity` | Same separation: observed activity association, not proof of story meaning or a standalone directional rule. |
| `context_global_flows_and_risk` | Mixed umbrella: dominance survived as relative information; several flows/risk sources were sparse or unstable. Do not label the whole bundle worthless. |
| `context_btc_dominance_relative_rotation` | Historical BTC-versus-meme relative-performance lead; cannot say whether either rose. |
| `context_orderbook_event_confirmation` | Recorded coverage passed, but only one scheduled event and one meme coin overlapped. Major-event usefulness remains untested. |
| `event_unscheduled_major_story` | Preserve; the one-major-story pilot cannot answer this broad question. |
| `event_multi_story_narrative_balance` | Preserve; distinct story identities and comparable market meaning are still prerequisites. |
| `event_family_confluence_activity` | Narrow ETH activity lead from overlapping families; not multiple independent findings or established direction. |
| `event_reaction_zone_direction_confirmation` | Carry into decision-time input assembly; single levels and clusters must both remain available. |
| `event_btc_alt_meme_transmission` | Keep raw amplification and participation. Later direction is exploratory; absence of extra amplification after adjustment does not erase ordinary sensitivity. |
| `event_one_minute_direction_replay` | Keep bounded microscope lane; distinguish selected known reactions from advance forecasts and reconcile existing replay defects first. |
| `event_live_media_three_hour_path` | Other AI's nine-coin lead lacked matching market-only/volatility-only/timing controls and used overlapping hourly labels. Secondary only. |
| `event_cpi_short_reaction_direction` | Retain prior proxy history, not an independent discovery from CPI expectation agreement. |
| `event_cpi_expectation_surprise` | Best current short-direction event lead; freeze and confirm on later whole releases, with actual information availability explicit. |
| `event_cpi_immediate_leader_transmission` | Leader added only one correct call over follower self-movement on 25 cases and failed search adjustment. Park this tested increment. |
| `event_cpi_meme_response` | Inadequate comparable history for the frozen ten-coin cohort; no reliable directional conclusion. |
| `event_global_central_bank_decisions` | Narrow BoJ clock-only activity result did not confirm. Expectations, guidance and other banks are not thereby disproven. |
| `event_bond_credit_liquidity_shocks` | Tested rating/refunding clocks did not retain activity; rare crises and genuine surprises remain coverage-limited separate questions. |
| `event_trade_geopolitical_policy_shocks` | Generic Council publication times did not repeat; unexpected severity, credible relief and background-conditioned effects remain unresolved. |
| `event_systemic_corporate_financial_shocks` | Keep earnings-time activity association only. First-public timing and equity-market controls do not support SEC causal attribution. |
| `event_non_us_regional_macro_releases` | UK/Japan/China tested clock warnings were weak; Eurostat was not market-tested. No blanket rejection of non-US influences. |
| `event_employment_expectation_surprise` | Promising descriptive direction, but later ordinary-period comparison support is insufficient. Preserve unchanged. |
| `event_freqai_recent_market_activity` | Keep simple volume persistence as baseline. Older repeated-anchor model is unusable; continuation prediction does not identify the initiating cause. |
| `event_freqai_cross_market_activity` | Old association did not provide a dependable later increment over simple volume. Old trained model is not reusable. |
| `event_freqai_level_modifiers` | Only a narrow historical BTC clue remains. Repair the evidence link; do not reuse the repeated-anchor model. |
| `event_freqai_breadth_direction` | Selected old direction cells did not survive integrity/search checks; not a deployable model. |
| `event_freqai_joint_call_calibration` | Too few issued calls and flawed training surface. Failure is not proof that joint prediction is impossible. |
| `event_whole_episode_interaction_reanalysis` | Completed integrity work, not an additional market edge. Keep its causal-role and episode-weighting corrections. |
| `event_cross_market_family_overlap_activity` | Later support was too weak for the family-matched version; no new model justified from it. |
| `event_aligned_signed_accumulation_semantics` | Adding incompatible signs was a representation error, not market evidence against accumulation. |
| `event_outer_range_discrete_activity_modifier` | Binary outer-fifth version did not retain a relationship. Other local/range roles remain distinct. |
| `event_crypto_structural_anticipated_events` | Source catalogue/case-study material only; few independent programmes and incomplete first-public clocks prevent general inference. |

### Corrections and technical verification completed

1. Replaced stale early-generation next actions for round levels, VP attributes,
   mirrored clusters and market-state modification with the later evidence.
2. Corrected the overly broad adjusted-meme-amplification status; clarified that raw
   larger movements remain a possible exposure-selection input, not proven profit.
3. Made the completed semantic pilot's narrow coverage explicit. Its negative averages
   do not answer the user's major-story and several-distinct-stories hypotheses.
4. Repaired the missing `event_freqai_level_modifiers` evidence path and resolved four
   shorthand historical evidence references to existing archive files. Archive reads
   here were explicitly for the user-requested historical reassessment.
5. Delegated one isolated code repair to Luna MAX. In
   `market_reaction_zone_generation17_one_minute_analysis.py`, rejection approached
   from below now points down; rejection from above points up. Only
   `approach_rejection` and `density_approach_rejection` change. Historical output
   artifacts and thresholds were not rewritten or rerun.
6. Parent inspected the repair and ran the combined one-minute/preflight tests:
   **21 passed**. Ruff passed for the two changed research/test files. Existing
   portfolio checksum/coverage validation passed: **five families, nine prototypes,
   104 coverage rows**. This validator does not prove every source was newly audited
   or that the event-time joins and performance tests have been completed.
7. A second Luna helper for evidence assembly hit its usage limit and returned no
   usable review. Parent completed the inventory review; no retries or unreviewed
   delegated interpretation were accepted.

The repaired one-minute sign does not retroactively validate a strategy. A bounded
score-impact replay remains queued. The old 40-event joint reaction-and-direction gate
was also capped below 55% by the observed reaction rate at every tested horizon; its
failure must not be used as a universal rejection of conditional direction.

### Next complete development batch: reuse before new discovery

Baseline: the existing Stage-1 source contracts and nine prototypes, plus the
reassessed checklist above. The development question is whether they can be assembled
at honest decision times without losing local levels or confusing one event with many
independent observations. This batch is input reconstruction and integrity checking,
not another broad search for winning parameters.

| Work item | Required result | Boundary |
|---|---|---|
| 1. Decision-time signal rows | Reuse existing source readers/contracts; preserve parent episode, coin, source availability, decision time, role and explicit no-call reason | Release information, initial response and later prediction must use different honest decision times. |
| 2. Local level map | Join only already-known single levels and independent-family clusters at source and confirmation decisions; preserve timeframe, distance, side and available opposing-level information | Stage-1's 60 map rows describe derivable support, not completed event-time joins. Do not invent missing geometry or automatic higher-timeframe precedence. |
| 3. Broad five-family coverage | Represent event information, background/leadership, calculated areas, local participation and cross-asset response before exploring combinations | Nine prototypes are not nine proven independent mechanisms; unsupported sources remain absent with reasons. |
| 4. Baseline and episode integrity | One parent-episode identity across components; comparable controls and non-overlapping forecast intervals; preserve already-exposed versus genuinely later periods | No random candle split of one event, copied old model, or unrecorded change of success definitions. |
| 5. Small repair-impact replay | Re-score only the two repaired one-minute calls on their unchanged historical sample, then report reaction, conditional direction, joint success and abstention separately | Keep old artifacts and original gate; this is an error-impact check, not fresh validation or threshold search. |

Complete or explicitly park every item before assessing possible next branches. The
parent owns the exact questions, comparison design, acceptance decisions and result
interpretation. Cheaper agents may implement bounded adapters/tests and mechanical
data assembly; they may not choose favourable subgroups, change labels/thresholds,
expand families or decide what evidence means. Review code and synthetic tests before
opening new market outcomes. If a helper is unavailable, preserve the bounded handoff
rather than repeatedly retrying it or starting a different research branch.

After reconstruction, freeze a small component-versus-combination batch. Compare
event input alone, later response alone, local context alone and a limited justified
combination at the SAME decision time on comparable whole events. Distinguish a useful
historical input from a repeatable forecast, and a repeatable forecast from a
cost/execution-tested trading candidate. No blanket AUC or raw hit-rate threshold can
substitute for those separate questions. Strategy promotion/live trading remains a
separate approval boundary.

The three previously frozen G25 confirmation questions still require the
2026-09-20 07:00 UTC candle and their declared coverage checks. They were not opened
early or redefined during this review. Reconstructing old inputs must not contaminate
their reserved periods.

### Documentation drift found, without refactoring

The hypothesis checklist still pointed at several G1/G2 next steps while the detailed
relationship register and later generation reports already contained their follow-ups.
The recent-summary file also predates the September 11-13 findings and September 14
preflight. This is a handoff/summary fragmentation problem, not grounds to delete older
evidence. Current corrections stay in the approved checklist and this outcomes ledger;
no rules, objective documents or directory structure were changed. Future agents should
use these two as the current index and follow exact historical links only when needed.

## 2026-09-19 - Separate Assessment Criteria And Earlier Research Inventory

- User removed the blanket requirement for an idea to predict both abnormal activity
  and direction at a high joint rate. Objective 02b Section 15.4 now governs separate
  direction, activity/magnitude/timing, reaction-location and conditional-modifier
  assessments. Relevant objective, roadmap, direct-test, FreqAI discovery/promotion,
  method-reference and theory wording was aligned to that authority.
- A direction input can qualify without abnormal volume; a volume/location input can
  qualify without direction. Evidence quality, fair baselines, independent support,
  uncertainty and later confirmation remain. Combined inputs require their own test;
  do not multiply success rates or assume independence. Direction on all issued calls
  must be separate from the subset where a reaction later occurred.
- Original frozen scores and verdicts remain intact. Old joint-gated results receive
  a separately dated component-role reassessment rather than automatic promotion or
  silently changed thresholds. Existing research evaluators were not rewritten in
  this documentation change; their hard-coded legacy gates must not be mistaken for
  the new policy when reused. No market test ran during this update.
- User also requested a search for research before the 65-record checklist. That
  checklist is not a census of all research ever done in this repository. The earlier
  review covered its 65 records, not the entire historical programme.

### Earlier evidence map found

| Existing location | What it provides | Reuse boundary |
|---|---|---|
| `user_data/research_news_data/context_features/reports/trader_concept_lifecycle_ledger.csv` | 513 records, 335 unique concept IDs and 306 distinct question strings, verified using the existing concept reader | Repeated IDs/variants and overlap with current questions require semantic reconciliation; these are not 335 new independent discoveries. |
| `user_data/research_news_data/context_features/reports/trader_concept_cards.csv` | Compact concept descriptions, claimed best results, controls, evidence paths and next actions | Convenient index, not independent verification of its ratings. |
| `user_data/research_news_data/context_features/reports/trader_concept_evidence_inventory.csv` | Artifact paths, statuses, sizes and dates | Locate exact supporting reports before opening bulky outputs. |
| `ai_guidance_docs/99_archive/original_uploaded_docs/history_logs/objectives_progress.md` | Earlier question batches and follow-ups, including beta/gamma/delta, structure, orderbook and confluence work | Historical evidence only; old promotion language is not current authorization. |
| `ai_guidance_docs/99_archive/original_uploaded_docs/history_logs/context_research_detailed_findings.md` | Detailed older results and method corrections, including trade-management work | Read targeted sections, not the whole 442 KB log every turn. |
| `ai_guidance_docs/99_archive/original_uploaded_docs/history_logs/goal_status_tracker.md` | June queue IDs, completion states, old file locations and handoff links | A historical snapshot, not current process/data availability. |
| `user_data/research_news_data/context_features/reports/freqai_research_results_ledger.csv` | Approximately 101 MB of model/profile/window scores, modified through August 7 | Not manually loaded; use existing summary readers and narrow experiment IDs. Score rows are not independent hypotheses. |
| `user_data/research_news_data/context_features/reports/general_exit_freqai_results_ledger.csv` | Approximately 4.8 MB of earlier exit/context evaluations | Source index found; substantive findings not reassessed in this search. |
| `user_data/research_news_data/context_features/3ki_freqai_2020_2022/runs/` | Separate technical-confluence, broad GDELT state, news/price-family, refinement and exit-context sweep ledgers | Group smoke/revised/full runs; the folder names describe data periods, not necessarily execution dates. |

The old concept ledger's record counts by branch are: structure/volume 200, news 111,
orderbook 89, downside risk 44, trade management 35, regime confluence 21, merged
confluence eight, and entry-family merge five. Its promising/rejected/promoted labels
are inherited claims, not this agent's newly accepted conclusions.

Documentation fragmentation: `trader_concept_lifecycle_summary.md` was generated June 5
and totals 460 records, whereas the June 7 lifecycle CSV now contains 513. Preserve
both as historical evidence, use the CSV for that inventory count, and do not create
another competing master ledger. No old files were moved, regenerated or deleted.

Next bounded task: group the older records by actual market question, match them to
the current checklist, separate revised/duplicate runs, then review missing promising
and prematurely rejected families under the new role-specific criteria. Start with
the small cards/index and linked summaries; do not rerun hundreds of models or accept
old backtest profits as confirmation. Add only distinct, relevant questions to the
existing checklist, preserving the five-family breadth and later combination plan.

## 2026-09-20 - Legacy Corpus Inventory Reconciliation And Five-Family Component Audit

### Scope and evidence strength

- This is the first bounded reconciliation batch requested after the 19 September
  component-role change. It preserves every frozen output and original verdict. No
  market experiment, model training, strategy change, collector action or download ran.
- The approved `concept_lifecycle_ledger._read_csv(...)` reader parsed the June 7
  lifecycle CSV as **513 records, 335 concept IDs and 306 question strings**. The
  builder/main was not run because it would rewrite historical artifacts and reapply
  obsolete status heuristics.
- All 513 rows are accounted for at inventory level. This is **not** exhaustive method
  validation of 335 concepts. Only the balanced five-family evidence sample below was
  traced through labels, decision-time definitions, controls and repetitions.
- The 48 concept cards are a promising-only convenience index. Their builder excludes
  deferred, needs-rework, rejected and candidate-for-strategy statuses, so they are not
  an unbiased census and cannot support a claim that omitted families were disproved.

### Complete inventory accounting

The apparent 335-concept corpus contains substantial repeated targets and model
versions. The following partition is mutually exclusive and totals 513:

| Inventory kind | Records | Concept IDs | Question strings | Interpretation |
|---|---:|---:|---:|---|
| Feature-transform/target screens | 170 | 114 | 170 | Forty-nine raw features were represented by levels, deltas, weekly z-scores or monthly ranks against four targets. Each transform/target row is not a separate mechanism. |
| FreqAI model/scope rows | 93 | 93 | 8 | Model family, profile and scope variants repeat eight generic market questions. `orderbook_present`, model type and smoke/revised/full siblings are not independent discoveries. |
| Named direct target rows | 198 | 76 | 76 | Named hypotheses repeat across as many as four binary or continuous targets on overlapping snapshots. Separate target roles, but not independent events. |
| Strategy/management comparisons | 52 | 52 | 52 | Thirty-five management, eight merged-confluence, five entry-merge and four other BTC candidate comparisons. These are iterative strategy questions, not component confirmations. |

The eight original branches reconcile as follows, including rejected and deferred rows:

| Legacy branch | Records / IDs / questions | Status coverage | Semantic checklist crosswalk and gap |
|---|---:|---|---|
| Structure/volume | 200 / 125 / 168 | 105 direct-promising, 44 rejected, 40 rework, 5 FreqAI-promising, 3 candidate, 3 sweep-promising | Existing compression, level, regime and cluster questions cover part of the family. Added distinct standalone price/pressure ranking and MTF-direction questions. |
| News/context | 111 / 73 / 33 | 108 deferred, 2 rejected, 1 rework | Existing macro stress, attention and later event-hierarchy rows cover active context. Added the missing quiet-background technical-break modifier; deferred status is source policy, not a negative result. |
| Orderbook state | 89 / 58 / 37 | 47 rejected, 17 FreqAI-promising, 14 rework, 9 direct-promising, 2 sweep-promising | Existing VAH rejection, support-touch fakeout and event confirmation rows cover some uses. Added liquidity transitions as a separate conditional risk/confirmation question. |
| Downside risk | 44 / 22 / 20 | 26 rejected, 8 rework, 4 sweep-promising, 4 direct-promising, 2 FreqAI-promising | Existing breakdown and event-risk rows do not preserve the reclaim-versus-continuation decision. Added that path question explicitly. |
| Trade management | 35 / 35 / 35 | 20 rejected-for-now, 9 candidate, 3 rework, 2 rejected, 1 historical promoted baseline | Current side-specific event/exit timing is not the same question. Added one evidence-only family-specific exit row; this does not reactivate strategy work. |
| Regime/confluence | 21 / 9 / 12 | 12 rejected, 7 direct-promising, 2 rework | Current regime and higher-timeframe checklist rows cover the modifier/location themes; its directional MTF subset is included in the new MTF-direction row. |
| Merged confluence | 8 / 8 / 8 | 6 candidate, 1 rejected, 1 rework | Semantically overlaps rare-pattern integration and portfolio assembly. Preserve as historical strategy combinations; no new component row is justified. |
| Entry-family merge | 5 / 5 / 5 | 5 rejected | Semantically maps to rare-pattern integration. Priority/addition variants are one strategy-integration family, not five new hypotheses. |

There is no unmapped remainder at this branch/family inventory level. The incompletely
method-reviewed remainder is still pending: the grouped crosswalk does not validate
every threshold, model window, source mask, exit rule or 3ki smoke/revised/full run.

### Balanced five-family evidence audit

| Family | Original question, label and decision time | Controls and repetition traced | Supplementary component-role conclusion |
|---|---|---|---|
| Standalone momentum/pressure and MTF structure direction | Completed one-hour price/pressure or MTF bullish-state fields attempted to rank `breakout_success_next_6h` or `breakdown_success_next_6h`. Those labels require a future six-hour extreme to cross the prior completed 24-hour high/low and the six-hour close to remain beyond it. Continuous upside targets are future path maxima, not binary success. | The feature screen covered 2,988 candidates and 11,761 feature-target rows; its tree used a chronological 70/30 split, but feature screening used the full surface, so the shortlist is development-selected. Direct MTF tests used same-setup, deterministic random, shuffled-score and price/structure controls. Several summary files repeat the same 55,971-row snapshot and are not replication. | Bearish pressure enriched a rare outcome: 23.5% in the selected tenth versus 9.5% overall, oriented AUC 0.669 and shuffled 0.502. That is ranking, not 23.5% direction accuracy. The MTF trigger subgroup was 75/131, 57.25%, versus 12.97% in same-setup rows, but score AUC 0.687 trailed the price/structure baseline 0.707; six-hour maximum upside was 0.852% versus 0.722%, and 24-hour 1.673% versus 1.526%, with no better correlation than baseline. Preserve both as development questions, not confirmed issued-call direction. |
| Support reclaim versus downside continuation | At the current completed hour, `early_downside_break_visible` used current/preceding price-break state. `support_reclaim_next_6h` means the current close is at or below the prior completed 24-hour low area and the six-hour close later finishes back above it. Continuation labels ask for further adverse path/close after this decision. | Direct tests compared trigger, same-regime and random rows plus shuffled and price/structure scores. The later FreqAI reclaim result had 227 Q1-2026 rows and a price-tree control. Two lifecycle scope names contain the same 227 rows and metrics, so they count once. The continuation trigger had only nine rows. | Reclaim score AUC 0.791 versus 0.570 baseline supports historical ranking, but the broad trigger's reclaim occurrence was 3.0% versus 16.8% in its same-regime complement, so it is not a ready trigger rule. FreqAI AUC 0.807, AP 0.180 and +0.032 AUC/+0.022 top-bucket deltas are modest incremental ranking evidence, not 80.7% success. Continuation evidence is underpowered rather than rejected for missing 55%. |
| Quiet-news background modifying technical breaks | The intended question is conditional: when timestamp-safe news activity is genuinely quiet, does the same completed technical break behave differently? The legacy beta row instead used a broader `context_present` quiet composite with a current technical break/volume trigger. | Same-setup, random, shuffled and price/structure controls were reported. The bullish trigger had 100 rows and 10/13 positive monthly AUC windows. The source audit shows only 112 usable article-activity rows, while the beta test admitted 5,670 broader context/technical rows. `_defer_if_unready` marks every news/context family deferred regardless of result. | The 60.0% trigger label rate versus 14.1% same-setup is a development clue, but AUC 0.619 trailed the 0.709 price/structure baseline and the representation did not isolate verified quiet news. Deferral is not negative evidence. Preserve a distinct conditional-modifier question; first separate source-present quietness from missingness and compare the same technical setup with/without quiet background. |
| Orderbook transitions as risk/confirmation | The named questions concern support removal, resistance retreat, wall movement and pressure changes around an already visible technical state. They are conditional risk or confirmation questions, not a generic orderbook-present filter. | Direct transition rows used technical setup, random and shuffled comparisons; later model rows used price controls. Historical source audits state maximum source timestamps were bounded by the candle and gaps were not carried, but this reassessment did not rebuild raw Bybit archives. Model/scope siblings often repeat the same rows and are not independent. | Historical support-removal triggers enriched breakdown labels. One February model had AUC 0.730, AP 0.422, +0.094 AUC and +0.047 top-bucket delta versus price; other setup-specific scopes were neutral/adverse and later direct ranking did not uniformly beat price/structure. Preserve **historical conditional increments/mixed evidence** for targeted risk/confirmation roles. Do not infer broad utility, profit or a complete trade. |
| Historical targeted exits/management | Questions asked whether named, already-open entry families should exit on orderbook escalation, reclaim, structure failure, pressure fade or time failure. Outcomes were full-strategy return/drawdown/trade statistics, not component probabilities. | The exact evidence path missing from the lifecycle row was recovered from the archived progress note: `production_alpha_current_best_targeted_exit_comparison_20260606.csv` and its backtest ZIP. The comparison tested several exits on the same 2020-2026 BTC development history, then selected descendants; this is iterative in-sample selection, not unseen confirmation. No matched still-open/earlier/later/no-exit causal counterfactual was reported. | The targeted long orderbook-exit variant improved the same-window historical comparison while broad exit stacks damaged it; strict short reclaim exchanged some return for lower drawdown. Preserve this as evidence that family-specific management deserves a proper counterfactual test in a future authorized exit objective. Do not verify the huge profit claims here, reactivate code, or treat AUC/backtest return as probability. |

The audit deliberately does not require every component to predict both abnormal
activity and direction or already form a complete trade. The old joint gate is not
replaced by a trading-profit gate. Each retained item is labelled only for its
supported ranking, conditional-modifier, risk/confirmation or management-evidence role.

### Checklist changes and limitations

- Updated `historical_research_inventory_reconciliation` from inventory-located to
  first-batch-complete with the remaining corpus explicitly pending.
- Added six materially distinct questions: standalone price/pressure ranking, MTF
  breakout direction, reclaim versus continuation, quiet-background modification,
  orderbook transitions, and family-specific management evidence. No old result was
  promoted and no hundreds-of-rows duplication was added.
- The lifecycle ledger's malformed timestamp in the `evidence_paths` field for
  `btc_current_best_pack2_long_orderbook_crash_exit` remains untouched as frozen
  history. This dated reassessment supplies the recovered exact paths alongside it.
- The approximately 101 MB FreqAI ledger and 4.8 MB exit ledger were not dumped or
  manually reviewed. Only compact lifecycle/card summaries and directly linked small
  reports were used. The 3ki run folders remain grouped because smoke, revised and full
  variants may overlap and are not independent ideas.
- Direction accuracy on all issued calls is unavailable for the audited legacy
  rankings. Trigger label occurrence, conditional outcome rates, AUC, continuous path
  means and strategy return remain distinct quantities.

### Smallest balanced next validation batch - not launched

1. **Price/pressure and MTF direction:** freeze one bearish-pressure feature, one
   price-momentum control and one MTF-structure block; compare each and the one justified
   combination on the same later chronological episodes. Report issued calls, coverage,
   direction accuracy, rare-outcome enrichment and continuous path magnitude separately.
2. **Reclaim versus continuation:** freeze one downside-break decision mask, then test
   reclaim score and continuation score on the same eligible episodes against price-only
   and structure-only controls. Count the duplicated 227-row scope once and require a
   genuinely later period.
3. **Quiet background:** run only after a source-readiness freeze proves source-present
   quiet rows. Compare technical break alone, quiet background alone and their modifier
   interaction; source-missing rows are a separate control, never quiet observations.
4. **Orderbook transition:** freeze one downside support-removal transition and its
   mirrored upside resistance-retreat transition. Compare each with the same technical
   state without the transition on later source-ready history; report risk/confirmation
   effects without a generic orderbook filter claim.
5. **Exit evidence:** under the current objective, perform no strategy test. If a future
   exit objective explicitly authorizes it, first reconstruct the selected entry
   families with matched still-open, earlier, later and no-exit paths on untouched
   chronology before selecting any exit action.

No new framework is needed to prepare the first four tests: reuse the current frozen
feature definitions, source-readiness masks and direct-result readers. Before any run,
freeze one compact batch manifest with exact later period, eligible episode definition,
controls, thresholds and pass/fail interpretation. Exit reconstruction remains a
separate authorization boundary and may require a narrow read-only adapter only after
the existing exit readers are checked; none was created in this batch.

## 2026-09-23 - Frozen fresh market-activity and location checks

- Scope: Reopened the three questions frozen on 2026-09-03 without changing their periods, features, controls or pass rules. This is one breadth batch, not a search for the best post-hoc rule.
- Source/data touched: Selectively appended 1h Binance futures OHLCV for the 19 distinct frozen pairs. All 20 pair/cohort coverage cells had zero missing timestamps, duplicate timestamps or hourly gaps through the required 2026-09-20 07:00 UTC candle. FreqAI ran 16 exact model/control profiles with two single-threaded profiles at a time. Bulky outputs are on D:; compact results remain under the existing `fresh_confirmation` research tree.
- Snapshot/window: `fresh_early` 2026-08-20 through 2026-09-04 and `fresh_late` 2026-09-05 through 2026-09-19, inclusive decision hours. The first block began before its 2026-09-03 freeze, so it is a later-unopened historical block, not wholly prospective evidence. The late block is later chronology; FreqAI walk-forward training may use older first-block outcomes when predicting later dates.
- Timestamp/lookahead checks: Frozen feature equations use completed prior candles and prior cross-asset states; target scale was fitted before 2026-08-20 with an eight-hour purge and held fixed. Support gates and the VWAP park decision were locked before future outcomes were read. No signed direction, profit or trade decision was read.
- Result in trader language: Recent own-coin and broad-market behaviour can flag when upcoming trading is likely to be unusually busy. It does not explain whether news caused the earlier activity, say which way price will move, or establish that a particular level causes a reaction. The exact pooled meme round-number/distribution combination did not repeat against all fair alternatives; some general calculated-location comparisons remained positive. The daily VWAP question could not be answered because its shifted historical comparison rarely contacted price.
- Key metrics: Market activity passed 14 of 15 preselected scope/target checks in both fresh periods across all four model/seed cells against constant, shuffled-label, simple-recent and 24h-old controls. This is **one correlated activity family**, not 14 independent discoveries. Meme two-hour range was the exception: all four models ranked it better but the simple rule had slightly lower absolute error in the late block. Across selected rows, minimum top-quarter above-training-median fractions were 0.64 for memes and 0.66 for normal coins. Pooled meme convergence failed its eight-control whole-cohort rule; late recent-analogue difference was -0.126 with only two of ten coins positive. VWAP support had zero eligible coins early and one late against its five-coin minimum, so VWAP future outcomes remain unopened.
- Artifact paths: `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings/g21_broad_siblings/g22_broad_siblings/g23_broad_siblings/g25_broad_siblings/fresh_confirmation/freqai_state/fresh_state_freqai_20260923a/fresh_state_review.json`; sibling `outcomes/two_supported_routes_20260923a/meme_crossing_review.json`; sibling `support/fresh_confirmation_support_20260923a/fresh_confirmation_support_result.json`. The three existing checklist records carry direct links.
- Verdict: Retain unsigned market-activity warning as one input; park this exact special convergence claim while preserving the broader level-location question; leave daily VWAP unanswered for sparse-control support. No activity-to-direction or trade-rule promotion.
- Next action: Complete the next broad event/background/leader/location batch before branching into detailed combinations. When activity warning is later combined with news, market leadership or nearby levels, compare identical event/setup states with and without it, and never label a downstream volume response a rival root cause of the event.

## 2026-09-23 - Repaired one-minute rejection-call impact

- Scope: Completed only work item 5 of the already-queued five-family reconstruction batch. Re-scored `approach_rejection` and `density_approach_rejection` after the documented sign repair; all other one-minute methods and their old results were left untouched.
- Source/data touched: Hash-verified the original Generation 17 40-episode contact-state, future-outcome and call CSVs through `g17_one_minute_analysis_record.json`. Reused `causal_calls`, `evaluated_calls` and `call_summary` from the existing research modules in a read-only replay. No new market data, threshold, sample, model or file was created.
- Snapshot/window: The same 40 historically selected calculated-area contacts, four post-contact horizons (15, 60, 240 and 720 minutes), 25 issued approach calls and 13 high-density subset calls. This is not a new confirmation sample.
- Timestamp/lookahead checks: The replay changed only the direction assigned to a known approach side. It joined old and new rows on the same episode, method and horizon, yielding 320 matched rows; 152 method/horizon rows across 25 episodes changed sign. Reaction outcomes and abstentions were unchanged.
- Result in trader language: The old approach-rejection direction was physically reversed. Once corrected, the broad approach idea was less accurate on this selected sample. The high-density subset happened to get 8 of 13 callable directions right overall, but that is far too few independent examples and its success when a reaction actually occurred varied sharply by horizon. Do not turn the old or repaired percentages into a trading claim.
- Key metrics: Corrected broad approach all-callable direction was 47.6%, 45.8%, 48.0% and 48.0% over 15/60/240/720 minutes. Corrected high-density direction was 61.5% on 13 calls at each horizon; among issued calls with a reaction, direction accuracy was 50%, 50%, 33% and 25%. Joint reaction-and-correct-direction among all 40 episodes was 15%, 10%, 2.5% and 5% for broad approach, and 12.5%, 10%, 2.5% and 2.5% for the high-density subset. Reaction rates alone were 52.5%, 42.5%, 20% and 25%; therefore a joint-rate failure cannot by itself reject direction, but no direction component is confirmed here.
- Artifact paths: Original unchanged `user_data/research_news_data/context_features/market_reaction_zones/generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/g17_broad_branch_layer/one_minute_direction/g17_one_minute_direction_20260822a/g17_one_minute_analysis_record.json`; repaired call code `user_data/Custom_Launcher/research/context_features/market_reaction_zone_generation17_one_minute_analysis.py`.
- Verdict: Integrity repair impact checked; no one-minute directional method promoted. The wider evidence-triggered one-minute lane stays queued, but requires fresh independently frozen episodes and separate reaction, direction, joint and abstention reports.
- Next action: Continue work items 1-4 of the five-family input reconstruction before any branch or portfolio combination test.

## 2026-09-24 - Stage-2 source decision rows, outcome-blind

- Scope: Began work item 1 of the queued five-family reconstruction. Reused the verified Stage-1 source contracts and assembled seven of nine prototypes at their own decision times. This is an input-integrity step, **not** a new market-effect test or trading result.
- Output: `user_data/research_news_data/context_features/event_hierarchy/market_signal_portfolio_stage2_20260924a/source_decision_rows.parquet` has 6,863 source-native decision rows; its sibling `source_assembly_result.json` records counts, hash, exclusions and limitations. Runner: `user_data/Custom_Launcher/research/context_features/market_signal_portfolio_stage2_sources.py`.
- Meaning: CPI and FOMC rows retain their official-event identity; live news/web and BTC-dominance rows retain source-specific identity and readiness; recent-volume rows preserve their multi-parent episode lists; the BTC-to-group rows begin only after one hour of leader observation; the negative-background/positive-initial-move rows begin after four hours. Five unready web observations are explicit no-calls, not quiet-news observations. No later price/volume outcomes were read or scored.
- Integrity correction during assembly: Matched background controls had their **own historical control clocks**. The first assembly check caught duplicate-looking rows caused by assigning the parent event clock to every control. The final artifact joins each control by event ID, control type and rank to its actual control timestamp, then places the decision four hours later. This is a source-clock correction, not a market result. Two targeted tests and Ruff pass.
- Remaining main-batch work: The other two prototypes—calculated-area contacts and the local multi-timeframe support/resistance map—are not in this artifact. Stage-1 had only metadata coverage for the map, not event-time coordinates. Existing G17-G24 builders can generate causal surfaces, but their persisted contact/control files cannot be mistaken for an arbitrary event-time map. Materialize a bounded per-pair coordinate surface, backward-as-of join it to source and later confirmation decisions, and retain single levels separately from true price-proximate independent-family clusters. Then complete the five-family coverage and episode/control integrity audit before testing combinations or interpreting outcomes.

## Archive Reference

For old detailed history, open only when verifying evidence:

- `ai_guidance_docs/99_archive/original_uploaded_docs/history_logs/context_research_detailed_findings.md`
- `ai_guidance_docs/99_archive/original_uploaded_docs/reviews/review_code_data_correctness_20260529.md`
- `ai_guidance_docs/99_archive/original_uploaded_docs/reviews/review_objective_alignment_20260529.md`
