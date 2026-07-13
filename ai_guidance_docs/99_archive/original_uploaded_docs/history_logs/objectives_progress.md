# Objectives Progress

This file is for progress, status, implementation notes, queue state, and blockers. It was migrated from the old mixed `current_objectives.md` file, so older sections below may contain objective wording as historical context. Do not treat this file as the canonical objective list.

Canonical objectives now live in `ai_guidance_docs/current_objectives.md`. Objective examples live in `ai_guidance_docs/objective_examples.md`.

## 2026-06-07 Multi-Path Strategy Plan Update

1. Added compact production-alpha dashboard script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_strategy_dashboard.py`
2. Generated compact decision artifacts:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_candidates_20260607.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_candidates_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_next_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_next_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_decision_snapshot.md`
3. Review fixes:
   - Dashboard now skips legacy promotion CSVs without the newer `role` column instead of crashing.
   - Lane classification now uses metrics as well as strategy names, so the current highest-return branch is correctly labelled `max_return`, not `lower_drawdown`.
4. Updated guidance:
   - `AGENTS.md` and `ai_guidance_docs\README.md` now tell production-alpha agents to read the compact decision snapshot first.

## 2026-06-07 Earlier Multi-Path Strategy Plan Update

1. Updated `ai_guidance_docs/current_objectives.md` with the full multi-path strategy development plan.
2. Added explicit lanes for:
   - max-return branch,
   - balanced return/drawdown branch,
   - cleaner high-profit-factor branch,
   - sparse high-quality branch,
   - position-management/stacking branch,
   - orderbook crash-exit/risk branch.
3. Added first-priority next tests:
   - smaller add sizes,
   - profitable-only adds,
   - different-family/source adds,
   - hold-longer-on-stacked behaviour,
   - family-specific exits,
   - orderbook as crash/invalidation warning.
4. Added promotion discipline:
   - candidate table first,
   - failure-mode audit,
   - separate lane development,
   - robustness checks,
   - later cross-coin testing.

## 2026-06-07 Signal Stacking / Position Management Test

1. Clarified `current_objectives.md`:
   - Do not collapse strategy research into one "best" answer.
   - Preserve multiple viable paths: max-return, balanced risk, lower-drawdown, sparse high-quality, and position-management variants.
   - Several strategies may eventually run side by side or behave differently on other coins.
2. Added two refinement branches after the sharp selective stacking run:
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackSharperReserveAddOnStrategy`
     - Keeps the weak overtrade resistance-break family as a normal entry, but removes it from reserve/add eligibility.
     - Result: `330` trades, `3867.94%` return, `4.33%` drawdown, `2.85` profit factor, `55.5%` win rate.
     - Stacked subset: `39` trades, `82.1%` win rate, `2.93%` average profit ratio, `23.36` profit factor.
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackSharpNoOvertradeFamilyStrategy`
     - Removes the weak overtrade resistance-break family entirely.
     - Result: `304` trades, `3722.38%` return, `4.32%` drawdown, `3.06` profit factor, `58.6%` win rate.
     - Stacked subset: `40` trades, `82.5%` win rate, `2.99%` average profit ratio, `24.12` profit factor.
3. Interpretation:
   - The sharper reserve branch is better for total return among stacking variants.
   - The no-overtrade-family branch is better for win rate/profit factor and is a cleaner risk-quality branch.
   - Both remain below the max-return baseline return, but both support the user's point that lower downside and clearer logic can justify keeping multiple paths alive.

## 2026-06-07 Earlier Signal Stacking / Position Management Notes

1. Updated guidance so overlapping signals are treated as position-management evidence, not only duplicate entries:
   - same-direction overlap can support add/hold/confidence,
   - opposite-direction overlap can support reduce/tighten/exit,
   - this remains a flexible test idea, not a hard production rule.
2. Added and ran a signal-stacking audit:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_signal_stacking_audit.py`
   - Audit source: `backtest-result-2026-06-07_00-46-44.zip`
   - Signal source: `trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_current_priority.parquet`
3. Direct audit result:
   - `125` overlap events inside open trades.
   - Same-direction overlaps were strong: `123` events, parent-trade win rate `88.6%`, average remaining trade-direction return `1.93%`.
   - Same-direction trades were much better than non-overlap trades:
     - same-only overlap: `92` trades, `85.9%` win rate, `2.98%` average profit, `14.47` profit factor.
     - no overlap: `238` trades, `45.4%` win rate, `0.57%` average profit, `1.67` profit factor.
   - Opposite-direction overlap had only `2` events, so it is a warning clue only, not enough for a production exit rule.
4. Added Freqtrade-compatible add-on strategy classes:
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackAddOnStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy`
5. Freqtrade implementation findings:
   - Add-on with full baseline stake and `--max-open-trades 1` produced `0` add orders because the baseline spends nearly the full wallet on entry.
   - Running without `--max-open-trades 1` was invalid for baseline comparison because it divided unlimited stake across ten possible lanes and/or traded extra configured pairs.
   - Reserve/add variant used `80%` initial stake and added the reserved `20%` only after a later same-direction signal.
6. Reserve/add Freqtrade result:
   - Strategy: `TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy`
   - `329` trades, `2052.49%` return, `4.24%` drawdown, `2.60` profit factor, `53.2%` win rate.
   - Baseline remains stronger overall: `332` trades, `4764.00%` return, `5.30%` drawdown, `2.75` profit factor, `56.6%` win rate.
7. Important sub-result:
   - `93` trades received same-direction add orders.
   - Stacked subset: `77.4%` win rate, `2.41%` average profit ratio, `8.86` profit factor.
   - Non-stacked subset: `236` trades, `1.37` profit factor.
8. Decision:
   - Do not promote the reserve/add strategy as the new whole-system baseline.
   - Do promote same-direction overlap as a validated position-sizing/confidence clue.
   - Next useful work is selective reserve/add logic by entry family or market state, plus a richer signal file that preserves multiple same-hour signals instead of one priority signal per hour.
9. Selective reserve/add follow-up:
   - Added `TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy`.
   - It reserves stake only for nine entry families where overlap audit evidence looked strong, and keeps full baseline stake for other families.
   - Freqtrade result: `330` trades, `3279.01%` return, `4.32%` drawdown, `2.81` profit factor, `54.5%` win rate.
   - Baseline remains stronger on return: `4764.00%`, `5.30%` drawdown, `2.75` profit factor, `56.6%` win rate.
   - Stacked subset: `54` add trades, `79.6%` win rate, `2.72%` average profit ratio, `14.29` profit factor.
   - Decision: do not replace the return baseline, but keep this as a risk-adjusted/position-management branch because drawdown and profit factor improved while return stayed well above market.
10. Sharp selective reserve/add follow-up:
   - Added `TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy`.
   - It removes the marginal prior-month-high VP node family from reserve/add eligibility after the selective run showed weaker behaviour there.
   - Freqtrade result: `330` trades, `3748.10%` return, `4.32%` drawdown, `2.87` profit factor, `55.2%` win rate.
   - Stacked subset: `47` add trades, `83.0%` win rate, `3.12%` average profit ratio, `22.98` profit factor.
   - Baseline remains stronger on return: `4764.00%`, `5.30%` drawdown, `2.75` profit factor, `56.6%` win rate.
   - Decision: keep the sharp selective branch as the best risk-adjusted signal-stacking branch so far, but do not replace the max-return baseline.
11. Audit tooling fix:
   - Fixed `production_alpha_backtest_audit.py` verdict logic so a branch cannot be labelled as replacing the current baseline when it underperforms the explicit baseline return.
   - Regenerated reserve, selective-reserve, and sharp-selective promotion reports with corrected verdicts.
12. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_current_strongest_signal_stacking.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_current_strongest_signal_stacking_signal_audit.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_09-52-42.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_selective_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_09-56-54.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharp_selective_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_10-01-03.zip`

## 2026-05-29 Source-Detail Reporting Split

1. Updated trader-confluence reporting so `context` and `structure` are no longer only opaque buckets.
2. Added granular source-detail groups for:
   - context: article/source activity, topic severity, GDELT events, GKG documents, Google Trends, BTC ETF flows, global/macro, availability metadata.
   - structure: VP, TLV2 support/resistance, BOS/CHoCH market structure, pattern geometry, cached price/volume state, composite setups.
3. Regenerated the column dictionary and direct-test markdown report with source-detail groups.
4. Verification: `py_compile` passed and `trader_confluence_direct_tests.py --tag source_detail_split --execute` completed.

## 2026-05-29 Beta Hypothesis Expansion

1. Added a beta pack of `31` extra named trader-confluence hypotheses after the strict source-detail gate reduced promoted evidence to one result.
2. Scope of the beta pack:
   - structure continuation/fakeout states
   - orderbook support/resistance removal, absorption, vacuum, and venue-pressure states
   - context risk/attention states
   - quiet-context technical break states
   - macro/context plus orderbook pressure-flip states
3. Implemented in:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_features.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_hypotheses.py`
4. Verification:
   - Python compile passed for both touched files.
   - Confluence snapshot rebuilt successfully with `55,971` rows and `2,096` columns.
   - Direct beta test completed with `557` detailed rows and `107` summary rows.
   - Beta threshold/component sweep completed with `24,455` evaluated threshold rows.
5. Direct beta watchlist positives:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`
   - `beta_context_risk_support_breakdown -> large_drawdown_next_6h`
   - `beta_structure_support_breakdown_continuation -> large_drawdown_next_6h`
6. Threshold sweep positives:
   - `831` passing threshold combinations across `7` hypothesis/target families.
   - Strongest incremental threshold result: `beta_ob_downside_vacuum_breakdown -> large_drawdown_next_6h`, AUC `0.646` versus baseline AUC `0.532`.
7. Caveats:
   - Many passing rows are near-duplicate threshold variants, so group-level interpretation matters more than raw pass count.
   - Snapshot rebuild produced pandas fragmentation warnings; this is a performance/code-cleanup issue, not a data-validity failure.
   - These are direct-test and threshold-sweep results only. They are not FreqAI strategy results yet.

## 2026-05-29 Gamma Bullish Breakout Suite

1. User clarified that structure-only trading hypotheses must not be rejected just because they are not news/orderbook confluence.
2. Added a gamma bullish breakout pack with `16` hypotheses:
   - `9` price/structure-only bullish breakout hypotheses.
   - `4` price/structure/orderbook bullish hypotheses.
   - `1` price/structure/context bullish hypothesis.
   - `2` broader confluence bullish hypotheses.
3. Added supporting state columns:
   - `conf_structure_bullish_state_score`
   - `conf_price_bull_trend_regime`
   - `conf_price_compression_24h`
   - `conf_volume_bullish_impulse_short`
4. Verification:
   - Python compile passed.
   - Snapshot rebuilt successfully with `55,971` rows and `2,352` columns.
   - Direct gamma test completed with `802` detailed rows and `156` summary rows.
   - Threshold sweep completed with `29,305` rows.
5. Result:
   - Best bullish breakout family so far is `gamma_bull_compression_breakout -> breakout_success_next_6h`.
   - Plain English: when price had been compressed/ranging, then broke upward with bullish volume, the next 6h breakout-success rate improved materially.
   - Representative threshold: `54` trigger rows, `51.9%` breakout success versus about `10.9%` in similar setup rows without trigger and `13.0%` in random eligible controls.
   - Model-style ranking score was better than price/structure baseline: AUC `0.658` versus baseline `0.629`.
   - Stability: positive in `11/13` monthly windows.
6. Caution:
   - Several bullish structure hypotheses scored well in absolute terms, but the price/structure baseline was slightly stronger, meaning they may not add much beyond the existing baseline formula.
   - The direct-test watchlist marked `gamma_bull_range_expansion_volume -> future_max_upside_24h`, but that is a continuous target lift, not a binary breakout-success result, so treat it as exploratory rather than promoted.
7. Storage check:
   - `C:\FreqTradeStuff\user_data\models` is about `28.2 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs` is about `0.45 GB`.
   - Reports and confluence cache are small enough to keep.
   - Do not delete model artifacts without explicit approval; old model directories can likely be reclaimed if CSV/Markdown reports are sufficient.

## 2026-05-29 Feature Discovery Layer V1

1. Implemented a feature discovery stage before more hand-authored hypotheses:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_discovery_candidate_builder.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_discovery_screen.py`
2. Builder behaviour:
   - Reads the frozen trader-confluence snapshot.
   - Selects a capped set of usable numeric base columns.
   - Excludes target labels, metadata, availability/debug columns, and hand-authored hypothesis outputs.
   - Builds controlled transforms only: current value, 3h/24h change, 168h z-score, 720h percentile rank, 24h above-mean, and rolling activity for binary flags.
3. Candidate build result:
   - Candidate parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_feature_discovery_v1.parquet`
   - Dictionary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_feature_discovery_v1_dictionary.csv`
   - Rows: `55,971`
   - Columns: `2,996`
   - Candidate feature columns: `2,988`
4. Screener behaviour:
   - Scores binary targets with top-decile lift, random controls, shuffled-label controls, monthly stability, and a tree-model shortlist check.
   - Rerun after fixing all-null train/test feature warnings in the tree-model path.
5. Clean screen result:
   - Combined report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_feature_discovery_v1_fixed_combined.csv`
   - Markdown report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_feature_discovery_v1_fixed.md`
   - Rows scored: `11,761`
   - Stable shortlist rows: `213`
6. Main discovery:
   - Price/range-position features dominate both breakout and breakdown success.
   - Low compression / high volatility state is strongly associated with large drawdown risk.
   - Bybit orderbook persistence/distance features add weaker but repeated drawdown-risk clues.
   - Context/news features did not materially survive this broad discovery screen.
7. Plain-English interpretation:
   - The data is saying "where price is inside its recent range" matters most for short-term breakout/breakdown success.
   - "Compressed or unstable range/volatility state plus volume" matters for drawdown risk.
   - Orderbook persistence disappearing or wall distances stretching looks more useful for risk detection than clean bullish breakout confirmation.
8. Remaining work:
   - Convert the top stable discovered feature clusters into new readable hypotheses.
   - Run direct hypothesis tests using those discovered components.
   - Use FreqAI only after the discovered clusters survive direct controls.

## 2026-05-29 Multi-Agent Objective Work Packages

1. Worker outputs created:
   - `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\source_coverage_audit.py`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_worker_b.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_worker_b.md`
   - `C:\FreqTradeStuff\ai_guidance_docs\hypothesis_redesign_plan.md`
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_promotion_readiness_plan.md`
2. Review outputs created:
   - `C:\FreqTradeStuff\ai_guidance_docs\review_objective_alignment_20260529.md`
   - `C:\FreqTradeStuff\ai_guidance_docs\review_code_data_correctness_20260529.md`
3. P1 review findings fixed or downgraded:
   - Coverage/window reports now distinguish coverage, active nonzero rows, and conservative usable rows.
   - Clean windows now use usable masks instead of non-null coverage masks.
   - `gdelt_only` no longer requires GKG; a separate `gdelt_gkg_documents` window is reported.
   - Timestamp safety filters are enforced in source coverage masks for orderbook and context rows.
   - Context source-count columns are no longer classified as availability metadata solely because they contain `source_`.
   - The macro/liquidity-stress FreqAI candidate is downgraded to conditional pilot pending a preflight artifact.
4. Regenerated fixed reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_20260529_review_fixed.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_20260529_review_fixed.md`

## 2026-05-29 Trader-Confluence Kickoff

1. Renamed the old indicator TODO to `user_data/Indicators/sieve_tasks_todo.md` to make clear it applies to Sieve/indicator strategy-stage work, not the external context/orderbook/FreqAI confluence research objective.
2. Updated `user_data/Indicators/agent.md` so Sieve strategy agents read `sieve_tasks_todo.md` only for Sieve-specific tasks.
3. Added `ai_guidance_docs/agents_structure.md` for orchestrator agents managing multi-agent trader-confluence research.
4. Updated `ai_guidance_docs/current_objectives.md` with research control requirements:
   - column dictionary first
   - direct tests before FreqAI
   - controls
   - ablations
   - promotion rules
   - source-family explainability
   - trader-language reporting
5. Launched parallel read-only agents:
   - Data readiness audit: `019e70e0-9ee9-7110-ade8-b4542ef4ecb9`
   - Column dictionary / feature semantics audit: `019e70e0-df07-7710-8f29-8078f0685136`
   - Trader hypothesis registry proposal: `019e70e1-1951-7142-afb2-47653a0fb897`
   - Implementation/test-harness scouting: `019e70e1-54a2-7b11-8f39-179083b6278d`
6. Sub-agent kickoff findings:
   - Bybit linear and inverse 1h trader-state parquet outputs now exist and are ready for validation/use.
   - Full structure + orderbook + context confluence is currently limited by structural cache coverage: `2025-04-29` to `2026-05-12`.
   - Context/GKG remains a moving layer; use frozen parquet snapshots for tests and record GKG coverage gaps.
   - A unified 1h snapshot, column dictionary, named hypothesis registry, direct-test controls, and ablations are required before more FreqAI queues.
7. Implemented the first trader-confluence foundation:
   - `user_data/Custom_Launcher/research/context_features/trader_confluence_hypotheses.py`
   - `user_data/Custom_Launcher/research/context_features/trader_confluence_features.py`
   - `user_data/Custom_Launcher/research/context_features/trader_confluence_snapshot.py`
   - `user_data/Custom_Launcher/research/context_features/trader_confluence_column_dictionary.py`
   - `user_data/Custom_Launcher/research/context_features/trader_confluence_direct_tests.py`
8. Generated frozen confluence artifacts:
   - Snapshot: `user_data/research_news_data/context_features/confluence_cache/trader_confluence_1h_latest.parquet`
   - Column dictionary: `user_data/research_news_data/context_features/confluence_cache/trader_confluence_column_dictionary.csv`
   - Validation: `user_data/research_news_data/context_features/confluence_cache/trader_confluence_1h.validation.json`
9. Snapshot validation summary:
   - Rows: `55,971`
   - Columns: `1,408`
   - Numeric columns: `1,386`
   - Duplicate dates: `0`
   - Base hourly gaps: `15`
   - Structure-present rows: `9,073`
   - Context-present rows: `54,579`
   - Spot orderbook-present rows: `9,320`
   - Bybit linear-present rows: `29,280`
   - Bybit inverse-present rows: `29,280`
   - Structure + context + orderbook overlap rows: `8,653`
10. First direct-test outputs:
   - Reviewed direct summary: `user_data/research_news_data/context_features/reports/trader_confluence_direct_tests_trader_confluence_kickoff_reviewed_summary.csv`
   - Behaviour-only model ablations: `user_data/research_news_data/context_features/reports/trader_confluence_direct_tests_trader_confluence_kickoff_fixed_ablation_model_ablations.csv`
11. First direct-test watchlist:
   - `orderbook_support_removal_breakdown -> breakdown_success_next_6h`: AUC about `0.681`, trigger event rate `0.610` versus same-regime `0.111`.
   - `orderbook_resistance_evaporation_breakout -> breakout_success_next_6h`: AUC about `0.627`, trigger event rate `0.459` versus same-regime `0.128`.
   - `context_pressure_plus_structure_break` for both breakout and breakdown success: AUC about `0.571` to `0.576`, with useful trigger lift.
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_24h`: AUC about `0.614`; 6h drawdown score is less stable monthly.
   - LVN fast-travel path metrics show potentially useful directional lift but have small row counts and need focused follow-up.
12. First model-ablation caution:
   - Behaviour-only model ablations often show `price` or `structure` alone outperforming the full feature set.
   - Treat that as an overfitting/feature-selection warning, not proof that confluence is useless.
   - Next FreqAI promotion should use only the strongest named hypotheses and compact feature sets.
13. Review checkpoint started:
   - Code/data-safety review agent: `019e713c-2d6d-74f1-bd0d-eb33b270a34b`
   - Logic/objective review agent: `019e713c-6761-7a40-aaa6-e44ce2214351`

## 2026-05-29 Trader-Confluence Review Fixes

1. Review agents found material issues in the first confluence pass:
   - confluence gates were too loose and could pass on one strong component
   - watchlist verdict used absolute control-rate difference instead of directional lift
   - model features could include source coverage/freshness/debug metadata
   - `minus_volume_pressure` removed unrelated orderbook/context pressure fields
   - the report did not list component columns or a price/structure baseline
   - past price features crossed base hourly gaps
2. Fixed the implementation:
   - stricter component-count gates for named hypotheses
   - gap-safe historical price features
   - directional same-regime/random control checks for watchlist status
   - explicit price/structure baseline score in direct-test summaries
   - diagnostic metadata excluded from model features
   - volume ablation now removes volume-dependent composites without deleting unrelated pressure columns
   - report now lists setup, trigger, and score component columns plus control/baseline metrics
   - column dictionary now classifies `px_return_*` as price action instead of a target label
3. Paused live web/global collectors via their stop-file mechanism for this research pass. Historical GKG backfill remains running because it is not a live collector.
4. Regenerated strict artifacts:
   - Snapshot: `user_data/research_news_data/context_features/confluence_cache/trader_confluence_1h_latest.parquet`
   - Validation: `user_data/research_news_data/context_features/confluence_cache/trader_confluence_1h.validation.json`
   - Direct report: `user_data/research_news_data/context_features/reports/trader_confluence_direct_tests_trader_confluence_review_fixed_strict_report.md`
   - Direct summary: `user_data/research_news_data/context_features/reports/trader_confluence_direct_tests_trader_confluence_review_fixed_strict_summary.csv`
   - Focused ablation: `user_data/research_news_data/context_features/reports/trader_confluence_direct_tests_trader_confluence_review_fixed_strict_ablation_model_ablations.csv`
5. Strict snapshot validation:
   - Rows: `55,971`
   - Columns: `1,618`
   - Duplicate dates: `0`
   - Source future violations: `0`
   - Inactive context signal rows: `0`
   - Inactive orderbook signal rows: `0`
   - Remaining warnings: base hourly gaps plus three low-coverage Bybit linear/inverse rows excluded from present flags
6. Strict direct-test result:
   - Only `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h` survived all current watchlist guards.
   - Rows: `4,750`; trigger rows: `127`.
   - Trigger event rate: `0.079`; same-regime event rate: `0.042`; random event rate: `0.024`.
   - Score AUC: `0.650`; price/structure baseline AUC: `0.606`; shuffled AUC: `0.512`.
   - Monthly positive AUC windows: `7/8`.
7. Strict focused ablation for the surviving candidate:
   - `confluence_only`: mean AUC `0.606`, positive windows `6/8`.
   - `price`: mean AUC `0.599`, positive windows `5/8`.
   - `only_ob_inverse`: mean AUC `0.581`, positive windows `3/8`.
   - `orderbook` / `all_venues_no_composites` / `minus_context`: mean AUC about `0.570`, positive windows `5/8`.
   - `context` / `minus_orderbook`: mean AUC about `0.497`, despite `5/8` positive windows, so context alone is unstable.
8. Interpretation:
   - The earlier orderbook breakout/breakdown signals are not invalid, but they no longer pass the stricter promotion rule because trigger samples became too small or price/structure baseline was stronger.
   - The next useful work is not a broad FreqAI queue. It is threshold calibration and hypothesis refinement so strict confluence keeps enough rows without allowing one-component triggers.
9. Completion/removal status:
   - No objective is 100% complete.
   - FreqAI promotion remains blocked until refined hypotheses survive direct controls, baseline, and ablations with adequate sample size.
10. Added and ran threshold/component sweep:
    - Script: `user_data/Custom_Launcher/research/context_features/trader_confluence_threshold_sweep.py`
    - CSV: `user_data/research_news_data/context_features/reports/trader_confluence_threshold_sweep_trader_confluence_review_fixed_strict.csv`
    - Markdown: `user_data/research_news_data/context_features/reports/trader_confluence_threshold_sweep_trader_confluence_review_fixed_strict.md`
    - Sweep rows: `9,857`
    - Passing candidates: `115`
11. Sweep interpretation:
    - All passing candidates are `macro_context_liquidity_stress_breakdown`, split across `large_drawdown_next_6h` and `large_drawdown_next_24h`.
    - No orderbook breakout, orderbook breakdown, support bounce, resistance rejection, context-plus-structure break, LVN travel, or multi-venue orderbook hypothesis currently passes full control/baseline rules.
    - Best 24h drawdown setting requires two setup components and two trigger components at component threshold `0.45`, with `71` to `79` trigger rows depending on trigger score threshold.
    - Best 6h drawdown settings are weaker but still pass, with component threshold `0.45`, two setup components, one trigger component, and about `156` to `509` trigger rows depending on trigger threshold.
12. Current next task:
    - Promote only the macro/liquidity-stress crash-risk family into a small FreqAI queue if the user wants immediate modelling.
    - For the broader objective, refine and retest failed hypothesis families with better component definitions rather than relaxing gates globally.

## Other Avenues While Waiting On Bybit Data

Query1 - "Other avenues while waiting on Bybit data?"
Answer - Yes. Best options are lightweight analyses that use already validated data and do not compete heavily with the Bybit feature builders. See section 1.

1. **Best Avenues Now**

1. **Meta-analysis of completed FreqAI runs**
   - We already have a large ledger of completed tests.
   - We can rank which signals are stable across months, not just which had one good score.
   - This may tell us whether VAH rejection, breakout acceptance, crash detection, or context blends deserve the next queue.

2. **Event-label quality review**
   - Before throwing more models at the data, check whether the labels actually match trader logic.
   - Example: does `breakout_failure_next_6h` really mean "failed breakout," or is it just a weak future-return proxy?
   - This is probably high value.

3. **Known-event window mapping**
   - Build a manual/event calendar for known BTC-moving periods:
     - ETF approval/run-up
     - halving narrative
     - COVID crash/recovery
     - Ukraine/Russia escalation
     - Iran conflict windows
     - major Fed/rate/liquidity shocks
     - China/property/market stress
     - major exchange/legal/security events
   - Then test whether existing context/news features light up around them.
   - This helps answer "are we measuring the right thing?" before FreqAI.

4. **News/context feature audit**
   - Review whether current news/GDELT/context features actually encode:
     - severity
     - persistence
     - first mention
     - source confluence
     - topic escalation
     - macro/geopolitical/liquidity channel
   - If they are mostly counts/z-scores, that explains weak results.

5. **Custom indicator prebuild audit**
   - Check your Volume Profile/support-resistance/structure indicators and decide which should become prebuilt parquet columns.
   - This can happen while Bybit builds run, because it is design/validation heavy rather than huge raw-data processing.

6. **Leakage/timestamp validation**
   - Very useful and not glamorous.
   - Confirm no feature uses future candles, stale forward-filled orderbook rows, or post-event news assigned to earlier candles.
   - If this is wrong, all results are suspect.

7. **Prepare next queue definitions**
   - Build the next experiment manifest design now, but do not launch until Bybit linear/inverse parquet validates.
   - The queue can be ready to compare:
     - price baseline
     - structure/VP
     - spot orderbook
     - linear orderbook
     - inverse orderbook
     - context/news
     - confluence combinations

2. **What I Would Prioritise**

1. Ledger meta-analysis.
2. Label/event-quality review.
3. Known-event mapping.
4. Context/news severity audit.
5. Prepare the next queue for when Bybit finishes.

3. **What I Would Avoid While Waiting**

1. More heavy FreqAI runs right now, because the Bybit feature builders are already using a lot of CPU/RAM.
2. New live database collector work.
3. More broad "try everything" tests without first tightening the objective and labels.

4. **Most Useful Next Step**

Yes: I'd start with a structured review of the completed result ledger and event labels. That gives us a better target for the next Bybit run instead of just feeding more data into weak definitions.

## Progress Log

### 2026-05-27

1. Started objective workstreams:
   - Meta-analysis of completed FreqAI runs.
   - Event-label quality review.
   - Known-event window mapping.
   - News/context feature audit.
   - Custom indicator prebuild audit.
   - Leakage/timestamp validation.
   - Next queue definition preparation.
2. Parallel sub-agent review started for:
   - FreqAI ledger meta-analysis.
   - Event-label and leakage/timestamp review.
   - News/context feature audit.
   - Custom-indicator prebuild and next-queue design.
3. Local sanity check:
   - `freqai_research_results_ledger.csv` exists with `2274` rows.
   - Latest FreqAI queue is exhausted: `72` completed, `0` pending.
   - No active FreqAI queue runner was found.
   - Bybit linear and inverse feature builders are still active.
   - No new linear/inverse Bybit parquet output is ready for validation yet.
4. Early ledger observation:
   - Strongest current candidates remain event-scoped signals, especially breakout failure / VAH rejection, breakout acceptance, and crash/breakdown detection.
   - Broad all-candle prediction remains lower priority than event-gated tests.
5. Completed sub-agent findings:
   - Ledger meta-analysis:
     - `price_tree_breakout_failure_6h` on monthly VAH rows is the strongest repeatable target so far: `9/9` monthly windows above AUC `0.55`, mean AUC about `0.6860`, best `spot_nov_2025` AUC `0.8750`.
     - `price_ridge_breakout_success_6h` is the cleanest breakout-acceptance setup: `9/9` monthly windows above AUC `0.55`, mean AUC about `0.6596`, best `spot_jan_2026` AUC `0.8394`.
     - Crash/breakdown detection is promising but less stable.
     - Fakeout labels, context-only tests, and generic future-return/drawdown regression are weak or noisy.
     - Ledger caveat: `_only` and `_orderbook_present` scopes often have identical metrics, so orderbook-present comparisons need validation before trusting them.
   - Event-label and leakage review:
     - Current event labels are better than plain future returns, but many are still future path/outcome proxies rather than strict "current trader setup then outcome" labels.
     - Context feature DB has `0` observed `max_source_available_at > date` violations across `56,041` rows.
     - Risk remains because strategy context merge uses backward `merge_asof` with no explicit max-age tolerance.
     - Orderbook exact timestamp merge avoids stale carry, but missing numeric orderbook features become `0.0`; this is only safe if coverage/orderbook-present gating is enforced.
     - `obts_feature_present` can be `1.0` even when `obts_coverage_ratio` is very low, so orderbook-present should require a coverage threshold.
   - News/context audit:
     - Current context builder partially encodes severity, persistence, source confluence, topic escalation, and macro/geopolitical/liquidity channels.
     - Explicit first-mention/follow-through features are missing.
     - GDELT/GKG layers are still mostly counts/tone/source sums/z-scores.
     - GKG remains stale in the built feature store; current GKG contribution is effectively sparse.
     - Recommended compact additions include `topic_first_mention_flag_24h`, `topic_escalation_score`, `severity_confluence_score`, `macro_liquidity_shock`, `risk_off_cross_market_confirmation`, and GDELT/GKG coverage-quality flags.
   - Custom-indicator and next-queue audit:
     - Existing structural cache already prebuilds VP, TLV2, BOS/CHoCH, volume-pressure, and range-position features into `st_*` columns.
     - `btc_structural_features_1h_latest.parquet` exists with about `9073` rows and `252` columns, from `2025-04-29` to `2026-05-12`; it likely needs refresh/coverage extension before the next full queue.
     - Next queues should wait until Bybit linear/inverse parquet outputs exist and validate.
     - Proposed next queue families: price, structure/VP, spot orderbook, linear orderbook, inverse orderbook, price+context, spot-linear confluence, linear-inverse confluence, structure+best-orderbook, price-context-best-orderbook, then full confluence.
6. New blockers before more broad queues:
   - Validate or fix orderbook-present scope scoring, because `_only` and `_orderbook_present` metrics are often identical.
   - Add/enforce orderbook coverage threshold for `orderbook_present`; low-coverage rows should not count as fully present.
   - Add/preflight max-age checks for context backward-asof merges.
   - Refresh/validate structural feature cache before full structure/VP queues.
   - Repair/rebuild GKG/context before judging context-only performance as final.
   - Audit event labels against manual chart/event windows before expanding fakeout or crash queues.
7. Next recommended implementation work:
   - Build a pre-queue audit report covering duplicate dates, gaps, source timestamp violations, feature latest vs timerange end, orderbook coverage distribution, and target class balance by month/mask.
   - Add explicit current-setup masks to event labels so tests mean "given this trader-observed setup, what happens next?" rather than "will a future setup/outcome happen from here?"
   - Prepare, but do not launch, the next Bybit linear/inverse queue until the builders finish and parquet validation passes.
8. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-06-02 GKG Hourly Metadata Prototype

1. Operational change:
   - Stopped the active GKG raw download loop because `D:` was full and the target scope changed.
   - Verified no active `gkg_raw_download`, `run_gkg_raw_download_loop`, or `gkg_hourly_metadata_extract` processes remained after the run.
2. Additive prototype:
   - Added `user_data\Custom_Launcher\research\context_features\gkg_hourly_metadata_extract.py`.
   - Streams raw GKG ZIPs directly into compact 1h FreqAI/Freqtrade-ready metadata features.
   - Avoids materializing full per-document Silver rows.
   - Joins existing `gdelt_hourly_features` event aggregates when available.
   - Outputs parquet, CSV, summary JSON, and audit sample CSV.
3. January 2020 v3 smoke:
   - Window: `2020-01-01T00:00:00+00:00` to `2020-02-01T00:00:00+00:00`.
   - Files read: `2,975` of `2,976` expected 15-minute GKG ZIPs.
   - Raw coverage ratio: `0.999664`.
   - Document rows streamed: `5,085,793`.
   - Rows skipped: `8`.
   - Corrupt files: `0`.
   - Feature rows: `744`.
   - Feature columns: `188`.
   - Output parquet size: `0.854 MB`.
   - January 2020 raw GKG size: `18.57 GB`.
4. Main outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_gkg_hourly_metadata_jan2020_review_v3.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_gkg_hourly_metadata_jan2020_review_v3.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_audit_gkg_hourly_metadata_jan2020_review_v3.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_summary_gkg_hourly_metadata_jan2020_review_v3.json`
5. Assessment:
   - Storage compression is excellent for the intended purpose: `18.57 GB` raw became a sub-`1 MB` compact hourly feature parquet for one month.
   - The first rule pass was too broad; audit found false positives such as `ETHNICITY` matching `eth` and generic `financial emergency` matching banking stress.
   - Rules were tightened to v3 before the final January artifact.
   - Remaining caveat: regulation, geopolitics, oil, macro, and cyber still fire in every hour because GKG is very high-volume. These features need baseline/percentile normalization and direct tests before promotion.
   - Do not delete raw years yet. Deletion should wait until the full target window is extracted with coverage reports, audit samples, and a deletion manifest.
6. Current raw deletion payoff:
   - Present 2020 GKG raw: `32,551` files / `201.13 GB`.
   - Present 2021 GKG raw: `13,968` files / `75.22 GB` through `2021-06-03T08:00`.
   - Present 2022 GKG raw: `0` files.

### 2026-05-31 GDELT/GKG Raw Download Restart

1. Scope:
   - Limited to historical GDELT/GKG raw archive completion first.
   - No full Silver/Gold formatting run was started in this step.
2. Sub-agent split:
   - Worker A updated the raw GKG downloader for safer per-file persistence and loop defaults.
   - Worker B added the raw GKG coverage report tool.
   - Reviewer C performed a functional/code review and identified download-critical and downstream formatting risks.
3. Implemented download-critical fixes:
   - GKG raw downloader now flushes completed status rows incrementally, defaulting to every completed future.
   - Worker exceptions are recorded as retryable download status rows instead of losing completed chunk progress.
   - Existing corrupt ZIP handling is supported through `--validate-existing`; corrupt files are moved aside as `.corrupt` before redownload.
   - Downloader status now also updates the unified `gdelt_raw_files` table using the local raw archive path as the canonical key.
   - The PowerShell loop defaults to `30` day chunks and `4` workers.
   - The loop supports optional `-ValidateExisting`.
   - The loop uses `-RetryDeferHours 72` by default so recent retryable failures are recorded and temporarily skipped during the forward sweep instead of blocking all later files.
4. Coverage/reporting added:
   - New CLI: `user_data.Custom_Launcher.research.context_features.gkg_raw_coverage_report`.
   - Latest report output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gkg_raw_coverage_gkg_raw_coverage_restart_check.json`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gkg_raw_coverage_gkg_raw_coverage_restart_check.csv`
   - Coverage before restart report:
     - `224,640` expected 15-minute GKG files from `2020-01-01` to `2026-05-29`.
     - `17,352` present ZIPs.
     - `7.7244%` complete.
     - `207,288` missing raw files.
     - `1` terminal 404.
     - `0` corrupt files in an 8-file validation sample.
     - Estimated remaining bytes: about `1.40 TB`.
5. Validation:
   - Focused GDELT/GKG test suite passed: `43 passed`.
   - PowerShell loop script parsed successfully.
   - Real one-file smoke download succeeded for `2020-04-20T18:30:00+00:00`.
6. Active run:
   - Started background loop at `2026-05-31 21:49:40 Europe/London`.
   - Command uses `ChunkDays=30`, `MaxWorkers=4`, `RetryDeferHours=72`.
   - Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_raw_download_loop_20260531_214940.log`.
   - First active chunk: `2020-04-20T19:45:00+00:00` to `2020-05-20T19:45:00+00:00`, `2,877` planned files.
   - Early direct counter after launch: `17,409` raw ZIPs and `56` successful rows in the active chunk.
7. Remaining open work:
   - Monitor the active loop for progress, retryable failures, disk space, and log updates.
   - After the forward sweep, run a retry/fill pass without treating deferred failures as complete.
   - Address downstream reviewer findings before trusting Silver/Gold formatted features: streaming Silver extraction, shared raw-file identity handling across all parsers, multi-topic extraction, topic taxonomy gaps, availability-window quality gates, and read-only dry-run behavior.

### 2026-05-31 Delta Discovery Hypothesis Update

1. Implemented the next feature-discovery loop step:
   - Converted stable discovery-screen findings into a compact delta hypothesis pack.
   - Added price/range states, compression/expansion states, bearish short-volume impulse, and orderbook persistence-weakness states.
   - Fixed the prior compression score to use a rolling historical percentile instead of full-column ranking, removing a lookahead risk from that metric.
2. New hypothesis families added:
   - Range-high breakout success.
   - Range-low breakdown success.
   - Expansion drawdown risk.
   - Compression breakout release.
   - Compression breakdown release.
   - Orderbook support weakness drawdown.
   - Orderbook resistance weakness breakout.
   - Structure+orderbook breakout confluence.
   - Structure+orderbook drawdown/breakdown confluence.
3. Validation:
   - Python compile passed for touched feature and hypothesis modules.
   - Trader confluence snapshot rebuilt successfully:
     - Rows: `55,971`
     - Columns: `2,510`
     - Latest parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Snapshot still emits pandas fragmentation warnings due the existing column-by-column hypothesis builder. This is performance debt, not a result validity failure.
4. Direct/sweep tests launched and completed:
   - Direct summary rows: `183`
   - Direct delta rows: `27`
   - Threshold sweep rows: `34,485`
   - Delta sweep rows: `5,180`
   - Delta passing threshold variants: `2,338`
5. Current blocker before FreqAI promotion:
   - Resolved for first-pass testing by adding a lightweight delta-confluence FreqAI strategy/profile bridge.
   - Remaining work is now refinement, not setup: decide which delta features should be exposed to future queue packs and whether to score stricter delta trigger scopes.

### 2026-05-31 Delta FreqAI Queue Update

1. Added first-pass FreqAI bridge:
   - Strategy class: `ContextTraderConfluenceDeltaFreqAIResearchStrategy`
   - Feature family: `trader_confluence_delta`
   - Source file: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Exposes discovery-derived `conf_delta_*`, range-position, compression/expansion, and orderbook persistence/pressure states as FreqAI `%` features.
2. Queue created and completed:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260531_210705_354735\freqai_experiment_queue.json`
   - Experiments: `12`
   - Windows: `spot_q4_2025`, `spot_q1_2026`, `spot_recent`
   - Profiles: price tree breakout/breakdown controls plus delta-confluence tree breakout/breakdown candidates.
3. Result headline:
   - Breakout: delta-confluence beat price-only in Q4 2025 on all rows and breakout-acceptance rows. It was mixed/weak in Q1 2026 and recent broad rows, but recent event-row top bucket improved.
   - Breakdown: delta-confluence did not beat price-only; price-only remained stronger across all tested breakdown/crash-detection windows.
4. Next refinement:
   - Keep delta breakout candidates alive.
   - Do not promote delta breakdown FreqAI yet.
   - Add stricter delta-trigger scoped FreqAI scoring if the next pass should match the direct-test evidence more closely.

### 2026-05-29 Source-Detail Gated Trader-Confluence Tests Started

1. Fixed the direct-test harness before running:
   - Source-detail gating now requires each hypothesis' required source blocks to have usable rows.
   - Walk-forward model ablations now train only on months before the holdout month.
   - Empty or all-null feature sets are skipped instead of crashing the ablation path.
2. Ran corrected direct tests:
   - Output report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_source_detail_gated_20260529_report.md`.
   - Detailed rows: `237`.
   - Summary rows: `43`.
   - Source-detail gate: enabled.
3. Result:
   - Only one watchlist-positive direct-test candidate survived: `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`.
   - Direct AUC: `0.6497`.
   - Baseline AUC: `0.6062`.
   - Shuffled AUC: `0.5348`.
   - Trigger rows: `127`.
   - Trigger positives: `10`.
4. Ran FreqAI preflight for the surviving candidate:
   - Output: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_freqai_preflight_source_detail_gated_20260529_macro_dd6h.md`.
   - Verdict: `conditional_pilot`.
   - Failed checks: `none`.
   - Reason it is not full promotion: trigger positives are still too low for a normal promotion claim.
5. Ran source-detail gated threshold/component sweep:
   - Output: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_source_detail_gated_20260529.md`.
   - Sweep rows: `8077`.
   - Passing candidates: `120`.
   - Passing candidates were only for `macro_context_liquidity_stress_breakdown`, split across `large_drawdown_next_6h` and `large_drawdown_next_24h`.
6. Current state:
   - Testing has started and produced corrected results.
   - FreqAI queueing for this advanced trader-confluence candidate should still use the conditional-pilot gate and not be treated as final validation.

### 2026-05-27 Review Fix Update

1. Review agent found a real context-age flaw:
   - Context strategies were using the hourly feature-row timestamp as freshness, not the underlying `max_source_available_at`.
   - Fixed: context presence and `%-ctx_feature_age_hours` now use source availability. Rows with null, future, or older-than-`48h` source timestamps are marked missing.
2. Review agent found a preflight gap:
   - Preflight checked only source timestamp after feature date.
   - Fixed: parquet preflight now records null source timestamp rows and window-scoped source age over `48h`.
3. Review agent found ambiguous acceleration-window logic:
   - `_window_between()` accepted `start, end` but internally treated them backwards.
   - Fixed: helper now accepts `earlier, later`, validates ordering, and the previous-six-hour acceleration window call now uses explicit chronological order.
4. Validation:
   - Python compile passed for the strategy, context builder, and queue preflight script.
   - `_window_between()` sanity test returned the expected rows for `(hour-12h, hour-6h]`.
5. Queue hygiene:
   - Paused the first queue after the review finding.
   - Created a fresh queue after the fixes: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_232436_720772\freqai_experiment_queue.json`.
   - Fresh queue preflight: `84` experiments, `0` errors, `14` warnings.
   - Fresh queue warnings include overall null `max_source_available_at` rows in the context parquet, but selected windows have no preflight errors and the strategy now marks stale/missing source rows as missing.
   - Fresh queue runner launched and initial status was `1` completed, `83` pending.
6. Known-event calendar:
   - Added additional sanity-check windows for Evergrande/property stress, Ukraine invasion, Fed hiking cycle, Celsius, Fed 75bp hike, SVB, Country Garden/property stress, 2025 Iran/Israel escalation, and 2025 AI valuation-risk warning.
   - These are validation windows only, not trading rules.
7. Still blocked/open:
   - Bybit linear/inverse feature parquet outputs are still running and not yet validated.
   - Active GKG repair means context source rebuild should wait; existing context parquet is a stable snapshot for current tests.
   - Structural cache still cannot extend past stale local `4h`/`1d` OHLCV without refreshing OHLCV.

### 2026-05-28 Queue Result Update

1. Completed clean tree setup queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_232436_720772\freqai_experiment_queue.json`.
   - Status: `84/84` completed, `0` failed.
2. Strongest early results:
   - Price-only VAH rejection / breakout-failure in `spot_oct_2025`: AUC about `0.768` on `62` event rows.
   - Price+context VAH rejection / breakout-failure in `spot_dec_2025`: AUC about `0.765` on `87` event rows.
   - Price+context crash-detection in `spot_nov_2025`: AUC about `0.728` on `74` event rows.
   - Price-only crash-detection in `spot_nov_2025`: AUC about `0.712` on `74` event rows.
3. Best positive lifts versus price controls:
   - Context breakout-success in `spot_jan_2026`: AUC lift about `+0.204` on breakout-acceptance rows.
   - Context breakout-failure in `spot_feb_2026`: AUC lift about `+0.214` on VAH rejection rows.
4. Main caution:
   - Context is not consistently helpful; it hurt several October/December breakout-failure slices.
   - Spot orderbook did not improve the strongest November crash-detection event rows.
   - April crash rows are low-count, so high AUC there is exploratory only.
5. Next queue launched:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_004256_909465\freqai_experiment_queue.json`.
   - Purpose: repeat the same current-setup hypotheses with Ridge as a simplicity/robustness check.
   - Preflight: `84` experiments, `0` errors, `14` warnings.

### 2026-05-28 Ridge Queue Result Update

1. Completed Ridge robustness queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_004256_909465\freqai_experiment_queue.json`.
   - Status: `84/84` completed, `0` failed.
2. Strongest results:
   - Orderbook Ridge VAH rejection / breakout-failure in `spot_oct_2025`: AUC about `0.866` on `62` event rows.
   - Context Ridge breakout-success in `spot_nov_2025`: AUC about `0.684` on `81` event rows.
   - Orderbook Ridge VAH rejection / breakout-failure in `spot_dec_2025`: AUC about `0.658` on `87` event rows.
3. Best positive lifts:
   - Context Ridge breakout-success in `spot_nov_2025`: AUC lift about `+0.236`, AP lift about `+0.157`.
   - Orderbook Ridge VAH rejection in `spot_oct_2025`: AUC lift about `+0.176`, but AP lift was negative.
   - Orderbook Ridge all-row breakout failure in `spot_jan_2026`: AUC lift about `+0.205`.
4. Cautions:
   - Context Ridge breakout-failure was strongly harmful in `spot_oct_2025`.
   - Ridge price+context crash detection was harmful in `spot_nov_2025`.
   - Ridge orderbook crash detection underperformed price in January and April event rows.
5. Interpretation:
   - Some context/orderbook effects are not purely tree-threshold artefacts, but they are still unstable month-to-month.
   - Continue with event-scoped tests and controls rather than broad all-row prediction.
6. Next queue launched:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_014316_660551\freqai_experiment_queue.json`.
   - Purpose: compare custom structure/VP features and structure+spot-orderbook confluence on validated structural/orderbook parquet.
   - Experiments: `21`.
   - Preflight: `0` errors, `1` low-orderbook-coverage warning.

### 2026-05-28 Structure/VP Queue Update

1. Structure/VP confluence queue finished with partial failure:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_014316_660551\freqai_experiment_queue.json`.
   - Completed: `15`.
   - Failed: `6`.
   - Pending: `0`.
2. Failure details:
   - All failed runs were structure+spot-orderbook on `ob_oct` and `ob_jan_feb`.
   - Error: FreqAI/datasieve `AttributeError: 'Pipeline' object has no attribute 'features_in'` during prediction transform.
   - Treat these as invalid runs, not negative model evidence.
3. Useful completed results:
   - Structure/VP Ridge on October VAH rejection rows: AUC about `0.677`, AP about `0.418`.
   - Structure/VP + spot-orderbook Ridge on `ob_jul_aug` all predicted rows: AUC about `0.717`, AP about `0.226`.
   - Structure/VP + spot-orderbook Tree on `ob_jul_aug` VAH rejection rows: AUC about `0.553`, AP about `0.180`.
4. Open issue:
   - Need root-cause fix or narrower setup before trusting structure+orderbook FreqAI on later orderbook windows.

### 2026-05-28 Bybit Raw Retention Cleanup Check

1. Guarded Bybit raw-retention cleanup was skipped again before `py_compile`, dry-runs, or any delete-capable command.
2. Active build evidence still present:
   - `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_linear_trader_state_feature_build_20260527_112834.out.log` was last written at `2026-05-28 11:14:38 +01:00`.
   - `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_inverse_trader_state_feature_build_20260527_165821.out.log` was last written at `2026-05-28 11:19:05 +01:00`.
   - Running Python processes were still present with start times `2026-05-28 09:55:03 +01:00`, `2026-05-28 10:19:16 +01:00`, and `2026-05-28 20:12:48 +01:00`, so the build chain cannot be treated as inactive yet.
3. Observed build state:
   - Linear log tail shows `files_processed: 1224/1224` and `feature_rows_1h: 29370`.
   - Inverse log tail shows `files_processed: 1224/1224` and `feature_rows_1h: 29370`.
   - Completion-looking output does not override the retention guard while the related logs are still receiving writes and Python workers remain live.
4. Outcome:
   - Spot skipped.
   - Linear skipped.
   - Inverse skipped.
   - Raw files deleted: `0`.
   - Reclaimed bytes: `0`.
5. Next run should re-check the same guard first and only proceed to `py_compile` plus per-market retention validation after the Bybit sync/feature-build chain is clearly inactive.

### 2026-05-27 Late Implementation Update

1. Removed a live-SQL dependency from queued context research:
   - Context FreqAI strategies now load `context_features_1h_latest.parquet` by default instead of reading `context_features.sqlite` during FreqAI runs.
   - Context feature caching is keyed by strategy class and source path so different context profiles cannot accidentally reuse the wrong loaded frame.
2. Updated context exports:
   - `export_features()` now writes metadata columns needed for timestamp validation: `generated_at`, `min_source_available_at`, `max_source_available_at`, `schema_version`, and `feature_config_hash`.
   - `export_features()` now also writes `context_features_1h_latest.parquet` for stable queued research paths.
3. Validated and exported the current context parquet snapshot:
   - Rows: `56,041`.
   - Columns: `409`.
   - Range: `2020-01-01T00:00:00+00:00` to `2026-05-24T00:00:00+00:00`.
   - Duplicate dates: `0`.
   - Source timestamp lookahead rows: `0`.
4. Created and launched a new hypothesis-led FreqAI queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_231751_233080\freqai_experiment_queue.json`.
   - Experiments: `84`.
   - Profiles: price baseline, spot orderbook, context-only, and price+context.
   - Targets: current-setup breakout failure, current-setup breakout success, and current-setup breakdown/crash continuation.
   - Windows: monthly slices from `2025-10` through `2026-04`.
   - Preflight errors: `0`.
   - Preflight warnings: acceptable low-label warnings for some breakdown slices and one low-coverage orderbook row.
5. Queue runner state:
   - Background runner started under the queue's `queue_runner_logs` directory.
   - Initial status after launch: `1` completed, `83` pending.
6. Review:
   - A review agent was started to check the parquet context loading, export metadata, registry path change, and queue preflight timestamp logic.
7. Still blocked/open:
   - Bybit linear and inverse feature parquet outputs are still running and not yet validated.
   - Structural cache still cannot extend past local `4h`/`1d` OHLCV coverage without refreshing those OHLCV files.
   - GKG repair/rebuild remains open before final context-only judgment.
8. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-05-29 GDELT/GKG Historical Rework Worker Wave

1. Scope:
   - Limited to GDELT event export and GKG historical data.
   - Did not change Bybit processing and did not run production Silver normalization against the live SQLite database.
2. Implemented additive first-milestone tooling:
   - Normalized schema helper: `user_data\Custom_Launcher\research\context_features\gdelt_gkg_normalized_schema.py`.
   - GDELT raw export normalizer: `user_data\Custom_Launcher\research\context_features\gdelt_event_normalize.py`.
   - GKG raw document normalizer: `user_data\Custom_Launcher\research\context_features\gkg_document_normalize.py`.
   - Quality report generator: `user_data\Custom_Launcher\research\context_features\gdelt_gkg_quality_reports.py`.
3. Integration corrections made after worker return:
   - Fixed GDELT shared-schema inserts to populate required `available_at` and trader-useful metadata columns instead of only the fallback schema.
   - Fixed GKG schema initialization to call shared `init_schema(conn)` directly before falling back.
   - Added shared-schema insertion tests for both GDELT events and GKG documents.
4. Verification:
   - Focused test command passed: `.\.venv\Scripts\python.exe -m pytest tests\research\test_gdelt_gkg_normalized_schema.py tests\research\test_gdelt_event_normalize.py tests\research\test_gkg_document_normalize.py tests\research\test_gdelt_gkg_quality_reports.py -q`.
   - Result: `17 passed`.
   - Dry-run GDELT parser against D-drive raw export data parsed 5 rows from 1 ZIP with 0 skipped.
   - Dry-run GKG parser against D-drive raw GKG data parsed 5 rows from 1 ZIP with 0 skipped.
   - Limited quality report generated under `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports` with tag `implementation_check`.
5. Active backfill status at check:
   - GKG loop still active under the existing PowerShell/Python process chain.
   - D-drive GKG raw cache had grown to `4175` ZIPs / `24.42` GB.
   - Highest observed raw GKG filename: `20200912150000.gkg.csv.zip`.
6. Remaining open work:
   - Let the current GKG backfill continue; do not run competing GKG download processes.
   - After raw coverage is materially complete, run production raw-to-Silver normalization in controlled chunks.
   - Then use quality gates to identify clean windows before rebuilding final trader-facing GDELT/GKG features.
7. Completion/removal status:
   - This completes the first additive implementation milestone only.
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-05-29 GDELT/GKG Historical Rework Second Wave

1. Scope:
   - Continued GDELT/GKG historical-only work.
   - No production raw-to-Silver normalization was run against the live SQLite database.
2. Added tooling:
   - Raw archive inventory: `user_data\Custom_Launcher\research\context_features\gdelt_gkg_raw_inventory.py`.
   - Read-only quality gates: `user_data\Custom_Launcher\research\context_features\gdelt_gkg_quality_gates.py`.
   - Date-window controls in `gdelt_event_normalize.py` and `gkg_document_normalize.py` with `--start` inclusive and `--end` exclusive filters applied before ZIP parsing.
3. Integration corrections:
   - GKG shared-schema insertion now populates `canonical_url` and `url_hash`.
   - Raw inventory now applies `--limit-files` before optional ZIP validation. The initial implementation applied the limit after validation, which made `--validate-zip --limit-files 4` start scanning/validating the whole archive set.
4. Verification:
   - Focused test command passed: `.\.venv\Scripts\python.exe -m pytest tests\research\test_gdelt_gkg_normalized_schema.py tests\research\test_gdelt_event_normalize.py tests\research\test_gkg_document_normalize.py tests\research\test_gdelt_gkg_quality_reports.py tests\research\test_gdelt_gkg_raw_inventory.py tests\research\test_gdelt_gkg_quality_gates.py -q`.
   - Result: `26 passed`.
   - GDELT event dry-run with a 2026-05-28 window parsed 3 rows from 1 raw ZIP with 0 skipped.
   - GKG document dry-run with a 2020-09-12 window parsed 3 rows from 1 raw ZIP with 0 skipped.
   - Raw inventory dry-run with `--validate-zip --limit-files 4` completed and validated 4 ZIPs successfully.
   - Quality gates for `2020-01-01T00:00:00+00:00` to `2020-01-01T02:00:00+00:00` wrote `implementation_gate_check` reports with `6` pass, `1` warn, and `4` fail rows; the fails are expected because production Silver tables are not built yet.
5. Active backfill status at check:
   - GKG loop remained active under PowerShell PID `34028` with Python workers visible.
   - D-drive GKG raw cache had grown to `4790` ZIPs / `28.16` GB.
   - Highest observed raw GKG filename: `20200919010000.gkg.csv.zip`.
6. Remaining open work:
   - Let the current GKG backfill keep running.
   - After a larger raw span is present, run raw inventory into `gdelt_raw_files`, then run date-windowed Silver normalization in controlled chunks.
   - Use quality gates to mark clean windows before rebuilding final trader-facing features.
7. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-05-29 GKG Raw-First Download Split

1. User decision:
   - Split GKG into raw-download-first and defer parsing/conditioning until raw coverage is complete.
2. Code added:
   - `user_data\Custom_Launcher\research\context_features\gkg_raw_download.py`
   - `user_data\Custom_Launcher\research\context_features\run_gkg_raw_download_loop.ps1`
   - `tests\research\test_gkg_raw_download.py`
3. Behaviour:
   - Downloads raw GKG ZIPs only.
   - Does not parse GKG CSV rows and does not build aggregate/Silver features.
   - Writes atomically via `.tmp` files and validates downloaded ZIP bytes before final rename.
   - Tracks raw download statuses in `gkg_raw_download_status` inside `gdelt_context.sqlite` so terminal 404s are not retried forever.
   - Existing raw ZIP files are skipped by file presence/size by default.
4. Process change:
   - Stopped the old combined `gkg_backfill` loop that downloaded, parsed, and wrote aggregate rows in one pass.
   - Started the raw-only loop:
     - `Start=2020-01-01`
     - `End=2026-05-29`
     - `ChunkDays=60`
     - `MaxWorkers=10`
     - `ProgressEvery=200`
   - Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_raw_download_loop_20260529_225154.log`.
5. Verification:
   - Smoke run for `2020-01-01T00:00:00+00:00` to `2020-01-01T01:00:00+00:00` downloaded 4 raw ZIPs and completed.
   - Focused tests passed: `24 passed`.
6. Current raw download state at launch check:
   - D-drive GKG raw cache: `6738` ZIPs / `40.07` GB.
   - Highest observed raw filename: `20201014000000.gkg.csv.zip`.
   - New raw-only loop was filling early missing files; newest write observed: `20200101061500.gkg.csv.zip`.
7. Remaining open work:
   - Let raw-only download complete before production parsing/conditioning.
   - Later run raw inventory, raw-to-Silver parsing, quality gates, and feature rebuild as separate controlled phases.

### 2026-05-29 GDELT/GKG Extraction And Formatting Scripts Prepared

1. User decision:
   - Keep raw download running now.
   - Prepare extraction/formatting scripts for later processing once raw coverage is complete.
2. Code added:
   - `user_data\Custom_Launcher\research\context_features\gdelt_gkg_silver_pipeline.py`
   - `user_data\Custom_Launcher\research\context_features\gdelt_gkg_gold_features.py`
   - `tests\research\test_gdelt_gkg_silver_pipeline.py`
   - `tests\research\test_gdelt_gkg_gold_features.py`
3. Pipeline behaviour:
   - Plans by default; production execution requires explicit `--execute`.
   - Supports chunked phases:
     - `inventory`
     - `gdelt_events`
     - `gkg_documents`
     - `gold_features`
     - `quality_gates`
   - Uses start-inclusive/end-exclusive date windows.
   - Can pass `--limit-files` and `--limit-rows` for smoke runs.
4. Gold feature behaviour:
   - Builds `gdelt_gkg_topic_features_1h` from Silver records.
   - Uses `available_at` alignment, not publication/download time.
   - Adds first-pass trader-readable fields:
     - `war_geopolitics_intensity_24h`
     - `banking_credit_confluence_24h`
     - `oil_energy_first_mention`
     - `regulation_persistence_72h`
     - `macro_release_severity_max_24h`
     - `topic_source_diversity_24h`
   - Keeps counts and Silver row coverage as supporting context.
   - Marks metadata as weak Silver labels pending classifier/enrichment.
5. Verification:
   - Plan-only pipeline command worked and did not mutate production data.
   - Focused tests passed: `35 passed`.
6. Active raw download status during this check:
   - Raw-only GKG downloader still active.
   - D-drive GKG raw cache: `7209` ZIPs / `42.29` GB.
   - Highest observed raw filename: `20201014000000.gkg.csv.zip`.
   - Newest observed write: `20200106040000.gkg.csv.zip`.
7. Remaining open work:
   - Wait for raw download to complete or reach a large enough checkpoint.
   - Run pipeline smoke with `--execute --limit-files --limit-rows` on a small window.
   - After successful smoke, run full chunked extraction/formatting.
   - Do classifier/story enrichment before treating Gold features as final trader-quality data.

### 2026-05-27 Implementation Update

1. Implemented pre-queue guardrails:
   - Queue creation now writes `preflight_audit.json`.
   - Required feature files are validated for duplicate dates, gaps, source timestamp violations, window coverage, and usable orderbook coverage.
   - Preflight now checks only the windows that each profile can actually run, avoiding false failures for mixed structure/orderbook queues.
   - Queue filesystem paths now use short `exp_0001` style folders and short FreqAI runtime identifiers to reduce Windows path-length risk.
   - Queue top-level status is updated by the runner instead of staying stale `pending` after completion.
2. Implemented safer feature gating:
   - Context features are merged with a `48h` max-age tolerance and feature age/present/missing flags.
   - Orderbook presence now requires `obts_feature_present > 0` and `obts_coverage_ratio >= 0.50` where coverage exists.
   - Direct structure/orderbook report loaders now apply the same orderbook coverage rule.
3. Implemented target balance reporting:
   - Preflight now reports event rows, label rows, positive rows, negative rows, and event rate per profile/window.
4. Implemented current-setup targets:
   - Added current resistance/support setup labels such as `breakout_failure_from_current_setup_6h`, `breakout_success_from_current_setup_6h`, `breakdown_success_from_current_setup_6h`, and `fakeout_from_current_setup_24h`.
   - Added setup-scoped profile targets alongside the legacy future-path labels.
   - Legacy VAH confluence target and setup-scoped VAH confluence target now use separate prediction columns to avoid scoring mismatches.
5. Review agents found and fixes were applied for:
   - Over-broad preflight window checks.
   - Windows path-length risk.
   - Missing usable orderbook coverage checks.
   - Shared strategy feature-cache keys.
   - Unnecessary import of the orderbook helper in price/context-only cases.
   - Target mismatch between setup-scoped VAH labels and legacy profiles.
   - Heavy scoring imports in queue-runner dry-run/no-pending paths.
6. Validation performed:
   - Python compile checks passed for the touched strategy, queue, scoring, registry, and report scripts.
   - Price-only queue smoke creation passed.
   - Structure cache stale-window smoke test correctly failed because current structural cache ends `2026-05-12` while `spot_recent` ends `2026-05-20`.
   - Mixed structure/orderbook queue smoke creation passed and audited only the relevant windows per required file.
   - Setup-scoped VAH target balance smoke test passed with both positive and negative rows.
7. Remaining open blockers:
   - Bybit linear and inverse feature parquet outputs are still not complete/validated.
   - Structural cache still needs refresh through intended queue end date.
   - GKG/context rebuild remains open before final context-only judgment.
   - Known-event calendar still needs broader Fed/liquidity, China/property, Ukraine/Russia, and other manually verified windows.
8. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-06-02 Communication And Test-Design Update

1. Added `result_communication_format.md`.
   - Future result explanations should start with the trader question, what was observed, what was asked afterwards, plain-English result, supporting numbers, verdict, and next step.
   - This is intended to prevent opaque ML/profile terminology from hiding the actual trading idea.
2. Updated `AGENTS.md` and `ai_guidance_docs/README.md` so agents know when to read the new communication format.
3. Added `Example 5: Downside Momentum After First Break` to `objective_examples.md`.
   - This captures the user's hypothesis that crashes/downside continuations may be easier to assess after the first break starts, using the next 1h-4h of orderbook, volume, support reclaim/failure, and momentum behaviour.
4. Added `next_trader_confluence_test_series.md`.
   - Contains the next plain-English test backlog for breakout continuation, breakout failure, downside momentum, downside slowing/failure, quiet technical breakouts, and orderbook panic state.
5. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-06-05 Readiness-Gated Feature Research Plan

1. Added `readiness_gated_feature_research_goal.md`.
   - Defines how a goal-mode agent should pursue feature-definition research when some data sources are incomplete.
   - Requires a source-readiness audit before tests.
   - Limits current work to validated price/volume, custom indicator, and orderbook windows.
   - Parks news/context until its separate rework is explicitly marked ready.
   - Adds stop rules so agents do not chase unavailable full confluence or huge unfocused feature sets.
2. Updated `AGENTS.md`, `current_objectives.md`, and `ai_guidance_docs/README.md` to reference the readiness-gated goal plan.
3. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-06-05 Goal Feature-Definition Research Pass

1. Read and followed the current objective guidance:
   - `current_objectives.md`
   - `objective_examples.md`
   - `result_communication_format.md`
   - `next_trader_confluence_test_series.md`
   - `context_research_detailed_findings.md`
2. Refreshed source readiness and clean-window reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_goal_feature_research_after_epsilon_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_goal_feature_research_after_epsilon_20260605.csv`
3. Current usable research scope:
   - Price/OHLCV: usable across the broad snapshot.
   - Structure/custom indicators: usable from `2025-04-29` to `2026-05-12`.
   - Structure + spot/linear/inverse orderbook: clean windows exist from `2025-04-29` to `2025-08-21` and `2025-08-21` to `2026-05-12`.
   - News/live context, Google Trends, ETF flow, and GKG documents remain too sparse for this confluence pass.
4. Added early-downside trader-state features and labels:
   - `downside_continuation_next_3h`
   - `downside_continuation_next_6h`
   - `downside_exhaustion_next_6h`
   - `support_reclaim_next_6h`
   - `conf_epsilon_downside_first_break_continuation_*`
   - `conf_epsilon_downside_first_break_exhaustion_*`
   - `conf_epsilon_support_reclaim_after_break_*`
   - `conf_epsilon_orderbook_panic_after_break_*`
5. Rebuilt frozen confluence snapshot:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Rows: `55,971`.
   - Columns: `2,576`.
6. Refreshed column dictionary:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.md`
7. Ran direct tests with source-detail gates and controls:
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_goal_feature_research_epsilon_20260605_summary.csv`
   - Markdown: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_goal_feature_research_epsilon_20260605_report.md`
   - Detail rows: `1,314`.
   - Summary rows: `197`.
8. Created an epsilon FreqAI validation queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_010227_427619\freqai_experiment_queue.json`
   - Profiles: price controls, epsilon-with-flags, and epsilon-components-only.
   - Targets: `downside_continuation_next_3h` and `support_reclaim_next_6h`.
   - Windows: `spot_q4_2025`, `spot_recent`.
9. Ran a tiny controlled feature-discovery pass:
   - Candidate parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_goal_feature_research_tiny_20260605.parquet`
   - Combined results: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_goal_feature_research_tiny_20260605_combined.csv`
   - Fast-path shortlisted rows: `172`.
10. Parked issue:
   - Full feature-discovery screening over `1,088` to `3,080` candidate columns is too slow for the current synchronous pass. Added `--skip-tree` so agents can produce fast univariate/control evidence before a later smaller tree-validation pass.
11. Completion/removal status:
   - The overall objective is not 100% complete because the new FreqAI validation queue has been created but not yet run/scored.
   - Do not remove any objective yet.

### 2026-06-05 Goal Feature-Definition Queue Launch

1. Verified the latest epsilon FreqAI validation queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_010227_427619\freqai_experiment_queue.json`
   - Experiments: `12`.
   - Initial status before launch: `12` pending.
2. Stopped obsolete duplicate broad feature-discovery screens:
   - These were still running against the full `goal_feature_research_20260605` candidate set.
   - That broad screen had already been parked as too slow and lower quality than the small shortlist route.
   - No collectors or unrelated live data jobs were stopped.
3. Verification before launch:
   - Python compile passed for the touched trader-confluence, feature-discovery, queue/scoring, registry, and FreqAI strategy files.
   - `git diff --check` passed for the touched research/FreqAI/guidance files.
   - Required artifacts exist:
     - `trader_confluence_1h_latest.parquet`
     - `trader_confluence_column_dictionary.csv`
     - epsilon direct-test summary CSV
     - tiny feature-discovery result CSV
     - epsilon queue JSON
4. Queue launch:
   - Started `run_freqai_experiment_queue_until_idle.py` in the background.
   - Current observed status after launch: `1` running, `11` pending.
   - First running experiment: price-only control for `downside_continuation_next_3h` over `spot_q4_2025`.
   - Runner logs are under:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_010227_427619\queue_runner_logs`
5. Scorer fix:
   - The first three Q4 downside-continuation runs completed their backtests but failed scoring because the scorer loaded actual labels from the older structure/orderbook frame.
   - Root fix: `score_freqai_experiment.py` now merges missing trader-confluence answer columns, such as `downside_continuation_next_3h`, from `trader_confluence_1h_latest.parquet`.
   - The failed Q4 runs were rescored after the fix.
6. Queue completion:
   - Final queue status: `completed`.
   - Completed experiments: `12/12`.
   - Ledger: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\freqai_research_results_ledger.csv`
7. Plain-English result:
   - Downside-continuation after the first break is not useful yet. In both Q4 2025 and the recent window, the epsilon features did not beat price-only in the actual crash-event rows.
   - Support-reclaim/exhaustion after a downside break is worth more testing. In the recent window, epsilon features improved the crash-event ranking over price-only, especially the component-only profile.
8. Completion/removal status:
   - The feature-definition milestone now has dictionary, direct-test, FreqAI validation, weak/rejected ideas, and next candidates documented.
   - Do not remove any objective yet.

### 2026-06-03 GKG Pre-2023 Conversion Orchestration

1. Objective:
   - Convert historical GKG raw ZIPs for 2020-2022 into compact 1h trader-readable metadata features.
   - Delete only raw GKG ZIPs after validated extraction to free D-drive space.
   - Preserve 2023-2026 raw archives for later overlap testing with historical Bybit orderbook data.
   - Track missing intervals and target-download/extract/delete those gaps until 2020-2022 have no known gaps.
2. Active automation:
   - Heartbeat `gkg-pre-2023-conversion-monitor` is active.
   - It runs `user_data\Custom_Launcher\research\context_features\run_gkg_pre2023_orchestrator_tick.ps1`.
   - Deletion is allowed only through the orchestrator wrapper, not by manual ad hoc file removal.
3. Orchestrator behaviour:
   - `gkg_pre2023_orchestrator.py` validates extraction summaries and parquet outputs before raw deletion.
   - It writes deletion manifests under `user_data\research_news_data\context_features\news_gkg_hourly\pre2023_orchestrator`.
   - If a year is partially covered, it records `pending_gap_intervals` from the coverage report before deleting available raw.
   - After space is freed, it downloads only the recorded missing intervals, extracts those gap windows, and deletes only the raw ZIPs in that gap window.
4. Current status:
   - 2020 extraction is running from existing D-drive raw files.
   - Raw coverage recorded before extraction: `32,551` of `35,136` expected files, `92.6429%`, with `2,585` missing 15-minute files.
   - Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_hourly_metadata_extract_2020_20260602_183055.log`.
   - Last observed progress: `28,000` of `32,551` files processed with `46,621,330` streamed GKG rows and `0` corrupt files.
5. Completion/removal status:
   - No objective is currently 100% complete.
   - Do not remove any objective yet.

### 2026-06-05 Parallel FreqAI Branch Goal Setup

1. Added a focused branch guidance document:
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_parallel_branch_goals.md`
2. Branches defined:
   - Branch A: structure + volume breakouts.
   - Branch B: downside risk and exhaustion.
   - Branch C: orderbook behaviour states.
   - Branch D: regime and confluence gates.
3. The branch plan keeps the research split while evidence is weak, then merges later as:
   - setup signal
   - risk state
   - orderbook confirmation
   - regime filter
4. News/context remains parked for these branch queues until its separate formatting pipeline is marked ready.
5. Next step:
   - Build executable queue manifests from existing validated profiles for each branch.

### 2026-06-05 Parallel FreqAI Branch Queues

1. Created guarded FreqAI branch queues from existing validated profiles. News/context profiles were intentionally excluded.
2. Branch A: structure + volume breakouts.
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020338_402028\freqai_experiment_queue.json`
   - Experiments: `16`.
   - Families: price, structure/VP, delta components, delta gated confluence.
   - Targets: breakout success and breakout failure.
   - Preflight: `0` errors, `0` warnings.
3. Branch B: downside risk and exhaustion.
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020454_865507\freqai_experiment_queue.json`
   - Experiments: `18`.
   - Families: price, epsilon components, epsilon gated confluence.
   - Targets: downside continuation, support reclaim, downside exhaustion.
   - Preflight: `0` errors, `0` warnings.
4. Branch C: orderbook behaviour states.
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020346_348724\freqai_experiment_queue.json`
   - Experiments: `24`.
   - Families: price, structure/VP, orderbook, structure/VP + orderbook.
   - Targets: breakout success, breakdown success, breakout failure.
   - Preflight: `0` errors, `1` warning for a single orderbook row below `0.50` coverage.
5. Branch D: regime and confluence gate.
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020348_711597\freqai_experiment_queue.json`
   - Experiments: `16`.
   - Families: price, structure/VP, delta components, delta gated, epsilon components, epsilon gated.
   - Targets: 6h return and 24h drawdown path risk.
   - Preflight: `0` errors, `0` warnings.
6. Fixed queue preflight target-balance audit:
   - `freqai_experiment_queue.py` now loads missing answer columns from `trader_confluence_1h_latest.parquet`, matching the scorer behaviour.
   - This removed false missing-column warnings for epsilon Branch B targets.
7. Completion/removal status:
   - The branch-goal setup is complete.
   - No canonical objective should be removed without user approval.
# 2026-06-05 - Concept lifecycle research process added

1. Added `ai_guidance_docs/concept_lifecycle_research_process.md` as the formal operating process for iterative trader-confluence research.
2. The new process defines concept cards, lifecycle statuses, branch quotas, pass/rework rules, and a formal lifecycle ledger target.
3. The process explicitly includes cleanup/consolidation of old scattered reports and queues by importing evidence paths into the lifecycle ledger before any user-approved cleanup.
4. Updated `ai_guidance_docs/README.md`, `AGENTS.md`, and `ai_guidance_docs/current_objectives.md` so future agents work from the new document when continuing this objective.

# 2026-06-05 - Concept lifecycle ledger and first cycle started

1. Added `user_data/Custom_Launcher/research/context_features/concept_lifecycle_ledger.py`.
2. Generated the formal lifecycle tracker:
   - `user_data/research_news_data/context_features/reports/trader_concept_lifecycle_ledger.csv`
   - `user_data/research_news_data/context_features/reports/trader_concept_lifecycle_summary.md`
   - `user_data/research_news_data/context_features/reports/trader_concept_evidence_inventory.csv`
   - `user_data/research_news_data/context_features/reports/trader_concept_next_cycle_plan.csv`
   - `user_data/research_news_data/context_features/reports/trader_concept_next_cycle_plan.md`
3. Imported existing direct-test, threshold-sweep, feature-discovery, FreqAI, report, and queue evidence into the lifecycle structure.
4. Context/news evidence is currently marked `deferred_for_data` rather than treated as active, because news/context extraction is being reworked separately and should not be mixed into current conclusions.
5. Ran a fresh direct-test pass:
   - `trader_confluence_direct_tests_concept_lifecycle_cycle_20260605_summary.csv`
   - 197 summary rows.
6. Ran the structure/volume FreqAI validation queue:
   - Queue: `queue_20260605_020338_402028`
   - Result: 14 completed, 2 failed.
   - Failure reason: Q1 structure-only profiles dropped almost all training rows due to NaNs, leaving too few samples. Treat as data/window readiness failure, not concept rejection.
7. Ran the downside/exhaustion FreqAI validation queue:
   - Queue: `queue_20260605_020342_609646`
   - Result: 18 completed, 0 failed.
8. Ran the orderbook-state FreqAI validation queue:
   - Queue: `queue_20260605_020346_348724`
   - Result: 18 completed, 6 failed.
   - Failure pattern: Q1 structure-dependent profiles failed from the same NaN-heavy structure coverage issue; orderbook-only profiles completed.
9. Ran the regime/confluence FreqAI validation queue:
   - Queue: `queue_20260605_020348_711597`
   - Result: 14 completed, 2 failed.
   - Failure pattern: Q1 structure-only regression profiles failed from the same NaN-heavy structure coverage issue.
10. Updated lifecycle import logic so FreqAI control-comparison rows and regression rows are classified properly.
11. Added independent lead-family outputs:
   - `user_data/research_news_data/context_features/reports/trader_independent_promising_leads.csv`
   - `user_data/research_news_data/context_features/reports/trader_independent_promising_leads.md`
12. Current lifecycle counts after import:
   - `freqai_promising`: 24 validation records.
   - `sweep_promising`: 9 concept/target records.
   - `direct_promising`: 44 concept/target records.
   - `needs_rework`: 49 records.
   - `deferred_for_data`: 108 records.
   - `rejected_for_now`: 116 records.
13. Current grouped independent promising lead families: 33.
14. Caveat: grouped lead families still need ranking, duplicate review, and concept-card cleanup before any are treated as strategy-ready.

# 2026-06-05 - Concept card cleanup and feature-discovery consolidation

1. Fixed `concept_lifecycle_ledger.py` so feature-discovery evidence is imported from all combined discovery reports, not only the newest tiny report.
2. Added trader-readable grouping for repeated raw clues:
   - price momentum breakout continuation
   - price momentum breakdown continuation
   - volatility expansion drawdown risk
   - market-structure invalidation drawdown
   - TLV2 resistance-line drawdown
   - bull-trend regime continuation filter
   - orderbook wall-distance shift
3. Rebuilt the lifecycle tracker.
4. Current lifecycle record counts:
   - `direct_promising`: 125
   - `freqai_promising`: 24
   - `sweep_promising`: 9
   - `needs_rework`: 65
   - `deferred_for_data`: 108
   - `rejected_for_now`: 129
5. Current cleaned concept-card counts:
   - `A - working lead`: 3
   - `B - promising validation`: 5
   - `B - sweep-promising`: 6
   - `C - strong feature clue`: 8
   - `D - convert or rework`: 26
6. The current milestone of at least 20 promising trader-readable leads is met at the research-lead level: 22 cards are A/B/C.
7. These are not strategy-ready rules. The next cycle should promote A/B leads into clean direct reports and convert C leads into explicit hypotheses with controls.

# 2026-06-05 - Sieve-borrowed trader rule backtest pass

1. Added a research-only standalone trading-rule harness:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_rule_backtests.py`
2. Purpose:
   - Convert promising trader-readable research leads into simple BTC 1h long/short entry rules.
   - Test each rule alone, then combine the strongest non-overlapping rules into a multi-scenario block.
   - Keep this separate from production strategy behaviour.
3. Initial broad rule pass:
   - Tested `18` rule families / `72` variants.
   - Result was weak: no rule passed the strict quality filter.
   - Best broad standalone rule was `bull_trend_regime_long__thr_0.75`: `461` trades, `45.1%` win rate, `51.8%` return, `-24.4%` drawdown.
   - This did not beat buy-and-hold and had too much drawdown for promotion.
4. Sieve-borrowed pass:
   - Borrowed plausible behaviours from `SIEVE3_GUARD_CANDIDATES.md` and `sieve3_candidates/from_20260603_revised_guard_profitable/MANIFEST.md`.
   - Added TLV2 resistance breakout, TLV2 support breakdown, VP support/reject, POC reclaim/reject, LVN fast travel, MTF BOS, and rectangle breakout/breakdown rule families.
5. Review correction:
   - The first active-period rate calculation let a 4-trade rule look positive.
   - Fixed by requiring at least `10` absolute trades before a variant can pass `positive_quality`.
6. Final corrected pass:
   - Command: `python user_data\Custom_Launcher\research\context_features\trader_rule_backtests.py --tag sieve_exact_min10_20260605 --execute`
   - Tested `36` rule families / `144` variants.
   - Final combined block:
     - `94` trades.
     - `53.2%` win rate.
     - `37.8%` compounded return.
     - `-8.6%` max drawdown.
     - `1.542` profit factor.
     - Did not beat buy-and-hold over the full 2020-2026 BTC snapshot.
7. Final selected standalone behaviours:
   - Rectangle/compression breakdown short: `14` trades, `42.9%` win rate, `9.5%` return, `-3.9%` drawdown, `2.713` profit factor.
   - Prior-day high break + VP/pressure long: `45` trades, `53.3%` win rate, `6.1%` return, `-9.1%` drawdown, `1.193` profit factor.
   - Prior-day low break + VP/pressure short: `34` trades, `50.0%` win rate, `3.4%` return, `-5.9%` drawdown, `1.137` profit factor.
   - Rectangle/compression breakout long: `18` trades, `33.3%` win rate, `3.2%` return, `-3.1%` drawdown, `1.427` profit factor.
8. Combined-rule year split:
   - 2025: `62` trades, `46.8%` win rate, `6.8%` return.
   - 2026: `32` trades, `65.6%` win rate, `29.0%` return.
9. Important caveat:
   - Custom structure/VP/TLV2 features only have meaningful coverage in the later part of the snapshot, so this is a promising 2025-2026 lead set, not a full 2020-2026 proof.
10. Storage check:
   - `C:\FreqTradeStuff\user_data\models`: about `28.34 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs`: about `0.45 GB`.
   - `C:\FreqTradeStuff\user_data\backtest_results`: about `0.00 GB`.
   - No deletion was performed.
11. Freqtrade research strategy added:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockResearchStrategy.py`
   - It consumes the exported signal parquet instead of recalculating custom indicators or reading live databases in `populate_indicators()`.
   - Signal file: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_signals_sieve_exact_min10_20260605.parquet`
12. Freqtrade verification:
   - Compile passed for `trader_rule_backtests.py` and `TraderRuleBlockResearchStrategy.py`.
   - Backtest command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_21-31-27.zip`
13. Freqtrade result with `--max-open-trades 1`:
   - `94` trades.
   - `57.4%` win rate.
   - `50.73%` account return.
   - `9.73%` max drawdown.
   - `1.66` profit factor.
   - Market change over the same backtest window: `-14.54%`.
   - Long / short split: `53 / 41`.
   - Long profit: `9.83%`; short profit: `40.90%`.
14. Sizing caveat:
   - A prior Freqtrade run without `--max-open-trades 1` reported only `4.30%` because the config split capital as if up to `10` positions could be open.
   - Use the `--max-open-trades 1` run as the relevant one-position BTC rule-block result.

# 2026-06-05 - 50-100 lead registry and first confluence screen

1. User clarified the real goal:
   - Build about `50` to `100` leads.
   - Test whether confluence improves entry quality.
   - Develop exits tailored to each successful entry family.
   - Add leverage/risk sizing from extra metrics such as orderbook and news/context when those data sources are ready.
2. Added a lightweight lead registry builder:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_registry.py`
3. Registry inputs:
   - `trader_concept_lifecycle_ledger.csv`
   - `trader_independent_promising_leads.csv`
   - `trader_concept_next_cycle_plan.csv`
   - `trader_rule_backtests_sieve_exact_min10_20260605_summary.csv`
   - `trader_rule_backtests_sieve_exact_min10_20260605_selected_rules.csv`
4. Registry output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260605.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_latest.csv`
5. Registry result:
   - `100` leads.
   - Branch counts:
     - `structure_volume`: `41`
     - `orderbook_state`: `18`
     - `downside_risk`: `18`
     - `sieve_borrowed`: `18`
     - `regime_confluence`: `5`
   - News/context was not forced into the active lead mix because source-specific context extraction and coverage are still under separate development.
6. Added confluence-plan output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_plan_20260605.csv`
   - `572` planned checks across structure, volume, orderbook, regime, context-readiness, and post-start momentum.
7. Added first confluence checker:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py`
8. First confluence screen target:
   - The `94` selected Sieve-derived entries from `trader_rule_backtests_sieve_exact_min10_20260605_combined_trades.csv`.
   - Question: if we keep the same entries but require another source family to agree, do entries become cleaner?
9. First strict confluence output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260605_strict.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260605_strict.md`
10. First strict confluence result:
   - Broad filters did not improve the full `94`-trade block.
   - The base selected block had `94` trades, `53.2%` win rate, `0.359%` average net return per trade, and `1.542` profit factor in the fast trade harness.
   - Strong volume, structure, range, compression, and orderbook filters all reduced the whole-block quality when applied globally.
11. Rule-specific watchlist:
   - `sieve_pattern_rectangle_breakout_long` plus VP context:
     - `9` filtered trades.
     - Win rate improved from `36.4%` to `44.4%`.
     - Profit factor improved from `1.603` to `2.719`.
     - Needs more sample size before promotion.
   - `sieve_prior_day_low_break_vp_short` plus strong volume and structure:
     - `11` filtered trades.
     - Win rate improved from `58.1%` to `63.6%`.
     - Profit factor improved from `1.545` to `1.616`.
     - Worth more testing as a tailored short-entry confluence filter.
12. Interpretation:
   - Confluence should be per-lead and trader-state-specific, not one broad global filter over every entry.
   - Some existing selected entries already contain volume/VP/structure logic, so adding the same broad filter can be redundant or harmful.
   - Orderbook-present alone is not a trading signal; orderbook must be tested as directional agreement, warning, or risk sizing.
13. Added next-stage queue builder:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_next_stage_queue.py`
14. Next-stage queue output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_next_stage_queue_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_next_stage_queue_20260605.md`
15. Queue result:
   - `46` concrete next tasks.
   - `6` tailored confluence retests.
   - `4` exit-research tasks for the selected Sieve-derived entry families.
   - `36` per-lead confluence tasks for the highest-ranked registry leads.
16. Verification:
   - Compile passed for:
     - `trading_lead_registry.py`
     - `trading_lead_confluence_checks.py`
     - `trading_lead_next_stage_queue.py`

# 2026-06-05 - Tailored exits and first risk overlay

1. Added tailored exit research script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_exit_research.py`
2. Exit research inputs:
   - Signals: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_signals_sieve_exact_min10_20260605.parquet`
   - Features/OHLCV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
3. Exit research outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605.md`
4. Exit sweep result:
   - Tested `408` stop/target/hold variants across the `4` selected entry families.
   - Selected one tailored exit per entry family.
5. Best tailored exits:
   - Rectangle breakout long:
     - Hold `48h`, stop `3.5%`, target `7.5%`.
     - `11` trades, `72.7%` win rate, `29.7%` fast-harness return, `-1.4%` drawdown, `13.17` profit factor.
   - Rectangle breakdown short:
     - Hold `10h`, stop `2.5%`, target `5.0%`.
     - `10` trades, `50.0%` win rate, `11.6%` fast-harness return, `-2.2%` drawdown, `4.68` profit factor.
   - Prior-day low break VP short:
     - Hold `36h`, stop `3.5%`, target `5.0%`.
     - `31` trades, `51.6%` win rate, `18.3%` fast-harness return, `-7.2%` drawdown, `1.61` profit factor.
   - Prior-day high break VP long:
     - Hold `72h`, stop `3.5%`, target `2.0%`.
     - `42` trades, `64.3%` win rate, `8.0%` fast-harness return, `-7.9%` drawdown, `1.25` profit factor.
6. Combined tailored-exit fast-harness result:
   - `76` trades.
   - `52.6%` win rate.
   - `39.3%` compounded return.
   - `-10.3%` max drawdown.
   - `1.563` profit factor.
   - Interpretation: average return and profit factor improved slightly versus the fast-harness original, but fewer trades and higher drawdown mean this is not a clean win.
7. Research strategy handling:
   - Baseline fixed-exit strategy remains:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockResearchStrategy.py`
   - Tailored-exit variant was added separately:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockTailoredExitResearchStrategy.py`
   - The tailored variant uses the selected take-profit and hold-hour values.
   - It uses global stoploss `-3.5%`, matching the widest selected stop. Exact per-rule stop-loss matching is not yet implemented in Freqtrade.
8. Tailored-exit Freqtrade validation:
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockTailoredExitResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-00-34.zip`
   - Result:
     - `83` trades.
     - `60.2%` win rate.
     - `49.19%` account return.
     - `11.56%` max account underwater.
     - `1.60` profit factor.
   - Comparison:
     - Previous fixed-exit Freqtrade result was `94` trades, `57.4%` win rate, `50.73%` return, `9.73%` max drawdown, `1.66` profit factor.
     - Tailored exits improved win rate but did not improve the whole strategy result.
9. Added first risk-overlay script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
10. Risk-overlay outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605.md`
11. Risk-overlay result:
   - Tested simple sizing overlays from signal score, volume, structure, orderbook, volatility, and a conservative combined mix.
   - None beat equal `1x` sizing on the current tailored-exit trade block.
   - Directional volume sizing raised return slightly but increased drawdown and reduced profit factor, so it was rejected for now.
   - Orderbook warning sizing reduced drawdown but cut return too much, so it was rejected for now.
12. Verification:
   - Compile passed for:
     - `trading_lead_exit_research.py`
     - `trading_lead_risk_overlay.py`
     - `TraderRuleBlockResearchStrategy.py`
     - `TraderRuleBlockTailoredExitResearchStrategy.py`

# 2026-06-05 - Per-rule confluence validation

1. Added per-rule confluence runner:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_per_rule_confluence.py`
2. Purpose:
   - Test confluence filters separately per selected entry family.
   - Compare baseline exits and tailored exits.
   - Export reusable filtered signal parquet files for Freqtrade validation.
3. Outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605.md`
4. Signal exports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_baseline_exit_selected_confluence.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_tailored_exit_selected_confluence.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_best_per_rule_selected_confluence.parquet`
5. Per-rule confluence selected rows:
   - Rectangle breakout long + VP context + tailored exit:
     - Kept `9` of `11` decision points.
     - Win rate improved from `72.7%` to `77.8%`.
     - Average return improved from `2.43%` to `2.99%`.
     - Profit factor improved from `13.17` to `15.25`.
   - Rectangle breakout long + VP context + baseline exit:
     - Kept `9` of `11` decision points.
     - Win rate improved from `36.4%` to `44.4%`.
     - Average return improved from `0.26%` to `0.54%`.
     - Profit factor improved from `1.60` to `2.72`.
   - Prior-day low break short + strong volume plus structure + baseline exit:
     - Kept `11` of `31` decision points.
     - Win rate improved from `58.1%` to `63.6%`.
     - Average return improved from `0.39%` to `0.47%`.
     - Profit factor improved from `1.55` to `1.62`.
   - Prior-day low break short + strong volume + baseline exit:
     - Kept `25` of `31` decision points.
     - Watchlist only.
6. Added Freqtrade confluence strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockConfluenceResearchStrategy.py`
   - Uses `trading_lead_signals_20260605_best_per_rule_selected_confluence.parquet`.
7. Freqtrade confluence validation:
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockConfluenceResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-05-53.zip`
   - Result:
     - `16` trades.
     - `50.0%` win rate.
     - `4.78%` account return.
     - `8.35%` max account underwater.
     - `1.25` profit factor.
8. Interpretation:
   - Per-rule confluence improves some fast-harness entry-family metrics.
   - It does not yet improve the actual Freqtrade strategy block.
   - The confluence-filtered strategy is too small and the filtered prior-day-low short subset lost money in Freqtrade.
   - Current best remains the unfiltered fixed-exit rule block.
9. Verification:
   - Compile passed for `trading_lead_per_rule_confluence.py`.
   - Compile passed for `TraderRuleBlockConfluenceResearchStrategy.py`.

# 2026-06-05 - Rectangle/VP expansion and quality split

1. Added expanded rectangle/compression plus VP tester:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_rectangle_vp_expansion.py`
2. Purpose:
   - Expand the strongest surviving clue, rectangle breakout long plus VP context, into nearby trader-readable variants.
   - Keep this separate from the existing fixed-exit rule block.
3. Variants covered:
   - 4h and 1d rectangle upper breaks with VP support.
   - Multi-timeframe rectangle upper breaks.
   - Rectangle/compression squeeze releases.
   - Rectangle breaks above VAH and below VAL.
   - Rectangle breaks into LVN/thin-liquidity areas.
   - Rectangle breaks with TLV2 or orderbook acceptance.
   - Compression plus recent range break with VP support.
4. Outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605.md`
5. First expansion result:
   - Tested `221` variant/threshold/exit combinations.
   - Selected `9` unique base variants after de-duplicating threshold variants.
   - Fast harness combined block: `66` trades, `53.0%` win rate, `16.3%` compounded return, `-11.9%` max drawdown, `1.320` profit factor.
6. Added Freqtrade validation strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpExpansionStrategy.py`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_expansion_20260605_selected.parquet`
7. Freqtrade expanded-block validation:
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockRectangleVpExpansionStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-15-56.zip`
   - Result:
     - `66` trades.
     - `56.1%` win rate.
     - `23.33%` account return.
     - `9.95%` max account underwater.
     - `1.51` profit factor.
8. Quality split:
   - Exported high-quality subset:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_expansion_20260605_quality.parquet`
   - Kept:
     - `rect_break_below_val_short`
     - `rect_break_orderbook_accept_long`
     - `rect_squeeze_upper_release_long`
   - Excluded broad `compression_range_break_vp_long` because it supplied many trades but weak average edge.
9. Added quality-subset Freqtrade strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpQualityStrategy.py`
10. Freqtrade quality-subset validation:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-16-45.zip`
   - Result:
     - `16` trades.
     - `62.5%` win rate.
     - `18.57%` account return.
     - `3.44%` max account underwater.
     - `3.89` profit factor.
11. Interpretation:
   - The broader rectangle/VP expansion is valid but not better than the current best 94-trade fixed-exit block.
   - The high-quality subset is sparse but much cleaner and should be promoted as a lead to develop, not as a finished strategy.
   - The strongest emerging trader story is: rectangle/compression break plus VP value-area/VAL/VAH context, with orderbook acceptance improving a small number of longs.
12. Verification:
   - Compile passed for:
     - `trading_lead_rectangle_vp_expansion.py`
     - `TraderRuleBlockRectangleVpExpansionStrategy.py`
     - `TraderRuleBlockRectangleVpQualityStrategy.py`

# 2026-06-05 - Rectangle/VP refinement pass

1. Added focused refinement script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_rectangle_vp_refinement.py`
2. Purpose:
   - Repair the noisy broad `compression_range_break_vp_long` idea.
   - Improve the `rect_break_below_val_short` family by filtering out cases where bullish volume/VP opposition is fighting the short.
3. Refinement ideas tested:
   - Compression breakout only when compression is genuinely strong.
   - Compression breakout with persistent 24h volume and positive pressure.
   - Compression breakout with VP value acceptance above VAH.
   - Compression breakout not already overextended at range high.
   - Compression breakout with orderbook acceptance.
   - VAL breakdown with weak bullish volume.
   - VAL breakdown with weak long-side VP opposition.
   - VAL breakdown as controlled/moderate-volume break.
   - VAL breakdown while price still has room toward range low.
   - Rechecks of the previous rectangle squeeze and orderbook-acceptance quality leads.
4. Direct harness outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605.md`
5. Direct harness result:
   - `185` variant/threshold/exit combinations tested.
   - `7` selected variants.
   - Combined direct-harness block: `159` trades, `52.2%` win rate, `190.5%` compounded return, `-18.4%` max drawdown, `1.672` profit factor.
6. Added Freqtrade refinement strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpRefinementStrategy.py`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_refinement_20260605_selected.parquet`
7. Freqtrade refinement result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-23-39.zip`
   - Result:
     - `54` trades.
     - `50.0%` win rate.
     - `13.30%` return.
     - `12.46%` drawdown.
     - `1.36` profit factor.
8. Interpretation of full refinement:
   - The direct-harness spike did not survive Freqtrade.
   - The broad `compression_break_vp_long_volume_persistent` rule is not promoted; in Freqtrade it contributed only `35` trades, `48.6%` win rate, and `-1.17%` profit.
   - This is a useful failure because it confirms broad compression-volume long still needs rework.
9. Added refined-quality signal subset:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_refinement_20260605_quality.parquet`
   - Kept:
     - `rect_val_short_vp_opposition_low`
     - `rect_squeeze_long_quality_keep`
     - `compression_break_vp_long_value_acceptance`
   - Excluded:
     - Broad failed compression-volume long.
     - Orderbook-acceptance long variant that weakened in Freqtrade.
     - VAL range/mid and moderate-volume variants that were too weak after overlap.
10. Added refined-quality strategy:
    - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpRefinedQualityStrategy.py`
11. Refined-quality Freqtrade result:
    - Artifact:
      - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-24-14.zip`
    - Result:
      - `10` trades.
      - `60.0%` win rate.
      - `20.88%` account return.
      - `2.94%` max account underwater.
      - `5.49` profit factor.
12. Trader-readable conclusion:
    - Current refined lead is sparse but strong:
      - Break below rectangle/value area.
      - Long-side VP opposition is weak.
      - Short can work, especially if it reaches target quickly.
    - Tiny long confirmations remain watchlist only.
    - Broad compression-volume breakout is a rework candidate, not a promoted lead.
13. Verification:
    - Compile passed for:
      - `trading_lead_rectangle_vp_refinement.py`
      - `TraderRuleBlockRectangleVpRefinementStrategy.py`
      - `TraderRuleBlockRectangleVpRefinedQualityStrategy.py`

# 2026-06-05 - Adjacent support-break short expansion

1. Added adjacent short-expansion script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_support_break_short_expansion.py`
2. Purpose:
   - Test whether the sparse `rect_val_short_vp_opposition_low` lead can expand into broader support-break short entries.
3. Lead families tested:
   - Prior-low break with weak VP long opposition.
   - Prior-low break with quiet bullish volume.
   - TLV2 support break with weak VP long opposition.
   - Bearish structure breakdown with weak VP long support.
   - VAL break plus LVN/thinness below.
   - Support break with orderbook breakdown acceptance.
   - Bearish pressure near TLV2 support.
   - Support break while price is still mid-range.
4. Direct harness outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605.md`
5. Direct harness result:
   - `160` variant/threshold/exit combinations tested.
   - `4` selected variants.
   - Combined direct-harness block: `92` trades, `38.0%` win rate, `4.7%` return, `-15.7%` max drawdown, `1.093` profit factor.
6. Added Freqtrade validation strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportBreakShortExpansionStrategy.py`
7. Freqtrade validation:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-28-04.zip`
   - Result:
     - `85` trades.
     - `42.4%` win rate.
     - `-0.03%` account return.
     - `12.27%` max account underwater.
     - `1.00` profit factor.
8. Interpretation:
   - Broad support-break expansion is rejected as currently defined.
   - The useful VAL/VP-opposition short did not generalise cleanly to broader support-break signals.
   - `support_break_orderbook_accept_short` was the only mildly positive sub-piece in Freqtrade (`30` entries, `2.8%` contribution), but it is too weak to promote.
9. Next action:
   - Keep the narrow VAL/VP-opposition short as a promoted sparse lead.
   - Park broad support-break expansion until orderbook support-removal features can be made more specific.
10. Verification:
    - Compile passed for:
      - `trading_lead_support_break_short_expansion.py`
      - `TraderRuleBlockSupportBreakShortExpansionStrategy.py`

# 2026-06-05 - Orderbook support-removal short refinement

1. Added focused orderbook-support-removal refinement:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_orderbook_support_removal_short_refinement.py`
2. Purpose:
   - Improve the rejected broad support-break short branch by requiring concrete orderbook support-clearing evidence.
3. Concepts tested:
   - VAL breakdown with support removed.
   - VAL breakdown with downside vacuum after support removal.
   - VAL breakdown with low bounce/failure warnings.
   - Recent support break where orderbook support is cleared.
   - Recent support break with support removed and bounce warnings low.
   - Recent support break with bearish book pressure.
   - Fast target-or-out support break.
4. Direct harness result:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605.md`
   - `118` variant/threshold/exit combinations tested.
   - `2` selected variants.
   - Combined direct-harness block: `43` trades, `60.5%` win rate, `26.2%` return, `-6.5%` max drawdown, `2.333` profit factor.
5. Selected direct variants:
   - `support_break_support_cleared_short`:
     - `18` trades, `66.7%` win rate, `21.8%` return, `-4.5%` max drawdown, `3.017` profit factor.
   - `val_break_downside_vacuum_after_support_removed_short`:
     - `27` trades, `55.6%` win rate, `5.4%` return, `-4.4%` max drawdown, `1.638` profit factor.
6. Added Freqtrade validation strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockOrderbookSupportRemovalShortStrategy.py`
7. Freqtrade validation:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-32-42.zip`
   - Result:
     - `43` trades.
     - `58.1%` win rate.
     - `12.40%` account return.
     - `6.69%` max account underwater.
     - `1.49` profit factor.
8. Interpretation:
   - Sharper orderbook support-removal works better than broad support-break expansion.
   - It is a promoted-watchlist family, not final production logic.
   - Trader story:
     - support/recent low breaks,
     - orderbook says support cleared or removed,
     - VAL/value break with downside vacuum can work as a quick short,
     - holding too broad support-break shorts without support-cleared evidence fails.
9. Verification:
   - Compile passed for:
     - `trading_lead_orderbook_support_removal_short_refinement.py`
     - `TraderRuleBlockOrderbookSupportRemovalShortStrategy.py`

# 2026-06-05 - Support-removal exit/risk split validation

1. Added exit/risk research for the promoted support-removal short family:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_support_removal_exit_risk_research.py`
2. Direct exit/risk screen:
   - `43` trades.
   - `60.5%` win rate.
   - `26.2%` return.
   - `-6.5%` max drawdown.
   - `2.333` profit factor.
3. Important logic finding:
   - `support_break_support_cleared_short` and `val_break_downside_vacuum_after_support_removed_short` do not want the same stop profile.
   - A global tight stop improved the quick VAL/vacuum short but damaged the support-cleared short.
4. Added standalone research-only validation strategies:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportRemovalSupportClearedStrategy.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportRemovalValVacuumTightStrategy.py`
5. Added split signal files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_support_removal_support_cleared_20260605.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_support_removal_val_vacuum_20260605.parquet`
6. Freqtrade standalone support-cleared result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-40-03.zip`
   - `18` trades.
   - `61.1%` win rate.
   - `10.65%` return.
   - `3.73%` max account underwater.
   - `1.95` profit factor.
7. Freqtrade standalone VAL/vacuum tight-stop result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-40-14.zip`
   - `25` trades.
   - `56.0%` win rate.
   - `5.67%` return.
   - `2.85%` max account underwater.
   - `1.65` profit factor.
8. Global tight-stop control:
   - `TraderRuleBlockOrderbookSupportRemovalTightStopStrategy` produced `43` trades, `48.8%` win rate, `1.93%` return, `5.16%` drawdown, and `1.09` profit factor.
   - Verdict: reject the global tight stop for the mixed block.
9. Interpretation:
   - Support-cleared short is the better standalone lead.
   - VAL/vacuum short is weaker but usable as a fast-invalidation quick short.
   - Future merged rule blocks should either keep these as separate entry families or add per-rule stop handling, not one shared stop.
10. Verification:
    - Compile passed for:
      - `trading_lead_support_removal_exit_risk_research.py`
      - `TraderRuleBlockOrderbookSupportRemovalTightStopStrategy.py`
      - `TraderRuleBlockSupportRemovalSupportClearedStrategy.py`
      - `TraderRuleBlockSupportRemovalValVacuumTightStrategy.py`

# 2026-06-05 - Support-removal lead registry and confluence pass

1. Added generic validated-entry lead ingestion to:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_registry.py`
2. Added validated lead source:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_entry_leads.csv`
3. Rebuilt registry:
   - `100` leads.
   - Branch mix:
     - `41` structure/volume.
     - `18` orderbook.
     - `18` downside-risk.
     - `18` Sieve-borrowed.
     - `5` regime/confluence.
   - `2` leads are now marked as Freqtrade-validated entry rules.
4. Rebuilt next-stage queue:
   - `48` tasks.
   - `36` per-lead confluence.
   - `6` tailored confluence retests.
   - `6` exit-research tasks.
5. Extended per-rule confluence runner:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_per_rule_confluence.py`
   - It now infers validated exit profiles for the two support-removal short rules instead of ignoring non-Sieve signal files.
6. Ran support-removal confluence screen:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal.md`
   - `36` filter/profile combinations tested.
   - `3` selected/watchlist filters.
7. Confluence Freqtrade validation:
   - Support-cleared with selected structure/VP confluence:
     - Strategy:
       - `TraderRuleBlockSupportRemovalSupportClearedConfluenceStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-44-55.zip`
     - `16` trades, `62.5%` win rate, `10.30%` return, `3.75%` drawdown, `2.06` profit factor.
   - VAL/vacuum with regime confluence:
     - Strategy:
       - `TraderRuleBlockSupportRemovalValVacuumConfluenceStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-45-06.zip`
     - `24` trades, `54.2%` win rate, `5.51%` return, `2.85%` drawdown, `1.63` profit factor.
8. Interpretation:
   - Confluence did not materially improve either support-removal short in Freqtrade.
   - Support-cleared confluence removed two trades while keeping nearly the same return, so it may be useful as a quality/risk filter but not a return booster.
   - VAL/vacuum confluence was neutral/slightly weaker than the standalone lead.
9. Next action:
   - Keep standalone support-removal leads as preferred.
   - Test more specific confluence next:
     - support-cleared plus no support reclaim/rebuild warning,
     - VAL/vacuum plus immediate downside momentum and no bounce warning.
10. Verification:
    - Compile passed for:
      - `trading_lead_registry.py`
      - `trading_lead_per_rule_confluence.py`
      - `TraderRuleBlockSupportRemovalSupportClearedConfluenceStrategy.py`
      - `TraderRuleBlockSupportRemovalValVacuumConfluenceStrategy.py`

# 2026-06-05 - Specific no-reclaim confluence for VAL/vacuum shorts

1. Added targeted downside-failure filters to:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py`
2. New filters:
   - `no_support_reclaim_warning`
   - `no_support_rebuild_warning`
   - `bearish_pressure_persists`
   - `downside_vacuum_still_open`
   - `breakdown_failure_warning_absent`
   - `panic_after_break_state`
3. Trader question:
   - When price breaks below value into a downside vacuum, does the quick short improve if the book does not warn that broken support is being reclaimed?
4. Direct confluence result:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_specific.md`
   - `no_support_reclaim_warning` kept `9` of `25` VAL/vacuum decision points.
   - Direct harness result:
     - `77.8%` win rate vs `56.0%` baseline.
     - `0.79%` average return vs `0.15%` baseline.
     - `4.90` profit factor vs `1.46` baseline.
5. Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-49-29.zip`
   - Result:
     - `9` trades.
     - `77.8%` win rate.
     - `7.89%` return.
     - `1.45%` max account underwater.
     - `4.41` profit factor.
6. Exit research:
   - Updated `trading_lead_exit_research.py` with support-removal/vacuum-specific exit grids.
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_val_vacuum_no_reclaim_exit.md`
   - Fast harness selected:
     - `8h` hold.
     - `1.0%` stop.
     - `3.5%` target.
7. Tuned-exit Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExitTunedStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-51-42.zip`
   - Result:
     - `9` trades.
     - `77.8%` win rate.
     - `7.35%` return.
     - `1.05%` max account underwater.
     - `4.31` profit factor.
8. Interpretation:
   - The no-reclaim filter is the first support-removal confluence filter that clearly improved the quick VAL/vacuum short in both direct testing and Freqtrade.
   - The original 6h/1.4% stop/2.5% target keeps slightly more return.
   - The tuned 8h/1.0% stop/3.5% target lowers drawdown.
   - Sample size is only `9` trades, so this is a promoted sparse lead, not production logic.
9. Registry/queue update:
   - Added `lead__validated__val_vacuum_no_support_reclaim_warning_short` to:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_entry_leads.csv`
   - Rebuilt the registry and queue.
   - Queue now has `49` tasks:
     - `36` per-lead confluence.
     - `7` exit-research.
     - `6` tailored confluence retests.
10. Next action:
    - Expand similar no-reclaim/no-bounce filters to increase trade count without losing quality.
    - Add month/window stability checks before risk sizing.
11. Verification:
    - Compile passed for:
      - `trading_lead_confluence_checks.py`
      - `trading_lead_per_rule_confluence.py`
      - `trading_lead_exit_research.py`
      - `TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy.py`
      - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExitTunedStrategy.py`

# 2026-06-05 - VAL/vacuum no-reclaim expansion and adjacent pressure validation

1. Objective:
   - Try to increase the useful trade count around the working VAL/vacuum no-reclaim short without hiding the trader story.
2. Code changes:
   - Added/fixed:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_val_vacuum_expansion.py`
   - Added validation strategies:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExpandedStrategy.py`
     - `TraderRuleBlockSupportRemovalValVacuumPressureExpandedStrategy.py`
3. Expansion sweep:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_val_vacuum_expansion_20260605_val_vacuum_no_reclaim_expansion.md`
   - Tested `59` variants across `2` exit profiles.
   - Exported selected signal parquets for strict, looser, pressure, rebuild, and panic variants.
4. Freqtrade validation - looser no-reclaim:
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExpandedStrategy`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_no_reclaim_lte_46.parquet`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-03-36.zip`
   - Result:
     - `12` trades.
     - `66.7%` win rate.
     - `7.04%` return.
     - `1.58%` max account underwater.
     - `2.84` profit factor.
     - Positive in both year buckets: `2025` and `2026`.
5. Freqtrade validation - broader bearish-pressure version:
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumPressureExpandedStrategy`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_pressure_gte_20.parquet`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-04-09.zip`
   - Result:
     - `21` trades.
     - `61.9%` win rate.
     - `7.99%` return.
     - `2.94%` max account underwater.
     - `2.36` profit factor.
     - Positive in both year buckets: `2025` and `2026`.
6. Interpretation:
   - Strict no-reclaim remains the cleanest version.
   - Looser no-reclaim adds `3` trades but lowers quality.
   - Bearish-pressure expansion adds more trade count but is a separate adjacent lead, not proof that no-reclaim can be widened indefinitely.
7. Verification:
   - Compile passed for:
     - `trading_lead_val_vacuum_expansion.py`
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExpandedStrategy.py`
     - `TraderRuleBlockSupportRemovalValVacuumPressureExpandedStrategy.py`
8. Registry/queue update:
   - Added the two new validated-watchlist leads to:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_entry_leads.csv`
   - Rebuilt registry:
     - `100` total leads.
     - `2` `freqtrade_validated_entry_rule`.
     - `3` `freqtrade_validated_confluence_entry_rule`.
   - Rebuilt next-stage queue:
     - `49` tasks.

# 2026-06-05 - Validated orderbook overlap and core block validation

1. Objective:
   - Check whether the five validated orderbook support-removal leads are genuinely separate or mostly duplicate the same entries.
2. Added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_validated_overlap.py`
   - `TraderRuleBlockValidatedOrderbookCombinedStrategy.py`
   - `TraderRuleBlockValidatedOrderbookCoreStrategy.py`
3. Overlap report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_overlap_20260605_validated_orderbook_overlap.md`
4. Main overlap finding:
   - VAL/vacuum variants are highly nested:
     - strict no-reclaim is fully inside loose no-reclaim and broad VAL/vacuum.
     - bearish-pressure VAL/vacuum is mostly inside broad VAL/vacuum.
   - Support-cleared has zero near-overlap with strict no-reclaim VAL/vacuum.
5. All-validated combined Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockValidatedOrderbookCombinedStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-10-44.zip`
   - Result:
     - `43` trades.
     - `58.1%` win rate.
     - `11.99%` return.
     - `8.39%` drawdown.
     - `1.46` profit factor.
   - Interpretation:
     - Profitable but diluted by nested lower-quality VAL/vacuum leftovers.
6. Core orderbook Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockValidatedOrderbookCoreStrategy`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_orderbook_overlap_core_validated_orderbook.parquet`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-11-52.zip`
   - Result:
     - `27` trades.
     - `66.7%` win rate.
     - `18.02%` return.
     - `3.68%` max account underwater.
     - `2.19` profit factor.
7. Registry/queue update:
   - Added:
     - `lead__validated__orderbook_core_support_cleared_plus_no_reclaim_short`
   - Rebuilt registry:
     - `100` total leads.
     - `1` `freqtrade_validated_rule_block`.
   - Rebuilt next-stage queue:
     - `52` tasks.
8. Next action:
   - Tailor exits for the two legs in the core block.
   - Test invalidation exits:
     - support-rebuild/reclaim after no-reclaim entry.
     - pressure fade after support-cleared entry.
   - Risk sizing only after exit refinement.

# 2026-06-05 - Core orderbook tailored-exit validation

1. Objective:
   - Validate whether the promoted core orderbook block improves when each leg keeps its own exit profile.
2. Exit sweep:
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_orderbook_overlap_core_validated_orderbook.parquet`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_orderbook_core_exit_research.md`
   - Fast harness selected:
     - Support-cleared:
       - `18h` hold.
       - `2.5%` stop.
       - `5%` target.
     - Strict no-reclaim VAL/vacuum:
       - `8h` hold.
       - `1.0%` stop.
       - `3.5%` target.
3. Implementation correction:
   - The base combined strategy only had one static Freqtrade stoploss.
   - Added:
     - `TraderRuleBlockValidatedOrderbookCoreTailoredExitStrategy.py`
   - It enables `custom_stoploss` so each lead id can use its selected stop.
4. Freqtrade validation:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-15-33.zip`
   - Result:
     - `27` trades.
     - `66.7%` win rate.
     - `19.22%` return.
     - `3.50%` max account underwater.
     - `2.46` profit factor.
5. Comparison:
   - Untailored core:
     - `18.02%` return.
     - `3.68%` drawdown.
     - `2.19` profit factor.
   - Tailored core:
     - `19.22%` return.
     - `3.50%` drawdown.
     - `2.46` profit factor.
6. Registry/queue:
   - Updated `lead__validated__orderbook_core_support_cleared_plus_no_reclaim_short`.
   - Rebuilt registry and queue:
     - `100` registry leads.
     - `52` queued next-stage tasks.

# 2026-06-05 - Core orderbook invalidation-exit screen

1. Objective:
   - Test whether the core orderbook short block should exit early when the original short thesis starts disappearing.
2. Added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_core_invalidation_exit.py`
3. Direct-screen report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_core_invalidation_exit_20260605_core_orderbook_invalidation_exit.md`
4. What was tested:
   - Support-cleared short:
     - exit when support reclaim/rebuild warnings appear,
     - exit when bearish pressure fades,
     - combined reclaim-or-pressure-fade exits.
   - Strict no-reclaim VAL/vacuum short:
     - exit when support reclaim/rebuild warnings appear,
     - exit when breakdown-failure warnings appear,
     - combined reclaim-or-breakdown-failure exits.
5. Result:
   - Best-ranked profile for both core legs remained the existing fixed tailored exit:
     - support-cleared: `18h` hold, `2.5%` stop, `5%` target.
     - strict no-reclaim: `8h` hold, `1.0%` stop, `3.5%` target.
   - Selected direct-screen result:
     - `27` trades.
     - `70.4%` win rate.
     - `33.8%` direct-harness compounded return.
     - `4.5%` drawdown.
     - `3.66` profit factor.
   - Direct-harness figures are not a replacement for the Freqtrade validation result; the Freqtrade validated baseline remains `19.22%` return, `3.50%` drawdown, and `2.46` profit factor.
6. Interpretation:
   - Generic reclaim/rebuild/pressure-fade exits did not improve the main exit.
   - Some reclaim-warning exits reduced drawdown but gave up too much profit, so they are risk-protection candidates rather than promoted exits.
7. Next action:
   - Keep the current tailored-exit Freqtrade strategy as the promoted orderbook-only short block.
   - Do not Freqtrade-validate the invalidation exits unless a later, more specific invalidation trigger beats the fixed baseline in direct tests.

# 2026-06-05 - Core orderbook risk-sizing screen

1. Objective:
   - Test whether additional metrics can size the promoted core orderbook short trades up/down.
2. Updated:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
   - Fixed the orderbook overlay to use current columns:
     - support removed/cleared,
     - downside vacuum,
     - bearish pressure,
     - reclaim/rebuild/bounce warnings.
3. Report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605_core_orderbook_risk_overlay.md`
4. Result:
   - Direct-harness baseline:
     - `27` trades.
     - `33.8%` compounded return.
     - `4.5%` drawdown.
     - `3.66` profit factor.
   - Watchlist overlays:
     - signal-score sizing:
       - `43.9%` direct-harness return.
       - `5.7%` drawdown.
       - `3.72` profit factor.
     - structure-agreement sizing:
       - `35.3%` direct-harness return.
       - `4.5%` drawdown.
       - `3.74` profit factor.
   - Rejected for now:
     - core orderbook quality sizing,
     - orderbook warning sizing,
     - directional volume sizing,
     - volatility risk-off sizing.
5. Interpretation:
   - Orderbook is currently useful for selecting the core short entries.
   - It is not yet proven as a position-size/leverage scaler for this block.
   - Score-based and structure-agreement sizing are watchlist ideas, but they need Freqtrade-compatible validation before any promotion.

# 2026-06-05 - Rectangle breakout VP-context isolated validation

1. Objective:
   - Validate the best long-side per-rule confluence clue without mixing it with the prior-day-low short filter.
2. Added:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleBreakoutVpConfluenceStrategy.py`
3. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_tailored_exit_selected_confluence.parquet`
4. Trader question:
   - When a rectangle/compression area breaks upward and VP/value-area context supports the move, does the breakout continue?
5. Freqtrade validation:
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockRectangleBreakoutVpConfluenceStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-30-19.zip`
   - Result:
     - `5` actual trades from `9` clustered signals.
     - `40.0%` win rate.
     - `7.97%` return.
     - `3.67%` max drawdown.
     - `2.73` profit factor.
     - Positive in both 2025 and 2026 year buckets.
6. Interpretation:
   - This is a profitable sparse long-side lead, not a complete standalone long block.
   - The confluence idea appears directionally useful, but max-open-trades reduced the 9 signal rows to 5 actual trades because the signals clustered.
7. Registry/queue:
   - Added:
     - `lead__validated__rectangle_breakout_vp_context_long`
   - Rebuilt registry:
     - `100` total leads.
     - `1` `freqtrade_validated_sparse_entry_rule`.
   - Rebuilt next-stage queue:
     - `53` tasks.
8. Next action:
   - Expand nearby rectangle/VP breakout variants and reduce clustering before exit or risk sizing.

# 2026-06-05 - Prior-day-low VP short confluence isolation

1. Objective:
   - Validate the prior-day-low VP short confluence candidates without mixing them with rectangle breakout longs.
2. Added:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockPriorDayLowVpShortVolumeStructureStrategy.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy.py`
3. Signal files:
   - Strict:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_prior_day_low_vp_short_strong_volume_structure.parquet`
   - Broader:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_prior_day_low_vp_short_volume_agrees.parquet`
4. Strict volume-plus-structure result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-33-32.zip`
   - Result:
     - `11` trades.
     - `45.5%` win rate.
     - `-3.57%` return.
     - `6.03%` drawdown.
     - `0.71` profit factor.
   - Verdict:
     - Reject/rework as currently defined.
5. Broader volume-agrees result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-33-45.zip`
   - Result:
     - `25` trades.
     - `60.0%` win rate.
     - `9.35%` return.
     - `8.62%` drawdown.
     - `1.44` profit factor.
     - Positive in both 2025 and 2026 year buckets.
6. Interpretation:
   - More confluence was not better here.
   - The broader volume-agreement short worked; the stricter volume-plus-structure filter removed useful winners and left too many stop losses.
7. Registry/queue:
   - Added:
     - `lead__validated__prior_day_low_vp_short_volume_agrees`
   - Rebuilt registry:
     - `100` total leads.
     - `4` `freqtrade_validated_confluence_entry_rule`.
   - Rebuilt next-stage queue:
     - `54` tasks.
8. Next action:
   - Run invalidation/risk-reduction research for this lead before adding it to a combined block because the current 2% stop caused `9` stop losses.
9. Exit research follow-up:
   - Fast-harness exit sweep selected:
     - `48h` hold.
     - `2.5%` stop.
     - `5.0%` target.
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_prior_day_low_vp_short_volume_agrees_exit_research.md`
   - Freqtrade validation strategy:
     - `TraderRuleBlockPriorDayLowVpShortVolumeAgreesTailoredExitStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-36-35.zip`
   - Freqtrade result:
     - `25` trades.
     - `52.0%` win rate.
     - `2.19%` return.
     - `10.40%` drawdown.
     - `1.08` profit factor.
   - Verdict:
     - Reject the wider-target tailored exit.
     - Keep the original `48h` hold, `2.0%` stop, `2.0%` target as current validated exit.

# 2026-06-05 - Multi-scenario lead block validation

1. Objective:
   - Start combining validated leads into a practical multi-scenario BTC trading block while preserving per-lead exits and stops.
2. Added:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockValidatedMultiScenarioStrategy.py`
3. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_multi_scenario_block.parquet`
4. Signal makeup:
   - `52` unique decision rows from `61` raw signals.
   - `9` exact overlaps removed by priority.
   - `43` short signals.
   - `9` long signals.
5. Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-41-53.zip`
6. Result:
   - `44` trades.
   - `63.6%` win rate.
   - `41.18%` return.
   - `9.81%` drawdown.
   - `2.26` profit factor.
7. Interpretation:
   - This is the first strong validated mixed long/short rule block.
   - It supports the revised objective path: build many trader-readable leads, test confluence, then tune exits and risk sizing per successful lead.
8. Next action:
   - Rework risk/conflict filters around the weaker legs.
   - Continue expanding the lead pool toward `50-100` candidates, but keep promotion gated by Freqtrade validation.

# 2026-06-06 - Goal cycle rectangle/VP and support-break lead expansion

1. Objective:
   - Continue building toward a broad pool of trader-readable BTC leads, then promote only those that survive Freqtrade validation.
2. Ran direct candidate expansion:
   - `trading_lead_rectangle_vp_expansion.py --tag 20260606_goal_cycle_rectangle_vp --min-trades 10 --execute`
   - `trading_lead_support_break_short_expansion.py --tag 20260606_goal_cycle_support_break --min-trades 10 --execute`
   - `trading_lead_orderbook_support_removal_short_refinement.py --tag 20260606_goal_cycle_orderbook_support --min-trades 10 --execute`
3. Direct-harness outputs:
   - Rectangle/VP:
     - `221` variants tested.
     - `9` selected.
     - selected-block direct return `16.27%`.
   - Support-break:
     - `160` variants tested.
     - `4` selected.
     - selected-block direct return `4.74%`.
   - Orderbook support-removal:
     - `118` variants tested.
     - `2` selected.
     - selected-block direct return `26.16%`.
4. Freqtrade validations added:
   - `TraderRuleBlockRectangleVpGoalCycleStrategy`
   - `TraderRuleBlockSupportBreakGoalCycleStrategy`
   - `TraderRuleBlockRectangleVpIsolatedGoalCycleStrategies`
5. Freqtrade results:
   - Rectangle/VP selected block:
     - `66` trades.
     - `56.1%` win rate.
     - `23.31%` return.
     - `9.80%` drawdown.
     - `1.51` profit factor.
   - Support-break selected block:
     - `92` trades.
     - `39.1%` win rate.
     - `-9.99%` return.
     - `16.69%` drawdown.
     - `0.87` profit factor.
6. New isolated rectangle leads promoted:
   - `lead__validated__rect_break_orderbook_accept_long`
     - `11` trades, `72.7%` win rate, `9.84%` return, `1.30%` drawdown, `4.41` profit factor.
   - `lead__validated__rect_squeeze_upper_release_long`
     - `10` trades, `80.0%` win rate, `9.42%` return, `1.31%` drawdown, `5.65` profit factor.
   - `lead__validated__rect_break_above_vah_long`
     - `20` trades, `60.0%` win rate, `8.95%` return, `1.69%` drawdown, `2.78` profit factor.
7. Rework/watchlist:
   - `lead__watchlist__rect_break_below_val_short_unstable`
     - `10` trades, `40.0%` win rate, `7.60%` return, `4.63%` drawdown, `2.13` profit factor.
     - Not promoted because 2025 failed and 2026 carried the result.
8. Registry state after update:
   - `105` total leads/candidates.
   - `12` validated entries/blocks.
9. Next action:
   - Check overlap between the three new rectangle longs and the existing mixed block.
   - Add only non-duplicative rectangle leads to the multi-scenario block.
   - Rework support-break shorts rather than promoting them.

# 2026-06-06 - Expanded multi-scenario block validation

1. Objective:
   - Test whether the new cleaner rectangle/VP long leads improve the existing validated mixed BTC rule block.
2. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_expanded_multi_scenario_block.parquet`
3. Overlap report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_expanded_multi_scenario_block_overlaps.csv`
4. Signal makeup:
   - `93` raw signals.
   - `75` unique decision rows after priority de-duplication.
   - `16` overlap rows.
   - `43` short signals.
   - `32` long signals.
5. Added strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockExpandedMultiScenarioStrategy.py`
6. Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-54-12.zip`
7. Result:
   - `53` trades.
   - `66.0%` win rate.
   - `40.71%` return.
   - `7.67%` drawdown.
   - `2.33` profit factor.
8. Comparison with previous mixed block:
   - Previous:
     - `44` trades.
     - `63.6%` win rate.
     - `41.18%` return.
     - `9.81%` drawdown.
     - `2.26` profit factor.
   - Expanded:
     - More trades.
     - Higher win rate.
     - Lower drawdown.
     - Slightly lower raw return.
9. Verdict:
   - Promote as current best quality mixed block candidate.
   - Not final: `rect_break_above_vah_long` was positive standalone but weak inside the merged block, so overlap/priority and risk filtering need refinement.

# 2026-06-06 - Refined squeeze-priority mixed block validation

1. Objective:
   - Test whether removing weak VAH-overlap rectangle rows and giving squeeze-release longs priority improves the mixed BTC rule block.
2. Signal files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah_overlaps.csv`
3. Added strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRefinedMultiScenarioStrategies.py`
4. Clean Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-59-39.zip`
5. Extracted trade CSV:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_refined_squeeze_priority_block.csv`
6. Result:
   - `50` trades.
   - `70.0%` win rate.
   - `43.94%` return.
   - `7.67%` max drawdown.
   - `2.56` profit factor.
7. Per-leg result:
   - Support-cleared short: `15` trades, `14.38%` return.
   - Rectangle squeeze-release long: `10` trades, `11.25%` return.
   - No-reclaim VAL/vacuum short: `9` trades, `8.70%` return.
   - Prior-day-low VP short: `15` trades, `8.23%` return.
   - Rectangle/orderbook-acceptance long: `1` trade, `1.37%` return.
8. Comparison:
   - Previous mixed block: `44` trades, `63.6%` win rate, `41.18%` return, `9.81%` drawdown, `2.26` profit factor.
   - Expanded mixed block: `53` trades, `66.0%` win rate, `40.71%` return, `7.67%` drawdown, `2.33` profit factor.
   - Refined squeeze-priority block: `50` trades, `70.0%` win rate, `43.94%` return, `7.67%` drawdown, `2.56` profit factor.
9. Verdict:
   - Promote as the current best mixed BTC rule-block candidate.
   - Plain English: removing the weak VAH rectangle overlap improved quality and return, while the squeeze-release long survived as a useful long-side scenario.
10. Next action:
   - Continue toward the user's real target of `50` to `100` promising leads.
   - Do not over-focus on this block; park it as the current best candidate, then add more independent leads and later test confluence, exits, and risk sizing per entry family.

# 2026-06-06 - MTF custom-indicator lead mining

1. Objective:
   - Broaden the lead pool using custom indicator logic across 1h, 4h, and 1d instead of over-focusing on the current best mixed block.
2. Added script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_mtf_indicator_mining.py`
3. What it tested:
   - `96` trader-readable rule families.
   - `480` threshold variants.
   - Source blocks:
     - VP: POC/VAH/VAL reclaim, rejection, acceptance, LVN travel.
     - TLV2: resistance/support breaks and retests.
     - Market structure: BOS/CHoCH continuation and reversal.
     - Pattern geometry: triangle, wedge, compression, rectangle breaks/reclaims/rejections.
4. Direct-mining output:
   - `41` selected fast-harness candidates.
   - Broad selected block had `372` direct-simulation trades, `48.7%` win rate, `58.3%` return, `16.7%` max drawdown, and `1.257` profit factor.
   - This broad block is not promoted; it is too noisy and is only useful as a lead source.
5. Output files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining.md`
6. Registry update:
   - `trading_lead_registry.py` now loads the MTF selected candidates as `direct_rule_mining_candidate` rows.
   - Latest registry has `100` leads and `572` confluence checks.
   - It includes `26` remaining direct MTF mining candidates after one candidate was promoted to Freqtrade validation.
7. First Freqtrade validation:
   - Lead:
     - `lead__validated__mtf_4h_tlv2_res_break_long`
   - Trader question:
     - When price breaks 4h TLV2 resistance with bullish pressure, does the breakout continue?
   - Signal:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_mtf_4h_tlv2_res_break_long_thr_045.parquet`
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-12-18.zip`
   - Extracted trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_4h_tlv2_res_break_long.csv`
8. Freqtrade result:
   - `50` trades.
   - `68.0%` win rate.
   - `29.24%` return.
   - `4.57%` max drawdown.
   - `2.99` profit factor.
   - Positive in both 2025 and 2026 annual buckets.
9. Verdict:
   - Promote as a strong long-side structure/volume lead.
   - Plain English: a scored 4h TLV2 resistance break with bullish pressure is a useful standalone breakout-following setup in the current clean feature window.
10. Next action:
   - Validate the next strongest MTF candidates one at a time, especially:
     - 1d TLV2 resistance retest long.
     - 4h triangle upper-break long.
     - 4h VAL acceptance short.
     - 1d rectangle lower-break short.
     - 1d TLV2 support retest short.
   - Then test confluence against orderbook resistance removal, volume persistence, VP acceptance, and regime filters.

# 2026-06-06 - MTF candidate validation batch

1. Objective:
   - Validate the next strongest direct-mined MTF candidates through Freqtrade before treating them as usable leads.
2. Added isolated strategy classes:
   - `TraderRuleBlockMtf1dTlv2ResRetestLongStrategy`
   - `TraderRuleBlockMtf4hTriangleUpperBreakLongStrategy`
   - `TraderRuleBlockMtf4hVpValAcceptShortStrategy`
   - `TraderRuleBlockMtf1dRectangleLowerBreakShortStrategy`
   - `TraderRuleBlockMtf1dTlv2SupRetestShortStrategy`
3. Promoted clean lead:
   - `lead__validated__mtf_1d_tlv2_res_retest_long`
   - Result:
     - `20` trades.
     - `75.0%` win rate.
     - `18.72%` return.
     - `2.05%` max drawdown.
     - `4.11` profit factor.
   - Plain English:
     - A daily TLV2 resistance break followed by a successful retest is a clean long-side continuation lead.
4. Promoted sparse lead:
   - `lead__validated__mtf_4h_triangle_upper_break_long`
   - Result:
     - `12` trades.
     - `83.3%` win rate.
     - `13.21%` return.
     - `1.97%` max drawdown.
     - `5.30` profit factor.
   - Plain English:
     - A 4h triangle upper break with volume is promising, but sample size is sparse.
5. Promoted filter-needed lead:
   - `lead__validated__mtf_4h_vp_val_accept_short`
   - Result:
     - `49` trades.
     - `55.1%` win rate.
     - `20.40%` return.
     - `7.02%` max drawdown.
     - `1.66` profit factor.
   - Plain English:
     - Accepting below 4h VAL is useful enough to keep, but it needs orderbook/support-reclaim filters before mixed-block use.
6. Parked/rework:
   - `lead__watchlist__mtf_1d_rectangle_lower_break_short_unstable`
     - `11` trades, `9.33%` return, `2.81` profit factor, but 2025 was negative and 2026 carried the result.
   - `lead__rework__mtf_1d_tlv2_sup_retest_short_weak`
     - `37` trades, `5.94%` return, `8.64%` drawdown, `1.25` profit factor, and 2025 was negative.
7. Registry state:
   - `100` leads in latest registry.
   - `20` validated/rework entry records.
   - `24` remaining direct MTF mining candidates.
8. Next action:
   - Test overlap between the new MTF longs and existing rectangle/squeeze longs.
   - Run per-lead confluence for:
     - 4h TLV2 resistance break long.
     - 1d TLV2 resistance retest long.
     - 4h triangle upper-break long.
     - 4h VP VAL acceptance short.
   - Do not promote the two weak/unstable shorts until reworked.

# 2026-06-06 - MTF per-rule confluence validation

1. Objective:
   - Check whether the newly validated MTF leads improve when each lead is filtered by its own matching structure, volume, or orderbook confluence condition.
2. Fixed issue before trusting results:
   - `trading_lead_per_rule_confluence.py` was dropping duplicate signal timestamps by `date` only.
   - This incorrectly removed same-hour signals from different MTF rules.
   - Fixed by deduplicating combined signal files by `date + rule_id` when `rule_id` exists.
3. Confluence screen:
   - Input signals after fix: `179`.
   - Selected filter rows: `10`.
   - Strongest plain-English filters:
     - 4h TLV2 resistance-break long improved when orderbook direction and custom structure agreed.
     - 4h TLV2 resistance-break long also improved when strong volume and custom structure agreed.
     - 4h VP VAL acceptance short improved when custom structure agreed and when support-break pressure persisted.
     - Daily lower-rectangle break short improved when there was no strong support-reclaim warning.
4. Freqtrade validation:
   - Added:
     - `TraderRuleBlockMtfValidatedConfluenceStrategy`
     - `TraderRuleBlockMtfBestPerRuleConfluenceStrategy`
   - Broader selected-confluence block:
     - `59` trades.
     - `59.3%` win rate.
     - `3.26%` return.
     - `0.42%` max account drawdown.
     - `2.25` profit factor.
   - Best-filter-per-rule block:
     - `50` trades.
     - `62.0%` win rate.
     - `3.17%` return.
     - `0.39%` max account drawdown.
     - `2.66` profit factor.
5. Registry:
   - Added `lead__validated__mtf_best_per_rule_confluence_block`.
   - Validated/rework entry records now: `21`.
   - Latest registry remains capped at `100` leads.
   - New confluence plan contains `572` checks.
6. Plain-English conclusion:
   - Confluence filtering did improve trade cleanliness.
   - It did not increase raw return yet, but it kept nearly the same return with fewer trades, higher win rate, higher profit factor, and lower drawdown.
7. Next action:
   - Compare this MTF confluence block against the current refined squeeze-priority block.
   - Start exit research per successful lead instead of using the same crude fixed target/time exit everywhere.
   - Add risk/leverage overlays only after the entry and exit behaviour are stable.

# 2026-06-06 - Tailored exit validation

1. Objective:
   - Start the next objective layer after entry/confluence validation: test whether successful entry blocks improve when exits are tailored to each entry story.
2. Fixed issue before testing:
   - `trading_lead_exit_research.py` also dropped duplicate timestamps by `date` only.
   - Fixed to deduplicate combined signal files by `date + rule_id` when `rule_id` exists.
3. Added MTF-specific exit grids:
   - 4h TLV2 resistance-break long.
   - Daily TLV2 resistance-retest long.
   - 4h triangle upper-break long.
   - 4h VAL acceptance short.
   - Daily rectangle lower-break short.
   - Daily TLV2 support-retest short.
4. Added Freqtrade validation classes:
   - `TraderRuleBlockMtfBestPerRuleTailoredExitStrategy`
   - `TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy`
5. MTF best-per-rule tailored-exit result:
   - `38` trades.
   - `52.6%` win rate.
   - `3.49%` return.
   - `0.48%` max account drawdown.
   - `3.08` profit factor.
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\mtf_tailored_20260606\backtest-result-2026-06-06_00-33-09.zip`
6. Refined squeeze-priority tailored-exit result:
   - `44` trades.
   - `75.0%` win rate.
   - `4.56%` return.
   - `0.58%` max account drawdown.
   - `3.24` profit factor.
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\refined_tailored_20260606\backtest-result-2026-06-06_00-33-22.zip`
7. Plain-English conclusion:
   - Tailored exits improved the refined squeeze-priority block: fewer trades, higher win rate, higher profit factor, and slightly higher return than the prior refined block.
   - Tailored exits also helped the MTF block's profit factor and return, but exposed a weak component: the 4h VAL acceptance short lost money under the selected tight-stop exit.
8. Registry:
   - Added:
     - `lead__validated__mtf_best_per_rule_tailored_exit_block`
     - `lead__validated__refined_squeeze_priority_tailored_exit_block`
   - Validated/rework entry records now: `23`.
   - Latest registry remains capped at `100` leads.
9. Next action:
   - Promote the refined squeeze-priority tailored-exit block as the current best quality block.
   - Rework or filter the MTF 4h VAL acceptance short before merging MTF block entries.
   - Start risk overlay tests only on stable entry+exit blocks.

# 2026-06-06 - First risk overlay screen on tailored-exit blocks

1. Objective:
   - Start the final objective layer cautiously: test whether extra metrics can size trades up/down after entries and exits are fixed.
2. Scope:
   - This is a first screen, not final leverage logic.
   - Inputs were the tailored-exit trade CSVs from the exit research tool, not live strategy code.
3. Refined squeeze-priority tailored-exit risk screen:
   - Trade file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_refined_squeeze_priority_exit_sweep_combined_trades.csv`
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_refined_tailored_risk_overlay.csv`
   - Baseline equal-size simulated result:
     - `40` trades.
     - `70.0%` win rate.
     - `63.57%` compounded trade-return in the exit simulator.
     - `5.77%` max simulated drawdown.
     - `3.32` profit factor.
   - Watchlist overlays:
     - Entry-score confidence sizing.
     - Conservative combined sizing.
     - Directional volume sizing.
     - Structure agreement sizing.
   - Rejected for now:
     - Current orderbook warning and core-orderbook sizing formulas reduced return.
4. MTF tailored-exit risk screen:
   - Trade file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_mtf_best_confluence_exit_sweep_combined_trades.csv`
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_mtf_tailored_risk_overlay.csv`
   - Baseline equal-size simulated result:
     - `35` trades.
     - `68.6%` win rate.
     - `94.02%` compounded trade-return in the exit simulator.
     - `2.61%` max simulated drawdown.
     - `6.83` profit factor.
   - Watchlist overlays:
     - Entry-score confidence sizing.
     - Directional volume sizing.
     - Volatility risk-off sizing.
     - Structure agreement sizing.
     - Conservative combined sizing.
   - Rejected for now:
     - Current orderbook warning and core-orderbook formulas reduced return.
5. Plain-English conclusion:
   - Sizing up higher-confidence, volume-confirmed, and structure-confirmed trades may help.
   - The current orderbook sizing formulas are too blunt; they reduce exposure but do not improve the result.
6. Next action:
   - Rework orderbook risk overlays so they are per-entry-story instead of generic.
   - Validate any chosen risk overlay in Freqtrade or a closer equivalent before treating it as leverage logic.

# 2026-06-06 - Story-specific risk overlay screen

1. Objective:
   - Replace the too-generic orderbook sizing idea with per-entry-story risk overlays.
2. Added overlays:
   - `story_specific_orderbook_size`
     - Applies support-reclaim warnings to support-break shorts.
     - Applies resistance/rejection warnings to breakout longs.
     - Applies stricter support-reclaim treatment to VAL acceptance shorts.
   - `story_specific_structure_volume_size`
     - Sizes up breakout longs when bullish structure and volume agree.
     - Sizes up breakdown shorts when bearish structure and volume agree.
     - Sizes down when structure conflicts.
   - `story_specific_conservative_size`
     - Blends entry confidence, story-specific structure/volume, story-specific orderbook, and volatility.
3. Refined tailored block result:
   - Best story-specific result:
     - `story_specific_structure_volume_size`
     - `40` trades.
     - `73.75%` simulated compounded return versus `63.57%` baseline.
     - `6.92%` simulated drawdown versus `5.77%` baseline.
     - `3.20` profit factor versus `3.32` baseline.
   - Verdict:
     - Watchlist, but not clearly better than entry-score sizing.
4. MTF tailored block result:
   - Best story-specific result:
     - `story_specific_structure_volume_size`
     - `35` trades.
     - `115.87%` simulated compounded return versus `94.02%` baseline.
     - `3.21%` simulated drawdown versus `2.61%` baseline.
     - `7.30` profit factor versus `6.83` baseline.
   - Verdict:
     - Strong watchlist for MTF sizing.
5. Rejected for now:
   - `story_specific_orderbook_size`
   - Reason:
     - It reduced return on both tested blocks despite lowering drawdown.
     - Plain English: the current orderbook risk rule is too defensive and cuts too many good trades.
6. Next action:
   - Keep structure/volume agreement as the best risk-sizing branch.
   - Rework orderbook sizing into separate warning-only rules:
     - skip/reduce only when support reclaim or resistance rejection is extreme.
     - avoid broad orderbook reductions on ordinary uncertainty.

# 2026-06-06 - Extreme-only orderbook risk overlay screen

1. Objective:
   - Test whether orderbook is more useful as a sharp invalidation guard than as a broad size reducer.
2. Added overlays:
   - `extreme_orderbook_invalidation_guard`
     - Reduces size only when a support-break short shows strong support reclaim/rebuild or a breakout long shows strong resistance rejection/failure.
   - `structure_volume_with_extreme_ob_guard`
     - Uses story-specific structure/volume sizing, then applies the extreme orderbook guard.
3. Refined tailored block:
   - `extreme_orderbook_invalidation_guard`:
     - `62.36%` simulated return versus `63.57%` baseline.
     - Same drawdown as baseline.
     - Reject for now.
   - `structure_volume_with_extreme_ob_guard`:
     - `72.16%` simulated return versus `63.57%` baseline.
     - `6.92%` drawdown versus `5.77%` baseline.
     - Watchlist, but weaker than structure/volume sizing alone.
4. MTF tailored block:
   - `extreme_orderbook_invalidation_guard`:
     - `84.07%` simulated return versus `94.02%` baseline.
     - `2.32%` drawdown versus `2.61%` baseline.
     - Reject for now because return loss is too high.
   - `structure_volume_with_extreme_ob_guard`:
     - `102.35%` simulated return versus `94.02%` baseline.
     - `2.78%` drawdown versus `2.61%` baseline.
     - `7.55` profit factor versus `6.83` baseline.
     - Watchlist. Better risk balance than plain structure/volume sizing, but lower return.
5. Plain-English conclusion:
   - Orderbook invalidation is not a standalone size-up/down answer yet.
   - It may be useful as a safety brake when combined with structure/volume sizing, especially on MTF entries.
6. Next action:
   - For risk sizing, prioritize:
     - structure/volume sizing for return seeking.
     - structure/volume plus extreme orderbook guard for risk-balanced sizing.
   - Next implementation should test these two as actual strategy variants or closer Freqtrade-equivalent equity curves.

# 2026-06-06 - Strategy-level risk sizing validation

1. Objective:
   - Move the best risk overlays from CSV simulation into Freqtrade strategy-level validation.
2. Implemented:
   - Risk-adjusted signal exports with `risk_multiplier`.
   - Research-only strategy variants using `custom_stake_amount`.
   - Strategy classes for refined and MTF story-specific structure/volume sizing.
   - Strategy classes for refined and MTF structure/volume sizing plus extreme orderbook guard.
3. Important validation detail:
   - Small BTC futures stake sizes hide risk multipliers because 0.001 BTC contract rounding collapses 65/100/120 USDT stakes into nearly the same real stake.
   - Valid risk-sizing comparisons must use a larger fixed stake/wallet or another method that avoids contract-size rounding.
4. Fixed-size Freqtrade validation:
   - Test setup:
     - `stake_amount=1000`
     - `dry_run_wallet=10000`
     - BTC/USDT:USDT only
     - Timerange `20250501-20260523`
   - Refined tailored baseline:
     - `44` trades, `75.0%` win, `4.51%` return, `0.86%` drawdown, `2.87` profit factor.
   - Refined story-specific structure/volume sizing:
     - `44` trades, `75.0%` win, `4.85%` return, `1.04%` drawdown, `2.72` profit factor.
     - Verdict: watchlist for return seeking, but drawdown and profit factor worsened.
   - Refined structure/volume plus extreme orderbook guard:
     - `44` trades, `75.0%` win, `4.58%` return, `1.04%` drawdown, `2.63` profit factor.
     - Verdict: reject for now versus pure structure/volume sizing.
   - MTF tailored baseline:
     - `38` trades, `52.6%` win, `3.89%` return, `0.38%` drawdown, `3.34` profit factor.
   - MTF story-specific structure/volume sizing:
     - `38` trades, `52.6%` win, `4.18%` return, `0.36%` drawdown, `3.39` profit factor.
     - Verdict: strongest risk-sizing candidate so far because return improved and drawdown fell slightly.
   - MTF structure/volume plus extreme orderbook guard:
     - Produced only `9` trades and `-0.02%` return despite a 53-row signal file.
     - Verdict: invalid/parked until the missing-entry behaviour is diagnosed.
5. Next action:
   - Carry forward MTF story-specific structure/volume sizing.
   - Keep refined structure/volume sizing as a return-seeking watchlist only.
   - Rework orderbook guard logic before using it in strategy-level sizing.

# 2026-06-06 - Sieve runtime lead borrowing added

1. Objective:
   - Move toward the user's real target of `50-100` trader-readable BTC leads by borrowing plausible good trades from existing entry-sieve evidence.
2. Implemented:
   - Added `trader_sieve_runtime_candidate_leads.py`.
   - The scanner reads completed entry-sieve JSONL result files and exports only positive usable candidate seeds.
   - It excludes failed runs and one-off curiosities by requiring:
     - completed `ok` status
     - at least `5` trades
     - positive return
     - profit factor at least `1.20`, or win rate at least `55%`
   - Wired the output into `trading_lead_registry.py` as a normal registry input source.
3. Outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260606_sieve_runtime_update.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_plan_20260606_sieve_runtime_update.csv`
4. Result:
   - Sieve scanner found `400` usable candidate rows after strategy-level dedupe.
   - Rebuilt registry contains `100` leads.
   - Registry now includes `25` sieve-runtime candidates, `28` structure/volume leads, `18` downside-risk leads, `18` orderbook-state leads, `6` multi-scenario blocks, and `5` regime-confluence leads.
   - Confluence plan now contains `572` planned checks.
5. Important interpretation:
   - Sieve-runtime rows are not final trading rules.
   - They are lead seeds that must still move through direct retest, confluence filtering, tailored exits, and risk/leverage overlays.
   - Some have low win rate but positive return/profit factor, so they may be useful trend/continuation leads but need trader-readable filtering before promotion.
6. Next action:
   - Run confluence checks against the strongest borrowed sieve leads first, especially:
     - BOS bullish continuation.
     - TLV2/VP resistance break longs.
     - TLV2 support breakdown shorts.
     - Rectangle breakdown shorts.
     - Triangle squeeze breakdown/upper-break leads.
   - Convert only the survivors into tailored-exit research.

# 2026-06-06 - BTC sieve confluence testing started

1. Objective:
   - Move from borrowed sieve lead seeds to actual BTC confluence testing.
2. Fixes/improvements:
   - Updated `trader_sieve_runtime_candidate_leads.py` to score borrowed sieve candidates from actual BTC/USDT futures trades inside backtest ZIP archives.
   - Added `--pair-required` so multi-pair-only winners are excluded when BTC evidence is required.
   - Added `trader_sieve_runtime_signal_exports.py` to export BTC decision-hour trades/signals from the selected sieve backtest archives.
   - Fixed a verdict bug in `trading_lead_confluence_checks.py` where `worth_more_testing` rows were overwritten as `watchlist`.
3. BTC-only candidate result:
   - `141` usable BTC-specific sieve candidate strategies.
   - Registry rebuilt to `100` leads with `25` BTC-specific sieve-runtime candidates.
4. Signal export:
   - Top `30` BTC sieve candidates exported.
   - `1,459` BTC trade rows.
   - `1,459` signal rows.
   - Signal alignment:
     - `signal_date` is `open_date - 1h`, so confluence features describe the hour before the actual backtest entry.
5. Confluence test result:
   - Input trades: `1,459`.
   - Result rows: `868`.
   - Verdict counts:
     - `86` worth more testing.
     - `47` watchlist.
     - `276` too few trades.
     - `428` reject for now.
     - `31` baselines.
6. Plain-English early findings:
   - Across all selected borrowed sieve entries, custom structure confirmation improved quality but cut trade count heavily.
   - Range-break, regime-plus-range-break, VP-plus-range-break, and compression-plus-range-break improved broad quality while keeping hundreds of trades.
   - Generic orderbook-only filters hurt the broad set, but orderbook helped a few specific rule families.
   - Some small-sample rows show huge profit factor because there were no losing trades after the filter; treat those as promising but not proven.
7. Next-stage queue:
   - Created `64` next tasks:
     - `12` tailored confluence retests.
     - `16` exit research tasks.
     - `36` per-lead confluence tasks.
8. Outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606_btc_only.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_btc_top30.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260606_btc_sieve_top30_confluence_fixed.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_next_stage_queue_20260606_btc_sieve_confluence_fixed.md`
9. Next action:
   - Retest the strongest broad filters as actual signal exports:
     - compression plus range break
     - regime plus range break
     - VP plus range break
     - structure trigger agrees
   - Then run tailored exits on the surviving filtered lead families.

# 2026-06-06 - BTC sieve compression/range branch moved to Freqtrade validation

1. Objective:
   - Advance one promising borrowed-sieve branch through the intended sequence:
     - BTC-only candidate scan.
     - Confluence filter.
     - Tailored exit sweep.
     - Normal Freqtrade validation.
2. Filtered signal export:
   - Exported confluence-filtered signal parquets from the top `30` BTC sieve candidates.
   - Strongest broad filter chosen first:
     - `compression_plus_range_break`
     - `683` signal rows before exit-rule filtering.
3. Exit research:
   - Ran `trading_lead_exit_research.py` on `compression_plus_range_break`.
   - Direct harness result:
     - `237` combined trades.
     - `66.67%` win rate.
     - `1.571%` average return per trade.
     - `3.6903` profit factor.
     - `-8.34%` max drawdown.
   - Caveat:
     - The shell command timed out after printing/writing complete outputs, so treat files as generated but keep the timeout note.
     - Direct-harness results are not final because they do not exactly match Freqtrade trade management.
4. Validation issue found and fixed:
   - First Freqtrade validation pointed at all `683` filtered signals.
   - Exit research had selected exits for only `21` of the `30` rule families.
   - Resulting problem:
     - An unhandled rule family opened a trade and held until forced exit, polluting the result.
   - Fix:
     - Exported selected-exit-rules-only signal parquet:
       - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_selected_exit_rules.parquet`
       - `653` signal rows.
       - `21` rule families.
     - Added research-only strategy:
       - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveCompressionRangeStrategy.py`
5. Corrected Freqtrade validation:
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockBtcSieveCompressionRangeStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20200101-20260115 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Result:
     - `270` trades.
     - `44.8%` win rate.
     - `0.52%` average profit per trade.
     - `264.54%` total return.
     - `1.40` profit factor.
     - `27.55%` max drawdown.
     - `121` winners and `149` losers.
     - `230` long trades and `40` short trades.
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-24-54.zip`
6. Plain-English interpretation:
   - This branch is genuinely useful enough to keep, but not production-ready.
   - The direct harness overstated quality because Freqtrade execution, overlapping signals, stop handling, and max-open-trade behaviour change the realised trade set.
   - The real Freqtrade result still shows a profitable multi-year BTC block, but win rate and drawdown need improvement before live consideration.
7. Next action:
   - Repeat exit research for the other broad confluence filters:
     - `regime_plus_range_break`
     - `range_break_agrees`
     - `vp_plus_range_break`
   - Compare the Freqtrade-validated survivors.
   - Then build risk/leverage overlays from structure/volume first, with orderbook reserved for story-specific invalidation.

# 2026-06-06 - BTC sieve broad branch comparison and first quality-family refinement

1. Objective:
   - Avoid over-focusing on the first working branch by testing nearby broad confluence alternatives and a cleaner sub-family refinement.
2. Added reusable validation infrastructure:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
   - Purpose:
     - Load pre-filtered signal parquets.
     - Load per-rule exit settings from the matching exit-research selected CSV.
     - Validate branches in Freqtrade without hand-copying large exit dictionaries.
3. Exit research direct-harness results:
   - `compression_plus_range_break`:
     - `237` trades, `66.67%` win rate, `1.571%` average return, `3.690` profit factor, `-8.34%` drawdown.
   - `regime_plus_range_break`:
     - `280` trades, `68.93%` win rate, `1.450%` average return, `2.889` profit factor, `-10.33%` drawdown.
   - `range_break_agrees`:
     - `280` trades, `68.93%` win rate, `1.448%` average return, `2.887` profit factor, `-10.33%` drawdown.
   - `vp_plus_range_break`:
     - `270` trades, `69.26%` win rate, `1.533%` average return, `3.076` profit factor, `-10.33%` drawdown.
4. Freqtrade validation results:
   - `compression_plus_range_break`:
     - `270` trades, `264.54%` return, `44.8%` win rate, `1.40` profit factor, `27.55%` drawdown.
   - `regime_plus_range_break`:
     - `329` trades, `269.76%` return, `48.6%` win rate, `1.31` profit factor, `29.06%` drawdown.
   - `range_break_agrees`:
     - `330` trades, `245.46%` return, `48.2%` win rate, `1.28` profit factor, `29.32%` drawdown.
   - `vp_plus_range_break`:
     - `317` trades, `247.60%` return, `47.6%` win rate, `1.30` profit factor, `26.14%` drawdown.
5. Quality-family refinement:
   - Built from compression/range Freqtrade output.
   - Kept `11` rule families that had:
     - at least `5` Freqtrade trades,
     - positive total contribution,
     - positive average return.
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet`
   - Freqtrade result:
     - `227` trades.
     - `371.08%` return.
     - `46.7%` win rate.
     - `1.56` profit factor.
     - `20.46%` drawdown.
6. Plain-English interpretation:
   - The best current branch is not the broadest branch.
   - Pruning weak rule families materially improved the system:
     - fewer trades,
     - higher return,
     - lower drawdown,
     - better profit factor.
   - This is now a serious lead block for further confluence/risk work.
7. Storage check:
   - `user_data\backtest_results`: about `16.7 MB`.
   - `user_data\models`: about `28.34 GB`.
   - `user_data\research_news_data\context_features\reports`: about `395 MB`.
   - Largest model folders:
     - `context-freqai-structure-lightgbm-20260520`: about `3.526 GB`.
     - `context-freqai-event-context-lgbm-20260521`: about `3.310 GB`.
     - `context-freqai-event-combined-lgbm-20260521`: about `3.310 GB`.
     - `context-freqai-event-price-lgbm-20260521`: about `3.309 GB`.
   - No deletion performed.
8. Next action:
   - Build risk/leverage overlays for the quality-family branch.
   - Start with structure/volume confirmation because previous risk work showed it was more useful than broad orderbook sizing.
   - Use orderbook only as a story-specific invalidation or caution brake, not as a generic size reducer.

# 2026-06-06 - BTC sieve quality-family risk overlay validation

1. Objective:
   - Keep the current best BTC lead block unchanged, then test whether trader-story sizing and orderbook invalidation improve realised Freqtrade results.
2. Added/updated:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
     - Added BTC-sieve-specific sizing logic for:
       - resistance breakouts,
       - support reclaims,
       - BOS bullish continuation,
       - LVN fast-traverse longs,
       - POC rejection shorts,
       - double-top shorts.
     - Added an orderbook guard that only adjusts size when orderbook evidence strongly agrees with or invalidates the specific rule direction.
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_signal_exports.py`
     - Exported risk-sized signal files for the quality-family block.
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
     - Added Freqtrade validation classes for the two risk-sized signal variants.
3. Export check:
   - Story-only risk file:
     - `361` signal rows.
     - `36` rows changed size.
     - average multiplier `1.0105`, min `0.65`, max `1.30`.
   - Story plus orderbook guard:
     - `361` signal rows.
     - `77` rows changed size.
     - average multiplier `0.9534`, min `0.40`, max `1.45`.
4. Fixed-stake Freqtrade baseline:
   - Strategy:
     - `TraderRuleBlockBtcSieveCompressionRangeQualityFamiliesStrategy`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-47-42.zip`
   - Result:
     - `227` trades.
     - `16.35%` return on fixed `1000 USDT` stake tests.
     - `46.7%` win rate.
     - `1.76` profit factor.
     - `1.83%` max account drawdown.
5. Story-only risk validation:
   - Strategy:
     - `TraderRuleBlockBtcSieveQualityStoryRiskStrategy`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-48-02.zip`
   - Result:
     - `227` trades.
     - `16.02%` return.
     - `46.7%` win rate.
     - `1.74` profit factor.
     - `1.81%` max account drawdown.
6. Story plus orderbook guard validation:
   - Strategy:
     - `TraderRuleBlockBtcSieveQualityStoryObGuardRiskStrategy`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-48-21.zip`
   - Result:
     - `227` trades.
     - `14.98%` return.
     - `46.7%` win rate.
     - `1.73` profit factor.
     - `2.02%` max account drawdown.
7. Plain-English interpretation:
   - The current entry/exit block remains useful.
   - The first risk-sizing overlays did not improve it.
   - Story-only sizing was close but slightly worse.
   - The orderbook guard cut too much good exposure or cut the wrong rows, so it should not be promoted as a leverage/risk layer yet.
8. Next action:
   - Do not discard the lead block.
   - Rework risk/leverage as a second-stage problem:
     - inspect winners/losers by retained rule family,
     - create rule-family-specific skip/reduce/size-up checks,
     - test skip filters separately from position sizing,
     - only then consider leverage multipliers.
   - Continue expanding toward `50-100` validated leads before trying to merge into a final multi-scenario rule block.

# 2026-06-06 - Objective scope update: drawdown exits and production alpha

1. User clarified that `20%+` drawdown is not acceptable as a final destination.
2. Canonical objectives were updated in:
   - `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md`
3. Added durable objectives for:
   - orderbook trade-invalidation exits,
   - orderbook crash/downturn-escalation exits,
   - volume failure exits,
   - structure level-failure exits,
   - volatility shock risk-off exits,
   - time-to-confirm exits,
   - multi-timeframe disagreement skips,
   - range/chop avoidance,
   - post-break acceptance checks,
   - support/resistance proximity targeting,
   - regime-specific exits,
   - drawdown-state throttle,
   - confidence-bucket sizing,
   - rule-family-specific risk,
   - later known-event and quiet-news modes when news/context quality is ready.
4. Added final objective:
   - turn validated research into a real production trade-alpha system for testing after evidence-based promotion gates are passed.
5. No implementation of those new objectives has been claimed yet.

# 2026-06-06 - First drawdown-management pass on BTC quality-family block

1. Objective:
   - Start testing the new drawdown-reduction objectives on the current best BTC quality-family lead block.
2. Added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_trade_management_research.py`
   - Purpose:
     - Keep the selected BTC lead block fixed.
     - Keep the selected per-rule exits fixed.
     - Test skip and early-exit concepts against the same signal set.
3. Direct-harness concepts tested:
   - orderbook trade-invalidation exit,
   - orderbook crash/downturn-escalation exit,
   - volume failure exit,
   - structure level-failure exit,
   - volatility shock risk-off exit,
   - time-to-confirm exit,
   - multi-timeframe disagreement skip,
   - range/chop avoidance skip,
   - post-break acceptance skip,
   - support/resistance proximity skip.
4. Direct-harness output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3.md`
5. Direct-harness finding:
   - `orderbook_crash_downturn_exit` was the cleanest candidate:
     - baseline direct harness: `221` trades, `3606.99%` compounded return, `-8.90%` drawdown, `3.789` profit factor.
     - orderbook crash exit: `242` trades, `3128.28%` compounded return, `-8.60%` drawdown, `4.155` profit factor.
   - Other early exits reduced drawdown more, but cut too much return and need rework rather than promotion.
6. Freqtrade validation added:
   - Strategy:
     - `TraderRuleBlockBtcSieveQualityOrderbookCrashExitStrategy`
   - File:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
   - Behaviour:
     - Same quality-family entries.
     - Same per-rule tailored exits.
     - Early long exit when orderbook/price state clusters into downside escalation.
7. Freqtrade result:
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_20-54-32.zip`
   - Compared with current quality-family baseline:
     - Baseline: `227` trades, `371.08%` return, `46.7%` win rate, `1.56` profit factor, `20.46%` drawdown.
     - Orderbook crash exit: `247` trades, `301.98%` return, `42.9%` win rate, `1.59` profit factor, `15.69%` drawdown.
8. Plain-English interpretation:
   - Orderbook is more useful as an exit/invalidation layer than as a broad entry filter.
   - The first crash-exit version reduced drawdown by about `4.77` percentage points and slightly improved profit factor.
   - It also gave up about `69.10` percentage points of return and reduced win rate.
9. Verdict:
   - Worth rework.
   - Not final production logic.
   - Next version should apply orderbook crash exits only to rule families where they actually help, because the broad long-exit rule cut some profitable continuation trades too early.
10. Sunflower note:
   - User requested inclusion of rare high-win-rate `complete sunflower` sieve entries.
   - Added to canonical objectives.
   - Current search did not find `sunflower` in the usual strategy/result metadata paths, so the relevant artifact still needs to be located before inclusion.

# 2026-06-06 - Selective orderbook crash-exit improvement

1. Objective:
   - Refine the broad orderbook crash exit so it only applies to rule families where the first Freqtrade audit showed benefit.
2. Family audit output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_crash_exit_family_audit_20260606.csv`
3. Harmed by broad crash exit:
   - `sieve2_multi2_vp_prior_month_high_break_vp_node_long`
   - `sieve1_bos_bull_continuation_long_8h`
   - `sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard`
   - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_1d`
4. Helped by broad crash exit:
   - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`
   - `sieve1_tlv2_resistance_breakout_long_4h`
   - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h`
   - `sieve1_vp_lvn_fast_traverse_long_1h`
5. Added Freqtrade strategy:
   - `TraderRuleBlockBtcSieveQualitySelectiveOrderbookCrashExitStrategy`
6. Freqtrade output:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_20-56-44.zip`
7. Result versus quality-family baseline:
   - Baseline:
     - `227` trades, `371.08%` return, `46.7%` win rate, `1.56` profit factor, `20.46%` drawdown.
   - Selective orderbook crash exit:
     - `233` trades, `426.75%` return, `46.8%` win rate, `1.75` profit factor, `14.88%` drawdown.
8. Plain-English interpretation:
   - This is the first drawdown-management layer that improved both return and drawdown in Freqtrade validation.
   - The key was not "use orderbook everywhere"; it was "use orderbook crash exits only where that trader story benefits from it."
9. Verdict:
   - Promote as current best BTC lead-block variant.
   - Still research-only, not production-alpha yet.
10. Next action:
   - Repeat the same rule-family-specific treatment for:
     - volume failure exits,
     - time-to-confirm exits,
     - volatility shock exits.
   - Locate the user-referenced `complete sunflower` sieve artifacts and add them as rare-entry lead candidates if evidence supports them.

# 2026-06-06 - Selective trade-management exit refinement

1. Objective:
   - Continue drawdown-management work from the current best BTC quality-family lead block.
   - Test only rule-family-specific trade-management exits rather than broad generic exits.
2. Added strategy variants in:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
3. Added/validated variants:
   - `TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy`
     - selective orderbook crash exit plus volume, structure, volatility, and time-to-confirm exits.
   - `TraderRuleBlockBtcSieveQualitySelectiveObVolumeExitStrategy`
   - `TraderRuleBlockBtcSieveQualitySelectiveObStructureExitStrategy`
   - `TraderRuleBlockBtcSieveQualitySelectiveObVolatilityExitStrategy`
   - `TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy`
   - `TraderRuleBlockBtcSieveQualitySelectiveObTimeStructureExitStrategy`
   - `TraderRuleBlockBtcSieveQualitySelectiveObTimeVolumeExitStrategy`
4. Generated comparison report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_selective_trade_management_freqtrade_comparison_20260606.csv`
5. Current comparison versus previous best selective orderbook crash exit:
   - Previous best:
     - `233` trades, `426.745%` return, `14.878%` drawdown, `1.7514` profit factor.
   - Selective orderbook plus time-to-confirm:
     - `233` trades, `447.648%` return, `14.471%` drawdown, `1.7959` profit factor.
   - Selective orderbook plus structure:
     - `239` trades, `383.836%` return, `13.718%` drawdown, `1.8244` profit factor.
   - Selective orderbook plus volume:
     - `238` trades, `391.576%` return, `14.681%` drawdown, `1.7967` profit factor.
   - Selective orderbook plus all tested trade-management exits:
     - `244` trades, `297.486%` return, `16.028%` drawdown, `1.7630` profit factor.
6. Plain-English interpretation:
   - The best improvement is not the broadest exit stack.
   - The time-to-confirm rule worked when applied only to selected rule families:
     - if the trade has not started working after about six hours and confirming evidence is absent, leave it.
   - Structure and volume exits reduced drawdown but cut too much return compared with the new best.
   - Combining too many exits damaged the strategy, confirming the objective that each entry family needs tailored management.
7. Verdict:
   - Promote `TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy` as the current best BTC quality-family research variant.
   - Keep structure and volume exits as rework candidates, not production candidates.
8. Sunflower/complete-sieve follow-up:
   - Filename search still did not find `sunflower`.
   - A likely related complete-sieve artifact was found:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\launcher_v2\runtime\entry_sieve\reports\sieve3_complete_folder_metrics_20260605.json`
   - It contains `sieve3_candidates` and `sieve2_complete_patterns`.
   - `sieve2_complete_patterns` has `22` strategy files, `2,278` total trades, and `55.27%` weighted average win rate in the complete-folder summary.
   - This should be used as a rare-entry expansion source, but it is not yet proven to be the user's referenced `complete sunflower`.
9. Candidate extraction:
   - Created:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_complete_pattern_candidates_20260606.csv`
   - Top complete-pattern rare-entry candidates by rough priority score:
     - `sieve2_geometry_wedge_breakout_long_1h`
     - `sieve2_geometry_ascending_channel_lower_bounce_long_8h`
     - `sieve2_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h`
     - `sieve2_multi2_tlv2_vp_res_break_vp_node_long_4h`
     - `sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h`
   - These are not yet promoted. They need the normal signal export, confluence, exit, and Freqtrade validation path.

# 2026-06-06 - Complete-pattern rare-entry validation and merge test

1. Objective:
   - Start including rare high-win-rate complete-pattern sieve entries as requested, while keeping them separate from the main BTC quality block until validated.
2. Created candidate lead file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_complete_pattern_candidate_leads_20260606.csv`
3. Exported decision-hour rare-entry signals:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_complete_pattern_rare_entries_signals.parquet`
   - `4` selected candidates.
   - `29` BTC signal rows.
4. Ran confluence checks:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260606_complete_pattern_rare_entries_confluence.csv`
   - Notable finding:
     - `compression_plus_range_break` improved all selected rare entries from `72.41%` win rate and `1.86%` average return to `83.33%` win rate and `2.48%` average return across `24` trades.
5. Ran tailored exit research:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_complete_pattern_rare_entries_exit_selected.csv`
   - Direct harness combined result:
     - `27` trades, `88.89%` win rate, `182.13%` compounded return, `1.92%` drawdown, `24.32` profit factor.
6. Added Freqtrade strategy:
   - `TraderRuleBlockBtcSieveCompletePatternRareEntriesStrategy`
7. Freqtrade validation:
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_21-22-24.zip`
   - Result:
     - `27` trades.
     - `73.20%` return.
     - `59.3%` win rate.
     - `4.54` profit factor.
     - `6.30%` drawdown.
8. Created merged signal/exit files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv`
9. Added merged Freqtrade strategy:
   - `TraderRuleBlockBtcSieveQualityPlusCompletePatternRareEntriesStrategy`
10. Merged Freqtrade validation:
    - Output:
      - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_21-23-22.zip`
    - Result:
      - `248` trades.
      - `555.26%` return.
      - `45.2%` win rate.
      - `1.80` profit factor.
      - `17.45%` drawdown.
11. Comparison report:
    - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_complete_pattern_merge_freqtrade_comparison_20260606.csv`
12. Plain-English interpretation:
    - The rare complete-pattern entries are useful as a small separate lead family.
    - Merging them into the current best quality block increased return by about `107.61` percentage points versus the quality-only best.
    - The merge also increased drawdown by about `2.98` percentage points, so it is a high-return candidate but not the current cleanest promoted variant.
13. Verdict:
    - Promote rare complete-pattern entries as a validated lead family.
    - Do not yet promote the merged strategy as final; it needs drawdown rework.

# 2026-06-06 - Rare complete-pattern confluence filtering and rare-priority merge test

1. Objective:
   - Check whether the rare complete-pattern entries become more useful when filtered by trader-readable compression/range-break context, and whether they can be merged into the current BTC quality block without hiding same-hour rare signals.
2. Built filtered rare-entry signal files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_compression_range.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_range_break.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_compression_release.parquet`
3. Built rare-priority merged signal files:
   - The research strategy keeps one signal per timestamp, so same-hour duplicates can hide one source.
   - The merge now gives rare complete-pattern entries priority over quality-family signals on same-hour conflicts.
4. Added Freqtrade strategy wrappers:
   - `TraderRuleBlockBtcSieveCompletePatternRareCompressionRangeStrategy`
   - `TraderRuleBlockBtcSieveQualityPlusRareCompressionRangePriorityStrategy`
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy`
   - `TraderRuleBlockBtcSieveQualityPlusRareAllPriorityStrategy`
5. Freqtrade validation results:
   - rare complete-pattern only:
     - `27` trades, `73.20%` return, `59.3%` win rate, `4.54` profit factor, `6.30%` drawdown.
   - rare complete-pattern plus compression/range context:
     - `22` trades, `77.66%` return, `68.2%` win rate, `7.05` profit factor, `4.08%` drawdown.
   - quality plus rare compression/range priority:
     - `244` trades, `589.17%` return, `45.5%` win rate, `1.86` profit factor, `18.26%` drawdown.
   - quality plus rare range-break priority:
     - `246` trades, `588.42%` return, `45.5%` win rate, `1.84` profit factor, `16.83%` drawdown.
   - quality plus all rare priority:
     - `248` trades, `571.38%` return, `45.2%` win rate, `1.81` profit factor, `16.83%` drawdown.
6. Comparison report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rare_priority_filter_merge_comparison_20260606.csv`
7. Plain-English interpretation:
   - The rare complete-pattern entries improve when they are aligned with compression/range-break context.
   - The standalone filtered rare block is better than the unfiltered rare block on return, win rate, profit factor, and drawdown, despite having fewer trades.
   - The merged variants increase total return versus the current clean best, but drawdown rises above the current clean best.
8. Verdict:
   - Promote `rare complete-pattern plus compression/range context` as a high-quality rare lead family.
   - Do not yet promote any merged rare-plus-quality block as the final system because drawdown remains too high.
   - Next work should test rare-family-specific exits and crash/downturn exits before accepting the merge.

# 2026-06-06 - Rare-specific exits and cleanup filters improve the merged candidate

1. Objective:
   - Test the user's assumption that rare complete-pattern entries should be included, but with dedicated exit/cleanup logic rather than a broad shared exit.
2. Added rare-specific exit variants:
   - rare time-to-confirm exit,
   - rare structure-failure exit,
   - rare time plus structure exit.
3. Rare-specific exit comparison:
   - baseline quality plus range-break rare priority:
     - `246` trades, `588.42%` return, `16.83%` drawdown, `1.84` profit factor.
   - rare time-to-confirm:
     - `247` trades, `558.02%` return, `20.09%` drawdown, `1.82` profit factor.
   - rare structure-failure:
     - `250` trades, `499.71%` return, `12.03%` drawdown, `1.96` profit factor.
   - rare time plus structure:
     - `251` trades, `465.73%` return, `16.38%` drawdown, `1.91` profit factor.
4. Plain-English interpretation:
   - Rare time-to-confirm is too aggressive or poorly aimed in this form.
   - Rare structure-failure is useful: it gives up some return but cuts drawdown materially.
5. Audit outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rare_specific_exit_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_family_audit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_year_audit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_day_audit_20260606.csv`
6. Family/day cleanup tests:
   - The audit showed one weak family:
     - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h`.
   - It also showed Sunday was negative, but skipping all Sunday entries removed too much good return.
7. Cleanup validation:
   - rare-structure candidate before cleanup:
     - `250` trades, `499.71%` return, `12.03%` drawdown, `1.96` profit factor.
   - no-Sunday filter:
     - `214` trades, `337.35%` return, `13.47%` drawdown, `1.87` profit factor.
   - no-weak-family filter:
     - `210` trades, `460.18%` return, `8.12%` drawdown, `2.14` profit factor.
   - no-Sunday plus no-weak-family:
     - `181` trades, `330.66%` return, `7.82%` drawdown, `2.12` profit factor.
8. Comparison report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_cleanup_filter_comparison_20260606.csv`
9. Verdict:
   - Promote `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoWeakFamilyRareStructureExitStrategy` as the current best BTC production-alpha research candidate.
   - It beats the previous clean best on both return and drawdown:
     - previous clean best: `447.65%` return, `14.47%` drawdown.
     - new best: `460.18%` return, `8.12%` drawdown.
   - Park the Sunday filter. It is too blunt.
   - Park rare time-to-confirm in its current form. It needs rework rather than promotion.

# 2026-06-06 - Expansion pack 1 validated and merged with current best

1. Objective:
   - Add more independent BTC lead families rather than over-tuning the current best block.
   - Test whether additional sieve-derived structure/pattern families can improve the current best production-alpha research candidate.
2. Expansion pack source:
   - Selected `25` BTC candidate families not already represented in the current best block and excluding known weak families.
   - Exported `969` BTC decision-hour signal rows.
3. Expansion pack validation:
   - Raw best-per-rule expansion block failed full Freqtrade validation:
     - `177` trades, `-12.17%` return, `85.47%` drawdown, `0.94` profit factor.
   - Cause:
     - one catastrophic family, `sieve2_multi2_vp_prior_month_high_break_vp_bullctx_long`, caused a large historical loss.
   - After removing that family:
     - `226` trades, `1140.33%` return, `7.26%` drawdown, `2.38` profit factor.
   - Positive-families-only version:
     - `201` trades, `888.11%` return, `4.90%` drawdown, `2.57` profit factor.
4. Merge validation:
   - Current clean best before merge:
     - `210` trades, `460.18%` return, `8.12%` drawdown, `2.14` profit factor.
   - Merged current-priority block:
     - `324` trades, `1693.90%` return, `11.70%` drawdown, `1.93` profit factor.
   - Merged expansion-priority block:
     - `324` trades, `1767.20%` return, `11.71%` drawdown, `1.96` profit factor.
5. Plain-English interpretation:
   - Expansion pack 1 is a real source of additional useful leads once the catastrophic family is removed.
   - The merged system greatly improves return and trade count, and drawdown remains far below the old `20%+` concern level.
   - Drawdown is still worse than the clean current best, so the merged version should be treated as a promoted research candidate, not final production alpha.
6. Reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_btc_expansion_pack1_freqtrade_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_expansion_pack1_freqtrade_comparison_20260606.csv`
7. Complete/sunflower note:
   - Complete-pattern rare entries are already included in the current best family through the validated complete-pattern artifacts.
   - Literal `sunflower` still has not been found in focused filename/content searches.
   - Do not claim sunflower is included until the exact artifact/rule naming is located or the user identifies the source.
8. Next work:
   - Treat `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1ExpansionPriorityStrategy` as the best high-return research candidate.
   - Run drawdown cleanup on the merged block, especially family-specific exits and crash/downturn invalidation.
   - Continue mining independent lead packs until the lead inventory approaches the `50-100` objective.

# 2026-06-06 - Merged expansion block drawdown cleanup found a new best candidate

1. Objective:
   - Reduce the `11.71%` drawdown in the high-return merged expansion-priority block without losing the expanded trade count advantage.
2. Audit finding:
   - Family audit of the merged expansion-priority block found three negative families:
     - `sieve1_bos_bull_continuation_long_8h`
     - `sieve1_tlv2_resistance_breakout_long_4h`
     - `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard`
3. Cleanup tests:
   - baseline merged expansion-priority:
     - `324` trades, `1767.20%` return, `11.71%` drawdown, `1.96` profit factor.
   - remove worst family only:
     - `317` trades, `1942.81%` return, `6.04%` drawdown, `2.03` profit factor.
   - remove two weak families:
     - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor.
   - remove three weak families:
     - `304` trades, `1753.60%` return, `6.05%` drawdown, `2.08` profit factor.
4. Plain-English interpretation:
   - Removing the worst one or two families improves both return and drawdown.
   - Removing all three is too aggressive because it loses return without improving drawdown.
5. Current best BTC production-alpha research candidate:
   - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`
   - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor, `55.7%` win rate.
6. Reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_merged_expansion_priority_family_audit_20260606_current_best_plus_expansion_pack1_expansion_priority.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_expansion_pack1_cleanup_comparison_20260606.csv`
7. Next work:
   - Audit this new best candidate for remaining drawdown events.
   - Test whether orderbook crash/downturn exits or structure-failure exits can reduce the remaining `6%` drawdown without cutting the biggest winners.
   - Keep mining additional independent lead packs; do not overfit only this candidate.

# 2026-06-06 - Trade-management follow-up on current best

1. Objective:
   - Test whether existing orderbook/time/structure trade-management layers or a targeted short-structure invalidation layer improve the current best BTC block.
2. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`
   - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor, `55.7%` win rate.
3. Existing trade-management tests:
   - full existing trade-management:
     - `315` trades, `1473.77%` return, `6.04%` drawdown, `2.04` profit factor.
   - orderbook plus time-to-confirm:
     - `311` trades, `1811.93%` return, `6.03%` drawdown, `2.04` profit factor.
   - orderbook plus structure:
     - `314` trades, `1784.91%` return, `6.04%` drawdown, `2.06` profit factor.
4. Targeted short-structure invalidation:
   - Added `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesShortStructureExitStrategy`.
   - It applies structure-failure exits only to the short families that appeared in the latest drawdown audit:
     - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`
     - `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h`
     - `sieve2_reversal_double_top_present_short_1h`
   - Result:
     - `319` trades, `1338.65%` return, `10.83%` drawdown, `2.06` profit factor, `51.4%` win rate.
5. Verdict:
   - Do not promote any of these trade-management variants.
   - Existing broad wrappers and the targeted short-structure layer cut too much edge and do not improve drawdown enough.
   - Current best remains `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_no_two_weak_families_trade_management_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-33-32.zip`
7. Complete/sunflower note:
   - Rare complete-pattern entries are already represented in the validated current-best path.
   - Literal `complete sunflower` remains unresolved until the exact artifact/rule name is found.

# 2026-06-06 - Expansion pack 2 created, validated, and merged

1. Objective:
   - Continue toward the `50-100` trader-readable BTC lead objective by adding another independent pack instead of only tuning the current best.
2. Candidate selection:
   - Started from `141` BTC sieve-derived candidate leads.
   - Excluded current-best rules, expansion-pack-1 rules, and known weak/catastrophic families.
   - Selected `32` remaining candidates with:
     - at least `5` BTC trades,
     - profit factor at least `1.25`,
     - candidate drawdown no worse than `15%`,
     - balanced source behaviours:
       - `9` pattern,
       - `7` resistance,
       - `7` support,
       - `9` other.
3. Signal/export process:
   - Exported `489` historical BTC decision-hour signal rows.
   - Per-rule confluence selected `154` signal rows.
   - Tailored exit sweep selected `18` rule-family exits.
4. Freqtrade validation:
   - Pack 2 standalone:
     - `131` trades, `421.99%` return, `7.06%` drawdown, `2.58` profit factor, `55.0%` win rate.
   - Current best baseline:
     - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor.
   - Current best plus pack 2, current-priority conflicts:
     - `357` trades, `3228.90%` return, `7.04%` drawdown, `2.01` profit factor, `54.9%` win rate.
   - Current best plus pack 2, pack2-priority conflicts:
     - `358` trades, `3091.94%` return, `7.03%` drawdown, `2.01` profit factor, `54.5%` win rate.
5. Plain-English interpretation:
   - Pack 2 is a useful independent lead block.
   - Merging pack 2 with the current best materially increased return and trade count.
   - Current-priority conflict handling is better than pack2-priority.
   - Drawdown increased from `6.05%` to about `7.04%`, so this is the new high-return research candidate, but it still needs drawdown cleanup before production-alpha promotion.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606_btc_expansion_pack2_balanced.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260606_btc_expansion_pack2_balanced_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_expansion_pack2_balanced_best_per_rule_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_btc_expansion_pack2_freqtrade_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-42-30.zip`
7. Next work:
   - Treat `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy` as the new high-return current candidate.
   - Audit its weak families and drawdown period.
   - Continue independent lead-pack mining toward the `50-100` objective, but do not ignore the new `7.04%` drawdown cleanup requirement.

# 2026-06-06 - Expansion pack 2 weak-family cleanup

1. Objective:
   - Check whether the new high-return pack-2 merged candidate can be cleaned without losing the extra return and trade count.
2. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy`
   - `357` trades, `3228.90%` return, `7.04%` drawdown, `2.01` profit factor, `54.9%` win rate.
3. Cleanup variants:
   - remove worst overall family:
     - `352` trades, `2968.85%` return, `7.04%` drawdown, `2.04` profit factor, `55.1%` win rate.
     - Rejected: lower return with no drawdown gain.
   - remove top 3 weak overall families:
     - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor, `55.5%` win rate.
     - Promoted as the current best research candidate.
   - remove pack-2 weak families:
     - `342` trades, `2957.66%` return, `6.82%` drawdown, `2.15` profit factor, `55.3%` win rate.
     - Parked: cleaner but lower return.
   - remove drawdown-window losing families:
     - `275` trades, `1513.50%` return, `12.56%` drawdown, `2.25` profit factor, `57.8%` win rate.
     - Rejected: over-cleaned and worsened drawdown.
4. Current best:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
   - This keeps the rare complete-pattern/sunflower-style research path represented through the existing complete-pattern artifacts and dedicated exit handling.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_pack2_cleanup_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-50-09.zip`
6. Next work:
   - Treat this as the new BTC current-best research candidate.
   - Continue adding independent lead packs toward `50-100` leads.
   - Revisit drawdown using precise failure-mode exits, not broad family deletion.

# 2026-06-06 - Objective audit created

1. Objective:
   - Add a separate objective-level audit so future agents can check completion against the canonical objectives without modifying `current_objectives.md`.
2. Added:
   - `C:\FreqTradeStuff\ai_guidance_docs\objective_audit.md`
3. Guidance map update:
   - Added `objective_audit.md` to `C:\FreqTradeStuff\ai_guidance_docs\README.md`.
4. Current objective-level findings:
   - Strongest current BTC research candidate:
     - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
     - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor, `55.5%` win rate.
   - The system is still research-only, not production-alpha complete.
   - The lifecycle ledger has many concepts, but only a smaller subset are strategy-level leads.
   - News/context remains mostly deferred for trading confluence because several source blocks are sparse or under separate rework.
5. Highest-priority remaining work recorded in the audit:
   - Build a dedicated column dictionary.
   - Promote more direct-promising concepts into strategy-level leads.
   - Test precise current-best trade-management exits.
   - Add buy-and-hold and baseline comparisons to production-alpha promotion reports.
   - Resolve whether `complete sunflower` is a separate artifact or a nickname for complete-pattern entries.

# 2026-06-06 - Enhanced column dictionary created

1. Objective:
   - Make the 1h feature snapshot explainable in trader language so future reports can name exact source-detail groups and column meanings.
2. Input:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_goal_feature_research_after_epsilon_20260605.csv`
3. Output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary_enhanced_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary_enhanced_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary_enhanced_20260606_summary.json`
4. Coverage:
   - `2576` columns documented.
   - Added:
     - source detail,
     - plain-English meaning,
     - trader-use hint,
     - source readiness,
     - timestamp-safety note.
5. Important notes:
   - `2434` columns are from source-detail groups marked `covered_active`.
   - `100` columns are from `too_sparse` groups and should not be used as full-history confluence.
   - `33` target-label columns are explicitly marked as research answer-sheet columns, not live inputs.
6. Audit update:
   - `objective_audit.md` now marks the column dictionary objective as `ready`, with the caveat that descriptions are inferred and should be refined when exact indicator semantics matter.

# 2026-06-06 - Current best production-promotion gate report

1. Objective:
   - Add the missing production-promotion gate comparison for the current best BTC research candidate, especially against same-window buy-and-hold/market change.
2. Candidate:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
   - Evidence artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-50-09.zip`
3. Generated reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_years_20260606.csv`
4. Key result:
   - Strategy return:
     - `3236.58%`
   - Same-window market change / buy-and-hold proxy from Freqtrade artifact:
     - `1247.17%`
   - Return ratio:
     - `2.60x`
   - Drawdown:
     - `6.82%`
   - Trades:
     - `346`, about `57.3` trades/year.
   - Long/short:
     - long return `1878.02%`,
     - short return `1358.56%`.
   - Full-year stability:
     - `6/6` full historical years positive from 2020 through 2025.
5. Gate verdict:
   - Passes research-promotion gates for buy-and-hold comparison, drawdown, trade count, long/short contribution, profit factor, and full-year stability.
   - Still not production-alpha complete because source coverage refresh, more lead breadth, targeted exit validation, risk validation, and dry-run-specific wrapper/gates remain incomplete.

# 2026-06-06 - Current best failure-mode audit

1. Objective:
   - Identify where the current best still loses money so the next trade-management work is targeted rather than generic.
2. Candidate:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
3. Generated reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_failure_mode_audit_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_trades_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_families_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_months_20260606.csv`
4. Main finding:
   - Losses are concentrated enough for targeted exit tests.
   - Worst month:
     - `2025-12`, `5` trades, `-2044.96` absolute profit, worst trade `-887.13`.
   - Worst single-trade families include:
     - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`
     - `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h`
     - `sieve1_ladder_long_sup_hold`
     - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`
     - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`
5. Next targeted tests:
   - December-2025 drawdown replay with orderbook crash/downturn and volatility risk states.
   - VP resistance-break long acceptance/volume-failure exits.
   - Support-break short support-reclaim/structure-failure exits.
   - Ladder support-hold time-to-confirm and volatility-shock exits.

# 2026-06-06 - Targeted current-best exit tests

1. Objective:
   - Test whether precise current-best failure-mode exits improve the pack-2 current best without over-managing the system.
2. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
   - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor, `55.5%` win rate.
3. Variants tested:
   - short structure-failure exit:
     - `356` trades, `2280.07%` return, `10.09%` drawdown, `2.12` profit factor.
     - Rejected: worse return and worse drawdown.
   - long volume/time failure exit:
     - `363` trades, `2126.85%` return, `6.99%` drawdown, `2.08` profit factor.
     - Rejected: cut return and slightly worsened drawdown.
   - long orderbook crash exit:
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
     - Promoted as new current best research candidate.
   - combined targeted failure stack:
     - `376` trades, `1594.13%` return, `9.35%` drawdown, `2.36` profit factor.
     - Rejected: over-managed and cut too much edge.
4. Interpretation:
   - Orderbook works when used as a targeted exit/risk layer for vulnerable long families.
   - It should not be treated as a broad entry filter or combined blindly with every possible exit.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_comparison_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-04-43.zip`
6. Current best:
   - `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`
   - Still research-only; production-alpha gates remain incomplete.

# 2026-06-06 - Refreshed gates for orderbook-exit current best

1. Objective:
   - Refresh promotion-gate and failure-mode evidence after the targeted orderbook exit became the current best.
2. Current best:
   - `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`
3. Promotion-gate reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_orderbook_exit_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_orderbook_exit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_orderbook_exit_years_20260606.csv`
4. Gate result:
   - Strategy return:
     - `3721.77%`
   - Same-window market change:
     - `1247.17%`
   - Ratio:
     - `2.98x`
   - Drawdown:
     - `6.04%`
   - Profit factor:
     - `2.44`
   - Trades:
     - `356`, about `58.9` trades/year.
   - Full-year stability:
     - `6/6` full historical years positive.
5. Refreshed failure audit:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_orderbook_exit_failure_mode_audit_20260606.md`
6. Next branch:
   - Rework short-side exits with stricter support-reclaim logic.
   - Test VP resistance-break long acceptance-failure exits only after price loses the broken level.
   - Test isolated ladder/support-hold exits with volatility shock plus loss-of-support.

# 2026-06-06 - Strict current-best exit rework

1. Objective:
   - Rework the failed broad short-structure and long acceptance-failure exits into stricter family-specific failure rules.
2. Variants added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestObExitLongAcceptanceFailureStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestObExitStrictReclaimAndAcceptanceFailureStrategy`
3. Result:
   - Current best return branch remains:
     - `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Best drawdown branch is now:
     - `TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy`
     - `357` trades, `3685.89%` return, `5.30%` drawdown, `2.47` profit factor.
4. Interpretation:
   - Strict short reclaim is useful as a drawdown-control candidate.
   - Long acceptance-failure did not materially fire, so that logic needs a different diagnostic built from the actual worst long trades.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_strict_exit_rework_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_strict_exit_rework_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-16-12.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-16-43.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-17-14.zip`
6. Next step:
   - Park strict short reclaim as a lower-drawdown branch.
   - Rework long failures using concrete worst-trade replay instead of broad acceptance-failure thresholds.

# 2026-06-06 - Long failure replay and expanded orderbook exit test

1. Objective:
   - Use actual worst long trades from the current best to design the next long-exit diagnostic.
2. Replay result:
   - Sampled the `30` worst remaining long losses from `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`.
   - `27/30` clustered under `orderbook_stress`.
   - `3/30` remained `unclear`.
3. Follow-up variants:
   - `TraderRuleBlockBtcSieveCurrentBestReplayExpandedOrderbookStressExitStrategy`
     - `374` trades, `3374.76%` return, `6.04%` drawdown, `2.52` profit factor.
     - Rejected: return fell by about `347` percentage points without improving drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestReplayExpandedObStressStrictShortStrategy`
     - `375` trades, `3342.79%` return, `5.29%` drawdown, `2.55` profit factor.
     - Rejected versus strict short branch: nearly same drawdown, much lower return.
4. Interpretation:
   - Orderbook stress does appear in many losing longs, but stress alone is too broad because it also appears during recoverable winners.
   - Future long-exit work should require orderbook stress plus failed reclaim/support and negative trade progress.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_failure_replay_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_replay_expanded_orderbook_exit_20260606.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-22-48.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-23-19.zip`

# 2026-06-06 - Tighter long orderbook-stress invalidation test

1. Objective:
   - Test whether orderbook stress becomes useful for more long families when combined with a losing trade and failed support/structure.
2. Variants:
   - `TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationExitStrategy`
     - `350` trades, `3311.53%` return, `6.04%` drawdown, `2.15` profit factor.
   - `TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationStrictShortStrategy`
     - `351` trades, `3281.12%` return, `5.30%` drawdown, `2.16` profit factor.
3. Interpretation:
   - The shared long invalidation rule still removes too much useful long-side edge.
   - It does not improve drawdown versus current best unless strict short reclaim is also included, and then it performs far worse than strict short reclaim alone.
4. Decision:
   - Reject shared long orderbook-stress invalidation in this form.
   - Rework long failures by individual family, not as a shared long-side exit.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_ob_stress_invalidation_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_ob_stress_invalidation_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-26-58.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-27-29.zip`

# 2026-06-06 - Current-best long family diagnostics

1. Objective:
   - Stop treating all long failures as one market story and identify which individual long entry families have distinct loser-skewed failure states.
2. Tool added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\current_best_family_diagnostics.py`
3. Reports generated:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606_signals.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606_trades.csv`
4. Key findings:
   - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard` has the most long losses, but remains profitable overall. Its loser-skewed clues are support removal and spread fragility, not a reason for blanket removal yet.
   - `sieve2_bos_bull_continuation_long_1h` is net negative in the current best sample and should be tested as a removal or strict-filter candidate.
   - `sieve1_geometry_triangle_squeeze_breakout_long_1h` losers show support-cleared / breakout-failure style exit clues.
   - `sieve1_ladder_long_sup_hold` losers show failed breakdown/breakout structure risk and support-cleared clues.
   - `sieve1_vp_lvn_fast_traverse_long_1h` losers show orderbook support removal, downside vacuum, and spread fragility.
5. Decision:
   - Promote per-family long exit/removal tests as the next branch.
   - Do not run more shared long-side exits until per-family tests show a reusable state.

# 2026-06-06 - Current-best family-specific tests

1. Objective:
   - Test the first per-family actions from the long-family diagnostic.
2. Variants added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestFamilySpecificLongExitV1Strategy`
3. Results:
   - Prior current best:
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Remove only `sieve2_bos_bull_continuation_long_1h`:
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
   - Bundled family-specific long exit V1:
     - `359` trades, `3606.94%` return, `6.05%` drawdown, `2.39` profit factor.
4. Decision:
   - Promote the BOS-removal branch as the new max-return research candidate.
   - Reject bundled family-specific long exit V1.
   - Continue with one-family-at-a-time exit tests.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_family_specific_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_family_specific_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-37-49.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-38-20.zip`

# 2026-06-06 - Single-family long exit tests

1. Objective:
   - Starting from the BOS-removal max-return branch, test one long-family exit at a time.
2. Variants added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosTriangleSupportClearedExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoBosLvnOrderbookFragilityExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoBosLadderStructureFailureExitStrategy`
3. Results:
   - No-BOS baseline:
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
   - Triangle support-cleared exit:
     - `352` trades, `4162.88%` return, `6.04%` drawdown, `2.52` profit factor.
   - LVN orderbook-fragility exit:
     - no measurable change versus no-BOS baseline.
   - Ladder structure-failure exit:
     - no measurable change versus no-BOS baseline.
4. Decision:
   - Promote the triangle support-cleared exit as the new max-return branch.
   - Park/rework LVN and ladder triggers.
   - Next test should combine triangle branch with strict short reclaim, or test another single family.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_single_family_exit_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_single_family_exit_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-42-45.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-43-16.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-43-48.zip`

# 2026-06-06 - Combined triangle and strict-short exit test

1. Objective:
   - Test whether the highest-return no-BOS triangle branch can borrow the strict short-reclaim exit that previously reduced drawdown.
2. Variant added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy`
3. Result:
   - `353` trades.
   - `4125.17%` return.
   - `5.30%` drawdown.
   - `2.55` profit factor.
   - `54.4%` win rate.
4. Comparison:
   - Triangle max-return branch:
     - `4162.88%` return, `6.04%` drawdown, `2.52` profit factor.
   - Strict short-reclaim branch:
     - `3685.89%` return, `5.30%` drawdown, `2.47` profit factor.
5. Decision:
   - Promote the combined branch as the best balanced production-alpha research candidate.
   - Keep the triangle-only branch as the highest-return candidate.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_combined_exit_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_combined_exit_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-49-19.zip`
7. Promotion-gate refresh:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_families.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_months.csv`

# 2026-06-07 - Balanced family removal tests

1. Objective:
   - Test whether weak/low-efficiency families should be removed rather than managed with more exit logic.
2. Variants added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakLongStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakAndPocShortStrategy`
3. Results:
   - Balanced baseline:
     - `353` trades, `4125.17%` return, `5.30%` drawdown, `2.55` profit factor.
   - Remove overtrade resistance-break long:
     - `327` trades, `3965.75%` return, `5.30%` drawdown, `2.70` profit factor.
   - Remove POC reject short:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Remove both:
     - `310` trades, `4528.92%` return, `5.30%` drawdown, `2.91` profit factor.
4. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy` as the current strongest research candidate.
   - Keep the both-removed branch as a cleaner, higher-profit-factor alternative.
   - Do not remove the overtrade resistance-break long alone; rework it with filtering/exits if needed.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_balanced_family_removal_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_balanced_family_removal_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_reject_short_20260607_promotion_gate.md`

# 2026-06-07 - Short volume-pressure exit tests rejected

1. Objective:
   - Test whether the current strongest no-POC branch can reduce short-side damage by exiting when volume pressure turns against vulnerable short entries.
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitExpandedStrategy`
3. Results:
   - Current strongest no-POC baseline:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Top short-family volume-pressure exit:
     - `339` trades, `3062.21%` return, `10.77%` drawdown, `2.51` profit factor.
   - Expanded short-family volume-pressure exit:
     - `341` trades, `2906.33%` return, `10.78%` drawdown, `2.50` profit factor.
4. Decision:
   - Reject volume pressure alone as a short invalidation trigger.
   - Keep `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy` as the current strongest research candidate.
   - Rework short exits only when volume pressure is paired with clearer trader-state evidence such as reclaim, acceptance against the short, bid absorption, or breakdown failure.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_volume_pressure_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_volume_pressure_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-03-07.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-03-30.zip`

# 2026-06-07 - Contextual short-failure exits rejected

1. Objective:
   - Rework the failed short volume-pressure exits into more trader-like contextual exits using breakdown failure, bid absorption, reclaim pressure, and loss-only thresholds.
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitExpandedStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitExpandedStrategy`
3. Results:
   - Current strongest no-POC baseline:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Top-family contextual exit:
     - `343` trades, `3794.52%` return, `10.70%` drawdown, `2.75` profit factor.
   - Expanded contextual exit:
     - `348` trades, `3097.61%` return, `9.72%` drawdown, `2.75` profit factor.
   - Top-family loss-only contextual exit:
     - `338` trades, `4142.23%` return, `9.48%` drawdown, `2.52` profit factor.
   - Expanded loss-only contextual exit:
     - `338` trades, `4021.28%` return, `8.76%` drawdown, `2.54` profit factor.
4. Decision:
   - Reject all four variants.
   - Current strongest remains `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`.
   - Park broad short-failure exits unless the next test is built from exact worst-short replay evidence.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_failure_context_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_failure_context_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-10-47.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-11-16.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-12-42.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-13-14.zip`

# 2026-06-07 - Current strongest long-family exits rejected

1. Objective:
   - Test whether two narrow long-family exits could improve the current strongest no-POC branch using orderbook/structure failure clues found in the family diagnostics.
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocLadderSupportFailureExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocLvnFragilityLossExitStrategy`
3. Results:
   - Current strongest no-POC baseline:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Ladder support-failure exit:
     - `336` trades, `4513.27%` return, `5.30%` drawdown, `2.64` profit factor.
   - LVN fragility loss exit:
     - `336` trades, `4507.71%` return, `5.30%` drawdown, `2.64` profit factor.
4. Decision:
   - Reject both variants for now.
   - Keep `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy` as the current strongest branch.
   - Move the next entry-family expansion toward locating and validating rare high-win-rate `complete sunflower` sieve entries as separate candidate families.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_long_family_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_long_family_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-18-10.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-20-09.zip`

# 2026-06-08 - Acceptance criteria tightened for exits, long/short balance, and buy-and-hold

1. Objective:
   - Update production-alpha acceptance criteria based on user clarification before continuing further strategy testing.
2. Changes made:
   - Current objectives now state that each entry condition/family should have its own associated exit/invalidation logic.
   - Broad generic exits are now only acceptable for true crash/downturn escalation or multi-indicator/data-source reversal confluence.
   - Normal candidate strategies must keep a reasonable long/short trade-count ratio, initially `80/20` one way or the other, unless explicitly treated as specialist lanes.
   - Promoted production-alpha candidates must beat same-window buy-and-hold/market-change and relevant baselines.
3. Current strongest branch checked against the new ratio rule:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
   - `332` total trades.
   - `208` long trades.
   - `124` short trades.
   - `63/37` long/short ratio.
   - Result: passes the new `80/20` ratio rule.
4. Decision:
   - Keep `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the current research baseline.
   - Continue next work from targeted entry-family failure replay, long/short-aware promotion gates, and position-management variants.
   - Do not promote the December short-failure broad exit branch; it remains rejected.

# 2026-06-08 - Current strongest family diagnostics refreshed

1. Objective:
   - Continue the next step without changing strategy logic by refreshing family-level diagnostics for the current strongest branch.
2. Branch audited:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
   - Backtest archive: `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-46-44.zip`
3. Main failure-family candidates:
   - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`: `51` trades, `24` losses, `52.94%` win rate, `65.31%` total profit, worst loss `-2.50%`.
   - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`: `34` trades, `23` losses, `32.35%` win rate, `23.60%` total profit, worst loss `-1.80%`.
   - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`: `27` trades, `11` losses, `59.26%` win rate, `21.58%` total profit, worst loss `-3.50%`.
4. Practical next test:
   - Build one-family-at-a-time exit/filter tests from the loser-skewed exit clues.
   - Do not add another broad generic exit.
   - First sensible candidates are:
     - short support-break family: volume-pressure/reclaim-style invalidation,
     - overtrade resistance-break long: support-removed/spread-fragility invalidation,
     - VP resistance-break VAL long: bid-absorption or local breakdown invalidation.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_family_diagnostics_20260608.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_family_diagnostics_20260608.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_family_diagnostics_20260608_signals.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_family_diagnostics_20260608_trades.csv`

# 2026-06-07 - Complete-pattern rare entries tested against current strongest branch

1. Objective:
   - Include the user's requested rare high-win-rate sieve-complete family where evidence supports it, without blindly damaging the current strongest branch.
2. What was found:
   - Existing complete-pattern rare-entry artifacts already exist.
   - Literal `sunflower` naming was not found in the strategy/report files searched, so the current tested family is `complete-pattern rare entries`, not confirmed as a separate Sunflower-specific family.
3. Existing standalone rare evidence:
   - Rare-only Freqtrade branch:
     - `27` trades, `73.20%` return, `6.30%` drawdown, `4.54` profit factor, `59.3%` win rate.
   - Tailored exit research on the rare entries:
     - `27` trades, `88.89%` win rate, `182.13%` simple total return, `-1.92%` max drawdown, `24.315` profit factor.
4. New merge tests:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCurrentPriorityStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRarePriorityStrategy`
5. Results:
   - Current strongest no-POC baseline:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Current-best priority merge:
     - `339` trades, `4489.73%` return, `5.30%` drawdown, `2.67` profit factor.
   - Rare-priority merge:
     - `339` trades, `4489.73%` return, `5.30%` drawdown, `2.67` profit factor.
6. Decision:
   - Reject simple merge into the current strongest branch for now.
   - Keep complete-pattern rare entries as a promising standalone lead family.
   - Next rare tests should be filtered rare-only, rare as replacement for weak current-best families, or Sunflower-specific if a separate named source is located.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-26-01.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-26-31.zip`

# 2026-06-07 - Filtered complete-pattern rare merges also rejected

1. Objective:
   - Test whether the rare complete-pattern family improves the current strongest branch when restricted to the filtered contexts that helped rare-only evidence.
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCompressionRangeStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRangeBreakStrategy`
3. Results:
   - Current strongest no-POC baseline:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Compression/range rare-priority merge:
     - `338` trades, `4351.32%` return, `5.30%` drawdown, `2.63` profit factor.
   - Range-break rare-priority merge:
     - `337` trades, `4593.56%` return, `5.30%` drawdown, `2.72` profit factor.
4. Decision:
   - Reject both filtered simple merges for now.
   - Range-break rare merge was closest, but still did not beat baseline.
   - Keep complete-pattern rare entries as a separate lead family rather than simple add-on.
5. Sunflower search:
   - Focused `rg -i sunflower ai_guidance_docs user_data` found only guidance/progress mentions, not a strategy/report/config artifact.
   - Literal `sunflower` remains unresolved unless the user identifies the exact artifact/rule name.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-30-45.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-31-14.zip`

# 2026-06-07 - Rare range-break replacement branch rejected

1. Objective:
   - Test whether rare range-break complete-pattern entries can help if used as replacements for weaker current-best families instead of simple additions.
2. Variant tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoOvertradePlusRareRangeBreakReplacementStrategy`
3. Setup:
   - Removed `sieve2_reframed_vp_poc_reject_short_1h`.
   - Removed `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`.
   - Added `12` range-break complete-pattern rare replacement rows.
   - Final signal snapshot had `439` rows and no duplicate timestamps.
4. Result:
   - `311` trades, `4198.00%` return, `5.30%` drawdown, `14.21%` max underwater, `2.80` profit factor, `58.5%` win rate.
5. Decision:
   - Reject this replacement branch for now.
   - It improved win rate versus the current strongest branch but sacrificed too much return and did not reduce drawdown.
   - Keep rare complete-pattern entries as a standalone/market-state-specific lead family, not as this replacement.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260607_current_strongest_no_poc_no_overtrade_plus_rare_range_break_replacement.parquet`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-34-39.zip`

# 2026-06-07 - Complete-pattern rare overlap audit completed

1. Objective:
   - Determine why high-quality complete-pattern rare entries did not improve the current strongest branch when merged.
2. Result:
   - `23` of `29` rare signals were contested by an active current-strongest trade at the next-hour entry point.
   - Rare standalone contested-row average return was `2.68%`.
   - Current active contested-row average return was `3.22%`.
   - Rare beat the active current trade in `7` of `23` contested rows.
3. Decision:
   - The rare family remains valuable, but mostly as a priority/gating problem rather than a blind merge.
   - Next work should test when rare should override current entries, not whether every rare entry should be added.
4. Formal audit file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_complete_pattern_rare_overlap_audit_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_complete_pattern_rare_overlap_audit_20260607.csv`
5. Follow-up decision:
   - Do not backtest a new rare-priority gate from this tiny sample.
   - Non-contested rare entries were weak, while contested rare entries were usually already competing with stronger active current-best trades.
   - Move to another lead source unless a separate literal `sunflower` artifact is located.

# 2026-06-07 - Tiny negative long-family removal creates new strongest branch

1. Objective:
   - Test whether tiny but clearly weak current-best long families should be removed from the current strongest no-POC branch before adding more entries or exits.
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTripleBottomLongStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `13.24%` max underwater, `2.73` profit factor, `56.0%` win rate.
4. Results:
   - Remove only triple-bottom long:
     - `334` trades, `4679.24%` return, `5.29%` drawdown, `13.24%` max underwater, `2.75` profit factor, `56.3%` win rate.
   - Remove tiny negative long families:
     - `332` trades, `4764.00%` return, `5.30%` drawdown, `12.69%` max underwater, `2.75` profit factor, `56.6%` win rate.
   - Remove all net-negative absolute families including overtrade resistance-break long:
     - `306` trades, `4607.96%` return, `5.30%` drawdown, `13.73%` max underwater, `2.94` profit factor, `59.8%` win rate.
5. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the new strongest return branch for research.
   - Keep `TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy` as a cleaner high-profit-factor alternative.
   - Reject removing only triple-bottom long as a main branch because it did not improve return.
6. Complete/sunflower handling:
   - The user wants rare `complete sunflower`-style sieve entries included in the final system if they are genuinely high-win-rate.
   - Current repo search still has no literal `sunflower` artifact, so this remains represented by the complete-pattern rare-entry family unless a separate artifact is identified.
   - Final-system design should allow each entry family to have a dedicated exit, but rare entries also need priority/gating logic because simple merges weakened the current strongest branch.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_tiny_family_removal_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_tiny_family_removal_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_family_audit_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-46-44.zip`

# 2026-06-07 - New strongest branch promotion/failure audit

1. Objective:
   - Verify the new strongest branch against promotion-gate evidence rather than relying only on the headline backtest table.
2. Implementation:
   - Added reusable audit script:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_backtest_audit.py`
   - The script reads a Freqtrade backtest ZIP and writes:
     - promotion summary,
     - completed-year stability,
     - worst months,
     - worst trades,
     - worst entry families,
     - markdown verdict.
3. Audited branch:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
4. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
5. Promotion result:
   - `332` trades.
   - `4764.00%` return versus baseline `4700.59%`.
   - `5.30%` drawdown versus baseline `5.30%`.
   - `2.75` profit factor versus baseline `2.73`.
   - `56.6%` win rate versus baseline `56.0%`.
   - `3.82x` same-window market-change return ratio.
   - `6/6` completed yearly buckets positive.
6. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the current research baseline.
   - Keep failure-mode work open because drawdown was not materially reduced.
7. Main failure clues:
   - Worst month: December 2025, `-2217.96`.
   - Worst family by absolute profit: `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`, `-68.40` across `34` trades.
   - Worst trade: `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`, opened `2025-12-17 08:00:00+00:00`, `-1282.86`.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_years.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_families.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_months.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_trades.csv`

# 2026-06-07 - December short-failure exit rejected

1. Objective:
   - Use the new failure audit to test whether a narrow December-style short-failure exit could reduce drawdown without damaging the new strongest branch.
2. Diagnostic evidence:
   - December 2025 short cluster:
     - `5` short trades.
     - `1` win, `4` losses.
     - `-2219.03` total profit in the diagnostic CSV scale.
   - The cluster showed elevated failed-breakdown/reclaim-style evidence, including `st_failed_breakdown_structure_risk`, support bounce, and bearish pressure not cleanly resolving downward.
3. Variant added and tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyNegativeDecShortFailureExitStrategy`
4. Result versus current research baseline:
   - Candidate:
     - `332` trades, `4533.53%` return, `6.06%` drawdown, `2.73` profit factor, `55.7%` win rate.
   - Baseline:
     - `332` trades, `4764.00%` return, `5.30%` drawdown, `2.75` profit factor, `56.6%` win rate.
5. Decision:
   - Reject this exit variant.
   - It improved the worst December month but over-managed the full branch and worsened the main promotion metrics.
   - The diagnostic remains useful, but the next rework should be entry filtering or exact per-family replay, not this broad post-entry exit.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_failure_cluster.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_cluster_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_feature_comparison.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec_short_failure_exit_20260607_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_01-02-37.zip`

# 2026-06-08 - Same-window targeted family exit batch

1. Objective:
   - Continue the production-alpha objective by testing family-specific exits for the current BTC sieve baseline without stopping Sieve runs.
   - Add small reusable tooling only where it reduces repeated result parsing.
2. Tooling added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_batch_result_summary.py`
   - Purpose: summarize multi-strategy Freqtrade ZIPs against a same-window baseline with return, market-change, drawdown, profit factor, trade count, and long/short balance gates.
3. Strategy variants added:
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyShortSupportBreakReclaimExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyOvertradeResBreakSupportLossExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyVpValLongFailureExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyTargetedFamilyExitStackStrategy`
4. Important correction:
   - The first batch was accidentally broader than the saved BTC-only baseline because it ran all whitelisted pairs.
   - The batch was rerun with `--pairs BTC/USDT:USDT`.
   - The original baseline was also rerun with the same current BTC-only command to create a fair comparison anchor.
5. Same-window BTC baseline:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
   - `332` trades, `208` long / `124` short.
   - `41.93%` return, `1.14%` drawdown, `2.86` profit factor, `56.6%` win rate.
   - Same-window market change was `690.64%`, so this current rerun does not satisfy the new buy-and-hold promotion gate.
6. Targeted exit results:
   - Overtrade resistance-break support-loss exit: `41.70%` return, `1.14%` drawdown, `2.85` profit factor.
   - VP VAL long failure exit: `41.56%` return, `1.14%` drawdown, `2.83` profit factor.
   - Short support-break reclaim exit: `39.98%` return, `1.10%` drawdown, `2.76` profit factor.
   - Stacked targeted exits: `39.98%` return, `1.10%` drawdown, `2.76` profit factor.
7. Decision:
   - Do not promote any targeted exit variant.
   - The small drawdown improvement in the short reclaim variant is not worth the return loss.
   - The stacked variant inherited the short-exit damage and did not add value.
8. Same-window family diagnostics:
   - Refreshed diagnostics from `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-06-49.zip`.
   - No current entry family is net negative in the current same-window baseline.
   - Therefore, simple family deletion is not the next path. Prefer selective per-family invalidation or entry-quality filters.
9. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_targeted_family_exit_btc_same_window.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_targeted_family_exit_btc_same_window.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_baseline_same_window_family_diagnostics_20260608.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-05-38.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-06-49.zip`

# 2026-06-08 - Same-window targeted family exit rerun and note cleanup

1. Objective:
   - Rerun the BTC-only baseline and targeted family-exit batch.
   - Remove duplicated/outdated high-level notes so future agents do not treat old archive promotion numbers as current acceptance evidence.
2. Rerun results:
   - Baseline rerun remained unchanged: `332` trades, `41.93%` return, `1.14%` drawdown, `2.86` profit factor, `208` long / `124` short.
   - Targeted family-exit variants again failed to improve the same-window baseline enough to promote.
3. Note cleanup:
   - Replaced the oversized `objectives_high_level_results_summary.md` with a concise current-state file.
   - Historical detail remains in this progress file and in machine-readable/markdown reports.
4. Decision:
   - Do not promote the targeted exits.
   - Keep the same-window baseline as the diagnostic anchor.
   - Next work should focus on selective entry-quality filters, signal priority, and per-family invalidation/risk logic.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-12-35.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-14-46.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_targeted_family_exit_btc_rerun.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_baseline_same_window_family_diagnostics_20260608_rerun.md`

# 2026-06-08 - Entry-quality refinement batch

1. Objective:
   - Continue improving entries/exits without discarding weak-looking families that may become useful once refined.
   - Test family-specific entry-quality filters before adding more broad exits.
2. New variants:
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestEntryQualityFilterStackStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeAndSoftShortCleanEntryStrategy`
3. Same-window baseline:
   - `332` trades, `41.93%` return, `1.14%` drawdown, `2.86` profit factor.
4. Useful result:
   - Overtrade resistance-break clean-entry improved the same-window baseline slightly:
     - `327` trades,
     - `42.23%` return,
     - `1.14%` drawdown,
     - `2.92` profit factor,
     - `57.5%` win rate.
   - This is the best max-return refinement from this batch.
5. Balanced-risk result:
   - Soft support-break short clean-entry reduced drawdown and improved profit factor, but gave up return:
     - `310` trades,
     - `40.53%` return,
     - `0.75%` drawdown,
     - `3.06` profit factor.
   - Overtrade plus soft-short stack was similar:
     - `305` trades,
     - `40.83%` return,
     - `0.74%` drawdown,
     - `3.13` profit factor.
6. Rejected/rework result:
   - VP-VAL long clean-entry was too aggressive and worsened the branch:
     - `320` trades,
     - `39.83%` return,
     - `1.43%` drawdown,
     - `2.77` profit factor.
   - Full three-filter stack removed too much edge.
7. Decision:
   - Keep overtrade clean-entry as the next max-return path.
   - Keep soft-short clean-entry as a balanced-risk path.
   - Rework VP-VAL using exit or sizing logic rather than entry removal.
   - Continue refining families individually before merging.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_entry_quality_refinement_btc.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_soft_short_entry_refinement_btc.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-19-41.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_21-21-28.zip`

# 2026-06-08 - Finish-line lane pivot

1. Objective:
   - Refocus production-alpha work on older buy-and-hold-beating strategy branches instead of chasing weaker diagnostic rerun paths.
2. Update made:
   - Added `C:\FreqTradeStuff\ai_guidance_docs\production_alpha_finish_line_lanes.md`.
   - Updated `C:\FreqTradeStuff\ai_guidance_docs\README.md` to route production-alpha finishing agents to that file.
   - Rewrote `C:\FreqTradeStuff\ai_guidance_docs\objectives_high_level_results_summary.md` so the active direction is clear.
3. Active finish-line anchors:
   - Max-return: `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`, `4764.00%` return, `5.30%` drawdown, `2.75` profit factor.
   - Balanced high-return: `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
   - Orderbook-risk: `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Lower-drawdown strict-exit: `TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy`, `3685.89%` return, `5.30%` drawdown, `2.47` profit factor.
   - Structure/triangle max-return: `TraderRuleBlockBtcSieveCurrentBestNoBosTriangleSupportClearedExitStrategy`, `4162.88%` return, `6.04%` drawdown, `2.52` profit factor.
4. Decision:
   - Recent same-window `41-42%` variants are kept as diagnostic/rework ideas only.
   - Future changes should be accepted only if they improve the relevant finish-line lane anchor, or clearly improve drawdown/profit factor/failure-mode control enough to suit that lane's purpose.
   - Do not remove weaker branches; park them for later pipeline revision.

# 2026-06-08 - Finish-line multi-plan BTC batches

1. Objective:
   - Trial multiple entry, exit, and risk-management plans for each active finish-line lane.
   - Compare each variant against its own lane anchor, not against a weaker unrelated branch.
2. Scope:
   - BTC-only.
   - Timerange `20200101-20260115`.
   - Existing strategy variants only; no new trading logic was added in this step.
3. Batches run:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_22-48-55.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_22-58-46.zip`
4. Combined report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_finish_line_btc_combined_multi_plan.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260608_finish_line_btc_combined_multi_plan.csv`
5. Important reproducibility note:
   - Current-code BTC reruns did not reproduce the older archived `4764.00%` return scale.
   - The old archive had `332` BTC-only trades and `4764.00%` return.
   - The current-code BTC anchor had the same `332` trades but `41.93%` return.
   - Treat this batch as variant-ranking evidence until the return-scale/config mismatch is fully explained.
6. Useful candidates found:
   - Max-return entry improvement:
     - `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy`
     - `327` trades, `42.23%` return, `1.14%` drawdown, `2.92` profit factor.
     - Versus max-return current-code anchor: `+0.30%` return, unchanged drawdown, `+0.05` profit factor.
   - Max-return risk tradeoffs:
     - `TraderRuleBlockBtcSieveCurrentBestOvertradeAndSoftShortCleanEntryStrategy`
     - `305` trades, `40.83%` return, `0.74%` drawdown, `3.13` profit factor.
     - Lower return, but drawdown improved by `0.40%` and profit factor improved by `0.27`.
     - `TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy`
     - `310` trades, `40.53%` return, `0.75%` drawdown, `3.06` profit factor.
   - Position-management risk tradeoff:
     - `TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy`
     - `332` trades, `36.60%` return, `0.87%` drawdown, `3.00` profit factor.
     - It reduced drawdown but gave up too much return to promote as-is.
7. Parked findings:
   - Balanced lane variants did not beat the balanced anchor.
   - Orderbook-risk expansion variants did not improve the orderbook-risk anchor.
   - Strict-exit expansion variants did not improve the strict-exit anchor.
   - Triangle/structure exit variants did not improve the triangle anchor.
8. Decision:
   - Keep `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy` as the next max-return improvement candidate.
   - Keep the soft-short and selective-reserve variants as risk-control ideas, not promoted finish-line replacements.
   - Before final promotion, resolve why current-code reruns no longer match the older archived return scale.

# 2026-06-08 - Return-scale mismatch resolved

1. Objective:
   - Investigate why the older archived finish-line result showed `4764.00%` return while the current BTC rerun showed `41.93%` with the same `332` trades.
2. Finding:
   - The trades and per-trade profit ratios are effectively identical.
   - The difference is stake allocation.
   - Old archive first trade stake: about `995.97 USDT`.
   - Current `max_open_trades = 10` rerun first trade stake: about `92.84 USDT`.
   - Old archive last trade stake: about `48807.63 USDT`.
   - Current `max_open_trades = 10` rerun last trade stake: about `87.63 USDT`.
3. Reproduction:
   - Reran current code with `--max-open-trades 1`.
   - Result exactly matched the old archive:
     - `332` trades,
     - `4764.00%` return,
     - `5.30%` drawdown,
     - `2.7548` profit factor,
     - first stake `995.96956`,
     - final balance `48639.95723963004`.
4. Evidence:
   - Old archive: `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-46-44.zip`
   - Reproduction archive: `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_23-18-51.zip`
   - Multi-position rerun archive: `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_22-48-55.zip`
5. Decision:
   - The old `4764.00%` result is not a signal mismatch or code drift.
   - It is a capital-allocation difference caused by how unlimited stake is divided when `max_open_trades` is higher than the number of BTC positions that can actually be opened.
   - Do not treat `max_open_trades` itself as strategy edge for BTC-only tests.
   - The production-relevant next step is explicit position management: initial stake reserve, same-direction stacking, partial exits, risk reductions, and later leverage/risk sizing.

# 2026-06-09 - BTC position-management component tests

1. Objective:
   - Move away from entry-only testing and trial production-relevant position management:
     - stake reserve,
     - same-direction stacking,
     - partial profit exits,
     - risk reductions when the run weakens,
     - combinations with the best clean-entry branch.
2. Code added:
   - `_CurrentBestPositionManagementBase`
   - `TraderRuleBlockBtcSieveCurrentBestPmBalancedStackPartialStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestPmAggressiveStackPartialStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestPmDefensivePartialStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestPmStackOnlyStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestPmProfitRunnerStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestPmRiskReduceOnlyStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmStackOnlyStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmRiskReduceStrategy`
3. Batches:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-08_23-58-47.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-09_00-03-28.zip`
4. Combined report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260609_position_management_combined_btc.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_20260609_position_management_combined_btc.csv`
5. Key result:
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy`
   - `327` trades, `4890.10%` return, `5.30%` drawdown, `2.83` profit factor.
   - Versus current best: about `+126.10` return points, essentially unchanged drawdown, improved profit factor.
6. Position-management result:
   - PM mechanics fired correctly:
     - balanced PM fired `78` stack adds, `89` first partial exits, `23` second partial exits, and `63` risk reductions.
     - defensive PM fired `64` stack adds, `97` first partial exits, `59` second partial exits, and `89` risk reductions.
   - The bundled PM variants reduced drawdown but cut too much winner exposure.
   - `PmProfitRunner` was closest to neutral:
     - `4759.10%` return versus base `4764.00%`,
     - `5.30%` drawdown,
     - `2.7549` profit factor.
7. Decision:
   - Promote clean-entry branch as the next main branch to develop.
   - Do not promote the first PM bundle.
   - Rework position management around later profit-runner partials and selective risk reduction, not early broad partial exits or broad reserve/add behaviour.

# 2026-06-09 - Exploratory finish-line checkpoint

1. Objective:
   - Launch broad existing-strategy exploratory batches before the next analysis pass.
   - Preserve this point in time so later agents can resume from the same evidence.
2. Scope:
   - BTC-only.
   - Timerange `20200101-20260115`.
   - `--max-open-trades 1`.
   - No Sieve or collector processes were stopped.
3. Result artifact:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\exploratory_batch_ranked_20260609.csv`
4. Key result:
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy`
   - `327` trades, `4890.10%` return, `5.30%` drawdown, `57.5%` win rate.
   - This is the current best raw BTC finish-line result from the broad exploratory pass.
5. Other useful lanes:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`: `4764.00%`, `5.30%` drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestPmProfitRunnerStrategy`: `4759.10%`, `5.30%` drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy`: `4607.96%`, `5.30%` drawdown, `59.8%` win rate.
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakAndPocShortStrategy`: `4528.92%`, `5.30%` drawdown, `59.0%` win rate.
6. Decision:
   - Preserve the top clean-entry lane as the near-term anchor.
   - Revisit older overtrading/high-drawdown lanes with better position protection before discarding them.
   - Next test should combine higher-risk entries with partial exits and profit-locking stop logic.

# 2026-06-09 - Generic TA/candle MTF feature-discovery branch

1. Objective:
   - Test whether broad generic indicators and candle-pattern relationships have useful signal before moving them into trader hypotheses or FreqAI strategy lanes.
   - Include BTC `1h`, `4h`, and `1d` timeframes.
2. Code added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\generic_ta_feature_discovery_builder.py`
3. Feature cache built:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_20260609_generic_ta_mtf_btc_debug.parquet`
   - `55,971` hourly rows from `2020-01-01` to `2026-05-22`.
   - `564` feature columns.
4. Feature families included:
   - RSI, EMA, SMA, TEMA, ATR, ROC, ADX/DI, CCI, MFI, Williams %R, OBV, ADOSC, TRIX, ULTOSC, CMO, NATR, PPO, MACD, Stoch, Bollinger bands.
   - TA-Lib candle-pattern scores and rolling candle-pattern counts.
   - Relationship/state features such as body pressure, volume z-scores, EMA crosses, MACD signal crosses, RSI reclaim/loss, Bollinger squeeze/cross, and 3-bar higher-high/lower-low structure.
5. Screening completed:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_20260609_generic_ta_mtf_btc.md`
   - `2,349` univariate scored rows.
   - `1,540` tree shortlist rows.
   - `676` stable shortlist rows.
6. Early read:
   - Strongest clusters were simple multi-timeframe momentum/body-pressure and volatility-risk states, especially `1d` and `4h`.
   - Best examples:
     - `1d body_pressure` high aligned with bullish `+3% before -2%` and breakout labels.
     - `1d body_pressure` low aligned with bearish `-3% before +2%`, breakdown, and drawdown labels.
     - `4h/1h RSI`, `4h/1h Williams %R`, `4h Stoch`, `1d/4h NATR`, and volume z-score features showed repeated stable rows.
   - Relationship/cross features existed, but EMA/Bollinger cross events did not lead the first shortlist.
7. Decision:
   - Promote this branch to hypothesis-writing next, but do not treat the large stable count as final proof.
   - Next step should convert the strongest generic TA clusters into a small number of trader-readable rules and test them as entries/exits/risk filters against current production-alpha anchors.

# 2026-06-10 - BTC dry-run candidate hardening

1. Objective:
   - Start from `Pre-Dry-Run Production Lane Selection`.
   - Take one best BTC lane to dry-run-ready state instead of continuing broad research.
2. Checkpoint added:
   - `C:\FreqTradeStuff\ai_guidance_docs\pre_dry_run_production_lane_selection_checkpoint.md`
3. Selected lane:
   - Source anchor: `TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy`
   - Dry-run wrapper: `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanDryRunBaseStrategy`
4. Code added:
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanProfitLockStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanTightProfitLockStrategy`
   - `_ConservativeDryRunLeverageMixin`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanDryRunBaseStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestOvertradeCleanDryRunCandidateStrategy`
5. Focused backtest pack:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-10_00-35-23.zip`
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_batch_summary_pre_dry_run_candidate_pack_20260610_0001.md`
6. Result:
   - Anchor: `327` trades, `4890.10%` return, `5.30%` drawdown, `2.83` profit factor.
   - Runner profit-lock: `329` trades, `3654.46%` return, `5.29%` drawdown, `2.91` profit factor.
   - Normal profit-lock: `332` trades, `2014.16%` return, `5.29%` drawdown, `2.52` profit factor.
   - Tight profit-lock: `343` trades, `618.46%` return, `26.02%` drawdown, `2.11` profit factor.
7. Decision:
   - Do not promote partial/profit-lock variants as the main lane because they cut too much return.
   - Promote the dry-run base wrapper, which preserves the anchor and adds only 1x leverage safety.
8. Dry-run config:
   - `C:\FreqTradeStuff\user_data\configs\config_btc_overtrade_clean_dry_run.example.json`
   - `dry_run: true`, empty keys, BTC-only, `stake_amount: 100`, `tradable_balance_ratio: 0.25`, isolated futures, strategy-level 1x leverage cap.
   - Smoke test passed on `20250101-20250201`.
9. Candidate note:
   - `C:\FreqTradeStuff\ai_guidance_docs\production_btc_dry_run_candidate_20260610.md`
