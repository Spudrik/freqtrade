# Concept Lifecycle Research Process

This is the operating document for iterative trader-confluence research. Agents should use this process when the user asks to find useful trading information from indicators, price/volume, orderbook, news/context, macro, FreqAI, or direct tests.

## Purpose

1. Find at least 20 trader-readable concepts that show repeatable promise.
2. Avoid getting trapped in one early winner.
3. Avoid random opaque FreqAI runs.
4. Convert scattered reports, queues, and test outputs into one formal tracking structure.
5. Keep the research adaptive: test, classify, park, rework, add new ideas, repeat.

## Core Rule

Every test must answer a trader question first.

Good:

1. "If price breaks range high with rising volume pressure, does it usually continue?"
2. "If support is broken but quickly reclaimed, does downside risk fade?"
3. "If orderbook support disappears and downside liquidity is thin, does drawdown risk increase?"

Bad:

1. "Run all features and see what happens."
2. "AUC was 0.61, therefore this is good."
3. "Context performed well" without naming the source block and behaviour.

## Position Management Note

Do not judge overlapping leads only as fresh entries. Same-direction overlap may be useful add/hold/confidence evidence; opposite-direction overlap may be useful reduce/tighten/exit evidence. Treat this as a testable signal-weighting layer, not a fixed rule.

## Concept Card

Each concept must be tracked as a card before it is promoted.

Required fields:

1. `concept_id`
   - Stable snake_case name.
2. `branch`
   - One of: `structure_volume`, `downside_risk`, `orderbook_state`, `regime_confluence`, `news_context`, `macro_global`, `merged_confluence`.
3. `trader_question`
   - Plain-English question the concept answers.
4. `visible_market_state`
   - What a trader would see at the decision hour.
5. `source_detail_blocks`
   - Exact inputs used, such as VP, TLV2, BOS/CHoCH, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook, GDELT events, GKG documents, Google Trends, macro/global.
6. `feature_columns`
   - Exact dataframe columns or compact feature groups used.
7. `target_label`
   - What is asked afterwards, such as breakout success next 6h, support reclaim next 6h, drawdown next 24h.
8. `controls`
   - Random rows, same-regime rows, shuffled labels, opposite-direction setups, price-only baseline, and missing-ingredient ablations where relevant.
9. `latest_evidence`
   - Direct-test report, threshold sweep, FreqAI queue result, or feature-discovery report path.
10. `status`
   - Use the lifecycle statuses below.
11. `next_action`
   - Promote, rework, park, retest, defer for data, or reject for now.

## Lifecycle Statuses

1. `idea`
   - Trader-readable concept exists but has not been tested.
2. `direct_promising`
   - Direct test beats controls and makes market sense.
3. `sweep_promising`
   - Threshold sweep shows more than one viable parameter setting.
4. `freqai_promising`
   - FreqAI improves over the relevant baseline/control.
5. `parked_working`
   - Looks useful but should not consume every future cycle.
6. `needs_rework`
   - Failed or unclear, but has a specific repair path.
7. `deferred_for_data`
   - Cannot be tested honestly until source coverage improves.
8. `rejected_for_now`
   - Failed controls, weak sample size, no trader logic, or likely lookahead/noise.
9. `candidate_for_strategy_research`
   - Has enough evidence to explore as a rule, filter, risk flag, or FreqAI input.

## Branch Quotas

Each research cycle should keep multiple paths alive. Do not let one promising idea consume the whole process.

Minimum cycle shape:

1. Structure/volume: 3 to 5 concepts.
2. Downside risk/exhaustion: 3 to 5 concepts.
3. Orderbook state: 3 to 5 concepts.
4. Regime/confluence gate: 3 to 5 concepts.
5. New feature-discovery ideas: 2 to 4 concepts.

News/context and macro/global should be included only in windows where source coverage and timestamp safety are proven.

## Iteration Loop

1. Freeze usable data.
   - Use validated parquet snapshots.
   - Do not read live SQLite collectors during research runs unless explicitly testing source readiness.
2. Build or update concept cards.
   - Include existing winners, rework candidates, and new ideas.
3. Run direct tests first.
   - Use random, same-regime, shuffled-label, opposite-direction, and price-only controls.
4. Classify results.
   - Promote, park, rework, defer, or reject.
5. Sweep thresholds only for concepts with direct-test promise.
   - Do not sweep concepts that failed basic controls.
6. Queue FreqAI only for concepts that survive direct tests and threshold review.
   - FreqAI validates and ranks; it is not the first fishing stage.
7. Update the formal concept ledger.
   - Add status, evidence paths, verdict, and next action.
8. Consolidate old outputs into the ledger.
   - Reference old reports/queues by path.
   - Do not delete or move generated reports without explicit user approval.
9. Start the next cycle.
   - Keep the best working ideas parked.
   - Rework some failures.
   - Add new concepts.
   - Continue until at least 20 concepts are `direct_promising` or better.

## Pass Rules

A concept can be promoted when most of these are true:

1. It beats random rows.
2. It beats shuffled labels.
3. It beats same-regime rows without the trigger.
4. It beats or adds to a price/structure baseline.
5. It has enough rows to be meaningful.
6. It works across multiple months/windows.
7. It survives source-detail coverage checks.
8. It makes trader sense in plain English.

## Rework Rules

1. Too few rows:
   - Loosen the trigger.
   - Test the strongest component separately.
   - Use a wider clean window.
   - Park if data is genuinely missing.
2. Good direct result, weak FreqAI:
   - Treat the concept as a rule/gate first.
   - Try component-only versus gated features.
   - Do not discard the concept solely because FreqAI did not use it well.
3. Wrong direction:
   - Flip the trader question.
   - A failed bullish continuation may be a useful rejection/risk signal.
4. One-window result:
   - Retest on another clean window.
   - Park until more data exists if no fair window is available.
5. Feature too opaque:
   - Replace it with trader-readable state columns.
   - Report exact source-detail blocks and column meanings.
6. Context/news source weak or incomplete:
   - Split by source.
   - Defer until extraction and coverage are improved.

## Formal Tracking Structure

Use one machine-readable lifecycle ledger as the primary tracker once implemented:

`C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_lifecycle_ledger.csv`

Recommended columns:

1. `concept_id`
2. `branch`
3. `status`
4. `trader_question`
5. `visible_market_state`
6. `source_detail_blocks`
7. `feature_columns`
8. `target_label`
9. `test_window`
10. `rows`
11. `trigger_rows`
12. `direct_result`
13. `baseline_result`
14. `control_result`
15. `freqai_result`
16. `stability_result`
17. `verdict`
18. `next_action`
19. `evidence_paths`
20. `last_updated`

Use one concise Markdown summary for human review:

`C:\FreqTradeStuff\ai_guidance_docs\objectives_high_level_results_summary.md`

Use detailed findings for nuance:

`C:\FreqTradeStuff\ai_guidance_docs\context_research_detailed_findings.md`

## Old Report And Queue Cleanup

The existing reports and queues are evidence, not the primary tracking structure.

Cleanup means:

1. Inventory old artifacts:
   - `context_features\reports\*.csv`
   - `context_features\reports\*.md`
   - `context_features\freqai_queue\queue_*\freqai_experiment_queue.json`
   - `context_features\feature_discovery\*`
   - `context_features\freqai_runs\*`
2. Extract the important result into the concept lifecycle ledger.
3. Store the original file path in `evidence_paths`.
4. Mark old queue/report status as:
   - imported
   - still_needed
   - duplicate_evidence
   - obsolete_but_keep
   - candidate_for_user_approved_cleanup
5. Do not delete old artifacts automatically.
6. Only propose deletion/archival after:
   - the result is represented in the lifecycle ledger
   - the file is not needed by a current queue
   - the user explicitly approves cleanup

## First Seed Concept Pool

Use these as the first concept pool to classify into the ledger:

1. Range high breakout continuation.
2. Range low breakdown continuation.
3. Compression breakout release.
4. Compression breakdown release.
5. Structure plus orderbook breakout confluence.
6. Structure plus orderbook drawdown confluence.
7. Support reclaim after first breakdown.
8. Downside first-break exhaustion.
9. VAH acceptance.
10. VAH rejection.
11. VAL breakdown.
12. VAL reclaim.
13. Volume pressure breakout.
14. Volume pressure breakdown.
15. TLV2 resistance break.
16. TLV2 support break.
17. Orderbook resistance removed before breakout.
18. Orderbook support removed before breakdown.
19. Liquidity vacuum above after resistance removal.
20. Liquidity vacuum below after support removal.

These are seed concepts, not all proven concepts. The process should promote, rework, defer, or reject them based on evidence.

## Agent Operating Rule

When the user asks to continue the objective, find correlations, run a new test cycle, or build toward useful trading information:

1. Read this document.
2. Read `current_objectives.md`.
3. Read `objective_examples.md`.
4. Read `result_communication_format.md`.
5. Work from the lifecycle ledger and current source coverage, not chat memory.
6. Do not claim a concept is useful unless the evidence path and controls are recorded.
