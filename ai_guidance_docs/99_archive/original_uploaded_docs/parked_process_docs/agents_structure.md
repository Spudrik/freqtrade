# Agent Structure For Trader-Confluence Research

This document is for orchestrator agents that manage other agents or split confluence research work. Do not require every coding agent to read it.

## Orchestrator Responsibilities

1. Read `current_objectives.md`, `objective_examples.md`, and this file before splitting work.
2. Keep objective control separate from progress:
   - objectives: `current_objectives.md`
   - progress/status: `objectives_progress.md`
   - broad results: `objectives_high_level_results_summary.md`
   - detailed results: `context_research_detailed_findings.md`
3. Assign disjoint work scopes to agents.
4. Prevent agents from reducing a complex confluence idea to one source, one indicator, or one timeframe.
5. Require every hypothesis to state the trader-readable behaviour first, then the numeric encoding.

## Recommended Agent Roles

1. Data Readiness Agent
   - Validate parquet/data sources, date ranges, gaps, duplicate timestamps, all-null columns, source freshness, and missing-data flags.

2. Column Dictionary Agent
   - Map dataframe columns to source, timeframe, meaning, safe timestamp basis, and whether the column is raw, derived, or composite.

3. Trader Concept Agent
   - Convert trader ideas into named hypotheses with setup logic, trigger logic, targets, and expected direction.

4. Feature Builder Agent
   - Build numeric state/transition/confluence features from validated inputs.

5. Logic Review Agent
   - Review whether each feature and hypothesis actually matches the intended market behaviour.

6. Code Review Agent
   - Review implementation correctness, timestamp safety, missing-data handling, and no-lookahead behaviour.

7. Direct Test Agent
   - Run direct tests, controls, ablations, sample-size checks, and month/window stability checks before FreqAI.

8. FreqAI Queue Agent
   - Promote only promising hypotheses into queued FreqAI profiles with baselines, ablations, and consistent reports.

9. Results Review Agent
   - Summarise what worked, what failed, what is invalid, and what should be refined or retired.

## Research Control Loop

1. Define hypothesis in plain trader language.
2. Identify required columns and source families.
3. Build setup mask and trigger mask.
4. Review the logic against trader intent.
5. Run direct tests.
6. Run controls:
   - random rows
   - same regime without trigger
   - shuffled labels
   - opposite-direction setup
7. Run ablations:
   - full confluence
   - minus orderbook
   - minus news/context
   - minus structure/custom indicators
   - minus volume
   - minus multi-timeframe confirmation
8. Promote to FreqAI only if direct evidence is useful.
9. Review feature importance or coefficient direction grouped by source family.
10. Update progress and result docs, not `current_objectives.md`.
11. Refine, repeat, or retire the hypothesis.

## Promotion Rules

Promote a hypothesis from direct testing to FreqAI only when it has:

1. Enough rows to avoid obvious small-sample noise.
2. A defined baseline/control.
3. Positive top-vs-bottom bucket separation or better event rate than controls.
4. No timestamp/lookahead violation.
5. No hidden forward-fill through missing data.
6. A stable enough result across more than one window, or a documented reason it is event-window-specific.
7. A plain-English reason why the behaviour could matter.

## Result Reporting Standard

Every report should include:

1. Hypothesis.
2. Exact columns used.
3. Setup logic.
4. Trigger logic.
5. Target.
6. Sample size and event rate.
7. AUC/AP/correlation where applicable.
8. Top/bottom bucket result.
9. Control and ablation outcome.
10. Stability across windows.
11. Good/bad/worth-investigating verdict.
12. Next action.
