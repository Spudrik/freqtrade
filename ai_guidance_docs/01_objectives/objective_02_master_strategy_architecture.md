---
doc_status: active
default_read: routed
owner: user+agent
purpose: Router for the wider master strategy architecture objective.
do_not_use_for: Detailed implementation history or single-candidate tuning.
last_rebuilt: 2026-06-11
---

# Objective 02 - Master Strategy Architecture

## Purpose

Route work for the wider Sieve-first strategy architecture: many entry families, confluence/conflict assessment, entry-specific exits, position management, and later data-source integration.

This file is a router. Do not treat it as a full implementation plan. Open the matching sub-objective and rule files for the actual task.

## Target Architecture Summary

1. Multiple valid entry families can coexist.
2. Each entry should have a clear reason and, where possible, target/invalidation levels.
3. Same-direction signals from independent families/sources can support add/hold/confidence.
4. Opposite-direction signals can support reduce/tighten/exit.
5. Rare high-success custom pattern entries should be preserved and used as entries/adds/exits depending on direction.
6. Sieve is the only approved Hyperopt/discovery system unless the user explicitly approves a non-Sieve exception.
7. Orderbook can later refine target zones, confirmation, wall/resistance/support behaviour, liquidity-path evidence, and invalidation when explicitly routed into Sieve work.
8. News/context can later provide market-regime/crash/risk-off confirmation only once data is ready and explicitly routed.
9. The master system should maintain a pool of high-potential branches instead of obsessing over one current best.

## Sub-Objective Router

| Work type | Read |
|---|---|
| Existing Sieve candidate refinement | `objective_02a_strategy_refinement.md` |
| Sieve3 exits/adds/reductions/risk | `../00_project_control/objective_current.md` plus routed Sieve3 exit rules |
| Orderbook confluence/targets/risk | `objective_02c_orderbook_confluence.md` |
| News/context integration | `objective_02d_news_context_integration.md` |
| Generic TA side-lane | `objective_02e_generic_ta_integration.md` |

## Required Rules For Broad Objective 2 Work

- `../02_rules/rules_hyperopt_general.md`
- `../02_rules/rules_direct_tests.md`
- `../02_rules/rules_strategy_success.md`
- `../02_rules/rules_entry_exit_position.md`
- `../02_rules/rules_result_reporting.md`

Add task-specific rules as needed:

- `../02_rules/rules_orderbook_sources.md`
- `../02_rules/rules_news_context_sources.md`
- `../02_rules/rules_source_detail_taxonomy.md`
- `../02_rules/rules_freqai_feature_discovery.md`
- `../02_rules/rules_freqai_promotion.md`

## Research Cycle Shape

Keep multiple concept paths alive. A normal broad Sieve research cycle should include roughly:

1. Structure/volume: `3-5` concepts.
2. Downside risk/exhaustion: `3-5` concepts.
3. Orderbook state: `3-5` concepts only when explicitly routed and data-ready.
4. Regime/confluence gate: `3-5` concepts.
5. New feature-discovery ideas: `2-4` concepts.

These are quotas for breadth, not hard promotion targets. Do not let one promising idea consume the whole process unless the user explicitly narrows the objective.

## Required Ledgers / Results

- Current Sieve batch/result outputs named by the active objective.
- `../04_results/hypothesis_ledger.csv` only when adding/checking test history.

## Non-Goals

1. Do not make a broad monolithic exit that damages all entry families.
2. Do not use news/context before readiness is proven.
3. Do not collapse rare pattern entries into vague names.
4. Do not run opaque FreqAI fishing as a substitute for trader-readable hypotheses.
5. Do not create non-Sieve Hyperopt packages unless explicitly approved.
