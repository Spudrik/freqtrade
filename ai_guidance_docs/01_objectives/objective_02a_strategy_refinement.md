---
doc_status: active
default_read: routed
owner: user+agent
purpose: Objective routing contract for Sieve candidate refinement.
do_not_use_for: Detailed implementation history.
last_rebuilt: 2026-06-11
---

# Objective 02a - Strategy Refinement

## Purpose

Refine existing high-potential Sieve candidates while keeping multiple paths alive.

## In Scope

1. Improve one Sieve candidate/search family at a time.
2. Compare every variant against that Sieve search family's own anchor.
3. Preserve alternative candidates that optimize different purposes: max return, balanced risk, lower drawdown, orderbook-risk overlay, sparse high quality, position management.
4. Audit failure modes by family, month, side, exit reason, and market context.
5. Promote ideas to the master architecture only when they improve a defined role.

## Out Of Scope

1. Do not erase a candidate because another candidate has higher return.
2. Do not merge candidates just because both are profitable.
3. Do not count raw feature-discovery rows as finished trading leads.
4. Do not treat `current best` as the whole project.
5. Do not revive retired non-Sieve exit/risk lane work.

## Required Rules

- `../02_rules/rules_hyperopt_general.md`
- `../02_rules/rules_strategy_success.md`
- `../02_rules/rules_direct_tests.md`
- `../02_rules/rules_result_reporting.md`
- `../02_rules/rules_codebase_workflow.md`

## Required Status / Results

- Current Sieve result files named by the active run and strategy stage.
- `../04_results/results_recent_summary.md`

## Completion Criteria For A Refinement

1. Baseline/anchor named.
2. Variant metrics compared against the correct anchor.
3. Trade count, return, drawdown, profit factor, win rate, long/short split, and same-window market comparison considered where available.
4. Failure-mode impact described in trader language.
5. Verdict is one of: promote, keep as candidate, park for rework, reject current form.
