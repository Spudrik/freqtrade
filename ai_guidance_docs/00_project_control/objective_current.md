---
doc_status: active
default_read: yes
owner: user
purpose: Ephemeral current task contract. This is the active objective, not a historical log.
do_not_use_for: Static long-term objectives, old progress logs, or result dumps.
last_rebuilt: 2026-06-11
---

# Current Objective

## Active Objective

Use Sieve as the only approved Hyperopt/discovery system for entries, exits, position adjustment, and risk management unless the user explicitly approves a non-Sieve exception.

The previous non-Sieve exit/risk lane objective is retired. Do not resume it, route agents to it, or use its old strategy/report artifacts as active evidence.

## Core Direction

1. Sieve is the primary and default tool for Hyperopt batches.
2. Backtests are validation/sanity checks after Sieve identifies candidates, not the main discovery method.
3. FreqAI, confluence, orderbook, news/context, and generic-TA discoveries are parked as separate future feature/overlay sources.
4. Do not use parked FreqAI/orderbook/context material as live strategy logic unless the user explicitly starts a new objective for it.
5. Do not create new non-Sieve Hyperopt strategy packages for exit/risk, position management, or lane testing without explicit user approval.
6. If a future Sieve strategy needs orderbook/confluence/news data, import only the specific proven feature source and keep it clearly separated from Sieve's own Hyperopt workflow.
7. Current active strategy-folder work is Sieve3 exit development. Active Sieve3 files live as top-level `user_data/strategies/sieve3*.py` files; old generated/candidate Sieve folders are not active working folders.
8. Sieve3 pass/fail has no fixed rule threshold yet. Rejection and promotion are user-discretionary: rejected files move to `Archive/`, and passed files move to a future successful-strategy folder when the user decides.
9. The current campaign goal is to Hyperopt all active top-level Sieve3 files across multiple windows and multiple random seeds, then report results for user review.
10. Temporary backlog-clearing guidance: while clearing the current top-level Sieve3 exit backlog, files that produce fewer than `10` validation trades across multiple windows should be parked in `user_data/strategies/sieve3_low_trade_long_window_retest/` for later longer-window testing. This is not a permanent Sieve3 pass/fail rule and does not mark those files rejected or passed.

## Required Rule Files

- `../02_rules/rules_codebase_workflow.md`
- `../02_rules/rules_runtime_environment.md`
- `../02_rules/rules_hyperopt_general.md`
- `../02_rules/rules_entry_exit_position.md`
- `../02_rules/rules_sieve3_exit_hyperopt.md`
- `../02_rules/rules_sieve3_exit_regeneration_agent.md`
- `../02_rules/rules_strategy_success.md`
- `../02_rules/rules_result_reporting.md`
- `../02_rules/rules_document_maintenance.md`

## Optional Rule Files

Read only if the user explicitly asks to revisit these parked systems:

- `../02_rules/rules_orderbook_sources.md`
- `../02_rules/rules_news_context_sources.md`
- `../02_rules/rules_freqai_feature_discovery.md`
- `../02_rules/rules_freqai_promotion.md`

## Ask The User Before Proceeding If

1. The task would create or run a non-Sieve Hyperopt package.
2. The task would revive retired non-Sieve exit/risk lane work.
3. The task would use FreqAI/orderbook/news/context as active trading logic.
4. The task would delete raw data, archives, or canonical feature stores.
5. The task would touch live trading settings.
6. The task would alter dry-run settings, stake, leverage, exchange, or pair scope.

## Retired Work

The recent non-Sieve exit/risk lane objective was retired because it produced limited additional value versus Sieve while consuming excessive agent context and tokens. Keep only the general lesson:

> Sieve should own Hyperopt/discovery. Park FreqAI/orderbook/context/confluence work as later optional overlays after strong Sieve strategies exist.
