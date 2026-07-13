---
doc_status: active
default_read: routed
owner: user
purpose: Rules for exit and risk research.
do_not_use_for: Entry discovery, Sieve generation, or broad management wording.
last_rebuilt: 2026-06-11
---

# Exit And Risk Research Rules

## Scope

This rule doc is only for exit logic and risk research:

1. Exit modes.
2. Partial exits.
3. Stop tightening.
4. Adaptive trailing.
5. Same-direction adds.
6. Opposite-direction reductions.
7. Stake sizing.
8. Leverage tiers.
9. Exposure caps and risk-off behaviour.

## Method

1. Use Sieve as the default and only approved Hyperopt/discovery system for explicit exit/risk theory search.
2. Use FreqAI, orderbook, or context data only as parked future input sources unless the user explicitly approves them for the current Sieve objective.
3. Do not use manual one-off backtests to declare an exit/risk concept dead.
4. Do not run ordinary backtests unless the user approves or the current objective explicitly permits it.
5. Treat backtests as sanity/validation after Sieve discovery, not as the main testing engine.

## Strategy Shape

1. Build focused Sieve searches or Sieve-compatible strategy files.
2. Encode explicit theories as named categorical modes.
3. Keep each strategy narrow enough that Hyperopt can resolve the search.
4. Do not create separate non-Sieve research packages unless the user explicitly approves an exception.
5. Every major entry tag/family should be allowed its own candidate exit/risk logic where feasible.
6. Sieve3 exit regeneration must follow `rules_sieve3_exit_regeneration_agent.md` for file splitting, categorical branch design, trigger/guard/PnL state, standalone strategy modules, no-lookahead handling, and valid Freqtrade method usage.
7. Keep ordinary branch files near `5-10` active HyperOpt parameters unless compact combined categorical plans keep the result interpretable.
8. Do not use wide decimal sweeps or broad Cartesian products where named categorical plans answer the same trader question.

## Exit / Risk Branches

Test these as mode families:

1. Fixed target exits.
2. VP target exits based on POC, VAH, VAL, node hold/exit, and value acceptance/rejection.
3. TLV2 target exits based on support/resistance, break/retest, and next-level interaction.
4. Pivot target exits based on confirmed pivot high/low, prominence, and side-aware high/low logic.
5. BOS/CHoCH exits based on opposite structure event, structure-state flip, and swing invalidation.
6. Pattern target exits split by Geometry V2, reversal, continuation, multi-peak, and Wolfe contracts where columns exist.
7. Invalidation exits.
8. Partial at first obstacle.
9. Partial after profit threshold.
10. Partial on opposing signal.
11. Partial or stop tightening on trigger damage.
12. Partial or stop tightening on guard flip, based on current PnL and partial state.
13. Breakeven after target touch.
14. Structure, VP, TLV2, swing, and volatility-based trailing.
15. Stop tightening during crash/risk-off states.
16. Add on same-family confirmation.
17. Add on independent-family confirmation.
18. Reduce on opposite-direction confluence.
19. Stake by confluence strength.
20. Reduce stake in chop or risk-off states.
21. Leverage only after a stable rule family is proven.

Orderbook, FreqAI, news/context, and broader confluence sources are parked future inputs unless the active objective explicitly routes them into Sieve. Do not use them as active exit evidence in current Sieve3 regeneration merely because older examples mention them.

## State And Implementation Guardrails

1. Exit actions should account for primary trigger, primary guard, target provider, invalidation provider, current PnL bucket, target touch state, and partial state.
2. Opposing evidence should not blindly force a full exit. Test hold, tighten, reduce, partial, lock, trail, and full exit as coherent categorical action plans.
3. Indicator target exits must use explicit source-family mappings. Do not keyword-scan columns for likely targets.
4. Use only levels available at the decision candle. Do not use future-confirmed pivots, retroactive pattern targets, or future candle scans.
5. Claimed partial exits, stop shifts, and trailing modes must use valid local Freqtrade method patterns.
6. Generated Sieve3 strategy files must remain standalone HyperOpt modules.

## Classification

Use these labels:

1. `hyperopt_promising`: strong candidate from train window.
2. `needs_holdout`: promising but not yet checked out of sample.
3. `propose_backtest`: ready to ask the user for validation backtest approval.
4. `park`: interesting but not current-best.
5. `reject`: failed after the allowed search attempts.

## Reporting

Report in trader-readable terms:

1. What the mode was trying to do.
2. What it changed in exits/risk.
3. Whether it improved return, drawdown, trade count, win rate, or profit factor.
4. Whether it is broad, specialist, or only useful for one entry family.
5. Whether it should continue, hold out, be proposed for backtest, be parked, or be rejected.
