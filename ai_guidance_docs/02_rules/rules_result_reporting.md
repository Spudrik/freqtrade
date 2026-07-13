---
doc_status: active
default_read: routed
owner: user+agent
purpose: Plain-English result reporting standard.
do_not_use_for: Storing full history.
last_rebuilt: 2026-06-25
---

# Rules - Result Reporting

## Required shape

Every result should answer:

1. Trader question.
2. What we looked at.
3. What we asked afterwards.
4. Result in plain English.
5. Numbers that support it.
6. Verdict.
7. Next step.

## Plain translations

- Feature: something known at that hour.
- Label/target: what happened after that hour.
- Event scope: only rows where this market situation was active.
- AUC: how well the score ranked good outcomes above bad outcomes.
- Top bucket: rows the model liked most.

## Avoid

1. Do not lead with profile names, AUC, AP, model class, or internal columns.
2. Do not say “post-break label” without explaining the trader event.
3. Do not call a result good/bad without saying what a trader saw.
4. Do not collapse complex confluence into one indicator/source/timeframe.
5. Do not hide sample size or controls.
6. Do not compare high-exposure legacy results with bounded-stake results without stating the exposure difference.

## Strategy result fields

When reporting Hyperopt/backtest strategy results, include these fields where available:

1. Method: Hyperopt, safety check, direct test, FreqAI, or backtest.
2. Timerange and timeframe.
3. Pair scope.
4. Stake/exposure settings, including wallet reference, max open trades, and whether the run used legacy high exposure.
5. Trade count.
6. Long/short count.
7. Win/draw/loss count and win rate.
8. Profit and objective/loss score.
9. Drawdown.
10. Buy-and-hold comparison when available.
11. Exact result ZIP, CSV, log, or Hyperopt result path for any claim.
12. Normalized return: report profit as return on the configured test wallet/exposure, and state wallet, stake mode, max open trades, pair scope, and leverage assumptions. Do not compare raw profit values across runs with different exposure settings without normalizing or flagging the difference.

## Sieve3 exit result analysis

When reporting Sieve3 exit results, follow the specific interpretation rules in `rules_sieve3_exit_hyperopt.md`.

Default Sieve3 exit analysis should group by entry/source condition first, then by exit family, and should select the best representative row per entry/source plus meaningful alternates. Do not lead with a raw top-N table when repeated variants of the same entry or exit family dominate the list.

Use averages, medians, hit-rates, and percentiles only as secondary diagnostics. They can highlight entries that work across many exit conditions or exit families that work across many entries, but they do not replace row-level evidence. Specific rows with their seed, window, selected params, profit, drawdown, winrate, and trade count remain the primary evidence.

Flag suspicious duplicates clearly: same strategy, seed, and training window should not produce conflicting metrics unless selected params or runtime inputs differ. Same strategy and seed across different training windows is valid, but the windows must be shown.


## Metric translation

When reporting research metrics, translate them before giving numbers:

1. AUC: “how well the score ranked good outcomes above bad outcomes.”
2. Average precision: “whether the rows ranked highest actually contained more rare good/bad events.”
3. Top-bucket lift: “whether the rows the score liked most actually behaved better than ordinary rows.”
4. Spearman: “whether stronger signal values generally matched stronger future outcomes.”
5. Threshold sweep: “whether the idea worked across a sensible range of settings, not one lucky cutoff.”
6. Setup/trigger/positive counts: “how many real examples the claim is based on.”

Do not lead with metric acronyms. Start with the trader question and what happened.
