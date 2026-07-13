---
doc_status: active
default_read: routed
owner: user+agent
purpose: Strict rules for promoting named hypotheses into FreqAI queues.
do_not_use_for: Loose feature discovery or strategy promotion.
last_rebuilt: 2026-06-10
---


# Rules - FreqAI Promotion

## Promotion principle

FreqAI promotion is not feature fishing. A candidate must be a named trader-readable hypothesis that survived data checks, direct tests, controls, baselines, and ablations.

Use `rules_freqai_feature_discovery.md` for loose early exploration. Use this file only when deciding whether a named hypothesis is ready for a FreqAI queue.

## Required before queueing

1. Hypothesis ID exists in the hypothesis registry or explicit test manifest.
2. Trader theory, direction, setup, trigger, score, target, and required source-detail groups are defined before queueing.
3. Frozen 1h snapshot or equivalent dataset is used.
4. Current column dictionary exists or a run-specific feature manifest maps columns to source-detail groups.
5. Duplicate dates: `0`.
6. Source future violations: `0` for required sources.
7. Infinite numeric cells: `0`.
8. Required source groups have real coverage in the selected window.
9. Missing source data remains missing and is not converted into quiet/balanced states.
10. The direct-test report row for the exact selected threshold settings exists.

## Direct-test minimums before FreqAI

Use the metric meanings in `rules_direct_tests.md`. A metric only supports promotion when it matches the trader story, beats controls, and survives the relevant data-readiness checks.

For binary targets, preferred minimums:

1. AUC `>= 0.55`.
2. AUC beats shuffled-label AUC by at least `0.02`.
3. AUC beats price/structure baseline by at least `0.01`; preferred `0.02`.
4. Trigger event rate beats same-regime and random controls by at least `0.03` absolute.
5. At least half of valid monthly windows have AUC `>= 0.55`, with a minimum of two positive windows.
6. Setup rows, trigger rows, and positive examples are adequate for the event type. If sample size is small, mark as `conditional_pilot_pending_preflight`, not full queue-ready.
7. Average precision must support the AUC story on rare-event labels. A rare-event result with weak AP is not a strong promotion candidate even if AUC looks acceptable.
8. Top bucket should have a materially better actual event rate than ordinary/control rows; bottom bucket should be clearly worse or lower-risk where applicable.
9. Event-rate lift should be explainable in trader terms, not only statistically positive.

For continuous/path targets:

1. Oriented top-bucket lift must be material and in the expected direction.
2. Spearman correlation and p-value should be reported where applicable. Treat p-value as a noise warning, not proof.
3. Same-regime and random-control means must be reported.
4. Price/structure baseline lift must be reported.
5. Monthly/rolling stability should be reported where enough rows exist.
6. Continuous/path results must say whether the score helps with target selection, stop tightening, add/reduce decisions, drawdown warning, or continuation ranking.

## Threshold-sweep promotion rule

A threshold-sweep result is promotion-ready only if:

1. the exact selected thresholds are reported,
2. nearby threshold settings are not all failures,
3. multi-component confluence is preserved where the hypothesis requires confluence,
4. setup/trigger sample sizes remain adequate,
5. controls and baselines still pass at the selected threshold,
6. the trader explanation remains sensible after thresholding.

## Required preflight record

Each queued experiment needs:

1. date range,
2. source-detail groups,
3. setup rows,
4. trigger rows,
5. positive target examples,
6. active rows,
7. nonzero rows,
8. missing rows,
9. low-coverage rows,
10. future-violation rows,
11. late-available-at rows where relevant,
12. max source age,
13. clean-window class: full confluence, structure+orderbook, orderbook-only, GDELT-only, context-source-specific, or missing-data/control.

## Source-specific warnings

1. Historical GDELT coverage is not equivalent to live news/web/global/Trends/ETF coverage.
2. Multi-venue orderbook hypotheses require the exact venue groups used by the features.
3. Do not carry orderbook state through missing archive gaps.
4. Do not call a window full confluence if a required source block is absent.
5. Do not promote source-family labels such as `context` or `orderbook` without naming source-detail groups.

## Metric misuse warnings

1. Do not promote a result from AUC alone.
2. Do not ignore average precision on rare events.
3. Do not claim a top-bucket edge unless the actual future outcomes in that bucket improved versus controls.
4. Do not treat one good threshold as robust unless neighbouring thresholds also work or there is a clear trader reason for a sharp threshold.
5. Do not treat one event window as general unless the result is explicitly labelled event-window-specific.
6. Do not promote a model that beats random but fails the price/structure baseline, unless the objective is explicitly to create a non-price diagnostic feature rather than an entry/risk rule.

## Output verdicts

Use one of:

- `queue_ready`,
- `conditional_pilot_pending_preflight`,
- `needs_direct_test`,
- `deferred_for_data`,
- `rejected_for_now`.
