---
doc_status: active
default_read: routed
owner: user+agent
purpose: Strict rules for promoting named hypotheses into FreqAI queues.
do_not_use_for: Loose feature discovery or strategy promotion.
last_rebuilt: 2026-09-09
---


# Rules - FreqAI Promotion

## Promotion principle

FreqAI promotion is not feature fishing. A candidate must be a named trader-readable hypothesis that survived data checks, direct tests, controls, baselines, and ablations.

Use `rules_freqai_feature_discovery.md` for loose early exploration. Use this file only when deciding whether a named hypothesis is ready for a FreqAI queue.

Objective 02b Section 15.4, revised by the user on 19 September 2026, allows components
to qualify for direction, activity, location or conditional roles separately; a
mandatory joint reaction-and-direction floor no longer gates every candidate. This
does not change the queue-specific AUC thresholds below or bypass direct controls,
queue preflight, portability, or later strategy and risk validation. A proposed
combination must be evaluated for its own named role, not assumed to inherit the
components' success rates.

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
11. The intended market scope is frozen as broad top-coin, coin-group-specific, or asset-specific, with exact eligible pairs and independent grouping rationale.
12. For event work, the hypothesis names the exact link, pair, or maximum three-block
    chain in the event hierarchy rather than submitting every available feature.
13. Event selection is independent of later price movement, and scheduled and
    unexpected events are kept as separate evidence groups.
14. Development, validation, and untouched confirmation are split by whole event;
    observations from one event never cross those boundaries.
15. The call/abstention rule is frozen and its expected coverage is reported.
16. Accepted market facts are listed so the queue does not retest an established
    premise.
17. Every input block has a decision-time role: background, driver/trigger,
    confirmation, modifier, accumulator, duplicate/substitute, or outcome.
18. Raw outcome and change from a matched expected path are defined separately.
19. The hypothesis states whether it claims independent prediction, later
    confirmation, or a conditional change in another influence.

## Market-Scope Promotion Rule

1. A candidate that fails a broad top-coin threshold may still proceed as a coin-group or asset-specific candidate when that narrower scope was predeclared or independently confirmed on later unseen data.
2. A group selected because its members happened to be positive in the same evaluated result is not promotion evidence. Freeze it as a new hypothesis and repeat it later.
3. A group-specific candidate must report every member, adequate support per member, chronological repetition, non-member comparison where relevant, and whether one member dominates.
4. A BTC-only or other asset-specific candidate must remain restricted to that asset and reproduce across multiple windows and later unseen data.
5. Promotion thresholds apply inside the declared eligible scope; they do not convert a specialist result into a general crypto claim.
6. A meme-cohort candidate is restricted to the exact frozen `10` most traded eligible
   meme coins ranked under Objective 02b's pre-outcome turnover and coverage contract.

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

For event-driven targets, monthly stability does not substitute for independent-event
repetition. Report results per whole event, check whether one event dominates, compare
BTC, ETH, and broad-market leadership where relevant, and reserve untouched events for
the final confirmation layer.

## Conditional-Interaction Promotion Rule

A factor does not need a strong unconditional average when its declared role is to
modify another influence. Such a candidate may enter a bounded FreqAI queue only when:

1. the primary influence and proposed modifier are trader-readable and timestamp-safe;
2. the modifier role is named as amplification, suppression, reversal, gating,
   accumulation, override, delay, shortening, or transmission blocking;
3. development evidence compares the primary influence with and without the modifier
   on comparable eligible rows, including neither/A-only/B-only/both cells where
   support permits;
4. the context definition is frozen before validation and untouched whole-event
   confirmation;
5. activation frequency, cell support, conflicting episodes, and concentration are
   reported;
6. duplicated stories, correlated indicators, adjacent timeframes, and multiple coins
   inside one event are not counted as independent components or confirmations;
7. post-event confirmation is not presented as a rival root cause or pre-event
   prediction; and
8. FreqAI feature importance is treated only as a discovery clue and the conditional
   relationship survives readable summaries, block removal, and later data.

For an independent or additive forecast claim, retain the normal requirement to beat
the relevant simpler models. For a modifier claim, require reproducible change in the
primary relationship inside the frozen context; do not reject it merely because the
modifier alone has weak AUC or correlation.

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
7. Do not count many candles, coins, or forecast horizons from one event as many
   independent confirmations.
8. Do not improve apparent accuracy by silently discarding abstentions or shrinking
   coverage after seeing outcomes.
9. Do not control on a market response caused after an event when estimating the
   event's total influence.
10. Do not treat a final negative candle as proof that positive information had no
    effect, or vice versa; compare the complete path and matched expected outcome.
11. Do not promote a conditional story that was invented after opening confirmation
    outcomes. Development-discovered conditions require a newly frozen later test.

## Output verdicts

Use one of:

- `queue_ready`,
- `conditional_pilot_pending_preflight`,
- `needs_direct_test`,
- `deferred_for_data`,
- `rejected_for_now`.
