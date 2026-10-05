---
doc_status: active
default_read: routed
owner: user+agent
purpose: Rules for named hypothesis/direct tests and threshold sweeps.
do_not_use_for: Running broad exploratory queues without hypotheses.
last_rebuilt: 2026-09-09
---


# Rules - Direct Tests

## Core rule

Every test must answer a trader question first.

Good:
- If price breaks range high with rising volume pressure, does it usually continue?
- If support is broken but quickly reclaimed, does downside risk fade?
- If orderbook support disappears and downside liquidity is thin, does drawdown risk increase?

Bad:
- Run all features and see what happens.
- AUC was 0.61, therefore this is good.
- Context performed well without naming the source block and behaviour.


## Snapshot/runtime discipline

1. Use frozen parquet snapshots where possible.
2. Pause relevant collectors and export a snapshot before tests that would otherwise read live SQLite databases.
3. Record the snapshot path, date range, duplicate timestamp count, known gaps, and source future-violation count.
4. Do not read live collectors during direct tests unless the task is explicitly source-readiness validation.

## Required test definition

1. Trader question.
2. Visible market state at the decision hour.
3. Exact source-detail groups.
4. Numeric inputs / feature columns or compact feature groups.
5. Target: what happened afterwards.
6. Setup rows, trigger rows, and positive target examples.
7. Controls.
8. Pass, conditional-lead, confirmation-only, unresolved, park, and reject meanings
   defined before interpretation; do not force a complex result into binary pass/fail.
9. Exact threshold settings if thresholds are used.
10. Intended market scope: broad top-coin, frozen coin group, or asset-specific, including the grouping rationale and per-member support rule.
11. For meme-cohort work, the exact frozen `10` most traded eligible meme coins,
    taxonomy source, venue/market type, prior-30-completed-day median quote-turnover
    ranking, ranking timestamp, and coverage rule.
12. For event work, the event identifier, scheduled/unexpected class, first knowable
    time, information cutoff, prior expectation where available, released outcome or
    surprise, observation window, and outcome window.
13. For event work, a whole-event development/holdout split fixed before model or
    threshold selection. Candles from one event must never appear on both sides.
14. Accepted facts that the test will not redundantly question.
15. The proposed decision-time role of each input: background, driver/trigger,
    confirmation, modifier, accumulator, duplicate/substitute, or outcome.
16. The expected path under comparable pre-event conditions and how both the raw and
    ordinary-adjusted outcome will be measured.
17. The evidence decision that follows from a positive, negative, or unresolved
    result. Do not run a test whose possible outcomes cannot change the evidence map.

## Required controls

Use as applicable:

1. Random eligible rows inside the same clean window.
2. Same regime without trigger.
3. Opposite-direction setup.
4. Shuffled labels.
5. Price/structure-only baseline.
6. Missing-ingredient ablations: minus volume, minus orderbook, minus structure, minus context/news, minus multi-timeframe confirmation.
7. Same-regime non-member assets when the claim is specific to a coin group.
8. For event chains, compare event-only, market-only, group-only, local-only, the
   frozen combination, and a matched no-event or no-confirmation control where data
   permits.
9. Test BTC, ETH, and broad-market breadth as alternative leaders rather than giving
   BTC permanent precedence.
10. Compare the same trigger under materially different predeclared backgrounds and
    comparable backgrounds with and without the trigger.
11. For a named two-part interaction, compare neither, A only, B only, and A plus B
    where each cell has usable support.
12. Compare strong, weak, delayed, reversed, and absent responses, then use only
    timestamp-safe development-period differences to define a later confirmation.
13. Check whether multiple articles, indicators, timeframes, coins, or horizons repeat
    one underlying event rather than supply independent evidence.

## Whole-Episode And Interaction Rules

1. Do not interpret the market as a contest in which one input must be the sole cause.
   Ask how each timestamp-safe layer changes the balance established by the others.
2. The same measurement can change roles over time. Pre-event volume may be
   background or anticipation; immediate post-event volume may confirm the response;
   later volume may be the outcome.
3. When estimating an event's total influence, do not control on a price, volume,
   pressure, or volatility response caused after the event. Doing so can remove the
   path through which the event acted.
4. It is valid to use an early response to predict a later response, but describe this
   as a later-decision confirmation or continuation test, not as evidence that the
   trigger did not matter.
5. A factor with a weak standalone average may still be a rational amplifier,
   suppressor, reverser, gate, accumulator, delay, shortener, transmission blocker, or
   override. Test its declared conditional role rather than rejecting it solely for
   weak average prediction.
6. Measure whether an influence changed magnitude, timing, path, or loss relative to
   the matched expected path. A positive influence can matter even when the final
   return remains negative, and vice versa.
7. Treat the same event family under different dominant concerns, expectations,
   positioning, liquidity, narratives, and technical structures as conditional
   evidence rather than assuming one permanent meaning.
8. Conditions discovered by contrasting successful and unsuccessful episodes remain
   provisional. Freeze them and test later whole events before accepting the
   explanation.
9. Do not average contradictions away. Report when a response was amplified,
   suppressed, delayed, reversed, absent, or overwhelmed and which pre-event
   conditions distinguish those paths.
10. Do not rescue every failure. A conditional explanation must be measurable before
    the outcome, market-rational, adequately supported, and reproducible.

## Metric interpretation rules

Metrics are evidence, not the objective. Always translate every metric back into the trader question: did the visible market state help predict target-before-stop, continuation, failure, drawdown, or recovery?

### Binary event targets

Binary event targets answer yes/no questions such as “did breakout succeed in the next 6h?” or “did price hit target before stop?”. Report and interpret:

1. `roc_auc`:
   - Plain meaning: how well the score ranks good outcomes above bad outcomes.
   - Use when: the concept creates a score or model ranking.
   - Caution: AUC can look acceptable even when the top-ranked rows are not tradable or the event is rare.
2. `average_precision`:
   - Plain meaning: when the score picks its favourite rows, are those rows actually enriched with the rare event?
   - Use when: positives are uncommon, such as crash risk, rare pattern success, breakout failure, or stop-before-target labels.
   - Caution: if AP does not improve over the base event rate, a good AUC may still be weak for trading.
3. Trigger event rate:
   - Plain meaning: when the setup/trigger fires, how often does the desired event happen?
   - Must be compared against same-regime and random controls.
4. Same-regime-without-trigger event rate:
   - Plain meaning: what happens in similar market conditions when the trigger is absent?
   - This prevents giving credit to a trigger when the whole regime already has the event.
5. Random eligible control event rate:
   - Plain meaning: what happens in ordinary eligible rows from the same clean data window?
   - This is the basic “did we beat random rows?” check.
6. Shuffled-label AUC:
   - Plain meaning: what score appears when the answer sheet is deliberately scrambled?
   - The real result must beat this, otherwise it may be noise.
7. Price/structure baseline AUC:
   - Plain meaning: does the new feature/source add anything beyond existing price/structure information?
   - Do not claim orderbook/news/context lift if price/structure alone does as well or better.
8. Top-bucket actual rate and bottom-bucket actual rate:
   - Plain meaning: the rows the score likes most should have more good outcomes than the rows it dislikes.
   - Report top/bottom decile or quintile where sample size allows.
9. Setup rows, trigger rows, and positives:
   - Plain meaning: how many real examples are being tested?
   - A strong-looking result with tiny trigger/positive counts is a lead, not proof.

### Joint reaction-and-direction calls

When the question asks whether volume will react and which way price will respond,
report three separate rates:

1. Reaction success: the frozen abnormal volume response occurred against its causal
   baseline/control.
2. Conditional direction success: among real reactions, the signed direction/path
   prediction was correct.
3. Joint success: both were correct on the same issued call; either failure counts as
   a joint failure.

Also report eligible independent episodes, issued calls, abstentions, coverage,
uncertainty, per-coin/per-window concentration, and the strongest base-rate,
majority-direction, simple-trend, and no-level comparators. Also report direction on
all issued directional calls, not only on cases that later reacted. Under the
19 September 2026 revision, Objective 02b Section 15.4 assesses each component for its
own role: a joint failure does not veto useful direction, activity, location or a
conditional modifier. Joint scoring applies to joint claims, not every direct test.
Keep historical frozen scores and label component reassessments separately. Evidence
quality, later confirmation and trading safeguards are unchanged.

For event tests, count whole events as independent confirmation units. Repeated
candles, coins, or horizons around one event can describe that event but cannot turn it
into many independent successes.

### Continuous/path targets

Continuous/path targets answer questions like “how far did price move?”, “how bad was max drawdown?”, or “how much upside came before downside?”. Report and interpret:

1. Oriented top-bucket lift:
   - Plain meaning: the rows the score likes most should have better future path outcomes in the intended direction.
   - For bullish signals, top bucket should show better upside/return or less drawdown. For bearish/risk signals, top bucket should show larger downside risk or better warning value.
2. Spearman correlation and p-value:
   - Plain meaning: as the signal gets stronger, does the future outcome generally move in the expected direction?
   - Use Spearman because trader signals are often threshold-like or monotonic, not perfectly linear.
   - Treat the p-value as a noise warning only; it is not a promotion decision by itself.
3. Same-regime and random-control target means:
   - Plain meaning: did the scored rows beat similar non-trigger rows and ordinary eligible rows?
4. Price/structure baseline lift:
   - Plain meaning: does the new source improve over what price/structure already knew?
5. Monthly or rolling-window stability:
   - Plain meaning: does the relationship repeat across time, or is it one lucky period?

### Threshold sweep interpretation

Threshold sweeps test whether an idea survives different cutoffs, such as volume expansion level, orderbook wall-removal score, confluence component count, or distance to resistance/support.

For every promising threshold setting, report:

1. exact setup threshold(s), trigger threshold(s), and component-count requirements,
2. setup rows, trigger rows, and positives,
3. control rates / baseline score,
4. monthly stability,
5. whether adjacent threshold settings also work,
6. whether the threshold still matches the trader story.

Do not promote a one-off threshold that only works at a single fragile setting. Prefer a small stable region of working thresholds.

### Exact report-row requirement

Do not promote vague claims such as “breakout confluence looked good”. A promoted direct-test result must point to the exact tested row/report with:

1. hypothesis ID,
2. target label,
3. source-detail groups,
4. setup and trigger thresholds,
5. setup rows, trigger rows, positives,
6. controls and baselines,
7. broad, coin-group, or asset-specific scope and per-member results,
8. verdict.

## Result classification

- `direct_promising`: beats controls and makes market sense.
- `sweep_promising`: threshold sweep shows several viable settings.
- `conditional_modifier`: changes a named trigger's probability, magnitude, timing,
  path, or transmission under a frozen context without requiring a strong standalone
  effect.
- `confirmation_only`: usefully describes or updates an episode after the initiating
  event but does not predict the event or replace it as the driver.
- `amplified_suppressed_or_reversed`: a real-looking influence changes materially
  across declared contexts and requires later whole-event confirmation.
- `unexplained_inconsistency`: contradictory episodes remain informative but no
  timestamp-safe conditional explanation has yet reproduced.
- `needs_rework`: unclear/failed but repair path exists.
- `deferred_for_data`: cannot test honestly yet.
- `rejected_for_now`: failed controls, weak sample, no trader logic, likely lookahead/noise.
- `candidate_for_strategy_research`: strong enough to test as rule/filter/risk input.

## Anti-leakage rules

1. Feature values must be known at the decision hour.
2. Targets must be future-only and not leak into setup/trigger.
3. Historical features must not cross base timeline gaps unless explicitly gap-safe.
4. Missing data must not be silently forward-filled into valid states.
5. Source coverage and timestamp filters must be applied before interpreting metrics.
