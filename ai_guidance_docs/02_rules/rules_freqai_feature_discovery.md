---
doc_status: active
default_read: routed
owner: user+agent
purpose: Rules for loose/early FreqAI feature exploration, including Objective 02b reaction and event-scoped direction questions.
do_not_use_for: Promoting strategies or final production decisions.
last_rebuilt: 2026-09-09
---

# Rules - FreqAI Feature Discovery

## Purpose

Use this for early feature discovery. The standard is intentionally lighter than FreqAI promotion or strategy acceptance.

If the user says to use FreqAI, use FreqAI. Do not substitute custom scripts for FreqAI theory testing, candidate filtering, or ML inference. If the current FreqAI setup does not fit the requested data or theory, create or update a dedicated FreqAI environment/profile for that request.

Under current Objective 02b, apply this file to reaction-zone targets and to the
bounded event-scoped direction hierarchy authorized on 3 September 2026. Direction is
valid only inside predeclared event, market-confirmation, coin-group, coin-local, or
post-event-range questions, plus the evidence-triggered `1m` replay lane. Unrestricted
every-candle direction fishing, entry/exit construction, profit optimization, and
trading actions remain outside the objective. The objective's stricter boundary
controls.


## Runtime/snapshot requirements

1. Use validated snapshots or exported parquet caches for repeatable feature discovery.
2. Do not run feature discovery directly against live SQLite collectors unless the task is explicitly source-readiness validation.
3. If a source requires collector pause/export, follow `rules_runtime_environment.md` before testing.

## Horizon selection

For live media/news/web/global feature discovery, include fast targets by default: `1h`, `2h`, and `4h`, alongside slower `6h` and `24h` targets when data volume allows. Live media can move markets within minutes to a few hours, so do not rely only on `6h`/`24h` labels unless the user explicitly asks for slower path analysis.

Scale the grid to the event type. Scheduled macro releases and sudden shocks may need
minute-to-day outcomes; persistent geopolitical or financial changes may need
day-to-week outcomes; structurally anticipated events may need week-to-month
anticipation and aftermath windows. Freeze the full applicable grid before reading
outcomes rather than reporting only the most attractive horizon.

## Objective 02b Event-Hierarchy Discipline

Do not ask one unrestricted model to discover the whole market story. Use the
following order:

1. reconstruct the slow background and event from information available at the time;
2. test whether the event adds activity or signed information for BTC, ETH, or broad
   crypto participation;
3. test which eligible leader moved first without selecting it from the final path;
4. test established-alt and meme-cohort transmission separately;
5. measure a coin's extra response beyond its earlier-estimated ordinary market
   sensitivity and volatility;
6. test local levels, clusters, volume, pressure, compression, attention, and liquidity
   as modifiers rather than assumed causes; and
7. test continuation, reversal, settlement, revisit, and failure of post-event ranges.

Complete the single-block generation before pairwise work. Complete all frozen pairs
before limited three-block chains. For an independent forecasting claim, a pair or
chain must beat each immediately simpler component on identical eligible rows. A
factor may still be retained as a conditional modifier, gate, suppressor, amplifier,
delay, or override when its standalone average is weak, but that role must be defined
before confirmation and must improve the conditional explanation or prediction on
later whole episodes. Hold out whole later events, not random candles from the same
continuing episode. Conflicting leadership, unconfirmed events, missing expectations,
or unusable sources must permit abstention rather than guessed direction.

## Conditional Markets And Provisional Edges

Treat markets as noisy, adaptive, and condition-dependent. No entry, exit, level, indicator, or model is expected to work reliably in every pair, direction, timeframe, regime, or external environment.

The research objective is not to discover absolute rules. It is to find small, repeatable changes in market-response probability under identifiable and trader-readable conditions.

An entry or exit event may create almost no measurable response when considered across every occurrence. It may nevertheless become useful when direction and pressure agree, volume changes as the level is approached, suitable volatility is present, nearby same-direction evidence supports it, and there is sufficient room before an opposing obstacle. The same event may weaken, reverse, or disappear when those conditions are absent.

Several signals occurring together may create a stronger combined response. They may instead be duplicate descriptions of the same underlying event, mutually conflicting evidence, or irrelevant because a stronger external event dominates the market. These possibilities must be measured rather than assumed.

Use the following interpretation rules:

1. Treat markets as noisy, adaptive, and condition-dependent. No entry, exit, level, indicator, or model is expected to work reliably in every pair, direction, timeframe, regime, or external environment.

2. The research objective is not to discover absolute rules. It is to find small, repeatable changes in market-response probability under identifiable and trader-readable conditions.

3. A weak average result does not necessarily mean a signal contains no information. Its effect may depend on direction, regime, timeframe, nearby levels, volume, pressure, volatility, signal ordering, or wider-market conditions. Test plausible conditional relationships before rejecting the underlying idea.

4. Conversely, do not rescue every weak result with an after-the-fact explanation. Conditions must be defined numerically, tested against appropriate controls, and reproduced on later unseen data.

5. Treat every result as a confidence update:

   - `lead`: an effect worth testing again;
   - `conditional_lead`: an effect appearing only under named conditions;
   - `unresolved`: evidence is mixed or the sample is inadequate;
   - `not_reproduced`: the effect failed a valid repeat test;
   - `parked`: further testing is currently lower value;
   - `candidate_edge`: repeated evidence survives controls and unseen data; and
   - `promotable`: a candidate edge also satisfies the separate direct-test, FreqAI-promotion, operational, and risk requirements.

6. Do not convert one failed model, feature representation, target, horizon, or initial aggregate test into rejection of the underlying market idea. State exactly what was tested, what was not established, and whether a narrower conditional test is justified.

7. Do not treat one successful pair, direction, period, threshold, or model as a general edge. Pair-specific findings are leads with lower initial confidence until they reproduce across coins, directions, periods, or clearly defined market classes.

8. Study reactions as paths, not only as profit at one future candle. Measure timing, direction, favourable and adverse excursion, continuation, rejection, reversal, volatility, volume, and pressure across multiple horizons.

9. Treat signal combinations as interaction hypotheses. Test whether aligned signals add information beyond their individual effects. Do not assume that more agreeing signals automatically mean a stronger edge.

10. Treat nearby opposing levels as possible obstacles and same-direction levels as possible support. Investigate distance, order, timeframe, age, prior touches, and source-family independence. Do not assume that higher-timeframe levels always take precedence; test when they do and whether they define structure, timing, invalidation, or another role.

11. News, major-market moves, liquidity disruptions, and broader risk conditions may override or distort crypto-specific technical behaviour. Until those sources are incorporated and validated, describe unexplained variation as potentially externally confounded, not as proof that a technical effect does or does not exist.

12. `Quiet external conditions` is itself a hypothesis. Do not assume a period was quiet, normal, or news-free merely because context data is unavailable. Once wider-market and news data are ready, define quiet and disrupted conditions explicitly and retest earlier leads within them.

13. Prefer modest effects that reproduce across suitable markets, sides, periods, and adjacent conditions over large effects isolated to one pair or window. Genuine edges may still be conditional, but their conditions must make market sense and remain identifiable before the outcome is known.

14. Preserve promising, unresolved, and negative evidence in the approved ledger. Revisit leads as more data, contexts, and controls become available. The purpose of each round is to improve the map of when, where, how, and under what conditions a reaction appears, not to force an immediate final verdict.

15. Promotion thresholds are decision safeguards, not exploration filters. Apply them only when deciding whether accumulated evidence is strong enough for strategy or risk-management use.

Generate explanations freely, but accept them only as hypotheses. Define the proposed condition before the next test, then require it to reproduce on unseen data. This preserves broad exploration without turning every result into an unfalsifiable story.

## Whole-Episode Conditional Interpretation

Do not reduce a market question to whether one input `works`. For each decision time,
interpret the complete episode as a balance of timestamp-safe layers:

1. long-running macro, financial, geopolitical, liquidity, and crypto background;
2. the rolling balance of distinct positive, negative, and conflicting stories;
3. contemporaneous expectations and what may already be reflected in price;
4. the new event, its novelty, credibility, severity, and surprise;
5. the first observable BTC, ETH, or broad-market response;
6. technical structure, levels, clusters, liquidity, and positioning that may modify
   the response;
7. transmission into established coins, memes, and the coin-local state; and
8. the later path: amplification, suppression, delay, continuation, reversal, or
   settlement.

The role of a measurement depends on when it is observed. Pre-event volume may
describe background activity or anticipation. Volume measured after a release may be
confirmation of the news response. Future volume is an outcome. Never compare a
downstream response with its possible driver as if they were competing root causes,
and never control away the response when estimating the driver's total effect.

Every feature block must be assigned one or more explicit decision-time roles:

1. `background_condition` - changes the starting probability or likely range;
2. `driver_or_trigger` - introduces new information or a new market condition;
3. `confirmation` - shows that participants noticed or accepted the trigger;
4. `modifier` - amplifies, suppresses, reverses, delays, shortens, gates, or blocks
   transmission of another influence;
5. `accumulator` - represents several genuinely distinct small influences building
   together;
6. `substitute_or_duplicate` - repeats substantially the same underlying information;
   or
7. `outcome` - the future behaviour being measured.

Roles are hypotheses, not permanent properties of a column. Record the information
cutoff and assign them again when the decision time changes.

Measure an influence relative to the path expected from comparable pre-event states,
not only by the final candle sign. Positive information may still have mattered when
price fell less than comparable bearish controls; a strong initial response may still
have been suppressed or reversed later. Report raw direction, ordinary-adjusted
change, reaction magnitude, timing, transmission, continuation, and reversal
separately.

Weak standalone evidence does not disqualify a rational interaction. Test the four
comparable states where support permits: neither ingredient, A only, B only, and both.
An interaction lead must identify whether B changes A's probability, size, timing, or
path and must survive a frozen later test. Do not require B to predict the outcome by
itself when the named theory is that B only modifies A.

Prevent flexible interpretation from becoming hindsight storytelling:

1. freeze broad context families and causal ordering before reading confirmation
   outcomes;
2. use development data to discover possible suppressors or amplifiers, then freeze
   them for later whole-event confirmation;
3. compare strong, weak, delayed, reversed, and absent responses instead of averaging
   contradictions away;
4. require every proposed explanation to use information knowable at the relevant
   decision time;
5. report plausible competing explanations and source limitations;
6. treat many assets, horizons, or articles inside one continuing event as descriptive
   breadth, not independent confirmation; and
7. do not run a test unless either possible result would change the evidence map or a
   later decision.

FreqAI may discover nonlinear conditional patterns, but feature importance is not a
causal explanation. Confirm a machine-discovered relationship with readable
conditional summaries, source-family removal, matched episodes, later chronological
data, and prospective capture where history is inadequate.

## Hierarchical Market Scope

Do not use one fixed coin-count threshold for every type of claim.

1. Use a broad threshold such as seven of ten suitable coins only when claiming that an effect generalizes across the frozen top-coin universe.
2. Retain a smaller coherent coin-group result as `coin_group_specific_lead` when the group was predeclared or formed without target outcomes, each member has adequate support, the sign and trader meaning agree across a clear majority, the effect repeats across at least two windows, and no one member dominates.
3. Retain a repeated BTC-only or other one-asset result as `asset_specific_lead`, not as a broad edge.
4. If the positive results themselves suggest a group, record the group as post-hoc and confirm it only in a later frozen batch or unseen window.
5. Do not repeatedly regroup coins until a positive subset appears.
6. Compare a group-specific model with the same price, regime, stale/shifted, and non-member controls required by its claim.

Meme-coin investigation uses only the frozen `10` most traded eligible meme coins on
the selected venue and market type. Rank by median daily quote turnover over the prior
`30` completed UTC days before reading event outcomes, and retain the taxonomy source,
ranking timestamp, exact members, and coverage rule. Do not replace members after
results are opened. Use the full contract in
`reference_freqai_event_reaction_research_method.md`.

Possible grouping inputs include past return correlation, BTC beta or response lag, volatility, liquidity, market-cap tier, trend persistence, broad-risk response, and a trader-readable economic role. Group membership must use only information available before the tested outcome. See `reference_freqai_event_reaction_research_method.md` for the full grouping, validation, and reporting procedure.

## Reaction And Relevance Research

For routed event, level, entry, exit, path, or interaction research, also read `reference_freqai_event_reaction_research_method.md`. It provides the reusable detailed sequence, target meanings, controls, model ladder, market-scope hierarchy, integrity audit, and reporting checklist.

The direction-versus-level theory in
`theory_freqai_regime_direction_level_reaction.md` is active during current Objective
02b only for the bounded evidence-triggered `1m` replay lane. It remains parked for
broad ordinary-timestamp direction searches, long/short selection, entry/exit action,
and profit optimization.

Forward profit or price after a fixed number of hours or days is only a baseline target. It is not sufficient by itself to establish why a feature, event, or market level may be useful.

For research involving levels, zones, structures, or market events, investigate whether approaching, touching, crossing, or rejecting the feature coincides with changes in relevant market behaviour. Depending on the hypothesis, this may include:

1. Price direction, excursion, reversal, continuation, volatility, or path.
2. Volume level or change in volume regime.
3. Buying/selling pressure and changes in directional pressure.
4. Momentum, trend strength, acceleration, weakening, or rejection.
5. Reaction timing across the next several candles, not only one fixed endpoint.
6. Whether the observed behaviour differs from normal periods, unrelated locations, and stale or shifted placebo levels.

Treat relevance as conditional and continuous, not as a yes/no claim that a level or feature "works." Where feasible, produce separate trader-readable scores for:

1. Importance or expected reaction magnitude.
2. Directional or reversal tendency.
3. Proximity and immediacy.
4. Quality, age, confirmations, and timeframe.
5. Same-asset and cross-asset confluence.
6. Confidence, sample support, and stability across windows.

Investigate whether the current event resembles recent historical interactions with the same or similar level under comparable market conditions. Test whether those analogues are more informative than the asset's normal behaviour, and whether recency, repeated tests, prior reactions, market regime, or changing level quality alter its present relevance.

A statistical difference is the start of the investigation, not its conclusion. Interpret:

1. What market behaviour changed.
2. When and where the difference appeared.
3. Whether the relationship has a coherent market explanation.
4. What additional conditions strengthen or weaken it.
5. Whether the effect repeats in development and holdout data.
6. Whether it survives appropriate controls, including stale, shifted, unrelated, or unconditional baselines.

Small repeatable advantages may be combined with other coherent evidence. For an
independent/additive claim, test each added layer against the immediately simpler
version. For a modifier claim, test whether the named context changes the primary
relationship within comparable rows. Do not assume that more inputs create a stronger
signal.

The intended output is an evidence-based relevance model answering:

1. Could this information be useful?
2. How useful does it appear?
3. Under what conditions?
4. What else was happening?
5. How much additional value did that context provide?

Do not reduce this work to "price was profitable after X hours," and do not treat proximity to a level alone as an entry or exit instruction.

## Breadth-First Candidate Queue

New indicators, mathematical representations, timeframes, level/cluster definitions,
convergence patterns, coins/cohorts, and contextual sources inspired by interim or
final results are valid branch candidates. Record their source result, plain-language
hypothesis, alternative explanation/control, required data, smallest useful test, and
earliest eligible generation. Do not launch them or alter the frozen surface until all
current-generation batches are terminal and the joint review freezes the next batch.

Objective 02b must assess and eventually fairly test at least five materially distinct
route families. Cosmetic feature variants do not count. This five-route breadth rule
is separate from the evidence-contingent branch horizon currently authorized in the
objective.

## Optional Multi-Timeframe Indicator Context

Conventional technical indicators may be added as an optional context layer after the event/level baseline for the current research generation is complete. Their purpose is not to rediscover an indicator-only strategy or to ask whether an event was profitable after one fixed delay. Their purpose is to describe the market state in which an entry, exit, or level interaction occurred and test whether that state changes the subsequent reaction path.

This is a branch candidate, not an automatic requirement for every event-reaction run. It must obey the batch-gated branching rules in the active objective. Do not interrupt a frozen generation, add indicator combinations because they are available, or allow an early indicator result to dominate unfinished main-scope tests.

### Sensible initial indicator set

Use a compact, fixed, trader-readable set before considering alternative periods or thresholds:

1. RSI `14`: level, distance from `50`, overbought/oversold state, and recent slope.
2. Bollinger Bands `20, 2`: percent-B position, width, and width contraction or expansion.
3. MACD `12, 26, 9`: line/signal relationship, normalized histogram, recent cross, and histogram acceleration or weakening.
4. Simple moving averages: price distance from `20`, `50`, and `200`; ordered or disordered state; and slope where enough history exists.
5. Exponential moving averages: price distance from `12`, `26`, and `50`; ordered or disordered state; and slope.
6. ATR `14` and realised volatility: normalized volatility level and expansion/contraction state.
7. Volume context: current volume versus a trailing `20`-candle baseline, volume z-score, and one cumulative pressure measure such as OBV slope or CMF `20`.
8. ADX `14`: trend-strength context, kept separate from direction.

Do not include several near-duplicate versions of every indicator in the first test. Do not tune periods and thresholds while deciding whether the feature family has any useful relationship. Conventional parameters provide the first controlled representation; period or threshold variation is a later branch only if the initial result gives a named reason.

### Timeframe coverage

The available research ladder may include `1h`, `2h`, `4h`, `8h`, `12h`, `1d`, `3d`, and `1w` when local coverage is validated. Building timestamp-safe features for the full ladder does not mean every experiment must consume every timeframe simultaneously.

Start with the event's native timeframe, one adjacent higher timeframe, and one structural higher timeframe. Expand to the remaining validated timeframes as a controlled comparison. Adjacent timeframes are highly correlated, so report whether apparent multi-timeframe agreement is genuinely additive or merely repeats the same price history.

For every timeframe:

1. Use only the most recently completed candle available at the event decision time.
2. Never forward-fill a partially formed higher-timeframe candle as if it were complete.
3. Record the informative candle timestamp and age at the event.
4. Treat missing history as missing, not as a neutral or zero indicator state.
5. Validate pair/timeframe coverage and duplicate timestamps before the batch.

### Structured question ladder

Test indicator context in this order:

1. **State description:** Does the entry, exit, or level event occur disproportionately in a named RSI, volatility, volume, trend, or band state compared with matched ordinary candles?
2. **Reaction conditioning:** Within the same event identity, does one indicator state change direction, favourable/adverse excursion, peak timing, threshold-touch order, volume, pressure, volatility, continuation, or reversal?
3. **Incremental FreqAI value:** Does one indicator family improve the relevant out-of-sample target family beyond the identical OHLCV/event baseline and the shifted-event placebo?
4. **Timeframe alignment:** Does agreement between native and higher-timeframe states add information beyond either state alone? Test disagreement as well as agreement.
5. **Level interaction:** Does indicator state alter reaction when same-direction, opposing, mixed, or higher-timeframe levels are nearby?
6. **Limited interaction:** Only after the one-family tests are complete, test a small
   predeclared interaction such as event plus volume expansion plus MACD acceleration.
   Compare additive claims with every immediately simpler component; for a modifier
   claim, compare the primary event with and without the modifier on the same eligible
   rows.

Do not enumerate all indicator × timeframe × direction × level combinations. Finish the registered one-family batch, review all results together, and freeze only evidence-backed interaction questions for the next branch generation.

### Controls and interpretation

Use the same pair, window, horizon, target, and model settings for each comparison. Retain the OHLCV-only model, exact-event model, shifted-event placebo, matched ordinary candles, and relevant stale-level controls. A shifted indicator-state placebo may be added where regime persistence is explicitly considered; it does not replace the event and level controls.

Separate target families in reporting:

1. direction and terminal return;
2. favourable and adverse excursion;
3. path order and peak timing;
4. volume activity;
5. buying/selling pressure; and
6. volatility activity.

An indicator family is an exploratory lead when it adds a small, coherent difference across several suitable coins or windows and the sign makes sense for the named market condition. A result isolated to one coin, side, threshold, or adjacent timeframe is more likely noise and remains unresolved until repeated. A failed bundled model rejects that bundle, not every indicator or conditional market idea inside it.

Record null and contradictory results. Indicator context is optional and should be
parked when it adds only noise, duplicates the OHLCV baseline, depends on a tiny cell,
or fails its declared additive or modifier comparison. It must never be promoted
merely because it improves profit in one backtest or predicts one fixed future return
horizon.

## Acceptable early evidence

A feature or feature family may be kept as a research lead if it:

1. maps to a trader-readable state,
2. is timestamp-safe in the tested window,
3. has enough non-null/active rows for exploration,
4. shows coherent direction or ranking in at least one relevant target/window,
5. does not obviously duplicate the answer label,
6. is recorded as exploratory, not promoted.

## Not acceptable

1. Broad “run everything and see what happens” without source groups and targets.
2. Treating raw AUC/correlation as a trading edge.
3. Treating a single good window as stable proof.
4. Treating missing source rows as zero signal.
5. Allowing debug/source-availability columns as predictive features unless explicitly testing missingness.

## Exploratory metric handling

Early feature discovery may use loose metrics, but the interpretation must stay modest:

1. AUC/correlation/AP/top-bucket lift can mark a feature family as interesting.
2. A single metric cannot make a feature strategy-ready.
3. For rare events, prefer AP and top-bucket enrichment over broad accuracy claims.
4. For path targets, prefer oriented top-bucket lift and Spearman direction over raw correlation alone.
5. Any promising exploratory metric must move to direct tests before FreqAI promotion or strategy integration.
6. For Objective 02b, follow Section 15.4's 19 September 2026 component assessment:
   direction and activity/location/modifier inputs can qualify separately. Report
   direction on all issued calls separately from direction among later reactions,
   and report joint success when both outcomes are claimed. A low joint rate is not
   a universal rejection gate. Controls, support, uncertainty, later confirmation
   and separate promotion rules remain; combinations need their own tests.

## Required output

Record exploratory results in a concise result file or append to `../04_results/hypothesis_ledger.csv` with status such as:

- `idea`,
- `exploratory_signal`,
- `needs_direct_test`,
- `deferred_for_data`,
- `rejected_for_now`.

## Escalation

Before FreqAI promotion, move to:

- `rules_direct_tests.md`, then
- `rules_freqai_promotion.md`.
