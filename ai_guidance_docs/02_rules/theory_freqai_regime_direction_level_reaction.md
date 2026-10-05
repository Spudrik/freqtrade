---
doc_status: active
default_read: routed
owner: user+agent
purpose: Detailed theory and method for Objective 02b event-driven market direction, reaction-zone interaction, coin-group transmission, post-event ranges, and evidence-triggered one-minute replay.
do_not_use_for: Unrestricted directional feature fishing, assuming levels cause price moves, selecting memorable historical winners, profit optimization, or selecting trades from unvalidated model outputs.
last_rebuilt: 2026-09-09
---

# Theory - Market Regime Supplies Direction, Levels Locate Reactions

## 1. Status And Intended Use

This document preserves and controls a possible directional theory:

> The broader crypto-market regime and the coin's recent multi-timeframe price path may supply much of the directional bias, while technical entry, exit, structure, and proximity levels may identify locations where market behaviour is more likely to change, intensify, stall, reject, reverse, or continue.

Objective 02b still preserves causal reaction-zone discovery and the three already
frozen fresh-confirmation questions. On 3 September 2026 the user also authorized a
bounded event-driven direction stage. That stage tests a causal hierarchy from slow
background conditions and major events through BTC, ETH, or broad-market confirmation,
established-alt and meme response, coin-local technical or liquidity modification,
and post-event range formation.

Signed direction is permitted only inside predeclared event, market-confirmation,
coin-group, coin-local, or post-event-range episodes, plus the evidence-triggered `1m`
replay. Agents must not use this file to introduce unrestricted every-candle up/down
modelling, long/short selection, entry/exit logic, position sizing, leverage, profit
optimization, or live/dry-run changes.

This remains a theory to test, not an established market law. Agents must not
describe it as proved merely because price sometimes reacts near levels or because a
model score improves.

Earlier Sieve-linked Generations 1 and 2 did not establish a non-redundant level
representation for this directional theory. That historical result is retained as a
warning against rescuing a failed representation by adding features. It does not gate
the current reaction-and-event-driven objective, whose level lane starts from
indicator levels and matched controls rather than Sieve events. The historical
evidence remains
`../../user_data/research_news_data/context_features/sieve3_event_reaction/generation2/generation2_review.json`.

Read this document when a task involves any of:

1. using FreqAI to investigate the direction of a reaction at or near a level;
2. separating broad market direction from coin-local technical reaction;
3. modelling breakout, rejection, reversal, continuation, stalling, or acceleration at levels;
4. combining recent price paths, market regime, Sieve events, exits, obstacles, or multi-timeframe levels; or
5. interpreting whether a model learned direction, activity, timing, magnitude, or only redundant price state;
6. reconstructing major scheduled or unexpected events and their expectations,
   confirmation stages, or slow background;
7. identifying whether BTC, ETH, or broad participation led a market response;
8. measuring established-alt or meme transmission and genuine amplification; or
9. testing how event moves settle into, revisit, or break a new range.

Do not load this document for unrelated Sieve discovery, launcher maintenance, data downloads, collectors, or ordinary code work.

## 2. Core Market Theory

Markets are noisy, adaptive, and condition-dependent. A technical level is not expected to create the same response every time it is approached. A level may matter because market participants, liquidity, stops, prior positions, resting orders, breakout traders, or earlier reactions are concentrated around a similar area. That can make the area a reaction point without determining the direction of the reaction.

Direction may instead be influenced by a wider state that is already developing before contact, including:

1. the direction and strength of the overall crypto market;
2. the coin's own trend on several timeframes;
3. the recent sequence of swings and candles;
4. whether momentum, volume, pressure, and volatility are strengthening or weakening;
5. how price approaches the level;
6. the available room before another level or invalidation;
7. agreement or conflict among independent levels and signals; and
8. external news, macro markets, liquidity events, or other causes not yet observed by this programme.

The current Objective 02b theory orders those influences rather than flattening them
into one feature bundle:

> slow background -> scheduled or unexpected event -> first valid market response ->
> BTC, ETH, or broad-participation leadership -> established-alt or meme transmission
> -> coin-local amplification or obstruction -> continuation, reversal, or settled
> post-event range.

This ordering is an information path, not a claim that one layer always defeats the
others. The realised response is the changing balance of long-running pressure,
current narratives, expectations, the new event, market confirmation, technical and
liquidity conditions, cross-coin transmission, and later competing information. A
factor may amplify, suppress, reverse, delay, shorten, gate, accumulate with, or be
overridden by another factor. Its historical average is therefore only a starting
description, not its permanent market meaning.

Measure influence relative to what comparable pre-event market states would otherwise
imply. A positive influence may reduce an expected decline without making the final
return positive. A negative influence may cap a rally without producing a negative
terminal candle. Preserve the initial impulse, strongest move, timing, propagation,
fade, reversal, and later resumption of the broader regime.

Not every move follows this story. A valid result may show simultaneous movement, no
stable leader, activity without direction, an event rejected by price, coin-specific
news, or no distinct response. The model must be allowed to abstain when the event is
unconfirmed, expectations are unavailable, leadership conflicts, or source coverage is
not usable.

Keep scheduled and unexpected events separate. Before a scheduled release, a model may
estimate reaction likelihood without knowing direction. After the result is available,
direction may use surprise relative to a genuinely reconstructable expectation. An
unexpected shock cannot be predicted before its first safe report; only detection,
confirmation, and subsequent propagation may be tested.

Under this theory, a level contact can have several different roles:

1. **Continuation or breakout point:** the prevailing direction carries through the level and may accelerate.
2. **Rejection point:** price tests the area and moves away from it.
3. **Reversal point:** the contact is followed by a material directional change, not merely a brief rejection.
4. **Stall or absorption point:** movement slows, becomes two-sided, or consumes liquidity before the next move.
5. **Volatility or activity point:** volume or range expands, but direction remains uncertain.
6. **No distinct response:** behaviour is not meaningfully different from comparable non-level states.

The main research question is therefore not simply, "Does this level work?" It is:

> Given the market regime, the coin's recent path, the way price approaches the level, and other levels currently present, does contact with this level change the probability, magnitude, timing, or direction of the next market response beyond what those background conditions already predicted?

## 3. Required Conceptual Separation

Agents must keep the following concepts separate throughout feature design, modelling, scoring, and reporting.

### 3.1 Broad crypto-market regime

The timestamp-safe state of the wider crypto market before the decision candle. It may include BTC, ETH, and frozen top-10 cross-crypto breadth, direction, dispersion, volatility, volume, and correlation measures.

It does not mean news, macro markets, equities, rates, commodities, or global liquidity unless those sources have separately passed readiness and timestamp-safety checks.

### 3.2 Coin-local trend

The coin's directional state over several completed timeframes. This can include returns, price relative to moving averages or ranges, moving-average slopes and ordering, swing structure, trend strength, and directional pressure.

Trend strength and trend direction must remain separate. ADX-like strength cannot be interpreted as bullish or bearish direction by itself.

### 3.3 Recent approach path

How price reached the level, not only where price is now. Relevant descriptions may include:

1. approach direction and speed;
2. acceleration or deceleration;
3. higher-high/higher-low or lower-high/lower-low sequence;
4. compression followed by expansion;
5. candle-body and wick balance;
6. volume and pressure changes during the approach;
7. favourable and adverse swings before contact;
8. distance travelled in ATR units;
9. whether price arrived directly or after repeated tests; and
10. whether the most recent move agrees with or opposes the higher-timeframe trend.

### 3.4 Level state

A timestamp-safe representation of the technical area being approached, touched, crossed, or rejected. Preserve:

1. exact source family and variant;
2. side and proposed role;
3. source timeframe;
4. volatility-normalized distance;
5. age and prior touch count;
6. freshness or staleness;
7. same-direction and opposing nearby levels;
8. highest and lowest contributing timeframes;
9. independent-family confluence versus duplicate variants;
10. room to target, obstacle, and invalidation; and
11. whether the event is an entry, exit, target, warning, or generic structure level.

Level proximity alone is not an entry or exit instruction.

### 3.5 Reaction relevance

Whether behaviour changes near the level. Relevance may appear as greater reaction magnitude, faster threshold contact, volume expansion, volatility expansion, pressure change, stalling, rejection, or altered path order. It need not be directional.

### 3.6 Directional response

Whether the subsequent path has a repeatable signed bias after controlling for the broad regime, local trend, recent approach path, and suitable placebos. Direction cannot be inferred merely from greater volume, volatility, absolute movement, faster reaction, or a profitable fixed-horizon return in one subset.

### 3.7 Slow background

The timestamp-safe state accumulated before an event across weeks or months. It may
include persistent verified event pressure, rates, equities, currencies, commodities,
market volatility, liquidity, crypto breadth, and financial or geopolitical stress only
where each source has passed readiness. Do not assign `negative background` by reading
later history or current opinion.

### 3.8 Major event state

A predeclared scheduled release, unexpected shock, persistent change, or structurally
anticipated crypto event. Preserve the first safe report, prior expectation when
available, actual outcome, independent confirmation, official confirmation,
implementation, correction, escalation, relief, and source availability as separate
states. A large later price move cannot define event importance.

### 3.9 Market leadership

Whether BTC, ETH, or broad cross-coin participation first supplies a usable response.
Leadership must be identified from a frozen early window and must not be assigned to the
asset with the largest final move. Simultaneous or conflicting movement is a valid
`no stable leader` state.

### 3.10 Group transmission and amplification

Transmission means that an established-alt or meme cohort follows a market response.
Amplification means that the response is unusually large after allowing for each coin's
earlier-estimated ordinary sensitivity, delay, and volatility. A larger raw meme
percentage move alone is not amplification.

### 3.11 Post-event range

A range whose boundaries become knowable only after a frozen settling rule has been
met. Preserve the first availability time, construction, residence, revisits, smaller
event ripples, boundary reactions, and permanent failures. A hoped-for return to the
range is not evidence that a wrong call can safely be held longer.

### 3.12 Decision-time roles and whole-episode state

At each prediction time, assign every information block a declared role:

1. background condition;
2. possible driver or trigger;
3. market confirmation;
4. modifier or gate;
5. accumulator of distinct smaller influences;
6. duplicate or substitute measurement; or
7. future outcome.

The role belongs to the measurement at that time, not permanently to the column.
Volume before an event may be background or anticipation; volume immediately after it
may confirm that the event was noticed; later volume may be the outcome. Do not
condition on a downstream response and then claim that its possible driver was
unimportant. A later-decision continuation forecast is valid, but it must not be
reported as a pre-event prediction or root-cause comparison.

## 4. Trader-Readable Questions

Every FreqAI experiment derived from this theory must begin with one or more exact trader questions, such as:

1. In a broad crypto uptrend, does resistance contact with rising approach volume break more often than it rejects?
2. In a broad downtrend, does support contact continue downward unless the coin's recent downside pressure is weakening?
3. Does a coin-local trend aligned with top-10 market breadth produce more continuation at a level than a coin-local trend opposed to the wider market?
4. Does a fast, high-pressure approach react differently from a slow, compressing approach at the same level type?
5. Does a fresh level behave differently from a repeatedly tested level after controlling for the same regime and approach path?
6. Do aligned higher-timeframe levels add directional information, or do they only increase reaction magnitude?
7. Does an opposing level leave too little room for entry even when regime direction supports the trade?
8. Does a level interaction improve prediction beyond regime and recent path alone, or is the level only marking where the already-predicted move becomes visible?
9. Does a predeclared major event add unusual activity beyond a matched market state
   with no such event?
10. When a historical expectation is available, does the event surprise add signed
    information beyond headline tone and price state?
11. After an unexpected event is independently confirmed, does the first BTC, ETH, or
    broad-market response persist or reverse?
12. Does BTC lead the coin group by a usable interval, or are the moves simultaneous?
13. Do memes respond more than their own ordinary market sensitivity would imply?
14. Does a meme-local level, cluster, volume, compression, attention, or liquidity
    state explain additional amplification beyond event and leader information?
15. Does a causally defined negative background strengthen negative reactions or cap
    positive reactions compared with other backgrounds?
16. After an event move settles, do smaller events and technical areas produce
    repeatable movement within or through the new range?

These questions are hypotheses. The wording must not assume the expected answer is true.

## 5. Feature Blocks

Keep feature blocks separately identifiable so ablations and incremental comparisons remain possible.

### 5.1 Broad-market block

Use a frozen and recorded coin universe. For the current objective, any suitable top-10 crypto asset may contribute when timestamp-safe comparable OHLCV data exists.

Candidate broad-market features include:

1. BTC and ETH signed returns and normalized trends over several horizons;
2. top-10 breadth: share rising, falling, above trend, or breaking recent structure;
3. median and weighted cross-coin return;
4. cross-coin dispersion and volatility;
5. volume breadth and volume expansion;
6. directional-pressure breadth;
7. rolling correlation or decoupling from BTC and the top-10 basket; and
8. whether movement is broad, concentrated, or internally conflicting.

Do not retrospectively alter the basket after observing results. Record ranking source/date, included instruments, market type, history, and gaps.

### 5.2 Coin-local trend and path block

Candidate timestamp-safe features include:

1. signed returns over a fixed multi-horizon grid;
2. SMA/EMA distance, ordering, and slope;
3. RSI level and slope;
4. MACD relationship, histogram, and acceleration;
5. ADX or another separate trend-strength measure;
6. Bollinger position, width, compression, and expansion;
7. ATR and realised-volatility state;
8. volume relative to a trailing baseline;
9. OBV, CMF, or another fixed pressure description;
10. recent swing direction and structure;
11. range location and distance from recent highs/lows;
12. candle-body, wick, gap, and directional-sequence summaries; and
13. approach speed, acceleration, and pressure as the level is reached.

Use the compact conventional indicator definitions routed by `rules_freqai_feature_discovery.md` before testing alternative periods. Do not create hundreds of slightly different trend periods and then interpret whichever combination wins.

### 5.3 Level block

Use exact Sieve and structural identities where available. Candidate features include:

1. ATR-normalized distance to each relevant level role;
2. contact, crossing, reclaim, and rejection state;
3. age, touch count, and time since prior reaction;
4. source timeframe and timeframe relation to the event;
5. same-direction, opposing, and mixed-level counts;
6. independent-family confluence;
7. room to the next obstacle, target, or invalidation;
8. stale and shifted representations used only as controls; and
9. current trade state when the level is a plausible exit or position-management event.

### 5.4 Explicit interaction block

Do not rely only on placing regime and level columns in the same unrestricted model and assuming the model found the intended interaction. Test named interactions or carefully controlled nonlinear ladders, for example:

1. market direction x level side;
2. coin-local trend alignment x level role;
3. approach direction x level side;
4. approach speed x level freshness;
5. volume or pressure expansion x contact/crossing state;
6. volatility compression/expansion x level confluence;
7. broad-market agreement x coin-local disagreement;
8. supporting/opposing room x entry direction;
9. higher-timeframe structure x lower-timeframe timing; and
10. previous reaction history x repeated touch number.

Only interactions with trader-readable component definitions may proceed. Three-way or deeper interactions require a surviving simpler relationship and must obey the active branch-generation gate.

`Surviving simpler relationship` does not require every component to predict the
outcome independently. A technically rational factor may proceed as a modifier when
development evidence suggests that it changes another factor's probability,
magnitude, timing, path, or transmission. Where support permits, compare neither
factor, A only, B only, and A plus B on comparable rows. Freeze any development-found
amplifier, suppressor, reverser, gate, accumulator, override, delay, or transmission
blocker before later whole-event confirmation.

Do not infer interaction merely because a flexible model contains both columns or
assigns them importance. Require readable conditional summaries, suitable source-
family removal, activation/support counts, and untouched chronological confirmation.
Correlated indicators, adjacent timeframes, duplicate news coverage, and several coins
inside one common shock may describe breadth but do not supply independent causal
votes.

### 5.5 Event-hierarchy blocks

Keep these separately identifiable:

1. slow-background state;
2. scheduled or unexpected event identity and first availability;
3. prior expectation, actual outcome, and surprise when reconstructable;
4. first report, independent confirmation, official confirmation, implementation,
   correction, escalation, and relief stages;
5. BTC, ETH, and broad-participation candidate-leader state;
6. established-alt and meme-cohort transmission state;
7. coin-local response beyond its pre-event ordinary market sensitivity;
8. local technical, attention, flow, and liquidity modifiers; and
9. post-event range availability, position, residence, revisit, and failure.

Complete single-block tests before pairwise relationships and all frozen pairs before
limited three-block chains. During the current chain layer, use no more than three
materially distinct predictive blocks. Independent or additive forecast claims require
the full chain to beat every relevant immediately simpler model. Modifier claims must
instead show that the named modifier changes the primary relationship within the
frozen context and repeats on later whole events; a weak broad standalone average does
not by itself disqualify a modifier.

## 6. Timeframes And Horizons

Do not reduce this theory to a single `24h` question or profit after six candles.

### 6.1 Feature timeframes

Use completed candles only. The research ladder may include `1h`, `2h`, `4h`, `8h`, `12h`, `1d`, `3d`, and `1w` when coverage is valid. Start with the event's native timeframe, one adjacent higher timeframe, and one structural timeframe; expand only as a frozen comparison.

For every higher timeframe, record the informative candle timestamp and age. Never treat a partially formed candle as complete.

### 6.2 Outcome horizons

Use a common predeclared grid appropriate to the event and data, potentially including `1h`, `2h`, `4h`, `8h`, `12h`, `24h`, and `48h`, with longer horizons only when the trader question and source timeframe justify them.

Do not choose a different winning horizon for every family after seeing results. Report the path across horizons so early continuation, later protection, reversal, or delayed reaction remain distinguishable.

For scheduled releases and sudden shocks, include minute-to-hour outcomes where the
source timestamps and market data support them. Persistent geopolitical or financial
changes may require day-to-week outcomes; structurally anticipated events may require
week-to-month anticipation and aftermath windows. Keep these event families separate
rather than forcing one horizon grid to represent all of them.

### 6.3 Evidence-triggered `1m` directional replay

The `1m` lane is a microscope for a previously identified reaction pattern. It is not
a general search over every minute and it must not be used to select a few attractive
historical charts.

#### 6.3.1 Candidate selection and queue contract

Queue a pattern only when its parent evidence is direction-neutral and trader-readable,
for example repeated abnormal volume, pressure magnitude, range expansion, dwell,
crossing, or volatility around a causal level or exact cluster composition.

Before examining the signed `1m` path, freeze:

1. the exact qualifying rule;
2. every qualifying pair and UTC episode timestamp on the tested surface;
3. the causal level price, zone width, family, source timeframe, age, and role;
4. every cluster component and its independence/dependency group where relevant;
5. the approach side and higher-timeframe state known at contact;
6. the direction-neutral reaction measurement that caused the pattern to qualify;
7. episode de-duplication and minimum separation rules; and
8. whether the price reference was knowable at the time or is retrospective-only.

Do not queue an episode because its later up/down result looked clean. Normally freeze
`6-12` independent episodes for the first diagnostic replay. This is a bounded-work
range, not a statistical pass mark. Expand only through a later frozen batch if the
initial direct study exposes a coherent question rather than merely an attractive
example.

#### 6.3.2 Data window and storage

Use two nested context layers. The first is a dense `1m` replay envelope around the
actual interaction. The second is a coarser structural envelope long enough to explain
how the level was created and what other higher-timeframe conditions surrounded it.

Set the **causal anchor timeframe** as follows:

1. a single level uses its source timeframe;
2. a cluster uses the highest timeframe among its independently contributing
   components while retaining every component's individual timeframe;
3. a historically significant price reference uses the timeframe on which its frozen
   significance rule became knowable; and
4. a result may use a shorter anchor only when that choice was frozen before reading
   signed `1m` outcomes and the parent hypothesis is explicitly about the shorter
   component.

Use these default dense replay envelopes:

| Causal anchor | Pre-contact `1m` envelope | Post-contact `1m` envelope |
|---|---:|---:|
| `1h` | `24h` | `12h` |
| `4h` | `72h` | `48h` |
| `8h` | `7d` | `4d` |
| `1d` | `30d` | `14d` |

For an intermediate or larger anchor, use the next more conservative tier and record
the choice. These are defaults, not winning parameters to optimize. They provide
enough history to see whether the approach was building for hours or days and enough
aftermath to distinguish a brief reaction from a sustained move.

The structural envelope must additionally contain:

1. the exact source-timeframe lookback and warm-up used to calculate the level;
2. completed source and higher-timeframe candles needed to describe trend, volatility,
   volume, prior contacts, and nearby major levels;
3. synchronized BTC, ETH, and frozen cross-crypto context required by the hypothesis;
   and
4. optional external or microstructure history only where source readiness proves the
   interval was observed.

Use coarser candles for long construction history rather than downloading unnecessary
minute rows. For example, a rolling daily high built from `720` daily candles requires
that daily construction history, but normally only `30d` before and `14d` after the
contact need dense `1m` replay.

Before interpreting direction, run a boundary-sufficiency audit using only path length
and direction-neutral activity facts. The default window is insufficient when:

1. the measurable approach or abnormal participation already began before its left
   edge;
2. the contact/retest episode remains open at the right edge;
3. volume or volatility has not returned to a usable comparison state by the right
   edge; or
4. a nearby causal higher-timeframe event needed by the frozen hypothesis falls just
   outside the envelope.

If insufficient, expand both sides once—normally to twice the default—for **every**
episode in that pattern. Never extend only examples with an attractive signed path.
Any second expansion becomes a later branch decision at joint generation review.

Acquire data in this order:

1. inventory existing `1m` coverage by pair, market type, UTC interval, continuity,
   and source;
2. add feature warm-up before each visible pre-window;
3. merge overlapping or adjacent padded intervals for the same pair;
4. download only missing merged intervals through approved Freqtrade launcher/data
   tools, preserving the exact exchange and futures/spot contract of the parent event;
5. when the downloader can only maintain a broad continuous pair file, compare that
   cost with separate staged downloads and extracts rather than silently fetching the
   entire gap between distant episodes; and
6. verify earliest/latest timestamps, expected minute count, duplicates, gaps, and
   file readability before analysis.

Do not download all top-ten pairs merely because one queued pattern needs two pairs.
Keep reusable raw Freqtrade data in its approved data location when already present.
Keep compact manifests, episode inventories, interval coverage, commands, and hashes
on the normal project path. Put bulky deduplicated replay slices under the approved
`D:` research root and remove superseded temporary downloads/extracts only after the
retained replacement passes its coverage and hash checks.

#### 6.3.3 Causal feature blocks

Keep the blocks separate so each explanation can be removed and tested.

1. **Higher-timeframe anchor:** exact level/cluster, source timeframe, zone width,
   approach side, age, prior contacts, surrounding levels, and completed higher-
   timeframe market state.
2. **Pre-contact `1m` path:** signed returns, swing sequence, distance and speed toward
   the zone, acceleration/deceleration, compression/expansion, candle bodies, wicks,
   close location, gaps, and repeated tests.
3. **Volume and pressure:** volume relative to causal rolling baselines, acceleration,
   directional candle-volume proxies, OBV/CMF or another frozen pressure measure,
   pressure divergence, and whether participation broadens or fades.
4. **Momentum and volatility:** ATR, realised volatility, RSI, MACD/histogram, fixed
   moving-average distance/slope/order, ADX strength kept separate from direction,
   and distance from recent `1m` highs/lows.
5. **Cross-crypto state:** synchronized BTC/ETH movement, top-coin breadth, dispersion,
   common volume, and coin decoupling, all known before the decision minute.
6. **Optional microstructure:** spread, depth, imbalance, replenishment/depletion,
   large displayed bids/asks, executed buy/sell flow, trade-size concentration,
   liquidations, funding, or open interest only where timestamp-safe historical
   coverage genuinely exists.
7. **Optional external context:** news, macro, global-market, or web context only under
   source-specific readiness and availability masks.

Do not describe a large displayed order as a whale position or intention. It may be
cancelled, moved, or spoofed. OHLCV does not identify whales. Report observable large
liquidity, executed flow, liquidation, or open-interest evidence under its actual
source limitations.

#### 6.3.4 Directional outcomes

Measure the path before compressing it into a label:

1. signed displacement on a common micro grid of `1m`, `3m`, `5m`, `10m`, `15m`,
   `30m`, `60m`, and `120m`;
2. maximum upward and downward excursion normalized by pre-contact `1m` ATR and zone
   width;
3. which side first reaches each symmetric movement threshold;
4. time to first break, rejection, reclaim, retest, or return to the zone;
5. breakout-before-rejection and rejection-before-breakout order;
6. path balance between upward and downward excursion;
7. persistence after the first move and whether price remains outside or returns;
8. dwell and recrossing around the zone; and
9. contemporaneous and subsequent volume/pressure response kept separate from signed
   price direction.

Also freeze source-scaled checkpoints through the end of the post-contact envelope:

| Causal anchor | Additional path checkpoints |
|---|---|
| `1h` | `4h`, `8h`, `12h` |
| `4h` | `4h`, `8h`, `12h`, `24h`, `48h` |
| `8h` | `4h`, `8h`, `12h`, `24h`, `48h`, `96h` |
| `1d` | `4h`, `8h`, `12h`, `24h`, `48h`, `96h`, `7d`, `14d` |

These checkpoints describe one continuous path; they are not independent events and
must not be counted as repeated evidence. Do not choose a different best checkpoint
for each episode after viewing its direction.

Post-contact volume, orderbook, pressure, or momentum can help explain how the path
developed. It cannot be used as a feature for a prediction made at first contact. To
test a later update decision, create a separately frozen decision timestamp—such as
`15m` after contact—use only information available by then, and evaluate outcomes
strictly after that timestamp.

Regression and time-to-event targets come first. A later breakout/rejection class may
be used only after its continuous path definition is frozen. `Profitable after N
minutes` is not an acceptable first target.

#### 6.3.5 Direct controls and model ladder

Run direct aligned-path comparisons before FreqAI:

1. the qualifying real contacts;
2. matched `1m` noncontact windows with similar pre-contact regime, volatility,
   volume, approach speed, and local path;
3. stale and symmetrically shifted level locations;
4. the same higher-timeframe regime and recent path without the current level;
5. the level/cluster without the proposed pressure or momentum condition;
6. current versus delayed microstructure fields when a like-for-like delay is valid;
   and
7. leave-one-episode, leave-one-date, and leave-one-coin checks where support permits.

Only when support is sufficient for chronological training and untouched evaluation,
use this smallest useful FreqAI ladder:

1. pre-contact `1m` path and local OHLCV baseline;
2. higher-timeframe and cross-crypto regime without exact level identity;
3. level or cluster state without detailed `1m` approach information;
4. regime plus `1m` approach without the current level;
5. additive regime, approach, and current-level blocks;
6. one explicitly named low-order interaction justified by the direct study;
7. matching stale/shifted-placebo representation; and
8. optional orderbook/flow or external-context block as a final ablation, never as an
   assumed explanation.

The current level interaction must improve beyond both the regime/approach model and
the level-only model, and it must outperform its matching placebo on the same rows.
Feature importance alone is not evidence.

#### 6.3.6 Interpretation and stopping

The replay may establish a directional lead, conditional lead, activity-only result,
representation failure, insufficient coverage, or not-reproduced pattern. It cannot
establish a trading action.

Stop and return to the main objective when:

1. the pattern was selected using future direction or remembered examples;
2. fewer independent episodes remain after de-duplication than the claim requires;
3. `1m`, orderbook, or external-source gaps cannot be distinguished from ordinary
   state;
4. regime/recent path explains the direction and the level adds nothing;
5. the current level matches its stale/shifted placebo;
6. one episode, date, coin, or threshold supplies the apparent result;
7. the initial direct pass and one frozen model ladder fail; or
8. the work begins tuning parent indicators to the selected episodes.

Record any further question for the joint generation review. Do not allow a replay to
spawn its own immediate branch.

## 7. Target Families

Separate target families in both modelling and interpretation.

### 7.1 Reaction magnitude and timing

1. absolute or ATR-normalized reaction magnitude;
2. time to first fixed movement threshold;
3. unconditional probability of reaching the threshold;
4. censored time-to-threshold when it is not reached;
5. volume and volatility change; and
6. pressure change.

These targets answer whether the level appears relevant. They do not establish direction.

### 7.2 Direction and path

1. signed return at fixed horizons;
2. favourable and adverse excursion;
3. path balance;
4. target-before-invalidation;
5. breakout versus rejection;
6. continuation versus reversal;
7. favourable-before-adverse ordering; and
8. peak timing and persistence.

These targets are required before calling a relationship directional or action-relevant.

### 7.3 Exit and position-management targets

When evaluating exits, keep avoided adverse excursion, forfeited favourable excursion, net exit regret, peak order, target progress, remaining fraction, and trade age separate. A level may be useful as a partial, warning, stop-tightening signal, runner cue, or invalidation without being a primary full exit.

## 8. Required Test Ladder

Use the smallest useful ladder that preserves attribution. A suitable initial branch batch is:

1. **Direct matched-state baseline:** compare level contacts with comparable non-level states and shifted/stale levels before relying on ML.
2. **Price/structure control:** coin-local OHLCV and structure without broad-market or level identity.
3. **Broad-regime model:** broad crypto state without the exact level.
4. **Recent-path model:** coin-local trend and approach path without the exact level.
5. **Regime-plus-path model:** combine broad and local directional context without the level.
6. **Level-only model:** exact level state without broad regime and detailed approach path.
7. **Additive model:** regime, recent path, and level supplied as distinct blocks.
8. **Explicit interaction model:** only the predeclared regime/path x level relationships.
9. **Ablations:** remove broad regime, local path, approach behaviour, level identity, timeframe, volume, pressure, or freshness one block at a time.
10. **Chronological and cross-coin validation:** evaluate the frozen ladder on later windows and the frozen top-10 surface.

Each layer must be compared with every immediately simpler model required to answer
its exact claim. Independent/additive claims require broad incremental performance;
modifier claims require a conditional change in the primary relationship with versus
without the modifier. An interaction is not supported merely because the largest
combined model performs best against an OHLCV-only baseline.

For the event-driven hierarchy, use the corresponding attributable ladder:

1. matched OHLCV and slow-background baseline with no qualifying event;
2. event-only model without the early market response;
3. BTC-only, ETH-only, and broad-participation candidate-leader models;
4. event plus the frozen market-leader representation;
5. group-transmission model without coin-local modifiers;
6. coin-local level/activity/attention/liquidity model without event or leader context;
7. frozen pairwise event, leader, group, or local relationships;
8. no more than three retained blocks in the complete hierarchy model;
9. remove each block in turn; and
10. confirm on later whole events and realistic source-outage states.

An event hierarchy is not supported unless each claimed link meets the comparison
required by its role: incremental lift for independent/additive links, or a repeatable
conditional change for a modifier link. A model that only recognizes an already large
simultaneous move is market-state description or later confirmation, not event
prediction or leader-to-group forecasting.

## 9. Required Controls

Use controls appropriate to the exact question:

1. same regime and recent path without a nearby level;
2. matched ordinary candles from the same chronological window;
3. an exact level shifted by a predeclared interval such as `168h`;
4. stale levels represented with the same fields as current levels;
5. same level type approached from the opposite direction;
6. regime-only and level-only baselines;
7. broad-market-only versus coin-local-only ablations;
8. missing-ingredient ablations;
9. shuffled labels for model sanity where appropriate;
10. repeated-event episode de-duplication; and
11. pair, side, period, and timeframe concentration checks;
12. matched no-event times under a similar slow background;
13. high-news-activity times without an independently confirmed major event;
14. similar BTC or broad-market movement without the named event;
15. the named event without a confirming early market response;
16. the same local coin state without market-leader movement;
17. deliberately false event times, leaders, or group membership; and
18. whole-event, source, coin, and event-family concentration checks.

The shifted/stale control must use a like-for-like feature contract. A weak or malformed placebo cannot justify the live level.

## 10. Evidence And Interpretation Rules

### 10.1 Level-reaction evidence

A level may be called a reaction-point lead when direct and model evidence indicates repeatable changes in magnitude, timing, volume, volatility, pressure, rejection, or path relative to matched and stale/shifted controls.

This classification must say exactly which behaviour changed. Do not shorten it to "the level works."

### 10.2 Directional-context evidence

Regime or recent path may be called a directional lead only when signed direction, path balance, target-before-invalidation, favourable/adverse excursion, or breakout/rejection ordering improves across suitable coins and chronological windows.

### 10.3 Interaction evidence

An independent/additive directional interaction is supported only when the interaction
model adds coherent directional or path information beyond both:

1. the regime/recent-path model without the level; and
2. the level-only model without the directional context.

A conditional modifier has a different burden: show that the primary relationship
changes with versus without the modifier on comparable eligible rows, then reproduce
that difference on later whole episodes. The modifier does not need to predict the
outcome broadly by itself. Report whether it changes probability, magnitude, timing,
path, or transmission and how often the relevant condition occurs.

Activity, faster threshold contact, greater absolute movement, or higher volume cannot be relabelled directional interaction evidence.

### 10.4 Portability

Judge consistency at the scope actually being claimed:

1. A **broad top-coin claim** should normally require majority-consistent evidence across at least seven suitable coins and at least two chronological FreqAI windows.
2. A **coin-group claim** may remain a valid conditional lead with fewer coins when the group is measurably or economically coherent, was defined before reading the tested outcomes or frozen after development-only discovery, repeats across at least two windows, and is not dominated by one member.
3. An **asset-specific claim**, including a BTC-only lead, may remain valid when it repeats across periods and later unseen data, but it must not be described as a general crypto edge.

Failure to reach seven of ten means `not broad across the tested top-coin universe`; it does not automatically reject a coherent group-specific or asset-specific effect. Conversely, do not select the positive coins after seeing the result and call that confirmation. A result-discovered group is a later hypothesis that requires a frozen repeat test.

For a two-coin group, both coins should agree before calling it a group lead. For a three-or-more-coin group, use a predeclared clear-majority rule, report every member, and test whether one coin supplies most of the pooled effect. Exact group membership, support, effect, window, and stopping criteria must be frozen in the branch manifest before execution.

Meme-coin work uses a separate frozen cohort of the `10` most traded eligible meme
coins on the selected venue and market type. Determine the ranking before event
outcomes are inspected using median daily quote turnover over the prior `30` completed
UTC days. Record taxonomy source, venue, market type, ranking timestamp, eligibility
rule, and every member/rank. Do not replace an inadequately covered member after
opening outcomes; report the coverage limitation and refresh only in a later frozen
batch. Compare the cohort with suitable non-meme controls and report every member.

For event-driven meme work, estimate each member's ordinary response magnitude and lag
from earlier data before scoring the event. Report raw response and extra response
beyond that ordinary relationship separately. A meme cohort does not amplify a market
move merely because its normal volatility produces a larger percentage change.

Use `reference_freqai_event_reaction_research_method.md` for the complete broad-market, coin-group, asset-specific, clustering, leave-one-coin-out, and post-hoc-grouping procedure.

### 10.5 Valid classifications

Use explicit conclusions such as:

1. `level_reaction_only`;
2. `regime_direction_only`;
3. `recent_path_direction_only`;
4. `context_dependent_directional_level`;
5. `breakout_context_lead`;
6. `rejection_or_reversal_context_lead`;
7. `activity_without_direction`;
8. `redundant_with_price_state`;
9. `inverse_or_regime_dependent`;
10. `insufficient_support`;
11. `not_reproduced`; or
12. `externally_confounded_or_unresolved`;
13. `coin_group_specific_lead`; or
14. `asset_specific_lead`;
15. `conditional_modifier`;
16. `confirmation_only`;
17. `amplified_suppressed_delayed_or_reversed`;
18. `dominant_override_candidate`; or
19. `unexplained_episode_difference`.

### 10.6 Separate Useful Components And Combined Outcomes

For every issued directional call at a qualified event, market-confirmation, level, or
post-event-range context, freeze and score:

1. whether the declared abnormal volume reaction occurred;
2. whether the declared signed direction/path occurred; and
3. whether both were correct on that same call.

The third value is the joint success rate; either component failing makes that joint
call wrong. Under the user-approved 19 September 2026 revision, it is NOT the gate for
retaining each useful component. Follow Objective 02b Section 15.4: directional bias
may qualify without abnormal volume, and a reaction-location or activity input may
qualify without direction. A justified modifier may qualify within its known context
without being a strong standalone predictor of either outcome.

Report direction on all issued directional calls separately from direction among
later reacting cases, plus activity, relevant joint rates, coverage, abstention,
independent support and uncertainty. Compare the stated role with appropriate simple
controls and later data. A combined predictive claim must demonstrate its own gain
against its applicable ingredients on common episodes; a modifier claim must show its
conditional effect. Do not infer combined accuracy from multiplying input accuracies.
The directional 55%/65% aspirations are not universal joint or activity floors.
Preserve old frozen verdicts with dated supplementary component assessments. This
relaxes the joint requirement, not evidence standards or trading safeguards.

### 10.7 Whole-Episode Conditional Evidence

Do not summarize a complex episode only as `event up`, `event down`, `worked`, or
`failed`. For each retained relationship, report:

1. the timestamp-safe longer background and rolling narrative balance;
2. what was expected or already reflected in price;
3. the new information and the pressure it plausibly added;
4. initial market confirmation or rejection;
5. whether technical structure, liquidity, positioning, or concurrent stories
   amplified, suppressed, delayed, reversed, shortened, or overrode the response;
6. whether the response propagated from the market leader into coin groups;
7. the raw outcome and the change from matched expected-path controls;
8. the response shape across the complete horizon grid; and
9. strong, weak, delayed, reversed, and absent episodes that remain unexplained.

The overall average is a starting description. A weak average can contain a genuine
conditional relationship, and a positive average can conceal rational failure states.
Use development-only contrastive analysis to identify possible conditions, freeze
them, and require later whole-event confirmation. Do not create context partitions
repeatedly until a positive subset appears.

Before accepting an interpretation, state separately:

1. direct observation;
2. proposed mechanism;
3. plausible competing explanation;
4. evidence scope and independent episode count;
5. conditions under which the influence appears stronger, weaker, or reversed;
6. whether those conditions were predeclared, development-discovered, or later
   confirmed; and
7. the evidence decision changed by the result.

## 11. Examples Of How To Interpret Outcomes

### 11.1 Broad uptrend approaching resistance

If top-10 breadth, BTC direction, coin-local structure, approach pressure, and volume are aligned upward, resistance contact may become a breakout candidate. The test must compare:

1. the same directional context away from resistance;
2. resistance without the aligned directional context;
3. stale or shifted resistance; and
4. the explicit context x resistance interaction.

If only reaction magnitude rises, classify it as a reaction point. If signed continuation and target-before-invalidation improve beyond both component models, it may become a contextual breakout lead.

### 11.2 Broad downtrend approaching support

Support contact may still create a bounce, stall, or temporary rejection, but the wider downtrend may favour eventual breakdown. Separate immediate rejection from later continuation across the horizon grid. Do not label a brief bounce as a durable reversal or a later decline as proof that support was irrelevant.

### 11.3 Coin trend conflicts with the wider market

A coin may be locally strong while the wider market is weak, or vice versa. Test whether decoupling changes level response rather than forcing one source to have automatic precedence. This may identify either relative-strength continuation or increased failure risk.

### 11.4 Higher-timeframe level near a lower-timeframe event

The higher-timeframe level may define the structural reaction zone while the lower-timeframe path controls timing. It may instead be stale or too distant to matter. Higher timeframe does not receive automatic precedence; test the interaction and ablate timeframe identity.

### 11.5 Multiple nearby entry and exit ideas

An entry event near an opposing exit or obstacle may have too little usable room even when the regime supports direction. Same-direction evidence may support continuation, or it may duplicate the same price move. Model normalized distance, ordering, independent-family identity, and available corridor explicitly.

## 12. Timestamp And Missing-Data Safety

1. Every feature must be knowable at the decision candle.
2. Future outcomes begin after the decision boundary and must never appear in features.
3. Use only completed higher-timeframe candles.
4. Cross-coin features must be aligned to information available at the same decision time.
5. Preserve missing source data as missing and include a valid availability representation when needed.
6. Zero may represent a genuine zero count or inactive state only when the source was observed.
7. Do not call missing news or wider-market data quiet, normal, balanced, or inactive.
8. Repeated adjacent event firings must be de-duplicated into episodes or modelled as dependent observations.
9. Chronological splits must remain sealed; do not repeatedly tune against the final holdout.

## 13. External Context Boundary

Crypto-specific OHLCV may reveal conditional technical patterns when wider external conditions are relatively undisturbed. Objective 02b now actively investigates external events and slow background, but only inside news, macro, global-market, and liquidity source windows that have passed their own readiness and timestamp checks.

Until then:

1. treat unexplained variation as potentially externally confounded;
2. do not claim a period was quiet merely because no external data is present;
3. do not reject a coherent technical lead solely because an unobserved external shock may have distorted some occurrences;
4. do not rescue weak technical evidence by vaguely blaming news; and
5. coverage-park unsupported event families instead of converting missing observations
   into neutral states; and
6. keep aggregate event activity separate from story-level event meaning when the
   latter cannot be reconstructed.

## 14. FreqAI Role And Limits

FreqAI is suitable here because the proposed interaction may be nonlinear, continuous, multi-timeframe, and conditional. It can estimate several named outputs together and compare complex but frozen feature blocks.

Its preferred interpretation is a sequence of probability updates. Background sets
the starting distribution; event information changes it; immediate market response
updates it at a later decision time; technical, liquidity, group, and coin-local
conditions modify it. The model must be allowed to retain conflicting evidence and
abstain rather than force one permanent rule or deterministic direction.

FreqAI must not:

1. replace exact Sieve entry or exit identity with an opaque master score;
2. consume every available indicator and discover an explanation afterward;
3. use profit at one future candle as the only target;
4. turn proximity into an automatic action;
5. interpret feature importance as causation;
6. infer direction from activity targets;
7. promote a one-pair result as a general edge; or
8. activate live or dry-run trading decisions from exploratory outputs;
9. treat a post-event confirmation variable as a rival root cause of the event-driven
   response; or
10. discard a rational modifier solely because its unconditional average is weak.

If this theory survives direct controls and FreqAI validation, later bounded uses may include:

1. enter versus skip;
2. confidence or bounded sizing input;
3. continuation versus rejection expectation;
4. add, hold, or runner permission;
5. opposing-obstacle caution;
6. tighten, reduce, or partial evidence; and
7. exit or invalidation support.

Those actions require separate deterministic integration, risk rules, ablations, and full-system validation.

## 15. Falsification And Stopping Conditions

Park or reject the tested representation when:

1. current levels do not differ from matched no-level, stale, or shifted controls;
2. regime/path models explain the result and the level adds no incremental information;
3. the level model explains activity but no directional interaction survives;
4. an independent/additive combined claim fails its immediately simpler component
   models, or a declared modifier fails to change the primary relationship within its
   frozen context;
5. apparent lift is concentrated in one coin, side, period, or fragile threshold;
6. adjacent definitions reverse the result without a coherent reason;
7. support is too sparse for the declared claim;
8. future leakage, incomplete higher-timeframe candles, duplicate episodes, or invalid missing-data handling is found;
9. the trader-readable question cannot be recovered from the features and targets; or
10. repeated bounded refinements fail and further work becomes post-hoc rescue.

A failed model rejects that exact representation and tested scope. It does not prove that every possible market-regime or level interaction is false. Record what was tested, what failed, and whether a materially different representation has an evidence-backed reason to become a later branch.

## 16. Required Experiment Record

Every batch derived from this theory must record:

1. hypothesis ID and generation/batch ID;
2. trader-readable question;
3. frozen pairs, market type, periods, and timeframes;
4. exact regime, recent-path, level, and interaction feature blocks;
5. event-time boundary and higher-timeframe alignment;
6. targets and common horizon grid;
7. direct, model, stale, shifted, and ablation controls;
8. setup, event, episode, and positive-target support;
9. duplicate dates, gaps, missingness, and infinite-cell checks;
10. model profiles and identifiers;
11. pair/window portability;
12. activity, direction, path, and timing conclusions kept separate;
13. contradictions and plausible confounders;
14. exact evidence artifact paths;
15. terminal verdict; and
16. any held next-generation branch question.

For every held question also retain its source result and artifact, observation in
ordinary market language, route family, alternative explanation/control, required
coverage, smallest useful test, expected cost, earliest eligible generation, and queue
status. Mark ideas observed before batch completion as provisional.

## 17. Batch-Gating Rule

This theory does not bypass breadth-first research generations.

1. Finish every batch in the active generation before allowing any one result to redirect the programme.
2. Jointly review the whole generation and distinguish supported relationships, unresolved attribution, contradictions, and failed representations.
3. Complete the event-history construction layer, then every single-link sibling,
   before admitting pairwise relationships; complete all frozen pairs before admitting
   limited three-block chains.
4. Consolidate it with every other valid candidate for that generation rather than promoting it ahead of the queue.
5. Freeze the complete next-generation scope, controls, criteria, outputs, and stop conditions before executing any of its batches.
6. Execute every frozen batch in that generation before following any deeper result.

Maintain a rolling candidate queue while work runs. A new indicator, mathematical
representation, timeframe, convergence pattern, level definition, coin/cohort, or data
source suggested by either an interim or final result is valid to capture immediately,
but it cannot launch or alter the frozen surface. At the joint review, confirm interim
ideas, deduplicate equivalent mechanisms, compare alternatives and controls, then
freeze a balanced later batch. Questions produced by that batch wait for the following
review.

While the objective remains unresolved, assess at least five materially different
routes and eventually terminally test at least five. The portfolio may include
isolated levels; level clusters/convergence; rational alternative level mathematics;
multi-timeframe regime/path/obstacle interactions; frozen asset/cohort portability;
evidence-triggered `1m` OHLCV/pressure/flow behaviour; and source-ready external
context. The current event-driven portfolio also includes slow background, major-event
state, market leadership, coin-group transmission, genuine meme amplification, local
modification, and post-event ranges. Five parameter changes within one mechanism do not
count. The five-route minimum is independent of the five-layer branch authorization.

The earlier Sieve-linked Generations 1 and 2 did not identify the required
non-redundant level representation. That historical outcome remains a warning against
retrospective rescue; it does not decide the current market-first level and cluster
programme.

Objective 02b now permits both its five-layer event-driven hierarchy and
evidence-triggered `1m` replay. The replay remains a bounded microscope side lane: it
may not redirect an unfinished main batch, spawn an immediate descendant, or alter
parent indicators. Its result waits for the joint generation review.

Up to five branch layers are pre-approved only when each layer asks valid evidence-backed questions. Do not manufacture layers merely because authorization exists.

## 18. Relationship To Other Guidance

Apply this theory together with:

1. `../01_objectives/objective_02b_market_reaction_zone_discovery.md` for the active
   reaction-zone confirmations, five-layer event-driven hierarchy, and bounded
   evidence-triggered `1m` replay;
2. `reference_freqai_event_reaction_research_method.md` for the reusable detailed event-reaction method and hierarchical market-scope rules;
3. `rules_freqai_feature_discovery.md` for conditional-market interpretation and initial feature discipline;
4. `rules_direct_tests.md` for controls and trader-readable metrics;
5. `rules_freqai_promotion.md` for later promotion safeguards;
6. `rules_entry_exit_position.md` for entry, exit, and position-action meaning; and
7. `rules_runtime_environment.md` for frozen data, worker lanes, and long-run ownership.
