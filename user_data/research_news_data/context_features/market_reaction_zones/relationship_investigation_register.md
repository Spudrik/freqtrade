# Market Reaction Relationship Register

Last updated: 2026-08-22

## Purpose

This is the readable running record of what the market-reaction programme has tested.
It records unsuccessful, contradictory, under-supported, and pending questions as well
as useful leads. It does not convert a reaction into a trading entry, exit, or direction
prediction.

Names such as G2D, G2E, and G2F are only short file/run identifiers. They do not mean
that a test passed, reached a quality grade, or received a machine-learning score. The
plain-language heading and conclusion define what each test actually investigated.

For every investigation, this register states:

1. the calculated level, indicator, or cluster;
2. every source and observation timeframe involved;
3. additional inputs used to make comparisons fair;
4. what was measured after price reached or approached the area;
5. which artificial, stale, shifted, isolated, or no-level controls were used;
6. what happened on established large coins;
7. what happened on the separately frozen meme-coin group;
8. what the evidence does and does not mean; and
9. the next test, if one is justified.

## Market groups

### Established large-coin group

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, TRX, AVAX, and LINK perpetual futures. BTC is
reported separately where its behaviour differs materially from the other coins.

### Frozen active meme-coin group

DOGE, PEPE, PUMP, SHIB, TRUMP, PENGU, BONK, FARTCOIN, WIF, and ORDI perpetual
futures. This group was frozen before its reaction outcomes were opened. The selection
used prior 30-completed-day Binance futures turnover among assets independently
classified as meme coins.

The two groups are not interchangeable. A relationship may be broad, specific to a
rational subgroup, or unsupported. It must be labelled honestly.

Their standard chronological periods are also different because several meme coins
have much shorter histories. Established-coin early validation runs from January 2024
to April 2025 and late validation from April 2025 to April 2026; its April-July 2026
period is diagnostic. Meme validation uses January-April 2026 and April-July 2026.
Therefore a full-period normal result must not be compared directly with a meme result
as if only coin type changed. Cross-group interpretation will first use shared calendar
dates where both actual and control events fall inside the same overlap window. If that
subset is too small or its matching was inherited from a broader period, it remains a
diagnostic and a later same-window rematch must be frozen before claiming a normal/meme
difference.

DOGE appears in both frozen groups. It is useful as a bridge showing how the same asset
looks under each cohort context, but it must not be counted twice as independent support
for a normal-versus-meme difference. Also, `top-ten breadth`, movement, dispersion, and
common volume describe the established group in one run and the meme group in the
other. They are valid within-cohort matching inputs but are not identical cross-cohort
variables. A later confirmatory group comparison must use the same shared dates and a
common external-market context, while retaining cohort-specific context as a separately
named input.

## Inputs commonly used to make comparisons fair

The direct comparisons match or balance the following information when applicable:

- recent 1-hour, 4-hour, and 24-hour price change;
- recent trading range and ATR-scaled volatility;
- recent volume relative to its trailing baseline;
- recent candle pressure calculated from OHLCV;
- RSI, Bollinger position and width, MACD histogram, and EMA distance;
- the state of the indicator on its own source timeframe;
- BTC's recent state;
- breadth, average movement, and dispersion across the coin group;
- the number of other calculated levels near price;
- the side from which price approached the zone;
- distance from price to the zone before contact; and
- zone width relative to ATR.

These are controls, not automatically useful predictors. A test records when they do
or do not change the relationship.

## Completed and active relationships

### 1. Nearest 8-hour round number when another independent level is nearby

Calculated object: the nearest psychologically round price calculated on the 8-hour
timeframe. The focused case requires a separately calculated level to be close enough
to make the round number a member of a tight cluster.

Crossed indicators and timeframes: 8-hour round number crossed with an independently
generated level from the multi-timeframe level atlas. Exact companion families are
retained in event-level data rather than treated as interchangeable.

Measured behaviour: contact-candle range and volume, largest absolute movement during
the next hour, volume over the next four hours, and one-hour crossings.

Established large coins: the earlier location test across ten coins found more
activity than shuffled and symmetrically shifted locations, but lacked a fair cluster
without a round number. The new comparison supplied that missing control. DOGE's
source OHLCV had changed after its old cache was built, so the cache validator
correctly rejected it; the current comparison therefore uses the other nine coins and
is diagnostic rather than a replacement ten-coin result. There were 960 independent
development comparisons, 421 early-validation comparisons, and 345 late-validation
comparisons. Relative to equally constructed independent clusters containing no round
number, round-number clusters had more contact range and volume in development and a
small positive difference in early validation. Every main activity measure reversed
in late validation: median contact range was about 0.09 trailing-range units lower,
contact volume was about 0.19 trailing-volume units lower, and four-hour volume was
about 0.15 units lower. State balance was good, so the reversal cannot simply be
dismissed as obviously incomparable prior conditions.

Meme coins: the cluster-without-round comparison contained 223 independent
development pairs, 102 early-validation pairs, and 100 late-validation pairs across
all ten coins. The contact candle did not show stable extra range or volume. However,
the largest absolute movement during the next hour was higher in early validation by
about 0.28 ATR in 8 of 10 coins and in late validation by about 0.20 ATR in 9 of 10.
Four-hour volume was also moderately higher in those two periods. This followed a
negative development result and did not include a stable contact-volume increase.

Normal-versus-meme difference: normal coins showed a development/early immediate-
activity difference that disappeared and reversed late. Memes showed little stable
contact-candle difference but a later one-hour movement-magnitude difference in the
two validation periods. This is evidence that market group and period change what a
round-number cluster represents; it is not evidence for one universal round-number
reaction rule.

Current meaning: the round number has not passed its frozen requirement of beating
every required control with the same behaviour in repeated validation periods. A
narrow meme hypothesis about post-contact movement magnitude remains a valid later
branch, but it must be tested as a new conditional question rather than used to rescue
the original broad claim.

### 2. Four-hour high-volume-node strength

Calculated object: high-volume nodes from the existing Volume Profile, with the
indicator's emitted node-strength value.

Source timeframe and zone: 4-hour Volume Profile with a tight contact zone.

Measured behaviour: two-hour volume, four-hour range, and four-hour crossing count.
The expected acceptance pattern is lower range and volume with more repeated crossing
or residence near a stronger high-volume node.

Established large coins: stronger nodes initially ordered behaviour in the expected
acceptance-like direction and beat shuffled strength values in the Generation-1
screen.

Meme coins: the stronger chronological test did not reproduce the expected ordering
in both validation periods. Early validation was under-supported and late validation
was inconsistent. Exact-location comparisons have adequate evidence for several
controls but do not show one stable response against all shifted locations.

Current meaning: the original broad lead has weakened on the independent meme group.
It remains under review until near-miss, matched-no-level, stale, and opposite-role
comparisons are jointly closed; it is not currently a repeatable lead.

### 3. One-hour low-volume-node thinness

Calculated object: low-volume nodes from the existing Volume Profile, ordered by how
thin or low-participation the indicator says the node is.

Source timeframe and zone: 1-hour Volume Profile with a wide contact zone.

Measured behaviour: contact-candle volume, time spent near the zone over four hours,
and total volume over 48 hours.

Established large coins: thinner nodes showed more activity/transit-like behaviour.
Contact-volume ordering repeated across ten coins and three periods and beat shuffled
thinness values.

Meme coins: this is the strongest current attribute result. In early validation,
thinner nodes had higher contact volume in 9 of 10 coins and higher 48-hour volume in
8 of 10. In late validation, those counts were 6 of 10 and 7 of 10. Shuffling which
nodes were labelled thin removed much of the relationship. Omitting each coin in turn
did not destroy it. Four-hour dwell did not reproduce consistently in that attribute
ordering test.

The stronger contact-location comparison adds a separate result. Actual entry into a
thin LVN was compared with price approaching the same live LVN to within two zone
widths but not entering it, after matching prior price, volatility, volume, pressure,
technical state, broad-market state, approach side, level density, distance, and zone
width. In early validation there were 92 independent pairs across all ten coins; in
late validation there were 104. Actual contacts had a contact-volume ratio higher by
about 1.19 and 1.36 trailing-median-volume units respectively, with a positive
difference in every coin in both periods. Mean volume over the following 48 hours was
higher by about 0.25 and 0.14 trailing-median-volume units, positive in 9 of 10 and 8
of 10 coins. Actual contacts also spent less of the next four hours near the LVN in
both periods. Omitting any one coin left every median effect with the same sign.

A matched pseudo-zone with no selected real level nearby could not provide adequate
LVN coverage. The wide LVN contacts in this sample were almost always also members of
a broader level cluster. This is an evidence limitation, not a negative result.

Current meaning: thinness contains repeatable information about subsequent trading
activity across two different coin groups, and actual entry into the live zone marks
more activity than a closely matched near miss. In the current representation this is
a cluster-conditioned activity/transit lead. It does not yet prove that the exact LVN
price is uniquely special relative to every no-level or stale-location control, and
it says nothing about whether price will bounce or break through.

Next required comparison: cluster-component attribution, improved stale-location
coverage, and a fair no-level control that does not remove nearly the entire eligible
LVN surface. Any revised representation must be motivated by this coverage problem,
not tuned to preserve the observed events.

### 4. One-hour point-of-control profile evidence

Calculated object: the Volume Profile point of control, ordered by the amount and
quality of profile evidence supporting it.

Source timeframe and zone: 1-hour Volume Profile with a tight contact zone.

Measured behaviour: volume and range over 24 and 48 hours. The proposed acceptance
relationship expects stronger evidence to be associated with quieter price activity
or greater residence around the point of control.

Established large coins: stronger profile evidence showed a conditional
acceptance-like ordering, particularly for 48-hour volume, and usually beat shuffled
evidence values.

Meme coins: early validation did not contain enough independent paired episodes for
the frozen cohort threshold. Late validation produced mixed signs rather than one
clear acceptance pattern. Nearby shifted locations also behaved similarly in several
comparisons.

Current meaning: insufficient and inconsistent on the meme group. Do not treat the
original large-coin result as broadly confirmed.

### 5. One-hour recent extreme crossed with the matching four-hour Bollinger boundary

Calculated objects:

- 1-hour rolling 24-hour low plus the 4-hour lower Bollinger Band; and
- 1-hour rolling 24-hour high plus the 4-hour upper Bollinger Band.

Crossed indicators and timeframes: a short-timeframe recent price extreme and a
slower volatility envelope touching at approximately the same price. Upper and lower
patterns are kept separate and then compared as mirrored forms of the same idea.

Measured behaviour: contact range and contact volume beyond what followed contact
with each individual component during comparable market conditions.

Additional inputs used for fair matching: recent 1-hour, 4-hour, and 24-hour returns;
ATR; recent range and volume; six-hour OHLCV pressure; RSI; Bollinger position and
width; MACD histogram; EMA50 distance; BTC 24-hour state; group breadth, average
movement and dispersion; number and density of other levels; complete cluster width;
approach side; and pre-contact distance. RSI, MACD, EMA, pressure, and broad-market
fields were matching inputs here, not claimed causal indicators.

Controls checked: contact with only the rolling extreme while the two zones were
connected; contact with only the Bollinger boundary; eight symmetric shifts from
minus two to plus two ATR that did not overlap any selected real level; the explicitly
paired artificial cluster; other independent component combinations at comparable
state; and same/shared-mechanism clusters. Matches from either side of the level were
kept separate before pooling. Actual and control future paths were separated by at
least five hours and then purged so four-hour outcome windows did not overlap.

Established large coins: the current full comparison uses BTC, ETH, BNB, SOL, XRP,
ADA, TRX, AVAX, and LINK. DOGE was parked because its OHLCV changed after the source
cache was built; stale-cache protection was not bypassed. Across the nine usable coins
there were 24,544 raw matched comparisons and 12,975 non-overlapping comparisons. In
development, both lower and upper patterns had much larger contact range and volume
than either component alone. For example, the lower pattern's median contact range
was about 0.75 to 0.86 trailing-range units higher and contact volume about 0.99 to
1.10 trailing-volume units higher, positive in all eight eligible coins. The upper
pattern showed similarly large development differences. These decisive component
and paired-artificial controls did not retain enough eligible coins and independent
episodes in the validation periods. Comparisons with ordinary independent clusters
had much more coverage, but changed sign: the lower pattern was below ordinary
clusters in early validation and mildly above them late; the upper pattern was mildly
above early and below late. Several shifted controls looked positive but had poor
state balance, so they cannot override the component-control coverage failure.

Meme coins: all ten pair workers passed integrity. The atlas supplied 3,611 raw
matches and 2,403 independent comparisons. Contact with just one of the two named
components, the explicitly paired artificial clusters, and most shifted locations
were too sparse to pass the frozen cohort gate in validation. Against ordinary
independent clusters, the lower combination moved from mostly negative development
differences to modest early positives and mixed late results. The upper combination
also changed by period: immediate range and volume were higher in early validation,
while its four-hour range and volume were lower; late results were small or mixed.
Shared-mechanism comparisons had poorer state balance and also failed to give the same
range-and-volume relationship in both validation periods.

Normal-versus-meme difference: established coins supplied far more historical
episodes and showed a striking development-only immediate-activity separation from
single components. Memes supplied fewer fully controlled episodes and mostly small,
period-dependent differences from other clusters. Neither group produced the frozen
requirement: both mirrored patterns beating every component and every required
control for range or volume in repeated validation periods.

Current meaning: park the broad mirrored-cluster claim. The original 12-13-path lead
was useful enough to justify the stronger test, but it did not become repeatable
cross-market evidence. Development-only component separation and the opposite early
versus late behaviour can later motivate a regime-conditioned branch, after all
frozen Generation-2 questions are complete. No FreqAI cluster rerun or trading rule is
justified from this result.

### 6. Broad collections of native levels, Volume Profile fields, clusters, and market state

Inputs compared: local OHLCV behaviour; BTC and top-ten market state; generic and
custom level identities and attributes; Volume Profile fields; and cluster fields.

Exact local and generic-indicator inventory: the one-hour contact stream supplied
lagged 1-hour, 4-hour, and 24-hour returns; ATR as a percentage of price; candle body
and closing position; prior range relative to its 24-hour median; prior volume relative
to its 24-hour median and its 24-hour z-score; six-hour and 24-hour volume-weighted
candle pressure and their difference; RSI(14); MACD(12,26) and its signal-histogram,
both scaled by ATR; price distance from SMA(20), SMA(50), SMA(200), EMA(12), EMA(26),
and EMA(50); ADX(14); and 24-hour realised volatility. These were market-context
inputs, not assumed reaction levels.

Exact wider-market inventory: Bitcoin's completed 1-hour and 24-hour return and ATR
percentage; the ten-coin group's fraction rising, typical absolute one-hour movement,
return dispersion, and typical relative volume. The level blocks crossed the one-hour
contact stream with 1-hour, 4-hour, 8-hour, and daily source timeframes. Native levels
retained signed and absolute ATR distance, any emitted score, source age, nearest
activity/acceptance distance, and counts inside 0.10, 0.25, and 0.50 ATR. Volume
Profile retained distances to upper/lower high-volume and low-volume nodes, current
and prior point of control, value-area width, node strength or thinness, profile score,
and source age. Cluster inputs retained whether a cluster was present; component,
family, timeframe, dependency-group, source, and exact-pattern counts; cluster width
and component spread; local level density; tight or standard scale; projected status;
cross-timeframe status; and whether different families crossed timeframes. Every native,
Volume Profile, and cluster block had an otherwise identical version delayed by 168
hours as a stale-information control.

Method: separate non-directional FreqAI regression models predicted continuous future
movement, range, volume, pressure change, dwell, and crossings. Current level blocks
were compared with market-state-only and delayed or shuffled versions on the same
rows.

Established large coins: adding BTC and broad-market state was almost neutral on
average. Adding the broad native-level or Volume Profile blocks slightly worsened the
median unseen ranking and error. Cluster features were present and non-constant, but
the fitted models did not use them to improve predictions.

Meme coins: no equivalent broad FreqAI claim has been made. Focused direct tests are
being completed first.

Current meaning: more inputs do not automatically create more knowledge. Broad mixed
feature blocks can hide narrow relationships and add noise. Future FreqAI work must
use small, named blocks after a direct relationship survives controls.

### 7. Generic calculated levels as one undifferentiated category

Calculated objects: rolling highs and lows, prior ranges, moving averages, Bollinger
boundaries, round numbers, Volume Profile levels, and custom structural levels pooled
too broadly.

Established large coins: price often reacts around calculated levels, but a generic
"some level is nearby" representation did not reliably beat market-state, shifted,
stale, shuffled, and artificial-location controls. Most apparently comparable cells
were too different before contact to support a clean claim.

Meme coins: the focused atlas is available, but no broad pooled success has been
claimed.

Current meaning: level family, attribute, source timeframe, cluster composition,
distance, and prior market state must remain explicit. Pooling them obscures rather
than proves a reaction relationship.

## Frozen Generation-2 tests and terminal results

### 8. Causally anchored VWAP zones

Status: the ten-meme evidence batch `g2d_anchored_vwap_meme_full_20260814a` and the
ten-established-coin batch `g2d_anchored_vwap_normal10_full_20260814a` both completed
cleanly. Neither produced an anchored-VWAP response that survived every required
control in repeated validation periods. The anchored-VWAP branch is therefore parked.
Nothing in this subsection is an accepted market edge.

What the indicator means in plain language: anchored VWAP is the average price paid
since a named starting point, with candles carrying more weight when their volume was
higher. The test uses four fixed starting-point families:

- the start of the current UTC day;
- the start of the current UTC week;
- the most recent confirmed local high;
- the most recent confirmed local low.

A local high or low needs three completed one-hour candles on its left and three on
its right. It is not made available to the test until the following decision candle.
The level at a contact candle uses only earlier candles, never the contact candle or
later data. Direct mutation tests confirm this.

Each family creates five one-hour price references: the volume-weighted centre, one
weighted price-dispersion above and below it, and two weighted dispersions above and
below it. The dispersion is the volume-weighted spread of earlier prices around the
anchor average. These settings are fixed; they are not selected from reaction results.
The contact zone extends 0.10 of the coin's prior one-hour ATR on either side, with a
0.05% price floor so extremely quiet candles do not create a zero-width zone.

Crossed levels and clusters are kept explicit. A contact is labelled as either:

- an isolated anchor-family contact, where no zone from another anchor family overlaps;
- a cross-family anchor cluster, where, for example, a daily anchored band and a
  confirmed-swing anchored band occupy the same price area.

The source family, centre or band, approach from below, approach from above, or already
inside/unclear approach are retained. Centre and band results are not pooled blindly.

Timeframes and response windows:

- contacts and all causal calculations use one-hour candles;
- daily and weekly reset points deliberately cross calendar horizons into the one-hour
  contact frame;
- confirmed swing anchors cross the earlier multi-candle structure into the current
  one-hour frame;
- response is measured on the contact candle and over the next 1, 4, and 24 hours;
- matched events are purged separately so one-hour, four-hour, and twenty-four-hour
  future paths are not repeatedly counted as independent evidence. Contact-candle
  measures use the four-hour independence group.

Market behaviour recorded after every contact includes absolute movement in ATR,
range expansion, volume change, the size of pressure change without assuming its
direction, time spent near the zone, and the number of crossings. It does not score
profit or predict up versus down.

Additional inputs used only to make the actual and control events comparable are:

- the coin's prior 1-hour, 4-hour, and 24-hour return;
- prior ATR as a percentage of price, range activity, and volume activity;
- six-hour candle pressure;
- RSI(14), position and width inside Bollinger Bands, MACD histogram relative to ATR,
  and distance from EMA(50);
- Bitcoin's prior 24-hour return;
- top-ten-coin breadth, common absolute movement, and return dispersion;
- age of the anchor and average volume since it began;
- the number of active anchor levels, the number of different anchor families within
  two ATR of price, and how much of a four-ATR price window is covered by anchor zones.

Every actual contact is compared with all of the following rather than with a single
generic random candle:

- an ordinary arithmetic average using exactly the same anchor history;
- an ordinary exponential average using exactly the same anchor history;
- the real anchor delayed by 24 hours for daily anchors or 168 hours for weekly and
  swing anchors;
- copies moved by minus 1, minus 0.5, plus 0.5, and plus 1 ATR, excluding copies that
  accidentally overlap a genuine anchor zone;
- a random eligible location;
- a same-market-state location with no anchor level;
- a no-level location also matched on active-level density and chart coverage.

Technical preflight only: PUMP produced 307,221 raw matched comparisons and 86,105
independent comparisons after the response-window purge. All four anchor families and
all control types appeared, with zero future-source or anchor-open timing violations.
Those counts prove the test surface works; they are not evidence that PUMP or anchored
VWAP reacts better than its controls. An initial long-form implementation was stopped
at about 7.8 GB because it duplicated the same state once per outcome. The corrected
wide representation keeps every outcome once per matched response window and passed
37 research tests. A second smoke was stopped only during an inefficient repeated
summary calculation; its retained comparisons were used to verify the corrected
summary contract, not to claim a result.

Full meme-cohort integrity: all ten pair tasks completed, producing 3,013,126 matched
comparisons before overlapping response paths were removed and 840,468 independent
comparisons afterwards. All four anchor families and every declared control appeared.
There were no future-source, anchor-open, direction, or profit violations. The contact
distance mismatch stayed below the frozen 0.10-ATR limit. The output contains 23,852
cohort summaries; 7,566 had enough coins and episodes and also passed the state-balance
check. Passing that gate only means a comparison was fair enough to inspect. It is not
itself evidence of a reaction.

No named centre or band, isolated level or cross-family cluster, and primary behaviour
measure passed all ten applicable controls in even one validation period. The best
candidate-period had six eligible controls. Therefore no candidate could repeat across
both early and late validation, and none reached the final leave-one-coin-out rule.

The control-by-control pattern explains why. Across the 640 primary validation cells,
the ordinary same-history averages were usually well matched: 632 cells were eligible
against EMA and 616 against SMA. By contrast, only 256 and 176 were eligible against
the minus and plus 0.5-ATR locations, 24 and 64 against the minus and plus 1-ATR
locations. The saved first-pass summary reported only 48 eligible random controls and
eight each for the same-state and same-density no-level controls, plus none against the
168-hour delayed anchor. These failures were a mixture of insufficient episodes and
actual/control market-state imbalance. They are coverage or comparability limits, not
proof of no market effect.

Summary-code review found that the first-pass balance gate judged every control on all
21 state fields even when that control had deliberately been matched on a smaller,
explicit field list. A read-only recomputation used each row's recorded matching list.
Random-control eligibility rose from 48 to 192 primary validation cells and same-state
no-level eligibility rose from eight to 56; the same-density control correctly remained
at eight because it already used the full list. Shift-control eligibility did not
change, and the sparse 168-hour delayed control still had zero eligible cells. No
candidate passed all controls in either validation period under the corrected balance,
so the meme conclusion does not change. Future summaries should nevertheless use the
declared feature list because the original all-field gate understates weaker-control
coverage.

Partial relationships are retained as diagnostics, not promoted findings. Eighteen
centre/band and behaviour combinations differed with the same sign from both ordinary
SMA and EMA in both validation periods and kept that sign when any one coin was
removed. Examples include:

- a weekly minus-one-dispersion cross-family cluster with contact-volume ratio about
  0.20 higher than the ordinary-average controls;
- an isolated weekly centre with following four-hour volume ratio about 0.14 lower;
- an isolated confirmed-swing-low plus-two-dispersion band with contact-range ratio
  about 0.13 higher.

These say volume weighting sometimes differs from ordinary price averaging. They do
not show that the exact anchored-VWAP location matters. Only one comparison had the
same sign against both nearby 0.5-ATR shifted locations in both validations, and it
failed the leave-one-coin-out check. None survived all four shifted locations. None
survived the random, same-state no-level, and same-density no-level ladder together.

The only leave-one-coin-out-stable delayed-anchor diagnostic was the daily centre when
it was part of a cross-family cluster: its contact-volume ratio was about 0.30 lower
than the same daily anchor delayed by 24 hours, with only about 15% of coins showing a
positive difference. It failed the other location and no-level controls, so it cannot
be called an anchored-VWAP reaction.

Plain conclusion for memes: anchored VWAP can produce values that differ from an
ordinary average, but this test did not find a repeatable response tied specifically
to the calculated VWAP price. Neither single anchor levels nor cross-family anchor
clusters cleared the complete control ladder. The branch remains parked unless the
normal-coin comparison supplies a genuinely different, fully controlled result or a
later batch obtains better balanced no-level and delayed-anchor coverage.

Provenance note: the meme process loaded and recorded source hash
`53e6e3e1...440be`. A faster but logically equivalent leave-one-coin-out summary was
added to the file after that process had already started, so the current file hash is
different. The completed outputs remain tied to the recorded launch request and every
pair file passed that request contract; the source change did not enter the running
process.

Full established-coin integrity: all ten pair tasks completed and produced 17,012,375
matched comparisons before overlap removal and 4,523,431 independent comparisons
afterwards. The compact evidence consists of 319,508 coin-and-period summaries,
31,988 cohort summaries, and 312,408 leave-one-coin-out rows. All four anchor families,
all applicable controls, all ten coins, and the one-, four-, and twenty-four-hour
response windows were present. There were no pair failures, future-source violations,
anchor-open timing violations, direction calculations, or profit calculations.

The summary balance defect described above was repaired after the established-coin
source run exited. The saved comparison rows already state exactly which prior-market
fields each control used for matching. Pair and cohort summaries now score balance
only on that declared list. A new summary-only mode verifies the completed source run,
its integrity record, request hash, row count, no-direction/no-profit boundary, and
the hash of the independent evidence table. It then writes a new summary run without
rebuilding or copying any pair evidence. Seven focused G2D tests and all 55 consolidated
market-reaction tests pass.

The corrected saved runs are
`g2d_anchored_vwap_meme_declared_balance_20260814a` and
`g2d_anchored_vwap_normal10_declared_balance_20260814a`. Their row counts exactly
match their original source runs. For memes, fair primary-validation random controls
increased from 48 to 192 cells and same-state/no-level controls from eight to 56. For
established coins, random controls increased from 64 to 344 and same-state/no-level
controls from 24 to 64. SMA, EMA, price-shift, delayed-anchor, and same-density results
did not change because those comparisons already declared the full state list or
failed for genuine coverage or geometry reasons.

The corrected meme conclusion is unchanged. No candidate-period had all ten
applicable controls; the best had seven. The 168-hour delayed-anchor comparison still
had no eligible primary-validation cells, and the same-density/no-level control had
only eight. No candidate survived all four price shifts or the random plus two
no-level controls in both validation periods.

The established-coin result has more usable control coverage but still does not retain
a VWAP level. There were 3,072 fair primary-validation control cells. All ten controls
were usable for sixteen candidate-periods: these were the eight four-hour behaviour
measures at the isolated weekly VWAP centre in each of the two validation periods.
Having all controls available exposed disagreement rather than confirmation. In early
validation, for example, absolute movement and range were lower than random,
same-state/no-level, and same-density/no-level times, while crossings were higher.
In late validation, absolute pressure change was higher than those three controls.
No one behaviour repeated with the same sign across the entire ten-control ladder,
and between two and eight controls also lost their sign when individual coins were
removed, depending on the behaviour measured.

One established-coin partial deserves to remain visible without being promoted. A
weekly plus-two-dispersion band that was part of a cross-family anchor cluster had
lower contact volume than all four nearby shifted locations in both validation
periods. The differences ranged from about -0.05 to -0.42 in early validation and
-0.21 to -0.56 in late validation. However, it was slightly above both ordinary
averages early, changed sign against EMA late, had a positive raw difference from the
random control, and lacked fair random, no-level, and delayed-anchor comparisons for
that exact question. This may describe local geometry around a band, but it does not
show that the anchored-VWAP location is uniquely reactive.

A shared-calendar diagnostic kept both the real and control event inside the same
2025-H2, 2026-Q1, or 2026-Q2 window. Matching was inherited from the wider original
period, so this is diagnostic rather than a fresh same-window experiment. Neither
market group produced an all-control or four-shift result on those shared dates. Meme
weekly-centre contacts in 2025 H2 had lower four-hour absolute movement, contact range,
and following volume than the three no-level controls, but the established group did
not reproduce that pattern and it did not repeat through 2026. The only same-sign
normal-and-meme comparison was higher following four-hour volume than ordinary
averages at the clustered weekly upper band in 2025 H2; nearby shifted locations
behaved more strongly and the result did not repeat in 2026. There is therefore no
simple normal-versus-meme anchored-VWAP relationship.

Two no-write analysis-query problems are retained as technical context, not evidence.
The first query used generic early/late labels instead of the saved cohort-specific
period names and stopped before analysis. The first shared-date print used tuple keys
that JSON cannot represent and stopped only while formatting output. Correct period
labels and ordinary labelled count rows resolved both; no saved evidence was changed.

Plain combined conclusion: volume weighting sometimes differs from ordinary moving
averages, and the isolated weekly centre can look like an acceptance or choppy area in
one period. Those differences are not stable across exact-location, no-level, delayed,
coin-removal, time, and market-group controls. Anchored VWAP is parked as a broad
reaction-zone family. The weekly-centre acceptance idea remains a low-priority queued
candidate, not a frozen descendant, because the complete fair comparison already
showed strong sign conflict.

The eventual retain rule remains strict: one named anchor family and band must show the
same understandable activity or acceptance response beyond every required control in
at least two validation periods, at least five eligible coins, at least fifty
independent comparisons, acceptable state balance, and leave-one-coin-out stability.
Normal coins and meme coins will be reported separately before any rational subgroup
comparison. A one-coin result cannot pass.

### 9. Causal swing-price and repeated-close density zones

Status: the frozen ten-meme coverage-only preflight completed cleanly. The 0.25-ATR
bin construction is now fixed before reaction outcomes are calculated. The fixed-width
normal-coin coverage audit also completed cleanly. Two end-to-end one-coin technical
smokes completed with identical retained events after the purge-speed correction. The
controlled reaction code passes 54 research tests. The full ten-meme evidence run
completed cleanly; the unchanged ten-established-coin comparison is now active with
four workers. No density-zone reaction result has been accepted.

The two proposed indicators mean:

- confirmed-swing price density: places where several fully confirmed local highs or
  lows occurred at similar prices;
- repeated-close density: places where many earlier one-hour candles closed at similar
  prices.

Both use only earlier one-hour candles. A swing requires three completed candles on
each side and becomes usable only on the next decision candle. Current and future
price mutation tests must leave every earlier zone unchanged; this currently passes.
The proposed historical views are the previous 168 hours and previous 720 hours. They
will remain separate rather than being pooled as one vague multi-timeframe feature.

The coverage-only preflight compares fixed bin widths of 0.20, 0.25, and one third of
the current causal one-hour ATR. It uses development data only and records availability,
number of zones, distance from prior price, zone width, support, and how much of the
nearby chart the zones cover. It contains no future reaction, profit, return-after,
or direction measure. Therefore a width cannot be selected because later price happened
to react well. The middle 0.25-ATR construction is the prior preference and will change
only if its coverage is unusably sparse or broad.

PUMP construction check, not reaction evidence: the source contained 8,849 one-hour
candles, with 4,179 development candles eligible for comparison. The causal mutation
and fully confirmed-swing tests passed, the frozen manifest and batch hashes matched,
all twelve expected family/history/width rows were present and finite, and the output
contained no future reaction score. The run took 99.5 seconds on one worker.

At the provisional 0.25-ATR bin width, the 168-hour confirmed-swing zones were
available on 96.1% of eligible PUMP candles and had a zone within two ATR of prior
price on 66.9%. Their median coverage of the local four-ATR price window was 8.6%,
and their median half-width was 0.17 ATR. The 168-hour repeated-close zones were
available on 96.3%, had a zone within two ATR on 64.8%, covered a median 13.6% of the
local window, and had a median half-width of 0.28 ATR. These values are neither so
sparse that the indicators are unusable nor so broad that almost every price becomes
a contact.

The older 720-hour PUMP zones were available on 83.1% of eligible candles but were
usually farther from current price: a confirmed-swing zone was within two ATR on
44.1%, and a repeated-close zone on 41.3%. Their median local coverage was zero
because the retained historical zones were often completely outside the four-ATR
window around current price; this does not mean that no zones existed. The upper-tenth
local coverage was 23.3% for confirmed swings and 56.1% for repeated closes. The
720-hour view must therefore remain a separate older-market-memory test, not be pooled
with the much more local 168-hour view.

The full coverage run contained all ten frozen meme coins and all 120 expected
coin/family/history/width surfaces with no duplicates or missing values. All task,
source-code, manifest, and frozen-batch integrity checks passed, and all 43 current
causal and unit tests passed. The 0.20-ATR choice made zones narrower and the
one-third-ATR choice made them broader across the cohort. The prior 0.25-ATR middle
choice is therefore fixed for the reaction test; no reaction outcome was used to make
that choice.

At fixed 0.25 ATR, the typical coin had a 168-hour confirmed-swing zone available on
almost every eligible candle, a zone within two ATR on 72.5%, median local-chart
coverage of 10.4%, and upper-tenth coverage of 28.2%. The typical 168-hour
repeated-close surface was also available almost continuously, had a zone within two
ATR on 69.2%, median local coverage of 15.6%, and upper-tenth coverage of 51.3%.
Repeated-close zones are therefore wider and sometimes cover much of the nearby chart;
the later same-density and same-coverage controls are mandatory, not optional.

For the 720-hour views, typical local coverage at a random eligible candle was zero
because the retained historical zones often sat outside the local four-ATR window.
When the older zones were local, upper-tenth coverage was 25.3% for confirmed swings
and 55.3% for repeated closes. A zone was within two ATR on 46.1% and 41.3%
respectively. This supports keeping 168-hour local memory and 720-hour older market
memory as separate crossed timeframes. It does not show that either causes a reaction.

Normal-versus-meme construction comparison at the same fixed 0.25 ATR: the 168-hour
surfaces were very similar. For the typical normal coin, a confirmed-swing zone was
within two ATR on 74.0%, versus 72.5% for memes; repeated-close values were 69.9% and
69.2%. Typical local coverage was slightly higher for normal coins: 12.8% versus 10.4%
for confirmed swings and 18.8% versus 15.6% for repeated closes. Median zone widths
were effectively the same, about 0.17 ATR for confirmed swings and 0.28 ATR for
repeated closes. This supports using one common construction rather than tuning a meme
version and a normal version.

The 720-hour normal surfaces were more often near current price than the meme surfaces.
Confirmed-swing zones were within two ATR on 48.2% for the typical normal coin versus
46.1% for memes; repeated-close zones were 47.0% versus 41.3%. Normal older zones also
covered more of the nearby chart when local: upper-tenth coverage was 30.7% versus
25.3% for confirmed swings and 65.9% versus 55.3% for repeated closes. A likely
construction explanation is that every established coin has deep continuous history,
whereas several young meme coins have shorter or less stable old-price memory. This is
only an inference about availability and geometry. It is not evidence that normal coins
react more strongly.

No volume, pressure, movement, dwell, crossing, direction, or profit conclusion can be
drawn from either coverage preflight.

Technical failure retained as non-evidence: the first normal-coin coverage attempt
stopped before calculating any zones because the research script expected the meme
manifest's explicit analysis-end field, while the established normal manifest declares
its dated end through chronological periods. All ten tasks reported that same missing
field. The root fix now accepts either documented manifest form and raises an error if
neither supplies a dated boundary; it does not guess a date. The fixed source-frame
check resolved the normal end to 20 July 2026, loaded 60,137 BTC one-hour candles, and
identified 22,656 development candles. Lint, compilation, and all 43 tests passed. The
replacement run completed all ten coins and all 40 expected fixed-width surfaces with
matching code, manifest, and frozen-batch hashes, no duplicates, no missing numbers,
and every reaction/direction/profit flag false. The failed run remains clearly marked
failed and cannot be used as evidence.

Volume Profile control readiness: nine established-coin one-hour caches validated
immediately. DOGE alone was stale because its underlying OHLCV file had changed. The
approved unchanged-indicator builder rebuilt only DOGE's one-hour core/generic cache;
the new metadata and OHLCV source hash now validate. No indicator definition or setting
was changed.

A bin must meet a fixed minimum count and the upper-quartile occupied-bin count for its
own history. Directly adjacent selected bins are joined into one connected zone;
non-adjacent bins stay separate. At most the three strongest zones per family/history
are retained at a candle. A direct unit example confirms that adjacent price bins merge
while a separated bin remains its own zone.

The reaction test keeps each 168-hour and 720-hour density family separate. Exact-rank
duplicates on the same contact candle are reduced to the nearest contacted zone, but
the source rank and support fraction remain descriptive fields. A contact is labelled
as a single density zone, a cluster of density zones, or a density zone overlapping a
Volume Profile or same-history prior-high/low reference. This prevents a repeated
cluster pattern from being hidden inside a pooled single-level result.

The full control ladder is now explicit:

- current one-hour high-volume nodes, low-volume nodes, and point of control remain
  separate Volume Profile controls because earlier tests show they do different jobs;
- the simple prior high and prior low use the same 168-hour or 720-hour history as the
  density zone being tested;
- copies of the density zone moved one ATR below and one ATR above test whether the
  exact price matters rather than the surrounding market area;
- the confirmed-swing control rotates swing-size residuals by one third inside each
  already-completed history and rebuilds them on mismatched historical pivot prices.
  This preserves causal history, swing counts, and approximate swing size while
  destroying the real pairing between a pivot's price and extremeness. Simply
  permuting raw prices would leave a density histogram unchanged and is therefore not
  used as a false placebo;
- a random eligible location, a same-market-state location with no considered level,
  and a no-level location additionally matched on density-zone count and chart
  coverage complete the ladder.

Additional inputs are used to match comparable pre-contact conditions, not to predict
up or down. They are the coin's prior 1-hour, 4-hour, and 24-hour return; ATR percentage;
recent range, volume, and six-hour candle pressure; RSI(14); Bollinger position and
width; MACD histogram relative to ATR; distance from EMA(50); Bitcoin's previous
24-hour return; top-ten breadth, common absolute movement, and dispersion; the number
of active zones in the selected density surface and across all four surfaces; how many
surfaces sit within two ATR; local chart coverage; and how many Volume Profile or
prior-high/low references sit nearby. These crossed indicators and timeframes are
matching context only. The measured outputs remain absolute movement, range, volume,
absolute pressure change, dwell, and crossings over the contact candle and the next
1, 4, and 24 hours. Profit and directional prediction remain disabled.

Causal-timing audit on 14 August 2026: every one of those matching inputs is available
before the contact candle opens. Local returns, RSI, Bollinger values, MACD, EMA gap,
range, volume, pressure, ATR, and prior close all use the previous completed one-hour
candle or older data. Bitcoin and top-ten market context are moved to the hour after
their source candles close before they are joined to a contact. A repeated-close zone
uses closes strictly before the contact. A swing price becomes usable only after all
right-hand confirmation candles have closed; a swing confirmed exactly at a contact
hour is therefore based on information completed before that hour opens. This audit
found no future candle in the matching context or density construction. It is a data-
timing check, not evidence that the zones cause a reaction.

Technical failure retained as non-evidence: the first reaction smoke stopped before
loading any market data because its frozen-branch guard used the wording
`g2e_causal_swing_and_close_density_zones` instead of the exact existing identifier
`g2e_causal_swing_price_density_zones`. The failed attempt produced only stderr and no
reaction rows. The guard now uses the frozen identifier, runs inside the recorded
failure boundary, and has a direct test. Lint and all 52 tests pass. The replacement
smoke uses a new run ID so the empty failed attempt cannot be confused with evidence.

Second technical failure retained as non-evidence: replacement smoke `...20260814b`
completed PUMP construction and wrote a 13.9-MB raw matched-pair file, then stopped
before the independence purge because the shared helper was called with
`key_columns` rather than its actual `group_columns`, response-key, and horizon-map
arguments. The raw file is not a reaction result and is not promoted. The root fix now
routes the density matches through one tested wrapper using the existing 1-hour,
4-hour, and 24-hour separation map. The rows also carry the required deterministic
level name used by the shared purge sorter. Dependency hashes for construction,
matching, period, event, and summary helpers are now recorded. Lint, compilation, and
all 53 tests pass before the next new-ID smoke.

Clean technical smoke `...20260814c` then completed end to end. PUMP produced 103,606
matched comparisons and 40,309 independent comparisons after the response-window
purge. Both density families, both 168-hour and 720-hour histories, single density
zones, density-only clusters, density-plus-reference clusters, and all eleven controls
were present. The maximum actual/control distance mismatch stayed below 0.10 ATR.
There were no future-source, source-open, response-separation, duplicate, direction,
or profit violations. Its 6,808 cohort rows are deliberately ineligible because one
coin cannot satisfy the multi-coin rule. This proves the complete reaction path works;
it says nothing about whether PUMP reacts.

Performance diagnostic retained as non-evidence: a no-write replay of the technical
raw pairs exceeded its 120-second limit inside the generic greedy overlap purge. The
largest group held only 179 rows, so the problem was thousands of repeated small
sorts, not one dominant event. An indexed-time version first matched the shared
reference exactly on bounded groups. Moving the same stable priority ordering into one
global sort then retained the same 40,309 PUMP comparisons and reduced the measured
purge to about 32 seconds. G2E now uses that equivalent path; a unit test compares its
selected event, order, and separation directly with the shared reference. This changes
runtime only, not contacts, controls, outcomes, or pass criteria.

Optimized end-to-end smoke `...20260814d` independently reproduced exactly 103,606
matched and 40,309 retained comparisons, with the same 7,340 pair-period rows and
6,808 one-coin cohort rows. Its integrity checks also passed. This confirms the faster
path did not change the selected PUMP events.

Before the full cohort run, the G2D balance review was applied to G2E summaries as a
root-cause correction. Each control is now balanced only on the pre-contact fields its
saved `matching_state_features` list declares. Random and same-state controls therefore
cannot fail merely because they intentionally omit density geometry; the
same-density-and-coverage control still declares and is tested on the full list. A
unit test makes an unmatched artificial field extremely imbalanced and confirms it is
excluded while the declared fields remain scored. Lint, compilation, and all 54 tests
pass.

Full meme reaction run `g2e_density_reaction_meme_full_20260814a` completed on 14
August 2026. All ten frozen pairs completed. The 1,008,169 matched comparisons became
388,745 independent comparisons after the one-hour, four-hour, and twenty-four-hour
future-path purge. The run wrote 73,300 pair/period rows, 7,188 cohort rows, and 66,774
leave-one-coin-out rows. Integrity passed: both indicators, both histories, all three
single/cluster scopes, and every applicable control were present; the pair set was
exact; actual/control approach distance differed by less than 0.10 ATR; and there were
no future-source, source-open, response-separation, duplicate, direction, or profit
violations.

The main limitation is fair control balance, not a lack of contacts. Of 7,188 cohort
rows, 6,592 had enough coins and events, but only 1,894 also had acceptable pre-contact
balance. Random-location controls supplied 720 evidence-eligible rows and same-state
no-level controls supplied 680. In contrast, high-volume-node, low-volume-node, and
same-density/no-level controls supplied zero balanced rows; point-of-control supplied
eight; one-ATR upward and downward shifts supplied 218 and 214; the swing shuffle
supplied 54 of its 360 rows; and same-history prior highs and lows supplied zero. The
largest imbalance was usually not RSI, MACD, moving-average distance, volume, pressure,
or broad-market state. It was the amount of the nearby chart covered by density zones,
the selected surface's coverage, and the number of density surfaces close to price.
For prior highs and lows, EMA distance and RSI also remained very different. Therefore
the strict all-control retain rule cannot pass, and the run cannot fairly say either
that density levels work or that they do not.

Balanced market-state diagnostic: thirteen primary four-hour relationships repeated
against both random-location and same-state/no-level controls, kept the same sign in
both validation periods, had a majority of eligible coins on that sign, and remained
stable when each coin was removed. Eight of those thirteen were higher crossing counts;
the others were two higher contact-volume comparisons, one higher contact range, one
higher absolute movement, and one lower following volume. None retained that complete
evidence status against both one-ATR shifted locations. This makes repeated crossing
the recurring partial behaviour, not a general claim that density contact increases
every type of activity.

No four-hour absolute pressure-change relationship survived that balanced two-control
and leave-one-coin-out screen. No width-safe dwell relationship survived either. RSI,
Bollinger, MACD, EMA distance, prior volume, six-hour pressure, and BTC/top-ten state
remain matching inputs here; this run does not show that any one of them caused the
crossing pattern. Direction was not tested.

The clearest example was a cluster of 168-hour repeated-close zones. Its four-hour
crossing count was higher than random locations by about 0.13 crossings in early
validation and 0.30 late, and higher than same-state/no-level times by about 0.19 and
0.23. Seven of ten then all ten coins agreed against random; seven of ten then nine of
ten agreed against same-state/no-level. However, the cluster had about 0.21 and 0.29
fewer crossings than current point-of-control contacts, and its difference from the
one-ATR-lower shifted location was negative early and effectively zero late. The exact
density price therefore was not demonstrated, and point of control remained a simpler
acceptance explanation.

Approach-state diagnostic explains much of that crossing result. When the prior close
was already inside the zone or the approach was unclear, the same 168-hour repeated-
close cluster had about 0.47 and 0.65 more crossings than random locations and about
0.71 and 0.45 more than same-state/no-level times in early and late validation. Almost
every coin agreed. Contacts arriving clearly from above or below were small, mixed, or
reversed across periods and controls. The repeated-crossing motif is therefore evidence
about occupancy in a choppy acceptance area, not evidence that arriving at the level
triggered a new reaction.

The longer view was coherent only against the two balanced market-state controls. For
the same 168-hour repeated-close clusters, twenty-four-hour absolute excursion was
about 0.59 and 0.32 ATR lower than random and 0.66 and 0.60 ATR lower than same-state/no-
level times. Mean range and volume were also lower in both validations, while crossings
were higher. Several Volume Profile and shifted-location comparisons changed sign, and
point of control was often quieter still. This supports a provisional description of
low-movement, lower-activity, repeatedly traversed acceptance conditions. It does not
establish a uniquely localized density reaction.

One 168-hour repeated-close density-plus-reference cluster had higher four-hour dwell
against most controls in both validations, but it is not retained. The weakest increase
was only about 0.008 of the observation window; its early prior-high comparison had only
43 rather than 50 independent events; most geometry controls were imbalanced; and the
saved pair rows do not retain or explicitly balance the contacted zone width. Because
wider zones mechanically raise dwell, this is a width-confounded diagnostic.

Plain meme conclusion: current G2E is a representation-limited acceptance/chop result,
not a repeatable reaction-zone lead. The repaired follow-up question is to trim events
to genuine common support in zone coverage, retain and match contacted width, separate
first arrivals from already-inside occupancy, and compare the repeated-close cluster
directly with point of control and its individual components. That question remains
queued until every frozen Generation-2 branch and the normal-cohort comparison receive
their joint review.

Full established-coin reaction run
`g2e_density_reaction_normal10_full_20260814a` also completed on 14 August 2026.
All ten requested pairs completed. Its 5,893,464 matched comparisons became 2,130,247
independent comparisons after overlapping future windows were removed. The run wrote
100,200 pair/period summaries, 10,022 cohort summaries, and 96,930 leave-one-coin-out
summaries. Integrity passed: both density indicators, 168- and 720-hour histories,
single zones and both cluster scopes, and every applicable control were present. There
were no failed pairs, duplicate retained rows, event-separation errors, future-source
errors, source-open timing errors, direction predictions, or profit calculations. The
largest actual/control approach-distance mismatch remained below 0.10 ATR.

The established-coin result has even more contact coverage than the meme result, but
the same geometry-comparison problem. Of 10,022 cohort rows, 9,710 had enough coins and
episodes, while only 2,604 also had acceptable pre-contact balance. Random locations
supplied 960 balanced rows and same-state/no-level supplied 932. The one-ATR upward and
downward shifts supplied 300 and 320, and the swing-price shuffle supplied 92. Current
high-volume nodes, low-volume nodes, points of control, same-density/no-level times,
and same-history prior highs and lows supplied no balanced cohort rows. This is not a
failure to find events: for example, many of those comparisons had all ten coins and
hundreds of independent pairs. Their density coverage or reference-level geometry was
too different before contact for a fair causal comparison.

A streaming no-write diagnostic checked 433,506 validation comparisons in 468
family/history/scope/control groups without loading the full independent table at
once. For high-volume nodes, low-volume nodes, and point of control, the field with the
largest difference was almost always total density-zone coverage or the selected
surface's own coverage. Their typical largest standardized difference was about 1.40
to 1.62, far above the 0.50 usable limit. The same-density/no-level control was most
often separated by selected-surface coverage, total coverage, or the number of density
surfaces near price; its typical largest difference was about 1.65. The swing shuffle
had the same coverage problem. Prior highs and lows additionally differed strongly in
EMA50 distance and nearby density-surface count, with typical largest differences of
about 2.15 and 1.87. This independently confirms that the proposed repair must create
genuine common support in zone geometry rather than merely loosen the balance limit.

Against the two well-balanced basic controls, 45 primary four-hour relationships kept
one sign in both validation periods and when every coin was omitted in turn. They
comprised twelve higher-crossing patterns, nine dwell patterns, seven contact-range
patterns, seven contact-volume patterns, six absolute-pressure-change patterns, two
absolute-movement patterns, and two following-range patterns. Twenty-two used confirmed
swing-price density and twenty-three used repeated-close density. This is much broader
than the thirteen meme basic-control diagnostics, but it is still only a preliminary
screen: a genuine calculated price location must also survive nearby shifted locations
and the applicable alternative-level controls.

Only one of those 45 relationships retained the same sign against random, same-state/
no-level, one-ATR-lower, and one-ATR-higher locations in both validations with
leave-one-coin-out stability. It was a single 720-hour confirmed-swing-price density
zone and contact-candle range. In early validation its equal-coin median range increase
was about 0.121 against random, 0.018 against same-state/no-level, 0.075 against the
lower shift, and 0.017 against the upper shift. Late values were about 0.140, 0.070,
0.042, and 0.023. All ten coins contributed, but the weakest comparisons had only six
of ten coins on the positive side. The Volume Profile, prior-extreme, density-matched,
and shuffled-swing controls were not balanced; their raw signs were also mixed. The
candidate therefore remains incomplete rather than retained.

A shared-calendar diagnostic further weakens a simple normal-coin interpretation.
Both actual and control events were restricted to the same 2025-H2, 2026-Q1, and
2026-Q2 windows in the normal and meme files. Matching was inherited from the wider
original periods, so this is diagnostic rather than a new confirmatory test. The
normal 720-hour swing-zone range difference was negative against random and the lower
shift in parts of 2025 H2, positive against the basic, shifted, and shuffled controls
in 2026 Q1, then negative again against the lower shift and swing shuffle in 2026 Q2.
The meme file showed comparable sign changes, including shifted or shuffled negatives
in 2025 H2 and 2026 Q2 and an effectively zero same-state/no-level difference in 2026
Q1. The full-period normal result therefore hides calendar variation. If a relationship
exists, it is more plausibly conditional on a market regime than universal to normal
coins, and it needs a separately frozen same-window rematch before that claim can be
made.

The normal crossing result also reproduces the meme occupancy explanation. All twelve
combinations of density family, history, and single/cluster scope had more four-hour
crossings than both random and same-state/no-level controls in both validations when
the prior close was already inside the zone or the approach was unclear. Every one of
those twelve survived removal of each coin; the weakest median increase was about 0.25
crossings and nearly all had every coin on the positive side. Only two of twelve
repeated for a clear approach from above and two of twelve for a clear approach from
below, with much smaller effects. Starting inside a zone naturally makes repeated
traversal easier to record. This is useful evidence of an occupied acceptance area,
but it does not show that arriving at the calculated level triggered a fresh reaction.

Operational pause retained as technical context, not market evidence: running the
normal G2D and G2E four-worker waves together reduced available RAM below 4 GB. Only
the exact G2E controller and its four owned workers were stopped after BTC, ETH, BNB,
and SOL had sealed. Those files were preserved. After G2D sealed its second group,
G2E resumed under the same run ID and request contract, validated and reused the four
files, and completed the remaining six. No saved pair was rebuilt or interpreted
partially, stderr remained empty, and final integrity passed.

Two no-write result probes also remain recorded as technical non-evidence. The first
leave-one-out filter repeatedly scanned the full omission table and was terminated;
the replacement grouped lookup returned the same required fields efficiently. A later
summary expression initially used `cov` as attribute syntax, colliding with pandas'
covariance method; explicit column indexing corrected it. Neither issue changed the
research code or any saved result.

Current result class: representation failure with a provisional acceptance/traversal
diagnostic in both market groups. The normal cohort adds one weak exact-location range
candidate, but its shared-date signs change and its geometry controls remain unfair.
This is neither a repeatable reaction-zone lead nor an unstable/null rejection of the
two density-indicator ideas.

### 10. Pre-contact market regime changing a level reaction

Status: the direct meme-cohort interaction run completed with valid files and is parked
for insufficient repeatable coverage. No regime interaction was accepted. This is not
evidence that market state has no effect; it means this event sample could not test the
declared interactions across enough coins, controls, and chronological periods at once.

The reused level surfaces are kept separate and clearly named: the eight-hour round
number when it is or is not part of an independent cluster; four-hour high-volume
nodes; one-hour wide low-volume nodes; and the one-hour Volume Profile point of
control. Their existing controls remain separate: matched no-level times, same-level
near misses, isolated current levels, causally shuffled locations, the 168-hour stale
location where available, and symmetric ATR-shifted locations. The test starts from
19,735 already independent event pairs, so repeated rows from one long market move do
not masquerade as separate evidence.

The added pre-contact inputs, in plain terms, are:

- realized volatility: how variable the previous 24 one-hour returns were;
- relative volume: the last completed candle's volume compared with the earlier
  24-hour median;
- ADX: how strongly the recent path behaved like a trend, without saying whether the
  trend pointed up or down;
- absolute EMA20 and SMA50 slopes: how steeply the averages were moving, again
  ignoring up versus down;
- BTC's absolute 24-hour move and ATR percentage: whether Bitcoin itself was moving
  unusually far or was unusually volatile;
- top-ten breadth alignment: whether the cohort mostly moved together rather than in
  mixed directions;
- top-ten return dispersion: how differently the coins moved from each other;
- top-ten common volume: whether volume was broadly raised across the cohort.

Each input is first checked alone. Three predeclared combinations are also checked:

- quiet versus expanding requires realized volatility and relative volume to agree;
- range-like versus directional-trend requires at least two of ADX, EMA slope, and SMA
  slope to agree;
- calm versus crypto-wide stress requires at least three of BTC movement, BTC
  volatility, breadth alignment, coin dispersion, and common volume to agree.

"Low" and "high" are causal: the threshold at a candle is calculated only from the
preceding 720 hours, with at least 168 earlier hours required. Current and future data
cannot move an earlier candle between regimes. An event pair is used for a regime lens
only when the genuine level contact and its control both fall in the same low or high
state. This avoids comparing a level in a panic with a no-level control from a quiet
market.

The reported interaction is a difference of differences. First calculate how much the
genuine level differs from its matched control in the low state. Do the same in the
high state. Then subtract low from high. For example, if a level adds 0.10 to relative
volume in quiet conditions but 0.40 in expanding conditions, the regime interaction is
+0.30. This still does not prove direction or profit. It says the level-specific
activity difference was larger in the expanding state. Movement magnitude, range,
volume, dwell, crossings, and absolute pressure change at 1, 4, and 24 hours remain
direction-neutral outputs.

Retain rule: both state sides need enough independent evidence, at least five supported
coins and fifty independent pairs per state at cohort level, acceptable continuous
state balance, repetition in at least two validation periods, applicable control
survival, and leave-one-coin-out stability. Sparse combinations are insufficient, not
negative. A state that raises activity equally at the level and at the control is not a
level interaction.

Technical failure retained as non-evidence: the first launch stopped before reading
outcomes because the older G2AB integrity schema has explicit audit fields but no
blanket `passed` field. The corrected validator now checks zero pair failures, valid
before/after event counts, the frozen 0.10-ATR matching caliper, positive event
separation, the 48-hour boundary embargo, populated periods and routes, and the
direction/profit boundaries. It does not bypass source integrity. Lint, compilation,
and all 47 current tests pass.

The replacement run completed from all 19,735 independent meme event pairs. It kept
all ten coins, all four routed level families, and thirteen distinct control labels.
There were no event-time lookup errors, no missing requested outcomes, and no use of
direction or profit. The number of event pairs where the real contact and control
agreed on a usable low or high state varied substantially by input: 7,190 for top-ten
breadth alignment; 5,977 for realized volatility; 5,875 for EMA slope; 5,769 for ADX;
5,740 for relative volume; 5,412 for common cohort volume; 5,154 for BTC volatility;
5,111 for BTC absolute movement; 5,077 for SMA slope; and 5,949 for return dispersion.
The combined states were much thinner: 4,866 for range-like versus directional-trend,
3,695 for calm versus crypto-wide stress, and only 798 for quiet versus expanding.
This is a useful negative coverage result: combining several sensible inputs can remove
most otherwise usable episodes even before any reaction comparison is made.

The run produced 62,768 outcome rows, representing 9,187 distinct combinations of one
coin, period, level route, control, and regime lens. Only 66 of those distinct
combinations had at least five low-state and five high-state observations for that
individual coin. After that conservative per-coin filter, no cohort comparison reached
the declared minimum of five coins and fifty observations on each state side. The best
supported remaining comparison had six coins, 45 low-state pairs, and 43 high-state
pairs. Consequently all 254 cohort summaries correctly remained ineligible.

A read-only sensitivity check tested whether the per-coin five-per-state rule itself
had hidden a valid cohort. Keeping coins with at least three observations per state
revealed one development-period comparison: isolated four-hour high-volume nodes
against matched no-level times, divided by top-ten breadth alignment. It contained all
ten meme coins and 62 low-state versus 66 high-state observations, with acceptable
continuous-state balance. Allowing even one observation per coin revealed the same
single comparison and no additional one. This was a diagnostic check only; it changed
no saved run or rule.

That lone comparison does not rescue the branch. It exists only in the meme development
period, not in two chronological validation periods, and it has not survived every
applicable no-level, near-miss, shifted, and shuffled control. Lowering the filter would
therefore create one exploratory lead but still could not satisfy the frozen retain
rule. The conservative completed output is left unchanged, and G2F is parked rather
than labelled negative or promoted. A later batch may revisit the specific
high-volume-node and breadth idea only after the full Generation-2 review, with more
independent dates or a predeclared less-fragmented design.

Technical diagnostic failure retained as non-evidence: the first no-write threshold
sensitivity probe stopped during Python module loading, before reading or comparing
results. Registering the temporary module correctly resolved that diagnostic-only
issue, and the same probe then completed. It did not change the research script or any
saved evidence file.

## Generation-2 joint review and frozen Generation-3 routes

Generation 2 is complete. Every one of its six frozen questions reached a terminal
state before a descendant was selected. No interim result changed the batch, and no
direction or profit result was used to choose a branch.

The six joint classifications are:

1. **Eight-hour round-number clusters:** reject the broad immediate-activity claim.
   Revise one narrower meme question about larger movement during the hour after
   contact, because that repeated in both meme validation periods but did not include
   stable contact-candle activity or the complete control ladder.
2. **Volume Profile roles:** keep the one-hour wide thin-LVN activity/transit result as
   the strongest current lead. Actual entry beat a closely matched near miss across
   both market groups and both validation periods, but the sample is cluster-conditioned
   and still lacks fair component, stale, and no-level attribution. Park the broad
   four-hour HVN-strength and one-hour POC-evidence claims.
3. **Mirrored rolling-extreme plus Bollinger clusters:** park the broad cluster claim.
   Its striking component separation was confined to development or lacked validation
   coverage, and ordinary-cluster comparisons changed sign by period.
4. **Anchored VWAP:** park the broad family. The corrected established-coin comparison
   supplied a complete fair control ladder at the weekly centre, but the controls and
   periods disagreed. Volume weighting differing from SMA or EMA is not exact-location
   evidence.
5. **Swing-price and repeated-close density zones:** classify as a representation-
   limited acceptance/chop result. Repeated crossings mostly came from price already
   residing inside broad zones. No exact density location survived the complete VP,
   structure, shifted, shuffled, and no-level ladder.
6. **Pre-contact regime modulation:** park for insufficient interaction coverage. The
   one development-only HVN and top-ten-breadth diagnostic remains queued, but it did
   not repeat in validation and must not be rescued by lowering the rule after seeing
   it.

The next generation is branch layer three of the user-authorized maximum five. The
following eight routes are frozen as one breadth-first batch. Every route must finish,
be honestly deferred for data, or be parked before any Generation-4 descendant is
selected. A result may add an idea to the queue but cannot launch it inside this batch.

### G3A. Attribute the thin-LVN activity/transit lead

Important correction: the original `20260814a` attribution runs described below
counted separately named peer lines as independent even when those lines came from the
same dependency group. Their numerical conclusions are superseded by the corrected
`20260814b` runs. The original account remains here as an audit trail; use the
correction subsection immediately before G3B for the current result.

Trader question: when price actually enters a causally available, unusually thin
one-hour LVN, does that LVN contribute extra participation and traversal beyond the
other levels that usually cluster around it?

Keep single LVNs and LVN-containing clusters separate. Compare actual entry with the
existing same-LVN near miss, the same cluster with the LVN component removed where
common support exists, an isolated thin LVN where available, a matched cluster without
an LVN, a 168-hour stale profile, symmetric shifted zones, and a same-state/no-level
location matched on total level density and chart coverage. Measure contact volume,
four-hour dwell and crossings, and forty-eight-hour volume. Retain only a repeated
multi-coin effect that survives component attribution, not merely another near-miss
comparison. Park if fair no-level or component support remains structurally absent.

#### G3A progress log: component-geometry preflight

Baseline: the earlier one-hour Volume Profile test ranked each causal `lvn_above` or
`lvn_below` value by its recorded thinness score inside the same coin, period, approach
side and wide half-ATR zone. The upper third was called unusually thin. It showed more
contact volume and later activity than the lower third or a shuffled thinness label.
The location test also found more activity after actual entry than after approaching
the same live LVN without entering it. Neither result had separated the LVN from all
the other levels around it.

Frozen question for this preflight: when an unusually thin one-hour LVN is removed
from the local geometry, is there still a connected cluster of independently derived
levels on the same candle? This stage measured coverage only. It did not inspect a
price direction, profit, or choose a favourable reaction outcome.

Inputs inspected were the one-hour LVN and its recorded thinness score; a wide zone
whose half-width is the larger of one-half of the prior one-hour ATR or 0.05% of the
level price; the candle high and low; and causally available independent levels from
one-hour, four-hour, eight-hour and daily sources. The independent companion families
in this frozen surface were rolling prior highs and lows, Bollinger upper and lower
bands, and round-number levels. Other Volume Profile values were excluded as
independent companions because they come from the same profile calculation as the LVN.
The most common companion family was a rolling prior high or low, followed by a
Bollinger boundary and then a round number. More than 92% of validation contacts in
both market groups had at least one four-hour, eight-hour or daily companion; a daily
companion was present in roughly 27-36%, depending on the group and period.

Established-coin coverage was 2,760 unusually thin LVN contacts in early validation
and 2,265 in late validation. After removing the LVN, independently calculated peer
levels still formed a connected cluster in 2,615 early contacts (94.75%) and 2,140
late contacts (94.48%). The contact candle directly touched at least two companion
levels in 2,505 early contacts (90.76%) and 2,062 late contacts (91.04%). Only 24
contacts in each period were isolated from every independent companion. Only 39 early
and 31 late contacts entered the LVN while touching none of the companions in its
otherwise overlapping cluster.

Meme-coin coverage was 546 unusually thin LVN contacts in early validation and 610
in late validation. The non-LVN peer cluster survived in 522 early contacts (95.60%)
and 582 late contacts (95.41%). The candle directly touched at least two companion
levels in 497 early contacts (91.03%) and 558 late contacts (91.48%). Only two early
and six late contacts were isolated. Only six early and twelve late contacts entered
the LVN without also touching a companion level.

Plain meaning: the current half-ATR LVN zone is almost never a clean single-level
experiment. Removing the LVN usually leaves a real cluster, and the same candle usually
touches several members of that cluster. This does not prove that the LVN adds nothing,
but it makes the broader cluster a strong alternative cause. The genuinely isolated
and LVN-only samples are too small to support a ten-coin claim by themselves. The next
smallest fair outcome test is therefore high-thinness versus low-thinness LVN contact
after matching both market state and the surviving peer-cluster geometry. If the
thinness effect disappears, park the exact-LVN claim. If it survives, continue to the
near-miss, stale, shifted, no-level and cluster-without-LVN controls.

Source-integrity note: the normal-coin cache validator correctly stopped when it saw
that source OHLCV had been extended after the frozen Generation-0 evidence. No stale
cache was silently accepted. A semantic prefix audit then compared the present data
with the frozen evidence at every relevant historical event. Across 33,311 normal-coin
LVN events, prior ATR, contact range, contact volume, pressure change, zone width and
pre-contact distance matched exactly. Across 455,508 selected-level event rows from
all four source timeframes, level price and source-availability time also matched
exactly; all indicator hashes and manifest hashes matched. Only one of forty cache-file
hashes had changed, the DOGE one-hour cache, and its entire retained selected-level
event surface still matched exactly. This establishes an unchanged historical prefix
for the event times used here; it is not permission to mix newly accumulated future
rows into validation.

Technical corrections kept out of the evidence: an initial join matched an LVN event
to a cluster only when both six-candle episode definitions began on the same exact
candle. It falsely labelled continuing clusters as absent and was discarded. A first
source resolver also rejected the older normal atlas because it expected the newer
two-zone metadata contract, and a datetime join initially encountered nanosecond versus
microsecond storage resolution. Both were query-construction issues; neither changed
saved evidence or a market conclusion. The replacement reconstructs the component
geometry directly on each LVN candle.

The missing projected wide-cluster atlas for the frozen meme cohort was then built as
`g3a_cluster_atlas_meme_wide_20260814a` with four workers. It completed all ten pairs,
created 1,734 control-bearing event rows and 47,334 summary rows, used 10.82 MB on the
large-data drive, passed its integrity audit, found no future-availability violations,
and kept direction and profit disabled. Its wide-cluster controls are available for
the next direct attribution comparison.

#### G3A direct result: thin LVN after removing the cluster explanation

Question actually tested: among one-hour LVN contacts where the LVN was not acting
alone, did an unusually thin LVN still mark different market behaviour from a less
thin LVN after the surrounding cluster and the market state were made similar? This
was a direct relationship test, not a profit test, direction forecast, FreqAI model,
or search for a favourable time horizon.

The contacted level was the causally available one-hour Volume Profile `lvn_above` or
`lvn_below`. Thinness was ranked separately within the same coin, chronological
period, LVN side and approach side. The top third was compared with the bottom third.
Every retained event had two safeguards against falsely crediting the LVN for a broad
cluster: after deleting the LVN, at least two independently calculated peer levels
still formed a connected price cluster, and the contact candle directly touched at
least two of those peers.

The peer levels were reconstructed from one-hour, four-hour, eight-hour and daily
causal data. In this frozen selected-level surface they were rolling prior highs and
lows, Bollinger upper and lower boundaries, and round-number levels. Values from the
same Volume Profile calculation were not counted as independent peers. The match also
made the following pre-contact inputs similar: local 24-hour return; local ATR as a
percentage of price; recent range, volume and six-hour pressure; source-timeframe ATR
and volume; BTC 24-hour return; top-ten market breadth; total selected-level density;
Volume Profile value-area width and persistence; number of peer levels; number touched
on the candle; number from a higher timeframe; and number from the daily timeframe.

Bollinger boundaries therefore took two roles in this test: they could be a literal
member of the surrounding level cluster, and their presence/count was preserved in
the geometry comparison. RSI, MACD, EMA and SMA state values were not added to this
primary G3A match. They remain valid later sensitivity inputs, but adding them here
would have changed the already frozen low-dimensional question. Moving-average levels
receive their own explicit test in G3F. Pressure was used as a prior-state matching
input here; post-contact pressure was not one of G3A's four frozen outcomes.

The four outcomes were contact-candle volume relative to the preceding median volume,
the fraction of the next four candles spent in the zone, the number of crossings over
four hours, and mean volume over the following 48 hours. Contact, dwell and crossings
used a six-hour independence gap. The 48-hour volume outcome used a 48-hour gap so one
future path could not be counted repeatedly. A deterministic shuffle of the thinness
labels was run on the same surface as a placebo. Results were calculated per coin and
period, then combined with equal coin weight. Each coin was also omitted in turn.

The completed meme run was
`g3a_lvn_cluster_attribution_meme_full_20260814a`. It produced 2,204 matched event
pairs before the outcome-specific overlap removal and 3,507 independent
outcome-comparisons afterwards. The established-coin run was
`g3a_lvn_cluster_attribution_normal10_full_20260814a`. It produced 15,895 matched
event pairs before that removal and 22,366 independent outcome-comparisons afterwards.
Both runs completed all ten coins, passed their source, timing, overlap, direction and
profit-boundary audits, and had no pair-task failures. The focused regression suite
also passed all 58 tests.

Established large coins did not retain the earlier broad contact-volume claim. In
early validation, the median of the ten per-coin mean differences was about +0.236
trailing-volume units and 8 of 10 coins were positive. After subtracting the shuffled
label result, the median advantage was about +0.360 and 8 of 10 coins were positive.
In late validation, however, the actual difference fell to about +0.015 with only 5
of 10 positive, while the actual-minus-shuffle result reversed to about -0.159 with
only 3 of 10 positive. Leaving out any one coin did not repair that late-period
failure. A normal-coin claim that thin LVNs themselves cause extra contact volume is
therefore parked.

Established coins did show a smaller repeated path-shape clue. Compared with matched
less-thin LVNs in the same kind of peer cluster, the unusually thin LVNs spent about
0.023 less of the next four-candle interval inside the zone in early validation and
0.020 less in late validation. Seven of ten and six of ten coins respectively had the
expected lower dwell. The improvement over shuffled labels was about 0.032 early and
0.009 late, with eight and six coins agreeing. Every leave-one-coin-out version kept
the same sign. At the same time, crossings were lower, not higher, by about 0.055 and
0.049. In plain terms, established coins tended to leave the thin-LVN cluster slightly
more cleanly and revisit it less, rather than oscillate through it repeatedly. This
does not reveal whether price continued through the LVN or rejected back from it, so
it is a small direction-neutral departure lead, not proof of directional transit.

Meme coins behaved differently. Their contact-volume result survived this first
component-attribution comparison. In early validation there were 93 independent
actual comparisons across ten coins. The median of the per-coin mean high-minus-low
thinness differences was about +0.360 trailing-volume units, positive in 6 of 10
coins. The actual-minus-shuffled-label advantage was about +0.514, also positive in 6
of 10. In late validation there were 124 independent actual comparisons; the values
were about +0.567 and +0.680, positive in 7 of 10 coins for both. Omitting any one coin
left both the actual difference and the advantage over shuffled labels positive in
both validation periods. The result is therefore not caused by DOGE, PUMP, or another
single meme coin.

That meme lead has an important weakness. Each meme coin contributed only about 6-18
independent actual pairs in a validation period. Pooled state balance passed, but many
individual coin-period balance scores were weak because their samples were small. In
early validation, only 4 of 10 coins had a positive median event difference even
though 6 of 10 had a positive mean; late validation had 8 of 10 positive medians. This
means the early meme result is influenced by occasional large volume bursts and does
not yet describe the typical contact reliably. It is a legitimate conditional lead
for the remaining controls, but it is not a retained trading rule.

The other meme outcomes did not form one repeated story. Four-hour dwell changed from
slightly more dwell at thin LVNs early to clearly less dwell late. Crossings were
lower, especially late, rather than showing more two-way traffic. Forty-eight-hour
volume was almost neutral early and moderately positive late. Established-coin
48-hour volume was neutral early and negative late. None of these results justifies a
general claim that a thin LVN causes prolonged volume or repeated traversal.

Normal-versus-meme interpretation: after making the surrounding cluster comparable,
established coins mainly offered a small, repeated cleaner-departure clue; meme coins
mainly offered a larger but more burst-driven contact-volume clue. Both are conditional
on a surviving multi-timeframe cluster. The data still cannot tell us what an isolated
thin LVN does because isolated and LVN-only contacts were structurally too rare. It
would be wrong to describe either result as proof that the exact LVN price caused the
whole reaction.

Technical issue and repair: the first reproducible smoke run asked the meme event
atlas for a daily event file. The frozen meme atlas intentionally contains event files
at one hour, four hours and eight hours, while its daily output is a causal peer-level
cache rather than an event surface. The runner now treats that declared daily case as
a pre-outcome cache snapshot, verifies that it predates the G3 freeze, checks its
OHLCV and indicator hashes, and freezes its exact cache hash in the run request. A
regression test now prevents the code from asking for a nonexistent daily meme event
file. No failed query contributed market evidence.

G3A status after its smallest direct test: the broad all-market claim is parked, but
two narrower leads remain queued for the later G3A control batch: meme contact-volume
bursts and established-coin cleaner departure. They must still face the applicable
stale-profile, shifted-zone, matched-no-level and cluster-without-LVN comparisons.
They are not allowed to spawn a Generation-4 experiment until the other frozen G3
routes have received their own primary tests and the whole batch is reviewed.

#### G3A correction: independent means independent calculation mechanisms

The error: the first implementation required two contacted peer lines, but two names
from one calculation family could satisfy that rule. For example, several moving
average or Bollinger-derived lines could look like several independent confirmations
even though they shared the `price_average_family` dependency group. This inflated the
number of supposedly independent clusters and contaminated the parent evidence used
to select the one-minute replay.

The repair: peer intervals are still joined by transitive price overlap, but the
corrected code now counts unique dependency groups inside each connected contacted
component. A qualifying parent requires at least two connected, contacted peer
dependency groups after the LVN's own Volume Profile group is removed. It records both
line counts and dependency-group counts so this distinction cannot be hidden again.
The focused research suite passed after the repair.

The corrected established-coin run was
`g3a_lvn_cluster_attribution_normal10_full_20260814b`. It retained 14,077 matched
comparisons before outcome-specific independence filtering and 20,623 independent
outcome comparisons afterwards. For contact volume, unusually thin LVNs exceeded
matched less-thin LVNs by about 0.245 trailing-volume units in development, 0.191 in
early validation, and 0.091 in late validation. Nine, eight, and seven of ten coins
respectively had a positive difference. The improvement over the shuffled-thinness
placebo was about 0.216, 0.260, and only 0.021, with nine, nine, and five of ten coins
positive. Thus the normal result remains positive in raw terms but becomes very small
and coin-inconsistent beyond the placebo late in time. It is a lead, not a retained
broad effect.

The corrected meme run was
`g3a_lvn_cluster_attribution_meme_full_20260814b`. It retained 1,883 matched
comparisons and 3,277 independent outcome comparisons. Contact-volume differences
were about 0.186 in development, 0.379 in early validation, and 0.450 in late
validation. Six of ten, seven of nine, and six of ten supported the positive sign.
The corresponding improvement over shuffled thinness was -0.080, +0.348, and +0.474,
supported by four of ten, five of nine, and six of ten. This means the meme effect is
encouraging in later periods but was not present beyond placebo in development. It
cannot yet be called a stable all-period relationship.

The earlier established-coin cleaner-departure story also weakened after dependency
groups were counted correctly. Early dwell was slightly lower, late dwell slightly
higher, and neither comparison showed broad coin agreement beyond placebo. Crossings
were close to neutral early and lower late, but the placebo-adjusted signs were not
consistent enough to retain one clean path-shape claim. Forty-eight-hour volume was
small early and negative late for established coins. Meme dwell, crossings, and
forty-eight-hour volume also did not form one stable all-period account.

Current G3A meaning: a thin LVN inside a genuinely mixed cluster still has a plausible
contact-volume association, especially in later meme data, but the evidence is
conditional and period-dependent. The corrected result justifies the bounded G3G
microscope because it was selected from direction-neutral activity, not because G3A
proved a directional edge.

Corrected evidence folders:

- `generation3_branches/g3a_thin_lvn_attribution/g3a_lvn_cluster_attribution_normal10_full_20260814b/`;
- `generation3_branches/g3a_thin_lvn_attribution/g3a_lvn_cluster_attribution_meme_full_20260814b/`;
- bulky matched and independent rows under
  `D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation3_branches\g3a_thin_lvn_attribution\`.

### G3B. Test higher-timeframe room and obstacles around a thin LVN

Trader question: does a one-hour thin-LVN contact produce more activity or faster
traversal when there is open price room beyond it, and less when a four-hour or daily
acceptance/activity level sits close ahead?

Freeze the approach side, nearest higher-timeframe level identity, distance in prior
ATR, whether that level is an activity or acceptance reference, and whether several
independent levels conflict or converge. Compare the same current LVN family under
similar market state but different pre-contact room, shuffle the higher-timeframe
identity, and remove the obstacle block. Measure approach-relative through/away
magnitude, volume, dwell, crossings, and time to leave the zone without predicting
up versus down. Retain only if the geometry changes the LVN-specific response more
than it changes matched controls in repeated periods.

#### G3B progress log: outcome-blind room and obstacle coverage

The first G3B step deliberately loaded no reaction outcomes. It asked only whether
unusually thin one-hour LVN contacts could be divided into a fair approach-relative
obstacle group and an open-room group before seeing what price, volume or pressure did
afterwards. This prevents a distance boundary from being moved to favour a known
reaction.

The target was the upper thinness third of the causal one-hour `lvn_above` and
`lvn_below` events. Already-inside or unclear approaches were excluded. For an approach
from below, "ahead" means above the LVN; for an approach from above, it means below the
LVN. This is a relative path description and not a prediction that price will continue
in that direction.

Every causally available selected level from four-hour, eight-hour and daily data was
then inspected on the ahead side. These included rolling prior highs and lows,
Bollinger boundaries, round numbers, higher-timeframe LVNs and HVNs, POC and prior POC.
The record stores the nearest level's exact family, name, timeframe and role. Rolling
extremes, Bollinger boundaries, round numbers and LVNs are labelled activity/transit
references; HVNs, POC and prior POC are labelled acceptance/stickiness references. It
also stores how many close levels came from 4h, 8h and 1d, how many independent
calculation groups converged, and whether activity and acceptance roles were both
present.

Distance was measured in prior one-hour ATR on the approach-relative side. The gap is
the empty price space between the outer edge of the wide LVN zone and the near edge of
the higher-timeframe level's equally causal wide zone. A gap at or below zero means
the zones overlap. A positive gap no greater than 0.50 ATR is a close obstacle. A gap
from 0.50 to 1.50 ATR is intermediate room. A gap above 1.50 ATR, or no valid level
ahead, is open room. The primary later contrast was frozen as overlap/close obstacle
versus open room. Before outcomes, each side had to contain at least 50 events from at
least five coins in both validation periods.

The meme preflight
`g3b_room_obstacle_preflight_meme_full_20260814a` inspected 1,360 unusually thin LVN
events across the frozen ten memes. Early validation contained 292 close-obstacle
events from all ten coins but only 19 open-room events from nine coins. Late validation
contained 346 close-obstacle events from all ten coins but only 20 open-room events
from seven. The open-room side therefore failed the predeclared 50-event minimum in
both periods. No G3B meme reaction outcome will be read under this representation.
This is insufficient support, not evidence that room is irrelevant to meme coins.

The established-coin preflight
`g3b_room_obstacle_preflight_normal10_full_20260814a` inspected 7,069 unusually thin
LVN events. Early validation contained 1,535 close-obstacle events and 73 open-room
events, all ten coins represented. Late validation contained 1,225 and 88 respectively,
again with all ten coins. Both sides passed the frozen raw coverage gate in both
periods, so the normal-coin direct comparison may proceed. The later matching and
future-path overlap removal may still reduce this support; if it falls below the same
minimum, the outcome comparison must be marked insufficient rather than rescued by
loosening the boundary.

The coverage itself reveals an important level relationship. The nearest ahead
higher-timeframe zone overlapped the wide one-hour LVN zone in 1,445 of 1,678 early
normal validation events and 1,171 of 1,391 late events. For memes the figures were
281 of 330 and 331 of 381. The nearest reference was most often 4h and most often an
activity/transit level. In close-obstacle meme events, roughly 63-68% contained at
least two independent calculation groups; the normal figure was roughly 62-64%.
Around 10-17% also contained both activity and acceptance roles. Thus "obstacle" here
usually means a genuine convergence of several causal zones, not a single clean line.

Both preflights passed all source-prefix, timestamp, cache, direction and profit
boundaries. Their saved geometry contains no contact-volume, post-contact pressure,
through/away excursion, dwell, crossing or time-to-leave result columns. The expanded
focused regression file passed all 64 tests. The next smallest test is normal coins
only: match obstacle and open-room events on the same LVN family, approach side,
thinness, trend, volatility, prior volume and pressure, broad-market state and total
level density, then compare the already frozen direction-neutral reaction outcomes.

Common-support refinement before outcomes: the raw normal counts above did not yet
hold the nearest higher-timeframe identity fixed. When the comparison also required
the same nearest timeframe and activity/acceptance role, the mathematical maximum was
57 early and 63 late obstacle/open-room pairs before any state matching or future-path
overlap removal. When identity was kept in its full declared form--nearest timeframe,
calculation family, named level and role, together with the target LVN side and approach
side--the maximum fell to 40 early and 36 late. Nine coins had at least one possible
pair early and all ten did late, but only three coins in either period could supply at
least five. State matching and independence could only reduce those numbers.

G3B is therefore deferred for data under its frozen question for both market groups.
No reaction outcomes were opened. The normal cohort passed a broad room-versus-obstacle
count but failed the fair same-identity comparison; the meme cohort failed even the
broad count. Weakening identity to "some higher-timeframe thing" would change the
question and could confuse a Bollinger boundary, POC, round number and prior high as
equivalent obstacles. That broader relationship may be queued for the joint batch
review, as may a continuous-distance model, but neither is allowed to replace this
failed common-support check inside G3B. The useful structural finding remains that
open room beyond a thin one-hour LVN is uncommon because higher-timeframe zones usually
overlap it.

### G3C. Focus the meme round-number question on post-contact movement

Trader question: for the frozen ten meme coins, does an eight-hour round number that
is genuinely part of an independent cluster precede larger absolute movement during
the next hour, even when it does not raise contact-candle range or volume?

Compare with a same-density cluster containing no round number, an isolated round
number, a 168-hour stale round level, symmetric ATR shifts, causal location shuffles,
and state/density/width-matched no-level times. Keep approach from above, approach from
below, and already-inside contacts explicit. The primary outcome is one-hour absolute
movement; four-hour volume is secondary and contact activity is a contrary-result
check. Retain only if the movement effect repeats across validation periods and a
supported multi-coin scope without being an already-active-market effect.

#### G3C result: repeated movement association, failed exact-location attribution

No new outcome run was needed for G3C. The exact cluster-without-round comparison was
already saved by `g2c_mirrored_meme_full_20260814a`, while the separate current-level,
isolated-level, shifted-location, shuffled-location, near-miss and no-level ladder was
already saved by `g2ab_missingcontrols_meme_full_20260814e`. Reusing those audited
event pairs avoids changing the event definition or counting the same evidence as a
new experiment. Both sources keep approach state explicit and match prior return,
volatility, range, volume, six-hour pressure, RSI, Bollinger position and width, MACD,
EMA distance, BTC movement, top-ten breadth and dispersion, total level density and
cluster width. Neither predicts direction or optimizes profit.

The narrow association reproduced. An actual cluster in which the contacted
components included the causal nearest eight-hour round number was compared with an
equally constructed independent cluster whose contacted components contained no round
number. There were 102 independent pairs across all ten memes in early validation and
100 in late validation. The round-containing cluster's largest absolute movement in
the following hour was higher by about 0.275 prior ATR early, positive in 8 of 10
coins, and by about 0.203 ATR late, positive in 9 of 10. Pooled state balance was usable
in both periods. This confirms that the earlier result was not a transcription error
or a one-coin accident.

Contact activity gave the promised contrary check. Relative to the cluster without a
round number, contact range was about 0.036 trailing-range units lower early and 0.041
higher late, with only 4 and 5 coins positive. Contact volume was about 0.416
trailing-volume units lower early and 0.057 lower late, with only 3 and 5 coins
positive. Four-hour volume was moderately higher, about 0.195 early and 0.167 late,
positive in 7 and 6 coins. The repeatable observation is therefore post-contact
movement magnitude, not an immediate volume or range reaction.

The wider location ladder explains why that association cannot be promoted as an
eight-hour round-number edge. When a current round level that was already an
independent cluster member was compared with the same current round level without an
independent neighbour, the one-hour movement advantage was only about 0.043 ATR early
and reversed to about -0.119 late; late support was below the cohort floor. The
matched-no-level comparison was about -0.027 early and +0.205 late, but its maximum
state imbalance was about 0.76 and 0.58, so neither period was a clean attribution.
The causal-location shuffle was positive, about +0.432 and +0.176, but early coverage
was only 38 pairs and both periods were at or just above the maximum imbalance limit.

Most importantly, the symmetric shifted locations disagreed. The current round
cluster showed more one-hour movement than the +0.5-ATR and +1-ATR locations in both
periods, but less movement than the -0.25-ATR and -0.5-ATR locations in both periods.
Against the -2-ATR location it was also lower in both periods. Other shifts were mixed
or under-supported. A real reaction point should not require choosing only the side of
the shift ladder that makes it look favourable after results are visible. Near-miss
coverage was too small and changed from positive early to negative late. The 168-hour
stale construction produced no usable cohort row under the no-overlap rules, which is
an evidence gap but cannot repair the already failed shifted-location tests.

Plain interpretation: meme coins did tend to move farther during the hour after this
kind of round-containing cluster than after a cluster with no round number. However,
several nearby artificial prices produced equally large or larger movement, and a
fair no-level comparison was not balanced. The result may describe a broader active
price region, cluster geometry, or approach condition that often happens to include a
round number. It does not show that the exact round price localized the movement.

G3C is parked as a unique eight-hour round-number reaction edge. The descriptive
meme-specific movement association remains in the evidence register and may motivate
a later, separately frozen question about broader active-region geometry after the
whole G3 batch is reviewed. It cannot be used to tune a round-number indicator or to
claim a directional or profitable trade rule.

### G3D. Separate first arrival from occupancy in density zones

Trader question: do confirmed-swing or repeated-close density zones mark a fresh
reaction when price first arrives, or do they merely describe price already chopping
inside a broad accepted area?

Predeclare first arrival after a minimum outside-zone interval, near miss, already
inside, and repeat-contact episodes. Match current VP nodes, prior highs/lows, shifted
density zones, shuffled swing locations, and no-level controls on the common density,
coverage, and reference-level geometry that failed in G2E. Keep 168-hour and 720-hour
histories, single zones, density clusters, and density-plus-reference clusters
separate. Retain a fresh-arrival response only if range, volume, pressure magnitude,
dwell, or crossings survive the fair geometry ladder; otherwise keep only the honest
occupancy/acceptance description.

#### G3D progress log: fresh-arrival coverage and geometry controls, 2026-08-14

The first G3D step rewrote the vague old `already_inside_or_unclear` label into
observable candle states before opening any outcome. A fresh arrival means that the
same causal density area was already present during each of the previous six completed
one-hour candles, none of those six candle ranges touched the event area, the previous
close was clearly above or below it, and the current candle then touched it. An
already-inside event means the previous close was within that continuously available
area. A repeat contact means the previous close was outside but at least one of the
previous six candle ranges had already touched the area. A clean near miss means price
entered a band extending to twice the zone half-width without touching any density zone
on that surface. If the zone was new or had moved so that no overlapping same-surface
zone existed throughout the prior six candles, the timestamp was excluded rather than
mislabelled as a reaction.

The calculated level inputs were confirmed-swing price density and repeated-close
density, each built causally from completed one-hour candles over 168 hours and 720
hours. The fixed density bin was 0.25 prior ATR and at most three connected high-density
areas were retained per surface. Each timestamp kept zone width, rank, source-support
fraction, other density-zone overlap, other density-surface overlap, and overlap with
independent reference families. The reference families used only to classify the
geometry were current VP LVN/HVN/POC, rolling prior highs and lows, Bollinger boundaries,
round numbers, and their causally aligned 1h, 4h, 8h and 1d surfaces. Single density
zones, density clusters, and density-plus-independent-reference clusters stayed
separate. No RSI, MACD, moving average, direction label, or profit result was allowed to
choose these states or coverage cells.

The outcome-blind preflight retained 154,853 eligible normal-coin timestamps and 29,383
meme timestamps after excluding unstable/new-zone cases. A cell was allowed to open
reaction outcomes only when fresh arrival, clean near miss, and combined
already-inside/repeat occupancy each had at least 50 timestamps from at least five coins
in both validation periods. Nine normal-coin surface/scope cells passed. Four meme cells
passed: 168-hour swing-density clusters, 168-hour repeated-close density plus reference
clusters, and both the density-only and density-plus-reference 720-hour repeated-close
clusters. No isolated meme density-zone cell passed. This is a meaningful normal-versus-
meme difference: the normal set has enough isolated fresh arrivals for a direct test,
while the meme evidence remains cluster-conditioned under this definition.

One technical smoke initially joined every density cell to outcomes even though its
request named only the supported cells. Its integrity record failed because observed
cells exceeded approved cells. That smoke is rejected. The source join was repaired,
a regression test now verifies that only preflight-supported cells can pass, and a
second smoke passed with exactly the four approved meme cells. Another harmless summary
bug counted the same supported cell once per chronological period; the final preflights
count unique cells and report nine normal and four meme cells. These failed technical
runs are not evidence.

The first bounded outcome step then filtered the previously audited G2E matched pairs
to genuine fresh arrivals and repeated the one-, four-, and twenty-four-hour future-path
independence purge. It compared fresh arrivals with current VP HVN/LVN/POC, simple prior
highs/lows of the same history, zones shifted one ATR up and down, causal swing locations
shuffled within completed histories, market-state-matched no-level times, and
density-coverage-matched no-level times. Matching inputs were pre-event 1h/4h/24h local
returns, prior ATR, recent range and volume, six-hour candle pressure, RSI14, Bollinger
position and width, MACD histogram, EMA50 distance, BTC 24-hour movement, top-ten breadth,
mean absolute movement and dispersion, plus density and nearby-reference coverage.
Those generic indicators were context controls only; none was crossed with another or
treated as a reaction level in this test.

For memes, 15,082 independent comparisons survived. Only the 168-hour confirmed-swing
density-cluster cell had adequate coverage and usable state balance against the plain
same-market-state no-level control in both validation periods. Fresh contacts had about
0.440 and 0.300 more contact-volume units, 0.050 and 0.193 more contact-range units, and
0.102 and 0.167 ATR more one-hour absolute movement in early and late validation. They
also crossed and dwelled in the zone slightly more. However, the VP, shifted, shuffled,
prior-level, and density-coverage controls were sparse or materially state-imbalanced.
The meme result therefore says that a fresh clustered contact occurs during more active
behaviour than a comparable no-level time; it does not yet attribute that activity to
the exact density location.

For normal coins, 166,079 independent comparisons survived and several descriptions
were repeatable. The clearest exact-location lead was the isolated 168-hour
confirmed-swing density zone. Against same-state no-level times, contact volume was
about 0.423 higher early and 0.249 higher late; one-hour volume was about 0.290 and
0.035 higher; and one-hour absolute movement was about 0.097 and 0.113 ATR higher.
Contact volume was positive in 8 of 10 coins in both periods. One-hour volume and
absolute movement also remained positive against both one-ATR shifts and the causal
shuffled-swing location in both periods. The one-hour volume advantage was modest in
late validation against the shifts, and contact volume itself reversed slightly against
the +1-ATR shift late, so the narrow repeatable description is immediate one-hour
activity/movement rather than every later outcome.

The same isolated normal cell was also descriptively positive against the
density-coverage no-level control, VP HVN, and several other levels, but those matches
failed the maximum state-balance rule even when their median imbalance was small. Prior
high/low controls were especially unlike the fresh-arrival state. They cannot be used
as favourable evidence. Other supported density clusters mostly had fair comparisons
only with the plain same-state no-level control, or showed lower activity and more dwell
for repeated-close clusters. This means the broad claim that all density zones cause a
fresh reaction is already rejected. The isolated normal 168-hour swing-density result
is a lead, not a retained rule, until the predeclared clean near-miss, already-inside,
and repeat-contact outcome comparisons are completed.

Normal-versus-meme interpretation at this stage: normal coins supply an isolated
confirmed-swing density lead that survives both nearby shifted prices and shuffled swing
locations for one-hour activity. Meme coins supply only a cluster-conditioned activity
description whose exact-location controls lack fair common support. This difference may
reflect the shorter histories, faster participation shifts, and denser overlapping
levels in the meme set, but those possible explanations have not been tested and are
not used to create a subgroup.

#### G3D direct event-state result and present conclusion

The final predeclared G3D comparison rebuilt outcomes for every supported timestamp and
matched a fresh arrival directly with three different states at the same density
surface and scope. Clean near misses and repeat contacts were matched separately for
approaches from above and below on the previous close's distance from the outer zone
edge, within 0.10 prior ATR. Already-inside events have no approach side, so they were
matched on zone half-width within 0.10 ATR. All comparisons also matched the complete
prior-state indicator block listed above plus zone width, source-support fraction,
other-density-zone count and independent-reference count. The one-, four-, and
twenty-four-hour future paths were then independently purged again. The normal run kept
56,667 independent comparisons; the meme run kept 5,749. Integrity passed in both.

The clearest relationship was fresh arrival versus already-inside occupancy. For the
normal isolated 168-hour confirmed-swing density cell, fresh arrivals had about 0.889
more contact-volume units early and 1.414 more late, 0.709 and 1.031 more contact-range
units, 0.797 and 0.687 more one-hour volume, and 0.165 and 0.223 ATR more one-hour
absolute movement. Contact volume was higher in every one of the ten coins in both
periods; omitting any one coin left the equal-coin median positive. For the meme
168-hour swing-density cluster, fresh arrival versus already inside was also stable:
contact volume was about 0.816 and 0.949 higher, contact range about 0.631 and 0.685
higher, one-hour volume about 0.409 and 0.339 higher, and one-hour absolute movement
about 0.230 and 0.196 ATR higher. Nine or more memes supported contact volume and eight
supported one-hour volume; every leave-one-coin-out median stayed positive.

This does not by itself prove that the density calculation caused the reaction. A
candle arriving from outside must travel farther than a candle whose previous close is
already inside, so some contact-range difference is built into the event definitions.
The accompanying volume difference is not mathematically forced, but it can still mean
that fresh approaches generally occur during more active market episodes. The fair
same-state matching reduces that explanation but cannot eliminate unobserved news,
order flow or global-market causes.

Fresh arrivals also usually had higher contact volume than repeat contacts, but the
following-hour volume was not universally higher. In the normal isolated 168-hour
swing-density cell, contact volume was about 0.635 and 0.346 higher than at repeat
contacts. The future one-hour volume and movement differences weakened or changed sign
in late validation. In the meme 168-hour swing-density cluster, fresh arrival had about
0.092 and 0.216 more contact volume than a repeat contact, but one-hour volume was about
0.074 and 0.114 lower. Repeat touches can therefore occur within continuing high-volume
episodes. Freshness changes the type of event, but it is not a simple rule that every
first touch is more active than every later touch.

Near-miss evidence separated normal and meme markets again. Meme near-miss matches
collapsed to only 5-18 pairs from one to three coins in the validation cells, so no meme
near-miss result is usable. Normal coins supplied fair near-miss support for several
cluster cells but not for the isolated 168-hour swing-density lead. In the normal
168-hour swing-density cluster, a fresh contact had about 0.903 and 0.700 more contact
volume than a near miss and about 0.512 and 0.325 more one-hour volume; all ten coins
were represented. In repeated-close density-plus-reference clusters, fresh contacts
also had much higher contact range, volume, dwell and crossings than near misses, but
their one-hour absolute excursion was actually lower by about 0.549 and 0.419 ATR for
the 168-hour surface and by about 0.718 and 0.428 ATR for the 720-hour surface. Near
misses can therefore precede large movement without price ever touching the calculated
zone. Contact activity and later movement are different questions and must not be
collapsed into a single reaction score.

G3D conclusion: the original broad density-zone crossing result was substantially an
occupancy/event-definition effect and is rejected as a general level edge. Fresh arrival
is a real and repeatable market-behaviour distinction: across normal coins and memes it
usually brings more immediate range and volume than already-inside acceptance. The
narrowest remaining calculated-level lead is a fresh arrival at an isolated normal-coin
168-hour confirmed-swing density area. It survives one-ATR shifted locations, a causal
shuffled-swing location, same-state no-level times, and direct inside/repeat states for
immediate activity. It cannot yet be retained as an exact reaction-zone rule because
its clean near-miss cell lacks common support and its VP, prior-level and
density-coverage no-level comparisons were materially state-imbalanced. Keep that lead
queued for the post-batch review; do not tune the density indicator, predict direction,
or promote a strategy from it now. The meme result remains a cluster-conditioned fresh-
versus-occupancy behaviour description, not an exact density-location edge.

### G3E. Test a causal participation-change origin as a new level family

Trader question: when a completed one-hour candle marks an abrupt, jointly unusual
change in volume and range, does the price area where that participation shift began
remain a reaction zone on later revisits?

Run a coverage-only preflight first. Define change points from prior-only volume,
range, volatility, and pressure distributions; freeze a small threshold region without
looking at future reactions; and construct a research-only origin price or bounded
volume-weighted zone. Compare calendar and swing anchored VWAP, prior extremes,
current VP nodes, stale origins, shuffled origins, price shifts, near misses, and
same-state/no-level times. Do not edit a production indicator or tune the construction
to remembered events. Retain only a causal, localized, repeated response beyond those
existing families.

Work completed on 2026-08-14 so far:

The level construction was fixed before reaction outcomes were opened. The base
timeframe is one hour. For every completed candle, the code looks back over the prior
168 completed one-hour candles only. A candle becomes a participation-change source
when its volume is at or above the prior 90th percentile and its high-to-low range,
measured in ATR, is at or above the prior 75th percentile. ATR means the recent typical
price range and is used only to put differently priced coins on a common scale. At
least 134 earlier candles must exist. The source zone is the middle of the source
candle's real body, with a half-width equal to half that body but bounded between 0.10
and 0.50 of the prior ATR. The zone becomes usable only on the next one-hour candle and
is remembered for 168 hours.

Source-candle pressure and volatility were recorded but did not select the source.
Pressure here means the absolute open-to-close body divided by the complete high-to-low
range. The register records whether source pressure and ATR were above their own prior
75th percentiles, the source volume relative to the prior median and 90th percentile,
the source range relative to ATR and its prior 75th percentile, source-body direction,
zone age, and zone width.

A later event is not counted immediately. Price must first complete six consecutive
one-hour candles outside the source zone. The first later approach is then classified
as either a contact, where that candle's high-to-low range touches the source zone, or
a clean near miss, where it enters a zone twice as wide but does not touch the real
source zone. Each coin and timestamp contributes only one event. This avoids counting
many nearly identical active zones on the same candle as separate evidence.

Every origin was also crossed with levels already present at that timestamp. The
crossed timeframes were 1 hour, 4 hours, 8 hours, and 1 day. Existing levels included
causal confirmed-swing and repeated-close price-density zones, Volume Profile high-
volume nodes, low-volume nodes and point of control, previous price extremes,
Bollinger boundaries, and round-number references. The four explicit arrangements
were the origin alone, origin plus density, origin plus another reference, and origin
plus both density and another reference. A cluster therefore remains named as a
cluster; it is not silently treated as evidence for the new origin by itself.

The outcome-blind coverage result was strong enough to test. The ten established
coins supplied 40,114 source candles and 25,508 later contact or near-miss events. All
four arrangements had at least 50 contacts and 50 near misses from at least five coins
in each of the two internal validation periods. The ten meme coins supplied 8,217
source candles and 4,974 later events. The origin alone, origin plus density, and origin
plus density plus another reference passed the same gate in both meme validation
periods. Origin plus only another reference did not: the early meme period had 41
contacts and 48 near misses, so outcomes for that meme arrangement remain closed.

The first direct comparison matched real contacts to clean near misses on the same
coin, validation period, origin arrangement, approach from above or below, and
source-candle direction. It also matched the following information known before the
event: one-, four-, and twenty-four-hour local returns; ATR as a percentage of price;
recent range and volume; six-hour candle pressure; RSI(14); position and width within
the Bollinger(20) band; MACD histogram divided by ATR; distance from EMA(50); Bitcoin's
twenty-four-hour return; top-ten-coin breadth, average absolute movement, and return
dispersion; origin width and age; source volume, range, and pressure; whether source
pressure and ATR were unusually high; and nearby density/reference counts. RSI is a
bounded recent-momentum measure, Bollinger position says where price sits within a
moving volatility envelope, MACD compares faster and slower moving averages, and
EMA(50) is a fifty-candle exponentially weighted average. These inputs make the two
groups comparable; none was treated as an extra reaction level or as a direction call.

Measured behaviour remained direction-neutral: event-candle volume and range relative
to their prior twenty-four-hour medians; largest absolute movement in ATR; average
range and volume; absolute pressure change; fraction of closes remaining within the
zone; and number of closes crossing the level over 1, 4, and 24 hours. In files these
are still named `contact_*` because they use the shared event calculator. On a near-
miss row, that label means the near-miss or encounter candle, not a hidden contact.

The exact-direction matching run produced 11,342 independent established-coin
comparisons and 854 meme comparisons. Its one usable repeated normal-coin result was
the three-way cluster: participation origin plus a density zone plus another reference
level. Relative to a clean near miss, an actual contact had an event-candle range ratio
larger by about 0.75 in early validation and 0.65 in late validation. Event-candle
volume ratio was larger by about 1.07 and 0.78. Every one of the ten established coins
had a positive mean difference for both measures in both periods, and removing any one
coin kept the result positive. One-hour average range was higher by about 0.29 and
0.18, and one-hour average volume by about 0.39 and 0.31. Four-hour dwell and crossings
were also higher. In plain terms, contact with this three-way cluster coincided with
more immediate trading, wider candles, and more movement back and forth around the
level.

That did not mean price travelled farther away. Largest absolute movement in the first
hour was lower than for the matched near misses by about 0.36 ATR early and 0.47 ATR
late. The four-hour difference was also lower, about 0.28 and 0.25 ATR. This combination
looks more like local two-way trading, absorption, or acceptance around a busy cluster
than a breakout. It is a reaction lead, not a direction forecast. The result also
cannot yet say the new origin caused the behaviour, because the qualifying events
already contain a density zone and another known reference.

The meme comparison was technically underpowered after matching. Although the
coverage preflight included all ten meme coins, splitting every pool by source-candle
direction left only one or two coins with enough independent matched episodes in the
validation cells. No meme result from that run is accepted as evidence.

A second technical run removed that unnecessary split. Source-candle direction stayed
in the state-balancing inputs, but events were no longer divided into small separate
pools. This raised independent comparisons to 13,202 for established coins and 1,510
for memes. It did not rescue the claim. The meme early period still had no comparison
with both five-coin/fifty-event coverage and acceptable prior-state balance. The meme
late period had enough events for one arrangement but the contact and near-miss groups
still differed too much before the event. For established coins, the broader pooling
also revealed source-direction/state imbalance in early validation. This second run is
therefore recorded as a coverage repair and a warning about source-direction context,
not selected because it produced a preferable number.

Current normal-versus-meme interpretation: established coins presently support a
narrow, conditional three-way-cluster activity lead when source-candle direction is
held equal. Meme coins do not yet supply a fair matched confirmation or rejection.
This may mean the meme group has fewer independent episodes, more abrupt state changes,
or a genuinely different relationship, but these data do not distinguish those
possibilities. No outcome-discovered meme subgroup is allowed here.

Controls still required before G3E can close are component comparisons against current
Volume Profile nodes, prior extremes and anchored VWAP; artificial shifted and shuffled
origin locations; an origin after it has become stale; and a matched time with the same
market state but no calculated level. The clean near miss is complete. The next control
stage must say explicitly whether it tests the complete three-way cluster or the added
value of the participation origin. Until those controls repeat, this is a queued lead,
not a retained level, indicator edit, FreqAI target, or trading rule.

The density/reference component stage is now complete. It reused the audited G2E
density comparisons but did not accept a same-candle coincidence. For each candidate
it reconstructed the exact causal density family, 168- or 720-hour history and zone
rank, then required that specific density zone to overlap the participation-origin
zone. It re-applied the one-, four-, and twenty-four-hour future-path separation after
filtering. The tested controls were current VP high-volume node, low-volume node and
point of control; prior high and prior low of the matching history; density zones
shifted one ATR up and down; causal shuffled-swing density; a matched market state with
no level; and a matched state with similar broad density coverage but no level at the
tested price.

The established coins supplied 42,797 independent component comparisons. The meme
coins supplied 6,018, but no meme validation comparison had both the required coverage
and acceptable prior-state balance, so the meme component result is insufficient.
For established coins, most VP, prior, shifted, shuffled, and density-coverage controls
were either too sparse or too different in prior state for a fair repeated comparison.
The one complete repeated control for the three-way lead was the matched same-state
no-level time for a 168-hour repeated-close density zone.

Against that fair no-level control, the proposed cluster still spent more time around
the level and crossed it more often over four hours in both validation periods. But
the apparent immediate range and volume advantage did not repeat: event-candle range
was only about +0.03 early and then -0.52 late, while event-candle volume was about
+0.26 early and -0.56 late. One-hour range and volume also changed from approximately
flat or positive early to negative late. One-hour largest absolute movement remained
lower, about -0.16 ATR early and -0.74 ATR late. This weakens the interpretation from
"the origin cluster causes more participation" to the narrower description "some
contacts remain local, cross repeatedly, and do not travel as far." That description
could reflect an existing repeated-close density/reference cluster without any added
information from the new origin.

The low-volume-node comparison had enough coins and events but failed the prior-state
balance requirement. Its descriptive signs were still a warning: the origin cluster
had lower one-hour absolute movement, range, and volume than a current LVN in both
periods, while four-hour dwell was higher. Shifted and density-coverage controls also
had incomplete or imbalanced support. Therefore the component ladder does not retain
the new origin-level claim. Artificial shifted, shuffled and stale origin controls,
plus the anchored-VWAP overlap check, remain the final G3E closure work; they may add a
diagnostic description but cannot rescue the failed requirement to beat existing
families consistently.

The final artificial-origin stage is complete. Four controls were built with the same
one-hour causal data and first-revisit rule. The first moved every real origin centre
down by one source-time ATR, the second moved it up by one ATR, and both kept the real
source's width and volume/range/pressure properties. The third moved source identity by
one third of each already-fixed chronological period, constructed an ordinary body
zone at that new timestamp, and excluded timestamps that were real participation
origins. The fourth kept the exact real source zone but restarted the six-outside-
candle test only after the normal 168-hour memory expired, so its contacts occurred at
source ages from 169 through 336 hours. All controls were matched on coin, period,
origin arrangement, approach side, source-body sign, outer-edge distance, local OHLCV
and indicator state, Bitcoin/top-ten state, and source properties. Age was deliberately
not matched for the stale comparison because age is the property being tested.

The established coins supplied 32,718 independent artificial-control comparisons.
The meme coins supplied 2,282, but no meme validation cell reached both five-coin/
fifty-pair coverage and acceptable prior-state balance. This is again insufficient
meme evidence, not a negative meme result.

Several normal-coin comparisons were fair in both validation periods, but they did not
describe one universal origin reaction. For the origin alone, the real location had
slightly more event-candle range and volume than the one-ATR-lower location: range was
about +0.04 and +0.19, while volume was about +0.15 and +0.02 in early and late
validation. The one-ATR-higher control did not repeat the same complete result. This
one-sided difference is not a robust exact-location effect and may reflect the market's
position and movement around the source rather than the origin calculation.

Fresh isolated origins were less active than the same source zones after expiry.
Relative to stale contacts, one-hour largest absolute movement was about -0.21 ATR in
both validation periods, one-hour average range was about -0.27 and -0.10, and one-hour
average volume was about -0.47 and -0.14. Origin-plus-density contacts were also lower
than stale contacts in event-candle range, about -0.07 and -0.10, and volume, about
-0.17 and -0.33. Relative to the one-ATR-higher location, origin-plus-density contacts
had lower one-hour absolute movement, range, and volume in both periods. This is the
opposite of a simple claim that a fresh participation origin creates stronger
activity.

The three-way origin-plus-density-plus-reference cluster had enough fair evidence in
both periods only against the one-ATR-lower origin control. Its one-hour absolute
movement difference was about +0.20 ATR early but only +0.02 late. The one-ATR-higher
side lacked late coverage, stale three-way contacts lacked coverage, and shuffled
source comparisons did not achieve fair prior-state balance. A rule cannot be retained
from the one surviving side while the other required alternatives are absent.

G3E conclusion: park the proposed participation-change origin as a new standalone
level family. A useful descriptive pattern remains—some established-coin contacts at
busy density/reference clusters dwell and cross locally while travelling less—but the
new origin does not consistently add reaction strength beyond near misses, no-level
market states, existing density/reference components, shifted locations, shuffled
sources, or stale sources. Calendar and swing anchored VWAP had already failed their
own complete G2D localization ladder and remain an alternative-family negative rather
than support for this origin. An exact origin-plus-anchored-VWAP overlap study can be
queued for a later branch only if another batch supplies a new reason; the G3E four-
iteration limit is reached. Do not tune the thresholds, edit an indicator, open a
FreqAI target, predict direction, or create a trading rule from G3E.

### G3F. Test generic calculated levels explicitly instead of as feature soup

Trader question: do any simple, visible reference levels mark a repeatable reaction
when each family is named and tested separately?

The fixed source families are previous completed week and month high, low, and
midpoint; SMA(20), SMA(50), and SMA(200); EMA(12), EMA(26), and EMA(50); and the upper,
middle, and lower Bollinger(20) references. Preserve source timeframe, age, approach,
single versus cluster scope, and cross-timeframe convergence. Do not pool them as
"some indicator nearby." Use shifted, stale, shuffled-identity, matched no-level,
simple-prior-level, density, and component-ablation controls. RSI, MACD, ATR, trend,
volume, pressure, and broad-market state may make comparisons fair but are not extra
level claims. Retain only a named family with repeated localized behaviour.

#### G3F completed investigation log: direct associations did not survive attribution

Work ran on 14 August 2026. The outcome-blind normal preflight completed at
15:29 UTC, the corrected normal direct run at 15:43 UTC, the artificial normal and
meme runs at 16:26 and 16:14 UTC, and the frozen all-control review at 16:27 UTC.
This entry records what was calculated, why it was calculated, every stable
indicator/timeframe relationship opened by the direct review, the normal/meme/platform
difference, and the resulting interpretation. It is deliberately longer than a status
summary so a later agent does not have to reconstruct the question from Parquet files.

##### Exact calculated levels and timeframes tested

The one-hour analysis clock was used only as the common observation clock. Indicator
values were calculated on their own source candles and became available only after
those source candles had closed. The fixed level set was:

- SMA(20), SMA(50), and SMA(200) on 1h, 4h, 8h, and 1d candles;
- EMA(12), EMA(26), and EMA(50) on 1h, 4h, 8h, and 1d candles;
- Bollinger(20) lower, middle, and upper lines on 1h, 4h, 8h, and 1d candles;
- the high, low, and midpoint of the previous fully completed UTC week; and
- the high, low, and midpoint of the previous fully completed UTC month.

An unfinished 4h, 8h, daily, weekly, or monthly candle was never backfilled into an
earlier one-hour observation. A weekly level required all 168 hourly candles. A monthly
level required every hourly candle in that actual UTC calendar month. This means a
previous-month line was not assumed to have a fixed thirty-day source period merely
because its nominal comparison history is recorded as 720 hours.

The contact zone around every line was the larger of 0.10 causal ATR or 0.05% of the
line price. A contact episode started only after at least six candles outside the zone,
or after the dynamic line itself had moved enough to create a genuinely new encounter.
This prevented each candle of one prolonged touch from being counted as a new market
reaction.

##### What “crossed” meant in this test

Every contact row retained the exact names, families, source timeframes, dependency
groups, and prices of every other calculated line whose zone overlapped the contacted
line. The stable cell labels mean:

- `isolated_generic_level`: no other tested calculated line overlapped the contacted
  line at that event;
- `same_timeframe_same_family`: another line from the same indicator family and source
  timeframe overlapped it;
- `same_timeframe_cross_family`: an SMA, EMA, Bollinger, or calendar line from the same
  source timeframe overlapped it but carried a different family name;
- `cross_timeframe_same_family`: the same family overlapped from another source
  timeframe;
- `cross_timeframe_cross_family`: at least one other timeframe and at least one other
  named family overlapped; and
- `same_dependency_group` versus `cross_dependency_group`: the former contains only
  lines ultimately derived from moving price averages, while the latter also contains
  a genuinely different construction such as completed calendar structure.

This distinction matters. An SMA, EMA, and Bollinger middle can have different names
without being independent market mechanisms. SMA(20) and Bollinger(20) middle are
mathematically identical on the same source timeframe. The first diagnostic direct
runs, `g3f_generic_reaction_normal10_full_20260814a` and
`g3f_generic_reaction_meme_full_20260814a`, counted that identity too favourably and
are invalid for evidence. Schema 2 keeps both visible names but records the alias and
does not treat the contacted SMA(20) and Bollinger middle as two independent cluster
components. All conclusions below use corrected preflights ending `20260814c` and
corrected direct runs ending `20260814b`.

One relationship cell can contain several exact peer combinations at different events.
It would therefore be false to describe, for example, every 4h EMA(12)
cross-timeframe/same-family contact as one fixed EMA pair. The exhaustive event-level
log is retained in the preflight `pair_geometry` files through
`overlap_level_names`, `overlap_level_families`, `overlap_level_timeframes`,
`overlap_dependency_groups`, and the corresponding exact keys. The plain-text cell
inventory below records every stable relationship that reached the frozen direct-lead
stage; the row-level files preserve every varying peer combination inside it.

##### Additional inputs used to make comparisons fair

These values described conditions before contact. They were matching inputs, not
extra claims that RSI, MACD, or another indicator caused a reaction:

- the coin's return over the preceding 1, 4, and 24 hours;
- ATR as a percentage of price, recent candle range relative to its prior median, and
  recent volume relative to its prior median;
- six-hour candle pressure, where pressure describes where candles closed within their
  ranges rather than a buy/sell direction forecast;
- RSI(14);
- the prior close's position inside the local Bollinger envelope and the envelope width
  in ATR;
- the MACD histogram divided by ATR;
- distance from EMA(50) divided by ATR;
- BTC's prior 24-hour return;
- top-ten breadth, meaning the fraction of the broad coin group rising;
- the top-ten group's mean absolute movement and return dispersion;
- the contacted zone width in ATR;
- the number of formula-deduplicated generic levels within two ATR of price;
- the fraction of the four-ATR neighbourhood covered by generic zones; and
- source age divided by the line's nominal history where source age was a fair matching
  field.

Source age was deliberately not matched for the stale control because being old is the
point of that control. It was also omitted for the continuously refreshed prior-high
and prior-low controls. Dropping it there does not make those controls favourable: all
other declared state, approach geometry, level density, and zone width still had to be
comparable.

##### Market behaviour measured after contact

No signed direction and no profit were measured. The saved output names translate as:

- `contact_range_ratio`: contact-candle high-to-low range divided by the preceding
  24-hour median candle range;
- `contact_volume_ratio`: contact-candle volume divided by the preceding 24-hour median
  volume;
- `abs_excursion_atr_h1/h4/h24`: the largest move either above or below the level during
  the next 1, 4, or 24 candles, measured in the ATR known at contact;
- `range_ratio_h1/h4/h24`: mean candle range during that future window divided by the
  preceding range median;
- `volume_ratio_h1/h4/h24`: mean future volume divided by the preceding volume median;
- `pressure_change_abs_h1/h4/h24`: the size of the change in average candle pressure,
  ignoring whether that change was upward or downward;
- `dwell_fraction_h1/h4/h24`: the fraction of future closes still inside the declared
  level zone; and
- `crossings_h1/h4/h24`: how many times consecutive closes changed from one side of the
  centre line to the other.

“Higher” therefore means more of that named behaviour than the control, not “price went
up.” “Lower” means less of that behaviour, not a short prediction.

##### Direct and artificial controls

The direct no-level control placed a pseudo-zone on another candle of the same coin and
period. It matched approach side, distance to the zone, all declared market-state
inputs, zone width, nearby generic-level count, and total generic-zone coverage, but no
real generic line was allowed to overlap the pseudo-zone. This one comparison therefore
supplied both the frozen same-state/no-level and same-density requirements.

For a clustered contact, component ablation compared the exact same named anchor while
it was isolated. This asked whether convergence added behaviour beyond the anchor
alone. The artificial ladder then used six controls:

- the same line shifted down by exactly one causal ATR;
- the same line shifted up by exactly one causal ATR;
- the same line value delayed by one complete nominal indicator history;
- a fixed identity swap inside the same family and timeframe, such as SMA(20) using the
  SMA(50) location, while the donor line itself was excluded from cluster membership;
- the highest fully completed one-hour high over the same nominal history; and
- the lowest fully completed one-hour low over the same nominal history.

An artificial event was discarded if its zone also touched the current real target or
a mathematical alias of that target. Its remaining peer lines then had to reproduce
the candidate's exact same/cross-timeframe, same/cross-family, and dependency
relationship. This is stricter than asking whether any arbitrary fake line happened to
be nearby.

##### Coverage and frozen lead counts

The corrected outcome-blind preflight opened 153 exact normal-coin cells and 87 exact
meme cells with at least five events per coin/period and at least 50 events across five
coins in both validation periods. It did not inspect any reaction result.

The direct normal run retained 540,409 independent comparisons. The direct meme run
retained 74,155. A strict two-period, coin-agreement, and leave-one-coin-out review
froze 94 direct lead rows before artificial outcomes were opened. Two normal outcome
claims appeared twice because they passed two direct controls with opposite signs, so
the actual frozen set was 92 distinct outcome claims in 50 structural cells:
65 claims in 33 normal cells, 12 claims in four meme cells, and 15 claims in thirteen
predeclared platform cells. The platform cells are a separate claim scope drawn from
ETH, BNB, SOL, ADA, AVAX, and TRX; they are not extra rows double-counted as a second
broad normal result.

The full artificial normal run produced 649,674 independent comparisons across all ten
coins and all six controls. It produced 13,366 cohort result rows, of which 2,578 had
both the required event/coin coverage and usable state balance. The meme artificial
run produced 15,205 independent comparisons and 1,136 cohort rows, but only 42 were
coverage-and-balance eligible; those 42 were in the development period rather than in
both frozen validation periods. Both runs had zero future-source violations, zero
event-separation violations, all requested coins, all six controls, and a maximum
approach-distance difference below 0.10 ATR.

##### Every frozen structural cell tested

The output names in this inventory use the plain definitions above. “Best 2/8,” for
example, means the best outcome in that cell passed only two of eight required controls;
it is not a two-out-of-eight acceptance rule.

Normal ten-coin cells:

- 1h Bollinger lower, cross-timeframe/cross-family and cross-dependency:
  24h absolute pressure change; best 1/8.
- 1h Bollinger lower, cross-timeframe/same-family and same-dependency:
  4h absolute movement, contact range, 24h crossings, 4h and 24h range, and 24h volume;
  best 1/8.
- isolated 1h Bollinger middle: 4h absolute pressure change; best 1/7.
- 1h Bollinger upper, cross-timeframe/cross-family and same-dependency:
  1h and 4h range and 1h volume; best 1/8.
- 1h Bollinger upper, cross-timeframe/same-family and same-dependency:
  contact range and 24h crossings; best 1/8.
- 1h EMA(12), cross-timeframe/cross-family and cross-dependency:
  24h volume; best 1/8.
- isolated 1h EMA(12): 1h absolute movement and 1h volume; best 1/7.
- 1h EMA(12), same-timeframe/same-family and same-dependency:
  24h crossings, 4h and 24h range, and 24h volume; best 1/8.
- 1h EMA(26), same-timeframe/cross-family and same-dependency:
  contact volume, 1h absolute pressure change, and 24h volume; best 1/8.
- 1h EMA(50), cross-timeframe/cross-family and same-dependency:
  24h dwell; best 1/8.
- 1h EMA(50), cross-timeframe/same-family and same-dependency:
  1h absolute movement; best 1/8.
- isolated 1h SMA(20): 24h volume; best 1/7.
- 1h SMA(20), same-timeframe/cross-family and same-dependency:
  1h absolute movement; best 1/8.
- isolated 1h SMA(50): 24h absolute pressure change and 4h volume; best 2/7.
- 1h SMA(50), same-timeframe/cross-family and same-dependency:
  4h absolute movement; best 1/8.
- previous-week high, cross-timeframe/cross-family and cross-dependency:
  4h crossings; best 1/8.
- previous-week midpoint, cross-timeframe/cross-family and cross-dependency:
  contact volume and 1h absolute pressure change; best 1/8.
- 4h Bollinger lower, cross-timeframe/cross-family and cross-dependency:
  4h absolute movement, contact range, 1h and 24h range, and 24h volume; best 1/8.
- 4h Bollinger lower, cross-timeframe/same-family and same-dependency:
  4h range; best 1/8.
- 4h Bollinger middle, cross-timeframe/same-family and same-dependency:
  4h absolute movement and pressure change; best 1/8.
- isolated 4h Bollinger middle: contact volume and 1h range; best 1/7.
- 4h Bollinger upper, cross-timeframe/cross-family and cross-dependency:
  24h absolute movement and volume; best 1/8.
- 4h Bollinger upper, cross-timeframe/same-family and same-dependency:
  contact range; best 1/8.
- 4h EMA(12), cross-timeframe/cross-family and same-dependency:
  1h and 4h range; best 1/8.
- 4h EMA(12), cross-timeframe/same-family and same-dependency:
  1h and 4h absolute movement, 4h crossings, 1h and 4h range, and 1h and 4h volume;
  best 2/8, with contradictory controls on three outcomes.
- 4h SMA(20), cross-timeframe/cross-family and same-dependency:
  1h absolute movement; best 1/8.
- 4h SMA(20), cross-timeframe/same-family and same-dependency:
  1h absolute pressure change; best 1/8.
- 8h Bollinger lower, cross-timeframe/cross-family and same-dependency:
  4h absolute pressure change; best 1/8.
- 8h Bollinger middle, cross-timeframe/same-family and same-dependency:
  1h and 24h absolute movement and 1h absolute pressure change; best 1/8.
- 8h Bollinger upper, cross-timeframe/cross-family and same-dependency:
  4h dwell; best 1/8.
- isolated 8h Bollinger upper: 1h volume; best 1/7.
- 8h SMA(20), cross-timeframe/cross-family and same-dependency:
  1h absolute movement; best 2/8 with contradictory controls.
- 8h SMA(50), cross-timeframe/cross-family and same-dependency:
  4h volume; best 1/8.

Meme ten-coin cells:

- 1h Bollinger lower, cross-timeframe/same-family and same-dependency:
  1h absolute movement, contact range, and 24h absolute pressure change; best 1/8.
- 1h Bollinger upper, cross-timeframe/cross-family and cross-dependency:
  24h crossings; best 1/8.
- 1h Bollinger upper, cross-timeframe/cross-family and same-dependency:
  4h and 24h absolute movement, range, and volume; best 1/8.
- 1h Bollinger upper, cross-timeframe/same-family and same-dependency:
  24h crossings and range; best 1/8.

Predeclared six-platform-coin cells:

- 1h Bollinger lower, cross-timeframe/same-family and same-dependency:
  contact range; best 1/8.
- isolated 1h Bollinger middle: 4h absolute pressure change; best 1/7.
- 1h Bollinger upper, cross-timeframe/cross-family and same-dependency:
  4h range; best 1/8.
- 1h Bollinger upper, cross-timeframe/same-family and same-dependency:
  1h range and 4h volume; best 1/8.
- 1h EMA(12), same-timeframe/same-family and same-dependency:
  1h crossings and 4h range; best 1/8.
- 1h EMA(26), same-timeframe/cross-family and same-dependency:
  1h volume; best 1/8.
- 1h EMA(50), cross-timeframe/cross-family and same-dependency:
  24h dwell; best 1/8.
- isolated 1h SMA(20): 24h dwell; best 1/7.
- isolated 4h Bollinger middle: 24h dwell; best 1/7.
- 4h EMA(12), cross-timeframe/same-family and same-dependency:
  4h range; best 1/8.
- 4h SMA(20), cross-timeframe/cross-family and same-dependency:
  1h absolute pressure change; best 1/8.
- 8h Bollinger upper, cross-timeframe/cross-family and same-dependency:
  4h dwell; best 1/8.
- 8h SMA(20), cross-timeframe/cross-family and same-dependency:
  1h absolute movement; best 1/8.

No daily, previous-month, previous-week-low, SMA(200), or equivalent unlisted cell is
being silently called negative. Those families were tested in the outcome-blind
coverage/direct surface, but they did not reach this frozen lead inventory with a
repeatable direct outcome under the declared support rule.

##### Final normal-coin result

None of the 65 normal outcome claims passed its complete seven-control isolated ladder
or eight-control cluster ladder. Four claims became explicitly contradictory and 61
had an incomplete fair-control ladder.

The clearest contradiction was the 4h EMA(12) when it overlapped a same-family line
from another timeframe. Compared with isolated 4h EMA(12), the cluster had 0.465 and
0.215 ATR more one-hour absolute movement in early and late validation, and 0.641 and
0.354 ATR more four-hour absolute movement. Compared with matched no-level times,
however, one-hour absolute movement was 0.034 and 0.186 ATR lower. Four-hour range was
0.284 and 0.166 units higher than isolated EMA(12), but 0.039 and 0.121 units lower than
no-level times. The honest interpretation is not that the convergence caused a strong
reaction. The isolated EMA(12) comparison baseline was unusually quiet, so adding a
second EMA looked active relative to that narrow baseline while remaining quieter than
ordinary comparable market times.

The same cluster's four-hour absolute movement was higher than isolated EMA(12) but
0.512 and 0.103 ATR lower than the stale-line control. Its crossings were lower than
both the isolated anchor and the identity-shuffled location, but its no-level crossing
sign changed between validation periods. These are different conditional comparisons,
not one coherent reaction signature.

The other explicit contradiction was an 8h SMA(20) in a cross-timeframe/cross-family,
same-dependency cluster. One-hour absolute movement was 0.091 and 0.062 ATR higher than
isolated 8h SMA(20), but 0.139 and 0.091 ATR lower than the fixed shuffled-identity
location. Again, the cluster beat a quiet component but did not establish the exact
SMA(20) location.

One narrow partial remains useful as a later question without being promoted. An
isolated current 1h SMA(50) had four-hour mean volume 0.247 and 0.231 units lower than
matched no-level times, and 0.153 and 0.129 units lower than the same SMA(50) value
delayed by its 50-hour nominal history. Eight of ten coins were on the lower-volume side
in both comparisons. It did not have fair support against either one-ATR shift, the
identity shuffle produced no complete two-period comparison, and prior highs/lows were
state-imbalanced. This is a possible “current isolated SMA(50) marks quieter conditions”
lead, not proof that SMA(50) is the cause or a trading rule.

Across the 65 normal claims, the plus-one-ATR control had 42 state-imbalanced, 15
insufficient-coverage, one missing, five sign-unstable, and two member-unstable results;
none passed. The minus-one-ATR control had 46 imbalanced, nine insufficient, three
missing, five sign-unstable, and two member-unstable; none passed. The identity shuffle
passed two claims, but had 25 imbalanced, 20 insufficient, nine missing, eight
sign-unstable, and one member-unstable. Prior highs and lows passed none. The stale
control passed two claims but had 21 imbalanced, ten insufficient, six missing,
seventeen sign-unstable, and nine member-unstable.

The main state differences were not random bookkeeping noise. Shifted locations were
most often separated by prior Bollinger position or distance from EMA(50). Prior
high/low controls were even more strongly separated by those two fields because prior
extremes naturally occur at different chart locations. Identity shuffles also differed
in nearby level count. Stale controls were closer, but their remaining failures were
often Bollinger width, Bollinger position, or generic-zone coverage. This means the
data cannot yet separate “the exact line matters” from “price occupied a different
technical location before contact.” Removing those fields would answer a looser
question, so it is queued as a later decomposition rather than used to rescue G3F.

##### Final meme result and how it differs

All twelve meme claims came from component ablation: a Bollinger cluster behaved
differently from the same Bollinger anchor when isolated. None passed the matched
no-level control, and none of the six artificial controls passed a complete two-period
validation. The plus-one shift had eight imbalanced, one insufficient, and three
missing claims; the minus-one shift had six imbalanced, five insufficient, and one
missing. The identity shuffle had six imbalanced, two insufficient, and four missing.
Prior highs had six imbalanced and six missing; prior lows had nine insufficient and
three missing. The stale control had six imbalanced, three insufficient, and three
missing.

For the coverage-valid meme artificial comparisons, prior Bollinger position was the
largest state difference in every frozen validation group. Typical maximum standardized
differences were about 1.38 to 1.65 for the shifted locations, about 1.17 for stale
locations, and roughly 10 for the identity shuffle or the one supported prior-low
group, far above the usable 0.50 limit. This is expected geometry, not permission to
ignore the control: a one-hour Bollinger boundary contact inherently occurs at a
different normalized band position from many fake locations.

The meme result is therefore narrower than the normal result. Memes supplied only
one-hour Bollinger cluster leads and did not supply even the limited no-level-plus-stale
SMA(50) pattern seen in established coins. The present evidence says some meme
Bollinger clusters differ from their isolated band, but cannot tell whether the exact
band, general price stretch, surrounding trend, or another nearby level produced that
difference.

##### Final platform-subgroup result

The fixed ETH/BNB/SOL/ADA/AVAX/TRX subgroup produced fifteen direct claims. Ten
component comparisons and five matched no-level comparisons individually passed the
direct repeatability rule, but no artificial control passed any complete platform
claim. Sixty-six of 117 required control rows were state-imbalanced, thirteen lacked
coverage, nine were missing, ten changed effect sign, and four failed the five-of-six
member rule. The rational narrower group therefore does not rescue a generic-level
reaction claim.

##### G3F conclusion and queued implications

G3F is parked, not promoted. Simple visible lines do correlate with different market
behaviour in selected comparisons, but none of the named levels or convergence cells
proved that its exact current location marked a repeatable reaction beyond every fair
alternative. The study also shows why “a cluster beats one component” is not enough:
the component can be unusually quiet, another fake location can be stronger, or a
matched no-level baseline can reverse the story.

The following result-inspired ideas are queued for a later breadth-reviewed batch and
must not interrupt G3G or G3H:

- decompose a Bollinger boundary into exact-line contact versus the broader condition
  that price is stretched within its recent distribution;
- decompose moving-average contact into exact-line location versus ordinary distance
  from the moving-average bundle;
- revisit the isolated 1h SMA(50) low-volume partial with a control specifically
  designed to share support in EMA distance and Bollinger position;
- compare genuinely independent clusters, such as price-average plus completed
  calendar structure, separately from multiple labels derived from the same average;
  and
- preserve the actual peer-name/timeframe signature rather than pooling all
  cross-timeframe events if a future coverage audit can support those smaller cells.

These are leads about how to ask the next question, not evidence for direction, profit,
entry, exit, or an indicator edit.

##### G3F evidence files

- corrected normal preflight:
  `generation3_branches/g3f_generic_levels/g3f_generic_levels_normal10_full_20260814c/`;
- corrected meme preflight:
  `generation3_branches/g3f_generic_levels/g3f_generic_levels_meme_full_20260814c/`;
- corrected normal direct results:
  `generation3_branches/g3f_generic_levels_reaction/g3f_generic_reaction_normal10_full_20260814b/`;
- corrected meme direct results:
  `generation3_branches/g3f_generic_levels_reaction/g3f_generic_reaction_meme_full_20260814b/`;
- frozen direct-lead and platform review:
  `generation3_branches/g3f_generic_levels_review/g3f_direct_review_20260814a/`;
- normal artificial controls:
  `generation3_branches/g3f_generic_levels_artificial/g3f_generic_artificial_normal10_full_20260814a/`;
- meme artificial controls:
  `generation3_branches/g3f_generic_levels_artificial/g3f_generic_artificial_meme_full_20260814a/`;
- final all-control outcome and cell assessment:
  `generation3_branches/g3f_generic_levels_attribution_review/g3f_attribution_review_20260814a/`;
- bulky pair geometry and independent comparison files:
  `D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation3_branches\`;
  and
- technical smoke `g3f_generic_artificial_normal_eth_smoke_20260814a` is not evidence
  and is superseded by the full normal run.

### G3G. Run one evidence-triggered one-minute replay for the thin-LVN lead

Status: completed as one direct diagnostic pass. No FreqAI model was fitted because
twelve episodes cannot support chronological training, validation, and untouched
confirmation. No profit, entry, exit, leverage, or indicator parameter was optimized.

#### G3G question, baseline, and declared diagnostic criteria

Baseline: corrected G3A found that high-thinness one-hour LVNs inside genuinely mixed
peer clusters sometimes coincided with unusually high contact volume, especially in
later meme data. It did not say whether price would break through or reject the area.

Hypothesis: after price approaches one of these qualified areas, the area produces a
larger change in price activity than a similar pre-event minute with no causally
available calculated level. Causal recent direction, pressure, momentum, volatility,
the exact cluster, and wider market state may then help explain whether the next move
continues through the area or moves away.

Strongest alternative: the event was already in an unusual momentum, volume, or
volatility state, so any ordinary minute with the same state would have moved just as
much. In that case the LVN and cluster are labels placed on an already active move,
not useful reaction locations.

The diagnostic reaction label was declared as both of the following within sixty
minutes:

1. post-contact price travels at least one frozen parent-zone half-width from the
   completed contact-minute close; and
2. mean volume from the contact minute onward is at least 1.5 times mean volume over
   the preceding sixty minutes.

This threshold is an agent-declared diagnostic, not a native Freqtrade, FreqAI, or
machine-learning relevance score. Continuous movement, volume, volatility, pressure,
dwell, crossings, and timing remain more informative than the label.

Direction begins after the contact candle has closed. Up and down first passage use
symmetric one-zone-half-width moves from that close. Zone resolution is a separate
geometric measurement:

- breakout means price first reaches the far edge of the frozen zone after contact;
- rejection means price first travels one additional zone half-width beyond the near
  edge from which it entered;
- a same-minute tie remains a tie; and
- if neither boundary is reached inside sixty minutes, the zone is unresolved.

The small diagnostic reaction comparison required at least four balanced actual/control
pairs in each cohort, at least four actual reactions among those pairs, and a higher
actual reaction rate than the no-level control. This is only a keep-or-park rule for
the diagnostic. It is not the programme's 55% or 65% success target.

#### G3G frozen selection and exact episodes

Selection used no signed one-minute future path. A parent had to be:

1. an actual high-thinness one-hour `lvn_above` or `lvn_below` from corrected G3A;
2. approached clearly from outside on the expected side;
3. accompanied by a surviving connected peer cluster with at least two contacted,
   independent dependency groups after the LVN's own Volume Profile group was removed;
4. followed by contact-hour volume at least twice the causal preceding twenty-four-hour
   median; and
5. at least forty-eight hours from another selected episode for the same pair.

There were 837 qualifying established-coin events and 98 meme events before the
forty-eight-hour independence filter, then 770 and 94 respectively. A deterministic
hash selected one event for each cohort, chronological period, and LVN side while
preferring six distinct pairs per cohort. This produced the following twelve-event
microscope sample:

- established development: ETH `lvn_above`, 2022-03-15 16:00 UTC;
- established development: LINK `lvn_below`, 2023-10-19 01:00 UTC;
- established early validation: SOL `lvn_above`, 2024-06-27 12:00 UTC;
- established early validation: AVAX `lvn_below`, 2024-04-10 02:00 UTC;
- established late validation: XRP `lvn_above`, 2025-06-12 20:00 UTC;
- established late validation: DOGE `lvn_below`, 2025-10-29 15:00 UTC;
- meme development: 1000PEPE `lvn_above`, 2025-07-27 22:00 UTC;
- meme development: DOGE `lvn_below`, 2025-10-27 00:00 UTC;
- meme early validation: PUMP `lvn_above`, 2026-01-06 14:00 UTC;
- meme early validation: FARTCOIN `lvn_below`, 2026-01-25 16:00 UTC;
- meme late validation: TRUMP `lvn_above`, 2026-05-05 08:00 UTC; and
- meme late validation: 1000SHIB `lvn_below`, 2026-05-13 15:00 UTC.

The highest independent cluster component set the context anchor: one event used a
one-hour anchor, three used four hours, and eight used eight hours. The initial exact
coverage audit checked 154,920 requested one-minute rows across eleven merged pair
intervals and found no missing minute, duplicate, gap, or invalid OHLCV row.

The direction-neutral boundary audit found that five episodes still had activity or
zone context touching an initial edge. Following the objective, every episode was
expanded once rather than extending only attractive examples. Effective windows became
48h before / 24h after for a one-hour anchor, 144h / 96h for four hours, and 14d / 8d
for eight hours, plus twenty-four warm-up hours. The final per-episode audit checked
318,960 rows with zero duplicates or gaps. Overlapping DOGE windows mean this count is
not a count of unique stored minutes. No second expansion is allowed inside this pass.

Seven expanded windows still have a boundary warning, primarily because volume or
volatility was elevated again days later; none remained inside the zone at the final
right edge. Immediate sixty-minute measurements remain usable. The longest source-
scale checkpoints are descriptive and must be read cautiously because expanding again
would start chasing later, potentially unrelated market episodes.

#### G3G exact level and cluster information retained

The parent object was the exact causal one-hour LVN centre, frozen half-width, ATR,
thinness score, source-open time, source-availability time, persistence, prior contacts,
and approach side. Every nearby causal peer retained its family, name, column,
representation, timeframe, price, width, availability, age, dependency group, contact
status, connected-component identity, and whether it belonged to the primary contacted
cluster.

The primary peers came from price-average mechanisms, rolling prior-price extremes,
and round-number mechanisms on one-hour, four-hour, and eight-hour data. These were not
treated as interchangeable lines. Established episodes had a median of six peer lines,
three dependency groups, and three peer timeframes; meme episodes had a median of four
peer lines, two dependency groups, and 2.5 timeframes. Established clusters contained
more round-number companions. Memes nevertheless produced the larger immediate
movement and activity in this sample, so raw line count is not evidence that a cluster
is stronger.

#### G3G complete generic-indicator and cross inventory

Every technical value used the last fully completed bar before the actual or pseudo
contact. The same compact calculation set was evaluated independently on `1m`, `5m`,
`15m`, `1h`, `4h`, and `8h`. The twenty-nine ordinary values on each timeframe were:

1. one-, three-, and twelve-bar returns;
2. fourteen-bar realised return volatility;
3. volume divided by its twenty-bar mean and volume standardized over sixty bars;
4. five-, twenty-, and sixty-bar volume-weighted candle-body pressure;
5. RSI(14);
6. Bollinger(20,2) position and normalized width;
7. MACD(12,26,9) histogram divided by price;
8. distance from SMA(20), SMA(50), and SMA(200);
9. SMA(20)-minus-SMA(50) and SMA(50)-minus-SMA(200), divided by price;
10. distance from EMA(12), EMA(26), and EMA(50);
11. EMA(12)-minus-EMA(26), divided by price;
12. ATR(14) divided by price;
13. ADX(14) and positive-DI minus negative-DI;
14. CMF(20) and normalized twenty-bar OBV change; and
15. price position inside the preceding twenty- and fifty-bar ranges.

The following twenty-four cross recencies were also recorded separately on every one
of the six timeframes. A value means completed bars since the cross, not a pass/fail
score:

1. RSI crossing above and below 50;
2. RSI recovering above 30 and falling below 70;
3. close crossing above and below the upper Bollinger band;
4. close crossing below and back above the lower Bollinger band;
5. MACD line crossing above and below its signal;
6. close crossing above and below SMA(20);
7. close crossing above and below SMA(50);
8. close crossing above and below SMA(200);
9. SMA(20) crossing above and below SMA(50);
10. SMA(50) crossing above and below SMA(200);
11. EMA(12) crossing above and below EMA(26); and
12. positive DI crossing above and below negative DI.

Seven explicit cross-timeframe alignments counted how many of the six timeframes had:

- RSI above 50;
- a positive MACD histogram;
- close above SMA(20);
- EMA(12) above EMA(26);
- positive twenty-bar pressure;
- volume above its twenty-bar mean; and
- ADX at least 25.

This is 318 timeframe-specific indicator/cross inputs plus seven cross-timeframe
alignments. They were logged exhaustively, but they were not fed into a model or used
to tune a rule.

#### G3G other inputs and relationships logged

Eight one-minute pre-contact values were used only to find the same-state no-level
control: fifteen-, sixty-, and 240-minute return; sixty-minute realised volatility;
one-minute ATR(14) as a percentage of price; current volume relative to twenty minutes;
twenty-minute pressure; and position in the preceding 240-minute range.

Twenty corrected G3A state fields were retained for relationship screening: BTC
twenty-four-hour return; local ATR percentage, six-hour pressure, range ratio,
twenty-four-hour return, and volume ratio; peer connected-group count, contacted
connected-group count, contacted line count, contacted dependency-group count, daily
count, all dependency-group count, higher-timeframe count, and all peer-line count;
selected-level density within two ATR; source ATR percentage and volume ratio; top-ten
breadth; Volume Profile level persistence; and Volume Profile value-area width.

Twelve numeric cluster summaries were screened: parent thinness; peer-line count;
independent-group count; timeframe count; cluster span in ATR; median parent-to-peer
distance in ATR; price-average, rolling-extreme, and round-number counts; and one-hour,
four-hour, and eight-hour peer counts. Exact component strings remain in the event file
for nonnumeric inspection.

Ten timestamp-safe global BTC orderbook values were available from the historical
Bybit linear surface: source coverage; mean spread; latest microprice offset; mean,
latest, and within-hour slope of 25-basis-point pressure; latest 25-basis-point depth
imbalance; distance to BTC resistance and support zones; and BTC volatility-expansion
state. Source time had to precede the decision, age could not exceed two hours, and
coverage had to be at least 80% to be called usable. Eleven of twelve actual episodes
had usable coverage; ETH in March 2022 did not. These fields describe global BTC
liquidity context, not local orderbooks for the sampled altcoins.

News and wider global-market fields were not silently converted to quiet or zero.
Their validated historical overlap was not broad enough for a like-for-like twelve-
episode statement, so they remain for G3H rather than being manufactured here.

In total, the catalogue contains 375 explicit pre-contact numeric inputs. It is a
plain inventory of what was checked, not evidence that using hundreds of indicators
improves prediction.

#### G3G continuous outcomes and controls

At 1, 3, 5, 10, 15, 30, 60, and 120 minutes, plus anchor-scaled 4h through 96h
checkpoints where the frozen window allowed, the replay recorded:

- close displacement from the contact close, in raw up/down and through-positive
  orientation;
- maximum symmetric up, down, through, away, and absolute excursion;
- first symmetric up/down and through/away ordering;
- first far-edge breakout versus one-extra-half-width rejection ordering;
- movement beyond the breakout and rejection boundaries;
- volume before versus after contact;
- raw pressure change and pressure oriented toward through or away;
- realised-volatility change;
- fraction of later candles overlapping the zone; and
- close crossings of the exact zone centre.

The ten screened outcomes were absolute excursion, post/pre volume, through excursion,
away excursion, away-oriented pressure change, realised-volatility ratio, zone dwell,
absolute up/down first path, through/away first path, and breakout/rejection resolution.

Each episode received one deterministic same-pair no-level pseudo-contact strictly
before the real event. Candidates had the same sixty-minute trend sign, could not
overlap the actual event's future path, and were rejected if the candidate minute
touched any causally available selected level. Matching used the eight values listed
above and no future outcome. A control was called balanced only when mean robust
distance was at most 0.75 and its largest input distance at most 2.0. These are declared
research calipers, not tool-native scores. Five of six controls were usable in each
cohort; AVAX and TRUMP remained outside the calipers and were not used in the balanced
comparison.

The other comparators were the sign of the causal preceding sixty-minute return, the
retrospective majority direction, and the cohort's base reaction rate. No-level
comparisons used paired event identities. Later checkpoints from one episode were
never counted as independent events.

#### G3G implementation and measurement corrections

The following failures and repairs are part of the result rather than hidden setup
noise:

1. The first acquisition smoke used an unsupported minute-form timerange and failed
   before data download. Exact ten-digit UTC Unix-second boundaries replaced it.
2. G3A's peer-independence defect was repaired before the G3G sample was frozen.
3. The first technical execution stopped before outcomes because Pandas 3 no longer
   accepts `5m` as a resampling alias. Explicit `5min` and `15min` parser mappings are
   now tested.
4. The first expanded invocation passed the new window map to the wrong helper and
   stopped before completing an episode. The corrected call is regression-tested.
5. The expansion download wrapper returned a non-zero shell status after logging its
   final download. It was not trusted; the independent 318,960-row audit proved exact
   coverage with zero gaps and duplicates.
6. A pre-final path definition counted the near entry edge during the contact candle
   as an away threshold. Because the approach had already visited that edge, this
   created a false five-of-six away-first pattern in both groups. The result was
   discarded. Final direction starts after the contact candle from its close, and a
   rejection must travel one additional half-width beyond the entry edge.

#### G3G corrected results: reaction before direction

Across all six actual episodes in each group, four established events and four meme
events met the declared reaction label: 66.7% in each cohort. This is eight of twelve,
but it is the same diagnostic sample on which the observation was made and therefore
is not a 66.7% confirmed forecasting rate.

Ten episodes had a usable matched no-level control. Seven of those ten actual contacts
reacted versus three of ten controls. Within established coins, four of five paired
actual contacts reacted versus two of five controls. Within memes, three of five
reacted versus one of five controls. Both groups therefore showed a forty-percentage-
point diagnostic uplift, but the meme cell had only three reacting balanced actuals,
below the declared minimum of four. The reaction comparison passed its small diagnostic
rule for established coins and failed it for memes; the combined all-market claim is
not retained.

Continuous activity gives useful context. Across the ten balanced pairs, actual
contacts had a median sixty-minute excursion about 0.22 parent half-widths larger than
controls, and the median paired difference was about +0.66. Seven of ten actuals moved
more. Actual post/pre volume was about 0.54 higher by separate medians and +0.81 by the
median paired difference; seven of ten were higher. Actual realised-volatility ratio
was about 0.35 higher, again with seven of ten positive. These differences are small-
sample descriptive evidence, not estimates of a stable population effect.

The normal/meme difference matters. For established coins, actual excursion magnitude
was not stronger than control by separate medians and only two of five actuals were
larger, although actual volume was higher in three of five. For memes, actual excursion
was larger in all five balanced pairs, volume higher in four of five, and realised-
volatility expansion larger in all five. Meme actuals also spent a median 0.38 of the
hour overlapping the pseudo-zone versus 0.97 for controls and had zero median centre
crossings versus five. In this diagnostic, memes made larger, cleaner departures;
established coins were more mixed and revisited the area more.

Pressure did not provide a clean explanation. Away-oriented pressure became positive
in parts of both actual cohorts, but actual-minus-control pressure differences changed
sign across pairs and were positive in only four of ten. OHLCV candle pressure is
therefore logged as inconclusive, not described as buying/selling proof.

#### G3G corrected results: direction and level resolution

Once the contact-candle artifact was removed, established coins split evenly: three
first moved through and three first moved away. Memes first moved through in four and
away in two. There is no universal rejection rule.

The sign of the preceding sixty-minute return correctly predicted the next symmetric
direction in four of six established episodes and five of six meme episodes. The
retrospective majority baselines were four of six and three of six respectively. The
simple trend therefore added nothing beyond majority in established coins but looked
better in memes. Reaction plus correct simple-trend direction occurred in three of six
established events and four of six meme events: 50.0% and 66.7%, or seven of twelve
pooled. The meme number happens to exceed the programme's 65% target, but it is six
examples with no untouched confirmation and must not be reported as target attainment.

Exact zone geometry was also group-dependent. Established events produced four
rejections, no far-edge breakout, and two unresolved zones inside sixty minutes. Meme
events split three rejections and three breakouts. Among the ten balanced pairs,
actual zones rejected first in six versus three no-level pseudo-zones. However the
established rejection rate was the same as its controls; the difference came from
memes, where three of five actuals rejected and none of five controls did. Conversely,
three meme actuals did break out, so a meme level is not simply a resistance/support
rejection rule.

Rejection distance was more informative than the class alone. Actual price travelled
farther beyond the entry edge than control in nine of ten balanced pairs, including
four of five established and all five meme pairs. Breakout distance was not stronger.
This suggests the frozen areas may mark a sharper two-way resolution point, especially
for memes, while recent direction helps decide which side wins. It is a lead for a
larger frozen confirmation batch, not a direction rule.

At longer checkpoints, paths crossed and revisited the area increasingly often. This
supports separating the immediate reaction from eventual market direction: the level
can mark a short-lived response while the broader trend later carries price elsewhere.
Because several expanded windows still ended during elevated later activity and the
number of supported episodes falls at the longest horizons, no specific 24h-96h path
is promoted from this microscope sample.

#### G3G indicator, timeframe, cluster, and external-input interpretation

The 375 inputs were screened against ten continuous or ordinal outcomes in normal,
meme, and pooled views. This created 3,750 feature/outcome combinations per view and
11,250 recorded correlation rows. The complete screen is retained so nulls and
contradictions are not lost.

None is eligible as an indicator relationship. Each cohort contains only six episodes,
while a cross-cohort queue signal requires at least twelve independent rows per cohort
before correlation size and leave-one-out stability are even considered. The absolute
direction outcome is also largely tied to whether the frozen level was above or below
price, and the corrected through/away and breakout/rejection outcomes contain only a
few examples per class. Selecting an RSI, MACD, moving-average, Bollinger, ADX, volume,
pressure, cross-recency, cluster-shape, or orderbook rule from this screen would be
feature fishing.

Therefore the correct result is not that generic indicators had no effect. It is that
this small replay cannot distinguish their effects reliably. Every value and cross is
logged for reproduction and later hypothesis design; none changed an indicator, became
a strategy condition, or earned promotion.

#### G3G conclusion and queued follow-up

Classification: retain as a reaction-location lead and a method result; do not retain
a directional rule.

What the replay supports weakly is a two-part market account:

1. qualified thin-LVN mixed clusters were followed by the declared price/volume
   reaction more often than balanced no-level moments in this small sample; and
2. recent direction appeared more useful for the immediate post-contact path than a
   universal bounce/rejection assumption, especially in memes.

What it does not establish is that the exact LVN caused the move, that all such areas
react, that one technical indicator predicts resolution, that memes are generally
easier to trade, or that any 55%/65% target has been met.

One later breadth-reviewed branch is justified: freeze a substantially larger set of
corrected G3A episodes before opening their one-minute paths, use the corrected
post-contact-close geometry, retain normal and meme cohorts separately, and test the
already named simple recent-trend comparator plus level-resolution outcomes. The batch
must include more balanced no-level controls and should not add hundreds of technical
features unless a specific G3G or G3H result names a low-dimensional question. This
candidate waits for the joint Generation-3 review; it does not launch now.

#### G3G evidence files

- frozen sample and original coverage:
  `generation3_branches/g3g_thin_lvn_one_minute_replay/g3g_thin_lvn_1m_freeze_20260814a/`;
- final expanded analysis, summary, feature catalogue, control comparison, boundary
  audit, control-match audit, coverage audit, and aligned plot:
  `generation3_branches/g3g_thin_lvn_one_minute_replay/g3g_thin_lvn_1m_analysis_expanded_20260814b/`;
- bulky event rows, checkpoint paths, and full relationship screen:
  `D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation3_branches\g3g_thin_lvn_one_minute_replay\g3g_thin_lvn_1m_analysis_expanded_20260814b\`;
- the preliminary nonexpanded `g3g_thin_lvn_1m_analysis_20260814a` is boundary/control
  QA only and is superseded by the expanded corrected run; and
- one-minute Binance futures caches remain under `user_data/data/binance/futures` for
  the eleven selected pair files. No unrelated continuous one-minute archive was
  downloaded.

### G3H. Audit and condition on external context only where timestamps are real

Inventory actual overlap between retained level episodes and historical orderbook,
executed-flow, news, and broad/global-market sources. Do not turn missing observations
into zero. If one source has enough independent contacts, ask one bounded question:
does the same named level response differ between comparable source-quiet and
source-active conditions, and does the added source improve over OHLCV and market
state on identical rows? If coverage is short, biased, or not timestamp-safe, finish
this route as deferred-for-data and record the gap rather than manufacturing a model.

#### G3H frozen orderbook subtest before reading the relationship outcome

The first usable source is the Bybit BTC linear orderbook. This is broad BTC/crypto
context, not the local orderbook of ETH, SOL, a meme coin, or any other sampled asset.
The frozen question is deliberately one-dimensional: among corrected G3A event pairs
that already match the local OHLCV state, BTC OHLCV state, top-ten breadth, level
density, approach, period, level side, and surviving cluster geometry, does the event
that occurs during unusually large absolute BTC orderbook pressure show a stronger
reaction than the matched event that occurs during quiet pressure?

"Pressure" here means the mean difference between bid depth and ask depth within 25
basis points of BTC's current price. Its absolute value is used because this test asks
about reaction strength, not up/down direction. Active means the top third of that
absolute value relative to the preceding thirty days of usable source hours. Quiet
means the bottom third. The middle third is logged but is not used in the direct
active-versus-quiet comparison. At least seven days of earlier source history is
required before a label can exist. A missing, low-coverage, stale, or future-timestamped
row is unavailable and can never be called quiet.

The primary outcome is contact-hour volume relative to earlier volume. The predeclared
secondary descriptions are four-hour dwell in the area, four-hour centre crossings,
and forty-eight-hour volume. Higher contact and later volume mean more activity;
lower dwell and fewer crossings mean a cleaner departure. These outcomes do not say
which price direction won and are not profit measures.

Each result slice forbids exact event reuse and keeps every selected global event time
at least the outcome horizon away from every other selected time, even across coins.
Established coins excluding BTC and memes are the two decision views. The ten-coin
established view and BTC alone are descriptive checks. Development and validation are
reported separately, no coin may exceed 35% of a decision sample, and leave-one-coin-
out results are retained. As a same-row negative control, the outcome is also oriented
by the BTC pressure value from seven days earlier. A queue-worthy lead needs at least
30 independent pairs, five coins, ten pairs in both development and validation, the
expected median in both phases, at least 55% of comparisons in the expected direction,
a better result than stale context, and positive leave-one-coin-out medians in at least
80% of omissions. These are declared review rules, not native machine-learning scores,
and a pass would only queue frozen confirmation.

The source-readiness audit also found that executed-trade-flow history is absent.
Validated GDELT/GKG/global aggregate windows overlap only the older established-coin
development data and have no meme overlap. They will be inventoried but not used to
manufacture a normal-versus-meme test. Story-level news and silver news remain blocked.

#### G3H completed result: absolute BTC orderbook pressure did not repeat

The project source audit first checked the frozen confluence snapshot rather than
assuming that a non-null table meant every source was present. Bybit linear and inverse
orderbook state had long timestamp-safe windows from January 2023, while spot started in
April 2025. The recent article/news feed, ETF flow, Google Trends, and historical GKG
blocks were too sparse for this question. Executed-trade-flow history was not present.
The validated aggregate news windows overlap only older established-coin development
events: 2,394 actual matched events fall in a permitted GDELT window, 1,441 in a GKG
window, and 450 have both permitted GDELT timing and usable Bybit orderbook state. No
meme event falls in one of those validated historical news windows. News and global
context were therefore inventoried and honestly deferred rather than interpreted as
zero or quiet.

After the shuffled G3A placebo rows were excluded, the orderbook inventory contained
5,506 unique actual established-coin matched events, of which 3,508 had usable BTC
linear orderbook state. It contained 890 unique meme events, of which 792 were usable.
No source timestamp was later than its event. Orderbook coverage was at least 80%, the
source row was at most two hours old, and at least seven earlier days existed before an
active, middle, or quiet label was allowed.

No RSI, MACD, moving average, Bollinger, timeframe cross, or new calculated line was
selected from outcomes in G3H. The only new input was absolute BTC orderbook pressure
within 25 basis points. The underlying G3A pairs already held similar the local return,
ATR, range, volume and six-hour candle pressure; BTC return; top-ten breadth; VP value-
area width and persistence; selected-level density; exact one-hour LVN side and
approach; and the number, timeframes, and independent calculation groups of contacted
cluster members. This means G3H asked whether orderbook context explained differences
left over after those market, level, and cluster inputs were matched.

For established altcoins excluding BTC, 208 globally time-independent active-versus-
quiet comparisons survived across all nine coins. Development initially looked
positive: active pressure had 0.418 more contact-volume units by the median and 23 of
31 comparisons, or 74.2%, favoured active pressure. Validation reversed. The median
became -0.157 and only 81 of 177, or 45.8%, favoured active pressure. Across all 208,
the median was essentially zero (-0.001) and exactly 50.0% favoured active pressure.
The current value did beat the seven-day-stale orientation by about five percentage
points overall, but only five of nine leave-one-coin-out medians were positive and the
development/validation sign did not repeat. ADA, ETH, and SOL were slightly positive
in validation; the other six established alts were neutral or negative. This is not a
stable similar-coin relationship.

BTC alone supplied 37 comparisons. Its development cell had only three examples. In
validation, active absolute pressure had lower contact volume by a median of about
0.214, only 41.2% favoured active pressure, and the stale value did better. BTC is both
too small and contrary; it cannot support an all-market result.

Memes supplied 54 independent active-versus-quiet comparisons across all ten coins.
Development was close to chance: the median active-minus-quiet contact-volume
difference was +0.132 and 17 of 32, or 53.1%, favoured active pressure. Validation
reversed to -0.194 with only 9 of 22, or 40.9%, positive. Across all meme comparisons,
the median was -0.029, 48.1% favoured active pressure, and the seven-day-stale value
actually scored 55.6%. Only one of ten leave-one-meme-out medians stayed positive.
The few positive per-coin cells did not repeat between development and validation.

Secondary outcomes did not reveal a hidden cleaner reaction. For established alts,
active pressure produced the expected lower dwell in only 34.6% of comparisons, fewer
crossings in 33.7%, and higher forty-eight-hour volume in 46.8%. Memes were 33.3%,
24.1%, and 55.0% respectively; the forty-eight-hour meme cell had only twenty pairs and
did no better than stale context. Many dwell and crossing differences were exactly
zero. None reached the 55% lead reference, none repeated coherently between development
and validation, and neither market group approached the 65% main target.

G3H conclusion: unusually large absolute BTC orderbook pressure is not retained as a
condition that strengthens this thin-LVN/mixed-cluster reaction. The promising
established-coin development result was a clean example of why chronological validation
is needed: it disappeared and reversed later. Memes were weaker and stale context was
better. This does not show that orderbooks are useless. It shows that one global BTC
pressure-magnitude input did not explain altcoin or meme reaction strength. Signed
pressure aligned to a later directional question, local coin orderbooks, wall distance,
depletion, or executed flow remain different hypotheses, but they wait for a later
generation and real coverage. G3H is complete: the tested orderbook route is parked and
the other sources are deferred for data.

G3H evidence is under
`generation3_branches/g3h_external_context/g3h_source_readiness_20260814a/` and
`generation3_branches/g3h_external_context/g3h_btc_orderbook_pressure_20260814a/`.
The 1,770 selected detail rows across all views and outcomes are stored at
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation3_branches\g3h_external_context\g3h_btc_orderbook_pressure_20260814a\`.
The final integrity audit found zero future-source rows, zero exact event reuse, and
zero global-time-separation violations; the focused research suite passed 106 tests.

All eight routes report the established and frozen meme cohorts separately. BTC is
reported separately but can never provide sufficient support alone. DOGE remains a
bridge and is never double-counted across cohorts. A predeclared smart-contract
platform subgroup of ETH, BNB, SOL, ADA, AVAX, and TRX may support a narrower result
only when at least five members have adequate evidence and no one member dominates;
its rationale and membership are fixed here before G3 outcomes. No outcome-discovered
coin group may be promoted inside the generation.

The weekly-VWAP-centre acceptance idea, development-only HVN-and-breadth interaction,
and mirrored-cluster regime idea remain in the rolling queue as parked candidates.
They are not part of the frozen G3 batch because their complete G2 evidence was either
contradictory or too sparse. A new G3 result may justify them for Generation 4, but
cannot quietly activate them now.

## Generation-3 joint review and frozen Generation-4 routes

All eight frozen G3 routes are now terminal. G3A retained only a conditional thin-LVN
contact-volume lead. G3B was deferred without outcomes because exact room/obstacle
common support was too small. G3C parked the exact round-number claim while preserving
the broader meme active-region association. G3D retained fresh arrival versus occupancy
as the strongest behaviour distinction but did not finish exact density-location
attribution. G3E parked participation-origin levels. G3F parked exact generic lines and
crosses while preserving representation questions. G3G retained a small reaction-
location and recent-path lead for breadth confirmation. G3H parked absolute global BTC
orderbook pressure and deferred the sources without real overlap.

The combined interpretation is that causal areas can mark short-term activity or
resolution, but visible lines frequently sit inside clusters and often proxy for
arrival state, congestion, distribution stretch, or the recent path. Fresh arrival
from outside is more repeatable than already-inside occupancy. Immediate direction is
more plausibly supplied by the approach and recent market path than by a universal
bounce rule. Established coins and memes remain separate because established coins
supplied the narrow isolated density lead while memes were more cluster-conditioned and
made cleaner two-way departures in the small one-minute sample.

Five justified G4 routes were frozen together before any descendant outcome was opened:

1. a larger outcome-blind one-minute thin-LVN reaction and recent-path confirmation;
2. fresh-arrival versus exact density-location attribution on genuine common support;
3. continuous room, independent-mechanism congestion, and distribution-stretch
   geometry;
4. one low-dimensional FreqAI regression/ablation ladder for continuous reaction
   behaviour rather than profit; and
5. exact generic-line contact versus the broader Bollinger-stretch, MA-bundle-distance,
   and SMA(50) quiet-state representations suggested by G3F.

All five must finish, defer, park, or reject before their results can spawn Generation
5. The full review is
`generation3_review/g3_generation3_review.md`; the frozen request surface is
`generation3_review/g4_frozen_branch_batch.json`.

## G4A larger one-minute thin-LVN confirmation: first pass and validity repair

Status: the first 111-event pass completed, but it is not accepted as a confirmation
because a post-contact volume condition was inherited from G3G selection. A second,
outcome-unconditioned sample is required within G4A's existing iteration allowance.
The first pass remains recorded because its setup problem is important evidence about
how not to test reaction zones.

### What the first pass froze before opening reaction paths

The first pass selected sixty established-coin episodes and all fifty-one surviving
meme episodes from the G3G-qualified pool. Selection was deterministic and did not use
the signed one-minute future path. Events were at least forty-eight hours apart across
all pairs inside each cohort. Normal data covered all ten established pairs; meme data
covered the fixed ten most-traded meme pairs. DOGE did not duplicate the same event
across cohorts. BTC was retained in the data but reported separately.

The sample contained one-hour thin LVNs above and below price, always approached from
the corresponding outside side. Each LVN sat inside a causally available contacted
cluster containing at least two independent calculation mechanisms. The exact parent
centre, half-width, thinness, persistence, prior contacts, source time, approach, and
all peer component families, dependency groups, prices, widths, source timeframes, and
availability times were retained. Cluster anchors were one hour, four hours, eight
hours, or one day. Single levels and complete clusters therefore remained separately
visible; the LVN was not treated as interchangeable with its companions.

Only eight causal OHLCV state inputs were used to match an earlier same-pair no-level
minute:

1. return over the preceding fifteen completed minutes;
2. return over the preceding sixty completed minutes;
3. return over the preceding 240 completed minutes;
4. realized return volatility over the preceding sixty minutes;
5. one-minute ATR(14) divided by price;
6. last completed minute volume divided by its trailing twenty-minute mean;
7. twenty-minute volume-weighted candle-body pressure; and
8. position inside the preceding 240-minute high-low range.

The control had to be strictly earlier, have the same sign of sixty-minute movement,
touch no causally available selected level, possess a complete two-hour path on both
sides, stay at least four hours from every actual event and other selected control in
the cohort, and pass the inherited mean and maximum robust-distance limits of 0.75 and
2.0 to enter paired comparisons. These limits are declared research calipers, not
native FreqAI or machine-learning scores. Controls with the fewest alternatives were
assigned first, then the frozen selection hash broke ties. There were 110 assignments;
thirty-nine established and thirty-nine meme controls met the balance limits. One
normal TRX development event had no globally independent control and stayed only in
the unpaired actual-event description.

No RSI, MACD, Bollinger, moving-average, EMA/SMA cross, ADX, generic line, orderbook,
news, or global-market value was screened against these outcomes. G4A deliberately
did not reopen G3G's 375-input catalogue. The only directional call was the sign of the
causal preceding sixty-minute return. The retrospective majority direction was an
optimistic comparator, not a usable live input.

Immediate direction began with the first candle after the contact candle closed. The
fixed diagnostic reaction required both at least one parent-zone half-width of price
excursion and post/contact-hour mean volume at least 1.5 times the preceding sixty-
minute mean within one hour. Continuous excursion, volume, pressure, volatility, zone
overlap, centre crossings, through versus away, and breakout versus rejection were
kept so the binary label could not hide the underlying behaviour.

Initial targeted acquisition covered ninety-eight merged intervals and 2,600,280
one-minute rows. Freqtrade correctly refused to insert older internal gaps into an
existing Feather file without `--prepend`. Rather than download years of unrelated
history, each failed frozen interval was downloaded into an isolated D-drive folder,
checked minute by minute, and atomically merged into the approved Binance file. The
temporary copies were then removed. The independent audit passed all 2,600,280 rows
with zero duplicate minutes, gaps, or invalid OHLCV rows.

Thirty-nine episodes triggered the predeclared source-scale boundary audit. Only those
episodes received one doubled context window. Thirty-two exact missing edge runs,
413,816 minutes in total, were downloaded and merged by the same method. The expanded
audit then found exactly 1,939,140 of 1,939,140 required rows. Expansion did not change
the sample, controls, reaction definition, or primary one-hour path.

### First-pass descriptive normal, meme, and BTC results

Among established altcoins excluding BTC, unseen early and late validation contained
thirty-six actual events across nine coins and twenty-three balanced pairs. Actual
contacts met the binary reaction label in 60.9% of those paired cases versus 26.1% of
no-level controls, an apparent uplift of 34.8 percentage points. The paired median
actual-minus-control differences were +1.04 zone half-widths of excursion, +0.68 in
post/pre volume ratio, and +0.15 in realized-volatility ratio. Actuals spent 0.18 less
of the hour overlapping the pseudo-zone by the median, while centre crossings had a
zero median difference. The apparent reaction uplift stayed positive in early
validation (+20.0 points) and late validation (+46.2 points).

The sixty-minute recent-path call got the raw first direction right in 62.9% of the
thirty-five callable established-alt validation events. The retrospective majority
would score 57.1%, so recent path was 5.7 points better. Reaction and correct direction
occurred together in eighteen of thirty-six events, or 50.0%. Successes covered all
nine altcoins and no coin contributed more than 16.7%, but the joint rate was below the
declared 55% lead reference and 65% main target. First movement was through in twenty-
two cases and away in thirteen. Far-edge breakouts and extra-half-width rejections
split sixteen to sixteen, with four unresolved or tied. There was no universal bounce
or breakout rule.

Meme validation contained twenty-eight actual events across all ten memes and eighteen
balanced pairs. Actual contacts met the binary label in 72.2% of paired cases versus
11.1% of controls, an apparent uplift of 61.1 percentage points. Paired median
differences were +0.95 half-widths of excursion, +1.84 in post/pre volume ratio, and
+0.36 in realized-volatility ratio. Actuals spent 0.38 less of the hour overlapping
the zone and crossed its centre three fewer times by the median. Apparent uplift was
positive in early validation (+55.6 points) and late validation (+66.7 points).

Recent path got meme direction right in 63.0% of twenty-seven callable validation
events versus a 55.6% retrospective majority, an advantage of 7.4 points. Reaction and
correct direction occurred together in fifteen of twenty-eight, or 53.6%. One more
success would have crossed 55%, but the frozen result did not. Successes covered eight
memes and no coin contributed more than 20.0%. First movement was through in nineteen
cases and away in eight. There were thirteen breakouts, eight rejections, and seven
unresolved or tied zones. Memes again made cleaner departures than established coins,
but not with a reliable universal direction.

BTC validation had only four actual events and two balanced controls, so it is
descriptive. Two actuals reacted, recent path got one of four directions right, and
the joint result was one of four. BTC cannot support or overturn either altcoin result.

### Why those attractive reaction numbers are not confirmation evidence

The post-run integrity review found that every selected episode inherited G3G's rule
that the completed contact hour must already have volume at least two times its causal
preceding twenty-four-hour median. The first-pass sample's smallest contact-hour volume
ratio was 2.018, its median was 3.219, and its maximum was 26.468. The later binary
reaction label then required elevated post-contact volume, while the strongest
continuous difference was also post/pre volume. This is outcome conditioning: the
sample was chosen partly because the volume reaction had already happened.

That conditioning also weakens the price-excursion comparison. High-volume hours often
move farther, so selecting future high volume can manufacture larger later excursion
even if the calculated level did not identify the activity beforehand. Earlier
same-state controls cannot repair this, because they were not selected for a future
high-volume outcome. The code executed the frozen first-pass question correctly, but
the inherited question itself was not a valid prospective confirmation question.

Therefore the honest first-pass classification is:

- the combined reaction-plus-direction package misses its frozen 55% retain rule;
- the apparent reaction-location uplift is selection-contaminated and cannot be
  promoted or called confirmed;
- recent-path direction is descriptive only because it was measured on the same
  contaminated event set;
- the exact level, cluster, volume, volatility, normal/meme, and BTC measurements stay
  logged rather than deleted; and
- G4A uses its second permitted iteration to draw from corrected G3A high-thinness,
  clear-approach, independent-cluster contacts without any post-contact volume,
  movement, pressure, profit, or direction filter.

The repair keeps the same level geometry, global independence, cohort separation,
control rules, reaction definition, continuous outcomes, recent-path comparator, 55%
lead reference, and 65% main target. Only the invalid future-volume selection condition
is removed. This is a root-cause correction, not a result-inspired threshold change.

First-pass evidence is under
`generation4_branches/g4a_thin_lvn_one_minute_breadth_confirmation/g4a_thin_lvn_1m_confirmation_20260814a/`;
bulky event and checkpoint rows are under the matching Generation-4 output folder on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\`.

### Corrected outcome-independent G4A run and full relationship log

Status: the root-cause repair completed on 14 August 2026. The sample was frozen at
19:38 UTC, the matching method and controls were frozen before outcomes at 19:46 UTC,
the fixed confirmation completed at 19:55 UTC, and the descriptive breadth review
completed at 20:10 UTC. The fixed overall classification is
`parked_larger_confirmation_did_not_repeat`. That classification remains authoritative
for the complete reaction-plus-direction package. A narrower reaction-location lead
inside the predeclared meme level-side split is recorded separately below; it does not
change the package classification.

#### What was repaired and what stayed unchanged

The corrected freeze rebuilt the causal G3A candidate surface instead of selecting
events that had already passed the later contact-hour volume outcome. No completed
contact-hour volume, later cluster contact, future price movement, future pressure,
future direction, or profit value was allowed to select an episode. The selected
one-hour LVN and every cluster companion had to be calculable before the first tested
one-minute contact.

The repair did not change the market question after seeing the first-pass numbers. It
kept the same:

1. one-hour Volume Profile LVN construction;
2. high-thinness selection idea;
3. clear approach from outside the zone;
4. independent cluster requirement;
5. normal and meme cohorts;
6. chronological development, early-validation, and late-validation periods;
7. forty-eight-hour global event separation inside each cohort;
8. earlier same-pair, same-state no-level controls;
9. one-hour reaction horizon;
10. binary reaction definition and continuous path measurements;
11. recent-sixty-minute direction comparator;
12. `55%` minimum joint lead reference and `65%` main programme target; and
13. BTC separation from the established-altcoin claim.

Only the invalid future-outcome selection rule was removed.

The corrected candidate builder found 5,593 eligible normal events and 1,157 eligible
meme events before global separation. Five hundred and seventy-two normal events and
114 meme events survived the forty-eight-hour cross-pair independence rule. The frozen
sample then took sixty normal and sixty meme episodes. Each cohort contained all ten
frozen pairs, twenty development episodes, twenty early-validation episodes, twenty
late-validation episodes, thirty LVNs above price, and thirty LVNs below price. DOGE
had no duplicate event shared between the two cohort interpretations.

The cluster anchor distribution was seventeen one-day, two one-hour, fourteen
four-hour, and twenty-seven eight-hour episodes for memes, and fifteen one-day, two
one-hour, fourteen four-hour, and twenty-nine eight-hour episodes for normal coins.
These are the timeframes of the causally connected cluster as a whole. The tested LVN
itself was always calculated from completed one-hour candles.

Fifty of the 120 selected events had completed parent contact-hour volume below the
old two-times cutoff, and thirty-three were below 1.5 times. The full sample ranged
from 0.670 to 11.657 times its earlier volume reference, with a median of 2.341. These
values were opened only after the corrected sample had been frozen. They are an audit
of the repaired selection, not another selector. All 120 records explicitly state
that later parent-hour peer contact was not used for selection.

#### One-minute data and control integrity

The corrected sample required 107 merged one-minute acquisition intervals containing
3,081,660 exact minutes. Sixty-four intervals were already complete. Forty-three
missing intervals containing 1,216,440 minutes were downloaded into isolated D-drive
working folders with four workers, checked, merged into the approved Binance futures
files, and independently audited. The audit found every expected minute, no duplicate
minute, no internal gap, and no invalid OHLCV row.

The fixed control search assigned 119 earlier same-pair no-level controls. Eighty-seven
met both robust-distance balance limits and entered paired comparisons. The balanced
sets were fourteen normal and eighteen meme development pairs, thirteen normal and
sixteen meme early-validation pairs, and eleven normal and twelve meme late-validation
pairs. One ORDI meme early-validation event had no globally independent candidate and
remained in the unpaired actual description. BTC supplied only two balanced validation
pairs.

Forty-three actual or control episodes reached a source-window boundary warning.
Selective expansion required thirty-six exact missing runs across eleven pairs and
411,960 additional minutes. The expanded audit found all 1,873,140 expected minutes
for the forty-three expanded windows. The event selection, control identities, main
one-hour horizon, and reaction rules did not change during expansion.

#### Complete input-role inventory for this test

One field can have a very different role from another. This log therefore separates
selection inputs, control-matching inputs, measured outcomes, and values merely carried
in the replay file.

The selected calculated level was a high-thinness one-hour Volume Profile LVN. An
`lvn_above` lay above the pre-contact price and was approached from below. An
`lvn_below` lay below the pre-contact price and was approached from above. This level
side was frozen and balanced before the one-minute outcomes were opened.

The LVN also had to be causally connected to at least two independent calculation
mechanisms. The companion mechanisms present in this sample were:

1. generic Bollinger boundaries, classed as a price-average family;
2. rolling or prior-range highs and lows, classed as a rolling-price-extreme family;
3. round-number levels; and
4. combinations of those mechanisms across one-hour, four-hour, eight-hour, and
   one-day source timeframes.

The eight causal values used to match each actual event to an earlier same-pair
no-level minute were:

1. return over the preceding fifteen completed one-minute candles;
2. return over the preceding sixty completed one-minute candles;
3. return over the preceding 240 completed one-minute candles;
4. realized return volatility over the preceding sixty minutes;
5. one-minute ATR(14) divided by price;
6. the last completed minute's volume divided by its trailing twenty-minute mean;
7. twenty-minute volume-weighted candle-body pressure; and
8. position inside the preceding 240-minute high-low range.

These eight values describe the path and activity before contact. They were not fitted
to the future outcome. The declared mean and maximum robust-distance calipers remained
0.75 and 2.0.

The fixed outputs examined after contact were:

1. whether the one-hour reaction label occurred;
2. maximum price excursion measured in parent-zone half-widths;
3. post-contact volume divided by pre-contact volume;
4. raw candle-body pressure change;
5. pressure change oriented toward moving through the zone;
6. pressure change oriented toward moving away from the zone;
7. absolute pressure-change magnitude;
8. post/pre realized-volatility ratio;
9. fraction of the hour spent overlapping the zone;
10. number of crossings of the level centre;
11. through-side excursion;
12. away-side excursion;
13. whether the first resolved path was through or away;
14. whether the first resolved zone outcome was breakout or rejection;
15. whether the sign of the preceding sixty-minute return predicted the first
    one-minute direction after contact;
16. direction accuracy among all callable events;
17. direction accuracy among events that actually met the reaction definition; and
18. reaction and correct direction occurring together in the same episode.

The replay file also carries causal RSI(14), Bollinger position, Bollinger width divided
by ATR, MACD histogram divided by ATR, EMA(50) gap divided by ATR, local returns, local
ATR, local volume and pressure state, BTC returns and volatility, top-ten breadth,
top-ten mean movement, return dispersion, common volume, nearby level density, source-
timeframe state, Volume Profile value-area width and persistence, and peer-level
counts. None of those carried fields was screened against the corrected G4A outcome.
They are not silently being credited for this result and remain available only for a
separately frozen later question.

No RSI cross, MACD cross, moving-average cross, EMA/SMA cross, Bollinger cross, ADX
cross, oscillator combination, or generic-indicator threshold was tested in G4A. No
orderbook, news, web, social-media, global-equity, rates, dollar, or other external
input was used. The only causal direction input was the sign of the preceding
sixty-minute return. The review did not reopen the earlier 375-input catalogue.

#### Corrected normal-coin result, excluding BTC

Early validation contained eighteen actual events and thirteen balanced pairs. The
paired actual reaction rate was 46.2% versus 15.4% for controls, an uplift of 30.8
percentage points. Median actual-minus-control differences were +0.75 zone
half-widths of excursion, +0.54 in post/pre volume ratio, and +0.26 in realized-
volatility ratio.

Late validation also contained eighteen actual events but only eleven balanced pairs.
Actual reactions were 27.3% versus 45.5% for controls, an uplift of minus 18.2 points.
Median excursion remained +0.39 half-width, but the volume-ratio difference changed
to -0.05 and the volatility-ratio difference changed to -0.12. The reaction advantage
therefore reversed rather than repeated.

Across the two validation periods, thirty-six actual events covered all nine normal
altcoins outside BTC and twenty-four balanced actual-control pairs. Eleven of the
thirty-six actual events reacted. In paired cases, actual reactions were 37.5% versus
29.2% for controls, only +8.3 points. There were six actual-only reactions, four
control-only reactions, three cases where both reacted, and eleven where neither
reacted.

The paired median actual-minus-control changes were:

1. +0.42 zone half-width of maximum excursion;
2. +0.03 in post/pre volume ratio, effectively little typical volume difference;
3. +0.10 in post/pre realized-volatility ratio;
4. +0.04 in pressure oriented through the zone;
5. -0.01 in absolute pressure-change magnitude;
6. -0.09 of the hour overlapping the zone; and
7. one fewer centre crossing.

Actual excursion was larger in 70.8% of balanced pairs, but actual volume was larger
in exactly 50.0%. This looks more like a weak displacement-location difference than a
repeatable abnormal-volume signal. It did not repeat chronologically.

The preceding-sixty-minute return sign predicted the immediate first direction in
60.6% of thirty-three callable validation events. The retrospective majority direction
was 63.6%, so recent trend did not beat it. Among the eleven callable events that
actually reacted, recent trend was correct in nine, or 81.8%, versus a 54.5%
retrospective reaction-subset majority. That is interesting but based on eleven
future-defined reactions and does not rescue the low reaction frequency. Reaction and
correct direction occurred together in nine of thirty-six events, or 25.0%, well
below both programme targets. Call coverage was 91.7%, with three direction ties or
abstentions.

At member level, three normal pairs had positive paired reaction uplift, four were
zero, and two were negative. Removing one normal pair at a time left pooled uplift
between zero and +15.0 points; eight of nine removals remained positive and one fell
to exactly zero. The effect was therefore not purely one coin, but it was small and
chronologically unstable.

Both normal level sides had the same pooled +8.3-point paired uplift: twelve balanced
`lvn_above` cases and twelve balanced `lvn_below` cases. Both were positive in early
validation and negative in late validation. There is no retained normal above-versus-
below relationship.

#### Corrected meme-coin result

Early validation contained twenty actual events and sixteen balanced pairs. Paired
actual reactions were 56.3% versus 12.5% for controls, an uplift of 43.8 points. Median
actual-minus-control differences were +0.83 half-width of excursion, +0.27 in volume
ratio, and +0.22 in volatility ratio.

Late validation contained twenty actual events and twelve balanced pairs. Actual and
control reactions were both 33.3%, so uplift was zero. Median excursion remained
+0.08 half-width, while the volume and volatility differences reversed to -0.17 and
-0.13. The overall meme effect did not reverse below zero, but the activity advantage
did not reproduce in the later period.

Across both validation periods, forty actual events covered all ten frozen meme pairs
and twenty-eight balanced actual-control pairs. Twenty of forty actual events reacted.
In paired cases, actual reactions were 46.4% versus 21.4% for controls, an uplift of
25.0 points. There were ten actual-only reactions, three control-only reactions, three
where both reacted, and twelve where neither reacted.

The paired median actual-minus-control changes were:

1. +0.54 zone half-width of maximum excursion;
2. +0.26 in post/pre volume ratio;
3. +0.14 in post/pre realized-volatility ratio;
4. +0.14 in pressure oriented through the zone;
5. -0.10 in absolute pressure-change magnitude;
6. -0.18 of the hour overlapping the zone; and
7. three fewer centre crossings.

Actual excursion was larger in 64.3% of balanced pairs, actual volume was larger in
57.1%, and actual volatility was larger in 60.7%. Pressure aligned with movement
through the zone was larger in 78.6%, but absolute pressure change was larger in only
25.0%. In plain terms, the corrected meme episodes did not create a generally stronger
pressure pulse. Their pressure was more often aligned with crossing the area, while
their absolute pressure movement was usually smaller than the matched control. This is
not evidence that pressure magnitude itself caused the reaction.

The recent-sixty-minute return sign predicted immediate direction in 54.1% of
thirty-seven callable events versus a 64.9% retrospective majority. Among the twenty
callable events that reacted, it was right in fifteen, or 75.0%, versus a 70.0%
reaction-subset majority. Reaction and correct direction occurred together in fifteen
of forty events, or 37.5%. Call coverage was 92.5%, with three abstentions. The
direction part remains inadequate despite the reaction difference.

Four individual meme pairs had positive paired reaction uplift, five were zero, and
one was negative. Removing any one of the ten meme pairs left overall pooled uplift
positive, between +16.7 and +32.0 points. This means the pooled effect was not produced
by one meme alone, but individual pair cells remain only two to four balanced events
and cannot establish ten separate coin edges.

#### Predeclared level-side relationship: the useful G4A lead

The balanced above/below split exposes an important asymmetry that the pooled meme
result hides.

For `lvn_above`, twenty validation events and fifteen balanced pairs produced actual
reactions of 26.7% versus 33.3% for controls, an uplift of -6.7 points. Early uplift
was +22.2 points, but late uplift reversed to -50.0 points. Median volume and
volatility differences were -0.11 and approximately zero. This side is unstable or
null in the corrected sample.

For `lvn_below`, twenty validation events and thirteen balanced pairs produced actual
reactions of 69.2% versus 7.7% for controls, an uplift of +61.5 points. The unpaired
actual rate was thirteen of twenty, or 65.0%. Early paired uplift was +71.4 points and
late paired uplift was +50.0 points. Eight balanced cases reacted only at the actual
LVN, none reacted only at the control, one reacted in both, and four reacted in
neither. This is the first corrected G4A reaction-location relationship that repeats
with the same sign in both untouched periods.

For meme `lvn_below` contacts, the paired median actual-minus-control differences were:

1. +0.85 zone half-width of maximum excursion;
2. +0.95 in post/pre volume ratio;
3. +0.65 in post/pre realized-volatility ratio;
4. +0.07 in pressure oriented through the zone;
5. -0.07 in absolute pressure-change magnitude;
6. -0.12 of the hour overlapping the zone; and
7. three fewer centre crossings.

Actual values exceeded controls in 69.2% of excursion pairs, 76.9% of volume pairs,
76.9% of volatility pairs, and 69.2% of through-oriented pressure pairs. Absolute
pressure change exceeded control in only 23.1%. The ordinary-language pattern is:
when these memes came down from above into a causally known thin one-hour Volume
Profile area that overlapped independent generic levels, they more often produced
higher volume, higher volatility, a larger displacement, less lingering, and fewer
back-and-forth centre crossings than an earlier same-market minute with the same
recent path but no selected level. This describes a reaction area. It does not say
whether price should bounce upward or continue downward.

The twenty `lvn_below` events covered nine meme pairs; PEPE had no validation event on
this side. Eight represented pairs had at least one balanced control. Five had
positive pair-level uplift, three were neutral, none were negative, and DOGE lacked a
balanced pair on this side. Removing each represented pair in turn left the pooled
paired uplift between +50.0 and +66.7 points. SHIB contributed three of the eight
actual-only reactions, but removing SHIB still left +50.0 points. The sign is therefore
not a single-pair accident, although thirteen balanced pairs remain a modest sample.

First resolved movement at meme `lvn_below` contacts was through in thirteen cases and
away in five, with two unresolved or tied. Eight were classed as breakouts and six as
rejections, so reaction did not equal one universal chart pattern. Recent trend was
correct on eleven of the thirteen reacting events, or 84.6%, but the retrospective
majority among reacting events was also 84.6%. The simple direction rule therefore
added nothing beyond the dominant outcome inside this small observed subset. Joint
reaction and correct direction occurred in eleven of twenty events, exactly 55.0%.
That reaches the user's minimum numerical floor but does not beat the relevant
direction comparator and is not a retained direction edge.

The honest classification of this subset is a queued reaction-location lead specific
to the current meme cohort, approach side, and level construction. It is not a broad
crypto claim, not a directional indication, not an entry, and not a promotion. It may
receive a predeclared replication/attribution batch only after all frozen G4 routes
finish and the Generation-4 review freezes Generation 5 together.

#### Cluster-family and indicator-combination relationships examined

The review kept the exact companion mechanisms instead of replacing them with an
opaque cluster score. The following family combinations occurred in validation:

1. Bollinger plus prior-range;
2. Bollinger plus prior-range plus round number;
3. Bollinger plus round number; and
4. prior-range plus round number.

For normal altcoins, Bollinger plus prior-range had fourteen balanced pairs and +14.3
points pooled reaction uplift; adding round numbers had ten balanced pairs and zero
uplift. Both normal combinations were positive in early validation and negative in
late validation. The two remaining family combinations had no balanced normal controls.
There is no retained normal cluster-family relationship.

For memes, Bollinger plus prior-range had sixteen balanced pairs and +18.8 points
pooled uplift. Bollinger plus prior-range plus round number had eight balanced pairs
and +37.5 points. Prior-range plus round number had three balanced pairs and +33.3
points, while Bollinger plus round number had one balanced pair and zero. The two
larger meme family cells were positive in early validation but fell to zero or became
too sparse in late validation. This test also did not remove one component at a time
while holding location and density fixed. It therefore cannot say that Bollinger,
prior-range, or round-number companionship caused the useful meme-below result.

Exact dependency-group count, peer-level count, number of price-average-family
components, number of rolling-extreme components, number of round-number components,
and one-hour/four-hour/eight-hour peer counts were also tabulated. No stable monotonic
relationship appeared in which more components consistently meant a stronger later
reaction. Those cells remain descriptive and do not justify a component-count
threshold.

#### Cross-timeframe relationships examined

All target LVNs came from the one-hour Volume Profile. Companion levels produced exact
cluster timeframe combinations of one hour alone, one plus four hours, one plus four
plus eight hours, one plus four plus eight hours plus one day, one plus eight hours,
four plus one day, four plus eight hours, or four plus eight hours plus one day where
available.

For normal altcoins, the largest one-plus-four-plus-eight-hour cell had twelve balanced
pairs and -8.3 points pooled uplift. It was +50.0 points in early validation and -37.5
points in late validation. The one-plus-four-hour cell had six balanced pairs and zero
pooled uplift. The full one-hour/four-hour/eight-hour/one-day cell had five balanced
pairs and +40.0 points pooled uplift, but only one balanced late event and no repeated
support. No normal timeframe combination is retained.

For memes, one-plus-four hours had nine balanced pairs and +11.1 points pooled uplift;
it fell from +16.7 early to zero late. One-plus-four-plus-eight hours had eight balanced
pairs and zero pooled uplift; it changed from +25.0 early to -25.0 late. The full one-
hour/four-hour/eight-hour/one-day combination had eight balanced pairs and +50.0 points
pooled uplift, remaining positive at +75.0 early and +25.0 late. This daily-connected
cell is interesting but contains only four balanced episodes per period and overlaps
the more general meme `lvn_below` lead. It is a queued attribution question, not proof
that a daily level takes precedence.

The meme `lvn_below` uplift stayed positive in its small one-hour, four-hour, eight-hour,
and one-day anchor cells. That argues against the pooled side effect being solely one
anchor timeframe. Sample sizes ranged from one to five balanced events per anchor,
however, so no timeframe precedence rule is justified. Higher timeframes remain
possible context, not automatic authority over lower-timeframe levels.

The review explicitly tabulated source timeframe, level side, causal anchor timeframe,
primary peer count, independent dependency-group count, cluster timeframe count,
exact dependency-group combination, exact timeframe combination, exact indicator-
family combination, counts of each family, peer counts by one/four/eight hours,
Bollinger presence, prior-range presence, round-number presence, higher-timeframe
presence, daily presence, pair crossed with level side, level side crossed with anchor
timeframe, level side crossed with exact timeframe combination, and level side crossed
with exact family combination. Every exact cell and continuous outcome is stored in
`g4a_confirmation_review_strata.csv`. Sparse cells are logged but are not treated as
discoveries.

#### BTC descriptive result

BTC supplied four validation events and two balanced controls. Two of four actuals
reacted, but both paired controls reacted, so paired uplift was -50.0 points. Recent
trend predicted three of four immediate directions and both reacting directions; joint
success was two of four. With only two balanced pairs, BTC cannot confirm or refute
the altcoin or meme relationships and remains a separate descriptive line.

#### G4A terminal interpretation and later queue

The invalid first pass showed why selecting on future volume can manufacture an
attractive result. The corrected run removes that problem and leaves a much narrower
picture:

1. the complete normal reaction-plus-direction package is parked;
2. the complete meme reaction-plus-direction package is parked;
3. normal one-hour thin-LVN contacts have a small, chronologically reversing reaction
   difference;
4. pooled meme contacts have a moderate reaction difference that weakens to zero in
   late validation;
5. meme `lvn_above` contacts are unstable or null;
6. meme `lvn_below` contacts retain a sizeable reaction-location difference in both
   validation periods and across pair removals;
7. that meme-below result does not yet isolate whether the LVN, the approach side, the
   companion cluster, or their combination supplies the information;
8. no tested recent-trend rule establishes direction beyond the relevant comparator;
9. no generic indicator cross or external context source was tested here; and
10. no strategy, entry, exit, profit, sizing, or trading action follows from G4A.

The Generation-5 queue should retain one explicitly worded question: on a larger,
outcome-independent meme sample, does arrival from above at a thin one-hour LVN still
produce abnormal price/volume/volatility displacement versus same-state no-level,
opposite-side LVN, shifted-level, near-miss, and cluster-component controls? That later
batch should preserve pair breadth, early/late chronology, exact cluster components,
level-density fairness, and side balance. It should first confirm the reaction and
only then test whether any causal lower-timeframe pressure or market-state input adds
direction beyond majority and recent trend. This question remains queued until G4B,
G4C, G4D, and G4E have terminal results and the full G4 review freezes the next batch.

Corrected compact evidence is under
`generation4_branches/g4a_thin_lvn_one_minute_breadth_confirmation/g4a_thin_lvn_1m_unconditioned_confirmation_20260814b/`.
The full immutable one-minute event replay remains under the matching G4A folder on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\`. The descriptive
review adds exact scope/phase, pair, leave-one-pair-out, and crossed-stratum CSVs plus a
machine-readable review summary without changing the frozen confirmation record.

## G4B fresh-arrival and exact density-location common-support audit

Status: completed and parked on 20 August 2026. None of the three frozen scopes had
fair support for the complete comparison ladder in both chronological validation
periods. G4B therefore did not open any new reaction values. Its terminal
classification is `parked_exact_location_controls_lack_common_support`.

### What question was tested

G3D had already shown a clear and repeated behaviour difference: a candle arriving at
a continuously known density area from outside usually carried more immediate range
and volume than a candle whose previous close was already inside that area. G4B asked
the narrower attribution question. After allowing for the fact that price had arrived
from outside, did the exact 168-hour confirmed-swing density location still add
information, or would an ordinary artificial boundary or another plausible level look
the same?

The three scopes were fixed before this audit:

1. an isolated 168-hour confirmed-swing density zone on the ten normal coins;
2. a 168-hour confirmed-swing density cluster on the ten normal coins; and
3. a 168-hour confirmed-swing density cluster on the fixed ten-meme cohort.

For each scope the intended evidence needed fresh arrivals, clean near misses,
already-inside periods, and repeat contacts. It also needed fair comparisons with a
same-width/same-approach artificial boundary, market-state-matched times without a real
level, density-coverage-matched no-level times, both one-ATR shifts, a causal shuffled
swing location, a current Volume Profile node, and a same-history prior high or low.
The frozen G2E no-level pool supplied the artificial boundary: it placed a pseudo-level
from the previous close using matched ATR distance, zone width, and approach side, then
excluded pseudo-level contacts that overlapped a real tested level.

### What “enough fair support” meant

G4B reused the G3D gate rather than inventing a result-dependent threshold. A
comparison needed at least fifty independent event pairs from at least five coins,
usable pre-event market-state balance, and the same support in both early and late
chronological validation. These are declared research rules, not native machine-
learning relevance scores. No AUC, profit, signed direction, or price-after-a-fixed-
candle success rule was involved.

The audit read only period, scope, control identity, independent-pair count, coin count,
coverage status, and pre-event state-balance metadata. It explicitly requested no
actual, control, difference, or named reaction-value column. Earlier G3D files contain
reaction results, but G4B did not use those values to decide which comparisons were
allowed to proceed.

### Normal isolated density-zone result

The raw event inventory was large enough for all four states. Clean near misses had
150 timestamps from all ten coins in early validation and 117 from all ten in late
validation. The problem appeared when fresh arrivals and near misses had to be made
genuinely comparable. Only 43 independent pairs from seven coins survived in early
validation and 24 pairs from four coins survived in late validation. Both are below the
declared fifty-pair/five-coin gate.

The same-state artificial/no-level boundary, both one-ATR shifts, and shuffled swing
location were fair in both periods. The density-coverage no-level comparison, current
VP nodes, and prior high/low comparisons had enough rows in several cells but their
starting market and level-density conditions were too different. More rows do not fix
that imbalance. This scope therefore cannot distinguish the exact density location
from generic arrival using the complete frozen control set.

### Normal density-cluster result

This was the only scope with fair matched support for all three alternative event
states in both validation periods. The clean-near-miss comparison retained 238
independent pairs from all ten coins early and 181 from all ten late. Already-inside and
repeat-contact comparisons also passed.

However, only the core same-state artificial/no-level comparison had usable starting-
condition balance. The density-coverage no-level, both shifted locations, shuffled
swing location, VP nodes, and prior extremes all remained materially different from
the real fresh-arrival cluster before the reaction window began. Their row counts were
often large, but their pre-event state imbalance was not. The cluster can support the
statement that fresh arrival differs from occupancy and near miss; it cannot fairly
attribute that difference to this exact calculated cluster location.

### Meme density-cluster result

The raw inventory again contained all four states, including 82 clean near misses from
all ten memes early and 79 from all ten late. Fair matching reduced those near misses
to only five independent pairs from one meme early and eighteen from three memes late.
The clean-near-miss comparison therefore failed before any reaction value could be
considered.

The same-state artificial/no-level comparison was usable. Density-matched no-level,
shifted, shuffled, VP-node, and prior-extreme comparisons were state-imbalanced, and
some VP and prior-extreme cells were also sparse. The current meme evidence remains a
cluster-conditioned fresh-arrival-versus-occupancy description, not exact density-
location attribution.

### G4B conclusion and what remains valid

G4B does not show that a confirmed-swing density location has no market effect. It
shows that the available rows cannot answer that causal-location question fairly under
the frozen controls. Rejecting or promoting the level from imbalanced comparisons
would both be unjustified.

The earlier, simpler finding remains valid as a market-behaviour lead: fresh approaches
from outside tend to be more active than price already accepted inside the same broad
area. What remains unproven is the stronger claim that the exact price emitted by the
density calculation supplies extra reaction information beyond arrival, activity,
nearby structure, and other possible boundaries.

No G4B descendant launches now. A different support-building representation may be
considered only at the joint Generation-4 review, after G4C, G4D, and G4E are terminal.
This prevents an unsuccessful attribution branch from consuming the rest of the frozen
generation.

Compact evidence is under
`generation4_branches/g4b_fresh_arrival_location_attribution/g4b_common_support_20260820a/`.
The run record stores source hashes, the exact metadata-only column list, all failed
control groups, the no-direction/no-profit boundary, and the explicit statement that
G4B opened no new reaction outcomes.

## G4C continuous room, congestion, and distribution-stretch test

Status: completed and parked on 20 August 2026. The outcome-blind preparation admitted
eight continuous geometry measurements for each cohort. The reaction screen then found
seven raw normal-altcoin relationships and six raw meme relationships. None survived
the complete frozen control ladder, so G4C retained no controlled continuous-geometry
finding. The result is terminal for this Generation-4 route; it does not launch a
Generation-5 test before G4D, G4E, and the joint review are complete.

### What question was tested

Earlier work suggested that named indicator lines may often be labels for broader
conditions. For example, price may react because it is stretched far from its recent
distribution, because many independently calculated zones occupy the same area, or
because there is unusually clear space before the next higher-timeframe zone. G4C
therefore asked whether these continuous conditions changed direction-neutral market
behaviour in a smooth low-to-middle-to-high order. It did not ask whether price went up
or down and did not use profit.

Every event remained a causally available one-hour contact with the thin side of the
unchanged one-hour Volume Profile LVN. The geometry was calculated using information
known before contact. The candidate measurements were:

1. forward empty room, in prior ATR, before the nearest causal four-hour, eight-hour,
   or daily zone in the approach direction;
2. counts of independent calculation groups ahead within 0.5, 1.0, and 2.0 ATR;
3. the fraction of a four-ATR local interval covered by causal higher-timeframe zones;
4. counts of same-role and opposite-role independent groups within one ATR;
5. absolute stretch within the preceding 168-hour price distribution;
6. absolute one-hour Bollinger stretch in rolling standard deviations; and
7. distance in ATR to the nearest causal SMA/EMA in the complete MA bundle.

The first third, middle third, and final third of each measurement were defined from
the development period before any validation reaction was opened. Tied values that
could not form three real bands were left unsupported rather than forced apart. This
prevented G4C from searching the validation outcomes for an attractive threshold.

### Outcome-blind support result

The normal run contained 7,069 eligible events. Eight of ten candidate measurements
had at least fifty events from at least five coins in every low, middle, and high band
in both validation periods. The unsupported normal measurements were the one-ATR
independent-group count and the opposite-role group count.

The fixed ten-meme run contained 1,360 eligible events. Eight measurements also passed
the band-support check. The unsupported meme measurements were the two-ATR independent-
group count and the opposite-role group count. Passing this step meant only that a fair
reaction screen could be attempted; it was not evidence that the measurement mattered.

### Raw reaction screen

For each supported measurement, G4C averaged each behaviour within each coin and band,
then used the median coin so that a high-event coin could not dominate. A raw lead
needed low, middle, and high values to have the same increasing or decreasing order in
both early and late validation. Eight direction-neutral behaviours were checked:
contact-hour volume and range, absolute one-hour excursion, four-hour realised
volatility relative to its preceding baseline, absolute pressure change, zone dwell,
centre-line crossings, and the first post-contact hour in which price closed outside
the zone.

Seven normal-altcoin cells and six meme cells passed that deliberately permissive raw
screen. They were leads only. Several already showed weak coin agreement: some pooled
orders were shared by only half of the individual coins. Opening controls at this point
was therefore necessary; reporting the pooled order alone would have overstated the
evidence.

### Frozen control ladder

Every raw cell faced the following controls, with both adjacent steps tested rather
than only comparing the endpoints:

1. **OHLCV and broad-market state:** adjacent geometry bands were matched within coin,
   validation period, LVN identity, and approach side using causal recent returns, ATR,
   range, volume, candle pressure, RSI, Bollinger state, MACD state, EMA distance, BTC
   return, and cohort breadth/activity/dispersion.
2. **Total density without geometry:** the same comparison additionally held constant
   the total nearby higher-timeframe line count, independent dependency-group count,
   and number of available MA lines.
3. **Matched geometry permutation:** the two members of each matched pair were
   deterministically swapped 1,024 times. The actual equal-coin effect had to exceed
   the one-sided 95% matched null in both adjacent steps and both validation periods.
4. **Same-state no-level times:** each real geometry band was matched to the frozen
   Generation-0 random-time LVN control pool. Those controls preserve approach distance
   and zone width and exclude the same LVN's real contacts nearby. The actual-minus-
   no-level difference still had to progress through both adjacent bands.
5. **Named LVN identity:** controlled effects were reviewed separately for `lvn_above`
   and `lvn_below`; one named side was not allowed to carry a generic geometry claim.
6. **Market separation:** normal alts, memes, and BTC were never pooled. BTC remained a
   one-coin descriptive check using the normal-alt development bands.

A normal adjacent comparison had ample support in most cells: commonly 70 to 457
matched event pairs from all nine normal altcoins. Meme state-matched comparisons were
much thinner, normally 9 to 31 pairs. The declared common-support requirement was
fifty matched pairs from at least five coins with usable starting-state balance. These
are project review rules, not native machine-learning scores.

### Normal-altcoin result

No normal relationship kept both adjacent steps in both periods through all controls.
The clearest near-pattern involved the number of independent higher-timeframe groups
within two ATR and contact volume. The raw result said volume fell as the number of
nearby independent groups rose. After OHLCV/broad-state matching, three of the four
period-by-adjacent-step checks passed. In early validation, both adjacent steps passed.
In late validation, the middle-versus-low step passed, while the high-versus-middle
step still had a positive equal-coin difference and six of nine agreeing coins but did
not exceed its matched permutation control (`p = 0.126`).

That near-pattern did not survive the independent no-level test. Three of four no-level-
adjusted steps were positive, but the late middle-versus-low step reversed: its equal-
coin adjusted difference was -0.296 and only one of nine coins agreed with the raw
ordering. This means the complete monotonic claim is not stable after comparable market
times without the LVN contact are considered. It remains a possible non-monotonic or
threshold-shaped question for later review, not a retained effect.

The forward-room/contact-volume cell also passed only two of four state/permutation
checks and three of four no-level-adjusted checks. Bollinger stretch versus departure
time passed one state/permutation step. The MA-distance and 168-hour distribution-
stretch range cells reversed or became null after state matching. The same-role-group
volatility cell worked only on one adjacent step in each of different periods. None is
a complete repeated ordinal relationship.

Named-side checks reached usable sample size for almost every normal cell, so failure
was not merely caused by splitting `lvn_above` from `lvn_below`. The effects themselves
did not remain positive through every side, period, and adjacent step.

### Meme result

The meme raw leads were higher-timeframe zone coverage versus volume and realised
volatility, one-ATR independent-group count versus volume, MA distance versus
crossings, and 168-hour distribution stretch versus crossings and absolute pressure
change. None had enough state-matched adjacent comparisons to meet the fifty-pair
control gate. Individual cells retained six to ten represented memes but only 9 to 31
matched event pairs, and several signs reversed between the two adjacent steps.

The larger random-time pool supplied roughly 54 to 92 same-state no-level matches per
band in many meme cells, but starting-state balance was often unacceptable and no cell
kept all four adjusted steps. Splitting by `lvn_above` and `lvn_below` left every meme
identity cell below its declared event-pair support floor. The correct conclusion is
not that continuous geometry never matters for memes. It is that the current 1,360-
event surface cannot support this high-dimensional matched attribution, and the raw
orders are too inconsistent to justify relaxing the rules after seeing them.

### BTC descriptive separation

BTC sometimes showed the same endpoint direction as a normal-alt raw lead, but the
middle band commonly broke the order or the order changed between periods. For example,
BTC contact volume across the two-ATR independent-group bands was low 4.18, middle
3.89, high 3.11 in early validation, but low 3.59, middle 4.58, high 3.45 in late
validation. BTC is one distinct market and received no pooled or promotion status.

### G4C conclusion and later queue

The useful conclusion is narrower than the raw screen. Continuous room, congestion,
and stretch values can line up with market activity in pooled summaries, but none of
the tested values presently gives a complete, repeated low-to-middle-to-high reaction
relationship beyond causal market state, total level density, matched permutations,
same-state no-level times, and both LVN identities.

One later question may be considered only during the joint Generation-4 review: does
very high independent-mechanism congestion suppress contact volume relative to low and
medium congestion as a threshold or plateau rather than a three-step monotonic rule?
That question is justified by the normal two-ATR near-pattern, but it must keep the
development-frozen bands, use a new untouched period or a newly frozen batch, retain
the no-level comparison that reversed, and avoid choosing a new cut point from these
outcomes. It remains queued, not active.

Compact preflight, raw-screen, and control summaries are under
`generation4_branches/g4c_continuous_active_region_geometry/` in the six run folders
ending in `preflight_normal10_20260820a`, `preflight_meme10_20260820a`,
`reaction_screen_normal10_20260820a`, `reaction_screen_meme10_20260820a`,
`controls_normal10_20260820a`, and `controls_meme10_20260820a`. Detailed event, band,
and matched-pair rows are in the matching G4C folders on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\`.

## Generation 4D FreqAI reaction-ablation result (2026-08-20)

### Terminal status

G4D is complete on all five frozen source surfaces. Thirty-five full FreqAI profiles
were trained: seven feature profiles on each of five surfaces. Every profile used the
same event rows, target definitions, chronological windows, model class, training
settings, and one-thread model limit. No common validation event was lost when the
profile predictions were intersected. Nine surface/target cells passed the deliberately
permissive frozen research-lead gate and six parked. The strong repeated finding is
about reaction volume. The forward excursion and dwell findings are smaller and need
the qualifications below. Nothing here predicts direction or establishes a trading
rule.

### Question and model ladder

G4D asked whether FreqAI estimates direction-neutral reaction behaviour more accurately
when it is told what level or zone price is interacting with, compared with seeing only
recent OHLCV, fixed indicators, BTC state, and cohort-wide state. It also asked whether
an identically shaped but shuffled level block could reproduce any improvement.

The seven profiles were:

1. a time-only rolling baseline, with no price, volume, level, or outcome inputs;
2. causal local OHLCV/fixed-indicator state plus BTC and cohort state;
3. current level identity and attributes without market state;
4. market state plus the current level and local geometry;
5. market state plus the same level/geometry columns reassigned to another event in the
   same pair and chronological period, with no row assigned to itself;
6. the current combined profile plus the predeclared multi-timeframe geometry block;
7. that profile plus only two predeclared low-order interactions.

The fixed continuous targets were contact-candle volume relative to prior typical
volume, absolute movement during the next hour in prior ATR, and the fraction of the
next four hourly closes that remained inside the level or zone. FreqAI used
`LightGBMRegressorMultiTarget`. Normal coins used rolling 730-day training windows and
two validation periods; memes used the shorter frozen history with rolling 160-day
training windows and their two validation periods. Development outcomes trained the
models and defined per-coin low/middle/high calibration bands; no development score was
reported as evidence.

### Exact source surfaces and support

The corrected G3A thin-LVN/mixed-cluster surface retained 8,271 independent normal
events and 1,247 independent meme events across development and validation. Its current
block described LVN side, causal thinness/value-area/persistence attributes, prior
distance, approach, nearby independent calculation groups, and higher-timeframe peer
counts.

The G3D confirmed-swing-density surface retained 2,291 normal isolated-zone events,
6,813 normal density-cluster events, and 1,271 meme density-cluster events across all
periods. It included first arrival, near miss, already-inside, and repeat-contact states
plus causal zone width, support fraction, prior distance, and nearby density/reference
geometry. The normal validation scoring surfaces contained 954 isolated-zone events and
3,166 density-cluster events including BTC. The meme density-cluster validation surface
contained 666 events. Some thin G3D rolling subsegments had no event for an individual
coin, which was reported rather than filled or replaced; all ten pairs still produced
common validation predictions.

Missing feature values were never changed to zero. An event entered the common mask
only when every required source, feature, and target existed and the source timestamp
was no later than the event. Retained events were more than four hours apart so their
future target paths did not overlap. Every profile predicted the exact same common
event keys, with no missing pair and no event excluded by the final profile
intersection.

### Technical faults found and repaired before evidence

The outcome-blind preflight found that G3D's single `approach_state` field belonged to a
matched event pair rather than uniquely to the actual event. The same actual event could
therefore inherit different approach labels from different controls. G4D removed that
field from G3D inputs instead of choosing one label. G3D also stored the one-hour
excursion target in its one-hour rows and the four-hour dwell target in its four-hour
rows. The repair joined the one-hour value by exact pair, base index, timestamp, side,
and event-state identity; unmatched rows were excluded. A structurally absent density-
overlap field was removed rather than zero-filled.

A literal constant-only FreqAI baseline then failed because FreqAI correctly removes
zero-variance features. The repaired baseline uses one causal calendar-time coordinate,
shared by all seven profiles. It allows a rolling time-only mean to train without
providing market or level information. A separate one-pair FreqAI smoke confirmed the
repair before the five full runs. The first failed full run remains preserved; all
reported evidence comes from the new `time_baseline_repair` runs.

### Frozen research-lead gate

A surface/target became a research lead only when the current combined profile had
lower mean absolute error than market-state-only in both validation periods, rank
agreement was no worse in both, at least five non-BTC normal coins or five memes had
lower error in each period, and the current block beat its shuffled block in both
periods. Calibration-band error, exact band accuracy, BTC, and leave-one-coin-out results
were reported as confirmation checks rather than turned into extra thresholds after
seeing results. Passing means “worth a narrower follow-up”, not “proven” or “trade it”.

### Strong repeated result: reaction volume

Current level/zone information made the clearest difference when estimating how unusual
the event candle's volume was:

- On normal thin-LVN mixed clusters, error improved by 11.45% in early validation and
  11.02% in late validation versus market-state-only. It improved by 11.33% and 11.44%
  versus shuffled level information. All nine non-BTC coins improved in both periods.
- On normal isolated density zones, error improved by 11.37% and 15.68% versus market
  state and by 9.86% and 17.05% versus shuffled information. Seven of nine non-BTC coins
  improved early and all nine improved late.
- On normal density clusters, error improved by 16.68% and 10.48% versus market state and
  by 18.00% and 10.49% versus shuffled information. Every non-BTC coin improved in both
  periods. BTC, kept separate, also improved by 12.40% and 18.07%.
- On meme density clusters, error improved by 8.76% and 11.76% versus market state and by
  10.03% and 11.90% versus shuffled information. Seven memes improved early and nine
  improved late. Removing any one meme left the pooled improvement positive in every
  leave-one-out check.
- The thin-LVN result did not transfer to the fixed meme cohort. It improved error by
  5.55% early but only 0.60% late, was worse than shuffled level information late, and
  only four memes improved in either period. That surface was correctly parked.

The normal thin-LVN, normal isolated-density, normal density-cluster, and meme density-
cluster volume results also stayed positive in every leave-one-coin-out aggregate. The
worst leave-one-out improvement was 8.03%, 9.94%, 9.48%, and 6.73%, respectively. For
the strongest normal and meme density-cluster surfaces, calibration-band error fell and
exact low/middle/high band accuracy rose in both periods. This is substantially stronger
than a pooled average carried by one coin.

The timing meaning is important. Arrival state and contacted-cluster classification are
known during or by completion of the event candle. Contact-volume estimation is
therefore a same-candle description or nowcast of a detected interaction, not a
pre-contact volume forecast. It still answers whether calculated level context helps
explain the observed activity, but a later live-prediction question must either use only
information known before the event candle or predict volume beginning with the next
candle.

### Smaller forward-reaction leads

Absolute next-hour excursion produced a smaller normal-market lead. Current information
reduced error by 5.93% then 2.91% on normal thin-LVN clusters, 2.99% then 1.67% on normal
isolated density zones, and 3.00% then 1.26% on normal density clusters. The result did
not repeat on meme thin-LVN or meme density-cluster surfaces. Rank correlations and
calibration generally improved, but their absolute values remained low, and the early
isolated-density advantage over shuffled information was only 0.06%. This is a normal-
market follow-up lead, not yet a reliable estimator.

Four-hour dwell passed the frozen gate only for normal and meme density clusters. The
normal-cluster error improvements were 1.19% and 2.54%; the meme improvements were 2.95%
and only 0.17%. Meme calibration-band error improved early but worsened late. This is a
weak, fragile lead despite its formal gate result. Thin-LVN dwell and isolated-density
dwell parked.

The forward targets begin with the next hourly candle, not the contact candle. Arrival
and event classification are therefore available before the excursion and dwell paths
start. Those targets have the appropriate causal ordering for a decision made after the
event candle has completed, although they still require new-period confirmation.

### What did not add value

The selected multi-timeframe block changed G3A error by less than about one percent and
did not repeat in sign. On all G3D surfaces its two reference-level counts were genuinely
zero for every retained event, so that profile was an audited no-op rather than evidence
against useful multi-timeframe geometry. A later MTF question needs a surface with real
reference-level coverage. The two predeclared interactions also produced only tiny,
inconsistent changes and are parked. More feature combinations are not justified from
this run.

Market-state-only sometimes improved volume over the time baseline, but often worsened
excursion or dwell. Level-only volume inputs were useful on normal thin-LVN and both
normal and meme density clusters. The combined and shuffled comparisons show that the
main volume result was not merely caused by giving the model more columns.

### BTC separation

BTC was not counted toward the five-coin gate. Its normal density-cluster volume result
was strong in both periods, while its isolated-density result improved early and was
flat/slightly worse late. BTC next-hour excursion differed by surface: it improved on
thin-LVN events, changed sign across density-cluster periods, and did not repeat for
isolated zones. This supports continuing to treat BTC as its own market rather than
assuming every normal-coin relationship transfers to it.

### Queued G4D follow-ups, not yet active

No descendant starts before G4E and the complete Generation-4 joint review. The current
results justify queuing the following narrow questions:

1. On the strong volume surfaces, ablate arrival state, support fraction, prior distance,
   causal nearby-zone count, and event-candle contacted-cluster fields one group at a
   time. This must distinguish pre-event geometry from information learned during the
   event candle.
2. Test a strictly pre-contact version of the volume question using only features known
   before the contact candle, and separately test next-candle volume after the level
   interaction. Do not call the completed same-candle result a forecast.
3. Recheck the smaller normal-only next-hour excursion lead in a newly frozen period and
   require a clearer margin over shuffled information, especially for isolated density
   zones.
4. Treat density-cluster dwell as weak until it repeats with improved calibration; the
   meme late-period 0.17% numeric gain is too small to rely on.
5. Build an MTF follow-up only on events with genuine nonzero higher-timeframe/reference-
   level coverage. The completed G3D MTF profile cannot answer that question.
6. Preserve the cohort difference: normal thin-LVN structure was useful for reaction
   volume, while the same representation on the meme cohort was not. Density clusters,
   not thin LVNs, supplied the meme lead.

Compact manifests, preflights, exact score tables, comparison tables, decisions, and
prediction audits are under
`generation4_branches/g4d_freqai_reaction_ablation/`. The five evidence run folders end
in `g3a_normal_20260820b_time_baseline_repair`,
`g3a_meme_20260820b_time_baseline_repair`,
`g3d_normal_isolated_20260820b_time_baseline_repair`,
`g3d_normal_cluster_20260820b_time_baseline_repair`, and
`g3d_meme_cluster_20260820b_time_baseline_repair`. The joint decision is in
`g4d_joint_review_20260820a/`. Detailed common event predictions and isolated FreqAI
artifacts are in the matching G4D folders on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\`.

## Generation 4E generic-line representation decomposition (2026-08-20)

### Terminal status

G4E is complete. It tested whether two broad market-state descriptions were more
repeatable than contact with their plotted indicator lines: absolute position within a
causal 168-hour price-rank distribution for Bollinger contacts, and prior-close
distance from the causal SMA/EMA bundle for moving-average contacts. It separately
revisited the isolated one-hour SMA(50) low-volume partial under explicit common
support in EMA distance and Bollinger position.

No broad representation and no exact-line residual survived the frozen control
ladder. Seventy-five representation/outcome cells and 150 exact-residual cells were
reviewed across the normal, meme, and descriptive BTC scopes. Zero were retained. The
terminal classifications are `unstable_or_null` for the tested representations,
`redundant_or_unattributed_exact_line` for the exact locations, and
`insufficient_coverage` for the isolated SMA(50) partial. No direction, profit, entry,
exit, or indicator edit follows.

### What was fixed before outcomes were opened

The first meme preflight correctly stopped because the meme manifest calls its periods
`meme_development`, `meme_validation_early`, and `meme_validation_late`, while the
normal manifest uses the shorter names. The repair reads the declared manifest roles
and maps one development plus two ordered chronological validations to common review
labels. It does not rename dates by assumption.

The next meme preflight exposed young-coin rows that lacked enough causal history for a
representation. Missing representation now means excluded from common support. It is
never converted to zero or ordinary market state. Ten focused tests, code lint, and the
full outcome-blind preflights passed after those repairs.

The final preflight contained 450,470 normal/BTC hourly representation rows and 95,880
meme rows. It opened no reaction values. Development-only thirds froze the low, middle,
and high representation bands separately for each source timeframe. Every exact-versus-
control comparison then required the same frozen band plus an absolute representation
caliper: 0.15 on the zero-to-one distribution-stretch scale or 0.25 prior ATR for MA
bundle distance. The SMA(50) question additionally required Bollinger-position
difference no more than 0.15 and EMA(50)-gap difference no more than 0.25 ATR.

### Outcome-blind support result

All six normal representation/timeframe questions had sufficient direct no-level
support for the predeclared one-hour and four-hour response rows, so their five fixed
direction-neutral outcomes were opened. Memes supported one-hour Bollinger stretch,
one-hour MA-bundle distance, and four-hour MA-bundle distance. Meme four-hour and
eight-hour Bollinger stretch and eight-hour MA-bundle distance were parked without
opening a G4E outcome interpretation.

The isolated one-hour SMA(50) partial failed before outcomes. Normal common support had
at most nineteen rows across seven or eight coins against an ATR-shifted control, only
three rows from two coins against the early no-level control, and zero late no-level
rows. Memes had zero early and one late no-level row. Shuffled and simple-prior
comparisons frequently had no shared representation row at all. Relaxing those
calipers after seeing this would recreate the state-imbalance problem that G4E was
designed to solve, so the partial is parked as insufficient coverage.

### Broader representation result

For each supported question, the exact-line-removed development rows first had to
define a genuinely monotonic low-middle-high outcome order. That order then had to
repeat on both the actual-contact surface and the no-line surface in both validation
periods. Both adjacent steps were matched by coin and inherited indicator identity on
causal returns, ATR, range, volume, pressure, RSI, volatility envelope width, MACD,
BTC, and cohort state. The direct proxy for the tested representation was omitted from
that match. Each matched step also faced 1,024 deterministic sign swaps.

The clearest near-pattern was one-hour Bollinger distribution stretch versus absolute
next-hour excursion. It still failed:

- For normal altcoins, the raw four surface/period checks passed only once, and only
  three of eight matched adjacent-step checks passed. In late validation, low-to-middle
  reversed on actual contacts (-0.096 prior ATR after orientation) and on no-line rows
  (-0.098). Middle-to-high remained positive, so the relationship was threshold-shaped
  or unstable rather than monotonic.
- For memes, three of four raw surface/period checks passed, but only two of eight
  matched adjacent steps passed. Late actual low-to-middle was positive, while
  middle-to-high reversed (-0.193); the no-line steps were small and did not beat their
  matched permutations.

Other cells were weaker. Normal eight-hour MA-bundle distance versus next-hour
excursion passed two of eight matched checks but none of its four raw surface/period
checks. Normal four-hour Bollinger stretch versus four-hour range passed one of eight
matched checks. Meme four-hour MA distance versus four-hour volume had a monotonic
development order but passed zero validation controls. Most remaining cells were not
even monotonic in development. BTC produced several descriptive development orders,
but one coin cannot satisfy a multi-market gate and none repeated through its controls.

### Exact plotted-line residual result

G4E reported the original state-matched line-control difference and then repeated it
only on rows sharing the broad representation. An exact residual had to retain its
development sign in early and late validation against no-level, both one-ATR shifts,
stale nominal history, shuffled indicator identity, and both simple-prior controls.
Independent cluster-versus-isolated component comparisons were reported separately so
several labels derived from one price average could not masquerade as independent
mechanisms.

No exact residual passed all fourteen required control-period checks. The largest
partial was normal eight-hour SMA contact versus contact-hour range: seven of fourteen
checks passed. Its direct no-level difference changed from -0.132 in early validation
to +0.011 late; shuffled identity was negative in both periods, while both ATR shifts
and stale history were positive. Simple-prior controls lacked balanced support. This is
mixed control behaviour, not a localized eight-hour SMA effect. Other exact cells
passed six or fewer checks. Meme artificial controls usually lost common support after
the representation restriction, so no meme exact-location claim could be attributed.

Cluster component results also changed sign through time or lacked complete support.
They do not show that a generic-indicator cluster adds information beyond its isolated
anchor.

### G4E interpretation and source evidence

The result does not prove that recent distribution position or average distance never
affects markets. It says these two fixed three-band encodings did not produce a stable,
monotonic, multi-coin reaction relationship on the G3F contact/control surface. It also
confirms that the earlier exact generic-line associations cannot be rescued by merely
renaming their broader technical state.

Final evidence is under
`generation4_branches/g4e_generic_level_representation_decomposition/` in:

- `g4e_preflight_normal_20260820c_missing_policy/`;
- `g4e_preflight_meme_20260820c_missing_policy/`;
- `g4e_analysis_normal_20260820b_groupby_fix/`;
- `g4e_analysis_meme_20260820b_groupby_fix/`; and
- `g4e_joint_review_20260820a/`.

The matching pair-level decomposition rows are under the same G4E path on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\`. Earlier G4E
attempts ending in `20260820a`, `period_roles`, or the failed normal analysis are
technical diagnostics, not evidence.

## Queued contextual and generic-indicator avenues

These are authorized rolling ideas, not completed evidence. The exact G4 subsets named
above are now frozen; anything not assigned to those five routes remains parked until a
later joint review supplies a reason to activate it.

- Generic moving averages and EMA/SMA crossings treated as explicit dynamic price
  levels rather than broad feature soup.
- Bollinger, Keltner, Donchian, and volatility-envelope boundaries, including genuine
  cross-timeframe convergence.
- Confirmed pivots, prior session/week/month levels, trendlines, channels, geometry,
  and rational clusters that preserve every component.
- Volume, range, candle pressure, wick behaviour, compression, expansion, RSI, MACD,
  ADX, and relative-strength state as pre-contact conditions.
- News and web-media attention, topic risk, sentiment or stress, with explicit source
  availability and timestamp safety.
- Global risk, rates, liquidity, equity, dollar, volatility, and crypto-wide state,
  using frozen snapshots and only periods with real coverage.
- Historical orderbook pressure, liquidity walls, depletion, and imbalance where the
  snapshot history overlaps enough independent level contacts.
- If a density-zone or other new calculated-level lead survives its frozen first
  control ladder, compare it with stale/time-shifted copies and near-misses that approach
  without contact. For a cluster lead, also compare the complete cluster with each
  component alone and with a matched-density cluster. This is a later queued attribution
  batch, not a change to the active frozen test or its pass rule.
- The queued density repair must retain contacted zone width, support fraction, rank,
  and component count; trim actual and control events to genuine common support in local
  zone coverage; and report first arrivals separately from candles already inside the
  zone. Any low/medium/high support bands must be defined from development data before
  validation. This will test whether a value emitted by the density indicator changes
  reaction strength, rather than merely testing whether a density zone exists.
- Normal-coin versus meme-coin contrasts, BTC separated where appropriate, and only
  rational subgroups retained when their members repeat the same behaviour.
- A possible meme subgroup suggested by differing participation histories must not be
  selected from these reaction outcomes. A later batch may define age, continuous-data
  history, liquidity, turnover, or attention/event concentration from information known
  before its test periods, freeze the members, and then ask whether the same acceptance
  behaviour repeats. Until then, per-coin differences remain descriptive rather than a
  claimed group edge.

## Source evidence

- `generation1_review/g1e_generation1_review.md`
- `generation1_review/g2_frozen_branch_batch.json`
- `generation2_branches/g2ab_localization/g2ab_horizon_meme_full_20260813d/`
- `generation2_branches/g2ab_localization/g2ab_missingcontrols_meme_full_20260814e/`
- `generation2_branches/g2b_volume_profile_roles/g2b_horizon_meme_full_20260813d/`
- `generation2_shared/cluster_reports/g2c_cluster_atlas_meme_full_20260814a/`
- `generation2_branches/g2c_mirrored_clusters/g2c_mirrored_meme_full_20260814a/`
- `generation2_branches/g2c_mirrored_clusters/g2c_mirrored_large9_full_20260814a/`
- `generation2_branches/g2d_anchored_vwap/g2d_anchored_vwap_meme_full_20260814a/`
- `generation2_branches/g2d_anchored_vwap/g2d_anchored_vwap_meme_declared_balance_20260814a/`
- `generation2_branches/g2d_anchored_vwap/g2d_anchored_vwap_normal10_full_20260814a/`
- `generation2_branches/g2d_anchored_vwap/g2d_anchored_vwap_normal10_declared_balance_20260814a/`
- `generation2_branches/g2e_density_zones/g2e_density_reaction_meme_full_20260814a/`
- `generation2_branches/g2e_density_zones/g2e_density_reaction_normal10_full_20260814a/`
- `generation2_branches/g2f_regime_modulation/g2f_regime_meme_full_20260814b/`
- `generation2_review/g3_frozen_branch_batch.json`
- `generation4_branches/g4b_fresh_arrival_location_attribution/g4b_common_support_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_preflight_normal10_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_preflight_meme10_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_reaction_screen_normal10_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_reaction_screen_meme10_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_controls_normal10_20260820a/`
- `generation4_branches/g4c_continuous_active_region_geometry/g4c_continuous_geometry_controls_meme10_20260820a/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_full_g3a_normal_20260820b_time_baseline_repair/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_full_g3a_meme_20260820b_time_baseline_repair/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_full_g3d_normal_isolated_20260820b_time_baseline_repair/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_full_g3d_normal_cluster_20260820b_time_baseline_repair/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_full_g3d_meme_cluster_20260820b_time_baseline_repair/`
- `generation4_branches/g4d_freqai_reaction_ablation/g4d_joint_review_20260820a/`
- `meme_cohort/meme_top10_selection_20260813.json`

This register is updated from compact verified summaries. Bulky raw event files stay
outside this document and are not manually copied into it.

## Generation 5 final branch review (2026-08-21)

### Terminal status and research boundary

Generation 5 is the fifth and final branch layer that the user authorized in advance.
All five frozen Generation 5 questions now have terminal decisions. One branch retained
a repeatable volume-prediction lead, two branches were tested and parked because their
effects did not repeat beyond the declared controls, and two branches were stopped
outcome-blind because a fair test could not be formed. No Generation 6 experiment was
started.

The table below uses ordinary meanings. `Retained` means that a direction-neutral
volume relationship survived the branch's frozen validation checks. `Parked` means
that the tested explanation did not survive or that the required comparison data did
not exist. It never means that a level should be traded.

| Branch | Question | Terminal decision | What was allowed to be concluded |
| --- | --- | --- | --- |
| G5A | Does the earlier meme thin-LVN-below reaction repeat on an unused independent sample? | Parked before outcomes: insufficient independent support | The replication could not be judged. It neither confirmed nor disproved the earlier lead. |
| G5B | Can causal information predict abnormal contact volume or next-candle volume beyond ordinary market state and false level controls? | Retained causal-volume leads on four positive surfaces | Selected pre-contact paths and calculated level geometry contain repeatable information about volume behaviour. No price direction was tested. |
| G5C | Does the normal-coin density-cluster model still estimate next-hour absolute price excursion in newly accumulated data? | Parked: not repeated beyond all controls | Average error improvements remained suggestive, but breadth, ranking, uncertainty, and calibration did not jointly repeat. |
| G5D | Does fresh-arrival path shape explain reactions at both real density zones and comparable synthetic boundaries? | Parked before outcomes: no common synthetic support | The current synthetic-boundary construction cannot fairly separate generic arrival behaviour from exact-location behaviour. |
| G5E | Do GDELT activity, global/macro state, or BTC orderbook pressure condition the level-volume relationship? | Eight tested cells parked; four macro cells parked before modelling | The available external context did not add a repeatable explanation, while the calculated level block retained substantial information. |

No branch predicted up versus down, optimized profit, selected an entry or exit, or
changed an indicator. The Generation 5 outcome therefore remains inside the
reaction-first boundary of Objective 02b.

### G5A: independent meme LVN-below replication

The frozen question was deliberately narrow. It asked whether memes arriving from
above at a causally known one-hour thin low-volume node below price would again show an
abnormal first-hour reaction, using events that did not reuse Generation 4's future
price paths.

There were 317 initially eligible validation events. Removing the exact prior events
left 297; removing events that shared their following 48-hour path with the earlier
test left 55; enforcing global independence left 14; and balancing the two untouched
periods produced only eight events from seven meme pairs, four in each period. The
frozen minimum was 30 events, with 60 preferred.

The branch therefore stopped with status
`parked_insufficient_independent_support`. Reaction outcomes, direction outcomes, and
profit were never opened. The correct interpretation is not that the meme LVN-below
lead failed. The correct interpretation is that the existing history cannot provide a
large enough genuinely unused replication sample after strict path separation. More
new data would be needed for the same question.

### G5B: causal volume clock and feature ablation

G5B separated three different information times so that information observed during a
reaction could not be mistaken for a forecast:

1. information available strictly before the contact candle was used to estimate the
   contact candle's volume;
2. the completed contact candle was then allowed when estimating the following
   candle's volume; and
3. same-contact-candle reconstruction was labelled a descriptive nowcast, not a
   forecast.

Each forecast had to beat market-state-only, shuffled-level, and stale-level profiles
in both chronological validation periods, retain broad coin support, remain positive
when each coin was removed in turn, and preserve useful ranking and calibration. The
four positive surfaces were the normal-coin density cluster, normal isolated density
zone, normal thin-LVN mixed cluster, and meme density cluster. The meme thin-LVN mixed
cluster was retained as a negative/descriptive comparator.

Eight forecast decisions survived: two forecast clocks on each of the four positive
surfaces. Compared with market state alone, the strictly pre-contact forecast reduced
equal-coin volume error by:

1. 15.46% early and 7.62% late for normal density clusters;
2. 10.40% early and 16.63% late for normal isolated density zones;
3. 10.73% early and 12.25% late for normal thin-LVN mixed clusters; and
4. 9.89% early and 8.34% late for meme density clusters.

All eight uncertainty intervals for those state comparisons excluded no improvement.
The normal surfaces covered nine non-BTC coins and the meme surface covered all ten
frozen meme coins. Seven to ten coins supported the sign in each period, depending on
the surface.

After the contact candle closed, adding its completed OHLCV behaviour reduced error in
estimating next-candle volume by 18.33% and 14.13% for normal density clusters, 15.41%
and 15.49% for normal isolated density zones, 20.30% and 20.21% for normal thin-LVN
mixed clusters, and 14.27% and 6.95% for meme density clusters. These comparisons also
survived the state, shuffled, and stale controls in both periods.

The feature removals make the result understandable. The recent approach path supplied
most of the useful strictly pre-contact information. Once contact had occurred, the
completed contact candle's own volume, range, body, and related OHLCV behaviour supplied
most of the extra information about the next candle. This is consistent with a market
approaching a calculated reaction area in a measurable state and the first contact
revealing whether participation is accelerating.

The negative meme thin-LVN comparator parked all three routes because shuffled or
stale representations could match or beat the real one. Three other same-candle
nowcasts were also parked. Only the normal density-cluster same-candle nowcast repeated,
and it remains descriptive because it uses information from the candle it describes.
This selectivity matters: the model did not simply declare every level surface useful.

`Eight retained leads` must not be read as eight independent trading edges. They are
two volume clocks repeated over four related density/LVN surfaces using one modelling
route family. At programme level this is one well-supported activity/volume mechanism
with several market-surface instances, not eight independent mechanisms.

### G5C: later-period normal-coin excursion confirmation

G5C used data accumulated after the earlier Generation 4 validation boundary. It asked
whether current density-cluster geometry still helped estimate the magnitude of the
largest next-hour price movement, regardless of direction. It did not ask whether
price would rise or fall.

The outcome-blind coverage check admitted only the normal confirmed-swing density
cluster. Fresh isolated-density and thin-LVN rows had no adequate non-BTC confirmation
support, so their outcomes were not judged. After control eligibility and independence
rules, the admitted cluster supplied 21 events from six non-BTC coins in the early
slice and 28 from six in the late slice. BTC supplied seven descriptive rows in each
slice, which was insufficient for a BTC claim. Four profiles were trained once before
the new slices and then used without retraining: state only, current level, shuffled
level, and stale level.

Against market state alone, the current-level profile reduced equal-coin error by
8.55% early and 8.15% late. This is suggestive but did not meet the frozen retention
rule:

1. the early 95% interval ran from -0.002 to +0.100 prior ATR and therefore included no
   improvement;
2. early rank quality became worse even though average error improved;
3. the late result was positive on only four of six coins, below the required five;
4. late calibration slope reversed below zero, meaning larger predictions no longer
   mapped sensibly to larger observed movements; and
5. BTC's seven-row result was unsupported and changed from small positive point
   estimates early to negative point estimates late.

The shuffled and stale comparisons also had positive average non-BTC improvements, but
they did not repair the breadth, ranking, and calibration failures. The branch is
therefore `parked_not_repeated_beyond_controls`, not declared completely null. The
later data still contains a possible small relationship, but it is too unstable and
sample-sensitive to rely on.

This does not contradict G5B. G5B predicted volume behaviour around contact. G5C tried
to estimate the size of the following price excursion. A level can help identify when
participation is likely to change without reliably telling us how far price will move.

### G5D: fresh-arrival path decomposition

G5D attempted to separate two explanations:

1. a generic arrival mechanism, where speed, starting distance, distance contraction,
   time outside the zone, and prior near-misses matter at any comparable boundary; and
2. an exact density-location mechanism, where those same arrivals behave differently
   specifically at a calculated density zone.

The outcome-blind preflight evaluated 46,648 candidate selection rows before global
independence and 36,398 afterwards. It produced 270 support rows and 45 frozen
scope/path-feature routes. Actual density-zone arrivals had two-period support for all
45 routes, but no route had enough comparable synthetic no-level arrivals in both
periods. Only some shuffled-path cells had support; zero routes could open either the
generic or exact-location outcome phase.

The branch stopped with integrity checks passing and no reaction values opened. This
is a control-representation failure, not evidence that approach path does or does not
matter. The actual path geometry may remain an outcome-blind input candidate, but it
cannot be interpreted until a synthetic or matched no-level construction shares real
common support.

### G5E: external context and residual level information

G5E formed twelve context/surface cells: four retained level surfaces crossed with
GDELT aggregate activity, global-market/macro state, and BTC Bybit orderbook pressure.
Missing source observations were kept unavailable and were never converted into an
ordinary quiet or zero state. Current context also faced a seven-day-stale version.

The macro snapshot initially appeared broad because it recorded many availability
rows, but its numeric daily values existed only on publication rows. A causal carry was
limited to 30 hours. That left only 227 genuinely model-ready hours, from 22 May to 23
June 2026, which could not provide the frozen 50-event/five-coin support in both
periods. All four macro cells were parked before outcome modelling.

GDELT and BTC orderbook pressure retained fair support on all four surfaces. Eight
cells were modelled with five profiles each, for 40 completed FreqAI profiles. Each
cell compared state only, level, context without level, level plus current context, and
level plus seven-day-stale context. Validation contacts were separated by four hours so
overlapping future paths could not masquerade as independent evidence.

Current external context did not improve contact-volume prediction repeatably. Across
the tested cells its incremental error change was generally small, changed sign, and
had uncertainty intervals that included no improvement. Current context also failed to
beat its seven-day-stale copy in both periods. Quiet-versus-shock differences did not
repeat. The strongest partial was isolated density plus GDELT in the early period,
where the level contribution appeared stronger in quiet conditions by 0.206 error
units; the late interval crossed zero, so it was parked rather than promoted.

The important control result points the other way: the calculated level block retained
substantial information after external context was present. Across the six normal-coin
GDELT/orderbook cells, adding the level block reduced error by approximately 8.42% to
15.80% in every period, and every interval excluded no improvement. Most cells had all
nine non-BTC coins supporting the sign; isolated density had seven or eight of nine.
For memes, density-cluster level information inside BTC orderbook context reduced error
by 17.05% early and 14.65% late, with eight of ten coins supporting both periods. The
GDELT meme level residual was 9.76% early and 11.58% late, although its early interval
included no improvement.

This means the tested GDELT activity and BTC-wide orderbook pressure neither created
nor explained away the volume relationship associated with these calculated areas.
It does not prove that external events are irrelevant. It says these particular
timestamp-safe representations did not supply additional repeatable information over
their actual overlap periods. Pair-local orderbooks, better-conditioned news, and a
longer genuine macro history remain different questions.

### Joint Generation 5 conclusion

Generation 5 strengthens one part of the working market theory and weakens several
overly broad interpretations.

The strengthened claim is:

> Around selected causal density and thin-LVN surfaces, contact volume and immediate
> next-candle volume are more predictable than they are from ordinary recent market
> state alone. The useful information is already present partly in the approach path,
> and the completed contact candle adds further information about the next volume
> response.

This repeated across nine established non-BTC markets and across the frozen ten-meme
cohort on the density-cluster surface. It survived shuffled and stale level controls
and did not disappear when GDELT or BTC orderbook context was added. This is credible
evidence for an `activity-only lead`: some calculated areas mark where market
participation changes in a structured way.

The evidence does not establish:

1. that the exact level mathematically causes the participation change;
2. that all LVNs, density zones, or clusters behave alike;
3. that the next price movement has a stable magnitude;
4. that price will bounce, break through, rise, or fall;
5. that news, macro state, or the BTC orderbook has been comprehensively ruled out;
6. that eight related model decisions are eight independent edges; or
7. that any result is profitable or ready for a trading rule.

The normal-coin later-period excursion test is a useful warning: predictable activity
does not automatically mean predictable price distance. G5A also shows that a strong
small historical subset can exhaust its independent evidence once overlapping future
paths are removed. G5D shows that a plausible market story must still be parked when
the control construction cannot compare like with like.

Objective 02b is therefore unresolved rather than complete. Generation 5 produced one
robust direction-neutral route family, but the programme does not yet have three
materially independent reaction-and-direction indications from at least two route
families. No Generation 5 branch made a direction call, so neither the 55% minimum nor
the 65% main joint reaction-and-direction target has been tested or reached here.

### Questions preserved for the approved breadth-first continuation

The following are evidence-led questions, not active tests:

1. confirm G5B's two volume clocks on a genuinely later untouched period, keeping the
   four surfaces and negative meme comparator frozen;
2. collect enough new meme history to repeat G5A without sharing any prior 48-hour
   outcome path;
3. redesign G5D's synthetic-boundary sampling before outcomes are opened, so real and
   no-level arrivals share path support rather than relaxing the gate after failure;
4. revisit macro conditioning only after a substantially longer span of real numeric
   observations exists;
5. test pair-local orderbook or better-conditioned event/news representations where
   honest multi-coin overlap exists; and
6. if a later independent reaction set qualifies, use a bounded one-minute microscope
   to ask whether lower-timeframe path and pressure distinguish immediate away-versus-
   through resolution beyond majority and recent-trend comparators.

On 21 August 2026 the user approved the complete breadth-first continuation recorded in
Objective 02b Section 15.5. Generation 6 will freeze a diverse seven-sibling portfolio
covering calculated price areas, timeframe relationships, OHLCV/indicator state,
cross-market/global state, orderbook, news/media/web context, and market-group
transfer. The questions above are candidates inside that portfolio, not permission for
one result to dominate it. Exact cells must still pass the outcome-blind coverage
crosswalk and enter one frozen batch before launch.

### Generation 5 source evidence

- Frozen plan: `generation4_review/g5_frozen_branch_batch.json`.
- G5A: `generation5_branches/g5a_meme_lvn_below_independent_replication/g5a_meme_lvn_below_freeze_20260820a/`.
- G5B: `generation5_branches/g5b_freqai_volume_clock_and_feature_ablation/g5b_joint_review_20260821a/` plus the five named full-run folders beside it. The corrected isolated-density evidence run ends in `20260821b`; the earlier `20260821a` folder is an incomplete technical attempt.
- G5C coverage: `generation5_branches/g5c_new_period_normal_excursion_confirmation/g5c_fresh_preflight_20260821a/`.
- G5C outcome: `generation5_branches/g5c_new_period_normal_excursion_confirmation/g5c_full_normal_density_cluster_fresh_20260821a/`.
- G5D: `generation5_branches/g5d_fresh_arrival_path_state_decomposition/g5d_path_preflight_20260821b/`.
- G5E coverage: `generation5_branches/g5e_external_context_common_support_and_conditioning/g5e_external_overlap_20260820a/`.
- G5E outcome: `generation5_branches/g5e_external_context_common_support_and_conditioning/g5e_context_full_20260821a/`.

Detailed event predictions remain in their routed Generation 5 run folders and the
matching bulky-output locations on `D:\FreqTradeStuffLargeData`. This register records
only the compact verified interpretation.

## Generation 6: broad source portfolio and controlled FreqAI review

Generation 6 deliberately tested a wide set of possible explanations before allowing
any one result to spawn more work. Its seven siblings covered calculated price areas,
relationships between level timeframes, recent OHLCV and standard indicators,
cross-market crypto state, historical order-book state, timestamp-safe GDELT context,
and frozen market groups. All seven direct screens and both complete FreqAI cohorts
were terminal before this joint review was opened.

### What was measured

The FreqAI work estimated five continuous, direction-neutral reactions:

1. next-hour volume divided by its typical pre-contact volume;
2. next-hour high-low range divided by its causal prior range baseline;
3. the largest absolute four-hour movement around the level in prior ATR units;
4. the size of the next-hour buying/selling-pressure change, with its sign removed; and
5. the fraction of the next four closes that remained inside the contacted zone.

These are not profit labels. A large absolute movement can be up or down. A change in
pressure magnitude does not say whether buyers or sellers win. No result below is an
entry, exit, directional call, or trading promotion.

Each supported source used the same comparison ladder on identical event timestamps:
market state alone, level geometry alone, the additional source alone, level plus the
source, and causal copies of the level or source that were at least 72 hours old. The
complete level-plus-source model was required to beat level alone, source alone, stale
source, and stale level in both chronological validation periods. Results were scored
per coin, combined with equal coin weight, and challenged with weekly-block
uncertainty. Contacts were separated by more than the four-hour maximum target so one
future price path could not appear as several independent events.

The normal and meme runs each completed 84 profiles and 72 controlled comparisons,
which produced 360 target-specific decisions per cohort. Every comparison used equal
prediction keys, no duplicate prediction rows were removed, all profile commands
finished, and the integrity records confirm that profit and future up/down direction
were not used.

### 1. Calculated location is a repeatable activity input

Adding the aggregate calculated-level description to recent market state improved all
four non-pressure reaction estimates in both cohorts and both validation periods. The
average equal-coin reductions in prediction error were:

1. **Next-hour volume:** 9.68% in both normal periods; 11.29% and 9.39% in the two meme
   periods.
2. **Next-hour range:** 5.83% and 4.46% for normal coins; 4.78% and 5.87% for memes.
3. **Largest absolute four-hour movement:** 5.01% and 5.60% for normal coins; 4.83%
   and 6.01% for memes.
4. **Four-hour dwell inside the zone:** 4.62% and 5.02% for normal coins; 4.41% and
   4.29% for memes.

All ten normal coins supported the first four normal comparisons in both periods. Meme
support was normally nine or ten of ten, and the movement result was positive on all
ten meme coins in both periods. Current level geometry also beat its 72-hour-old copy,
and level-only geometry beat stale level-only geometry. This makes calculated location
a credible reaction-estimation input rather than a line that merely labels whatever
the recent candles were already doing.

The pressure-change target is the exception. Calculated levels did not improve the
magnitude of the next-hour pressure change consistently, and all three meme level
checks failed. Pressure magnitude therefore remains an input candidate and descriptive
measurement, not a retained reaction target from this generation.

### 2. Prior highs and lows were the clearest direct level family

The direct matched-control screen kept individual level families visible. One-hour
generic prior-range references, such as causal prior session/week/month extremes,
showed the same positive behaviour sign in both cohorts for next-hour volume, next-hour
range, four-hour absolute movement, and pressure-change magnitude.

For next-hour volume, the median difference at one-hour prior-range levels was about
+0.33 baseline-volume units versus matched random times in both normal periods and
+0.31 to +0.34 in the meme periods. The same actual levels also beat symmetrically
shifted prices and 72-hour-stale level locations in both periods. Four-hour absolute
movement was about 0.17-0.18 ATR above matched random normal events and about
0.24-0.26 ATR above matched random meme events. These are repeated activity
differences, not directional forecasts.

A different, smaller pattern also matters: explicit prior four-hour Volume Profile
areas had slightly *lower* next-hour range than random-time, shifted-price, and stale
controls in both cohorts. The typical difference was only a few hundredths of a prior
range unit. A negative reaction difference is valid evidence; it may describe an
acceptance or dampening area rather than an expansion point. Generation 7 will test
whether current compression explains this instead of assuming every useful area must
increase movement.

### 3. Recent OHLCV state adds a second, smaller source of information

Adding the complete recent-market-state block to level geometry improved next-hour
volume error by 4.09% and 5.73% in the normal periods and 7.37% and 4.68% in the meme
periods. Next-hour range error improved by 3.52% and 3.82% for normal coins and 6.52%
and 2.34% for memes. The smaller late meme range result was supported by only six of
ten coins, so it is less secure than the volume result.

Within the separately tested blocks, local volume/pressure was the most consistent.
When added to level geometry it reduced next-hour volume error by 3.69% and 5.10% in
normal coins and 5.13% and 4.29% in memes. It passed all four component and stale
comparisons in both cohorts. Its range addition passed the complete standard for
normal coins, but the late meme increment fell to 0.44% with only five positive coins.
This supports a broad local-participation/volume mechanism, with range conditioning
stronger in established markets than in memes.

Volatility/compression state passed all four comparisons for normal next-hour volume
and range. Meme volume was one uncertainty check short of the same standard, although
its point improvements were 4.04% and 3.00%. Trend/momentum state passed all four
checks for normal range and meme volume, but did not give one common target that was
equally strong across both cohorts. These blocks are therefore distinct conditional
components, not additional universal edges.

### 4. BTC activity supplies a small broad-market increment

At the same calculated levels, adding timestamp-safe BTC activity reduced next-hour
volume error by 1.42% and 2.66% for normal coins and 2.48% and 1.86% for meme coins.
The four component/stale comparisons passed in both cohorts. The BTC-only state did
not beat the larger recent-market-state block, which is expected because that block
already contains overlapping information, but current BTC state did beat its stale
copy and added information beyond the level-only model.

For next-hour range, BTC state passed all four checks in normal coins and was one
uncertainty check short in memes. The point improvements were 1.57% and 1.66% for
normal coins and 4.51% and 0.96% for memes. This is consistent with the working theory
that the wider crypto market supplies some force while a calculated area supplies
location, but the effect is small and does not establish causation.

### 5. Cross-timeframe clusters need a more exact attribution test

Direct comparisons repeatedly found more next-hour volume, range, and four-hour
absolute movement when levels calculated by different mechanisms agreed across 4h,
8h, or 1d. The sign repeated in both cohorts. For example, different-mechanism 4h
agreement had about +0.31 to +0.34 baseline-volume units in normal periods and +0.21
to +0.26 in meme periods relative to the direct comparison group.

Those direct rows still carry an explicit width-matching warning. In the FreqAI ladder,
the broad timeframe blocks did not beat level-only, timeframe-only, stale-level, and
stale-timeframe models together for any full-cohort target. This does not justify
automatic higher-timeframe precedence. It says the present broad representation is
either redundant with level geometry or has not yet isolated genuine cross-timeframe
agreement from wider, older, or more frequently contacted zones.

Generation 7 therefore retains one exact different-mechanism cluster question with
component identity, dependency, distance, width, and contact-frequency controls. It
also retains an independent two-level-family cluster comparison. Single levels remain
equally valid and are required as component controls.

### 6. News, order-book, ETH, and cohort-relative context remain conditional or parked

Aggregate GDELT activity, historical BTC order-book state, ETH state, and
cohort-relative activity all produced repeated direct conditional differences in some
level families. None passed all four full-cohort FreqAI component/stale comparisons for
one common target in both cohorts. In most cases two checks passed because the current
combined model beat stale representations, while the comparisons against level-only
or source-only models showed that the apparent information was redundant.

The frozen group summaries identified narrower point-estimate patterns, including
established-altcoin volume under BTC-wide order-book context and meme dwell under ETH
state. These group checks did not use the full weekly-block uncertainty standard, so
they are labelled provisional group patterns rather than retained broad edges. The
late established-altcoin order-book addition beyond level geometry averaged only
0.02%, which is effectively no useful average increment even though five of eight
coins were slightly positive.

News and order-book are not being silently discarded. Generation 7 grants exactly two
weak-component exceptions:

1. order-book state may be tested only in combination with current BTC activity, on
   the theory that displayed liquidity matters primarily when BTC is active; and
2. aggregate GDELT may be tested only with the already retained local
   volume/pressure block, on the theory that information activity needs observable
   participation before it changes a reaction.

Both complete models must beat the stronger component alone, every immediately simpler
model, and stale inputs. Topic news, macro, live-news/web, and pair-local altcoin
order-book claims remain parked because their frozen coverage or representation did
not support this batch.

### Joint Generation 6 conclusion

Generation 6 supports the following loose market description:

> Calculated price areas contain repeatable information about where activity changes.
> Recent local participation and volatility describe how ready that market is to
> react, while BTC activity supplies a smaller wider-market condition. These inputs
> improve estimates of volume and movement size, but still do not say which direction
> price will take.

The strict cross-cohort intersection contains eight target rows, but it must not be
called eight independent edges. Four are different behaviours estimated from the same
calculated-location mechanism; two are the aggregate recent-state mechanism; one is
local volume/pressure; and one is BTC activity. The useful programme-level conclusion
is several components with small incremental value, not a finished predictive system.

The result also answers an important negative question. Simply observing that many
indicator, timeframe, news, or order-book conditions correlate with a reaction is not
enough. Once the level and source are each given their own model, much of that apparent
relationship is duplicate information. The next tests must show that selected
components work better together than either does alone.

### Frozen Generation 7 pairwise portfolio

The next complete batch contains ten sibling questions, frozen together before their
outcomes are opened:

1. local volume/pressure plus BTC activity at a calculated level;
2. local volume/pressure plus volatility/compression;
3. local volume/pressure plus direction-neutral trend/momentum strength;
4. genuine cross-timeframe agreement plus local participation;
5. prior session/week/month range references plus local participation;
6. prior four-hour Volume Profile areas plus volatility/compression, including the
   possibility of lower rather than higher activity;
7. two dependency-independent level families forming one cluster;
8. meme-coin local participation plus ETH activity;
9. historical BTC order-book state plus BTC activity, as a bounded weak-component
   exception; and
10. aggregate GDELT activity plus local participation, as the second and final bounded
    weak-component exception.

For every sibling, the complete interaction must beat level-only, each
level-plus-one-component model, both context components without the level, and current
versus stale copies on identical eligible timestamps. Every sibling must finish or be
honestly parked before a Generation 8 question is allowed to launch. Pressure-change
prediction, broad direction, profit, entries, and exits remain closed.

### Generation 6 evidence

- Joint review:
  `generation6_review/g6_joint_review_20260821a/g6_joint_review.json`.
- Source/target controls:
  `generation6_review/g6_joint_review_20260821a/g6_source_target_review.csv`.
- Cross-cohort classifications:
  `generation6_review/g6_joint_review_20260821a/g6_cross_cohort_review.csv`.
- Frozen group checks:
  `generation6_review/g6_joint_review_20260821a/g6_group_interaction_review.csv`.
- Frozen Generation 7 batch:
  `generation6_review/g7_frozen_pairwise_batch.json`.
- Normal FreqAI run:
  `generation6_branches/g6_freqai_source_ladders/g6_freqai_full_normal_20260821a/`.
- Meme FreqAI run:
  `generation6_branches/g6_freqai_source_ladders/g6_freqai_full_meme_20260821a/`.
- Seven direct screens:
  `generation6_branches/g6_direct_screen/g6_direct_full_20260821a/`.

Large prediction and family-slice artifacts remain in the corresponding Generation 6
folder under `D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones`.

## 2026-08-21 - Generation 7 Complete Pairwise Interaction Review

### What was tested

Generation 7 tested ten frozen questions about whether two conditions become more
useful together at a calculated price area. Nine questions applied to each eligible
cohort; the ETH question was meme-only and the historical BTC order-book question was
normal-market-only. The run therefore contained `18` branch/cohort cells, `144`
FreqAI profiles, and `126` complete-versus-control comparisons.

Every complete model was compared on exactly the same pair/timestamps with:

1. the calculated level alone;
2. the level plus the first condition;
3. the level plus the second condition;
4. both conditions without a current level;
5. a stale level plus both current conditions;
6. the current level plus a stale first condition; and
7. the current level plus a stale second condition.

A finding was called strict only when the complete model improved all seven
comparisons in both chronological validation periods, the weekly-block uncertainty
range stayed above no improvement, enough predeclared group members were positive,
and one coin did not dominate. This creates `14` required control-period checks per
target and group. Profit and signed future direction were not targets.

Before reaction outcomes were opened, the exact feature and stale-control support was
frozen. All `18` cells passed. The narrowest meme cell, prior four-hour Volume Profile
geometry under compression, still had `468` development contacts and `242`/`305`
validation contacts across all ten coins. A two-family cluster required zone overlap
or a gap no larger than `0.10 ATR`; merely appearing during the same hour did not count.

### The one strict complete interaction

For the full normal-coin cohort, the combination of:

- a current calculated price area;
- current local participation, described by relative volume, volume acceleration,
  absolute pressure, and pressure persistence; and
- current movement capacity, described by ATR/range and compression measures,

improved estimates of **next-hour volume** against every simpler and stale model in
both validation periods.

The complete model reduced equal-coin absolute error by about:

- `4.3%` and `6.0%` versus the calculated level alone;
- `0.76%` and `0.82%` versus level plus local participation, showing the small extra
  contribution from current volatility/compression;
- `2.5%` and `3.3%` versus level plus volatility/compression, showing the larger extra
  contribution from local participation;
- `10.2%` and `10.4%` versus both market-state blocks without a current level; and
- `10.4%` and `10.7%` versus a stale level with both current conditions.

The smallest improvements were under one percent. That is not a flaw or a universal
prediction claim: it is consistent with the programme theory that technical inputs
may supply small conditional information. Current location, participation, and
movement capacity each contributed something; none of the three alone explains the
complete result.

### Cross-cohort provisional patterns

Two branch/target relationships were point-positive across all seven controls and
both periods in both broad cohort views, but at least one uncertainty range still
included no improvement:

1. local participation plus volatility/compression for **next-hour range**; and
2. local participation plus direction-neutral trend/momentum strength for
   **next-hour volume**.

For the meme range result, every one of the `14` point comparisons was positive. The
complete model's mean error reduction ranged from about `0.5%` against the hardest
stale-component control to about `8.7%` against both context components without the
current level. Only `9/14` uncertainty checks were strict, so this is a coherent lead,
not confirmation.

The trend/momentum-volume result was provisional for the full meme cohort and for BTC
inside the normal run. That difference matters: it may describe a behaviour shared by
BTC and fast meme markets rather than a universal established-altcoin relationship.
It needs feature attribution and group-separated reporting before any stronger claim.

### Normal-cohort leads that did not transfer broadly to memes

Several normal-market relationships were positive across all controls but remained
uncertain:

- local participation plus BTC activity for next-hour volume and range;
- different-mechanism cross-timeframe agreement plus local participation for
  next-hour volume, with a narrower BTC four-hour-excursion pattern; and
- historical BTC order-book state plus BTC activity for next-hour volume in the
  predeclared smart-contract-platform group.

These are valid detailed-attribution questions because their complete models beat all
seven controls at the point-estimate level for at least one predeclared group. They
are not market-wide edges and are not evidence that higher timeframes, BTC, or the
order book always take precedence.

### What did not survive the complete interaction standard

The following complete interactions did not pass and will not receive
outcome-selected threshold repair in Generation 8:

- specific prior-range geometry plus local participation;
- prior four-hour Volume Profile geometry plus compression;
- two dependency-independent calculated level families as one explicit cluster;
- meme local participation plus ETH activity; and
- aggregate GDELT activity plus local participation.

The failed cluster test does not say that individual levels or all direct cluster
reactions are imaginary. It says that the tested two-family cluster representation
did not improve reaction-magnitude estimation beyond both components and the stale
controls. Similarly, the failed prior-range interaction does not erase the earlier
matched-control evidence that prior highs and lows coincide with unusual activity; it
means local participation did not robustly add the proposed extra interaction.

The two weak-component exceptions were therefore resolved honestly. Historical
order-book plus BTC activity remains only a provisional smart-contract-group lead.
Aggregate GDELT failed and is parked. No further weak news exception is carried into
Generation 8.

### Frozen Generation 8 attribution portfolio

The complete Generation 7 review froze eight detailed siblings together:

1. level strength, width, age, persistence, touch count, proximity, and timeframe
   attribution inside the retained participation/volatility interaction;
2. relative volume, volume acceleration, pressure, and persistence attribution;
3. ATR/range versus Bollinger/compression attribution;
4. persistent trend strength versus oscillator/momentum-magnitude attribution;
5. BTC 1h/4h/24h activity, relative volume, freshness, duration, and adjacent
   intensity-range attribution;
6. separate 4h/8h/1d incremental value plus opposing-obstacle and open-room geometry;
7. historical BTC order-book pressure, coverage, freshness, duration, and BTC-horizon
   attribution for the smart-contract-platform lead; and
8. first contact, occupancy, prior-contact age, and repeated-retest state across the
   retained interaction families.

Stable adjacent predictor ranges must be frozen from predictor distributions without
reading reactions. Every retained attribution must survive three deterministic model
seeds, immediately simpler representations, causal stale copies, and within-period
nonself shuffles. Cluster composition and GDELT are deliberately absent because their
Generation 7 parents failed. Direction, profit, entries, exits, and canonical
indicator edits remain closed.

### Generation 7 evidence

- Joint review:
  `generation7_review/g7_joint_review_20260821a/g7_joint_review.json`.
- All branch/target classifications:
  `generation7_review/g7_joint_review_20260821a/g7_joint_branch_target_review.csv`.
- Retained-control and uncertainty details:
  `generation7_review/g7_joint_review_20260821a/g7_retained_control_bottlenecks.csv`.
- Frozen Generation 8 batch:
  `generation7_review/g8_frozen_attribution_batch.json`.
- Normal FreqAI run:
  `generation7_branches/g7_freqai_pairwise_interactions/g7_freqai_full_normal_20260821a/`.
- Meme FreqAI run:
  `generation7_branches/g7_freqai_pairwise_interactions/g7_freqai_full_meme_20260821a/`.

Large feature caches, predictions, and level-family slices remain under the matching
Generation 7 folders on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones`.

## 2026-08-21 - Generation 8 Detailed Attribution And Three-Seed Review

### What was tested

Generation 8 completed all eight attribution questions frozen after Generation 7.
The purpose was to separate broad feature bundles into trader-readable pieces rather
than assume that every ingredient in a successful bundle mattered. The questions
covered:

1. calculated-level identity, prominence, width, proximity, timeframe composition,
   age, persistence, and prior contacts;
2. current relative volume, volume acceleration, absolute pressure, pressure
   persistence, and the time spent in each participation state;
3. absolute volatility, compression, and the duration of those states;
4. persistent trend strength, return acceleration, oscillator displacement, and the
   duration of those states, with every input kept direction-neutral;
5. BTC activity over one, four, and twenty-four hours, BTC relative volume, and state
   duration;
6. separate four-hour, eight-hour, and daily location information plus open room and
   nearby obstacles;
7. historical BTC order-book pressure, coverage/freshness, pressure regime, and
   persistence for the predeclared smart-contract-platform group; and
8. first contact, continued occupancy, retest state, and level/contact history across
   three retained parent contexts.

Every add-one test compared the current ingredient with the immediately simpler
model, a causal 72-hour-old copy, and a deterministic within-period nonself shuffle.
The full attribution models also had complete-versus-base and leave-one-part-out
controls. Normal and meme cohorts retained their predeclared coin groups and two
chronological validation periods. Profit, entry/exit logic, and future signed price
direction were absent.

### Outcome-blind construction and integrity

Before targets were opened, all `19` branch/cohort/surface cells passed the causal
support preflight. The cache reconstructed level contact lineage from the existing
one-hour, four-hour, eight-hour, and daily level files; separated first contact,
continued occupancy, first retest, and later retests; and reconstructed opposing-level
room geometry. Level identity starts a new causal segment only when the level moves by
more than its active half-width or returns after being unavailable. Stable low/middle/
high feature bands came from development predictors, not reaction outcomes.

The initial seed-42 runs completed `404` profiles and `385` comparisons. Only routes
whose current feature beat every frozen control on point estimates in both periods
were frozen for seed confirmation. This produced `138` normal and `62` meme profiles
for seeds `17` and `73`, covering `174` additional comparisons. Across all initial and
confirmation runs, `604/604` profiles and `559/559` comparisons completed with zero
duplicate prediction rows, zero unequal prediction-key comparisons, and no profit or
future-direction fields.

A relationship is strict only when all controls in both periods remained positive and
their weekly-block uncertainty stayed above no improvement at all three seeds. A
relationship is provisional when every point estimate stayed positive at all three
seeds but at least one uncertainty interval included no improvement. Any non-positive
control at either confirmation seed is a failed replication; it was not retuned.

### Strong result: current participation intensity and its duration forecast reaction volume

For normal coins near calculated price areas, two local-participation descriptions
strictly improved prediction of next-hour trading volume:

1. **Current relative volume.** This tells the model how unusual the current completed
   candle's volume is relative to its causal recent baseline.
2. **Participation-state duration.** For relative volume, volume acceleration,
   absolute pressure, and pressure persistence, this records whether the current value
   is in its development-frozen low, middle, or high band and how many consecutive
   hours that band has persisted.

Both relationships passed `18/18` control-period-seed checks in each of the
established-altcoin, full-normal, and smart-contract-platform groups. Relative volume
reduced prediction error by at least about `2.7-2.9%` in every comparison for these
groups; median reductions were about `3.0-3.3%`. Participation duration reduced error
by at least about `1.3-1.4%`; median reductions were about `1.6-1.8%`.

This is not a claim that high volume always means price will move up or down. It says
that, once the model already knows it is near a calculated price area and knows the
current movement-capacity state, the current strength and persistence of local market
participation contain additional repeatable information about how abnormal the next
hour's trading volume will be.

### Cross-cohort corroboration

The same two participation relationships stayed positive across all `18` meme-cohort
checks. Meme relative volume was strict in `17/18` checks, reduced error by at least
about `0.98%`, and had a median reduction of about `2.04%`; one uncertainty interval
crossed no improvement, so the final label remains provisional. Meme participation
duration was positive in `18/18` checks but strict in only `8/18`, with a minimum
reduction near `0.34%` and median near `0.97%`. It is a weaker but coherent
cross-market lead.

First-contact state also survived provisionally for next-hour volume in both cohorts
when added to calculated level, local participation, and volatility/compression. It
was point-positive in `18/18` checks, strict in `14/18` for the full normal cohort and
`15/18` for the frozen top-ten meme cohort. Minimum relative error reductions were
about `0.35%` and `0.67%`, with medians near `0.52%` and `1.53%`, respectively. This
supports the idea that the first arrival at a level may differ from remaining inside
it, but it is not yet a strict edge.

These three repeated cross-cohort relationships are related, not three independent
trading signals. They collectively describe current participation intensity,
participation persistence, and the phase of the level contact.

### Smaller normal-market leads

Several normal-market properties remained point-positive at all three seeds but had
wider uncertainty:

- level-family identity added small information about next-hour range for the full
  cohort and next-hour volume for BTC;
- level prominence added small range information for BTC and volume information for
  established altcoins/the full cohort;
- contact proximity added small range information for BTC;
- compression added small next-hour-range information for the full normal cohort;
- direction-neutral oscillator displacement added small range information for the
  smart-contract-platform group; and
- trend-state duration added small range information for the full normal cohort.

The smallest relative improvements for several level-property results were below
`0.2%`. They are useful as queued explanatory ingredients, not reasons to build a
large model around them.

### What failed or was parked

The seed confirmations rejected many attractive seed-42 results. Forty of the `61`
frozen group-specific candidates failed at least one later-seed control. In particular:

- meme level width and level/contact history did not remain seed-stable;
- volume acceleration and pressure persistence did not independently reproduce the
  meme volume/range leads;
- absolute volatility did not reproduce as the useful movement-capacity component;
- one-hour and twenty-four-hour BTC activity did not remain positive across the
  confirmation seeds;
- the eight-hour location contribution to four-hour absolute movement failed; and
- most occupancy, retest, and prior-contact-history results failed, leaving only the
  first-contact volume lead.

The historical BTC order-book attribution had no seed-42 route that beat its complete
control ladder, so it received no seed confirmation. The broad BTC-context and
cross-timeframe branches therefore finished without a seed-stable relationship. This
does not prove those sources never matter; it means the tested representations did not
add stable information beyond the stronger current level and local-market inputs.

### Generation 8 conclusion

The evidence sharpens the working market description:

> Calculated areas help locate where market behaviour may change. The clearest extra
> information about how much activity follows is the market's own current volume and
> how long elevated or depressed participation has persisted. First arrival at the
> area may also matter. Compression, trend state, and some level properties may refine
> movement magnitude in narrower conditions, but BTC horizon, cross-timeframe, and
> historical order-book additions did not remain stable in this batch. None of these
> findings yet predicts price direction or profit.

All eight Generation 8 siblings reached a terminal state before any descendant was
opened. The next authorized layer may therefore freeze both a limited three-source
portfolio and an evidence-triggered one-minute replay portfolio. It must preserve the
strict participation leads, include selected independent provisional mechanisms, and
avoid reviving failed BTC/order-book/timeframe routes through outcome-selected repair.

### Generation 8 evidence

- Three-seed joint review:
  `generation8_review/g8_three_seed_joint_review_20260821a/g8_three_seed_joint_review.json`.
- Group-specific seed review:
  `generation8_review/g8_three_seed_joint_review_20260821a/g8_three_seed_candidate_review.csv`.
- Relationship summary:
  `generation8_review/g8_three_seed_joint_review_20260821a/g8_three_seed_relationship_summary.csv`.
- Eight-sibling completion table:
  `generation8_review/g8_three_seed_joint_review_20260821a/g8_branch_completion.csv`.
- Cross-cohort summary:
  `generation8_review/g8_three_seed_joint_review_20260821a/g8_cross_cohort_relationships.csv`.
- Frozen seed-confirmation selection:
  `generation8_review/g8_frozen_seed_confirmation_20260821a.json`.
- Initial normal and meme runs:
  `generation8_branches/g8_freqai_attribution/g8_freqai_full_normal_20260821a/` and
  `generation8_branches/g8_freqai_attribution/g8_freqai_full_meme_20260821a/`.
- Seed-17/73 normal and meme runs:
  `generation8_branches/g8_freqai_attribution/g8_freqai_confirm_normal_20260821a/` and
  `generation8_branches/g8_freqai_attribution/g8_freqai_confirm_meme_20260821a/`.

Large caches and model artifacts remain under the corresponding Generation 8 folder
on `D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones`.

## 2026-08-21 - Generation 9 limited combinations and one-minute replay

### What this batch asked

Generation 9 completed the two sibling families that were frozen together before
either family's outcomes were inspected.

The first family asked seven deliberately small FreqAI questions. Every complete model
contained exactly three named information blocks. It had to improve prediction of an
unsigned reaction measure after each block was removed in turn, after each block was
replaced by a causal stale copy, and after each block was independently shuffled. The
same result had to repeat in both validation periods and all three seeds. This was a
hard test of whether the ingredients genuinely worked together instead of one useful
ingredient carrying the rest.

The second family selected twelve independent one-minute episodes without looking at
their future direction. Each episode was a fresh contact with an actual calculated
`1h` or `4h` level, had been in the frozen high-relative-volume state for at least two
hours, and was followed by at least twice its recent next-hour volume. The sample held
one episode from every cohort, period, and source-timeframe cell, used twelve different
pairs, and kept normal and meme coins separate. The replay then asked two separate
questions: did minute-scale price and volume react, and, only where they did, could a
simple pre-contact method tell whether price would move away from or through the level?

### Limited three-source result

All `210` FreqAI profiles and `189` frozen comparisons completed. Prediction rows were
equal, duplicate rows were zero, every command was terminal, and neither profit nor
future signed direction was used.

No complete three-source model passed the full control ladder across all three seeds.
Of `19` eligible question/group routes, `16` failed replication and `3` were only
provisional:

1. For established altcoins, calculated level + current relative volume + fresh first
   contact remained point-positive at all three seeds for next-hour volume. Its weakest
   equal-coin error improvement was about `0.20%`, but the weakest uncertainty lower
   bound was about `-0.15%`.
2. The same first-contact combination was point-positive for the full normal cohort,
   but its weakest improvement was only about `0.04%` and its uncertainty crossed no
   improvement.
3. Calculated level + current relative volume + trend-state duration was point-positive
   for full-normal next-hour range. Its weakest improvement was about `0.05%`, again
   with uncertainty crossing no improvement.

These are preserved as small normal-market observations, not promoted mechanisms.
Neither tested meme combination survived the three-seed control ladder. BTC-only and
most narrower normal groups also failed. Most importantly, combining the two strong
Generation 8 participation descriptions did not improve prediction reliably after
each ingredient was removed, made stale, or shuffled. The correct conclusion is not
that current relative volume and participation duration stopped mattering. It is that
this particular three-input merge did not add stable information beyond its simpler
parts.

### One-minute reaction result

The exact missing `1m` ranges for TRX, XRP, and WIF were downloaded and atomically
merged. Final coverage was `74,160/74,160` expected rows across all twelve intervals,
with no duplicates, gaps, invalid rows, or overlap conflicts.

Six of the twelve selected episodes met the stricter minute-scale reaction rule: price
had to travel at least one active zone half-width and post-contact volume had to be at
least `1.5` times the pre-contact baseline within sixty minutes. Normal coins reacted
in `4/6` examples and meme coins in `2/6`. Because selection deliberately required an
abnormally large parent next-hour volume result, these percentages do not estimate how
often ordinary level contacts react.

The paths were mixed: `5` first resolved as breakouts, `6` as rejections, and `1` was a
tie. This supports the basic market picture that calculated areas can coincide with a
decision or activity point, but the existence of the area does not say which direction
wins.

Nine pre-contact direction methods were checked. The best full-coverage observation,
a leave-one-out majority from the other coins in the same cohort, was correct on `4/6`
reacted and callable episodes, but joint reaction-and-direction success was only
`4/12 = 33.3%`. A multi-timeframe EMA vote was correct on `4/5` reacted episodes where
it issued a call, but it also achieved only `4/12 = 33.3%` joint success. No method met
the `55%` joint floor or the `65%` main target, and twelve episodes are below the
predeclared minimum for a direction lead.

The raw matched controls sometimes showed larger movement or volume at the actual
contacts, but only `2/12` same-state no-level controls, `1/4` available stale-level
controls, and `1/5` available shifted-level controls were sufficiently state-balanced.
That is too little clean comparison evidence to claim that the calculated level caused
the reaction. Three episode windows also touched a boundary-extension condition, so
their wider context should not be overinterpreted from the current clipped view.

### Joint conclusion and queued next layer

Generation 9 does not justify a larger three-source model or a broader one-minute
direction search. The provisional normal-only combinations and the one-minute path
ideas remain recorded but parked.

Two mechanisms remain eligible for Generation 10 because they were already strict
across all three Generation 8 seeds: current relative volume and participation-state
duration. They should be confirmed separately on later untouched periods, declared
normal-coin groups, relevant regimes, and realistic stale or missing-source states.
Their failed Generation 9 merge is evidence to keep the tests separate. The programme
still has not produced three materially independent indications from two route families
at `55%` unseen joint reaction-and-direction success, so Objective 02b remains
unresolved and no trading rule is promoted.

### Generation 9 evidence

- Joint review:
  `generation9_review/g9_joint_review_20260821a/g9_joint_review.json`.
- Three-source group decisions and question summary:
  `generation9_review/g9_joint_review_20260821a/g9_model_joint_decisions.csv` and
  `generation9_review/g9_joint_review_20260821a/g9_model_question_summary.csv`.
- One-minute cohort and direction summaries:
  `generation9_review/g9_joint_review_20260821a/g9_one_minute_cohort_summary.csv` and
  `generation9_review/g9_joint_review_20260821a/g9_one_minute_direction_summary.csv`.
- Frozen result-inspired queue:
  `generation9_review/g9_joint_review_20260821a/g9_result_inspired_queue.csv`.
- Original frozen sibling definition:
  `generation9_review/g9_frozen_limited_multisource_and_one_minute_batch_20260821a.json`.

Large one-minute event/control paths remain under
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation9_branches`.

## 2026-08-21 - Generation 10 later-period confirmation

### What this final authorized batch asked

Generation 10 did not search for another attractive combination. It took the two
strongest single-source Generation 8 findings and asked whether they still worked when
tested separately on a later month whose relevant next-hour-volume outcomes had not
been opened for either mechanism:

1. **Current relative volume:** does knowing how large the just-completed candle's
   volume is compared with its causal recent median improve the estimate of unsigned
   next-hour volume near a calculated area?
2. **Participation-state duration:** does knowing whether volume is in a frozen low,
   middle, or high participation band, plus how many consecutive hours that state has
   lasted, improve the same estimate?

Both models retained the same calculated-level, absolute-volatility, and compression
base. Each current input had to beat three alternatives: a model without it, the same
input delayed by `72h`, and the same values shuffled to different timestamps within
the same period. A useful current market measurement should beat all three. Beating
the plain model but not the shuffled input would mean that adding values with a similar
distribution helped the learner, but the values were not reliably useful at their real
times.

The checks were frozen across the established-altcoin, full-normal, and
smart-contract-platform groups; the `2026-07-20` to `2026-08-05` and `2026-08-05` to
`2026-08-20` halves; and seeds `42`, `17`, and `73`. A group could pass only if every
control was point-positive in both periods and all three seeds, with adequate coin and
row support, no single-coin domination, and a weekly-block uncertainty lower bound
above no improvement. Profit and future price direction were never targets.

These dates are honestly described as **target-and-mechanism-unopened chronological
confirmation**, not never-before-used market history. The same candles had previously
been used by an unrelated absolute-price-excursion study, but their next-hour volume
outcomes had not been compared for either mechanism before this freeze.

### Runtime repair and evidence integrity

The first launch stopped four profiles before model training because the new cache
carried only the requested volume target while the reused research strategy validates
the complete standard target-column interface. This was a cache/schema mismatch, not
a market result. The cache was rebuilt with the unused standard targets present for
interface compatibility, while the frozen configurations still trained only on the
declared next-hour-volume target. A new run id was used. The failed technical run is
retained for traceability and contributes no evidence.

The repaired run completed all `24/24` profiles and `18/18` frozen comparisons. Every
candidate/control comparison used identical prediction keys, no duplicate prediction
rows were removed, and every profile command was terminal.

### Broad result

Neither mechanism survived. All six mechanism/group routes failed the complete
three-seed confirmation; there were zero strict and zero provisional routes.

For **current relative volume**:

- Against the model without relative volume, all `9/9` group/seed checks were
  point-positive in the early half, but only `6/9` were point-positive in the late
  half. Only `8/18` of those period checks were strict.
- Against the `72h`-old version, all `18/18` checks were point-positive and `15/18`
  were strict. Current volume therefore usually described the next hour better than
  old volume.
- Against the shuffled-time version, `14/18` checks were point-positive but only
  `1/18` was strict. The average improvement was small, several late-period checks
  were negative, and uncertainty usually included no improvement.

For **participation-state duration**:

- Against the model without duration, `13/18` checks were point-positive but only
  `1/18` was strict. The mean late-period result was slightly negative.
- Against the `72h`-old version, `17/18` checks were point-positive but none was
  strict.
- Against the shuffled-time version, only `4/18` checks were point-positive and none
  was strict. In most comparisons the values attached to the wrong timestamps did
  better than the real current duration.

In plain terms, the strong Generation 8 relationships did not carry forward as broad,
stable, current-timing signals. Relative volume retained more information than a
three-day-old copy, but the inability to beat shuffled timing consistently means its
earlier benefit cannot be attributed confidently to the current value at the current
level contact. Participation duration failed that timing test more clearly. This does
not erase the wider evidence that calculated areas coincide with changes in market
activity; it narrows the claim about these two extra inputs.

### Result-inspired observations, with limits

The frozen run also produced descriptive activity-state slices against the plain base
model. These were not allowed to rescue a failed route because they were inspected
after the broad result, pooled the two periods, and did not repeat the full stale and
shuffle ladder.

- Relative volume was point-positive in every active-market slice (`9/9`) and every
  quiet-market slice (`9/9`), but in only `3/9` ordinary-activity slices.
- Participation duration was point-positive in every active-market slice (`9/9`), in
  `6/9` quiet slices, and in only `1/9` ordinary-activity slices.

This suggests a rational conditional question: these measurements may matter when
activity is clearly displaced from normal and may add noise when activity is ordinary.
It does not yet show that the current timestamp matters inside those states. A future
test would need pre-frozen regime definitions, a genuinely later period, and the full
base/stale/shuffle controls within each regime.

BNB was the only coin for which both mechanisms were point-positive against all three
controls in both periods and all three seeds: `18/18` pair/control/period/seed checks
for relative volume and `18/18` for participation duration. This is deliberately not
called an edge. BNB was noticed after opening outcomes, is one coin, and the pair rows
do not carry the group-level uncertainty gate. The correct follow-up would first audit
BNB's data and model behaviour, then predeclare a rational peer group and test a later
period. Tuning a feature to preserve BNB's historical examples is prohibited.

### Terminal conclusion and queued choices

The authorized Generation 10 horizon is complete. Objective 02b remains unresolved:
there are not three materially independent indications from two route families at the
`55%` unseen joint reaction-and-direction floor, no direction method is confirmed, and
the `65%` main target has not been approached. No entry, exit, profit, direction, or
strategy claim is promoted.

The terminal queue preserves three possible future choices without launching them:

1. a new untouched conditional-regime replication with the complete timing-control
   ladder;
2. a BNB integrity audit followed by an outcome-blind peer-group replication; and
3. a separately authorized, reaction-selected directional objective using only
   information known before contact and stronger matched controls.

The broad relative-volume and participation-duration timing claims are parked rather
than repeatedly retuned. Further branch work requires new user authorization.

### Generation 10 evidence

- Terminal review:
  `generation10_review/g10_terminal_review_20260821a/g10_terminal_review.json`.
- Period/control summary:
  `generation10_review/g10_terminal_review_20260821a/g10_control_period_summary.csv`.
- Descriptive regime and pair audits:
  `generation10_review/g10_terminal_review_20260821a/g10_activity_regime_summary.csv`
  and `generation10_review/g10_terminal_review_20260821a/g10_pair_observations.csv`.
- Result-inspired queue:
  `generation10_review/g10_terminal_review_20260821a/g10_result_inspired_queue.csv`.
- Original frozen definition:
  `generation10_review/g10_frozen_untouched_confirmation_20260821a.json`.
- Terminal repaired run:
  `generation10_branches/g10_untouched_participation_confirmation/`
  `g10_participation_confirmation_20260821b_target_schema_repair/`.

Large model artifacts remain under the matching Generation 10 branch on
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones`.

## 2026-08-22 - Generation 11 broad multi-horizon reaction attribution

### What was measured

Generation 11 deliberately stopped asking whether a row was profitable after an
arbitrary holding time. At every independent calculated-level contact it asked two
direction-neutral questions over the next `1h`, `2h`, `4h`, and `8h`:

1. Did price travel far enough to count as a material reaction while volume also rose?
   A reaction required unsigned price travel of at least the larger of `0.5 ATR` and
   the active zone half-width, together with volume of at least `1.25` times its causal
   baseline.
2. How much volume followed the contact relative to its causal baseline?

The inputs were kept in understandable families: calculated-level identity and source
timeframe, isolated-versus-cluster geometry, recent local activity/volatility, local
trend/momentum, wider crypto-market state, historical BTC order-book context, and
aggregate historical news context. The final low-dimensional combination was also
tested, but only after all individual families had their own complete comparisons.

Every real input family was compared with an immediately simpler model, a causal copy
delayed by `72h`, and a deterministic within-period shuffled-time copy. Combination
models also had to beat their component-removal versions. Candidate and control models
were scored on identical coin/timestamp rows. The established and meme cohorts were run
separately, and results were reviewed jointly only after all `62/62` requested profiles
were terminal.

### Broad result

The clearest initial lead was the **recent local activity and volatility state**. It
improved estimates of the combined price-and-volume reaction at every `1h`-to-`8h`
horizon in both cohorts. It also improved near-term volume estimates, especially through
`4h`. The full family included current relative volume, volume acceleration, candle
pressure and persistence, ATR-scaled range, Bollinger width, and compression measures.

Local trend/momentum, wider crypto-market state, and calculated-level identity retained
smaller reaction leads. The historical BTC order-book block retained no target. The
news route had zero eligible rows in these periods and is labelled **not tested**, not a
negative market result.

The broad geometry-plus-trend-plus-market combination mostly failed its component and
placebo ladder. More features were therefore not assumed to be better: the simpler
families explained more of the useful result than the large merge.

Across the two cohorts, `10` of `56` route/target cells were strict in both and another
`24` were point-positive in both. These are related target horizons, not 34 independent
edges. Balanced reaction accuracy for the strongest activity family was roughly
`0.58-0.65` depending on cohort and horizon, while the raw reaction rate was only about
`0.36-0.46`. That is evidence of reaction discrimination, not direction prediction.

### Why this did not overrule Generation 10

Generation 10 had already shown that current relative volume and participation duration
alone failed a genuinely later timing-specific confirmation, especially against
shuffled timestamps. Generation 11 reused older validation periods and its activity
family was broader. Its strong initial result could therefore mean that volatility,
compression, pressure, or interactions inside the family mattered; it could also be a
reused-period effect. It was not promoted. A later chronological stress test and a
separate component decomposition were frozen before any descendant result was opened.

### Generation 11 evidence

- Joint review:
  `generation11_review/generation11_branches/g11_freqai_initial_attribution/`
  `g11_initial_joint_review_20260822a/g11_joint_freqai_review.json`.
- Plain route and reaction-skill summaries live beside that joint review as
  `g11_plain_route_summary.csv` and `g11_reaction_skill_summary.csv`.
- The causal normal and meme cache manifests are under
  `generation11_shared/g11_freqai_cache_20260822a/`.

## 2026-08-22 - Generation 12 later chronology, component attribution, and restrained combinations

### Frozen sibling design

Generation 12 completed three sibling runs before any result was interpreted:

1. a later normal-coin chronology from `2026-04-01` through `2026-08-20`, split at
   `2026-07-01`, for the five broad Generation 11 families;
2. separate activity, volatility, trend, oscillator, BTC/ETH-leader, wider-cohort, and
   pair-relative-to-BTC components on the existing normal validation periods; and
3. the identical component questions on the frozen top-ten meme cohort.

Only four restrained combinations were included: activity plus trend, activity plus
wider-market state, activity plus geometry, and activity plus trend plus wider-market
state. Every combination had to beat each named component as well as stale and shuffled
controls. The complete run finished `104/104` profiles with zero failed commands, zero
unequal prediction-key comparisons, and zero duplicate prediction rows. Profit and
future signed direction were absent.

### Later-date result

The later chronology supported one important but limited statement: **the local
activity/volatility state helped distinguish whether a calculated-level contact would
produce a material unsigned price-and-volume reaction**.

- It passed the full simpler/stale/shuffled ladder for the `1h`, `2h`, `4h`, and `8h`
  reaction targets in both later date halves.
- Its minimum equal-coin reduction in reaction-probability error was about `1.3%`,
  `1.6%`, `1.8%`, and `1.6%` respectively. Weekly-block uncertainty stayed above zero.
- Across the ten coins, two date halves, and three controls, the gain was positive in
  `55/60`, `54/60`, `55/60`, and `57/60` checks respectively.
- Balanced reaction accuracy was about `0.60-0.62` in both later halves. The reaction
  base rate was about `0.38-0.48`, so this was not achieved by always predicting the
  majority outcome.
- BTC, ETH, AVAX, TRX, and XRP were positive in all `24/24` horizon/period/control
  checks. ADA was `23/24`, LINK and SOL were `22/24`, while BNB and DOGE were weaker at
  `17/24`. The result was therefore broad but not uniform.

The same family did **not** retain later-date future-volume strength beyond `1h`; its
`2h`, `4h`, and `8h` volume targets failed. This narrows the claim. Current conditions
appear more useful for deciding whether a contact becomes an active reaction than for
predicting that high volume will persist for many hours.

Wider crypto-market alignment was strict for `1h`, `2h`, and `8h` reaction and
point-positive at `4h`. Local trend/momentum and calculated-level identity were
point-positive but did not clear uncertainty against every control. Isolated/cluster
geometry failed all eight later targets. This does not prove clusters are useless; it
means the current geometry representation added no stable information in this test.

### Component and combination result

On the reused normal and meme validation periods, two halves of the activity family
were each useful:

- **Participation/pressure:** relative volume, acceleration, absolute candle pressure,
  and pressure persistence were strict in both cohorts for reaction through `4h` and
  future volume through `4h`; their `8h` results were weaker.
- **Volatility/compression:** ATR-scaled range, prior range, Bollinger width, and range
  contraction were strict in both cohorts for reaction through `8h` and future volume
  through `2h`.

Two other materially different families survived more narrowly. BTC/ETH leader state
was strict in both cohorts for `1h` and `2h` reaction. RSI/MACD oscillator and momentum
change was strict in both for `2h` and `4h` reaction and `1h` future volume. Direction/
strength indicators such as EMA slope, moving-average separation, return slope, and ADX
were only point-positive for `4h` and `8h` reaction.

No tested combination was strict in both cohorts. The three-source activity plus trend
plus market model failed all targets, and the pairwise combinations retained only small
point leads. The practical conclusion is to keep useful context families separate until
a later test shows that combining them adds information beyond the best component.

The component results explain plausible parts of the broad activity lead, but they reuse
earlier validation dates and are not new chronological confirmation. Only the broad
activity/volatility family has passed the current later-date stress test.

### What this means for the programme

This is a reaction-only lead, not a complete trading edge. The model says that current
market activity can improve the odds of recognizing a calculated area where price and
volume are likely to become unusually active. It does not say whether price will bounce
up, reject down, or break through. Therefore the `55%` joint reaction-and-direction
floor and `65%` main target have not been reached.

The next balanced generation preserves three siblings before any deeper branch:

1. matched ordinary/no-level and near-miss timestamps to test whether the activity
   family predicts reactions specifically around calculated areas or merely predicts
   busy markets everywhere;
2. timestamp-safe `4h`, `8h`, and `1d` indicator state to test whether higher-timeframe
   context adds beyond the current `1h` information; and
3. `12h`, `24h`, and `48h` reaction paths, especially for `4h`, `8h`, and `1d` source
   levels, so conclusions are not restricted to short forecasts.

These siblings must all finish before any one of them can spawn a descendant. Order-book
and news remain parked until their real timestamp coverage supports a fair test.

### Generation 12 evidence

- Frozen plan:
  `generation11_review/generation11_branches/g11_freqai_initial_attribution/`
  `g12_balanced_followup_freeze_20260822a.json`.
- Joint review:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g12_joint_review_20260822a/g12_joint_review.json`.
- Later-date readable decisions:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g12_joint_review_20260822a/g12_recent_route_decisions.csv`.
- Cross-cohort component decisions:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g12_joint_review_20260822a/g12_joint_attribution_decisions.csv`.

Large model artifacts remain under
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones\generation12_branches`.

## 2026-08-22 - Generation 13 real-level controls, higher-timeframe context, and longer reactions

### Why this batch was needed

Generation 12 showed that recent activity and volatility helped estimate whether a
calculated-level contact became unusually active. That left three important alternative
explanations:

1. perhaps the model was only recognizing a busy market and the calculated price area
   did not matter;
2. perhaps completed `4h`, `8h`, or `1d` market state added useful context that the
   existing `1h` inputs missed; and
3. perhaps useful reactions unfolded over `12h`, `24h`, or `48h`, outside the earlier
   short forecast windows.

All three sibling questions, their controls, feature columns, target definitions, and
interpretation rules were frozen before any result was read. The frozen surface contained
`58` unique profiles and `54` planned comparisons. The supporting cache covered all ten
established coins and all ten frozen top-traded meme coins, with zero unsupported profiles
and zero timestamp or placebo violations. The complete FreqAI run finished `87/87` models
on the first attempt with equal prediction keys and no duplicate prediction rows.

### Do genuine calculated areas matter beyond busy-market persistence?

The direct control test compared real calculated-level contacts with two harder controls:

1. ordinary timestamps where a matched pseudo-level was contacted; and
2. genuine near misses where price approached a calculated area but did not contact it.

Controls were matched without looking at future outcomes. Matching required the same
source timeframe and approach state, similar starting distance, and similar current
relative volume, volume acceleration, pressure, pressure persistence, ATR-scaled range,
prior range, Bollinger width, and compression. Any control timestamp coinciding with any
tracked real-level contact was removed. This excluded `933,395` contaminated candidate
control rows and left `296,218` matched rows before the forecast-horizon purges.

A reaction meant that unsigned price travel reached at least the larger of `0.5 ATR` and
the calculated zone half-width while mean future volume reached at least `1.25` times its
causal prior baseline. Across all source timeframes together, real contacts reacted more
often than matched ordinary timestamps by approximately:

- `5.6-8.2` percentage points over the next `1h`;
- `3.7-8.4` points over `2h`;
- `3.9-5.6` points over `4h`; and
- roughly `1.0-4.8` points over `8h`, where the result became less reliable.

The `1h`, `2h`, and `4h` differences passed the strict control and uncertainty ladder in
the original established-coin periods, the later normal-coin chronology, and the meme
cohort. At `8h`, the original normal result remained strict, the meme result was only a
small point lead whose uncertainty crossed zero, and the later normal chronology did not
pass every comparison. Real contacts also generally exceeded genuine near misses by a
larger margin at `1h` through `4h`.

In plain language, the earlier activity model was not merely identifying busy hours.
Contact with a causally calculated price area supplied additional short-window reaction
information after current market activity had been closely matched. This still does not
show which direction price will move, and it does not prove that every level family is
equally useful.

### Which completed higher-timeframe state helped?

The higher-timeframe FreqAI sibling compared completed `4h`, `8h`, and `1d` activity/
volatility and trend/momentum state with three controls: the existing `1h` information,
a `72h`-old higher-timeframe copy, and a within-period shuffled higher-timeframe copy.
Candidate and control models were compared on identical coin/timestamp rows.

One result completed the entire original-normal, later-normal, and meme ladder:
**completed `8h` activity/volatility improved prediction of whether an unsigned reaction
would occur within `8h`**. It beat all three controls in every evaluation period, with at
least five coins contributing positively to every comparison. The smallest equal-coin
reduction in probability error was `0.0043`, and the weakest weekly-block uncertainty
lower bound remained positive at `0.00033`.

Balanced reaction accuracy for the candidate was approximately:

- `62.8-63.1%` in the original established-coin periods;
- `61.4-61.8%` in the later normal-coin chronology; and
- `60.6-60.8%` in the meme cohort.

The underlying reaction rate was about `44-48%`, so these figures were not produced by
always choosing the majority class. They are also reaction-only figures: they say
whether unusually active price-and-volume behaviour follows, not whether price goes up
or down.

The coin result was broad but not identical. BTC, ETH, BNB, TRX, AVAX, and LINK were
consistently positive in the normal tests. ADA, SOL, DOGE, and XRP varied more by
chronology. In memes, ORDI, FARTCOIN, PEPE, and WIF were positive in every comparison;
the other six generally contributed but less uniformly. This supports cohort and
smaller coherent-group analysis rather than requiring every coin to behave identically.

Completed `8h` activity also had a weaker `2h` lead in the older normal and meme periods,
but it failed the later normal chronology and was not retained. The tested `4h` and `1d`
state, and higher-timeframe trend/momentum, did not complete the three-cell ladder. This
does not establish that those timeframes can never help; it means their current compact
representations did not add stable information in this test.

### What happened over `12h`, `24h`, and `48h`?

The only long-horizon result positive in all three cells was **local activity/volatility
at contacts with `4h` source levels for a reaction within `48h`**. It beat the level-only,
stale, and shuffled controls by at least `0.0382` probability-error units across the
complete joint comparison. The original normal and meme cohorts passed strictly, but the
later normal late period had only `100` fairly comparable rows across six coins and its
uncertainty crossed zero. The joint classification is therefore a cross-window point
lead, not a strict chronological confirmation.

Balanced reaction accuracy was about `58-63%` in the original normal periods, `61-62%`
in memes, and `55-63%` in the later normal periods. The weak `55%` later-period value and
sparse support are reasons to test stability, not reasons to advertise a 48-hour edge.
The `12h` and `24h` reaction results and the long future-volume targets worked only in the
older standard cohorts. They failed the later normal ladder. Long outcomes around `8h`
and `1d` source levels, oscillator/momentum inputs, and BTC/ETH leader state also failed
the full cross-window requirement.

### Joint interpretation and boundary

Generation 13 supplies two materially different reaction leads and one useful control
finding:

1. actual calculated-level contacts have more short-window price-and-volume reactions
   than closely matched ordinary times and near misses;
2. completed `8h` activity state adds repeatable information about an `8h` reaction; and
3. local activity at `4h` levels may help with a `48h` reaction, but the newest late
   window is too sparse and variable for strict confirmation.

It also rejected or parked many alternatives: broader higher-timeframe trend, the tested
`4h`/`1d` state, most longer-horizon source/target combinations, and the idea that adding
more inputs automatically improves a model. Negative results remain part of the evidence
base and prevent one early winner from dominating the programme.

No future signed direction, profit, entry, exit, or trading rule was tested. Reaction
accuracy above `60%` is not the user's `55%` goal: that goal requires the same unseen call
to predict both a real reaction and its direction. The joint target remains untested and
unmet.

### Generation 13 evidence

- Frozen complete sibling plan:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings_freeze_20260822a.json`.
- Direct level-control result:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/direct_level_controls/g13_direct_controls_20260822a/`
  `g13_direct_control_result.json`.
- Joint FreqAI review:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/freqai/g13_joint_review_20260822a/g13_joint_review.json`.
- The authoritative full-model run stem is `g13_broad_freqai_20260822b`. The earlier
  `20260822a` attempt stopped during preflight because its multi-timeframe cache had not
  yet materialized the frozen target/readiness columns; it is not evidence.

The next broad combination batch must be frozen as a complete sibling set before it
launches. It should include reaction-definition sensitivity, attribution inside the
`8h` activity family, single-level and cluster interaction, wider-market conditioning,
the interaction between the `8h` activity lead and the provisional `4h`-level/`48h`
lead, and coin-family/chronology stability. Any directional `1m` replay remains a queued,
bounded later lane rather than the main batch.

## 2026-08-22 - Generation 14 robustness, attribution, and restrained combinations

### Why this batch was needed

Generation 13 established that genuine calculated-level contacts were followed by
short-window price-and-volume reactions more often than closely matched ordinary times
and near misses. It also retained completed `8h` activity/volatility as useful context
for estimating an `8h` reaction, and left a provisional suggestion that local activity
at `4h` source levels might help estimate a `48h` reaction. Generation 14 challenged
those findings without letting the first positive result choose the rest of the batch.

The complete frozen sibling set contained `64` feature profiles and `64` declared
comparisons. It tested nine adjacent reaction definitions, separated the completed `8h`
activity block into participation/pressure and volatility/compression, tested level
identity and cluster geometry, added wider-market and oscillator context separately,
combined local and completed `8h` activity at `4h` levels over `12h` to `48h`, examined
coin/time stability, and refreshed news/order-book coverage. Every sibling reached a
terminal result or an explicit coverage park before the joint review.

### Did the calculated-area effect depend on one convenient reaction definition?

No. The direct test used every combination of three unsigned price thresholds (`0.35`,
`0.50`, and `0.75 ATR`) and three future-volume thresholds (`1.10`, `1.25`, and `1.50`
times the causal trailing baseline). It preserved the same outcome-blind matched ordinary
timestamps and genuine near-miss controls used in Generation 13. Across `296,218` matched
rows, all nine definitions remained positive over `1h`, `2h`, and `4h` in the original
normal, later normal, and meme cells.

The original `0.50 ATR` plus `1.25x` volume definition illustrates the size of the
location difference. Depending on period and cohort, real contacts exceeded matched
ordinary times by about `3.7-8.4` percentage points at `1h-4h`, and exceeded genuine near
misses by about `7.6-15.4` points. The adjacent definitions retained the same conclusion,
so this is not an artefact of one hand-picked cutoff.

At `8h`, the original normal and meme cells remained point-positive, but their uncertainty
was weaker. The later normal `8h` matched-random comparison had only four eligible coins;
its observed difference remained positive but did not meet the predeclared five-coin
support rule and its uncertainty crossed zero. Overall, `11/12` cohort/window/horizon rows
passed the robust point rule and `9/12` passed the stricter uncertainty rule. The reliable
part of the direct finding therefore remains the `1h-4h` reaction window.

### Did more context improve FreqAI reaction estimates?

Mostly not. The supported FreqAI surface completed `89/89` model commands: `25` profiles
for each of the original normal, later normal, and meme multi-timeframe cells, plus seven
profiles for each of the two supported normal long-horizon cells. Commands used one model
thread each and no more than four concurrent profile workers. Prediction keys were equal,
no duplicate prediction rows were removed, and no profit or future signed direction was
used.

Only one row completed the looser three-cell ladder. For a reaction within `1h`, adding
the current calculated-level family and source-timeframe identity to local and completed
`8h` market activity reduced equal-coin absolute probability error against no identity,
a causal `72h`-old identity, and a within-period shuffled identity in both periods of all
three cells. The weakest improvement was only `0.00073`, however, and the weakest weekly
uncertainty lower bound was `-0.00313`. Balanced reaction accuracy for the complete model
was about `60.9-65.0%`, but the model without current level identity was already almost the
same; thresholded balanced-accuracy changes ranged from slightly negative to about half a
percentage point positive. This is a small attribution lead, not a reliable edge.

Leave-one-coin-out checks were positive in `60/60` original-normal comparisons and
`58/60` comparisons in both the later-normal and meme cells, so the aggregate result was
not created by one obvious coin. It was not uniformly stable inside the predeclared coin
groups: BTC changed sign between periods, and both the smart-contract and other-alt groups
had one later or control cell below zero. The sensible interpretation is a weak broad lead
whose useful level families or timeframes have not yet been identified.

No other tested combination completed the three-cell ladder. The full completed `8h`
activity block did not beat both of its participation/pressure and volatility/compression
halves. Adding the current cluster block, BTC/ETH and cohort market state, or RSI/MACD
oscillator state to completed `8h` activity failed at every `1h`, `2h`, `4h`, and `8h`
joint row. These failures reject the tested bundles; they do not prove that every more
specific cluster, market-state, or approach-path condition is useless.

### Did the provisional long-horizon combination survive?

No. At contacts with `4h` source levels, combining current local activity with completed
`8h` activity failed to improve `12h`, `24h`, or `48h` reaction or future-volume estimates
against level-only, either activity scale alone, causal `72h`-old blocks, and shuffled
blocks in both normal chronological cells. The prior Generation 13 `48h` suggestion was
therefore not strengthened and is parked rather than refined again.

The meme long-horizon cell was not modelled: none of the ten frozen meme pairs had the
minimum `30` pre-window training contacts required for the `4h`-level long-horizon
profiles. This was recorded as a coverage park without weakening the support rule or
reading future outcomes to choose pairs.

### External context and current boundary

Historical news had zero eligible rows in every declared validation period. Historical
order-book context covered the older normal periods, the first later-normal period, and
both meme periods, but not the newest later-normal period. Because the frozen external
question required every declared period, both sources were parked for this generation.
Order-book data may support a later explicitly limited exploratory cell, but it cannot
currently supply complete chronological confirmation.

Generation 14 strengthens the claim that calculated areas identify short-window market
reaction locations. It does not show that large feature bundles improve prediction, and
it still says nothing about whether the reaction goes up, down, away from, or through the
area. The user's `55%` unseen joint reaction-and-direction floor and `65%` main target
remain untested and unmet.

### Generation 14 evidence

- Frozen complete sibling plan:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combination_freeze_20260822a.json`.
- Reaction-definition sensitivity:
  `g13_broad_siblings/g14_broad_combinations/label_sensitivity/`
  `g14_label_sensitivity_20260822a/g14_label_sensitivity_result.json`.
- External-source readiness:
  `g13_broad_siblings/g14_broad_combinations/external_readiness/`
  `g14_external_readiness_20260822a/g14_external_readiness_result.json`.
- Joint FreqAI review:
  `g13_broad_siblings/g14_broad_combinations/freqai/`
  `g14_joint_review_20260822a/g14_joint_review.json`.
- The authoritative full-model run stem is `g14_broad_freqai_20260822b`; technical smoke
  and incomplete preparation stems are not evidence.

The next complete batch should stay broad. It should separate the combined reaction into
price, volume, volatility, timing, dwell, and traversal behaviours; attribute the small
`1h` level-identity lead to actual level families and source timeframes; test contact age
and approach/lifecycle state; repair cluster representation using explicit component
composition rather than a generic geometry bundle; compare predeclared market groups;
and use historical order-book context only on its honestly supported common rows. These
siblings must be frozen together before any one result receives a descendant.

## 2026-08-22 - Generation 15 reaction paths and direct attribution

### What was frozen before outcomes were reviewed

Generation 15 was one seven-sibling batch. It was designed to answer several different
questions before allowing any one interesting result to dominate the work:

1. Which parts of market behaviour actually change after contact with a calculated area?
2. Which level families repeatedly locate those changes?
3. Which completed source timeframes repeatedly locate them?
4. Do high contact-candle volume plus range, or volume plus pressure, add information
   beyond either component alone?
5. Do first/repeated contacts or fast/slow arrival change what happens next?
6. Do explicit cluster types add something beyond an isolated calculated level?
7. Does timestamp-safe historical BTC order-book activity change the response after
   local and wider-crypto state are matched?

The complete set was frozen before any Generation 15 outcome was opened. All seven
siblings then reached a terminal result before this review. The analysis covered the ten
established coins and the frozen ten most-traded meme coins, original validation periods,
the later April-August 2026 normal-coin chronology, and `1h`, `2h`, `4h`, and `8h` future
paths. It did not use profit, entries, exits, or future signed direction.

The first three siblings produced `383,679` outcome-blind matched rows before each
horizon's overlapping paths were removed. Real contacts were compared separately with:

- an ordinary timestamp whose market state and pseudo-level geometry were closely
  matched; and
- a genuine near miss of the same level family and source timeframe.

The matching used only information already available at contact time. It required the
same coin, period, source timeframe, level family, and approach side, a starting-distance
caliper of `0.10 ATR`, and similar pre-contact volume, pressure, volatility, Bollinger,
and compression state. Controls occurring at any real tracked contact time were removed.

The remaining four siblings produced `94,308` matched contact-to-contact contrasts.
Activity combinations had to beat both simpler components and a quiet contact. Lifecycle
and arrival tests had to beat their shuffled labels. Cluster types were compared with
matched isolated contacts, with overlapping "any presence" descriptions kept separate
from clean single-component attribution. Order-book activity had to beat quiet state and
also exceed causal `72h`-old and within-period shuffled order-book controls. A strict
result also required weekly-block uncertainty to stay on one side of zero and no one coin
to contribute more than half of the absolute equal-coin effect.

### What a calculated-area contact reliably located

Across the original normal, later normal, and meme cells, a real calculated-area contact
was followed by three repeatable changes relative to both controls:

- **more future volume**;
- **a wider future high-low range**; and
- **more crossings back and forth through the calculated price**.

All three remained point-positive at every `1h`, `2h`, `4h`, and `8h` horizon. After the
uncertainty and one-coin-dominance rules were applied, the strongest general findings
were:

- future volume at `1h`, `2h`, and `4h`;
- future range at `1h` and `2h`; and
- crossings at `1h`, `2h`, and `4h`.

The effect weakened with time. Depending on cohort, period, and control, the equal-coin
future-volume difference was approximately `+0.17` to `+0.44` at `1h`, `+0.17` to
`+0.35` at `2h`, `+0.12` to `+0.25` at `4h`, and `+0.07` to `+0.19` at `8h`. These are
differences in a volume ratio measured against its causal prior baseline, not percentage
points of trading profit. Future-range differences were approximately `+0.05` to
`+0.22`, `+0.06` to `+0.18`, `+0.06` to `+0.14`, and `+0.03` to `+0.12` over the same
horizons. Crossings increased by about `0.15-0.26` at `1h`, `0.14-0.34` at `2h`,
`0.11-0.41` at `4h`, and `0.06-0.46` at `8h`; the late crossing estimates were less
certain.

The same broad ladder did **not** retain larger absolute price excursion, faster arrival
at `0.5 ATR`, hit rate, dwell, or absolute pressure change. One generic-prior-range
`4h` excursion row was point-positive but not strict. Therefore the evidence says that
calculated areas locate **market traffic and activity**, not that they reliably cause a
large move away from the area.

The crossing result is especially important. A useful calculated area is not automatically
support or resistance. Many of these areas behave like magnets or busy junctions that
price traverses repeatedly. Direction, rejection, acceptance, and breakout still require
a separate explanation.

### Which families and timeframes repeated the activity result

Several level families independently beat their ordinary-time and near-miss controls:

- prior-range levels retained volume through `8h`, plus short range and crossing effects;
- round numbers retained volume and range through `4h` and crossings through `4h`;
- explicit prior volume-profile levels retained short volume, range, and crossing effects;
- volume-profile nodes retained short volume and crossings through `4h`; and
- settled volume-profile levels retained `1h` volume and crossings.

Confirmed swings had a few three-cell point results but did not pass the full strict
uncertainty ladder. The tested TLV2 families did not complete it. These tests show which
families locate activity; they were not head-to-head family rankings, so a longer list of
passes does not prove one family is universally better than another.

Source timeframe also changed the duration of the evidence:

- `1h` levels retained strict volume, range, and crossing effects through `4h`;
- `4h` levels retained strict future volume through `4h`, with range and crossings
  strongest through `2h`;
- `8h` levels retained only a strict `1h` future-volume result; and
- `1d` levels did not complete the broad support ladder.

This does not justify automatic higher-timeframe precedence. It shows that the currently
calculated `1h` and `4h` areas have the clearest short-window activity association. A
higher timeframe cannot be given authority merely because it is higher.

The activity-location finding also appeared inside BTC, the five smart-contract-platform
coins, the four other established alts, and the frozen meme cohort. The group review found
many point-consistent rows in each group, including the same general range/volume/crossing
behaviours. Those hundreds of rows are correlated horizons, metrics, families, and
timeframes, not hundreds of independent trading edges. Their value is that the result is
not explained by one pair or one market type.

### What did not add portable information

None of the four contextual siblings completed the full original-normal, later-normal,
and meme ladder:

- contact volume-plus-range and volume-plus-pressure combinations produced several old
  normal-period results but did not reproduce in the later normal or meme cells;
- exact single-level lifecycle bins were sparse, while the broader multi-level mean
  lifecycle lane was descriptive and also failed portability;
- cluster "any presence" often differed from isolated contacts in the original normal
  and meme cells, but failed the later chronology and mixed several cluster mechanisms;
- clean exclusive cluster components did not complete the portability ladder; and
- historical BTC order-book activity produced a few old-normal cells, but did not pass
  the meme or later-normal comparison and still lacks the newest later-period source
  coverage.

No attribution-eligible contextual combination was portable even within the predeclared
BTC, smart-contract, other-alt, or meme groups. The only group-level contextual repeats
were descriptive meme cluster-any-presence rows, which deliberately remain ineligible
for a causal or predictive claim because their components overlap.

These failures matter. They say that adding plausible context to a genuine location
effect does not automatically improve it. They also prevent the original-normal cluster
or activity-combination observations from becoming the next rabbit hole.

### Current interpretation in plain language

The most defensible theory is now narrower and clearer:

1. A calculated area can mark **where trading is likely to become busier**.
2. The broader market trend, recent price path, pressure, and external state may still
   determine **which way that busy trading resolves**.
3. The level itself does not yet tell us whether price will reject, break through, or
   oscillate around it.
4. Some apparent confluence is merely multiple labels describing the same busy location;
   it receives no credit until a clean component adds information on new dates.

The location could attract resting orders, or active markets could simply be more likely
to reach widely watched levels and then stay active. This test cannot distinguish those
causal explanations. For practical research, that distinction is not yet necessary: the
level may still be useful as an activity alert, but a separate direction model is needed.

Generation 15 did not test the user's `55%` joint target. That target means the same
unseen decision must correctly identify both that a meaningful reaction occurs and its
direction. Reaction-only accuracy, a positive volume difference, or frequent crossings
cannot satisfy it.

### Generation 15 evidence and next batch boundary

- Frozen seven-sibling plan:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combinations/g15_broad_direct_freeze_20260822a.json`.
- Joint review:
  `g13_broad_siblings/g14_broad_combinations/g15_broad_direct/joint_review/`
  `g15_joint_review_20260822a/g15_joint_review.json`.
- Matched path details and one-coin-dominance audit are in the joint review's
  `matched_level_leads.csv` and `matched_scores_with_dominance.csv`.
- Context failures and group checks are in `context_cell_leads.csv` and
  `context_group_leads.csv`.

The next main batch should remain direction-neutral and broad. It should determine
whether the calculated area adds anything after completed contact-candle activity is
matched, challenge level-density and chart-coverage explanations, compare families and
timeframes head to head at actual contacts, separate persistent traffic from genuinely
new post-contact activity, and test whether local or wider-market regimes change those
unsigned behaviours. Separately, this completed pattern may seed one bounded,
evidence-triggered `1m` replay using a frozen representative episode queue. Only that
limited lane may examine direction under the current objective. Both sibling sets must
be frozen before outcomes, and no result may receive a descendant until the full batch
review is complete.

## 2026-08-22 - Generation 16 completed-contact attribution, FreqAI density, and bounded direction replay

Generation 16 asked whether the Generation 15 activity-location result remained useful
after harder explanations were controlled. Seven main questions and one bounded
direction lane were frozen together before the new outcomes were opened:

1. Does the calculated area add anything after the completed contact candle's volume,
   range, and absolute pressure are matched?
2. Does a current calculated coordinate outperform causal `72h`-old, price-shifted, and
   ordinary-time coordinates?
3. Does activity start after contact, or was it already underway before and during the
   contact candle?
4. Does any retained level family or source timeframe beat other genuine contacts under
   the same market state?
5. Is the result explained by level density, chart coverage, or overlapping levels?
6. Do local activity, wider-crypto activity, or completed `8h` activity add information,
   alone or in combinations that beat every simpler component?
7. Does FreqAI gain unseen predictive information when current level, density, or context
   blocks are added to completed contact-candle activity?
8. On a separate frozen set of `12` distinct-pair episodes, can causal `1m` information
   predict both that a reaction occurs and its direction?

The main direction-neutral surface used the ten established coins and the frozen ten
most-traded meme coins. It retained the five Generation 15 families that had strict
activity evidence: generic prior ranges, round numbers, explicit-prior volume-profile
levels, volume-profile nodes, and settled volume-profile levels. The retained source
timeframes were `1h`, `4h`, and `8h`, and outcomes were measured over `1h`, `2h`, and
`4h`. Profit, entries, exits, and future signed direction were absent from the main
surface.

### Direct tests after matching the completed contact candle

The direct run completed all `20/20` pair tasks with no failure. It produced `44,664`
pair-level contrast summaries. Matching used only information known by the end of the
contact candle:

- pre-contact relative volume and volume acceleration;
- pre-contact absolute pressure and pressure persistence;
- ATR, prior range, Bollinger width, and range contraction;
- completed contact-candle volume ratio;
- completed contact-candle range ratio; and
- completed contact-candle absolute pressure change.

Controls were genuine near misses, matched ordinary times, causal `72h`-old levels, and
deterministically price-shifted levels. A strict result had to repeat in both halves of
the original normal chronology and both halves of the April-August 2026 normal
chronology, retain enough coins, keep weekly uncertainty on one side of zero, and avoid
one coin contributing more than half of the equal-coin effect.

After all required comparisons were collapsed into whole questions, five broad normal-
coin results remained strict:

- more crossings through the calculated area within `1h`;
- more crossings within `2h`;
- more crossings within `4h`;
- more direction-neutral price-plus-volume reaction within `1h`; and
- more future volume within `2h`.

For crossings, the paired equal-coin differences were approximately:

- `1h`: `+0.13` to `+0.14` crossings versus ordinary times and `+0.21` to `+0.22`
  versus genuine near misses;
- `2h`: `+0.12` to `+0.16` versus ordinary times and `+0.27` to `+0.29` versus near
  misses; and
- `4h`: `+0.11` to `+0.16` versus ordinary times and `+0.36` to `+0.39` versus near
  misses.

The `1h` direction-neutral reaction difference was approximately `+4.5` to `+5.3`
percentage points versus ordinary times and `+7.9` to `+10.3` points versus near misses.
The `2h` future-volume-ratio difference was approximately `+0.08` to `+0.11` versus
ordinary times and `+0.14` to `+0.20` versus near misses. These are changes in measured
market behaviour, not profit percentages.

The crossing result repeated inside BTC, the smart-contract-platform group, and the
other-established-alt group at several horizons. BTC remains a separate asset-specific
claim, while the two similar-coin groups are valid subgroup evidence. The direct meme
surface did not complete a whole-question lead under the same gate, so it is not merged
into the normal claim.

The current coordinate also beat the stale, shifted, and ordinary controls for several
crossing, reaction, and volume questions at the point level, but none of those complete
three-control questions was strict across weekly uncertainty. They remain supporting
leads, not proof that the exact present coordinate is uniquely causal.

### What the direct tests ruled out

No retained family or source timeframe beat the other genuine contacts in both normal
chronologies. Earlier family-specific pass counts therefore must not be interpreted as a
head-to-head ranking, and a higher source timeframe still receives no automatic
precedence.

No pre-contact-to-post-contact change question survived all of its controls across both
normal chronologies. This means the tests cannot say that contact *started* the activity.
The safer explanation is that the calculated area locates where already-active trading
often continues or repeatedly interacts.

Simple high-density-versus-isolated contrasts produced several point-level normal
results but no strict result. No direct local, wider-market, completed-`8h`, or combined
context question became strict after every immediate component comparison. This avoids
turning one threshold or one standard-period context result into a rabbit hole.

### What FreqAI added

The FreqAI support cache contained `40,090` retained normal contact rows and `7,854`
retained meme contact rows. Eight frozen profiles were run in each of three cells:

- original normal validation;
- later April-August 2026 normal chronology; and
- frozen top-ten meme validation.

All `24/24` full model commands completed with identical prediction keys and no duplicate
prediction rows. Every profile started with completed contact-candle volume, range, and
absolute pressure. Separate profiles then added current level identity, continuous level
density plus cluster geometry, wider-crypto state, completed `8h` activity, a restrained
level-plus-wider combination, causal `72h`-old blocks, or within-period shuffled blocks.

The strongest new result was **continuous level density plus cluster geometry for future
crossing count**. It was strict in all three cells at `1h`, `2h`, and `4h`:

- `1h`: average relative error reduction was about `5.8%` in original normal, `6.9%`
  in later normal, and `5.2%` in memes; all `60/60` coin-period rows improved;
- `2h`: average reduction was about `6.5%`, `5.1%`, and `5.0%`; `59/60` rows improved;
  and
- `4h`: average reduction was about `3.8%`, `3.5%`, and `2.6%`; `54/60` rows improved.

This explains why a binary high-density threshold could look unstable while the model
still found useful information: the useful relationship is probably gradual and
multi-part. The combined block includes both the number/concentration of nearby levels
and cluster geometry. It does not yet prove which component matters, so decomposition is
the first queued branch.

Three other FreqAI results completed the looser all-three-cell point ladder:

- completed `8h` activity improved the estimate of a direction-neutral reaction within
  `4h`;
- current family/timeframe identity gave a very small improvement for `1h` crossings;
  and
- wider-crypto state improved `4h` crossing estimates.

For the completed-`8h` reaction profile, reaction-only accuracy ranged from about
`61.9%` to `68.4%` across six validation periods. Balanced accuracy was about
`61.4-64.7%`, rank AUC was about `0.64-0.71`, and raw accuracy was `4.8-8.7` percentage
points above the majority guess. This is useful enough to investigate again, but it is
not a strict incremental result and it says nothing about up versus down. The complete
level-plus-wider model did not beat every simpler component.

### Bounded `1m` direction result

The separate `1m` lane used `12` episodes from `12` distinct pairs, including eight
normal and four meme episodes. It tested `14` causal methods at `15m`, `60m`, `240m`, and
`720m`, including pre-contact trend, pressure, contact-candle pressure, rolling pressure,
pressure acceleration, multi-timeframe votes, timestamp-safe order book where available,
and a non-deployable cohort comparator.

The best causal method was contact-candle pressure at `15m`. It correctly called both a
reaction and its direction in `4/12` episodes, or `33.3%`. Its direction was correct in
`5/10` callable cases overall. Zero methods reached the user's `55%` joint floor, and the
sample was too small for a repeatable lead. Five episodes indicated that the frozen
analysis window should be extended when the bounded lane is revisited.

### Current conclusion and complete next branch layer

Generation 16 strengthens one narrow theory:

1. causal calculated areas can identify where price is likely to remain busy or recross;
2. continuous density and geometry of nearby calculated areas improve estimates of how
   much repeated crossing occurs;
3. completed `8h` activity may help estimate whether a direction-neutral reaction occurs;
4. neither the level nor the tested lower-timeframe methods currently predict the
   direction reliably; and
5. no result is a trading rule, profit claim, or promotion candidate.

Only after every Generation 16 sibling was terminal, six Generation 17 branch batches
were queued together:

1. decompose continuous density from individual cluster-geometry components;
2. distinguish useful rejection, traversal, repeated recrossing, dwell, and false-break
   behaviour rather than treating every crossing as equivalent;
3. calibrate reaction probability from completed `8h`, local, wider-market, and density
   components with full ablations;
4. test additional rational and causal level calculations, including single levels and
   clusters equally, without event-specific tuning;
5. test news, global-market, and order-book quiet/aligned regimes only where source
   readiness and missing-source placebos permit; and
6. enlarge the bounded `1m` lane around robust reaction candidates, counting abstentions
   as failures and retaining `55%` joint success as the minimum lead and `65%` as the
   target.

These six batches form one branch layer. They must all complete or be honestly parked
before any one result spawns a deeper descendant.

### Generation 16 evidence

- Joint review:
  `generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
  `g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/`
  `joint_review/g16_joint_review_20260822a/g16_joint_review.json`.
- Whole direct-question decisions and the six-batch branch queue are beside that joint
  review.
- Full FreqAI cell and joint results are under the sibling `freqai/` folder.
- The bounded replay result is under
  `one_minute_direct/g16_one_minute_direct_20260822a/`.

## 2026-08-23 - Generation 17 density, path form, rational levels, context, and `1m` replay

Generation 17 completed every sibling in its frozen batch before interpreting one result
as a reason for more testing. The common market question was whether calculated price
areas carry repeatable information about activity near them, while deliberately keeping
up/down prediction outside the main surface.

### Inputs and comparisons

The batch covered ten established coins and the frozen ten most-traded meme coins. DOGE
appears in both separately labelled cohorts; it was not double-counted as two distinct
symbols. Main outcomes used `1h`, `2h`, and `4h` horizons. The separate `1m` microscope
used `15m`, `60m`, `240m`, and `720m` horizons.

The inputs included:

1. contact state, completed-contact volume, range, and absolute pressure;
2. the number, distance, width, overlap, and independent-family composition of nearby
   calculated areas;
3. completed `8h` activity, local OHLCV/indicator state, and wider-crypto state;
4. adaptive Volume Profile nodes over `72h`, `168h`, and `720h`, with fixed and
   Freedman-Diaconis bin counts;
5. rolling VWAP and one/two-deviation bands, rolling high/low boundaries, weekly pivots,
   and generic moving-average/Bollinger comparison levels;
6. actual contacts, matched ordinary times, genuine near misses, causal `72h`-old
   levels, and deterministic price-shift controls;
7. source-ready orderbook rows plus stale and shuffled equivalents; and
8. causal `1m` trend, pressure, RSI, MACD, directional movement, CMF, OBV, Bollinger,
   range, density-interaction, approach, and orderbook methods.

Profit, entries, exits, and signed future direction were absent from the main level and
reaction batches.

### What repeated

FreqAI found that continuous location geometry adds information about future crossing
traffic. Current density beat a shuffled copy and a causal `72h`-old copy for crossing
count over `2h`. Proximity and zone-width information added to the simpler contact
description for crossing count over `1h` and `2h`. The effect repeated in original
normal, later normal, and meme evaluation cells. It says that the arrangement of nearby
areas helps estimate how busy the location will be; it does not say whether price rises
or falls.

Direct comparisons gave the same broad physical picture. Actual contacts were more
likely than ordinary times and genuine near misses to cross the area again. Any recross
repeated at `1h` and `2h` across all normal subgroups and memes and remained broadly
present at `4h`. Repeated recrossing over `2h` survived in the all-normal and meme
scopes. Meme contacts also had more one-sided breakthrough at `1h`, `2h`, and `4h`.
Breakthrough here only means price travelled through the zone on the approach path; it
is not a forecast made before contact.

The rational atlas calculated `113` causal level definitions and produced about
`10.35 million` source/control event rows. Its clearest portable family result was:

1. adaptive Volume Profile nodes located extra crossing traffic at `1h` and `2h`, with
   weaker but still repeated support at `4h`; and
2. rolling high/low boundaries located more unsigned price-plus-volume reaction over
   `1h`, `2h`, and `4h`.

Several neighbouring `72h` VP definitions survived—especially 48/96-bin high-volume
nodes and 48/96-bin low-volume nodes—so the lead is not just one isolated winning number.
The atlas reported `303` strict rows, but those rows reuse related level variants,
metrics, horizons, and market scopes. They are not `303` independent edges. Bitcoin
alone accounts for many strict cells and remains a separate market-specific claim.

### What did not repeat or could not be tested fairly

News had no timestamp-ready rows on the required surface, and no compatible historical
non-crypto global-market feature block existed. Those routes were parked as coverage
gaps rather than pretending missing data meant quiet conditions. Source-ready orderbook
rows did not establish a broad retained addition beyond level/density information.

The `40`-episode `1m` replay found no usable direction method. The best same-call result,
completed-contact pressure over `15m`, got both the reaction and direction correct in
`14/40 = 35%` of all frozen episodes. Direction was correct in `14/21 = 66.7%` only
after excluding episodes that did not react or did not produce a callable case. That
conditional subset does not meet the user's `55%` joint target and is not retained.

### Interpretation and next whole batch

The evidence supports a probabilistic location claim: some calculated areas identify
places where trading is more likely to remain busy, cross repeatedly, or show an
unsigned reaction. It does not yet support bounce, breakout, final direction, causation,
profit, or a trading action.

The next batch stays broad and treats these as siblings:

1. confirm the density/proximity relationship on new time blocks and with components
   removed one at a time;
2. confirm whether recross, repeated recross, traversal, rejection, breakthrough, and
   dwell remain physically distinct after strong controls;
3. confirm adaptive VP and rolling-boundary families on later chronology and neighbouring
   rational settings rather than selecting a remembered event or single best parameter;
4. test quiet, trending, and choppy wider-market regimes plus genuinely source-ready
   orderbook state against level-only and regime-only descriptions; and
5. test transfer across BTC separately, similar established-coin groups, memes, and
   `1h`/`4h`/`8h` source calculations.

The direction descendant remains parked because no current method reached `55%`. No
early member of the next batch may create another descendant until all siblings are
terminal and jointly reviewed.

Generation 17 evidence:
`generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
`g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/`
`g17_broad_branch_layer/joint_review/g17_joint_review_20260823a/g17_joint_review.json`.

## 2026-08-23 - Generation 18 later-block and multi-timeframe confirmation

Generation 18 did not start by choosing the largest Generation 17 score. It froze five
different confirmation routes together—density, physical path form, exact level source,
market/indicator context, and coin/timeframe transfer—and kept direction parked. All
five active routes finished before their results were combined.

### Data and controls

The direct confirmation used two later time blocks for each cohort:

1. normal coins: 20 July to 5 August 2026, then 5 August to 20 August 2026;
2. meme coins: 14 July to 29 July 2026, then 29 July to 13 August 2026.

The same ten established pairs and frozen ten most-traded meme pairs were retained.
Main outcomes were measured over `1h`, `2h`, and `4h`. Actual contacts were compared
with matched ordinary times, genuine near misses, causal `72h`-old levels, and
deterministically shifted price coordinates where the question required them.

The separate timeframe test causally rebuilt eight compact definitions on completed
`1h`, `4h`, and `8h` source candles:

1. a 72-source-bar/48-bin nearest high-volume node;
2. a 72-source-bar/48-bin nearest low-volume node;
3. 24- and 72-source-bar rolling highs and lows; and
4. 72-source-bar rolling VWAP plus or minus two volume-weighted deviations.

Every source value became available only after its source candle closed. A higher
timeframe was never presumed better.

### Strongest repeated behaviour: price revisits busy areas

The most consistent result was not a bounce or breakout. It was that price crossed the
area again.

For adaptive Volume Profile areas and for the combined adaptive-VP/rolling-high-low
surface:

1. any recross repeated over `1h`, `2h`, and `4h` in both normal coins and memes;
2. repeated recross repeated over `2h` and `4h` in both groups;
3. the minimum equal-coin any-recross difference against ordinary time and near miss was
   roughly `+15` to `+30` percentage points, depending on surface and horizon; and
4. the minimum repeated-recross difference was roughly `+5` to `+8` percentage points.

This means the calculated areas behave like busy junctions or acceptance locations more
often than nearby almost-touched or ordinary locations. It does not tell us which side
price will leave on.

### Exact coordinate was less portable than the broad area effect

The broad recross question used ordinary-time and near-miss controls. When the test also
required the current coordinate to beat causal stale and price-shifted versions, the
evidence became much narrower:

1. BTC retained adaptive-VP crossing effects at `1h` and `2h`;
2. the meme group retained more unsigned reaction over `4h` at rolling highs/lows; and
3. one exact `72h`/96-bin nearest low-volume-node setting retained extra `1h` crossings
   in memes.

The neighbouring VP settings did not confirm as one cross-market plateau. The correct
interpretation is that calculated active areas robustly locate repeated traffic, while
the unique value of one exact coordinate or parameter remains conditional.

### Wider market and ordinary indicators added small activity clues

Each context comparison asked whether the extra level effect in a high context state was
larger than in a low state, after the same context change at the control locations was
subtracted.

Two non-BTC-only relationships repeated through both later blocks:

1. Across normal coins, high cross-coin return dispersion increased the extra future
   volume response around the area over `2h` by at least about `0.106` volume-ratio units
   across required controls and periods.
2. Inside the predeclared smart-contract-platform group, high completed local
   volume/range activity increased the extra `1h` volume response by at least about
   `0.142` volume-ratio units.

Other strict rows—Bollinger-width state, RSI extremes, local activity, and wider-crypto
activity—were mostly BTC-specific. These findings fit the theory that wider or recent
market state helps determine how active the response becomes while a technical area
helps locate it. They still do not predict direction.

### No higher-timeframe precedence and no cluster superiority

Some `4h` or `8h` source contacts had larger outcomes than `1h` contacts, and some
cross-timeframe clusters had larger outcomes than isolated same-family contacts. Those
are incomplete comparisons by themselves.

The joint audit required the same candidate to also beat ordinary, near-miss, stale, and
shifted coordinates. Results were:

1. higher-timeframe rows that beat both `1h` and every location control: `0`;
2. cluster rows that beat both isolated contacts and every location control: `0`.

Therefore higher timeframe does not receive precedence, and the tested clusters are not
better than single levels under the full evidence ladder. This does not invalidate all
clusters; it only rejects a superiority claim for this frozen surface.

### Density, external sources, and direction

The gradual density relationship did not confirm broadly on these later blocks. Only
BTC proximity to the calculated area completed the density/control ladder for `1h` and
`2h` crossings.

Orderbook had no timestamp-ready observations spanning both later blocks. News and
non-crypto global-market histories also lacked the required aligned surface, so they
remained unavailable rather than being treated as quiet or neutral.

No new direction test ran. Generation 17's best same-call reaction-plus-direction result
was `35%`, so another signed search would not have been evidence-led. The `55%` floor and
`65%` main target remain unmet.

### Next complete branch layer

Five active Generation 19 siblings are queued together:

1. test whether exact current coordinates specifically explain recrossing after every
   stale, shifted, ordinary, and near-miss control uses the same path definition;
2. compare level-only, context-only, and level-plus-context models for crypto dispersion
   and completed local activity;
3. confirm the separate meme rolling-boundary/LVN, established-coin rolling-VWAP, and
   BTC-VP mechanisms under their exact market labels;
4. distinguish newly started recrossing from activity already underway before contact;
5. test one compact causal repeated-acceptance or event-anchored VWAP area against the
   existing rolling VWAP and controls.

External context remains queued but coverage-parked. Direction remains evidence-parked.
All Generation 19 siblings must be terminal or honestly parked before any descendant.

Generation 18 evidence:
`generation11_review/generation11_branches/g12_chronology_attribution_and_combinations/`
`g13_broad_siblings/g14_broad_combinations/g15_broad_direct/g16_broad_attribution/`
`g17_broad_branch_layer/g18_broad_confirmation/joint_review/`
`g18_joint_review_20260823a/g18_joint_review.json`.
