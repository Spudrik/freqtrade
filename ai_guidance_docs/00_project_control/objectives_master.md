---
doc_status: active
default_read: routed
owner: user
purpose: Static master objectives and route map.
do_not_use_for: Detailed progress logging or evidence dumps.
last_rebuilt: 2026-09-28
---

# Master Objectives

## Objective 1 - Dry-Run Preparation / Startup Readiness

Prepare selected strategy/config/files so the user can launch a dry run and perform startup/smoke checks.

Status: retired from active guidance. Do not reactivate it unless the user explicitly reopens it.

Scope:

- confirm the selected strategy file exists,
- confirm imports/classes/config references are correct,
- confirm dry-run configuration is syntactically valid if a config is supplied,
- confirm no live/API assumptions are introduced by the agent,
- produce startup/readiness notes only when requested.

Non-scope:

- do not launch or monitor the dry run unless explicitly asked,
- do not create pass/fail rules for dry-run performance unless the user defines them,
- do not invent leverage, stake, balance, pair, or risk limits,
- do not replace the selected strategy with a different candidate because it looks like current best.

## Objective 2 - Master Strategy Architecture

Build toward a flexible Sieve-first multi-entry, multi-source strategy system where entries, exits, adds, reductions, stop tightening, and risk/leverage decisions are driven by confluence and conflict between independent signals.

The long-term aim is not one narrow winner. The aim is a pool of high-potential entry families and risk signals that can be combined into a master strategy.

Required ideas:

- Sieve is the only approved Hyperopt system for trading entry/exit candidate discovery unless the user explicitly approves a non-Sieve exception; Objective 02b reaction-zone and bounded event-direction research is a separate direct-test/FreqAI lane,
- many independent entry families may coexist,
- each entry family should have its own objective, invalidation, and likely target logic where possible,
- each trade should retain enough state to know why it entered, what target/invalidation applies, and what later evidence changed,
- rare high-success custom pattern entries should be preserved as entry/add/exit evidence,
- same-direction stacked signals may support add/hold/confidence,
- opposite-direction signals may support reduce/tighten/exit,
- crash/risk-off states should usually tighten stops or reduce exposure rather than act like generic broad exits,
- orderbook/news/context may reinforce or challenge entries only when the data is proven ready and explicitly routed into later entry integration; separately, Objective 02b may use proven source blocks for its approved research questions.

Primary objective file:

- `../01_objectives/objective_02_master_strategy_architecture.md`

### Objective 2a - Strategy Refinement

Refine existing high-potential Sieve candidates without obsessing over a single current best.

Primary objective file:

- `../01_objectives/objective_02a_strategy_refinement.md`

### Objective 2b - Market Reaction And Event-Driven Direction Discovery

Status: historical research reference; superseded as the active work programme by the user's 26 September 2026 bounded paper-trial instruction. Preserve its evidence and unanswered questions, but do not resume its open-ended batch/branch queue without a new request.

Start from causal OHLCV, existing custom and generic reference levels, rational new
level-producing indicators, and clusters across indicators and timeframes. Preserve the
reaction-zone work, then test the newly authorized hierarchy from slow background and
major events through BTC/ETH/broad-market confirmation, established-alt and meme-group
response, coin-local technical or liquidity modification, and post-event range formation.

Sieve3 entries and exits are not prerequisites. Result-inspired indicators, maths,
timeframes, level/cluster definitions, coins, and contextual ideas enter a rolling
breadth-first queue so the frozen initial and generation batches finish before later
branches launch. At least five materially distinct investigation routes must receive
fair tests, and up to five evidence-justified branch layers are authorized without any
requirement to invent all five. The current event hierarchy proceeds through honest
event reconstruction, individual links, frozen pairs, limited three-block chains and
range work, then untouched whole-event or later live confirmation. Meme-coin work uses
a frozen cohort of the ten most traded eligible meme coins.

On 3 September 2026 the user authorized bounded event-scoped signed-direction research
inside this objective. It separates scheduled-event anticipation, unexpected-event
detection, market leadership, group transmission, meme amplification relative to each
coin's normal behaviour, local level/liquidity effects, and later range settlement.
Whole events rather than random candles are the independent confirmation units. The
three already-frozen September reaction/activity confirmations remain unchanged while
outcome-blind event and source reconstruction may proceed.

Unrestricted every-candle direction fishing, entry/exit selection, profit optimization,
position sizing, leverage, and live/dry-run changes remain deferred. The research aims
for several complementary indications of direction, activity, location, timing or
conditional modification. The 19 September 2026 user revision removes the blanket
joint activity-and-direction pass requirement; Objective 02b Section 15.4 governs
separate component assessment and later combination tests, retaining coverage,
controls, abstention, uncertainty and chronological confirmation. FreqAI is
used as a controlled incremental research tool after direct baselines. News, web,
global-context, and orderbook sources may be used only under their real timestamp-safe
coverage.

Primary objective file:

- `../01_objectives/objective_02b_market_reaction_zone_discovery.md`

## Objective 3 - Comparative And Discretionary Paper Trading

Status: current active stage, user-authorized 26 September and extended 28 September 2026. Develop and operate actively trading, paper-only approaches using retained research leads and usable free/current information; stop expanding the old open-ended research queue. The active set is IntegratedPaper auto, its automatic-with-manual-adjustments sibling, V01/V10, FastPivot, LeaderImpulse, and two isolated fully manual accounts. The original A-F definitions are retired, not an execution queue.

Keep automated rules frozen except explicitly approved defect repairs. The main agent manages the two discretionary accounts using sourced news/global-market context, BTC/ETH confirmation and local multi-timeframe levels/volume/orderbook where coverage permits. Luna provides bounded briefs and proposes extra announcement checks; the main agent approves scheduling and alone makes journalled paper trade decisions. The user delegates paper risk selection; never transfer that discretion to live trading.

Aim for useful returns after fees/funding while assessing drawdown, exposure, decision quality and operational reliability. Preserve losses, complementary ideas and the distinction between prediction, confirmation, location and risk information. Compare matched opportunities where possible rather than racing unlike accounts by headline profit. Review weekly and after the initial eight-week observation window, then recommend keep/revise/park decisions and a bounded next phase. No automatic research promotion, perpetual branching, erased failed accounts or live orders. The detailed current contract is Objective 03; changing run facts remain in its existing run record.

Primary objective file:

- `../01_objectives/objective_03_comparative_paper_trial.md`

### Objective 2c - Orderbook Confluence

Use orderbook as confirmation, contradiction, risk acceleration, invalidation, or target-zone evidence. Do not use it as a blunt generic filter unless tests prove that role.

Primary objective file:

- `../01_objectives/objective_02c_orderbook_confluence.md`

### Objective 2d - News / Context Integration

News/GDELT/GKG/web/global/context sources remain a long-term goal, but are parked until the user says the data is ready or a source-readiness report proves a specific window/source block is usable.

Primary objective file:

- `../01_objectives/objective_02d_news_context_integration.md`

### Objective 2e - Generic TA Integration

Generic TA discovery is a side lane. Track useful ideas for later integration, but do not let it replace custom indicators, sieve-derived families, or the master confluence architecture.

Primary objective file:

- `../01_objectives/objective_02e_generic_ta_integration.md`

## Objective Maintenance Rule

There is no temporary current-objective file. This master file should be changed rarely and only when the user changes the long-term programme direction.
