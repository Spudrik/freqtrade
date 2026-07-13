---
doc_status: active
default_read: routed
owner: user
purpose: Static master objectives and route map.
do_not_use_for: Detailed progress logging or evidence dumps.
last_rebuilt: 2026-06-11
---

# Master Objectives

## Objective 1 - Dry-Run Preparation / Startup Readiness

Prepare selected strategy/config/files so the user can launch a dry run and perform startup/smoke checks.

Status: retired from active guidance. Do not treat this as the current objective unless the user reopens it.

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

- Sieve is the only approved Hyperopt/discovery system unless the user explicitly approves a non-Sieve exception,
- many independent entry families may coexist,
- each entry family should have its own objective, invalidation, and likely target logic where possible,
- each trade should retain enough state to know why it entered, what target/invalidation applies, and what later evidence changed,
- rare high-success custom pattern entries should be preserved as entry/add/exit evidence,
- same-direction stacked signals may support add/hold/confidence,
- opposite-direction signals may support reduce/tighten/exit,
- crash/risk-off states should usually tighten stops or reduce exposure rather than act like generic broad exits,
- orderbook/news/context should reinforce or challenge entries only when the data is proven ready and explicitly routed into Sieve work.

Primary objective file:

- `../01_objectives/objective_02_master_strategy_architecture.md`

### Objective 2a - Strategy Refinement

Refine existing high-potential Sieve candidates without obsessing over a single current best.

Primary objective file:

- `../01_objectives/objective_02a_strategy_refinement.md`

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

`objective_current.md` is temporary and should be rewritten at stage boundaries. This master file should be changed rarely and only when the user changes the long-term direction.
