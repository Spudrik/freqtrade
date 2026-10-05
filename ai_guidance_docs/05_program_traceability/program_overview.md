---
doc_status: active
default_read: no
owner: user+agent
purpose: Compact programme-level objective, milestone, workstream, and roadmap traceability.
do_not_use_for: Current task authority, subsystem implementation rules, raw results, or source-readiness details.
last_rebuilt: 2026-10-05
---

# Programme Overview

## 1. Authority

This file is a non-governing traceability point. Start actual work from the repo-root `AGENTS.md` and the user's latest explicit request. Use this overview only to understand how the requested task fits the wider programme, then read the relevant track.

## 2. Long-term objective

Build one Freqtrade master strategy that:

1. retains multiple independently tested entry families;
2. assigns each family an evidence-based prior weight and its own target, invalidation, guards, and exit roles;
3. combines active candle-level evidence without double-counting correlated variants;
4. uses timestamp-safe FreqAI estimates as a bounded risk/weighting layer rather than an unexplained entry replacement;
5. converts the combined state into enter/skip, initial size, later leverage, add/reduce, stop, partial, runner, and exit decisions; and
6. manages price zones and evolving trade state rather than assuming an exact buy-at-X/sell-at-Y path.

## 3. Current programme snapshot

- **Active stage:** bounded comparative PAPER trial under Objective 03; PAPER is the primary evidence surface, not another open-ended research queue.
- **Snapshot:** 16 paper identities (9 active, 5 draining, 2 parked), including four distinct Sieve-derived entry sources. Recent startup/heartbeat activity is not proof of trade quality or profitability.
- **Evidence boundary:** the trial is still evaluating behavior; it does not establish durable profitability or authorize live trading.
- **Research disposition:** Sieve discovery, Objective 02b reaction/event-direction work, and FreqAI risk research are preserved as historical leads. Reopen only through explicit scope review and the applicable objective/rules.
- **Data and interpretation caveats:** short/uneven source coverage limits conclusions; activity/readiness is not direction or causation, and a downstream volume response is not a competing news cause.
- **Configuration recovery:** the four credential-bearing local configs remain excluded. Their `.example.json` templates preserve strategy/risk/schema fields with credential placeholders; copy a template to its expected local name and configure local credentials before use. Placeholders are not a launch guard, so the examples are not launch-ready. Recovery pins stay local and require approved review; do not launch clones blindly from historical run records.

## 4. Governing objective map

| Document | Relationship to this programme | Authority |
|---|---|---|
| Repo-root `AGENTS.md` plus the user's latest request | Active task routing and scope. | Governing for current work. |
| `../00_project_control/objectives_master.md` | Static map of the wider Sieve-first programme. | Governing when stage or scope is unclear/changing. |
| `../01_objectives/objective_02_master_strategy_architecture.md` | Long-term multi-entry, multi-source integration architecture. | Routed; governing when integration is active. |
| `../01_objectives/objective_02a_strategy_refinement.md` | Candidate refinement objective. | Routed; use only when explicitly active. |
| `../01_objectives/objective_02b_market_reaction_zone_discovery.md` | Historical reaction-zone and event-driven market-hierarchy research. | Preserve evidence; not the current execution queue. |
| `../01_objectives/objective_03_comparative_paper_trial.md` | Bounded comparative paper-trial stage. | Active and governing for the current stage. |
| `../01_objectives/objective_02c_orderbook_confluence.md` | Historical standalone orderbook confirmation/risk objective. | Any bounded paper context use follows active Objective 03 and source-specific readiness; Objective 02b is historical. |
| `../01_objectives/objective_02d_news_context_integration.md` | Historical standalone news/context integration objective. | Any bounded paper context use follows active Objective 03 and source-specific readiness; Objective 02b is historical. |
| `../01_objectives/objective_02e_generic_ta_integration.md` | Generic-TA side-lane objective. | Routed/parked; not the current discovery lane. |

Traceability does not declare an objective obsolete. If an older objective appears inconsistent, mark it `needs reconciliation` here and ask before editing the objective itself.

## 5. Workstream map

| Track | Current state | Next material milestone |
|---|---|---|
| Sieve entry/exit | Historical discovery/exit evidence retained; four distinct selected sources feed the paper comparison. | Use Objective 03 ledgers for current paper dispositions; no new batch by default. |
| Core and pattern indicators | Historical reaction-zone vocabulary and coverage limitations retained. | No active indicator redesign in the paper stage. |
| FreqAI risk/weighting | Historical leads only; audited caveats limit old headline/model claims. | Parked unless explicitly reopened under a named objective. |
| Historical and live data | Partial history and short/uneven external-source overlap constrain inference. | Keep quality/coverage limits attached to retained conclusions. |
| Master integration | Existing comparative PAPER combinations are authorized by Objective 03; the canonical full/live master remains unassembled and unapproved. | Keep canonical-master assembly parked pending separate approval. |

## 6. Major chronology

| Period | Programme milestone | Durable conclusion |
|---|---|---|
| Late April-mid May 2026 | Custom pivots, TLV2, VP, structure, and separate pattern families were iteratively developed. | Indicators became a reusable market vocabulary, not one universal signal. |
| 15-21 May | Sieve1 established breadth-first entry diagnostics. | Disposable probes can cheaply reveal entry shape but are not production strategies. |
| 21 May-12 June | Sieve2 refined, guarded, loosened, or reframed promising foundations. | Entry refinement is distinct from exit optimization; stage names are generation-based. |
| 23 May onward | Historical/live orderbook, news, GDELT/GKG, and FreqAI work expanded. | Continuous timestamp-safe source overlap is the main constraint. |
| 5-9 June | Same-direction signal stacking and broader position management were tested. | Selective adds can be strong; reserve sizing and broad reductions can damage total results. |
| 11 June | Competing non-Sieve discovery was retired. | Sieve owns discovery; FreqAI/context/orderbook remain later overlays. |
| 13 June-10 July | Large Sieve3 exit families were generated and partly tested. | Exit utility varies by source; generic naming does not prove coherent behaviour. |
| 11 July onward | Refined target/invalidation/layered exit comparison began. | This is a validation gate for the exit design and possible legacy regeneration. |
| 7 August | Post-Sieve3 evidence gate defined | A Sieve-linked event-reaction stage was defined and was later superseded by the market-first Objective 02b on 12 August. | Superseded programme stage. |
| 12 August | Market-first reaction-zone objective activated | Sieve entries/exits stopped being the prerequisite surface. Research now starts from causal indicator levels, level clusters, controls, and non-directional reaction measurements. | `../01_objectives/objective_02b_market_reaction_zone_discovery.md`. |
| 3 September | Event-driven market hierarchy activated | Existing reaction confirmations remain frozen; the next direction stage now tests slow background, major events, market leadership, coin-group transmission, local modifiers, and post-event ranges in five breadth-first layers. | `../01_objectives/objective_02b_market_reaction_zone_discovery.md`. |
| 26 September-5 October | Comparative PAPER trial became the primary stage | Four distinct Sieve-derived entry sources are compared in bounded paper operation. The 5 October snapshot records 16 identities (9 active, 5 draining, 2 parked); this is operational evidence, not a profitability claim. | `../01_objectives/objective_03_comparative_paper_trial.md`; `../04_results/results_recent_summary.md`. |

## 7. Cross-track roadmap

1. Keep the comparative paper trial bounded and use Objective 03 plus its linked ledgers as the current authority.
2. Preserve the 16-identity disposition (9 active, 5 draining, 2 parked) and the rule that startup/heartbeat activity is not a quality or profitability result.
3. Carry historical Sieve, reaction-zone, FreqAI, and context findings with their audit and source-coverage caveats; do not restart them as an unbounded queue.
4. Keep new canonical full/live master assembly and live trading outside this stage unless explicitly approved; the existing comparative PAPER combinations remain governed by Objective 03.

## 8. Track links

- Sieve entry/exit: `tracks/sieve_entry_exit.md`
- Current paper trial: `../01_objectives/objective_03_comparative_paper_trial.md`; `../04_results/results_recent_summary.md`
- Core indicators: `tracks/indicators_core.md`
- Pattern indicators: `tracks/indicators_patterns.md`
- FreqAI risk/weighting: `tracks/freqai_risk_weighting.md`
- Historical data: `tracks/data_historical.md`
- Live data: `tracks/data_live.md`
- Master integration: `tracks/master_integration.md`
