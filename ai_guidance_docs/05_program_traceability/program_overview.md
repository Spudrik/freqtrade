---
doc_status: active
default_read: no
owner: user+agent
purpose: Compact programme-level objective, milestone, workstream, and roadmap traceability.
do_not_use_for: Current task authority, subsystem implementation rules, raw results, or source-readiness details.
last_rebuilt: 2026-07-13
---

# Programme Overview

## 1. Authority

This file is a non-governing traceability point. Start actual work from `../00_project_control/objective_current.md` and its routed documents. Use this overview to understand how the active objective fits the wider programme, then read only the relevant track.

## 2. Long-term objective

Build one Freqtrade master strategy that:

1. retains multiple independently tested entry families;
2. assigns each family an evidence-based prior weight and its own target, invalidation, guards, and exit roles;
3. combines active candle-level evidence without double-counting correlated variants;
4. uses timestamp-safe FreqAI estimates as a bounded risk/weighting layer rather than an unexplained entry replacement;
5. converts the combined state into enter/skip, initial size, later leverage, add/reduce, stop, partial, runner, and exit decisions; and
6. manages price zones and evolving trade state rather than assuming an exact buy-at-X/sell-at-Y path.

## 3. Current programme snapshot

- **Active stage:** Sieve3 exit development with selected entry parameters locked.
- **Immediate question:** does the refined target-zone/invalidation/layered design encode more coherent and effective exits than the older generic Sieve3 branches?
- **Decision consequence:** if the refined design passes matched evidence and execution review, untested legacy exits become bounded regeneration/rework candidates rather than automatically continuing unchanged.
- **Entry-position gap:** later same-direction signal adds were tested, but a systematic structural top/middle/bottom entry-zone ladder was not found.
- **Pattern lane:** pattern entries remain logically separate because exact conditions are rare and need family-specific coverage and exit contracts.
- **FreqAI/data position:** useful correlations exist, but source history and continuous timestamp-safe overlap remain too short or inconsistent for a dependable final risk layer.
- **Integration position:** entry registry, evidence weights, exit-role selection, staggered-entry validation, and dynamic risk integration remain unfinished.

## 4. Governing objective map

| Document | Relationship to this programme | Authority |
|---|---|---|
| `../00_project_control/objective_current.md` | Current Sieve3 exit-development task contract. | Governing and always current. |
| `../00_project_control/objectives_master.md` | Static map of the wider Sieve-first programme. | Governing when stage or scope is unclear/changing. |
| `../01_objectives/objective_02_master_strategy_architecture.md` | Long-term multi-entry, multi-source integration architecture. | Routed; governing when integration is active. |
| `../01_objectives/objective_02a_strategy_refinement.md` | Candidate refinement objective. | Routed; use only when explicitly active. |
| `../01_objectives/objective_02c_orderbook_confluence.md` | Orderbook confirmation/risk objective. | Parked by the current objective unless reopened. |
| `../01_objectives/objective_02d_news_context_integration.md` | News/context integration objective. | Parked unless source readiness and the user reopen it. |
| `../01_objectives/objective_02e_generic_ta_integration.md` | Generic-TA side-lane objective. | Routed/parked; not the current discovery lane. |

Traceability does not declare an objective obsolete. If an older objective appears inconsistent, mark it `needs reconciliation` here and ask before editing the objective itself.

## 5. Workstream map

| Track | Current state | Next material milestone |
|---|---|---|
| Sieve entry/exit | Refined Sieve3 exit validation active; broad legacy testing incomplete. | Complete comparison and decide continue versus regenerate/rework. |
| Core indicators | Structure/level vocabulary is usable; some reference/code drift may exist. | Record only a material indicator redesign or completed new contract. |
| Pattern indicators | Separate rare-signal lane with family-specific rules and sparse evidence. | Validate adequate pair/window coverage and coherent pattern exits. |
| FreqAI risk/weighting | Several trader-readable leads; no promotion-grade continuous evidence. | Reopen with named hypotheses and clean temporal/cross-pair controls. |
| Historical data | Large local repository is partial; historical gaps constrain clean blocks. | Produce a materially improved continuous, timestamp-safe block. |
| Live data | Collectors/snapshots exist; history and strict multi-source overlap are short. | Accumulate and validate longer live continuity. |
| Master integration | Architecture defined; evidence registry and weighting are not built. | Consolidate retained entries/exits after Sieve3 design settles. |

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

## 7. Cross-track roadmap

1. Settle the refined Sieve3 exit-design question.
2. Build the compact entry/exit evidence view and retain genuinely distinct exit roles.
3. Consolidate the retained entry registry and measure overlap.
4. Test structural price-zone staggering against one-shot and later-signal adds.
5. Improve historical and live data continuity while preserving explicit gaps/freshness.
6. Revalidate a small set of trader-readable FreqAI risk hypotheses.
7. Define static entry weights, bounded dynamic risk multipliers, and stateful position management.
8. Assemble and ablate the final master strategy before any dry-run freeze.

## 8. Track links

- Sieve entry/exit: `tracks/sieve_entry_exit.md`
- Core indicators: `tracks/indicators_core.md`
- Pattern indicators: `tracks/indicators_patterns.md`
- FreqAI risk/weighting: `tracks/freqai_risk_weighting.md`
- Historical data: `tracks/data_historical.md`
- Live data: `tracks/data_live.md`
- Master integration: `tracks/master_integration.md`

