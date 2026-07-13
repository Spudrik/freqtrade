---
doc_status: active
default_read: no
owner: user
purpose: Conditional router and edit contract for the programme traceability documents.
do_not_use_for: Routine task context, current instructions, raw evidence, or detailed run reporting.
last_rebuilt: 2026-07-13
---

# Programme Traceability Router

## Role and precedence

This folder explains how the research programme evolved, where each workstream stands, and why later stages exist. It is a handoff and review layer, not a governing layer.

1. `../00_project_control/objective_current.md` remains the active objective.
2. Routed objectives, rules, status files, result ledgers, runtime JSONL, and backtest archives remain authoritative for their own content.
3. Traceability files summarize milestones and link to that evidence. They do not supersede it.
4. If traceability conflicts with a current routed document, follow the routed document and flag the traceability note for coordinator review.

## When to read this folder

Read traceability only when:

1. the user starts a new programme-level agent chat and needs orientation;
2. the user asks what has been done, what is active, or what remains;
3. a named research stage starts, finishes, reopens, is parked, or changes direction;
4. a cross-track decision requires the wider programme context; or
5. the current objective or coordinating agent explicitly routes here.

For orientation, read `program_overview.md` and then only the relevant track. Do not read every track. Routine implementation and individual batch work should normally skip this folder.

## Ownership

1. The user and the coordinating programme-level agent control this file and `program_overview.md`.
2. Specialist or lower-level agents must not edit the router or overview unless the user or coordinating agent explicitly assigns that edit.
3. A specialist agent may update its one routed track only when its task explicitly includes traceability maintenance and the milestone gate below passes.
4. Otherwise, the specialist should return a proposed one- or two-sentence milestone note to the coordinating agent.

## Milestone gate

Before changing a track, ask:

1. Did a named stage start, finish, reopen, or get parked?
2. Did a test series reach a conclusion that changes subsequent work?
3. Was an indicator or data subsystem materially completed, replaced, or redesigned?
4. Did source readiness materially change?
5. Did a concept move between demonstrated, promising, insufficient evidence, blocked, parked, or rejected?
6. Would a future agent make a materially wrong decision without this update?

If every answer is no, do not update traceability. A completed batch is not automatically a milestone.

## Minimal update format

For a valid milestone:

1. update the track's short `Current state` only if it changed;
2. add at most one compact milestone row: `date | milestone | conclusion/change | evidence`;
3. adjust only the affected outstanding task or classification;
4. link the canonical evidence instead of copying metrics or parameters; and
5. update `last_rebuilt`.

Do not add seed rows, batch narratives, parameter dumps, broad metric tables, generated reports, or duplicate result ledgers. When a track grows, compress old rows into a stage summary while preserving evidence links; do not create another history log without approval.

## Track routing

| Track | Read for |
|---|---|
| `tracks/sieve_entry_exit.md` | Sieve generations, entry/exit research, position adjustment, and refined exit decisions. |
| `tracks/indicators_core.md` | Core custom indicator development and material redesigns. |
| `tracks/indicators_patterns.md` | Rare geometry, reversal, continuation, multi-peak, and Wolfe pattern work. |
| `tracks/freqai_risk_weighting.md` | FreqAI labels, controls, risk estimates, and future dynamic weighting. |
| `tracks/data_historical.md` | Historical acquisition, backfills, extraction, gaps, and reproducible coverage. |
| `tracks/data_live.md` | Live collectors, freshness, outages, snapshots, and live overlap. |
| `tracks/master_integration.md` | Static entry weights, overlap, signal voting, trade state, and final integration. |

