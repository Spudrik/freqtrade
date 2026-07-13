---
doc_status: active
default_read: routed
owner: user+agent
purpose: Rules for keeping guidance docs clean.
do_not_use_for: Research methodology.
last_rebuilt: 2026-06-11
---

# Rules - Document Maintenance

## Status Header Required

Every active Markdown guidance file should start with:

- `doc_status`: active / parked / archive / append-only / generated
- `default_read`: yes / no / routed / code tasks
- `owner`: user / agent / user+agent / generated
- `purpose`: one sentence
- `do_not_use_for`: one sentence
- `last_rebuilt`: date

## Current Objective Lifecycle

`objective_current.md` is ephemeral. Replace it at stage boundaries. Do not append long run logs.

## Master Objective Lifecycle

`objectives_master.md` is static. Change it only when the user changes the long-term direction.

## Ledgers

1. `hypothesis_ledger.csv` is append-only and not default context.
2. `source_readiness_matrix.csv` stores source usability facts.
3. `context_research_ledger.md` stores compact durable context/orderbook/FreqAI handoff notes.
4. Large generated reports and old detailed histories stay referenced by path, not copied into active docs.

## Anti-Slop Rules

1. No new "current best" claims unless the current Sieve objective and Sieve result files support them.
2. No duplicate process docs.
3. No long run history in active guidance.
4. No generated reports in active context.
5. No deletion of old docs unless the user explicitly approves.
6. Park instead of delete when uncertain.
7. Add routing pointers instead of copying the same rules into many files.
8. Do not preserve obsolete objective-specific lane docs after the user retires that objective.
