---
doc_status: active
default_read: no
owner: agent
purpose: Map of old result/history locations and when to open them.
do_not_use_for: Default context loading.
last_rebuilt: 2026-06-10
---


# Results Archive Map

## Rule

Do not read old result/history ledgers by default. Open them only when verifying a claim, recovering an evidence path, rebuilding a ledger, or reviewing historical work.

## Key archive locations

| Need | Open |
|---|---|
| Large run/progress history | `../99_archive/original_uploaded_docs/history_logs/objectives_progress.md` |
| Detailed context/orderbook/FreqAI research findings | `../99_archive/original_uploaded_docs/history_logs/context_research_detailed_findings.md` |
| Old queue/process baseline | `../99_archive/original_uploaded_docs/history_logs/goal_status_tracker.md` |
| News/context rebuild details | `../99_archive/original_uploaded_docs/parked_news_context/` |
| Old FreqAI/confluence plans | `../99_archive/original_uploaded_docs/parked_freqai_confluence_plans/` |
| 2026-05-29 correctness reviews | `../99_archive/original_uploaded_docs/reviews/` |

## Evidence recovery rule

If a lane or hypothesis has `missing_evidence_path`, search archived result ledgers/reports and update the ledger with the exact path. Do not invent evidence paths.
