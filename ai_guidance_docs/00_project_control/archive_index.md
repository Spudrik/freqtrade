---
doc_status: active
default_read: no
owner: agent
purpose: Archive map and when to open old files.
do_not_use_for: Default context loading.
last_rebuilt: 2026-06-11
---

# Archive Index

Archive rule: do not read archive files unless the user asks for historical review, evidence verification, migration audit, or a current objective explicitly lists an archive file.

## Archive Folders

| Folder | Contents | Open only when |
|---|---|---|
| `99_archive/original_uploaded_docs/original_top_level/` | Original repo-level runtime/environment notes supplied after v6. | Auditing top-level routing/runtime migration. |
| `99_archive/original_uploaded_docs/history_logs/` | Large historical ledgers and old progress logs. | Reconstructing what happened or verifying old evidence. |
| `99_archive/original_uploaded_docs/parked_news_context/` | GDELT/GKG/news formatting plans. | Resuming news/context integration. |
| `99_archive/original_uploaded_docs/parked_freqai_confluence_plans/` | Old FreqAI/confluence plans. | Designing a new confluence/FreqAI work package. |
| `99_archive/original_uploaded_docs/reviews/` | 2026-05-29 review findings. | Checking past data/correctness issues. |
| `99_archive/original_uploaded_docs/parked_process_docs/` | Process scaffolding and examples. | Rebuilding process docs or examples. |
| `../01_objectives/archive/` | Completed or superseded routed objective files, when present. | Auditing prior objective scope or verifying why it was archived. |

## Important Archived Docs To Know About

- `repo_runtime_environment_notes_original.md`: original top-level runtime/environment contract; now factored into `AGENTS.md` and `rules_runtime_environment.md`.
- `freqai_promotion_readiness_plan.md`: strict historical FreqAI promotion gates.
- `news_formatting_objectives.md`: detailed future news/context normalized-layer design.
- `gdelt_gkg_historical_rework_plan.md`: historical GDELT/GKG rework plan.
- `objectives_progress.md`: large run-history ledger; not canonical.
- `context_research_detailed_findings.md`: detailed historical research findings; not default context.

## Related Active Map

- `../04_results/results_archive_map.md`: compact map for old result/history locations.
