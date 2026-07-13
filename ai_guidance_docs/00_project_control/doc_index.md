---
doc_status: active
default_read: no
owner: agent
purpose: Index of active guidance docs and when to read them.
do_not_use_for: Research evidence details or broad context loading.
last_rebuilt: 2026-07-13
---

# Document Index

## Read Policy Values

- `yes`: read by default.
- `routed`: read only when named by the current objective/objective file.
- `no`: do not read unless explicitly needed.

## 00_project_control

| File | Default read | Purpose |
|---|---:|---|
| `objective_current.md` | yes | Current active task and required docs. This is the active objective, not only a router. |
| `objectives_master.md` | routed | Static objective map; read when stage/routing is unclear or changing. |
| `doc_index.md` | no | Use when unsure where guidance lives. |
| `archive_index.md` | no | Use only when checking old documents. |

## 01_objectives

| File | Default read | Read when |
|---|---:|---|
| `objective_02_master_strategy_architecture.md` | routed | Returning to wider Sieve-first confluence architecture. |
| `objective_02a_strategy_refinement.md` | routed | Refining Sieve candidates. |
| `objective_02c_orderbook_confluence.md` | routed | Using orderbook for confirmation/risk/targets. |
| `objective_02d_news_context_integration.md` | routed | News/context work; parked unless user says ready or readiness proves a block/window. |
| `objective_02e_generic_ta_integration.md` | routed | Generic TA side-lane integration. |

## 02_rules

| File | Default read | Purpose |
|---|---:|---|
| `rules_codebase_workflow.md` | routed | Investigate/reuse/refactor-breakpoint rules. |
| `rules_runtime_environment.md` | routed | Python envs, backtest lanes, raw archive paths, snapshot discipline. |
| `rules_hyperopt_general.md` | routed | Sieve-first Hyperopt design, candidate extraction, and reporting rules. |
| `rules_exit_and_risk_research.md` | routed | Exit logic and risk research rules. |
| `rules_goal_mode_iteration_control.md` | routed | Bounded long-run iteration, stop, and park rules. |
| `rules_source_detail_taxonomy.md` | routed | Standard source-detail group names. |
| `rules_freqai_feature_discovery.md` | routed | Parked feature discovery standards. |
| `rules_freqai_promotion.md` | routed | Strict FreqAI promotion thresholds and preflight. |
| `rules_direct_tests.md` | routed | Hypothesis tests, controls, ablations, metrics. |
| `rules_strategy_success.md` | routed | Strategy/candidate acceptance and promotion rules. |
| `rules_entry_exit_position.md` | routed | Entry-specific exit/add/reduce/risk architecture. |
| `rules_sieve3_exit_hyperopt.md` | routed | Sieve3 exit-stage branching, Hyperopt test families, and result interpretation. |
| `rules_sieve3_exit_regeneration_agent.md` | routed | Controlling implementation spec for regenerating Sieve3 exit files with nuanced categorical branch logic. |
| `rules_orderbook_sources.md` | routed | Orderbook data readiness and usage. |
| `rules_news_context_sources.md` | routed | News/GDELT/GKG/web/global readiness and future formatting. |
| `rules_result_reporting.md` | routed | Plain-English result reporting. |
| `rules_document_maintenance.md` | routed | How to keep docs clean. |

## 03_status

| File | Default read | Purpose |
|---|---:|---|
| `status_source_readiness.md` | routed | Current source readiness and parked data. |
| `status_open_questions.md` | routed | Known questions for the user. |

## 04_results

| File | Default read | Purpose |
|---|---:|---|
| `results_recent_summary.md` | routed | Concise recent evidence summary. |
| `context_research_ledger.md` | routed | Compact context/orderbook/FreqAI handoff ledger. |
| `hypothesis_ledger.csv` | no | Append-only idea/test ledger. Do not read by default. |
| `source_readiness_matrix.csv` | routed | Source usability matrix. |
| `results_archive_map.md` | no | Where old reports live and when to open them. |

## 05_program_traceability

Non-governing programme history and handoff context. Read only for a new programme-level chat, explicit review, material stage transition, or when routed. Start with the local router, then read the overview and only one relevant track.

| File | Default read | Purpose |
|---|---:|---|
| `AGENTS.md` | no | Conditional read/edit router and milestone gate for this folder. |
| `program_overview.md` | no | Compact end goal, current programme position, objective map, chronology, and roadmap. |
| `tracks/sieve_entry_exit.md` | no | Sieve generations, entry/exit research, position adjustment, and refined exit decisions. |
| `tracks/indicators_core.md` | no | Core custom-indicator development and material redesigns. |
| `tracks/indicators_patterns.md` | no | Separate rare-pattern indicator history and research requirements. |
| `tracks/freqai_risk_weighting.md` | no | FreqAI research and future dynamic risk/weighting role. |
| `tracks/data_historical.md` | no | Historical source acquisition, extraction, coverage, and gaps. |
| `tracks/data_live.md` | no | Live collection, freshness, snapshots, and accumulated coverage. |
| `tracks/master_integration.md` | no | Entry/exit weighting, overlap control, trade state, and final integration. |

## 99_archive

Original and superseded docs. Do not read by default.
