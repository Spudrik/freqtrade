---
doc_status: active
default_read: yes
owner: user
purpose: Start-here router and top-level repo contract for coding/research agents.
do_not_use_for: Detailed research history, generated reports, or evidence dumps.
last_rebuilt: 2026-07-13
---

# AGENTS.md - Start Here

## Core rule

Do not read every guidance file. Read only the current objective and the files routed by that objective.

## Upstream Freqtrade core protection

1. Treat the upstream Freqtrade codebase as protected and read-only. Agents must not edit `freqtrade/**`, upstream-owned `tests/**`, root dependency/build files, or otherwise change Freqtrade/FreqAI runtime behaviour without explicit user approval naming the affected files and intended behaviour.
2. A general request to implement, fix, investigate, improve, or run project work is not permission to patch Freqtrade or FreqAI internals. Formatting-only core edits also require approval.
3. If a task appears to require a core edit, stop before editing and present:
   - the root cause,
   - the preferred solution under `user_data/**`,
   - the exact core files and behavioural change that would otherwise be required.
4. Project customizations should live under `user_data/**`, including strategies, indicators, FreqAI models, launchers, collectors, research tools, custom tests, and research-only dependency manifests. Approved project guidance remains under `ai_guidance_docs/**`.
5. Agents may read and inspect protected core files and may propose an exact patch, but they must wait for explicit user approval before applying it.
6. Existing approved maintained-fork exceptions are limited to:
   - the historical Bybit orderbook integration while it remains compatible with upstream,
   - the Pandas dtype compatibility adjustment in `strategy_helper.py`,
   - the user-approved Hyperopt initial sampling value `INITIAL_POINTS = 60`.
7. Do not expand an existing exception or create another one without explicit user approval. If an upstream update conflicts with the historical Bybit integration, propose extracting it into a standalone `user_data/**` tool before modifying upstream code further.
8. Do not add project/research packages to Freqtrade's root requirements. Put them in an approved dependency manifest under `user_data/**`.

This file incorporates the original repo-level runtime notes. The old top-level instruction that `current_objectives.md` was always canonical is superseded by the new routed structure: start at `objective_current.md`; use `objectives_master.md` only when the active stage is unclear or changing. `objective_current.md` is the active current objective, not just a pointer to another objective file.

Agents may suggest objective document creation, archival, deletion, movement, splitting, or replacement, but must not do those actions without explicit user approval.

## User interaction contract

1. When the user asks a question, answer first. Do not change code, move files, launch runs, or modify configuration until the user explicitly asks for action.
2. When the user asks for implementation, act directly inside the current objective and routed rules.
3. Use concise, structured answers. Start with short query/answer lines using abbreviated versions of the user's questions, then expand only where useful.
4. Number or letter sections so the user can reference them quickly.
5. When sharing plots or charts, share one at a time with context before the link, and link to the local file so it opens in Codex side panel.

## Fix discipline

1. Prefer root-cause fixes over defensive fallback code.
2. If a proposed fix looks like workaround bulk, stop and identify the real source of failure first.
3. Do not add interpreter guessing, broad fallback paths, duplicate parsers, or catch-all retry logic unless the user explicitly requests defensive behaviour.
4. Preserve existing Sieve result interpretation and strategy acceptance rules unless the user explicitly changes them.
5. If launcher, UI, preset, and runner settings drift apart, fix the drift at the shared source rather than patching only one caller.

## Required read order

1. Read `ai_guidance_docs/00_project_control/objective_current.md`.
2. Read only the rule/status/result files listed by that objective.
3. Read a file from `ai_guidance_docs/01_objectives/` only when `objective_current.md` explicitly routes to it.
4. For coding tasks, also read `ai_guidance_docs/02_rules/rules_codebase_workflow.md`.
5. For commands, backtests, dry-run startup, split-worker lanes, raw archive paths, or source data work, also read `ai_guidance_docs/02_rules/rules_runtime_environment.md`.
6. Read `ai_guidance_docs/00_project_control/objectives_master.md` only if the current objective is missing, unclear, contradicted by the user request, or the task appears to move to a new stage.
7. Do not read `ai_guidance_docs/99_archive/` unless the current objective explicitly says to verify historical evidence or the user asks for historical review.
8. For Hyperopt tasks, read `ai_guidance_docs/02_rules/rules_hyperopt_general.md`.
9. For exit/risk tasks, read `ai_guidance_docs/02_rules/rules_exit_and_risk_research.md`.
10. Sieve is the only approved Hyperopt/discovery system unless the user explicitly approves a non-Sieve exception.
11. Read `ai_guidance_docs/05_program_traceability/AGENTS.md` only for a new programme-level chat, an explicit history/roadmap review, a material stage transition, or when the current objective/coordinating agent routes there. It is not routine task context.

## Runtime anchors

1. Repo root: `C:\FreqTradeStuff`.
2. Guidance docs: `C:\FreqTradeStuff\ai_guidance_docs`.
3. Controller Python: `C:\FreqTradeStuff\.venv`.
4. Dedicated backtest worker venvs: `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest` and numbered `freqtrade-backtest-*` lanes under `runtime\venvs`.
5. Do not use system Python or the controller `.venv` as a backtest worker lane.
6. Use the controller `.venv` for launcher services, Entry Sieve orchestration, and Hyperopt commands. Job-level interpreter settings must override stale launcher preset values.
7. Do not guess interpreters or add fallback behaviour. Missing configured executables should fail early with the missing path.
8. Bybit raw ZIPs live on `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`, not under `user_data`.
9. GDELT/GKG raw ZIPs live on `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`, not under `user_data`.
10. Never infer raw coverage is missing just because the mirrored `C:\FreqTradeStuff\user_data\...\raw` path is absent or empty.

## Parallel process safety

1. This repo intentionally has several Freqtrade-capable venv installs so different agents can run backtests, Hyperopts, dry-run checks, Sieve jobs, collectors, and research tools in parallel.
2. Assume there may already be multiple Python, Freqtrade, Hyperopt, backtest, collector, or Sieve processes running for other agents or user tasks.
3. Stop only processes that this agent started, or processes that are explicitly identified by exact command line, PID, log path, queue/job file, and user-approved scope.
4. Do not kill broad `python.exe`, `freqtrade`, Hyperopt, backtest, collector, launcher, or Sieve processes just because they look duplicated. Windows venv launches can show both the venv Python and the base interpreter for one logical job.
5. Prefer targeted worker-env commands, explicit `--logfile`/result paths, and isolated report folders so later agents can tell which process owns which task.

## Long-run ownership and polling

1. The active coding/research agent owns code edits, strategy/config changes, batch design, launching Hyperopt/backtest/FreqAI/Freqtrade processes, checking completion, running summary scripts, and interpreting results.
2. Do not use heartbeat automations as the default way to continue an active goal. A heartbeat does not preserve or resume goal execution reliably; it is a detached reminder/monitor, not a substitute for the active agent.
3. For active goal-owned runs, prefer a bounded in-thread polling loop: launch the process, sleep for a sensible interval, check exact PIDs/logs/outputs, run summary scripts when complete, then continue the goal.
4. Use coarse polling intervals matched to expected runtime to reduce token/tool churn. Short runs can be checked more often; large multi-coin/high-epoch runs should sleep longer between checks.
5. If the user explicitly asks to keep chatting while a long run continues, then launch the run, record exact command/PID/log/output paths, and end the turn with a clear handoff. Do not claim the goal will automatically resume; a later user/heartbeat turn must pick it up from the run record.
6. Use heartbeat automations only when explicitly requested for detached monitoring, reminders, or periodic status checks. They must not edit strategy/config/code, launch unrelated experiments, broaden scope, or stop processes unless the run record explicitly identifies the target process and cleanup rule.
7. Per-run details should live in a machine-readable run record or ledger entry: exact command, PID/process tree where available, worker venv, log path, expected outputs, summary script, result doc/ledger, check interval, and stop/blocker conditions.

## Hyperopt search standard

1. Hyperopt/discovery work must route through Sieve unless the user explicitly approves a non-Sieve exception.
2. Hyperopt should search broad, explicit theory spaces that would be inefficient to check manually.
3. If a proposed Hyperopt run could be replaced by roughly `10-15` ordinary backtests, the search design is probably too small and should be expanded or downgraded to a simple comparison.
4. For Hyperopt tasks, read and follow `ai_guidance_docs/02_rules/rules_hyperopt_general.md` before creating or launching a search.

## Sieve promotion integrity

1. A promoted Sieve strategy must carry forward the exact selected Hyperopt parameters that justified promotion.
2. When promoting or branching from Sieve2 into Sieve3, lock the entry surface first: either embed the promoted Sieve2 `buy` parameter values as strategy defaults or provide an explicit params overlay that Freqtrade will load for every Sieve3 run.
3. Do not assume a promoted strategy will keep its winning values just because the Python file was copied. If no matching params JSON is loaded, Freqtrade uses the parameter defaults in the strategy file.
4. Sieve3 exit research must keep the promoted entry parameters fixed while optimizing only exit/risk parameters. Before launching exit-stage batches, audit at least one branch per source entry stem and confirm its `buy` defaults or loaded params match the promoted Sieve2 result.
5. If traceability from a Sieve3 branch back to its promoted Sieve2 params is missing or ambiguous, stop that branch group and resolve the source mapping before running exit tests.
6. Do not shrink Sieve3 exit research to BTC-only unless the source entry is known to trade frequently on BTC alone. Sparse, MTF, structure, and pattern entries must use enough pairs and windows to give Hyperopt real training trades.
7. For rare pattern entries, validate pair/window coverage before launching exit Hyperopt. If the training surface has zero or only a few trades, broaden pairs or use larger routed windows rather than judging the exit logic from a starved run.
8. Treat tiny Sieve3 exit training samples as a blocking setup error, not a weak result. If the first live Hyperopt window for a sparse/pattern branch shows fewer than roughly `10` training trades, stop that batch and restart with broader pair/window coverage before accepting any result.

## Goal-mode operating rule

When the active objective is a bounded goal-mode run, follow the iteration caps, stop rules, and output requirements in `rules_goal_mode_iteration_control.md`. Do not turn an overnight goal into an open-ended architecture rewrite. Produce useful ledgers and reports, not long chat summaries.

## If a needed rule is not routed

If a clearly relevant rule file is not listed by the current objective but the work still fits the active objective, read the needed rule and note that the objective routing may need a small update.

Pause and ask the user only when the request changes scope, promotes parked work, contradicts the current objective, risks live trading, or turns research evidence into strategy/promotion logic without an updated objective.

## Deviation prompt rule

Ask the user whether the current objective should be updated if the requested work:

1. changes the active objective,
2. promotes a parked objective,
3. contradicts the master objectives,
4. starts using parked news/context data as if it is ready,
5. turns a research idea into strategy promotion,
6. starts a broad FreqAI queue without a named hypothesis,
7. requires new pass/fail rules not already defined for the current task,
8. would alter the verified dry-run strategy/config after it has been frozen,
9. would require live-trading mode or ambiguous order placement,
10. would require large new data downloads not authorized by the current objective.

Do not ask for guidance for ordinary implementation details inside the active objective. Continue using the current objective, routed rules, and the codebase-aware workflow.

## Research runtime discipline

1. Before context/FreqAI research tests, state the objective and pass/fail condition first.
2. Prefer low-dimensional, hypothesis-led tests with controls before broad FreqAI runs.
3. Pause relevant collectors and export parquet snapshots before research runs that would otherwise read live SQLite databases.
4. If the user says to use FreqAI, use FreqAI.
5. Do not substitute custom scripts for FreqAI theory testing, candidate filtering, or ML inference when the user requested FreqAI.
6. If the existing FreqAI setup does not fit the requested data or theory, create or update a dedicated FreqAI environment/profile to run that request.
7. Define trader-readable behaviour first, then encode it numerically for the requested FreqAI run.
8. Treat overlapping signals as position-management evidence: same-direction may support add/hold/confidence; opposite-direction may support reduce/tighten/exit.

## Global non-negotiables

1. Preserve behaviour unless the current objective explicitly changes it.
2. Investigate existing code before adding new systems.
3. Reuse existing helpers and ledgers before creating new ones.
4. Do not invent dry-run safety settings, leverage caps, stake sizes, pair limits, or live-trading rules.
5. Dry-run launch/monitoring details are user-controlled unless explicitly provided in the current task.
6. Do not claim one permanent current best unless the current Sieve objective and result files support it.
7. Do not delete archives or generated reports without explicit user approval.
8. Indicators and heavy dataframe calculations should stay vectorized.
9. No news/context source may be treated as ready unless source-specific readiness proves it.
10. No FreqAI promotion may happen without a named trader-readable hypothesis, controls, baselines, and clean-window/source checks.
11. Do not edit the verified dry-run class/config during later research; create separate variant classes/configs for research.
12. Avoid token waste: compact investigation first, bounded tests second, ledgers/reports third.
