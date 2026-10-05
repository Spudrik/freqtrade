---
doc_status: active
default_read: yes
owner: user
purpose: Start-here router and top-level repo contract for coding/research agents.
do_not_use_for: Detailed research history, generated reports, or evidence dumps.
last_rebuilt: 2026-09-09
---

# AGENTS.md - Start Here

## Core rule

Do not read every guidance file. Route directly from the user's request through this file to the minimum relevant subsystem `AGENTS.md` and rule files.

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

This file incorporates the original repo-level runtime notes. There is no ephemeral current-objective file. The user's latest explicit request defines the active task; use `objectives_master.md` only for programme-roadmap context or material stage changes.

Agents may suggest objective document creation, archival, deletion, movement, splitting, or replacement, but must not do those actions without explicit user approval.

## User interaction contract

1. When the user asks a question, answer first. Do not change code, move files, launch runs, or modify configuration until the user explicitly asks for action.
2. When the user asks for implementation, act directly inside the requested scope and routed rules.
3. Use concise, structured answers. Start with short query/answer lines using abbreviated versions of the user's questions, then expand only where useful.
4. Number or letter sections so the user can reference them quickly.
5. When sharing plots or charts, share one at a time with context before the link, and link to the local file so it opens in Codex side panel.
6. Code-edit permission is scoped to the files, subsystem, and behaviour the user explicitly approved. A broad objective, investigation request, or root-cause finding does not authorize edits outside that scope.
7. Before changing code outside the approved scope, state the exact files and intended behavioural change, explain why it is needed, and ask for explicit approval. No response is not approval: leave the change pending and raise the request again on a later relevant turn until the user explicitly approves or rejects it.
8. A one-time approval for named files does not widen standing permission. For current Objective 02b, the user has authorized research implementations and explicit indicator variants under `user_data/**` that are necessary to investigate reaction zones and the event-driven market hierarchy. Before editing `user_data/Indicators/**`, commit or record the clean fully tracked indicator baseline, create and switch to a dedicated experiment branch, and keep variants distinct from canonical indicators. Upstream core remains protected, and merging a variant into a canonical indicator still requires explicit user approval.
9. Do not spawn or use sub-agents unless the user explicitly authorizes sub-agents for the specific current task. Large edits, repetitive work, reviews, urgency, or available thread capacity do not imply permission.

## Fix discipline

1. Prefer root-cause fixes over defensive fallback code.
2. If a proposed fix looks like workaround bulk, stop and identify the real source of failure first.
3. Do not add interpreter guessing, broad fallback paths, duplicate parsers, or catch-all retry logic unless the user explicitly requests defensive behaviour.
4. Preserve existing Sieve result interpretation and strategy acceptance rules unless the user explicitly changes them.
5. If launcher, UI, preset, and runner settings drift apart, fix the drift at the shared source rather than patching only one caller.

## Required read order

1. Start from the user's latest explicit request and this router.
2. Read the nearest subsystem `AGENTS.md` when one exists, then only the minimum relevant rule files.
3. For coding tasks, read `ai_guidance_docs/02_rules/rules_codebase_workflow.md`.
4. For commands, backtests, dry-run startup, split-worker lanes, raw archive paths, or source data work, read `ai_guidance_docs/02_rules/rules_runtime_environment.md`.
5. For Hyperopt tasks, read `ai_guidance_docs/02_rules/rules_hyperopt_general.md`.
6. For exit/risk tasks, read `ai_guidance_docs/02_rules/rules_exit_and_risk_research.md` and the relevant Sieve-stage rules.
7. For the current bounded paper-trial stage, read
   `ai_guidance_docs/01_objectives/objective_03_comparative_paper_trial.md`.
   The old Objective 02b batch/branch queue is superseded; read it only when tracing
   specific research evidence or if the user explicitly reopens that research.
8. Read another file from `ai_guidance_docs/01_objectives/` only for programme architecture, a named parked objective, or a material stage change.
9. Read `ai_guidance_docs/00_project_control/objectives_master.md` only for programme-roadmap context or when the user asks to revisit programme priorities.
10. Do not read `ai_guidance_docs/99_archive/` unless the user asks for historical review or a routed rule requires specific historical evidence.
11. Sieve is the only approved Hyperopt system for trading entry/exit candidate discovery unless the user explicitly approves an exception. Objective 02b direct-test/FreqAI reaction-zone research was a separate approved historical lane, not a competing entry/exit Hyperopt system; it is not the current execution queue.
12. Read `ai_guidance_docs/05_program_traceability/AGENTS.md` only for a new programme-level chat, an explicit history/roadmap review, or a material stage transition. It is not routine task context.

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
6. The machine has `20` logical processors. Never copy or infer an `84`-thread setting from another environment.
7. Before starting a new multithreaded or multi-process task, inspect current processor use and the exact existing project jobs. Do not infer available capacity from logical-processor count alone.
8. Cap a new research workload at `4` worker threads by default, and reduce it when current load requires.
9. Preserve at least `4` logical processors for the user. A temporary reserve of only `2` is allowed during daytime in the `Europe/London` timezone for a bounded attended run; restore the four-processor reserve afterwards. Do not use the reduced reserve for unattended or overnight work.
10. Sub-agents and compute workers are separate limits. Do not spawn sub-agents without explicit task-specific authorization, regardless of available processor capacity.

## Long-run ownership and polling

1. The active coding/research agent owns code edits, strategy/config changes, batch design, launching Hyperopt/backtest/FreqAI/Freqtrade processes, checking completion, running summary scripts, and interpreting results.
2. Do not use heartbeat automations as the default way to continue an active goal. A heartbeat does not preserve or resume goal execution reliably; it is a detached reminder/monitor, not a substitute for the active agent.
3. For active goal-owned runs, prefer a bounded in-thread polling loop: launch the process, sleep for a sensible interval, check exact PIDs/logs/outputs, run summary scripts when complete, then continue the goal.
4. Use coarse polling intervals matched to expected runtime to reduce token/tool churn. Short runs can be checked more often; large multi-coin/high-epoch runs should sleep longer between checks.
5. If the user explicitly asks to keep chatting while a long run continues, then launch the run, record exact command/PID/log/output paths, and end the turn with a clear handoff. Do not claim the goal will automatically resume; a later user/heartbeat turn must pick it up from the run record.
6. Use heartbeat automations only when explicitly requested for detached monitoring, reminders, or periodic status checks. They must not edit strategy/config/code, launch unrelated experiments, broaden scope, or stop processes unless the run record explicitly identifies the target process and cleanup rule.
7. Per-run details should live in a machine-readable run record or ledger entry: exact command, PID/process tree where available, worker venv, log path, expected outputs, summary script, result doc/ledger, check interval, and stop/blocker conditions.

## Hyperopt search standard

1. Hyperopt work for trading entry/exit candidate discovery must route through Sieve unless the user explicitly approves a non-Sieve exception. Do not misapply this restriction to the direct-test and FreqAI reaction-zone or bounded event-scoped direction research explicitly governed by Objective 02b.
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

When the user starts a bounded goal-mode run, follow the iteration caps, stop rules, and output requirements in `rules_goal_mode_iteration_control.md`. Do not turn an overnight goal into an open-ended architecture rewrite. Produce useful ledgers and reports, not long chat summaries.

## If a needed rule is not routed

If a clearly relevant rule is not named by a subsystem router, read the minimum needed rule and flag the routing gap without creating an ephemeral task document.

Pause and ask the user only when the request materially broadens the approved scope, promotes parked work, conflicts with standing rules, risks live trading, or turns research evidence into strategy/promotion logic without explicit approval.

## Scope-change prompt rule

Ask the user for explicit scope approval if the requested work:

1. changes the programme stage or requested subsystem,
2. promotes a parked objective,
3. contradicts the master objectives,
4. starts using parked news/context data as if it is ready,
5. turns a research idea into strategy promotion,
6. starts a broad FreqAI queue without a named hypothesis,
7. requires new pass/fail rules not already defined for the current task,
8. would alter the verified dry-run strategy/config after it has been frozen,
9. would require live-trading mode or ambiguous order placement,
10. would require large new data downloads not authorized by the user.

Do not ask for guidance for ordinary implementation details inside the explicit request. Continue using the routed rules and codebase-aware workflow.

## Research runtime discipline

1. Before context/FreqAI research tests, state the objective and pass/fail condition first.
2. Prefer low-dimensional, hypothesis-led tests with controls before broad FreqAI runs.
3. Pause relevant collectors and export parquet snapshots before research runs that would otherwise read live SQLite databases.
4. If the user says to use FreqAI, use FreqAI.
5. Do not substitute custom scripts for FreqAI theory testing, candidate filtering, or ML inference when the user requested FreqAI.
6. If the existing FreqAI setup does not fit the requested data or theory, create or update a dedicated FreqAI environment/profile to run that request.
7. Define trader-readable behaviour first, then encode it numerically for the requested FreqAI run.
8. Treat overlapping signals as position-management evidence: same-direction may support add/hold/confidence; opposite-direction may support reduce/tighten/exit.
9. For Objective 02b event-scoped direction, test event, BTC/ETH/broad-market leader,
   coin-group transmission, coin-local modification, and post-event range information
   separately before pairwise and limited three-block combinations.
10. Select and hold out whole events rather than random candles from one continuing
    episode, and allow abstention when event confirmation, market leadership, or source
    coverage is unclear.
11. For event and interaction interpretation, use the whole-episode, decision-time
    role, expected-path, and conditional-modifier protocol in
    `ai_guidance_docs/02_rules/reference_freqai_event_reaction_research_method.md`.
    Never treat a downstream market response as a competing root cause merely because
    it predicts continuation.
12. Weak standalone evidence does not by itself reject a rational modifier, gate,
    suppressor, amplifier, accumulator, delay, or override. Development-discovered
    conditions must be frozen and confirmed on later whole episodes before acceptance.

## Global non-negotiables

1. Preserve behaviour unless the user explicitly requests a change.
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
