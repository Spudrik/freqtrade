---
doc_status: active
default_read: routed
owner: user
purpose: Runtime paths, Python environments, worker lanes, raw archive locations, and snapshot discipline.
do_not_use_for: Strategy acceptance, research interpretation, or source feature design.
last_rebuilt: 2026-06-10
---

# Rules - Runtime Environment

## Purpose

Use this file when a task runs commands, launches the launcher, starts/stops dry-run or backtest processes, uses split worker lanes, touches raw archives, or reads/writes context/orderbook research data.

This file incorporates the original top-level repo runtime notes. It replaces the old instruction that `current_objectives.md` is always canonical with the new routed guidance structure: start at repo-root `AGENTS.md`, then `ai_guidance_docs/00_project_control/objective_current.md`.

## Guidance document routing

1. Focused guidance lives under `C:\FreqTradeStuff\ai_guidance_docs`.
2. The active task contract is now `ai_guidance_docs\00_project_control\objective_current.md`.
3. Static long-term direction is `ai_guidance_docs\00_project_control\objectives_master.md`.
4. Do not append progress logs to objective files.
5. Use routed rule/status/result files for the current objective.
6. Use `ai_guidance_docs\99_archive` only for evidence verification, migration audit, or explicitly routed historical review.
7. If an objective appears complete, flag it to the user. Do not remove objectives without approval.

## Python environments

1. Controller environment:
   - `C:\FreqTradeStuff\.venv`
   - Use for PyCharm, launcher services, Entry Sieve orchestration, and Hyperopt commands.
2. First dedicated Freqtrade backtest worker:
   - `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest`
3. Additional dedicated Freqtrade worker lanes:
   - `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest-01` through the highest configured numbered lane present under `runtime\venvs`.
4. Worker venvs are ignored local installs only. They must not hold OHLCV data, configs, launcher state, reports, or backtest archives.
5. Each worker venv should have `freqtrade` installed editable from `C:\FreqTradeStuff` so it uses the main repo source.
6. Do not use `C:\Python3-11\python.exe` or any system Python for Entry Sieve orchestration, Hyperopt, or backtest lanes.
7. Job-level interpreter settings must override stale preset values. If a launcher preset contains an old `python_exe`, fix the merge order or preset source instead of relying on ad hoc command overrides.
8. On Windows, venv-launched Python processes may spawn child processes whose executable path displays as `C:\Python3-11\python.exe`. Judge ownership by the full command line and parent process tree, not the executable path alone.

## Parallel process safety

1. Multiple Freqtrade-capable venvs are intentional. They allow separate agents to run backtests, Hyperopts, dry-run checks, Sieve jobs, collectors, and research tools in parallel.
2. Before stopping or restarting anything, assume other Python/Freqtrade processes may belong to another active user task or agent.
3. Stop only processes this agent started, or processes explicitly matched by exact command line, PID, log path, queue/job file, and user-approved scope.
4. Do not kill broad `python.exe`, `freqtrade`, Hyperopt, backtest, collector, launcher, or Sieve processes just because there appear to be duplicates.
5. On Windows, one logical venv-launched job can show both the venv Python executable and the base interpreter. Treat that as normal unless command lines prove a real duplicate.
6. Prefer targeted worker-env commands, explicit `--logfile`/result paths, and isolated report folders so ownership and cleanup are auditable.

## Long-running run ownership and polling

1. The active coding/research agent owns code edits, strategy/config changes, batch design, launching Hyperopt/backtest/FreqAI/Freqtrade processes, checking completion, running summary scripts, and interpreting results.
2. Do not use heartbeat automations as the default way to continue active goals. Heartbeats are detached reminders/monitors and do not reliably preserve or resume the active goal execution path.
3. For active goal-owned runs, prefer bounded in-thread polling: launch the process, sleep for a sensible interval, check exact PIDs/logs/outputs, run summary scripts when complete, and continue from those results.
4. Tune polling intervals to runtime. A small BTC-only 100-epoch batch may justify frequent checks; a large multi-coin 400-epoch batch should sleep longer between checks.
5. If the user explicitly asks to keep chatting while a long run continues, launch the run, record exact command/PID/log/output paths, and end the turn with a clear handoff. Do not claim the goal will automatically resume.
6. Use heartbeat automations only when explicitly requested for detached monitoring, reminders, or periodic status checks.
7. Detached automations must not edit strategy/config/code, launch unrelated experiments, broaden scope, or stop processes unless the run record explicitly identifies the target process and cleanup rule.
8. Every long run should have a machine-readable run record or ledger entry:
   - exact command,
   - PID/process tree where available,
   - worker venv,
   - log path,
   - expected output files,
   - summary script to run,
   - result doc/ledger to update,
   - suggested check interval,
   - stop/blocker conditions.
9. If a run was not launched by the current agent/thread, treat it as external unless its PID/command/log/output path exactly matches the run record.

## Backtest lane rules

1. Split-venv worker count means dedicated worker venvs only: from `1` to `9`.
2. Do not count the controller `.venv` or system Python as a backtest lane.
3. Explorer and Entry Sieve split-venv runs must treat worker count as a concurrency limit, not permission for unbounded subprocesses.
4. Backtest tasks must be queued across configured lanes and keep result directories isolated per task/window.
5. Do not add interpreter guessing or fallback behaviour. If a configured Python executable is missing, fail early with the missing path.
6. Keep backtest worker venvs under `runtime\venvs`; do not move them into `user_data` or launcher packages.
7. Do not run broad overnight tests on live trading mode. Dry-run startup verification must remain dry-run only.

## Raw archive locations - Bybit orderbook

1. Historical Bybit raw ZIP archives are stored on `D:`, not under `C:\FreqTradeStuff\user_data`:
   - root: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`
   - spot: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\spot_raw`
   - linear: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`
   - inverse: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`
2. Bybit features, logs, and manifests remain under:
   - `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit`
3. Do not infer incomplete Bybit raw coverage from this folder being absent or empty:
   - `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw`

## Raw archive locations - GDELT/GKG

1. Historical GDELT/GKG raw ZIP archives are stored on `D:`, not under `C:\FreqTradeStuff\user_data`:
   - root: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`
   - GKG: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`
   - event export: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`
2. SQLite database:
   - `C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite`
3. Do not infer incomplete GDELT/GKG raw coverage from this folder being absent or empty:
   - `C:\FreqTradeStuff\user_data\research_news_data\gdelt\raw`

## Research runtime and snapshot discipline

1. Before context/FreqAI research tests, state the objective and pass/fail rule.
2. Prefer low-dimensional, hypothesis-led tests with controls before broad FreqAI runs.
3. Pause relevant collectors and export parquet snapshots before research runs that would otherwise read live SQLite databases.
4. Do not move scraping/API fetching into Freqtrade strategy code.
5. Treat FreqAI as validation/ranking after direct evidence, not the first fishing tool.
6. Use frozen parquet snapshots where possible for repeatability.
7. Record data coverage and timestamp flaws separately from model metrics.
8. Do not remove open issues without a resolved entry.

## Context/orderbook research ledger updates

When adding context/orderbook sources, changing feature conditioning, running FreqAI research, validating timestamp alignment, or resolving data gaps, update the active compact ledgers/results routed by the current objective. Current replacements for the old monolithic ledger pattern are:

1. `ai_guidance_docs\04_results\context_research_ledger.md` for durable context/orderbook/FreqAI handoff notes.
2. `ai_guidance_docs\04_results\results_recent_summary.md` only when broad findings change.
3. `ai_guidance_docs\04_results\source_readiness_matrix.csv` when source usability changes.
4. Detailed generated reports stay in their generated report folders and are referenced by path, not copied into active guidance.

## External context feature design anchors

1. Build context/orderbook datasets around trader-readable market behaviour, not raw counts or isolated values.
2. Orderbook examples: persistent resistance/support zones, wall evaporation, pressure shock, liquidity vacuum, spread shock, post-breakout support rebuild.
3. News/GDELT examples: event severity, source confluence, topic persistence, first mention versus follow-through, plausible impact channel.
4. Global/macro examples: surprise versus expectation, risk-on/risk-off pressure, rate/liquidity shock, persistence, cross-market confirmation.
5. Prefer compact descriptive features such as `resistance_removed_score`, `topic_confluence_24h`, or `macro_liquidity_shock` over hundreds of weak raw columns.
6. Do not hardcode bullish/bearish article opinions as truth. Encode observable facts and let tests decide.
7. Do not reduce complex confluence examples to one source, indicator, or timeframe.
