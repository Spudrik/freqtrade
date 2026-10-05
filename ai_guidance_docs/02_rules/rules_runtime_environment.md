---
doc_status: active
default_read: routed
owner: user
purpose: Runtime paths, Python environments, worker lanes, raw archive locations, and snapshot discipline.
do_not_use_for: Strategy acceptance, research interpretation, or source feature design.
last_rebuilt: 2026-10-05
---

# Rules - Runtime Environment

## Purpose

Use this file when a task runs commands, launches the launcher, starts/stops dry-run or backtest processes, uses split worker lanes, touches raw archives, or reads/writes context/orderbook research data.

This file incorporates the original top-level repo runtime notes. Start at repo-root `AGENTS.md`, then route directly from the user's request to the relevant subsystem and runtime rules.

## Guidance document routing

Follow repo-root `AGENTS.md` §§2, 4, and 5 for request-first routing, minimum relevant guidance, and archive access. Long-term direction is recorded in `ai_guidance_docs\00_project_control\objectives_master.md`; do not append progress logs to objective files. If an objective appears complete, flag it to the user; do not remove objectives without approval.

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
7. This host has `20` logical processors. Never reuse or infer an `84`-thread configuration from another system.
8. Before launching a multithreaded or multi-process task, inspect current processor use and exact existing jobs. Logical-processor count is not the same as available capacity.
9. Limit each new research workload to `4` worker threads by default, and lower it when current load requires.
10. Preserve at least `4` logical processors for the user. The reserve may temporarily fall to `2` only during daytime in the `Europe/London` timezone for a bounded attended run, after which the four-processor reserve must be restored. Never use the reduced reserve for unattended or overnight work.
11. Agent delegation and local compute-worker limits are separate. Delegation follows the repo-root orchestration contract; sub-agent count does not increase the processor budget.

## Long-running run ownership and polling

1. Follow the orchestration contract in repo-root `AGENTS.md` §1. For each run, record the exact assigned owner and authorized scope; assignment clarifies responsibility and does not add an approval gate.
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
3. For research that would otherwise read collector-fed live SQLite databases, pause only the relevant collectors and export immutable parquet snapshots before testing. This research snapshot rule does not apply to routine paper-trial health checks or monitoring; do not pause collectors for those checks.
4. Do not move scraping/API fetching into Freqtrade strategy code.
5. Treat FreqAI as validation/ranking after direct evidence, not the first fishing tool.
6. Use frozen parquet snapshots where possible for repeatability.
7. Record data coverage and timestamp flaws separately from model metrics.
8. Do not remove open issues without a resolved entry.

## Targeted `1m` data acquisition for Objective 02b

The user has authorized downloading missing `1m` OHLCV needed by the bounded
evidence-triggered replay lane. This authorization covers selected episode windows and
their causal warm-up/context only; it is not permission for an automatic all-pair,
all-history minute-data mirror.

Before each acquisition:

1. freeze the queued pairs, exact exchange/market type, UTC contact timestamps, causal
   anchor timeframes, default or approved expanded replay envelopes, and feature
   warm-up;
2. inventory existing local `1m` files and continuity before assuming anything is
   missing;
3. construct per-pair intervals, merge overlaps and adjacent padded windows, and
   calculate expected row counts;
4. use existing Freqtrade/launcher download helpers from the controller `.venv` and
   the explicitly selected data directory; do not use system Python or add downloader
   logic to a strategy;
5. do not download unselected pairs or silently fetch years between distant episodes
   when targeted staged ranges and extracts are materially smaller;
6. record the exact command, pair, market type, timeframe, timerange, data directory,
   format, source, and completion status;
7. verify timestamps, minute continuity, duplicates, gaps, expected/actual rows, and
   readable output before accepting coverage; and
8. align the replay to the first `1m` entry into the already-known frozen zone, not
   merely to the opening timestamp of the parent `1h`/`4h`/`8h`/`1d` candle.

Long level-construction history should normally reuse existing source-timeframe
candles. Dense `1m` data belongs only in the source-scaled replay envelope unless a
recorded boundary audit justifies one whole-pattern expansion.

Retain existing reusable Freqtrade-formatted OHLCV in its approved data location. Put
bulky deduplicated replay slices and staged extracts under
`D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones`, with compact
coverage manifests and analysis records under the routed `C:` research path. Remove
failed, duplicate, or superseded extracts only after verifying the retained
replacement; never delete shared source data merely because one replay is complete.

## Context/orderbook research ledger updates

When adding context/orderbook sources, changing feature conditioning, running FreqAI research, validating timestamp alignment, or resolving data gaps, update the active compact ledgers/results routed by the relevant subsystem rules. Current replacements for the old monolithic ledger pattern are:

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
