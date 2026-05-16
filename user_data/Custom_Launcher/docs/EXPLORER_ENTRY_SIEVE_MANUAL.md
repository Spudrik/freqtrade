# Explorer And Entry Sieve Manual

Status: review draft.

Audience:

- User reviewing/running Explorer and Entry Sieve.
- Agents making strategy, launcher, or Entry Sieve changes.

Scope:

- Explorer tab modes.
- Entry Sieve runs and batch queues.
- Result review.
- Strategy-file rules.
- Architecture and runtime state.

## 1. Core Intent

- This project is a research machine for Freqtrade strategy development.
- Explorer is the controlled optimizer for an existing strategy parameter surface.
- Entry Sieve is the broad entry-discovery pass over many small strategy files.
- Sieve output is diagnostic, not final proof.
- Final judgement comes from broader backtests, normally `full_cycle_2020_2026`.
- Hyperopt windows are seed windows: good enough data to get candidate params.
- Do not over-tune hyperopt windows; use them to generate candidates for later review.
- Avoid adding defensive fallback code unless explicitly requested.
- Prefer root-cause fixes for config, path, state, or strategy-mapping issues.

## 2. Launcher Map

- Active launcher: `user_data/Custom_Launcher/launcher_v2/app.py`.
- Main UI tab for this work: `launcher_v2/tabs/explorer_tab.py`.
- Explorer service: `launcher_v2/services/explorer_service.py`.
- Entry Sieve service: `launcher_v2/services/entry_sieve_service.py`.
- Entry Sieve runner: `launcher_v2/services/entry_sieve_runner.py`.
- Entry Sieve batch queue runner: `launcher_v2/services/entry_sieve_batch_queue_runner.py`.
- Normal Explorer runner: `explorer/explorer_runner.py`.
- Split-venv Explorer runner: `explorer/explorer_pipeline_runner.py`.
- Shared command helpers: `explorer/explorer_commands.py`.
- Shared windows helpers: `explorer/explorer_windows.py`.
- Target/catalog helpers: `explorer/explorer_catalog.py`, `explorer/explorer_targets.py`.

## 3. Important Config Files

- Launcher presets: `launcher_v2/config/presets.json`.
- Explorer market windows: `explorer/config/market_windows.json`.
- Entry Sieve auto windows: `explorer/config/sieve_auto_windows.json`.
- Entry Sieve strategy batches: `explorer/config/sieve_strategy_batches.json`.
- Strategy-agent rules: `user_data/strategies/AGENTS.md`.
- Launcher source-of-truth note: `user_data/Custom_Launcher/README_SOURCE_OF_TRUTH.md`.

## 4. Runtime State

- Entry Sieve runtime root: `launcher_v2/runtime/entry_sieve`.
- Jobs: `runtime/entry_sieve/jobs/*.json`.
- Queue definitions: `runtime/entry_sieve/queues/*.json`.
- Active pointer: `runtime/entry_sieve/active.json`.
- Status files: `runtime/entry_sieve/status/*.json`.
- Result files: `runtime/entry_sieve/results/*.jsonl`.
- Result summaries: `runtime/entry_sieve/results/*.summary.json`.
- Runtime strategy copies: `runtime/entry_sieve/runs/<strategy>__<window>/strategy`.
- Hyperopted params archive: `runtime/entry_sieve/params/*.json`.
- Backtest artifacts: `runtime/entry_sieve/backtests/...`.
- `active.json` means a run is active or recently active; confirm the PID/status before assuming.
- Empty result files should not be kept; zero-row/error-only debug outputs are disposable.

## 5. Explorer Modes

- Target type:
  - `family`: choose a parameter family, such as `entries`.
  - `mode`: choose a mode/tag within a family.
- Target selection:
  - `random`: chooses from available targets using usage state.
  - `specific`: requires a named target.
- Search breadth:
  - `targeted`: tunes the selected target's direct params.
  - `open`: broader surface around that target.
- Loop count:
  - `0`: continuous/infinite.
  - `1`: single loop.
- Epoch modes:
  - Manual epochs: fixed count.
  - Auto epochs: `20 * resolved_param_count`, optionally capped.
- Split-venv pipeline:
  - Uses dedicated backtest worker venvs under `runtime/venvs`.
  - Worker count is a concurrency limit, not permission for unbounded subprocesses.
- Normal Explorer flow:
  - Select target.
  - Randomly pick one selected training window per loop.
  - Hyperopt selected surface.
  - Backtest champion and challenger on validation windows.
  - Accept only if scoring/guards improve.
  - Persist accepted params to the strategy param file.

## 6. Explorer Acceptance

- Normal Explorer has champion/challenger acceptance logic.
- Explorer can accept or reject parameter changes.
- Acceptance compares validation-window score, drawdown, profit, trades, and guard state.
- Normal Explorer is for improving a known strategy, not mass-testing hundreds of seed entry files.
- Do not confuse Explorer acceptance with Entry Sieve ranking.

## 7. Entry Sieve Purpose

- Entry Sieve is Sieve1: entry-quality diagnostics.
- It is designed to answer: "Does this entry idea show an obvious edge under coarse tuning?"
- It is not designed to tune exits, leverage, adds, stake, global risk, or position management.
- Future sieves may test exits/risk/etc, but only after explicit user request.
- Sieve1 files should be disposable probes, not production strategies.
- Sieve1 results have no pass/fail gate.
- Good Sieve1 rows become candidates for deeper strategy work.
- Bad Sieve1 rows can be archived or ignored.

## 8. Entry Sieve Flow

- UI writes an Entry Sieve job JSON.
- Job contains selected strategies, windows, pairs, epochs, TP/SL, and worker settings.
- Runner creates a runtime copy of each strategy file.
- Runner hyperopts only `family:entries`.
- Runner writes best entry params beside the runtime copy.
- Runner backtests the runtime copy on validation windows.
- Runner appends one JSONL row per strategy/training-window/validation-window/TP-SL.
- UI loads results from `results/*.jsonl`.
- Latest result pointer is stored in `latest.json`.

## 9. Entry Sieve Controls

- Strategy batch:
  - Runs one configured batch from `sieve_strategy_batches.json`.
  - Use this for focused family runs.
- Batch queue:
  - Comma-separated batch ids.
  - Runs batches sequentially.
  - Good for overnight or unattended sweeps.
- Batch priority:
  - `least_run_first`: orders requested batches by fewest existing non-empty result runs.
  - `configured`: uses the entered order.
- Strategy filter:
  - Usually `sieve1_*.py`.
  - Keeps daily/TOP10 management strategies out of Sieve1.
- Speed run:
  - Forces 5 pairs, 1 auto window, target sweep off, and normal epoch resolution capped at 120.
  - Useful for faster breadth-first collection.
- Auto windows:
  - Picks 1-3 windows per strategy.
  - Uses timeframe, pattern family, pattern type, priority, and candle-count scoring.
- Target sweep:
  - Tests multiple TP/SL pairs.
  - Uses target-sweep lane distribution and the configured worker count directly.
  - Speed mode disables it.

## 10. Entry Sieve Batches

- Batch config file: `explorer/config/sieve_strategy_batches.json`.
- `all`: all active `sieve1_*.py`.
- `volume_profile`: `sieve1_vp_*.py`.
- `geometry`: `sieve1_geometry_*.py`.
- `continuation_patterns`: `sieve1_continuation_*.py`.
- `reversal_patterns`: `sieve1_reversal_*.py`, `sieve1_triple_*.py`, `sieve1_wolfe_*.py`.
- `structure_levels`: BOS, CHoCH, TLV2, prior/equal levels, ladder, pivot, liquidity.
- `market_state_pressure`: relative strength, capitulation, regime, volatility.
- `tlv2_vp`: focused `sieve1_multi2_tlv2_vp_*.py` confluence probes.
- `tlv2_boschoch`: focused `sieve1_multi2_tlv2_boschoch_*.py` confluence probes.
- `vp_prior_levels`: focused `sieve1_multi2_vp_prior_*.py` confluence probes.
- `multi_confluence`: remaining explicit `sieve1_multi*.py` confluence probes that are not split into the focused confluence batches above.
- `avwap`: `sieve1_avwap_*.py`.
- `zones`: supply/demand zone probes.
- `small_concepts`: short mixed health-check group.
- Batches can overlap; this is intentional in places such as `small_concepts`.
- Avoid assuming a batch name maps to exactly one indicator.

## 11. Batch Queue Behavior

- Queue file is written under `runtime/entry_sieve/queues`.
- Queue runner runs one batch job at a time.
- Queue runner waits for active Entry Sieve runs instead of launching duplicates.
- Queue runner rechecks for active runs before each queued batch.
- Each batch gets a separate job id and result file.
- Job id format is currently like `YYYYMMDDTHHMMSS_entry_<batch-token>`.
- Result names are intended to make entry/exit/risk layers distinguishable later.
- Queue jobs read settings when they start; already-running jobs do not live-reload code or preset changes.
- Current queue progress is stored in the queue JSON:
  - `status`
  - `phase`
  - `current_batch`
  - `current_job_id`
  - `completed_batches`
  - `failed_batches`

## 12. Backtest Scheduling

- Non-sweep mode:
  - Hyperopts keep running as the primary pipeline.
  - Backtests are queued behind completed hyperopts.
  - During hyperopt, backtest launches are capped at 3 lanes.
  - After all hyperopts finish, final backtest drain defaults to `hyperopt_jobs + 3`, capped by configured lanes.
  - With `hyperopt_jobs = 15`, this means 3 backtests during hyperopt and 18 backtests during final drain.
  - New backtest starts are staggered by about 15 seconds.
  - New backtests do not launch if RAM usage is at/above 80%.
  - Scheduler changes apply to newly launched jobs; active jobs keep the code/settings they loaded at process start.
- Target-sweep mode:
  - Uses target-sweep lane distribution.
  - Can use the configured worker count directly.
- Worker venvs:
  - Controller: `C:\FreqTradeStuff\.venv`.
  - Backtest lanes: `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest` plus `freqtrade-backtest-01` through `freqtrade-backtest-19`.
  - Current lane ceiling: 20 dedicated backtest venvs.
  - Do not store OHLCV data/configs/results inside worker venvs.
- Run locking:
  - `runtime/entry_sieve/entry_sieve.lock` is the hard single-run lock.
  - Direct runner starts refuse to run when another live Entry Sieve job exists.
  - UI launch buttons preflight against the same live-run state.
  - `active.json` is display state, not locking authority.
  - Closing the UI terminates Entry Sieve subprocess trees launched by that UI.
- Console feedback:
  - UI-owned Entry Sieve runs stream stdout through the normal Raw Console path.
  - Batch-queue jobs tee child-runner stdout to `runtime/entry_sieve/logs/<job_id>.log`.
  - The UI polls Entry Sieve queue/status state and tails the active job log when the run is background-owned.

## 13. Auto Window Selection

- Auto-window manifest: `explorer/config/sieve_auto_windows.json`.
- Market-window fallback: `explorer/config/market_windows.json`.
- Strategy profile is inferred from:
  - file name
  - class name
  - `TIMEFRAME`
  - class `timeframe`
  - `ENTRY_MODE`
  - `ENTRY_TAG`
  - `core_behavior`
- Window selection scores:
  - timeframe match
  - pattern family match
  - pattern type match
  - window priority
  - enough candles for the strategy timeframe
  - penalty for excessive length
- Pattern files need pattern-rich windows.
- Generic indicator files can use generic/regime windows.
- Validation should normally stay `full_cycle_2020_2026`.

## 14. Result Review

- Result dropdown loads non-empty result batches.
- Result file filter filters the result-file list, not rows.
- Row filter filters loaded rows.
- Multiple result batches can be loaded together for comparison.
- Result rows include:
  - score
  - batch
  - speed mode
  - winrate
  - TP/SL
  - profit
  - trade count
  - status
  - strategy
  - side/core behavior
  - windows
  - params/backtest files
- Score is a quick ranking heuristic, not a proof.
- Score currently combines:
  - sample-adjusted winrate
  - TP/SL expectancy in R terms
  - profit percentage
  - activity/trade-count bonus
- Score penalizes:
  - error rows
  - zero trades
  - low activity
- Strong rows should still be reviewed manually.

## 15. Strategy File Rules

- Current entry files use `sieve1_` prefix.
- Current entry classes use `Sieve1` prefix.
- One core entry concept per normal Sieve1 file.
- Confluence probes are special:
  - filename `sieve1_multiN_*`
  - class `Sieve1MultiN*`
  - clearly labelled as confluence.
- Do not merge many unrelated entry ideas into one file.
- Do not split every guard combination into separate files.
- Optional guards can be hyperoptable enable/disable params.
- Guards should be broad and coarse, not fine-tuned decimals.
- Good guards:
  - volume spike/rising volume
  - directional volume pressure
  - accumulation/distribution pressure
  - candle direction
  - close-vs-previous-close direction
  - reclaim/rejection/cross behaviour around project-indicator levels
  - small N-candle follow-through above/below project-indicator thresholds
  - price action confirmation
- Avoid standard indicators by default:
  - RSI, MACD, Bollinger, etc are not preferred.
  - Moving averages, EMA/SMA trend filters, and generic mean-reversion bands are not Sieve entry guards.
- Prefer project indicators and direct price-action logic.
- Strategy files must be standalone Freqtrade strategy modules.
- Do not use parent strategy classes, mixins, or external strategy helper files for hyperopted strategies.
- Indicator modules under `user_data/Indicators/` are allowed dependencies.
- Do not edit indicators from strategy-agent work unless explicitly asked.

## 16. Strategy Parameter Rules

- Hyperopt only a small focused surface per file.
- Think broad strokes, not precision fitting.
- Good file shape:
  - one entry trigger concept
  - 2-5 direct levers for that concept
  - a few optional validation guards
  - coarse categorical/discrete ranges
  - enable flags for guards
- Bad file shape:
  - many unrelated indicator surfaces
  - 20-40 tunables at once
  - fine decimal ranges everywhere
  - hidden indicator retuning
  - multiple unrelated entry concepts without `multiN`
- Entry tags should describe the entry idea.
- Avoid vague generator-origin names such as `test` or `codex`.
- Keep startup candles to the real minimum needed.
- Do not inflate startup candles just because an indicator has optional long windows.

## 17. Indicator Boundary

- Indicators are contracts.
- Indicator defaults take priority.
- Strategy agents adapt to current indicator outputs.
- Strategy agents cannot retune or reshape indicators.
- Strategy agents may suggest:
  - missing outputs
  - useful extra levers
  - unclear contracts
  - suspected indicator bugs
- Strategy agents must not implement those indicator changes without explicit approval.
- Indicator strategy-facing docs should be read before generating new Sieve files.

## 18. Architecture Principles For Agents

- Read current docs and config before coding.
- Check local definitions for:
  - sieve
  - strategy
  - entry
  - exit
  - guard
  - acceptance
  - result interpretation
- If docs and code disagree, report the mismatch before broad edits.
- Keep changes scoped.
- Avoid persistent workaround logic.
- Prefer one-time migrations for stale state/names/results.
- Do not revert user changes.
- Do not kill active runs unless user asks.
- Before changing scheduling, inspect:
  - `active.json`
  - `status/*.json`
  - `queues/*.json`
  - running Python command lines
- Before changing strategy generation, inspect existing Sieve files and batch config.

## 19. Common Operations

- Launch UI:
  - from `user_data/Custom_Launcher`
  - `python -B -m launcher_v2.app`
- Run one Entry Sieve batch:
  - Explorer tab -> Entry Sieve -> choose batch -> Run Entry Sieve.
- Run queued batches:
  - Enter comma-separated batch ids.
  - Pick priority.
  - Run batch queue.
- Monitor active run:
  - inspect `runtime/entry_sieve/active.json`
  - inspect matching `status/<job_id>.json`
  - inspect current `queues/<queue_id>.json`
- Compare results:
  - use result file filter to narrow result files.
  - select/load multiple result files.
  - use row filter and sort by score/winrate/profit/trades.

## 20. Troubleshooting Notes

- If a result file is empty, it should not show in review.
- If a run errors before rows are written, delete empty/debug outputs after inspection.
- If `active.json` exists but PID is gone, status reconciliation should mark stale.
- If batch order looks wrong, check `batch_queue_priority`.
- If a batch includes surprising filenames, inspect `sieve_strategy_batches.json` include patterns.
- If auto windows look wrong, inspect inferred timeframe/family from strategy filename/class/constants.
- If Freqtrade warns about excessive startup candles, inspect strategy `startup_candle_count` and actual indicator lookbacks.
- If throughput drops:
  - check RAM percentage.
  - check hyperopt process count.
  - check waiting/running backtest batch counts.
  - check whether target sweep is enabled.
  - check worker venv paths.

## 21. Future Layers

- Current naming reserves `entry` in result/job names.
- Future layers may include:
  - exits
  - leverage
  - position sizing
  - convergence
  - risk layers
  - adds/peels
- Do not implement future sieve layers without explicit request.
- Keep naming layer-aware so results remain filterable later.
