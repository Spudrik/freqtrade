# Repo Runtime Environment Notes

## Python environments

- `C:\FreqTradeStuff\.venv` is the controller environment. Use it for PyCharm and for launching `user_data\Custom_Launcher\launcher_v2\app.py`.
- `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest` is the first dedicated Freqtrade backtest worker environment.
- `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest-01` through `freqtrade-backtest-08` are additional backtest worker lanes.
- Worker venvs are local ignored installs only. They must not hold OHLCV data, configs, launcher state, reports, or backtest result archives.
- Each worker venv should have `freqtrade` installed in editable mode from `C:\FreqTradeStuff` so it can run independently while still using the main repo source.

## Backtest lane rules

- The split-venv worker count means the number of dedicated worker venvs to use, from 1 to 9. Do not count the controller `.venv` or system Python as backtest lanes.
- Explorer and Entry Sieve split-venv runs should treat the worker count as a concurrency limit, not as permission to launch unbounded subprocesses.
- Backtest tasks must be queued across configured lanes and keep result directories isolated per task/window.
- Do not add interpreter guessing or fallback behavior. If a configured Python executable is missing, fail early with the missing path.
- Keep the backtest worker venvs under `runtime\venvs`; do not move them into `user_data` or the launcher package.
