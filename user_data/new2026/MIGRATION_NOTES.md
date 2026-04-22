# New2026 Migration Notes

This worktree starts from `upstream/stable` and keeps only the New2026 work that should
survive the cleanup.

## Canonical Layout

- Strategy: `user_data/strategies/HybridRecoveryGridStrategy_v11.py`
- Strategy params: `user_data/strategies/HybridRecoveryGridStrategy_v11.json`
- Hyperopt losses: `user_data/hyperopts/hyperopt_custom.py` and
  `user_data/hyperopts/hyperopt_custom_profit_winrate.py`
- Launcher/tooling: `user_data/new2026/`
- Launcher compatibility shim: `New2026/freqtrade_launcher.py`
- Runtime bulk output: `user_data/runtime/`
- Runtime data, backtest results, hyperopt results, logs, caches, and research output are
  intentionally ignored by git.

## Custom Hyperopt Printing

`freqtrade/optimize/hyperopt/hyperopt.py` has a small custom patch that logs one compact
summary after each evaluated epoch. The intent is to make launcher monitoring easier by
printing:

- epoch number and total epochs
- best/random flags
- loss
- total trades
- winrate
- absolute and percent profit
- max drawdown

The patch also contains an optional fuller report path that extracts grid usage from order
tags such as `seed`, `add_L02`, `rebuy_L02`, and `peel_L02`. That report summarizes entry,
rebuy, and peel level usage across levels 1 through 9.

This patch touches upstream Freqtrade code directly, so it should be reviewed whenever
upstream `hyperopt.py` changes.

## Custom Hyperloss

`user_data/hyperopts/hyperopt_custom.py` preserves the older custom loss intent from the
old workspace. It biases optimization toward:

- positive daily profit
- winrate above roughly 60%
- reasonable trade frequency
- lower relative drawdown

The loss has separate branches for losing runs, low-trade profitable runs, non-ideal
profitable runs, and preferred profitable/high-winrate runs. It uses `calculate_underwater`
to estimate drawdown impact.

`user_data/hyperopts/hyperopt_custom_profit_winrate.py` preserves the newer
`ProfitWinrateModerateRiskHyperOptLoss`. It keeps profit and winrate as the primary
objective, uses trade count as a low-activity guardrail, and treats drawdown as a moderate
risk-shaping penalty instead of the main score.

## Launcher Runtime Paths

The launcher presets should point at the clean worktree root, `user_data`, and
`user_data/data/binance`. Explorer receives an explicit `--strategy-param-file` argument
pointing at `user_data/strategies/HybridRecoveryGridStrategy_v11.json`.

Launcher runtime files are now intentionally redirected to `user_data/runtime`:

- News collector data: `user_data/runtime/news`
- Web collector data: `user_data/runtime/web`
- Order book lab data: `user_data/runtime/orderbook`
- Explorer metadata and temp backtests: `user_data/runtime/explorer_metadata` and
  `user_data/runtime/tempbacktest`

## Launcher Surface Cleanup

- The standalone Plotting launcher surface is removed.
  - Plot/result exploration is expected to happen in FreqUI.
- The standalone Mirror/source-mirror launcher surface is removed.
- The standalone FreqUI tab is removed.
  - FreqUI is now exposed as a compact utility on Mode Options (`FreqUI URL` + `Launch FreqUI`).

## Hyperopt UI Semantics (Updated)

- `Job workers` now uses direct core counts in the launcher UI.
  - Example: `1` = one core, `20` = twenty cores.
  - Legacy negative values from older presets (such as `-1`, `-4`) are converted to direct core counts when loaded.
- `Epochs` / `Epoch override` fields are now `Epoch multiplier`.
  - Effective epochs are computed as: `job workers * epoch multiplier`.
  - Example: workers `10` and multiplier `10` runs `100` epochs.
- Any effective hyperopt epoch count is capped at `800`.
  - This cap is applied in launcher-generated commands, Explorer support flow, and core hyperopt runtime.

## Removed From The Clean Base

The old `GPT_GUI`, `Backups`, and package-internal `freqtrade/user_data` folders are treated
as obsolete unless a future check finds a direct dependency from the New2026 files. The old
large `.fthypt`, backtest zip, market data, and research output files are generated artifacts
and should not be committed.
