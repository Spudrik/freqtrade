# Entry Sieve Agent Notes

## Purpose

Entry Sieve is a disposable first-pass research workflow for comparing entry conditions across top-level strategy files.

The intent is:

- Use selected Explorer market windows as training windows.
- Hyperopt only `family:entries` parameters.
- Validate each trained result against the selected validation window(s), normally `full_cycle_2020_2026`.
- Keep capital and management simple: fixed stake, no leverage experiments, no adds, no peels.
- Keep exits standardized so results mainly reflect entry/filter quality.
- Store generated metadata under `launcher_v2/runtime/entry_sieve` so it can be deleted later.

This is not meant to replace normal Explorer. It reuses Explorer's lower-level command helpers, but it does not use Explorer's champion/challenger acceptance loop.

## Current Linkage

Entry Sieve touches three areas.

Strategy files:

- `user_data/strategies/entry_sieve_tools.py`
- `user_data/strategies/test_entry_research_base.py`
- `user_data/strategies/DailyStructureLadderStrategyTOP10.py`
- `user_data/strategies/DailyStructureShortModesTOP10Normal.py`
- `user_data/strategies/DailyPivotBreakoutLongTOP10.py`
- `user_data/strategies/codex_*.py`

Launcher UI/orchestration:

- `launcher_v2/tabs/explorer_tab.py`
- `launcher_v2/services/entry_sieve_service.py`
- `launcher_v2/services/entry_sieve_runner.py`

Shared Explorer catalog support:

- `explorer/tag_catalog.py`
- `explorer/explorer_support.py`

## How It Runs

The `Entry Sieve` subtab lives inside the existing Explorer tab.

Flow:

1. The UI reads the same selected training and validation windows as normal Explorer.
2. `EntrySieveService` discovers top-level strategy files.
3. It excludes helper/archive/non-sieve files, especially `PivotTrendlineMTFResearchStrategy.py`.
4. It writes a disposable job JSON under `launcher_v2/runtime/entry_sieve`.
5. `EntrySieveRunner` creates a runtime strategy copy per strategy/window.
6. It writes the best hyperopt params next to that runtime strategy copy so Freqtrade definitely loads the tuned params during backtesting.
7. It runs HyperOpt directly through `explorer.explorer_commands.run_hyperopt`.
8. It filters the best epoch to resolved `family:entries` params.
9. It backtests the tuned runtime strategy copy through `explorer.explorer_commands.run_backtest`.
10. Entry Sieve appends one result row per strategy/training-window/validation-window to `launcher_v2/runtime/entry_sieve/results.json`.
11. The UI reads `results.json` and provides sorting/filtering.

There is no accepted/rejected decision in Entry Sieve. A row is only marked `error` if HyperOpt or backtesting actually fails.

## Tagging Rules

Entry Sieve does not introduce a new tag system.

It relies on the existing standard:

- Only `family:*` and `mode:*` tags.
- Families are only:
  - `entries`
  - `exits`
  - `adjust_position`
  - `stake`
  - `risk`
- Entry Sieve targets only `family:entries`.

## Why `tag_catalog.py` Was Touched

Normal Explorer previously relied mostly on AST parsing. That misses some valid strategy surfaces:

- inherited parameters in split `test_` strategies
- tags applied after class definition
- mixin-provided surfaces

Entry Sieve needs those to be visible because the split strategy files are intentionally thin wrappers around parent strategies.

The catalog now loads the runtime strategy class and reads actual parameter objects when possible. This is a source-of-truth improvement, not a second tagging system.

If this causes issues in normal Explorer, revert only the runtime-loading portion of `explorer/tag_catalog.py`. Normal Explorer will return to AST-only behavior, but inherited split-strategy params may disappear from the catalog again.

## Strategy Baseline

Sieve-prepped strategies should use:

- ROI target: `+2%`
- stoploss: `-2%`
- `use_exit_signal = False`
- `use_custom_stoploss = False`
- `position_adjustment_enable = False`
- `max_entry_position_adjustment = 0`
- no leverage scaling
- no custom stake scaling
- no adds or peels

`PivotTrendlineMTFResearchStrategy.py` is intentionally excluded and must not be changed by Entry Sieve work.

## Runtime Metadata

Disposable files live here:

- `launcher_v2/runtime/entry_sieve/jobs`
- `launcher_v2/runtime/entry_sieve/runs`
- `launcher_v2/runtime/entry_sieve/params`
- `launcher_v2/runtime/entry_sieve/results.json`

Current job JSON files are written directly under `launcher_v2/runtime/entry_sieve` using the job id as the filename.

Runtime strategy copies are stored under `launcher_v2/runtime/entry_sieve/runs/<strategy>__<training_window>/strategy`.

These are not source-of-truth strategy files. They can be deleted when cleaning up Entry Sieve experiments.

## Untangling / Removal Plan

To remove Entry Sieve without breaking normal Explorer:

1. Remove the Entry Sieve UI from `launcher_v2/tabs/explorer_tab.py`.
2. Delete `launcher_v2/services/entry_sieve_service.py`.
3. Delete `launcher_v2/services/entry_sieve_runner.py`.
4. Delete `launcher_v2/runtime/entry_sieve`.
5. Keep or revert the runtime catalog enhancement in `explorer/tag_catalog.py` depending on whether normal Explorer should continue seeing inherited/runtime-applied strategy params.
6. If reverting the catalog enhancement, also review `explorer/explorer_support.py` for any sieve-specific active-param filtering.
7. Remove `user_data/strategies/entry_sieve_tools.py` only after removing imports from all strategy files that use it.
8. Restore strategy baseline values if the strategies should return to non-sieve behavior.

Safe minimal removal:

- Remove only the UI and `entry_sieve_*` launcher services.
- Keep strategy baseline and catalog changes.
- This removes the feature surface but keeps normal Explorer working with the improved catalog.

Full removal:

- Remove the UI, services, runtime folder, strategy helper, strategy imports, and catalog runtime loading.
- This returns the codebase closer to pre-sieve behavior but may make inherited split-strategy params invisible again.

## Current Risk / Incomplete Work

The Entry Sieve feature was added while still in implementation. Before relying on it for a long run:

- Confirm the UI subtab opens.
- Confirm one short low-epoch Entry Sieve run completes.
- Confirm `results.json` is populated after a real run.

Static syntax checks, strategy discovery, inherited/runtime catalog visibility, MTF trendline exclusion, and take-profit/stoploss environment pass-through have been smoke-tested.
The runner records neutral hyperopt/backtest metrics and no longer uses Explorer accepted/rejected logic.
