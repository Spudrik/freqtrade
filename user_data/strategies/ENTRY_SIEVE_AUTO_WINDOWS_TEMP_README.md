# Entry Sieve Auto Windows - Temp Review Notes

## Current Implementation

1. Entry Sieve now defaults to automatic training-window selection.
2. The UI exposes a single user-facing count: `1`, `2`, or `3` hyperopt windows per strategy file.
3. Auto mode ignores manually selected training windows.
4. Auto mode forces validation/backtest selection to `full_cycle_2020_2026`.
5. Manual mode is still available by disabling `Auto windows`.

## Window Selection Logic

1. The runner parses each strategy file with `ast` and reads strategy-facing constants such as `TIMEFRAME`, `ENTRY_MODE`, and `ENTRY_TAG`.
2. It infers the strategy timeframe from constants first, then filename suffixes such as `_1h`, `_4h`, `_8h`, `_1d`, or `_3d`.
3. It infers the entry family from names/tags:
   - continuation: `flag`, `pennant`, or generic `continuation`
   - geometry: `triangle`, `wedge`, `rectangle`, `channel`, `compression`
   - reversal: `double_top`, `double_bottom`, `head_shoulders`, `inverse_head_shoulders`, `choch`, `capitulation`
   - multi peak: `triple_top`, `triple_bottom`
   - Wolfe: `wolfe`
   - generic/supporting families: `volume_profile`, `relative_strength`, `trendline`, `prior_levels`, `avwap`
4. Candidate windows are ranked by exact pattern match, family match, timeframe match, priority, and candle count.
5. Cross-family pattern windows are penalized so a generic fallback should beat an unrelated pattern window.

## Window Sources

1. Generic/manual regime windows still come from `user_data/Custom_Launcher/explorer/config/market_windows.json`.
2. Auto Sieve seed windows now come from `user_data/Custom_Launcher/explorer/config/sieve_auto_windows.json`.
3. The new auto manifest contains:
   - generic timeframe-aware cycle windows
   - continuation windows
   - geometry windows
   - reversal/H&S windows
   - multi-peak windows
   - Wolfe windows
4. The auto windows are seed windows only. They are not intended to be the final acceptance test.

## Verification Done

1. Python syntax compile passed for the modified service, runner, and UI files.
2. JSON validation passed for the new auto-window manifest.
3. Service validation passed for auto mode with `3` windows per file.
4. Full `sieve1_*.py` dry selection succeeded:
   - 446 strategy files
   - 3 windows per file
   - 1,338 planned hyperopt rows
   - no auto-window assignment failures
5. Spot check examples:
   - `sieve1_vp_entry_trigger_long_1h` selected 1h generic windows.
   - `sieve1_continuation_flag_present_long_1d` selected a daily continuation window plus daily generic fallbacks.
   - `sieve1_wolfe_bullish_present_long_4h` selected 4h Wolfe windows plus a 4h generic fallback.

## Smoke Test Notes

1. A filtered 1-epoch auto smoke used exactly three strategy files:
   - `sieve1_continuation_flag_present_long_1d.py`
   - `sieve1_vp_entry_trigger_long_1h.py`
   - `sieve1_wolfe_bullish_present_long_4h.py`
2. The first attempted smoke exposed two runner handoff bugs:
   - Freqtrade commands inherited the preset timeframe instead of using the strategy timeframe.
   - Runtime-copied strategy files could not import top-level indicator helpers such as `pattern_common`.
3. Fixes applied:
   - Entry Sieve now writes the inferred strategy timeframe into the runtime preset used by hyperopt/backtest.
   - Child Freqtrade processes now receive the real `user_data/Indicators` path in `PYTHONPATH`.
4. The rerun passed:
   - `3/3` hyperopts completed.
   - `3/3` backtests completed.
   - runtime strategy copies were resolved by Freqtrade.
   - generated runtime parameter JSON files were loaded by backtests.
5. A manual-window compatibility smoke also passed with Auto windows disabled.
6. A target-sweep smoke passed with one file and two effective TP/SL rows. The baseline `take_profit_pct/stoploss_pct` pair is always included, and duplicate sweep pairs are deduped.

## Review Points

1. Reversal strategies currently share reversal/H&S-heavy windows where no more specific double-top or double-bottom window exists.
2. Non-pattern structure ideas such as CHOCH/capitulation are treated as reversal-family entries.
3. Volume-profile, AVWAP, relative-strength, trendline, prior-level, ladder, zone, volatility, and mixed-confluence probes mostly use generic timeframe windows.
4. If an indicator agent gives better exact pattern date ranges later, add them to `sieve_auto_windows.json` rather than changing strategy files.
