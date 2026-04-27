# Custom_Launcher Migration Notes

## Canonical Layout

- Launcher/tooling root: `user_data/Custom_Launcher/`
- Active launcher: `user_data/Custom_Launcher/launcher_v2/app.py`
- Explorer core: `user_data/Custom_Launcher/explorer/`
- Orderbook core: `user_data/Custom_Launcher/orderbook/`
- Research collectors/config: `user_data/Custom_Launcher/research/`

## Runtime/Data Split (27/04/2026)

Mixed `user_data/runtime/` storage was split into isolated folders:

- News data: `user_data/research_news_data/news`
- Web data: `user_data/research_news_data/web`
- Orderbook data: `user_data/orderbook_data/live`
- Explorer summaries/audits/state: `user_data/explorer_reports/`
- Launcher state: `user_data/Custom_Launcher/launcher_v2/runtime/launcher_v2_state.json`
- Indicator external validator runtime output: `user_data/Indicator_External_Validator/`

Removed as obsolete runtime clutter:

- `source_mirror`
- `indicator_validation_test`
- `indicator_validation_test_base`
- `_inspect_bt_*`
- `_patch_tests`
- `tempbacktest`

## Launcher Surface

- Monolithic launcher remains retired.
- Active work targets `Custom_Launcher/launcher_v2/`.
- FreqUI launch surface is on the Run tab utility (`Launch FreqUI`), not a standalone tab.
- File Converter is integrated in LauncherV2 tab surface.

## Explorer/Orderbook/Research Path Notes

- Explorer default files:
  - `launcher_v2/config/presets.json`
  - `explorer/config/market_windows.json`
  - `explorer/explorer_runner.py`
  - `user_data/explorer_reports/latest_summary.json`
  - `user_data/explorer_reports/hyperopt_explorer_state.json`
- News/Web collector defaults now target `user_data/research_news_data/*`.
- Orderbook collector defaults target `user_data/orderbook_data/live`.

## Indicator External Validator (Draft)

- Launcher tab: `Indicator External Validator`
- Runner: `Custom_Launcher/indicator_validation_runner.py`
- Runtime output: `user_data/Indicator_External_Validator/`
- Draft documentation: `Custom_Launcher/docs/INDICATOR_EXTERNAL_VALIDATOR_DRAFT.md`

This feature is explicitly marked as draft and subject to deletion/rework.

## Root Policy

Only these source-of-truth files stay at `Custom_Launcher` root:

- `indicator_validation_runner.py`
- `README_SOURCE_OF_TRUTH.md`
- `MIGRATION_NOTES.md`
