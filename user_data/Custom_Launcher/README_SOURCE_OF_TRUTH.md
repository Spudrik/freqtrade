# Custom_Launcher Source Of Truth

Active launcher:

```text
Custom_Launcher/launcher_v2/app.py
```

Run from `user_data/Custom_Launcher`:

```text
python -B -m launcher_v2.app
```

## Canonical Operational Paths

```text
launcher_v2/config/presets.json
launcher_v2/runtime/launcher_v2_state.json
explorer/explorer_runner.py
explorer/explorer_support.py
explorer/tag_catalog.py
explorer/config/market_windows.json
docs/EXPLORER_ENTRY_SIEVE_MANUAL.md
orderbook/collector.py
orderbook/metrics.py
orderbook/store.py
orderbook/config/sources.json
research/collectors/
research/config/
```

FreqUI launch surface is the Run tab utility (`Launch FreqUI`), not a standalone tab.

## Root Policy (`Custom_Launcher/`)

Keep only:

```text
README_SOURCE_OF_TRUTH.md
MIGRATION_NOTES.md
```

All operational code/config belongs in organized subdirectories.

## Runtime/Data Policy

Generated output is now isolated by purpose:

```text
user_data/research_news_data/news
user_data/research_news_data/web
user_data/orderbook_data/live
user_data/explorer_reports
```

`user_data/runtime/` is no longer a mixed source of truth for these tools.
