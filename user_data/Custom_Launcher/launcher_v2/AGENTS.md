# LauncherV2 Agent Rules

LauncherV2 is the destination architecture for the launcher.

## Objective

Use simple tab modules and shared helpers to replace the monolithic launcher gradually.

## Rules

- One tab = one file under `launcher_v2/tabs/`.
- Keep `app.py` small.
- Do not put feature logic into `app.py`.
- Do not create another architecture pattern.
- Do not add promotion/comparison/random-tag/namespace controls to ExplorerV2.
- Keep FreqUI launch available as a simple `freqtrade webserver` surface using Setup config/userdir/datadir.
- Any foreground process launched through `ProcessRunner` must be terminated as a process tree on Stop and app close; this includes Freqtrade/Explorer hyperopt/backtest workers. Detached news/web/orderbook collectors are intentionally outside this shutdown path.
- Do not change strategy trading logic unless the current task is explicitly strategy research work.
- Do not delete old launcher code unless the current phase explicitly says deletion is safe.
- If unsure, stop and report instead of broadening scope.
- Root-cause first: fix mapping/config/state issues at source before adding fallback/workaround code.
- Do not add runtime rescue logic by default; use one-time migration when needed.
- If a workaround is truly unavoidable, stop and ask for explicit approval before adding it.
- Runtime Python environments are documented in the repo-root `AGENTS.md`. Keep Explorer/Sieve split-venv worker lanes under `runtime/venvs`, queue work by configured worker count, and fail early when a configured interpreter path is missing.

## Strategy Research Notes

- Prefer one trading hypothesis per strategy file.
- Keep the first-pass capital model simple: fixed stake, no leverage experiments, no adds, no peels.
- Standardize exits early when comparing entry quality.
- Use clear prefixes for experimental strategy files so they are easy to group and filter. Current entry-sieve strategy files use the `sieve1_` file prefix and `Sieve1` class prefix.
- Strategy files intended for HyperOpt must be standalone modules. Do not use parent strategy classes, mixin strategy bases, or external strategy helper files in hyperopted strategy logic.
- Sieve1 results are informative diagnostics for refining entry signals; do not treat them as pass/fail or acceptance decisions.
- Future Sieve2/Sieve3/Sieve4 passes may test exits, adds, global guards, or other ideas, but do not implement those without explicit user request.
- Check that local definitions for sieve, strategies, entries, exits, guards, and result interpretation are still accurate before changing this area. Report stale definitions or deviations to the user.
- Keep the ladder and daily-structure family split into separate files when the goal is to isolate entry edge.
- Strategy parameter tags are limited to `family:*` and `mode:*` only.
- Use `family` as the top-level block, limited to only: `entries`, `exits`, `adjust_position`, `stake`, `risk`.
- Use `mode` for each isolated logic block that may be swapped/tested later, including entry, exit, and adjust-position variants.
- Mode composition is allowed and encouraged when collaboration between blocks is being tested.
- Use explicit composed mode names when combining logic, for example: `mode:x_with_exits`, `mode:x_with_risk`, or `mode:x_with_y`.
- Treat composed modes as first-class test candidates, not temporary labels.
- Each `family+mode` block should expose at least 2 tunable parameters.
- No `family` block and no `mode` block may exist with only 1 parameter.
- If a mode is sparse, merge it into a related composed mode (for example combine entry logic with exits or risk using `_with_` mode naming).
- Entry mode shape is a strategy-level decision; there is no fixed guard/trigger template required across all strategies.
- Deprecated dataframe/pandas usage is not allowed in strategy code and must be cleaned when touched.
- Revisit these notes when the research workflow changes materially.

## Data Management Context Source Backlog

- The Data Management tabs include TODO/catalog surfaces for historical context sources. These are reminders and planning surfaces, not proof that a source is ready.
- Before adding any importer/downloader, check timestamp semantics, license/provenance, storage size, and no-lookahead safety.
- Prefer extending existing systems instead of adding unique ingestion frameworks:
  - article/news-like sources should merge into News Lab or Web Lab patterns where practical;
  - numeric macro/market sources should extend Global Context patterns where practical;
  - conditioned outputs should flow into `context_features`, not strategies;
  - raw imports should stay under `user_data/research_news_data/<source_family>/`.
- Sources that still need explicit review include GDELT, Common Crawl CC-NEWS, Guardian Open Platform, NYT Archive metadata, Media Cloud, Internet Archive TV News captions, Coin Metrics Community, FRED/ALFRED vintages, EIA/oil-energy data, Reddit archives, and timestamped crypto social datasets.
- Random file-share datasets are not acceptable unless license, provenance, fields, and timestamps are clear.
- Do not add orderbook data to this context-source backlog; orderbook remains a separate task.

## Preferred migration order

1. Shell/helpers.
2. Run/Common/Pairs/Mode Options.
3. Simplified Explorer.
4. Review.
5. News/Web shared collector tabs.
6. OrderBook.
