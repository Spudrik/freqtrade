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
- Use clear prefixes for experimental strategy files so they are easy to group and filter. Current sieve strategy files must use the active sieve pass prefix and class prefix consistently, for example `sieve2_` files with `Sieve2` classes.
- Strategy files intended for HyperOpt must be standalone modules. Do not use parent strategy classes, mixin strategy bases, or external strategy helper files in hyperopted strategy logic.
- Sieve results are informative diagnostics for refining entry signals; do not treat them as pass/fail or acceptance decisions.
- Current Sieve2 work may test promising Sieve1 entries with better guard structure, multi-timeframe context, and cleaner confluence. Do not add exits, adds, leverage, or broader strategy-management logic unless explicitly requested.
- Sieve progression is generation-based, not idea-age-based. When Sieve3 is active, refined Sieve2 foundations and totally new ideas both use `sieve3_` files/classes.
- Refined Sieve3 files must preserve lineage with metadata such as `SIEVE_STAGE`, `SOURCE_STRATEGY`, `SOURCE_RESULT_BATCH`, and `RESEARCH_PATH`.
- New ideas during Sieve3 should use `sieve3_novel_*` naming and explicit novel metadata such as `NOVEL_IDEA = True`.
- One source strategy may branch into multiple Sieve3 files when testing distinct refinements. Keep branches narrow and named by path, for example VP guard, TLV2 guard, HTF context, retest, local structure, VP+TLV2 guard, or retest+HTF guard.
- Keep all interesting foundations available for next-generation work, not only the currently discussed subset. Interesting foundations include high-winrate low-trade ideas, profitable high-trade sub-50% winrate ideas, strong profit relative to trade count, strong `4/2` target behaviour around 40%+ winrate, low-drawdown stable ideas, and structurally sensible near-misses with clear repair paths.
- Do not promote or archive solely from a synthetic score. Use result shape, trade count, drawdown, target profile, and trading logic quality together.
- Weak strategies may be archived out of active batches after result review and logic review, but do not park a file silently. Add a short module-level comment to each archived strategy explaining why it was moved.
- Archive comments should make later review easy by naming the main reason, for example consistently negative results, excessive noisy trade count, generic/non-structural trigger, redundant duplicate, lookahead-risk concern, or no clear repair path.
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
- Entry mode shape is a strategy-level decision, but MTF sieve files must use a real hierarchy: higher-timeframe trading condition plus lower-timeframe structural execution.
- Useful MTF higher-timeframe contexts include 1d/3d/4h patterns, TLV2 levels, volume-profile VAH/VAL/POC/HVN/LVN behaviour, BOS/CHOCH, HH/HL/LH/LL, supply/demand, and prior high/low levels.
- Useful lower-timeframe execution includes breakout, break-and-retest, rejection/reclaim, failed break, local BOS/CHOCH, or higher-low/lower-high continuation at the relevant level.
- Retests are first-class variants, especially for TLV2, volume-profile levels, and 1d/3d pattern lines such as flags, pennants, head-and-shoulders, triangles, channels, and rectangles.
- Do not use generic momentum, one-candle close direction, EMA/SMA pullback, or oscillator/average-cross logic as the primary trigger for MTF entries. Those can only support a structural trigger as optional guards.
- MTF strategy names and tags must state the higher timeframe/concept and lower timeframe/execution trigger.
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
