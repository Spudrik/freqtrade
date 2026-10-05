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
- For general repository safety and root-cause/workaround policy, follow repo-root `AGENTS.md` and `ai_guidance_docs/02_rules/rules_codebase_workflow.md`.
- Runtime Python environments are documented in `ai_guidance_docs/02_rules/rules_runtime_environment.md`. Keep Explorer/Sieve split-venv worker lanes under `runtime/venvs`, queue work by configured worker count, and fail early when a configured interpreter path is missing.

## Strategy Research Notes

- Prefer one trading hypothesis per strategy file.
- Keep capital/leverage experiments outside an entry/exit task unless the user explicitly requests them.
- Standardize exits when comparing entry quality; disable Sieve exit control when exit behaviour itself is being researched.
- Use clear stage prefixes. Current Sieve3 V2 exit files use `sieve3_V2_` and current refined queues use `sieve3_v2_refined_*` identifiers.
- Strategy files intended for HyperOpt must be standalone modules. Do not use parent strategy classes, mixin strategy bases, or external strategy helper files in hyperopted strategy logic.
- Sieve1 results are informative diagnostics for refining entry signals; do not treat them as pass/fail or acceptance decisions.
- Later Sieve stages may test exits, adds, global guards, or weighted composition only when the user explicitly requests that work.
- Use materiality-gated analytical depth when interpreting research results. Clear, decision-changing separation (for example, a solid profit versus a loss, without disqualifying drawdown) can be prioritized directly from the metrics. Close, mixed, or suspicious comparisons (for example, `6%` versus `5%`, or `22` row wins versus `18`) require deeper analysis before naming a winner: assess effect size, drawdown and trade-count changes, whether the variants executed comparable trade opportunities, repeatability across genuinely independent windows or runs, parameter convergence where relevant, and whether the proposed mechanism fits the trading logic. Treat small differences as equivalent or as descriptive leads until that evidence distinguishes them; do not equate rank with causation or proof.
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
