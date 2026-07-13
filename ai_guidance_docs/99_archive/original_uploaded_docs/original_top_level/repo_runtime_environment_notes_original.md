# Repo Runtime Environment Notes

## AI guidance documents

- Focused AI guidance docs live under `C:\FreqTradeStuff\ai_guidance_docs`.
- Treat `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md` as the canonical current objective list.
- Any user or agent mention of "objectives" means first refer to `C:\FreqTradeStuff\ai_guidance_docs\current_objectives.md`.
- Do not add progress notes, queue status, implementation logs, or result details to `current_objectives.md`.
- Track progress/status in `C:\FreqTradeStuff\ai_guidance_docs\objectives_progress.md`.
- Track broad concise findings in `C:\FreqTradeStuff\ai_guidance_docs\objectives_high_level_results_summary.md`.
- Track detailed context/orderbook/FreqAI research findings in `C:\FreqTradeStuff\ai_guidance_docs\context_research_detailed_findings.md`.
- Read `C:\FreqTradeStuff\ai_guidance_docs\objective_examples.md` before designing confluence features, hypotheses, labels, or reports.
- Read `C:\FreqTradeStuff\ai_guidance_docs\result_communication_format.md` before explaining FreqAI, direct-test, confluence, orderbook, structure, or feature-discovery results to the user.
- Read `C:\FreqTradeStuff\ai_guidance_docs\readiness_gated_feature_research_goal.md` before using goal mode or launching a long-running agent to discover feature definitions while some data sources are incomplete.
- Read `C:\FreqTradeStuff\ai_guidance_docs\concept_lifecycle_research_process.md` before running iterative trader-confluence research, creating broad test cycles, deciding whether a concept should be promoted/parked/reworked/rejected, or consolidating old scattered reports and queues into the formal tracking structure.
- For production-alpha strategy development or next-test selection, read `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_strategy_decision_snapshot.md` first, then read `C:\FreqTradeStuff\ai_guidance_docs\production_alpha_finish_line_lanes.md`, then open only the detailed report for the lane being changed.
- Orchestrator agents that manage sub-agents or split confluence research work should also read `C:\FreqTradeStuff\ai_guidance_docs\agents_structure.md`.
- Use `C:\FreqTradeStuff\ai_guidance_docs\README.md` as the document map. Read only the focused docs relevant to the current task.
- The agent's job is to check that objectives are being completed, not to rely on chat memory.
- If an objective appears truly 100% complete, flag it to the user for removal approval. Do not remove completed objectives without explicit user approval.

## Python environments

- `C:\FreqTradeStuff\.venv` is the controller environment. Use it for PyCharm and for launching `user_data\Custom_Launcher\launcher_v2\app.py`.
- `C:\FreqTradeStuff\runtime\venvs\fraeqtrade-backtest` is the first dedicated Freqtrade backtest worker environment.
- `C:\FreqTradeStuff\runtime\venvs\freqtrade-backtest-01` through `freqtrade-backtest-08` are additional backtest worker lanes.
- Worker venvs are local ignored installs only. They must not hold OHLCV data, configs, launcher state, reports, or backtest result archives.
- Each worker venv should have `freqtrade` installed in editable mode from `C:\FreqTradeStuff` so it can run independently while still using the main repo source.

## Backtest lane rules

- The split-venv worker count means the number of dedicated worker venvs to use, from 1 to 9. Do not count the controller `.venv` or system Python as backtest lanes.
- Explorer and Entry Sieve split-venv runs should treat the worker count as a concurrency limit, not as permission to launch unbounded subprocesses.
- Backtest tasks must be queued across configured lanes and keep result directories isolated per task/window.
- Do not add interpreter guessing or fallback behavior. If a configured Python executable is missing, fail early with the missing path.
- Keep the backtest worker venvs under `runtime\venvs`; do not move them into `user_data` or the launcher package.

## External context research ledger

- Keep `C:\FreqTradeStuff\ai_guidance_docs\context_research_detailed_findings.md` updated whenever adding context sources, changing feature conditioning, running FreqAI research, validating timestamp alignment, or resolving data gaps.
- Keep `C:\FreqTradeStuff\ai_guidance_docs\objectives_high_level_results_summary.md` updated only when broad findings change.
- Use the detailed findings file as the first stop after context compaction before repeating news, web, global, GDELT/GKG, orderbook, or confluence research work.
- Record both model results and data coverage/timestamp flaws. Do not remove open issues without adding a resolved entry.

## Bybit historical raw archive locations

- Historical Bybit raw ZIP archives are intentionally stored on `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`, not under `C:\FreqTradeStuff\user_data`.
- Spot raw ZIPs live in `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\spot_raw`.
- Linear raw ZIPs live in `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`.
- Inverse raw ZIPs live in `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`.
- Features, logs, and manifests remain under `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit`.
- Do not infer incomplete Bybit raw coverage from `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw` being absent or empty.

## GDELT/GKG historical raw archive locations

- Historical GDELT/GKG raw ZIP archives are intentionally stored on `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`, not under `C:\FreqTradeStuff\user_data`.
- GKG raw ZIPs live in `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
- GDELT event export raw ZIPs live in `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`.
- The SQLite database remains at `C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite`.
- Do not infer incomplete GDELT/GKG raw coverage from `C:\FreqTradeStuff\user_data\research_news_data\gdelt\raw` being absent or empty.

## Research test discipline

- Before running context/FreqAI research tests, state the objective and pass/fail condition first.
- When reporting research results, use the simple trader-readable result format from `ai_guidance_docs\result_communication_format.md`: trader question, what was observed, what was asked afterwards, plain-English result, supporting numbers, verdict, next step.
- For feature-definition research while data is incomplete, follow `ai_guidance_docs\readiness_gated_feature_research_goal.md`: audit readiness first, test only usable source windows, park unavailable sources, and stop rather than wandering into unachievable full confluence.
- Prefer low-dimensional, hypothesis-led tests with controls before broad FreqAI runs.
- Pause relevant collectors and export parquet snapshots before research runs that would otherwise read live SQLite databases.
- Treat FreqAI as a validation step after a feature family shows stable direct-test evidence, not as the first fishing tool.
- For confluence research, define trader-readable behaviour first, then encode it numerically, direct-test it, and only then promote promising states into FreqAI queues.
- Treat overlapping signals as position-management evidence, not only duplicate entries: same-direction overlap may support add/hold/confidence, while opposite-direction overlap may support reduce/tighten/exit. Keep this flexible and test it as signal weighting, not a hard rule.
- For iterative concept research, follow `ai_guidance_docs\concept_lifecycle_research_process.md`: keep concept cards, test across branch quotas, park working concepts, rework failures, add new ideas, and consolidate old evidence into the formal lifecycle ledger instead of relying on scattered reports or queue names.

## External context feature design priority

- Build context datasets around how a competent trader would describe market behaviour, not just raw counts or isolated values.
- For every data family, first ask what real-world state or transition the feature represents, then encode that state numerically.
- Examples:
  - Orderbook: persistent resistance, support zones, wall evaporation, pressure shock, liquidity vacuum, spread shock, post-breakout support rebuild.
  - News/web/GDELT: event severity, source confluence, persistence of topic coverage, first mention versus follow-through, plausible market impact channel.
  - Global/macro: surprise versus expectation, risk-on/risk-off pressure, rate/liquidity shock, persistence across sessions, cross-market confirmation.
- Prefer compact descriptive features such as `resistance_removed_score`, `topic_confluence_24h`, or `macro_liquidity_shock` over hundreds of weak raw columns.
- Do not hardcode "bullish" or "bearish" article/trader opinions as truth. Encode observable facts and let direct tests/FreqAI decide whether they matter.
- Treat the advanced trader-confluence objectives as superseding but not deleting the earlier Objective List Alpha orderbook baseline.
- Do not reduce complex user examples to one indicator, one source, or one timeframe. Confluence across structure, orderbook, volume, context/news, and multiple timeframes is the end goal.
- For raw orderbook archive work, never carry book state, zone state, or hourly features through missing archive gaps. Each archive/day must start from its own snapshot, and sparse historical coverage must remain sparse in FreqAI/test datasets.
