---
doc_status: active
default_read: routed
owner: agent
purpose: Concise recent evidence summary.
do_not_use_for: Full detailed historical log.
last_rebuilt: 2026-06-11
---

# Recent Results Summary

## Current Direction

Sieve is now the only approved Hyperopt/discovery system for entries, exits, position adjustment, and risk management unless the user explicitly approves a non-Sieve exception.

Manual one-off backtests are not valid concept-discovery evidence. Backtests are validation/sanity checks after Sieve identifies candidates.

## Retired Non-Sieve Exit/Risk Branch

The recent non-Sieve exit/risk branch was not a good use of tokens compared with Sieve. It added only narrow value, created too much context clutter, and should not be resumed as an active objective.

Keep only the general lesson: Sieve should own Hyperopt/discovery. Park FreqAI, orderbook, context, confluence, and generic-TA work as optional future overlays after strong Sieve strategies exist.

## Research Lessons To Preserve

1. Custom indicators, sieve-derived entries, structure, VP, TLV2, BOS/CHoCH, pattern geometry, price action, and volume pressure are core inputs.
2. Orderbook is most useful as targeted confluence/risk/invalidation/target evidence, not a broad generic filter by default.
3. News/context remains parked until source-specific extraction, coverage, and timestamp safety are ready.
4. Exit/risk logic must now be tested through focused Sieve Hyperopt batches, not manual one-off backtest probes.
5. Entry-family-specific exits, partial exits, stop tightening, add/reduce logic, and risk sizing remain the correct research direction.
6. 2026-06-11 top-10 OHLCV correction: canonical futures OHLCV lives under `C:\FreqTradeStuff\user_data\data\binance\futures`; duplicate direct `.feather` files were removed and canonical futures files remained `80/80`. Cleanup evidence: `C:\FreqTradeStuff\ai_guidance_docs\04_results\top10_ohlcv_duplicate_cleanup_20260611.md`.

## Historical News Overlay Leads - 2026-06-15

Source-masked aggregate GDELT/GKG testing found exploratory overlay leads through end-2022. These are not strategy rules yet, but they are worth joining to Sieve entry/export rows:

1. Relief bounce: stress was high but fades near support or after selloff.
2. Bad-news breakdown continuation: lower-low break plus bad news increases downside risk.
3. Attention/compression expansion: high news attention after compression raises expansion risk; price lean helps direction.
4. High-attention breakout follow-through: breakouts with volume and high attention worked better when not overextended.
5. Bad-news absorption/strength: bad news that fails to break support, or is ignored by rising price, can mark strength rather than weakness.

Evidence: `C:\FreqTradeStuff\ai_guidance_docs\04_results\historical_news_overlay_results_20260615.md`.

## Live Formatted Snapshot FreqAI Smoke Leads - 2026-06-26

User explicitly approved a non-Sieve FreqAI feature-discovery run on the strict live aligned snapshot. The useful early signals were risk-overlay style, not standalone strategy rules:

1. Context/news ranked next-24h large drawdown risk better than price-only in the recent BTC window.
2. Market breadth also improved large-drawdown and breakdown-success ranking versus price-only.
3. Strict orderbook/cross-source coverage was too small for a strong conclusion; treat it as source-readiness-limited.

Evidence CSVs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\formatted_freqai_hypothesis_20260626_032201\formatted_freqai_hypothesis_metrics.csv` and `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\formatted_freqai_hypothesis_20260626_032247\formatted_freqai_hypothesis_metrics.csv`.

## Refined Live Snapshot Leads - 2026-06-26

The broad context/news drawdown result was split into narrower FreqAI profiles. Crypto-native and topic-risk context looked strong in the recent stress window, but older comparison windows mostly had GDELT-style archive context rather than the live news/web/global block. Do not treat those older checks as live-media validation.

The better carry-forward leads are:

1. Market-volatility as a short-risk/reward overlay. It repeatedly helped identify when short-side reward-before-danger was more likely.
2. Raw market breadth as a downside/breakdown warning. It was not perfect, but it gave the best repeatable breakdown-success separation.
3. Context topic-risk as a possible conditional overlay, not a standalone risk signal yet.

Evidence summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\refined_best_results_cross_window_summary_20260626.csv`.

## Live Media Short-Horizon Rerun - 2026-06-26

Live media now means the live news + web + global block. GDELT/GKG is archive context and was excluded from this rerun.

The valid live-media window in the current snapshot is roughly `2026-05-08 12:00 UTC` to `2026-06-11 05:00 UTC`. Main FreqAI scoring used `624` live-media-ready BTC rows from `2026-05-16` onward after enough training history existed.

Best short-horizon exploratory signals:

1. `1h` long follow-through improved with live media.
2. `1h` drawdown risk improved with live topic-risk media.
3. `1h` and `2h` short reward/risk improved with live media.
4. `2h` and `4h` upside burst labels improved modestly.

Live Binance orderbook overlap was only `93` scored rows in the FreqAI check, so it is insufficient for a conclusion.

Evidence summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\live_media_short_horizon_summary_20260626.csv`.

## Live Media Tactical Shapes + VP - 2026-06-26

The added `1h`/`2h`/`4h` trader-shape layer was more useful than raw compact live-media alone for short/risk overlays. FreqAI and direct threshold checks point to:

1. Source burst plus volume expansion as a fast downside/short-risk warning.
2. Financial-stress acceleration and risk-off acceleration as useful risk-overlay states.
3. Crypto-attention high as an active-market warning/confirmation state, not automatically bullish.
4. VP helped some long follow-through labels but weakened several downside/risk labels, so VP should be conditional rather than globally enabled.

Evidence: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\formatted_freqai_hypothesis_20260626_103907_154173\live_media_threshold_confluence_sweep_20260626.csv` and `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\formatted_freqai_hypothesis\formatted_freqai_hypothesis_20260626_105806_066500\live_media_vp_comparison_summary_20260626.csv`.

## Runtime / Top-Level Contract Integration

The original repo-level runtime notes were factored back into active guidance on 2026-06-10. Key preserved constraints:

1. Use documented Python environments and worker venvs; do not guess or fallback interpreters.
2. Treat split-venv worker count as a concurrency limit, not unbounded subprocess permission.
3. Check `D:` raw archive roots for Bybit and GDELT/GKG before claiming raw coverage is missing.
4. Pause collectors and export snapshots before research tests that would otherwise read live SQLite databases.
5. Keep context/orderbook/FreqAI source updates in compact ledgers with exact artifact paths.
