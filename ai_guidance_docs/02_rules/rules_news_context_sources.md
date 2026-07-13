---
doc_status: active
default_read: routed
owner: user+agent
purpose: News/GDELT/GKG/web/global source readiness, timestamp safety, and formatting rules.
do_not_use_for: Orderbook readiness.
last_rebuilt: 2026-06-10
---


# Rules - News/Context Sources

## Current status

Parked. News, GDELT, GKG, web, global, macro, trends, ETF-flow, and similar context sources are long-term objectives but not active default inputs.

## Use only when

1. The user says the data is ready, or
2. a source-readiness report proves a specific source block/window is usable.

## Important nuance

News/context is not globally unusable. Some blocks or months may be usable if proven with coverage, timestamp, and completeness reports. Use only those proven windows/source-detail blocks.

## User terminology for source families

When the user says `live`, `media`, `meta`, or `news data` in this project, treat that as the live media block unless they say otherwise:

1. `context_live_news_web`
2. `context_global_market_macro`
3. web/news downloader outputs

The user treats those live media sources as one practical block for research planning. Do not split them into separate user-facing objectives unless the test specifically needs source-family ablation.

GDELT/GKG is different. Treat GDELT/GKG as historical/archive context data for long backtests and separate archive research. Do not mix GDELT/GKG into a `live media/news` result unless the user explicitly asks for archive GDELT/GKG or a combined live-plus-archive comparison.

If the user asks for `live plus orderbook`, clarify whether they mean live Binance streamed orderbook or historical Bybit archive orderbook. If the user says `live orderbook` or `Binance orderbook`, use streamed Binance orderbook data. If the user says `historic orderbook`, `historical orderbook`, `Bybit`, or `archive orderbook`, use the Bybit archive path/rules.


## Raw archive and database locations

Historical GDELT/GKG raw ZIP archives are intentionally stored on `D:`, not under `C:\FreqTradeStuff\user_data`:

1. Root: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`
2. GKG raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`
3. GDELT event export raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`
4. SQLite database: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite`

Do not infer incomplete GDELT/GKG raw coverage from `C:\FreqTradeStuff\user_data\research_news_data\gdelt\raw` being absent or empty.

## Required source-detail naming

Name exact blocks rather than `context`:

1. `context_gdelt_events`
2. `context_gkg_documents`
3. `context_article_source_activity`
4. `context_live_news_web`
5. `context_google_trends`
6. `context_btc_etf_flows`
7. `context_global_market_macro`
8. `context_topic_severity`
9. `context_story_cluster`
10. `context_availability_metadata`

## Anti-lookahead readiness masks

Context availability must be enforced in actual masks, not only report text.

Minimum mask logic:

1. Source-specific present/completeness flag is true.
2. `context_source_future_violation == false` where available.
3. If `max_source_available_at` exists: `max_source_available_at <= feature_hour`.
4. If source age/freshness exists: source age must be within the selected max-age rule.
5. Missing source rows remain missing; they are not converted into quiet/no-news states unless the test is explicitly a missing-data/quietness experiment with proven source availability.

A readable equivalent is:

`usable_context_row = source_present & ~context_source_future_violation & (max_source_available_at <= feature_hour when available) & within_max_age`

## Required readiness checks

1. Source-specific present/completeness flags.
2. Conservative `available_at <= feature_hour` timing.
3. Raw/source coverage and parser status known for the tested window.
4. Terminal source holes are explicit and excluded from normal clean windows.
5. Topic/severity/entity features are not based on single keywords alone.
6. GDELT/GKG aggregates are not treated as final normalized truth.
7. Historical GDELT coverage is not treated as equivalent to live news/web/global/Trends/ETF coverage.

## News/context feature architecture

When implementing or rebuilding news/context features, preserve:

1. Bronze raw/source layer:
   - raw file/API metadata,
   - fetched time, source URL/query, status, raw path, byte size, content hash, parser status.
2. Silver normalized layer:
   - one canonical article/event/document record shape,
   - safe `available_at`, `published_at`, `fetched_at`, raw reference,
   - topic/entity/severity/version fields.
3. Gold hourly feature layer:
   - trader-readable story/topic/severity/source-confluence features,
   - coverage and quality flags beside signal columns,
   - counts as context, not the main signal family.

## Desired future feature qualities

1. First mention.
2. Topic escalation.
3. Severity proxy.
4. Source confluence.
5. Persistence.
6. Impact channel.
7. Entity/source diversity.
8. Relief versus escalation updates.
9. Market availability timing.
10. Story clustering and duplicate control.

## GDELT/GKG-specific rules

1. Build any new GDELT/GKG path additively beside old aggregate backfill tables.
2. Do not treat existing aggregate GDELT/GKG rows as recoverable Silver truth.
3. Do not call aggregate-only GDELT/GKG features final trader-readable news features.
4. Do not zero-fill source holes into quiet/no-news states.
5. Do not assign final topic labels from single keywords.
6. Do not use `downloaded_at` as market availability.
7. Use GDELT event `DATEADDED` as safe `available_at` where applicable.
8. Use GKG batch/file stamp as conservative `available_at` where applicable.

## Required reporting columns

Reports should include, where available:

1. present rows,
2. usable rows,
3. active/nonzero rows,
4. missing rows,
5. low-coverage or incomplete rows excluded,
6. future-timestamp rows excluded,
7. late-available-at rows excluded,
8. max source age,
9. source-detail groups required,
10. whether inactive rows mean true quietness or unavailable/no-signal data.

## Out-of-scope while parked

1. Full news/context confluence claims.
2. Broad FreqAI over incomplete context data.
3. Treating missing GKG/news rows as zero/quiet.
4. Using old aggregate rows as final trader-readable features.
