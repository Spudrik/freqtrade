---
doc_status: active
default_read: routed
owner: user+agent
purpose: News/GDELT/GKG/web/global source readiness, timestamp safety, and formatting rules.
do_not_use_for: Orderbook readiness.
last_rebuilt: 2026-09-09
---


# Rules - News/Context Sources

## Current status

Parked as broad or default inputs. Current Objective 02b explicitly reopens bounded
use of news, GDELT/GKG, web, global, macro, trends, ETF-flow, and similar context for
causal event reconstruction and the event-driven direction hierarchy, but only inside
source blocks whose real coverage, availability timing, and missingness are proved.
This exception does not make the sources globally ready.

## Use only when

1. The user says the data is ready, or
2. a source-readiness report proves a specific source block/window is usable.

For current Objective 02b, the user's 12 August authorization permits exploratory use
of these sources and the 3 September authorization permits their bounded use in the
event-driven direction hierarchy. The readiness report still controls which exact
rows and claims are valid.

## Important nuance

News/context is not globally unusable. Some blocks or months may be usable if proven with coverage, timestamp, and completeness reports. Use only those proven windows/source-detail blocks.

## Objective 02b event-reconstruction discipline

When these sources are used to reconstruct a historical event or test direction:

1. define why the event would have mattered using information available before the
   market outcome; do not select importance because price later moved sharply;
2. separate scheduled events, whose timing was knowable, from unexpected events,
   whose occurrence was not predictable in advance;
3. separate the prior expectation, the released outcome, and the surprise relative
   to that expectation where the evidence permits it;
4. timestamp every fact by when a trader could first have received it, not merely by
   the date the event concerned;
5. keep all observations from one event in the same development or holdout partition;
6. preserve source-present, source-missing, stale, and low-coverage states rather than
   interpreting missing material as quiet or neutral news; and
7. call a relationship causal only when source timing and coverage support that claim;
8. distinguish the news driver from the market's later volume, pressure, volatility,
   or price confirmation rather than treating the response as a competing cause; and
9. evaluate whether the event amplified, opposed, reduced, delayed, or was overwhelmed
   by the pre-existing market background instead of assigning one permanent bullish or
   bearish meaning to the event family.

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

## Story Families, Narrative Accumulation, And Conditional Meaning

News features must represent both individual major events and the rolling balance of
several distinct smaller developments. Do not force every story to predict a price
move by itself. A story may add pressure that only becomes visible when other stories,
market conditions, and technical structure align.

For each story or event family, preserve where source quality permits:

1. first safe market-availability time;
2. scheduled versus unexpected status;
3. prior expectation and surprise relative to that expectation;
4. novelty versus repetition of already-known information;
5. credibility and confirmation stage;
6. severity and likely duration;
7. geographic, financial-system, crypto-market, sector, and coin scope;
8. escalation, relief, correction, reversal, or implementation state;
9. the dominant market concern the story addresses; and
10. a causal decay rule for how long its pressure may remain active.

Separate these cases:

1. many sources repeating the same underlying story - one event with wider attention,
   not many independent positive or negative events;
2. several genuinely different stories pointing in the same direction - possible
   accumulated narrative pressure;
3. different stories applying opposing pressure - a conflicted background rather than
   an automatic neutral state; and
4. one exceptional event plausibly overriding the accumulated smaller background.

Do not judge a story only by the final market sign. Positive information can reduce an
otherwise likely fall, cap negative pressure, cause only an initial rise, or be
subdued by a dominant negative background, positioning, liquidity, or technical
obstacle. Negative information can analogously weaken an expected rise without
turning the final candle negative. Where feasible, report raw movement and movement
relative to matched pre-event states.

The same release can carry different market meaning at different times. For example,
lower inflation may help when inflation and rates dominate attention but may be read
as weak growth when recession fear dominates. Determine the contemporaneous concern
from timestamp-safe source and market information; never assign it from hindsight.

Narrative interaction hypotheses may include amplification, suppression, reversal,
delay, shortening, accumulation, transmission blocking, and dominant override. A
weak standalone story can remain a rational modifier or accumulator. Discover such
conditions only in development data, freeze the definition, and require later whole-
event or prospective confirmation.

Reports and models must keep story identity, source breadth, duplicate count, distinct
story-family count, rolling positive/negative/conflicting pressure, source readiness,
and market-response confirmation separately identifiable. Feature importance or raw
article count cannot establish which story caused a market move.

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
10. whether inactive rows mean true quietness or unavailable/no-signal data,
11. unique underlying stories versus duplicate-source coverage,
12. distinct concurrent story families and their positive/negative/conflicting
    balance,
13. expectation, surprise, novelty, severity, credibility, and confirmation state
    where reconstructable,
14. the contemporaneous dominant concern and whether it was predeclared,
    development-inferred, or later confirmed, and
15. raw market outcome versus change from a matched expected path.

## Still out of scope

1. News/context confluence claims outside a predeclared causal event design and proven
   source block.
2. Broad FreqAI over incomplete context data.
3. Treating missing GKG/news rows as zero/quiet.
4. Using old aggregate rows as final trader-readable features.
