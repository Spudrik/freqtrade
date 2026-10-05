---
doc_status: active
default_read: routed
owner: agent
purpose: Current source readiness summary.
do_not_use_for: Detailed source implementation plans.
last_rebuilt: 2026-09-04
---

# Status - Source Readiness

## Current summary

| Source area | Status | Use now? | Notes |
|---|---|---:|---|
| Price/OHLCV | usable | yes | Still validate timerange/gaps per task. |
| Custom indicators / sieve-derived entries | usable | yes | Core evidence source. Do not replace with generic TA. |
| Structure cache / VP / TLV2 / BOS/CHoCH / patterns | usable but verify freshness | yes with checks | Important for entry logic and target zones. |
| Orderbook historical/live | partial; recent live coverage passed | yes for validated windows | All 40 recent streams passed the bounded coverage preflight, but only one frozen scheduled event and one meme coin overlap. Use for later reaction confirmation; do not infer broad event or meme evidence. |
| News/GDELT/GKG/web/global/context | collection active; semantic and timestamp preparation incomplete | validated windows and bounded Objective 02b pilots only | News/web lacks usable signed semantics. Global data retains collection times but needs a causal extraction because frequent polls, early source timestamps, and revisions make raw rows unsafe for direct modelling. |
| Generic TA discovery | side lane | yes as side research | Track for later merge; do not displace custom indicators. |

See `../04_results/source_readiness_matrix.csv` for machine-readable status.

## News/context audited exceptions - 2026-06-15

News/GDELT/GKG remains parked by default, but these source-masked windows are now acceptable for first-pass aggregate overlay research:

1. `2020-01-01` to `2020-10-31`: GDELT events plus GKG aggregate topics, with October partial.
2. `2021-03-01` to `2022-07-31`: best first window for GDELT events plus GKG aggregate topics.
3. `2022-08-01` to `2022-12-31`: GDELT events only; do not use GKG topic/document aggregates as present.
4. `2023-01-18 07:00 UTC` to `2023-06-30`: GDELT events plus Bybit orderbook; GKG aggregates are not usable here.

Historical article/story-level news remains blocked because `gdelt_events_silver`, `gkg_documents_silver`, and `story_clusters` currently have zero rows, and the `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg` folder is empty on disk.

## Live collection audit - 2026-09-04

These are point-in-time counts from the audit and will continue to increase:

1. News: `77,274` articles, `34` enabled sources, and a fresh 15-minute collector. Two
   ECB feeds repeatedly fail certificate verification and have never succeeded; other
   audited feeds continue to collect.
2. Web: `4,238` items from `9` enabled sources, with a fresh six-hour collector and no
   current source error in the inspected cycle.
3. Global context: `56,388` ticks at the bounded preflight cutoff, with `9` of `15` configured sources enabled. Crypto
   attention, ETF flows, fear/greed, global crypto, stablecoin/chain, equity, and rate
   groups are live. Several macro/stress/systemic-risk Trends and Stooq groups are
   disabled, stopped, or historically short and require isolated source tests.
4. Live orderbook: `451,196,711` messages and `34,118,160` metric rows across the
   top-ten configured coins and `40` active streams at audit time. The database is on
   `D:` through the approved junction. A recent ping timeout was transient because
   fresh messages and metrics continued afterwards.

News and web enrichment tables already contain relevance, category, tag, urgency, and
duplicate information. However, every audited news/web row had
`impact_direction='unknown'`, with `severity` and `confidence` unset; both
`article_forward_returns` tables were empty. The sources may therefore support
availability, activity, topic, and duplicate-aware baselines, but not signed or
severity-aware historical claims yet.

Objective 02b Section 7.7 now permits a maximum-`300`-story outcome-blind semantic
pilot. It must use immutable snapshots and pass timestamp, traceability, schema,
duplicate, and review-agreement gates before any market-outcome comparison. Objective
02d remains parked outside that exception.

## Global-context causal preflight - 2026-09-04

The read-only preflight inspected `56,388` rows across `43` source/metric histories
without reading prices or outcomes. All rows retain `created_at`, so none is
irrecoverably blocked, but only one history is directly timestamp-safe as stored. The
other `42` require a frozen causal extraction using collection time; `10` also contain
later value revisions at the same source timestamp.

The enabled Google Trends crypto group is suitable for a small first test after this
repair: each of its six metrics has `182` distinct observations at roughly twelve-hour
collection cadence and no duplicate source timestamps. Its stored timestamp precedes
actual collection by about `20.6` minutes at the median and as much as `62.9` minutes,
so tests must use `created_at`, not `ts`.

ETF flow, fear/greed, FRED equity/rate, and similar daily histories were polled roughly
every thirty minutes. Their database row counts therefore greatly overstate independent
information. Freeze first-seen and changed vintages using `created_at`; preserve real
revisions and remove unchanged repeat pulls. CoinGecko snapshots are much closer to
independent observations. The current DeFi-chain change metric was unchanged throughout
the inspected history and should be parked unless new variation appears.

Evidence: `user_data/research_news_data/context_features/source_preflight/`
`global_context_source_preflight_20260904a.json`.

## Recent orderbook overlap preflight - 2026-09-04

The read-only check covered `2026-08-23 20:00 UTC` through `2026-09-04 19:00
UTC`. All `40` one-minute streams passed the declared minimums across ten coins and
Binance spot/futures plus Bybit spot/futures: at least `95%` clock coverage, at least
`80%` valid samples, no duplicate minute keys, and no bar recorded before it ended.

This makes the source suitable for later confirmation around causally frozen reaction
contacts. It does not yet support a major-event claim because only one frozen scheduled
event falls inside the complete live window. It also cannot represent the top-ten meme
cohort because DOGE is the only collected meme coin. The collector was not paused or
changed for this outcome-blind audit; an actual outcome test must first freeze only the
required windows under the runtime snapshot rules.

Evidence: `user_data/research_news_data/context_features/source_preflight/`
`orderbook_event_coverage_preflight_20260904a.json`.

## Source-detail caution

Do not treat source-family names as enough. A usable window must name the exact source-detail groups and prove timestamp-safe coverage for each required group.


## Raw archive location caution

1. Bybit raw archives are on `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`, split by `spot_raw`, `linear_raw`, and `inverse_raw`.
2. GDELT/GKG raw archives are on `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`.
3. Do not mark coverage incomplete just because a mirrored `C:\FreqTradeStuff\user_data\...\raw` folder is absent or empty.
