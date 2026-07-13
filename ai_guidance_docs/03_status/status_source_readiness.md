---
doc_status: active
default_read: routed
owner: agent
purpose: Current source readiness summary.
do_not_use_for: Detailed source implementation plans.
last_rebuilt: 2026-06-10
---

# Status - Source Readiness

## Current summary

| Source area | Status | Use now? | Notes |
|---|---|---:|---|
| Price/OHLCV | usable | yes | Still validate timerange/gaps per task. |
| Custom indicators / sieve-derived entries | usable | yes | Core evidence source. Do not replace with generic TA. |
| Structure cache / VP / TLV2 / BOS/CHoCH / patterns | usable but verify freshness | yes with checks | Important for entry logic and target zones. |
| Orderbook historical Bybit | partial | yes for validated windows | Use only with present/coverage/timestamp/gap rules. |
| News/GDELT/GKG/web/global/context | parked/incomplete | no by default | Some windows/blocks may be usable if readiness report proves them. |
| Generic TA discovery | side lane | yes as side research | Track for later merge; do not displace custom indicators. |

See `../04_results/source_readiness_matrix.csv` for machine-readable status.

## News/context audited exceptions - 2026-06-15

News/GDELT/GKG remains parked by default, but these source-masked windows are now acceptable for first-pass aggregate overlay research:

1. `2020-01-01` to `2020-10-31`: GDELT events plus GKG aggregate topics, with October partial.
2. `2021-03-01` to `2022-07-31`: best first window for GDELT events plus GKG aggregate topics.
3. `2022-08-01` to `2022-12-31`: GDELT events only; do not use GKG topic/document aggregates as present.
4. `2023-01-18 07:00 UTC` to `2023-06-30`: GDELT events plus Bybit orderbook; GKG aggregates are not usable here.

Historical article/story-level news remains blocked because `gdelt_events_silver`, `gkg_documents_silver`, and `story_clusters` currently have zero rows, and the `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg` folder is empty on disk.

## Source-detail caution

Do not treat source-family names as enough. A usable window must name the exact source-detail groups and prove timestamp-safe coverage for each required group.


## Raw archive location caution

1. Bybit raw archives are on `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`, split by `spot_raw`, `linear_raw`, and `inverse_raw`.
2. GDELT/GKG raw archives are on `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`.
3. Do not mark coverage incomplete just because a mirrored `C:\FreqTradeStuff\user_data\...\raw` folder is absent or empty.
