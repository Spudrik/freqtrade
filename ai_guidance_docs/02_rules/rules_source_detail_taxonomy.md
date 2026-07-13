---
doc_status: active
default_read: routed
owner: user+agent
purpose: Standard source-detail group names for tests, reports, and FreqAI routing.
do_not_use_for: Replacing real column dictionaries or source readiness reports.
last_rebuilt: 2026-06-10
---


# Rules - Source Detail Taxonomy

## Purpose

Avoid vague buckets such as `context`, `structure`, or `orderbook` when the exact source block matters. Reports and tests must name the source-detail groups used.

## Standard source-detail groups

Use these names unless the codebase already has a stricter current taxonomy:

### Price / volume

1. `price_ohlcv`
2. `price_return_range_volatility`
3. `volume_pressure`
4. `range_compression_expansion`

### Structure / custom indicators

1. `structure_volume_profile`
2. `structure_tlv2_support_resistance`
3. `structure_bos_choch_market_structure`
4. `structure_pattern_geometry`
5. `structure_cached_price_volume_state`
6. `structure_composite_setups`

### Sieve / strategy families

1. `sieve_entry_family`
2. `sieve_rare_pattern_family`
3. `sieve_strategy_lane`
4. `sieve_exit_or_invalidation_family`

### Orderbook

1. `orderbook_spot`
2. `orderbook_bybit_linear`
3. `orderbook_bybit_inverse`
4. `orderbook_wall_persistence`
5. `orderbook_wall_evaporation_removal`
6. `orderbook_support_resistance_rebuild`
7. `orderbook_liquidity_vacuum`
8. `orderbook_pressure_flip`
9. `orderbook_spread_fragility`
10. `orderbook_venue_agreement_divergence`

### News/context/macro

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

## Reporting rule

Every direct test, FreqAI promotion report, strategy-lane result, and source-readiness report should list the exact source-detail groups used.

Bad:
- `context performed well`
- `orderbook improved the result`
- `structure plus confluence worked`

Good:
- `structure_volume_profile + structure_tlv2_support_resistance + orderbook_bybit_linear wall-removal features improved support-reclaim filtering`

## Column dictionary rule

When a current column dictionary exists, use it as the source of truth and map its columns to the closest standard source-detail group above. If no group fits, add a new group deliberately and document it.
