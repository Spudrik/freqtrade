---
doc_status: parked
default_read: routed
owner: user+agent
purpose: Objective routing contract for news/context integration.
do_not_use_for: Default active trading research while parked.
last_rebuilt: 2026-06-10
---


# Objective 02d - News/Context Integration

## Purpose

Keep news/GDELT/GKG/web/global/context as a long-term integration goal, while preventing agents from treating incomplete data as ready.

## Current status

Parked. Resume only when the user says a dataset is ready or a source-readiness report proves a specific source block/window is usable.

## In scope when resumed

1. Source coverage and timestamp safety.
2. Raw/bronze, normalized/silver, and feature/gold separation where applicable.
3. Story/topic/severity/source-confluence features.
4. Crash/risk-off/regime signals that can influence entries, exits, stop tightening, exposure, or leverage.
5. Specific windows or source blocks if proven usable.
6. Comparing whether news/context changes the probability of hitting target before stop, or warns that an open trade should tighten/reduce.

## Future news/context architecture requirements

When resuming news/context implementation, preserve these layers:

1. Bronze raw layer:
   - immutable raw files/API responses where feasible,
   - source family, URL/query, fetched time, HTTP status, byte size, content hash, parser status, and raw path,
   - raw caches retained until normalized layer and coverage reports pass.
2. Silver normalized layer:
   - one canonical record format across RSS/news, web pages, GDELT events, GKG documents, official macro/global feeds, and Google Trends,
   - article/event/document facts before hourly aggregation,
   - parser/taxonomy/model/source-config version fields so features can be rebuilt and compared.
3. Gold feature layer:
   - hourly trader-readable features derived from normalized records and story clusters,
   - coverage and quality flags beside signal columns,
   - counts are supporting context, not the main signal family.

## Canonical normalized fields to preserve

Use these where applicable:

1. Source and timing: `source_family`, `source_name`, `source_group`, `source_quality`, `published_at`, `fetched_at`, `available_at`, `raw_ref`.
2. Content identity: `title`, `summary`, `body_excerpt` or permitted full body, `url`, `url_hash`, `content_hash`, `duplicate_group`.
3. Trader facts: `assets_mentioned`, `entities_mentioned`, `topic`, `subtopic`, `impact_channel`, `severity_proxy`, `scope`, `direction`, `confidence`, `evidence_json`.
4. Versioning: `parser_version`, `taxonomy_version`, `entity_model_version`, `classifier_version`, `source_config_hash`.

## Story clustering requirements

Articles/documents should be grouped into stories before feature generation where possible. Story features should include:

1. `story_id`, `first_seen_at`, `latest_seen_at`, `article_count`.
2. independent source-group count and duplicate ratio.
3. official confirmation flag.
4. severity start/latest/escalation.
5. relief/update flag.
6. dominant topic and impact channel.

## GDELT/GKG-specific rules

1. Build any new GDELT/GKG path additively beside old aggregate backfills.
2. Do not treat existing aggregate GDELT/GKG rows as recoverable Silver truth.
3. Do not call aggregate-only GDELT/GKG features final trader-readable news features.
4. Do not zero-fill source holes into quiet/no-news states.
5. Do not assign final topic labels from single keywords.
6. Do not use `downloaded_at` as market availability.
7. Use GDELT event `DATEADDED` as safe `available_at` where applicable.
8. Use GKG batch/file stamp as conservative `available_at` where applicable.

## Out of scope while parked

1. Full confluence claims.
2. Treating GDELT/GKG aggregate rows as final trader-readable features.
3. Treating missing data as quiet/no-news.
4. Topic labels from single keywords.
5. Broad FreqAI runs over incomplete context features.

## Required rules if resumed

- `../02_rules/rules_news_context_sources.md`
- `../02_rules/rules_source_detail_taxonomy.md`
- `../02_rules/rules_freqai_feature_discovery.md` for feature exploration.
- `../02_rules/rules_direct_tests.md` for hypotheses.
- `../02_rules/rules_freqai_promotion.md` for FreqAI promotion.

## Required archive docs if implementing formatting/rebuild work

Open these archived plans before editing news/context pipeline code:

- `../99_archive/original_uploaded_docs/parked_news_context/news_formatting_objectives.md`
- `../99_archive/original_uploaded_docs/parked_news_context/gdelt_gkg_historical_rework_plan.md`
