# News Formatting Objectives

This document defines the durable direction for rebuilding external news/context formatting so future agents do not drift back into weak keyword counts or early lossy aggregation.

## Purpose

Build a reproducible news/context pipeline that captures market-relevant facts, story development, source confluence, and timing quality before producing FreqAI-ready hourly features.

The goal is not to guess bullish or bearish truth from text. The goal is to encode observable facts a trader would care about and let direct tests/FreqAI decide whether those facts are predictive.

## Core Principle

Do not treat raw article counts, keyword hits, generic sentiment, or GDELT/GKG aggregate rows as the source of truth.

The durable source of truth should be normalized article/event/document records with enough metadata to rebuild topic features, story clusters, severity signals, and coverage reports when the taxonomy improves.

## Target Data Layers

1. Bronze raw layer:
   - Keep immutable raw files/API responses where feasible.
   - Store raw reference metadata: source family, URL/query, fetched time, HTTP status, byte size, hash, parser status, and raw path.
   - Do not delete raw caches until the normalized layer and coverage reports pass.

2. Silver normalized layer:
   - One canonical record format across RSS/news, web pages, GDELT events, GKG documents, official macro/global feeds, and Google Trends.
   - Store article/event/document facts before hourly aggregation.
   - Preserve parser/taxonomy/model version fields so the layer can be rebuilt and compared.

3. Gold feature layer:
   - Hourly trader-readable features derived from normalized records and story clusters.
   - Include coverage and quality flags beside signal columns.
   - Keep counts as supporting context, not the main signal family.

## Canonical Normalized Fields

Each normalized item should support these fields where applicable:

1. Source and timing:
   - `source_family`: `rss_news`, `web_page`, `gdelt_event`, `gkg_document`, `official_macro`, `google_trends`, etc.
   - `source_name`
   - `source_group`
   - `source_quality`
   - `published_at`
   - `fetched_at`
   - `available_at`
   - `raw_ref`

2. Content identity:
   - `title`
   - `summary`
   - `body_excerpt` or full body where available and permitted
   - `url`
   - `url_hash`
   - `content_hash`
   - `duplicate_group`

3. Extracted trader facts:
   - `assets_mentioned`
   - `entities_mentioned`
   - `topic`
   - `subtopic`
   - `impact_channel`
   - `severity_proxy`
   - `scope`: single company, sector, country, region, global
   - `direction`: factual state such as `stress_increase`, `stress_relief`, `supply_disruption`, `supply_restoration`, `policy_tightening`, `policy_easing`, `neutral_update`
   - `confidence`
   - `evidence_json`

4. Versioning:
   - `parser_version`
   - `taxonomy_version`
   - `entity_model_version`
   - `classifier_version`
   - `source_config_hash`

## Topic Definition Rules

Topic labels must not be assigned from single keywords alone.

1. A topic requires a combination of:
   - source context
   - recognized entities
   - event/action terms
   - impact channel
   - severity evidence
   - confidence score

2. Example banking distinction:
   - Central bank policy speech: `central_bank_policy`, not `banking_credit`.
   - Bank opens a branch: likely benign or irrelevant.
   - Deposit run, emergency liquidity, solvency concern, credit downgrade, contagion language: `banking_credit` with stress evidence.
   - Regulator says liquidity facility calmed stress: `banking_credit` plus `stress_relief` or `liquidity_support`.

3. Example oil distinction:
   - Company earnings from an oil producer: not automatically an oil shock.
   - OPEC cut, pipeline outage, tanker disruption, Hormuz risk: `oil_energy` with `supply_disruption`.
   - Inventory build or weak demand report: `oil_energy` with `demand_weakness` or `inventory_build`.
   - Production restoration or ceasefire easing shipping risk: `oil_energy` with `supply_restoration` or `stress_relief`.

## Story Clustering Objectives

Articles should be grouped into stories before feature generation.

Story clusters should use:

1. Entity overlap.
2. URL/content hashes for duplicates.
3. Title/summary/body embeddings for semantic similarity.
4. Time proximity.
5. Source group diversity.
6. Topic/subtopic compatibility.

The story layer should expose:

1. `story_id`
2. `first_seen_at`
3. `latest_seen_at`
4. `article_count`
5. `independent_source_group_count`
6. `duplicate_ratio`
7. `official_confirmation_flag`
8. `severity_start`
9. `severity_latest`
10. `severity_escalation`
11. `relief_update_flag`
12. `dominant_topic`
13. `dominant_impact_channel`

## Existing Systems To Use

Use existing systems where they reduce risk:

1. GDELT CAMEO/GKG/GCAM:
   - Use as baseline event/theme/content metadata and weak labels.
   - Do not treat GDELT aggregate rows as final normalized truth.

2. spaCy or GLiNER:
   - Use for entity extraction and entity rulers for known assets, exchanges, regulators, central banks, countries, commodities, protocols, and macro institutions.

3. Sentence Transformers:
   - Use embeddings for article/story similarity.

4. HDBSCAN or equivalent clustering:
   - Use for story clustering and uncertain/noise membership.

5. LLM structured extraction:
   - Use strict JSON schemas for nuanced event facts, impact channels, direction, severity evidence, and evidence spans.
   - Use batch processing for historical backfills when cost and throughput allow.

6. Optional commercial benchmark:
   - RavenPack-like products represent the institutional target shape: entity/event relevance, novelty, impact, topic tagging, and temporal scoring.
   - If no subscription is available, replicate only the parts needed for this project.

## Required Feature Families

Gold hourly features should include compact, trader-readable metrics such as:

1. `war_geopolitics_intensity_24h`
2. `banking_credit_confluence_24h`
3. `oil_energy_first_mention`
4. `regulation_persistence_72h`
5. `macro_release_severity_max_24h`
6. `topic_source_diversity_24h`
7. `official_confirmation_24h`
8. `duplicate_wave_ratio_24h`
9. `severity_escalation_6h`
10. `stress_relief_update_24h`
11. `first_mention_age_hours`
12. `story_follow_through_24h`

Counts such as article count, GDELT event count, or GKG document count should remain supporting context and coverage diagnostics.

## Coverage Reports Required Before Testing

Every source family must produce a coverage report before features are promoted into direct tests or FreqAI queues.

Required report fields:

1. Active date range.
2. Expected raw files/responses.
3. Successful raw files/responses.
4. Raw validation failures.
5. Normalized row count.
6. Nonzero feature rows.
7. Missing `available_at` count.
8. Future `available_at` count.
9. Duplicate rate.
10. Topic coverage.
11. Source coverage.
12. Empty title/body rate where applicable.
13. Parse error counts by type.
14. Known upstream source holes.
15. Feature rows with no valid source records.

## Acceptance Criteria For "Done"

The news/context formatting rework is not complete until:

1. Raw captures for each enabled source family have validation manifests.
2. Normalized records exist for RSS/news, web, GDELT events, GKG documents, official macro/global feeds, and Google Trends where available.
3. Every normalized record has a safe `available_at`, or is excluded with a documented reason.
4. Topic labels are based on taxonomy plus entity/action/evidence logic, not single keywords.
5. Story clustering exists and exposes first mention, persistence, source confluence, duplicate ratio, and escalation/relief state.
6. Coverage reports pass for the target historical window.
7. Gold hourly features include quality/coverage flags.
8. A small hand-reviewed audit set exists for each major topic family.
9. Direct tests compare new story/fact features against raw count baselines.
10. FreqAI promotion only happens after direct tests show stable evidence and data coverage passes.

## Current Known Gap

The current GDELT/GKG historical extraction compresses data too early into aggregate rows. It is useful for rough coverage and quick features, but it is not the final target design for high-quality trader-readable news data.

Future work should preserve per-event/per-document normalized rows before hourly aggregation.
