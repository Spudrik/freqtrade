# GDELT/GKG Historical Rework Plan

This plan narrows `news_formatting_objectives.md` to historical GDELT event exports and GKG documents only.

## Current Decision

Build the new GDELT/GKG path additively beside the existing aggregate backfill tables. Do not swap the current context feature builder to the new path until raw, normalized, timing, coverage, and feature gates pass.

## Current State

1. GDELT event aggregates:
   - Stored in `C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite`.
   - Table: `gdelt_hourly_features`.
   - Success reaches `2026-05-28T23:00:00+00:00`.
   - Remaining failures are confirmed `HTTP 404` upstream source holes.

2. GKG aggregates:
   - Stored in the same SQLite database.
   - Table: `gdelt_gkg_file_features`.
   - GKG is not complete and remains the historical critical path.
   - Active backfill loop uses 30-day chunks with 6 workers.

3. Raw files:
   - GDELT event export raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`.
   - GKG raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
   - Raw files stay on `D:`.
   - SQLite metadata, reports, and compact feature outputs stay on `C:`.

## Implementation Phases

### Phase 1: Additive Schema And Reports

1. Add Bronze raw-file metadata:
   - Table: `gdelt_raw_files`.
   - Purpose: inventory raw ZIP files, hashes, validation status, row counts, parse status, and source-hole state.

2. Add Silver normalized tables:
   - `gdelt_events_silver`: one row per raw GDELT export event row.
   - `gkg_documents_silver`: one row per raw GKG document row.

3. Add story scaffolding:
   - `story_clusters`
   - `story_members`

4. Add Gold GDELT/GKG feature table:
   - Purpose: future hourly features generated from Silver/story records, not current aggregate tables.

5. Add quality reports:
   - aggregate SQLite coverage
   - source holes
   - raw ZIP validation
   - normalized completeness
   - clean windows
   - feature promotion preflight

### Phase 2: Raw-To-Silver Parsers

1. GDELT event export parser:
   - One normalized event per raw export TSV row.
   - Use `DATEADDED` as safe `available_at`; fallback to file stamp.
   - Store `Day` as event date only, not availability.
   - Treat CAMEO codes, Goldstein, QuadClass, source URL, and tone as weak metadata.

2. GKG document parser:
   - One normalized document per raw GKG TSV row.
   - Use GKG batch/file stamp as conservative `available_at`.
   - Use GKG `DATE` or precise timestamp extras as `published_at` where valid.
   - Preserve themes, V2 themes, persons, orgs, locations, all names, counts, amounts, dates, GCAM, quotations, and extras as JSON/metadata.
   - Treat GKG-derived topic/entity fields as weak candidates until classifier/enrichment confirms them.

### Phase 3: Coverage Gate And Clean Windows

Before using GDELT/GKG in direct tests or FreqAI:

1. Raw gate passes.
2. No retryable failures remain in target windows.
3. Terminal source holes are explicit and excluded from normal clean windows.
4. Normalized records have safe `available_at`.
5. Feature rows have no source timestamps after feature hour.
6. GDELT/GKG features include source-hole and completeness flags.
7. Direct tests use only clean windows or explicitly labelled source-hole experiments.

### Phase 4: Story And Enrichment

1. Add source quality mapping.
2. Add entity resolution for assets, countries, regulators, exchanges, commodities, protocols, and central banks.
3. Add story clustering from URLs, entities, themes, source diversity, time proximity, and embeddings where available.
4. Add classifier-backed `topic`, `subtopic`, `impact_channel`, `direction`, `severity_proxy`, `confidence`, and evidence JSON.

## Non-Negotiable Rules

1. Do not treat existing aggregate GDELT/GKG rows as recoverable Silver truth.
2. Do not call aggregate-only GDELT/GKG features final trader-readable news features.
3. Do not zero-fill source holes into quiet/no-news states.
4. Do not assign final topic labels from single keywords.
5. Do not use `downloaded_at` as market availability.
6. Do not interrupt the active GKG backfill unless it is clearly dead or harming the machine.

## Active Worker Wave

1. Schema worker:
   - Owns normalized schema helpers and schema tests.

2. GDELT parser worker:
   - Owns raw export row normalization and tests.

3. GKG parser worker:
   - Owns raw GKG document normalization and tests.

4. Quality report worker:
   - Owns GDELT/GKG quality report tooling and tests.

## First Acceptance Target

The first acceptable implementation milestone is not a model result. It is:

1. Additive schema initializes cleanly in a temp SQLite database.
2. Raw-file metadata can be inventoried idempotently.
3. Synthetic GDELT export rows normalize correctly.
4. Synthetic GKG rows normalize correctly.
5. Quality reports can be generated from temp fixtures and from the current aggregate DB without altering existing data.
6. Existing GKG backfill continues running independently.
