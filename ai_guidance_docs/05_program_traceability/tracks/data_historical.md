---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for historical source acquisition, backfills, extraction, continuity, and reproducible research blocks.
do_not_use_for: Current source-readiness authority, live collector operation, raw file inspection, or FreqAI conclusions.
last_rebuilt: 2026-07-13
---

# Track - Historical Data

## 1. Boundary

This track owns the history of historical OHLCV, orderbook, news, GDELT/GKG, extraction, alignment, and gap-management work. Current usability remains governed by `../../03_status/status_source_readiness.md` and the source-specific rules.

## 2. Current state

- OHLCV and derived structural/indicator state are the most usable base, subject to ordinary coverage checks.
- Bybit historical orderbook coverage is partial and should be used only on validated windows.
- Historical aggregate GDELT/GKG has some usable blocks, but continuous clean coverage and taxonomy remain incomplete.
- Historical article/story silver layers are not ready; missing rows cannot be interpreted as no news.
- Large raw acquisition, missing intervals, extraction cost, and disk capacity repeatedly constrained progress.
- Reproducible tests should use frozen, gap-aware snapshots rather than changing live SQLite stores.

## 3. Storage anchors

- Bybit raw ZIPs: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`
- GDELT/GKG raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw`

An absent or empty mirrored `C:\FreqTradeStuff\user_data\...\raw` directory is not proof that raw coverage is missing.

## 4. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| May 2026 | Bybit import/validation/compaction | Created partial historical orderbook research windows; broad continuity was not achieved. | Source-readiness status and context ledger. |
| Late May | Source age, masks, gaps, and snapshots added | Established that missing/stale source rows must remain explicit. | Data/model audit and source rules. |
| 29 May | False-clean audit | Found zero-fill, timestamp-mask, taxonomy, and eligibility defects that overstated usable history. | Review docs linked from the context ledger. |
| 31 May-2 June | Large GKG acquisition/extraction prototypes | Compression was effective, but acquisition remained partial and taxonomy false positives required more work. | Historical progress log and source status. |
| June | Aggregate historical blocks tested | Produced exploratory context leads, not article/story promotion evidence. | Recent summary and context ledger. |

## 5. Current conclusions

- **Usable with checks:** OHLCV and derived deterministic features.
- **Partial:** Bybit historical orderbook and aggregate GDELT/GKG windows.
- **Blocked/parked:** continuous article/story historical news and full multi-source overlap.
- **Critical rule:** preserve `available_at`, source age, coverage, and gaps; never silently zero-fill absence.

## 6. Outstanding work

- Maintain a compact coverage map before adding more feature breadth.
- Complete only backfills that create a named usable research block or resolve a specific gap.
- Use approved scripts for raw archives, extraction, coverage, and snapshots; do not manually load bulky artifacts into agent context.
- Validate taxonomy and timestamp semantics before treating a new block as ready.
- Record a milestone only when a materially larger clean block becomes usable, a source is decisively parked, or storage architecture changes.

## 7. Canonical evidence

- Source readiness: `../../03_status/status_source_readiness.md`
- Runtime/storage rules: `../../02_rules/rules_runtime_environment.md`
- News/context rules: `../../02_rules/rules_news_context_sources.md`
- Orderbook rules: `../../02_rules/rules_orderbook_sources.md`
- Source matrix: `../../04_results/source_readiness_matrix.csv`
- Context ledger: `../../04_results/context_research_ledger.md`

