---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for live collectors, source freshness, outages, snapshots, and accumulating live research coverage.
do_not_use_for: Historical backfill management, current source-readiness authority, live-trading settings, or FreqAI promotion.
last_rebuilt: 2026-07-13
---

# Track - Live Data

## 1. Boundary

This track owns live-source collection and continuity history. It does not authorize live trading, collector changes, or source use; those remain governed by explicit user scope, source rules, and readiness status.

## 2. Current state

- Live media, web/global context, orderbook, and related collectors/snapshots exist in varying states of readiness.
- The true live-media history used in the June tests spans only a short recent window.
- Strict multi-source overlap can be much smaller than each source's individual coverage.
- Cross-pair gap-fill tests increased breadth but did not create long temporal validation.
- Freshness, outage, age, and source eligibility must be explicit on every decision row.
- Live SQLite data should be snapshotted/frozen before research when collector writes would make the test non-reproducible.

## 3. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| May-June 2026 | Live-source alignment and snapshot workflow | Added available-time/freshness concepts and reproducible parquet snapshots. | Source rules and context ledger. |
| 26 June | True live-media window tested | Several labels improved price controls; orderbook strict overlap remained small. | Recent summary. |
| 28 June | BTC combined and nine-pair gap-fill tests | Found useful cross-pair shapes, but only over a short live period; combined overlap remained sparse. | Context ledger and linked reports. |

## 4. Current conclusions

- **Promising:** accumulated live media/context shapes and selected exact orderbook states.
- **Insufficient:** long temporal history and stable all-source overlap.
- **Do not infer:** cross-pair repeatability over a short window equals temporal validation.
- **Critical rule:** source absence/staleness is an eligibility state, not a neutral feature value.

## 5. Outstanding work

- Accumulate longer uninterrupted live coverage without silently patching gaps.
- Monitor and document material collector outages or schema/timestamp changes through the governed readiness system.
- Export frozen snapshots before model/research runs that would otherwise read changing stores.
- Reassess overlap only when added history materially changes the usable decision rows.
- Keep live data collection separate from live-trading authorization and configuration.

## 6. Canonical evidence

- Source readiness: `../../03_status/status_source_readiness.md`
- News/context rules: `../../02_rules/rules_news_context_sources.md`
- Orderbook rules: `../../02_rules/rules_orderbook_sources.md`
- Runtime/snapshot rules: `../../02_rules/rules_runtime_environment.md`
- Recent summary: `../../04_results/results_recent_summary.md`
- Context ledger: `../../04_results/context_research_ledger.md`
