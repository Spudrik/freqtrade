---
doc_status: active
default_read: routed
owner: user+agent
purpose: Objective routing contract for objective_02c_orderbook_confluence.md.
do_not_use_for: Detailed implementation history.
last_rebuilt: 2026-06-10
---

# Objective 02c - Orderbook Confluence

## Purpose

Use orderbook data to confirm, contradict, target, invalidate, or adjust trades. Do not make orderbook a blunt generic filter unless focused tests prove that role.

## In scope

1. Wall persistence/removal/evaporation.
2. Support/resistance rebuild.
3. Liquidity vacuum/path-through zones.
4. Pressure flips and venue agreement/divergence.
5. Spread/fragility as risk acceleration.
6. Target-zone alignment with VP, trendlines, pivots, and pattern projections.
7. Stop tightening or partial exits when price reaches a likely resistance/support shelf and momentum fades.

## Out of scope

1. Carrying book state through missing archive gaps.
2. Treating missing orderbook as zero pressure.
3. Broad orderbook filters without setup-specific evidence.
4. Treating one venue as universal truth without venue/coverage checks.

## Required rules

- `../02_rules/rules_orderbook_sources.md`
- `../02_rules/rules_source_detail_taxonomy.md`
- `../02_rules/rules_direct_tests.md`
- `../02_rules/rules_entry_exit_position.md`
- `../02_rules/rules_result_reporting.md`

## Required status/results

- `../03_status/status_source_readiness.md`
- `../04_results/source_readiness_matrix.csv`
- `../04_results/hypothesis_ledger.csv` only when adding/checking historical orderbook tests.
