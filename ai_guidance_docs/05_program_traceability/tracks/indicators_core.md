---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for core custom indicator development and material redesigns.
do_not_use_for: Indicator API contracts, Sieve acceptance, per-commit history, or routine indicator implementation context.
last_rebuilt: 2026-07-13
---

# Track - Core Indicators

## 1. Boundary

This track owns the development history of reusable non-pattern indicator foundations. Indicator contracts remain in `user_data/Indicators`; Sieve evidence belongs to the Sieve track.

## 2. Current state

- The core vocabulary includes canonical pivots/prior/equal levels, BOS/CHoCH, TLV2 support/resistance, volume-profile locations, zones/liquidity, and selected market-state side lanes.
- These indicators provide triggers, context, targets, invalidations, or guards; they are not one combined entry signal.
- TLV2, VP, BOS/CHoCH, and prior levels have produced credible Sieve foundations when used with structurally meaningful execution.
- Raw score or flip outputs can overtrade badly when promoted directly to entries.
- Indicator documentation and current file placement may have some drift, particularly around relative strength; inspect current code before future work.

## 3. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| 29 April-1 May 2026 | Pivot/VP/trendline review and TLV2 development | Consolidated structural anchors and produced strategy-ready trendline support/resistance outputs. | Git history; indicator reference. |
| 6-7 May | Foundation consolidation and lifecycle helpers | Reduced competing foundations and made outputs reusable by generated strategies. | Git history; indicator code. |
| 8-14 May | Runtime, filtering, naming, and threshold refinements | Removed costly geometry sampling, improved filters/scaling, and standardized strategy-facing outputs. | Git history; indicator guides. |
| Mid-May onward | Sieve integration | Indicator usefulness became evidence-dependent by exact trigger/context role rather than module existence. | Sieve results and baseline lookup. |

## 4. Current conclusions

- **Demonstrated:** structural levels and location-aware context are reusable foundations.
- **Promising:** source-specific target/invalidation providers derived from TLV2, VP, pivots, and structure.
- **Needs refinement:** raw indicator scores used as direct entries; reference/code placement drift.
- **Do not infer:** an indicator is successful merely because many generated strategies import it.

## 5. Outstanding work

- Audit a core indicator only when a routed task identifies a concrete implementation or contract problem.
- Keep outputs vectorized and reuse canonical foundations rather than creating duplicate parsers/calculators.
- Record a new traceability milestone only after a material redesign, completed indicator contract, replacement, or retirement.
- Reconcile relative-strength documentation and file placement when that lane is explicitly reopened.

## 6. Canonical evidence

- Indicator agent/router: `../../../user_data/Indicators/agent.md`
- Strategy reference: `../../../user_data/Indicators/INDICATOR_STRATEGY_REFERENCE.md`
- Score guide: `../../../user_data/Indicators/INDICATOR_SCORE_GUIDE.md`
- Current indicator code: `../../../user_data/Indicators/`
- Sieve evidence: `sieve_entry_exit.md`

