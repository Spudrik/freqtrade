---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for core custom indicator development and material redesigns.
do_not_use_for: Indicator API contracts, Sieve acceptance, per-commit history, or routine indicator implementation context.
last_rebuilt: 2026-09-03
---

# Track - Core Indicators

## 1. Boundary

This track owns the development history of reusable non-pattern indicator foundations. Indicator contracts remain in `user_data/Indicators`; Sieve evidence belongs to the Sieve track.

## 2. Current state

- The core vocabulary includes canonical pivots/prior/equal levels, BOS/CHoCH, TLV2 support/resistance, volume-profile locations, zones/liquidity, and selected market-state side lanes.
- These indicators provide triggers, context, targets, invalidations, or guards; they are not one combined entry signal.
- TLV2, VP, BOS/CHoCH, and prior levels have produced credible Sieve foundations when used with structurally meaningful execution.
- TLV2 now builds chronological proximity families: two-pivot seeds are optional developing evidence, while canonical ranked lines require three to six confirmed pivots and start only when knowable.
- TLV2 candidate work is bounded by recent-pivot, active-seed, and active-family limits. Its reusable state is shared with Geometry V2 and the structural feature cache to avoid duplicate construction.
- Raw score or flip outputs can overtrade badly when promoted directly to entries.
- Isolated causality, contract, runtime, and plot checks passed; the ranked output remains relatively short-lived and should be treated as structural context until strategy-level evidence says otherwise.
- Indicator documentation and current file placement may have some drift, particularly around relative strength; inspect current code before future work.
- Current Objective 02b uses these causal levels as the initial unchanged market-reaction surface. It first tests whether locations add reaction information, then may test supported levels and clusters as coin-local context inside the bounded event-driven direction hierarchy; neither result is permission to turn them into entries, exits, or trade actions.
- Any later indicator refinement must begin from a clean tracked baseline and a dedicated experiment branch; canonical indicators are not silently overwritten by research variants.

## 3. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| 29 April-1 May 2026 | Pivot/VP/trendline review and TLV2 development | Consolidated structural anchors and produced strategy-ready trendline support/resistance outputs. | Git history; indicator reference. |
| 6-7 May | Foundation consolidation and lifecycle helpers | Reduced competing foundations and made outputs reusable by generated strategies. | Git history; indicator code. |
| 8-14 May | Runtime, filtering, naming, and threshold refinements | Removed costly geometry sampling, improved filters/scaling, and standardized strategy-facing outputs. | Git history; indicator guides. |
| Mid-May onward | Sieve integration | Indicator usefulness became evidence-dependent by exact trigger/context role rather than module existence. | Sieve results and baseline lookup. |
| 21 July 2026 | TLV2 causal proximity-family redesign | Replaced all-candidate joining/absorption with a bounded chronological state machine, separated optional two-pivot hypotheses from confirmed three-plus-pivot ranks, and reused one state across TLV2/geometry consumers. Prefix invariance, pivot timing, bounded runtime, and multi-market plots were checked in isolation; strategies were not executed. | Current indicator code and detailed redesign commit. |
| 12 August 2026 | Market-first reaction-zone objective activated | Existing indicator levels will be tested unchanged against causal reaction and density controls before rational configuration or branch-protected code refinement. | `../../01_objectives/objective_02b_market_reaction_zone_discovery.md`. |
| 3 September 2026 | Coin-local role extended within the event hierarchy | Supported levels and clusters may be tested as bounded local modifiers of event-led direction after their standalone reaction value is measured; trade-action meaning remains out of scope. | `../../01_objectives/objective_02b_market_reaction_zone_discovery.md`. |

## 4. Current conclusions

- **Demonstrated:** structural levels and location-aware context are reusable foundations.
- **Promising:** source-specific target/invalidation providers derived from TLV2, VP, pivots, and structure.
- **Needs refinement:** raw indicator scores used as direct entries; short-lived TLV2 rank segments; reference/code placement drift.
- **Do not infer:** an indicator is successful merely because many generated strategies import it.

## 5. Outstanding work

- Audit causal availability, output meaning, level density, and timestamp behaviour for the initial Objective 02b surface.
- Test supported levels and clusters alone before adding them as coin-local context to event/leader/group chains, and preserve whole-event holdouts when direction is scored.
- Test current defaults before changing construction, and distinguish an uninformative level family from a failed representation.
- Keep outputs vectorized and reuse canonical foundations rather than creating duplicate parsers/calculators.
- Record a new traceability milestone only after a material redesign, completed indicator contract, replacement, or retirement.
- Reconcile relative-strength documentation and file placement when that lane is explicitly reopened.
- Validate any variant with causal, contract, runtime, and untouched-data reaction evidence before proposing canonical adoption.

## 6. Canonical evidence

- Indicator agent/router: `../../../user_data/Indicators/agent.md`
- Strategy reference: `../../../user_data/Indicators/INDICATOR_STRATEGY_REFERENCE.md`
- Score guide: `../../../user_data/Indicators/INDICATOR_SCORE_GUIDE.md`
- Current indicator code: `../../../user_data/Indicators/`
- Sieve evidence: `sieve_entry_exit.md`
