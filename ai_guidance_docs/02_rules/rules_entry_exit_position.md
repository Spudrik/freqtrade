---
doc_status: active
default_read: routed
owner: user
purpose: Rules for nuanced entry/exit/add/reduce/risk design.
do_not_use_for: Broad feature discovery or data ingestion.
last_rebuilt: 2026-06-10
---


# Rules - Entry, Exit, And Position Management

## Core architecture

Each entry should have:

1. an entry family/reason,
2. expected direction,
3. expected path if available,
4. target zones if available,
5. invalidation conditions,
6. confluence/conflict monitoring after entry.

## Baseline interpretation

Many sieve-derived entry-only strategies were originally screened with simple `3%` take-profit and `3%` stop-loss style logic. The master strategy should improve on that by asking:

1. Is this entry more likely to hit target before stop?
2. Where is the expected target or obstacle if the entry works?
3. What would invalidate the entry before the fixed stop is hit?
4. What later evidence justifies add, hold, reduce, stop tightening, or exit?

The `3%/3%` idea is a historical baseline concept, not a universal final exit rule.

## Trade-state model

When implementing position management, prefer explicit trade state instead of hidden generic exits. Each open trade should retain or reconstruct:

1. `entry_family`: exact rule/family that opened the trade.
2. `entry_direction`: long/short.
3. `entry_reason`: trader-readable reason.
4. `initial_target_source`: VP, TLV2/trendline, pattern projection, prior pivot, or none.
5. `target_levels`: one or more projected levels where partial exit/stop tightening may be considered.
6. `invalidation_source`: structure break, VP acceptance/rejection failure, TLV2 retest failure, BOS/CHoCH flip, pattern confirmation failure, opposing pattern, crash state, etc.
7. `current_confluence_state`: same-direction and opposing signals since entry.
8. `position_action_state`: hold/add/reduce/tighten/exit state and reason.
9. `superseded_by`: later signal/family if it replaces the original target/invalidation logic.
10. `primary_trigger`: the indicator/rule family that actually opened the trade.
11. `primary_guard`: the strongest non-trigger evidence that made the trade acceptable.
12. `target_provider`: the source family providing the next obstacle or objective.
13. `invalidation_provider`: the source family defining where the entry idea is broken.
14. `profit_bucket`: loss, flat, small profit, material profit, strong profit, or post-partial state.
15. `partial_state`: no partial, first partial done, second partial done, or runner-only.

## Target and obstacle sources

Target/invalidation levels may come from:

1. Volume Profile: POC, VAH, VAL, HVN/LVN, value-area acceptance/rejection.
2. Trendlines / TLV2 support-resistance.
3. Pattern projections: wedges, triangles, pennants, head and shoulders, Wolfe waves, etc.
4. Prior pivots, highs/lows, higher highs/lower lows.
5. BOS/CHoCH and market-structure state.
6. Orderbook shelves, walls, liquidity vacuums, wall removal/rebuild, and venue agreement only when a future objective explicitly routes orderbook into Sieve.
7. Later news/context crash/risk-off signals only when data readiness and the active objective explicitly route them into Sieve.

## Entry-specific exits

Prefer entry-family-specific exits over broad generic exits.

Examples of valid exit/reduce logic:

1. Price reaches a projected VP POC/VAH/VAL, TLV2 level, pivot, or pattern target and the source-specific guard weakens: partial exit or stop tightening.
2. VP acceptance fails after a VP-triggered breakout or reclaim: exit, tighten, or partial based on current PnL and target proximity.
3. TLV2 retest fails after a TLV2-triggered break: tighten to structure, reduce, or exit based on PnL bucket.
4. BOS/CHoCH flips against the trade: reduce/tighten/full exit based on whether the original trigger remains intact.
5. Pattern confirmation fails, pattern target is touched, or exposed pattern score/direction flips: use family-specific pattern action logic.
6. Crash/risk-off signals appear while profitable: tighten stop to protect profit rather than blindly exit every trade.
7. Trigger remains intact but a guard weakens: ignore, tighten, reduce, or partial based on current profit and target proximity.
8. Trigger invalidates and guard also flips against the trade: compare full exit, partial plus lock, and tight structure stop based on PnL bucket.
9. Target is touched while same-direction evidence remains aligned: partial, stop shift, or trail to the next target rather than automatic full exit.

For current Sieve3 exit regeneration, active project indicator target/invalidation families are VP, TLV2, pivots, BOS/CHoCH, and custom pattern families. Orderbook, FreqAI, news/context, and broader confluence sources remain parked unless the active objective explicitly routes them in.

## Adds / stacking

Same-direction signals may justify add/hold/confidence only when:

1. they are independent or different-family signals,
2. they appear after or near the original entry,
3. they do not simply duplicate the original trigger,
4. the added risk is tested against the baseline,
5. later signals have their own target/invalidation logic or clearly strengthen the original logic.

## Reductions / stop tightening

Opposite-direction or risk signals may justify exposure reduction, stop tightening, or full exit when:

1. they come from independent evidence,
2. they invalidate the original entry reason,
3. they signal broad crash/risk-off state,
4. they occur near a known target/resistance/support zone,
5. they appear after profit is already available and the better action is to protect it.

## Rare pattern entries

The old "complete sunflower" wording is invalid. Use actual pattern/family names only: e.g. head and shoulders, inverse head and shoulders, Wolfe waves, pennants, triangles, wedges, and other custom indicator pattern families.

Known user intent:
- rare custom pattern entries may have high success but low frequency,
- same-direction rare pattern during an open trade can be add/hold evidence,
- opposite-direction rare pattern can be reduce/exit evidence,
- each pattern should remain separately named.

## Leverage/risk

Do not optimize leverage before entry, exit, add/reduce, and invalidation logic are validated. Leverage/risk can later use confluence/conflict strength, crash risk, volatility, and source readiness.

## Bounded overnight testing guidance

For improvement loops, test the smallest coherent exit/risk change first:

1. A full exit that damages winners should be converted once into partial exit, stop tightening, or reduce-exposure logic.
2. A same-direction signal overlap should first be tested as add/hold/confidence evidence, not as a global stake-reserve rule.
3. An opposite-direction signal should first be tested as tighten/reduce/exit evidence, not automatic reversal.
4. A target-zone rule should name the target source: VP, TLV2, prior pivot, pattern projection, or liquidity zone.
5. A crash/risk-off rule should usually tighten stops or reduce exposure, especially when the trade is already profitable, unless evidence supports a full exit.
