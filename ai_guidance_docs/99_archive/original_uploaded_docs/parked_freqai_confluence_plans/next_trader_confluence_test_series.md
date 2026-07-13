# Next Trader-Confluence Test Series

Use this as the next plain-English design backlog for price, structure, volume, and orderbook tests. News/context is intentionally excluded until its separate rework is ready.

## Reporting Rule

Every test must be explained using `result_communication_format.md` before giving model metrics.

## Test 1: Breakout Keeps Going

1. Trader question:
   - When price breaks above resistance with rising volume, does it keep going?
2. What we look at:
   - Price near resistance.
   - Resistance from VP, TLV2, prior highs, or pattern boundary.
   - Volume expands as price breaks.
   - Orderbook resistance above price is weak, removed, or thin.
3. What we ask afterwards:
   - Did price keep moving up over the next 3h or 6h?
   - Did it reach the next visible wall/value area before falling back?
4. Useful result would mean:
   - The setup helps identify breakouts worth following.

## Test 2: Breakout Fails

1. Trader question:
   - When price breaks resistance, can we tell whether it is likely to fail?
2. What we look at:
   - Price breaks or pokes above resistance.
   - Volume does not expand enough, or fades quickly.
   - Orderbook resistance remains strong or support does not rebuild below price.
   - Price struggles to stay above the level.
3. What we ask afterwards:
   - Did price fall back under the broken level?
   - Did price return into the old range/value area?
4. Useful result would mean:
   - The setup helps filter bad breakouts.

## Test 3: Downside Momentum After First Break

1. Trader question:
   - After the first downside break has started, can we tell whether the drop is likely to keep going?
2. What we look at before the break:
   - Price was near support, range low, or value-area low.
   - Support was weakening.
   - Buy-side orderbook walls were losing persistence.
3. What we look at during the first 1h to 4h after the break:
   - Price keeps making lower lows.
   - Sell pressure expands.
   - Buy walls disappear, move lower, or fail to rebuild.
   - Spread/fragility increases.
   - Volume remains high instead of fading.
4. What we ask afterwards:
   - Did price keep dropping over the next 3h, 6h, or 24h?
   - Did it fail to reclaim broken support?
5. Useful result would mean:
   - The setup helps identify crash continuation risk after the first break has already started.

## Test 4: Downside Move Slows Or Fails

1. Trader question:
   - After an initial drop, can we tell when the move is running out of steam?
2. What we look at:
   - Price has already dropped or broken support.
   - Sell pressure stops increasing.
   - Volume fades.
   - Buy walls rebuild near or below price.
   - Price stops making lower lows or reclaims broken support.
3. What we ask afterwards:
   - Did price stop falling?
   - Did price bounce or chop instead of continuing down?
4. Useful result would mean:
   - The setup helps avoid chasing late downside moves.

## Test 5: Technical Breakout With Quiet Book

1. Trader question:
   - When news/context is ignored and orderbook is balanced, do technical levels alone guide the move?
2. What we look at:
   - Price is at a meaningful technical level.
   - Orderbook is balanced, not extreme.
   - Volume confirms the break.
3. What we ask afterwards:
   - Did price travel toward the next VP/orderbook/technical level?
4. Useful result would mean:
   - Some market states are mostly technical and do not need extra source confluence.

## Test 6: Orderbook Panic State

1. Trader question:
   - Does the orderbook show a panic/risk state after a move starts?
2. What we look at:
   - Multiple venues shift in the same direction.
   - Walls move away from price.
   - Liquidity thins in the direction of travel.
   - Spread or fragility rises.
3. What we ask afterwards:
   - Did price move faster or further after the book entered this state?
4. Useful result would mean:
   - Orderbook is useful as a risk accelerator or continuation filter, not just as a standalone signal.

## Controls Required For These Tests

1. Same setup without volume confirmation.
2. Same setup without orderbook confirmation.
3. Same setup without structure confirmation.
4. Similar random rows from the same market regime.
5. Opposite-direction setup where relevant.
6. Month/window stability check.

