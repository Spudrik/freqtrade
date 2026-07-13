# FreqAI Parallel Branch Goals

Use this document when splitting the next FreqAI research work into parallel goals. The branches are designed to stay separate while evidence is weak, then merge later into trading research as `setup signal + risk state + orderbook confirmation + regime filter`.

## Shared Rules

1. Every branch must start with a trader question, not a model/profile name.
2. Every branch must state:
   - what a trader would see
   - which numeric feature groups represent it
   - what was asked afterwards
   - what control rows were compared
   - whether the result is useful in plain English
3. Use only validated timestamp-safe 1h feature caches.
4. Park news/context in these branches until its separate formatting work is marked ready.
5. Prefer component-only profiles for discovery. Use flag/gated profiles only to validate a defined trader state.
6. Do not reject a good structure-only or price/volume-only signal because it lacks orderbook/news confluence.

## Branch A: Structure + Volume Breakouts

1. Objective:
   - Find technical breakout/failure behaviours that work from price, volume, and custom indicators.
2. Trader questions:
   - When price breaks resistance with rising volume, does it keep going?
   - When price pokes through resistance but volume fades, does it fail?
3. Numeric feature groups:
   - OHLCV and volume pressure.
   - VP levels such as POC, VAH, VAL, HVN/LVN where available.
   - TLV2 support/resistance.
   - BOS/CHoCH and market structure.
   - Range/compression and pattern geometry states.
4. Answer columns:
   - `breakout_success_next_6h`
   - `breakout_failure_next_6h`
   - `future_max_upside_6h`
   - `future_return_6h`
5. Controls:
   - Price-only.
   - Same setup without volume confirmation.
   - Random rows from the same regime.
   - Opposite-direction breakdown setup.
6. Merge role:
   - This branch becomes the setup engine.

## Branch B: Downside Risk And Exhaustion

1. Objective:
   - Separate dangerous downside continuation from late/exhausted downside moves.
2. Trader questions:
   - After the first downside break is visible, does the drop keep falling?
   - After support breaks, can we tell when price is likely to reclaim support?
3. Numeric feature groups:
   - Lower lows, support breaks, range-low breaks, value-area loss.
   - Volume expansion or fade after the break.
   - Support-reclaim evidence.
   - Orderbook support removal, bid rebuild, spread/fragility.
4. Answer columns:
   - `downside_continuation_next_3h`
   - `downside_continuation_next_6h`
   - `downside_exhaustion_next_6h`
   - `support_reclaim_next_6h`
   - `large_drawdown_next_6h`
5. Controls:
   - Price-only.
   - Same break without orderbook confirmation.
   - Same break without volume persistence.
   - Random crash-regime rows.
6. Merge role:
   - This branch becomes the risk-avoidance and late-move filter.

## Branch C: Orderbook Behaviour States

1. Objective:
   - Test whether the book confirms, rejects, or warns against the chart signal.
2. Trader questions:
   - Does wall removal make a structure break more likely to continue?
   - Does support/resistance rebuilding warn that a move is failing?
   - Does venue agreement matter more than one venue alone?
3. Numeric feature groups:
   - Wall persistence.
   - Wall evaporation/removal.
   - Support/resistance rebuild.
   - Liquidity vacuum.
   - Pressure flip.
   - Spread shock/fragility.
   - Spot/linear/inverse agreement where available.
4. Answer columns:
   - `breakout_success_next_6h`
   - `breakdown_success_next_6h`
   - `breakout_failure_next_6h`
   - `support_reclaim_next_6h`
5. Controls:
   - Matching structure-only setup.
   - Price-only.
   - Same setup without orderbook-present rows.
   - Opposite-direction orderbook pressure.
6. Merge role:
   - This branch becomes the confirmation or warning layer.

## Branch D: Regime And Confluence Gate

1. Objective:
   - Find market states where other branch signals should be trusted, reduced, or ignored.
2. Trader questions:
   - Does a breakout work better in compression, trend, or balanced-book regimes?
   - Does support reclaim matter more after panic than during ordinary chop?
3. Numeric feature groups:
   - Trend/range/chop state.
   - Volatility compression/expansion.
   - Higher-timeframe alignment.
   - Balanced versus extreme orderbook.
   - Later: news/context severity and persistence when ready.
4. Answer columns:
   - Branch-specific success/failure columns.
   - `future_return_6h`
   - `future_max_drawdown_24h`
5. Controls:
   - Same signal outside the regime.
   - Random rows inside the same regime.
   - Branch result with and without the regime gate.
6. Merge role:
   - This branch becomes the final trust/risk gate.

## Goal Completion Per Branch

1. Direct-test evidence exists with controls.
2. A FreqAI validation queue exists for promoted profiles.
3. Results are scored against price-only and relevant source-family controls.
4. Weak ideas are explicitly rejected.
5. Next refinements are listed in trader language.

