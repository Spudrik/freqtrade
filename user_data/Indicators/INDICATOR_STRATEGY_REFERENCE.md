# Indicator Strategy Reference

This file is reference material for transferring active indicators into strategies. It is not an agent instruction file.

## Active Outputs Summary

Volume Profile:
- Provides market-profile levels and directional context.
- Context convention: `+2` bull, `+1` bullish chop, `0` undefined, `-1` bearish chop, `-2` bear.
- Strategies may use profile levels, value migration, HVN/LVN behaviour, entry triggers, hold evidence, and exit evidence as inputs.
- The indicator must not own final trade, stop, target, stake, or risk decisions.

Pivot Foundation:
- Canonical cleaned body pivots.
- Shared pivot behaviour belongs here first.
- Other indicators should consume this source instead of defining private pivots.

Pivot-Based Market Structure:
- Consumes Pivot Foundation.
- Provides strategy-facing pivot, structural-zone, market-structure, and context evidence.
- It does not detect channels.

Trendline V2:
- Consumes Pivot Foundation.
- Provides ranked support/resistance trendlines.
- It is lines only; shapes and patterns belong to pattern indicators.

Orderbook Context:
- Provides real order-book/liquidity context where collected data exists.
- Prefer this over OHLCV proxy orderflow when both are available.
- Treat orderbook columns as context, filter, risk, and timing evidence until strategy tests prove otherwise.

Global Context And News/Web Sentiment:
- First-pass external context/filter inputs.
- Ready for guarded research tests.
- Do not treat them as standalone entry signals until forward-return and drawdown buckets prove stable usefulness.

Pattern Indicators:
- Pattern outputs are split by family rather than aggregated through the old Pattern Structure layer.
- Geometry V2 emits triangle, wedge, compression, rectangle, ascending channel, and descending channel evidence through `pg2_*` slot columns plus row-level compression/channel context.
- Reversal, continuation, range, multi-peak, and Wolfe wave logic should be consumed from their own files.
- Multi-peak triple top/bottom identity uses dynamic level breach checks. `triple_level_breach_tolerance_mult` and `triple_level_breach_pivot_grace_bars` are strategy/hyperopt levers: tolerance controls body-level invalidation after reversal, while grace only forgives near-pivot fuzz before the interval has armed on a real opposing reaction.
- Strategies decide whether to trade, wait for breakout, use `1h` as trigger data, treat channel rails as avoid/context evidence, or ignore lower-timeframe patterns.

Multi-Peak Strategy / Hyperopt Guidance:
- Treat multi-peak output as pattern identity evidence only. A valid triple top/bottom is not a trade by itself; strategy logic still owns confirmation, direction, stop, target, timeframe weighting, and context filters.
- First-pass hyperopt surface should stay small and identity-focused: `min_triple_spacing_bars`, `min_triple_touch_similarity_score`, `min_triple_touch_turn_score`, `triple_level_breach_tolerance_mult`, `triple_level_breach_pivot_grace_bars`, and `min_triple_quality`.
- Second-pass surface may add pivot and impulse controls if plots show identity problems: `pivot_strength`, `pattern_pivot_strength`, `min_triple_neckline_depth_pct`, and `peak_prior_impulse_min_efficiency`.
- Avoid broad hyperopt over every internal scale knob initially. The dynamic body/ATR/prominence multipliers can interact heavily and may overfit samples before the pattern identity surface is stable.
- Sensible breach-grace search should keep `triple_level_breach_pivot_grace_bars` small, for example 0-4 bars. Larger values can forgive real breaks and turn failed structures back into accepted patterns.
- `min_triple_touch_turn_score` is a clipped score, so useful values should stay at or below 1.0.

## Naming Migration Target

Advice-only columns should use:
- `go_long`
- `go_short`
- `exit_long`
- `exit_short`
- `hold_long`
- `hold_short`

Advice plus reason should use:
- `go_long__bounce_at_support`
- `go_short__rejection_at_resistance`
- `exit_long__support_lost`
- `exit_short__resistance_reclaimed`

Neutral evidence should use:
- `bounce_at_support`
- `rejection_at_resistance`
- `breakout_above_resistance`
- `breakdown_below_support`
- `near_support`
- `near_resistance`

Existing pattern columns that do not follow this style should be migrated before the relevant pattern file is called production-ready.
