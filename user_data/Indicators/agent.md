# Indicator Agent Notes

This file is for active instructions only. Strategy-output inventories, old archive notes, and historical experiment logs belong in separate reference files, not here.

## Indicator Objective

Build a multilayer Freqtrade indicator stack that finds real market edge instead of relying on lagging confirmation alone.

Core rules:
- Prefer no-lookahead, confirmed structure over repainting signals.
- Use vectorized NumPy and pandas methods wherever possible.
- Avoid row-by-row dataframe loops.
- Non-vectorized indicator logic is not acceptable unless the user gives explicit written permission first.
- If a non-vectorized exception is approved, add a code comment at the exception site explaining why vectorization was not practical.
- Bounded loops over strengths, horizons, ranks, or candidate slots are acceptable only when the per-candle work remains vectorized over Series or arrays.
- Expose tuning knobs so Freqtrade hyperopt can optimize feature logic and thresholds, but document only the small coarse first-pass levers that strategies should start with.
- Indicator modules produce dataframe evidence, context, confidence scores, and optional advice flags. Strategies own final entry, exit, sizing, stake, add, peel, and risk decisions.
- Advice flags such as `go_long`, `go_short`, `exit_long`, and `exit_short` are allowed when they are simple, explicit, and derived from that indicator's own evidence. Strategies still own final trade decisions and additional guards.
- Do not build indicator or strategy ideas around standard technical indicators such as RSI, MACD, Bollinger Bands, Stochastic, ADX, generic oscillator crosses, or similar TA-library staples.
- Do not use moving averages, EMA/SMA trend filters, MACD-style average crosses, or generic mean-reversion bands as Sieve entry guards.
- If a proposed indicator mainly repackages standard TA under new names, stop and tell the user before spending refinement time.
- Prefer structure, liquidity, volume profile, and pattern evidence where the output is genuinely different.
- Prefer price-action logic over lagging confirmation: reversal behaviour, continuation behaviour, breakout/breakdown behaviour, rejection/reclaim behaviour, and whether price is moving in the intended direction after the event.
- Volume is the standard-indicator exception. Accumulation, relative volume, volume spikes, directional volume pressure, CVD-like pressure, and volume-profile participation are valid guards or supporting evidence.
- External order-book, global-context, and news/web sentiment modules are parked in `work_in_progress/`. They require much deeper historical data coverage and should not be imported by strategy files unless the user explicitly reopens external-context research.
- Strategy-facing outputs must be lean, practical evidence a strategy can directly consume.
- Do not emit duplicate booleans when one numeric value lets the strategy set its own threshold.
- Do not emit hidden proof columns or blended magic-number scores from normal outputs unless explicitly requested.
- Prefer raw evidence plus exposed tunable gates over opaque quality formulas.

Priority indicator families:
- Confirmed pivot structure and BOS/CHoCH market-structure events.
- Trendline projection from pivots.
- Volume profile, HVN/LVN, VAH/VAL, and POC migration.
- Pattern detection: flags, pennants, wedges, triangles, double/triple peaks, head-and-shoulders, Wolfe waves, channels, rectangles, and HH/HL sequence scoring.

## Strategy-File Guidance For Indicator Agents

- When indicator agents write strategy-facing documentation or seed strategy ideas, they must describe targeted Sieve strategy files, not broad all-knob experiments.
- A normal Sieve strategy file should test one labelled entry concept, for example value-area-low reclaim, value-area-high rejection, support-line reclaim, resistance-line reject, pattern confirmation, or BOS continuation.
- Explicitly labelled confluence probes are allowed when the point of the file is to test a specific combination of named ideas.
- Hyperopt surfaces must be coarse and small. Prefer `CategoricalParameter` choices with a few meaningful values over wide decimal or integer ranges.
- Strategy files should not retune full indicator construction. Use indicator defaults unless the strategy-facing reference identifies a small number of first-pass levers for that indicator.
- Optional validation guards should use enable/disable flags plus coarse ranges. Volume and pressure guards are preferred supporting checks.
- Do not suggest RSI, MACD, Bollinger, stochastic, moving-average trend filters, or generic TA-stack guards. If a guard is needed, prefer volume, pressure, local price-action direction, proximity/reclaim/rejection behaviour, or an output from the active custom indicators.
- If directional confirmation is needed, use direct price-action checks rather than moving averages: close above/below the previous close, close above/below a project-indicator level for a small number of candles, reclaim/rejection of an indicator level, candle body direction, or pressure alignment with the trade side.
- The strategy objective is to find statistically useful entries through Sieve diagnostics. It is not to optimize every available lever or make a final production trading system.

## Active Indicator Stack

Current foundation order:
- `pivot_foundation.py` is the canonical cleaned-pivot source. Shared pivot behaviour belongs here first.
- `pattern_bos_choch.py` consumes the foundation for focused break-of-structure and change-of-character event evidence. It does not score setups, draw channels, or create structural zones.
- `complex_trendline_projection_v2.py` consumes the foundation and is the canonical ranked trendline generator.
- All indicators may use `pivot_foundation.py` as input. They should not define their own pivots.
- Active indicators should expose a public `add_*` entrypoint callable from strategy code with an OHLCV dataframe, timeframe, optional config, and overrides, returning the input dataframe plus output columns.
- No indicator may inherit unrelated dataframe plumbing from another indicator. Cross-indicator inputs are allowed only when consuming that indicator's actual concept.
- Trendlines are explicitly lines only. Shapes and patterns are defined exclusively in pattern-focused indicators.

Current disposition:
- Active strategy candidates: Volume Profile, Relative Strength, BOS/CHoCH Structure, Trendline V2, Geometry V2, Reversal, Continuation, Multi-Peak, and Wolfe Wave.
- Strategy-facing documentation must describe active indicators neutrally: output columns, meanings, and starting hyperopt levers/ranges. Do not include reliability labels, return summaries, or quality opinions that could bias strategy construction.
- Pattern detectors are split by family. Geometry v2 covers triangle, wedge, compression, and channel/rectangle context. Reversal, continuation, multi-peak, and Wolfe wave logic live in their own modules.
- Parked external context: Orderbook Context, Global Context, and News/Web Sentiment live under `work_in_progress/` and are not active strategy-facing indicators.

Do not add archive history to this file. If a module is archived or removed, new agents usually do not need that context. Archived code is out of scope unless the user explicitly reopens it.

## Foundation Dependency Hard Rule

- Indicators must not depend on an unrelated indicator namespace just to get shared foundation data.
- If a module needs pivots, call `pivot_foundation.py` directly or consume a clearly documented canonical pivot dataframe layer.
- Do not route pattern detectors through TLV2 pivot columns just to get pivots.
- Do not route Trendline V2 through Pattern columns.
- Do not create hidden dependency chains that work only because another indicator happens to re-emit foundation columns.
- Cross-indicator inputs are allowed only when the actual concept is being consumed, for example Geometry V2 using TLV2 ranked lines as trendline evidence.
- Cross-indicator inputs are not allowed as plumbing shortcuts for pivots, volume, ATR, or other shared base data.

## Height And Level Scaling Hard Rule

- Every active indicator height, level, boundary-distance, pivot-distance, and same-price comparison must adjust to timeframe.
- Use dynamic scale from rolling body size, ATR/range, pivot prominence, volume-profile width, order-book/liquidity scale, or another documented market-scale source appropriate to the indicator.
- Do not use a fixed raw percentage as the primary decision tolerance for height comparisons.
- Static percent parameters may exist only as safety ceilings, sanity caps, or disabled compatibility aliases. Comments must make that clear.
- If an indicator still contains legacy fixed-percent height logic, flag it before calling that indicator production-ready.

## Strategy-Facing Naming Standard

Use names that make the advice/evidence split obvious.

Advice-only:
- Use `go_long`, `go_short`, `exit_long`, `exit_short`, `hold_long`, or `hold_short`.

Advice plus cause/context:
- Use double underscores to separate advice from reason.
- Preferred examples: `go_long__bounce_at_support`, `go_short__rejection_at_resistance`, `exit_long__support_lost`, `exit_short__resistance_reclaimed`.

Cause/evidence only:
- Use neutral behaviour names with no trade-side suffix when possible.
- Preferred examples: `bounce_at_support`, `rejection_at_resistance`, `breakout_above_resistance`, `breakdown_below_support`, `near_support`, `near_resistance`.

Avoid:
- `long_bounce_support`
- `short_reject_resistance`
- `resistance_reject_short`
- Any name where the reader cannot tell whether `_long` or `_short` is advice, bias, position side, or market behaviour.

Existing strategy-facing pattern columns that do not follow this style should be cleaned up before the relevant pattern file is called production-ready.

## Outstanding Reminders

- Pattern families are now split by indicator. Do not resurrect the old Pattern Structure aggregation layer unless the user explicitly asks for a new aggregation design.
- `pattern_geometry_v2.py` owns triangle, wedge, compression, rectangle, ascending channel, and descending channel outputs.
- Keep future geometry changes practical, tunable, timeframe-aware where the strategy passes timeframe, and free of duplicate strategy columns.
