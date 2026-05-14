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
- Expose tuning knobs so Freqtrade hyperopt can optimize feature logic and thresholds.
- Indicator modules produce dataframe evidence, context, confidence scores, and optional advice flags. Strategies own final entry, exit, sizing, stake, add, peel, and risk decisions.
- Advice flags such as `go_long`, `go_short`, `exit_long`, and `exit_short` are allowed when they are simple, explicit, and derived from that indicator's own evidence. Strategies still own final trade decisions and additional guards.
- Avoid redesigning standard technical indicators unless the output is materially different from TA-Lib, qtpylib, ATR/range, relative volume, or simple rolling statistics.
- If a proposed indicator mainly repackages standard TA under new names, stop and tell the user before spending refinement time.
- Prefer structure, liquidity, volume profile, order-book, and pattern evidence where the output is genuinely different.
- OHLCV-only volume and volatility proxies are lower priority when real order-book or richer market microstructure data is available.
- Strategy-facing outputs must be lean, practical evidence a strategy can directly consume.
- Do not emit duplicate booleans when one numeric value lets the strategy set its own threshold.
- Do not emit hidden diagnostics, proof columns, or blended magic-number scores from normal outputs unless explicitly requested.
- Prefer raw evidence plus exposed tunable gates over opaque quality formulas.

Priority indicator families:
- Confirmed pivot structure and BOS/CHoCH market-structure events.
- Trendline projection from pivots.
- Volume profile, HVN/LVN, VAH/VAL, and POC migration.
- Real order-book and liquidity behaviour.
- Pattern detection: flags, pennants, wedges, triangles, double/triple peaks, head-and-shoulders, ranges, and HH/HL sequence scoring.

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
- Done/ready for strategy tests: Volume Profile, Pivot Foundation, BOS/CHoCH Structure, Trendline V2, and Orderbook Context.
- Pattern detectors are split by family. Geometry v2 covers triangle, wedge, compression, and channel/rectangle context. Reversal, continuation, range, multi-peak, and Wolfe wave logic live in their own modules.
- Work in progress: broad pattern regression review for hypertesting, plus any individual pattern files the user explicitly reopens.
- Strategy-facing first-pass external context: Global Context and News/Web Sentiment. These are context/filter inputs only until forward-return and drawdown buckets prove stable usefulness.

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
- If an indicator still contains legacy fixed-percent height logic, mark it for review before calling that indicator production-ready.

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

Existing strategy-facing pattern columns that do not follow this style should be marked for cleanup before the relevant pattern file is called production-ready.

## Indicator Review Method

Visual review is the primary semantic validation method. Automated score or forward-return scripts are not proof of semantic correctness and should not be rebuilt for indicator review.

Before tuning an indicator:
- Define the expected market behaviour in plain English.
- Manually identify the expected pattern/structure/window first.
- Use targeted validation windows where the expected behaviour exists.
- Use control windows where the behaviour should not appear.
- Start focused, usually BTC only, then expand to ETH/SOL and other windows once the core definition improves.
- Source extra historical windows online if needed, but manually review the chart before judging the indicator.
- If a requested timeframe is missing locally, fetch it through Freqtrade before designing detection fixes.

Layered implementation requirement:
- Build indicators in explicit staged layers.
- Each stage should be inspectable with a focused plot.
- Tune the first bad stage, not the final output.
- Re-run the original failed plot after each change.
- Only move forward when the current stage is visually rational.

Standard stage flow:
- Stage 1: raw candles.
- Stage 2: foundation pivots or base market data.
- Stage 3: candidate window or search region.
- Stage 4: selected anchors/levels/events.
- Stage 5: raw shape/line/zone construction.
- Stage 6: core filters and dynamic scaling gates.
- Stage 7: merge, de-duplicate, and score.
- Stage 8: strategy-facing columns.
- Stage 9: no-lookahead prefix check.

Review targets:
- Initial pass target: roughly `70%` hit rate on known positive windows and no worse than about `1` confirmed bad pattern per `200` candles.
- Current acceptance target: `<=15%` miss rate on known real-pattern windows, equivalent to `>=85%` rational identification.
- Control-window events are not automatically false positives. Count them as errors only after visual review shows the confirmed pattern was unjustified noise rather than a plausible developing setup.
- Longer-term control baseline: report confirmed pattern errors/events per candles reviewed, with events per `500` candles as the first standard metric.

Plot sharing standard:
- Share one plot at a time.
- Prefer direct links that open the file in the Codex side panel.
- Do not embed multi-plot grids when a single focused plot is more readable.
- Before each plot link, state whether it is a validation or control plot.
- For validation plots, state how many known target structures are expected and how many were hit, for example: "expected 3 triangles, hit 2/3".
- Keep a reusable pool of known validation/control windows so progress can be compared across refinement loops.

Validation source hard rule:
- Validation plots must run the production indicator file against imported OHLCV data.
- Every plotted indicator line, marker, trace, or event must come from dataframe columns emitted by that production indicator run.
- Do not use parallel detector scripts, copied algorithm variants, synthetic trace lines, or internal-only arrays as proof of strategy-facing indicator output.
- Passes from isolated review slices are provisional. Re-run on continuous history before accepting a tuning change.
- Stage/debug plots may exist only for root-cause diagnosis, must be clearly labelled as non-strategy-output diagnostics, and must be deleted before calling an indicator finished.

Suggested plotting windows:
- `1h` local review: 250-600 candles.
- `4h` tactical/medium review: 300-900 candles.
- `12h/1d` structural review: 500-1200 candles.
- `3d` bigger-picture review: as much history as available, usually 500-1000 candles.

## Pattern Review Nuance

Pattern modules detect plausible developing structure. They do not prove the structure will play out.

Multi-timeframe interpretation:
- The agent's job is to assess whether a detected pattern is valid relative to the timeframe being reviewed.
- The agent must not downgrade or dismiss timeframe usefulness on the user's behalf.
- Larger structures do not need to be forced onto `1h` if the same market structure is captured cleanly on `4h`, `1d`, or `3d`.
- A miss of a large structure on `1h` can be acceptable when the higher-timeframe detector captures it.
- If a lower-timeframe broad pattern is missed but a higher timeframe captures the same valid structure cleanly, that can count as a pass only after the agent flags it to the user and gets approval.
- If a valid large structure is missed on both lower and higher timeframes, improve the higher-timeframe detector first.
- In most strategies, `1h` is trigger/timing data. `4h` is tactical and may be traded depending on testing. `1d` and `3d` are higher-priority setup/context timeframes.
- The indicator should still be robust across all timeframes: if it emits a `1h` pattern, that pattern should be visually valid on `1h`, even if strategies later ignore it.
- Resampled higher-timeframe validation is provisional. Prefer native exchange timeframe data when available.

Peak-pattern definition preference:
- Double/triple tops and bottoms should use deliberately simple structure first.
- Price must move meaningfully into the first pivot.
- Price must react away within a bounded number of candles.
- Price must retest the same level within dynamic tolerance.
- Move, reaction, base-hold buffer, and same-level tolerance must scale across timeframes using typical candle-body size, ATR/range, and confirmed pivot prominence where available.
- Neckline/body-reaction levels are management evidence for strategies, not final trade decisions.

## Development Support Code Standard

- Production indicator modules should contain dataframe feature logic only.
- Plotting scripts, diagnostic runners, validation-window scripts, prototypes, and one-off review code belong under `user_data/Indicators/development_support/`.
- Generated plots and bulk runtime outputs should not be committed.
- Write review images to dated folders under `user_data/plot/` and keep them unstaged.
- Do not leave plotting-only proof code, temporary diagnostics, or visual-review orchestration inside finished indicator modules.
- If diagnostic columns are needed for no-lookahead or visual inspection, gate them behind an explicit config flag such as `include_pattern_diagnostics=False` and document that they are not normal strategy-facing outputs.
- Do not create clone indicator files, `v2` files, or alternative experimental implementations unless the user explicitly asks for that exact experiment.

## Outstanding Reminders

- Pattern families are now split by indicator. Do not resurrect the old Pattern Structure aggregation layer unless the user explicitly asks for a new aggregation design.
- `pattern_geometry_v2.py` owns triangle, wedge, compression, rectangle, ascending channel, and descending channel outputs. Treat triangle/wedge/compression as the stronger validated family; treat channel outputs primarily as context/avoidance evidence unless future testing upgrades that confidence.
- Geometry v2 validation focus is 1h and 4h. Higher timeframe outputs may be useful context, but do not retune the file around 8h/1d/3d without explicit user approval.
- Run all reviewed pattern indicators across a broad range of validation and control windows to confirm they still function well enough for hypertesting after recent changes.
- Keep future geometry changes practical, tunable, timeframe-aware where the strategy passes timeframe, and free of duplicate strategy columns.
