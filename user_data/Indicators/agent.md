# Indicator Objective

Build a multilayer Freqtrade indicator stack that finds real market edge instead of relying on lagging confirmation alone.

Core principles:
- Prefer no-lookahead, confirmed structure over repainting signals.
- Use vectorized NumPy and pandas methods wherever possible.
- Avoid row-by-row dataframe loops.
- Expose tuning knobs on every indicator so Freqtrade hyperopt can optimize the feature logic and thresholds.
- Indicator modules are for Freqtrade strategy consumption only; they should produce dataframes, evidence columns, and confidence scores, not final trading decisions.
- Score market state, not just single events.
- Combine signals across timeframes into a confluence score.
- Favor structure, liquidity, and regime context over basic lagging indicators alone.

Priority indicator families:
- Confirmed pivot structure and market structure state
- Trendline and channel projection from pivots
- Volume profile, HVN / LVN, VAH / VAL, POC migration
- Advanced volume behavior, effort-vs-result, anchored VWAP, liquidity sweeps
- Pattern detection such as flags, pennants, wedges, triangles, and HH / HL sequence scoring
- Market regime classification and transition scoring

Target outcome:
- Each module emits evidence columns plus a normalized confidence score.
- All key thresholds and windows should be configurable for hyperopt.
- Scores can be summed across multiple timeframes.
- The final system should produce trade setups that are logical, explainable, and testable against buy and hold.
