# Indicator Score Guide

This directory contains Freqtrade-focused indicator modules. They should be
used as dataframe feature builders, not as standalone trading systems.

## Shared Score Contract

Every base indicator should expose the same strategy-facing score contract:

- `*_score_long`: normalized `0..1` evidence favoring long setups.
- `*_score_short`: normalized `0..1` evidence favoring short setups.
- `*_score_abs`: normalized `0..1` evidence strength, direction ignored.
- `*_state`: directional lean, usually `-1`, `0`, or `1`; this is not a score.

The scores are comparable in shape, not in proven predictive power. A high
`vp_score_long` and a high `pa_score_long` both mean "this module sees long-side
evidence," but real-data validation must decide which score is actually useful.

## Hyperopt Usage

Indicator configs are dataclasses and public function kwargs mirror the config
fields. Freqtrade strategies can pass `IntParameter`, `DecimalParameter`, and
`CategoricalParameter` values into those configs during `populate_indicators`.

The goal is to hyperopt indicator behavior, then measure whether the output
scores have useful forward-return behavior before combining them.

## Module Meanings

`complex_pivot_structure.py`

Confirmed no-lookahead pivots, HH/HL/LH/LL, BOS/CHoCH, swing state, and channel
targets. Long scores rise with constructive higher-high/higher-low structure,
bullish breaks, upward channel bias, and available upside target space. Short
scores mirror this for bearish structure.

`complex_trendline_projection.py`

Trendline and channel evidence derived from confirmed pivots. Long scores rise
when support/channel lines are valid, respected, reclaimed, and breaking upward.
Short scores rise when resistance lines are valid, respected, rejected, and
breaking downward. Compression contributes to both because it can precede
directional expansion.

`complex_volume_indicators.py`

OHLCV-only volume behavior: pressure proxy, CVD proxy, volume regimes,
effort-vs-result, liquidity sweeps, and anchored VWAP. Long scores rise with
bullish pressure, absorption/spring behavior, stop-runs, AVWAP reclaim, and
volume-confirmed breakouts. Short scores mirror those ideas.

`complex_volume_profile.py`

Rolling volume profile with POC, VAH/VAL, HVN/LVN, profile shape, and node
interaction behavior. Long scores rise when value migrates higher, price accepts
above value, failed downside auctions reclaim value/HVN/LVN zones, or price
traverses thin LVN liquidity upward with pressure. Short scores mirror this.

POC migration means the point of control is moving, showing where the market is
accepting value over time. LVN rejection/acceptance means price is interacting
with thin historical liquidity zones that may reject sharply or traverse quickly.

`complex_pattern_structure.py`

Flag/pennant-style structure built from impulse, controlled retracement, range
contraction, dry volume, and optional pivot/trendline context. This treats flags
as a variant of pivot/trendline behavior rather than a separate visual-only
pattern.

`complex_volatility_cycles.py`

Compression, expansion, and exhaustion states. Compression scores rise when ATR,
range, and volume contract. Expansion scores rise when range and volume expand.
Exhaustion scores rise when a large range, high relative volume, and extreme
close location suggest a stretched move.

`complex_relative_strength.py`

Pair performance relative to a supplied benchmark such as BTC, ETH, or a market
proxy. Long scores rise when the pair is outperforming the benchmark across
multiple windows and the relative-strength line is high in its rolling range.
Short scores rise when it underperforms.

`external_global_context_features.py`

First-pass external-data feature module. It reads the launcher Global Context
SQLite database, aligns effective 0..100 context scores to strategy candles, and
emits only rolling score, score delta, and fear/greed persistence columns. Treat
it as context/filter research, not a standalone entry system.

`simple_confluence_indicator.py`

Broad textbook TA confluence: RSI, MACD/PPO, Bollinger, EMA/SMA/DEMA/TEMA/WMA/HMA,
stochastic, StochRSI, ADX/DMI, CCI, MFI, CMO, ROC, Williams %R, Aroon, TRIX,
Ultimate Oscillator, Awesome Oscillator, KST, ATR/Keltner/Donchian, VWAP, OBV,
CMF, ADL, Elder Force Index, Ichimoku, z-score, volume breakouts, and common
candlestick patterns. It emits paired 0/1 `*_long` and `*_short` flags for each
rule, unbounded long/short signal counts, normalized count-based scores, and a
directional state. Treat it as a research sieve for simple TA confluence, not as
a refined edge model.

## Validation Priority

Before confluence, validate each score independently:

- Bucket future returns by score decile.
- Check long and short scores separately.
- Check score usefulness per timeframe.
- Check whether high absolute scores produce cleaner trades or just more noise.
- Keep orderbook features separate until the collected data is stable.

## Automated Validation Tool

The launcher-integrated validation runner lives at
`user_data/new2026/indicator_validation_runner.py`. It is designed to answer a
different question from hyperopt:

- Hyperopt asks, "Can a strategy trade this profitably?"
- Indicator validation asks, "Does this score sort future market behaviour in
  the direction the indicator claims?"

The runner loads Freqtrade OHLCV files, computes selected indicators, and writes
repeatable reports under `user_data/runtime/indicator_validation` by default.
By default it validates only base indicator scores: `*_score_long`,
`*_score_short`, and `*_score_abs`. Set score scope to `all` when you want to
research internal component scores as possible future strategy features.

Outputs:

- `latest_indicator_validation_summary.json`: run config, pass rates, failures,
  and top score candidates.
- `latest_indicator_score_metrics.csv`: score-vs-forward-outcome metrics per
  pair, timeframe, score column, and forward candle window.
- `latest_indicator_deciles.csv`: raw decile buckets so score shape can be
  inspected without manually plotting every chart.
- `latest_indicator_behavior.csv`: checks whether high scores align with their
  internal evidence columns.
- `latest_indicator_contract.csv`: normalized score and state contract checks.

Primary metrics:

- `validity_score`: normalized `0..1` quality estimate combining decile
  monotonicity, top-vs-bottom edge, MFE/MAE improvement, and raw score/outcome
  correlation.
- `top_bottom_edge`: mean forward outcome in the highest score bucket minus the
  lowest score bucket. For short scores, returns are direction-adjusted so
  positive still means useful short-side sorting.
- `decile_expected_corr`: correlation between score bucket rank and forward
  outcome. Higher is better when a score is meant to become more useful as it
  rises.
- `top_mfe_mae_ratio`: whether high scores produce cleaner opportunity relative
  to adverse excursion.
- `contract_pass`: confirms score columns stay normalized to `0..1` and state
  columns stay directional.
- `alignment_pass`: confirms high scores are backed by the evidence columns the
  indicator claims to represent.

Interpretation rules:

- Treat a high `validity_score` as a research lead, not proof of a complete
  trading strategy.
- Prefer indicators whose decile tables improve smoothly, not just one lucky
  bucket.
- Compare long and short scores independently; one side may be useful while the
  other side is noise.
- Validate per timeframe. A structure score that works on `4h` may be noise on
  `5m`.
- If hyperopt likes a score but this tool shows poor contract, evidence, or
  decile behaviour, assume the strategy is curve-fitting around noise until
  proven otherwise.
