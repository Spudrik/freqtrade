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

`pivot_based_market_structure.py`

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

`pattern_continuation.py`

Flag/pennant-style structure built from impulse, controlled retracement, range
contraction, dry volume, and pivot context. This file owns continuation pattern
outputs directly; the old composite `complex_pattern_structure.py` layer is
archived.

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

## Review Priority

Before confluence, review each score independently:

- Visually confirm the indicator does what its name claims.
- Check long and short scores separately.
- Check score usefulness per timeframe.
- Check whether high absolute scores produce cleaner trades or just more noise.
- Keep orderbook features separate until the collected data is stable.
