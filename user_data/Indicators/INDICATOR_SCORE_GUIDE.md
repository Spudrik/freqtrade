# Indicator Score Guide

This directory contains Freqtrade-focused indicator modules. Active indicators are dataframe feature builders for strategy and hyperopt work.

Strategy agents should assume active indicators are usable candidates. This guide intentionally describes output shape and meaning only; it does not rank indicators or comment on expected strategy quality.

## Shared Score Contract

Most base indicators expose the same strategy-facing score contract:

- `*_score_long`: normalized `0..1` evidence favoring long-side interpretation.
- `*_score_short`: normalized `0..1` evidence favoring short-side interpretation.
- `*_score_abs`: normalized `0..1` evidence strength, direction ignored.
- `*_state`: directional lean, usually `-1`, `0`, or `1`; this is not a score.

Event-only indicators may omit scores when their primary output is an event code or event flag. BOS/CHoCH is one such case.

## Hyperopt Usage

Indicator configs are dataclasses and public function kwargs mirror the config fields. Freqtrade strategies can pass `IntParameter`, `DecimalParameter`, and `CategoricalParameter` values into those configs during `populate_indicators`.

Hyperopt should start from the strategy-facing output columns and the initial lever ranges in `INDICATOR_STRATEGY_REFERENCE.md`.

## Active Module Meanings

`pattern_bos_choch.py`

Confirmed pivots converted into BOS/CHoCH event context. It emits event flags, structure event code, directional structure state, and optional HH/HL/LH/LL sequence labels.

`complex_trendline_projection_v2.py`

Ranked support and resistance trendline evidence derived from confirmed pivots. Main strategy-facing values are line price, line score, and distance in ATR units for support/resistance ranks.

`pattern_geometry_v2.py`

Line-pair geometry evidence for triangle, wedge, compression, rectangle, ascending-channel, and descending-channel families. Main strategy-facing values are family presence, score, direction, width, squeeze state, upper rail, and lower rail.

`complex_volume_profile.py`

Rolling volume profile with POC, VAH/VAL, HVN/LVN, profile context, score columns, and node/value-area interaction flags.

`complex_relative_strength.py`

Pair performance relative to a supplied benchmark such as BTC, ETH, or a market proxy. It emits relative score columns, directional state, trend z-scores, and regime-aware advice/context columns:

- `rs_go_long`
- `rs_go_short`
- `rs_hold_long`
- `rs_hold_short`
- `rs_exit_long`
- `rs_exit_short`
- `rs_long_caution`

`pattern_reversal.py`

Double top, double bottom, head-and-shoulders, and inverse head-and-shoulders outputs. Main strategy-facing values are pattern presence, confirmation, indicator score, confirmation level, target level, and pivot indices.

`pattern_continuation.py`

Flag and pennant outputs. Main strategy-facing values are setup presence, pattern presence, confirmation, direction, indicator score, upper/lower rails, confirmation level, invalidation flag, and invalidation level.

`pattern_multi_peak.py`

Triple top and triple bottom outputs. Main strategy-facing values are pattern presence, confirmation, indicator score, confirmation level, target level, and three pivot indices.

`pattern_wolfe_waves.py`

Bullish and bearish Wolfe wave outputs. Main strategy-facing values are pattern presence, confirmation, indicator score, confirmation level, and target level.

## Parked Modules

These files live under `work_in_progress/` and are not part of the active strategy-facing indicator set:

- `work_in_progress/external_orderbook_context_features.py`
- `work_in_progress/external_global_context_features.py`
- `work_in_progress/external_news_web_sentiment_features.py`

Do not import parked modules into strategies unless external-context research is explicitly reopened.
