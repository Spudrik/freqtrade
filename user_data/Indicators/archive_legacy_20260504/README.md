# Legacy Indicator Archive

These modules are archived prototype code:

- `complex_line_engine.py`
- `complex_pattern_structure.py`
- `complex_structural_trendlines.py`
- `complex_trendline_projection.py`
- `complex_trendline_channels.py`
- `complex_volatility_cycles.py`
- `complex_volume_indicators.py`
- `simple_confluence_indicator.py`

They are superseded by the active foundation stack:

- `pivot_foundation.py`
- `complex_pivot_structure.py`
- `complex_trendline_projection_v2.py`
- `complex_pattern_structure.py`

Do not import archived modules into new strategies or indicators. Old strategies that still import these files should be treated as prototype strategies requiring retuning against the active stack.

`complex_volatility_cycles.py` and `complex_volume_indicators.py` were archived later because they mostly duplicate standard ATR/range/relative-volume ideas or OHLCV proxy orderflow. Future work should prefer genuinely different structure/profile/pattern/order-book evidence.

`complex_trendline_channels.py` was archived because channel-like evidence is now treated as part of pattern detection. Reintroduce it only if the pattern layer proves it needs a shared standalone channel foundation.

`market_regime.py` was removed rather than archived because it mostly repackaged standard EMA/ADX/ATR/range/volume state. Future regime work should start from a clean premise.
