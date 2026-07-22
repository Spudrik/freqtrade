from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def add_market_state(
    dataframe: DataFrame,
    window: int = 24,
    prefix: str = "s2m",
) -> DataFrame:
    """Append volume-pressure, volatility-scaled trend, and close-location evidence."""

    _validate_inputs(dataframe, window, prefix)
    result = dataframe.copy()

    close = _num(result["close"])
    open_ = _num(result["open"])
    high = _num(result["high"])
    low = _num(result["low"])
    volume = _num(result["volume"]).clip(lower=0.0).fillna(0.0)

    candle_range = (high - low).replace(0.0, np.nan)
    body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
    close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
    directional_volume = (pressure * volume).fillna(0.0)

    pressure_min_periods = max(2, window // 3)
    baseline = volume.rolling(window, min_periods=pressure_min_periods).sum().replace(0.0, np.nan)
    returns = _finite(close.pct_change(fill_method=None))
    window_return = _finite(close.pct_change(window, fill_method=None))
    volatility = _finite(
        returns.rolling(window, min_periods=max(3, window // 3)).std()
        * np.sqrt(float(window))
    )

    result[f"{prefix}_pressure_ratio"] = _finite(
        directional_volume.rolling(window, min_periods=pressure_min_periods).sum() / baseline
    )
    result[f"{prefix}_trend_z"] = _finite(
        window_return / volatility.replace(0.0, np.nan)
    )
    result[f"{prefix}_close_location"] = _finite(close_location)
    return result


def _validate_inputs(dataframe: DataFrame, window: int, prefix: str) -> None:
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"DataFrame missing required OHLCV columns: {', '.join(missing)}")
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer greater than or equal to 3")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")


def _num(series: Series) -> Series:
    return _finite(pd.to_numeric(series, errors="coerce"))


def _finite(series: Series) -> Series:
    return series.replace([np.inf, -np.inf], np.nan)


__all__ = ["add_market_state"]
