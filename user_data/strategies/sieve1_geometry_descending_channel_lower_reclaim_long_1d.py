from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

_INDICATOR_DIR = Path(__file__).resolve().parents[1] / "Indicators"
if str(_INDICATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_INDICATOR_DIR))

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
ENTRY_MODE = "entry_geometry_descending_channel_lower_reclaim_long_1d"
ENTRY_TAG = "geometry_descending_channel_lower_reclaim_long_1d"
SIDE = "long"
TIMEFRAME = "1d"


def split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:
    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    if not selected:
        return
    for name in dir(strategy_cls):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = str(name) in selected


def _pct_env(name: str, default_ratio: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default_ratio)
    try:
        return max(0.0, float(raw)) / 100.0
    except ValueError:
        return float(default_ratio)


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
    return param


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))

class Sieve1GeometryDescendingChannelLowerReclaimLong1D(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, 0.02)}
    stoploss = -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_volume_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    volume_guard_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=1.0, space="buy", optimize=True, load=True))
    use_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    pressure_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(CategoricalParameter([0.05, 0.1, 0.15, 0.2, 0.35], default=0.15, space="buy", optimize=True, load=True))
    use_accumulation_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    use_body_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    use_close_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))

    min_pattern_bars = tagged_parameter(IntParameter(6, 24, default=12, space="buy", optimize=False, load=True))
    max_pattern_bars = tagged_parameter(IntParameter(36, 120, default=72, space="buy", optimize=False, load=True))
    compression_max_width_atr = tagged_parameter(DecimalParameter(1.20, 3.50, decimals=2, default=2.00, space="buy", optimize=False, load=True))
    squeeze_active_width_atr = tagged_parameter(DecimalParameter(1.00, 3.50, decimals=2, default=2.00, space="buy", optimize=False, load=True))
    local_narrowing_min_ratio = tagged_parameter(DecimalParameter(0.03, 0.25, decimals=2, default=0.10, space="buy", optimize=False, load=True))
    min_line_score = tagged_parameter(CategoricalParameter([0.4, 0.5, 0.55, 0.7], default=0.5, space="buy", optimize=True, load=True))
    min_containment = tagged_parameter(CategoricalParameter([0.7, 0.8, 0.88, 0.9], default=0.88, space="buy", optimize=True, load=True))
    max_recent_touch_age_bars = tagged_parameter(IntParameter(4, 24, default=12, space="buy", optimize=False, load=True))
    channel_min_pattern_bars = tagged_parameter(IntParameter(8, 36, default=18, space="buy", optimize=False, load=True))
    channel_max_pattern_bars = tagged_parameter(IntParameter(48, 180, default=96, space="buy", optimize=False, load=True))
    channel_min_quality = tagged_parameter(DecimalParameter(0.65, 0.95, decimals=2, default=0.82, space="buy", optimize=False, load=True))
    channel_min_containment = tagged_parameter(DecimalParameter(0.55, 0.90, decimals=2, default=0.68, space="buy", optimize=False, load=True))
    channel_near_boundary_atr_mult = tagged_parameter(DecimalParameter(0.30, 1.20, decimals=2, default=0.70, space="buy", optimize=False, load=True))
    channel_breakout_atr_mult = tagged_parameter(DecimalParameter(0.15, 0.80, decimals=2, default=0.35, space="buy", optimize=False, load=True))
    channel_lifecycle_confirm_break_bars = tagged_parameter(IntParameter(1, 4, default=2, space="buy", optimize=False, load=True))
    score_min = tagged_parameter(CategoricalParameter([0.55, 0.65, 0.72, 0.82], default=0.65, space="buy", optimize=False, load=True))
    width_atr_max = tagged_parameter(DecimalParameter(0.75, 10.00, decimals=2, default=6.00, space="buy", optimize=False, load=True))
    rail_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.004, 0.005, 0.01, 0.02], default=0.004, space="buy", optimize=False, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(
            dataframe,
            timeframe=self.timeframe,
            output_slots=1,
            include_triangle_patterns=False,
            include_wedge_patterns=False,
            include_compression_patterns=False,
            include_rectangle_patterns=False,
            include_ascending_channel_patterns=False,
            include_descending_channel_patterns=True,
            min_pattern_bars=int(self.min_pattern_bars.value),
            max_pattern_bars=int(self.max_pattern_bars.value),
            compression_max_width_atr=float(self.compression_max_width_atr.value),
            squeeze_active_width_atr=float(self.squeeze_active_width_atr.value),
            local_narrowing_min_ratio=float(self.local_narrowing_min_ratio.value),
            min_line_score=float(self.min_line_score.value),
            min_containment=float(self.min_containment.value),
            max_recent_touch_age_bars=int(self.max_recent_touch_age_bars.value),
            channel_min_pattern_bars=int(self.channel_min_pattern_bars.value),
            channel_max_pattern_bars=int(self.channel_max_pattern_bars.value),
            channel_min_quality=float(self.channel_min_quality.value),
            channel_min_containment=float(self.channel_min_containment.value),
            channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult.value),
            channel_breakout_atr_mult=float(self.channel_breakout_atr_mult.value),
            channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars.value),
            output_prefix="pg2",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard.value):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window.value)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min.value)))
        if bool(self.use_pressure_guard.value) or bool(self.use_accumulation_guard.value):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard.value):
            window = int(self.pressure_window.value)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure_ratio.ge(float(self.pressure_min.value))
        if bool(self.use_accumulation_guard.value):
            window = int(self.pressure_window.value)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.le(0.0) if SIDE == "short" else accumulation.ge(0.0)
        if bool(self.use_body_direction_guard.value):
            open_ = _num(dataframe, "open")
            guard &= close.lt(open_) if SIDE == "short" else close.gt(open_)
        if bool(self.use_close_direction_guard.value):
            guard &= close.lt(close.shift(1)) if SIDE == "short" else close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_descending_channel_lower", np.nan)
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min.value)) & _num(dataframe, "pg2_descending_channel_width_atr", np.nan).le(float(self.width_atr_max.value)) & _num(dataframe, "low").le(lower.mul(1.0 + float(self.rail_buffer_pct.value))) & _num(dataframe, "close").gt(lower) & _num(dataframe, "close").gt(_num(dataframe, "open"))
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(Sieve1GeometryDescendingChannelLowerReclaimLong1D)
