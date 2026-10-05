from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from freqtrade.exchange import timeframe_to_minutes
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2






def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)




PRIMARY_GUARD = 'locked volume/pressure guards; optional Sieve2 VP and market guards are frozen off'

PRIMARY_INVALIDATION = 'entry-frozen triangle lower-rail reclaim; fixed 3% adverse stop'
ENTRY_INVALIDATION_STATE_KEY = "s3v2_triangle_short_lower_boundary"








EXIT_FAMILY = 'reclaimed_lower_boundary_invalidation'
PRIMARY_TRIGGER = '4h close below the emitted triangle lower rail while squeeze, score, and non-positive direction hold'
PRIMARY_GUARD = 'locked volume/pressure guards; optional Sieve2 VP and market guards are frozen off'
PRIMARY_TARGET = 'none'
PRIMARY_INVALIDATION = 'entry-frozen triangle lower-rail reclaim; fixed 3% adverse stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'Sieve2GeometryTriangleSqueezeBreakdownShort4H'
SOURCE_RESULT_BATCH = '20260521T012740_entry_all_resume.jsonl:242,380,446'
RESEARCH_PATH = 'sieve3_exit_reclaimed_lower_boundary_invalidation'
EXIT_HYPOTHESIS = 'A confirmed reclaim of the exact broken lower rail invalidates the breakdown before a generic stop.'
ACTIVE_SELL_PARAMS = ('exit_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_triangle_squeeze_breakdown_short_4h'

class Sieve3V2ReclaimedLowerBoundaryInvalidationFromCompletePatternGeometryTriangleSqueezeBreakdownShort4H(IStrategy):
    """Invalidate the short when price closes back above the frozen broken lower rail."""

    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True

    SIEVE_STAGE = "sieve3"
    ENTRY_SOURCE_STAGE = 'sieve2'
    SOURCE_STRATEGY = "Sieve2GeometryTriangleSqueezeBreakdownShort4H"
    SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume.jsonl:242,380,446"
    PRIMARY_TRIGGER = "4h close below the emitted triangle lower rail while squeeze, score, and non-positive direction hold"
    PRIMARY_GUARD = "locked volume/pressure guards; optional Sieve2 VP and market guard selectors are frozen off"
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    RESEARCH_PATH = 'sieve3_exit_reclaimed_lower_boundary_invalidation'
    EXIT_HYPOTHESIS = "A confirmed reclaim of the exact broken lower rail invalidates the breakdown before a generic stop."
    LOCK_STATUS = "verified promoted params plus tested-snapshot defaults"

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.05
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.4
    min_containment = 0.9

    exit_plan = CategoricalParameter(
        ["one_close", "two_closes", "one_close_buffer_0_5", "two_closes_buffer_0_5"],
        default="two_closes",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        missing = sorted(set(("pg2_triangle_lower",)).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_triangle_pattern_present") & _bool(dataframe, "pg2_triangle_squeeze_active") & _num(dataframe, "pg2_triangle_indicator_score").ge(float(self.min_line_score)) & _num(dataframe, "pg2_triangle_direction").le(0) & _num(dataframe, "close").lt(_num(dataframe, "pg2_triangle_lower", np.nan))
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _freeze_entry_lower(self, pair: str, trade: Any, cutoff_time: datetime) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("triangle invalidation requires an analyzed entry dataframe")
        if "date" not in dataframe.columns or "pg2_triangle_lower" not in dataframe.columns:
            raise KeyError("triangle invalidation requires date and pg2_triangle_lower columns")
        opened = pd.Timestamp(cutoff_time)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        rows = dataframe.loc[close_times.le(opened)].sort_values("date")
        if rows.empty:
            raise RuntimeError("entry signal candle is unavailable for triangle invalidation")
        value = float(rows.iloc[-1].get("pg2_triangle_lower", np.nan))
        if not np.isfinite(value) or value <= 0.0 or value <= float(trade.open_rate):
            raise RuntimeError("entry signal has no usable triangle lower-boundary invalidation level")
        trade.set_custom_data(key=ENTRY_INVALIDATION_STATE_KEY, value={"lower": value})
        return value

    def _entry_lower(self, pair: str, trade: Any) -> float:
        state = trade.get_custom_data(key=ENTRY_INVALIDATION_STATE_KEY)
        if state is None:
            cutoff = getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
            return self._freeze_entry_lower(pair, trade, cutoff)
        if not isinstance(state, dict):
            raise RuntimeError("triangle invalidation entry snapshot is invalid")
        value = float(state["lower"])
        if not np.isfinite(value) or value <= 0.0 or value <= float(trade.open_rate):
            raise RuntimeError("persisted triangle lower-boundary invalidation level is unusable")
        return value

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=ENTRY_INVALIDATION_STATE_KEY) is None:
            self._freeze_entry_lower(pair, trade, current_time)

    def _closed_trade_candles(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return DataFrame()
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        opened = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return dataframe.loc[dates.ge(opened) & close_times.le(cutoff)].sort_values("date")

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        stop_price = float(trade.open_rate) * 1.03
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = (current_rate, current_profit, kwargs)
        lower = self._entry_lower(pair, trade)
        dataframe = self._closed_trade_candles(pair, trade, current_time)
        if dataframe.empty:
            return None
        plan = str(self.exit_plan.value)
        buffer = 0.005 if "buffer_0_5" in plan else 0.0
        threshold = lower * (1.0 + buffer)
        one = float(dataframe.iloc[-1]["close"]) > threshold
        confirmed = one and ("two_closes" not in plan or (len(dataframe) >= 2 and float(dataframe.iloc[-2]["close"]) > threshold))
        return f"lower_reclaim_{plan}" if confirmed else None
