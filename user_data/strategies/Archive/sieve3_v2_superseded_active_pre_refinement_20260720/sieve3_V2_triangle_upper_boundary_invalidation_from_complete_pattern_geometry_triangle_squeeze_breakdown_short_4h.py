from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from freqtrade.exchange import timeframe_to_minutes
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_triangle_squeeze_breakdown_short_4h"
SIDE = "short"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = "Sieve2GeometryTriangleSqueezeBreakdownShort4H"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume.jsonl:242,380,446"
RESEARCH_PATH = "sieve3_exit_triangle_upper_boundary_invalidation"
EXIT_HYPOTHESIS = "Compare the frozen triangle upper rail with the entry-signal high as invalidation for the short breakdown."
ACTIVE_SELL_PARAMS = ("exit_plan",)


def _num(
    frame: DataFrame, column: str, default: float | Series | object = ...
) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f"Required dataframe column not found: {column}")
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"Required dataframe column not found: {column}")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


class Sieve3V2TriangleUpperBoundaryInvalidationFromCompletePatternGeometryTriangleSqueezeBreakdownShort4H(IStrategy):
    """Compare frozen triangle upper-rail and entry signal-swing invalidation."""

    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True

    SIEVE_STAGE = "sieve3"
    ENTRY_SOURCE_STAGE = "sieve2"
    SOURCE_STRATEGY = "Sieve2GeometryTriangleSqueezeBreakdownShort4H"
    SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume.jsonl:242,380,446"
    PRIMARY_TRIGGER = "4h close below the emitted triangle lower rail while squeeze, score, and non-positive direction hold"
    PRIMARY_GUARD = "locked volume/pressure guards; optional Sieve2 VP and market guard selectors are frozen off"
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    RESEARCH_PATH = "sieve3_exit_triangle_upper_boundary_invalidation"
    EXIT_HYPOTHESIS = "The frozen upper rail or entry-candle high is the hard structural failure level for the short thesis."
    LOCK_STATUS = "verified promoted params plus tested-snapshot defaults"

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    

    exit_plan = CategoricalParameter(
        ["upper_rail_close", "upper_rail_high", "signal_high_close", "signal_high_high"],
        default="upper_rail_close",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    _STATE_KEY = "triangle_upper_boundary_entry_state"

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(12), max_pattern_bars=int(72), compression_max_width_atr=float(2.0), squeeze_active_width_atr=float(2.0), min_line_score=float(0.4), min_containment=float(0.9), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(24)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(1.6)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        window = int(24)
        pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
        guard &= pressure_ratio.le(-float(0.05))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_triangle_pattern_present") & _bool(dataframe, "pg2_triangle_squeeze_active") & _num(dataframe, "pg2_triangle_indicator_score").ge(float(0.4)) & _num(dataframe, "pg2_triangle_direction").le(0) & _num(dataframe, "close").lt(_num(dataframe, "pg2_triangle_lower", np.nan))
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
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

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def _entry_levels(self, pair: str, trade: Any, entry_order_time: datetime | None = None) -> dict[str, Any] | None:
        saved = trade.get_custom_data(key=self._STATE_KEY)
        if saved is not None:
            try:
                saved = dict(saved)
                required = {"upper", "signal_high", "plan", "entry_filled_at"}
                if required - set(saved):
                    raise ValueError
                upper = float(saved["upper"]); signal_high = float(saved["signal_high"])
                filled_at = pd.Timestamp(saved["entry_filled_at"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("restored triangle invalidation state is incompatible") from exc
            if (
                str(saved["plan"]) not in {"upper_rail_close", "upper_rail_high", "signal_high_close", "signal_high_high"}
                or not np.isfinite([upper, signal_high]).all()
                or not float(trade.open_rate) < upper
                or not float(trade.open_rate) < signal_high
                or pd.isna(filled_at)
            ):
                raise RuntimeError("restored triangle invalidation state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("triangle invalidation state is missing after entry restoration")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        filled_at = self._utc(entry_order_time)
        closes = pd.to_datetime(dataframe["date"], utc=True) + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        rows = dataframe.loc[closes.le(filled_at)].sort_values("date")
        if rows.empty:
            raise RuntimeError("triangle invalidation cannot locate the closed entry signal candle")
        row = rows.iloc[-1]
        upper = float(row["pg2_triangle_upper"]); signal_high = float(row["high"])
        if not np.isfinite([upper, signal_high]).all() or not float(trade.open_rate) < upper or not float(trade.open_rate) < signal_high:
            raise ValueError("entry triangle invalidation levels must be above the short trade open rate")
        saved = {"upper": upper, "signal_high": signal_high, "plan": str(self.exit_plan.value), "entry_filled_at": filled_at.isoformat()}
        trade.set_custom_data(key=self._STATE_KEY, value=saved)
        return saved

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=self._STATE_KEY) is None:
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None:
                raise RuntimeError("triangle invalidation requires the actual entry fill timestamp")
            self._entry_levels(pair, trade, filled_at)

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool = False, **kwargs: Any) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        stop_price = float(trade.open_rate) * 1.06
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = (current_rate, current_profit, kwargs)
        state = self._entry_levels(pair, trade)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if state is None or dataframe is None or dataframe.empty:
            return None
        dates = pd.to_datetime(dataframe["date"], utc=True)
        closes = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = dataframe.loc[dates.ge(self._utc(state["entry_filled_at"])) & closes.le(self._utc(current_time))].sort_values("date")
        if closed.empty:
            return None
        plan = str(state["plan"]); last = closed.iloc[-1]
        level = float(state["upper"] if plan.startswith("upper_rail") else state["signal_high"])
        observed = float(last["high"] if plan.endswith("_high") else last["close"])
        return f"structural_failure_{plan}" if observed > level else None
