"""Triangle lower-rail versus signal-candle-low invalidation for the long breakout."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_triangle_squeeze_breakout_long_1h"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = "complete_pattern_geometry_triangle_squeeze_breakout_long_1h"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
RESEARCH_PATH = "sieve3_exit_triangle_lower_boundary_signal_swing_invalidation"
EXIT_HYPOTHESIS = "Compare the frozen triangle lower boundary with the entry-signal swing low as invalidation for the long breakout."


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
    return frame[column].astype("boolean").fillna(False).astype(bool)


class Sieve3V2TriangleLowerBoundarySignalSwingInvalidationFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"; startup_candle_count = 180; process_only_new_candles = True; can_short = False
    position_adjustment_enable = False; max_entry_position_adjustment = 0; use_custom_stoploss = True; use_exit_signal = True
    exit_profit_only = False; ignore_roi_if_entry_signal = False; trailing_stop = False; minimal_roi = {"0": 100.0}; stoploss = -0.99

    
    
    
    
    
    
    
    
    
    
    
    
    

    invalidation_plan = CategoricalParameter(
        [
            "lower_intrabar", "signal_low_intrabar", "lower_close1", "lower_close2", "lower_close3",
            "signal_low_close1", "signal_low_close2", "signal_low_close3",
            "lower_bear_close1", "lower_bear_close2", "signal_low_bear_close1", "signal_low_bear_close2",
            "lower_close1_buffer025", "lower_close2_buffer025", "signal_low_close1_buffer025", "signal_low_close2_buffer025",
        ], default="signal_low_close1", space="sell", optimize=True, load=True,
    )
    invalidation_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ("invalidation_plan",)
    _GEOMETRY_KEY = "triangle_lower_signal_swing_entry_geometry"

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float: return 1.0
    def informative_pairs(self) -> list[tuple[str, str]]: return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(12), max_pattern_bars=int(72), compression_max_width_atr=float(2.0), squeeze_active_width_atr=float(2.0), min_line_score=float(0.5), min_containment=float(0.88), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool"); close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0); window = int(48); baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan); guard &= volume.ge(baseline.mul(float(1.6)))
        open_ = _num(dataframe, "open"); high = _num(dataframe, "high"); low = _num(dataframe, "low"); volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0); candle_range = (high - low).replace(0.0, np.nan); pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0).clip(-1.0, 1.0); directional_volume = (pressure * volume).fillna(0.0)
        window = int(12); denominator = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan); guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().div(denominator).ge(float(0.35))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0; dataframe["enter_short"] = 0; dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_triangle_pattern_present") & _bool(dataframe, "pg2_triangle_squeeze_active") & _num(dataframe, "pg2_triangle_indicator_score").ge(float(0.5)) & _num(dataframe, "pg2_triangle_direction").ge(0) & _num(dataframe, "close").gt(_num(dataframe, "pg2_triangle_upper", np.nan)); condition &= self._common_guards(dataframe); condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna(); dataframe.loc[valid, "enter_long"] = 1; dataframe.loc[valid, "enter_tag"] = ENTRY_TAG; return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None; return dataframe

    def _entry_geometry(
        self, pair: str, trade: Any, entry_order_time: datetime | None = None
    ) -> dict[str, Any]:
        saved = trade.get_custom_data(key=self._GEOMETRY_KEY)
        plans = {
            "lower_intrabar", "signal_low_intrabar", "lower_close1", "lower_close2", "lower_close3",
            "signal_low_close1", "signal_low_close2", "signal_low_close3",
            "lower_bear_close1", "lower_bear_close2", "signal_low_bear_close1", "signal_low_bear_close2",
            "lower_close1_buffer025", "lower_close2_buffer025", "signal_low_close1_buffer025", "signal_low_close2_buffer025",
        }
        if saved is not None:
            try:
                saved = dict(saved)
                required = {"upper", "lower", "signal_low", "plan", "entry_filled_at"}
                if required - set(saved):
                    raise ValueError
                upper = float(saved["upper"]); lower = float(saved["lower"]); signal_low = float(saved["signal_low"])
                filled_at = pd.Timestamp(saved["entry_filled_at"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("restored triangle invalidation state is incompatible") from exc
            if (
                str(saved["plan"]) not in plans
                or not np.isfinite([upper, lower, signal_low]).all()
                or not 0.0 < lower < upper
                or not 0.0 < lower < float(trade.open_rate)
                or not 0.0 < signal_low < float(trade.open_rate)
                or pd.isna(filled_at)
            ):
                raise RuntimeError("restored triangle invalidation state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("triangle invalidation state is missing after entry restoration")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe); dates = pd.to_datetime(frame["date"], utc=True); opened = pd.Timestamp(entry_order_time); opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        rows = frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= opened].sort_values("date")
        if rows.empty: raise RuntimeError("triangle invalidation cannot locate the closed entry signal candle")
        row = rows.iloc[-1]
        upper = float(row["pg2_triangle_upper"]); lower = float(row["pg2_triangle_lower"]); signal_low = float(row["low"])
        if not np.isfinite([upper, lower, signal_low]).all() or not 0.0 < lower < upper or lower >= float(trade.open_rate) or not 0.0 < signal_low < float(trade.open_rate): raise ValueError("entry triangle/signal levels are unavailable or incoherent")
        saved = {"upper": upper, "lower": lower, "signal_low": signal_low, "plan": str(self.invalidation_plan.value), "entry_filled_at": opened.isoformat()}; trade.set_custom_data(key=self._GEOMETRY_KEY, value=saved); return saved

    def _closed(self, pair: str, current_time: datetime, entry_filled_at: Any) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe); dates = pd.to_datetime(frame["date"], utc=True); cutoff = pd.Timestamp(current_time); cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        filled = pd.Timestamp(entry_filled_at); filled = filled.tz_localize("UTC") if filled.tzinfo is None else filled.tz_convert("UTC")
        return frame.loc[dates.ge(filled) & (dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= cutoff)].sort_values("date")

    def _selected_level(self, geometry: dict[str, float], plan: str) -> float:
        return geometry["signal_low"] if plan.startswith("signal_low") else geometry["lower"]

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        geometry = self._entry_geometry(pair, trade); plan = str(geometry["plan"])
        if plan.endswith("intrabar"): return None
        count = 3 if "close3" in plan else 2 if "close2" in plan else 1; level = self._selected_level(geometry, plan); level *= 0.9975 if plan.endswith("buffer025") else 1.0; frame = self._closed(pair, current_time, geometry["entry_filled_at"])
        if len(frame) < count: return None
        failed = _num(frame, "close").tail(count).lt(level)
        if "bear_close" in plan: failed &= _num(frame, "close").tail(count).lt(_num(frame, "open").tail(count))
        return "triangle_lower_or_signal_swing_failed" if bool(failed.all()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        geometry = self._entry_geometry(pair, trade); plan = str(geometry["plan"]); stop_price = self._selected_level(geometry, plan) if plan.endswith("intrabar") else min(geometry["lower"], geometry["signal_low"]) * 0.995
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=self._GEOMETRY_KEY) is None:
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None: raise RuntimeError("triangle invalidation requires the actual entry fill timestamp")
            self._entry_geometry(pair, trade, filled_at)
