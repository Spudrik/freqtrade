"""Frozen triangle-height measured-move targets for the 1h long breakout."""
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
RESEARCH_PATH = "sieve3_exit_triangle_measured_move_target"
EXIT_HYPOTHESIS = "Test entry-frozen triangle-height measured-move targets and target-zone confirmation for the long breakout."
TARGET_PLANS = (
    "mm50_touch", "mm75_touch", "mm100_touch", "mm125_touch", "mm150_touch",
    "mm50_zone025", "mm75_zone025", "mm100_zone025", "mm125_zone025",
    "mm75_zone050", "mm100_zone050", "mm125_zone050", "mm150_zone050",
    "mm75_bear_confirm", "mm100_bear_confirm", "mm125_bear_confirm", "mm150_bear_confirm",
)


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


class Sieve3V2TriangleMeasuredMoveTargetFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"; startup_candle_count = 180; process_only_new_candles = True; can_short = False
    position_adjustment_enable = False; max_entry_position_adjustment = 0
    use_custom_stoploss = True; use_exit_signal = True; exit_profit_only = False; ignore_roi_if_entry_signal = False; trailing_stop = False
    minimal_roi = {"0": 100.0}; stoploss = -0.99

    
    
    
    
    
    
    
    
    
    
    
    
    

    target_plan = CategoricalParameter(
        list(TARGET_PLANS), default="mm100_touch", space="sell", optimize=True, load=True,
    )
    target_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ("target_plan",)
    _GEOMETRY_KEY = "triangle_measured_move_entry_geometry"

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
        self,
        pair: str,
        trade: Any,
        entry_order_time: datetime | None = None,
    ) -> dict[str, float]:
        saved = trade.get_custom_data(key=self._GEOMETRY_KEY)
        if saved is not None:
            try:
                saved = dict(saved)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("restored triangle measured-move state is incompatible") from exc
            required = {"upper", "lower", "height", "plan", "entry_filled_at"}
            if required - set(saved):
                raise RuntimeError("restored triangle measured-move state is incomplete")
            try:
                upper = float(saved["upper"]); lower = float(saved["lower"]); height = float(saved["height"])
                filled_at = pd.Timestamp(saved["entry_filled_at"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("restored triangle measured-move state is incompatible") from exc
            if (
                str(saved["plan"]) not in TARGET_PLANS
                or not np.isfinite([upper, lower, height]).all()
                or not 0.0 < lower < upper
                or not np.isclose(height, upper - lower)
                or pd.isna(filled_at)
            ):
                raise RuntimeError("restored triangle measured-move state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("triangle measured-move state is missing after entry restoration")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True)
        opened = pd.Timestamp(entry_order_time)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        row = frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= opened].sort_values("date").iloc[-1]
        upper = float(row["pg2_triangle_upper"]); lower = float(row["pg2_triangle_lower"])
        if not np.isfinite([upper, lower]).all() or not 0.0 < lower < upper or lower >= float(trade.open_rate): raise ValueError("entry triangle rails are unavailable or incoherent")
        saved = {"upper": upper, "lower": lower, "height": upper - lower, "plan": str(self.target_plan.value), "entry_filled_at": opened.isoformat()}; trade.set_custom_data(key=self._GEOMETRY_KEY, value=saved); return saved

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = current_time, kwargs
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None)
            and trade.get_custom_data(key=self._GEOMETRY_KEY) is None
        ):
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None: raise RuntimeError("triangle measured-move exit requires the actual entry fill timestamp")
            self._entry_geometry(pair, trade, filled_at)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        geometry = self._entry_geometry(pair, trade); plan = str(geometry["plan"]); multiplier = float(plan.split("_", 1)[0].removeprefix("mm")) / 100.0; target = geometry["upper"] + geometry["height"] * multiplier
        if target <= float(trade.open_rate):
            return None
        zone = 0.0025 if "zone025" in plan else 0.005 if "zone050" in plan else 0.0
        trigger_rate = max(target * (1.0 - zone), float(trade.open_rate))
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        filled_at = pd.Timestamp(geometry["entry_filled_at"])
        filled_at = filled_at.tz_localize("UTC") if filled_at.tzinfo is None else filled_at.tz_convert("UTC")
        closed = frame.loc[dates.ge(filled_at) & (dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= cutoff)].sort_values("date")
        reached = not closed.empty and bool(pd.to_numeric(closed["high"], errors="coerce").ge(trigger_rate).any())
        if reached and "bear_confirm" in plan:
            row = closed.iloc[-1]
            reached = bool(float(row["close"]) < float(row["open"]))
        return "triangle_measured_move_target" if reached else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        stop_price = self._entry_geometry(pair, trade)["lower"]
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))
