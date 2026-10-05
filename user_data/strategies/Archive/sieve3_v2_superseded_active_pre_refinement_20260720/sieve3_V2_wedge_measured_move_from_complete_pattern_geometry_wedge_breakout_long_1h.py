from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_wedge_breakout_long_1h"
ENTRY_TAG = "geometry_wedge_breakout_long_1h"
SIDE = "long"
TIMEFRAME = "1h"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_ENTRY_STAGE = "sieve2"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:523"
SOURCE_STRATEGY = "D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\sieve2_geometry_wedge_breakout_long_1h__pattern_geometry_1h_2023_24_compression\\full_cycle_2020_2026\\tp_4_sl_2\\backtest-result-2026-05-26_02-54-35.zip!backtest-result-2026-05-26_02-54-35_Sieve2GeometryWedgeBreakoutLong1H.py:Sieve2GeometryWedgeBreakoutLong1H"
RESEARCH_PATH = "sieve3_exit_wedge_measured_move"
EXIT_HYPOTHESIS = "Use the entry-signal wedge height above its upper rail as the breakout's source-defined upside objective."

# Preflight: wedge upper breakout; frozen signal-candle upper/lower rails; wedge-height target;
# fixed 3% invalidation. The plan binds projection multiple and early-touch offset.
ACTIVE_SELL_PARAMS = ("target_plan",)
BRANCH_LOCAL_PARAMS: tuple[str, ...] = ()




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


class Sieve3V2WedgeMeasuredMoveFromCompletePatternGeometryWedgeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    

    target_plan = CategoricalParameter(["height_0_50", "height_0_75", "height_1_00", "height_1_25", "height_1_50", "height_0_75_early_0_5", "height_1_00_early_0_5", "height_1_00_early_1_0"], default="height_1_00", space="sell", optimize=True, load=True)
    target_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    ACTIVE_SELL_PARAMS = ("target_plan",)
    TARGET_PLANS = {
        "height_0_50": (0.50, 0.0), "height_0_75": (0.75, 0.0), "height_1_00": (1.00, 0.0),
        "height_1_25": (1.25, 0.0), "height_1_50": (1.50, 0.0), "height_0_75_early_0_5": (0.75, 0.005),
        "height_1_00_early_0_5": (1.00, 0.005), "height_1_00_early_1_0": (1.00, 0.01),
    }
    _STATE_KEY = "wedge_measured_move_entry_state"

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=True, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(12), max_pattern_bars=int(72), compression_max_width_atr=float(2.0), squeeze_active_width_atr=float(2.0), min_line_score=float(0.55), min_containment=float(0.88), output_prefix="pg2")
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
        directional_volume = (((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0) * volume).fillna(0.0)
        window = int(48)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        guard &= (directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline).ge(float(0.15))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_wedge_pattern_present") & _bool(dataframe, "pg2_wedge_squeeze_active")
        condition &= _num(dataframe, "pg2_wedge_indicator_score").ge(float(0.55))
        condition &= _num(dataframe, "pg2_wedge_direction").ge(0)
        condition &= _num(dataframe, "close").gt(_num(dataframe, "pg2_wedge_upper"))
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
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

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def _entry_signal_row(self, pair: str, entry_order_time: datetime) -> Series:
        if self.dp is None:
            raise RuntimeError("measured-move exit requires the analyzed dataframe")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "pg2_wedge_upper", "pg2_wedge_lower"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"measured-move exit missing columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        opened = self._utc(entry_order_time)
        candidates = frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= opened]
        if candidates.empty:
            raise RuntimeError("measured-move exit cannot locate the entry signal candle")
        return candidates.iloc[-1]

    def _state(self, pair: str, trade: Any, entry_order_time: datetime | None = None) -> dict[str, Any]:
        saved = trade.get_custom_data(key=self._STATE_KEY)
        if saved is not None:
            try:
                saved = dict(saved)
                required = {"plan", "target", "entry_filled_at"}
                if required - set(saved):
                    raise ValueError
                plan = str(saved["plan"]); target = float(saved["target"])
                filled_at = pd.Timestamp(saved["entry_filled_at"])
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("restored wedge measured-move state is incompatible") from exc
            if plan not in self.TARGET_PLANS or not np.isfinite(target) or target <= float(trade.open_rate) or pd.isna(filled_at):
                raise RuntimeError("restored wedge measured-move state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("wedge measured-move state is missing after entry restoration")
        filled_at = self._utc(entry_order_time)
        row = self._entry_signal_row(pair, filled_at)
        upper = float(row["pg2_wedge_upper"])
        lower = float(row["pg2_wedge_lower"])
        if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
            raise ValueError("entry wedge rails do not define a positive measured height")
        plan = str(self.target_plan.value)
        multiple, early = self.TARGET_PLANS[plan]
        target = (upper + (upper - lower) * multiple) * (1.0 - early)
        if target <= float(trade.open_rate):
            raise ValueError("entry wedge measured target is not above the trade open rate")
        saved = {"plan": plan, "target": target, "entry_filled_at": filled_at.isoformat()}
        trade.set_custom_data(key=self._STATE_KEY, value=saved)
        return saved

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=self._STATE_KEY) is None:
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None:
                raise RuntimeError("wedge measured-move exit requires the actual entry fill timestamp")
            self._state(pair, trade, filled_at)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        state = self._state(pair, trade)
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        closed = frame.loc[dates.ge(self._utc(state["entry_filled_at"])) & (dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= self._utc(current_time))]
        if not closed.empty and bool(pd.to_numeric(closed["high"], errors="coerce").ge(float(state["target"])).any()):
            return f"wedge_measured_move_{state['plan']}"
        return None


LOCKED_BUY_PARAMS = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context", "sieve2_vp_window": 96, "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28, "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24, "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25, "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True, "volume_guard_window": 24, "volume_ratio_min": 1.6, "use_pressure_guard": True,
    "pressure_window": 48, "pressure_min": 0.15, "use_accumulation_guard": False, "use_body_direction_guard": False,
    "use_close_direction_guard": False, "min_pattern_bars": 12, "max_pattern_bars": 72, "compression_max_width_atr": 2.0,
    "squeeze_active_width_atr": 2.0, "min_line_score": 0.55, "min_containment": 0.88,
}
