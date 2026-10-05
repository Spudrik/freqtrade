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
RESEARCH_PATH = "sieve3_exit_wedge_invalidation"
EXIT_HYPOTHESIS = "Compare loss of the breakout rail with deeper failure of the lower wedge rail or signal swing."

# Preflight: wedge upper breakout; entry-frozen upper/lower rails and pre-entry swing low;
# source invalidation only, with a complementary 5% catastrophic stop. One plan selects one failure thesis.
ACTIVE_SELL_PARAMS = ("invalidation_plan",)
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


class Sieve3V2WedgeInvalidationFromCompletePatternGeometryWedgeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.05
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    

    invalidation_plan = CategoricalParameter(["upper_close1", "upper_close2", "upper_close1_band_0_25", "upper_close2_band_0_50", "lower_close1", "lower_close2", "signal_swing6_close1", "signal_swing12_close1", "signal_swing12_close2"], default="upper_close2", space="sell", optimize=True, load=True)
    invalidation_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    ACTIVE_SELL_PARAMS = ("invalidation_plan",)
    _STATE_KEY = "wedge_invalidation_entry_state"
    INVALIDATION_PLANS = {
        "upper_close1": ("upper", 1, 0.0, 0), "upper_close2": ("upper", 2, 0.0, 0),
        "upper_close1_band_0_25": ("upper", 1, 0.0025, 0), "upper_close2_band_0_50": ("upper", 2, 0.005, 0),
        "lower_close1": ("lower", 1, 0.0, 0), "lower_close2": ("lower", 2, 0.0, 0),
        "signal_swing6_close1": ("swing", 1, 0.0, 6), "signal_swing12_close1": ("swing", 1, 0.0, 12),
        "signal_swing12_close2": ("swing", 2, 0.0, 12),
    }

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

    def _state(self, pair: str, trade: Any, entry_order_time: datetime | None = None) -> dict[str, Any]:
        saved = trade.get_custom_data(key=self._STATE_KEY)
        if saved is not None:
            try:
                saved = dict(saved)
                required = {"plan", "level", "confirmations", "band", "entry_filled_at"}
                if required - set(saved):
                    raise ValueError
                plan = str(saved["plan"])
                level = float(saved["level"]); confirmations = int(saved["confirmations"]); band = float(saved["band"])
                filled_at = pd.Timestamp(saved["entry_filled_at"])
                expected = self.INVALIDATION_PLANS[plan]
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("restored wedge invalidation state is incompatible") from exc
            if (
                not np.isfinite([level, band]).all()
                or not 0.0 < level < float(trade.open_rate)
                or confirmations != expected[1]
                or not np.isclose(band, expected[2])
                or pd.isna(filled_at)
            ):
                raise RuntimeError("restored wedge invalidation state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("wedge invalidation state is missing after entry restoration")
        if self.dp is None:
            raise RuntimeError("wedge invalidation requires the analyzed dataframe")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "low", "close", "pg2_wedge_upper", "pg2_wedge_lower"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"wedge invalidation missing columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        opened = self._utc(entry_order_time)
        pre_entry = frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= opened].sort_values("date")
        if pre_entry.empty:
            raise RuntimeError("wedge invalidation cannot locate the entry signal candle")
        plan = str(self.invalidation_plan.value)
        source, confirmations, band, lookback = self.INVALIDATION_PLANS[plan]
        signal = pre_entry.iloc[-1]
        if source == "upper":
            level = float(signal["pg2_wedge_upper"])
        elif source == "lower":
            level = float(signal["pg2_wedge_lower"])
        else:
            level = float(pd.to_numeric(pre_entry["low"].tail(lookback), errors="coerce").min())
        if not np.isfinite(level) or not 0.0 < level < float(trade.open_rate):
            raise ValueError(f"{source} invalidation level must be below the long trade open rate")
        saved = {"plan": plan, "level": level, "confirmations": confirmations, "band": band, "entry_filled_at": opened.isoformat()}
        trade.set_custom_data(key=self._STATE_KEY, value=saved)
        return saved

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=self._STATE_KEY) is None:
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None:
                raise RuntimeError("wedge invalidation requires the actual entry fill timestamp")
            self._state(pair, trade, filled_at)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        state = self._state(pair, trade)
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        closed = frame.loc[dates.ge(self._utc(state["entry_filled_at"])) & (dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= self._utc(current_time))].sort_values("date")
        confirmations = int(state["confirmations"])
        if closed.empty or len(closed) < confirmations:
            return None
        breached = pd.to_numeric(closed["close"].tail(confirmations), errors="coerce").lt(float(state["level"]) * (1.0 - float(state["band"])))
        if len(breached) == confirmations and bool(breached.all()):
            return f"wedge_invalidation_{state['plan']}"
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
