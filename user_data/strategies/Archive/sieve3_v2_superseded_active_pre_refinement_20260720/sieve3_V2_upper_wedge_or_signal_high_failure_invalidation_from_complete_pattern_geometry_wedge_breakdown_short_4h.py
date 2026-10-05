from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_wedge_breakdown_short_4h"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = "sieve2_complete_pattern_geometry_wedge_breakdown_short_4h"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
RESEARCH_PATH = "sieve3_exit_upper_wedge_or_signal_high_failure_invalidation"
EXIT_HYPOTHESIS = "Compare reclaim of the frozen upper wedge rail with one- or two-close failure above the entry-signal high for the short breakdown."
ACTIVE_SELL_PARAMS = ("failure_plan",)


class Sieve3V2UpperWedgeOrSignalHighFailureInvalidationFromCompletePatternGeometryWedgeBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    trailing_stop = False
    position_adjustment_enable = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.03

    # Legacy fixed-off entry parameter: use_sieve2_vp_guard=False (BooleanParameter, buy space).
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.25
    sieve2_vp_context_min = 0.28
    # Legacy fixed-off entry parameter: use_sieve2_market_guard=False (BooleanParameter, buy space).
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.07
    sieve2_market_trend_min = 0.25
    sieve2_rs_benchmark_pair = 'BTC/USDT:USDT'
    sieve2_rs_score_min = 0.45
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.6
    # Legacy fixed-off entry parameter: use_pressure_guard=False (BooleanParameter, buy space).
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.55
    min_containment = 0.88

    failure_plan = CategoricalParameter(
        ["upper_wedge_close", "signal_high_close", "signal_high_two_closes"],
        default="upper_wedge_close", space="sell", optimize=True, load=True,
    )
    failure_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    ACTIVE_SELL_PARAMS = ("failure_plan",)
    _STATE_KEY = "upper_wedge_failure_entry_state"

    @staticmethod
    def _num(frame: DataFrame, column: str) -> Series:
        return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _flag(frame: DataFrame, column: str) -> Series:
        return frame[column].astype("boolean").fillna(False).astype(bool)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return add_pattern_geometry_v2(
            dataframe, timeframe=self.timeframe, output_slots=1,
            include_triangle_patterns=False, include_wedge_patterns=True,
            include_compression_patterns=False, include_rectangle_patterns=False,
            include_ascending_channel_patterns=False, include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2",
        )

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        volume = self._num(dataframe, 'volume').clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = (
            self._flag(dataframe, "pg2_wedge_pattern_present")
            & self._flag(dataframe, "pg2_wedge_squeeze_active")
            & self._num(dataframe, "pg2_wedge_indicator_score").ge(float(self.min_line_score))
            & self._num(dataframe, "pg2_wedge_direction").le(0)
            & self._num(dataframe, "close").lt(self._num(dataframe, "pg2_wedge_lower"))
        )
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = "geometry_wedge_breakdown_short_4h"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def _entry_state(self, pair: str, trade: Any, entry_order_time: datetime | None = None) -> dict[str, Any] | None:
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
                raise RuntimeError("restored wedge failure state is incompatible") from exc
            if (
                str(saved["plan"]) not in {"upper_wedge_close", "signal_high_close", "signal_high_two_closes"}
                or not np.isfinite([upper, signal_high]).all()
                or not float(trade.open_rate) < upper
                or not float(trade.open_rate) < signal_high
                or pd.isna(filled_at)
            ):
                raise RuntimeError("restored wedge failure state is incompatible")
            return saved
        if entry_order_time is None:
            raise RuntimeError("wedge failure state is missing after entry restoration")
        if not self.dp:
            raise RuntimeError("wedge failure exit requires the analyzed dataframe")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("wedge failure exit requires a non-empty analyzed dataframe")
        filled_at = self._utc(entry_order_time)
        dates = pd.to_datetime(dataframe["date"], utc=True)
        closes = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        entry_rows = dataframe.loc[closes.le(filled_at)].sort_values("date")
        if entry_rows.empty:
            raise RuntimeError("wedge failure exit cannot locate the closed entry signal candle")
        row = entry_rows.iloc[-1]
        upper = float(row["pg2_wedge_upper"]); signal_high = float(row["high"])
        if not np.isfinite([upper, signal_high]).all() or not float(trade.open_rate) < upper or not float(trade.open_rate) < signal_high:
            raise ValueError("entry wedge failure levels must be above the short trade open rate")
        saved = {"upper": upper, "signal_high": signal_high, "plan": str(self.failure_plan.value), "entry_filled_at": filled_at.isoformat()}
        trade.set_custom_data(key=self._STATE_KEY, value=saved)
        return saved

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=self._STATE_KEY) is None:
            filled_at = getattr(order, "order_filled_utc", None) or getattr(trade, "date_entry_fill_utc", None)
            if filled_at is None:
                raise RuntimeError("wedge failure exit requires the actual entry fill timestamp")
            self._entry_state(pair, trade, filled_at)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        state = self._entry_state(pair, trade)
        if state is None:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(dataframe["date"], utc=True)
        closes_at = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        after_entry = dataframe.loc[dates.ge(self._utc(state["entry_filled_at"])) & closes_at.le(self._utc(current_time))].sort_values("date")
        if after_entry.empty:
            return None
        plan = str(state["plan"])
        level = float(state["upper"] if plan == "upper_wedge_close" else state["signal_high"])
        if not np.isfinite(level) or level <= 0.0:
            return None
        closes = pd.to_numeric(after_entry["close"], errors="coerce")
        failed = closes.iloc[-1] > level
        if plan == "signal_high_two_closes":
            failed = len(closes) >= 2 and closes.iloc[-2:].gt(level).all()
        return f"wedge_failure_{plan}" if failed else None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0
