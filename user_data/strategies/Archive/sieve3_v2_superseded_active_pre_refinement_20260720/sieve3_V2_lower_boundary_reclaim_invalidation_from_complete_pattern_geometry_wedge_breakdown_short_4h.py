from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ACTIVE_SELL_PARAMS = ("reclaim_plan",)
STATE_KEY = "s3v2_wedge_breakdown_signal_geometry"


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_lower_boundary_reclaim_invalidation"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_complete_pattern_geometry_wedge_breakdown_short_4h"
EXIT_HYPOTHESIS = "Test the lower boundary reclaim invalidation exit family while preserving the complete pattern geometry wedge breakdown short 4h entry behavior."


class Sieve3V2LowerBoundaryReclaimInvalidationFromCompletePatternGeometryWedgeBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    ACTIVE_SELL_PARAMS = ("reclaim_plan",)
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

    reclaim_plan = CategoricalParameter(
        ["one_close", "two_closes", "one_close_buffer_0_5pct"],
        default="one_close", space="sell", optimize=True, load=True,
    )
    reclaim_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _signal_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        saved = trade.get_custom_data(key=STATE_KEY)
        if isinstance(saved, dict):
            return {"lower": float(saved["lower"])}
        if not self.dp:
            raise RuntimeError("signal geometry requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("signal geometry requires a non-empty analyzed dataframe")
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        closes_at = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        signal_rows = dataframe.loc[closes_at.le(filled_at)]
        if signal_rows.empty:
            raise RuntimeError("no fully closed pre-entry signal candle is available")
        lower = float(signal_rows.iloc[-1]["pg2_wedge_lower"])
        if not np.isfinite(lower) or lower <= 0.0:
            raise ValueError("entry-frozen wedge lower boundary must be finite and positive")
        state = {"lower": lower}
        trade.set_custom_data(key=STATE_KEY, value=state)
        return state

    def _trade_frames(
        self, pair: str, trade: Any, current_time: datetime
    ) -> tuple[dict[str, float], DataFrame] | None:
        if not self.dp:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        candle_open = self._utc(timeframe_to_prev_date(self.timeframe, filled_at.to_pydatetime()))
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        closes_at = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = closes_at.le(self._utc(current_time))
        post_fill = dates.ge(candle_open) if filled_at == candle_open else dates.gt(candle_open)
        after_entry = dataframe.loc[closed & post_fill]
        if after_entry.empty:
            return None
        return self._signal_geometry(pair, trade), after_entry

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        frames = self._trade_frames(pair, trade, current_time)
        if frames is None:
            return None
        entry, after_entry = frames
        lower = float(entry["lower"])
        plan = str(self.reclaim_plan.value)
        threshold = lower * (1.005 if plan == "one_close_buffer_0_5pct" else 1.0)
        closes = pd.to_numeric(after_entry["close"], errors="coerce")
        reclaimed = closes.iloc[-1] > threshold
        if plan == "two_closes":
            reclaimed = len(closes) >= 2 and closes.iloc[-2:].gt(threshold).all()
        return f"lower_boundary_reclaim_{plan}" if reclaimed else None

    def order_filled(
        self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any
    ) -> None:
        _ = current_time, kwargs
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None)
            and trade.get_custom_data(key=STATE_KEY) is None
        ):
            self._signal_geometry(pair, trade)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0
