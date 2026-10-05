from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


EXIT_FAMILY = 'profit_armed_bullish_reversal'
PRIMARY_TRIGGER = '4h close below the wedge lower rail with squeeze, score, and non-positive direction confirmation'
PRIMARY_GUARD = 'locked entry volume guard'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'selected bullish reversal with optional rising-volume confirmation'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'filename-derived source foundation: complete_pattern_geometry_wedge_breakdown_short_4h'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_reversal'
EXIT_HYPOTHESIS = 'After profit is armed, a selected bullish closed-candle reversal should close the 4h wedge-breakdown short.'
ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_rising_volume')
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_wedge_breakdown_short_4h'

class Sieve3V2ProfitArmedBullishReversalFromCompletePatternGeometryWedgeBreakdownShort4H(IStrategy):
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

    reversal_plan = CategoricalParameter(
        ["bullish_close_after_1_5pct", "two_bullish_closes_after_2pct", "bullish_engulf_after_3pct", "prior_high_reclaim_after_2pct"],
        default="two_bullish_closes_after_2pct", space="sell", optimize=True, load=True,
    )
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    require_rising_volume = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    require_rising_volume.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_rising_volume')

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

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        if not self.dp:
            return None
        dataframe = self._closed_trade_candles(pair, trade, current_time)
        if len(dataframe) < 2:
            return None
        current = dataframe.iloc[-1]
        previous = dataframe.iloc[-2]
        plan = str(self.reversal_plan.value)
        profit_floor = {
            "bullish_close_after_1_5pct": 0.015,
            "two_bullish_closes_after_2pct": 0.02,
            "bullish_engulf_after_3pct": 0.03,
            "prior_high_reclaim_after_2pct": 0.02,
        }[plan]
        if current_profit < profit_floor:
            return None
        if bool(self.require_rising_volume.value) and float(current["volume"]) <= float(previous["volume"]):
            return None
        bullish = float(current["close"]) > float(current["open"])
        if plan == "bullish_close_after_1_5pct":
            reversal = bullish
        elif plan == "two_bullish_closes_after_2pct":
            reversal = bullish and float(previous["close"]) > float(previous["open"])
        elif plan == "bullish_engulf_after_3pct":
            reversal = bullish and float(previous["close"]) < float(previous["open"]) and float(current["open"]) <= float(previous["close"]) and float(current["close"]) >= float(previous["open"])
        else:
            reversal = bullish and float(current["close"]) > float(previous["high"])
        return f"profit_armed_{plan}" if reversal else None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0
