from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_choch_bull_reversal_long_4h"
ENTRY_TAG = "choch_bull_reversal_long_4h"
SIDE = "long"
TIMEFRAME = "4h"
ENTRY_LOCK_STATUS = "verified_archived_snapshot_defaults_plus_selected_buy_params"
EXIT_THESIS = "Use a fixed 4% control target while testing the bullish CHoCH impulse candle as initial protection."
INVALIDATION = "Entry-frozen CHoCH signal-candle low, optionally buffered below by a small ATR fraction."
STOP_DATA_KEY = "s3v2_choch4h_signal_candle_low_stop"

BUFFER_PLANS = {
    "exact_signal_candle_low": 0.0,
    "quarter_atr_below_signal_candle_low": 0.25,
    "half_atr_below_signal_candle_low": 0.50,
}

LOCKED_BUY_PARAMS = {
    "use_sieve2_vp_guard": False,
    "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96,
    "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25,
    "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT",
    "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True,
    "volume_guard_window": 48,
    "volume_ratio_min": 0.8,
    "use_pressure_guard": True,
    "pressure_window": 24,
    "pressure_min": 0.2,
    "use_accumulation_guard": False,
    "use_body_direction_guard": False,
    "use_close_direction_guard": False,
    "strength": 3,
    "min_prominence_atr": 0.35,
    "min_pivot_spacing_bars": 2,
    "max_pivot_age_bars": 96,
    "breakout_buffer_atr": 0.3,
    "use_state_guard": True,
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


ACTIVE_SELL_PARAMS = ("invalidation_buffer",)


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_initial_invalidation_signal_candle_low"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_choch_bull_reversal_long_4h"
EXIT_HYPOTHESIS = "Test the initial invalidation signal candle low exit family while preserving the choch bull reversal long 4h entry behavior."


class Sieve3V2InitialInvalidationSignalCandleLowFromChochBullReversalLong4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("invalidation_buffer",)

    minimal_roi = {"0": 0.04}
    stoploss = -0.99
    use_exit_signal = False
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False; its branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    invalidation_buffer = CategoricalParameter(tuple(BUFFER_PLANS), default="quarter_atr_below_signal_candle_low", space="sell", optimize=True, load=True)
    invalidation_buffer.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(
            dataframe,
            strength=int(self.strength),
            min_prominence_atr=float(self.min_prominence_atr),
            min_pivot_spacing_bars=int(self.min_pivot_spacing_bars),
            max_pivot_age_bars=int(self.max_pivot_age_bars),
            breakout_buffer_atr=float(self.breakout_buffer_atr),
            include_diagnostics=True,
            prefix="ms",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if self.use_volume_guard:
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if self.use_pressure_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if self.use_pressure_guard:
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_choch_to_bull")
        if self.use_state_guard:
            condition &= _num(dataframe, "ms_state").ge(0)
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

    def _entry_signal_row(self, pair: str, order_time: datetime) -> Series:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        close_times = pd.to_datetime(dataframe["date"], utc=True) + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        row = dataframe.loc[close_times.le(pd.Timestamp(order_time))].iloc[-1]
        if not bool(row["ms_choch_to_bull"]):
            raise RuntimeError("entry order does not map to a closed bullish CHoCH signal candle")
        return row

    def order_filled(self, pair, trade, order, current_time, **kwargs) -> None:
        _ = (current_time, kwargs)
        if order.ft_order_side != trade.entry_side or trade.get_custom_data(key=STOP_DATA_KEY) is not None:
            return
        row = self._entry_signal_row(pair, order.order_date_utc)
        buffer = BUFFER_PLANS[str(self.invalidation_buffer.value)]
        stop_rate = float(row["low"]) - float(row["ms_atr"]) * buffer
        entry_rate = float(order.safe_price)
        if not math.isfinite(stop_rate) or stop_rate <= 0.0 or stop_rate >= entry_rate:
            raise RuntimeError("signal-candle-low invalidation must be finite, positive, and below entry")
        trade.set_custom_data(key=STOP_DATA_KEY, value=stop_rate)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        stop_rate = trade.get_custom_data(key=STOP_DATA_KEY)
        if stop_rate is None:
            raise RuntimeError("entry-frozen signal-candle-low invalidation is missing")
        return stoploss_from_absolute(float(stop_rate), current_rate, is_short=False, leverage=float(trade.leverage or 1.0))
