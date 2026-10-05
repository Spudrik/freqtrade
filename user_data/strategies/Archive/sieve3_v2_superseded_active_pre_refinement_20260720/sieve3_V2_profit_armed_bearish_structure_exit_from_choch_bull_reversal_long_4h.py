from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_choch_bull_reversal_long_4h"


TIMEFRAME = "4h"
ENTRY_LOCK_STATUS = "verified_archived_snapshot_defaults_plus_selected_buy_params"
EXIT_THESIS = "Exit a profitable long when 4h structure reverses through bearish CHoCH or sustained bearish state."
INVALIDATION = "Fixed 2% initial stop; reversal evidence is ignored until its named profit arm is reached."

REVERSAL_PLANS = {
    "bearish_choch_after_profit1": ("choch", 0.01),
    "bearish_choch_after_profit2": ("choch", 0.02),
    "bearish_state_two_closes_after_profit2": ("state_two", 0.02),
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


EXIT_FAMILY = 'profit_armed_bearish_structure_exit'
PRIMARY_TRIGGER = 'confirmed 4h ms_choch_to_bull'
PRIMARY_GUARD = 'non-negative ms_state plus locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'selected open-profit arm'
PRIMARY_INVALIDATION = 'bearish CHoCH or two negative ms_state closes'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'filename-derived source foundation: choch_bull_reversal_long_4h'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bearish_structure_exit'
EXIT_HYPOTHESIS = 'After profit is armed, bearish CHoCH or persistent negative structure state should close the 4h bullish reversal.'
ACTIVE_SELL_PARAMS = ('reversal_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'choch_bull_reversal_long_4h'

class Sieve3V2ProfitArmedBearishStructureExitFromChochBullReversalLong4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.02
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    reversal_plan = CategoricalParameter(tuple(REVERSAL_PLANS), default="bearish_choch_after_profit1", space="sell", optimize=True, load=True)
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('reversal_plan',)

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
            prefix="ms",
        )
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
            close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
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
        if bool(self.use_state_guard):
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

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> str | None:
        _ = (current_rate, kwargs)
        mode, profit_arm = REVERSAL_PLANS[str(self.reversal_plan.value)]
        if current_profit < profit_arm:
            return None
        dataframe = self._closed_trade_candles(pair, trade, current_time)
        if dataframe.empty:
            return None
        if mode == "choch" and bool(_bool(dataframe, "ms_choch_to_bear").iloc[-1]):
            return "profit_armed_bearish_choch"
        if mode == "state_two" and len(dataframe) >= 2 and bool(_num(dataframe, "ms_state").tail(2).lt(0).all()):
            return "profit_armed_bearish_state_two"
        return None
