from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_MODE = "entry_choch_bull_reversal_long_8h"


TIMEFRAME = "8h"


SOURCE_STRATEGY = (
    "user_data/strategies/sieve3_exit_fixed_tp_sl_from_choch_bull_reversal_long_8h.py:"
    "Sieve3ExitFixedTpSlFromChochBullReversalLong8H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = 'sieve3_exit_profit_armed_bearish_structure_reversal'
ENTRY_SOURCE_STAGE = 'sieve2'
ENTRY_LOCK_STATUS = "legacy_executable_defaults_locked; promoted_params_overlay_missing"


# Primary trigger: confirmed ms_choch_to_bull event.
# Primary guard: locked 12-candle volume >= 1.3x its shifted baseline.
# Target provider: open profit; reversal provider: ms_choch_to_bear/ms_bos_to_bear.
# Invalidation: fixed 4% initial stop. Active sell params: reversal_plan, require_bearish_close.
# Kept separate because bearish structure is profit management here, not entry invalidation.




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'profit_armed_bearish_structure_reversal'
PRIMARY_TRIGGER = 'confirmed 8h ms_choch_to_bull'
PRIMARY_GUARD = 'locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bearish 8h CHoCH or BOS with optional bearish-close confirmation; fixed 4% stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Once profit is available, a confirmed bearish 8h CHoCH or BOS should close the reversal long before it gives back the move.'
ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_bearish_close')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'choch_bull_reversal_long_8h'

class Sieve3V2ProfitArmedBearishStructureReversalFromChochBullReversalLong8H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False
    exit_profit_only = False

    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.3
    # Legacy fixed-off entry parameter: use_pressure_guard=False (BooleanParameter, buy space).
    pressure_window = 12
    pressure_min = 0.1
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.15
    # Legacy fixed-off entry parameter: use_state_guard=False (BooleanParameter, buy space).

    REVERSAL_PLANS = {
        "bear_choch_after_profit_1": ("choch", 0.01),
        "bear_choch_after_profit_2": ("choch", 0.02),
        "bear_choch_after_profit_3": ("choch", 0.03),
        "bear_bos_after_profit_1": ("bos", 0.01),
        "bear_bos_after_profit_2": ("bos", 0.02),
        "either_bear_event_after_profit_2": ("either", 0.02),
    }
    reversal_plan = CategoricalParameter(
        tuple(REVERSAL_PLANS),
        default="either_bear_event_after_profit_2",
        space="sell",
        optimize=True,
        load=True,
    )
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    require_bearish_close = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    require_bearish_close.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ("reversal_plan", "require_bearish_close")

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = (
            pair,
            current_time,
            current_rate,
            proposed_leverage,
            max_leverage,
            entry_tag,
            side,
            kwargs,
        )
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
        _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = (
                volume.shift(1)
                .rolling(window, min_periods=max(2, window // 3))
                .mean()
                .replace(0.0, np.nan)
            )
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_choch_to_bull")
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

    def _current_row(self, pair: str, current_time: datetime, trade: Any) -> Series | None:
        if self.dp is None:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty or "date" not in dataframe.columns:
            return None
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        eligible = dataframe.loc[dates.ge(entered) & close_times.le(cutoff)]
        return None if eligible.empty else eligible.iloc[-1]

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = (trade, current_rate, kwargs)
        event_mode, profit_gate = self.REVERSAL_PLANS[str(self.reversal_plan.value)]
        if current_profit < profit_gate:
            return None
        row = self._current_row(pair, current_time, trade)
        if row is None:
            return None
        bearish_choch = bool(row["ms_choch_to_bear"])
        bearish_bos = bool(row["ms_bos_to_bear"])
        event_matches = (
            bearish_choch
            if event_mode == "choch"
            else bearish_bos
            if event_mode == "bos"
            else bearish_choch or bearish_bos
        )
        if not event_matches:
            return None
        if bool(self.require_bearish_close.value) and not float(row["close"]) < float(row["open"]):
            return None
        return f"profit_armed_{self.reversal_plan.value}"

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        return stoploss_from_absolute(
            float(trade.open_rate) * 0.96,
            current_rate=current_rate,
            is_short=False,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
