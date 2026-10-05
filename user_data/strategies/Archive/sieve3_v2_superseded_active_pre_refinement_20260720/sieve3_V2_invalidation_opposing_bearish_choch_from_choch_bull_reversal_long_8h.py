from __future__ import annotations

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


ENTRY_MODE = "entry_choch_bull_reversal_long_8h"
ENTRY_TAG = "choch_bull_reversal_long_8h"
SIDE = "long"
TIMEFRAME = "8h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = (
    "user_data/strategies/sieve3_exit_fixed_tp_sl_from_choch_bull_reversal_long_8h.py:"
    "Sieve3ExitFixedTpSlFromChochBullReversalLong8H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = "sieve3_exit_invalidation_opposing_bearish_choch"
ENTRY_SOURCE_STAGE = "sieve2"
ENTRY_LOCK_STATUS = "legacy_executable_defaults_locked; promoted_params_overlay_missing"
EXIT_HYPOTHESIS = (
    "A confirmed opposing bearish 8h CHoCH invalidates the bullish reversal entry "
    "before a fixed catastrophe cap."
)

# Primary trigger: confirmed ms_choch_to_bull event.
# Primary guard: locked 12-candle volume >= 1.3x its shifted baseline.
# Target: fixed 6% control; invalidation: confirmed ms_choch_to_bear with optional
# candle confirmation.
# Active sell params: choch_confirmation, catastrophe_stop. Both affect every selected plan.
# Kept separate because event invalidation is materially different from an
# entry-time swing-low stop.




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)




class Sieve3V2InvalidationOpposingBearishChochFromChochBullReversalLong8H(IStrategy):
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
    # Legacy fixed-off entry parameters: use_pressure_guard=False, pressure_window=12,
    # pressure_min=0.1, and use_accumulation_guard=False; their pressure branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.15
    # Legacy fixed-off entry parameter: use_state_guard=False; the state branch is unreachable.

    CHOCH_CONFIRMATIONS = ("event", "event_bearish_close", "event_two_bearish_closes")
    CATASTROPHE_STOPS = {"cap_4pct": 0.04, "cap_6pct": 0.06, "cap_8pct": 0.08}
    choch_confirmation = CategoricalParameter(
        CHOCH_CONFIRMATIONS, default="event", space="sell", optimize=True, load=True
    )
    choch_confirmation.batch_tags = ('family:exits', 'mode:sieve3_exit')
    catastrophe_stop = CategoricalParameter(
        tuple(CATASTROPHE_STOPS), default="cap_6pct", space="sell", optimize=True, load=True
    )
    catastrophe_stop.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ("choch_confirmation", "catastrophe_stop")

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
        if self.use_volume_guard:
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

    def _post_entry_rows_through(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
    ) -> DataFrame:
        if self.dp is None:
            raise RuntimeError("opposing CHoCH invalidation requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return DataFrame()
        required = {"date", "open", "close", "ms_choch_to_bear"}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise KeyError(f"opposing CHoCH invalidation missing required source columns: {missing}")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        filled_at = pd.Timestamp(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        filled_at = (
            filled_at.tz_localize("UTC")
            if filled_at.tzinfo is None
            else filled_at.tz_convert("UTC")
        )
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return dataframe.loc[dates.ge(filled_at) & close_times.le(cutoff)]

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = (current_profit, kwargs)
        rows = self._post_entry_rows_through(pair, trade, current_time)
        if current_rate >= float(trade.open_rate) * 1.06:
            return "fixed_reward_6pct"
        if rows.empty or not bool(rows.iloc[-1]["ms_choch_to_bear"]):
            return None
        confirmation = str(self.choch_confirmation.value)
        if confirmation == "event_bearish_close" and not float(rows.iloc[-1]["close"]) < float(
            rows.iloc[-1]["open"]
        ):
            return None
        if confirmation == "event_two_bearish_closes":
            if len(rows) < 2 or not (
                float(rows.iloc[-1]["close"]) < float(rows.iloc[-1]["open"])
                and float(rows.iloc[-2]["close"]) < float(rows.iloc[-2]["open"])
            ):
                return None
        return f"opposing_bearish_choch_{confirmation}"

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
        stop_ratio = self.CATASTROPHE_STOPS[str(self.catastrophe_stop.value)]
        return stoploss_from_absolute(
            float(trade.open_rate) * (1.0 - stop_ratio),
            current_rate=current_rate,
            is_short=False,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
