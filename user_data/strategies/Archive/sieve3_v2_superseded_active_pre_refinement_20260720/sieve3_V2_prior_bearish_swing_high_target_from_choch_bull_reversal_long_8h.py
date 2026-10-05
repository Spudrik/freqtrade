from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_MODE = "entry_choch_bull_reversal_long_8h"


TIMEFRAME = "8h"

SOURCE_STRATEGY = (
    "user_data/strategies/sieve3_exit_fixed_tp_sl_from_choch_bull_reversal_long_8h.py:"
    "Sieve3ExitFixedTpSlFromChochBullReversalLong8H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = 'sieve3_exit_prior_bearish_swing_high_target'
ENTRY_SOURCE_STAGE = 'sieve2'
ENTRY_LOCK_STATUS = "legacy_executable_defaults_locked; promoted_params_overlay_missing"
ENTRY_TARGET_STATE_KEY = "s3v2_prior_bearish_swing_high_entry_target"


# Primary trigger: confirmed ms_choch_to_bull event.
# Primary guard: locked 12-candle volume >= 1.3x its shifted baseline.
# Target: entry-time ms_prev_swing_high; invalidation: fixed 4% initial stop.
# Active sell parameter: target_response. Its named plans combine distance gate and confirmation.
# Kept separate because this asks only whether the genuine source level should close the full trade.




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'prior_bearish_swing_high_target'
PRIMARY_TRIGGER = 'confirmed 8h ms_choch_to_bull'
PRIMARY_GUARD = 'locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'older confirmed bearish-structure swing high'
PRIMARY_INVALIDATION = 'fixed 4% adverse stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'The older confirmed swing high from the bearish structure is the natural recovery target after an 8h bullish CHoCH.'
ACTIVE_SELL_PARAMS = ('target_response',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'choch_bull_reversal_long_8h'

class Sieve3V2PriorBearishSwingHighTargetFromChochBullReversalLong8H(IStrategy):
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

    TARGET_RESPONSES = {
        "touch_full_min_distance_1pct": ("touch", 0.01),
        "touch_full_min_distance_2pct": ("touch", 0.02),
        "touch_full_min_distance_3pct": ("touch", 0.03),
        "bearish_rejection_full_min_distance_1pct": ("rejection", 0.01),
        "bearish_rejection_full_min_distance_2pct": ("rejection", 0.02),
        "bearish_choch_after_touch_full_min_distance_1pct": ("bear_choch", 0.01),
    }
    target_response = CategoricalParameter(
        tuple(TARGET_RESPONSES),
        default="touch_full_min_distance_2pct",
        space="sell",
        optimize=True,
        load=True,
    )
    target_response.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ("target_response",)

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
            include_diagnostics=True,
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

    def _closed_rows(self, pair: str, cutoff_time: datetime) -> DataFrame:
        if self.dp is None:
            raise RuntimeError("structure target exit requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "high", "close", "ms_prev_swing_high", "ms_choch_to_bear"}
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise KeyError(f"structure target exit missing columns: {missing}")
        cutoff = pd.Timestamp(cutoff_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return dataframe.loc[close_times.le(cutoff)]

    def _freeze_entry_target(self, pair: str, trade: Any, cutoff_time: datetime) -> float:
        rows = self._closed_rows(pair, cutoff_time)
        if rows.empty:
            raise RuntimeError("no closed entry-signal candle is available for structure target")
        target = float(rows.iloc[-1]["ms_prev_swing_high"])
        if not np.isfinite(target) or target <= 0.0:
            raise RuntimeError("entry signal has no usable prior bearish swing-high target")
        trade.set_custom_data(key=ENTRY_TARGET_STATE_KEY, value={"target": target})
        return target

    def _entry_target(self, pair: str, trade: Any, min_distance: float) -> float | None:
        state = trade.get_custom_data(key=ENTRY_TARGET_STATE_KEY)
        if state is None:
            cutoff = getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
            target = self._freeze_entry_target(pair, trade, cutoff)
        elif not isinstance(state, dict):
            raise RuntimeError("structure target entry snapshot is invalid")
        else:
            target = float(state["target"])
        minimum = float(trade.open_rate) * (1.0 + min_distance)
        return target if np.isfinite(target) and target >= minimum else None

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=ENTRY_TARGET_STATE_KEY) is None:
            self._freeze_entry_target(pair, trade, current_time)

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
        response, min_distance = self.TARGET_RESPONSES[str(self.target_response.value)]
        target = self._entry_target(pair, trade, min_distance)
        if target is None:
            return None
        rows = self._closed_rows(pair, current_time)
        dates = pd.to_datetime(rows["date"], utc=True, errors="raise")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        rows = rows.loc[dates.ge(entered)]
        if rows.empty:
            return None
        row = rows.iloc[-1]
        touched = (
            max(float(getattr(trade, "max_rate", current_rate) or current_rate), float(row["high"]))
            >= target
        )
        if not touched:
            return None
        if response == "touch":
            return f"prior_bearish_swing_high_{self.target_response.value}"
        if (
            response == "rejection"
            and float(row["close"]) < target
            and float(row["close"]) < float(row["open"])
        ):
            return f"prior_bearish_swing_high_{self.target_response.value}"
        if response == "bear_choch" and bool(row["ms_choch_to_bear"]):
            return f"prior_bearish_swing_high_{self.target_response.value}"
        return None

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
