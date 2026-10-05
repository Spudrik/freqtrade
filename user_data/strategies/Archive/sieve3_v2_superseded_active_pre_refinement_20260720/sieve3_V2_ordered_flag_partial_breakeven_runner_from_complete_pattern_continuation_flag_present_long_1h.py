"""Ordered measured-target partial, exact breakeven, and pole runner for the bullish 1h flag entry.

Source strategy: sieve2_continuation_flag_present_long_1h
Exit family: ordered source-geometry position management
Primary trigger: bullish pat_flag_pattern_present at the promoted score floor
Primary guard: promoted volume and bullish-pressure guards
Target provider: entry-frozen flag width and pat_impulse_up_pct pole projections
Invalidation provider: entry-frozen pat_flag_lower, then exactly trade.open_rate after the partial fills
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds first target, partial size, confirmation, and runner
Split rationale: every mode executes one sequence: measured partial, exact entry stop, full-pole runner.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_continuation import add_pattern_continuation

ENTRY_MODE = "entry_continuation_flag_present_long_1h"
ENTRY_TAG = "continuation_flag_present_long_1h"
SIDE = "long"
TIMEFRAME = "1h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "sieve2_continuation_flag_present_long_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = "sieve3_exit_ordered_flag_partial_breakeven_runner"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = "Realize part of the continuation at a measured target, protect the remainder exactly at entry, and hold for the full pole projection."
ENTRY_LOCK_STATUS = "authoritative exported buy params over archived executable defaults"

ORDERED_LADDER_PLANS: dict[str, tuple[str, float, str]] = {
    "flag_width_touch_p25_remaining_be_full_pole_runner": ("flag_width", 0.25, "touch"),
    "flag_width_touch_p33_remaining_be_full_pole_runner": ("flag_width", 0.33, "touch"),
    "flag_width_touch_p50_remaining_be_full_pole_runner": ("flag_width", 0.50, "touch"),
    "flag_width_rejection_p33_remaining_be_full_pole_runner": ("flag_width", 0.33, "rejection"),
    "half_pole_touch_p25_remaining_be_full_pole_runner": ("half_pole", 0.25, "touch"),
    "half_pole_touch_p33_remaining_be_full_pole_runner": ("half_pole", 0.33, "touch"),
    "half_pole_touch_p50_remaining_be_full_pole_runner": ("half_pole", 0.50, "touch"),
    "half_pole_rejection_p33_remaining_be_full_pole_runner": ("half_pole", 0.33, "rejection"),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



ACTIVE_SELL_PARAMS = ("exit_plan",)



class Sieve3V2OrderedFlagPartialBreakevenRunnerFromCompletePatternContinuationFlagPresentLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    volume_guard_window = 12
    volume_ratio_min = 1.3
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    pressure_window = 48
    pressure_min = 0.1
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    score_min = 0.55
    # Inactive buy declaration: rail_buffer_pct = CategoricalParameter(default=0.0, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    exit_plan = CategoricalParameter(tuple(ORDERED_LADDER_PLANS), default="half_pole_touch_p33_remaining_be_full_pole_runner", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_continuation(dataframe, timeframe=self.timeframe, min_flag_quality=float(self.score_min), min_pennant_quality=float(self.score_min))
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        window = int(self.pressure_window)
        pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
        guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pat_flag_pattern_present") & _num(dataframe, "pat_flag_direction").ge(1) & _num(dataframe, "pat_flag_indicator_score").ge(float(self.score_min))
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

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(self._utc(current_time))].sort_values("date").copy()

    def _entry_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed flag signal candle is available at trade entry.")
        row = frame.iloc[-1]
        upper = float(row["pat_flag_upper"])
        lower = float(row["pat_flag_lower"])
        pole_pct = float(row["pat_impulse_up_pct"])
        if not math.isfinite(upper) or not math.isfinite(lower) or not math.isfinite(pole_pct):
            raise RuntimeError("The flag signal candle has incomplete rail or pole geometry.")
        if lower <= 0.0 or upper <= lower or pole_pct <= 0.0 or lower >= float(trade.open_rate):
            raise RuntimeError("The flag signal candle has invalid long-side geometry.")
        return {
            "flag_low": lower,
            "flag_width": upper + (upper - lower),
            "half_pole": upper * (1.0 + 0.5 * pole_pct),
            "full_pole": upper * (1.0 + pole_pct),
        }

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        return frame.loc[dates.ge(filled_at)]

    @staticmethod
    def _first_target_confirmed(frame: DataFrame, target: float, confirmation: str) -> bool:
        touched = _num(frame, "high").ge(target)
        positions = np.flatnonzero(touched.to_numpy())
        if len(positions) == 0:
            return False
        if confirmation == "touch":
            return True
        post_touch = frame.iloc[int(positions[0]):]
        return bool(_num(post_touch, "close").iloc[-1] < _num(post_touch, "open").iloc[-1] and _num(post_touch, "close").iloc[-1] < target)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        if not self._partial_complete(trade, current_time):
            return None
        runner = self._entry_geometry(pair, trade)["full_pole"]
        frame = self._post_entry_frame(pair, trade, current_time)
        return "s3v2_flag_full_pole_runner" if bool(_num(frame, "high").ge(runner).any()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        stop_price = self._entry_geometry(pair, trade)["flag_low"]
        if self._partial_complete(trade, current_time):
            stop_price = float(trade.open_rate)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(trade.leverage or 1.0))

    PARTIAL_STAGE_TAG = "s3v2_flag_measured_partial"
    PARTIAL_STATE_KEY = "s3v2_partial_stage_state"

    @classmethod
    def _partial_state(cls, trade: Any) -> dict[str, Any]:
        raw = trade.get_custom_data(key=cls.PARTIAL_STATE_KEY)
        if isinstance(raw, dict):
            return dict(raw)
        return {
            "status": "ready",
            "target_stake": None,
            "credited_stake": 0.0,
            "order_credits": {},
            "requested_at": None,
            "filled_at": None,
        }

    @staticmethod
    def _partial_order_stake(order: Any, trade: Any, filled: bool) -> float:
        amount_name = "safe_filled" if filled else "safe_amount"
        amount = float(getattr(order, amount_name, 0.0) or 0.0)
        price = float(getattr(order, "safe_price", 0.0) or 0.0)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        return amount * price / leverage

    def _reconcile_partial(self, trade: Any, current_time: Any, extra_order: Any | None = None) -> dict[str, Any]:
        state = self._partial_state(trade)
        before = repr(state)
        orders = list(getattr(trade, "orders", ()) or ())
        if extra_order is not None and all(order is not extra_order for order in orders):
            orders.append(extra_order)
        matching = [
            order
            for order in orders
            if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ]
        if state.get("target_stake") is None and matching:
            state["target_stake"] = self._partial_order_stake(matching[0], trade, filled=False)
        credits = dict(state.get("order_credits") or {})
        credited = float(state.get("credited_stake") or 0.0)
        has_open = False
        fully_filled = False
        for order in matching:
            order_id = str(getattr(order, "order_id", None) or "")
            if not order_id:
                raise ValueError("partial-stage reconciliation requires an order id")
            filled_stake = self._partial_order_stake(order, trade, filled=True)
            previous = float(credits.get(order_id) or 0.0)
            if filled_stake > previous:
                credited += filled_stake - previous
                credits[order_id] = filled_stake
            status = str(getattr(order, "status", None) or "").casefold()
            is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
                "canceled", "cancelled", "closed", "expired", "failed", "rejected",
            }
            requested_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
            filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
            has_open |= is_open
            fully_filled |= requested_amount > 0.0 and filled_amount >= requested_amount - max(1e-12, requested_amount * 1e-9)
        state["order_credits"] = credits
        state["credited_stake"] = credited
        target = float(state.get("target_stake") or 0.0)
        tolerance = max(1e-8, target * 0.005)
        if fully_filled or (target > 0.0 and credited >= target - tolerance):
            state["status"] = "filled"
            state["requested_at"] = None
            if state.get("filled_at") is None:
                state["filled_at"] = pd.Timestamp(current_time).isoformat()
        elif has_open or (
            state.get("status") == "requested"
            and not matching
            and state.get("requested_at") == pd.Timestamp(current_time).isoformat()
        ):
            state["status"] = "requested"
        else:
            state["status"] = "ready"
            state["requested_at"] = None
        if repr(state) != before:
            trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return state

    def _request_partial(self, trade: Any, current_time: Any, desired: float, min_stake: float | None) -> float | None:
        state = self._reconcile_partial(trade, current_time)
        if state["status"] != "ready":
            return None
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        target = state.get("target_stake")
        request = min(stake, desired) if target is None else min(stake, max(0.0, float(target) - float(state["credited_stake"])))
        if min_stake is not None and 0.0 < stake - request < float(min_stake):
            request = stake - float(min_stake)
        if request <= 0.0 or request >= stake:
            return None
        if target is None:
            state["target_stake"] = request
        state["status"] = "requested"
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return request

    def _partial_complete(self, trade: Any, current_time: Any) -> bool:
        return self._reconcile_partial(trade, current_time)["status"] == "filled"


    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float, str]:
        _ = current_rate, current_profit, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if trade.has_open_orders:
            return None
        first_name, fraction, confirmation = ORDERED_LADDER_PLANS[str(self.exit_plan.value)]
        geometry = self._entry_geometry(str(trade.pair), trade)
        first_target = geometry[first_name]
        runner = geometry["full_pole"]
        if first_target <= float(trade.open_rate) or runner <= first_target:
            return None
        frame = self._post_entry_frame(str(trade.pair), trade, current_time)
        if not self._first_target_confirmed(frame, first_target, confirmation):
            return None
        remaining_stake = float(trade.stake_amount)
        partial_stake = remaining_stake * fraction
        if partial_stake <= 0.0 or partial_stake >= remaining_stake:
            return None
        if min_stake is not None and partial_stake < min_stake:
            return None
        request = self._request_partial(trade, current_time, partial_stake, min_stake)
        return (-request, self.PARTIAL_STAGE_TAG) if request is not None else None

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = (pair, kwargs)
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ):
            self._reconcile_partial(trade, current_time, order)
