"""Ordered channel midpoint partial, exact-entry stop, and upper-rail runner.

Sequence: reach a favorable-side channel midpoint on a closed 8h candle, fill
one partial, move the remainder stop to exactly ``trade.open_rate``, then let
the remainder run to the active upper rail.  No trailing or unrelated target
is included.  Active sell parameter: ``runner_plan``.
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
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = "entry_geometry_ascending_channel_lower_bounce_long_8h"
ENTRY_TAG = "geometry_ascending_channel_lower_bounce_long_8h"
SIDE = "long"
TIMEFRAME = "8h"
SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_ascending_channel_lower_bounce_long_8h.py"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_geometry_8h_2020_23"
ENTRY_LOCK_STATUS = "executable_source_defaults_frozen_historical_promoted_snapshots_conflict"
EXIT_HYPOTHESIS = "Bank once at channel midpoint, protect the remainder at exact entry, and run to upper rail."

RUNNER_PLANS: dict[str, tuple[str, float, str]] = {
    "midline_close_p33_entry_stop_upper_touch": ("midline_close", 0.33, "upper_touch"),
    "midline_close_p50_entry_stop_upper_close": ("midline_close", 0.50, "upper_close"),
    "midline_touch_p33_entry_stop_upper_close": ("midline_touch", 0.33, "upper_close"),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



ACTIVE_SELL_PARAMS = ("runner_plan",)


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_midline_partial_entry_stop_upper_runner"


class Sieve3V2MidlinePartialEntryStopUpperRunnerFromCompletePatternGeometryAscendingChannelLowerBounceLong8H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("runner_plan",)
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
    volume_ratio_min = 1.0
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    pressure_window = 24
    pressure_min = 0.1
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.55
    min_containment = 0.88
    max_recent_touch_age_bars = 12
    channel_min_pattern_bars = 18
    channel_max_pattern_bars = 96
    channel_min_quality = 0.82
    channel_min_containment = 0.68
    channel_near_boundary_atr_mult = 0.7
    channel_breakout_atr_mult = 0.35
    channel_lifecycle_confirm_break_bars = 2
    score_min = 0.65
    width_atr_max = 6.0
    rail_buffer_pct = 0.004

    runner_plan = CategoricalParameter(tuple(RUNNER_PLANS), default="midline_close_p33_entry_stop_upper_touch", space="sell", optimize=True, load=True)
    runner_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(
            dataframe, timeframe=self.timeframe, output_slots=1,
            include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False,
            include_rectangle_patterns=False, include_ascending_channel_patterns=True, include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score),
            min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),
            channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars),
            channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment),
            channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),
            channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2",
        )
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
        guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == "short" else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_ascending_channel_lower")
        condition = _bool(dataframe, "pg2_ascending_channel_pattern_present") & _num(dataframe, "pg2_ascending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_ascending_channel_width_atr").le(float(self.width_atr_max)) & _num(dataframe, "low").le(lower.mul(1.0 + float(self.rail_buffer_pct))) & _num(dataframe, "close").gt(lower) & _num(dataframe, "close").gt(_num(dataframe, "open"))
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

    def _latest_closed_channel(
        self,
        pair: str,
        current_time: datetime,
        after: datetime | None = None,
    ) -> tuple[Series, float, float]:
        if self.dp is None:
            raise RuntimeError("channel runner requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        close_times = pd.to_datetime(dataframe["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        now = pd.Timestamp(current_time)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        closed = dataframe.loc[close_times.le(now)].sort_values("date")
        if after is not None:
            filled_at = pd.Timestamp(after)
            filled_at = (
                filled_at.tz_localize("UTC")
                if filled_at.tzinfo is None
                else filled_at.tz_convert("UTC")
            )
            candle_opens = pd.to_datetime(closed["date"], utc=True, errors="raise")
            closed = closed.loc[candle_opens.ge(filled_at)]
        if closed.empty:
            raise RuntimeError("channel runner has no closed 8h candle")
        row = closed.iloc[-1]
        if not bool(row["pg2_ascending_channel_pattern_present"]):
            return row, math.nan, math.nan
        upper = float(row["pg2_ascending_channel_upper"])
        lower = float(row["pg2_ascending_channel_lower"])
        if not np.isfinite([upper, lower]).all() or upper <= lower:
            raise RuntimeError("active ascending channel has invalid runner rails")
        return row, upper, lower

    PARTIAL_STAGE_TAG = "s3v2_midline_partial"
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


    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> tuple[float, str] | None:
        _ = current_rate, current_profit, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(trade.has_open_orders):
            return None
        partial_mode, fraction, _ = RUNNER_PLANS[str(self.runner_plan.value)]
        row, upper, lower = self._latest_closed_channel(
            trade.pair,
            current_time,
            after=getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc,
        )
        if not np.isfinite([upper, lower]).all():
            return None
        midpoint = 0.5 * (upper + lower)
        if midpoint <= float(trade.open_rate):
            return None
        reached = float(row["close"]) >= midpoint if partial_mode == "midline_close" else float(row["high"]) >= midpoint
        if not reached:
            return None
        amount = float(trade.stake_amount) * fraction
        if amount <= 0.0 or (min_stake is not None and amount < min_stake):
            return None
        request = self._request_partial(trade, current_time, amount, min_stake)
        return (-request, self.PARTIAL_STAGE_TAG) if request is not None else None

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = (pair, kwargs)
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ):
            self._reconcile_partial(trade, current_time, order)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        if not self._partial_complete(trade, current_time):
            return None
        _, _, runner_mode = RUNNER_PLANS[str(self.runner_plan.value)]
        row, upper, lower = self._latest_closed_channel(
            pair,
            current_time,
            after=getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc,
        )
        _ = lower
        if not np.isfinite(upper) or upper <= float(trade.open_rate):
            return None
        reached = float(row["high"]) >= upper if runner_mode == "upper_touch" else float(row["close"]) >= upper
        return f"s3v2_{runner_mode}_runner" if reached else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = pair, current_time, current_profit, after_fill, kwargs
        stop_price = float(trade.open_rate) if self._partial_complete(trade, current_time) else float(trade.open_rate) * 0.96
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(trade.leverage or 1.0))
