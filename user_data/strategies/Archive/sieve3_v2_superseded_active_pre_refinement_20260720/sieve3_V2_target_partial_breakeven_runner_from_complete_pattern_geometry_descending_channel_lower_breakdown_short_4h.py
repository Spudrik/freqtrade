from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_4h"
ENTRY_TAG, SIDE, TIMEFRAME = "geometry_descending_channel_lower_breakdown_short_4h", "short", "4h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "user_data/strategies/sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_4h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"
SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"
RESEARCH_PATH = 'sieve3_exit_target_partial_breakeven_runner'
ENTRY_SOURCE_STAGE = 'sieve2'
EXIT_HYPOTHESIS = "Take one ordered measured-target partial, move the remainder stop to exact entry, then run to a deeper downside target or bullish reversal."
ENTRY_LOCK_STATUS = "unanimous_current_executable_defaults_locked; historical promoted snapshots conflict"
ENTRY_LOCK_SHA256 = "8b900a491a755ddc11adb21c8bd307b92af2b3848ae48a9e06766233dc8673b9"


def _num(frame: DataFrame, column: str, _default: float | Series | object = ...) -> Series:
    if column not in frame.columns:
        if _default is ...:
            raise KeyError(f"required numeric column is missing: {column!r}")
        if isinstance(_default, Series):
            return pd.to_numeric(_default, errors="coerce").reindex(frame.index)
        return pd.Series(float(_default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


EXIT_FAMILY = 'target_partial_breakeven_runner'
ACTIVE_SELL_PARAMS = ('management_plan',)




class Sieve3V2TargetPartialBreakevenRunnerFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H(IStrategy):
    ACTIVE_SELL_PARAMS = ('management_plan',)
    INTERFACE_VERSION = 3
    timeframe, startup_candle_count, process_only_new_candles, can_short = TIMEFRAME, 180, True, True
    minimal_roi, stoploss = {"0": 100.0}, -0.12
    use_exit_signal, use_custom_stoploss, position_adjustment_enable = True, True, True
    max_entry_position_adjustment = 0
    trailing_stop, ignore_roi_if_entry_signal = False, False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.15
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.5
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

    management_plan = CategoricalParameter(("half_to_one_engulf_33", "half_to_one_two_bull_50", "three_quarter_to_one_half_engulf_33", "three_quarter_to_one_half_two_bull_50", "one_to_one_half_engulf_33", "one_to_one_half_two_bull_50", "one_to_two_engulf_33", "one_to_two_two_bull_50", "one_quarter_to_two_engulf_33", "one_quarter_to_two_two_bull_50", "one_half_to_two_half_engulf_33", "one_half_to_two_half_two_bull_50"), default="one_to_two_two_bull_50", space="sell", optimize=True, load=True)
    management_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    MANAGEMENT_PLANS = {
        "half_to_one_engulf_33": (0.50, 0.33, 1.00, "engulf", 0.04), "half_to_one_two_bull_50": (0.50, 0.50, 1.00, "two_bull", 0.04),
        "three_quarter_to_one_half_engulf_33": (0.75, 0.33, 1.50, "engulf", 0.05), "three_quarter_to_one_half_two_bull_50": (0.75, 0.50, 1.50, "two_bull", 0.05),
        "one_to_one_half_engulf_33": (1.00, 0.33, 1.50, "engulf", 0.05), "one_to_one_half_two_bull_50": (1.00, 0.50, 1.50, "two_bull", 0.05),
        "one_to_two_engulf_33": (1.00, 0.33, 2.00, "engulf", 0.06), "one_to_two_two_bull_50": (1.00, 0.50, 2.00, "two_bull", 0.06),
        "one_quarter_to_two_engulf_33": (1.25, 0.33, 2.00, "engulf", 0.06), "one_quarter_to_two_two_bull_50": (1.25, 0.50, 2.00, "two_bull", 0.06),
        "one_half_to_two_half_engulf_33": (1.50, 0.33, 2.50, "engulf", 0.08), "one_half_to_two_half_two_bull_50": (1.50, 0.50, 2.50, "two_bull", 0.08),
    }

    def _partial_fill_state(self, trade: Any) -> dict[str, Any]:
        key = f"sieve3_v2_partial_fill:{self.__class__.__name__}"
        state = trade.get_custom_data(key=key)
        if state is None:
            state = {
                "status": "ready",
                "target_stake": None,
                "credited_stake": 0.0,
                "request_stake": None,
                "requested_at": None,
                "order_id": None,
                "resolved_order_ids": [],
            }
            trade.set_custom_data(key=key, value=state)
        return dict(state)

    def _partial_fill_save(self, trade: Any, state: dict[str, Any]) -> None:
        key = f"sieve3_v2_partial_fill:{self.__class__.__name__}"
        trade.set_custom_data(key=key, value=state)

    @staticmethod
    def _partial_fill_order_key(order: Any) -> str:
        order_key = str(getattr(order, "order_id", None) or getattr(order, "id", None) or "")
        if not order_key:
            raise ValueError("partial exit order requires a persistent identity")
        return order_key

    @staticmethod
    def _partial_fill_snapshot(order: Any) -> tuple[bool, float, bool]:
        status = str(getattr(order, "status", None) or "").casefold()
        is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "closed",
        }
        filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
        requested_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
        filled_stake = max(0.0, float(getattr(order, "stake_amount_filled", 0.0) or 0.0))
        tolerance = max(1e-12, requested_amount * 1e-9)
        return is_open, filled_stake, requested_amount > 0.0 and filled_amount >= requested_amount - tolerance

    def _partial_fill_apply(
        self,
        trade: Any,
        state: dict[str, Any],
        order: Any,
        current_time: datetime,
    ) -> None:
        order_key = self._partial_fill_order_key(order)
        resolved = state.setdefault("resolved_order_ids", [])
        if order_key in {str(value) for value in resolved}:
            return
        is_open, filled_stake, fully_filled = self._partial_fill_snapshot(order)
        if is_open:
            state["status"] = "requested"
            state["order_id"] = order_key
            return
        target_stake = float(state.get("target_stake") or 0.0)
        if filled_stake > 0.0:
            credited = float(state.get("credited_stake") or 0.0) + filled_stake
            state["credited_stake"] = min(target_stake, credited) if target_stake > 0.0 else credited
        resolved.append(order_key)
        tolerance = max(1e-9, target_stake * 1e-9)
        target_reached = target_stake > 0.0 and float(state.get("credited_stake") or 0.0) >= target_stake - tolerance
        if fully_filled or target_reached:
            state["status"] = "filled"
            state["order_id"] = order_key
        else:
            state["status"] = "ready"
            state["order_id"] = None
        state["request_stake"] = None
        state["requested_at"] = None

    def _partial_fill_reconcile(self, trade: Any, current_time: datetime, tag: str) -> dict[str, Any]:
        state = self._partial_fill_state(trade)
        if state["status"] == "filled":
            return state
        resolved = {str(value) for value in state.setdefault("resolved_order_ids", [])}
        orders = [
            order
            for order in (getattr(trade, "orders", ()) or ())
            if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == tag
            and self._partial_fill_order_key(order) not in resolved
        ]
        terminal = [order for order in orders if not self._partial_fill_snapshot(order)[0]]
        for order in terminal:
            self._partial_fill_apply(trade, state, order, current_time)
            if state["status"] == "filled":
                break
        if state["status"] != "filled":
            open_order = next((order for order in reversed(orders) if self._partial_fill_snapshot(order)[0]), None)
            if open_order is not None:
                self._partial_fill_apply(trade, state, open_order, current_time)
            elif not terminal and state["status"] == "requested":
                requested_at = state.get("requested_at")
                if requested_at != pd.Timestamp(current_time).isoformat():
                    state["status"] = "ready"
                    state["request_stake"] = None
                    state["requested_at"] = None
                    state["order_id"] = None
        self._partial_fill_save(trade, state)
        return state

    def _partial_fill_complete(self, trade: Any, current_time: datetime, tag: str) -> bool:
        return self._partial_fill_reconcile(trade, current_time, tag)["status"] == "filled"

    def _partial_fill_request(
        self,
        trade: Any,
        current_time: datetime,
        fraction: float,
        min_stake: float | None,
        tag: str,
    ) -> float | None:
        state = self._partial_fill_reconcile(trade, current_time, tag)
        if state["status"] != "ready" or bool(getattr(trade, "has_open_orders", False)):
            return None
        current_stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        target_stake = state.get("target_stake")
        if target_stake is None:
            target_stake = current_stake * float(fraction)
            if min_stake is not None and 0.0 < current_stake - target_stake < float(min_stake):
                target_stake = current_stake - float(min_stake)
            if target_stake <= 0.0 or target_stake >= current_stake:
                return None
            state["target_stake"] = target_stake
        target_stake = float(target_stake)
        credited_stake = max(0.0, float(state.get("credited_stake") or 0.0))
        remaining_stake = max(0.0, target_stake - credited_stake)
        tolerance = max(1e-9, target_stake * 1e-9)
        if remaining_stake <= tolerance:
            state["status"] = "filled"
            self._partial_fill_save(trade, state)
            return None
        request = min(current_stake, remaining_stake)
        if request <= 0.0 or request >= current_stake:
            return None
        state["status"] = "requested"
        state["request_stake"] = request
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        state["order_id"] = None
        self._partial_fill_save(trade, state)
        return request


    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        close, volume = _num(dataframe, "close"), _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        guard = volume.ge(volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan).mul(float(self.volume_ratio_min)))
        open_, high, low = (_num(dataframe, name) for name in ("open", "high", "low"))
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        window = int(self.pressure_window)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        guard &= (pressure * volume.fillna(0.0)).rolling(window, min_periods=max(2, window // 3)).sum().div(baseline).le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"], dataframe["enter_short"], dataframe["enter_tag"] = 0, 0, None
        lower = _num(dataframe, "pg2_descending_channel_lower").mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr").le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"], dataframe["exit_short"], dataframe["exit_tag"] = 0, 0, None
        return dataframe

    def _context(self, pair: str, trade: Any, current_time: datetime) -> tuple[DataFrame, float, float]:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
        opened = pd.Timestamp(trade.open_date_utc)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        else:
            opened = opened.tz_convert("UTC")
        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        close_times = result["date"] + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = result.loc[close_times.le(now)]
        entry_candle = pd.Timestamp(timeframe_to_prev_date(self.timeframe, opened.to_pydatetime()))
        signal = closed.loc[closed["date"].lt(entry_candle)]
        if signal.empty:
            raise RuntimeError("entry signal candle is unavailable for target-runner reconstruction")
        row = signal.iloc[-1]
        lower, upper = float(row["pg2_descending_channel_lower"]), float(row["pg2_descending_channel_upper"])
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise RuntimeError("entry signal has no valid descending-channel rails")
        post_entry = closed.loc[close_times.loc[closed.index].gt(opened)].sort_values("date")
        return post_entry, lower, upper - lower

    def _bullish_reversal(self, frame: DataFrame, mode: str) -> bool:
        if len(frame) < 2:
            return False
        latest, previous = frame.iloc[-1], frame.iloc[-2]
        if mode == "engulf":
            return bool(latest["close"] > latest["open"] and previous["close"] < previous["open"] and latest["close"] >= previous["open"] and latest["open"] <= previous["close"])
        tail = frame.tail(2)
        return bool((tail["close"] > tail["open"]).all() and tail["close"].is_monotonic_increasing)

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | tuple[float, str] | None:
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        first_multiplier, fraction, _, _, _ = self.MANAGEMENT_PLANS[str(self.management_plan.value)]
        _, lower, width = self._context(trade.pair, trade, current_time)
        target = lower - width * first_multiplier
        if target >= float(trade.open_rate) or current_rate > target:
            return None
        request = self._partial_fill_request(trade, current_time, fraction, min_stake, "measured_target_partial_1")
        return (-request, "measured_target_partial_1") if request is not None else None

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        if not self._partial_fill_complete(trade, current_time, "measured_target_partial_1"):
            return None
        _, _, final_multiplier, reversal_mode, _ = self.MANAGEMENT_PLANS[str(self.management_plan.value)]
        post_entry, lower, width = self._context(pair, trade, current_time)
        final_target = lower - width * final_multiplier
        if final_target < float(trade.open_rate) and current_rate <= final_target:
            return "downside_runner_target"
        return f"runner_bullish_{reversal_mode}" if self._bullish_reversal(post_entry, reversal_mode) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        completed = self._partial_fill_complete(trade, current_time, "measured_target_partial_1")
        hard_stop = self.MANAGEMENT_PLANS[str(self.management_plan.value)][4]
        stop_price = float(trade.open_rate) if completed else float(trade.open_rate) * (1.0 + hard_stop)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))


__all__ = ["Sieve3V2TargetPartialBreakevenRunnerFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"]
