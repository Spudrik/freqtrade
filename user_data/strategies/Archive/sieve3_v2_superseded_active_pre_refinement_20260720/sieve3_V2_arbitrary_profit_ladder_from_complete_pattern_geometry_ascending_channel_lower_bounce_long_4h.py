"""Arbitrary profit ladders for the 4h ascending-channel lower-rail bounce.

Source entry: the current executable Sieve3 fixed-TP/SL representative.
Primary trigger: bullish bounce close above pg2_ascending_channel_lower.
Primary guard: source volume/pressure guards; optional Sieve2 guards are locked off.
Target provider: fixed profit thresholds. Invalidation provider: fixed 3% stop.
Active Hyperopt parameter: ladder_plan. No branch-local inactive parameters.
The plans stay together because each is the same ordered two-partial/final-target experiment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
)
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = "entry_geometry_ascending_channel_lower_bounce_long_4h"
ENTRY_TAG = "geometry_ascending_channel_lower_bounce_long_4h"
SIDE = "long"
TIMEFRAME = "4h"

SIEVE_STAGE = 'sieve3'
SOURCE_STRATEGY = (
    "user_data/strategies/"
    "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_ascending_channel_lower_bounce_long_4h.py:"
    "Sieve3ExitFixedTpSlFromCompletePatternGeometryAscendingChannelLowerBounceLong4H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
ENTRY_SOURCE_STAGE = 'sieve2'
ENTRY_LOCK_STATUS = "current executable defaults locked; promoted parameter artifact not found"
EXIT_HYPOTHESIS = 'Test ordered arbitrary profit-taking ladders before adding channel structure.'

LADDER_PLANS = {
    "p25_at_1_5_p25_at_3_final_5": (0.015, 0.25, 0.03, 0.25, 0.05),
    "p33_at_2_p33_at_4_final_6": (0.02, 0.33, 0.04, 0.33, 0.06),
    "p50_at_2_5_p25_at_5_final_8": (0.025, 0.50, 0.05, 0.25, 0.08),
}




def tagged_exit_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:exits", "mode:sieve3_exit"))
    return param


def _num(frame: DataFrame, column: str, _default: float | Series=...) -> Series:
    if column in frame.columns:
        value = frame[column]
    elif _default is not ...:
        value = pd.Series(_default, index=frame.index)
    else:
        raise KeyError(column)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


class Sieve3V2ArbitraryProfitLadderFromCompletePatternGeometryAscendingChannelLowerBounceLong4H(
    IStrategy
):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
    EXIT_HYPOTHESIS = 'Test ordered arbitrary profit-taking ladders before adding channel structure.'
    ACTIVE_SELL_PARAMS = ('ladder_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False


    # Legacy fixed-on entry parameter: use_volume_guard=True; guard is now unconditional.
    volume_guard_window = 48
    volume_ratio_min = 1.0
    # Legacy fixed-on entry parameter: use_pressure_guard=True; guard is now unconditional.
    pressure_window = 12
    pressure_min = 0.05
    # Legacy fixed-off entry parameters: accumulation/body/close-direction guards were unreachable.

    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.55
    min_containment = 0.9
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

    ladder_plan = tagged_exit_parameter(
        CategoricalParameter(
            tuple(LADDER_PLANS),
            default="p33_at_2_p33_at_4_final_6",
            space="sell",
            optimize=True,
            load=True,
        )
    )

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
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(
            dataframe,
            timeframe=self.timeframe,
            output_slots=1,
            include_triangle_patterns=False,
            include_wedge_patterns=False,
            include_compression_patterns=False,
            include_rectangle_patterns=False,
            include_ascending_channel_patterns=True,
            include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars),
            max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr),
            squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),
            min_line_score=float(self.min_line_score),
            min_containment=float(self.min_containment),
            max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),
            channel_min_pattern_bars=int(self.channel_min_pattern_bars),
            channel_max_pattern_bars=int(self.channel_max_pattern_bars),
            channel_min_quality=float(self.channel_min_quality),
            channel_min_containment=float(self.channel_min_containment),
            channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),
            channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),
            channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),
            output_prefix="pg2",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(
            window, min_periods=max(2, window // 3)
        ).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume.fillna(0.0)).fillna(0.0)
        window = int(self.pressure_window)
        pressure_baseline = volume.rolling(
            window, min_periods=max(2, window // 3)
        ).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(
            window, min_periods=max(2, window // 3)
        ).sum() / pressure_baseline
        guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_ascending_channel_lower', 'pg2_ascending_channel_width_atr')).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        lower = _num(dataframe, 'pg2_ascending_channel_lower', np.nan)
        condition = _bool(dataframe, 'pg2_ascending_channel_pattern_present') & _num(dataframe, 'pg2_ascending_channel_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'pg2_ascending_channel_width_atr', np.nan).le(float(self.width_atr_max)) & _num(dataframe, 'low').le(lower.mul(1.0 + float(self.rail_buffer_pct))) & _num(dataframe, 'close').gt(lower) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _partial_identity(self) -> tuple[str, tuple[str, ...]]:
        return str(self.ladder_plan.value), ("arbitrary_ladder_stage_1", "arbitrary_ladder_stage_2")

    @staticmethod
    def _partial_order_key(order: Any) -> str:
        identifier = getattr(order, "order_id", None) or getattr(order, "id", None)
        if identifier is not None:
            return str(identifier)
        return "|".join(
            (
                str(getattr(order, "ft_order_tag", None) or ""),
                str(getattr(order, "order_date_utc", None) or getattr(order, "order_date", None) or ""),
                str(getattr(order, "safe_amount", 0.0) or 0.0),
                str(getattr(order, "safe_price", 0.0) or 0.0),
            )
        )

    @staticmethod
    def _partial_time_key(value: Any) -> str:
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def _partial_sync(
        self,
        trade: Any,
        current_time: Any,
        plan: str,
        tags: tuple[str, ...],
        order: Any | None = None,
    ) -> dict[str, Any]:
        key = f"sieve3_v2_partial:{type(self).__name__}"
        stored = trade.get_custom_data(key=key)
        if stored is None:
            state = {
                "version": 1,
                "plan": plan,
                "stages": [
                    {
                        "tag": tag,
                        "target_stake": None,
                        "credited_stake": 0.0,
                        "fills": {},
                        "complete": False,
                        "skipped": False,
                        "skip_reason": None,
                        "pending": False,
                        "requested_at": None,
                    }
                    for tag in tags
                ],
            }
        else:
            if not isinstance(stored, dict) or stored.get("version") != 1:
                raise ValueError("partial ladder state version mismatch")
            state = dict(stored)
            state["stages"] = [
                {**dict(stage), "fills": dict(stage.get("fills") or {})}
                for stage in state.get("stages", ())
            ]
            state_tags = tuple(str(stage.get("tag") or "") for stage in state["stages"])
            if state.get("plan") != plan or state_tags != tags:
                raise ValueError("partial ladder state does not match the selected plan")

        orders = list(getattr(trade, "orders", ()) or ())
        if order is not None:
            order_key = self._partial_order_key(order)
            if not any(self._partial_order_key(candidate) == order_key for candidate in orders):
                orders.append(order)
        exit_side = getattr(trade, "exit_side", None)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        now_key = self._partial_time_key(current_time)

        for stage in state["stages"]:
            matching = [
                candidate
                for candidate in orders
                if getattr(candidate, "ft_order_side", None) == exit_side
                and str(getattr(candidate, "ft_order_tag", None) or "") == stage["tag"]
            ]
            fills = dict(stage.get("fills") or {})
            requested_stakes: list[float] = []
            open_order = False
            terminal_order = False
            fully_filled = False
            for candidate in matching:
                candidate_key = self._partial_order_key(candidate)
                amount = float(getattr(candidate, "safe_amount", 0.0) or 0.0)
                filled = float(getattr(candidate, "safe_filled", 0.0) or 0.0)
                price = float(getattr(candidate, "safe_price", 0.0) or 0.0)
                if amount > 0.0 and price > 0.0:
                    requested_stakes.append(amount * price / leverage)
                filled_stake = filled * price / leverage if filled > 0.0 and price > 0.0 else 0.0
                if filled_stake > float(fills.get(candidate_key, 0.0) or 0.0):
                    fills[candidate_key] = filled_stake
                status = str(getattr(candidate, "status", None) or "").casefold()
                is_open = bool(getattr(candidate, "ft_is_open", False)) and status not in NON_OPEN_EXCHANGE_STATES
                open_order = open_order or is_open
                if not is_open:
                    terminal_order = True
                    remaining = float(getattr(candidate, "safe_remaining", max(0.0, amount - filled)) or 0.0)
                    tolerance = max(1e-12, amount * 1e-6)
                    fully_filled = fully_filled or (
                        filled > 0.0
                        and (
                            (amount > 0.0 and filled >= amount - tolerance)
                            or remaining <= tolerance
                        )
                    )

            stage["fills"] = fills
            stage["credited_stake"] = sum(float(value) for value in fills.values())
            if stage.get("target_stake") is None and requested_stakes:
                stage["target_stake"] = requested_stakes[0]
            target = float(stage.get("target_stake") or 0.0)
            complete = bool(stage.get("complete")) or fully_filled
            if target > 0.0:
                complete = complete or stage["credited_stake"] >= target - max(1e-8, target * 1e-6)
            stage["complete"] = complete
            if complete:
                stage["pending"] = False
            elif open_order:
                stage["pending"] = True
            elif terminal_order or stage.get("requested_at") != now_key:
                stage["pending"] = False

        if stored != state:
            trade.set_custom_data(key=key, value=state)
        return state

    def _partial_request(
        self,
        trade: Any,
        current_time: Any,
        state: dict[str, Any],
        stage_index: int,
        current_stake: float,
        fraction: float,
        min_stake: float | None = None,
        enforce_minimum: bool = False,
        preserve_remainder_minimum: bool = False,
    ) -> float | None:
        stage = state["stages"][stage_index]
        if stage["complete"] or stage.get("skipped") or stage["pending"]:
            return None
        if stage.get("target_stake") is None:
            stage["target_stake"] = current_stake * fraction
        target = float(stage["target_stake"])
        remaining = max(0.0, target - float(stage.get("credited_stake") or 0.0))
        if remaining <= max(1e-8, target * 1e-6):
            stage["complete"] = True
            trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
            return None
        request = min(current_stake, remaining)
        skip_reason = None
        if request <= 0.0:
            skip_reason = "nonpositive_reduction"
        elif request >= current_stake:
            skip_reason = "partial_would_close_position"
        elif enforce_minimum and min_stake is not None and request < float(min_stake):
            skip_reason = "reduction_below_min_stake"
        elif (
            preserve_remainder_minimum
            and min_stake is not None
            and 0.0 < current_stake - request < float(min_stake)
        ):
            skip_reason = "remainder_below_min_stake"
        if skip_reason is not None:
            stage["skipped"] = True
            stage["skip_reason"] = skip_reason
            stage["pending"] = False
            trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
            return None
        stage["pending"] = True
        stage["requested_at"] = self._partial_time_key(current_time)
        trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
        return request

    @staticmethod
    def _partials_complete(state: dict[str, Any]) -> bool:
        return bool(state["stages"]) and all(
            bool(stage["complete"]) or bool(stage.get("skipped"))
            for stage in state["stages"]
        )

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = (pair, kwargs)
        if getattr(order, "ft_order_side", None) != getattr(trade, "exit_side", None):
            return None
        plan, tags = self._partial_identity()
        self._partial_sync(trade, current_time, plan, tags, order)
        return None


    def adjust_trade_position(
        self, trade: Any, current_time: datetime, current_rate: float, current_profit: float,
        min_stake: float | None, max_stake: float, current_entry_rate: float,
        current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
        **kwargs: Any,
    ) -> tuple[float, str] | None:
        _ = (current_rate, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        if bool(trade.has_open_orders):
            return None
        stage_index = next((index for index, stage in enumerate(state["stages"]) if not stage["complete"] and not stage.get("skipped")), None)
        if stage_index is None:
            return None
        first_profit, first_fraction, second_profit, second_fraction, _ = LADDER_PLANS[plan]
        trigger, fraction = ((first_profit, first_fraction) if stage_index == 0 else (second_profit, second_fraction))
        if current_profit < trigger:
            return None
        stake = float(trade.stake_amount)
        if stake <= 0.0:
            raise ValueError("profit ladder requires positive remaining stake")
        request = self._partial_request(
            trade, current_time, state, stage_index, stake, fraction,
            min_stake, enforce_minimum=True, preserve_remainder_minimum=True,
        )
        return (-request, tags[stage_index]) if request is not None else None

    def custom_exit(
        self, pair: str, trade: Any, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs: Any,
    ) -> str | None:
        _ = (pair, current_rate, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        final_profit = LADDER_PLANS[plan][4]
        if self._partials_complete(state) and current_profit >= final_profit:
            return "arbitrary_ladder_final_target"
        return None


ACTIVE_SELL_PARAMS = ('ladder_plan',)
