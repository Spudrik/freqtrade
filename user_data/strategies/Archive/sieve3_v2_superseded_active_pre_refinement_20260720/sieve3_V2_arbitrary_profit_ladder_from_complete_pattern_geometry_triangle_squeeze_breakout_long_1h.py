"""Arbitrary percentage-profit ladders for the frozen triangle breakout entry."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.strategy import CategoricalParameter, IntParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_TAG = "geometry_triangle_squeeze_breakout_long_1h"
def _num(frame: DataFrame, column: str, _default: float | Series=...) -> Series:
    if column in frame.columns:
        value = frame[column]
    elif _default is not ...:
        value = pd.Series(_default, index=frame.index)
    else:
        raise KeyError(column)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


SIEVE_STAGE = 'sieve3'
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
EXIT_HYPOTHESIS = 'Compare ordered fixed-profit scale-out plans while preserving the locked entry.'
ACTIVE_SELL_PARAMS = ('ladder_plan', 'hard_stop_percent')
class Sieve3V2ArbitraryProfitLadderFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
    EXIT_HYPOTHESIS = 'Compare ordered fixed-profit scale-out plans while preserving the locked entry.'
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    trailing_stop = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.35
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.5
    min_containment = 0.88

    ladder_plan = CategoricalParameter(
        [
            "rem25_at2_rem33_at4_final8", "rem25_at2_rem33_at5_final10",
            "rem25_at3_rem33_at6_final12", "rem33_at2_rem50_at5_final8",
            "rem33_at3_rem50_at6_final10", "rem33_at4_rem50_at8_final12",
            "rem50_at2_rem50_at4_final6", "rem50_at3_rem50_at6_final9",
            "rem50_at4_rem50_at8_final12", "rem20_at1_rem25_at3_final6",
            "rem20_at2_rem25_at5_final10", "rem25_at4_rem33_at8_final16",
            "rem33_at5_rem50_at10_final20", "rem50_at5_rem50_at10_final15",
            "rem25_at6_rem33_at12_final24", "rem33_at8_rem50_at16_final30",
        ],
        default="rem25_at2_rem33_at5_final10", space="sell", optimize=True, load=True,
    )
    ladder_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 8, default=3, space="sell", optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')

    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ('ladder_plan', 'hard_stop_percent')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
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
            open_ = _num(dataframe, "open"); high = _num(dataframe, "high"); low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            denominator = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().div(denominator).ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_triangle_upper',)).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'pg2_triangle_pattern_present') & _bool(dataframe, 'pg2_triangle_squeeze_active') & _num(dataframe, 'pg2_triangle_indicator_score').ge(float(self.min_line_score)) & _num(dataframe, 'pg2_triangle_direction').ge(0) & _num(dataframe, 'close').gt(_num(dataframe, 'pg2_triangle_upper', np.nan))
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None
        return dataframe

    def _ladder(self) -> tuple[float, float, float, float, float]:
        parts = str(self.ladder_plan.value).split("_")
        if len(parts) != 5 or not parts[0].startswith("rem") or not parts[1].startswith("at") or not parts[2].startswith("rem") or not parts[3].startswith("at") or not parts[4].startswith("final"):
            raise ValueError(f"invalid arbitrary ladder plan: {self.ladder_plan.value}")
        first_fraction = parts[0].removeprefix("rem")
        first_profit = parts[1].removeprefix("at")
        second_fraction = parts[2].removeprefix("rem")
        second_profit = parts[3].removeprefix("at")
        final_profit = parts[4].removeprefix("final")
        return float(first_fraction) / 100.0, float(first_profit) / 100.0, float(second_fraction) / 100.0, float(second_profit) / 100.0, float(final_profit) / 100.0

    def _partial_identity(self) -> tuple[str, tuple[str, ...]]:
        return str(self.ladder_plan.value), ("arbitrary_ladder_partial_1", "arbitrary_ladder_partial_2")

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
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        stage_index = next((index for index, stage in enumerate(state["stages"]) if not stage["complete"] and not stage.get("skipped")), None)
        if stage_index is None:
            return None
        first_fraction, first_profit, second_fraction, second_profit, _ = self._ladder()
        trigger, fraction = ((first_profit, first_fraction) if stage_index == 0 else (second_profit, second_fraction))
        if current_profit < trigger:
            return None
        stake = float(trade.stake_amount)
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
        return "arbitrary_ladder_final" if self._partials_complete(state) and current_profit >= self._ladder()[4] else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        hard_stop = int(self.hard_stop_percent.value) * 0.01
        stop_price = float(trade.open_rate) * (1.0 - hard_stop)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))
