"""Arbitrary fixed-profit ladder for the promoted bearish 1h BOS entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: fixed-percentage partial ladder
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: fixed profit percentages
Invalidation provider: fixed 3% initial stop
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category fixes steps, remaining-stake fractions, and final target
Split rationale: every mode follows the same ordered fixed-profit realization sequence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_MODE = "entry_bos_bear_continuation_short_1h"


TIMEFRAME = "1h"


SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_profit_ladder'
ENTRY_SOURCE_STAGE = 'sieve2'


PROFIT_LADDER_PLANS: dict[str, dict[str, Any]] = {
    "p25_at_1_5_final_4": {"steps": ((0.015, 0.25),), "final": 0.04, "breakeven": False},
    "p33_at_2_final_5": {"steps": ((0.02, 0.33),), "final": 0.05, "breakeven": False},
    "p50_at_2_final_5": {"steps": ((0.02, 0.50),), "final": 0.05, "breakeven": False},
    "p33_at_3_final_6": {"steps": ((0.03, 0.33),), "final": 0.06, "breakeven": False},
    "p50_at_3_final_8": {"steps": ((0.03, 0.50),), "final": 0.08, "breakeven": False},
    "p66_at_4_final_10": {"steps": ((0.04, 0.66),), "final": 0.10, "breakeven": False},
    "p25_at_1_5_be_final_5": {"steps": ((0.015, 0.25),), "final": 0.05, "breakeven": True},
    "p33_at_2_be_final_6": {"steps": ((0.02, 0.33),), "final": 0.06, "breakeven": True},
    "p50_at_3_be_final_8": {"steps": ((0.03, 0.50),), "final": 0.08, "breakeven": True},
    "p50_at_4_be_final_10": {"steps": ((0.04, 0.50),), "final": 0.10, "breakeven": True},
    "p25_at_2_then_p33_remaining_at_5_final_8": {
        "steps": ((0.02, 0.25), (0.05, 0.33)), "final": 0.08, "breakeven": False
    },
    "p33_at_2_then_p50_remaining_at_6_final_10": {
        "steps": ((0.02, 0.33), (0.06, 0.50)), "final": 0.10, "breakeven": False
    },
    "p50_at_3_then_p50_remaining_at_8_final_12": {
        "steps": ((0.03, 0.50), (0.08, 0.50)), "final": 0.12, "breakeven": False
    },
    "p25_at_2_be_then_p33_remaining_at_5_final_8": {
        "steps": ((0.02, 0.25), (0.05, 0.33)), "final": 0.08, "breakeven": True
    },
    "p33_at_2_be_then_p50_remaining_at_6_final_10": {
        "steps": ((0.02, 0.33), (0.06, 0.50)), "final": 0.10, "breakeven": True
    },
}




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'profit_ladder'
PRIMARY_TRIGGER = 'confirmed 1h ms_bos_to_bear'
PRIMARY_GUARD = 'locked volume and bearish-pressure guards'
PRIMARY_TARGET = 'named fixed-profit ladder stages'
PRIMARY_INVALIDATION = 'fixed 3% initial stop with optional breakeven after the first partial'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Ordered fixed-profit partials may realize gains while retaining a larger short runner.'
ACTIVE_SELL_PARAMS = ('exit_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'bos_bear_continuation_short_1h'

class Sieve3V2ProfitLadderFromBosBearContinuationShort1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    minimal_roi: dict[str, float] = {}
    stoploss = -0.03
    use_exit_signal = False
    use_custom_roi = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    # Legacy fixed-off entry parameter: use_state_guard=False (BooleanParameter, buy space).

    exit_plan = CategoricalParameter(tuple(PROFIT_LADDER_PLANS), default="p33_at_2_be_final_6", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('exit_plan',)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(dataframe, strength=int(self.strength), min_prominence_atr=float(self.min_prominence_atr), min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), max_pivot_age_bars=int(self.max_pivot_age_bars), breakout_buffer_atr=float(self.breakout_buffer_atr), prefix="ms")
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
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_bos_to_bear")
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def custom_roi(self, pair: str, trade: Any, current_time: datetime, trade_duration: int, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, trade, current_time, trade_duration, entry_tag, side, kwargs
        return float(PROFIT_LADDER_PLANS[str(self.exit_plan.value)]["final"])

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = pair, current_time, current_profit, after_fill, kwargs
        plan = PROFIT_LADDER_PLANS[str(self.exit_plan.value)]
        stop_price = float(trade.open_rate) * 1.03
        if bool(plan["breakeven"]) and self._ladder_completed(
            trade, current_time, str(self.exit_plan.value), len(plan["steps"])
        ) >= 1:
            stop_price = float(trade.open_rate)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def _ladder_state_key(self) -> str:
        return f"sieve3_v2_ladder:{self.__class__.__name__}"

    def _ladder_load_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self._ladder_state_key())
        if state is None:
            return None
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("profit ladder trade state is invalid")
        return dict(state)

    def _ladder_state(self, trade: Any, plan: str, stage_count: int) -> dict[str, Any]:
        state = self._ladder_load_state(trade)
        if state is None:
            state = {
                "version": 1,
                "plan": plan,
                "stages": [
                    {
                        "status": "ready",
                        "tag": f"s3v2_ladder_partial_{index + 1}",
                        "target_stake": None,
                        "credited_stake": 0.0,
                        "request_stake": None,
                        "requested_at": None,
                        "order_id": None,
                        "resolved_order_ids": [],
                    }
                    for index in range(stage_count)
                ],
            }
            trade.set_custom_data(key=self._ladder_state_key(), value=state)
        if state.get("plan") != plan or len(state.get("stages", ())) != stage_count:
            raise ValueError("profit ladder state does not match the selected plan")
        return state

    @staticmethod
    def _ladder_reset_stage(stage: dict[str, Any]) -> None:
        stage["status"] = "ready"
        stage["request_stake"] = None
        stage["requested_at"] = None
        stage["order_id"] = None

    def _ladder_apply_order(self, stage: dict[str, Any], order: Any) -> None:
        order_id = str(getattr(order, "order_id", None) or "")
        if not order_id:
            raise ValueError("profit ladder exit order requires an order_id")
        resolved = stage.setdefault("resolved_order_ids", [])
        if order_id in {str(value) for value in resolved}:
            return
        str(getattr(order, "status", None) or "").casefold()
        if bool(getattr(order, "ft_is_open", False)):
            stage["status"] = "requested"
            stage["order_id"] = order_id
            return
        filled = float(getattr(order, "safe_filled", 0.0) or 0.0)
        filled_stake = float(getattr(order, "stake_amount_filled", 0.0) or 0.0)
        if filled > 0.0 and filled_stake <= 0.0:
            raise ValueError("profit ladder fill requires positive filled stake")
        if filled_stake > 0.0:
            stage["credited_stake"] = float(stage.get("credited_stake") or 0.0) + filled_stake
        resolved.append(order_id)
        safe_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
        fully_filled = (
            filled > 0.0
            and safe_amount > 0.0
            and filled >= safe_amount - max(1e-12, abs(safe_amount) * 1e-9)
        )
        target = float(stage.get("target_stake") or 0.0)
        target_credited = target > 0.0 and float(stage.get("credited_stake") or 0.0) >= target - 1e-12
        if fully_filled or target_credited:
            stage["status"] = "filled"
            stage["request_stake"] = None
            stage["order_id"] = order_id
            return
        self._ladder_reset_stage(stage)

    def _ladder_reconcile(self, trade: Any, state: dict[str, Any], current_time: datetime) -> None:
        orders = tuple(getattr(trade, "orders", ()) or ())
        exit_side = getattr(trade, "exit_side", None)
        now = current_time.isoformat()
        for stage in state["stages"]:
            if stage.get("status") == "filled":
                continue
            resolved = {str(value) for value in stage.setdefault("resolved_order_ids", [])}
            matching = [
                order
                for order in orders
                if getattr(order, "ft_order_side", None) == exit_side
                and str(getattr(order, "ft_order_tag", None) or "") == str(stage["tag"])
                and str(getattr(order, "order_id", None) or "") not in resolved
            ]
            for order in matching:
                if not bool(getattr(order, "ft_is_open", False)):
                    self._ladder_apply_order(stage, order)
                    if stage.get("status") == "filled":
                        break
            if stage.get("status") == "filled":
                continue
            resolved = {str(value) for value in stage.setdefault("resolved_order_ids", [])}
            open_order = next(
                (
                    order
                    for order in reversed(orders)
                    if getattr(order, "ft_order_side", None) == exit_side
                    and str(getattr(order, "ft_order_tag", None) or "") == str(stage["tag"])
                    and str(getattr(order, "order_id", None) or "") not in resolved
                    and bool(getattr(order, "ft_is_open", False))
                ),
                None,
            )
            if open_order is not None:
                self._ladder_apply_order(stage, open_order)
            elif stage.get("status") == "requested" and stage.get("requested_at") != now:
                self._ladder_reset_stage(stage)
        trade.set_custom_data(key=self._ladder_state_key(), value=state)

    def _ladder_request(
        self,
        trade: Any,
        current_time: datetime,
        current_profit: float,
        min_stake: float | None,
        plan: str,
        steps: tuple[tuple[float, float], ...],
    ) -> tuple[float, str] | None:
        state = self._ladder_state(trade, plan, len(steps))
        self._ladder_reconcile(trade, state, current_time)
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        for index, (trigger, fraction) in enumerate(steps):
            stage = state["stages"][index]
            if stage["status"] == "filled":
                continue
            if stage["status"] == "requested" or current_profit < float(trigger):
                return None
            stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
            if stake <= 0.0:
                return None
            target = stage.get("target_stake")
            if target is None:
                target = stake * float(fraction)
                stage["target_stake"] = target
            request = min(stake, max(0.0, float(target) - float(stage.get("credited_stake") or 0.0)))
            if request <= 0.0 or request >= stake:
                return None
            if min_stake is not None and (
                request < float(min_stake) or stake - request < float(min_stake)
            ):
                return None
            stage["status"] = "requested"
            stage["request_stake"] = request
            stage["requested_at"] = current_time.isoformat()
            stage["order_id"] = None
            trade.set_custom_data(key=self._ladder_state_key(), value=state)
            return -request, str(stage["tag"])
        return None

    def _ladder_completed(
        self,
        trade: Any,
        current_time: datetime,
        plan: str,
        stage_count: int,
    ) -> int:
        state = self._ladder_state(trade, plan, stage_count)
        self._ladder_reconcile(trade, state, current_time)
        return sum(stage.get("status") == "filled" for stage in state["stages"])

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = pair, kwargs
        if getattr(order, "ft_order_side", None) != getattr(trade, "exit_side", None):
            return None
        state = self._ladder_load_state(trade)
        if state is None:
            return None
        stage = next(
            (
                candidate
                for candidate in state["stages"]
                if str(candidate.get("tag") or "") == str(getattr(order, "ft_order_tag", None) or "")
            ),
            None,
        )
        if stage is None:
            return None
        self._ladder_apply_order(stage, order)
        trade.set_custom_data(key=self._ladder_state_key(), value=state)
        return None

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
        _ = current_rate, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        plan_name = str(self.exit_plan.value)
        steps = tuple(PROFIT_LADDER_PLANS[plan_name]["steps"])
        return self._ladder_request(trade, current_time, current_profit, min_stake, plan_name, steps)
