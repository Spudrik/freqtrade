from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_MODE = "entry_choch_bull_reversal_long_4h"


TIMEFRAME = "4h"
ENTRY_LOCK_STATUS = "verified_archived_snapshot_defaults_plus_selected_buy_params"
EXIT_THESIS = "Take two ordered profit partials, move the remainder stop to entry, then use a final profit target."
INVALIDATION = "Fixed 3% initial stop; exactly trade.open_rate after the first filled partial."

LADDER_PLANS = {
    "balanced_p25_2_p33_4_final7": (0.02, 0.25, 0.04, 0.33, 0.07),
    "frontload_p50_2_p50_3_final6": (0.02, 0.50, 0.03, 0.50, 0.06),
    "extended_p25_3_p33_6_final10": (0.03, 0.25, 0.06, 0.33, 0.10),
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


EXIT_FAMILY = 'profit_ladder'
PRIMARY_TRIGGER = 'confirmed 4h ms_choch_to_bull'
PRIMARY_GUARD = 'non-negative ms_state plus locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'named two-partial profit ladder and final objective'
PRIMARY_INVALIDATION = 'source stop shifts to exact entry after the first partial'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'filename-derived source foundation: choch_bull_reversal_long_4h'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_ladder'
EXIT_HYPOTHESIS = 'Ordered profit partials may de-risk the 4h bullish CHoCH reversal while preserving a final runner objective.'
ACTIVE_SELL_PARAMS = ('ladder_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'choch_bull_reversal_long_4h'

class Sieve3V2ProfitLadderFromChochBullReversalLong4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
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

    ladder_plan = CategoricalParameter(tuple(LADDER_PLANS), default="balanced_p25_2_p33_4_final7", space="sell", optimize=True, load=True)
    ladder_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('ladder_plan',)

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
        trade,
        current_time,
        current_rate,
        current_profit,
        min_stake,
        max_stake,
        current_entry_rate,
        current_exit_rate,
        current_entry_profit,
        current_exit_profit,
        **kwargs,
    ):
        _ = (current_rate, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        plan_name = str(self.ladder_plan.value)
        first_profit, first_fraction, second_profit, second_fraction, _ = LADDER_PLANS[plan_name]
        steps = ((first_profit, first_fraction), (second_profit, second_fraction))
        return self._ladder_request(trade, current_time, current_profit, min_stake, plan_name, steps)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        if self._ladder_completed(trade, current_time, str(self.ladder_plan.value), 2) < 1:
            return None
        return stoploss_from_absolute(
            float(trade.open_rate),
            current_rate,
            is_short=False,
            leverage=float(trade.leverage or 1.0),
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> str | None:
        _ = (pair, current_time, current_rate, kwargs)
        final_profit = LADDER_PLANS[str(self.ladder_plan.value)][4]
        if self._ladder_completed(trade, current_time, str(self.ladder_plan.value), 2) >= 2 and current_profit >= final_profit:
            return "profit_ladder_final"
        return None
