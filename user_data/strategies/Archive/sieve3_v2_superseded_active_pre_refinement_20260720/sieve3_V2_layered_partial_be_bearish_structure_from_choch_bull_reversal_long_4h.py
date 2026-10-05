from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_choch_bull_reversal_long_4h"
ENTRY_TAG = "choch_bull_reversal_long_4h"
SIDE = "long"
TIMEFRAME = "4h"
ENTRY_LOCK_STATUS = "verified_archived_snapshot_defaults_plus_selected_buy_params"
EXIT_THESIS = "Take one profit partial, move the remainder stop exactly to entry, then exit on bearish structure or the final target."
INVALIDATION = "Fixed 2% initial stop; exactly trade.open_rate after the filled partial."

LAYERED_PLANS = {
    "p33_at2_be_choch_or6": (0.02, 0.33, "choch", 0.06),
    "p50_at3_be_choch_or8": (0.03, 0.50, "choch", 0.08),
    "p33_at2_be_state_two_or8": (0.02, 0.33, "state_two", 0.08),
}
PARTIAL_TAG = "layered_profit_partial"
STATE_VERSION = 2
STATE_KEY = "sieve3_v2:layered_partial_be_bearish_structure"
TERMINAL_PARTIAL_STATUSES = frozenset(
    {"canceled", "cancelled", "closed", "expired", "failed", "rejected"}
)

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


ACTIVE_SELL_PARAMS = ("layered_plan",)


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_layered_partial_be_bearish_structure"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_choch_bull_reversal_long_4h"
EXIT_HYPOTHESIS = "Test the layered partial be bearish structure exit family while preserving the choch bull reversal long 4h entry behavior."


class Sieve3V2LayeredPartialBeBearishStructureFromChochBullReversalLong4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("layered_plan",)

    minimal_roi = {"0": 100.0}
    stoploss = -0.02
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
    # Legacy fixed-off entry parameter: use_accumulation_guard=False; its branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    layered_plan = CategoricalParameter(tuple(LAYERED_PLANS), default="p33_at2_be_choch_or6", space="sell", optimize=True, load=True)
    layered_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        if self.use_volume_guard:
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if self.use_pressure_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if self.use_pressure_guard:
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
        if self.use_state_guard:
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

    @staticmethod
    def _layered_state(trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=STATE_KEY)
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            return None
        normalized = dict(state)
        normalized["credited_order_stakes"] = dict(
            normalized.get("credited_order_stakes") or {}
        )
        normalized["resolved_order_ids"] = list(
            normalized.get("resolved_order_ids") or []
        )
        return normalized

    @staticmethod
    def _save_layered_state(trade: Any, state: dict[str, Any]) -> None:
        trade.set_custom_data(key=STATE_KEY, value=dict(state))

    def _ensure_layered_state(self, trade: Any) -> dict[str, Any]:
        state = self._layered_state(trade)
        if state is None:
            state = {
                "version": STATE_VERSION,
                "plan": str(self.layered_plan.value),
                "initial_stake": float(getattr(trade, "stake_amount", 0.0) or 0.0),
                "target_stake": None,
                "requested_stake": None,
                "requested_at": None,
                "credited_stake": 0.0,
                "credited_order_stakes": {},
                "resolved_order_ids": [],
                "status": "ready",
                "partial_complete": False,
                "partial_filled_at": None,
                "active_order_id": None,
            }
            self._save_layered_state(trade, state)
        return state

    @staticmethod
    def _layered_order_key(order: Any) -> str:
        order_id = str(getattr(order, "order_id", None) or "")
        if not order_id:
            raise ValueError("layered partial reconciliation requires an order id")
        return order_id

    @staticmethod
    def _trade_leverage(trade: Any) -> float:
        leverage = float(trade.leverage)
        if not np.isfinite(leverage) or leverage <= 0.0:
            raise ValueError("layered partial management requires positive finite trade leverage")
        return leverage

    @classmethod
    def _layered_order_stake(cls, trade: Any, order: Any) -> float:
        filled = float(getattr(order, "safe_filled", 0.0) or 0.0)
        fill_price = float(getattr(order, "safe_price", 0.0) or 0.0)
        if filled < 0.0 or fill_price < 0.0:
            raise ValueError("layered partial order fill values cannot be negative")
        return filled * fill_price / cls._trade_leverage(trade)

    @staticmethod
    def _layered_order_is_open(order: Any) -> bool:
        status = str(getattr(order, "status", None) or "").casefold()
        return bool(getattr(order, "ft_is_open", False)) and status not in TERMINAL_PARTIAL_STATUSES

    def _credit_layered_order(self, trade: Any, state: dict[str, Any], order: Any, current_time: Any) -> None:
        key = self._layered_order_key(order)
        credited_for_order = self._layered_order_stake(trade, order)
        credits = state.setdefault("credited_order_stakes", {})
        previous_credit = float(credits.get(key) or 0.0)
        if credited_for_order > previous_credit:
            state["credited_stake"] = float(state.get("credited_stake") or 0.0) + credited_for_order - previous_credit
            credits[key] = credited_for_order

        target_stake = state.get("target_stake")
        credited_stake = float(state.get("credited_stake") or 0.0)
        target_filled = target_stake is not None and credited_stake >= float(target_stake) - max(
            1e-8, float(target_stake) * 1e-9
        )
        is_open = self._layered_order_is_open(order)
        if target_filled:
            state["status"] = "filled"
            state["partial_complete"] = True
            state["active_order_id"] = None
            state["requested_stake"] = None
            state["requested_at"] = None
            if state.get("partial_filled_at") is None:
                filled_at = getattr(order, "order_filled_utc", None) or current_time
                state["partial_filled_at"] = pd.Timestamp(filled_at).isoformat()
            return

        state["partial_complete"] = False
        if is_open:
            state["status"] = "open"
            state["active_order_id"] = key
            return

        resolved = state.setdefault("resolved_order_ids", [])
        if key not in resolved:
            resolved.append(key)
        state["status"] = "partially_filled" if credited_stake > 0.0 else "ready"
        state["active_order_id"] = None
        state["requested_stake"] = None
        state["requested_at"] = None

    def _reconcile_layered_orders(
        self,
        trade: Any,
        state: dict[str, Any],
        current_time: Any,
        extra_order: Any | None = None,
    ) -> None:
        before = repr(state)
        exit_side = getattr(trade, "exit_side", None)
        orders = list(getattr(trade, "orders", ()) or ())
        if extra_order is not None and all(order is not extra_order for order in orders):
            orders.append(extra_order)
        matching = [
            order
            for order in orders
            if getattr(order, "ft_order_side", None) == exit_side
            and str(getattr(order, "ft_order_tag", None) or "") == PARTIAL_TAG
        ]
        for order in matching:
            self._credit_layered_order(trade, state, order, current_time)
        if state.get("partial_complete"):
            if repr(state) != before:
                self._save_layered_state(trade, state)
            return

        open_order = next(
            (order for order in reversed(matching) if self._layered_order_is_open(order)),
            None,
        )
        if open_order is not None:
            state["status"] = "open"
            state["active_order_id"] = self._layered_order_key(open_order)
        elif state.get("status") == "requested":
            requested_at = state.get("requested_at")
            now = pd.Timestamp(current_time).isoformat()
            if requested_at != now:
                credited = float(state.get("credited_stake") or 0.0)
                state["status"] = "partially_filled" if credited > 0.0 else "ready"
                state["active_order_id"] = None
                state["requested_stake"] = None
                state["requested_at"] = None
        if repr(state) != before:
            self._save_layered_state(trade, state)

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = pair, kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None):
            self._ensure_layered_state(trade)
            return None
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == PARTIAL_TAG
        ):
            state = self._ensure_layered_state(trade)
            self._reconcile_layered_orders(trade, state, current_time, order)
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
        state = self._ensure_layered_state(trade)
        self._reconcile_layered_orders(trade, state, current_time)
        if trade.has_open_orders or state.get("status") in {"requested", "open", "filled"}:
            return None
        partial_profit, partial_fraction, _, _ = LAYERED_PLANS[str(state["plan"])]
        if current_profit < partial_profit:
            return None
        current_stake = float(trade.stake_amount)
        target_stake = state.get("target_stake")
        new_target = target_stake is None
        if new_target:
            amount = min(current_stake, float(state["initial_stake"]) * partial_fraction)
        else:
            amount = min(current_stake, max(0.0, float(target_stake) - float(state.get("credited_stake") or 0.0)))
        if min_stake is not None:
            minimum = float(min_stake)
            if amount < minimum:
                return None
            if 0.0 < current_stake - amount < minimum:
                amount = current_stake - minimum
            if amount < minimum:
                return None
        if amount <= 0.0 or amount >= current_stake:
            return None
        if new_target:
            state["target_stake"] = amount
        state["status"] = "requested"
        state["requested_stake"] = amount
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        state["active_order_id"] = None
        self._save_layered_state(trade, state)
        return -amount, PARTIAL_TAG

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        state = self._ensure_layered_state(trade)
        self._reconcile_layered_orders(trade, state, current_time)
        if not state.get("partial_complete"):
            return None
        return stoploss_from_absolute(
            float(trade.open_rate),
            current_rate,
            is_short=False,
            leverage=self._trade_leverage(trade),
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> str | None:
        _ = (current_rate, kwargs)
        state = self._ensure_layered_state(trade)
        self._reconcile_layered_orders(trade, state, current_time)
        if not state.get("partial_complete"):
            return None
        _, _, reversal_mode, final_profit = LAYERED_PLANS[str(state["plan"])]
        if current_profit >= final_profit:
            return "layered_final_profit"
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if reversal_mode == "choch" and bool(_bool(dataframe, "ms_choch_to_bear").iloc[-1]):
            return "layered_bearish_choch"
        if reversal_mode == "state_two" and bool(_num(dataframe, "ms_state").tail(2).lt(0).all()):
            return "layered_bearish_state_two"
        return None
