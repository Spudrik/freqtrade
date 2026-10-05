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
ENTRY_TAG = "choch_bull_reversal_long_8h"
SIDE = "long"
TIMEFRAME = "8h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = (
    "user_data/strategies/sieve3_exit_fixed_tp_sl_from_choch_bull_reversal_long_8h.py:"
    "Sieve3ExitFixedTpSlFromChochBullReversalLong8H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = 'sieve3_exit_structure_target_partial_be_bearish_structure_remainder'
ENTRY_SOURCE_STAGE = 'sieve2'
ENTRY_LOCK_STATUS = "legacy_executable_defaults_locked; promoted_params_overlay_missing"
EXIT_HYPOTHESIS = (
    "Take one partial at the prior bearish swing high, stop the remainder exactly "
    "at entry, then exit it on bearish structure or extension."
)

# Primary trigger: confirmed ms_choch_to_bull event.
# Primary guard: locked 12-candle volume >= 1.3x its shifted baseline.
# Target: entry-time ms_prev_swing_high; invalidation: 4% initial stop, then exact
# open-rate stop after partial.
# Active sell parameter: layer_plan. Partial, BE, remainder event, and extension
# are one ordered named plan.
# Kept as the sole layered file because it tests a distinct target -> partial ->
# BE -> remainder sequence.




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


EXIT_FAMILY = 'structure_target_partial_be_bearish_structure_remainder'




class Sieve3V2StructureTargetPartialBeBearishStructureRemainderFromChochBullReversalLong8H(
    IStrategy
):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False
    exit_profit_only = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.3
    use_pressure_guard = False
    pressure_window = 12
    pressure_min = 0.1
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.15
    use_state_guard = False

    LAYER_PLANS = {
        "p33_at_target_be_choch_or_6pct": (0.00, 0.01, 0.33, "choch", 0.06),
        "p50_at_target_be_choch_or_6pct": (0.00, 0.01, 0.50, "choch", 0.06),
        "p33_1pct_before_target_be_choch_or_6pct": (0.01, 0.02, 0.33, "choch", 0.06),
        "p50_1pct_before_target_be_choch_or_8pct": (0.01, 0.02, 0.50, "choch", 0.08),
        "p33_at_target_be_bos_or_8pct": (0.00, 0.01, 0.33, "bos", 0.08),
        "p50_at_target_be_either_or_10pct": (0.00, 0.02, 0.50, "either", 0.10),
    }
    layer_plan = CategoricalParameter(
        tuple(LAYER_PLANS),
        default="p33_at_target_be_choch_or_6pct",
        space="sell",
        optimize=True,
        load=True,
    )
    layer_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ("layer_plan",)

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
        close = _num(dataframe, "close")
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
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(
                -1.0, 1.0
            )
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = (
                volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            )
            pressure_ratio = (
                directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
                / pressure_baseline
            )
            guard &= pressure_ratio.ge(float(self.pressure_min))
        if bool(self.use_accumulation_guard):
            window = int(self.pressure_window)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.ge(0.0)
        if bool(self.use_body_direction_guard):
            guard &= close.gt(_num(dataframe, "open"))
        if bool(self.use_close_direction_guard):
            guard &= close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_choch_to_bull")
        if bool(self.use_state_guard):
            condition &= _num(dataframe, "ms_state").ge(0)
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
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
            raise RuntimeError("layered structure exit requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "high", "ms_prev_swing_high", "ms_choch_to_bear", "ms_bos_to_bear"}
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise KeyError(f"layered structure exit missing columns: {missing}")
        cutoff = pd.Timestamp(cutoff_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return dataframe.loc[close_times.le(cutoff)]

    def _entry_target(self, pair: str, trade: Any, min_distance: float) -> float | None:
        rows = self._closed_rows(pair, trade.open_date_utc)
        if rows.empty:
            raise RuntimeError(
                "no closed entry-signal candle is available for layered structure exit"
            )
        target = float(rows.iloc[-1]["ms_prev_swing_high"])
        minimum = float(trade.open_rate) * (1.0 + min_distance)
        return target if np.isfinite(target) and target >= minimum else None

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
    ) -> float | None | tuple[float | None, str | None]:
        _ = (
            current_profit,
            min_stake,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            kwargs,
        )
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        early_offset, min_distance, fraction, _, _ = self.LAYER_PLANS[str(self.layer_plan.value)]
        target = self._entry_target(str(trade.pair), trade, min_distance)
        if target is None:
            return None
        rows = self._closed_rows(str(trade.pair), current_time)
        opened = pd.Timestamp(trade.open_date_utc)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        close_times = pd.to_datetime(rows["date"], utc=True, errors="raise") + pd.to_timedelta(
            timeframe_to_minutes(self.timeframe), unit="m"
        )
        rows = rows.loc[close_times.gt(opened)]
        if rows.empty:
            return None
        trigger_rate = target * (1.0 - early_offset)
        touched = pd.to_numeric(rows["high"], errors="coerce").max() >= trigger_rate
        if not touched:
            return None
        tag = f"structure_target_partial_{self.layer_plan.value}"
        request = self._partial_fill_request(trade, current_time, fraction, min_stake, tag)
        return (-request, tag) if request is not None else None

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
        tag = f"structure_target_partial_{self.layer_plan.value}"
        if not self._partial_fill_complete(trade, current_time, tag):
            return None
        _, _, _, event_mode, final_ratio = self.LAYER_PLANS[str(self.layer_plan.value)]
        if current_rate >= float(trade.open_rate) * (1.0 + final_ratio):
            return f"layered_extension_{self.layer_plan.value}"
        rows = self._closed_rows(pair, current_time)
        if rows.empty:
            return None
        row = rows.iloc[-1]
        bearish_choch = bool(row["ms_choch_to_bear"])
        bearish_bos = bool(row["ms_bos_to_bear"])
        event_matches = (
            bearish_choch
            if event_mode == "choch"
            else bearish_bos
            if event_mode == "bos"
            else bearish_choch or bearish_bos
        )
        return f"layered_bearish_structure_{self.layer_plan.value}" if event_matches else None

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
        tag = f"structure_target_partial_{self.layer_plan.value}"
        completed = self._partial_fill_complete(trade, current_time, tag)
        stop_rate = float(trade.open_rate) if completed else float(trade.open_rate) * 0.96
        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=False,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
