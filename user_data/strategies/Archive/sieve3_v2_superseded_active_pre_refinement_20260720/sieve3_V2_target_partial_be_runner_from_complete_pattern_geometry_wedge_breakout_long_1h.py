from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_wedge_breakout_long_1h"
ENTRY_TAG = "geometry_wedge_breakout_long_1h"
SIDE = "long"
TIMEFRAME = "1h"
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = "sieve2"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:523"
SOURCE_STRATEGY = "D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\sieve2_geometry_wedge_breakout_long_1h__pattern_geometry_1h_2023_24_compression\\full_cycle_2020_2026\\tp_4_sl_2\\backtest-result-2026-05-26_02-54-35.zip!backtest-result-2026-05-26_02-54-35_Sieve2GeometryWedgeBreakoutLong1H.py:Sieve2GeometryWedgeBreakoutLong1H"
RESEARCH_PATH = 'sieve3_exit_target_partial_be_runner'
EXIT_HYPOTHESIS = "Take one ordered partial at a wedge projection, move the remainder stop to exact entry, then exit at an extended projection or bearish reversal."

# Preflight: frozen wedge-height targets; source/candle reversal guard; exact open-rate post-partial stop.
# One named plan owns target order, partial size, runner target, and remainder reversal confirmation.
ACTIVE_SELL_PARAMS = ("sequence_plan",)
BRANCH_LOCAL_PARAMS: tuple[str, ...] = ()




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


EXIT_FAMILY = 'target_partial_be_runner'


ENTRY_SOURCE_STAGE = 'sieve2'


class Sieve3V2TargetPartialBeRunnerFromCompletePatternGeometryWedgeBreakoutLong1H(IStrategy):
    ACTIVE_SELL_PARAMS = ('sequence_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.15
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.55
    min_containment = 0.88

    sequence_plan = CategoricalParameter(["h50_p33_h100_bear2", "h50_p50_h100_upper_bear", "h75_p33_h125_bear2", "h75_p50_h125_direction", "h100_p33_h150_bear2", "h100_p50_h150_two_of_three", "h75_p33_h150_upper_direction", "h100_p33_h200_strong_bear"], default="h75_p33_h125_bear2", space="sell", optimize=True, load=True)
    sequence_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    SEQUENCE_PLANS = {
        "h50_p33_h100_bear2": (0.50, 1.0 / 3.0, 1.00, "bear2"),
        "h50_p50_h100_upper_bear": (0.50, 0.50, 1.00, "upper_bear"),
        "h75_p33_h125_bear2": (0.75, 1.0 / 3.0, 1.25, "bear2"),
        "h75_p50_h125_direction": (0.75, 0.50, 1.25, "direction"),
        "h100_p33_h150_bear2": (1.00, 1.0 / 3.0, 1.50, "bear2"),
        "h100_p50_h150_two_of_three": (1.00, 0.50, 1.50, "two_of_three"),
        "h75_p33_h150_upper_direction": (0.75, 1.0 / 3.0, 1.50, "upper_direction"),
        "h100_p33_h200_strong_bear": (1.00, 1.0 / 3.0, 2.00, "strong_bear"),
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

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=True, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            directional_volume = (((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0) * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            guard &= (directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline).ge(float(self.pressure_min))
        if bool(self.use_accumulation_guard):
            window = int(self.pressure_window)
            guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().ge(0.0)
        if bool(self.use_body_direction_guard):
            guard &= close.gt(_num(dataframe, "open"))
        if bool(self.use_close_direction_guard):
            guard &= close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_wedge_pattern_present") & _bool(dataframe, "pg2_wedge_squeeze_active")
        condition &= _num(dataframe, "pg2_wedge_indicator_score").ge(float(self.min_line_score))
        condition &= _num(dataframe, "pg2_wedge_direction").ge(0)
        condition &= _num(dataframe, "close").gt(_num(dataframe, "pg2_wedge_upper"))
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

    def _context(self, pair: str, trade: Any, current_time: datetime) -> tuple[Series, DataFrame]:
        if self.dp is None:
            raise RuntimeError("ordered wedge sequence requires the analyzed dataframe")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "high", "low", "close", "pg2_wedge_upper", "pg2_wedge_lower", "pg2_wedge_indicator_score", "pg2_wedge_direction"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"ordered wedge sequence missing columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        opened = pd.Timestamp(trade.open_date_utc)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        now = pd.Timestamp(current_time)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        signal_rows = frame.loc[dates < opened]
        closed = frame.loc[(dates >= opened) & (dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= now)]
        if signal_rows.empty:
            raise RuntimeError("ordered wedge sequence cannot locate the entry signal candle")
        return signal_rows.iloc[-1], closed

    @staticmethod
    def _projected_target(signal: Series, multiple: float) -> float:
        upper = float(signal["pg2_wedge_upper"])
        lower = float(signal["pg2_wedge_lower"])
        if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
            raise ValueError("ordered wedge sequence has invalid entry rails")
        return upper + (upper - lower) * multiple

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | tuple[float, str] | None:
        _ = current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        signal, _ = self._context(str(trade.pair), trade, current_time)
        first_multiple, fraction, _, _ = self.SEQUENCE_PLANS[str(self.sequence_plan.value)]
        if current_rate >= self._projected_target(signal, first_multiple):
            request = self._partial_fill_request(trade, current_time, fraction, min_stake, "wedge_target_partial")
            return (-request, "wedge_target_partial") if request is not None else None
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = pair, current_time, current_profit, after_fill, kwargs
        completed = self._partial_fill_complete(trade, current_time, "wedge_target_partial")
        stop_price = float(trade.open_rate) if completed else float(trade.open_rate) * 0.97
        if current_rate <= stop_price:
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = kwargs
        completed = self._partial_fill_complete(trade, current_time, "wedge_target_partial")
        if not completed:
            return "ordered_initial_hard_stop" if current_profit <= -0.03 else None
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        if current_rate <= float(trade.open_rate):
            return "ordered_post_partial_exact_entry_stop"
        signal, closed = self._context(pair, trade, current_time)
        _, _, runner_multiple, reversal_mode = self.SEQUENCE_PLANS[str(self.sequence_plan.value)]
        if current_rate >= self._projected_target(signal, runner_multiple):
            return f"ordered_runner_target_{self.sequence_plan.value}"
        if closed.empty:
            return None
        rows = closed.tail(2)
        last = rows.iloc[-1]
        bearish = pd.to_numeric(rows["close"], errors="coerce").lt(pd.to_numeric(rows["open"], errors="coerce"))
        bear2 = len(rows) == 2 and bool(bearish.all()) and float(rows.iloc[-1]["close"]) < float(rows.iloc[-2]["close"])
        upper = float(last["pg2_wedge_upper"]) if pd.notna(last["pg2_wedge_upper"]) else np.nan
        direction = float(last["pg2_wedge_direction"]) if pd.notna(last["pg2_wedge_direction"]) else np.nan
        score = float(last["pg2_wedge_indicator_score"]) if pd.notna(last["pg2_wedge_indicator_score"]) else np.nan
        upper_loss = np.isfinite(upper) and float(last["close"]) < upper
        direction_negative = np.isfinite(direction) and direction < 0.0
        score_weak = np.isfinite(score) and score < 0.40
        candle_range = float(last["high"] - last["low"])
        strong_bear = candle_range > 0.0 and float(last["open"] - last["close"]) / candle_range >= 0.60
        reversal = {
            "bear2": bear2,
            "upper_bear": upper_loss and bool(bearish.iloc[-1]),
            "direction": direction_negative,
            "two_of_three": sum((bear2, direction_negative, score_weak)) >= 2,
            "upper_direction": upper_loss and direction_negative,
            "strong_bear": strong_bear,
        }[reversal_mode]
        return f"ordered_runner_reversal_{reversal_mode}" if reversal else None


LOCKED_BUY_PARAMS = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context", "sieve2_vp_window": 96, "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28, "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24, "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25, "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True, "volume_guard_window": 24, "volume_ratio_min": 1.6, "use_pressure_guard": True,
    "pressure_window": 48, "pressure_min": 0.15, "use_accumulation_guard": False, "use_body_direction_guard": False,
    "use_close_direction_guard": False, "min_pattern_bars": 12, "max_pattern_bars": 72, "compression_max_width_atr": 2.0,
    "squeeze_active_width_atr": 2.0, "min_line_score": 0.55, "min_containment": 0.88,
}
