"""Measured target partial, exact entry stop, then trail/reversal remainder."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_triangle_squeeze_breakout_long_1h"
SIDE = "long"
SIEVE_STAGE = "sieve3"


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


EXIT_FAMILY = 'target_partial_entry_stop_runner_reversal'


ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_target_partial_entry_stop_runner_reversal'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
SOURCE_STRATEGY = 'complete_pattern_geometry_triangle_squeeze_breakout_long_1h'
EXIT_HYPOTHESIS = 'Take one partial at the entry-frozen triangle target, move the remainder stop to entry, then run to the extension unless a bearish reversal appears.'

class Sieve3V2TargetPartialEntryStopRunnerReversalFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"; startup_candle_count = 180; process_only_new_candles = True; can_short = False
    position_adjustment_enable = True; max_entry_position_adjustment = 0; use_custom_stoploss = True; use_exit_signal = True
    exit_profit_only = False; ignore_roi_if_entry_signal = False; trailing_stop = False; minimal_roi = {"0": 100.0}; stoploss = -0.99

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.35
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.5
    min_containment = 0.88

    sequence_plan = CategoricalParameter(
        [
            "p25_mm50_trail075_bear1", "p25_mm75_trail100_bear1", "p25_mm100_trail125_bear1",
            "p33_mm50_trail075_bear1", "p33_mm75_trail100_bear1", "p33_mm100_trail125_bear1",
            "p50_mm50_trail075_bear1", "p50_mm75_trail100_bear1", "p50_mm100_trail125_bear1",
            "p25_mm75_trail100_bear2", "p25_mm100_trail150_bear2", "p33_mm75_trail100_bear2",
            "p33_mm100_trail150_bear2", "p50_mm75_trail125_bear2", "p50_mm100_trail150_bear2",
            "p33_mm125_trail200_bear2", "p50_mm125_trail200_bear2",
        ], default="p33_mm100_trail125_bear1", space="sell", optimize=True, load=True,
    )
    sequence_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ("sequence_plan",)
    _GEOMETRY_KEY = "triangle_ordered_sequence_entry_geometry"
    _PARTIAL_KEY = "triangle_ordered_sequence_partial_requested"
    _RUNNER_ANCHOR_KEY = "triangle_ordered_sequence_runner_anchor"

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


    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float: return 1.0
    def informative_pairs(self) -> list[tuple[str, str]]: return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool"); close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0); window = int(self.volume_guard_window); baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan); guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):
            open_ = _num(dataframe, "open"); high = _num(dataframe, "high"); low = _num(dataframe, "low"); volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0); candle_range = (high - low).replace(0.0, np.nan); pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0).clip(-1.0, 1.0); directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window); denominator = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan); guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().div(denominator).ge(float(self.pressure_min))
        if bool(self.use_accumulation_guard): window = int(self.pressure_window); guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().ge(0.0)
        if bool(self.use_body_direction_guard): guard &= close.gt(_num(dataframe, "open"))
        if bool(self.use_close_direction_guard): guard &= close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(("pg2_triangle_upper",)).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_triangle_pattern_present") & _bool(dataframe, "pg2_triangle_squeeze_active") & _num(dataframe, "pg2_triangle_indicator_score").ge(float(self.min_line_score)) & _num(dataframe, "pg2_triangle_direction").ge(0) & _num(dataframe, "close").gt(_num(dataframe, "pg2_triangle_upper", np.nan))
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None; return dataframe

    def _entry_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        saved = trade.get_custom_data(key=self._GEOMETRY_KEY)
        if saved is not None: return saved
        opened = pd.Timestamp(trade.open_date_utc); opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        frame = self._closed(pair, opened)
        if frame.empty: raise RuntimeError("no fully closed entry candle is available for triangle geometry")
        row = frame.iloc[-1]
        upper = float(row["pg2_triangle_upper"]); lower = float(row["pg2_triangle_lower"])
        if not np.isfinite([upper, lower]).all() or not 0.0 < lower < upper or lower >= float(trade.open_rate): raise ValueError("entry triangle rails are unavailable or incoherent")
        saved = {"upper": upper, "lower": lower, "height": upper - lower}; trade.set_custom_data(key=self._GEOMETRY_KEY, value=saved); return saved

    def _plan(self) -> tuple[float, float, float, int]:
        partial, target, trail, bearish = str(self.sequence_plan.value).split("_")
        return float(partial[1:]) / 100.0, float(target.removeprefix("mm")) / 100.0, float(trail.removeprefix("trail")) / 10000.0, int(bearish.removeprefix("bear"))

    def _closed(self, pair: str, current_time: datetime) -> DataFrame:
        if self.dp is None: raise RuntimeError("triangle exit requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"close", "date", "high", "open", "pg2_triangle_lower", "pg2_triangle_upper"}
        missing = sorted(required.difference(frame.columns))
        if missing: raise KeyError(f"triangle exit dataframe is missing required columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise"); cutoff = pd.Timestamp(current_time); cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= cutoff].sort_values("date")

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> tuple[float, str] | None:
        if bool(getattr(trade, "has_open_orders", False)): return None
        partial, multiplier, _, _ = self._plan(); geometry = self._entry_geometry(trade.pair, trade); target = geometry["upper"] + geometry["height"] * multiplier
        frame = self._closed(trade.pair, current_time); opened = pd.Timestamp(trade.open_date_utc); opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC"); close_times = pd.to_datetime(frame["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m"); post_entry = frame.loc[close_times.gt(opened)]
        if post_entry.empty or pd.to_numeric(post_entry["high"], errors="coerce").max() < target: return None
        request = self._partial_fill_request(trade, current_time, partial, min_stake, "triangle_target_partial")
        return (-request, "triangle_target_partial") if request is not None else None

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None) and str(getattr(order, "ft_order_tag", None) or "") == "triangle_target_partial":
            state = self._partial_fill_state(trade)
            self._partial_fill_apply(trade, state, order, current_time)
            self._partial_fill_save(trade, state)
            if trade.get_custom_data(key=self._RUNNER_ANCHOR_KEY) is None:
                trade.set_custom_data(key=self._RUNNER_ANCHOR_KEY, value=float(getattr(trade, "max_rate", trade.open_rate) or trade.open_rate))

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        if not self._partial_fill_complete(trade, current_time, "triangle_target_partial"): return None
        frame = self._closed(pair, current_time); bearish_count = self._plan()[3]
        if len(frame) < bearish_count + 1: return None
        close = _num(frame, "close"); open_ = _num(frame, "open"); bearish = close.tail(bearish_count).lt(open_.tail(bearish_count)) & close.tail(bearish_count).lt(close.shift(1).tail(bearish_count))
        return "post_partial_bearish_reversal" if bool(bearish.all()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        geometry = self._entry_geometry(pair, trade); completed = self._partial_fill_complete(trade, current_time, "triangle_target_partial")
        stop_price = geometry["lower"]
        if completed:
            trail = self._plan()[2]
            max_rate = max(float(getattr(trade, "max_rate", trade.open_rate) or trade.open_rate), float(trade.open_rate))
            anchor = trade.get_custom_data(key=self._RUNNER_ANCHOR_KEY)
            if anchor is None:
                anchor = max_rate
                trade.set_custom_data(key=self._RUNNER_ANCHOR_KEY, value=anchor)
            stop_price = float(trade.open_rate)
            if max_rate >= float(anchor) * (1.0 + trail):
                stop_price = max(stop_price, max_rate * (1.0 - trail))
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))
