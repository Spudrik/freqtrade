from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import amount_to_contract_precision, timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_triangle_squeeze_breakdown_short_4h"
SIDE = "short"


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


EXIT_FAMILY = "target_partial_entry_stop_runner"

ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_target_partial_entry_stop_runner"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume.jsonl:242,380,446"
SOURCE_STRATEGY = "Sieve2GeometryTriangleSqueezeBreakdownShort4H"
EXIT_HYPOTHESIS = (
    "After a measured-move partial, an exact entry stop can protect the remainder while "
    "continuation runs."
)


class Sieve3V2TargetPartialEntryStopRunnerFromCompletePatternGeometryTriangleSqueezeBreakdownShort4H(  # noqa: E501
    IStrategy
):
    """Ordered half-height partial, exact-entry stop, then measured target or reversal runner."""

    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True

    SIEVE_STAGE = "sieve3"
    ENTRY_SOURCE_STAGE = "sieve2"
    SOURCE_STRATEGY = "Sieve2GeometryTriangleSqueezeBreakdownShort4H"
    SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume.jsonl:242,380,446"
    PRIMARY_TRIGGER = (
        "4h close below the emitted triangle lower rail while squeeze, score, and "
        "non-positive direction hold"
    )
    PRIMARY_GUARD = (
        "locked volume/pressure guards; optional Sieve2 VP and market guard selectors "
        "are frozen off"
    )
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    RESEARCH_PATH = "sieve3_exit_target_partial_entry_stop_runner"
    EXIT_HYPOTHESIS = (
        "After a measured-move partial, an exact entry stop can protect the remainder while "
        "continuation runs."
    )
    LOCK_STATUS = "verified promoted params plus tested-snapshot defaults"

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
    pressure_window = 24
    pressure_min = 0.05
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.4
    min_containment = 0.9

    exit_plan = CategoricalParameter(
        [
            "partial33_half_final_full",
            "partial50_half_final_full",
            "partial33_half_reversal_runner",
        ],
        default="partial33_half_final_full",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_plan.batch_tags = ("family:exits", "mode:sieve3_exit")

    def _partial_fill_state(self, trade: Any) -> dict[str, Any]:
        key = f"sieve3_v2_partial_fill:{self.__class__.__name__}"
        state = trade.get_custom_data(key=key)
        if state is None:
            state = {
                "status": "ready",
                "target_amount": None,
                "target_stake": None,
                "credited_amount": 0.0,
                "credited_stake": 0.0,
                "request_stake": None,
                "requested_at": None,
                "order_id": None,
                "resolved_order_ids": [],
                "failure_reason": None,
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
    def _partial_fill_target_tolerance(target_stake: float) -> float:
        if not math.isfinite(target_stake) or target_stake <= 0.0:
            raise ValueError("partial exit target stake must be finite and positive")
        # Covers only proportional conversion and accumulated IEEE-754 error.
        return max(math.ulp(target_stake) * 64.0, target_stake * 1e-12)

    @staticmethod
    def _partial_fill_trade_basis(trade: Any) -> tuple[float, float]:
        trade_amount = float(getattr(trade, "amount", 0.0) or 0.0)
        trade_stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if not math.isfinite(trade_amount) or trade_amount <= 0.0:
            raise ValueError("partial exit trade amount must be finite and positive")
        if not math.isfinite(trade_stake) or trade_stake <= 0.0:
            raise ValueError("partial exit trade stake must be finite and positive")
        return trade_amount, trade_stake

    @staticmethod
    def _partial_fill_quantize_amount(trade: Any, amount: float) -> float:
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError("partial exit amount to quantize must be finite and non-negative")
        rounded = float(
            amount_to_contract_precision(
                amount,
                getattr(trade, "amount_precision", None),
                getattr(trade, "precision_mode", None),
                getattr(trade, "contract_size", None),
            )
        )
        if not math.isfinite(rounded) or rounded < 0.0:
            raise ValueError("rounded partial exit amount must be finite and non-negative")
        return rounded

    def _partial_fill_initialize(
        self,
        trade: Any,
        state: dict[str, Any],
        fraction: float,
    ) -> None:
        if state.get("target_stake") is not None or state.get("status") != "ready":
            return
        fraction = float(fraction)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise ValueError(
                "partial exit fraction must be finite and strictly between zero and one"
            )
        trade_amount, trade_stake = self._partial_fill_trade_basis(trade)
        target_amount = self._partial_fill_quantize_amount(trade, trade_amount * fraction)
        amount_tolerance = max(math.ulp(trade_amount) * 64.0, trade_amount * 1e-12)
        if target_amount <= 0.0:
            state["status"] = "unfillable"
            state["failure_reason"] = "target_rounds_to_zero"
            return
        if target_amount >= trade_amount - amount_tolerance:
            state["status"] = "unfillable"
            state["failure_reason"] = "target_would_consume_trade"
            return
        target_stake = target_amount * trade_stake / trade_amount
        if not math.isfinite(target_stake) or target_stake <= 0.0:
            raise ValueError("reachable partial exit target stake must be finite and positive")
        if target_stake >= trade_stake:
            state["status"] = "unfillable"
            state["failure_reason"] = "target_would_consume_trade"
            return
        state["target_amount"] = target_amount
        state["target_stake"] = target_stake
        state["failure_reason"] = None

    @staticmethod
    def _partial_fill_snapshot(order: Any) -> tuple[bool, float]:
        status = str(getattr(order, "status", None) or "").casefold()
        is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "closed",
        }
        filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
        if not math.isfinite(filled_amount) or filled_amount < 0.0:
            raise ValueError("partial exit filled amount must be finite and non-negative")
        return is_open, filled_amount

    def _partial_fill_cost_basis_stake(self, trade: Any, filled_amount: float) -> float:
        if not math.isfinite(filled_amount) or filled_amount <= 0.0:
            raise ValueError("credited partial exit amount must be finite and positive")
        trade_amount, trade_stake = self._partial_fill_trade_basis(trade)
        credited_stake = filled_amount * trade_stake / trade_amount
        if not math.isfinite(credited_stake) or credited_stake <= 0.0:
            raise ValueError("partial exit cost-basis credit must be finite and positive")
        return credited_stake

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
        is_open, filled_amount = self._partial_fill_snapshot(order)
        if is_open:
            state["status"] = "requested"
            state["order_id"] = order_key
            return
        target_stake = float(state.get("target_stake") or 0.0)
        tolerance = self._partial_fill_target_tolerance(target_stake)
        credited_stake = float(state.get("credited_stake") or 0.0)
        credited_amount = float(state.get("credited_amount") or 0.0)
        if not math.isfinite(credited_stake) or credited_stake < 0.0:
            raise ValueError("partial exit credited stake must be finite and non-negative")
        if not math.isfinite(credited_amount) or credited_amount < 0.0:
            raise ValueError("partial exit credited amount must be finite and non-negative")
        if filled_amount > 0.0:
            actual_credit = self._partial_fill_cost_basis_stake(trade, filled_amount)
            state["credited_amount"] = credited_amount + filled_amount
            state["credited_stake"] = credited_stake + actual_credit
        resolved.append(order_key)
        credited_stake = float(state.get("credited_stake") or 0.0)
        if target_stake > 0.0 and math.isclose(
            credited_stake, target_stake, rel_tol=0.0, abs_tol=tolerance
        ):
            state["status"] = "filled"
            state["order_id"] = order_key
        elif credited_stake > target_stake + tolerance:
            state["status"] = "failed"
            state["failure_reason"] = "credited_stake_exceeds_target"
            state["order_id"] = order_key
        else:
            state["status"] = "ready"
            state["order_id"] = None
        state["request_stake"] = None
        state["requested_at"] = None

    def _partial_fill_reconcile(
        self, trade: Any, current_time: datetime, tag: str
    ) -> dict[str, Any]:
        state = self._partial_fill_state(trade)
        if state["status"] in {"filled", "failed", "unfillable"}:
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
            open_order = next(
                (order for order in reversed(orders) if self._partial_fill_snapshot(order)[0]), None
            )
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

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = (pair, kwargs)
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None):
            state = self._partial_fill_state(trade)
            fraction = 0.50 if str(self.exit_plan.value).startswith("partial50") else 0.33
            self._partial_fill_initialize(trade, state, fraction)
            self._partial_fill_save(trade, state)
            return
        if (
            getattr(order, "ft_order_side", None) != getattr(trade, "exit_side", None)
            or str(getattr(order, "ft_order_tag", None) or "") != "half_height_partial"
        ):
            return
        state = self._partial_fill_state(trade)
        if state["status"] != "filled":
            self._partial_fill_apply(trade, state, order, current_time)
            self._partial_fill_save(trade, state)

    def _partial_fill_complete(self, trade: Any, current_time: datetime, tag: str) -> bool:
        return self._partial_fill_reconcile(trade, current_time, tag)["status"] == "filled"

    def _partial_fill_residual_request(
        self,
        trade: Any,
        state: dict[str, Any],
        current_time: datetime,
        min_stake: float | None,
    ) -> float | None:
        current_amount, current_stake = self._partial_fill_trade_basis(trade)
        target_stake = float(state.get("target_stake") or 0.0)
        target_amount = float(state.get("target_amount") or 0.0)
        if not math.isfinite(target_amount) or target_amount <= 0.0:
            raise ValueError("partial exit target amount must be finite and positive")
        credited_stake = float(state.get("credited_stake") or 0.0)
        credited_amount = float(state.get("credited_amount") or 0.0)
        if not math.isfinite(credited_stake) or credited_stake < 0.0:
            raise ValueError("partial exit credited stake must be finite and non-negative")
        if not math.isfinite(credited_amount) or credited_amount < 0.0:
            raise ValueError("partial exit credited amount must be finite and non-negative")
        tolerance = self._partial_fill_target_tolerance(target_stake)
        if math.isclose(credited_stake, target_stake, rel_tol=0.0, abs_tol=tolerance):
            state["status"] = "filled"
            self._partial_fill_save(trade, state)
            return None
        if credited_stake > target_stake + tolerance:
            state["status"] = "failed"
            state["failure_reason"] = "credited_stake_exceeds_target"
            self._partial_fill_save(trade, state)
            return None
        remaining_amount = target_amount - credited_amount
        amount_tolerance = max(math.ulp(current_amount) * 64.0, current_amount * 1e-12)
        if remaining_amount < -amount_tolerance:
            state["status"] = "failed"
            state["failure_reason"] = "credited_amount_exceeds_target"
            self._partial_fill_save(trade, state)
            return None
        residual_amount = self._partial_fill_quantize_amount(trade, max(0.0, remaining_amount))
        if residual_amount <= 0.0:
            state["status"] = "unfillable"
            state["failure_reason"] = "residual_rounds_to_zero"
            self._partial_fill_save(trade, state)
            return None
        if residual_amount >= current_amount - amount_tolerance:
            state["status"] = "unfillable"
            state["failure_reason"] = "residual_would_consume_trade"
            self._partial_fill_save(trade, state)
            return None
        request = residual_amount * current_stake / current_amount
        if not math.isfinite(request) or request <= 0.0 or request >= current_stake:
            raise ValueError(
                "partial exit request stake must be finite, positive, and below trade stake"
            )
        if min_stake is not None and 0.0 < current_stake - request < float(min_stake):
            state["status"] = "unfillable"
            state["failure_reason"] = "residual_would_leave_below_minimum"
            self._partial_fill_save(trade, state)
            return None
        state["status"] = "requested"
        state["request_stake"] = request
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        state["order_id"] = None
        self._partial_fill_save(trade, state)
        return request

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
        if state.get("target_stake") is None:
            self._partial_fill_initialize(trade, state, fraction)
            if state["status"] != "ready":
                self._partial_fill_save(trade, state)
                return None
        return self._partial_fill_residual_request(trade, state, current_time, min_stake)

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
        dataframe = add_pattern_geometry_v2(
            dataframe,
            timeframe=self.timeframe,
            output_slots=1,
            include_triangle_patterns=True,
            include_wedge_patterns=False,
            include_compression_patterns=False,
            include_rectangle_patterns=False,
            include_ascending_channel_patterns=False,
            include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars),
            max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr),
            squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            min_line_score=float(self.min_line_score),
            min_containment=float(self.min_containment),
            output_prefix="pg2",
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
            guard &= pressure_ratio.le(-float(self.pressure_min))
        if bool(self.use_accumulation_guard):
            window = int(self.pressure_window)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.le(0.0)
        if bool(self.use_body_direction_guard):
            guard &= close.lt(_num(dataframe, "open"))
        if bool(self.use_close_direction_guard):
            guard &= close.lt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = (
            _bool(dataframe, "pg2_triangle_pattern_present")
            & _bool(dataframe, "pg2_triangle_squeeze_active")
            & _num(dataframe, "pg2_triangle_indicator_score").ge(float(self.min_line_score))
            & _num(dataframe, "pg2_triangle_direction").le(0)
            & _num(dataframe, "close").lt(_num(dataframe, "pg2_triangle_lower"))
        )
        condition &= self._common_guards(dataframe)
        condition = (
            pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
        )
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

    def _geometry(self, pair: str, trade: Any) -> tuple[float, float] | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {
            "date",
            "pg2_slot_1_upper_start",
            "pg2_slot_1_lower_start",
            "pg2_triangle_lower",
        }
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise KeyError(f"missing exact triangle geometry columns: {missing}")
        opened = pd.Timestamp(trade.open_date_utc)
        opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
        close_times = pd.to_datetime(dataframe["date"], utc=True, errors="raise") + pd.to_timedelta(
            timeframe_to_minutes(self.timeframe), unit="m"
        )
        rows = dataframe.loc[close_times.le(opened)]
        if rows.empty:
            return None
        row = rows.iloc[-1]
        upper_start = float(row["pg2_slot_1_upper_start"])
        lower_start = float(row["pg2_slot_1_lower_start"])
        lower = float(row["pg2_triangle_lower"])
        return (
            (lower, upper_start - lower_start)
            if np.isfinite([upper_start, lower_start, lower]).all() and upper_start > lower_start
            else None
        )

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs: Any,
    ) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        post_partial = self._partial_fill_complete(trade, current_time, "half_height_partial")
        stop_price = float(trade.open_rate) if post_partial else float(trade.open_rate) * 1.03
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=True,
            leverage=float(trade.leverage or 1.0),
        )

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
    ) -> tuple[float, str] | None:
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        geometry = self._geometry(trade.pair, trade)
        if geometry is None:
            return None
        lower, height = geometry
        if current_rate > lower - height * 0.5:
            return None
        fraction = 0.50 if str(self.exit_plan.value).startswith("partial50") else 0.33
        request = self._partial_fill_request(
            trade, current_time, fraction, min_stake, "half_height_partial"
        )
        return (-request, "half_height_partial") if request is not None else None

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = (current_time, current_profit, kwargs)
        if not self._partial_fill_complete(trade, current_time, "half_height_partial"):
            return None
        geometry = self._geometry(pair, trade)
        if geometry is None:
            return None
        lower, height = geometry
        plan = str(self.exit_plan.value)
        if plan.endswith("final_full") and current_rate <= lower - height:
            return "full_height_runner_target"
        if plan.endswith("reversal_runner"):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            required = {"date", "close"}
            missing = sorted(required - set(dataframe.columns))
            if missing:
                raise KeyError(f"missing exact reversal runner columns: {missing}")
            now = pd.Timestamp(current_time)
            now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
            opened = pd.Timestamp(
                getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
            )
            opened = (
                opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC")
            )
            open_times = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
            close_times = open_times + pd.to_timedelta(
                timeframe_to_minutes(self.timeframe), unit="m"
            )
            dataframe = dataframe.loc[open_times.ge(opened) & close_times.le(now)].sort_values(
                "date"
            )
            if (
                dataframe is not None
                and len(dataframe) >= 2
                and float(dataframe.iloc[-1]["close"]) > float(dataframe.iloc[-2]["close"])
            ):
                return "post_partial_bullish_reversal"
        return None
