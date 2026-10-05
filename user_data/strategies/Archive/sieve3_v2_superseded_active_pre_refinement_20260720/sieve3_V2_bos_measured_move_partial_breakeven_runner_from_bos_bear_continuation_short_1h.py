"""Ordered BOS projection partial, breakeven, and runner for the bearish 1h entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: ordered source-structure position management
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: entry-candle break level projected by the break-to-swing-high range
Invalidation provider: entry-candle active swing high, then exact entry after a filled partial
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds first projection, partial fraction, and runner target
Split rationale: every mode executes one sequence: projection partial, exact breakeven, deeper projection.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_bos_bear_continuation_short_1h"
ENTRY_TAG = "bos_bear_continuation_short_1h"
SIDE = "short"
TIMEFRAME = "1h"

SIEVE_STAGE = 'sieve3'
SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_bos_measured_move_partial_breakeven_runner'
ENTRY_SOURCE_STAGE = 'sieve2'
EXIT_HYPOTHESIS = 'Realize part of a valid BOS projection, protect the remainder at exact entry, and pursue a deeper projection.'

LAYERED_PLANS: dict[str, tuple[float, float, float, str]] = {
    "half_touch_p25_remaining_be_full_runner": (0.50, 0.25, 1.00, "touch"),
    "half_touch_p33_remaining_be_full_runner": (0.50, 0.33, 1.00, "touch"),
    "half_touch_p50_remaining_be_full_runner": (0.50, 0.50, 1.00, "touch"),
    "half_touch_p33_remaining_be_one_and_half_runner": (0.50, 0.33, 1.50, "touch"),
    "half_touch_p50_remaining_be_one_and_half_runner": (0.50, 0.50, 1.50, "touch"),
    "three_quarter_touch_p33_remaining_be_one_and_half_runner": (0.75, 0.33, 1.50, "touch"),
    "three_quarter_touch_p50_remaining_be_one_and_half_runner": (0.75, 0.50, 1.50, "touch"),
    "full_touch_p33_remaining_be_double_runner": (1.00, 0.33, 2.00, "touch"),
    "full_touch_p50_remaining_be_double_runner": (1.00, 0.50, 2.00, "touch"),
    "half_bull_close_p33_remaining_be_full_runner": (0.50, 0.33, 1.00, "bull_close"),
    "half_bull_close_p50_remaining_be_one_and_half_runner": (0.50, 0.50, 1.50, "bull_close"),
    "three_quarter_bull_close_p33_remaining_be_one_and_half_runner": (0.75, 0.33, 1.50, "bull_close"),
    "half_bull_2of3_p33_remaining_be_full_runner": (0.50, 0.33, 1.00, "bull_2of3"),
    "three_quarter_bull_2of3_p50_remaining_be_one_and_half_runner": (0.75, 0.50, 1.50, "bull_2of3"),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


ACTIVE_SELL_PARAMS = ('exit_plan',)
class Sieve3V2BosMeasuredMovePartialBreakevenRunnerFromBosBearContinuationShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_bos_measured_move_partial_breakeven_runner'
    EXIT_HYPOTHESIS = 'Realize part of a valid BOS projection, protect the remainder at exact entry, and pursue a deeper projection.'
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
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
    pressure_min = 0.2
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = False

    exit_plan = CategoricalParameter(tuple(LAYERED_PLANS), default="half_touch_p33_remaining_be_full_runner", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(dataframe, strength=int(self.strength), min_prominence_atr=float(self.min_prominence_atr), min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), max_pivot_age_bars=int(self.max_pivot_age_bars), breakout_buffer_atr=float(self.breakout_buffer_atr), include_diagnostics=True, prefix="ms")
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
        if bool(self.use_state_guard):
            condition &= _num(dataframe, "ms_state").le(0)
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

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(self._utc(current_time))].sort_values("date").copy()

    def _entry_structure(self, pair: str, trade: Any) -> tuple[float, float]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed BOS signal candle is available at trade entry.")
        row = frame.iloc[-1]
        break_level = float(row["ms_break_level"])
        invalidation = float(row["ms_invalidation_level"])
        if not math.isfinite(break_level) or not math.isfinite(invalidation) or invalidation <= break_level:
            raise RuntimeError("The BOS signal candle does not contain a valid break-to-invalidation range.")
        return break_level, invalidation

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        return frame.loc[dates.ge(self._utc(trade.open_date_utc))]

    @staticmethod
    def _target_confirmed(frame: DataFrame, target: float, confirmation: str) -> bool:
        touched = _num(frame, "low").le(target)
        positions = np.flatnonzero(touched.to_numpy())
        if len(positions) == 0:
            return False
        if confirmation == "touch":
            return True
        post_touch = frame.iloc[int(positions[0]):]
        bullish = _num(post_touch, "close").gt(_num(post_touch, "open"))
        if confirmation == "bull_close":
            return bool(bullish.iloc[-1])
        return len(bullish) >= 3 and int(bullish.tail(3).sum()) >= 2

    def _projection(self, pair: str, trade: Any, multiplier: float) -> float:
        break_level, invalidation = self._entry_structure(pair, trade)
        return break_level - (invalidation - break_level) * multiplier

    def _partial_identity(self) -> tuple[str, tuple[str, ...]]:
        return str(self.exit_plan.value), ("s3v2_bos_projection_partial",)

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


    def custom_exit(
        self, pair: str, trade: Any, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs: Any,
    ) -> str | None:
        _ = (current_rate, current_profit, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        if not self._partials_complete(state):
            return None
        final_multiplier = LAYERED_PLANS[plan][2]
        final_target = self._projection(pair, trade, final_multiplier)
        frame = self._post_entry_frame(pair, trade, current_time)
        if bool(_num(frame, "low").le(final_target).any()):
            return "s3v2_bos_projection_runner_target"
        return None

    def custom_stoploss(
        self, pair: str, trade: Any, current_time: datetime, current_rate: float,
        current_profit: float, after_fill: bool, **kwargs: Any,
    ) -> float:
        _ = (current_profit, after_fill, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        _, invalidation = self._entry_structure(pair, trade)
        stop_price = float(trade.open_rate) if self._partials_complete(state) else invalidation
        return stoploss_from_absolute(
            stop_price, current_rate=current_rate, is_short=True,
            leverage=float(trade.leverage or 1.0),
        )

    def adjust_trade_position(
        self, trade: Any, current_time: datetime, current_rate: float, current_profit: float,
        min_stake: float | None, max_stake: float, current_entry_rate: float,
        current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float, str]:
        _ = (current_rate, current_profit, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        if trade.has_open_orders or self._partials_complete(state):
            return None
        first_multiplier, partial_fraction, _, confirmation = LAYERED_PLANS[plan]
        first_target = self._projection(str(trade.pair), trade, first_multiplier)
        if first_target >= float(trade.open_rate):
            return None
        frame = self._post_entry_frame(str(trade.pair), trade, current_time)
        if not self._target_confirmed(frame, first_target, confirmation):
            return None
        remaining_stake = float(trade.stake_amount)
        request = self._partial_request(
            trade, current_time, state, 0, remaining_stake, partial_fraction, min_stake,
            enforce_minimum=True,
        )
        return (-request, tags[0]) if request is not None else None
