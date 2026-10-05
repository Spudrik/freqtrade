from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ACTIVE_SELL_PARAMS = ("management_plan",)
GEOMETRY_STATE_KEY = "s3v2_wedge_breakdown_signal_geometry"


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_measured_move_partial_entry_stop_runner"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_complete_pattern_geometry_wedge_breakdown_short_4h"
EXIT_HYPOTHESIS = "Test the measured move partial entry stop runner exit family while preserving the complete pattern geometry wedge breakdown short 4h entry behavior."


class Sieve3V2MeasuredMovePartialEntryStopRunnerFromCompletePatternGeometryWedgeBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    ACTIVE_SELL_PARAMS = ("management_plan",)
    use_exit_signal = True
    use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    # Legacy fixed-off entry parameter: use_sieve2_vp_guard=False (BooleanParameter, buy space).
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.25
    sieve2_vp_context_min = 0.28
    # Legacy fixed-off entry parameter: use_sieve2_market_guard=False (BooleanParameter, buy space).
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.07
    sieve2_market_trend_min = 0.25
    sieve2_rs_benchmark_pair = 'BTC/USDT:USDT'
    sieve2_rs_score_min = 0.45
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.6
    # Legacy fixed-off entry parameter: use_pressure_guard=False (BooleanParameter, buy space).
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.55
    min_containment = 0.88

    management_plan = CategoricalParameter(
        [
            "half_height_partial_50_be_full_height",
            "three_quarter_height_partial_50_be_one_and_half_height",
            "half_height_partial_33_be_bullish_reversal",
            "three_quarter_height_partial_50_be_full_height_or_reversal",
        ],
        default="half_height_partial_50_be_full_height",
        space="sell", optimize=True, load=True,
    )
    management_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    @staticmethod
    def _num(frame: DataFrame, column: str) -> Series:
        return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _flag(frame: DataFrame, column: str) -> Series:
        return frame[column].astype("boolean").fillna(False).astype(bool)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return add_pattern_geometry_v2(
            dataframe, timeframe=self.timeframe, output_slots=1,
            include_triangle_patterns=False, include_wedge_patterns=True,
            include_compression_patterns=False, include_rectangle_patterns=False,
            include_ascending_channel_patterns=False, include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2",
        )

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        volume = self._num(dataframe, 'volume').clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = (
            self._flag(dataframe, "pg2_wedge_pattern_present")
            & self._flag(dataframe, "pg2_wedge_squeeze_active")
            & self._num(dataframe, "pg2_wedge_indicator_score").ge(float(self.min_line_score))
            & self._num(dataframe, "pg2_wedge_direction").le(0)
            & self._num(dataframe, "close").lt(self._num(dataframe, "pg2_wedge_lower"))
        )
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = "geometry_wedge_breakdown_short_4h"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _signal_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        saved = trade.get_custom_data(key=GEOMETRY_STATE_KEY)
        if isinstance(saved, dict):
            return {
                "pg2_wedge_upper": float(saved["pg2_wedge_upper"]),
                "pg2_wedge_lower": float(saved["pg2_wedge_lower"]),
            }
        if not self.dp:
            raise RuntimeError("signal geometry requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("signal geometry requires a non-empty analyzed dataframe")
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        closes_at = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        signal_rows = dataframe.loc[closes_at.le(filled_at)]
        if signal_rows.empty:
            raise RuntimeError("no fully closed pre-entry signal candle is available")
        row = signal_rows.iloc[-1]
        upper = float(row["pg2_wedge_upper"])
        lower = float(row["pg2_wedge_lower"])
        if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower or lower <= 0.0:
            raise ValueError("entry-frozen wedge geometry must be finite, positive, and ordered")
        state = {"pg2_wedge_upper": upper, "pg2_wedge_lower": lower}
        trade.set_custom_data(key=GEOMETRY_STATE_KEY, value=state)
        return state

    def _entry_and_current(
        self, pair: str, trade: Any, current_time: datetime
    ) -> tuple[Series, DataFrame] | None:
        if not self.dp:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        candle_open = self._utc(timeframe_to_prev_date(self.timeframe, filled_at.to_pydatetime()))
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        closes_at = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = closes_at.le(self._utc(current_time))
        post_fill = dates.ge(candle_open) if filled_at == candle_open else dates.gt(candle_open)
        return pd.Series(self._signal_geometry(pair, trade)), dataframe.loc[closed & post_fill]

    def _plan(self) -> tuple[float, float, float, bool]:
        return {
            "half_height_partial_50_be_full_height": (0.5, 0.5, 1.0, False),
            "three_quarter_height_partial_50_be_one_and_half_height": (0.75, 0.5, 1.5, False),
            "half_height_partial_33_be_bullish_reversal": (0.5, 0.33, 0.0, True),
            "three_quarter_height_partial_50_be_full_height_or_reversal": (0.75, 0.5, 1.0, True),
        }[str(self.management_plan.value)]

    @staticmethod
    def _target(entry: Series, multiplier: float) -> float | None:
        upper = float(entry["pg2_wedge_upper"])
        lower = float(entry["pg2_wedge_lower"])
        if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
            return None
        target = lower - (upper - lower) * multiplier
        return target if target > 0.0 else None

    PARTIAL_STAGE_TAG = "wedge_target_partial"
    PARTIAL_STATE_KEY = "s3v2_partial_stage_state"

    @classmethod
    def _partial_state(cls, trade: Any) -> dict[str, Any]:
        raw = trade.get_custom_data(key=cls.PARTIAL_STATE_KEY)
        if isinstance(raw, dict):
            return dict(raw)
        return {
            "status": "ready",
            "target_stake": None,
            "credited_stake": 0.0,
            "order_credits": {},
            "requested_at": None,
            "filled_at": None,
        }

    @staticmethod
    def _partial_order_stake(order: Any, trade: Any, filled: bool) -> float:
        amount_name = "safe_filled" if filled else "safe_amount"
        amount = float(getattr(order, amount_name, 0.0) or 0.0)
        price = float(getattr(order, "safe_price", 0.0) or 0.0)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        return amount * price / leverage

    def _reconcile_partial(self, trade: Any, current_time: Any, extra_order: Any | None = None) -> dict[str, Any]:
        state = self._partial_state(trade)
        before = repr(state)
        orders = list(getattr(trade, "orders", ()) or ())
        if extra_order is not None and all(order is not extra_order for order in orders):
            orders.append(extra_order)
        matching = [
            order
            for order in orders
            if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ]
        if state.get("target_stake") is None and matching:
            state["target_stake"] = self._partial_order_stake(matching[0], trade, filled=False)
        credits = dict(state.get("order_credits") or {})
        credited = float(state.get("credited_stake") or 0.0)
        has_open = False
        fully_filled = False
        for order in matching:
            order_id = str(getattr(order, "order_id", None) or "")
            if not order_id:
                raise ValueError("partial-stage reconciliation requires an order id")
            filled_stake = self._partial_order_stake(order, trade, filled=True)
            previous = float(credits.get(order_id) or 0.0)
            if filled_stake > previous:
                credited += filled_stake - previous
                credits[order_id] = filled_stake
            status = str(getattr(order, "status", None) or "").casefold()
            is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
                "canceled", "cancelled", "closed", "expired", "failed", "rejected",
            }
            requested_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
            filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
            has_open |= is_open
            fully_filled |= requested_amount > 0.0 and filled_amount >= requested_amount - max(1e-12, requested_amount * 1e-9)
        state["order_credits"] = credits
        state["credited_stake"] = credited
        target = float(state.get("target_stake") or 0.0)
        tolerance = max(1e-8, target * 0.005)
        if fully_filled or (target > 0.0 and credited >= target - tolerance):
            state["status"] = "filled"
            state["requested_at"] = None
            if state.get("filled_at") is None:
                state["filled_at"] = pd.Timestamp(current_time).isoformat()
        elif has_open or (
            state.get("status") == "requested"
            and not matching
            and state.get("requested_at") == pd.Timestamp(current_time).isoformat()
        ):
            state["status"] = "requested"
        else:
            state["status"] = "ready"
            state["requested_at"] = None
        if repr(state) != before:
            trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return state

    def _request_partial(self, trade: Any, current_time: Any, desired: float, min_stake: float | None) -> float | None:
        state = self._reconcile_partial(trade, current_time)
        if state["status"] != "ready":
            return None
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        target = state.get("target_stake")
        request = min(stake, desired) if target is None else min(stake, max(0.0, float(target) - float(state["credited_stake"])))
        if min_stake is not None and 0.0 < stake - request < float(min_stake):
            request = stake - float(min_stake)
        if request <= 0.0 or request >= stake:
            return None
        if target is None:
            state["target_stake"] = request
        state["status"] = "requested"
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return request

    def _partial_complete(self, trade: Any, current_time: Any) -> bool:
        return self._reconcile_partial(trade, current_time)["status"] == "filled"


    def adjust_trade_position(
        self, trade: Any, current_time: datetime, current_rate: float, current_profit: float,
        min_stake: float | None, max_stake: float, current_entry_rate: float,
        current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
        **kwargs: Any,
    ) -> tuple[float, str] | None:
        if bool(trade.has_open_orders):
            return None
        context = self._entry_and_current(trade.pair, trade, current_time)
        if context is None:
            return None
        entry, _ = context
        partial_multiplier, fraction, _, _ = self._plan()
        target = self._target(entry, partial_multiplier)
        if target is not None and current_rate <= target:
            request = self._request_partial(
                trade,
                current_time,
                float(trade.stake_amount) * fraction,
                min_stake,
            )
            return (-request, self.PARTIAL_STAGE_TAG) if request is not None else None
        return None

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = (pair, kwargs)
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None)
            and trade.get_custom_data(key=GEOMETRY_STATE_KEY) is None
        ):
            self._signal_geometry(pair, trade)
            return
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ):
            self._reconcile_partial(trade, current_time, order)

    def custom_stoploss(
        self, pair: str, trade: Any, current_time: datetime, current_rate: float,
        current_profit: float, after_fill: bool = False, **kwargs: Any,
    ) -> float:
        stop_rate = float(trade.open_rate) if self._partial_complete(trade, current_time) else float(trade.open_rate) * 1.03
        return stoploss_from_absolute(stop_rate, current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        if not self._partial_complete(trade, current_time):
            return None
        context = self._entry_and_current(pair, trade, current_time)
        if context is None:
            return None
        entry, dataframe = context
        _, _, remainder_multiplier, reversal_enabled = self._plan()
        target = self._target(entry, remainder_multiplier) if remainder_multiplier > 0.0 else None
        if target is not None and current_rate <= target:
            return "wedge_runner_target"
        if reversal_enabled and len(dataframe) >= 2:
            current = dataframe.iloc[-1]
            previous = dataframe.iloc[-2]
            two_bullish = float(current["close"]) > float(current["open"]) and float(previous["close"]) > float(previous["open"])
            if two_bullish:
                return "wedge_runner_bullish_reversal"
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0
