from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_prev_date  # noqa: E402
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch  # noqa: E402

# Source: selected Sieve2 snapshot in the 2026-05-21 8h BOS result archive.
# Exit family: ordered structure partial -> exact breakeven -> target/trail remainder.
# Trigger/guard: bearish 8h BOS; bearish state plus locked volume/pressure guards.
# Target: fill-anchored frozen BOS-height projections; invalidation: swing high.
# Active HyperOpt parameter: exit_plan. No branch-local inactive parameters.
# Kept together because every plan uses the same ordered management sequence.
ENTRY_MODE = "entry_bos_bear_continuation_short_8h"
ENTRY_TAG = "bos_bear_continuation_short_8h"
SIDE = "short"
TIMEFRAME = "8h"

SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = (
    "Sieve2BOSBearContinuationShort8H snapshot:"
    "backtest-result-2026-05-21_18-40-06_Sieve2BOSBearContinuationShort8H.py"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
SOURCE_PARAMS_FILE = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/"
    "Sieve2BOSBearContinuationShort8H__pattern_continuation_8h_2020_23.json"
)
LINEAGE_STATUS = "selected_sieve2_params_verified; promotion_label_ambiguous"
RESEARCH_PATH = "sieve3_exit_layered_structure_partial_be_remainder"
EXIT_HYPOTHESIS = (
    "A first BOS structure objective can fund a one-shot reduction, after which "
    "the remainder is protected at exact entry before a final target or trail."
)
PRIMARY_TRIGGER = "confirmed 8h ms_bos_to_bear"
PRIMARY_GUARD = "ms_state <= 0 with locked 24-candle volume and pressure guards"
TARGET_PROVIDER = "fill-anchored projections of entry-frozen BOS structure height"
INVALIDATION_PROVIDER = "entry-frozen ms_invalidation_level, then exact entry"
ACTIVE_SELL_PARAMS = ("exit_plan",)
STATE_KEY = "s3v2_bos_bear_8h_layered_structure"
PARTIAL_TAG = "s3v2_structure_partial"

LOCKED_BUY_PARAMS: dict[str, Any] = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96, "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24, "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True, "volume_guard_window": 24, "volume_ratio_min": 1.0,
    "use_pressure_guard": True, "pressure_window": 24, "pressure_min": 0.1,
    "use_accumulation_guard": False, "use_body_direction_guard": False,
    "use_close_direction_guard": False, "strength": 3,
    "min_prominence_atr": 0.35, "min_pivot_spacing_bars": 2,
    "max_pivot_age_bars": 96, "breakout_buffer_atr": 0.3,
    "use_state_guard": True,
}

# target, confirmation, fraction of current stake, remainder, trail activation,
# trail distance. A target remainder always exits at measured_full.
EXIT_PLANS = {
    "measured_half_touch_p33_be_measured_full": ("measured_half", "touch", 0.33, "target", 0.0, 0.0),
    "measured_half_close_p50_be_measured_full": ("measured_half", "close", 0.50, "target", 0.0, 0.0),
    "measured_three_quarter_touch_p33_be_measured_full": ("measured_three_quarter", "touch", 0.33, "target", 0.0, 0.0),
    "measured_three_quarter_close_p50_be_measured_full": ("measured_three_quarter", "close", 0.50, "target", 0.0, 0.0),
    "measured_half_touch_p33_be_trail4_1": ("measured_half", "touch", 0.33, "trail", 0.04, 0.01),
    "measured_half_close_p50_be_trail5_1_5": ("measured_half", "close", 0.50, "trail", 0.05, 0.015),
    "measured_three_quarter_touch_p33_be_trail4_1": ("measured_three_quarter", "touch", 0.33, "trail", 0.04, 0.01),
    "measured_three_quarter_close_p50_be_trail5_1_5": ("measured_three_quarter", "close", 0.50, "trail", 0.05, 0.015),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)




class Sieve3V2LayeredStructurePartialBeRemainderFromBosBearContinuationShort8H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 1.0}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    volume_guard_window = 24
    volume_ratio_min = 1.0
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    pressure_window = 24
    pressure_min = 0.1
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed entry gate or mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed entry gate or mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed entry gate or mode makes its branch unreachable.
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    # Inactive buy declaration: use_state_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.

    ACTIVE_SELL_PARAMS = ('exit_plan',)

    exit_plan = CategoricalParameter(tuple(EXIT_PLANS), default="measured_half_touch_p33_be_measured_full", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
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
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        directional_volume = (((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0) * volume).fillna(0.0)
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

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "high", "low", "close", "ms_choch_to_bull"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"missing analyzed columns: {missing}")
        boundary = pd.Timestamp(timeframe_to_prev_date(self.timeframe, current_time))
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        return frame.loc[dates.lt(boundary)].copy()

    def _post_entry_closed(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        filled_at = pd.Timestamp(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        filled_at = filled_at.tz_localize("UTC") if filled_at.tzinfo is None else filled_at.tz_convert("UTC")
        entry_boundary = pd.Timestamp(timeframe_to_prev_date(self.timeframe, filled_at))
        entry_boundary = entry_boundary.tz_localize("UTC") if entry_boundary.tzinfo is None else entry_boundary.tz_convert("UTC")
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        eligible = dates.ge(entry_boundary) if filled_at == entry_boundary else dates.gt(entry_boundary)
        return frame.loc[eligible].copy()

    def _freeze_structure(self, pair: str, trade: Any) -> dict[str, Any]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        required = {"ms_break_level", "ms_invalidation_level"}
        missing = sorted(required.difference(frame.columns))
        if missing or frame.empty:
            raise KeyError(f"cannot freeze source structure; missing={missing}")
        row = frame.iloc[-1]
        entry = float(trade.open_rate)
        break_level = float(row["ms_break_level"])
        invalidation = float(row["ms_invalidation_level"])
        if not all(np.isfinite(value) for value in (break_level, invalidation)):
            raise ValueError("entry structure levels must be finite")
        if invalidation <= break_level:
            raise ValueError("short source invalidation must be above the BOS break")
        structure_height = invalidation - break_level
        targets = {
            "measured_half": entry - structure_height * 0.5,
            "measured_three_quarter": entry - structure_height * 0.75,
            "measured_full": entry - structure_height,
        }
        if any(value >= entry for value in targets.values()):
            raise ValueError("all short structure targets must be below entry")
        state = {
            "invalidation": invalidation,
            "targets": targets,
            "partial_target_confirmed": False,
            "partial_target_confirmed_at": None,
            "partial_filled_at": None,
            "stop_rate": invalidation,
        }
        trade.set_custom_data(STATE_KEY, state)
        return state

    def _state(self, pair: str, trade: Any) -> dict[str, Any]:
        state = trade.get_custom_data(STATE_KEY)
        return dict(state) if isinstance(state, Mapping) else self._freeze_structure(pair, trade)

    def _plan(self) -> tuple[str, str, float, str, float, float]:
        return EXIT_PLANS[str(self.exit_plan.value)]

    @staticmethod
    def _target_reached(frame: DataFrame, target: float, confirmation: str) -> bool:
        if frame.empty:
            return False
        if confirmation == "touch":
            return float(frame.iloc[-1]["low"]) <= target
        if confirmation == "close":
            return float(frame.iloc[-1]["close"]) <= target
        raise ValueError(f"unsupported target confirmation: {confirmation}")

    def _partial_target_confirmed(
        self,
        trade: Any,
        state: dict[str, Any],
        frame: DataFrame,
        target: float,
        confirmation: str,
        current_time: datetime,
    ) -> bool:
        if bool(state.get("partial_target_confirmed")):
            return True
        if not self._target_reached(frame, target, confirmation):
            return False
        state["partial_target_confirmed"] = True
        state["partial_target_confirmed_at"] = pd.Timestamp(current_time).isoformat()
        trade.set_custom_data(STATE_KEY, state)
        return True

    @staticmethod
    def _valid_reduction(stake: float, fraction: float, min_stake: float | None) -> bool:
        reduction = stake * fraction
        remaining = stake - reduction
        if min_stake is None:
            return reduction > 0.0 and remaining > 0.0
        return reduction >= min_stake and remaining >= min_stake

    PARTIAL_STAGE_TAG = "s3v2_structure_partial"
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
        _ = current_rate, current_profit, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if trade.has_open_orders:
            return None
        target_name, confirmation, fraction, _, _, _ = self._plan()
        state = self._state(trade.pair, trade)
        frame = self._post_entry_closed(trade.pair, trade, current_time)
        if not self._partial_target_confirmed(
            trade,
            state,
            frame,
            float(state["targets"][target_name]),
            confirmation,
            current_time,
        ):
            return None
        stake = float(trade.stake_amount)
        if not self._valid_reduction(stake, fraction, min_stake):
            return None
        request = self._request_partial(trade, current_time, stake * fraction, min_stake)
        return (-request, PARTIAL_TAG) if request is not None else None

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_profit, kwargs
        if trade.has_open_orders:
            return None
        completed = self._partial_complete(trade, current_time)
        target_name, confirmation, _, remainder, _, _ = self._plan()
        state = self._state(pair, trade)
        if current_rate >= float(state["invalidation"]):
            return "s3v2_source_invalidation"
        frame = self._post_entry_closed(pair, trade, current_time)
        if not completed and self._partial_target_confirmed(
            trade,
            state,
            frame,
            float(state["targets"][target_name]),
            confirmation,
            current_time,
        ):
            return None
        if completed:
            if not frame.empty and bool(_bool(frame.tail(1), "ms_choch_to_bull").iloc[-1]):
                return "s3v2_choch_after_partial"
            if remainder == "target" and self._target_reached(frame, float(state["targets"]["measured_full"]), "touch"):
                return "s3v2_measured_full_remainder"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = kwargs
        state = self._state(pair, trade)
        desired = float(state["invalidation"])
        completed = self._partial_complete(trade, current_time)
        _, _, _, remainder, trail_activation, trail_ratio = self._plan()
        if completed:
            desired = float(trade.open_rate)
            filled_at_raw = self._partial_state(trade).get("filled_at")
            if remainder == "trail" and filled_at_raw is not None and not after_fill:
                filled_at = pd.Timestamp(filled_at_raw).to_pydatetime()
                trail_ready = current_time >= filled_at + timedelta(hours=8)
                if trail_ready and current_profit >= trail_activation:
                    frame = self._post_entry_closed(pair, trade, current_time)
                    if not frame.empty:
                        favorable_low = float(pd.to_numeric(frame["low"], errors="coerce").min())
                        trail_stop = favorable_low * (1.0 + trail_ratio)
                        desired = min(desired, trail_stop)
        previous = float(state.get("stop_rate", state["invalidation"]))
        desired = min(previous, desired)
        state["stop_rate"] = desired
        trade.set_custom_data(STATE_KEY, state)
        return stoploss_from_absolute(desired, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if order.ft_order_side == trade.entry_side and trade.get_custom_data(STATE_KEY) is None:
            self._freeze_structure(pair, trade)
            return
        if order.ft_order_side == trade.exit_side and order.ft_order_tag == PARTIAL_TAG:
            self._reconcile_partial(trade, current_time, order)
