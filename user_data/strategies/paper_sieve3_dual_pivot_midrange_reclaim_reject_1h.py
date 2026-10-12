"""Paper-only dual-side probe of the selected Sieve3 pivot-midrange entry.

The saved pivot-midrange reclaim long and zone-reversal exit remain the tested
side. The symmetric midrange-reject short is experimental; combinations are
not validated. Pivot levels are emitted only after confirmation and frozen at
entry, matching the saved source's causal target/invalidation behavior.
Selection: entry_sieve_v2_speed_recovery/results/20260724T073022_entry_sieve3_v2_refined_standard_batch_008.jsonl:3;
backtest-result-2026-07-24_08-28-39.zip and best_params.checkpoint.json.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import IStrategy, stoploss_from_absolute

from user_data.Indicators.pivot_foundation import build_clean_pivot_source


TIMEFRAME = "1h"
LOCKED_BUY_PARAMS = {
    "pivot_strength": 4,
    "min_prominence_atr": 0.23,
    "min_pivot_spacing_bars": 1,
    "min_pivot_distance_atr": 1.16,
    "reclaim_buffer_pct": 0.001,
    "score_min": 0.35,  # Saved source marks this entry parameter inert.
    "range_atr_mult": 0.87,
    "freshness_max_bars": 17,  # Saved source marks this entry parameter inert.
}
SELECTED_EXIT_PARAMS = {
    "exit_plan": "zone_reversal",
    "target_band_quarter_percent": 0,
    "reversal_confirmations": 1,
    "invalidation_band_quarter_percent": 2,
    "invalidation_confirmations": 3,
}
HARD_STOP_RATIO = 0.03
MAX_HOLD_CANDLES = 336
ENTRY_TAG_LONG = "tested_sieve3_pivot_midrange_reclaim_long_1h"
ENTRY_TAG_SHORT = "experimental_symmetric_pivot_midrange_reject_short_1h"


def _safe_div(numer: Series, denom: Series) -> Series:
    return numer / denom.replace(0.0, np.nan)


def _normalized_roi(roi: dict[Any, Any]) -> dict[int, float]:
    return {int(duration): float(value) for duration, value in roi.items()}


def _atr(frame: DataFrame, length: int = 14) -> Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length, min_periods=max(2, length // 2)).mean()


class PaperSieve3DualPivotMidrangeReclaimReject1h(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    can_short = True
    startup_candle_count = 360
    process_only_new_candles = True

    pivot_strength = LOCKED_BUY_PARAMS["pivot_strength"]
    min_prominence_atr = LOCKED_BUY_PARAMS["min_prominence_atr"]
    min_pivot_spacing_bars = LOCKED_BUY_PARAMS["min_pivot_spacing_bars"]
    min_pivot_distance_atr = LOCKED_BUY_PARAMS["min_pivot_distance_atr"]
    reclaim_buffer_pct = LOCKED_BUY_PARAMS["reclaim_buffer_pct"]
    score_min = LOCKED_BUY_PARAMS["score_min"]
    range_atr_mult = LOCKED_BUY_PARAMS["range_atr_mult"]
    freshness_max_bars = LOCKED_BUY_PARAMS["freshness_max_bars"]

    minimal_roi = {"0": 100.0}
    # Paper safety floor matches the selected 3% hard stop, not the source's
    # 99% native fallback while open orders suppress custom_stoploss.
    stoploss = -0.03
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False

    FOCUSED_EXIT_CONTRACT = "entry_target_full_or_zone_reversal"
    FOCUSED_SOURCE_PROFILE = {
        "side": "both",
        "min_level_distance": 0.001,
        "targets": {
            "target_1": {
                "provider": "entry_selected_confirmed_pivot_extreme",
                "long": {"level": "pivot_high_last", "available": None},
                "short": {"level": "pivot_low_last", "available": None},
            }
        },
        "invalidation": {
            "provider": "entry_selected_pivot_midrange_loss",
            "mode": "level",
            "long": {"level": "pivot_mid", "available": None},
            "short": {"level": "pivot_mid", "available": None},
        },
    }
    FOCUSED_EXIT_PLANS = {
        "zone_reversal": {
            "contract": FOCUSED_EXIT_CONTRACT,
            "name": "zone_reversal",
            "hard_stop_ratio": HARD_STOP_RATIO,
            "max_hold_candles": MAX_HOLD_CANDLES,
        }
    }
    FOCUSED_REQUIRED_COLUMNS = (
        "close", "date", "high", "low", "open", "pivot_high_last", "pivot_low_last", "pivot_mid",
    )
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = "paper_sieve3_dual_pivot_midrange_v1"

    def bot_start(self, **kwargs: Any) -> None:
        _ = kwargs
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperSieve3DualPivotMidrangeReclaimReject1h is dry-run only")
        expected = {
            "timeframe": TIMEFRAME,
            "startup_candle_count": 360,
            "process_only_new_candles": True,
            "stoploss": -0.03,
            "use_exit_signal": True,
            "exit_profit_only": False,
            "ignore_roi_if_entry_signal": False,
            "use_custom_stoploss": True,
            "position_adjustment_enable": False,
            "max_entry_position_adjustment": 0,
            "trailing_stop": False,
            "can_short": True,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError(f"Frozen Sieve3 paper setting drifted: {name}")
        if _normalized_roi(self.minimal_roi) != {0: 100.0}:
            raise RuntimeError("Frozen Sieve3 paper setting drifted: minimal_roi")
        for name, value in LOCKED_BUY_PARAMS.items():
            if getattr(self, name) != value:
                raise RuntimeError(f"Frozen Sieve3 entry setting drifted: {name}")
        if SELECTED_EXIT_PARAMS != {
            "exit_plan": "zone_reversal",
            "target_band_quarter_percent": 0,
            "reversal_confirmations": 1,
            "invalidation_band_quarter_percent": 2,
            "invalidation_confirmations": 3,
        }:
            raise RuntimeError("Frozen Sieve3 exit parameters drifted")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        frame["atr"] = _atr(frame, 14)
        frame["bar_index"] = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
        pivots = build_clean_pivot_source(
            body_high=pd.to_numeric(frame["high"], errors="coerce"),
            body_low=pd.to_numeric(frame["low"], errors="coerce"),
            atr=pd.to_numeric(frame["atr"], errors="coerce"),
            bar_index=frame["bar_index"],
            strength=self.pivot_strength,
            min_prominence_atr=self.min_prominence_atr,
            min_prominence_pct=0.0,
            min_pivot_spacing_bars=self.min_pivot_spacing_bars,
            min_pivot_distance_atr=self.min_pivot_distance_atr,
            min_pivot_distance_pct=0.0,
        )
        frame = frame.join(pd.DataFrame(pivots))
        frame["pivot_high_last"] = frame["pivot_high"].ffill()
        frame["pivot_low_last"] = frame["pivot_low"].ffill()
        frame["pivot_high_prev"] = frame["pivot_high"].where(frame["pivot_high"].notna()).shift(1).ffill()
        frame["pivot_low_prev"] = frame["pivot_low"].where(frame["pivot_low"].notna()).shift(1).ffill()
        frame["pivot_mid"] = (frame["pivot_high_last"] + frame["pivot_low_last"]) / 2.0
        frame["pivot_range"] = (frame["pivot_high_last"] - frame["pivot_low_last"]).abs()
        frame["pivot_age_high"] = (frame["bar_index"] - frame["pivot_high_available_index"].ffill()).fillna(999.0)
        frame["pivot_age_low"] = (frame["bar_index"] - frame["pivot_low_available_index"].ffill()).fillna(999.0)
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        frame["volume_ratio"] = _safe_div(volume, volume.rolling(20, min_periods=5).mean()).fillna(0.0)
        return frame

    def _entry_conditions(self, frame: DataFrame) -> tuple[Series, Series]:
        close = pd.to_numeric(frame["close"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        atr = pd.to_numeric(frame["atr"], errors="coerce")
        midpoint = pd.to_numeric(frame["pivot_mid"], errors="coerce")
        pivot_range = pd.to_numeric(frame["pivot_range"], errors="coerce")
        buffer = self.reclaim_buffer_pct
        range_ok = pivot_range.ge(atr * self.range_atr_mult)
        reclaim_long = low.lt(midpoint * (1.0 - buffer)) & close.gt(midpoint) & range_ok
        reject_short = high.gt(midpoint * (1.0 + buffer)) & close.lt(midpoint) & range_ok
        return reclaim_long.fillna(False), reject_short.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        long_condition, short_condition = self._entry_conditions(dataframe)
        both = long_condition & short_condition
        long_condition &= ~both
        short_condition &= ~both
        dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG_LONG)
        dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = (1, ENTRY_TAG_SHORT)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @staticmethod
    def _focused_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    @staticmethod
    def _focused_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _focused_number(frame: DataFrame, column: str) -> Series:
        return pd.to_numeric(frame[column], errors="coerce")

    def _focused_close_times(self, frame: DataFrame) -> Series:
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        return dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")

    def _focused_closed_frame(self, frame: DataFrame, current_time: Any) -> DataFrame:
        if frame is None or frame.empty:
            raise RuntimeError("focused exit runtime requires a non-empty analyzed dataframe")
        if "date" not in frame.columns:
            raise KeyError("focused exit runtime requires the dataframe date column")
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError("current_time is required for closed-candle evaluation")
        closed = frame.loc[self._focused_close_times(frame).le(now)].copy()
        return closed.sort_values("date").drop_duplicates("date", keep="last")

    def _focused_analyzed_frame(self, pair: str, current_time: Any) -> DataFrame:
        _ = current_time
        if getattr(self, "dp", None) is None:
            raise RuntimeError("focused exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError("focused exit runtime requires a non-empty analyzed dataframe")
        if "date" not in frame.columns:
            raise KeyError("focused exit runtime requires the dataframe date column")
        return frame

    def _focused_require_columns(self, frame: DataFrame) -> None:
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f"focused source profile is missing exact columns: {missing}")

    def _focused_side(self, trade: Any) -> str:
        side = "short" if bool(getattr(trade, "is_short", False)) else "long"
        if self.FOCUSED_SOURCE_PROFILE["side"] not in {"both", side}:
            raise ValueError(f"trade side {side!r} violates exact profile side")
        return side

    def _focused_plan(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        name = str((state or {}).get("plan") or SELECTED_EXIT_PARAMS["exit_plan"])
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f"unknown focused exit plan: {name}")
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan["target_confirmation"] = f"reversal_{SELECTED_EXIT_PARAMS['reversal_confirmations']}"
        plan["target_band"] = SELECTED_EXIT_PARAMS["target_band_quarter_percent"] * 0.0025
        plan["invalidation_band"] = SELECTED_EXIT_PARAMS["invalidation_band_quarter_percent"] * 0.0025
        plan["invalidation_confirmations"] = SELECTED_EXIT_PARAMS["invalidation_confirmations"]
        return plan

    @staticmethod
    def _focused_decision(action: str = "hold", tag: str | None = None) -> dict[str, Any]:
        return {"action": action, "tag": tag}

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        cache_attr = f"_sieve3_v2_state_cache_{self.FOCUSED_STATE_KEY}"
        state = getattr(trade, cache_attr, None)
        if state is None:
            state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError("focused exit trade state must be a mapping")
        if state.get("version") != self.FOCUSED_STATE_VERSION or state.get("contract") != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError("focused exit trade state version or contract mismatch")
        if getattr(trade, cache_attr, None) is None:
            setattr(trade, cache_attr, deepcopy(state))
        return deepcopy(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        cache_attr = f"_sieve3_v2_state_cache_{self.FOCUSED_STATE_KEY}"
        snapshot = deepcopy(dict(state))
        setattr(trade, cache_attr, snapshot)
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=deepcopy(snapshot))

    def _focused_side_binding(self, role: str, side: str, slot: str | None = None) -> Mapping[str, Any] | None:
        provider = self.FOCUSED_SOURCE_PROFILE.get("targets", {}).get(str(slot)) if role == "target" else self.FOCUSED_SOURCE_PROFILE.get(role)
        if not isinstance(provider, Mapping):
            return None
        binding = provider.get(side)
        return binding if isinstance(binding, Mapping) else None

    def _focused_frozen_level(
        self,
        row: Series,
        binding: Mapping[str, Any] | None,
        side: str,
        entry_rate: float,
        role: str,
    ) -> float | None:
        if binding is None:
            return None
        available_column = binding.get("available")
        if available_column:
            availability = row[str(available_column)]
            if bool(pd.isna(availability)) or not bool(availability):
                return None
        level_column = binding.get("level")
        if not level_column:
            return None
        level = self._focused_float(row[str(level_column)])
        if level is None or level <= 0.0:
            return None
        if role == "invalidation":
            return level
        if role != "target":
            raise ValueError(f"unsupported frozen level role: {role}")
        minimum = self.FOCUSED_SOURCE_PROFILE["min_level_distance"]
        valid = level < entry_rate * (1.0 - minimum) if side == "short" else level > entry_rate * (1.0 + minimum)
        return level if valid else None

    def _focused_new_state(self, pair: str, trade: Any, current_time: Any, order: Any | None = None) -> dict[str, Any] | None:
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, "safe_price", None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, "open_rate", None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError("focused exit runtime requires a positive entry rate")
        freeze_time = (
            (getattr(order, "order_date_utc", None) if order is not None else None)
            or (getattr(order, "order_date", None) if order is not None else None)
            or getattr(trade, "open_date_utc", None)
            or getattr(trade, "date_entry_fill_utc", None)
            or current_time
        )
        frame = self._focused_closed_frame(self._focused_analyzed_frame(pair, freeze_time), freeze_time)
        if frame.empty:
            return None
        self._focused_require_columns(frame)
        row = frame.iloc[-1]
        target_binding = self._focused_side_binding("target", side, "target_1")
        levels = {
            "target_1": self._focused_frozen_level(row, target_binding, side, entry_rate, "target"),
            "invalidation": self._focused_frozen_level(
                row, self._focused_side_binding("invalidation", side), side, entry_rate, "invalidation"
            ),
        }
        if levels["target_1"] is not None and levels["invalidation"] is not None and math.isclose(
            levels["target_1"], levels["invalidation"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("entry-frozen target equals the source invalidation level")
        fill_time = (
            (getattr(order, "order_filled_utc", None) if order is not None else None)
            or getattr(trade, "date_entry_fill_utc", None)
            or getattr(trade, "open_date_utc", None)
            or current_time
        )
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row["date"])
        return {
            "version": self.FOCUSED_STATE_VERSION,
            "contract": self.FOCUSED_EXIT_CONTRACT,
            "plan": SELECTED_EXIT_PARAMS["exit_plan"],
            "side": side,
            "phase": "pre_target",
            "entry_rate": float(entry_rate),
            "entry_filled_at": fill_timestamp.isoformat() if fill_timestamp is not None else None,
            "entry_snapshot_candle": snapshot_date.isoformat() if snapshot_date is not None else None,
            "levels": levels,
            "target_states": {"target_1": {"status": "available" if levels["target_1"] is not None else "unavailable", "touched_at": None}},
            "invalidation_seen": False,
            "invalidation_streak": 0,
            "last_processed_candle": None,
        }

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None = None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any]) -> DataFrame:
        cursor = self._focused_utc(state.get("last_processed_candle") or state.get("entry_snapshot_candle"))
        if cursor is None:
            raise ValueError("focused exit state has no processed-candle cursor")
        filled_at = self._focused_utc(state.get("entry_filled_at"))
        if filled_at is None:
            raise ValueError("focused exit state has no entry fill timestamp")
        latest = self._focused_utc(frame["date"].iat[-1])
        if latest is None or latest <= cursor:
            return frame.iloc[0:0]
        # Skip a fill-straddling OHLC bar: it may hide a true touch, but avoids false pre-fill evidence.
        fill_start = int(frame["date"].searchsorted(filled_at, side="left"))
        start = max(int(frame["date"].searchsorted(cursor, side="right")), fill_start)
        return frame.iloc[start:]

    @staticmethod
    def _focused_hard_stop_price(plan: Mapping[str, Any], state: Mapping[str, Any]) -> float:
        entry_rate = float(state["entry_rate"])
        ratio = float(plan["hard_stop_ratio"])
        return entry_rate * (1.0 + ratio if state["side"] == "short" else 1.0 - ratio)

    def _focused_hard_stop_breached(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_rate: float) -> bool:
        stop_price = self._focused_hard_stop_price(plan, state)
        return current_rate >= stop_price if state["side"] == "short" else current_rate <= stop_price

    def _focused_context(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
        current_profit: float,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return self._focused_plan(), {}, {}, self._focused_decision()
        before = repr(state)
        plan = self._focused_plan(state)
        now = self._focused_utc(current_time)
        filled_at = self._focused_utc(state.get("entry_filled_at"))
        if now is not None and filled_at is not None:
            timeframe_seconds = max(60, timeframe_to_minutes(self.timeframe) * 60)
            age_candles = max(0, int((now - filled_at).total_seconds() // timeframe_seconds))
        else:
            age_candles = 0

        side = state["side"]
        entry_rate = float(state["entry_rate"])
        favorable = self._focused_float(getattr(trade, "min_rate" if side == "short" else "max_rate", None)) or entry_rate
        adverse = self._focused_float(getattr(trade, "max_rate" if side == "short" else "min_rate", None)) or entry_rate
        levels = state.get("levels", {})
        target = self._focused_float(levels.get("target_1")) if isinstance(levels, Mapping) else None
        invalidation = self._focused_float(levels.get("invalidation")) if isinstance(levels, Mapping) else None
        target_state = (state.get("target_states") or {}).get("target_1", {})
        target_status = str(target_state.get("status") or "unavailable") if isinstance(target_state, Mapping) else "unavailable"
        target_done = target_status in {"accepted", "rejected", "confirmed"}

        target_near = False
        if target is not None and not target_done:
            threshold = target * (1.0 + plan["target_band"]) if side == "short" else target * (1.0 - plan["target_band"])
            target_near = target_status == "touched" or (favorable <= threshold if side == "short" else favorable >= threshold)
        invalidation_near = False
        if invalidation is not None and not state.get("invalidation_seen"):
            threshold = invalidation * (1.0 + plan["invalidation_band"]) if side == "short" else invalidation * (1.0 - plan["invalidation_band"])
            invalidation_near = int(state.get("invalidation_streak") or 0) > 0 or (adverse >= threshold if side == "short" else adverse <= threshold)

        if target_near or invalidation_near:
            frame = self._focused_analyzed_frame(pair, current_time)
            self._focused_require_columns(frame)
            post_entry = self._focused_post_entry(frame, state)
            events = self._focused_contract_events(post_entry, state, plan, current_profit)
            if not post_entry.empty:
                processed = self._focused_utc(post_entry["date"].iat[-1])
                state["last_processed_candle"] = processed.isoformat() if processed is not None else None
        else:
            events = {"invalidation": bool(state.get("invalidation_seen")), "target_1": target_done}
        events["age_candles"] = age_candles

        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision("full", "focused_hard_stop")
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision["action"] == "hold" and age_candles >= plan["max_hold_candles"]:
            decision = self._focused_decision("full", "focused_max_hold")
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return plan, state, events, decision

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = kwargs
        _, _, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        return str(decision["tag"]) if decision["action"] == "full" else None

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        plan = self._focused_plan()
        entry_rate = self._focused_float(getattr(trade, "open_rate", None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError("focused exit runtime requires a positive trade open rate")
        is_short = bool(getattr(trade, "is_short", False))
        stop_price = entry_rate * (1.0 + plan["hard_stop_ratio"] if is_short else 1.0 - plan["hard_stop_ratio"])
        state = self._focused_state(trade)
        if state is not None:
            target_state = (state.get("target_states") or {}).get("target_1", {})
            target_done = isinstance(target_state, Mapping) and target_state.get("status") in {"accepted", "rejected", "confirmed"}
            if state.get("invalidation_seen") or target_done:
                return None
        if (is_short and stop_price <= current_rate) or (not is_short and stop_price >= current_rate):
            return None
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=is_short,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )

    def _focused_target_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        target = self._focused_float((state.get("levels") or {}).get("target_1"))
        target_states = state.setdefault("target_states", {})
        target_state = target_states.setdefault("target_1", {"status": "unavailable", "touched_at": None})
        if target_state.get("status") in {"accepted", "rejected", "confirmed"}:
            return True
        if target is None or post_entry.empty:
            if target is None:
                target_state["status"] = "unavailable"
            return False
        band = float(plan["target_band"])
        close_times = self._focused_close_times(post_entry)
        touched_at = self._focused_utc(target_state.get("touched_at"))
        confirmations = int(plan["target_confirmation"].rsplit("_", 1)[1])
        streak = int(target_state.get("reversal_streak") or 0)
        for position in range(len(post_entry)):
            row = post_entry.iloc[position]
            candle_close = self._focused_utc(close_times.iloc[position])
            low = self._focused_float(row["low"])
            high = self._focused_float(row["high"])
            touched = low is not None and low <= target * (1.0 + band) if state["side"] == "short" else high is not None and high >= target * (1.0 - band)
            if touched_at is None and touched:
                touched_at = candle_close
                target_state["touched_at"] = touched_at.isoformat() if touched_at is not None else None
                target_state["status"] = "touched"
                state["phase"] = "target_1_zone"
            if touched_at is None or candle_close is None or candle_close <= touched_at:
                continue
            close = self._focused_float(row["close"])
            open_ = self._focused_float(row["open"])
            reversal = (
                close is not None and open_ is not None
                and (close > open_ and close > target * (1.0 - band) if state["side"] == "short"
                     else close < open_ and close < target * (1.0 + band))
            )
            streak = streak + 1 if reversal else 0
            target_state["reversal_streak"] = streak
            if streak >= confirmations:
                target_state["status"] = "rejected"
                return True
        if touched_at is None:
            target_state["status"] = "available"
        return False

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get("invalidation_seen") or post_entry.empty:
            return bool(state.get("invalidation_seen"))
        level = self._focused_float((state.get("levels") or {}).get("invalidation"))
        if level is None:
            return False
        close = self._focused_number(post_entry, "close")
        band = float(plan["invalidation_band"])
        breached = close.gt(level * (1.0 + band)) if state["side"] == "short" else close.lt(level * (1.0 - band))
        streak = int(state.get("invalidation_streak") or 0)
        confirmations = int(plan["invalidation_confirmations"])
        for value in breached.fillna(False).tolist():
            streak = streak + 1 if bool(value) else 0
            state["invalidation_streak"] = streak
            if streak >= confirmations:
                state["invalidation_seen"] = True
                state["phase"] = "invalidated"
                return True
        return False

    def _focused_contract_events(
        self,
        post_entry: DataFrame,
        state: dict[str, Any],
        plan: Mapping[str, Any],
        current_profit: float,
    ) -> dict[str, Any]:
        _ = current_profit
        invalidation = self._focused_invalidation_event(post_entry, state, plan)
        target = self._focused_target_event(post_entry, state, plan)
        return {"invalidation": invalidation, "target_1": target}

    def _focused_contract_decision(
        self,
        state: dict[str, Any],
        plan: Mapping[str, Any],
        events: Mapping[str, Any],
        current_profit: float,
    ) -> dict[str, Any]:
        _ = (state, plan, current_profit)
        if events.get("invalidation"):
            return self._focused_decision("full", "focused_source_invalidation")
        if events.get("target_1"):
            return self._focused_decision("full", "focused_entry_target")
        return self._focused_decision()

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and self._focused_state(trade) is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is not None:
                self._focused_save_state(trade, state)
