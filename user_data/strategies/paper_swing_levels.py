"""Two isolated 1h PAPER strategies around completed substantial market levels."""
from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import merge_informative_pair
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.strategies.paper_fast_reaction import (
    FEE_AND_SLIPPAGE,
    PLAN_KEY,
    PaperFastAuto,
    validate_fast_plan,
)
from user_data.strategies.integrated_paper_context import CONTEXT_FILE
from user_data.strategies.paper_fast_context import FastControl, load_fast_control

LOGICAL_LEVELS = (
    ("range_high_20", "4h_rolling20_high"),
    ("range_low_20", "4h_rolling20_low"),
    ("vp_poc", "4h_profile_poc"),
    ("vp_hvn_above", "4h_profile_hvn_above"),
    ("vp_hvn_below", "4h_profile_hvn_below"),
    ("day_high", "previous_daily_high"),
    ("day_low", "previous_daily_low"),
)
LEVEL_COLUMNS = tuple(f"{column}_{'1d' if column.startswith('day_') else '4h'}" for column, _ in LOGICAL_LEVELS)
LEVEL_SOURCES = frozenset(source for _, source in LOGICAL_LEVELS)
MAX_ENTRY_SLIPPAGE_ATR = 0.25
FILL_RESERVE_ATR = 0.30  # Confirmation envelope plus a small market-fill/rounding cushion.
MIN_FILL_RESERVE_PCT = 0.0005
MIN_STOP_ATR = 1.5
LEVEL_STOP_BUFFER_ATR = 0.5
INVALIDATION_BUFFER_ATR = 0.25
MIN_LEVEL_TARGET_R = 1.5
VOLATILITY_TARGET_R = 2.0
EMERGENCY_MARGIN_LOSS_PCT = 0.20
COLUMN_SOURCE = {f"{column}_{'1d' if column.startswith('day_') else '4h'}": source
                 for column, source in LOGICAL_LEVELS}
REPORT = CONTEXT_FILE.parent
SWING_CONTEXT_ACCOUNT_KEYS = (
    "swing_level_bounce_context", "swing_level_break_hold_context",
)
SWING_CONTEXT_CONTROL_FILES = {
    key: REPORT / f"{key}_control.json" for key in SWING_CONTEXT_ACCOUNT_KEYS
}
SWING_CONTEXT_CONTROL_JOURNALS = {
    key: REPORT / f"{key}_control_decisions.jsonl" for key in SWING_CONTEXT_ACCOUNT_KEYS
}


class SwingContextControl:
    """Fast schema-1 controls adapted to fixed-3x, stake-only swing decisions."""

    def __init__(self, control: FastControl) -> None:
        self._control = control

    @property
    def status(self):
        return self._control.status

    @property
    def decision_id(self):
        return self._control.decision_id

    @property
    def bias(self):
        return self._control.bias

    @property
    def exposure(self):
        return self._control.exposure

    @property
    def entry_permission(self):
        return self._control.entry_permission

    @property
    def blackouts(self):
        return self._control.blackouts

    @property
    def size_factor(self):
        return 1.0 if self.exposure == "normal" else 0.5

    def leverage_for(self, side: str, now: datetime) -> float:
        """Gate entries without using fast_context's bias-to-leverage table."""
        if (side not in {"long", "short"} or self.status != "observed"
                or self.entry_permission == "paused"
                or any(start <= now < end for start, end in self.blackouts)):
            return 0.0
        return 3.0


def swing_stop_price(side: str, entry_rate: float, atr: float, level: float) -> float:
    """Place protection beyond the frozen level and at least 1.5 entry ATR away."""
    if side not in {"long", "short"} or not all(isfinite(float(x)) for x in (entry_rate, atr, level)):
        raise ValueError("Invalid swing stop inputs")
    if entry_rate <= 0 or atr <= 0:
        raise ValueError("Invalid swing stop inputs")
    if side == "long":
        return min(level - LEVEL_STOP_BUFFER_ATR * atr, entry_rate - MIN_STOP_ATR * atr)
    return max(level + LEVEL_STOP_BUFFER_ATR * atr, entry_rate + MIN_STOP_ATR * atr)


def swing_target_price(
    side: str, entry_rate: float, stop_price: float, target_level: float | None,
) -> tuple[float, str]:
    """Use the next useful level only when its net-of-cost reward is at least 1.5R."""
    if side not in {"long", "short"} or entry_rate <= 0:
        raise ValueError("Invalid swing target inputs")
    risk = abs(entry_rate - stop_price)
    if risk <= 0:
        raise ValueError("Invalid swing target risk")
    direction = 1.0 if side == "long" else -1.0
    if target_level is not None and isfinite(float(target_level)):
        target_level = float(target_level)
        favorable = target_level > entry_rate if side == "long" else target_level < entry_rate
        net_reward = abs(target_level - entry_rate) / entry_rate - FEE_AND_SLIPPAGE
        net_risk = risk / entry_rate + FEE_AND_SLIPPAGE
        if favorable and net_reward >= MIN_LEVEL_TARGET_R * net_risk:
            return target_level, "substantial_level"
    return entry_rate + direction * VOLATILITY_TARGET_R * risk, "two_r_volatility"


def swing_stop_within_emergency(entry_rate: float, stop_price: float, leverage: float) -> bool:
    return (entry_rate > 0 and leverage > 0
            and leverage * abs(entry_rate - stop_price) / entry_rate <= EMERGENCY_MARGIN_LOSS_PCT + 1e-9)


def cap_swing_stop_to_reserve(
    side: str, entry_rate: float, desired_stop: float, stake_usdt: float,
    leverage: float, reserved_loss_usdt: float,
) -> tuple[float, bool]:
    """Tighten—not widen—an out-of-envelope fill to its persisted loss reserve."""
    if side not in {"long", "short"} or min(entry_rate, stake_usdt, leverage, reserved_loss_usdt) <= 0:
        raise ValueError("Invalid swing slippage-reserve inputs")
    actual_loss = stake_usdt * leverage * (abs(entry_rate - desired_stop) / entry_rate + FEE_AND_SLIPPAGE)
    if actual_loss <= reserved_loss_usdt + 1e-6:
        return desired_stop, False
    allowed_distance = (reserved_loss_usdt / (stake_usdt * leverage) - FEE_AND_SLIPPAGE) * entry_rate
    if allowed_distance <= 0:
        raise ValueError("Swing fill consumed its entire persisted risk reserve")
    direction = 1.0 if side == "long" else -1.0
    return entry_rate - direction * allowed_distance, True


def validate_swing_plan(plan: dict, *, actual_leverage: float | None = None) -> None:
    """Validate durable fill-frozen protection and its conservative risk reserve."""
    validate_fast_plan(plan)
    required = ("trigger_level", "level_source", "signal_kind", "target_kind", "stake_usdt")
    if any(key not in plan for key in required):
        raise ValueError("Swing protection plan is incomplete")
    for key in ("trigger_level", "stake_usdt"):
        if not isfinite(float(plan[key])) or float(plan[key]) <= 0:
            raise ValueError(f"Invalid swing protection value: {key}")
    if plan["level_source"] not in LEVEL_SOURCES:
        raise ValueError("Swing protection uses an unapproved level source")
    if plan["signal_kind"] not in {"bounce", "break_hold"}:
        raise ValueError("Swing protection has an unknown signal family")
    if plan["target_kind"] not in {"substantial_level", "two_r_volatility"}:
        raise ValueError("Swing protection has an unknown target type")
    if abs(float(plan["leverage"]) - 3.0) > 1e-9:
        raise ValueError("Swing protection must preserve fixed 3x leverage")
    if actual_leverage is not None and abs(float(actual_leverage) - float(plan["leverage"])) > 1e-9:
        raise ValueError("Swing protection leverage differs from the persisted trade")

    signal_fields = ("signal_at_utc", "signal_close", "confirmation_buffer")
    signal_fields_present = [key in plan for key in signal_fields]
    if any(signal_fields_present):
        if not all(signal_fields_present):
            raise ValueError("Swing signal confirmation metadata is incomplete")
        signal_at = pd.Timestamp(plan["signal_at_utc"])
        if pd.isna(signal_at) or signal_at.tzinfo is None:
            raise ValueError("Swing signal clock must be timezone-aware")
        for key in ("signal_close", "confirmation_buffer"):
            if not isfinite(float(plan[key])) or float(plan[key]) <= 0:
                raise ValueError(f"Invalid swing signal value: {key}")

    side = plan["side"]
    entry = float(plan["open_rate"])
    atr = float(plan["entry_atr"])
    level = float(plan["trigger_level"])
    stop = float(plan["stop_price"])
    target = float(plan["target_price"])
    direction = 1.0 if side == "long" else -1.0
    if side == "long":
        structural = stop <= level - LEVEL_STOP_BUFFER_ATR * atr + 1e-9
    else:
        structural = stop >= level + LEVEL_STOP_BUFFER_ATR * atr - 1e-9
    minimum_room = abs(entry - stop) + 1e-9 >= MIN_STOP_ATR * atr
    emergency_room = swing_stop_within_emergency(entry, stop, float(plan["leverage"]))
    if not structural or not minimum_room:
        if plan.get("protection_adjustment") != "risk_capped_slippage":
            raise ValueError("Swing stop does not preserve its level buffer and minimum ATR room")
    if not emergency_room:
        raise ValueError("Swing stop exceeds the 20% emergency margin-loss ceiling")
    actual_loss = float(plan["stake_usdt"]) * float(plan["leverage"]) * (
        abs(entry - stop) / entry + FEE_AND_SLIPPAGE
    )
    if actual_loss > float(plan["planned_loss_usdt"]) + 1e-6:
        raise ValueError("Filled swing protection exceeds its reserved planned loss")

    reward = abs(target - entry) / entry - FEE_AND_SLIPPAGE
    risk = abs(entry - stop) / entry + FEE_AND_SLIPPAGE
    if reward <= 0 or risk <= 0:
        raise ValueError("Swing target is not favorable after costs")
    if plan["target_kind"] == "substantial_level":
        if plan.get("target_level") is None or not isfinite(float(plan["target_level"])):
            raise ValueError("Swing level target is not frozen")
        if plan.get("target_source") not in LEVEL_SOURCES:
            raise ValueError("Swing level target source is not frozen")
        if abs(target - float(plan["target_level"])) > 1e-9 or reward < MIN_LEVEL_TARGET_R * risk:
            raise ValueError("Swing level target does not meet its cost-adjusted 1.5R floor")
    elif abs(abs(target - entry) - VOLATILITY_TARGET_R * abs(entry - stop)) > 1e-7:
        if plan.get("protection_adjustment") != "risk_capped_slippage":
            raise ValueError("Swing volatility target is not the planned 2R fallback")


class PaperSwingLevels(PaperFastAuto):
    """Shared 1h data and frozen-plan behaviour for the two swing hypotheses."""

    timeframe = "1h"
    startup_candle_count = 400
    stoploss = -EMERGENCY_MARGIN_LOSS_PCT
    paper_hold_hours = 48
    account_key = "swing_level_bounce"
    signal_kind = "bounce"
    control_file: Path | None = None

    def _control(self, now: datetime):
        if not self.contextual:
            return FastControl("observed", "automatic_baseline", 0, "normal", "normal")
        if self.control_file is None:
            raise RuntimeError("Contextual swing account has no account-specific control path")
        return SwingContextControl(load_fast_control(now, path=self.control_file))

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        # Context changes entry size only. The matched swing pair remains fixed at 3x.
        return max(1.0, min(3.0, float(max_leverage)))

    def informative_pairs(self):
        return sorted({(pair, tf) for pair in self.dp.current_whitelist() for tf in ("4h", "1d")})

    def _higher(self, pair: str, tf: str) -> DataFrame:
        source = self.dp.get_pair_dataframe(pair=pair, timeframe=tf).copy()
        if source.empty:
            raise ValueError(f"Missing {tf} candles for {pair}")
        last = source.iloc[-1]
        signature = (len(source), *(last[k] for k in ("date", "open", "high", "low", "close", "volume")))
        cached = self._higher_cache.get((pair, tf))
        if cached is not None and cached[0] == signature:
            return cached[1]
        if tf == "4h":
            source = add_volume_profile(source)
            source["range_high_20"] = source["high"].rolling(20, min_periods=20).max()
            source["range_low_20"] = source["low"].rolling(20, min_periods=20).min()
            columns = ["date", "range_high_20", "range_low_20", "vp_poc", "vp_hvn_above", "vp_hvn_below"]
        elif tf == "1d":
            # merge_informative_pair exposes each daily candle only at its close.
            source = source[["date", "high", "low"]].rename(columns={"high": "day_high", "low": "day_low"})
            columns = ["date", "day_high", "day_low"]
        else:
            raise ValueError(f"Unsupported swing level timeframe: {tf}")
        result = source[columns]
        self._higher_cache[(pair, tf)] = (signature, result)
        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["paper_atr"] = ta.ATR(frame, timeperiod=14)
        frame["prior_volume"] = frame["volume"].shift(1).rolling(36).median()
        pair = metadata["pair"]
        for tf in ("4h", "1d"):
            frame = merge_informative_pair(frame, self._higher(pair, tf), self.timeframe, tf, ffill=True)
        return frame

    @staticmethod
    def _chosen_level(candidates: dict[str, pd.Series], side: str) -> tuple[pd.Series, pd.Series]:
        matrix = pd.DataFrame(candidates)
        valid = matrix.notna().any(axis=1)
        if side == "long":
            price = matrix.max(axis=1)
            source = matrix.fillna(-np.inf).idxmax(axis=1)
        else:
            price = matrix.min(axis=1)
            source = matrix.fillna(np.inf).idxmin(axis=1)
        return price.where(valid), source.where(valid)

    def _signals(self, frame: DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        close = frame["close"]
        atr = frame["paper_atr"]
        zone = 0.25 * atr
        previous = close.shift(1)
        ready = frame[["paper_atr", "prior_volume"]].notna().all(axis=1) & atr.gt(0) & frame["prior_volume"].gt(0)
        participation = frame["volume"].ge(frame["prior_volume"])
        long_candidates, short_candidates = {}, {}

        for column in LEVEL_COLUMNS:
            level = frame[column]
            if self.signal_kind == "bounce":
                frozen = level.shift(1)
                touch = frame["low"].le(frozen + zone) & frame["high"].ge(frozen - zone)
                long_hit = (previous.gt(frozen + zone) & touch & close.gt(frozen + zone)
                            & (close - frozen).le(1.25 * atr))
                short_hit = (previous.lt(frozen - zone) & touch & close.lt(frozen - zone)
                             & (frozen - close).le(1.25 * atr))
                long_candidates[column] = frozen.where(ready & participation & long_hit)
                short_candidates[column] = frozen.where(ready & participation & short_hit)
            else:
                # Freeze the level and buffer before the crossing candle starts.
                frozen = level.shift(2)
                frozen_zone = 0.25 * atr.shift(2)
                crossed_up = close.shift(2).le(frozen + frozen_zone) & close.shift(1).gt(frozen + frozen_zone)
                crossed_down = close.shift(2).ge(frozen - frozen_zone) & close.shift(1).lt(frozen - frozen_zone)
                held_up = close.gt(frozen + frozen_zone) & (close - frozen).le(1.25 * atr)
                held_down = close.lt(frozen - frozen_zone) & (frozen - close).le(1.25 * atr)
                long_candidates[column] = frozen.where(ready & participation & crossed_up & held_up)
                short_candidates[column] = frozen.where(ready & participation & crossed_down & held_down)

        long_level, long_source = self._chosen_level(long_candidates, "long")
        short_level, short_source = self._chosen_level(short_candidates, "short")
        long_hit, short_hit = long_level.notna(), short_level.notna()
        conflict = long_hit & short_hit
        long_level, long_source = long_level.where(~conflict), long_source.where(~conflict)
        short_level, short_source = short_level.where(~conflict), short_source.where(~conflict)
        return long_level, long_source, short_level, short_source

    def _target_levels(self, frame: DataFrame, side: str) -> tuple[pd.Series, pd.Series]:
        close = frame["close"]
        levels = {column: frame[column] for column in LEVEL_COLUMNS}
        if side == "long":
            candidates = {column: value.where(value.gt(close)) for column, value in levels.items()}
            matrix = pd.DataFrame(candidates)
            valid = matrix.notna().any(axis=1)
            source = matrix.fillna(np.inf).idxmin(axis=1).where(valid)
            return matrix.min(axis=1).where(valid), source
        candidates = {column: value.where(value.lt(close)) for column, value in levels.items()}
        matrix = pd.DataFrame(candidates)
        valid = matrix.notna().any(axis=1)
        source = matrix.fillna(-np.inf).idxmax(axis=1).where(valid)
        return matrix.max(axis=1).where(valid), source

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe
        frame["enter_long"], frame["enter_short"], frame["enter_tag"] = 0, 0, None
        frame["fast_trigger_level"] = np.nan
        frame["fast_strong"] = False
        frame["swing_level_source"], frame["swing_target_source"] = None, None
        frame["swing_target_level"] = np.nan
        frame["swing_signal_close"], frame["swing_confirmation_buffer"] = np.nan, np.nan
        frame["swing_signal_at"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        frame["swing_signal_side"] = None
        long_level, long_source, short_level, short_source = self._signals(frame)
        long_target, long_target_source = self._target_levels(frame, "long")
        short_target, short_target_source = self._target_levels(frame, "short")
        long, short = long_level.notna(), short_level.notna()
        frame.loc[long, "enter_long"] = 1
        frame.loc[short, "enter_short"] = 1
        label = "bounce" if self.signal_kind == "bounce" else "break_hold"
        frame.loc[long, "enter_tag"] = f"swing_{label}_long"
        frame.loc[short, "enter_tag"] = f"swing_{label}_short"
        frame.loc[long, "fast_trigger_level"] = long_level[long]
        frame.loc[short, "fast_trigger_level"] = short_level[short]
        frame.loc[long, "swing_level_source"] = long_source[long].map(COLUMN_SOURCE)
        frame.loc[short, "swing_level_source"] = short_source[short].map(COLUMN_SOURCE)
        frame.loc[long, "swing_target_level"] = long_target[long]
        frame.loc[short, "swing_target_level"] = short_target[short]
        frame.loc[long, "swing_target_source"] = long_target_source[long].map(COLUMN_SOURCE)
        frame.loc[short, "swing_target_source"] = short_target_source[short].map(COLUMN_SOURCE)
        signal = long | short
        confirmation_atr = frame["paper_atr"].shift(2) if self.signal_kind == "break_hold" else frame["paper_atr"]
        frame.loc[signal, "swing_signal_close"] = frame.loc[signal, "close"]
        frame.loc[signal, "swing_confirmation_buffer"] = (0.25 * confirmation_atr[signal])
        frame.loc[signal, "swing_signal_at"] = (
            pd.to_datetime(frame.loc[signal, "date"], utc=True) + pd.Timedelta(hours=1)
        )
        frame.loc[long, "swing_signal_side"] = "long"
        frame.loc[short, "swing_signal_side"] = "short"
        frame["fast_strong"] = frame["volume"].ge(1.5 * frame["prior_volume"]) & (frame["close"]-frame["open"]).abs().ge(.5*frame["paper_atr"])
        return frame

    def _stop_price_for_entry(self, side: str, entry_rate: float, atr: float, level: float) -> float:
        return swing_stop_price(side, entry_rate, atr, level)

    def _target_price_for_entry(
        self, side: str, entry_rate: float, stop_price: float, target_level: float | None,
    ) -> tuple[float, str]:
        return swing_target_price(side, entry_rate, stop_price, target_level)

    def _entry_plan_context(self, row: pd.Series, side: str) -> dict:
        level_source = row.get("swing_level_source")
        if level_source not in LEVEL_SOURCES:
            raise ValueError("Swing entry is missing its frozen substantial level source")
        raw_target = row.get("swing_target_level")
        target_level = None if raw_target is None or pd.isna(raw_target) else float(raw_target)
        if target_level is not None and (not isfinite(target_level) or target_level <= 0):
            raise ValueError("Swing entry has an invalid favorable target level")
        target_source = row.get("swing_target_source")
        if target_level is None:
            target_source = None
        elif target_source not in LEVEL_SOURCES:
            raise ValueError("Swing entry has a target level without its frozen source")
        signal_at = pd.Timestamp(row.get("swing_signal_at"))
        if pd.isna(signal_at):
            raise ValueError("Swing entry is missing its confirming H1 close clock")
        if signal_at.tzinfo is None:
            signal_at = signal_at.tz_localize("UTC")
        signal_close = float(row.get("swing_signal_close"))
        confirmation_buffer = float(row.get("swing_confirmation_buffer"))
        if (row.get("swing_signal_side") != side or not isfinite(signal_close)
                or signal_close <= 0 or not isfinite(confirmation_buffer)
                or confirmation_buffer <= 0):
            raise ValueError("Swing entry is missing its exact confirmation evidence")
        return {"level_source": level_source, "signal_kind": self.signal_kind,
            "target_level": target_level, "target_source": target_source,
            "signal_at_utc": signal_at.isoformat(), "signal_close": signal_close,
            "confirmation_buffer": confirmation_buffer}

    @staticmethod
    def _entry_price_gate(*, signal_at, signal_close, confirmation_buffer,
                          signal_atr, level, signal_side, current_rate,
                          side, current_time) -> bool:
        """Enforce the original H1 confirmation and bounded post-close entry window."""
        if side not in {"long", "short"} or signal_side != side:
            return False
        try:
            signal_clock = pd.Timestamp(signal_at)
            now = pd.Timestamp(current_time)
            if pd.isna(signal_clock) or pd.isna(now):
                return False
            if signal_clock.tzinfo is None:
                signal_clock = signal_clock.tz_localize("UTC")
            else:
                signal_clock = signal_clock.tz_convert("UTC")
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            else:
                now = now.tz_convert("UTC")
            signal_close = float(signal_close)
            confirmation_buffer = float(confirmation_buffer)
            signal_atr = float(signal_atr)
            level = float(level)
            current_rate = float(current_rate)
        except (TypeError, ValueError, OverflowError):
            return False
        if (not all(isfinite(value) for value in
                    (signal_close, confirmation_buffer, signal_atr, level, current_rate))
                or min(signal_close, confirmation_buffer, signal_atr, level, current_rate) <= 0
                or not timedelta(0) <= now - signal_clock <= timedelta(minutes=5)
                or abs(current_rate - signal_close) > MAX_ENTRY_SLIPPAGE_ATR * signal_atr):
            return False
        return (current_rate > level + confirmation_buffer if side == "long"
                else current_rate < level - confirmation_buffer)

    def _entry_row_allowed(self, row: pd.Series, current_rate: float,
                           side: str, current_time: datetime) -> bool:
        return self._entry_price_gate(
            signal_at=row.get("swing_signal_at"),
            signal_close=row.get("swing_signal_close"),
            confirmation_buffer=row.get("swing_confirmation_buffer"),
            signal_atr=row.get("paper_atr"),
            level=row.get("fast_trigger_level"),
            signal_side=row.get("swing_signal_side"),
            current_rate=current_rate, side=side, current_time=current_time,
        )

    def _entry_plan_allowed(self, plan: dict, current_rate: float,
                            side: str, current_time: datetime) -> bool:
        return self._entry_price_gate(
            signal_at=plan.get("signal_at_utc"),
            signal_close=plan.get("signal_close"),
            confirmation_buffer=plan.get("confirmation_buffer"),
            signal_atr=plan.get("entry_atr"),
            level=plan.get("trigger_level"),
            signal_side=plan.get("side"),
            current_rate=current_rate, side=side, current_time=current_time,
        )

    @staticmethod
    def _directional_size_factor(bias: int, side: str) -> float:
        if type(bias) is not int or bias not in {-2, -1, 0, 1, 2} or side not in {"long", "short"}:
            raise ValueError("Invalid contextual swing sizing inputs")
        if bias == 0:
            return 1.0
        favored_side = "long" if bias > 0 else "short"
        if side == favored_side:
            return 1.0
        return 0.75 if abs(bias) == 1 else 0.5

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        key = (pair, side, str(entry_tag))
        try:
            row = self._current_row(pair, current_time)
        except (KeyError, TypeError, ValueError, OverflowError):
            self._pending.pop(key, None)
            return 0.0
        if not self._entry_row_allowed(row, current_rate, side, current_time):
            self._pending.pop(key, None)
            return 0.0

        stake = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake, min_stake,
            max_stake, leverage, entry_tag, side, **kwargs,
        )
        if not self.contextual or stake <= 0:
            return stake

        pending = self._pending.get(key)
        control = self._control(current_time)
        if (pending is None or control.status != "observed"
                or pending.get("decision_id") != control.decision_id):
            self._pending.pop(key, None)
            return 0.0
        directional = self._directional_size_factor(control.bias, side)
        scaled_stake = float(stake) * directional
        if (not isfinite(scaled_stake) or scaled_stake <= 0
                or (min_stake is not None and scaled_stake + 1e-9 < float(min_stake))):
            self._pending.pop(key, None)
            return 0.0
        pending["stake_usdt"] = float(pending["stake_usdt"]) * directional
        pending["planned_loss_usdt"] = float(pending["planned_loss_usdt"]) * directional
        pending.update(
            context_bias=control.bias,
            context_exposure=control.exposure,
            context_directional_size_factor=directional,
            context_effective_size_factor=control.size_factor * directional,
        )
        return scaled_stake

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        key = (pair, side, str(entry_tag))
        pending = self._pending.get(key)
        if pending is None or not self._entry_plan_allowed(pending, rate, side, current_time):
            return False
        if self.contextual:
            control = self._control(current_time)
            if (control.status != "observed"
                    or control.decision_id != pending.get("decision_id")
                    or control.bias != pending.get("context_bias")
                    or control.exposure != pending.get("context_exposure")
                    or self._directional_size_factor(control.bias, side)
                       != pending.get("context_directional_size_factor")
                    or control.size_factor * self._directional_size_factor(control.bias, side)
                       != pending.get("context_effective_size_factor")):
                return False
        return super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, current_time,
            entry_tag, side, **kwargs,
        )

    def _sizing_geometry(
        self, side: str, reference_rate: float, atr: float, level: float,
        target_level: float | None, leverage: float,
    ) -> dict:
        if reference_rate <= 0 or atr <= 0 or leverage <= 0:
            raise ValueError("Invalid swing sizing inputs")
        if abs(leverage - 3.0) > 1e-9:
            raise ValueError("Swing entries require the approved fixed 3x leverage")
        fill_radius = max(FILL_RESERVE_ATR * atr, MIN_FILL_RESERVE_PCT * reference_rate)
        fill_bounds = (reference_rate - fill_radius, reference_rate + fill_radius)
        candidates = []
        for reserve_rate in fill_bounds:
            if reserve_rate <= 0:
                raise ValueError("Invalid swing fill-reserve bounds")
            stop = self._stop_price_for_entry(side, reserve_rate, atr, level)
            if not swing_stop_within_emergency(reserve_rate, stop, leverage):
                raise ValueError("Planned swing stop exceeds the 20% emergency margin-loss ceiling")
            unit_loss = leverage * (abs(reserve_rate - stop) / reserve_rate + FEE_AND_SLIPPAGE)
            candidates.append((unit_loss, reserve_rate, stop))
        unit_loss, reserve_rate, stop = max(candidates, key=lambda candidate: candidate[0])
        target, target_kind = self._target_price_for_entry(side, reserve_rate, stop, target_level)
        return {"reserved_entry_rate": reserve_rate, "reserved_stop_price": stop,
            "reserved_target_price": target, "target_kind": target_kind,
            "fill_rate_bounds": fill_bounds, "unit_loss_fraction": unit_loss}

    def _filled_plan_after_entry(self, pending: dict, trade, current_time: datetime) -> dict:
        """Freeze protection at the exchange-reported fill without exceeding reserve."""
        plan = {key: value for key, value in pending.items() if key not in {"at", "confirmed"}}
        side = "short" if trade.is_short else "long"
        entry = float(trade.open_rate)
        atr = float(pending["entry_atr"])
        level = float(pending["trigger_level"])
        stake = float(trade.stake_amount)
        leverage = float(trade.leverage)
        desired_stop = self._stop_price_for_entry(side, entry, atr, level)
        stop, capped = cap_swing_stop_to_reserve(
            side, entry, desired_stop, stake, leverage, float(pending["planned_loss_usdt"]),
        )
        target_level = pending.get("target_level")
        target, target_kind = self._target_price_for_entry(side, entry, stop, target_level)
        plan.update(
            stop_price=stop,
            target_price=target,
            target_kind=target_kind,
            open_rate=entry,
            stake_usdt=stake,
            leverage=leverage,
            filled_at_utc=current_time.isoformat(),
        )
        if capped:
            # The fill exceeded its reserved envelope. Keep the tightened
            # protective stop, then let the normal strategy loop exit promptly.
            plan["protection_adjustment"] = "risk_capped_slippage"
        if target_kind == "two_r_volatility":
            # A nearby level that failed the net-R threshold is only a rejected
            # candidate, not the plan's target; do not present it as one.
            plan["target_level"] = None
            plan["target_source"] = None
        return plan

    def _current_row(self, pair: str, now: datetime):
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise ValueError(f"No analyzed swing candle for {pair}")
        row = frame.iloc[-1]
        duration = timedelta(hours=1)
        closed_at = pd.Timestamp(row["date"]) + duration
        now_ts = pd.Timestamp(now)
        if closed_at.tzinfo is None and now_ts.tzinfo is not None:
            closed_at = closed_at.tz_localize("UTC")
        elif closed_at.tzinfo is not None and now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        if not timedelta(0) <= now_ts - closed_at < timedelta(hours=1):
            raise ValueError(f"Stale/future swing candle for {pair}")
        return row

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        plan = trade.get_custom_data(PLAN_KEY)
        validate_swing_plan(plan, actual_leverage=float(trade.leverage))
        target = float(plan["target_price"])
        if (trade.is_short and current_rate <= target) or (not trade.is_short and current_rate >= target):
            return "swing_price_target"
        if current_time - trade.open_date_utc >= timedelta(hours=48):
            return "swing_time_limit"
        if plan.get("protection_adjustment") == "risk_capped_slippage":
            return "swing_slippage_risk_cap"

        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            return None
        closes_at = pd.to_datetime(frame["date"], utc=True) + pd.Timedelta(hours=1)
        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        opened = pd.Timestamp(trade.open_date_utc)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        completed = frame.loc[(closes_at > opened) & (closes_at <= now), "close"].tail(2)
        if len(completed) < 2:
            return None
        level = float(plan["trigger_level"])
        buffer = INVALIDATION_BUFFER_ATR * float(plan["entry_atr"])
        if trade.is_short:
            invalid = completed.gt(level + buffer).all()
        else:
            invalid = completed.lt(level - buffer).all()
        return "swing_level_invalidated" if bool(invalid) else None


class PaperSwingLevelBounce(PaperSwingLevels):
    account_key = "swing_level_bounce"
    signal_kind = "bounce"


class PaperSwingLevelBreakHold(PaperSwingLevels):
    account_key = "swing_level_break_hold"
    signal_kind = "break_hold"


class PaperSwingLevelBounceContext(PaperSwingLevelBounce):
    account_key = "swing_level_bounce_context"
    contextual = True
    control_file = SWING_CONTEXT_CONTROL_FILES[account_key]


class PaperSwingLevelBreakHoldContext(PaperSwingLevelBreakHold):
    account_key = "swing_level_break_hold_context"
    contextual = True
    control_file = SWING_CONTEXT_CONTROL_FILES[account_key]
