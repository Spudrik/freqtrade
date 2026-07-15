"""Codex Sieve3 strategy file.

Testing focus:
- Short after price rejects prior rolling volume-profile POC; cover into prior value low unless rejection accepts back through the level.
- Entry trigger: vp_prior_poc_reject; guard: negative volume pressure plus value-side rejection.
- Target provider: volume profile VAL; invalidation provider: vwap reclaim with bullish pressure.
- Hyperopt surface: trader-readable categorical entry/management modes, with explicit target, stop, partial, trailing, and guard-response choices.
- Scope: standalone top-level Sieve3 strategy using OHLCV/project indicator evidence only; parked orderbook, FreqAI, and news/context sources are not used.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile


SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "codex_novel_theory"
SOURCE_RESULT_BATCH = "codex_sieve3_nuanced_generation_20260617"
RESEARCH_PATH = "codex_sieve3_nuanced_single_theory"
ENTRY_SOURCE_STAGE = "codex_novel"
EXIT_HYPOTHESIS = "Short after price rejects prior rolling volume-profile POC; cover into prior value low unless rejection accepts back through the level."
ENTRY_TAG = "codex_sieve3_vp_poc_reject_target_short_1h"
SOURCE_ENTRY_STEM = "codex_sieve3_vp_poc_reject_target_short_1h"
SOURCE_ENTRY_CLASS = "CodexSieve3VPPOCRejectTargetShortTf1H"
PRIMARY_TRIGGER = "vp_prior_poc_reject"
PRIMARY_GUARD = "negative volume pressure plus value-side rejection"
TARGET_PROVIDER = "volume profile VAL"
INVALIDATION_PROVIDER = "vwap reclaim with bullish pressure"
BRANCH_SPLIT_RATIONALE = "One strategy file tests one explicit entry and management theory with compact categorical state/action modes."

SIDE = "short"
TIMEFRAME = "1h"
ENTRY_LEVEL_KIND = "vp_prior_poc"
TRIGGER_KIND = "reject"
TARGET_KIND = "vp_prior_val"
STRUCTURE_MEMORY = 72
USES_VOLUME_PROFILE = True

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
LOOKBACKS = (24, 48, 96, 168, 336)


def split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:
    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    if not selected:
        return
    for name in dir(strategy_cls):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = str(name) in selected


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _cross_above(left: Series, right: Series) -> Series:
    return left.gt(right) & left.shift(1).le(right.shift(1))


def _cross_below(left: Series, right: Series) -> Series:
    return left.lt(right) & left.shift(1).ge(right.shift(1))


def _true_range(dataframe: DataFrame) -> Series:
    high = _num(dataframe, "high")
    low = _num(dataframe, "low")
    close = _num(dataframe, "close")
    return pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)


def _profit_bucket(current_profit: float) -> str:
    profit = float(current_profit or 0.0)
    if profit <= -0.015:
        return "loss_beyond_half_stop"
    if profit < -0.002:
        return "small_loss"
    if profit < 0.005:
        return "flat"
    if profit < 0.010:
        return "profit_0_5"
    if profit < 0.020:
        return "profit_1"
    if profit < 0.030:
        return "profit_2"
    return "profit_3_plus"


class CodexSieve3VPPOCRejectTargetShortTf1H(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = TIMEFRAME
    startup_candle_count = 380
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    entry_plan = CategoricalParameter(
        ["balanced", "strict_parallel", "trigger_first_context_soft", "retest_only", "acceptance_followthrough"],
        default="balanced",
        space="buy",
        optimize=True,
        load=True,
    )
    context_mode = CategoricalParameter(["ignore", "soft", "aligned", "strict"], default="aligned", space="buy", optimize=True, load=True)
    trigger_quality_mode = CategoricalParameter(
        ["normal", "require_pressure", "require_retest", "require_expansion", "require_wick"],
        default="normal",
        space="buy",
        optimize=True,
        load=True,
    )
    context_lookback = CategoricalParameter([48, 96, 168, 336], default=96, space="buy", optimize=True, load=True)
    trigger_buffer_pct = CategoricalParameter([0.002, 0.004, 0.006, 0.010], default=0.004, space="buy", optimize=True, load=True)
    volume_pressure_mode = CategoricalParameter(["ignore", "soft", "aligned", "strict"], default="soft", space="buy", optimize=True, load=True)

    target_action_mode = CategoricalParameter(
        ["full_on_touch", "full_on_reject", "partial_33_then_be", "partial_50_then_trail", "runner_until_guard_flip", "tighten_only"],
        default="partial_33_then_be",
        space="sell",
        optimize=True,
        load=True,
    )
    trigger_guard_response_mode = CategoricalParameter(
        [
            "ignore_guard_unless_trigger_fails",
            "guard_weak_tighten_if_profit",
            "guard_flip_reduce_if_profit",
            "trigger_fail_lock_if_profit",
            "trigger_fail_full_if_guard_opposes",
        ],
        default="guard_weak_tighten_if_profit",
        space="sell",
        optimize=True,
        load=True,
    )
    pnl_overlay_mode = CategoricalParameter(
        ["off", "loss_protect", "profit_lock", "profit_scale", "asymmetric_loss_hold_profit_reduce"],
        default="profit_lock",
        space="sell",
        optimize=True,
        load=True,
    )
    target_distance_mode = CategoricalParameter(
        ["ignore_distance", "skip_if_too_close", "require_useful", "cap_far", "nearest_only"],
        default="require_useful",
        space="sell",
        optimize=True,
        load=True,
    )
    stop_movement_mode = CategoricalParameter(
        ["fixed_3pct", "fixed_4pct", "target_touch_be", "guard_tighten", "trail_after_target", "structure_then_be"],
        default="target_touch_be",
        space="sell",
        optimize=True,
        load=True,
    )
    partial_plan = CategoricalParameter(
        ["none", "p33_at_target", "p50_at_target", "p33_at_2pct", "p25_at_1pct_then_runner"],
        default="p33_at_target",
        space="sell",
        optimize=True,
        load=True,
    )
    trail_plan = CategoricalParameter(
        ["off", "after_target_1pct", "after_target_1_5pct", "after_partial_1pct", "structure_lookback"],
        default="after_target_1pct",
        space="sell",
        optimize=True,
        load=True,
    )
    time_exit_mode = CategoricalParameter(
        ["off", "stale_flat_exit", "stale_lock_profit", "stale_tighten_only"],
        default="stale_lock_profit",
        space="sell",
        optimize=True,
        load=True,
    )

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
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        if USES_VOLUME_PROFILE:
            frame = add_volume_profile(
                frame,
                window=int(self.context_lookback.value),
                bins=36,
                value_area_pct=0.70,
                prefix="vp",
            )
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        close = _num(frame, "close")
        volume = _num(frame, "volume").clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)

        frame["s3_atr"] = _true_range(frame).rolling(14, min_periods=5).mean()
        frame["s3_body_pressure"] = ((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0)
        frame["s3_close_location"] = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)
        frame["s3_pressure"] = ((frame["s3_body_pressure"] + frame["s3_close_location"]) / 2.0).clip(-1.0, 1.0)
        frame["s3_directional_volume"] = frame["s3_pressure"] * volume
        frame["s3_lower_wick_ratio"] = ((np.minimum(open_, close) - low) / candle_range).clip(0.0, 1.0).fillna(0.0)
        frame["s3_upper_wick_ratio"] = ((high - np.maximum(open_, close)) / candle_range).clip(0.0, 1.0).fillna(0.0)
        frame["s3_range_expand"] = candle_range / frame["s3_atr"].replace(0.0, np.nan)

        for lookback in LOOKBACKS:
            prior_volume = volume.shift(1)
            volume_sum = prior_volume.rolling(lookback, min_periods=max(4, lookback // 4)).sum().replace(0.0, np.nan)
            frame[f"s3_prior_high_{lookback}"] = high.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).max()
            frame[f"s3_prior_low_{lookback}"] = low.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).min()
            frame[f"s3_range_mid_{lookback}"] = (frame[f"s3_prior_high_{lookback}"] + frame[f"s3_prior_low_{lookback}"]) / 2.0
            frame[f"s3_vwap_{lookback}"] = (close.shift(1) * prior_volume).rolling(lookback, min_periods=max(4, lookback // 4)).sum() / volume_sum
            frame[f"s3_value_high_{lookback}"] = frame[f"s3_vwap_{lookback}"] + frame["s3_atr"].mul(1.2)
            frame[f"s3_value_low_{lookback}"] = frame[f"s3_vwap_{lookback}"] - frame["s3_atr"].mul(1.2)
            frame[f"s3_pressure_ratio_{lookback}"] = (
                frame["s3_directional_volume"].rolling(lookback, min_periods=max(4, lookback // 4)).sum()
                / volume.rolling(lookback, min_periods=max(4, lookback // 4)).sum().replace(0.0, np.nan)
            )
            frame[f"s3_volume_ratio_{lookback}"] = volume / volume.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).mean().replace(0.0, np.nan)
            frame[f"s3_compression_high_{lookback}"] = high.shift(1).rolling(max(8, lookback // 2), min_periods=4).max()
            frame[f"s3_compression_low_{lookback}"] = low.shift(1).rolling(max(8, lookback // 2), min_periods=4).min()
            frame[f"s3_compression_width_{lookback}"] = (
                frame[f"s3_compression_high_{lookback}"] - frame[f"s3_compression_low_{lookback}"]
            ) / frame["s3_atr"].replace(0.0, np.nan)

        strength = 3
        rolling_high = high.rolling(strength * 2 + 1, min_periods=strength + 1).max()
        rolling_low = low.rolling(strength * 2 + 1, min_periods=strength + 1).min()
        confirmed_high = high.shift(strength).where(high.shift(strength).eq(rolling_high))
        confirmed_low = low.shift(strength).where(low.shift(strength).eq(rolling_low))
        frame["s3_confirmed_pivot_high"] = confirmed_high.ffill()
        frame["s3_confirmed_pivot_low"] = confirmed_low.ffill()
        return frame

    def _lookback(self) -> int:
        return int(self.context_lookback.value)

    def _level(self, dataframe: DataFrame, kind: str, side: str | None = None) -> Series:
        side = side or SIDE
        lookback = self._lookback()
        if kind == "vwap":
            return _num(dataframe, f"s3_vwap_{lookback}", np.nan)
        if kind == "vp_poc":
            return _num(dataframe, "vp_poc", _num(dataframe, f"s3_vwap_{lookback}", np.nan))
        if kind == "vp_vah":
            return _num(dataframe, "vp_vah", _num(dataframe, f"s3_value_high_{lookback}", np.nan))
        if kind == "vp_val":
            return _num(dataframe, "vp_val", _num(dataframe, f"s3_value_low_{lookback}", np.nan))
        if kind == "vp_prior_poc":
            return _num(dataframe, "vp_prior_poc", _num(dataframe, f"s3_vwap_{lookback}", np.nan))
        if kind == "vp_prior_vah":
            return _num(dataframe, "vp_prior_vah", _num(dataframe, f"s3_value_high_{lookback}", np.nan))
        if kind == "vp_prior_val":
            return _num(dataframe, "vp_prior_val", _num(dataframe, f"s3_value_low_{lookback}", np.nan))
        if kind == "vp_prior_mid":
            vah = _num(dataframe, "vp_prior_vah", _num(dataframe, f"s3_value_high_{lookback}", np.nan))
            val = _num(dataframe, "vp_prior_val", _num(dataframe, f"s3_value_low_{lookback}", np.nan))
            return (vah + val) / 2.0
        if kind == "next_prior_high":
            level = _num(dataframe, f"s3_prior_high_{lookback}", np.nan)
            current = _num(dataframe, "close")
            atr = _num(dataframe, "s3_atr").replace(0.0, np.nan)
            return level.where(level.gt(current + atr.mul(0.5)), _num(dataframe, f"s3_value_high_{lookback}", level))
        if kind == "next_prior_low":
            level = _num(dataframe, f"s3_prior_low_{lookback}", np.nan)
            current = _num(dataframe, "close")
            atr = _num(dataframe, "s3_atr").replace(0.0, np.nan)
            return level.where(level.lt(current - atr.mul(0.5)), _num(dataframe, f"s3_value_low_{lookback}", level))
        if kind == "value_low":
            return _num(dataframe, f"s3_value_low_{lookback}", np.nan)
        if kind == "value_high":
            return _num(dataframe, f"s3_value_high_{lookback}", np.nan)
        if kind == "range_mid":
            return _num(dataframe, f"s3_range_mid_{lookback}", np.nan)
        if kind == "prior_high":
            return _num(dataframe, f"s3_prior_high_{lookback}", np.nan)
        if kind == "prior_low":
            return _num(dataframe, f"s3_prior_low_{lookback}", np.nan)
        if kind == "compression_high":
            return _num(dataframe, f"s3_compression_high_{lookback}", np.nan)
        if kind == "compression_low":
            return _num(dataframe, f"s3_compression_low_{lookback}", np.nan)
        if kind == "confirmed_pivot":
            return _num(dataframe, "s3_confirmed_pivot_low" if side == "short" else "s3_confirmed_pivot_high", np.nan)
        if kind == "entry_opposite_pivot":
            return _num(dataframe, "s3_confirmed_pivot_high" if side == "short" else "s3_confirmed_pivot_low", np.nan)
        return _num(dataframe, f"s3_range_mid_{lookback}", np.nan)

    def _target_level(self, dataframe: DataFrame) -> Series:
        return self._level(dataframe, TARGET_KIND, SIDE)

    def _entry_level(self, dataframe: DataFrame) -> Series:
        return self._level(dataframe, ENTRY_LEVEL_KIND, SIDE)

    def _pressure_ok(self, dataframe: DataFrame, mode: str) -> Series:
        if mode == "ignore":
            return pd.Series(True, index=dataframe.index)
        lookback = self._lookback()
        pressure = _num(dataframe, f"s3_pressure_ratio_{lookback}", 0.0)
        volume_ratio = _num(dataframe, f"s3_volume_ratio_{lookback}", 0.0)
        threshold = {"soft": 0.02, "aligned": 0.06, "strict": 0.12}.get(mode, 0.06)
        volume_min = {"soft": 0.8, "aligned": 1.0, "strict": 1.25}.get(mode, 1.0)
        if SIDE == "short":
            return pressure.le(-threshold) & volume_ratio.ge(volume_min)
        return pressure.ge(threshold) & volume_ratio.ge(volume_min)

    def _context_ok(self, dataframe: DataFrame) -> Series:
        mode = str(self.context_mode.value)
        if mode == "ignore":
            return pd.Series(True, index=dataframe.index)
        close = _num(dataframe, "close")
        lookback = self._lookback()
        vwap = _num(dataframe, f"s3_vwap_{lookback}", np.nan)
        mid = _num(dataframe, f"s3_range_mid_{lookback}", np.nan)
        soft = close.le(vwap) | close.le(mid) if SIDE == "short" else close.ge(vwap) | close.ge(mid)
        aligned = (close.le(vwap) & close.le(mid)) if SIDE == "short" else (close.ge(vwap) & close.ge(mid))
        pressure = self._pressure_ok(dataframe, str(self.volume_pressure_mode.value))
        if mode == "soft":
            return soft & pressure
        if mode == "strict":
            return aligned & self._pressure_ok(dataframe, "strict")
        return aligned & pressure

    def _entry_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = self._entry_level(dataframe)
        buffer = float(self.trigger_buffer_pct.value)
        near = buffer * 1.5
        if TRIGGER_KIND == "reclaim":
            return low.le(level.mul(1.0 + near)) & close.gt(level.mul(1.0 + buffer)) & close.gt(open_)
        if TRIGGER_KIND == "reject":
            return high.ge(level.mul(1.0 - near)) & close.lt(level.mul(1.0 - buffer)) & close.lt(open_)
        if TRIGGER_KIND == "breakout":
            return _cross_above(close, level.mul(1.0 + buffer))
        if TRIGGER_KIND == "breakdown":
            return _cross_below(close, level.mul(1.0 - buffer))
        if TRIGGER_KIND == "retest_long":
            return low.le(level.mul(1.0 + near)) & close.gt(level.mul(1.0 + buffer)) & close.gt(open_)
        if TRIGGER_KIND == "retest_short":
            return high.ge(level.mul(1.0 - near)) & close.lt(level.mul(1.0 - buffer)) & close.lt(open_)
        if TRIGGER_KIND == "sweep_low_snapback":
            return low.lt(level.mul(1.0 - buffer)) & close.gt(level) & _num(dataframe, "s3_lower_wick_ratio").ge(0.45)
        if TRIGGER_KIND == "sweep_high_reject":
            return high.gt(level.mul(1.0 + buffer)) & close.lt(level) & _num(dataframe, "s3_upper_wick_ratio").ge(0.45)
        if TRIGGER_KIND == "compression_breakout":
            width_ok = _num(dataframe, f"s3_compression_width_{self._lookback()}").le(4.0)
            return width_ok & _cross_above(close, level.mul(1.0 + buffer))
        if TRIGGER_KIND == "compression_breakdown":
            width_ok = _num(dataframe, f"s3_compression_width_{self._lookback()}").le(4.0)
            return width_ok & _cross_below(close, level.mul(1.0 - buffer))
        return pd.Series(False, index=dataframe.index)

    def _quality_ok(self, dataframe: DataFrame) -> Series:
        mode = str(self.trigger_quality_mode.value)
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = self._entry_level(dataframe)
        if mode == "require_pressure":
            return self._pressure_ok(dataframe, "aligned")
        if mode == "require_retest":
            near = float(self.trigger_buffer_pct.value) * 2.0
            return (low.le(level.mul(1.0 + near)) if SIDE == "long" else high.ge(level.mul(1.0 - near)))
        if mode == "require_expansion":
            return _num(dataframe, "s3_range_expand").ge(1.0)
        if mode == "require_wick":
            return _num(dataframe, "s3_lower_wick_ratio").ge(0.35) if SIDE == "long" else _num(dataframe, "s3_upper_wick_ratio").ge(0.35)
        return close.notna()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        trigger = self._entry_trigger(dataframe)
        context = self._context_ok(dataframe)
        quality = self._quality_ok(dataframe)
        pressure_soft = self._pressure_ok(dataframe, "soft")
        plan = str(self.entry_plan.value)
        if plan == "strict_parallel":
            condition = trigger & context & quality & self._pressure_ok(dataframe, "strict")
        elif plan == "trigger_first_context_soft":
            condition = trigger & quality & (context | pressure_soft)
        elif plan == "retest_only":
            retest_memory = _num(dataframe, "low").le(self._entry_level(dataframe).mul(1.0 + float(self.trigger_buffer_pct.value) * 2.0)) if SIDE == "long" else _num(dataframe, "high").ge(self._entry_level(dataframe).mul(1.0 - float(self.trigger_buffer_pct.value) * 2.0))
            condition = trigger & quality & retest_memory
        elif plan == "acceptance_followthrough":
            condition = trigger & context & _num(dataframe, "s3_range_expand").ge(0.85)
        else:
            condition = trigger & context & quality
        if "compression" in SOURCE_ENTRY_STEM and plan != "trigger_first_context_soft":
            compression = _num(dataframe, f"s3_compression_width_{self._lookback()}", 999.0).le(4.0)
            condition &= compression
        valid = condition.fillna(False) & _num(dataframe, "volume").gt(0.0) & _num(dataframe, "close").notna()
        if SIDE == "short":
            dataframe.loc[valid, "enter_short"] = 1
        else:
            dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _latest(self, pair: str) -> Series | None:
        if not getattr(self, "dp", None):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        return dataframe.iloc[-1]

    def _profit_to_level(self, trade, level: float) -> float | None:
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        if open_rate <= 0.0 or not np.isfinite(level) or level <= 0.0:
            return None
        if bool(getattr(trade, "is_short", False)):
            return (open_rate - level) / open_rate
        return (level - open_rate) / open_rate

    def _target_state(self, pair: str, trade, current_rate: float) -> dict[str, Any]:
        last = self._latest(pair)
        if last is None:
            return {"available": False, "valid": False}
        level_series = self._target_level(pd.DataFrame([last]))
        level = pd.to_numeric(level_series, errors="coerce").iloc[0]
        current_rate = float(current_rate or 0.0)
        target_profit = self._profit_to_level(trade, float(level)) if current_rate > 0.0 else None
        if target_profit is None or target_profit <= 0.0:
            return {"available": False, "valid": False}
        distance = abs(float(level) - current_rate) / current_rate
        mode = str(self.target_distance_mode.value)
        valid_distance = True
        if mode == "skip_if_too_close":
            valid_distance = target_profit >= 0.006
        elif mode == "require_useful":
            valid_distance = 0.008 <= target_profit <= 0.12
        elif mode == "cap_far":
            valid_distance = target_profit <= 0.16
        elif mode == "nearest_only":
            valid_distance = distance <= 0.030 or target_profit <= 0.050
        touched = current_rate >= float(level) * 0.997 if SIDE == "long" else current_rate <= float(level) * 1.003
        close = float(pd.to_numeric(pd.Series([last.get("close", np.nan)]), errors="coerce").iloc[0])
        pressure = float(pd.to_numeric(pd.Series([last.get(f"s3_pressure_ratio_{self._lookback()}", 0.0)]), errors="coerce").iloc[0])
        rejected = bool(touched and ((SIDE == "long" and close < float(level) and pressure < -0.02) or (SIDE == "short" and close > float(level) and pressure > 0.02)))
        accepted = bool((SIDE == "long" and close > float(level) * 1.004) or (SIDE == "short" and close < float(level) * 0.996))
        return {
            "available": True,
            "valid": bool(valid_distance),
            "level": float(level),
            "profit": float(target_profit),
            "distance": float(distance),
            "touched": bool(touched),
            "rejected": rejected,
            "accepted": accepted,
            "provider": TARGET_KIND,
        }

    def _guard_state(self, pair: str, trade, current_rate: float, current_profit: float) -> str:
        last = self._latest(pair)
        if last is None:
            return "neutral"
        pressure = float(pd.to_numeric(pd.Series([last.get(f"s3_pressure_ratio_{self._lookback()}", 0.0)]), errors="coerce").iloc[0])
        close = float(pd.to_numeric(pd.Series([last.get("close", np.nan)]), errors="coerce").iloc[0])
        level = pd.to_numeric(self._entry_level(pd.DataFrame([last])), errors="coerce").iloc[0]
        adverse_pressure = pressure < -0.08 if SIDE == "long" else pressure > 0.08
        lost_level = close < float(level) * 0.996 if SIDE == "long" else close > float(level) * 1.004
        target = self._target_state(pair, trade, current_rate)
        adverse_target = bool(target.get("rejected", False))
        score = int(bool(adverse_pressure)) + int(bool(lost_level)) + int(bool(adverse_target))
        if score >= 2:
            return "opposite_multi"
        if score == 1:
            return "weakening" if float(current_profit or 0.0) > 0.0 else "opposite_one"
        aligned_pressure = pressure > 0.05 if SIDE == "long" else pressure < -0.05
        return "aligned" if aligned_pressure else "neutral"

    def _trigger_state(self, pair: str) -> str:
        last = self._latest(pair)
        if last is None:
            return "unknown"
        close = float(pd.to_numeric(pd.Series([last.get("close", np.nan)]), errors="coerce").iloc[0])
        low = float(pd.to_numeric(pd.Series([last.get("low", np.nan)]), errors="coerce").iloc[0])
        high = float(pd.to_numeric(pd.Series([last.get("high", np.nan)]), errors="coerce").iloc[0])
        level = pd.to_numeric(self._entry_level(pd.DataFrame([last])), errors="coerce").iloc[0]
        buffer = float(self.trigger_buffer_pct.value)
        if not np.isfinite(level):
            return "unknown"
        if SIDE == "long":
            if close < float(level) * (1.0 - buffer * 2.0):
                return "invalidated"
            if low < float(level) * (1.0 - buffer) and close < float(level):
                return "retest_failed"
            if close < float(level):
                return "damaged"
        else:
            if close > float(level) * (1.0 + buffer * 2.0):
                return "invalidated"
            if high > float(level) * (1.0 + buffer) and close > float(level):
                return "retest_failed"
            if close > float(level):
                return "damaged"
        return "intact"

    def _trade_age_candles(self, trade, current_time: datetime | None) -> int:
        if current_time is None:
            return 0
        opened = getattr(trade, "open_date_utc", None)
        if opened is None:
            return 0
        minutes = 60 if self.timeframe.endswith("h") else 240
        try:
            minutes = int(self.timeframe[:-1]) * 60 if self.timeframe.endswith("h") else int(self.timeframe[:-1])
        except ValueError:
            minutes = 60
        return int(max(0.0, (current_time - opened).total_seconds()) // max(60, minutes * 60))

    def _trade_state(self, pair: str, trade, current_time: datetime | None, current_rate: float, current_profit: float) -> dict[str, Any]:
        target = self._target_state(pair, trade, current_rate)
        guard = self._guard_state(pair, trade, current_rate, current_profit)
        trigger = self._trigger_state(pair)
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        age = self._trade_age_candles(trade, current_time)
        return {
            "target": target,
            "guard_state": guard,
            "trigger_state": trigger,
            "profit_bucket": _profit_bucket(float(current_profit or 0.0)),
            "partial_state": "no_partial" if exits_done <= 0 else ("first_partial_done" if exits_done == 1 else "runner_only"),
            "age_candles": age,
        }

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = after_fill, kwargs
        state = self._trade_state(pair, trade, current_time, current_rate, current_profit)
        stop_profit = -0.04 if str(self.stop_movement_mode.value) == "fixed_4pct" else -0.03
        target = state["target"]
        guard = state["guard_state"]
        trigger = state["trigger_state"]
        stop_mode = str(self.stop_movement_mode.value)
        response = str(self.trigger_guard_response_mode.value)
        overlay = str(self.pnl_overlay_mode.value)
        trail = str(self.trail_plan.value)
        profit = float(current_profit or 0.0)
        if target.get("touched") and stop_mode in {"target_touch_be", "structure_then_be"}:
            stop_profit = max(stop_profit, 0.001)
        if state["partial_state"] != "no_partial" and overlay in {"profit_lock", "profit_scale"}:
            stop_profit = max(stop_profit, 0.004)
        if guard in {"weakening", "opposite_one", "opposite_multi"} and stop_mode == "guard_tighten" and float(current_profit or 0.0) > 0.004:
            stop_profit = max(stop_profit, float(current_profit) - 0.012)
        if trigger in {"damaged", "retest_failed", "invalidated"} and response == "trigger_fail_lock_if_profit" and float(current_profit or 0.0) > 0.006:
            stop_profit = max(stop_profit, 0.002)
        if target.get("touched") and trail in {"after_target_1pct", "after_target_1_5pct"}:
            distance = 0.010 if trail == "after_target_1pct" else 0.015
            stop_profit = max(stop_profit, float(current_profit or 0.0) - distance)
        if target.get("touched") and stop_mode == "trail_after_target":
            stop_profit = max(stop_profit, float(current_profit or 0.0) - 0.012)
        if state["partial_state"] != "no_partial" and trail == "after_partial_1pct":
            stop_profit = max(stop_profit, float(current_profit or 0.0) - 0.010)
        if (target.get("touched") or state["partial_state"] != "no_partial") and trail == "structure_lookback":
            stop_profit = max(stop_profit, float(current_profit or 0.0) - 0.018)
        if str(self.time_exit_mode.value) == "stale_tighten_only" and state["age_candles"] >= STRUCTURE_MEMORY:
            stop_profit = max(stop_profit, min(float(current_profit or 0.0) - 0.020, 0.002))
        if overlay == "loss_protect" and trigger == "invalidated":
            stop_profit = max(stop_profit, -0.015)
        if SOURCE_ENTRY_STEM.find("crash_flush") >= 0:
            if state["age_candles"] >= max(8, STRUCTURE_MEMORY // 2) and profit < 0.008:
                stop_profit = max(stop_profit, -0.012)
        if SOURCE_ENTRY_STEM.find("exhaustion_sweep") >= 0 and trigger in {"damaged", "retest_failed"} and float(current_profit or 0.0) > 0.004:
            stop_profit = max(stop_profit, 0.001)
        open_rate = float(getattr(trade, "open_rate", 0.0) or current_rate or 0.0)
        current_rate = float(current_rate or 0.0)
        if open_rate <= 0.0 or current_rate <= 0.0:
            return stop_profit
        stop_price = open_rate * (1.0 + stop_profit) if not bool(getattr(trade, "is_short", False)) else open_rate * (1.0 - stop_profit)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=bool(getattr(trade, "is_short", False)),
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )

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
        _ = min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        if exits_done > 0:
            return None
        plan = str(self.partial_plan.value)
        action = str(self.target_action_mode.value)
        if plan == "none" and "partial" not in action:
            return None
        state = self._trade_state(getattr(trade, "pair", ""), trade, current_time, current_rate, current_profit)
        target = state["target"]
        guard = state["guard_state"]
        should_partial = False
        fraction = 0.33
        if plan == "p33_at_target" and target.get("touched") and target.get("valid"):
            should_partial = True
            fraction = 0.33
        elif plan == "p50_at_target" and target.get("touched") and target.get("valid"):
            should_partial = True
            fraction = 0.50
        elif plan == "p33_at_2pct" and float(current_profit or 0.0) >= 0.020:
            should_partial = True
            fraction = 0.33
        elif plan == "p25_at_1pct_then_runner" and float(current_profit or 0.0) >= 0.010:
            should_partial = True
            fraction = 0.25
        elif action in {"partial_33_then_be", "partial_50_then_trail"} and target.get("touched") and target.get("valid"):
            should_partial = True
            fraction = 0.50 if action == "partial_50_then_trail" else 0.33
        if str(self.trigger_guard_response_mode.value) == "guard_flip_reduce_if_profit" and guard in {"weakening", "opposite_one", "opposite_multi"} and float(current_profit or 0.0) >= 0.010:
            should_partial = True
            fraction = max(fraction, 0.33)
        if not should_partial:
            return None
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        return -(stake * fraction), f"{ENTRY_TAG}_partial_{int(fraction * 100)}"

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = kwargs
        state = self._trade_state(pair, trade, current_time, current_rate, current_profit)
        target = state["target"]
        guard = state["guard_state"]
        trigger = state["trigger_state"]
        action = str(self.target_action_mode.value)
        response = str(self.trigger_guard_response_mode.value)
        overlay = str(self.pnl_overlay_mode.value)
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        profit = float(current_profit or 0.0)
        if target.get("valid") and target.get("touched"):
            if action == "full_on_touch":
                return f"{ENTRY_TAG}_target_touch_full"
            if action == "full_on_reject" and target.get("rejected"):
                return f"{ENTRY_TAG}_target_reject_full"
            if action == "runner_until_guard_flip" and exits_done >= 1 and guard in {"weakening", "opposite_one", "opposite_multi"}:
                return f"{ENTRY_TAG}_runner_guard_flip"
        if response == "trigger_fail_full_if_guard_opposes" and trigger in {"retest_failed", "invalidated"} and guard in {"opposite_one", "opposite_multi"}:
            return f"{ENTRY_TAG}_trigger_guard_full"
        if overlay == "asymmetric_loss_hold_profit_reduce" and profit >= 0.025 and guard == "opposite_multi":
            return f"{ENTRY_TAG}_profit_opposition_exit"
        if SOURCE_ENTRY_STEM.find("crash_flush") >= 0 and state["age_candles"] >= max(8, STRUCTURE_MEMORY // 2) and -0.002 <= profit <= 0.008:
            return f"{ENTRY_TAG}_failed_bounce_time_exit"
        if SOURCE_ENTRY_STEM.find("exhaustion_sweep") >= 0 and target.get("accepted", False) and profit >= 0.012:
            return f"{ENTRY_TAG}_accepted_sweep_followthrough_exit"
        if str(self.time_exit_mode.value) == "stale_flat_exit" and state["age_candles"] >= STRUCTURE_MEMORY and -0.003 <= profit <= 0.006:
            return f"{ENTRY_TAG}_stale_flat_exit"
        if str(self.time_exit_mode.value) == "stale_lock_profit" and state["age_candles"] >= STRUCTURE_MEMORY and profit >= 0.012 and guard in {"weakening", "opposite_one", "opposite_multi"}:
            return f"{ENTRY_TAG}_stale_profit_exit"
        if profit >= 0.045 and action == "tighten_only" and guard == "opposite_multi":
            return f"{ENTRY_TAG}_large_profit_guard_exit"
        if profit <= -0.04:
            return f"{ENTRY_TAG}_fixed_risk_exit"
        return None


apply_explicit_hyperopt_surface(CodexSieve3VPPOCRejectTargetShortTf1H)
