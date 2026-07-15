"""Advanced Codex Sieve3 novel strategy.

Testing focus:
- Short entries managed by speed of favorable breakdown, slow drift, and fast adverse reclaim.
- Trader thesis: A short trade with fast downside progress should be managed differently from slow grind or fast adverse reclaim.
- Hyperopt question: Can short-side MFE/MAE path state identify when to hold continuation versus cut failed breakdowns?
- Active parameters: setup_mode, trigger_mode, context_mode, lookback_mode, quality_mode, lifecycle_policy, target_route, stop_policy, partial_plan, add_reduce_policy.
- Scope: standalone top-level Sieve3 strategy using OHLCV and rolling volume-profile evidence only; parked orderbook, FreqAI, and news/context sources are not used.
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
NOVEL_IDEA = True
SOURCE_STRATEGY = "codex_advanced_novel_theory"
SOURCE_RESULT_BATCH = "codex_sieve3_advanced_spin1_20260621"
RESEARCH_PATH = "codex_sieve3_advanced_feature_discovery"
ENTRY_SOURCE_STAGE = "codex_novel"
EXIT_HYPOTHESIS = "A short trade with fast downside progress should be managed differently from slow grind or fast adverse reclaim."
ENTRY_TAG = "codex_sieve3_novel_velocity_path_management_short_1h"
SOURCE_ENTRY_STEM = "codex_sieve3_novel_velocity_path_management_short_1h"
SOURCE_ENTRY_CLASS = "CodexSieve3NovelVelocityPathManagementShortTf1H"
TESTING_FOCUS = "Short entries managed by speed of favorable breakdown, slow drift, and fast adverse reclaim."
PRIMARY_TRIGGER = "velocity-supported rejection or breakdown"
PRIMARY_GUARD = "negative pressure and path cleanliness"
TARGET_PROVIDER = "first useful downside obstacle or projection target"
INVALIDATION_PROVIDER = "fast adverse reclaim or lost breakdown level"
BRANCH_SPLIT_RATIONALE = "One advanced file tests one coherent path-management feature with compact categorical plans instead of tiny backtest-like toggles."

SIDE = "short"
FAMILY = "velocity_path_management_short"
TIMEFRAME = "1h"
PRIMARY_LEVEL_KIND = "range_mid"
DEFAULT_TARGET_KIND = "prior_low"
STRUCTURE_MEMORY = 36

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


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


class CodexSieve3NovelVelocityPathManagementShortTf1H(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = TIMEFRAME
    startup_candle_count = 420
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 1

    setup_mode = CategoricalParameter(['fast_aligned_move', 'slow_reclaim', 'adverse_reclaim', 'velocity_breakdown', 'adaptive_path'], default="fast_aligned_move", space="buy", optimize=True, load=True)
    trigger_mode = CategoricalParameter(['close_break', 'retest_reject', 'pressure_acceptance', 'two_close_acceptance', 'loss_level'], default="close_break", space="buy", optimize=True, load=True)
    context_mode = CategoricalParameter(
        ["ignore", "range_location", "trend_structure", "value_context", "pressure_regime"],
        default="value_context",
        space="buy",
        optimize=True,
        load=True,
    )
    lookback_mode = CategoricalParameter([48, 96, 168, 336], default=96, space="buy", optimize=True, load=True)
    quality_mode = CategoricalParameter(['normal', 'pressure_aligned', 'velocity_confirmed', 'volume_confirmed', 'strict_acceptance'], default="normal", space="buy", optimize=True, load=True)

    lifecycle_policy = CategoricalParameter(['clean_path_runner', 'fast_adverse_exit', 'target_acceleration_partial', 'stalled_path_exit', 'profit_reduce_on_opposition', 'continuation_after_target'], default="clean_path_runner", space="sell", optimize=True, load=True)
    target_route = CategoricalParameter(['nearest_useful_obstacle', 'range_mid', 'prior_low', 'range_projection_1x', 'range_projection_1_5x', 'vp_poc', 'vp_val', 'confluence_target'], default="nearest_useful_obstacle", space="sell", optimize=True, load=True)
    stop_policy = CategoricalParameter(['static_3pct', 'static_4pct', 'entry_level_damage', 'breakeven_after_partial', 'tighten_on_guard_flip', 'trail_after_target'], default="entry_level_damage", space="sell", optimize=True, load=True)
    partial_plan = CategoricalParameter(
        ["none", "p25_at_1_5pct", "p33_at_first_target", "p50_at_first_target", "p33_then_runner", "p50_fast_move_then_trail"],
        default="p33_at_first_target",
        space="sell",
        optimize=True,
        load=True,
    )
    add_reduce_policy = CategoricalParameter(['off', 'add_on_clean_retest', 'add_after_partial', 'reduce_on_guard_flip', 'tighten_only_no_add', 'add_or_reduce_by_guard'], default="off", space="sell", optimize=True, load=True)

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
        frame = add_volume_profile(
            dataframe.copy(),
            window=int(self.lookback_mode.value),
            bins=40,
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
            minp = max(4, lookback // 4)
            volume_sum = prior_volume.rolling(lookback, min_periods=minp).sum().replace(0.0, np.nan)
            frame[f"s3_prior_high_{lookback}"] = high.shift(1).rolling(lookback, min_periods=minp).max()
            frame[f"s3_prior_low_{lookback}"] = low.shift(1).rolling(lookback, min_periods=minp).min()
            frame[f"s3_range_mid_{lookback}"] = (frame[f"s3_prior_high_{lookback}"] + frame[f"s3_prior_low_{lookback}"]) / 2.0
            frame[f"s3_range_width_{lookback}"] = (frame[f"s3_prior_high_{lookback}"] - frame[f"s3_prior_low_{lookback}"]).clip(lower=0.0)
            frame[f"s3_projection_up_1x_{lookback}"] = frame[f"s3_prior_high_{lookback}"] + frame[f"s3_range_width_{lookback}"]
            frame[f"s3_projection_down_1x_{lookback}"] = frame[f"s3_prior_low_{lookback}"] - frame[f"s3_range_width_{lookback}"]
            frame[f"s3_projection_up_1_5x_{lookback}"] = frame[f"s3_prior_high_{lookback}"] + frame[f"s3_range_width_{lookback}"].mul(1.5)
            frame[f"s3_projection_down_1_5x_{lookback}"] = frame[f"s3_prior_low_{lookback}"] - frame[f"s3_range_width_{lookback}"].mul(1.5)
            frame[f"s3_vwap_{lookback}"] = (close.shift(1) * prior_volume).rolling(lookback, min_periods=minp).sum() / volume_sum
            frame[f"s3_pressure_ratio_{lookback}"] = (
                frame["s3_directional_volume"].rolling(lookback, min_periods=minp).sum()
                / volume.rolling(lookback, min_periods=minp).sum().replace(0.0, np.nan)
            )
            frame[f"s3_volume_ratio_{lookback}"] = volume / volume.shift(1).rolling(lookback, min_periods=minp).mean().replace(0.0, np.nan)
            frame[f"s3_velocity_{lookback}"] = (close - close.shift(max(3, lookback // 12))).abs() / frame["s3_atr"].replace(0.0, np.nan)
            leg = (frame[f"s3_prior_high_{lookback}"] - frame[f"s3_prior_low_{lookback}"]).replace(0.0, np.nan)
            frame[f"s3_pullback_depth_long_{lookback}"] = ((frame[f"s3_prior_high_{lookback}"] - close) / leg).clip(0.0, 1.5)
            frame[f"s3_pullback_depth_short_{lookback}"] = ((close - frame[f"s3_prior_low_{lookback}"]) / leg).clip(0.0, 1.5)

        strength = 3
        rolling_high = high.rolling(strength * 2 + 1, min_periods=strength + 1).max()
        rolling_low = low.rolling(strength * 2 + 1, min_periods=strength + 1).min()
        frame["s3_confirmed_pivot_high"] = high.shift(strength).where(high.shift(strength).eq(rolling_high)).ffill()
        frame["s3_confirmed_pivot_low"] = low.shift(strength).where(low.shift(strength).eq(rolling_low)).ffill()
        return frame

    def _lookback(self) -> int:
        return int(self.lookback_mode.value)

    def _level(self, dataframe: DataFrame, kind: str) -> Series:
        lookback = self._lookback()
        if kind == "prior_high":
            return _num(dataframe, f"s3_prior_high_{lookback}", np.nan)
        if kind == "prior_low":
            return _num(dataframe, f"s3_prior_low_{lookback}", np.nan)
        if kind == "range_mid":
            return _num(dataframe, f"s3_range_mid_{lookback}", np.nan)
        if kind == "range_high":
            return _num(dataframe, f"s3_prior_high_{lookback}", np.nan)
        if kind == "range_low":
            return _num(dataframe, f"s3_prior_low_{lookback}", np.nan)
        if kind == "projection_up_1x":
            return _num(dataframe, f"s3_projection_up_1x_{lookback}", np.nan)
        if kind == "projection_down_1x":
            return _num(dataframe, f"s3_projection_down_1x_{lookback}", np.nan)
        if kind == "projection_up_1_5x":
            return _num(dataframe, f"s3_projection_up_1_5x_{lookback}", np.nan)
        if kind == "projection_down_1_5x":
            return _num(dataframe, f"s3_projection_down_1_5x_{lookback}", np.nan)
        if kind == "vp_poc":
            return _num(dataframe, "vp_prior_poc", _num(dataframe, f"s3_vwap_{lookback}", np.nan))
        if kind == "vp_vah":
            return _num(dataframe, "vp_prior_vah", _num(dataframe, f"s3_prior_high_{lookback}", np.nan))
        if kind == "vp_val":
            return _num(dataframe, "vp_prior_val", _num(dataframe, f"s3_prior_low_{lookback}", np.nan))
        if kind == "confirmed_pivot_high":
            return _num(dataframe, "s3_confirmed_pivot_high", np.nan)
        if kind == "confirmed_pivot_low":
            return _num(dataframe, "s3_confirmed_pivot_low", np.nan)
        return _num(dataframe, f"s3_range_mid_{lookback}", np.nan)

    def _pressure_ok(self, dataframe: DataFrame, strength: str = "aligned") -> Series:
        lookback = self._lookback()
        pressure = _num(dataframe, f"s3_pressure_ratio_{lookback}", 0.0)
        volume_ratio = _num(dataframe, f"s3_volume_ratio_{lookback}", 0.0)
        pressure_min = {"soft": 0.025, "aligned": 0.060, "strict": 0.120}.get(strength, 0.060)
        volume_min = {"soft": 0.75, "aligned": 0.95, "strict": 1.20}.get(strength, 0.95)
        if SIDE == "short":
            return pressure.le(-pressure_min) & volume_ratio.ge(volume_min)
        return pressure.ge(pressure_min) & volume_ratio.ge(volume_min)

    def _context_ok(self, dataframe: DataFrame) -> Series:
        mode = str(self.context_mode.value)
        if mode == "ignore":
            return pd.Series(True, index=dataframe.index)
        close = _num(dataframe, "close")
        lookback = self._lookback()
        mid = self._level(dataframe, "range_mid")
        high = self._level(dataframe, "prior_high")
        low = self._level(dataframe, "prior_low")
        poc = self._level(dataframe, "vp_poc")
        pressure = self._pressure_ok(dataframe, "soft")
        if mode == "range_location":
            return (close.ge(mid) if SIDE == "long" else close.le(mid)) & high.notna() & low.notna()
        if mode == "trend_structure":
            prior_high = _num(dataframe, f"s3_prior_high_{lookback}", np.nan)
            prior_low = _num(dataframe, f"s3_prior_low_{lookback}", np.nan)
            if SIDE == "short":
                return prior_high.lt(prior_high.shift(8)) & prior_low.lt(prior_low.shift(8))
            return prior_high.gt(prior_high.shift(8)) & prior_low.gt(prior_low.shift(8))
        if mode == "value_context":
            return close.ge(poc) if SIDE == "long" else close.le(poc)
        return pressure

    def _generic_trigger(self, dataframe: DataFrame) -> Series:
        mode = str(self.trigger_mode.value)
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = self._level(dataframe, PRIMARY_LEVEL_KIND)
        buffer = 0.0035
        near = 0.008
        if SIDE == "long":
            if "two_close" in mode:
                return close.gt(level.mul(1.0 + buffer)) & close.shift(1).gt(level.shift(1))
            if "retest" in mode or "hold" in mode:
                return low.le(level.mul(1.0 + near)) & close.gt(level.mul(1.0 + buffer)) & close.gt(open_)
            if "reclaim" in mode:
                return low.lt(level.mul(1.0 - buffer)) & close.gt(level.mul(1.0 + buffer))
            if "pressure" in mode:
                return close.gt(level) & self._pressure_ok(dataframe, "aligned")
            return _cross_above(close, level.mul(1.0 + buffer))
        if "two_close" in mode:
            return close.lt(level.mul(1.0 - buffer)) & close.shift(1).lt(level.shift(1))
        if "retest" in mode or "reject" in mode:
            return high.ge(level.mul(1.0 - near)) & close.lt(level.mul(1.0 - buffer)) & close.lt(open_)
        if "loss" in mode or "lost" in mode or "reclaim" in mode:
            return high.gt(level.mul(1.0 + buffer)) & close.lt(level.mul(1.0 - buffer))
        if "pressure" in mode:
            return close.lt(level) & self._pressure_ok(dataframe, "aligned")
        return _cross_below(close, level.mul(1.0 - buffer))

    def _family_setup_ok(self, dataframe: DataFrame) -> Series:
        setup = str(self.setup_mode.value)
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        lookback = self._lookback()
        prior_high = self._level(dataframe, "prior_high")
        prior_low = self._level(dataframe, "prior_low")
        mid = self._level(dataframe, "range_mid")
        width = _num(dataframe, f"s3_range_width_{lookback}", np.nan)
        velocity = _num(dataframe, f"s3_velocity_{lookback}", 0.0)
        pressure = _num(dataframe, f"s3_pressure_ratio_{lookback}", 0.0)
        volume_ratio = _num(dataframe, f"s3_volume_ratio_{lookback}", 0.0)
        range_expand = _num(dataframe, "s3_range_expand", 0.0)
        pullback_long = _num(dataframe, f"s3_pullback_depth_long_{lookback}", 1.0)
        pullback_short = _num(dataframe, f"s3_pullback_depth_short_{lookback}", 1.0)

        if FAMILY == "range_expansion_exhaustion_short":
            projection = prior_high + width.mul(0.55)
            final_drive = high.gt(projection) & close.lt(prior_high)
            weakening = pressure.lt(pressure.shift(6)) | _num(dataframe, "s3_upper_wick_ratio").ge(0.35)
            if "strict" in setup:
                return final_drive & weakening & volume_ratio.ge(1.0)
            if "projection" in setup:
                return final_drive
            if "wick" in setup:
                return high.gt(projection) & _num(dataframe, "s3_upper_wick_ratio").ge(0.40)
            return final_drive | (high.gt(prior_high) & close.lt(mid))

        if "second_chance" in FAMILY and SIDE == "long":
            first_fail = low.rolling(8, min_periods=3).min().lt(prior_low.mul(0.994))
            reclaim = close.gt(prior_low) & close.shift(1).lt(prior_low.shift(1).mul(1.006))
            if "two_reclaims" in setup:
                return first_fail & reclaim & close.shift(1).gt(prior_low.shift(1))
            if "pressure" in setup:
                return first_fail & reclaim & pressure.gt(0.03)
            return first_fail & reclaim

        if "second_chance" in FAMILY and SIDE == "short":
            first_fail = high.rolling(8, min_periods=3).max().gt(prior_high.mul(1.006))
            reject = close.lt(prior_high) & close.shift(1).gt(prior_high.shift(1).mul(0.994))
            if "two_rejects" in setup:
                return first_fail & reject & close.shift(1).lt(prior_high.shift(1))
            if "pressure" in setup:
                return first_fail & reject & pressure.lt(-0.03)
            return first_fail & reject

        if "pullback_depth" in FAMILY and SIDE == "long":
            shallow = pullback_long.between(0.18, 0.38)
            medium = pullback_long.between(0.38, 0.62)
            deep = pullback_long.between(0.62, 0.82) & self._pressure_ok(dataframe, "soft")
            if "shallow" in setup:
                return shallow & close.gt(mid)
            if "medium" in setup:
                return medium & close.gt(mid)
            if "deep" in setup:
                return deep
            return (shallow | medium | deep) & close.gt(prior_low)

        if "velocity" in FAMILY:
            fast_aligned = velocity.ge(1.15) & self._pressure_ok(dataframe, "soft")
            slow_reclaim = velocity.between(0.35, 1.10) & (close.gt(mid) if SIDE == "long" else close.lt(mid))
            adverse_reclaim = (low.lt(mid) & close.gt(mid)) if SIDE == "long" else (high.gt(mid) & close.lt(mid))
            if "fast" in setup:
                return fast_aligned
            if "adverse" in setup:
                return adverse_reclaim
            if "slow" in setup:
                return slow_reclaim
            return fast_aligned | slow_reclaim | adverse_reclaim

        if "runner_continuation" in FAMILY:
            acceptance = close.gt(prior_high) if SIDE == "long" else close.lt(prior_low)
            retest = (low.le(prior_high.mul(1.008)) & close.gt(prior_high)) if SIDE == "long" else (high.ge(prior_low.mul(0.992)) & close.lt(prior_low))
            if "retest" in setup:
                return retest
            if "two_close" in setup:
                return acceptance & acceptance.shift(1).fillna(False)
            return acceptance | retest

        if "multi_obstacle" in FAMILY:
            value_ok = close.gt(self._level(dataframe, "vp_poc")) if SIDE == "long" else close.lt(self._level(dataframe, "vp_poc"))
            structure_ok = close.gt(mid) if SIDE == "long" else close.lt(mid)
            if "confluence" in setup:
                return value_ok & structure_ok & self._pressure_ok(dataframe, "soft")
            if "value" in setup:
                return value_ok
            return structure_ok

        if "regime_adaptive" in FAMILY:
            range_width = width / close.replace(0.0, np.nan)
            tight = range_width.le(0.055)
            expanding = range_expand.ge(0.95)
            if "range_rotation" in setup:
                return tight & (close.gt(mid) if SIDE == "long" else close.lt(mid))
            if "expansion" in setup:
                return expanding & ((close.gt(prior_high)) if SIDE == "long" else (close.lt(prior_low)))
            if "value" in setup:
                return close.gt(self._level(dataframe, "vp_poc")) if SIDE == "long" else close.lt(self._level(dataframe, "vp_poc"))
            return tight | expanding

        if "add_reduce" in FAMILY:
            return (close.gt(mid) if SIDE == "long" else close.lt(mid)) & self._pressure_ok(dataframe, "soft")

        return self._generic_trigger(dataframe)

    def _quality_ok(self, dataframe: DataFrame) -> Series:
        mode = str(self.quality_mode.value)
        close = _num(dataframe, "close")
        level = self._level(dataframe, PRIMARY_LEVEL_KIND)
        if "strict" in mode:
            return self._pressure_ok(dataframe, "strict") & _num(dataframe, "s3_range_expand", 0.0).ge(0.80)
        if "pressure" in mode:
            return self._pressure_ok(dataframe, "aligned")
        if "volume" in mode:
            return _num(dataframe, f"s3_volume_ratio_{self._lookback()}", 0.0).ge(1.10)
        if "wick" in mode:
            return _num(dataframe, "s3_lower_wick_ratio").ge(0.30) if SIDE == "long" else _num(dataframe, "s3_upper_wick_ratio").ge(0.30)
        if "acceptance" in mode or "two_close" in mode:
            return (close.gt(level) & close.shift(1).gt(level.shift(1))) if SIDE == "long" else (close.lt(level) & close.shift(1).lt(level.shift(1)))
        if "velocity" in mode:
            return _num(dataframe, f"s3_velocity_{self._lookback()}", 0.0).ge(0.75)
        return close.notna()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = (
            self._family_setup_ok(dataframe)
            & self._generic_trigger(dataframe)
            & self._context_ok(dataframe)
            & self._quality_ok(dataframe)
            & _num(dataframe, "volume").gt(0.0)
        ).fillna(False)
        if SIDE == "short":
            dataframe.loc[condition, "enter_short"] = 1
        else:
            dataframe.loc[condition, "enter_long"] = 1
        dataframe.loc[condition, "enter_tag"] = ENTRY_TAG
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

    def _target_candidates(self, dataframe: DataFrame) -> DataFrame:
        if SIDE == "short":
            return pd.DataFrame(
                {
                    "prior_low": self._level(dataframe, "prior_low"),
                    "range_mid": self._level(dataframe, "range_mid"),
                    "projection_1x": self._level(dataframe, "projection_down_1x"),
                    "projection_1_5x": self._level(dataframe, "projection_down_1_5x"),
                    "vp_poc": self._level(dataframe, "vp_poc"),
                    "vp_val": self._level(dataframe, "vp_val"),
                },
                index=dataframe.index,
            )
        return pd.DataFrame(
            {
                "prior_high": self._level(dataframe, "prior_high"),
                "range_mid": self._level(dataframe, "range_mid"),
                "projection_1x": self._level(dataframe, "projection_up_1x"),
                "projection_1_5x": self._level(dataframe, "projection_up_1_5x"),
                "vp_poc": self._level(dataframe, "vp_poc"),
                "vp_vah": self._level(dataframe, "vp_vah"),
            },
            index=dataframe.index,
        )

    def _target_level(self, dataframe: DataFrame) -> Series:
        route = str(self.target_route.value)
        close = _num(dataframe, "close")
        if route == "range_projection_1x":
            return self._level(dataframe, "projection_down_1x" if SIDE == "short" else "projection_up_1x")
        if route == "range_projection_1_5x":
            return self._level(dataframe, "projection_down_1_5x" if SIDE == "short" else "projection_up_1_5x")
        if route == "range_mid":
            return self._level(dataframe, "range_mid")
        if route == "prior_high":
            return self._level(dataframe, "prior_high")
        if route == "prior_low":
            return self._level(dataframe, "prior_low")
        if route == "vp_poc":
            return self._level(dataframe, "vp_poc")
        if route == "vp_vah":
            return self._level(dataframe, "vp_vah")
        if route == "vp_val":
            return self._level(dataframe, "vp_val")
        candidates = self._target_candidates(dataframe)
        if SIDE == "short":
            useful = candidates.where(candidates.lt(close.mul(0.996)))
            nearest = useful.max(axis=1)
        else:
            useful = candidates.where(candidates.gt(close.mul(1.004)))
            nearest = useful.min(axis=1)
        if route == "confluence_target":
            default = self._level(dataframe, DEFAULT_TARGET_KIND)
            return ((nearest + default) / 2.0).where(nearest.notna() & default.notna(), nearest.fillna(default))
        return nearest.fillna(self._level(dataframe, DEFAULT_TARGET_KIND))

    def _profit_to_level(self, trade, level: float) -> float | None:
        open_rate = _as_float(getattr(trade, "open_rate", 0.0), 0.0)
        if open_rate <= 0.0 or not np.isfinite(level) or level <= 0.0:
            return None
        if bool(getattr(trade, "is_short", False)):
            return (open_rate - level) / open_rate
        return (level - open_rate) / open_rate

    def _target_state(self, pair: str, trade, current_rate: float) -> dict[str, Any]:
        last = self._latest(pair)
        if last is None:
            return {"available": False, "valid": False, "touched": False}
        level = pd.to_numeric(self._target_level(pd.DataFrame([last])), errors="coerce").iloc[0]
        target_profit = self._profit_to_level(trade, _as_float(level))
        current_rate = _as_float(current_rate, 0.0)
        if target_profit is None or target_profit <= 0.0 or current_rate <= 0.0:
            return {"available": False, "valid": False, "touched": False}
        touched = current_rate <= _as_float(level) * 1.003 if SIDE == "short" else current_rate >= _as_float(level) * 0.997
        close = _as_float(last.get("close", np.nan))
        pressure = _as_float(last.get(f"s3_pressure_ratio_{self._lookback()}", 0.0), 0.0)
        rejected = bool(touched and ((SIDE == "long" and close < _as_float(level) and pressure < -0.02) or (SIDE == "short" and close > _as_float(level) and pressure > 0.02)))
        return {
            "available": True,
            "valid": 0.006 <= float(target_profit) <= 0.18,
            "level": _as_float(level),
            "profit": float(target_profit),
            "touched": bool(touched),
            "rejected": rejected,
        }

    def _trigger_state(self, pair: str) -> str:
        last = self._latest(pair)
        if last is None:
            return "unknown"
        close = _as_float(last.get("close", np.nan))
        level = pd.to_numeric(self._level(pd.DataFrame([last]), PRIMARY_LEVEL_KIND), errors="coerce").iloc[0]
        if not np.isfinite(level) or not np.isfinite(close):
            return "unknown"
        if SIDE == "short":
            if close > float(level) * 1.008:
                return "invalidated"
            if close > float(level) * 1.003:
                return "damaged"
        else:
            if close < float(level) * 0.992:
                return "invalidated"
            if close < float(level) * 0.997:
                return "damaged"
        return "intact"

    def _guard_state(self, pair: str, current_profit: float) -> str:
        last = self._latest(pair)
        if last is None:
            return "neutral"
        pressure = _as_float(last.get(f"s3_pressure_ratio_{self._lookback()}", 0.0), 0.0)
        adverse_pressure = pressure > 0.08 if SIDE == "short" else pressure < -0.08
        aligned_pressure = pressure < -0.05 if SIDE == "short" else pressure > 0.05
        if adverse_pressure and float(current_profit or 0.0) >= 0.0:
            return "opposes_profit"
        if adverse_pressure:
            return "opposes_loss"
        return "aligned" if aligned_pressure else "neutral"

    def _trade_age_candles(self, trade, current_time: datetime | None) -> int:
        if current_time is None:
            return 0
        opened = getattr(trade, "open_date_utc", None)
        if opened is None:
            return 0
        return int(max(0.0, (current_time - opened).total_seconds()) // 3600)

    def _trade_path(self, trade, current_profit: float) -> dict[str, float]:
        open_rate = _as_float(getattr(trade, "open_rate", 0.0), 0.0)
        if open_rate <= 0.0:
            return {"mfe": float(current_profit or 0.0), "mae": float(current_profit or 0.0)}
        max_rate = _as_float(getattr(trade, "max_rate", np.nan), np.nan)
        min_rate = _as_float(getattr(trade, "min_rate", np.nan), np.nan)
        if bool(getattr(trade, "is_short", False)):
            mfe = (open_rate - min_rate) / open_rate if np.isfinite(min_rate) else float(current_profit or 0.0)
            mae = (open_rate - max_rate) / open_rate if np.isfinite(max_rate) else float(current_profit or 0.0)
        else:
            mfe = (max_rate - open_rate) / open_rate if np.isfinite(max_rate) else float(current_profit or 0.0)
            mae = (min_rate - open_rate) / open_rate if np.isfinite(min_rate) else float(current_profit or 0.0)
        return {"mfe": float(mfe), "mae": float(mae)}

    def _state(self, pair: str, trade, current_time: datetime | None, current_rate: float, current_profit: float) -> dict[str, Any]:
        return {
            "target": self._target_state(pair, trade, current_rate),
            "trigger": self._trigger_state(pair),
            "guard": self._guard_state(pair, current_profit),
            "age": self._trade_age_candles(trade, current_time),
            "exits_done": int(getattr(trade, "nr_of_successful_exits", 0) or 0),
            "entries_done": int(getattr(trade, "nr_of_successful_entries", 1) or 1),
            "path": self._trade_path(trade, current_profit),
            "profit": float(current_profit or 0.0),
        }

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = after_fill, kwargs
        state = self._state(pair, trade, current_time, current_rate, current_profit)
        policy = str(self.stop_policy.value)
        stop_profit = -0.04 if policy == "static_4pct" else -0.03
        profit = float(current_profit or 0.0)
        if "structure" in policy or "entry_level" in policy or "failed_level" in policy:
            if state["trigger"] in {"damaged", "invalidated"}:
                stop_profit = max(stop_profit, -0.015)
        if "breakeven" in policy and (state["exits_done"] > 0 or profit >= 0.015):
            stop_profit = max(stop_profit, 0.001)
        if "tighten" in policy and state["guard"] in {"opposes_profit", "opposes_loss"}:
            stop_profit = max(stop_profit, profit - 0.012 if profit > 0.006 else -0.018)
        if "trail" in policy and (state["target"].get("touched") or state["exits_done"] > 0):
            stop_profit = max(stop_profit, profit - 0.015)
        if "mfe" in policy and state["path"]["mfe"] >= 0.025:
            stop_profit = max(stop_profit, 0.003)
        if "time" in policy and state["age"] >= STRUCTURE_MEMORY and -0.004 <= profit <= 0.010:
            stop_profit = max(stop_profit, -0.012)
        open_rate = _as_float(getattr(trade, "open_rate", 0.0), _as_float(current_rate, 0.0))
        current_rate = _as_float(current_rate, 0.0)
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
        state = self._state(getattr(trade, "pair", ""), trade, current_time, current_rate, current_profit)
        plan = str(self.partial_plan.value)
        target = state["target"]
        profit = float(current_profit or 0.0)
        exits_done = state["exits_done"]
        if exits_done <= 0 and plan != "none":
            should_partial = False
            fraction = 0.33
            if plan == "p25_at_1_5pct" and profit >= 0.015:
                should_partial, fraction = True, 0.25
            elif plan == "p33_at_first_target" and target.get("touched") and target.get("valid"):
                should_partial, fraction = True, 0.33
            elif plan == "p50_at_first_target" and target.get("touched") and target.get("valid"):
                should_partial, fraction = True, 0.50
            elif plan == "p33_then_runner" and (target.get("touched") or profit >= 0.020):
                should_partial, fraction = True, 0.33
            elif plan == "p50_fast_move_then_trail" and state["path"]["mfe"] >= 0.025:
                should_partial, fraction = True, 0.50
            if should_partial:
                stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
                if stake > 0.0:
                    return -(stake * fraction), f"{ENTRY_TAG}_partial_{int(fraction * 100)}"
        add_policy = str(self.add_reduce_policy.value)
        if "add" in add_policy and state["entries_done"] < 2 and exits_done <= 0 and profit >= 0.010 and state["trigger"] == "intact" and state["guard"] in {"aligned", "neutral"}:
            stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
            if stake > 0.0:
                return stake * 0.25, f"{ENTRY_TAG}_add_25"
        if "reduce" in add_policy and exits_done <= 0 and profit >= 0.010 and state["guard"] == "opposes_profit":
            stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
            if stake > 0.0:
                return -(stake * 0.33), f"{ENTRY_TAG}_guard_reduce_33"
        return None

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = kwargs
        state = self._state(pair, trade, current_time, current_rate, current_profit)
        policy = str(self.lifecycle_policy.value)
        profit = float(current_profit or 0.0)
        target = state["target"]
        trigger = state["trigger"]
        guard = state["guard"]
        path = state["path"]
        if target.get("touched") and target.get("valid"):
            if "full" in policy:
                return f"{ENTRY_TAG}_target_full"
            if "continuation" in policy and guard == "opposes_profit":
                return f"{ENTRY_TAG}_target_opposed_exit"
            if "runner" in policy and state["exits_done"] > 0 and guard == "opposes_profit":
                return f"{ENTRY_TAG}_runner_opposed_exit"
        if "fast_adverse" in policy and path["mae"] <= -0.025 and profit < -0.010:
            return f"{ENTRY_TAG}_fast_adverse_exit"
        if "failed" in policy and trigger == "invalidated":
            return f"{ENTRY_TAG}_trigger_failed_exit"
        if "second" in policy and trigger == "damaged" and guard in {"opposes_profit", "opposes_loss"}:
            return f"{ENTRY_TAG}_second_signal_failed_exit"
        if "stall" in policy and state["age"] >= STRUCTURE_MEMORY and -0.004 <= profit <= 0.010:
            return f"{ENTRY_TAG}_stalled_path_exit"
        if "exhaustion" in policy and profit >= 0.018 and guard == "opposes_profit":
            return f"{ENTRY_TAG}_exhaustion_guard_exit"
        if "reduce" in policy and profit >= 0.030 and guard == "opposes_profit":
            return f"{ENTRY_TAG}_profit_reduce_exit"
        if profit <= -0.045:
            return f"{ENTRY_TAG}_hard_risk_exit"
        return None


apply_explicit_hyperopt_surface(CodexSieve3NovelVelocityPathManagementShortTf1H)
