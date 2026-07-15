from __future__ import annotations

from abc import update_abstractmethods
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute
from freqtrade.strategy.parameters import BaseParameter
from user_data.Indicators.complex_volume_profile import add_volume_profile


REWORK_LOOKBACKS = (24, 48, 96, 168, 336)
SOURCE_VP_PREFIXES = {
    "prior_month_breakout_long": ("vp1d", "vp4h", "vp"),
    "tlv2_vp_bull_long": ("vp",),
    "mtfx_tlv2_break_short": ("htfvp",),
}


def rework_exit_parameter(values: Iterable[Any], default: Any) -> CategoricalParameter:
    parameter = CategoricalParameter(list(values), default=default, space="sell", optimize=True, load=True)
    parameter.batch_tags = ("family:exits", "mode:sieve3_exit_rework")
    return parameter


def _numeric(frame: DataFrame, name: str, default: float = np.nan) -> Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _boolean(frame: DataFrame, name: str) -> Series:
    if name not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return frame[name].astype("boolean").fillna(False).astype(bool)


class _LegacyExitSurfaceDisabled:
    """Shadow the inherited universal sell surface with non-parameter values."""

    exit_path_mode = "rework_only"
    fixed_tp_pct = 0.03
    fixed_sl_pct = 0.03
    partial_1_profit = 0.02
    partial_2_profit = 0.05
    partial_1_fraction = 0.33
    partial_2_fraction = 0.33
    breakeven_trigger = 0.02
    breakeven_offset = 0.001
    trailing_activation = 0.03
    trailing_distance = 0.015
    indicator_min_profit = 0.01
    indicator_max_profit = 0.12
    indicator_near_pct = 0.01
    indicator_stop_buffer = 0.015
    structural_min_profit = 0.01
    structural_max_profit = 0.12
    structural_proximity_pct = 0.01
    structural_stop_buffer = 0.015
    guard_tighten_buffer = 0.015
    guard_profit_floor = 0.005
    time_stop_candles = 96
    time_stop_min_profit = 0.0


class ExitReworkCoreMixin(_LegacyExitSurfaceDisabled):
    """Shared mechanics for source-aware, compact Sieve3 exit comparisons."""

    startup_candle_count = 400
    use_exit_signal = False
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    REWORK_SOURCE_PROFILE = "generic"
    REWORK_SOURCE_TARGET_COLUMNS: tuple[str, ...] = ()
    REWORK_VP_WINDOW = 96
    REWORK_VP_BINS = 48

    def _rwx_add_indicators(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        high = _numeric(frame, "high")
        low = _numeric(frame, "low")
        for lookback in REWORK_LOOKBACKS:
            minimum = max(4, lookback // 4)
            frame[f"rwx_prior_high_{lookback}"] = high.shift(1).rolling(lookback, min_periods=minimum).max()
            frame[f"rwx_prior_low_{lookback}"] = low.shift(1).rolling(lookback, min_periods=minimum).min()

        profile_prefix = next(
            (
                prefix
                for prefix in SOURCE_VP_PREFIXES.get(str(self.REWORK_SOURCE_PROFILE), ())
                if all(f"{prefix}_prior_{level}" in frame.columns for level in ("poc", "vah", "val"))
            ),
            "",
        )
        if profile_prefix:
            for level in ("poc", "vah", "val"):
                frame[f"rwxvp_prior_{level}"] = _numeric(frame, f"{profile_prefix}_prior_{level}")
            frame["rwxvp_prior_hvn_above"] = _numeric(frame, f"{profile_prefix}_hvn_above").shift(1)
            frame["rwxvp_prior_hvn_below"] = _numeric(frame, f"{profile_prefix}_hvn_below").shift(1)
        else:
            frame = add_volume_profile(
                frame,
                window=self.REWORK_VP_WINDOW,
                bins=self.REWORK_VP_BINS,
                value_area_pct=0.70,
                price_source="hlc3",
                smooth_bins=3,
                include_diagnostics=False,
                prefix="rwxvp",
            )
            frame["rwxvp_prior_hvn_above"] = _numeric(frame, "rwxvp_hvn_above").shift(1)
            frame["rwxvp_prior_hvn_below"] = _numeric(frame, "rwxvp_hvn_below").shift(1)
        return frame

    @staticmethod
    def _rwx_utc_timestamp(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _rwx_closed_frame(self, frame: DataFrame, current_time: Any, timeframe: str) -> DataFrame:
        if frame is None or frame.empty:
            return DataFrame()
        result = frame.copy()
        if current_time is None or "date" not in result.columns:
            return result
        now = self._rwx_utc_timestamp(current_time)
        if now is None:
            return result
        dates = pd.to_datetime(result["date"], utc=True, errors="coerce")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(timeframe), unit="m")
        return result.loc[close_times.le(now)].copy()

    def _rwx_frame(self, pair: str, current_time: Any) -> DataFrame:
        if not getattr(self, "dp", None):
            return DataFrame()
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return self._rwx_closed_frame(frame, current_time, self.timeframe)

    def _rwx_confirmation_frame(self, pair: str, current_time: Any, timeframe: str) -> DataFrame:
        if not getattr(self, "dp", None):
            return DataFrame()
        if timeframe == self.timeframe:
            return self._rwx_frame(pair, current_time)
        frame = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        return self._rwx_closed_frame(frame, current_time, timeframe)

    @staticmethod
    def _rwx_value(row: Series, name: str) -> float | None:
        if name not in row.index:
            return None
        value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) and np.isfinite(value) else None

    @staticmethod
    def _rwx_unique(values: Iterable[float | None]) -> list[float]:
        result: list[float] = []
        for value in values:
            if value is None or not np.isfinite(value):
                continue
            if not any(np.isclose(value, existing, rtol=0.0, atol=1e-10) for existing in result):
                result.append(float(value))
        return result

    def _rwx_target_level(
        self,
        frame: DataFrame,
        trade: Any,
        current_rate: float,
        source_mode: str,
        lookback: int,
    ) -> tuple[float | None, str]:
        if frame.empty:
            return None, "missing_frame"
        row = frame.iloc[-1]
        is_short = bool(getattr(trade, "is_short", False))
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        if open_rate <= 0.0:
            return None, "missing_open_rate"

        source_values = [(self._rwx_value(row, name), name) for name in self.REWORK_SOURCE_TARGET_COLUMNS]
        swing_name = f"rwx_prior_low_{lookback}" if is_short else f"rwx_prior_high_{lookback}"
        swing_values = [(self._rwx_value(row, swing_name), swing_name)]
        poc_values = [(self._rwx_value(row, "rwxvp_prior_poc"), "rwxvp_prior_poc")]
        edge_names = (
            ("rwxvp_prior_val", "rwxvp_prior_hvn_below")
            if is_short
            else ("rwxvp_prior_vah", "rwxvp_prior_hvn_above")
        )
        edge_values = [(self._rwx_value(row, name), name) for name in edge_names]

        groups = {
            "source_level": source_values,
            "prior_swing": swing_values,
            "vp_poc": poc_values,
            "vp_value_edge": edge_values,
            "nearest_obstacle": source_values + swing_values + poc_values + edge_values,
        }
        candidates = groups.get(source_mode, groups["nearest_obstacle"])
        minimum_move = 0.002
        directional = [
            (float(value), name)
            for value, name in candidates
            if value is not None
            and ((is_short and float(value) < open_rate * (1.0 - minimum_move)) or (not is_short and float(value) > open_rate * (1.0 + minimum_move)))
        ]
        if not directional:
            return None, f"{source_mode}_unavailable"
        target, provider = min(directional, key=lambda item: abs(item[0] - current_rate))
        return target, provider

    def _rwx_target_confirmed(
        self,
        pair: str,
        current_time: Any,
        target: float,
        is_short: bool,
        band: float,
        timeframe: str,
        confirmation: str,
    ) -> bool:
        frame = self._rwx_confirmation_frame(pair, current_time, timeframe)
        if frame.empty or len(frame) < 2:
            return False
        recent = frame.tail(4)
        high = _numeric(recent, "high")
        low = _numeric(recent, "low")
        open_ = _numeric(recent, "open")
        close = _numeric(recent, "close")
        touched = bool(low.min() <= target * (1.0 + band)) if is_short else bool(high.max() >= target * (1.0 - band))
        if not touched:
            return False
        if confirmation == "touch":
            return True
        opposite = close.gt(open_) if is_short else close.lt(open_)
        if confirmation == "one_reversal":
            return bool(opposite.tail(1).all())
        if confirmation == "two_reversal":
            return len(opposite) >= 2 and bool(opposite.tail(2).all())
        if confirmation == "two_of_three":
            return len(opposite) >= 3 and int(opposite.tail(3).sum()) >= 2
        latest = recent.iloc[-1]
        if confirmation == "close_reject":
            if is_short:
                return float(latest["low"]) <= target * (1.0 + band) and float(latest["close"]) > target
            return float(latest["high"]) >= target * (1.0 - band) and float(latest["close"]) < target
        return False

    def _rwx_target_state(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
        *,
        source_mode: str,
        lookback: int,
        band: float,
        timeframe: str,
        confirmation: str,
    ) -> dict[str, Any]:
        frame = self._rwx_frame(pair, current_time)
        target, provider = self._rwx_target_level(frame, trade, float(current_rate), source_mode, lookback)
        confirmed = False
        if target is not None:
            confirmed = self._rwx_target_confirmed(
                pair,
                current_time,
                target,
                bool(getattr(trade, "is_short", False)),
                band,
                timeframe,
                confirmation,
            )
        return {"target": target, "provider": provider, "confirmed": confirmed}

    def _rwx_invalidation_components(self, frame: DataFrame, band: float) -> dict[str, Series]:
        if frame.empty:
            empty = pd.Series(False, index=frame.index, dtype="bool")
            return {"ltf": empty, "htf": empty, "source": empty}
        close = _numeric(frame, "close")
        profile = str(self.REWORK_SOURCE_PROFILE)

        if profile == "bos_bear_short":
            pivot_high = _numeric(frame, "ms_pivot_high")
            break_level = _numeric(frame, "ms_bearish_break_level")
            state = _numeric(frame, "ms_state")
            ltf = state.gt(0) | close.gt(pivot_high)
            htf = state.gt(0)
            source = close.gt(break_level.mul(1.0 + band))
        elif profile == "prior_month_breakout_long":
            level = _numeric(frame, "prior_month_high")
            market = _numeric(frame, "vp1d_market_context")
            if market.isna().all():
                market = _numeric(frame, "vp_market_context")
            ltf = close.lt(level.mul(1.0 - band))
            htf = market.lt(0)
            source = close.lt(level.mul(1.0 - band))
        elif profile == "tlv2_vp_bull_long":
            level = _numeric(frame, "tlv2_resistance_line_rank0")
            bull = _numeric(frame, "vp_context_score_bull")
            bear = _numeric(frame, "vp_context_score_bear")
            market = _numeric(frame, "vp_market_context")
            ltf = close.lt(level.mul(1.0 - band))
            htf = market.lt(0) | bear.gt(bull)
            source = close.lt(level.mul(1.0 - band)) & bear.gt(bull)
        elif profile == "mtfx_tlv2_break_short":
            level = _numeric(frame, "mtfx_htf_level")
            ltf_state = _numeric(frame, "ltfms_state")
            ltf = ltf_state.gt(0) | _boolean(frame, "ltfms_choch_to_bull")
            htf = ~_boolean(frame, "mtfx_htf_context")
            source = close.gt(level.mul(1.0 + band))
        elif profile == "mtf_macd_breakout_long":
            ema_fast_1h = _numeric(frame, "ema_fast_1h")
            ema_fast_1d = _numeric(frame, "ema_fast_1d")
            ema_slow_1d = _numeric(frame, "ema_slow_1d")
            macd_1d = _numeric(frame, "macd_hist_1d")
            level = _numeric(frame, "prior_high_1d")
            ltf = close.lt(ema_fast_1h)
            htf = macd_1d.le(0.0) | ema_fast_1d.le(ema_slow_1d)
            source = close.lt(level.mul(1.0 - band))
        else:
            prior_low = _numeric(frame, "rwx_prior_low_48")
            prior_high = _numeric(frame, "rwx_prior_high_48")
            ltf = close.lt(prior_low) | close.gt(prior_high)
            htf = pd.Series(False, index=frame.index, dtype="bool")
            source = ltf

        return {
            "ltf": ltf.fillna(False),
            "htf": htf.fillna(False),
            "source": source.fillna(False),
        }

    def _rwx_invalidation_state(
        self,
        pair: str,
        current_time: Any,
        source_mode: str,
        confirmation_bars: int,
        band: float,
    ) -> bool:
        frame = self._rwx_frame(pair, current_time)
        if frame.empty:
            return False
        components = self._rwx_invalidation_components(frame, band)
        if source_mode == "ltf_trigger_failure":
            condition = components["ltf"]
        elif source_mode == "htf_context_flip":
            condition = components["htf"]
        elif source_mode == "source_level_failure":
            condition = components["source"]
        else:
            count = components["ltf"].astype(int) + components["htf"].astype(int) + components["source"].astype(int)
            condition = count.ge(2)
        needed = max(1, int(confirmation_bars))
        return len(condition) >= needed and bool(condition.tail(needed).all())

    @staticmethod
    def _rwx_exits_done(trade: Any) -> int:
        return int(getattr(trade, "nr_of_successful_exits", 0) or 0)

    @staticmethod
    def _rwx_stake(trade: Any) -> float:
        return float(getattr(trade, "stake_amount", 0.0) or 0.0)

    @staticmethod
    def _rwx_stop_from_profit(trade: Any, current_rate: float, stop_profit: float) -> float:
        open_rate = float(getattr(trade, "open_rate", 0.0) or current_rate or 0.0)
        rate = float(current_rate or 0.0)
        if open_rate <= 0.0 or rate <= 0.0:
            return -0.99
        is_short = bool(getattr(trade, "is_short", False))
        stop_price = open_rate * (1.0 - stop_profit) if is_short else open_rate * (1.0 + stop_profit)
        return stoploss_from_absolute(
            stop_price,
            current_rate=rate,
            is_short=is_short,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )

    def _rwx_base_stop(self, trade: Any, current_rate: float, hard_stop: float) -> float:
        return self._rwx_stop_from_profit(trade, current_rate, -float(hard_stop))


class TargetZoneExitMixin(ExitReworkCoreMixin):
    S3_BRANCH_FAMILY = "rework_target_zone"
    EXIT_FAMILY = S3_BRANCH_FAMILY
    EXIT_HYPOTHESIS = "Source-aware obstacle target with explicit band, confirmation timeframe, and action."
    ACTIVE_EXIT_PARAMETERS = (
        "target_source_mode",
        "target_lookback",
        "target_band_pct",
        "target_confirmation_timeframe",
        "target_confirmation_mode",
        "target_action_plan",
        "hard_stop_pct",
        "trail_distance_pct",
        "fallback_profit_pct",
    )

    target_source_mode = rework_exit_parameter(
        ["source_level", "prior_swing", "vp_poc", "vp_value_edge", "nearest_obstacle"],
        "nearest_obstacle",
    )
    target_lookback = rework_exit_parameter(REWORK_LOOKBACKS, 96)
    target_band_pct = rework_exit_parameter([0.005, 0.010, 0.020, 0.030], 0.010)
    target_confirmation_timeframe = rework_exit_parameter(["1h", "4h", "1d"], "1h")
    target_confirmation_mode = rework_exit_parameter(
        ["touch", "one_reversal", "two_reversal", "two_of_three", "close_reject"],
        "one_reversal",
    )
    target_action_plan = rework_exit_parameter(
        ["full_exit", "partial_33_then_be", "partial_50_then_trail", "tighten_only"],
        "partial_33_then_be",
    )
    hard_stop_pct = rework_exit_parameter([0.020, 0.030, 0.050, 0.080], 0.030)
    trail_distance_pct = rework_exit_parameter([0.010, 0.015, 0.025, 0.040], 0.015)
    fallback_profit_pct = rework_exit_parameter([0.030, 0.050, 0.080, 0.120], 0.050)

    def _rwx_current_target(self, pair: str, trade: Any, current_time: Any, current_rate: float) -> dict[str, Any]:
        return self._rwx_target_state(
            pair,
            trade,
            current_time,
            current_rate,
            source_mode=str(self.target_source_mode.value),
            lookback=int(self.target_lookback.value),
            band=float(self.target_band_pct.value),
            timeframe=str(self.target_confirmation_timeframe.value),
            confirmation=str(self.target_confirmation_mode.value),
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = kwargs
        if float(current_profit) >= float(self.fallback_profit_pct.value):
            return "rwx_target_fallback_profit"
        state = self._rwx_current_target(pair, trade, current_time, current_rate)
        if state["confirmed"] and str(self.target_action_plan.value) == "full_exit":
            return f"rwx_target_full_{state['provider']}"
        return None

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):
        _ = min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(getattr(trade, "has_open_orders", False)) or self._rwx_exits_done(trade) > 0:
            return None
        action = str(self.target_action_plan.value)
        if action not in {"partial_33_then_be", "partial_50_then_trail"}:
            return None
        state = self._rwx_current_target(getattr(trade, "pair", ""), trade, current_time, current_rate)
        if not state["confirmed"]:
            return None
        fraction = 0.33 if action == "partial_33_then_be" else 0.50
        stake = self._rwx_stake(trade)
        return (-(stake * fraction), f"rwx_target_partial_{state['provider']}") if stake > 0.0 else None

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = after_fill, kwargs
        stop_profit = -float(self.hard_stop_pct.value)
        action = str(self.target_action_plan.value)
        exits_done = self._rwx_exits_done(trade)
        if action == "partial_33_then_be" and exits_done >= 1:
            stop_profit = max(stop_profit, 0.001)
        elif action == "partial_50_then_trail" and exits_done >= 1:
            stop_profit = max(stop_profit, float(current_profit) - float(self.trail_distance_pct.value))
        elif action == "tighten_only":
            state = self._rwx_current_target(pair, trade, current_time, current_rate)
            if state["confirmed"]:
                stop_profit = max(stop_profit, float(current_profit) - float(self.trail_distance_pct.value))
        return self._rwx_stop_from_profit(trade, current_rate, stop_profit)


class SourceInvalidationExitMixin(ExitReworkCoreMixin):
    S3_BRANCH_FAMILY = "rework_source_invalidation"
    EXIT_FAMILY = S3_BRANCH_FAMILY
    EXIT_HYPOTHESIS = "Entry-family trigger, context, or broken-level invalidation with PnL-aware action."
    ACTIVE_EXIT_PARAMETERS = (
        "invalidation_source_mode",
        "invalidation_confirm_bars",
        "invalidation_action_plan",
        "invalidation_band_pct",
        "hard_stop_pct",
        "lock_distance_pct",
        "fallback_profit_pct",
    )

    invalidation_source_mode = rework_exit_parameter(
        ["ltf_trigger_failure", "htf_context_flip", "source_level_failure", "any_two_sources"],
        "source_level_failure",
    )
    invalidation_confirm_bars = rework_exit_parameter([1, 2, 3], 2)
    invalidation_action_plan = rework_exit_parameter(
        ["full_exit", "loss_exit_profit_lock", "partial_50_then_trail", "tighten_only"],
        "loss_exit_profit_lock",
    )
    invalidation_band_pct = rework_exit_parameter([0.000, 0.005, 0.010, 0.020], 0.005)
    hard_stop_pct = rework_exit_parameter([0.020, 0.030, 0.050, 0.080], 0.030)
    lock_distance_pct = rework_exit_parameter([0.005, 0.010, 0.015, 0.025], 0.010)
    fallback_profit_pct = rework_exit_parameter([0.030, 0.050, 0.080, 0.120], 0.050)

    def _rwx_invalidated(self, pair: str, current_time: Any) -> bool:
        return self._rwx_invalidation_state(
            pair,
            current_time,
            str(self.invalidation_source_mode.value),
            int(self.invalidation_confirm_bars.value),
            float(self.invalidation_band_pct.value),
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = current_rate, kwargs
        if float(current_profit) >= float(self.fallback_profit_pct.value):
            return "rwx_invalidation_fallback_profit"
        if not self._rwx_invalidated(pair, current_time):
            return None
        action = str(self.invalidation_action_plan.value)
        if action == "full_exit" or (action == "loss_exit_profit_lock" and float(current_profit) <= 0.0):
            return f"rwx_invalidation_{self.invalidation_source_mode.value}"
        return None

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):
        _ = current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if str(self.invalidation_action_plan.value) != "partial_50_then_trail":
            return None
        if bool(getattr(trade, "has_open_orders", False)) or self._rwx_exits_done(trade) > 0:
            return None
        if not self._rwx_invalidated(getattr(trade, "pair", ""), current_time):
            return None
        stake = self._rwx_stake(trade)
        return (-(stake * 0.50), "rwx_invalidation_partial_50") if stake > 0.0 else None

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = after_fill, kwargs
        stop_profit = -float(self.hard_stop_pct.value)
        if self._rwx_invalidated(pair, current_time):
            action = str(self.invalidation_action_plan.value)
            if action in {"loss_exit_profit_lock", "tighten_only"} and float(current_profit) > 0.0:
                stop_profit = max(stop_profit, float(current_profit) - float(self.lock_distance_pct.value))
            elif action == "tighten_only":
                stop_profit = max(stop_profit, float(current_profit) - float(self.lock_distance_pct.value))
        if str(self.invalidation_action_plan.value) == "partial_50_then_trail" and self._rwx_exits_done(trade) >= 1:
            stop_profit = max(stop_profit, float(current_profit) - float(self.lock_distance_pct.value))
        return self._rwx_stop_from_profit(trade, current_rate, stop_profit)


class LayeredTargetExitMixin(ExitReworkCoreMixin):
    S3_BRANCH_FAMILY = "rework_layered_target"
    EXIT_FAMILY = S3_BRANCH_FAMILY
    EXIT_HYPOTHESIS = "Target partial, explicit stop transition, and source invalidation for the remainder."
    ACTIVE_EXIT_PARAMETERS = (
        "layer_target_source",
        "layer_target_lookback",
        "layer_target_band_pct",
        "layer_confirmation_timeframe",
        "layer_target_plan",
        "layer_invalidation_plan",
        "layer_hard_stop_pct",
        "layer_trail_distance_pct",
        "layer_fallback_profit_pct",
    )

    layer_target_source = rework_exit_parameter(
        ["source_level", "prior_swing", "vp_poc", "vp_value_edge", "nearest_obstacle"],
        "nearest_obstacle",
    )
    layer_target_lookback = rework_exit_parameter(REWORK_LOOKBACKS, 96)
    layer_target_band_pct = rework_exit_parameter([0.005, 0.010, 0.020, 0.030], 0.010)
    layer_confirmation_timeframe = rework_exit_parameter(["1h", "4h", "1d"], "1h")
    layer_target_plan = rework_exit_parameter(
        [
            "touch_partial_33_be",
            "one_reversal_partial_50_trail",
            "two_of_three_partial_33_trail",
            "close_reject_partial_50_be",
        ],
        "one_reversal_partial_50_trail",
    )
    layer_invalidation_plan = rework_exit_parameter(
        ["source_full", "htf_tighten", "any_two_full_after_partial", "ltf_partial_then_trail"],
        "any_two_full_after_partial",
    )
    layer_hard_stop_pct = rework_exit_parameter([0.020, 0.030, 0.050, 0.080], 0.030)
    layer_trail_distance_pct = rework_exit_parameter([0.010, 0.015, 0.025, 0.040], 0.015)
    layer_fallback_profit_pct = rework_exit_parameter([0.050, 0.080, 0.120, 0.180], 0.080)

    def _rwx_layer_plan(self) -> tuple[str, float, str]:
        plan = str(self.layer_target_plan.value)
        mapping = {
            "touch_partial_33_be": ("touch", 0.33, "be"),
            "one_reversal_partial_50_trail": ("one_reversal", 0.50, "trail"),
            "two_of_three_partial_33_trail": ("two_of_three", 0.33, "trail"),
            "close_reject_partial_50_be": ("close_reject", 0.50, "be"),
        }
        return mapping[plan]

    def _rwx_layer_target(self, pair: str, trade: Any, current_time: Any, current_rate: float) -> dict[str, Any]:
        confirmation, _, _ = self._rwx_layer_plan()
        return self._rwx_target_state(
            pair,
            trade,
            current_time,
            current_rate,
            source_mode=str(self.layer_target_source.value),
            lookback=int(self.layer_target_lookback.value),
            band=float(self.layer_target_band_pct.value),
            timeframe=str(self.layer_confirmation_timeframe.value),
            confirmation=confirmation,
        )

    def _rwx_layer_invalidation(self, pair: str, current_time: Any) -> bool:
        plan = str(self.layer_invalidation_plan.value)
        source = {
            "source_full": "source_level_failure",
            "htf_tighten": "htf_context_flip",
            "any_two_full_after_partial": "any_two_sources",
            "ltf_partial_then_trail": "ltf_trigger_failure",
        }[plan]
        return self._rwx_invalidation_state(
            pair,
            current_time,
            source,
            2,
            float(self.layer_target_band_pct.value),
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = current_rate, kwargs
        if float(current_profit) >= float(self.layer_fallback_profit_pct.value):
            return "rwx_layer_fallback_profit"
        if not self._rwx_layer_invalidation(pair, current_time):
            return None
        plan = str(self.layer_invalidation_plan.value)
        exits_done = self._rwx_exits_done(trade)
        if plan == "source_full" or (plan == "any_two_full_after_partial" and exits_done >= 1):
            return f"rwx_layer_invalidation_{plan}"
        return None

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):
        _ = current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(getattr(trade, "has_open_orders", False)) or self._rwx_exits_done(trade) > 0:
            return None
        _, fraction, _ = self._rwx_layer_plan()
        target = self._rwx_layer_target(getattr(trade, "pair", ""), trade, current_time, current_rate)
        invalidation_partial = str(self.layer_invalidation_plan.value) == "ltf_partial_then_trail" and self._rwx_layer_invalidation(getattr(trade, "pair", ""), current_time)
        if not target["confirmed"] and not invalidation_partial:
            return None
        stake = self._rwx_stake(trade)
        reason = f"rwx_layer_partial_{target['provider']}" if target["confirmed"] else "rwx_layer_partial_ltf_failure"
        return (-(stake * fraction), reason) if stake > 0.0 else None

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = after_fill, kwargs
        stop_profit = -float(self.layer_hard_stop_pct.value)
        _, _, remainder = self._rwx_layer_plan()
        exits_done = self._rwx_exits_done(trade)
        if exits_done >= 1:
            if remainder == "be":
                stop_profit = max(stop_profit, 0.001)
            else:
                stop_profit = max(stop_profit, float(current_profit) - float(self.layer_trail_distance_pct.value))
        if self._rwx_layer_invalidation(pair, current_time):
            plan = str(self.layer_invalidation_plan.value)
            if plan in {"htf_tighten", "any_two_full_after_partial", "ltf_partial_then_trail"}:
                stop_profit = max(stop_profit, float(current_profit) - float(self.layer_trail_distance_pct.value))
        return self._rwx_stop_from_profit(trade, current_rate, stop_profit)


_MATERIALIZED_EXIT_METHODS = {
    "custom_exit",
    "custom_stoploss",
    "adjust_trade_position",
    "populate_exit_trend",
}


def _materialized_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    frame = self._rwx_source_populate_indicators(dataframe, metadata)
    return self._rwx_add_indicators(frame)


def _materialized_populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    _ = metadata
    return dataframe


def materialize_rework_strategy(
    strategy_cls: type,
    source_cls: type,
    exit_mixin_cls: type,
    *,
    source_profile: str,
    source_target_columns: tuple[str, ...],
) -> None:
    """Flatten source entry behavior and one refined exit family onto a direct IStrategy class."""

    source_populate = source_cls.__dict__.get("populate_indicators")
    if source_populate is None:
        raise TypeError(f"{source_cls.__name__} does not define populate_indicators")

    for name, value in source_cls.__dict__.items():
        if name.startswith("__") or name in _MATERIALIZED_EXIT_METHODS or name == "populate_indicators":
            continue
        if name.startswith("_s3_") or name.startswith("S3_") or name.startswith("EXIT_"):
            continue
        if isinstance(value, BaseParameter) and str(value.space or "") == "sell":
            continue
        setattr(strategy_cls, name, value)

    for mixin in reversed(exit_mixin_cls.__mro__[:-1]):
        for name, value in mixin.__dict__.items():
            if name.startswith("__") or name == "populate_indicators":
                continue
            setattr(strategy_cls, name, value)

    strategy_cls._rwx_source_populate_indicators = source_populate
    strategy_cls.populate_indicators = _materialized_populate_indicators
    strategy_cls.populate_exit_trend = _materialized_populate_exit_trend
    strategy_cls.REWORK_SOURCE_PROFILE = str(source_profile)
    strategy_cls.REWORK_SOURCE_TARGET_COLUMNS = tuple(source_target_columns)
    strategy_cls.SOURCE_ENTRY_STEM = str(getattr(source_cls, "SOURCE_ENTRY_STEM", "") or "")
    strategy_cls.SOURCE_ENTRY_CLASS = str(getattr(source_cls, "SOURCE_ENTRY_CLASS", source_cls.__name__))
    update_abstractmethods(strategy_cls)
