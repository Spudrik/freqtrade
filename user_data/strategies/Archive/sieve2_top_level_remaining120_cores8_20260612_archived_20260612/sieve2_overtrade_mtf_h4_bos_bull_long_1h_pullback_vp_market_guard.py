from __future__ import annotations

# Updated Sieve2 rescue clone from sieve2_mtf_h4_bos_bull_long_1h_pullback; entry-only research variant.

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.strategies.sieve_guard_helpers import (
    SIEVE2_MARKET_GUARD_MODES,
    SIEVE2_VP_GUARD_MODES,
    add_sieve2_guard_indicators,
    apply_sieve2_optional_guards,
)

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
ENTRY_MODE = "entry_overtrade_mtf_h4_bos_bull_long_1h_pullback_vp_market_guard"
ENTRY_TAG = "overtrade_mtf_h4_bos_bull_long_1h_pullback_vp_market_guard"
SIEVE_STAGE = "sieve2"
SOURCE_STRATEGY = "sieve2_mtf_h4_bos_bull_long_1h_pullback"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = "overtrade_vp_market_guard"
UPDATE_HYPOTHESIS = "High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution."
SIDE = "long"
TIMEFRAME = "1h"
CONTEXT_TIMEFRAME = "4h"
HTF_BEHAVIOR = "bos_bull_long"
LTF_TRIGGER = "pullback_reclaim"


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


def _pct_env(name: str, default_ratio: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default_ratio)
    try:
        return max(0.0, float(raw)) / 100.0
    except ValueError:
        return float(default_ratio)


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
    return param


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


class Sieve2OvertradeMTFH4BOSBullLong1hPullbackVPMarketGuard(IStrategy):
    """MTF Sieve2 probe: 4h bullish BOS context; 1h pullback reclaim."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 420
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, 0.02)}
    stoploss = -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_sieve2_vp_guard = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    sieve2_vp_guard_mode = CategoricalParameter(SIEVE2_VP_GUARD_MODES, default="score_or_context", space="buy", optimize=True, load=True)
    sieve2_vp_window = CategoricalParameter([48, 96, 168], default=96, space="buy", optimize=False, load=True)
    sieve2_vp_bins = CategoricalParameter([24, 36, 48], default=36, space="buy", optimize=False, load=True)
    sieve2_vp_score_min = CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True)
    sieve2_vp_context_min = CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=True, load=True)
    use_sieve2_market_guard = CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True)
    sieve2_market_guard_mode = CategoricalParameter(SIEVE2_MARKET_GUARD_MODES, default="pressure_or_trend", space="buy", optimize=True, load=True)
    sieve2_market_window = CategoricalParameter([12, 24, 48, 96], default=24, space="buy", optimize=False, load=True)
    sieve2_market_pressure_min = CategoricalParameter([0.03, 0.07, 0.12, 0.18, 0.25], default=0.07, space="buy", optimize=True, load=True)
    sieve2_market_trend_min = CategoricalParameter([0.0, 0.25, 0.50, 0.80], default=0.25, space="buy", optimize=True, load=True)
    sieve2_rs_benchmark_pair = CategoricalParameter(["BTC/USDT:USDT", "ETH/USDT:USDT"], default="BTC/USDT:USDT", space="buy", optimize=False, load=True)
    sieve2_rs_score_min = CategoricalParameter([0.35, 0.45, 0.55, 0.65], default=0.45, space="buy", optimize=True, load=True)

    context_lookback = tagged_parameter(CategoricalParameter([24, 48, 72, 120], default=48, space="buy", optimize=True, load=True))
    context_recent_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=3, space="buy", optimize=True, load=True))
    local_lookback = tagged_parameter(CategoricalParameter([6, 12, 24, 48], default=12, space="buy", optimize=True, load=True))
    retest_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012], default=0.004, space="buy", optimize=True, load=True))
    breakout_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.002, 0.004, 0.008], default=0.002, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.2, 1.5], default=1.0, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18], default=0.07, space="buy", optimize=True, load=True))
    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True))
    vp_context_min = tagged_parameter(CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=True, load=True))
    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    use_htf_turn_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        try:
            return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]
        except Exception:
            return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        window = int(self.local_lookback.value)
        candle_range = (high - low).replace(0.0, np.nan)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        volume_baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        frame["mtf_ltf_pressure"] = pressure
        frame["mtf_ltf_volume_ratio"] = volume / volume_baseline
        frame["mtf_ltf_ema"] = close.ewm(span=window, min_periods=max(2, window // 3), adjust=False).mean()
        frame["mtf_ltf_local_high"] = high.shift(1).rolling(window, min_periods=max(2, window // 3)).max()
        frame["mtf_ltf_local_low"] = low.shift(1).rolling(window, min_periods=max(2, window // 3)).min()
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        for column, default in (("mtf_htf_context", False), ("mtf_htf_turn_bad", True)):
            frame[column] = default
        frame["mtf_htf_level"] = np.nan
        if "date" not in frame.columns or not getattr(self, "dp", None):
            return frame
        pair = str((metadata or {}).get("pair") or "")
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or "date" not in informative.columns:
            return frame
        informative = self._add_htf_context(informative.copy())
        keep = ["date", "mtf_htf_context", "mtf_htf_level", "mtf_htf_turn_bad"]
        informative = informative[[column for column in keep if column in informative.columns]].copy().sort_values("date")
        informative["date_merge"] = pd.to_datetime(informative["date"], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")
        base = frame.drop(columns=["mtf_htf_context", "mtf_htf_level", "mtf_htf_turn_bad"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        base["__date_merge"] = pd.to_datetime(base["date"], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")
        merged = merged.sort_values("__row_index").drop(columns=["__row_index", "__date_merge", "date_merge"], errors="ignore")
        merged.index = dataframe.index
        merged["mtf_htf_context"] = pd.Series(merged["mtf_htf_context"], index=merged.index).astype("boolean").fillna(False).astype(bool)
        merged["mtf_htf_turn_bad"] = pd.Series(merged["mtf_htf_turn_bad"], index=merged.index).astype("boolean").fillna(True).astype(bool)
        return merged

    def _add_htf_context(self, informative: DataFrame) -> DataFrame:
        frame = informative.copy()
        lookback = int(self.context_lookback.value)
        recent = int(self.context_recent_bars.value)
        buffer = float(self.breakout_buffer_pct.value)
        pressure_min = float(self.pressure_min.value)
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0
        volume_sum = volume.rolling(lookback, min_periods=max(3, lookback // 4)).sum().replace(0.0, np.nan)
        pressure_ratio = (pressure * volume).rolling(lookback, min_periods=max(3, lookback // 4)).sum() / volume_sum
        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()
        trend_ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()
        frame = add_volume_profile(frame, window=min(max(lookback, 24), 168), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="htfvp")
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix="htfms")
        vp_score_long = _num(frame, "htfvp_score_long")
        vp_score_short = _num(frame, "htfvp_score_short")
        vp_ctx_long = _num(frame, "htfvp_context_score_bull")
        vp_ctx_short = _num(frame, "htfvp_context_score_bear")
        prior_vah = _num(frame, "htfvp_prior_vah", np.nan)
        prior_val = _num(frame, "htfvp_prior_val", np.nan)
        node_long = _bool(frame, "htfvp_node_entry_long") | _bool(frame, "htfvp_node_hold_long")
        node_short = _bool(frame, "htfvp_node_entry_short") | _bool(frame, "htfvp_node_hold_short")
        vp_long_ok = (vp_score_long.ge(float(self.vp_score_min.value)) | vp_ctx_long.ge(float(self.vp_context_min.value))) & vp_score_long.ge(vp_score_short)
        vp_short_ok = (vp_score_short.ge(float(self.vp_score_min.value)) | vp_ctx_short.ge(float(self.vp_context_min.value))) & vp_score_short.ge(vp_score_long)
        behavior = HTF_BEHAVIOR
        if behavior in {"res_break_long", "prior_high_break_long", "supply_break_long"}:
            context = close.gt(prior_high.mul(1.0 + buffer))
            level = prior_high
        elif behavior == "vp_value_accept_long":
            context = close.gt(prior_vah) & vp_long_ok
            level = prior_vah
        elif behavior == "vp_node_break_long":
            context = (node_long | vp_long_ok) & close.gt(prior_vah)
            level = prior_vah.fillna(prior_high)
        elif behavior == "support_reclaim_long":
            context = low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low) & pressure_ratio.ge(-pressure_min)
            level = prior_low
        elif behavior == "bos_bull_long":
            context = _bool(frame, "htfms_bos_to_bull") | (_num(frame, "htfms_state").ge(0.0) & close.gt(trend_ema) & pressure_ratio.ge(-pressure_min))
            level = prior_high.fillna(trend_ema)
        elif behavior in {"prior_low_break_short", "support_break_short"}:
            context = close.lt(prior_low.mul(1.0 - buffer))
            level = prior_low
        elif behavior == "vp_value_break_short":
            context = close.lt(prior_val) & vp_short_ok
            level = prior_val
        elif behavior in {"vp_vah_reject_short", "vp_node_reject_short"}:
            context = high.ge(prior_vah) & close.lt(prior_vah) & (vp_short_ok | node_short)
            level = prior_vah
        elif behavior == "res_reject_short":
            context = high.gt(prior_high.mul(1.0 - buffer)) & close.lt(prior_high) & pressure_ratio.le(pressure_min)
            level = prior_high
        elif behavior == "bos_bear_short":
            context = _bool(frame, "htfms_bos_to_bear") | (_num(frame, "htfms_state").le(0.0) & close.lt(trend_ema) & pressure_ratio.le(pressure_min))
            level = prior_low.fillna(trend_ema)
        else:
            context = pd.Series(False, index=frame.index)
            level = pd.Series(np.nan, index=frame.index)
        frame["mtf_htf_context"] = context.fillna(False).astype("int8").rolling(recent, min_periods=1).max().gt(0)
        frame["mtf_htf_level"] = level
        if SIDE == "short":
            frame["mtf_htf_turn_bad"] = (close.gt(trend_ema) & pressure_ratio.ge(pressure_min)).fillna(False)
        else:
            frame["mtf_htf_turn_bad"] = (close.lt(trend_ema) & pressure_ratio.le(-pressure_min)).fillna(False)
        return frame

    def _ltf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = _num(dataframe, "mtf_htf_level", np.nan)
        ema = _num(dataframe, "mtf_ltf_ema", np.nan)
        local_high = _num(dataframe, "mtf_ltf_local_high", np.nan)
        local_low = _num(dataframe, "mtf_ltf_local_low", np.nan)
        retest_buffer = float(self.retest_buffer_pct.value)
        breakout_buffer = float(self.breakout_buffer_pct.value)
        pressure = _num(dataframe, "mtf_ltf_pressure")
        if LTF_TRIGGER == "local_break":
            return close.lt(local_low.mul(1.0 - breakout_buffer)) if SIDE == "short" else close.gt(local_high.mul(1.0 + breakout_buffer))
        if LTF_TRIGGER == "pullback_reclaim":
            return (high.ge(ema.mul(1.0 - retest_buffer)) & close.lt(ema) & close.lt(open_)) if SIDE == "short" else (low.le(ema.mul(1.0 + retest_buffer)) & close.gt(ema) & close.gt(open_))
        if LTF_TRIGGER == "momentum_confirm":
            return (close.lt(close.shift(1)) & pressure.le(-float(self.pressure_min.value))) if SIDE == "short" else (close.gt(close.shift(1)) & pressure.ge(float(self.pressure_min.value)))
        if LTF_TRIGGER == "rejection_reclaim":
            return (high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_)) if SIDE == "short" else (low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_))
        return (high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_)) if SIDE == "short" else (low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_))

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "mtf_htf_context") & self._ltf_trigger(dataframe)
        if bool(self.use_ltf_volume_guard.value):
            condition &= _num(dataframe, "mtf_ltf_volume_ratio").ge(float(self.volume_ratio_min.value))
        if bool(self.use_ltf_pressure_guard.value):
            pressure = _num(dataframe, "mtf_ltf_pressure")
            condition &= pressure.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure.ge(float(self.pressure_min.value))
        if bool(self.use_htf_turn_guard.value):
            condition &= ~_bool(dataframe, "mtf_htf_turn_bad")
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
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


apply_explicit_hyperopt_surface(Sieve2OvertradeMTFH4BOSBullLong1hPullbackVPMarketGuard)
