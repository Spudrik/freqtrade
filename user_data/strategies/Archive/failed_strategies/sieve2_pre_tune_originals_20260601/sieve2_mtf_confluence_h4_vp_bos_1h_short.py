from __future__ import annotations
# Archived pre-tune copy from 2026-06-01 review. Active top-level file remains in Sieve2;
# reason: tune/fix candidate for entry-only rerun after high-performer 4/2 validation.

# New Sieve2 MTF confluence strategy; entry-only research variant.

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
ENTRY_MODE = "entry_mtf_confluence_h4_vp_bos_1h_short"
ENTRY_TAG = "mtf_confluence_h4_vp_bos_1h_short"
SIEVE_STAGE = "sieve2"
SOURCE_STRATEGY = "new_mtf_confluence_concept"
SOURCE_RESULT_BATCH = "20260529_manual_mtf_confluence_batch"
RESEARCH_PATH = "mtf_confluence_flexible_gate"
UPDATE_HYPOTHESIS = "H4 VP plus BOS/structure context, 1H breakdown/failed-reclaim execution. Uses flexible count-based gates so hyperopt can test two-gate and three-gate confluence rather than one rigid chain."
SIDE = "short"
CONTEXT_TIMEFRAME = "4h"
TIMEFRAME = "1h"


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


def _date_naive(value: Series) -> Series:
    return pd.to_datetime(value, utc=True, errors="coerce").dt.tz_convert(None)


class Sieve2MTFConfluenceH4VPBOS1hShort(IStrategy):
    """Flexible MTF confluence probe: HTF confidence, LTF execution."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 520
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, 0.02)}
    stoploss = -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_sieve2_vp_guard = CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True)
    sieve2_vp_guard_mode = CategoricalParameter(SIEVE2_VP_GUARD_MODES, default="score_or_context", space="buy", optimize=True, load=True)
    sieve2_vp_window = CategoricalParameter([48, 96, 168], default=96, space="buy", optimize=False, load=True)
    sieve2_vp_bins = CategoricalParameter([24, 36, 48], default=36, space="buy", optimize=False, load=True)
    sieve2_vp_score_min = CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True)
    sieve2_vp_context_min = CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=True, load=True)
    use_sieve2_market_guard = CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True)
    sieve2_market_guard_mode = CategoricalParameter(SIEVE2_MARKET_GUARD_MODES, default="pressure_or_trend", space="buy", optimize=True, load=True)
    sieve2_market_window = CategoricalParameter([12, 24, 48, 96], default=24, space="buy", optimize=False, load=True)
    sieve2_market_pressure_min = CategoricalParameter([0.03, 0.07, 0.12, 0.18, 0.25], default=0.07, space="buy", optimize=True, load=True)
    sieve2_market_trend_min = CategoricalParameter([0.0, 0.25, 0.50, 0.80], default=0.25, space="buy", optimize=True, load=True)
    sieve2_rs_benchmark_pair = CategoricalParameter(["BTC/USDT:USDT", "ETH/USDT:USDT"], default="BTC/USDT:USDT", space="buy", optimize=False, load=True)
    sieve2_rs_score_min = CategoricalParameter([0.35, 0.45, 0.55, 0.65], default=0.45, space="buy", optimize=True, load=True)

    min_htf_gates = tagged_parameter(CategoricalParameter([1, 2, 3], default=1, space="buy", optimize=True, load=True))
    min_ltf_gates = tagged_parameter(CategoricalParameter([1, 2], default=1, space="buy", optimize=True, load=True))
    min_total_gates = tagged_parameter(CategoricalParameter([2, 3, 4, 5], default=3, space="buy", optimize=True, load=True))
    htf_lookback = tagged_parameter(CategoricalParameter([24, 48, 72, 120], default=48, space="buy", optimize=True, load=True))
    ltf_lookback = tagged_parameter(CategoricalParameter([6, 12, 24, 48], default=12, space="buy", optimize=True, load=True))
    recent_htf_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=3, space="buy", optimize=True, load=True))
    level_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.002, 0.004, 0.008], default=0.002, space="buy", optimize=True, load=True))
    retest_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012], default=0.004, space="buy", optimize=True, load=True))
    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True))
    vp_context_min = tagged_parameter(CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=True, load=True))
    ltf_volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.2, 1.5], default=1.0, space="buy", optimize=True, load=True))
    ltf_pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18], default=0.07, space="buy", optimize=True, load=True))
    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))

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
        lookback = int(self.ltf_lookback.value)
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        baseline = volume.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).mean().replace(0.0, np.nan)
        frame["mtf_ltf_pressure"] = pressure
        frame["mtf_ltf_volume_ratio"] = volume / baseline
        frame["mtf_ltf_local_high"] = high.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).max()
        frame["mtf_ltf_local_low"] = low.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).min()
        frame = add_volume_profile(frame, window=max(48, min(168, lookback * 4)), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="ltfvp")
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 6, 160)), breakout_buffer_atr=0.15, prefix="ltfms")
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["mtf_htf_vp_gate"] = False
        frame["mtf_htf_structure_gate"] = False
        frame["mtf_htf_prior_gate"] = False
        if "date" not in frame.columns or not getattr(self, "dp", None):
            return frame
        pair = str((metadata or {}).get("pair") or "")
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or "date" not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        keep = ["date", "mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_prior_gate"]
        informative = informative[keep].copy().sort_values("date")
        informative["date_merge"] = _date_naive(informative["date"]) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")
        base = frame.drop(columns=["mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_prior_gate"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        base["__date_merge"] = _date_naive(base["date"])
        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")
        merged = merged.sort_values("__row_index").drop(columns=["__row_index", "__date_merge", "date_merge"], errors="ignore")
        merged.index = dataframe.index
        for column in ("mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_prior_gate"):
            merged[column] = pd.Series(merged[column], index=merged.index).astype("boolean").fillna(False).astype(bool)
        return merged

    def _add_htf_features(self, informative: DataFrame) -> DataFrame:
        frame = informative.copy()
        lookback = int(self.htf_lookback.value)
        buffer = float(self.level_buffer_pct.value)
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()
        ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()
        frame = add_volume_profile(frame, window=max(24, min(168, lookback)), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="htfvp")
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix="htfms")
        score_long = _num(frame, "htfvp_score_long")
        score_short = _num(frame, "htfvp_score_short")
        ctx_long = _num(frame, "htfvp_context_score_bull")
        ctx_short = _num(frame, "htfvp_context_score_bear")
        prior_vah = _num(frame, "htfvp_prior_vah", np.nan)
        prior_val = _num(frame, "htfvp_prior_val", np.nan)
        if SIDE == "short":
            vp_gate = ((score_short.ge(float(self.vp_score_min.value)) | ctx_short.ge(float(self.vp_context_min.value)) | close.lt(prior_val)) & score_short.ge(score_long)).fillna(False)
            structure_gate = (_bool(frame, "htfms_bos_to_bear") | (_num(frame, "htfms_state").le(0.0) & close.lt(ema))).fillna(False)
            prior_gate = (close.lt(prior_low.mul(1.0 - buffer)) | (high.gt(prior_high.mul(1.0 + buffer)) & close.lt(prior_high))).fillna(False)
        else:
            vp_gate = ((score_long.ge(float(self.vp_score_min.value)) | ctx_long.ge(float(self.vp_context_min.value)) | close.gt(prior_vah)) & score_long.ge(score_short)).fillna(False)
            structure_gate = (_bool(frame, "htfms_bos_to_bull") | (_num(frame, "htfms_state").ge(0.0) & close.gt(ema))).fillna(False)
            prior_gate = (close.gt(prior_high.mul(1.0 + buffer)) | (low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low))).fillna(False)
        recent = int(self.recent_htf_bars.value)
        frame["mtf_htf_vp_gate"] = vp_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        frame["mtf_htf_structure_gate"] = structure_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        frame["mtf_htf_prior_gate"] = prior_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        return frame

    def _gate_count(self, gates: list[Series], index: pd.Index) -> Series:
        total = pd.Series(0, index=index, dtype="int64")
        for gate in gates:
            total = total + pd.Series(gate, index=index).fillna(False).astype(bool).astype("int64")
        return total

    def _ltf_gates(self, dataframe: DataFrame) -> list[Series]:
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        buffer = float(self.level_buffer_pct.value)
        retest = float(self.retest_buffer_pct.value)
        local_high = _num(dataframe, "mtf_ltf_local_high", np.nan)
        local_low = _num(dataframe, "mtf_ltf_local_low", np.nan)
        pressure = _num(dataframe, "mtf_ltf_pressure")
        volume_ratio = _num(dataframe, "mtf_ltf_volume_ratio")
        if SIDE == "short":
            local_break_or_retest = _cross_below(close, local_low.mul(1.0 - buffer)) | (high.ge(local_low.mul(1.0 - retest)) & close.lt(local_low))
            structure = _bool(dataframe, "ltfms_bos_to_bear") | _num(dataframe, "ltfms_state").le(0.0)
            vp_or_pressure = _bool(dataframe, "ltfvp_node_entry_short") | _bool(dataframe, "ltfvp_node_hold_short") | _num(dataframe, "ltfvp_score_short").ge(float(self.vp_score_min.value)) | pressure.le(-float(self.ltf_pressure_min.value))
            volume_gate = volume_ratio.ge(float(self.ltf_volume_ratio_min.value))
        else:
            local_break_or_retest = _cross_above(close, local_high.mul(1.0 + buffer)) | (low.le(local_high.mul(1.0 + retest)) & close.gt(local_high))
            structure = _bool(dataframe, "ltfms_bos_to_bull") | _num(dataframe, "ltfms_state").ge(0.0)
            vp_or_pressure = _bool(dataframe, "ltfvp_node_entry_long") | _bool(dataframe, "ltfvp_node_hold_long") | _num(dataframe, "ltfvp_score_long").ge(float(self.vp_score_min.value)) | pressure.ge(float(self.ltf_pressure_min.value))
            volume_gate = volume_ratio.ge(float(self.ltf_volume_ratio_min.value))
        context_gate = vp_or_pressure
        if bool(self.use_ltf_volume_guard.value):
            context_gate &= volume_gate
        if bool(self.use_ltf_pressure_guard.value):
            context_gate &= pressure.le(-float(self.ltf_pressure_min.value)) if SIDE == "short" else pressure.ge(float(self.ltf_pressure_min.value))
        return [local_break_or_retest.fillna(False), structure.fillna(False), context_gate.fillna(False)]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        htf_gates = [_bool(dataframe, "mtf_htf_vp_gate"), _bool(dataframe, "mtf_htf_structure_gate"), _bool(dataframe, "mtf_htf_prior_gate")]
        ltf_gates = self._ltf_gates(dataframe)
        htf_count = self._gate_count(htf_gates, dataframe.index)
        ltf_count = self._gate_count(ltf_gates, dataframe.index)
        total_count = htf_count + ltf_count
        condition = htf_count.ge(int(self.min_htf_gates.value)) & ltf_count.ge(int(self.min_ltf_gates.value)) & total_count.ge(int(self.min_total_gates.value))
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


apply_explicit_hyperopt_surface(Sieve2MTFConfluenceH4VPBOS1hShort)
