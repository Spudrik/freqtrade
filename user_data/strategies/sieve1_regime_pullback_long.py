from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile


HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"

GUARD_MODE_CHOICES = ["direction", "score", "context", "score_or_context", "balance"]
PRICE_SOURCE_CHOICES = ["close", "hl2", "hlc3", "ohlc4"]
VP_COLUMNS = [
    "score_long",
    "score_short",
    "context_score_bull",
    "context_score_bear",
    "context_score_balance",
    "market_context",
]


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


def entry_sieve_minimal_roi(default: float = 0.02) -> dict[str, float]:
    return {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, default)}


def entry_sieve_stoploss(default: float = -0.02) -> float:
    return -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, abs(default))


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def tagged_parameter(param: Any, *tags: str) -> Any:
    family_map = {
        "entry": "entries",
        "entries": "entries",
        "structure": "entries",
        "volume": "entries",
        "entry_enables": "entries",
        "entry_confirmation": "entries",
        "entry_structure": "entries",
        "trend_filter": "entries",
        "daily_structure": "entries",
        "hourly_execution": "entries",
        "line_quality": "entries",
        "target_space": "entries",
        "trendline_projection": "entries",
        "breakout_long": "entries",
        "exit": "exits",
        "exits": "exits",
        "exit_profile": "exits",
        "exit_peel": "exits",
        "exit_target": "exits",
        "entry_family_exit": "exits",
        "stoploss": "exits",
        "adjust_position": "adjust_position",
        "add_enables": "adjust_position",
        "add_rules": "adjust_position",
        "capital": "stake",
        "stake": "stake",
        "risk": "risk",
    }
    family: str | None = None
    mode: str | None = None
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag or ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "family":
            mapped = family_map.get(value.lower())
            if mapped and family is None:
                family = mapped
        elif key == "mode" and value and mode is None:
            mode = value

    if family is None:
        if mode and mode.startswith("exit_"):
            family = "exits"
        elif mode and mode.startswith("adjust_"):
            family = "adjust_position"
        elif mode and mode.startswith("stake_"):
            family = "stake"
        elif mode and mode.startswith("risk_"):
            family = "risk"
        else:
            family = "entries"

    if mode is None:
        mode_defaults = {
            "entries": "entry_core",
            "exits": "exit_core",
            "adjust_position": "adjust_core",
            "stake": "stake_core",
            "risk": "risk_core",
        }
        mode = mode_defaults.get(family, "entry_core")

    setattr(param, "batch_tags", (f"family:{family}", f"mode:{mode}"))
    return param


class Sieve1RegimePullbackLong(IStrategy):
    """
    Trend-regime pullback strategy.

    Hypothesis:
    - Only trade in a directional EMA regime.
    - Enter when price pulls back toward the faster trend mean, momentum resets,
      then closes back in the trend direction on acceptable volume.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = False

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    ema_fast_period = tagged_parameter(
        IntParameter(18, 36, default=24, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback_long",
    )
    ema_slow_period = tagged_parameter(
        IntParameter(72, 144, default=96, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback_long",
    )
    rsi_pullback_low = tagged_parameter(
        IntParameter(36, 48, default=42, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback_long",
    )
    pullback_atr_max = tagged_parameter(
        DecimalParameter(0.10, 1.20, decimals=2, default=0.55, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback_long",
    )
    volume_ratio_min = tagged_parameter(
        DecimalParameter(0.60, 1.60, decimals=2, default=0.85, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback_long",
    )


    use_vp_4h_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_4h_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="score_or_context", space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_4h_window = tagged_parameter(IntParameter(12, 96, default=48, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_4h_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_4h_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_4h_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    use_vp_1d_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_1d_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="context", space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_1d_window = tagged_parameter(IntParameter(10, 84, default=30, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_1d_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_1d_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_1d_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_guard_value_area_pct = tagged_parameter(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_guard_price_source = tagged_parameter(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_guard_node_near_pct = tagged_parameter(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")
    vp_guard_pressure_delta_min = tagged_parameter(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=True, load=True), "family:entry", "mode:entry_vp_guard")

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        try:
            whitelist = self.dp.current_whitelist()
            return [(pair, "4h") for pair in whitelist] + [(pair, "1d") for pair in whitelist]
        except Exception:
            return []

    @staticmethod
    def _rsi(close: Series, period: int = 14) -> Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = gain / loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14) -> Series:
        prev_close = dataframe["close"].shift(1)
        tr = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - prev_close).abs(),
                (dataframe["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        fast = int(self.ema_fast_period.value)
        slow = int(self.ema_slow_period.value)
        dataframe["ema_fast"] = dataframe["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(48, min_periods=24).mean()
        dataframe["pullback_dist_atr"] = (
            (dataframe["close"] - dataframe["ema_fast"]).abs() / dataframe["atr"].replace(0.0, np.nan)
        )
        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))
        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        long_regime = dataframe["ema_fast"] > dataframe["ema_slow"]
        volume_ok = dataframe["volume_ratio"] >= float(self.volume_ratio_min.value)
        near_mean = dataframe["pullback_dist_atr"] <= float(self.pullback_atr_max.value)

        long_trigger = (
            long_regime
            & near_mean
            & volume_ok
            & dataframe["rsi"].between(float(self.rsi_pullback_low.value), 55.0)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["ema_fast"])
        )
        if bool(self.use_vp_4h_guard.value):
            long_trigger &= self._vp_guard(dataframe, "vp4h", "long", str(self.vp_4h_guard_mode.value), float(self.vp_4h_score_min.value), float(self.vp_4h_context_min.value))
        if bool(self.use_vp_1d_guard.value):
            long_trigger &= self._vp_guard(dataframe, "vp1d", "long", str(self.vp_1d_guard_mode.value), float(self.vp_1d_score_min.value), float(self.vp_1d_context_min.value))

        dataframe.loc[long_trigger.fillna(False), "enter_long"] = 1
        dataframe.loc[long_trigger.fillna(False), "enter_tag"] = "regime_pullback_long"
        return dataframe


    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:
        if "date" not in dataframe.columns or not getattr(self, "dp", None):
            return dataframe
        pair = str(metadata.get("pair") or "")
        if not pair:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative is None or informative.empty or "date" not in informative.columns:
            return dataframe
        informative = add_volume_profile(
            informative.copy(),
            window=window,
            bins=bins,
            value_area_pct=float(self.vp_guard_value_area_pct.value),
            price_source=str(self.vp_guard_price_source.value),
            pressure_delta_min=float(self.vp_guard_pressure_delta_min.value),
            node_near_pct=float(self.vp_guard_node_near_pct.value),
            prefix=prefix,
        )
        merge_columns = [f"{prefix}_{name}" for name in VP_COLUMNS if f"{prefix}_{name}" in informative.columns]
        if not merge_columns:
            return dataframe
        minutes = timeframe_to_minutes(timeframe)
        inf = informative[["date", *merge_columns]].copy().sort_values("date")
        inf["date_merge"] = inf["date"] + pd.to_timedelta(minutes, unit="m")
        base = dataframe.reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        merged = pd.merge_asof(
            base,
            inf[["date_merge", *merge_columns]].sort_values("date_merge"),
            left_on="date",
            right_on="date_merge",
            direction="backward",
        ).sort_values("__row_index")
        for column in merge_columns:
            dataframe[column] = merged[column].to_numpy()
        return dataframe

    @staticmethod
    def _score_guard(dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:
        score = _num(dataframe, f"{prefix}_score_{side}")
        opposite = _num(dataframe, f"{prefix}_score_{'short' if side == 'long' else 'long'}")
        return score.ge(score_min) & score.ge(opposite)

    @staticmethod
    def _context_guard(dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:
        if side == "long":
            context = _num(dataframe, f"{prefix}_context_score_bull")
            opposite = _num(dataframe, f"{prefix}_context_score_bear")
            market_ok = _num(dataframe, f"{prefix}_market_context").ge(0)
        else:
            context = _num(dataframe, f"{prefix}_context_score_bear")
            opposite = _num(dataframe, f"{prefix}_context_score_bull")
            market_ok = _num(dataframe, f"{prefix}_market_context").le(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:
        score_ok = self._score_guard(dataframe, prefix, side, score_min)
        context_ok = self._context_guard(dataframe, prefix, side, context_min)
        balance_ok = _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)
        if mode == "score":
            return score_ok
        if mode == "context":
            return context_ok
        if mode == "score_or_context":
            return score_ok | context_ok
        if mode == "balance":
            return balance_ok
        if side == "long":
            return _num(dataframe, f"{prefix}_market_context").ge(0)
        return _num(dataframe, f"{prefix}_market_context").le(0)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(Sieve1RegimePullbackLong)



