from __future__ import annotations

import os
from datetime import datetime
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
ENTRY_MODE = "entry_vp_lvn_fast_traverse_long"
ENTRY_TAG = "vp_lvn_fast_traverse_long"
SIDE = "long"
TRIGGER_COLUMN = "vp_lvn_fast_traverse_long"
CORE_BEHAVIOR = "fast upside traverse through LVN"
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


class Sieve1VpLvnFastTraverseLong(IStrategy):
    """
    Sieve1 volume-profile entry concept: fast upside traverse through LVN.

    The trigger is intentionally narrow. Optional 4h/1d VP guards are exposed as
    hyperoptable enables so Explorer can decide whether higher-timeframe value
    context improves this entry idea.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 1200
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

    vp_window = tagged_parameter(IntParameter(24, 168, default=96, space="buy", optimize=True, load=True))
    vp_bins = tagged_parameter(IntParameter(24, 72, default=48, space="buy", optimize=True, load=True))
    vp_value_area_pct = tagged_parameter(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    vp_price_source = tagged_parameter(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=True, load=True))
    vp_smooth_bins = tagged_parameter(IntParameter(1, 6, default=3, space="buy", optimize=True, load=True))
    vp_hvn_threshold = tagged_parameter(DecimalParameter(0.50, 0.90, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    vp_lvn_threshold = tagged_parameter(DecimalParameter(0.10, 0.55, decimals=2, default=0.35, space="buy", optimize=True, load=True))
    vp_pressure_delta_min = tagged_parameter(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=True, load=True))
    vp_node_near_pct = tagged_parameter(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True))
    vp_node_hvn_strength_min = tagged_parameter(DecimalParameter(0.40, 0.95, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    vp_node_lvn_thinness_min = tagged_parameter(DecimalParameter(0.25, 0.95, decimals=2, default=0.55, space="buy", optimize=True, load=True))
    vp_volume_percentile_min = tagged_parameter(DecimalParameter(0.00, 0.90, decimals=2, default=0.55, space="buy", optimize=True, load=True))
    vp_poc_migration_window = tagged_parameter(IntParameter(4, 36, default=12, space="buy", optimize=True, load=True))
    vp_score_window = tagged_parameter(IntParameter(12, 120, default=48, space="buy", optimize=True, load=True))
    vp_fast_traverse_atr_mult = tagged_parameter(DecimalParameter(0.50, 2.80, decimals=2, default=1.20, space="buy", optimize=True, load=True))
    vp_entry_score_margin = tagged_parameter(DecimalParameter(0.00, 0.20, decimals=2, default=0.02, space="buy", optimize=True, load=True))

    use_entry_score_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    entry_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.20, space="buy", optimize=True, load=True))
    use_entry_context_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    entry_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.26, space="buy", optimize=True, load=True))

    use_vp_4h_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    vp_4h_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="score_or_context", space="buy", optimize=True, load=True))
    vp_4h_window = tagged_parameter(IntParameter(12, 96, default=48, space="buy", optimize=True, load=True))
    vp_4h_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True))
    vp_4h_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    vp_4h_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))

    use_vp_1d_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    vp_1d_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="context", space="buy", optimize=True, load=True))
    vp_1d_window = tagged_parameter(IntParameter(10, 84, default=30, space="buy", optimize=True, load=True))
    vp_1d_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True))
    vp_1d_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    vp_1d_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        try:
            return [(pair, "4h") for pair in self.dp.current_whitelist()] + [(pair, "1d") for pair in self.dp.current_whitelist()]
        except Exception:
            return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_volume_profile(
            dataframe,
            window=int(self.vp_window.value),
            bins=int(self.vp_bins.value),
            value_area_pct=float(self.vp_value_area_pct.value),
            price_source=str(self.vp_price_source.value),
            smooth_bins=int(self.vp_smooth_bins.value),
            hvn_threshold=float(self.vp_hvn_threshold.value),
            lvn_threshold=float(self.vp_lvn_threshold.value),
            pressure_delta_min=float(self.vp_pressure_delta_min.value),
            node_near_pct=float(self.vp_node_near_pct.value),
            node_hvn_strength_min=float(self.vp_node_hvn_strength_min.value),
            node_lvn_thinness_min=float(self.vp_node_lvn_thinness_min.value),
            volume_percentile_min=float(self.vp_volume_percentile_min.value),
            poc_migration_window=int(self.vp_poc_migration_window.value),
            score_window=int(self.vp_score_window.value),
            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult.value),
            entry_score_margin=float(self.vp_entry_score_margin.value),
            prefix="vp",
        )
        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))
        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        condition = _bool(dataframe, TRIGGER_COLUMN)
        if bool(self.use_entry_score_guard.value):
            condition &= self._score_guard(dataframe, "vp", SIDE, float(self.entry_score_min.value))
        if bool(self.use_entry_context_guard.value):
            condition &= self._context_guard(dataframe, "vp", SIDE, float(self.entry_context_min.value))
        if bool(self.use_vp_4h_guard.value):
            condition &= self._vp_guard(dataframe, "vp4h", SIDE, str(self.vp_4h_guard_mode.value), float(self.vp_4h_score_min.value), float(self.vp_4h_context_min.value))
        if bool(self.use_vp_1d_guard.value):
            condition &= self._vp_guard(dataframe, "vp1d", SIDE, str(self.vp_1d_guard_mode.value), float(self.vp_1d_score_min.value), float(self.vp_1d_context_min.value))

        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
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
            value_area_pct=float(self.vp_value_area_pct.value),
            price_source=str(self.vp_price_source.value),
            smooth_bins=int(self.vp_smooth_bins.value),
            hvn_threshold=float(self.vp_hvn_threshold.value),
            lvn_threshold=float(self.vp_lvn_threshold.value),
            pressure_delta_min=float(self.vp_pressure_delta_min.value),
            node_near_pct=float(self.vp_node_near_pct.value),
            node_hvn_strength_min=float(self.vp_node_hvn_strength_min.value),
            node_lvn_thinness_min=float(self.vp_node_lvn_thinness_min.value),
            volume_percentile_min=float(self.vp_volume_percentile_min.value),
            poc_migration_window=int(self.vp_poc_migration_window.value),
            score_window=int(self.vp_score_window.value),
            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult.value),
            entry_score_margin=float(self.vp_entry_score_margin.value),
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


apply_explicit_hyperopt_surface(Sieve1VpLvnFastTraverseLong)

