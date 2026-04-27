from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

REGIME_CODE_CRASH = -2
REGIME_CODE_BEAR = -1
REGIME_CODE_CHOP = 0
REGIME_CODE_BULL = 1
REGIME_CODE_UNKNOWN = 9


def _float_col(dataframe: DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column], errors="coerce")


def _bool_col(dataframe: DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype="bool")
    return pd.Series(dataframe[column], index=dataframe.index).astype("boolean").fillna(False).astype(bool)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _confirm_choices(params: dict[str, Any]) -> list[int]:
    raw = params.get("regime_confirm_choices", ())
    values: list[int] = []
    for value in raw:
        try:
            parsed = int(value)
            if parsed > 0:
                values.append(parsed)
        except (TypeError, ValueError):
            continue
    if not values:
        values = [max(1, int(params.get("regime_confirm_bars", 1)))]
    return sorted(set(values))


def add_regime_input_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()
    atr_pct_col = str(params.get("atr_pct_col") or "atr_pct_96")
    atr_pct_ema_col = str(params.get("atr_pct_ema_col") or "atr_pct_ema_96_40")
    volume_ratio_col = str(params.get("volume_ratio_col") or "volume_ratio_10")

    close = _float_col(frame, "close")
    high = _float_col(frame, "high")
    low = _float_col(frame, "low")
    atr_pct = _float_col(frame, atr_pct_col)
    atr_pct_ema = _float_col(frame, atr_pct_ema_col)

    frame["regime_adx"] = _float_col(frame, "adx")
    frame["regime_atr_pct"] = atr_pct
    frame["regime_atr_pct_ema"] = atr_pct_ema
    frame["regime_atr_ratio"] = _ratio(atr_pct, atr_pct_ema).replace([np.inf, -np.inf], np.nan)
    frame["regime_volume_ratio"] = _float_col(frame, volume_ratio_col)
    frame["regime_btc_1d_rsi"] = _float_col(frame, "btc_1d_rsi")

    recent_high = high.rolling(72, min_periods=1).max()
    frame["regime_drawdown_from_recent_high"] = _ratio(recent_high - close, recent_high).clip(lower=0.0)

    range_span = _ratio(high.rolling(24, min_periods=1).max() - low.rolling(24, min_periods=1).min(), close)
    range_peak = range_span.rolling(24, min_periods=1).max()
    frame["regime_range_compression_score"] = (1.0 - _ratio(range_span, range_peak)).clip(lower=0.0, upper=1.0).fillna(0.0)

    btc_bull_stack = _bool_col(frame, "btc_1d_trend_stack_bull")
    btc_bear_stack = _bool_col(frame, "btc_1d_trend_stack_bear")
    pair_bull_stack = _bool_col(frame, "pair_1d_trend_stack_bull")
    pair_bear_stack = _bool_col(frame, "pair_1d_trend_stack_bear")
    btc_breakout = _bool_col(frame, "btc_1d_breakout_confirmed") | _bool_col(frame, "btc_1d_breakout_retest_hold")
    pair_breakout = _bool_col(frame, "pair_1d_breakout_confirmed") | _bool_col(frame, "pair_1d_breakout_retest_hold")
    btc_support = _bool_col(frame, "btc_1d_support_reversal_confirmed") | _bool_col(frame, "btc_1d_support_bounce_confirmed")
    pair_support = _bool_col(frame, "pair_1d_support_reversal_confirmed") | _bool_col(frame, "pair_1d_support_bounce_confirmed")

    frame["regime_btc_1d_trend_score"] = (
        btc_bull_stack.astype("int8")
        + btc_breakout.astype("int8")
        + btc_support.astype("int8")
        - btc_bear_stack.astype("int8")
    ).astype("float64")
    frame["regime_pair_1d_trend_score"] = (
        pair_bull_stack.astype("int8")
        + pair_breakout.astype("int8")
        + pair_support.astype("int8")
        - pair_bear_stack.astype("int8")
    ).astype("float64")

    trend_stack_bull = _bool_col(frame, "trend_stack_bull")
    trend_stack_bear = _bool_col(frame, "trend_stack_bear")
    di_bull = _bool_col(frame, "di_bull")
    di_bear = _bool_col(frame, "di_bear")
    regime_bull_core = trend_stack_bull & di_bull
    regime_bear_core = trend_stack_bear & di_bear
    frame["regime_bull_core"] = regime_bull_core
    frame["regime_bear_core"] = regime_bear_core

    for confirm in _confirm_choices(params):
        window = max(1, int(confirm))
        frame[f"regime_bull_confirm_{window}"] = (regime_bull_core.astype("int8").rolling(window).sum() >= window)
        frame[f"regime_bear_confirm_{window}"] = (regime_bear_core.astype("int8").rolling(window).sum() >= window)

    selected_confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    bull_confirm_count = regime_bull_core.astype("int8").rolling(selected_confirm).sum().fillna(0.0)
    bear_confirm_count = regime_bear_core.astype("int8").rolling(selected_confirm).sum().fillna(0.0)
    frame["regime_confirm_count"] = np.maximum(bull_confirm_count, bear_confirm_count).astype("float64")
    return frame


def add_regime_score_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()
    trend_min = float(params.get("regime_adx_trend_min", 25.0))
    bear_spike = float(params.get("regime_bear_atr_spike", 1.5))
    confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    use_btc_regime = bool(params.get("use_btc_1d_rsi_regime", False))
    use_pair_context = bool(params.get("enable_pair_1d_context", False))
    btc_bull_min = float(params.get("btc_1d_rsi_bull_min", 55.0))
    btc_bear_max = float(params.get("btc_1d_rsi_bear_max", 40.0))

    adx = _float_col(frame, "regime_adx")
    atr_pct = _float_col(frame, "regime_atr_pct")
    atr_pct_ema = _float_col(frame, "regime_atr_pct_ema")
    atr_ratio = _float_col(frame, "regime_atr_ratio")
    btc_rsi = _float_col(frame, "regime_btc_1d_rsi")
    pair_rsi = _float_col(frame, "pair_1d_rsi")
    btc_adx = _float_col(frame, "btc_1d_adx")
    pair_adx = _float_col(frame, "pair_1d_adx")

    bull_confirm = _bool_col(frame, f"regime_bull_confirm_{confirm}")
    bear_confirm = _bool_col(frame, f"regime_bear_confirm_{confirm}")

    btc_bull_stack = _bool_col(frame, "btc_1d_trend_stack_bull")
    btc_bear_stack = _bool_col(frame, "btc_1d_trend_stack_bear")
    pair_bull_stack = _bool_col(frame, "pair_1d_trend_stack_bull")
    pair_bear_stack = _bool_col(frame, "pair_1d_trend_stack_bear")
    btc_breakout = _bool_col(frame, "btc_1d_breakout_confirmed") | _bool_col(frame, "btc_1d_breakout_retest_hold")
    pair_breakout = _bool_col(frame, "pair_1d_breakout_confirmed") | _bool_col(frame, "pair_1d_breakout_retest_hold")
    btc_support = _bool_col(frame, "btc_1d_support_reversal_confirmed") | _bool_col(frame, "btc_1d_support_bounce_confirmed")
    pair_support = _bool_col(frame, "pair_1d_support_reversal_confirmed") | _bool_col(frame, "pair_1d_support_bounce_confirmed")

    pair_bull_context = pair_bull_stack | pair_breakout | pair_support
    pair_bear_context = pair_bear_stack
    false_mask = pd.Series(False, index=frame.index, dtype="bool")

    bull_htf_ok = pd.Series(True, index=frame.index, dtype="bool")
    bear_htf_ok = pd.Series(True, index=frame.index, dtype="bool")
    btc_context_bull_extra = pair_bull_context if use_pair_context else false_mask
    btc_context_bear_extra = pair_bear_context if use_pair_context else false_mask

    if use_btc_regime:
        bull_htf_ok = (
            btc_rsi.notna()
            & (btc_rsi >= btc_bull_min)
            & (btc_bull_stack | btc_breakout | btc_support | btc_context_bull_extra)
        )
        bear_htf_ok = (btc_rsi <= btc_bear_max) | btc_bear_stack | btc_context_bear_extra

    if use_pair_context:
        pair_known = pair_rsi.notna()
        pair_bull_ok = (pair_rsi >= 50.0) | pair_bull_context
        pair_bear_ok = (pair_rsi <= 45.0) & pair_bear_context
        bull_htf_ok = pd.Series(np.where(pair_known, bull_htf_ok & pair_bull_ok, bull_htf_ok), index=frame.index, dtype="bool")
        bear_htf_ok = pd.Series(np.where(pair_known, bear_htf_ok | pair_bear_ok, bear_htf_ok), index=frame.index, dtype="bool")

    bear_htf_ok = bear_htf_ok | (btc_adx.notna() & btc_bear_stack & (btc_adx >= trend_min))
    if use_pair_context:
        bear_htf_ok = bear_htf_ok | (pair_adx.notna() & pair_bear_stack & (pair_adx >= trend_min))

    base_valid = adx.notna() & atr_pct.notna() & atr_pct_ema.notna()
    if use_btc_regime:
        base_valid &= btc_rsi.notna()

    bear_base = (adx >= trend_min) | (atr_pct >= atr_pct_ema * bear_spike)
    bull_flag = bull_confirm & (adx >= trend_min) & bull_htf_ok
    bear_flag = bear_confirm & bear_base & bear_htf_ok

    drawdown = _float_col(frame, "regime_drawdown_from_recent_high").fillna(0.0)
    crash_spike_threshold = max(bear_spike * 1.25, bear_spike + 0.5)
    crash_flag = bear_flag & (atr_ratio >= crash_spike_threshold) & (drawdown >= 0.08)

    bull_score = (
        0.40 * bull_confirm.astype("float64")
        + 0.30 * (adx >= trend_min).astype("float64")
        + 0.30 * bull_htf_ok.astype("float64")
    ).clip(lower=0.0, upper=1.0)
    bear_score = (
        0.35 * bear_confirm.astype("float64")
        + 0.35 * bear_base.astype("float64")
        + 0.30 * bear_htf_ok.astype("float64")
    ).clip(lower=0.0, upper=1.0)
    chop_score = (
        0.50 * (~bull_flag & ~bear_flag).astype("float64")
        + 0.30 * (adx < trend_min).astype("float64")
        + 0.20 * (1.0 - np.clip((atr_ratio.fillna(1.0) - 1.0), 0.0, 1.0))
    ).clip(lower=0.0, upper=1.0)
    crash_score = (
        0.45 * bear_flag.astype("float64")
        + 0.35 * np.clip((atr_ratio.fillna(0.0) - bear_spike) / max(bear_spike, 1.0), 0.0, 1.0)
        + 0.20 * np.clip(drawdown / 0.15, 0.0, 1.0)
    ).clip(lower=0.0, upper=1.0)

    frame["regime_bull_score"] = bull_score.where(base_valid, 0.0)
    frame["regime_bear_score"] = bear_score.where(base_valid, 0.0)
    frame["regime_chop_score"] = chop_score.where(base_valid, 0.0)
    frame["regime_crash_score"] = crash_score.where(base_valid, 0.0)
    frame["regime_crash_pressure"] = frame["regime_crash_score"].clip(lower=0.0, upper=1.0)
    frame["regime_bear_pressure"] = (
        np.maximum(frame["regime_bear_score"], frame["regime_crash_score"])
        + 0.25 * frame["regime_crash_score"]
    ).clip(lower=0.0, upper=1.0)
    frame["regime_bull_pressure"] = (
        frame["regime_bull_score"] * (1.0 - frame["regime_bear_pressure"])
    ).clip(lower=0.0, upper=1.0)
    frame["regime_directional_pressure"] = frame["regime_bull_pressure"] - frame["regime_bear_pressure"]
    frame["regime_bull_flag"] = bull_flag & base_valid
    frame["regime_bear_flag"] = bear_flag & base_valid
    frame["regime_crash_flag"] = crash_flag & base_valid
    frame["regime_core_valid"] = base_valid
    return frame


def add_regime_state_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()
    valid = _bool_col(frame, "regime_core_valid")
    confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    bull_pressure = _float_col(frame, "regime_bull_pressure").fillna(0.0)
    bear_pressure = _float_col(frame, "regime_bear_pressure").fillna(0.0)
    crash_pressure = _float_col(frame, "regime_crash_pressure").fillna(0.0)
    directional_pressure = _float_col(frame, "regime_directional_pressure").fillna(0.0)
    chop_score = _float_col(frame, "regime_chop_score").fillna(0.0)

    crash_candidate = valid & (crash_pressure >= 0.70)
    bear_candidate = valid & (bear_pressure >= 0.45) & (directional_pressure <= -0.20)
    bull_candidate = valid & (bull_pressure >= 0.50) & (directional_pressure >= 0.25)
    chop_candidate = valid & (chop_score >= 0.70) & (bull_pressure < 0.25) & (bear_pressure < 0.25)

    def confirmed(candidate: pd.Series) -> pd.Series:
        return candidate.astype("int8").rolling(confirm).sum().fillna(0.0) >= confirm

    regime_code = pd.Series(REGIME_CODE_UNKNOWN, index=frame.index, dtype="int64")
    regime_code = regime_code.where(~confirmed(chop_candidate), REGIME_CODE_CHOP)
    regime_code = regime_code.where(~confirmed(bull_candidate), REGIME_CODE_BULL)
    regime_code = regime_code.where(~confirmed(bear_candidate), REGIME_CODE_BEAR)
    regime_code = regime_code.where(~confirmed(crash_candidate), REGIME_CODE_CRASH)
    frame["regime_code"] = regime_code

    scores = np.column_stack(
        [
            _float_col(frame, "regime_bull_score").fillna(0.0).to_numpy(),
            _float_col(frame, "regime_bear_score").fillna(0.0).to_numpy(),
            _float_col(frame, "regime_chop_score").fillna(0.0).to_numpy(),
            _float_col(frame, "regime_crash_score").fillna(0.0).to_numpy(),
        ]
    )
    top = scores.max(axis=1)
    second = np.partition(scores, -2, axis=1)[:, -2]
    confidence = np.clip(top - second, 0.0, 1.0)
    confidence_series = pd.Series(confidence, index=frame.index, dtype="float64").where(valid, 0.0)
    frame["regime_confidence"] = confidence_series

    prev_code = regime_code.shift(1).fillna(REGIME_CODE_UNKNOWN).astype("int64")
    frame["regime_changed"] = (regime_code != prev_code) & valid & (prev_code != REGIME_CODE_UNKNOWN)
    return frame


def add_regime_response_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()

    gap_bull = float(params.get("regime_gap_mult_bull", 1.0))
    gap_chop = float(params.get("regime_gap_mult_chop", 1.0))
    gap_bear = float(params.get("regime_gap_mult_bear", 1.0))
    exit_bull = float(params.get("regime_exit_mult_bull", 1.0))
    exit_chop = float(params.get("regime_exit_mult_chop", 1.0))
    exit_bear = float(params.get("regime_exit_mult_bear", 1.0))

    bull_weight = _float_col(frame, "regime_bull_pressure").fillna(0.0).clip(lower=0.0, upper=1.0)
    bear_weight = _float_col(frame, "regime_bear_pressure").fillna(0.0).clip(lower=0.0, upper=1.0)
    weight_sum = bull_weight + bear_weight
    overweight = weight_sum > 1.0
    bull_weight = bull_weight.where(~overweight, bull_weight / weight_sum.replace(0.0, np.nan)).fillna(0.0)
    bear_weight = bear_weight.where(~overweight, bear_weight / weight_sum.replace(0.0, np.nan)).fillna(0.0)
    chop_weight = (1.0 - bull_weight - bear_weight).clip(lower=0.0, upper=1.0)

    gap_active = chop_weight * gap_chop + bull_weight * gap_bull + bear_weight * gap_bear
    exit_active = chop_weight * exit_chop + bull_weight * exit_bull + bear_weight * exit_bear

    stake_active = chop_weight * 1.0 + bull_weight * 1.2 + bear_weight * 0.75
    spacing_active = chop_weight * 1.1 + bull_weight * 0.8 + bear_weight * 1.5
    peel_fraction_active = chop_weight * 1.2 + bull_weight * 0.9 + bear_weight * 1.4

    frame["regime_gap_mult_active"] = gap_active
    frame["regime_exit_mult_active"] = exit_active
    frame["regime_stake_mult_active"] = stake_active
    frame["regime_spacing_mult_active"] = spacing_active
    frame["regime_peel_fraction_mult_active"] = peel_fraction_active
    return frame
