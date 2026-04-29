from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

REGIME_CODE_CRASH = -2
REGIME_CODE_BEAR = -1
REGIME_CODE_CHOP = 0
REGIME_CODE_BULL = 1
REGIME_CODE_UNKNOWN = 9


def add_regime_input_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    """Append reusable OHLCV market-regime inputs.

    This module intentionally avoids strategy-specific BTC/pair context,
    position-adjustment, and stake logic. It emits evidence about the traded
    market itself: EMA trend stack, ADX/DMI trend quality, ATR expansion,
    drawdown from recent highs, range compression, and relative volume.
    """

    _validate_ohlcv(dataframe)
    frame = dataframe.copy()
    close = _float_col(frame, "close").replace(0.0, np.nan)
    high = _float_col(frame, "high")
    low = _float_col(frame, "low")
    volume = _float_col(frame, "volume").clip(lower=0.0).fillna(0.0)

    atr_period = _positive_int(params.get("regime_atr_period", 14), 14)
    atr_ema_window = _positive_int(params.get("regime_atr_ema_window", 96), 96)
    adx_period = _positive_int(params.get("regime_adx_period", 14), 14)
    ema_fast_window = _positive_int(params.get("regime_ema_fast", 48), 48)
    ema_slow_window = _positive_int(params.get("regime_ema_slow", 144), 144)
    ema_long_window = _positive_int(params.get("regime_ema_long", 480), 480)
    volume_window = _positive_int(params.get("regime_volume_window", 96), 96)
    range_window = _positive_int(params.get("regime_range_window", 72), 72)
    slope_window = _positive_int(params.get("regime_slope_window", 48), 48)
    drawdown_window = _positive_int(params.get("regime_drawdown_window", 240), 240)

    atr = _true_range(frame).ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()
    atr_pct = _ratio(atr, close)
    atr_pct_ema = atr_pct.ewm(span=atr_ema_window, min_periods=max(2, atr_ema_window // 4), adjust=False).mean()
    dmi = _dmi(frame, adx_period)

    ema_fast = close.ewm(span=ema_fast_window, min_periods=ema_fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_window, min_periods=ema_slow_window, adjust=False).mean()
    ema_long = close.ewm(span=ema_long_window, min_periods=max(2, ema_long_window // 4), adjust=False).mean()
    trend_slope_pct = ema_slow.pct_change(slope_window)
    price_location_atr = _ratio(close - ema_slow, atr)

    volume_mean = volume.rolling(volume_window, min_periods=max(2, volume_window // 4)).mean()
    recent_high = high.rolling(drawdown_window, min_periods=max(2, drawdown_window // 4)).max()
    range_span = _ratio(
        high.rolling(range_window, min_periods=max(2, range_window // 4)).max()
        - low.rolling(range_window, min_periods=max(2, range_window // 4)).min(),
        close,
    )
    range_peak = range_span.rolling(range_window, min_periods=max(2, range_window // 4)).max()

    bull_core = close.gt(ema_fast) & ema_fast.gt(ema_slow) & ema_slow.gt(ema_long) & dmi["plus_di"].gt(dmi["minus_di"])
    bear_core = close.lt(ema_fast) & ema_fast.lt(ema_slow) & ema_slow.lt(ema_long) & dmi["minus_di"].gt(dmi["plus_di"])

    frame["regime_atr"] = atr
    frame["regime_atr_pct"] = atr_pct
    frame["regime_atr_pct_ema"] = atr_pct_ema
    frame["regime_atr_ratio"] = _ratio(atr_pct, atr_pct_ema).replace([np.inf, -np.inf], np.nan)
    frame["regime_adx"] = dmi["adx"]
    frame["regime_plus_di"] = dmi["plus_di"]
    frame["regime_minus_di"] = dmi["minus_di"]
    frame["regime_ema_fast"] = ema_fast
    frame["regime_ema_slow"] = ema_slow
    frame["regime_ema_long"] = ema_long
    frame["regime_trend_slope_pct"] = trend_slope_pct
    frame["regime_price_location_atr"] = price_location_atr
    frame["regime_volume_ratio"] = _ratio(volume, volume_mean)
    frame["regime_drawdown_from_recent_high"] = _ratio(recent_high - close, recent_high).clip(lower=0.0)
    frame["regime_range_compression_score"] = (1.0 - _ratio(range_span, range_peak)).clip(lower=0.0, upper=1.0).fillna(0.0)
    frame["regime_bull_core"] = bull_core.fillna(False)
    frame["regime_bear_core"] = bear_core.fillna(False)

    for confirm in _confirm_choices(params):
        window = max(1, int(confirm))
        frame[f"regime_bull_confirm_{window}"] = bull_core.astype("int8").rolling(window, min_periods=window).sum().ge(window)
        frame[f"regime_bear_confirm_{window}"] = bear_core.astype("int8").rolling(window, min_periods=window).sum().ge(window)

    selected_confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    bull_confirm_count = bull_core.astype("int8").rolling(selected_confirm, min_periods=1).sum().fillna(0.0)
    bear_confirm_count = bear_core.astype("int8").rolling(selected_confirm, min_periods=1).sum().fillna(0.0)
    frame["regime_confirm_count"] = np.maximum(bull_confirm_count, bear_confirm_count).astype("float64")
    return frame


def add_regime_score_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()
    trend_min = float(params.get("regime_adx_trend_min", 22.0))
    bear_spike = float(params.get("regime_bear_atr_spike", 1.5))
    confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    slope_scale = max(float(params.get("regime_slope_scale", 0.035)), 1e-9)
    volume_confirm_min = max(float(params.get("regime_volume_confirm_min", 1.05)), 1e-9)

    adx = _float_col(frame, "regime_adx")
    atr_ratio = _float_col(frame, "regime_atr_ratio")
    volume_ratio = _float_col(frame, "regime_volume_ratio")
    slope = _float_col(frame, "regime_trend_slope_pct")
    price_location = _float_col(frame, "regime_price_location_atr")
    drawdown = _float_col(frame, "regime_drawdown_from_recent_high").fillna(0.0)
    compression = _float_col(frame, "regime_range_compression_score").fillna(0.0)

    bull_confirm = _bool_col(frame, f"regime_bull_confirm_{confirm}") | _bool_col(frame, "regime_bull_core")
    bear_confirm = _bool_col(frame, f"regime_bear_confirm_{confirm}") | _bool_col(frame, "regime_bear_core")
    trend_strength = _clip01((adx - trend_min * 0.55) / max(trend_min * 0.75, 1e-9))
    volume_confirm = _clip01((volume_ratio - 0.75) / max(volume_confirm_min - 0.75, 1e-9))
    slope_up = _clip01(slope / slope_scale)
    slope_down = _clip01(-slope / slope_scale)
    price_above = _clip01(price_location / 2.0)
    price_below = _clip01(-price_location / 2.0)
    volatility_expansion = _clip01((atr_ratio - 1.0) / max(bear_spike - 1.0, 1e-9))
    drawdown_pressure = _clip01(drawdown / 0.15)

    base_valid = adx.notna() & atr_ratio.notna() & slope.notna()
    bull_score = _clip01(
        0.30 * bull_confirm.astype("float64")
        + 0.25 * trend_strength
        + 0.20 * slope_up
        + 0.15 * price_above
        + 0.10 * volume_confirm
    ).where(base_valid, 0.0)
    bear_score = _clip01(
        0.30 * bear_confirm.astype("float64")
        + 0.25 * trend_strength
        + 0.20 * slope_down
        + 0.15 * price_below
        + 0.10 * drawdown_pressure
    ).where(base_valid, 0.0)
    chop_score = _clip01(
        0.35 * (adx.lt(trend_min * 0.75)).astype("float64")
        + 0.30 * compression
        + 0.20 * atr_ratio.lt(1.10).astype("float64")
        + 0.15 * (~bull_confirm & ~bear_confirm).astype("float64")
    ).where(base_valid, 0.0)
    crash_score = _clip01(
        0.40 * bear_score
        + 0.35 * volatility_expansion
        + 0.25 * drawdown_pressure
    ).where(base_valid, 0.0)

    frame["regime_bull_score"] = bull_score
    frame["regime_bear_score"] = bear_score
    frame["regime_chop_score"] = chop_score
    frame["regime_crash_score"] = crash_score
    frame["regime_crash_pressure"] = crash_score
    frame["regime_bear_pressure"] = np.maximum(bear_score, crash_score).clip(0.0, 1.0)
    frame["regime_bull_pressure"] = (bull_score * (1.0 - frame["regime_bear_pressure"])).clip(0.0, 1.0)
    frame["regime_directional_pressure"] = frame["regime_bull_pressure"] - frame["regime_bear_pressure"]
    frame["regime_bull_flag"] = bull_score.ge(0.55) & bull_score.gt(bear_score) & base_valid
    frame["regime_bear_flag"] = bear_score.ge(0.55) & bear_score.gt(bull_score) & base_valid
    frame["regime_crash_flag"] = crash_score.ge(0.70) & drawdown.ge(0.06) & base_valid
    frame["regime_core_valid"] = base_valid
    return frame


def add_regime_state_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    frame = dataframe.copy()
    valid = _bool_col(frame, "regime_core_valid")
    confirm = max(1, int(params.get("regime_confirm_bars", 1)))
    cooldown = max(2, int(params.get("regime_event_cooldown_bars", 8)))
    early_bear_min = float(params.get("regime_early_bear_min", 0.42))
    early_crash_min = float(params.get("regime_early_crash_min", 0.58))
    bull_recovery_min = float(params.get("regime_bull_recovery_min", 0.48))
    bull_pressure = _float_col(frame, "regime_bull_pressure").fillna(0.0)
    bear_pressure = _float_col(frame, "regime_bear_pressure").fillna(0.0)
    crash_pressure = _float_col(frame, "regime_crash_pressure").fillna(0.0)
    directional_pressure = _float_col(frame, "regime_directional_pressure").fillna(0.0)
    chop_score = _float_col(frame, "regime_chop_score").fillna(0.0)
    atr_ratio = _float_col(frame, "regime_atr_ratio").fillna(1.0)
    drawdown = _float_col(frame, "regime_drawdown_from_recent_high").fillna(0.0)
    slope = _float_col(frame, "regime_trend_slope_pct").fillna(0.0)
    price_location = _float_col(frame, "regime_price_location_atr").fillna(0.0)

    early_bear_warning = valid & (
        (
            bear_pressure.ge(early_bear_min)
            & directional_pressure.le(-0.08)
            & (price_location.lt(-0.25) | slope.lt(0.0) | drawdown.ge(0.04))
        )
        | (drawdown.ge(0.07) & price_location.lt(0.0) & atr_ratio.ge(1.05))
    )
    crash_warning = valid & (
        crash_pressure.ge(early_crash_min)
        | (drawdown.ge(0.10) & atr_ratio.ge(1.20) & price_location.lt(-0.50))
    )
    bull_recovery = valid & (
        bull_pressure.ge(bull_recovery_min)
        & directional_pressure.ge(0.08)
        & slope.ge(0.0)
        & price_location.gt(0.0)
    )

    crash_candidate = valid & _confirmed(crash_warning, confirm)
    bear_candidate = valid & _confirmed(early_bear_warning, confirm)
    bull_candidate = valid & _confirmed(bull_recovery, confirm)
    chop_candidate = valid & chop_score.ge(0.62) & bull_pressure.lt(0.40) & bear_pressure.lt(0.40)

    regime_code = pd.Series(REGIME_CODE_UNKNOWN, index=frame.index, dtype="int64")
    regime_code = regime_code.where(~valid, REGIME_CODE_CHOP)
    regime_code = regime_code.where(~chop_candidate, REGIME_CODE_CHOP)
    regime_code = regime_code.where(~bull_candidate, REGIME_CODE_BULL)
    regime_code = regime_code.where(~bear_candidate, REGIME_CODE_BEAR)
    regime_code = regime_code.where(~crash_candidate, REGIME_CODE_CRASH)
    frame["regime_code"] = regime_code

    bullish_chop = valid & regime_code.eq(REGIME_CODE_CHOP) & bull_pressure.gt(bear_pressure + 0.08)
    bearish_chop = valid & regime_code.eq(REGIME_CODE_CHOP) & (
        bear_pressure.gt(bull_pressure + 0.08) | early_bear_warning
    )
    market_context = pd.Series(
        np.select(
            [
                regime_code.eq(REGIME_CODE_BULL),
                bullish_chop,
                regime_code.isin([REGIME_CODE_BEAR, REGIME_CODE_CRASH]),
                bearish_chop,
            ],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    ).where(valid, 0.0)
    frame["regime_market_context"] = market_context
    frame["regime_bear_warning"] = early_bear_warning
    frame["regime_crash_warning"] = crash_warning
    frame["regime_bull_recovery"] = bull_recovery
    frame["regime_entry_risk_on_long"] = _dedupe_events(bull_recovery & market_context.gt(0.0), cooldown)
    frame["regime_entry_risk_off_short"] = _dedupe_events(early_bear_warning | crash_warning, cooldown)
    frame["regime_suggested_entry_long"] = frame["regime_entry_risk_on_long"]
    frame["regime_suggested_entry_short"] = frame["regime_entry_risk_off_short"]
    frame["regime_hold_long"] = valid & market_context.gt(0.0) & ~early_bear_warning & ~crash_warning
    frame["regime_hold_short"] = valid & market_context.lt(0.0)
    frame["regime_exit_long"] = _dedupe_events(early_bear_warning | crash_warning | regime_code.isin([REGIME_CODE_BEAR, REGIME_CODE_CRASH]), cooldown)
    frame["regime_exit_short"] = _dedupe_events(bull_recovery | regime_code.eq(REGIME_CODE_BULL), cooldown)
    frame["regime_score_long"] = bull_pressure
    frame["regime_score_short"] = bear_pressure
    frame["regime_score_abs"] = pd.concat([bull_pressure, bear_pressure, crash_pressure], axis=1).max(axis=1)
    frame["regime_state"] = pd.Series(
        np.select([market_context.gt(0.0), market_context.lt(0.0)], [1.0, -1.0], default=0.0),
        index=frame.index,
        dtype="float64",
    )

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
    frame["regime_confidence"] = pd.Series(np.clip(top - second, 0.0, 1.0), index=frame.index).where(valid, 0.0)
    prev_code = regime_code.shift(1).fillna(REGIME_CODE_UNKNOWN).astype("int64")
    frame["regime_changed"] = regime_code.ne(prev_code) & valid & prev_code.ne(REGIME_CODE_UNKNOWN)
    return frame


def add_regime_response_columns(dataframe: DataFrame, params: dict[str, Any]) -> DataFrame:
    """Append simple regime response multipliers for legacy callers.

    New strategy research should treat these as optional outputs. The core
    indicator value is in the input, score, and state columns above.
    """

    frame = dataframe.copy()
    bull = _float_col(frame, "regime_bull_pressure").fillna(0.0).clip(0.0, 1.0)
    bear = _float_col(frame, "regime_bear_pressure").fillna(0.0).clip(0.0, 1.0)
    crash = _float_col(frame, "regime_crash_pressure").fillna(0.0).clip(0.0, 1.0)
    weight_sum = bull + bear
    overweight = weight_sum.gt(1.0)
    bull_weight = bull.where(~overweight, _ratio(bull, weight_sum)).fillna(0.0)
    bear_weight = bear.where(~overweight, _ratio(bear, weight_sum)).fillna(0.0)
    chop = (1.0 - bull_weight - bear_weight).clip(0.0, 1.0)

    frame["regime_gap_mult_active"] = (
        chop * float(params.get("regime_gap_mult_chop", 1.0))
        + bull_weight * float(params.get("regime_gap_mult_bull", 0.85))
        + bear_weight * float(params.get("regime_gap_mult_bear", 1.25))
    )
    frame["regime_exit_mult_active"] = (
        chop * float(params.get("regime_exit_mult_chop", 1.0))
        + bull_weight * float(params.get("regime_exit_mult_bull", 1.15))
        + bear_weight * float(params.get("regime_exit_mult_bear", 0.85))
    )
    frame["regime_risk_mult_active"] = (1.0 + bull_weight * 0.15 - bear_weight * 0.25 - crash * 0.35).clip(0.25, 1.25)
    frame["regime_stake_mult_active"] = (1.0 + bull_weight * 0.20 - bear_weight * 0.25 - crash * 0.40).clip(0.20, 1.25)
    frame["regime_spacing_mult_active"] = (1.0 - bull_weight * 0.15 + bear_weight * 0.35 + crash * 0.50).clip(0.75, 1.75)
    frame["regime_peel_fraction_mult_active"] = (1.0 - bull_weight * 0.10 + bear_weight * 0.30 + crash * 0.45).clip(0.75, 1.75)
    return frame


def add_market_regime(dataframe: DataFrame, params: dict[str, Any] | None = None) -> DataFrame:
    """Convenience wrapper for the full regime indicator pass."""

    resolved = params or {}
    out = add_regime_input_columns(dataframe, resolved)
    out = add_regime_score_columns(out, resolved)
    return add_regime_state_columns(out, resolved)


def _dmi(frame: DataFrame, period: int) -> dict[str, Series]:
    high = _float_col(frame, "high")
    low = _float_col(frame, "low")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index, dtype="float64")
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index, dtype="float64")
    atr = _true_range(frame).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * _ratio(plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean(), atr)
    minus_di = 100.0 * _ratio(minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean(), atr)
    dx = 100.0 * _ratio((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def _true_range(frame: DataFrame) -> Series:
    high = _float_col(frame, "high")
    low = _float_col(frame, "low")
    close = _float_col(frame, "close")
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _confirmed(candidate: Series, window: int) -> Series:
    return candidate.astype("int8").rolling(max(1, int(window)), min_periods=max(1, int(window))).sum().fillna(0.0).ge(window)


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    event = pd.Series(mask, index=mask.index).astype("boolean").fillna(False).astype(bool)
    previous_count = event.astype("float64").shift(1).rolling(max(1, int(cooldown_bars)), min_periods=1).sum().fillna(0.0)
    return event & previous_count.eq(0.0)


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _validate_ohlcv(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _float_col(dataframe: DataFrame, column: str) -> Series:
    if column not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool_col(dataframe: DataFrame, column: str) -> Series:
    if column not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype="bool")
    return pd.Series(dataframe[column], index=dataframe.index).astype("boolean").fillna(False).astype(bool)


def _ratio(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)
