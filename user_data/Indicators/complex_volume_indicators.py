from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class ComplexVolumeConfig:
    """Settings for the standalone complex volume indicator suite.

    This module is designed for Freqtrade strategies and hyperopt. It uses only
    OHLCV data, so delta/CVD outputs are pressure proxies rather than true
    orderflow. The ``*_score_*`` outputs are normalized to 0..1 and summarize
    volume evidence only; strategy code remains responsible for final decisions.

    Score meaning:
    - ``*_score_long`` rises with bullish pressure proxy, CVD trend confirmation,
      spring/stop-run behavior, bullish absorption, anchored VWAP reclaim, and
      volume-confirmed breakouts.
    - ``*_score_short`` mirrors that for bearish pressure, upthrust/stop-run
      behavior, bearish absorption, anchored VWAP rejection, and downside
      breakouts.
    - ``*_score_abs`` is max(long, short). It measures volume-event strength.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.

    Liquidity sweeping:
    The rolling-window sweep columns mark when price takes a recent high/low
    and then closes back inside with sufficient volume/pressure. This is a
    proxy version; pivot-level sweep detection can be tested later once pivot
    score quality is validated.
    """

    short_window: int = 20
    medium_window: int = 48
    long_window: int = 96
    divergence_window: int = 48
    sweep_window: int = 24
    vwap_window: int = 48
    anchor_volume_zscore: float = 2.50
    vwap_band_mult: float = 1.50
    dry_rvol: float = 0.60
    expansion_rvol: float = 1.35
    climax_rvol: float = 2.25
    absorption_zscore: float = 1.00
    low_result_atr: float = 0.45
    wide_result_atr: float = 1.35
    pressure_zscore_min: float = 0.25
    accumulation_close_location: float = 0.35
    absorption_close_location: float = 0.15
    absorption_delta_tolerance: float = 0.25
    spring_close_location_min: float = -0.25
    upthrust_close_location_max: float = 0.25
    sweep_reclaim_close_location: float = 0.20
    stoprun_delta_tolerance: float = 0.25
    exhaustion_close_location: float = 0.50
    score_window: int = 48
    prefix: str = "vol"


def add_complex_volume_indicators(
    dataframe: DataFrame,
    config: ComplexVolumeConfig | None = None,
    *,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    divergence_window: int | None = None,
    sweep_window: int | None = None,
    vwap_window: int | None = None,
    anchor_volume_zscore: float | None = None,
    vwap_band_mult: float | None = None,
    dry_rvol: float | None = None,
    expansion_rvol: float | None = None,
    climax_rvol: float | None = None,
    absorption_zscore: float | None = None,
    low_result_atr: float | None = None,
    wide_result_atr: float | None = None,
    pressure_zscore_min: float | None = None,
    accumulation_close_location: float | None = None,
    absorption_close_location: float | None = None,
    absorption_delta_tolerance: float | None = None,
    spring_close_location_min: float | None = None,
    upthrust_close_location_max: float | None = None,
    sweep_reclaim_close_location: float | None = None,
    stoprun_delta_tolerance: float | None = None,
    exhaustion_close_location: float | None = None,
    score_window: int | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append a complex OHLCV-only volume suite to a strategy DataFrame.

    The output is designed for Freqtrade strategies that want richer volume
    behavior without orderbook or tick data. Delta/CVD columns are therefore a
    proxy: they infer pressure from candle body, close location, and volume.

    The module exposes evidence columns only. It does not create final
    enter/exit/signal columns; the strategy owns all trading decisions.

    Key output groups are pressure/CVD proxy, volume regime, effort-vs-result,
    liquidity sweeps, anchored VWAP behavior, and normalized score columns.
    """

    cfg = _resolve_config(
        config,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        divergence_window=divergence_window,
        sweep_window=sweep_window,
        vwap_window=vwap_window,
        anchor_volume_zscore=anchor_volume_zscore,
        vwap_band_mult=vwap_band_mult,
        dry_rvol=dry_rvol,
        expansion_rvol=expansion_rvol,
        climax_rvol=climax_rvol,
        absorption_zscore=absorption_zscore,
        low_result_atr=low_result_atr,
        wide_result_atr=wide_result_atr,
        pressure_zscore_min=pressure_zscore_min,
        accumulation_close_location=accumulation_close_location,
        absorption_close_location=absorption_close_location,
        absorption_delta_tolerance=absorption_delta_tolerance,
        spring_close_location_min=spring_close_location_min,
        upthrust_close_location_max=upthrust_close_location_max,
        sweep_reclaim_close_location=sweep_reclaim_close_location,
        stoprun_delta_tolerance=stoprun_delta_tolerance,
        exhaustion_close_location=exhaustion_close_location,
        score_window=score_window,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    frame = _add_base_volume_columns(frame, cfg)
    frame = _add_cvd_columns(frame, cfg)
    frame = _add_volume_regime_columns(frame, cfg)
    frame = _add_effort_result_columns(frame, cfg)
    frame = _add_liquidity_sweep_columns(frame, cfg)
    frame = _add_vwap_columns(frame, cfg)
    frame = _add_score_columns(frame, cfg)
    return frame


def _add_base_volume_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    prev_close = close.shift(1)

    candle_range = (high - low).clip(lower=0.0)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(span=cfg.medium_window, min_periods=1, adjust=False).mean()

    body = close - open_
    hlc3 = (high + low + close) / 3.0
    close_location = ((_safe_div(close - low, candle_range) * 2.0) - 1.0).clip(
        lower=-1.0,
        upper=1.0,
    )
    body_pressure = _safe_div(body, candle_range).clip(lower=-1.0, upper=1.0)

    # With only OHLCV, "delta" must be inferred. Combining body direction with
    # close location avoids treating every green candle with a poor close as
    # strong buying, and vice versa.
    delta_pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(
        lower=-1.0,
        upper=1.0,
    )

    frame[f"{p}_hlc3"] = hlc3
    frame[f"{p}_true_range"] = true_range
    frame[f"{p}_atr"] = atr
    frame[f"{p}_range_atr"] = _safe_div(true_range, atr)
    frame[f"{p}_body_atr"] = _safe_div(body.abs(), atr)
    frame[f"{p}_close_location"] = close_location
    frame[f"{p}_delta_pressure"] = delta_pressure
    return frame


def _add_cvd_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    close = _num(frame, "close")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    delta_pressure = _num(frame, f"{p}_delta_pressure").fillna(0.0)
    delta = volume * delta_pressure
    cvd = delta.cumsum()
    fast = cvd.ewm(span=cfg.short_window, min_periods=1, adjust=False).mean()
    slow = cvd.ewm(span=cfg.medium_window, min_periods=1, adjust=False).mean()
    delta_zscore = _zscore(delta, cfg.long_window)

    prior_price_low = close.rolling(cfg.divergence_window, min_periods=2).min().shift(1)
    prior_price_high = close.rolling(cfg.divergence_window, min_periods=2).max().shift(1)
    prior_cvd_low = cvd.rolling(cfg.divergence_window, min_periods=2).min().shift(1)
    prior_cvd_high = cvd.rolling(cfg.divergence_window, min_periods=2).max().shift(1)

    bull_pressure = delta_zscore >= cfg.pressure_zscore_min
    bear_pressure = delta_zscore <= -cfg.pressure_zscore_min

    frame[f"{p}_delta"] = delta
    frame[f"{p}_delta_zscore"] = delta_zscore
    frame[f"{p}_cvd"] = cvd
    frame[f"{p}_cvd_ema_fast"] = fast
    frame[f"{p}_cvd_ema_slow"] = slow
    frame[f"{p}_cvd_trend"] = fast - slow
    frame[f"{p}_cvd_bull_divergence"] = close.lt(prior_price_low) & cvd.gt(prior_cvd_low)
    frame[f"{p}_cvd_bear_divergence"] = close.gt(prior_price_high) & cvd.lt(prior_cvd_high)
    frame[f"{p}_cvd_trend_confirm_long"] = fast.gt(slow) & bull_pressure
    frame[f"{p}_cvd_trend_confirm_short"] = fast.lt(slow) & bear_pressure
    return frame


def _add_volume_regime_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    open_ = _num(frame, "open")
    close = _num(frame, "close")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    volume_mean = volume.rolling(cfg.long_window, min_periods=1).mean()
    rvol = _safe_div(volume, volume_mean)
    volume_zscore = _zscore(volume, cfg.long_window)
    close_location = _num(frame, f"{p}_close_location")

    dry = rvol <= cfg.dry_rvol
    expansion = rvol >= cfg.expansion_rvol
    climax = rvol >= cfg.climax_rvol
    capitulation = climax & close.lt(open_) & close_location.lt(-float(cfg.accumulation_close_location))
    accumulation = climax & close.gt(open_) & close_location.gt(float(cfg.accumulation_close_location))

    regime = np.select(
        [capitulation, dry, climax, expansion],
        [-2.0, -1.0, 2.0, 1.0],
        default=0.0,
    )

    frame[f"{p}_volume_mean"] = volume_mean
    frame[f"{p}_rvol"] = rvol
    frame[f"{p}_volume_zscore"] = volume_zscore
    frame[f"{p}_regime_code"] = regime
    frame[f"{p}_regime_dry"] = dry
    frame[f"{p}_regime_expansion"] = expansion
    frame[f"{p}_regime_climax"] = climax
    frame[f"{p}_regime_capitulation"] = capitulation
    frame[f"{p}_regime_accumulation"] = accumulation
    return frame


def _add_effort_result_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    prior_high = high.rolling(cfg.sweep_window, min_periods=2).max().shift(1)
    prior_low = low.rolling(cfg.sweep_window, min_periods=2).min().shift(1)
    volume_zscore = _num(frame, f"{p}_volume_zscore")
    rvol = _num(frame, f"{p}_rvol")
    range_atr = _num(frame, f"{p}_range_atr")
    body_atr = _num(frame, f"{p}_body_atr")
    close_location = _num(frame, f"{p}_close_location")
    delta_zscore = _num(frame, f"{p}_delta_zscore")

    high_effort = (volume_zscore >= cfg.absorption_zscore) | (rvol >= cfg.expansion_rvol)
    low_result = body_atr <= cfg.low_result_atr
    wide_result = range_atr >= cfg.wide_result_atr

    # Effort-vs-result separates "lots of volume went nowhere" from "lots of
    # volume moved price." That distinction is useful for absorption and
    # exhaustion filters.
    bull_absorption = high_effort & low_result & close_location.gt(float(cfg.absorption_close_location)) & delta_zscore.gt(-float(cfg.absorption_delta_tolerance))
    bear_absorption = high_effort & low_result & close_location.lt(-float(cfg.absorption_close_location)) & delta_zscore.lt(float(cfg.absorption_delta_tolerance))
    spring = low.lt(prior_low) & close.gt(prior_low) & high_effort & close_location.gt(float(cfg.spring_close_location_min))
    upthrust = high.gt(prior_high) & close.lt(prior_high) & high_effort & close_location.lt(float(cfg.upthrust_close_location_max))
    exhaustion_up = _bool(frame, f"{p}_regime_climax") & wide_result & close_location.gt(float(cfg.exhaustion_close_location))
    exhaustion_down = _bool(frame, f"{p}_regime_climax") & wide_result & close_location.lt(-float(cfg.exhaustion_close_location))

    frame[f"{p}_effort_high"] = high_effort
    frame[f"{p}_result_low"] = low_result
    frame[f"{p}_result_wide"] = wide_result
    frame[f"{p}_evr_bull_absorption"] = bull_absorption
    frame[f"{p}_evr_bear_absorption"] = bear_absorption
    frame[f"{p}_evr_spring"] = spring
    frame[f"{p}_evr_upthrust"] = upthrust
    frame[f"{p}_evr_exhaustion_up"] = exhaustion_up
    frame[f"{p}_evr_exhaustion_down"] = exhaustion_down
    return frame


def _add_liquidity_sweep_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    delta_zscore = _num(frame, f"{p}_delta_zscore")
    close_location = _num(frame, f"{p}_close_location")
    prior_high = high.rolling(cfg.sweep_window, min_periods=2).max().shift(1)
    prior_low = low.rolling(cfg.sweep_window, min_periods=2).min().shift(1)
    volume_ok = _bool(frame, f"{p}_regime_expansion") | _bool(frame, f"{p}_regime_climax")

    sweep_low = low.lt(prior_low)
    sweep_high = high.gt(prior_high)
    low_reclaim = sweep_low & close.gt(prior_low) & close_location.gt(-float(cfg.sweep_reclaim_close_location))
    high_reject = sweep_high & close.lt(prior_high) & close_location.lt(float(cfg.sweep_reclaim_close_location))
    stoprun_long = low_reclaim & volume_ok & delta_zscore.gt(-float(cfg.stoprun_delta_tolerance))
    stoprun_short = high_reject & volume_ok & delta_zscore.lt(float(cfg.stoprun_delta_tolerance))

    breakout_long = close.gt(prior_high) & volume_ok & delta_zscore.gt(cfg.pressure_zscore_min)
    breakout_short = close.lt(prior_low) & volume_ok & delta_zscore.lt(-cfg.pressure_zscore_min)

    frame[f"{p}_prior_sweep_high"] = prior_high
    frame[f"{p}_prior_sweep_low"] = prior_low
    frame[f"{p}_liq_sweep_low"] = sweep_low
    frame[f"{p}_liq_sweep_high"] = sweep_high
    frame[f"{p}_liq_sweep_low_reclaim_long"] = low_reclaim & volume_ok
    frame[f"{p}_liq_sweep_high_reject_short"] = high_reject & volume_ok
    frame[f"{p}_liq_stoprun_long"] = stoprun_long
    frame[f"{p}_liq_stoprun_short"] = stoprun_short
    frame[f"{p}_vol_breakout_confirm_long"] = breakout_long
    frame[f"{p}_vol_breakout_confirm_short"] = breakout_short
    frame[f"{p}_vol_fakeout_risk_long"] = high_reject & volume_ok
    frame[f"{p}_vol_fakeout_risk_short"] = low_reclaim & volume_ok
    return frame


def _add_vwap_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    tp = _num(frame, f"{p}_hlc3")
    pv = tp * volume
    pv2 = tp * tp * volume

    rolling_volume = volume.rolling(cfg.vwap_window, min_periods=1).sum()
    rolling_vwap = _safe_div(pv.rolling(cfg.vwap_window, min_periods=1).sum(), rolling_volume)
    rolling_second = _safe_div(pv2.rolling(cfg.vwap_window, min_periods=1).sum(), rolling_volume)
    rolling_std = np.sqrt((rolling_second - rolling_vwap * rolling_vwap).clip(lower=0.0))

    anchor_event = (
        _num(frame, f"{p}_volume_zscore").ge(cfg.anchor_volume_zscore)
        | _bool(frame, f"{p}_liq_stoprun_long")
        | _bool(frame, f"{p}_liq_stoprun_short")
    )
    if len(anchor_event) > 0:
        anchor_event = anchor_event.copy()
        anchor_event.iloc[0] = True
    anchor_id = anchor_event.astype("int64").cumsum()
    anchored_volume = volume.groupby(anchor_id).cumsum()
    anchored_vwap = _safe_div(pv.groupby(anchor_id).cumsum(), anchored_volume)
    anchored_second = _safe_div(pv2.groupby(anchor_id).cumsum(), anchored_volume)
    anchored_std = np.sqrt((anchored_second - anchored_vwap * anchored_vwap).clip(lower=0.0))
    upper = anchored_vwap + (anchored_std * cfg.vwap_band_mult)
    lower = anchored_vwap - (anchored_std * cfg.vwap_band_mult)

    prev_close = close.shift(1)
    prev_avwap = anchored_vwap.shift(1)
    cross_above = close.gt(anchored_vwap) & prev_close.le(prev_avwap)
    cross_below = close.lt(anchored_vwap) & prev_close.ge(prev_avwap)
    bull_pressure = _num(frame, f"{p}_delta_zscore").ge(cfg.pressure_zscore_min)
    bear_pressure = _num(frame, f"{p}_delta_zscore").le(-cfg.pressure_zscore_min)

    # The anchored VWAP resets at unusually important volume events. This gives
    # a lightweight "market memory" line without hand-picking sessions.
    frame[f"{p}_rvwap"] = rolling_vwap
    frame[f"{p}_rvwap_upper"] = rolling_vwap + (rolling_std * cfg.vwap_band_mult)
    frame[f"{p}_rvwap_lower"] = rolling_vwap - (rolling_std * cfg.vwap_band_mult)
    frame[f"{p}_anchor_event"] = anchor_event
    frame[f"{p}_anchor_id"] = anchor_id.astype("float64")
    frame[f"{p}_avwap"] = anchored_vwap
    frame[f"{p}_avwap_upper"] = upper
    frame[f"{p}_avwap_lower"] = lower
    frame[f"{p}_avwap_reclaim_long"] = cross_above & bull_pressure
    frame[f"{p}_avwap_reject_short"] = cross_below & bear_pressure
    frame[f"{p}_avwap_upper_extension"] = high.ge(upper)
    frame[f"{p}_avwap_lower_extension"] = low.le(lower)
    frame[f"{p}_avwap_mean_reversion_long"] = low.le(lower) & close.gt(lower) & bull_pressure
    frame[f"{p}_avwap_mean_reversion_short"] = high.ge(upper) & close.lt(upper) & bear_pressure
    return frame


def _add_score_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    w = int(cfg.score_window)
    delta_z = _num(frame, f"{p}_delta_zscore").fillna(0.0)
    volume_z = _num(frame, f"{p}_volume_zscore").fillna(0.0)
    pressure_long = _clip01(delta_z / max(float(cfg.pressure_zscore_min) * 4.0, 1e-9))
    pressure_short = _clip01(-delta_z / max(float(cfg.pressure_zscore_min) * 4.0, 1e-9))
    volume_quality = _clip01(volume_z / max(float(cfg.anchor_volume_zscore), 1e-9))

    long_events = (
        1.00 * _bool(frame, f"{p}_cvd_trend_confirm_long").astype("float64")
        + 1.00 * _bool(frame, f"{p}_evr_bull_absorption").astype("float64")
        + 1.25 * _bool(frame, f"{p}_evr_spring").astype("float64")
        + 1.25 * _bool(frame, f"{p}_liq_stoprun_long").astype("float64")
        + 1.00 * _bool(frame, f"{p}_avwap_reclaim_long").astype("float64")
        + 0.75 * _bool(frame, f"{p}_avwap_mean_reversion_long").astype("float64")
        + 0.75 * _bool(frame, f"{p}_vol_breakout_confirm_long").astype("float64")
    )
    short_events = (
        1.00 * _bool(frame, f"{p}_cvd_trend_confirm_short").astype("float64")
        + 1.00 * _bool(frame, f"{p}_evr_bear_absorption").astype("float64")
        + 1.25 * _bool(frame, f"{p}_evr_upthrust").astype("float64")
        + 1.25 * _bool(frame, f"{p}_liq_stoprun_short").astype("float64")
        + 1.00 * _bool(frame, f"{p}_avwap_reject_short").astype("float64")
        + 0.75 * _bool(frame, f"{p}_avwap_mean_reversion_short").astype("float64")
        + 0.75 * _bool(frame, f"{p}_vol_breakout_confirm_short").astype("float64")
    )
    long_recent = _clip01(long_events.rolling(w, min_periods=1).sum() / 4.0)
    short_recent = _clip01(short_events.rolling(w, min_periods=1).sum() / 4.0)
    fakeout_long_penalty = 0.20 * _bool(frame, f"{p}_vol_fakeout_risk_long").astype("float64")
    fakeout_short_penalty = 0.20 * _bool(frame, f"{p}_vol_fakeout_risk_short").astype("float64")

    long_score = _clip01(0.45 * long_recent + 0.35 * pressure_long + 0.20 * volume_quality - fakeout_long_penalty)
    short_score = _clip01(0.45 * short_recent + 0.35 * pressure_short + 0.20 * volume_quality - fakeout_short_penalty)
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )

    frame[f"{p}_score_long"] = long_score
    frame[f"{p}_score_short"] = short_score
    frame[f"{p}_score_abs"] = abs_score
    frame[f"{p}_state"] = score_state
    return frame


def _resolve_config(config: ComplexVolumeConfig | None, **overrides: object) -> ComplexVolumeConfig:
    cfg = config or ComplexVolumeConfig()
    values = {
        "short_window": cfg.short_window,
        "medium_window": cfg.medium_window,
        "long_window": cfg.long_window,
        "divergence_window": cfg.divergence_window,
        "sweep_window": cfg.sweep_window,
        "vwap_window": cfg.vwap_window,
        "anchor_volume_zscore": cfg.anchor_volume_zscore,
        "vwap_band_mult": cfg.vwap_band_mult,
        "dry_rvol": cfg.dry_rvol,
        "expansion_rvol": cfg.expansion_rvol,
        "climax_rvol": cfg.climax_rvol,
        "absorption_zscore": cfg.absorption_zscore,
        "low_result_atr": cfg.low_result_atr,
        "wide_result_atr": cfg.wide_result_atr,
        "pressure_zscore_min": cfg.pressure_zscore_min,
        "accumulation_close_location": cfg.accumulation_close_location,
        "absorption_close_location": cfg.absorption_close_location,
        "absorption_delta_tolerance": cfg.absorption_delta_tolerance,
        "spring_close_location_min": cfg.spring_close_location_min,
        "upthrust_close_location_max": cfg.upthrust_close_location_max,
        "sweep_reclaim_close_location": cfg.sweep_reclaim_close_location,
        "stoprun_delta_tolerance": cfg.stoprun_delta_tolerance,
        "exhaustion_close_location": cfg.exhaustion_close_location,
        "score_window": cfg.score_window,
        "prefix": cfg.prefix,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return ComplexVolumeConfig(**values)


def _validate_config(cfg: ComplexVolumeConfig) -> None:
    window_values = [
        cfg.short_window,
        cfg.medium_window,
        cfg.long_window,
        cfg.divergence_window,
        cfg.sweep_window,
        cfg.vwap_window,
    ]
    if any(value < 2 for value in window_values):
        raise ValueError("all volume indicator windows must be at least 2")
    if cfg.vwap_band_mult <= 0.0:
        raise ValueError("vwap_band_mult must be positive")
    if cfg.dry_rvol <= 0.0 or cfg.expansion_rvol <= 0.0 or cfg.climax_rvol <= 0.0:
        raise ValueError("relative volume thresholds must be positive")
    if cfg.dry_rvol >= cfg.expansion_rvol or cfg.expansion_rvol >= cfg.climax_rvol:
        raise ValueError("expected dry_rvol < expansion_rvol < climax_rvol")
    if cfg.low_result_atr <= 0.0 or cfg.wide_result_atr <= 0.0:
        raise ValueError("effort/result ATR thresholds must be positive")
    if cfg.score_window < 2:
        raise ValueError("score_window must be at least 2")
    bounded = [
        cfg.accumulation_close_location,
        cfg.absorption_close_location,
        cfg.absorption_delta_tolerance,
        cfg.sweep_reclaim_close_location,
        cfg.stoprun_delta_tolerance,
        cfg.exhaustion_close_location,
    ]
    if any(value < 0.0 or value > 1.0 for value in bounded):
        raise ValueError("close-location and delta tolerance thresholds must be between 0.0 and 1.0")
    if not -1.0 <= cfg.spring_close_location_min <= 1.0:
        raise ValueError("spring_close_location_min must be between -1.0 and 1.0")
    if not -1.0 <= cfg.upthrust_close_location_max <= 1.0:
        raise ValueError("upthrust_close_location_max must be between -1.0 and 1.0")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _zscore(series: Series, window: int) -> Series:
    mean = series.rolling(window, min_periods=2).mean()
    std = series.rolling(window, min_periods=2).std(ddof=0)
    return _safe_div(series - mean, std)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)
