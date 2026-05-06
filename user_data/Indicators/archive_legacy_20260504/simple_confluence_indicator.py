from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
import talib
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame, Series

_TALIB_CDL_PATTERNS = (
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3INSIDE",
    "CDL3LINESTRIKE",
    "CDL3OUTSIDE",
    "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK",
    "CDLBELTHOLD",
    "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU",
    "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDOJI",
    "CDLDOJISTAR",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI",
    "CDLHAMMER",
    "CDLHANGINGMAN",
    "CDLHARAMI",
    "CDLHARAMICROSS",
    "CDLHIGHWAVE",
    "CDLHIKKAKE",
    "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS",
    "CDLINNECK",
    "CDLINVERTEDHAMMER",
    "CDLKICKING",
    "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLMATCHINGLOW",
    "CDLMATHOLD",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLONNECK",
    "CDLPIERCING",
    "CDLRICKSHAWMAN",
    "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN",
    "CDLSTICKSANDWICH",
    "CDLTAKURI",
    "CDLTASUKIGAP",
    "CDLTHRUSTING",
    "CDLTRISTAR",
    "CDLUNIQUE3RIVER",
    "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS",
)

_SIGNAL_WEIGHTS = {
    "stoch_cross_midline": 1.80,
    "stoch_rsi_cross_midline": 1.60,
    "adx_directional_trend": 1.70,
    "bb_reversal": 1.40,
    "cmf_pressure": 1.30,
    "efi_cross_zero": 1.20,
    "atr_expansion_direction": 1.30,
    "ema_price_reclaim_fast": 1.20,
    "cci_cross_extreme": 1.20,
    "stoch_rsi_cross_extreme": 1.20,
    "ultimate_cross_midline": 0.45,
    "adx_di_cross": 1.10,
    "sma_price_reclaim_mid": 1.10,
    "dema_cross": 1.10,
    "macd_cross_signal": 1.10,
    "ppo_cross_signal": 1.10,
    "cmo_cross_zero": 1.00,
    "tema_price_reclaim": 1.00,
    "vwap_reclaim": 0.80,
    "bb_midline_reclaim": 0.90,
    "bb_percent_b_midline": 0.90,
    "rsi_cross_midline": 0.90,
    "engulfing": 0.90,
    "wma_cross": 1.20,
    "keltner_breakout": 1.10,
    "trix_cross_signal": 1.10,
    "aroon_cross": 1.00,
    "cmf_cross_zero": 0.35,
    "ma_cluster": 0.35,
    "ma_slope_consensus": 0.45,
    "rsi_range_momentum": 0.25,
    "vwap_trend": 0.30,
    "mfi_money_flow_bias": 0.30,
    "ichimoku_cloud_position": 0.25,
    "bb_squeeze_breakout": 0.20,
    "ema_pullback_hold": 0.20,
    "donchian_breakout": 0.45,
    "macd_hist_reversal": 0.40,
    "volume_confirmed_breakout": 0.00,
    "volume_climax_reversal": 0.00,
    "obv_breakout": 0.00,
    "inside_bar_break": 0.00,
    "keltner_reversal": 0.25,
    "donchian_failed_break": 0.30,
}

_DIRECTIONAL_VOLUME_WEIGHTS = {
    "cmf_pressure": 1.30,
    "efi_cross_zero": 1.20,
    "vwap_reclaim": 0.80,
    "adl_cross": 0.70,
    "obv_cross": 0.60,
    "cmf_cross_zero": 0.35,
    "vwap_trend": 0.30,
}

_CDL_DEFAULT_WEIGHT = 0.25

# First-pass BTC/SOL futures validation weights. The low default keeps sparse
# TA-Lib candle patterns from dominating broader confluence.
_CDL_SIGNAL_WEIGHTS = {
    "cdl_3inside": (0.60, 0.80),
    "cdl_3outside": (0.05, 0.05),
    "cdl_belthold": (0.65, 0.20),
    "cdl_closingmarubozu": (0.35, 0.20),
    "cdl_doji": (0.40, 0.25),
    "cdl_dragonflydoji": (0.65, 0.25),
    "cdl_engulfing": (0.75, 0.05),
    "cdl_eveningstar": (0.25, 0.05),
    "cdl_hammer": (0.60, 0.25),
    "cdl_harami": (0.50, 0.15),
    "cdl_haramicross": (0.05, 0.25),
    "cdl_highwave": (0.40, 0.30),
    "cdl_hikkake": (0.45, 0.05),
    "cdl_longleggeddoji": (0.40, 0.25),
    "cdl_longline": (0.35, 0.45),
    "cdl_marubozu": (0.45, 0.20),
    "cdl_morningstar": (0.05, 0.25),
    "cdl_separatinglines": (0.70, 0.05),
    "cdl_shortline": (0.45, 0.60),
    "cdl_spinningtop": (0.25, 0.35),
    "cdl_takuri": (0.65, 0.25),
}


@dataclass(frozen=True)
class SimpleConfluenceConfig:
    """Broad standard-TA confluence evidence for Freqtrade research.

    This module intentionally collects common textbook signals rather than
    trying to be a refined edge model. It emits many 0/1 long and short event
    columns so validation can measure whether any simple TA family contributes
    useful confluence across pairs and timeframes.

    Score meaning:
    - ``*_signal_count_long`` and ``*_signal_count_short`` are unbounded counts
      of currently triggered long/short rules.
    - ``*_recent_signal_count_long`` and ``*_recent_signal_count_short`` count
      triggers over the rolling memory window.
    - ``*_weighted_signal_sum_long`` and ``*_weighted_signal_sum_short`` apply
      first-pass usefulness weights while leaving raw counts intact for audit.
    - ``*_score_long`` and ``*_score_short`` normalize weighted evidence to
      0..1.
    - ``*_score_abs`` is max(long, short).
    - ``*_state`` is -1/0/1 directional lean and is not a score.

    The raw counts are deliberately broad. The normalized score is weighted so
    always-on or non-directional rules do not contribute as much as signals that
    showed better early BTC/SOL validation behavior.
    """

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_midline: float = 50.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    bb_squeeze_window: int = 120
    bb_squeeze_quantile: float = 0.20
    ema_fast: int = 12
    ema_mid: int = 26
    ema_slow: int = 50
    ema_long: int = 200
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 100
    sma_long: int = 200
    tema_fast: int = 9
    tema_slow: int = 21
    dema_fast: int = 9
    dema_slow: int = 21
    wma_fast: int = 20
    wma_slow: int = 50
    hma_fast: int = 16
    hma_slow: int = 49
    stoch_period: int = 14
    stoch_smooth: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    stoch_rsi_period: int = 14
    adx_period: int = 14
    adx_trend_min: float = 20.0
    cci_period: int = 20
    cci_oversold: float = -100.0
    cci_overbought: float = 100.0
    mfi_period: int = 14
    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0
    cmo_period: int = 14
    cmo_oversold: float = -50.0
    cmo_overbought: float = 50.0
    roc_period: int = 12
    momentum_period: int = 10
    williams_period: int = 14
    williams_oversold: float = -80.0
    williams_overbought: float = -20.0
    aroon_period: int = 25
    aroon_trend_min: float = 70.0
    trix_period: int = 15
    trix_signal: int = 9
    ultimate_short: int = 7
    ultimate_medium: int = 14
    ultimate_long: int = 28
    ultimate_oversold: float = 30.0
    ultimate_overbought: float = 70.0
    kst_signal: int = 9
    atr_period: int = 14
    atr_expansion_mult: float = 1.25
    keltner_window: int = 20
    keltner_atr_mult: float = 2.0
    donchian_window: int = 20
    zscore_window: int = 50
    zscore_extreme: float = 2.0
    vwap_window: int = 20
    volume_window: int = 20
    volume_breakout_mult: float = 1.50
    cmf_period: int = 20
    obv_fast: int = 10
    obv_slow: int = 30
    adl_fast: int = 10
    adl_slow: int = 30
    efi_period: int = 13
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_span_b: int = 52
    signal_memory_window: int = 3
    score_signal_cap: float = 12.0
    weighted_score_cap: float = 20.0
    weighted_family_score_cap: float = 8.0
    recent_score_weight: float = 0.25
    min_confluence_signals: int = 5
    min_confluence_families: int = 4
    family_score_cap: float = 6.0
    context_window: int = 24
    entry_cooldown_bars: int = 12
    context_full_min: float = 0.70
    context_full_margin: float = 0.18
    context_soft_min: float = 0.55
    context_soft_margin: float = 0.10
    prefix: str = "sci"


def add_simple_confluence_indicator(
    dataframe: DataFrame,
    config: SimpleConfluenceConfig | None = None,
    *,
    rsi_period: int | None = None,
    rsi_oversold: float | None = None,
    rsi_midline: float | None = None,
    rsi_overbought: float | None = None,
    macd_fast: int | None = None,
    macd_slow: int | None = None,
    macd_signal: int | None = None,
    bb_window: int | None = None,
    bb_std: float | None = None,
    bb_squeeze_window: int | None = None,
    bb_squeeze_quantile: float | None = None,
    ema_fast: int | None = None,
    ema_mid: int | None = None,
    ema_slow: int | None = None,
    ema_long: int | None = None,
    sma_fast: int | None = None,
    sma_mid: int | None = None,
    sma_slow: int | None = None,
    sma_long: int | None = None,
    tema_fast: int | None = None,
    tema_slow: int | None = None,
    dema_fast: int | None = None,
    dema_slow: int | None = None,
    wma_fast: int | None = None,
    wma_slow: int | None = None,
    hma_fast: int | None = None,
    hma_slow: int | None = None,
    stoch_period: int | None = None,
    stoch_smooth: int | None = None,
    stoch_oversold: float | None = None,
    stoch_overbought: float | None = None,
    stoch_rsi_period: int | None = None,
    adx_period: int | None = None,
    adx_trend_min: float | None = None,
    cci_period: int | None = None,
    cci_oversold: float | None = None,
    cci_overbought: float | None = None,
    mfi_period: int | None = None,
    mfi_oversold: float | None = None,
    mfi_overbought: float | None = None,
    cmo_period: int | None = None,
    cmo_oversold: float | None = None,
    cmo_overbought: float | None = None,
    roc_period: int | None = None,
    momentum_period: int | None = None,
    williams_period: int | None = None,
    williams_oversold: float | None = None,
    williams_overbought: float | None = None,
    aroon_period: int | None = None,
    aroon_trend_min: float | None = None,
    trix_period: int | None = None,
    trix_signal: int | None = None,
    ultimate_short: int | None = None,
    ultimate_medium: int | None = None,
    ultimate_long: int | None = None,
    ultimate_oversold: float | None = None,
    ultimate_overbought: float | None = None,
    kst_signal: int | None = None,
    atr_period: int | None = None,
    atr_expansion_mult: float | None = None,
    keltner_window: int | None = None,
    keltner_atr_mult: float | None = None,
    donchian_window: int | None = None,
    zscore_window: int | None = None,
    zscore_extreme: float | None = None,
    vwap_window: int | None = None,
    volume_window: int | None = None,
    volume_breakout_mult: float | None = None,
    cmf_period: int | None = None,
    obv_fast: int | None = None,
    obv_slow: int | None = None,
    adl_fast: int | None = None,
    adl_slow: int | None = None,
    efi_period: int | None = None,
    ichimoku_tenkan: int | None = None,
    ichimoku_kijun: int | None = None,
    ichimoku_span_b: int | None = None,
    signal_memory_window: int | None = None,
    score_signal_cap: float | None = None,
    weighted_score_cap: float | None = None,
    weighted_family_score_cap: float | None = None,
    recent_score_weight: float | None = None,
    min_confluence_signals: int | None = None,
    min_confluence_families: int | None = None,
    family_score_cap: float | None = None,
    context_window: int | None = None,
    entry_cooldown_bars: int | None = None,
    context_full_min: float | None = None,
    context_full_margin: float | None = None,
    context_soft_min: float | None = None,
    context_soft_margin: float | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """Append common TA values, long/short flags, and confluence scores."""

    cfg = _resolve_config(
        config,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_midline=rsi_midline,
        rsi_overbought=rsi_overbought,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_window=bb_window,
        bb_std=bb_std,
        bb_squeeze_window=bb_squeeze_window,
        bb_squeeze_quantile=bb_squeeze_quantile,
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        ema_long=ema_long,
        sma_fast=sma_fast,
        sma_mid=sma_mid,
        sma_slow=sma_slow,
        sma_long=sma_long,
        tema_fast=tema_fast,
        tema_slow=tema_slow,
        dema_fast=dema_fast,
        dema_slow=dema_slow,
        wma_fast=wma_fast,
        wma_slow=wma_slow,
        hma_fast=hma_fast,
        hma_slow=hma_slow,
        stoch_period=stoch_period,
        stoch_smooth=stoch_smooth,
        stoch_oversold=stoch_oversold,
        stoch_overbought=stoch_overbought,
        stoch_rsi_period=stoch_rsi_period,
        adx_period=adx_period,
        adx_trend_min=adx_trend_min,
        cci_period=cci_period,
        cci_oversold=cci_oversold,
        cci_overbought=cci_overbought,
        mfi_period=mfi_period,
        mfi_oversold=mfi_oversold,
        mfi_overbought=mfi_overbought,
        cmo_period=cmo_period,
        cmo_oversold=cmo_oversold,
        cmo_overbought=cmo_overbought,
        roc_period=roc_period,
        momentum_period=momentum_period,
        williams_period=williams_period,
        williams_oversold=williams_oversold,
        williams_overbought=williams_overbought,
        aroon_period=aroon_period,
        aroon_trend_min=aroon_trend_min,
        trix_period=trix_period,
        trix_signal=trix_signal,
        ultimate_short=ultimate_short,
        ultimate_medium=ultimate_medium,
        ultimate_long=ultimate_long,
        ultimate_oversold=ultimate_oversold,
        ultimate_overbought=ultimate_overbought,
        kst_signal=kst_signal,
        atr_period=atr_period,
        atr_expansion_mult=atr_expansion_mult,
        keltner_window=keltner_window,
        keltner_atr_mult=keltner_atr_mult,
        donchian_window=donchian_window,
        zscore_window=zscore_window,
        zscore_extreme=zscore_extreme,
        vwap_window=vwap_window,
        volume_window=volume_window,
        volume_breakout_mult=volume_breakout_mult,
        cmf_period=cmf_period,
        obv_fast=obv_fast,
        obv_slow=obv_slow,
        adl_fast=adl_fast,
        adl_slow=adl_slow,
        efi_period=efi_period,
        ichimoku_tenkan=ichimoku_tenkan,
        ichimoku_kijun=ichimoku_kijun,
        ichimoku_span_b=ichimoku_span_b,
        signal_memory_window=signal_memory_window,
        score_signal_cap=score_signal_cap,
        weighted_score_cap=weighted_score_cap,
        weighted_family_score_cap=weighted_family_score_cap,
        recent_score_weight=recent_score_weight,
        min_confluence_signals=min_confluence_signals,
        min_confluence_families=min_confluence_families,
        family_score_cap=family_score_cap,
        context_window=context_window,
        entry_cooldown_bars=entry_cooldown_bars,
        context_full_min=context_full_min,
        context_full_margin=context_full_margin,
        context_soft_min=context_soft_min,
        context_soft_margin=context_soft_margin,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    p = cfg.prefix
    raw = _build_raw_columns(frame, cfg)
    long_flags, short_flags = _build_signal_flags(frame, cfg, raw)
    score_cols = _build_score_columns(frame.index, cfg, long_flags, short_flags)

    new_cols: dict[str, Series] = {}
    new_cols.update({f"{p}_{name}": value for name, value in raw.items()})
    new_cols.update({f"{p}_{name}_long": value for name, value in long_flags.items()})
    new_cols.update({f"{p}_{name}_short": value for name, value in short_flags.items()})
    new_cols.update({f"{p}_{name}": value for name, value in score_cols.items()})

    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def add_simple_confluence(
    dataframe: DataFrame,
    config: SimpleConfluenceConfig | None = None,
    **kwargs: object,
) -> DataFrame:
    """Short alias for strategy code that prefers concise indicator names."""

    return add_simple_confluence_indicator(dataframe, config=config, **kwargs)


def _build_raw_columns(frame: DataFrame, cfg: SimpleConfluenceConfig) -> dict[str, Series]:
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close").replace(0.0, np.nan)
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    typical = (high + low + close) / 3.0

    ema_fast = _ema(close, cfg.ema_fast)
    ema_mid = _ema(close, cfg.ema_mid)
    ema_slow = _ema(close, cfg.ema_slow)
    ema_long = _ema(close, cfg.ema_long)
    sma_fast = _sma(close, cfg.sma_fast)
    sma_mid = _sma(close, cfg.sma_mid)
    sma_slow = _sma(close, cfg.sma_slow)
    sma_long = _sma(close, cfg.sma_long)
    wma_fast = _wma(close, cfg.wma_fast)
    wma_slow = _wma(close, cfg.wma_slow)
    hma_fast = _hma(close, cfg.hma_fast)
    hma_slow = _hma(close, cfg.hma_slow)
    dema_fast = _dema(close, cfg.dema_fast)
    dema_slow = _dema(close, cfg.dema_slow)
    tema_fast = _tema(close, cfg.tema_fast)
    tema_slow = _tema(close, cfg.tema_slow)

    rsi = _rsi(close, cfg.rsi_period)
    macd_line, macd_signal, macd_hist = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    ppo_line, ppo_signal, ppo_hist = _ppo(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    bb_mid, bb_upper, bb_lower, bb_width, bb_percent_b = _bollinger(close, cfg.bb_window, cfg.bb_std)
    tr = _true_range(frame)
    atr = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    natr = 100.0 * _safe_div(atr, close)
    keltner_mid = typical.ewm(span=cfg.keltner_window, adjust=False, min_periods=cfg.keltner_window).mean()
    keltner_upper = keltner_mid + (cfg.keltner_atr_mult * atr)
    keltner_lower = keltner_mid - (cfg.keltner_atr_mult * atr)
    donchian_high = high.rolling(cfg.donchian_window, min_periods=2).max().shift(1)
    donchian_low = low.rolling(cfg.donchian_window, min_periods=2).min().shift(1)

    stoch_k, stoch_d = _stochastic(high, low, close, cfg.stoch_period, cfg.stoch_smooth)
    stoch_rsi_k, stoch_rsi_d = _stoch_rsi(rsi, cfg.stoch_rsi_period, cfg.stoch_smooth)
    dmi = _dmi(frame, cfg.adx_period)
    cci = _cci(high, low, close, cfg.cci_period)
    mfi = _mfi(high, low, close, volume, cfg.mfi_period)
    cmo = _cmo(close, cfg.cmo_period)
    roc = close.pct_change(cfg.roc_period) * 100.0
    momentum = close - close.shift(cfg.momentum_period)
    williams_r = _williams_r(high, low, close, cfg.williams_period)
    aroon_up, aroon_down = _aroon(high, low, cfg.aroon_period)
    trix, trix_signal = _trix(close, cfg.trix_period, cfg.trix_signal)
    ultimate = _ultimate_oscillator(high, low, close, cfg.ultimate_short, cfg.ultimate_medium, cfg.ultimate_long)
    ao = _awesome_oscillator(high, low)
    kst, kst_signal = _kst(close, cfg.kst_signal)
    vwap = _rolling_vwap(typical, volume, cfg.vwap_window)
    obv = _obv(close, volume)
    cmf = _cmf(high, low, close, volume, cfg.cmf_period)
    adl = _adl(high, low, close, volume)
    efi = _efi(close, volume, cfg.efi_period)
    ichimoku = _ichimoku(high, low, cfg.ichimoku_tenkan, cfg.ichimoku_kijun, cfg.ichimoku_span_b)
    volume_mean = volume.rolling(cfg.volume_window, min_periods=1).mean()
    volume_ratio = _safe_div(volume, volume_mean)
    price_zscore = _zscore(close, cfg.zscore_window)
    candle_range = (high - low).clip(lower=0.0)
    close_location = ((_safe_div(close - low, candle_range) * 2.0) - 1.0).clip(-1.0, 1.0)

    raw = {
        "hlc3": typical,
        "close_location": close_location,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "ppo": ppo_line,
        "ppo_signal": ppo_signal,
        "ppo_hist": ppo_hist,
        "bb_mid": bb_mid,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "bb_percent_b": bb_percent_b,
        "ema_fast": ema_fast,
        "ema_mid": ema_mid,
        "ema_slow": ema_slow,
        "ema_long": ema_long,
        "sma_fast": sma_fast,
        "sma_mid": sma_mid,
        "sma_slow": sma_slow,
        "sma_long": sma_long,
        "dema_fast": dema_fast,
        "dema_slow": dema_slow,
        "tema_fast": tema_fast,
        "tema_slow": tema_slow,
        "wma_fast": wma_fast,
        "wma_slow": wma_slow,
        "hma_fast": hma_fast,
        "hma_slow": hma_slow,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_rsi_k": stoch_rsi_k,
        "stoch_rsi_d": stoch_rsi_d,
        "adx": dmi["adx"],
        "plus_di": dmi["plus_di"],
        "minus_di": dmi["minus_di"],
        "cci": cci,
        "mfi": mfi,
        "cmo": cmo,
        "roc": roc,
        "momentum": momentum,
        "williams_r": williams_r,
        "aroon_up": aroon_up,
        "aroon_down": aroon_down,
        "aroon_osc": aroon_up - aroon_down,
        "trix": trix,
        "trix_signal": trix_signal,
        "trix_hist": trix - trix_signal,
        "ultimate_osc": ultimate,
        "ao": ao,
        "kst": kst,
        "kst_signal": kst_signal,
        "kst_hist": kst - kst_signal,
        "true_range": tr,
        "atr": atr,
        "natr": natr,
        "keltner_mid": keltner_mid,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,
        "donchian_high": donchian_high,
        "donchian_low": donchian_low,
        "zscore": price_zscore,
        "vwap": vwap,
        "obv": obv,
        "obv_fast": _ema(obv, cfg.obv_fast),
        "obv_slow": _ema(obv, cfg.obv_slow),
        "cmf": cmf,
        "adl": adl,
        "adl_fast": _ema(adl, cfg.adl_fast),
        "adl_slow": _ema(adl, cfg.adl_slow),
        "efi": efi,
        "efi_signal": _ema(efi, cfg.efi_period),
        "ichimoku_tenkan": ichimoku["tenkan"],
        "ichimoku_kijun": ichimoku["kijun"],
        "ichimoku_span_a": ichimoku["span_a"],
        "ichimoku_span_b": ichimoku["span_b"],
    }
    raw.update(_talib_candle_columns(open_, high, low, close))
    return raw


def _build_signal_flags(
    frame: DataFrame,
    cfg: SimpleConfluenceConfig,
    raw: dict[str, Series],
) -> tuple[dict[str, Series], dict[str, Series]]:
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close").replace(0.0, np.nan)
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    body = close - open_
    body_abs = body.abs()
    candle_range = (high - low).clip(lower=0.0)
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    bb_squeeze_level = raw["bb_width"].rolling(cfg.bb_squeeze_window, min_periods=5).quantile(cfg.bb_squeeze_quantile)
    bb_squeeze = raw["bb_width"].le(bb_squeeze_level)
    bb_squeeze_prev = bb_squeeze.shift(1, fill_value=False).astype(bool)
    atr_expanding = raw["true_range"].gt(raw["atr"] * cfg.atr_expansion_mult)
    volume_expanding = raw["volume_ratio"].gt(cfg.volume_breakout_mult)
    prev_high = high.rolling(3, min_periods=2).max().shift(1)
    prev_low = low.rolling(3, min_periods=2).min().shift(1)

    long: dict[str, Series] = {}
    short: dict[str, Series] = {}

    _add_pair(long, short, "rsi_cross_oversold", _crossed_above(raw["rsi"], cfg.rsi_oversold), _crossed_below(raw["rsi"], cfg.rsi_overbought))
    _add_pair(long, short, "rsi_cross_midline", _crossed_above(raw["rsi"], cfg.rsi_midline), _crossed_below(raw["rsi"], cfg.rsi_midline))
    _add_pair(
        long,
        short,
        "rsi_extreme_reversal",
        raw["rsi"].shift(1).lt(cfg.rsi_oversold) & raw["rsi"].gt(raw["rsi"].shift(1)),
        raw["rsi"].shift(1).gt(cfg.rsi_overbought) & raw["rsi"].lt(raw["rsi"].shift(1)),
    )
    _add_pair(
        long,
        short,
        "rsi_range_momentum",
        raw["rsi"].gt(cfg.rsi_midline) & raw["rsi"].lt(cfg.rsi_overbought) & raw["rsi"].diff().gt(0.0),
        raw["rsi"].lt(cfg.rsi_midline) & raw["rsi"].gt(cfg.rsi_oversold) & raw["rsi"].diff().lt(0.0),
    )
    _add_pair(long, short, "macd_cross_signal", _crossed_above(raw["macd"], raw["macd_signal"]), _crossed_below(raw["macd"], raw["macd_signal"]))
    _add_pair(long, short, "macd_cross_zero", _crossed_above(raw["macd"], 0.0), _crossed_below(raw["macd"], 0.0))
    _add_pair(
        long,
        short,
        "macd_hist_reversal",
        raw["macd_hist"].lt(0.0) & raw["macd_hist"].diff().gt(0.0) & raw["macd_hist"].diff().shift(1).le(0.0),
        raw["macd_hist"].gt(0.0) & raw["macd_hist"].diff().lt(0.0) & raw["macd_hist"].diff().shift(1).ge(0.0),
    )
    _add_pair(long, short, "ppo_cross_signal", _crossed_above(raw["ppo"], raw["ppo_signal"]), _crossed_below(raw["ppo"], raw["ppo_signal"]))
    _add_pair(long, short, "ppo_cross_zero", _crossed_above(raw["ppo"], 0.0), _crossed_below(raw["ppo"], 0.0))
    _add_pair(
        long,
        short,
        "stoch_cross_extreme",
        _crossed_above(raw["stoch_k"], raw["stoch_d"]) & raw["stoch_k"].shift(1).lt(cfg.stoch_oversold),
        _crossed_below(raw["stoch_k"], raw["stoch_d"]) & raw["stoch_k"].shift(1).gt(cfg.stoch_overbought),
    )
    _add_pair(long, short, "stoch_cross_midline", _crossed_above(raw["stoch_k"], 50.0), _crossed_below(raw["stoch_k"], 50.0))
    _add_pair(
        long,
        short,
        "stoch_rsi_cross_extreme",
        _crossed_above(raw["stoch_rsi_k"], raw["stoch_rsi_d"]) & raw["stoch_rsi_k"].shift(1).lt(cfg.stoch_oversold),
        _crossed_below(raw["stoch_rsi_k"], raw["stoch_rsi_d"]) & raw["stoch_rsi_k"].shift(1).gt(cfg.stoch_overbought),
    )
    _add_pair(long, short, "stoch_rsi_cross_midline", _crossed_above(raw["stoch_rsi_k"], 50.0), _crossed_below(raw["stoch_rsi_k"], 50.0))
    _add_pair(long, short, "cci_cross_extreme", _crossed_above(raw["cci"], cfg.cci_oversold), _crossed_below(raw["cci"], cfg.cci_overbought))
    _add_pair(long, short, "cci_cross_zero", _crossed_above(raw["cci"], 0.0), _crossed_below(raw["cci"], 0.0))
    _add_pair(long, short, "mfi_cross_extreme", _crossed_above(raw["mfi"], cfg.mfi_oversold), _crossed_below(raw["mfi"], cfg.mfi_overbought))
    _add_pair(
        long,
        short,
        "mfi_money_flow_bias",
        raw["mfi"].gt(50.0) & raw["mfi"].diff().gt(0.0),
        raw["mfi"].lt(50.0) & raw["mfi"].diff().lt(0.0),
    )
    _add_pair(long, short, "cmo_cross_extreme", _crossed_above(raw["cmo"], cfg.cmo_oversold), _crossed_below(raw["cmo"], cfg.cmo_overbought))
    _add_pair(long, short, "cmo_cross_zero", _crossed_above(raw["cmo"], 0.0), _crossed_below(raw["cmo"], 0.0))
    _add_pair(long, short, "williams_cross_extreme", _crossed_above(raw["williams_r"], cfg.williams_oversold), _crossed_below(raw["williams_r"], cfg.williams_overbought))
    _add_pair(long, short, "ultimate_cross_extreme", _crossed_above(raw["ultimate_osc"], cfg.ultimate_oversold), _crossed_below(raw["ultimate_osc"], cfg.ultimate_overbought))
    _add_pair(long, short, "ultimate_cross_midline", _crossed_above(raw["ultimate_osc"], 50.0), _crossed_below(raw["ultimate_osc"], 50.0))
    _add_pair(long, short, "roc_cross_zero", _crossed_above(raw["roc"], 0.0), _crossed_below(raw["roc"], 0.0))
    _add_pair(long, short, "momentum_cross_zero", _crossed_above(raw["momentum"], 0.0), _crossed_below(raw["momentum"], 0.0))
    _add_pair(long, short, "trix_cross_signal", _crossed_above(raw["trix"], raw["trix_signal"]), _crossed_below(raw["trix"], raw["trix_signal"]))
    _add_pair(long, short, "trix_cross_zero", _crossed_above(raw["trix"], 0.0), _crossed_below(raw["trix"], 0.0))
    _add_pair(long, short, "ao_cross_zero", _crossed_above(raw["ao"], 0.0), _crossed_below(raw["ao"], 0.0))
    _add_pair(
        long,
        short,
        "ao_saucer",
        raw["ao"].gt(0.0) & raw["ao"].diff().gt(0.0) & raw["ao"].diff().shift(1).lt(0.0),
        raw["ao"].lt(0.0) & raw["ao"].diff().lt(0.0) & raw["ao"].diff().shift(1).gt(0.0),
    )
    _add_pair(long, short, "kst_cross_signal", _crossed_above(raw["kst"], raw["kst_signal"]), _crossed_below(raw["kst"], raw["kst_signal"]))
    _add_pair(long, short, "kst_cross_zero", _crossed_above(raw["kst"], 0.0), _crossed_below(raw["kst"], 0.0))

    _add_pair(long, short, "ema_fast_cross_mid", _crossed_above(raw["ema_fast"], raw["ema_mid"]), _crossed_below(raw["ema_fast"], raw["ema_mid"]))
    _add_pair(long, short, "ema_mid_cross_slow", _crossed_above(raw["ema_mid"], raw["ema_slow"]), _crossed_below(raw["ema_mid"], raw["ema_slow"]))
    _add_pair(long, short, "ema_slow_cross_long", _crossed_above(raw["ema_slow"], raw["ema_long"]), _crossed_below(raw["ema_slow"], raw["ema_long"]))
    _add_pair(long, short, "ema_price_reclaim_fast", _crossed_above(close, raw["ema_fast"]), _crossed_below(close, raw["ema_fast"]))
    _add_pair(long, short, "ema_price_reclaim_slow", _crossed_above(close, raw["ema_slow"]), _crossed_below(close, raw["ema_slow"]))
    ema_bull_stack = close.gt(raw["ema_fast"]) & raw["ema_fast"].gt(raw["ema_mid"]) & raw["ema_mid"].gt(raw["ema_slow"]) & raw["ema_slow"].gt(raw["ema_long"])
    ema_bear_stack = close.lt(raw["ema_fast"]) & raw["ema_fast"].lt(raw["ema_mid"]) & raw["ema_mid"].lt(raw["ema_slow"]) & raw["ema_slow"].lt(raw["ema_long"])
    _add_pair(long, short, "ema_stack", ema_bull_stack, ema_bear_stack)
    _add_pair(
        long,
        short,
        "ema_pullback_hold",
        ema_bull_stack & low.le(raw["ema_fast"]) & close.gt(raw["ema_fast"]),
        ema_bear_stack & high.ge(raw["ema_fast"]) & close.lt(raw["ema_fast"]),
    )
    _add_pair(long, short, "sma_fast_cross_mid", _crossed_above(raw["sma_fast"], raw["sma_mid"]), _crossed_below(raw["sma_fast"], raw["sma_mid"]))
    _add_pair(long, short, "sma_mid_cross_slow", _crossed_above(raw["sma_mid"], raw["sma_slow"]), _crossed_below(raw["sma_mid"], raw["sma_slow"]))
    _add_pair(long, short, "sma_slow_cross_long", _crossed_above(raw["sma_slow"], raw["sma_long"]), _crossed_below(raw["sma_slow"], raw["sma_long"]))
    _add_pair(long, short, "sma_price_reclaim_mid", _crossed_above(close, raw["sma_mid"]), _crossed_below(close, raw["sma_mid"]))
    sma_bull_stack = close.gt(raw["sma_fast"]) & raw["sma_fast"].gt(raw["sma_mid"]) & raw["sma_mid"].gt(raw["sma_slow"]) & raw["sma_slow"].gt(raw["sma_long"])
    sma_bear_stack = close.lt(raw["sma_fast"]) & raw["sma_fast"].lt(raw["sma_mid"]) & raw["sma_mid"].lt(raw["sma_slow"]) & raw["sma_slow"].lt(raw["sma_long"])
    _add_pair(long, short, "sma_stack", sma_bull_stack, sma_bear_stack)
    _add_pair(long, short, "tema_cross", _crossed_above(raw["tema_fast"], raw["tema_slow"]), _crossed_below(raw["tema_fast"], raw["tema_slow"]))
    _add_pair(long, short, "tema_price_reclaim", _crossed_above(close, raw["tema_fast"]), _crossed_below(close, raw["tema_fast"]))
    _add_pair(long, short, "dema_cross", _crossed_above(raw["dema_fast"], raw["dema_slow"]), _crossed_below(raw["dema_fast"], raw["dema_slow"]))
    _add_pair(long, short, "wma_cross", _crossed_above(raw["wma_fast"], raw["wma_slow"]), _crossed_below(raw["wma_fast"], raw["wma_slow"]))
    _add_pair(long, short, "hma_cross", _crossed_above(raw["hma_fast"], raw["hma_slow"]), _crossed_below(raw["hma_fast"], raw["hma_slow"]))
    _add_pair(long, short, "hma_slope", raw["hma_fast"].diff().gt(0.0) & raw["hma_slow"].diff().gt(0.0), raw["hma_fast"].diff().lt(0.0) & raw["hma_slow"].diff().lt(0.0))
    _add_pair(
        long,
        short,
        "ma_cluster",
        close.gt(raw["ema_mid"]) & close.gt(raw["sma_mid"]) & close.gt(raw["hma_fast"]) & raw["ema_mid"].diff().gt(0.0),
        close.lt(raw["ema_mid"]) & close.lt(raw["sma_mid"]) & close.lt(raw["hma_fast"]) & raw["ema_mid"].diff().lt(0.0),
    )
    _add_pair(
        long,
        short,
        "ma_slope_consensus",
        raw["ema_mid"].diff().gt(0.0) & raw["sma_mid"].diff().gt(0.0) & raw["hma_fast"].diff().gt(0.0),
        raw["ema_mid"].diff().lt(0.0) & raw["sma_mid"].diff().lt(0.0) & raw["hma_fast"].diff().lt(0.0),
    )

    _add_pair(
        long,
        short,
        "adx_di_cross",
        _crossed_above(raw["plus_di"], raw["minus_di"]) & raw["adx"].ge(cfg.adx_trend_min),
        _crossed_above(raw["minus_di"], raw["plus_di"]) & raw["adx"].ge(cfg.adx_trend_min),
    )
    _add_pair(
        long,
        short,
        "adx_directional_trend",
        raw["plus_di"].gt(raw["minus_di"]) & raw["adx"].ge(cfg.adx_trend_min) & raw["adx"].diff().gt(0.0),
        raw["minus_di"].gt(raw["plus_di"]) & raw["adx"].ge(cfg.adx_trend_min) & raw["adx"].diff().gt(0.0),
    )
    _add_pair(long, short, "aroon_cross", _crossed_above(raw["aroon_up"], raw["aroon_down"]), _crossed_above(raw["aroon_down"], raw["aroon_up"]))
    _add_pair(
        long,
        short,
        "aroon_trend",
        raw["aroon_up"].ge(cfg.aroon_trend_min) & raw["aroon_down"].le(100.0 - cfg.aroon_trend_min),
        raw["aroon_down"].ge(cfg.aroon_trend_min) & raw["aroon_up"].le(100.0 - cfg.aroon_trend_min),
    )
    _add_pair(long, short, "ichimoku_tenkan_kijun_cross", _crossed_above(raw["ichimoku_tenkan"], raw["ichimoku_kijun"]), _crossed_below(raw["ichimoku_tenkan"], raw["ichimoku_kijun"]))
    cloud_top = pd.concat([raw["ichimoku_span_a"], raw["ichimoku_span_b"]], axis=1).max(axis=1)
    cloud_bottom = pd.concat([raw["ichimoku_span_a"], raw["ichimoku_span_b"]], axis=1).min(axis=1)
    _add_pair(
        long,
        short,
        "ichimoku_cloud_position",
        close.gt(cloud_top) & raw["ichimoku_span_a"].gt(raw["ichimoku_span_b"]),
        close.lt(cloud_bottom) & raw["ichimoku_span_a"].lt(raw["ichimoku_span_b"]),
    )

    _add_pair(
        long,
        short,
        "bb_reversal",
        low.le(raw["bb_lower"]) & close.gt(raw["bb_lower"]) & body.gt(0.0),
        high.ge(raw["bb_upper"]) & close.lt(raw["bb_upper"]) & body.lt(0.0),
    )
    _add_pair(
        long,
        short,
        "bb_reentry",
        close.shift(1).lt(raw["bb_lower"].shift(1)) & close.gt(raw["bb_lower"]),
        close.shift(1).gt(raw["bb_upper"].shift(1)) & close.lt(raw["bb_upper"]),
    )
    _add_pair(long, short, "bb_midline_reclaim", _crossed_above(close, raw["bb_mid"]), _crossed_below(close, raw["bb_mid"]))
    _add_pair(long, short, "bb_percent_b_zero_one", _crossed_above(raw["bb_percent_b"], 0.0), _crossed_below(raw["bb_percent_b"], 1.0))
    _add_pair(long, short, "bb_percent_b_midline", _crossed_above(raw["bb_percent_b"], 0.5), _crossed_below(raw["bb_percent_b"], 0.5))
    _add_pair(
        long,
        short,
        "bb_squeeze_breakout",
        bb_squeeze_prev & close.gt(raw["bb_upper"]),
        bb_squeeze_prev & close.lt(raw["bb_lower"]),
    )
    _add_pair(
        long,
        short,
        "bb_bandwalk",
        close.gt(raw["bb_upper"]) & raw["rsi"].gt(cfg.rsi_midline) & raw["ema_fast"].gt(raw["ema_mid"]),
        close.lt(raw["bb_lower"]) & raw["rsi"].lt(cfg.rsi_midline) & raw["ema_fast"].lt(raw["ema_mid"]),
    )
    _add_pair(
        long,
        short,
        "keltner_reversal",
        low.le(raw["keltner_lower"]) & close.gt(raw["keltner_lower"]) & body.gt(0.0),
        high.ge(raw["keltner_upper"]) & close.lt(raw["keltner_upper"]) & body.lt(0.0),
    )
    _add_pair(long, short, "keltner_breakout", close.gt(raw["keltner_upper"]), close.lt(raw["keltner_lower"]))
    _add_pair(long, short, "donchian_breakout", close.gt(raw["donchian_high"]), close.lt(raw["donchian_low"]))
    _add_pair(
        long,
        short,
        "donchian_failed_break",
        low.lt(raw["donchian_low"]) & close.gt(raw["donchian_low"]),
        high.gt(raw["donchian_high"]) & close.lt(raw["donchian_high"]),
    )
    _add_pair(
        long,
        short,
        "atr_expansion_direction",
        atr_expanding & body.gt(0.0) & close.gt(prev_high),
        atr_expanding & body.lt(0.0) & close.lt(prev_low),
    )
    _add_pair(
        long,
        short,
        "zscore_reversal",
        raw["zscore"].shift(1).lt(-cfg.zscore_extreme) & raw["zscore"].gt(raw["zscore"].shift(1)),
        raw["zscore"].shift(1).gt(cfg.zscore_extreme) & raw["zscore"].lt(raw["zscore"].shift(1)),
    )

    _add_pair(long, short, "vwap_reclaim", _crossed_above(close, raw["vwap"]), _crossed_below(close, raw["vwap"]))
    _add_pair(
        long,
        short,
        "vwap_trend",
        close.gt(raw["vwap"]) & raw["vwap"].diff().gt(0.0),
        close.lt(raw["vwap"]) & raw["vwap"].diff().lt(0.0),
    )
    _add_pair(long, short, "obv_cross", _crossed_above(raw["obv_fast"], raw["obv_slow"]), _crossed_below(raw["obv_fast"], raw["obv_slow"]))
    _add_pair(
        long,
        short,
        "obv_breakout",
        raw["obv"].gt(raw["obv"].rolling(cfg.obv_slow, min_periods=2).max().shift(1)),
        raw["obv"].lt(raw["obv"].rolling(cfg.obv_slow, min_periods=2).min().shift(1)),
    )
    _add_pair(long, short, "cmf_cross_zero", _crossed_above(raw["cmf"], 0.0), _crossed_below(raw["cmf"], 0.0))
    _add_pair(
        long,
        short,
        "cmf_pressure",
        raw["cmf"].gt(0.05) & raw["cmf"].diff().gt(0.0),
        raw["cmf"].lt(-0.05) & raw["cmf"].diff().lt(0.0),
    )
    _add_pair(long, short, "adl_cross", _crossed_above(raw["adl_fast"], raw["adl_slow"]), _crossed_below(raw["adl_fast"], raw["adl_slow"]))
    _add_pair(long, short, "efi_cross_zero", _crossed_above(raw["efi"], 0.0), _crossed_below(raw["efi"], 0.0))
    _add_pair(
        long,
        short,
        "volume_confirmed_breakout",
        volume_expanding & close.gt(raw["donchian_high"]) & body.gt(0.0),
        volume_expanding & close.lt(raw["donchian_low"]) & body.lt(0.0),
    )
    _add_pair(
        long,
        short,
        "volume_climax_reversal",
        volume_expanding & low.lt(raw["donchian_low"]) & close.gt(open_) & raw["close_location"].gt(0.25),
        volume_expanding & high.gt(raw["donchian_high"]) & close.lt(open_) & raw["close_location"].lt(-0.25),
    )

    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_body_abs = body_abs.shift(1)
    small_body = body_abs.le(candle_range * 0.30)
    _add_pair(
        long,
        short,
        "engulfing",
        prev_close.lt(prev_open) & close.gt(open_) & close.ge(prev_open) & open_.le(prev_close),
        prev_close.gt(prev_open) & close.lt(open_) & close.le(prev_open) & open_.ge(prev_close),
    )
    _add_pair(
        long,
        short,
        "hammer_shooting_star",
        lower_wick.ge(body_abs * 2.0) & upper_wick.le(body_abs) & close.gt(open_),
        upper_wick.ge(body_abs * 2.0) & lower_wick.le(body_abs) & close.lt(open_),
    )
    _add_pair(
        long,
        short,
        "piercing_dark_cloud",
        prev_close.lt(prev_open) & open_.lt(prev_close) & close.gt((prev_open + prev_close) / 2.0) & close.lt(prev_open),
        prev_close.gt(prev_open) & open_.gt(prev_close) & close.lt((prev_open + prev_close) / 2.0) & close.gt(prev_open),
    )
    _add_pair(
        long,
        short,
        "morning_evening_star",
        close.shift(2).lt(open_.shift(2)) & small_body.shift(1) & close.gt(open_) & close.gt((open_.shift(2) + close.shift(2)) / 2.0),
        close.shift(2).gt(open_.shift(2)) & small_body.shift(1) & close.lt(open_) & close.lt((open_.shift(2) + close.shift(2)) / 2.0),
    )
    _add_pair(
        long,
        short,
        "three_candle_drive",
        close.gt(open_) & close.shift(1).gt(open_.shift(1)) & close.shift(2).gt(open_.shift(2)) & body_abs.gt(prev_body_abs),
        close.lt(open_) & close.shift(1).lt(open_.shift(1)) & close.shift(2).lt(open_.shift(2)) & body_abs.gt(prev_body_abs),
    )
    _add_pair(
        long,
        short,
        "inside_bar_break",
        high.shift(1).lt(high.shift(2)) & low.shift(1).gt(low.shift(2)) & close.gt(high.shift(1)),
        high.shift(1).lt(high.shift(2)) & low.shift(1).gt(low.shift(2)) & close.lt(low.shift(1)),
    )
    for pattern in _TALIB_CDL_PATTERNS:
        name = _talib_candle_name(pattern)
        value = raw[name]
        _add_pair(long, short, name, value.gt(0.0), value.lt(0.0))

    return long, short


def _talib_candle_columns(open_: Series, high: Series, low: Series, close: Series) -> dict[str, Series]:
    open_values = pd.to_numeric(open_, errors="coerce").to_numpy(dtype="float64")
    high_values = pd.to_numeric(high, errors="coerce").to_numpy(dtype="float64")
    low_values = pd.to_numeric(low, errors="coerce").to_numpy(dtype="float64")
    close_values = pd.to_numeric(close, errors="coerce").to_numpy(dtype="float64")
    columns: dict[str, Series] = {}
    for pattern in _TALIB_CDL_PATTERNS:
        values = getattr(talib, pattern)(open_values, high_values, low_values, close_values)
        columns[_talib_candle_name(pattern)] = pd.Series(values, index=close.index, dtype="float64")
    return columns


def _talib_candle_name(pattern: str) -> str:
    return f"cdl_{pattern[3:].lower()}"


def _build_score_columns(
    index: pd.Index,
    cfg: SimpleConfluenceConfig,
    long_flags: dict[str, Series],
    short_flags: dict[str, Series],
) -> dict[str, Series]:
    long_frame = pd.DataFrame(long_flags, index=index).astype("int8")
    short_frame = pd.DataFrame(short_flags, index=index).astype("int8")
    long_count = long_frame.sum(axis=1).astype("float64")
    short_count = short_frame.sum(axis=1).astype("float64")
    memory = int(cfg.signal_memory_window)
    long_recent = long_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    short_recent = short_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    long_family_frame = _family_vote_frame(long_frame)
    short_family_frame = _family_vote_frame(short_frame)
    long_family_count = long_family_frame.sum(axis=1).astype("float64")
    short_family_count = short_family_frame.sum(axis=1).astype("float64")
    long_family_recent = long_family_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    short_family_recent = short_family_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    long_weighted_sum = _weighted_vote_sum(long_frame, "long")
    short_weighted_sum = _weighted_vote_sum(short_frame, "short")
    long_weighted_recent = _weighted_rolling_sum(long_frame, memory, "long")
    short_weighted_recent = _weighted_rolling_sum(short_frame, memory, "short")
    long_weighted_family_frame = _weighted_family_vote_frame(long_frame, "long")
    short_weighted_family_frame = _weighted_family_vote_frame(short_frame, "short")
    long_weighted_family_sum = long_weighted_family_frame.sum(axis=1).astype("float64")
    short_weighted_family_sum = short_weighted_family_frame.sum(axis=1).astype("float64")
    long_weighted_family_recent = long_weighted_family_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    short_weighted_family_recent = short_weighted_family_frame.rolling(memory, min_periods=1).sum().sum(axis=1).astype("float64")
    cap = max(float(cfg.score_signal_cap), 1.0)
    family_cap = max(float(cfg.family_score_cap), 1.0)
    weighted_cap = max(float(cfg.weighted_score_cap), 1.0)
    weighted_family_cap = max(float(cfg.weighted_family_score_cap), 1.0)
    current_weight = 1.0 - float(cfg.recent_score_weight)
    recent_denominator = cap * max(float(memory), 1.0)
    family_recent_denominator = family_cap * max(float(memory), 1.0)
    weighted_recent_denominator = weighted_cap * max(float(memory), 1.0)
    weighted_family_recent_denominator = weighted_family_cap * max(float(memory), 1.0)
    raw_long_current_score = _clip01(long_count / cap)
    raw_short_current_score = _clip01(short_count / cap)
    raw_long_recent_score = _clip01(long_recent / recent_denominator)
    raw_short_recent_score = _clip01(short_recent / recent_denominator)
    long_family_score = _clip01(long_family_count / family_cap)
    short_family_score = _clip01(short_family_count / family_cap)
    long_family_recent_score = _clip01(long_family_recent / family_recent_denominator)
    short_family_recent_score = _clip01(short_family_recent / family_recent_denominator)
    weighted_long_current_score = _clip01(long_weighted_sum / weighted_cap)
    weighted_short_current_score = _clip01(short_weighted_sum / weighted_cap)
    weighted_long_recent_score = _clip01(long_weighted_recent / weighted_recent_denominator)
    weighted_short_recent_score = _clip01(short_weighted_recent / weighted_recent_denominator)
    weighted_long_family_score = _clip01(long_weighted_family_sum / weighted_family_cap)
    weighted_short_family_score = _clip01(short_weighted_family_sum / weighted_family_cap)
    weighted_long_family_recent_score = _clip01(long_weighted_family_recent / weighted_family_recent_denominator)
    weighted_short_family_recent_score = _clip01(short_weighted_family_recent / weighted_family_recent_denominator)
    long_current_score = _clip01(0.75 * weighted_long_family_score + 0.25 * weighted_long_current_score)
    short_current_score = _clip01(0.75 * weighted_short_family_score + 0.25 * weighted_short_current_score)
    long_recent_score = _clip01(0.75 * weighted_long_family_recent_score + 0.25 * weighted_long_recent_score)
    short_recent_score = _clip01(0.75 * weighted_short_family_recent_score + 0.25 * weighted_short_recent_score)
    long_score = _clip01(current_weight * long_current_score + cfg.recent_score_weight * long_recent_score)
    short_score = _clip01(current_weight * short_current_score + cfg.recent_score_weight * short_recent_score)
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_margin = long_score - short_score
    context_long = long_score.ewm(span=cfg.context_window, min_periods=1, adjust=False).mean()
    context_short = short_score.ewm(span=cfg.context_window, min_periods=1, adjust=False).mean()
    context_margin = context_long - context_short
    state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1, -1], default=0),
        index=index,
        dtype="int8",
    )
    confluence_long_state = (
        long_family_count.ge(cfg.min_confluence_families)
        & long_family_count.gt(short_family_count)
        & score_margin.ge(cfg.context_soft_margin)
    )
    confluence_short_state = (
        short_family_count.ge(cfg.min_confluence_families)
        & short_family_count.gt(long_family_count)
        & score_margin.le(-cfg.context_soft_margin)
    )
    full_bull = (
        context_long.ge(cfg.context_full_min)
        & context_margin.ge(cfg.context_full_margin)
        & long_family_count.ge(cfg.min_confluence_families)
    )
    full_bear = (
        context_short.ge(cfg.context_full_min)
        & context_margin.le(-cfg.context_full_margin)
        & short_family_count.ge(cfg.min_confluence_families)
    )
    bullish_chop = (
        ~full_bull
        & ~full_bear
        & context_long.ge(cfg.context_soft_min)
        & context_margin.ge(cfg.context_soft_margin)
    )
    bearish_chop = (
        ~full_bull
        & ~full_bear
        & context_short.ge(cfg.context_soft_min)
        & context_margin.le(-cfg.context_soft_margin)
    )
    market_context = pd.Series(
        np.select(
            [full_bull, bullish_chop, full_bear, bearish_chop],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=index,
        dtype="float64",
    )
    entry_long = _dedupe_events(confluence_long_state & market_context.gt(0.0), cfg.entry_cooldown_bars)
    entry_short = _dedupe_events(confluence_short_state & market_context.lt(0.0), cfg.entry_cooldown_bars)
    exit_long = _dedupe_events(confluence_short_state | full_bear, cfg.entry_cooldown_bars)
    exit_short = _dedupe_events(confluence_long_state | full_bull, cfg.entry_cooldown_bars)
    return {
        "signal_count_long": long_count,
        "signal_count_short": short_count,
        "recent_signal_count_long": long_recent,
        "recent_signal_count_short": short_recent,
        "signal_count_total": long_count + short_count,
        "signal_delta": long_count - short_count,
        "family_count_long": long_family_count,
        "family_count_short": short_family_count,
        "recent_family_count_long": long_family_recent,
        "recent_family_count_short": short_family_recent,
        "family_delta": long_family_count - short_family_count,
        "weighted_signal_sum_long": long_weighted_sum,
        "weighted_signal_sum_short": short_weighted_sum,
        "weighted_recent_signal_sum_long": long_weighted_recent,
        "weighted_recent_signal_sum_short": short_weighted_recent,
        "weighted_family_sum_long": long_weighted_family_sum,
        "weighted_family_sum_short": short_weighted_family_sum,
        "weighted_recent_family_sum_long": long_weighted_family_recent,
        "weighted_recent_family_sum_short": short_weighted_family_recent,
        "weighted_signal_delta": long_weighted_sum - short_weighted_sum,
        "weighted_family_delta": long_weighted_family_sum - short_weighted_family_sum,
        "raw_score_long_current": raw_long_current_score,
        "raw_score_short_current": raw_short_current_score,
        "raw_score_long_recent": raw_long_recent_score,
        "raw_score_short_recent": raw_short_recent_score,
        "family_score_long": long_family_score,
        "family_score_short": short_family_score,
        "family_score_long_recent": long_family_recent_score,
        "family_score_short_recent": short_family_recent_score,
        "weighted_score_long_current": weighted_long_current_score,
        "weighted_score_short_current": weighted_short_current_score,
        "weighted_score_long_recent": weighted_long_recent_score,
        "weighted_score_short_recent": weighted_short_recent_score,
        "weighted_family_score_long": weighted_long_family_score,
        "weighted_family_score_short": weighted_short_family_score,
        "weighted_family_score_long_recent": weighted_long_family_recent_score,
        "weighted_family_score_short_recent": weighted_short_family_recent_score,
        "score_long_current": long_current_score,
        "score_short_current": short_current_score,
        "score_long_recent": long_recent_score,
        "score_short_recent": short_recent_score,
        "score_long": long_score,
        "score_short": short_score,
        "score_abs": abs_score,
        "market_context": market_context,
        "state": state,
        "confluence_long": confluence_long_state.astype("int8"),
        "confluence_short": confluence_short_state.astype("int8"),
        "entry_confluence_long": entry_long.astype("int8"),
        "entry_confluence_short": entry_short.astype("int8"),
        "suggested_entry_long": entry_long.astype("int8"),
        "suggested_entry_short": entry_short.astype("int8"),
        "hold_long": (market_context.gt(0.0) & ~full_bear).astype("int8"),
        "hold_short": (market_context.lt(0.0) & ~full_bull).astype("int8"),
        "exit_long": exit_long.astype("int8"),
        "exit_short": exit_short.astype("int8"),
    }


def _family_vote_frame(flag_frame: DataFrame) -> DataFrame:
    family_cols: dict[str, Series] = {}
    for family in sorted({_signal_family(column) for column in flag_frame.columns}):
        members = [column for column in flag_frame.columns if _signal_family(column) == family]
        family_cols[family] = flag_frame[members].sum(axis=1).gt(0).astype("int8")
    return pd.DataFrame(family_cols, index=flag_frame.index)


def _weighted_vote_sum(flag_frame: DataFrame, side: str) -> Series:
    weighted = {
        column: flag_frame[column].astype("float64") * _signal_weight(column, side)
        for column in flag_frame.columns
    }
    return pd.DataFrame(weighted, index=flag_frame.index).sum(axis=1).astype("float64")


def _weighted_rolling_sum(flag_frame: DataFrame, window: int, side: str) -> Series:
    weighted = {
        column: flag_frame[column].astype("float64") * _signal_weight(column, side)
        for column in flag_frame.columns
    }
    weighted_frame = pd.DataFrame(weighted, index=flag_frame.index)
    return weighted_frame.rolling(window, min_periods=1).sum().sum(axis=1).astype("float64")


def _weighted_family_vote_frame(flag_frame: DataFrame, side: str) -> DataFrame:
    family_cols: dict[str, Series] = {}
    for family in sorted({_signal_family(column) for column in flag_frame.columns}):
        members = [column for column in flag_frame.columns if _signal_family(column) == family]
        weighted_members = pd.DataFrame(
            {
                column: flag_frame[column].astype("float64") * _signal_weight(column, side)
                for column in members
            },
            index=flag_frame.index,
        )
        family_cols[family] = weighted_members.max(axis=1)
    return pd.DataFrame(family_cols, index=flag_frame.index)


def _signal_weight(name: str, side: str | None = None) -> float:
    if name in _SIGNAL_WEIGHTS:
        return _SIGNAL_WEIGHTS[name]
    if name.startswith("cdl_"):
        return _cdl_signal_weight(name, side)
    if _signal_family(name) == "volume":
        return _DIRECTIONAL_VOLUME_WEIGHTS.get(name, 0.0)
    return 0.75


def _cdl_signal_weight(name: str, side: str | None) -> float:
    weights = _CDL_SIGNAL_WEIGHTS.get(name)
    if weights is None:
        return _CDL_DEFAULT_WEIGHT
    if side == "long":
        return weights[0]
    if side == "short":
        return weights[1]
    return max(weights)


def _signal_family(name: str) -> str:
    if name.startswith(("rsi_", "stoch", "cci_", "mfi_", "cmo_", "williams_", "ultimate_")):
        return "oscillator"
    if name.startswith(("macd_", "ppo_", "roc_", "momentum_", "trix_", "ao_", "kst_")):
        return "momentum"
    if name.startswith(("ema_", "sma_", "tema_", "dema_", "wma_", "hma_", "aroon_", "ichimoku_", "adx_", "ma_")):
        return "trend"
    if name.startswith(("bb_", "keltner_", "donchian_", "zscore_", "atr_")):
        return "volatility"
    if name.startswith(("volume_", "vwap_", "obv_", "cmf_", "adl_", "efi_")):
        return "volume"
    if name.startswith("cdl_"):
        return "talib_candle"
    return "candle"


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    event = pd.Series(mask, index=mask.index).astype("boolean").fillna(False).astype(bool)
    previous_count = event.astype("float64").shift(1).rolling(max(1, int(cooldown_bars)), min_periods=1).sum().fillna(0.0)
    return event & previous_count.eq(0.0)


def _resolve_config(config: SimpleConfluenceConfig | None, **overrides: object) -> SimpleConfluenceConfig:
    base = config or SimpleConfluenceConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return SimpleConfluenceConfig(**values)


def _validate_config(cfg: SimpleConfluenceConfig) -> None:
    windows = [
        cfg.rsi_period,
        cfg.macd_fast,
        cfg.macd_slow,
        cfg.macd_signal,
        cfg.bb_window,
        cfg.bb_squeeze_window,
        cfg.ema_fast,
        cfg.ema_mid,
        cfg.ema_slow,
        cfg.ema_long,
        cfg.sma_fast,
        cfg.sma_mid,
        cfg.sma_slow,
        cfg.sma_long,
        cfg.tema_fast,
        cfg.tema_slow,
        cfg.dema_fast,
        cfg.dema_slow,
        cfg.wma_fast,
        cfg.wma_slow,
        cfg.hma_fast,
        cfg.hma_slow,
        cfg.stoch_period,
        cfg.stoch_smooth,
        cfg.stoch_rsi_period,
        cfg.adx_period,
        cfg.cci_period,
        cfg.mfi_period,
        cfg.cmo_period,
        cfg.roc_period,
        cfg.momentum_period,
        cfg.williams_period,
        cfg.aroon_period,
        cfg.trix_period,
        cfg.trix_signal,
        cfg.ultimate_short,
        cfg.ultimate_medium,
        cfg.ultimate_long,
        cfg.kst_signal,
        cfg.atr_period,
        cfg.keltner_window,
        cfg.donchian_window,
        cfg.zscore_window,
        cfg.vwap_window,
        cfg.volume_window,
        cfg.cmf_period,
        cfg.obv_fast,
        cfg.obv_slow,
        cfg.adl_fast,
        cfg.adl_slow,
        cfg.efi_period,
        cfg.ichimoku_tenkan,
        cfg.ichimoku_kijun,
        cfg.ichimoku_span_b,
        cfg.signal_memory_window,
        cfg.min_confluence_families,
        cfg.context_window,
        cfg.entry_cooldown_bars,
    ]
    if any(value < 2 for value in windows):
        raise ValueError("all simple confluence windows must be at least 2")
    if not cfg.macd_fast < cfg.macd_slow:
        raise ValueError("macd_fast must be smaller than macd_slow")
    if not cfg.rsi_oversold < cfg.rsi_midline < cfg.rsi_overbought:
        raise ValueError("expected rsi_oversold < rsi_midline < rsi_overbought")
    if not cfg.stoch_oversold < cfg.stoch_overbought:
        raise ValueError("expected stoch_oversold < stoch_overbought")
    if not cfg.mfi_oversold < cfg.mfi_overbought:
        raise ValueError("expected mfi_oversold < mfi_overbought")
    if not cfg.cmo_oversold < 0.0 < cfg.cmo_overbought:
        raise ValueError("expected cmo_oversold < 0 < cmo_overbought")
    if not cfg.williams_oversold < cfg.williams_overbought:
        raise ValueError("expected williams_oversold < williams_overbought")
    if not cfg.ultimate_oversold < cfg.ultimate_overbought:
        raise ValueError("expected ultimate_oversold < ultimate_overbought")
    if not 0.0 < cfg.bb_squeeze_quantile < 1.0:
        raise ValueError("bb_squeeze_quantile must be between 0 and 1")
    positives = [
        cfg.bb_std,
        cfg.adx_trend_min,
        cfg.aroon_trend_min,
        cfg.atr_expansion_mult,
        cfg.keltner_atr_mult,
        cfg.zscore_extreme,
        cfg.volume_breakout_mult,
        cfg.score_signal_cap,
        cfg.weighted_score_cap,
        cfg.weighted_family_score_cap,
    ]
    if any(value <= 0.0 for value in positives):
        raise ValueError("thresholds and multipliers must be positive")
    if not 0.0 <= cfg.recent_score_weight <= 1.0:
        raise ValueError("recent_score_weight must be between 0 and 1")
    if cfg.min_confluence_signals < 1:
        raise ValueError("min_confluence_signals must be at least 1")
    if cfg.family_score_cap <= 0.0:
        raise ValueError("family_score_cap must be positive")
    bounded = [
        cfg.context_full_min,
        cfg.context_full_margin,
        cfg.context_soft_min,
        cfg.context_soft_margin,
    ]
    if any(value < 0.0 or value > 1.0 for value in bounded):
        raise ValueError("context thresholds must be between 0 and 1")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _add_pair(
    long: dict[str, Series],
    short: dict[str, Series],
    name: str,
    long_condition: Series,
    short_condition: Series,
) -> None:
    long[name] = _flag(long_condition)
    short[name] = _flag(short_condition)


def _flag(condition: Series) -> Series:
    return pd.Series(condition).astype("boolean").fillna(False).astype("int8")


def _crossed_above(left: Series, right: Series | float) -> Series:
    rhs = _series_like(right, left.index)
    return left.gt(rhs) & left.shift(1).le(rhs.shift(1))


def _crossed_below(left: Series, right: Series | float) -> Series:
    rhs = _series_like(right, left.index)
    return left.lt(rhs) & left.shift(1).ge(rhs.shift(1))


def _series_like(value: Series | float, index: pd.Index) -> Series:
    if isinstance(value, Series):
        return pd.to_numeric(value, errors="coerce").reindex(index)
    return pd.Series(float(value), index=index, dtype="float64")


def _ema(series: Series, window: int) -> Series:
    return series.ewm(span=window, min_periods=window, adjust=False).mean()


def _sma(series: Series, window: int) -> Series:
    return series.rolling(window, min_periods=window).mean()


def _dema(series: Series, window: int) -> Series:
    ema1 = _ema(series, window)
    ema2 = _ema(ema1, window)
    return (2.0 * ema1) - ema2


def _tema(series: Series, window: int) -> Series:
    ema1 = _ema(series, window)
    ema2 = _ema(ema1, window)
    ema3 = _ema(ema2, window)
    return (3.0 * ema1) - (3.0 * ema2) + ema3


def _wma(series: Series, window: int) -> Series:
    if len(series) < window:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    weights = np.arange(1.0, float(window) + 1.0)
    weighted = np.correlate(clean, weights, mode="valid")
    valid_weight = np.correlate(valid.astype("float64"), weights, mode="valid")
    required_weight = weights.sum()
    result = np.full(len(values), np.nan, dtype="float64")
    result[window - 1 :] = np.where(valid_weight >= required_weight, weighted / required_weight, np.nan)
    return pd.Series(result, index=series.index)


def _hma(series: Series, window: int) -> Series:
    half = max(2, window // 2)
    root = max(2, int(round(sqrt(window))))
    return _wma((2.0 * _wma(series, half)) - _wma(series, window), root)


def _rsi(series: Series, window: int) -> Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: Series, fast: int, slow: int, signal: int) -> tuple[Series, Series, Series]:
    line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(line, signal)
    return line, signal_line, line - signal_line


def _ppo(series: Series, fast: int, slow: int, signal: int) -> tuple[Series, Series, Series]:
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    line = 100.0 * _safe_div(fast_ema - slow_ema, slow_ema)
    signal_line = _ema(line, signal)
    return line, signal_line, line - signal_line


def _bollinger(series: Series, window: int, std_mult: float) -> tuple[Series, Series, Series, Series, Series]:
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + (std * std_mult)
    lower = mid - (std * std_mult)
    width = _safe_div(upper - lower, mid.abs())
    percent_b = _safe_div(series - lower, upper - lower)
    return mid, upper, lower, width, percent_b


def _stochastic(high: Series, low: Series, close: Series, period: int, smooth: int) -> tuple[Series, Series]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    k = 100.0 * _safe_div(close - lowest, highest - lowest)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def _stoch_rsi(rsi: Series, period: int, smooth: int) -> tuple[Series, Series]:
    lowest = rsi.rolling(period, min_periods=period).min()
    highest = rsi.rolling(period, min_periods=period).max()
    k = 100.0 * _safe_div(rsi - lowest, highest - lowest)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def _dmi(frame: DataFrame, period: int) -> dict[str, Series]:
    high = _num(frame, "high")
    low = _num(frame, "low")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index, dtype="float64")
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index, dtype="float64")
    atr = _true_range(frame).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * _safe_div(plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean(), atr)
    minus_di = 100.0 * _safe_div(minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean(), atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def _cci(high: Series, low: Series, close: Series, window: int) -> Series:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(window, min_periods=window).mean()
    mad = _rolling_mad(typical, window)
    return _safe_div(typical - mean, 0.015 * mad)


def _rolling_mad(series: Series, window: int) -> Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    result = np.full(len(values), np.nan, dtype="float64")
    if len(values) < window:
        return pd.Series(result, index=series.index)
    windows = sliding_window_view(values, window)
    valid = np.isfinite(windows).all(axis=1)
    means = windows.mean(axis=1)
    mad = np.abs(windows - means[:, None]).mean(axis=1)
    out = np.where(valid, mad, np.nan)
    result[window - 1 :] = out
    return pd.Series(result, index=series.index)


def _mfi(high: Series, low: Series, close: Series, volume: Series, window: int) -> Series:
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    direction = typical.diff()
    positive = money_flow.where(direction.gt(0.0), 0.0)
    negative = money_flow.where(direction.lt(0.0), 0.0).abs()
    ratio = _safe_div(positive.rolling(window, min_periods=window).sum(), negative.rolling(window, min_periods=window).sum())
    return 100.0 - (100.0 / (1.0 + ratio))


def _cmo(close: Series, window: int) -> Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window, min_periods=window).sum()
    losses = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).sum()
    return 100.0 * _safe_div(gains - losses, gains + losses)


def _williams_r(high: Series, low: Series, close: Series, window: int) -> Series:
    highest = high.rolling(window, min_periods=window).max()
    lowest = low.rolling(window, min_periods=window).min()
    return -100.0 * _safe_div(highest - close, highest - lowest)


def _aroon(high: Series, low: Series, window: int) -> tuple[Series, Series]:
    high_values = pd.to_numeric(high, errors="coerce").to_numpy(dtype="float64")
    low_values = pd.to_numeric(low, errors="coerce").to_numpy(dtype="float64")
    up = np.full(len(high_values), np.nan, dtype="float64")
    down = np.full(len(low_values), np.nan, dtype="float64")
    if len(high_values) >= window:
        high_windows = sliding_window_view(high_values, window)
        low_windows = sliding_window_view(low_values, window)
        high_valid = np.isfinite(high_windows).all(axis=1)
        low_valid = np.isfinite(low_windows).all(axis=1)
        high_pos = np.argmax(high_windows, axis=1)
        low_pos = np.argmin(low_windows, axis=1)
        up[window - 1 :] = np.where(high_valid, 100.0 * (high_pos + 1.0) / float(window), np.nan)
        down[window - 1 :] = np.where(low_valid, 100.0 * (low_pos + 1.0) / float(window), np.nan)
    return pd.Series(up, index=high.index), pd.Series(down, index=low.index)


def _trix(close: Series, window: int, signal: int) -> tuple[Series, Series]:
    triple = _ema(_ema(_ema(close, window), window), window)
    trix = triple.pct_change() * 100.0
    return trix, _ema(trix, signal)


def _ultimate_oscillator(high: Series, low: Series, close: Series, short: int, medium: int, long: int) -> Series:
    prev_close = close.shift(1)
    low_or_close = pd.concat([low, prev_close], axis=1).min(axis=1)
    high_or_close = pd.concat([high, prev_close], axis=1).max(axis=1)
    buying_pressure = close - low_or_close
    true_range = high_or_close - low_or_close
    avg_short = _safe_div(buying_pressure.rolling(short, min_periods=short).sum(), true_range.rolling(short, min_periods=short).sum())
    avg_medium = _safe_div(buying_pressure.rolling(medium, min_periods=medium).sum(), true_range.rolling(medium, min_periods=medium).sum())
    avg_long = _safe_div(buying_pressure.rolling(long, min_periods=long).sum(), true_range.rolling(long, min_periods=long).sum())
    return 100.0 * ((4.0 * avg_short) + (2.0 * avg_medium) + avg_long) / 7.0


def _awesome_oscillator(high: Series, low: Series) -> Series:
    median = (high + low) / 2.0
    return median.rolling(5, min_periods=5).mean() - median.rolling(34, min_periods=34).mean()


def _kst(close: Series, signal: int) -> tuple[Series, Series]:
    roc10 = close.pct_change(10) * 100.0
    roc15 = close.pct_change(15) * 100.0
    roc20 = close.pct_change(20) * 100.0
    roc30 = close.pct_change(30) * 100.0
    kst = (
        roc10.rolling(10, min_periods=10).mean()
        + 2.0 * roc15.rolling(10, min_periods=10).mean()
        + 3.0 * roc20.rolling(10, min_periods=10).mean()
        + 4.0 * roc30.rolling(15, min_periods=15).mean()
    )
    return kst, kst.rolling(signal, min_periods=signal).mean()


def _rolling_vwap(typical: Series, volume: Series, window: int) -> Series:
    return _safe_div((typical * volume).rolling(window, min_periods=1).sum(), volume.rolling(window, min_periods=1).sum())


def _obv(close: Series, volume: Series) -> Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).fillna(0.0).cumsum()


def _cmf(high: Series, low: Series, close: Series, volume: Series, window: int) -> Series:
    multiplier = _safe_div((close - low) - (high - close), high - low).fillna(0.0)
    money_flow_volume = multiplier * volume
    return _safe_div(money_flow_volume.rolling(window, min_periods=window).sum(), volume.rolling(window, min_periods=window).sum())


def _adl(high: Series, low: Series, close: Series, volume: Series) -> Series:
    multiplier = _safe_div((close - low) - (high - close), high - low).fillna(0.0)
    return (multiplier * volume).fillna(0.0).cumsum()


def _efi(close: Series, volume: Series, window: int) -> Series:
    force = close.diff() * volume
    return _ema(force, window)


def _ichimoku(high: Series, low: Series, tenkan: int, kijun: int, span_b: int) -> dict[str, Series]:
    tenkan_line = (high.rolling(tenkan, min_periods=tenkan).max() + low.rolling(tenkan, min_periods=tenkan).min()) / 2.0
    kijun_line = (high.rolling(kijun, min_periods=kijun).max() + low.rolling(kijun, min_periods=kijun).min()) / 2.0
    span_a = (tenkan_line + kijun_line) / 2.0
    span_b_line = (high.rolling(span_b, min_periods=span_b).max() + low.rolling(span_b, min_periods=span_b).min()) / 2.0
    return {"tenkan": tenkan_line, "kijun": kijun_line, "span_a": span_a, "span_b": span_b_line}


def _zscore(series: Series, window: int) -> Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return _safe_div(series - mean, std)


def _true_range(frame: DataFrame) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
