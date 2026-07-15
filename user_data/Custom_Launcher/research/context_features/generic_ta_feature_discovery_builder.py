from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib
import talib.abstract as ta
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = USER_DATA_DIR / "data" / "binance"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "feature_discovery"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generic TA/candle/relationship feature-discovery candidates.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_generic_ta"))
    parser.add_argument("--pair", default="BTC_USDT")
    parser.add_argument("--timeframes", default="1h,4h,1d")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "pair": args.pair,
        "timeframes": timeframes,
        "purpose": "Build broad generic TA/candle/relationship features for screening before promotion into trader hypotheses.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    frames = {tf: read_ohlcv(args.data_dir / f"{args.pair}-{tf}.feather") for tf in timeframes}
    base = frames["1h"][["date", "open", "high", "low", "close", "volume"]].copy()
    features = DataFrame({"date": base["date"]})
    dictionary_rows: list[dict[str, Any]] = []

    for tf, frame in frames.items():
        tf_features, tf_dictionary = build_tf_features(frame, tf)
        if tf == "1h":
            aligned = tf_features
        else:
            aligned = pd.merge_asof(
                base[["date"]].sort_values("date"),
                tf_features.sort_values("date"),
                on="date",
                direction="backward",
            )
        aligned = aligned.drop(columns=["date"], errors="ignore")
        features = pd.concat([features, aligned.reset_index(drop=True)], axis=1)
        dictionary_rows.extend(tf_dictionary)

    targets = build_targets(base)
    features = pd.concat([features, targets], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    parquet_path = args.output_dir / f"feature_discovery_candidates_{tag}.parquet"
    dictionary_path = args.output_dir / f"feature_discovery_candidates_{tag}_dictionary.csv"
    meta_path = args.output_dir / f"feature_discovery_candidates_{tag}_meta.json"
    features.to_parquet(parquet_path, index=False)
    pd.DataFrame(dictionary_rows).to_csv(dictionary_path, index=False)
    meta = {
        **plan,
        "rows": int(len(features)),
        "feature_columns": int(sum(column.startswith("fd__") for column in features.columns)),
        "columns": int(len(features.columns)),
        "start": str(features["date"].min()),
        "end": str(features["date"].max()),
        "outputs": {"parquet": str(parquet_path), "dictionary": str(dictionary_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parquet": str(parquet_path), "dictionary": str(dictionary_path), "meta": str(meta_path), **meta}, indent=2))
    return 0


def read_ohlcv(path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_tf_features(frame: DataFrame, tf: str) -> tuple[DataFrame, list[dict[str, Any]]]:
    out = DataFrame({"date": frame["date"]})
    dictionary: list[dict[str, Any]] = []
    inputs = frame[["open", "high", "low", "close", "volume"]].copy()

    raw: dict[str, Series] = {}
    for period in (5, 8, 10, 14, 20, 21, 30, 50, 100, 200):
        add_indicator(raw, f"rsi_{period}", lambda p=period: ta.RSI(inputs, timeperiod=p))
        add_indicator(raw, f"ema_{period}", lambda p=period: ta.EMA(inputs, timeperiod=p))
        add_indicator(raw, f"sma_{period}", lambda p=period: ta.SMA(inputs, timeperiod=p))
        add_indicator(raw, f"tema_{period}", lambda p=period: ta.TEMA(inputs, timeperiod=p))
        add_indicator(raw, f"atr_{period}", lambda p=period: ta.ATR(inputs, timeperiod=p))
        add_indicator(raw, f"roc_{period}", lambda p=period: ta.ROC(inputs, timeperiod=p))

    add_indicator(raw, "adx_14", lambda: ta.ADX(inputs, timeperiod=14))
    add_indicator(raw, "plus_di_14", lambda: ta.PLUS_DI(inputs, timeperiod=14))
    add_indicator(raw, "minus_di_14", lambda: ta.MINUS_DI(inputs, timeperiod=14))
    add_indicator(raw, "cci_20", lambda: ta.CCI(inputs, timeperiod=20))
    add_indicator(raw, "mfi_14", lambda: ta.MFI(inputs, timeperiod=14))
    add_indicator(raw, "willr_14", lambda: ta.WILLR(inputs, timeperiod=14))
    add_indicator(raw, "obv", lambda: ta.OBV(inputs))
    add_indicator(raw, "adosc", lambda: ta.ADOSC(inputs))
    add_indicator(raw, "trix_30", lambda: ta.TRIX(inputs, timeperiod=30))
    add_indicator(raw, "ultosc", lambda: ta.ULTOSC(inputs))
    add_indicator(raw, "cmo_14", lambda: ta.CMO(inputs, timeperiod=14))
    add_indicator(raw, "natr_14", lambda: ta.NATR(inputs, timeperiod=14))
    add_indicator(raw, "ppo", lambda: ta.PPO(inputs))
    macd = ta.MACD(inputs)
    for column in macd.columns:
        raw[f"macd_{column.lower()}"] = macd[column]
    stoch = ta.STOCH(inputs)
    for column in stoch.columns:
        raw[f"stoch_{column.lower()}"] = stoch[column]
    bbands = ta.BBANDS(inputs, timeperiod=20)
    for column in bbands.columns:
        raw[f"bbands_{column.lower()}"] = bbands[column]

    candle_scores = build_candle_features(inputs)
    raw.update(candle_scores)
    relationships = build_relationships(frame, raw)
    raw.update(relationships)

    columns: dict[str, Series] = {}
    for name, series in raw.items():
        feature = f"fd__generic_ta_{tf}__{safe_name(name)}"
        columns[feature] = pd.to_numeric(series, errors="coerce").astype("float32")
        dictionary.append(
            {
                "feature": feature,
                "base_column": name,
                "transform": "generic_ta_state",
                "source_family": "generic_ta",
                "source_detail": f"generic_ta_{tf}",
                "plain_english": describe_feature(tf, name),
            }
        )
    return pd.concat([out, DataFrame(columns)], axis=1), dictionary


def add_indicator(raw: dict[str, Series], name: str, fn: Any) -> None:
    try:
        value = fn()
    except Exception:
        return
    if isinstance(value, DataFrame):
        for column in value.columns:
            raw[f"{name}_{str(column).lower()}"] = value[column]
    else:
        raw[name] = value


def build_candle_features(inputs: DataFrame) -> dict[str, Series]:
    raw: dict[str, Series] = {}
    patterns = talib.get_function_groups().get("Pattern Recognition", [])
    bullish_count = pd.Series(0.0, index=inputs.index)
    bearish_count = pd.Series(0.0, index=inputs.index)
    for pattern in patterns:
        try:
            values = getattr(ta, pattern)(inputs)
        except Exception:
            continue
        signed = pd.to_numeric(values, errors="coerce").fillna(0.0) / 100.0
        name = pattern.lower().replace("cdl", "candle_")
        raw[name] = signed
        bullish_count = bullish_count + signed.gt(0).astype(float)
        bearish_count = bearish_count + signed.lt(0).astype(float)
    raw["bullish_candle_pattern_count"] = bullish_count
    raw["bearish_candle_pattern_count"] = bearish_count
    for window in (3, 6, 12):
        raw[f"bullish_candle_patterns_{window}bar"] = bullish_count.rolling(window, min_periods=1).sum()
        raw[f"bearish_candle_patterns_{window}bar"] = bearish_count.rolling(window, min_periods=1).sum()
    return raw


def build_relationships(frame: DataFrame, raw: dict[str, Series]) -> dict[str, Series]:
    out: dict[str, Series] = {}
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    body = (frame["close"] - frame["open"]) / (frame["high"] - frame["low"]).replace(0.0, np.nan)
    out["body_pressure"] = body.clip(-1.0, 1.0)
    out["higher_high_3bar"] = high.gt(high.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)
    out["higher_low_3bar"] = low.gt(low.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)
    out["lower_high_3bar"] = high.lt(high.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)
    out["lower_low_3bar"] = low.lt(low.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)
    out["volume_z_24"] = zscore(volume, 24, 12)
    out["volume_z_168"] = zscore(volume, 168, 48)

    for fast, slow in ((8, 21), (10, 30), (20, 50), (50, 200)):
        ema_fast = raw.get(f"ema_{fast}")
        ema_slow = raw.get(f"ema_{slow}")
        sma_slow = raw.get(f"sma_{slow}")
        if ema_fast is not None and ema_slow is not None:
            out[f"ema_{fast}_above_ema_{slow}"] = ema_fast.gt(ema_slow).astype(float)
            out[f"ema_{fast}_cross_above_ema_{slow}"] = cross_above(ema_fast, ema_slow)
            out[f"ema_{fast}_cross_below_ema_{slow}"] = cross_below(ema_fast, ema_slow)
        if ema_fast is not None and sma_slow is not None:
            out[f"ema_{fast}_above_sma_{slow}"] = ema_fast.gt(sma_slow).astype(float)

    macd = raw.get("macd_macd")
    macdsignal = raw.get("macd_macdsignal")
    macdhist = raw.get("macd_macdhist")
    if macd is not None and macdsignal is not None:
        out["macd_above_signal"] = macd.gt(macdsignal).astype(float)
        out["macd_cross_above_signal"] = cross_above(macd, macdsignal)
        out["macd_cross_below_signal"] = cross_below(macd, macdsignal)
    if macdhist is not None:
        out["macd_hist_rising_3bar"] = macdhist.diff().gt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)

    rsi = raw.get("rsi_14")
    if rsi is not None:
        out["rsi_14_oversold_reclaim"] = (rsi.shift(1).lt(30) & rsi.ge(30)).astype(float)
        out["rsi_14_overbought_loss"] = (rsi.shift(1).gt(70) & rsi.le(70)).astype(float)
        out["rsi_14_rising_3bar"] = rsi.diff().gt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)

    bb_upper = raw.get("bbands_upperband")
    bb_middle = raw.get("bbands_middleband")
    bb_lower = raw.get("bbands_lowerband")
    if bb_upper is not None and bb_lower is not None and bb_middle is not None:
        width = (bb_upper - bb_lower) / bb_middle.replace(0.0, np.nan)
        out["bb_width"] = width
        out["bb_squeeze_120"] = width.le(width.rolling(120, min_periods=40).quantile(0.20)).astype(float)
        out["close_cross_above_bb_upper"] = cross_above(close, bb_upper)
        out["close_cross_below_bb_lower"] = cross_below(close, bb_lower)

    bull = raw.get("bullish_candle_patterns_3bar")
    bear = raw.get("bearish_candle_patterns_3bar")
    if bull is not None:
        out["three_bullish_patterns_plus_hh"] = bull.ge(3).astype(float) * out["higher_high_3bar"]
    if bear is not None:
        out["three_bearish_patterns_plus_ll"] = bear.ge(3).astype(float) * out["lower_low_3bar"]
    if bull is not None and macdhist is not None:
        out["bullish_patterns_macd_rising"] = bull.ge(2).astype(float) * out.get("macd_hist_rising_3bar", 0.0)
    if bear is not None and rsi is not None:
        out["bearish_patterns_rsi_overbought_loss"] = bear.ge(2).astype(float) * out["rsi_14_overbought_loss"]
    return out


def build_targets(frame: DataFrame) -> DataFrame:
    out = DataFrame(index=frame.index)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    prior_high_24h = high.shift(1).rolling(24, min_periods=12).max()
    prior_low_24h = low.shift(1).rolling(24, min_periods=12).min()
    future_close_6h = close.shift(-6)
    out["future_return_6h"] = future_close_6h / close - 1.0
    out["future_return_24h"] = close.shift(-24) / close - 1.0
    out["future_max_upside_6h"] = future_window_max(high, 6) / close - 1.0
    out["future_max_drawdown_6h"] = future_window_min(low, 6) / close - 1.0
    out["future_max_upside_24h"] = future_window_max(high, 24) / close - 1.0
    out["future_max_drawdown_24h"] = future_window_min(low, 24) / close - 1.0
    breakout_attempt_6h = future_window_max(high, 6).gt(prior_high_24h * 1.001)
    breakdown_attempt_6h = future_window_min(low, 6).lt(prior_low_24h * 0.999)
    out["breakout_success_next_6h"] = (breakout_attempt_6h & future_close_6h.gt(prior_high_24h * 1.001)).astype(float).where(prior_high_24h.notna())
    out["breakdown_success_next_6h"] = (breakdown_attempt_6h & future_close_6h.lt(prior_low_24h * 0.999)).astype(float).where(prior_low_24h.notna())
    out["large_drawdown_next_6h"] = out["future_max_drawdown_6h"].le(-0.03).astype(float).where(out["future_max_drawdown_6h"].notna())
    out["large_drawdown_next_24h"] = out["future_max_drawdown_24h"].le(-0.04).astype(float).where(out["future_max_drawdown_24h"].notna())
    out["hit_plus_3pct_before_minus_2pct"] = hit_before(high, low, close, up_pct=0.03, down_pct=-0.02, horizon=24)
    out["hit_minus_3pct_before_plus_2pct"] = hit_before(high, low, close, up_pct=0.02, down_pct=-0.03, horizon=24, downside=True)
    out["fakeout_next_24h"] = ((future_window_max(high, 6).gt(prior_high_24h * 1.001)) & (future_window_min(low, 24).lt(prior_high_24h * 0.995))).astype(float).where(prior_high_24h.notna())
    return out


def future_window_max(series: Series, horizon: int) -> Series:
    return pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1)


def future_window_min(series: Series, horizon: int) -> Series:
    return pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1).min(axis=1)


def hit_before(high: Series, low: Series, close: Series, *, up_pct: float, down_pct: float, horizon: int, downside: bool = False) -> Series:
    values: list[float] = []
    for idx in range(len(close)):
        base = close.iloc[idx]
        if not np.isfinite(base):
            values.append(np.nan)
            continue
        up_level = base * (1.0 + up_pct)
        down_level = base * (1.0 + down_pct)
        result = np.nan
        for step in range(1, horizon + 1):
            pos = idx + step
            if pos >= len(close):
                break
            up_hit = high.iloc[pos] >= up_level
            down_hit = low.iloc[pos] <= down_level
            if downside:
                if down_hit:
                    result = 1.0
                    break
                if up_hit:
                    result = 0.0
                    break
            else:
                if up_hit:
                    result = 1.0
                    break
                if down_hit:
                    result = 0.0
                    break
        values.append(result)
    return pd.Series(values, index=close.index)


def cross_above(left: Series, right: Series) -> Series:
    return (left.gt(right) & left.shift(1).le(right.shift(1))).astype(float)


def cross_below(left: Series, right: Series) -> Series:
    return (left.lt(right) & left.shift(1).ge(right.shift(1))).astype(float)


def zscore(series: Series, window: int, min_periods: int) -> Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (series - mean) / std


def describe_feature(tf: str, name: str) -> str:
    return f"{tf} generic technical/candle feature: {name.replace('_', ' ')}."


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()
    return clean[:140] or "feature"


if __name__ == "__main__":
    raise SystemExit(main())
