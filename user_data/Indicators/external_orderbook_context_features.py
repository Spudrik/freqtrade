from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import sqlite3

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


DEFAULT_MARKET_KEYS: tuple[str, ...] = (
    "binance_spot",
    "binance_usdm_futures",
    "bybit_spot",
    "bybit_linear",
)

DEFAULT_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("binance_spot", "binance_usdm_futures"),
    ("bybit_spot", "bybit_linear"),
    ("binance_usdm_futures", "bybit_linear"),
    ("binance_spot", "bybit_spot"),
)

BAR_NUMERIC_COLUMNS: tuple[str, ...] = (
    "valid_samples",
    "expected_samples",
    "spread_bps_mean",
    "spread_bps_max",
    "microprice_offset_bps_mean",
    "imbalance_top20_mean",
    "imbalance_top20_min",
    "imbalance_top20_max",
    "imbalance_10bps_mean",
    "imbalance_25bps_mean",
    "bid_pressure_seconds",
    "ask_pressure_seconds",
    "bid_pressure_ratio",
    "ask_pressure_ratio",
    "max_bid_pressure_streak_seconds",
    "max_ask_pressure_streak_seconds",
    "nearest_bid_wall_min_distance_bps",
    "nearest_ask_wall_min_distance_bps",
    "strongest_bid_wall_score",
    "strongest_ask_wall_score",
)

MARKET_CONTEXT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "funding_rate",
    "open_interest",
    "long_ratio",
    "short_ratio",
    "long_short_ratio",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_sell_ratio",
)


@dataclass(frozen=True)
class OrderbookContextFeatureConfig:
    """First-pass Order Book formatter and feature settings.

    Objective:
    Convert the launcher Order Book SQLite data into candle-aligned,
    no-trading-decision dataframe features for 1h strategy research.

    First-pass scope:
    - Use compact ``orderbook_metric_bars`` rows by default, not raw snapshots.
    - Keep gap handling explicit with coverage, missing ratio, gap flags, bar
      age, and stale-period columns.
    - Treat historical/backtest rows and live/dry-run rows differently.
    - Add market comparison features where both selected markets exist.
    - Optionally append derivative market-context rows already stored beside
      order book data: funding, open interest, long/short, and taker ratio.
    - Do not infer entries, exits, sizing, or execution simulation.

    Gap convention:
    News gaps can usually mean "no article arrived". Order book gaps mean
    "measurement unavailable". This module therefore does not forward-fill
    order book pressure/spread/imbalance values across missing candle bins.
    Missing bins get ``coverage_ratio=0`` and ``gap_flag=1``. Rolling features
    are accompanied by rolling coverage/gap ratios so another agent can reject
    low-quality periods instead of accidentally treating them as neutral.

    Relevant SQLite tables:
    ``orderbook_metric_bars(ts_start, ts_end, timeframe_seconds, stream_id,
    market_key, canonical_pair, valid_samples, expected_samples,
    spread_bps_mean, microprice_offset_bps_mean, imbalance_top20_mean,
    bid_pressure_ratio, ask_pressure_ratio, nearest_*_wall_*, ...)``

    ``market_context_ticks(ts, source_ts, market_key, canonical_pair,
    funding_rate, open_interest, long_ratio, short_ratio, long_short_ratio,
    taker_buy_sell_ratio, ...)``

    Agent notes:
    - The safest default is closed-bar alignment: bar ``ts_end`` is grouped into
      the candle bin it closed inside.
    - For backtest/hyperopt, pass a historical order-book SQLite path and keep
      ``data_mode="historical"``. Do not use the live collector DB in backtests
      unless it genuinely contains the historical range being tested.
    - For dry/live, use ``data_mode="auto"``, pass the strategy runmode, and set
      ``enable_live_stream=True``. This switches to the configured live bar
      timeframe and adds freshness checks.
    - Basis bps and depth ratio are not emitted here because the current bar
      schema does not store mid price or top-book notionals. Add those to the
      collector bar schema, or build a separate tick-resample formatter, before
      using basis/depth features in strategy research.
    - Validate coverage thresholds before using these columns in hyperopt.
    """

    db_path: str | Path | None = None
    canonical_pair: str | None = None
    market_keys: tuple[str, ...] = DEFAULT_MARKET_KEYS
    primary_market_key: str = "binance_usdm_futures"
    comparison_pairs: tuple[tuple[str, str], ...] = DEFAULT_COMPARISON_PAIRS
    data_mode: str = "historical"
    runmode: str | None = None
    enable_live_stream: bool = False
    resample_rule: str = "1h"
    bar_timeframe_seconds: int = 3600
    live_bar_timeframe_seconds: int = 300
    availability_lag_candles: int = 0
    short_window: int = 6
    medium_window: int = 24
    long_window: int = 72
    persistence_window: int = 24
    min_coverage_ratio: float = 0.80
    pressure_threshold: float = 0.35
    wall_near_threshold_bps: float = 10.0
    microprice_scale_bps: float = 5.0
    spread_penalty_bps: float = 10.0
    max_bar_age_seconds: int | None = None
    live_max_bar_age_seconds: int = 900
    include_market_context: bool = True
    market_context_max_age_periods: int = 8
    prefix: str = "obctx"
    allow_missing: bool = False


def add_orderbook_context_features(
    dataframe: DataFrame,
    pair: str | None = None,
    config: OrderbookContextFeatureConfig | None = None,
    *,
    db_path: str | Path | None = None,
    canonical_pair: str | None = None,
    market_keys: tuple[str, ...] | None = None,
    primary_market_key: str | None = None,
    comparison_pairs: tuple[tuple[str, str], ...] | None = None,
    data_mode: str | None = None,
    runmode: str | None = None,
    enable_live_stream: bool | None = None,
    resample_rule: str | None = None,
    bar_timeframe_seconds: int | None = None,
    live_bar_timeframe_seconds: int | None = None,
    availability_lag_candles: int | None = None,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    persistence_window: int | None = None,
    min_coverage_ratio: float | None = None,
    pressure_threshold: float | None = None,
    wall_near_threshold_bps: float | None = None,
    microprice_scale_bps: float | None = None,
    spread_penalty_bps: float | None = None,
    max_bar_age_seconds: int | None = None,
    live_max_bar_age_seconds: int | None = None,
    include_market_context: bool | None = None,
    market_context_max_age_periods: int | None = None,
    prefix: str | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Append first-pass order book context features to a candle dataframe.

    Expected use in a strategy:
    ``dataframe = add_orderbook_context_features(dataframe, pair=metadata["pair"])``

    Output columns use the configured prefix, default ``obctx``:
    - ``obctx_mode_live`` / ``obctx_source_bar_timeframe_seconds``: confirms
      whether the formatter used historical or live-stream settings.
    - ``obctx_coverage_ratio`` and ``obctx_gap_flag``: primary market data
      quality for each candle.
    - ``obctx_gap_ratio_roll_medium``: how much of the recent window is missing
      or below coverage threshold.
    - ``obctx_spread_bps_roll_medium``: rolling primary-market spread.
    - ``obctx_spread_zscore_medium`` and spread widening/volatility columns:
      detect spread stress better than a simple average.
    - ``obctx_imbalance_roll_short/medium/long`` plus imbalance delta and
      volatility columns: track pressure direction and instability.
    - ``obctx_bid_pressure_persistence`` and
      ``obctx_ask_pressure_persistence``: sustained one-sided pressure.
    - ``obctx_bid_wall_near_persistence`` and
      ``obctx_ask_wall_near_persistence``: recurring nearby wall presence.
    - ``obctx_liquidity_stress_score`` and support/resistance pressure scores:
      compact diagnostics for later validation, not entry rules.
    - ``obctx_score_long/short/abs/state``: shared score contract derived only
      from order-book pressure, not a strategy decision.
    - ``obctx_<market>_*``: selected per-market raw/aligned bar features.
    - ``obctx_cmp_<market_a>_vs_<market_b>_*``: spread, imbalance, pressure,
      wall, and coverage divergence where both markets have usable rows.

    Order-book values are not forward-filled through missing candle bins.
    Coverage and gap columns should be used before trusting any rolling score.

    ``data_mode`` controls runtime behavior:
    - ``historical``: default for backtest/hyperopt; uses
      ``bar_timeframe_seconds`` and strict missing-measurement columns.
    - ``live``: requires ``enable_live_stream=True``; uses
      ``live_bar_timeframe_seconds`` unless a non-default bar timeframe is
      explicitly set, and applies live freshness checks.
    - ``auto``: live only when runmode looks like dry/live and live stream is
      enabled; otherwise historical.
    """

    cfg = _resolve_config(
        config,
        db_path=db_path,
        canonical_pair=canonical_pair,
        market_keys=market_keys,
        primary_market_key=primary_market_key,
        comparison_pairs=comparison_pairs,
        data_mode=data_mode,
        runmode=runmode,
        enable_live_stream=enable_live_stream,
        resample_rule=resample_rule,
        bar_timeframe_seconds=bar_timeframe_seconds,
        live_bar_timeframe_seconds=live_bar_timeframe_seconds,
        availability_lag_candles=availability_lag_candles,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        persistence_window=persistence_window,
        min_coverage_ratio=min_coverage_ratio,
        pressure_threshold=pressure_threshold,
        wall_near_threshold_bps=wall_near_threshold_bps,
        microprice_scale_bps=microprice_scale_bps,
        spread_penalty_bps=spread_penalty_bps,
        max_bar_age_seconds=max_bar_age_seconds,
        live_max_bar_age_seconds=live_max_bar_age_seconds,
        include_market_context=include_market_context,
        market_context_max_age_periods=market_context_max_age_periods,
        prefix=prefix,
        allow_missing=allow_missing,
    )
    cfg = _effective_mode_config(cfg)
    _validate_config(cfg)
    if dataframe.empty:
        return dataframe.copy()

    canonical = _resolve_canonical_pair(pair, cfg)
    frame = dataframe.copy()
    candle_times = _candle_times(frame)
    query_start, query_end = _query_bounds(candle_times, cfg)
    bars = _load_metric_bars(cfg, canonical, query_start, query_end)
    if bars.empty:
        if cfg.allow_missing:
            return _append_empty_features(frame, cfg)
        raise ValueError(
            f"No orderbook_metric_bars rows found for {canonical} at {cfg.bar_timeframe_seconds}s. "
            "Start the collector with matching bar_intervals_seconds before using this formatter."
        )

    aligned = _align_bars_by_market(bars, candle_times, frame.index, cfg)
    features = _build_orderbook_features(aligned, cfg)
    if cfg.include_market_context:
        context = _load_market_context(cfg, canonical, query_start, query_end)
        context_features = _align_market_context(context, candle_times, frame.index, cfg)
        features = pd.concat([features, context_features], axis=1)

    existing = [column for column in frame.columns if str(column).startswith(f"{cfg.prefix}_")]
    if existing:
        frame = frame.drop(columns=existing)
    return pd.concat([frame, features], axis=1)


def format_orderbook_bars(
    pair: str | None = None,
    config: OrderbookContextFeatureConfig | None = None,
    *,
    db_path: str | Path | None = None,
    canonical_pair: str | None = None,
    market_keys: tuple[str, ...] | None = None,
    bar_timeframe_seconds: int | None = None,
    data_mode: str | None = None,
    runmode: str | None = None,
    enable_live_stream: bool | None = None,
    live_bar_timeframe_seconds: int | None = None,
    live_max_bar_age_seconds: int | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Load compact order-book bar rows with coverage columns.

    This is the inspection/formatter function. It returns raw bar rows filtered
    to one canonical pair and selected markets, with ``coverage_ratio`` and
    ``missing_ratio`` added. It does not align to a strategy dataframe.
    """

    cfg = _resolve_config(
        config,
        db_path=db_path,
        canonical_pair=canonical_pair,
        market_keys=market_keys,
        bar_timeframe_seconds=bar_timeframe_seconds,
        data_mode=data_mode,
        runmode=runmode,
        enable_live_stream=enable_live_stream,
        live_bar_timeframe_seconds=live_bar_timeframe_seconds,
        live_max_bar_age_seconds=live_max_bar_age_seconds,
        allow_missing=allow_missing,
    )
    cfg = _effective_mode_config(cfg)
    _validate_config(cfg)
    canonical = _resolve_canonical_pair(pair, cfg)
    bars = _load_metric_bars(cfg, canonical, None, None)
    if bars.empty:
        return bars
    return _prepare_bars(bars, cfg)


def _resolve_config(config: OrderbookContextFeatureConfig | None, **overrides: Any) -> OrderbookContextFeatureConfig:
    cfg = config or OrderbookContextFeatureConfig()
    values = {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return OrderbookContextFeatureConfig(**values)


def _effective_mode_config(cfg: OrderbookContextFeatureConfig) -> OrderbookContextFeatureConfig:
    mode = _effective_data_mode(cfg)
    if mode != "live":
        return cfg
    values: dict[str, Any] = {}
    if int(cfg.bar_timeframe_seconds) == 3600:
        values["bar_timeframe_seconds"] = int(cfg.live_bar_timeframe_seconds)
    if cfg.max_bar_age_seconds is None:
        values["max_bar_age_seconds"] = int(cfg.live_max_bar_age_seconds)
    if int(cfg.availability_lag_candles) > 0:
        values["availability_lag_candles"] = 0
    return replace(cfg, **values) if values else cfg


def _effective_data_mode(cfg: OrderbookContextFeatureConfig) -> str:
    mode = str(cfg.data_mode or "historical").strip().lower()
    if mode in {"dry_run", "live_run", "livestream", "live_stream"}:
        mode = "live"
    if mode == "backtest" or mode == "hyperopt":
        mode = "historical"
    if mode != "auto":
        return mode
    runmode = str(cfg.runmode or "").strip().lower()
    if cfg.enable_live_stream and runmode in {"dry_run", "dry-run", "live", "live_run", "live-run"}:
        return "live"
    return "historical"


def _validate_config(cfg: OrderbookContextFeatureConfig) -> None:
    mode = _effective_data_mode(cfg)
    if mode not in {"historical", "live"}:
        raise ValueError("Orderbook data_mode must be 'historical', 'live', or 'auto'.")
    if mode == "live" and not cfg.enable_live_stream:
        raise ValueError("Orderbook live mode requires enable_live_stream=True.")
    if not cfg.market_keys:
        raise ValueError("Orderbook market_keys must not be empty.")
    if cfg.primary_market_key not in cfg.market_keys:
        raise ValueError("Orderbook primary_market_key must be included in market_keys.")
    if cfg.bar_timeframe_seconds < 1:
        raise ValueError("Orderbook bar_timeframe_seconds must be >= 1.")
    if cfg.live_bar_timeframe_seconds < 1:
        raise ValueError("Orderbook live_bar_timeframe_seconds must be >= 1.")
    if cfg.availability_lag_candles < 0:
        raise ValueError("Orderbook availability_lag_candles must be >= 0.")
    if cfg.short_window < 1 or cfg.medium_window < 1 or cfg.long_window < 1:
        raise ValueError("Orderbook rolling windows must be >= 1.")
    if cfg.persistence_window < 1:
        raise ValueError("Orderbook persistence_window must be >= 1.")
    if not 0.0 <= cfg.min_coverage_ratio <= 1.0:
        raise ValueError("Orderbook min_coverage_ratio must be normalized 0..1.")
    if cfg.pressure_threshold < 0:
        raise ValueError("Orderbook pressure_threshold must be >= 0.")
    if cfg.wall_near_threshold_bps < 0:
        raise ValueError("Orderbook wall_near_threshold_bps must be >= 0.")
    if cfg.microprice_scale_bps <= 0 or cfg.spread_penalty_bps <= 0:
        raise ValueError("Orderbook microprice/spread scales must be > 0.")
    if cfg.max_bar_age_seconds is not None and cfg.max_bar_age_seconds < 0:
        raise ValueError("Orderbook max_bar_age_seconds must be >= 0 when provided.")
    if cfg.live_max_bar_age_seconds < 0:
        raise ValueError("Orderbook live_max_bar_age_seconds must be >= 0.")
    if cfg.market_context_max_age_periods < 1:
        raise ValueError("Orderbook market_context_max_age_periods must be >= 1.")
    if not cfg.prefix:
        raise ValueError("Orderbook prefix must not be empty.")


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "orderbook_data" / "live" / "orderbook_events.sqlite"


def _resolved_db_path(cfg: OrderbookContextFeatureConfig) -> Path:
    return Path(cfg.db_path).expanduser() if cfg.db_path else _default_db_path()


def _resolve_canonical_pair(pair: str | None, cfg: OrderbookContextFeatureConfig) -> str:
    raw = cfg.canonical_pair or pair
    canonical = _canonical_pair(raw)
    if not canonical:
        raise ValueError("Orderbook features require pair='BTC/USDT:USDT' or config.canonical_pair='BTC/USDT'.")
    return canonical


def _canonical_pair(pair: str | None) -> str | None:
    raw = str(pair or "").strip().upper()
    if not raw or "/" not in raw:
        return None
    base, quote_part = raw.split("/", 1)
    quote = quote_part.split(":", 1)[0]
    base = base.strip()
    quote = quote.strip()
    if not base or not quote:
        return None
    return f"{base}/{quote}"


def _candle_times(dataframe: DataFrame) -> pd.DatetimeIndex:
    if "date" in dataframe.columns:
        values = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    elif isinstance(dataframe.index, pd.DatetimeIndex):
        values = pd.to_datetime(dataframe.index, utc=True, errors="coerce")
    else:
        raise ValueError("Orderbook features require a 'date' column or a DatetimeIndex.")
    if values.isna().any():
        raise ValueError("Orderbook features found invalid candle timestamps.")
    return pd.DatetimeIndex(values)


def _query_bounds(candle_times: pd.DatetimeIndex, cfg: OrderbookContextFeatureConfig) -> tuple[str, str]:
    lookback_periods = max(cfg.long_window, cfg.persistence_window) + cfg.availability_lag_candles + 2
    lookback = pd.Timedelta(seconds=cfg.bar_timeframe_seconds * lookback_periods)
    start = candle_times.min() - lookback
    end = candle_times.max() + pd.Timedelta(seconds=cfg.bar_timeframe_seconds)
    return start.isoformat(), end.isoformat()


def _load_metric_bars(
    cfg: OrderbookContextFeatureConfig,
    canonical_pair: str,
    query_start: str | None,
    query_end: str | None,
) -> DataFrame:
    db_path = _resolved_db_path(cfg)
    if not db_path.exists():
        if cfg.allow_missing:
            return DataFrame()
        raise FileNotFoundError(f"Orderbook SQLite not found: {db_path}")

    where = [
        "canonical_pair = ?",
        "timeframe_seconds = ?",
    ]
    params: list[Any] = [canonical_pair, int(cfg.bar_timeframe_seconds)]
    if cfg.market_keys:
        placeholders = ", ".join("?" for _ in cfg.market_keys)
        where.append(f"market_key IN ({placeholders})")
        params.extend(cfg.market_keys)
    if query_start is not None:
        where.append("ts_end >= ?")
        params.append(query_start)
    if query_end is not None:
        where.append("ts_end <= ?")
        params.append(query_end)

    query = f"""
        SELECT
            id,
            ts_start,
            ts_end,
            timeframe_seconds,
            stream_id,
            market_key,
            venue,
            exchange,
            market_type,
            margin_type,
            quote_asset,
            canonical_pair,
            pair,
            symbol,
            valid_samples,
            expected_samples,
            spread_bps_mean,
            spread_bps_max,
            microprice_offset_bps_mean,
            imbalance_top20_mean,
            imbalance_top20_min,
            imbalance_top20_max,
            imbalance_10bps_mean,
            imbalance_25bps_mean,
            bid_pressure_seconds,
            ask_pressure_seconds,
            bid_pressure_ratio,
            ask_pressure_ratio,
            max_bid_pressure_streak_seconds,
            max_ask_pressure_streak_seconds,
            nearest_bid_wall_min_distance_bps,
            nearest_ask_wall_min_distance_bps,
            strongest_bid_wall_score,
            strongest_ask_wall_score
        FROM orderbook_metric_bars
        WHERE {" AND ".join(where)}
        ORDER BY ts_end, market_key, id
    """

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        return pd.read_sql_query(query, conn, params=params)
    except sqlite3.Error as exc:
        if cfg.allow_missing:
            return DataFrame()
        raise RuntimeError(f"Could not read Orderbook SQLite: {db_path}") from exc
    finally:
        if conn is not None:
            conn.close()


def _prepare_bars(bars: DataFrame, cfg: OrderbookContextFeatureConfig) -> DataFrame:
    prepared = bars.copy()
    prepared["ts_start"] = pd.to_datetime(prepared["ts_start"], utc=True, errors="coerce")
    prepared["ts_end"] = pd.to_datetime(prepared["ts_end"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["ts_end"])
    for column in BAR_NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["expected_samples"] = prepared["expected_samples"].replace(0, np.nan)
    prepared["coverage_ratio"] = (prepared["valid_samples"] / prepared["expected_samples"]).clip(0.0, 1.0)
    prepared["coverage_ratio"] = prepared["coverage_ratio"].fillna(0.0)
    prepared["missing_ratio"] = (1.0 - prepared["coverage_ratio"]).clip(0.0, 1.0)
    prepared["gap_flag"] = prepared["coverage_ratio"].lt(cfg.min_coverage_ratio).astype(float)
    return prepared.sort_values(["ts_end", "market_key", "id"])


def _align_bars_by_market(
    bars: DataFrame,
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: OrderbookContextFeatureConfig,
) -> dict[str, DataFrame]:
    prepared = _prepare_bars(bars, cfg)
    aligned: dict[str, DataFrame] = {}
    for market_key in cfg.market_keys:
        market_rows = prepared.loc[prepared["market_key"].eq(market_key)].copy()
        if market_rows.empty:
            aligned[market_key] = _empty_market_frame(candle_times, output_index, cfg)
            continue
        market_rows["bar_observed_at"] = market_rows["ts_end"]
        market_rows = market_rows.set_index("ts_end").sort_index()
        resampled = market_rows.resample(cfg.resample_rule, label="right", closed="right").last()
        market_aligned = resampled.reindex(candle_times)
        if cfg.availability_lag_candles:
            market_aligned = market_aligned.shift(cfg.availability_lag_candles)
        candle_series = Series(candle_times, index=market_aligned.index)
        market_aligned["bar_age_seconds"] = (
            candle_series - pd.to_datetime(market_aligned["bar_observed_at"], utc=True, errors="coerce")
        ).dt.total_seconds()
        market_aligned.index = output_index
        market_aligned = _finalize_aligned_market(market_aligned, cfg)
        aligned[market_key] = market_aligned
    return aligned


def _empty_market_frame(
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: OrderbookContextFeatureConfig,
) -> DataFrame:
    frame = DataFrame(index=candle_times)
    for column in BAR_NUMERIC_COLUMNS:
        frame[column] = np.nan
    frame["coverage_ratio"] = 0.0
    frame["missing_ratio"] = 1.0
    frame["gap_flag"] = 1.0
    frame["bar_age_seconds"] = np.nan
    frame.index = output_index
    return frame


def _finalize_aligned_market(frame: DataFrame, cfg: OrderbookContextFeatureConfig) -> DataFrame:
    output = frame.copy()
    for column in BAR_NUMERIC_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["coverage_ratio"] = pd.to_numeric(output.get("coverage_ratio"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    output["missing_ratio"] = (1.0 - output["coverage_ratio"]).clip(0.0, 1.0)
    age = pd.to_numeric(output.get("bar_age_seconds"), errors="coerce")
    output["bar_age_seconds"] = age
    too_old = Series(False, index=output.index)
    if cfg.max_bar_age_seconds is not None:
        too_old = age.gt(cfg.max_bar_age_seconds).fillna(True)
    missing = output["coverage_ratio"].le(0.0)
    low_coverage = output["coverage_ratio"].lt(cfg.min_coverage_ratio)
    output["gap_flag"] = (missing | low_coverage | too_old).astype(float)
    output["stale_periods"] = _stale_periods(output["gap_flag"].eq(0.0))
    return output


def _build_orderbook_features(aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> DataFrame:
    p = cfg.prefix
    features = DataFrame(index=next(iter(aligned.values())).index)
    primary = aligned[cfg.primary_market_key]
    valid_primary = primary["gap_flag"].eq(0.0)
    mode = _effective_data_mode(cfg)

    features[f"{p}_mode_live"] = 1.0 if mode == "live" else 0.0
    features[f"{p}_live_stream_enabled"] = 1.0 if cfg.enable_live_stream else 0.0
    features[f"{p}_source_bar_timeframe_seconds"] = float(cfg.bar_timeframe_seconds)
    features[f"{p}_coverage_ratio"] = primary["coverage_ratio"]
    features[f"{p}_missing_ratio"] = primary["missing_ratio"]
    features[f"{p}_gap_flag"] = primary["gap_flag"]
    features[f"{p}_stale_periods"] = primary["stale_periods"]
    features[f"{p}_bar_age_seconds"] = primary["bar_age_seconds"]
    features[f"{p}_coverage_roll_medium"] = primary["coverage_ratio"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_gap_ratio_roll_medium"] = primary["gap_flag"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_valid_observation_count_roll_medium"] = valid_primary.astype(float).rolling(cfg.medium_window, min_periods=1).sum()
    features[f"{p}_spread_bps_mean"] = primary["spread_bps_mean"]
    features[f"{p}_spread_bps_roll_medium"] = primary["spread_bps_mean"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_spread_bps_max_roll_medium"] = primary["spread_bps_max"].rolling(cfg.medium_window, min_periods=1).max()
    features[f"{p}_spread_widening_bps"] = (primary["spread_bps_max"] - primary["spread_bps_mean"]).clip(lower=0.0)
    features[f"{p}_spread_widening_bps_roll_medium"] = features[f"{p}_spread_widening_bps"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_spread_volatility_roll_medium"] = primary["spread_bps_mean"].rolling(cfg.medium_window, min_periods=2).std()
    features[f"{p}_spread_zscore_medium"] = _rolling_zscore(primary["spread_bps_mean"], cfg.medium_window)
    features[f"{p}_wide_spread_persistence"] = _valid_persistence(
        primary["spread_bps_mean"].ge(cfg.spread_penalty_bps),
        valid_primary,
        cfg.persistence_window,
    )
    features[f"{p}_microprice_offset_bps_mean"] = primary["microprice_offset_bps_mean"]
    features[f"{p}_microprice_offset_bps_roll_medium"] = primary["microprice_offset_bps_mean"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_microprice_delta_1"] = primary["microprice_offset_bps_mean"].diff()
    features[f"{p}_microprice_volatility_roll_medium"] = primary["microprice_offset_bps_mean"].rolling(cfg.medium_window, min_periods=2).std()
    features[f"{p}_imbalance_top20_mean"] = primary["imbalance_top20_mean"]
    features[f"{p}_imbalance_roll_short"] = primary["imbalance_top20_mean"].rolling(cfg.short_window, min_periods=1).mean()
    features[f"{p}_imbalance_roll_medium"] = primary["imbalance_top20_mean"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_imbalance_roll_long"] = primary["imbalance_top20_mean"].rolling(cfg.long_window, min_periods=1).mean()
    features[f"{p}_imbalance_coverage_weighted_roll_medium"] = _rolling_weighted_mean(
        primary["imbalance_top20_mean"],
        primary["coverage_ratio"],
        cfg.medium_window,
    )
    features[f"{p}_imbalance_delta_1"] = primary["imbalance_top20_mean"].diff()
    features[f"{p}_imbalance_delta_short"] = primary["imbalance_top20_mean"] - primary["imbalance_top20_mean"].shift(cfg.short_window)
    features[f"{p}_imbalance_volatility_roll_medium"] = primary["imbalance_top20_mean"].rolling(cfg.medium_window, min_periods=2).std()
    features[f"{p}_imbalance_10bps_roll_medium"] = primary["imbalance_10bps_mean"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_imbalance_25bps_roll_medium"] = primary["imbalance_25bps_mean"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_bid_pressure_ratio"] = primary["bid_pressure_ratio"]
    features[f"{p}_ask_pressure_ratio"] = primary["ask_pressure_ratio"]
    features[f"{p}_bid_pressure_roll_short"] = primary["bid_pressure_ratio"].rolling(cfg.short_window, min_periods=1).mean()
    features[f"{p}_ask_pressure_roll_short"] = primary["ask_pressure_ratio"].rolling(cfg.short_window, min_periods=1).mean()
    features[f"{p}_pressure_delta"] = primary["bid_pressure_ratio"] - primary["ask_pressure_ratio"]
    features[f"{p}_pressure_delta_roll_short"] = features[f"{p}_pressure_delta"].rolling(cfg.short_window, min_periods=1).mean()
    features[f"{p}_pressure_delta_roll_medium"] = features[f"{p}_pressure_delta"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_pressure_flip_count_roll_medium"] = _rolling_sign_flip_count(features[f"{p}_pressure_delta"], cfg.medium_window)
    features[f"{p}_bid_pressure_persistence"] = _valid_persistence(
        primary["bid_pressure_ratio"].ge(cfg.pressure_threshold),
        valid_primary,
        cfg.persistence_window,
    )
    features[f"{p}_ask_pressure_persistence"] = _valid_persistence(
        primary["ask_pressure_ratio"].ge(cfg.pressure_threshold),
        valid_primary,
        cfg.persistence_window,
    )
    features[f"{p}_bid_wall_near_persistence"] = _valid_persistence(
        primary["nearest_bid_wall_min_distance_bps"].le(cfg.wall_near_threshold_bps),
        valid_primary,
        cfg.persistence_window,
    )
    features[f"{p}_ask_wall_near_persistence"] = _valid_persistence(
        primary["nearest_ask_wall_min_distance_bps"].le(cfg.wall_near_threshold_bps),
        valid_primary,
        cfg.persistence_window,
    )
    features[f"{p}_wall_distance_delta_bps"] = (
        primary["nearest_ask_wall_min_distance_bps"] - primary["nearest_bid_wall_min_distance_bps"]
    )
    features[f"{p}_wall_support_score"] = _near_wall_score(primary["nearest_bid_wall_min_distance_bps"], cfg.wall_near_threshold_bps)
    features[f"{p}_wall_resistance_score"] = _near_wall_score(primary["nearest_ask_wall_min_distance_bps"], cfg.wall_near_threshold_bps)
    features[f"{p}_wall_support_resistance_delta"] = features[f"{p}_wall_support_score"] - features[f"{p}_wall_resistance_score"]
    features[f"{p}_bid_wall_score_roll_medium"] = primary["strongest_bid_wall_score"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_ask_wall_score_roll_medium"] = primary["strongest_ask_wall_score"].rolling(cfg.medium_window, min_periods=1).mean()
    features[f"{p}_liquidity_stress_score"] = _clip01(
        (0.35 * primary["gap_flag"])
        + (0.30 * (primary["spread_bps_mean"] / cfg.spread_penalty_bps).clip(0.0, 1.0))
        + (0.20 * primary["ask_pressure_ratio"].clip(0.0, 1.0))
        + (0.15 * features[f"{p}_wall_resistance_score"])
    )
    features[f"{p}_support_pressure_score"] = _clip01(
        (0.45 * primary["bid_pressure_ratio"].clip(0.0, 1.0))
        + (0.30 * features[f"{p}_wall_support_score"])
        + (0.25 * ((primary["imbalance_top20_mean"].clip(-1.0, 1.0) + 1.0) / 2.0))
    ).where(valid_primary)
    features[f"{p}_resistance_pressure_score"] = _clip01(
        (0.45 * primary["ask_pressure_ratio"].clip(0.0, 1.0))
        + (0.30 * features[f"{p}_wall_resistance_score"])
        + (0.25 * ((1.0 - primary["imbalance_top20_mean"].clip(-1.0, 1.0)) / 2.0))
    ).where(valid_primary)

    _append_per_market_features(features, aligned, cfg)
    _append_comparison_features(features, aligned, cfg)
    _append_score_features(features, primary, cfg)
    return features


def _append_per_market_features(features: DataFrame, aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> None:
    p = cfg.prefix
    for market_key, frame in aligned.items():
        market = _safe_column_part(market_key)
        features[f"{p}_{market}_coverage_ratio"] = frame["coverage_ratio"]
        features[f"{p}_{market}_gap_flag"] = frame["gap_flag"]
        features[f"{p}_{market}_spread_bps_mean"] = frame["spread_bps_mean"]
        features[f"{p}_{market}_imbalance_top20_mean"] = frame["imbalance_top20_mean"]
        features[f"{p}_{market}_microprice_offset_bps_mean"] = frame["microprice_offset_bps_mean"]
        features[f"{p}_{market}_bid_pressure_ratio"] = frame["bid_pressure_ratio"]
        features[f"{p}_{market}_ask_pressure_ratio"] = frame["ask_pressure_ratio"]
        features[f"{p}_{market}_nearest_bid_wall_distance_bps"] = frame["nearest_bid_wall_min_distance_bps"]
        features[f"{p}_{market}_nearest_ask_wall_distance_bps"] = frame["nearest_ask_wall_min_distance_bps"]
        features[f"{p}_{market}_bar_age_seconds"] = frame["bar_age_seconds"]


def _append_comparison_features(features: DataFrame, aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> None:
    p = cfg.prefix
    for left_key, right_key in cfg.comparison_pairs:
        if left_key not in aligned or right_key not in aligned:
            continue
        left = aligned[left_key]
        right = aligned[right_key]
        name = f"{_safe_column_part(left_key)}_vs_{_safe_column_part(right_key)}"
        both_valid = left["gap_flag"].eq(0.0) & right["gap_flag"].eq(0.0)
        features[f"{p}_cmp_{name}_coverage_min"] = pd.concat(
            [left["coverage_ratio"], right["coverage_ratio"]], axis=1
        ).min(axis=1)
        features[f"{p}_cmp_{name}_gap_flag"] = (~both_valid).astype(float)
        features[f"{p}_cmp_{name}_spread_diff_bps"] = (right["spread_bps_mean"] - left["spread_bps_mean"]).where(both_valid)
        features[f"{p}_cmp_{name}_spread_ratio"] = (
            right["spread_bps_mean"] / left["spread_bps_mean"].replace(0.0, np.nan)
        ).where(both_valid)
        features[f"{p}_cmp_{name}_imbalance_divergence"] = (
            right["imbalance_top20_mean"] - left["imbalance_top20_mean"]
        ).where(both_valid)
        features[f"{p}_cmp_{name}_microprice_divergence_bps"] = (
            right["microprice_offset_bps_mean"] - left["microprice_offset_bps_mean"]
        ).where(both_valid)
        features[f"{p}_cmp_{name}_bid_pressure_diff"] = (right["bid_pressure_ratio"] - left["bid_pressure_ratio"]).where(both_valid)
        features[f"{p}_cmp_{name}_ask_pressure_diff"] = (right["ask_pressure_ratio"] - left["ask_pressure_ratio"]).where(both_valid)
        left_pressure_delta = left["bid_pressure_ratio"] - left["ask_pressure_ratio"]
        right_pressure_delta = right["bid_pressure_ratio"] - right["ask_pressure_ratio"]
        features[f"{p}_cmp_{name}_pressure_delta_diff"] = (right_pressure_delta - left_pressure_delta).where(both_valid)
        features[f"{p}_cmp_{name}_pressure_direction_disagree"] = (
            np.sign(left_pressure_delta).ne(np.sign(right_pressure_delta)).astype(float)
        ).where(both_valid)
        features[f"{p}_cmp_{name}_bid_wall_distance_diff_bps"] = (
            right["nearest_bid_wall_min_distance_bps"] - left["nearest_bid_wall_min_distance_bps"]
        ).where(both_valid)
        features[f"{p}_cmp_{name}_ask_wall_distance_diff_bps"] = (
            right["nearest_ask_wall_min_distance_bps"] - left["nearest_ask_wall_min_distance_bps"]
        ).where(both_valid)
        left_wall_delta = left["nearest_ask_wall_min_distance_bps"] - left["nearest_bid_wall_min_distance_bps"]
        right_wall_delta = right["nearest_ask_wall_min_distance_bps"] - right["nearest_bid_wall_min_distance_bps"]
        features[f"{p}_cmp_{name}_wall_delta_divergence_bps"] = (right_wall_delta - left_wall_delta).where(both_valid)


def _append_score_features(features: DataFrame, primary: DataFrame, cfg: OrderbookContextFeatureConfig) -> None:
    p = cfg.prefix
    valid = primary["gap_flag"].eq(0.0)
    imbalance = primary["imbalance_top20_mean"].clip(-1.0, 1.0)
    microprice = (primary["microprice_offset_bps_mean"] / cfg.microprice_scale_bps).clip(-1.0, 1.0)
    spread_penalty = (primary["spread_bps_mean"] / cfg.spread_penalty_bps).clip(0.0, 1.0)
    bid_pressure = primary["bid_pressure_ratio"].clip(0.0, 1.0)
    ask_pressure = primary["ask_pressure_ratio"].clip(0.0, 1.0)
    bid_wall_support = _near_wall_score(primary["nearest_bid_wall_min_distance_bps"], cfg.wall_near_threshold_bps)
    ask_wall_resistance = _near_wall_score(primary["nearest_ask_wall_min_distance_bps"], cfg.wall_near_threshold_bps)

    long_score = (
        (0.35 * ((imbalance + 1.0) / 2.0))
        + (0.20 * ((microprice + 1.0) / 2.0))
        + (0.20 * bid_pressure)
        + (0.15 * bid_wall_support)
        + (0.10 * (1.0 - spread_penalty))
    )
    short_score = (
        (0.35 * ((1.0 - imbalance) / 2.0))
        + (0.20 * ((1.0 - microprice) / 2.0))
        + (0.20 * ask_pressure)
        + (0.15 * ask_wall_resistance)
        + (0.10 * (1.0 - spread_penalty))
    )
    long_score = _clip01(long_score).where(valid)
    short_score = _clip01(short_score).where(valid)
    abs_score = _clip01((pd.concat([long_score, short_score], axis=1).max(axis=1) - 0.5).abs() * 2.0).where(valid)
    long_bias = long_score.ge(0.60) & features[f"{p}_bid_pressure_persistence"].ge(0.30)
    short_bias = short_score.ge(0.60) & features[f"{p}_ask_pressure_persistence"].ge(0.30)
    state = Series(np.select([long_bias, short_bias], [1.0, -1.0], default=0.0), index=features.index)
    state = state.where(valid, np.nan)

    features[f"{p}_score_long"] = long_score
    features[f"{p}_score_short"] = short_score
    features[f"{p}_score_abs"] = abs_score
    features[f"{p}_state"] = state


def _load_market_context(
    cfg: OrderbookContextFeatureConfig,
    canonical_pair: str,
    query_start: str,
    query_end: str,
) -> DataFrame:
    db_path = _resolved_db_path(cfg)
    if not db_path.exists():
        return DataFrame()
    placeholders = ", ".join("?" for _ in cfg.market_keys)
    query = f"""
        SELECT
            id,
            ts,
            source_ts,
            market_key,
            venue,
            market_type,
            margin_type,
            quote_asset,
            canonical_pair,
            symbol,
            funding_rate,
            open_interest,
            long_ratio,
            short_ratio,
            long_short_ratio,
            taker_buy_volume,
            taker_sell_volume,
            taker_buy_sell_ratio
        FROM market_context_ticks
        WHERE canonical_pair = ?
          AND market_key IN ({placeholders})
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts, market_key, id
    """
    params: list[Any] = [canonical_pair, *cfg.market_keys, query_start, query_end]
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        return pd.read_sql_query(query, conn, params=params)
    except sqlite3.Error as exc:
        if cfg.allow_missing:
            return DataFrame()
        raise RuntimeError(f"Could not read market_context_ticks from Orderbook SQLite: {db_path}") from exc
    finally:
        if conn is not None:
            conn.close()


def _align_market_context(
    context: DataFrame,
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: OrderbookContextFeatureConfig,
) -> DataFrame:
    p = cfg.prefix
    features = DataFrame(index=output_index)
    for market_key in cfg.market_keys:
        market = _safe_column_part(market_key)
        for column in MARKET_CONTEXT_NUMERIC_COLUMNS:
            features[f"{p}_{market}_{column}"] = np.nan
        features[f"{p}_{market}_market_context_age_seconds"] = np.nan

    if context.empty:
        return features

    prepared = context.copy()
    prepared["ts"] = pd.to_datetime(prepared["ts"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["ts"])
    for column in MARKET_CONTEXT_NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    max_age = pd.Timedelta(seconds=cfg.bar_timeframe_seconds * cfg.market_context_max_age_periods)
    candle_frame = DataFrame({"candle_time": candle_times}).sort_values("candle_time")

    for market_key in cfg.market_keys:
        market_rows = prepared.loc[prepared["market_key"].eq(market_key)].sort_values("ts")
        if market_rows.empty:
            continue
        merged = pd.merge_asof(
            candle_frame,
            market_rows,
            left_on="candle_time",
            right_on="ts",
            direction="backward",
            tolerance=max_age,
        )
        if cfg.availability_lag_candles:
            merged[MARKET_CONTEXT_NUMERIC_COLUMNS] = merged[list(MARKET_CONTEXT_NUMERIC_COLUMNS)].shift(cfg.availability_lag_candles)
            merged["ts"] = merged["ts"].shift(cfg.availability_lag_candles)
        age_seconds = (merged["candle_time"] - pd.to_datetime(merged["ts"], utc=True, errors="coerce")).dt.total_seconds()
        market = _safe_column_part(market_key)
        for column in MARKET_CONTEXT_NUMERIC_COLUMNS:
            features[f"{p}_{market}_{column}"] = Series(merged[column].to_numpy(), index=output_index)
        features[f"{p}_{market}_market_context_age_seconds"] = Series(age_seconds.to_numpy(), index=output_index)
    return features


def _append_empty_features(dataframe: DataFrame, cfg: OrderbookContextFeatureConfig) -> DataFrame:
    frame = dataframe.copy()
    empty = DataFrame(index=frame.index)
    p = cfg.prefix
    base_columns = (
        f"{p}_mode_live",
        f"{p}_live_stream_enabled",
        f"{p}_source_bar_timeframe_seconds",
        f"{p}_coverage_ratio",
        f"{p}_missing_ratio",
        f"{p}_gap_flag",
        f"{p}_stale_periods",
        f"{p}_bar_age_seconds",
        f"{p}_coverage_roll_medium",
        f"{p}_gap_ratio_roll_medium",
        f"{p}_valid_observation_count_roll_medium",
        f"{p}_spread_bps_mean",
        f"{p}_spread_bps_roll_medium",
        f"{p}_spread_bps_max_roll_medium",
        f"{p}_spread_widening_bps",
        f"{p}_spread_widening_bps_roll_medium",
        f"{p}_spread_volatility_roll_medium",
        f"{p}_spread_zscore_medium",
        f"{p}_wide_spread_persistence",
        f"{p}_microprice_offset_bps_mean",
        f"{p}_microprice_offset_bps_roll_medium",
        f"{p}_microprice_delta_1",
        f"{p}_microprice_volatility_roll_medium",
        f"{p}_imbalance_top20_mean",
        f"{p}_imbalance_roll_short",
        f"{p}_imbalance_roll_medium",
        f"{p}_imbalance_roll_long",
        f"{p}_imbalance_coverage_weighted_roll_medium",
        f"{p}_imbalance_delta_1",
        f"{p}_imbalance_delta_short",
        f"{p}_imbalance_volatility_roll_medium",
        f"{p}_imbalance_10bps_roll_medium",
        f"{p}_imbalance_25bps_roll_medium",
        f"{p}_bid_pressure_ratio",
        f"{p}_ask_pressure_ratio",
        f"{p}_bid_pressure_roll_short",
        f"{p}_ask_pressure_roll_short",
        f"{p}_pressure_delta",
        f"{p}_pressure_delta_roll_short",
        f"{p}_pressure_delta_roll_medium",
        f"{p}_pressure_flip_count_roll_medium",
        f"{p}_bid_pressure_persistence",
        f"{p}_ask_pressure_persistence",
        f"{p}_bid_wall_near_persistence",
        f"{p}_ask_wall_near_persistence",
        f"{p}_wall_distance_delta_bps",
        f"{p}_wall_support_score",
        f"{p}_wall_resistance_score",
        f"{p}_wall_support_resistance_delta",
        f"{p}_bid_wall_score_roll_medium",
        f"{p}_ask_wall_score_roll_medium",
        f"{p}_liquidity_stress_score",
        f"{p}_support_pressure_score",
        f"{p}_resistance_pressure_score",
        f"{p}_score_long",
        f"{p}_score_short",
        f"{p}_score_abs",
        f"{p}_state",
    )
    for column in base_columns:
        empty[column] = np.nan
    empty[f"{p}_mode_live"] = 1.0 if _effective_data_mode(cfg) == "live" else 0.0
    empty[f"{p}_live_stream_enabled"] = 1.0 if cfg.enable_live_stream else 0.0
    empty[f"{p}_source_bar_timeframe_seconds"] = float(cfg.bar_timeframe_seconds)
    for market_key in cfg.market_keys:
        market = _safe_column_part(market_key)
        for suffix in (
            "coverage_ratio",
            "gap_flag",
            "spread_bps_mean",
            "imbalance_top20_mean",
            "microprice_offset_bps_mean",
            "bid_pressure_ratio",
            "ask_pressure_ratio",
            "nearest_bid_wall_distance_bps",
            "nearest_ask_wall_distance_bps",
            "bar_age_seconds",
        ):
            empty[f"{p}_{market}_{suffix}"] = np.nan
        if cfg.include_market_context:
            for column in MARKET_CONTEXT_NUMERIC_COLUMNS:
                empty[f"{p}_{market}_{column}"] = np.nan
            empty[f"{p}_{market}_market_context_age_seconds"] = np.nan
    for left_key, right_key in cfg.comparison_pairs:
        name = f"{_safe_column_part(left_key)}_vs_{_safe_column_part(right_key)}"
        for suffix in (
            "coverage_min",
            "gap_flag",
            "spread_diff_bps",
            "spread_ratio",
            "imbalance_divergence",
            "microprice_divergence_bps",
            "bid_pressure_diff",
            "ask_pressure_diff",
            "pressure_delta_diff",
            "pressure_direction_disagree",
            "bid_wall_distance_diff_bps",
            "ask_wall_distance_diff_bps",
            "wall_delta_divergence_bps",
        ):
            empty[f"{p}_cmp_{name}_{suffix}"] = np.nan
    return pd.concat([frame, empty], axis=1)


def _stale_periods(valid: Series) -> Series:
    valid_bool = valid.fillna(False).astype(bool)
    groups = valid_bool.cumsum()
    stale = (~valid_bool).groupby(groups).cumsum().astype(float)
    stale[valid_bool] = 0.0
    return stale


def _near_wall_score(distance_bps: Series, threshold_bps: float) -> Series:
    distance = pd.to_numeric(distance_bps, errors="coerce")
    if threshold_bps <= 0:
        return distance.le(0).astype(float)
    return (1.0 - (distance / threshold_bps)).clip(0.0, 1.0).fillna(0.0)


def _rolling_weighted_mean(values: Series, weights: Series, window: int) -> Series:
    clean_values = pd.to_numeric(values, errors="coerce")
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    clean_weights = clean_weights.where(clean_values.notna(), 0.0)
    numerator = (clean_values.fillna(0.0) * clean_weights).rolling(window, min_periods=1).sum()
    denominator = clean_weights.rolling(window, min_periods=1).sum()
    return numerator / denominator.replace(0.0, np.nan)


def _rolling_zscore(series: Series, window: int) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    rolling_mean = numeric.rolling(window, min_periods=2).mean()
    rolling_std = numeric.rolling(window, min_periods=2).std()
    return (numeric - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _rolling_sign_flip_count(series: Series, window: int) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sign = Series(np.sign(numeric), index=numeric.index).replace(0.0, np.nan)
    flip = sign.ne(sign.shift()) & sign.notna() & sign.shift().notna()
    return flip.astype(float).rolling(window, min_periods=1).sum()


def _valid_persistence(condition: Series, valid: Series, window: int) -> Series:
    observed = valid.fillna(False).astype(bool)
    hits = (condition.fillna(False).astype(bool) & observed).astype(float)
    observed_count = observed.astype(float).rolling(window, min_periods=1).sum()
    return (hits.rolling(window, min_periods=1).sum() / observed_count.replace(0.0, np.nan)).clip(0.0, 1.0)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def _safe_column_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "market"


__all__ = [
    "OrderbookContextFeatureConfig",
    "add_orderbook_context_features",
    "format_orderbook_bars",
]
