from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json
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

BAR_JSON_COLUMNS: tuple[str, ...] = (
    "bid_wall_blocks_json",
    "ask_wall_blocks_json",
    "bid_liquidity_zones_json",
    "ask_liquidity_zones_json",
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

WALL_BLOCK_OUTPUT_SUFFIXES: tuple[str, ...] = tuple(
    f"{side}_wall{slot}_{field}"
    for side in ("bid", "ask")
    for slot in range(1, 4)
    for field in ("price", "distance_bps", "score", "notional", "persistence", "age_seconds", "resilience_score")
)

LIQUIDITY_ZONE_OUTPUT_SUFFIXES: tuple[str, ...] = tuple(
    f"{side}_liq_{kind}{slot}_{field}"
    for side in ("bid", "ask")
    for kind in ("high", "low")
    for slot in range(1, 3)
    for field in ("lower_price", "upper_price", "mid_price", "distance_bps", "score", "notional", "persistence")
)

STRATEGY_OUTPUT_SUFFIXES: tuple[str, ...] = (
    "ready",
    "quality_score",
    "coverage_ratio",
    "gap_flag",
    "risk_score",
    "risk_state",
    "risk_block",
    "spread_bps",
    "spread_state",
    "score_long",
    "score_short",
    "score_abs",
    "state",
    "book_bias_score",
    "book_state",
    "pressure_score",
    "pressure_state",
    "wall_score",
    "wall_state",
    "wall_box_score",
    "wall_support_score",
    "wall_resistance_score",
    "market_ready_ratio",
    "market_pressure_score",
    "market_agreement_score",
    "market_state",
    *WALL_BLOCK_OUTPUT_SUFFIXES,
    *LIQUIDITY_ZONE_OUTPUT_SUFFIXES,
)


@dataclass(frozen=True)
class OrderbookContextFeatureConfig:
    """First-pass Order Book formatter and feature settings.

    Objective:
    Convert the launcher Order Book SQLite data into candle-aligned,
    no-trading-decision dataframe features for 1h strategy research.

    First-pass scope:
    - Use compact ``orderbook_metric_bars`` rows by default, not raw snapshots.
      Source bars are aggregated by interval overlap into each closed candle
      window, so collector-phase offsets do not decide the candle assignment.
    - Keep gap handling explicit with coverage, missing ratio, gap flags, bar
      age, and stale-period columns.
    - Treat historical/backtest rows and live/dry-run rows differently.
    - Add market comparison features where both selected markets exist.
    - Optionally append derivative market-context rows already stored beside
      order book data: funding, open interest, long/short, and taker ratio.
    - Do not infer entries, exits, sizing, or execution simulation.

    Strategy output profile:
    Normal strategy runs get a compact headline packet by default. The packet
    answers: is the row usable, is order-book pressure long/short/neutral, is
    liquidity risk acceptable, are nearby walls supportive or restrictive, and
    do other configured markets broadly agree. Detailed per-market and
    comparison columns are available through ``include_diagnostics=True`` for
    review plots and root-cause analysis, but they are not normal strategy
    inputs.

    Gap convention:
    News gaps can usually mean "no article arrived". Order book gaps mean
    "measurement unavailable". This module therefore does not forward-fill
    order book pressure/spread/imbalance values across missing candle bins.
    Missing bins get ``coverage_ratio=0`` and ``gap_flag=1``. Rolling features
    are accompanied by rolling coverage/gap ratios so another agent can reject
    low-quality periods instead of accidentally treating them as neutral.

    Strategy-facing readiness contract:
    Strategies should normally gate order-book context with ``obctx_ready``.
    A ready row means the primary market's closed summary window had enough
    valid source observations, was not stale, and passed the configured
    coverage threshold. The module may still expose raw partial diagnostics on
    unready rows for plotting and investigation, but the shared score/state
    columns are masked when the row is not ready.

    Relevant SQLite tables:
    ``orderbook_metric_bars(ts_start, ts_end, timeframe_seconds, stream_id,
    market_key, canonical_pair, valid_samples, expected_samples,
    spread_bps_mean, microprice_offset_bps_mean, imbalance_top20_mean,
    bid_pressure_ratio, ask_pressure_ratio, nearest_*_wall_*, ...)``

    ``market_context_ticks(ts, source_ts, market_key, canonical_pair,
    funding_rate, open_interest, long_ratio, short_ratio, long_short_ratio,
    taker_buy_sell_ratio, ...)``

    Agent notes:
    - The safest default is closed-window alignment: a row timestamped 13:00
      summarizes the 12:00-13:00 candle window.
    - ``summary_lag_seconds`` defaults to zero so the formatter acts as soon as
      the candle has closed. If the collector still emits source bars that end
      after the candle close because they are phase-offset, those late-ending
      bars are not used for the just-closed candle. This avoids lookahead and
      lowers coverage instead of pretending the window was complete.
    - For backtest/hyperopt, pass a historical order-book SQLite path and keep
      ``data_mode="historical"``. Do not use the live collector DB in backtests
      unless it genuinely contains the historical range being tested.
    - For dry/live, use ``data_mode="auto"``, pass the strategy runmode, and set
      ``enable_live_stream=True``. This switches to the configured live bar
      timeframe and adds freshness checks.
    - TODO(live): when the collector supports wall-clock anchored metric bars,
      keep ``summary_lag_seconds=0`` and prefer source bars that close exactly
      on candle boundaries. If live strategy dataframes include an actively
      forming candle, the strategy integration should only consume rows where
      ``obctx_ready=1`` and the strategy knows the candle is closed.
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
    bar_timeframe_seconds: int = 60
    live_bar_timeframe_seconds: int = 60
    availability_lag_candles: int = 0
    summary_lag_seconds: int = 0
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
    include_market_context: bool = False
    include_diagnostics: bool = False
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
    summary_lag_seconds: int | None = None,
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
    include_diagnostics: bool | None = None,
    market_context_max_age_periods: int | None = None,
    prefix: str | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Append first-pass order book context features to a candle dataframe.

    Expected use in a strategy:
    ``dataframe = add_orderbook_context_features(dataframe, pair=metadata["pair"])``

    Output columns use the configured prefix, default ``obctx``. By default the
    dataframe is intentionally small and strategy-facing:
    - ``obctx_ready`` / ``obctx_quality_score``: whether the closed summary row
      is usable and how complete the recent primary-market data is.
    - ``obctx_risk_score`` / ``obctx_risk_state`` / ``obctx_risk_block``:
      liquidity and data-quality risk, where higher score/state is worse.
    - ``obctx_score_long`` / ``obctx_score_short`` / ``obctx_state``:
      shared score contract derived only from order-book context.
    - ``obctx_book_bias_score`` / ``obctx_book_state``: signed headline bias;
      positive favours long context, negative favours short context.
    - ``obctx_pressure_score`` / ``obctx_wall_score`` /
      ``obctx_market_agreement_score``: compact explanations for the bias.
    - ``obctx_bid_wallN_*`` / ``obctx_ask_wallN_*``: up to three aggregated
      support/resistance wall blocks per side when the collector bar schema has
      wall-block JSON.
    - ``obctx_bid_liq_highN_*`` / ``obctx_ask_liq_lowN_*`` etc.: high/low
      liquidity zones below and above price when the collector bar schema has
      liquidity-zone JSON.

    Pass ``include_diagnostics=True`` to also export detailed per-market,
    rolling, spread, imbalance, wall, comparison, and market-context columns for
    visual review or root-cause analysis.

    Order-book values are not forward-filled through missing candle bins.
    Coverage and gap columns should be used before trusting any rolling score.

    ``data_mode`` controls runtime behavior:
    - ``historical``: default for backtest/hyperopt; uses
      ``bar_timeframe_seconds`` source rows and strict missing-measurement
      columns.
    - ``live``: requires ``enable_live_stream=True``; uses
      ``live_bar_timeframe_seconds`` unless a non-default bar timeframe is
      explicitly set, and applies live freshness checks.
    - ``auto``: live only when runmode looks like dry/live and live stream is
      enabled; otherwise historical.

    Candle alignment convention:
    A feature row timestamped 13:00 summarizes source rows overlapping the
    12:00-13:00 candle window. Source rows ending after
    ``13:00 + summary_lag_seconds`` are not used for that row, even if part of
    the source row overlaps the closed candle, because using them would require
    data that was not yet available.
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
        summary_lag_seconds=summary_lag_seconds,
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
        include_diagnostics=include_diagnostics,
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
    summary_lag_seconds: int | None = None,
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
        summary_lag_seconds=summary_lag_seconds,
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
    if int(cfg.bar_timeframe_seconds) == 60:
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
    if cfg.summary_lag_seconds < 0:
        raise ValueError("Orderbook summary_lag_seconds must be >= 0.")
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
    _summary_timedelta(cfg)


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


def _summary_timedelta(cfg: OrderbookContextFeatureConfig) -> pd.Timedelta:
    try:
        delta = pd.to_timedelta(cfg.resample_rule)
    except (TypeError, ValueError):
        offset = pd.tseries.frequencies.to_offset(cfg.resample_rule)
        try:
            delta = pd.Timedelta(offset.nanos, unit="ns")
        except ValueError as exc:
            raise ValueError("Orderbook resample_rule must be a fixed-width duration like '1h' or '5min'.") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError("Orderbook resample_rule must be a positive duration.")
    return delta


def _query_bounds(candle_times: pd.DatetimeIndex, cfg: OrderbookContextFeatureConfig) -> tuple[str, str]:
    lookback_periods = max(cfg.long_window, cfg.persistence_window) + cfg.availability_lag_candles + 2
    summary_delta = _summary_timedelta(cfg)
    source_margin = pd.Timedelta(seconds=int(cfg.bar_timeframe_seconds) + int(cfg.summary_lag_seconds))
    lookback = (summary_delta * lookback_periods) + source_margin
    start = candle_times.min() - lookback
    end = candle_times.max() + source_margin
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

    select_columns = (
        "id",
        "ts_start",
        "ts_end",
        "timeframe_seconds",
        "stream_id",
        "market_key",
        "venue",
        "exchange",
        "market_type",
        "margin_type",
        "quote_asset",
        "canonical_pair",
        "pair",
        "symbol",
        "valid_samples",
        "expected_samples",
        *BAR_NUMERIC_COLUMNS[2:],
        *BAR_JSON_COLUMNS,
    )

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        existing_columns = _table_columns(conn, "orderbook_metric_bars")
        select_exprs = [
            column if column in existing_columns else f"NULL AS {column}"
            for column in select_columns
        ]
        query = f"""
            SELECT
                {", ".join(select_exprs)}
            FROM orderbook_metric_bars
            WHERE {" AND ".join(where)}
            ORDER BY ts_end, market_key, id
        """
        return pd.read_sql_query(query, conn, params=params)
    except sqlite3.Error as exc:
        if cfg.allow_missing:
            return DataFrame()
        raise RuntimeError(f"Could not read Orderbook SQLite: {db_path}") from exc
    finally:
        if conn is not None:
            conn.close()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


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
            aligned[market_key] = _empty_market_frame(candle_times, output_index)
            continue
        market_aligned = _aggregate_market_to_candles(
            market_rows,
            candle_times,
            output_index,
            cfg,
            include_json=market_key == cfg.primary_market_key,
        )
        market_aligned = _finalize_aligned_market(market_aligned, cfg)
        aligned[market_key] = market_aligned
    return aligned


def _aggregate_market_to_candles(
    market_rows: DataFrame,
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: OrderbookContextFeatureConfig,
    *,
    include_json: bool = True,
) -> DataFrame:
    """Summarize source bars into closed candle windows without lookahead.

    The collector may store source bars that are not aligned to the strategy
    clock. This function therefore treats each source row as an interval,
    measures the overlap with each target candle window, and contributes only
    the overlapping slice that would be known by ``candle_close +
    summary_lag_seconds``. Coverage is measured against the whole target
    window, so missing source intervals reduce ``coverage_ratio`` and make the
    row unready instead of making a partial window look clean.
    """

    rows = market_rows.dropna(subset=["ts_start", "ts_end"]).sort_values("ts_end").reset_index(drop=True)
    summary_delta = _summary_timedelta(cfg)
    summary_delta_ns = int(summary_delta.value)
    summary_lag = pd.Timedelta(seconds=int(cfg.summary_lag_seconds))
    summary_lag_ns = int(summary_lag.value)
    candle_index = pd.DatetimeIndex(candle_times)
    target_ns = candle_index.asi8

    output = DataFrame(index=pd.RangeIndex(len(candle_index)))
    for column in BAR_NUMERIC_COLUMNS:
        output[column] = np.nan
    for column in BAR_JSON_COLUMNS:
        output[column] = None
    output["summary_window_start"] = Series(candle_index - summary_delta, index=output.index)
    output["summary_window_end"] = Series(candle_index, index=output.index)
    output["summary_available_at"] = Series(candle_index + summary_lag, index=output.index)
    output["bar_observed_at"] = Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["bar_age_seconds"] = np.nan

    if rows.empty or len(candle_index) == 0:
        output.index = output_index
        return output

    starts = pd.DatetimeIndex(rows["ts_start"]).asi8
    ends = pd.DatetimeIndex(rows["ts_end"]).asi8
    source_duration_ns = ends - starts
    target_positions: list[int] = []
    source_positions: list[int] = []
    overlap_ns_values: list[int] = []

    for source_pos, (start_ns, end_ns, duration_ns) in enumerate(zip(starts, ends, source_duration_ns)):
        if duration_ns <= 0:
            continue
        first_target = int(np.searchsorted(target_ns, start_ns, side="right"))
        last_target = int(np.searchsorted(target_ns, end_ns + summary_delta_ns, side="left"))
        for target_pos in range(first_target, last_target):
            target_end_ns = int(target_ns[target_pos])
            if end_ns > target_end_ns + summary_lag_ns:
                continue
            target_start_ns = target_end_ns - summary_delta_ns
            overlap_ns = min(end_ns, target_end_ns) - max(start_ns, target_start_ns)
            if overlap_ns <= 0:
                continue
            target_positions.append(target_pos)
            source_positions.append(source_pos)
            overlap_ns_values.append(int(overlap_ns))

    if not target_positions:
        output.index = output_index
        return output

    target_array = np.asarray(target_positions, dtype=np.int64)
    source_array = np.asarray(source_positions, dtype=np.int64)
    overlap_ns_array = np.asarray(overlap_ns_values, dtype=np.float64)
    duration_array = source_duration_ns[source_array].astype(np.float64)
    fraction = np.divide(overlap_ns_array, duration_array, out=np.zeros_like(overlap_ns_array), where=duration_array > 0)

    valid_samples = pd.to_numeric(rows["valid_samples"], errors="coerce").to_numpy(dtype=float)[source_array] * fraction
    expected_samples = pd.to_numeric(rows["expected_samples"], errors="coerce").to_numpy(dtype=float)[source_array] * fraction
    overlap_seconds = overlap_ns_array / 1_000_000_000.0
    contributions = DataFrame(
        {
            "target_pos": target_array,
            "source_pos": source_array,
            "valid_samples": valid_samples,
            "expected_samples": expected_samples,
            "overlap_seconds": overlap_seconds,
        }
    )
    grouped = contributions.groupby("target_pos", sort=False)
    valid_sum = grouped["valid_samples"].sum(min_count=1)
    observed_expected_sum = grouped["expected_samples"].sum(min_count=1)
    observed_seconds_sum = grouped["overlap_seconds"].sum(min_count=1)
    observed_rate = observed_expected_sum / observed_seconds_sum.replace(0.0, np.nan)
    full_expected = observed_rate * float(summary_delta.total_seconds())
    output.loc[valid_sum.index, "valid_samples"] = valid_sum
    output.loc[full_expected.index, "expected_samples"] = full_expected
    output.loc[full_expected.index, "coverage_ratio"] = (valid_sum / full_expected.replace(0.0, np.nan)).clip(0.0, 1.0)
    output["coverage_ratio"] = output["coverage_ratio"].fillna(0.0)
    output["missing_ratio"] = (1.0 - output["coverage_ratio"]).clip(0.0, 1.0)

    weight = np.nan_to_num(valid_samples, nan=0.0)
    for column in (
        "spread_bps_mean",
        "microprice_offset_bps_mean",
        "imbalance_top20_mean",
        "imbalance_10bps_mean",
        "imbalance_25bps_mean",
    ):
        output.loc[:, column] = _weighted_source_mean(rows, column, target_array, source_array, weight).reindex(output.index)

    for column in ("spread_bps_max", "imbalance_top20_max", "max_bid_pressure_streak_seconds", "max_ask_pressure_streak_seconds"):
        output.loc[:, column] = _source_reduction(rows, column, target_array, source_array, "max").reindex(output.index)
    for column in ("imbalance_top20_min", "nearest_bid_wall_min_distance_bps", "nearest_ask_wall_min_distance_bps"):
        output.loc[:, column] = _source_reduction(rows, column, target_array, source_array, "min").reindex(output.index)
    for column in ("strongest_bid_wall_score", "strongest_ask_wall_score"):
        output.loc[:, column] = _source_reduction(rows, column, target_array, source_array, "max").reindex(output.index)

    for column in ("bid_pressure_seconds", "ask_pressure_seconds"):
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)[source_array] * fraction
        reduced = DataFrame({"target_pos": target_array, "value": values}).groupby("target_pos", sort=False)["value"].sum(min_count=1)
        output.loc[reduced.index, column] = reduced
    pressure_denominator = float(summary_delta.total_seconds())
    output["bid_pressure_ratio"] = (output["bid_pressure_seconds"] / pressure_denominator).clip(0.0, 1.0)
    output["ask_pressure_ratio"] = (output["ask_pressure_seconds"] / pressure_denominator).clip(0.0, 1.0)

    observed = DataFrame({"target_pos": target_array, "ts_end_ns": ends[source_array]}).groupby("target_pos", sort=False)[
        "ts_end_ns"
    ].max()
    output.loc[observed.index, "bar_observed_at"] = pd.to_datetime(observed, utc=True)

    if include_json:
        for column in BAR_JSON_COLUMNS:
            if column in rows.columns:
                output.loc[:, column] = _aggregate_json_column_to_targets(rows, column, target_array, source_array, output.index)

    if cfg.availability_lag_candles:
        output = output.shift(int(cfg.availability_lag_candles))

    available_at = Series(candle_index + summary_lag, index=output.index)
    observed_at = pd.to_datetime(output["bar_observed_at"], utc=True, errors="coerce")
    output["bar_age_seconds"] = (available_at - observed_at).dt.total_seconds()
    output.index = output_index
    return output


def _weighted_source_mean(
    rows: DataFrame,
    column: str,
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    weights: np.ndarray,
) -> Series:
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)[source_positions]
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return Series(dtype=float)
    frame = DataFrame(
        {
            "target_pos": target_positions[valid],
            "numerator": values[valid] * weights[valid],
            "weight": weights[valid],
        }
    )
    grouped = frame.groupby("target_pos", sort=False).sum()
    return grouped["numerator"] / grouped["weight"].replace(0.0, np.nan)


def _source_reduction(
    rows: DataFrame,
    column: str,
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    reduction: str,
) -> Series:
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)[source_positions]
    valid = np.isfinite(values)
    if not valid.any():
        return Series(dtype=float)
    grouped = DataFrame({"target_pos": target_positions[valid], "value": values[valid]}).groupby("target_pos", sort=False)[
        "value"
    ]
    if reduction == "min":
        return grouped.min()
    return grouped.max()


def _aggregate_json_column_to_targets(
    rows: DataFrame,
    column: str,
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    output_index: Any,
) -> Series:
    membership = DataFrame({"target_pos": target_positions, "source_pos": source_positions}).drop_duplicates()
    output = Series([None] * len(output_index), index=output_index, dtype=object)
    for target_pos, group in membership.groupby("target_pos", sort=False):
        items: list[dict[str, Any]] = []
        for source_pos in group["source_pos"]:
            items.extend(_json_records(rows.iloc[int(source_pos)].get(column)))
        if not items:
            continue
        if "wall_blocks" in column:
            aggregated = _merge_wall_records(items)
        else:
            aggregated = _merge_liquidity_zone_records(items)
        output.iloc[int(target_pos)] = json.dumps(aggregated, separators=(",", ":"), sort_keys=True) if aggregated else None
    return output


def _json_records(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return value if isinstance(value, list) else []


def _merge_wall_records(items: list[dict[str, Any]], max_count: int = 3, tolerance_bps: float = 3.0) -> list[dict[str, float]]:
    groups: list[dict[str, Any]] = []
    for item in items:
        try:
            price = float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        target = None
        for group in groups:
            distance = _price_distance_bps(price, float(group["price"]))
            if distance <= tolerance_bps:
                target = group
                break
        if target is None:
            target = {"price": price, "items": []}
            groups.append(target)
        target["items"].append(item)
        prices = [float(record.get("price", price)) for record in target["items"]]
        weights = [max(float(record.get("resilience_score", record.get("score_max", record.get("score", 0.0))) or 0.0), 0.01) for record in target["items"]]
        weight_sum = sum(weights)
        target["price"] = sum(price_value * weight for price_value, weight in zip(prices, weights)) / weight_sum if weight_sum > 0 else sum(prices) / len(prices)

    merged: list[dict[str, float]] = []
    for group in groups:
        records = group["items"]
        def values(name: str) -> list[float]:
            output: list[float] = []
            for record in records:
                try:
                    output.append(float(record[name]))
                except (KeyError, TypeError, ValueError):
                    continue
            return output

        distance_values = values("distance_bps")
        resilience_values = values("resilience_score") or values("score_max") or values("score")
        score_values = values("score_max") or values("score")
        notional_values = values("notional_max") or values("notional")
        persistence_values = values("persistence")
        age_values = values("age_seconds")
        if not distance_values or not resilience_values:
            continue
        merged.append(
            {
                "price": float(group["price"]),
                "distance_bps": float(min(distance_values)),
                "score": float(max(score_values) if score_values else max(resilience_values)),
                "notional": float(max(notional_values) if notional_values else np.nan),
                "persistence": float(max(persistence_values) if persistence_values else np.nan),
                "age_seconds": float(max(age_values) if age_values else np.nan),
                "resilience_score": float(max(resilience_values)),
            }
        )
    merged.sort(key=lambda record: (-record["resilience_score"], record["distance_bps"]))
    for slot, record in enumerate(merged[:max_count], start=1):
        record["slot"] = float(slot)
    return merged[:max_count]


def _merge_liquidity_zone_records(items: list[dict[str, Any]], per_kind_count: int = 2) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in items:
        try:
            kind = str(item["kind"])
            bucket_index = int(float(item["bucket_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault((kind, bucket_index), []).append(item)

    merged: list[dict[str, float | str]] = []
    for (kind, bucket_index), records in groups.items():
        def mean_value(name: str) -> float:
            values = []
            for record in records:
                try:
                    values.append(float(record[name]))
                except (KeyError, TypeError, ValueError):
                    continue
            return float(sum(values) / len(values)) if values else np.nan

        score = mean_value("score_mean") if "score_mean" in records[0] else mean_value("score")
        persistence = mean_value("persistence")
        rank_score = (score * persistence) if kind == "high" else ((1.0 / (1.0 + max(score, 0.0))) * persistence)
        merged.append(
            {
                "kind": kind,
                "bucket_index": float(bucket_index),
                "lower_price": mean_value("lower_price"),
                "upper_price": mean_value("upper_price"),
                "mid_price": mean_value("mid_price"),
                "distance_bps": mean_value("distance_bps"),
                "score": float(score),
                "notional": mean_value("notional_mean") if "notional_mean" in records[0] else mean_value("notional"),
                "persistence": float(persistence),
                "rank_score": float(rank_score),
            }
        )

    selected: list[dict[str, float | str]] = []
    for kind in ("high", "low"):
        kind_records = [record for record in merged if record["kind"] == kind]
        kind_records.sort(key=lambda record: (-float(record["rank_score"]), float(record["distance_bps"])))
        for slot, record in enumerate(kind_records[:per_kind_count], start=1):
            record = dict(record)
            record["slot"] = float(slot)
            selected.append(record)
    return selected


def _price_distance_bps(left: float, right: float) -> float:
    reference = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / reference * 10000.0


def _empty_market_frame(
    candle_times: pd.DatetimeIndex,
    output_index: Any,
) -> DataFrame:
    frame = DataFrame(index=candle_times)
    for column in BAR_NUMERIC_COLUMNS:
        frame[column] = np.nan
    for column in BAR_JSON_COLUMNS:
        frame[column] = None
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
    for column in BAR_JSON_COLUMNS:
        if column not in output.columns:
            output[column] = None
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
    features[f"{p}_summary_window_seconds"] = float(_summary_timedelta(cfg).total_seconds())
    features[f"{p}_summary_lag_seconds"] = float(cfg.summary_lag_seconds)
    features[f"{p}_coverage_ratio"] = primary["coverage_ratio"]
    features[f"{p}_missing_ratio"] = primary["missing_ratio"]
    features[f"{p}_gap_flag"] = primary["gap_flag"]
    features[f"{p}_ready"] = valid_primary.astype(float)
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

    _append_score_features(features, primary, cfg)
    features = _append_headline_features(features, aligned, primary, cfg)
    if cfg.include_diagnostics:
        features = _append_per_market_features(features, aligned, cfg)
        features = _append_comparison_features(features, aligned, cfg)
        return features
    return features.loc[:, _strategy_output_columns(features, cfg)]


def _append_headline_features(features: DataFrame, aligned: dict[str, DataFrame], primary: DataFrame, cfg: OrderbookContextFeatureConfig) -> DataFrame:
    """Build the compact strategy packet from lower-level order-book evidence.

    Narrative:
    Strategies should not need to interpret every raw order-book measurement.
    This packet compresses the candle into a small set of questions:
    usable data, long/short pressure, entry-context risk, wall bias,
    cross-market agreement, and the strongest wall/liquidity blocks. Scores
    stay numeric for Freqtrade compatibility; state columns are simple
    sign/severity codes for plots and rule filters.
    """

    p = cfg.prefix
    valid = features[f"{p}_ready"].eq(1.0)
    coverage = pd.to_numeric(features[f"{p}_coverage_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    recent_gap_ratio = pd.to_numeric(features[f"{p}_gap_ratio_roll_medium"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    features[f"{p}_quality_score"] = _clip01((0.75 * coverage) + (0.25 * (1.0 - recent_gap_ratio))).where(valid, 0.0)
    features[f"{p}_spread_bps"] = primary["spread_bps_mean"].where(valid)
    spread_load = (primary["spread_bps_mean"] / cfg.spread_penalty_bps).clip(0.0, 1.0)
    features[f"{p}_spread_state"] = Series(
        np.select([spread_load.ge(1.0), spread_load.ge(0.65)], [2.0, 1.0], default=0.0),
        index=features.index,
    ).where(valid)

    pressure_current = pd.to_numeric(features[f"{p}_pressure_delta"], errors="coerce").clip(-1.0, 1.0)
    pressure_medium = pd.to_numeric(features[f"{p}_pressure_delta_roll_medium"], errors="coerce").clip(-1.0, 1.0)
    features[f"{p}_pressure_score"] = ((0.65 * pressure_current) + (0.35 * pressure_medium)).clip(-1.0, 1.0).where(valid)
    features[f"{p}_pressure_state"] = _signed_state(features[f"{p}_pressure_score"], weak=0.02, strong=0.06).where(valid)

    features[f"{p}_wall_score"] = features[f"{p}_wall_support_resistance_delta"].clip(-1.0, 1.0).where(valid)
    features[f"{p}_wall_state"] = _signed_state(features[f"{p}_wall_score"], weak=0.20, strong=0.55).where(valid)
    features[f"{p}_wall_box_score"] = pd.concat(
        [features[f"{p}_wall_support_score"], features[f"{p}_wall_resistance_score"]],
        axis=1,
    ).min(axis=1).where(valid)

    _append_market_headline_features(features, aligned, cfg)

    book_bias = (features[f"{p}_score_long"] - features[f"{p}_score_short"]).clip(-1.0, 1.0)
    features[f"{p}_book_bias_score"] = book_bias.where(valid)
    features[f"{p}_book_state"] = _signed_state(features[f"{p}_book_bias_score"], weak=0.02, strong=0.05).where(valid)
    features[f"{p}_state"] = Series(
        np.select([features[f"{p}_book_bias_score"].ge(0.02), features[f"{p}_book_bias_score"].le(-0.02)], [1.0, -1.0], default=0.0),
        index=features.index,
    ).where(valid, np.nan)
    features[f"{p}_score_abs"] = features[f"{p}_book_bias_score"].abs().clip(0.0, 1.0).where(valid)

    flip_denominator = float(max(int(cfg.medium_window) - 1, 1))
    pressure_flip_risk = (features[f"{p}_pressure_flip_count_roll_medium"] / flip_denominator).clip(0.0, 1.0)
    market_ready_risk = 1.0 - pd.to_numeric(features[f"{p}_market_ready_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    market_disagreement_risk = (-pd.to_numeric(features[f"{p}_market_agreement_score"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    risk_score = _clip01(
        (0.30 * primary["missing_ratio"].fillna(1.0).clip(0.0, 1.0))
        + (0.30 * spread_load.fillna(1.0))
        + (0.20 * pressure_flip_risk.fillna(1.0))
        + (0.10 * market_ready_risk)
        + (0.10 * market_disagreement_risk)
    )
    features[f"{p}_risk_score"] = risk_score.where(valid, 1.0)
    features[f"{p}_risk_state"] = Series(
        np.select([features[f"{p}_risk_score"].ge(0.65), features[f"{p}_risk_score"].ge(0.40)], [2.0, 1.0], default=0.0),
        index=features.index,
    )
    features[f"{p}_risk_block"] = ((~valid) | features[f"{p}_risk_state"].ge(2.0)).astype(float)
    return pd.concat(
        [
            features,
            _wall_block_feature_frame(primary, cfg, valid),
            _liquidity_zone_feature_frame(primary, cfg, valid),
        ],
        axis=1,
    )


def _wall_block_feature_frame(primary: DataFrame, cfg: OrderbookContextFeatureConfig, valid: Series) -> DataFrame:
    p = cfg.prefix
    columns: dict[str, Series] = {}
    for side in ("bid", "ask"):
        column = f"{side}_wall_blocks_json"
        records_by_row = primary[column].apply(_json_records) if column in primary.columns else Series([[]] * len(primary), index=primary.index)
        for slot in range(1, 4):
            slot_records = records_by_row.apply(lambda records, slot=slot: _slot_record(records, slot))
            columns[f"{p}_{side}_wall{slot}_price"] = slot_records.apply(lambda record: _record_float(record, "price")).where(valid)
            columns[f"{p}_{side}_wall{slot}_distance_bps"] = slot_records.apply(lambda record: _record_float(record, "distance_bps")).where(valid)
            columns[f"{p}_{side}_wall{slot}_score"] = slot_records.apply(lambda record: _record_float(record, "score")).where(valid)
            columns[f"{p}_{side}_wall{slot}_notional"] = slot_records.apply(lambda record: _record_float(record, "notional")).where(valid)
            columns[f"{p}_{side}_wall{slot}_persistence"] = slot_records.apply(lambda record: _record_float(record, "persistence")).where(valid)
            columns[f"{p}_{side}_wall{slot}_age_seconds"] = slot_records.apply(lambda record: _record_float(record, "age_seconds")).where(valid)
            columns[f"{p}_{side}_wall{slot}_resilience_score"] = slot_records.apply(lambda record: _record_float(record, "resilience_score")).where(valid)
    return DataFrame(columns, index=primary.index)


def _liquidity_zone_feature_frame(primary: DataFrame, cfg: OrderbookContextFeatureConfig, valid: Series) -> DataFrame:
    p = cfg.prefix
    columns: dict[str, Series] = {}
    for side in ("bid", "ask"):
        column = f"{side}_liquidity_zones_json"
        records_by_row = primary[column].apply(_json_records) if column in primary.columns else Series([[]] * len(primary), index=primary.index)
        for kind in ("high", "low"):
            for slot in range(1, 3):
                slot_records = records_by_row.apply(lambda records, kind=kind, slot=slot: _zone_slot_record(records, kind, slot))
                base = f"{p}_{side}_liq_{kind}{slot}"
                columns[f"{base}_lower_price"] = slot_records.apply(lambda record: _record_float(record, "lower_price")).where(valid)
                columns[f"{base}_upper_price"] = slot_records.apply(lambda record: _record_float(record, "upper_price")).where(valid)
                columns[f"{base}_mid_price"] = slot_records.apply(lambda record: _record_float(record, "mid_price")).where(valid)
                columns[f"{base}_distance_bps"] = slot_records.apply(lambda record: _record_float(record, "distance_bps")).where(valid)
                columns[f"{base}_score"] = slot_records.apply(lambda record: _record_float(record, "score")).where(valid)
                columns[f"{base}_notional"] = slot_records.apply(lambda record: _record_float(record, "notional")).where(valid)
                columns[f"{base}_persistence"] = slot_records.apply(lambda record: _record_float(record, "persistence")).where(valid)
    return DataFrame(columns, index=primary.index)


def _slot_record(records: list[dict[str, Any]], slot: int) -> dict[str, Any]:
    for record in records:
        try:
            if int(float(record.get("slot", 0))) == slot:
                return record
        except (TypeError, ValueError):
            continue
    return {}


def _zone_slot_record(records: list[dict[str, Any]], kind: str, slot: int) -> dict[str, Any]:
    for record in records:
        try:
            if str(record.get("kind")) == kind and int(float(record.get("slot", 0))) == slot:
                return record
        except (TypeError, ValueError):
            continue
    return {}


def _record_float(record: dict[str, Any], key: str) -> float:
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _append_market_headline_features(features: DataFrame, aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> None:
    p = cfg.prefix
    ready_columns: list[Series] = []
    pressure_columns: list[Series] = []
    for frame in aligned.values():
        market_valid = frame["gap_flag"].eq(0.0)
        ready_columns.append(market_valid.astype(float))
        pressure = (frame["bid_pressure_ratio"] - frame["ask_pressure_ratio"]).clip(-1.0, 1.0).where(market_valid)
        pressure_columns.append(pressure)

    if not ready_columns:
        features[f"{p}_market_ready_ratio"] = 0.0
        features[f"{p}_market_pressure_score"] = np.nan
        features[f"{p}_market_agreement_score"] = np.nan
        features[f"{p}_market_state"] = np.nan
        return

    ready_frame = pd.concat(ready_columns, axis=1)
    pressure_frame = pd.concat(pressure_columns, axis=1)
    features[f"{p}_market_ready_ratio"] = ready_frame.mean(axis=1).clip(0.0, 1.0)
    features[f"{p}_market_pressure_score"] = pressure_frame.mean(axis=1).clip(-1.0, 1.0)

    primary_sign = _direction_sign(features[f"{p}_pressure_score"], threshold=0.02)
    market_signs = pressure_frame.apply(lambda column: _direction_sign(column, threshold=0.02))
    directional_count = market_signs.ne(0.0).sum(axis=1).astype(float)
    primary_directional = primary_sign.ne(0.0).astype(float)
    same_count = market_signs.eq(primary_sign, axis=0).mul(primary_directional, axis=0).sum(axis=1).astype(float)
    opposed_count = market_signs.eq(-primary_sign, axis=0).mul(primary_directional, axis=0).sum(axis=1).astype(float)
    agreement = ((same_count - opposed_count) / directional_count.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
    features[f"{p}_market_agreement_score"] = agreement
    aligned_score = (agreement * features[f"{p}_market_ready_ratio"]).clip(-1.0, 1.0)
    features[f"{p}_market_state"] = Series(
        np.select([aligned_score.ge(0.66), aligned_score.ge(0.25), aligned_score.le(-0.66), aligned_score.le(-0.25)], [2.0, 1.0, -2.0, -1.0], default=0.0),
        index=features.index,
    )


def _append_per_market_features(features: DataFrame, aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> DataFrame:
    p = cfg.prefix
    columns: dict[str, Series] = {}
    for market_key, frame in aligned.items():
        market = _safe_column_part(market_key)
        columns[f"{p}_{market}_coverage_ratio"] = frame["coverage_ratio"]
        columns[f"{p}_{market}_gap_flag"] = frame["gap_flag"]
        columns[f"{p}_{market}_spread_bps_mean"] = frame["spread_bps_mean"]
        columns[f"{p}_{market}_imbalance_top20_mean"] = frame["imbalance_top20_mean"]
        columns[f"{p}_{market}_microprice_offset_bps_mean"] = frame["microprice_offset_bps_mean"]
        columns[f"{p}_{market}_bid_pressure_ratio"] = frame["bid_pressure_ratio"]
        columns[f"{p}_{market}_ask_pressure_ratio"] = frame["ask_pressure_ratio"]
        columns[f"{p}_{market}_nearest_bid_wall_distance_bps"] = frame["nearest_bid_wall_min_distance_bps"]
        columns[f"{p}_{market}_nearest_ask_wall_distance_bps"] = frame["nearest_ask_wall_min_distance_bps"]
        columns[f"{p}_{market}_bar_age_seconds"] = frame["bar_age_seconds"]
    if not columns:
        return features
    return pd.concat([features, DataFrame(columns, index=features.index)], axis=1)


def _append_comparison_features(features: DataFrame, aligned: dict[str, DataFrame], cfg: OrderbookContextFeatureConfig) -> DataFrame:
    p = cfg.prefix
    columns: dict[str, Series] = {}
    for left_key, right_key in cfg.comparison_pairs:
        if left_key not in aligned or right_key not in aligned:
            continue
        left = aligned[left_key]
        right = aligned[right_key]
        name = f"{_safe_column_part(left_key)}_vs_{_safe_column_part(right_key)}"
        both_valid = left["gap_flag"].eq(0.0) & right["gap_flag"].eq(0.0)
        columns[f"{p}_cmp_{name}_coverage_min"] = pd.concat(
            [left["coverage_ratio"], right["coverage_ratio"]], axis=1
        ).min(axis=1)
        columns[f"{p}_cmp_{name}_gap_flag"] = (~both_valid).astype(float)
        columns[f"{p}_cmp_{name}_spread_diff_bps"] = (right["spread_bps_mean"] - left["spread_bps_mean"]).where(both_valid)
        columns[f"{p}_cmp_{name}_spread_ratio"] = (
            right["spread_bps_mean"] / left["spread_bps_mean"].replace(0.0, np.nan)
        ).where(both_valid)
        columns[f"{p}_cmp_{name}_imbalance_divergence"] = (
            right["imbalance_top20_mean"] - left["imbalance_top20_mean"]
        ).where(both_valid)
        columns[f"{p}_cmp_{name}_microprice_divergence_bps"] = (
            right["microprice_offset_bps_mean"] - left["microprice_offset_bps_mean"]
        ).where(both_valid)
        columns[f"{p}_cmp_{name}_bid_pressure_diff"] = (right["bid_pressure_ratio"] - left["bid_pressure_ratio"]).where(both_valid)
        columns[f"{p}_cmp_{name}_ask_pressure_diff"] = (right["ask_pressure_ratio"] - left["ask_pressure_ratio"]).where(both_valid)
        left_pressure_delta = left["bid_pressure_ratio"] - left["ask_pressure_ratio"]
        right_pressure_delta = right["bid_pressure_ratio"] - right["ask_pressure_ratio"]
        columns[f"{p}_cmp_{name}_pressure_delta_diff"] = (right_pressure_delta - left_pressure_delta).where(both_valid)
        columns[f"{p}_cmp_{name}_pressure_direction_disagree"] = (
            np.sign(left_pressure_delta).ne(np.sign(right_pressure_delta)).astype(float)
        ).where(both_valid)
        columns[f"{p}_cmp_{name}_bid_wall_distance_diff_bps"] = (
            right["nearest_bid_wall_min_distance_bps"] - left["nearest_bid_wall_min_distance_bps"]
        ).where(both_valid)
        columns[f"{p}_cmp_{name}_ask_wall_distance_diff_bps"] = (
            right["nearest_ask_wall_min_distance_bps"] - left["nearest_ask_wall_min_distance_bps"]
        ).where(both_valid)
        left_wall_delta = left["nearest_ask_wall_min_distance_bps"] - left["nearest_bid_wall_min_distance_bps"]
        right_wall_delta = right["nearest_ask_wall_min_distance_bps"] - right["nearest_bid_wall_min_distance_bps"]
        columns[f"{p}_cmp_{name}_wall_delta_divergence_bps"] = (right_wall_delta - left_wall_delta).where(both_valid)
    if not columns:
        return features
    return pd.concat([features, DataFrame(columns, index=features.index)], axis=1)


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
    max_age = _summary_timedelta(cfg) * int(cfg.market_context_max_age_periods)
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
    for column in (f"{p}_{suffix}" for suffix in STRATEGY_OUTPUT_SUFFIXES):
        empty[column] = np.nan
    empty[f"{p}_ready"] = 0.0
    empty[f"{p}_quality_score"] = 0.0
    empty[f"{p}_coverage_ratio"] = 0.0
    empty[f"{p}_gap_flag"] = 1.0
    empty[f"{p}_risk_score"] = 1.0
    empty[f"{p}_risk_state"] = 2.0
    empty[f"{p}_risk_block"] = 1.0
    empty[f"{p}_market_ready_ratio"] = 0.0

    if cfg.include_diagnostics:
        diagnostic_columns = (
            f"{p}_mode_live",
            f"{p}_live_stream_enabled",
            f"{p}_source_bar_timeframe_seconds",
            f"{p}_summary_window_seconds",
            f"{p}_summary_lag_seconds",
            f"{p}_missing_ratio",
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
            f"{p}_wall_support_resistance_delta",
            f"{p}_bid_wall_score_roll_medium",
            f"{p}_ask_wall_score_roll_medium",
            f"{p}_liquidity_stress_score",
            f"{p}_support_pressure_score",
            f"{p}_resistance_pressure_score",
        )
        for column in diagnostic_columns:
            if column not in empty.columns:
                empty[column] = np.nan
        empty[f"{p}_mode_live"] = 1.0 if _effective_data_mode(cfg) == "live" else 0.0
        empty[f"{p}_live_stream_enabled"] = 1.0 if cfg.enable_live_stream else 0.0
        empty[f"{p}_source_bar_timeframe_seconds"] = float(cfg.bar_timeframe_seconds)
        empty[f"{p}_summary_window_seconds"] = float(_summary_timedelta(cfg).total_seconds())
        empty[f"{p}_summary_lag_seconds"] = float(cfg.summary_lag_seconds)

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

    if cfg.include_market_context:
        for market_key in cfg.market_keys:
            market = _safe_column_part(market_key)
            for column in MARKET_CONTEXT_NUMERIC_COLUMNS:
                empty[f"{p}_{market}_{column}"] = np.nan
            empty[f"{p}_{market}_market_context_age_seconds"] = np.nan
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


def _signed_state(series: Series, *, weak: float, strong: float) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return Series(
        np.select(
            [numeric.ge(strong), numeric.ge(weak), numeric.le(-strong), numeric.le(-weak)],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=numeric.index,
    )


def _direction_sign(series: Series, *, threshold: float) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return Series(
        np.select([numeric.ge(threshold), numeric.le(-threshold)], [1.0, -1.0], default=0.0),
        index=numeric.index,
    )


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


def _strategy_output_columns(features: DataFrame, cfg: OrderbookContextFeatureConfig) -> list[str]:
    p = cfg.prefix
    return [f"{p}_{suffix}" for suffix in STRATEGY_OUTPUT_SUFFIXES if f"{p}_{suffix}" in features.columns]


__all__ = [
    "OrderbookContextFeatureConfig",
    "add_orderbook_context_features",
    "format_orderbook_bars",
]
