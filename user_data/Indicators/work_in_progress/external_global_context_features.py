from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sqlite3

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class GlobalContextFeatureConfig:
    """First-pass Global Context feature settings.

    Objective:
    Convert the launcher Global Context SQLite scores into candle-aligned,
    no-trading-decision dataframe features for 1h strategy research.

    First-pass scope:
    - Use only existing effective scores from ``global_context_ticks.score``.
    - Emit rolling score and persistence columns only.
    - Do not infer article sentiment, entries, exits, sizing, or risk actions.
    - Keep the output easy for another agent to validate and extend.

    Data source:
    The collector stores rows in:
    ``user_data/research_news_data/global_context/global_context.sqlite``

    Relevant SQLite table:
    ``global_context_ticks(ts, source_ts, source_id, source_group, source_type,
    metric_key, score, source_score, calc_score, signal, value, unit, notes,
    raw_json, created_at)``

    Score convention:
    ``score`` is the effective 0..100 score shown in the launcher UI. Lower
    values mean Fear / weaker risk appetite. Higher values mean Greed / stronger
    risk appetite. Neutral is around 50.

    Strategy contract:
    This module appends context columns only. Strategies own final entry, exit,
    stake, and risk decisions.
    """

    db_path: str | Path | None = None
    resample_rule: str = "1h"
    availability_lag_candles: int = 1
    max_age_periods: int = 72
    short_window: int = 6
    medium_window: int = 24
    long_window: int = 72
    persistence_window: int = 24
    fear_threshold: float = 0.40
    greed_threshold: float = 0.60
    persistence_min: float = 0.50
    source_groups: tuple[str, ...] = ("sentiment", "market", "defi", "equity_indices", "rates_fx")
    prefix: str = "gctx"
    allow_missing: bool = False


def add_global_context_features(
    dataframe: DataFrame,
    config: GlobalContextFeatureConfig | None = None,
    *,
    db_path: str | Path | None = None,
    resample_rule: str | None = None,
    availability_lag_candles: int | None = None,
    max_age_periods: int | None = None,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    persistence_window: int | None = None,
    fear_threshold: float | None = None,
    greed_threshold: float | None = None,
    persistence_min: float | None = None,
    source_groups: tuple[str, ...] | None = None,
    prefix: str | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Append first-pass Global Context rolling score and persistence features.

    Expected use in a strategy:
    ``dataframe = add_global_context_features(dataframe, db_path=...)``

    Output columns use the configured prefix, default ``gctx``:
    - ``gctx_score_latest``: latest aligned effective score, normalized 0..1.
    - ``gctx_score_roll_short/medium/long``: rolling score means.
    - ``gctx_score_delta_short/medium``: score change over configured windows.
    - ``gctx_greed_persistence``: fraction of recent candles above greed threshold.
    - ``gctx_fear_persistence``: fraction of recent candles below fear threshold.
    - ``gctx_score_long/short/abs/state``: shared indicator score contract based
      only on the medium rolling score and persistence.
    - ``gctx_<group>_score_roll_medium`` plus group fear/greed persistence for
      source groups present in SQLite.

    This is intentionally not a final signal layer. The next agent should first
    validate whether rolling Global Context scores sort forward returns or act
    as useful filters before combining them with structure/order-book features.
    """

    cfg = _resolve_config(
        config,
        db_path=db_path,
        resample_rule=resample_rule,
        availability_lag_candles=availability_lag_candles,
        max_age_periods=max_age_periods,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        persistence_window=persistence_window,
        fear_threshold=fear_threshold,
        greed_threshold=greed_threshold,
        persistence_min=persistence_min,
        source_groups=source_groups,
        prefix=prefix,
        allow_missing=allow_missing,
    )
    _validate_config(cfg)
    if dataframe.empty:
        return dataframe.copy()

    frame = dataframe.copy()
    candle_times = _candle_times(frame)
    ticks = _load_context_ticks(cfg)
    if ticks.empty:
        if cfg.allow_missing:
            return _append_empty_features(frame, cfg)
        raise ValueError("Global Context SQLite exists but contains no scored rows.")

    context = _context_scores_by_period(ticks, cfg)
    aligned = _align_context_to_candles(context, candle_times, frame.index, cfg)
    features = _rolling_context_features(aligned, cfg)
    existing = [column for column in frame.columns if str(column).startswith(f"{cfg.prefix}_")]
    if existing:
        frame = frame.drop(columns=existing)
    return pd.concat([frame, features], axis=1)


def _resolve_config(config: GlobalContextFeatureConfig | None, **overrides: Any) -> GlobalContextFeatureConfig:
    cfg = config or GlobalContextFeatureConfig()
    values = {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return GlobalContextFeatureConfig(**values)


def _validate_config(cfg: GlobalContextFeatureConfig) -> None:
    if cfg.short_window < 1 or cfg.medium_window < 1 or cfg.long_window < 1:
        raise ValueError("Global Context rolling windows must be >= 1.")
    if cfg.persistence_window < 1:
        raise ValueError("Global Context persistence_window must be >= 1.")
    if cfg.max_age_periods < 1:
        raise ValueError("Global Context max_age_periods must be >= 1.")
    if cfg.availability_lag_candles < 0:
        raise ValueError("Global Context availability_lag_candles must be >= 0.")
    if not 0.0 <= cfg.fear_threshold <= 1.0 or not 0.0 <= cfg.greed_threshold <= 1.0:
        raise ValueError("Global Context fear/greed thresholds must be normalized 0..1 values.")
    if cfg.fear_threshold >= cfg.greed_threshold:
        raise ValueError("Global Context fear_threshold must be below greed_threshold.")
    if not cfg.prefix:
        raise ValueError("Global Context prefix must not be empty.")


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research_news_data" / "global_context" / "global_context.sqlite"


def _resolved_db_path(cfg: GlobalContextFeatureConfig) -> Path:
    return Path(cfg.db_path).expanduser() if cfg.db_path else _default_db_path()


def _candle_times(dataframe: DataFrame) -> pd.DatetimeIndex:
    if "date" in dataframe.columns:
        values = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    elif isinstance(dataframe.index, pd.DatetimeIndex):
        values = pd.to_datetime(dataframe.index, utc=True, errors="coerce")
    else:
        raise ValueError("Global Context features require a 'date' column or a DatetimeIndex.")
    if values.isna().any():
        raise ValueError("Global Context features found invalid candle timestamps.")
    return pd.DatetimeIndex(values)


def _load_context_ticks(cfg: GlobalContextFeatureConfig) -> DataFrame:
    db_path = _resolved_db_path(cfg)
    if not db_path.exists():
        if cfg.allow_missing:
            return DataFrame()
        raise FileNotFoundError(f"Global Context SQLite not found: {db_path}")

    query = """
        SELECT ts, source_group, metric_key, score
        FROM global_context_ticks
        WHERE score IS NOT NULL
    """
    params: list[Any] = []
    if cfg.source_groups:
        placeholders = ", ".join("?" for _ in cfg.source_groups)
        query += f" AND source_group IN ({placeholders})"
        params.extend(cfg.source_groups)
    query += " ORDER BY ts"

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        ticks = pd.read_sql_query(query, conn, params=params)
    except sqlite3.Error as exc:
        if cfg.allow_missing:
            return DataFrame()
        raise RuntimeError(f"Could not read Global Context SQLite: {db_path}") from exc
    finally:
        if conn is not None:
            conn.close()
    if ticks.empty:
        return ticks
    ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True, errors="coerce")
    ticks["score_norm"] = pd.to_numeric(ticks["score"], errors="coerce").clip(0.0, 100.0) / 100.0
    ticks["source_group"] = ticks["source_group"].astype(str)
    ticks["metric_key"] = ticks["metric_key"].astype(str)
    return ticks.dropna(subset=["ts", "score_norm"])


def _context_scores_by_period(ticks: DataFrame, cfg: GlobalContextFeatureConfig) -> DataFrame:
    prepared = ticks.copy()
    prepared["metric_id"] = prepared["source_group"] + ":" + prepared["metric_key"]
    wide = prepared.pivot_table(index="ts", columns="metric_id", values="score_norm", aggfunc="last").sort_index()
    wide = wide.resample(cfg.resample_rule).last().ffill(limit=cfg.max_age_periods)

    output = DataFrame(index=wide.index)
    output[f"{cfg.prefix}_score_latest"] = wide.mean(axis=1)
    for group in cfg.source_groups:
        group_cols = [column for column in wide.columns if str(column).startswith(f"{group}:")]
        if not group_cols:
            continue
        output[f"{cfg.prefix}_{_safe_column_part(group)}_score_latest"] = wide[group_cols].mean(axis=1)
    return output


def _align_context_to_candles(
    context: DataFrame,
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: GlobalContextFeatureConfig,
) -> DataFrame:
    aligned = context.reindex(candle_times, method="ffill", limit=cfg.max_age_periods)
    if cfg.availability_lag_candles:
        aligned = aligned.shift(cfg.availability_lag_candles)
    aligned.index = output_index
    return aligned


def _rolling_context_features(aligned: DataFrame, cfg: GlobalContextFeatureConfig) -> DataFrame:
    p = cfg.prefix
    features = DataFrame(index=aligned.index)
    latest = _clip01(aligned[f"{p}_score_latest"])
    short_roll = latest.rolling(cfg.short_window, min_periods=1).mean()
    medium_roll = latest.rolling(cfg.medium_window, min_periods=1).mean()
    long_roll = latest.rolling(cfg.long_window, min_periods=1).mean()
    greed_persistence = latest.ge(cfg.greed_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
    fear_persistence = latest.le(cfg.fear_threshold).rolling(cfg.persistence_window, min_periods=1).mean()

    features[f"{p}_score_latest"] = latest
    features[f"{p}_score_roll_short"] = short_roll
    features[f"{p}_score_roll_medium"] = medium_roll
    features[f"{p}_score_roll_long"] = long_roll
    features[f"{p}_score_delta_short"] = latest - latest.shift(cfg.short_window)
    features[f"{p}_score_delta_medium"] = latest - latest.shift(cfg.medium_window)
    features[f"{p}_greed_persistence"] = greed_persistence
    features[f"{p}_fear_persistence"] = fear_persistence

    long_score = _clip01(medium_roll)
    short_score = _clip01(1.0 - medium_roll)
    abs_score = _clip01((medium_roll - 0.5).abs() * 2.0)
    bull_state = medium_roll.ge(cfg.greed_threshold) & greed_persistence.ge(cfg.persistence_min)
    bear_state = medium_roll.le(cfg.fear_threshold) & fear_persistence.ge(cfg.persistence_min)
    state = Series(np.select([bull_state, bear_state], [1.0, -1.0], default=0.0), index=features.index)
    features[f"{p}_score_long"] = long_score
    features[f"{p}_score_short"] = short_score
    features[f"{p}_score_abs"] = abs_score
    features[f"{p}_state"] = state

    group_prefix = f"{p}_"
    for column in aligned.columns:
        if column == f"{p}_score_latest" or not str(column).startswith(group_prefix):
            continue
        group_name = str(column).removeprefix(group_prefix).removesuffix("_score_latest")
        group_latest = _clip01(aligned[column])
        features[f"{p}_{group_name}_score_roll_medium"] = group_latest.rolling(cfg.medium_window, min_periods=1).mean()
        features[f"{p}_{group_name}_greed_persistence"] = group_latest.ge(cfg.greed_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
        features[f"{p}_{group_name}_fear_persistence"] = group_latest.le(cfg.fear_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
    return features


def _append_empty_features(dataframe: DataFrame, cfg: GlobalContextFeatureConfig) -> DataFrame:
    frame = dataframe.copy()
    empty = DataFrame(index=frame.index)
    p = cfg.prefix
    for column in (
        f"{p}_score_latest",
        f"{p}_score_roll_short",
        f"{p}_score_roll_medium",
        f"{p}_score_roll_long",
        f"{p}_score_delta_short",
        f"{p}_score_delta_medium",
        f"{p}_greed_persistence",
        f"{p}_fear_persistence",
        f"{p}_score_long",
        f"{p}_score_short",
        f"{p}_score_abs",
        f"{p}_state",
    ):
        empty[column] = np.nan
    return pd.concat([frame, empty], axis=1)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def _safe_column_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_") or "group"


__all__ = ["GlobalContextFeatureConfig", "add_global_context_features"]
