from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import sqlite3

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


DEFAULT_RELEVANCE_WEIGHTS: dict[str, float] = {
    "high": 1.00,
    "medium": 0.65,
    "low": 0.35,
    "unknown": 0.45,
    "": 0.45,
}

DEFAULT_SOURCE_GROUP_WEIGHTS: dict[str, float] = {
    "exchange_announcements": 1.00,
    "regulators": 1.00,
    "central_banks": 0.95,
    "macro": 0.90,
    "crypto_media": 0.75,
    "defi_protocols": 0.70,
    "protocol_foundations": 0.70,
    "infrastructure": 0.65,
    "aggregators": 0.60,
    "general_news": 0.55,
    "unknown": 0.55,
    "": 0.55,
}

DEFAULT_SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    "rss": 0.75,
    "api": 0.85,
    "web": 0.80,
    "official": 1.00,
    "exchange": 1.00,
    "regulator": 1.00,
    "unknown": 0.60,
    "": 0.60,
}

DEFAULT_TOPIC_WEIGHTS: dict[str, float] = {
    "crypto": 0.90,
    "btc": 1.00,
    "bitcoin": 1.00,
    "ethereum": 0.85,
    "stablecoins": 0.90,
    "defi": 0.80,
    "exchange": 0.95,
    "regulation": 1.00,
    "macro": 0.90,
    "markets": 0.75,
    "unknown": 0.55,
    "": 0.55,
}

POSITIVE_TERMS: tuple[str, ...] = (
    "accumulat",
    "adoption",
    "approved",
    "approval",
    "breakout",
    "bullish",
    "buyback",
    "court win",
    "etf inflow",
    "expands",
    "funding",
    "inflows",
    "integrates",
    "investment",
    "launch",
    "listing",
    "partnership",
    "record high",
    "reopen",
    "rally",
    "settlement",
    "surge",
    "upgrade",
    "wins lawsuit",
)

NEGATIVE_TERMS: tuple[str, ...] = (
    "attack",
    "bankrupt",
    "ban",
    "charged",
    "crash",
    "cftc sues",
    "delist",
    "exploit",
    "fine",
    "fraud",
    "hack",
    "halt",
    "insolvency",
    "investigation",
    "lawsuit",
    "liquidation",
    "outage",
    "plunge",
    "probe",
    "rug",
    "sanction",
    "sec sues",
    "selloff",
    "suspend",
    "withdrawal freeze",
)

SEVERITY_TERMS: tuple[str, ...] = (
    "bankruptcy",
    "ban",
    "billion",
    "breaking",
    "cftc",
    "court",
    "critical",
    "delist",
    "emergency",
    "exploit",
    "hack",
    "lawsuit",
    "liquidation",
    "major",
    "record",
    "sec",
    "systemic",
    "urgent",
)

POSITIVE_IMPACT_VALUES = {"positive", "bullish", "greed", "up", "long"}
NEGATIVE_IMPACT_VALUES = {"negative", "bearish", "fear", "down", "short"}


@dataclass(frozen=True)
class NewsWebSentimentFeatureConfig:
    """First-pass News/Web formatter and feature settings.

    Objective:
    Convert existing launcher News and Web SQLite article rows into
    candle-aligned, no-trading-decision dataframe features for 1h research.

    First-pass scope:
    - Use free collected data already in ``news_events.sqlite`` and
      ``web_events.sqlite``.
    - Define explicit, auditable rule weights for source relevance, article
      direction, severity, counts, and persistence.
    - Output formatter/features only. This module does not decide entries,
      exits, sizing, or risk actions.

    Why this lives in ``Indicators``:
    Freqtrade strategies can import it like an indicator helper, but the file is
    intentionally a data formatter/feature builder. A later strategy agent
    should validate whether these columns sort forward returns before using
    them as filters.

    Relevant SQLite tables:
    ``articles(id, source_id, source_group, source_type, region, topic,
    market_relevance, source_url, canonical_url, title, summary, published_at,
    collected_at, event_type, impact_direction, impact_scope, severity,
    confidence, ...)``

    ``article_scores(article_id, market_relevance_score,
    crypto_relevance_score, tradfi_relevance_score, macro_relevance_score,
    urgency_score, duplicate_penalty, final_priority_score, ...)``

    Agent notes:
    - Treat the current keyword lists as transparent defaults, not truth.
    - Prefer validating score usefulness with forward-return buckets before
      adding heavier NLP or LLM classification.
    - If heavier NLP is added later, store model outputs beside event-level
      scores rather than hiding them inside strategy logic.
    """

    news_db_path: str | Path | None = None
    web_db_path: str | Path | None = None
    datasets: tuple[str, ...] = ("news", "web")
    resample_rule: str = "1h"
    availability_lag_candles: int = 1
    short_window: int = 6
    medium_window: int = 24
    long_window: int = 72
    persistence_window: int = 24
    positive_threshold: float = 0.20
    negative_threshold: float = 0.20
    severity_threshold: float = 0.45
    persistence_min: float = 0.35
    activity_scale: float = 8.0
    prefix: str = "nwctx"
    allow_missing: bool = False
    relevance_weights: dict[str, float] | None = None
    source_group_weights: dict[str, float] | None = None
    source_type_weights: dict[str, float] | None = None
    topic_weights: dict[str, float] | None = None


def format_news_web_events(
    config: NewsWebSentimentFeatureConfig | None = None,
    *,
    news_db_path: str | Path | None = None,
    web_db_path: str | Path | None = None,
    datasets: tuple[str, ...] | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Load, dedupe, and score event-level News/Web rows.

    This returns one row per deduped article/event with explicit formatter
    columns such as ``sentiment_score``, ``severity_score``,
    ``relevance_weight``, and weighted positive/negative event scores. It is
    useful for inspecting why rolling dataframe features move.
    """

    cfg = _resolve_config(
        config,
        news_db_path=news_db_path,
        web_db_path=web_db_path,
        datasets=datasets,
        allow_missing=allow_missing,
    )
    _validate_config(cfg)
    raw_events = _load_raw_events(cfg)
    if raw_events.empty:
        return raw_events
    return _format_events(raw_events, cfg)


def add_news_web_sentiment_features(
    dataframe: DataFrame,
    config: NewsWebSentimentFeatureConfig | None = None,
    *,
    news_db_path: str | Path | None = None,
    web_db_path: str | Path | None = None,
    datasets: tuple[str, ...] | None = None,
    resample_rule: str | None = None,
    availability_lag_candles: int | None = None,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    persistence_window: int | None = None,
    positive_threshold: float | None = None,
    negative_threshold: float | None = None,
    severity_threshold: float | None = None,
    persistence_min: float | None = None,
    activity_scale: float | None = None,
    prefix: str | None = None,
    allow_missing: bool | None = None,
) -> DataFrame:
    """Append first-pass News/Web rolling sentiment and persistence features.

    Output columns use the configured prefix, default ``nwctx``:
    - ``nwctx_article_count_roll_short/medium/long``: rolling event counts.
    - ``nwctx_weighted_count_roll_medium``: source/relevance-weighted volume.
    - ``nwctx_positive_score_roll_medium``: rolling positive event pressure.
    - ``nwctx_negative_score_roll_medium``: rolling negative event pressure.
    - ``nwctx_net_score_roll_medium``: positive minus negative pressure.
    - ``nwctx_severity_score_roll_medium``: weighted severity average.
    - ``nwctx_positive_persistence``: fraction of recent candles with positive
      event pressure above threshold.
    - ``nwctx_negative_persistence``: fraction of recent candles with negative
      event pressure above threshold.
    - ``nwctx_severity_persistence``: fraction of recent candles with notable
      event severity.
    - ``nwctx_score_long/short/abs/state``: shared score contract derived only
      from rolling event pressure and persistence. State is -1 Fear, 0 Neutral,
      1 Greed.

    This is a formatter/feature layer, not a final tradable indicator. The next
    agent should test whether rolling score and persistence outputs have stable
    forward-return value before wiring them into strategies.
    """

    cfg = _resolve_config(
        config,
        news_db_path=news_db_path,
        web_db_path=web_db_path,
        datasets=datasets,
        resample_rule=resample_rule,
        availability_lag_candles=availability_lag_candles,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        persistence_window=persistence_window,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        severity_threshold=severity_threshold,
        persistence_min=persistence_min,
        activity_scale=activity_scale,
        prefix=prefix,
        allow_missing=allow_missing,
    )
    _validate_config(cfg)
    if dataframe.empty:
        return dataframe.copy()

    frame = dataframe.copy()
    candle_times = _candle_times(frame)
    events = format_news_web_events(cfg)
    if events.empty:
        if cfg.allow_missing:
            return _append_empty_features(frame, cfg)
        raise ValueError("News/Web SQLite exists but contains no article rows.")

    periods = _events_by_period(events, cfg)
    aligned = _align_periods_to_candles(periods, candle_times, frame.index, cfg)
    features = _rolling_news_web_features(aligned, cfg)
    existing = [column for column in frame.columns if str(column).startswith(f"{cfg.prefix}_")]
    if existing:
        frame = frame.drop(columns=existing)
    return pd.concat([frame, features], axis=1)


def _resolve_config(config: NewsWebSentimentFeatureConfig | None, **overrides: Any) -> NewsWebSentimentFeatureConfig:
    cfg = config or NewsWebSentimentFeatureConfig()
    values = {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return NewsWebSentimentFeatureConfig(**values)


def _validate_config(cfg: NewsWebSentimentFeatureConfig) -> None:
    valid_datasets = {"news", "web"}
    unknown = sorted(set(cfg.datasets) - valid_datasets)
    if unknown:
        raise ValueError(f"News/Web datasets must be drawn from {sorted(valid_datasets)}. Got: {unknown}")
    if not cfg.datasets:
        raise ValueError("News/Web datasets must not be empty.")
    if cfg.short_window < 1 or cfg.medium_window < 1 or cfg.long_window < 1:
        raise ValueError("News/Web rolling windows must be >= 1.")
    if cfg.persistence_window < 1:
        raise ValueError("News/Web persistence_window must be >= 1.")
    if cfg.availability_lag_candles < 0:
        raise ValueError("News/Web availability_lag_candles must be >= 0.")
    if cfg.activity_scale <= 0:
        raise ValueError("News/Web activity_scale must be > 0.")
    if not 0.0 <= cfg.positive_threshold <= 1.0 or not 0.0 <= cfg.negative_threshold <= 1.0:
        raise ValueError("News/Web positive/negative thresholds must be normalized 0..1 values.")
    if not 0.0 <= cfg.severity_threshold <= 1.0:
        raise ValueError("News/Web severity_threshold must be a normalized 0..1 value.")
    if not 0.0 <= cfg.persistence_min <= 1.0:
        raise ValueError("News/Web persistence_min must be a normalized 0..1 value.")
    if not cfg.prefix:
        raise ValueError("News/Web prefix must not be empty.")


def _user_data_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_db_path(dataset: str) -> Path:
    if dataset == "news":
        return _user_data_root() / "research_news_data" / "news" / "news_events.sqlite"
    if dataset == "web":
        return _user_data_root() / "research_news_data" / "web" / "web_events.sqlite"
    raise ValueError(f"Unknown News/Web dataset: {dataset}")


def _resolved_db_paths(cfg: NewsWebSentimentFeatureConfig) -> dict[str, Path]:
    return {
        "news": Path(cfg.news_db_path).expanduser() if cfg.news_db_path else _default_db_path("news"),
        "web": Path(cfg.web_db_path).expanduser() if cfg.web_db_path else _default_db_path("web"),
    }


def _candle_times(dataframe: DataFrame) -> pd.DatetimeIndex:
    if "date" in dataframe.columns:
        values = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
    elif isinstance(dataframe.index, pd.DatetimeIndex):
        values = pd.to_datetime(dataframe.index, utc=True, errors="coerce")
    else:
        raise ValueError("News/Web features require a 'date' column or a DatetimeIndex.")
    if values.isna().any():
        raise ValueError("News/Web features found invalid candle timestamps.")
    return pd.DatetimeIndex(values)


def _load_raw_events(cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    frames: list[DataFrame] = []
    paths = _resolved_db_paths(cfg)
    for dataset in cfg.datasets:
        db_path = paths[dataset]
        if not db_path.exists():
            if cfg.allow_missing:
                continue
            raise FileNotFoundError(f"News/Web SQLite not found for {dataset}: {db_path}")
        frame = _read_events_from_db(db_path, dataset, cfg)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_events_from_db(db_path: Path, dataset: str, cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    query = """
        SELECT
            a.id,
            ? AS dataset,
            a.source_id,
            a.source_group,
            a.source_type,
            a.region,
            a.topic,
            a.market_relevance,
            a.source_url,
            a.canonical_url,
            a.title,
            a.summary,
            a.published_at,
            a.collected_at,
            a.event_type,
            a.impact_direction,
            a.impact_scope,
            a.severity,
            a.confidence,
            s.market_relevance_score,
            s.crypto_relevance_score,
            s.tradfi_relevance_score,
            s.macro_relevance_score,
            s.urgency_score,
            s.duplicate_penalty,
            s.final_priority_score
        FROM articles a
        LEFT JOIN article_scores s ON s.article_id = a.id
        WHERE COALESCE(a.published_at, a.collected_at) IS NOT NULL
        ORDER BY COALESCE(a.published_at, a.collected_at), a.source_id
    """

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        return pd.read_sql_query(query, conn, params=[dataset])
    except sqlite3.Error as exc:
        if cfg.allow_missing:
            return DataFrame()
        raise RuntimeError(f"Could not read News/Web SQLite for {dataset}: {db_path}") from exc
    finally:
        if conn is not None:
            conn.close()


def _format_events(events: DataFrame, cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    prepared = events.copy()
    published_at = pd.to_datetime(prepared["published_at"], utc=True, errors="coerce")
    collected_at = pd.to_datetime(prepared["collected_at"], utc=True, errors="coerce")
    prepared["event_time"] = published_at.fillna(collected_at)
    prepared = prepared.dropna(subset=["event_time"])

    text_cols = [
        "dataset",
        "source_id",
        "source_group",
        "source_type",
        "region",
        "topic",
        "market_relevance",
        "source_url",
        "canonical_url",
        "title",
        "summary",
        "event_type",
        "impact_direction",
        "impact_scope",
    ]
    for column in text_cols:
        prepared[column] = prepared[column].fillna("").astype(str)

    prepared["event_key"] = prepared.apply(_event_key, axis=1)
    prepared = prepared.sort_values(["event_time", "dataset", "source_id", "title"])
    prepared = prepared.drop_duplicates("event_key", keep="first")
    prepared["event_text"] = (
        prepared["title"] + " " + prepared["summary"] + " " + prepared["topic"] + " " + prepared["event_type"]
    ).str.lower()

    prepared["positive_hits"] = prepared["event_text"].map(lambda value: _keyword_hits(value, POSITIVE_TERMS))
    prepared["negative_hits"] = prepared["event_text"].map(lambda value: _keyword_hits(value, NEGATIVE_TERMS))
    prepared["severity_hits"] = prepared["event_text"].map(lambda value: _keyword_hits(value, SEVERITY_TERMS))

    prepared["collector_priority_norm"] = _normalise_score_0_100(prepared["final_priority_score"])
    prepared["collector_urgency_norm"] = _normalise_score_0_100(prepared["urgency_score"])
    prepared["collector_severity_norm"] = _normalise_score_0_100(prepared["severity"])
    prepared["confidence_norm"] = _normalise_score_0_100(prepared["confidence"]).fillna(0.50).clip(0.25, 1.00)
    prepared["relevance_weight"] = _event_relevance_weight(prepared, cfg)
    prepared["sentiment_score"] = _event_sentiment_score(prepared)
    prepared["severity_score"] = _event_severity_score(prepared)
    prepared["event_weight"] = (
        prepared["relevance_weight"]
        * prepared["confidence_norm"]
        * (0.50 + (prepared["severity_score"] * 0.50))
    ).clip(0.0, 1.5)
    prepared["positive_weighted"] = prepared["sentiment_score"].clip(lower=0.0) * prepared["event_weight"]
    prepared["negative_weighted"] = (-prepared["sentiment_score"].clip(upper=0.0)) * prepared["event_weight"]
    prepared["net_weighted"] = prepared["positive_weighted"] - prepared["negative_weighted"]
    prepared["severity_weighted"] = prepared["severity_score"] * prepared["event_weight"]
    prepared["weighted_count"] = prepared["event_weight"]

    output_columns = [
        "event_time",
        "event_key",
        "dataset",
        "source_id",
        "source_group",
        "source_type",
        "market_relevance",
        "topic",
        "title",
        "canonical_url",
        "source_url",
        "positive_hits",
        "negative_hits",
        "severity_hits",
        "collector_priority_norm",
        "collector_urgency_norm",
        "confidence_norm",
        "relevance_weight",
        "sentiment_score",
        "severity_score",
        "event_weight",
        "weighted_count",
        "positive_weighted",
        "negative_weighted",
        "net_weighted",
        "severity_weighted",
    ]
    return prepared[output_columns].sort_values("event_time").reset_index(drop=True)


def _events_by_period(events: DataFrame, cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    prepared = events.set_index("event_time").sort_index()
    resampled = prepared.resample(cfg.resample_rule)

    periods = DataFrame(index=resampled.size().index)
    periods["article_count"] = resampled["event_key"].count().astype(float)
    periods["weighted_count"] = resampled["weighted_count"].sum()
    periods["positive_weighted"] = resampled["positive_weighted"].sum()
    periods["negative_weighted"] = resampled["negative_weighted"].sum()
    periods["net_weighted"] = periods["positive_weighted"] - periods["negative_weighted"]
    periods["severity_weighted"] = resampled["severity_weighted"].sum()
    periods["source_breadth"] = resampled["source_id"].nunique().astype(float)
    periods["max_priority"] = resampled["collector_priority_norm"].max().fillna(0.0)

    for dataset in cfg.datasets:
        periods[f"{dataset}_article_count"] = (
            prepared["dataset"].eq(dataset).astype(float).resample(cfg.resample_rule).sum()
        )

    return periods.fillna(0.0)


def _align_periods_to_candles(
    periods: DataFrame,
    candle_times: pd.DatetimeIndex,
    output_index: Any,
    cfg: NewsWebSentimentFeatureConfig,
) -> DataFrame:
    aligned = periods.reindex(candle_times, fill_value=0.0)
    if cfg.availability_lag_candles:
        aligned = aligned.shift(cfg.availability_lag_candles).fillna(0.0)
    aligned.index = output_index
    return aligned


def _rolling_news_web_features(aligned: DataFrame, cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    p = cfg.prefix
    features = DataFrame(index=aligned.index)
    weighted_count_medium = aligned["weighted_count"].rolling(cfg.medium_window, min_periods=1).sum()
    weighted_count_safe = weighted_count_medium.replace(0.0, np.nan)
    positive_medium = aligned["positive_weighted"].rolling(cfg.medium_window, min_periods=1).sum()
    negative_medium = aligned["negative_weighted"].rolling(cfg.medium_window, min_periods=1).sum()
    net_medium = positive_medium - negative_medium
    severity_medium = (
        aligned["severity_weighted"].rolling(cfg.medium_window, min_periods=1).sum() / weighted_count_safe
    ).fillna(0.0)

    positive_intensity = (positive_medium / weighted_count_safe).fillna(0.0).clip(0.0, 1.0)
    negative_intensity = (negative_medium / weighted_count_safe).fillna(0.0).clip(0.0, 1.0)
    net_intensity = (net_medium / weighted_count_safe).fillna(0.0).clip(-1.0, 1.0)
    attention_score = (1.0 - np.exp(-weighted_count_medium / cfg.activity_scale)).clip(0.0, 1.0)

    features[f"{p}_article_count_roll_short"] = aligned["article_count"].rolling(cfg.short_window, min_periods=1).sum()
    features[f"{p}_article_count_roll_medium"] = aligned["article_count"].rolling(cfg.medium_window, min_periods=1).sum()
    features[f"{p}_article_count_roll_long"] = aligned["article_count"].rolling(cfg.long_window, min_periods=1).sum()
    features[f"{p}_weighted_count_roll_medium"] = weighted_count_medium
    features[f"{p}_source_breadth_roll_medium"] = aligned["source_breadth"].rolling(cfg.medium_window, min_periods=1).sum()
    features[f"{p}_positive_score_roll_medium"] = positive_intensity
    features[f"{p}_negative_score_roll_medium"] = negative_intensity
    features[f"{p}_net_score_roll_medium"] = net_intensity
    features[f"{p}_severity_score_roll_medium"] = severity_medium.clip(0.0, 1.0)
    features[f"{p}_attention_score"] = attention_score
    features[f"{p}_max_priority_roll_medium"] = aligned["max_priority"].rolling(cfg.medium_window, min_periods=1).max()
    features[f"{p}_positive_persistence"] = (
        aligned["positive_weighted"].gt(cfg.positive_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
    )
    features[f"{p}_negative_persistence"] = (
        aligned["negative_weighted"].gt(cfg.negative_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
    )
    features[f"{p}_severity_persistence"] = (
        severity_medium.gt(cfg.severity_threshold).rolling(cfg.persistence_window, min_periods=1).mean()
    )

    for dataset in cfg.datasets:
        column = f"{dataset}_article_count"
        if column in aligned.columns:
            features[f"{p}_{dataset}_count_roll_medium"] = aligned[column].rolling(cfg.medium_window, min_periods=1).sum()

    long_score = _clip01(0.50 + (net_intensity * 0.50))
    short_score = _clip01(0.50 - (net_intensity * 0.50))
    abs_score = _clip01(np.maximum(positive_intensity, negative_intensity) * attention_score)
    greed_state = (
        positive_intensity.ge(cfg.positive_threshold)
        & features[f"{p}_positive_persistence"].ge(cfg.persistence_min)
        & net_intensity.gt(0.0)
    )
    fear_state = (
        negative_intensity.ge(cfg.negative_threshold)
        & features[f"{p}_negative_persistence"].ge(cfg.persistence_min)
        & net_intensity.lt(0.0)
    )
    state = Series(np.select([greed_state, fear_state], [1.0, -1.0], default=0.0), index=features.index)
    features[f"{p}_score_long"] = long_score
    features[f"{p}_score_short"] = short_score
    features[f"{p}_score_abs"] = abs_score
    features[f"{p}_state"] = state
    return features


def _append_empty_features(dataframe: DataFrame, cfg: NewsWebSentimentFeatureConfig) -> DataFrame:
    frame = dataframe.copy()
    empty = DataFrame(index=frame.index)
    p = cfg.prefix
    for column in (
        f"{p}_article_count_roll_short",
        f"{p}_article_count_roll_medium",
        f"{p}_article_count_roll_long",
        f"{p}_weighted_count_roll_medium",
        f"{p}_source_breadth_roll_medium",
        f"{p}_positive_score_roll_medium",
        f"{p}_negative_score_roll_medium",
        f"{p}_net_score_roll_medium",
        f"{p}_severity_score_roll_medium",
        f"{p}_attention_score",
        f"{p}_max_priority_roll_medium",
        f"{p}_positive_persistence",
        f"{p}_negative_persistence",
        f"{p}_severity_persistence",
        f"{p}_score_long",
        f"{p}_score_short",
        f"{p}_score_abs",
        f"{p}_state",
    ):
        empty[column] = np.nan
    for dataset in cfg.datasets:
        empty[f"{p}_{dataset}_count_roll_medium"] = np.nan
    return pd.concat([frame, empty], axis=1)


def _event_key(row: Series) -> str:
    url = _normalise_url(row.get("canonical_url") or row.get("source_url") or "")
    if url:
        return f"url:{url}"
    title = _normalise_text(row.get("title") or "")
    source_id = _normalise_text(row.get("source_id") or "")
    date_part = str(row.get("event_time") or "")[:10]
    return f"title:{source_id}:{date_part}:{title}"


def _event_relevance_weight(events: DataFrame, cfg: NewsWebSentimentFeatureConfig) -> Series:
    relevance_weights = _merged_weights(DEFAULT_RELEVANCE_WEIGHTS, cfg.relevance_weights)
    source_group_weights = _merged_weights(DEFAULT_SOURCE_GROUP_WEIGHTS, cfg.source_group_weights)
    source_type_weights = _merged_weights(DEFAULT_SOURCE_TYPE_WEIGHTS, cfg.source_type_weights)
    topic_weights = _merged_weights(DEFAULT_TOPIC_WEIGHTS, cfg.topic_weights)

    relevance = _map_weight(events["market_relevance"], relevance_weights)
    source_group = _map_weight(events["source_group"], source_group_weights)
    source_type = _map_weight(events["source_type"], source_type_weights)
    topic = _map_weight(events["topic"], topic_weights)
    priority = events["collector_priority_norm"].fillna(0.0)

    combined = (
        (relevance * 0.35)
        + (source_group * 0.25)
        + (source_type * 0.15)
        + (topic * 0.10)
        + ((0.50 + (priority * 0.50)) * 0.15)
    )
    return combined.clip(0.0, 1.0)


def _event_sentiment_score(events: DataFrame) -> Series:
    positive_hits = pd.to_numeric(events["positive_hits"], errors="coerce").fillna(0.0)
    negative_hits = pd.to_numeric(events["negative_hits"], errors="coerce").fillna(0.0)
    total_hits = (positive_hits + negative_hits).replace(0.0, np.nan)
    lexicon_score = ((positive_hits - negative_hits) / total_hits).fillna(0.0).clip(-1.0, 1.0)
    impact_score = events["impact_direction"].map(_impact_direction_score).astype(float)
    has_impact = impact_score.ne(0.0)
    combined = lexicon_score.where(~has_impact, (lexicon_score * 0.70) + (impact_score * 0.30))
    return combined.clip(-1.0, 1.0)


def _event_severity_score(events: DataFrame) -> Series:
    severity_hits = pd.to_numeric(events["severity_hits"], errors="coerce").fillna(0.0)
    keyword_severity = (severity_hits / 4.0).clip(0.0, 1.0)
    collector_severity = events["collector_severity_norm"].fillna(0.0)
    urgency = events["collector_urgency_norm"].fillna(0.0)
    priority = events["collector_priority_norm"].fillna(0.0)
    values = DataFrame(
        {
            "keyword": keyword_severity,
            "collector": collector_severity,
            "urgency": urgency * 0.80,
            "priority": priority * 0.50,
        },
        index=events.index,
    )
    return values.max(axis=1).clip(0.0, 1.0)


def _keyword_hits(text: str, terms: tuple[str, ...]) -> int:
    normalised = str(text or "").lower()
    return sum(1 for term in terms if term in normalised)


def _impact_direction_score(value: Any) -> float:
    normalised = _normalise_text(value)
    if normalised in POSITIVE_IMPACT_VALUES:
        return 1.0
    if normalised in NEGATIVE_IMPACT_VALUES:
        return -1.0
    return 0.0


def _normalise_score_0_100(series: Series) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return numeric
    max_value = float(non_null.max())
    if max_value <= 1.0:
        return numeric.clip(0.0, 1.0)
    if max_value <= 10.0:
        return (numeric / 10.0).clip(0.0, 1.0)
    return (numeric / 100.0).clip(0.0, 1.0)


def _map_weight(series: Series, weights: dict[str, float]) -> Series:
    fallback = weights.get("unknown", 0.50)
    return series.map(lambda value: weights.get(_normalise_text(value), fallback)).astype(float)


def _merged_weights(defaults: dict[str, float], overrides: dict[str, float] | None) -> dict[str, float]:
    merged = {key: float(value) for key, value in defaults.items()}
    if overrides:
        for key, value in overrides.items():
            merged[_normalise_text(key)] = float(value)
    return merged


def _normalise_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"#.*$", "", text)
    text = re.sub(r"/+$", "", text)
    return text.lower()


def _normalise_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _clip01(value: Series | np.ndarray) -> Series:
    return pd.Series(value).clip(0.0, 1.0)


__all__ = [
    "NewsWebSentimentFeatureConfig",
    "add_news_web_sentiment_features",
    "format_news_web_events",
]
