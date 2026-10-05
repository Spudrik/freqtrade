"""Freeze a small, outcome-blind news/web story pilot.

The pilot exports only selected source records and their traceability metadata.  It
does not read price data, infer market direction, alter a collector database, or
enable a source.  The compact snapshot is intended to test whether the accumulated
live collection can support trustworthy story-level enrichment before any wider
historical backfill or confluence run.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import html
import json
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.builder import _duplicate_key


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_NEWS_DB = (
    USER_DATA_DIR / "research_news_data" / "news" / "news_events.sqlite"
)
DEFAULT_WEB_DB = USER_DATA_DIR / "research_news_data" / "web" / "web_events.sqlite"
DEFAULT_OUTPUT_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "semantic_data_pilot_20260904a"
)

SCHEMA_VERSION = "market_event_semantic_pilot_v1"
DEFAULT_INFORMATION_CUTOFF = pd.Timestamp("2026-09-04T19:00:00Z")
DEFAULT_SAMPLE_SIZE = 120
MAX_SAMPLE_SIZE = 300
FRESH_PUBLICATION_MIN_LAG_HOURS = -6.0
FRESH_PUBLICATION_MAX_LAG_HOURS = 72.0
EXACT_EPISODE_GAP_HOURS = 48.0
NEAR_DUPLICATE_GAP_HOURS = 48.0
NEAR_DUPLICATE_JACCARD = 0.82
NEAR_DUPLICATE_CONTAINMENT = 0.92
MIN_NEAR_DUPLICATE_TOKENS = 5
CANDIDATE_POOL_MULTIPLIER = 4
BLOCKS: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...] = (
    (
        "development_2026-06-01_2026-07-15",
        pd.Timestamp("2026-06-01T00:00:00Z"),
        pd.Timestamp("2026-07-16T00:00:00Z"),
    ),
    (
        "validation_2026-07-16_2026-08-31",
        pd.Timestamp("2026-07-16T00:00:00Z"),
        pd.Timestamp("2026-09-01T00:00:00Z"),
    ),
)

ALLOWED_DIRECTIONS = {"positive", "negative", "mixed", "unclear"}
ALLOWED_SEVERITIES = {"minor", "moderate", "major", "unclear"}
ALLOWED_LABEL_STATUS = {"pending", "complete"}
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class SourceSnapshot:
    source_family: str
    db_path: Path
    articles: DataFrame
    health_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


class UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def iso_utc(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def parse_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def normalise_title(value: Any) -> str:
    text = html.unescape(str(value or "")).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in normalise_title(value).split()
        if len(token) >= 3 and token not in TITLE_STOPWORDS
    )


def near_duplicate_titles(left: Any, right: Any) -> bool:
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if (
        len(left_tokens) < MIN_NEAR_DUPLICATE_TOKENS
        or len(right_tokens) < MIN_NEAR_DUPLICATE_TOKENS
    ):
        return False
    intersection = len(left_tokens & right_tokens)
    if intersection < 4:
        return False
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union if union else 0.0
    containment = intersection / min(len(left_tokens), len(right_tokens))
    return (
        jaccard >= NEAR_DUPLICATE_JACCARD
        or containment >= NEAR_DUPLICATE_CONTAINMENT
    )


def priority_band(value: Any) -> str:
    score = float(value or 0.0)
    if score >= 70.0:
        return "high"
    if score >= 30.0:
        return "medium"
    return "low"


def block_for_timestamp(value: pd.Timestamp) -> str | None:
    for name, start, end in BLOCKS:
        if start <= value < end:
            return name
    return None


def source_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def load_source_snapshot(
    path: Path,
    source_family: str,
    information_cutoff: pd.Timestamp,
) -> SourceSnapshot:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {source_family} database: {path}")
    file_before = path.stat()
    with closing(
        sqlite3.connect(source_uri(path), uri=True, timeout=30.0)
    ) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("BEGIN")
        data_version_start = int(conn.execute("PRAGMA data_version").fetchone()[0])
        articles = pd.read_sql_query(
            """
            SELECT
                a.id,
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
                a.updated_at,
                a.language,
                a.detected_assets_json,
                a.detected_entities_json,
                a.event_type,
                a.impact_direction,
                a.impact_scope,
                a.severity,
                a.confidence,
                a.raw_file_path,
                COALESCE(s.market_relevance_score, 0) AS market_relevance_score,
                COALESCE(s.crypto_relevance_score, 0) AS crypto_relevance_score,
                COALESCE(s.tradfi_relevance_score, 0) AS tradfi_relevance_score,
                COALESCE(s.macro_relevance_score, 0) AS macro_relevance_score,
                COALESCE(s.urgency_score, 0) AS urgency_score,
                COALESCE(s.duplicate_penalty, 0) AS duplicate_penalty,
                COALESCE(s.final_priority_score, 0) AS final_priority_score,
                COALESCE(s.scoring_version, 'missing') AS scoring_version
            FROM articles a
            LEFT JOIN article_scores s ON s.article_id = a.id
            ORDER BY a.collected_at, a.id
            """,
            conn,
        )
        health_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT source_id, source_group, source_type, enabled, last_success_at,
                       last_failure_at, last_error, last_http_status,
                       items_last_fetch, inserted_last_fetch, duplicates_last_fetch
                FROM sources
                ORDER BY source_group, source_id
                """
            )
        ]
        schema_rows = {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key, value FROM schema_meta ORDER BY key")
        }
        forward_return_count = int(
            conn.execute("SELECT COUNT(*) FROM article_forward_returns").fetchone()[0]
        )
        data_version_end = int(conn.execute("PRAGMA data_version").fetchone()[0])
        conn.rollback()

    file_after = path.stat()
    articles["source_family"] = source_family
    articles["article_key"] = source_family + ":" + articles["id"].astype(str)
    articles["collected_ts"] = parse_utc(articles["collected_at"])
    articles["published_ts"] = parse_utc(articles["published_at"])
    articles["publication_lag_hours"] = (
        articles["collected_ts"] - articles["published_ts"]
    ).dt.total_seconds() / 3600.0
    articles["whole_story_partition"] = articles["collected_ts"].map(
        lambda value: block_for_timestamp(value) if not pd.isna(value) else None
    )
    articles = articles[
        articles["collected_ts"].notna()
        & (articles["collected_ts"] <= information_cutoff)
    ].copy()

    publication_missing = articles["published_at"].fillna("").str.strip().eq("")
    publication_parsed = articles["published_ts"].notna()
    fresh_lag = articles["publication_lag_hours"].between(
        FRESH_PUBLICATION_MIN_LAG_HOURS,
        FRESH_PUBLICATION_MAX_LAG_HOURS,
        inclusive="both",
    )
    articles["publication_state"] = "fresh"
    articles.loc[publication_missing, "publication_state"] = "missing_use_observed_time"
    articles.loc[
        ~publication_missing & ~publication_parsed, "publication_state"
    ] = "unparseable_excluded"
    articles.loc[
        publication_parsed
        & (articles["publication_lag_hours"] > FRESH_PUBLICATION_MAX_LAG_HOURS),
        "publication_state",
    ] = "stale_backfill_excluded"
    articles.loc[
        publication_parsed
        & (articles["publication_lag_hours"] < FRESH_PUBLICATION_MIN_LAG_HOURS),
        "publication_state",
    ] = "future_dated_excluded"
    articles["eligible_for_pilot"] = (
        articles["whole_story_partition"].notna()
        & (publication_missing | (publication_parsed & fresh_lag))
    )
    articles["priority_band"] = articles["final_priority_score"].map(priority_band)
    articles["exact_title_key"] = articles["title"].map(_duplicate_key)

    metadata = {
        "source_family": source_family,
        "db_path": str(path.resolve()),
        "db_size_bytes_before": int(file_before.st_size),
        "db_size_bytes_after": int(file_after.st_size),
        "db_mtime_utc_before": iso_utc(pd.Timestamp(file_before.st_mtime, unit="s", tz="UTC")),
        "db_mtime_utc_after": iso_utc(pd.Timestamp(file_after.st_mtime, unit="s", tz="UTC")),
        "data_version_start": data_version_start,
        "data_version_end": data_version_end,
        "schema_meta": schema_rows,
        "forward_return_rows": forward_return_count,
        "rows_at_snapshot": len(articles),
        "max_collected_at": iso_utc(articles["collected_ts"].max()),
        "information_cutoff": iso_utc(information_cutoff),
        "query_transaction": "read-only SQLite snapshot transaction",
    }
    return SourceSnapshot(source_family, path, articles, health_rows, metadata)


def coverage_summary(frame: DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(frame),
        "missing_or_unparseable_collected_at": int(frame["collected_ts"].isna().sum()),
        "publication_states": {
            str(key): int(value)
            for key, value in frame["publication_state"].value_counts().items()
        },
        "semantic_fields": {
            "unknown_direction": int(
                frame["impact_direction"]
                .fillna("unknown")
                .str.strip()
                .str.lower()
                .isin({"", "unknown"})
                .sum()
            ),
            "missing_severity": int(frame["severity"].isna().sum()),
            "missing_confidence": int(frame["confidence"].isna().sum()),
        },
        "partitions": {},
    }
    for name, _start, _end in BLOCKS:
        block = frame[frame["whole_story_partition"] == name]
        eligible = block[block["eligible_for_pilot"]]
        result["partitions"][name] = {
            "all_rows": len(block),
            "eligible_rows": len(eligible),
            "source_ids": int(eligible["source_id"].nunique()),
            "source_groups": {
                str(key): int(value)
                for key, value in eligible["source_group"]
                .fillna("(missing)")
                .value_counts()
                .items()
            },
            "priority_bands": {
                str(key): int(value)
                for key, value in eligible["priority_band"].value_counts().items()
            },
        }
    return result


def assign_exact_episodes(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    eligible = frame[frame["eligible_for_pilot"]].copy()
    eligible = eligible.sort_values(["exact_title_key", "collected_ts", "article_key"])
    gap_hours = (
        eligible.groupby("exact_title_key", sort=False)["collected_ts"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )
    episode_start = gap_hours.isna() | gap_hours.gt(EXACT_EPISODE_GAP_HOURS)
    eligible["_episode_number"] = (
        episode_start.groupby(eligible["exact_title_key"]).cumsum().astype(int) - 1
    )
    eligible["_episode_first_seen"] = eligible.groupby(
        ["exact_title_key", "_episode_number"], sort=False
    )["collected_ts"].transform("min")
    eligible["exact_episode_id"] = [
        stable_hash(f"{title_key}|{number}|{iso_utc(first_seen)}")
        for title_key, number, first_seen in eligible[
            ["exact_title_key", "_episode_number", "_episode_first_seen"]
        ].itertuples(index=False, name=None)
    ]

    representatives = eligible.sort_values(
        ["final_priority_score", "collected_ts", "article_key"],
        ascending=[False, True, True],
    ).drop_duplicates("exact_episode_id")
    representatives = representatives[
        [
            "exact_episode_id",
            "article_key",
            "title",
            "summary",
            "source_family",
            "source_group",
            "source_id",
            "final_priority_score",
            "priority_band",
        ]
    ].rename(
        columns={
            "article_key": "representative_article_key",
            "title": "representative_title",
            "summary": "representative_summary",
            "source_family": "representative_source_family",
            "source_group": "representative_source_group",
            "source_id": "representative_source_id",
            "final_priority_score": "representative_priority",
            "priority_band": "representative_priority_band",
        }
    )

    exact = (
        eligible.groupby("exact_episode_id", sort=True)
        .agg(
            exact_title_key=("exact_title_key", "first"),
            article_keys=(
                "article_key",
                lambda values: sorted(values.astype(str).tolist()),
            ),
            article_count=("article_key", "size"),
            first_seen_at=("collected_ts", "min"),
            last_seen_at=("collected_ts", "max"),
            source_ids=(
                "source_id",
                lambda values: sorted(values.fillna("").astype(str).unique()),
            ),
            source_groups=(
                "source_group",
                lambda values: sorted(
                    values.fillna("(missing)").astype(str).unique()
                ),
            ),
        )
        .reset_index()
        .merge(representatives, on="exact_episode_id", how="left", validate="one_to_one")
    )
    exact["whole_story_partition"] = exact["first_seen_at"].map(
        block_for_timestamp
    )
    exact["representative_summary"] = exact["representative_summary"].fillna("")
    exact["representative_source_group"] = exact[
        "representative_source_group"
    ].fillna("(missing)")
    eligible = eligible.drop(columns=["_episode_number", "_episode_first_seen"])
    return eligible, exact


def deterministic_order(value: Any, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def balanced_select(
    frame: DataFrame,
    target: int,
    *,
    id_column: str,
    stratum_columns: Sequence[str],
    seed: str,
) -> DataFrame:
    if target <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    working["_stratum"] = working[list(stratum_columns)].fillna("(missing)").agg(
        "|".join, axis=1
    )
    working["_order"] = working[id_column].map(
        lambda value: deterministic_order(value, seed)
    )
    queues: dict[str, deque[int]] = {}
    for stratum, rows in working.groupby("_stratum", sort=True):
        ordered = rows.sort_values(["_order", id_column])
        queues[str(stratum)] = deque(ordered.index.tolist())
    selected: list[int] = []
    while len(selected) < min(target, len(working)):
        progressed = False
        for stratum in sorted(queues):
            queue = queues[stratum]
            if not queue:
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return working.loc[selected].drop(columns=["_stratum", "_order"]).copy()


def build_candidate_pool(exact: DataFrame, sample_size: int) -> DataFrame:
    per_block_targets = partition_targets(sample_size)
    selected: list[DataFrame] = []
    for block, target in per_block_targets.items():
        block_rows = exact[exact["whole_story_partition"] == block]
        pool_target = min(
            len(block_rows),
            max(target * CANDIDATE_POOL_MULTIPLIER, target + 40),
        )
        selected.append(
            balanced_select(
                block_rows,
                pool_target,
                id_column="exact_episode_id",
                stratum_columns=(
                    "representative_source_family",
                    "representative_source_group",
                    "representative_priority_band",
                ),
                seed=f"{SCHEMA_VERSION}|candidate|{block}",
            )
        )
    return pd.concat(selected, ignore_index=True) if selected else exact.iloc[0:0]


def merge_near_duplicate_candidates(candidates: DataFrame) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    records = candidates.sort_values(
        ["first_seen_at", "exact_episode_id"]
    ).to_dict(orient="records")
    union_find = UnionFind(str(row["exact_episode_id"]) for row in records)
    for left_index, left in enumerate(records):
        left_time = pd.Timestamp(left["first_seen_at"])
        for right in records[left_index + 1 :]:
            right_time = pd.Timestamp(right["first_seen_at"])
            gap = (right_time - left_time).total_seconds() / 3600.0
            if gap > NEAR_DUPLICATE_GAP_HOURS:
                break
            if near_duplicate_titles(
                left["representative_title"], right["representative_title"]
            ):
                union_find.union(
                    str(left["exact_episode_id"]), str(right["exact_episode_id"])
                )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[union_find.find(str(row["exact_episode_id"]))].append(row)
    stories = [summarise_story(group) for group in groups.values()]
    return sorted(stories, key=lambda row: (row["first_seen_at"], row["story_id"]))


def summarise_story(exact_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_groups = sorted(
        exact_groups,
        key=lambda row: (
            -float(row["representative_priority"]),
            pd.Timestamp(row["first_seen_at"]),
            str(row["exact_episode_id"]),
        ),
    )
    representative = exact_groups[0]
    article_keys = sorted(
        {
            str(article_key)
            for row in exact_groups
            for article_key in row["article_keys"]
        }
    )
    source_ids = sorted(
        {str(value) for row in exact_groups for value in row["source_ids"] if value}
    )
    source_groups = sorted(
        {str(value) for row in exact_groups for value in row["source_groups"] if value}
    )
    first_seen = min(pd.Timestamp(row["first_seen_at"]) for row in exact_groups)
    last_seen = max(pd.Timestamp(row["last_seen_at"]) for row in exact_groups)
    exact_ids = sorted(str(row["exact_episode_id"]) for row in exact_groups)
    if len(exact_groups) > 1:
        method = "near_title"
    elif len(article_keys) > 1:
        method = "exact_title"
    else:
        method = "singleton"
    return {
        "story_id": stable_hash("|".join(exact_ids)),
        "exact_episode_ids": exact_ids,
        "article_keys": article_keys,
        "article_count": len(article_keys),
        "first_seen_at": first_seen,
        "last_seen_at": last_seen,
        "whole_story_partition": block_for_timestamp(first_seen),
        "representative_article_key": str(
            representative["representative_article_key"]
        ),
        "representative_title": str(representative["representative_title"]),
        "representative_summary": str(representative["representative_summary"]),
        "representative_source_family": str(
            representative["representative_source_family"]
        ),
        "representative_source_group": str(
            representative["representative_source_group"]
        ),
        "representative_source_id": str(representative["representative_source_id"]),
        "representative_priority": float(
            representative["representative_priority"]
        ),
        "representative_priority_band": str(
            representative["representative_priority_band"]
        ),
        "source_ids": source_ids,
        "source_groups": source_groups,
        "independent_source_count": len(source_ids),
        "independent_source_group_count": len(source_groups),
        "duplicate_ratio": 0.0
        if len(article_keys) <= 1
        else round((len(article_keys) - 1) / len(article_keys), 6),
        "cluster_method": method,
    }


def partition_targets(sample_size: int) -> dict[str, int]:
    first = sample_size // 2
    return {BLOCKS[0][0]: first, BLOCKS[1][0]: sample_size - first}


def select_final_stories(stories: Sequence[Mapping[str, Any]], sample_size: int) -> DataFrame:
    frame = pd.DataFrame(stories)
    selected: list[DataFrame] = []
    for block, target in partition_targets(sample_size).items():
        block_rows = frame[frame["whole_story_partition"] == block]
        selected.append(
            balanced_select(
                block_rows,
                target,
                id_column="story_id",
                stratum_columns=(
                    "representative_source_family",
                    "representative_source_group",
                    "representative_priority_band",
                ),
                seed=f"{SCHEMA_VERSION}|final|{block}",
            )
        )
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True).sort_values(
        ["whole_story_partition", "first_seen_at", "story_id"]
    )


def read_relations(
    db_path: Path, source_family: str, article_ids: Sequence[str]
) -> dict[str, DataFrame]:
    relation_frames: dict[str, list[DataFrame]] = {
        "tags": [],
        "categories": [],
        "assets": [],
    }
    if not article_ids:
        return {key: DataFrame() for key in relation_frames}
    with closing(
        sqlite3.connect(source_uri(db_path), uri=True, timeout=30.0)
    ) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("BEGIN")
        for offset in range(0, len(article_ids), 800):
            chunk = list(article_ids[offset : offset + 800])
            placeholders = ",".join("?" for _ in chunk)
            relation_frames["tags"].append(
                pd.read_sql_query(
                    f"""
                    SELECT at.article_id, t.namespace, t.value, at.matched_by,
                           at.confidence
                    FROM article_tags at
                    JOIN tags t ON t.id = at.tag_id
                    WHERE at.article_id IN ({placeholders})
                    ORDER BY at.article_id, t.namespace, t.value
                    """,
                    conn,
                    params=chunk,
                )
            )
            relation_frames["categories"].append(
                pd.read_sql_query(
                    f"""
                    SELECT article_id, category, confidence
                    FROM article_categories
                    WHERE article_id IN ({placeholders})
                    ORDER BY article_id, category
                    """,
                    conn,
                    params=chunk,
                )
            )
            relation_frames["assets"].append(
                pd.read_sql_query(
                    f"""
                    SELECT article_id, asset, confidence
                    FROM article_assets
                    WHERE article_id IN ({placeholders})
                    ORDER BY article_id, asset
                    """,
                    conn,
                    params=chunk,
                )
            )
        conn.rollback()
    output: dict[str, DataFrame] = {}
    for key, frames in relation_frames.items():
        nonempty = [frame for frame in frames if not frame.empty]
        output[key] = pd.concat(nonempty, ignore_index=True) if nonempty else DataFrame()
        if not output[key].empty:
            output[key]["source_family"] = source_family
            output[key]["article_key"] = (
                source_family + ":" + output[key]["article_id"].astype(str)
            )
    return output


def json_values(frame: DataFrame, article_key: str, columns: Sequence[str]) -> list[Any]:
    if frame.empty or "article_key" not in frame:
        return []
    rows = frame[frame["article_key"] == article_key]
    values: list[Any] = []
    for row in rows[list(columns)].itertuples(index=False, name=None):
        values.append(
            row[0] if len(row) == 1 else dict(zip(columns, row, strict=True))
        )
    return values


def build_label_template(
    stories: DataFrame,
    relations: Mapping[str, DataFrame],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in stories.itertuples(index=False):
        member_keys = list(row.article_keys)
        tags: list[dict[str, Any]] = []
        categories: list[str] = []
        assets: list[str] = []
        for article_key in member_keys:
            tags.extend(
                json_values(
                    relations.get("tags", DataFrame()),
                    article_key,
                    ("namespace", "value", "matched_by", "confidence"),
                )
            )
            categories.extend(
                json_values(
                    relations.get("categories", DataFrame()),
                    article_key,
                    ("category",),
                )
            )
            assets.extend(
                json_values(
                    relations.get("assets", DataFrame()), article_key, ("asset",)
                )
            )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "story_id": str(row.story_id),
                "whole_story_partition": str(row.whole_story_partition),
                "first_seen_at": iso_utc(row.first_seen_at),
                "latest_seen_at": iso_utc(row.last_seen_at),
                "representative_article_key": str(row.representative_article_key),
                "representative_title": str(row.representative_title),
                "representative_summary": str(row.representative_summary),
                "article_keys": member_keys,
                "source_ids": list(row.source_ids),
                "source_groups": list(row.source_groups),
                "article_count": int(row.article_count),
                "independent_source_count": int(row.independent_source_count),
                "independent_source_group_count": int(
                    row.independent_source_group_count
                ),
                "duplicate_ratio": float(row.duplicate_ratio),
                "cluster_method": str(row.cluster_method),
                "existing_tags": sorted(
                    tags,
                    key=lambda value: (
                        str(value.get("namespace")),
                        str(value.get("value")),
                    ),
                ),
                "existing_categories": sorted(set(categories)),
                "existing_assets": sorted(set(assets)),
                "label_status": "pending",
                "topics": [],
                "impact_channels": [],
                "affected_entities": [],
                "affected_assets": [],
                "direction": None,
                "severity": None,
                "novelty": None,
                "update_state": None,
                "confidence": None,
                "evidence_article_keys": [],
                "label_notes": "",
                "classifier_model": "",
                "classifier_version": "",
                "prompt_version": "",
            }
        )
    return records


def dataframe_for_sqlite(frame: DataFrame) -> DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].map(iso_utc)
        elif output[column].map(
            lambda value: isinstance(value, (list, dict, tuple, set))
        ).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict, tuple, set))
                else value
            )
    return output


def write_compact_snapshot(
    path: Path,
    articles: DataFrame,
    stories: DataFrame,
    relations: Mapping[str, DataFrame],
    metadata: Mapping[str, Any],
) -> None:
    with closing(sqlite3.connect(path)) as conn:
        dataframe_for_sqlite(articles).to_sql(
            "pilot_articles", conn, index=False, if_exists="replace"
        )
        story_table = stories.copy()
        story_members = story_table[["story_id", "article_keys"]].explode(
            "article_keys"
        )
        story_members = story_members.rename(columns={"article_keys": "article_key"})
        story_table = story_table.drop(columns=["article_keys"])
        dataframe_for_sqlite(story_table).to_sql(
            "pilot_story_clusters", conn, index=False, if_exists="replace"
        )
        dataframe_for_sqlite(story_members).to_sql(
            "pilot_story_members", conn, index=False, if_exists="replace"
        )
        for key, frame in relations.items():
            dataframe_for_sqlite(frame).to_sql(
                f"pilot_{key}", conn, index=False, if_exists="replace"
            )
        conn.execute(
            "CREATE TABLE pilot_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO pilot_meta(key, value_json) VALUES (?, ?)",
            [
                (str(key), json.dumps(value, sort_keys=True))
                for key, value in metadata.items()
            ],
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_pilot_articles_key ON pilot_articles(article_key)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_pilot_stories_id ON pilot_story_clusters(story_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_pilot_members ON pilot_story_members(story_id, article_key)"
        )
        conn.commit()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def freeze_quality_report(
    stories: DataFrame,
    articles: DataFrame,
    labels: Sequence[Mapping[str, Any]],
    sample_size: int,
    information_cutoff: pd.Timestamp,
) -> dict[str, Any]:
    story_ids = stories["story_id"].astype(str)
    member_story_ids = {
        str(story_id)
        for story_id, keys in stories[["story_id", "article_keys"]].itertuples(
            index=False, name=None
        )
        if keys
    }
    article_keys = set(articles["article_key"].astype(str))
    referenced_keys = {
        str(key) for keys in stories["article_keys"] for key in list(keys)
    }
    partition_counts = {
        str(key): int(value)
        for key, value in stories["whole_story_partition"].value_counts().items()
    }
    checks = {
        "requested_sample_not_above_maximum": sample_size <= MAX_SAMPLE_SIZE,
        "selected_sample_matches_request": len(stories) == sample_size,
        "unique_story_ids": not story_ids.duplicated().any(),
        "all_stories_have_members": len(member_story_ids) == len(stories),
        "all_members_exported": referenced_keys == article_keys,
        "all_observed_by_information_cutoff": bool(
            (articles["collected_ts"] <= information_cutoff).all()
        ),
        "all_articles_fresh_or_publication_missing": bool(
            articles["publication_state"]
            .isin({"fresh", "missing_use_observed_time"})
            .all()
        ),
        "whole_story_partition_unique": bool(
            stories.groupby("story_id")["whole_story_partition"].nunique().le(1).all()
        ),
        "label_template_matches_story_ids": {
            str(record.get("story_id")) for record in labels
        }
        == set(story_ids),
        "no_price_or_outcome_data_read": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "freeze_and_sampling_quality",
        "checks": checks,
        "passed": all(checks.values()),
        "semantic_labels_ready": False,
        "market_test_ready": False,
        "market_test_scope_when_ready": (
            "direction-neutral BTC/ETH activity, volume, range/volatility, "
            "and pressure only"
        ),
        "selected_story_count": len(stories),
        "selected_article_count": len(articles),
        "partition_counts": partition_counts,
        "cluster_methods": {
            str(key): int(value)
            for key, value in stories["cluster_method"].value_counts().items()
        },
        "explanation": (
            "The freeze may pass while semantic_labels_ready remains false; labels "
            "and review are a separate gate."
        ),
    }


def validate_complete_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("direction") not in ALLOWED_DIRECTIONS:
        errors.append("invalid_direction")
    if record.get("severity") not in ALLOWED_SEVERITIES:
        errors.append("invalid_severity")
    confidence = record.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("invalid_confidence")
    if not isinstance(record.get("topics"), list) or not record.get("topics"):
        errors.append("topics_required")
    if not isinstance(record.get("impact_channels"), list):
        errors.append("impact_channels_must_be_list")
    evidence = record.get("evidence_article_keys")
    article_keys = set(record.get("article_keys") or [])
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence_article_keys_required")
    elif not set(evidence).issubset(article_keys):
        errors.append("evidence_not_in_story_members")
    for version_field in (
        "classifier_model",
        "classifier_version",
        "prompt_version",
    ):
        if not str(record.get(version_field) or "").strip():
            errors.append(f"{version_field}_required")
    return errors


def validate_completed_labels(
    records: Sequence[Mapping[str, Any]],
    expected_story_ids: set[str],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    completed = 0
    for line_number, record in enumerate(records, start=1):
        story_id = str(record.get("story_id") or "")
        row_errors: list[str] = []
        if not story_id or story_id not in expected_story_ids:
            row_errors.append("unknown_or_missing_story_id")
        if story_id in seen:
            row_errors.append("duplicate_story_id")
        seen.add(story_id)
        status = str(record.get("label_status") or "")
        if status not in ALLOWED_LABEL_STATUS:
            row_errors.append("invalid_label_status")
        if status == "complete":
            completed += 1
            row_errors.extend(validate_complete_record(record))
        if row_errors:
            errors.append(
                {
                    "line_number": line_number,
                    "story_id": story_id,
                    "errors": row_errors,
                }
            )
    schema_valid_rate = 0.0 if not records else (len(records) - len(errors)) / len(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(records),
        "expected_records": len(expected_story_ids),
        "completed_records": completed,
        "schema_valid_rate": schema_valid_rate,
        "all_expected_story_ids_present": seen == expected_story_ids,
        "errors": errors,
        "semantic_schema_gate_passed": (
            seen == expected_story_ids
            and completed == len(expected_story_ids)
            and schema_valid_rate >= 0.95
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(payload)
    return records


def validate_existing_labels(output_dir: Path, labels_path: Path) -> dict[str, Any]:
    snapshot = output_dir / "pilot_snapshot.sqlite"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing pilot snapshot: {snapshot}")
    with closing(sqlite3.connect(source_uri(snapshot), uri=True)) as conn:
        expected_story_ids = {
            str(row[0]) for row in conn.execute("SELECT story_id FROM pilot_story_clusters")
        }
    report = validate_completed_labels(read_jsonl(labels_path), expected_story_ids)
    report_path = output_dir / "semantic_label_quality_report.json"
    write_json(report_path, report)
    return {"report": str(report_path.resolve()), **report}


def execute_freeze(
    news_db: Path,
    web_db: Path,
    output_dir: Path,
    information_cutoff: pd.Timestamp,
    sample_size: int,
) -> dict[str, Any]:
    if sample_size < 2 or sample_size > MAX_SAMPLE_SIZE:
        raise ValueError(f"sample_size must be between 2 and {MAX_SAMPLE_SIZE}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen output directory: {output_dir}"
        )

    source_snapshots = [
        load_source_snapshot(news_db, "news", information_cutoff),
        load_source_snapshot(web_db, "web", information_cutoff),
    ]
    all_articles = pd.concat(
        [snapshot.articles for snapshot in source_snapshots], ignore_index=True
    )
    eligible_articles, exact = assign_exact_episodes(all_articles)
    candidate_pool = build_candidate_pool(exact, sample_size)
    candidate_stories = merge_near_duplicate_candidates(candidate_pool)
    stories = select_final_stories(candidate_stories, sample_size)
    if len(stories) != sample_size:
        raise RuntimeError(
            f"Could select only {len(stories)} stories for requested sample {sample_size}"
        )
    selected_story_ids = set(stories["story_id"].astype(str))
    selected_article_keys = {
        str(key)
        for keys in stories.loc[
            stories["story_id"].astype(str).isin(selected_story_ids), "article_keys"
        ]
        for key in list(keys)
    }
    selected_articles = eligible_articles[
        eligible_articles["article_key"].isin(selected_article_keys)
    ].copy()
    article_to_story = {
        str(key): str(row.story_id)
        for row in stories.itertuples(index=False)
        for key in row.article_keys
    }
    selected_articles["story_id"] = selected_articles["article_key"].map(
        article_to_story
    )

    relation_parts: dict[str, list[DataFrame]] = defaultdict(list)
    for snapshot in source_snapshots:
        source_ids = selected_articles.loc[
            selected_articles["source_family"] == snapshot.source_family, "id"
        ].astype(str)
        relations = read_relations(
            snapshot.db_path, snapshot.source_family, source_ids.tolist()
        )
        for key, frame in relations.items():
            if not frame.empty:
                relation_parts[key].append(frame)
    relation_output = {
        key: pd.concat(frames, ignore_index=True) if frames else DataFrame()
        for key, frames in relation_parts.items()
    }
    empty_relation_columns = {
        "tags": (
            "article_id",
            "namespace",
            "value",
            "matched_by",
            "confidence",
            "source_family",
            "article_key",
        ),
        "categories": (
            "article_id",
            "category",
            "confidence",
            "source_family",
            "article_key",
        ),
        "assets": (
            "article_id",
            "asset",
            "confidence",
            "source_family",
            "article_key",
        ),
    }
    for key, columns in empty_relation_columns.items():
        relation_output.setdefault(key, DataFrame(columns=columns))

    labels = build_label_template(stories, relation_output)
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "information_cutoff": iso_utc(information_cutoff),
        "freshness_rule": {
            "safe_available_at": (
                "collected_at because it proves the collector had the item by that "
                "time"
            ),
            "missing_published_at": "eligible using collected_at and explicitly flagged",
            "minimum_publication_lag_hours": FRESH_PUBLICATION_MIN_LAG_HOURS,
            "maximum_publication_lag_hours": FRESH_PUBLICATION_MAX_LAG_HOURS,
            "reason": (
                "exclude stale startup/backfill material and implausibly future-dated "
                "feed items"
            ),
        },
        "blocks": [
            {"name": name, "start": iso_utc(start), "end_exclusive": iso_utc(end)}
            for name, start, end in BLOCKS
        ],
        "sources": {
            snapshot.source_family: {
                "database": snapshot.metadata,
                "coverage": coverage_summary(snapshot.articles),
                "source_health": snapshot.health_rows,
            }
            for snapshot in source_snapshots
        },
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": (
            "Can story-level structure add direction-neutral information about later "
            "BTC/ETH activity beyond simple article counts?"
        ),
        "outcome_blind": True,
        "price_or_market_outcomes_read": False,
        "sample_size": sample_size,
        "candidate_pool_exact_episodes": len(candidate_pool),
        "candidate_story_clusters": len(candidate_stories),
        "selected_articles": len(selected_articles),
        "information_cutoff": iso_utc(information_cutoff),
        "cluster_rules": {
            "exact_title_episode_gap_hours": EXACT_EPISODE_GAP_HOURS,
            "near_title_gap_hours": NEAR_DUPLICATE_GAP_HOURS,
            "near_title_jaccard": NEAR_DUPLICATE_JACCARD,
            "near_title_containment": NEAR_DUPLICATE_CONTAINMENT,
            "minimum_near_duplicate_tokens": MIN_NEAR_DUPLICATE_TOKENS,
            "limitation": (
                "Near-title matching is a conservative pilot rule, not final "
                "embedding-based story resolution."
            ),
        },
        "selection_rule": (
            "deterministic round-robin across source family, source group, and "
            "pre-outcome priority band"
        ),
    }
    quality = freeze_quality_report(
        stories, selected_articles, labels, sample_size, information_cutoff
    )
    if not quality["passed"]:
        raise RuntimeError(f"Freeze quality gate failed: {quality['checks']}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent)
    )
    try:
        snapshot_path = temporary_dir / "pilot_snapshot.sqlite"
        coverage_path = temporary_dir / "coverage_report.json"
        labels_path = temporary_dir / "story_label_template.jsonl"
        quality_path = temporary_dir / "freeze_quality_report.json"
        manifest_path = temporary_dir / "sample_manifest.json"
        write_compact_snapshot(
            snapshot_path,
            selected_articles,
            stories,
            relation_output,
            metadata,
        )
        write_json(coverage_path, coverage)
        write_jsonl(labels_path, labels)
        write_json(quality_path, quality)
        manifest = {
            **metadata,
            "artifacts": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in (snapshot_path, coverage_path, labels_path, quality_path)
            },
            "next_gate": (
                "Complete semantic labels and independent bounded review before "
                "reading market outcomes."
            ),
        }
        write_json(manifest_path, manifest)
        temporary_dir.rename(output_dir)
    except Exception:
        for path in temporary_dir.glob("*"):
            path.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise

    return {
        "output_dir": str(output_dir.resolve()),
        "sample_manifest": str((output_dir / "sample_manifest.json").resolve()),
        "coverage_report": str((output_dir / "coverage_report.json").resolve()),
        "quality_report": str(
            (output_dir / "freeze_quality_report.json").resolve()
        ),
        "label_template": str(
            (output_dir / "story_label_template.jsonl").resolve()
        ),
        "snapshot": str((output_dir / "pilot_snapshot.sqlite").resolve()),
        "stories": len(stories),
        "articles": len(selected_articles),
        "freeze_gate_passed": True,
        "semantic_labels_ready": False,
        "market_test_ready": False,
    }


def plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "outcome-blind semantic data pilot freeze",
        "news_db": str(args.news_db.resolve()),
        "web_db": str(args.web_db.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "information_cutoff": iso_utc(args.information_cutoff),
        "sample_size": args.sample_size,
        "maximum_sample_size": MAX_SAMPLE_SIZE,
        "blocks": [
            {"name": name, "start": iso_utc(start), "end_exclusive": iso_utc(end)}
            for name, start, end in BLOCKS
        ],
        "market_outcomes": "not read",
        "collector_changes": "none",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, default=DEFAULT_NEWS_DB)
    parser.add_argument("--web-db", type=Path, default=DEFAULT_WEB_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--information-cutoff",
        type=pd.Timestamp,
        default=DEFAULT_INFORMATION_CUTOFF,
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--execute", action="store_true", help="Write the frozen compact snapshot."
    )
    parser.add_argument(
        "--validate-labels",
        type=Path,
        default=None,
        help="Validate a completed JSONL label file against an existing snapshot.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cutoff = pd.Timestamp(args.information_cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    args.information_cutoff = cutoff
    if args.validate_labels is not None:
        print(
            json.dumps(
                validate_existing_labels(args.output_dir, args.validate_labels),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute:
        print(json.dumps(plan_payload(args), indent=2, sort_keys=True))
        return 0
    result = execute_freeze(
        args.news_db,
        args.web_db,
        args.output_dir,
        cutoff,
        args.sample_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
