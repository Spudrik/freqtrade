"""Freeze the first independent context-source batch before reading market outcomes.

This batch keeps news, web, Google attention, global/financial measurements, and the
already-frozen historical cross-market shocks as separate siblings.  It exports only
causally available source observations, source-defined event anchors, and outcome-blind
control mappings.  Price files, returns, reaction labels, and trading outcomes are not
read here.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_pilot_freeze as story_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
USER_DATA_DIR = REPO_ROOT / "user_data"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "independent_context_sources_20260904a"
)
GLOBAL_CAUSAL_PATH = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "source_preflight"
    / "global_context_source_preflight_20260904a_causal.parquet"
)
CROSS_ASSET_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "cross_asset_relevance_20260904a"
)
CROSS_ASSET_RESULT = CROSS_ASSET_ROOT / "cross_asset_freeze_result.json"
CROSS_ASSET_FREEZE = CROSS_ASSET_ROOT / "cross_asset_freeze.json"
CROSS_ASSET_CATALOG = CROSS_ASSET_ROOT / "cross_asset_event_catalog.csv"

SOURCE_HISTORY_START = pd.Timestamp("2026-05-08T00:00:00Z")
SOURCE_TEST_START = pd.Timestamp("2026-06-01T00:00:00Z")
SOURCE_TEST_END = pd.Timestamp("2026-08-31T00:00:00Z")
SOURCE_SNAPSHOT_CUTOFF = pd.Timestamp("2026-09-01T00:00:00Z")
PARTITIONS: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...] = (
    (
        "development_2026_06_01_to_07_15",
        pd.Timestamp("2026-06-01T00:00:00Z"),
        pd.Timestamp("2026-07-16T00:00:00Z"),
    ),
    (
        "validation_2026_07_16_to_08_30",
        pd.Timestamp("2026-07-16T00:00:00Z"),
        SOURCE_TEST_END,
    ),
)

HORIZONS_HOURS = (1, 4, 8, 24, 72)
ACTIVITY_MULTIPLE = 1.20
MINIMUM_EVENT_COUNT = 10
MINIMUM_CONTROL_COUNT = 6
CONTROL_COUNT = 12
MINIMUM_ACTIVITY_RATE = 0.55
MINIMUM_DIRECTION_RATE = 0.55
MINIMUM_DIRECTION_EDGE = 0.03

# Every row is an independent trader question.  Closely related raw metrics are not
# opened as dozens of implicit multiple tests.
GLOBAL_FAMILIES: dict[str, dict[str, Any]] = {
    "google_crypto_attention_change": {
        "label": "Change in broad crypto Google attention",
        "source_id": "google_trends_crypto_attention",
        "metric_key": "google_trends_crypto_attention_composite",
        "value_column": "value",
        "transform": "difference",
        "sample_mode": "native",
        "selection_quantile": 0.80,
        "minimum_history": 20,
        "rolling_history": 90,
        "cooldown_hours": 24,
        "horizons_hours": (1, 4, 8, 24),
        "direction_relation": "abstain_unsigned_attention",
        "target_kind": "market",
        "source_role": "independent_attention",
    },
    "fear_greed_change": {
        "label": "Change in crypto fear and greed",
        "source_id": "alternative_fear_greed",
        "metric_key": "fear_greed_index",
        "value_column": "value",
        "transform": "difference",
        "sample_mode": "first_source_timestamp",
        "selection_quantile": 0.50,
        "minimum_history": 14,
        "rolling_history": 60,
        "cooldown_hours": 24,
        "horizons_hours": (4, 8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "market_derived_sentiment",
    },
    "btc_etf_net_flow": {
        "label": "US spot Bitcoin ETF net flow",
        "source_id": "farside_btc_spot_etf_flows",
        "metric_key": "btc_spot_etf_net_flow_musd",
        "value_column": "value",
        "transform": "level",
        "sample_mode": "first_source_timestamp",
        "selection_quantile": 0.50,
        "minimum_history": 14,
        "rolling_history": 60,
        "cooldown_hours": 24,
        "horizons_hours": (8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "institutional_flow",
    },
    "stablecoin_supply_change": {
        "label": "One-day stablecoin supply change",
        "source_id": "defillama_stablecoins",
        "metric_key": "stablecoin_supply_change_1d",
        "value_column": "value",
        "transform": "level",
        "sample_mode": "first_daily_bucket",
        "selection_quantile": 0.50,
        "minimum_history": 14,
        "rolling_history": 60,
        "cooldown_hours": 24,
        "horizons_hours": (8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "slow_liquidity_proxy",
    },
    "fred_equity_risk_change": {
        "label": "US equity and volatility risk basket",
        "source_id": "fred_us_equity_daily",
        "metric_key": "fred_us_equity_daily_change",
        "value_column": "score",
        "transform": "centered_score",
        "sample_mode": "first_source_timestamp",
        "selection_quantile": 0.50,
        "minimum_history": 14,
        "rolling_history": 60,
        "cooldown_hours": 24,
        "horizons_hours": (8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "external_risk_on_off",
    },
    "fred_rates_risk_change": {
        "label": "US rates and dollar risk basket",
        "source_id": "fred_rates_risk_daily",
        "metric_key": "fred_rates_risk_daily_change",
        "value_column": "score",
        "transform": "centered_score",
        "sample_mode": "first_source_timestamp",
        "selection_quantile": 0.50,
        "minimum_history": 14,
        "rolling_history": 60,
        "cooldown_hours": 24,
        "horizons_hours": (8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "external_risk_on_off",
    },
    "global_crypto_market_change": {
        "label": "Broad crypto market-cap change",
        "source_id": "coingecko_global",
        "metric_key": "global_market_cap_change_24h",
        "value_column": "value",
        "transform": "level",
        "sample_mode": "first_6h_bucket",
        "selection_quantile": 0.80,
        "minimum_history": 20,
        "rolling_history": 120,
        "cooldown_hours": 24,
        "horizons_hours": (1, 4, 8, 24),
        "direction_relation": "same",
        "target_kind": "market",
        "source_role": "market_derived_state",
    },
    "btc_dominance_change": {
        "label": "Change in Bitcoin market dominance",
        "source_id": "coingecko_global",
        "metric_key": "btc_dominance_pct",
        "value_column": "value",
        "transform": "difference",
        "sample_mode": "first_6h_bucket",
        "selection_quantile": 0.80,
        "minimum_history": 20,
        "rolling_history": 120,
        "cooldown_hours": 24,
        "horizons_hours": (1, 4, 8, 24),
        "direction_relation": "same",
        "target_kind": "btc_relative_to_groups",
        "source_role": "market_derived_relative_state",
    },
}

NEWS_WEB_FAMILIES: dict[str, dict[str, Any]] = {
    "live_news_activity_spike": {
        "label": "Live news story-activity spike",
        "source_family": "news",
        "db_path": story_freeze.DEFAULT_NEWS_DB,
        "rolling_history": 168,
        "minimum_history": 48,
        "selection_quantile": 0.80,
        "cooldown_hours": 24,
        "horizons_hours": (1, 4, 8, 24),
    },
    "live_web_activity_spike": {
        "label": "Live web and announcement activity spike",
        "source_family": "web",
        "db_path": story_freeze.DEFAULT_WEB_DB,
        "rolling_history": 28,
        "minimum_history": 12,
        "selection_quantile": 0.80,
        "cooldown_hours": 24,
        "horizons_hours": (1, 4, 8, 24),
    },
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def partition_for(value: Any) -> str | None:
    stamp = pd.Timestamp(value)
    for name, start, end in PARTITIONS:
        if start <= stamp < end:
            return name
    return None


def sign(value: Any) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or float(number) == 0.0:
        return 0
    return 1 if float(number) > 0.0 else -1


def first_per_bucket(frame: DataFrame, hours: int) -> DataFrame:
    out = frame.copy()
    out["_sample_bucket"] = out["available_at"].dt.floor(f"{hours}h")
    return (
        out.sort_values(["available_at", "extract_sequence"], kind="stable")
        .drop_duplicates("_sample_bucket", keep="first")
        .drop(columns="_sample_bucket")
    )


def sample_global_family(source: DataFrame, family: str, spec: Mapping[str, Any]) -> DataFrame:
    rows = source.loc[
        source["source_id"].eq(spec["source_id"])
        & source["metric_key"].eq(spec["metric_key"])
        & source["configured_enabled"].fillna(False)
    ].copy()
    if rows.empty:
        raise ValueError(f"No enabled causal observations for {family}.")
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["available_at"])
    rows = rows.loc[
        rows["available_at"].ge(SOURCE_HISTORY_START)
        & rows["available_at"].lt(SOURCE_SNAPSHOT_CUTOFF)
    ].sort_values(["available_at", "extract_sequence"], kind="stable")

    mode = str(spec["sample_mode"])
    if mode == "first_source_timestamp":
        usable_source_ts = rows["source_ts"].fillna("").astype(str).str.strip().ne("")
        rows = rows.loc[usable_source_ts].drop_duplicates("source_ts", keep="first")
    elif mode == "first_daily_bucket":
        rows = first_per_bucket(rows, 24)
    elif mode == "first_6h_bucket":
        rows = first_per_bucket(rows, 6)
    elif mode != "native":
        raise ValueError(f"Unknown sample mode for {family}: {mode}")

    values = pd.to_numeric(rows[str(spec["value_column"])], errors="coerce")
    transform = str(spec["transform"])
    if transform == "difference":
        rows["source_signal"] = values.diff()
    elif transform == "centered_score":
        rows["source_signal"] = values - 50.0
    elif transform == "level":
        rows["source_signal"] = values
    else:
        raise ValueError(f"Unknown source transform for {family}: {transform}")
    rows["source_value"] = values
    rows["anchor_utc"] = rows["available_at"].dt.ceil("h")
    rows["whole_event_partition"] = rows["anchor_utc"].map(partition_for)

    absolute_signal = rows["source_signal"].abs()
    rows["event_threshold"] = (
        absolute_signal.shift(1)
        .rolling(
            int(spec["rolling_history"]),
            min_periods=int(spec["minimum_history"]),
        )
        .quantile(float(spec["selection_quantile"]))
    )
    rows["candidate_event"] = (
        rows["whole_event_partition"].notna()
        & rows["source_signal"].notna()
        & rows["event_threshold"].notna()
        & absolute_signal.gt(rows["event_threshold"])
    )
    rows["is_event"] = apply_cooldown(
        rows["anchor_utc"],
        rows["candidate_event"],
        int(spec["cooldown_hours"]),
    )
    rows["family"] = family
    rows["family_label"] = spec["label"]
    rows["direction_relation"] = spec["direction_relation"]
    rows["target_kind"] = spec["target_kind"]
    rows["source_role"] = spec["source_role"]
    rows["horizons_hours"] = ";".join(
        str(value) for value in spec["horizons_hours"]
    )
    rows["source_direction"] = rows["source_signal"].map(sign)
    rows["predicted_direction"] = rows["source_direction"].where(
        rows["direction_relation"].ne("abstain_unsigned_attention"), 0
    )
    return rows[
        [
            "family",
            "family_label",
            "source_id",
            "metric_key",
            "source_role",
            "available_at",
            "anchor_utc",
            "whole_event_partition",
            "source_value",
            "source_signal",
            "event_threshold",
            "source_direction",
            "predicted_direction",
            "direction_relation",
            "target_kind",
            "horizons_hours",
            "candidate_event",
            "is_event",
            "source_ts",
            "change_kind",
        ]
    ].reset_index(drop=True)


def apply_cooldown(anchors: pd.Series, candidates: pd.Series, hours: int) -> pd.Series:
    selected = pd.Series(False, index=anchors.index, dtype=bool)
    last_anchor: pd.Timestamp | None = None
    for index in anchors.loc[candidates.fillna(False)].sort_values(kind="stable").index:
        anchor = pd.Timestamp(anchors.loc[index])
        if last_anchor is not None and anchor - last_anchor < pd.Timedelta(hours=hours):
            continue
        selected.loc[index] = True
        last_anchor = anchor
    return selected


def load_fetch_hours(db_path: Path, cutoff: pd.Timestamp) -> DataFrame:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30.0) as conn:
        frame = pd.read_sql_query(
            """
            SELECT source_id, started_at, finished_at, status, fetched_count,
                   inserted_count, duplicate_count
            FROM source_fetch_log
            WHERE started_at < ?
            ORDER BY started_at, source_id, id
            """,
            conn,
            params=(cutoff.isoformat(),),
        )
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["started_at"])
    frame["anchor_utc"] = frame["started_at"].dt.floor("h") + pd.Timedelta(hours=1)
    frame["success"] = frame["status"].eq("success")
    for column in ("fetched_count", "inserted_count", "duplicate_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return (
        frame.groupby("anchor_utc", as_index=False)
        .agg(
            fetch_attempts=("source_id", "size"),
            attempted_sources=("source_id", "nunique"),
            successful_sources=("source_id", lambda values: 0),
            successful_fetches=("success", "sum"),
            fetched_rows=("fetched_count", "sum"),
            inserted_rows=("inserted_count", "sum"),
            duplicate_rows=("duplicate_count", "sum"),
        )
        .merge(
            frame.loc[frame["success"]]
            .groupby("anchor_utc")["source_id"]
            .nunique()
            .rename("successful_sources_actual"),
            left_on="anchor_utc",
            right_index=True,
            how="left",
        )
        .drop(columns="successful_sources")
        .rename(columns={"successful_sources_actual": "successful_sources"})
        .fillna({"successful_sources": 0})
    )


def episode_hours(eligible: DataFrame, exact: DataFrame) -> DataFrame:
    episodes = exact.copy()
    episodes["anchor_utc"] = episodes["first_seen_at"].dt.floor("h") + pd.Timedelta(hours=1)
    episodes["high_priority"] = pd.to_numeric(
        episodes["representative_priority"], errors="coerce"
    ).ge(70.0)
    episodes["multi_source"] = episodes["source_ids"].map(len).ge(2)
    hourly = (
        episodes.groupby("anchor_utc", as_index=False)
        .agg(
            episode_count=("exact_episode_id", "nunique"),
            article_count=("article_count", "sum"),
            high_priority_episode_count=("high_priority", "sum"),
            multi_source_episode_count=("multi_source", "sum"),
        )
    )
    articles = eligible.copy()
    articles["anchor_utc"] = articles["collected_ts"].dt.floor("h") + pd.Timedelta(hours=1)
    diversity = (
        articles.groupby("anchor_utc", as_index=False)
        .agg(
            article_source_count=("source_id", "nunique"),
            article_source_group_count=("source_group", "nunique"),
        )
    )
    return hourly.merge(diversity, on="anchor_utc", how="outer")


def freeze_news_web_family(
    family: str, spec: Mapping[str, Any]
) -> tuple[DataFrame, dict[str, Any]]:
    snapshot = story_freeze.load_source_snapshot(
        Path(spec["db_path"]),
        str(spec["source_family"]),
        SOURCE_SNAPSHOT_CUTOFF,
    )
    eligible, exact = story_freeze.assign_exact_episodes(snapshot.articles)
    fetch = load_fetch_hours(Path(spec["db_path"]), SOURCE_SNAPSHOT_CUTOFF)
    episodes = episode_hours(eligible, exact)
    rows = fetch.merge(episodes, on="anchor_utc", how="left")
    count_columns = (
        "episode_count",
        "article_count",
        "high_priority_episode_count",
        "multi_source_episode_count",
        "article_source_count",
        "article_source_group_count",
    )
    for column in count_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0).astype(int)
    rows["fetch_success_ratio"] = rows["successful_fetches"] / rows[
        "fetch_attempts"
    ].replace(0, np.nan)
    historical_source_median = (
        rows["successful_sources"]
        .shift(1)
        .rolling(int(spec["rolling_history"]), min_periods=4)
        .median()
    )
    rows["source_coverage_floor"] = (historical_source_median * 0.50).clip(lower=1)
    rows["source_available"] = (
        rows["fetch_success_ratio"].ge(0.75)
        & rows["successful_sources"].ge(rows["source_coverage_floor"])
    )
    rows["whole_event_partition"] = rows["anchor_utc"].map(partition_for)
    rows["source_signal"] = rows["episode_count"].astype(float)
    eligible_signal = rows["source_signal"].where(rows["source_available"])
    rows["event_threshold"] = (
        eligible_signal.shift(1)
        .rolling(
            int(spec["rolling_history"]),
            min_periods=int(spec["minimum_history"]),
        )
        .quantile(float(spec["selection_quantile"]))
    )
    rows["candidate_event"] = (
        rows["source_available"]
        & rows["whole_event_partition"].notna()
        & rows["event_threshold"].notna()
        & rows["source_signal"].gt(rows["event_threshold"])
        & rows["source_signal"].gt(0)
    )
    rows["is_event"] = apply_cooldown(
        rows["anchor_utc"], rows["candidate_event"], int(spec["cooldown_hours"])
    )
    rows["family"] = family
    rows["family_label"] = spec["label"]
    rows["source_id"] = spec["source_family"]
    rows["metric_key"] = "new_exact_title_story_episodes"
    rows["source_role"] = "unsigned_story_activity"
    rows["source_value"] = rows["episode_count"].astype(float)
    rows["source_direction"] = 0
    rows["predicted_direction"] = 0
    rows["direction_relation"] = "abstain_missing_story_semantics"
    rows["target_kind"] = "market"
    rows["horizons_hours"] = ";".join(
        str(value) for value in spec["horizons_hours"]
    )
    rows["available_at"] = rows["anchor_utc"]
    rows["source_ts"] = ""
    rows["change_kind"] = "hourly_fetch_snapshot"
    metadata = {
        **snapshot.metadata,
        "eligible_articles": len(eligible),
        "exact_title_episodes": len(exact),
        "fetch_hours": len(fetch),
        "available_fetch_hours": int(rows["source_available"].sum()),
        "direction_semantics_available": False,
    }
    return rows.reset_index(drop=True), metadata


def event_catalog(observations: DataFrame) -> DataFrame:
    events = observations.loc[observations["is_event"]].copy()
    events = events.sort_values(["anchor_utc", "family"], kind="stable").reset_index(
        drop=True
    )
    events["event_id"] = [
        f"{family}_{pd.Timestamp(anchor).strftime('%Y%m%d_%H%M')}_{index:03d}"
        for index, (family, anchor) in enumerate(
            events[["family", "anchor_utc"]].itertuples(index=False, name=None), start=1
        )
    ]
    if events["event_id"].duplicated().any():
        raise ValueError("Independent source event identifiers are not unique.")
    return events


def far_from_events(
    anchors: pd.Series, event_anchors: Sequence[pd.Timestamp], exclusion_hours: int
) -> pd.Series:
    if not event_anchors:
        return pd.Series(True, index=anchors.index, dtype=bool)
    keep = pd.Series(True, index=anchors.index, dtype=bool)
    exclusion = pd.Timedelta(hours=exclusion_hours)
    for event_anchor in event_anchors:
        keep &= (anchors - event_anchor).abs().gt(exclusion)
    return keep


def control_catalog(observations: DataFrame, events: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for family, family_events in events.groupby("family", sort=False):
        if family not in NEWS_WEB_FAMILIES:
            event_anchors = [pd.Timestamp(value) for value in family_events["anchor_utc"]]
            for event in family_events.itertuples(index=False):
                rank = 0
                for weeks in range(1, 27):
                    control_anchor = pd.Timestamp(event.anchor_utc) - pd.Timedelta(
                        weeks=weeks
                    )
                    if any(
                        abs(control_anchor - blocked) <= pd.Timedelta(hours=72)
                        for blocked in event_anchors
                    ):
                        continue
                    rank += 1
                    records.append(
                        {
                            "event_id": event.event_id,
                            "family": family,
                            "event_anchor_utc": event.anchor_utc,
                            "control_anchor_utc": control_anchor,
                            "control_rank": rank,
                            "event_source_direction": int(event.source_direction),
                            "control_source_direction": 0,
                            "matching_rule": (
                                "same UTC weekday and hour in an earlier week; outside "
                                "the 72h window of every same-family source event"
                            ),
                        }
                    )
                    if rank >= CONTROL_COUNT:
                        break
            continue

        source = observations.loc[observations["family"].eq(family)].copy()
        event_anchors = [pd.Timestamp(value) for value in family_events["anchor_utc"]]
        exclusion_hours = 72 if "news" not in family and "web" not in family else 24
        source["far_from_event"] = far_from_events(
            source["anchor_utc"], event_anchors, exclusion_hours
        )
        for event in family_events.itertuples(index=False):
            candidates = source.loc[
                source["whole_event_partition"].notna()
                & ~source["is_event"]
                & source["far_from_event"]
                & source["anchor_utc"].lt(event.anchor_utc)
            ].copy()
            if "source_available" in candidates:
                availability = candidates["source_available"].fillna(True).astype(bool)
                candidates = candidates.loc[availability]
            if int(event.predicted_direction) != 0:
                same_sign = candidates["source_direction"].eq(int(event.source_direction))
                if same_sign.sum() >= MINIMUM_CONTROL_COUNT:
                    candidates = candidates.loc[same_sign]
            candidates = candidates.sort_values("anchor_utc", ascending=False).head(
                CONTROL_COUNT
            )
            for rank, control in enumerate(candidates.itertuples(index=False), start=1):
                records.append(
                    {
                        "event_id": event.event_id,
                        "family": family,
                        "event_anchor_utc": event.anchor_utc,
                        "control_anchor_utc": control.anchor_utc,
                        "control_rank": rank,
                        "event_source_direction": int(event.source_direction),
                        "control_source_direction": int(control.source_direction),
                        "matching_rule": (
                            "prior ordinary source observation; same source-sign when at "
                            "least six were available; outside event exclusion window"
                        ),
                    }
                )
    return DataFrame.from_records(records)


def verify_cross_asset_freeze() -> dict[str, Any]:
    result = json.loads(CROSS_ASSET_RESULT.read_text(encoding="utf-8"))
    freeze = json.loads(CROSS_ASSET_FREEZE.read_text(encoding="utf-8"))
    if result.get("status") != "completed_cross_asset_relevance_freeze":
        raise ValueError("Historical cross-asset freeze is not terminal.")
    if freeze.get("outcomes_read") or result.get("outcomes_read"):
        raise ValueError("Historical cross-asset freeze read outcomes unexpectedly.")
    for name, path in (("freeze", CROSS_ASSET_FREEZE), ("catalog", CROSS_ASSET_CATALOG)):
        expected = result["artifacts"][name]["sha256"]
        if expected != g0.sha256_file(path):
            raise ValueError(f"Historical cross-asset artifact changed: {path}")
    catalog = pd.read_csv(CROSS_ASSET_CATALOG)
    return {
        "freeze": artifact(CROSS_ASSET_FREEZE),
        "catalog": artifact(CROSS_ASSET_CATALOG),
        "events": len(catalog),
        "families": int(catalog["event_family"].nunique()),
        "direction_contract": {
            "technology_equities": "same",
            "market_fear": "opposite",
            "broad_us_dollar": "opposite",
            "financial_conditions": "opposite",
            "crude_oil": "abstain_no_predeclared_relation",
            "ten_year_yield": "abstain_no_predeclared_relation",
            "yield_curve": "abstain_no_predeclared_relation",
        },
    }


def render_report(freeze: Mapping[str, Any]) -> str:
    rows = [
        "# Independent Context-Source Batch Freeze",
        "",
        f"- Status: `{freeze['status']}`",
        "- Market outcomes read: **No**",
        f"- Independent live source families: `{freeze['live_source_family_count']}`",
        f"- Frozen live source events: `{freeze['live_source_event_count']}`",
        "- Historical wider-market families referenced unchanged: `7`",
        "",
        "## Questions",
        "",
        "1. Do source-defined shocks precede unusual market activity?",
        "2. Where a source has defensible sign, does it add short direction beyond "
        "recent price trend?",
        "3. Do unsigned news, web, or Google-attention counts remain activity-only "
        "until story meaning exists?",
        "",
        "All source families finish before any source combination or level interaction is opened.",
        "",
    ]
    return "\n".join(rows)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "independent_context_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_independent_context_source_freeze":
            raise ValueError("Existing independent-source freeze is not terminal.")
        return result
    if not GLOBAL_CAUSAL_PATH.is_file():
        raise FileNotFoundError(GLOBAL_CAUSAL_PATH)

    global_source = pd.read_parquet(GLOBAL_CAUSAL_PATH)
    global_frames = [
        sample_global_family(global_source, family, spec)
        for family, spec in GLOBAL_FAMILIES.items()
    ]
    source_metadata: dict[str, Any] = {
        "global_causal_extract": artifact(GLOBAL_CAUSAL_PATH)
    }
    live_frames = list(global_frames)
    for family, spec in NEWS_WEB_FAMILIES.items():
        rows, metadata = freeze_news_web_family(family, spec)
        live_frames.append(rows)
        source_metadata[family] = metadata

    observations = pd.concat(live_frames, ignore_index=True, sort=False)
    observations["anchor_utc"] = pd.to_datetime(observations["anchor_utc"], utc=True)
    observations["available_at"] = pd.to_datetime(observations["available_at"], utc=True)
    if (observations["available_at"] > observations["anchor_utc"]).any():
        raise ValueError("A source observation is available after its market anchor.")
    events = event_catalog(observations)
    controls = control_catalog(observations, events)
    control_counts = (
        controls.groupby("event_id").size()
        if not controls.empty
        else pd.Series(dtype=int)
    )
    events["control_count"] = events["event_id"].map(control_counts).fillna(0).astype(int)
    events["coverage_ready"] = events["control_count"].ge(MINIMUM_CONTROL_COUNT)
    cross_asset = verify_cross_asset_freeze()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    observations_path = OUTPUT_ROOT / "source_observations.parquet"
    events_path = OUTPUT_ROOT / "source_event_catalog.csv"
    controls_path = OUTPUT_ROOT / "source_control_catalog.csv"
    freeze_path = OUTPUT_ROOT / "independent_context_freeze.json"
    report_path = OUTPUT_ROOT / "independent_context_freeze_report.md"
    g0.atomic_write_parquet(observations, observations_path)
    g0.atomic_write_csv(events, events_path)
    g0.atomic_write_csv(controls, controls_path)
    freeze = {
        "schema_version": 1,
        "status": "frozen_independent_context_sources_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "live_source_family_count": int(events["family"].nunique()),
        "live_source_event_count": len(events),
        "coverage_ready_events": int(events["coverage_ready"].sum()),
        "plain_question": (
            "Which independently observed news, web, attention, flow, liquidity, "
            "sentiment, and wider-market changes identify unusual crypto activity, and "
            "which defensibly signed sources add direction beyond recent price trend?"
        ),
        "source_test_window": {
            "history_start": SOURCE_HISTORY_START.isoformat(),
            "test_start": SOURCE_TEST_START.isoformat(),
            "test_end_exclusive": SOURCE_TEST_END.isoformat(),
            "snapshot_cutoff_exclusive": SOURCE_SNAPSHOT_CUTOFF.isoformat(),
            "partitions": [
                {"name": name, "start": start.isoformat(), "end": end.isoformat()}
                for name, start, end in PARTITIONS
            ],
        },
        "horizons_hours": list(HORIZONS_HOURS),
        "live_source_definitions": json.loads(
            json.dumps(GLOBAL_FAMILIES | NEWS_WEB_FAMILIES, default=str)
        ),
        "selection_rule": (
            "Each family uses only earlier source observations to define an absolute "
            "source-change/activity threshold, then applies a fixed 24h or 72h cooldown."
        ),
        "control_rule": (
            "Up to twelve earlier ordinary observations from the same source family, "
            "outside every event exclusion window; sign-matched where at least six exist."
        ),
        "market_scopes": [
            "BTC",
            "ETH",
            "frozen established-alt group",
            "frozen top-ten traded meme group",
            "BTC relative to established alts and memes for dominance only",
        ],
        "screening_rule": {
            "minimum_whole_events_per_partition": MINIMUM_EVENT_COUNT,
            "minimum_controls_per_event": MINIMUM_CONTROL_COUNT,
            "abnormal_activity_multiple": ACTIVITY_MULTIPLE,
            "minimum_activity_success_rate": MINIMUM_ACTIVITY_RATE,
            "minimum_direction_success_rate": MINIMUM_DIRECTION_RATE,
            "minimum_direction_edge_over_each_simple_comparator": MINIMUM_DIRECTION_EDGE,
            "comparators": [
                "ordinary source observations",
                "recent 24h market direction",
                "development-majority direction",
                "time-rotated source direction",
            ],
        },
        "direction_rules": [
            "News, web, and Google-attention activity issue no signed call.",
            "Market-derived sentiment/state must beat the same recent-price baseline.",
            "Bitcoin-dominance change predicts relative BTC-versus-group movement only.",
            "Reaction, conditional direction, and joint success are reported separately.",
        ],
        "limits": [
            "This batch does not label story meaning or infer major events from counts.",
            "Source association is not proof that the source caused the market move.",
            "No profit, entry, exit, position, leverage, or trading target is used.",
            "All independent siblings finish before combinations or levels are added.",
        ],
        "source_metadata": source_metadata,
        "historical_cross_asset_contract": cross_asset,
        "event_counts": (
            events.groupby(["family", "whole_event_partition"], dropna=False)
            .size()
            .rename("events")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_independent_context_source_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "live_source_family_count": int(events["family"].nunique()),
        "live_source_event_count": len(events),
        "coverage_ready_events": int(events["coverage_ready"].sum()),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "observations": artifact(observations_path),
            "events": artifact(events_path),
            "controls": artifact(controls_path),
            "report": artifact(report_path),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "setup_only",
                    "outcomes_read": False,
                    "output_root": str(OUTPUT_ROOT),
                    "live_source_families": list(GLOBAL_FAMILIES | NEWS_WEB_FAMILIES),
                    "historical_cross_asset_reference": str(CROSS_ASSET_CATALOG),
                },
                indent=2,
            )
        )
        return 0
    result = execute(overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
