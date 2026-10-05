"""Freeze the next broad event-source and transmission batch before outcomes."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_esma_sovereign_rating_catalogue as esma,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as us_cpi,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_japan_tankan_catalogue as tankan,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_sec_hyperscaler_8k_catalogue as sec,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_treasury_refunding_activity_direct as treasury_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_uk_ons_cpi_catalogue as uk_cpi,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "layer2_breadth_batch_20260907a"
)
EVENTS_PATH = OUTPUT_ROOT / "source_activity_event_catalogue.csv"
CONTROLS_PATH = OUTPUT_ROOT / "source_activity_control_map.csv"
CPI_LINK_EVENTS_PATH = OUTPUT_ROOT / "cpi_link_event_catalogue.csv"
FREEZE_PATH = OUTPUT_ROOT / "layer2_breadth_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "layer2_breadth_freeze_report.md"
RESULT_PATH = OUTPUT_ROOT / "layer2_breadth_freeze_result.json"

CPI_CATALOGUE_PATH = us_cpi.OUTPUT_ROOT / "cpi_release_catalog.csv"
CPI_RESULT_PATH = us_cpi.OUTPUT_ROOT / "cpi_freeze_result.json"
CPI_FREEZE_PATH = us_cpi.OUTPUT_ROOT / "cpi_freeze.json"
MEME_COHORT_PATH = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/market_reaction_zones/"
    "meme_cohort/meme_top10_selection_20260813.json"
)

PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
ACTIVITY_ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
ESTABLISHED_FOLLOWERS = (
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
)
CONTROL_COUNT = 12
CONTROL_SEARCH_WEEKS = 52
COLLISION_HOURS = 4
MAX_ACTIVITY_HORIZON_MINUTES = 240
ACTIVITY_ROUTES: dict[str, dict[str, Any]] = {
    "uk_ons_cpi": {
        "plain_name": "UK consumer-price release",
        "horizons_minutes": [5, 15, 30, 60],
        "minimum_events_by_partition": {
            "development_2021_2023": 12,
            "internal_validation_2024_2025": 8,
        },
        "timestamp_limit": "exact official ONS release-page minute",
    },
    "japan_tankan": {
        "plain_name": "Japan Tankan business survey",
        "horizons_minutes": [5, 15, 30, 60],
        "minimum_events_by_partition": {
            "development_2021_2023": 8,
            "internal_validation_2024_2025": 6,
        },
        "timestamp_limit": "official always-08:50-JST rule; not server posting time",
    },
    "esma_euro_signed_rating": {
        "plain_name": "European sovereign-rating change publication",
        "horizons_minutes": [15, 60, 240],
        "minimum_events_by_partition": {
            "development_2021_2023": 30,
            "internal_validation_2024_2025": 30,
        },
        "timestamp_limit": "exact ESMA action-validity publication clock",
    },
    "sec_hyperscaler_all": {
        "plain_name": "Any fixed-hyperscaler SEC 8-K acceptance",
        "horizons_minutes": [15, 60, 240],
        "minimum_events_by_partition": {
            "development_2021_2023": 30,
            "internal_validation_2024_2025": 20,
        },
        "timestamp_limit": "EDGAR acceptance, not proven first-public news",
    },
    "sec_hyperscaler_earnings": {
        "plain_name": "Fixed-hyperscaler SEC Item 2.02 acceptance",
        "horizons_minutes": [15, 60, 240],
        "minimum_events_by_partition": {
            "development_2021_2023": 20,
            "internal_validation_2024_2025": 12,
        },
        "timestamp_limit": "EDGAR acceptance, often after an earlier earnings release",
    },
}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def validate_source(
    *,
    result_path: Path,
    freeze_path: Path,
    catalogue_path: Path,
    result_status: str,
    freeze_status: str,
    catalogue_artifact_key: str = "catalogue",
) -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if result.get("status") != result_status or freeze.get("status") != freeze_status:
        raise ValueError(f"Source catalogue is not terminal: {catalogue_path}")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError(f"Source catalogue opened market outcomes: {catalogue_path}")
    if result.get("profit_used") or freeze.get("profit_used"):
        raise ValueError(f"Source catalogue used profit: {catalogue_path}")
    for key, path in (
        (catalogue_artifact_key, catalogue_path),
        ("freeze", freeze_path),
    ):
        recorded = result["artifacts"][key]["sha256"]
        if recorded != g0.sha256_file(path):
            raise ValueError(f"Frozen source artifact changed: {path}")
    return result, freeze, pd.read_csv(catalogue_path)


def fixed_window_clusters(
    frame: DataFrame,
    *,
    anchor_column: str,
    prefix: str,
    window_minutes: int = MAX_ACTIVITY_HORIZON_MINUTES,
) -> DataFrame:
    ordered = frame.copy()
    ordered[anchor_column] = pd.to_datetime(ordered[anchor_column], utc=True)
    ordered = ordered.sort_values(anchor_column, kind="stable").reset_index(drop=True)
    cluster_ids: list[str] = []
    cluster_starts: list[pd.Timestamp] = []
    cluster_number = 0
    cluster_start: pd.Timestamp | None = None
    for anchor in ordered[anchor_column]:
        stamp = pd.Timestamp(anchor)
        if cluster_start is None or stamp >= cluster_start + pd.Timedelta(
            minutes=window_minutes
        ):
            cluster_number += 1
            cluster_start = stamp
        cluster_ids.append(f"{prefix}_{cluster_number:04d}")
        cluster_starts.append(cluster_start)
    ordered["market_window_group_id"] = cluster_ids
    ordered["cluster_anchor_utc"] = cluster_starts
    return ordered


def _base_row(
    *,
    event_id: str,
    source_family: str,
    anchor: pd.Timestamp,
    partition: str,
    timestamp_quality: str,
    member_event_count: int = 1,
    market_window_group_id: str | None = None,
) -> dict[str, Any]:
    decision = pd.Timestamp(anchor).ceil("1min")
    return {
        "event_id": event_id,
        "base_event_id": market_window_group_id or event_id,
        "event_source": source_family,
        "source_family": source_family.split("_all")[0].split("_earnings")[0],
        "anchor_utc": pd.Timestamp(anchor),
        "decision_utc": decision,
        "decision_delay_seconds": int((decision - pd.Timestamp(anchor)).total_seconds()),
        "whole_event_partition": partition,
        "timestamp_quality": timestamp_quality,
        "member_event_count": int(member_event_count),
        "market_window_group_id": market_window_group_id or event_id,
    }


def normalized_source_events() -> tuple[DataFrame, list[dict[str, Any]]]:
    source_artifacts: list[dict[str, Any]] = []
    _, _, uk = validate_source(
        result_path=uk_cpi.RESULT_PATH,
        freeze_path=uk_cpi.FREEZE_PATH,
        catalogue_path=uk_cpi.CATALOGUE_PATH,
        result_status="completed_uk_ons_cpi_catalogue",
        freeze_status="frozen_uk_ons_cpi_catalogue_before_market_outcomes",
    )
    source_artifacts.append(
        {
            "source": "uk_ons_cpi",
            "result": artifact(uk_cpi.RESULT_PATH),
            "freeze": artifact(uk_cpi.FREEZE_PATH),
            "catalogue": artifact(uk_cpi.CATALOGUE_PATH),
        }
    )
    _, _, jp = validate_source(
        result_path=tankan.RESULT_PATH,
        freeze_path=tankan.FREEZE_PATH,
        catalogue_path=tankan.CATALOGUE_PATH,
        result_status="completed_japan_tankan_catalogue",
        freeze_status="frozen_japan_tankan_catalogue_before_market_outcomes",
    )
    source_artifacts.append(
        {
            "source": "japan_tankan",
            "result": artifact(tankan.RESULT_PATH),
            "freeze": artifact(tankan.FREEZE_PATH),
            "catalogue": artifact(tankan.CATALOGUE_PATH),
        }
    )
    _, _, rating = validate_source(
        result_path=esma.RESULT_PATH,
        freeze_path=esma.FREEZE_PATH,
        catalogue_path=esma.CATALOG_PATH,
        result_status="completed_esma_sovereign_rating_catalogue",
        freeze_status="frozen_esma_sovereign_ratings_before_market_outcomes",
    )
    source_artifacts.append(
        {
            "source": "esma_euro_signed_rating",
            "result": artifact(esma.RESULT_PATH),
            "freeze": artifact(esma.FREEZE_PATH),
            "catalogue": artifact(esma.CATALOG_PATH),
        }
    )
    _, _, filings = validate_source(
        result_path=sec.RESULT_PATH,
        freeze_path=sec.FREEZE_PATH,
        catalogue_path=sec.CATALOGUE_PATH,
        result_status="completed_sec_hyperscaler_8k_catalogue",
        freeze_status="frozen_sec_hyperscaler_8k_catalogue_before_market_outcomes",
    )
    source_artifacts.append(
        {
            "source": "sec_hyperscaler",
            "result": artifact(sec.RESULT_PATH),
            "freeze": artifact(sec.FREEZE_PATH),
            "catalogue": artifact(sec.CATALOGUE_PATH),
        }
    )

    rows: list[dict[str, Any]] = []
    for event in uk.loc[uk["whole_event_partition"].isin(PARTITIONS)].itertuples(
        index=False
    ):
        rows.append(
            _base_row(
                event_id=event.event_id,
                source_family="uk_ons_cpi",
                anchor=pd.Timestamp(event.anchor_utc),
                partition=event.whole_event_partition,
                timestamp_quality=event.time_precision,
            )
        )
    for event in jp.loc[jp["whole_event_partition"].isin(PARTITIONS)].itertuples(
        index=False
    ):
        rows.append(
            _base_row(
                event_id=event.event_id,
                source_family="japan_tankan",
                anchor=pd.Timestamp(event.anchor_utc),
                partition=event.whole_event_partition,
                timestamp_quality=event.timestamp_quality,
            )
        )

    rating = rating.loc[
        rating["whole_event_partition"].isin(PARTITIONS)
        & rating["jurisdiction_group"].eq("selected_euro_area")
        & ~rating["source_action_sign"].eq("unsigned_or_lifecycle_action")
    ].copy()
    rating = fixed_window_clusters(
        rating,
        anchor_column="available_at_utc",
        prefix="esma_euro_signed_window",
    )
    for group_id, group in rating.groupby("market_window_group_id", sort=False):
        first = group.iloc[0]
        rows.append(
            _base_row(
                event_id=str(group_id),
                source_family="esma_euro_signed_rating",
                anchor=pd.Timestamp(first["cluster_anchor_utc"]),
                partition=str(first["whole_event_partition"]),
                timestamp_quality="esma_exact_clock_clustered_into_nonoverlapping_4h_windows",
                member_event_count=len(group),
                market_window_group_id=str(group_id),
            )
        )

    filings = filings.loc[filings["whole_event_partition"].isin(PARTITIONS)].copy()
    filings = fixed_window_clusters(
        filings,
        anchor_column="acceptance_datetime_utc",
        prefix="sec_hyperscaler_window",
    )
    for group_id, group in filings.groupby("market_window_group_id", sort=False):
        first = group.iloc[0]
        common = dict(
            anchor=pd.Timestamp(first["cluster_anchor_utc"]),
            partition=str(first["whole_event_partition"]),
            timestamp_quality="sec_acceptance_clustered_into_nonoverlapping_4h_windows",
            member_event_count=len(group),
            market_window_group_id=str(group_id),
        )
        rows.append(
            _base_row(
                event_id=f"{group_id}_all",
                source_family="sec_hyperscaler_all",
                **common,
            )
        )
        if group["earnings_like_candidate"].astype(bool).any():
            rows.append(
                _base_row(
                    event_id=f"{group_id}_earnings",
                    source_family="sec_hyperscaler_earnings",
                    **common,
                )
            )
    output = DataFrame.from_records(rows).sort_values(
        ["decision_utc", "event_source"], kind="stable"
    ).reset_index(drop=True)
    validate_normalized_events(output)
    return output, source_artifacts


def validate_normalized_events(events: DataFrame) -> None:
    if events["event_id"].duplicated().any():
        raise ValueError("Normalized source event identifiers are not unique")
    if set(events["event_source"]) != set(ACTIVITY_ROUTES):
        raise ValueError("Normalized source routes differ from the frozen route set")
    if not events["whole_event_partition"].isin(PARTITIONS).all():
        raise ValueError("Normalized events contain an unfrozen partition")
    delay = pd.to_numeric(events["decision_delay_seconds"], errors="raise")
    if not delay.between(0, 59).all():
        raise ValueError("Decision-minute alignment delay is outside 0-59 seconds")
    if events["decision_utc"].isna().any():
        raise ValueError("Normalized events contain a missing decision minute")


def add_collision_flags(
    events: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> DataFrame:
    output = events.copy()
    bases = output.drop_duplicates("base_event_id")[["base_event_id", "decision_utc"]]
    anchor_by_base = {
        str(row.base_event_id): pd.Timestamp(row.decision_utc)
        for row in bases.itertuples(index=False)
    }
    flags: list[bool] = []
    for event in output.itertuples(index=False):
        others = [
            *external_anchors,
            *(
                anchor
                for base_id, anchor in anchor_by_base.items()
                if base_id != str(event.base_event_id)
            ),
        ]
        distance = treasury_activity.shared.minimum_event_distance_hours(
            others, pd.Timestamp(event.decision_utc)
        )
        flags.append(distance <= COLLISION_HOURS)
    output["other_frozen_event_within_4h"] = flags
    return output


def build_control_map(
    events: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> DataFrame:
    blocked = [
        *external_anchors,
        *events.drop_duplicates("base_event_id")["decision_utc"].tolist(),
    ]
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        rank = 0
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            candidate = pd.Timestamp(event.decision_utc) - pd.Timedelta(weeks=weeks)
            distance = treasury_activity.shared.minimum_event_distance_hours(
                blocked, candidate
            )
            if distance <= COLLISION_HOURS:
                continue
            rank += 1
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_source": event.event_source,
                    "event_decision_utc": event.decision_utc,
                    "control_anchor_utc": candidate,
                    "control_type": "prior_same_weekday_and_utc_clock",
                    "control_rank": rank,
                }
            )
            if rank >= CONTROL_COUNT:
                break
    controls = DataFrame.from_records(rows)
    if controls.duplicated(["event_id", "control_rank"]).any():
        raise ValueError("Frozen source controls are not unique")
    return controls


def load_cpi_link_events() -> tuple[DataFrame, dict[str, Any]]:
    result = json.loads(CPI_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(CPI_FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_cpi_family_freeze":
        raise ValueError("US CPI family freeze is not terminal")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("US CPI source freeze unexpectedly opened outcomes")
    if result["artifacts"]["catalog"]["sha256"] != g0.sha256_file(CPI_CATALOGUE_PATH):
        raise ValueError("Frozen US CPI catalogue changed")
    if result["artifacts"]["freeze"]["sha256"] != g0.sha256_file(CPI_FREEZE_PATH):
        raise ValueError("Frozen US CPI definitions changed")
    events = pd.read_csv(CPI_CATALOGUE_PATH)
    events["anchor_utc"] = pd.to_datetime(events["anchor_utc"], utc=True)
    events["leader_eligible"] = events["whole_event_partition"].isin(PARTITIONS)
    meme_start = pd.Timestamp("2025-07-10T08:00:00Z")
    meme_end = pd.Timestamp("2026-07-14T00:00:00Z")
    events["meme_eligible"] = events["anchor_utc"].ge(meme_start) & events[
        "anchor_utc"
    ].lt(meme_end)
    events["meme_partition"] = pd.NA
    events.loc[
        events["meme_eligible"] & events["anchor_utc"].lt("2026-01-01"),
        "meme_partition",
    ] = "meme_development_2025"
    events.loc[
        events["meme_eligible"] & events["anchor_utc"].ge("2026-01-01"),
        "meme_partition",
    ] = "meme_internal_validation_2026_h1"
    selected = events.loc[events["leader_eligible"] | events["meme_eligible"], [
        "event_id",
        "anchor_utc",
        "whole_event_partition",
        "leader_eligible",
        "meme_eligible",
        "meme_partition",
    ]].copy()
    return selected, {
        "source": "us_cpi",
        "result": artifact(CPI_RESULT_PATH),
        "freeze": artifact(CPI_FREEZE_PATH),
        "catalogue": artifact(CPI_CATALOGUE_PATH),
    }


def load_meme_cohort() -> tuple[list[str], dict[str, Any]]:
    document = json.loads(MEME_COHORT_PATH.read_text(encoding="utf-8"))
    if document.get("status") != "frozen_before_reaction_outcomes":
        raise ValueError("Top-ten meme cohort is not frozen")
    members = [str(row["freqtrade_pair"]) for row in document["members"]]
    if len(members) != 10 or len(set(members)) != 10:
        raise ValueError("Top-ten meme cohort does not contain ten unique pairs")
    return members, artifact(MEME_COHORT_PATH)


def freeze_document(
    events: DataFrame,
    controls: DataFrame,
    cpi_events: DataFrame,
    source_artifacts: Sequence[Mapping[str, Any]],
    external_artifacts: Sequence[Mapping[str, Any]],
    cpi_artifact: Mapping[str, Any],
    meme_members: Sequence[str],
    meme_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    event_counts = (
        events.groupby(["event_source", "whole_event_partition"], sort=True)
        .size()
        .rename("events")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "schema_version": 1,
        "status": "frozen_layer2_breadth_batch_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "objective": (
            "Complete several independent event-source activity tests and the already-"
            "queued CPI leader and meme links before following any result-led branch."
        ),
        "all_siblings_frozen_together": True,
        "activity_routes": ACTIVITY_ROUTES,
        "activity_test": {
            "assets": list(ACTIVITY_ASSETS),
            "response": (
                "median of absolute return, full high-low range, and traded-volume "
                "ratios against clean controls"
            ),
            "controls": (
                "up to 12 prior same-weekday and UTC-clock weeks, excluding any frozen "
                "major event within four hours"
            ),
            "variants": ["all_events", "exclude_other_frozen_event_within_4h"],
            "minimum_median_activity_score": 1.2,
            "minimum_above_control_rate": 0.55,
            "retention_rule": (
                "The same source, asset, and horizon must pass development and validation "
                "with enough whole events in both variants. Any survivor remains an "
                "exploratory activity-only lead pending whole-family shuffle validation."
            ),
        },
        "cpi_immediate_leader_transmission": {
            "events": int(cpi_events["leader_eligible"].sum()),
            "leader_candidates": ["btc_first_move", "eth_first_move", "btc_eth_agreement"],
            "leader_windows_minutes": [5, 15],
            "follower_end_minutes": [15, 30, 60],
            "followers": list(ESTABLISHED_FOLLOWERS),
            "rule": (
                "Use only movement after the frozen leader window. Issue a directional "
                "call only when the first move exceeds its clean matched-control median; "
                "compare with matched controls and each follower's own first move."
            ),
            "minimum_issued_whole_events": 10,
            "minimum_overall_direction_rate": 0.55,
            "minimum_each_partition_rate": 0.50,
            "minimum_control_uplift": 0.05,
        },
        "cpi_meme_response": {
            "events": int(cpi_events["meme_eligible"].sum()),
            "cohort": list(meme_members),
            "event_window": "2025-07-10T08:00:00Z to 2026-07-14T00:00:00Z exclusive",
            "leader_windows_minutes": [5, 15],
            "response_end_minutes": [15, 30, 60],
            "rule": (
                "Test memes only when the same CPI event has a measurable BTC first move. "
                "Report simultaneous reaction separately from later direction and remove "
                "each coin's rolling ordinary BTC sensitivity and volatility."
            ),
            "minimum_complete_whole_events": 10,
            "minimum_reaction_or_direction_rate": 0.55,
            "selection_limit": (
                "This is recent exploratory evidence because the 2026 cohort ranking was "
                "not historically knowable; no old nonexistent coin is backfilled."
            ),
        },
        "breadth_boundary": [
            "UK CPI, Tankan, ESMA ratings, and SEC filings are different event families.",
            "The all-filing and earnings SEC routes are related views, not independent proof.",
            "No event direction is inferred without expectation or a causal post-event leader.",
            "No profit, entry, exit, or trading promotion is part of this batch.",
            "Complete and jointly review every sibling before any result-led refinement.",
        ],
        "event_counts": event_counts,
        "frozen_control_rows": len(controls),
        "source_artifacts": list(source_artifacts),
        "external_collision_sources": list(external_artifacts),
        "cpi_source": dict(cpi_artifact),
        "meme_cohort_source": dict(meme_artifact),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    lines = [
        "# Layer 2 Breadth Batch Freeze",
        "",
        "No market outcome or profit was read while freezing this batch.",
        "",
        "## Independent source activity siblings",
        "",
    ]
    for source, config in freeze["activity_routes"].items():
        lines.append(
            f"- `{source}`: {config['plain_name']}; horizons "
            f"`{config['horizons_minutes']}` minutes."
        )
    lines.extend(
        [
            "",
            "## Separately frozen causal links",
            "",
            "- CPI first BTC/ETH move -> later BNB/ADA/TRX movement.",
            "- CPI plus measurable BTC response -> top-ten meme reaction and later movement.",
            "",
            "Every source test is activity-first. Direction is confined to the post-release "
            "leader/follower windows, with abstention when the first move is not measurable.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_layer2_breadth_outcome_blind_freeze":
        raise ValueError("Existing Layer 2 breadth freeze result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing Layer 2 breadth artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    events, source_artifacts = normalized_source_events()
    external_anchors, external_artifacts = treasury_activity.load_external_anchors()
    events = add_collision_flags(events, external_anchors)
    controls = build_control_map(events, external_anchors)
    cpi_events, cpi_artifact = load_cpi_link_events()
    meme_members, meme_artifact = load_meme_cohort()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(events, EVENTS_PATH)
    g0.atomic_write_csv(controls, CONTROLS_PATH)
    g0.atomic_write_csv(cpi_events, CPI_LINK_EVENTS_PATH)
    freeze = freeze_document(
        events,
        controls,
        cpi_events,
        source_artifacts,
        external_artifacts,
        cpi_artifact,
        meme_members,
        meme_artifact,
    )
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_layer2_breadth_outcome_blind_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "activity_event_rows": len(events),
        "control_rows": len(controls),
        "cpi_link_event_rows": len(cpi_events),
        "artifacts": {
            "events": artifact(EVENTS_PATH),
            "controls": artifact(CONTROLS_PATH),
            "cpi_link_events": artifact(CPI_LINK_EVENTS_PATH),
            "freeze": artifact(FREEZE_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
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
                    "status": "ready_not_executed",
                    "outcomes_will_be_read": False,
                    "activity_routes": list(ACTIVITY_ROUTES),
                    "separate_links": [
                        "cpi_immediate_leader_transmission",
                        "cpi_meme_response",
                    ],
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
