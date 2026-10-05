"""Run the frozen Council sanctions whole-episode unsigned-activity analysis."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as shared,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_activity_direct as central_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_regional_nominal_clock_direct as regional,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
SOURCE_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "eu_council_sanctions_catalogue_20260913a"
)
CONTRACT_PATH = SOURCE_ROOT / "eu_sanctions_activity_analysis_contract_v2.json"
CONTRACT_SHA256 = "a1cf03ac1aecce429a6951af171d75eb8723825ea7fa680d03552f20e7b2b792"
DIRECT_ROOT = SOURCE_ROOT / "direct_review_20260913a"
CONTROL_PATH = DIRECT_ROOT / "eu_sanctions_activity_controls.csv"
MEMBER_METRICS_PATH = DIRECT_ROOT / "eu_sanctions_activity_member_metrics.csv"
EPISODE_METRICS_PATH = DIRECT_ROOT / "eu_sanctions_activity_episode_metrics.csv"
CELL_PATH = DIRECT_ROOT / "eu_sanctions_activity_route_cells.csv"
ROUTE_PATH = DIRECT_ROOT / "eu_sanctions_activity_route_decisions.csv"
NULL_PATH = DIRECT_ROOT / "eu_sanctions_activity_permutation_summary.csv"
REPORT_PATH = DIRECT_ROOT / "eu_sanctions_activity_plain_review.md"
RESULT_PATH = DIRECT_ROOT / "eu_sanctions_activity_analysis_result.json"

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
HORIZONS = (60, 240, 1440)
OFFSETS = (-120, -60, 0, 60, 120)
PARTITIONS = ("development_2023", "internal_validation_2024", "holdout_2025")
VIEWS = ("natural_background_all_episodes",)
CONTROL_COUNT = 12
CANDIDATE_COUNT = 13
CONTROL_SEARCH_WEEKS = 52
COLLISION_HOURS = 26
MINIMUM_EPISODES = 10
PERMUTATIONS = 2000
PERMUTATION_SEED = 20260913
METRIC_COLUMNS = ("absolute_end_return", "full_high_low_range", "total_volume")
SCORE_COLUMNS = tuple(f"candidate_score_{slot:02d}" for slot in range(CANDIDATE_COUNT))
ADJACENT_PAIRS = (
    (-120, -60),
    (-60, -120),
    (-60, 0),
    (0, -60),
    (0, 60),
    (60, 0),
    (60, 120),
    (120, 60),
)
PLANNED_OUTPUT_PATHS = (
    CONTROL_PATH,
    MEMBER_METRICS_PATH,
    EPISODE_METRICS_PATH,
    CELL_PATH,
    ROUTE_PATH,
    NULL_PATH,
    REPORT_PATH,
    RESULT_PATH,
)
EXPECTED_RESULT_ARTIFACT_PATHS = {
    "controls": CONTROL_PATH,
    "member_metrics": MEMBER_METRICS_PATH,
    "episode_metrics": EPISODE_METRICS_PATH,
    "route_cells": CELL_PATH,
    "route_decisions": ROUTE_PATH,
    "permutation_summary": NULL_PATH,
    "plain_review": REPORT_PATH,
}


@dataclass(frozen=True)
class CellSpec:
    """One partition/view/offset leg in a route."""

    row_indices: np.ndarray
    minimum_episodes: int = MINIMUM_EPISODES


@dataclass(frozen=True)
class RouteSpec:
    """All linked legs for one asset and horizon."""

    key: tuple[str, int]
    cells_by_offset: Mapping[int, tuple[CellSpec, ...]]


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def validate_artifact_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or g0.sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"Frozen artifact changed: {label} -> {path}")
    return path


def _partition_for_year(year: int) -> str:
    return {
        2023: "development_2023",
        2024: "internal_validation_2024",
        2025: "holdout_2025",
    }[year]


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("status") != (
        "frozen_outcome_blind_eu_sanctions_activity_analysis_v2_after_source_only_preflight"
    ):
        raise ValueError("EU sanctions activity contract is not frozen")
    prohibited = (
        contract.get("market_outcomes_opened_before_contract")
        or contract.get("profit_used")
        or contract.get("direction_tested")
        or contract.get("causal_claim_permitted")
    )
    if prohibited:
        raise ValueError("EU sanctions activity contract permits a prohibited outcome")
    supersession = contract.get("supersession", {})
    v1_path = validate_artifact_record(
        supersession.get("v1_contract", {}), label="v1_contract"
    )
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if v1.get("status") != "frozen_outcome_blind_eu_sanctions_activity_analysis":
        raise ValueError("R2B v1 lineage is not the frozen outcome-blind contract")
    if any(
        v1.get(key)
        for key in (
            "market_outcomes_opened_before_contract",
            "profit_used",
            "direction_tested",
            "causal_claim_permitted",
        )
    ):
        raise ValueError("R2B v1 lineage contains a prohibited outcome flag")
    facts = supersession.get("v1_source_only_preflight", {})
    expected = {
        "scored_episodes": 67,
        "episodes_with_12_controls": 39,
        "episodes_with_incomplete_controls": 28,
        "eligible_controls_minimum": 2,
        "eligible_controls_maximum": 33,
        "eligible_controls_median": 17,
    }
    if any(facts.get(key) != value for key, value in expected.items()):
        raise ValueError("R2B v1 source-only preflight lineage changed")
    if facts.get("market_outcomes_opened") is not False:
        raise ValueError("R2B v1 supersession was not outcome-blind")
    assertions = contract.get("pre_run_assertions", {})
    expected_assertions = {
        "scored_whole_episodes": 67,
        "context_only_whole_episodes": 2,
        "asset_horizon_routes": len(ASSETS) * len(HORIZONS),
        "offsets": len(OFFSETS),
        "candidates_per_episode": CANDIDATE_COUNT,
        "randomizations": PERMUTATIONS,
        "primary_scored_views": len(VIEWS),
    }
    if any(assertions.get(key) != value for key, value in expected_assertions.items()):
        raise ValueError("R2B v2 pre-run assertions changed")


def _validate_source_documents(
    source_result: Mapping[str, Any],
    source_freeze: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    if source_result.get("status") != "completed_eu_council_sanctions_source_pilot":
        raise ValueError("EU sanctions source result is not terminal")
    if source_freeze.get("status") != "frozen_outcome_blind_eu_sanctions_source_pilot":
        raise ValueError("EU sanctions source definitions are not frozen")
    if (
        source_result.get("outcomes_read")
        or source_result.get("profit_used")
        or source_freeze.get("outcomes_read")
        or source_freeze.get("profit_used")
    ):
        raise ValueError("EU sanctions source result is not outcome-blind")
    pages = source_manifest.get("pages", [])
    if len(pages) != 18 or [row.get("page") for row in pages] != list(range(1, 19)):
        raise ValueError("EU sanctions official source manifest is incomplete")
    if any(row.get("http_status") != 200 for row in pages):
        raise ValueError("EU sanctions official source page was not HTTP 200")


def load_contract_and_source() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    """Validate every source-only freeze before any market loader can run."""

    if g0.sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("EU sanctions activity contract changed after freeze")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    _validate_contract(contract)
    lineage = {
        name: validate_artifact_record(record, label=name)
        for name, record in contract["source_lineage"].items()
    }
    source_result = json.loads(lineage["source_result"].read_text(encoding="utf-8"))
    source_freeze = json.loads(lineage["source_freeze"].read_text(encoding="utf-8"))
    source_manifest = json.loads(lineage["source_manifest"].read_text(encoding="utf-8"))
    _validate_source_documents(source_result, source_freeze, source_manifest)

    catalogue = pd.read_csv(lineage["corrected_source_catalogue"])
    forbidden = ("profit", "direction", "future_return", "ohlcv")
    if any(token in column.casefold() for column in catalogue for token in forbidden):
        raise ValueError("EU source catalogue contains a prohibited outcome field")
    catalogue["member_clock"] = pd.to_datetime(
        catalogue["displayed_publication_clock"], utc=True, errors="raise"
    )
    if len(catalogue) != 357 or catalogue["event_id"].duplicated().any():
        raise ValueError("EU source catalogue row count or identity changed")
    status_counts = catalogue["selection_status"].value_counts().to_dict()
    if status_counts != {"exclude": 257, "include": 98, "manual_review": 2}:
        raise ValueError(f"EU source selection counts changed: {status_counts}")

    selected = catalogue.loc[catalogue["selection_status"].eq("include")].copy()
    if selected["episode_id"].nunique() != 69:
        raise ValueError("EU selected whole-episode count changed")
    years = selected["member_clock"].dt.year
    context = selected.loc[years.isin([2021, 2022])].copy()
    scored = selected.loc[years.isin([2023, 2024, 2025])].copy()
    scored["analysis_partition"] = scored["member_clock"].dt.year.map(
        _partition_for_year
    )
    expected = {
        "development_2023": 22,
        "internal_validation_2024": 25,
        "holdout_2025": 20,
    }
    actual = scored.groupby("analysis_partition")["episode_id"].nunique().to_dict()
    if actual != expected or context["episode_id"].nunique() != 2:
        raise ValueError(f"EU scored/context whole-episode counts changed: {actual}")
    if scored["episode_id"].nunique() != 67:
        raise ValueError("EU scored whole-episode total changed")
    episode_dates = scored.groupby("episode_id")["member_clock"].agg(
        lambda values: pd.Series(values).dt.date.nunique()
    )
    if not episode_dates.eq(1).all():
        raise ValueError("An EU whole episode spans more than one displayed date")
    return contract, selected.reset_index(drop=True), scored.reset_index(drop=True)


def _flatten_artifacts(value: Any, prefix: str = "") -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        if {"path", "sha256"}.issubset(value):
            records.append(
                {
                    "label": prefix or "artifact",
                    "path": str(value["path"]),
                    "sha256": str(value["sha256"]),
                }
            )
        else:
            for key, child in value.items():
                records.extend(_flatten_artifacts(child, f"{prefix}.{key}".strip(".")))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            records.extend(_flatten_artifacts(child, f"{prefix}[{index}]"))
    return records


def load_external_anchors() -> tuple[DataFrame, list[dict[str, str]]]:
    """Reuse validated external source clocks without opening market outcomes."""

    external, upstream_artifacts = regional.load_external_anchors()
    _, china = regional.load_contract_and_sources()
    china_rows = DataFrame(
        {
            "anchor_utc": pd.to_datetime(china["anchor_utc"], utc=True),
            "source_group": "china_nbs_nominal_clock",
        }
    )
    external = pd.concat([external, china_rows], ignore_index=True)
    external["anchor_utc"] = pd.to_datetime(external["anchor_utc"], utc=True)
    external = external.drop_duplicates(["anchor_utc", "source_group"]).sort_values(
        ["anchor_utc", "source_group"], kind="stable"
    )
    china_artifacts = {
        "analysis_contract": artifact(regional.CONTRACT_PATH),
        "source_catalogue": artifact(regional.source_freeze.CATALOGUE_PATH),
        "source_freeze": artifact(regional.source_freeze.FREEZE_PATH),
        "source_result": artifact(regional.source_freeze.RESULT_PATH),
    }
    flattened = _flatten_artifacts(upstream_artifacts, "regional_external")
    flattened.extend(_flatten_artifacts(china_artifacts, "china_nbs"))
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for record in flattened:
        validate_artifact_record(record, label=record["label"])
        unique[(record["path"], record["sha256"])] = record
    return external.reset_index(drop=True), list(unique.values())


def _timestamp_ns(values: Sequence[pd.Timestamp]) -> np.ndarray:
    """Normalize all pandas timestamp resolutions to integer nanoseconds."""

    return np.asarray([pd.Timestamp(value).value for value in values], dtype=np.int64)


def _within_hours(anchor: pd.Timestamp, blocked_ns: np.ndarray) -> bool:
    if not len(blocked_ns):
        return False
    distance = np.abs(blocked_ns - pd.Timestamp(anchor).value)
    return bool(np.min(distance) <= pd.Timedelta(hours=COLLISION_HOURS).value)


def _external_overlap(
    member_clocks: Sequence[pd.Timestamp], external: DataFrame, hours: int
) -> tuple[int, int]:
    member_ns = _timestamp_ns(member_clocks)
    threshold = pd.Timedelta(hours=hours).value
    external_ns = _timestamp_ns(external["anchor_utc"].tolist())
    if not len(member_ns) or not len(external_ns):
        return 0, 0
    distances = np.abs(external_ns[:, None] - member_ns[None, :])
    matched = external.loc[np.min(distances, axis=1) <= threshold]
    return int(matched["anchor_utc"].nunique()), int(matched["source_group"].nunique())


def add_external_context_diagnostics(
    candidates: DataFrame, selected: DataFrame, external: DataFrame
) -> DataFrame:
    """Add descriptive external overlap without selecting controls or market labels."""

    output = candidates.copy()
    diagnostic_rows: list[dict[str, Any]] = []
    for (episode_id, slot), members in output.groupby(
        ["episode_id", "candidate_slot"], sort=True
    ):
        clocks = members["candidate_member_clock"].tolist()
        anchors_4h, groups_4h = _external_overlap(clocks, external, 4)
        anchors_26h, groups_26h = _external_overlap(clocks, external, 26)
        other_eu = selected.loc[
            ~selected["episode_id"].astype(str).eq(str(episode_id)), "member_clock"
        ]
        other_eu_ns = _timestamp_ns(other_eu.tolist())
        other_eu_collision = any(
            _within_hours(pd.Timestamp(clock), other_eu_ns) for clock in clocks
        )
        diagnostic_rows.append(
            {
                "episode_id": str(episode_id),
                "candidate_slot": int(slot),
                "external_anchor_count_within_4h_of_any_member": anchors_4h,
                "external_source_group_count_within_4h_of_any_member": groups_4h,
                "external_anchor_count_within_26h_of_any_member": anchors_26h,
                "external_source_group_count_within_26h_of_any_member": groups_26h,
                "descriptive_collision_clean": bool(
                    anchors_26h == 0 and not other_eu_collision
                ),
            }
        )
    diagnostics = DataFrame.from_records(diagnostic_rows)
    return output.merge(
        diagnostics,
        on=["episode_id", "candidate_slot"],
        how="left",
        validate="many_to_one",
    )


def _shift_order() -> list[int]:
    return [
        shift
        for distance in range(1, CONTROL_SEARCH_WEEKS + 1)
        for shift in (-distance, distance)
    ]


def build_control_map(scored: DataFrame, selected: DataFrame) -> DataFrame:
    blocked_ns = _timestamp_ns(selected["member_clock"].tolist())
    rows: list[dict[str, Any]] = []
    for episode_id, members in scored.groupby("episode_id", sort=True):
        members = members.sort_values(["member_clock", "event_id"], kind="stable")
        partition = str(members["analysis_partition"].iloc[0])
        year = int(members["member_clock"].dt.year.iloc[0])
        rank = 0
        for shift_weeks in _shift_order():
            shifted = members["member_clock"] + pd.Timedelta(weeks=shift_weeks)
            if not shifted.dt.year.eq(year).all():
                continue
            if any(_within_hours(pd.Timestamp(anchor), blocked_ns) for anchor in shifted):
                continue
            rank += 1
            rows.append(
                {
                    "episode_id": str(episode_id),
                    "analysis_partition": partition,
                    "candidate_slot": rank,
                    "shift_weeks": shift_weeks,
                    "member_count": len(members),
                    "first_shifted_member_clock": shifted.iloc[0],
                    "last_shifted_member_clock": shifted.iloc[-1],
                    "control_type": "same_year_same_weekday_displayed_clock_whole_episode_shift",
                }
            )
            if rank == CONTROL_COUNT:
                break
        if rank != CONTROL_COUNT:
            raise ValueError(f"EU episode has only {rank} source-clean controls: {episode_id}")
    controls = DataFrame.from_records(rows)
    if controls.duplicated(["episode_id", "candidate_slot"]).any():
        raise ValueError("EU control candidate slots are duplicated")
    return controls


def candidate_member_rows(scored: DataFrame, controls: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    shifts = {
        str(episode_id): {
            int(row.candidate_slot): int(row.shift_weeks)
            for row in group.itertuples()
        }
        for episode_id, group in controls.groupby("episode_id", sort=False)
    }
    for episode_id, members in scored.groupby("episode_id", sort=True):
        ordered = members.sort_values(["member_clock", "event_id"], kind="stable")
        episode_shifts = {0: 0, **shifts[str(episode_id)]}
        if sorted(episode_shifts) != list(range(CANDIDATE_COUNT)):
            raise ValueError(f"EU matched set is incomplete: {episode_id}")
        for slot, shift_weeks in episode_shifts.items():
            for member_order, member in enumerate(ordered.itertuples(index=False), start=1):
                rows.append(
                    {
                        "episode_id": str(episode_id),
                        "analysis_partition": str(member.analysis_partition),
                        "candidate_slot": slot,
                        "candidate_kind": "event" if slot == 0 else "control",
                        "shift_weeks": shift_weeks,
                        "member_order": member_order,
                        "member_count": len(ordered),
                        "source_event_id": str(member.event_id),
                        "source_member_clock": pd.Timestamp(member.member_clock),
                        "candidate_member_clock": pd.Timestamp(member.member_clock)
                        + pd.Timedelta(weeks=shift_weeks),
                    }
                )
    return DataFrame.from_records(rows)


def candidate_structure_violations(candidates: DataFrame) -> dict[str, int]:
    same_year = int(
        candidates["candidate_member_clock"].dt.year.ne(
            candidates["source_member_clock"].dt.year
        ).sum()
    )
    same_weekday = int(
        candidates["candidate_member_clock"].dt.weekday.ne(
            candidates["source_member_clock"].dt.weekday
        ).sum()
    )
    spacing = 0
    for _, members in candidates.groupby(["episode_id", "candidate_slot"], sort=True):
        ordered = members.sort_values("member_order", kind="stable")
        source_spacing = np.diff(_timestamp_ns(ordered["source_member_clock"].tolist()))
        candidate_spacing = np.diff(
            _timestamp_ns(ordered["candidate_member_clock"].tolist())
        )
        spacing += int(not np.array_equal(source_spacing, candidate_spacing))
    return {
        "same_year_violations": same_year,
        "same_weekday_violations": same_weekday,
        "internal_member_spacing_violations": spacing,
    }


def source_control_preflight() -> dict[str, Any]:
    contract, selected, scored = load_contract_and_source()
    external, artifacts = load_external_anchors()
    controls = build_control_map(scored, selected)
    candidates = add_external_context_diagnostics(
        candidate_member_rows(scored, controls), selected, external
    )
    candidate_context = candidates.drop_duplicates(["episode_id", "candidate_slot"])
    control_members = candidates.loc[candidates["candidate_kind"].eq("control")]
    selected_ns = _timestamp_ns(selected["member_clock"].tolist())
    violations = int(
        sum(
            _within_hours(pd.Timestamp(clock), selected_ns)
            for clock in control_members["candidate_member_clock"]
        )
    )
    overlap: dict[str, Any] = {}
    for kind, group in candidate_context.groupby("candidate_kind", sort=True):
        overlap[str(kind)] = {
            "candidates": len(group),
            "with_external_anchor_within_4h": int(
                group["external_anchor_count_within_4h_of_any_member"].gt(0).sum()
            ),
            "with_external_anchor_within_26h": int(
                group["external_anchor_count_within_26h_of_any_member"].gt(0).sum()
            ),
            "descriptive_collision_clean": int(
                group["descriptive_collision_clean"].sum()
            ),
            "mean_external_anchors_within_4h": float(
                group["external_anchor_count_within_4h_of_any_member"].mean()
            ),
            "mean_external_source_groups_within_4h": float(
                group["external_source_group_count_within_4h_of_any_member"].mean()
            ),
            "mean_external_anchors_within_26h": float(
                group["external_anchor_count_within_26h_of_any_member"].mean()
            ),
            "mean_external_source_groups_within_26h": float(
                group["external_source_group_count_within_26h_of_any_member"].mean()
            ),
        }
    clean_support: dict[str, dict[str, int]] = {}
    for kind, group in candidate_context.groupby("candidate_kind", sort=True):
        clean = group.groupby("analysis_partition")["descriptive_collision_clean"].sum()
        total = group.groupby("analysis_partition").size()
        clean_support[str(kind)] = {
            partition: int(clean.get(partition, 0)) for partition in PARTITIONS
        }
        clean_support[str(kind)].update(
            {f"{partition}_total": int(total.get(partition, 0)) for partition in PARTITIONS}
        )
    return {
        "status": "v2_source_control_preflight_complete_no_market_opened",
        "contract_status": contract["status"],
        "scored_episodes": int(scored["episode_id"].nunique()),
        "controls": len(controls),
        "controls_per_episode": sorted(controls.groupby("episode_id").size().unique().tolist()),
        "candidate_structure_violations": candidate_structure_violations(candidates),
        "selected_eu_proximity_violations": violations,
        "external_overlap_diagnostics": overlap,
        "descriptive_clean_support_by_partition": clean_support,
        "external_anchor_rows": len(external),
        "external_source_artifacts": len(artifacts),
        "market_outcomes_opened": False,
    }


def _trailing_metrics(
    frame: DataFrame, position: int | None
) -> tuple[str, float, str, float, float]:
    if position is None:
        return "missing_anchor_candle", np.nan, "missing_anchor_candle", np.nan, np.nan
    status_60 = central_activity.minute_window_status(frame, position - 60, 60)
    status_240 = central_activity.minute_window_status(frame, position - 240, 240)
    metrics_60 = (
        central_activity.contiguous_window_metrics(frame, position - 60, 60)
        if status_60 == "usable"
        else None
    )
    metrics_240 = (
        central_activity.contiguous_window_metrics(frame, position - 240, 240)
        if status_240 == "usable"
        else None
    )
    return (
        status_60,
        float(metrics_60["volume"]) if metrics_60 else np.nan,
        status_240,
        float(metrics_240["abs_return"]) if metrics_240 else np.nan,
        float(metrics_240["range"]) if metrics_240 else np.nan,
    )


def extract_member_metrics(
    candidates: DataFrame, frames: Mapping[str, DataFrame]
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for pair, frame in frames.items():
        positions = shared.position_by_date(frame)
        for member in candidates.itertuples(index=False):
            for offset in OFFSETS:
                anchor = pd.Timestamp(member.candidate_member_clock) + pd.Timedelta(
                    minutes=offset
                )
                position = positions.get(anchor)
                trailing = _trailing_metrics(frame, position)
                for horizon in HORIZONS:
                    if position is None:
                        status = "missing_anchor_candle"
                        raw = None
                    else:
                        status = central_activity.minute_window_status(
                            frame, position, horizon
                        )
                        raw = (
                            central_activity.contiguous_window_metrics(
                                frame, position, horizon
                            )
                            if status == "usable"
                            else None
                        )
                    rows.append(
                        {
                            **member._asdict(),
                            "pair": pair,
                            "horizon_minutes": horizon,
                            "offset_minutes": offset,
                            "analysis_anchor": anchor,
                            "window_status": status,
                            "absolute_end_return": (
                                float(raw["abs_return"]) if raw is not None else np.nan
                            ),
                            "full_high_low_range": (
                                float(raw["range"]) if raw is not None else np.nan
                            ),
                            "total_volume": (
                                float(raw["volume"]) if raw is not None else np.nan
                            ),
                            "prior_60m_status": trailing[0],
                            "prior_60m_total_volume": trailing[1],
                            "prior_240m_status": trailing[2],
                            "prior_240m_absolute_end_return": trailing[3],
                            "prior_240m_full_high_low_range": trailing[4],
                        }
                    )
    return DataFrame.from_records(rows)


def aggregate_episode_metrics(member_metrics: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        "episode_id",
        "analysis_partition",
        "candidate_slot",
        "candidate_kind",
        "shift_weeks",
        "pair",
        "horizon_minutes",
        "offset_minutes",
        "external_anchor_count_within_4h_of_any_member",
        "external_source_group_count_within_4h_of_any_member",
        "external_anchor_count_within_26h_of_any_member",
        "external_source_group_count_within_26h_of_any_member",
        "descriptive_collision_clean",
    ]
    for key, group in member_metrics.groupby(groups, sort=True, dropna=False):
        expected = int(group["member_count"].iloc[0])
        complete = len(group) == expected and group["window_status"].eq("usable").all()
        row = dict(zip(groups, key, strict=True))
        row["member_count"] = expected
        row["usable_member_windows"] = int(group["window_status"].eq("usable").sum())
        row["episode_window_status"] = "usable" if complete else "incomplete_member_set"
        for column in METRIC_COLUMNS:
            row[column] = float(group[column].median()) if complete else np.nan
        for column in (
            "prior_60m_total_volume",
            "prior_240m_absolute_end_return",
            "prior_240m_full_high_low_range",
        ):
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.median()) if values.notna().all() else np.nan
        rows.append(row)
    output = DataFrame.from_records(rows)
    for column in (
        "absolute_end_return_ratio",
        "full_high_low_range_ratio",
        "total_volume_ratio",
        "activity_score",
    ):
        output[column] = np.nan
    group_columns = ["episode_id", "pair", "horizon_minutes", "offset_minutes"]
    for key, group in output.groupby(group_columns, sort=True):
        ordered = group.sort_values("candidate_slot", kind="stable")
        if ordered["candidate_slot"].tolist() != list(range(CANDIDATE_COUNT)):
            raise ValueError(f"EU episode candidate slots changed: {key}")
        scores, ratios = regional.symmetric_candidate_scores(
            ordered.loc[:, METRIC_COLUMNS].to_numpy(dtype=float)
        )
        output.loc[ordered.index, "activity_score"] = scores
        output.loc[ordered.index, "absolute_end_return_ratio"] = ratios[:, 0]
        output.loc[ordered.index, "full_high_low_range_ratio"] = ratios[:, 1]
        output.loc[ordered.index, "total_volume_ratio"] = ratios[:, 2]
    return output


def build_score_matrix(episode_metrics: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        "episode_id",
        "analysis_partition",
        "pair",
        "horizon_minutes",
        "offset_minutes",
    ]
    for key, group in episode_metrics.groupby(groups, sort=True):
        ordered = group.sort_values("candidate_slot", kind="stable")
        if ordered["candidate_slot"].tolist() != list(range(CANDIDATE_COUNT)):
            raise ValueError(f"EU score matrix candidate slots changed: {key}")
        row = dict(zip(groups, key, strict=True))
        row.update(dict(zip(SCORE_COLUMNS, ordered["activity_score"], strict=True)))
        rows.append(row)
    return DataFrame.from_records(rows)


def summarize_cells(scores: DataFrame, score_column: str = SCORE_COLUMNS[0]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["pair", "horizon_minutes", "offset_minutes", "analysis_partition"]
    for key, group in scores.groupby(groups, sort=True):
        pair, horizon, offset, partition = key
        for view in VIEWS:
            values = pd.to_numeric(group[score_column], errors="coerce")
            values = values[np.isfinite(values)]
            support = len(values)
            median = float(values.median()) if support else np.nan
            rate = float(values.gt(1.0).mean()) if support else np.nan
            rows.append(
                {
                    "pair": pair,
                    "horizon_minutes": int(horizon),
                    "offset_minutes": int(offset),
                    "analysis_partition": partition,
                    "view": view,
                    "whole_episodes": support,
                    "minimum_whole_episodes": MINIMUM_EPISODES,
                    "median_activity_score": median,
                    "above_control_rate": rate,
                    "studentized_log_activity": regional.studentized_log_activity(
                        values, minimum_events=MINIMUM_EPISODES
                    ),
                    "primary_cell_pass": bool(
                        support >= MINIMUM_EPISODES
                        and np.isfinite(median)
                        and median >= 1.20
                        and np.isfinite(rate)
                        and rate >= 0.55
                    ),
                    "adjacent_cell_pass": bool(
                        support >= MINIMUM_EPISODES
                        and np.isfinite(median)
                        and median >= 1.10
                        and np.isfinite(rate)
                        and rate >= 0.50
                    ),
                }
            )
    return DataFrame.from_records(rows)


def _offset_gate(cells: DataFrame, offset: int, gate_column: str) -> bool:
    selected = cells.loc[cells["offset_minutes"].eq(offset)]
    return bool(
        len(selected) == len(PARTITIONS) * len(VIEWS)
        and selected[gate_column].all()
    )


def _offset_strength(cells: DataFrame, offset: int) -> float:
    values = cells.loc[
        cells["offset_minutes"].eq(offset), "studentized_log_activity"
    ].to_numpy(dtype=float)
    if len(values) != len(PARTITIONS) * len(VIEWS) or not np.isfinite(values).all():
        return np.nan
    return float(np.min(values))


def observed_route_decisions(cells: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (pair, horizon), route in cells.groupby(["pair", "horizon_minutes"], sort=True):
        strengths = {offset: _offset_strength(route, offset) for offset in OFFSETS}
        candidates: list[dict[str, Any]] = []
        for primary, adjacent in ADJACENT_PAIRS:
            strength = (
                min(strengths[primary], strengths[adjacent])
                if np.isfinite(strengths[primary]) and np.isfinite(strengths[adjacent])
                else np.nan
            )
            candidates.append(
                {
                    "primary": primary,
                    "adjacent": adjacent,
                    "strength": strength,
                    "direct_pass": _offset_gate(route, primary, "primary_cell_pass")
                    and _offset_gate(route, adjacent, "adjacent_cell_pass"),
                }
            )
        qualifying = [
            row
            for row in candidates
            if row["direct_pass"] and np.isfinite(row["strength"])
        ]
        complete = [row for row in candidates if np.isfinite(row["strength"])]
        chosen = max(qualifying or complete, key=lambda row: row["strength"], default=None)
        rows.append(
            {
                "pair": pair,
                "horizon_minutes": int(horizon),
                "chosen_primary_offset_minutes": chosen["primary"] if chosen else np.nan,
                "chosen_adjacent_offset_minutes": chosen["adjacent"] if chosen else np.nan,
                "all_direct_gates_pass": bool(qualifying),
                "observed_route_strength": chosen["strength"] if chosen else np.nan,
            }
        )
    return DataFrame.from_records(rows)


def build_route_specs(scores: DataFrame) -> list[RouteSpec]:
    specs: list[RouteSpec] = []
    for key, route in scores.groupby(["pair", "horizon_minutes"], sort=True):
        cells: dict[int, tuple[CellSpec, ...]] = {}
        for offset in OFFSETS:
            legs: list[CellSpec] = []
            for partition in PARTITIONS:
                for view in VIEWS:
                    mask = route["offset_minutes"].eq(offset) & route[
                        "analysis_partition"
                    ].eq(partition)
                    legs.append(CellSpec(route.index[mask].to_numpy(dtype=int)))
            cells[offset] = tuple(legs)
        specs.append(RouteSpec((str(key[0]), int(key[1])), cells))
    return specs


def route_strength_from_scores(selected_scores: np.ndarray, spec: RouteSpec) -> float:
    strengths: dict[int, float] = {}
    for offset, cells in spec.cells_by_offset.items():
        legs = [
            regional.studentized_log_activity(
                selected_scores[cell.row_indices], minimum_events=cell.minimum_episodes
            )
            for cell in cells
        ]
        strengths[offset] = float(np.min(legs)) if np.isfinite(legs).all() else np.nan
    pairs = [
        min(strengths[primary], strengths[adjacent])
        for primary, adjacent in ADJACENT_PAIRS
        if np.isfinite(strengths[primary]) and np.isfinite(strengths[adjacent])
    ]
    return float(max(pairs)) if pairs else np.nan


def run_global_randomization(
    scores: DataFrame,
    specs: Sequence[RouteSpec],
    *,
    iterations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> DataFrame:
    matrix = scores.loc[:, SCORE_COLUMNS].to_numpy(dtype=float)
    codes, event_count = regional.event_codes(scores.rename(columns={"episode_id": "event_id"}))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        choices = rng.integers(0, CANDIDATE_COUNT, size=event_count)
        selected = regional.select_permuted_scores(matrix, codes, choices)
        strengths = [route_strength_from_scores(selected, spec) for spec in specs]
        finite = [value for value in strengths if np.isfinite(value)]
        rows.append(
            {
                "randomization": iteration + 1,
                "familywide_max_route_strength": max(finite) if finite else np.nan,
            }
        )
    return DataFrame.from_records(rows)


def add_familywide_decisions(routes: DataFrame, null: DataFrame) -> DataFrame:
    output = routes.copy()
    maxima = pd.to_numeric(null["familywide_max_route_strength"], errors="coerce")
    maxima = maxima[np.isfinite(maxima)].to_numpy(dtype=float)
    probabilities: list[float] = []
    verdicts: list[str] = []
    for row in output.itertuples(index=False):
        if not row.all_direct_gates_pass or not np.isfinite(row.observed_route_strength):
            probabilities.append(np.nan)
            verdicts.append("direct_gates_not_met")
            continue
        count = int(np.sum(maxima >= float(row.observed_route_strength)))
        probability = float((count + 1) / (len(maxima) + 1))
        probabilities.append(probability)
        verdicts.append(
            "retained_unsigned_activity_association"
            if probability <= 0.05
            else "not_retained_after_familywide_chance_control"
        )
    output["familywide_probability"] = probabilities
    output["retained_unsigned_activity_association"] = (
        output["all_direct_gates_pass"] & output["familywide_probability"].le(0.05)
    )
    output["verdict"] = verdicts
    output["interpretation_limit"] = (
        "Unsigned activity association only; no causation, direction, profit, or trading rule."
    )
    return output


def render_report(
    routes: DataFrame,
    member_metrics: DataFrame,
    scored: DataFrame,
    controls: DataFrame,
    candidates: DataFrame,
) -> str:
    retained = int(routes["retained_unsigned_activity_association"].sum())
    prior_60 = int(member_metrics["prior_60m_total_volume"].notna().sum())
    later_usable = int(member_metrics["window_status"].eq("usable").sum())
    context = candidates.drop_duplicates(["episode_id", "candidate_slot"])
    event_context = context.loc[context["candidate_kind"].eq("event")]
    control_context = context.loc[context["candidate_kind"].eq("control")]
    event_overlap = int(
        event_context["external_anchor_count_within_26h_of_any_member"].gt(0).sum()
    )
    control_overlap = int(
        control_context["external_anchor_count_within_26h_of_any_member"].gt(0).sum()
    )
    return "\n".join(
        [
            "# EU Council Sanctions Unsigned-Activity Review",
            "",
            "## Driver / event clock",
            "",
            (
                "- The source exposure is the frozen Council episode clock in the real "
                "mixed event background. Association does not prove that the Council "
                "release alone caused activity."
            ),
            (
                f"- Scored whole episodes: `{scored['episode_id'].nunique()}`; "
                f"controls: `{len(controls)}`."
            ),
            (
                "- Historical displayed clocks are uncertain and were tested only "
                "through the frozen offsets."
            ),
            "",
            "## Pre-event readiness diagnostics",
            "",
            f"- Member rows with prior-60-minute volume available: `{prior_60}`.",
            (
                "- Readiness diagnostics are trailing context only. They did not "
                "select controls or replace the event."
            ),
            "",
            "## Collisions / confluence",
            "",
            (
                f"- Candidates with an external anchor within 26 hours: events "
                f"`{event_overlap}` of `{len(event_context)}`, controls "
                f"`{control_overlap}` of `{len(control_context)}`."
            ),
            (
                "- Any event/control overlap imbalance or activity concentration in "
                "overlapped episodes is descriptive confluence evidence for a later "
                "batch. It does not dismiss either the EU layer or other-news layer."
            ),
            "",
            "## Later activity outcome",
            "",
            f"- Usable member outcome windows: `{later_usable}`.",
            f"- Retained unsigned-activity routes: `{retained}` of `{len(routes)}`.",
            (
                "- Retention means a warning in naturally mixed conditions, not "
                "source-specific causation. This review does not test direction, profit, "
                "entries, exits, or a trading rule."
            ),
            "",
        ]
    )


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_eu_sanctions_unsigned_activity_analysis":
        raise ValueError("Existing EU sanctions activity result is invalid")
    if result.get("market_outcomes_opened_before_contract") is not False:
        raise ValueError("Existing EU sanctions result lacks an outcome-blind contract flag")
    if (
        result.get("profit_used")
        or result.get("direction_tested")
        or result.get("causal_claim_permitted")
    ):
        raise ValueError("Existing EU sanctions result contains a prohibited claim")
    records = result.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(
        EXPECTED_RESULT_ARTIFACT_PATHS
    ):
        raise ValueError("Existing EU sanctions result artifact key set changed")
    for label, expected_path in EXPECTED_RESULT_ARTIFACT_PATHS.items():
        path = validate_artifact_record(records[label], label=label)
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"Existing EU sanctions artifact path changed: {label}")
    script_path = validate_artifact_record(
        result["analysis_script"], label="analysis_script"
    )
    if script_path.resolve() != ANALYSIS_PATH.resolve():
        raise ValueError("Existing EU sanctions analysis-script path changed")
    contract_path = validate_artifact_record(
        result["analysis_contract"], label="analysis_contract"
    )
    if contract_path.resolve() != CONTRACT_PATH.resolve():
        raise ValueError("Existing EU sanctions analysis-contract path changed")
    for record in result.get("external_source_artifacts", []):
        validate_artifact_record(record, label=record.get("label", "external"))
    for record in result.get("market_source_artifacts", []):
        validate_artifact_record(record, label=record.get("pair", "market"))


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    partial = [
        path for path in PLANNED_OUTPUT_PATHS if path != RESULT_PATH and path.exists()
    ]
    if partial and not overwrite:
        names = ", ".join(path.name for path in partial)
        raise FileExistsError(
            f"R2B partial outputs exist without a result; refusing to mix artifacts: {names}"
        )
    contract, selected, scored = load_contract_and_source()
    external, external_artifacts = load_external_anchors()
    controls = build_control_map(scored, selected)
    candidates = add_external_context_diagnostics(
        candidate_member_rows(scored, controls), selected, external
    )

    frames, market_coverage = regional.load_market_frames()
    member_metrics = extract_member_metrics(candidates, frames)
    episode_metrics = aggregate_episode_metrics(member_metrics)
    scores = build_score_matrix(episode_metrics)
    cells = summarize_cells(scores)
    routes = observed_route_decisions(cells)
    specs = build_route_specs(scores)
    null = run_global_randomization(scores, specs)
    routes = add_familywide_decisions(routes, null)
    report = render_report(routes, member_metrics, scored, controls, candidates)

    DIRECT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(controls, CONTROL_PATH)
    g0.atomic_write_csv(member_metrics, MEMBER_METRICS_PATH)
    g0.atomic_write_csv(episode_metrics, EPISODE_METRICS_PATH)
    g0.atomic_write_csv(cells, CELL_PATH)
    g0.atomic_write_csv(routes, ROUTE_PATH)
    g0.atomic_write_csv(null, NULL_PATH)
    _atomic_write_text(report, REPORT_PATH)
    market_artifacts = [
        {"pair": str(row.pair), "path": str(row.path), "sha256": str(row.sha256)}
        for row in market_coverage.itertuples(index=False)
    ]
    result = {
        "schema_version": 1,
        "status": "completed_eu_sanctions_unsigned_activity_analysis",
        "created_at_utc": g0.utc_now(),
        "market_outcomes_opened_before_contract": False,
        "profit_used": False,
        "direction_tested": False,
        "causal_claim_permitted": False,
        "scored_whole_episodes": int(scored["episode_id"].nunique()),
        "context_only_whole_episodes": int(
            selected.loc[
                selected["member_clock"].dt.year.isin([2021, 2022]), "episode_id"
            ].nunique()
        ),
        "candidate_sets": int(scores["episode_id"].nunique()),
        "routes": len(routes),
        "randomizations": len(null),
        "retained_routes": int(routes["retained_unsigned_activity_association"].sum()),
        "scored_view": "natural_background_all_episodes",
        "analysis_contract": artifact(CONTRACT_PATH),
        "analysis_script": artifact(ANALYSIS_PATH),
        "external_source_artifacts": external_artifacts,
        "market_source_artifacts": market_artifacts,
        "artifacts": {
            "controls": artifact(CONTROL_PATH),
            "member_metrics": artifact(MEMBER_METRICS_PATH),
            "episode_metrics": artifact(EPISODE_METRICS_PATH),
            "route_cells": artifact(CELL_PATH),
            "route_decisions": artifact(ROUTE_PATH),
            "permutation_summary": artifact(NULL_PATH),
            "plain_review": artifact(REPORT_PATH),
        },
        "interpretation_limit": (
            "Repeatable unsigned activity association only; no causation, direction, "
            "profit, entry, exit, or trading rule."
        ),
        "contract_status": contract["status"],
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--preflight-controls", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_controls:
        print(json.dumps(source_control_preflight(), indent=2))
        return os.EX_OK
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "market_outcomes_opened": False,
                    "planned_output_count": len(PLANNED_OUTPUT_PATHS),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
