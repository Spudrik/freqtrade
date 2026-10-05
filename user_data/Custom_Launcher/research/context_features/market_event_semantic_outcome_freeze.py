"""Freeze bounded semantic-story activity and direction questions before outcomes."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
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
    market_event_confirmation_breadth_freeze as confirmation_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_pilot_freeze as semantic_pilot,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


PILOT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "semantic_data_pilot_20260904a"
)
SAMPLE_MANIFEST = PILOT_ROOT / "sample_manifest.json"
LABEL_TEMPLATE = PILOT_ROOT / "story_label_template.jsonl"
REVIEWED_LABELS = PILOT_ROOT / "story_labels_reviewed.jsonl"
EXPECTED_REVIEWED_LABELS_SHA256 = (
    "f5972bc41e102910c8025200d6c7ee2ec4695e9b08b345f71b0c0d8009793ce4"
)
COHORT_FREEZE_RESULT = (
    confirmation_freeze.OUTPUT_ROOT / "event_confirmation_breadth_freeze_result.json"
)
COHORT_FREEZE = confirmation_freeze.OUTPUT_ROOT / "event_confirmation_breadth_freeze.json"

OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "semantic_outcome_20260905a"
)
ROUTES_PATH = OUTPUT_ROOT / "semantic_story_route_catalog.csv"
COVERAGE_PATH = OUTPUT_ROOT / "semantic_story_route_coverage.csv"
CONTROLS_PATH = OUTPUT_ROOT / "semantic_story_control_catalog.csv"
FREEZE_PATH = OUTPUT_ROOT / "semantic_story_outcome_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "semantic_story_outcome_freeze_result.json"

PARTITIONS = (
    "development_2026-06-01_2026-07-15",
    "validation_2026-07-16_2026-08-31",
)
ROUTES = {
    "all_cooled_stories": "All causally observed stories after the common cooldown",
    "moderate_or_major": "Stories labelled moderate or major before market outcomes",
    "positive_or_negative": "Stories with a clear broad-crypto positive or negative sign",
}
EXPECTED_COVERAGE = {
    "all_cooled_stories": {
        PARTITIONS[0]: 38,
        PARTITIONS[1]: 48,
    },
    "moderate_or_major": {
        PARTITIONS[0]: 15,
        PARTITIONS[1]: 24,
    },
    "positive_or_negative": {
        PARTITIONS[0]: 11,
        PARTITIONS[1]: 13,
    },
}
COOLDOWN_HOURS = 6
CONTROL_EXCLUSION_HOURS = 24
CONTROL_COUNT = 12
CONTROL_SEARCH_WEEKS = 52
HORIZONS_HOURS = (1, 4, 8, 24)
MINIMUM_EVENTS_PER_PARTITION = 3


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _stable_id(*values: Any) -> str:
    text = "|".join(map(str, values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for name, metadata in manifest.get("artifacts", {}).items():
        path = PILOT_ROOT / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing semantic pilot artifact: {path}")
        expected = str(metadata.get("sha256", ""))
        if g0.sha256_file(path) != expected:
            raise ValueError(f"Semantic pilot artifact hash changed: {path}")
        contracts[str(name)] = artifact(path)
    if "story_label_template.jsonl" not in contracts:
        raise ValueError("Semantic pilot manifest has no label-template contract.")
    return contracts


def _verify_cohort_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_json(COHORT_FREEZE_RESULT)
    freeze = _read_json(COHORT_FREEZE)
    if result.get("status") != "completed_event_confirmation_breadth_freeze":
        raise ValueError("Parent cohort freeze result is not terminal.")
    if freeze.get("status") != "frozen_event_confirmation_breadth_before_market_outcomes":
        raise ValueError("Parent cohort document is not frozen.")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Parent cohort freeze unexpectedly opened outcomes.")
    if result["artifacts"]["freeze"]["sha256"] != g0.sha256_file(COHORT_FREEZE):
        raise ValueError("Parent cohort freeze hash changed.")
    for source in ("normal_cohort_source", "meme_cohort_source"):
        contract = freeze["cohorts"][source]
        path = Path(contract["path"])
        if g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen cohort source changed: {path}")
    return result, freeze


def load_reviewed_labels() -> tuple[DataFrame, dict[str, Any]]:
    manifest = _read_json(SAMPLE_MANIFEST)
    if manifest.get("schema_version") != semantic_pilot.SCHEMA_VERSION:
        raise ValueError("Semantic pilot manifest schema changed.")
    if not manifest.get("outcome_blind") or manifest.get("price_or_market_outcomes_read"):
        raise ValueError("Semantic pilot was not outcome blind.")
    parents = _verify_manifest_artifacts(manifest)
    if g0.sha256_file(REVIEWED_LABELS) != EXPECTED_REVIEWED_LABELS_SHA256:
        raise ValueError("Reviewed semantic-label hash changed.")
    template = semantic_pilot.read_jsonl(LABEL_TEMPLATE)
    records = semantic_pilot.read_jsonl(REVIEWED_LABELS)
    expected_ids = {str(record["story_id"]) for record in template}
    report = semantic_pilot.validate_completed_labels(records, expected_ids)
    if not report["semantic_schema_gate_passed"]:
        raise ValueError(f"Reviewed semantic labels failed: {report['errors'][:3]}")
    if len(records) != 120:
        raise ValueError(f"Expected 120 reviewed stories, found {len(records)}.")
    frame = DataFrame.from_records(records)
    frame["first_seen_at"] = pd.to_datetime(
        frame["first_seen_at"], utc=True, errors="raise", format="mixed"
    )
    if set(frame["whole_story_partition"].astype(str)) != set(PARTITIONS):
        raise ValueError("Reviewed stories do not preserve the two frozen partitions.")
    if frame["story_id"].astype(str).duplicated().any():
        raise ValueError("Reviewed semantic story IDs are not unique.")
    if (pd.to_numeric(frame["independent_source_group_count"]) >= 2).any():
        raise ValueError(
            "This pilot unexpectedly contains independent-source confluence; "
            "freeze a separate predeclared route before testing it."
        )
    return frame, {
        "sample_manifest": artifact(SAMPLE_MANIFEST),
        "reviewed_labels": artifact(REVIEWED_LABELS),
        "pilot_artifacts": parents,
        "label_validation": report,
    }


def common_cooldown(frame: DataFrame, hours: int = COOLDOWN_HOURS) -> DataFrame:
    ordered = frame.sort_values(["first_seen_at", "story_id"], kind="stable")
    keep: list[int] = []
    last: pd.Timestamp | None = None
    gap = pd.Timedelta(hours=hours)
    for index, timestamp in ordered["first_seen_at"].items():
        current = pd.Timestamp(timestamp)
        if last is None or current - last >= gap:
            keep.append(index)
            last = current
    return ordered.loc[keep].reset_index(drop=True)


def build_routes(labels: DataFrame) -> DataFrame:
    cooled = common_cooldown(labels)
    pieces: list[DataFrame] = []
    masks = {
        "all_cooled_stories": pd.Series(True, index=cooled.index),
        "moderate_or_major": cooled["severity"].isin(["moderate", "major"]),
        "positive_or_negative": cooled["direction"].isin(["positive", "negative"]),
    }
    columns = [
        "story_id",
        "first_seen_at",
        "whole_story_partition",
        "direction",
        "severity",
        "novelty",
        "confidence",
        "article_count",
        "independent_source_count",
        "independent_source_group_count",
        "representative_title",
    ]
    for route_id, mask in masks.items():
        selected = cooled.loc[mask, columns].copy()
        selected["analysis_group"] = "semantic_story"
        selected["route_id"] = route_id
        selected["route_label"] = ROUTES[route_id]
        selected = selected.rename(
            columns={
                "first_seen_at": "anchor_utc",
                "whole_story_partition": "whole_event_partition",
            }
        )
        pieces.append(selected)
    routes = pd.concat(pieces, ignore_index=True)
    routes["route_event_id"] = [
        _stable_id("semantic_story", route, story_id, pd.Timestamp(anchor).isoformat())
        for route, story_id, anchor in routes[
            ["route_id", "story_id", "anchor_utc"]
        ].itertuples(index=False, name=None)
    ]
    if routes["route_event_id"].duplicated().any():
        raise ValueError("Semantic route-event IDs are not unique.")
    return routes


def route_coverage(routes: DataFrame) -> DataFrame:
    counts = (
        routes.groupby(
            ["analysis_group", "route_id", "route_label", "whole_event_partition"],
            as_index=False,
            sort=True,
        )["route_event_id"]
        .nunique()
        .rename(columns={"route_event_id": "event_count"})
    )
    for route_id, expected_by_partition in EXPECTED_COVERAGE.items():
        actual = counts.loc[counts["route_id"].eq(route_id)].set_index(
            "whole_event_partition"
        )["event_count"]
        for partition, expected in expected_by_partition.items():
            if int(actual.get(partition, 0)) != expected:
                raise ValueError(
                    f"{route_id} {partition} coverage changed: "
                    f"{int(actual.get(partition, 0))} != {expected}."
                )
    counts["coverage_eligible"] = counts["event_count"].ge(10)
    if not counts["coverage_eligible"].all():
        raise ValueError("A predeclared semantic route has fewer than ten events.")
    counts["status"] = "frozen_ready"
    return counts


def prior_week_controls(routes: DataFrame) -> DataFrame:
    blocked = tuple(pd.Timestamp(value) for value in routes["anchor_utc"].unique())
    exclusion = pd.Timedelta(hours=CONTROL_EXCLUSION_HOURS)
    records: list[dict[str, Any]] = []
    for event in routes.itertuples(index=False):
        rank = 0
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            anchor = pd.Timestamp(event.anchor_utc) - pd.Timedelta(weeks=weeks)
            if any(abs(anchor - other) <= exclusion for other in blocked):
                continue
            rank += 1
            records.append(
                {
                    "route_event_id": event.route_event_id,
                    "analysis_group": event.analysis_group,
                    "route_id": event.route_id,
                    "story_id": event.story_id,
                    "event_anchor_utc": event.anchor_utc,
                    "whole_event_partition": event.whole_event_partition,
                    "control_anchor_utc": anchor,
                    "control_rank": rank,
                    "control_type": "same_weekday_hour_prior_week",
                    "exclusion_hours": CONTROL_EXCLUSION_HOURS,
                }
            )
            if rank == CONTROL_COUNT:
                break
        if rank != CONTROL_COUNT:
            raise ValueError(
                f"Only {rank} ordinary controls found for {event.route_event_id}."
            )
    controls = DataFrame.from_records(records)
    if controls.duplicated(["route_event_id", "control_rank"]).any():
        raise ValueError("Semantic control IDs are not unique within route events.")
    if (controls.groupby("route_event_id").size() != CONTROL_COUNT).any():
        raise ValueError("Every semantic route event must have twelve controls.")
    return controls


def freeze_document(
    routes: DataFrame,
    controls: DataFrame,
    parent_contracts: Mapping[str, Any],
    cohort_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_semantic_story_questions_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "plain_questions": {
            "all_cooled_stories": (
                "Do causally observed stories precede more market activity than "
                "matched ordinary periods?"
            ),
            "moderate_or_major": (
                "Do moderate-or-major stories mark more activity than all stories, "
                "minor-or-unclear stories, and matched ordinary periods?"
            ),
            "positive_or_negative": (
                "When a reviewed story has a defensible broad-crypto sign, does price "
                "move in that direction more often than simple controls?"
            ),
        },
        "route_rows": len(routes),
        "control_rows": len(controls),
        "partitions": list(PARTITIONS),
        "common_cooldown_hours": COOLDOWN_HOURS,
        "horizons_hours": list(HORIZONS_HOURS),
        "hourly_alignment": {
            "rule": (
                "Start at the first 1h candle whose opening timestamp is at or after "
                "the story first-seen time; an exact-hour story may use that hour."
            ),
            "operation": "ceil_to_1h",
            "maximum_alignment_delay_minutes": 60,
            "controls_use_identical_alignment": True,
        },
        "cohorts": cohort_freeze["cohorts"],
        "activity_target": {
            "definition": (
                "Combined absolute movement, full range, and volume score at least "
                "1.20 times its matched ordinary-period baseline."
            ),
            "minimum_ratio": 1.20,
            "comparators": [
                "same_weekday_hour_prior_week",
                "all_story_reference",
                "minor_or_unclear_story_reference_for_severity",
            ],
            "retention_rule": {
                "minimum_events_per_partition": 10,
                "minimum_success_rate_each_partition": 0.55,
                "moderate_or_major_minimum_lift_over_minor_or_unclear": 0.05,
                "moderate_or_major_minimum_lift_over_all_stories": 0.05,
                "single_isolated_scope_or_horizon": "provisional_not_retained",
            },
        },
        "direction_target": {
            "issued_only_for": ["positive", "negative"],
            "unclear_or_mixed": "abstain",
            "definition": "Reviewed story sign agrees with the future signed return.",
            "comparators": [
                "development_majority_direction",
                "causal_prior_30d_trend",
                "rotated_story_sign_within_partition",
                "matched_ordinary_period_with_story_sign",
            ],
            "retention_rule": {
                "minimum_events": 10,
                "minimum_events_per_partition": MINIMUM_EVENTS_PER_PARTITION,
                "minimum_overall_rate": 0.55,
                "minimum_partition_rate": 0.50,
                "minimum_lift_over_strongest_comparator": 0.03,
            },
        },
        "excluded_routes": {
            "major_only": "Only one reviewed story is labelled major.",
            "independent_source_confluence": (
                "No reviewed story has two independent source groups."
            ),
            "narrative_accumulation": (
                "Requires a separately frozen trailing-window balance test after this "
                "complete semantic sibling batch."
            ),
        },
        "control_rule": {
            "count_per_event": CONTROL_COUNT,
            "search_weeks": CONTROL_SEARCH_WEEKS,
            "same_weekday_hour": True,
            "exclude_any_cooled_story_within_hours": CONTROL_EXCLUSION_HOURS,
        },
        "research_boundary": (
            "Direct semantic source test only; no FreqAI, profit, entry, exit, or "
            "trading-rule promotion is tested."
        ),
        "parent_contracts": dict(parent_contracts),
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = _read_json(RESULT_PATH)
        if result.get("status") != "completed_semantic_story_outcome_freeze":
            raise ValueError("Existing semantic outcome freeze is not terminal.")
        return result
    labels, parents = load_reviewed_labels()
    cohort_result, cohort = _verify_cohort_freeze()
    parents = {
        **parents,
        "cohort_freeze_result": artifact(COHORT_FREEZE_RESULT),
        "cohort_freeze": artifact(COHORT_FREEZE),
        "cohort_freeze_status": cohort_result["status"],
    }
    routes = build_routes(labels)
    coverage = route_coverage(routes)
    controls = prior_week_controls(routes)
    freeze = freeze_document(routes, controls, parents, cohort)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(routes, ROUTES_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_csv(controls, CONTROLS_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_semantic_story_outcome_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "route_count": int(routes["route_id"].nunique()),
        "route_rows": len(routes),
        "control_rows": len(controls),
        "coverage": {
            route: {
                partition: int(
                    routes.loc[
                        routes["route_id"].eq(route)
                        & routes["whole_event_partition"].eq(partition)
                    ].shape[0]
                )
                for partition in PARTITIONS
            }
            for route in ROUTES
        },
        "artifacts": {
            "routes": artifact(ROUTES_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "controls": artifact(CONTROLS_PATH),
            "freeze": artifact(FREEZE_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "outcomes_read": False,
                    "reviewed_labels": str(REVIEWED_LABELS),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
