"""Freeze the compact 2026 confirmation of five event-signal families."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import copy
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_portfolio as portfolio,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_signal_fresh_2026_20260909a"
OUTPUT_ROOT = breadth.OUTPUT_ROOT.parent / RUN_ID
EVENTS_PATH = OUTPUT_ROOT / "event_signal_fresh_event_catalog.csv"
SAMPLES_PATH = OUTPUT_ROOT / "event_signal_fresh_sample_catalog.csv"
REGISTRY_PATH = OUTPUT_ROOT / "event_signal_fresh_profile_registry.json"
FREEZE_PATH = OUTPUT_ROOT / "event_signal_fresh_freeze.json"

SOURCE_START_UTC = pd.Timestamp("2021-06-01T00:00:00Z")
CURRENT_START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
CURRENT_SPLIT_UTC = pd.Timestamp("2026-05-01T00:00:00Z")
SOURCE_STOP_UTC = pd.Timestamp("2026-08-21T00:00:00Z")
CURRENT_PARTITION = "current_diagnostic_2026_pre_freeze"
VALIDATION_PERIODS = (
    "untouched_confirmation_2026_jan_apr",
    "untouched_confirmation_2026_may_aug",
)

HORIZONS = breadth.HORIZONS
READY_BLOCK = "event_signal_fresh_2026_samples"
NORMAL_PAIRS = breadth.NORMAL_PAIRS
FAMILIES = breadth.FAMILIES
KINDS = breadth.KINDS
ALL_FEATURES = breadth.ALL_FEATURES
EVENT_IDENTITY_FEATURES = breadth.EVENT_IDENTITY_FEATURES
SIGNED_SOURCE_FEATURES = breadth.SIGNED_SOURCE_FEATURES
EVENT_CONFLUENCE_FEATURES = breadth.EVENT_CONFLUENCE_FEATURES

PRIMARY_TARGET = "&-meb_log_volume_ratio_h1"
TARGETS = (PRIMARY_TARGET,)
TARGET_METADATA = (
    {
        "target": PRIMARY_TARGET,
        "metric": "log_volume_ratio",
        "horizon_hours": 1,
        "plain_meaning": (
            "Next-hour volume relative to the typical volume known before the hour."
        ),
    },
)

REPRESENTATIVES: tuple[dict[str, str], ...] = (
    {
        "family_id": "major_event_information",
        "route_id": "event_with_recent_confirmation",
        "profile_id": "event_plus_recent_market",
        "reason": "A known event and immediately visible market activity agree.",
    },
    {
        "family_id": "background_and_market_leadership",
        "route_id": "event_with_background",
        "profile_id": "event_plus_background",
        "reason": "A known event is interpreted within the preceding market background.",
    },
    {
        "family_id": "calculated_reaction_areas",
        "route_id": "event_with_level_cluster",
        "profile_id": "event_plus_cluster",
        "reason": "A known event occurs near several independent calculated areas.",
    },
    {
        "family_id": "local_participation_and_pressure",
        "route_id": "recent_local_participation",
        "profile_id": "recent_market_only",
        "reason": "Recent price, range, and volume show changing local participation.",
    },
    {
        "family_id": "cross_asset_transmission_and_amplification",
        "route_id": "event_with_cross_market_confirmation",
        "profile_id": "event_plus_cross_market",
        "reason": "A known event coincides with activity across established crypto markets.",
    },
)

PROFILE_DEFINITIONS = {
    item["profile_id"]: copy.deepcopy(breadth.PROFILE_DEFINITIONS[item["profile_id"]])
    for item in REPRESENTATIVES
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def model_period(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if timestamp.year <= 2023:
        return "development_2021_2023"
    if timestamp.year == 2024:
        return "walk_forward_validation_2024"
    if timestamp.year == 2025:
        return "walk_forward_validation_2025"
    if CURRENT_START_UTC <= timestamp < CURRENT_SPLIT_UTC:
        return VALIDATION_PERIODS[0]
    if CURRENT_SPLIT_UTC <= timestamp < SOURCE_STOP_UTC:
        return VALIDATION_PERIODS[1]
    return "outside_scored_period"


@contextmanager
def _breadth_scope(
    *, partitions: tuple[str, ...], start: pd.Timestamp, stop: pd.Timestamp
) -> Iterator[None]:
    names = ("PARTITIONS", "SOURCE_START_UTC", "SOURCE_STOP_UTC", "model_period")
    previous = {name: getattr(breadth, name) for name in names}
    try:
        breadth.PARTITIONS = partitions
        breadth.SOURCE_START_UTC = start
        breadth.SOURCE_STOP_UTC = stop
        breadth.model_period = model_period
        yield
    finally:
        for name, value in previous.items():
            setattr(breadth, name, value)


def build_event_catalog() -> DataFrame:
    breadth.load_freeze()
    historical = pd.read_csv(breadth.EVENTS_PATH)
    for column in ("decision_utc", "model_anchor_utc"):
        historical[column] = pd.to_datetime(historical[column], utc=True, errors="raise")
    historical = historical.loc[
        historical["model_anchor_utc"].lt(CURRENT_START_UTC)
    ].copy()
    old_episode_map = historical.set_index("event_id")["event_episode_id"].astype(str)

    with _breadth_scope(
        partitions=(CURRENT_PARTITION,),
        start=CURRENT_START_UTC,
        stop=SOURCE_STOP_UTC,
    ):
        current = breadth.build_event_catalog()
    if current.empty:
        raise ValueError("The frozen 2026 confirmation has no eligible events.")

    combined = pd.concat((historical, current), ignore_index=True)
    if combined["event_id"].duplicated().any():
        raise ValueError("The combined historical/current event identifiers are not unique.")
    combined["model_period"] = combined["model_anchor_utc"].map(model_period)
    combined = breadth.assign_event_episodes(combined).sort_values(
        ["model_anchor_utc", "event_family", "event_id"], kind="stable"
    ).reset_index(drop=True)
    rebuilt_old = combined.loc[
        combined["model_anchor_utc"].lt(CURRENT_START_UTC)
    ].set_index("event_id")["event_episode_id"].astype(str)
    if not rebuilt_old.reindex(old_episode_map.index).equals(old_episode_map):
        raise ValueError("Adding 2026 events changed a frozen historical episode identifier.")
    return combined


def build_control_catalog(events: DataFrame) -> DataFrame:
    partitions = (*breadth.PARTITIONS, CURRENT_PARTITION)
    with _breadth_scope(
        partitions=partitions,
        start=SOURCE_START_UTC,
        stop=SOURCE_STOP_UTC,
    ):
        return breadth.build_control_catalog(events)


def build_sample_catalog(events: DataFrame, controls: DataFrame) -> DataFrame:
    with _breadth_scope(
        partitions=(*breadth.PARTITIONS, CURRENT_PARTITION),
        start=SOURCE_START_UTC,
        stop=SOURCE_STOP_UTC,
    ):
        return breadth.build_sample_catalog(events, controls)


def build_registry() -> dict[str, Any]:
    profiles = {
        name: {
            "profile_id": name,
            "role": name,
            "feature_blocks": definition["blocks"],
            "feature_columns": definition["features"],
            "required_ready_blocks": [READY_BLOCK],
            "targets": list(TARGETS),
            "seed": 2026090901,
        }
        for name, definition in PROFILE_DEFINITIONS.items()
    }
    return {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_event_freqai_outcome_materialization",
        "profile_count": len(profiles),
        "profiles": profiles,
        "comparisons": [],
        "all_feature_columns": list(ALL_FEATURES),
        "targets_declared_without_values": list(TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "future_outcomes_read": False,
    }


def source_contracts() -> dict[str, Any]:
    return {
        "historical_breadth_freeze": artifact(breadth.FREEZE_PATH),
        "historical_signal_portfolio_freeze": artifact(portfolio.FREEZE_PATH),
        "layer2_events": artifact(breadth.layer2.EVENTS_PATH),
        "layer2_controls": artifact(breadth.layer2.CONTROLS_PATH),
        "cpi_catalog": artifact(breadth.cpi.CATALOG_PATH),
        "source_activity_events": artifact(breadth.breadth.EVENTS_PATH),
        "source_activity_controls": artifact(breadth.breadth.CONTROLS_PATH),
        "cross_asset_catalog": artifact(breadth.cross_asset.CATALOG_PATH),
        "esma_catalog": artifact(breadth.esma.CATALOG_PATH),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_event_signal_fresh_2026_before_outcomes":
        raise ValueError("The fresh 2026 event-signal freeze is not valid.")
    if frozen.get("future_outcomes_read") or frozen.get("profit_used"):
        raise ValueError("The fresh 2026 event-signal freeze crossed its research boundary.")
    for contract in frozen["source_contracts"].values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen fresh-confirmation source changed: {path}")
    for contract in frozen["frozen_artifacts"].values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen fresh-confirmation artifact changed: {path}")
    return frozen


def execute() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        return load_freeze()
    portfolio.verify_freeze()
    events = build_event_catalog()
    controls = build_control_catalog(events)
    samples = build_sample_catalog(events, controls)
    current_events = events.loc[events["model_period"].isin(VALIDATION_PERIODS)]
    support = current_events.groupby("model_period")["event_episode_id"].nunique()
    if set(support.index) != set(VALIDATION_PERIODS) or support.lt(20).any():
        raise ValueError(f"Fresh period support is below the frozen gate: {support.to_dict()}")

    registry = build_registry()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(events, EVENTS_PATH)
    g0.atomic_write_csv(samples, SAMPLES_PATH)
    g0.atomic_write_json(registry, REGISTRY_PATH)
    freeze = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_event_signal_fresh_2026_before_outcomes",
        "plain_objective": (
            "Confirm on later 2026 whole events whether one representative from each of "
            "five signal families still identifies unusually high next-hour volume."
        ),
        "event_count": len(events),
        "event_episode_count": int(events["event_episode_id"].nunique()),
        "current_event_count": len(current_events),
        "current_event_episode_count": int(current_events["event_episode_id"].nunique()),
        "current_episode_support": {key: int(value) for key, value in support.items()},
        "sample_count": len(samples),
        "actual_sample_count": int(samples["sample_kind"].eq("actual_event").sum()),
        "control_sample_count": int(samples["sample_kind"].eq("matched_control").sum()),
        "current_source_families": sorted(current_events["event_family"].unique()),
        "pairs": list(NORMAL_PAIRS),
        "profiles": [dict(item) for item in REPRESENTATIVES],
        "profile_count": len(PROFILE_DEFINITIONS),
        "primary_target": PRIMARY_TARGET,
        "source_start_utc": SOURCE_START_UTC,
        "current_start_utc": CURRENT_START_UTC,
        "current_split_utc": CURRENT_SPLIT_UTC,
        "source_stop_exclusive_utc": SOURCE_STOP_UTC,
        "evaluation_periods": list(VALIDATION_PERIODS),
        "calibration": {
            "lookback_hours": portfolio.CALIBRATION_LOOKBACK_HOURS,
            "minimum_prior_hours": portfolio.MIN_CALIBRATION_HOURS,
            "activity_issue_quantile": portfolio.ACTIVITY_SCORE_QUANTILE,
            "rule_source": str(portfolio.FREEZE_PATH.resolve()),
            "thresholds_retuned_on_2026": False,
        },
        "decision_rule": {
            "minimum_unique_event_episodes_each_period": 20,
            "minimum_reaction_rate": portfolio.LEAD_RATE,
            "strong_reaction_rate": portfolio.STRONG_RATE,
            "minimum_enrichment_over_non_calls": portfolio.MIN_REACTION_ENRICHMENT,
            "matched_controls": "Reported as a diagnostic, not a newly tuned gate.",
        },
        "scope_boundary": {
            "source_families_absent_in_2026_are_not_confirmed": True,
            "direction_tested": False,
            "profit_used": False,
            "trading_rule_tested": False,
            "meme_coins_tested": False,
        },
        "source_contracts": source_contracts(),
        "frozen_artifacts": {
            "events": artifact(EVENTS_PATH),
            "samples": artifact(SAMPLES_PATH),
            "profile_registry": artifact(REGISTRY_PATH),
        },
        "future_outcomes_read": False,
        "profit_used": False,
    }
    g0.atomic_write_json(freeze, FREEZE_PATH)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = execute()
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
