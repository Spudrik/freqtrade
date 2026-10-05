from __future__ import annotations

# Freeze G5A without inspecting reaction, direction, volume, range, pressure, or profit.
# The expensive causal event construction is reused from the corrected G4A implementation.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3g,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_breadth as g4a_breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_unconditioned_freeze as g4a_freeze,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)


SCHEMA_VERSION = 1
BRANCH_ID = "g5a_meme_lvn_below_independent_replication"
SELECTION_SEED = "g5a_meme_lvn_below_independent_v1"
G5_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
G4A_FREEZE_RECORD = (
    OUTPUT_ROOT
    / "generation4_branches"
    / "g4a_thin_lvn_one_minute_breadth_confirmation"
    / "g4a_thin_lvn_1m_unconditioned_freeze_20260814b"
    / "g4a_freeze_record.json"
)
REPORT_ROOT = OUTPUT_ROOT / "generation5_branches" / BRANCH_ID
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation5_branches" / BRANCH_ID
CONFIRMATION_PERIODS = ("meme_validation_early", "meme_validation_late")
LEVEL_NAME = "lvn_below"
APPROACH_STATE = "from_above"
PRIOR_PATH_SEPARATION_HOURS = 48
MINIMUM_EPISODES = 30
TARGET_EPISODES = 60
MINIMUM_PAIRS = 5

# This projection is the complete information surface admitted to sample selection.
# It deliberately excludes every contact/future outcome column carried by G3A/G4A.
SELECTION_COLUMNS = (
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "level_column",
    "representation",
    "control",
    "zone_method",
    "base_index",
    "event_time",
    "period",
    "source_available_at",
    "source_open",
    "level_price",
    "zone_half_width",
    "zone_half_width_atr",
    "base_atr",
    "approach_state",
    "approach_code",
    "pre_distance_atr",
    "level_score",
    "level_identity",
    "attr_vp_value_area_width_pct",
    "state_vp_level_persistence_bars",
    "state_peer_connected_dependency_group_count",
    "peer_cluster_survives_without_lvn",
    "attribute_value_for_tier",
    "attribute_tier",
    "parent_high_thinness_score",
    "cohort",
    "parent_run_id",
    "selection_key",
    "selection_hash",
    "episode_id",
    "g4a_selection_hash",
    "qualification_reason",
    "direction_used_for_selection",
    "retrospective_price_reference",
    "price_reference_known_at_event",
    "level_role",
)

COMPONENT_SOURCE_COLUMNS = (
    "base_index",
    "event_time",
    "level_name",
    "episode_id",
    "level_price",
    "base_atr",
    "zone_half_width",
    "state_vp_level_persistence_bars",
    "cohort",
    "pair",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the outcome-blind G5A meme lvn_below replication sample. "
            "This command does not read one-minute paths or select on reaction outcomes."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return freeze_replication_sample(args.run_id, overwrite=bool(args.overwrite))


def freeze_replication_sample(run_id: str, *, overwrite: bool) -> int:
    branch = validate_g5_branch()
    parent, parent_sample_path = validate_g4a_parent()
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g5a_freeze_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"G5A freeze already exists: {record_path}")

    request = request_contract(run_id, branch, parent, parent_sample_path)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "branch_id": BRANCH_ID,
        "status": "freezing_outcome_blind_support",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        source = next(item for item in g3g.COHORT_SOURCES if item.cohort == "meme")
        raw = g4a_freeze.build_unconditioned_cohort(source)
        selection = selection_projection(raw)
        parent_sample = pd.read_csv(parent_sample_path, low_memory=False)
        parent_sample["event_time"] = pd.to_datetime(parent_sample["event_time"], utc=True)

        stage_rows: list[dict[str, Any]] = []
        stage_rows.extend(inventory_rows(selection, "eligible_lvn_below_validation"))
        exact_unused = selection.loc[
            ~selection["episode_id"].isin(set(parent_sample["episode_id"].astype(str)))
        ].copy()
        stage_rows.extend(inventory_rows(exact_unused, "after_exact_episode_exclusion"))
        path_unused = exclude_prior_outcome_paths(
            exact_unused,
            exposed_times=parent_sample["event_time"],
            hours=PRIOR_PATH_SEPARATION_HOURS,
        )
        stage_rows.extend(inventory_rows(path_unused, "after_prior_path_exclusion"))
        independent = select_globally_independent(path_unused)
        stage_rows.extend(inventory_rows(independent, "globally_independent"))
        sample = select_balanced_confirmation_sample(independent)
        stage_rows.extend(inventory_rows(sample, "frozen_sample"))

        support = support_decision(sample)
        support_path = run_dir / "g5a_outcome_blind_support_inventory.csv"
        candidate_path = artifact_dir / "g5a_independent_candidate_metadata.parquet"
        atomic_write_csv(DataFrame(stage_rows), support_path)
        atomic_write_parquet(g3g.sort_episode_rows(independent), candidate_path)
        record.update(
            {
                "outcome_columns_present_in_parent_surface": len(
                    set(raw.columns).difference(SELECTION_COLUMNS)
                ),
                "outcome_columns_admitted_to_selection": 0,
                "reaction_outcomes_opened": False,
                "direction_outcomes_opened": False,
                "profit_used": False,
                "support_decision": support,
                "stage_counts": stage_count_map(DataFrame(stage_rows)),
                "artifacts": {
                    "support_inventory_csv": g3g.artifact_record(support_path),
                    "independent_candidate_metadata_parquet": g3g.artifact_record(
                        candidate_path
                    ),
                },
            }
        )
        if not support["passed"]:
            record.update(
                {
                    "status": "parked_insufficient_independent_support",
                    "completed_at_utc": utc_now(),
                    "plain_language_result": (
                        "The level-specific unused sample did not reach the frozen minimum "
                        "without sharing a 48-hour outcome path with Generation 4. No reaction "
                        "or direction outcome was opened."
                    ),
                }
            )
            atomic_write_json(record, record_path)
            print(json.dumps(record["support_decision"], indent=2))
            return 0

        component_source = raw.loc[
            raw["episode_id"].astype(str).isin(set(sample["episode_id"].astype(str))),
            list(COMPONENT_SOURCE_COLUMNS),
        ].copy()
        components, summaries = g4a_freeze.freeze_geometric_components(component_source)
        frozen = sample.merge(
            summaries,
            on=["episode_id", "cohort", "pair", "event_time"],
            how="left",
            validate="one_to_one",
        )
        validate_frozen_sample(frozen)
        acquisition = g3g.acquisition_intervals(frozen)

        sample_path = run_dir / "g5a_frozen_episode_sample.csv"
        component_path = run_dir / "g5a_frozen_sample_components.csv"
        acquisition_path = run_dir / "g5a_one_minute_acquisition_intervals.csv"
        atomic_write_csv(g3g.sort_episode_rows(frozen), sample_path)
        atomic_write_csv(g3g.sort_component_rows(components), component_path)
        atomic_write_csv(
            acquisition.sort_values(["pair", "interval_start_utc"]), acquisition_path
        )
        record["artifacts"].update(
            {
                "frozen_sample_csv": g3g.artifact_record(sample_path),
                "frozen_sample_components_csv": g3g.artifact_record(component_path),
                "acquisition_intervals_csv": g3g.artifact_record(acquisition_path),
            }
        )
        record.update(
            {
                "status": "frozen_before_reaction_outcomes",
                "completed_at_utc": utc_now(),
                "sample_episode_count": len(frozen),
                "sample_pair_count": int(frozen["pair"].nunique()),
                "sample_period_counts": g3g.nested_counts(frozen, ["period"]),
                "sample_pair_counts": g3g.nested_counts(frozen, ["pair"]),
                "sample_anchor_counts": g3g.nested_counts(
                    frozen, ["cluster_causal_anchor_timeframe"]
                ),
                "acquisition_intervals": len(acquisition),
                "expected_one_minute_rows_before_file_overlap": int(
                    acquisition["expected_one_minute_rows"].sum()
                ),
                "plain_language_result": (
                    "The independent meme lvn_below sample passed the frozen support gate "
                    "using only causal selection fields. One-minute reaction and direction "
                    "outcomes remain unopened."
                ),
            }
        )
        atomic_write_json(record, record_path)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "support": support,
                    "periods": record["sample_period_counts"],
                    "pairs": record["sample_pair_count"],
                    "acquisition_intervals": len(acquisition),
                },
                indent=2,
            )
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def validate_g5_branch() -> dict[str, Any]:
    if not G5_BATCH.is_file():
        raise FileNotFoundError(f"Missing frozen G5 batch: {G5_BATCH}")
    batch = json.loads(G5_BATCH.read_text(encoding="utf-8"))
    if int(batch.get("generation", -1)) != 5:
        raise ValueError("G5 batch generation is not 5")
    if batch.get("status") != "frozen_before_generation5_reaction_outcomes":
        raise ValueError("G5 batch is not frozen before outcomes")
    matches = [item for item in batch.get("branches", []) if item.get("id") == BRANCH_ID]
    if len(matches) != 1 or matches[0].get("status") != "frozen_next_batch":
        raise ValueError(f"Frozen branch missing or invalid: {BRANCH_ID}")
    return matches[0]


def validate_g4a_parent() -> tuple[dict[str, Any], Path]:
    if not G4A_FREEZE_RECORD.is_file():
        raise FileNotFoundError(f"Missing G4A parent record: {G4A_FREEZE_RECORD}")
    parent = json.loads(G4A_FREEZE_RECORD.read_text(encoding="utf-8"))
    if parent.get("status") != "frozen_before_one_minute_paths":
        raise ValueError("G4A parent freeze status changed")
    artifact = parent.get("artifacts", {}).get("frozen_sample_csv", {})
    path = Path(str(artifact.get("path", "")))
    if not path.is_file() or sha256_file(path) != str(artifact.get("sha256", "")):
        raise ValueError(f"G4A frozen sample is missing or changed: {path}")
    return parent, path


def request_contract(
    run_id: str,
    branch: dict[str, Any],
    parent: dict[str, Any],
    parent_sample_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "branch": branch,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "g5_batch": str(G5_BATCH),
        "g5_batch_sha256": sha256_file(G5_BATCH),
        "g4a_parent_record": str(G4A_FREEZE_RECORD),
        "g4a_parent_record_sha256": sha256_file(G4A_FREEZE_RECORD),
        "g4a_parent_request_sha256": parent["request_sha256"],
        "g4a_exposed_sample": str(parent_sample_path),
        "selection": {
            "seed": SELECTION_SEED,
            "cohort": "meme",
            "level_name": LEVEL_NAME,
            "approach_state": APPROACH_STATE,
            "periods": list(CONFIRMATION_PERIODS),
            "prior_outcome_path_separation_hours": PRIOR_PATH_SEPARATION_HOURS,
            "global_independence_hours": g3g.INDEPENDENCE_HOURS,
            "balance": "equal episode quota across the two frozen confirmation periods",
            "minimum_episodes": MINIMUM_EPISODES,
            "target_episodes": TARGET_EPISODES,
            "minimum_pairs": MINIMUM_PAIRS,
            "selection_columns": list(SELECTION_COLUMNS),
            "reaction_outcome_used": False,
            "direction_outcome_used": False,
            "contact_volume_used": False,
            "contact_range_used": False,
            "profit_used": False,
        },
        "reaction_outcomes_open_after_freeze_only": True,
        "direction_phase_open_only_after_reaction_gate": True,
    }


def selection_projection(raw: DataFrame) -> DataFrame:
    missing = sorted(set(SELECTION_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"G5A parent surface is missing selection columns: {missing}")
    view = raw.loc[:, list(SELECTION_COLUMNS)].copy()
    view["event_time"] = pd.to_datetime(view["event_time"], utc=True)
    view["source_available_at"] = pd.to_datetime(view["source_available_at"], utc=True)
    view["source_open"] = pd.to_datetime(view["source_open"], utc=True)
    view = view.loc[
        view["cohort"].eq("meme")
        & view["level_name"].eq(LEVEL_NAME)
        & view["approach_state"].eq(APPROACH_STATE)
        & view["period"].isin(CONFIRMATION_PERIODS)
    ].copy()
    if (view["source_available_at"] > view["event_time"]).any():
        raise ValueError("Future LVN source entered the G5A selection view")
    if view["episode_id"].duplicated().any():
        raise ValueError("G5A selection episode ids are not unique")
    view["source_g4a_selection_hash"] = view["g4a_selection_hash"].astype(str)
    view["g5a_selection_key"] = view["episode_id"].astype(str).map(
        lambda value: f"{SELECTION_SEED}|{value}"
    )
    view["g5a_selection_hash"] = view["g5a_selection_key"].map(sha256_text)
    return view.reset_index(drop=True)


def exclude_prior_outcome_paths(
    candidates: DataFrame,
    *,
    exposed_times: Series,
    hours: int,
) -> DataFrame:
    output = candidates.copy()
    distances = nearest_time_distance_minutes(output["event_time"], exposed_times)
    output["minutes_to_nearest_g4a_episode"] = distances
    return output.loc[distances.ge(hours * 60.0)].reset_index(drop=True)


def nearest_time_distance_minutes(events: Series, references: Series) -> Series:
    event_dates = pd.Series(pd.to_datetime(events, utc=True), index=events.index).dt.as_unit("ns")
    reference_dates = pd.Series(pd.to_datetime(references, utc=True)).dt.as_unit("ns")
    if event_dates.isna().any() or reference_dates.isna().any():
        raise ValueError("Missing timestamp in G5A independence audit")
    if reference_dates.empty:
        return Series(np.inf, index=events.index, dtype="float64")
    event_ns = event_dates.astype("int64").to_numpy()
    reference_ns = np.sort(reference_dates.astype("int64").to_numpy())
    positions = np.searchsorted(reference_ns, event_ns)
    left_index = np.clip(positions - 1, 0, len(reference_ns) - 1)
    right_index = np.clip(positions, 0, len(reference_ns) - 1)
    left = np.abs(event_ns - reference_ns[left_index])
    right = np.abs(event_ns - reference_ns[right_index])
    minutes = np.minimum(left, right) / (60.0 * 1_000_000_000.0)
    return Series(minutes, index=events.index, dtype="float64")


def select_globally_independent(candidates: DataFrame) -> DataFrame:
    if candidates.empty:
        return candidates.copy()
    source = candidates.copy()
    source["g4a_selection_hash"] = source["g5a_selection_hash"]
    selected = g4a_breadth.select_globally_independent_pool(source)
    selected["g4a_selection_hash"] = selected["source_g4a_selection_hash"]
    selected["g5a_global_independence_order"] = np.arange(1, len(selected) + 1)
    return selected.drop(columns="g4a_global_independence_order", errors="ignore").assign(
        g5a_global_independence_order=np.arange(1, len(selected) + 1)
    )


def select_balanced_confirmation_sample(candidates: DataFrame) -> DataFrame:
    if candidates.empty:
        return candidates.copy()
    counts = candidates.groupby("period", observed=True).size().to_dict()
    per_period = min(
        TARGET_EPISODES // len(CONFIRMATION_PERIODS),
        *(int(counts.get(period, 0)) for period in CONFIRMATION_PERIODS),
    )
    parts: list[DataFrame] = []
    for period in CONFIRMATION_PERIODS:
        cell = candidates.loc[candidates["period"].eq(period)].copy()
        cell = cell.sort_values(["pair", "g5a_selection_hash"], kind="mergesort")
        cell["_pair_rank"] = cell.groupby("pair", observed=True).cumcount()
        cell = cell.sort_values(["_pair_rank", "g5a_selection_hash"], kind="mergesort")
        parts.append(cell.head(per_period))
    if not parts:
        return candidates.iloc[0:0].copy()
    sample = pd.concat(parts, ignore_index=True, sort=False).drop(
        columns="_pair_rank", errors="ignore"
    )
    sample = sample.sort_values("g5a_selection_hash", kind="mergesort").reset_index(drop=True)
    sample["sample_selection_order"] = np.arange(1, len(sample) + 1)
    sample["selected_for_g5a_reaction_replication"] = True
    return sample


def support_decision(sample: DataFrame) -> dict[str, Any]:
    period_counts = {
        period: int(sample["period"].eq(period).sum()) for period in CONFIRMATION_PERIODS
    }
    pair_count = int(sample["pair"].nunique()) if not sample.empty else 0
    balanced = len(set(period_counts.values())) == 1 and all(period_counts.values())
    passed = (
        len(sample) >= MINIMUM_EPISODES
        and pair_count >= MINIMUM_PAIRS
        and balanced
        and set(sample["period"].astype(str)) == set(CONFIRMATION_PERIODS)
    )
    return {
        "passed": bool(passed),
        "episodes": len(sample),
        "minimum_episodes": MINIMUM_EPISODES,
        "target_episodes": TARGET_EPISODES,
        "pairs": pair_count,
        "minimum_pairs": MINIMUM_PAIRS,
        "period_counts": period_counts,
        "balanced_periods": bool(balanced),
        "classification": (
            "adequate_to_freeze_before_reaction_outcomes"
            if passed
            else "insufficient_independent_support_park_without_outcomes"
        ),
    }


def validate_frozen_sample(sample: DataFrame) -> None:
    decision = support_decision(sample)
    if not decision["passed"]:
        raise ValueError(f"Frozen G5A sample failed its support gate: {decision}")
    normalized_dates = pd.to_datetime(sample["event_time"], utc=True).dt.as_unit("ns")
    dates = np.sort(normalized_dates.astype("int64").to_numpy())
    if len(dates) > 1:
        minutes = np.diff(dates) / (60.0 * 1_000_000_000.0)
        if np.any(minutes < g3g.INDEPENDENCE_HOURS * 60.0):
            raise ValueError("Frozen G5A events are not globally 48-hour independent")
    if sample["minutes_to_nearest_g4a_episode"].lt(
        PRIOR_PATH_SEPARATION_HOURS * 60.0
    ).any():
        raise ValueError("Frozen G5A sample overlaps a previously opened G4A outcome path")
    forbidden = [
        column
        for column in sample.columns
        if column.startswith(
            ("contact_", "abs_excursion_", "away_excursion_", "through_excursion_")
        )
        or column.startswith(("range_ratio_h", "volume_ratio_h", "pressure_change_h"))
    ]
    if forbidden:
        raise ValueError(f"Outcome columns leaked into frozen G5A sample: {forbidden}")


def inventory_rows(frame: DataFrame, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period in (*CONFIRMATION_PERIODS, "all"):
        view = frame if period == "all" else frame.loc[frame["period"].eq(period)]
        rows.append(
            {
                "stage": stage,
                "period": period,
                "episodes": len(view),
                "pairs": int(view["pair"].nunique()) if not view.empty else 0,
                "first_event_utc": (
                    iso_or_none(view["event_time"].min()) if not view.empty else None
                ),
                "last_event_utc": iso_or_none(view["event_time"].max()) if not view.empty else None,
                "reaction_outcomes_opened": False,
            }
        )
    return rows


def stage_count_map(inventory: DataFrame) -> dict[str, int]:
    return {
        str(row["stage"]): int(row["episodes"])
        for _, row in inventory.loc[inventory["period"].eq("all")].iterrows()
    }


def iso_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
