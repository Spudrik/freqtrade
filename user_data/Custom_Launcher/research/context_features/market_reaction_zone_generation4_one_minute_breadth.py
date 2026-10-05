from __future__ import annotations

# Selection and coverage auditing are single-threaded. Downloads are launched separately
# through the existing Freqtrade controller environment.
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
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_one_minute_replay import (  # noqa: E501
    INDEPENDENCE_HOURS,
    LEVEL_NAMES,
    WINDOW_HOURS,
    acquisition_intervals,
    artifact_record,
    audit_one_minute_coverage,
    nested_counts,
    sort_component_rows,
    sort_episode_rows,
)


SCHEMA_VERSION = 1
SELECTION_SEED = "g4a_thin_lvn_breadth_v1"
MINIMUM_EPISODES_PER_COHORT = 30
TARGET_EPISODES_PER_COHORT = 60
MINIMUM_PAIRS_PER_COHORT = 8
PARENT_RUN_ID = "g3g_thin_lvn_1m_freeze_20260814a"
G4_BATCH = OUTPUT_ROOT / "generation3_review" / "g4_frozen_branch_batch.json"
PARENT_RECORD = (
    OUTPUT_ROOT
    / "generation3_branches"
    / "g3g_thin_lvn_one_minute_replay"
    / PARENT_RUN_ID
    / "g3g_freeze_record.json"
)
REPORT_ROOT = (
    OUTPUT_ROOT
    / "generation4_branches"
    / "g4a_thin_lvn_one_minute_breadth_confirmation"
)
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT
    / "generation4_branches"
    / "g4a_thin_lvn_one_minute_breadth_confirmation"
)
PERIODS = {
    "normal": ("development", "validation_early", "validation_late"),
    "meme": (
        "meme_development",
        "meme_validation_early",
        "meme_validation_late",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the outcome-blind G4A breadth-confirmation sample and targeted "
            "one-minute acquisition intervals, or audit the frozen local coverage."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--coverage-audit", action="store_true")
    args = parser.parse_args(argv)

    validate_g4_branch()
    if args.coverage_audit:
        return audit_one_minute_coverage(
            args.run_id,
            report_root=REPORT_ROOT,
            record_filename="g4a_freeze_record.json",
            coverage_prefix="g4a",
        )
    return freeze_sample(args.run_id, overwrite=bool(args.overwrite))


def validate_g4_branch() -> dict[str, Any]:
    batch = json.loads(G4_BATCH.read_text(encoding="utf-8"))
    if batch.get("status") != "frozen_before_generation4_reaction_outcomes":
        raise ValueError(f"Generation 4 batch is not frozen: {G4_BATCH}")
    branch = next(
        (
            item
            for item in batch.get("branches", [])
            if item.get("id") == "g4a_thin_lvn_one_minute_breadth_confirmation"
        ),
        None,
    )
    if branch is None or branch.get("status") != "frozen_next_batch":
        raise ValueError("The frozen Generation 4 batch does not authorize G4A")
    return branch


def freeze_sample(run_id: str, *, overwrite: bool) -> int:
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g4a_freeze_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"Run already exists: {record_path}")

    parent, pool_path, components_path = validated_parent_artifacts()
    request = request_contract(run_id, parent, pool_path, components_path)
    request_sha256 = stable_json_sha256(request)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "freezing",
        "started_at_utc": utc_now(),
        "request_sha256": request_sha256,
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        pool = pd.read_csv(pool_path)
        pool["event_time"] = pd.to_datetime(pool["event_time"], utc=True, errors="coerce")
        if pool["event_time"].isna().any():
            raise ValueError("Parent pool contains invalid event timestamps")
        pool = add_g4_selection_hash(pool)
        globally_independent = select_globally_independent_pool(pool)
        sample = select_breadth_sample(globally_independent)
        validate_sample(sample)

        components = pd.read_csv(components_path)
        sample_components = components.loc[
            components["episode_id"].astype(str).isin(set(sample["episode_id"].astype(str)))
        ].copy()
        if set(sample["episode_id"].astype(str)) != set(
            sample_components["episode_id"].astype(str)
        ):
            raise ValueError("At least one G4A episode lacks frozen cluster components")
        acquisition = acquisition_intervals(sample)

        sample_path = run_dir / "g4a_frozen_episode_sample.csv"
        components_output = run_dir / "g4a_frozen_sample_components.csv"
        acquisition_path = run_dir / "g4a_one_minute_acquisition_intervals.csv"
        candidate_path = artifact_dir / "g4a_globally_independent_candidate_pool.csv"
        atomic_write_csv(sort_episode_rows(sample), sample_path)
        atomic_write_csv(sort_component_rows(sample_components), components_output)
        atomic_write_csv(
            acquisition.sort_values(["pair", "interval_start_utc"]), acquisition_path
        )
        atomic_write_csv(sort_episode_rows(globally_independent), candidate_path)

        duplicate_bridge = cross_cohort_bridge_duplicates(sample)
        record.update(
            {
                "status": "frozen_before_one_minute_paths",
                "completed_at_utc": utc_now(),
                "parent_independent_episode_counts": count_by(pool, "cohort"),
                "global_48h_candidate_counts": count_by(globally_independent, "cohort"),
                "sample_episode_counts": count_by(sample, "cohort"),
                "sample_pair_counts": {
                    str(key): int(value)
                    for key, value in sample.groupby("cohort")["pair"].nunique().items()
                },
                "sample_period_counts": nested_counts(sample, ["cohort", "period"]),
                "sample_side_counts": nested_counts(sample, ["cohort", "level_name"]),
                "sample_anchor_counts": nested_counts(
                    sample, ["cohort", "cluster_causal_anchor_timeframe"]
                ),
                "cross_cohort_doge_bridge_duplicate_events": duplicate_bridge,
                "acquisition_intervals": len(acquisition),
                "expected_one_minute_rows_before_file_overlap": int(
                    acquisition["expected_one_minute_rows"].sum()
                ),
                "artifacts": {
                    "frozen_sample_csv": artifact_record(sample_path),
                    "frozen_sample_components_csv": artifact_record(components_output),
                    "acquisition_intervals_csv": artifact_record(acquisition_path),
                    "globally_independent_candidate_pool_csv": artifact_record(candidate_path),
                },
            }
        )
        atomic_write_json(record, record_path)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise


def validated_parent_artifacts() -> tuple[dict[str, Any], Path, Path]:
    parent = json.loads(PARENT_RECORD.read_text(encoding="utf-8"))
    if parent.get("status") != "frozen_before_one_minute_paths":
        raise ValueError(f"Parent G3G freeze is not valid: {PARENT_RECORD}")
    artifacts = parent.get("artifacts", {})
    pool_path = validated_artifact_path(artifacts, "complete_episode_pool_csv")
    components_path = validated_artifact_path(
        artifacts, "complete_cluster_components_csv"
    )
    return parent, pool_path, components_path


def validated_artifact_path(artifacts: dict[str, Any], name: str) -> Path:
    artifact = artifacts.get(name, {})
    path = Path(str(artifact.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Missing parent artifact {name}: {path}")
    if sha256_file(path) != str(artifact.get("sha256", "")):
        raise ValueError(f"Parent artifact changed after freeze: {path}")
    return path


def request_contract(
    run_id: str,
    parent: dict[str, Any],
    pool_path: Path,
    components_path: Path,
) -> dict[str, Any]:
    branch = validate_g4_branch()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "branch": branch,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "g4_batch_sha256": sha256_file(G4_BATCH),
        "parent_record_sha256": sha256_file(PARENT_RECORD),
        "parent_request_sha256": parent["request_sha256"],
        "parent_pool": str(pool_path),
        "parent_components": str(components_path),
        "selection": {
            "selection_seed": SELECTION_SEED,
            "future_signed_path_used": False,
            "global_independence_hours": INDEPENDENCE_HOURS,
            "global_independence_scope": "within cohort across every pair",
            "minimum_episodes_per_cohort": MINIMUM_EPISODES_PER_COHORT,
            "target_episodes_per_cohort": TARGET_EPISODES_PER_COHORT,
            "minimum_pairs_per_cohort": MINIMUM_PAIRS_PER_COHORT,
            "normal_stratification": (
                "equal quota across three periods and two LVN sides, with one episode "
                "per pair before a second episode in each cell"
            ),
            "meme_shortfall_rule": (
                "retain every globally independent candidate when fewer than the target "
                "sixty remain, provided the minimum thirty and eight pairs survive"
            ),
        },
        "one_minute_windows": {
            timeframe: {"pre_hours": values[0], "post_hours": values[1]}
            for timeframe, values in WINDOW_HOURS.items()
        },
        "boundary_rule": (
            "Start with the source-scaled default windows. Expand only episodes whose "
            "outcome-blind boundary audit shows continuing zone, volume, or volatility "
            "activity; do not automatically double every window."
        ),
        "direction_prediction": "immediate through-versus-away only after selection",
        "profit_optimization": False,
    }


def add_g4_selection_hash(pool: DataFrame) -> DataFrame:
    output = pool.copy()
    output["g4a_selection_hash"] = output["episode_id"].astype(str).map(
        lambda value: hashlib.sha256(
            f"{SELECTION_SEED}|{value}".encode()
        ).hexdigest()
    )
    return output


def select_globally_independent_pool(pool: DataFrame) -> DataFrame:
    rows: list[pd.Series] = []
    for cohort, group in pool.groupby("cohort", sort=True, observed=True):
        kept_times: list[pd.Timestamp] = []
        for _, row in group.sort_values("g4a_selection_hash", kind="mergesort").iterrows():
            event_time = pd.Timestamp(row["event_time"])
            if any(
                abs((event_time - previous).total_seconds())
                < INDEPENDENCE_HOURS * 3600.0
                for previous in kept_times
            ):
                continue
            rows.append(row)
            kept_times.append(event_time)
    output = DataFrame(rows).reset_index(drop=True)
    output["g4a_global_independence_order"] = (
        output.groupby("cohort", observed=True).cumcount() + 1
    )
    return output


def select_breadth_sample(candidates: DataFrame) -> DataFrame:
    parts: list[DataFrame] = []
    for cohort, periods in PERIODS.items():
        source = candidates.loc[candidates["cohort"].eq(cohort)].copy()
        if len(source) <= TARGET_EPISODES_PER_COHORT:
            chosen = source
        else:
            cells = [(period, level_name) for period in periods for level_name in LEVEL_NAMES]
            quotas = balanced_cell_quotas(
                source,
                cells=cells,
                target=TARGET_EPISODES_PER_COHORT,
            )
            chosen_parts: list[DataFrame] = []
            for period, level_name in cells:
                cell = source.loc[
                    source["period"].eq(period)
                    & source["level_name"].eq(level_name)
                ].copy()
                cell["_pair_rank"] = cell.groupby("pair", observed=True).cumcount()
                cell = cell.sort_values(
                    ["_pair_rank", "g4a_selection_hash"], kind="mergesort"
                )
                chosen_parts.append(cell.head(quotas[(period, level_name)]))
            chosen = pd.concat(chosen_parts, ignore_index=True, sort=False)
        parts.append(chosen)
    sample = pd.concat(parts, ignore_index=True, sort=False)
    sample = sample.sort_values(
        ["cohort", "g4a_selection_hash"], kind="mergesort"
    ).reset_index(drop=True)
    sample["sample_selection_order"] = np.arange(len(sample), dtype=np.int64)
    sample["selected_for_one_minute_replay"] = True
    sample["sample_stratum"] = (
        sample["cohort"].astype(str)
        + "|"
        + sample["period"].astype(str)
        + "|"
        + sample["level_name"].astype(str)
    )
    return sample.drop(columns=["_pair_rank"], errors="ignore")


def balanced_cell_quotas(
    source: DataFrame,
    *,
    cells: list[tuple[str, str]],
    target: int,
) -> dict[tuple[str, str], int]:
    available = {
        cell: int(
            (
                source["period"].eq(cell[0])
                & source["level_name"].eq(cell[1])
            ).sum()
        )
        for cell in cells
    }
    base = target // len(cells)
    quotas = {cell: min(base, available[cell]) for cell in cells}
    remaining = target - sum(quotas.values())
    while remaining > 0:
        eligible = [cell for cell in cells if quotas[cell] < available[cell]]
        if not eligible:
            break
        eligible.sort(key=lambda cell: (-(available[cell] - quotas[cell]), cell))
        for cell in eligible:
            if remaining <= 0:
                break
            quotas[cell] += 1
            remaining -= 1
    if sum(quotas.values()) < min(target, len(source)):
        raise ValueError("Could not allocate the frozen G4A cell quotas")
    return quotas


def validate_sample(sample: DataFrame) -> None:
    for cohort, periods in PERIODS.items():
        source = sample.loc[sample["cohort"].eq(cohort)]
        if len(source) < MINIMUM_EPISODES_PER_COHORT:
            raise ValueError(f"{cohort} has fewer than thirty G4A episodes")
        if source["pair"].nunique() < MINIMUM_PAIRS_PER_COHORT:
            raise ValueError(f"{cohort} has fewer than eight G4A pairs")
        for period in periods:
            for level_name in LEVEL_NAMES:
                cell = source.loc[
                    source["period"].eq(period)
                    & source["level_name"].eq(level_name)
                ]
                if cell.empty:
                    raise ValueError(f"Empty G4A sample cell: {cohort} {period} {level_name}")
        times = source["event_time"].sort_values()
        if times.diff().dropna().lt(pd.Timedelta(hours=INDEPENDENCE_HOURS)).any():
            raise ValueError(f"{cohort} G4A sample is not globally 48h independent")
    if sample["episode_id"].duplicated().any():
        raise ValueError("G4A sample contains duplicate episode IDs")


def cross_cohort_bridge_duplicates(sample: DataFrame) -> int:
    doge = sample.loc[sample["pair"].eq("DOGE/USDT:USDT")].copy()
    if doge.empty:
        return 0
    return int(doge.duplicated(["pair", "event_time", "level_name"], keep=False).sum() // 2)


def count_by(frame: DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
