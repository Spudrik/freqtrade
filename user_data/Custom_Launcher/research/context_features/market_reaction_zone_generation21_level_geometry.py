"""Run Generation 21 Volume Profile role and single/cluster siblings A and C."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_freeze as g21z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g21_level_geometry_20260827a"
DEFAULT_SUPPORT_ID = "g21_level_geometry_support_20260827a"
RECORD_ROOT = g21z.OUTPUT_ROOT / "level_geometry"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation21_branches"
    / "g21_broad_siblings"
    / "level_geometry_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g21_level_geometry_support_freeze.json"
ATLAS_ROOT = g17l.ARTIFACT_ROOT / g17l.DEFAULT_RUN_ID / "pair_events"
HORIZONS = (2, 4, 8)
CONTROLS = g21z.LOCATION_CONTROLS
PRE_WINDOW = 4
READ_COLUMNS = (
    *g20s.META_COLUMNS,
    "independent_family_count",
)
ROLE_SUFFIXES = {
    "point_of_control": ("_poc",),
    "value_area_boundary": ("_nearest_value_boundary",),
    "high_volume_node": ("_nearest_hvn_q80", "_nearest_hvn_q90"),
    "low_volume_node": ("_nearest_lvn_q10", "_nearest_lvn_q20"),
}


def artifact(path: Path) -> dict[str, Any]:
    return g21z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g21z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation21_outcomes":
        raise ValueError("Generation 21 batch is not frozen.")
    return frozen


def role_group(level_name: str) -> str | None:
    for group, suffixes in ROLE_SUFFIXES.items():
        if str(level_name).endswith(suffixes):
            return group
    return None


def geometry_state(value: object) -> str:
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return "unavailable"
    return (
        "isolated_single"
        if count == 1
        else "two_family_cluster"
        if count == 2
        else "three_plus_family_cluster"
        if count >= 3
        else "unavailable"
    )


def later_vp_rows(path: Path) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(READ_COLUMNS))
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    cohorts = frame["cohort"].dropna().astype(str).unique()
    if len(cohorts) != 1:
        raise ValueError(f"Unexpected cohorts in {path}: {cohorts}")
    frame["period"] = g18d.assign_confirmation_period(frame["event_time"], cohorts[0])
    frame = frame.loc[
        frame["period"].ne("outside_g18_confirmation")
        & frame["level_family"].eq("adaptive_volume_profile_nodes")
        & frame["control"].isin({"actual", *CONTROLS})
    ].copy()
    frame["role_group"] = frame["level_name"].map(role_group)
    frame["geometry_state"] = frame["independent_family_count"].map(geometry_state)
    return frame.loc[
        frame["role_group"].notna() & frame["geometry_state"].ne("unavailable")
    ].copy()


def scoped_support(frame: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "period", "control", "event_time"]
    role = g18d.nearest_anchor(frame, [*keys, "role_group"])
    role["branch_id"] = "g21a_btc_volume_profile_role_specificity"
    geometry = g18d.nearest_anchor(frame, [*keys, "geometry_state"])
    geometry["branch_id"] = "g21c_state_matched_single_and_cluster_levels"
    return pd.concat([role, geometry], ignore_index=True, sort=False)


def build_pair_support(path: Path, overwrite: bool) -> dict[str, Any]:
    raw = later_vp_rows(path)
    pair = str(raw["pair"].iloc[0])
    cohort = str(raw["cohort"].iloc[0])
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["branch_id"])
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "existing",
        }
    base, state = g20s.base_and_state(pair, cohort)
    state, calibration = g20s.add_activity_states(state, cohort)
    support = scoped_support(raw)
    g20s.attach_causal_state(support, base, state)
    support.sort_values(
        ["branch_id", "control", "event_time", "level_name"],
        inplace=True,
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "roles": sorted(support["role_group"].dropna().unique()),
        "geometries": sorted(support["geometry_state"].dropna().unique()),
        "activity_calibration": calibration,
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation21_level_outcomes":
            raise ValueError("Invalid Generation 21 level support freeze.")
        return manifest
    paths = sorted(ATLAS_ROOT.glob("*.parquet"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 atlas files, got {len(paths)}.")
    inventory = []
    for number, path in enumerate(paths, start=1):
        inventory.append(build_pair_support(path, overwrite))
        print(
            json.dumps(
                {"phase": "g21_level_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_level_outcomes",
        "branches": [
            "g21a_btc_volume_profile_role_specificity",
            "g21c_state_matched_single_and_cluster_levels",
        ],
        "future_outcomes_opened": False,
        "role_groups": {key: list(value) for key, value in ROLE_SUFFIXES.items()},
        "geometry_states": [
            "isolated_single",
            "two_family_cluster",
            "three_plus_family_cluster",
        ],
        "inventory": inventory,
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def compare_scopes(events: DataFrame) -> DataFrame:
    targets = [
        (metric, horizon, f"metric__{metric}_h{horizon}")
        for metric in ("any_recross", "repeated_recross", "crossing_count")
        for horizon in HORIZONS
    ]
    state_columns = (
        f"pre_crossings_h{PRE_WINDOW}",
        "relative_volume",
        f"pre_range_over_atr_h{PRE_WINDOW}",
        f"pre_abs_return_over_atr_h{PRE_WINDOW}",
    )
    rows: list[dict[str, Any]] = []
    a = events.loc[
        events["branch_id"].eq("g21a_btc_volume_profile_role_specificity")
    ]
    for role in ROLE_SUFFIXES:
        rows.extend(
            g20s.compare_cell(
                a.loc[a["role_group"].eq(role)],
                comparison_names=CONTROLS,
                state_columns=state_columns,
                targets=targets,
                dimensions={
                    "branch_id": "g21a_btc_volume_profile_role_specificity",
                    "scope_kind": "role_group",
                    "scope_value": role,
                },
            )
        )
    c = events.loc[
        events["branch_id"].eq("g21c_state_matched_single_and_cluster_levels")
    ]
    for geometry in (
        "isolated_single",
        "two_family_cluster",
        "three_plus_family_cluster",
    ):
        rows.extend(
            g20s.compare_cell(
                c.loc[c["geometry_state"].eq(geometry)],
                comparison_names=CONTROLS,
                state_columns=state_columns,
                targets=targets,
                dimensions={
                    "branch_id": "g21c_state_matched_single_and_cluster_levels",
                    "scope_kind": "geometry_state",
                    "scope_value": geometry,
                },
            )
        )
    return DataFrame.from_records(rows)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    if g0.sha256_file(ANALYSIS_PATH) != manifest["source_contracts"]["analysis_script"][
        "sha256"
    ]:
        raise ValueError("Generation 21 level analysis changed after support freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g21_level_geometry_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen Generation 21 level support changed: {path}")
        frame = pd.read_parquet(path)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(frame, base)
        parts.append(frame)
        print(
            json.dumps(
                {"phase": "g21_level_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    contrasts = compare_scopes(pd.concat(parts, ignore_index=True, sort=False))
    if contrasts.empty:
        raise ValueError("No Generation 21 level contrasts were produced.")
    question_keys = ("branch_id", "scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g21_level_pair_contrasts.csv",
        "scores": run_dir / "g21_level_period_scores.csv",
        "decisions": run_dir / "g21_level_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation21_level_geometry",
        "branches_completed": [
            "g21a_btc_volume_profile_role_specificity",
            "g21c_state_matched_single_and_cluster_levels",
        ],
        "strict_rows": int(
            decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "point_rows": int(
            decisions["status"].eq("point_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
        },
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--prepare-support", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_support:
        print(json.dumps(freeze_support(overwrite=args.overwrite), indent=2))
        return 0
    return execute(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
