"""Attribute the fixed Generation 22 multitimeframe cluster near-miss."""

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

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_multitimeframe_convergence as g22b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_common as g23c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g23_multitimeframe_attribution_20260828a"
DEFAULT_SUPPORT_ID = "g23_multitimeframe_attribution_support_20260828a"
RECORD_ROOT = g23z.OUTPUT_ROOT / "multitimeframe_attribution"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "multitimeframe_attribution_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g23_multitimeframe_attribution_support_freeze.json"
GEOMETRY = "three_plus_timeframe_cluster"
ATTRIBUTION_COMPARISONS = (
    "matched_isolated_anchor_component",
    "highest_timeframe_component",
    "nearest_non_anchor_component",
    "same_time_current_close_pseudo_cluster",
)
PARENT_CONTROLS = g23z.LEVEL_CONTROLS
CONTROLS = (*PARENT_CONTROLS, *ATTRIBUTION_COMPARISONS)
CLUSTER_RADIUS_ATR = 0.50
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g23c.load_branch(
        "g23a_multitimeframe_cluster_incremental_attribution"
    )
    if tuple(branch["attribution_comparisons"]) != ATTRIBUTION_COMPARISONS:
        raise ValueError("Generation 23 cluster attribution registry drifted.")
    if tuple(branch["parent_controls"]) != PARENT_CONTROLS:
        raise ValueError("Generation 23 parent control registry drifted.")
    if float(branch["cluster_radius_atr"]) != CLUSTER_RADIUS_ATR:
        raise ValueError("Generation 23 cluster radius drifted.")
    return frozen


def load_parent_support() -> dict[str, Any]:
    manifest = json.loads(g22b.SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_generation22_multitimeframe_outcomes":
        raise ValueError("Generation 22 multitimeframe support is not frozen.")
    return manifest


def approach_state(pre_close: float, level: float, width: float) -> str:
    if pre_close < level - width:
        return "from_below"
    if pre_close > level + width:
        return "from_above"
    return "already_inside_or_unclear"


def component_record(
    row: pd.Series,
    *,
    base: DataFrame,
    surface: dict[str, Any] | None,
    comparison: str,
) -> dict[str, Any]:
    index = int(row["base_index"])
    atr = float(base["base_atr"].iloc[index])
    pre_close = float(base["pre_close"].iloc[index])
    if surface is None:
        level = float(base["close"].iloc[index])
        source_open = pd.to_datetime(base["date"].iloc[index], utc=True)
        name = "same_time_current_close_pseudo_cluster"
        family = "artificial_current_price"
        role = "pseudo_cluster_centre"
        timeframe = "artificial"
    else:
        level = float(surface["level"][index])
        source_open = pd.to_datetime(surface["source_open"].iloc[index], utc=True)
        name = str(surface["level_name"])
        family = str(surface["family"])
        role = str(surface["role"])
        timeframe = str(surface["source_timeframe"])
    record = row.to_dict()
    record.update(
        {
            "control": comparison,
            "level_price": level,
            "level_name": name,
            "source_open": source_open,
            "anchor_source_family": family,
            "anchor_role": role,
            "anchor_source_timeframe": timeframe,
            "pre_distance_atr": abs(level - pre_close) / atr,
            "approach_state": approach_state(
                pre_close, level, ZONE_HALF_WIDTH_ATR * atr
            ),
            "attribution_comparison": comparison,
        }
    )
    return record


def same_time_component_controls(
    actual_clusters: DataFrame,
    base: DataFrame,
    surfaces: list[dict[str, Any]],
) -> DataFrame:
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for _, row in actual_clusters.iterrows():
        index = int(row["base_index"])
        anchor = float(row["level_price"])
        candidates = [
            surface
            for surface in surfaces
            if np.isfinite(surface["level"][index])
            and abs(float(surface["level"][index]) - anchor)
            <= CLUSTER_RADIUS_ATR * atr[index]
        ]
        if len({surface["source_timeframe"] for surface in candidates}) < 3:
            continue
        highest_hours = max(int(surface["timeframe_hours"]) for surface in candidates)
        highest = min(
            (
                surface
                for surface in candidates
                if int(surface["timeframe_hours"]) == highest_hours
            ),
            key=lambda surface: abs(float(surface["level"][index]) - anchor),
        )
        non_anchor = [
            surface
            for surface in candidates
            if str(surface["level_name"]) != str(row["level_name"])
        ]
        if not non_anchor:
            continue
        nearest_other = min(
            non_anchor,
            key=lambda surface: abs(float(surface["level"][index]) - anchor),
        )
        records.append(
            component_record(
                row,
                base=base,
                surface=highest,
                comparison="highest_timeframe_component",
            )
        )
        records.append(
            component_record(
                row,
                base=base,
                surface=nearest_other,
                comparison="nearest_non_anchor_component",
            )
        )
        records.append(
            component_record(
                row,
                base=base,
                surface=None,
                comparison="same_time_current_close_pseudo_cluster",
            )
        )
    return DataFrame.from_records(records)


def matched_isolated_anchor_controls(
    actual_clusters: DataFrame, isolated: DataFrame, pair: str
) -> DataFrame:
    keys = [
        "period",
        "anchor_source_family",
        "anchor_role",
        "anchor_source_timeframe",
    ]
    parts: list[DataFrame] = []
    for identity, clusters in actual_clusters.groupby(keys, observed=True, sort=False):
        mask = np.ones(len(isolated), dtype=bool)
        for key, value in zip(keys, identity, strict=True):
            mask &= isolated[key].astype(str).eq(str(value)).to_numpy()
        candidates = isolated.loc[mask].copy()
        if candidates.empty:
            continue
        candidates["_order"] = [
            g0.stable_hash_int(
                f"g23-isolated-anchor|{pair}|{identity}|{pd.Timestamp(value).isoformat()}"
            )
            for value in candidates["event_time"]
        ]
        selected = candidates.sort_values("_order").head(len(clusters)).drop(
            columns="_order"
        )
        selected["control"] = "matched_isolated_anchor_component"
        selected["attribution_comparison"] = "matched_isolated_anchor_component"
        parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else DataFrame()


def pair_support(
    item: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]:
    pair = str(item["pair"])
    cohort = str(item["cohort"])
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control"])
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "existing",
        }
    parent_path = Path(item["path"])
    if g0.sha256_file(parent_path) != item["sha256"]:
        raise ValueError(f"Frozen Generation 22 multitimeframe support changed: {parent_path}")
    parent = pd.read_parquet(parent_path)
    parent["event_time"] = pd.to_datetime(parent["event_time"], utc=True)
    base, _ = g20s.base_and_state(pair, cohort)
    actual_clusters = parent.loc[
        parent["control"].eq("actual") & parent["geometry_state"].eq(GEOMETRY)
    ].copy()
    isolated = parent.loc[
        parent["control"].eq("actual")
        & parent["geometry_state"].eq("isolated_single_timeframe")
    ].copy()
    parent_ladder = parent.loc[
        parent["geometry_state"].eq(GEOMETRY)
        & parent["control"].isin(("actual", *PARENT_CONTROLS))
    ].copy()
    surfaces = g22b.multitimeframe_surfaces(base)
    same_time = same_time_component_controls(actual_clusters, base, surfaces)
    isolated_controls = matched_isolated_anchor_controls(actual_clusters, isolated, pair)
    parts = [parent_ladder, same_time, isolated_controls]
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support["attribution_comparison"] = support["control"].where(
        support["control"].ne("actual"), "actual_three_plus_cluster"
    )
    support.sort_values(
        ["control", "period", "event_time", "pre_distance_atr"],
        inplace=True,
        kind="stable",
    )
    support = support.drop_duplicates(["control", "event_time"], keep="first").reset_index(
        drop=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "control_counts": support["control"].value_counts().to_dict(),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    parent = load_parent_support()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        expected = "frozen_before_generation23_multitimeframe_attribution_outcomes"
        if manifest.get("status") != expected:
            raise ValueError("Invalid Generation 23 multitimeframe attribution freeze.")
        return manifest
    inventory = []
    for number, item in enumerate(parent["inventory"], start=1):
        inventory.append(pair_support(item, overwrite=overwrite))
        print(
            json.dumps(
                {"phase": "g23_mtf_attribution_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_multitimeframe_attribution_outcomes",
        "branch_id": "g23a_multitimeframe_cluster_incremental_attribution",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "geometry": GEOMETRY,
        "controls": list(CONTROLS),
        "same_holdout_attribution_only": True,
        "inventory": inventory,
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation23_common": artifact(g23c.ANALYSIS_PATH),
            "generation22_multitimeframe_builder": artifact(g22b.ANALYSIS_PATH),
            "generation22_multitimeframe_support": artifact(g22b.SUPPORT_MANIFEST),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    output = events.copy()
    output["scope_kind"] = "cluster_attribution"
    output["scope_value"] = GEOMETRY
    return output


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch_supports = g23c.require_all_frozen_supports()
    for path, key in (
        (ANALYSIS_PATH, "analysis_script"),
        (g23c.ANALYSIS_PATH, "generation23_common"),
        (g22b.ANALYSIS_PATH, "generation22_multitimeframe_builder"),
        (g22b.SUPPORT_MANIFEST, "generation22_multitimeframe_support"),
    ):
        if g0.sha256_file(path) != manifest["source_contracts"][key]["sha256"]:
            raise ValueError(f"Frozen multitimeframe attribution dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g23_multitimeframe_attribution_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    events = g23c.open_frozen_support_outcomes(
        manifest, phase="g23_mtf_attribution_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g23c.score_control_ladders(
        events, controls=CONTROLS
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g23c.write_scored_tables(
        run_dir,
        prefix="g23_multitimeframe_attribution",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_multitimeframe_attribution",
        "branch_completed": "g23a_multitimeframe_cluster_incremental_attribution",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "same_holdout_attribution_only": True,
            "untouched_confirmation_claimed": False,
        },
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation23_sibling_supports": batch_supports,
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
