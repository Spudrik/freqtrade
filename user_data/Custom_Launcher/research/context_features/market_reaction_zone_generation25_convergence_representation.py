"""Repair convergence controls by pooling all frozen definitions before outcomes."""

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
    market_reaction_zone_generation24_round_distribution_convergence as g24r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_common as g25c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g25b_convergence_control_representation_repair"
DEFAULT_SUPPORT_ID = "g25_convergence_representation_support_20260828a"
DEFAULT_RUN_ID = "g25_convergence_representation_20260828a"
RECORD_ROOT = g25z.OUTPUT_ROOT / "convergence_representation"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
        "generation25_branches/g25_broad_siblings/convergence_representation"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g25_convergence_representation_support_freeze.json"
CONTROLS = tuple(g24r.CONTROLS)
SOURCE_SURFACE_COUNT = (
    len(g24r.STEP_MULTIPLIERS) * len(g24r.LOOKBACKS) * len(g24r.QUANTILES)
)
MIN_REPRESENTATION_ROWS = 10


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def _spread_positions(length: int, wanted: int) -> np.ndarray:
    if wanted <= 0 or length <= 0:
        return np.array([], dtype=np.int64)
    if wanted >= length:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, wanted)).astype(np.int64))


def _deduplicate_pool(frame: DataFrame) -> DataFrame:
    """Collapse repeated definitions at one timestamp without using outcomes."""
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    counts = (
        output.groupby(["control", "period", "event_time"], observed=True)["level_name"]
        .transform("nunique")
        .astype(int)
    )
    output["source_definition_count"] = counts
    output.sort_values(
        ["control", "period", "event_time", "pre_distance_atr", "level_name"],
        inplace=True,
        kind="stable",
    )
    return output.drop_duplicates(["control", "period", "event_time"], keep="first")


def _geometry_match(reference: DataFrame, candidates: DataFrame, wanted: int) -> DataFrame:
    """Select unique controls nearest to frozen approach geometry."""
    if wanted <= 0 or reference.empty or candidates.empty:
        return DataFrame(columns=candidates.columns)
    available = candidates.sort_values(["event_time", "pre_distance_atr"], kind="stable").copy()
    chosen: list[int] = []
    for _, row in reference.iloc[:wanted].iterrows():
        remaining = available.loc[~available.index.isin(chosen)]
        if remaining.empty:
            break
        same_approach = remaining.loc[
            remaining["approach_state"].astype(str).eq(str(row["approach_state"]))
        ]
        pool = same_approach if not same_approach.empty else remaining
        distances = (
            pd.to_numeric(pool["pre_distance_atr"], errors="coerce")
            - float(row["pre_distance_atr"])
        ).abs()
        chosen.append(int(distances.sort_values(kind="stable").index[0]))
    return available.loc[chosen].copy()


def equal_support_surface(source: DataFrame) -> tuple[DataFrame, list[dict[str, Any]]]:
    """Return one equal-density pooled ladder per period for a single pair."""
    pooled = _deduplicate_pool(source)
    selected_parts: list[DataFrame] = []
    audits: list[dict[str, Any]] = []
    expected = ("actual", *CONTROLS)
    for period, period_frame in pooled.groupby("period", observed=True, sort=False):
        pools = {
            control: period_frame.loc[period_frame["control"].eq(control)].copy()
            for control in expected
        }
        counts = {control: len(frame) for control, frame in pools.items()}
        common_count = min(counts.values()) if counts else 0
        actual_pool = pools["actual"].sort_values("event_time", kind="stable")
        actual = actual_pool.iloc[_spread_positions(len(actual_pool), common_count)].copy()
        selected_parts.append(actual)
        matched_counts = {"actual": len(actual)}
        for control in CONTROLS:
            matched = _geometry_match(actual, pools[control], common_count)
            matched["control"] = control
            selected_parts.append(matched)
            matched_counts[control] = len(matched)
        audits.append(
            {
                "period": str(period),
                "source_counts": counts,
                "common_count": common_count,
                "matched_counts": matched_counts,
                "representation_gate_pass": bool(
                    common_count >= MIN_REPRESENTATION_ROWS
                    and set(matched_counts.values()) == {common_count}
                ),
            }
        )
    available = [part for part in selected_parts if not part.empty]
    if not available:
        return DataFrame(), audits
    output = pd.concat(available, ignore_index=True, sort=False)
    output["scope_kind"] = "round_distribution_convergence_pooled"
    output["scope_value"] = "all_60_frozen_definitions"
    output["control_match_method"] = "same_period_approach_and_distance_equal_density"
    output.sort_values(["period", "control", "event_time"], inplace=True, kind="stable")
    output.reset_index(drop=True, inplace=True)
    return output, audits


def _source_manifest() -> dict[str, Any]:
    manifest = json.loads(g24r.SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_generation24_convergence_outcomes":
        raise ValueError("Generation 24 convergence support is unavailable.")
    if int(manifest.get("combination_count_per_pair", -1)) != SOURCE_SURFACE_COUNT:
        raise ValueError("Generation 24 convergence surface count drifted.")
    return manifest


def pair_support(item: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    pair = str(item["pair"])
    cohort = str(item["cohort"])
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    audit_path = output.with_suffix(".support.json")
    if output.is_file() and audit_path.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control", "period"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "control_counts": existing["control"].value_counts().to_dict(),
            "representation_gate_pass": bool(audit["representation_gate_pass"]),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "audit": artifact(audit_path),
            "status": "existing",
        }
    source_path = Path(item["path"])
    if g0.sha256_file(source_path) != item["sha256"]:
        raise ValueError(f"Generation 24 convergence support changed: {source_path}")
    source = pd.read_parquet(source_path)
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True)
    support, period_audits = equal_support_surface(source)
    if support.empty:
        raise ValueError(f"No pooled Generation 25 convergence support for {pair}.")
    passed = bool(period_audits and all(row["representation_gate_pass"] for row in period_audits))
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    audit = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "pair": pair,
        "cohort": cohort,
        "source_surface_count": SOURCE_SURFACE_COUNT,
        "periods": period_audits,
        "representation_gate_pass": passed,
        "future_outcome_values_read": False,
    }
    g0.atomic_write_json(audit, audit_path)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "control_counts": support["control"].value_counts().to_dict(),
        "representation_gate_pass": passed,
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "audit": artifact(audit_path),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    g25c.load_branch(BRANCH_ID)
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation25_convergence_outcomes":
            raise ValueError("Invalid Generation 25 convergence support.")
        return manifest
    parent = _source_manifest()
    inventory: list[dict[str, Any]] = []
    for number, item in enumerate(parent["inventory"], start=1):
        inventory.append(pair_support(item, overwrite=overwrite))
        print(
            json.dumps({"phase": "g25_convergence_support", "processed": number, "total": 20}),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation25_convergence_outcomes",
        "branch_id": BRANCH_ID,
        "source_surface_count_per_pair": SOURCE_SURFACE_COUNT,
        "entire_frozen_family_pooled_without_winner_filtering": True,
        "controls": list(CONTROLS),
        "inventory": inventory,
        "pairs_passing_representation_gate": sum(
            bool(item["representation_gate_pass"]) for item in inventory
        ),
        "future_outcome_values_read": False,
        "source_contracts": {
            "generation25_freeze": artifact(g25z.FREEZE_PATH),
            "generation24_convergence_support": artifact(g24r.SUPPORT_MANIFEST),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    sibling_supports = g25c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g25_convergence_representation_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    events = g25c.open_direct_support_outcomes(
        manifest,
        phase="g25_convergence_outcomes",
    )
    contrasts, scores, decisions = g25c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g25c.write_scored_tables(
        run_dir,
        prefix="g25_convergence_representation",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation25_convergence_representation",
        "representation_pairs_passed": manifest["pairs_passing_representation_gate"],
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "winner_filtering_used": False,
            "same_holdout_exploratory": True,
        },
        "source_contracts": {
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation25_sibling_supports": sibling_supports,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
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
