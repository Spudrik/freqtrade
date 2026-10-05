from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)


OUTPUT_SCHEMA_VERSION = 1
FROZEN_BATCH = OUTPUT_ROOT / "generation3_review" / "g4_frozen_branch_batch.json"
REPORT_ROOT = (
    OUTPUT_ROOT
    / "generation4_branches"
    / "g4b_fresh_arrival_location_attribution"
)
PREFLIGHT_ROOT = (
    OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival"
)
CONTROL_ROOT = (
    OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival_controls"
)
STATE_ROOT = (
    OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival_states"
)

DENSITY_FAMILY = "confirmed_swing_price_density"
HISTORY_HOURS = 168
MIN_EVENTS_OR_PAIRS = 50
MIN_COINS = 5
RESPONSE_WINDOW = "h1"

FIXED_SCOPES: tuple[dict[str, str], ...] = (
    {
        "scope_id": "normal_isolated_168h_confirmed_swing_density",
        "cohort": "large",
        "actual_scope": "single_density_zone",
    },
    {
        "scope_id": "normal_168h_confirmed_swing_density_cluster",
        "cohort": "large",
        "actual_scope": "density_cluster",
    },
    {
        "scope_id": "meme_168h_confirmed_swing_density_cluster",
        "cohort": "meme",
        "actual_scope": "density_cluster",
    },
)

RAW_STATES = (
    "first_arrival_after_outside_interval",
    "near_miss",
    "already_inside",
    "repeat_contact",
)
MATCHED_STATE_CONTROLS = (
    "event_state__near_miss",
    "event_state__already_inside",
    "event_state__repeat_contact",
)
LOCATION_CONTROL_GROUPS: dict[str, dict[str, Any]] = {
    "same_width_same_approach_synthetic_boundary": {
        "mode": "all",
        "controls": ("same_state_no_level",),
        "interpretation": (
            "The frozen G2E no-level pool places a pseudo-boundary from the prior "
            "close using matched ATR distance, width, and approach, then excludes "
            "timestamps overlapping a real level."
        ),
    },
    "same_state_no_level": {
        "mode": "all",
        "controls": ("same_state_no_level",),
        "interpretation": "The same pseudo-boundary pool matched on the core causal state.",
    },
    "density_coverage_matched_no_level": {
        "mode": "all",
        "controls": ("same_density_coverage_no_level",),
        "interpretation": (
            "The no-level pseudo-boundary additionally matches density and nearby-level coverage."
        ),
    },
    "plus_and_minus_one_atr": {
        "mode": "all",
        "controls": ("shift_-1atr", "shift_+1atr"),
        "interpretation": "Both symmetric one-ATR locations must be fair in both validations.",
    },
    "causal_shuffled_swing_location": {
        "mode": "all",
        "controls": ("shuffled_swing_residual_assignment",),
        "interpretation": (
            "The causal swing identity is retained while its price residual is shuffled."
        ),
    },
    "current_vp_node": {
        "mode": "any_same_control",
        "controls": ("current_vp_hvn", "current_vp_lvn", "current_vp_poc"),
        "interpretation": (
            "At least one identical current VP-node type must be fair in both validations."
        ),
    },
    "prior_high_or_low": {
        "mode": "any_same_control",
        "controls": (
            "simple_prior_high_same_history",
            "simple_prior_low_same_history",
        ),
        "interpretation": (
            "At least one identical same-history prior-extreme type must be fair in "
            "both validations."
        ),
    },
}

METADATA_COLUMNS = (
    "density_family",
    "history_hours",
    "actual_scope",
    "control",
    "period",
    "response_window",
    "eligible_coin_count",
    "independent_event_pairs",
    "coverage_gate_passed",
    "state_balance_usable",
    "evidence_eligible",
    "max_absolute_state_smd",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 4B outcome-blind common-support audit for fresh-arrival "
            "event state and exact 168-hour swing-density location attribution."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--normal-preflight-run-id", required=True)
    parser.add_argument("--normal-control-run-id", required=True)
    parser.add_argument("--normal-state-run-id", required=True)
    parser.add_argument("--meme-preflight-run-id", required=True)
    parser.add_argument("--meme-control-run-id", required=True)
    parser.add_argument("--meme-state-run-id", required=True)
    parser.add_argument("--frozen-batch", type=Path, default=FROZEN_BATCH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    frozen_batch_path = args.frozen_batch.resolve()
    frozen_batch = json.loads(frozen_batch_path.read_text(encoding="utf-8"))
    branch = validate_frozen_branch(frozen_batch)
    sources = source_paths(args)
    source_contracts = validate_sources(sources)

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_batch_path": str(frozen_batch_path),
        "frozen_batch_sha256": sha256_file(frozen_batch_path),
        "frozen_branch": branch,
        "fixed_scopes": list(FIXED_SCOPES),
        "density_family": DENSITY_FAMILY,
        "history_hours": HISTORY_HOURS,
        "response_window_for_matching_support": RESPONSE_WINDOW,
        "coverage_gate": {
            "minimum_independent_events_or_pairs": MIN_EVENTS_OR_PAIRS,
            "minimum_coins": MIN_COINS,
            "must_repeat_in_both_chronological_validation_periods": True,
            "origin": "unchanged G3D declared support and state-balance gates",
        },
        "raw_states": list(RAW_STATES),
        "matched_state_controls": list(MATCHED_STATE_CONTROLS),
        "location_control_groups": LOCATION_CONTROL_GROUPS,
        "metadata_columns_read": list(METADATA_COLUMNS),
        "reaction_value_columns_read": [],
        "reaction_outcomes_opened_in_g4b": False,
        "direction_prediction": False,
        "profit_optimization": False,
        "source_contracts": source_contracts,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    record_path = run_dir / "g4b_support_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request contract.")
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "G3D found a repeatable fresh-arrival versus already-inside activity difference, "
            "but exact density-location attribution remained incomplete."
        ),
        "hypothesis": (
            "Fresh arrival is a real event-state effect and the actual 168-hour confirmed-"
            "swing density location adds a smaller response beyond generic arrival."
        ),
        "pass_fail": (
            "Open no new reaction values unless every required real state, matched event "
            "state, artificial/no-level boundary, shifted/shuffled location, VP-node, and "
            "prior-extreme group has fair support in both chronological validations."
        ),
        "workers": 1,
        "reaction_outcomes_opened_in_g4b": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)

    try:
        audit_frames: list[DataFrame] = []
        decision_rows: list[dict[str, Any]] = []
        for scope in FIXED_SCOPES:
            cohort_sources = sources[scope["cohort"]]
            audit, decision = audit_scope(
                scope,
                coverage_path=cohort_sources["coverage"],
                control_results_path=cohort_sources["control_results"],
                state_results_path=cohort_sources["state_results"],
            )
            audit_frames.append(audit)
            decision_rows.append(decision)

        audit_table = pd.concat(audit_frames, ignore_index=True)
        decisions = DataFrame(decision_rows)
        any_scope_passed = bool(decisions["common_support_passed"].any())
        terminal_classification = (
            "support_passed_reaction_attribution_may_open"
            if any_scope_passed
            else "parked_exact_location_controls_lack_common_support"
        )
        plain_language = plain_language_result(decisions)

        audit_path = run_dir / "g4b_common_support_audit.parquet"
        decision_path = run_dir / "g4b_scope_decisions.parquet"
        atomic_write_parquet(audit_table, audit_path)
        atomic_write_parquet(decisions, decision_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "terminal_classification": terminal_classification,
                "scope_count": len(decisions),
                "scopes_with_complete_common_support": int(
                    decisions["common_support_passed"].sum()
                ),
                "scope_decisions": decisions.to_dict(orient="records"),
                "plain_language_result": plain_language,
                "what_this_establishes": (
                    "Whether the frozen G4B comparisons are structurally fair enough "
                    "to inspect, without using any reaction value to make that decision."
                ),
                "what_this_does_not_establish": (
                    "It does not show that density locations cause or fail to cause a "
                    "reaction; unsupported comparisons cannot answer that question."
                ),
                "later_question": (
                    "No G4B descendant launches inside Generation 4. Any new support-"
                    "building method waits for the joint Generation 4 review."
                ),
                "outputs": {
                    "common_support_audit": str(audit_path),
                    "scope_decisions": str(decision_path),
                },
            }
        )
        atomic_write_json(record, record_path)
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

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def source_paths(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    return {
        "large": source_paths_for_cohort(
            preflight_run_id=args.normal_preflight_run_id,
            control_run_id=args.normal_control_run_id,
            state_run_id=args.normal_state_run_id,
        ),
        "meme": source_paths_for_cohort(
            preflight_run_id=args.meme_preflight_run_id,
            control_run_id=args.meme_control_run_id,
            state_run_id=args.meme_state_run_id,
        ),
    }


def source_paths_for_cohort(
    *, preflight_run_id: str, control_run_id: str, state_run_id: str
) -> dict[str, Path]:
    preflight_dir = PREFLIGHT_ROOT / preflight_run_id
    control_dir = CONTROL_ROOT / control_run_id
    state_dir = STATE_ROOT / state_run_id
    return {
        "preflight_record": preflight_dir / "g3d_preflight_run_record.json",
        "preflight_integrity": preflight_dir / "g3d_preflight_integrity.json",
        "coverage": preflight_dir / "g3d_state_coverage.parquet",
        "control_record": control_dir / "g3d_control_run_record.json",
        "control_integrity": control_dir / "g3d_control_integrity.json",
        "control_results": control_dir / "g3d_cohort_period_results.parquet",
        "state_record": state_dir / "g3d_state_run_record.json",
        "state_integrity": state_dir / "g3d_state_integrity.json",
        "state_results": state_dir / "g3d_cohort_period_results.parquet",
    }


def validate_frozen_branch(frozen_batch: dict[str, Any]) -> dict[str, Any]:
    branches = frozen_batch.get("branches", [])
    matches = [
        branch
        for branch in branches
        if branch.get("id") == "g4b_fresh_arrival_location_attribution"
    ]
    if len(matches) != 1:
        raise ValueError("The frozen Generation 4 batch lacks exactly one G4B branch.")
    branch = matches[0]
    if branch.get("status") != "frozen_next_batch":
        raise ValueError("G4B is not frozen for execution.")
    if int(branch.get("iteration_cap", 0)) != 4:
        raise ValueError("G4B iteration cap changed from the frozen value of four.")
    return branch


def validate_sources(
    sources: dict[str, dict[str, Path]],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for cohort, paths in sources.items():
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        for stage in ("preflight", "control", "state"):
            record_path = paths[f"{stage}_record"]
            integrity_path = paths[f"{stage}_integrity"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            request = record.get("request_contract", {})
            if record.get("status") != "completed" or not integrity.get("passed"):
                raise ValueError(f"Unclean G3D {stage} source for {cohort}.")
            if request.get("cohort") != cohort:
                raise ValueError(f"G3D {stage} source cohort mismatch for {cohort}.")
            if request.get("direction_prediction") is not False:
                raise ValueError(f"G3D {stage} source crossed the direction boundary.")
            if request.get("profit_optimization") is not False:
                raise ValueError(f"G3D {stage} source crossed the profit boundary.")
            contracts.append(
                {
                    "cohort": cohort,
                    "stage": stage,
                    "record_path": str(record_path),
                    "record_sha256": sha256_file(record_path),
                    "integrity_path": str(integrity_path),
                    "integrity_sha256": sha256_file(integrity_path),
                }
            )
        for name in ("coverage", "control_results", "state_results"):
            path = paths[name]
            contracts.append(
                {
                    "cohort": cohort,
                    "stage": name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return contracts


def audit_scope(
    scope: dict[str, str],
    *,
    coverage_path: Path,
    control_results_path: Path,
    state_results_path: Path,
) -> tuple[DataFrame, dict[str, Any]]:
    coverage = pd.read_parquet(
        coverage_path,
        columns=[
            "period",
            "period_role",
            "density_family",
            "history_hours",
            "actual_scope",
            "event_state",
            "events",
            "coins",
            "state_coverage_gate_passed",
        ],
    )
    selected_coverage = select_scope(coverage, scope)
    validation_periods = tuple(
        selected_coverage.loc[
            selected_coverage["period_role"].eq("chronological_internal_validation"),
            "period",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if len(validation_periods) != 2:
        raise ValueError(
            f"{scope['scope_id']} has {len(validation_periods)} validation periods, expected two."
        )

    control_metadata = read_matching_metadata(control_results_path, scope)
    state_metadata = read_matching_metadata(state_results_path, scope)
    rows: list[dict[str, Any]] = []

    for period in validation_periods:
        for state in RAW_STATES:
            cell = selected_coverage.loc[
                selected_coverage["period"].astype(str).eq(period)
                & selected_coverage["event_state"].astype(str).eq(state)
            ]
            if len(cell) > 1:
                raise ValueError(
                    f"Duplicate raw state coverage for {scope['scope_id']} {period} {state}."
                )
            events = int(cell["events"].iloc[0]) if len(cell) else 0
            coins = int(cell["coins"].iloc[0]) if len(cell) else 0
            stored_gate = bool(cell["state_coverage_gate_passed"].iloc[0]) if len(cell) else False
            recomputed_gate = events >= MIN_EVENTS_OR_PAIRS and coins >= MIN_COINS
            if stored_gate != recomputed_gate:
                raise ValueError(
                    "Stored raw-state coverage gate disagrees with the declared G3D gate."
                )
            rows.append(
                audit_row(
                    scope,
                    period=period,
                    check_family="raw_event_state",
                    check_group=state,
                    check_name=state,
                    source_control=state,
                    events_or_pairs=events,
                    coins=coins,
                    coverage_gate_passed=stored_gate,
                    state_balance_usable=True,
                    source_path=coverage_path,
                )
            )

        for control in MATCHED_STATE_CONTROLS:
            rows.append(
                matching_audit_row(
                    scope,
                    metadata=state_metadata,
                    period=period,
                    check_family="matched_event_state",
                    check_group=control,
                    control=control,
                    source_path=state_results_path,
                )
            )

        location_controls = sorted(
            {
                control
                for group in LOCATION_CONTROL_GROUPS.values()
                for control in group["controls"]
            }
        )
        for control in location_controls:
            groups = ";".join(
                name
                for name, group in LOCATION_CONTROL_GROUPS.items()
                if control in group["controls"]
            )
            rows.append(
                matching_audit_row(
                    scope,
                    metadata=control_metadata,
                    period=period,
                    check_family="location_control",
                    check_group=groups,
                    control=control,
                    source_path=control_results_path,
                )
            )

    audit = DataFrame(rows)
    raw_passed = family_passed(
        audit,
        family="raw_event_state",
        required=RAW_STATES,
        validation_periods=validation_periods,
    )
    state_passed = family_passed(
        audit,
        family="matched_event_state",
        required=MATCHED_STATE_CONTROLS,
        validation_periods=validation_periods,
    )
    location_group_status = {
        group_name: control_group_passed(
            audit,
            group=group,
            validation_periods=validation_periods,
        )
        for group_name, group in LOCATION_CONTROL_GROUPS.items()
    }
    failed_raw = [state for state, passed in raw_passed.items() if not passed]
    failed_state = [control for control, passed in state_passed.items() if not passed]
    failed_location = [
        group for group, passed in location_group_status.items() if not passed
    ]
    blockers = [
        *(f"raw_state:{name}" for name in failed_raw),
        *(f"matched_state:{name}" for name in failed_state),
        *(f"location:{name}" for name in failed_location),
    ]
    return audit, {
        **scope,
        "validation_periods": ";".join(validation_periods),
        "raw_states_passed": not failed_raw,
        "matched_states_passed": not failed_state,
        "location_controls_passed": not failed_location,
        "common_support_passed": not blockers,
        "blockers": ";".join(blockers),
        "reaction_outcomes_opened_in_g4b": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def select_scope(frame: DataFrame, scope: dict[str, str]) -> DataFrame:
    return frame.loc[
        frame["density_family"].astype(str).eq(DENSITY_FAMILY)
        & pd.to_numeric(frame["history_hours"], errors="coerce").eq(HISTORY_HOURS)
        & frame["actual_scope"].astype(str).eq(scope["actual_scope"])
    ].copy()


def read_matching_metadata(path: Path, scope: dict[str, str]) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(METADATA_COLUMNS))
    frame = select_scope(frame, scope)
    frame = frame.loc[frame["response_window"].astype(str).eq(RESPONSE_WINDOW)].copy()
    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    key_columns = ["period", "control"]
    conflicting = deduplicated.groupby(key_columns, observed=True).size()
    if bool(conflicting.gt(1).any()):
        keys = conflicting.loc[conflicting.gt(1)].index.tolist()
        raise ValueError(f"Conflicting outcome-independent support metadata at {path}: {keys}")
    return deduplicated


def matching_audit_row(
    scope: dict[str, str],
    *,
    metadata: DataFrame,
    period: str,
    check_family: str,
    check_group: str,
    control: str,
    source_path: Path,
) -> dict[str, Any]:
    cell = metadata.loc[
        metadata["period"].astype(str).eq(period)
        & metadata["control"].astype(str).eq(control)
    ]
    if len(cell) > 1:
        raise ValueError(f"Duplicate support metadata for {period} {control}.")
    pairs = int(cell["independent_event_pairs"].iloc[0]) if len(cell) else 0
    coins = int(cell["eligible_coin_count"].iloc[0]) if len(cell) else 0
    coverage_gate = bool(cell["coverage_gate_passed"].iloc[0]) if len(cell) else False
    balance_gate = bool(cell["state_balance_usable"].iloc[0]) if len(cell) else False
    evidence_gate = bool(cell["evidence_eligible"].iloc[0]) if len(cell) else False
    if evidence_gate != (coverage_gate and balance_gate):
        raise ValueError("Stored evidence gate disagrees with coverage and state balance.")
    return audit_row(
        scope,
        period=period,
        check_family=check_family,
        check_group=check_group,
        check_name=control,
        source_control=control,
        events_or_pairs=pairs,
        coins=coins,
        coverage_gate_passed=coverage_gate,
        state_balance_usable=balance_gate,
        source_path=source_path,
        max_absolute_state_smd=(
            float(cell["max_absolute_state_smd"].iloc[0]) if len(cell) else None
        ),
    )


def audit_row(
    scope: dict[str, str],
    *,
    period: str,
    check_family: str,
    check_group: str,
    check_name: str,
    source_control: str,
    events_or_pairs: int,
    coins: int,
    coverage_gate_passed: bool,
    state_balance_usable: bool,
    source_path: Path,
    max_absolute_state_smd: float | None = None,
) -> dict[str, Any]:
    return {
        **scope,
        "period": period,
        "check_family": check_family,
        "check_group": check_group,
        "check_name": check_name,
        "source_control": source_control,
        "events_or_independent_pairs": events_or_pairs,
        "eligible_coins": coins,
        "coverage_gate_passed": coverage_gate_passed,
        "state_balance_usable": state_balance_usable,
        "support_passed": coverage_gate_passed and state_balance_usable,
        "max_absolute_state_smd": max_absolute_state_smd,
        "source_path": str(source_path),
        "reaction_value_read": False,
    }


def family_passed(
    audit: DataFrame,
    *,
    family: str,
    required: Sequence[str],
    validation_periods: Sequence[str],
) -> dict[str, bool]:
    selected = audit.loc[audit["check_family"].eq(family)]
    return {
        name: all(
            bool(
                selected.loc[
                    selected["period"].eq(period)
                    & selected["check_name"].eq(name),
                    "support_passed",
                ].all()
            )
            and bool(
                len(
                    selected.loc[
                        selected["period"].eq(period)
                        & selected["check_name"].eq(name)
                    ]
                )
            )
            for period in validation_periods
        )
        for name in required
    }


def control_group_passed(
    audit: DataFrame,
    *,
    group: dict[str, Any],
    validation_periods: Sequence[str],
) -> bool:
    controls = tuple(group["controls"])
    control_status = family_passed(
        audit,
        family="location_control",
        required=controls,
        validation_periods=validation_periods,
    )
    if group["mode"] == "all":
        return all(control_status.values())
    if group["mode"] == "any_same_control":
        return any(control_status.values())
    raise ValueError(f"Unknown location-control group mode: {group['mode']}")


def plain_language_result(decisions: DataFrame) -> str:
    passed = decisions.loc[decisions["common_support_passed"]]
    if not passed.empty:
        names = ", ".join(passed["scope_id"].astype(str))
        return (
            f"Complete fair support exists for {names}. Only those scopes may proceed "
            "to a separately frozen reaction-value comparison."
        )
    return (
        "None of the three frozen scopes can fairly compare the real density location "
        "with every required alternative in both validation periods. The normal isolated "
        "scope loses clean-near-miss support after matching; the normal cluster keeps the "
        "event states but not the complete fair location-control ladder; and the meme "
        "cluster loses near-miss support and the fair location-control ladder. G4B is "
        "therefore parked without opening new reaction values."
    )


if __name__ == "__main__":
    raise SystemExit(main())
