from __future__ import annotations

# The repository root is inserted before project-local imports.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_artificial_controls import (  # noqa: E501
    CONTROL_NAMES as ARTIFICIAL_CONTROLS,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_artificial_controls import (  # noqa: E501
    REPORT_ROOT as ARTIFICIAL_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    COMPONENT_CONTROL,
    NO_LEVEL_CONTROL,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    REPORT_ROOT as DIRECT_REACTION_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_review import (  # noqa: E501
    REPORT_ROOT as DIRECT_REVIEW_ROOT,
)


REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels_attribution_review"
OUTPUT_SCHEMA_VERSION = 1
CLAIM_KEYS = (
    "route_id",
    "density_family",
    "history_hours",
    "actual_scope",
    "response_window",
    "outcome",
)


@dataclass(frozen=True)
class ScopeRule:
    name: str
    periods: tuple[str, str]
    minimum_positive_fraction: float
    maximum_positive_fraction: float


SCOPE_RULES = {
    "normal_top10": ScopeRule(
        name="normal_top10",
        periods=("validation_early", "validation_late"),
        minimum_positive_fraction=0.70,
        maximum_positive_fraction=0.30,
    ),
    "meme_top10": ScopeRule(
        name="meme_top10",
        periods=("meme_validation_early", "meme_validation_late"),
        minimum_positive_fraction=0.70,
        maximum_positive_fraction=0.30,
    ),
    "platform6": ScopeRule(
        name="platform6",
        periods=("validation_early", "validation_late"),
        minimum_positive_fraction=5.0 / 6.0,
        maximum_positive_fraction=1.0 / 6.0,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Close Generation 3F by applying every direct and artificial control to "
            "the outcome claims frozen before the artificial tests."
        )
    )
    parser.add_argument("--direct-review-run-id", required=True)
    parser.add_argument("--normal-artificial-run-id", required=True)
    parser.add_argument("--meme-artificial-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    direct_review_dir = DIRECT_REVIEW_ROOT / args.direct_review_run_id
    direct_review_record_path = direct_review_dir / "g3f_review_run_record.json"
    direct_review_integrity_path = direct_review_dir / "g3f_review_integrity.json"
    candidate_path = direct_review_dir / "g3f_candidate_outcomes.parquet"
    direct_review_record = validated_record(
        direct_review_record_path,
        direct_review_integrity_path,
        label="G3F direct review",
    )
    candidates = pd.read_parquet(candidate_path)
    if candidates.empty:
        raise ValueError("G3F direct review exposed no frozen candidate outcomes.")

    direct_request = direct_review_record["request_contract"]
    normal_direct_run = str(direct_request["normal_run_id"])
    meme_direct_run = str(direct_request["meme_run_id"])
    direct_sources = {
        "normal_top10": load_direct_source(normal_direct_run),
        "meme_top10": load_direct_source(meme_direct_run),
        "platform6": {
            "cohort": pd.read_parquet(direct_review_dir / "g3f_platform_cohort_period.parquet"),
            "leave_one_out": pd.read_parquet(
                direct_review_dir / "g3f_platform_leave_one_out.parquet"
            ),
        },
    }
    artificial_sources = {
        "normal_top10": load_artificial_source(
            args.normal_artificial_run_id,
            expected_cohort="large",
            platform=False,
        ),
        "meme_top10": load_artificial_source(
            args.meme_artificial_run_id,
            expected_cohort="meme",
            platform=False,
        ),
        "platform6": load_artificial_source(
            args.normal_artificial_run_id,
            expected_cohort="large",
            platform=True,
        ),
    }
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "direct_review_run_id": args.direct_review_run_id,
        "direct_review_record_sha256": sha256_file(direct_review_record_path),
        "direct_review_integrity_sha256": sha256_file(direct_review_integrity_path),
        "frozen_candidate_outcomes_sha256": sha256_file(candidate_path),
        "normal_direct_run_id": normal_direct_run,
        "meme_direct_run_id": meme_direct_run,
        "normal_artificial_run_id": args.normal_artificial_run_id,
        "meme_artificial_run_id": args.meme_artificial_run_id,
        "required_controls": {
            "all_cells": [NO_LEVEL_CONTROL, *ARTIFICIAL_CONTROLS],
            "cluster_cells_additionally": [COMPONENT_CONTROL],
            "same_density_interpretation": (
                "The direct no-level control matched the complete generic-density state "
                "as well as OHLCV, technical and broad-market state."
            ),
        },
        "retain_rule": (
            "Every applicable control must have both validation periods, adequate "
            "coverage, usable declared-state balance, one non-zero pooled/equal-coin "
            "sign, declared member agreement and leave-one-coin-out sign stability. "
            "Every control must then agree on that sign."
        ),
        "missing_control_rule": "Missing fair coverage is insufficient evidence, never a pass.",
        "scope_rules": {name: rule.__dict__ for name, rule in SCOPE_RULES.items()},
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3f_attribution_review_run_record.json"
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "request_sha256": request_sha256,
        "request_contract": request,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    try:
        claims = candidates.drop_duplicates(["scope", *CLAIM_KEYS]).reset_index(drop=True)
        detail_rows: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        for _, claim in claims.iterrows():
            scope = str(claim["scope"])
            rule = SCOPE_RULES[scope]
            relationship = str(claim["actual_scope"]).split("__", maxsplit=1)[0]
            direct_controls = [NO_LEVEL_CONTROL]
            if relationship != "isolated_generic_level":
                direct_controls.append(COMPONENT_CONTROL)
            required = [*direct_controls, *ARTIFICIAL_CONTROLS]
            claim_details: list[dict[str, Any]] = []
            for control in direct_controls:
                claim_details.append(
                    assess_control(
                        claim=claim,
                        control=control,
                        source_stage="direct",
                        source=direct_sources[scope],
                        rule=rule,
                    )
                )
            for control in ARTIFICIAL_CONTROLS:
                claim_details.append(
                    assess_control(
                        claim=claim,
                        control=control,
                        source_stage="artificial",
                        source=artificial_sources[scope],
                        rule=rule,
                    )
                )
            detail_rows.extend(claim_details)
            passed = [row for row in claim_details if row["control_status"] == "passed"]
            signs = sorted({str(row["effect_sign"]) for row in passed})
            all_passed = len(passed) == len(required)
            if all_passed and len(signs) == 1:
                final_status = "retained_full_attribution"
                final_sign = signs[0]
            elif len(signs) > 1:
                final_status = "parked_contradictory_controls"
                final_sign = "contradictory"
            else:
                final_status = "parked_incomplete_control_ladder"
                final_sign = signs[0] if signs else "not_established"
            status_counts = pd.Series(
                [row["control_status"] for row in claim_details]
            ).value_counts()
            final_rows.append(
                {
                    "scope": scope,
                    **{key: claim[key] for key in CLAIM_KEYS},
                    "required_control_count": len(required),
                    "passed_control_count": len(passed),
                    "passed_controls": ";".join(
                        sorted(str(row["control"]) for row in passed)
                    ),
                    "passed_effect_signs": ";".join(signs),
                    "nonpassing_controls": ";".join(
                        sorted(
                            f"{row['control']}={row['control_status']}"
                            for row in claim_details
                            if row["control_status"] != "passed"
                        )
                    ),
                    "coverage_failure_controls": int(
                        status_counts.get("insufficient_coverage", 0)
                    ),
                    "state_imbalance_controls": int(status_counts.get("state_imbalance", 0)),
                    "effect_instability_controls": int(
                        status_counts.get("effect_sign_instability", 0)
                    ),
                    "member_instability_controls": int(
                        status_counts.get("member_agreement_failed", 0)
                    ),
                    "leave_one_out_failure_controls": int(
                        status_counts.get("leave_one_out_failed", 0)
                    ),
                    "missing_comparison_controls": int(
                        status_counts.get("missing_comparison", 0)
                    ),
                    "final_status": final_status,
                    "final_effect_sign": final_sign,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
        details = DataFrame(detail_rows)
        final = DataFrame(final_rows)
        cells = summarize_cells(final)
        summary = summarize_scope_results(final, details)
        atomic_write_parquet(details, run_dir / "g3f_control_outcome_details.parquet")
        atomic_write_parquet(final, run_dir / "g3f_final_outcome_assessment.parquet")
        atomic_write_parquet(cells, run_dir / "g3f_final_cell_assessment.parquet")
        atomic_write_parquet(summary, run_dir / "g3f_scope_control_summary.parquet")
        retained = int(final["final_status"].eq("retained_full_attribution").sum())
        integrity = {
            "created_at_utc": utc_now(),
            "passed": bool(
                len(final) == len(claims)
                and len(details) == int(final["required_control_count"].sum())
                and details["direction_prediction"].eq(False).all()
                and details["profit_optimization"].eq(False).all()
                and final["direction_prediction"].eq(False).all()
                and final["profit_optimization"].eq(False).all()
            ),
            "frozen_claims": len(claims),
            "control_detail_rows": len(details),
            "final_claim_rows": len(final),
            "final_cell_rows": len(cells),
            "retained_full_attribution_rows": retained,
            "direction_prediction": False,
            "profit_optimization": False,
        }
        atomic_write_json(integrity, run_dir / "g3f_attribution_review_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3F attribution-review integrity failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "frozen_claims": len(claims),
                "control_detail_rows": len(details),
                "final_claim_rows": len(final),
                "final_cell_rows": len(cells),
                "retained_full_attribution_rows": retained,
                "final_status_counts": final["final_status"].value_counts().to_dict(),
                "integrity": str(run_dir / "g3f_attribution_review_integrity.json"),
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
    return 0


def validated_record(record_path: Path, integrity_path: Path, *, label: str) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or integrity.get("passed") is not True:
        raise ValueError(f"{label} is incomplete or invalid.")
    for source in (record, integrity):
        if source.get("direction_prediction") is not False:
            raise ValueError(f"{label} violates direction_prediction=False.")
        if source.get("profit_optimization") is not False:
            raise ValueError(f"{label} violates profit_optimization=False.")
    return record


def load_direct_source(run_id: str) -> dict[str, DataFrame]:
    run_dir = DIRECT_REACTION_ROOT / run_id
    validated_record(
        run_dir / "g3f_reaction_run_record.json",
        run_dir / "g3f_reaction_integrity.json",
        label=f"G3F direct run {run_id}",
    )
    return {
        "cohort": pd.read_parquet(run_dir / "g3f_cohort_period_results.parquet"),
        "leave_one_out": pd.read_parquet(run_dir / "g3f_leave_one_coin_out.parquet"),
    }


def load_artificial_source(
    run_id: str,
    *,
    expected_cohort: str,
    platform: bool,
) -> dict[str, DataFrame]:
    run_dir = ARTIFICIAL_REPORT_ROOT / run_id
    record = validated_record(
        run_dir / "g3f_artificial_run_record.json",
        run_dir / "g3f_artificial_integrity.json",
        label=f"G3F artificial run {run_id}",
    )
    if record.get("request_contract", {}).get("cohort") != expected_cohort:
        raise ValueError(f"G3F artificial cohort mismatch: {run_id}")
    cohort_name = (
        "g3f_platform_cohort_period.parquet"
        if platform
        else "g3f_cohort_period_results.parquet"
    )
    leave_one_out_name = (
        "g3f_platform_leave_one_out.parquet"
        if platform
        else "g3f_leave_one_coin_out.parquet"
    )
    return {
        "cohort": pd.read_parquet(run_dir / cohort_name),
        "leave_one_out": pd.read_parquet(run_dir / leave_one_out_name),
    }


def claim_mask(frame: DataFrame, claim: pd.Series, *, control: str) -> np.ndarray:
    mask = frame["control"].astype(str).eq(control).to_numpy(copy=True)
    for key in CLAIM_KEYS:
        mask &= frame[key].eq(claim[key]).to_numpy()
    return mask


def assess_control(
    *,
    claim: pd.Series,
    control: str,
    source_stage: str,
    source: dict[str, DataFrame],
    rule: ScopeRule,
) -> dict[str, Any]:
    cohort = source["cohort"]
    leave_one_out = source["leave_one_out"]
    group = cohort.loc[claim_mask(cohort, claim, control=control)].copy()
    validation = group.loc[group["period"].astype(str).isin(rule.periods)].copy()
    base = {
        "scope": rule.name,
        **{key: claim[key] for key in CLAIM_KEYS},
        "source_stage": source_stage,
        "control": control,
        "required_periods": ";".join(rule.periods),
        "observed_periods": ";".join(sorted(validation["period"].astype(str).unique())),
        "control_status": "missing_comparison",
        "effect_sign": "not_established",
        "early_coverage_gate_passed": False,
        "late_coverage_gate_passed": False,
        "early_state_balance_usable": False,
        "late_state_balance_usable": False,
        "early_independent_event_pairs": 0,
        "late_independent_event_pairs": 0,
        "early_equal_coin_delta_median": np.nan,
        "late_equal_coin_delta_median": np.nan,
        "early_pooled_event_delta_mean": np.nan,
        "late_pooled_event_delta_mean": np.nan,
        "early_positive_coin_fraction": np.nan,
        "late_positive_coin_fraction": np.nan,
        "leave_one_out_rows": 0,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    if set(validation["period"].astype(str)) != set(rule.periods) or len(validation) != 2:
        return base
    by_period = {str(row.period): row for row in validation.itertuples()}
    early = by_period[rule.periods[0]]
    late = by_period[rule.periods[1]]
    base.update(
        {
            "early_coverage_gate_passed": bool(early.coverage_gate_passed),
            "late_coverage_gate_passed": bool(late.coverage_gate_passed),
            "early_state_balance_usable": bool(early.state_balance_usable),
            "late_state_balance_usable": bool(late.state_balance_usable),
            "early_independent_event_pairs": int(early.independent_event_pairs),
            "late_independent_event_pairs": int(late.independent_event_pairs),
            "early_equal_coin_delta_median": float(early.equal_coin_delta_median),
            "late_equal_coin_delta_median": float(late.equal_coin_delta_median),
            "early_pooled_event_delta_mean": float(early.pooled_event_delta_mean),
            "late_pooled_event_delta_mean": float(late.pooled_event_delta_mean),
            "early_positive_coin_fraction": float(early.equal_coin_positive_fraction),
            "late_positive_coin_fraction": float(late.equal_coin_positive_fraction),
        }
    )
    if not validation["coverage_gate_passed"].fillna(False).all():
        base["control_status"] = "insufficient_coverage"
        return base
    if not validation["state_balance_usable"].fillna(False).all():
        base["control_status"] = "state_imbalance"
        return base
    pooled_sign = np.sign(validation["pooled_event_delta_mean"].to_numpy(dtype=float))
    equal_sign = np.sign(validation["equal_coin_delta_median"].to_numpy(dtype=float))
    if (
        0.0 in pooled_sign
        or 0.0 in equal_sign
        or len(set(pooled_sign)) != 1
        or len(set(equal_sign)) != 1
        or pooled_sign[0] != equal_sign[0]
    ):
        base["control_status"] = "effect_sign_instability"
        return base
    sign = int(pooled_sign[0])
    base["effect_sign"] = "higher" if sign > 0 else "lower"
    positive = validation["equal_coin_positive_fraction"]
    member_agreement = (
        positive.ge(rule.minimum_positive_fraction).all()
        if sign > 0
        else positive.le(rule.maximum_positive_fraction).all()
    )
    if not member_agreement:
        base["control_status"] = "member_agreement_failed"
        return base
    loo_mask = leave_one_out["control"].astype(str).eq(control).to_numpy(copy=True)
    for key in CLAIM_KEYS:
        loo_mask &= leave_one_out[key].eq(claim[key]).to_numpy()
    loo = leave_one_out.loc[
        loo_mask & leave_one_out["period"].astype(str).isin(rule.periods)
    ].copy()
    base["leave_one_out_rows"] = len(loo)
    if (
        set(loo["period"].astype(str)) != set(rule.periods)
        or loo.empty
        or not loo["coverage_gate_passed"].fillna(False).all()
        or not (np.sign(loo["equal_coin_delta_median"].to_numpy(dtype=float)) == sign).all()
    ):
        base["control_status"] = "leave_one_out_failed"
        return base
    base["control_status"] = "passed"
    return base


def summarize_cells(final: DataFrame) -> DataFrame:
    cell_keys = ["scope", "route_id", "density_family", "history_hours", "actual_scope"]
    return (
        final.groupby(cell_keys, observed=True, dropna=False)
        .agg(
            tested_outcomes=("outcome", "size"),
            retained_outcomes=(
                "final_status",
                lambda values: int(values.eq("retained_full_attribution").sum()),
            ),
            final_statuses=(
                "final_status",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
            tested_outcome_names=(
                "outcome",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
        )
        .reset_index()
    )


def summarize_scope_results(final: DataFrame, details: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_RULES:
        scoped_final = final.loc[final["scope"].eq(scope)]
        scoped_detail = details.loc[details["scope"].eq(scope)]
        statuses = scoped_detail["control_status"].value_counts()
        rows.append(
            {
                "scope": scope,
                "frozen_claims": len(scoped_final),
                "retained_full_attribution": int(
                    scoped_final["final_status"].eq("retained_full_attribution").sum()
                ),
                "contradictory_claims": int(
                    scoped_final["final_status"].eq("parked_contradictory_controls").sum()
                ),
                "incomplete_control_ladder_claims": int(
                    scoped_final["final_status"].eq(
                        "parked_incomplete_control_ladder"
                    ).sum()
                ),
                "control_detail_rows": len(scoped_detail),
                "passed_control_rows": int(statuses.get("passed", 0)),
                "missing_comparison_rows": int(statuses.get("missing_comparison", 0)),
                "insufficient_coverage_rows": int(statuses.get("insufficient_coverage", 0)),
                "state_imbalance_rows": int(statuses.get("state_imbalance", 0)),
                "effect_sign_instability_rows": int(
                    statuses.get("effect_sign_instability", 0)
                ),
                "member_agreement_failed_rows": int(
                    statuses.get("member_agreement_failed", 0)
                ),
                "leave_one_out_failed_rows": int(
                    statuses.get("leave_one_out_failed", 0)
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
