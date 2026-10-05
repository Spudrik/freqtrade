from __future__ import annotations

# The repository root is inserted before project-local imports.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    ARTIFACT_ROOT as REACTION_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    REPORT_ROOT as REACTION_REPORT_ROOT,
)


REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels_review"
OUTPUT_SCHEMA_VERSION = 1
PLATFORM_PAIRS = (
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "TRX/USDT:USDT",
)
GROUP_KEYS = (
    "route_id",
    "density_family",
    "history_hours",
    "actual_scope",
    "control",
    "response_window",
    "outcome",
)
CELL_KEYS = (
    "route_id",
    "density_family",
    "history_hours",
    "actual_scope",
    "control",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze corrected Generation 3F direct leads and evaluate the predeclared "
            "six-platform-coin subgroup before artificial controls."
        )
    )
    parser.add_argument("--normal-run-id", required=True)
    parser.add_argument("--meme-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    normal = validated_run(args.normal_run_id, expected_cohort="large")
    meme = validated_run(args.meme_run_id, expected_cohort="meme")
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "normal_run_id": args.normal_run_id,
        "normal_record_sha256": sha256_file(normal["record_path"]),
        "normal_integrity_sha256": sha256_file(normal["integrity_path"]),
        "meme_run_id": args.meme_run_id,
        "meme_record_sha256": sha256_file(meme["record_path"]),
        "meme_integrity_sha256": sha256_file(meme["integrity_path"]),
        "claim_scopes": {
            "normal_top10": {
                "pairs": normal["pairs"],
                "minimum_member_agreement": "7 of 10",
            },
            "meme_top10": {
                "pairs": meme["pairs"],
                "minimum_member_agreement": "7 of 10",
            },
            "platform6": {
                "pairs": list(PLATFORM_PAIRS),
                "minimum_member_agreement": "5 of 6",
                "predeclared_before_G3_outcomes": True,
            },
        },
        "selection_rule": (
            "Both validation periods must pass coverage and state balance; pooled-event "
            "and equal-coin effects must share one non-zero sign in both periods; the "
            "declared member fraction must agree in both periods; and every leave-one-"
            "coin-out result must retain coverage and sign. Positive and negative "
            "direction-neutral behaviour are both retained for stronger controls."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3f_review_run_record.json"
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
        normal_cohort = pd.read_parquet(normal["cohort_period_path"])
        normal_loo = pd.read_parquet(normal["leave_one_out_path"])
        meme_cohort = pd.read_parquet(meme["cohort_period_path"])
        meme_loo = pd.read_parquet(meme["leave_one_out_path"])

        platform_independent = load_pair_outputs(
            run_id=args.normal_run_id,
            pairs=PLATFORM_PAIRS,
        )
        platform_pair = pair_period_results(platform_independent)
        platform_cohort = cohort_period_results(platform_pair, platform_independent)
        platform_loo = leave_one_coin_out_results(platform_pair)
        atomic_write_parquet(platform_pair, run_dir / "g3f_platform_pair_period.parquet")
        atomic_write_parquet(platform_cohort, run_dir / "g3f_platform_cohort_period.parquet")
        atomic_write_parquet(platform_loo, run_dir / "g3f_platform_leave_one_out.parquet")

        candidate_frames = [
            select_repeated_candidates(
                cohort=normal_cohort,
                leave_one_out=normal_loo,
                scope="normal_top10",
                periods=("validation_early", "validation_late"),
                minimum_positive_fraction=0.70,
                maximum_positive_fraction=0.30,
            ),
            select_repeated_candidates(
                cohort=meme_cohort,
                leave_one_out=meme_loo,
                scope="meme_top10",
                periods=("meme_validation_early", "meme_validation_late"),
                minimum_positive_fraction=0.70,
                maximum_positive_fraction=0.30,
            ),
            select_repeated_candidates(
                cohort=platform_cohort,
                leave_one_out=platform_loo,
                scope="platform6",
                periods=("validation_early", "validation_late"),
                minimum_positive_fraction=5.0 / 6.0,
                maximum_positive_fraction=1.0 / 6.0,
            ),
        ]
        outcomes = pd.concat(candidate_frames, ignore_index=True)
        if outcomes.empty:
            cells = DataFrame(columns=["scope", *CELL_KEYS, "candidate_outcome_count"])
        else:
            cells = (
                outcomes.groupby(["scope", *CELL_KEYS], observed=True, dropna=False)
                .agg(
                    candidate_outcome_count=("outcome", "size"),
                    candidate_outcomes=(
                        "outcome",
                        lambda values: ";".join(sorted(set(values.astype(str)))),
                    ),
                    candidate_signs=(
                        "effect_sign",
                        lambda values: ";".join(sorted(set(values.astype(str)))),
                    ),
                )
                .reset_index()
            )
        atomic_write_parquet(outcomes, run_dir / "g3f_candidate_outcomes.parquet")
        atomic_write_parquet(cells, run_dir / "g3f_candidate_cells.parquet")
        integrity = {
            "created_at_utc": utc_now(),
            "passed": True,
            "normal_pairs": normal["pairs"],
            "meme_pairs": meme["pairs"],
            "platform_pairs": list(PLATFORM_PAIRS),
            "platform_pairs_present": sorted(platform_independent["pair"].unique()),
            "candidate_outcome_rows": len(outcomes),
            "candidate_cells": len(cells),
            "candidate_cells_by_scope": (
                cells.groupby("scope").size().astype(int).to_dict() if not cells.empty else {}
            ),
            "direction_prediction": False,
            "profit_optimization": False,
        }
        atomic_write_json(integrity, run_dir / "g3f_review_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "candidate_outcome_rows": len(outcomes),
                "candidate_cells": len(cells),
                "candidate_cells_by_scope": integrity["candidate_cells_by_scope"],
                "integrity": str(run_dir / "g3f_review_integrity.json"),
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


def validated_run(run_id: str, *, expected_cohort: str) -> dict[str, Any]:
    run_dir = REACTION_REPORT_ROOT / run_id
    record_path = run_dir / "g3f_reaction_run_record.json"
    integrity_path = run_dir / "g3f_reaction_integrity.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or integrity.get("passed") is not True:
        raise ValueError(f"G3F reaction source is incomplete or invalid: {run_id}")
    request = record.get("request_contract", {})
    if request.get("cohort") != expected_cohort:
        raise ValueError(f"G3F reaction cohort mismatch: {run_id}")
    if record.get("schema_version") != 2:
        raise ValueError(f"G3F review requires the formula-alias-corrected schema: {run_id}")
    for source in (record, integrity):
        if source.get("direction_prediction") is not False:
            raise ValueError(f"G3F source violates direction_prediction=False: {run_id}")
        if source.get("profit_optimization") is not False:
            raise ValueError(f"G3F source violates profit_optimization=False: {run_id}")
    return {
        "record_path": record_path,
        "integrity_path": integrity_path,
        "pairs": list(request["pairs"]),
        "pair_period_path": run_dir / "g3f_pair_period_results.parquet",
        "cohort_period_path": run_dir / "g3f_cohort_period_results.parquet",
        "leave_one_out_path": run_dir / "g3f_leave_one_coin_out.parquet",
    }


def load_pair_outputs(*, run_id: str, pairs: Sequence[str]) -> DataFrame:
    pair_dir = REACTION_ARTIFACT_ROOT / run_id / "pair_independent_matches"
    frames = [pd.read_parquet(pair_dir / f"{pair_stem(pair)}.parquet") for pair in pairs]
    output = pd.concat(frames, ignore_index=True)
    if set(output["pair"].astype(str)) != set(pairs):
        raise ValueError("Platform subgroup pair artifacts do not match the frozen membership.")
    return output


def select_repeated_candidates(
    *,
    cohort: DataFrame,
    leave_one_out: DataFrame,
    scope: str,
    periods: tuple[str, str],
    minimum_positive_fraction: float,
    maximum_positive_fraction: float,
) -> DataFrame:
    validation = cohort.loc[
        cohort["period"].isin(periods) & cohort["evidence_eligible"].fillna(False)
    ].copy()
    rows: list[dict[str, Any]] = []
    for key, group in validation.groupby(list(GROUP_KEYS), observed=True, dropna=False, sort=False):
        if set(group["period"].astype(str)) != set(periods) or len(group) != 2:
            continue
        pooled_sign = np.sign(group["pooled_event_delta_mean"].to_numpy(dtype=float))
        equal_coin_sign = np.sign(group["equal_coin_delta_median"].to_numpy(dtype=float))
        if (
            0.0 in pooled_sign
            or 0.0 in equal_coin_sign
            or len(set(pooled_sign)) != 1
            or len(set(equal_coin_sign)) != 1
            or pooled_sign[0] != equal_coin_sign[0]
        ):
            continue
        sign = int(pooled_sign[0])
        positive_fraction = group["equal_coin_positive_fraction"]
        member_agreement = (
            positive_fraction.ge(minimum_positive_fraction).all()
            if sign > 0
            else positive_fraction.le(maximum_positive_fraction).all()
        )
        if not member_agreement:
            continue
        loo_mask = np.ones(len(leave_one_out), dtype=bool)
        for column, value in zip(GROUP_KEYS, key, strict=True):
            loo_mask &= leave_one_out[column].eq(value).to_numpy()
        loo = leave_one_out.loc[loo_mask & leave_one_out["period"].isin(periods)].copy()
        if (
            set(loo["period"].astype(str)) != set(periods)
            or loo.empty
            or not loo["coverage_gate_passed"].all()
            or not (np.sign(loo["equal_coin_delta_median"].to_numpy(dtype=float)) == sign).all()
        ):
            continue
        by_period = {str(row.period): row for row in group.itertuples()}
        early = by_period[periods[0]]
        late = by_period[periods[1]]
        rows.append(
            {
                "scope": scope,
                **dict(zip(GROUP_KEYS, key, strict=True)),
                "effect_sign": "higher" if sign > 0 else "lower",
                "early_equal_coin_delta_median": float(early.equal_coin_delta_median),
                "late_equal_coin_delta_median": float(late.equal_coin_delta_median),
                "early_pooled_event_delta_mean": float(early.pooled_event_delta_mean),
                "late_pooled_event_delta_mean": float(late.pooled_event_delta_mean),
                "early_positive_coin_fraction": float(early.equal_coin_positive_fraction),
                "late_positive_coin_fraction": float(late.equal_coin_positive_fraction),
                "early_independent_event_pairs": int(early.independent_event_pairs),
                "late_independent_event_pairs": int(late.independent_event_pairs),
                "leave_one_out_rows": len(loo),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
