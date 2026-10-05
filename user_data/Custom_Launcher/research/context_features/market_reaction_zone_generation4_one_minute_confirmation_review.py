from __future__ import annotations

# This is descriptive post-processing of the immutable G4A confirmation replay.
# It does not change selection, controls, labels, thresholds, or acceptance rules.
# ruff: noqa: E402
import argparse
import json
import math
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_one_minute_replay import (  # noqa: E501
    artifact_record,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_breadth import (  # noqa: E501
    REPORT_ROOT,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIRMATION_RUN_ID = "g4a_thin_lvn_1m_unconditioned_confirmation_20260814b"
PRIMARY_SCOPES = ("normal_excluding_btc", "meme")
REPORT_SCOPES = (*PRIMARY_SCOPES, "btc_only")
REPORT_PHASES = (
    "development",
    "validation_early",
    "validation_late",
    "validation_combined",
)
BOOL_COLUMNS = (
    "reaction_60m",
    "first_path_away",
    "first_path_through",
    "first_zone_rejection",
    "first_zone_breakout",
    "direction_callable",
    "simple_trend_direction_correct",
    "simple_trend_joint_reaction_and_direction",
    "control_balance_usable",
)
NUMERIC_COLUMNS = (
    "excursion_strength_60_half_widths",
    "volume_ratio_post_pre_60m",
    "pressure_change_60m",
    "pressure_change_toward_through_60m",
    "pressure_change_toward_away_60m",
    "realized_volatility_ratio_60m",
    "zone_overlap_fraction_60m",
    "reference_crossings_60m",
    "through_excursion_60_half_widths",
    "away_excursion_60_half_widths",
    "control_state_distance_mean",
    "control_state_distance_max",
    "causal_level_contacts_at_event",
)
PAIRED_DIFFERENCE_COLUMNS = (
    "excursion_strength_60_half_widths",
    "volume_ratio_post_pre_60m",
    "pressure_change_60m",
    "pressure_change_toward_through_60m",
    "pressure_change_toward_away_60m",
    "realized_volatility_ratio_60m",
    "zone_overlap_fraction_60m",
    "reference_crossings_60m",
    "through_excursion_60_half_widths",
    "away_excursion_60_half_widths",
)
STRATUM_COLUMNS = (
    "source_timeframe",
    "level_name",
    "cluster_causal_anchor_timeframe",
    "cluster_primary_peer_level_count",
    "cluster_primary_dependency_group_count",
    "cluster_primary_timeframe_count",
    "cluster_dependency_groups",
    "cluster_timeframes",
    "cluster_families",
    "cluster_price_average_family_count",
    "cluster_rolling_price_extreme_count",
    "cluster_round_number_count",
    "cluster_peer_1h_count",
    "cluster_peer_4h_count",
    "cluster_peer_8h_count",
    "has_bollinger_peer",
    "has_prior_range_peer",
    "has_round_number_peer",
    "has_higher_timeframe_peer",
    "has_daily_peer",
    "pair_x_level_name",
    "level_name_x_anchor_timeframe",
    "level_name_x_cluster_timeframes",
    "level_name_x_cluster_families",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the corrected G4A result is broad across pairs, periods, "
            "level sides, cluster timeframes, and continuous reaction measurements."
        )
    )
    parser.add_argument(
        "--confirmation-run-id",
        default=DEFAULT_CONFIRMATION_RUN_ID,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return review_confirmation(
        confirmation_run_id=args.confirmation_run_id,
        overwrite=bool(args.overwrite),
    )


def review_confirmation(*, confirmation_run_id: str, overwrite: bool) -> int:
    run_dir = REPORT_ROOT / confirmation_run_id
    input_record_path = run_dir / "g4a_confirmation_record.json"
    review_record_path = run_dir / "g4a_confirmation_review_record.json"
    if review_record_path.is_file() and not overwrite:
        raise FileExistsError(f"Review already exists: {review_record_path}")

    input_record, replay_path = validated_confirmation_input(input_record_path)
    request = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "confirmation_run_id": confirmation_run_id,
        "confirmation_record_sha256": sha256_file(input_record_path),
        "event_replay_sha256": sha256_file(replay_path),
        "purpose": (
            "Descriptive robustness review of the already frozen G4A confirmation. "
            "No feature selection or new decision threshold is performed."
        ),
        "primary_scopes": list(PRIMARY_SCOPES),
        "btc_is_descriptive_only": True,
        "direction_prediction": "unchanged_fixed_recent_60m_return_sign_comparator",
        "profit_optimization": False,
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "confirmation_run_id": confirmation_run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "request_contract": request,
    }
    atomic_write_json(record, review_record_path)
    try:
        replay = normalized_replay(pd.read_csv(replay_path))
        actual, paired = actual_and_balanced_pairs(replay)
        scope_phase = scope_phase_summary(actual, paired)
        pair_summary = per_pair_summary(actual, paired)
        leave_one_out = leave_one_pair_out_summary(actual, paired)
        strata = stratum_summary(actual, paired)

        output_paths = {
            "scope_phase_csv": run_dir / "g4a_confirmation_review_scope_phase.csv",
            "pair_summary_csv": run_dir / "g4a_confirmation_review_pairs.csv",
            "leave_one_pair_out_csv": (run_dir / "g4a_confirmation_review_leave_one_pair_out.csv"),
            "strata_csv": run_dir / "g4a_confirmation_review_strata.csv",
            "summary_json": run_dir / "g4a_confirmation_review_summary.json",
        }
        atomic_write_csv(scope_phase, output_paths["scope_phase_csv"])
        atomic_write_csv(pair_summary, output_paths["pair_summary_csv"])
        atomic_write_csv(leave_one_out, output_paths["leave_one_pair_out_csv"])
        atomic_write_csv(strata, output_paths["strata_csv"])

        summary = build_review_summary(
            input_record=input_record,
            actual=actual,
            paired=paired,
            scope_phase=scope_phase,
            pair_summary=pair_summary,
            leave_one_out=leave_one_out,
        )
        atomic_write_json(summary, output_paths["summary_json"])
        record.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "input_actual_episode_count": len(actual),
                "balanced_pair_count": len(paired),
                "artifacts": {name: artifact_record(path) for name, path in output_paths.items()},
                "summary": summary,
            }
        )
        atomic_write_json(record, review_record_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(record, review_record_path)
        raise


def validated_confirmation_input(record_path: Path) -> tuple[dict[str, Any], Path]:
    if not record_path.is_file():
        raise FileNotFoundError(f"Missing confirmation record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"Confirmation is not complete: {record.get('status')}")
    replay = record.get("artifacts", {}).get("event_replay_csv", {})
    replay_path = Path(str(replay.get("path", "")))
    if not replay_path.is_file():
        raise FileNotFoundError(f"Missing immutable event replay: {replay_path}")
    if sha256_file(replay_path) != replay.get("sha256"):
        raise ValueError("Immutable event replay hash does not match its record")
    return record, replay_path


def normalized_replay(frame: DataFrame) -> DataFrame:
    required = {
        "episode_id",
        "cohort",
        "period",
        "pair",
        "event_kind",
        *BOOL_COLUMNS,
        *NUMERIC_COLUMNS,
        *STRATUM_COLUMNS[:15],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Replay is missing required columns: {missing}")
    output = frame.copy()
    for column in BOOL_COLUMNS:
        output[column] = output[column].map(normalize_bool)
        if output[column].isna().any():
            raise ValueError(f"Replay contains invalid booleans in {column}")
        output[column] = output[column].astype(bool)
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["scope"] = scope_labels(output)
    output["phase"] = phase_labels(output["period"])
    families = output["cluster_families"].fillna("").astype(str)
    output["has_bollinger_peer"] = families.str.contains("generic_bollinger", regex=False)
    output["has_prior_range_peer"] = families.str.contains("generic_prior_range", regex=False)
    output["has_round_number_peer"] = families.str.contains("generic_round_number", regex=False)
    output["has_higher_timeframe_peer"] = pd.to_numeric(
        output["parent_state_peer_higher_tf_count"], errors="coerce"
    ).gt(0)
    output["has_daily_peer"] = pd.to_numeric(
        output["parent_state_peer_daily_count"], errors="coerce"
    ).gt(0)
    output["pair_x_level_name"] = (
        output["pair"].astype(str) + " | " + output["level_name"].astype(str)
    )
    output["level_name_x_anchor_timeframe"] = (
        output["level_name"].astype(str)
        + " | "
        + output["cluster_causal_anchor_timeframe"].astype(str)
    )
    output["level_name_x_cluster_timeframes"] = (
        output["level_name"].astype(str) + " | " + output["cluster_timeframes"].astype(str)
    )
    output["level_name_x_cluster_families"] = (
        output["level_name"].astype(str) + " | " + output["cluster_families"].astype(str)
    )
    return output


def normalize_bool(value: Any) -> bool | float:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if value in (0, 1):
        return bool(value)
    return math.nan


def scope_labels(frame: DataFrame) -> Series:
    labels = pd.Series("outside_report_scope", index=frame.index, dtype="object")
    labels.loc[frame["cohort"].eq("meme")] = "meme"
    labels.loc[frame["cohort"].eq("normal") & ~frame["pair"].eq("BTC/USDT:USDT")] = (
        "normal_excluding_btc"
    )
    labels.loc[frame["pair"].eq("BTC/USDT:USDT")] = "btc_only"
    return labels


def phase_labels(periods: Series) -> Series:
    text = periods.astype(str)
    labels = pd.Series("unknown", index=periods.index, dtype="object")
    labels.loc[text.str.contains("development", regex=False)] = "development"
    labels.loc[text.str.contains("validation_early", regex=False)] = "validation_early"
    labels.loc[text.str.contains("validation_late", regex=False)] = "validation_late"
    return labels


def actual_and_balanced_pairs(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    actual = frame.loc[frame["event_kind"].eq("actual_cluster_contact")].copy()
    controls = frame.loc[
        frame["event_kind"].eq("same_state_no_level") & frame["control_balance_usable"].eq(True)
    ].copy()
    if actual["episode_id"].duplicated().any():
        raise ValueError("Actual replay contains duplicate episode IDs")
    if controls["episode_id"].duplicated().any():
        raise ValueError("Balanced controls contain duplicate episode IDs")
    control_columns = [
        "episode_id",
        "pair",
        "reaction_60m",
        *PAIRED_DIFFERENCE_COLUMNS,
        "control_state_distance_mean",
        "control_state_distance_max",
        "causal_level_contacts_at_event",
    ]
    renamed = controls[control_columns].rename(
        columns={
            column: f"control_{column}" for column in control_columns if column != "episode_id"
        }
    )
    paired = actual.merge(renamed, on="episode_id", how="inner", validate="one_to_one")
    if not paired["pair"].eq(paired["control_pair"]).all():
        raise ValueError("A balanced control does not use the same pair as its actual event")
    if not paired["control_causal_level_contacts_at_event"].eq(0).all():
        raise ValueError("A balanced no-level control contains a causal selected-level contact")
    return actual, paired


def scope_phase_summary(actual: DataFrame, paired: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in REPORT_SCOPES:
        for phase in REPORT_PHASES:
            selected_actual = select_scope_phase(actual, scope=scope, phase=phase)
            selected_paired = paired.loc[paired["episode_id"].isin(selected_actual["episode_id"])]
            rows.append(
                {
                    "scope": scope,
                    "phase": phase,
                    **summary_metrics(selected_actual, selected_paired),
                }
            )
    return DataFrame(rows)


def per_pair_summary(actual: DataFrame, paired: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in REPORT_SCOPES:
        for phase in REPORT_PHASES:
            scoped = select_scope_phase(actual, scope=scope, phase=phase)
            for pair in sorted(scoped["pair"].unique()):
                selected_actual = scoped.loc[scoped["pair"].eq(pair)]
                selected_paired = paired.loc[
                    paired["episode_id"].isin(selected_actual["episode_id"])
                ]
                rows.append(
                    {
                        "scope": scope,
                        "phase": phase,
                        "pair": pair,
                        **summary_metrics(selected_actual, selected_paired),
                    }
                )
    return DataFrame(rows)


def leave_one_pair_out_summary(actual: DataFrame, paired: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in PRIMARY_SCOPES:
        scoped = select_scope_phase(actual, scope=scope, phase="validation_combined")
        scoped_paired = paired.loc[paired["episode_id"].isin(scoped["episode_id"])]
        for omitted_pair in sorted(scoped["pair"].unique()):
            selected_actual = scoped.loc[~scoped["pair"].eq(omitted_pair)]
            selected_paired = scoped_paired.loc[~scoped_paired["pair"].eq(omitted_pair)]
            metrics = summary_metrics(selected_actual, selected_paired)
            rows.append(
                {
                    "scope": scope,
                    "phase": "validation_combined",
                    "omitted_pair": omitted_pair,
                    "remaining_pair_count": int(selected_actual["pair"].nunique()),
                    **metrics,
                    "paired_reaction_uplift_positive": bool(metrics["paired_reaction_uplift"] > 0),
                }
            )
    return DataFrame(rows)


def stratum_summary(actual: DataFrame, paired: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in REPORT_SCOPES:
        for phase in ("validation_early", "validation_late", "validation_combined"):
            scoped = select_scope_phase(actual, scope=scope, phase=phase)
            for dimension in STRATUM_COLUMNS:
                values = scoped[dimension].fillna("missing").astype(str)
                for value in sorted(values.unique()):
                    selected_actual = scoped.loc[values.eq(value)]
                    selected_paired = paired.loc[
                        paired["episode_id"].isin(selected_actual["episode_id"])
                    ]
                    rows.append(
                        {
                            "scope": scope,
                            "phase": phase,
                            "dimension": dimension,
                            "value": value,
                            "descriptive_only": True,
                            **summary_metrics(selected_actual, selected_paired),
                        }
                    )
    return DataFrame(rows)


def select_scope_phase(frame: DataFrame, *, scope: str, phase: str) -> DataFrame:
    selected = frame.loc[frame["scope"].eq(scope)]
    if phase == "validation_combined":
        return selected.loc[selected["phase"].isin(("validation_early", "validation_late"))]
    return selected.loc[selected["phase"].eq(phase)]


def summary_metrics(actual: DataFrame, paired: DataFrame) -> dict[str, Any]:
    callable_rows = actual.loc[actual["direction_callable"].eq(True)]
    reaction_callable_rows = callable_rows.loc[callable_rows["reaction_60m"].eq(True)]
    direction_counts = callable_rows["first_direction"].value_counts()
    majority = (
        float(direction_counts.max() / len(callable_rows)) if len(callable_rows) else math.nan
    )
    reaction_direction_counts = reaction_callable_rows["first_direction"].value_counts()
    reaction_majority = (
        float(reaction_direction_counts.max() / len(reaction_callable_rows))
        if len(reaction_callable_rows)
        else math.nan
    )
    conditional_direction_accuracy = safe_mean(
        reaction_callable_rows["simple_trend_direction_correct"]
    )
    output: dict[str, Any] = {
        "actual_episode_count": len(actual),
        "actual_pair_count": int(actual["pair"].nunique()),
        "actual_reaction_count": int(actual["reaction_60m"].sum()),
        "actual_reaction_rate": safe_mean(actual["reaction_60m"]),
        "direction_callable_count": len(callable_rows),
        "direction_abstention_count": len(actual) - len(callable_rows),
        "direction_call_coverage": len(callable_rows) / len(actual) if len(actual) else math.nan,
        "simple_trend_direction_accuracy": safe_mean(
            callable_rows["simple_trend_direction_correct"]
        ),
        "retrospective_majority_direction_accuracy": majority,
        "trend_minus_majority_accuracy": (
            safe_mean(callable_rows["simple_trend_direction_correct"]) - majority
            if np.isfinite(majority)
            else math.nan
        ),
        "reaction_direction_callable_count": len(reaction_callable_rows),
        "conditional_direction_accuracy_given_reaction": conditional_direction_accuracy,
        "retrospective_reaction_majority_direction_accuracy": reaction_majority,
        "conditional_trend_minus_reaction_majority_accuracy": (
            conditional_direction_accuracy - reaction_majority
            if np.isfinite(conditional_direction_accuracy) and np.isfinite(reaction_majority)
            else math.nan
        ),
        "joint_reaction_direction_count": int(
            actual["simple_trend_joint_reaction_and_direction"].sum()
        ),
        "joint_reaction_direction_rate": safe_mean(
            actual["simple_trend_joint_reaction_and_direction"]
        ),
        "first_through_count": int(actual["first_path_through"].sum()),
        "first_away_count": int(actual["first_path_away"].sum()),
        "zone_breakout_count": int(actual["first_zone_breakout"].sum()),
        "zone_rejection_count": int(actual["first_zone_rejection"].sum()),
    }
    output.update(paired_metrics(paired))
    return output


def paired_metrics(paired: DataFrame) -> dict[str, Any]:
    if paired.empty:
        output: dict[str, Any] = {
            "balanced_pair_count": 0,
            "paired_actual_reaction_rate": math.nan,
            "paired_control_reaction_rate": math.nan,
            "paired_reaction_uplift": math.nan,
            "actual_only_reaction_count": 0,
            "control_only_reaction_count": 0,
            "both_reacted_count": 0,
            "neither_reacted_count": 0,
            "control_state_distance_mean_median": math.nan,
            "control_state_distance_max_median": math.nan,
        }
        for column in PAIRED_DIFFERENCE_COLUMNS:
            output[f"{column}_difference_median"] = math.nan
            output[f"{column}_actual_greater_fraction"] = math.nan
        output["absolute_pressure_change_difference_median"] = math.nan
        output["absolute_pressure_change_actual_greater_fraction"] = math.nan
        return output

    actual_reaction = paired["reaction_60m"].astype(bool)
    control_reaction = paired["control_reaction_60m"].astype(bool)
    output = {
        "balanced_pair_count": len(paired),
        "paired_actual_reaction_rate": float(actual_reaction.mean()),
        "paired_control_reaction_rate": float(control_reaction.mean()),
        "paired_reaction_uplift": float(actual_reaction.mean() - control_reaction.mean()),
        "actual_only_reaction_count": int((actual_reaction & ~control_reaction).sum()),
        "control_only_reaction_count": int((~actual_reaction & control_reaction).sum()),
        "both_reacted_count": int((actual_reaction & control_reaction).sum()),
        "neither_reacted_count": int((~actual_reaction & ~control_reaction).sum()),
        "control_state_distance_mean_median": safe_median(
            paired["control_control_state_distance_mean"]
        ),
        "control_state_distance_max_median": safe_median(
            paired["control_control_state_distance_max"]
        ),
    }
    for column in PAIRED_DIFFERENCE_COLUMNS:
        difference = paired[column] - paired[f"control_{column}"]
        output[f"{column}_difference_median"] = safe_median(difference)
        output[f"{column}_actual_greater_fraction"] = safe_mean(difference.gt(0))
    absolute_pressure_difference = (
        paired["pressure_change_60m"].abs() - paired["control_pressure_change_60m"].abs()
    )
    output["absolute_pressure_change_difference_median"] = safe_median(absolute_pressure_difference)
    output["absolute_pressure_change_actual_greater_fraction"] = safe_mean(
        absolute_pressure_difference.gt(0)
    )
    return output


def build_review_summary(
    *,
    input_record: dict[str, Any],
    actual: DataFrame,
    paired: DataFrame,
    scope_phase: DataFrame,
    pair_summary: DataFrame,
    leave_one_out: DataFrame,
) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    for scope in PRIMARY_SCOPES:
        row = scope_phase.loc[
            scope_phase["scope"].eq(scope) & scope_phase["phase"].eq("validation_combined")
        ].iloc[0]
        pair_rows = pair_summary.loc[
            pair_summary["scope"].eq(scope)
            & pair_summary["phase"].eq("validation_combined")
            & pair_summary["balanced_pair_count"].gt(0)
        ]
        loo = leave_one_out.loc[leave_one_out["scope"].eq(scope)]
        primary[scope] = {
            "fixed_confirmation_retained": bool(
                input_record["summary"]["primary_scope_results"][scope]["retained"]
            ),
            "validation_combined": serializable_row(row),
            "balanced_member_pair_rows": len(pair_rows),
            "member_pairs_with_positive_reaction_uplift": int(
                pair_rows["paired_reaction_uplift"].gt(0).sum()
            ),
            "member_pairs_with_zero_reaction_uplift": int(
                pair_rows["paired_reaction_uplift"].eq(0).sum()
            ),
            "member_pairs_with_negative_reaction_uplift": int(
                pair_rows["paired_reaction_uplift"].lt(0).sum()
            ),
            "leave_one_pair_out_count": len(loo),
            "leave_one_pair_out_positive_uplift_count": int(
                loo["paired_reaction_uplift"].gt(0).sum()
            ),
            "leave_one_pair_out_minimum_uplift": safe_min(loo["paired_reaction_uplift"]),
            "leave_one_pair_out_maximum_uplift": safe_max(loo["paired_reaction_uplift"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "fixed_confirmation_classification": input_record["classification"],
        "review_changed_fixed_classification": False,
        "actual_episode_count": len(actual),
        "balanced_pair_count": len(paired),
        "primary_scope_review": primary,
        "input_roles": {
            "selected_level": "causal one-hour high-thinness Volume Profile LVN",
            "cluster_components": (
                "causally connected generic Bollinger, prior-range, and round-number "
                "levels, preserving exact source timeframes"
            ),
            "control_matching_inputs": [
                "15m return",
                "60m return",
                "240m return",
                "60m realized volatility",
                "one-minute ATR(14) divided by price",
                "last-completed-minute volume / trailing 20m mean",
                "20m volume-weighted candle-body pressure",
                "position inside the preceding 240m range",
            ],
            "tested_outputs": [
                "60m reaction label",
                "price excursion",
                "post/pre volume",
                "raw and approach-oriented pressure change",
                "realized volatility change",
                "zone residence",
                "centre crossings",
                "through and away excursion",
                "breakout and rejection",
                "recent-60m-trend immediate-path comparator",
            ],
            "recorded_but_not_screened_here": [
                "RSI(14)",
                "Bollinger position and width",
                "MACD histogram / ATR",
                "EMA(50) gap / ATR",
                "BTC returns and volatility",
                "top-10 breadth, dispersion, and common volume",
                "level-density and Volume Profile state fields",
            ],
            "indicator_crosses_tested": [],
            "external_inputs_tested": [],
        },
        "interpretation_limits": [
            "This review is descriptive post-processing, not a new feature search.",
            "Sparse pair and stratum cells do not create new pass rules or branches.",
            "The reaction label includes a 1.5x post/pre volume condition and is not profit.",
            "BTC has only four validation events and remains descriptive.",
            "The immutable fixed confirmation classification remains authoritative.",
        ],
    }


def serializable_row(row: Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.bool_, bool)):
            output[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            output[key] = float(value) if np.isfinite(value) else None
        else:
            output[key] = value
    return output


def safe_mean(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else math.nan


def safe_median(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.median()) if numeric.notna().any() else math.nan


def safe_min(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min()) if numeric.notna().any() else math.nan


def safe_max(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
