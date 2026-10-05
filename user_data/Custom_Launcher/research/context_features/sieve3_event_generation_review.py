"""Build the batch-gated Generation 0 Sieve3 event-reaction review surfaces.

The review combines already-approved compact outputs.  It does not rerun models,
alter Sieve definitions, promote strategies, or launch branch experiments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame


TERMINAL_EVIDENCE_STATES = {
    "completed",
    "completed_insufficient_evidence",
    "unsupported_in_tested_scope",
    "deferred_for_data",
    "blocked_integrity_defect",
}
LEVEL_METRICS = (
    "signed_return",
    "best_move_if_held_atr",
    "worst_move_if_held_atr",
    "path_balance_atr",
    "reaction_magnitude_atr",
    "volume_ratio",
    "pressure_alignment",
    "volatility_ratio",
    "target_before_invalidation",
    "first_1atr_touch_step",
)
PORTABILITY_COLUMNS = [
    "domain",
    "candidate_id",
    "family",
    "side",
    "scope",
    "horizon_hours",
    "metric",
    "comparator_primary",
    "tests",
    "distinct_pairs",
    "distinct_windows",
    "sample_rows",
    "primary_median_delta",
    "primary_positive_share",
    "comparator_secondary",
    "secondary_median_delta",
    "secondary_positive_share",
    "tertiary_median_delta",
    "tertiary_positive_share",
    "evidence_shape",
    "classification",
    "source_artifact",
    "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if len(values) else None


def positive_share(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.gt(0).mean()) if len(values) else None


def negative_share(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.lt(0).mean()) if len(values) else None


def as_int_sum(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def atomic_csv(frame: DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def require_generation_gate(manifest: dict[str, Any]) -> None:
    batches = {item["id"]: item for item in manifest.get("batches", [])}
    required = [f"G0-B{index}" for index in range(1, 7)]
    missing = [batch for batch in required if batch not in batches]
    if missing:
        raise ValueError(f"Generation manifest is missing evidence batches: {missing}")
    nonterminal = {
        batch: batches[batch].get("status")
        for batch in required
        if batches[batch].get("status") not in TERMINAL_EVIDENCE_STATES
    }
    if nonterminal:
        raise ValueError(f"Generation review gate is closed; nonterminal batches: {nonterminal}")


def direct_entry_rows(base: Path) -> DataFrame:
    source = base / "direct_entry_portability_summary.csv"
    frame = pd.read_csv(source)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        shape = str(record.get("descriptive_evidence_shape"))
        precursor = str(record.get("precursor_relation"))
        metric = str(record["metric"])
        tests = int(record["pair_window_tests"])
        samples = int(record["summed_event_rows"])
        samples_per_test = samples / tests if tests else 0.0
        isolated = bool(record.get("isolation_warning")) or bool(
            record.get("single_window_warning")
        )
        if isolated:
            classification = "isolated_or_sparse"
        elif samples < 20:
            classification = "insufficient_event_support"
        elif samples < 50 or samples_per_test < 3:
            if shape == "broad_positive_lead" and precursor == "trigger_stronger_than_precursor":
                classification = "sparse_trigger_timing_lead"
            elif shape == "broad_positive_lead" and precursor == "precursor_stronger_than_trigger":
                classification = "sparse_precursor_or_regime_lead"
            else:
                classification = "sparse_or_context_dependent"
        elif shape == "broad_positive_lead" and precursor == "trigger_stronger_than_precursor":
            if metric in {"volume_ratio", "volatility_ratio", "reaction_magnitude_atr"}:
                classification = "direct_trigger_activity_lead"
            else:
                classification = "direct_trigger_directional_lead"
        elif shape == "broad_positive_lead" and precursor == "precursor_stronger_than_trigger":
            classification = "precursor_or_regime_lead"
        elif shape == "broad_positive_lead":
            classification = "direct_positive_context_lead"
        elif shape == "broad_negative_lead":
            classification = "direct_negative_or_contrarian_lead"
        else:
            classification = "mixed_or_context_dependent"
        rows.append(
            {
                "domain": "direct_entry",
                "candidate_id": record["entry_id"],
                "family": record["family"],
                "side": record["side"],
                "scope": record["entry_timeframe"],
                "horizon_hours": record["horizon_hours"],
                "metric": record["metric"],
                "comparator_primary": "matched_visible_state",
                "tests": tests,
                "distinct_pairs": record["distinct_pairs"],
                "distinct_windows": record["distinct_windows"],
                "sample_rows": samples,
                "primary_median_delta": record["matched_median_delta"],
                "primary_positive_share": record["matched_positive_share"],
                "comparator_secondary": "shifted_168h_placebo",
                "secondary_median_delta": record["placebo_median_delta"],
                "secondary_positive_share": record["placebo_positive_share"],
                "tertiary_median_delta": record["precursor_median_delta"],
                "tertiary_positive_share": record["precursor_positive_share"],
                "evidence_shape": shape,
                "classification": classification,
                "source_artifact": str(source),
                "notes": (
                    f"precursor_relation={precursor}; samples_per_test={samples_per_test}; "
                    f"native_horizon_multiple={record.get('native_horizon_multiple')}"
                ),
            }
        )
    return DataFrame(rows, columns=PORTABILITY_COLUMNS)


def _majority_count(frame: DataFrame, group: str, column: str, threshold: float) -> int:
    return int(frame.groupby(group, dropna=False)[column].mean().ge(threshold).sum())


def freqai_rows(base: Path) -> DataFrame:
    source = base / "freqai" / "freqai_reaction_scores.csv"
    columns = [
        "profile_id",
        "pair",
        "window",
        "scope",
        "target",
        "rows",
        "status",
        "target_family",
        "horizon_hours",
        "prediction_actual_spearman_delta_vs_price",
        "prediction_actual_spearman_delta_vs_event_placebo",
        "mean_absolute_error_skill_vs_price",
        "mean_absolute_error_skill_vs_event_placebo",
        "root_mean_squared_error_skill_vs_price",
        "root_mean_squared_error_skill_vs_event_placebo",
        "r_squared_delta_vs_price",
        "r_squared_delta_vs_event_placebo",
    ]
    frame = pd.read_csv(source, usecols=columns, low_memory=False)
    frame = frame[
        frame["profile_id"].eq("event_identity")
        & frame["status"].eq("scored")
        & frame["scope"].isin(["all", "any_entry_onset"])
    ].copy()
    for column in columns[9:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["rank_both_positive"] = (
        frame["prediction_actual_spearman_delta_vs_price"].gt(0)
        & frame["prediction_actual_spearman_delta_vs_event_placebo"].gt(0)
    )
    frame["mae_both_positive"] = (
        frame["mean_absolute_error_skill_vs_price"].gt(0)
        & frame["mean_absolute_error_skill_vs_event_placebo"].gt(0)
    )
    frame["rank_and_mae_both_positive"] = (
        frame["rank_both_positive"] & frame["mae_both_positive"]
    )
    rows: list[dict[str, Any]] = []
    grouping = ["scope", "target", "target_family", "horizon_hours"]
    for keys, group in frame.groupby(grouping, dropna=False, observed=True):
        scope, target, family, horizon = keys
        rank_share = float(group["rank_both_positive"].mean())
        mae_share = float(group["mae_both_positive"].mean())
        joint_share = float(group["rank_and_mae_both_positive"].mean())
        rank_price = finite_median(group["prediction_actual_spearman_delta_vs_price"])
        rank_placebo = finite_median(group["prediction_actual_spearman_delta_vs_event_placebo"])
        mae_price = finite_median(group["mean_absolute_error_skill_vs_price"])
        mae_placebo = finite_median(group["mean_absolute_error_skill_vs_event_placebo"])
        pair_majority = _majority_count(
            group, "pair", "rank_and_mae_both_positive", 2 / 3
        )
        window_majority = _majority_count(
            group, "window", "rank_and_mae_both_positive", 0.6
        )
        support_ok = group["pair"].nunique() >= 7 and group["window"].nunique() >= 2
        rank_ok = (
            rank_price is not None
            and rank_placebo is not None
            and rank_price > 0
            and rank_placebo > 0
            and rank_share >= 0.65
        )
        mae_ok = (
            mae_price is not None
            and mae_placebo is not None
            and mae_price > 0
            and mae_placebo > 0
            and mae_share >= 0.65
        )
        portable_ok = pair_majority >= 7 and window_majority >= 2
        if support_ok and rank_ok and mae_ok and joint_share >= 0.60 and portable_ok:
            classification = "repeatable_incremental_regression_lead"
        elif support_ok and rank_ok and not mae_ok:
            classification = "ranking_only_lead"
        elif support_ok and mae_ok and not rank_ok:
            classification = "magnitude_error_only_lead"
        elif (
            rank_price is not None
            and rank_placebo is not None
            and mae_price is not None
            and mae_placebo is not None
            and max(rank_price, rank_placebo, mae_price, mae_placebo) <= 0
        ):
            classification = "inverse_or_redundant_in_tested_scope"
        else:
            classification = "mixed_or_too_small"
        rows.append(
            {
                "domain": "freqai_incremental",
                "candidate_id": target,
                "family": family,
                "side": "side_agnostic_target",
                "scope": scope,
                "horizon_hours": horizon,
                "metric": "rank_and_magnitude_prediction",
                "comparator_primary": "price_only_control",
                "tests": len(group),
                "distinct_pairs": group["pair"].nunique(),
                "distinct_windows": group["window"].nunique(),
                "sample_rows": as_int_sum(group["rows"]),
                "primary_median_delta": rank_price,
                "primary_positive_share": rank_share,
                "comparator_secondary": "shifted_168h_event_placebo",
                "secondary_median_delta": rank_placebo,
                "secondary_positive_share": mae_share,
                "tertiary_median_delta": mae_price,
                "tertiary_positive_share": joint_share,
                "evidence_shape": (
                    f"pair_joint_majorities={pair_majority}; "
                    f"window_joint_majorities={window_majority}"
                ),
                "classification": classification,
                "source_artifact": str(source),
                "notes": (
                    f"median_mae_skill_vs_placebo={mae_placebo}; "
                    f"median_rmse_skill_vs_price="
                    f"{finite_median(group['root_mean_squared_error_skill_vs_price'])}; "
                    f"median_rmse_skill_vs_placebo="
                    f"{finite_median(group['root_mean_squared_error_skill_vs_event_placebo'])}; "
                    f"median_r2_delta_vs_price="
                    f"{finite_median(group['r_squared_delta_vs_price'])}; "
                    f"median_r2_delta_vs_placebo="
                    f"{finite_median(group['r_squared_delta_vs_event_placebo'])}"
                ),
            }
        )
    return DataFrame(rows, columns=PORTABILITY_COLUMNS)


def level_rows(base: Path) -> DataFrame:
    repaired = base / "level_relationships_repaired" / "entry_level_reaction_summary.csv"
    source = repaired if repaired.is_file() else (
        base / "level_relationships" / "entry_level_reaction_summary.csv"
    )
    key_columns = [
        "evidence_source",
        "entry_id",
        "family",
        "entry_side",
        "window",
        "pair_scope",
        "proximity_band_pct",
        "relationship_state",
        "mtf_precedence_state",
        "event_rows",
        "horizon_hours",
    ]
    metric_columns = [
        item
        for metric in LEVEL_METRICS
        for item in (f"{metric}_rows", f"{metric}_mean")
    ]
    frame = pd.read_csv(source, usecols=[*key_columns, *metric_columns], low_memory=False)
    frame = frame[~frame["pair_scope"].eq("pooled")].copy()
    exact_key = [
        "entry_id",
        "family",
        "entry_side",
        "window",
        "pair_scope",
        "proximity_band_pct",
        "relationship_state",
        "mtf_precedence_state",
        "horizon_hours",
    ]
    baseline_key = [
        "entry_id",
        "family",
        "entry_side",
        "window",
        "pair_scope",
        "proximity_band_pct",
        "horizon_hours",
    ]
    output: list[dict[str, Any]] = []
    for metric in LEVEL_METRICS:
        mean_column = f"{metric}_mean"
        rows_column = f"{metric}_rows"
        relevant = frame[[*exact_key, "evidence_source", rows_column, mean_column]].copy()
        actual = relevant[
            relevant["evidence_source"].eq("contemporaneous_levels")
        ].drop(columns="evidence_source")
        stale = relevant[
            relevant["evidence_source"].eq("stale_168h_levels")
        ].drop(columns="evidence_source").rename(
            columns={rows_column: "stale_rows", mean_column: "stale_mean"}
        )
        if actual.duplicated(exact_key).any() or stale.duplicated(exact_key).any():
            raise ValueError(f"Duplicate level relationship key for metric {metric}")
        baseline = actual[actual["relationship_state"].eq("no_nearby_level")][
            [*baseline_key, mean_column]
        ].rename(columns={mean_column: "no_near_mean"})
        if baseline.duplicated(baseline_key).any():
            baseline = (
                baseline.groupby(baseline_key, dropna=False, observed=True)["no_near_mean"]
                .median()
                .reset_index()
            )
        contrasts = actual[
            ~actual["relationship_state"].eq("no_nearby_level")
        ].merge(baseline, on=baseline_key, how="left", validate="many_to_one")
        contrasts = contrasts.merge(stale, on=exact_key, how="left", validate="one_to_one")
        contrasts["delta_vs_no_near"] = pd.to_numeric(
            contrasts[mean_column], errors="coerce"
        ) - pd.to_numeric(contrasts["no_near_mean"], errors="coerce")
        contrasts["delta_vs_stale"] = pd.to_numeric(
            contrasts[mean_column], errors="coerce"
        ) - pd.to_numeric(contrasts["stale_mean"], errors="coerce")
        contrasts["paired_rows"] = np.minimum(
            pd.to_numeric(contrasts[rows_column], errors="coerce"),
            pd.to_numeric(contrasts["stale_rows"], errors="coerce"),
        )
        grouping = [
            "relationship_state",
            "mtf_precedence_state",
            "proximity_band_pct",
            "horizon_hours",
        ]
        for keys, group in contrasts.groupby(grouping, dropna=False, observed=True):
            relationship, mtf_state, band, horizon = keys
            primary_median = finite_median(group["delta_vs_no_near"])
            primary_share = positive_share(group["delta_vs_no_near"])
            secondary_median = finite_median(group["delta_vs_stale"])
            secondary_share = positive_share(group["delta_vs_stale"])
            tests = int(group["delta_vs_no_near"].notna().sum())
            stale_tests = int(group["delta_vs_stale"].notna().sum())
            paired_sample_rows = as_int_sum(group["paired_rows"])
            support_ok = (
                tests >= 20
                and stale_tests >= 20
                and paired_sample_rows >= 100
                and group["entry_id"].nunique() >= 5
                and group["pair_scope"].nunique() == 3
                and group["window"].nunique() >= 3
            )
            if (
                support_ok
                and primary_median is not None
                and secondary_median is not None
                and primary_median > 0
                and secondary_median > 0
                and primary_share is not None
                and secondary_share is not None
                and primary_share >= 0.65
                and secondary_share >= 0.65
            ):
                classification = "coherent_positive_level_context"
            elif (
                support_ok
                and primary_median is not None
                and secondary_median is not None
                and primary_median < 0
                and secondary_median < 0
                and primary_share is not None
                and secondary_share is not None
                and primary_share <= 0.35
                and secondary_share <= 0.35
            ):
                classification = "coherent_negative_level_context"
            elif (
                support_ok
                and primary_share is not None
                and (primary_share >= 0.65 or primary_share <= 0.35)
                and secondary_share is not None
                and 0.35 < secondary_share < 0.65
            ):
                classification = "state_difference_but_stale_equivalent"
            elif not support_ok:
                classification = "insufficient_level_support"
            else:
                classification = "mixed_level_context"
            output.append(
                {
                    "domain": "level_relationship",
                    "candidate_id": f"{relationship}|{mtf_state}|band={band}",
                    "family": relationship,
                    "side": "pooled_sides",
                    "scope": mtf_state,
                    "horizon_hours": horizon,
                    "metric": metric,
                    "comparator_primary": "contemporaneous_no_nearby_level",
                    "tests": tests,
                    "distinct_pairs": group["pair_scope"].nunique(),
                    "distinct_windows": group["window"].nunique(),
                    "sample_rows": paired_sample_rows,
                    "primary_median_delta": primary_median,
                    "primary_positive_share": primary_share,
                    "comparator_secondary": "same_state_stale_168h_levels",
                    "secondary_median_delta": secondary_median,
                    "secondary_positive_share": secondary_share,
                    "tertiary_median_delta": None,
                    "tertiary_positive_share": None,
                    "evidence_shape": (
                        f"distinct_entries={group['entry_id'].nunique()}; "
                        f"paired_stale_tests={stale_tests}; band_pct={band}"
                    ),
                    "classification": classification,
                    "source_artifact": str(source),
                    "notes": (
                        "BTC/ETH/SOL only; higher timeframe is a tested state, "
                        "not automatic precedence"
                    ),
                }
            )
    return DataFrame(output, columns=PORTABILITY_COLUMNS)


def repaired_action_exit_rows(source: Path) -> DataFrame:
    frame = pd.read_csv(source, low_memory=False)
    rows: list[dict[str, Any]] = []
    grouping = [
        "exit_family",
        "entry_side",
        "analysis_reason_category",
        "analysis_action_role",
        "horizon_hours",
    ]
    for keys, group in frame.groupby(grouping, dropna=False, observed=True):
        family, side, reason, role, horizon = keys
        adequate = group[
            group["unique_event_paths"].ge(20)
            & group["unique_event_times"].ge(20)
            & group["control_rows"].ge(20)
        ]
        event_net = pd.to_numeric(
            adequate["event_net_exit_regret_mean"], errors="coerce"
        )
        event_hold = pd.to_numeric(
            adequate["event_hold_instead_delta_mean"], errors="coerce"
        )
        event_missed = pd.to_numeric(
            adequate["event_missed_additional_profit_mean"], errors="coerce"
        )
        event_avoided = pd.to_numeric(
            adequate["event_avoided_loss_after_exit_mean"], errors="coerce"
        )
        matched_delta = pd.to_numeric(
            adequate["event_minus_control_net_exit_regret"], errors="coerce"
        )
        net_median = finite_median(event_net)
        hold_median = finite_median(event_hold)
        matched_median = finite_median(matched_delta)
        net_positive = positive_share(event_net)
        net_negative = negative_share(event_net)
        matched_positive = positive_share(matched_delta)
        matched_negative = negative_share(matched_delta)
        support_ok = (
            adequate["pair"].nunique() >= 3
            and adequate["window"].nunique() >= 2
            and as_int_sum(adequate["unique_event_paths"]) >= 20
        )
        absolute_protective = (
            net_median is not None
            and net_median < 0
            and net_negative is not None
            and net_negative >= 0.65
            and hold_median is not None
            and hold_median <= 0
        )
        absolute_runner = (
            net_median is not None
            and net_median > 0
            and net_positive is not None
            and net_positive >= 0.65
            and hold_median is not None
            and hold_median > 0
        )
        if (
            support_ok
            and absolute_protective
            and matched_median is not None
            and matched_median < 0
            and matched_negative is not None
            and matched_negative >= 0.65
        ):
            classification = "incremental_protective_exit_lead"
        elif (
            support_ok
            and absolute_runner
            and matched_median is not None
            and matched_median > 0
            and matched_positive is not None
            and matched_positive >= 0.65
        ):
            classification = "incremental_runner_or_partial_lead"
        elif support_ok and absolute_protective:
            classification = "protective_but_trade_state_redundant"
        elif support_ok and absolute_runner:
            classification = "runner_but_trade_state_redundant"
        elif (
            support_ok
            and finite_median(event_missed) is not None
            and finite_median(event_avoided) is not None
            and finite_median(event_missed) > 0
            and finite_median(event_avoided) > 0
            and net_positive is not None
            and 0.35 < net_positive < 0.65
        ):
            classification = "two_sided_path_warning"
        elif not support_ok:
            classification = "insufficient_exit_support"
        else:
            classification = "mixed_or_context_dependent_exit"
        candidate = f"{family}|{reason}|{role}"
        rows.append(
            {
                "domain": "exit_counterfactual",
                "candidate_id": candidate,
                "family": family,
                "side": side,
                "scope": f"executed_order:{reason}:{role}",
                "horizon_hours": horizon,
                "metric": "net_exit_regret",
                "comparator_primary": "exact_order_price_missed_minus_avoided",
                "tests": len(adequate),
                "distinct_pairs": adequate["pair"].nunique(),
                "distinct_windows": adequate["window"].nunique(),
                "sample_rows": as_int_sum(adequate["event_rows"]),
                "primary_median_delta": net_median,
                "primary_positive_share": net_positive,
                "comparator_secondary": "hold_instead_of_exit",
                "secondary_median_delta": hold_median,
                "secondary_positive_share": positive_share(event_hold),
                "tertiary_median_delta": matched_median,
                "tertiary_positive_share": matched_positive,
                "evidence_shape": (
                    f"adequate_pair_windows={len(adequate)}; "
                    f"unique_path_sum={as_int_sum(adequate['unique_event_paths'])}; "
                    f"matched_negative_share={matched_negative}"
                ),
                "classification": classification,
                "source_artifact": str(source),
                "notes": (
                    f"median_missed={finite_median(event_missed)}; "
                    f"median_avoided={finite_median(event_avoided)}; "
                    "matched controls are different still-open trade paths in the same family/side/pair/window; outcomes begin with the first complete post-event candle"
                ),
            }
        )
    return DataFrame(rows, columns=PORTABILITY_COLUMNS)


def exit_rows(base: Path) -> DataFrame:
    repaired = (
        base.parent.parent
        / "generation1"
        / "g1-b4"
        / "direct_exit_role_summary.csv"
    )
    if repaired.is_file():
        return repaired_action_exit_rows(repaired)
    source = base / "exit_counterfactual_summary.csv"
    frame = pd.read_csv(source, low_memory=False)
    rows: list[dict[str, Any]] = []
    grouping = ["exit_family", "entry_side", "horizon_hours"]
    for keys, group in frame.groupby(grouping, dropna=False, observed=True):
        family, side, horizon = keys
        net = pd.to_numeric(group["net_exit_regret_median"], errors="coerce")
        hold = pd.to_numeric(group["hold_instead_delta_median"], errors="coerce")
        missed = pd.to_numeric(group["missed_additional_profit_median"], errors="coerce")
        avoided = pd.to_numeric(group["avoided_loss_after_exit_median"], errors="coerce")
        net_median = finite_median(net)
        net_positive = positive_share(net)
        net_negative = negative_share(net)
        hold_median = finite_median(hold)
        hold_positive = positive_share(hold)
        support_ok = (
            group["pair"].nunique() >= 2
            and group["window"].nunique() >= 2
            and as_int_sum(group["unique_trade_paths"]) >= 20
        )
        if (
            support_ok
            and net_median is not None
            and net_median < 0
            and net_negative is not None
            and net_negative >= 0.65
            and hold_median is not None
            and hold_median <= 0
        ):
            classification = "protective_exit_lead"
        elif (
            support_ok
            and net_median is not None
            and net_median > 0
            and net_positive is not None
            and net_positive >= 0.65
            and hold_median is not None
            and hold_median > 0
        ):
            classification = "runner_or_later_exit_lead"
        elif (
            support_ok
            and finite_median(missed) is not None
            and finite_median(avoided) is not None
            and finite_median(missed) > 0
            and finite_median(avoided) > 0
            and net_positive is not None
            and 0.35 < net_positive < 0.65
        ):
            classification = "two_sided_path_warning"
        elif not support_ok:
            classification = "insufficient_exit_support"
        else:
            classification = "mixed_or_context_dependent_exit"
        rows.append(
            {
                "domain": "exit_counterfactual",
                "candidate_id": family,
                "family": family,
                "side": side,
                "scope": "executed_exit_trade_state",
                "horizon_hours": horizon,
                "metric": "net_exit_regret",
                "comparator_primary": "missed_profit_minus_avoided_loss",
                "tests": len(group),
                "distinct_pairs": group["pair"].nunique(),
                "distinct_windows": group["window"].nunique(),
                "sample_rows": as_int_sum(group["exit_rows"]),
                "primary_median_delta": net_median,
                "primary_positive_share": net_positive,
                "comparator_secondary": "hold_instead_of_exit",
                "secondary_median_delta": hold_median,
                "secondary_positive_share": hold_positive,
                "tertiary_median_delta": finite_median(avoided),
                "tertiary_positive_share": finite_median(missed),
                "evidence_shape": (
                    f"negative_net_regret_share={net_negative}; "
                    f"distinct_entries={group['entry_id'].nunique()}; "
                    f"unique_trade_path_sum={as_int_sum(group['unique_trade_paths'])}"
                ),
                "classification": classification,
                "source_artifact": str(source),
                "notes": (
                    f"median_missed_additional_profit={finite_median(missed)}; "
                    "rows reuse entries/backtests and are not independent"
                ),
            }
        )
    return DataFrame(rows, columns=PORTABILITY_COLUMNS)


def contradiction_rows(portability: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []

    def add(frame: DataFrame, question: str, conflict: str, alternative: str) -> None:
        for row in frame.to_dict(orient="records"):
            records.append(
                {
                    "domain": row["domain"],
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "scope": row["scope"],
                    "horizon_hours": row["horizon_hours"],
                    "metric": row["metric"],
                    "question": question,
                    "observed_conflict": conflict,
                    "alternative_explanation": alternative,
                    "classification": row["classification"],
                    "source_artifact": row["source_artifact"],
                }
            )

    direct = portability[portability["domain"].eq("direct_entry")]
    add(
        direct[
            direct["classification"].isin(
                ["precursor_or_regime_lead", "sparse_precursor_or_regime_lead"]
            )
        ],
        "Does the exact trigger add timing information beyond its precursor?",
        "Broad positive event/control difference but the native precursor is usually stronger than the exact onset.",
        "The entry may mark an already-developing regime rather than initiate or precisely time the reaction.",
    )
    freqai = portability[portability["domain"].eq("freqai_incremental")]
    add(
        freqai[freqai["classification"].isin(["ranking_only_lead", "magnitude_error_only_lead"])],
        "Does event identity improve both ordering and magnitude accuracy?",
        "Rank and magnitude-error comparisons do not agree across both controls.",
        "The model may reorder outcomes slightly without calibrating them, or reduce average error without stable ranking.",
    )
    levels = portability[portability["domain"].eq("level_relationship")].copy()
    primary = pd.to_numeric(levels["primary_median_delta"], errors="coerce")
    secondary = pd.to_numeric(levels["secondary_median_delta"], errors="coerce")
    add(
        levels[primary.mul(secondary).lt(0)],
        "Is the nearby-level state incremental rather than a generic state partition?",
        "The contemporaneous-vs-no-level and contemporaneous-vs-stale comparisons have opposite signs.",
        "State composition, rather than the live level itself, may explain the apparent relationship.",
    )
    exits = portability[portability["domain"].eq("exit_counterfactual")].copy()
    net = pd.to_numeric(exits["primary_median_delta"], errors="coerce")
    hold = pd.to_numeric(exits["secondary_median_delta"], errors="coerce")
    add(
        exits[net.mul(hold).lt(0)],
        "Does the path-based regret balance agree with the terminal hold counterfactual?",
        "Net exit regret and hold-to-horizon direction disagree.",
        "The post-exit path may contain both favourable continuation and later adverse movement, making the action horizon-dependent.",
    )
    columns = [
        "domain",
        "candidate_id",
        "family",
        "scope",
        "horizon_hours",
        "metric",
        "question",
        "observed_conflict",
        "alternative_explanation",
        "classification",
        "source_artifact",
    ]
    return DataFrame(records, columns=columns)


def classified_counts(frame: DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for domain, group in frame.groupby("domain", dropna=False):
        result[str(domain)] = {
            str(key): int(value)
            for key, value in group["classification"].value_counts().items()
        }
    return result


def build_review(
    base: Path,
    generation_manifest: Path,
) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    manifest = json.loads(generation_manifest.read_text(encoding="utf-8"))
    require_generation_gate(manifest)
    exit_repair_audit_path = (
        base.parent.parent / "generation1" / "g1-b4" / "cache_audit.json"
    )
    exit_repair_audit = (
        json.loads(exit_repair_audit_path.read_text(encoding="utf-8"))
        if exit_repair_audit_path.is_file()
        else None
    )
    parts = [
        direct_entry_rows(base),
        freqai_rows(base),
        level_rows(base),
        exit_rows(base),
    ]
    portability = pd.concat(parts, ignore_index=True)
    contradictions = contradiction_rows(portability)
    batches = {
        item["id"]: item["status"]
        for item in manifest.get("batches", [])
        if item["id"].startswith("G0-B")
    }
    review = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "generation": 0,
        "status": "evidence_synthesized_branch_selection_pending",
        "gate": {
            "evidence_batches": {key: batches[key] for key in sorted(batches) if key != "G0-B7"},
            "all_B1_to_B6_terminal": True,
            "branch_execution_authorized": False,
            "rule": "Review all Generation 0 domains together before freezing Generation 1; descriptive classifications are not promotions.",
        },
        "integrity": {
            "freqai_score_audit": manifest.get("freqai_run", {}).get("final_score_audit"),
            "direct_baseline_audit": json.loads(
                (base / "direct_baseline_audit.json").read_text(encoding="utf-8")
            ),
            "exit_order_repair_audit": exit_repair_audit,
            "catalogue_parked_sources": 3,
            "catalogue_parked_reason": "Source-strategy datetime millisecond/microsecond MergeError; parked rather than silently repaired outside the approved harness scope.",
        },
        "surface": {
            "portability_rows": int(len(portability)),
            "contradiction_rows": int(len(contradictions)),
            "domain_rows": {
                str(key): int(value)
                for key, value in portability["domain"].value_counts().items()
            },
            "classification_counts": classified_counts(portability),
        },
        "interpretation_contract": [
            "Association and incremental predictive information are not causation.",
            "No single metric, profitable horizon, pair, direction, or exposed window is promotion-grade evidence.",
            "Direct reactions, price-state controls, shifted placebos, precursor controls, cross-pair/window portability, and contradictions remain separate evidence dimensions.",
            "The 0.65/0.35 consistency bands reuse the direct-atlas descriptive convention; FreqAI strict leads additionally require agreement in rank and magnitude error against both controls plus pair/window breadth.",
            "Level relationships require agreement against both no-nearby-level and stale-level controls; higher timeframe is never automatic precedence.",
            "Exit regret, missed continuation, avoided downside, and hold-to-horizon remain separate even when a compact role label is assigned.",
        ],
        "limitations": [
            "All FreqAI evaluation windows are exposed research windows; no sealed promotion claim is made.",
            "The FreqAI exact-identity profile is a broad 78-event-feature model, so null average increment does not rule out named conditional event families.",
            "Level relationships cover BTC, ETH, and SOL rather than the full frozen top-10 universe.",
            "Exit observations are conditioned on historical open-trade paths; repaired matched controls use different paths but entries, pairs, windows, timestamps, and source strategies remain dependent.",
            "News and wider-market causes remain unobserved; periods are not labelled quiet or normal.",
        ],
        "outputs": {
            "portability_matrix": str(base / "generation0_portability_matrix.csv"),
            "contradiction_matrix": str(base / "generation0_contradiction_matrix.csv"),
            "review": str(base / "generation0_review.json"),
        },
        "branch_candidates": [],
    }
    return portability, contradictions, json_safe(review)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("generation_manifest", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    portability, contradictions, review = build_review(
        base, args.generation_manifest.resolve()
    )
    atomic_csv(portability, base / "generation0_portability_matrix.csv")
    atomic_csv(contradictions, base / "generation0_contradiction_matrix.csv")
    atomic_json(review, base / "generation0_review.json")
    print(
        json.dumps(
            {
                "status": review["status"],
                "portability_rows": len(portability),
                "contradiction_rows": len(contradictions),
                "classification_counts": review["surface"]["classification_counts"],
                "outputs": review["outputs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
