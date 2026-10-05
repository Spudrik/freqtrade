"""Independent integrity audit for Sieve3 event-reaction FreqAI score artifacts.

This verifier reads the completed score outputs only.  It does not retrain models,
rewrite scores, or apply research verdict thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


PROFILES = ("price_control", "event_identity_placebo", "event_identity")
KEY = ["profile_id", "pair", "window", "scope", "target"]
PAIR_KEY = ["pair", "window", "scope", "target"]
PORTABILITY_KEY = ["profile_id", "scope", "target"]
FAMILY_KEY = ["profile_id", "scope", "target_family", "horizon_hours"]

DIRECT_METRICS = (
    "prediction_actual_spearman",
    "top_minus_bottom",
    "r_squared",
    "roc_auc",
    "average_precision",
)
ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "brier_score_clipped",
)
CONTROL_METRICS = (
    "prediction_actual_spearman",
    "top_minus_bottom",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r_squared",
    "prediction_bias",
    "roc_auc",
    "average_precision",
    "brier_score_clipped",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _mismatch_count(left: pd.Series, right: pd.Series) -> int:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_finite = left_num.notna() & right_num.notna()
    close = pd.Series(False, index=left.index)
    close.loc[both_finite] = np.isclose(
        left_num.loc[both_finite],
        right_num.loc[both_finite],
        rtol=1e-9,
        atol=1e-12,
    )
    return int((~(both_missing | close)).sum())


def _compare_frames(
    actual: DataFrame,
    expected: DataFrame,
    key: list[str],
) -> list[str]:
    errors: list[str] = []
    actual_columns = list(actual.columns)
    expected_columns = list(expected.columns)
    if actual_columns != expected_columns:
        return [
            "summary column mismatch: "
            f"actual={actual_columns!r} expected={expected_columns!r}"
        ]
    left = actual.sort_values(key).reset_index(drop=True)
    right = expected.sort_values(key).reset_index(drop=True)
    if len(left) != len(right):
        errors.append(f"summary row mismatch: actual={len(left)} expected={len(right)}")
        return errors
    for column in key:
        if not left[column].fillna("<NA>").astype(str).equals(
            right[column].fillna("<NA>").astype(str)
        ):
            errors.append(f"summary key mismatch in {column}")
    for column in (item for item in left.columns if item not in key):
        mismatches = _mismatch_count(left[column], right[column])
        if mismatches:
            errors.append(f"summary value mismatch in {column}: {mismatches} rows")
    return errors


def recompute_summaries(scores: DataFrame) -> tuple[DataFrame, DataFrame]:
    scored = scores[
        scores["status"].eq("scored")
        & scores["scope"].isin(["all", "any_entry_onset"])
        & scores["profile_id"].eq("event_identity")
    ].copy()
    scored["positive_spearman_delta"] = pd.to_numeric(
        scored["prediction_actual_spearman_delta_vs_price"], errors="coerce"
    ).gt(0)
    scored["positive_mae_skill"] = pd.to_numeric(
        scored["mean_absolute_error_skill_vs_price"], errors="coerce"
    ).gt(0)
    portability = (
        scored.groupby(["profile_id", "scope", "target"], dropna=False)
        .agg(
            pair_window_tests=("pair", "size"),
            distinct_pairs=("pair", "nunique"),
            positive_pair_windows=("positive_spearman_delta", "sum"),
            positive_mae_skill_pair_windows=("positive_mae_skill", "sum"),
            median_spearman=("prediction_actual_spearman", "median"),
            median_spearman_delta_vs_price=(
                "prediction_actual_spearman_delta_vs_price",
                "median",
            ),
            median_mae_skill_vs_price=("mean_absolute_error_skill_vs_price", "median"),
            median_rmse_skill_vs_price=("root_mean_squared_error_skill_vs_price", "median"),
            median_r_squared_delta_vs_price=("r_squared_delta_vs_price", "median"),
            median_absolute_bias_skill_vs_price=("absolute_bias_skill_vs_price", "median"),
            median_top_bottom=("top_minus_bottom", "median"),
            median_top_bottom_delta_vs_price=("top_minus_bottom_delta_vs_price", "median"),
            median_spearman_delta_vs_event_placebo=(
                "prediction_actual_spearman_delta_vs_event_placebo",
                "median",
            ),
            median_top_bottom_delta_vs_event_placebo=(
                "top_minus_bottom_delta_vs_event_placebo",
                "median",
            ),
            median_mae_skill_vs_event_placebo=(
                "mean_absolute_error_skill_vs_event_placebo",
                "median",
            ),
            median_rmse_skill_vs_event_placebo=(
                "root_mean_squared_error_skill_vs_event_placebo",
                "median",
            ),
            median_r_squared_delta_vs_event_placebo=(
                "r_squared_delta_vs_event_placebo",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            ["positive_pair_windows", "median_spearman_delta_vs_price"],
            ascending=False,
        )
    )
    family = (
        scored.groupby(
            ["profile_id", "scope", "target_family", "horizon_hours"],
            dropna=False,
        )
        .agg(
            target_pair_window_tests=("target", "size"),
            distinct_targets=("target", "nunique"),
            distinct_pairs=("pair", "nunique"),
            positive_delta_share=("positive_spearman_delta", "mean"),
            positive_mae_skill_share=("positive_mae_skill", "mean"),
            median_spearman_delta_vs_price=(
                "prediction_actual_spearman_delta_vs_price",
                "median",
            ),
            median_mae_skill_vs_price=("mean_absolute_error_skill_vs_price", "median"),
            median_rmse_skill_vs_price=("root_mean_squared_error_skill_vs_price", "median"),
            median_r_squared_delta_vs_price=("r_squared_delta_vs_price", "median"),
            median_top_bottom_delta_vs_price=("top_minus_bottom_delta_vs_price", "median"),
            median_spearman_delta_vs_event_placebo=(
                "prediction_actual_spearman_delta_vs_event_placebo",
                "median",
            ),
            median_top_bottom_delta_vs_event_placebo=(
                "top_minus_bottom_delta_vs_event_placebo",
                "median",
            ),
            median_mae_skill_vs_event_placebo=(
                "mean_absolute_error_skill_vs_event_placebo",
                "median",
            ),
            median_rmse_skill_vs_event_placebo=(
                "root_mean_squared_error_skill_vs_event_placebo",
                "median",
            ),
            median_r_squared_delta_vs_event_placebo=(
                "r_squared_delta_vs_event_placebo",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            ["positive_delta_share", "median_spearman_delta_vs_price"],
            ascending=False,
        )
    )
    return portability, family


def audit(run_dir: Path) -> dict[str, Any]:
    score_path = run_dir / "freqai_reaction_scores.csv"
    portability_path = run_dir / "freqai_portability_summary.csv"
    family_path = run_dir / "freqai_target_family_summary.csv"
    manifest_path = run_dir / "manifest.json"
    for path in (score_path, portability_path, family_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(score_path, low_memory=False)
    portability = pd.read_csv(portability_path)
    family = pd.read_csv(family_path)
    errors: list[str] = []

    if len(scores) != 180_456:
        errors.append(f"unexpected score row count: {len(scores)}")
    if int(scores.duplicated(KEY).sum()):
        errors.append(f"duplicate score keys: {int(scores.duplicated(KEY).sum())}")
    if set(scores["profile_id"].dropna()) != set(PROFILES):
        errors.append(f"profile mismatch: {sorted(scores['profile_id'].dropna().unique())}")
    expected_pairs = set(manifest["pairs"])
    if set(scores["pair"].dropna()) != expected_pairs:
        errors.append("pair set differs from manifest")
    expected_windows = set(manifest["windows"])
    if set(scores["window"].dropna()) != expected_windows:
        errors.append("window set differs from manifest")
    if scores["target"].nunique() != 73:
        errors.append(f"unexpected target count: {scores['target'].nunique()}")
    unexpected_status = set(scores["status"].dropna()) - {"scored", "unscorable"}
    if unexpected_status:
        errors.append(f"unexpected statuses: {sorted(unexpected_status)}")
    numeric = scores.select_dtypes(include=[np.number])
    infinity_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if infinity_count:
        errors.append(f"infinite score cells: {infinity_count}")

    key_counts = scores.groupby(PAIR_KEY, dropna=False)["profile_id"].nunique()
    if not key_counts.eq(len(PROFILES)).all():
        errors.append(f"base keys missing profiles: {int((~key_counts.eq(len(PROFILES))).sum())}")
    signature = scores.pivot(index=PAIR_KEY, columns="profile_id", values=["rows", "status"])
    for profile in PROFILES[1:]:
        if not signature[("rows", "price_control")].equals(signature[("rows", profile)]):
            errors.append(f"row support differs: price_control vs {profile}")
        if not signature[("status", "price_control")].equals(signature[("status", profile)]):
            errors.append(f"score status differs: price_control vs {profile}")

    indexed = scores.set_index(PAIR_KEY)
    control = indexed[indexed["profile_id"].eq("price_control")]
    placebo = indexed[indexed["profile_id"].eq("event_identity_placebo")]
    for metric in CONTROL_METRICS:
        expected_control = scores.set_index(PAIR_KEY).index.map(control[metric])
        mismatches = _mismatch_count(
            scores[f"control_{metric}"],
            pd.Series(expected_control, index=scores.index),
        )
        if mismatches:
            errors.append(f"control pairing mismatch for {metric}: {mismatches}")
        expected_placebo = scores.set_index(PAIR_KEY).index.map(placebo[metric])
        mismatches = _mismatch_count(
            scores[f"event_placebo_{metric}"],
            pd.Series(expected_placebo, index=scores.index),
        )
        if mismatches:
            errors.append(f"placebo pairing mismatch for {metric}: {mismatches}")

    for metric in DIRECT_METRICS:
        for comparator, prefix in (("price", "control"), ("event_placebo", "event_placebo")):
            expected = pd.to_numeric(scores[metric], errors="coerce") - pd.to_numeric(
                scores[f"{prefix}_{metric}"], errors="coerce"
            )
            column = f"{metric}_delta_vs_{comparator}"
            mismatches = _mismatch_count(scores[column], expected)
            if mismatches:
                errors.append(f"arithmetic mismatch in {column}: {mismatches}")
    for metric in ERROR_METRICS:
        for comparator, prefix in (("price", "control"), ("event_placebo", "event_placebo")):
            expected = pd.to_numeric(scores[f"{prefix}_{metric}"], errors="coerce") - pd.to_numeric(
                scores[metric], errors="coerce"
            )
            column = f"{metric}_skill_vs_{comparator}"
            mismatches = _mismatch_count(scores[column], expected)
            if mismatches:
                errors.append(f"arithmetic mismatch in {column}: {mismatches}")
    for comparator, prefix in (("price", "control"), ("event_placebo", "event_placebo")):
        expected = pd.to_numeric(scores[f"{prefix}_prediction_bias"], errors="coerce").abs() - pd.to_numeric(
            scores["prediction_bias"], errors="coerce"
        ).abs()
        column = f"absolute_bias_skill_vs_{comparator}"
        mismatches = _mismatch_count(scores[column], expected)
        if mismatches:
            errors.append(f"arithmetic mismatch in {column}: {mismatches}")

    control_rows = scores["profile_id"].eq("price_control")
    placebo_rows = scores["profile_id"].eq("event_identity_placebo")
    self_columns = [
        *(f"{metric}_delta_vs_price" for metric in DIRECT_METRICS),
        *(f"{metric}_skill_vs_price" for metric in ERROR_METRICS),
        "absolute_bias_skill_vs_price",
    ]
    for column in self_columns:
        values = pd.to_numeric(scores.loc[control_rows, column], errors="coerce").dropna()
        if not np.isclose(values, 0.0, rtol=1e-9, atol=1e-12).all():
            errors.append(f"price-control self comparison is nonzero: {column}")
    placebo_self_columns = [
        *(f"{metric}_delta_vs_event_placebo" for metric in DIRECT_METRICS),
        *(f"{metric}_skill_vs_event_placebo" for metric in ERROR_METRICS),
        "absolute_bias_skill_vs_event_placebo",
    ]
    for column in placebo_self_columns:
        values = pd.to_numeric(scores.loc[placebo_rows, column], errors="coerce").dropna()
        if not np.isclose(values, 0.0, rtol=1e-9, atol=1e-12).all():
            errors.append(f"placebo self comparison is nonzero: {column}")

    fresh = scores[scores["window"].eq("fresh_to_round1")]
    if not fresh["status"].eq("unscorable").all() or not pd.to_numeric(
        fresh["rows"], errors="coerce"
    ).fillna(-1).eq(0).all():
        errors.append("pre-prediction fresh_to_round1 window is not uniformly zero-row/unscorable")
    post_fresh_all = scores[
        ~scores["window"].eq("fresh_to_round1") & scores["scope"].eq("all")
    ]
    definitional_peak_targets = {
        "&-future_downside_peak_step_1h",
        "&-future_upside_peak_step_1h",
    }
    post_fresh_unscored = post_fresh_all[~post_fresh_all["status"].eq("scored")]
    if (
        set(post_fresh_unscored["target"].dropna()) != definitional_peak_targets
        or not post_fresh_unscored["status"].eq("unscorable").all()
        or not pd.to_numeric(post_fresh_unscored["rows"], errors="coerce").ge(20).all()
    ):
        errors.append(
            "post-fresh all-scope unscored rows are not limited to the two "
            "definitionally constant one-hour peak-step targets"
        )
    expected_definitional_rows = (
        len(PROFILES)
        * len(expected_pairs)
        * (len(expected_windows) - 1)
        * len(definitional_peak_targets)
    )
    if len(post_fresh_unscored) != expected_definitional_rows:
        errors.append(
            "unexpected one-hour peak-step unscored support: "
            f"actual={len(post_fresh_unscored)} expected={expected_definitional_rows}"
        )
    identity_scored = scores[
        scores["profile_id"].eq("event_identity") & scores["status"].eq("scored")
    ]
    paired_columns = [
        *(f"control_{metric}" for metric in CONTROL_METRICS),
        *(f"event_placebo_{metric}" for metric in CONTROL_METRICS),
    ]
    missing_paired = int(identity_scored[paired_columns].isna().all(axis=1).sum())
    if missing_paired:
        errors.append(f"identity scored rows wholly missing paired metrics: {missing_paired}")

    metadata_variants = scores.groupby("target", dropna=False)[
        ["target_family", "horizon_hours"]
    ].nunique(dropna=False)
    if (metadata_variants > 1).any(axis=None):
        errors.append("target metadata is inconsistent across score rows")

    expected_portability, expected_family = recompute_summaries(scores)
    errors.extend(
        f"portability: {item}"
        for item in _compare_frames(portability, expected_portability, PORTABILITY_KEY)
    )
    errors.extend(
        f"family: {item}"
        for item in _compare_frames(family, expected_family, FAMILY_KEY)
    )
    if int(portability.duplicated(PORTABILITY_KEY).sum()):
        errors.append("duplicate portability keys")
    if int(family.duplicated(FAMILY_KEY).sum()):
        errors.append("duplicate family-summary keys")
    if set(portability["profile_id"].dropna()) != {"event_identity"}:
        errors.append("portability contains a non-identity profile")
    if set(family["profile_id"].dropna()) != {"event_identity"}:
        errors.append("family summary contains a non-identity profile")
    for label, frame in (("portability", portability), ("family", family)):
        numeric_frame = frame.select_dtypes(include=[np.number])
        count = int(np.isinf(numeric_frame.to_numpy(dtype=float)).sum())
        if count:
            errors.append(f"infinite {label} cells: {count}")

    status_counts = (
        scores.groupby(["profile_id", "window", "status"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    all_scope_support = (
        scores[scores["scope"].eq("all")]
        .groupby(["profile_id", "window", "target"], dropna=False)["rows"]
        .first()
        .groupby(["profile_id", "window"])
        .agg(["min", "max"])
        .reset_index()
    )
    return {
        "passed": not errors,
        "errors": errors,
        "score_rows": int(len(scores)),
        "score_unique_keys": int(scores[KEY].drop_duplicates().shape[0]),
        "profiles": sorted(scores["profile_id"].dropna().unique().tolist()),
        "pairs": int(scores["pair"].nunique()),
        "windows": int(scores["window"].nunique()),
        "targets": int(scores["target"].nunique()),
        "scopes": int(scores["scope"].nunique()),
        "infinite_score_cells": infinity_count,
        "portability_rows": int(len(portability)),
        "family_summary_rows": int(len(family)),
        "status_counts": [
            {key: _json_value(value) for key, value in row.items()}
            for row in status_counts.to_dict(orient="records")
        ],
        "all_scope_support": [
            {key: _json_value(value) for key, value in row.items()}
            for row in all_scope_support.to_dict(orient="records")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
