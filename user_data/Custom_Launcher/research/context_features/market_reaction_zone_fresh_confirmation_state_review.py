"""Score only the frozen fresh-period market-activity questions and controls."""

from __future__ import annotations

# Bound numerical pools before research imports.
# ruff: noqa: E402
import json
import os
import sys
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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation0 as g0f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation23 as g23f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation24 as g24f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_state_run as run,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_market_state_activity as g25state,
)


ANALYSIS_PATH = Path(__file__).resolve()
ROLES = (
    g25state.CANDIDATE_ROLE,
    g25state.SHUFFLED_ROLE,
    g25state.SIMPLE_ROLE,
    g25state.STALE_ROLE,
    "constant_training_median",
)
CONTROL_ROLES = {
    "constant_training_median": "constant_training_median",
    "within_pair_time_shuffled_training_labels": g25state.SHUFFLED_ROLE,
    "simple_recent_activity": g25state.SIMPLE_ROLE,
    "causal_stale_activity_24h": g25state.STALE_ROLE,
}
PAIR_METRICS = tuple(g23f.PAIR_METRICS)


def _load_manifest() -> tuple[dict[str, Any], Path]:
    path = run.RECORD_ROOT / run.DEFAULT_RUN_ID / "fresh_state_freqai_run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "running" or manifest["technical_smoke_not_evidence"]:
        raise ValueError("The full frozen FreqAI manifest is not complete/runnable.")
    if len(manifest["commands"]) != 16 or any(
        item["status"] != "completed" for item in manifest["commands"]
    ):
        raise ValueError("All 16 frozen FreqAI profiles must complete before scoring.")
    for source in manifest["source_contracts"].values():
        run.verify_artifact(source)
    return manifest, path


def _inputs(
    manifest: dict[str, Any], cohort: str
) -> tuple[
    DataFrame,
    DataFrame,
    dict[tuple[str, int], dict[str, DataFrame]],
    DataFrame,
    DataFrame,
]:
    frozen, _split, support, outcome = run.sources()
    pairs = tuple(frozen["data_contract"]["cohort_pairs"][cohort])
    support_rows = {
        row["pair"]: row
        for row in support["routes"]["fresh_market_state_activity"]["inventory"]
        if row["cohort"] == cohort
    }
    outcome_rows = {
        row["pair"]: row
        for row in outcome["routes"]["fresh_market_state_activity"]["inventory"]
        if row["cohort"] == cohort
    }
    targets = tuple(outcome_rows[pairs[0]]["targets"])
    actual_parts = []
    threshold_rows = []
    for pair in pairs:
        if tuple(outcome_rows[pair]["targets"]) != targets:
            raise ValueError(f"Frozen targets differ within {cohort}.")
        frame = pd.read_parquet(
            outcome_rows[pair]["actual_target_cache"]["path"],
            columns=["date", "period", run.READY_COLUMN, *targets],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        for period in frozen["fresh_periods"]:
            start = pd.Timestamp(period["start_utc"])
            cutoff = start - pd.Timedelta(hours=run.MAX_HORIZON_HOURS)
            training_start = start - pd.Timedelta(
                days=int(frozen["data_contract"]["training_days_by_cohort"][cohort])
            )
            training = frame.loc[
                frame[run.READY_COLUMN].fillna(False).astype(bool)
                & frame["date"].ge(training_start)
                & frame["date"].lt(cutoff)
            ]
            for target in targets:
                values = pd.to_numeric(training[target], errors="coerce").dropna()
                if len(values) < 1000:
                    raise ValueError(f"Insufficient median training rows: {cohort}/{pair}/{target}")
                threshold_rows.append(
                    {
                        "cohort": cohort,
                        "pair": pair,
                        "period": period["id"],
                        "target": target,
                        "training_rows": len(values),
                        "training_median": float(values.median()),
                    }
                )
        actual_parts.append(frame[["date", "period", *targets]].assign(pair=pair))
    actual = pd.concat(actual_parts, ignore_index=True)
    feature_dir = Path(support_rows[pairs[0]]["feature"]["path"]).parent
    simple = g25state.simple_baseline_predictions(
        {"storage": {"actual_feature_cache_dir": str(feature_dir)}, "pairs": pairs}
    )
    predictions: dict[tuple[str, int], dict[str, DataFrame]] = {}
    audits = []
    commands = [item for item in manifest["commands"] if item["cohort"] == cohort]
    for item in commands:
        frame, audit = g0f.load_predictions(Path(item["model_dir"]), pairs)
        if frame.empty or not set(targets).issubset(frame.columns):
            raise ValueError(f"No frozen target predictions: {item['profile_id']}")
        cell = str(item["model_class"]), int(item["seed"])
        predictions.setdefault(cell, {})[item["role"]] = frame
        audits.append({"cohort": cohort, "profile_id": item["profile_id"], **audit})
    if len(predictions) != 4 or any(
        set(cell) != set(run.freshz.STATE_ROLES) for cell in predictions.values()
    ):
        raise ValueError(f"Incomplete four-cell profile ladder: {cohort}")
    return (
        actual,
        simple,
        predictions,
        DataFrame.from_records(threshold_rows),
        DataFrame.from_records(audits),
    )


def _pair_scores(cohort: str, inputs: tuple[Any, ...]) -> tuple[DataFrame, DataFrame]:
    actual, simple, predictions, thresholds, audits = inputs
    target_columns = tuple(
        column for column in actual.columns if column.startswith("&-g24_percentile_")
    )
    threshold_lookup = thresholds.set_index(["pair", "period", "target"])[
        "training_median"
    ].to_dict()
    all_predictions = {
        f"{model}_{seed}_{role}": frame
        for (model, seed), roles in predictions.items()
        for role, frame in roles.items()
    }
    common, common_audit = g0f.common_prediction_keys(
        all_predictions, tuple(actual["pair"].unique())
    )
    if common.empty or common_audit.attrs.get("missing_pairs"):
        raise ValueError(f"No complete prediction intersection for {cohort}.")
    fair = common.merge(actual, on=["pair", "date"], how="left", validate="one_to_one")
    fair = fair.merge(simple, on=["pair", "date"], how="left", validate="one_to_one")
    rows = []
    for (model_class, seed), roles in predictions.items():
        candidate = roles[g25state.CANDIDATE_ROLE][["pair", "date", *target_columns]].rename(
            columns={target: f"candidate__{target}" for target in target_columns}
        )
        shuffled = roles[g25state.SHUFFLED_ROLE][["pair", "date", *target_columns]].rename(
            columns={target: f"shuffled__{target}" for target in target_columns}
        )
        merged = fair.merge(candidate, on=["pair", "date"], validate="one_to_one")
        merged = merged.merge(shuffled, on=["pair", "date"], validate="one_to_one")
        for (pair, period), group in merged.groupby(["pair", "period"], observed=True):
            if period not in {"fresh_early", "fresh_late"}:
                continue
            for target in target_columns:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                if not np.isfinite(threshold):
                    raise ValueError(f"Missing training median: {cohort}/{pair}/{period}/{target}")
                actual_values = group[f"{target}_x"]
                role_predictions = {
                    g25state.CANDIDATE_ROLE: group[f"candidate__{target}"],
                    g25state.SHUFFLED_ROLE: group[f"shuffled__{target}"],
                    g25state.SIMPLE_ROLE: group[f"{target}_y"],
                    g25state.STALE_ROLE: group[f"stale__{target}"],
                    "constant_training_median": Series(threshold, index=group.index),
                }
                for role, values in role_predictions.items():
                    rows.append(
                        {
                            "cohort": cohort,
                            "pair": pair,
                            "period": period,
                            "model_class": model_class,
                            "seed": seed,
                            "role": role,
                            "target": target,
                            "training_median": threshold,
                            **g23f.prediction_metrics(actual_values, values, threshold),
                        }
                    )
    return DataFrame.from_records(rows), pd.concat([audits, common_audit], ignore_index=True)


def _decisions(
    scopes: DataFrame, selected: list[dict[str, str]]
) -> tuple[DataFrame, DataFrame, DataFrame]:
    rows = []
    keys = ["cohort", "market_scope", "period", "model_class", "seed", "target"]
    for key, group in scopes.groupby(keys, observed=True, sort=False):
        indexed = group.set_index("role")
        if set(indexed.index) != set(ROLES):
            raise ValueError(f"Incomplete fresh control ladder: {key}")
        candidate = indexed.loc[g25state.CANDIDATE_ROLE]
        candidate_gate = bool(
            candidate["coin_support_pass"]
            and candidate["top_minus_bottom_difference"] > 0
            and candidate["top_quartile_above_training_median_fraction"] >= 0.55
        )
        for control, role in CONTROL_ROLES.items():
            baseline = indexed.loc[role]
            metrics_pass = bool(
                candidate["median_absolute_error"] < baseline["median_absolute_error"]
            )
            if control != "constant_training_median":
                metrics_pass &= bool(
                    candidate["spearman_rank_correlation"] > baseline["spearman_rank_correlation"]
                    and candidate["top_minus_bottom_difference"]
                    > baseline["top_minus_bottom_difference"]
                )
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "control": control,
                    "candidate_gate_pass": candidate_gate,
                    "candidate_mae": candidate["median_absolute_error"],
                    "control_mae": baseline["median_absolute_error"],
                    "candidate_spearman": candidate["spearman_rank_correlation"],
                    "control_spearman": baseline["spearman_rank_correlation"],
                    "candidate_top_minus_bottom": candidate["top_minus_bottom_difference"],
                    "control_top_minus_bottom": baseline["top_minus_bottom_difference"],
                    "candidate_top_above_training_median_fraction": candidate[
                        "top_quartile_above_training_median_fraction"
                    ],
                    "period_control_pass": bool(candidate_gate and metrics_pass),
                }
            )
    comparisons = DataFrame.from_records(rows)
    cells = []
    cell_keys = ["cohort", "market_scope", "model_class", "seed", "target"]
    for key, group in comparisons.groupby(cell_keys, observed=True, sort=False):
        complete = bool(
            len(group) == 2 * len(CONTROL_ROLES)
            and set(group["period"]) == {"fresh_early", "fresh_late"}
            and set(group["control"]) == set(CONTROL_ROLES)
        )
        cells.append(
            {
                **dict(zip(cell_keys, key, strict=True)),
                "complete_fresh_ladder": complete,
                "all_controls_both_periods_pass": bool(
                    complete and group["period_control_pass"].all()
                ),
                "minimum_top_above_training_median_fraction": float(
                    group["candidate_top_above_training_median_fraction"].min()
                ),
                "minimum_top_minus_bottom": float(group["candidate_top_minus_bottom"].min()),
                "controls_passed": int(group["period_control_pass"].sum()),
            }
        )
    cells_frame = DataFrame.from_records(cells)
    final = []
    for question in selected:
        found = cells_frame.loc[
            cells_frame["cohort"].eq(question["cohort"])
            & cells_frame["market_scope"].eq(question["market_scope"])
            & cells_frame["target"].eq(question["target"])
        ]
        if len(found) != 4:
            raise ValueError(f"Expected four frozen model/seed cells: {question}")
        final.append(
            {
                **question,
                "model_seed_cells_present": len(found),
                "all_model_seed_cells_pass": bool(found["all_controls_both_periods_pass"].all()),
                "cells_passing": int(found["all_controls_both_periods_pass"].sum()),
                "controls_passed_of_32": int(found["controls_passed"].sum()),
                "minimum_top_above_training_median_fraction": float(
                    found["minimum_top_above_training_median_fraction"].min()
                ),
                "minimum_top_minus_bottom": float(found["minimum_top_minus_bottom"].min()),
            }
        )
    return comparisons, cells_frame, DataFrame.from_records(final)


def main() -> int:
    manifest, manifest_path = _load_manifest()
    frozen = run.freshz.freeze()
    selected = next(
        item for item in frozen["siblings"] if item["id"] == "fresh_market_state_activity"
    )["retained_scope_target_questions"]
    pair_parts = []
    audit_parts = []
    threshold_parts = []
    for cohort in ("normal", "meme"):
        inputs = _inputs(manifest, cohort)
        pairs, audits = _pair_scores(cohort, inputs)
        pair_parts.append(pairs)
        audit_parts.append(audits)
        threshold_parts.append(inputs[3])
    pair_frame = pd.concat(pair_parts, ignore_index=True)
    scopes = g24f.scope_scores(pair_frame)
    comparisons, cells, decisions = _decisions(scopes, selected)
    record_dir = manifest_path.parent
    artifacts = {}
    for name, frame in (
        ("training_thresholds", pd.concat(threshold_parts, ignore_index=True)),
        ("prediction_audit", pd.concat(audit_parts, ignore_index=True)),
        ("pair_scores", pair_frame),
        ("scope_scores", scopes),
        ("control_comparisons", comparisons),
        ("model_cells", cells),
        ("decisions", decisions),
    ):
        path = record_dir / f"fresh_state_{name}.csv"
        g0.atomic_write_csv(frame, path)
        artifacts[name] = run.artifact(path)
    result = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "batch_id": manifest["batch_id"],
        "status": "completed_frozen_fresh_market_state_activity_review",
        "questions": len(decisions),
        "questions_confirmed": int(decisions["all_model_seed_cells_pass"].sum()),
        "periods": ["fresh_early", "fresh_late"],
        "model_seed_cells": 4,
        "controls_per_period": list(CONTROL_ROLES),
        "source_contracts": {
            "manifest": run.artifact(manifest_path),
            "analysis_script": run.artifact(ANALYSIS_PATH),
        },
        "artifacts": artifacts,
        "signed_direction_read": False,
        "profit_read": False,
    }
    result_path = record_dir / "fresh_state_review.json"
    g0.atomic_write_json(result, result_path)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "questions", "questions_confirmed")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
