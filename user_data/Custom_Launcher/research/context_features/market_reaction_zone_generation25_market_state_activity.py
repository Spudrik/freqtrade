"""Test whether FreqAI market-state activity forecasts beat simple causal baselines."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import ConstantInputWarning


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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_common as g25c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g25d_market_state_only_activity_portability"
DEFAULT_RUN_ID = "g25_market_state_activity_20260828a"
RECORD_ROOT = g25z.OUTPUT_ROOT / "market_state_activity"
SUPPORT_MANIFEST = RECORD_ROOT / "g25_market_state_activity_support_freeze.json"
CONTROLS = (
    "constant_training_median",
    "within_pair_time_shuffled_training_labels",
    "simple_recent_activity",
    "causal_stale_activity_24h",
)
CANDIDATE_ROLE = "market_state_only"
SHUFFLED_ROLE = "within_pair_time_shuffled_training_labels"
SIMPLE_ROLE = "simple_recent_activity"
STALE_ROLE = "causal_stale_activity_24h"
ROLLING_PERCENTILE_HOURS = 720
ROLLING_PERCENTILE_MIN_HOURS = 168
STALE_HOURS = 24
PAIR_METRICS = tuple(g24f.PAIR_METRICS)


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def parent_manifest_path(cohort: str) -> Path:
    return (
        g24f.RECORD_ROOT
        / f"{g24f.DEFAULT_RUN_STEM}_{cohort}"
        / "g24_freqai_run_manifest.json"
    )


def load_parent_manifest(cohort: str) -> dict[str, Any]:
    path = parent_manifest_path(cohort)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation24_freqai_cohort":
        raise ValueError(f"Generation 24 {cohort} FreqAI run is not terminal.")
    if manifest.get("technical_smoke_not_evidence"):
        raise ValueError("Technical smoke cannot supply Generation 25 evidence.")
    return manifest


def _selected_commands(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        item
        for item in manifest["commands"]
        if item["role"] in (CANDIDATE_ROLE, SHUFFLED_ROLE)
    ]
    cells = {
        (str(item["model_class"]), int(item["seed"]))
        for item in selected
        if item["role"] == CANDIDATE_ROLE
    }
    shuffled_cells = {
        (str(item["model_class"]), int(item["seed"]))
        for item in selected
        if item["role"] == SHUFFLED_ROLE
    }
    if cells != set(g24cache.MODEL_CELLS) or shuffled_cells != cells:
        raise ValueError("Generation 24 market-state/shuffled model cells are incomplete.")
    return selected


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    g25c.load_branch(BRANCH_ID)
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation25_market_state_outcomes":
            raise ValueError("Invalid Generation 25 market-state support.")
        return manifest
    cohorts: dict[str, Any] = {}
    for cohort in ("normal", "meme"):
        parent = load_parent_manifest(cohort)
        selected = _selected_commands(parent)
        feature_dir = Path(parent["storage"]["actual_feature_cache_dir"])
        event_dir = Path(parent["storage"]["actual_event_cache_dir"])
        features = []
        targets = []
        for pair in parent["pairs"]:
            stem = g0.pair_file_stem(pair)
            feature_path = feature_dir / f"{stem}.parquet"
            target_path = event_dir / f"{stem}.parquet"
            features.append({"pair": pair, **artifact(feature_path)})
            targets.append({"pair": pair, **artifact(target_path)})
        cohorts[cohort] = {
            "pairs": list(parent["pairs"]),
            "validation_periods": list(parent["validation_periods"]),
            "selected_profiles": [
                {
                    "profile_id": item["profile_id"],
                    "role": item["role"],
                    "model_class": item["model_class"],
                    "seed": item["seed"],
                    "model_dir": item["model_dir"],
                }
                for item in selected
            ],
            "feature_inventory": features,
            "target_inventory": targets,
            "thresholds": artifact(Path(parent["storage"]["training_median_thresholds"])),
            "parent_manifest": artifact(parent_manifest_path(cohort)),
        }
    manifest = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation25_market_state_outcomes",
        "branch_id": BRANCH_ID,
        "candidate_role": CANDIDATE_ROLE,
        "controls": list(CONTROLS),
        "targets": list(g24cache.TARGETS),
        "model_cells": [list(cell) for cell in g24cache.MODEL_CELLS],
        "simple_baseline": {
            "volume_feature": "relative_volume",
            "range_feature": "range_over_atr_4h",
            "rolling_percentile_hours": ROLLING_PERCENTILE_HOURS,
            "minimum_hours": ROLLING_PERCENTILE_MIN_HOURS,
            "stale_hours": STALE_HOURS,
        },
        "cohorts": cohorts,
        "same_holdout_exploratory": True,
        "fresh_confirmation": False,
        "future_outcome_values_read": False,
        "source_contracts": {
            "generation25_freeze": artifact(g25z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def causal_rolling_percentile(values: Series) -> Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rolling(
        ROLLING_PERCENTILE_HOURS,
        min_periods=ROLLING_PERCENTILE_MIN_HOURS,
    ).rank(pct=True)


def simple_baseline_predictions(parent: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    feature_dir = Path(parent["storage"]["actual_feature_cache_dir"])
    for pair in parent["pairs"]:
        frame = pd.read_parquet(
            feature_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "relative_volume", "range_over_atr_4h"],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        volume = causal_rolling_percentile(frame["relative_volume"])
        range_activity = causal_rolling_percentile(frame["range_over_atr_4h"])
        output = frame[["date"]].copy()
        output["pair"] = pair
        for target in g24cache.TARGETS:
            source = volume if "future_volume_ratio" in target else range_activity
            output[target] = source
            output[f"stale__{target}"] = source.shift(STALE_HOURS)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _prediction_lookup(
    parent: dict[str, Any],
) -> tuple[
    dict[tuple[str, int], DataFrame],
    dict[tuple[str, int], DataFrame],
    list[dict[str, Any]],
]:
    candidate: dict[tuple[str, int], DataFrame] = {}
    shuffled: dict[tuple[str, int], DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in _selected_commands(parent):
        frame, audit = g0f.load_predictions(Path(item["model_dir"]), tuple(item["pairs"]))
        missing = sorted(set(g24cache.TARGETS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        cell = (str(item["model_class"]), int(item["seed"]))
        target = candidate if item["role"] == CANDIDATE_ROLE else shuffled
        target[cell] = frame
        audit.update({"profile_id": item["profile_id"], "role": item["role"]})
        audits.append(audit)
    return candidate, shuffled, audits


def pair_scores(parent: dict[str, Any]) -> tuple[DataFrame, DataFrame]:
    candidate, shuffled, audits = _prediction_lookup(parent)
    simple = simple_baseline_predictions(parent)
    actual = g24f.load_actual(parent)
    common_inputs = {
        **{f"candidate_{model}_{seed}": frame for (model, seed), frame in candidate.items()},
        **{f"shuffled_{model}_{seed}": frame for (model, seed), frame in shuffled.items()},
    }
    common, _ = g0f.common_prediction_keys(common_inputs, tuple(parent["pairs"]))
    fair = common.merge(actual, on=["pair", "date"], how="left", validate="one_to_one")
    fair = fair.merge(simple, on=["pair", "date"], how="left", validate="one_to_one")
    thresholds = pd.read_csv(parent["storage"]["training_median_thresholds"])
    threshold_lookup = thresholds.set_index(["pair", "period", "target"])[
        "training_median"
    ].to_dict()
    rows: list[dict[str, Any]] = []
    for cell_key, prediction in candidate.items():
        shuffled_prediction = shuffled[cell_key]
        model_class, seed = cell_key
        candidate_columns = prediction[["pair", "date", *g24cache.TARGETS]].rename(
            columns={target: f"candidate__{target}" for target in g24cache.TARGETS}
        )
        shuffled_columns = shuffled_prediction[["pair", "date", *g24cache.TARGETS]].rename(
            columns={target: f"shuffled__{target}" for target in g24cache.TARGETS}
        )
        merged = fair.merge(
            candidate_columns, on=["pair", "date"], how="left", validate="one_to_one"
        ).merge(shuffled_columns, on=["pair", "date"], how="left", validate="one_to_one")
        for (pair, period), group in merged.groupby(["pair", "period"], observed=True):
            if period not in parent["validation_periods"]:
                continue
            for target in g24cache.TARGETS:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                predictions = {
                    CANDIDATE_ROLE: group[f"candidate__{target}"],
                    SHUFFLED_ROLE: group[f"shuffled__{target}"],
                    SIMPLE_ROLE: group[target + "_y"] if target + "_y" in group else group[target],
                    STALE_ROLE: group[f"stale__{target}"],
                    "constant_training_median": Series(threshold, index=group.index),
                }
                actual_column = target + "_x" if target + "_x" in group else target
                for role, prediction_values in predictions.items():
                    rows.append(
                        {
                            "cohort": parent["cohort"],
                            "pair": pair,
                            "period": period,
                            "model_class": model_class,
                            "seed": seed,
                            "role": role,
                            "target": target,
                            "training_median": threshold,
                            **g23f.prediction_metrics(
                                group[actual_column], prediction_values, threshold
                            ),
                        }
                    )
    audit_frame = DataFrame.from_records(audits)
    return DataFrame.from_records(rows), audit_frame


def scope_scores(pair_frame: DataFrame) -> DataFrame:
    return g24f.scope_scores(pair_frame)


def control_comparisons(
    scopes: DataFrame,
    validation_periods: Sequence[str],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    keys = ["cohort", "market_scope", "period", "model_class", "seed", "target"]
    role_for_control = {
        "constant_training_median": "constant_training_median",
        "within_pair_time_shuffled_training_labels": SHUFFLED_ROLE,
        "simple_recent_activity": SIMPLE_ROLE,
        "causal_stale_activity_24h": STALE_ROLE,
    }
    for key, group in scopes.groupby(keys, observed=True, sort=False):
        indexed = group.set_index("role")
        if CANDIDATE_ROLE not in indexed.index:
            continue
        candidate = indexed.loc[CANDIDATE_ROLE]
        candidate_gate = bool(
            candidate["coin_support_pass"]
            and candidate["top_minus_bottom_difference"] > 0.0
            and candidate["top_quartile_above_training_median_fraction"] >= 0.55
        )
        for control, baseline_role in role_for_control.items():
            baseline = indexed.loc[baseline_role]
            if control == "constant_training_median":
                metrics_pass = bool(
                    candidate["median_absolute_error"] < baseline["median_absolute_error"]
                )
            else:
                metrics_pass = bool(
                    candidate["median_absolute_error"] < baseline["median_absolute_error"]
                    and candidate["spearman_rank_correlation"]
                    > baseline["spearman_rank_correlation"]
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
    cell_rows: list[dict[str, Any]] = []
    cell_keys = ["cohort", "market_scope", "model_class", "seed", "target"]
    for key, group in comparisons.groupby(cell_keys, observed=True, sort=False):
        cohort = str(key[0])
        expected_periods = {
            period for period in validation_periods if str(period).startswith(f"g18_{cohort}_")
        }
        complete = bool(
            set(group["control"]) == set(CONTROLS)
            and set(group["period"]) == expected_periods
            and len(group) == len(CONTROLS) * len(expected_periods)
        )
        cell_rows.append(
            {
                **dict(zip(cell_keys, key, strict=True)),
                "complete_control_and_period_ladder": complete,
                "all_controls_both_periods_pass": bool(
                    complete and group["period_control_pass"].all()
                ),
            }
        )
    cells = DataFrame.from_records(cell_rows)
    final_rows: list[dict[str, Any]] = []
    final_keys = ["cohort", "market_scope", "target"]
    for key, group in cells.groupby(final_keys, observed=True, sort=False):
        retained = bool(
            len(group) == len(g24cache.MODEL_CELLS)
            and group["all_controls_both_periods_pass"].all()
        )
        final_rows.append(
            {
                **dict(zip(final_keys, key, strict=True)),
                "status": (
                    "exploratory_nonlevel_activity_context"
                    if retained
                    else "not_retained_across_all_model_seed_cells"
                ),
                "model_seed_cells_present": len(group),
                "required_model_seed_cells": len(g24cache.MODEL_CELLS),
                "all_model_seed_cells_pass": retained,
            }
        )
    return comparisons, cells, DataFrame.from_records(final_rows)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    freeze_support(overwrite=False)
    sibling_supports = g25c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g25_market_state_activity_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    pair_frames: list[DataFrame] = []
    audits: list[DataFrame] = []
    validation_periods: set[str] = set()
    for cohort in ("normal", "meme"):
        parent = load_parent_manifest(cohort)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            pair, audit = pair_scores(parent)
        pair_frames.append(pair)
        audits.append(audit)
        validation_periods.update(parent["validation_periods"])
    pairs = pd.concat(pair_frames, ignore_index=True)
    scopes = scope_scores(pairs)
    comparisons, cells, decisions = control_comparisons(scopes, sorted(validation_periods))
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prediction_audit": run_dir / "g25_market_state_prediction_audit.csv",
        "pair_scores": run_dir / "g25_market_state_pair_scores.csv",
        "scope_scores": run_dir / "g25_market_state_scope_scores.csv",
        "control_comparisons": run_dir / "g25_market_state_control_comparisons.csv",
        "model_seed_decisions": run_dir / "g25_market_state_model_seed_decisions.csv",
        "decisions": run_dir / "g25_market_state_decisions.csv",
    }
    for frame, path in (
        (pd.concat(audits, ignore_index=True), paths["prediction_audit"]),
        (pairs, paths["pair_scores"]),
        (scopes, paths["scope_scores"]),
        (comparisons, paths["control_comparisons"]),
        (cells, paths["model_seed_decisions"]),
        (decisions, paths["decisions"]),
    ):
        g0.atomic_write_csv(frame, path)
    retained = int(decisions["all_model_seed_cells_pass"].sum())
    result = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation25_market_state_activity",
        "exploratory_nonlevel_activity_contexts": retained,
        "not_retained": len(decisions) - retained,
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "level_claim_made": False,
            "same_holdout_exploratory": True,
            "fresh_confirmation": False,
        },
        "source_contracts": {
            "frozen_support": artifact(SUPPORT_MANIFEST),
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
