"""Test whether Generation 13's direct level effect survives nearby reaction labels."""

from __future__ import annotations

# Bound numerical pools before pandas/NumPy imports.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_direct_screen as g11d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)


DEFAULT_RUN_ID = "g14_label_sensitivity_20260822a"
RECORD_ROOT = (
    g14z.FREEZE_PATH.parent / "g14_broad_combinations" / "label_sensitivity"
)
MAX_WORKERS = 4
DIRECT_RESULT = g13d.RECORD_ROOT / g13d.DEFAULT_RUN_ID / "g13_direct_control_result.json"
OUTCOME_COLUMNS = tuple(
    column
    for horizon in g14z.REACTION_HORIZONS
    for column in (f"abs_excursion_atr_h{horizon}", f"volume_ratio_h{horizon}")
)
SOURCE_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "control",
    "event_time",
    "period",
    "contact_close_distance_atr",
    "zone_half_width_atr",
    *OUTCOME_COLUMNS,
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def definition_id(price_threshold_atr: float, volume_threshold: float) -> str:
    return f"price_{price_threshold_atr:g}atr__volume_{volume_threshold:g}x"


def reaction_label(
    frame: DataFrame,
    *,
    prefix: str,
    horizon: int,
    price_threshold_atr: float,
    volume_threshold: float,
) -> Series:
    zone = pd.to_numeric(
        frame[f"{prefix}__zone_half_width_atr"], errors="coerce"
    ).to_numpy(dtype=float)
    excursion = pd.to_numeric(
        frame[f"{prefix}__abs_excursion_atr_h{horizon}"], errors="coerce"
    ).to_numpy(dtype=float)
    volume = pd.to_numeric(
        frame[f"{prefix}__volume_ratio_h{horizon}"], errors="coerce"
    ).to_numpy(dtype=float)
    threshold = np.maximum(price_threshold_atr, zone)
    valid = np.isfinite(threshold) & np.isfinite(excursion) & np.isfinite(volume)
    values = np.full(len(frame), np.nan, dtype=float)
    values[valid] = (
        (excursion[valid] >= threshold[valid])
        & (volume[valid] >= volume_threshold)
    ).astype(float)
    return Series(values, index=frame.index, dtype=float)


def validate_sources() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    frozen = g14z.freeze_generation14()
    if frozen.get("status") != "frozen_before_generation14_outcomes":
        raise ValueError("Generation 14 label grid was not frozen.")
    direct = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    if direct.get("status") != "completed_generation13_direct_controls":
        raise ValueError("Generation 13 direct control result is not terminal.")
    g13_review = json.loads(g14z.G13_JOINT_REVIEW.read_text(encoding="utf-8"))
    direct_contract = g13_review["source_direct_control"]
    if g0.sha256_file(DIRECT_RESULT) != direct_contract["sha256"]:
        raise ValueError("Generation 13 direct result changed after joint review.")
    inventory_path = Path(direct["inventory"]["path"])
    if g0.sha256_file(inventory_path) != direct["inventory"]["sha256"]:
        raise ValueError("Generation 13 pair-match inventory changed.")
    return frozen, direct, pd.read_csv(inventory_path)


def source_tasks() -> dict[tuple[str, str], dict[str, Any]]:
    manifest = json.loads(g14z.g13z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    return {
        (str(item["cohort"]), str(item["pair"])): dict(item)
        for item in manifest["tasks"]
    }


def prefixed_lookup(frame: DataFrame, *, control: str, prefix: str) -> DataFrame:
    selected = frame.loc[frame["control"].eq(control)].copy()
    keys = ["window", "analysis_period", "source_timeframe", "event_time"]
    selected = selected[[*keys, "zone_half_width_atr", *OUTCOME_COLUMNS]]
    return selected.rename(
        columns={
            "event_time": f"{prefix}_event_time",
            "zone_half_width_atr": f"{prefix}__zone_half_width_atr",
            **{column: f"{prefix}__{column}" for column in OUTCOME_COLUMNS},
        }
    )


def enrich_pair(row: dict[str, Any], source: dict[str, Any]) -> DataFrame:
    matched_path = Path(str(row["path"]))
    if g0.sha256_file(matched_path) != str(row["sha256"]):
        raise ValueError(f"Generation 13 matched pairs changed: {matched_path}")
    source_path = Path(str(source["event_path"]))
    if g0.sha256_file(source_path) != str(source["event_sha256"]):
        raise ValueError(f"Generation 6 event source changed: {source_path}")
    matched = pd.read_parquet(matched_path)
    frame = pd.read_parquet(
        source_path,
        columns=list(SOURCE_COLUMNS),
        filters=[("control", "in", ["actual", *g13d.CONTROLS])],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = g13d.causal_deduplicate(
        g13d.add_analysis_windows(frame.dropna(subset=["event_time"]), str(row["cohort"]))
    )
    actual = prefixed_lookup(frame, control="actual", prefix="actual")
    keys = ["window", "analysis_period", "source_timeframe"]
    output = matched.merge(
        actual,
        on=[*keys, "actual_event_time"],
        how="left",
        validate="many_to_one",
    )
    controls: list[DataFrame] = []
    for control in g13d.CONTROLS:
        control_lookup = prefixed_lookup(frame, control=control, prefix="control")
        selected = output.loc[output["control"].eq(control)].merge(
            control_lookup,
            on=[*keys, "control_event_time"],
            how="left",
            validate="many_to_one",
        )
        controls.append(selected)
    output = pd.concat(controls, ignore_index=True, sort=False)
    output["event_time"] = pd.to_datetime(output["actual_event_time"], utc=True)
    required = [
        *(f"actual__{column}" for column in ("zone_half_width_atr", *OUTCOME_COLUMNS)),
        *(f"control__{column}" for column in ("zone_half_width_atr", *OUTCOME_COLUMNS)),
    ]
    output["raw_outcomes_joined"] = output[required].notna().all(axis=1)
    return output


def enrich_all(inventory: DataFrame, *, workers: int) -> DataFrame:
    source_map = source_tasks()
    frames: list[DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {}
        for row in inventory.to_dict(orient="records"):
            key = (str(row["cohort"]), str(row["pair"]))
            if key not in source_map:
                raise KeyError(f"No causal event source for {key}")
            futures[pool.submit(enrich_pair, row, source_map[key])] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                failures.append(
                    {"cohort": key[0], "pair": key[1], "error": f"{type(exc).__name__}: {exc}"}
                )
    if failures:
        raise RuntimeError(f"Generation 14 raw-outcome enrichment failed: {failures}")
    return pd.concat(frames, ignore_index=True, sort=False)


def score_definition(
    independent: DataFrame,
    *,
    horizon: int,
    price_threshold_atr: float,
    volume_threshold: float,
) -> tuple[DataFrame, DataFrame]:
    frame = independent.copy()
    frame["actual_reaction"] = reaction_label(
        frame,
        prefix="actual",
        horizon=horizon,
        price_threshold_atr=price_threshold_atr,
        volume_threshold=volume_threshold,
    )
    frame["control_reaction"] = reaction_label(
        frame,
        prefix="control",
        horizon=horizon,
        price_threshold_atr=price_threshold_atr,
        volume_threshold=volume_threshold,
    )
    frame = frame.dropna(subset=["actual_reaction", "control_reaction"]).copy()
    frame["reaction_delta"] = frame["actual_reaction"] - frame["control_reaction"]
    label = definition_id(price_threshold_atr, volume_threshold)
    pair_rows: list[dict[str, Any]] = []
    keys = ("cohort", "pair", "window", "analysis_period", "control")
    for key, cell in frame.groupby(list(keys), observed=True, sort=False):
        match_balance = g13d.paired_balance(cell, g13d.MATCH_FEATURES)
        audit_balance = g13d.paired_balance(cell, g13d.AUDIT_FEATURES)
        match_quality = g13d.balance_quality(match_balance)
        audit_quality = g13d.balance_quality(audit_balance)
        pair_rows.append(
            {
                "definition_id": label,
                "price_threshold_atr": price_threshold_atr,
                "volume_threshold": volume_threshold,
                "horizon_hours": horizon,
                "cohort": key[0],
                "pair": key[1],
                "window": key[2],
                "analysis_period": key[3],
                "control": key[4],
                "matched_independent_rows": len(cell),
                "actual_reaction_rate": float(cell["actual_reaction"].mean()),
                "control_reaction_rate": float(cell["control_reaction"].mean()),
                "reaction_rate_difference": float(cell["reaction_delta"].mean()),
                "adequately_balanced": (
                    match_quality in {"strong", "usable"}
                    and audit_quality in {"strong", "usable"}
                ),
            }
        )
    pair_summary = DataFrame.from_records(pair_rows)
    score_rows: list[dict[str, Any]] = []
    group_keys = ("cohort", "window", "analysis_period", "control")
    for key, cell in pair_summary.groupby(list(group_keys), observed=True, sort=False):
        eligible = cell.loc[
            cell["adequately_balanced"]
            & cell["matched_independent_rows"].ge(g13d.MIN_PAIR_ROWS)
        ]
        pairs = eligible["pair"].astype(str).tolist()
        matching = frame.loc[
            frame["cohort"].eq(key[0])
            & frame["window"].eq(key[1])
            & frame["analysis_period"].eq(key[2])
            & frame["control"].eq(key[3])
            & frame["pair"].isin(pairs)
        ]
        bootstrap = g11d.block_bootstrap(
            matching,
            value_columns=("reaction_delta",),
            seed_key=f"g14-label|{label}|{horizon}|" + "|".join(map(str, key)),
        )["reaction_delta"]
        score_rows.append(
            {
                "definition_id": label,
                "price_threshold_atr": price_threshold_atr,
                "volume_threshold": volume_threshold,
                "horizon_hours": horizon,
                "cohort": key[0],
                "window": key[1],
                "analysis_period": key[2],
                "control": key[3],
                "eligible_coins": len(eligible),
                "positive_coins": int(eligible["reaction_rate_difference"].gt(0).sum()),
                "matched_independent_rows": int(eligible["matched_independent_rows"].sum()),
                "equal_coin_actual_reaction_rate": float(
                    eligible["actual_reaction_rate"].mean()
                )
                if len(eligible)
                else np.nan,
                "equal_coin_control_reaction_rate": float(
                    eligible["control_reaction_rate"].mean()
                )
                if len(eligible)
                else np.nan,
                "equal_coin_reaction_rate_difference": bootstrap[0],
                "bootstrap_lower_95": bootstrap[1],
                "bootstrap_upper_95": bootstrap[2],
                "point_cell_pass": bool(
                    len(eligible) >= g13d.MIN_COINS and bootstrap[0] > 0
                ),
                "strict_cell_pass": bool(
                    len(eligible) >= g13d.MIN_COINS and bootstrap[1] > 0
                ),
            }
        )
    return pair_summary, DataFrame.from_records(score_rows)


def definition_decisions(scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = (
        "definition_id",
        "price_threshold_atr",
        "volume_threshold",
        "cohort",
        "window",
        "horizon_hours",
    )
    for key, cell in scores.groupby(list(keys), observed=True, sort=False):
        expected_periods = (
            list(g14z.g13z.g11base.VALIDATION_PERIODS[key[3]])
            if key[4] == g14z.STANDARD
            else [item["period"] for item in g14z.g12z.RECENT_PERIODS]
        )
        expected = {
            (period, control)
            for period in expected_periods
            for control in g13d.CONTROLS
        }
        observed = set(zip(cell["analysis_period"], cell["control"], strict=False))
        complete = observed == expected
        rows.append(
            {
                "definition_id": key[0],
                "price_threshold_atr": float(key[1]),
                "volume_threshold": float(key[2]),
                "cohort": key[3],
                "window": key[4],
                "horizon_hours": int(key[5]),
                "complete_control_period_ladder": complete,
                "point_pass": bool(complete and cell["point_cell_pass"].all()),
                "strict_pass": bool(complete and cell["strict_cell_pass"].all()),
                "minimum_equal_coin_difference": float(
                    cell["equal_coin_reaction_rate_difference"].min()
                ),
                "minimum_bootstrap_lower_95": float(cell["bootstrap_lower_95"].min()),
                "minimum_eligible_coins": int(cell["eligible_coins"].min()),
            }
        )
    return DataFrame.from_records(rows)


def robust_decisions(decisions: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    original = definition_id(0.5, 1.25)
    keys = ("cohort", "window", "horizon_hours")
    expected_definitions = len(g14z.REACTION_PRICE_THRESHOLDS_ATR) * len(
        g14z.REACTION_VOLUME_THRESHOLDS
    )
    for key, cell in decisions.groupby(list(keys), observed=True, sort=False):
        original_cell = cell.loc[cell["definition_id"].eq(original)]
        point_count = int(cell["point_pass"].astype(bool).sum())
        strict_count = int(cell["strict_pass"].astype(bool).sum())
        complete = bool(
            len(cell) == expected_definitions
            and cell["complete_control_period_ladder"].astype(bool).all()
        )
        original_point = bool(
            len(original_cell) == 1 and original_cell["point_pass"].iloc[0]
        )
        robust_point = bool(complete and original_point and point_count >= 7)
        rows.append(
            {
                "cohort": key[0],
                "window": key[1],
                "horizon_hours": int(key[2]),
                "definitions_completed": len(cell),
                "point_positive_definitions": point_count,
                "strict_positive_definitions": strict_count,
                "original_definition_point_positive": original_point,
                "robust_point": robust_point,
                "robust_strict": bool(robust_point and strict_count >= 5),
                "minimum_definition_difference": float(
                    cell["minimum_equal_coin_difference"].min()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(list(keys)).reset_index(drop=True)


def run_sensitivity(*, run_id: str, workers: int) -> dict[str, Any]:
    frozen, direct, inventory = validate_sources()
    enriched = enrich_all(inventory, workers=workers)
    if not bool(enriched["raw_outcomes_joined"].all()):
        missing = int((~enriched["raw_outcomes_joined"]).sum())
        raise ValueError(f"Generation 14 raw outcomes failed to join for {missing} rows.")
    pair_frames: list[DataFrame] = []
    score_frames: list[DataFrame] = []
    independent_counts: list[dict[str, Any]] = []
    for horizon in g14z.REACTION_HORIZONS:
        independent = g13d.independent_rows(
            enriched, horizon=horizon, scope="all_source_timeframes"
        )
        independent_counts.append({"horizon_hours": horizon, "rows": len(independent)})
        for price_threshold in g14z.REACTION_PRICE_THRESHOLDS_ATR:
            for volume_threshold in g14z.REACTION_VOLUME_THRESHOLDS:
                pair, scores = score_definition(
                    independent,
                    horizon=horizon,
                    price_threshold_atr=price_threshold,
                    volume_threshold=volume_threshold,
                )
                pair_frames.append(pair)
                score_frames.append(scores)
    pair_summary = pd.concat(pair_frames, ignore_index=True, sort=False)
    scores = pd.concat(score_frames, ignore_index=True, sort=False)
    definitions = definition_decisions(scores)
    robust = robust_decisions(definitions)
    record_dir = RECORD_ROOT / run_id
    paths = {
        "pair_scores": record_dir / "label_pair_scores.csv",
        "group_scores": record_dir / "label_group_scores.csv",
        "definition_decisions": record_dir / "label_definition_decisions.csv",
        "robust_decisions": record_dir / "label_robust_decisions.csv",
    }
    g0.atomic_write_csv(pair_summary, paths["pair_scores"])
    g0.atomic_write_csv(scores, paths["group_scores"])
    g0.atomic_write_csv(definitions, paths["definition_decisions"])
    g0.atomic_write_csv(robust, paths["robust_decisions"])
    result = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation14_label_sensitivity",
        "plain_question": frozen["siblings"]["reaction_definition_sensitivity"][
            "plain_question"
        ],
        "matched_rows": len(enriched),
        "independent_rows_by_horizon": independent_counts,
        "definitions": len(g14z.REACTION_PRICE_THRESHOLDS_ATR)
        * len(g14z.REACTION_VOLUME_THRESHOLDS),
        "robust_point_rows": int(robust["robust_point"].sum()),
        "robust_strict_rows": int(robust["robust_strict"].sum()),
        "future_signed_direction_used": False,
        "profit_used": False,
        "source_contracts": {
            "generation14_freeze": artifact(g14z.FREEZE_PATH),
            "generation13_direct_result": artifact(DIRECT_RESULT),
            "generation13_pair_inventory": direct["inventory"],
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g14_label_sensitivity_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Generation 14 label sensitivity.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    print(
        json.dumps(
            run_sensitivity(run_id=args.run_id, workers=args.workers),
            indent=2,
            default=g0.json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
