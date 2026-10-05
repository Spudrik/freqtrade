"""Materialize only the frozen unsigned outcomes after joint support is complete.

This module cannot open outcomes from the current partial fresh windows.  Its hard
input is the jointly completed support result produced by
``market_reaction_zone_fresh_confirmation_support``.  A preflight is safe to run at
any time and creates no D-drive outcome directory.
"""

from __future__ import annotations

# Bound native numerical pools before importing numpy/pandas.
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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as freshz,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_support as freshs,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24cache,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_OUTCOME_ID = "fresh_confirmation_outcomes_20260903a"
RECORD_ROOT = freshz.OUTPUT_ROOT / "outcomes"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
    "fresh_confirmation_outcomes"
)
COMPLETED_SUPPORT_STATUS = "frozen_all_three_supports_before_fresh_outcomes"
COMPLETED_OUTCOME_STATUS = "completed_fresh_unsigned_outcome_materialization"
ACTIVITY_METRICS = ("future_volume_ratio", "future_range_ratio")
ACTIVITY_HORIZONS = (1, 2, 4, 8)
ALL_ACTIVITY_TARGETS = tuple(
    f"&-g24_percentile_{metric}_h{horizon}"
    for metric in ACTIVITY_METRICS
    for horizon in ACTIVITY_HORIZONS
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def outcome_result_path(outcome_id: str) -> Path:
    return RECORD_ROOT / outcome_id / "fresh_confirmation_outcome_result.json"


def load_support(support_id: str) -> dict[str, Any]:
    path = freshs.support_result_path(support_id)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("batch_id") != freshz.BATCH_ID:
        raise ValueError("Fresh support belongs to a different batch.")
    if result.get("future_reaction_outcomes_read") is not False:
        raise ValueError("Support input is not outcome blind.")
    if result.get("status") == COMPLETED_SUPPORT_STATUS:
        freshs._verify_existing_result(result)
        if not result.get("all_three_support_gates_pass"):
            raise ValueError("Completed fresh support did not pass all three gates.")
        contracts = result.get("source_contracts", {})
        if contracts.get("support_builder", {}).get("sha256") != g0.sha256_file(
            freshs.ANALYSIS_PATH
        ):
            raise ValueError("Fresh support builder changed after support freeze.")
    return result


def preflight(support_id: str, outcome_id: str) -> dict[str, Any]:
    support = load_support(support_id)
    ready = support.get("status") == COMPLETED_SUPPORT_STATUS
    return {
        "schema_version": 1,
        "batch_id": freshz.BATCH_ID,
        "outcome_id": outcome_id,
        "status": "ready_to_materialize_frozen_unsigned_outcomes"
        if ready
        else "waiting_for_joint_outcome_blind_support_freeze",
        "support_status": support.get("status"),
        "all_three_support_routes_required": True,
        "outcome_directory_created": False,
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
        "next_action": (
            "Materialize the frozen unsigned targets for all three routes together."
            if ready
            else "Wait for complete chronology and the joint support freeze; do not open outcomes."
        ),
    }


def selected_activity_targets(frozen: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    sibling = next(
        item for item in frozen["siblings"] if item["id"] == "fresh_market_state_activity"
    )
    targets: dict[str, set[str]] = {"normal": set(), "meme": set()}
    for item in sibling["retained_scope_target_questions"]:
        cohort = str(item["cohort"])
        target = str(item["target"])
        if cohort not in targets or target not in ALL_ACTIVITY_TARGETS:
            raise ValueError(f"Unexpected frozen activity target: {cohort}/{target}")
        targets[cohort].add(target)
    output = {cohort: tuple(sorted(values)) for cohort, values in targets.items()}
    if len(output["normal"]) != 4 or len(output["meme"]) != 8:
        raise ValueError(f"Unexpected fresh activity target counts: {output}")
    return output


def _target_parts(target: str) -> tuple[str, int]:
    prefix = "&-g24_percentile_"
    if not target.startswith(prefix) or "_h" not in target:
        raise ValueError(f"Invalid activity target: {target}")
    metric, raw_horizon = target.removeprefix(prefix).rsplit("_h", 1)
    horizon = int(raw_horizon)
    if metric not in ACTIVITY_METRICS or horizon not in ACTIVITY_HORIZONS:
        raise ValueError(f"Unknown activity target: {target}")
    return metric, horizon


def _positions_for_dates(base: DataFrame, dates: Series) -> np.ndarray:
    base_dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    if base_dates.duplicated().any() or not base_dates.is_monotonic_increasing:
        raise ValueError("Outcome OHLCV dates must be unique and ordered.")
    lookup = Series(np.arange(len(base), dtype=np.int64), index=base_dates)
    positions = pd.to_datetime(dates, utc=True, errors="raise").map(lookup).to_numpy(float)
    if not np.isfinite(positions).all():
        raise ValueError("Frozen support contains a date absent from outcome OHLCV.")
    return positions.astype(np.int64)


def raw_activity_outcomes(
    base: DataFrame, dates: Series, targets: Sequence[str]
) -> DataFrame:
    """Calculate only unsigned future volume/range ratios named by the freeze."""
    positions = _positions_for_dates(base, dates)
    paths = g0.future_path_matrices(base, max(ACTIVITY_HORIZONS))
    pre_volume = pd.to_numeric(
        base["pre_volume_median_24"], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    pre_range = pd.to_numeric(
        base["pre_range_median_24"], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    output = DataFrame({"date": pd.to_datetime(dates, utc=True, errors="raise")})
    for target in targets:
        metric, horizon = _target_parts(target)
        high = paths["high"][positions, :horizon]
        low = paths["low"][positions, :horizon]
        volume = paths["volume"][positions, :horizon]
        if metric == "future_volume_ratio":
            numerator = np.mean(volume, axis=1)
            denominator = pre_volume
            complete = np.isfinite(volume).all(axis=1)
        else:
            numerator = np.mean(high - low, axis=1)
            denominator = pre_range
            complete = np.isfinite(high).all(axis=1) & np.isfinite(low).all(axis=1)
        values = np.divide(
            numerator,
            denominator,
            out=np.full(len(positions), np.nan),
            where=complete & np.isfinite(denominator) & (denominator > 0.0),
        )
        output[target] = values
    return output


def calibrated_activity_targets(
    raw: DataFrame,
    feature: DataFrame,
    *,
    pair: str,
    cohort: str,
    targets: Sequence[str],
) -> tuple[DataFrame, list[dict[str, Any]]]:
    """Fit one pre-fresh empirical scale and reuse it across both fresh blocks."""
    dates = pd.to_datetime(raw["date"], utc=True, errors="raise")
    feature_dates = pd.to_datetime(feature["date"], utc=True, errors="raise")
    if not dates.equals(feature_dates):
        raise ValueError(f"Raw activity outcomes and features do not align for {pair}.")
    ready_column = f"ready__{freshs.STATE_READY_BLOCK}"
    ready = feature[ready_column].fillna(False).astype(bool)
    first_start = pd.Timestamp(freshz.FRESH_PERIODS[0]["start_utc"])
    calibration_start = first_start - pd.Timedelta(days=freshz.STATE_TRAINING_DAYS[cohort])
    calibration_cutoff = first_start - pd.Timedelta(hours=freshs.TARGET_PURGE_HOURS)
    calibration_mask = ready & dates.ge(calibration_start) & dates.lt(calibration_cutoff)
    output = DataFrame(
        {
            "date": dates,
            "period": freshz.assign_fresh_period(dates),
            ready_column: ready,
        }
    )
    audits = []
    for target in targets:
        reference = (
            pd.to_numeric(raw.loc[calibration_mask, target], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(reference) < freshs.MIN_TRAINING_ROWS:
            raise ValueError(f"Insufficient target calibration rows for {pair}/{target}.")
        output[target] = g23cache.empirical_percentile(raw[target], reference)
        audits.append(
            {
                "cohort": cohort,
                "pair": pair,
                "target": target,
                "calibration_start_utc": calibration_start.isoformat(),
                "calibration_cutoff_exclusive_utc": calibration_cutoff.isoformat(),
                "calibration_rows": len(reference),
                "raw_training_median": float(reference.median()),
                "raw_training_minimum": float(reference.min()),
                "raw_training_maximum": float(reference.max()),
            }
        )
    return output, audits


def shuffled_activity_targets(
    actual: DataFrame, mapping: DataFrame, targets: Sequence[str]
) -> DataFrame:
    dates = pd.to_datetime(actual["date"], utc=True, errors="raise")
    mapping_dates = pd.to_datetime(mapping["date"], utc=True, errors="raise")
    if not dates.equals(mapping_dates):
        raise ValueError("Activity target and shuffled-source mapping dates do not align.")
    ready_column = f"ready__{freshs.STATE_READY_BLOCK}"
    current = actual[["date", "period", ready_column]].copy()
    current["shuffled_source_date"] = pd.to_datetime(
        mapping["shuffled_source_date"], utc=True, errors="coerce"
    )
    source_columns = {
        "date": "shuffled_source_date",
        **{target: f"source__{target}" for target in targets},
    }
    source = actual[["date", *targets]].rename(columns=source_columns)
    merged = current.merge(source, on="shuffled_source_date", how="left", validate="many_to_one")
    for target in targets:
        merged[target] = pd.to_numeric(merged.pop(f"source__{target}"), errors="coerce")
    return merged[["date", "period", ready_column, *targets]]


def _state_target_audit(
    actual: DataFrame,
    shuffled: DataFrame,
    *,
    pair: str,
    cohort: str,
    targets: Sequence[str],
) -> dict[str, Any]:
    dates = pd.to_datetime(actual["date"], utc=True, errors="raise")
    ready_column = f"ready__{freshs.STATE_READY_BLOCK}"
    ready = actual[ready_column].fillna(False).astype(bool)
    rows = []
    for item in freshz.FRESH_PERIODS:
        start = pd.Timestamp(item["start_utc"])
        end = pd.Timestamp(item["end_utc_exclusive"])
        period_mask = ready & dates.ge(start) & dates.lt(end)
        counts = {
            target: int(pd.to_numeric(actual.loc[period_mask, target], errors="coerce").notna().sum())
            for target in targets
        }
        rows.append(
            {
                "period": str(item["id"]),
                "actual_target_rows": counts,
                "minimum_actual_target_rows": min(counts.values()),
                "gate_pass": min(counts.values()) >= freshs.MIN_PREDICTION_ROWS,
            }
        )
    first_start = pd.Timestamp(freshz.FRESH_PERIODS[0]["start_utc"])
    training_start = first_start - pd.Timedelta(days=freshz.STATE_TRAINING_DAYS[cohort])
    training_cutoff = first_start - pd.Timedelta(hours=freshs.TARGET_PURGE_HOURS)
    training = ready & dates.ge(training_start) & dates.lt(training_cutoff)
    shuffled_counts = {
        target: int(
            pd.to_numeric(shuffled.loc[training, target], errors="coerce").notna().sum()
        )
        for target in targets
    }
    shuffled_pass = min(shuffled_counts.values()) >= freshs.MIN_TRAINING_ROWS
    return {
        "pair": pair,
        "cohort": cohort,
        "periods": rows,
        "shuffled_training_target_rows": shuffled_counts,
        "shuffled_training_gate_pass": shuffled_pass,
        "gate_pass": bool(rows and all(row["gate_pass"] for row in rows) and shuffled_pass),
    }


def materialize_state_targets(
    frozen: dict[str, Any], support: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    route = support["routes"]["fresh_market_state_activity"]
    targets_by_cohort = selected_activity_targets(frozen)
    inventory = []
    calibration_rows: list[dict[str, Any]] = []
    for item in route["inventory"]:
        pair = str(item["pair"])
        cohort = str(item["cohort"])
        targets = targets_by_cohort[cohort]
        feature_path = Path(item["feature"]["path"])
        mapping_path = Path(item["label_mapping"]["path"])
        for path, contract in (
            (feature_path, item["feature"]),
            (mapping_path, item["label_mapping"]),
        ):
            if g0.sha256_file(path) != contract["sha256"]:
                raise ValueError(f"Frozen state support changed: {path}")
        feature = pd.read_parquet(feature_path)
        mapping = pd.read_parquet(mapping_path)
        base, _ = g20s.base_and_state(pair, cohort)
        raw = raw_activity_outcomes(base, feature["date"], targets)
        actual, audits = calibrated_activity_targets(
            raw,
            feature,
            pair=pair,
            cohort=cohort,
            targets=targets,
        )
        shuffled = shuffled_activity_targets(actual, mapping, targets)
        audit = _state_target_audit(
            actual,
            shuffled,
            pair=pair,
            cohort=cohort,
            targets=targets,
        )
        calibration_rows.extend(audits)
        stem = g0.pair_file_stem(pair)
        root = output_root / "state" / cohort
        actual_path = root / "actual_target_cache" / f"{stem}.parquet"
        shuffled_path = root / "shuffled_target_cache" / f"{stem}.parquet"
        evaluation_path = root / "evaluation_cache" / f"{stem}.parquet"
        for path in (actual_path, shuffled_path, evaluation_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(actual, actual_path)
        g0.atomic_write_parquet(shuffled, shuffled_path)
        g0.atomic_write_parquet(actual[["date", "period", *targets]], evaluation_path)
        inventory.append(
            {
                **audit,
                "targets": list(targets),
                "actual_target_cache": artifact(actual_path),
                "shuffled_target_cache": artifact(shuffled_path),
                "evaluation_cache": artifact(evaluation_path),
            }
        )
    calibration_path = output_root / "state" / "target_calibration.csv"
    g0.atomic_write_csv(DataFrame.from_records(calibration_rows), calibration_path)
    return {
        "route_id": "fresh_market_state_activity",
        "targets_by_cohort": {
            cohort: list(targets) for cohort, targets in targets_by_cohort.items()
        },
        "calibration_rule": (
            "One pair-specific empirical distribution from the frozen training window "
            "ending eight hours before fresh_early; reused unchanged in both blocks."
        ),
        "inventory": inventory,
        "target_calibration": artifact(calibration_path),
        "outcome_gate_pass": bool(inventory and all(item["gate_pass"] for item in inventory)),
    }


def crossing_count_outcomes(
    base: DataFrame, events: DataFrame, horizon: int
) -> DataFrame:
    """Open only the frozen unsigned crossing-count outcome."""
    output = events.copy()
    positions = _positions_for_dates(base, output["event_time"])
    paths = g0.future_path_matrices(base, horizon)
    future_close = paths["close"][positions, :horizon]
    level = pd.to_numeric(output["level_price"], errors="coerce").to_numpy(dtype=float)
    previous = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)[
        positions
    ]
    crossing = g20s.g19_crossing_count(future_close, level, previous)
    complete = np.isfinite(future_close).all(axis=1)
    crossing[~complete] = np.nan
    column = f"outcome__crossing_count_h{horizon}"
    output[column] = crossing
    if output[column].isna().any():
        raise ValueError("Frozen direct support lacks a complete future crossing path.")
    output["g18_period"] = output["period"].astype(str)
    return output


def materialize_direct_route(
    support_route: dict[str, Any],
    *,
    cohort: str,
    horizon: int,
    scope_kind: str,
    scope_value: str,
    output_root: Path,
) -> dict[str, Any]:
    inventory = []
    for item in support_route["inventory"]:
        pair = str(item["pair"])
        if str(item["cohort"]) != cohort:
            raise ValueError(f"Unexpected cohort in {support_route['route_id']}: {pair}")
        source_path = Path(item["support"]["path"])
        if g0.sha256_file(source_path) != item["support"]["sha256"]:
            raise ValueError(f"Frozen direct support changed: {source_path}")
        events = pd.read_parquet(source_path)
        base, _ = g20s.base_and_state(pair, cohort)
        outcomes = crossing_count_outcomes(base, events, horizon)
        outcomes["scope_kind"] = scope_kind
        outcomes["scope_value"] = scope_value
        path = output_root / support_route["route_id"] / f"{g0.pair_file_stem(pair)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(outcomes, path)
        inventory.append(
            {
                "pair": pair,
                "cohort": cohort,
                "rows": len(outcomes),
                "period_control_counts": item["period_control_counts"],
                "outcomes": artifact(path),
            }
        )
    return {
        "route_id": support_route["route_id"],
        "cohort": cohort,
        "metric": "crossing_count",
        "horizon_hours": horizon,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "controls": list(support_route["controls"]),
        "inventory": inventory,
        "outcome_gate_pass": bool(inventory),
    }


def _verify_existing_result(result: dict[str, Any]) -> None:
    if result.get("status") != COMPLETED_OUTCOME_STATUS:
        return
    state = result["routes"]["fresh_market_state_activity"]
    state_items = [state["target_calibration"]]
    state_items.extend(
        contract
        for row in state["inventory"]
        for contract in (
            row["actual_target_cache"],
            row["shuffled_target_cache"],
            row["evaluation_cache"],
        )
    )
    direct_items = [
        row["outcomes"]
        for route_id, route in result["routes"].items()
        if route_id != "fresh_market_state_activity"
        for row in route["inventory"]
    ]
    for item in (*state_items, *direct_items):
        path = Path(item["path"])
        if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen unsigned outcome artifact changed: {path}")


def materialize(
    support_id: str,
    outcome_id: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    result_path = outcome_result_path(outcome_id)
    if result_path.is_file() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETED_OUTCOME_STATUS:
            _verify_existing_result(existing)
            return existing
    support = load_support(support_id)
    if support.get("status") != COMPLETED_SUPPORT_STATUS:
        raise RuntimeError(
            "Fresh outcomes remain locked until all three outcome-blind support routes "
            "are jointly frozen."
        )
    frozen = freshz.freeze()
    output_root = ARTIFACT_ROOT / outcome_id
    output_root.mkdir(parents=True, exist_ok=True)
    state = materialize_state_targets(frozen, support, output_root)
    convergence_support = support["routes"]["fresh_meme_convergence_crossing"]
    convergence = materialize_direct_route(
        convergence_support,
        cohort="meme",
        horizon=2,
        scope_kind="round_distribution_convergence_pooled",
        scope_value="all_60_frozen_definitions",
        output_root=output_root,
    )
    vwap_support = support["routes"]["fresh_daily_current_session_vwap"]
    vwap = materialize_direct_route(
        vwap_support,
        cohort="normal",
        horizon=8,
        scope_kind="anchored_vwap_band",
        scope_value=freshs.VWAP_SCOPE_VALUE,
        output_root=output_root,
    )
    routes = {
        state["route_id"]: state,
        convergence["route_id"]: convergence,
        vwap["route_id"]: vwap,
    }
    all_pass = all(bool(route["outcome_gate_pass"]) for route in routes.values())
    if not all_pass:
        raise ValueError("At least one frozen route failed unsigned outcome materialization.")
    support_path = freshs.support_result_path(support_id)
    result = {
        "schema_version": 1,
        "batch_id": freshz.BATCH_ID,
        "outcome_id": outcome_id,
        "created_at_utc": g0.utc_now(),
        "status": COMPLETED_OUTCOME_STATUS,
        "all_three_routes_materialized_together": True,
        "routes": routes,
        "source_contracts": {
            "joint_outcome_blind_support": artifact(support_path),
            "question_freeze": artifact(freshz.FREEZE_PATH),
            "outcome_materializer": artifact(ANALYSIS_PATH),
            "activity_percentile_helper": artifact(Path(g23cache.__file__)),
            "frozen_activity_target_registry": artifact(Path(g24cache.__file__)),
        },
        "future_reaction_outcomes_read": True,
        "future_signed_direction_read": False,
        "profit_read": False,
        "next_action": (
            "Run the frozen FreqAI profiles and direct control scorers; do not inspect "
            "unlisted outcomes."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(result, result_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-id", default=freshs.DEFAULT_SUPPORT_ID)
    parser.add_argument("--outcome-id", default=DEFAULT_OUTCOME_ID)
    parser.add_argument("--phase", choices=("preflight", "materialize"), default="preflight")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        result = preflight(args.support_id, args.outcome_id)
    else:
        result = materialize(args.support_id, args.outcome_id, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
