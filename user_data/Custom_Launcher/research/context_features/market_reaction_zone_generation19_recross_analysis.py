"""Run Generation 19 exact-location, market-scope, and recross-onset siblings."""

from __future__ import annotations

# Bound numerical pools before pandas imports.
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
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

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
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freeze as g19z,
)


DEFAULT_RUN_ID = "g19_recross_analysis_20260823a"
RECORD_ROOT = g19z.OUTPUT_ROOT / "recross_analysis"
ATLAS_EVENT_ROOT = g17l.ARTIFACT_ROOT / g17l.DEFAULT_RUN_ID / "pair_events"
HORIZONS = g19z.HORIZONS_HOURS
CONTROLS = g19z.LOCATION_CONTROLS
FAMILIES = (
    "adaptive_volume_profile_nodes",
    "donchian_boundaries",
    "rolling_vwap_deviation_bands",
)
EXACT_LEVELS = frozenset(
    {
        "vp_lb72_bins48_nearest_hvn_q80",
        "vp_lb72_bins96_nearest_hvn_q80",
        "vp_lb72_bins48_nearest_lvn_q10",
        "vp_lb72_bins96_nearest_lvn_q10",
        "vp_lb72_bins48_nearest_lvn_q20",
        "vp_lb72_bins96_nearest_lvn_q20",
        "donchian_lb24_upper",
        "donchian_lb24_lower",
        "donchian_lb72_upper",
        "donchian_lb72_lower",
        "rolling_vwap_lb72_upper_2sd",
        "rolling_vwap_lb72_lower_2sd",
    }
)
READ_COLUMNS = (
    "cohort",
    "pair",
    "control",
    "event_time",
    "level_family",
    "level_name",
    "level_price",
    "zone_half_width",
    "zone_half_width_atr",
    "base_atr",
    "pre_distance_atr",
    "approach_state",
    "crossings_h1",
    "crossings_h2",
    "crossings_h4",
)


def artifact(path: Path) -> dict[str, Any]:
    return g19z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g19z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation19_outcomes":
        raise ValueError("Generation 19 branch layer is not frozen.")
    return frozen


def read_later_rows(path: Path) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(READ_COLUMNS))
    frame = frame.loc[
        frame["level_family"].isin(FAMILIES)
        & frame["control"].isin({"actual", *CONTROLS})
    ].copy()
    if frame.empty:
        return frame
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    cohort_values = frame["cohort"].dropna().astype(str).unique()
    if len(cohort_values) != 1:
        raise ValueError(f"Unexpected cohort values in {path}: {cohort_values}")
    frame["g18_period"] = g18d.assign_confirmation_period(
        frame["event_time"], cohort_values[0]
    )
    return frame.loc[frame["g18_period"].ne("outside_g18_confirmation")].copy()


def coordinate_scopes(frame: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "g18_period", "control", "event_time"]
    family = g18d.nearest_anchor(frame, [*keys, "level_family"])
    family["scope_kind"] = "family_any_level"
    family["scope_value"] = family["level_family"]
    exact = frame.loc[frame["level_name"].isin(EXACT_LEVELS)].copy()
    exact["scope_kind"] = "single_level"
    exact["scope_value"] = exact["level_name"]
    return pd.concat([family, exact], ignore_index=True, sort=False)


def crossing_count(close_path: np.ndarray, levels: np.ndarray, previous: np.ndarray) -> np.ndarray:
    signs = np.sign(close_path - levels[:, None])
    combined = np.column_stack([np.sign(previous - levels), signs])
    valid = np.isfinite(combined).all(axis=1)
    output = np.full(len(levels), np.nan, dtype=float)
    output[valid] = (
        (combined[valid, :-1] * combined[valid, 1:]) < 0.0
    ).sum(axis=1)
    return output


def first_crossing_hour(
    close_path: np.ndarray, levels: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    signs = np.sign(close_path - levels[:, None])
    combined = np.column_stack([np.sign(previous - levels), signs])
    crossed = (combined[:, :-1] * combined[:, 1:]) < 0.0
    valid = np.isfinite(combined).all(axis=1)
    output = np.full(len(levels), np.nan, dtype=float)
    any_cross = crossed.any(axis=1) & valid
    output[valid & ~any_cross] = 0.0
    output[any_cross] = np.argmax(crossed[any_cross], axis=1) + 1.0
    return output


def indexed_close_windows(
    close: np.ndarray, indexes: np.ndarray, offsets: Sequence[int]
) -> np.ndarray:
    output = np.full((len(indexes), len(offsets)), np.nan, dtype=float)
    for column, offset in enumerate(offsets):
        positions = indexes + int(offset)
        valid = (positions >= 0) & (positions < len(close))
        output[valid, column] = close[positions[valid]]
    return output


def add_crossing_paths(frame: DataFrame, pair: str) -> None:
    source = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
    source = source.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    dates = pd.to_datetime(source["date"], utc=True, errors="raise")
    position_map = Series(np.arange(len(source), dtype=np.int64), index=dates)
    positions = frame["event_time"].map(position_map).to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        missing = int((~np.isfinite(positions)).sum())
        raise ValueError(f"{pair} has {missing} event timestamps absent from 1h OHLCV.")
    indexes = positions.astype(np.int64)
    close = pd.to_numeric(source["close"], errors="coerce").to_numpy(dtype=float)
    levels = pd.to_numeric(frame["level_price"], errors="coerce").to_numpy(dtype=float)
    previous = indexed_close_windows(close, indexes, (-1,))[:, 0]
    future = indexed_close_windows(close, indexes, tuple(range(1, max(HORIZONS) + 1)))
    for horizon in HORIZONS:
        if horizon <= 4:
            count = pd.to_numeric(
                frame[f"crossings_h{horizon}"], errors="coerce"
            ).to_numpy(dtype=float)
        else:
            count = crossing_count(future[:, :horizon], levels, previous)
        frame[f"metric__crossing_count_h{horizon}"] = count
        frame[f"metric__any_recross_h{horizon}"] = np.where(
            np.isfinite(count), (count >= 1.0).astype(float), np.nan
        )
        frame[f"metric__repeated_recross_h{horizon}"] = np.where(
            np.isfinite(count), (count >= 2.0).astype(float), np.nan
        )
    first = first_crossing_hour(future, levels, previous)
    frame["metric__recross_immediacy_h8"] = np.where(
        first > 0.0, (max(HORIZONS) + 1.0 - first) / max(HORIZONS), 0.0
    )
    for pre_window in g19z.RECROSS_PRE_WINDOWS_HOURS:
        pre_offsets = tuple(range(-pre_window, 0))
        before = indexed_close_windows(close, indexes, pre_offsets)
        before_previous = indexed_close_windows(
            close, indexes, (-pre_window - 1,)
        )[:, 0]
        pre_count = crossing_count(before, levels, before_previous)
        frame[f"pre_crossings_h{pre_window}"] = pre_count
        for horizon in HORIZONS:
            post_count = pd.to_numeric(
                frame[f"metric__crossing_count_h{horizon}"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(pre_count) & np.isfinite(post_count)
            states = {
                "new_onset": (pre_count == 0.0) & (post_count >= 1.0),
                "persistent": (pre_count >= 1.0) & (post_count >= 1.0),
                "pre_only": (pre_count >= 1.0) & (post_count == 0.0),
                "quiet": (pre_count == 0.0) & (post_count == 0.0),
            }
            for state, mask in states.items():
                frame[
                    f"metric__{state}_pre{pre_window}_post{horizon}"
                ] = np.where(valid, mask.astype(float), np.nan)


def metric_summary(frame: DataFrame, *, timing: bool) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    rows: list[DataFrame] = []
    if not timing:
        questions = [
            (metric, horizon, f"metric__{metric}_h{horizon}")
            for metric in ("crossing_count", "any_recross", "repeated_recross")
            for horizon in HORIZONS
        ]
    else:
        questions = [
            (
                f"{state}_after_pre{pre_window}",
                horizon,
                f"metric__{state}_pre{pre_window}_post{horizon}",
            )
            for state in ("new_onset", "persistent", "pre_only", "quiet")
            for pre_window in g19z.RECROSS_PRE_WINDOWS_HOURS
            for horizon in HORIZONS
        ]
        questions.append(("recross_immediacy", 8, "metric__recross_immediacy_h8"))
    for metric, horizon, column in questions:
        selected = frame[keys + [column]].dropna(subset=[column])
        grouped = (
            selected.groupby(keys, observed=True, sort=False)[column]
            .agg(["mean", "size"])
            .reset_index()
            .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
        )
        grouped["metric"] = metric
        grouped["horizon_hours"] = horizon
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def analyse(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    coordinate_summary = metric_summary(frame, timing=False)
    coordinate_contrasts = g18d.paired_contrasts(coordinate_summary, CONTROLS)
    timing = frame.loc[frame["scope_kind"].eq("family_any_level")].copy()
    timing_summary = metric_summary(timing, timing=True)
    timing_contrasts = g18d.paired_contrasts(timing_summary, CONTROLS)
    return coordinate_contrasts, timing_contrasts


def decisions(contrasts: DataFrame) -> tuple[DataFrame, DataFrame]:
    keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, keys)
    decided = g18d.whole_decisions(scores, keys, CONTROLS)
    return scores, decided


def market_specific_decisions(coordinate: DataFrame) -> DataFrame:
    definitions = (
        (
            "top_ten_memes__donchian_boundaries",
            "top_ten_memes",
            "family_any_level",
            "donchian_boundaries",
            "all_normal",
        ),
        (
            "top_ten_memes__vp_lb72_bins96_nearest_lvn_q10",
            "top_ten_memes",
            "single_level",
            "vp_lb72_bins96_nearest_lvn_q10",
            "all_normal",
        ),
        (
            "all_normal__rolling_vwap_deviation_bands",
            "all_normal",
            "family_any_level",
            "rolling_vwap_deviation_bands",
            "top_ten_memes",
        ),
        (
            "btc_separate__adaptive_volume_profile_nodes",
            "btc_separate",
            "family_any_level",
            "adaptive_volume_profile_nodes",
            "other_established_alts",
        ),
    )
    rows: list[DataFrame] = []
    for mechanism, scope, kind, value, nonmember_scope in definitions:
        selected = coordinate.loc[
            coordinate["market_scope"].eq(scope)
            & coordinate["scope_kind"].eq(kind)
            & coordinate["scope_value"].eq(value)
        ].copy()
        if selected.empty:
            continue
        nonmember = coordinate.loc[
            coordinate["market_scope"].eq(nonmember_scope)
            & coordinate["scope_kind"].eq(kind)
            & coordinate["scope_value"].eq(value),
            ["metric", "horizon_hours", "status"],
        ].rename(columns={"status": "matched_nonmember_status"})
        selected = selected.merge(
            nonmember,
            on=["metric", "horizon_hours"],
            how="left",
            validate="one_to_one",
        )
        selected["mechanism_id"] = mechanism
        selected["matched_nonmember_scope"] = nonmember_scope
        selected["market_specific_interpretation"] = np.where(
            selected["status"].eq("strict_holdout_confirmation")
            & ~selected["matched_nonmember_status"].eq(
                "strict_holdout_confirmation"
            ),
            "strict_only_under_predeclared_market_label",
            np.where(
                selected["status"].eq("strict_holdout_confirmation"),
                "strict_but_not_unique_to_market_label",
                "not_retained_under_predeclared_market_label",
            ),
        )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else DataFrame()


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "coordinate_contrasts": run_dir / "g19_coordinate_pair_contrasts.csv",
        "coordinate_scores": run_dir / "g19_coordinate_period_scores.csv",
        "coordinate_decisions": run_dir / "g19_coordinate_decisions.csv",
        "market_mechanism_decisions": run_dir / "g19_market_mechanism_decisions.csv",
        "timing_contrasts": run_dir / "g19_recross_timing_pair_contrasts.csv",
        "timing_scores": run_dir / "g19_recross_timing_period_scores.csv",
        "timing_decisions": run_dir / "g19_recross_timing_decisions.csv",
        "result": run_dir / "g19_recross_analysis_result.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    load_freeze()
    run_dir = RECORD_ROOT / args.run_id
    paths = output_paths(run_dir)
    if paths["result"].is_file() and not args.overwrite:
        print(paths["result"].read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    coordinate_parts: list[DataFrame] = []
    timing_parts: list[DataFrame] = []
    inventory: list[dict[str, Any]] = []
    event_paths = sorted(ATLAS_EVENT_ROOT.glob("*.parquet"))
    if len(event_paths) != 20:
        raise ValueError(f"Expected 20 pair-event files, got {len(event_paths)}.")
    for number, path in enumerate(event_paths, start=1):
        raw = read_later_rows(path)
        if raw.empty:
            raise ValueError(f"No later rows in {path}.")
        scoped = coordinate_scopes(raw)
        pair = str(scoped["pair"].iloc[0])
        add_crossing_paths(scoped, pair)
        coordinate, timing = analyse(scoped)
        coordinate_parts.append(coordinate)
        timing_parts.append(timing)
        inventory.append(
            {
                "pair": pair,
                "cohort": str(scoped["cohort"].iloc[0]),
                "scoped_rows": len(scoped),
                "source_path": str(path.resolve()),
                "source_sha256": g0.sha256_file(path),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "g19_recross_pair",
                    "processed": number,
                    "total": len(event_paths),
                    "pair": pair,
                }
            ),
            flush=True,
        )
    coordinate_contrasts = pd.concat(coordinate_parts, ignore_index=True, sort=False)
    timing_contrasts = pd.concat(timing_parts, ignore_index=True, sort=False)
    coordinate_scores, coordinate_decisions = decisions(coordinate_contrasts)
    timing_scores, timing_decisions = decisions(timing_contrasts)
    mechanisms = market_specific_decisions(coordinate_decisions)
    outputs = {
        "coordinate_contrasts": coordinate_contrasts,
        "coordinate_scores": coordinate_scores,
        "coordinate_decisions": coordinate_decisions,
        "market_mechanism_decisions": mechanisms,
        "timing_contrasts": timing_contrasts,
        "timing_scores": timing_scores,
        "timing_decisions": timing_decisions,
    }
    for name, frame in outputs.items():
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation19_recross_analysis",
        "branches_completed": [
            "g19a_exact_coordinate_recross_specificity",
            "g19c_market_specific_level_mechanisms",
            "g19d_recross_timing_and_activity_onset",
        ],
        "pair_tasks_completed": len(inventory),
        "strict_coordinate_rows": int(
            coordinate_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "strict_market_specific_rows": int(
            mechanisms["market_specific_interpretation"]
            .eq("strict_only_under_predeclared_market_label")
            .sum()
        )
        if len(mechanisms)
        else 0,
        "strict_timing_rows": int(
            timing_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "direction_branch_remained_parked": True,
        },
        "source_contracts": {"generation19_freeze": artifact(g19z.FREEZE_PATH)},
        "inventory": inventory,
        "artifacts": {name: artifact(paths[name]) for name in outputs},
        "result_path": str(paths["result"].resolve()),
    }
    g0.atomic_write_json(result, paths["result"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
