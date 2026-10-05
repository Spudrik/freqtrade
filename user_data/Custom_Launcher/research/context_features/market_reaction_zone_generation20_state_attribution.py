"""Run Generation 20 siblings A-C with outcome-blind state matching."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation1_localization as g1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_freeze as g20z,
)


DEFAULT_RUN_ID = "g20_state_attribution_20260827a"
DEFAULT_SUPPORT_ID = "g20_state_support_20260827a"
ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = g20z.OUTPUT_ROOT / "state_attribution"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation20_branches"
    / "g20_broad_siblings"
    / "state_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g20_state_support_freeze.json"
ATLAS_EVENT_ROOT = g17l.ARTIFACT_ROOT / g17l.DEFAULT_RUN_ID / "pair_events"
HORIZONS = g20z.HORIZONS_HOURS
LOCATION_CONTROLS = ("matched_random_time", "near_miss", "stale_72h", "price_shift")
DONCHIAN_CONTROLS = (
    "breakout_state_matched_no_current_boundary_contact",
    "near_miss",
    "stale_72h",
    "price_shift",
)
PRE_WINDOWS = (2, 4)
MIN_PAIR_ROWS = 10
RATIONAL_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "rolling_vwap_deviation_bands",
    "donchian_boundaries",
    "weekly_pivot_grid",
)
META_COLUMNS = (
    "cohort",
    "pair",
    "control",
    "event_time",
    "level_family",
    "level_name",
    "level_price",
    "zone_half_width",
    "base_atr",
    "pre_distance_atr",
    "approach_state",
)
ACTIVITY_COLUMNS = (
    "relative_volume",
    "volume_acceleration",
    "prior_range_atr",
    "atr_fraction",
)


def artifact(path: Path) -> dict[str, Any]:
    return g20z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g20z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation20_outcomes":
        raise ValueError("Generation 20 batch is not frozen.")
    return frozen


def manifest_for_cohort(cohort: str) -> dict[str, Any]:
    path = g0.DEFAULT_MANIFEST if cohort == "normal" else g17l.MEME_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def base_and_state(pair: str, cohort: str) -> tuple[DataFrame, DataFrame]:
    base = g0.prepare_base_market_frame(pair, manifest_for_cohort(cohort))
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    prior_close = close.shift(1)
    prior_volume = volume.shift(1)
    prior_range = (high - low).shift(1)
    prior_atr = pd.to_numeric(base["base_atr"], errors="coerce")
    state = DataFrame({"date": pd.to_datetime(base["date"], utc=True)})
    state["relative_volume"] = prior_volume.div(
        prior_volume.rolling(24, min_periods=24).median()
    )
    state["volume_acceleration"] = np.log1p(prior_volume.clip(lower=0.0)).diff(3)
    state["prior_range_atr"] = prior_range.div(prior_atr)
    state["atr_fraction"] = prior_atr.div(prior_close.abs())
    return base, state


def empirical_percentile(values: Series, calibration: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    calibration = np.sort(calibration[np.isfinite(calibration)])
    valid = np.isfinite(numeric) & (len(calibration) > 0)
    if valid.any():
        result[valid] = np.searchsorted(
            calibration, numeric[valid], side="right"
        ) / len(calibration)
    return result


def add_activity_states(
    state: DataFrame, cohort: str
) -> tuple[DataFrame, dict[str, Any]]:
    first_period = min(
        pd.Timestamp(item["start_utc"])
        for item in g18d.g18z.CONFIRMATION_PERIODS[cohort]
    )
    calibration_mask = state["date"] < first_period
    percentiles: list[np.ndarray] = []
    calibration_counts: dict[str, int] = {}
    for column in ACTIVITY_COLUMNS:
        calibration = pd.to_numeric(
            state.loc[calibration_mask, column], errors="coerce"
        ).to_numpy(dtype=float)
        if column == "volume_acceleration":
            calibration = np.abs(calibration)
            values = pd.to_numeric(state[column], errors="coerce").abs()
        else:
            values = state[column]
        calibration_counts[column] = int(np.isfinite(calibration).sum())
        percentiles.append(empirical_percentile(values, calibration))
    matrix = np.column_stack(percentiles)
    state = state.copy()
    finite_count = np.isfinite(matrix).sum(axis=1)
    state["activity_score"] = np.divide(
        np.nansum(matrix, axis=1),
        finite_count,
        out=np.full(len(state), np.nan),
        where=finite_count > 0,
    )
    calibration_score = pd.to_numeric(
        state.loc[calibration_mask, "activity_score"], errors="coerce"
    ).dropna()
    if len(calibration_score) < 168:
        raise ValueError("Insufficient causal development rows for activity tertiles.")
    lower, upper = calibration_score.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
    state["activity_state"] = np.select(
        [state["activity_score"] <= lower, state["activity_score"] > upper],
        ["quiet", "extreme"],
        default="moderate",
    )
    state.loc[state["activity_score"].isna(), "activity_state"] = "unavailable"
    return state, {
        "calibration_end_exclusive_utc": first_period.isoformat(),
        "lower_tertile": float(lower),
        "upper_tertile": float(upper),
        "calibration_rows": len(calibration_score),
        "feature_calibration_rows": calibration_counts,
        "volume_acceleration_transform": "absolute magnitude before percentile rank",
    }


def indexed_windows(values: np.ndarray, indexes: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    output = np.full((len(indexes), len(offsets)), np.nan, dtype=float)
    for column, offset in enumerate(offsets):
        positions = indexes + int(offset)
        valid = (positions >= 0) & (positions < len(values))
        output[valid, column] = values[positions[valid]]
    return output


def attach_causal_state(events: DataFrame, base: DataFrame, state: DataFrame) -> None:
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    positions_by_date = Series(np.arange(len(base), dtype=np.int64), index=dates)
    positions = events["event_time"].map(positions_by_date).to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("An outcome-blind support event is absent from causal OHLCV.")
    indexes = positions.astype(np.int64)
    state_by_date = state.set_index("date")
    for column in (*ACTIVITY_COLUMNS, "activity_score", "activity_state"):
        events[column] = events["event_time"].map(state_by_date[column]).to_numpy()
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(events["base_atr"], errors="coerce").to_numpy(dtype=float)
    levels = pd.to_numeric(events["level_price"], errors="coerce").to_numpy(dtype=float)
    for window in PRE_WINDOWS:
        offsets = tuple(range(-window, 0))
        prior_close = indexed_windows(close, indexes, offsets)
        before = indexed_windows(close, indexes, (-window - 1,))[:, 0]
        prior_high = indexed_windows(high, indexes, offsets)
        prior_low = indexed_windows(low, indexes, offsets)
        events[f"pre_crossings_h{window}"] = g19_crossing_count(
            prior_close, levels, before
        )
        events[f"pre_range_over_atr_h{window}"] = np.nanmean(
            prior_high - prior_low, axis=1
        ) / atr
        events[f"pre_abs_return_over_atr_h{window}"] = np.abs(
            prior_close[:, -1] - before
        ) / atr
        span_high = np.nanmax(prior_high, axis=1)
        span_low = np.nanmin(prior_low, axis=1)
        span = span_high - span_low
        events[f"pre_close_location_h{window}"] = np.divide(
            prior_close[:, -1] - span_low,
            span,
            out=np.full(len(events), np.nan),
            where=np.isfinite(span) & (span > 0.0),
        )
        events[f"pre_trend_over_atr_h{window}"] = (
            prior_close[:, -1] - before
        ) / atr


def g19_crossing_count(
    close_path: np.ndarray, levels: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    signs = np.sign(close_path - levels[:, None])
    combined = np.column_stack([np.sign(previous - levels), signs])
    valid = np.isfinite(combined).all(axis=1)
    output = np.full(len(levels), np.nan, dtype=float)
    output[valid] = ((combined[valid, :-1] * combined[valid, 1:]) < 0.0).sum(
        axis=1
    )
    return output


def later_meta(path: Path) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(META_COLUMNS))
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    cohorts = frame["cohort"].dropna().astype(str).unique()
    if len(cohorts) != 1:
        raise ValueError(f"Unexpected cohorts in {path}: {cohorts}")
    frame["period"] = g18d.assign_confirmation_period(frame["event_time"], cohorts[0])
    return frame.loc[frame["period"].ne("outside_g18_confirmation")].copy()


def nearest_family(frame: DataFrame, family: str) -> DataFrame:
    selected = frame.loc[frame["level_family"].eq(family)].copy()
    keys = ["cohort", "pair", "period", "control", "event_time", "level_family"]
    return g18d.nearest_anchor(selected, keys)


def nearest_rational_level(frame: DataFrame) -> DataFrame:
    selected = frame.loc[frame["level_family"].isin(RATIONAL_FAMILIES)].copy()
    keys = ["cohort", "pair", "period", "control", "event_time"]
    return g18d.nearest_anchor(selected, keys)


def donchian_no_contact_pool(base: DataFrame, pair: str, cohort: str) -> DataFrame:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    atr = pd.to_numeric(base["base_atr"], errors="coerce")
    dates = pd.to_datetime(base["date"], utc=True)
    period = g18d.assign_confirmation_period(dates, cohort)
    rows: list[DataFrame] = []
    for lookback in (24, 72):
        levels = {
            f"donchian_lb{lookback}_upper": high.shift(1).rolling(
                lookback, min_periods=lookback
            ).max(),
            f"donchian_lb{lookback}_lower": low.shift(1).rolling(
                lookback, min_periods=lookback
            ).min(),
        }
        for level_name, level in levels.items():
            width = 0.25 * atr
            contact = (high >= level - width) & (low <= level + width)
            valid = (
                level.notna()
                & width.gt(0.0)
                & ~contact
                & period.ne("outside_g18_confirmation")
            )
            pre_close = close.shift(1)
            approach = np.select(
                [pre_close < level - width, pre_close > level + width],
                ["from_below", "from_above"],
                default="already_inside_or_unclear",
            )
            part = DataFrame(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "control": "breakout_state_matched_no_current_boundary_contact",
                    "event_time": dates,
                    "period": period,
                    "level_family": "donchian_boundaries",
                    "level_name": level_name,
                    "level_price": level,
                    "zone_half_width": width,
                    "base_atr": atr,
                    "pre_distance_atr": (pre_close - level).abs().div(atr),
                    "approach_state": approach,
                    "source_row": -1,
                }
            )
            rows.append(part.loc[valid])
    return pd.concat(rows, ignore_index=True, sort=False)


def build_pair_support(path: Path, overwrite: bool) -> dict[str, Any]:
    raw = later_meta(path)
    pair = str(raw["pair"].iloc[0])
    cohort = str(raw["cohort"].iloc[0])
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["branch_id"])
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "existing",
        }
    base, state = base_and_state(pair, cohort)
    state, calibration = add_activity_states(state, cohort)
    parts: list[DataFrame] = []
    a = nearest_family(raw, "adaptive_volume_profile_nodes")
    a = a.loc[a["control"].isin({"actual", *LOCATION_CONTROLS})].copy()
    a["branch_id"] = "g20a_volume_profile_traffic_state_attribution"
    parts.append(a)
    if cohort == "normal":
        b = nearest_family(raw, "donchian_boundaries")
        b = b.loc[b["control"].isin({"actual", *DONCHIAN_CONTROLS})].copy()
        no_contact = donchian_no_contact_pool(base, pair, cohort)
        b = pd.concat([b, no_contact], ignore_index=True, sort=False)
        b["branch_id"] = "g20b_donchian_onset_causal_isolation"
        parts.append(b)
        c = nearest_rational_level(raw)
        c = c.loc[c["control"].isin({"actual", *LOCATION_CONTROLS})].copy()
        c["branch_id"] = "g20c_level_local_activity_regime_stability"
        parts.append(c)
    support = pd.concat(parts, ignore_index=True, sort=False)
    attach_causal_state(support, base, state)
    support.sort_values(
        ["branch_id", "control", "event_time", "level_name"],
        inplace=True,
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "branches": sorted(support["branch_id"].unique()),
        "activity_calibration": calibration,
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation20_state_outcomes":
            raise ValueError("Invalid Generation 20 state support freeze.")
        return manifest
    event_paths = sorted(ATLAS_EVENT_ROOT.glob("*.parquet"))
    if len(event_paths) != 20:
        raise ValueError(f"Expected 20 atlas pair files, got {len(event_paths)}.")
    inventory = []
    for number, path in enumerate(event_paths, start=1):
        inventory.append(build_pair_support(path, overwrite))
        print(
            json.dumps(
                {"phase": "g20_state_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation20_state_outcomes",
        "branches": [
            "g20a_volume_profile_traffic_state_attribution",
            "g20b_donchian_onset_causal_isolation",
            "g20c_level_local_activity_regime_stability",
        ],
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "activity_tertiles_frozen_from_pre_confirmation_rows": True,
        "activity_score_inputs": list(ACTIVITY_COLUMNS),
        "inventory": inventory,
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def add_future_metrics(events: DataFrame, base: DataFrame) -> None:
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    positions_by_date = Series(np.arange(len(base), dtype=np.int64), index=dates)
    positions = events["event_time"].map(positions_by_date).to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("A frozen support event is absent from outcome OHLCV.")
    indexes = positions.astype(np.int64)
    paths = g0.future_path_matrices(base, max(HORIZONS))
    levels = pd.to_numeric(events["level_price"], errors="coerce").to_numpy(dtype=float)
    widths = pd.to_numeric(events["zone_half_width"], errors="coerce").to_numpy(dtype=float)
    previous = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)[
        indexes
    ]
    pre_volume = pd.to_numeric(
        base["pre_volume_median_24"], errors="coerce"
    ).to_numpy(dtype=float)[indexes]
    future_close = paths["close"][indexes]
    future_volume = paths["volume"][indexes]
    for horizon in HORIZONS:
        valid = np.isfinite(future_close[:, :horizon]).all(axis=1)
        crossing = g19_crossing_count(
            future_close[:, :horizon], levels, previous
        )
        crossing[~valid] = np.nan
        events[f"metric__crossing_count_h{horizon}"] = crossing
        events[f"metric__any_recross_h{horizon}"] = np.where(
            np.isfinite(crossing), (crossing >= 1.0).astype(float), np.nan
        )
        events[f"metric__repeated_recross_h{horizon}"] = np.where(
            np.isfinite(crossing), (crossing >= 2.0).astype(float), np.nan
        )
        events[f"metric__future_volume_ratio_h{horizon}"] = np.divide(
            np.nanmean(future_volume[:, :horizon], axis=1),
            pre_volume,
            out=np.full(len(events), np.nan),
            where=np.isfinite(pre_volume) & (pre_volume > 0.0) & valid,
        )
        events[f"metric__dwell_fraction_h{horizon}"] = np.where(
            valid,
            np.mean(
                np.abs(future_close[:, :horizon] - levels[:, None])
                <= widths[:, None],
                axis=1,
            ),
            np.nan,
        )


def matched_pair_rows(
    actual: DataFrame,
    control: DataFrame,
    *,
    state_columns: Sequence[str],
) -> tuple[DataFrame, dict[str, int]]:
    left = actual.copy().reset_index(drop=True)
    right = control.copy().reset_index(drop=True)
    # Geometry is deliberately neutral here: the frozen questions match market state,
    # while all source rows already obey their contact/no-contact control definition.
    left["pre_distance_atr"] = 0.0
    right["pre_distance_atr"] = 0.0
    pairs, audit = g1.nearest_state_pairs(
        left,
        right,
        state_columns=state_columns,
        pre_distance_atr_caliper=0.10,
        minimum_event_separation_hours=max(HORIZONS),
    )
    if not pairs:
        return DataFrame(), audit
    left_selected = left.iloc[[item[0] for item in pairs]].reset_index(drop=True)
    right_selected = right.iloc[[item[1] for item in pairs]].reset_index(drop=True)
    output = DataFrame(
        {
            "cohort": left_selected["cohort"].astype(str),
            "pair": left_selected["pair"].astype(str),
            "period": left_selected["period"].astype(str),
            "actual_event_time": left_selected["event_time"],
            "control_event_time": right_selected["event_time"],
            "level_name": left_selected["level_name"].astype(str),
            "approach_state": left_selected["approach_state"].astype(str),
            "match_distance": [item[2] for item in pairs],
            "pre_distance_atr_abs_difference": 0.0,
        }
    )
    for column in left.columns:
        if column.startswith("metric__"):
            output[f"actual__{column}"] = pd.to_numeric(
                left_selected[column], errors="coerce"
            ).to_numpy(dtype=float)
            output[f"control__{column}"] = pd.to_numeric(
                right_selected[column], errors="coerce"
            ).to_numpy(dtype=float)
    return output, audit


def compare_cell(
    frame: DataFrame,
    *,
    comparison_names: Sequence[str],
    state_columns: Sequence[str],
    targets: Sequence[tuple[str, int, str]],
    dimensions: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (pair, period), pair_period in frame.groupby(
        ["pair", "period"], observed=True, sort=False
    ):
        actual = pair_period.loc[pair_period["control"].eq("actual")]
        for comparison in comparison_names:
            control = pair_period.loc[pair_period["control"].eq(comparison)]
            matched, audit = matched_pair_rows(
                actual, control, state_columns=state_columns
            )
            if matched.empty:
                continue
            matched["comparison"] = comparison
            matched = g13d.purge_overlapping_hourly_pairs(
                matched,
                separation_hours=max(HORIZONS),
                group_columns=["pair", "period", "comparison"],
            )
            for metric, horizon, column in targets:
                actual_values = pd.to_numeric(
                    matched[f"actual__{column}"], errors="coerce"
                )
                control_values = pd.to_numeric(
                    matched[f"control__{column}"], errors="coerce"
                )
                valid = actual_values.notna() & control_values.notna()
                difference = actual_values.loc[valid] - control_values.loc[valid]
                rows.append(
                    {
                        **dimensions,
                        "cohort": str(actual["cohort"].iloc[0]),
                        "pair": str(pair),
                        "period": str(period),
                        "comparison": comparison,
                        "metric": metric,
                        "horizon_hours": horizon,
                        "event_rows_actual": int(valid.sum()),
                        "event_rows_control": int(valid.sum()),
                        "outcome_mean_actual": float(actual_values.loc[valid].mean())
                        if valid.any()
                        else np.nan,
                        "outcome_mean_control": float(control_values.loc[valid].mean())
                        if valid.any()
                        else np.nan,
                        "difference": float(difference.mean())
                        if len(difference)
                        else np.nan,
                        "eligible_pair": bool(valid.sum() >= MIN_PAIR_ROWS),
                        "eligible_actual_before_matching": int(
                            audit["eligible_actual"]
                        ),
                        "state_matchable_actual": int(
                            audit["state_matchable_actual"]
                        ),
                    }
                )
    return rows


def route_results(events: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    contrasts: list[dict[str, Any]] = []
    a = events.loc[
        events["branch_id"].eq("g20a_volume_profile_traffic_state_attribution")
    ]
    a_targets = [
        (metric, horizon, f"metric__{metric}_h{horizon}")
        for metric in ("any_recross", "repeated_recross", "crossing_count")
        for horizon in HORIZONS
    ]
    for window in PRE_WINDOWS:
        contrasts.extend(
            compare_cell(
                a,
                comparison_names=LOCATION_CONTROLS,
                state_columns=(
                    f"pre_crossings_h{window}",
                    "relative_volume",
                    f"pre_range_over_atr_h{window}",
                    f"pre_abs_return_over_atr_h{window}",
                ),
                targets=a_targets,
                dimensions={
                    "branch_id": "g20a_volume_profile_traffic_state_attribution",
                    "pre_window_hours": window,
                    "activity_state": "all",
                },
            )
        )
    b = events.loc[
        events["branch_id"].eq("g20b_donchian_onset_causal_isolation")
    ]
    for window in PRE_WINDOWS:
        local = b.copy()
        targets = []
        for horizon in HORIZONS:
            column = f"metric__new_onset_pre{window}_post{horizon}"
            crossing = pd.to_numeric(
                local[f"metric__crossing_count_h{horizon}"], errors="coerce"
            )
            pre = pd.to_numeric(local[f"pre_crossings_h{window}"], errors="coerce")
            valid = crossing.notna() & pre.notna()
            local[column] = np.where(
                valid, ((pre == 0.0) & (crossing >= 1.0)).astype(float), np.nan
            )
            targets.append(("new_onset", horizon, column))
        contrasts.extend(
            compare_cell(
                local,
                comparison_names=DONCHIAN_CONTROLS,
                state_columns=(
                    f"pre_abs_return_over_atr_h{window}",
                    f"pre_range_over_atr_h{window}",
                    "relative_volume",
                    f"pre_close_location_h{window}",
                    f"pre_trend_over_atr_h{window}",
                ),
                targets=targets,
                dimensions={
                    "branch_id": "g20b_donchian_onset_causal_isolation",
                    "pre_window_hours": window,
                    "activity_state": "all",
                },
            )
        )
    c = events.loc[
        events["branch_id"].eq("g20c_level_local_activity_regime_stability")
    ]
    c_targets = [
        (
            "future_volume_ratio",
            horizon,
            f"metric__future_volume_ratio_h{horizon}",
        )
        for horizon in HORIZONS
    ]
    for state in ("quiet", "moderate", "extreme"):
        contrasts.extend(
            compare_cell(
                c.loc[c["activity_state"].eq(state)],
                comparison_names=LOCATION_CONTROLS,
                state_columns=ACTIVITY_COLUMNS,
                targets=c_targets,
                dimensions={
                    "branch_id": "g20c_level_local_activity_regime_stability",
                    "pre_window_hours": 0,
                    "activity_state": state,
                },
            )
        )
    contrast_frame = DataFrame.from_records(contrasts)
    if contrast_frame.empty:
        raise ValueError("No Generation 20 A-C matched contrasts were produced.")
    question_keys = (
        "branch_id",
        "pre_window_hours",
        "activity_state",
        "metric",
        "horizon_hours",
    )
    scores = g18d.period_scores(contrast_frame, question_keys)
    decisions: list[DataFrame] = []
    for branch_id, controls in (
        ("g20a_volume_profile_traffic_state_attribution", LOCATION_CONTROLS),
        ("g20b_donchian_onset_causal_isolation", DONCHIAN_CONTROLS),
        ("g20c_level_local_activity_regime_stability", LOCATION_CONTROLS),
    ):
        selected = scores.loc[scores["branch_id"].eq(branch_id)]
        decisions.append(g18d.whole_decisions(selected, question_keys, controls))
    return contrast_frame, scores, pd.concat(decisions, ignore_index=True, sort=False)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    if (
        g0.sha256_file(ANALYSIS_PATH)
        != manifest["source_contracts"]["analysis_script"]["sha256"]
    ):
        raise ValueError("Generation 20 state analysis changed after support freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g20_state_attribution_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        support_path = Path(item["path"])
        if g0.sha256_file(support_path) != item["sha256"]:
            raise ValueError(f"Frozen support changed: {support_path}")
        frame = pd.read_parquet(support_path)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        base, _ = base_and_state(str(item["pair"]), str(item["cohort"]))
        add_future_metrics(frame, base)
        parts.append(frame)
        print(
            json.dumps(
                {"phase": "g20_state_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    events = pd.concat(parts, ignore_index=True, sort=False)
    contrasts, scores, decisions = route_results(events)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g20_state_pair_contrasts.csv",
        "scores": run_dir / "g20_state_period_scores.csv",
        "decisions": run_dir / "g20_state_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation20_state_attribution",
        "branches_completed": [
            "g20a_volume_profile_traffic_state_attribution",
            "g20b_donchian_onset_causal_isolation",
            "g20c_level_local_activity_regime_stability",
        ],
        "strict_rows": int(
            decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "point_rows": int(
            decisions["status"].eq("point_holdout_confirmation").sum()
        ),
        "strict_by_branch": {
            str(key): int(value)
            for key, value in decisions.loc[
                decisions["status"].eq("strict_holdout_confirmation")
            ]
            .groupby("branch_id")
            .size()
            .items()
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "freqai_descendant_run": False,
        },
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
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
