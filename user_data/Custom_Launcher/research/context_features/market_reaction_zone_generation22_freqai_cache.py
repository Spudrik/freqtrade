"""Build Generation 22's outcome-blind features and unsigned FreqAI targets."""

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
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_multitimeframe_convergence as g22b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)


ANALYSIS_PATH = Path(__file__).resolve()
CACHE_ID = "g22_freqai_unsigned_reaction_20260827a"
RECORD_ROOT = g22z.OUTPUT_ROOT / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "freqai_cache"
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g22_freqai_profile_registry.json"
CONTEXT_PATH = ARTIFACT_ROOT / "g22_causal_cross_asset_context.parquet"
SEED = 2026082722
HORIZONS = tuple(g22z.HORIZONS_HOURS)
READY_BLOCK = "g22_unsigned_reaction_features"
ZONE_HALF_WIDTH_ATR = 0.25
SHUFFLE_LAGS_HOURS = (168, 336, 504, 672)

FEATURE_COLUMNS = (
    "nearest_level_distance_atr",
    "independent_level_family_count_within_0p5atr",
    "source_role_fixed_encoding",
    "source_timeframe_fixed_encoding",
    "pre_crossing_count_4h",
    "relative_volume",
    "range_over_atr_4h",
    "absolute_return_over_atr_4h",
    "atr_fraction",
    "btc_absolute_activity_4h",
    "equal_weight_market_activity_4h",
    "cross_coin_dispersion_4h",
)
LEVEL_FEATURES = FEATURE_COLUMNS[:5]
STATE_FEATURES = FEATURE_COLUMNS[5:]
TARGET_METRICS = (
    "maximum_absolute_excursion_atr",
    "future_volume_ratio",
    "future_range_ratio",
    "crossing_count",
    "dwell_fraction",
)
TARGETS = tuple(
    f"&-g22_{metric}_h{horizon}"
    for metric in TARGET_METRICS
    for horizon in HORIZONS
)
ROLE_ENCODING = {"lower": 0.0, "low": 1.0, "pivot": 2.0, "poc": 3.0, "upper": 4.0, "high": 5.0}
TIMEFRAME_ENCODING = {"1h": 0.0, "4h": 1.0, "1d": 2.0, "1w": 3.0}
PROFILE_ROLES = (
    "full_interaction",
    "level_geometry_only",
    "market_state_only",
    "within_pair_time_shuffled_training_labels",
)


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def support_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"


def cache_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_manifest.json"


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g22z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22e_freqai_unsigned_reaction_interaction_regression"
    )
    if frozen.get("status") != "frozen_before_generation22_outcomes":
        raise ValueError("Generation 22 batch is not frozen.")
    if tuple(branch["features"]) != FEATURE_COLUMNS:
        raise ValueError("Generation 22 FreqAI feature contract drifted.")
    expected_targets = tuple(
        f"{metric}_h{horizon}" for metric in TARGET_METRICS for horizon in HORIZONS
    )
    if tuple(branch["continuous_targets"]) != expected_targets:
        raise ValueError("Generation 22 FreqAI target contract drifted.")
    if branch.get("required_tool") != "FreqAI":
        raise ValueError("Generation 22 FreqAI route lost its required tool.")
    return frozen


def profile_id(cohort: str, role: str) -> str:
    return f"g22__{cohort}__{role}__seed{SEED}"


def build_registry() -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        for role in PROFILE_ROLES:
            identifier = profile_id(cohort, role)
            if role == "level_geometry_only":
                columns = LEVEL_FEATURES
            elif role == "market_state_only":
                columns = STATE_FEATURES
            else:
                columns = FEATURE_COLUMNS
            profiles[identifier] = {
                "profile_id": identifier,
                "cohort": cohort,
                "role": role,
                "feature_columns": list(columns),
                "required_ready_blocks": [READY_BLOCK],
                "targets": list(TARGETS),
                "seed": SEED,
                "target_cache": (
                    "shuffled"
                    if role == "within_pair_time_shuffled_training_labels"
                    else "actual"
                ),
            }
        candidate = profile_id(cohort, "full_interaction")
        for control in (
            "constant_training_median",
            "level_geometry_only_model",
            "market_state_only_model",
            "within_pair_time_shuffled_training_labels",
        ):
            baseline_role = control.removesuffix("_model")
            comparisons.append(
                {
                    "comparison_id": f"{cohort}__full_vs_{control}",
                    "cohort": cohort,
                    "route_id": "full_interaction",
                    "control_type": control,
                    "candidate": candidate,
                    "baseline": (
                        None
                        if control == "constant_training_median"
                        else profile_id(cohort, baseline_role)
                    ),
                    "targets": list(TARGETS),
                    "expected_controls_for_route": 4,
                }
            )
    return {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_freqai_target_materialization",
        "profiles": profiles,
        "comparisons": comparisons,
        "feature_columns": list(FEATURE_COLUMNS),
        "targets": list(TARGETS),
        "role_encoding": ROLE_ENCODING,
        "timeframe_encoding": TIMEFRAME_ENCODING,
        "shuffled_label_control": {
            "method": "within-pair deterministic earlier-date time shift",
            "lags_hours": list(SHUFFLE_LAGS_HOURS),
            "future_label_source_allowed": False,
        },
    }


def freeze_registry() -> dict[str, Any]:
    registry = build_registry()
    if REGISTRY_PATH.is_file():
        existing = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 22 FreqAI registry changed after its freeze.")
        return existing
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(registry, REGISTRY_PATH)
    return registry


def feature_surface(base: DataFrame, context: DataFrame, pair: str) -> tuple[DataFrame, DataFrame]:
    surfaces = g22b.multitimeframe_surfaces(base)
    levels = np.column_stack([np.asarray(item["level"], dtype=float) for item in surfaces])
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    valid_scale = np.isfinite(atr) & (atr > 0.0) & np.isfinite(pre_close)
    distance = np.abs(levels - pre_close[:, None]) / atr[:, None]
    distance[~np.isfinite(distance)] = np.inf
    nearest_index = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(len(base)), nearest_index]
    nearest_valid = np.isfinite(nearest_distance) & valid_scale
    nearest_distance[~nearest_valid] = np.nan
    nearest_level = levels[np.arange(len(base)), nearest_index]
    nearest_level[~nearest_valid] = np.nan

    roles = np.asarray([str(item["role"]) for item in surfaces], dtype=object)
    timeframes = np.asarray([str(item["source_timeframe"]) for item in surfaces], dtype=object)
    role_encoding = np.asarray([ROLE_ENCODING[str(value)] for value in roles], dtype=float)
    timeframe_encoding = np.asarray(
        [TIMEFRAME_ENCODING[str(value)] for value in timeframes], dtype=float
    )
    nearest_role = role_encoding[nearest_index]
    nearest_timeframe = timeframe_encoding[nearest_index]
    nearest_role[~nearest_valid] = np.nan
    nearest_timeframe[~nearest_valid] = np.nan

    family_count = np.zeros(len(base), dtype=float)
    families = np.asarray([str(item["family"]) for item in surfaces], dtype=object)
    close_to_price = distance <= 0.5
    for family in sorted(set(families)):
        family_count += close_to_price[:, families == family].any(axis=1).astype(float)
    family_count[~valid_scale] = np.nan

    close = pd.to_numeric(base["close"], errors="coerce")
    signs = np.column_stack(
        [np.sign(close.shift(lag).to_numpy(dtype=float) - nearest_level) for lag in range(1, 6)]
    )
    pre_crossings = ((signs[:, :-1] * signs[:, 1:]) < 0.0).sum(axis=1).astype(float)
    pre_crossings[~np.isfinite(signs).all(axis=1) | ~nearest_valid] = np.nan

    volume = pd.to_numeric(base["volume"], errors="coerce")
    prior_volume = volume.shift(1)
    relative_volume = prior_volume / prior_volume.rolling(168, min_periods=72).median().replace(
        0.0, np.nan
    )
    candle_range = pd.to_numeric(base["high"], errors="coerce") - pd.to_numeric(
        base["low"], errors="coerce"
    )
    range_over_atr = candle_range.shift(1).rolling(4, min_periods=4).sum() / Series(
        atr, index=base.index
    ).replace(0.0, np.nan)
    absolute_return = (close.shift(1) - close.shift(5)).abs() / Series(
        atr, index=base.index
    ).replace(0.0, np.nan)
    absolute_pre_close = Series(pre_close, index=base.index).abs().replace(0.0, np.nan)
    atr_fraction = Series(atr, index=base.index) / absolute_pre_close

    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    local = DataFrame(
        {
            "date": dates,
            "nearest_level_distance_atr": nearest_distance,
            "independent_level_family_count_within_0p5atr": family_count,
            "source_role_fixed_encoding": nearest_role,
            "source_timeframe_fixed_encoding": nearest_timeframe,
            "pre_crossing_count_4h": pre_crossings,
            "relative_volume": relative_volume,
            "range_over_atr_4h": range_over_atr,
            "absolute_return_over_atr_4h": absolute_return,
            "atr_fraction": atr_fraction,
            "nearest_level_price": nearest_level,
            "zone_half_width": ZONE_HALF_WIDTH_ATR * atr,
        }
    )
    wider = context[
        [
            "date",
            "btc_absolute_return_over_atr_w4",
            "equal_weight_median_absolute_return_over_atr_w4",
            "cross_coin_return_dispersion_w4",
        ]
    ].rename(
        columns={
            "btc_absolute_return_over_atr_w4": "btc_absolute_activity_4h",
            "equal_weight_median_absolute_return_over_atr_w4": (
                "equal_weight_market_activity_4h"
            ),
            "cross_coin_return_dispersion_w4": "cross_coin_dispersion_4h",
        }
    )
    output = local.merge(wider, on="date", how="left", validate="one_to_one")
    output[f"ready__{READY_BLOCK}"] = output[list(FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    lag_choices = np.asarray(SHUFFLE_LAGS_HOURS, dtype=int)
    lags = np.asarray(
        [
            lag_choices[
                g0.stable_hash_int(f"g22-shuffle|{pair}|{date.isoformat()}")
                % len(lag_choices)
            ]
            for date in output["date"]
        ],
        dtype=int,
    )
    support = output[
        ["date", "nearest_level_price", "zone_half_width", f"ready__{READY_BLOCK}"]
    ].copy()
    support["shuffled_source_date"] = support["date"] - pd.to_timedelta(lags, unit="h")
    features = output[["date", *FEATURE_COLUMNS, f"ready__{READY_BLOCK}"]].copy()
    return features, support


def build_context(*, overwrite: bool = False) -> DataFrame:
    if CONTEXT_PATH.is_file() and not overwrite:
        return pd.read_parquet(CONTEXT_PATH)
    context = g22d.causal_cross_asset_context(g22d.aligned_pair_inputs())
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(context, CONTEXT_PATH)
    return context


def support_pair(
    pair: str, cohort: str, context: DataFrame, *, overwrite: bool = False
) -> dict[str, Any]:
    cohort_root = ARTIFACT_ROOT / cohort
    stem = g0.pair_file_stem(pair)
    feature_path = cohort_root / "feature_cache" / f"{stem}.parquet"
    support_path = cohort_root / "support_cache" / f"{stem}.parquet"
    if feature_path.is_file() and support_path.is_file() and not overwrite:
        feature = pd.read_parquet(feature_path, columns=[f"ready__{READY_BLOCK}"])
    else:
        base, _ = g20s.base_and_state(pair, cohort)
        feature, support = feature_surface(base, context, pair)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        support_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(feature, feature_path)
        g0.atomic_write_parquet(support, support_path)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(feature),
        "ready_rows": int(feature[f"ready__{READY_BLOCK}"].sum()),
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": g0.sha256_file(feature_path),
        "support_path": str(support_path.resolve()),
        "support_sha256": g0.sha256_file(support_path),
        "future_outcomes_read": False,
    }


def cohort_pairs(cohort: str) -> list[str]:
    return [pair for item_cohort, pair in g22a.cohort_pairs() if item_cohort == cohort]


def freeze_support(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    freeze_registry()
    path = support_manifest_path(cohort)
    if path.is_file() and not overwrite:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_outcome_blind_generation22_freqai_support":
            raise ValueError(f"Invalid Generation 22 FreqAI support freeze: {path}")
        return manifest
    context = build_context(overwrite=overwrite)
    inventory: list[dict[str, Any]] = []
    pairs = cohort_pairs(cohort)
    if len(pairs) != 10:
        raise ValueError(f"Expected ten {cohort} pairs, got {len(pairs)}.")
    for number, pair in enumerate(pairs, start=1):
        inventory.append(support_pair(pair, cohort, context, overwrite=overwrite))
        print(
            json.dumps(
                {
                    "phase": "g22_freqai_support",
                    "cohort": cohort,
                    "processed": number,
                    "total": len(pairs),
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_generation22_freqai_support",
        "branch_id": "g22e_freqai_unsigned_reaction_interaction_regression",
        "cohort": cohort,
        "pairs": pairs,
        "inventory": inventory,
        "feature_columns": list(FEATURE_COLUMNS),
        "future_outcome_values_read": False,
        "profile_registry": artifact(REGISTRY_PATH),
        "context_artifact": artifact(CONTEXT_PATH),
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "multitimeframe_builder": artifact(g22b.ANALYSIS_PATH),
            "cross_asset_builder": artifact(g22d.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, path)
    return manifest


def target_frame(base: DataFrame, support: DataFrame, cohort: str) -> DataFrame:
    if len(base) != len(support):
        raise ValueError("Generation 22 FreqAI base/support row mismatch.")
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    support_dates = pd.to_datetime(support["date"], utc=True, errors="raise")
    if not dates.equals(support_dates):
        raise ValueError("Generation 22 FreqAI base/support dates do not align.")
    paths = g0.future_path_matrices(base, max(HORIZONS))
    level = pd.to_numeric(support["nearest_level_price"], errors="coerce").to_numpy(float)
    width = pd.to_numeric(support["zone_half_width"], errors="coerce").to_numpy(float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(float)
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(float)
    pre_volume = pd.to_numeric(base["pre_volume_median_24"], errors="coerce").to_numpy(float)
    pre_range = pd.to_numeric(base["pre_range_median_24"], errors="coerce").to_numpy(float)
    output = DataFrame(
        {
            "date": dates,
            "period": g18d.assign_confirmation_period(dates, cohort),
            f"ready__{READY_BLOCK}": support[f"ready__{READY_BLOCK}"].astype(bool),
        }
    )
    level_matrix = level[:, None]
    atr_matrix = atr[:, None]
    high_distance = np.abs(paths["high"] - level_matrix) / atr_matrix
    low_distance = np.abs(paths["low"] - level_matrix) / atr_matrix
    absolute_path = np.maximum(high_distance, low_distance)
    previous_sign = np.sign(pre_close - level)
    for horizon in HORIZONS:
        sl = slice(0, horizon)
        high = paths["high"][:, sl]
        low = paths["low"][:, sl]
        close = paths["close"][:, sl]
        volume = paths["volume"][:, sl]
        complete = (
            np.isfinite(high).all(axis=1)
            & np.isfinite(low).all(axis=1)
            & np.isfinite(close).all(axis=1)
            & np.isfinite(volume).all(axis=1)
            & np.isfinite(level)
            & np.isfinite(width)
            & np.isfinite(atr)
            & (atr > 0.0)
        )
        finite_absolute = np.where(
            np.isfinite(absolute_path[:, sl]), absolute_path[:, sl], -np.inf
        )
        maximum = np.max(finite_absolute, axis=1)
        volume_ratio = g0.safe_ratio(np.mean(volume, axis=1), pre_volume)
        range_ratio = g0.safe_ratio(np.mean(high - low, axis=1), pre_range)
        crossings = g0.crossing_counts(close, level, previous_sign).astype(float)
        dwell = np.mean(np.abs(close - level_matrix) <= width[:, None], axis=1)
        for values in (maximum, volume_ratio, range_ratio, crossings, dwell):
            values[~complete] = np.nan
        output[f"&-g22_maximum_absolute_excursion_atr_h{horizon}"] = maximum
        output[f"&-g22_future_volume_ratio_h{horizon}"] = volume_ratio
        output[f"&-g22_future_range_ratio_h{horizon}"] = range_ratio
        output[f"&-g22_crossing_count_h{horizon}"] = crossings
        output[f"&-g22_dwell_fraction_h{horizon}"] = dwell
    return output


def shuffled_targets(actual: DataFrame, support: DataFrame) -> DataFrame:
    current = actual[["date", "period", f"ready__{READY_BLOCK}"]].copy()
    current["shuffled_source_date"] = pd.to_datetime(
        support["shuffled_source_date"], utc=True, errors="raise"
    )
    source_columns = {
        "date": "shuffled_source_date",
        **{target: f"source__{target}" for target in TARGETS},
    }
    source = actual[["date", *TARGETS]].rename(columns=source_columns)
    merged = current.merge(source, on="shuffled_source_date", how="left", validate="many_to_one")
    for target in TARGETS:
        merged[target] = pd.to_numeric(merged.pop(f"source__{target}"), errors="coerce")
    return merged[["date", "period", f"ready__{READY_BLOCK}", *TARGETS]]


def materialize_targets(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    support_freeze = freeze_support(cohort, overwrite=False)
    contracts = support_freeze["source_contracts"]
    for key, path in (
        ("generation22_freeze", g22z.FREEZE_PATH),
        ("analysis_script", ANALYSIS_PATH),
        ("multitimeframe_builder", g22b.ANALYSIS_PATH),
        ("cross_asset_builder", g22d.ANALYSIS_PATH),
    ):
        if g0.sha256_file(path) != contracts[key]["sha256"]:
            raise ValueError(f"Frozen Generation 22 FreqAI dependency changed: {key}")
    manifest_path = cache_manifest_path(cohort)
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_generation22_freqai_cache":
            raise ValueError(f"Invalid Generation 22 FreqAI cache: {manifest_path}")
        return existing
    completed: list[dict[str, Any]] = []
    for number, item in enumerate(support_freeze["inventory"], start=1):
        pair = str(item["pair"])
        base, _ = g20s.base_and_state(pair, cohort)
        support = pd.read_parquet(item["support_path"])
        actual = target_frame(base, support, cohort)
        shuffled = shuffled_targets(actual, support)
        root = ARTIFACT_ROOT / cohort
        stem = g0.pair_file_stem(pair)
        event_path = root / "event_cache" / f"{stem}.parquet"
        shuffled_path = root / "shuffled_event_cache" / f"{stem}.parquet"
        evaluation_path = root / "evaluation_cache" / f"{stem}.parquet"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        shuffled_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(actual, event_path)
        g0.atomic_write_parquet(shuffled, shuffled_path)
        g0.atomic_write_parquet(actual[["date", "period", *TARGETS]], evaluation_path)
        completed.append(
            {
                **item,
                "event_path": str(event_path.resolve()),
                "event_sha256": g0.sha256_file(event_path),
                "evaluation_path": str(evaluation_path.resolve()),
                "evaluation_sha256": g0.sha256_file(evaluation_path),
                "shuffled_event_path": str(shuffled_path.resolve()),
                "shuffled_event_sha256": g0.sha256_file(shuffled_path),
                "event_rows": len(actual),
                "targets_opened_after_support_freeze": True,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "g22_freqai_targets",
                    "cohort": cohort,
                    "processed": number,
                    "total": len(support_freeze["inventory"]),
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_freqai_cache",
        "cohort": cohort,
        "pairs": list(support_freeze["pairs"]),
        "inventory": completed,
        "profile_registry": artifact(REGISTRY_PATH),
        "outcome_blind_support_freeze": artifact(support_manifest_path(cohort)),
        "source_contracts": support_freeze["source_contracts"],
        "integrity": {
            "features_frozen_before_targets": True,
            "shuffled_source_dates_always_earlier": True,
            "targets_joined_after_support_freeze": True,
            "profit_used": False,
            "future_signed_direction_used": False,
        },
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "normal", "meme"), default="all")
    parser.add_argument("--phase", choices=("support", "targets", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    results: list[dict[str, Any]] = []
    if args.phase in {"support", "all"}:
        results.extend(freeze_support(cohort, overwrite=args.overwrite) for cohort in cohorts)
    if args.phase in {"targets", "all"}:
        results.extend(materialize_targets(cohort, overwrite=args.overwrite) for cohort in cohorts)
    print(
        json.dumps(
            [
                {
                    "status": item["status"],
                    "cohort": item["cohort"],
                    "pairs": len(item["pairs"]),
                }
                for item in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
