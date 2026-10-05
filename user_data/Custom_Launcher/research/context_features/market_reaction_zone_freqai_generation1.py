from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas/sklearn.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    common_prediction_keys,
    eligibility_digest,
    load_predictions,
    score_values,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    bool_array,
    level_cache_path,
    level_specs,
    load_manifest,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    resolved_level_values,
    sha256_file,
    utc_now,
    validate_cache_metadata,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    ACCEPTANCE_LEVELS,
    ACTIVITY_LEVELS,
    LOCAL_MATCH_FEATURES,
    SELECTED_LEVEL_COLUMNS,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    generation0_event_source,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration1Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
    TARGET_HORIZONS,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
DEFAULT_RECORD_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation1_freqai"
)
DEFAULT_ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation1_freqai"
)
FROZEN_BATCH = DEFAULT_MANIFEST.parent / "generation0_review" / "g1_frozen_branch_batch.json"
CLUSTER_CONTACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation1_cluster_increment"
    / "g1c_contact_preflight_full_20260813b_dependency"
    / "pair_contacts"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration1Strategy.py"
DATA_DIR = USER_DATA_DIR / "data" / "binance"

DEFAULT_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
)
SOURCE_TIMEFRAMES = ("1h", "4h", "8h", "1d")
ZONE_METHOD = "standard_base_atr"
EVENT_CONTROLS = ("actual", "matched_random_time")
EVENT_MATCH_FEATURES = tuple(
    column
    for column in LOCAL_MATCH_FEATURES
    if not column.startswith("state_source_") and column != "state_selected_level_density_2atr"
)
MAX_PROFILE_WORKERS = 8
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1

VP_COLUMNS = {
    "vp_lvn_above",
    "vp_lvn_below",
    "vp_hvn_above",
    "vp_hvn_below",
    "vp_poc",
    "vp_prior_poc",
}
VP_ATTRIBUTE_COLUMNS = (
    "vp_value_area_width_pct",
    "vp_hvn_above_strength",
    "vp_hvn_below_strength",
    "vp_lvn_above_thinness",
    "vp_lvn_below_thinness",
    "vp_score_abs",
)

PROFILES: dict[str, dict[str, Any]] = {
    "local_state": {
        "strategy": "MarketReactionZoneG1DLocalStateFreqAIResearchStrategy",
        "role": "Pre-contact local OHLCV and fixed-indicator state only.",
        "baseline": None,
        "placebo": None,
    },
    "crypto_wide_state": {
        "strategy": "MarketReactionZoneG1DCryptoWideStateFreqAIResearchStrategy",
        "role": "Local state plus causal BTC and top-ten crypto-wide state.",
        "baseline": "local_state",
        "placebo": None,
    },
    "native_current": {
        "strategy": "MarketReactionZoneG1DNativeCurrentFreqAIResearchStrategy",
        "role": (
            "Crypto-wide state plus current multi-timeframe level identity, distance, "
            "role, and score fields."
        ),
        "baseline": "crypto_wide_state",
        "placebo": "native_placebo",
    },
    "native_placebo": {
        "strategy": "MarketReactionZoneG1DNativePlaceboFreqAIResearchStrategy",
        "role": "The identical native level block delayed by 168 hours.",
        "baseline": "crypto_wide_state",
        "placebo": None,
    },
    "vp_current": {
        "strategy": "MarketReactionZoneG1DVolumeProfileCurrentFreqAIResearchStrategy",
        "role": (
            "Crypto-wide state plus current VP distances, evidence, thinness, strength, "
            "and value-area width."
        ),
        "baseline": "crypto_wide_state",
        "placebo": "vp_placebo",
    },
    "vp_placebo": {
        "strategy": "MarketReactionZoneG1DVolumeProfilePlaceboFreqAIResearchStrategy",
        "role": "The identical Volume Profile block delayed by 168 hours.",
        "baseline": "crypto_wide_state",
        "placebo": None,
    },
    "cluster_current": {
        "strategy": "MarketReactionZoneG1DClusterCurrentFreqAIResearchStrategy",
        "role": (
            "Crypto-wide state plus current independent cluster composition and density fields."
        ),
        "baseline": "crypto_wide_state",
        "placebo": "cluster_placebo",
    },
    "cluster_placebo": {
        "strategy": "MarketReactionZoneG1DClusterPlaceboFreqAIResearchStrategy",
        "role": "The identical independent-cluster block delayed by 168 hours.",
        "baseline": "crypto_wide_state",
        "placebo": None,
    },
}


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def target_metadata(target: str) -> tuple[str, int]:
    raw = target.removeprefix("&-event_")
    for horizon in TARGET_HORIZONS:
        suffix = f"_{horizon}h"
        if raw.endswith(suffix):
            return raw.removesuffix(suffix), horizon
    raise ValueError(f"Unknown Generation 1D target: {target}")


def validate_frozen_branch() -> dict[str, Any]:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branch = next(
        (
            item
            for item in frozen["branches"]
            if item["id"] == "g1d_contact_focused_freqai_residual_ladder"
        ),
        None,
    )
    if frozen.get("status") != "frozen_ready_for_execution" or branch is None:
        raise ValueError("The frozen Generation 1D branch is unavailable.")
    if frozen["common_scope"].get("direction_prediction") is not False:
        raise ValueError("Generation 1D must keep direction prediction disabled.")
    if frozen["common_scope"].get("profit_optimization") is not False:
        raise ValueError("Generation 1D must keep profit optimization disabled.")
    return branch


def selected_specs(cache: DataFrame) -> list[LevelSpec]:
    selected: dict[str, LevelSpec] = {}
    for spec in level_specs(cache, ("g0b1", "g0b2")):
        if spec.column not in SELECTED_LEVEL_COLUMNS:
            continue
        previous = selected.get(spec.column)
        if previous is not None and previous.representation != spec.representation:
            raise ValueError(f"Ambiguous selected representation for {spec.column}")
        selected[spec.column] = spec
    return list(selected.values())


def causal_level_feature_blocks(  # noqa: C901 - explicit causal feature families are auditable
    *, pair: str, base: DataFrame, manifest_path: Path
) -> tuple[DataFrame, list[dict[str, Any]]]:
    features: dict[str, Any] = {"date": normalize_dates(base["date"])}
    pre_close = numeric_array(base["pre_close"])
    base_atr = numeric_array(base["base_atr"])
    contracts: list[dict[str, Any]] = []
    native_role_distances: dict[tuple[str, str], list[np.ndarray]] = {}
    for timeframe in SOURCE_TIMEFRAMES:
        cache_path = level_cache_path(pair, timeframe, ("core", "generic"))
        metadata = validate_cache_metadata(cache_path, manifest_path)
        cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        cache["available_at"] = normalize_dates(cache["available_at"])
        cache["source_open"] = normalize_dates(cache["source_open"])
        aligned = pd.merge_asof(
            base[["date"]].sort_values("date"),
            cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        causal = aligned["available_at"].notna()
        if (aligned.loc[causal, "available_at"] > aligned.loc[causal, "date"]).any():
            raise AssertionError(f"Future level cache admitted for {pair} {timeframe}")
        source_age = (
            (normalize_dates(aligned["date"]) - normalize_dates(aligned["available_at"]))
            .dt.total_seconds()
            .div(3600.0)
        )
        features[f"native__{timeframe}_source_age_hours"] = source_age.clip(0.0, 240.0)
        features[f"vp__{timeframe}_source_age_hours"] = source_age.clip(0.0, 240.0)
        for spec in selected_specs(cache):
            level = resolved_level_values(aligned, spec, timeframe)
            valid = np.isfinite(level) & (level > 0.0) & np.isfinite(base_atr) & (base_atr > 0.0)
            if spec.active_columns:
                active = np.zeros(len(aligned), dtype=bool)
                for column in spec.active_columns:
                    if column in aligned:
                        active |= bool_array(aligned[column])
                valid &= active
            signed = np.full(len(base), np.nan, dtype=np.float64)
            signed[valid] = (level[valid] - pre_close[valid]) / base_atr[valid]
            absolute = np.abs(signed)
            key = f"{timeframe}_{spec.column}"
            features[f"native__{key}_signed_distance_atr"] = np.clip(signed, -25.0, 25.0)
            features[f"native__{key}_absolute_distance_atr"] = np.clip(absolute, 0.0, 25.0)
            if spec.score_column and spec.score_column in aligned:
                score = pd.to_numeric(aligned[spec.score_column], errors="coerce")
                features[f"native__{key}_score"] = score.clip(-100.0, 100.0)
            role = (
                "activity"
                if spec.name in ACTIVITY_LEVELS
                else "acceptance"
                if spec.name in ACCEPTANCE_LEVELS
                else "other"
            )
            native_role_distances.setdefault((timeframe, role), []).append(absolute)
            if spec.column in VP_COLUMNS:
                features[f"vp__{key}_absolute_distance_atr"] = np.clip(absolute, 0.0, 25.0)
        for column in VP_ATTRIBUTE_COLUMNS:
            if column in aligned:
                features[f"vp__{timeframe}_{column}"] = pd.to_numeric(
                    aligned[column], errors="coerce"
                ).clip(-100.0, 100.0)
        contracts.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "cache": str(cache_path.resolve()),
                "cache_sha256": sha256_file(cache_path),
                "source_sha256": metadata["source_sha256"],
            }
        )
    for (timeframe, role), values in native_role_distances.items():
        matrix = np.column_stack(values)
        features[f"native__{timeframe}_{role}_nearest_distance_atr"] = np.nanmin(matrix, axis=1)
        for threshold, label in ((0.10, "10"), (0.25, "25"), (0.50, "50")):
            features[f"native__{timeframe}_{role}_count_within_{label}pct_atr"] = np.sum(
                np.isfinite(matrix) & (matrix <= threshold), axis=1
            )
    return DataFrame(features), contracts


def causal_cluster_feature_block(pair: str, base: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    path = CLUSTER_CONTACT_ROOT / f"{pair_stem(pair)}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = [
        "event_time",
        "cluster_scale",
        "representation_mode",
        "contact_structure_class",
        "contacted_component_count",
        "contacted_family_count",
        "contacted_timeframe_count",
        "contacted_dependency_group_count",
        "contacted_distinct_source_count",
        "contacted_independent_dependency_groups",
        "cluster_width_atr",
        "component_dispersion_atr",
        "local_level_density_2atr",
        "contacted_component_signature",
        "future_path_key",
    ]
    events = pd.read_parquet(path, columns=columns)
    events["event_time"] = normalize_dates(events["event_time"])
    events = events.loc[events["contacted_independent_dependency_groups"].fillna(False)].copy()
    output = DataFrame({"date": normalize_dates(base["date"])})
    if events.empty:
        for column in cluster_feature_columns():
            output[column] = 0.0
        return output, {"path": str(path), "sha256": sha256_file(path), "rows": 0}
    events["cross_timeframe"] = events["contacted_timeframe_count"].gt(1).astype(float)
    events["tight"] = events["cluster_scale"].eq("tight").astype(float)
    events["standard"] = events["cluster_scale"].eq("standard").astype(float)
    events["projected"] = events["representation_mode"].eq("projected").astype(float)
    events["different_family_cross_timeframe"] = (
        events["contact_structure_class"].eq("different_family_cross_timeframe").astype(float)
    )
    grouped = events.groupby("event_time", sort=False, observed=True)
    aggregate = (
        grouped.agg(
            cluster__present=("future_path_key", "size"),
            cluster__unique_future_paths=("future_path_key", "nunique"),
            cluster__exact_pattern_count=("contacted_component_signature", "nunique"),
            cluster__max_contacted_components=("contacted_component_count", "max"),
            cluster__max_family_count=("contacted_family_count", "max"),
            cluster__max_timeframe_count=("contacted_timeframe_count", "max"),
            cluster__max_dependency_group_count=("contacted_dependency_group_count", "max"),
            cluster__max_distinct_source_count=("contacted_distinct_source_count", "max"),
            cluster__minimum_width_atr=("cluster_width_atr", "min"),
            cluster__median_component_dispersion_atr=("component_dispersion_atr", "median"),
            cluster__max_local_level_density_2atr=("local_level_density_2atr", "max"),
            cluster__cross_timeframe_present=("cross_timeframe", "max"),
            cluster__tight_present=("tight", "max"),
            cluster__standard_present=("standard", "max"),
            cluster__projected_present=("projected", "max"),
            cluster__different_family_cross_timeframe_present=(
                "different_family_cross_timeframe",
                "max",
            ),
        )
        .reset_index()
        .rename(columns={"event_time": "date"})
    )
    output = output.merge(aggregate, on="date", how="left", validate="one_to_one")
    for column in cluster_feature_columns():
        if column not in output:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["cluster__present"] = output["cluster__present"].gt(0).astype(float)
    return output, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(events),
    }


def cluster_feature_columns() -> tuple[str, ...]:
    return (
        "cluster__present",
        "cluster__unique_future_paths",
        "cluster__exact_pattern_count",
        "cluster__max_contacted_components",
        "cluster__max_family_count",
        "cluster__max_timeframe_count",
        "cluster__max_dependency_group_count",
        "cluster__max_distinct_source_count",
        "cluster__minimum_width_atr",
        "cluster__median_component_dispersion_atr",
        "cluster__max_local_level_density_2atr",
        "cluster__cross_timeframe_present",
        "cluster__tight_present",
        "cluster__standard_present",
        "cluster__projected_present",
        "cluster__different_family_cross_timeframe_present",
    )


def canonical_event_candidates(
    *, pair: str, base: DataFrame
) -> tuple[DataFrame, list[dict[str, Any]]]:
    columns = [
        "pair",
        "source_timeframe",
        "level_family",
        "level_name",
        "control",
        "base_index",
        "event_time",
        "period",
        "approach_state",
        "pre_distance_atr",
        "random_match_tier",
        *event_outcome_columns(),
    ]
    frames: list[DataFrame] = []
    contracts: list[dict[str, Any]] = []
    for timeframe in SOURCE_TIMEFRAMES:
        path, metadata_path = generation0_event_source(pair, timeframe)
        available = set(pq.ParquetFile(path).schema.names)
        missing = sorted(set(columns).difference(available))
        if missing:
            raise ValueError(f"Generation 0 event source lacks G1D columns: {path}: {missing}")
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("control", "in", list(EVENT_CONTROLS)),
                ("zone_method", "==", ZONE_METHOD),
            ],
        )
        frame = frame.loc[frame["level_name"].isin(ACTIVITY_LEVELS | ACCEPTANCE_LEVELS)]
        frames.append(frame)
        contracts.append(
            {
                "pair": pair,
                "timeframe": timeframe,
                "event_path": str(path.resolve()),
                "event_sha256": sha256_file(path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    events = pd.concat(frames, ignore_index=True)
    events["event_time"] = normalize_dates(events["event_time"])
    events["source_timeframe_order"] = events["source_timeframe"].map(
        {timeframe: index for index, timeframe in enumerate(SOURCE_TIMEFRAMES)}
    )
    events.sort_values(
        [
            "control",
            "event_time",
            "pre_distance_atr",
            "source_timeframe_order",
            "level_name",
        ],
        inplace=True,
        kind="stable",
    )
    events = events.drop_duplicates(["control", "event_time"], keep="first")
    actual_dates = set(events.loc[events["control"].eq("actual"), "event_time"])
    events = events.loc[
        events["control"].eq("actual") | ~events["event_time"].isin(actual_dates)
    ].copy()
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    if indexes.min(initial=0) < 0 or indexes.max(initial=-1) >= len(base):
        raise ValueError(f"Generation 0 event base indexes are incompatible for {pair}")
    expected_dates = normalize_dates(base["date"]).iloc[indexes].reset_index(drop=True)
    actual_event_dates = events["event_time"].reset_index(drop=True)
    if not expected_dates.equals(actual_event_dates):
        raise ValueError(f"Generation 0 event dates do not match base indexes for {pair}")
    return events.drop(columns=["source_timeframe_order"]), contracts


def event_outcome_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for horizon in TARGET_HORIZONS:
        columns.extend(
            (
                f"abs_excursion_atr_h{horizon}",
                f"range_ratio_h{horizon}",
                f"volume_ratio_h{horizon}",
                f"pressure_change_h{horizon}",
                f"dwell_fraction_h{horizon}",
                f"crossings_h{horizon}",
            )
        )
    return tuple(columns)


def attach_event_match_state(
    events: DataFrame, *, base: DataFrame, market_context: DataFrame
) -> DataFrame:
    state = causal_local_state(base).merge(
        market_context,
        on="date",
        how="left",
        validate="one_to_one",
    )
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    for column in EVENT_MATCH_FEATURES:
        output[column] = numeric_array(state[column])[indexes]
    return output


def matched_event_pairs(events: DataFrame) -> list[tuple[int, int, float]]:
    actual = events.loc[events["control"].eq("actual")]
    control = events.loc[events["control"].eq("matched_random_time")]
    candidates: list[tuple[int, int, float]] = []
    common_periods = sorted(set(actual["period"]).intersection(control["period"]))
    for period in common_periods:
        for approach in (
            "from_below",
            "from_above",
            "already_inside_or_unclear",
        ):
            left = actual.loc[actual["period"].eq(period) & actual["approach_state"].eq(approach)]
            right = control.loc[
                control["period"].eq(period) & control["approach_state"].eq(approach)
            ]
            pairs, _ = nearest_state_pairs(
                left,
                right,
                state_columns=EVENT_MATCH_FEATURES,
                pre_distance_atr_caliper=0.10,
                minimum_event_separation_hours=MAX_TARGET_HORIZON_HOURS,
            )
            candidates.extend(
                (int(left.index[a]), int(right.index[b]), float(distance))
                for a, b, distance in pairs
            )
    return select_non_overlapping_event_pairs(events, candidates)


def select_non_overlapping_event_pairs(
    events: DataFrame,
    candidates: Sequence[tuple[int, int, float]],
    *,
    minimum_gap_hours: int = MAX_TARGET_HORIZON_HOURS,
) -> list[tuple[int, int, float]]:
    if minimum_gap_hours < 0:
        raise ValueError("minimum_gap_hours cannot be negative")
    gap_ns = int(pd.Timedelta(hours=minimum_gap_hours).value)
    times = normalize_dates(events["event_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    index_to_position = {index: position for position, index in enumerate(events.index)}
    selected: list[tuple[int, int, float]] = []
    selected_times: list[int] = []
    used_actual: set[int] = set()
    used_control: set[int] = set()
    for actual_index, control_index, distance in sorted(candidates, key=lambda row: row[2]):
        if actual_index in used_actual or control_index in used_control:
            continue
        actual_time = int(times[index_to_position[actual_index]])
        control_time = int(times[index_to_position[control_index]])
        if abs(actual_time - control_time) <= gap_ns:
            continue
        if selected_times and (
            np.min(np.abs(np.asarray(selected_times, dtype=np.int64) - actual_time)) <= gap_ns
            or np.min(np.abs(np.asarray(selected_times, dtype=np.int64) - control_time)) <= gap_ns
        ):
            continue
        selected.append((actual_index, control_index, distance))
        used_actual.add(actual_index)
        used_control.add(control_index)
        selected_times.extend((actual_time, control_time))
    return selected


def event_targets(frame: DataFrame) -> DataFrame:
    output = DataFrame(index=frame.index)
    for horizon in TARGET_HORIZONS:
        output[f"&-event_absolute_excursion_atr_{horizon}h"] = pd.to_numeric(
            frame[f"abs_excursion_atr_h{horizon}"], errors="coerce"
        ).clip(0.0, 30.0)
        output[f"&-event_range_ratio_{horizon}h"] = pd.to_numeric(
            frame[f"range_ratio_h{horizon}"], errors="coerce"
        ).clip(0.0, 30.0)
        output[f"&-event_volume_ratio_{horizon}h"] = pd.to_numeric(
            frame[f"volume_ratio_h{horizon}"], errors="coerce"
        ).clip(0.0, 30.0)
        output[f"&-event_pressure_change_magnitude_{horizon}h"] = (
            pd.to_numeric(frame[f"pressure_change_h{horizon}"], errors="coerce")
            .abs()
            .clip(0.0, 2.0)
        )
        output[f"&-event_dwell_fraction_{horizon}h"] = pd.to_numeric(
            frame[f"dwell_fraction_h{horizon}"], errors="coerce"
        ).clip(0.0, 1.0)
        output[f"&-event_crossing_rate_{horizon}h"] = (
            pd.to_numeric(frame[f"crossings_h{horizon}"], errors="coerce")
            .div(float(horizon))
            .clip(0.0, 1.0)
        )
    missing = sorted(set(TARGET_COLUMNS).difference(output.columns))
    if missing:
        raise AssertionError(f"G1D target mapping is incomplete: {missing}")
    return output[list(TARGET_COLUMNS)]


def build_event_cache(
    *,
    pair: str,
    base: DataFrame,
    manifest: dict[str, Any],
    market_context: DataFrame,
    feature_cache: DataFrame,
) -> tuple[DataFrame, dict[str, Any]]:
    candidates, contracts = canonical_event_candidates(pair=pair, base=base)
    _ = manifest
    candidates = attach_event_match_state(
        candidates,
        base=base,
        market_context=market_context,
    )
    pairs = matched_event_pairs(candidates)
    if len(pairs) < 40:
        raise ValueError(f"Too few purged actual/control event pairs for {pair}: {len(pairs)}")
    rows: list[DataFrame] = []
    match_rows: list[dict[str, Any]] = []
    for match_number, (actual_index, control_index, distance) in enumerate(pairs):
        for scope, index in (
            ("actual_contact", actual_index),
            ("matched_noncontact", control_index),
        ):
            selected = candidates.loc[[index]].copy()
            selected["event_scope"] = scope
            selected["event_match_id"] = f"{pair_stem(pair)}-{match_number:05d}"
            rows.append(selected)
        match_rows.append(
            {
                "event_match_id": f"{pair_stem(pair)}-{match_number:05d}",
                "state_distance": distance,
            }
        )
    events = pd.concat(rows, ignore_index=True)
    targets = event_targets(events)
    output = events[
        [
            "event_time",
            "event_scope",
            "event_match_id",
            "period",
            "approach_state",
            "source_timeframe",
            "level_family",
            "level_name",
            "pre_distance_atr",
        ]
    ].rename(columns={"event_time": "date"})
    output = pd.concat([output.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    feature_scope = feature_cache[
        [
            "date",
            "cluster__present",
            "native__1h_activity_count_within_25pct_atr",
            "native__1h_acceptance_count_within_25pct_atr",
        ]
    ]
    output = output.merge(feature_scope, on="date", how="left", validate="many_to_one")
    cluster_context = output["cluster__present"].fillna(0.0).gt(0.0)
    output["event_scope_detail"] = np.where(
        output["event_scope"].eq("matched_noncontact"),
        "matched_noncontact",
        np.where(cluster_context, "actual_cluster_context", "actual_isolated_context"),
    )
    output.drop(
        columns=[
            "cluster__present",
            "native__1h_activity_count_within_25pct_atr",
            "native__1h_acceptance_count_within_25pct_atr",
        ],
        inplace=True,
    )
    output.dropna(subset=list(TARGET_COLUMNS), inplace=True)
    output.sort_values("date", inplace=True, ignore_index=True)
    if output["date"].duplicated().any():
        raise AssertionError(f"G1D event cache has duplicate dates for {pair}")
    event_ns = output["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    if len(event_ns) > 1:
        minimum_gap = np.min(np.diff(np.sort(event_ns))) / 3.6e12
        if minimum_gap <= MAX_TARGET_HORIZON_HOURS:
            raise AssertionError(f"G1D future paths overlap for {pair}: {minimum_gap}h")
    counts = output["event_scope"].value_counts().to_dict()
    if counts.get("actual_contact") != counts.get("matched_noncontact"):
        raise AssertionError(f"G1D event scopes are unbalanced for {pair}: {counts}")
    return output, {
        "event_sources": contracts,
        "candidate_rows": len(candidates),
        "matched_pairs_before_target_filter": len(pairs),
        "event_rows": len(output),
        "scope_counts": {str(key): int(value) for key, value in counts.items()},
        "match_state_distance_median": float(
            np.median([row["state_distance"] for row in match_rows])
        ),
        "minimum_event_gap_hours": float(
            np.min(np.diff(np.sort(event_ns))) / 3.6e12 if len(event_ns) > 1 else np.nan
        ),
    }


def build_pair_caches(
    *,
    pair: str,
    manifest_path: Path,
    feature_dir: Path,
    event_dir: Path,
    market_context: DataFrame | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(pair, manifest)
    level_features, level_contracts = causal_level_feature_blocks(
        pair=pair,
        base=base,
        manifest_path=manifest_path,
    )
    raw_market = (
        causal_market_context(manifest) if market_context is None else market_context.copy()
    )
    market = raw_market.copy()
    market = market.rename(
        columns={
            column: f"wide__{column.removeprefix('state_')}"
            for column in market.columns
            if column.startswith("state_")
        }
    )
    features = level_features.merge(market, on="date", how="left", validate="one_to_one")
    cluster, cluster_contract = causal_cluster_feature_block(pair, base)
    features = features.merge(cluster, on="date", how="left", validate="one_to_one")
    feature_columns = [column for column in features if column != "date"]
    for column in feature_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        if "distance_atr" in column:
            values = values.fillna(25.0)
        elif column.endswith("source_age_hours"):
            values = values.fillna(240.0)
        else:
            values = values.fillna(0.0)
        features[column] = values.astype("float32")
    feature_path = feature_dir / f"{pair_stem(pair)}.parquet"
    event_path = event_dir / f"{pair_stem(pair)}.parquet"
    atomic_write_parquet(features, feature_path)
    events, event_audit = build_event_cache(
        pair=pair,
        base=base,
        manifest=manifest,
        market_context=raw_market,
        feature_cache=features,
    )
    atomic_write_parquet(events, event_path)
    return {
        "pair": pair,
        "feature_rows": len(features),
        "feature_columns": len(feature_columns),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "event_path": str(event_path),
        "event_sha256": sha256_file(event_path),
        "event_audit": event_audit,
        "level_cache_contracts": level_contracts,
        "cluster_contract": cluster_contract,
    }


def prepare_caches(
    *,
    pairs: Sequence[str],
    manifest_path: Path,
    feature_dir: Path,
    event_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    if workers < 1 or workers > MAX_PROFILE_WORKERS:
        raise ValueError(f"Cache workers must be between 1 and {MAX_PROFILE_WORKERS}")
    feature_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    market_context = causal_market_context(manifest)
    kwargs = [
        {
            "pair": pair,
            "manifest_path": manifest_path,
            "feature_dir": feature_dir,
            "event_dir": event_dir,
            "market_context": market_context,
        }
        for pair in pairs
    ]
    if workers == 1:
        return [build_pair_caches(**item) for item in kwargs]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_pair_caches, **item): item["pair"] for item in kwargs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g1d_cache_build",
                        "pair": result["pair"],
                        "event_rows": result["event_audit"]["event_rows"],
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda row: pairs.index(row["pair"]))


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["exchange"]["pair_whitelist"] = list(pairs)
    freqai = config["freqai"]
    freqai["enabled"] = True
    freqai["identifier"] = identifier
    freqai["train_period_days"] = int(train_days)
    freqai["backtest_period_days"] = int(backtest_days)
    freqai["save_backtest_models"] = False
    freqai["multitarget_parallel_training"] = False
    freqai["feature_parameters"]["plot_feature_importances"] = 0
    freqai["feature_parameters"]["include_corr_pairlist"] = []
    freqai["feature_parameters"]["include_timeframes"] = ["1h"]
    freqai["feature_parameters"]["include_shifted_candles"] = 0
    freqai["data_split_parameters"] = {
        "test_size": 0,
        "shuffle": False,
    }
    training = freqai["model_training_parameters"]
    training["n_jobs"] = 1
    if technical_smoke:
        training["n_estimators"] = min(20, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_g1d"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
    }
    return config


def build_manifest(
    *,
    run_id: str,
    record_dir: Path,
    artifact_dir: Path,
    base_config: Path,
    python_exe: Path,
    profiles: Sequence[str],
    pairs: Sequence[str],
    timerange: str,
    train_days: int,
    backtest_days: int,
    profile_workers: int,
    technical_smoke: bool,
    cache_inventory: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    commands: list[dict[str, Any]] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        profile_dir = artifact_dir / "profiles" / profile_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"{run_id}_{profile_id}"
        config_path = record_dir / f"config_{profile_id}.json"
        atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                train_days=train_days,
                backtest_days=backtest_days,
                technical_smoke=technical_smoke,
            ),
            config_path,
        )
        command = [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(userdir),
            "--strategy-path",
            str(STRATEGY_PATH),
            "--datadir",
            str(DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            str(definition["strategy"]),
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            timerange,
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
            "--cache",
            "none",
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "role": definition["role"],
                "baseline": definition["baseline"],
                "placebo": definition["placebo"],
                "identifier": identifier,
                "config_path": str(config_path),
                "artifact_dir": str(profile_dir),
                "user_data_dir": str(userdir),
                "model_dir": str(userdir / "models" / identifier),
                "export_dir": str(export_dir),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "frozen_branch": "g1d_contact_focused_freqai_residual_ladder",
        "objective": (
            "Use FreqAI on purged actual-contact and matched-noncontact episodes to test "
            "whether current single-level, Volume Profile, or independent-cluster fields "
            "improve unseen non-directional reaction estimates beyond pre-contact market state."
        ),
        "residual_interpretation": (
            "Incremental residual information is measured as lower unseen error or better "
            "ranking than the identical market-state baseline on the same prediction rows. "
            "The raw label is not replaced by a separately learned residual label."
        ),
        "hypothesis": (
            "A current trader-readable level block should improve unseen reaction estimates "
            "over both crypto-wide market state and its identically shaped 168-hour delayed "
            "placebo specifically at actual contacts more than matched noncontacts."
        ),
        "pass_interpretation": (
            "Retain only blocks that improve calibration and rank beyond baseline and placebo "
            "at actual contacts in repeated chronological periods and an honest coin or cohort "
            "scope. A lead is not a trading rule or promotion."
        ),
        "park_interpretation": (
            "Park blocks that merely estimate general activity, match their placebo, harm "
            "calibration, or depend on one pair, period, or horizon."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
        "timerange": timerange,
        "train_period_days": int(train_days),
        "backtest_period_days": int(backtest_days),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "pairs": list(pairs),
        "profiles": list(profiles),
        "targets": list(TARGET_COLUMNS),
        "target_horizons_hours": list(TARGET_HORIZONS),
        "event_surface": {
            "zone_method": ZONE_METHOD,
            "actual_scope": "canonical nearest selected level contact",
            "control_scope": (
                "canonical matched random pseudo-level contact with every selected "
                "actual-contact date excluded"
            ),
            "matching_features": list(EVENT_MATCH_FEATURES),
            "pre_distance_atr_difference_caliper": 0.10,
            "maximum_control_reuse": 1,
            "minimum_gap_between_every_retained_event_hours": MAX_TARGET_HORIZON_HOURS,
        },
        "leakage_controls": {
            "contact_candle_used_as_feature": False,
            "future_outcomes_used_for_event_matching": False,
            "internal_train_test_split": False,
            "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
            "embargo_method": (
                "set_freqai_targets blanks every event label inside the maximum target horizon "
                "of each FreqAI training-window end; test_size=0 removes an unpurged "
                "internal split."
            ),
            "same_event_rows_for_every_profile": True,
            "same_model_class_and_parameters": True,
        },
        "source_contracts": {
            "generation0_manifest": str(DEFAULT_MANIFEST.resolve()),
            "generation0_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
            "frozen_batch": str(FROZEN_BATCH.resolve()),
            "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "strategy_sha256": sha256_file(STRATEGY_FILE),
            "cache_inventory": list(cache_inventory),
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
            "cleanup": (
                "Technical smoke and failed temporary artifacts may be removed after a verified "
                "full replacement; retain compact records and the verified full evidence."
            ),
        },
        "commands": commands,
    }


def validate_manifest_request(
    manifest: dict[str, Any],
    *,
    profiles: Sequence[str],
    pairs: Sequence[str],
    timerange: str,
    train_days: int,
    backtest_days: int,
    profile_workers: int,
    technical_smoke: bool,
    python_exe: Path,
) -> None:
    checks = (
        ("profiles", tuple(manifest["profiles"]), tuple(profiles)),
        ("pairs", tuple(manifest["pairs"]), tuple(pairs)),
        ("timerange", str(manifest["timerange"]), timerange),
        ("train_period_days", int(manifest["train_period_days"]), int(train_days)),
        ("backtest_period_days", int(manifest["backtest_period_days"]), int(backtest_days)),
        ("profile_workers", int(manifest["profile_workers"]), int(profile_workers)),
        (
            "technical_smoke_not_evidence",
            bool(manifest["technical_smoke_not_evidence"]),
            bool(technical_smoke),
        ),
        (
            "worker_interpreter",
            {str(Path(item["command"][0]).resolve()) for item in manifest["commands"]},
            {str(python_exe.resolve())},
        ),
    )
    mismatches = [
        f"{name}: recorded={actual!r}, requested={expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            f"Run id {manifest['run_id']!r} has incompatible settings: " + "; ".join(mismatches)
        )


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "load_note": "Inspect the exact live process snapshot before increasing profile workers.",
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot.update(
            {
                "cpu_percent_sample": psutil.cpu_percent(interval=0.25),
                "memory_available_gib": round(memory.available / (1024**3), 3),
                "visible_freqtrade_processes": [],
            }
        )
        for process in psutil.process_iter(["pid", "name", "cmdline", "num_threads"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                if "freqtrade" in command.lower() or "hyperopt" in command.lower():
                    snapshot["visible_freqtrade_processes"].append(
                        {
                            "pid": process.info["pid"],
                            "name": process.info.get("name"),
                            "num_threads": process.info.get("num_threads"),
                            "command": command[:1000],
                        }
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
                continue
    except ImportError:
        snapshot["load_note"] = "psutil unavailable; inspect live load outside this preflight."
    return snapshot


def preflight(  # noqa: C901 - one explicit audit keeps every launch gate in one record
    manifest: dict[str, Any], python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    cache_audit: list[dict[str, Any]] = []
    sources = manifest["source_contracts"]
    for label, path, expected in (
        ("analysis script", Path(__file__), sources["analysis_script_sha256"]),
        ("research strategy", STRATEGY_FILE, sources["strategy_sha256"]),
        (
            "Generation 0 manifest",
            Path(sources["generation0_manifest"]),
            sources["generation0_manifest_sha256"],
        ),
        (
            "frozen branch batch",
            Path(sources["frozen_batch"]),
            sources["frozen_batch_sha256"],
        ),
    ):
        if not path.is_file():
            problems.append(f"missing {label}: {path}")
        elif sha256_file(path) != expected:
            problems.append(f"changed {label}: {path}")
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")
    expected_feature_dir = Path(manifest["storage"]["feature_cache_dir"])
    expected_event_dir = Path(manifest["storage"]["event_cache_dir"])
    inventory = {row["pair"]: row for row in manifest["source_contracts"]["cache_inventory"]}
    for pair in manifest["pairs"]:
        feature_path = expected_feature_dir / f"{pair_stem(pair)}.parquet"
        event_path = expected_event_dir / f"{pair_stem(pair)}.parquet"
        recorded = inventory.get(pair)
        if recorded is None:
            problems.append(f"cache inventory missing pair: {pair}")
            continue
        for label, path, digest_key in (
            ("feature", feature_path, "feature_sha256"),
            ("event", event_path, "event_sha256"),
        ):
            if not path.is_file():
                problems.append(f"missing {label} cache: {path}")
                continue
            actual_sha = sha256_file(path)
            if actual_sha != recorded[digest_key]:
                problems.append(f"changed {label} cache: {path}")
        if event_path.is_file():
            events = pd.read_parquet(
                event_path,
                columns=["date", "event_scope", *TARGET_COLUMNS],
            )
            counts = events["event_scope"].value_counts().to_dict()
            dates = normalize_dates(events["date"])
            ordered = np.sort(dates.to_numpy(dtype="datetime64[ns]").astype("int64"))
            minimum_gap = float(np.min(np.diff(ordered)) / 3.6e12) if len(ordered) > 1 else np.nan
            if counts.get("actual_contact") != counts.get("matched_noncontact"):
                problems.append(f"unbalanced event scopes for {pair}: {counts}")
            if np.isfinite(minimum_gap) and minimum_gap <= MAX_TARGET_HORIZON_HOURS:
                problems.append(f"overlapping event future paths for {pair}: {minimum_gap}h")
            if events[list(TARGET_COLUMNS)].isna().any().any():
                problems.append(f"event cache contains missing target values: {event_path}")
            cache_audit.append(
                {
                    "pair": pair,
                    "event_rows": len(events),
                    "scope_counts": {str(key): int(value) for key, value in counts.items()},
                    "minimum_event_gap_hours": minimum_gap,
                    "feature_columns": len(pq.ParquetFile(feature_path).schema.names) - 1
                    if feature_path.is_file()
                    else 0,
                }
            )
    dependency = None
    if python_exe.is_file():
        result = subprocess.run(
            [str(python_exe), "-c", "import lightgbm, freqtrade; print('ok')"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dependency = {
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        if result.returncode != 0:
            problems.append("worker interpreter cannot import lightgbm and freqtrade")
    artifact_root = Path(manifest["storage"]["bulky_artifact_dir"])
    free_gib = round(shutil.disk_usage(artifact_root.anchor).free / (1024**3), 3)
    return {
        "created_at_utc": utc_now(),
        "profiles": len(manifest["profiles"]),
        "pairs": len(manifest["pairs"]),
        "targets": len(TARGET_COLUMNS),
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "worker_interpreter": str(python_exe.resolve()),
        "dependency_check": dependency,
        "cache_audit": cache_audit,
        "runtime_snapshot": runtime_snapshot(),
        "bulky_storage_free_gib": free_gib,
        "internal_test_size": 0,
        "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
        "problems": problems,
        "passed": not problems,
    }


def run_profile(item: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(item["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    Path(item["user_data_dir"]).mkdir(parents=True, exist_ok=True)
    attempt = int(item.get("attempts", 0)) + 1
    stdout_path = artifact_dir / f"freqai_stdout_attempt_{attempt}.log"
    stderr_path = artifact_dir / f"freqai_stderr_attempt_{attempt}.log"
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYARROW_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            [str(part) for part in item["command"]],
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    return {
        "profile_id": item["profile_id"],
        "attempts": attempt,
        "returncode": int(result.returncode),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "finished_at_utc": utc_now(),
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    items_by_id = {item["profile_id"]: item for item in manifest["commands"]}
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    workers = int(manifest["profile_workers"])
    processed = len(manifest["commands"]) - len(pending)
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        for item in batch:
            item["status"] = "running"
            item["started_at_utc"] = utc_now()
        atomic_write_json(manifest, manifest_path)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_profile, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                item = items_by_id[result["profile_id"]]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g1d_freqai_profiles",
                            "processed": processed,
                            "total": len(manifest["commands"]),
                            "profile_id": item["profile_id"],
                            "status": item["status"],
                            "stderr_log": item["stderr_log"],
                        }
                    ),
                    flush=True,
                )
        failures = [item for item in batch if item["status"] == "failed"]
        if failures:
            manifest["status"] = "failed"
            manifest["failed_profile_ids"] = [item["profile_id"] for item in failures]
            manifest["finished_at_utc"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            return int(failures[0]["returncode"])
    return 0


def event_scopes(events: DataFrame) -> dict[str, DataFrame]:
    actual = events["event_scope"].eq("actual_contact")
    return {
        "all_events": events,
        "actual_contacts": events.loc[actual],
        "matched_noncontacts": events.loc[events["event_scope"].eq("matched_noncontact")],
        "actual_isolated_context": events.loc[
            events["event_scope_detail"].eq("actual_isolated_context")
        ],
        "actual_cluster_context": events.loc[
            events["event_scope_detail"].eq("actual_cluster_context")
        ],
        "actual_activity_levels": events.loc[actual & events["level_name"].isin(ACTIVITY_LEVELS)],
        "actual_acceptance_levels": events.loc[
            actual & events["level_name"].isin(ACCEPTANCE_LEVELS)
        ],
    }


def load_event_cache(event_dir: Path, pair: str) -> DataFrame:
    path = event_dir / f"{pair_stem(pair)}.parquet"
    frame = pd.read_parquet(path)
    frame["date"] = normalize_dates(frame["date"])
    return frame


def score_profiles(
    manifest: dict[str, Any], record_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    predictions: dict[str, DataFrame] = {}
    prediction_audit: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        profile_id = str(item["profile_id"])
        frame, audit = load_predictions(Path(item["model_dir"]), pairs)
        audit["profile_id"] = profile_id
        prediction_audit.append(audit)
        if frame.empty:
            raise ValueError(f"No predictions found for completed profile {profile_id}")
        missing = sorted(set(TARGET_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {profile_id} is missing targets: {missing}")
        predictions[profile_id] = frame
    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError("Generation 1D profiles have no common prediction keys")
    missing_pairs = sorted(set(pairs).difference(common_keys["pair"].unique()))
    if missing_pairs:
        raise ValueError(f"Common prediction surface lacks pairs: {missing_pairs}")
    eligibility_path = record_dir / "g1d_eligibility_audit.parquet"
    eligibility.to_parquet(eligibility_path, index=False)
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    events_by_pair = {pair: load_event_cache(event_dir, pair) for pair in pairs}
    rows: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        fair = frame.merge(common_keys, on=["pair", "date"], how="inner")
        for pair in pairs:
            pair_predictions = fair.loc[fair["pair"].eq(pair)]
            scopes = event_scopes(events_by_pair[pair])
            for scope, scope_events in scopes.items():
                labelled_events = scope_events.rename(
                    columns={target: f"{target}_actual" for target in TARGET_COLUMNS}
                )
                merged = labelled_events.merge(
                    pair_predictions,
                    on="date",
                    how="inner",
                    validate="one_to_one",
                )
                for period, period_frame in merged.groupby("period", sort=False, observed=True):
                    for target in TARGET_COLUMNS:
                        family, horizon = target_metadata(target)
                        rows.append(
                            {
                                "profile_id": profile_id,
                                "pair": pair,
                                "period": str(period),
                                "scope": scope,
                                "target": target,
                                "target_family": family,
                                "horizon_hours": horizon,
                                **score_values(
                                    period_frame,
                                    target,
                                    f"{target}_actual",
                                ),
                            }
                        )
    scores = attach_profile_deltas(DataFrame(rows))
    scores = attach_contact_specificity(scores)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = artifact_dir / "g1d_pair_period_scores.parquet"
    scores.to_parquet(detailed_path, index=False)
    summary = summarize_scores(scores, manifest)
    summary_path = record_dir / "g1d_freqai_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    prediction_audit_path = record_dir / "g1d_prediction_file_audit.json"
    atomic_write_json(
        {
            "created_at_utc": utc_now(),
            "profiles": prediction_audit,
            "common_prediction_rows": len(common_keys),
            "common_prediction_key_digest": eligibility_digest(common_keys),
            "missing_pairs": missing_pairs,
        },
        prediction_audit_path,
    )
    result = {
        "created_at_utc": utc_now(),
        "profiles": len(predictions),
        "pairs": len(pairs),
        "targets": len(TARGET_COLUMNS),
        "common_prediction_rows": len(common_keys),
        "detailed_score_rows": len(scores),
        "summary_rows": len(summary),
        "detailed_scores": str(detailed_path),
        "summary": str(summary_path),
        "eligibility_audit": str(eligibility_path),
        "prediction_file_audit": str(prediction_audit_path),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "g1d_freqai_result_record.json")
    return result


SCORE_METRICS = (
    "prediction_actual_spearman",
    "top_minus_bottom",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r_squared",
    "prediction_bias",
)


def assign_control_metrics(
    output: DataFrame,
    *,
    scores: DataFrame,
    selected: pd.Series,
    control_profile: str,
    prefix: str,
    keys: list[str],
) -> None:
    """Assign one profile's metrics by exact score-cell identity without merge suffixes."""
    controls = scores.loc[
        scores["profile_id"].eq(control_profile),
        [*keys, *SCORE_METRICS],
    ].set_index(keys)
    if controls.index.duplicated().any():
        raise ValueError(
            f"Profile {control_profile!r} has duplicate score cells for keys {keys}."
        )
    wanted = pd.MultiIndex.from_frame(output.loc[selected, keys])
    for metric in SCORE_METRICS:
        output.loc[selected, f"{prefix}{metric}"] = controls[metric].reindex(wanted).to_numpy()


def attach_profile_deltas(
    scores: DataFrame,
) -> DataFrame:
    keys = ["pair", "period", "scope", "target"]
    output = scores.copy()
    local = scores.loc[scores["profile_id"].eq("local_state"), [*keys, *SCORE_METRICS]]
    local = local.rename(columns={metric: f"local_state_{metric}" for metric in SCORE_METRICS})
    output = output.merge(local, on=keys, how="left", validate="many_to_one")
    baseline_lookup = {profile: definition["baseline"] for profile, definition in PROFILES.items()}
    for profile_id, baseline_id in baseline_lookup.items():
        if baseline_id is None:
            continue
        selected = output["profile_id"].eq(profile_id)
        assign_control_metrics(
            output,
            scores=scores,
            selected=selected,
            control_profile=baseline_id,
            prefix="incremental_baseline_",
            keys=keys,
        )
    for metric in ("prediction_actual_spearman", "top_minus_bottom", "r_squared"):
        output[f"{metric}_delta_vs_local_state"] = output[metric] - output[f"local_state_{metric}"]
        output[f"{metric}_delta_vs_incremental_baseline"] = (
            output[metric] - output[f"incremental_baseline_{metric}"]
        )
    for metric in ("mean_absolute_error", "root_mean_squared_error"):
        output[f"{metric}_skill_vs_local_state"] = output[f"local_state_{metric}"] - output[metric]
        output[f"{metric}_skill_vs_incremental_baseline"] = (
            output[f"incremental_baseline_{metric}"] - output[metric]
        )
    output["absolute_bias_skill_vs_incremental_baseline"] = (
        output["incremental_baseline_prediction_bias"].abs() - output["prediction_bias"].abs()
    )
    for current, definition in PROFILES.items():
        placebo_id = definition["placebo"]
        if placebo_id is None:
            continue
        selected = output["profile_id"].eq(current)
        assign_control_metrics(
            output,
            scores=scores,
            selected=selected,
            control_profile=placebo_id,
            prefix="matching_placebo_",
            keys=keys,
        )
        for metric in ("prediction_actual_spearman", "top_minus_bottom", "r_squared"):
            output.loc[selected, f"{metric}_delta_vs_matching_placebo"] = (
                output.loc[selected, metric] - output.loc[selected, f"matching_placebo_{metric}"]
            )
        for metric in ("mean_absolute_error", "root_mean_squared_error"):
            output.loc[selected, f"{metric}_skill_vs_matching_placebo"] = (
                output.loc[selected, f"matching_placebo_{metric}"] - output.loc[selected, metric]
            )
    return output


def attach_contact_specificity(scores: DataFrame) -> DataFrame:
    output = scores.copy()
    keys = ["profile_id", "pair", "period", "target"]
    metrics = (
        "mean_absolute_error_skill_vs_incremental_baseline",
        "prediction_actual_spearman_delta_vs_incremental_baseline",
        "mean_absolute_error_skill_vs_matching_placebo",
        "prediction_actual_spearman_delta_vs_matching_placebo",
    )
    controls = output.loc[output["scope"].eq("matched_noncontacts"), [*keys, *metrics]]
    controls = controls.rename(columns={metric: f"control_scope_{metric}" for metric in metrics})
    output = output.merge(controls, on=keys, how="left", validate="many_to_one")
    actual = output["scope"].eq("actual_contacts")
    for metric in metrics:
        output.loc[actual, f"actual_minus_control_{metric}"] = (
            output.loc[actual, metric] - output.loc[actual, f"control_scope_{metric}"]
        )
    return output


def manifest_cohorts(manifest: dict[str, Any]) -> dict[str, str]:
    generation0 = load_manifest(Path(manifest["source_contracts"]["generation0_manifest"]))
    output: dict[str, str] = {}
    for cohort, pairs in generation0["reporting_groups"]["coin_cohorts"].items():
        for pair in pairs:
            output[pair] = cohort
    missing = sorted(set(manifest["pairs"]).difference(output))
    if missing:
        raise ValueError(f"Pairs lack a frozen reporting cohort: {missing}")
    return output


def summarize_scores(scores: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    cohorts = manifest_cohorts(manifest)
    primary = scores.copy()
    primary["market_scope"] = "frozen_cohort:" + primary["pair"].map(cohorts)
    broad = scores.copy()
    broad["market_scope"] = "all_top10"
    non_btc = scores.loc[~scores["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["market_scope"] = "all_non_btc"
    working = pd.concat([primary, broad, non_btc], ignore_index=True)
    working = working.loc[working["status"].eq("scored")].copy()
    numeric_metrics = (
        "prediction_actual_spearman",
        "mean_absolute_error",
        "top_minus_bottom",
        "prediction_actual_spearman_delta_vs_incremental_baseline",
        "mean_absolute_error_skill_vs_incremental_baseline",
        "prediction_actual_spearman_delta_vs_matching_placebo",
        "mean_absolute_error_skill_vs_matching_placebo",
        "actual_minus_control_mean_absolute_error_skill_vs_incremental_baseline",
        "actual_minus_control_prediction_actual_spearman_delta_vs_incremental_baseline",
        "actual_minus_control_mean_absolute_error_skill_vs_matching_placebo",
        "actual_minus_control_prediction_actual_spearman_delta_vs_matching_placebo",
    )
    for column in numeric_metrics:
        if column in working:
            working[f"{column}_positive"] = working[column].gt(0.0).astype(float)
    keys = [
        "profile_id",
        "market_scope",
        "scope",
        "target_family",
        "horizon_hours",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "pair_period_tests": ("pair", "size"),
        "distinct_pairs": ("pair", "nunique"),
        "distinct_periods": ("period", "nunique"),
        "scored_rows": ("rows", "sum"),
    }
    for column in numeric_metrics:
        if column not in working:
            continue
        aggregations[f"median_{column}"] = (column, "median")
        positive = f"{column}_positive"
        if positive in working:
            aggregations[f"positive_fraction_{column}"] = (positive, "mean")
    return (
        working.groupby(keys, observed=True, dropna=False, sort=False)
        .agg(**aggregations)
        .reset_index()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the purged, contact-focused Generation 1D FreqAI residual-information ladder."
        )
    )
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--timerange", default="20220601-20260719")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--backtest-days", type=int, default=180)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    validate_frozen_branch()
    profiles = tuple(PROFILES) if args.profiles == "all" else parse_csv(args.profiles)
    pairs = DEFAULT_PAIRS if args.pairs == "all" else parse_csv(args.pairs)
    unknown_profiles = sorted(set(profiles).difference(PROFILES))
    unknown_pairs = sorted(set(pairs).difference(DEFAULT_PAIRS))
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {unknown_profiles}")
    if unknown_pairs:
        raise ValueError(f"Pairs are outside the frozen Generation 1 universe: {unknown_pairs}")
    if args.train_days <= 0 or args.backtest_days <= 0:
        raise ValueError("train-days and backtest-days must be positive")
    for label, workers in (
        ("cache-workers", args.cache_workers),
        ("profile-workers", args.profile_workers),
    ):
        if workers < 1 or workers > MAX_PROFILE_WORKERS:
            raise ValueError(f"{label} must be between 1 and {MAX_PROFILE_WORKERS}")
    run_id = args.run_id or f"g1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_dir = args.record_root / run_id
    artifact_dir = args.artifact_root / run_id
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if args.score_only:
            raise FileNotFoundError(f"Cannot score an unprepared run: {manifest_path}")
        cache_inventory = prepare_caches(
            pairs=pairs,
            manifest_path=args.manifest,
            feature_dir=artifact_dir / "feature_cache",
            event_dir=artifact_dir / "event_cache",
            workers=args.cache_workers,
        )
        manifest = build_manifest(
            run_id=run_id,
            record_dir=record_dir,
            artifact_dir=artifact_dir,
            base_config=args.config,
            python_exe=args.python_exe,
            profiles=profiles,
            pairs=pairs,
            timerange=args.timerange,
            train_days=args.train_days,
            backtest_days=args.backtest_days,
            profile_workers=args.profile_workers,
            technical_smoke=args.technical_smoke,
            cache_inventory=cache_inventory,
        )
        manifest["command"] = [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        atomic_write_json(manifest, manifest_path)
    validate_manifest_request(
        manifest,
        profiles=profiles,
        pairs=pairs,
        timerange=args.timerange,
        train_days=args.train_days,
        backtest_days=args.backtest_days,
        profile_workers=args.profile_workers,
        technical_smoke=args.technical_smoke,
        python_exe=args.python_exe,
    )
    audit = preflight(manifest, args.python_exe)
    atomic_write_json(audit, record_dir / "preflight.json")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2))
        return 2
    if args.prepare_only or args.dry_run:
        manifest["status"] = "prepared"
        atomic_write_json(manifest, manifest_path)
        print(json.dumps({"manifest": str(manifest_path), "preflight": audit}, indent=2))
        return 0
    if not args.score_only:
        returncode = run_manifest(manifest, manifest_path)
        if returncode != 0:
            return returncode
    manifest["status"] = "scoring"
    atomic_write_json(manifest, manifest_path)
    result = score_profiles(manifest, record_dir, artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result_record"] = str(record_dir / "g1d_freqai_result_record.json")
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
