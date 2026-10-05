from __future__ import annotations

# Bound numerical libraries before pandas/FreqAI helpers are imported.
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
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    common_prediction_keys,
    eligibility_digest,
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (  # noqa: E501
    MIN_SCORABLE_ROWS,
    WIDE_SOURCE_COLUMNS,
    deterministic_shift,
    g3a_event_path_by_pair,
    regression_metrics,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    normalize_dates,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation5_arrival_path_preflight import (  # noqa: E501
    PATH_FEATURES,
    attach_path_features,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration5Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration5Strategy.py"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
RECORD_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5b_freqai_volume_clock_and_feature_ablation"
)
LARGE_ROOT = Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
ARTIFACT_ROOT = (
    LARGE_ROOT
    / "generation5_branches"
    / "g5b_freqai_volume_clock_and_feature_ablation"
)
G4D_RECORD_ROOT = (
    OUTPUT_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
G4D_ARTIFACT_ROOT = (
    LARGE_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
G3A_RECORD_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3a_thin_lvn_attribution"
G5D_RECORD_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5d_fresh_arrival_path_state_decomposition"
    / "g5d_path_preflight_20260821b"
)
G5D_ARTIFACT_ROOT = (
    LARGE_ROOT
    / "generation5_branches"
    / "g5d_fresh_arrival_path_state_decomposition"
    / "g5d_path_preflight_20260821b"
    / "pair_path_surfaces"
)
NORMAL_MANIFEST = OUTPUT_ROOT / "generation0_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation2_shared" / "g2_meme_reaction_manifest.json"

OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MAX_WORKERS = 4
MIN_RETAIN_COINS = 5
BLOCK_BOOTSTRAP_SAMPLES = 1024
BLOCK_DAYS = 7
STRUCTURALLY_OPTIONAL_BLOCKS = frozenset({"pre_proximity"})

CONTACT_TARGET = TARGET_COLUMNS[0]
NEXT_VOLUME_TARGET = TARGET_COLUMNS[1]
G4D_CONTACT_TARGET = "&-g4d_contact_volume_ratio"

SURFACES: dict[str, dict[str, Any]] = {
    "normal_thin_lvn_mixed_cluster": {
        "cohort": "normal",
        "source_family": "g3a",
        "g4d_run_id": "g4d_full_g3a_normal_20260820b_time_baseline_repair",
        "g3a_run_id": "g3a_lvn_cluster_attribution_normal10_full_20260814b",
        "manifest": NORMAL_MANIFEST,
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
        "role": "positive_g4d_surface",
    },
    "meme_thin_lvn_mixed_cluster": {
        "cohort": "meme",
        "source_family": "g3a",
        "g4d_run_id": "g4d_full_g3a_meme_20260820b_time_baseline_repair",
        "g3a_run_id": "g3a_lvn_cluster_attribution_meme_full_20260814b",
        "manifest": MEME_MANIFEST,
        "timerange": "20260101-20260714",
        "train_days": 160,
        "backtest_days": 30,
        "validation_periods": ("meme_validation_early", "meme_validation_late"),
        "role": "negative_or_descriptive_surface",
    },
    "normal_isolated_confirmed_swing_density": {
        "cohort": "normal",
        "source_family": "g3d",
        "g4d_run_id": "g4d_full_g3d_normal_isolated_20260820b_time_baseline_repair",
        "manifest": NORMAL_MANIFEST,
        "actual_scope": "single_density_zone",
        "g5d_scope_id": "normal_isolated_168h_confirmed_swing_density",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
        "role": "positive_g4d_surface",
    },
    "normal_confirmed_swing_density_cluster": {
        "cohort": "normal",
        "source_family": "g3d",
        "g4d_run_id": "g4d_full_g3d_normal_cluster_20260820b_time_baseline_repair",
        "manifest": NORMAL_MANIFEST,
        "actual_scope": "density_cluster",
        "g5d_scope_id": "normal_168h_confirmed_swing_density_cluster",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
        "role": "positive_g4d_surface",
    },
    "meme_confirmed_swing_density_cluster": {
        "cohort": "meme",
        "source_family": "g3d",
        "g4d_run_id": "g4d_full_g3d_meme_cluster_20260820b_time_baseline_repair",
        "manifest": MEME_MANIFEST,
        "actual_scope": "density_cluster",
        "g5d_scope_id": "meme_168h_confirmed_swing_density_cluster",
        "timerange": "20260101-20260714",
        "train_days": 160,
        "backtest_days": 30,
        "validation_periods": ("meme_validation_early", "meme_validation_late"),
        "role": "positive_g4d_surface",
    },
}

PROFILES: dict[str, dict[str, Any]] = {
    "state_only": {
        "strategy": "MarketReactionZoneG5BStateOnlyFreqAIResearchStrategy",
        "clock": "strict_pre_contact",
        "role": "Prior-candle OHLCV, indicators, BTC, and cohort state only.",
    },
    "pre_contact": {
        "strategy": "MarketReactionZoneG5BPreContactFreqAIResearchStrategy",
        "clock": "strict_pre_contact",
        "role": "State plus causal path, level, and nearby-structure groups.",
    },
    "pre_shuffled": {
        "strategy": "MarketReactionZoneG5BPreContactShuffledFreqAIResearchStrategy",
        "clock": "strict_pre_contact_placebo",
        "role": "State plus non-self candidate groups shuffled within pair-period.",
    },
    "pre_stale": {
        "strategy": "MarketReactionZoneG5BPreContactStaleFreqAIResearchStrategy",
        "clock": "strict_pre_contact_placebo",
        "role": "State plus candidate groups copied from the prior chronological event.",
    },
    "event_resolution": {
        "strategy": "MarketReactionZoneG5BEventResolutionFreqAIResearchStrategy",
        "clock": "contact_close",
        "role": "Pre-contact groups plus contacted-component/event-state geometry.",
    },
    "contact_close": {
        "strategy": "MarketReactionZoneG5BContactCloseFreqAIResearchStrategy",
        "clock": "contact_close",
        "role": "Event-resolution profile plus completed contact-candle OHLCV.",
    },
    "no_path": {
        "strategy": "MarketReactionZoneG5BNoPathFreqAIResearchStrategy",
        "clock": "strict_pre_contact_ablation",
        "role": "Pre-contact profile with the approach-path group removed.",
    },
    "no_level": {
        "strategy": "MarketReactionZoneG5BNoLevelFreqAIResearchStrategy",
        "clock": "strict_pre_contact_ablation",
        "role": "Pre-contact profile with level identity/width/support removed.",
    },
    "no_proximity": {
        "strategy": "MarketReactionZoneG5BNoProximityFreqAIResearchStrategy",
        "clock": "strict_pre_contact_ablation",
        "role": "Pre-contact profile with nearby-zone/cluster geometry removed.",
    },
    "candidate_only": {
        "strategy": "MarketReactionZoneG5BCandidateOnlyFreqAIResearchStrategy",
        "clock": "strict_pre_contact_ablation",
        "role": "Candidate groups with local and broad market state removed.",
    },
}

CLOCK_COMPARISONS = (
    (
        "strict_precontact_contact_volume_vs_state",
        CONTACT_TARGET,
        "pre_contact",
        "state_only",
        "strict_pre_contact_forecast",
    ),
    (
        "strict_precontact_contact_volume_vs_shuffle",
        CONTACT_TARGET,
        "pre_contact",
        "pre_shuffled",
        "strict_pre_contact_forecast",
    ),
    (
        "strict_precontact_contact_volume_vs_stale",
        CONTACT_TARGET,
        "pre_contact",
        "pre_stale",
        "strict_pre_contact_forecast",
    ),
    (
        "contact_close_nowcast_increment",
        CONTACT_TARGET,
        "event_resolution",
        "pre_contact",
        "same_candle_description_or_nowcast",
    ),
    (
        "contact_close_next_volume_vs_state",
        NEXT_VOLUME_TARGET,
        "contact_close",
        "state_only",
        "contact_close_causal_next_candle_forecast",
    ),
    (
        "contact_close_next_volume_vs_shuffle",
        NEXT_VOLUME_TARGET,
        "contact_close",
        "pre_shuffled",
        "contact_close_causal_next_candle_forecast",
    ),
    (
        "contact_close_next_volume_vs_stale",
        NEXT_VOLUME_TARGET,
        "contact_close",
        "pre_stale",
        "contact_close_causal_next_candle_forecast",
    ),
    (
        "path_group_increment",
        CONTACT_TARGET,
        "pre_contact",
        "no_path",
        "strict_pre_contact_group_ablation",
    ),
    (
        "level_group_increment",
        CONTACT_TARGET,
        "pre_contact",
        "no_level",
        "strict_pre_contact_group_ablation",
    ),
    (
        "proximity_group_increment",
        CONTACT_TARGET,
        "pre_contact",
        "no_proximity",
        "strict_pre_contact_group_ablation",
    ),
    (
        "market_state_group_increment",
        CONTACT_TARGET,
        "pre_contact",
        "candidate_only",
        "strict_pre_contact_group_ablation",
    ),
    (
        "contact_ohlcv_increment_for_next_volume",
        NEXT_VOLUME_TARGET,
        "contact_close",
        "event_resolution",
        "contact_close_group_ablation",
    ),
)


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def pair_stem(pair: str) -> str:
    return pair.strip().upper().replace("/", "_").replace(":", "_")


def validate_frozen_branch() -> dict[str, Any]:
    payload = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branch = next(
        (
            item
            for item in payload.get("branches", [])
            if item.get("id") == "g5b_freqai_volume_clock_and_feature_ablation"
        ),
        None,
    )
    if branch is None or branch.get("status") != "frozen_next_batch":
        raise ValueError("The frozen Generation 5 batch does not expose G5B.")
    if payload.get("generation") != 5 or payload.get("branch_layer") != 5:
        raise ValueError("G5B requires the frozen fifth-generation branch layer.")
    expected_surfaces = {
        "normal_thin_lvn_mixed_cluster",
        "normal_isolated_confirmed_swing_density",
        "normal_confirmed_swing_density_cluster",
        "meme_confirmed_swing_density_cluster",
    }
    if set(branch.get("fixed_surfaces", [])) != expected_surfaces:
        raise ValueError("The G5B positive surface contract changed.")
    if branch.get("negative_or_descriptive_surface") != "meme_thin_lvn_mixed_cluster":
        raise ValueError("The G5B negative/descriptive surface changed.")
    return branch


def load_g4d_source(surface_id: str, surface: dict[str, Any]) -> dict[str, Any]:
    run_id = str(surface["g4d_run_id"])
    record_dir = G4D_RECORD_ROOT / run_id
    artifact_dir = G4D_ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "manifest.json"
    result_path = record_dir / "g4d_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else result_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or result.get("surface_id") is None:
        raise ValueError(f"The G4D source run is incomplete: {run_id}")
    expected_source_surface = {
        "normal_thin_lvn_mixed_cluster": "g3a_normal_thin_lvn_cluster",
        "meme_thin_lvn_mixed_cluster": "g3a_meme_thin_lvn_cluster",
        "normal_isolated_confirmed_swing_density": "g3d_normal_isolated_density",
        "normal_confirmed_swing_density_cluster": "g3d_normal_density_cluster",
        "meme_confirmed_swing_density_cluster": "g3d_meme_density_cluster",
    }[surface_id]
    if manifest.get("surface_id") != expected_source_surface:
        raise ValueError(f"G4D source surface mismatch for {surface_id}.")
    return {
        "run_id": run_id,
        "record_dir": record_dir,
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "result_path": result_path,
        "result_sha256": sha256_file(result_path),
        "pairs": tuple(str(pair) for pair in manifest["pairs"]),
    }


def select_pairs(allowed: Sequence[str], requested: str) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return tuple(allowed)
    selected = parse_csv(requested)
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid G5B pairs: selected={selected}, unknown={unknown}")
    return selected


def contact_clock_frame(base: DataFrame) -> DataFrame:
    output = base[["date", "open", "high", "low", "close", "volume"]].copy()
    high = pd.to_numeric(output["high"], errors="coerce")
    low = pd.to_numeric(output["low"], errors="coerce")
    open_ = pd.to_numeric(output["open"], errors="coerce")
    close = pd.to_numeric(output["close"], errors="coerce")
    volume = pd.to_numeric(output["volume"], errors="coerce").clip(lower=0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    prior_volume = volume.shift(1).rolling(24, min_periods=24).median().replace(0.0, np.nan)
    through_contact_volume = volume.rolling(24, min_periods=24).median().replace(0.0, np.nan)
    prior_range = candle_range.shift(1).rolling(24, min_periods=24).median().replace(0.0, np.nan)
    prior_close = close.shift(1).replace(0.0, np.nan)
    output[CONTACT_TARGET] = volume / prior_volume
    output[NEXT_VOLUME_TARGET] = volume.shift(-1) / through_contact_volume
    output["contact__volume_ratio"] = volume / prior_volume
    output["contact__range_ratio"] = candle_range / prior_range
    output["contact__body_fraction"] = ((close - open_) / candle_range).clip(-1.0, 1.0)
    output["contact__close_location"] = (((close - low) / candle_range) * 2.0 - 1.0).clip(
        -1.0,
        1.0,
    )
    output["contact__pressure"] = (2.0 * close - high - low) / candle_range
    output["contact__return_from_prior_close"] = close / prior_close - 1.0
    return output.drop(columns=["open", "high", "low", "close", "volume"])


def source_event_rows(
    *,
    pair: str,
    source: dict[str, Any],
) -> tuple[DataFrame, DataFrame]:
    feature_path = source["artifact_dir"] / "feature_cache" / f"{pair_stem(pair)}.parquet"
    event_path = source["artifact_dir"] / "event_cache" / f"{pair_stem(pair)}.parquet"
    if not feature_path.is_file() or not event_path.is_file():
        raise FileNotFoundError(feature_path if not feature_path.is_file() else event_path)
    features = pd.read_parquet(feature_path)
    events = pd.read_parquet(event_path)
    features["date"] = normalize_dates(features["date"])
    events["date"] = normalize_dates(events["date"])
    if features["date"].duplicated().any() or events["date"].duplicated().any():
        raise ValueError(f"Duplicate G4D source dates for {pair}.")
    return features, events


def classify_source_features(
    event_features: DataFrame,
) -> dict[str, tuple[str, ...]]:
    columns = [
        column
        for column in event_features
        if column.startswith(("level__", "geometry__", "mtf__"))
    ]
    path = tuple(
        column
        for column in columns
        if column == "level__pre_distance_atr" or column.startswith("level__approach_")
    )
    resolution = tuple(
        column
        for column in columns
        if column.startswith("level__arrival_")
        or (column.startswith("geometry__") and "contact" in column)
    )
    proximity = tuple(
        column
        for column in columns
        if column.startswith("mtf__")
        or (column.startswith("geometry__") and column not in resolution)
    )
    excluded = set((*path, *resolution, *proximity))
    level = tuple(column for column in columns if column not in excluded)
    return {
        "path": path,
        "level": level,
        "proximity": proximity,
        "resolution": resolution,
    }


def nearest_candidate_rows(
    events: DataFrame,
    candidates: DataFrame,
    *,
    value_column: str,
    candidate_value_column: str,
    extra_keys: Sequence[str],
) -> DataFrame:
    rows: list[pd.Series] = []
    grouped = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in candidates.groupby(list(extra_keys), dropna=False, sort=False)
    }
    for event in events.itertuples(index=False):
        key = tuple(getattr(event, column) for column in extra_keys)
        options = grouped.get(key)
        if options is None or options.empty:
            continue
        wanted = float(getattr(event, value_column))
        distance = (
            pd.to_numeric(options[candidate_value_column], errors="coerce") - wanted
        ).abs()
        if distance.notna().sum() == 0:
            continue
        chosen = options.loc[distance.idxmin()].copy()
        chosen["_match_distance"] = float(distance.min())
        rows.append(chosen)
    return DataFrame(rows).reset_index(drop=True) if rows else DataFrame()


def g3a_path_features(
    *,
    pair: str,
    events: DataFrame,
    event_features: DataFrame,
    base: DataFrame,
    surface: dict[str, Any],
) -> tuple[DataFrame, dict[str, Any]]:
    record_path = G3A_RECORD_ROOT / str(surface["g3a_run_id"]) / "g3a_run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    raw_path = g3a_event_path_by_pair(record).get(pair)
    if raw_path is None or not raw_path.is_file():
        raise FileNotFoundError(raw_path or f"G3A event source for {pair}")
    raw = pd.read_parquet(
        raw_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", ["lvn_above", "lvn_below"]),
            ("zone_method", "==", "wide_base_atr"),
            ("source_timeframe", "==", "1h"),
        ],
        columns=[
            "pair",
            "event_time",
            "base_index",
            "level_name",
            "approach_state",
            "level_price",
            "zone_half_width",
            "zone_half_width_atr",
            "base_atr",
            "pre_distance_atr",
        ],
    ).rename(columns={"event_time": "date", "level_name": "level_identity"})
    raw["date"] = normalize_dates(raw["date"])
    lookup = events[["date", "level_identity", "approach_state"]].merge(
        event_features[["date", "level__pre_distance_atr"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    matched = nearest_candidate_rows(
        lookup,
        raw,
        value_column="level__pre_distance_atr",
        candidate_value_column="pre_distance_atr",
        extra_keys=("date", "level_identity", "approach_state"),
    )
    if len(matched) != len(events) or matched["_match_distance"].gt(1e-10).any():
        raise ValueError(f"G3A path geometry did not map exactly for {pair}.")
    matched["event_state"] = events.set_index("date").loc[matched["date"], "event_state"].to_numpy()
    path = attach_path_features(matched, base=base, zone_surface=None)
    return path[["date", *PATH_FEATURES]], {
        "path": str(raw_path),
        "sha256": sha256_file(raw_path),
        "rows": len(path),
        "maximum_match_distance": float(matched["_match_distance"].max()),
    }


def g3d_path_features(
    *,
    pair: str,
    events: DataFrame,
    event_features: DataFrame,
    surface: dict[str, Any],
) -> tuple[DataFrame, dict[str, Any]]:
    cohort = str(surface["cohort"])
    path = G5D_ARTIFACT_ROOT / f"{cohort}__{pair_stem(pair)}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates = pd.read_parquet(path)
    candidates = candidates.loc[
        candidates["boundary_kind"].eq("actual_density_zone")
        & candidates["scope_id"].eq(surface["g5d_scope_id"])
    ].copy()
    candidates["date"] = normalize_dates(candidates["event_time"])
    lookup = events[["date", "event_state"]].merge(
        event_features[["date", "level__pre_distance_atr"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    matched = nearest_candidate_rows(
        lookup,
        candidates,
        value_column="level__pre_distance_atr",
        candidate_value_column="pre_distance_atr",
        extra_keys=("date", "event_state"),
    )
    if len(matched) != len(events) or matched["_match_distance"].gt(1e-10).any():
        raise ValueError(f"G3D path geometry did not map exactly for {pair}.")
    return matched[["date", *PATH_FEATURES]], {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(matched),
        "maximum_match_distance": float(matched["_match_distance"].max()),
        "g5d_terminal_status": "parked_insufficient_path_common_support",
        "reuse_boundary": (
            "Only the outcome-blind actual path geometry is reused; G5D's unsupported "
            "actual-versus-pseudo reaction claim remains parked."
        ),
    }


def rename_block(frame: DataFrame, columns: Sequence[str], prefix: str) -> DataFrame:
    output = frame[["date", *columns]].copy()
    output = output.rename(
        columns={
            column: f"{prefix}__{column.replace('__', '_', 1)}" for column in columns
        }
    )
    return output


def complete_nonconstant_columns(frame: DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(f"{prefix}__")]
    return [
        column
        for column in columns
        if frame[column].notna().all() and frame[column].nunique(dropna=True) > 1
    ]


def prune_feature_block(
    frame: DataFrame,
    *,
    prefix: str,
    pair: str,
    surface_id: str,
) -> tuple[DataFrame, list[str]]:
    keep = complete_nonconstant_columns(frame, prefix)
    drop = [
        column
        for column in frame
        if column.startswith(f"{prefix}__") and column not in keep
    ]
    output = frame.drop(columns=drop)
    if not keep and prefix not in STRUCTURALLY_OPTIONAL_BLOCKS:
        raise ValueError(f"G5B {prefix} block is empty for {pair} and {surface_id}.")
    return output, keep


def attach_placebo_groups(
    events: DataFrame,
    *,
    candidate_columns: Sequence[str],
    pair: str,
    surface_id: str,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    output = events.sort_values("date", kind="stable").reset_index(drop=True).copy()
    audits: list[dict[str, Any]] = []
    for source_prefix, destination_prefix in (
        ("pre_path__", "shuffle_path__"),
        ("pre_level__", "shuffle_level__"),
        ("pre_proximity__", "shuffle_proximity__"),
        ("pre_path__", "stale_path__"),
        ("pre_level__", "stale_level__"),
        ("pre_proximity__", "stale_proximity__"),
    ):
        for column in candidate_columns:
            if column.startswith(source_prefix):
                output[column.replace(source_prefix, destination_prefix, 1)] = np.nan
    for period, positions in output.groupby("period", sort=False).groups.items():
        indexes = np.asarray(list(positions), dtype=np.int64)
        if len(indexes) < 2:
            continue
        shift = deterministic_shift(len(indexes), f"g5b|{surface_id}|{pair}|{period}")
        shuffled_source = np.roll(indexes, shift)
        self_assignments = int(np.sum(shuffled_source == indexes))
        if self_assignments:
            raise AssertionError("G5B shuffled placebo retained self assignments.")
        stale_source = np.r_[[-1], indexes[:-1]]
        for source_prefix, shuffle_prefix, stale_prefix in (
            ("pre_path__", "shuffle_path__", "stale_path__"),
            ("pre_level__", "shuffle_level__", "stale_level__"),
            ("pre_proximity__", "shuffle_proximity__", "stale_proximity__"),
        ):
            columns = [column for column in candidate_columns if column.startswith(source_prefix)]
            for column in columns:
                output.loc[indexes, column.replace(source_prefix, shuffle_prefix, 1)] = (
                    output.loc[shuffled_source, column].to_numpy()
                )
                output.loc[indexes[1:], column.replace(source_prefix, stale_prefix, 1)] = (
                    output.loc[stale_source[1:], column].to_numpy()
                )
        audits.append(
            {
                "pair": pair,
                "period": str(period),
                "rows": len(indexes),
                "shuffle": shift,
                "shuffle_self_assignments": self_assignments,
                "stale_missing_first_row": 1,
                "stale_future_source_violations": 0,
            }
        )
    return output, audits


def build_pair_cache(
    pair: str,
    *,
    surface_id: str,
    surface: dict[str, Any],
    source: dict[str, Any],
    feature_dir: Path,
    event_dir: Path,
) -> dict[str, Any]:
    source_features, source_events = source_event_rows(pair=pair, source=source)
    manifest = load_manifest(Path(surface["manifest"]))
    base = prepare_base_market_frame(pair, manifest)
    base["date"] = normalize_dates(base["date"])
    clock = contact_clock_frame(base)
    events = source_events.merge(
        clock,
        on="date",
        how="left",
        validate="one_to_one",
    )
    contact_difference = (
        pd.to_numeric(events[G4D_CONTACT_TARGET], errors="coerce")
        - pd.to_numeric(events[CONTACT_TARGET], errors="coerce")
    ).abs()
    if contact_difference.max(skipna=True) > 1e-10:
        raise ValueError(f"Recomputed G5B contact-volume target differs for {pair}.")
    event_source_features = source_features.loc[
        source_features["date"].isin(events["date"])
    ].copy()
    event_source_features = events[["date"]].merge(
        event_source_features,
        on="date",
        how="left",
        validate="one_to_one",
    )
    groups = classify_source_features(event_source_features)
    if surface["source_family"] == "g3a":
        path_frame, path_contract = g3a_path_features(
            pair=pair,
            events=events,
            event_features=event_source_features,
            base=base,
            surface=surface,
        )
    else:
        path_frame, path_contract = g3d_path_features(
            pair=pair,
            events=events,
            event_features=event_source_features,
            surface=surface,
        )

    path_columns = [column for column in PATH_FEATURES if path_frame[column].notna().all()]
    pre_path = rename_block(path_frame, path_columns, "pre_path")
    source_path = rename_block(event_source_features, groups["path"], "pre_path_source")
    pre_path = pre_path.merge(source_path, on="date", how="left", validate="one_to_one")
    pre_path = pre_path.rename(
        columns={
            column: column.replace("pre_path_source__", "pre_path__", 1)
            for column in pre_path
            if column.startswith("pre_path_source__")
        }
    )
    pre_level = rename_block(event_source_features, groups["level"], "pre_level")
    pre_proximity = rename_block(
        event_source_features,
        groups["proximity"],
        "pre_proximity",
    )
    resolution = rename_block(event_source_features, groups["resolution"], "resolution")
    for state in sorted(events["event_state"].astype(str).unique()):
        resolution[f"resolution__event_state_{state}"] = (
            events["event_state"].astype(str).eq(state).astype(float).to_numpy()
        )
    contact_columns = [column for column in clock if column.startswith("contact__")]
    contact = events[["date", *contact_columns]].copy()
    blocks = events[
        [
            "date",
            "period",
            "event_state",
            "level_identity",
            "approach_state",
            "source_available_at",
            CONTACT_TARGET,
            NEXT_VOLUME_TARGET,
        ]
    ].merge(pre_path, on="date", how="left", validate="one_to_one")
    for block in (pre_level, pre_proximity, resolution, contact):
        blocks = blocks.merge(block, on="date", how="left", validate="one_to_one")

    retained_blocks: dict[str, list[str]] = {}
    for prefix in ("pre_path", "pre_level", "pre_proximity", "resolution", "contact"):
        blocks, retained_blocks[prefix] = prune_feature_block(
            blocks,
            prefix=prefix,
            pair=pair,
            surface_id=surface_id,
        )

    candidate_columns = [
        column
        for column in blocks
        if column.startswith(("pre_path__", "pre_level__", "pre_proximity__"))
    ]
    blocks, placebo_audit = attach_placebo_groups(
        blocks,
        candidate_columns=candidate_columns,
        pair=pair,
        surface_id=surface_id,
    )
    required_features = [
        column
        for column in blocks
        if column.startswith(
            (
                "pre_path__",
                "pre_level__",
                "pre_proximity__",
                "resolution__",
                "contact__",
                "shuffle_path__",
                "shuffle_level__",
                "shuffle_proximity__",
                "stale_path__",
                "stale_level__",
                "stale_proximity__",
            )
        )
    ]
    numeric_required = blocks[[*required_features, *TARGET_COLUMNS]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric_required.notna().all(axis=1)
    incomplete_rows = int((~complete).sum())
    blocks = blocks.loc[complete].reset_index(drop=True)
    if len(blocks) < MIN_SCORABLE_ROWS:
        raise ValueError(f"Too few complete G5B events remain for {pair}: {len(blocks)}")
    if blocks["date"].duplicated().any():
        raise ValueError(f"G5B event rows are not unique for {pair}.")
    source_available = normalize_dates(blocks["source_available_at"])
    if source_available.isna().any() or source_available.gt(blocks["date"]).any():
        raise ValueError(f"G5B source availability is not causal for {pair}.")

    wide_columns = [column for column in source_features if column.startswith("wide__")]
    if set(WIDE_SOURCE_COLUMNS) and len(wide_columns) != len(WIDE_SOURCE_COLUMNS):
        raise ValueError(f"G5B wide-state block is incomplete for {pair}.")
    feature_cache = source_features[["date", *wide_columns]].merge(
        blocks[["date", *required_features]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    event_cache = blocks[
        [
            "date",
            "period",
            "event_state",
            "level_identity",
            "approach_state",
            "source_available_at",
            *TARGET_COLUMNS,
        ]
    ].copy()
    feature_path = feature_dir / f"{pair_stem(pair)}.parquet"
    event_path = event_dir / f"{pair_stem(pair)}.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(feature_cache, feature_path)
    atomic_write_parquet(event_cache, event_path)
    return {
        "pair": pair,
        "status": "completed",
        "source_feature_path": str(
            source["artifact_dir"] / "feature_cache" / f"{pair_stem(pair)}.parquet"
        ),
        "source_event_path": str(
            source["artifact_dir"] / "event_cache" / f"{pair_stem(pair)}.parquet"
        ),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "event_path": str(event_path),
        "event_sha256": sha256_file(event_path),
        "event_key_digest": eligibility_digest(event_cache.assign(pair=pair)[["pair", "date"]]),
        "source_event_rows": len(source_events),
        "complete_event_rows": len(event_cache),
        "incomplete_rows_dropped_without_outcome_comparison": incomplete_rows,
        "period_rows": event_cache.groupby("period").size().to_dict(),
        "feature_blocks": {
            prefix: len([column for column in feature_cache if column.startswith(f"{prefix}__")])
            for prefix in (
                "wide",
                "pre_path",
                "pre_level",
                "pre_proximity",
                "resolution",
                "contact",
                "shuffle_path",
                "shuffle_level",
                "shuffle_proximity",
                "stale_path",
                "stale_level",
                "stale_proximity",
            )
        },
        "structurally_absent_feature_blocks": [
            prefix for prefix, columns in retained_blocks.items() if not columns
        ],
        "path_source": path_contract,
        "placebo_audit": placebo_audit,
        "contact_target_maximum_absolute_reconstruction_difference": float(
            contact_difference.max(skipna=True)
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }


def build_caches(
    *,
    surface_id: str,
    surface: dict[str, Any],
    source: dict[str, Any],
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    def safe(pair: str) -> dict[str, Any]:
        try:
            return build_pair_cache(
                pair,
                surface_id=surface_id,
                surface=surface,
                source=source,
                feature_dir=feature_dir,
                event_dir=event_dir,
            )
        except Exception as exc:
            return {
                "pair": pair,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if workers == 1:
        return [safe(pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(safe, pair): pair for pair in pairs}
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda row: row["pair"])


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
    freqai["feature_parameters"]["label_period_candles"] = MAX_TARGET_HORIZON_HOURS
    freqai["data_split_parameters"] = {"test_size": 0, "shuffle": False}
    training = freqai["model_training_parameters"]
    training["n_jobs"] = 1
    training["min_child_samples"] = 10
    if technical_smoke:
        training["n_estimators"] = min(20, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_g5b"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "information_clock_contract": "profile_specific_cache_blocks",
        "missing_source_policy": "retain_nan_and_exclude_from_common_event_mask",
    }
    return config


def build_commands(
    *,
    run_id: str,
    record_dir: Path,
    artifact_dir: Path,
    surface: dict[str, Any],
    profiles: Sequence[str],
    pairs: Sequence[str],
    base_config: Path,
    python_exe: Path,
    feature_dir: Path,
    event_dir: Path,
    technical_smoke: bool,
) -> list[dict[str, Any]]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
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
                train_days=int(surface["train_days"]),
                backtest_days=int(surface["backtest_days"]),
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
            str(surface["timerange"]),
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
                "clock": definition["clock"],
                "role": definition["role"],
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
    return commands


def prepare_run(
    *,
    run_id: str,
    surface_id: str,
    pairs: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    cache_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    branch = validate_frozen_branch()
    surface = SURFACES[surface_id]
    source = load_g4d_source(surface_id, surface)
    if not set(pairs).issubset(source["pairs"]):
        raise ValueError("G5B selected pairs are not all present in the G4D source.")
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    cache_results = build_caches(
        surface_id=surface_id,
        surface=surface,
        source=source,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        workers=cache_workers,
    )
    atomic_write_parquet(DataFrame(cache_results), record_dir / "g5b_cache_inventory.parquet")
    failures = [row for row in cache_results if row["status"] == "failed"]
    if failures:
        atomic_write_json(
            {
                "run_id": run_id,
                "status": "cache_failed",
                "surface_id": surface_id,
                "failures": failures,
            },
            manifest_path,
        )
        raise ValueError(f"{len(failures)} G5B pair cache(s) failed.")
    commands = build_commands(
        run_id=run_id,
        record_dir=record_dir,
        artifact_dir=artifact_dir,
        surface=surface,
        profiles=profiles,
        pairs=pairs,
        base_config=base_config,
        python_exe=python_exe,
        feature_dir=feature_dir,
        event_dir=event_dir,
        technical_smoke=technical_smoke,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "frozen_branch_id": branch["id"],
        "surface_id": surface_id,
        "surface_role": surface["role"],
        "cohort": surface["cohort"],
        "objective": branch["hypothesis"],
        "strongest_alternative": branch["strongest_alternative"],
        "retain_interpretation": branch["retain_interpretation"],
        "park_interpretation": branch["park_interpretation"],
        "iteration_cap": int(branch["iteration_cap"]),
        "timerange": surface["timerange"],
        "validation_periods": list(surface["validation_periods"]),
        "train_period_days": int(surface["train_days"]),
        "backtest_period_days": int(surface["backtest_days"]),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "pairs": list(pairs),
        "profiles": list(profiles),
        "targets": list(TARGET_COLUMNS),
        "target_clocks": {
            CONTACT_TARGET: (
                "Contact-candle volume divided by the prior 24-candle median. Strict "
                "pre-contact profiles are forecasts; event-resolution profiles are "
                "same-candle descriptions/nowcasts."
            ),
            NEXT_VOLUME_TARGET: (
                "Next-candle volume divided by the 24-candle median known at contact close. "
                "Contact-close profiles are causal one-candle-ahead forecasts."
            ),
        },
        "clock_comparisons": [list(item) for item in CLOCK_COMPARISONS],
        "leakage_controls": {
            "same_rows_for_every_profile": True,
            "pre_contact_blocks_use_values_known_by_previous_candle_close": True,
            "event_resolution_excluded_from_strict_precontact_profiles": True,
            "contact_ohlcv_excluded_from_contact_volume_models": True,
            "contact_ohlcv_only_interpreted_for_next_candle_target": True,
            "development_outcomes_used_to_select_features": False,
            "shuffled_control_is_nonself_within_pair_period": True,
            "stale_control_uses_only_prior_event_values": True,
            "missing_stays_missing": True,
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_scorable_rows": MIN_SCORABLE_ROWS,
            "minimum_positive_coins": MIN_RETAIN_COINS,
            "required_validation_periods": list(surface["validation_periods"]),
            "bootstrap_samples": BLOCK_BOOTSTRAP_SAMPLES,
            "bootstrap_block_days": BLOCK_DAYS,
            "positive_means": (
                "lower paired absolute error for the current profile; same-candle-only "
                "gains remain descriptive"
            ),
        },
        "source_contracts": {
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "strategy_sha256": sha256_file(STRATEGY_FILE),
            "g4d_source_manifest": str(source["manifest_path"]),
            "g4d_source_manifest_sha256": source["manifest_sha256"],
            "g4d_source_result": str(source["result_path"]),
            "g4d_source_result_sha256": source["result_sha256"],
            "g5d_path_integrity": str(G5D_RECORD_ROOT / "g5d_preflight_integrity.json"),
            "g5d_path_integrity_sha256": sha256_file(
                G5D_RECORD_ROOT / "g5d_preflight_integrity.json"
            ),
            "cache_inventory": cache_results,
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "direction_prediction": False,
        "profit_optimization": False,
        "commands": commands,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def runtime_preflight(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"missing strategy: {STRATEGY_FILE}")
    if int(manifest["profile_workers"]) < 1 or int(manifest["profile_workers"]) > MAX_WORKERS:
        problems.append("profile worker count is outside 1..4")
    event_digests: set[str] = set()
    for item in manifest["source_contracts"]["cache_inventory"]:
        for key in ("feature_path", "event_path"):
            path = Path(str(item[key]))
            if not path.is_file():
                problems.append(f"missing cache: {path}")
            elif sha256_file(path) != item[f"{key.removesuffix('_path')}_sha256"]:
                problems.append(f"cache hash changed: {path}")
        event_digests.add(str(item["event_key_digest"]))
    if len(event_digests) != len(manifest["pairs"]):
        warnings.append(
            "Pair event digests are not unique; this is valid only across distinct pairs."
        )
    dependency: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": ""}
    if python_exe.is_file():
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import freqtrade, lightgbm, pyarrow; print('g5b-worker-dependencies-ok')",
            ],
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
        if result.returncode:
            problems.append("worker dependency import failed")
    free_gib = round(shutil.disk_usage(ARTIFACT_ROOT).free / (1024**3), 3)
    if free_gib < 5.0:
        warnings.append(f"Only {free_gib} GiB is free on the bulky artifact drive.")
    return {
        "created_at_utc": utc_now(),
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "worker_interpreter": str(python_exe),
        "dependency_check": dependency,
        "profiles": len(manifest["profiles"]),
        "pairs": len(manifest["pairs"]),
        "targets": len(TARGET_COLUMNS),
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "bulky_storage_free_gib": free_gib,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    items_by_id = {item["profile_id"]: item for item in manifest["commands"]}
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    processed = len(manifest["commands"]) - len(pending)
    workers = int(manifest["profile_workers"])
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        for item in batch:
            item["status"] = "running"
            item["started_at_utc"] = utc_now()
        atomic_write_json(manifest, manifest_path)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_freqai_profile, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                item = items_by_id[str(result["profile_id"])]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g5b_freqai_profiles",
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


def load_event_cache(event_dir: Path, pair: str) -> DataFrame:
    path = event_dir / f"{pair_stem(pair)}.parquet"
    frame = pd.read_parquet(path)
    frame["date"] = normalize_dates(frame["date"])
    return frame


def development_period(surface_id: str) -> str:
    source = load_manifest(Path(SURFACES[surface_id]["manifest"]))
    periods = [
        str(item["id"])
        for item in source["data"]["chronological_periods"]
        if item.get("role") == "development"
    ]
    if len(periods) != 1:
        raise ValueError(f"Expected one G5B development period; found {periods}.")
    return periods[0]


def target_band_thresholds(
    events_by_pair: dict[str, DataFrame],
    *,
    development: str,
) -> tuple[dict[tuple[str, str], tuple[float, float]], DataFrame]:
    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for pair, events in events_by_pair.items():
        selected = events.loc[events["period"].eq(development)]
        for target in TARGET_COLUMNS:
            values = pd.to_numeric(selected[target], errors="coerce").dropna()
            low = float(values.quantile(1.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
            high = (
                float(values.quantile(2.0 / 3.0))
                if len(values) >= MIN_SCORABLE_ROWS
                else np.nan
            )
            supported = bool(np.isfinite(low) and np.isfinite(high) and low < high)
            if supported:
                thresholds[(pair, target)] = (low, high)
            rows.append(
                {
                    "pair": pair,
                    "target": target,
                    "development_rows": len(values),
                    "lower_to_middle_boundary": low,
                    "middle_to_upper_boundary": high,
                    "status": "supported" if supported else "unscorable",
                    "outcome_use": "calibration_bands_only_not_feature_selection",
                }
            )
    return thresholds, DataFrame(rows)


def assign_bands(
    frame: DataFrame,
    *,
    thresholds: dict[tuple[str, str], tuple[float, float]],
) -> DataFrame:
    output = frame.copy()
    output["actual_band"] = np.nan
    output["prediction_band"] = np.nan
    for (pair, target), (low, high) in thresholds.items():
        selected = output["pair"].eq(pair) & output["target"].eq(target)
        for source, destination in (
            ("actual", "actual_band"),
            ("prediction", "prediction_band"),
        ):
            values = pd.to_numeric(output.loc[selected, source], errors="coerce")
            output.loc[selected, destination] = np.select(
                (values.le(low), values.le(high)),
                (0.0, 1.0),
                default=2.0,
            )
    return output


def prediction_event_surface(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
) -> tuple[DataFrame, dict[str, Any], DataFrame]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    predictions: dict[str, DataFrame] = {}
    file_audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        if item.get("status") != "completed":
            continue
        profile_id = str(item["profile_id"])
        frame, audit = load_predictions(Path(item["model_dir"]), pairs)
        audit["profile_id"] = profile_id
        file_audits.append(audit)
        if frame.empty:
            raise ValueError(f"No G5B FreqAI predictions for {profile_id}.")
        missing = sorted(set(TARGET_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"G5B profile {profile_id} lacks targets: {missing}")
        predictions[profile_id] = frame
    if not predictions:
        raise ValueError("No completed G5B prediction profiles are available.")
    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError("G5B profiles have no common prediction keys.")
    atomic_write_parquet(eligibility, record_dir / "g5b_prediction_eligibility.parquet")
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    events_by_pair = {pair: load_event_cache(event_dir, pair) for pair in pairs}
    thresholds, threshold_frame = target_band_thresholds(
        events_by_pair,
        development=development_period(str(manifest["surface_id"])),
    )
    validation = set(str(value) for value in manifest["validation_periods"])
    rows: list[DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        fair = frame.merge(
            common_keys,
            on=["pair", "date"],
            how="inner",
            validate="one_to_one",
        )
        for pair in pairs:
            events = events_by_pair[pair]
            events = events.loc[events["period"].isin(validation)].copy()
            pair_predictions = fair.loc[fair["pair"].eq(pair)]
            merged = events.merge(
                pair_predictions[["date", *TARGET_COLUMNS]],
                on="date",
                how="inner",
                suffixes=("_actual", "_prediction"),
                validate="one_to_one",
            )
            coverage.append(
                {
                    "profile_id": profile_id,
                    "pair": pair,
                    "eligible_validation_events": len(events),
                    "common_prediction_events": len(merged),
                    "excluded_validation_events": len(events) - len(merged),
                }
            )
            for target in TARGET_COLUMNS:
                block = merged[
                    ["date", "period", f"{target}_actual", f"{target}_prediction"]
                ].rename(
                    columns={
                        f"{target}_actual": "actual",
                        f"{target}_prediction": "prediction",
                    }
                )
                block["profile_id"] = profile_id
                block["pair"] = pair
                block["target"] = target
                rows.append(block)
    output = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    output = assign_bands(output, thresholds=thresholds)
    if output[["prediction", "actual"]].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError("G5B common event predictions contain missing values.")
    duplicates = int(output.duplicated(["profile_id", "pair", "date", "target"]).sum())
    if duplicates:
        raise ValueError(f"G5B scoring surface has {duplicates} duplicate keys.")
    audit = {
        "created_at_utc": utc_now(),
        "profile_files": file_audits,
        "common_prediction_rows": len(common_keys),
        "common_prediction_key_digest": eligibility_digest(common_keys),
        "common_event_prediction_rows": len(output),
        "event_coverage": coverage,
        "development_band_thresholds_used_only_for_calibration": True,
    }
    return output, audit, threshold_frame


def score_scopes(manifest: dict[str, Any], predictions: DataFrame) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    primary = (
        tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        if manifest["cohort"] == "normal"
        else pairs
    )
    scopes: dict[str, tuple[str, ...]] = {
        "primary_cohort": primary,
        **{f"pair::{pair}": (pair,) for pair in pairs},
    }
    if manifest["cohort"] == "normal":
        scopes["btc_only"] = tuple(pair for pair in pairs if pair.startswith("BTC/"))
    rows: list[dict[str, Any]] = []
    for scope, members in scopes.items():
        selected = predictions.loc[predictions["pair"].isin(members)]
        scope_type = (
            "pair"
            if scope.startswith("pair::")
            else "btc"
            if scope == "btc_only"
            else "cohort"
        )
        for (profile_id, period, target), frame in selected.groupby(
            ["profile_id", "period", "target"],
            observed=True,
            sort=False,
        ):
            rows.append(
                {
                    "scope": scope,
                    "scope_type": scope_type,
                    "member_pairs": json.dumps(list(members)),
                    "member_pair_count": len(members),
                    "profile_id": profile_id,
                    "period": period,
                    "target": target,
                    **regression_metrics(frame),
                }
            )
    return DataFrame(rows)


def deterministic_block_bootstrap(
    paired: DataFrame,
    *,
    seed_key: str,
    samples: int = BLOCK_BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float]:
    if paired.empty:
        return np.nan, np.nan, np.nan
    block = paired.copy()
    block["week"] = pd.to_datetime(block["date"], utc=True).dt.floor(f"{BLOCK_DAYS}D")
    block_means = (
        block.groupby(["pair", "week"], observed=True)["paired_error_gain"]
        .agg(block_mean="mean", block_rows="size")
        .reset_index()
    )
    coin_point = block.groupby("pair", observed=True)["paired_error_gain"].mean()
    point = float(coin_point.mean())
    if len(block_means) < 2:
        return point, np.nan, np.nan
    seed = int.from_bytes(
        __import__("hashlib").sha256(seed_key.encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    values: list[float] = []
    groups = {
        pair: (
            group["block_mean"].to_numpy(dtype=float),
            group["block_rows"].to_numpy(dtype=float),
        )
        for pair, group in block_means.groupby("pair", observed=True)
    }
    for _ in range(samples):
        coin_values = []
        for means, row_counts in groups.values():
            positions = rng.integers(0, len(means), len(means))
            sampled_means = means[positions]
            sampled_rows = row_counts[positions]
            coin_values.append(
                float(np.average(sampled_means, weights=sampled_rows))
            )
        values.append(float(np.mean(coin_values)))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_profile_comparisons(
    manifest: dict[str, Any],
    predictions: DataFrame,
    scores: DataFrame,
) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    primary = (
        tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        if manifest["cohort"] == "normal"
        else pairs
    )
    rows: list[dict[str, Any]] = []
    for comparison_id, target, profile, control, interpretation in CLOCK_COMPARISONS:
        selected = predictions.loc[
            predictions["target"].eq(target)
            & predictions["profile_id"].isin([profile, control])
            & predictions["pair"].isin(primary)
        ].copy()
        pivot = selected.pivot(
            index=["pair", "date", "period", "actual"],
            columns="profile_id",
            values="prediction",
        ).reset_index()
        if profile not in pivot or control not in pivot:
            continue
        pivot["paired_error_gain"] = (
            (pivot[control] - pivot["actual"]).abs()
            - (pivot[profile] - pivot["actual"]).abs()
        )
        for period in manifest["validation_periods"]:
            frame = pivot.loc[pivot["period"].eq(period)].copy()
            point, lower, upper = deterministic_block_bootstrap(
                frame,
                seed_key=f"{manifest['run_id']}|{comparison_id}|{period}",
            )
            coin = frame.groupby("pair", observed=True)["paired_error_gain"].mean()
            loo = [float(coin.drop(index=pair).mean()) for pair in coin.index if len(coin) > 1]
            control_mae = float(
                frame.assign(
                    control_absolute_error=(frame[control] - frame["actual"]).abs()
                )
                .groupby("pair", observed=True)["control_absolute_error"]
                .mean()
                .mean()
            )
            profile_score = scores.loc[
                scores["scope"].eq("primary_cohort")
                & scores["profile_id"].eq(profile)
                & scores["period"].eq(period)
                & scores["target"].eq(target)
            ]
            control_score = scores.loc[
                scores["scope"].eq("primary_cohort")
                & scores["profile_id"].eq(control)
                & scores["period"].eq(period)
                & scores["target"].eq(target)
            ]
            supported = (
                len(frame) >= MIN_SCORABLE_ROWS
                and len(coin) >= MIN_RETAIN_COINS
                and len(profile_score) == 1
                and len(control_score) == 1
                and profile_score.iloc[0]["status"] == "scored"
                and control_score.iloc[0]["status"] == "scored"
            )
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "interpretation_clock": interpretation,
                    "surface_id": manifest["surface_id"],
                    "cohort": manifest["cohort"],
                    "period": period,
                    "target": target,
                    "profile_id": profile,
                    "control_profile_id": control,
                    "rows": len(frame),
                    "coins": len(coin),
                    "supported": supported,
                    "equal_coin_paired_mae_gain": point,
                    "paired_mae_gain_percent_of_control": (
                        100.0 * point / control_mae
                        if np.isfinite(point) and control_mae > 0.0
                        else np.nan
                    ),
                    "block_bootstrap_lower_95": lower,
                    "block_bootstrap_upper_95": upper,
                    "positive_coin_count": int(coin.gt(0.0).sum()),
                    "scoreable_coin_count": len(coin),
                    "leave_one_coin_out_minimum_gain": min(loo) if loo else np.nan,
                    "leave_one_coin_out_positive": bool(loo and min(loo) > 0.0),
                    "spearman_change": (
                        float(
                            profile_score.iloc[0]["prediction_actual_spearman"]
                            - control_score.iloc[0]["prediction_actual_spearman"]
                        )
                        if supported
                        else np.nan
                    ),
                    "profile_calibration_slope": (
                        float(profile_score.iloc[0]["calibration_slope"])
                        if supported
                        else np.nan
                    ),
                    "profile_band_rows": (
                        int(profile_score.iloc[0]["band_rows"]) if supported else 0
                    ),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def comparison_passed(row: pd.Series) -> bool:
    return bool(
        row["supported"]
        and row["equal_coin_paired_mae_gain"] > 0.0
        and row["positive_coin_count"] >= MIN_RETAIN_COINS
        and row["leave_one_coin_out_positive"]
        and row["spearman_change"] >= 0.0
        and np.isfinite(row["profile_calibration_slope"])
        and row["profile_calibration_slope"] > 0.0
        and row["profile_band_rows"] >= MIN_SCORABLE_ROWS
    )


def retention_decisions(manifest: dict[str, Any], comparisons: DataFrame) -> DataFrame:
    routes = {
        "strict_precontact_contact_volume_forecast": (
            "strict_precontact_contact_volume_vs_state",
            "strict_precontact_contact_volume_vs_shuffle",
            "strict_precontact_contact_volume_vs_stale",
        ),
        "contact_close_volume_nowcast": ("contact_close_nowcast_increment",),
        "contact_close_next_volume_forecast": (
            "contact_close_next_volume_vs_state",
            "contact_close_next_volume_vs_shuffle",
            "contact_close_next_volume_vs_stale",
        ),
    }
    rows: list[dict[str, Any]] = []
    for route, required in routes.items():
        period_checks: list[dict[str, Any]] = []
        complete = True
        repeated = True
        for period in manifest["validation_periods"]:
            selected = comparisons.loc[
                comparisons["comparison_id"].isin(required)
                & comparisons["period"].eq(period)
            ]
            present = set(selected["comparison_id"]) == set(required)
            passed = bool(present and selected.apply(comparison_passed, axis=1).all())
            complete &= present
            repeated &= passed
            period_checks.append(
                {
                    "period": period,
                    "required_comparisons": list(required),
                    "present_comparisons": sorted(selected["comparison_id"].tolist()),
                    "comparison_passes": {
                        str(row.comparison_id): comparison_passed(pd.Series(row._asdict()))
                        for row in selected.itertuples(index=False)
                    },
                    "passed": passed,
                }
            )
        if manifest.get("technical_smoke_not_evidence"):
            status = "technical_smoke_not_evidence"
        elif not complete:
            status = "parked_insufficient_common_support"
        elif repeated and route == "contact_close_volume_nowcast":
            status = "retained_descriptive_nowcast_only"
        elif repeated:
            status = "retained_causal_volume_lead"
        else:
            status = "parked_not_repeated_beyond_controls"
        rows.append(
            {
                "surface_id": manifest["surface_id"],
                "surface_role": manifest["surface_role"],
                "cohort": manifest["cohort"],
                "route": route,
                "status": status,
                "period_checks": json.dumps(period_checks, sort_keys=True),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def score_run(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    predictions, audit, thresholds = prediction_event_surface(
        manifest,
        record_dir=record_dir,
    )
    scores = score_scopes(manifest, predictions)
    comparisons = paired_profile_comparisons(manifest, predictions, scores)
    decisions = retention_decisions(manifest, comparisons)
    prediction_path = artifact_dir / "g5b_common_event_predictions.parquet"
    atomic_write_parquet(predictions, prediction_path)
    atomic_write_parquet(scores, record_dir / "g5b_regression_scores.parquet")
    atomic_write_parquet(comparisons, record_dir / "g5b_paired_profile_comparisons.parquet")
    atomic_write_parquet(decisions, record_dir / "g5b_surface_decisions.parquet")
    atomic_write_parquet(thresholds, record_dir / "g5b_development_band_thresholds.parquet")
    atomic_write_json(audit, record_dir / "g5b_prediction_audit.json")
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "surface_id": manifest["surface_id"],
        "status": "completed",
        "completed_at_utc": utc_now(),
        "profiles": int(predictions["profile_id"].nunique()),
        "pairs": int(predictions["pair"].nunique()),
        "targets": int(predictions["target"].nunique()),
        "prediction_rows": len(predictions),
        "paired_comparison_rows": len(comparisons),
        "decision_counts": decisions["status"].value_counts().to_dict(),
        "prediction_path": str(prediction_path),
        "interpretation": (
            "Strict-precontact and next-candle routes are causal timing tests. Any "
            "contact-volume gain that appears only after event-resolution geometry is "
            "added is a description/nowcast, not a pre-contact forecast."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "g5b_result.json")
    return result


def batch_review(run_ids: Sequence[str], *, review_id: str) -> dict[str, Any]:
    frames: list[DataFrame] = []
    sources: list[dict[str, Any]] = []
    for run_id in run_ids:
        record_dir = RECORD_ROOT / run_id
        manifest_path = record_dir / "manifest.json"
        decision_path = record_dir / "g5b_surface_decisions.parquet"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or not decision_path.is_file():
            raise ValueError(f"Incomplete G5B review source: {run_id}")
        frame = pd.read_parquet(decision_path)
        frame["run_id"] = run_id
        frames.append(frame)
        sources.append(
            {
                "run_id": run_id,
                "manifest_sha256": sha256_file(manifest_path),
                "decision_sha256": sha256_file(decision_path),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    expected = set(SURFACES)
    present = set(combined["surface_id"])
    if present != expected:
        raise ValueError(f"G5B batch review needs all surfaces: missing={expected - present}")
    review_dir = RECORD_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    combined_path = review_dir / "g5b_all_surface_decisions.parquet"
    atomic_write_parquet(combined, combined_path)
    causal = combined.loc[combined["status"].eq("retained_causal_volume_lead")]
    nowcast = combined.loc[combined["status"].eq("retained_descriptive_nowcast_only")]
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "review_id": review_id,
        "created_at_utc": utc_now(),
        "status": (
            "retained_causal_volume_leads"
            if not causal.empty
            else "parked_causal_volume_prediction"
        ),
        "source_runs": sources,
        "decision_counts": combined["status"].value_counts().to_dict(),
        "causal_leads": causal.to_dict(orient="records"),
        "descriptive_nowcasts": nowcast.to_dict(orient="records"),
        "interpretation": (
            "A causal lead requires repetition beyond state, shuffled, and stale controls. "
            "A same-contact-candle gain is reported separately as a nowcast."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
        "combined_decisions": str(combined_path),
    }
    atomic_write_json(result, review_dir / "g5b_batch_review.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Generation 5B FreqAI regression tests separating strict pre-contact, "
            "contact-close nowcast, and next-candle volume clocks."
        )
    )
    parser.add_argument("--surface", choices=tuple(SURFACES))
    parser.add_argument("--run-id")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-review-run-ids")
    parser.add_argument("--batch-review-id")
    args = parser.parse_args(argv)
    if args.batch_review_run_ids:
        if not args.batch_review_id:
            raise ValueError("--batch-review-id is required.")
        result = batch_review(
            parse_csv(args.batch_review_run_ids),
            review_id=args.batch_review_id,
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if not args.surface or not args.run_id:
        raise ValueError("--surface and --run-id are required.")
    for label, value in (
        ("profile-workers", args.profile_workers),
        ("cache-workers", args.cache_workers),
    ):
        if value < 1 or value > MAX_WORKERS:
            raise ValueError(f"{label} must be in 1..{MAX_WORKERS}; received {value}.")
    surface = SURFACES[args.surface]
    source = load_g4d_source(args.surface, surface)
    pairs = select_pairs(source["pairs"], args.pairs)
    profiles = tuple(PROFILES) if args.profiles == "all" else parse_csv(args.profiles)
    unknown = sorted(set(profiles).difference(PROFILES))
    if not profiles or unknown:
        raise ValueError(f"Invalid G5B profiles: {unknown}")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    manifest, manifest_path, artifact_dir = prepare_run(
        run_id=args.run_id,
        surface_id=args.surface,
        pairs=pairs,
        profiles=profiles,
        base_config=args.base_config,
        python_exe=args.python,
        profile_workers=args.profile_workers,
        cache_workers=args.cache_workers,
        technical_smoke=args.technical_smoke,
    )
    record_dir = manifest_path.parent
    preflight = runtime_preflight(manifest, python_exe=args.python)
    atomic_write_json(preflight, record_dir / "g5b_launch_preflight.json")
    manifest["launch_preflight"] = preflight
    atomic_write_json(manifest, manifest_path)
    if not preflight["passed"]:
        manifest["status"] = "preflight_failed"
        atomic_write_json(manifest, manifest_path)
        return 2
    if args.prepare_only:
        return 0
    if args.retry_failed:
        for item in manifest["commands"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
        manifest.pop("failed_profile_ids", None)
        manifest["status"] = "prepared"
        atomic_write_json(manifest, manifest_path)
    elif manifest.get("status") in {"failed", "preflight_failed"}:
        raise ValueError("Use --retry-failed after diagnosing a failed G5B profile.")
    if not args.score_only and manifest.get("status") != "completed":
        returncode = run_manifest(manifest, manifest_path)
        if returncode:
            return returncode
    incomplete = [
        item["profile_id"]
        for item in manifest["commands"]
        if item.get("status") != "completed"
    ]
    if incomplete:
        raise ValueError(f"Cannot score incomplete G5B profiles: {incomplete}")
    result = score_run(manifest, record_dir=record_dir, artifact_dir=artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result"] = result
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
