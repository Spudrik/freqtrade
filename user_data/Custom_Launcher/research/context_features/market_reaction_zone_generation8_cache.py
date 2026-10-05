from __future__ import annotations

# Bound numerical libraries before importing pandas/pyarrow.
# ruff: noqa: E402, E501
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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
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
    market_reaction_zone_generation3_external_context as g3h,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_preflight as g8p,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration6Strategy import (
    TARGET_COLUMNS,
)


OUTPUT_ROOT = g8p.OUTPUT_ROOT
PREFLIGHT_MANIFEST = (
    OUTPUT_ROOT
    / "generation8_branches"
    / "g8_preflight"
    / g8p.DEFAULT_RUN_ID
    / "manifest.json"
)
EVENT_MANIFEST = (
    OUTPUT_ROOT
    / "generation6_branches"
    / "g6_common_event_cache"
    / "g6_common_event_full_20260821a"
    / "manifest.json"
)
SHARED_CACHE_ID = "g8_attribution_cache_20260821a"
RECORD_ROOT = OUTPUT_ROOT / "generation8_shared" / SHARED_CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation8_shared"
    / SHARED_CACHE_ID
)
MAX_WORKERS = 4
STALE_HOURS = 72

DEPENDENCY_GROUP = {
    "confirmed_swing": "confirmed_swing",
    "generic_prior_range": "prior_range",
    "generic_round_number": "round_number",
    "tlv2_forecast_zone": "trendline",
    "tlv2_ranked": "trendline",
    "volume_profile_explicit_prior": "volume_profile",
    "volume_profile_nodes": "volume_profile",
    "volume_profile_settled": "volume_profile",
}

G7_COPY_BLOCKS: dict[str, tuple[str, ...]] = {
    "g8_level_base": tuple(
        column.removeprefix("level__")
        for column in g8p.PREDICTOR_GROUPS["level_summary"]
    ),
    "g8_participation_relative_volume": ("relative_volume",),
    "g8_participation_volume_acceleration": ("volume_acceleration_magnitude",),
    "g8_participation_absolute_pressure": ("absolute_pressure",),
    "g8_participation_pressure_persistence": ("pressure_persistence",),
    "g8_volatility_absolute": tuple(
        column.removeprefix("ohlcv_volatility_range__")
        for column in g8p.PREDICTOR_GROUPS["absolute_volatility"]
    ),
    "g8_volatility_compression": tuple(
        column.removeprefix("ohlcv_volatility_range__")
        for column in g8p.PREDICTOR_GROUPS["compression"]
    ),
    "g8_trend_persistent": tuple(
        column.removeprefix("ohlcv_trend_momentum__")
        for column in g8p.PREDICTOR_GROUPS["persistent_trend"]
    ),
    "g8_trend_acceleration": tuple(
        column.removeprefix("ohlcv_trend_momentum__")
        for column in g8p.PREDICTOR_GROUPS["return_acceleration"]
    ),
    "g8_momentum_oscillator": tuple(
        column.removeprefix("ohlcv_trend_momentum__")
        for column in g8p.PREDICTOR_GROUPS["oscillator_displacement"]
    ),
    "g8_btc_horizon_1h": ("btc_activity_1h",),
    "g8_btc_horizon_4h": ("btc_activity_4h",),
    "g8_btc_horizon_24h": ("btc_activity_24h",),
    "g8_btc_relative_volume": ("btc_relative_volume",),
}

G7_SOURCE_PREFIX = {
    "g8_level_base": "level",
    "g8_participation_relative_volume": "ohlcv_volume_pressure",
    "g8_participation_volume_acceleration": "ohlcv_volume_pressure",
    "g8_participation_absolute_pressure": "ohlcv_volume_pressure",
    "g8_participation_pressure_persistence": "ohlcv_volume_pressure",
    "g8_volatility_absolute": "ohlcv_volatility_range",
    "g8_volatility_compression": "ohlcv_volatility_range",
    "g8_trend_persistent": "ohlcv_trend_momentum",
    "g8_trend_acceleration": "ohlcv_trend_momentum",
    "g8_momentum_oscillator": "ohlcv_trend_momentum",
    "g8_btc_horizon_1h": "cross_market_btc",
    "g8_btc_horizon_4h": "cross_market_btc",
    "g8_btc_horizon_24h": "cross_market_btc",
    "g8_btc_relative_volume": "cross_market_btc",
}

TIMEFRAME_BLOCKS = {
    f"g8_timeframe_{timeframe}": tuple(
        column.removeprefix(f"timeframe_{timeframe}__")
        for column in g8p.PREDICTOR_GROUPS["timeframe_relationships"]
        if column.startswith(f"timeframe_{timeframe}__")
    )
    for timeframe in ("4h", "8h", "1d")
}

DERIVED_LEVEL_BLOCKS = (
    "g8_level_identity",
    "g8_level_prominence",
    "g8_level_width",
    "g8_level_proximity",
    "g8_level_timeframe",
    "g8_level_history",
    "g8_touch_first",
    "g8_touch_occupancy",
    "g8_touch_retest",
    "g8_room_geometry",
)


@dataclass(frozen=True)
class CacheTask:
    cohort: str
    pair: str
    source_manifest: Path
    g7_item: dict[str, Any]
    feature_dir: Path
    support_dir: Path


def cell_ready_name(cell: g8p.CellSpec) -> str:
    branch = cell.branch_id.split("_", maxsplit=1)[0]
    surface = "" if cell.surface == "main" else f"_{cell.surface}"
    return f"{branch}_{cell.cohort}{surface}"


def cell_blocks(cell: g8p.CellSpec) -> tuple[str, ...]:
    branch = cell.branch_id.split("_", maxsplit=1)[0]
    participation = (
        "g8_participation_relative_volume",
        "g8_participation_volume_acceleration",
        "g8_participation_absolute_pressure",
        "g8_participation_pressure_persistence",
    )
    base = ("g8_level_base",)
    if branch == "g8a":
        return (
            *base,
            *participation,
            "g8_volatility_absolute",
            "g8_volatility_compression",
            "g8_level_identity",
            "g8_level_prominence",
            "g8_level_width",
            "g8_level_proximity",
            "g8_level_timeframe",
            "g8_level_history",
        )
    if branch == "g8b":
        return (
            *base,
            *participation,
            "g8_participation_duration",
            "g8_volatility_absolute",
            "g8_volatility_compression",
        )
    if branch == "g8c":
        return (
            *base,
            *participation,
            "g8_volatility_absolute",
            "g8_volatility_compression",
            "g8_volatility_duration",
        )
    if branch == "g8d":
        return (
            *base,
            *participation,
            "g8_trend_persistent",
            "g8_trend_acceleration",
            "g8_momentum_oscillator",
            "g8_trend_duration",
        )
    if branch == "g8e":
        return (
            *base,
            *participation,
            "g8_btc_horizon_1h",
            "g8_btc_horizon_4h",
            "g8_btc_horizon_24h",
            "g8_btc_relative_volume",
            "g8_btc_duration",
        )
    if branch == "g8f":
        return (
            *base,
            *participation,
            "g8_timeframe_4h",
            "g8_timeframe_8h",
            "g8_timeframe_1d",
            "g8_room_geometry",
        )
    if branch == "g8g":
        return (
            *base,
            "g8_btc_horizon_1h",
            "g8_btc_horizon_4h",
            "g8_btc_horizon_24h",
            "g8_btc_relative_volume",
            "g8_orderbook_pressure",
            "g8_orderbook_coverage",
            "g8_orderbook_regime",
            "g8_orderbook_persistence",
        )
    if branch == "g8h":
        context = {
            "participation_volatility": (
                "g8_volatility_absolute",
                "g8_volatility_compression",
            ),
            "participation_trend": (
                "g8_trend_persistent",
                "g8_trend_acceleration",
                "g8_momentum_oscillator",
            ),
            "participation_btc": (
                "g8_btc_horizon_1h",
                "g8_btc_horizon_4h",
                "g8_btc_horizon_24h",
                "g8_btc_relative_volume",
            ),
        }[cell.surface]
        return (
            *base,
            *participation,
            *context,
            "g8_touch_first",
            "g8_touch_occupancy",
            "g8_touch_retest",
            "g8_level_history",
        )
    raise ValueError(f"Unknown Generation 8 branch {branch!r}")


def load_preflight() -> dict[str, Any]:
    if not PREFLIGHT_MANIFEST.is_file():
        raise FileNotFoundError(PREFLIGHT_MANIFEST)
    manifest = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_all_generation8_siblings_supported_outcome_blind":
        raise ValueError("Generation 8 outcome-blind preflight is not fully supported.")
    sequence = manifest.get("outcome_blind_sequence", {})
    if sequence.get("reaction_outcome_columns_read") is not False:
        raise ValueError("Generation 8 preflight opened reaction outcomes.")
    return manifest


def load_g7_manifest(cohort: str) -> dict[str, Any]:
    return g8p.load_g7_manifest(cohort)


@cache
def event_manifest() -> dict[str, Any]:
    return json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))


@cache
def frozen_bins() -> dict[tuple[str, str], tuple[float, float]]:
    preflight = load_preflight()
    path = Path(preflight["artifacts"]["frozen_adjacent_predictor_ranges"])
    frame = pd.read_csv(path)
    frame = frame.loc[frame["status"].eq("frozen_adjacent_tertiles")]
    return {
        (str(row.cell_id), str(row.predictor)): (
            float(row.lower_boundary),
            float(row.upper_boundary),
        )
        for row in frame.itertuples(index=False)
    }


def split_g7_blocks(features: DataFrame) -> DataFrame:
    output = DataFrame({"date": pd.to_datetime(features["date"], utc=True)})
    for target_block, suffixes in G7_COPY_BLOCKS.items():
        source = G7_SOURCE_PREFIX[target_block]
        for suffix in suffixes:
            source_column = f"{source}__{suffix}"
            if source_column not in features:
                raise ValueError(f"Missing Generation 7 feature {source_column!r}")
            output[f"{target_block}__{suffix}"] = pd.to_numeric(
                features[source_column], errors="coerce"
            )
    for target_block, suffixes in TIMEFRAME_BLOCKS.items():
        source = target_block.removeprefix("g8_")
        for suffix in suffixes:
            source_column = f"{source}__{suffix}"
            if source_column not in features:
                raise ValueError(f"Missing Generation 7 feature {source_column!r}")
            output[f"{target_block}__{suffix}"] = pd.to_numeric(
                features[source_column], errors="coerce"
            )
    return output


def numeric_array(values: Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def run_length(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    changed = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        changed[1:] = (~finite[1:]) | (~finite[:-1]) | (values[1:] != values[:-1])
    groups = np.cumsum(changed)
    result = (
        Series(np.arange(len(values), dtype=np.int64))
        .groupby(groups, sort=False)
        .cumcount()
        .add(1)
        .to_numpy(dtype=np.float64)
    )
    result[~finite] = np.nan
    return result


def band_and_duration(
    values: Series, *, lower: float, upper: float
) -> tuple[np.ndarray, np.ndarray]:
    numeric = numeric_array(values)
    band = np.full(len(numeric), np.nan, dtype=np.float64)
    finite = np.isfinite(numeric)
    band[finite] = 0.0
    band[finite & (numeric <= lower)] = -1.0
    band[finite & (numeric >= upper)] = 1.0
    return band, run_length(band)


def cell_for_branch(cohort: str, branch: str, surface: str = "main") -> g8p.CellSpec:
    return next(
        cell
        for cell in g8p.CELLS
        if cell.cohort == cohort
        and cell.branch_id.startswith(f"{branch}_")
        and cell.surface == surface
    )


def duration_block(
    *,
    block: str,
    cell: g8p.CellSpec,
    timeline: DataFrame,
    source: DataFrame,
    definitions: dict[str, tuple[str, Callable[[Series], Series]]],
    predictor_prefix: str,
) -> DataFrame:
    continuous = DataFrame({"date": pd.to_datetime(source["date"], utc=True)})
    thresholds = frozen_bins()
    for feature, (source_column, transform) in definitions.items():
        predictor = f"{predictor_prefix}__{feature}"
        bounds = thresholds.get((cell.cell_id, predictor))
        if bounds is None:
            continue
        values = transform(pd.to_numeric(source[source_column], errors="coerce"))
        band, duration = band_and_duration(values, lower=bounds[0], upper=bounds[1])
        continuous[f"{block}__{feature}_band"] = band
        continuous[f"{block}__{feature}_band_duration_hours"] = duration
    if len(continuous.columns) == 1:
        raise ValueError(f"No frozen duration features for {cell.cell_id} and {block}")
    return timeline[["date"]].merge(
        continuous, on="date", how="left", validate="one_to_one"
    )


def continuous_state(base: DataFrame) -> DataFrame:
    state, _ = g6.causal_state_features(base)
    output = DataFrame({"date": pd.to_datetime(base["date"], utc=True)})
    for source, _transform in g6d.STATE_FEATURES.values():
        raw = source.removeprefix("state__")
        output[source] = pd.to_numeric(state[raw], errors="coerce")
    return output


def continuous_cross_market(cohort: str) -> DataFrame:
    key = f"{cohort}_cross_market"
    path = Path(event_manifest()["shared_features"][key]["path"])
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    columns = [
        g6d.CROSS_MARKET_FEATURES[name][0]
        for name in (
            "btc_activity_1h",
            "btc_activity_4h",
            "btc_activity_24h",
            "btc_relative_volume",
        )
    ]
    return frame[["date", *dict.fromkeys(columns)]].drop_duplicates("date", keep="last")


@cache
def orderbook_surface() -> DataFrame:
    causal = g3h.load_causal_pressure_surface(g8p.HISTORICAL_ORDERBOOK)
    full_columns = list(dict.fromkeys(g8p.ORDERBOOK_REQUIRED_COLUMNS))
    full = pd.read_parquet(g8p.HISTORICAL_ORDERBOOK, columns=full_columns)
    full["date"] = pd.to_datetime(full["date"], utc=True, errors="coerce")
    full["source_max_ts"] = pd.to_datetime(
        full["source_max_ts"], utc=True, errors="coerce"
    )
    keep = [
        "date",
        "source_max_ts",
        "source_row_usable",
        "absolute_pressure",
        "pressure_lower_tertile",
        "pressure_upper_tertile",
        "pressure_regime",
    ]
    output = full.merge(causal[keep], on=["date", "source_max_ts"], how="left")
    return output.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def orderbook_blocks(timeline: DataFrame) -> DataFrame:
    source = orderbook_surface()
    frame = timeline[["date"]].merge(source, on="date", how="left", validate="one_to_one")
    safe = frame["source_row_usable"].fillna(False).astype(bool)
    age = (
        pd.to_datetime(frame["date"], utc=True)
        - pd.to_datetime(frame["source_max_ts"], utc=True)
    ).dt.total_seconds().div(3600.0)
    output = DataFrame({"date": frame["date"]})
    pressure = {
        "absolute_pressure": frame["absolute_pressure"],
        "absolute_last_pressure": pd.to_numeric(
            frame["obts_pressure_25bps_last"], errors="coerce"
        ).abs(),
        "absolute_pressure_slope": pd.to_numeric(
            frame["obts_pressure_25bps_slope"], errors="coerce"
        ).abs(),
        "absolute_pressure_acceleration": pd.to_numeric(
            frame["obts_pressure_acceleration"], errors="coerce"
        ).abs(),
        "absolute_flip_strength": pd.to_numeric(
            frame["obts_pressure_flip_strength"], errors="coerce"
        ).abs(),
    }
    for name, values in pressure.items():
        output[f"g8_orderbook_pressure__{name}"] = pd.to_numeric(
            values, errors="coerce"
        ).where(safe)
    output["g8_orderbook_coverage__coverage_ratio"] = pd.to_numeric(
        frame["obts_coverage_ratio"], errors="coerce"
    ).where(safe)
    output["g8_orderbook_coverage__source_age_hours"] = age.where(safe)
    output["g8_orderbook_coverage__source_ready"] = safe.astype(float)
    regime = frame["pressure_regime"].astype("string")
    for name in ("quiet", "middle", "active"):
        output[f"g8_orderbook_regime__{name}"] = regime.eq(name).astype(float).where(safe)
    output["g8_orderbook_persistence__pressure_duration_hours"] = pd.to_numeric(
        frame["obts_pressure_duration_hours"], errors="coerce"
    ).where(safe)
    output["g8_orderbook_persistence__flip_count_1h"] = pd.to_numeric(
        frame["obts_pressure_flip_count_1h"], errors="coerce"
    ).where(safe)
    return output


def update_sum(
    total: np.ndarray, count: np.ndarray, values: np.ndarray, mask: np.ndarray
) -> None:
    finite = mask & np.isfinite(values)
    total[finite] += values[finite]
    count[finite] += 1.0


def safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)


def segment_history(
    *,
    valid: np.ndarray,
    contact: np.ndarray,
    level: np.ndarray,
    width: np.ndarray,
    explicit_identity: Series | None,
) -> dict[str, np.ndarray]:
    previous_valid = np.concatenate(([False], valid[:-1]))
    moved = np.zeros(len(valid), dtype=bool)
    moved[1:] = (
        valid[1:]
        & valid[:-1]
        & (np.abs(level[1:] - level[:-1]) > width[1:])
    )
    if explicit_identity is None:
        new_segment = valid & (~previous_valid | moved)
    else:
        identity = explicit_identity.astype("string")
        present = identity.notna().to_numpy()
        prior_present = np.concatenate(([False], present[:-1]))
        same = np.zeros(len(valid), dtype=bool)
        same[1:] = (
            present[1:]
            & present[:-1]
            & identity.iloc[1:].reset_index(drop=True).eq(
                identity.iloc[:-1].reset_index(drop=True)
            ).to_numpy()
        )
        use_explicit = present & prior_present
        new_segment = valid & (
            ~previous_valid | (use_explicit & ~same) | (~use_explicit & moved)
        )
    segment = np.cumsum(new_segment.astype(np.int64))
    segment_start = np.maximum.accumulate(
        np.where(new_segment, np.arange(len(valid), dtype=np.int64), 0)
    )
    episode_start = g0.episode_start_mask(contact, level, width, cooldown=6)
    cumulative = np.cumsum(episode_start.astype(np.int64))
    prior_global = np.concatenate(([0], cumulative[:-1]))
    segment_base = np.maximum.accumulate(np.where(new_segment, prior_global, -1))
    prior_episodes = np.maximum(prior_global - segment_base, 0).astype(np.float64)
    previous_contact = np.concatenate(([False], contact[:-1]))
    previous_segment = np.concatenate(([-1], segment[:-1]))
    occupancy = contact & previous_contact & (segment == previous_segment)
    first = contact & (prior_episodes == 0.0)
    first_retest = contact & ~occupancy & (prior_episodes == 1.0)
    later_retest = contact & ~occupancy & (prior_episodes >= 2.0)
    index = np.arange(len(valid), dtype=np.float64)
    last_contact = (
        Series(np.where(contact, index, np.nan))
        .groupby(segment, sort=False)
        .ffill()
        .shift(1)
        .to_numpy(dtype=np.float64, copy=True)
    )
    same_segment = segment == previous_segment
    last_contact[~same_segment] = np.nan
    since_contact = index - last_contact
    persistence = index - segment_start.astype(np.float64)
    persistence[~valid] = np.nan
    return {
        "first": first.astype(float),
        "occupancy": occupancy.astype(float),
        "first_retest": first_retest.astype(float),
        "later_retest": later_retest.astype(float),
        "prior_episodes": prior_episodes,
        "time_since_contact": since_contact,
        "persistence": persistence,
    }


def reconstruct_level_features(  # noqa: C901 - explicit causal level families remain auditable
    *,
    pair: str,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    timeline: DataFrame,
) -> tuple[DataFrame, dict[str, Any]]:
    base = g0.prepare_base_market_frame(pair, source_manifest)
    end = pd.Timestamp(source_manifest["data"]["analysis_end_utc_exclusive"])
    base = base.loc[base["date"] < end].reset_index(drop=True)
    base["date"] = pd.to_datetime(base["date"], utc=True, errors="coerce")
    positions = (
        timeline[["date"]]
        .merge(
            base[["date"]].reset_index(names="base_position"),
            on="date",
            how="left",
            validate="one_to_one",
        )["base_position"]
    )
    if positions.isna().any():
        raise ValueError(f"Generation 8 event date missing from base candles for {pair}")
    event_index = positions.to_numpy(dtype=np.int64)
    n = len(timeline)
    contact_count = np.zeros(n)
    explicit_count = np.zeros(n)
    score_sum = np.zeros(n)
    score_count = np.zeros(n)
    score_max = np.full(n, -np.inf)
    native_width_sum = np.zeros(n)
    native_width_count = np.zeros(n)
    standard_width_sum = np.zeros(n)
    standard_width_count = np.zeros(n)
    proximity_sum = np.zeros(n)
    proximity_count = np.zeros(n)
    age_sum = np.zeros(n)
    age_count = np.zeros(n)
    persistence_sum = np.zeros(n)
    persistence_count = np.zeros(n)
    prior_episode_sum = np.zeros(n)
    prior_episode_count = np.zeros(n)
    since_contact_sum = np.zeros(n)
    since_contact_count = np.zeros(n)
    touch_state_sums = {
        name: np.zeros(n)
        for name in ("first", "occupancy", "first_retest", "later_retest")
    }
    family_flags: dict[str, np.ndarray] = {}
    dependency_flags: dict[str, np.ndarray] = {}
    timeframe_counts = {timeframe: np.zeros(n) for timeframe in ("1h", "4h", "8h", "1d")}
    lvn_count = np.zeros(n)
    pivot_sum = np.zeros(n)
    pivot_count = np.zeros(n)
    absorbed_sum = np.zeros(n)
    absorbed_count = np.zeros(n)
    vp_width_sum = np.zeros(n)
    vp_width_count = np.zeros(n)
    room_above = np.full(n, np.inf)
    room_below = np.full(n, np.inf)
    available_level_count = np.zeros(n)
    within = {threshold: np.zeros(n) for threshold in (0.5, 1.0, 2.0)}
    available_upper = np.zeros(n)
    available_lower = np.zeros(n)
    causal_violations = 0
    opened_specs = 0
    cache_dir = Path(source_manifest["storage"]["cache_dir"])
    base_atr_all = numeric_array(base["base_atr"])
    pre_close_all = numeric_array(base["pre_close"])
    high_all = numeric_array(base["high"])
    low_all = numeric_array(base["low"])
    base_dates = pd.to_datetime(base["date"], utc=True)
    for timeframe in ("1h", "4h", "8h", "1d"):
        cache_path = g0.level_cache_path(
            pair, timeframe, ("core", "generic"), cache_dir=cache_dir
        )
        g0.validate_cache_metadata(cache_path, source_manifest_path)
        level_cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        level_cache["available_at"] = pd.to_datetime(
            level_cache["available_at"], utc=True, errors="coerce"
        )
        level_cache["source_open"] = pd.to_datetime(
            level_cache["source_open"], utc=True, errors="coerce"
        )
        merged = pd.merge_asof(
            base.sort_values("date"),
            level_cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        present = merged["available_at"].notna()
        causal_violations += int(
            (merged.loc[present, "available_at"] > merged.loc[present, "date"]).sum()
        )
        for spec in g6.selected_level_specs(level_cache):
            opened_specs += 1
            level = g0.resolved_level_values(merged, spec, timeframe)
            native = g0.resolved_native_width(merged, spec)
            valid = (
                np.isfinite(level)
                & (level > 0.0)
                & np.isfinite(base_atr_all)
                & (base_atr_all > 0.0)
            )
            if spec.active_columns:
                active = np.zeros(len(merged), dtype=bool)
                for column in spec.active_columns:
                    if column in merged:
                        active |= g0.bool_array(merged[column])
                valid &= active
            width = g0.zone_half_width(
                "standard_base_atr", level, base_atr_all, native
            )
            valid &= np.isfinite(width) & (width > 0.0)
            contact = valid & (high_all >= level - width) & (low_all <= level + width)
            explicit = merged[spec.identity_column] if spec.identity_column in merged else None
            history = segment_history(
                valid=valid,
                contact=contact,
                level=level,
                width=width,
                explicit_identity=explicit,
            )
            level_event = level[event_index]
            width_event = width[event_index]
            valid_event = valid[event_index]
            contact_event = contact[event_index]
            atr_event = base_atr_all[event_index]
            pre_event = pre_close_all[event_index]
            distance = np.abs(level_event - pre_event) / atr_event
            available_level_count[valid_event] += 1.0
            for threshold in within:
                within[threshold][valid_event & (distance <= threshold)] += 1.0
            side = g6.level_side(spec)
            gap_above = (level_event - width_event - pre_event) / atr_event
            gap_below = (pre_event - level_event - width_event) / atr_event
            above = valid_event & (gap_above > 0.0)
            below = valid_event & (gap_below > 0.0)
            room_above[above] = np.minimum(room_above[above], gap_above[above])
            room_below[below] = np.minimum(room_below[below], gap_below[below])
            if side == "upper":
                available_upper[valid_event] += 1.0
            elif side == "lower":
                available_lower[valid_event] += 1.0
            if not contact_event.any():
                continue
            contact_count[contact_event] += 1.0
            timeframe_counts[timeframe][contact_event] += 1.0
            family_flags.setdefault(spec.family, np.zeros(n, dtype=bool))[contact_event] = True
            dependency = DEPENDENCY_GROUP[spec.family]
            dependency_flags.setdefault(dependency, np.zeros(n, dtype=bool))[contact_event] = True
            if explicit is not None:
                explicit_event = explicit.iloc[event_index].notna().to_numpy() & contact_event
                explicit_count[explicit_event] += 1.0
            standard_width = width_event / atr_event
            update_sum(
                standard_width_sum,
                standard_width_count,
                standard_width,
                contact_event,
            )
            native_atr = np.abs(native[event_index]) / atr_event
            update_sum(
                native_width_sum,
                native_width_count,
                native_atr,
                contact_event,
            )
            update_sum(proximity_sum, proximity_count, distance, contact_event)
            available_at = pd.to_datetime(
                merged["available_at"], utc=True, errors="coerce"
            ).iloc[event_index]
            age = (
                base_dates.iloc[event_index].reset_index(drop=True)
                - available_at.reset_index(drop=True)
            ).dt.total_seconds().div(3600.0).to_numpy(dtype=float)
            update_sum(age_sum, age_count, age, contact_event & (age >= 0.0))
            for name, total in touch_state_sums.items():
                total[contact_event] += history[name][event_index][contact_event]
            update_sum(
                persistence_sum,
                persistence_count,
                history["persistence"][event_index],
                contact_event,
            )
            update_sum(
                prior_episode_sum,
                prior_episode_count,
                history["prior_episodes"][event_index],
                contact_event,
            )
            update_sum(
                since_contact_sum,
                since_contact_count,
                history["time_since_contact"][event_index],
                contact_event,
            )
            if spec.score_column and spec.score_column in merged:
                score = numeric_array(merged[spec.score_column])[event_index]
                update_sum(score_sum, score_count, score, contact_event)
                finite_score = contact_event & np.isfinite(score)
                score_max[finite_score] = np.maximum(
                    score_max[finite_score], score[finite_score]
                )
            if spec.family == "volume_profile_nodes" and "lvn" in spec.name.lower():
                lvn_count[contact_event] += 1.0
            for column in spec.attribute_columns:
                if column not in merged:
                    continue
                values = numeric_array(merged[column])[event_index]
                if "absorbed_pivot_count" in column:
                    update_sum(absorbed_sum, absorbed_count, values, contact_event)
                elif "pivot_count" in column:
                    update_sum(pivot_sum, pivot_count, values, contact_event)
                elif "value_area_width_pct" in column:
                    update_sum(vp_width_sum, vp_width_count, values, contact_event)
    if causal_violations:
        raise ValueError(f"Future level-cache rows for {pair}: {causal_violations}")
    contact_denominator = np.maximum(contact_count, 1.0)
    score_max[~np.isfinite(score_max)] = 0.0
    output = DataFrame({"date": timeline["date"]})
    output["g8_level_identity__contacted_level_count"] = contact_count
    output["g8_level_identity__distinct_family_count"] = sum(
        flag.astype(float) for flag in family_flags.values()
    )
    output["g8_level_identity__independent_dependency_group_count"] = sum(
        flag.astype(float) for flag in dependency_flags.values()
    )
    output["g8_level_identity__explicit_identity_fraction"] = explicit_count / contact_denominator
    output["g8_level_prominence__score_available_fraction"] = score_count / contact_denominator
    output["g8_level_prominence__mean_score"] = safe_mean(score_sum, score_count)
    output["g8_level_prominence__maximum_score"] = score_max
    output["g8_level_prominence__lvn_contact_fraction"] = lvn_count / contact_denominator
    output["g8_level_prominence__mean_pivot_count"] = safe_mean(pivot_sum, pivot_count)
    output["g8_level_prominence__mean_absorbed_pivot_count"] = safe_mean(
        absorbed_sum, absorbed_count
    )
    output["g8_level_prominence__mean_vp_value_area_width_pct"] = safe_mean(
        vp_width_sum, vp_width_count
    )
    output["g8_level_width__mean_standard_half_width_atr"] = safe_mean(
        standard_width_sum, standard_width_count
    )
    output["g8_level_width__native_width_available_fraction"] = (
        native_width_count / contact_denominator
    )
    output["g8_level_width__mean_native_half_width_atr"] = safe_mean(
        native_width_sum, native_width_count
    )
    output["g8_level_proximity__mean_contact_distance_atr"] = safe_mean(
        proximity_sum, proximity_count
    )
    for timeframe, count in timeframe_counts.items():
        output[f"g8_level_timeframe__{timeframe}_contact_fraction"] = (
            count / contact_denominator
        )
    output["g8_level_timeframe__distinct_contact_timeframes"] = sum(
        count.gt(0.0).astype(float) if isinstance(count, Series) else (count > 0.0).astype(float)
        for count in timeframe_counts.values()
    )
    output["g8_level_history__mean_level_age_hours"] = safe_mean(age_sum, age_count)
    output["g8_level_history__mean_lineage_persistence_hours"] = safe_mean(
        persistence_sum, persistence_count
    )
    output["g8_level_history__mean_prior_episode_count"] = safe_mean(
        prior_episode_sum, prior_episode_count
    )
    output["g8_level_history__mean_hours_since_prior_contact"] = safe_mean(
        since_contact_sum, since_contact_count
    )
    output["g8_level_history__prior_contact_available_fraction"] = (
        since_contact_count / contact_denominator
    )
    touch_blocks = {
        "first": "g8_touch_first",
        "occupancy": "g8_touch_occupancy",
        "first_retest": "g8_touch_retest",
        "later_retest": "g8_touch_retest",
    }
    for name, total in touch_state_sums.items():
        block = touch_blocks[name]
        output[f"{block}__{name}_fraction"] = total / contact_denominator
        output[f"{block}__{name}_count"] = total
    has_above = np.isfinite(room_above)
    has_below = np.isfinite(room_below)
    output["g8_room_geometry__has_level_above"] = has_above.astype(float)
    output["g8_room_geometry__has_level_below"] = has_below.astype(float)
    output["g8_room_geometry__nearest_gap_above_atr"] = np.where(
        has_above, room_above, 0.0
    )
    output["g8_room_geometry__nearest_gap_below_atr"] = np.where(
        has_below, room_below, 0.0
    )
    both = has_above & has_below
    output["g8_room_geometry__minimum_two_sided_room_atr"] = np.where(
        both, np.minimum(room_above, room_below), 0.0
    )
    output["g8_room_geometry__available_level_count"] = available_level_count
    output["g8_room_geometry__available_upper_fraction"] = np.divide(
        available_upper,
        np.maximum(available_level_count, 1.0),
    )
    output["g8_room_geometry__available_lower_fraction"] = np.divide(
        available_lower,
        np.maximum(available_level_count, 1.0),
    )
    for threshold, values in within.items():
        name = str(threshold).replace(".", "_")
        output[f"g8_room_geometry__level_count_within_{name}_atr"] = values
    audit = {
        "pair": pair,
        "base_rows": len(base),
        "event_rows": n,
        "opened_level_specs": opened_specs,
        "causal_timestamp_violations": causal_violations,
        "contacted_level_rows": int(contact_count.sum()),
        "first_contact_rows": int(touch_state_sums["first"].sum()),
        "occupancy_rows": int(touch_state_sums["occupancy"].sum()),
        "first_retest_rows": int(touch_state_sums["first_retest"].sum()),
        "later_retest_rows": int(touch_state_sums["later_retest"].sum()),
        "reaction_outcome_columns_read": False,
    }
    return output, audit


def current_block_names(frame: DataFrame) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            column.split("__", maxsplit=1)[0]
            for column in frame
            if "__" in column
            and not column.endswith("_stale")
            and not column.endswith("_shuffled")
        )
    )


def complete_block(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No Generation 8 columns for {block!r}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def attach_stale_and_shuffled(
    frame: DataFrame, *, pair: str, blocks: Sequence[str]
) -> tuple[DataFrame, list[dict[str, Any]]]:
    output = frame.copy()
    audits: list[dict[str, Any]] = []
    for block in blocks:
        columns = [column for column in frame if column.startswith(f"{block}__")]
        stale_parts = []
        shuffled_parts = []
        self_matches = 0
        stale_violations = 0
        for period, period_frame in frame.groupby("period", sort=False):
            target = period_frame[["date"]].sort_values("date").copy()
            target["cutoff"] = target["date"] - pd.Timedelta(hours=STALE_HOURS)
            source = period_frame.loc[
                complete_block(period_frame, block), ["date", *columns]
            ].sort_values("date").copy()
            source.rename(columns={"date": "source_date"}, inplace=True)
            if source.empty:
                stale = target.copy()
                stale["source_date"] = pd.Series(
                    pd.NaT,
                    index=stale.index,
                    dtype=target["date"].dtype,
                )
                for column in columns:
                    stale[column] = np.nan
            else:
                stale = pd.merge_asof(
                    target.sort_values("cutoff"),
                    source.sort_values("source_date"),
                    left_on="cutoff",
                    right_on="source_date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            valid = stale["source_date"].notna()
            stale_violations += int(
                (
                    valid
                    & (
                        stale["source_date"]
                        > stale["date"] - pd.Timedelta(hours=STALE_HOURS)
                    )
                ).sum()
            )
            stale.rename(
                columns={
                    column: f"{block}_stale__{column.removeprefix(f'{block}__')}"
                    for column in columns
                },
                inplace=True,
            )
            stale_parts.append(
                stale[
                    [
                        "date",
                        *(
                            f"{block}_stale__{column.removeprefix(f'{block}__')}"
                            for column in columns
                        ),
                    ]
                ]
            )
            ordered_all = period_frame.sort_values("date").copy()
            ordered = ordered_all.loc[complete_block(ordered_all, block)].copy()
            n = len(ordered)
            shuffled = ordered_all[["date"]].copy()
            if n < 2:
                for column in columns:
                    shuffled[
                        f"{block}_shuffled__{column.removeprefix(f'{block}__')}"
                    ] = np.nan
            else:
                offset = 1 + g0.stable_hash_int(f"g8|{pair}|{period}|{block}") % (n - 1)
                source_dates = np.roll(ordered["date"].to_numpy(), offset)
                self_matches += int((source_dates == ordered["date"].to_numpy()).sum())
                assigned = DataFrame({"date": ordered["date"].to_numpy()})
                for column in columns:
                    assigned[
                        f"{block}_shuffled__{column.removeprefix(f'{block}__')}"
                    ] = np.roll(numeric_array(ordered[column]), offset)
                shuffled = shuffled.merge(
                    assigned, on="date", how="left", validate="one_to_one"
                )
            shuffled_parts.append(shuffled)
        stale_frame = pd.concat(stale_parts, ignore_index=True)
        shuffled_frame = pd.concat(shuffled_parts, ignore_index=True)
        output = output.merge(stale_frame, on="date", how="left", validate="one_to_one")
        output = output.merge(
            shuffled_frame, on="date", how="left", validate="one_to_one"
        )
        audits.append(
            {
                "block": block,
                "stale_hours": STALE_HOURS,
                "stale_timestamp_violations": stale_violations,
                "shuffled_self_matches": self_matches,
            }
        )
    return output, audits


def build_pair_feature_support(task: CacheTask) -> dict[str, Any]:
    g7_features = pd.read_parquet(task.g7_item["feature_path"])
    g7_features["date"] = pd.to_datetime(
        g7_features["date"], utc=True, errors="coerce"
    )
    support = pd.read_parquet(task.g7_item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    if g7_features["date"].duplicated().any() or support["date"].duplicated().any():
        raise ValueError(f"Duplicate Generation 7 cache dates for {task.pair}")
    timeline = support[["date", "period", *(
        column for column in support if column.startswith("ready__")
    )]].copy()
    current = split_g7_blocks(g7_features)
    current = timeline[["date", "period"]].merge(
        current, on="date", how="inner", validate="one_to_one"
    )
    source_manifest = json.loads(task.source_manifest.read_text(encoding="utf-8"))
    level_features, level_audit = reconstruct_level_features(
        pair=task.pair,
        source_manifest=source_manifest,
        source_manifest_path=task.source_manifest,
        timeline=timeline,
    )
    current = current.merge(level_features, on="date", how="left", validate="one_to_one")
    base = g0.prepare_base_market_frame(task.pair, source_manifest)
    end = pd.Timestamp(source_manifest["data"]["analysis_end_utc_exclusive"])
    base = base.loc[base["date"] < end].reset_index(drop=True)
    state = continuous_state(base)
    participation_cell = cell_for_branch(task.cohort, "g8b")
    volatility_cell = cell_for_branch(task.cohort, "g8c")
    trend_cell = cell_for_branch(task.cohort, "g8d")
    for duration in (
        duration_block(
            block="g8_participation_duration",
            cell=participation_cell,
            timeline=timeline,
            source=state,
            definitions={
                name: definition
                for name, definition in g6d.STATE_FEATURES.items()
                if name in {"relative_volume", "volume_acceleration_magnitude", "absolute_pressure", "pressure_persistence"}
            },
            predictor_prefix="ohlcv_volume_pressure",
        ),
        duration_block(
            block="g8_volatility_duration",
            cell=volatility_cell,
            timeline=timeline,
            source=state,
            definitions={
                name: definition
                for name, definition in g6d.STATE_FEATURES.items()
                if name in {"atr_fraction", "prior_range_atr", "bollinger_width", "range_contraction_ratio"}
            },
            predictor_prefix="ohlcv_volatility_range",
        ),
        duration_block(
            block="g8_trend_duration",
            cell=trend_cell,
            timeline=timeline,
            source=state,
            definitions={
                name: definition
                for name, definition in g6d.STATE_FEATURES.items()
                if name in {
                    "ema20_slope_magnitude",
                    "ma_separation",
                    "return_slope_magnitude",
                    "return_acceleration_magnitude",
                    "adx14",
                    "rsi_distance_from_50",
                    "rsi_change_magnitude",
                    "macd_histogram_magnitude",
                    "macd_change_magnitude",
                }
            },
            predictor_prefix="ohlcv_trend_momentum",
        ),
    ):
        current = current.merge(duration, on="date", how="left", validate="one_to_one")
    cross = continuous_cross_market(task.cohort)
    btc_cell = cell_for_branch(task.cohort, "g8e")
    btc_duration = duration_block(
        block="g8_btc_duration",
        cell=btc_cell,
        timeline=timeline,
        source=cross,
        definitions={
            name: definition
            for name, definition in g6d.CROSS_MARKET_FEATURES.items()
            if name in {"btc_activity_1h", "btc_activity_4h", "btc_activity_24h", "btc_relative_volume"}
        },
        predictor_prefix="cross_market_btc",
    )
    current = current.merge(btc_duration, on="date", how="left", validate="one_to_one")
    if task.cohort == "normal":
        current = current.merge(
            orderbook_blocks(timeline), on="date", how="left", validate="one_to_one"
        )
    blocks = current_block_names(current)
    full, placebo_audit = attach_stale_and_shuffled(
        current, pair=task.pair, blocks=blocks
    )
    if any(
        row["stale_timestamp_violations"] or row["shuffled_self_matches"]
        for row in placebo_audit
    ):
        raise ValueError(f"Generation 8 placebo violation for {task.pair}")
    output_support = timeline[["date", "period"]].copy()
    cell_support: list[dict[str, Any]] = []
    for cell in g8p.CELLS:
        if cell.cohort != task.cohort:
            continue
        if cell.pair_scope == "smart_contract_platforms" and task.pair not in g8p.SMART_CONTRACT_PLATFORMS:
            continue
        parent = f"ready__{cell.parent_ready}"
        ready = timeline[parent].fillna(False).astype(bool)
        for block in cell_blocks(cell):
            for variant in (block, f"{block}_stale", f"{block}_shuffled"):
                ready &= complete_block(full, variant)
        column = f"ready__{cell_ready_name(cell)}"
        output_support[column] = ready
        for period, period_frame in output_support.groupby("period", sort=False):
            cell_support.append(
                {
                    "cell_id": cell.cell_id,
                    "ready_block": cell_ready_name(cell),
                    "cohort": task.cohort,
                    "pair": task.pair,
                    "period": period,
                    "events": int(period_frame[column].sum()),
                }
            )
    feature_columns = [column for column in full if "__" in column]
    feature_cache = full[["date", *feature_columns]].copy()
    for column in feature_columns:
        feature_cache[column] = pd.to_numeric(
            feature_cache[column], errors="coerce"
        ).astype("float32")
    stem = g0.pair_file_stem(task.pair)
    feature_path = task.feature_dir / f"{stem}.parquet"
    support_path = task.support_dir / f"{stem}.parquet"
    g0.atomic_write_parquet(feature_cache, feature_path)
    g0.atomic_write_parquet(output_support, support_path)
    return {
        "cohort": task.cohort,
        "pair": task.pair,
        "feature_path": str(feature_path),
        "feature_sha256": g0.sha256_file(feature_path),
        "support_path": str(support_path),
        "support_sha256": g0.sha256_file(support_path),
        "g7_feature_path": task.g7_item["feature_path"],
        "g7_feature_sha256": task.g7_item["feature_sha256"],
        "g7_event_path": task.g7_item["event_path"],
        "g7_event_sha256": task.g7_item["event_sha256"],
        "evaluation_path": task.g7_item["evaluation_path"],
        "evaluation_sha256": task.g7_item["evaluation_sha256"],
        "raw_provenance_path": task.g7_item["source_path"],
        "raw_provenance_sha256": task.g7_item["source_sha256"],
        "feature_rows": len(feature_cache),
        "feature_columns": len(feature_columns),
        "cell_support": cell_support,
        "level_audit": level_audit,
        "placebo_audit": placebo_audit,
        "reaction_outcome_columns_read": False,
    }


def exact_support_decisions(
    cohort: str, inventory: Sequence[dict[str, Any]]
) -> tuple[DataFrame, list[dict[str, Any]]]:
    support = DataFrame.from_records(
        row for item in inventory for row in item["cell_support"]
    )
    decisions: list[dict[str, Any]] = []
    for cell in g8p.CELLS:
        if cell.cohort != cohort:
            continue
        selected = support.loc[support["cell_id"].eq(cell.cell_id)]
        periods = (
            g8p.DEVELOPMENT_PERIOD[cohort],
            *g8p.VALIDATION_PERIODS[cohort],
        )
        checks = []
        for period in periods:
            frame = selected.loc[selected["period"].eq(period)]
            positive = frame.loc[frame["events"].gt(0)]
            checks.append(
                {
                    "period": period,
                    "events": int(frame["events"].sum()),
                    "coins": int(positive["pair"].nunique()),
                    "minimum_pair_events": int(positive["events"].min()) if len(positive) else 0,
                }
            )
        development, *validation = checks
        supported = (
            development["events"] >= 100
            and development["coins"] >= 5
            and all(check["events"] >= 50 and check["coins"] >= 5 for check in validation)
        )
        decisions.append(
            {
                "cell_id": cell.cell_id,
                "branch_id": cell.branch_id,
                "cohort": cohort,
                "surface": cell.surface,
                "ready_block": cell_ready_name(cell),
                "blocks": list(cell_blocks(cell)),
                "period_checks": checks,
                "status": (
                    "supported_for_generation8_models"
                    if supported
                    else "parked_outcome_blind_insufficient_exact_common_support"
                ),
                "reaction_outcomes_opened": False,
            }
        )
    return support, decisions


def materialize_event_cache(item: dict[str, Any], event_dir: Path) -> dict[str, Any]:
    source_columns = [
        "date",
        "period",
        "pair",
        "cohort",
        "market_group",
        "smart_contract_platform",
        *TARGET_COLUMNS,
    ]
    events = pd.read_parquet(item["g7_event_path"], columns=source_columns)
    events["date"] = pd.to_datetime(events["date"], utc=True, errors="coerce")
    support = pd.read_parquet(item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    ready = [column for column in support if column.startswith("ready__")]
    output = events.merge(
        support[["date", *ready]], on="date", how="inner", validate="one_to_one"
    )
    path = event_dir / f"{g0.pair_file_stem(item['pair'])}.parquet"
    g0.atomic_write_parquet(output, path)
    return {
        **item,
        "event_path": str(path),
        "event_sha256": g0.sha256_file(path),
        "event_rows": len(output),
        "reaction_outcomes_opened_after_support_freeze": True,
    }


def build_cohort_cache(
    *, cohort: str, workers: int, overwrite: bool = False
) -> dict[str, Any]:
    load_preflight()
    record_dir = RECORD_ROOT
    record_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = record_dir / f"{cohort}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_generation8_attribution_shared_cache":
            raise ValueError(f"Incomplete existing Generation 8 cache: {manifest_path}")
        return manifest
    g7_manifest = load_g7_manifest(cohort)
    source_manifest_path = g8p.G6_SOURCE_MANIFESTS[cohort]
    artifact_dir = ARTIFACT_ROOT / cohort
    feature_dir = artifact_dir / "feature_cache"
    support_dir = artifact_dir / "outcome_blind_support_cache"
    event_dir = artifact_dir / "event_cache"
    for path in (feature_dir, support_dir, event_dir):
        path.mkdir(parents=True, exist_ok=True)
    tasks = [
        CacheTask(
            cohort=cohort,
            pair=str(item["pair"]),
            source_manifest=source_manifest_path,
            g7_item=item,
            feature_dir=feature_dir,
            support_dir=support_dir,
        )
        for item in g7_manifest["inventory"]
    ]
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {pool.submit(build_pair_feature_support, task): task.pair for task in tasks}
        for future in as_completed(futures):
            item = future.result()
            inventory.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g8_outcome_blind_feature_cache",
                        "cohort": cohort,
                        "pair": item["pair"],
                        "features": item["feature_columns"],
                    }
                ),
                flush=True,
            )
    pair_order = [str(item["pair"]) for item in g7_manifest["inventory"]]
    inventory.sort(key=lambda item: pair_order.index(str(item["pair"])))
    support, decisions = exact_support_decisions(cohort, inventory)
    support_csv = record_dir / f"{cohort}_exact_model_support.csv"
    decisions_json = record_dir / f"{cohort}_exact_model_support_decisions.json"
    g0.atomic_write_csv(support, support_csv)
    g0.atomic_write_json(decisions, decisions_json)
    supported = [row for row in decisions if row["status"].startswith("supported")]
    if len(supported) != len(decisions):
        raise ValueError(f"Generation 8 {cohort} exact support parked one or more cells.")
    # Only after exact predictor support is frozen may current reaction labels be opened.
    inventory = [materialize_event_cache(item, event_dir) for item in inventory]
    manifest = {
        "schema_version": 1,
        "cache_id": SHARED_CACHE_ID,
        "cohort": cohort,
        "status": "completed_generation8_attribution_shared_cache",
        "created_at_utc": g0.utc_now(),
        "pairs": pair_order,
        "supported_cells": [row["cell_id"] for row in supported],
        "inventory": inventory,
        "source_contracts": {
            "frozen_batch": str(g8p.FROZEN_BATCH.resolve()),
            "frozen_batch_sha256": g0.sha256_file(g8p.FROZEN_BATCH),
            "outcome_blind_preflight": str(PREFLIGHT_MANIFEST.resolve()),
            "outcome_blind_preflight_sha256": g0.sha256_file(PREFLIGHT_MANIFEST),
            "g7_parent_manifest": str(
                (g8p.G7_SHARED_RECORD / f"{cohort}_manifest.json").resolve()
            ),
            "g7_parent_manifest_sha256": g0.sha256_file(
                g8p.G7_SHARED_RECORD / f"{cohort}_manifest.json"
            ),
        },
        "outcome_blind_sequence": {
            "feature_and_placebo_cache_completed_first": True,
            "exact_support_decisions_written_before_target_read": True,
            "reaction_targets_materialized_only_after_support_freeze": True,
            "support_decisions_path": str(decisions_json.resolve()),
            "support_decisions_sha256": g0.sha256_file(decisions_json),
            "prior_reaction_history_subdimension": (
                "parked_before_outcomes because cross-family lineage history requires a "
                "separate causal availability audit; contact-state attribution remains open"
            ),
        },
        "integrity": {
            "causal_level_timestamp_violations": sum(
                item["level_audit"]["causal_timestamp_violations"] for item in inventory
            ),
            "stale_timestamp_violations": sum(
                row["stale_timestamp_violations"]
                for item in inventory
                for row in item["placebo_audit"]
            ),
            "shuffled_self_matches": sum(
                row["shuffled_self_matches"]
                for item in inventory
                for row in item["placebo_audit"]
            ),
            "profit_features": 0,
            "future_signed_direction_features": 0,
        },
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the outcome-blind Generation 8 causal attribution cache."
    )
    parser.add_argument("--cohort", choices=("normal", "meme", "both"), default="both")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cohorts = ("normal", "meme") if args.cohort == "both" else (args.cohort,)
    manifests = [
        build_cohort_cache(
            cohort=cohort, workers=args.workers, overwrite=args.overwrite
        )
        for cohort in cohorts
    ]
    print(
        json.dumps(
            [
                {
                    "cohort": manifest["cohort"],
                    "status": manifest["status"],
                    "pairs": len(manifest["pairs"]),
                    "supported_cells": len(manifest["supported_cells"]),
                }
                for manifest in manifests
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
