from __future__ import annotations

# Bound numerical libraries before importing pandas/FreqAI helpers.
# ruff: noqa: E402, E501
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
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation6 as g6f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation7_preflight as g7p,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (
    runtime_snapshot,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation5 import (
    deterministic_block_bootstrap,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration7Strategy import (
    TARGET_COLUMNS,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration7Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG7ConfigurableFreqAIResearchStrategy"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation6_review" / "g7_frozen_pairwise_batch.json"
OUTCOME_BLIND_PREFLIGHT = (
    OUTPUT_ROOT
    / "generation7_branches"
    / "g7_preflight"
    / "g7_outcome_blind_preflight_20260821a"
    / "manifest.json"
)
RECORD_ROOT = OUTPUT_ROOT / "generation7_branches" / "g7_freqai_pairwise_interactions"
SHARED_RECORD_ROOT = OUTPUT_ROOT / "generation7_shared"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation7_branches"
    / "g7_freqai_pairwise_interactions"
)
SHARED_ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation7_shared"
    / "g7_enriched_cache_20260821a"
)
SHARED_CACHE_ID = "g7_enriched_cache_20260821a"

MAX_WORKERS = 4
STALE_HOURS = 72
MIN_PAIR_SCORABLE_ROWS = 15
MIN_AGGREGATE_SCORABLE_ROWS = 50
MIN_TRAINING_ROWS = 30
BOOTSTRAP_SAMPLES = 512
CLUSTER_MAX_ZONE_GAP_ATR = 0.10

SPECIAL_PRIOR_RANGE = "g7_prior_range_geometry"
SPECIAL_VP4H = "g7_vp4h_geometry"
SPECIAL_CLUSTER_A = "g7_cluster_family_a"
SPECIAL_CLUSTER_B = "g7_cluster_family_b"


@dataclass(frozen=True)
class BranchRuntimeSpec:
    branch_id: str
    cohorts: tuple[str, ...]
    level_blocks: tuple[str, ...]
    component_a_blocks: tuple[str, ...]
    component_b_blocks: tuple[str, ...]
    scope: str

    @property
    def ready_block(self) -> str:
        return f"g7_common_{self.branch_id.removeprefix('g7').split('_', 1)[0]}"


BRANCH_RUNTIME: dict[str, BranchRuntimeSpec] = {
    "g7a_local_participation_and_btc_activity": BranchRuntimeSpec(
        "g7a_local_participation_and_btc_activity",
        ("normal", "meme"),
        ("level",),
        ("ohlcv_volume_pressure",),
        ("cross_market_btc",),
        "all",
    ),
    "g7b_local_participation_and_volatility": BranchRuntimeSpec(
        "g7b_local_participation_and_volatility",
        ("normal", "meme"),
        ("level",),
        ("ohlcv_volume_pressure",),
        ("ohlcv_volatility_range",),
        "all",
    ),
    "g7c_local_participation_and_trend_momentum": BranchRuntimeSpec(
        "g7c_local_participation_and_trend_momentum",
        ("normal", "meme"),
        ("level",),
        ("ohlcv_volume_pressure",),
        ("ohlcv_trend_momentum",),
        "all",
    ),
    "g7d_cross_timeframe_agreement_and_local_participation": BranchRuntimeSpec(
        "g7d_cross_timeframe_agreement_and_local_participation",
        ("normal", "meme"),
        ("level",),
        ("timeframe_4h", "timeframe_8h", "timeframe_1d"),
        ("ohlcv_volume_pressure",),
        "different_mechanism_cross_timeframe",
    ),
    "g7e_prior_range_level_and_local_participation": BranchRuntimeSpec(
        "g7e_prior_range_level_and_local_participation",
        ("normal", "meme"),
        ("level",),
        (SPECIAL_PRIOR_RANGE,),
        ("ohlcv_volume_pressure",),
        "generic_prior_range",
    ),
    "g7f_prior_volume_profile_area_and_compression": BranchRuntimeSpec(
        "g7f_prior_volume_profile_area_and_compression",
        ("normal", "meme"),
        ("level",),
        (SPECIAL_VP4H,),
        ("ohlcv_volatility_range",),
        "volume_profile_explicit_prior_4h",
    ),
    "g7g_independent_level_family_cluster": BranchRuntimeSpec(
        "g7g_independent_level_family_cluster",
        ("normal", "meme"),
        ("level",),
        (SPECIAL_CLUSTER_A,),
        (SPECIAL_CLUSTER_B,),
        "independent_level_family_cluster",
    ),
    "g7h_meme_eth_activity_and_local_participation": BranchRuntimeSpec(
        "g7h_meme_eth_activity_and_local_participation",
        ("meme",),
        ("level",),
        ("ohlcv_volume_pressure",),
        ("cross_market_eth",),
        "all",
    ),
    "g7i_orderbook_and_btc_activity_at_levels": BranchRuntimeSpec(
        "g7i_orderbook_and_btc_activity_at_levels",
        ("normal",),
        ("level",),
        ("orderbook_btc",),
        ("cross_market_btc",),
        "all",
    ),
    "g7j_gdelt_activity_and_local_participation": BranchRuntimeSpec(
        "g7j_gdelt_activity_and_local_participation",
        ("normal", "meme"),
        ("level",),
        ("news_gdelt",),
        ("ohlcv_volume_pressure",),
        "all",
    ),
}

PROFILE_ROLES = (
    "level_only",
    "level_plus_component_a",
    "level_plus_component_b",
    "component_a_plus_component_b_without_level",
    "level_plus_both_components",
    "stale_level_plus_both_components",
    "level_plus_stale_component_a_plus_component_b",
    "level_plus_component_a_plus_stale_component_b",
)
COMPLETE_ROLE = "level_plus_both_components"


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def stale_blocks(blocks: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{block}_placebo" for block in blocks)


def load_frozen_batch() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    batch = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    if batch.get("status") != "frozen_before_generation7_reaction_outcomes":
        raise ValueError("Generation 7 was not frozen before outcomes.")
    branches = {str(item["id"]): item for item in batch["branches"]}
    if set(branches) != set(BRANCH_RUNTIME):
        raise ValueError("Generation 7 runtime registry differs from the frozen batch.")
    if batch["research_boundary"].get("profit_optimization") is not False:
        raise ValueError("Profit is outside Generation 7.")
    if batch["research_boundary"].get("direction_prediction") is not False:
        raise ValueError("Direction is outside Generation 7.")
    return batch, branches


FROZEN, FROZEN_BRANCHES = load_frozen_batch()


def profile_blocks(spec: BranchRuntimeSpec, role: str) -> tuple[str, ...]:
    level = spec.level_blocks
    component_a = spec.component_a_blocks
    component_b = spec.component_b_blocks
    if role == "level_only":
        return level
    if role == "level_plus_component_a":
        return (*level, *component_a)
    if role == "level_plus_component_b":
        return (*level, *component_b)
    if role == "component_a_plus_component_b_without_level":
        return (*component_a, *component_b)
    if role == COMPLETE_ROLE:
        return (*level, *component_a, *component_b)
    if role == "stale_level_plus_both_components":
        return (*stale_blocks(level), *component_a, *component_b)
    if role == "level_plus_stale_component_a_plus_component_b":
        return (*level, *stale_blocks(component_a), *component_b)
    if role == "level_plus_component_a_plus_stale_component_b":
        return (*level, *component_a, *stale_blocks(component_b))
    raise ValueError(f"Unknown Generation 7 profile role: {role}")


def build_profile_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for branch_id, spec in BRANCH_RUNTIME.items():
        frozen = FROZEN_BRANCHES[branch_id]
        if tuple(frozen["profile_ladder"]) != PROFILE_ROLES:
            raise ValueError(f"Profile ladder drift for {branch_id}")
        targets = tuple(str(target) for target in frozen["targets"])
        unknown_targets = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown_targets:
            raise ValueError(f"Unknown targets for {branch_id}: {unknown_targets}")
        for role in PROFILE_ROLES:
            profile_id = f"{branch_id}__{role}"
            profiles[profile_id] = {
                "branch_id": branch_id,
                "role": role,
                "blocks": profile_blocks(spec, role),
                "required_ready_blocks": (spec.ready_block,),
                "targets": targets,
                "plain_name": frozen["plain_name"],
            }
        candidate = f"{branch_id}__{COMPLETE_ROLE}"
        for role in PROFILE_ROLES:
            if role == COMPLETE_ROLE:
                continue
            comparisons.append(
                {
                    "comparison_id": f"{branch_id}__complete_vs_{role}",
                    "branch_id": branch_id,
                    "candidate": candidate,
                    "baseline": f"{branch_id}__{role}",
                    "baseline_role": role,
                    "targets": list(targets),
                    "question": (
                        f"Does the complete interaction improve {frozen['plain_name']} "
                        f"beyond {role.replace('_', ' ')}?"
                    ),
                }
            )
    return profiles, comparisons


PROFILES, COMPARISONS = build_profile_registry()


def complete_block_mask(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No feature columns found for block {block!r}.")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def aggregate_specific_geometry(rows: DataFrame, *, block: str) -> DataFrame:
    columns = [
        f"{block}__event_count",
        f"{block}__distinct_timeframe_count",
        f"{block}__distinct_level_count",
        f"{block}__mean_zone_half_width_atr",
        f"{block}__median_pre_distance_atr",
        f"{block}__mean_contact_close_distance_atr",
        f"{block}__level_span_atr",
        *(f"{block}__timeframe_{timeframe}_count" for timeframe in ("1h", "4h", "8h", "1d")),
    ]
    if block == SPECIAL_VP4H:
        columns.extend(
            (
                f"{block}__mean_value_area_width_pct",
                f"{block}__mean_vp_state_magnitude",
            )
        )
    if rows.empty:
        return DataFrame(columns=["date", *columns])
    grouped = rows.groupby("date", sort=True)
    output = grouped.agg(
        event_count=("level_family", "size"),
        distinct_timeframe_count=("source_timeframe", "nunique"),
        distinct_level_count=("level_name", "nunique"),
        mean_zone_half_width_atr=("zone_half_width_atr", "mean"),
        median_pre_distance_atr=("pre_distance_atr", "median"),
        mean_contact_close_distance_atr=("contact_close_distance_atr", "mean"),
        minimum_level_price=("level_price", "min"),
        maximum_level_price=("level_price", "max"),
        median_base_atr=("base_atr", "median"),
    ).reset_index()
    output["level_span_atr"] = (
        (output["maximum_level_price"] - output["minimum_level_price"])
        / output["median_base_atr"].replace(0.0, np.nan)
    )
    output.drop(
        columns=["minimum_level_price", "maximum_level_price", "median_base_atr"],
        inplace=True,
    )
    output.rename(
        columns={column: f"{block}__{column}" for column in output if column != "date"},
        inplace=True,
    )
    for timeframe in ("1h", "4h", "8h", "1d"):
        counts = (
            rows["source_timeframe"]
            .eq(timeframe)
            .groupby(rows["date"])
            .sum()
            .reindex(output["date"], fill_value=0)
            .to_numpy(dtype=float)
        )
        output[f"{block}__timeframe_{timeframe}_count"] = counts
    if block == SPECIAL_VP4H:
        extra = grouped.agg(
            mean_value_area_width_pct=("attr_vp_value_area_width_pct", "mean"),
            mean_vp_state_magnitude=("attr_vp_state", lambda values: values.abs().mean()),
        ).reset_index()
        extra.rename(
            columns={column: f"{block}__{column}" for column in extra if column != "date"},
            inplace=True,
        )
        output = output.merge(extra, on="date", how="left", validate="one_to_one")
    return output[["date", *columns]]


def dependency_group_summary(rows: DataFrame) -> DataFrame:
    frame = rows.copy()
    frame["dependency_group"] = frame["level_family"].map(
        g7p.LEVEL_DEPENDENCY_GROUPS
    )
    if frame["dependency_group"].isna().any():
        unknown = sorted(
            frame.loc[frame["dependency_group"].isna(), "level_family"].unique()
        )
        raise ValueError(f"Unknown cluster dependency families: {unknown}")
    return (
        frame.groupby(["date", "dependency_group"], sort=True, as_index=False)
        .agg(
            event_count=("level_family", "size"),
            distinct_timeframe_count=("source_timeframe", "nunique"),
            representative_level_price=("level_price", "median"),
            median_base_atr=("base_atr", "median"),
            mean_zone_half_width_atr=("zone_half_width_atr", "mean"),
            median_pre_distance_atr=("pre_distance_atr", "median"),
            mean_contact_close_distance_atr=("contact_close_distance_atr", "mean"),
        )
        .reset_index(drop=True)
    )


def aggregate_independent_cluster(rows: DataFrame) -> tuple[DataFrame, DataFrame]:
    summary = dependency_group_summary(rows)
    group_names = tuple(sorted(set(g7p.LEVEL_DEPENDENCY_GROUPS.values())))
    component_columns = (
        "event_count",
        "distinct_timeframe_count",
        "mean_zone_half_width_atr",
        "median_pre_distance_atr",
        "mean_contact_close_distance_atr",
    )
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    for date, frame in summary.groupby("date", sort=True):
        if len(frame) < 2:
            continue
        candidates: list[tuple[float, float, str, str, Series, Series]] = []
        for (_, left), (_, right) in combinations(frame.iterrows(), 2):
            atr = float(np.nanmedian([left["median_base_atr"], right["median_base_atr"]]))
            if not np.isfinite(atr) or atr <= 0.0:
                continue
            separation = abs(
                float(left["representative_level_price"])
                - float(right["representative_level_price"])
            ) / atr
            combined_width = float(left["mean_zone_half_width_atr"]) + float(
                right["mean_zone_half_width_atr"]
            )
            zone_gap = max(0.0, separation - combined_width)
            candidates.append(
                (
                    zone_gap,
                    separation,
                    str(left["dependency_group"]),
                    str(right["dependency_group"]),
                    left,
                    right,
                )
            )
        if not candidates:
            continue
        zone_gap, separation, _, _, left, right = min(
            candidates, key=lambda item: item[:4]
        )
        ordered = sorted(
            (left, right), key=lambda item: str(item["dependency_group"])
        )
        first, second = ordered
        record_a: dict[str, Any] = {"date": date}
        record_b: dict[str, Any] = {"date": date}
        for name in group_names:
            record_a[f"{SPECIAL_CLUSTER_A}__group_{name}"] = float(
                first["dependency_group"] == name
            )
            record_b[f"{SPECIAL_CLUSTER_B}__group_{name}"] = float(
                second["dependency_group"] == name
            )
        for column in component_columns:
            record_a[f"{SPECIAL_CLUSTER_A}__{column}"] = float(first[column])
            record_b[f"{SPECIAL_CLUSTER_B}__{column}"] = float(second[column])
        combined_width = float(first["mean_zone_half_width_atr"]) + float(
            second["mean_zone_half_width_atr"]
        )
        record_b[f"{SPECIAL_CLUSTER_B}__separation_atr"] = separation
        record_b[f"{SPECIAL_CLUSTER_B}__zone_gap_atr"] = zone_gap
        record_b[f"{SPECIAL_CLUSTER_B}__overlap_atr"] = max(
            0.0, combined_width - separation
        )
        record_b[f"{SPECIAL_CLUSTER_B}__combined_half_width_atr"] = combined_width
        record_b[f"{SPECIAL_CLUSTER_B}__combined_contact_count"] = float(
            first["event_count"] + second["event_count"]
        )
        rows_a.append(record_a)
        rows_b.append(record_b)
    return DataFrame.from_records(rows_a), DataFrame.from_records(rows_b)


def attach_sparse_stale_placebo(
    features: DataFrame, *, block: str, stale_hours: int = STALE_HOURS
) -> DataFrame:
    columns = [column for column in features if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"Cannot make sparse stale placebo for {block!r}.")
    outputs: list[DataFrame] = []
    for _, period_frame in features.groupby("period", sort=False):
        target = period_frame[["date"]].sort_values("date").copy()
        target["cutoff"] = target["date"] - pd.Timedelta(hours=stale_hours)
        available = complete_block_mask(period_frame, block)
        source = period_frame.loc[available, ["date", *columns]].copy()
        source.rename(columns={"date": "source_date"}, inplace=True)
        if source.empty:
            matched = target.copy()
            matched["source_date"] = pd.NaT
            for column in columns:
                matched[column] = np.nan
        else:
            matched = pd.merge_asof(
                target.sort_values("cutoff"),
                source.sort_values("source_date"),
                left_on="cutoff",
                right_on="source_date",
                direction="backward",
                allow_exact_matches=True,
            )
        matched.rename(
            columns={
                column: f"{block}_placebo__{column.removeprefix(f'{block}__')}"
                for column in columns
            },
            inplace=True,
        )
        outputs.append(matched[["date", "source_date", *(
            f"{block}_placebo__{column.removeprefix(f'{block}__')}"
            for column in columns
        )]])
    return pd.concat(outputs, ignore_index=True).sort_values("date").reset_index(drop=True)


def branch_scope_mask(features: DataFrame, spec: BranchRuntimeSpec) -> Series:
    if spec.scope == "all":
        mask = pd.Series(True, index=features.index)
    elif spec.scope == "different_mechanism_cross_timeframe":
        columns = [
            f"timeframe_{timeframe}__different_mechanism_agreement"
            for timeframe in ("4h", "8h", "1d")
        ]
        mask = features[columns].fillna(0.0).gt(0.0).any(axis=1)
    elif spec.scope in {"generic_prior_range", "volume_profile_explicit_prior_4h"}:
        mask = complete_block_mask(features, spec.component_a_blocks[0])
    elif spec.scope == "independent_level_family_cluster":
        gap = pd.to_numeric(
            features[f"{SPECIAL_CLUSTER_B}__zone_gap_atr"], errors="coerce"
        )
        mask = gap.le(CLUSTER_MAX_ZONE_GAP_ATR)
    else:
        raise ValueError(f"Unknown Generation 7 scope: {spec.scope}")
    all_current = (
        *spec.level_blocks,
        *spec.component_a_blocks,
        *spec.component_b_blocks,
    )
    for block in (*all_current, *stale_blocks(all_current)):
        mask &= complete_block_mask(features, block)
    return mask.fillna(False)


def raw_geometry_columns() -> tuple[str, ...]:
    return (
        "cohort",
        "pair",
        "event_time",
        "period",
        "control",
        "source_timeframe",
        "level_family",
        "level_name",
        "level_price",
        "zone_half_width_atr",
        "base_atr",
        "pre_distance_atr",
        "contact_close_distance_atr",
        "attr_vp_value_area_width_pct",
        "attr_vp_state",
    )


def build_pair_feature_support(
    *,
    cohort: str,
    pair: str,
    feature_dir: Path,
    support_dir: Path,
) -> dict[str, Any]:
    source = g7p.cache_inventory(cohort)[pair]
    old_feature_path = Path(source["feature_path"])
    old_event_path = Path(source["event_path"])
    raw_path = Path(source["source_path"])
    features = pd.read_parquet(old_feature_path)
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="coerce")
    event = pd.read_parquet(old_event_path, columns=["date", "period"])
    event["date"] = pd.to_datetime(event["date"], utc=True, errors="coerce")
    if features["date"].duplicated().any() or event["date"].duplicated().any():
        raise ValueError(f"Duplicate Generation 6 cache dates for {pair}")
    timeline = event.merge(features, on="date", how="inner", validate="one_to_one")
    raw = pd.read_parquet(
        raw_path,
        columns=list(raw_geometry_columns()),
        filters=[("control", "==", "actual")],
    )
    raw.rename(columns={"event_time": "date"}, inplace=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.merge(event[["date"]], on="date", how="inner", validate="many_to_one")
    raw = g6d.causal_deduplicate(
        raw,
        keys=(
            "cohort",
            "pair",
            "date",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )
    prior_range = aggregate_specific_geometry(
        raw.loc[raw["level_family"].eq("generic_prior_range")],
        block=SPECIAL_PRIOR_RANGE,
    )
    vp4h = aggregate_specific_geometry(
        raw.loc[
            raw["level_family"].eq("volume_profile_explicit_prior")
            & raw["source_timeframe"].eq("4h")
        ],
        block=SPECIAL_VP4H,
    )
    cluster_a, cluster_b = aggregate_independent_cluster(raw)
    for block_frame in (prior_range, vp4h, cluster_a, cluster_b):
        timeline = timeline.merge(
            block_frame,
            on="date",
            how="left",
            validate="one_to_one",
        )
    specialty = (SPECIAL_PRIOR_RANGE, SPECIAL_VP4H, SPECIAL_CLUSTER_A, SPECIAL_CLUSTER_B)
    placebo_audit: list[dict[str, Any]] = []
    for block in specialty:
        placebo = attach_sparse_stale_placebo(timeline, block=block)
        source_date = f"placebo_source_date__{block}"
        placebo.rename(columns={"source_date": source_date}, inplace=True)
        timeline = timeline.merge(placebo, on="date", how="left", validate="one_to_one")
        valid = timeline[source_date].notna()
        violation = valid & (
            pd.to_datetime(timeline[source_date], utc=True)
            > timeline["date"] - pd.Timedelta(hours=STALE_HOURS)
        )
        placebo_audit.append(
            {
                "block": block,
                "ready_rows": int(complete_block_mask(timeline, f"{block}_placebo").sum()),
                "future_or_too_recent_source_violations": int(violation.sum()),
            }
        )
    violations = sum(
        int(item["future_or_too_recent_source_violations"]) for item in placebo_audit
    )
    if violations:
        raise ValueError(f"Generation 7 specialty-placebo violation for {pair}: {violations}")
    support = event.copy()
    for branch_id, spec in BRANCH_RUNTIME.items():
        if cohort in spec.cohorts and g7p.BRANCH_SPECS[branch_id].pair_filter(pair):
            support[f"ready__{spec.ready_block}"] = branch_scope_mask(timeline, spec)
    feature_columns = [
        column
        for column in timeline
        if "__" in column and not column.startswith("placebo_source_date__")
    ]
    feature_cache = timeline[["date", *feature_columns]].copy()
    for column in feature_columns:
        feature_cache[column] = pd.to_numeric(
            feature_cache[column], errors="coerce"
        ).astype("float32")
    stem = g0.pair_file_stem(pair)
    feature_path = feature_dir / f"{stem}.parquet"
    support_path = support_dir / f"{stem}.parquet"
    g0.atomic_write_parquet(feature_cache, feature_path)
    g0.atomic_write_parquet(support, support_path)
    support_rows = []
    for column in sorted(item for item in support if item.startswith("ready__")):
        for period, frame in support.groupby("period", observed=True):
            support_rows.append(
                {
                    "pair": pair,
                    "cohort": cohort,
                    "ready_block": column.removeprefix("ready__"),
                    "period": period,
                    "events": int(frame[column].fillna(False).astype(bool).sum()),
                }
            )
    return {
        "pair": pair,
        "cohort": cohort,
        "source_path": str(raw_path),
        "source_sha256": g0.sha256_file(raw_path),
        "g6_feature_path": str(old_feature_path),
        "g6_feature_sha256": g0.sha256_file(old_feature_path),
        "g6_event_path": str(old_event_path),
        "g6_event_sha256": g0.sha256_file(old_event_path),
        "evaluation_path": source["evaluation_path"],
        "evaluation_sha256": source["evaluation_sha256"],
        "feature_path": str(feature_path),
        "feature_sha256": g0.sha256_file(feature_path),
        "support_path": str(support_path),
        "support_sha256": g0.sha256_file(support_path),
        "feature_columns": len(feature_columns),
        "support_rows": support_rows,
        "placebo_audit": placebo_audit,
        "causal_placebo_violations": violations,
        "reaction_outcome_columns_read": False,
    }


def exact_support_decisions(
    cohort: str, inventory: Sequence[dict[str, Any]]
) -> tuple[DataFrame, list[dict[str, Any]]]:
    support = DataFrame.from_records(
        row for item in inventory for row in item["support_rows"]
    )
    decisions: list[dict[str, Any]] = []
    for branch_id, spec in BRANCH_RUNTIME.items():
        if cohort not in spec.cohorts:
            continue
        ready = spec.ready_block
        selected = support.loc[support["ready_block"].eq(ready)]
        periods = (
            g7p.DEVELOPMENT_PERIOD[cohort],
            *g7p.VALIDATION_PERIODS[cohort],
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
                    "minimum_pair_events": (
                        int(positive["events"].min()) if len(positive) else 0
                    ),
                }
            )
        development, *validation = checks
        supported = (
            development["events"] >= 100
            and development["coins"] >= 5
            and all(
                item["events"] >= MIN_AGGREGATE_SCORABLE_ROWS
                and item["coins"] >= 5
                for item in validation
            )
        )
        decisions.append(
            {
                "branch_id": branch_id,
                "cohort": cohort,
                "ready_block": ready,
                "status": (
                    "supported_for_generation7_models"
                    if supported
                    else "parked_outcome_blind_insufficient_exact_common_support"
                ),
                "period_checks": checks,
                "cluster_max_zone_gap_atr": (
                    CLUSTER_MAX_ZONE_GAP_ATR
                    if spec.scope == "independent_level_family_cluster"
                    else None
                ),
                "reaction_outcomes_opened": False,
            }
        )
    return support, decisions


def materialize_pair_event_cache(
    *,
    item: dict[str, Any],
    event_dir: Path,
) -> dict[str, Any]:
    pair = str(item["pair"])
    source_columns = [
        "date",
        "period",
        "pair",
        "cohort",
        "market_group",
        "smart_contract_platform",
        *TARGET_COLUMNS,
    ]
    events = pd.read_parquet(item["g6_event_path"], columns=source_columns)
    events["date"] = pd.to_datetime(events["date"], utc=True, errors="coerce")
    support = pd.read_parquet(item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    ready_columns = [column for column in support if column.startswith("ready__")]
    output = events.merge(
        support[["date", *ready_columns]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    event_path = event_dir / f"{g0.pair_file_stem(pair)}.parquet"
    g0.atomic_write_parquet(output, event_path)
    return {
        **item,
        "event_path": str(event_path),
        "event_sha256": g0.sha256_file(event_path),
        "event_rows": len(output),
        "ready_columns": ready_columns,
        "reaction_outcomes_opened_after_support_freeze": True,
    }


def validate_shared_cache_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "completed_generation7_enriched_shared_cache":
        raise ValueError("Generation 7 shared cache is not terminal.")
    for item in manifest["inventory"]:
        for key in ("feature", "support", "event", "evaluation"):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                raise ValueError(f"Generation 7 shared {key} cache changed: {path}")
        if item["causal_placebo_violations"]:
            raise ValueError(f"Causal placebo violation for {item['pair']}")


def prepare_shared_cache(
    *, cohort: str, pairs: Sequence[str], workers: int
) -> dict[str, Any]:
    record_dir = SHARED_RECORD_ROOT / SHARED_CACHE_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = record_dir / f"{cohort}_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_shared_cache_manifest(manifest)
        if set(manifest["pairs"]) != set(pairs):
            raise ValueError("Generation 7 shared-cache pair universe changed.")
        return manifest
    artifact_dir = SHARED_ARTIFACT_ROOT / cohort
    feature_dir = artifact_dir / "feature_cache"
    support_dir = artifact_dir / "outcome_blind_support_cache"
    event_dir = artifact_dir / "event_cache"
    for path in (feature_dir, support_dir, event_dir):
        path.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                build_pair_feature_support,
                cohort=cohort,
                pair=pair,
                feature_dir=feature_dir,
                support_dir=support_dir,
            ): pair
            for pair in pairs
        }
        for future in as_completed(futures):
            item = future.result()
            inventory.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g7_outcome_blind_feature_cache",
                        "cohort": cohort,
                        "pair": item["pair"],
                        "feature_columns": item["feature_columns"],
                    }
                ),
                flush=True,
            )
    inventory.sort(key=lambda item: pairs.index(str(item["pair"])))
    support, decisions = exact_support_decisions(cohort, inventory)
    support_path = record_dir / f"{cohort}_exact_model_support.csv"
    decisions_path = record_dir / f"{cohort}_exact_model_support_decisions.json"
    g0.atomic_write_csv(support, support_path)
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "cache_id": SHARED_CACHE_ID,
            "cohort": cohort,
            "created_at_utc": g0.utc_now(),
            "status": "frozen_exact_support_before_generation7_outcomes",
            "decisions": decisions,
            "reaction_outcome_columns_read": False,
        },
        decisions_path,
    )
    # Only after the exact decisions above are durable may target columns be opened.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        materialized = list(
            pool.map(
                lambda item: materialize_pair_event_cache(
                    item=item, event_dir=event_dir
                ),
                inventory,
            )
        )
    supported = {
        str(item["branch_id"])
        for item in decisions
        if item["status"] == "supported_for_generation7_models"
    }
    manifest = {
        "schema_version": 1,
        "cache_id": SHARED_CACHE_ID,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation7_enriched_shared_cache",
        "cohort": cohort,
        "pairs": list(pairs),
        "supported_branches": sorted(supported),
        "decisions": decisions,
        "outcome_blind_sequence": {
            "feature_and_support_cache_completed_first": True,
            "exact_support_decisions_written_before_target_read": True,
            "reaction_targets_materialized_only_after_support_freeze": True,
            "support_decisions_path": str(decisions_path),
            "support_decisions_sha256": g0.sha256_file(decisions_path),
        },
        "thresholds_declared_before_model_outcomes": {
            "minimum_pair_scorable_rows": MIN_PAIR_SCORABLE_ROWS,
            "minimum_aggregate_scorable_rows": MIN_AGGREGATE_SCORABLE_ROWS,
            "minimum_training_rows_per_pair": MIN_TRAINING_ROWS,
            "cluster_max_zone_gap_atr": CLUSTER_MAX_ZONE_GAP_ATR,
            "stale_hours": STALE_HOURS,
        },
        "inventory": materialized,
        "source_contracts": {
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": g0.sha256_file(FROZEN_BATCH),
            "outcome_blind_preflight": str(OUTCOME_BLIND_PREFLIGHT),
            "outcome_blind_preflight_sha256": g0.sha256_file(
                OUTCOME_BLIND_PREFLIGHT
            ),
            "cache_builder": str(Path(__file__).resolve()),
            "cache_builder_sha256": g0.sha256_file(Path(__file__)),
        },
        "integrity": {
            "profit_used": False,
            "direction_prediction": False,
            "causal_placebo_violations": int(
                sum(item["causal_placebo_violations"] for item in materialized)
            ),
        },
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def cohort_settings(cohort: str) -> dict[str, Any]:
    return g6f.cohort_settings(cohort)


def allowed_pairs(cohort: str) -> tuple[str, ...]:
    manifest = g6f.source_manifest(cohort)
    return tuple(str(pair) for pair in manifest["data"]["pairs"])


def select_pairs(allowed: Sequence[str], requested: str) -> tuple[str, ...]:
    if requested == "all":
        return tuple(allowed)
    selected = parse_csv(requested)
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: {unknown}")
    return selected


def profiles_for_cohort(cohort: str, supported: set[str]) -> tuple[str, ...]:
    return tuple(
        profile_id
        for profile_id, definition in PROFILES.items()
        if definition["branch_id"] in supported
        and cohort in BRANCH_RUNTIME[definition["branch_id"]].cohorts
    )


def technical_smoke_profiles(cohort: str, supported: set[str]) -> tuple[str, ...]:
    available = profiles_for_cohort(cohort, supported)
    first_branch = "g7a_local_participation_and_btc_activity"
    selected = [
        profile for profile in available if profile.startswith(f"{first_branch}__")
    ]
    special_branches = (
        "g7d_cross_timeframe_agreement_and_local_participation",
        "g7e_prior_range_level_and_local_participation",
        "g7f_prior_volume_profile_area_and_compression",
        "g7g_independent_level_family_cluster",
        "g7i_orderbook_and_btc_activity_at_levels",
        "g7h_meme_eth_activity_and_local_participation",
        "g7j_gdelt_activity_and_local_participation",
    )
    for branch_id in special_branches:
        for role in (
            COMPLETE_ROLE,
            "level_plus_stale_component_a_plus_component_b",
            "level_plus_component_a_plus_stale_component_b",
        ):
            profile_id = f"{branch_id}__{role}"
            if profile_id in available:
                selected.append(profile_id)
    return tuple(dict.fromkeys(selected))


def profile_pairs(
    profile: dict[str, Any], selected_pairs: Sequence[str]
) -> tuple[str, ...]:
    branch_id = str(profile["branch_id"])
    pair_filter = g7p.BRANCH_SPECS[branch_id].pair_filter
    return tuple(pair for pair in selected_pairs if pair_filter(pair))


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    profile: dict[str, Any],
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = g6f.profile_config(
        base,
        identifier=identifier,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        profile=profile,
        train_days=train_days,
        backtest_days=backtest_days,
        technical_smoke=technical_smoke,
    )
    research = config.pop("market_reaction_zone_g6")
    research["target_columns"] = list(profile["targets"])
    research["frozen_branch_id"] = profile["branch_id"]
    research["profile_role"] = profile["role"]
    config["market_reaction_zone_g7"] = research
    return config


def build_manifest(
    *,
    run_id: str,
    cohort: str,
    pairs: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    record_dir: Path,
    artifact_dir: Path,
    shared_cache: dict[str, Any],
    profile_workers: int,
    technical_smoke: bool,
    settings: dict[str, Any],
    timerange: str,
) -> dict[str, Any]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = Path(shared_cache["inventory"][0]["feature_path"]).parent
    event_dir = Path(shared_cache["inventory"][0]["event_path"]).parent
    commands: list[dict[str, Any]] = []
    active_profiles: list[str] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        pair_subset = profile_pairs(definition, pairs)
        if not pair_subset:
            continue
        active_profiles.append(profile_id)
        index = len(active_profiles)
        short_id = f"p{index:03d}_{g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g7-{g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pair_subset,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=definition,
                train_days=int(settings["train_days"]),
                backtest_days=int(settings["backtest_days"]),
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
            STRATEGY_CLASS,
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
                "short_id": short_id,
                "branch_id": definition["branch_id"],
                "role": definition["role"],
                "blocks": list(definition["blocks"]),
                "required_ready_blocks": list(
                    definition["required_ready_blocks"]
                ),
                "targets": list(definition["targets"]),
                "pairs": list(pair_subset),
                "strategy": STRATEGY_CLASS,
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
    active_comparisons = [
        item
        for item in COMPARISONS
        if item["candidate"] in active_profiles and item["baseline"] in active_profiles
    ]
    inventory = [
        item for item in shared_cache["inventory"] if item["pair"] in pairs
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": g0.utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "cohort": cohort,
        "pairs": list(pairs),
        "timerange": timerange,
        "train_period_days": int(settings["train_days"]),
        "backtest_period_days": int(settings["backtest_days"]),
        "validation_periods": list(settings["validation_periods"]),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": active_profiles,
        "profile_count": len(active_profiles),
        "comparisons": active_comparisons,
        "targets": sorted(
            {target for profile in active_profiles for target in PROFILES[profile]["targets"]}
        ),
        "objective": (
            "Test the complete frozen Generation 7 pairwise source interactions against "
            "both simpler components and every causal stale-information control."
        ),
        "decision_clock": (
            "Completion of the one-hour level-contact candle. Every target starts after "
            "that candle and contains reaction magnitude only, never future direction."
        ),
        "research_boundary": {
            "profit_used": False,
            "direction_prediction": False,
            "entry_or_exit_action": False,
            "result_driven_threshold_tuning": False,
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_pair_scorable_rows": MIN_PAIR_SCORABLE_ROWS,
            "minimum_aggregate_scorable_rows": MIN_AGGREGATE_SCORABLE_ROWS,
            "minimum_training_rows_per_pair": MIN_TRAINING_ROWS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "stale_hours": STALE_HOURS,
            "cluster_max_zone_gap_atr": CLUSTER_MAX_ZONE_GAP_ATR,
            "complete_must_beat_all_seven_controls": True,
        },
        "source_contracts": {
            "base_config": str(base_config.resolve()),
            "base_config_sha256": g0.sha256_file(base_config),
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": g0.sha256_file(FROZEN_BATCH),
            "outcome_blind_preflight": str(OUTCOME_BLIND_PREFLIGHT),
            "outcome_blind_preflight_sha256": g0.sha256_file(
                OUTCOME_BLIND_PREFLIGHT
            ),
            "shared_cache_manifest": str(
                SHARED_RECORD_ROOT / SHARED_CACHE_ID / f"{cohort}_manifest.json"
            ),
            "shared_cache_manifest_sha256": g0.sha256_file(
                SHARED_RECORD_ROOT / SHARED_CACHE_ID / f"{cohort}_manifest.json"
            ),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": g0.sha256_file(Path(__file__)),
            "strategy": str(STRATEGY_FILE.resolve()),
            "strategy_sha256": g0.sha256_file(STRATEGY_FILE),
            "cache_inventory": inventory,
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": runtime_snapshot(),
        "commands": commands,
    }


def prepare_run(
    *,
    run_id: str,
    cohort: str,
    pairs: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    cache_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g7_freqai_run_manifest.json"
    if manifest_path.is_file():
        return (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path,
            artifact_dir,
        )
    universe = allowed_pairs(cohort)
    shared_cache = prepare_shared_cache(
        cohort=cohort,
        pairs=universe,
        workers=cache_workers,
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    settings = cohort_settings(cohort)
    timerange = str(settings["timerange"])
    if technical_smoke:
        if cohort == "normal":
            timerange = "20250401-20250701"
            settings = {**settings, "train_days": 365, "backtest_days": 90}
        else:
            timerange = "20260401-20260516"
            settings = {**settings, "train_days": 160, "backtest_days": 45}
    manifest = build_manifest(
        run_id=run_id,
        cohort=cohort,
        pairs=pairs,
        profiles=profiles,
        base_config=base_config,
        python_exe=python_exe,
        record_dir=record_dir,
        artifact_dir=artifact_dir,
        shared_cache=shared_cache,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
        settings=settings,
        timerange=timerange,
    )
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def preflight_run(  # noqa: C901 - runtime and cache gates remain explicit
    manifest: dict[str, Any], *, python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    if not python_exe.is_file():
        problems.append(f"Missing worker interpreter: {python_exe}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 7 strategy: {STRATEGY_FILE}")
    base_config = Path(manifest["source_contracts"]["base_config"])
    if not base_config.is_file():
        problems.append(f"Missing base config: {base_config}")
    elif g0.sha256_file(base_config) != manifest["source_contracts"]["base_config_sha256"]:
        problems.append("The base config changed after run preparation.")
    dependency: dict[str, Any] = {}
    if python_exe.is_file():
        check = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import freqtrade, lightgbm, sklearn; print(freqtrade.__version__)",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dependency = {
            "returncode": check.returncode,
            "stdout": check.stdout.strip(),
            "stderr": check.stderr.strip(),
        }
        if check.returncode:
            problems.append("The worker cannot import Freqtrade and LightGBM.")
    cache_by_pair = {
        str(item["pair"]): item
        for item in manifest["source_contracts"]["cache_inventory"]
    }
    cache_audit = []
    for pair, item in cache_by_pair.items():
        valid = True
        for key in ("feature", "event", "evaluation"):
            path = Path(item[f"{key}_path"])
            valid &= path.is_file() and g0.sha256_file(path) == item[f"{key}_sha256"]
        if not valid:
            problems.append(f"Cache hash mismatch for {pair}")
        cache_audit.append({"pair": pair, "hashes_valid": bool(valid)})
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    readiness_audit: list[dict[str, Any]] = []
    requirements = {
        (
            tuple(item["pairs"]),
            tuple(item["required_ready_blocks"]),
            tuple(item["targets"]),
        )
        for item in manifest["commands"]
    }
    for pair_subset, ready_blocks, targets in sorted(requirements):
        for pair in pair_subset:
            path = event_dir / f"{g0.pair_file_stem(pair)}.parquet"
            ready_columns = [f"ready__{block}" for block in ready_blocks]
            frame = pd.read_parquet(
                path,
                columns=["date", "period", *targets, *ready_columns],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            eligible = frame[ready_columns].fillna(False).astype(bool).all(axis=1)
            eligible &= frame[list(targets)].notna().all(axis=1)
            training_rows = int(
                (
                    eligible
                    & frame["date"].ge(training_start)
                    & frame["date"].lt(prediction_start)
                ).sum()
            )
            prediction_rows = int(
                (
                    eligible
                    & frame["date"].ge(prediction_start)
                    & frame["date"].lt(prediction_end)
                ).sum()
            )
            period_counts = {
                period: int((eligible & frame["period"].eq(period)).sum())
                for period in manifest["validation_periods"]
            }
            readiness_audit.append(
                {
                    "pair": pair,
                    "ready_blocks": ",".join(ready_blocks),
                    "targets": ",".join(targets),
                    "training_rows_before_first_prediction": training_rows,
                    "prediction_timerange_rows": prediction_rows,
                    "validation_period_rows": json.dumps(period_counts, sort_keys=True),
                }
            )
            if training_rows < MIN_TRAINING_ROWS:
                problems.append(
                    f"{pair} has only {training_rows} training rows for {ready_blocks}."
                )
            if prediction_rows < MIN_PAIR_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {prediction_rows} prediction rows for {ready_blocks}."
                )
    free_gib = shutil.disk_usage(Path(manifest["storage"]["bulky_artifact_dir"])).free / (
        1024**3
    )
    if free_gib < 20.0:
        problems.append(f"Only {free_gib:.2f} GiB free on the artifact drive.")
    return {
        "created_at_utc": g0.utc_now(),
        "passed": not problems,
        "problems": problems,
        "dependency_check": dependency,
        "cache_audit": cache_audit,
        "readiness_audit": readiness_audit,
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", g0.utc_now())
    g0.atomic_write_json(manifest, manifest_path)
    items_by_id = {item["profile_id"]: item for item in manifest["commands"]}
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    workers = int(manifest["profile_workers"])
    processed = len(manifest["commands"]) - len(pending)
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        for item in batch:
            item["status"] = "running"
            item["started_at_utc"] = g0.utc_now()
        g0.atomic_write_json(manifest, manifest_path)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_freqai_profile, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                item = items_by_id[result["profile_id"]]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                g0.atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g7_freqai_profiles",
                            "processed": processed,
                            "total": len(manifest["commands"]),
                            "profile": item["profile_id"],
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
            manifest["finished_at_utc"] = g0.utc_now()
            g0.atomic_write_json(manifest, manifest_path)
            return int(failures[0]["returncode"])
    return 0


def load_event_targets(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    for pair in manifest["pairs"]:
        path = event_dir / f"{g0.pair_file_stem(pair)}.parquet"
        frame = pd.read_parquet(path, columns=["date", "period", *TARGET_COLUMNS])
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date keys in Generation 7 targets.")
    return output


def load_evaluation_rows(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    for item in manifest["source_contracts"]["cache_inventory"]:
        frame = pd.read_parquet(item["evaluation_path"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def regression_metrics(
    frame: DataFrame,
    *,
    target: str,
    candidate_column: str,
    baseline_column: str,
) -> dict[str, Any]:
    actual_column = f"{target}__actual"
    numeric = frame[[actual_column, candidate_column, baseline_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.loc[np.isfinite(numeric).all(axis=1)]
    if len(values) < MIN_PAIR_SCORABLE_ROWS:
        return {
            "rows": len(values),
            "candidate_mae": np.nan,
            "baseline_mae": np.nan,
            "paired_mae_gain": np.nan,
            "relative_mae_gain": np.nan,
            "candidate_spearman": np.nan,
            "baseline_spearman": np.nan,
            "spearman_change": np.nan,
        }
    actual = values[actual_column]
    candidate = values[candidate_column]
    baseline = values[baseline_column]
    candidate_mae = float((actual - candidate).abs().mean())
    baseline_mae = float((actual - baseline).abs().mean())
    candidate_rank = float(candidate.corr(actual, method="spearman"))
    baseline_rank = float(baseline.corr(actual, method="spearman"))
    return {
        "rows": len(values),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "paired_mae_gain": baseline_mae - candidate_mae,
        "relative_mae_gain": (
            (baseline_mae - candidate_mae) / baseline_mae
            if baseline_mae > 0.0
            else np.nan
        ),
        "candidate_spearman": candidate_rank,
        "baseline_spearman": baseline_rank,
        "spearman_change": candidate_rank - baseline_rank,
    }


def fair_comparison_frame(
    *,
    candidate: DataFrame,
    baseline: DataFrame,
    actual: DataFrame,
    targets: Sequence[str],
) -> tuple[DataFrame, dict[str, Any]]:
    candidate_keys = candidate[["pair", "date"]].drop_duplicates()
    baseline_keys = baseline[["pair", "date"]].drop_duplicates()
    common = candidate_keys.merge(
        baseline_keys,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    )
    candidate_selected = candidate[["pair", "date", *targets]].merge(
        common, on=["pair", "date"], how="inner", validate="one_to_one"
    )
    baseline_selected = baseline[["pair", "date", *targets]].merge(
        common, on=["pair", "date"], how="inner", validate="one_to_one"
    )
    candidate_selected.rename(
        columns={target: f"{target}__candidate" for target in targets},
        inplace=True,
    )
    baseline_selected.rename(
        columns={target: f"{target}__baseline" for target in targets},
        inplace=True,
    )
    actual_selected = actual[["pair", "date", "period", *targets]].rename(
        columns={target: f"{target}__actual" for target in targets}
    )
    fair = actual_selected.merge(
        candidate_selected,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    ).merge(
        baseline_selected,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    )
    return fair, {
        "candidate_prediction_rows": len(candidate_keys),
        "baseline_prediction_rows": len(baseline_keys),
        "common_prediction_rows": len(common),
        "candidate_only_rows": len(candidate_keys) - len(common),
        "baseline_only_rows": len(baseline_keys) - len(common),
        "fair_labelled_rows": len(fair),
        "identical_prediction_keys": (
            len(candidate_keys) == len(common) == len(baseline_keys)
        ),
    }


def group_definitions(cohort: str, available_pairs: Sequence[str]) -> list[dict[str, Any]]:
    available = set(available_pairs)
    if cohort == "meme":
        members = tuple(
            pair for pair in g6.GROUPS["frozen_top_ten_memes"] if pair in available
        )
        return [
            {
                "group_id": "frozen_top_ten_memes",
                "members": members,
                "minimum_positive_coins": 5,
                "dominance_limit": 0.60,
            }
        ]
    definitions = [
        {
            "group_id": "full_normal_cohort",
            "members": tuple(pair for pair in available_pairs if pair in available),
            "minimum_positive_coins": 5,
            "dominance_limit": 0.60,
        },
        {
            "group_id": "btc_separate",
            "members": tuple(
                pair for pair in g6.GROUPS["btc_separate"] if pair in available
            ),
            "minimum_positive_coins": 1,
            "dominance_limit": None,
        },
        {
            "group_id": "established_altcoins",
            "members": tuple(
                pair for pair in g6.GROUPS["established_altcoins"] if pair in available
            ),
            "minimum_positive_coins": 5,
            "dominance_limit": 0.60,
        },
        {
            "group_id": "smart_contract_platforms",
            "members": tuple(
                pair
                for pair in g6.GROUPS["smart_contract_platforms"]
                if pair in available
            ),
            "minimum_positive_coins": 3,
            "dominance_limit": 0.60,
        },
    ]
    return [item for item in definitions if item["members"]]


def group_comparison_row(
    frame: DataFrame,
    *,
    manifest: dict[str, Any],
    definition: dict[str, Any],
    target: str,
    period: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    actual_column = f"{target}__actual"
    candidate_column = f"{target}__candidate"
    baseline_column = f"{target}__baseline"
    selected = frame.loc[
        frame["pair"].isin(group["members"]) & frame["period"].eq(period),
        ["pair", "date", actual_column, candidate_column, baseline_column],
    ].copy()
    numeric = selected[[actual_column, candidate_column, baseline_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    selected = selected.loc[np.isfinite(numeric).all(axis=1)].copy()
    selected["paired_error_gain"] = (
        selected[actual_column] - selected[baseline_column]
    ).abs() - (
        selected[actual_column] - selected[candidate_column]
    ).abs()
    coin_rows = selected.groupby("pair", observed=True).size()
    scorable_coins = tuple(
        str(pair)
        for pair in coin_rows.loc[coin_rows.ge(MIN_PAIR_SCORABLE_ROWS)].index
    )
    selected = selected.loc[selected["pair"].isin(scorable_coins)]
    coin_gain = selected.groupby("pair", observed=True)["paired_error_gain"].mean()
    point, lower, upper = deterministic_block_bootstrap(
        selected[["pair", "date", "paired_error_gain"]],
        seed_key=(
            f"g7|{manifest['run_id']}|{definition['comparison_id']}|"
            f"{target}|{period}|{group['group_id']}"
        ),
        samples=BOOTSTRAP_SAMPLES,
    )
    absolute_sum = float(coin_gain.abs().sum()) if len(coin_gain) else 0.0
    largest_share = (
        float(coin_gain.abs().max() / absolute_sum) if absolute_sum > 0.0 else np.nan
    )
    enough_rows = len(selected) >= MIN_AGGREGATE_SCORABLE_ROWS
    enough_coins = int(coin_gain.gt(0.0).sum()) >= int(
        group["minimum_positive_coins"]
    )
    dominance_limit = group["dominance_limit"]
    not_dominated = (
        True
        if dominance_limit is None
        else bool(np.isfinite(largest_share) and largest_share <= dominance_limit)
    )
    point_positive = bool(np.isfinite(point) and point > 0.0)
    lower_positive = bool(np.isfinite(lower) and lower > 0.0)
    strict = enough_rows and enough_coins and not_dominated and lower_positive
    provisional = enough_rows and enough_coins and not_dominated and point_positive
    return {
        **definition,
        "target": target,
        "period": period,
        "group_id": group["group_id"],
        "group_members": ",".join(group["members"]),
        "rows": len(selected),
        "scorable_coins": len(coin_gain),
        "positive_coins": int(coin_gain.gt(0.0).sum()),
        "negative_coins": int(coin_gain.lt(0.0).sum()),
        "minimum_positive_coins": int(group["minimum_positive_coins"]),
        "equal_coin_paired_mae_gain": point,
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
        "largest_absolute_coin_share": largest_share,
        "not_dominated_by_one_coin": not_dominated,
        "point_positive": point_positive,
        "bootstrap_lower_positive": lower_positive,
        "strict_period_pass": strict,
        "provisional_period_pass": provisional,
    }


def score_comparisons(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
    evaluation: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for definition in manifest["comparisons"]:
        candidate_id = str(definition["candidate"])
        baseline_id = str(definition["baseline"])
        targets = tuple(str(target) for target in definition["targets"])
        fair, audit = fair_comparison_frame(
            candidate=predictions[candidate_id],
            baseline=predictions[baseline_id],
            actual=actual,
            targets=targets,
        )
        eligibility_rows.append({**definition, **audit})
        for (pair, period), frame in fair.groupby(
            ["pair", "period"], sort=False, observed=True
        ):
            for target in targets:
                pair_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "target": target,
                        **regression_metrics(
                            frame,
                            target=target,
                            candidate_column=f"{target}__candidate",
                            baseline_column=f"{target}__baseline",
                        ),
                    }
                )
        comparison_pairs = sorted(fair["pair"].unique())
        for group in group_definitions(str(manifest["cohort"]), comparison_pairs):
            for period in manifest["validation_periods"]:
                for target in targets:
                    group_rows.append(
                        group_comparison_row(
                            fair,
                            manifest=manifest,
                            definition=definition,
                            target=target,
                            period=period,
                            group=group,
                        )
                    )
        candidate_predictions = predictions[candidate_id][
            ["pair", "date", *targets]
        ].rename(columns={target: f"{target}__candidate" for target in targets})
        baseline_predictions = predictions[baseline_id][
            ["pair", "date", *targets]
        ].rename(columns={target: f"{target}__baseline" for target in targets})
        detailed = evaluation.merge(
            candidate_predictions,
            on=["pair", "date"],
            how="inner",
            validate="many_to_one",
        ).merge(
            baseline_predictions,
            on=["pair", "date"],
            how="inner",
            validate="many_to_one",
        )
        detailed.rename(
            columns={target: f"{target}__actual" for target in targets},
            inplace=True,
        )
        for key, frame in detailed.groupby(
            ["pair", "period", "level_family", "source_timeframe"],
            sort=False,
            observed=True,
        ):
            pair, period, family, timeframe = key
            for target in targets:
                family_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "level_family": family,
                        "source_timeframe": timeframe,
                        "target": target,
                        **regression_metrics(
                            frame,
                            target=target,
                            candidate_column=f"{target}__candidate",
                            baseline_column=f"{target}__baseline",
                        ),
                        "dependence_warning": (
                            "Descriptive slice; multiple level rows can share one market path."
                        ),
                    }
                )
    return (
        DataFrame.from_records(pair_rows),
        DataFrame.from_records(group_rows),
        DataFrame.from_records(eligibility_rows),
        DataFrame.from_records(family_rows),
    )


def branch_target_group_decisions(
    manifest: dict[str, Any], group_scores: DataFrame
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    expected_comparisons = 7
    expected_periods = set(manifest["validation_periods"])
    keys = ["branch_id", "target", "group_id", "group_members"]
    for key, frame in group_scores.groupby(keys, dropna=False):
        branch_id, target, group_id, members = key
        comparison_count = int(frame["comparison_id"].nunique())
        complete_periods = all(
            set(group["period"]) == expected_periods
            for _, group in frame.groupby("comparison_id", observed=True)
        )
        strict = (
            comparison_count == expected_comparisons
            and complete_periods
            and frame["strict_period_pass"].fillna(False).astype(bool).all()
        )
        provisional = (
            comparison_count == expected_comparisons
            and complete_periods
            and frame["provisional_period_pass"].fillna(False).astype(bool).all()
        )
        if strict:
            status = "control_resistant_complete_interaction"
        elif provisional:
            status = "provisional_complete_interaction"
        else:
            status = "complete_interaction_not_reproduced"
        failed = frame.loc[
            ~frame["strict_period_pass"].fillna(False).astype(bool),
            [
                "comparison_id",
                "baseline_role",
                "period",
                "rows",
                "positive_coins",
                "equal_coin_paired_mae_gain",
                "bootstrap_lower",
                "not_dominated_by_one_coin",
            ],
        ]
        rows.append(
            {
                "branch_id": branch_id,
                "plain_name": FROZEN_BRANCHES[str(branch_id)]["plain_name"],
                "target": target,
                "group_id": group_id,
                "group_members": members,
                "status": status,
                "all_seven_controls_present": comparison_count == expected_comparisons,
                "both_validation_periods_present": complete_periods,
                "all_seven_controls_strict": strict,
                "all_seven_controls_point_positive": provisional,
                "failed_strict_checks": json.dumps(
                    failed.to_dict("records"),
                    sort_keys=True,
                    default=g0.json_default,
                ),
                "meaning": (
                    "Reaction-magnitude estimation only; not causation, direction, profit, "
                    "entry, exit, or strategy promotion."
                ),
            }
        )
    return DataFrame.from_records(rows)


def branch_target_decisions(group_decisions: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (branch_id, target), frame in group_decisions.groupby(
        ["branch_id", "target"], dropna=False
    ):
        strict = frame.loc[
            frame["status"].eq("control_resistant_complete_interaction"), "group_id"
        ].tolist()
        provisional = frame.loc[
            frame["status"].eq("provisional_complete_interaction"), "group_id"
        ].tolist()
        if strict:
            status = "control_resistant_complete_interaction"
        elif provisional:
            status = "provisional_complete_interaction"
        else:
            status = "complete_interaction_not_reproduced"
        rows.append(
            {
                "branch_id": branch_id,
                "plain_name": FROZEN_BRANCHES[str(branch_id)]["plain_name"],
                "target": target,
                "status": status,
                "strict_groups": ",".join(strict),
                "provisional_groups": ",".join(provisional),
                "groups_checked": ",".join(frame["group_id"].astype(str)),
                "decision_rule": (
                    "Retain only if one predeclared group beats all seven simpler/stale "
                    "controls in both validation periods with positive uncertainty bounds."
                ),
            }
        )
    return DataFrame.from_records(rows)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def score_run(
    manifest: dict[str, Any], *, record_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    prediction_audit: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        profile_id = str(item["profile_id"])
        pairs = tuple(str(pair) for pair in item["pairs"])
        frame, audit = load_predictions(Path(item["model_dir"]), pairs)
        audit["profile_id"] = profile_id
        prediction_audit.append(audit)
        if frame.empty:
            raise ValueError(f"No predictions for Generation 7 profile {profile_id}")
        missing = sorted(set(item["targets"]).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {profile_id} lacks targets: {missing}")
        predictions[profile_id] = frame
    actual = load_event_targets(manifest)
    evaluation = load_evaluation_rows(manifest)
    pair_scores, group_scores, eligibility, family = score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
        evaluation=evaluation,
    )
    group_decisions = branch_target_group_decisions(manifest, group_scores)
    decisions = branch_target_decisions(group_decisions)
    record_dir.mkdir(parents=True, exist_ok=True)
    score_paths = {
        "prediction_audit": record_dir / "g7_freqai_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g7_freqai_comparison_eligibility.csv",
        "pair_scores": record_dir / "g7_freqai_pair_scores.csv",
        "group_scores": record_dir / "g7_freqai_group_scores.csv",
        "group_decisions": record_dir / "g7_freqai_group_decisions.csv",
        "branch_target_decisions": record_dir / "g7_freqai_branch_target_decisions.csv",
        "family_slices": artifact_dir / "g7_freqai_level_family_slices.parquet",
    }
    g0.atomic_write_csv(DataFrame.from_records(prediction_audit), score_paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, score_paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, score_paths["pair_scores"])
    g0.atomic_write_csv(group_scores, score_paths["group_scores"])
    g0.atomic_write_csv(group_decisions, score_paths["group_decisions"])
    g0.atomic_write_csv(decisions, score_paths["branch_target_decisions"])
    g0.atomic_write_parquet(family, score_paths["family_slices"])
    unequal = int((~eligibility["identical_prediction_keys"]).sum()) if len(eligibility) else 0
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": "completed_generation7_freqai_pairwise_interactions",
        "created_at_utc": g0.utc_now(),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "branches_completed": int(decisions["branch_id"].nunique()) if len(decisions) else 0,
        "branch_targets_completed": len(decisions),
        "control_resistant_branch_targets": int(
            decisions["status"].eq("control_resistant_complete_interaction").sum()
        ),
        "provisional_branch_targets": int(
            decisions["status"].eq("provisional_complete_interaction").sum()
        ),
        "not_reproduced_branch_targets": int(
            decisions["status"].eq("complete_interaction_not_reproduced").sum()
        ),
        "integrity": {
            "unequal_prediction_key_comparisons": unequal,
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in prediction_audit)
            ),
            "profit_used": False,
            "direction_prediction": False,
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "all_complete_models_challenged_by_seven_controls": bool(
                len(group_decisions)
                and group_decisions["all_seven_controls_present"].astype(bool).all()
            ),
        },
        "artifacts": {key: artifact_record(path) for key, path in score_paths.items()},
        "interpretation_boundary": (
            "These results estimate direction-neutral reaction magnitude. They do not "
            "establish causation, direction, profitability, or a trading rule."
        ),
    }
    result_path = record_dir / "g7_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def validate_authorization_contract() -> None:
    if not OUTCOME_BLIND_PREFLIGHT.is_file():
        raise FileNotFoundError(OUTCOME_BLIND_PREFLIGHT)
    preflight = json.loads(OUTCOME_BLIND_PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "completed_generation7_outcome_blind_preflight":
        raise ValueError("Generation 7 outcome-blind preflight is incomplete.")
    if preflight.get("supported_branch_cohort_cells") != 18:
        raise ValueError("All 18 frozen branch/cohort cells did not pass preflight.")
    integrity = preflight.get("integrity", {})
    if integrity.get("reaction_outcome_columns_read") is not False:
        raise ValueError("The Generation 7 preflight was not outcome-blind.")
    if integrity.get("profit_used") is not False:
        raise ValueError("Profit entered the Generation 7 preflight.")
    if integrity.get("direction_prediction") is not False:
        raise ValueError("Direction entered the Generation 7 preflight.")


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Generation 7 pairwise FreqAI reaction-magnitude interactions "
            "against simpler and causal stale controls."
        )
    )
    parser.add_argument("--cohort", choices=("normal", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
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
    args = parser.parse_args(argv)
    for label, value in (
        ("profile-workers", args.profile_workers),
        ("cache-workers", args.cache_workers),
    ):
        if value < 1 or value > MAX_WORKERS:
            raise ValueError(f"{label} must be between 1 and {MAX_WORKERS}.")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    validate_authorization_contract()
    universe = allowed_pairs(args.cohort)
    pairs = select_pairs(universe, args.pairs)
    shared = prepare_shared_cache(
        cohort=args.cohort,
        pairs=universe,
        workers=args.cache_workers,
    )
    supported = set(str(branch) for branch in shared["supported_branches"])
    available_profiles = profiles_for_cohort(args.cohort, supported)
    if args.profiles == "all":
        profiles = available_profiles
    elif args.profiles == "smoke":
        profiles = technical_smoke_profiles(args.cohort, supported)
    else:
        profiles = parse_csv(args.profiles)
    unknown_profiles = sorted(set(profiles).difference(available_profiles))
    if not profiles or unknown_profiles:
        raise ValueError(f"Invalid Generation 7 profiles: {unknown_profiles}")
    manifest, manifest_path, artifact_dir = prepare_run(
        run_id=args.run_id,
        cohort=args.cohort,
        pairs=pairs,
        profiles=profiles,
        base_config=args.base_config,
        python_exe=args.python,
        profile_workers=args.profile_workers,
        cache_workers=args.cache_workers,
        technical_smoke=args.technical_smoke,
    )
    record_dir = manifest_path.parent
    preflight = preflight_run(manifest, python_exe=args.python)
    preflight_path = record_dir / "g7_freqai_launch_preflight.json"
    g0.atomic_write_json(preflight, preflight_path)
    manifest["launch_preflight"] = artifact_record(preflight_path)
    g0.atomic_write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "phase": "g7_freqai_launch_preflight",
                "passed": preflight["passed"],
                "profiles": len(manifest["commands"]),
                "problems": preflight["problems"],
            }
        ),
        flush=True,
    )
    if not preflight["passed"]:
        manifest["status"] = "preflight_failed"
        g0.atomic_write_json(manifest, manifest_path)
        return 2
    if args.prepare_only:
        return 0
    if args.retry_failed:
        for item in manifest["commands"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
        manifest.pop("failed_profile_ids", None)
        manifest["status"] = "prepared"
        g0.atomic_write_json(manifest, manifest_path)
    elif manifest.get("status") in {"failed", "preflight_failed"}:
        raise ValueError("Use --retry-failed only after diagnosing a failed profile.")
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
        raise ValueError(f"Cannot score incomplete Generation 7 profiles: {incomplete}")
    result = score_run(manifest, record_dir=record_dir, artifact_dir=artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = g0.utc_now()
    manifest["result"] = result
    g0.atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
