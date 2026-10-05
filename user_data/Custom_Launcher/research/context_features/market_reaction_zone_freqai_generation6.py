from __future__ import annotations

# Bound numerical libraries before importing pandas/FreqAI helpers.
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
from collections.abc import Callable, Sequence
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
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (  # noqa: E501
    runtime_snapshot,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation5 import (  # noqa: E501
    deterministic_block_bootstrap,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration6Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration6Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG6ConfigurableFreqAIResearchStrategy"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
EVENT_MANIFEST = (
    OUTPUT_ROOT
    / "generation6_branches"
    / "g6_common_event_cache"
    / "g6_common_event_full_20260821a"
    / "manifest.json"
)
DIRECT_RESULT = (
    OUTPUT_ROOT
    / "generation6_branches"
    / "g6_direct_screen"
    / "g6_direct_full_20260821a"
    / "g6_direct_screen_result.json"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation5_review" / "g6_frozen_branch_batch.json"
NORMAL_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_normal_source_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_meme_source_manifest.json"
RECORD_ROOT = OUTPUT_ROOT / "generation6_branches" / "g6_freqai_source_ladders"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation6_branches"
    / "g6_freqai_source_ladders"
)

MAX_WORKERS = 4
STALE_HOURS = 72
MIN_SCORABLE_ROWS = 30
BOOTSTRAP_SAMPLES = 512
TARGET_SOURCE_COLUMNS = {
    TARGET_COLUMNS[0]: "volume_ratio_h1",
    TARGET_COLUMNS[1]: "range_ratio_h1",
    TARGET_COLUMNS[2]: "abs_excursion_atr_h4",
    TARGET_COLUMNS[3]: "absolute_pressure_change_h1",
    TARGET_COLUMNS[4]: "dwell_fraction_h4",
}
TARGET_MEANINGS = {
    TARGET_COLUMNS[0]: (
        "Average volume in the next completed hour divided by typical pre-contact volume."
    ),
    TARGET_COLUMNS[1]: (
        "Next-hour high-low range divided by the typical range known at contact."
    ),
    TARGET_COLUMNS[2]: (
        "Largest absolute four-hour movement around the contacted level in prior ATR units; "
        "it contains no up/down prediction."
    ),
    TARGET_COLUMNS[3]: (
        "Magnitude of the next-hour buying/selling-pressure change, ignoring its direction."
    ),
    TARGET_COLUMNS[4]: (
        "Fraction of the next four closes that remain inside the contacted zone."
    ),
}


STATE_GROUPS = {
    "ohlcv_volume_pressure": (
        "relative_volume",
        "volume_acceleration_magnitude",
        "absolute_pressure",
        "pressure_persistence",
    ),
    "ohlcv_volatility_range": (
        "atr_fraction",
        "prior_range_atr",
        "bollinger_width",
        "range_contraction_ratio",
    ),
    "ohlcv_trend_momentum": (
        "ema20_slope_magnitude",
        "ma_separation",
        "return_slope_magnitude",
        "return_acceleration_magnitude",
        "adx14",
        "rsi_distance_from_50",
        "rsi_change_magnitude",
        "macd_histogram_magnitude",
        "macd_change_magnitude",
    ),
}
CROSS_MARKET_GROUPS = {
    "cross_market_btc": (
        "btc_activity_1h",
        "btc_activity_4h",
        "btc_activity_24h",
        "btc_relative_volume",
    ),
    "cross_market_eth": (
        "eth_activity_1h",
        "eth_activity_4h",
    ),
    "cross_market_cohort_relative": (
        "cohort_breadth_extremity",
        "cohort_dispersion",
        "cohort_absolute_activity",
        "pair_btc_relative_activity_1h",
        "pair_btc_relative_activity_24h",
        "pair_btc_decoupling",
    ),
}
NEWS_GROUPS = {
    "news_gdelt": (
        "gdelt_activity",
        "gdelt_tone_magnitude",
        "gdelt_conflict_intensity",
    ),
}
PARKED_SOURCE_BLOCKS = {
    "news_topics": (
        "The development-only threshold audit found three topic groups constant and "
        "one entirely missing in both cohorts, so FreqAI cannot learn a topic-state "
        "relationship from the current representation."
    ),
    "global_macro": (
        "The outcome-blind coverage audit found only one short May-June 2026 block, "
        "which cannot support both chronological validation slices."
    ),
    "live_pair_local_orderbook": (
        "The frozen live export is stale and too short. Historical BTC orderbook state "
        "remains available as BTC-pair-local and BTC-wide context only."
    ),
}
TIMEFRAME_GROUPS = {
    f"timeframe_{timeframe}": tuple(
        column
        for column in g6d.RELATIONSHIP_COLUMNS
        if column.startswith(f"rel__{timeframe}__")
    )
    for timeframe in ("4h", "8h", "1d")
}
ORDERBOOK_GROUPS = {"orderbook_btc": tuple(g6d.ORDERBOOK_FEATURES)}
SOURCE_BLOCKS: dict[str, dict[str, Any]] = {
    **{
        block: {
            "lane": "recent_ohlcv_and_standard_indicator_state",
            "plain_name": block.replace("_", " "),
        }
        for block in STATE_GROUPS
    },
    **{
        block: {
            "lane": "timeframe_relationships",
            "plain_name": block.replace("_", " "),
        }
        for block in TIMEFRAME_GROUPS
    },
    **{
        block: {
            "lane": "cross_market_and_global_state",
            "plain_name": block.replace("_", " "),
        }
        for block in CROSS_MARKET_GROUPS
    },
    **{
        block: {
            "lane": "orderbook_state",
            "plain_name": "historical BTC orderbook state",
        }
        for block in ORDERBOOK_GROUPS
    },
    **{
        block: {
            "lane": "news_media_and_web_context",
            "plain_name": block.replace("_", " "),
        }
        for block in NEWS_GROUPS
    },
}


def build_profile_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    core_ready = ("state", "state_placebo", "level", "level_placebo")
    core_profiles = {
        "core__market_state": ("state",),
        "core__market_state_placebo": ("state_placebo",),
        "core__level_only": ("level",),
        "core__level_placebo": ("level_placebo",),
        "core__market_plus_level": ("state", "level"),
        "core__market_plus_level_placebo": ("state", "level_placebo"),
        "core__market_placebo_plus_level": ("state_placebo", "level"),
    }
    for profile_id, blocks in core_profiles.items():
        profiles[profile_id] = {
            "surface": "core",
            "blocks": blocks,
            "required_ready_blocks": core_ready,
            "plain_role": profile_id.replace("core__", "").replace("_", " "),
        }
    comparisons.extend(
        [
            comparison(
                "calculated_areas__level_adds_to_market",
                "calculated_price_areas",
                "core__market_plus_level",
                "core__market_state",
                "Does current level information improve reaction estimates beyond market state?",
            ),
            comparison(
                "calculated_areas__current_beats_stale_level",
                "calculated_price_areas",
                "core__market_plus_level",
                "core__market_plus_level_placebo",
                "Does the current level beat an otherwise identical 72-hour-stale level block?",
            ),
            comparison(
                "calculated_areas__level_only_beats_stale",
                "calculated_price_areas",
                "core__level_only",
                "core__level_placebo",
                "Does current level geometry rank reactions better than stale geometry alone?",
            ),
            comparison(
                "ohlcv_state__market_adds_to_level",
                "recent_ohlcv_and_standard_indicator_state",
                "core__market_plus_level",
                "core__level_only",
                "Does recent market state improve reaction estimates beyond level information?",
            ),
            comparison(
                "ohlcv_state__current_beats_stale_state",
                "recent_ohlcv_and_standard_indicator_state",
                "core__market_plus_level",
                "core__market_placebo_plus_level",
                "Does current market state beat a 72-hour-stale state at the same levels?",
            ),
            comparison(
                "ohlcv_state__state_only_beats_stale",
                "recent_ohlcv_and_standard_indicator_state",
                "core__market_state",
                "core__market_state_placebo",
                "Does current market state rank reactions better than stale market state alone?",
            ),
        ]
    )
    for source, definition in SOURCE_BLOCKS.items():
        stale = f"{source}_placebo"
        ready = ("state", "level", "level_placebo", source, stale)
        source_profiles = {
            f"{source}__market_control": ("state",),
            f"{source}__level_control": ("level",),
            f"{source}__source_only": (source,),
            f"{source}__source_placebo": (stale,),
            f"{source}__level_plus_source": ("level", source),
            f"{source}__level_plus_source_placebo": ("level", stale),
            f"{source}__level_placebo_plus_source": ("level_placebo", source),
        }
        for profile_id, blocks in source_profiles.items():
            profiles[profile_id] = {
                "surface": source,
                "blocks": blocks,
                "required_ready_blocks": ready,
                "plain_role": profile_id.replace(f"{source}__", "").replace("_", " "),
            }
        combined = f"{source}__level_plus_source"
        source_only = f"{source}__source_only"
        lane = str(definition["lane"])
        plain = str(definition["plain_name"])
        comparisons.extend(
            [
                comparison(
                    f"{source}__source_vs_market",
                    lane,
                    source_only,
                    f"{source}__market_control",
                    f"Does {plain} estimate reactions better than recent market state?",
                ),
                comparison(
                    f"{source}__source_adds_to_level",
                    lane,
                    combined,
                    f"{source}__level_control",
                    f"Does {plain} add information beyond level geometry?",
                ),
                comparison(
                    f"{source}__level_adds_to_source",
                    lane,
                    combined,
                    source_only,
                    f"Does level geometry add information beyond {plain}?",
                ),
                comparison(
                    f"{source}__combined_beats_stale_source",
                    lane,
                    combined,
                    f"{source}__level_plus_source_placebo",
                    f"Does current {plain} beat its 72-hour-stale version at the same levels?",
                ),
                comparison(
                    f"{source}__combined_beats_stale_level",
                    lane,
                    combined,
                    f"{source}__level_placebo_plus_source",
                    f"Do current levels beat stale levels with the same {plain}?",
                ),
                comparison(
                    f"{source}__source_only_beats_stale",
                    lane,
                    source_only,
                    f"{source}__source_placebo",
                    f"Does current {plain} beat stale {plain} without level information?",
                ),
            ]
        )
    return profiles, comparisons


def comparison(
    comparison_id: str,
    lane: str,
    candidate: str,
    baseline: str,
    question: str,
) -> dict[str, str]:
    return {
        "comparison_id": comparison_id,
        "lane": lane,
        "candidate": candidate,
        "baseline": baseline,
        "question": question,
    }


PROFILES, COMPARISONS = build_profile_registry()


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def event_task_map(event_manifest: dict[str, Any], cohort: str) -> dict[str, dict[str, Any]]:
    return {
        str(item["pair"]): item
        for item in event_manifest["tasks"]
        if item["cohort"] == cohort
    }


def validate_generation_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (EVENT_MANIFEST, DIRECT_RESULT, FROZEN_BATCH):
        if not path.is_file():
            raise FileNotFoundError(path)
    event_manifest = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    if event_manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("The Generation 6 causal event cache is incomplete.")
    if event_manifest.get("summary", {}).get("causal_timestamp_violations") != 0:
        raise ValueError("The Generation 6 event cache contains timestamp violations.")
    if direct.get("status") != "completed_all_seven_sibling_direct_screens":
        raise ValueError("All seven direct sibling screens must finish before FreqAI.")
    if direct.get("summary", {}).get("branches_completed") != 7:
        raise ValueError("The direct sibling portfolio is incomplete.")
    if direct.get("summary", {}).get("profit_used") is not False:
        raise ValueError("Profit must remain outside the Generation 6 FreqAI target set.")
    if direct.get("summary", {}).get("direction_prediction") is not False:
        raise ValueError("Broad direction must remain closed in Generation 6.")
    for task in event_manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Generation 6 event artifact changed or is missing: {path}")
    return event_manifest, direct


def source_manifest(cohort: str) -> dict[str, Any]:
    path = NORMAL_MANIFEST if cohort == "normal" else MEME_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def cohort_settings(cohort: str) -> dict[str, Any]:
    if cohort == "normal":
        return {
            "timerange": "20240101-20260401",
            "train_days": 730,
            "backtest_days": 90,
            "validation_periods": ("validation_early", "validation_late"),
        }
    if cohort == "meme":
        return {
            "timerange": "20260101-20260714",
            "train_days": 160,
            "backtest_days": 30,
            "validation_periods": ("meme_validation_early", "meme_validation_late"),
        }
    raise ValueError(f"Unknown cohort: {cohort}")


def target_values(frame: DataFrame) -> DataFrame:
    output = DataFrame(index=frame.index)
    for target, source in TARGET_SOURCE_COLUMNS.items():
        if source == "absolute_pressure_change_h1":
            output[target] = pd.to_numeric(
                frame["pressure_change_h1"], errors="coerce"
            ).abs()
        else:
            output[target] = pd.to_numeric(frame[source], errors="coerce")
    return output


def independent_event_dates(frame: DataFrame) -> DataFrame:
    retained: list[int] = []
    for _, group in frame.sort_values("date").groupby("period", sort=False):
        previous: pd.Timestamp | None = None
        for index, date in zip(group.index, group["date"], strict=True):
            if previous is None or date - previous > pd.Timedelta(
                hours=MAX_TARGET_HORIZON_HOURS
            ):
                retained.append(index)
                previous = date
    return frame.loc[retained].sort_values("date").reset_index(drop=True)


def feature_definition_subset(
    definitions: dict[str, tuple[str, Callable[[Series], Series]]],
    names: Sequence[str],
) -> dict[str, tuple[str, Callable[[Series], Series]]]:
    missing = sorted(set(names).difference(definitions))
    if missing:
        raise ValueError(f"Unknown frozen feature names: {missing}")
    return {name: definitions[name] for name in names}


def aggregate_transformed_block(
    rows: DataFrame,
    *,
    block: str,
    definitions: dict[str, tuple[str, Callable[[Series], Series]]],
) -> DataFrame:
    transformed = DataFrame({"date": rows["date"]})
    for name, definition in definitions.items():
        transformed[f"{block}__{name}"] = g6d.transformed_feature(rows, definition)
    return transformed.groupby("date", sort=True, as_index=False).first()


def aggregate_level_block(rows: DataFrame) -> DataFrame:
    grouped = rows.groupby("date", sort=True)
    output = grouped.agg(
        level__event_count=("level_family", "size"),
        level__distinct_family_count=("level_family", "nunique"),
        level__distinct_timeframe_count=("source_timeframe", "nunique"),
        level__mean_zone_half_width_atr=("zone_half_width_atr", "mean"),
        level__median_pre_distance_atr=("pre_distance_atr", "median"),
        level__mean_contact_close_distance_atr=("contact_close_distance_atr", "mean"),
    ).reset_index()
    for family in sorted(g6.SELECTED_LEVEL_FAMILIES):
        values = (
            rows["level_family"]
            .eq(family)
            .groupby(rows["date"])
            .sum()
            .reindex(output["date"], fill_value=0)
            .to_numpy(dtype=float)
        )
        output[f"level__family_{family}_count"] = values
    for timeframe in ("1h", "4h", "8h", "1d"):
        values = (
            rows["source_timeframe"]
            .eq(timeframe)
            .groupby(rows["date"])
            .sum()
            .reindex(output["date"], fill_value=0)
            .to_numpy(dtype=float)
        )
        output[f"level__timeframe_{timeframe}_count"] = values
    for approach in ("from_above", "from_below", "already_inside_or_unclear"):
        fractions = (
            rows["approach_state"]
            .eq(approach)
            .groupby(rows["date"])
            .mean()
            .reindex(output["date"])
            .to_numpy(dtype=float)
        )
        output[f"level__approach_{approach}_fraction"] = fractions
    return output


def aggregate_timeframe_blocks(rows: DataFrame) -> list[DataFrame]:
    outputs: list[DataFrame] = []
    for block, columns in TIMEFRAME_GROUPS.items():
        selected = rows[["date", *columns]].copy()
        for column in columns:
            selected[column] = selected[column].fillna(False).astype(float)
        aggregated = selected.groupby("date", sort=True, as_index=False).max()
        aggregated.rename(
            columns={column: f"{block}__{column.rsplit('__', 1)[-1]}" for column in columns},
            inplace=True,
        )
        outputs.append(aggregated)
    return outputs


def attach_stale_placebo(
    features: DataFrame,
    *,
    block: str,
    stale_hours: int = STALE_HOURS,
) -> DataFrame:
    columns = [column for column in features if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"Cannot build a stale placebo for empty block {block!r}.")
    outputs: list[DataFrame] = []
    for _, target in features.groupby("period", sort=False):
        target = target.sort_values("date").copy()
        source = target[["date", *columns]].rename(columns={"date": "source_date"})
        lookup = target[["date"]].copy()
        lookup["cutoff"] = lookup["date"] - pd.Timedelta(hours=stale_hours)
        matched = pd.merge_asof(
            lookup.sort_values("cutoff"),
            source.sort_values("source_date"),
            left_on="cutoff",
            right_on="source_date",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("date")
        renamed = matched[["date", "source_date", *columns]].rename(
            columns={
                column: f"{block}_placebo__{column.removeprefix(f'{block}__')}"
                for column in columns
            }
        )
        outputs.append(renamed)
    return pd.concat(outputs, ignore_index=True).sort_values("date").reset_index(drop=True)


def complete_block_mask(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No feature columns found for block {block!r}.")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def raw_event_columns() -> tuple[str, ...]:
    columns = {
        "cohort",
        "pair",
        "source_timeframe",
        "level_family",
        "level_name",
        "representation",
        "control",
        "event_time",
        "period",
        "zone_half_width_atr",
        "approach_state",
        "pre_distance_atr",
        "contact_close_distance_atr",
        "market_group",
        "smart_contract_platform",
        "pressure_change_h1",
        "volume_ratio_h1",
        "range_ratio_h1",
        "abs_excursion_atr_h4",
        "dwell_fraction_h4",
        *g6d.RELATIONSHIP_COLUMNS,
        "news_gdelt_ready",
        "news_topics_ready",
        "ob_btc_model_ready",
    }
    for definitions in (
        g6d.STATE_FEATURES,
        g6d.CROSS_MARKET_FEATURES,
        g6d.NEWS_FEATURES,
        g6d.ORDERBOOK_FEATURES,
    ):
        columns.update(source for source, _ in definitions.values())
    return tuple(sorted(columns))


def build_pair_cache(
    *,
    cohort: str,
    pair: str,
    source_path: Path,
    feature_dir: Path,
    event_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    rows = pd.read_parquet(
        source_path,
        columns=list(raw_event_columns()),
        filters=[("control", "==", "actual")],
    )
    rows.rename(columns={"event_time": "date"}, inplace=True)
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["date"]).reset_index(drop=True)
    rows = g6d.causal_deduplicate(
        rows,
        keys=(
            "cohort",
            "pair",
            "date",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )
    deduplicated_actual_rows = len(rows)
    targets = target_values(rows)
    rows = pd.concat([rows, targets], axis=1)
    grouped = rows.groupby("date", sort=True)
    event_dates = grouped.agg(
        period=("period", "first"),
        pair=("pair", "first"),
        cohort=("cohort", "first"),
        market_group=("market_group", "first"),
        smart_contract_platform=("smart_contract_platform", "first"),
        raw_level_rows=("level_family", "size"),
    ).reset_index()
    target_medians = grouped[list(TARGET_COLUMNS)].median().reset_index()
    event_dates = event_dates.merge(
        target_medians,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    complete_targets = event_dates[list(TARGET_COLUMNS)].notna().all(axis=1)
    event_dates = event_dates.loc[complete_targets].reset_index(drop=True)
    event_dates = independent_event_dates(event_dates)
    retained_dates = event_dates[["date"]]
    rows = rows.merge(retained_dates, on="date", how="inner", validate="many_to_one")

    feature_frames: list[DataFrame] = [aggregate_level_block(rows)]
    feature_frames.append(
        aggregate_transformed_block(
            rows,
            block="state",
            definitions=g6d.STATE_FEATURES,
        )
    )
    for block, names in STATE_GROUPS.items():
        feature_frames.append(
            aggregate_transformed_block(
                rows,
                block=block,
                definitions=feature_definition_subset(g6d.STATE_FEATURES, names),
            )
        )
    feature_frames.extend(aggregate_timeframe_blocks(rows))
    for block, names in CROSS_MARKET_GROUPS.items():
        feature_frames.append(
            aggregate_transformed_block(
                rows,
                block=block,
                definitions=feature_definition_subset(g6d.CROSS_MARKET_FEATURES, names),
            )
        )
    for block, names in ORDERBOOK_GROUPS.items():
        feature_frames.append(
            aggregate_transformed_block(
                rows,
                block=block,
                definitions=feature_definition_subset(g6d.ORDERBOOK_FEATURES, names),
            )
        )
    for block, names in NEWS_GROUPS.items():
        feature_frames.append(
            aggregate_transformed_block(
                rows,
                block=block,
                definitions=feature_definition_subset(g6d.NEWS_FEATURES, names),
            )
        )

    features = event_dates[["date", "period"]].copy()
    for block_frame in feature_frames:
        features = features.merge(
            block_frame,
            on="date",
            how="left",
            validate="one_to_one",
        )

    declared = grouped.agg(
        orderbook_btc=("ob_btc_model_ready", "first"),
        news_gdelt=("news_gdelt_ready", "first"),
        news_topics=("news_topics_ready", "first"),
    ).reset_index()
    features = features.merge(declared, on="date", how="left", validate="one_to_one")
    for block in ("orderbook_btc", "news_gdelt", "news_topics"):
        mask = features[block].fillna(False).astype(bool)
        block_columns = [
            column for column in features if column.startswith(f"{block}__")
        ]
        features.loc[~mask, block_columns] = np.nan

    current_blocks = (
        "state",
        "level",
        *STATE_GROUPS,
        *TIMEFRAME_GROUPS,
        *CROSS_MARKET_GROUPS,
        *ORDERBOOK_GROUPS,
        *NEWS_GROUPS,
    )
    placebo_audit: list[dict[str, Any]] = []
    for block in current_blocks:
        placebo = attach_stale_placebo(features, block=block)
        source_column = f"placebo_source_date__{block}"
        placebo.rename(columns={"source_date": source_column}, inplace=True)
        features = features.merge(
            placebo,
            on="date",
            how="left",
            validate="one_to_one",
        )
        valid_sources = features[source_column].notna()
        future = valid_sources & (
            pd.to_datetime(features[source_column], utc=True)
            > features["date"] - pd.Timedelta(hours=STALE_HOURS)
        )
        placebo_audit.append(
            {
                "block": block,
                "ready_rows": int(complete_block_mask(features, f"{block}_placebo").sum()),
                "future_or_too_recent_source_violations": int(future.sum()),
            }
        )
    violations = sum(
        int(item["future_or_too_recent_source_violations"]) for item in placebo_audit
    )
    if violations:
        raise ValueError(f"Causal stale-placebo violations for {pair}: {violations}")

    ready = DataFrame({"date": features["date"]})
    for block in current_blocks:
        ready[f"ready__{block}"] = complete_block_mask(features, block)
        ready[f"ready__{block}_placebo"] = complete_block_mask(
            features, f"{block}_placebo"
        )
    event_cache = event_dates.merge(ready, on="date", how="left", validate="one_to_one")
    valid_feature_prefixes = tuple(
        [
            *(f"{block}__" for block in current_blocks),
            *(f"{block}_placebo__" for block in current_blocks),
        ]
    )
    feature_columns = [
        column for column in features if column.startswith(valid_feature_prefixes)
    ]
    feature_cache = features[["date", *feature_columns]].copy()
    for column in feature_columns:
        feature_cache[column] = pd.to_numeric(feature_cache[column], errors="coerce").astype(
            "float32"
        )

    evaluation_columns = [
        "date",
        "pair",
        "cohort",
        "period",
        "market_group",
        "smart_contract_platform",
        "level_family",
        "level_name",
        "source_timeframe",
        *TARGET_COLUMNS,
    ]
    evaluation = rows[evaluation_columns].copy()
    evaluation = (
        evaluation.groupby(
            [
                "date",
                "pair",
                "cohort",
                "period",
                "market_group",
                "smart_contract_platform",
                "level_family",
                "level_name",
                "source_timeframe",
            ],
            dropna=False,
            observed=True,
            as_index=False,
        )[list(TARGET_COLUMNS)]
        .median()
        .sort_values(["date", "level_family", "source_timeframe"])
        .reset_index(drop=True)
    )

    stem = g0.pair_file_stem(pair)
    feature_path = feature_dir / f"{stem}.parquet"
    event_path = event_dir / f"{stem}.parquet"
    evaluation_path = evaluation_dir / f"{stem}.parquet"
    g0.atomic_write_parquet(feature_cache, feature_path)
    g0.atomic_write_parquet(event_cache, event_path)
    g0.atomic_write_parquet(evaluation, evaluation_path)
    period_rows = (
        event_cache.groupby("period", observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict("records")
    )
    readiness_rows = []
    for column in sorted(column for column in event_cache if column.startswith("ready__")):
        readiness_rows.append(
            {
                "block": column.removeprefix("ready__"),
                "rows": int(event_cache[column].fillna(False).astype(bool).sum()),
            }
        )
    return {
        "cohort": cohort,
        "pair": pair,
        "source_path": str(source_path),
        "source_sha256": g0.sha256_file(source_path),
        "deduplicated_actual_rows": deduplicated_actual_rows,
        "independent_event_rows": len(event_cache),
        "evaluation_rows": len(evaluation),
        "feature_columns": len(feature_columns),
        "feature_path": str(feature_path),
        "feature_sha256": g0.sha256_file(feature_path),
        "event_path": str(event_path),
        "event_sha256": g0.sha256_file(event_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": g0.sha256_file(evaluation_path),
        "period_rows": period_rows,
        "readiness_rows": readiness_rows,
        "placebo_audit": placebo_audit,
        "causal_placebo_violations": violations,
    }


def build_caches(
    *,
    cohort: str,
    pairs: Sequence[str],
    task_map: dict[str, dict[str, Any]],
    feature_dir: Path,
    event_dir: Path,
    evaluation_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    for path in (feature_dir, event_dir, evaluation_dir):
        path.mkdir(parents=True, exist_ok=True)
    kwargs = [
        {
            "cohort": cohort,
            "pair": pair,
            "source_path": Path(task_map[pair]["event_path"]),
            "feature_dir": feature_dir,
            "event_dir": event_dir,
            "evaluation_dir": evaluation_dir,
        }
        for pair in pairs
    ]
    if workers == 1:
        return [build_pair_cache(**item) for item in kwargs]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_pair_cache, **item): item["pair"] for item in kwargs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g6_freqai_cache",
                        "pair": result["pair"],
                        "independent_events": result["independent_event_rows"],
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: pairs.index(item["pair"]))


def select_pairs(allowed: Sequence[str], requested: str) -> tuple[str, ...]:
    if requested == "all":
        return tuple(allowed)
    selected = parse_csv(requested)
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: {unknown}")
    return selected


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
        training["n_estimators"] = min(12, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_g6"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "feature_blocks": list(profile["blocks"]),
        "required_ready_blocks": list(profile["required_ready_blocks"]),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
        "missing_source_policy": (
            "missing remains NaN and removes that event from every ladder profile"
        ),
    }
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
    cache_inventory: Sequence[dict[str, Any]],
    profile_workers: int,
    technical_smoke: bool,
    settings: dict[str, Any],
    timerange: str,
) -> dict[str, Any]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    commands: list[dict[str, Any]] = []
    for index, profile_id in enumerate(profiles, start=1):
        definition = PROFILES[profile_id]
        short_id = f"p{index:03d}_{stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g6-{stable_digest(f'{run_id}|{profile_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
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
                "surface": definition["surface"],
                "blocks": list(definition["blocks"]),
                "required_ready_blocks": list(definition["required_ready_blocks"]),
                "plain_role": definition["plain_role"],
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
    cache_contract = [
        {
            key: item[key]
            for key in (
                "pair",
                "source_path",
                "source_sha256",
                "independent_event_rows",
                "evaluation_rows",
                "feature_columns",
                "feature_path",
                "feature_sha256",
                "event_path",
                "event_sha256",
                "evaluation_path",
                "evaluation_sha256",
                "causal_placebo_violations",
            )
        }
        for item in cache_inventory
    ]
    active_comparisons = [
        item
        for item in COMPARISONS
        if item["candidate"] in profiles and item["baseline"] in profiles
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
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": active_comparisons,
        "parked_source_blocks": PARKED_SOURCE_BLOCKS,
        "targets": list(TARGET_COLUMNS),
        "target_meanings": TARGET_MEANINGS,
        "objective": (
            "Challenge the complete Generation 6 direct-source portfolio with controlled "
            "FreqAI regressions on independent level-contact timestamps."
        ),
        "model_ladder": (
            "For each named source block, compare recent market behaviour, level geometry, "
            "the source alone, level plus source, and causal 72-hour-stale copies on the "
            "same eligible training and prediction rows."
        ),
        "decision_clock": (
            "Completion of the one-hour contact candle. Targets begin with the next candle; "
            "contact-candle volume is not a FreqAI target in this batch."
        ),
        "research_boundary": {
            "profit_used": False,
            "direction_prediction": False,
            "entry_or_exit_action": False,
            "direct_screen_candidates_used_to_select_features": False,
        },
        "controls": {
            "same_rows_within_each_ladder": True,
            "stale_placebo_hours": STALE_HOURS,
            "stale_placebo_is_causal": True,
            "direct_location_controls": str(DIRECT_RESULT),
            "independent_event_gap": f"strictly greater than {MAX_TARGET_HORIZON_HOURS}h",
        },
        "source_contracts": {
            "base_config": str(base_config.resolve()),
            "base_config_sha256": g0.sha256_file(base_config),
            "event_manifest": str(EVENT_MANIFEST),
            "event_manifest_sha256": g0.sha256_file(EVENT_MANIFEST),
            "direct_result": str(DIRECT_RESULT),
            "direct_result_sha256": g0.sha256_file(DIRECT_RESULT),
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": g0.sha256_file(FROZEN_BATCH),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": g0.sha256_file(Path(__file__)),
            "strategy": str(STRATEGY_FILE.resolve()),
            "strategy_sha256": g0.sha256_file(STRATEGY_FILE),
            "cache_inventory": cache_contract,
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "evaluation_cache_dir": str(artifact_dir / "evaluation_cache"),
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
    event_manifest, _ = validate_generation_contract()
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g6_freqai_run_manifest.json"
    if manifest_path.is_file():
        return (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path,
            artifact_dir,
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    evaluation_dir = artifact_dir / "evaluation_cache"
    task_map = event_task_map(event_manifest, cohort)
    missing = sorted(set(pairs).difference(task_map))
    if missing:
        raise ValueError(f"Missing shared Generation 6 event tasks: {missing}")
    inventory = build_caches(
        cohort=cohort,
        pairs=pairs,
        task_map=task_map,
        feature_dir=feature_dir,
        event_dir=event_dir,
        evaluation_dir=evaluation_dir,
        workers=cache_workers,
    )
    inventory_path = record_dir / "g6_freqai_cache_inventory.json"
    g0.atomic_write_json(inventory, inventory_path)
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
        cache_inventory=inventory,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
        settings=settings,
        timerange=timerange,
    )
    manifest["cache_inventory_record"] = {
        "path": str(inventory_path),
        "sha256": g0.sha256_file(inventory_path),
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def preflight_run(  # noqa: C901 - source and runtime gates stay explicit and auditable
    manifest: dict[str, Any], *, python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    if not python_exe.is_file():
        problems.append(f"Missing worker interpreter: {python_exe}")
    base_config = Path(manifest["source_contracts"]["base_config"])
    if not base_config.is_file():
        problems.append(f"Missing base config: {base_config}")
    elif g0.sha256_file(base_config) != manifest["source_contracts"]["base_config_sha256"]:
        problems.append(f"Base config changed after the run was prepared: {base_config}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 6 strategy: {STRATEGY_FILE}")
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
            problems.append("The configured worker cannot import Freqtrade and LightGBM.")
    cache_audit: list[dict[str, Any]] = []
    for item in manifest["source_contracts"]["cache_inventory"]:
        event_path = Path(item["event_path"])
        feature_path = Path(item["feature_path"])
        evaluation_path = Path(item["evaluation_path"])
        valid = (
            event_path.is_file()
            and feature_path.is_file()
            and evaluation_path.is_file()
            and g0.sha256_file(event_path) == item["event_sha256"]
            and g0.sha256_file(feature_path) == item["feature_sha256"]
            and g0.sha256_file(evaluation_path) == item["evaluation_sha256"]
        )
        if not valid:
            problems.append(f"Cache hash mismatch for {item['pair']}")
        if item["causal_placebo_violations"]:
            problems.append(f"Causal placebo violation for {item['pair']}")
        cache_audit.append(
            {
                "pair": item["pair"],
                "independent_events": item["independent_event_rows"],
                "feature_columns": item["feature_columns"],
                "hashes_valid": valid,
            }
        )
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    readiness_audit: list[dict[str, Any]] = []
    unique_requirements = sorted(
        {
            tuple(item["required_ready_blocks"])
            for item in manifest["commands"]
        }
    )
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    for pair in manifest["pairs"]:
        path = event_dir / f"{g0.pair_file_stem(pair)}.parquet"
        ready_columns = sorted(
            {f"ready__{block}" for blocks in unique_requirements for block in blocks}
        )
        frame = pd.read_parquet(
            path,
            columns=["date", "period", *TARGET_COLUMNS, *ready_columns],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        for blocks in unique_requirements:
            columns = [f"ready__{block}" for block in blocks]
            eligible = frame[columns].fillna(False).astype(bool).all(axis=1)
            eligible &= frame[list(TARGET_COLUMNS)].notna().all(axis=1)
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
                    "required_ready_blocks": ",".join(blocks),
                    "training_rows_before_first_prediction": training_rows,
                    "prediction_timerange_rows": prediction_rows,
                    "validation_period_rows": json.dumps(period_counts, sort_keys=True),
                }
            )
            if training_rows < MIN_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {training_rows} training rows for blocks {blocks}."
                )
            if prediction_rows < MIN_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {prediction_rows} prediction rows for blocks {blocks}."
                )
            if not manifest["technical_smoke_not_evidence"]:
                sparse_periods = {
                    period: count
                    for period, count in period_counts.items()
                    if count < MIN_SCORABLE_ROWS
                }
                if sparse_periods:
                    problems.append(
                        f"{pair} lacks validation support for blocks {blocks}: {sparse_periods}."
                    )
    free_gib = shutil.disk_usage(artifact_root_for_manifest(manifest)).free / (1024**3)
    if free_gib < 20.0:
        problems.append(f"Only {free_gib:.2f} GiB free on the bulky artifact drive.")
    return {
        "created_at_utc": g0.utc_now(),
        "passed": not problems,
        "problems": problems,
        "worker_interpreter": (
            str(python_exe.resolve()) if python_exe.is_file() else str(python_exe)
        ),
        "dependency_check": dependency,
        "cache_audit": cache_audit,
        "readiness_audit": readiness_audit,
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def artifact_root_for_manifest(manifest: dict[str, Any]) -> Path:
    return Path(manifest["storage"]["bulky_artifact_dir"])


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
                            "phase": "g6_freqai_profiles",
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
        raise ValueError("Duplicate pair/date keys in the Generation 6 target cache.")
    return output


def load_evaluation_rows(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    evaluation_dir = Path(manifest["storage"]["evaluation_cache_dir"])
    for pair in manifest["pairs"]:
        path = evaluation_dir / f"{g0.pair_file_stem(pair)}.parquet"
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def regression_row(
    frame: DataFrame,
    *,
    target: str,
    candidate_column: str,
    baseline_column: str,
) -> dict[str, Any]:
    actual_column = f"{target}__actual"
    complete = frame[[actual_column, candidate_column, baseline_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(complete).all(axis=1)
    values = complete.loc[finite]
    if len(values) < MIN_SCORABLE_ROWS:
        return {
            "rows": len(values),
            "candidate_mae": np.nan,
            "baseline_mae": np.nan,
            "paired_mae_gain": np.nan,
            "relative_mae_gain": np.nan,
            "candidate_spearman": np.nan,
            "baseline_spearman": np.nan,
            "spearman_change": np.nan,
            "candidate_calibration_slope": np.nan,
        }
    actual = values[actual_column]
    candidate = values[candidate_column]
    baseline = values[baseline_column]
    candidate_error = (actual - candidate).abs()
    baseline_error = (actual - baseline).abs()
    candidate_mae = float(candidate_error.mean())
    baseline_mae = float(baseline_error.mean())
    variance = float(candidate.var(ddof=0))
    calibration = (
        float(np.cov(candidate, actual, ddof=0)[0, 1] / variance)
        if variance > 0.0
        else np.nan
    )
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
        "candidate_spearman": float(candidate.corr(actual, method="spearman")),
        "baseline_spearman": float(baseline.corr(actual, method="spearman")),
        "spearman_change": float(
            candidate.corr(actual, method="spearman")
            - baseline.corr(actual, method="spearman")
        ),
        "candidate_calibration_slope": calibration,
    }


def fair_comparison_frame(
    *,
    candidate: DataFrame,
    baseline: DataFrame,
    actual: DataFrame,
) -> tuple[DataFrame, dict[str, Any]]:
    candidate_keys = candidate[["pair", "date"]].drop_duplicates()
    baseline_keys = baseline[["pair", "date"]].drop_duplicates()
    common = candidate_keys.merge(
        baseline_keys,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    )
    candidate_only = len(candidate_keys) - len(common)
    baseline_only = len(baseline_keys) - len(common)
    candidate_selected = candidate[["pair", "date", *TARGET_COLUMNS]].merge(
        common,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    )
    baseline_selected = baseline[["pair", "date", *TARGET_COLUMNS]].merge(
        common,
        on=["pair", "date"],
        how="inner",
        validate="one_to_one",
    )
    candidate_selected.rename(
        columns={target: f"{target}__candidate" for target in TARGET_COLUMNS},
        inplace=True,
    )
    baseline_selected.rename(
        columns={target: f"{target}__baseline" for target in TARGET_COLUMNS},
        inplace=True,
    )
    actual_selected = actual.rename(
        columns={target: f"{target}__actual" for target in TARGET_COLUMNS}
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
        "candidate_only_rows": candidate_only,
        "baseline_only_rows": baseline_only,
        "fair_labelled_rows": len(fair),
        "identical_prediction_keys": candidate_only == 0 and baseline_only == 0,
    }


def score_comparisons(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
    evaluation: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for definition in manifest["comparisons"]:
        comparison_id = str(definition["comparison_id"])
        candidate_id = str(definition["candidate"])
        baseline_id = str(definition["baseline"])
        fair, audit = fair_comparison_frame(
            candidate=predictions[candidate_id],
            baseline=predictions[baseline_id],
            actual=actual,
        )
        eligibility_rows.append({**definition, **audit})
        for (pair, period), group in fair.groupby(
            ["pair", "period"], sort=False, observed=True
        ):
            for target in TARGET_COLUMNS:
                metrics = regression_row(
                    group,
                    target=target,
                    candidate_column=f"{target}__candidate",
                    baseline_column=f"{target}__baseline",
                )
                pair_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "target": target,
                        **metrics,
                    }
                )
        for period, group in fair.groupby("period", sort=False, observed=True):
            for target in TARGET_COLUMNS:
                actual_column = f"{target}__actual"
                candidate_column = f"{target}__candidate"
                baseline_column = f"{target}__baseline"
                complete = group[
                    ["pair", "date", actual_column, candidate_column, baseline_column]
                ].copy()
                numeric = complete[
                    [actual_column, candidate_column, baseline_column]
                ].apply(pd.to_numeric, errors="coerce")
                complete = complete.loc[np.isfinite(numeric).all(axis=1)].copy()
                complete["paired_error_gain"] = (
                    complete[actual_column] - complete[baseline_column]
                ).abs() - (
                    complete[actual_column] - complete[candidate_column]
                ).abs()
                point, lower, upper = deterministic_block_bootstrap(
                    complete[["pair", "date", "paired_error_gain"]],
                    seed_key=f"g6|{manifest['run_id']}|{comparison_id}|{period}|{target}",
                    samples=BOOTSTRAP_SAMPLES,
                )
                pair_subset = [
                    row
                    for row in pair_rows
                    if row["comparison_id"] == comparison_id
                    and row["period"] == period
                    and row["target"] == target
                    and np.isfinite(row["paired_mae_gain"])
                ]
                aggregate_rows.append(
                    {
                        **definition,
                        "period": period,
                        "target": target,
                        "rows": len(complete),
                        "coins": len(pair_subset),
                        "positive_coins": sum(
                            float(row["paired_mae_gain"]) > 0.0 for row in pair_subset
                        ),
                        "negative_coins": sum(
                            float(row["paired_mae_gain"]) < 0.0 for row in pair_subset
                        ),
                        "equal_coin_paired_mae_gain": point,
                        "bootstrap_lower": lower,
                        "bootstrap_upper": upper,
                        "equal_coin_spearman_change": (
                            float(
                                np.nanmean(
                                    [float(row["spearman_change"]) for row in pair_subset]
                                )
                            )
                            if pair_subset
                            else np.nan
                        ),
                    }
                )
        candidate_predictions = predictions[candidate_id][
            ["pair", "date", *TARGET_COLUMNS]
        ].rename(columns={target: f"{target}__candidate" for target in TARGET_COLUMNS})
        baseline_predictions = predictions[baseline_id][
            ["pair", "date", *TARGET_COLUMNS]
        ].rename(columns={target: f"{target}__baseline" for target in TARGET_COLUMNS})
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
            columns={target: f"{target}__actual" for target in TARGET_COLUMNS},
            inplace=True,
        )
        for key, group in detailed.groupby(
            ["pair", "period", "level_family", "source_timeframe"],
            sort=False,
            observed=True,
        ):
            pair, period, level_family, timeframe = key
            for target in TARGET_COLUMNS:
                metrics = regression_row(
                    group,
                    target=target,
                    candidate_column=f"{target}__candidate",
                    baseline_column=f"{target}__baseline",
                )
                family_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "level_family": level_family,
                        "source_timeframe": timeframe,
                        "target": target,
                        **metrics,
                        "dependence_warning": (
                            "Descriptive family slice; several family rows may share one "
                            "market path."
                        ),
                    }
                )
    return (
        DataFrame.from_records(pair_rows),
        DataFrame.from_records(aggregate_rows),
        DataFrame.from_records(eligibility_rows),
        DataFrame.from_records(family_rows),
    )


def comparison_decisions(
    manifest: dict[str, Any], aggregate: DataFrame
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    periods = set(manifest["validation_periods"])
    required_positive_coins = 5 if len(manifest["pairs"]) >= 5 else len(manifest["pairs"])
    for key, group in aggregate.groupby(
        ["comparison_id", "lane", "candidate", "baseline", "question", "target"],
        dropna=False,
    ):
        comparison_id, lane, candidate, baseline, question, target = key
        checks = group.loc[group["period"].isin(periods)]
        complete_periods = set(checks["period"]) == periods
        enough_rows = checks["rows"].ge(MIN_SCORABLE_ROWS).all() if len(checks) else False
        point_positive = (
            checks["equal_coin_paired_mae_gain"].gt(0.0).all() if len(checks) else False
        )
        lower_positive = checks["bootstrap_lower"].gt(0.0).all() if len(checks) else False
        enough_coins = (
            checks["positive_coins"].ge(required_positive_coins).all()
            if len(checks)
            else False
        )
        rank_not_both_worse = not (
            len(checks)
            and checks["equal_coin_spearman_change"].lt(0.0).all()
        )
        if complete_periods and enough_rows and lower_positive and enough_coins:
            status = "control_resistant_incremental_freqai_lead"
        elif (
            complete_periods
            and enough_rows
            and point_positive
            and enough_coins
            and rank_not_both_worse
        ):
            status = "provisional_incremental_freqai_lead"
        elif not complete_periods or not enough_rows:
            status = "insufficient_freqai_support"
        else:
            status = "not_reproduced_by_freqai"
        rows.append(
            {
                "comparison_id": comparison_id,
                "lane": lane,
                "candidate": candidate,
                "baseline": baseline,
                "question": question,
                "target": target,
                "status": status,
                "validation_periods_complete": complete_periods,
                "positive_point_both_periods": point_positive,
                "bootstrap_lower_positive_both_periods": lower_positive,
                "required_positive_coins": required_positive_coins,
                "positive_coin_rule_met": enough_coins,
                "rank_not_worse_in_both_periods": rank_not_both_worse,
                "period_results": json.dumps(
                    checks[
                        [
                            "period",
                            "rows",
                            "coins",
                            "positive_coins",
                            "negative_coins",
                            "equal_coin_paired_mae_gain",
                            "bootstrap_lower",
                            "bootstrap_upper",
                            "equal_coin_spearman_change",
                        ]
                    ].to_dict("records"),
                    sort_keys=True,
                    default=g0.json_default,
                ),
                "meaning": (
                    "Exploratory reaction-estimation evidence only; not direction, profit, "
                    "causation, or a trading rule."
                ),
            }
        )
    return DataFrame.from_records(rows)


def group_memberships(cohort: str) -> dict[str, tuple[str, ...]]:
    if cohort == "meme":
        return {"frozen_top_ten_memes": g6.GROUPS["frozen_top_ten_memes"]}
    return {
        "btc_separate": g6.GROUPS["btc_separate"],
        "established_altcoins": g6.GROUPS["established_altcoins"],
        "smart_contract_platforms": g6.GROUPS["smart_contract_platforms"],
    }


def group_transfer_decisions(manifest: dict[str, Any], pair_scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    periods = set(manifest["validation_periods"])
    for group_id, members in group_memberships(str(manifest["cohort"])).items():
        if group_id == "btc_separate":
            required_positive = 1
            claim = "asset_specific_lead"
        elif group_id == "smart_contract_platforms":
            required_positive = g6.MIN_SUBGROUP_COINS
            claim = "coin_group_specific_lead"
        else:
            required_positive = g6.MIN_FULL_COINS
            claim = "coin_group_specific_lead"
        selected = pair_scores.loc[pair_scores["pair"].isin(members)]
        keys = ["comparison_id", "lane", "candidate", "baseline", "question", "target"]
        for key, group in selected.groupby(keys, dropna=False):
            record = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
            period_rows: list[dict[str, Any]] = []
            for period in manifest["validation_periods"]:
                period_frame = group.loc[
                    group["period"].eq(period) & group["paired_mae_gain"].notna()
                ]
                gains = pd.to_numeric(period_frame["paired_mae_gain"], errors="coerce")
                period_rows.append(
                    {
                        "period": period,
                        "coins": len(period_frame),
                        "positive_coins": int(gains.gt(0.0).sum()),
                        "negative_coins": int(gains.lt(0.0).sum()),
                        "equal_coin_mean_gain": (
                            float(gains.mean()) if len(gains) else np.nan
                        ),
                        "largest_absolute_coin_share": (
                            float(gains.abs().max() / gains.abs().sum())
                            if len(gains) and gains.abs().sum() > 0.0
                            else np.nan
                        ),
                    }
                )
            checks = DataFrame.from_records(period_rows)
            complete = set(checks["period"]) == periods and checks["coins"].gt(0).all()
            positive = checks["equal_coin_mean_gain"].gt(0.0).all()
            majority = checks["positive_coins"].ge(required_positive).all()
            not_dominated = (
                checks["largest_absolute_coin_share"].le(0.60).all()
                if group_id != "btc_separate"
                else True
            )
            retained = bool(complete and positive and majority and not_dominated)
            rows.append(
                {
                    **record,
                    "group_id": group_id,
                    "members": ",".join(members),
                    "required_positive_coins": required_positive,
                    "retained": retained,
                    "classification": claim if retained else "not_reproduced_for_group",
                    "period_results": json.dumps(
                        period_rows, sort_keys=True, default=g0.json_default
                    ),
                    "note": (
                        "Membership was frozen before these outcomes; no result-driven regrouping."
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
            raise ValueError(f"Profile {profile_id} lacks prediction targets: {missing}")
        predictions[profile_id] = frame
    actual = load_event_targets(manifest)
    evaluation = load_evaluation_rows(manifest)
    pair_scores, aggregate, eligibility, family = score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
        evaluation=evaluation,
    )
    decisions = comparison_decisions(manifest, aggregate)
    groups = group_transfer_decisions(manifest, pair_scores)

    record_dir.mkdir(parents=True, exist_ok=True)
    score_paths = {
        "prediction_audit": record_dir / "g6_freqai_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g6_freqai_comparison_eligibility.csv",
        "pair_scores": record_dir / "g6_freqai_pair_scores.csv",
        "aggregate_scores": record_dir / "g6_freqai_aggregate_scores.csv",
        "comparison_decisions": record_dir / "g6_freqai_comparison_decisions.csv",
        "group_transfer": record_dir / "g6_freqai_group_transfer.csv",
        "family_slices": artifact_dir / "g6_freqai_level_family_slices.parquet",
    }
    g0.atomic_write_csv(DataFrame.from_records(prediction_audit), score_paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, score_paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, score_paths["pair_scores"])
    g0.atomic_write_csv(aggregate, score_paths["aggregate_scores"])
    g0.atomic_write_csv(decisions, score_paths["comparison_decisions"])
    g0.atomic_write_csv(groups, score_paths["group_transfer"])
    g0.atomic_write_parquet(family, score_paths["family_slices"])
    unequal_keys = int((~eligibility["identical_prediction_keys"]).sum()) if len(eligibility) else 0
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": "completed_freqai_source_ladders",
        "created_at_utc": g0.utc_now(),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "comparisons_completed": int(decisions["comparison_id"].nunique()) if len(decisions) else 0,
        "targets": list(TARGET_COLUMNS),
        "control_resistant_leads": int(
            decisions["status"].eq("control_resistant_incremental_freqai_lead").sum()
        ),
        "provisional_leads": int(
            decisions["status"].eq("provisional_incremental_freqai_lead").sum()
        ),
        "not_reproduced": int(decisions["status"].eq("not_reproduced_by_freqai").sum()),
        "insufficient": int(decisions["status"].eq("insufficient_freqai_support").sum()),
        "retained_group_specific_rows": int(groups["retained"].sum()) if len(groups) else 0,
        "integrity": {
            "unequal_prediction_key_comparisons": unequal_keys,
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in prediction_audit)
            ),
            "profit_used": False,
            "direction_prediction": False,
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
        },
        "artifacts": {key: artifact_record(path) for key, path in score_paths.items()},
        "interpretation_boundary": (
            "These are incremental reaction-estimation results. They do not establish "
            "causation, direction, profitability, entry, exit, or strategy promotion."
        ),
    }
    result_path = record_dir / "g6_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def technical_smoke_profiles() -> tuple[str, ...]:
    selected = [profile for profile in PROFILES if profile.startswith("core__")]
    full_ladders = {
        "ohlcv_volume_pressure",
        "timeframe_4h",
        "cross_market_btc",
        "orderbook_btc",
        "news_gdelt",
    }
    for source in SOURCE_BLOCKS:
        source_profiles = [
            profile for profile in PROFILES if profile.startswith(f"{source}__")
        ]
        if source in full_ladders:
            selected.extend(source_profiles)
        else:
            selected.extend(
                profile
                for profile in source_profiles
                if profile.endswith("__source_only")
                or profile.endswith("__level_plus_source")
            )
    return tuple(dict.fromkeys(selected))


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled Generation 6 FreqAI regression ladders for all broad "
            "reaction-source siblings without profit or direction targets."
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
            raise ValueError(f"{label} must be between 1 and {MAX_WORKERS}; received {value}.")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    manifest_source = source_manifest(args.cohort)
    allowed_pairs = tuple(str(pair) for pair in manifest_source["data"]["pairs"])
    pairs = select_pairs(allowed_pairs, args.pairs)
    if args.profiles == "all":
        profiles = tuple(PROFILES)
    elif args.profiles == "smoke":
        profiles = technical_smoke_profiles()
    else:
        profiles = parse_csv(args.profiles)
    unknown_profiles = sorted(set(profiles).difference(PROFILES))
    if not profiles or unknown_profiles:
        raise ValueError(f"Invalid Generation 6 profiles: {unknown_profiles}")

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
    preflight_path = record_dir / "g6_freqai_launch_preflight.json"
    g0.atomic_write_json(preflight, preflight_path)
    manifest["launch_preflight"] = artifact_record(preflight_path)
    g0.atomic_write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "phase": "g6_freqai_launch_preflight",
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
        raise ValueError(
            "Use --retry-failed only after diagnosing a failed Generation 6 profile."
        )

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
        raise ValueError(f"Cannot score incomplete Generation 6 profiles: {incomplete}")
    result = score_run(manifest, record_dir=record_dir, artifact_dir=artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = g0.utc_now()
    manifest["result"] = result
    g0.atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
