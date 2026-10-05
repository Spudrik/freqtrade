"""Build the frozen Generation 11 event-level FreqAI cache in two stages.

Predictor support is materialized and frozen before reaction targets are opened.
"""

from __future__ import annotations

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
import gc
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    market_reaction_zone_generation8_cache as g8c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as g11f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_preflight as g11p,
)


CACHE_ID = "g11_freqai_cache_20260822a"
RECORD_ROOT = g11f.g11z.OUTPUT_ROOT / "generation11_shared" / CACHE_ID
ARTIFACT_ROOT = g0.LARGE_ARTIFACT_ROOT / "generation11_shared" / CACHE_ID
MAX_WORKERS = 4
COOLDOWN_HOURS = 8
MIN_DEVELOPMENT_EVENTS = 100
MIN_VALIDATION_EVENTS = 50
MIN_COINS = 5

META_COLUMNS = (
    "cohort",
    "pair",
    "event_time",
    "period",
    "source_timeframe",
    "level_family",
    "level_name",
    "representation",
    "control",
    "base_index",
    "zone_method",
    "approach_code",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "level_score",
    "zone_half_width_atr",
    "mechanism_group",
    "level_side",
    "market_group",
    "smart_contract_platform",
)
STATE_COLUMNS = g11p.STATE_COLUMNS
CROSS_MARKET_COLUMNS = (
    *g11p.CROSS_MARKET_COLUMNS,
    "xm_pair_minus_btc_1h",
    "xm_pair_minus_btc_24h",
)
NEWS_COLUMNS = (
    "news_gdelt_ready",
    "news_gdelt_regime",
    "news_gdelt_activity",
    "news_gdelt__log_event_count_1h",
    "news_gdelt__log_event_count_24h",
    "news_gdelt__log_num_mentions_24h",
    "news_gdelt__log_num_sources_24h",
    "news_gdelt__avg_tone_24h",
    "news_gdelt__goldstein_24h",
    "news_topics_ready",
    "news_topic__geopolitics_and_energy",
    "news_topic__macro_policy_and_growth",
    "news_topic__crypto_policy_and_institutions",
    "news_topic__crypto_liquidity_and_security",
)
ORDERBOOK_COLUMNS = (
    "ob_btc_source_ready",
    "ob_btc_model_ready",
    "ob_btc_regime",
    "ob_btc_coverage_ratio",
    "ob_btc_pressure_25bps",
    "ob_btc_absolute_pressure",
    "ob_pair_local",
)
FEATURE_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            *META_COLUMNS,
            *g11p.RELATIONSHIP_COLUMNS,
            *STATE_COLUMNS,
            *CROSS_MARKET_COLUMNS,
            *NEWS_COLUMNS,
            *ORDERBOOK_COLUMNS,
        )
    )
)
OUTCOME_COLUMNS = tuple(
    column
    for horizon in g11f.g11z.HORIZONS
    for column in (f"abs_excursion_atr_h{horizon}", f"volume_ratio_h{horizon}")
)
TARGET_INPUT_COLUMNS = tuple(dict.fromkeys((*META_COLUMNS, *OUTCOME_COLUMNS)))


@dataclass(frozen=True)
class CacheTask:
    cohort: str
    pair: str
    source_path: Path
    source_sha256: str
    feature_dir: Path
    support_dir: Path


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_feature_column_contract(columns: Sequence[str] = FEATURE_INPUT_COLUMNS) -> None:
    forbidden = (
        "excursion_atr_h",
        "volume_ratio_h",
        "range_ratio_h",
        "dwell_fraction_h",
        "crossings_h",
        "time_to_",
    )
    found = [column for column in columns if any(item in column for item in forbidden)]
    if found:
        raise ValueError(f"Outcome columns entered feature-cache stage: {found}")


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = g11f.validate_existing_freeze()
    manifest = json.loads(g11f.g11z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    frozen_manifest = frozen["source_contracts"]["generation6_causal_event_manifest"]
    if g0.sha256_file(g11f.g11z.G6_EVENT_MANIFEST) != frozen_manifest["sha256"]:
        raise ValueError("Generation 6 event manifest changed after the FreqAI freeze.")
    for task in manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Frozen event source changed: {path}")
    return frozen, manifest


def numeric(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def causal_deduplicate(frame: DataFrame) -> DataFrame:
    return g6d.causal_deduplicate(
        frame,
        keys=(
            "cohort",
            "pair",
            "event_time",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )


def select_independent_anchors(frame: DataFrame) -> DataFrame:
    source = causal_deduplicate(frame.loc[frame["control"].eq("actual")].copy())
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    source["_distance"] = numeric(source, "contact_close_distance_atr")
    source["_score"] = numeric(source, "level_score")
    anchors = source.sort_values(
        ["pair", "period", "event_time", "_distance", "_score", "source_timeframe"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    ).drop_duplicates(["pair", "period", "event_time"], keep="first")
    keep = Series(False, index=anchors.index)
    cooldown = pd.Timedelta(hours=COOLDOWN_HOURS)
    for _, group in anchors.groupby(["pair", "period"], observed=True, sort=False):
        last: pd.Timestamp | None = None
        for index, timestamp in zip(group.index, group["event_time"], strict=True):
            if last is None or timestamp - last >= cooldown:
                keep.loc[index] = True
                last = timestamp
    return (
        anchors.loc[keep]
        .drop(columns=["_distance", "_score"])
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def group_numeric_first(source: DataFrame, dates: Series, column: str) -> Series:
    mapping = source.groupby("event_time", observed=True)[column].first()
    return dates.map(mapping)


def relationship_any(source: DataFrame, suffix: str) -> Series:
    columns = [
        column for column in g11p.RELATIONSHIP_COLUMNS if column.endswith(suffix)
    ]
    return source[columns].fillna(False).astype(bool).any(axis=1)


def build_feature_frame(frame: DataFrame, anchors: DataFrame) -> DataFrame:  # noqa: C901
    selected_dates = set(pd.to_datetime(anchors["event_time"], utc=True))
    source = causal_deduplicate(frame.loc[frame["event_time"].isin(selected_dates)].copy())
    source = source.loc[source["control"].eq("actual")].copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True)
    anchor = anchors.set_index("event_time")
    dates = pd.to_datetime(anchors["event_time"], utc=True).reset_index(drop=True)
    output = DataFrame({"date": dates, "period": anchors["period"].reset_index(drop=True)})
    grouped = source.groupby("event_time", observed=True)
    count = dates.map(grouped.size()).astype(float)
    prefix = g11f.MINIMAL
    output[f"{prefix}__contacted_level_count"] = count
    output[f"{prefix}__distinct_family_count"] = dates.map(
        grouped["level_family"].nunique()
    ).astype(float)
    output[f"{prefix}__distinct_timeframe_count"] = dates.map(
        grouped["source_timeframe"].nunique()
    ).astype(float)
    output[f"{prefix}__distinct_mechanism_count"] = dates.map(
        grouped["mechanism_group"].nunique()
    ).astype(float)
    output[f"{prefix}__mean_zone_half_width_atr"] = dates.map(
        grouped["zone_half_width_atr"].mean()
    )
    output[f"{prefix}__minimum_contact_distance_atr"] = dates.map(
        grouped["contact_close_distance_atr"].min()
    )
    anchor_rows = anchor.loc[dates].reset_index()
    output[f"{prefix}__anchor_zone_half_width_atr"] = numeric(
        anchor_rows, "zone_half_width_atr"
    )
    output[f"{prefix}__anchor_pre_distance_atr"] = numeric(
        anchor_rows, "pre_distance_atr"
    )
    output[f"{prefix}__anchor_contact_distance_atr"] = numeric(
        anchor_rows, "contact_close_distance_atr"
    )
    output[f"{prefix}__anchor_approach_code"] = numeric(anchor_rows, "approach_code")
    score = numeric(anchor_rows, "level_score")
    output[f"{prefix}__anchor_level_score_available"] = score.notna().astype(float)
    output[f"{prefix}__anchor_level_score"] = score.fillna(0.0)

    prefix = g11f.LEVEL
    anchor_tf = anchor_rows["source_timeframe"].astype(str)
    anchor_family = anchor_rows["level_family"].astype(str)
    for timeframe in g11f.g11z.SOURCE_TIMEFRAMES:
        output[f"{prefix}__anchor_source_{timeframe}"] = anchor_tf.eq(timeframe).astype(float)
        counts = source["source_timeframe"].eq(timeframe).groupby(source["event_time"]).sum()
        output[f"{prefix}__contact_fraction_source_{timeframe}"] = (
            dates.map(counts).fillna(0.0).to_numpy(dtype=float) / count
        )
    for family in g11f.LEVEL_FAMILIES:
        output[f"{prefix}__anchor_family_{family}"] = anchor_family.eq(family).astype(float)
        counts = source["level_family"].eq(family).groupby(source["event_time"]).sum()
        output[f"{prefix}__contact_fraction_family_{family}"] = (
            dates.map(counts).fillna(0.0).to_numpy(dtype=float) / count
        )

    prefix = g11f.GEOMETRY
    for suffix in (
        "same_mechanism_agreement",
        "different_mechanism_agreement",
        "any_cross_timeframe_cluster",
        "opposing_side_overlap",
    ):
        values = relationship_any(source, suffix)
        output[f"{prefix}__{suffix}"] = (
            values.groupby(source["event_time"]).max().reindex(dates).fillna(False).astype(float).to_numpy()
        )
    cluster_columns = [
        f"{prefix}__same_mechanism_agreement",
        f"{prefix}__different_mechanism_agreement",
        f"{prefix}__any_cross_timeframe_cluster",
        f"{prefix}__opposing_side_overlap",
    ]
    output[f"{prefix}__isolated_anchor"] = (
        ~output[cluster_columns].astype(bool).any(axis=1)
    ).astype(float)
    output[f"{prefix}__independent_family_count"] = output[
        f"{g11f.MINIMAL}__distinct_mechanism_count"
    ]

    prefix = g11f.TREND
    trend_map = {
        "ema20_slope": "state__ema20_slope",
        "ma_separation": "state__ma_separation",
        "return_slope": "state__return_slope",
        "return_acceleration": "state__return_acceleration",
        "adx14": "state__adx14",
        "rsi14_centered": "state__rsi14",
        "rsi_change": "state__rsi_change",
        "macd_histogram": "state__macd_histogram",
        "macd_histogram_change": "state__macd_histogram_change",
    }
    for suffix, column in trend_map.items():
        values = group_numeric_first(source, dates, column)
        if suffix == "rsi14_centered":
            values = values - 50.0
        output[f"{prefix}__{suffix}"] = values

    prefix = g11f.ACTIVITY
    activity_map = {
        "relative_volume": "state__relative_volume",
        "volume_acceleration": "state__volume_acceleration",
        "absolute_pressure": "state__absolute_pressure",
        "pressure_persistence": "state__pressure_persistence",
        "atr_fraction": "state__atr_fraction",
        "prior_range_atr": "state__prior_range_atr",
        "bollinger_width": "state__bollinger_width",
        "range_contraction": "state__range_contraction",
    }
    for suffix, column in activity_map.items():
        output[f"{prefix}__{suffix}"] = group_numeric_first(source, dates, column)

    prefix = g11f.MARKET
    market_map = {
        suffix.removeprefix("xm_"): suffix
        for suffix in CROSS_MARKET_COLUMNS
    }
    for suffix, column in market_map.items():
        output[f"{prefix}__{suffix}"] = group_numeric_first(source, dates, column)

    prefix = g11f.NEWS
    news_map = {
        "source_ready": "news_gdelt_ready",
        "topics_ready": "news_topics_ready",
        "activity": "news_gdelt_activity",
        "log_event_count_1h": "news_gdelt__log_event_count_1h",
        "log_event_count_24h": "news_gdelt__log_event_count_24h",
        "log_num_mentions_24h": "news_gdelt__log_num_mentions_24h",
        "log_num_sources_24h": "news_gdelt__log_num_sources_24h",
        "average_tone_24h": "news_gdelt__avg_tone_24h",
        "goldstein_24h": "news_gdelt__goldstein_24h",
        "topic_geopolitics_and_energy": "news_topic__geopolitics_and_energy",
        "topic_macro_policy_and_growth": "news_topic__macro_policy_and_growth",
        "topic_crypto_policy_and_institutions": "news_topic__crypto_policy_and_institutions",
        "topic_crypto_liquidity_and_security": "news_topic__crypto_liquidity_and_security",
    }
    for suffix, column in news_map.items():
        output[f"{prefix}__{suffix}"] = group_numeric_first(source, dates, column)
    news_regime = source.groupby("event_time", observed=True)["news_gdelt_regime"].first()
    news_regime = dates.map(news_regime).astype("string")
    for regime in ("quiet", "middle", "shock"):
        output[f"{prefix}__regime_{regime}"] = news_regime.eq(regime).astype(float)

    prefix = g11f.ORDERBOOK
    orderbook_map = {
        "source_ready": "ob_btc_source_ready",
        "model_ready": "ob_btc_model_ready",
        "coverage_ratio": "ob_btc_coverage_ratio",
        "pressure_25bps": "ob_btc_pressure_25bps",
        "absolute_pressure": "ob_btc_absolute_pressure",
        "pair_is_btc": "ob_pair_local",
    }
    for suffix, column in orderbook_map.items():
        output[f"{prefix}__{suffix}"] = group_numeric_first(source, dates, column)
    ob_regime = source.groupby("event_time", observed=True)["ob_btc_regime"].first()
    ob_regime = dates.map(ob_regime).astype("string")
    for regime in ("quiet", "middle", "active"):
        output[f"{prefix}__regime_{regime}"] = ob_regime.eq(regime).astype(float)

    expected = {
        f"{block}__{suffix}"
        for block, suffixes in g11f.FEATURE_SUFFIXES.items()
        for suffix in suffixes
    }
    present = {column for column in output if "__" in column}
    if present != expected:
        raise ValueError(
            f"Generation 11 feature contract drift: missing={sorted(expected - present)}, "
            f"extra={sorted(present - expected)}"
        )
    return output.replace([np.inf, -np.inf], np.nan)


def complete_block(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No Generation 11 columns for block {block!r}")
    return frame[columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)


def build_pair_feature_support(task: CacheTask) -> dict[str, Any]:
    validate_feature_column_contract()
    frame = pd.read_parquet(
        task.source_path,
        columns=list(FEATURE_INPUT_COLUMNS),
        filters=[("control", "==", "actual")],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    anchors = select_independent_anchors(frame)
    current = build_feature_frame(frame, anchors)
    full, placebo_audit = g8c.attach_stale_and_shuffled(
        current, pair=task.pair, blocks=g11f.CURRENT_BLOCKS
    )
    violations = sum(
        int(item["stale_timestamp_violations"] or item["shuffled_self_matches"])
        for item in placebo_audit
    )
    if violations:
        raise ValueError(f"Generation 11 placebo violation for {task.pair}")
    support = full[["date", "period"]].copy()
    all_blocks = [
        *g11f.CURRENT_BLOCKS,
        *(f"{block}_stale" for block in g11f.CURRENT_BLOCKS),
        *(f"{block}_shuffled" for block in g11f.CURRENT_BLOCKS),
    ]
    for block in all_blocks:
        support[f"ready__{block}"] = complete_block(full, block)
    feature_columns = [column for column in full if "__" in column]
    features = full[["date", *feature_columns]].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    stem = g0.pair_file_stem(task.pair)
    feature_path = task.feature_dir / f"{stem}.parquet"
    support_path = task.support_dir / f"{stem}.parquet"
    g0.atomic_write_parquet(features, feature_path)
    g0.atomic_write_parquet(support, support_path)
    del frame, current, full, features, support
    gc.collect()
    return {
        "cohort": task.cohort,
        "pair": task.pair,
        "source_path": str(task.source_path),
        "source_sha256": task.source_sha256,
        "feature_path": str(feature_path),
        "feature_sha256": g0.sha256_file(feature_path),
        "support_path": str(support_path),
        "support_sha256": g0.sha256_file(support_path),
        "independent_contact_rows": len(anchors),
        "feature_columns": len(feature_columns),
        "placebo_audit": placebo_audit,
        "reaction_outcome_columns_read": False,
    }


def support_decisions(
    frozen: dict[str, Any],
    cohort: str,
    inventory: Sequence[dict[str, Any]],
) -> DataFrame:
    frames: list[DataFrame] = []
    for item in inventory:
        support = pd.read_parquet(item["support_path"])
        support["pair"] = item["pair"]
        frames.append(support)
    all_support = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    settings = frozen["cohort_settings"][cohort]
    development = "development" if cohort == "normal" else "meme_development"
    periods = (development, *settings["validation_periods"])
    for profile_id, profile in frozen["profiles"].items():
        if profile["cohort"] != cohort:
            continue
        ready_columns = [f"ready__{block}" for block in profile["required_ready_blocks"]]
        ready = all_support[ready_columns].fillna(False).astype(bool).all(axis=1)
        checks: list[dict[str, Any]] = []
        for period in periods:
            selected = all_support.loc[ready & all_support["period"].eq(period)]
            per_pair = selected.groupby("pair", observed=True).size()
            checks.append(
                {
                    "period": period,
                    "events": len(selected),
                    "coins": int((per_pair > 0).sum()),
                    "minimum_pair_events": int(per_pair.min()) if len(per_pair) else 0,
                }
            )
        development_check, *validation_checks = checks
        supported = (
            development_check["events"] >= MIN_DEVELOPMENT_EVENTS
            and development_check["coins"] >= MIN_COINS
            and all(
                item["events"] >= MIN_VALIDATION_EVENTS and item["coins"] >= MIN_COINS
                for item in validation_checks
            )
        )
        rows.append(
            {
                "profile_id": profile_id,
                "cohort": cohort,
                "role": profile["role"],
                "blocks": ",".join(profile["blocks"]),
                "required_ready_blocks": ",".join(profile["required_ready_blocks"]),
                "status": (
                    "supported_for_generation11_freqai"
                    if supported
                    else "parked_outcome_blind_insufficient_common_support"
                ),
                "period_checks": json.dumps(checks, sort_keys=True),
                "reaction_outcomes_opened": False,
            }
        )
    return DataFrame.from_records(rows)


def target_frame(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    anchors = select_independent_anchors(frame)
    output = DataFrame(
        {
            "date": pd.to_datetime(anchors["event_time"], utc=True),
            "period": anchors["period"].astype(str),
            "pair": anchors["pair"].astype(str),
            "cohort": anchors["cohort"].astype(str),
            "market_group": anchors["market_group"].astype(str),
            "smart_contract_platform": (
                anchors["smart_contract_platform"].fillna(False).astype(bool)
            ),
        }
    ).reset_index(drop=True)
    evaluation = output.copy()
    evaluation["anchor_level_family"] = anchors["level_family"].astype(str).to_numpy()
    evaluation["anchor_level_name"] = anchors["level_name"].astype(str).to_numpy()
    evaluation["anchor_source_timeframe"] = anchors["source_timeframe"].astype(str).to_numpy()
    evaluation["anchor_representation"] = anchors["representation"].astype(str).to_numpy()
    width = numeric(anchors, "zone_half_width_atr").fillna(0.5)
    threshold = np.maximum(0.5, width.to_numpy(dtype=float))
    for horizon in g11f.g11z.HORIZONS:
        excursion = numeric(anchors, f"abs_excursion_atr_h{horizon}")
        volume = numeric(anchors, f"volume_ratio_h{horizon}")
        reaction = (excursion.ge(threshold) & volume.ge(1.25)).astype(float)
        output[f"&-g11_reaction_h{horizon}"] = reaction.to_numpy()
        output[f"&-g11_volume_ratio_h{horizon}"] = volume.to_numpy()
        evaluation[f"&-g11_reaction_h{horizon}"] = reaction.to_numpy()
        evaluation[f"&-g11_volume_ratio_h{horizon}"] = volume.to_numpy()
    return output, evaluation


def materialize_targets(
    item: dict[str, Any], event_dir: Path, evaluation_dir: Path
) -> dict[str, Any]:
    source = Path(item["source_path"])
    frame = pd.read_parquet(
        source,
        columns=list(TARGET_INPUT_COLUMNS),
        filters=[("control", "==", "actual")],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    targets, evaluation = target_frame(frame)
    support = pd.read_parquet(item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    ready_columns = [column for column in support if column.startswith("ready__")]
    output = targets.merge(
        support[["date", *ready_columns]], on="date", how="inner", validate="one_to_one"
    )
    evaluation = evaluation.merge(
        support[["date", *ready_columns]], on="date", how="inner", validate="one_to_one"
    )
    stem = g0.pair_file_stem(item["pair"])
    event_path = event_dir / f"{stem}.parquet"
    evaluation_path = evaluation_dir / f"{stem}.parquet"
    g0.atomic_write_parquet(output, event_path)
    g0.atomic_write_parquet(evaluation, evaluation_path)
    return {
        **item,
        "event_path": str(event_path),
        "event_sha256": g0.sha256_file(event_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": g0.sha256_file(evaluation_path),
        "event_rows": len(output),
        "reaction_outcomes_opened_after_support_freeze": True,
    }


def build_cohort_cache(
    *, cohort: str, workers: int, overwrite: bool = False
) -> dict[str, Any]:
    frozen, event_manifest = load_contracts()
    manifest_path = RECORD_ROOT / f"{cohort}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_generation11_freqai_cache":
            raise ValueError(f"Incomplete existing Generation 11 cache: {manifest_path}")
        return existing
    artifact_dir = ARTIFACT_ROOT / cohort
    feature_dir = artifact_dir / "feature_cache"
    support_dir = artifact_dir / "outcome_blind_support_cache"
    event_dir = artifact_dir / "event_cache"
    evaluation_dir = artifact_dir / "evaluation_cache"
    for path in (feature_dir, support_dir, event_dir, evaluation_dir):
        path.mkdir(parents=True, exist_ok=True)
    tasks = [
        CacheTask(
            cohort=cohort,
            pair=str(item["pair"]),
            source_path=Path(item["event_path"]),
            source_sha256=str(item["event_sha256"]),
            feature_dir=feature_dir,
            support_dir=support_dir,
        )
        for item in event_manifest["tasks"]
        if str(item["cohort"]) == cohort
    ]
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {pool.submit(build_pair_feature_support, task): task for task in tasks}
        for future in as_completed(futures):
            item = future.result()
            inventory.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g11_outcome_blind_features",
                        "cohort": cohort,
                        "pair": item["pair"],
                        "independent_contacts": item["independent_contact_rows"],
                    }
                ),
                flush=True,
            )
    pair_order = [task.pair for task in tasks]
    inventory.sort(key=lambda item: pair_order.index(str(item["pair"])))
    decisions = support_decisions(frozen, cohort, inventory)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    decisions_path = RECORD_ROOT / f"{cohort}_profile_support.csv"
    g0.atomic_write_csv(decisions, decisions_path)
    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    support_freeze = {
        "schema_version": 1,
        "cohort": cohort,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation11_cache_targets",
        "profiles": len(decisions),
        "supported_profiles": int(decisions["status"].str.startswith("supported").sum()),
        "parked_profiles": decisions.loc[
            ~decisions["status"].str.startswith("supported"), "profile_id"
        ].tolist(),
        "reaction_outcome_columns_read": False,
        "feature_inventory": [
            {
                key: item[key]
                for key in (
                    "pair",
                    "feature_path",
                    "feature_sha256",
                    "support_path",
                    "support_sha256",
                    "independent_contact_rows",
                )
            }
            for item in inventory
        ],
        "decisions": artifact(decisions_path),
        "generation11_freqai_freeze": artifact(g11f.FREEZE_PATH),
    }
    g0.atomic_write_json(support_freeze, support_freeze_path)
    # The exact predictor support is now frozen; only here may targets be opened.
    materialized = [
        materialize_targets(item, event_dir, evaluation_dir) for item in inventory
    ]
    manifest = {
        "schema_version": 1,
        "cache_id": CACHE_ID,
        "cohort": cohort,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation11_freqai_cache",
        "pairs": pair_order,
        "feature_blocks": list(g11f.CURRENT_BLOCKS),
        "targets": list(g11f.TARGETS),
        "inventory": materialized,
        "profile_support": artifact(decisions_path),
        "supported_profiles": decisions.loc[
            decisions["status"].str.startswith("supported"), "profile_id"
        ].tolist(),
        "parked_profiles": decisions.loc[
            ~decisions["status"].str.startswith("supported"), "profile_id"
        ].tolist(),
        "source_contracts": {
            "generation11_freqai_freeze": artifact(g11f.FREEZE_PATH),
            "outcome_blind_support_freeze": artifact(support_freeze_path),
            "generation6_event_manifest": artifact(g11f.g11z.G6_EVENT_MANIFEST),
        },
        "integrity": {
            "outcome_blind_support_frozen_before_targets": True,
            "causal_placebo_violations": int(
                sum(
                    row["stale_timestamp_violations"] + row["shuffled_self_matches"]
                    for item in materialized
                    for row in item["placebo_audit"]
                )
            ),
            "profit_used": False,
            "future_signed_direction": False,
        },
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Generation 11 FreqAI caches.")
    parser.add_argument("--cohort", choices=("normal", "meme", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    summaries = []
    for cohort in cohorts:
        manifest = build_cohort_cache(
            cohort=cohort,
            workers=max(1, min(args.workers, MAX_WORKERS)),
            overwrite=args.overwrite,
        )
        summaries.append(
            {
                "cohort": cohort,
                "status": manifest["status"],
                "pairs": len(manifest["pairs"]),
                "supported_profiles": len(manifest["supported_profiles"]),
                "parked_profiles": len(manifest["parked_profiles"]),
                "manifest": str((RECORD_ROOT / f"{cohort}_manifest.json").resolve()),
            }
        )
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
