"""Build outcome-blind Generation 13 MTF support, then long-horizon targets."""

from __future__ import annotations

# Bind numerical pools before pandas/NumPy imports.
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
import gc
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    market_reaction_zone_generation6_preflight as g6p,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as g11c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)


CACHE_ID = "g13_broad_support_20260822a"
RECORD_ROOT = g13z.OUTPUT_ROOT / "generation13_shared" / CACHE_ID
ARTIFACT_ROOT = g0.LARGE_ARTIFACT_ROOT / "generation13_shared" / CACHE_ID
MAX_WORKERS = 4
LONG_COOLDOWN_HOURS = 48
MIN_DEVELOPMENT_EVENTS = 100
MIN_VALIDATION_EVENTS = 50
MIN_COINS = 5
MIN_SOURCE_TIMEFRAME_EVENTS = 20
SOURCE_TIMEFRAMES = ("4h", "8h", "1d")
LONG_ANCHOR_INPUT_COLUMNS = ("level_price", "zone_half_width", "base_atr")


@dataclass(frozen=True)
class CacheTask:
    cohort: str
    pair: str
    source_path: str
    source_sha256: str
    g11_feature_path: str
    g11_support_path: str
    manifest_path: str
    mtf_feature_dir: str
    mtf_support_dir: str
    long_feature_dir: str
    long_support_dir: str
    long_anchor_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_freeze() -> dict[str, Any]:
    if not g13z.FREEZE_PATH.is_file():
        raise FileNotFoundError(g13z.FREEZE_PATH)
    frozen = json.loads(g13z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation13_outcomes":
        raise ValueError("Generation 13 definition is not frozen.")
    return frozen


def load_g11_manifest(cohort: str) -> dict[str, Any]:
    path = g13z.G11_CACHE_ROOT / f"{cohort}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation11_freqai_cache":
        raise ValueError(f"Generation 11 cache is not terminal for {cohort}.")
    return manifest


def manifest_for_cohort(cohort: str) -> Path:
    event_manifest = json.loads(g13z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    source_manifests = event_manifest.get("source_manifests", {})
    if cohort not in source_manifests:
        raise KeyError(f"No Generation 6 source manifest for {cohort}.")
    return Path(source_manifests[cohort]["path"])


def resample_ohlcv(frame: DataFrame, timeframe: str) -> DataFrame:
    rule = {"4h": "4h", "8h": "8h", "1d": "1D"}[timeframe]
    source = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    source = source.dropna(subset=["date"]).set_index("date").sort_index()
    output = source.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    output = output.dropna(subset=["open", "high", "low", "close"]).reset_index()
    output["base_atr"] = g0.wilder_atr(output, 14).shift(1)
    return output


def higher_timeframe_states(
    raw: DataFrame, dates: Series, timeframe: str
) -> tuple[DataFrame, int]:
    resampled = resample_ohlcv(raw, timeframe)
    state, _ = g6p.causal_state_features(resampled)
    state.insert(0, "state_available_at", resampled["date"])
    target = DataFrame({"date": pd.to_datetime(dates, utc=True, errors="coerce")})
    merged = pd.merge_asof(
        target.sort_values("date"),
        state.sort_values("state_available_at"),
        left_on="date",
        right_on="state_available_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_index()
    valid = merged["state_available_at"].notna()
    violations = int((merged.loc[valid, "state_available_at"] > merged.loc[valid, "date"]).sum())
    output = DataFrame({"date": merged["date"]})
    for family, suffixes in g13z.MTF_FAMILIES.items():
        block = g13z.mtf_block(timeframe, family)
        for suffix in suffixes:
            source_column = "rsi14" if suffix == "rsi14_centered" else suffix
            values = pd.to_numeric(merged[source_column], errors="coerce")
            if suffix == "rsi14_centered":
                values = values - 50.0
            output[f"{block}__{suffix}"] = values.to_numpy()
    return output, violations


def complete_block(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No Generation 13 columns for block {block!r}.")
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )


def select_long_anchors(frame: DataFrame) -> DataFrame:
    source = g11c.causal_deduplicate(frame.loc[frame["control"].eq("actual")].copy())
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    source = source.loc[source["source_timeframe"].isin(SOURCE_TIMEFRAMES)].copy()
    source["_distance"] = pd.to_numeric(
        source["contact_close_distance_atr"], errors="coerce"
    )
    source["_score"] = pd.to_numeric(source["level_score"], errors="coerce")
    anchors = source.sort_values(
        ["pair", "period", "event_time", "_distance", "_score", "level_name"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    ).drop_duplicates(["pair", "period", "event_time"], keep="first")
    keep = Series(False, index=anchors.index)
    cooldown = pd.Timedelta(hours=LONG_COOLDOWN_HOURS)
    for _, group in anchors.groupby(["pair", "period"], observed=True, sort=False):
        last: pd.Timestamp | None = None
        for index, timestamp in zip(group.index, group["event_time"], strict=True):
            if last is None or timestamp - last >= cooldown:
                keep.loc[index] = True
                last = timestamp
    return anchors.loc[keep].drop(columns=["_distance", "_score"]).reset_index(drop=True)


def float32_features(frame: DataFrame) -> DataFrame:
    feature_columns = [column for column in frame if "__" in column]
    output = frame[["date", *feature_columns]].copy()
    for column in feature_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output


def support_frame(frame: DataFrame, blocks: Sequence[str]) -> DataFrame:
    output = frame[["date", "period"]].copy()
    if "anchor_source_timeframe" in frame:
        output["anchor_source_timeframe"] = frame["anchor_source_timeframe"].astype(str)
    for block in blocks:
        output[f"ready__{block}"] = complete_block(frame, block)
    return output


def build_pair_support(task: CacheTask) -> dict[str, Any]:
    stem = g0.pair_file_stem(task.pair)
    mtf_feature_path = Path(task.mtf_feature_dir) / f"{stem}.parquet"
    mtf_support_path = Path(task.mtf_support_dir) / f"{stem}.parquet"
    long_feature_path = Path(task.long_feature_dir) / f"{stem}.parquet"
    long_support_path = Path(task.long_support_dir) / f"{stem}.parquet"
    long_anchor_path = Path(task.long_anchor_dir) / f"{stem}.parquet"
    paths = (
        mtf_feature_path,
        mtf_support_path,
        long_feature_path,
        long_support_path,
        long_anchor_path,
    )
    if all(path.is_file() for path in paths) and not task.overwrite:
        return {
            "cohort": task.cohort,
            "pair": task.pair,
            "source_path": task.source_path,
            "source_sha256": task.source_sha256,
            "mtf_feature_path": str(mtf_feature_path),
            "mtf_feature_sha256": g0.sha256_file(mtf_feature_path),
            "mtf_support_path": str(mtf_support_path),
            "mtf_support_sha256": g0.sha256_file(mtf_support_path),
            "long_feature_path": str(long_feature_path),
            "long_feature_sha256": g0.sha256_file(long_feature_path),
            "long_support_path": str(long_support_path),
            "long_support_sha256": g0.sha256_file(long_support_path),
            "long_anchor_path": str(long_anchor_path),
            "long_anchor_sha256": g0.sha256_file(long_anchor_path),
            "status": "existing_outcome_blind_support",
        }

    manifest = g0.load_manifest(Path(task.manifest_path))
    raw = g0.load_ohlcv(g0.ohlcv_path(task.pair, manifest["data"]["base_timeframe"]))
    g11_features = pd.read_parquet(task.g11_feature_path)
    g11_features["date"] = pd.to_datetime(g11_features["date"], utc=True, errors="coerce")
    g11_support = pd.read_parquet(task.g11_support_path)
    g11_support["date"] = pd.to_datetime(g11_support["date"], utc=True, errors="coerce")
    if g11_features["date"].duplicated().any() or g11_support["date"].duplicated().any():
        raise ValueError(f"Duplicate Generation 11 support dates for {task.pair}.")

    mtf = g11_support[["date", "period"]].copy()
    clock_violations = 0
    mtf_blocks: list[str] = []
    for timeframe in g13z.MTF_TIMEFRAMES:
        state, violations = higher_timeframe_states(raw, mtf["date"], timeframe)
        clock_violations += violations
        mtf = mtf.merge(state, on="date", how="left", validate="one_to_one")
        for family in g13z.MTF_FAMILIES:
            mtf_blocks.append(g13z.mtf_block(timeframe, family))
    if clock_violations:
        raise ValueError(f"Future higher-timeframe state for {task.pair}: {clock_violations}")
    mtf, mtf_placebo_audit = g8c.attach_stale_and_shuffled(
        mtf,
        pair=task.pair,
        blocks=mtf_blocks,
    )
    mtf_placebo_violations = sum(
        int(row["stale_timestamp_violations"] or row["shuffled_self_matches"])
        for row in mtf_placebo_audit
    )
    if mtf_placebo_violations:
        raise ValueError(f"Generation 13 MTF placebo violation for {task.pair}.")
    all_mtf_blocks = [
        *mtf_blocks,
        *(f"{block}_stale" for block in mtf_blocks),
        *(f"{block}_shuffled" for block in mtf_blocks),
    ]
    mtf_features = g11_features.merge(
        float32_features(mtf), on="date", how="inner", validate="one_to_one"
    )
    mtf_support = g11_support.merge(
        support_frame(mtf, all_mtf_blocks),
        on=["date", "period"],
        how="inner",
        validate="one_to_one",
    )

    source = pd.read_parquet(
        task.source_path,
        columns=list(
            dict.fromkeys((*g11c.FEATURE_INPUT_COLUMNS, *LONG_ANCHOR_INPUT_COLUMNS))
        ),
        filters=[("control", "==", "actual")],
    )
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="coerce")
    anchors = select_long_anchors(source)
    current = g11c.build_feature_frame(source, anchors)
    long_full, long_placebo_audit = g8c.attach_stale_and_shuffled(
        current,
        pair=task.pair,
        blocks=g11c.g11f.CURRENT_BLOCKS,
    )
    long_placebo_violations = sum(
        int(row["stale_timestamp_violations"] or row["shuffled_self_matches"])
        for row in long_placebo_audit
    )
    if long_placebo_violations:
        raise ValueError(f"Generation 13 long placebo violation for {task.pair}.")
    long_full = long_full.copy()
    long_full["anchor_source_timeframe"] = (
        anchors["source_timeframe"].astype(str).to_numpy()
    )
    all_long_blocks = [
        *g11c.g11f.CURRENT_BLOCKS,
        *(f"{block}_stale" for block in g11c.g11f.CURRENT_BLOCKS),
        *(f"{block}_shuffled" for block in g11c.g11f.CURRENT_BLOCKS),
    ]
    long_features = float32_features(long_full)
    long_support = support_frame(long_full, all_long_blocks)
    anchor_columns = [
        "cohort",
        "pair",
        "event_time",
        "period",
        "base_index",
        "level_family",
        "level_name",
        "source_timeframe",
        "representation",
        "level_price",
        "zone_half_width",
        "zone_half_width_atr",
        "base_atr",
        "approach_code",
        "market_group",
        "smart_contract_platform",
    ]
    long_anchors = anchors[anchor_columns].rename(columns={"event_time": "date"})

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(mtf_features, mtf_feature_path)
    g0.atomic_write_parquet(mtf_support, mtf_support_path)
    g0.atomic_write_parquet(long_features, long_feature_path)
    g0.atomic_write_parquet(long_support, long_support_path)
    g0.atomic_write_parquet(long_anchors, long_anchor_path)
    result = {
        "cohort": task.cohort,
        "pair": task.pair,
        "source_path": task.source_path,
        "source_sha256": task.source_sha256,
        "mtf_feature_path": str(mtf_feature_path),
        "mtf_feature_sha256": g0.sha256_file(mtf_feature_path),
        "mtf_support_path": str(mtf_support_path),
        "mtf_support_sha256": g0.sha256_file(mtf_support_path),
        "mtf_rows": len(mtf_support),
        "mtf_placebo_audit": mtf_placebo_audit,
        "higher_timeframe_clock_violations": clock_violations,
        "long_feature_path": str(long_feature_path),
        "long_feature_sha256": g0.sha256_file(long_feature_path),
        "long_support_path": str(long_support_path),
        "long_support_sha256": g0.sha256_file(long_support_path),
        "long_anchor_path": str(long_anchor_path),
        "long_anchor_sha256": g0.sha256_file(long_anchor_path),
        "long_rows": len(long_support),
        "long_placebo_audit": long_placebo_audit,
        "reaction_outcome_columns_read": False,
        "status": "built_outcome_blind_support",
    }
    del raw, g11_features, g11_support, mtf, source, current, long_full
    gc.collect()
    return result


def relabel_periods(frame: DataFrame, cohort: str) -> DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
    if cohort != "normal":
        return output
    for definition in g13z.g12z.RECENT_PERIODS:
        start = pd.Timestamp(definition["start"])
        stop = pd.Timestamp(definition["stop"])
        selected = output["date"].ge(start) & output["date"].lt(stop)
        output.loc[selected, "period"] = definition["period"]
    return output


def profile_support_decisions(
    frozen: dict[str, Any],
    cohort: str,
    inventory: Sequence[dict[str, Any]],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    period_sets = {
        "normal": (
            "development",
            *g13z.g11base.VALIDATION_PERIODS["normal"],
            *(item["period"] for item in g13z.g12z.RECENT_PERIODS),
        ),
        "meme": (
            "meme_development",
            *g13z.g11base.VALIDATION_PERIODS["meme"],
        ),
    }
    for stage, support_key in (
        (g13z.MTF_STAGE, "mtf_support_path"),
        (g13z.LONG_STAGE, "long_support_path"),
    ):
        frames = []
        for item in inventory:
            support = relabel_periods(pd.read_parquet(item[support_key]), cohort)
            support["pair"] = item["pair"]
            frames.append(support)
        combined = pd.concat(frames, ignore_index=True)
        for profile_id, profile in frozen["profiles"].items():
            if profile["cohort"] != cohort or profile["stage"] != stage:
                continue
            ready_columns = [
                f"ready__{block}" for block in profile["required_ready_blocks"]
            ]
            missing = sorted(set(ready_columns).difference(combined.columns))
            if missing:
                raise ValueError(f"Generation 13 support lacks {missing} for {profile_id}.")
            ready = combined[ready_columns].fillna(False).astype(bool).all(axis=1)
            checks = []
            for period in period_sets[cohort]:
                selected = combined.loc[ready & combined["period"].eq(period)]
                per_pair = selected.groupby("pair", observed=True).size()
                check: dict[str, Any] = {
                    "period": period,
                    "events": len(selected),
                    "coins": int((per_pair > 0).sum()),
                    "minimum_pair_events": int(per_pair.min()) if len(per_pair) else 0,
                }
                if stage == g13z.LONG_STAGE and period not in {
                    "development",
                    "meme_development",
                }:
                    by_tf = {
                        timeframe: int(
                            selected["anchor_source_timeframe"].eq(timeframe).sum()
                        )
                        for timeframe in SOURCE_TIMEFRAMES
                    }
                    check["events_by_source_timeframe"] = by_tf
                checks.append(check)
            development, *evaluation = checks
            basic = (
                development["events"] >= MIN_DEVELOPMENT_EVENTS
                and development["coins"] >= MIN_COINS
                and all(
                    row["events"] >= MIN_VALIDATION_EVENTS and row["coins"] >= MIN_COINS
                    for row in evaluation
                )
            )
            timeframe_support = all(
                all(
                    count >= MIN_SOURCE_TIMEFRAME_EVENTS
                    for count in row.get("events_by_source_timeframe", {}).values()
                )
                for row in evaluation
            )
            supported = basic and timeframe_support
            rows.append(
                {
                    "profile_id": profile_id,
                    "stage": stage,
                    "cohort": cohort,
                    "role": profile["role"],
                    "required_ready_blocks": ",".join(profile["required_ready_blocks"]),
                    "status": (
                        "supported_for_generation13_freqai"
                        if supported
                        else "parked_outcome_blind_insufficient_common_support"
                    ),
                    "period_checks": json.dumps(checks, sort_keys=True),
                    "reaction_outcomes_opened": False,
                }
            )
    return DataFrame.from_records(rows)


def materialize_long_targets(
    item: dict[str, Any],
    *,
    manifest_path: Path,
    event_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    anchors = pd.read_parquet(item["long_anchor_path"])
    anchors["date"] = pd.to_datetime(anchors["date"], utc=True, errors="coerce")
    manifest = g0.load_manifest(manifest_path)
    base = g0.prepare_base_market_frame(item["pair"], manifest)
    end = pd.Timestamp(manifest["data"]["analysis_end_utc_exclusive"])
    base = base.loc[base["date"] < end].reset_index(drop=True)
    indexes = pd.to_numeric(anchors["base_index"], errors="raise").to_numpy(dtype=np.int64)
    if (indexes < 0).any() or (indexes >= len(base)).any():
        raise ValueError(f"Generation 13 anchor index outside base data for {item['pair']}.")
    base_dates = pd.to_datetime(base.iloc[indexes]["date"], utc=True).reset_index(drop=True)
    if not base_dates.equals(anchors["date"].reset_index(drop=True)):
        raise ValueError(f"Generation 13 anchor/base clock mismatch for {item['pair']}.")
    paths = g0.future_path_matrices(base, max(g13z.LONG_HORIZONS))
    outcomes = DataFrame(index=anchors.index)
    g0.append_reaction_metrics(
        output=outcomes,
        merged=base,
        paths=paths,
        indexes=indexes,
        levels=pd.to_numeric(anchors["level_price"], errors="coerce").to_numpy(dtype=float),
        widths=pd.to_numeric(anchors["zone_half_width"], errors="coerce").to_numpy(dtype=float),
        atr=pd.to_numeric(anchors["base_atr"], errors="coerce").to_numpy(dtype=float),
        approach_code=pd.to_numeric(
            anchors["approach_code"], errors="coerce"
        ).to_numpy(dtype=np.int8),
        horizons=g13z.LONG_HORIZONS,
    )
    target = anchors[
        [
            "date",
            "period",
            "pair",
            "cohort",
            "market_group",
            "smart_contract_platform",
        ]
    ].copy()
    evaluation = target.copy()
    evaluation["anchor_level_family"] = anchors["level_family"].astype(str)
    evaluation["anchor_level_name"] = anchors["level_name"].astype(str)
    evaluation["anchor_source_timeframe"] = anchors["source_timeframe"].astype(str)
    evaluation["anchor_representation"] = anchors["representation"].astype(str)
    threshold = np.maximum(
        0.5,
        pd.to_numeric(anchors["zone_half_width_atr"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=float),
    )
    for horizon in g13z.LONG_HORIZONS:
        excursion = pd.to_numeric(
            outcomes[f"abs_excursion_atr_h{horizon}"], errors="coerce"
        )
        volume = pd.to_numeric(outcomes[f"volume_ratio_h{horizon}"], errors="coerce")
        reaction = (excursion.ge(threshold) & volume.ge(1.25)).astype(float)
        reaction.loc[excursion.isna() | volume.isna()] = np.nan
        reaction_name = f"&-g13_reaction_h{horizon}"
        volume_name = f"&-g13_volume_ratio_h{horizon}"
        target[reaction_name] = reaction.to_numpy()
        target[volume_name] = volume.to_numpy()
        evaluation[reaction_name] = reaction.to_numpy()
        evaluation[volume_name] = volume.to_numpy()
    support = pd.read_parquet(item["long_support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    ready_columns = [column for column in support if column.startswith("ready__")]
    target = target.merge(
        support[["date", *ready_columns]], on="date", how="inner", validate="one_to_one"
    )
    evaluation = evaluation.merge(
        support[["date", *ready_columns]], on="date", how="inner", validate="one_to_one"
    )
    stem = g0.pair_file_stem(item["pair"])
    event_path = event_dir / f"{stem}.parquet"
    evaluation_path = evaluation_dir / f"{stem}.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(target, event_path)
    g0.atomic_write_parquet(evaluation, evaluation_path)
    return {
        **item,
        "long_event_path": str(event_path),
        "long_event_sha256": g0.sha256_file(event_path),
        "long_evaluation_path": str(evaluation_path),
        "long_evaluation_sha256": g0.sha256_file(evaluation_path),
        "long_event_rows": len(target),
        "long_targets_opened_after_support_freeze": True,
    }


def materialize_mtf_targets(
    item: dict[str, Any],
    *,
    g11_item: dict[str, Any],
    event_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    support = pd.read_parquet(item["mtf_support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    ready_columns = [column for column in support if column.startswith("ready__")]
    outputs: dict[str, DataFrame] = {}
    for kind in ("event", "evaluation"):
        source = pd.read_parquet(g11_item[f"{kind}_path"])
        source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
        source_ready = [column for column in source if column.startswith("ready__")]
        merged = source.drop(columns=source_ready).merge(
            support[["date", *ready_columns]],
            on="date",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(support) or len(merged) != len(source):
            raise ValueError(
                f"Generation 13 MTF target/support clock mismatch for {item['pair']}."
            )
        outputs[kind] = merged
    stem = g0.pair_file_stem(item["pair"])
    event_path = event_dir / f"{stem}.parquet"
    evaluation_path = evaluation_dir / f"{stem}.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(outputs["event"], event_path)
    g0.atomic_write_parquet(outputs["evaluation"], evaluation_path)
    return {
        **item,
        "mtf_event_path": str(event_path),
        "mtf_event_sha256": g0.sha256_file(event_path),
        "mtf_evaluation_path": str(evaluation_path),
        "mtf_evaluation_sha256": g0.sha256_file(evaluation_path),
        "mtf_event_rows": len(outputs["event"]),
        "mtf_targets_joined_after_support_freeze": True,
    }


def tasks_for_cohort(cohort: str, *, overwrite: bool) -> tuple[list[CacheTask], dict[str, Any]]:
    g11 = load_g11_manifest(cohort)
    manifest_path = manifest_for_cohort(cohort)
    mtf_feature_dir = ARTIFACT_ROOT / cohort / "mtf_feature_cache"
    mtf_support_dir = ARTIFACT_ROOT / cohort / "mtf_support_cache"
    long_feature_dir = ARTIFACT_ROOT / cohort / "long_feature_cache"
    long_support_dir = ARTIFACT_ROOT / cohort / "long_support_cache"
    long_anchor_dir = ARTIFACT_ROOT / cohort / "long_anchor_support"
    tasks = [
        CacheTask(
            cohort=cohort,
            pair=str(item["pair"]),
            source_path=str(item["source_path"]),
            source_sha256=str(item["source_sha256"]),
            g11_feature_path=str(item["feature_path"]),
            g11_support_path=str(item["support_path"]),
            manifest_path=str(manifest_path),
            mtf_feature_dir=str(mtf_feature_dir),
            mtf_support_dir=str(mtf_support_dir),
            long_feature_dir=str(long_feature_dir),
            long_support_dir=str(long_support_dir),
            long_anchor_dir=str(long_anchor_dir),
            overwrite=overwrite,
        )
        for item in g11["inventory"]
    ]
    return tasks, g11


def run_tasks(tasks: Sequence[CacheTask], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(build_pair_support, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "cohort": task.cohort,
                    "pair": task.pair,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g13_outcome_blind_support",
                        "processed": len(results),
                        "total": len(tasks),
                        "pair": task.pair,
                        "status": result["status"],
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda row: str(row["pair"]))


def build_cohort_cache(
    *, cohort: str, workers: int, overwrite: bool = False
) -> dict[str, Any]:
    frozen = load_freeze()
    manifest_path = RECORD_ROOT / f"{cohort}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_generation13_cache":
            return existing
    tasks, g11 = tasks_for_cohort(cohort, overwrite=overwrite)
    inventory = run_tasks(tasks, max(1, min(workers, MAX_WORKERS)))
    failures = [row for row in inventory if row.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"Generation 13 support failures: {failures}")
    decisions = profile_support_decisions(frozen, cohort, inventory)
    decisions_path = RECORD_ROOT / f"{cohort}_profile_support.csv"
    g0.atomic_write_csv(decisions, decisions_path)
    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    support_freeze = {
        "schema_version": 1,
        "generation": 13,
        "cohort": cohort,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation13_long_targets",
        "profiles": len(decisions),
        "supported_profiles": int(
            decisions["status"].eq("supported_for_generation13_freqai").sum()
        ),
        "parked_profiles": decisions.loc[
            ~decisions["status"].eq("supported_for_generation13_freqai"), "profile_id"
        ].tolist(),
        "integrity": {
            "reaction_outcome_columns_read": False,
            "higher_timeframe_clock_violations": int(
                sum(row.get("higher_timeframe_clock_violations", 0) for row in inventory)
            ),
            "placebo_violations": int(
                sum(
                    int(audit["stale_timestamp_violations"] or audit["shuffled_self_matches"])
                    for row in inventory
                    for key in ("mtf_placebo_audit", "long_placebo_audit")
                    for audit in row.get(key, [])
                )
            ),
            "profit_used": False,
            "future_signed_direction": False,
        },
        "profile_support": artifact(decisions_path),
        "generation13_freeze": artifact(g13z.FREEZE_PATH),
    }
    g0.atomic_write_json(support_freeze, support_freeze_path)
    event_dir = ARTIFACT_ROOT / cohort / "long_event_cache"
    evaluation_dir = ARTIFACT_ROOT / cohort / "long_evaluation_cache"
    materialized = [
        materialize_long_targets(
            item,
            manifest_path=manifest_for_cohort(cohort),
            event_dir=event_dir,
            evaluation_dir=evaluation_dir,
        )
        for item in inventory
    ]
    g11_by_pair = {str(item["pair"]): item for item in g11["inventory"]}
    mtf_event_dir = ARTIFACT_ROOT / cohort / "mtf_event_cache"
    mtf_evaluation_dir = ARTIFACT_ROOT / cohort / "mtf_evaluation_cache"
    materialized = [
        materialize_mtf_targets(
            item,
            g11_item=g11_by_pair[str(item["pair"])],
            event_dir=mtf_event_dir,
            evaluation_dir=mtf_evaluation_dir,
        )
        for item in materialized
    ]
    manifest = {
        "schema_version": 1,
        "generation": 13,
        "cache_id": CACHE_ID,
        "cohort": cohort,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation13_cache",
        "pairs": [str(item["pair"]) for item in materialized],
        "inventory": materialized,
        "profile_support": artifact(decisions_path),
        "supported_profiles": decisions.loc[
            decisions["status"].eq("supported_for_generation13_freqai"), "profile_id"
        ].tolist(),
        "parked_profiles": decisions.loc[
            ~decisions["status"].eq("supported_for_generation13_freqai"), "profile_id"
        ].tolist(),
        "source_contracts": {
            "generation13_freeze": artifact(g13z.FREEZE_PATH),
            "generation13_outcome_blind_support_freeze": artifact(support_freeze_path),
            "generation11_cache": artifact(
                g13z.G11_CACHE_ROOT / f"{cohort}_manifest.json"
            ),
            "generation6_event_manifest": artifact(g13z.G6_EVENT_MANIFEST),
        },
        "integrity": {
            "support_frozen_before_long_targets": True,
            "higher_timeframe_clock_violations": 0,
            "placebo_violations": 0,
            "long_outcome_cooldown_hours": LONG_COOLDOWN_HOURS,
            "profit_used": False,
            "future_signed_direction": False,
        },
        "g11_event_dir": str(
            Path(g11["inventory"][0]["event_path"]).parent.resolve()
        ),
        "g11_evaluation_dir": str(
            Path(g11["inventory"][0]["evaluation_path"]).parent.resolve()
        ),
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Generation 13 causal caches.")
    parser.add_argument("--cohort", choices=("all", *g13z.COHORTS), default="all")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cohorts = g13z.COHORTS if args.cohort == "all" else (args.cohort,)
    results = [
        build_cohort_cache(
            cohort=cohort,
            workers=max(1, min(args.workers, MAX_WORKERS)),
            overwrite=args.overwrite,
        )
        for cohort in cohorts
    ]
    print(
        json.dumps(
            [
                {
                    "cohort": result["cohort"],
                    "status": result["status"],
                    "pairs": len(result["pairs"]),
                    "supported_profiles": len(result["supported_profiles"]),
                    "parked_profiles": len(result["parked_profiles"]),
                    "manifest": str((RECORD_ROOT / f"{result['cohort']}_manifest.json").resolve()),
                }
                for result in results
            ],
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
