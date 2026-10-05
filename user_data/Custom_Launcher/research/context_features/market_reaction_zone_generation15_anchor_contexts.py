"""Score Generation 15 contact, lifecycle, cluster, and order-book siblings.

The sibling definitions were frozen before this module opens any Generation 15
outcome.  All matching variables are available no later than the completed contact
candle; every scored path starts on the following 1h candle.
"""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
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
import hashlib
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
    market_reaction_zone_generation1_localization as g1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as g11c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_freeze as g15z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_matched_paths as g15m,
)


DEFAULT_RUN_ID = "g15_anchor_contexts_20260822a"
RECORD_ROOT = g15m.RECORD_ROOT.parent / "anchor_contexts"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation15_branches"
    / "anchor_contexts"
)
G8_MANIFESTS = {
    "normal": (
        REPO_ROOT
        / "user_data/research_news_data/context_features/market_reaction_zones"
        / "generation8_shared/g8_attribution_cache_20260821a/normal_manifest.json"
    ),
    "meme": (
        REPO_ROOT
        / "user_data/research_news_data/context_features/market_reaction_zones"
        / "generation8_shared/g8_attribution_cache_20260821a/meme_manifest.json"
    ),
}
MAX_WORKERS = 4
MIN_PAIR_ROWS = 20
MIN_COINS = 5
MAX_COIN_ABSOLUTE_SHARE = 0.50
HORIZONS = tuple(g15z.HORIZONS)
METRICS = tuple(g15z.PATH_METRICS)

RELATIONSHIP_STATES = (
    "same_mechanism_agreement",
    "different_mechanism_agreement",
    "any_cross_timeframe_cluster",
    "opposing_side_overlap",
)
LOCAL_MATCH_FEATURES = tuple(g13d.MATCH_FEATURES)
WIDER_MATCH_FEATURES = (
    "xm_btc_return_1h",
    "xm_btc_return_4h",
    "xm_btc_return_24h",
    "xm_btc_relative_volume",
    "xm_cohort_breadth_positive",
    "xm_cohort_dispersion",
    "xm_cohort_absolute_activity",
)
RELATIONSHIP_COLUMNS = tuple(
    f"rel__{timeframe}__{state}"
    for timeframe in ("4h", "8h", "1d")
    for state in RELATIONSHIP_STATES
)
G8_HISTORY_COLUMNS = tuple(
    f"g8_level_history{suffix}__{name}"
    for suffix in ("", "_shuffled")
    for name in (
        "mean_hours_since_prior_contact",
        "prior_contact_available_fraction",
    )
)
G8_FEATURE_COLUMNS = (
    "date",
    "g8_level_identity__contacted_level_count",
    *G8_HISTORY_COLUMNS,
)
META_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "control",
    "event_time",
    "period",
    "approach_state",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "level_score",
    "contact_range_ratio",
    "contact_volume_ratio",
    "contact_pressure_change",
    "ob_btc_model_ready",
    "ob_btc_regime",
)
OUTCOME_COLUMNS = tuple(
    dict.fromkeys(
        [
            "time_to_abs_0_5atr",
            *(
                column
                for horizon in HORIZONS
                for column in (
                    f"abs_excursion_atr_h{horizon}",
                    f"range_ratio_h{horizon}",
                    f"volume_ratio_h{horizon}",
                    f"pressure_change_h{horizon}",
                    f"dwell_fraction_h{horizon}",
                    f"crossings_h{horizon}",
                )
            ),
        ]
    )
)
READ_COLUMNS = tuple(
    dict.fromkeys(
        (
            *META_COLUMNS,
            *LOCAL_MATCH_FEATURES,
            *WIDER_MATCH_FEATURES,
            *RELATIONSHIP_COLUMNS,
            *g13d.AUDIT_SOURCE_COLUMNS,
            *OUTCOME_COLUMNS,
        )
    )
)


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    source_path: str
    source_sha256: str
    feature_path: str
    feature_sha256: str
    orderbook_path: str
    orderbook_sha256: str
    artifact_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_orderbook_source(source_manifest: dict[str, Any]) -> dict[str, Any]:
    record = source_manifest["shared_features"]["btc_orderbook"]
    path = Path(record["path"])
    if not path.is_file() or g0.sha256_file(path) != record["sha256"]:
        raise ValueError("Frozen Generation 6 BTC order-book feature source changed.")
    return record


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], list[PairTask]]:
    if not g15z.FREEZE_PATH.is_file():
        raise FileNotFoundError(g15z.FREEZE_PATH)
    freeze = json.loads(g15z.FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_generation15_outcomes":
        raise ValueError("Generation 15 plan is not frozen.")
    source_record = freeze["source_contracts"]["generation6_event_manifest"]
    source_manifest_path = Path(source_record["path"])
    if (
        not source_manifest_path.is_file()
        or g0.sha256_file(source_manifest_path) != source_record["sha256"]
    ):
        raise ValueError("Frozen Generation 6 event manifest changed.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    orderbook_record = validate_orderbook_source(source_manifest)
    orderbook_path = Path(orderbook_record["path"])
    source_by_key = {
        (str(item["cohort"]), str(item["pair"])): item
        for item in source_manifest["tasks"]
    }
    g8_records: dict[tuple[str, str], dict[str, Any]] = {}
    for cohort, path in G8_MANIFESTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_generation8_attribution_shared_cache":
            raise ValueError(f"Generation 8 {cohort} cache is not terminal.")
        for item in manifest["inventory"]:
            key = (cohort, str(item["pair"]))
            source = source_by_key.get(key)
            if source is None:
                raise ValueError(f"Generation 8 pair is outside the frozen G6 source: {key}")
            if (
                Path(item["raw_provenance_path"]).resolve()
                != Path(source["event_path"]).resolve()
                or item["raw_provenance_sha256"] != source["event_sha256"]
            ):
                raise ValueError(f"Generation 8 raw provenance drifted for {key}")
            feature_path = Path(item["feature_path"])
            if (
                not feature_path.is_file()
                or g0.sha256_file(feature_path) != item["feature_sha256"]
            ):
                raise ValueError(f"Generation 8 feature cache changed for {key}")
            g8_records[key] = item
    if set(g8_records) != set(source_by_key):
        raise ValueError("Generation 8 feature coverage does not match the frozen G6 pairs.")
    tasks = [
        PairTask(
            cohort=key[0],
            pair=key[1],
            source_path=str(source["event_path"]),
            source_sha256=str(source["event_sha256"]),
            feature_path=str(g8_records[key]["feature_path"]),
            feature_sha256=str(g8_records[key]["feature_sha256"]),
            orderbook_path=str(orderbook_path),
            orderbook_sha256=str(orderbook_record["sha256"]),
            artifact_dir="",
            overwrite=False,
        )
        for key, source in sorted(source_by_key.items())
    ]
    return freeze, source_manifest, tasks


def relationship_flags(frame: DataFrame) -> DataFrame:
    output = DataFrame({"event_time": pd.to_datetime(frame["event_time"], utc=True)})
    for state in RELATIONSHIP_STATES:
        columns = [column for column in RELATIONSHIP_COLUMNS if column.endswith(state)]
        output[state] = frame[columns].fillna(False).astype(bool).any(axis=1)
    return (
        output.groupby("event_time", observed=True, sort=False)[list(RELATIONSHIP_STATES)]
        .max()
        .reset_index()
    )


def select_g8_timeline_anchors(frame: DataFrame, dates: Series) -> DataFrame:
    selected_dates = set(pd.to_datetime(dates, utc=True, errors="coerce").dropna())
    source = g11c.causal_deduplicate(
        frame.loc[frame["event_time"].isin(selected_dates)].copy()
    )
    source["_distance"] = pd.to_numeric(
        source["contact_close_distance_atr"], errors="coerce"
    )
    source["_score"] = pd.to_numeric(source["level_score"], errors="coerce")
    anchors = (
        source.sort_values(
            ["pair", "period", "event_time", "_distance", "_score", "source_timeframe"],
            ascending=[True, True, True, True, False, True],
            na_position="last",
        )
        .drop_duplicates(["pair", "period", "event_time"], keep="first")
        .drop(columns=["_distance", "_score"])
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    missing = selected_dates.difference(set(anchors["event_time"]))
    if missing:
        raise ValueError(f"{len(missing)} frozen G8 event dates lack an actual level contact.")
    return anchors


def shuffled_within_period(values: Series, periods: Series, *, seed_key: str) -> Series:
    output = Series(index=values.index, dtype=float)
    for period, positions in periods.groupby(periods, observed=True, sort=False).groups.items():
        source = pd.to_numeric(values.loc[positions], errors="coerce").to_numpy(dtype=float)
        if len(source) < 2:
            output.loc[positions] = np.nan
            continue
        seed = int.from_bytes(
            hashlib.sha256(f"{seed_key}|{period}".encode()).digest()[:8], "big"
        )
        shift = 1 + int(np.random.default_rng(seed).integers(0, len(source) - 1))
        output.loc[positions] = np.roll(source, shift)
    return output


def classify_lifecycle(
    frame: DataFrame, suffix: str = "", *, single_contact_only: bool = True
) -> Series:
    prefix = f"g8_level_history{suffix}"
    prior_fraction = pd.to_numeric(
        frame[f"{prefix}__prior_contact_available_fraction"], errors="coerce"
    )
    hours = pd.to_numeric(
        frame[f"{prefix}__mean_hours_since_prior_contact"], errors="coerce"
    )
    single = pd.to_numeric(
        frame["g8_level_identity__contacted_level_count"], errors="coerce"
    ).eq(1.0)
    eligible = single if single_contact_only else Series(True, index=frame.index)
    output = Series("ambiguous", index=frame.index, dtype="string")
    output.loc[eligible & prior_fraction.eq(0.0)] = "fresh_30d"
    has_prior = eligible & prior_fraction.gt(0.0) & hours.notna()
    output.loc[has_prior & hours.ge(720.0)] = "fresh_30d"
    output.loc[has_prior & hours.lt(24.0)] = "repeat_within_24h"
    output.loc[has_prior & hours.ge(24.0) & hours.lt(168.0)] = "repeat_1d_to_7d"
    output.loc[has_prior & hours.ge(168.0) & hours.lt(720.0)] = "repeat_7d_to_30d"
    return output


def orderbook_state(ready_values: Series, regime_values: Series) -> Series:
    ready = ready_values.fillna(False).astype(bool)
    regime = regime_values.astype("string")
    output = Series("unavailable_or_middle", index=ready_values.index, dtype="string")
    output.loc[ready & regime.eq("active")] = "active"
    output.loc[ready & regime.eq("quiet")] = "quiet"
    return output


def shuffled_labels_within_cells(frame: DataFrame, column: str, *, seed_key: str) -> Series:
    output = Series(index=frame.index, dtype="string")
    keys = ["window", "analysis_period"]
    for key, positions in frame.groupby(keys, observed=True, sort=False).groups.items():
        source = frame.loc[positions, column].astype("string").to_numpy()
        if len(source) < 2:
            output.loc[positions] = pd.NA
            continue
        seed = int.from_bytes(
            hashlib.sha256(f"{seed_key}|{key}".encode()).digest()[:8], "big"
        )
        shift = 1 + int(np.random.default_rng(seed).integers(0, len(source) - 1))
        output.loc[positions] = np.roll(source, shift)
    return output


def prepare_anchors(task: PairTask) -> DataFrame:
    source_path = Path(task.source_path)
    feature_path = Path(task.feature_path)
    orderbook_path = Path(task.orderbook_path)
    if not source_path.is_file() or g0.sha256_file(source_path) != task.source_sha256:
        raise ValueError(f"Frozen event source changed: {source_path}")
    if not feature_path.is_file() or g0.sha256_file(feature_path) != task.feature_sha256:
        raise ValueError(f"Frozen causal feature source changed: {feature_path}")
    if (
        not orderbook_path.is_file()
        or g0.sha256_file(orderbook_path) != task.orderbook_sha256
    ):
        raise ValueError(f"Frozen BTC order-book source changed: {orderbook_path}")
    frame = pd.read_parquet(
        source_path,
        columns=list(READ_COLUMNS),
        filters=[("control", "=", "actual")],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = g13d.derive_audit_features(frame.dropna(subset=["event_time"]))
    relationships = relationship_flags(frame)
    features = pd.read_parquet(feature_path, columns=list(G8_FEATURE_COLUMNS)).rename(
        columns={"date": "event_time"}
    )
    features["event_time"] = pd.to_datetime(
        features["event_time"], utc=True, errors="coerce"
    )
    anchors = select_g8_timeline_anchors(frame, features["event_time"])
    anchors = anchors.merge(
        relationships, on="event_time", how="left", validate="many_to_one"
    )
    anchors = anchors.merge(features, on="event_time", how="left", validate="one_to_one")
    missing_features = anchors["g8_level_identity__contacted_level_count"].isna()
    if missing_features.any():
        raise ValueError(
            f"{int(missing_features.sum())} anchors lack the frozen G8 causal feature row."
        )
    anchors["arrival_value_shuffled"] = shuffled_within_period(
        anchors["pre_distance_atr"],
        anchors["period"],
        seed_key=f"g15-arrival|{task.cohort}|{task.pair}",
    )
    anchors["lifecycle_current"] = classify_lifecycle(anchors)
    anchors["lifecycle_shuffled"] = classify_lifecycle(anchors, "_shuffled")
    anchors["lifecycle_current_broad"] = classify_lifecycle(
        anchors, single_contact_only=False
    )
    anchors["lifecycle_shuffled_broad"] = classify_lifecycle(
        anchors, "_shuffled", single_contact_only=False
    )
    orderbook = pd.read_parquet(
        orderbook_path, columns=["date", "ob_btc_model_ready", "ob_btc_regime"]
    )
    orderbook["date"] = pd.to_datetime(orderbook["date"], utc=True, errors="coerce")
    current = orderbook.rename(
        columns={
            "date": "event_time",
            "ob_btc_model_ready": "ob_current_ready",
            "ob_btc_regime": "ob_current_regime",
        }
    )
    stale = orderbook.assign(date=orderbook["date"] + pd.Timedelta(hours=72)).rename(
        columns={
            "date": "event_time",
            "ob_btc_model_ready": "ob_stale_ready",
            "ob_btc_regime": "ob_stale_regime",
        }
    )
    anchors = anchors.merge(current, on="event_time", how="left", validate="one_to_one")
    anchors = anchors.merge(stale, on="event_time", how="left", validate="one_to_one")
    anchors["orderbook_current"] = orderbook_state(
        anchors["ob_current_ready"], anchors["ob_current_regime"]
    )
    anchors["orderbook_stale"] = orderbook_state(
        anchors["ob_stale_ready"], anchors["ob_stale_regime"]
    )
    anchors["cluster_count"] = anchors[list(RELATIONSHIP_STATES)].sum(axis=1)
    anchors = g13d.add_analysis_windows(anchors, task.cohort)
    anchors["orderbook_shuffled"] = shuffled_labels_within_cells(
        anchors,
        "orderbook_current",
        seed_key=f"g15-orderbook|{task.cohort}|{task.pair}",
    )
    return anchors.reset_index(drop=True)


def nearest_pairs(
    treatment: DataFrame,
    control: DataFrame,
    *,
    state_columns: Sequence[str],
    match_pre_distance: bool,
) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    left = treatment.copy()
    right = control.copy()
    if not match_pre_distance:
        left["pre_distance_atr"] = 0.0
        right["pre_distance_atr"] = 0.0
    return g1.nearest_state_pairs(
        left,
        right,
        state_columns=state_columns,
        minimum_event_separation_hours=max(HORIZONS),
    )


def match_contrast(
    frame: DataFrame,
    *,
    treatment: Series,
    control: Series,
    route_id: str,
    scope_value: str,
    comparison: str,
    state_columns: Sequence[str] = LOCAL_MATCH_FEATURES,
    match_pre_distance: bool = True,
) -> DataFrame:
    left_source = frame.loc[treatment.fillna(False)].copy()
    right_source = frame.loc[control.fillna(False)].copy()
    rows: list[DataFrame] = []
    keys = (
        "window",
        "analysis_period",
        "source_timeframe",
        "level_family",
        "approach_state",
    )
    left_groups = left_source.groupby(list(keys), observed=True, sort=False)
    right_groups = right_source.groupby(list(keys), observed=True, sort=False)
    for key in sorted(set(left_groups.groups).intersection(right_groups.groups)):
        left = left_groups.get_group(key).reset_index(drop=True)
        right = right_groups.get_group(key).reset_index(drop=True)
        pairs, audit = nearest_pairs(
            left,
            right,
            state_columns=state_columns,
            match_pre_distance=match_pre_distance,
        )
        if not pairs:
            continue
        left_positions = [item[0] for item in pairs]
        right_positions = [item[1] for item in pairs]
        selected_left = left.iloc[left_positions].reset_index(drop=True)
        selected_right = right.iloc[right_positions].reset_index(drop=True)
        output = DataFrame(
            {
                "route_id": route_id,
                "scope_value": scope_value,
                "comparison": comparison,
                "cohort": selected_left["cohort"].astype(str),
                "pair": selected_left["pair"].astype(str),
                "window": key[0],
                "analysis_period": key[1],
                "source_timeframe": key[2],
                "level_family": key[3],
                "approach_state": key[4],
                "actual_event_time": pd.to_datetime(
                    selected_left["event_time"], utc=True
                ),
                "control_event_time": pd.to_datetime(
                    selected_right["event_time"], utc=True
                ),
                "level_name": selected_left["level_name"].astype(str),
                "match_distance": [item[2] for item in pairs],
                "pre_distance_atr_abs_difference": (
                    pd.to_numeric(selected_left["pre_distance_atr"], errors="coerce")
                    - pd.to_numeric(selected_right["pre_distance_atr"], errors="coerce")
                ).abs(),
                "eligible_treatment": audit["eligible_actual"],
                "eligible_control": audit["eligible_control"],
                "state_matchable_treatment": audit["state_matchable_actual"],
            }
        )
        for column in OUTCOME_COLUMNS:
            output[f"actual__{column}"] = pd.to_numeric(
                selected_left[column], errors="coerce"
            ).to_numpy(dtype=float)
            output[f"control__{column}"] = pd.to_numeric(
                selected_right[column], errors="coerce"
            ).to_numpy(dtype=float)
        rows.append(output)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else DataFrame()


def activity_contrasts(frame: DataFrame) -> list[DataFrame]:
    volume = pd.to_numeric(frame["contact_volume_ratio"], errors="coerce").ge(1.25)
    candle_range = pd.to_numeric(frame["contact_range_ratio"], errors="coerce").ge(1.25)
    pressure = pd.to_numeric(frame["contact_pressure_change"], errors="coerce").abs().ge(0.50)
    definitions = (
        (
            "high_volume_and_range",
            volume & candle_range,
            {
                "versus_high_volume_only": volume & ~candle_range,
                "versus_high_range_only": ~volume & candle_range,
                "versus_quiet_contact": ~volume & ~candle_range,
            },
        ),
        (
            "high_volume_and_absolute_pressure",
            volume & pressure,
            {
                "versus_high_volume_only": volume & ~pressure,
                "versus_high_absolute_pressure_only": ~volume & pressure,
                "versus_quiet_contact": ~volume & ~pressure,
            },
        ),
    )
    rows: list[DataFrame] = []
    for scope, treatment, controls in definitions:
        for comparison, control in controls.items():
            rows.append(
                match_contrast(
                    frame,
                    treatment=treatment,
                    control=control,
                    route_id="contact_activity_combinations",
                    scope_value=scope,
                    comparison=comparison,
                )
            )
    return rows


def lifecycle_contrasts(frame: DataFrame) -> list[DataFrame]:
    rows: list[DataFrame] = []
    repeat_states = (
        "repeat_within_24h",
        "repeat_1d_to_7d",
        "repeat_7d_to_30d",
    )
    for state in repeat_states:
        for scope_suffix, column_suffix in (
            ("single_level_exact", ""),
            ("all_contacts_mean_descriptive", "_broad"),
        ):
            for variant in ("current", "shuffled"):
                labels = frame[f"lifecycle_{variant}{column_suffix}"]
                rows.append(
                    match_contrast(
                        frame,
                        treatment=labels.eq(state),
                        control=labels.eq("fresh_30d"),
                        route_id="contact_lifecycle_and_approach",
                        scope_value=f"{state}__{scope_suffix}",
                        comparison=f"{variant}_repeat_versus_fresh_30d",
                    )
                )
    current_distance = pd.to_numeric(frame["pre_distance_atr"], errors="coerce")
    shuffled_distance = pd.to_numeric(frame["arrival_value_shuffled"], errors="coerce")
    for variant, distance in (
        ("current", current_distance),
        ("shuffled", shuffled_distance),
    ):
        rows.append(
            match_contrast(
                frame,
                treatment=distance.ge(1.0),
                control=distance.le(0.25),
                route_id="contact_lifecycle_and_approach",
                scope_value="fast_arrival_versus_slow_arrival",
                comparison=f"{variant}_fast_versus_slow",
                match_pre_distance=False,
            )
        )
    return rows


def cluster_contrasts(frame: DataFrame) -> list[DataFrame]:
    rows: list[DataFrame] = []
    isolated = frame["cluster_count"].eq(0)
    for state in RELATIONSHIP_STATES:
        present = frame[state].fillna(False).astype(bool)
        for suffix, treatment in (
            ("any_presence", present),
            ("exclusive", present & frame["cluster_count"].eq(1)),
        ):
            rows.append(
                match_contrast(
                    frame,
                    treatment=treatment,
                    control=isolated,
                    route_id="explicit_cluster_composition",
                    scope_value=f"{state}__{suffix}",
                    comparison="versus_matched_isolated_contact",
                )
            )
    return rows


def orderbook_contrasts(frame: DataFrame) -> list[DataFrame]:
    rows: list[DataFrame] = []
    match_features = (*LOCAL_MATCH_FEATURES, *WIDER_MATCH_FEATURES)
    for variant in ("current", "stale", "shuffled"):
        labels = frame[f"orderbook_{variant}"]
        rows.append(
            match_contrast(
                frame,
                treatment=labels.eq("active"),
                control=labels.eq("quiet"),
                route_id="historical_btc_orderbook_conditioning",
                scope_value="active_versus_quiet",
                comparison=f"{variant}_active_versus_quiet",
                state_columns=match_features,
            )
        )
    return rows


def build_pair(task: PairTask) -> dict[str, Any]:
    output_path = Path(task.artifact_dir) / (
        f"{task.cohort}__{g0.pair_file_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(output_path, columns=["pair", "route_id"])
        return {
            "status": "existing",
            "cohort": task.cohort,
            "pair": task.pair,
            "contrast_rows": len(existing),
            "path": str(output_path),
            "sha256": g0.sha256_file(output_path),
        }
    anchors = prepare_anchors(task)
    outputs = [
        *activity_contrasts(anchors),
        *lifecycle_contrasts(anchors),
        *cluster_contrasts(anchors),
        *orderbook_contrasts(anchors),
    ]
    nonempty = [item for item in outputs if not item.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else DataFrame()
    )
    if combined.empty:
        raise ValueError(f"No Generation 15 anchor-context contrasts for {task.pair}.")
    g0.atomic_write_parquet(combined, output_path)
    return {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "anchor_rows": len(anchors),
        "contrast_rows": len(combined),
        "routes": sorted(combined["route_id"].astype(str).unique()),
        "path": str(output_path),
        "sha256": g0.sha256_file(output_path),
    }


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {
            "status": "failed",
            "cohort": task.cohort,
            "pair": task.pair,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tasks(tasks: Sequence[PairTask], workers: int) -> list[dict[str, Any]]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(safe_build_pair, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g15_anchor_context_pair",
                        "processed": index,
                        "total": len(tasks),
                        "cohort": result.get("cohort"),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def summarize_contrasts(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    route_keys = ["route_id", "scope_value", "comparison"]
    for route_key, source in frame.groupby(route_keys, observed=True, sort=False):
        for horizon in HORIZONS:
            independent = g13d.purge_overlapping_hourly_pairs(
                source,
                separation_hours=horizon,
                group_columns=["pair", "window", "analysis_period"],
            )
            if independent.empty:
                continue
            for metric in METRICS:
                values = g15m.metric_values(independent, metric, horizon)
                scored = pd.concat(
                    [
                        independent[
                            [
                                "cohort",
                                "pair",
                                "window",
                                "analysis_period",
                                "actual_event_time",
                            ]
                        ].reset_index(drop=True),
                        values.reset_index(drop=True),
                    ],
                    axis=1,
                ).dropna(subset=["actual_value", "control_value", "difference"])
                if scored.empty:
                    continue
                for key, cell in scored.groupby(
                    ["cohort", "pair", "window", "analysis_period"],
                    observed=True,
                    sort=False,
                ):
                    pair_rows.append(
                        {
                            "route_id": route_key[0],
                            "scope_value": route_key[1],
                            "comparison": route_key[2],
                            "metric": metric,
                            "horizon_hours": horizon,
                            "cohort": key[0],
                            "pair": key[1],
                            "window": key[2],
                            "analysis_period": key[3],
                            "independent_rows": len(cell),
                            "treatment_mean": float(cell["actual_value"].mean()),
                            "control_mean": float(cell["control_value"].mean()),
                            "paired_mean_difference": float(cell["difference"].mean()),
                        }
                    )
                scored["week"] = pd.to_datetime(
                    scored["actual_event_time"], utc=True
                ).dt.floor("7D")
                weekly = (
                    scored.groupby(
                        ["cohort", "pair", "window", "analysis_period", "week"],
                        observed=True,
                        sort=False,
                    )["difference"]
                    .agg(["sum", "size"])
                    .reset_index()
                )
                for row in weekly.itertuples(index=False):
                    weekly_rows.append(
                        {
                            "route_id": route_key[0],
                            "scope_value": route_key[1],
                            "comparison": route_key[2],
                            "metric": metric,
                            "horizon_hours": horizon,
                            "cohort": row.cohort,
                            "pair": row.pair,
                            "window": row.window,
                            "analysis_period": row.analysis_period,
                            "week": row.week,
                            "difference_sum": float(row.sum),
                            "rows": int(row.size),
                        }
                    )
    return DataFrame.from_records(pair_rows), DataFrame.from_records(weekly_rows)


def equal_coin_scores(pair_summary: DataFrame, weekly_blocks: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "comparison",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
        "analysis_period",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in pair_summary.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["independent_rows"].ge(MIN_PAIR_ROWS)].copy()
        pairs = set(eligible["pair"].astype(str))
        blocks = weekly_blocks.loc[
            weekly_blocks["route_id"].eq(key[0])
            & weekly_blocks["scope_value"].eq(key[1])
            & weekly_blocks["comparison"].eq(key[2])
            & weekly_blocks["metric"].eq(key[3])
            & weekly_blocks["horizon_hours"].eq(key[4])
            & weekly_blocks["cohort"].eq(key[5])
            & weekly_blocks["window"].eq(key[6])
            & weekly_blocks["analysis_period"].eq(key[7])
            & weekly_blocks["pair"].isin(pairs)
        ]
        point, lower, upper = g15m.bootstrap_weekly_blocks(
            blocks, seed_key="|".join(map(str, key))
        )
        absolute = eligible["paired_mean_difference"].abs()
        absolute_sum = float(absolute.sum())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": len(eligible),
                "positive_coins": int(eligible["paired_mean_difference"].gt(0).sum()),
                "negative_coins": int(eligible["paired_mean_difference"].lt(0).sum()),
                "independent_rows": int(eligible["independent_rows"].sum()),
                "equal_coin_treatment_mean": float(eligible["treatment_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_control_mean": float(eligible["control_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_paired_difference": point,
                "bootstrap_lower_95": lower,
                "bootstrap_upper_95": upper,
                "maximum_absolute_coin_share": (
                    float(absolute.max() / absolute_sum) if absolute_sum > 0.0 else np.nan
                ),
                "not_dominated_by_one_coin": bool(
                    len(eligible)
                    and absolute_sum > 0.0
                    and float(absolute.max() / absolute_sum) <= MAX_COIN_ABSOLUTE_SHARE
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def expected_periods(route_id: str, cohort: str, window: str) -> tuple[str, ...]:
    if cohort == "normal" and window == "recent_normal_chronology":
        periods = ("recent_confirmation_early", "recent_confirmation_late")
        if route_id == "historical_btc_orderbook_conditioning":
            return periods[:1]
        return periods
    if cohort == "normal":
        return ("validation_early", "validation_late")
    return ("meme_validation_early", "meme_validation_late")


def comparison_contract(route_id: str, scope_value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if route_id == "contact_activity_combinations":
        if scope_value == "high_volume_and_range":
            return (
                (
                    "versus_high_volume_only",
                    "versus_high_range_only",
                    "versus_quiet_contact",
                ),
                (),
            )
        return (
            (
                "versus_high_volume_only",
                "versus_high_absolute_pressure_only",
                "versus_quiet_contact",
            ),
            (),
        )
    if route_id == "contact_lifecycle_and_approach":
        if scope_value == "fast_arrival_versus_slow_arrival":
            return (("current_fast_versus_slow",), ("shuffled_fast_versus_slow",))
        return (
            ("current_repeat_versus_fresh_30d",),
            ("shuffled_repeat_versus_fresh_30d",),
        )
    if route_id == "explicit_cluster_composition":
        return (("versus_matched_isolated_contact",), ())
    if route_id == "historical_btc_orderbook_conditioning":
        return (
            ("current_active_versus_quiet",),
            ("stale_active_versus_quiet", "shuffled_active_versus_quiet"),
        )
    raise KeyError(route_id)


def cell_decisions(scores: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        periods = expected_periods(key[0], key[4], key[5])
        primary, placebos = comparison_contract(key[0], key[1])
        required = {(period, comparison) for period in periods for comparison in primary}
        available = set(zip(cell["analysis_period"], cell["comparison"], strict=True))
        supported = cell.loc[
            cell["eligible_coins"].ge(MIN_COINS)
            & cell["analysis_period"].isin(periods)
        ].copy()
        if key[0] == "historical_btc_orderbook_conditioning":
            supported = supported.loc[supported["independent_rows"].ge(50)]
        supported_lookup = {
            (row.analysis_period, row.comparison): row
            for row in supported.itertuples(index=False)
        }
        primary_rows = [supported_lookup[item] for item in required if item in supported_lookup]
        effects = np.asarray(
            [float(row.equal_coin_paired_difference) for row in primary_rows], dtype=float
        )
        complete_primary = required.issubset(set(supported_lookup))
        positive = complete_primary and len(effects) == len(required) and np.all(effects > 0.0)
        negative = complete_primary and len(effects) == len(required) and np.all(effects < 0.0)
        sign = "positive" if positive else "negative" if negative else "mixed"
        placebo_required = {
            (period, comparison) for period in periods for comparison in placebos
        }
        complete_placebos = placebo_required.issubset(set(supported_lookup))
        placebo_beaten = True
        if placebos:
            placebo_beaten = complete_placebos
            for period in periods:
                primary_effect = float(
                    supported_lookup[(period, primary[0])].equal_coin_paired_difference
                ) if (period, primary[0]) in supported_lookup else np.nan
                for comparison in placebos:
                    placebo_effect = float(
                        supported_lookup[(period, comparison)].equal_coin_paired_difference
                    ) if (period, comparison) in supported_lookup else np.nan
                    placebo_beaten = bool(
                        placebo_beaten
                        and np.isfinite(primary_effect)
                        and np.isfinite(placebo_effect)
                        and abs(primary_effect) > abs(placebo_effect)
                    )
        point = bool((positive or negative) and placebo_beaten)
        strict_uncertainty = point and all(
            (
                float(row.bootstrap_lower_95) > 0.0
                if sign == "positive"
                else float(row.bootstrap_upper_95) < 0.0
            )
            for row in primary_rows
        )
        dominance = bool(
            primary_rows and all(bool(row.not_dominated_by_one_coin) for row in primary_rows)
        )
        attribution_eligible = not (
            (
                key[0] == "explicit_cluster_composition"
                and str(key[1]).endswith("__any_presence")
            )
            or str(key[1]).endswith("__all_contacts_mean_descriptive")
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "expected_periods": len(periods),
                "expected_primary_cells": len(required),
                "supported_primary_cells": len(primary_rows),
                "expected_placebo_cells": len(placebo_required),
                "available_cells_before_support": len(available),
                "consistent_effect_sign": sign,
                "placebo_magnitude_beaten_every_period": bool(placebo_beaten),
                "not_dominated_by_one_coin": dominance,
                "attribution_eligible": attribution_eligible,
                "source_limited": key[0] == "historical_btc_orderbook_conditioning",
                "point_pass": point,
                "strict_pass": bool(strict_uncertainty and dominance),
                "minimum_primary_effect": float(np.min(effects)) if len(effects) else np.nan,
                "maximum_primary_effect": float(np.max(effects)) if len(effects) else np.nan,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def joint_decisions(cells: DataFrame) -> DataFrame:
    keys = ["route_id", "scope_value", "metric", "horizon_hours"]
    expected = {
        ("normal", "standard_validation"),
        ("normal", "recent_normal_chronology"),
        ("meme", "standard_validation"),
    }
    rows: list[dict[str, Any]] = []
    for key, cell in cells.groupby(keys, observed=True, sort=False):
        selected = cell.loc[
            [
                (cohort, window) in expected
                for cohort, window in zip(cell["cohort"], cell["window"], strict=True)
            ]
        ]
        available = set(zip(selected["cohort"], selected["window"], strict=True))
        complete = expected.issubset(available)
        point_rows = selected.loc[selected["point_pass"]]
        signs = set(point_rows["consistent_effect_sign"])
        same_sign = len(signs) == 1 and len(point_rows) == len(expected)
        attribution = selected["attribution_eligible"].all() if len(selected) else False
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete_three_cell_ladder": complete,
                "same_sign_across_three_cells": same_sign,
                "attribution_eligible": bool(attribution),
                "source_limited": bool(selected["source_limited"].any())
                if len(selected)
                else False,
                "point_all_three_cells": bool(
                    complete and same_sign and selected["point_pass"].all()
                ),
                "strict_all_three_cells": bool(
                    complete and same_sign and selected["strict_pass"].all()
                ),
                "point_cells": int(selected["point_pass"].sum()),
                "strict_cells": int(selected["strict_pass"].sum()),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze, source_manifest, base_tasks = load_contracts()
    record_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id / "pair_contrasts"
    result_path = record_dir / "g15_anchor_contexts_result.json"
    run_record_path = record_dir / "g15_anchor_contexts_run.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    tasks = [
        PairTask(
            cohort=task.cohort,
            pair=task.pair,
            source_path=task.source_path,
            source_sha256=task.source_sha256,
            feature_path=task.feature_path,
            feature_sha256=task.feature_sha256,
            orderbook_path=task.orderbook_path,
            orderbook_sha256=task.orderbook_sha256,
            artifact_dir=str(artifact_dir),
            overwrite=args.overwrite,
        )
        for task in base_tasks
    ]
    run_record = {
        "schema_version": 1,
        "generation": 15,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "workers": args.workers,
        "worker_threads_each": 1,
        "pair_tasks": len(tasks),
        "artifact_dir": str(artifact_dir.resolve()),
        "expected_result": str(result_path.resolve()),
    }
    g0.atomic_write_json(run_record, run_record_path)
    try:
        inventory = run_tasks(tasks, args.workers)
        failures = [item for item in inventory if item["status"] == "failed"]
        if failures:
            raise RuntimeError(json.dumps(failures, indent=2, sort_keys=True))
        frames = [pd.read_parquet(item["path"]) for item in inventory]
        contrasts = pd.concat(frames, ignore_index=True, sort=False)
        pair_summary, weekly_blocks = summarize_contrasts(contrasts)
        scores = equal_coin_scores(pair_summary, weekly_blocks)
        decisions = cell_decisions(scores)
        joint = joint_decisions(decisions)

        inventory_path = record_dir / "pair_contrast_inventory.csv"
        pair_path = record_dir / "pair_context_differences.csv"
        scores_path = record_dir / "equal_coin_context_scores.csv"
        decisions_path = record_dir / "context_cell_decisions.csv"
        joint_path = record_dir / "context_joint_decisions.csv"
        g0.atomic_write_csv(DataFrame.from_records(inventory), inventory_path)
        g0.atomic_write_csv(pair_summary, pair_path)
        g0.atomic_write_csv(scores, scores_path)
        g0.atomic_write_csv(decisions, decisions_path)
        g0.atomic_write_csv(joint, joint_path)
        result = {
            "schema_version": 1,
            "generation": 15,
            "run_id": args.run_id,
            "status": "completed_generation15_anchor_contexts",
            "routes_completed": [
                "contact_activity_combinations",
                "contact_lifecycle_and_approach",
                "explicit_cluster_composition",
                "historical_btc_orderbook_conditioning",
            ],
            "source_contracts": {
                "generation15_freeze": artifact(g15z.FREEZE_PATH),
                "generation6_event_manifest": freeze["source_contracts"][
                    "generation6_event_manifest"
                ],
                "generation8_normal_features": artifact(G8_MANIFESTS["normal"]),
                "generation8_meme_features": artifact(G8_MANIFESTS["meme"]),
            },
            "implementation_contract": {
                "anchor_timeline": "frozen Generation 8 actual-contact timeline",
                "overlapping_paths_purged_separately_for_each_horizon": True,
                "contact_targets_begin_after_completed_contact_candle": True,
                "lifecycle_single_contact_only": True,
                "lifecycle_reason": (
                    "A single contacted lineage makes the 30d and repeat bins exact; "
                    "multi-level lifecycle averages are excluded as ambiguous."
                ),
                "lifecycle_all_contact_mean_lane": (
                    "reported separately as descriptive and never attribution-eligible"
                ),
                "arrival_pre_distance_not_matched_away": True,
                "cluster_any_presence_is_descriptive": True,
                "cluster_exclusive_is_attribution_eligible": True,
                "strict_maximum_absolute_one_coin_share": MAX_COIN_ABSOLUTE_SHARE,
                "orderbook_newest_later_period_available": False,
            },
            "runtime": {"workers": args.workers, "worker_threads_each": 1},
            "integrity": {
                "pair_tasks": len(tasks),
                "pair_failures": 0,
                "future_outcomes_used_for_matching": False,
                "future_signed_direction_used": False,
                "profit_used": False,
                "generation6_task_count": len(source_manifest["tasks"]),
            },
            "summary": {
                "contrast_rows_before_horizon_purges": len(contrasts),
                "pair_metric_rows": len(pair_summary),
                "score_rows": len(scores),
                "cell_decision_rows": len(decisions),
                "joint_decision_rows": len(joint),
                "point_all_three_rows": int(joint["point_all_three_cells"].sum()),
                "strict_all_three_rows": int(joint["strict_all_three_cells"].sum()),
                "attribution_eligible_point_all_three_rows": int(
                    (joint["point_all_three_cells"] & joint["attribution_eligible"]).sum()
                ),
                "joint_55_percent_direction_target_reached": False,
            },
            "artifacts": {
                "inventory": artifact(inventory_path),
                "pair_context_differences": artifact(pair_path),
                "equal_coin_context_scores": artifact(scores_path),
                "context_cell_decisions": artifact(decisions_path),
                "context_joint_decisions": artifact(joint_path),
            },
        }
        g0.atomic_write_json(result, result_path)
        run_record.update(
            {
                "status": "completed",
                "completed_at_utc": g0.utc_now(),
                "result_path": str(result_path.resolve()),
                "result_sha256": g0.sha256_file(result_path),
            }
        )
        g0.atomic_write_json(run_record, run_record_path)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    except Exception as exc:
        run_record.update(
            {
                "status": "failed",
                "failed_at_utc": g0.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        g0.atomic_write_json(run_record, run_record_path)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
