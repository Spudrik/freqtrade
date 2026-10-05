"""Run Generation 16's six frozen direction-neutral direct-attribution routes."""

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
):
    os.environ[_name] = "1"

import argparse
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
    market_reaction_zone_generation13_cache as g13c,
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
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freeze as g16z,
)


DEFAULT_RUN_ID = "g16_direct_attribution_20260822a"
RECORD_ROOT = g16z.FREEZE_PATH.parent / "g16_broad_attribution" / "direct"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation16"
    / "direct_attribution"
)
MAX_WORKERS = 4
HORIZONS = (1, 2, 4)
RAW_CONTROLS = (
    "actual",
    "matched_random_time",
    "near_miss",
    "stale_72h",
    "price_shift",
)
CONTACT_MATCH_FEATURES = (
    *g13d.MATCH_FEATURES,
    "contact_volume_ratio",
    "contact_range_ratio",
    "absolute_contact_pressure_change",
)
BASE_METRICS = (
    "volume_ratio",
    "range_ratio",
    "crossings",
    "unsigned_reaction",
)
TRANSITION_METRICS = (
    "volume_change_after_contact",
    "range_change_after_contact",
    "absolute_pressure_change_after_contact",
)
ALL_METRICS = (*BASE_METRICS, *TRANSITION_METRICS)
RAW_META_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "level_column",
    "representation",
    "control",
    "event_time",
    "period",
    "approach_state",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "zone_half_width_atr",
    "base_atr",
    "contact_range_ratio",
    "contact_volume_ratio",
    "contact_pressure_change",
)
RAW_OUTCOME_COLUMNS = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"abs_excursion_atr_h{horizon}",
        f"range_ratio_h{horizon}",
        f"volume_ratio_h{horizon}",
        f"pressure_change_h{horizon}",
        f"crossings_h{horizon}",
    )
)
RAW_READ_COLUMNS = tuple(
    dict.fromkeys((*RAW_META_COLUMNS, *g13d.MATCH_FEATURES, *RAW_OUTCOME_COLUMNS))
)
MIN_PAIR_ROWS = 20
MAX_COIN_ABSOLUTE_SHARE = 0.50
GROUP_MIN_COINS = {
    "all_normal": 5,
    "btc_separate": 1,
    "smart_contract_platforms": 5,
    "other_established_alts": 4,
    "top_ten_memes": 5,
}
LOCAL_ACTIVITY_COLUMNS = (
    "g11_local_activity_volatility__relative_volume",
    "g11_local_activity_volatility__volume_acceleration",
    "g11_local_activity_volatility__absolute_pressure",
    "g11_local_activity_volatility__pressure_persistence",
    "g11_local_activity_volatility__atr_fraction",
    "g11_local_activity_volatility__prior_range_atr",
    "g11_local_activity_volatility__bollinger_width",
    "g11_local_activity_volatility__range_contraction",
)
WIDER_ACTIVITY_COLUMNS = (
    "g11_wider_crypto_market__btc_return_1h",
    "g11_wider_crypto_market__btc_return_4h",
    "g11_wider_crypto_market__eth_return_1h",
    "g11_wider_crypto_market__eth_return_4h",
    "g11_wider_crypto_market__btc_relative_volume",
    "g11_wider_crypto_market__cohort_dispersion",
    "g11_wider_crypto_market__cohort_absolute_activity",
)
EIGHT_HOUR_ACTIVITY_COLUMNS = (
    "g13_mtf_8h_activity_volatility__relative_volume",
    "g13_mtf_8h_activity_volatility__volume_acceleration",
    "g13_mtf_8h_activity_volatility__absolute_pressure",
    "g13_mtf_8h_activity_volatility__pressure_persistence",
    "g13_mtf_8h_activity_volatility__atr_fraction",
    "g13_mtf_8h_activity_volatility__prior_range_atr",
    "g13_mtf_8h_activity_volatility__bollinger_width",
    "g13_mtf_8h_activity_volatility__range_contraction",
)
DENSITY_COLUMNS = (
    "g11_minimal_contact__contacted_level_count",
    "g11_minimal_contact__distinct_family_count",
    "g11_minimal_contact__distinct_timeframe_count",
    "g11_minimal_contact__distinct_mechanism_count",
)
ANCHOR_MATCH_FEATURES = (
    "contact_volume_ratio",
    "contact_range_ratio",
    "absolute_contact_pressure_change",
    "g11_minimal_contact__contacted_level_count",
    "g11_minimal_contact__anchor_zone_half_width_atr",
    "g11_minimal_contact__anchor_contact_distance_atr",
    "g11_local_trend_momentum__ema20_slope",
    "g11_local_trend_momentum__adx14",
)
DENSITY_MATCH_FEATURES = tuple(
    column
    for column in ANCHOR_MATCH_FEATURES
    if column != "g11_minimal_contact__contacted_level_count"
)


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    source_path: str
    source_sha256: str
    feature_path: str
    feature_sha256: str
    evaluation_path: str
    evaluation_sha256: str
    strict_families: tuple[str, ...]
    strict_timeframes: tuple[str, ...]
    output_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    frozen = json.loads(g16z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation16_outcomes":
        raise ValueError("Generation 16 plan is not frozen.")
    direct = frozen["main_direction_neutral_family"]
    if direct.get("future_signed_direction_used") or direct.get("profit_used"):
        raise ValueError("Generation 16 direct boundary changed.")
    source = frozen["source_contracts"]["generation6_event_manifest"]
    source_path = Path(source["path"])
    if not source_path.is_file() or g0.sha256_file(source_path) != source["sha256"]:
        raise ValueError("The frozen Generation 6 event manifest changed.")
    event_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    g13_manifests: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_generation13_cache":
            raise ValueError(f"Generation 13 {cohort} cache is not terminal.")
        manifest["_artifact"] = artifact(path)
        g13_manifests.append(manifest)
    return frozen, event_manifest, g13_manifests


def raw_frame(task: PairTask) -> DataFrame:
    path = Path(task.source_path)
    if not path.is_file() or g0.sha256_file(path) != task.source_sha256:
        raise ValueError(f"Frozen source changed: {path}")
    frame = pd.read_parquet(
        path,
        columns=list(RAW_READ_COLUMNS),
        filters=[("control", "in", list(RAW_CONTROLS))],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    frame["absolute_contact_pressure_change"] = pd.to_numeric(
        frame["contact_pressure_change"], errors="coerce"
    ).abs()
    actual_clock = set(
        frame.loc[frame["control"].eq("actual"), "event_time"].astype("int64")
    )
    frame = g13d.add_analysis_windows(frame, task.cohort)
    frame = frame.loc[
        frame["source_timeframe"].isin(task.strict_timeframes)
        & frame["level_family"].isin(task.strict_families)
    ].copy()
    frame = g15m.causal_deduplicate_by_family(frame)
    contaminated = (
        ~frame["control"].eq("actual")
        & frame["event_time"].astype("int64").isin(actual_clock)
    )
    frame = frame.loc[~contaminated].copy()
    frame.attrs["removed_control_rows_at_real_contacts"] = int(contaminated.sum())
    return frame


def metric_values(frame: DataFrame, prefix: str, metric: str, horizon: int) -> Series:
    def numeric(column: str) -> Series:
        return pd.to_numeric(frame[f"{prefix}__{column}"], errors="coerce")

    if metric == "volume_ratio":
        return numeric(f"volume_ratio_h{horizon}")
    if metric == "range_ratio":
        return numeric(f"range_ratio_h{horizon}")
    if metric == "crossings":
        return numeric(f"crossings_h{horizon}")
    if metric == "unsigned_reaction":
        threshold = np.maximum(0.5, numeric("zone_half_width_atr"))
        return (
            numeric(f"abs_excursion_atr_h{horizon}").ge(threshold)
            & numeric(f"volume_ratio_h{horizon}").ge(1.25)
        ).astype(float)
    if metric == "volume_change_after_contact":
        return numeric(f"volume_ratio_h{horizon}") - numeric("contact_volume_ratio")
    if metric == "range_change_after_contact":
        return numeric(f"range_ratio_h{horizon}") - numeric("contact_range_ratio")
    if metric == "absolute_pressure_change_after_contact":
        return numeric(f"pressure_change_h{horizon}").abs() - numeric(
            "contact_pressure_change"
        ).abs()
    raise KeyError(metric)


def matched_frame(
    left: DataFrame,
    right: DataFrame,
    *,
    exact_keys: Sequence[str],
    state_columns: Sequence[str],
    horizon: int,
    comparison: str,
) -> tuple[DataFrame, dict[str, int]]:
    rows: list[DataFrame] = []
    totals = {
        "eligible_actual": 0,
        "eligible_control": 0,
        "geometry_eligible_actual": 0,
        "state_matchable_actual": 0,
    }
    left_groups = left.groupby(list(exact_keys), observed=True, sort=False)
    right_groups = right.groupby(list(exact_keys), observed=True, sort=False)
    for key in sorted(set(left_groups.groups).intersection(right_groups.groups)):
        actual = left_groups.get_group(key).reset_index(drop=True)
        control = right_groups.get_group(key).reset_index(drop=True)
        pairs, audit = g1.nearest_state_pairs(
            actual,
            control,
            state_columns=state_columns,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=horizon,
        )
        for name in totals:
            totals[name] += int(audit[name])
        if not pairs:
            continue
        actual_selected = actual.iloc[[item[0] for item in pairs]].reset_index(drop=True)
        control_selected = control.iloc[[item[1] for item in pairs]].reset_index(drop=True)
        output = DataFrame(
            {
                "cohort": actual_selected["cohort"].astype(str),
                "pair": actual_selected["pair"].astype(str),
                "window": actual_selected["window"].astype(str),
                "analysis_period": actual_selected["analysis_period"].astype(str),
                "source_timeframe": actual_selected["source_timeframe"].astype(str),
                "level_family": actual_selected["level_family"].astype(str),
                "level_name": actual_selected["level_name"].astype(str),
                "approach_state": actual_selected["approach_state"].astype(str),
                "comparison": comparison,
                "actual_event_time": actual_selected["event_time"],
                "control_event_time": control_selected["event_time"],
                "match_distance": [item[2] for item in pairs],
                "pre_distance_atr_abs_difference": (
                    pd.to_numeric(actual_selected["pre_distance_atr"], errors="coerce")
                    - pd.to_numeric(control_selected["pre_distance_atr"], errors="coerce")
                ).abs(),
            }
        )
        carried = tuple(
            dict.fromkeys(
                (
                    "zone_half_width_atr",
                    "contact_volume_ratio",
                    "contact_range_ratio",
                    "contact_pressure_change",
                    *RAW_OUTCOME_COLUMNS,
                )
            )
        )
        for column in carried:
            output[f"actual__{column}"] = pd.to_numeric(
                actual_selected[column], errors="coerce"
            ).to_numpy(dtype=float)
            output[f"control__{column}"] = pd.to_numeric(
                control_selected[column], errors="coerce"
            ).to_numpy(dtype=float)
        rows.append(output)
    if not rows:
        return DataFrame(), totals
    combined = pd.concat(rows, ignore_index=True, sort=False)
    independent = g13d.purge_overlapping_hourly_pairs(
        combined,
        separation_hours=horizon,
        group_columns=["pair", "window", "analysis_period", "comparison"],
    )
    return independent, totals


def summarize_matches(
    frame: DataFrame,
    *,
    route_id: str,
    scope_value: str,
    horizon: int,
    metrics: Sequence[str],
    zero_baseline: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    if frame.empty:
        return pair_rows, weekly_rows
    for metric in metrics:
        actual = metric_values(frame, "actual", metric, horizon)
        control = (
            Series(0.0, index=frame.index)
            if zero_baseline
            else metric_values(frame, "control", metric, horizon)
        )
        scored = frame[
            [
                "cohort",
                "pair",
                "window",
                "analysis_period",
                "comparison",
                "actual_event_time",
            ]
        ].copy()
        if zero_baseline:
            scored["comparison"] = "actual_transition_vs_zero"
        scored["actual_value"] = actual
        scored["control_value"] = control
        scored["difference"] = actual - control
        scored = scored.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["actual_value", "control_value", "difference"]
        )
        for key, cell in scored.groupby(
            ["cohort", "pair", "window", "analysis_period", "comparison"],
            observed=True,
            sort=False,
        ):
            pair_rows.append(
                {
                    "route_id": route_id,
                    "scope_value": scope_value,
                    "metric": metric,
                    "horizon_hours": horizon,
                    "cohort": key[0],
                    "pair": key[1],
                    "window": key[2],
                    "analysis_period": key[3],
                    "comparison": key[4],
                    "independent_rows": len(cell),
                    "actual_mean": float(cell["actual_value"].mean()),
                    "control_mean": float(cell["control_value"].mean()),
                    "paired_mean_difference": float(cell["difference"].mean()),
                }
            )
        scored["week"] = pd.to_datetime(
            scored["actual_event_time"], utc=True
        ).dt.floor("7D")
        grouped = (
            scored.groupby(
                [
                    "cohort",
                    "pair",
                    "window",
                    "analysis_period",
                    "comparison",
                    "week",
                ],
                observed=True,
                sort=False,
            )["difference"]
            .agg(["sum", "size"])
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            weekly_rows.append(
                {
                    "route_id": route_id,
                    "scope_value": scope_value,
                    "metric": metric,
                    "horizon_hours": horizon,
                    "cohort": row.cohort,
                    "pair": row.pair,
                    "window": row.window,
                    "analysis_period": row.analysis_period,
                    "comparison": row.comparison,
                    "week": row.week,
                    "difference_sum": float(row.sum),
                    "rows": int(row.size),
                }
            )
    return pair_rows, weekly_rows


def rank_score(frame: DataFrame, columns: Sequence[str]) -> Series:
    ranked: list[Series] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if "return_" in column:
            values = values.abs()
        ranked.append(values.rank(pct=True, method="average"))
    return pd.concat(ranked, axis=1).mean(axis=1, skipna=False)


def anchor_frame(task: PairTask, raw: DataFrame) -> DataFrame:
    feature_path = Path(task.feature_path)
    evaluation_path = Path(task.evaluation_path)
    for path, expected in (
        (feature_path, task.feature_sha256),
        (evaluation_path, task.evaluation_sha256),
    ):
        if not path.is_file() or g0.sha256_file(path) != expected:
            raise ValueError(f"Frozen Generation 13 source changed: {path}")
    features = pd.read_parquet(feature_path)
    evaluation_columns = (
        "date",
        "anchor_level_family",
        "anchor_level_name",
        "anchor_source_timeframe",
        "anchor_representation",
    )
    evaluation = pd.read_parquet(evaluation_path, columns=list(evaluation_columns))
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="raise")
    evaluation["date"] = pd.to_datetime(
        evaluation["date"], utc=True, errors="raise"
    )
    anchors = evaluation.merge(features, on="date", how="inner", validate="one_to_one")
    anchors.rename(
        columns={
            "anchor_level_family": "level_family",
            "anchor_level_name": "level_name",
            "anchor_source_timeframe": "source_timeframe",
            "anchor_representation": "representation",
        },
        inplace=True,
    )
    anchors = anchors.loc[
        anchors["level_family"].isin(task.strict_families)
        & anchors["source_timeframe"].isin(task.strict_timeframes)
    ].copy()
    actual = raw.loc[raw["control"].eq("actual")].copy()
    exact = [
        "event_time",
        "level_family",
        "level_name",
        "source_timeframe",
        "representation",
    ]
    actual.sort_values(
        [*exact, "contact_close_distance_atr"], inplace=True, kind="stable"
    )
    actual = actual.drop_duplicates(exact, keep="first")
    merged = anchors.merge(
        actual,
        left_on=[
            "date",
            "level_family",
            "level_name",
            "source_timeframe",
            "representation",
        ],
        right_on=exact,
        how="inner",
        validate="one_to_many",
        suffixes=("", "__raw"),
    )
    if merged.empty:
        raise ValueError(f"No Generation 16 anchor rows for {task.pair}.")
    merged["absolute_contact_pressure_change"] = pd.to_numeric(
        merged["contact_pressure_change"], errors="coerce"
    ).abs()
    merged["local_activity_score"] = rank_score(merged, LOCAL_ACTIVITY_COLUMNS)
    merged["wider_activity_score"] = rank_score(merged, WIDER_ACTIVITY_COLUMNS)
    merged["eight_hour_activity_score"] = rank_score(
        merged, EIGHT_HOUR_ACTIVITY_COLUMNS
    )
    for name in ("local", "wider", "eight_hour"):
        score = f"{name}_activity_score"
        merged[f"{name}_activity_high"] = (
            merged.groupby(["window", "analysis_period"], observed=True)[score]
            .rank(pct=True, method="average")
            .ge(0.5)
        )
    for variant in ("", "_stale", "_shuffled"):
        column = f"g11_minimal_contact{variant}__contacted_level_count"
        merged[f"density{variant}_high"] = pd.to_numeric(
            merged[column], errors="coerce"
        ).ge(2.0)
    merged["pre_distance_atr"] = pd.to_numeric(
        merged["pre_distance_atr"], errors="coerce"
    )
    return merged


def condition_contrast(
    frame: DataFrame,
    *,
    left_mask: Series,
    right_mask: Series,
    comparison: str,
    horizon: int,
    state_columns: Sequence[str],
) -> tuple[DataFrame, dict[str, int]]:
    return matched_frame(
        frame.loc[left_mask].copy(),
        frame.loc[right_mask].copy(),
        exact_keys=("window", "analysis_period"),
        state_columns=state_columns,
        horizon=horizon,
        comparison=comparison,
    )


def context_definitions(frame: DataFrame) -> list[tuple[str, str, Series, Series]]:
    local = frame["local_activity_high"]
    wider = frame["wider_activity_high"]
    eight = frame["eight_hour_activity_high"]
    return [
        ("local_activity", "high_vs_low", local, ~local),
        ("wider_activity", "high_vs_low", wider, ~wider),
        ("completed_8h_activity", "high_vs_low", eight, ~eight),
        ("local_plus_wider", "vs_local_only", local & wider, local & ~wider),
        ("local_plus_wider", "vs_wider_only", local & wider, ~local & wider),
        ("local_plus_wider", "vs_neither", local & wider, ~local & ~wider),
        ("local_plus_8h", "vs_local_only", local & eight, local & ~eight),
        ("local_plus_8h", "vs_8h_only", local & eight, ~local & eight),
        ("local_plus_8h", "vs_neither", local & eight, ~local & ~eight),
        ("wider_plus_8h", "vs_wider_only", wider & eight, wider & ~eight),
        ("wider_plus_8h", "vs_8h_only", wider & eight, ~wider & eight),
        ("wider_plus_8h", "vs_neither", wider & eight, ~wider & ~eight),
        (
            "local_plus_wider_plus_8h",
            "vs_local_plus_wider",
            local & wider & eight,
            local & wider & ~eight,
        ),
        (
            "local_plus_wider_plus_8h",
            "vs_local_plus_8h",
            local & wider & eight,
            local & ~wider & eight,
        ),
        (
            "local_plus_wider_plus_8h",
            "vs_wider_plus_8h",
            local & wider & eight,
            ~local & wider & eight,
        ),
        (
            "local_plus_wider_plus_8h",
            "vs_neither",
            local & wider & eight,
            ~local & ~wider & ~eight,
        ),
    ]


def density_definitions(frame: DataFrame) -> list[tuple[str, str, Series, Series, tuple[str, ...]]]:
    current = frame["density_high"]
    stale = frame["density_stale_high"]
    shuffled = frame["density_shuffled_high"]
    geometry = "g11_cluster_geometry"
    same = pd.to_numeric(
        frame[f"{geometry}__same_mechanism_agreement"], errors="coerce"
    ).eq(1.0)
    different = pd.to_numeric(
        frame[f"{geometry}__different_mechanism_agreement"], errors="coerce"
    ).eq(1.0)
    cross = pd.to_numeric(
        frame[f"{geometry}__any_cross_timeframe_cluster"], errors="coerce"
    ).eq(1.0)
    opposing = pd.to_numeric(
        frame[f"{geometry}__opposing_side_overlap"], errors="coerce"
    ).eq(1.0)
    return [
        ("current_density", "dense_vs_isolated", current, ~current, DENSITY_MATCH_FEATURES),
        ("stale_density_placebo", "dense_vs_isolated", stale, ~stale, DENSITY_MATCH_FEATURES),
        (
            "shuffled_density_placebo",
            "dense_vs_isolated",
            shuffled,
            ~shuffled,
            DENSITY_MATCH_FEATURES,
        ),
        (
            "same_vs_different_mechanism",
            "same_only_vs_different_only",
            same & ~different,
            different & ~same,
            ANCHOR_MATCH_FEATURES,
        ),
        (
            "cross_timeframe_overlap",
            "cross_vs_dense_no_cross",
            current & cross,
            current & ~cross,
            ANCHOR_MATCH_FEATURES,
        ),
        (
            "opposing_side_overlap",
            "opposing_vs_dense_no_opposing",
            current & opposing,
            current & ~opposing,
            ANCHOR_MATCH_FEATURES,
        ),
    ]


def append_summaries(
    target_pair_rows: list[dict[str, Any]],
    target_weekly_rows: list[dict[str, Any]],
    matched: DataFrame,
    *,
    route_id: str,
    scope_value: str,
    horizon: int,
    metrics: Sequence[str],
    zero_baseline: bool = False,
) -> None:
    pair_rows, weekly_rows = summarize_matches(
        matched,
        route_id=route_id,
        scope_value=scope_value,
        horizon=horizon,
        metrics=metrics,
        zero_baseline=zero_baseline,
    )
    target_pair_rows.extend(pair_rows)
    target_weekly_rows.extend(weekly_rows)


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901
    output_dir = Path(task.output_dir)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    pair_path = output_dir / "pair_summaries" / f"{stem}.parquet"
    weekly_path = output_dir / "weekly_blocks" / f"{stem}.parquet"
    audit_path = output_dir / "match_audits" / f"{stem}.parquet"
    if (
        pair_path.is_file()
        and weekly_path.is_file()
        and audit_path.is_file()
        and not task.overwrite
    ):
        return {
            "status": "existing",
            "cohort": task.cohort,
            "pair": task.pair,
            "pair_summary_path": str(pair_path),
            "pair_summary_sha256": g0.sha256_file(pair_path),
            "weekly_path": str(weekly_path),
            "weekly_sha256": g0.sha256_file(weekly_path),
            "audit_path": str(audit_path),
            "audit_sha256": g0.sha256_file(audit_path),
        }
    raw = raw_frame(task)
    anchors = anchor_frame(task, raw)
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    actual = raw.loc[raw["control"].eq("actual")].copy()
    control_route_map = {
        "matched_random_time": (
            "completed_contact_activity_control",
            "matched_ordinary_time",
        ),
        "near_miss": ("completed_contact_activity_control", "genuine_near_miss"),
        "stale_72h": ("stale_shift_and_density_controls", "causal_72h_old_level"),
        "price_shift": ("stale_shift_and_density_controls", "price_shifted_level"),
    }
    cached_matches: dict[tuple[str, int], DataFrame] = {}
    for control_name, (route_id, comparison) in control_route_map.items():
        for horizon in HORIZONS:
            matched, audit = matched_frame(
                actual,
                raw.loc[raw["control"].eq(control_name)].copy(),
                exact_keys=(
                    "window",
                    "analysis_period",
                    "source_timeframe",
                    "level_family",
                    "approach_state",
                ),
                state_columns=CONTACT_MATCH_FEATURES,
                horizon=horizon,
                comparison=comparison,
            )
            cached_matches[(control_name, horizon)] = matched
            audits.append(
                {
                    "route_id": route_id,
                    "scope_value": "all_retained_levels",
                    "comparison": comparison,
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
            routes = [(route_id, comparison)]
            if control_name == "matched_random_time":
                routes.append(
                    ("stale_shift_and_density_controls", "matched_ordinary_time")
                )
            for selected_route, _ in routes:
                append_summaries(
                    pair_rows,
                    weekly_rows,
                    matched,
                    route_id=selected_route,
                    scope_value="all_retained_levels",
                    horizon=horizon,
                    metrics=BASE_METRICS,
                )
            if control_name in {"matched_random_time", "near_miss"}:
                append_summaries(
                    pair_rows,
                    weekly_rows,
                    matched,
                    route_id="pre_contact_to_post_contact_change",
                    scope_value="all_retained_levels",
                    horizon=horizon,
                    metrics=TRANSITION_METRICS,
                )
    for horizon in HORIZONS:
        baseline = cached_matches[("matched_random_time", horizon)].drop_duplicates(
            ["pair", "window", "analysis_period", "actual_event_time"]
        )
        append_summaries(
            pair_rows,
            weekly_rows,
            baseline,
            route_id="pre_contact_to_post_contact_change",
            scope_value="all_retained_levels",
            horizon=horizon,
            metrics=TRANSITION_METRICS,
            zero_baseline=True,
        )

    for horizon in HORIZONS:
        for family in task.strict_families:
            matched, audit = matched_frame(
                anchors.loc[anchors["level_family"].eq(family)].copy(),
                anchors.loc[~anchors["level_family"].eq(family)].copy(),
                exact_keys=(
                    "window",
                    "analysis_period",
                    "source_timeframe",
                    "approach_state",
                ),
                state_columns=ANCHOR_MATCH_FEATURES,
                horizon=horizon,
                comparison="target_family_vs_other_retained_families",
            )
            scope = f"family::{family}"
            audits.append(
                {
                    "route_id": "head_to_head_family_and_timeframe",
                    "scope_value": scope,
                    "comparison": "target_family_vs_other_retained_families",
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
            append_summaries(
                pair_rows,
                weekly_rows,
                matched,
                route_id="head_to_head_family_and_timeframe",
                scope_value=scope,
                horizon=horizon,
                metrics=BASE_METRICS,
            )
        for timeframe in task.strict_timeframes:
            matched, audit = matched_frame(
                anchors.loc[anchors["source_timeframe"].eq(timeframe)].copy(),
                anchors.loc[~anchors["source_timeframe"].eq(timeframe)].copy(),
                exact_keys=(
                    "window",
                    "analysis_period",
                    "level_family",
                    "approach_state",
                ),
                state_columns=ANCHOR_MATCH_FEATURES,
                horizon=horizon,
                comparison="target_timeframe_vs_other_retained_timeframes",
            )
            scope = f"timeframe::{timeframe}"
            audits.append(
                {
                    "route_id": "head_to_head_family_and_timeframe",
                    "scope_value": scope,
                    "comparison": "target_timeframe_vs_other_retained_timeframes",
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
            append_summaries(
                pair_rows,
                weekly_rows,
                matched,
                route_id="head_to_head_family_and_timeframe",
                scope_value=scope,
                horizon=horizon,
                metrics=BASE_METRICS,
            )

        for scope, comparison, left_mask, right_mask, state_columns in density_definitions(
            anchors
        ):
            matched, audit = condition_contrast(
                anchors,
                left_mask=left_mask,
                right_mask=right_mask,
                comparison=comparison,
                horizon=horizon,
                state_columns=state_columns,
            )
            audits.append(
                {
                    "route_id": "level_density_and_overlap",
                    "scope_value": scope,
                    "comparison": comparison,
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
            append_summaries(
                pair_rows,
                weekly_rows,
                matched,
                route_id="level_density_and_overlap",
                scope_value=scope,
                horizon=horizon,
                metrics=BASE_METRICS,
            )

        for scope, comparison, left_mask, right_mask in context_definitions(anchors):
            matched, audit = condition_contrast(
                anchors,
                left_mask=left_mask,
                right_mask=right_mask,
                comparison=comparison,
                horizon=horizon,
                state_columns=ANCHOR_MATCH_FEATURES,
            )
            audits.append(
                {
                    "route_id": "local_wider_and_completed_8h_activity_interaction",
                    "scope_value": scope,
                    "comparison": comparison,
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
            append_summaries(
                pair_rows,
                weekly_rows,
                matched,
                route_id="local_wider_and_completed_8h_activity_interaction",
                scope_value=scope,
                horizon=horizon,
                metrics=BASE_METRICS,
            )
    pair_frame = DataFrame.from_records(pair_rows)
    weekly_frame = DataFrame.from_records(weekly_rows)
    audit_frame = DataFrame.from_records(audits)
    if pair_frame.empty or weekly_frame.empty:
        raise ValueError(f"No Generation 16 direct summaries for {task.pair}.")
    g0.atomic_write_parquet(pair_frame, pair_path)
    g0.atomic_write_parquet(weekly_frame, weekly_path)
    g0.atomic_write_parquet(audit_frame, audit_path)
    return {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "removed_control_rows_at_real_contacts": raw.attrs[
            "removed_control_rows_at_real_contacts"
        ],
        "anchor_rows": len(anchors),
        "pair_summary_rows": len(pair_frame),
        "weekly_rows": len(weekly_frame),
        "audit_rows": len(audit_frame),
        "pair_summary_path": str(pair_path),
        "pair_summary_sha256": g0.sha256_file(pair_path),
        "weekly_path": str(weekly_path),
        "weekly_sha256": g0.sha256_file(weekly_path),
        "audit_path": str(audit_path),
        "audit_sha256": g0.sha256_file(audit_path),
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
                        "phase": "g16_direct_pair",
                        "processed": index,
                        "total": len(tasks),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def group_for_pair(cohort: str, pair: str) -> tuple[str, ...]:
    if cohort == "meme":
        return ("top_ten_memes",)
    groups = ["all_normal"]
    for name, members in g15z.NORMAL_GROUPS.items():
        if pair in members:
            groups.append(name)
    return tuple(groups)


def expand_market_scopes(frame: DataFrame) -> DataFrame:
    rows: list[DataFrame] = []
    for _, item in frame.groupby(["cohort", "pair"], observed=True, sort=False):
        cohort = str(item.iloc[0]["cohort"])
        pair = str(item.iloc[0]["pair"])
        for scope in group_for_pair(cohort, pair):
            copy = item.copy()
            copy["market_scope"] = scope
            rows.append(copy)
    return pd.concat(rows, ignore_index=True, sort=False)


def equal_coin_scores(pair_summary: DataFrame, weekly: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "market_scope",
        "window",
        "analysis_period",
        "comparison",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in pair_summary.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["independent_rows"].ge(MIN_PAIR_ROWS)].copy()
        pair_names = set(eligible["pair"].astype(str))
        blocks = weekly.loc[
            weekly["route_id"].eq(key[0])
            & weekly["scope_value"].eq(key[1])
            & weekly["metric"].eq(key[2])
            & weekly["horizon_hours"].eq(key[3])
            & weekly["market_scope"].eq(key[4])
            & weekly["window"].eq(key[5])
            & weekly["analysis_period"].eq(key[6])
            & weekly["comparison"].eq(key[7])
            & weekly["pair"].isin(pair_names)
        ]
        point, lower, upper = g15m.bootstrap_weekly_blocks(
            blocks, seed_key="|".join(map(str, key))
        )
        absolute = eligible["paired_mean_difference"].abs()
        dominance = (
            float(absolute.max() / absolute.sum())
            if len(absolute) and absolute.sum() > 0.0
            else np.nan
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": len(eligible),
                "required_coins": GROUP_MIN_COINS[str(key[4])],
                "independent_rows": int(eligible["independent_rows"].sum()),
                "positive_coins": int(eligible["paired_mean_difference"].gt(0).sum()),
                "negative_coins": int(eligible["paired_mean_difference"].lt(0).sum()),
                "equal_coin_actual_mean": float(eligible["actual_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_control_mean": float(eligible["control_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_paired_difference": point,
                "bootstrap_lower_95": lower,
                "bootstrap_upper_95": upper,
                "maximum_one_coin_absolute_share": dominance,
                "coin_support_pass": len(eligible) >= GROUP_MIN_COINS[str(key[4])],
                "uncertainty_positive": bool(np.isfinite(lower) and lower > 0.0),
                "uncertainty_negative": bool(np.isfinite(upper) and upper < 0.0),
                "dominance_pass": bool(
                    str(key[4]) == "btc_separate"
                    or (np.isfinite(dominance) and dominance <= MAX_COIN_ABSOLUTE_SHARE)
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def score_leads(scores: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "market_scope",
        "window",
        "comparison",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        supported = cell.loc[cell["coin_support_pass"]].copy()
        expected_periods = 2
        positive = bool(
            len(supported) == expected_periods
            and supported["equal_coin_paired_difference"].gt(0.0).all()
        )
        negative = bool(
            len(supported) == expected_periods
            and supported["equal_coin_paired_difference"].lt(0.0).all()
        )
        strict = bool(
            (
                positive
                and supported["uncertainty_positive"].all()
                and supported["dominance_pass"].all()
            )
            or (
                negative
                and supported["uncertainty_negative"].all()
                and supported["dominance_pass"].all()
            )
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "supported_periods": len(supported),
                "expected_periods": expected_periods,
                "effect_sign": "positive" if positive else "negative" if negative else "mixed",
                "point_pass": positive or negative,
                "strict_pass": strict,
                "minimum_effect": float(supported["equal_coin_paired_difference"].min())
                if len(supported)
                else np.nan,
                "maximum_effect": float(supported["equal_coin_paired_difference"].max())
                if len(supported)
                else np.nan,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def selected_pairs(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen, event_manifest, g13_manifests = load_contracts()
    run_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    result_path = run_dir / "g16_direct_attribution_result.json"
    run_record_path = run_dir / "g16_direct_attribution_run.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "objective": "Complete the first six frozen Generation 16 direct routes.",
        "routes": [
            item["route_id"]
            for item in frozen["main_direction_neutral_family"]["routes"]
            if not item["route_id"].startswith("freqai")
        ],
        "horizons_hours": list(HORIZONS),
        "metrics": list(ALL_METRICS),
        "controls": list(RAW_CONTROLS),
        "match_features": list(CONTACT_MATCH_FEATURES),
        "future_outcomes_used_for_matching": False,
        "future_signed_direction_used": False,
        "profit_used": False,
        "workers": int(args.workers),
        "source_contracts": {
            "generation16_freeze": artifact(g16z.FREEZE_PATH),
            "generation6_event_manifest": frozen["source_contracts"][
                "generation6_event_manifest"
            ],
            "generation13_manifests": [item["_artifact"] for item in g13_manifests],
        },
    }
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "running",
            "started_at_utc": g0.utc_now(),
            "request": request,
        },
        run_record_path,
    )
    allowed = selected_pairs(str(args.pairs))
    strict_families = tuple(frozen["parent_surface"]["strict_level_families"])
    strict_timeframes = tuple(frozen["parent_surface"]["strict_source_timeframes"])
    g13_by_cohort = {str(item["cohort"]): item for item in g13_manifests}
    tasks: list[PairTask] = []
    for item in event_manifest["tasks"]:
        pair = str(item["pair"])
        cohort = str(item["cohort"])
        if allowed is not None and pair not in allowed:
            continue
        g13_item = next(
            row for row in g13_by_cohort[cohort]["inventory"] if row["pair"] == pair
        )
        tasks.append(
            PairTask(
                cohort=cohort,
                pair=pair,
                source_path=str(item["event_path"]),
                source_sha256=str(item["event_sha256"]),
                feature_path=str(g13_item["mtf_feature_path"]),
                feature_sha256=str(g13_item["mtf_feature_sha256"]),
                evaluation_path=str(g13_item["mtf_evaluation_path"]),
                evaluation_sha256=str(g13_item["mtf_evaluation_sha256"]),
                strict_families=strict_families,
                strict_timeframes=strict_timeframes,
                output_dir=str(artifact_dir),
                overwrite=bool(args.overwrite),
            )
        )
    inventory = run_tasks(tasks, int(args.workers))
    failed = [item for item in inventory if item["status"] == "failed"]
    if failed:
        g0.atomic_write_json(
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "status": "failed",
                "failed_at_utc": g0.utc_now(),
                "request": request,
                "failures": failed,
            },
            run_record_path,
        )
        raise RuntimeError(f"Generation 16 direct pair failures: {failed}")
    pair_parts = [pd.read_parquet(item["pair_summary_path"]) for item in inventory]
    weekly_parts = [pd.read_parquet(item["weekly_path"]) for item in inventory]
    audit_parts = [pd.read_parquet(item["audit_path"]) for item in inventory]
    pair_summary = expand_market_scopes(pd.concat(pair_parts, ignore_index=True))
    weekly = expand_market_scopes(pd.concat(weekly_parts, ignore_index=True))
    audits = pd.concat(audit_parts, ignore_index=True)
    scores = equal_coin_scores(pair_summary, weekly)
    leads = score_leads(scores)
    outputs = {
        "inventory": run_dir / "pair_inventory.csv",
        "pair_summary": run_dir / "pair_contrast_summary.csv",
        "weekly_blocks": artifact_dir / "weekly_contrast_blocks.parquet",
        "match_audits": run_dir / "match_audits.csv",
        "equal_coin_scores": run_dir / "equal_coin_scores.csv",
        "period_leads": run_dir / "period_leads.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(inventory), outputs["inventory"])
    g0.atomic_write_csv(pair_summary, outputs["pair_summary"])
    g0.atomic_write_parquet(weekly, outputs["weekly_blocks"])
    g0.atomic_write_csv(audits, outputs["match_audits"])
    g0.atomic_write_csv(scores, outputs["equal_coin_scores"])
    g0.atomic_write_csv(leads, outputs["period_leads"])
    summary = {
        "pair_tasks": len(tasks),
        "pair_failures": 0,
        "pair_summary_rows": len(pair_summary),
        "score_rows": len(scores),
        "period_lead_rows": len(leads),
        "point_period_leads": int(leads["point_pass"].sum()),
        "strict_period_leads": int(leads["strict_pass"].sum()),
        "routes_completed": sorted(pair_summary["route_id"].unique()),
        "future_signed_direction_used": False,
        "profit_used": False,
    }
    result = {
        "schema_version": 1,
        "generation": 16,
        "run_id": args.run_id,
        "status": "completed_generation16_direct_attribution",
        "completed_at_utc": g0.utc_now(),
        "request": request,
        "summary": summary,
        "artifacts": {name: artifact(path) for name, path in outputs.items()},
    }
    g0.atomic_write_json(result, result_path)
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": result["status"],
            "completed_at_utc": result["completed_at_utc"],
            "request": request,
            "result": artifact(result_path),
        },
        run_record_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
