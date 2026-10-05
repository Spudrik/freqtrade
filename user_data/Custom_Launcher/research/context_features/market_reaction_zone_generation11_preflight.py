from __future__ import annotations

# This preflight selects feature/meta columns only. Reaction outcomes are forbidden.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


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
    market_reaction_zone_generation11_freeze as g11z,
)


PREFLIGHT_ROOT = g11z.OUTPUT_ROOT / "generation11_preflight"
DEFAULT_PREFLIGHT_ID = "g11_broad_support_20260822a"
MIN_EVENTS = 50
MIN_COINS = 5

BASE_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "control",
    "event_time",
    "period",
    "approach_code",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "level_score",
    "zone_half_width_atr",
    "market_group",
    "smart_contract_platform",
)
STATE_COLUMNS = (
    "state__relative_volume",
    "state__volume_acceleration",
    "state__absolute_pressure",
    "state__pressure_persistence",
    "state__atr_fraction",
    "state__prior_range_atr",
    "state__bollinger_width",
    "state__range_contraction",
    "state__ema20_slope",
    "state__ma_separation",
    "state__return_slope",
    "state__return_acceleration",
    "state__adx14",
    "state__rsi14",
    "state__rsi_change",
    "state__macd_histogram",
    "state__macd_histogram_change",
)
CROSS_MARKET_COLUMNS = (
    "xm_btc_return_1h",
    "xm_btc_return_4h",
    "xm_btc_return_24h",
    "xm_btc_relative_volume",
    "xm_eth_return_1h",
    "xm_eth_return_4h",
    "xm_eth_return_24h",
    "xm_cohort_breadth_positive",
    "xm_cohort_dispersion",
    "xm_cohort_absolute_activity",
    "xm_cohort_ready_members",
    "xm_pair_btc_corr_168h",
    "xm_pair_btc_beta_168h",
)
EXTERNAL_COLUMNS = (
    "news_gdelt_ready",
    "news_gdelt_regime",
    "news_gdelt_activity",
    "news_topics_ready",
    "ob_btc_source_ready",
    "ob_btc_model_ready",
    "ob_btc_regime",
    "ob_btc_coverage_ratio",
    "ob_btc_pressure_25bps",
    "ob_btc_absolute_pressure",
    "ob_pair_local",
)
RELATIONSHIP_COLUMNS = tuple(
    f"rel__{timeframe}__{relationship}"
    for timeframe in ("4h", "8h", "1d")
    for relationship in (
        "isolated_native_1h",
        "isolated_higher_timeframe",
        "same_mechanism_agreement",
        "different_mechanism_agreement",
        "any_cross_timeframe_cluster",
        "opposing_side_overlap",
    )
)
INPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            *BASE_COLUMNS,
            *STATE_COLUMNS,
            *CROSS_MARKET_COLUMNS,
            *EXTERNAL_COLUMNS,
            *RELATIONSHIP_COLUMNS,
        )
    )
)
THRESHOLD_FEATURES = (
    "state__relative_volume",
    "state__volume_acceleration",
    "state__absolute_pressure",
    "state__pressure_persistence",
    "state__atr_fraction",
    "state__bollinger_width",
    "state__range_contraction",
    "state__adx14",
    "xm_cohort_dispersion",
    "xm_cohort_absolute_activity",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def development_period(cohort: str) -> str:
    return "development" if cohort == "normal" else "meme_development"


def validate_input_columns(columns: Sequence[str] = INPUT_COLUMNS) -> None:
    forbidden_fragments = (
        "excursion_atr_h",
        "volume_ratio_h",
        "range_ratio_h",
        "pressure_change_h",
        "dwell_fraction_h",
        "crossings_h",
        "time_to_away",
        "time_to_through",
        "time_to_abs",
    )
    forbidden = [
        column
        for column in columns
        if any(fragment in column for fragment in forbidden_fragments)
    ]
    if forbidden:
        raise ValueError(f"Outcome columns requested by preflight: {forbidden}")


def load_feature_rows(event_manifest: dict[str, Any]) -> DataFrame:
    validate_input_columns()
    frames: list[DataFrame] = []
    for task in event_manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Generation 11 event source changed: {path}")
        frame = pd.read_parquet(
            path,
            columns=list(INPUT_COLUMNS),
            filters=[("control", "in", ["actual", *g11z.LEVEL_CONTROLS])],
        )
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True, sort=False)
    output["event_time"] = pd.to_datetime(output["event_time"], utc=True, errors="coerce")
    if output["event_time"].isna().any():
        raise ValueError("Generation 11 feature rows contain invalid event timestamps.")
    return output


def deterministic_deduplicate(frame: DataFrame) -> DataFrame:
    keys = (
        "cohort",
        "pair",
        "event_time",
        "source_timeframe",
        "level_family",
        "control",
    )
    return g6d.causal_deduplicate(frame, keys=keys)


def build_thresholds(frame: DataFrame) -> DataFrame:
    actual = frame.loc[frame["control"].eq("actual")].copy()
    actual = g6d.causal_deduplicate(
        actual,
        keys=("cohort", "pair", "event_time", "control"),
    )
    rows: list[dict[str, Any]] = []
    for cohort in g11z.COHORTS:
        development = actual.loc[
            actual["cohort"].eq(cohort)
            & actual["period"].eq(development_period(cohort))
        ]
        for feature in THRESHOLD_FEATURES:
            values = pd.to_numeric(development[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            lower = float(values.quantile(1.0 / 3.0)) if len(values) else np.nan
            upper = float(values.quantile(2.0 / 3.0)) if len(values) else np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "development_period": development_period(cohort),
                    "development_rows": len(values),
                    "lower_tercile": lower,
                    "upper_tercile": upper,
                    "status": (
                        "frozen_from_development_predictors"
                        if len(values) >= MIN_EVENTS
                        and np.isfinite(lower)
                        and np.isfinite(upper)
                        and upper > lower
                        else "unsupported_degenerate_or_sparse"
                    ),
                }
            )
    return DataFrame.from_records(rows).sort_values(["cohort", "feature"])


def threshold_map(thresholds: DataFrame, feature: str) -> dict[str, tuple[float, float]]:
    selected = thresholds.loc[
        thresholds["feature"].eq(feature)
        & thresholds["status"].eq("frozen_from_development_predictors")
    ]
    return {
        str(row.cohort): (float(row.lower_tercile), float(row.upper_tercile))
        for row in selected.itertuples(index=False)
    }


def assign_activity_state(frame: DataFrame, thresholds: DataFrame) -> pd.Series:
    mapping = threshold_map(thresholds, "state__relative_volume")
    output = pd.Series("unavailable", index=frame.index, dtype="string")
    values = pd.to_numeric(frame["state__relative_volume"], errors="coerce")
    for cohort, (lower, upper) in mapping.items():
        scope = frame["cohort"].eq(cohort) & values.notna()
        output.loc[scope & values.le(lower)] = "quiet"
        output.loc[scope & values.gt(lower) & values.lt(upper)] = "typical"
        output.loc[scope & values.ge(upper)] = "active"
    return output


def support_summary(frame: DataFrame, thresholds: DataFrame) -> dict[str, DataFrame]:
    output = frame.copy()
    output["activity_state"] = assign_activity_state(output, thresholds)
    output["has_clear_approach"] = pd.to_numeric(
        output["approach_code"], errors="coerce"
    ).ne(0)
    relationship = output[list(RELATIONSHIP_COLUMNS)].fillna(False).astype(bool)
    output["has_cross_timeframe_cluster"] = relationship[
        [column for column in RELATIONSHIP_COLUMNS if column.endswith("cluster")]
    ].any(axis=1)
    output["has_different_mechanism_agreement"] = relationship[
        [
            column
            for column in RELATIONSHIP_COLUMNS
            if column.endswith("different_mechanism_agreement")
        ]
    ].any(axis=1)
    output["has_opposing_overlap"] = relationship[
        [column for column in RELATIONSHIP_COLUMNS if column.endswith("opposing_side_overlap")]
    ].any(axis=1)

    base = (
        output.groupby(
            ["cohort", "period", "source_timeframe", "control"],
            observed=True,
            dropna=False,
        )
        .agg(events=("event_time", "size"), coins=("pair", "nunique"))
        .reset_index()
    )
    actual = output.loc[output["control"].eq("actual")]
    activity = (
        actual.groupby(
            ["cohort", "period", "source_timeframe", "activity_state"],
            observed=True,
            dropna=False,
        )
        .agg(events=("event_time", "size"), coins=("pair", "nunique"))
        .reset_index()
    )
    direction = (
        actual.loc[actual["has_clear_approach"]]
        .groupby(["cohort", "period", "source_timeframe"], observed=True)
        .agg(events=("event_time", "size"), coins=("pair", "nunique"))
        .reset_index()
    )
    geometry = (
        actual.groupby(
            [
                "cohort",
                "period",
                "source_timeframe",
                "has_cross_timeframe_cluster",
                "has_different_mechanism_agreement",
                "has_opposing_overlap",
            ],
            observed=True,
            dropna=False,
        )
        .agg(events=("event_time", "size"), coins=("pair", "nunique"))
        .reset_index()
    )
    external_frames: list[DataFrame] = []
    for source, ready_column, regime_column in (
        ("gdelt", "news_gdelt_ready", "news_gdelt_regime"),
        ("btc_orderbook", "ob_btc_model_ready", "ob_btc_regime"),
    ):
        ready = actual[ready_column].fillna(False).astype(bool)
        selected = actual.loc[ready].copy()
        selected["external_source"] = source
        selected["external_regime"] = selected[regime_column].astype("string")
        external_frames.append(
            selected.groupby(
                ["external_source", "cohort", "period", "external_regime"],
                observed=True,
                dropna=False,
            )
            .agg(events=("event_time", "size"), coins=("pair", "nunique"))
            .reset_index()
        )
    external = pd.concat(external_frames, ignore_index=True)
    groups = (
        actual.groupby(
            ["cohort", "period", "market_group", "smart_contract_platform"],
            observed=True,
            dropna=False,
        )
        .agg(events=("event_time", "size"), coins=("pair", "nunique"))
        .reset_index()
    )
    return {
        "base_support": base,
        "activity_support": activity,
        "direction_support": direction,
        "geometry_support": geometry,
        "external_support": external,
        "market_group_support": groups,
    }


def route_decisions(support: dict[str, DataFrame]) -> DataFrame:
    validation_names = {
        period for periods in g11z.VALIDATION_PERIODS.values() for period in periods
    }
    rows: list[dict[str, Any]] = []
    for route in g11z.ROUTES:
        if route.route_id == "external_calm_and_stress":
            frame = support["external_support"]
            checks = frame.loc[frame["period"].isin(validation_names)]
            supported = bool(
                checks["events"].ge(MIN_EVENTS).any()
                and checks["coins"].ge(MIN_COINS).any()
            )
            note = "source-specific cells below support remain parked"
        elif route.route_id == "cluster_obstacle_geometry":
            frame = support["geometry_support"]
            checks = frame.loc[frame["period"].isin(validation_names)]
            supported = bool(
                checks["events"].ge(MIN_EVENTS).any()
                and checks["coins"].ge(MIN_COINS).any()
            )
            note = "each geometry state is scored only where supported"
        elif route.route_id == "market_group_portability":
            frame = support["market_group_support"]
            checks = frame.loc[frame["period"].isin(validation_names)]
            supported = bool(checks["coins"].ge(3).any())
            note = "single-coin rows remain descriptive audits"
        else:
            frame = support[
                "direction_support"
                if route.route_id == "crypto_market_alignment"
                else "activity_support"
                if route.route_id == "activity_displacement"
                else "base_support"
            ]
            checks = frame.loc[frame["period"].isin(validation_names)]
            supported = bool(
                checks["events"].ge(MIN_EVENTS).any()
                and checks["coins"].ge(MIN_COINS).any()
            )
            note = "complete cohort/timeframe cells are retained; sparse cells park"
        rows.append(
            {
                "route_id": route.route_id,
                "family": route.family,
                "supported_for_outcome_screen": supported,
                "status": (
                    "supported_with_cell_level_parking"
                    if supported
                    else "parked_without_opening_outcomes"
                ),
                "note": note,
            }
        )
    return DataFrame.from_records(rows)


def run_preflight(preflight_id: str) -> dict[str, Any]:
    g11z.validate_existing_freeze()
    event_manifest = json.loads(g11z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    rows = deterministic_deduplicate(load_feature_rows(event_manifest))
    thresholds = build_thresholds(rows)
    support = support_summary(rows, thresholds)
    decisions = route_decisions(support)
    if not decisions["supported_for_outcome_screen"].all():
        unsupported = decisions.loc[
            ~decisions["supported_for_outcome_screen"], "route_id"
        ].tolist()
    else:
        unsupported = []

    run_dir = PREFLIGHT_ROOT / preflight_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "thresholds": run_dir / "g11_frozen_feature_thresholds.csv",
        "route_decisions": run_dir / "g11_route_support_decisions.csv",
    }
    g0.atomic_write_csv(thresholds, paths["thresholds"])
    g0.atomic_write_csv(decisions, paths["route_decisions"])
    for key, frame in support.items():
        path = run_dir / f"g11_{key}.csv"
        g0.atomic_write_csv(frame, path)
        paths[key] = path

    result_path = run_dir / "g11_preflight_record.json"
    result = {
        "schema_version": 1,
        "preflight_id": preflight_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation11_outcome_blind_support_preflight",
        "feature_rows_after_deduplication": len(rows),
        "routes": len(decisions),
        "supported_routes": int(decisions["supported_for_outcome_screen"].sum()),
        "unsupported_routes": unsupported,
        "thresholds_frozen": int(
            thresholds["status"].eq("frozen_from_development_predictors").sum()
        ),
        "outcome_blind_sequence": {
            "reaction_outcome_columns_read": False,
            "profit_columns_read": False,
            "future_direction_columns_read": False,
            "thresholds_use_development_predictors_only": True,
        },
        "source_contracts": {
            "generation11_freeze": artifact(g11z.FREEZE_PATH),
            "generation6_event_manifest": artifact(g11z.G6_EVENT_MANIFEST),
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "next_action": (
            "Run every supported route as one complete outcome batch. Park unsupported "
            "cells without changing thresholds, horizons, groups, or questions."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the outcome-blind Generation 11 broad support preflight."
    )
    parser.add_argument("--preflight-id", default=DEFAULT_PREFLIGHT_ID)
    args = parser.parse_args(argv)
    result = run_preflight(args.preflight_id)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
