"""Breadth-first direct screens for all seven Generation 6 sibling lanes.

The screens compare continuous reaction behaviour and preserve inverse or unexpected
relationships. They are deliberately labelled provisional: FreqAI profile comparisons,
uncertainty checks, and fresh confirmation remain separate steps.
"""

from __future__ import annotations

# Limit numerical libraries before pandas imports.
# ruff: noqa: E402
import argparse
import gc
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

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
    market_reaction_zone_generation6_preflight as g6,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
EVENT_RUN_ID = "g6_common_event_full_20260821a"
EVENT_RECORD_DIR = (
    OUTPUT_ROOT / "generation6_branches" / "g6_common_event_cache" / EVENT_RUN_ID
)
EVENT_MANIFEST = EVENT_RECORD_DIR / "manifest.json"
DIRECT_ROOT = OUTPUT_ROOT / "generation6_branches" / "g6_direct_screen"

METRICS = (
    "contact_volume_ratio",
    "contact_range_ratio",
    "absolute_contact_pressure_change",
    "abs_excursion_atr_h1",
    "abs_excursion_atr_h4",
    "volume_ratio_h1",
    "volume_ratio_h4",
    "range_ratio_h1",
    "range_ratio_h4",
    "absolute_pressure_change_h1",
    "absolute_pressure_change_h4",
    "dwell_fraction_h4",
    "crossings_h4",
)
LEVEL_CONTROLS = ("matched_random_time", "price_shift", "stale_72h", "near_miss")
CORE_LOCATION_CONTROLS = ("matched_random_time", "price_shift", "stale_72h")
MIN_EVENTS = 50
MIN_COINS = 5

BASE_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "representation",
    "control",
    "event_time",
    "period",
    "contact_close_distance_atr",
    "mechanism_group",
    "market_group",
    "smart_contract_platform",
)
RAW_METRIC_COLUMNS = (
    "contact_volume_ratio",
    "contact_range_ratio",
    "contact_pressure_change",
    "abs_excursion_atr_h1",
    "abs_excursion_atr_h4",
    "volume_ratio_h1",
    "volume_ratio_h4",
    "range_ratio_h1",
    "range_ratio_h4",
    "pressure_change_h1",
    "pressure_change_h4",
    "dwell_fraction_h4",
    "crossings_h4",
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

STATE_FEATURES: dict[str, tuple[str, Callable[[Series], Series]]] = {
    "relative_volume": ("state__relative_volume", lambda value: value),
    "volume_acceleration_magnitude": ("state__volume_acceleration", lambda value: value.abs()),
    "absolute_pressure": ("state__absolute_pressure", lambda value: value),
    "pressure_persistence": ("state__pressure_persistence", lambda value: value),
    "atr_fraction": ("state__atr_fraction", lambda value: value),
    "prior_range_atr": ("state__prior_range_atr", lambda value: value),
    "bollinger_width": ("state__bollinger_width", lambda value: value),
    "range_contraction_ratio": ("state__range_contraction", lambda value: value),
    "ema20_slope_magnitude": ("state__ema20_slope", lambda value: value.abs()),
    "ma_separation": ("state__ma_separation", lambda value: value),
    "return_slope_magnitude": ("state__return_slope", lambda value: value.abs()),
    "return_acceleration_magnitude": (
        "state__return_acceleration",
        lambda value: value.abs(),
    ),
    "adx14": ("state__adx14", lambda value: value),
    "rsi_distance_from_50": ("state__rsi14", lambda value: (value - 50.0).abs()),
    "rsi_change_magnitude": ("state__rsi_change", lambda value: value.abs()),
    "macd_histogram_magnitude": ("state__macd_histogram", lambda value: value.abs()),
    "macd_change_magnitude": (
        "state__macd_histogram_change",
        lambda value: value.abs(),
    ),
}

CROSS_MARKET_FEATURES: dict[str, tuple[str, Callable[[Series], Series]]] = {
    "btc_activity_1h": ("xm_btc_return_1h", lambda value: value.abs()),
    "btc_activity_4h": ("xm_btc_return_4h", lambda value: value.abs()),
    "btc_activity_24h": ("xm_btc_return_24h", lambda value: value.abs()),
    "btc_relative_volume": ("xm_btc_relative_volume", lambda value: value),
    "eth_activity_1h": ("xm_eth_return_1h", lambda value: value.abs()),
    "eth_activity_4h": ("xm_eth_return_4h", lambda value: value.abs()),
    "cohort_breadth_extremity": (
        "xm_cohort_breadth_positive",
        lambda value: (value - 0.5).abs(),
    ),
    "cohort_dispersion": ("xm_cohort_dispersion", lambda value: value),
    "cohort_absolute_activity": ("xm_cohort_absolute_activity", lambda value: value),
    "pair_btc_relative_activity_1h": (
        "xm_pair_minus_btc_1h",
        lambda value: value.abs(),
    ),
    "pair_btc_relative_activity_24h": (
        "xm_pair_minus_btc_24h",
        lambda value: value.abs(),
    ),
    "pair_btc_decoupling": (
        "xm_pair_btc_corr_168h",
        lambda value: (1.0 - value).abs(),
    ),
}

NEWS_FEATURES: dict[str, tuple[str, Callable[[Series], Series]]] = {
    "gdelt_activity": ("news_gdelt_activity", lambda value: value),
    "gdelt_tone_magnitude": ("news_gdelt__avg_tone_24h", lambda value: value.abs()),
    "gdelt_conflict_intensity": (
        "news_gdelt__goldstein_24h",
        lambda value: value.abs(),
    ),
    "geopolitics_and_energy": ("news_topic__geopolitics_and_energy", lambda value: value),
    "macro_policy_and_growth": ("news_topic__macro_policy_and_growth", lambda value: value),
    "crypto_policy_and_institutions": (
        "news_topic__crypto_policy_and_institutions",
        lambda value: value,
    ),
    "crypto_liquidity_and_security": (
        "news_topic__crypto_liquidity_and_security",
        lambda value: value,
    ),
}

ORDERBOOK_FEATURES: dict[str, tuple[str, Callable[[Series], Series]]] = {
    "btc_absolute_pressure": ("ob_btc_absolute_pressure", lambda value: value),
    "btc_pressure_change_from_quiet_threshold": (
        "ob_btc_pressure_25bps",
        lambda value: value.abs(),
    ),
    "btc_orderbook_coverage": ("ob_btc_coverage_ratio", lambda value: value),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all seven frozen Generation 6 direct reaction screens."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    event_manifest = validate_event_contract()
    group_support = load_group_support(event_manifest)
    run_dir = DIRECT_ROOT / args.run_id
    result_path = run_dir / "g6_direct_screen_result.json"
    if result_path.is_file() and not args.overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(existing["summary"], indent=2, sort_keys=True))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    event_paths = [Path(task["event_path"]) for task in event_manifest["tasks"]]

    level_events = load_event_rows(
        event_paths,
        columns=(*BASE_COLUMNS, *RAW_METRIC_COLUMNS),
        controls=None,
    )
    level_events = add_metric_columns(level_events)
    level_events = causal_deduplicate(
        level_events,
        keys=(
            "cohort",
            "pair",
            "event_time",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )
    level_effects = control_effects(
        level_events,
        group_keys=("cohort", "level_family", "source_timeframe", "period"),
        controls=LEVEL_CONTROLS,
    )
    level_candidates = repeated_control_candidates(
        level_effects,
        cell_keys=("cohort", "level_family", "source_timeframe", "metric"),
        required_controls=CORE_LOCATION_CONTROLS,
    )
    level_effects_path = run_dir / "g6a_calculated_area_control_effects.csv"
    level_candidates_path = run_dir / "g6a_calculated_area_candidates.csv"
    g0.atomic_write_csv(level_effects, level_effects_path)
    g0.atomic_write_csv(level_candidates, level_candidates_path)
    del level_events
    gc.collect()

    feature_columns = tuple(
        dict.fromkeys(
            [
                *BASE_COLUMNS,
                *RAW_METRIC_COLUMNS,
                *RELATIONSHIP_COLUMNS,
                *(source for source, _ in STATE_FEATURES.values()),
                *(source for source, _ in CROSS_MARKET_FEATURES.values()),
                *(source for source, _ in NEWS_FEATURES.values()),
                "news_gdelt_ready",
                "news_topics_ready",
                *(source for source, _ in ORDERBOOK_FEATURES.values()),
                "ob_btc_model_ready",
                "ob_pair_local",
            ]
        )
    )
    events = load_event_rows(
        event_paths,
        columns=feature_columns,
        controls=("actual", "matched_random_time"),
    )
    events = add_metric_columns(events)
    events = causal_deduplicate(
        events,
        keys=(
            "cohort",
            "pair",
            "event_time",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )

    relationship_effects, relationship_candidates = relationship_screen(events)
    state_result = conditioned_screen(
        events,
        lane="ohlcv_and_standard_indicators",
        feature_definitions=STATE_FEATURES,
        ready_mask=Series(True, index=events.index),
    )
    cross_result = conditioned_screen(
        events,
        lane="cross_market_crypto_state",
        feature_definitions=CROSS_MARKET_FEATURES,
        ready_mask=Series(True, index=events.index),
    )
    orderbook_result = orderbook_screen(events)
    news_result = news_screen(events)
    group_effects, group_candidates = market_group_screen(events, group_support)

    paths = {
        "g6b_timeframe_effects": write_frame(
            relationship_effects, run_dir / "g6b_timeframe_relationship_effects.csv"
        ),
        "g6b_timeframe_candidates": write_frame(
            relationship_candidates,
            run_dir / "g6b_timeframe_relationship_candidates.csv",
        ),
        **write_condition_result(run_dir, "g6c_ohlcv_indicator", state_result),
        **write_condition_result(run_dir, "g6d_cross_market", cross_result),
        **write_condition_result(run_dir, "g6e_orderbook", orderbook_result),
        **write_condition_result(run_dir, "g6f_news_context", news_result),
        "g6g_group_effects": write_frame(
            group_effects, run_dir / "g6g_market_group_effects.csv"
        ),
        "g6g_group_candidates": write_frame(
            group_candidates, run_dir / "g6g_market_group_candidates.csv"
        ),
        "g6a_level_effects": artifact_record(level_effects_path),
        "g6a_level_candidates": artifact_record(level_candidates_path),
    }
    decisions = branch_decisions(
        level_candidates=level_candidates,
        relationship_candidates=relationship_candidates,
        state_candidates=state_result["candidates"],
        cross_candidates=cross_result["candidates"],
        orderbook_candidates=orderbook_result["candidates"],
        news_candidates=news_result["candidates"],
        group_candidates=group_candidates,
    )
    decision_path = run_dir / "g6_direct_screen_branch_decisions.csv"
    g0.atomic_write_csv(decisions, decision_path)
    paths["branch_decisions"] = artifact_record(decision_path)
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_all_seven_sibling_direct_screens",
        "created_at_utc": g0.utc_now(),
        "event_manifest": artifact_record(EVENT_MANIFEST),
        "method": {
            "outcomes": list(METRICS),
            "profit_used": False,
            "direction_prediction": False,
            "condition_thresholds": (
                "33rd and 67th percentiles frozen from each cohort's development "
                "events without reading validation outcomes"
            ),
            "combined_test": (
                "difference-in-differences: actual-level minus matched-random effect in "
                "the high source band versus the same level effect in the low band"
            ),
            "screen_warning": (
                "Repeated median differences are provisional direct-test leads. They are "
                "not statistical confirmation, FreqAI promotion, direction, or profit."
            ),
        },
        "artifacts": paths,
        "summary": {
            "event_rows_used_for_source_screens": len(events),
            "calculated_area_candidates": int(level_candidates["candidate"].sum()),
            "timeframe_candidates": int(relationship_candidates["candidate"].sum()),
            "ohlcv_indicator_candidates": int(state_result["candidates"]["candidate"].sum()),
            "cross_market_candidates": int(cross_result["candidates"]["candidate"].sum()),
            "orderbook_candidates": int(orderbook_result["candidates"]["candidate"].sum()),
            "news_context_candidates": int(news_result["candidates"]["candidate"].sum()),
            "market_group_candidates": int(group_candidates["candidate"].sum()),
            "branches_completed": len(decisions),
            "profit_used": False,
            "direction_prediction": False,
        },
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def validate_event_contract() -> dict[str, Any]:
    if not EVENT_MANIFEST.is_file():
        raise FileNotFoundError(EVENT_MANIFEST)
    manifest = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("The shared Generation 6 event cache is incomplete.")
    summary = manifest.get("summary", {})
    if summary.get("causal_timestamp_violations") != 0:
        raise ValueError("The shared event cache contains causal violations.")
    if summary.get("profit_used") is not False or summary.get("direction_prediction") is not False:
        raise ValueError("The shared event cache crossed the research boundary.")
    if len(manifest.get("tasks", [])) != 20:
        raise ValueError("The full normal-plus-meme event portfolio is incomplete.")
    preflight = manifest.get("preflight", {})
    preflight_path = Path(str(preflight.get("path", "")))
    if (
        not preflight_path.is_file()
        or g0.sha256_file(preflight_path) != preflight.get("sha256")
    ):
        raise ValueError("The frozen outcome-blind preflight changed or is missing.")
    for task in manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Event artifact changed or is missing: {path}")
    return manifest


def load_group_support(event_manifest: dict[str, Any]) -> DataFrame:
    preflight_path = Path(event_manifest["preflight"]["path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    artifact = preflight.get("artifacts", {}).get("market_group_transfer_support", {})
    support_path = Path(str(artifact.get("path", "")))
    if not support_path.is_file() or g0.sha256_file(support_path) != artifact.get("sha256"):
        raise ValueError("The market-group preflight support file changed or is missing.")
    support = pd.read_csv(support_path)
    required = {"group_id", "question", "supported"}
    missing = required.difference(support.columns)
    if missing:
        raise ValueError(f"Market-group support is missing columns: {sorted(missing)}")
    return support


def load_event_rows(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    controls: Sequence[str] | None,
) -> DataFrame:
    frames: list[DataFrame] = []
    for path in paths:
        filters = [("control", "in", list(controls))] if controls is not None else None
        frames.append(pd.read_parquet(path, columns=list(columns), filters=filters))
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    return frame.dropna(subset=["event_time"]).reset_index(drop=True)


def add_metric_columns(frame: DataFrame) -> DataFrame:
    output = frame.copy()
    output["absolute_contact_pressure_change"] = pd.to_numeric(
        output["contact_pressure_change"], errors="coerce"
    ).abs()
    output["absolute_pressure_change_h1"] = pd.to_numeric(
        output["pressure_change_h1"], errors="coerce"
    ).abs()
    output["absolute_pressure_change_h4"] = pd.to_numeric(
        output["pressure_change_h4"], errors="coerce"
    ).abs()
    for metric in METRICS:
        output[metric] = pd.to_numeric(output[metric], errors="coerce")
    return output


def causal_deduplicate(frame: DataFrame, *, keys: Sequence[str]) -> DataFrame:
    ordered = frame.assign(
        _distance=pd.to_numeric(frame["contact_close_distance_atr"], errors="coerce")
    ).sort_values([*keys, "_distance"], na_position="last")
    return (
        ordered.drop_duplicates(list(keys), keep="first")
        .drop(columns="_distance")
        .reset_index(drop=True)
    )


def metric_summary(frame: DataFrame, *, group_keys: Sequence[str]) -> DataFrame:
    rows: list[DataFrame] = []
    for metric in METRICS:
        source = frame.loc[frame[metric].notna()]
        if source.empty:
            continue
        summary = (
            source.groupby(list(group_keys), dropna=False, observed=True)
            .agg(
                median=(metric, "median"),
                mean=(metric, "mean"),
                rows=(metric, "size"),
                coins=("pair", "nunique"),
            )
            .reset_index()
        )
        summary["metric"] = metric
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else DataFrame()


def control_effects(
    frame: DataFrame,
    *,
    group_keys: Sequence[str],
    controls: Sequence[str],
) -> DataFrame:
    summary = metric_summary(frame, group_keys=(*group_keys, "control"))
    actual = summary.loc[summary["control"].eq("actual")].drop(columns="control")
    rows: list[DataFrame] = []
    merge_keys = [*group_keys, "metric"]
    for control in controls:
        comparison = summary.loc[summary["control"].eq(control)].drop(columns="control")
        joined = actual.merge(
            comparison,
            on=merge_keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        joined["comparison_control"] = control
        joined["median_difference"] = joined["median_actual"] - joined["median_control"]
        joined["relative_median_difference"] = joined["median_difference"].div(
            joined["median_control"].abs().replace(0.0, np.nan)
        )
        joined["effect_sign"] = np.sign(joined["median_difference"]).astype(int)
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else DataFrame()


def validation_periods(cohort: str) -> tuple[str, str]:
    return g6.validation_periods_for_cohort(cohort)


def development_period(cohort: str) -> str:
    return "development" if cohort == "normal" else "meme_development"


def repeated_control_candidates(
    effects: DataFrame,
    *,
    cell_keys: Sequence[str],
    required_controls: Sequence[str],
    min_events: int = MIN_EVENTS,
    min_coins: int = MIN_COINS,
) -> DataFrame:
    if effects.empty:
        return DataFrame(columns=[*cell_keys, "candidate"])
    candidate_keys = [*cell_keys, "comparison_control"]
    control_rows: list[dict[str, Any]] = []
    for key, cell in effects.groupby(candidate_keys, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(candidate_keys, values, strict=True))
        required_periods = validation_periods(str(record["cohort"]))
        checks = cell.loc[cell["period"].isin(required_periods)]
        signs = checks["effect_sign"].loc[checks["effect_sign"].ne(0)]
        repeated = (
            set(checks["period"]) == set(required_periods)
            and checks["rows_actual"].ge(min_events).all()
            and checks["rows_control"].ge(min_events).all()
            and checks["coins_actual"].ge(min_coins).all()
            and checks["coins_control"].ge(min_coins).all()
            and len(signs) == len(required_periods)
            and signs.nunique() == 1
        )
        record["repeated"] = bool(repeated)
        record["repeated_sign"] = int(signs.iloc[0]) if repeated else 0
        record["period_effects"] = json.dumps(
            checks[
                [
                    "period",
                    "median_actual",
                    "median_control",
                    "median_difference",
                    "rows_actual",
                    "rows_control",
                    "coins_actual",
                    "coins_control",
                ]
            ].to_dict("records"),
            sort_keys=True,
        )
        control_rows.append(record)
    by_control = DataFrame.from_records(control_rows)
    final_rows: list[dict[str, Any]] = []
    base_keys = list(cell_keys)
    for key, cell in by_control.groupby(base_keys, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(base_keys, values, strict=True))
        indexed = cell.set_index("comparison_control")
        present = [control for control in required_controls if control in indexed.index]
        repeated = [bool(indexed.loc[control, "repeated"]) for control in present]
        signs = [int(indexed.loc[control, "repeated_sign"]) for control in present]
        candidate = (
            len(present) == len(required_controls)
            and all(repeated)
            and len(set(signs)) == 1
            and signs[0] != 0
        )
        record["candidate"] = bool(candidate)
        record["behaviour_sign"] = signs[0] if candidate else 0
        record["controls_present"] = ",".join(present)
        record["control_checks"] = json.dumps(
            cell[
                [
                    "comparison_control",
                    "repeated",
                    "repeated_sign",
                    "period_effects",
                ]
            ].to_dict("records"),
            sort_keys=True,
        )
        record["interpretation"] = (
            "provisional_control_resistant_reaction_difference"
            if candidate
            else "mixed_or_control_reproduced_direct_screen"
        )
        final_rows.append(record)
    return DataFrame.from_records(final_rows)


def relationship_screen(events: DataFrame) -> tuple[DataFrame, DataFrame]:
    actual = events.loc[events["control"].eq("actual")].copy()
    rows: list[DataFrame] = []
    for column in RELATIONSHIP_COLUMNS:
        _, higher, relationship = column.split("__", maxsplit=2)
        anchor = higher if relationship == "isolated_higher_timeframe" else "1h"
        source = actual.loc[actual["source_timeframe"].eq(anchor)].copy()
        source["relationship_present"] = source[column].fillna(False).astype(bool)
        source = causal_deduplicate(
            source,
            keys=("cohort", "pair", "event_time", "mechanism_group", "relationship_present"),
        )
        summary = metric_summary(
            source,
            group_keys=("cohort", "period", "relationship_present"),
        )
        present = summary.loc[summary["relationship_present"]].drop(
            columns="relationship_present"
        )
        absent = summary.loc[~summary["relationship_present"]].drop(
            columns="relationship_present"
        )
        joined = present.merge(
            absent,
            on=["cohort", "period", "metric"],
            how="inner",
            suffixes=("_present", "_absent"),
            validate="one_to_one",
        )
        joined["higher_timeframe"] = higher
        joined["relationship"] = relationship
        joined["median_difference"] = joined["median_present"] - joined["median_absent"]
        joined["effect_sign"] = np.sign(joined["median_difference"]).astype(int)
        rows.append(joined)
    effects = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    candidates: list[dict[str, Any]] = []
    for key, cell in effects.groupby(
        ["cohort", "higher_timeframe", "relationship", "metric"], dropna=False
    ):
        cohort, higher, relationship, metric = key
        periods = validation_periods(str(cohort))
        checks = cell.loc[cell["period"].isin(periods)]
        signs = checks["effect_sign"].loc[checks["effect_sign"].ne(0)]
        candidate = (
            set(checks["period"]) == set(periods)
            and checks["rows_present"].ge(MIN_EVENTS).all()
            and checks["rows_absent"].ge(MIN_EVENTS).all()
            and checks["coins_present"].ge(MIN_COINS).all()
            and checks["coins_absent"].ge(MIN_COINS).all()
            and len(signs) == len(periods)
            and signs.nunique() == 1
        )
        candidates.append(
            {
                "cohort": cohort,
                "higher_timeframe": higher,
                "relationship": relationship,
                "metric": metric,
                "candidate": bool(candidate),
                "behaviour_sign": int(signs.iloc[0]) if candidate else 0,
                "interpretation": (
                    "provisional_repeated_relationship_difference_needs_width_matching"
                    if candidate
                    else "mixed_or_insufficient_relationship_screen"
                ),
                "period_effects": json.dumps(
                    checks[
                        [
                            "period",
                            "median_present",
                            "median_absent",
                            "median_difference",
                            "rows_present",
                            "rows_absent",
                            "coins_present",
                            "coins_absent",
                        ]
                    ].to_dict("records"),
                    sort_keys=True,
                ),
            }
        )
    return effects, DataFrame.from_records(candidates)


def transformed_feature(
    frame: DataFrame,
    definition: tuple[str, Callable[[Series], Series]],
) -> Series:
    source, transform = definition
    values = pd.to_numeric(frame[source], errors="coerce")
    return transform(values).replace([np.inf, -np.inf], np.nan)


def conditioned_screen(
    events: DataFrame,
    *,
    lane: str,
    feature_definitions: dict[str, tuple[str, Callable[[Series], Series]]],
    ready_mask: Series,
    extra_filter: Series | None = None,
    scope: str = "all_supported_one_hour_levels",
    min_events: int = MIN_EVENTS,
    min_coins: int = MIN_COINS,
) -> dict[str, DataFrame]:
    base_filter = events["source_timeframe"].eq("1h") & ready_mask.fillna(False)
    if extra_filter is not None:
        base_filter &= extra_filter.fillna(False)
    source = events.loc[base_filter].copy()
    thresholds: list[dict[str, Any]] = []
    effects: list[DataFrame] = []
    interactions: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        cohort_rows = source.loc[source["cohort"].eq(cohort)].copy()
        development = development_period(cohort)
        for feature, definition in feature_definitions.items():
            values = transformed_feature(cohort_rows, definition)
            development_values = values.loc[
                cohort_rows["period"].eq(development) & cohort_rows["control"].eq("actual")
            ].dropna()
            if len(development_values) < MIN_EVENTS:
                thresholds.append(
                    {
                        "lane": lane,
                        "scope": scope,
                        "cohort": cohort,
                        "feature": feature,
                        "development_rows": len(development_values),
                        "status": "park_without_outcomes_insufficient_development_values",
                    }
                )
                continue
            lower = float(development_values.quantile(1.0 / 3.0))
            upper = float(development_values.quantile(2.0 / 3.0))
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                thresholds.append(
                    {
                        "lane": lane,
                        "scope": scope,
                        "cohort": cohort,
                        "feature": feature,
                        "development_rows": len(development_values),
                        "lower": lower,
                        "upper": upper,
                        "status": "park_without_outcomes_degenerate_thresholds",
                    }
                )
                continue
            thresholds.append(
                {
                    "lane": lane,
                    "scope": scope,
                    "cohort": cohort,
                    "feature": feature,
                    "development_rows": len(development_values),
                    "lower": lower,
                    "upper": upper,
                    "status": "frozen_from_development",
                }
            )
            feature_rows = cohort_rows.copy()
            feature_rows["feature_value"] = values
            feature_rows["feature_band"] = pd.cut(
                feature_rows["feature_value"],
                bins=[-np.inf, lower, upper, np.inf],
                labels=["low", "middle", "high"],
                include_lowest=True,
            ).astype("string")
            comparison = control_effects(
                feature_rows.dropna(subset=["feature_band"]),
                group_keys=("cohort", "level_family", "period", "feature_band"),
                controls=("matched_random_time",),
            )
            comparison["lane"] = lane
            comparison["scope"] = scope
            comparison["feature"] = feature
            effects.append(comparison)
            interactions.extend(
                interaction_rows(
                    comparison,
                    lane=lane,
                    scope=scope,
                    cohort=cohort,
                    feature=feature,
                )
            )
    threshold_frame = DataFrame.from_records(thresholds)
    effect_frame = pd.concat(effects, ignore_index=True) if effects else DataFrame()
    interaction_frame = DataFrame.from_records(interactions)
    candidate_frame = repeated_interaction_candidates(
        interaction_frame,
        min_events=min_events,
        min_coins=min_coins,
    )
    return {
        "thresholds": threshold_frame,
        "effects": effect_frame,
        "interactions": interaction_frame,
        "candidates": candidate_frame,
    }


def interaction_rows(
    effects: DataFrame,
    *,
    lane: str,
    scope: str,
    cohort: str,
    feature: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, cell in effects.groupby(["level_family", "period", "metric"], dropna=False):
        level_family, period, metric = key
        indexed = cell.set_index("feature_band")
        if "low" not in indexed.index or "high" not in indexed.index:
            continue
        low = indexed.loc["low"]
        high = indexed.loc["high"]
        rows.append(
            {
                "lane": lane,
                "scope": scope,
                "cohort": cohort,
                "feature": feature,
                "level_family": level_family,
                "period": period,
                "metric": metric,
                "low_level_increment": float(low["median_difference"]),
                "high_level_increment": float(high["median_difference"]),
                "interaction_difference": float(
                    high["median_difference"] - low["median_difference"]
                ),
                "effect_sign": int(
                    np.sign(high["median_difference"] - low["median_difference"])
                ),
                "low_actual_rows": int(low["rows_actual"]),
                "low_control_rows": int(low["rows_control"]),
                "high_actual_rows": int(high["rows_actual"]),
                "high_control_rows": int(high["rows_control"]),
                "low_actual_coins": int(low["coins_actual"]),
                "low_control_coins": int(low["coins_control"]),
                "high_actual_coins": int(high["coins_actual"]),
                "high_control_coins": int(high["coins_control"]),
            }
        )
    return rows


def repeated_interaction_candidates(
    interactions: DataFrame,
    *,
    min_events: int = MIN_EVENTS,
    min_coins: int = MIN_COINS,
) -> DataFrame:
    if interactions.empty:
        return DataFrame(
            columns=["lane", "scope", "cohort", "feature", "level_family", "metric", "candidate"]
        )
    rows: list[dict[str, Any]] = []
    keys = ["lane", "scope", "cohort", "feature", "level_family", "metric"]
    for key, cell in interactions.groupby(keys, dropna=False):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        periods = validation_periods(str(record["cohort"]))
        checks = cell.loc[cell["period"].isin(periods)]
        signs = checks["effect_sign"].loc[checks["effect_sign"].ne(0)]
        count_columns = (
            "low_actual_rows",
            "low_control_rows",
            "high_actual_rows",
            "high_control_rows",
        )
        coin_columns = (
            "low_actual_coins",
            "low_control_coins",
            "high_actual_coins",
            "high_control_coins",
        )
        candidate = (
            set(checks["period"]) == set(periods)
            and checks[list(count_columns)].ge(min_events).all(axis=None)
            and checks[list(coin_columns)].ge(min_coins).all(axis=None)
            and len(signs) == len(periods)
            and signs.nunique() == 1
        )
        record["candidate"] = bool(candidate)
        record["behaviour_sign"] = int(signs.iloc[0]) if candidate else 0
        record["interpretation"] = (
            "provisional_combined_level_and_source_interaction"
            if candidate
            else "mixed_or_insufficient_direct_interaction"
        )
        record["period_interactions"] = json.dumps(
            checks.to_dict("records"), sort_keys=True, default=g0.json_default
        )
        rows.append(record)
    return DataFrame.from_records(rows)


def orderbook_screen(events: DataFrame) -> dict[str, DataFrame]:
    base = events.loc[~events["level_family"].eq("volume_profile_nodes")].copy()
    scopes: list[dict[str, Any]] = []
    scope_definitions = {
        "btc_pair_local": (base["pair"].eq("BTC/USDT:USDT"), 1),
        "btc_wide_established_alts": (
            base["market_group"].eq("established_altcoins"),
            MIN_COINS,
        ),
        "btc_wide_memes": (
            base["market_group"].eq("frozen_top_ten_memes"),
            MIN_COINS,
        ),
    }
    for scope, (mask, min_coins) in scope_definitions.items():
        result = conditioned_screen(
            base,
            lane="orderbook_state",
            feature_definitions=ORDERBOOK_FEATURES,
            ready_mask=base["ob_btc_model_ready"].fillna(False).astype(bool),
            extra_filter=mask,
            scope=scope,
            min_coins=min_coins,
        )
        for key, frame in result.items():
            scopes.append({"scope": scope, "kind": key, "frame": frame})
    return combine_condition_scopes(scopes)


def news_screen(events: DataFrame) -> dict[str, DataFrame]:
    base = events.loc[~events["level_family"].eq("volume_profile_nodes")].copy()
    gdelt_features = {
        key: value for key, value in NEWS_FEATURES.items() if key.startswith("gdelt_")
    }
    topic_features = {
        key: value for key, value in NEWS_FEATURES.items() if not key.startswith("gdelt_")
    }
    scopes: list[dict[str, Any]] = []
    for scope, features, ready in (
        (
            "gdelt_aggregate_distinct_levels",
            gdelt_features,
            base["news_gdelt_ready"].fillna(False).astype(bool),
        ),
        (
            "gdelt_topic_groups_distinct_levels",
            topic_features,
            base["news_topics_ready"].fillna(False).astype(bool),
        ),
    ):
        result = conditioned_screen(
            base,
            lane="news_media_and_web_context",
            feature_definitions=features,
            ready_mask=ready,
            scope=scope,
        )
        for key, frame in result.items():
            scopes.append({"scope": scope, "kind": key, "frame": frame})
    return combine_condition_scopes(scopes)


def combine_condition_scopes(scopes: Iterable[dict[str, Any]]) -> dict[str, DataFrame]:
    output: dict[str, list[DataFrame]] = {
        "thresholds": [],
        "effects": [],
        "interactions": [],
        "candidates": [],
    }
    for item in scopes:
        output[item["kind"]].append(item["frame"])
    return {
        key: pd.concat(frames, ignore_index=True) if frames else DataFrame()
        for key, frames in output.items()
    }


def market_group_screen(
    events: DataFrame,
    support: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    views: list[DataFrame] = []
    definitions = {
        "btc_separate": events["pair"].eq("BTC/USDT:USDT"),
        "established_altcoins": events["market_group"].eq("established_altcoins"),
        "smart_contract_platforms": events["smart_contract_platform"].fillna(False).astype(bool),
        "payments_and_transfer": events["pair"].isin(g6.GROUPS["payments_and_transfer"]),
        "frozen_top_ten_memes": events["market_group"].eq("frozen_top_ten_memes"),
    }
    for group_id, mask in definitions.items():
        selected = events.loc[mask].copy()
        selected["group_id"] = group_id
        views.append(selected)
    frame = pd.concat(views, ignore_index=True)
    questions: list[DataFrame] = []
    question_masks = {
        "one_hour_thin_lvn_activity_benchmark": (
            frame["source_timeframe"].eq("1h")
            & frame["level_family"].eq("volume_profile_nodes")
            & frame["level_name"].astype(str).str.contains("lvn")
        ),
        "volume_profile_value_area_reaction": (
            frame["source_timeframe"].eq("1h")
            & frame["level_family"].eq("volume_profile_settled")
            & frame["level_name"].isin(["vah", "val"])
        ),
        "prior_week_or_month_extreme_reaction": (
            frame["source_timeframe"].eq("1h")
            & frame["level_family"].eq("generic_prior_range")
            & frame["level_name"].astype(str).str.contains("168|720", regex=True)
        ),
        "higher_timeframe_agreement_increment": (
            frame["source_timeframe"].eq("1h")
            & frame[[
                column
                for column in RELATIONSHIP_COLUMNS
                if column.endswith("agreement")
            ]]
            .fillna(False)
            .any(axis=1)
        ),
    }
    for question, mask in question_masks.items():
        selected = frame.loc[mask].copy()
        selected["question"] = question
        questions.append(selected)
    question_frame = pd.concat(questions, ignore_index=True)
    supported = support.loc[
        support["supported"].astype(str).str.lower().eq("true"),
        ["group_id", "question"],
    ].drop_duplicates()
    question_frame = question_frame.merge(
        supported,
        on=["group_id", "question"],
        how="inner",
        validate="many_to_one",
    )
    effects = control_effects(
        question_frame,
        group_keys=("group_id", "cohort", "question", "period"),
        controls=("matched_random_time",),
    )
    candidate_frames: list[DataFrame] = []
    group_requirements = {
        "btc_separate": (g6.MIN_BTC_EVENTS, 1),
        "established_altcoins": (g6.MIN_FULL_EVENTS, g6.MIN_FULL_COINS),
        "smart_contract_platforms": (
            g6.MIN_SUBGROUP_EVENTS,
            g6.MIN_SUBGROUP_COINS,
        ),
        "frozen_top_ten_memes": (g6.MIN_FULL_EVENTS, g6.MIN_FULL_COINS),
    }
    for group_id, (min_events, min_coins) in group_requirements.items():
        group_effects = effects.loc[effects["group_id"].eq(group_id)]
        if group_effects.empty:
            continue
        candidate_frames.append(
            repeated_control_candidates(
                group_effects,
                cell_keys=("group_id", "cohort", "question", "metric"),
                required_controls=("matched_random_time",),
                min_events=min_events,
                min_coins=min_coins,
            )
        )
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else DataFrame()
    )
    candidates["note"] = (
        "The Generation 5 pre-contact/next-volume lead receives a separate fresh "
        "confirmation cell and is not inferred from this table."
    )
    return effects, candidates


def write_condition_result(
    run_dir: Path, prefix: str, result: dict[str, DataFrame]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for kind, frame in result.items():
        path = run_dir / f"{prefix}_{kind}.csv"
        output[f"{prefix}_{kind}"] = write_frame(frame, path)
    return output


def branch_decisions(
    *,
    level_candidates: DataFrame,
    relationship_candidates: DataFrame,
    state_candidates: DataFrame,
    cross_candidates: DataFrame,
    orderbook_candidates: DataFrame,
    news_candidates: DataFrame,
    group_candidates: DataFrame,
) -> DataFrame:
    rows = []
    definitions = (
        ("g6a_calculated_price_areas", "Calculated price areas", level_candidates),
        ("g6b_timeframe_relationships", "Timeframe relationships", relationship_candidates),
        (
            "g6c_recent_ohlcv_and_standard_indicator_state",
            "Recent OHLCV and standard-indicator state",
            state_candidates,
        ),
        ("g6d_cross_market_and_global_state", "Cross-market and global state", cross_candidates),
        ("g6e_orderbook_state", "Orderbook state", orderbook_candidates),
        ("g6f_news_media_and_web_context", "News, media, and web context", news_candidates),
        ("g6g_market_group_transfer", "Market-group transfer", group_candidates),
    )
    for branch_id, plain_name, candidates in definitions:
        candidate_count = int(candidates["candidate"].sum()) if len(candidates) else 0
        rows.append(
            {
                "branch_id": branch_id,
                "plain_name": plain_name,
                "screened_relationships": len(candidates),
                "provisional_candidates": candidate_count,
                "status": (
                    "completed_with_provisional_direct_leads"
                    if candidate_count
                    else "completed_without_repeatable_direct_lead"
                ),
                "next_action": (
                    "Do not branch yet. Complete the frozen FreqAI/control portfolio and "
                    "then review all seven siblings jointly."
                ),
            }
        )
    return DataFrame.from_records(rows)


def write_frame(frame: DataFrame, path: Path) -> dict[str, Any]:
    g0.atomic_write_csv(frame, path)
    return artifact_record(path)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
