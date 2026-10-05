"""Freeze seven historical news/market interaction families before outcome review.

The source parquet files already contain future-price columns, so this phase reads an
explicit feature-only column list.  It freezes whole episodes and matched component
controls without inspecting any future outcome or profit field.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
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


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "historical_news_interaction_retest_20260911a"
SOURCE_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "reports"
    / "historical_news_overlay"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / RUN_ID
)

WINDOW_PATHS: Mapping[str, Path] = {
    "w1_2020_jan_oct": SOURCE_ROOT
    / "historical_news_overlay_frame_batch3_refined_w2_w1_w3_w1_2020_jan_oct.parquet",
    "w2_2021_mar_2022_jul": SOURCE_ROOT
    / "historical_news_overlay_frame_batch3_refined_w2_w1_w3_w2_2021_mar_2022_jul.parquet",
    "w3_2022_aug_dec_gdelt_only": SOURCE_ROOT
    / "historical_news_overlay_frame_batch3_refined_w2_w1_w3_w3_2022_aug_dec_gdelt_only.parquet",
}
OLD_RESULT_PATH = (
    SOURCE_ROOT / "historical_news_overlay_theory_results_batch3_refined_w2_w1_w3.csv"
)

EPISODES_PATH = OUTPUT_ROOT / "historical_news_interaction_episodes.csv"
CONTROLS_PATH = OUTPUT_ROOT / "historical_news_interaction_controls.csv"
STALE_EPISODES_PATH = OUTPUT_ROOT / "historical_news_interaction_stale_episodes.csv"
COVERAGE_PATH = OUTPUT_ROOT / "historical_news_interaction_coverage.csv"
FREEZE_PATH = OUTPUT_ROOT / "historical_news_interaction_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "historical_news_interaction_freeze_result.json"

EPISODE_GAP_HOURS = 24
CONTROL_EXCLUSION_HOURS = 24
SOURCE_STALE_HOURS = 168
MINIMUM_EPISODES_PER_WINDOW = 6
MINIMUM_SUPPORTED_WINDOWS = 2
MINIMUM_JOINT_RATE = 0.55
MAIN_JOINT_TARGET = 0.65
MINIMUM_MARKET_ONLY_LIFT = 0.05
MINIMUM_NEWS_ONLY_LIFT = 0.03
MINIMUM_STALE_LIFT = 0.03
MAX_OPPOSITE_WINDOW_LIFT = -0.05
PERMUTATION_ITERATIONS = 2_000
PERMUTATION_SEED = 20260911

PRICE_MATCH_COLUMNS = (
    "ret_6h",
    "ret_24h",
    "volume_z_24h",
    "volatility_24h",
    "range_position_30d",
)

THEORIES: tuple[dict[str, Any], ...] = (
    {
        "theory_id": "macro_relief_near_low",
        "plain_question": (
            "When elevated macro pressure eases near the bottom of the recent range, "
            "does Bitcoin react upward more often than from the same price location alone?"
        ),
        "expected_direction": 1,
        "source_kind": "macro_relief",
        "market_kind": "near_range_low",
        "source_match_columns": (
            "macro_stress__z_168h",
            "macro_stress__delta_6h",
            "news_activity__pct_rank_720h",
        ),
        "supported_windows": tuple(WINDOW_PATHS),
    },
    {
        "theory_id": "bad_news_absorbed_near_low",
        "plain_question": (
            "When bad-news pressure is active but price holds near support and starts "
            "rising, is that absorption associated with a later upward reaction?"
        ),
        "expected_direction": 1,
        "source_kind": "bad_news",
        "market_kind": "absorbed_near_low",
        "source_match_columns": (
            "macro_stress__z_168h",
            "crypto_stress__z_168h",
            "news_activity__pct_rank_720h",
        ),
        "supported_windows": tuple(WINDOW_PATHS),
    },
    {
        "theory_id": "bad_news_after_breakdown",
        "plain_question": (
            "When bad-news pressure accompanies a lower-low break with elevated volume, "
            "does the downward reaction continue more often than after the break alone?"
        ),
        "expected_direction": -1,
        "source_kind": "bad_news",
        "market_kind": "lower_break_high_volume",
        "source_match_columns": (
            "macro_stress__z_168h",
            "crypto_stress__z_168h",
            "news_activity__pct_rank_720h",
        ),
        "supported_windows": tuple(WINDOW_PATHS),
    },
    {
        "theory_id": "high_attention_compression_down",
        "plain_question": (
            "When unusually high news attention meets compressed price already leaning "
            "down, is a meaningful downward reaction more likely?"
        ),
        "expected_direction": -1,
        "source_kind": "high_attention_80",
        "market_kind": "compression_down",
        "source_match_columns": ("news_activity__pct_rank_720h",),
        "supported_windows": tuple(WINDOW_PATHS),
    },
    {
        "theory_id": "high_attention_breakout_not_extended",
        "plain_question": (
            "When unusually high news attention meets a high-volume breakout that is "
            "not already at the range extreme, is upward follow-through more likely?"
        ),
        "expected_direction": 1,
        "source_kind": "high_attention_75",
        "market_kind": "breakout_not_extended",
        "source_match_columns": ("news_activity__pct_rank_720h",),
        "supported_windows": tuple(WINDOW_PATHS),
    },
    {
        "theory_id": "crypto_stress_near_range_high",
        "plain_question": (
            "When crypto-specific stress is elevated near the top of the recent range, "
            "is a meaningful downward rejection more likely?"
        ),
        "expected_direction": -1,
        "source_kind": "crypto_stress",
        "market_kind": "near_range_high",
        "source_match_columns": (
            "crypto_stress__z_168h",
            "news_activity__pct_rank_720h",
        ),
        "supported_windows": (
            "w1_2020_jan_oct",
            "w2_2021_mar_2022_jul",
        ),
    },
    {
        "theory_id": "macro_stress_ignored_while_rising",
        "plain_question": (
            "When macro stress is elevated but Bitcoin is already rising, does that "
            "relative strength precede a meaningful upward reaction?"
        ),
        "expected_direction": 1,
        "source_kind": "macro_stress",
        "market_kind": "price_rising",
        "source_match_columns": (
            "macro_stress__z_168h",
            "news_activity__pct_rank_720h",
        ),
        "supported_windows": tuple(WINDOW_PATHS),
    },
)

FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        (
            "date",
            "news_any_present",
            "gkg_present",
            "bad_news_escalating",
            "news_activity__pct_rank_720h",
            "macro_stress__z_168h",
            "macro_stress__delta_6h",
            "crypto_stress__z_168h",
            "near_range_low",
            "near_range_high",
            "break_lower_low_24h",
            "break_higher_high_24h",
            "compression_state",
            *PRICE_MATCH_COLUMNS,
        )
    )
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _strict_bool(values: Series) -> Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    unexpected = normalized[~normalized.isin(["true", "false", "1", "0", "<na>"])]
    if not unexpected.empty:
        raise ValueError(f"Unexpected boolean values: {sorted(unexpected.unique())}")
    return normalized.isin(["true", "1"])


def load_feature_frame(window_id: str) -> DataFrame:
    path = WINDOW_PATHS[window_id]
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=list(FEATURE_COLUMNS))
    if any(
        token in column.lower()
        for column in frame.columns
        for token in ("future", "profit", "target", "label")
    ):
        raise ValueError("Freeze phase unexpectedly loaded an outcome column.")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"Invalid hourly date index in {path}")
    for column in (
        "news_any_present",
        "gkg_present",
        "bad_news_escalating",
        "near_range_low",
        "near_range_high",
        "break_lower_low_24h",
        "break_higher_high_24h",
        "compression_state",
    ):
        frame[column] = _strict_bool(frame[column])
    return frame


def build_source_mask(
    frame: DataFrame,
    theory: Mapping[str, Any],
    *,
    source_lag_hours: int = 0,
) -> Series:
    lag = int(source_lag_hours)
    news = frame["news_any_present"].shift(lag, fill_value=False)
    gkg = frame["gkg_present"].shift(lag, fill_value=False)
    bad = frame["bad_news_escalating"].shift(lag, fill_value=False)
    attention = pd.to_numeric(
        frame["news_activity__pct_rank_720h"], errors="coerce"
    ).shift(lag)
    macro_z = pd.to_numeric(frame["macro_stress__z_168h"], errors="coerce")
    macro_delta = pd.to_numeric(
        frame["macro_stress__delta_6h"], errors="coerce"
    )
    crypto_z = pd.to_numeric(frame["crypto_stress__z_168h"], errors="coerce")

    source_kind = str(theory["source_kind"])
    if source_kind == "macro_relief":
        source = (
            news
            & macro_z.shift(6 + lag).gt(1.25)
            & macro_delta.shift(lag).lt(0)
        )
    elif source_kind == "bad_news":
        source = news & bad
    elif source_kind == "high_attention_80":
        source = news & attention.gt(0.80)
    elif source_kind == "high_attention_75":
        source = news & attention.gt(0.75)
    elif source_kind == "crypto_stress":
        source = gkg & crypto_z.shift(lag).gt(1.0)
    elif source_kind == "macro_stress":
        source = news & macro_z.shift(lag).gt(1.0)
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")
    return source.fillna(False).astype(bool)


def build_market_mask(frame: DataFrame, theory: Mapping[str, Any]) -> Series:
    market_kind = str(theory["market_kind"])
    if market_kind == "near_range_low":
        market = frame["near_range_low"]
    elif market_kind == "absorbed_near_low":
        market = (
            frame["near_range_low"]
            & ~frame["break_lower_low_24h"]
            & pd.to_numeric(frame["ret_6h"], errors="coerce").ge(0)
        )
    elif market_kind == "lower_break_high_volume":
        market = (
            frame["break_lower_low_24h"]
            & pd.to_numeric(frame["volume_z_24h"], errors="coerce").gt(0.5)
        )
    elif market_kind == "compression_down":
        market = frame["compression_state"] & pd.to_numeric(
            frame["ret_6h"], errors="coerce"
        ).lt(0)
    elif market_kind == "breakout_not_extended":
        market = (
            frame["break_higher_high_24h"]
            & pd.to_numeric(frame["volume_z_24h"], errors="coerce").gt(0.5)
            & pd.to_numeric(frame["range_position_30d"], errors="coerce").lt(0.95)
        )
    elif market_kind == "near_range_high":
        market = frame["near_range_high"]
    elif market_kind == "price_rising":
        market = pd.to_numeric(frame["ret_6h"], errors="coerce").gt(0.01)
    else:
        raise ValueError(f"Unknown market kind: {market_kind}")
    return market.fillna(False).astype(bool)


def build_component_masks(
    frame: DataFrame,
    theory: Mapping[str, Any],
    *,
    source_lag_hours: int = 0,
) -> tuple[Series, Series]:
    return (
        build_source_mask(frame, theory, source_lag_hours=source_lag_hours),
        build_market_mask(frame, theory),
    )


def cluster_mask(
    dates: Series,
    mask: Series,
    *,
    gap_hours: int = EPISODE_GAP_HOURS,
) -> DataFrame:
    raw = pd.DatetimeIndex(dates.loc[mask.fillna(False)].sort_values().unique())
    records: list[dict[str, Any]] = []
    cluster_start: pd.Timestamp | None = None
    cluster_end: pd.Timestamp | None = None
    count = 0
    for value in raw:
        stamp = pd.Timestamp(value)
        if cluster_end is None or stamp - cluster_end > pd.Timedelta(hours=gap_hours):
            if cluster_start is not None and cluster_end is not None:
                records.append(
                    {
                        "feature_candle_open_utc": cluster_start,
                        "last_trigger_candle_open_utc": cluster_end,
                        "raw_trigger_rows": count,
                    }
                )
            cluster_start = stamp
            count = 1
        else:
            count += 1
        cluster_end = stamp
    if cluster_start is not None and cluster_end is not None:
        records.append(
            {
                "feature_candle_open_utc": cluster_start,
                "last_trigger_candle_open_utc": cluster_end,
                "raw_trigger_rows": count,
            }
        )
    return DataFrame.from_records(
        records,
        columns=(
            "feature_candle_open_utc",
            "last_trigger_candle_open_utc",
            "raw_trigger_rows",
        ),
    )


def outside_event_exclusion(
    candidate_dates: Series,
    event_dates: Series,
    *,
    hours: int = CONTROL_EXCLUSION_HOURS,
) -> Series:
    candidates = pd.DatetimeIndex(
        pd.to_datetime(candidate_dates, utc=True)
    ).as_unit("ns").asi8
    events = np.sort(
        pd.DatetimeIndex(pd.to_datetime(event_dates, utc=True)).as_unit("ns").asi8
    )
    if not len(events):
        return Series(True, index=candidate_dates.index)
    positions = np.searchsorted(events, candidates)
    distance = np.full(len(candidates), np.iinfo(np.int64).max, dtype=np.int64)
    right = positions < len(events)
    distance[right] = np.minimum(
        distance[right], np.abs(events[positions[right]] - candidates[right])
    )
    left = positions > 0
    distance[left] = np.minimum(
        distance[left], np.abs(events[positions[left] - 1] - candidates[left])
    )
    limit = pd.Timedelta(hours=hours).value
    return Series(distance > limit, index=candidate_dates.index)


def _feature_lookup(frame: DataFrame, columns: Sequence[str]) -> DataFrame:
    return frame.set_index("date")[list(columns)].apply(pd.to_numeric, errors="coerce")


def match_controls(
    frame: DataFrame,
    events: DataFrame,
    candidates: DataFrame,
    *,
    theory_id: str,
    window_id: str,
    control_kind: str,
    match_columns: Sequence[str],
) -> DataFrame:
    output_columns = (
        "theory_id",
        "window_id",
        "control_kind",
        "episode_id",
        "event_feature_candle_open_utc",
        "control_feature_candle_open_utc",
        "event_decision_anchor_utc",
        "control_decision_anchor_utc",
        "match_distance",
        "match_columns",
    )
    if events.empty or candidates.empty:
        return DataFrame(columns=output_columns)

    lookup = _feature_lookup(frame, match_columns)
    candidate_dates = pd.DatetimeIndex(candidates["feature_candle_open_utc"])
    used: set[pd.Timestamp] = set()
    records: list[dict[str, Any]] = []
    for event in events.sort_values("feature_candle_open_utc").to_dict(orient="records"):
        event_date = pd.Timestamp(event["feature_candle_open_utc"])
        pool = [
            stamp
            for stamp in candidate_dates
            if stamp not in used
            and stamp.year == event_date.year
            and stamp.month == event_date.month
        ]
        if not pool or event_date not in lookup.index:
            continue
        event_values = lookup.loc[event_date]
        candidate_values = lookup.loc[pool]
        complete = candidate_values.notna().all(axis=1)
        if event_values.isna().any() or not complete.any():
            continue
        candidate_values = candidate_values.loc[complete]
        combined = pd.concat([candidate_values, event_values.to_frame().T])
        median = combined.median(axis=0)
        scale = (combined - median).abs().median(axis=0)
        fallback = combined.std(axis=0, ddof=0)
        scale = scale.where(scale.gt(0), fallback).where(lambda value: value.gt(0), 1.0)
        distances = ((candidate_values - event_values).abs() / scale).sum(axis=1)
        time_tiebreak = np.abs(
            (candidate_values.index - event_date).total_seconds()
        ) / (31 * 24 * 60 * 60 * 10_000)
        distances = distances + time_tiebreak
        chosen = pd.Timestamp(distances.idxmin())
        used.add(chosen)
        records.append(
            {
                "theory_id": theory_id,
                "window_id": window_id,
                "control_kind": control_kind,
                "episode_id": str(event["episode_id"]),
                "event_feature_candle_open_utc": event_date,
                "control_feature_candle_open_utc": chosen,
                "event_decision_anchor_utc": event_date + pd.Timedelta(hours=1),
                "control_decision_anchor_utc": chosen + pd.Timedelta(hours=1),
                "match_distance": float(distances.loc[chosen]),
                "match_columns": ";".join(match_columns),
            }
        )
    return DataFrame.from_records(records, columns=output_columns)


def _episode_rows(
    clusters: DataFrame,
    *,
    theory: Mapping[str, Any],
    window_id: str,
    episode_kind: str,
) -> DataFrame:
    rows = clusters.copy()
    if rows.empty:
        return DataFrame(
            columns=(
                "theory_id",
                "window_id",
                "episode_kind",
                "episode_id",
                "feature_candle_open_utc",
                "decision_anchor_utc",
                "last_trigger_candle_open_utc",
                "raw_trigger_rows",
                "expected_direction",
            )
        )
    rows.insert(0, "theory_id", str(theory["theory_id"]))
    rows.insert(1, "window_id", window_id)
    rows.insert(2, "episode_kind", episode_kind)
    rows["episode_id"] = rows["feature_candle_open_utc"].map(
        lambda stamp: (
            f"{theory['theory_id']}__{window_id}__{episode_kind}__"
            f"{pd.Timestamp(stamp).strftime('%Y%m%dT%H%M%SZ')}"
        )
    )
    rows["decision_anchor_utc"] = rows["feature_candle_open_utc"] + pd.Timedelta(
        hours=1
    )
    rows["expected_direction"] = int(theory["expected_direction"])
    return rows


def build_catalogues() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    all_episodes: list[DataFrame] = []
    all_controls: list[DataFrame] = []
    all_stale: list[DataFrame] = []
    coverage_records: list[dict[str, Any]] = []

    for window_id in WINDOW_PATHS:
        frame = load_feature_frame(window_id)
        for theory in THEORIES:
            supported = window_id in theory["supported_windows"]
            if not supported:
                coverage_records.append(
                    {
                        "theory_id": theory["theory_id"],
                        "window_id": window_id,
                        "supported": False,
                        "raw_full_rows": 0,
                        "whole_episodes": 0,
                        "market_only_candidate_episodes": 0,
                        "market_only_matches": 0,
                        "news_only_candidate_episodes": 0,
                        "news_only_matches": 0,
                        "stale_whole_episodes": 0,
                        "coverage_note": "required source unavailable in this window",
                    }
                )
                continue

            source, market = build_component_masks(frame, theory)
            full_mask = source & market
            full_clusters = cluster_mask(frame["date"], full_mask)
            episodes = _episode_rows(
                full_clusters,
                theory=theory,
                window_id=window_id,
                episode_kind="full_interaction",
            )
            all_episodes.append(episodes)

            outside = outside_event_exclusion(frame["date"], frame.loc[full_mask, "date"])
            control_specs = (
                (
                    "market_only",
                    market & ~source & outside,
                    PRICE_MATCH_COLUMNS,
                ),
                (
                    "news_only",
                    source & ~market & outside,
                    tuple(theory["source_match_columns"]),
                ),
            )
            matched_counts: dict[str, int] = {}
            candidate_counts: dict[str, int] = {}
            for kind, candidate_mask, columns in control_specs:
                candidate_clusters = cluster_mask(frame["date"], candidate_mask)
                candidate_counts[kind] = len(candidate_clusters)
                matches = match_controls(
                    frame,
                    episodes,
                    candidate_clusters,
                    theory_id=str(theory["theory_id"]),
                    window_id=window_id,
                    control_kind=kind,
                    match_columns=columns,
                )
                matched_counts[kind] = len(matches)
                all_controls.append(matches)

            stale_source, _ = build_component_masks(
                frame, theory, source_lag_hours=SOURCE_STALE_HOURS
            )
            stale_mask = stale_source & market & ~source
            stale_clusters = cluster_mask(frame["date"], stale_mask)
            stale_rows = _episode_rows(
                stale_clusters,
                theory=theory,
                window_id=window_id,
                episode_kind="stale_source_168h",
            )
            all_stale.append(stale_rows)

            coverage_records.append(
                {
                    "theory_id": theory["theory_id"],
                    "window_id": window_id,
                    "supported": True,
                    "raw_full_rows": int(full_mask.sum()),
                    "whole_episodes": len(episodes),
                    "market_only_candidate_episodes": candidate_counts["market_only"],
                    "market_only_matches": matched_counts["market_only"],
                    "news_only_candidate_episodes": candidate_counts["news_only"],
                    "news_only_matches": matched_counts["news_only"],
                    "stale_whole_episodes": len(stale_rows),
                    "coverage_note": (
                        "eligible"
                        if len(episodes) >= MINIMUM_EPISODES_PER_WINDOW
                        else "fewer than frozen minimum whole episodes"
                    ),
                }
            )

    episodes = pd.concat(all_episodes, ignore_index=True) if all_episodes else DataFrame()
    controls = pd.concat(all_controls, ignore_index=True) if all_controls else DataFrame()
    stale = pd.concat(all_stale, ignore_index=True) if all_stale else DataFrame()
    coverage = DataFrame.from_records(coverage_records)
    return episodes, controls, stale, coverage


def build_freeze() -> dict[str, Any]:
    missing = [path for path in (*WINDOW_PATHS.values(), OLD_RESULT_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen historical inputs: {missing}")
    return {
        "schema_version": 1,
        "status": "frozen_historical_news_interactions_before_outcomes",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "plain_objective": (
            "Retest all seven saved historical news-plus-market ideas together using "
            "whole episodes and component controls before following any branch."
        ),
        "scope": {
            "retrospective_selected_leads_only": True,
            "independent_confirmation": False,
            "profit_used": False,
            "freqai_used": False,
            "all_seven_completed_before_branching": True,
            "primary_outcome_horizon_hours": 24,
        },
        "timestamp_contract": {
            "feature_timestamp_meaning": "opening time of the completed one-hour candle",
            "decision_anchor": "feature timestamp plus one hour",
            "outcome_start": "first subsequent one-hour candle",
            "current_hour_source_and_market_values_known_only_at_decision_anchor": True,
        },
        "episode_contract": {
            "whole_episode_gap_hours": EPISODE_GAP_HOURS,
            "selection": "first trigger in each chain separated by no gap above 24 hours",
            "control_exclusion_hours_from_any_raw_full_trigger": CONTROL_EXCLUSION_HOURS,
            "one_control_per_event_without_reuse": True,
            "control_calendar_rule": "same calendar year and month",
            "market_only_match_columns": list(PRICE_MATCH_COLUMNS),
            "stale_source_hours": SOURCE_STALE_HOURS,
        },
        "outcome_contract": {
            "reaction_amount": (
                "larger of maximum upward excursion and absolute maximum downward "
                "excursion during the next 24 completed one-hour candles"
            ),
            "reaction_threshold": "median reaction amount across the same source window",
            "direction_success": (
                "expected-side maximum excursion exceeds opposite-side maximum excursion"
            ),
            "joint_success": "above-normal reaction and direction success",
            "ties": "abstain",
            "old_fixed_threshold_labels": "secondary description only",
        },
        "decision_rules": {
            "minimum_episodes_per_window": MINIMUM_EPISODES_PER_WINDOW,
            "minimum_supported_windows": MINIMUM_SUPPORTED_WINDOWS,
            "minimum_joint_rate": MINIMUM_JOINT_RATE,
            "main_joint_target": MAIN_JOINT_TARGET,
            "minimum_market_only_lift": MINIMUM_MARKET_ONLY_LIFT,
            "minimum_news_only_lift": MINIMUM_NEWS_ONLY_LIFT,
            "minimum_stale_lift": MINIMUM_STALE_LIFT,
            "maximum_opposite_window_lift": MAX_OPPOSITE_WINDOW_LIFT,
            "familywise_probability_limit": 0.05,
            "permutation_iterations": PERMUTATION_ITERATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "historical_only_rule": (
                "Even a retained result is a selected historical candidate and requires "
                "new whole-event confirmation before use."
            ),
        },
        "theories": [dict(value) for value in THEORIES],
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "old_selected_result_table": artifact(OLD_RESULT_PATH),
            "feature_frames": {
                window_id: artifact(path) for window_id, path in WINDOW_PATHS.items()
            },
        },
    }


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        existing = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_historical_news_interaction_freeze":
            raise ValueError("Existing freeze result is not terminal.")
        return existing

    freeze_record = build_freeze()
    episodes, controls, stale, coverage = build_catalogues()
    g0.atomic_write_csv(episodes, EPISODES_PATH)
    g0.atomic_write_csv(controls, CONTROLS_PATH)
    g0.atomic_write_csv(stale, STALE_EPISODES_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_json(freeze_record, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_historical_news_interaction_freeze",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "profit_used": False,
        "future_outcomes_read": False,
        "counts": {
            "theories": len(THEORIES),
            "full_whole_episodes": len(episodes),
            "matched_component_controls": len(controls),
            "stale_whole_episodes": len(stale),
        },
        "artifacts": {
            "freeze": artifact(FREEZE_PATH),
            "episodes": artifact(EPISODES_PATH),
            "controls": artifact(CONTROLS_PATH),
            "stale_episodes": artifact(STALE_EPISODES_PATH),
            "coverage": artifact(COVERAGE_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(freeze(overwrite=args.overwrite), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
