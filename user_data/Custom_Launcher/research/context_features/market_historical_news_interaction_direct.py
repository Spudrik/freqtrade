"""Directly retest the frozen historical news/market interaction families.

This phase reads future price paths only after the seven theories, timestamps,
whole-episode rules, controls, and decision gates have been frozen.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_historical_news_interaction_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260911a"
OUTCOME_ROWS_PATH = OUTPUT_ROOT / "historical_news_interaction_outcomes.parquet"
EPISODE_SUMMARY_PATH = OUTPUT_ROOT / "historical_news_interaction_episode_summary.csv"
PAIRED_ROWS_PATH = OUTPUT_ROOT / "historical_news_interaction_paired_controls.parquet"
PAIRED_SUMMARY_PATH = OUTPUT_ROOT / "historical_news_interaction_paired_summary.csv"
NULL_PATH = OUTPUT_ROOT / "historical_news_interaction_familywise_null.csv"
DECISIONS_PATH = OUTPUT_ROOT / "historical_news_interaction_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "historical_news_interaction_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "historical_news_interaction_result.json"

OUTCOME_COLUMNS = (
    "date",
    "future_return_24h",
    "future_max_upside_24h",
    "future_max_drawdown_24h",
    "up_2pct_next_24h",
    "down_2pct_next_24h",
    "large_upside_next_24h",
    "large_drawdown_next_24h",
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_artifact(contract: Mapping[str, Any]) -> None:
    path = Path(str(contract["path"]))
    if not path.is_file() or g0.sha256_file(path) != str(contract["sha256"]):
        raise ValueError(f"Frozen artifact changed or is missing: {path}")


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame, DataFrame]:
    result = _load_json(frozen.RESULT_PATH)
    if result.get("status") != "completed_historical_news_interaction_freeze":
        raise ValueError("Historical interaction freeze is not terminal.")
    if result.get("profit_used") or result.get("future_outcomes_read"):
        raise ValueError("Freeze phase violated its no-outcome contract.")
    for contract in result["artifacts"].values():
        _verify_artifact(contract)

    freeze_record = _load_json(frozen.FREEZE_PATH)
    if freeze_record.get("status") != "frozen_historical_news_interactions_before_outcomes":
        raise ValueError("Historical interaction design is not frozen.")
    if freeze_record["scope"].get("profit_used"):
        raise ValueError("Profit unexpectedly appears in the frozen design.")
    for contract in freeze_record["source_contracts"]["feature_frames"].values():
        _verify_artifact(contract)
    _verify_artifact(freeze_record["source_contracts"]["old_selected_result_table"])

    episodes = pd.read_csv(frozen.EPISODES_PATH)
    controls = pd.read_csv(frozen.CONTROLS_PATH)
    stale = pd.read_csv(frozen.STALE_EPISODES_PATH)
    coverage = pd.read_csv(frozen.COVERAGE_PATH)
    for frame, columns in (
        (
            episodes,
            (
                "feature_candle_open_utc",
                "decision_anchor_utc",
                "last_trigger_candle_open_utc",
            ),
        ),
        (
            controls,
            (
                "event_feature_candle_open_utc",
                "control_feature_candle_open_utc",
                "event_decision_anchor_utc",
                "control_decision_anchor_utc",
            ),
        ),
        (
            stale,
            (
                "feature_candle_open_utc",
                "decision_anchor_utc",
                "last_trigger_candle_open_utc",
            ),
        ),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    coverage["supported"] = frozen._strict_bool(coverage["supported"])
    return freeze_record, episodes, controls, stale, coverage


def load_outcomes() -> tuple[dict[str, DataFrame], dict[str, float]]:
    frames: dict[str, DataFrame] = {}
    thresholds: dict[str, float] = {}
    for window_id, path in frozen.WINDOW_PATHS.items():
        frame = pd.read_parquet(path, columns=list(OUTCOME_COLUMNS))
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError(f"Invalid outcome timestamp index in {path}")
        upside = pd.to_numeric(frame["future_max_upside_24h"], errors="coerce")
        downside = -pd.to_numeric(frame["future_max_drawdown_24h"], errors="coerce")
        frame["reaction_amount"] = pd.concat([upside, downside], axis=1).max(axis=1)
        complete = upside.notna() & downside.notna()
        threshold = float(frame.loc[complete, "reaction_amount"].median())
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid reaction threshold for {window_id}")
        thresholds[window_id] = threshold
        indexed = frame.set_index("date")
        if not indexed.index.is_unique:
            raise ValueError(f"Duplicate outcome timestamp in {path}")
        frames[window_id] = indexed
    return frames, thresholds


def label_outcomes(
    rows: DataFrame,
    outcomes: Mapping[str, DataFrame],
    thresholds: Mapping[str, float],
    *,
    date_column: str,
    sample_kind: str,
) -> DataFrame:
    records: list[dict[str, Any]] = []
    for window_id, group in rows.groupby("window_id", sort=False):
        source = outcomes[str(window_id)]
        dates = pd.DatetimeIndex(group[date_column])
        selected = source.reindex(dates)
        for metadata, (_, outcome) in zip(
            group.to_dict(orient="records"), selected.iterrows(), strict=True
        ):
            expected = int(metadata["expected_direction"])
            upside = pd.to_numeric(outcome["future_max_upside_24h"], errors="coerce")
            downside = -pd.to_numeric(
                outcome["future_max_drawdown_24h"], errors="coerce"
            )
            complete = bool(pd.notna(upside) and pd.notna(downside))
            direction_tie = bool(
                complete and np.isclose(float(upside), float(downside), atol=1e-12)
            )
            if not complete or direction_tie:
                direction_success: bool | float = np.nan
            elif expected > 0:
                direction_success = bool(upside > downside)
            else:
                direction_success = bool(downside > upside)
            reaction_amount = max(float(upside), float(downside)) if complete else np.nan
            reaction_success = (
                bool(reaction_amount > thresholds[str(window_id)])
                if complete
                else np.nan
            )
            joint_success = (
                bool(reaction_success and direction_success)
                if pd.notna(direction_success) and pd.notna(reaction_success)
                else np.nan
            )
            fixed_threshold = (
                outcome["up_2pct_next_24h"]
                if expected > 0
                else outcome["down_2pct_next_24h"]
            )
            records.append(
                {
                    **metadata,
                    "sample_kind": sample_kind,
                    "outcome_complete": complete,
                    "direction_tie_abstention": direction_tie,
                    "reaction_threshold": thresholds[str(window_id)],
                    "reaction_amount": reaction_amount,
                    "reaction_success": reaction_success,
                    "direction_success": direction_success,
                    "joint_success": joint_success,
                    "expected_side_excursion": (
                        float(upside) if expected > 0 else float(downside)
                    )
                    if complete
                    else np.nan,
                    "opposite_side_excursion": (
                        float(downside) if expected > 0 else float(upside)
                    )
                    if complete
                    else np.nan,
                    "future_return_24h": outcome["future_return_24h"],
                    "old_expected_2pct_hit": fixed_threshold,
                    "large_upside_next_24h": outcome["large_upside_next_24h"],
                    "large_drawdown_next_24h": outcome["large_drawdown_next_24h"],
                }
            )
    return DataFrame.from_records(records)


def outcome_rows(
    episodes: DataFrame,
    stale: DataFrame,
    outcomes: Mapping[str, DataFrame],
    thresholds: Mapping[str, float],
) -> DataFrame:
    full = label_outcomes(
        episodes,
        outcomes,
        thresholds,
        date_column="feature_candle_open_utc",
        sample_kind="full_interaction",
    )
    stale_rows = label_outcomes(
        stale,
        outcomes,
        thresholds,
        date_column="feature_candle_open_utc",
        sample_kind="stale_source_168h",
    )
    return pd.concat([full, stale_rows], ignore_index=True)


def paired_control_rows(
    controls: DataFrame,
    outcomes: Mapping[str, DataFrame],
    thresholds: Mapping[str, float],
    expected_directions: Mapping[str, int],
) -> DataFrame:
    base = controls.copy()
    base["expected_direction"] = base["theory_id"].map(expected_directions)
    event_labels = label_outcomes(
        base,
        outcomes,
        thresholds,
        date_column="event_feature_candle_open_utc",
        sample_kind="paired_event",
    )
    control_labels = label_outcomes(
        base,
        outcomes,
        thresholds,
        date_column="control_feature_candle_open_utc",
        sample_kind="paired_control",
    )
    identifiers = (
        "theory_id",
        "window_id",
        "control_kind",
        "episode_id",
        "event_feature_candle_open_utc",
        "control_feature_candle_open_utc",
        "match_distance",
        "expected_direction",
    )
    measures = (
        "outcome_complete",
        "direction_tie_abstention",
        "reaction_amount",
        "reaction_success",
        "direction_success",
        "joint_success",
        "expected_side_excursion",
        "opposite_side_excursion",
        "future_return_24h",
        "old_expected_2pct_hit",
    )
    result = event_labels[list(identifiers)].copy()
    for column in measures:
        result[f"event_{column}"] = event_labels[column].to_numpy()
        result[f"control_{column}"] = control_labels[column].to_numpy()
    for measure in ("reaction_success", "direction_success", "joint_success"):
        result[f"{measure}_difference"] = (
            pd.to_numeric(result[f"event_{measure}"], errors="coerce")
            - pd.to_numeric(result[f"control_{measure}"], errors="coerce")
        )
    return result


def _rate(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def summarize_episodes(rows: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["theory_id", "window_id", "sample_kind"], sort=False, dropna=False
    ):
        theory_id, window_id, sample_kind = keys
        records.append(
            {
                "theory_id": theory_id,
                "window_id": window_id,
                "sample_kind": sample_kind,
                "whole_episode_count": int(group["episode_id"].nunique()),
                "complete_outcome_count": int(group["outcome_complete"].sum()),
                "abstention_count": int(group["direction_tie_abstention"].sum()),
                "above_normal_reaction_rate": _rate(group["reaction_success"]),
                "expected_direction_dominance_rate": _rate(group["direction_success"]),
                "joint_reaction_and_direction_rate": _rate(group["joint_success"]),
                "old_expected_2pct_hit_rate": _rate(group["old_expected_2pct_hit"]),
                "median_reaction_amount": float(
                    pd.to_numeric(group["reaction_amount"], errors="coerce").median()
                ),
                "median_expected_side_excursion": float(
                    pd.to_numeric(
                        group["expected_side_excursion"], errors="coerce"
                    ).median()
                ),
                "median_opposite_side_excursion": float(
                    pd.to_numeric(
                        group["opposite_side_excursion"], errors="coerce"
                    ).median()
                ),
            }
        )
    return DataFrame.from_records(records)


def summarize_pairs(rows: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["theory_id", "window_id", "control_kind"], sort=False
    ):
        theory_id, window_id, control_kind = keys
        complete = group.loc[
            group["event_joint_success"].notna()
            & group["control_joint_success"].notna()
        ]
        records.append(
            {
                "theory_id": theory_id,
                "window_id": window_id,
                "control_kind": control_kind,
                "matched_pair_count": len(group),
                "complete_pair_count": len(complete),
                "event_reaction_rate": _rate(group["event_reaction_success"]),
                "control_reaction_rate": _rate(group["control_reaction_success"]),
                "reaction_rate_lift": _rate(group["reaction_success_difference"]),
                "event_direction_rate": _rate(group["event_direction_success"]),
                "control_direction_rate": _rate(group["control_direction_success"]),
                "direction_rate_lift": _rate(group["direction_success_difference"]),
                "event_joint_rate": _rate(group["event_joint_success"]),
                "control_joint_rate": _rate(group["control_joint_success"]),
                "joint_rate_lift": _rate(group["joint_success_difference"]),
                "median_match_distance": float(group["match_distance"].median()),
            }
        )
    return DataFrame.from_records(records)


def _records(frame: DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def paired_studentized_statistic(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return np.nan
    mean = float(finite.mean())
    standard_deviation = float(finite.std(ddof=1))
    if standard_deviation == 0:
        if mean > 0:
            return np.inf
        if mean < 0:
            return -np.inf
        return 0.0
    return mean / (standard_deviation / np.sqrt(len(finite)))


def familywise_null(rows: DataFrame) -> DataFrame:
    rng = np.random.default_rng(frozen.PERMUTATION_SEED)
    records: list[dict[str, Any]] = []
    for control_kind, raw_kind_rows in rows.groupby("control_kind", sort=False):
        for outcome_measure in ("reaction_success", "direction_success", "joint_success"):
            event_column = f"event_{outcome_measure}"
            control_column = f"control_{outcome_measure}"
            kind_rows = raw_kind_rows.loc[
                raw_kind_rows[event_column].notna()
                & raw_kind_rows[control_column].notna()
            ].copy()
            counts = kind_rows.groupby("theory_id")["episode_id"].transform("nunique")
            kind_rows = kind_rows.loc[
                counts.ge(frozen.MINIMUM_EPISODES_PER_WINDOW)
            ].copy()
            kind_rows["event_value"] = pd.to_numeric(
                kind_rows[event_column], errors="raise"
            ).astype(float)
            kind_rows["control_value"] = pd.to_numeric(
                kind_rows[control_column], errors="raise"
            ).astype(float)
            kind_rows["swap_unit"] = (
                kind_rows["window_id"].astype(str)
                + "__"
                + kind_rows["event_feature_candle_open_utc"].astype(str)
            )
            units = kind_rows["swap_unit"].unique()
            unit_codes = pd.Categorical(
                kind_rows["swap_unit"], categories=units
            ).codes
            observed_difference = (
                kind_rows["event_value"] - kind_rows["control_value"]
            ).to_numpy(dtype=float)
            theories = kind_rows["theory_id"].astype(str).to_numpy()
            for iteration in range(frozen.PERMUTATION_ITERATIONS):
                swap = rng.integers(0, 2, size=len(units), dtype=np.int8)
                signs = np.where(swap[unit_codes].astype(bool), -1.0, 1.0)
                shuffled = observed_difference * signs
                maximum = max(
                    paired_studentized_statistic(shuffled[theories == theory_id])
                    for theory_id in np.unique(theories)
                )
                records.append(
                    {
                        "control_kind": control_kind,
                        "outcome_measure": outcome_measure,
                        "iteration": iteration,
                        "maximum_null_studentized_statistic": maximum,
                    }
                )
    return DataFrame.from_records(records)


def _finite_mean(values: Iterable[Any]) -> float:
    numeric = pd.to_numeric(Series(list(values)), errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _familywise_probability(
    null: DataFrame,
    *,
    control_kind: str,
    outcome_measure: str,
    observed_statistic: float,
) -> float:
    if not np.isfinite(observed_statistic):
        return np.nan
    values = null.loc[
        null["control_kind"].eq(control_kind)
        & null["outcome_measure"].eq(outcome_measure),
        "maximum_null_studentized_statistic",
    ]
    if values.empty:
        return np.nan
    return float(
        (values.ge(observed_statistic).sum() + 1) / (len(values) + 1)
    )


def classify_primary_result(
    *,
    market_ok: bool,
    news_ok: bool,
    stale_ok: bool,
    repeat_ok: bool,
    no_opposite: bool,
    enough_market_windows: bool,
    market_lift: float,
) -> tuple[str, str]:
    if market_ok and news_ok and stale_ok and repeat_ok and no_opposite:
        return (
            "retained_historical_interaction_candidate",
            "The combined news and market condition beat both components and the "
            "week-old news timing, but still needs genuinely new events.",
        )
    if market_ok and repeat_ok and no_opposite:
        return (
            "promising_but_component_control_limited",
            "The news condition added to the matched market setup, but another "
            "component or stale-timing check lacked coverage or strength.",
        )
    if enough_market_windows and market_lift < 0.03:
        return (
            "market_condition_explains_most_of_old_result",
            "Once similar market setups were compared, the current news condition "
            "added little in this exact historical representation.",
        )
    if not no_opposite:
        return (
            "inconsistent_across_historical_windows",
            "The combination helped in one historical setting and hurt in another, "
            "so a missing background condition is likely.",
        )
    if not enough_market_windows:
        return (
            "coverage_limited",
            "There were not enough whole episodes and matched market comparisons "
            "in enough historical windows to judge the idea.",
        )
    return (
        "not_supported_in_whole_episode_retest",
        "The exact combination did not repeat strongly enough after overlapping "
        "hours and matched market conditions were controlled.",
    )


def classify_layer_result(
    *,
    direction_role_ok: bool,
    reaction_role_ok: bool,
    market_direction_lift: float,
    market_direction_probability: float,
    news_direction_probability: float,
    market_direction_event_rate: float,
    joint_market_probability: float,
    joint_news_probability: float,
    joint_market_lift: float,
    joint_news_lift: float,
    primary_decision: str,
    probability_limit: float,
    minimum_market_lift: float,
    primary_plain: str,
) -> tuple[str, str]:
    if direction_role_ok and reaction_role_ok:
        return (
            "historical_reaction_and_direction_modifier_lead",
            "The combination separately improved both reaction likelihood and "
            "direction, although it did not meet the stricter same-episode joint gate.",
        )
    if direction_role_ok:
        return (
            "historical_direction_modifier_lead",
            "The combination gave repeatable directional information beyond both "
            "components and week-old source timing; reaction size remains separate.",
        )
    if reaction_role_ok:
        return (
            "historical_reaction_modifier_lead",
            "The combination helped identify unusually large reactions, but did "
            "not provide dependable direction by itself.",
        )
    if (
        market_direction_event_rate >= 0.55
        and market_direction_lift >= minimum_market_lift
        and market_direction_probability <= probability_limit
        and news_direction_probability <= probability_limit
    ):
        return (
            "strong_direction_modifier_with_period_limit",
            "Direction improved strongly overall versus both components, but at least "
            "one historical component comparison was slightly contradictory.",
        )
    if (
        market_direction_lift >= minimum_market_lift
        and market_direction_probability <= probability_limit
    ):
        return (
            "direction_increment_below_complete_gate",
            "Direction improved versus the matched market setup, but absolute "
            "accuracy, another component, stale timing, or period consistency failed.",
        )
    if (
        joint_market_probability <= probability_limit
        and joint_news_probability <= probability_limit
        and joint_market_lift > 0
        and joint_news_lift > 0
    ):
        return (
            "joint_increment_missing_stable_background",
            "The joint outcome improved overall, but it did not remain coherent "
            "across time or current-versus-stale source timing.",
        )
    if primary_decision == "coverage_limited":
        return "coverage_limited", primary_plain
    return (
        "no_repeatable_incremental_role_found",
        "This exact broad hourly combination did not add a repeatable reaction "
        "or direction role under the available component controls.",
    )


def decide_theories(
    freeze_record: Mapping[str, Any],
    episode_summary: DataFrame,
    pair_summary: DataFrame,
    pairs: DataFrame,
    null: DataFrame,
) -> DataFrame:
    rules = freeze_record["decision_rules"]
    records: list[dict[str, Any]] = []
    for theory in freeze_record["theories"]:
        theory_id = str(theory["theory_id"])
        supported = tuple(theory["supported_windows"])
        full = episode_summary.loc[
            episode_summary["theory_id"].eq(theory_id)
            & episode_summary["sample_kind"].eq("full_interaction")
            & episode_summary["window_id"].isin(supported)
        ].set_index("window_id")
        stale = episode_summary.loc[
            episode_summary["theory_id"].eq(theory_id)
            & episode_summary["sample_kind"].eq("stale_source_168h")
            & episode_summary["window_id"].isin(supported)
        ].set_index("window_id")
        market = pair_summary.loc[
            pair_summary["theory_id"].eq(theory_id)
            & pair_summary["control_kind"].eq("market_only")
            & pair_summary["window_id"].isin(supported)
        ].set_index("window_id")
        news = pair_summary.loc[
            pair_summary["theory_id"].eq(theory_id)
            & pair_summary["control_kind"].eq("news_only")
            & pair_summary["window_id"].isin(supported)
        ].set_index("window_id")

        minimum = int(rules["minimum_episodes_per_window"])
        eligible_market_windows = [
            window
            for window in supported
            if window in full.index
            and window in market.index
            and int(full.loc[window, "complete_outcome_count"]) >= minimum
            and int(market.loc[window, "complete_pair_count"]) >= minimum
        ]
        eligible_news_windows = [
            window
            for window in supported
            if window in full.index
            and window in news.index
            and int(full.loc[window, "complete_outcome_count"]) >= minimum
            and int(news.loc[window, "complete_pair_count"]) >= minimum
        ]
        eligible_stale_windows = [
            window
            for window in supported
            if window in full.index
            and window in stale.index
            and int(full.loc[window, "complete_outcome_count"]) >= minimum
            and int(stale.loc[window, "complete_outcome_count"]) >= minimum
        ]

        theory_pairs = pairs.loc[pairs["theory_id"].eq(theory_id)]
        market_pairs = theory_pairs.loc[
            theory_pairs["control_kind"].eq("market_only")
            & theory_pairs["window_id"].isin(eligible_market_windows)
        ]
        news_pairs = theory_pairs.loc[
            theory_pairs["control_kind"].eq("news_only")
            & theory_pairs["window_id"].isin(eligible_news_windows)
        ]
        market_measure: dict[str, dict[str, float]] = {}
        news_measure: dict[str, dict[str, float]] = {}
        summary_columns = {
            "reaction_success": (
                "event_reaction_rate",
                "reaction_rate_lift",
                "above_normal_reaction_rate",
            ),
            "direction_success": (
                "event_direction_rate",
                "direction_rate_lift",
                "expected_direction_dominance_rate",
            ),
            "joint_success": (
                "event_joint_rate",
                "joint_rate_lift",
                "joint_reaction_and_direction_rate",
            ),
        }
        stale_measure_lifts: dict[str, float] = {}
        for measure, (_, _, episode_column) in summary_columns.items():
            for control_name, selected_pairs, target in (
                ("market_only", market_pairs, market_measure),
                ("news_only", news_pairs, news_measure),
            ):
                differences = pd.to_numeric(
                    selected_pairs[f"{measure}_difference"], errors="coerce"
                ).dropna()
                statistic = paired_studentized_statistic(
                    differences.to_numpy(dtype=float)
                )
                target[measure] = {
                    "event_rate": _rate(selected_pairs[f"event_{measure}"]),
                    "lift": _rate(differences),
                    "statistic": statistic,
                    "probability": _familywise_probability(
                        null,
                        control_kind=control_name,
                        outcome_measure=measure,
                        observed_statistic=statistic,
                    ),
                }
            stale_measure_lifts[measure] = _finite_mean(
                [
                    float(full.loc[window, episode_column])
                    - float(stale.loc[window, episode_column])
                    for window in eligible_stale_windows
                ]
            )

        market_lift = market_measure["joint_success"]["lift"]
        news_lift = news_measure["joint_success"]["lift"]
        market_event_rate = market_measure["joint_success"]["event_rate"]
        stale_lift = stale_measure_lifts["joint_success"]
        repeat_windows = sum(
            float(market.loc[window, "event_joint_rate"])
            >= float(rules["minimum_joint_rate"])
            and float(market.loc[window, "joint_rate_lift"])
            >= float(rules["minimum_market_only_lift"])
            for window in eligible_market_windows
        )
        opposite_windows = sum(
            float(market.loc[window, "joint_rate_lift"])
            < float(rules["maximum_opposite_window_lift"])
            for window in eligible_market_windows
        )
        market_probability = market_measure["joint_success"]["probability"]
        news_probability = news_measure["joint_success"]["probability"]

        enough_market_windows = len(eligible_market_windows) >= int(
            rules["minimum_supported_windows"]
        )
        enough_news_windows = len(eligible_news_windows) >= int(
            rules["minimum_supported_windows"]
        )
        enough_stale_windows = len(eligible_stale_windows) >= int(
            rules["minimum_supported_windows"]
        )
        repeat_ok = repeat_windows >= int(rules["minimum_supported_windows"])
        no_opposite = opposite_windows == 0
        market_ok = bool(
            enough_market_windows
            and np.isfinite(market_event_rate)
            and market_event_rate >= float(rules["minimum_joint_rate"])
            and market_lift >= float(rules["minimum_market_only_lift"])
            and market_probability <= float(rules["familywise_probability_limit"])
        )
        news_ok = bool(
            enough_news_windows
            and news_lift >= float(rules["minimum_news_only_lift"])
            and news_probability <= float(rules["familywise_probability_limit"])
        )
        stale_ok = bool(
            enough_stale_windows
            and stale_lift >= float(rules["minimum_stale_lift"])
        )

        direction_repeat_windows = sum(
            float(market.loc[window, "event_direction_rate"])
            >= float(rules["minimum_joint_rate"])
            and float(market.loc[window, "direction_rate_lift"])
            >= float(rules["minimum_market_only_lift"])
            for window in eligible_market_windows
        )
        direction_opposite_windows = sum(
            float(market.loc[window, "direction_rate_lift"])
            < float(rules["maximum_opposite_window_lift"])
            for window in eligible_market_windows
        )
        direction_news_repeat_windows = sum(
            float(news.loc[window, "event_direction_rate"])
            >= float(rules["minimum_joint_rate"])
            and float(news.loc[window, "direction_rate_lift"])
            >= float(rules["minimum_news_only_lift"])
            for window in eligible_news_windows
        )
        direction_news_opposite_windows = sum(
            float(news.loc[window, "direction_rate_lift"])
            < float(rules["maximum_opposite_window_lift"])
            for window in eligible_news_windows
        )
        reaction_repeat_windows = sum(
            float(market.loc[window, "event_reaction_rate"])
            >= float(rules["minimum_joint_rate"])
            and float(market.loc[window, "reaction_rate_lift"])
            >= float(rules["minimum_market_only_lift"])
            for window in eligible_market_windows
        )
        reaction_opposite_windows = sum(
            float(market.loc[window, "reaction_rate_lift"])
            < float(rules["maximum_opposite_window_lift"])
            for window in eligible_market_windows
        )
        direction_role_ok = bool(
            enough_market_windows
            and enough_news_windows
            and enough_stale_windows
            and market_measure["direction_success"]["event_rate"]
            >= float(rules["minimum_joint_rate"])
            and market_measure["direction_success"]["lift"]
            >= float(rules["minimum_market_only_lift"])
            and news_measure["direction_success"]["lift"]
            >= float(rules["minimum_news_only_lift"])
            and stale_measure_lifts["direction_success"]
            >= float(rules["minimum_stale_lift"])
            and market_measure["direction_success"]["probability"]
            <= float(rules["familywise_probability_limit"])
            and news_measure["direction_success"]["probability"]
            <= float(rules["familywise_probability_limit"])
            and direction_repeat_windows >= int(rules["minimum_supported_windows"])
            and direction_news_repeat_windows
            >= int(rules["minimum_supported_windows"])
            and direction_opposite_windows == 0
            and direction_news_opposite_windows == 0
        )
        reaction_role_ok = bool(
            enough_market_windows
            and enough_news_windows
            and enough_stale_windows
            and market_measure["reaction_success"]["event_rate"]
            >= float(rules["minimum_joint_rate"])
            and market_measure["reaction_success"]["lift"]
            >= float(rules["minimum_market_only_lift"])
            and news_measure["reaction_success"]["lift"]
            >= float(rules["minimum_news_only_lift"])
            and stale_measure_lifts["reaction_success"]
            >= float(rules["minimum_stale_lift"])
            and market_measure["reaction_success"]["probability"]
            <= float(rules["familywise_probability_limit"])
            and news_measure["reaction_success"]["probability"]
            <= float(rules["familywise_probability_limit"])
            and reaction_repeat_windows >= int(rules["minimum_supported_windows"])
            and reaction_opposite_windows == 0
        )

        decision, plain = classify_primary_result(
            market_ok=market_ok,
            news_ok=news_ok,
            stale_ok=stale_ok,
            repeat_ok=repeat_ok,
            no_opposite=no_opposite,
            enough_market_windows=enough_market_windows,
            market_lift=market_lift,
        )
        layer_decision, layer_plain = classify_layer_result(
            direction_role_ok=direction_role_ok,
            reaction_role_ok=reaction_role_ok,
            market_direction_lift=market_measure["direction_success"]["lift"],
            market_direction_probability=market_measure["direction_success"][
                "probability"
            ],
            news_direction_probability=news_measure["direction_success"][
                "probability"
            ],
            market_direction_event_rate=market_measure["direction_success"][
                "event_rate"
            ],
            joint_market_probability=market_probability,
            joint_news_probability=news_probability,
            joint_market_lift=market_lift,
            joint_news_lift=news_lift,
            primary_decision=decision,
            probability_limit=float(rules["familywise_probability_limit"]),
            minimum_market_lift=float(rules["minimum_market_only_lift"]),
            primary_plain=plain,
        )

        records.append(
            {
                "theory_id": theory_id,
                "plain_question": theory["plain_question"],
                "expected_direction": int(theory["expected_direction"]),
                "eligible_market_windows": len(eligible_market_windows),
                "eligible_news_windows": len(eligible_news_windows),
                "eligible_stale_windows": len(eligible_stale_windows),
                "market_matched_pairs": len(market_pairs),
                "news_matched_pairs": len(news_pairs),
                "market_matched_event_joint_rate": market_event_rate,
                "joint_lift_vs_market_only": market_lift,
                "joint_lift_vs_news_only": news_lift,
                "mean_joint_lift_vs_stale_source": stale_lift,
                "market_matched_event_reaction_rate": market_measure[
                    "reaction_success"
                ]["event_rate"],
                "reaction_lift_vs_market_only": market_measure[
                    "reaction_success"
                ]["lift"],
                "reaction_lift_vs_news_only": news_measure["reaction_success"][
                    "lift"
                ],
                "mean_reaction_lift_vs_stale_source": stale_measure_lifts[
                    "reaction_success"
                ],
                "market_matched_event_direction_rate": market_measure[
                    "direction_success"
                ]["event_rate"],
                "direction_lift_vs_market_only": market_measure[
                    "direction_success"
                ]["lift"],
                "direction_lift_vs_news_only": news_measure["direction_success"][
                    "lift"
                ],
                "mean_direction_lift_vs_stale_source": stale_measure_lifts[
                    "direction_success"
                ],
                "repeat_windows_at_frozen_gate": repeat_windows,
                "material_opposite_windows": opposite_windows,
                "familywise_probability_vs_market_only": market_probability,
                "familywise_probability_vs_news_only": news_probability,
                "familywise_reaction_probability_vs_market_only": market_measure[
                    "reaction_success"
                ]["probability"],
                "familywise_reaction_probability_vs_news_only": news_measure[
                    "reaction_success"
                ]["probability"],
                "familywise_direction_probability_vs_market_only": market_measure[
                    "direction_success"
                ]["probability"],
                "familywise_direction_probability_vs_news_only": news_measure[
                    "direction_success"
                ]["probability"],
                "direction_repeat_windows": direction_repeat_windows,
                "direction_material_opposite_windows": direction_opposite_windows,
                "direction_news_repeat_windows": direction_news_repeat_windows,
                "direction_news_material_opposite_windows": (
                    direction_news_opposite_windows
                ),
                "reaction_repeat_windows": reaction_repeat_windows,
                "reaction_material_opposite_windows": reaction_opposite_windows,
                "reaches_main_65pct_target": bool(
                    np.isfinite(market_event_rate)
                    and market_event_rate >= float(rules["main_joint_target"])
                ),
                "decision": decision,
                "plain_interpretation": plain,
                "layer_decision": layer_decision,
                "layer_plain_interpretation": layer_plain,
                "layer_review_is_posthoc": True,
                "independent_confirmation": False,
            }
        )
    return DataFrame.from_records(records)


def _pct(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "not available" if pd.isna(numeric) else f"{float(numeric) * 100:.1f}%"


def render_report(
    freeze_record: Mapping[str, Any],
    thresholds: Mapping[str, float],
    coverage: DataFrame,
    episode_summary: DataFrame,
    pair_summary: DataFrame,
    decisions: DataFrame,
) -> str:
    lines = [
        "# Historical News And Market Interaction Retest",
        "",
        "## Short answer",
        "",
    ]
    retained = decisions.loc[
        decisions["decision"].eq("retained_historical_interaction_candidate")
    ]
    promising = decisions.loc[
        decisions["decision"].eq("promising_but_component_control_limited")
    ]
    layer_leads = decisions.loc[
        decisions["layer_decision"].isin(
            [
                "historical_reaction_and_direction_modifier_lead",
                "historical_direction_modifier_lead",
                "historical_reaction_modifier_lead",
                "strong_direction_modifier_with_period_limit",
            ]
        )
    ]
    if len(retained):
        lines.append(
            f"{len(retained)} of seven combinations survived every frozen historical "
            "interaction check. They remain historical leads, not trading rules."
        )
    elif len(promising):
        lines.append(
            f"None passed every interaction check; {len(promising)} remained promising "
            "but lacked a complete component or timing comparison."
        )
    else:
        lines.append(
            "None of the seven old combinations survived the stronger whole-episode "
            "and component-control requirements in full."
        )
    if len(layer_leads):
        lines.append(
            f"However, {len(layer_leads)} combination retained a useful partial role "
            "when reaction size and direction were examined separately."
        )
    lines.extend(
        [
            "",
            "This does not mean news is unimportant. It means these exact broad hourly "
            "representations did or did not add measurable information beyond the market "
            "condition with the historical data available.",
            "",
            "## Results",
            "",
            (
                "| Idea | Matched events | Large-reaction rate | Reaction added versus "
                "market setup | Expected-direction rate | Direction added versus market "
                "setup | Useful layer finding |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in decisions.to_dict(orient="records"):
        lines.append(
            "| "
            + str(row["theory_id"]).replace("_", " ")
            + f" | {int(row['market_matched_pairs'])}"
            + f" | {_pct(row['market_matched_event_reaction_rate'])}"
            + f" | {_pct(row['reaction_lift_vs_market_only'])}"
            + f" | {_pct(row['market_matched_event_direction_rate'])}"
            + f" | {_pct(row['direction_lift_vs_market_only'])}"
            + " | "
            + str(row["layer_decision"]).replace("_", " ")
            + " |"
        )
    lines.extend(
        [
            "",
            "Large-reaction rate and expected-direction rate are separate. The stricter "
            "joint result requires both to occur in the same episode. None reached the "
            "frozen 55% joint target; that does not erase a useful partial role.",
            "",
            "## Partial-role interpretation",
            "",
            (
                "This layer interpretation was added after completing the frozen joint "
                "test. It uses the already-frozen reaction and direction measurements, "
                "the same thresholds, and the same whole-episode chance correction, but "
                "it is exploratory and cannot promote a rule."
            ),
            "",
        ]
    )
    informative = decisions.loc[
        ~decisions["layer_decision"].eq("no_repeatable_incremental_role_found")
    ]
    for row in informative.to_dict(orient="records"):
        lines.append(
            f"- **{str(row['theory_id']).replace('_', ' ')}:** "
            f"{row['layer_plain_interpretation']} Direction was "
            f"{_pct(row['market_matched_event_direction_rate'])}, adding "
            f"{_pct(row['direction_lift_vs_market_only'])} versus similar market "
            f"setups; its family-wide chance-check probability was "
            f"{_pct(row['familywise_direction_probability_vs_market_only'])}."
        )
    lines.extend(
        [
            "",
            "A meaningful joint reaction means that the 24-hour move was larger than "
            "normal for that historical window and that the larger side of the move was "
            "in the predeclared direction. It is not a profit measure.",
            "",
            "## Per-window evidence",
            "",
            (
                "| Idea | Window | Whole events | Joint rate | Market-only pairs | "
                "Difference from market-only | News-only pairs | Difference from "
                "news-only |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    full = episode_summary.loc[
        episode_summary["sample_kind"].eq("full_interaction")
    ][
        [
            "theory_id",
            "window_id",
            "whole_episode_count",
            "joint_reaction_and_direction_rate",
        ]
    ]
    market = pair_summary.loc[pair_summary["control_kind"].eq("market_only")][
        ["theory_id", "window_id", "complete_pair_count", "joint_rate_lift"]
    ].rename(
        columns={
            "complete_pair_count": "market_pairs",
            "joint_rate_lift": "market_lift",
        }
    )
    news = pair_summary.loc[pair_summary["control_kind"].eq("news_only")][
        ["theory_id", "window_id", "complete_pair_count", "joint_rate_lift"]
    ].rename(
        columns={
            "complete_pair_count": "news_pairs",
            "joint_rate_lift": "news_lift",
        }
    )
    detail = full.merge(market, on=["theory_id", "window_id"], how="left").merge(
        news, on=["theory_id", "window_id"], how="left"
    )
    for row in detail.to_dict(orient="records"):
        lines.append(
            f"| {str(row['theory_id']).replace('_', ' ')}"
            f" | {row['window_id']} | {int(row['whole_episode_count'])}"
            f" | {_pct(row['joint_reaction_and_direction_rate'])}"
            f" | {int(row['market_pairs']) if pd.notna(row['market_pairs']) else 0}"
            f" | {_pct(row['market_lift'])}"
            f" | {int(row['news_pairs']) if pd.notna(row['news_pairs']) else 0}"
            f" | {_pct(row['news_lift'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen safeguards",
            "",
            "- Seven ideas were frozen and completed as one batch before branching.",
            (
                f"- Raw hourly triggers were collapsed into "
                f"{frozen.EPISODE_GAP_HOURS}-hour whole episodes."
            ),
            "- Market-only controls used the same exact price setup without the news condition.",
            "- News-only controls used the news condition without the exact price setup.",
            (
                f"- A {frozen.SOURCE_STALE_HOURS}-hour-old source condition checked "
                "whether current timing mattered."
            ),
            (
                f"- {frozen.PERMUTATION_ITERATIONS:,} whole-episode reshuffles "
                "controlled the chance of selecting the best of seven ideas."
            ),
            "- Profit, trades, entries, exits, and FreqAI were not used.",
            "",
            "## Important limits",
            "",
            (
                "- These seven ideas were originally selected after viewing these "
                "historical periods. This is a robustness retest, not independent "
                "confirmation."
            ),
            (
                "- The original historical overlay builder is missing, so source "
                "construction cannot currently be regenerated from raw archives. "
                "Frozen source files passed timestamp and target arithmetic checks, "
                "but provenance remains incomplete."
            ),
            (
                "- The source variables are broad hourly media-pressure measures. A "
                "failed representation does not prove that the underlying real-world "
                "stories had no effect."
            ),
            (
                "- A market-only control is not a world with no news; it only lacks "
                "this exact frozen news condition."
            ),
            (
                "- Results describe interactions: news may initiate pressure, while "
                "background, liquidity, price structure, Bitcoin confirmation, and "
                "local levels can amplify, suppress, delay, or reverse what follows."
            ),
            "",
            "## Reaction thresholds by historical window",
            "",
        ]
    )
    for window_id, threshold in thresholds.items():
        lines.append(f"- {window_id}: {_pct(threshold)} 24-hour maximum excursion.")
    lines.extend(
        [
            "",
            "## Coverage notes",
            "",
        ]
    )
    limited = coverage.loc[~coverage["coverage_note"].eq("eligible")]
    for row in limited.to_dict(orient="records"):
        lines.append(
            f"- {row['theory_id']} / {row['window_id']}: {row['coverage_note']}."
        )
    lines.extend(
        [
            "",
            "## Frozen interpretation rule",
            "",
            str(freeze_record["decision_rules"]["historical_only_rule"]),
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        existing = _load_json(RESULT_PATH)
        if existing.get("status") != "completed_historical_news_interaction_review":
            raise ValueError("Existing historical interaction result is not terminal.")
        return existing

    freeze_record, episodes, controls, stale, coverage = load_frozen_inputs()
    outcomes, thresholds = load_outcomes()
    expected = {
        str(theory["theory_id"]): int(theory["expected_direction"])
        for theory in freeze_record["theories"]
    }
    rows = outcome_rows(episodes, stale, outcomes, thresholds)
    pairs = paired_control_rows(controls, outcomes, thresholds, expected)
    episode_summary = summarize_episodes(rows)
    pair_summary = summarize_pairs(pairs)
    null = familywise_null(pairs)
    decisions = decide_theories(
        freeze_record, episode_summary, pair_summary, pairs, null
    )
    report = render_report(
        freeze_record,
        thresholds,
        coverage,
        episode_summary,
        pair_summary,
        decisions,
    )

    g0.atomic_write_parquet(rows, OUTCOME_ROWS_PATH)
    g0.atomic_write_csv(episode_summary, EPISODE_SUMMARY_PATH)
    g0.atomic_write_parquet(pairs, PAIRED_ROWS_PATH)
    g0.atomic_write_csv(pair_summary, PAIRED_SUMMARY_PATH)
    g0.atomic_write_csv(null, NULL_PATH)
    g0.atomic_write_csv(decisions, DECISIONS_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")

    retained = decisions.loc[
        decisions["decision"].eq("retained_historical_interaction_candidate"),
        "theory_id",
    ].tolist()
    promising = decisions.loc[
        decisions["decision"].eq("promising_but_component_control_limited"),
        "theory_id",
    ].tolist()
    layer_leads = decisions.loc[
        decisions["layer_decision"].isin(
            [
                "historical_reaction_and_direction_modifier_lead",
                "historical_direction_modifier_lead",
                "historical_reaction_modifier_lead",
                "strong_direction_modifier_with_period_limit",
            ]
        ),
        "theory_id",
    ].tolist()
    result = {
        "schema_version": 1,
        "status": "completed_historical_news_interaction_review",
        "created_at_utc": g0.utc_now(),
        "run_id": frozen.RUN_ID,
        "profit_used": False,
        "freqai_used": False,
        "independent_confirmation": False,
        "counts": {
            "theories": len(decisions),
            "outcome_rows": len(rows),
            "paired_control_rows": len(pairs),
            "familywise_null_rows": len(null),
            "retained_historical_interactions": len(retained),
            "promising_component_limited": len(promising),
            "posthoc_useful_layer_leads": len(layer_leads),
        },
        "retained_theory_ids": retained,
        "promising_theory_ids": promising,
        "posthoc_useful_layer_theory_ids": layer_leads,
        "posthoc_layer_review_used_only_frozen_outcomes": True,
        "posthoc_layer_review_can_promote_rule": False,
        "reaction_thresholds": thresholds,
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "layer_decision_counts": decisions["layer_decision"].value_counts().to_dict(),
        "artifacts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "outcomes": artifact(OUTCOME_ROWS_PATH),
            "episode_summary": artifact(EPISODE_SUMMARY_PATH),
            "paired_controls": artifact(PAIRED_ROWS_PATH),
            "paired_summary": artifact(PAIRED_SUMMARY_PATH),
            "familywise_null": artifact(NULL_PATH),
            "decisions": artifact(DECISIONS_PATH),
            "plain_review": artifact(REPORT_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        raise SystemExit("Pass --execute to read the frozen future outcomes.")
    print(json.dumps(execute(overwrite=args.overwrite), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
