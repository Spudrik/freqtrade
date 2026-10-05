"""Freeze and run the 2026 Bitcoin-to-meme event transmission batch.

The freeze phase reads only event metadata, previously materialized causal Bitcoin
features, and OHLCV timestamps.  The run phase opens BTC and coin outcomes only after
the questions, cohorts, horizons, controls, and decision rules have been frozen.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
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
RUN_ID = "event_meme_transmission_20260909a"
SOURCE_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "event_signal_fresh_2026_20260909a"
)
SOURCE_FREEZE_PATH = SOURCE_ROOT / "event_signal_fresh_freeze.json"
SOURCE_SAMPLES_PATH = SOURCE_ROOT / "event_signal_fresh_sample_catalog.csv"
SOURCE_CACHE_MANIFEST_PATH = (
    SOURCE_ROOT / "freqai_cache" / "event_signal_fresh_cache_manifest.json"
)
MEME_COHORT_PATH = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "meme_cohort"
    / "meme_top10_selection_20260813.json"
)
OUTPUT_ROOT = SOURCE_ROOT / RUN_ID
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / RUN_ID
)
FREEZE_PATH = OUTPUT_ROOT / "meme_transmission_freeze.json"
FROZEN_SAMPLES_PATH = OUTPUT_ROOT / "meme_transmission_samples.csv"
SCORE_PATH = OUTPUT_ROOT / "meme_transmission_scores.csv"
DECISION_PATH = OUTPUT_ROOT / "meme_transmission_decisions.csv"
RESULT_PATH = OUTPUT_ROOT / "meme_transmission_result.json"

PERIODS = (
    "untouched_confirmation_2026_jan_apr",
    "untouched_confirmation_2026_may_aug",
)
BTC_PAIR = "BTC/USDT:USDT"
ESTABLISHED_PAIRS = (
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
)
HORIZONS = (1, 3)
RECENT_BASELINE_HOURS = 24 * 7
RECENT_BASELINE_MINIMUM = 24 * 3
SENSITIVITY_LOOKBACK_HOURS = 24 * 90
SENSITIVITY_MINIMUM_HOURS = 24 * 45
BTC_PRE_ACTIVITY_THRESHOLD = 1.0
BTC_CONFIRM_VOLUME_THRESHOLD = 1.0
BTC_CONFIRM_ABSOLUTE_RETURN_THRESHOLD = 1.0
COHORT_MAJORITY = 0.60
LEAD_RATE = 0.55
STRONG_RATE = 0.65
TEMPORAL_FLOOR = 0.50
MINIMUM_MAIN_EPISODES = 20
MINIMUM_JOINT_EPISODES = 10
MINIMUM_HALF_EPISODES = 5
MINIMUM_COMPARATOR_UPLIFT = 0.03


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list in frozen sample metadata.")
    return [str(item) for item in parsed]


def _btc_feature_path() -> Path:
    manifest = json.loads(SOURCE_CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_event_freqai_cache":
        raise ValueError("The parent feature cache is not terminal.")
    row = next(item for item in manifest["inventory"] if item["pair"] == BTC_PAIR)
    path = Path(row["feature_path"])
    if g0.sha256_file(path) != row["feature_sha256"]:
        raise ValueError("The frozen BTC feature cache hash changed.")
    return path


def _current_sample_units() -> DataFrame:
    source_freeze = json.loads(SOURCE_FREEZE_PATH.read_text(encoding="utf-8"))
    if source_freeze.get("status") != "frozen_event_signal_fresh_2026_before_outcomes":
        raise ValueError("The parent 2026 event freeze is not valid.")
    if (
        source_freeze["frozen_artifacts"]["samples"]["sha256"]
        != g0.sha256_file(SOURCE_SAMPLES_PATH)
    ):
        raise ValueError("The parent sample catalog changed after its freeze.")
    samples = pd.read_csv(SOURCE_SAMPLES_PATH)
    samples["model_anchor_utc"] = pd.to_datetime(samples["model_anchor_utc"], utc=True)
    samples = samples.loc[samples["model_period"].isin(PERIODS)].copy()
    samples["episode_ids"] = samples["parent_episode_ids_json"].map(_json_list)
    samples = samples.explode("episode_ids").rename(columns={"episode_ids": "episode_id"})
    samples = samples.sort_values(["episode_id", "model_anchor_utc", "sample_id"])
    actual = samples.loc[samples["sample_kind"].eq("actual_event")].drop_duplicates(
        "episode_id", keep="first"
    )
    controls = samples.loc[samples["sample_kind"].eq("matched_control")].drop_duplicates(
        ["episode_id", "sample_id"], keep="first"
    )
    units = pd.concat([actual, controls], ignore_index=True)
    features = pd.read_parquet(
        _btc_feature_path(), columns=["date", "recent__relative_volume"]
    )
    features["date"] = pd.to_datetime(features["date"], utc=True)
    units = units.merge(
        features,
        left_on="model_anchor_utc",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if units["recent__relative_volume"].isna().any():
        raise ValueError("A frozen 2026 sample lacks its causal BTC activity feature.")
    units["btc_pre_activity_signal"] = units["recent__relative_volume"].ge(
        BTC_PRE_ACTIVITY_THRESHOLD
    )
    return units[
        [
            "sample_id",
            "sample_kind",
            "episode_id",
            "model_anchor_utc",
            "model_period",
            "event_families_json",
            "event_kinds_json",
            "recent__relative_volume",
            "btc_pre_activity_signal",
        ]
    ].reset_index(drop=True)


def _cohorts() -> tuple[list[str], list[str]]:
    cohort = json.loads(MEME_COHORT_PATH.read_text(encoding="utf-8"))
    if cohort.get("status") != "frozen_before_reaction_outcomes":
        raise ValueError("The top-ten meme cohort is not frozen.")
    memes = [str(item["freqtrade_pair"]) for item in cohort["members"]]
    if len(memes) != 10 or len(set(memes)) != 10:
        raise ValueError("The frozen meme cohort is not ten unique contracts.")
    return memes, list(ESTABLISHED_PAIRS)


def _timestamp_coverage(pairs: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        dates = pd.read_feather(path, columns=["date"])["date"]
        dates = pd.to_datetime(dates, utc=True)
        rows.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_utc": dates.min(),
                "last_utc": dates.max(),
                "rows": len(dates),
            }
        )
    return rows


def freeze() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "frozen_before_meme_outcomes":
            raise ValueError("Existing meme transmission freeze is not terminal.")
        if existing["source_contracts"]["analysis_script"]["sha256"] != g0.sha256_file(
            ANALYSIS_PATH
        ):
            raise ValueError("Analysis code changed after the meme batch was frozen.")
        return existing

    samples = _current_sample_units()
    memes, established = _cohorts()
    coverage = _timestamp_coverage([BTC_PAIR, *established, *memes])
    required_end = samples["model_anchor_utc"].max() + pd.Timedelta(hours=max(HORIZONS) + 1)
    incomplete = [item["pair"] for item in coverage if item["last_utc"] < required_end]
    if incomplete:
        raise ValueError(f"OHLCV coverage ends too early for: {incomplete}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(samples, FROZEN_SAMPLES_PATH)
    actual = samples.loc[samples["sample_kind"].eq("actual_event")]
    controls = samples.loc[samples["sample_kind"].eq("matched_control")]
    result = {
        "schema_version": 1,
        "status": "frozen_before_meme_outcomes",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "plain_objective": (
            "Test whether a simple pre-event Bitcoin activity signal first predicts a "
            "Bitcoin reaction, whether a confirmed Bitcoin reaction is followed by the "
            "frozen top-ten meme cohort, and whether any meme move exceeds its ordinary "
            "pre-estimated relationship with Bitcoin."
        ),
        "evidence_state": {
            "btc_parent_results_previously_reviewed": True,
            "meme_outcomes_read_by_this_freeze": False,
            "classification": "retrospective_branch_with_unopened_meme_outcomes",
        },
        "questions": [
            {
                "route": "pre_signal_to_btc_volume_continuation",
                "meaning": (
                    "At a known event, did Bitcoin volume in the last completed hour at "
                    "or above its seven-day median precede above-median Bitcoin volume?"
                ),
            },
            {
                "route": "confirmed_btc_to_same_hour_group_activity",
                "meaning": (
                    "When Bitcoin actually reacted, did most memes also show elevated "
                    "volume or range in that same hour? This is co-movement, not a forecast."
                ),
            },
            {
                "route": "confirmed_btc_to_later_group_direction",
                "meaning": (
                    "After one Bitcoin reaction hour was observable, did the meme group "
                    "continue in Bitcoin's initial direction over the following one or "
                    "three hours?"
                ),
            },
            {
                "route": "group_amplification_beyond_ordinary_btc_link",
                "meaning": (
                    "Did memes move farther in Bitcoin's initial direction than a rolling "
                    "ninety-day Bitcoin sensitivity and residual volatility predicted?"
                ),
            },
            {
                "route": "meme_vs_established_amplification",
                "meaning": (
                    "On identical confirmed Bitcoin reactions, were meme moves larger "
                    "than established-coin moves both in raw size and after each group's "
                    "ordinary Bitcoin relationship was removed?"
                ),
            },
            {
                "route": "pre_signal_joint_btc_and_meme_response",
                "meaning": (
                    "From the event start, did the simple Bitcoin signal identify both a "
                    "real Bitcoin reaction and a broad meme response, with later direction "
                    "reported separately?"
                ),
            },
        ],
        "definitions": {
            "btc_pre_activity_signal": (
                f"prior completed BTC hour volume / prior {RECENT_BASELINE_HOURS}-hour "
                f"median >= {BTC_PRE_ACTIVITY_THRESHOLD}"
            ),
            "btc_confirmed_reaction": (
                "event-hour BTC volume and absolute open-to-close movement are each at "
                "least their own causal seven-day median; no combined opaque score"
            ),
            "cohort_activity": (
                f"at least {COHORT_MAJORITY:.0%} of the complete cohort individually "
                "exceed their own causal seven-day median"
            ),
            "later_direction": (
                "sign of the cohort median return after the Bitcoin confirmation hour "
                "versus the sign of Bitcoin's confirmation-hour return"
            ),
            "ordinary_sensitivity": (
                "rolling ninety-day hourly regression on Bitcoin, using only hours before "
                "the event and requiring at least forty-five days"
            ),
        },
        "horizons_after_btc_confirmation_hours": list(HORIZONS),
        "periods": list(PERIODS),
        "cohorts": {
            "top_ten_traded_memes": memes,
            "established_non_meme_comparator": established,
        },
        "decision_rule": {
            "minimum_main_independent_episodes": MINIMUM_MAIN_EPISODES,
            "minimum_joint_exploratory_episodes": MINIMUM_JOINT_EPISODES,
            "minimum_each_half_episodes": MINIMUM_HALF_EPISODES,
            "minimum_success_rate": LEAD_RATE,
            "main_target_rate": STRONG_RATE,
            "minimum_each_half_rate": TEMPORAL_FLOOR,
            "minimum_uplift_over_immediately_simpler_comparator": MINIMUM_COMPARATOR_UPLIFT,
            "same_hour_result_is_never_labelled_prediction": True,
            "event_family_slices_are_descriptive_only": True,
        },
        "sample_support_before_outcomes": {
            "actual_episode_count": int(actual["episode_id"].nunique()),
            "actual_pre_signal_episode_count": int(
                actual.loc[actual["btc_pre_activity_signal"], "episode_id"].nunique()
            ),
            "control_episode_count": int(controls["episode_id"].nunique()),
            "unique_control_anchor_count": int(controls["sample_id"].nunique()),
            "by_period": {
                period: {
                    "actual_episodes": int(
                        actual.loc[actual["model_period"].eq(period), "episode_id"].nunique()
                    ),
                    "actual_pre_signal_episodes": int(
                        actual.loc[
                            actual["model_period"].eq(period)
                            & actual["btc_pre_activity_signal"],
                            "episode_id",
                        ].nunique()
                    ),
                }
                for period in PERIODS
            },
        },
        "timestamp_only_coverage": coverage,
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "parent_freeze": artifact(SOURCE_FREEZE_PATH),
            "parent_samples": artifact(SOURCE_SAMPLES_PATH),
            "parent_cache_manifest": artifact(SOURCE_CACHE_MANIFEST_PATH),
            "btc_pre_event_features": artifact(_btc_feature_path()),
            "meme_cohort": artifact(MEME_COHORT_PATH),
            "frozen_samples": artifact(FROZEN_SAMPLES_PATH),
        },
        "scope": {
            "profit_used": False,
            "trading_rule_tested": False,
            "freqai_used": False,
            "same_hour_causality_claimed": False,
            "meme_test_requires_observed_btc_effect_or_joint_btc_success": True,
        },
    }
    g0.atomic_write_json(result, FREEZE_PATH)
    return result


def _safe_ratio(numerator: Series, denominator: Series) -> Series:
    return numerator.div(denominator.where(denominator.gt(0)))


def _coerce_bool(series: Series) -> Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    unknown = normalized.loc[~normalized.isin(["true", "false"])]
    if len(unknown):
        raise ValueError(f"Unexpected boolean values: {sorted(unknown.unique())}")
    return normalized.eq("true")


def _rolling_sensitivity(frame: DataFrame) -> DataFrame:
    x = frame["btc_close_log_return"].shift(1)
    y = frame["coin_close_log_return"].shift(1)
    window = SENSITIVITY_LOOKBACK_HOURS
    minimum = SENSITIVITY_MINIMUM_HOURS
    mean_x = x.rolling(window, min_periods=minimum).mean()
    mean_y = y.rolling(window, min_periods=minimum).mean()
    mean_xy = x.mul(y).rolling(window, min_periods=minimum).mean()
    mean_x2 = x.pow(2).rolling(window, min_periods=minimum).mean()
    mean_y2 = y.pow(2).rolling(window, min_periods=minimum).mean()
    covariance = mean_xy - mean_x.mul(mean_y)
    variance_x = mean_x2 - mean_x.pow(2)
    variance_y = mean_y2 - mean_y.pow(2)
    beta = covariance.div(variance_x.where(variance_x.gt(0)))
    alpha = mean_y - beta.mul(mean_x)
    residual_variance = variance_y - covariance.pow(2).div(variance_x.where(variance_x.gt(0)))
    return DataFrame(
        {
            "ordinary_alpha": alpha,
            "ordinary_beta_to_btc": beta,
            "ordinary_residual_std": residual_variance.clip(lower=0).pow(0.5),
        },
        index=frame.index,
    )


def pair_surface(pair: str, btc: DataFrame) -> DataFrame:
    coin = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
    coin = coin.sort_values("date", kind="stable").drop_duplicates("date")
    frame = coin.merge(
        btc[["date", "close"]].rename(columns={"close": "btc_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    frame["coin_close_log_return"] = np.log(frame["close"]).diff()
    frame["btc_close_log_return"] = np.log(frame["btc_close"]).diff()
    event_log_return = np.log(frame["close"].div(frame["open"]))
    range_fraction = frame["high"].sub(frame["low"]).div(frame["open"])
    volume_median = frame["volume"].shift(1).rolling(
        RECENT_BASELINE_HOURS, min_periods=RECENT_BASELINE_MINIMUM
    ).median()
    abs_return_median = event_log_return.abs().shift(1).rolling(
        RECENT_BASELINE_HOURS, min_periods=RECENT_BASELINE_MINIMUM
    ).median()
    range_median = range_fraction.shift(1).rolling(
        RECENT_BASELINE_HOURS, min_periods=RECENT_BASELINE_MINIMUM
    ).median()
    frame["event_log_return"] = event_log_return
    frame["event_volume_ratio"] = _safe_ratio(frame["volume"], volume_median)
    frame["event_abs_return_ratio"] = _safe_ratio(
        event_log_return.abs(), abs_return_median
    )
    frame["event_range_ratio"] = _safe_ratio(range_fraction, range_median)
    sensitivity = _rolling_sensitivity(frame)
    frame = pd.concat([frame, sensitivity], axis=1)
    for horizon in HORIZONS:
        frame[f"lag_log_return_{horizon}h"] = np.log(
            frame["close"].shift(-horizon).div(frame["open"].shift(-1))
        )
    return frame[
        [
            "date",
            "event_log_return",
            "event_volume_ratio",
            "event_abs_return_ratio",
            "event_range_ratio",
            "ordinary_alpha",
            "ordinary_beta_to_btc",
            "ordinary_residual_std",
            *[f"lag_log_return_{horizon}h" for horizon in HORIZONS],
        ]
    ]


def extract_pair_rows(pair: str, samples: DataFrame, btc: DataFrame) -> DataFrame:
    surface = pair_surface(pair, btc)
    joined = samples.merge(
        surface,
        left_on="model_anchor_utc",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if joined["event_log_return"].isna().any():
        raise ValueError(f"Missing outcome coverage for {pair}.")
    parts: list[DataFrame] = []
    fixed = [
        "sample_id",
        "sample_kind",
        "episode_id",
        "model_anchor_utc",
        "model_period",
        "event_families_json",
        "event_kinds_json",
        "recent__relative_volume",
        "btc_pre_activity_signal",
        "event_log_return",
        "event_volume_ratio",
        "event_abs_return_ratio",
        "event_range_ratio",
        "ordinary_alpha",
        "ordinary_beta_to_btc",
        "ordinary_residual_std",
    ]
    for horizon in HORIZONS:
        part = joined[fixed].copy()
        part["pair"] = pair
        part["horizon_hours"] = horizon
        part["lag_log_return"] = joined[f"lag_log_return_{horizon}h"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def add_btc_context(rows: DataFrame) -> DataFrame:
    btc = rows.loc[rows["pair"].eq(BTC_PAIR)][
        [
            "sample_id",
            "episode_id",
            "horizon_hours",
            "event_log_return",
            "event_volume_ratio",
            "event_abs_return_ratio",
            "lag_log_return",
        ]
    ].rename(
        columns={
            "event_log_return": "btc_event_log_return",
            "event_volume_ratio": "btc_event_volume_ratio",
            "event_abs_return_ratio": "btc_event_abs_return_ratio",
            "lag_log_return": "btc_lag_log_return",
        }
    )
    output = rows.merge(
        btc,
        on=["sample_id", "episode_id", "horizon_hours"],
        how="left",
        validate="many_to_one",
    )
    output["btc_confirmed_reaction"] = (
        output["btc_event_volume_ratio"].ge(BTC_CONFIRM_VOLUME_THRESHOLD)
        & output["btc_event_abs_return_ratio"].ge(
            BTC_CONFIRM_ABSOLUTE_RETURN_THRESHOLD
        )
        & output["btc_event_log_return"].ne(0)
    )
    output["btc_initial_sign"] = np.sign(output["btc_event_log_return"])
    expected = (
        output["ordinary_alpha"].mul(output["horizon_hours"])
        + output["ordinary_beta_to_btc"].mul(output["btc_lag_log_return"])
    )
    residual = output["lag_log_return"] - expected
    scale = output["ordinary_residual_std"].mul(
        np.sqrt(output["horizon_hours"].astype(float))
    )
    output["ordinary_residual_z"] = residual.div(scale.where(scale.gt(0)))
    output["residual_toward_initial_btc_z"] = (
        output["ordinary_residual_z"] * output["btc_initial_sign"]
    )
    output["direction_aligned_with_initial_btc"] = np.sign(
        output["lag_log_return"]
    ).eq(output["btc_initial_sign"])
    return output


def cohort_rows(
    pair_rows: DataFrame, cohort_name: str, pairs: Sequence[str]
) -> DataFrame:
    source = pair_rows.loc[pair_rows["pair"].isin(pairs)].copy()
    keys = [
        "sample_id",
        "sample_kind",
        "episode_id",
        "model_anchor_utc",
        "model_period",
        "event_families_json",
        "event_kinds_json",
        "recent__relative_volume",
        "btc_pre_activity_signal",
        "btc_confirmed_reaction",
        "btc_event_log_return",
        "btc_initial_sign",
        "horizon_hours",
    ]
    grouped = source.groupby(keys, dropna=False, sort=False)
    result = grouped.agg(
        coin_count=("pair", "nunique"),
        elevated_volume_fraction=("event_volume_ratio", lambda values: values.ge(1).mean()),
        elevated_range_fraction=("event_range_ratio", lambda values: values.ge(1).mean()),
        cohort_lag_log_return=("lag_log_return", "median"),
        cohort_absolute_lag_return=("lag_log_return", lambda values: values.abs().median()),
        median_residual_toward_btc_z=("residual_toward_initial_btc_z", "median"),
    ).reset_index()
    required = len(pairs)
    result["cohort"] = cohort_name
    result["complete_cohort"] = result["coin_count"].eq(required)
    result["volume_majority_reacted"] = result["elevated_volume_fraction"].ge(
        COHORT_MAJORITY
    ).where(result["complete_cohort"])
    result["range_majority_reacted"] = result["elevated_range_fraction"].ge(
        COHORT_MAJORITY
    ).where(result["complete_cohort"])
    result["later_direction_aligned"] = np.sign(
        result["cohort_lag_log_return"]
    ).eq(result["btc_initial_sign"]).where(result["complete_cohort"])
    result["amplified_beyond_ordinary"] = (
        result["later_direction_aligned"].fillna(False)
        & result["median_residual_toward_btc_z"].gt(0)
    ).where(result["complete_cohort"])
    return result


def _analysis_units(frame: DataFrame) -> DataFrame:
    actual = frame.loc[frame["sample_kind"].eq("actual_event")].drop_duplicates(
        ["episode_id", "horizon_hours"], keep="first"
    )
    controls = frame.loc[frame["sample_kind"].eq("matched_control")].drop_duplicates(
        ["sample_id", "horizon_hours"], keep="first"
    )
    return pd.concat([actual, controls], ignore_index=True)


def _score_binary(
    frame: DataFrame,
    *,
    route: str,
    outcome: str,
    success_column: str,
    condition: Series,
    cohort: str,
    horizon: int,
    condition_name: str,
) -> list[dict[str, Any]]:
    selected = _analysis_units(frame.loc[condition & frame[success_column].notna()].copy())
    records: list[dict[str, Any]] = []
    for sample_kind in ("actual_event", "matched_control"):
        kind = selected.loc[selected["sample_kind"].eq(sample_kind)]
        for period in ("all_2026", *PERIODS):
            group = kind if period == "all_2026" else kind.loc[kind["model_period"].eq(period)]
            success = group[success_column].astype(bool)
            records.append(
                {
                    "route": route,
                    "outcome": outcome,
                    "cohort": cohort,
                    "horizon_hours": horizon,
                    "condition": condition_name,
                    "sample_kind": sample_kind,
                    "period": period,
                    "support": len(group),
                    "successes": int(success.sum()),
                    "success_rate": float(success.mean()) if len(group) else np.nan,
                }
            )
    return records


def build_scores(pair_rows: DataFrame, groups: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    btc = pair_rows.loc[
        pair_rows["pair"].eq(BTC_PAIR) & pair_rows["horizon_hours"].eq(HORIZONS[0])
    ].copy()
    btc["btc_volume_continued"] = btc["btc_event_volume_ratio"].ge(1)
    for signal_value, name in ((True, "pre_signal"), (False, "no_pre_signal")):
        records.extend(
            _score_binary(
                btc,
                route="pre_signal_to_btc_volume_continuation",
                outcome="btc_event_hour_volume_above_prior_median",
                success_column="btc_volume_continued",
                condition=btc["btc_pre_activity_signal"].eq(signal_value),
                cohort="btc",
                horizon=0,
                condition_name=name,
            )
        )

    for cohort in ("memes", "established_alts"):
        for horizon in HORIZONS:
            part = groups.loc[
                groups["cohort"].eq(cohort) & groups["horizon_hours"].eq(horizon)
            ].copy()
            confirmed = part["btc_confirmed_reaction"] & part["complete_cohort"]
            for success_column, outcome in (
                ("volume_majority_reacted", "same_hour_volume_majority"),
                ("range_majority_reacted", "same_hour_range_majority"),
            ):
                if horizon == HORIZONS[0]:
                    records.extend(
                        _score_binary(
                            part,
                            route="confirmed_btc_to_same_hour_group_activity",
                            outcome=outcome,
                            success_column=success_column,
                            condition=confirmed,
                            cohort=cohort,
                            horizon=0,
                            condition_name="btc_reaction_observed",
                        )
                    )
            records.extend(
                _score_binary(
                    part,
                    route="confirmed_btc_to_later_group_direction",
                    outcome="later_direction_matches_initial_btc",
                    success_column="later_direction_aligned",
                    condition=confirmed,
                    cohort=cohort,
                    horizon=horizon,
                    condition_name="btc_reaction_observed",
                )
            )
            records.extend(
                _score_binary(
                    part,
                    route="group_amplification_beyond_ordinary_btc_link",
                    outcome="move_exceeds_ordinary_btc_link_in_initial_direction",
                    success_column="amplified_beyond_ordinary",
                    condition=confirmed,
                    cohort=cohort,
                    horizon=horizon,
                    condition_name="btc_reaction_observed",
                )
            )
            part["joint_activity"] = (
                part["btc_confirmed_reaction"]
                & part["volume_majority_reacted"].fillna(False)
            )
            part["joint_activity_and_direction"] = (
                part["joint_activity"]
                & part["later_direction_aligned"].fillna(False)
            )
            if cohort == "memes":
                if horizon == HORIZONS[0]:
                    records.extend(
                        _score_binary(
                            part,
                            route="pre_signal_joint_btc_and_meme_response",
                            outcome="btc_and_meme_activity",
                            success_column="joint_activity",
                            condition=part["btc_pre_activity_signal"]
                            & part["complete_cohort"],
                            cohort=cohort,
                            horizon=0,
                            condition_name="pre_signal",
                        )
                    )
                records.extend(
                    _score_binary(
                        part,
                        route="pre_signal_joint_btc_and_meme_response",
                        outcome="btc_and_meme_activity_plus_later_direction",
                        success_column="joint_activity_and_direction",
                        condition=part["btc_pre_activity_signal"]
                        & part["complete_cohort"],
                        cohort=cohort,
                        horizon=horizon,
                        condition_name="pre_signal",
                    )
                )

    comparison_keys = [
        "sample_id",
        "sample_kind",
        "episode_id",
        "model_anchor_utc",
        "model_period",
        "event_families_json",
        "event_kinds_json",
        "recent__relative_volume",
        "btc_pre_activity_signal",
        "btc_confirmed_reaction",
        "btc_event_log_return",
        "btc_initial_sign",
        "horizon_hours",
    ]
    meme_compare = groups.loc[groups["cohort"].eq("memes")].drop(columns="cohort")
    established_compare = groups.loc[groups["cohort"].eq("established_alts")].drop(
        columns="cohort"
    )
    comparison = meme_compare.merge(
        established_compare,
        on=comparison_keys,
        how="inner",
        suffixes=("_meme", "_established"),
        validate="one_to_one",
    )
    comparison["meme_raw_move_larger"] = comparison[
        "cohort_absolute_lag_return_meme"
    ].gt(comparison["cohort_absolute_lag_return_established"])
    comparison["meme_adjusted_move_larger"] = comparison[
        "median_residual_toward_btc_z_meme"
    ].gt(comparison["median_residual_toward_btc_z_established"])
    comparison_condition = (
        comparison["btc_confirmed_reaction"]
        & comparison["complete_cohort_meme"]
        & comparison["complete_cohort_established"]
    )
    for horizon in HORIZONS:
        part = comparison.loc[comparison["horizon_hours"].eq(horizon)]
        condition = comparison_condition.loc[part.index]
        for success_column, outcome in (
            ("meme_raw_move_larger", "meme_raw_move_larger_than_established"),
            (
                "meme_adjusted_move_larger",
                "meme_adjusted_move_larger_than_established",
            ),
        ):
            records.extend(
                _score_binary(
                    part,
                    route="meme_vs_established_amplification",
                    outcome=outcome,
                    success_column=success_column,
                    condition=condition,
                    cohort="memes_vs_established_alts",
                    horizon=horizon,
                    condition_name="btc_reaction_observed",
                )
            )
    return DataFrame.from_records(records)


def _find_rate(
    scores: DataFrame,
    row: Series,
    *,
    sample_kind: str,
    period: str,
    condition: str | None = None,
) -> tuple[int, float]:
    mask = (
        scores["route"].eq(row["route"])
        & scores["outcome"].eq(row["outcome"])
        & scores["cohort"].eq(row["cohort"])
        & scores["horizon_hours"].eq(row["horizon_hours"])
        & scores["sample_kind"].eq(sample_kind)
        & scores["period"].eq(period)
    )
    mask &= scores["condition"].eq(condition or str(row["condition"]))
    match = scores.loc[mask]
    if match.empty:
        return 0, np.nan
    return int(match.iloc[0]["support"]), float(match.iloc[0]["success_rate"])


def decide(scores: DataFrame) -> DataFrame:
    primary = scores.loc[
        scores["sample_kind"].eq("actual_event")
        & scores["period"].eq("all_2026")
        & ~scores["condition"].eq("no_pre_signal")
    ].copy()
    records: list[dict[str, Any]] = []
    for _, row in primary.iterrows():
        support = int(row["support"])
        rate = float(row["success_rate"]) if pd.notna(row["success_rate"]) else np.nan
        first_support, first_rate = _find_rate(
            scores, row, sample_kind="actual_event", period=PERIODS[0]
        )
        second_support, second_rate = _find_rate(
            scores, row, sample_kind="actual_event", period=PERIODS[1]
        )
        control_support, control_rate = _find_rate(
            scores, row, sample_kind="matched_control", period="all_2026"
        )
        minimum = (
            MINIMUM_JOINT_EPISODES
            if row["route"] == "pre_signal_joint_btc_and_meme_response"
            else MINIMUM_MAIN_EPISODES
        )
        support_pass = support >= minimum
        temporal_pass = (
            first_support >= MINIMUM_HALF_EPISODES
            and second_support >= MINIMUM_HALF_EPISODES
            and first_rate >= TEMPORAL_FLOOR
            and second_rate >= TEMPORAL_FLOOR
        )
        rate_pass = pd.notna(rate) and rate >= LEAD_RATE
        comparator_rate = np.nan
        comparator_uplift = np.nan
        comparator_pass = True
        if row["route"] == "pre_signal_to_btc_volume_continuation":
            _, comparator_rate = _find_rate(
                scores,
                row,
                sample_kind="actual_event",
                period="all_2026",
                condition="no_pre_signal",
            )
            comparator_uplift = rate - comparator_rate
            comparator_pass = comparator_uplift >= MINIMUM_COMPARATOR_UPLIFT
        retained = bool(support_pass and temporal_pass and rate_pass and comparator_pass)
        event_specific = bool(
            retained
            and pd.notna(control_rate)
            and rate - control_rate >= MINIMUM_COMPARATOR_UPLIFT
        )
        if not support_pass:
            verdict = "coverage_parked"
        elif not temporal_pass:
            verdict = "parked_not_temporally_consistent"
        elif not rate_pass:
            verdict = "not_retained_below_55_percent"
        elif not comparator_pass:
            verdict = "not_retained_did_not_beat_simple_comparator"
        elif row["route"] == "confirmed_btc_to_same_hour_group_activity":
            verdict = "retained_same_hour_description_not_prediction"
        elif event_specific:
            verdict = "retained_event_specific_lead"
        else:
            verdict = "retained_general_market_lead_not_event_specific"
        records.append(
            {
                "route": row["route"],
                "outcome": row["outcome"],
                "cohort": row["cohort"],
                "horizon_hours": int(row["horizon_hours"]),
                "support": support,
                "success_rate": rate,
                "first_half_support": first_support,
                "first_half_rate": first_rate,
                "second_half_support": second_support,
                "second_half_rate": second_rate,
                "matched_control_support": control_support,
                "matched_control_rate": control_rate,
                "event_minus_control": rate - control_rate,
                "simple_comparator_rate": comparator_rate,
                "uplift_over_simple_comparator": comparator_uplift,
                "meets_55_floor": bool(retained),
                "meets_65_target": bool(retained and rate >= STRONG_RATE),
                "event_specific": event_specific,
                "verdict": verdict,
            }
        )
    return DataFrame.from_records(records)


def run() -> dict[str, Any]:
    if RESULT_PATH.is_file():
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_meme_transmission_review":
            raise ValueError("Existing meme transmission result is not terminal.")
        return result
    frozen = freeze()
    for name, contract in frozen["source_contracts"].items():
        path = Path(contract["path"])
        if contract["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen source changed before outcome review: {name}")
    for contract in frozen["timestamp_only_coverage"]:
        path = Path(contract["path"])
        if contract["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen OHLCV changed before outcome review: {path}")
    samples = pd.read_csv(FROZEN_SAMPLES_PATH)
    samples["model_anchor_utc"] = pd.to_datetime(samples["model_anchor_utc"], utc=True)
    samples["btc_pre_activity_signal"] = _coerce_bool(
        samples["btc_pre_activity_signal"]
    )
    memes = frozen["cohorts"]["top_ten_traded_memes"]
    established = frozen["cohorts"]["established_non_meme_comparator"]
    btc = g0.load_ohlcv(g0.ohlcv_path(BTC_PAIR, "1h"))
    btc = btc.sort_values("date", kind="stable").drop_duplicates("date")
    parts = [
        extract_pair_rows(pair, samples, btc)
        for pair in [BTC_PAIR, *established, *memes]
    ]
    pair_rows = add_btc_context(pd.concat(parts, ignore_index=True))
    groups = pd.concat(
        [
            cohort_rows(pair_rows, "memes", memes),
            cohort_rows(pair_rows, "established_alts", established),
        ],
        ignore_index=True,
    )
    scores = build_scores(pair_rows, groups)
    decisions = decide(scores)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    pair_path = DETAIL_ROOT / "meme_transmission_pair_rows.parquet"
    group_path = DETAIL_ROOT / "meme_transmission_group_rows.parquet"
    g0.atomic_write_parquet(pair_rows, pair_path)
    g0.atomic_write_parquet(groups, group_path)
    g0.atomic_write_csv(scores, SCORE_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_event_meme_transmission_review",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "profit_used": False,
        "freqai_used": False,
        "trading_rule_tested": False,
        "same_hour_results_are_descriptive_only": True,
        "actual_episode_count": int(
            samples.loc[samples["sample_kind"].eq("actual_event"), "episode_id"].nunique()
        ),
        "retained_result_count": int(decisions["verdict"].str.startswith("retained").sum()),
        "main_target_result_count": int(decisions["meets_65_target"].sum()),
        "freeze_contract": artifact(FREEZE_PATH),
        "artifacts": {
            "scores": artifact(SCORE_PATH),
            "decisions": artifact(DECISION_PATH),
            "pair_details": artifact(pair_path),
            "group_details": artifact(group_path),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args(argv)
    result = freeze() if args.phase == "freeze" else run()
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
