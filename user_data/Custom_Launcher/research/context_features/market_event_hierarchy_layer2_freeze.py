"""Freeze the complete Layer 2 individual-link batch before reading outcomes."""

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
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
LAYER1_RESULT = layer1.OUTPUT_ROOT / "layer1_result.json"
LAYER1_FREEZE = layer1.OUTPUT_ROOT / "layer1_freeze.json"
LAYER1_EVENTS = layer1.OUTPUT_ROOT / "official_fomc_event_catalog.csv"
CONTEXT_SNAPSHOT = (
    layer1.USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "confluence_cache"
    / "trader_confluence_1h_latest.parquet"
)
OUTPUT_ROOT = (
    layer1.USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "layer2_individual_links_20260903a"
)

HORIZONS = (1, 2, 4, 8, 24)
CONTROL_COUNT = 4
CONTROL_SEARCH_WEEKS = 52
EVENT_EXCLUSION_HOURS = 168
NEWS_HISTORY_SAME_HOUR_OBSERVATIONS = 365
NEWS_MINIMUM_SAME_HOUR_OBSERVATIONS = 120
NEWS_SPIKE_QUANTILE = 0.999
NEWS_HIGH_CONTROL_LOW_QUANTILE = 0.95
NEWS_HIGH_CONTROL_HIGH_QUANTILE = 0.99
NEWS_EPISODE_COOLDOWN_HOURS = 168


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_layer1() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    result = json.loads(LAYER1_RESULT.read_text(encoding="utf-8"))
    freeze = json.loads(LAYER1_FREEZE.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_hierarchy_layer1":
        raise ValueError("Layer 1 result is not terminal.")
    if freeze.get("status") != "frozen_before_event_market_outcomes":
        raise ValueError("Layer 1 definitions are not frozen.")
    if freeze.get("outcomes_read"):
        raise ValueError("Layer 1 unexpectedly read market outcomes.")
    if result["artifacts"]["freeze"]["sha256"] != g0.sha256_file(LAYER1_FREEZE):
        raise ValueError("Layer 1 freeze changed after completion.")
    events = pd.read_csv(LAYER1_EVENTS)
    events["anchor_utc"] = pd.to_datetime(events["anchor_utc"], utc=True)
    historical = events.loc[events["historical_activity_test_eligible"].astype(bool)].copy()
    return result, freeze, historical


def load_context_activity() -> DataFrame:
    columns = [
        "date",
        "ctx_gdelt_event_count_1h",
        "ctx_gdelt_event_count_24h",
        "ctx_max_source_available_at",
    ]
    frame = pd.read_parquet(CONTEXT_SNAPSHOT, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["ctx_max_source_available_at"] = pd.to_datetime(
        frame["ctx_max_source_available_at"], utc=True, errors="coerce"
    )
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date")
    frame["gdelt_1h"] = pd.to_numeric(
        frame["ctx_gdelt_event_count_1h"], errors="coerce"
    )
    frame["gdelt_24h"] = pd.to_numeric(
        frame["ctx_gdelt_event_count_24h"], errors="coerce"
    )
    return frame[
        ["date", "gdelt_1h", "gdelt_24h", "ctx_max_source_available_at"]
    ].reset_index(drop=True)


def add_causal_news_thresholds(frame: DataFrame) -> DataFrame:
    """Normalize news within UTC hour and never bridge a source gap."""

    output = frame.copy()
    output["segment"] = output["date"].diff().ne(pd.Timedelta(hours=1)).cumsum()
    output["utc_hour"] = output["date"].dt.hour
    log_count = np.log1p(output["gdelt_1h"].clip(lower=0))
    grouping = [output["segment"], output["utc_hour"]]

    def quantile(values: Series, value: float) -> Series:
        return values.shift(1).rolling(
            NEWS_HISTORY_SAME_HOUR_OBSERVATIONS,
            min_periods=NEWS_MINIMUM_SAME_HOUR_OBSERVATIONS,
        ).quantile(value)

    output["news_log_count"] = log_count
    output["news_q999"] = log_count.groupby(grouping).transform(
        lambda values: quantile(values, NEWS_SPIKE_QUANTILE)
    )
    output["news_q95"] = log_count.groupby(grouping).transform(
        lambda values: quantile(values, NEWS_HIGH_CONTROL_LOW_QUANTILE)
    )
    output["news_q99"] = log_count.groupby(grouping).transform(
        lambda values: quantile(values, NEWS_HIGH_CONTROL_HIGH_QUANTILE)
    )
    output["source_timestamp_safe"] = output["ctx_max_source_available_at"].le(
        output["date"]
    )
    output["news_spike_raw"] = (
        output["source_timestamp_safe"]
        & output["gdelt_1h"].gt(0)
        & output["news_log_count"].gt(output["news_q999"])
    )
    output["news_high_not_spike"] = (
        output["source_timestamp_safe"]
        & output["news_log_count"].ge(output["news_q95"])
        & output["news_log_count"].lt(output["news_q99"])
    )
    return output


def select_news_episodes(frame: DataFrame) -> DataFrame:
    candidates = frame.loc[
        frame["news_spike_raw"]
        & frame["date"].ge(pd.Timestamp("2021-01-01T00:00:00Z"))
        & frame["date"].lt(pd.Timestamp(layer1.INFORMATION_CUTOFF_UTC))
    ].copy()
    selected: list[int] = []
    last_anchor: pd.Timestamp | None = None
    for row in candidates.itertuples():
        anchor = pd.Timestamp(row.date)
        if last_anchor is None or anchor - last_anchor >= pd.Timedelta(
            hours=NEWS_EPISODE_COOLDOWN_HOURS
        ):
            selected.append(int(row.Index))
            last_anchor = anchor
    output = frame.loc[selected].copy()
    output["event_id"] = output["date"].dt.strftime("gdelt_activity_%Y_%m_%d_%H")
    output["event_source"] = "gdelt_activity_spike"
    output["event_family"] = "unexpected_activity_without_story_semantics"
    output["selection_value"] = output["gdelt_1h"]
    output["selection_threshold"] = np.expm1(output["news_q999"])
    output["whole_event_partition"] = output["date"].map(layer1.whole_event_partition)
    return output[
        [
            "event_id",
            "event_source",
            "event_family",
            "date",
            "whole_event_partition",
            "selection_value",
            "selection_threshold",
        ]
    ].rename(columns={"date": "anchor_utc"})


def causal_btc_prestate() -> DataFrame:
    frame = g0.load_ohlcv(g0.ohlcv_path("BTC/USDT:USDT", "1h"))
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date")
    close = frame["close"]
    returns = close.pct_change()
    frame["pre_return_30d"] = close.shift(1).div(close.shift(721)).sub(1.0)
    frame["pre_volatility_24h"] = returns.shift(1).rolling(24, min_periods=24).std()
    frame["pre_volatility_30d"] = returns.shift(1).rolling(720, min_periods=360).std()
    frame["pre_volume_median_24h"] = frame["volume"].shift(1).rolling(
        24, min_periods=24
    ).median()
    return frame[
        [
            "date",
            "pre_return_30d",
            "pre_volatility_24h",
            "pre_volatility_30d",
            "pre_volume_median_24h",
        ]
    ]


def combine_event_catalogs(
    fomc: DataFrame, news: DataFrame
) -> tuple[DataFrame, DataFrame]:
    official = fomc[
        ["event_id", "event_family", "anchor_utc", "whole_event_partition"]
    ].copy()
    official["event_source"] = "official_fomc"
    official["selection_value"] = np.nan
    official["selection_threshold"] = np.nan
    official = official[
        [
            "event_id",
            "event_source",
            "event_family",
            "anchor_utc",
            "whole_event_partition",
            "selection_value",
            "selection_threshold",
        ]
    ]
    official_anchors = list(official["anchor_utc"])
    collision = news["anchor_utc"].map(
        lambda value: any(
            abs(pd.Timestamp(value) - anchor)
            <= pd.Timedelta(hours=EVENT_EXCLUSION_HOURS)
            for anchor in official_anchors
        )
    )
    collided = news.loc[collision].copy()
    collided["exclusion_reason"] = "within_168h_of_regular_fomc_release"
    clean_news = news.loc[~collision].copy()
    combined = pd.concat([official, clean_news], ignore_index=True)
    combined = combined.sort_values("anchor_utc", kind="stable").reset_index(drop=True)
    return combined, collided


def _state_distance(event: Series, candidates: DataFrame) -> Series:
    tiny = np.finfo(float).eps
    distance = (
        np.log(candidates["pre_volatility_24h"].clip(lower=tiny))
        .sub(np.log(max(float(event["pre_volatility_24h"]), tiny)))
        .abs()
    )
    distance += (
        np.log(candidates["pre_volatility_30d"].clip(lower=tiny))
        .sub(np.log(max(float(event["pre_volatility_30d"]), tiny)))
        .abs()
    )
    scale = max(float(event["pre_volatility_30d"]) * np.sqrt(720), 0.01)
    distance += candidates["pre_return_30d"].sub(float(event["pre_return_30d"])).abs() / scale
    return distance


def build_control_map(events: DataFrame, context: DataFrame) -> DataFrame:
    prestate = causal_btc_prestate().merge(
        context[["date", "gdelt_24h", "news_high_not_spike"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    prestate["pre_gdelt_24h"] = prestate["gdelt_24h"].shift(1)
    indexed = prestate.set_index("date", drop=False)
    blocked_anchors = list(events["anchor_utc"])

    def blocked(timestamp: pd.Timestamp) -> bool:
        return any(
            abs(timestamp - anchor) <= pd.Timedelta(hours=EVENT_EXCLUSION_HOURS)
            for anchor in blocked_anchors
        )

    records: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_utc)
        if anchor not in indexed.index:
            continue
        event_state = indexed.loc[anchor]
        weekly = [anchor - pd.Timedelta(weeks=week) for week in range(1, CONTROL_SEARCH_WEEKS + 1)]
        eligible_times = [
            value
            for value in weekly
            if value in indexed.index
            and not blocked(value)
            and pd.notna(indexed.at[value, "pre_volatility_24h"])
            and pd.notna(indexed.at[value, "pre_volatility_30d"])
            and pd.notna(indexed.at[value, "pre_return_30d"])
        ]
        eligible = indexed.loc[eligible_times].copy().reset_index(drop=True)
        if eligible.empty:
            continue
        eligible["state_distance"] = _state_distance(event_state, eligible)
        matched = eligible.sort_values(["state_distance", "date"], kind="stable").head(
            CONTROL_COUNT
        )
        fixed = eligible.sort_values("date", ascending=False, kind="stable").head(CONTROL_COUNT)
        high_news = eligible.loc[eligible["news_high_not_spike"].fillna(False)].sort_values(
            ["state_distance", "date"], kind="stable"
        ).head(CONTROL_COUNT)
        groups = {
            "matched_prior_state": matched,
            "fixed_prior_week": fixed,
            "high_news_without_selected_event": high_news,
        }
        for control_type, selected in groups.items():
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                records.append(
                    {
                        "event_id": str(event.event_id),
                        "event_source": str(event.event_source),
                        "event_anchor_utc": anchor,
                        "control_type": control_type,
                        "control_rank": rank,
                        "control_anchor_utc": pd.Timestamp(row.date),
                        "state_distance": float(row.state_distance),
                        "control_pre_return_30d": float(row.pre_return_30d),
                        "control_pre_volatility_24h": float(row.pre_volatility_24h),
                        "control_pre_volatility_30d": float(row.pre_volatility_30d),
                        "control_pre_gdelt_24h": (
                            float(row.pre_gdelt_24h) if pd.notna(row.pre_gdelt_24h) else np.nan
                        ),
                    }
                )
    return DataFrame.from_records(records)


def coverage_inventory(events: DataFrame, cohorts: dict[str, Any]) -> DataFrame:
    pairs = list(dict.fromkeys(
        ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        + list(cohorts["established_alts"])
        + list(cohorts["top_ten_traded_memes"])
    ))
    records: list[dict[str, Any]] = []
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        dates = pd.to_datetime(pd.read_feather(path, columns=["date"])["date"], utc=True)
        earliest = dates.min() + pd.Timedelta(days=90)
        latest = dates.max() - pd.Timedelta(hours=168)
        for source in ("official_fomc", "gdelt_activity_spike"):
            source_events = events.loc[events["event_source"].eq(source)]
            eligible = source_events["anchor_utc"].between(earliest, latest, inclusive="both")
            records.append(
                {
                    "pair": pair,
                    "event_source": source,
                    "available_start_utc": dates.min(),
                    "available_end_utc": dates.max(),
                    "eligible_event_count": int(eligible.sum()),
                    "source_event_count": len(source_events),
                }
            )
    return DataFrame.from_records(records)


def freeze_document(
    *,
    layer1_freeze: dict[str, Any],
    events: DataFrame,
    collisions: DataFrame,
    controls: DataFrame,
    coverage: DataFrame,
) -> dict[str, Any]:
    cohorts = layer1_freeze["cohorts"]
    common_meme_count = min(
        coverage.loc[
            coverage["pair"].isin(cohorts["top_ten_traded_memes"])
            & coverage["event_source"].eq("official_fomc"),
            "eligible_event_count",
        ]
    )
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "layer": 2,
        "status": "frozen_before_layer2_individual_link_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "all_siblings_frozen_together": True,
        "event_counts": {
            str(key): int(value)
            for key, value in events["event_source"].value_counts().to_dict().items()
        },
        "gdelt_collision_events_excluded": len(collisions),
        "controls": {
            "types": sorted(controls["control_type"].unique().tolist()),
            "target_per_type_per_event": CONTROL_COUNT,
            "all_controls_precede_their_event": bool(
                (controls["control_anchor_utc"] < controls["event_anchor_utc"]).all()
            ),
            "selection_uses_outcomes": False,
        },
        "news_activity_detector": {
            "input": "ctx_gdelt_event_count_1h",
            "transform": "log1p",
            "seasonality_control": "compare only with the same UTC hour",
            "history_observations_per_utc_hour": NEWS_HISTORY_SAME_HOUR_OBSERVATIONS,
            "minimum_observations_per_utc_hour": NEWS_MINIMUM_SAME_HOUR_OBSERVATIONS,
            "primary_quantile": NEWS_SPIKE_QUANTILE,
            "episode_cooldown_hours": NEWS_EPISODE_COOLDOWN_HOURS,
            "semantics": "activity/intensity only; no story or direction",
        },
        "individual_links": [
            {
                "id": "official_event_to_market_activity",
                "status": "frozen_ready",
                "scopes": ["BTC", "ETH", "broad established market"],
            },
            {
                "id": "gdelt_spike_to_market_activity",
                "status": "frozen_ready_activity_only",
                "scopes": ["BTC", "ETH", "broad established market"],
            },
            {
                "id": "event_to_signed_direction",
                "status": "coverage_parked_missing_expectation_and_surprise_sign",
            },
            {"id": "btc_eth_breadth_leadership", "status": "frozen_ready"},
            {"id": "leader_to_established_alts", "status": "frozen_ready"},
            {
                "id": "leader_to_memes",
                "status": "frozen_exploratory_limited_common_events",
                "common_top_ten_event_count": int(common_meme_count),
            },
            {
                "id": "meme_residual_amplification",
                "status": "frozen_exploratory_limited_common_events",
                "common_top_ten_event_count": int(common_meme_count),
            },
            {"id": "slow_background_modification", "status": "frozen_ohlcv_and_news_activity"},
            {"id": "coin_local_level_and_cluster_description", "status": "frozen_ready"},
            {"id": "post_event_range_behaviour", "status": "frozen_ready"},
        ],
        "outcome_definitions": {
            "horizons_hours": list(HORIZONS),
            "activity_components": [
                "absolute close return versus causal trailing median",
                "high-low range versus causal trailing median",
                "volume sum versus causal trailing median",
            ],
            "activity_score": "median of the three causal component ratios",
            "unusual_activity": "activity score above its causal trailing-90-day 95th percentile",
            "paired_success": "event score exceeds the median frozen control score",
            "leader": layer1_freeze["definitions"]["leader_definition"],
            "meme_amplification": layer1_freeze["definitions"]["ordinary_meme_sensitivity"],
            "local_levels": {
                "single_levels": [
                    "prior 7d high/low",
                    "prior 30d high/low",
                    "EMA50 and EMA200",
                    "Bollinger20 upper/mid/lower",
                    "rolling VWAP168",
                    "adaptive nearest round number",
                ],
                "near_primary": "event-open distance <= 0.25 causal ATR14",
                "near_sensitivity": "event-open distance <= 0.50 causal ATR14",
                "cluster": "at least two independently named levels within the active distance",
                "single_and_cluster_are_equally_valid": True,
            },
            "post_event_range": layer1_freeze["definitions"]["post_event_ranges"],
        },
        "decision_rules": layer1_freeze["definitions"]["layer2_decision_rules"],
        "cohorts": cohorts,
        "research_boundary": {
            "signed_direction_only_after_observed_market_leader": True,
            "event_sign_claim": False,
            "ordinary_timestamp_direction": False,
            "trades_or_profit": False,
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "layer1_freeze": artifact(LAYER1_FREEZE),
            "layer1_result": artifact(LAYER1_RESULT),
            "context_snapshot": artifact(CONTEXT_SNAPSHOT),
        },
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer2_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer2_outcome_blind_freeze":
            raise ValueError("Existing Layer 2 freeze result is not terminal.")
        return result
    _, layer1_freeze, fomc = load_layer1()
    context = add_causal_news_thresholds(load_context_activity())
    news = select_news_episodes(context)
    events, collisions = combine_event_catalogs(fomc, news)
    controls = build_control_map(events, context)
    coverage = coverage_inventory(events, layer1_freeze["cohorts"])
    freeze = freeze_document(
        layer1_freeze=layer1_freeze,
        events=events,
        collisions=collisions,
        controls=controls,
        coverage=coverage,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUT_ROOT / "layer2_event_catalog.csv"
    collisions_path = OUTPUT_ROOT / "layer2_excluded_event_collisions.csv"
    controls_path = OUTPUT_ROOT / "layer2_control_map.csv"
    coverage_path = OUTPUT_ROOT / "layer2_pair_event_coverage.csv"
    freeze_path = OUTPUT_ROOT / "layer2_freeze.json"
    g0.atomic_write_csv(events, events_path)
    g0.atomic_write_csv(collisions, collisions_path)
    g0.atomic_write_csv(controls, controls_path)
    g0.atomic_write_csv(coverage, coverage_path)
    g0.atomic_write_json(freeze, freeze_path)
    result = {
        "schema_version": 1,
        "status": "completed_layer2_outcome_blind_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "event_count": len(events),
        "control_rows": len(controls),
        "event_sources": freeze["event_counts"],
        "artifacts": {
            "freeze": artifact(freeze_path),
            "events": artifact(events_path),
            "collisions": artifact(collisions_path),
            "controls": artifact(controls_path),
            "coverage": artifact(coverage_path),
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "ready_not_executed", "outcomes_read": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
