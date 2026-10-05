"""Test frozen CPI leader transmission and CPI-conditioned meme response."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as shared,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.RESULT_PATH
FREEZE_PATH = frozen.FREEZE_PATH
EVENTS_PATH = frozen.CPI_LINK_EVENTS_PATH
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "cpi_links_direct_20260907a"
CONTROL_PATH = OUTPUT_ROOT / "cpi_link_control_map.csv"
TRANSMISSION_METRICS_PATH = OUTPUT_ROOT / "transmission_pair_metrics.parquet"
TRANSMISSION_ROWS_PATH = OUTPUT_ROOT / "transmission_rows.parquet"
TRANSMISSION_SUMMARY_PATH = OUTPUT_ROOT / "transmission_summary.csv"
MEME_METRICS_PATH = OUTPUT_ROOT / "meme_pair_metrics.parquet"
MEME_COIN_ROWS_PATH = OUTPUT_ROOT / "meme_coin_rows.parquet"
MEME_EVENT_ROWS_PATH = OUTPUT_ROOT / "meme_event_rows.parquet"
MEME_SUMMARY_PATH = OUTPUT_ROOT / "meme_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "cpi_link_route_decisions.csv"
COVERAGE_PATH = OUTPUT_ROOT / "cpi_link_market_coverage.csv"
REPORT_PATH = OUTPUT_ROOT / "cpi_link_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "cpi_link_result.json"

LEADER_PAIRS = {
    "btc_first_move": "BTC/USDT:USDT",
    "eth_first_move": "ETH/USDT:USDT",
}
TRANSMISSION_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    *frozen.ESTABLISHED_FOLLOWERS,
)
CONTROL_COUNT = 12
CONTROL_SEARCH_WEEKS = 52
COLLISION_HOURS = 4
MINIMUM_CONTROLS = 6
BETA_LOOKBACK_HOURS = 24 * 90
BETA_MINIMUM_HOURS = 24 * 45
TRANSMISSION_WINDOWS = (
    (0, 5),
    (0, 15),
    (5, 15),
    (5, 30),
    (5, 60),
    (15, 30),
    (15, 60),
)
MEME_WINDOWS = (*TRANSMISSION_WINDOWS, (0, 30), (0, 60))


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_layer2_breadth_outcome_blind_freeze":
        raise ValueError("Layer 2 breadth freeze result is not terminal")
    if freeze.get("status") != "frozen_layer2_breadth_batch_before_market_outcomes":
        raise ValueError("Layer 2 breadth definitions are not frozen")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Layer 2 breadth freeze unexpectedly opened outcomes")
    if result.get("profit_used") or freeze.get("profit_used"):
        raise ValueError("Layer 2 breadth freeze unexpectedly used profit")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("cpi_link_events", EVENTS_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen CPI link artifact changed: {path}")
    events = pd.read_csv(EVENTS_PATH)
    events["anchor_utc"] = pd.to_datetime(events["anchor_utc"], utc=True)
    for column in ("leader_eligible", "meme_eligible"):
        events[column] = events[column].astype(bool)
    return freeze, events


def build_control_map(events: DataFrame) -> DataFrame:
    external_anchors, _ = frozen.treasury_activity.load_external_anchors()
    event_anchors = list(events["anchor_utc"])
    blocked = [*external_anchors, *event_anchors]
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        rank = 0
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            candidate = pd.Timestamp(event.anchor_utc) - pd.Timedelta(weeks=weeks)
            if (
                shared.minimum_event_distance_hours(blocked, candidate)
                <= COLLISION_HOURS
            ):
                continue
            rank += 1
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_anchor_utc": event.anchor_utc,
                    "control_anchor_utc": candidate,
                    "control_rank": rank,
                    "control_type": "prior_same_weekday_and_utc_clock",
                }
            )
            if rank >= CONTROL_COUNT:
                break
    controls = DataFrame.from_records(rows)
    if controls.duplicated(["event_id", "control_rank"]).any():
        raise ValueError("CPI link control identifiers are not unique")
    return controls


def sample_table(
    events: DataFrame, controls: DataFrame, *, eligibility_column: str
) -> DataFrame:
    selected = events.loc[events[eligibility_column]].copy()
    event_rows = selected[
        [
            "event_id",
            "anchor_utc",
            "whole_event_partition",
            "meme_partition",
        ]
    ].copy()
    event_rows["sample_type"] = "event"
    event_rows["control_rank"] = 0
    event_rows["sample_anchor_utc"] = event_rows["anchor_utc"]
    event_rows = event_rows.drop(columns="anchor_utc")
    control_rows = controls.loc[controls["event_id"].isin(selected["event_id"])].copy()
    metadata = selected[
        ["event_id", "whole_event_partition", "meme_partition"]
    ].drop_duplicates("event_id")
    control_rows = control_rows.merge(
        metadata, on="event_id", how="left", validate="many_to_one"
    )
    control_rows["sample_type"] = "control"
    control_rows["sample_anchor_utc"] = control_rows["control_anchor_utc"]
    control_rows = control_rows[
        [
            "event_id",
            "whole_event_partition",
            "meme_partition",
            "sample_type",
            "control_rank",
            "sample_anchor_utc",
        ]
    ]
    output = pd.concat([event_rows, control_rows], ignore_index=True)
    output["sample_id"] = (
        output["event_id"].astype(str)
        + "|"
        + output["sample_type"].astype(str)
        + "|"
        + output["control_rank"].astype(str)
    )
    if output["sample_id"].duplicated().any():
        raise ValueError("CPI link sample identifiers are not unique")
    return output


def subwindow_metrics(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    start_minute: int,
    end_minute: int,
) -> tuple[str, dict[str, float] | None]:
    if end_minute <= start_minute:
        raise ValueError("CPI subwindow must end after it starts")
    start = pd.Timestamp(anchor) + pd.Timedelta(minutes=start_minute)
    position = positions.get(start)
    if position is None:
        return "missing_anchor", None
    length = end_minute - start_minute
    dates = frame.iloc[position : position + length]["date"]
    if len(dates) != length:
        return "incomplete_future_window", None
    if pd.Timestamp(dates.iloc[0]) != start:
        return "timestamp_gap", None
    expected = pd.date_range(dates.iloc[0], periods=length, freq="1min")
    if not dates.reset_index(drop=True).equals(pd.Series(expected)):
        return "timestamp_gap", None
    window = frame.iloc[position : position + length]
    opening = float(window.iloc[0]["open"])
    if not np.isfinite(opening) or opening <= 0:
        return "invalid_open", None
    closing = float(window.iloc[-1]["close"])
    return "usable", {
        "return": closing / opening - 1.0,
        "abs_return": abs(closing / opening - 1.0),
        "range": float((window["high"].max() - window["low"].min()) / opening),
        "volume": float(window["volume"].sum()),
    }


def extract_pair_metrics(
    samples: DataFrame,
    pairs: Iterable[str],
    *,
    windows: Sequence[tuple[int, int]],
) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicates = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = shared.position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
                "duplicate_timestamps_removed": duplicates,
            }
        )
        for sample in samples.itertuples(index=False):
            for start, end in windows:
                status, metrics = subwindow_metrics(
                    frame,
                    positions,
                    anchor=pd.Timestamp(sample.sample_anchor_utc),
                    start_minute=start,
                    end_minute=end,
                )
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "event_id": sample.event_id,
                        "sample_type": sample.sample_type,
                        "control_rank": sample.control_rank,
                        "sample_anchor_utc": sample.sample_anchor_utc,
                        "whole_event_partition": sample.whole_event_partition,
                        "meme_partition": sample.meme_partition,
                        "pair": pair,
                        "start_minute": start,
                        "end_minute": end,
                        "eligibility_status": status,
                        "return": metrics["return"] if metrics else np.nan,
                        "abs_return": metrics["abs_return"] if metrics else np.nan,
                        "range": metrics["range"] if metrics else np.nan,
                        "volume": metrics["volume"] if metrics else np.nan,
                    }
                )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def metric_lookup(metrics: DataFrame) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    output: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in metrics.to_dict(orient="records"):
        output[(row["sample_id"], row["pair"], row["start_minute"], row["end_minute"])] = {
            "return": row["return"],
            "abs_return": row["abs_return"],
            "range": row["range"],
            "volume": row["volume"],
            "eligibility_status": row["eligibility_status"],
        }
    return output


def comparison_controls(
    samples: DataFrame,
    lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    event_id: str,
    sample_id: str,
    pair: str,
    start: int,
    end: int,
) -> list[Mapping[str, Any]]:
    controls: list[Mapping[str, Any]] = []
    selected = samples.loc[
        samples["event_id"].eq(event_id) & samples["sample_type"].eq("control")
    ]
    for row in selected.itertuples(index=False):
        if row.sample_id == sample_id:
            continue
        metric = lookup.get((row.sample_id, pair, start, end))
        if metric is not None and metric["eligibility_status"] == "usable":
            controls.append(metric)
    return controls


def normalized_initial_move(
    samples: DataFrame,
    lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    event_id: str,
    sample_id: str,
    pair: str,
    window: int,
) -> tuple[float, float]:
    metric = lookup.get((sample_id, pair, 0, window))
    if metric is None or metric["eligibility_status"] != "usable":
        return np.nan, np.nan
    controls = comparison_controls(
        samples,
        lookup,
        event_id=event_id,
        sample_id=sample_id,
        pair=pair,
        start=0,
        end=window,
    )
    values = [float(row["abs_return"]) for row in controls]
    baseline = float(np.median(values)) if len(values) >= MINIMUM_CONTROLS else np.nan
    ratio = (
        float(metric["abs_return"]) / baseline
        if np.isfinite(baseline) and baseline > 0
        else np.nan
    )
    return float(metric["return"]), ratio


def leader_call(
    samples: DataFrame,
    lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    event_id: str,
    sample_id: str,
    leader: str,
    window: int,
) -> tuple[bool, float, float]:
    if leader in LEADER_PAIRS:
        value, ratio = normalized_initial_move(
            samples,
            lookup,
            event_id=event_id,
            sample_id=sample_id,
            pair=LEADER_PAIRS[leader],
            window=window,
        )
        sign = float(np.sign(value)) if np.isfinite(value) else np.nan
        issued = bool(np.isfinite(ratio) and ratio > 1.0 and sign != 0)
        return issued, sign, ratio
    if leader != "btc_eth_agreement":
        raise ValueError(f"Unknown CPI leader candidate: {leader}")
    btc, btc_ratio = normalized_initial_move(
        samples,
        lookup,
        event_id=event_id,
        sample_id=sample_id,
        pair="BTC/USDT:USDT",
        window=window,
    )
    eth, eth_ratio = normalized_initial_move(
        samples,
        lookup,
        event_id=event_id,
        sample_id=sample_id,
        pair="ETH/USDT:USDT",
        window=window,
    )
    signs = (np.sign(btc), np.sign(eth))
    issued = bool(
        all(np.isfinite(value) and value > 1.0 for value in (btc_ratio, eth_ratio))
        and signs[0] != 0
        and signs[0] == signs[1]
    )
    finite_ratios = [value for value in (btc_ratio, eth_ratio) if np.isfinite(value)]
    strength = float(np.median(finite_ratios)) if finite_ratios else np.nan
    return issued, float(signs[0]) if issued else np.nan, strength


def later_return(
    lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    sample_id: str,
    pair: str,
    start: int,
    end: int,
) -> float:
    metric = lookup.get((sample_id, pair, start, end))
    if metric is None or metric["eligibility_status"] != "usable":
        return np.nan
    return float(metric["return"])


def transmission_rows(samples: DataFrame, metrics: DataFrame) -> DataFrame:
    lookup = metric_lookup(metrics)
    rows: list[dict[str, Any]] = []
    follower_pairs = (*frozen.ESTABLISHED_FOLLOWERS, "established_group_median")
    for sample in samples.itertuples(index=False):
        for leader in (*LEADER_PAIRS, "btc_eth_agreement"):
            for leader_window in (5, 15):
                issued, sign, strength = leader_call(
                    samples,
                    lookup,
                    event_id=sample.event_id,
                    sample_id=sample.sample_id,
                    leader=leader,
                    window=leader_window,
                )
                for follower_end in (15, 30, 60):
                    if follower_end <= leader_window:
                        continue
                    later_by_pair = {
                        pair: later_return(
                            lookup,
                            sample_id=sample.sample_id,
                            pair=pair,
                            start=leader_window,
                            end=follower_end,
                        )
                        for pair in frozen.ESTABLISHED_FOLLOWERS
                    }
                    initial_by_pair = {
                        pair: later_return(
                            lookup,
                            sample_id=sample.sample_id,
                            pair=pair,
                            start=0,
                            end=leader_window,
                        )
                        for pair in frozen.ESTABLISHED_FOLLOWERS
                    }
                    for follower in follower_pairs:
                        if follower == "established_group_median":
                            later_values = [
                                value for value in later_by_pair.values() if np.isfinite(value)
                            ]
                            initial_values = [
                                value
                                for value in initial_by_pair.values()
                                if np.isfinite(value)
                            ]
                            later = (
                                float(np.median(later_values))
                                if len(later_values) == len(frozen.ESTABLISHED_FOLLOWERS)
                                else np.nan
                            )
                            initial = (
                                float(np.median(initial_values))
                                if len(initial_values) == len(frozen.ESTABLISHED_FOLLOWERS)
                                else np.nan
                            )
                        else:
                            later = later_by_pair[follower]
                            initial = initial_by_pair[follower]
                        usable = issued and np.isfinite(later)
                        rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "event_id": sample.event_id,
                                "sample_type": sample.sample_type,
                                "control_rank": sample.control_rank,
                                "whole_event_partition": sample.whole_event_partition,
                                "leader": leader,
                                "leader_window_minutes": leader_window,
                                "follower_end_minutes": follower_end,
                                "follower": follower,
                                "call_issued": issued,
                                "leader_sign": sign,
                                "leader_strength_ratio": strength,
                                "follower_later_return": later,
                                "direction_success": (
                                    bool(np.sign(later) == sign) if usable and later != 0 else pd.NA
                                ),
                                "follower_own_initial_sign": (
                                    float(np.sign(initial)) if np.isfinite(initial) else np.nan
                                ),
                                "own_initial_direction_success": (
                                    bool(np.sign(later) == np.sign(initial))
                                    if (
                                        usable
                                        and np.isfinite(initial)
                                        and initial != 0
                                        and later != 0
                                    )
                                    else pd.NA
                                ),
                            }
                        )
    return DataFrame.from_records(rows)


def transmission_summary(
    rows: DataFrame, freeze: Mapping[str, Any]
) -> DataFrame:
    rule = freeze["cpi_immediate_leader_transmission"]
    records: list[dict[str, Any]] = []
    keys = [
        "leader",
        "leader_window_minutes",
        "follower_end_minutes",
        "follower",
    ]
    for values, group in rows.groupby(keys, sort=True):
        event = group.loc[
            group["sample_type"].eq("event")
            & group["call_issued"]
            & group["direction_success"].notna()
        ]
        control = group.loc[
            group["sample_type"].eq("control")
            & group["call_issued"]
            & group["direction_success"].notna()
        ]
        event_rate = float(event["direction_success"].mean()) if len(event) else np.nan
        control_rate = (
            float(control["direction_success"].mean()) if len(control) else np.nan
        )
        own_rate = (
            float(event["own_initial_direction_success"].mean())
            if event["own_initial_direction_success"].notna().any()
            else np.nan
        )
        partition_rates = {
            partition: (
                float(
                    event.loc[
                        event["whole_event_partition"].eq(partition),
                        "direction_success",
                    ].mean()
                )
                if event["whole_event_partition"].eq(partition).any()
                else np.nan
            )
            for partition in frozen.PARTITIONS
        }
        enough = len(event) >= int(rule["minimum_issued_whole_events"])
        repeats = all(
            np.isfinite(rate) and rate >= float(rule["minimum_each_partition_rate"])
            for rate in partition_rates.values()
        )
        beats_control = bool(
            np.isfinite(event_rate)
            and np.isfinite(control_rate)
            and event_rate - control_rate >= float(rule["minimum_control_uplift"])
        )
        beats_own = bool(
            np.isfinite(event_rate) and np.isfinite(own_rate) and event_rate > own_rate
        )
        passed = bool(
            enough
            and repeats
            and event_rate >= float(rule["minimum_overall_direction_rate"])
            and beats_control
            and beats_own
        )
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "issued_event_count": len(event),
                "eligible_event_count": int(
                    group.loc[group["sample_type"].eq("event"), "event_id"].nunique()
                ),
                "event_direction_rate": event_rate,
                "development_direction_rate": partition_rates[
                    "development_2021_2023"
                ],
                "validation_direction_rate": partition_rates[
                    "internal_validation_2024_2025"
                ],
                "issued_control_count": len(control),
                "control_direction_rate": control_rate,
                "control_uplift": event_rate - control_rate,
                "follower_own_initial_direction_rate": own_rate,
                "beats_follower_own_first_move": beats_own,
                "meets_frozen_rule": passed,
                "interpretation": (
                    "exploratory_lead_pending_family_shuffle"
                    if passed
                    else "coverage_or_control_rule_not_met"
                ),
            }
        )
    return DataFrame.from_records(records)


def rolling_beta_fit(pair: str, anchors: Sequence[pd.Timestamp]) -> DataFrame:
    btc = g0.load_ohlcv(g0.ohlcv_path("BTC/USDT:USDT", "1h"))[["date", "close"]]
    coin = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))[["date", "close"]]
    btc = btc.sort_values("date").drop_duplicates("date")
    coin = coin.sort_values("date").drop_duplicates("date")
    btc["btc_log_return"] = np.log(btc["close"]).diff()
    coin["coin_log_return"] = np.log(coin["close"]).diff()
    aligned = btc[["date", "btc_log_return"]].merge(
        coin[["date", "coin_log_return"]], on="date", how="inner", validate="one_to_one"
    )
    aligned["available_at"] = aligned["date"] + pd.Timedelta(hours=1)
    records: list[dict[str, Any]] = []
    for anchor in sorted(set(pd.Timestamp(value) for value in anchors)):
        history = aligned.loc[aligned["available_at"].le(anchor)].tail(
            BETA_LOOKBACK_HOURS
        ).dropna()
        if len(history) < BETA_MINIMUM_HOURS:
            continue
        design = np.column_stack(
            [np.ones(len(history)), history["btc_log_return"].to_numpy()]
        )
        target = history["coin_log_return"].to_numpy()
        alpha, beta = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ np.array([alpha, beta])
        records.append(
            {
                "pair": pair,
                "sample_anchor_utc": anchor,
                "alpha_hourly": float(alpha),
                "beta_to_btc": float(beta),
                "residual_std_hourly": float(np.std(residual, ddof=2)),
                "training_hours": len(history),
            }
        )
    return DataFrame.from_records(records)


def activity_against_controls(
    samples: DataFrame,
    lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    event_id: str,
    sample_id: str,
    pair: str,
    end: int,
) -> float:
    metric = lookup.get((sample_id, pair, 0, end))
    if metric is None or metric["eligibility_status"] != "usable":
        return np.nan
    controls = comparison_controls(
        samples,
        lookup,
        event_id=event_id,
        sample_id=sample_id,
        pair=pair,
        start=0,
        end=end,
    )
    score, _ = shared.activity_score(metric, controls)
    return score


def meme_coin_rows(
    samples: DataFrame,
    metrics: DataFrame,
    meme_pairs: Sequence[str],
) -> DataFrame:
    lookup = metric_lookup(metrics)
    fits = pd.concat(
        [
            rolling_beta_fit(pair, list(samples["sample_anchor_utc"]))
            for pair in meme_pairs
        ],
        ignore_index=True,
    )
    fit_lookup = {
        (row.pair, pd.Timestamp(row.sample_anchor_utc)): row
        for row in fits.itertuples(index=False)
    }
    provisional: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        for leader_window in (5, 15):
            issued, sign, strength = leader_call(
                samples,
                lookup,
                event_id=sample.event_id,
                sample_id=sample.sample_id,
                leader="btc_first_move",
                window=leader_window,
            )
            for end in (15, 30, 60):
                if end <= leader_window:
                    continue
                btc_full = later_return(
                    lookup,
                    sample_id=sample.sample_id,
                    pair="BTC/USDT:USDT",
                    start=0,
                    end=end,
                )
                for pair in meme_pairs:
                    full = later_return(
                        lookup,
                        sample_id=sample.sample_id,
                        pair=pair,
                        start=0,
                        end=end,
                    )
                    later = later_return(
                        lookup,
                        sample_id=sample.sample_id,
                        pair=pair,
                        start=leader_window,
                        end=end,
                    )
                    fit = fit_lookup.get((pair, pd.Timestamp(sample.sample_anchor_utc)))
                    residual = np.nan
                    if (
                        fit is not None
                        and np.isfinite(full)
                        and np.isfinite(btc_full)
                        and 1 + full > 0
                        and 1 + btc_full > 0
                    ):
                        expected = float(fit.alpha_hourly) * (end / 60) + float(
                            fit.beta_to_btc
                        ) * math.log1p(btc_full)
                        residual = math.log1p(full) - expected
                    provisional.append(
                        {
                            "sample_id": sample.sample_id,
                            "event_id": sample.event_id,
                            "sample_type": sample.sample_type,
                            "control_rank": sample.control_rank,
                            "sample_anchor_utc": sample.sample_anchor_utc,
                            "meme_partition": sample.meme_partition,
                            "leader_window_minutes": leader_window,
                            "response_end_minutes": end,
                            "pair": pair,
                            "call_issued": issued,
                            "btc_initial_sign": sign,
                            "btc_initial_strength_ratio": strength,
                            "meme_activity_score": activity_against_controls(
                                samples,
                                lookup,
                                event_id=sample.event_id,
                                sample_id=sample.sample_id,
                                pair=pair,
                                end=end,
                            ),
                            "meme_full_return": full,
                            "meme_later_return": later,
                            "beta_residual_log_return": residual,
                            "direction_success": (
                                bool(np.sign(later) == sign)
                                if issued and np.isfinite(later) and later != 0
                                else pd.NA
                            ),
                        }
                    )
    rows = DataFrame.from_records(provisional)
    rows["abs_beta_residual"] = rows["beta_residual_log_return"].abs()
    rows["residual_ratio"] = np.nan
    group_keys = ["event_id", "pair", "response_end_minutes"]
    for _, indexes in rows.groupby(group_keys, sort=False).groups.items():
        indexes = list(indexes)
        controls = rows.loc[indexes].loc[rows.loc[indexes, "sample_type"].eq("control")]
        for index in indexes:
            current = rows.at[index, "abs_beta_residual"]
            if not np.isfinite(current):
                continue
            baseline_rows = controls.loc[controls["sample_id"].ne(rows.at[index, "sample_id"])]
            baseline_values = baseline_rows["abs_beta_residual"].dropna()
            if len(baseline_values) < MINIMUM_CONTROLS:
                continue
            baseline = float(baseline_values.median())
            if baseline > 0:
                rows.at[index, "residual_ratio"] = float(current) / baseline
    rows["coin_reaction_success"] = (
        rows["meme_activity_score"].gt(1.0) & rows["residual_ratio"].gt(1.0)
    ).where(rows[["meme_activity_score", "residual_ratio"]].notna().all(axis=1))
    return rows


def meme_event_rows(rows: DataFrame, required_coins: int) -> DataFrame:
    keys = [
        "sample_id",
        "event_id",
        "sample_type",
        "control_rank",
        "meme_partition",
        "leader_window_minutes",
        "response_end_minutes",
    ]
    grouped = rows.groupby(keys, dropna=False, sort=True)
    output = grouped.agg(
        call_issued=("call_issued", "all"),
        btc_initial_sign=("btc_initial_sign", "first"),
        btc_initial_strength_ratio=("btc_initial_strength_ratio", "first"),
        coin_count=("residual_ratio", "count"),
        median_meme_activity_score=("meme_activity_score", "median"),
        median_residual_ratio=("residual_ratio", "median"),
        coin_reaction_rate=("coin_reaction_success", "mean"),
        cohort_later_return=("meme_later_return", "median"),
    ).reset_index()
    output["complete_cohort"] = output["coin_count"].eq(required_coins)
    output["reaction_success"] = (
        output["complete_cohort"]
        & output["median_meme_activity_score"].gt(1.0)
        & output["median_residual_ratio"].gt(1.0)
    )
    output["direction_success"] = (
        np.sign(output["cohort_later_return"]).eq(output["btc_initial_sign"])
        & output["cohort_later_return"].ne(0)
    ).where(output["call_issued"] & output["complete_cohort"])
    output["joint_success"] = (
        output["reaction_success"] & output["direction_success"].eq(True)
    ).where(output["call_issued"] & output["complete_cohort"])
    return output


def meme_summary(
    event_rows: DataFrame, freeze: Mapping[str, Any]
) -> DataFrame:
    rule = freeze["cpi_meme_response"]
    records: list[dict[str, Any]] = []
    for values, group in event_rows.groupby(
        ["leader_window_minutes", "response_end_minutes"], sort=True
    ):
        event = group.loc[
            group["sample_type"].eq("event")
            & group["call_issued"]
            & group["complete_cohort"]
        ]
        control = group.loc[
            group["sample_type"].eq("control")
            & group["call_issued"]
            & group["complete_cohort"]
        ]
        reaction_rate = float(event["reaction_success"].mean()) if len(event) else np.nan
        direction_rate = (
            float(event["direction_success"].mean()) if len(event) else np.nan
        )
        joint_rate = float(event["joint_success"].mean()) if len(event) else np.nan
        control_joint = (
            float(control["joint_success"].mean()) if len(control) else np.nan
        )
        partition_counts = event["meme_partition"].value_counts().to_dict()
        enough = len(event) >= int(rule["minimum_complete_whole_events"])
        records.append(
            {
                "leader_window_minutes": values[0],
                "response_end_minutes": values[1],
                "eligible_cpi_events": int(
                    group.loc[group["sample_type"].eq("event"), "event_id"].nunique()
                ),
                "issued_complete_event_count": len(event),
                "development_events": int(
                    partition_counts.get("meme_development_2025", 0)
                ),
                "validation_events": int(
                    partition_counts.get("meme_internal_validation_2026_h1", 0)
                ),
                "reaction_success_rate": reaction_rate,
                "conditional_direction_rate": direction_rate,
                "joint_success_rate": joint_rate,
                "issued_complete_control_count": len(control),
                "control_joint_success_rate": control_joint,
                "joint_control_uplift": joint_rate - control_joint,
                "meets_55_reaction_floor": bool(
                    enough
                    and reaction_rate >= float(rule["minimum_reaction_or_direction_rate"])
                ),
                "meets_55_direction_floor": bool(
                    enough
                    and direction_rate >= float(rule["minimum_reaction_or_direction_rate"])
                ),
                "meets_55_joint_floor_and_control": bool(
                    enough
                    and joint_rate >= float(rule["minimum_reaction_or_direction_rate"])
                    and np.isfinite(control_joint)
                    and joint_rate > control_joint
                ),
                "coverage_status": (
                    "testable_recent_exploratory" if enough else "coverage_limited"
                ),
            }
        )
    return DataFrame.from_records(records)


def route_decisions(
    transmission: DataFrame, memes: DataFrame
) -> DataFrame:
    records: list[dict[str, Any]] = []
    for follower in (*frozen.ESTABLISHED_FOLLOWERS, "established_group_median"):
        group = transmission.loc[transmission["follower"].eq(follower)]
        passed = group.loc[group["meets_frozen_rule"]]
        records.append(
            {
                "route": f"cpi_first_move_to_{follower}",
                "verdict": (
                    "retained_exploratory_lead_pending_family_shuffle"
                    if len(passed)
                    else "parked_no_control_resistant_lead"
                ),
                "passing_cells": len(passed),
                "plain_result": (
                    f"{len(passed)} leader-window combinations beat matched controls "
                    "and the follower's own first move."
                ),
            }
        )
    meme_passed = memes.loc[memes["meets_55_joint_floor_and_control"]]
    meme_testable = memes.loc[memes["coverage_status"].eq("testable_recent_exploratory")]
    records.append(
        {
            "route": "cpi_conditioned_top10_meme_response",
            "verdict": (
                "retained_recent_exploratory_lead_pending_family_shuffle"
                if len(meme_passed)
                else (
                    "parked_no_joint_control_resistant_lead"
                    if len(meme_testable)
                    else "coverage_limited"
                )
            ),
            "passing_cells": len(meme_passed),
            "plain_result": (
                f"{len(meme_passed)} windows reached the joint 55% floor after BTC "
                "conditioning and ordinary-sensitivity adjustment."
            ),
        }
    )
    return DataFrame.from_records(records)


def render_report(
    decisions: DataFrame,
    transmission: DataFrame,
    memes: DataFrame,
) -> str:
    lines = [
        "# CPI Leader, Follower, And Meme Review",
        "",
        "This review uses only movement after a 5- or 15-minute BTC/ETH observation "
        "window for directional claims. It abstains when the first move is not larger "
        "than its clean ordinary-time control.",
        "",
        "## Decisions",
        "",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(f"- `{row.route}`: `{row.verdict}` — {row.plain_result}")
    lines.extend(
        [
            "",
            f"Transmission combinations assessed: `{len(transmission)}`.",
            f"Meme window combinations assessed: `{len(memes)}`.",
            "",
            "BTC/ETH candidates, overlapping horizons, and three established coins are "
            "related views, not independent discoveries. Any survivor remains exploratory "
            "until a whole-family shuffle and later whole-event confirmation.",
            "",
            "No profit, entry, exit, or trading promotion was tested.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_cpi_leader_and_meme_link_review":
        raise ValueError("Existing CPI link result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing CPI link artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    freeze, events = load_frozen_inputs()
    controls = build_control_map(events)
    transmission_samples = sample_table(
        events, controls, eligibility_column="leader_eligible"
    )
    transmission_metrics, transmission_coverage = extract_pair_metrics(
        transmission_samples,
        TRANSMISSION_PAIRS,
        windows=TRANSMISSION_WINDOWS,
    )
    transmission_detail = transmission_rows(
        transmission_samples, transmission_metrics
    )
    transmission_result = transmission_summary(transmission_detail, freeze)

    meme_samples = sample_table(events, controls, eligibility_column="meme_eligible")
    meme_pairs = list(freeze["cpi_meme_response"]["cohort"])
    meme_metrics, meme_coverage = extract_pair_metrics(
        meme_samples,
        ("BTC/USDT:USDT", *meme_pairs),
        windows=MEME_WINDOWS,
    )
    meme_coin_detail = meme_coin_rows(meme_samples, meme_metrics, meme_pairs)
    meme_event_detail = meme_event_rows(meme_coin_detail, len(meme_pairs))
    meme_result = meme_summary(meme_event_detail, freeze)
    decisions = route_decisions(transmission_result, meme_result)
    coverage = pd.concat(
        [
            transmission_coverage.assign(test_block="cpi_established_transmission"),
            meme_coverage.assign(test_block="cpi_meme_response"),
        ],
        ignore_index=True,
    ).drop_duplicates(["test_block", "pair"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(controls, CONTROL_PATH)
    g0.atomic_write_parquet(transmission_metrics, TRANSMISSION_METRICS_PATH)
    g0.atomic_write_parquet(transmission_detail, TRANSMISSION_ROWS_PATH)
    g0.atomic_write_csv(transmission_result, TRANSMISSION_SUMMARY_PATH)
    g0.atomic_write_parquet(meme_metrics, MEME_METRICS_PATH)
    g0.atomic_write_parquet(meme_coin_detail, MEME_COIN_ROWS_PATH)
    g0.atomic_write_parquet(meme_event_detail, MEME_EVENT_ROWS_PATH)
    g0.atomic_write_csv(meme_result, MEME_SUMMARY_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    REPORT_PATH.write_text(
        render_report(decisions, transmission_result, meme_result),
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "schema_version": 1,
        "status": "completed_cpi_leader_and_meme_link_review",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": True,
        "profit_used": False,
        "direction_tested": True,
        "transmission_summary_rows": len(transmission_result),
        "meme_summary_rows": len(meme_result),
        "retained_route_count": int(decisions["verdict"].str.startswith("retained").sum()),
        "artifacts": {
            "controls": artifact(CONTROL_PATH),
            "transmission_metrics": artifact(TRANSMISSION_METRICS_PATH),
            "transmission_rows": artifact(TRANSMISSION_ROWS_PATH),
            "transmission_summary": artifact(TRANSMISSION_SUMMARY_PATH),
            "meme_metrics": artifact(MEME_METRICS_PATH),
            "meme_coin_rows": artifact(MEME_COIN_ROWS_PATH),
            "meme_event_rows": artifact(MEME_EVENT_ROWS_PATH),
            "meme_summary": artifact(MEME_SUMMARY_PATH),
            "decisions": artifact(DECISION_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "freeze": artifact(FREEZE_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "outcomes_will_be_read": True,
                    "profit_will_be_used": False,
                    "direction_scope": "post_cpi_first_move_only",
                    "links": [
                        "cpi_immediate_leader_transmission",
                        "cpi_conditioned_meme_response",
                    ],
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
