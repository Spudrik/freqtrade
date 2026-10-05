# ruff: noqa: S101

"""Deterministic unit tests for the semantic-story outcome analysis helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_outcome_direct as direct,
)


def _pair_metric(
    *,
    signed_return: float = 0.01,
    activity_score: float = 1.30,
    control_count: int = 12,
) -> dict[str, float | int]:
    """Return the small metric mapping consumed by ``_scope_metrics``."""
    return {
        "signed_return": signed_return,
        "abs_return_ratio": activity_score,
        "range_ratio": activity_score,
        "volume_ratio": activity_score,
        "activity_score": activity_score,
        "control_count": control_count,
    }


def _hourly_close_frame(
    start: pd.Timestamp, closes: list[float]
) -> DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    return DataFrame({"close": closes}, index=index)


def test_aligned_hour_normalizes_timezone_and_ceils_partial_hours() -> None:
    assert direct.aligned_hour("2026-09-05T12:00:00Z") == pd.Timestamp(
        "2026-09-05T12:00:00Z"
    )
    assert direct.aligned_hour("2026-09-05T12:00:00.001Z") == pd.Timestamp(
        "2026-09-05T13:00:00Z"
    )
    assert direct.aligned_hour("2026-09-05T14:30:00+02:00") == pd.Timestamp(
        "2026-09-05T13:00:00Z"
    )
    assert direct.aligned_hour("2026-09-05T12:30:00") == pd.Timestamp(
        "2026-09-05T13:00:00Z"
    )


def test_sign_and_safe_rate_handle_zero_missing_and_empty_inputs() -> None:
    assert direct.sign(2.0) == 1
    assert direct.sign(-0.001) == -1
    assert direct.sign(0.0) == 0
    assert direct.sign(None) == 0
    assert direct.sign(np.nan) == 0

    assert direct.safe_rate(pd.Series([True, False, np.nan])) == pytest.approx(0.5)
    assert math.isnan(direct.safe_rate(pd.Series(dtype=float)))
    assert math.isnan(direct.safe_rate(pd.Series(["not-a-rate"])))


def test_causal_prior_return_uses_completed_30_day_window_only() -> None:
    anchor = pd.Timestamp("2026-09-01T00:00:00Z")
    hours = direct.PRIOR_TREND_HOURS
    # The expected period is [anchor - (hours + 1), anchor - 1h].
    closes = np.linspace(100.0, 125.0, hours + 1).tolist()
    frame = _hourly_close_frame(
        anchor - pd.Timedelta(hours=hours + 1), closes + [10_000.0]
    )

    assert direct.causal_prior_return(frame, anchor) == pytest.approx(0.25)
    # The event candle is deliberately extreme; it must not enter the prior trend.
    assert frame.index[-1] == anchor

    gapped = frame.drop(index=anchor - pd.Timedelta(hours=hours))
    assert math.isnan(direct.causal_prior_return(gapped, anchor))


def test_complete_hourly_window_rejects_missing_candles() -> None:
    anchor = pd.Timestamp("2026-09-05T12:00:00Z")
    frame = _hourly_close_frame(
        anchor - pd.Timedelta(hours=25), [100.0] * 27
    )

    assert direct._has_complete_hourly_window(frame, anchor, horizon=2)
    missing = frame.drop(index=anchor - pd.Timedelta(hours=3))
    assert not direct._has_complete_hourly_window(missing, anchor, horizon=2)


def test_scope_metrics_abstain_until_the_scope_member_floor_is_met() -> None:
    one_pair = {"a": _pair_metric()}
    assert direct._scope_metrics(
        "established_alts", ("a", "b", "c", "d"), one_pair
    ) is None
    assert direct._scope_metrics("memes", ("a", "b", "c"), one_pair) is None

    valid_rows = {
        pair: _pair_metric(signed_return=index * 0.01, activity_score=1.2 + index)
        for index, pair in enumerate(("a", "b", "c", "d"), start=1)
    }
    scoped = direct._scope_metrics(
        "established_alts", tuple(valid_rows), valid_rows
    )
    assert scoped is not None
    assert scoped["member_count"] == 4
    assert scoped["signed_return"] == pytest.approx(0.025)
    assert scoped["activity_score"] == pytest.approx(3.7)


def test_scope_prior_return_abstains_when_fewer_than_minimum_members_are_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {pair: DataFrame() for pair in ("a", "b", "c")}
    anchor = pd.Timestamp("2026-09-05T12:00:00Z")
    values = iter((0.10, math.nan, 0.30))
    monkeypatch.setattr(
        direct,
        "causal_prior_return",
        lambda _frame, _anchor: next(values),
    )

    assert math.isnan(
        direct._scope_prior_return(frames, tuple(frames), anchor, minimum_members=3)
    )

    values = iter((0.10, math.nan, 0.30))
    assert direct._scope_prior_return(
        frames, tuple(frames), anchor, minimum_members=2
    ) == pytest.approx(0.20)


def test_control_direction_rate_collapses_pairs_to_one_observation_per_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = tuple(
        pd.Timestamp("2026-08-01T00:00:00Z") + pd.Timedelta(days=offset)
        for offset in range(3)
    )
    frames = {"a": object(), "b": object()}
    calls: list[tuple[object, pd.Timestamp]] = []

    monkeypatch.setattr(
        direct,
        "_has_complete_hourly_window",
        lambda _frame, _anchor, _horizon: True,
    )

    def fake_window_metrics(frame: object, anchor: pd.Timestamp, _horizon: int):
        calls.append((frame, pd.Timestamp(anchor)))
        return {
            "signed_return": 0.01
            if pd.Timestamp(anchor) != anchors[-1]
            else -0.01
        }

    monkeypatch.setattr(direct.independent, "window_metrics", fake_window_metrics)
    count, rate = direct._scope_control_direction_rate(
        frames,
        tuple(frames),
        anchors,
        horizon=1,
        predicted_direction=1,
        minimum_members=2,
    )

    # Two pair observations are combined into one market outcome for each anchor.
    assert len(calls) == len(anchors) * len(frames)
    assert count == 3
    assert rate == pytest.approx(2 / 3)

    assert direct._scope_control_direction_rate(
        frames, tuple(frames), anchors, 1, predicted_direction=0, minimum_members=2
    ) == (0, math.nan)


def test_direction_controls_rotate_story_sign_within_each_partition_only() -> None:
    development, validation = direct.frozen.PARTITIONS
    rows = DataFrame.from_records(
        [
            # Development predictions in time order: 1, 1, -1.
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": development,
                "market_anchor_utc": pd.Timestamp("2026-08-01T01:00:00Z"),
                "predicted_direction": 1,
                "actual_direction": 1,
            },
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": development,
                "market_anchor_utc": pd.Timestamp("2026-08-01T02:00:00Z"),
                "predicted_direction": 1,
                "actual_direction": 1,
            },
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": development,
                "market_anchor_utc": pd.Timestamp("2026-08-01T03:00:00Z"),
                "predicted_direction": -1,
                "actual_direction": -1,
            },
            # Validation predictions in time order: 1, -1, 1.
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": validation,
                "market_anchor_utc": pd.Timestamp("2026-08-02T01:00:00Z"),
                "predicted_direction": 1,
                "actual_direction": -1,
            },
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": validation,
                "market_anchor_utc": pd.Timestamp("2026-08-02T02:00:00Z"),
                "predicted_direction": -1,
                "actual_direction": -1,
            },
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": validation,
                "market_anchor_utc": pd.Timestamp("2026-08-02T03:00:00Z"),
                "predicted_direction": 1,
                "actual_direction": -1,
            },
        ]
    )

    output = direct.add_direction_controls(rows)
    for partition, expected in (
        (development, [-1, 1, 1]),
        (validation, [1, 1, -1]),
    ):
        observed = (
            output.loc[output["whole_event_partition"].eq(partition)]
            .sort_values("market_anchor_utc")
            ["rotated_prediction"]
            .tolist()
        )
        assert observed == expected

    # The development majority is calculated from development rows and carried
    # to validation; validation outcomes cannot change that reference direction.
    assert output["development_majority_direction"].eq(1).all()
    validation_majority = output.loc[
        output["whole_event_partition"].eq(validation), "majority_direction_success"
    ]
    assert not validation_majority.any()


def _activity_rows(
    *,
    all_success_counts: tuple[int, int] = (6, 6),
    moderate_success_counts: tuple[int, int] = (8, 8),
) -> DataFrame:
    records: list[dict[str, object]] = []
    routes = (
        ("all_cooled_stories", all_success_counts, "minor"),
        ("moderate_or_major", moderate_success_counts, "major"),
    )
    for route_id, success_counts, severity in routes:
        for partition, success_count in zip(
            direct.frozen.PARTITIONS, success_counts, strict=True
        ):
            for index in range(10):
                success = index < success_count
                records.append(
                    {
                        "route_id": route_id,
                        "market_scope": "btc",
                        "horizon_hours": 1,
                        "route_event_id": f"{route_id}-{partition}-{index}",
                        "whole_event_partition": partition,
                        "reaction_success": success,
                        "activity_score": 1.30 if success else 1.00,
                        "severity": severity,
                    }
                )
    return DataFrame.from_records(records)


def test_synthetic_activity_routes_pass_when_both_partitions_and_lift_pass() -> None:
    summary = direct.summarize_activity(_activity_rows())

    all_stories = summary.loc[summary["route_id"].eq("all_cooled_stories")].iloc[0]
    moderate = summary.loc[summary["route_id"].eq("moderate_or_major")].iloc[0]
    assert all_stories["development_reaction_rate"] == pytest.approx(0.60)
    assert all_stories["validation_reaction_rate"] == pytest.approx(0.60)
    assert bool(all_stories["meets_activity_rule"])
    assert moderate["development_reaction_rate"] == pytest.approx(0.80)
    assert moderate["validation_reaction_rate"] == pytest.approx(0.80)
    assert bool(moderate["meets_activity_rule"])


@pytest.mark.parametrize(
    ("all_counts", "moderate_counts", "route_id"),
    [
        ((5, 6), (8, 8), "all_cooled_stories"),
        ((6, 6), (5, 8), "moderate_or_major"),
    ],
)
def test_synthetic_activity_routes_fail_when_a_partition_breaks_the_rule(
    all_counts: tuple[int, int],
    moderate_counts: tuple[int, int],
    route_id: str,
) -> None:
    summary = direct.summarize_activity(
        _activity_rows(
            all_success_counts=all_counts,
            moderate_success_counts=moderate_counts,
        )
    )
    row = summary.loc[summary["route_id"].eq(route_id)].iloc[0]
    assert not bool(row["meets_activity_rule"])


def _direction_rows(
    *,
    success_counts: tuple[int, int] = (4, 4),
    include_abstention: bool = True,
) -> DataFrame:
    records: list[dict[str, object]] = []
    for partition, success_count in zip(
        direct.frozen.PARTITIONS, success_counts, strict=True
    ):
        for index in range(5):
            records.append(
                {
                    "route_id": "positive_or_negative",
                    "market_scope": "btc",
                    "horizon_hours": 1,
                    "route_event_id": f"signed-{partition}-{index}",
                    "whole_event_partition": partition,
                    "issued_direction_call": True,
                    "direction_success": index < success_count,
                    "prior_30d_trend_success": False,
                    "majority_direction_success": False,
                    "rotated_direction_success": False,
                    "matched_control_direction_count": 12,
                    "matched_control_direction_rate": 0.40,
                }
            )
    if include_abstention:
        records.append(
            {
                "route_id": "positive_or_negative",
                "market_scope": "btc",
                "horizon_hours": 1,
                "route_event_id": "signed-abstain",
                "whole_event_partition": direct.frozen.PARTITIONS[0],
                "issued_direction_call": False,
                "direction_success": np.nan,
                "prior_30d_trend_success": np.nan,
                "majority_direction_success": np.nan,
                "rotated_direction_success": np.nan,
                "matched_control_direction_count": 0,
                "matched_control_direction_rate": np.nan,
            }
        )
    return DataFrame.from_records(records)


def test_synthetic_direction_pass_keeps_abstentions_out_of_the_call_rate() -> None:
    summary = direct.summarize_direction(_direction_rows())
    row = summary.iloc[0]

    assert row["issued_events"] == 10
    assert row["abstentions"] == 1
    assert row["direction_success_rate"] == pytest.approx(0.80)
    assert row["strongest_comparator_rate"] == pytest.approx(0.40)
    assert bool(row["meets_direction_rule"])


def test_synthetic_direction_fails_below_overall_and_partition_accuracy_floor() -> None:
    summary = direct.summarize_direction(
        _direction_rows(success_counts=(2, 2), include_abstention=False)
    )
    row = summary.iloc[0]

    assert row["issued_events"] == 10
    assert row["direction_success_rate"] == pytest.approx(0.40)
    assert not bool(row["meets_direction_rule"])
