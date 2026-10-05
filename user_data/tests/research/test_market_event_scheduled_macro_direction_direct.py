# ruff: noqa: S101

"""Focused synthetic tests for the scheduled-macro direction evaluator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_direct as direct,
)


def _minute_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    values = np.arange(periods, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values + 0.5,
            "volume": np.ones(periods),
        }
    )


def test_complete_future_and_pre_event_windows_reject_minute_gaps() -> None:
    frame = _minute_frame("2025-01-01 12:00:00", 12)
    value, reason = direct.future_return(frame, 5, 4)
    assert reason is None
    assert value == pytest.approx(frame.iloc[8]["close"] / frame.iloc[5]["open"] - 1)
    assert direct.pre_event_return(frame, 5, 4) == pytest.approx(
        frame.iloc[5]["open"] / frame.iloc[1]["open"] - 1
    )

    gapped = frame.drop(index=7).reset_index(drop=True)
    value, reason = direct.future_return(gapped, 5, 4)
    assert np.isnan(value)
    assert reason == "minute_gap_in_future_window"
    assert np.isnan(direct.pre_event_return(gapped, 8, 4))


def test_matched_controls_skip_blocked_weeks_and_keep_complete_windows() -> None:
    anchor = pd.Timestamp("2025-07-01 13:30:00", tz="UTC")
    records: list[pd.DataFrame] = []
    for weeks in range(1, 20):
        records.append(_minute_frame(str(anchor - pd.Timedelta(weeks=weeks)), 2))
    frame = pd.concat(records, ignore_index=True).sort_values("date").reset_index(drop=True)
    positions = direct.position_by_date(frame)
    controls = direct.matched_ordinary_returns(
        frame,
        positions,
        anchor=anchor,
        horizon=2,
        blocked_anchors=[anchor - pd.Timedelta(weeks=1)],
    )
    assert len(controls) == direct.frozen.CONTROL_COUNT
    first_position = positions[anchor - pd.Timedelta(weeks=2)]
    expected, reason = direct.future_return(frame, first_position, 2)
    assert reason is None
    assert controls[0] == pytest.approx(expected)


def test_missing_anchor_and_zero_return_are_explicit_abstentions(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "market.feather"
    path.write_bytes(b"synthetic")
    frame = _minute_frame("2025-01-01 13:30:00", 70)
    frame.loc[:, "open"] = 100.0
    frame.loc[:, "close"] = 100.0
    monkeypatch.setattr(direct.g0, "ohlcv_path", lambda pair, timeframe: path)
    monkeypatch.setattr(direct.g0, "load_ohlcv", lambda candidate: frame.copy())
    monkeypatch.setattr(direct.frozen, "ASSETS", ("BTC/USDT:USDT",))
    monkeypatch.setattr(direct.frozen, "HORIZONS_MINUTES", (5,))
    catalog = pd.DataFrame(
        [
            {
                "event_id": "zero",
                "release_cluster_id": "a",
                "event_family": "employment",
                "source_series_id": "PAYEMS",
                "anchor_utc": frame.iloc[0]["date"],
                "whole_event_partition": direct.frozen.EVALUATION_PARTITIONS[0],
                "predicted_crypto_direction": 1,
                "previous_release_predicted_direction": -1,
            },
            {
                "event_id": "missing",
                "release_cluster_id": "b",
                "event_family": "employment",
                "source_series_id": "PAYEMS",
                "anchor_utc": pd.Timestamp("2025-02-01 13:30:00", tz="UTC"),
                "whole_event_partition": direct.frozen.EVALUATION_PARTITIONS[1],
                "predicted_crypto_direction": -1,
                "previous_release_predicted_direction": 1,
            },
        ]
    )
    rows, coverage = direct.extract_direction_rows(catalog, blocked_anchors=[])
    reasons = rows.set_index("event_id")["abstention_reason"].to_dict()
    assert reasons == {"zero": "zero_future_return", "missing": "missing_anchor_candle"}
    assert rows["direction_hit"].isna().all()
    assert len(coverage) == 1


def _control_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for partition, actuals in (
        (direct.frozen.EVALUATION_PARTITIONS[0], [1, 1, 1, -1]),
        (direct.frozen.EVALUATION_PARTITIONS[1], [-1, -1, 1, -1]),
    ):
        for index, actual in enumerate(actuals):
            records.append(
                {
                    "event_id": f"{partition}-{index}",
                    "event_family": "employment",
                    "pair": "BTC/USDT:USDT",
                    "horizon_minutes": 5,
                    "whole_event_partition": partition,
                    "anchor_utc": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(days=index),
                    "predicted_direction": 1 if index % 2 == 0 else -1,
                    "actual_direction": actual,
                }
            )
    return pd.DataFrame.from_records(records)


def test_rotation_stays_within_partition_and_majority_uses_development_only() -> None:
    rows = direct.add_rotated_and_majority_controls(_control_rows())
    development = rows.loc[
        rows["whole_event_partition"].eq(direct.frozen.EVALUATION_PARTITIONS[0])
    ].sort_values("anchor_utc")
    validation = rows.loc[
        rows["whole_event_partition"].eq(direct.frozen.EVALUATION_PARTITIONS[1])
    ].sort_values("anchor_utc")
    assert development["rotated_predicted_direction"].tolist() == [-1, 1, -1, 1]
    assert validation["rotated_predicted_direction"].tolist() == [-1, 1, -1, 1]
    assert rows["development_majority_direction"].eq(1).all()


def _summary_for_decision(**overrides: float) -> pd.DataFrame:
    base = {
        "event_family": "employment",
        "source_series_id": "PAYEMS",
        "pair": "BTC/USDT:USDT",
        "horizon_minutes": 5,
        "events_total": 12,
        "events_usable": 12,
        "events_abstained": 0,
        "accuracy": 0.60,
        "development_majority_accuracy": 0.50,
        "pretrend_accuracy": 0.50,
        "previous_release_accuracy": 0.50,
        "rotated_sign_accuracy": 0.50,
        "ordinary_paired_event_accuracy": 0.60,
        "ordinary_period_accuracy": 0.50,
        "ordinary_paired_lift": 0.10,
        "ordinary_control_eligible_events": 12,
        "median_response_return": 0.0,
    }
    records = []
    for partition in (
        *direct.frozen.EVALUATION_PARTITIONS,
        "development_plus_validation",
    ):
        row = {**base, "partition": partition, **overrides}
        records.append(row)
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize(
    "control_column",
    [
        "development_majority_accuracy",
        "pretrend_accuracy",
        "previous_release_accuracy",
        "rotated_sign_accuracy",
        "ordinary_period_accuracy",
    ],
)
def test_candidate_must_beat_each_possible_strongest_control(
    control_column: str,
) -> None:
    passing = direct.route_decisions(_summary_for_decision())
    assert bool(passing.iloc[0]["candidate"])
    overrides = {control_column: 0.58}
    if control_column == "ordinary_period_accuracy":
        overrides["ordinary_paired_lift"] = 0.02
    failing = direct.route_decisions(_summary_for_decision(**overrides))
    assert not bool(failing.iloc[0]["candidate"])


def test_route_enforces_overall_partition_and_coverage_thresholds() -> None:
    overall_failure = direct.route_decisions(_summary_for_decision(accuracy=0.54))
    assert not bool(overall_failure.iloc[0]["candidate"])

    partition_failure = _summary_for_decision()
    partition_failure.loc[
        partition_failure["partition"].eq(direct.frozen.EVALUATION_PARTITIONS[1]),
        "accuracy",
    ] = 0.49
    assert not bool(direct.route_decisions(partition_failure).iloc[0]["candidate"])

    coverage_failure = _summary_for_decision()
    coverage_failure.loc[
        coverage_failure["partition"].eq(direct.frozen.EVALUATION_PARTITIONS[0]),
        "ordinary_control_eligible_events",
    ] = direct.frozen.MINIMUM_EVENTS_PER_PARTITION - 1
    decision = direct.route_decisions(coverage_failure).iloc[0]
    assert not bool(decision["candidate"])
    assert decision["decision"] == "coverage_limited"


def test_family_requires_both_assets_at_the_same_horizon() -> None:
    decisions = pd.DataFrame(
        [
            {
                "event_family": "employment",
                "pair": "BTC/USDT:USDT",
                "horizon_minutes": 5,
                "candidate": True,
                "decision": "provisional_requires_familywide_validation",
            },
            {
                "event_family": "employment",
                "pair": "ETH/USDT:USDT",
                "horizon_minutes": 15,
                "candidate": True,
                "decision": "provisional_requires_familywide_validation",
            },
        ]
    )
    narrow = direct.family_decisions(decisions)
    assert narrow.iloc[0]["decision"].startswith("narrow_one_market")
    decisions.loc[1, "horizon_minutes"] = 5
    shared = direct.family_decisions(decisions)
    assert shared.iloc[0]["decision"].startswith("provisional_two_market")


def test_load_rejects_changed_frozen_catalogue(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path = tmp_path / "freeze.json"
    catalog_path = tmp_path / "catalog.csv"
    result_path = tmp_path / "result.json"
    freeze_path.write_text(
        json.dumps(
            {
                "status": "frozen_scheduled_macro_direction_before_signed_outcomes",
                "outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    catalog_path.write_text("event_id,anchor_utc\na,2025-01-01T00:00:00Z\n", encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "status": "completed_scheduled_macro_direction_freeze",
                "outcomes_read": False,
                "artifacts": {
                    "freeze": {"sha256": direct.g0.sha256_file(freeze_path)},
                    "catalog": {"sha256": direct.g0.sha256_file(catalog_path)},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(direct, "FREEZE_RESULT_PATH", result_path)
    monkeypatch.setattr(direct, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(direct, "CATALOG_PATH", catalog_path)
    direct.load_frozen_inputs()
    catalog_path.write_text("event_id,anchor_utc\nb,2025-01-01T00:00:00Z\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        direct.load_frozen_inputs()


def test_no_execute_does_not_open_frozen_or_market_inputs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        direct,
        "load_frozen_inputs",
        lambda: (_ for _ in ()).throw(AssertionError("inputs were opened")),
    )
    assert direct.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready_not_executed"
    assert output["outcomes_will_be_read"] is True
