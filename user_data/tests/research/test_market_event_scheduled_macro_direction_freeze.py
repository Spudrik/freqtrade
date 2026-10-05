# ruff: noqa: S101

"""Deterministic, outcome-blind tests for scheduled macro direction freezing."""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_freeze as freeze,
)


RETRIEVED_AT = "2026-09-05T12:00:00Z"


def _short_source(values: list[float]) -> dict[str, list[dict[str, object]]]:
    observations = []
    for index, value in enumerate(values):
        observations.append(
            {
                "date": f"2021-{index + 1:02d}-01",
                "realtime_start": f"2021-{index + 1:02d}-15",
                "value": value,
            }
        )
    return {"observations": observations}


def _synthetic_events_and_sources() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Create six events per family in each frozen partition.

    All five families share each monthly release date.  That deliberately makes
    simultaneous announcements visible to the cluster-collapse tests.
    """
    release_dates = [
        *(f"2022-{month:02d}-15" for month in range(1, 7)),
        *(f"2024-{month:02d}-15" for month in range(1, 7)),
    ]
    periods = [f"2021-{month:02d}-01" for month in range(1, 13)]
    records: list[dict[str, object]] = []
    sources: dict[str, dict[str, Any]] = {}
    for family, definition in freeze.broad.SCHEDULED_FAMILIES.items():
        series_id = definition["series_id"]
        observations: list[dict[str, object]] = []
        values = (
            [1000.0 + (index + 1) ** 3 for index in range(14)]
            if series_id == "PAYEMS"
            else [100.0 + (index + 1) ** 2 for index in range(14)]
        )
        for index, value in enumerate(values):
            if index < 2:
                observation_period = f"2020-{index + 11:02d}-01"
                release_date = f"2020-{index + 11:02d}-15"
            else:
                event_index = index - 2
                observation_period = periods[event_index]
                release_date = release_dates[event_index]
            observations.append(
                {
                    "date": observation_period,
                    "realtime_start": release_date,
                    "value": value,
                }
            )
        sources[series_id] = {"observations": observations}

        for event_index, (period, release_date) in enumerate(
            zip(periods, release_dates, strict=True)
        ):
            records.append(
                {
                    "event_id": f"{family}_{period}",
                    "event_family": family,
                    "event_label": family,
                    "whole_event_partition": (
                        freeze.EVALUATION_PARTITIONS[0]
                        if event_index < 6
                        else freeze.EVALUATION_PARTITIONS[1]
                    ),
                    "source_column": series_id,
                    "observation_period": period,
                    "anchor_utc": freeze.broad.release_anchor(release_date),
                }
            )
    return pd.DataFrame.from_records(records), sources


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    return _synthetic_events_and_sources()


@pytest.fixture(scope="module")
def synthetic_catalogue(
    synthetic_inputs: tuple[pd.DataFrame, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    scheduled, sources = synthetic_inputs
    return freeze.build_direction_catalog(scheduled, sources)


@pytest.mark.parametrize(
    ("series_id", "expected_method"),
    [
        (
            "PPIACO",
            "published_index_monthly_rate_acceleration",
        ),
        (
            "PCEPI",
            "published_index_monthly_rate_acceleration",
        ),
        (
            "RSAFS",
            "published_level_monthly_rate_acceleration",
        ),
        (
            "GDPC1",
            "published_level_quarterly_rate_acceleration",
        ),
    ],
)
def test_index_series_use_percent_rate_acceleration_and_inverse_sign(
    series_id: str, expected_method: str
) -> None:
    values = [100.0, 101.0, 104.0, 110.0]
    transformed = freeze.transformed_source_rows(
        _short_source(values), series_id
    )

    first_scored = transformed.iloc[2]
    expected_previous_change = values[1] / values[0] - 1.0
    expected_current_change = values[2] / values[1] - 1.0
    expected_acceleration = expected_current_change - expected_previous_change
    assert freeze.SERIES_METHODS[series_id] == expected_method
    assert first_scored["published_change"] == pytest.approx(expected_current_change)
    assert first_scored["published_acceleration"] == pytest.approx(
        expected_acceleration
    )
    assert first_scored["acceleration_sign"] == 1
    assert first_scored["predicted_crypto_direction"] == -1


def test_payems_uses_payroll_difference_acceleration_and_inverse_sign() -> None:
    values = [1000.0, 1001.0, 1004.0, 1010.0]
    transformed = freeze.transformed_source_rows(
        _short_source(values), "PAYEMS"
    )

    first_scored = transformed.iloc[2]
    assert freeze.SERIES_METHODS["PAYEMS"] == (
        "published_payroll_monthly_change_acceleration"
    )
    assert first_scored["published_change"] == pytest.approx(3.0)
    assert first_scored["published_acceleration"] == pytest.approx(2.0)
    assert first_scored["acceleration_sign"] == 1
    assert first_scored["predicted_crypto_direction"] == -1


def test_every_nonzero_transformed_sign_maps_to_the_inverse_prediction() -> None:
    values = [100.0, 101.0, 104.0, 110.0, 121.0]
    for series_id in freeze.SERIES_METHODS:
        transformed = freeze.transformed_source_rows(
            _short_source(values), series_id
        )
        scored = transformed.loc[transformed["acceleration_sign"].ne(0)]
        assert (
            scored["predicted_crypto_direction"]
            == -scored["acceleration_sign"]
        ).all()


def test_release_date_from_source_must_agree_with_the_frozen_anchor(
    synthetic_inputs: tuple[pd.DataFrame, dict[str, dict[str, Any]]],
) -> None:
    scheduled, sources = synthetic_inputs
    bad_sources = copy.deepcopy(sources)
    bad_sources["PPIACO"]["observations"][2]["realtime_start"] = "2022-02-16"

    with pytest.raises(ValueError, match="PPIACO release dates disagree"):
        freeze.build_direction_catalog(scheduled, bad_sources)


def test_catalogue_counts_families_and_partitions_without_using_market_outcomes(
    synthetic_catalogue: pd.DataFrame,
) -> None:
    assert len(synthetic_catalogue) == 60
    counts = freeze.counts_table(synthetic_catalogue)
    assert len(counts) == len(freeze.SERIES_METHODS) * len(
        freeze.EVALUATION_PARTITIONS
    )
    assert counts["events"].eq(6).all()
    assert set(counts["event_family"]) == set(freeze.broad.SCHEDULED_FAMILIES)
    assert set(counts["whole_event_partition"]) == set(
        freeze.EVALUATION_PARTITIONS
    )
    assert (
        counts["positive_acceleration"] + counts["negative_acceleration"]
    ).eq(counts["events"]).all()


def test_simultaneous_release_dates_share_one_cluster_and_distinct_dates_do_not(
    synthetic_catalogue: pd.DataFrame,
) -> None:
    by_anchor = synthetic_catalogue.groupby("anchor_utc")
    assert by_anchor.size().eq(5).all()
    assert by_anchor["release_cluster_id"].nunique().eq(1).all()
    assert synthetic_catalogue["release_cluster_id"].nunique() == 12
    assert synthetic_catalogue["release_cluster_id"].str.startswith(
        "scheduled_release_"
    ).all()


def test_duplicate_missing_and_zero_direction_inputs_are_rejected(
    synthetic_inputs: tuple[pd.DataFrame, dict[str, dict[str, Any]]],
) -> None:
    scheduled, sources = synthetic_inputs

    duplicate_period = copy.deepcopy(sources)
    duplicate_period["PPIACO"]["observations"][3]["date"] = (
        duplicate_period["PPIACO"]["observations"][2]["date"]
    )
    with pytest.raises(ValueError, match="duplicate initial observation periods"):
        freeze.transformed_source_rows(duplicate_period["PPIACO"], "PPIACO")

    missing_observation = copy.deepcopy(sources)
    del missing_observation["PPIACO"]["observations"][2]
    with pytest.raises(ValueError, match="Could not reconstruct every PPIACO"):
        freeze.build_direction_catalog(scheduled, missing_observation)

    zero_direction = copy.deepcopy(sources)
    first = float(zero_direction["PPIACO"]["observations"][0]["value"])
    second = float(zero_direction["PPIACO"]["observations"][1]["value"])
    zero_direction["PPIACO"]["observations"][2]["value"] = second * second / first
    with pytest.raises(ValueError, match="non-zero frozen direction"):
        freeze.build_direction_catalog(scheduled, zero_direction)

    duplicate_event = scheduled.copy()
    duplicate_event.loc[1, "event_id"] = duplicate_event.loc[0, "event_id"]
    with pytest.raises(ValueError, match="event IDs are not unique"):
        freeze.build_direction_catalog(duplicate_event, sources)


def test_frozen_document_keeps_method_controls_and_outcome_boundaries(
    synthetic_catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = freeze.counts_table(synthetic_catalogue)
    parent_result = {
        "status": "completed_broad_event_relevance_freeze",
        "outcomes_read": False,
        "profit_used": False,
    }
    monkeypatch.setattr(
        freeze,
        "artifact",
        lambda path: {"path": str(path), "sha256": "synthetic"},
    )

    document = freeze.freeze_document(synthetic_catalogue, counts, parent_result)

    assert document["status"] == (
        "frozen_scheduled_macro_direction_before_signed_outcomes"
    )
    assert document["outcomes_read"] is False
    assert document["profit_used"] is False
    assert document["event_rows"] == 60
    assert document["release_clocks"] == 12
    assert document["series_methods"] == freeze.SERIES_METHODS
    assert document["families"] == list(freeze.broad.SCHEDULED_FAMILIES)
    assert document["assets"] == list(freeze.ASSETS)
    assert document["horizons_minutes"] == list(freeze.HORIZONS_MINUTES)
    assert document["retention_rule"]["minimum_events_per_partition"] == 6
    assert document["control_selection"] == {
        "search_weeks": 26,
        "controls_per_event": 12,
        "minimum_controls": 10,
        "exclude_within_hours_of_any_scheduled_event": 24,
    }
    assert len(document["controls"]) == 5
    assert "timestamped forecast" in document["critical_limit"]
    assert document["parent_contracts"]["parent_result"] == parent_result


def test_source_validation_rejects_missing_fields_and_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="source is missing fields"):
        freeze.transformed_source_rows(
            {"observations": [{"date": "2021-01-01", "value": 1.0}]},
            "PPIACO",
        )

    with pytest.raises(ValueError, match="missing published value"):
        freeze.transformed_source_rows(
            {
                "observations": [
                    {
                        "date": "2021-01-01",
                        "realtime_start": "2021-01-15",
                        "value": ".",
                    }
                ]
            },
            "PPIACO",
        )


def test_no_execute_path_reports_ready_without_reading_parent_or_market_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_if_called(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("execute must not run without --execute")

    monkeypatch.setattr(freeze, "execute", fail_if_called)
    assert freeze.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "ready_not_executed",
        "families": 5,
        "outcomes_will_be_read": False,
    }
