from __future__ import annotations

# ruff: noqa: S101
import json

import pandas as pd
import pytest
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_confluence_direct as direct,
)


def _summary_row(
    *,
    partition: str,
    route_id: str = "candidate",
    route_label: str = "Candidate",
    market_scope: str = "btc",
    horizon_hours: int = 1,
    whole_events: int = 10,
    reaction_success_rate: float = 0.60,
    median_activity_multiple: float = 1.30,
    direction_calls: int = 10,
    direction_success_rate: float = 0.61,
    recent_trend_success_rate: float = 0.55,
    majority_direction_success_rate: float = 0.56,
    rotated_direction_success_rate: float = 0.57,
    joint_success_rate: float = 0.60,
) -> dict[str, object]:
    return {
        "route_id": route_id,
        "route_label": route_label,
        "market_scope": market_scope,
        "horizon_hours": horizon_hours,
        "whole_event_partition": partition,
        "whole_events": whole_events,
        "reaction_success_rate": reaction_success_rate,
        "median_activity_multiple": median_activity_multiple,
        "direction_calls": direction_calls,
        "direction_success_rate": direction_success_rate,
        "recent_trend_success_rate": recent_trend_success_rate,
        "majority_direction_success_rate": majority_direction_success_rate,
        "rotated_direction_success_rate": rotated_direction_success_rate,
        "joint_success_rate": joint_success_rate,
    }


def _partition_summary_rows(**kwargs: object) -> dict[str, pd.Series]:
    return {
        partition: pd.Series(_summary_row(partition=partition, **kwargs))
        for partition in direct.REQUIRED_PARTITIONS
    }


def _result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "supported_cells": 1,
        "activity_cells": [],
        "enhanced_cells": [],
        "direction_cells": [],
        "joint_cells": [],
    }
    result.update(overrides)
    return result


def test_load_frozen_inputs_verifies_hashes_and_normalizes_mixed_timestamps(
    tmp_path, monkeypatch
) -> None:
    freeze_path = tmp_path / "freeze.json"
    catalog_path = tmp_path / "catalog.csv"
    sources_path = tmp_path / "sources.csv"
    result_path = tmp_path / "result.json"

    freeze_payload = {
        "status": "frozen_event_confluence_before_crypto_outcomes",
        "outcomes_read": False,
    }
    freeze_path.write_text(json.dumps(freeze_payload), encoding="utf-8")
    catalog_path.write_text(
        "route_id,anchor_utc\n"
        "synthetic,2026-01-01T00:00:00Z\n"
        "synthetic,2026-01-02T00:00:00.123456-05:00\n",
        encoding="utf-8",
    )
    sources_path.write_text(
        "decision_hour_utc\n2026-01-01T00:00:00.123Z\n",
        encoding="utf-8",
    )
    artifacts = {
        name: {"sha256": direct.g0.sha256_file(path)}
        for name, path in {
            "freeze": freeze_path,
            "catalog": catalog_path,
            "sources": sources_path,
        }.items()
    }
    result_path.write_text(
        json.dumps(
            {
                "status": "completed_event_confluence_freeze",
                "outcomes_read": False,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(direct, "FREEZE_RESULT_PATH", result_path)
    monkeypatch.setattr(direct, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(direct, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(direct, "SOURCE_PATH", sources_path)

    loaded_freeze, catalog, sources = direct.load_frozen_inputs()

    assert loaded_freeze == freeze_payload
    assert catalog["anchor_utc"].dt.tz == pd.Timestamp.now(tz="UTC").tz
    assert str(catalog.loc[1, "anchor_utc"]) == "2026-01-02 05:00:00.123456+00:00"
    assert str(sources.loc[0, "decision_hour_utc"]) == (
        "2026-01-01 00:00:00.123000+00:00"
    )

    catalog_path.write_text(catalog_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen event-confluence artifact changed"):
        direct.load_frozen_inputs()


def test_prior_week_controls_exclude_inside_and_exact_boundary_and_cap_at_twelve() -> None:
    anchor = pd.Timestamp("2026-09-04T14:00:00Z")
    blocked = [
        anchor - pd.Timedelta(weeks=1) + pd.Timedelta(hours=1),
        anchor - pd.Timedelta(weeks=2) + pd.Timedelta(hours=72),
        anchor - pd.Timedelta(weeks=3) + pd.Timedelta(hours=72, minutes=1),
    ]

    controls = direct.prior_week_control_anchors(anchor, blocked)

    assert len(controls) == direct.CONTROL_COUNT == 12
    assert controls[0] == anchor - pd.Timedelta(weeks=3)
    assert all(
        control.weekday() == anchor.weekday() and control.hour == anchor.hour
        for control in controls
    )
    assert all(
        abs(control - event) > pd.Timedelta(hours=72)
        for control in controls
        for event in blocked
    )


def test_partition_rows_requires_both_partitions_and_keeps_first_row() -> None:
    rows = DataFrame(
        {
            "whole_event_partition": [
                "other",
                "development_2021_2023",
                "development_2021_2023",
                "internal_validation_2024_2025",
                "internal_validation_2024_2025",
            ],
            "marker": ["other", "dev_first", "dev_later", "val_first", "val_later"],
        }
    )

    partitions = direct.partition_rows(rows)

    assert partitions is not None
    assert partitions["development_2021_2023"]["marker"] == "dev_first"
    assert partitions["internal_validation_2024_2025"]["marker"] == "val_first"
    assert direct.partition_rows(rows.iloc[:3]) is None


@pytest.mark.parametrize(
    ("partition", "field", "value"),
    [
        ("development_2021_2023", "whole_events", 9),
        ("internal_validation_2024_2025", "reaction_success_rate", 0.549),
        ("development_2021_2023", "median_activity_multiple", 1.199),
    ],
)
def test_activity_pass_requires_thresholds_in_both_partitions(
    partition: str, field: str, value: object
) -> None:
    rows = _partition_summary_rows()

    assert direct.activity_pass(rows)
    rows[partition][field] = value

    assert not direct.activity_pass(rows)


def test_exceeds_isolated_requires_both_metrics_to_improve_in_both_partitions() -> None:
    rows = _partition_summary_rows(
        reaction_success_rate=0.70,
        median_activity_multiple=1.40,
    )
    reference = _partition_summary_rows(
        reaction_success_rate=0.60,
        median_activity_multiple=1.20,
    )

    assert direct.exceeds_isolated(rows, reference)

    rows["development_2021_2023"]["median_activity_multiple"] = 1.20
    assert not direct.exceeds_isolated(rows, reference)

    rows = _partition_summary_rows(
        reaction_success_rate=0.70,
        median_activity_multiple=1.40,
    )
    rows["internal_validation_2024_2025"]["reaction_success_rate"] = 0.60
    assert not direct.exceeds_isolated(rows, reference)


@pytest.mark.parametrize(
    ("partition", "field", "value"),
    [
        ("development_2021_2023", "direction_calls", 9),
        ("internal_validation_2024_2025", "direction_success_rate", 0.549),
        ("development_2021_2023", "recent_trend_success_rate", 0.60),
        ("internal_validation_2024_2025", "majority_direction_success_rate", 0.60),
        ("development_2021_2023", "rotated_direction_success_rate", 0.60),
    ],
)
def test_direction_pass_requires_count_accuracy_and_edge_over_each_comparator(
    partition: str, field: str, value: object
) -> None:
    rows = _partition_summary_rows()

    assert direct.direction_pass(rows)
    rows[partition][field] = value

    assert not direct.direction_pass(rows)


def test_route_cell_results_checks_partitions_controls_and_joint_result() -> None:
    records: list[dict[str, object]] = []
    for partition in direct.REQUIRED_PARTITIONS:
        records.append(
            _summary_row(
                partition=partition,
                route_id="candidate",
                market_scope="btc",
                horizon_hours=1,
                reaction_success_rate=0.70,
                median_activity_multiple=1.40,
                joint_success_rate=0.60,
            )
        )
        records.append(
            _summary_row(
                partition=partition,
                route_id="candidate",
                market_scope="eth",
                horizon_hours=4,
                direction_success_rate=0.54,
                reaction_success_rate=0.60,
                median_activity_multiple=1.30,
                joint_success_rate=0.40,
            )
        )
        records.append(
            _summary_row(
                partition=partition,
                route_id=direct.ISOLATED_REFERENCE,
                route_label="Isolated reference",
                market_scope="btc",
                horizon_hours=1,
                reaction_success_rate=0.60,
                median_activity_multiple=1.20,
            )
        )
        records.append(
            _summary_row(
                partition=partition,
                route_id=direct.ISOLATED_REFERENCE,
                route_label="Isolated reference",
                market_scope="eth",
                horizon_hours=4,
                reaction_success_rate=0.55,
                median_activity_multiple=1.10,
            )
        )
    summary = DataFrame.from_records(records)

    result = direct.route_cell_results(
        "candidate", summary.loc[summary["route_id"].eq("candidate")], summary
    )

    assert result == {
        "supported_cells": 2,
        "activity_cells": ["btc:1h", "eth:4h"],
        "enhanced_cells": ["btc:1h", "eth:4h"],
        "direction_cells": ["btc:1h"],
        "joint_cells": ["btc:1h"],
    }


@pytest.mark.parametrize(
    ("route_id", "result", "expected"),
    [
        (
            "candidate",
            _result(
                joint_cells=["btc:1h"],
                direction_cells=["btc:1h"],
                enhanced_cells=["btc:1h"],
                activity_cells=["btc:1h"],
            ),
            "repeatable_joint_activity_and_direction_lead",
        ),
        (
            "candidate",
            _result(
                direction_cells=["btc:1h"],
                enhanced_cells=["btc:1h"],
                activity_cells=["btc:1h"],
            ),
            "repeatable_direction_lead",
        ),
        (
            "candidate",
            _result(enhanced_cells=["btc:1h"], activity_cells=["btc:1h"]),
            "repeatable_overlap_activity_lead",
        ),
        (
            direct.ISOLATED_REFERENCE,
            _result(activity_cells=["btc:1h"]),
            "isolated_event_activity_reference",
        ),
        (
            "candidate",
            _result(activity_cells=["btc:1h"]),
            "activity_repeats_but_overlap_adds_no_value",
        ),
        (
            "candidate",
            _result(supported_cells=0),
            "insufficient_two_partition_coverage",
        ),
        ("candidate", _result(), "weak_or_inconsistent"),
    ],
)
def test_route_verdict_precedence_and_coverage(
    route_id: str, result: dict[str, object], expected: str
) -> None:
    assert direct.route_verdict(route_id, result) == expected


def test_add_result_flags_integrates_majority_control_and_abstentions() -> None:
    rows = DataFrame(
        [
            {
                "route_id": "candidate",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": "development_2021_2023",
                "anchor_utc": pd.Timestamp("2022-01-01T00:00:00Z"),
                "activity_score": 1.30,
                "signed_return": 0.02,
                "signed_prediction": 1,
                "recent_24h_return": 0.01,
            },
            {
                "route_id": "candidate",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": "development_2021_2023",
                "anchor_utc": pd.Timestamp("2022-01-02T00:00:00Z"),
                "activity_score": 1.00,
                "signed_return": 0.01,
                "signed_prediction": 0,
                "recent_24h_return": 0.01,
            },
            {
                "route_id": "candidate",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": "development_2021_2023",
                "anchor_utc": pd.Timestamp("2022-01-03T00:00:00Z"),
                "activity_score": 1.30,
                "signed_return": -0.02,
                "signed_prediction": -1,
                "recent_24h_return": -0.01,
            },
            {
                "route_id": "candidate",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": "internal_validation_2024_2025",
                "anchor_utc": pd.Timestamp("2024-01-01T00:00:00Z"),
                "activity_score": 1.30,
                "signed_return": 0.03,
                "signed_prediction": 1,
                "recent_24h_return": 0.01,
            },
            {
                "route_id": "candidate",
                "market_scope": "btc",
                "horizon_hours": 1,
                "whole_event_partition": "internal_validation_2024_2025",
                "anchor_utc": pd.Timestamp("2024-01-02T00:00:00Z"),
                "activity_score": 1.30,
                "signed_return": -0.03,
                "signed_prediction": 0,
                "recent_24h_return": -0.01,
            },
        ]
    )

    flagged = direct.add_result_flags(rows)

    assert flagged["actual_direction"].tolist() == [1, 1, -1, 1, -1]
    assert flagged["issued_direction_call"].tolist() == [True, False, True, True, False]
    assert bool(flagged.loc[0, "direction_success"])
    assert pd.isna(flagged.loc[1, "direction_success"])
    assert bool(flagged.loc[0, "majority_direction_success"])
    assert not bool(flagged.loc[2, "majority_direction_success"])
    assert pd.isna(flagged.loc[1, "joint_success"])
    assert bool(flagged.loc[0, "joint_success"])
