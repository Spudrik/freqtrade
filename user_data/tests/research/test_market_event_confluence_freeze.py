from __future__ import annotations

# ruff: noqa: S101
import json

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_confluence_freeze as freeze,
)


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-04T14:00:00Z", "2026-09-04T14:00:00Z"),
        ("2026-09-04T14:00:01Z", "2026-09-04T15:00:00Z"),
        ("2026-09-04T10:15:00-04:00", "2026-09-04T15:00:00Z"),
    ],
)
def test_decision_hour_normalizes_timezone_and_ceils(value: str, expected: str) -> None:
    result = freeze.decision_hour(pd.Timestamp(value))

    assert result == pd.Timestamp(expected)
    assert str(result.tz) == "UTC"


@pytest.mark.parametrize(
    ("change", "relation", "expected"),
    [
        (2.0, "same_direction", 1),
        (-2.0, "same_direction", -1),
        (2.0, "opposite_direction", -1),
        (-2.0, "opposite_direction", 1),
        (2.0, "unknown", 0),
        (0.0, "same_direction", 0),
        (float("nan"), "same_direction", 0),
        (float("inf"), "same_direction", 0),
    ],
)
def test_signed_contribution_applies_relation_or_abstains(
    change: float, relation: str, expected: int
) -> None:
    assert freeze.signed_contribution(change, relation) == expected


def test_causal_surface_includes_current_and_inside_24h_but_not_lower_bound() -> None:
    target = _utc("2026-09-04 12:00:00")
    events = pd.DataFrame(
        [
            {
                "event_id": "exact_lower_bound",
                "event_family": "excluded_family",
                "source_group": "excluded_source",
                "decision_hour_utc": target - pd.Timedelta(hours=24),
                "signed_contribution": 1,
                "severity_multiple": 9.0,
            },
            {
                "event_id": "just_inside",
                "event_family": "inside_family",
                "source_group": "inside_source",
                "decision_hour_utc": target - pd.Timedelta(hours=24)
                + pd.Timedelta(microseconds=1),
                "signed_contribution": 1,
                "severity_multiple": 1.5,
            },
            {
                "event_id": "current_event",
                "event_family": "current_family",
                "source_group": "current_source",
                "decision_hour_utc": target,
                "signed_contribution": 1,
                "severity_multiple": 2.5,
            },
            {
                "event_id": "too_old",
                "event_family": "old_family",
                "source_group": "old_source",
                "decision_hour_utc": target - pd.Timedelta(hours=25),
                "signed_contribution": -1,
                "severity_multiple": 100.0,
            },
            {
                "event_id": "future_event",
                "event_family": "future_family",
                "source_group": "future_source",
                "decision_hour_utc": target + pd.Timedelta(hours=1),
                "signed_contribution": -1,
                "severity_multiple": 100.0,
            },
        ]
    )

    surface = freeze.causal_anchor_surface(events)
    row = surface.loc[surface["anchor_utc"].eq(target)].iloc[0]

    assert row["component_event_ids"] == "current_event;just_inside"
    assert row["family_count_24h"] == 2
    assert row["source_group_count_24h"] == 2
    assert row["positive_signed_count_24h"] == 2
    assert row["negative_signed_count_24h"] == 0
    assert row["signed_net_24h"] == 2
    assert row["max_measurable_severity_multiple_24h"] == 2.5


def _route_surface() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_utc": [
                _utc("2026-01-01 00:00:00") + pd.Timedelta(hours=100 * index)
                for index in range(7)
            ],
            "family_count_24h": [1, 2, 3, 1, 2, 2, 2],
            "source_group_count_24h": [1, 1, 2, 1, 2, 2, 2],
            "positive_signed_count_24h": [0, 0, 0, 0, 2, 1, 3],
            "negative_signed_count_24h": [0, 0, 0, 0, 0, 2, 2],
            "max_measurable_severity_multiple_24h": [2.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    ("route_id", "expected"),
    [
        ("two_plus_families", [False, True, True, False, True, True, True]),
        ("three_plus_families", [False, False, True, False, False, False, False]),
        ("two_plus_source_groups", [False, False, True, False, True, True, True]),
        ("aligned_positive_market_signs", [False, False, False, False, True, False, True]),
        ("aligned_negative_market_signs", [False, False, False, False, False, True, False]),
        ("isolated_extreme_measurable_shock", [True, False, False, False, False, False, False]),
        ("isolated_family_reference", [True, False, False, True, False, False, False]),
    ],
)
def test_route_mask_definitions(route_id: str, expected: list[bool]) -> None:
    actual = freeze.route_mask(_route_surface(), route_id)

    assert actual.tolist() == expected


def test_route_mask_rejects_unknown_route() -> None:
    with pytest.raises(ValueError, match="Unknown route"):
        freeze.route_mask(_route_surface(), "not_a_route")


def test_select_with_cooldown_keeps_exact_boundary_and_rejects_inside() -> None:
    candidates = pd.DataFrame(
        {
            "anchor_utc": [
                _utc("2026-01-01 00:00:00"),
                _utc("2026-01-03 23:59:59"),
                _utc("2026-01-04 00:00:00"),
            ],
            "value": [1, 2, 3],
        }
    )

    selected = freeze.select_with_cooldown(candidates)

    assert selected["value"].tolist() == [1, 3]


def test_build_route_catalog_attaches_partitions_and_unique_route_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _route_surface()
    monkeypatch.setattr(
        freeze.layer1,
        "whole_event_partition",
        lambda value: f"synthetic_{pd.Timestamp(value).year}",
    )

    catalog = freeze.build_route_catalog(surface)

    assert set(catalog["route_id"]) == set(freeze.ROUTES)
    assert catalog["route_event_id"].is_unique
    assert catalog["whole_event_partition"].eq("synthetic_2026").all()
    expected_predictions = catalog["route_id"].map(
        {route_id: definition["prediction"] for route_id, definition in freeze.ROUTES.items()}
    )
    assert catalog["signed_prediction"].tolist() == expected_predictions.tolist()
    assert catalog.apply(
        lambda row: row["route_event_id"].startswith(row["route_id"] + "_"),
        axis=1,
    ).all()


def test_count_table_counts_rows_by_route_and_whole_event_partition() -> None:
    catalog = pd.DataFrame(
        {
            "route_id": ["route_a", "route_a", "route_a", "route_b"],
            "route_label": ["A", "A", "A", "B"],
            "whole_event_partition": [
                "development",
                "development",
                "validation",
                "development",
            ],
        }
    )

    counts = freeze.count_table(catalog)
    actual = {
        (row.route_id, row.whole_event_partition): row.episodes
        for row in counts.itertuples()
    }

    assert actual == {
        ("route_a", "development"): 2,
        ("route_a", "validation"): 1,
        ("route_b", "development"): 1,
    }


def test_verify_frozen_catalog_rejects_changed_catalog_hash(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("event_id\nsynthetic\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "outcomes_read": False,
                "artifacts": {"catalog": {"sha256": "wrong-hash"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen catalog changed"):
        freeze.verify_frozen_catalog(result_path, catalog_path)
