from __future__ import annotations

# ruff: noqa: S101
import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_cross_asset_relevance_direct as cross_direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_cross_asset_relevance_freeze as cross_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_freeze as frozen,
)


def test_release_anchor_respects_new_york_daylight_saving() -> None:
    assert frozen.release_anchor("2024-01-10") == pd.Timestamp("2024-01-10T13:30:00Z")
    assert frozen.release_anchor("2024-07-10") == pd.Timestamp("2024-07-10T12:30:00Z")


def test_scheduled_catalog_uses_complete_families_without_direction() -> None:
    documents = {
        definition["series_id"]: {
            "observations": [
                {
                    "realtime_start": "2024-01-10",
                    "realtime_end": "2024-02-10",
                    "date": "2023-12-01",
                    "value": "1.0",
                }
            ]
        }
        for definition in frozen.SCHEDULED_FAMILIES.values()
    }
    catalog = frozen.build_scheduled_catalog(documents)

    assert len(catalog) == len(frozen.SCHEDULED_FAMILIES)
    assert set(catalog["event_family"]) == set(frozen.SCHEDULED_FAMILIES)
    assert catalog["anchor_utc"].eq(pd.Timestamp("2024-01-10T13:30:00Z")).all()
    assert catalog["source_semantics"].str.contains("direction are not used").all()


def test_causal_topic_threshold_does_not_read_current_value(monkeypatch) -> None:
    monkeypatch.setattr(frozen, "TOPIC_HISTORY_SAME_HOUR_OBSERVATIONS", 3)
    monkeypatch.setattr(frozen, "TOPIC_MINIMUM_SAME_HOUR_OBSERVATIONS", 2)
    dates = pd.date_range("2021-01-01", periods=24 * 6, freq="h", tz="UTC")
    frame = pd.DataFrame({"date": dates})
    base = pd.Series(np.arange(len(frame), dtype=float) % 7 + 1)
    changed = base.copy()
    changed.iloc[-1] = 1_000_000

    first = frozen.causal_topic_threshold(base, frame)
    second = frozen.causal_topic_threshold(changed, frame)

    assert first.iloc[-1] == second.iloc[-1]


def test_topic_selection_applies_family_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(
        frozen,
        "causal_topic_threshold",
        lambda values, frame: pd.Series(0.0, index=frame.index),
    )
    dates = pd.to_datetime(
        ["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z", "2021-01-09T00:00:00Z"]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "ctx_max_source_available_at": dates,
            **{
                definition["column"]: [1.0, 2.0, 3.0]
                for definition in frozen.TOPIC_FAMILIES.values()
            },
        }
    )

    catalog = frozen.select_topic_episodes(frame)

    counts = catalog.groupby("event_family").size()
    assert counts.eq(2).all()


def test_activity_score_is_median_of_three_control_ratios() -> None:
    response = {"abs_return": 2.0, "range": 3.0, "volume": 4.0}
    controls = [
        {"abs_return": 1.0, "range": 1.0, "volume": 2.0}
        for _ in range(direct.MINIMUM_CONTROLS)
    ]

    score, ratios = direct.activity_score(response, controls)

    assert ratios == {"abs_return": 2.0, "range": 3.0, "volume": 2.0}
    assert score == 2.0


def test_family_classification_requires_both_time_partitions() -> None:
    rows = []
    for pair in direct.ASSETS:
        for partition in (
            "development_2021_2023",
            "internal_validation_2024_2025",
        ):
            rows.append(
                {
                    "event_family": "example",
                    "pair": pair,
                    "horizon": 5,
                    "horizon_unit": "minutes",
                    "whole_event_partition": partition,
                    "events": frozen.MINIMUM_PARTITION_EVENTS,
                    "partition_pass": True,
                }
            )
    summary = pd.DataFrame(rows)
    catalog = pd.DataFrame(
        [
            {
                "event_group": "scheduled_release",
                "event_family": "example",
                "event_label": "Example",
            }
        ]
    )

    verdicts = direct.classify_families(summary, catalog)

    assert verdicts.iloc[0]["verdict"] == "repeatable_activity_lead_both_markets"


def test_cross_asset_anchor_is_after_reported_availability_day() -> None:
    assert cross_frozen.conservative_anchor("2024-07-10") == pd.Timestamp(
        "2024-07-11T00:00:00Z"
    )


def test_cross_asset_threshold_uses_only_earlier_source_changes(monkeypatch) -> None:
    monkeypatch.setattr(cross_frozen, "DAILY_HISTORY", 3)
    monkeypatch.setattr(cross_frozen, "DAILY_MINIMUM_HISTORY", 2)
    definition = {
        "series_id": "TEST",
        "change": "percent",
        "frequency": "daily",
    }
    observations = [
        {
            "date": f"2024-01-{day:02d}",
            "realtime_start": f"2024-01-{day:02d}",
            "value": str(value),
        }
        for day, value in enumerate([100, 101, 103, 104, 105], start=1)
    ]
    changed = [dict(row) for row in observations]
    changed[-1]["value"] = "1000"

    first = cross_frozen.series_frame({"observations": observations}, definition)
    second = cross_frozen.series_frame({"observations": changed}, definition)

    assert first.iloc[-1]["shock_threshold"] == second.iloc[-1]["shock_threshold"]


def test_cross_asset_classification_requires_both_markets_and_periods() -> None:
    rows = []
    for pair in direct.ASSETS:
        for partition in (
            "development_2021_2023",
            "internal_validation_2024_2025",
        ):
            rows.append(
                {
                    "event_family": "example",
                    "pair": pair,
                    "horizon": 4,
                    "whole_event_partition": partition,
                    "events": cross_frozen.MINIMUM_PARTITION_EVENTS,
                    "partition_pass": True,
                }
            )
    verdicts = cross_direct.classify(
        pd.DataFrame(rows),
        pd.DataFrame([{"event_family": "example", "event_label": "Example"}]),
    )

    assert verdicts.iloc[0]["verdict"] == "repeatable_activity_lead_both_markets"
