# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
import pytest
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_macro_expectation_actual_freeze"
)
freeze = importlib.import_module(MODULE)


def test_following_anchor_never_uses_a_pre_cutoff_release() -> None:
    official = DataFrame(
        {
            "anchor_utc": pd.to_datetime(
                ["2025-02-12T13:20:00Z", "2025-02-12T13:30:00Z"], utc=True
            ),
            "event_id": ["before", "after"],
        }
    )
    matched = freeze.match_following_anchor(
        pd.Timestamp("2025-02-12T13:25:00Z"), official
    )
    assert matched is not None
    assert matched["event_id"] == "after"


def test_following_anchor_uses_reference_period_when_releases_share_a_date() -> None:
    official = DataFrame(
        {
            "anchor_utc": pd.to_datetime(
                ["2025-12-16T13:30:00Z", "2025-12-16T13:30:00Z"], utc=True
            ),
            "observation_period": ["2025-10-01", "2025-11-01"],
            "event_id": ["october", "november"],
        }
    )
    matched = freeze.match_following_anchor(
        pd.Timestamp("2025-12-16T13:29:00Z"),
        official,
        reference_period="2025-11-01",
        period_column="observation_period",
    )
    assert matched is not None
    assert matched["event_id"] == "november"


@pytest.mark.parametrize(
    ("ticker", "period"),
    [
        ("CPI-21DEC", "2021-12-01"),
        ("PROLLS-23MAR", "2023-03-01"),
        ("PROLLS-23DECB", "2023-12-01"),
        ("KXCPICORE-25DECT", "2025-12-01"),
        ("KXPAYROLLS-26JUN", "2026-06-01"),
    ],
)
def test_reference_period_from_event_ticker(ticker: str, period: str) -> None:
    assert freeze.reference_period_from_event_ticker(ticker) == period


def test_following_anchor_keeps_valid_older_same_day_expectation() -> None:
    official = DataFrame(
        {
            "anchor_utc": pd.to_datetime(["2025-02-12T13:30:00Z"], utc=True),
            "event_id": ["release"],
        }
    )
    matched = freeze.match_following_anchor(
        pd.Timestamp("2025-02-12T05:00:00Z"), official
    )
    assert matched is not None
    assert matched["event_id"] == "release"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("False", False), (1, True), (0, False)],
)
def test_strict_bool_accepts_only_explicit_boolean_forms(value: object, expected: bool) -> None:
    assert freeze.strict_bool(value) is expected


def test_strict_bool_rejects_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="explicit boolean"):
        freeze.strict_bool("yes")


@pytest.mark.parametrize(
    ("change", "category"),
    [
        (-50, "Cut >25bps"),
        (-25, "Cut 25bps"),
        (0, "Hike 0bps"),
        (25, "Hike 25bps"),
        (50, "Hike >25bps"),
    ],
)
def test_fed_category(change: float, category: str) -> None:
    assert freeze.fed_category(change) == category


def test_jobs_actuals_use_one_release_vintage_for_both_months() -> None:
    employment = DataFrame(
        {
            "event_id": ["employment_2024_07"],
            "anchor_utc": pd.to_datetime(["2024-08-02T12:30:00Z"], utc=True),
            "observation_period": ["2024-07-01"],
        }
    )
    sources = {
        "documents": {
            "PAYEMS_initial": {
                "observations": [
                    {
                        "date": "2024-07-01",
                        "realtime_start": "2024-08-02",
                        "value": "158723",
                    }
                ]
            },
            "UNRATE_initial": {
                "observations": [
                    {
                        "date": "2024-07-01",
                        "realtime_start": "2024-08-02",
                        "value": "4.3",
                    }
                ]
            },
            "PAYEMS_release_vintages": {
                "observations": [
                    {"date": "2024-06-01", "PAYEMS_20240802": "158609"},
                    {"date": "2024-07-01", "PAYEMS_20240802": "158723"},
                ]
            },
            "UNRATE_release_vintages": {
                "observations": [
                    {"date": "2024-07-01", "UNRATE_20240802": "4.3"}
                ]
            },
        }
    }
    actual = freeze.build_jobs_actuals(employment, sources).iloc[0]
    assert actual["nonfarm_payrolls_actual"] == 114_000
    assert actual["unemployment_rate_actual"] == 4.3


def test_jobs_actuals_do_not_require_an_unpublished_other_component() -> None:
    employment = DataFrame(
        {
            "event_id": ["employment_2025_10"],
            "anchor_utc": pd.to_datetime(["2025-12-16T13:30:00Z"], utc=True),
            "observation_period": ["2025-10-01"],
        }
    )
    sources = {
        "documents": {
            "PAYEMS_initial": {
                "observations": [
                    {
                        "date": "2025-10-01",
                        "realtime_start": "2025-12-16",
                        "value": "159488",
                    }
                ]
            },
            "UNRATE_initial": {
                "observations": [
                    {
                        "date": "2025-10-01",
                        "realtime_start": "2025-12-16",
                        "value": ".",
                    }
                ]
            },
            "PAYEMS_release_vintages": {
                "observations": [
                    {"date": "2025-09-01", "PAYEMS_20251216": "159626"},
                    {"date": "2025-10-01", "PAYEMS_20251216": "159488"},
                ]
            },
            "UNRATE_release_vintages": {"observations": []},
        }
    }
    actual = freeze.build_jobs_actuals(employment, sources).iloc[0]
    assert actual["nonfarm_payrolls_actual"] == -138_000
    assert pd.isna(actual["unemployment_rate_actual"])


def test_jobs_components_do_not_receive_an_automatic_direction() -> None:
    semantics, rule = freeze.component_semantics("nonfarm_payrolls")
    assert "dual" in semantics
    assert rule == "not_preassigned"
