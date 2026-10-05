# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_kalshi_macro_expectation_catalogue"
)
catalogue = importlib.import_module(MODULE)


def test_failure_reason_is_specific() -> None:
    assert (
        catalogue.quote_failure_reason(
            {
                "usable_2h": False,
                "fresh_quotes_2h": 2,
                "median_spread_2h": 0.02,
                "probability_shape_valid": True,
                "expectation_value": 0.2,
            }
        )
        == "fewer_than_three_fresh_two_sided_quotes"
    )
    assert (
        catalogue.quote_failure_reason(
            {
                "usable_2h": False,
                "fresh_quotes_2h": 4,
                "median_spread_2h": 0.40,
                "probability_shape_valid": True,
                "expectation_value": 0.2,
            }
        )
        == "median_bid_ask_spread_too_wide"
    )


def test_source_summary_requires_both_blocks() -> None:
    rows = []
    for component, old_usable, later_usable in (
        ("headline_cpi", 6, 6),
        ("core_cpi", 8, 5),
    ):
        rows.extend(
            {
                "component": component,
                "source_block": "development_source_2022_2024",
                "usable_2h": index < old_usable,
            }
            for index in range(8)
        )
        rows.extend(
            {
                "component": component,
                "source_block": "later_source_2025_2026",
                "usable_2h": index < later_usable,
            }
            for index in range(8)
        )
    # Other configured components need rows so the summary remains a complete batch.
    for spec in catalogue.FULL_SOURCE_SPECS:
        if spec.component in {"headline_cpi", "core_cpi"}:
            continue
        rows.extend(
            {
                "component": spec.component,
                "source_block": block,
                "usable_2h": False,
            }
            for block in (
                "development_source_2022_2024",
                "later_source_2025_2026",
            )
        )
    summary = catalogue.source_summary(DataFrame.from_records(rows)).set_index(
        "component"
    )
    assert summary.loc["headline_cpi", "source_readiness"] == "cross_period_ready"
    assert (
        summary.loc["core_cpi", "source_readiness"]
        == "insufficient_clean_expectations"
    )


def test_rejected_event_has_no_expectation() -> None:
    spec = catalogue.FULL_SOURCE_SPECS[0]
    row = catalogue.rejected_event(
        spec,
        "KXCPI-25JAN",
        pd.Timestamp("2025-02-12T13:25:00Z"),
        8,
        "mixed_close",
    )
    assert row["usable_2h"] is False
    assert row["expectation_value"] is None
    assert row["failure_reason"] == "mixed_close"


def test_close_parser_accepts_mixed_iso_precision() -> None:
    parsed = catalogue.parse_close_times(
        ["2025-12-18T13:29:00Z", "2025-12-29T20:13:36.935Z"]
    )
    assert len(parsed) == 2
    assert str(parsed.tz) == "UTC"


def test_component_cache_rejects_a_different_source_spec() -> None:
    spec = catalogue.FULL_SOURCE_SPECS[0]
    document = {
        "cache_contract_version": catalogue.CACHE_CONTRACT_VERSION,
        "source_spec": catalogue.asdict(spec),
        "quotes": [],
        "events": [],
        "inventory": {},
    }
    catalogue.validate_component_cache(document, spec)
    document["source_spec"]["series_ticker"] = "WRONG"
    try:
        catalogue.validate_component_cache(document, spec)
    except ValueError as exc:
        assert "source mismatch" in str(exc)
    else:
        raise AssertionError("A mismatched source cache was accepted")
