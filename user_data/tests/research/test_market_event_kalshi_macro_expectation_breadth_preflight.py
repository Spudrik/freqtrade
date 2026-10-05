# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
import pytest
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_kalshi_macro_expectation_breadth_preflight"
)
macro = importlib.import_module(MODULE)


def test_threshold_parser_handles_both_legacy_negative_shapes() -> None:
    assert macro.parse_threshold("CPICORE-23OCT-TN0.2", -0.2) == pytest.approx(-0.2)
    assert macro.parse_threshold("PAYROLLS-24APR-T-100000", -100000) == -100000
    assert macro.parse_threshold("U3-24OCT-T4.1", 4.099999) == pytest.approx(4.099999)
    assert macro.parse_threshold("KXGDP-26JAN30-T.5", 0.5) == pytest.approx(0.5)
    assert macro.parse_threshold(
        "PCECORE-22NOV-TN0.1",
        0.1,
        title="Will core PCE be above -0.1%?",
        rules_primary="Resolves yes if core PCE is above -0.1%.",
    ) == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="disagree"):
        macro.parse_threshold("GDP-24JUL25-T2.4", 2.9)
    with pytest.raises(ValueError, match="disagree"):
        macro.parse_threshold("PCECORE-22NOV-TN0.1", 0.1)


def test_threshold_summary_requires_a_coherent_probability_curve() -> None:
    frame = DataFrame(
        {
            "floor_strike": [0.1, 0.2, 0.3],
            "yes_bid": [0.75, 0.50, 0.15],
            "yes_ask": [0.85, 0.60, 0.25],
            "probability_mid": [0.80, 0.55, 0.20],
            "spread": [0.10, 0.10, 0.10],
            "staleness_minutes": [5, 5, 5],
        }
    )
    assert macro.summarise_threshold_event(frame)["usable_2h"] is True
    contradictory = frame.copy()
    contradictory.loc[1, "yes_bid"] = 0.90
    contradictory.loc[1, "probability_mid"] = 0.95
    assert macro.summarise_threshold_event(contradictory)["usable_2h"] is False


def test_categorical_summary_requires_total_probability_bracket() -> None:
    frame = DataFrame(
        {
            "category": ["cut", "hold", "hike"],
            "yes_bid": [0.10, 0.65, 0.10],
            "yes_ask": [0.20, 0.75, 0.20],
            "probability_mid": [0.15, 0.70, 0.15],
            "spread": [0.10, 0.10, 0.10],
            "staleness_minutes": [10, 10, 10],
        }
    )
    summary = macro.summarise_categorical_event(frame)
    assert summary["usable_2h"] is True
    assert summary["expectation_value"] == "hold"
    incomplete = frame.iloc[:2].copy()
    assert macro.summarise_categorical_event(incomplete)["usable_2h"] is False


def test_source_blocks_preserve_old_and_later_periods() -> None:
    assert (
        macro.source_block(pd.Timestamp("2024-07-25T12:25:00Z"))
        == "development_source_2022_2024"
    )
    assert (
        macro.source_block(pd.Timestamp("2025-07-30T12:25:00Z"))
        == "later_source_2025_2026"
    )


def test_candle_url_encodes_categorical_ticker_in_path() -> None:
    cutoff = pd.Timestamp("2023-07-26T17:55:00Z")
    url, path = macro.candles_url("FEDDECISION-23JUL-H>25", cutoff)
    assert "H%3E25" in url
    assert path.endswith("/FEDDECISION-23JUL-H%3E25/candlesticks")
