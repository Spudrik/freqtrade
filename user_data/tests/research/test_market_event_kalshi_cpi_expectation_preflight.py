# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
import pytest
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_kalshi_cpi_expectation_preflight"
)
kalshi = importlib.import_module(MODULE)


def test_official_url_validation_rejects_wrong_host() -> None:
    url = kalshi.MARKETS_URL
    assert kalshi.valid_api_url(url, expected_path="/trade-api/v2/historical/markets")
    assert not kalshi.valid_api_url(
        url.replace(kalshi.API_HOST, "example.com"),
        expected_path="/trade-api/v2/historical/markets",
    )


def test_latest_quote_uses_last_valid_two_sided_quote() -> None:
    close = pd.Timestamp("2025-02-12T13:25:00Z")
    payload = {
        "candlesticks": [
            {
                "end_period_ts": int(pd.Timestamp("2025-02-12T12:00:00Z").timestamp()),
                "yes_bid": {"close": "0.40"},
                "yes_ask": {"close": "0.50"},
            },
            {
                "end_period_ts": int(pd.Timestamp("2025-02-12T13:20:00Z").timestamp()),
                "yes_bid": {"close": "0.55"},
                "yes_ask": {"close": "0.61"},
            },
        ]
    }
    quote = kalshi.latest_quote("ticker", close, payload)
    assert quote["probability_mid"] == pytest.approx(0.58)
    assert quote["staleness_minutes"] == 5.0


def test_event_summary_requires_fresh_bracketed_quotes() -> None:
    frame = DataFrame(
        {
            "market_close_time_utc": ["2025-02-12T13:25:00Z"] * 3,
            "floor_strike": [0.1, 0.2, 0.3],
            "probability_mid": [0.8, 0.55, 0.2],
            "spread": [0.05, 0.08, 0.1],
            "staleness_minutes": [10, 15, 20],
            "volume_fp": [100, 200, 300],
        }
    )
    result = kalshi.summarise_event("KXCPI-25JAN", frame)
    assert result["usable_2h"] is True
    assert 0.2 < result["implied_median_2h"] < 0.3
    stale = frame.copy()
    stale["staleness_minutes"] = 181
    assert kalshi.summarise_event("KXCPI-25JAN", stale)["usable_2h"] is False


def test_candle_url_uses_common_supplied_cutoff() -> None:
    cutoff = pd.Timestamp("2022-08-10T12:25:00Z")
    url, path = kalshi.candles_url("CPI-22JUL-T0.1", cutoff)
    assert path.endswith("/CPI-22JUL-T0.1/candlesticks")
    assert f"end_ts={int(cutoff.timestamp())}" in url


def test_legacy_ticker_supplies_missing_threshold_but_must_agree_with_field() -> None:
    assert kalshi.market_threshold("CPI-22JAN-T0.7", None) == pytest.approx(0.7)
    assert kalshi.market_threshold("CPI-23JUL-T-0.3", -0.3) == pytest.approx(-0.3)
    assert kalshi.market_threshold("CPI-23JUL-T.3", 0.3) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="disagree"):
        kalshi.market_threshold("CPI-22JAN-T0.7", 0.6)
