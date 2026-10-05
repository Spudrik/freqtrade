"""Audit whether free Kalshi history can supply pre-release CPI expectations."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/source_preflight/"
    "kalshi_cpi_expectation_20260910a"
)
QUOTE_PATH = OUTPUT_ROOT / "kalshi_cpi_sample_quotes.csv"
EVENT_PATH = OUTPUT_ROOT / "kalshi_cpi_sample_events.csv"
RESULT_PATH = OUTPUT_ROOT / "kalshi_cpi_expectation_preflight.json"

API_HOST = "external-api.kalshi.com"
API_ROOT = f"https://{API_HOST}/trade-api/v2"
SERIES_TICKER = "KXCPI"
MARKETS_URL = f"{API_ROOT}/historical/markets?series_ticker={SERIES_TICKER}&limit=1000"
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b Kalshi-CPI-source-audit/1.0"
SAMPLE_EVENTS = (
    "CPI-22JAN",
    "CPI-22JUL",
    "CPI-23JAN",
    "CPI-23JUL",
    "CPI-24JAN",
    "CPI-24JUL",
    "KXCPI-25JAN",
    "KXCPI-25JUL",
    "KXCPI-26JAN",
    "KXCPI-26MAY",
)
MIN_FRESH_THRESHOLDS = 3
MAX_MEDIAN_SPREAD = 0.25
TICKER_THRESHOLD_RE = re.compile(
    r"-T(?P<negative>N|-)?(?P<value>(?:\d+(?:\.\d+)?|\.\d+))$"
)
REQUEST_PACING_SECONDS = 0.25


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def valid_api_url(url: str, *, expected_path: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == API_HOST
        and parsed.path == expected_path
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.fragment
    )


def request_json(
    url: str,
    *,
    expected_path: str,
    requester: Callable[..., Any] = requests.get,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Kalshi API returned {response.status_code} for {expected_path}")
    if not valid_api_url(str(response.url), expected_path=expected_path):
        raise ValueError("Kalshi response did not come from the expected official API path")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Kalshi response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Kalshi response root is not an object")
    return payload, {
        "url": str(response.url),
        "http_status": int(response.status_code),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def fetch_markets() -> tuple[DataFrame, dict[str, Any]]:
    payload, source = request_json(
        MARKETS_URL,
        expected_path="/trade-api/v2/historical/markets",
    )
    rows = payload.get("markets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Kalshi historical market response is empty")
    if payload.get("cursor") not in (None, ""):
        raise ValueError("Kalshi market response unexpectedly requires another page")
    frame = DataFrame.from_records(rows)
    required = {
        "ticker",
        "event_ticker",
        "close_time",
        "floor_strike",
        "volume_fp",
    }
    if not required.issubset(frame):
        raise ValueError(f"Kalshi markets are missing fields: {required - set(frame)}")
    if frame["ticker"].astype(str).duplicated().any():
        raise ValueError("Kalshi market tickers are duplicated")
    return frame, source


def candles_url(ticker: str, close_time: pd.Timestamp) -> tuple[str, str]:
    path = f"/trade-api/v2/historical/markets/{ticker}/candlesticks"
    start_ts = int((close_time - pd.Timedelta(hours=24)).timestamp())
    end_ts = int(close_time.timestamp())
    return (
        f"{API_ROOT}/historical/markets/{ticker}/candlesticks?"
        f"start_ts={start_ts}&end_ts={end_ts}&period_interval=1",
        path,
    )


def latest_quote(
    ticker: str,
    close_time: pd.Timestamp,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    candles = payload.get("candlesticks")
    if not isinstance(candles, list):
        raise ValueError(f"Kalshi candle response is invalid for {ticker}")
    candidates: list[dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        bid_block = candle.get("yes_bid")
        ask_block = candle.get("yes_ask")
        if not isinstance(bid_block, Mapping) or not isinstance(ask_block, Mapping):
            continue
        bid = pd.to_numeric(bid_block.get("close"), errors="coerce")
        ask = pd.to_numeric(ask_block.get("close"), errors="coerce")
        timestamp = pd.to_datetime(candle.get("end_period_ts"), unit="s", utc=True)
        if pd.isna(bid) or pd.isna(ask) or pd.isna(timestamp) or timestamp > close_time:
            continue
        if not 0 <= float(bid) <= float(ask) <= 1:
            continue
        candidates.append(
            {
                "quote_time_utc": timestamp.isoformat(),
                "yes_bid": float(bid),
                "yes_ask": float(ask),
                "probability_mid": (float(bid) + float(ask)) / 2.0,
                "spread": float(ask) - float(bid),
                "staleness_minutes": (close_time - timestamp).total_seconds() / 60.0,
            }
        )
    if not candidates:
        return {
            "quote_time_utc": None,
            "yes_bid": None,
            "yes_ask": None,
            "probability_mid": None,
            "spread": None,
            "staleness_minutes": None,
        }
    return candidates[-1]


def market_threshold(ticker: str, floor_strike: Any) -> float:
    match = TICKER_THRESHOLD_RE.search(ticker)
    if match is None:
        raise ValueError(f"Kalshi CPI ticker does not encode a threshold: {ticker}")
    ticker_value = float(match.group("value"))
    if match.group("negative") in {"N", "-"}:
        ticker_value *= -1
    field_value = pd.to_numeric(floor_strike, errors="coerce")
    if pd.notna(field_value) and abs(float(field_value) - ticker_value) > 1e-9:
        raise ValueError(f"Kalshi ticker and floor strike disagree for {ticker}")
    return ticker_value


def probability_brackets_half(frame: DataFrame) -> bool:
    probabilities = pd.to_numeric(frame["probability_mid"], errors="coerce").dropna()
    return bool((probabilities >= 0.5).any() and (probabilities < 0.5).any())


def interpolate_median(frame: DataFrame) -> float | None:
    ordered = frame.dropna(subset=["probability_mid", "floor_strike"]).copy()
    ordered["floor_strike"] = pd.to_numeric(ordered["floor_strike"], errors="coerce")
    ordered = ordered.dropna(subset=["floor_strike"]).sort_values("floor_strike")
    if not probability_brackets_half(ordered):
        return None
    lower = ordered[ordered["probability_mid"] >= 0.5].tail(1)
    upper = ordered[ordered["probability_mid"] < 0.5].head(1)
    if lower.empty or upper.empty:
        return None
    x0 = float(lower.iloc[0]["floor_strike"])
    x1 = float(upper.iloc[0]["floor_strike"])
    p0 = float(lower.iloc[0]["probability_mid"])
    p1 = float(upper.iloc[0]["probability_mid"])
    if x1 <= x0 or p0 == p1:
        return None
    return x0 + (0.5 - p0) * (x1 - x0) / (p1 - p0)


def summarise_event(event_ticker: str, quotes: DataFrame) -> dict[str, Any]:
    close_time = pd.to_datetime(quotes["market_close_time_utc"].iloc[0], utc=True)
    block = "development_source_2022_2024" if close_time.year <= 2024 else "later_source_2025_2026"
    output: dict[str, Any] = {
        "event_ticker": event_ticker,
        "source_block": block,
        "market_close_time_utc": close_time.isoformat(),
        "threshold_markets": len(quotes),
        "total_contract_volume": float(pd.to_numeric(quotes["volume_fp"]).sum()),
    }
    for hours in (2, 12, 24):
        fresh = quotes[
            pd.to_numeric(quotes["staleness_minutes"], errors="coerce") <= hours * 60
        ].copy()
        median_spread = pd.to_numeric(fresh["spread"], errors="coerce").median()
        bracket = probability_brackets_half(fresh) if not fresh.empty else False
        implied_median = interpolate_median(fresh)
        usable = bool(
            len(fresh) >= MIN_FRESH_THRESHOLDS
            and bracket
            and pd.notna(median_spread)
            and float(median_spread) <= MAX_MEDIAN_SPREAD
            and implied_median is not None
        )
        output[f"fresh_thresholds_{hours}h"] = len(fresh)
        output[f"median_spread_{hours}h"] = (
            float(median_spread) if pd.notna(median_spread) else None
        )
        output[f"brackets_half_{hours}h"] = bracket
        output[f"usable_{hours}h"] = usable
        output[f"implied_median_{hours}h"] = implied_median if usable else None
    return output


def run_preflight() -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    markets, markets_source = fetch_markets()
    available = set(markets["event_ticker"].astype(str))
    missing = set(SAMPLE_EVENTS) - available
    if missing:
        raise ValueError(f"Frozen Kalshi sample events are missing: {sorted(missing)}")
    quote_rows: list[dict[str, Any]] = []
    source_rows = [markets_source]
    for event_ticker in SAMPLE_EVENTS:
        event_markets = markets[markets["event_ticker"].eq(event_ticker)].copy()
        close_times = pd.to_datetime(event_markets["close_time"], utc=True, errors="raise")
        close_spread = close_times.max() - close_times.min()
        if close_spread > pd.Timedelta(minutes=1):
            raise ValueError(
                f"Kalshi event {event_ticker} close times span more than one minute"
            )
        # Contracts can close a few seconds apart.  Use the earliest close as the
        # common causal cutoff so no threshold receives later information.
        close_time = close_times.min()
        for row in event_markets.to_dict(orient="records"):
            ticker = str(row["ticker"])
            url, expected_path = candles_url(ticker, close_time)
            payload, source = request_json(url, expected_path=expected_path)
            time.sleep(REQUEST_PACING_SECONDS)
            source_rows.append(source)
            quote_rows.append(
                {
                    "event_ticker": event_ticker,
                    "ticker": ticker,
                    "market_close_time_utc": close_time.isoformat(),
                    "floor_strike": market_threshold(ticker, row["floor_strike"]),
                    "volume_fp": pd.to_numeric(row["volume_fp"], errors="coerce"),
                    **latest_quote(ticker, close_time, payload),
                }
            )
    quotes = DataFrame.from_records(quote_rows)
    events = DataFrame.from_records(
        [
            summarise_event(event, quotes[quotes["event_ticker"].eq(event)])
            for event in SAMPLE_EVENTS
        ]
    )
    development_usable = int(
        events.loc[
            events["source_block"].eq("development_source_2022_2024"), "usable_2h"
        ].sum()
    )
    later_usable = int(
        events.loc[
            events["source_block"].eq("later_source_2025_2026"), "usable_2h"
        ].sum()
    )
    total_usable = int(events["usable_2h"].sum())
    passed = bool(development_usable >= 4 and later_usable >= 3 and total_usable >= 7)
    result = {
        "schema_version": 1,
        "status": "completed_kalshi_cpi_expectation_source_preflight",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "plain_question": (
            "Can free historical prediction-market quotes reconstruct what participants "
            "expected shortly before CPI releases?"
        ),
        "source": "Kalshi public historical market-data API",
        "series_ticker": SERIES_TICKER,
        "sample_events": list(SAMPLE_EVENTS),
        "sample_event_count": len(events),
        "sample_threshold_market_count": len(quotes),
        "usable_within_2h": total_usable,
        "development_usable_within_2h": development_usable,
        "later_usable_within_2h": later_usable,
        "source_pass": passed,
        "pass_rule": (
            "At least 7/10 sample events, including at least 4/6 in 2022-2024 and 3/4 "
            "in 2025-2026, need three two-sided threshold quotes within two hours, a "
            "50% probability bracket, and median bid-ask spread no wider than 0.25."
        ),
        "interpretation": (
            "A pass supports a full market-implied CPI expectation catalogue. It is not "
            "the economist survey consensus and is not evidence about crypto direction."
        ),
        "request_count": len(source_rows),
        "source_response_hashes": source_rows,
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    return quotes, events, result


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_kalshi_cpi_expectation_source_preflight":
        raise ValueError("Existing Kalshi CPI source preflight is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing Kalshi CPI artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    quotes, events, result = run_preflight()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(quotes, QUOTE_PATH)
    g0.atomic_write_csv(events, EVENT_PATH)
    result["artifacts"] = {
        "quotes": artifact(QUOTE_PATH),
        "events": artifact(EVENT_PATH),
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "sample_events": list(SAMPLE_EVENTS),
                    "crypto_outcomes_will_be_read": False,
                    "profit_will_be_used": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
