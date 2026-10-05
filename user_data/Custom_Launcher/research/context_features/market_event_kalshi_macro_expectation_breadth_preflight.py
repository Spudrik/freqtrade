"""Audit free pre-release expectation history across several US macro families.

This is an outcome-blind source test.  It does not read cryptocurrency prices or
judge whether any economic release predicted market direction.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_kalshi_cpi_expectation_preflight as cpi_source,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/source_preflight/"
    "kalshi_macro_expectation_breadth_20260910a"
)
QUOTE_PATH = OUTPUT_ROOT / "kalshi_macro_sample_quotes.csv"
EVENT_PATH = OUTPUT_ROOT / "kalshi_macro_sample_events.csv"
RESULT_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_breadth_preflight.json"

MIN_FRESH_QUOTES = 3
MAX_MEDIAN_SPREAD = 0.25
THRESHOLD_RE = re.compile(
    r"-T(?P<negative>N|-)?(?P<value>(?:\d+(?:\.\d+)?|\.\d+))$"
)


@dataclass(frozen=True)
class SourceSpec:
    family: str
    component: str
    series_ticker: str
    shape: Literal["threshold", "categorical"]
    sample_events: tuple[str, ...]


SOURCE_SPECS = (
    SourceSpec(
        "consumer_inflation",
        "core_cpi",
        "KXCPICORE",
        "threshold",
        (
            "CPICORE-22JUL",
            "CPICORE-24JUL",
            "KXCPICORE-25JUL",
            "KXCPICORE-26MAY",
        ),
    ),
    SourceSpec(
        "consumer_spending_inflation",
        "core_pce",
        "KXPCECORE",
        "threshold",
        (
            "PCECORE-22NOV",
            "PCECORE-23SEP",
            "KXPCECORE-25JUL",
            "KXPCECORE-26MAY",
        ),
    ),
    SourceSpec(
        "jobs_report",
        "nonfarm_payrolls",
        "KXPAYROLLS",
        "threshold",
        (
            "PROLLS-23JUL",
            "PAYROLLS-24JUL",
            "KXPAYROLLS-25JUL",
            "KXPAYROLLS-26MAY",
        ),
    ),
    SourceSpec(
        "jobs_report",
        "unemployment_rate",
        "KXU3",
        "threshold",
        (
            "U3-22JUL",
            "U3-24JUL",
            "KXU3-25JUL",
            "KXU3-26MAY",
        ),
    ),
    SourceSpec(
        "central_bank_decision",
        "federal_reserve_decision",
        "KXFEDDECISION",
        "categorical",
        (
            "FEDDECISION-23JUL",
            "FEDDECISION-24JUL",
            "KXFEDDECISION-25JUL",
            "KXFEDDECISION-26JUN",
        ),
    ),
    SourceSpec(
        "economic_growth",
        "real_gdp_growth",
        "KXGDP",
        "threshold",
        (
            "GDP-22JUN30",
            "GDP-24JUL25",
            "KXGDP-25JUL30",
            "KXGDP-26APR30",
        ),
    ),
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def fetch_markets(spec: SourceSpec) -> tuple[DataFrame, dict[str, Any]]:
    path = "/trade-api/v2/historical/markets"
    url = (
        f"{cpi_source.API_ROOT}/historical/markets?"
        f"series_ticker={spec.series_ticker}&limit=1000"
    )
    payload, source = cpi_source.request_json(url, expected_path=path)
    rows = payload.get("markets")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Kalshi history is empty for {spec.series_ticker}")
    if payload.get("cursor") not in (None, ""):
        raise ValueError(f"Kalshi history requires another page for {spec.series_ticker}")
    frame = DataFrame.from_records(rows)
    required = {"ticker", "event_ticker", "close_time", "volume_fp"}
    if not required.issubset(frame):
        raise ValueError(
            f"Kalshi markets for {spec.series_ticker} lack {required - set(frame)}"
        )
    if frame["ticker"].astype(str).duplicated().any():
        raise ValueError(f"Kalshi market tickers repeat for {spec.series_ticker}")
    missing = set(spec.sample_events) - set(frame["event_ticker"].astype(str))
    if missing:
        raise ValueError(
            f"Frozen samples are missing for {spec.series_ticker}: {sorted(missing)}"
        )
    return frame, source


def parse_threshold(
    ticker: str,
    floor_strike: Any,
    *,
    title: str = "",
    rules_primary: str = "",
) -> float:
    match = THRESHOLD_RE.search(ticker)
    if match is None:
        raise ValueError(f"Threshold ticker has an unsupported shape: {ticker}")
    ticker_value = float(match.group("value"))
    if match.group("negative") in {"N", "-"}:
        ticker_value *= -1
    field_value = pd.to_numeric(floor_strike, errors="coerce")
    if pd.isna(field_value):
        return ticker_value
    tolerance = max(1e-5, abs(ticker_value) * 1e-6)
    if abs(float(field_value) - ticker_value) > tolerance:
        signed_text = f"above {ticker_value:g}"
        legacy_negative_field_bug = bool(
            ticker_value < 0
            and abs(float(field_value) + ticker_value) <= tolerance
            and signed_text in title
            and signed_text in rules_primary
        )
        if legacy_negative_field_bug:
            return ticker_value
        raise ValueError(f"Ticker and floor strike disagree for {ticker}")
    return float(field_value)


def candles_url(ticker: str, close_time: pd.Timestamp) -> tuple[str, str]:
    encoded_ticker = quote(ticker, safe="-._~")
    path = f"/trade-api/v2/historical/markets/{encoded_ticker}/candlesticks"
    start_ts = int((close_time - pd.Timedelta(hours=24)).timestamp())
    end_ts = int(close_time.timestamp())
    return (
        f"{cpi_source.API_ROOT}/historical/markets/{encoded_ticker}/candlesticks?"
        f"start_ts={start_ts}&end_ts={end_ts}&period_interval=1",
        path,
    )


def probability_brackets_half(frame: DataFrame) -> bool:
    values = pd.to_numeric(frame["probability_mid"], errors="coerce").dropna()
    return bool((values >= 0.5).any() and (values < 0.5).any())


def threshold_curve_is_consistent(frame: DataFrame) -> bool:
    ordered = frame.dropna(
        subset=["floor_strike", "yes_bid", "yes_ask"]
    ).sort_values("floor_strike")
    if len(ordered) < 2:
        return False
    lower_asks = ordered["yes_ask"].astype(float).to_numpy()[:-1]
    higher_bids = ordered["yes_bid"].astype(float).to_numpy()[1:]
    return bool((higher_bids <= lower_asks).all())


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


def summarise_threshold_event(quotes: DataFrame) -> dict[str, Any]:
    fresh = quotes[
        pd.to_numeric(quotes["staleness_minutes"], errors="coerce") <= 120
    ].dropna(subset=["yes_bid", "yes_ask", "probability_mid", "floor_strike"])
    median_spread = pd.to_numeric(fresh["spread"], errors="coerce").median()
    brackets = probability_brackets_half(fresh) if not fresh.empty else False
    curve_consistent = threshold_curve_is_consistent(fresh)
    implied_median = interpolate_median(fresh)
    usable = bool(
        len(fresh) >= MIN_FRESH_QUOTES
        and pd.notna(median_spread)
        and float(median_spread) <= MAX_MEDIAN_SPREAD
        and brackets
        and curve_consistent
        and implied_median is not None
    )
    return {
        "fresh_quotes_2h": len(fresh),
        "median_spread_2h": (
            float(median_spread) if pd.notna(median_spread) else None
        ),
        "probability_shape_valid": bool(brackets and curve_consistent),
        "expectation_value": implied_median if usable else None,
        "usable_2h": usable,
    }


def summarise_categorical_event(quotes: DataFrame) -> dict[str, Any]:
    fresh = quotes[
        pd.to_numeric(quotes["staleness_minutes"], errors="coerce") <= 120
    ].dropna(subset=["yes_bid", "yes_ask", "probability_mid"])
    median_spread = pd.to_numeric(fresh["spread"], errors="coerce").median()
    bid_sum = float(pd.to_numeric(fresh["yes_bid"], errors="coerce").sum())
    ask_sum = float(pd.to_numeric(fresh["yes_ask"], errors="coerce").sum())
    mass_brackets_one = bool(bid_sum <= 1 <= ask_sum)
    top_category = None
    if not fresh.empty:
        top_category = str(
            fresh.sort_values("probability_mid", ascending=False).iloc[0]["category"]
        )
    usable = bool(
        len(fresh) >= MIN_FRESH_QUOTES
        and pd.notna(median_spread)
        and float(median_spread) <= MAX_MEDIAN_SPREAD
        and mass_brackets_one
        and top_category
    )
    return {
        "fresh_quotes_2h": len(fresh),
        "median_spread_2h": (
            float(median_spread) if pd.notna(median_spread) else None
        ),
        "probability_shape_valid": mass_brackets_one,
        "expectation_value": top_category if usable else None,
        "usable_2h": usable,
    }


def source_block(close_time: pd.Timestamp) -> str:
    return "development_source_2022_2024" if close_time.year <= 2024 else "later_source_2025_2026"


def response_manifest_sha256(source_rows: Sequence[Mapping[str, Any]]) -> str:
    entries = sorted(
        f"{row.get('url', '')}|{row.get('content_sha256', '')}" for row in source_rows
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def run_preflight() -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    quote_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    series_inventory: list[dict[str, Any]] = []

    for spec in SOURCE_SPECS:
        markets, market_source = fetch_markets(spec)
        source_rows.append(market_source)
        series_inventory.append(
            {
                "family": spec.family,
                "component": spec.component,
                "series_ticker": spec.series_ticker,
                "historical_markets": len(markets),
                "historical_events": int(markets["event_ticker"].nunique()),
            }
        )
        for event_ticker in spec.sample_events:
            selected = markets[markets["event_ticker"].eq(event_ticker)].copy()
            close_times = pd.to_datetime(selected["close_time"], utc=True, errors="raise")
            common_close = close_times.min()
            if close_times.max() - common_close > pd.Timedelta(minutes=1):
                event_rows.append(
                    {
                        "family": spec.family,
                        "component": spec.component,
                        "series_ticker": spec.series_ticker,
                        "shape": spec.shape,
                        "event_ticker": event_ticker,
                        "source_block": source_block(common_close),
                        "market_close_time_utc": common_close.isoformat(),
                        "threshold_or_category_markets": len(selected),
                        "fresh_quotes_2h": 0,
                        "median_spread_2h": None,
                        "probability_shape_valid": False,
                        "expectation_value": None,
                        "usable_2h": False,
                        "failure_reason": "contract_close_times_span_over_one_minute",
                    }
                )
                continue
            event_quotes: list[dict[str, Any]] = []
            for row in selected.to_dict(orient="records"):
                ticker = str(row["ticker"])
                url, expected_path = candles_url(ticker, common_close)
                try:
                    payload, candle_source = cpi_source.request_json(
                        url,
                        expected_path=expected_path,
                    )
                except (RuntimeError, ValueError) as exc:
                    raise type(exc)(
                        f"{exc}; series={spec.series_ticker}; "
                        f"event={event_ticker}; ticker={ticker}"
                    ) from exc
                source_rows.append(candle_source)
                quote = {
                    "family": spec.family,
                    "component": spec.component,
                    "series_ticker": spec.series_ticker,
                    "shape": spec.shape,
                    "event_ticker": event_ticker,
                    "ticker": ticker,
                    "market_close_time_utc": common_close.isoformat(),
                    "volume_fp": pd.to_numeric(row.get("volume_fp"), errors="coerce"),
                    "floor_strike": (
                        parse_threshold(
                            ticker,
                            row.get("floor_strike"),
                            title=str(row.get("title") or ""),
                            rules_primary=str(row.get("rules_primary") or ""),
                        )
                        if spec.shape == "threshold"
                        else None
                    ),
                    "category": str(row.get("subtitle") or row.get("title") or ticker),
                    **cpi_source.latest_quote(ticker, common_close, payload),
                }
                quote_rows.append(quote)
                event_quotes.append(quote)
            event_frame = DataFrame.from_records(event_quotes)
            detail = (
                summarise_threshold_event(event_frame)
                if spec.shape == "threshold"
                else summarise_categorical_event(event_frame)
            )
            event_rows.append(
                {
                    "family": spec.family,
                    "component": spec.component,
                    "series_ticker": spec.series_ticker,
                    "shape": spec.shape,
                    "event_ticker": event_ticker,
                    "source_block": source_block(common_close),
                    "market_close_time_utc": common_close.isoformat(),
                    "threshold_or_category_markets": len(event_frame),
                    "failure_reason": None,
                    **detail,
                }
            )

    quotes = DataFrame.from_records(quote_rows)
    events = DataFrame.from_records(event_rows)
    component_rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        selected = events[events["component"].eq(spec.component)]
        development_usable = int(
            selected.loc[
                selected["source_block"].eq("development_source_2022_2024"),
                "usable_2h",
            ].sum()
        )
        later_usable = int(
            selected.loc[
                selected["source_block"].eq("later_source_2025_2026"),
                "usable_2h",
            ].sum()
        )
        usable = int(selected["usable_2h"].sum())
        component_rows.append(
            {
                "family": spec.family,
                "component": spec.component,
                "series_ticker": spec.series_ticker,
                "sample_events": len(selected),
                "usable_events": usable,
                "development_usable": development_usable,
                "later_usable": later_usable,
                "source_pass": bool(
                    usable >= 3 and development_usable >= 1 and later_usable >= 1
                ),
            }
        )

    result = {
        "schema_version": 1,
        "status": "completed_kalshi_macro_expectation_breadth_preflight",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "plain_question": (
            "Does free prediction-market history contain usable pre-release expectations "
            "for several major economic-news families, rather than CPI alone?"
        ),
        "source": "Kalshi public historical market-data API",
        "sample_selection": (
            "Four dates per component were frozen before any cryptocurrency outcome was "
            "opened, spanning older and newer source periods."
        ),
        "pass_rule": (
            "A component needs at least 3/4 usable dates, including at least one in both "
            "2022-2024 and 2025-2026. Threshold curves need three fresh two-sided quotes, "
            "a 50% bracket, no bid/ask-proven ordering contradiction, and median spread "
            "at most 0.25. Fed categories must bracket total probability one."
        ),
        "independence_warning": (
            "Payrolls and unemployment are two components of one jobs-report episode. "
            "They must not be counted as independent event discoveries."
        ),
        "interpretation_limit": (
            "These are market-implied expectations, not economist survey consensus. A "
            "source pass is not evidence that a release predicts cryptocurrency direction."
        ),
        "series_inventory": series_inventory,
        "component_results": component_rows,
        "component_pass_count": sum(row["source_pass"] for row in component_rows),
        "request_count": len(source_rows),
        "response_manifest_sha256": response_manifest_sha256(source_rows),
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    return quotes, events, result


def validate_existing_result(result: Mapping[str, Any]) -> None:
    expected = "completed_kalshi_macro_expectation_breadth_preflight"
    if result.get("status") != expected:
        raise ValueError("Existing Kalshi macro expectation preflight is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing Kalshi macro artifact changed: {path}")


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
                    "families": [spec.family for spec in SOURCE_SPECS],
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
