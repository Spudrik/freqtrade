"""Build an outcome-blind full Kalshi macro-expectation source catalogue."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_kalshi_cpi_expectation_preflight as cpi_source,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_kalshi_macro_expectation_breadth_preflight as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "macro_expectation_catalogue_20260910a"
)
QUOTE_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_quotes.csv"
EVENT_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_events.csv"
SUMMARY_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_source_summary.csv"
FREEZE_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "kalshi_macro_expectation_catalogue_result.json"
COMPONENT_CACHE_ROOT = OUTPUT_ROOT / "_component_cache"

MAX_WORKERS = 1
REQUEST_PACING_SECONDS = 0.25
MIN_USABLE_PER_SOURCE_BLOCK = 6
CACHE_CONTRACT_VERSION = 1

FULL_SOURCE_SPECS = (
    breadth.SourceSpec(
        "consumer_inflation",
        "headline_cpi",
        "KXCPI",
        "threshold",
        (),
    ),
    *breadth.SOURCE_SPECS,
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def parse_close_times(values: Any) -> Any:
    return pd.to_datetime(values, utc=True, errors="raise", format="mixed")


def quote_failure_reason(summary: Mapping[str, Any]) -> str | None:
    if bool(summary["usable_2h"]):
        return None
    if int(summary["fresh_quotes_2h"]) < breadth.MIN_FRESH_QUOTES:
        return "fewer_than_three_fresh_two_sided_quotes"
    spread = pd.to_numeric(summary["median_spread_2h"], errors="coerce")
    if pd.isna(spread) or float(spread) > breadth.MAX_MEDIAN_SPREAD:
        return "median_bid_ask_spread_too_wide"
    if not bool(summary["probability_shape_valid"]):
        return "probability_shape_not_reconstructable"
    if summary["expectation_value"] in (None, ""):
        return "expectation_value_not_reconstructable"
    return "source_gate_failed"


def quote_market(
    spec: breadth.SourceSpec,
    event_ticker: str,
    common_close: pd.Timestamp,
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ticker = str(row["ticker"])
    url, expected_path = breadth.candles_url(ticker, common_close)
    try:
        payload, source = cpi_source.request_json(url, expected_path=expected_path)
    except (RuntimeError, ValueError) as exc:
        raise type(exc)(
            f"{exc}; series={spec.series_ticker}; "
            f"event={event_ticker}; ticker={ticker}"
        ) from exc
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
            breadth.parse_threshold(
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
    time.sleep(REQUEST_PACING_SECONDS)
    return quote, source


def rejected_event(
    spec: breadth.SourceSpec,
    event_ticker: str,
    common_close: pd.Timestamp,
    market_count: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "family": spec.family,
        "component": spec.component,
        "series_ticker": spec.series_ticker,
        "shape": spec.shape,
        "expectation_episode_id": (
            f"{spec.family}_{common_close.strftime('%Y%m%dT%H%MZ')}"
        ),
        "event_ticker": event_ticker,
        "source_block": breadth.source_block(common_close),
        "market_close_time_utc": common_close.isoformat(),
        "threshold_or_category_markets": market_count,
        "fresh_quotes_2h": 0,
        "median_spread_2h": None,
        "probability_shape_valid": False,
        "expectation_value": None,
        "usable_2h": False,
        "failure_reason": reason,
    }


def catalogue_component(
    spec: breadth.SourceSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    markets, market_source = breadth.fetch_markets(spec)
    all_market_count = len(markets)
    all_event_count = int(markets["event_ticker"].nunique())
    close_times = parse_close_times(markets["close_time"])
    markets = markets.loc[close_times >= pd.Timestamp("2022-01-01T00:00:00Z")].copy()
    quote_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    source_rows = [market_source]

    grouped = markets.groupby("event_ticker", sort=False)
    ordered_events = sorted(
        grouped.groups,
        key=lambda event: parse_close_times(
            markets.loc[grouped.groups[event], "close_time"]
        ).min(),
    )
    for event_ticker in ordered_events:
        selected = markets.loc[grouped.groups[event_ticker]].copy()
        close_times = parse_close_times(selected["close_time"])
        common_close = close_times.min()
        if close_times.max() - common_close > pd.Timedelta(minutes=1):
            event_rows.append(
                rejected_event(
                    spec,
                    str(event_ticker),
                    common_close,
                    len(selected),
                    "contract_close_times_span_over_one_minute",
                )
            )
            continue

        records = selected.to_dict(orient="records")
        quote_one = partial(
            quote_market,
            spec,
            str(event_ticker),
            common_close,
        )
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            outputs = list(executor.map(quote_one, records))
        event_quotes = [quote for quote, _source in outputs]
        quote_rows.extend(event_quotes)
        source_rows.extend(source for _quote, source in outputs)
        event_frame = DataFrame.from_records(event_quotes)
        summary = (
            breadth.summarise_threshold_event(event_frame)
            if spec.shape == "threshold"
            else breadth.summarise_categorical_event(event_frame)
        )
        event_rows.append(
            {
                "family": spec.family,
                "component": spec.component,
                "series_ticker": spec.series_ticker,
                "shape": spec.shape,
                "expectation_episode_id": (
                    f"{spec.family}_{common_close.strftime('%Y%m%dT%H%MZ')}"
                ),
                "event_ticker": str(event_ticker),
                "source_block": breadth.source_block(common_close),
                "market_close_time_utc": common_close.isoformat(),
                "threshold_or_category_markets": len(event_frame),
                **summary,
                "failure_reason": quote_failure_reason(summary),
            }
        )

    inventory = {
        "family": spec.family,
        "component": spec.component,
        "series_ticker": spec.series_ticker,
        "shape": spec.shape,
        "all_historical_markets": all_market_count,
        "all_historical_events": all_event_count,
        "eligible_2022_onward_markets": len(markets),
        "eligible_2022_onward_events": int(markets["event_ticker"].nunique()),
    }
    return quote_rows, event_rows, source_rows, inventory


def source_summary(events: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in FULL_SOURCE_SPECS:
        selected = events[events["component"].eq(spec.component)]
        old = selected[selected["source_block"].eq("development_source_2022_2024")]
        later = selected[selected["source_block"].eq("later_source_2025_2026")]
        old_usable = int(old["usable_2h"].sum())
        later_usable = int(later["usable_2h"].sum())
        if (
            old_usable >= MIN_USABLE_PER_SOURCE_BLOCK
            and later_usable >= MIN_USABLE_PER_SOURCE_BLOCK
        ):
            readiness = "cross_period_ready"
        elif later_usable >= MIN_USABLE_PER_SOURCE_BLOCK:
            readiness = "later_period_only"
        else:
            readiness = "insufficient_clean_expectations"
        rows.append(
            {
                "family": spec.family,
                "component": spec.component,
                "series_ticker": spec.series_ticker,
                "historical_events": len(selected),
                "usable_events": int(selected["usable_2h"].sum()),
                "development_events": len(old),
                "development_usable": old_usable,
                "later_events": len(later),
                "later_usable": later_usable,
                "source_readiness": readiness,
            }
        )
    return DataFrame.from_records(rows)


def response_manifest_sha256(source_rows: Sequence[Mapping[str, Any]]) -> str:
    entries = sorted(
        f"{row.get('url', '')}|{row.get('content_sha256', '')}" for row in source_rows
    )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def component_cache_path(spec: breadth.SourceSpec) -> Path:
    return COMPONENT_CACHE_ROOT / f"{spec.component}.json"


def validate_component_cache(
    document: Mapping[str, Any], spec: breadth.SourceSpec
) -> None:
    if document.get("cache_contract_version") != CACHE_CONTRACT_VERSION:
        raise ValueError(f"Stale component-cache contract for {spec.component}")
    if document.get("source_spec") != asdict(spec):
        raise ValueError(f"Component-cache source mismatch for {spec.component}")
    if not isinstance(document.get("quotes"), list) or not isinstance(
        document.get("events"), list
    ):
        raise ValueError(f"Malformed component cache for {spec.component}")
    if not isinstance(document.get("inventory"), Mapping):
        raise ValueError(f"Missing component inventory for {spec.component}")


def load_or_build_component(
    spec: breadth.SourceSpec,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    str,
    dict[str, Any],
    dict[str, str],
]:
    path = component_cache_path(spec)
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_component_cache(document, spec)
        print(f"reusing completed {spec.component} cache", file=sys.stderr, flush=True)
        return (
            list(document["quotes"]),
            list(document["events"]),
            int(document["request_count"]),
            str(document["response_manifest_sha256"]),
            dict(document["inventory"]),
            artifact(path),
        )

    quotes, events, sources, inventory = catalogue_component(spec)
    document = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "created_at_utc": g0.utc_now(),
        "source_spec": asdict(spec),
        "crypto_outcomes_read": False,
        "economic_contract_resolution_used": False,
        "request_count": len(sources),
        "response_manifest_sha256": response_manifest_sha256(sources),
        "inventory": inventory,
        "quotes": quotes,
        "events": events,
    }
    COMPONENT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(document, path)
    return (
        quotes,
        events,
        len(sources),
        str(document["response_manifest_sha256"]),
        inventory,
        artifact(path),
    )


def run_catalogue() -> tuple[DataFrame, DataFrame, DataFrame, dict[str, Any]]:
    quote_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    request_count = 0
    component_manifests: list[str] = []
    component_caches: dict[str, dict[str, str]] = {}
    for spec in FULL_SOURCE_SPECS:
        print(
            f"cataloguing {spec.component} ({spec.series_ticker})",
            file=sys.stderr,
            flush=True,
        )
        (
            quotes,
            events,
            component_request_count,
            component_manifest,
            component_inventory,
            cache_artifact,
        ) = load_or_build_component(spec)
        quote_rows.extend(quotes)
        event_rows.extend(events)
        inventory.append(component_inventory)
        request_count += component_request_count
        component_manifests.append(component_manifest)
        component_caches[spec.component] = cache_artifact
        print(
            f"completed {spec.component}: {len(events)} events",
            file=sys.stderr,
            flush=True,
        )

    quotes = DataFrame.from_records(quote_rows).sort_values(
        ["market_close_time_utc", "component", "floor_strike", "ticker"],
        kind="stable",
        na_position="last",
    )
    events = DataFrame.from_records(event_rows).sort_values(
        ["market_close_time_utc", "component"], kind="stable"
    )
    summary = source_summary(events)
    freeze = {
        "schema_version": 1,
        "status": "frozen_full_macro_expectations_before_crypto_outcomes",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "economic_contract_resolution_used": False,
        "plain_question": (
            "Which macro families have enough clean historical pre-release market-implied "
            "expectations to support a later whole-event crypto-direction test?"
        ),
        "source": "Kalshi public historical market-data API",
        "components": [asdict(spec) for spec in FULL_SOURCE_SPECS],
        "common_cutoff_rule": (
            "Use the earliest close across every contract in an event. Reject an event "
            "when contract closes span more than one minute."
        ),
        "event_quality_rule": (
            "Within two hours of the common cutoff require at least three fresh two-sided "
            "quotes and median spread at most 0.25. Threshold curves must bracket 50% "
            "without a bid/ask-proven ordering contradiction; Fed categories must bracket "
            "total probability one."
        ),
        "cross_period_source_rule": (
            "At least six usable expectations in both 2022-2024 and 2025-2026."
        ),
        "request_pacing": {
            "workers": MAX_WORKERS,
            "minimum_seconds_after_each_candle_request": REQUEST_PACING_SECONDS,
            "reason": (
                "Respect the public API token bucket after a four-worker run returned "
                "HTTP 429."
            ),
        },
        "independence_rules": [
            "Headline and core CPI are components of one CPI release episode.",
            "Payrolls and unemployment are components of one jobs-report episode.",
            "Components and overlapping thresholds are not independent discoveries.",
        ],
        "interpretation_limit": (
            "Market-implied expectations are not economist survey consensus and may "
            "contain crowd bias or risk premia. Source readiness is not direction evidence."
        ),
        "source_inventory": inventory,
        "summary": summary.to_dict(orient="records"),
        "event_rows": len(events),
        "usable_event_rows": int(events["usable_2h"].sum()),
        "unique_expectation_episodes": int(events["expectation_episode_id"].nunique()),
        "request_count": request_count,
        "response_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(component_manifests)).encode("utf-8")
        ).hexdigest(),
        "component_cache_artifacts": component_caches,
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    return quotes, events, summary, freeze


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_full_macro_expectation_catalogue":
        raise ValueError("Existing full macro expectation catalogue is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing macro expectation artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    quotes, events, summary, freeze = run_catalogue()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(quotes, QUOTE_PATH)
    g0.atomic_write_csv(events, EVENT_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_full_macro_expectation_catalogue",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "source_ready_components": summary.loc[
            summary["source_readiness"].eq("cross_period_ready"), "component"
        ].tolist(),
        "later_only_components": summary.loc[
            summary["source_readiness"].eq("later_period_only"), "component"
        ].tolist(),
        "insufficient_components": summary.loc[
            summary["source_readiness"].eq("insufficient_clean_expectations"),
            "component",
        ].tolist(),
        "artifacts": {
            "quotes": artifact(QUOTE_PATH),
            "events": artifact(EVENT_PATH),
            "summary": artifact(SUMMARY_PATH),
            "freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
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
                    "components": [spec.component for spec in FULL_SOURCE_SPECS],
                    "max_workers": MAX_WORKERS,
                    "crypto_outcomes_will_be_read": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
