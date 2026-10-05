"""Build an outcome-blind catalogue of official UK ONS CPI release clocks."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import calendar
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
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
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "uk_ons_cpi_catalogue_20260906a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "uk_ons_cpi_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "uk_ons_cpi_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "uk_ons_cpi_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "uk_ons_cpi_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "uk_ons_cpi_catalogue_result.json"

ONS_ROOT = "https://www.ons.gov.uk"
START_PUBLICATION_YEAR = 2020
END_PUBLICATION_YEAR = 2025
EXPECTED_ROWS = 72
EXPECTED_CLOCK_COUNTS = {"07:00": 69, "09:30": 3}
WORKERS = 4
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b ONS CPI source audit"
RELEASE_PATTERN = re.compile(
    r"\b(Released|Release date):\s*"
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}(?:am|pm))\b"
)
PARTITIONS = {
    2020: "context_only_2020",
    2021: "development_2021_2023",
    2022: "development_2021_2023",
    2023: "development_2021_2023",
    2024: "internal_validation_2024_2025",
    2025: "internal_validation_2024_2025",
}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def normal_text(document: str | bytes) -> str:
    soup = BeautifulSoup(document, "html.parser")
    return " ".join(soup.get_text(" ", strip=True).split())


def official_release_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "www.ons.gov.uk"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.query == ""
        and parsed.fragment == ""
        and re.fullmatch(r"/releases/[a-z0-9]+", parsed.path) is not None
    )


def expected_release_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    year, month = 2019, 12
    for _ in range(EXPECTED_ROWS):
        month_name = calendar.month_name[month]
        slug = (
            "ukconsumerpriceinflationdecember2019"
            if (year, month) == (2019, 12)
            else f"consumerpriceinflationuk{month_name.casefold()}{year}"
        )
        specs.append(
            {
                "event_id": f"uk_ons_cpi_{year}{month:02d}",
                "reference_month": f"{year}-{month:02d}",
                "expected_title": f"Consumer price inflation, UK: {month_name} {year}",
                "source_url": f"{ONS_ROOT}/releases/{slug}",
            }
        )
        month += 1
        if month == 13:
            year += 1
            month = 1
    if len({spec["event_id"] for spec in specs}) != EXPECTED_ROWS:
        raise ValueError("ONS expected CPI event identifiers are not unique")
    if len({spec["source_url"] for spec in specs}) != EXPECTED_ROWS:
        raise ValueError("ONS expected CPI release URLs are not unique")
    if not all(official_release_url(spec["source_url"]) for spec in specs):
        raise ValueError("ONS expected CPI release list contains a non-official URL")
    return specs


def fetch_page(
    url: str,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    if not official_release_url(url):
        raise ValueError(f"Non-official ONS release URL: {url}")
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    status = int(response.status_code)
    if status != 200:
        raise RuntimeError(f"ONS source returned {status}: {url}")
    if str(response.url) != url or not official_release_url(str(response.url)):
        raise ValueError("ONS response did not come from the exact approved release URL")
    try:
        document = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"ONS release is not valid UTF-8: {url}") from exc
    return document, {
        "url": url,
        "http_status": status,
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def fetch_pages(
    urls: Iterable[str], *, workers: int = WORKERS
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    unique = list(dict.fromkeys(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fetched = list(executor.map(fetch_page, unique))
    documents = {
        url: document for url, (document, _) in zip(unique, fetched, strict=True)
    }
    manifest = [entry for _, entry in fetched]
    return documents, manifest


def parse_release_page(document: str, spec: Mapping[str, str]) -> dict[str, Any]:
    soup = BeautifulSoup(document, "html.parser")
    headings = [" ".join(node.get_text(" ", strip=True).split()) for node in soup.select("h1")]
    if headings.count(spec["expected_title"]) != 1:
        raise ValueError(f"ONS release title is missing or ambiguous: {spec['reference_month']}")
    text = normal_text(document)
    matches = RELEASE_PATTERN.findall(text)
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) != 1:
        raise ValueError(f"ONS release clock is missing or ambiguous: {spec['reference_month']}")
    label, raw_timestamp = unique_matches[0]
    try:
        naive = datetime.strptime(raw_timestamp, "%d %B %Y %I:%M%p")
    except ValueError as exc:
        raise ValueError(f"ONS release clock cannot be parsed: {raw_timestamp}") from exc
    local = pd.Timestamp(naive, tz=ZoneInfo("Europe/London"))
    anchor = local.tz_convert("UTC")
    if local.year not in PARTITIONS:
        raise ValueError(f"ONS release is outside the frozen publication years: {local}")
    reference_end = pd.Period(spec["reference_month"], freq="M").end_time.date()
    lag_days = (local.date() - reference_end).days
    if not 1 <= lag_days <= 40:
        raise ValueError(f"ONS release date disagrees with its reference month: {spec}")
    changed = "Changes to this release date" in text
    return {
        "event_id": spec["event_id"],
        "event_family": "uk_ons_consumer_price_inflation",
        "reference_month": spec["reference_month"],
        "release_title": spec["expected_title"],
        "release_date_local": local.date().isoformat(),
        "official_release_clock_local": local.strftime("%H:%M"),
        "official_release_timestamp_local": local.isoformat(),
        "anchor_utc": anchor.isoformat(),
        "source_clock_label": label,
        "time_precision": "official_ons_release_page_minute",
        "intraday_eligible": True,
        "historical_schedule_change_noted": changed,
        "previous_schedule_count": text.count("Previous date") if changed else 0,
        "source_url": spec["source_url"],
        "market_expectation_status": "unavailable_not_imputed",
        "whole_event_partition": PARTITIONS[local.year],
        "selection_method": "complete_fixed_ons_cpi_release_pages_2020_2025",
    }


def build_catalogue(documents: Mapping[str, str]) -> DataFrame:
    specs = expected_release_specs()
    expected_urls = {spec["source_url"] for spec in specs}
    if set(documents) != expected_urls:
        raise ValueError("ONS fetched document set differs from the frozen release list")
    catalogue = DataFrame.from_records(
        [parse_release_page(documents[spec["source_url"]], spec) for spec in specs]
    )
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: DataFrame) -> None:
    required = {
        "event_id",
        "event_family",
        "reference_month",
        "release_date_local",
        "official_release_clock_local",
        "official_release_timestamp_local",
        "anchor_utc",
        "time_precision",
        "intraday_eligible",
        "source_url",
        "market_expectation_status",
        "whole_event_partition",
        "selection_method",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"ONS CPI catalogue is missing: {required - set(catalogue)}")
    if len(catalogue) != EXPECTED_ROWS:
        raise ValueError("ONS CPI catalogue row count changed")
    duplicated = [
        column
        for column in ("event_id", "reference_month", "anchor_utc", "source_url")
        if catalogue[column].duplicated().any()
    ]
    if duplicated:
        raise ValueError(f"ONS CPI catalogue has duplicate columns: {duplicated}")
    specs = expected_release_specs()
    if set(catalogue["source_url"]) != {spec["source_url"] for spec in specs}:
        raise ValueError("ONS CPI catalogue source URLs differ from the frozen list")
    if not catalogue["source_url"].map(official_release_url).all():
        raise ValueError("ONS CPI catalogue contains a non-official URL")
    fixed = {
        "event_family": "uk_ons_consumer_price_inflation",
        "time_precision": "official_ons_release_page_minute",
        "intraday_eligible": True,
        "market_expectation_status": "unavailable_not_imputed",
        "selection_method": "complete_fixed_ons_cpi_release_pages_2020_2025",
    }
    changed = [
        column for column, expected in fixed.items() if not catalogue[column].eq(expected).all()
    ]
    if changed:
        raise ValueError(f"ONS CPI catalogue fixed fields changed: {changed}")
    clock_counts = catalogue["official_release_clock_local"].value_counts().to_dict()
    if clock_counts != EXPECTED_CLOCK_COUNTS:
        raise ValueError(f"ONS CPI release clock counts changed: {clock_counts}")
    partition_counts = catalogue["whole_event_partition"].value_counts().to_dict()
    if partition_counts != {
        "context_only_2020": 12,
        "development_2021_2023": 36,
        "internal_validation_2024_2025": 24,
    }:
        raise ValueError(f"ONS CPI partition counts changed: {partition_counts}")
    anchors = pd.to_datetime(catalogue["anchor_utc"], utc=True, errors="raise")
    if not anchors.is_monotonic_increasing:
        raise ValueError("ONS CPI anchors are not chronological")
    forbidden_tokens = ("profit", "future_return", "direction", "ohlcv", "market_volume")
    if any(
        token in column.casefold()
        for column in catalogue.columns
        for token in forbidden_tokens
    ):
        raise ValueError("ONS CPI source catalogue contains a market outcome column")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby(
            ["whole_event_partition", "official_release_clock_local"], sort=True
        )
        .agg(
            releases=("event_id", "nunique"),
            first_release_utc=("anchor_utc", "min"),
            last_release_utc=("anchor_utc", "max"),
            schedule_change_pages=("historical_schedule_change_noted", "sum"),
        )
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_uk_ons_cpi_catalogue_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "plain_question": (
            "Can official release-specific ONS pages supply a complete minute-safe UK CPI "
            "clock set for publication years 2020-2025?"
        ),
        "publication_period": {
            "start_inclusive": "2020-01-01",
            "end_inclusive": "2025-12-31",
        },
        "release_rows": len(catalogue),
        "unique_release_clocks": int(catalogue["anchor_utc"].nunique()),
        "local_clock_counts": dict(
            sorted(Counter(catalogue["official_release_clock_local"]).items())
        ),
        "schedule_change_pages": int(
            catalogue["historical_schedule_change_noted"].sum()
        ),
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            "The ONS page's released minute is used; its old scheduled time is not used.",
            "Europe/London daylight-saving rules are applied before conversion to UTC.",
            "The page clock is an official release clock, not a measured HTTP first-byte time.",
            "Current revised time-series values are not used as first-release values.",
            "Historical market expectations are unavailable and are not imputed.",
            "No crypto, market outcome, direction, causation, or profit data was read.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# UK ONS CPI Source Catalogue",
            "",
            f"- Official monthly releases: `{freeze['release_rows']}`",
            f"- Unique exact release clocks: `{freeze['unique_release_clocks']}`",
            f"- Local clock counts: `{freeze['local_clock_counts']}`",
            f"- Pages recording schedule changes: `{freeze['schedule_change_pages']}`",
            "- Historical market expectations: **Unavailable; not guessed**",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "The fixed 2020-2025 publication catalogue is source-complete and suitable for "
            "a later activity-first test. Direction remains unavailable until expectation or "
            "separately frozen market-confirmation information is added.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_uk_ons_cpi_catalogue":
        raise ValueError("Existing ONS CPI catalogue result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing ONS CPI catalogue artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    specs = expected_release_specs()
    documents, source_entries = fetch_pages(spec["source_url"] for spec in specs)
    catalogue = build_catalogue(documents)
    counts = count_catalogue(catalogue)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        "schema_version": 1,
        "status": "frozen_official_ons_release_pages",
        "retrieved_at_utc": g0.utc_now(),
        "source_root": ONS_ROOT,
        "expected_release_pages": EXPECTED_ROWS,
        "documents": source_entries,
    }
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(source_manifest, SOURCE_MANIFEST_PATH)
    freeze = freeze_document(catalogue, counts)
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_uk_ons_cpi_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "release_rows": len(catalogue),
        "unique_release_clocks": int(catalogue["anchor_utc"].nunique()),
        "artifacts": {
            "catalogue": artifact(CATALOGUE_PATH),
            "counts": artifact(COUNTS_PATH),
            "source_manifest": artifact(SOURCE_MANIFEST_PATH),
            "freeze": artifact(FREEZE_PATH),
            "report": artifact(REPORT_PATH),
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
                    "outcomes_will_be_read": False,
                    "profit_will_be_used": False,
                    "expected_release_pages": EXPECTED_ROWS,
                    "workers": WORKERS,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
