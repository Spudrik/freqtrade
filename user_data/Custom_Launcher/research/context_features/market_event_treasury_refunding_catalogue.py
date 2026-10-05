"""Build an outcome-blind 2020-2025 Treasury refunding statement catalogue."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
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
    "treasury_refunding_catalogue_20260905a"
)
CATALOG_PATH = OUTPUT_ROOT / "treasury_refunding_event_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "treasury_refunding_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "treasury_refunding_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "treasury_refunding_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "treasury_refunding_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "treasury_refunding_catalogue_result.json"

TREASURY_ROOT = "https://home.treasury.gov"
ARCHIVE_URL = (
    TREASURY_ROOT
    + "/policy-issues/financing-the-government/quarterly-refunding/"
    "quarterly-refunding-archives/official-remarks-on-quarterly-refunding-by-calendar-year"
)
START_YEAR = 2020
END_YEAR = 2025
WORKERS = 4
REQUEST_TIMEOUT = 45
USER_AGENT = "Objective02b-Treasury-refunding-catalogue/1.0"
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


def official_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "home.treasury.gov"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
    )


def official_statement_url(url: str) -> bool:
    if not official_url(url):
        return False
    path = urlparse(url).path.rstrip("/")
    return re.fullmatch(r"/news/press-releases/[a-z]{2}\d{3,4}", path) is not None


def fetch_page(
    url: str,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    if not official_url(url):
        raise ValueError(f"Non-official Treasury URL: {url}")
    response = requester(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    content = bytes(response.content)
    status = int(response.status_code)
    if status != 200:
        raise RuntimeError(f"Treasury source returned {status}: {url}")
    return content.decode("utf-8", errors="replace"), {
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


def parse_archive(document: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(document, "html.parser")
    records: list[dict[str, Any]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_node = soup.find("th", id=str(year))
        row = year_node.parent.find_next_sibling("tr") if year_node else None
        if row is None:
            raise ValueError(f"Treasury archive has no quarterly row for {year}")
        links = row.select("a[href]")
        if len(links) != 4:
            raise ValueError(f"Treasury archive yielded {len(links)} quarters for {year}")
        quarters: set[int] = set()
        for anchor in links:
            label = " ".join(anchor.get_text(" ", strip=True).replace("\u200b", "").split())
            quarter_match = re.fullmatch(r"([1-4])(st|nd|rd|th) Quarter", label)
            if quarter_match is None:
                raise ValueError(f"Treasury archive has an invalid quarter label: {label}")
            quarter = int(quarter_match.group(1))
            expected_suffix = {1: "st", 2: "nd", 3: "rd", 4: "th"}[quarter]
            if quarter_match.group(2) != expected_suffix:
                raise ValueError(f"Treasury archive has an invalid quarter label: {label}")
            quarters.add(quarter)
            records.append(
                {
                    "archive_year": year,
                    "archive_quarter": quarter,
                    "archive_label": label,
                    "source_url": urljoin(TREASURY_ROOT, str(anchor["href"])),
                }
            )
        if quarters != {1, 2, 3, 4}:
            raise ValueError(f"Treasury archive quarters are incomplete for {year}")
    if len(records) != 24 or len({row["source_url"] for row in records}) != 24:
        raise ValueError("Treasury archive did not yield 24 unique statement pages")
    if not all(official_statement_url(str(row["source_url"])) for row in records):
        raise ValueError("Treasury archive contains a noncanonical statement URL")
    return records


def parse_statement(document: str, archive_row: Mapping[str, Any]) -> dict[str, Any]:
    soup = BeautifulSoup(document, "html.parser")
    headings = [
        " ".join(node.get_text(" ", strip=True).split())
        for node in soup.select("h1,h2")
    ]
    titles = [title for title in headings if title.startswith("Quarterly Refunding Statement")]
    if len(titles) != 1:
        raise ValueError(f"Treasury statement title is missing or ambiguous: {archive_row}")
    time_nodes = soup.select("time.datetime[datetime]")
    if len(time_nodes) != 1:
        raise ValueError(f"Treasury statement has no unique official time: {archive_row}")
    raw_timestamp = str(time_nodes[0]["datetime"])
    published = pd.Timestamp(raw_timestamp)
    if published.tz is None:
        raise ValueError(f"Treasury statement timestamp is timezone-naive: {archive_row}")
    local_clock = published.tz_localize(None)
    expected_eastern = local_clock.tz_localize(ZoneInfo("America/New_York"))
    if published.utcoffset() != expected_eastern.utcoffset():
        raise ValueError(f"Treasury statement timestamp is not Eastern time: {archive_row}")
    if published.year != int(archive_row["archive_year"]):
        raise ValueError(f"Treasury statement year disagrees with archive: {archive_row}")
    displayed_date = pd.Timestamp(time_nodes[0].get_text(" ", strip=True)).date()
    if displayed_date != published.date():
        raise ValueError(f"Treasury displayed date disagrees with timestamp: {archive_row}")
    quarter_months = {
        1: {1, 2},
        2: {4, 5},
        3: {7, 8},
        4: {10, 11},
    }
    if published.month not in quarter_months[int(archive_row["archive_quarter"])]:
        raise ValueError(f"Treasury release month disagrees with archive quarter: {archive_row}")
    anchor = published.tz_convert("UTC")
    page_id = str(archive_row["source_url"]).rstrip("/").rsplit("/", maxsplit=1)[-1]
    return {
        "event_id": f"treasury_refunding_{published.date().isoformat()}_{page_id}",
        "event_family": "treasury_quarterly_refunding_statement",
        "archive_year": int(archive_row["archive_year"]),
        "archive_quarter": int(archive_row["archive_quarter"]),
        "decision_date_local": published.date().isoformat(),
        "official_timestamp_text": raw_timestamp,
        "anchor_utc": anchor.isoformat(),
        "title": titles[0],
        "source_url": str(archive_row["source_url"]),
        "archive_url": ARCHIVE_URL,
        "timestamp_quality": "official_page_timezone_aware_datetime",
        "market_consensus_status": "unavailable_not_imputed",
        "whole_event_partition": PARTITIONS[published.year],
        "selection_method": "complete_official_archive_one_statement_per_quarter",
    }


def build_catalogue(
    archive_document: str, statement_documents: Mapping[str, str]
) -> DataFrame:
    archive_rows = parse_archive(archive_document)
    records = [
        parse_statement(statement_documents[str(row["source_url"])], row)
        for row in archive_rows
    ]
    catalogue = DataFrame.from_records(records)
    catalogue["anchor_utc"] = pd.to_datetime(catalogue["anchor_utc"], utc=True)
    catalogue = catalogue.sort_values("anchor_utc", kind="stable").reset_index(drop=True)
    validate_catalogue(
        catalogue, {str(row["source_url"]) for row in archive_rows}
    )
    return catalogue


def validate_catalogue_row(row: Any) -> None:
    published = pd.Timestamp(row.official_timestamp_text)
    if published.tz is None or published.tz_convert("UTC") != pd.Timestamp(row.anchor_utc):
        raise ValueError("Treasury official timestamp disagrees with UTC anchor")
    local_clock = published.tz_localize(None)
    expected_eastern = local_clock.tz_localize(ZoneInfo("America/New_York"))
    if published.utcoffset() != expected_eastern.utcoffset():
        raise ValueError("Treasury official timestamp is not Eastern time")
    if published.date().isoformat() != row.decision_date_local:
        raise ValueError("Treasury local date disagrees with official timestamp")
    quarter_months = {
        1: {1, 2},
        2: {4, 5},
        3: {7, 8},
        4: {10, 11},
    }
    if published.month not in quarter_months[int(row.archive_quarter)]:
        raise ValueError("Treasury release month disagrees with archive quarter")
    page_id = row.source_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    expected_id = f"treasury_refunding_{row.decision_date_local}_{page_id}"
    if row.event_id != expected_id:
        raise ValueError("Treasury event ID disagrees with its source")


def validate_catalogue_shape(
    catalogue: DataFrame, required: set[str], allowed_source_urls: set[str]
) -> None:
    if not required.issubset(catalogue):
        raise ValueError(f"Treasury catalogue is missing: {required - set(catalogue)}")
    if len(catalogue) != 24:
        raise ValueError(f"Treasury catalogue has {len(catalogue)} rows, expected 24")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("Treasury event IDs are not unique")
    if not catalogue["source_url"].map(official_statement_url).all():
        raise ValueError("Treasury catalogue contains a non-official URL")
    if set(catalogue["source_url"]) != allowed_source_urls:
        raise ValueError("Treasury catalogue URLs disagree with the official archive")
    counts = catalogue.groupby("archive_year")["event_id"].size().to_dict()
    if counts != {year: 4 for year in range(START_YEAR, END_YEAR + 1)}:
        raise ValueError(f"Treasury yearly counts are invalid: {counts}")


def validate_catalogue_constants(catalogue: DataFrame) -> None:
    expected_values = {
        "market_consensus_status": "unavailable_not_imputed",
        "event_family": "treasury_quarterly_refunding_statement",
        "archive_url": ARCHIVE_URL,
        "timestamp_quality": "official_page_timezone_aware_datetime",
        "selection_method": "complete_official_archive_one_statement_per_quarter",
    }
    for column, expected in expected_values.items():
        if not catalogue[column].eq(expected).all():
            raise ValueError(f"Treasury {column} changed")
    if not catalogue["title"].str.startswith("Quarterly Refunding Statement").all():
        raise ValueError("Treasury statement title changed")
    expected_partitions = catalogue["archive_year"].map(PARTITIONS)
    if not catalogue["whole_event_partition"].eq(expected_partitions).all():
        raise ValueError("Treasury whole-event partitions changed")
    timestamps = pd.to_datetime(catalogue["anchor_utc"], utc=True)
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Treasury release timestamps are not ordered")


def validate_catalogue(catalogue: DataFrame, allowed_source_urls: set[str]) -> None:
    required = {
        "event_id",
        "event_family",
        "archive_year",
        "archive_quarter",
        "decision_date_local",
        "official_timestamp_text",
        "anchor_utc",
        "title",
        "source_url",
        "archive_url",
        "timestamp_quality",
        "market_consensus_status",
        "whole_event_partition",
        "selection_method",
    }
    validate_catalogue_shape(catalogue, required, allowed_source_urls)
    validate_catalogue_constants(catalogue)
    for row in catalogue.itertuples(index=False):
        validate_catalogue_row(row)


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby(["archive_year", "whole_event_partition"], sort=True)
        .agg(events=("event_id", "nunique"))
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_treasury_refunding_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "plain_question": (
            "Do complete quarterly Treasury refunding statement clocks repeatedly make "
            "Bitcoin or Ethereum unusually active over the following 5-60 minutes?"
        ),
        "event_rows": len(catalogue),
        "archive_url": ARCHIVE_URL,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "expected_page_count": 24,
        "expected_page_ids": sorted(
            value.rstrip("/").rsplit("/", maxsplit=1)[-1]
            for value in catalogue["source_url"]
        ),
        "selection_rule": (
            "Exactly four archive-linked pages per year and one for each named quarter; "
            "each page title begins Quarterly Refunding Statement and supplies one "
            "timezone-aware Eastern publication timestamp."
        ),
        "market_test": {
            "assets": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "horizons_minutes": [5, 15, 30, 60],
            "minimum_events_by_partition": {
                "development_2021_2023": 10,
                "internal_validation_2024_2025": 6,
            },
            "minimum_median_activity_score": 1.20,
            "minimum_above_control_rate": 0.55,
            "variants": [
                "all_quarterly_statements",
                "exclude_other_major_scheduled_event_within_4h",
            ],
            "controls": (
                "Up to 12 clean prior same-weekday and UTC-clock weeks, rejecting "
                "non-contiguous minute windows."
            ),
        },
        "boundaries": [
            "This is a scheduled quarterly programme with only 24 observations and a "
            "small eight-release later period.",
            "The official page timestamp is observable publication evidence but does "
            "not prove that no dealer or journalist saw information earlier.",
            "Historical market consensus is unavailable and is not imputed.",
            "No direction, causation, profit, or trading-rule claim is allowed.",
        ],
        "counts": counts.to_dict(orient="records"),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Treasury Quarterly Refunding Source Catalogue",
            "",
            f"- Complete 2020-2025 quarterly statements: `{freeze['event_rows']}`",
            "- Exact source: one official Treasury archive link per quarter.",
            "- Time: timezone-aware publication value from each official page.",
            "- Historical market expectation: **unavailable, not imputed**.",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "This catalogue supports a small exploratory activity test only. It does "
            "not establish an unexpected funding shock or a directional rule.",
            "",
        ]
    )


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_treasury_refunding_catalogue":
            raise ValueError("Existing Treasury catalogue result is invalid")
        for value in result.get("artifacts", {}).values():
            path = Path(value["path"])
            if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
                raise ValueError(f"Existing Treasury catalogue artifact changed: {path}")
        return result
    archive_document, archive_manifest = fetch_page(ARCHIVE_URL)
    archive_rows = parse_archive(archive_document)
    documents, detail_manifest = fetch_pages(
        str(row["source_url"]) for row in archive_rows
    )
    catalogue = build_catalogue(archive_document, documents)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(catalogue, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOG_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(
        {
            "retrieved_at_utc": g0.utc_now(),
            "sources": [archive_manifest, *detail_manifest],
        },
        SOURCE_MANIFEST_PATH,
    )
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_treasury_refunding_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "event_rows": len(catalogue),
        "source_pages": 1 + len(detail_manifest),
        "artifacts": {
            "catalogue": artifact(CATALOG_PATH),
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
                    "expected_events": 24,
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
