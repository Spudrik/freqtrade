"""Build a complete outcome-blind 2020-2025 scheduled central-bank catalogue.

ECB and Bank of England scheduled decisions use official fixed publication-time
rules. Bank of Japan decisions use the actual time printed on each statement page.
No market outcome, historical consensus, direction, or profit is read.
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
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
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "central_bank_catalogue_20260905a"
)
CATALOG_PATH = OUTPUT_ROOT / "scheduled_central_bank_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "scheduled_central_bank_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "scheduled_central_bank_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "scheduled_central_bank_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "scheduled_central_bank_catalogue_result.json"

START_YEAR = 2020
END_YEAR = 2025
WORKERS = 4
REQUEST_TIMEOUT = 45
USER_AGENT = "Objective02b-central-bank-catalogue/1.0"

ECB_ROOT = "https://www.ecb.europa.eu"
ECB_ARCHIVE_TEMPLATE = (
    ECB_ROOT + "/press/govcdec/mopo/{year}/html/index_include.en.html"
)
ECB_TIME_RULE_URL = (
    ECB_ROOT + "/press/pr/date/2022/html/ecb.pr220627~73acedf868.en.html"
)
ECB_TIME_CHANGE_DATE = date(2022, 7, 21)

BOE_ROOT = "https://www.bankofengland.co.uk"
BOE_ARCHIVE_URL = BOE_ROOT + "/sitemap/minutes"
BOE_NEWS_ARCHIVE_URL = BOE_ROOT + "/sitemap/news"
BOE_TIME_RULE_URL = (
    BOE_ROOT
    + "/news/2025/may/statement-on-the-timing-of-the-mpr-and-mpc-minutes"
)
BOE_TIMING_NOTICES = {
    date(2020, 8, 6): {
        "clock": time(7, 0),
        "url": (
            BOE_ROOT
            + "/news/2020/july/boe-statement-august-monetary-policy-report-"
            "and-financial-stability-report"
        ),
        "proof": ("07:00 (BST)", "originally scheduled time of 12:00 (BST)"),
    },
    date(2020, 11, 5): {
        "clock": time(7, 0),
        "url": (
            BOE_ROOT
            + "/news/2020/november/time-change-november-monetary-policy-report-"
            "and-monetary-policy-committee-decision-and-minutes"
        ),
        "proof": ("7:00 hrs (GMT)", "not at 12:00 hrs (GMT)"),
    },
    date(2022, 9, 22): {
        "clock": time(12, 0),
        "url": BOE_ROOT + "/news/2022/september/mpc-announcement-postponed",
        "proof": ("12pm on 22 September", "postponed for a period of one week"),
    },
    date(2025, 5, 8): {
        "clock": time(12, 2),
        "url": BOE_TIME_RULE_URL,
        "proof": ("12.02pm", "regular time of 12pm"),
    },
}
BOE_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

BOJ_ROOT = "https://www.boj.or.jp"
BOJ_ARCHIVE_URL = BOJ_ROOT + "/en/mopo/mpmsche_minu/past.htm"
BOJ_REFERENCE_ARCHIVE_URL = BOJ_ROOT + "/en/mopo/mpmsche_minu/m_ref/index.htm"
BOJ_RESCHEDULE_NOTICE_URL = (
    BOJ_ROOT + "/en/mopo/mpmsche_minu/m_ref/rel200316l.htm"
)
BOJ_SCHEDULE_STATUS = {
    "200316b": "rescheduled_scheduled",
    "200522b": "unscheduled",
}

EXPECTED_BANK_COUNTS = {"ECB": 48, "BoE": 48, "BoJ": 49}
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


def _normal_text(document: str | bytes) -> str:
    soup = BeautifulSoup(document, "html.parser")
    return " ".join(soup.get_text(" ", strip=True).split())


def _official_domain(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    try:
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
    ):
        return False
    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in ("ecb.europa.eu", "bankofengland.co.uk", "boj.or.jp")
    )


def _utc_from_local(day: date, clock: time, timezone: str) -> pd.Timestamp:
    local = datetime.combine(day, clock).replace(tzinfo=ZoneInfo(timezone))
    return pd.Timestamp(local.astimezone(ZoneInfo("UTC")))


def _partition(day: date) -> str:
    if day.year not in PARTITIONS:
        raise ValueError(f"Decision date is outside the frozen years: {day}")
    return PARTITIONS[day.year]


def _manifest_entry(url: str, content: bytes, status: int) -> dict[str, Any]:
    return {
        "url": url,
        "http_status": status,
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def fetch_page(
    url: str,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    if not _official_domain(url):
        raise ValueError(f"Non-official central-bank URL: {url}")
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    content = bytes(response.content)
    status = int(response.status_code)
    if status != 200:
        raise RuntimeError(f"Official source returned {status}: {url}")
    return content.decode("utf-8", errors="replace"), _manifest_entry(
        url, content, status
    )


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


def verify_timing_sources(ecb_document: str, boe_document: str) -> None:
    ecb = _normal_text(ecb_document)
    if not all(value in ecb for value in ("14:15", "13:45", "21 July 2022")):
        raise ValueError("ECB timing-rule page no longer proves the frozen time change")
    boe = _normal_text(boe_document).casefold()
    if not all(value in boe for value in ("12.02pm", "regular time of 12pm")):
        raise ValueError("BoE timing page no longer proves its rule and exception")


def verify_boe_known_timing_notices(
    news_archive_document: str, documents: Mapping[str, str]
) -> None:
    soup = BeautifulSoup(news_archive_document, "html.parser")
    archive_urls = {
        str(anchor["href"]).rstrip("/") for anchor in soup.select("a[href]")
    }
    expected_urls = {
        str(notice["url"]).rstrip("/") for notice in BOE_TIMING_NOTICES.values()
    }
    missing = expected_urls - archive_urls
    if missing:
        raise ValueError(f"BoE news sitemap is missing timing notices: {sorted(missing)}")
    for notice in BOE_TIMING_NOTICES.values():
        url = str(notice["url"])
        text = _normal_text(documents[url]).casefold()
        proof = tuple(str(value).casefold() for value in notice["proof"])
        if not all(value in text for value in proof):
            raise ValueError(f"BoE timing notice no longer proves its exception: {url}")


def verify_boj_special_sources(
    reference_archive_document: str, reschedule_document: str
) -> None:
    reference = _normal_text(reference_archive_document)
    if not all(
        value in reference
        for value in ("May 19, 2020", "Calling of an Unscheduled Monetary Policy Meeting")
    ):
        raise ValueError("BoJ reference archive no longer proves the unscheduled meeting")
    reschedule = _normal_text(reschedule_document)
    if not all(
        value in reschedule
        for value in (
            "Change in the Scheduled Dates of Monetary Policy Meetings",
            "Mar. 16 (Mon.)",
            "Mar. 18 (Wed.), 19 (Thurs.)",
        )
    ):
        raise ValueError("BoJ notice no longer proves the rescheduled March meeting")


def parse_ecb_archive(document: str, *, year: int, archive_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(document, "html.parser")
    indexed_links: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        label = " ".join(anchor.get_text(" ", strip=True).split())
        href = str(anchor["href"])
        if label == "Monetary policy decisions" and href.endswith(".en.html"):
            date_node = anchor.find_parent("dd")
            date_node = date_node.find_previous_sibling("dt") if date_node else None
            displayed_date = str(date_node.get("isodate", "")) if date_node else ""
            indexed_links.append((urljoin(ECB_ROOT, href), displayed_date))
    indexed_links = list(dict.fromkeys(indexed_links))
    if len(indexed_links) != 8:
        raise ValueError(
            f"ECB {year} archive yielded {len(indexed_links)} decisions, expected 8"
        )

    records: list[dict[str, Any]] = []
    for source_url, displayed_date in indexed_links:
        match = re.search(r"ecb\.mp(\d{6})~", source_url)
        if match is None:
            raise ValueError(f"ECB decision URL has no release date: {source_url}")
        day = datetime.strptime(match.group(1), "%y%m%d").date()
        if day.year != year:
            raise ValueError(f"ECB decision year disagrees with archive: {source_url}")
        if displayed_date != day.isoformat():
            raise ValueError(
                f"ECB displayed date disagrees with decision URL: {source_url}"
            )
        clock = time(13, 45) if day < ECB_TIME_CHANGE_DATE else time(14, 15)
        records.append(
            {
                "bank": "ECB",
                "event_id": f"ecb_{day.isoformat()}_scheduled_decision",
                "decision_date": day.isoformat(),
                "available_at_utc": _utc_from_local(
                    day, clock, "Europe/Berlin"
                ).isoformat(),
                "local_release_timestamp_text": (
                    f"{day.isoformat()} {clock.strftime('%H:%M')} Frankfurt civil time"
                ),
                "local_timezone": "Europe/Berlin",
                "timing_quality": "official_fixed_schedule_time",
                "scheduled": "scheduled",
                "decision_label": "ECB monetary policy decisions",
                "source_url": source_url,
                "archive_url": archive_url,
                "timing_rule_url": ECB_TIME_RULE_URL,
                "expected_consensus": None,
                "whole_event_partition": _partition(day),
                "selection_method": "complete_annual_official_decision_index",
            }
        )
    return records


def parse_boe_archive(document: str) -> list[str]:
    soup = BeautifulSoup(document, "html.parser")
    month_pattern = "|".join(BOE_MONTHS)
    pattern = re.compile(
        rf"^https://www\.bankofengland\.co\.uk/monetary-policy-summary-and-minutes/"
        rf"(20(?:2[0-5]))/(?:{month_pattern})-\1/?$",
        re.IGNORECASE,
    )
    links = list(
        dict.fromkeys(
            str(anchor["href"]).rstrip("/")
            for anchor in soup.select("a[href]")
            if pattern.match(str(anchor["href"]).rstrip("/"))
        )
    )
    counts = {
        year: sum(f"/{year}/" in link for link in links)
        for year in range(START_YEAR, END_YEAR + 1)
    }
    if any(count != 8 for count in counts.values()):
        raise ValueError(f"BoE archive did not yield eight regular decisions/year: {counts}")
    return links


def parse_boe_page(document: str, *, source_url: str) -> dict[str, Any]:
    text = _normal_text(document)
    published = re.search(
        r"Published on (\d{2} [A-Za-z]+ 20\d{2})", text, flags=re.IGNORECASE
    )
    if published is None:
        raise ValueError(f"BoE page has no published date: {source_url}")
    day = datetime.strptime(published.group(1).title(), "%d %B %Y").date()
    url_year = int(source_url.split("/")[-2])
    if day.year != url_year:
        raise ValueError(f"BoE published year disagrees with URL: {source_url}")
    notice = BOE_TIMING_NOTICES.get(day)
    clock = notice["clock"] if notice is not None else time(12, 0)
    soup = BeautifulSoup(document, "html.parser")
    title = " ".join((soup.title.get_text(" ", strip=True) if soup.title else "").split())
    label = title.split(" | ", maxsplit=1)[0] or "BoE monetary policy decision"
    return {
        "bank": "BoE",
        "event_id": f"boe_{day.isoformat()}_scheduled_decision",
        "decision_date": day.isoformat(),
        "available_at_utc": _utc_from_local(
            day, clock, "Europe/London"
        ).isoformat(),
        "local_release_timestamp_text": (
            f"{day.isoformat()} {clock.strftime('%H:%M')} Europe/London"
        ),
        "local_timezone": "Europe/London",
        "timing_quality": (
            "official_notice_exact_time"
            if notice is not None
            else "official_standard_time_with_known_exceptions"
        ),
        "scheduled": "scheduled",
        "decision_label": label,
        "source_url": source_url,
        "archive_url": BOE_ARCHIVE_URL,
        "timing_rule_url": (
            str(notice["url"]) if notice is not None else BOE_TIME_RULE_URL
        ),
        "expected_consensus": None,
        "whole_event_partition": _partition(day),
        "selection_method": "complete_official_regular_mpc_sitemap",
    }


def parse_boj_archive(document: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(document, "html.parser")
    identifiers: list[str] = []
    for anchor in soup.select("a[href]"):
        match = re.search(r"/k(2[0-5]\d{4}[a-z]?)\.pdf$", str(anchor["href"]))
        if match is None:
            continue
        identifier = match.group(1)
        year = 2000 + int(identifier[:2])
        if START_YEAR <= year <= END_YEAR and identifier not in identifiers:
            identifiers.append(identifier)
    if len(identifiers) != EXPECTED_BANK_COUNTS["BoJ"]:
        raise ValueError(
            f"BoJ archive yielded {len(identifiers)} meetings, expected 49"
        )
    return [
        (
            identifier,
            BOJ_ROOT
            + f"/en/mopo/mpmdeci/state_20{identifier[:2]}/k{identifier}.htm",
        )
        for identifier in identifiers
    ]


def _parse_boj_release_date(value: str, *, identifier: str) -> date:
    cleaned = re.sub(r"^[A-Za-z]+,\s*", "", " ".join(value.split()))
    if re.search(r",\s*20\d{2}$", cleaned) is None:
        cleaned = f"{cleaned}, {2000 + int(identifier[:2])}"
    try:
        return datetime.strptime(cleaned, "%B %d, %Y").date()
    except ValueError as error:
        raise ValueError(
            f"BoJ page has an unparseable first release date: {value}"
        ) from error


def parse_boj_page(
    document: str, *, identifier: str, source_url: str
) -> dict[str, Any]:
    text = _normal_text(document)
    match = re.search(
        r"Release dates and times:\s*(.+?)\s*--\s*"
        r"([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2}(?:,\s*20\d{2})?)\s*"
        r"at\s*(\d{1,2}:\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"BoJ page has no exact first release time: {source_url}")
    day = datetime.strptime(identifier[:6], "%y%m%d").date()
    displayed_day = _parse_boj_release_date(match.group(2), identifier=identifier)
    if displayed_day != day:
        raise ValueError(f"BoJ displayed release date disagrees with ID: {source_url}")
    clock = datetime.strptime(match.group(3), "%H:%M").time()
    label = " ".join(match.group(1).split())
    if any(
        excluded in label.casefold()
        for excluded in ("summary of opinions", "minutes", "full text")
    ):
        raise ValueError(f"BoJ first release item is not a decision: {source_url}")
    schedule_status = BOJ_SCHEDULE_STATUS.get(identifier, "scheduled")
    return {
        "bank": "BoJ",
        "event_id": f"boj_{day.isoformat()}_{schedule_status}_decision",
        "decision_date": day.isoformat(),
        "available_at_utc": _utc_from_local(
            day, clock, "Asia/Tokyo"
        ).isoformat(),
        "local_release_timestamp_text": (
            f"{day.isoformat()} {clock.strftime('%H:%M')} Asia/Tokyo"
        ),
        "local_timezone": "Asia/Tokyo",
        "timing_quality": "actual_document_time",
        "scheduled": schedule_status,
        "decision_label": label,
        "source_url": source_url,
        "archive_url": BOJ_ARCHIVE_URL,
        "timing_rule_url": source_url,
        "expected_consensus": None,
        "whole_event_partition": _partition(day),
        "selection_method": "complete_official_meeting_archive_page_time",
    }


def build_catalogue(
    initial_documents: Mapping[str, str],
    detail_documents: Mapping[str, str],
) -> DataFrame:
    verify_timing_sources(
        initial_documents[ECB_TIME_RULE_URL], initial_documents[BOE_TIME_RULE_URL]
    )
    verify_boe_known_timing_notices(
        initial_documents[BOE_NEWS_ARCHIVE_URL], initial_documents
    )
    verify_boj_special_sources(
        initial_documents[BOJ_REFERENCE_ARCHIVE_URL],
        initial_documents[BOJ_RESCHEDULE_NOTICE_URL],
    )
    records: list[dict[str, Any]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        archive_url = ECB_ARCHIVE_TEMPLATE.format(year=year)
        records.extend(
            parse_ecb_archive(
                initial_documents[archive_url], year=year, archive_url=archive_url
            )
        )
    for source_url in parse_boe_archive(initial_documents[BOE_ARCHIVE_URL]):
        records.append(
            parse_boe_page(detail_documents[source_url], source_url=source_url)
        )
    for identifier, source_url in parse_boj_archive(
        initial_documents[BOJ_ARCHIVE_URL]
    ):
        records.append(
            parse_boj_page(
                detail_documents[source_url],
                identifier=identifier,
                source_url=source_url,
            )
        )
    catalogue = DataFrame.from_records(records)
    catalogue["available_at_utc"] = pd.to_datetime(
        catalogue["available_at_utc"], utc=True
    )
    catalogue["release_cluster_id"] = catalogue["available_at_utc"].map(
        lambda value: "central_bank_" + pd.Timestamp(value).strftime("%Y%m%dT%H%M%SZ")
    )
    catalogue = catalogue.sort_values(
        ["available_at_utc", "bank"], kind="stable"
    ).reset_index(drop=True)
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: DataFrame) -> None:
    required = {
        "bank",
        "event_id",
        "decision_date",
        "available_at_utc",
        "local_timezone",
        "timing_quality",
        "scheduled",
        "source_url",
        "expected_consensus",
        "whole_event_partition",
        "release_cluster_id",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"Central-bank catalogue is missing: {required - set(catalogue)}")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("Central-bank event IDs are not unique")
    counts = catalogue.groupby("bank")["event_id"].size().to_dict()
    if counts != EXPECTED_BANK_COUNTS:
        raise ValueError(f"Unexpected central-bank counts: {counts}")
    status_counts = catalogue["scheduled"].value_counts().to_dict()
    if status_counts.get("unscheduled", 0) != 1:
        raise ValueError("Exactly one separately labelled unscheduled BoJ event is expected")
    if status_counts.get("rescheduled_scheduled", 0) != 1:
        raise ValueError("Exactly one rescheduled BoJ event is expected")
    if catalogue["expected_consensus"].notna().any():
        raise ValueError("Historical consensus must remain explicitly absent")
    if not catalogue["source_url"].map(_official_domain).all():
        raise ValueError("Every event requires an official source URL")
    for row in catalogue.itertuples(index=False):
        local_day = pd.Timestamp(row.available_at_utc).tz_convert(row.local_timezone).date()
        if local_day != date.fromisoformat(row.decision_date):
            raise ValueError("Local release date disagrees after UTC conversion")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby(
            ["bank", "whole_event_partition", "scheduled", "timing_quality"],
            sort=True,
        )
        .agg(events=("event_id", "nunique"), release_clocks=("release_cluster_id", "nunique"))
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    scheduled = catalogue.loc[catalogue["scheduled"].eq("scheduled")]
    return {
        "schema_version": 1,
        "status": "frozen_scheduled_central_banks_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "historical_consensus_available": False,
        "plain_question": (
            "Do scheduled ECB Bank of England or Bank of Japan decisions repeatedly "
            "make Bitcoin and Ethereum more active over the following 5-60 minutes?"
        ),
        "banks": ["ECB", "BoE", "BoJ"],
        "event_rows": len(catalogue),
        "scheduled_event_rows": len(scheduled),
        "release_clocks": int(catalogue["release_cluster_id"].nunique()),
        "partitions": [
            "development_2021_2023",
            "internal_validation_2024_2025",
        ],
        "market_test": {
            "assets": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "horizons_minutes": [5, 15, 30, 60],
            "minimum_events_per_partition": 10,
            "activity_definition": (
                "Median of absolute-return range and volume ratios versus clean "
                "prior same-weekday-and-clock periods."
            ),
            "minimum_median_activity_score": 1.20,
            "minimum_above_control_rate": 0.55,
            "variants": [
                "all_scheduled_decisions",
                "exclude_other_major_scheduled_event_within_4h",
            ],
            "retention_rule": (
                "The same bank asset and horizon must pass development and validation "
                "in both variants. An all-only pass is overlap-dependent."
            ),
        },
        "boundaries": [
            "The unscheduled and same-day rescheduled BoJ meetings are catalogue "
            "context and are not scored.",
            "BoE default noon clocks apply the official standard time plus known "
            "documented exceptions; this is not proof that every historical exception "
            "notice was discovered.",
            "ECB and BoE emergency decisions are excluded without exact official clocks.",
            "No direction is tested because official historical consensus is absent.",
            "No profit or trading-rule conclusion is allowed.",
        ],
        "counts": counts.to_dict(orient="records"),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Scheduled Central-Bank Catalogue",
            "",
            "- Complete scheduled catalogue plus two special 2020 BoJ records: "
            f"`{freeze['event_rows']}`",
            f"- Scheduled records eligible for later tests: `{freeze['scheduled_event_rows']}`",
            f"- Distinct exact release clocks: `{freeze['release_clocks']}`",
            "- ECB: official fixed schedule with the July 2022 time change.",
            "- Bank of England: official standard noon time with known documented "
            "August/November 2020 and May 2025 clock exceptions, plus the September "
            "2022 postponement. The archive check does not prove that no other "
            "historical exception exists.",
            "- Bank of Japan: actual time parsed from every individual statement page.",
            "- Historical market consensus: **not available in these official sources**.",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "The catalogue supports an activity test only. Direction must wait for a "
            "causally timestamped expectation source or a separately frozen market-"
            "confirmation question.",
            "",
        ]
    )


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_scheduled_central_bank_catalogue":
            raise ValueError("Existing central-bank catalogue result is invalid")
        return result
    initial_urls = [
        *(ECB_ARCHIVE_TEMPLATE.format(year=year) for year in range(START_YEAR, END_YEAR + 1)),
        ECB_TIME_RULE_URL,
        BOE_ARCHIVE_URL,
        BOE_NEWS_ARCHIVE_URL,
        BOE_TIME_RULE_URL,
        *(str(notice["url"]) for notice in BOE_TIMING_NOTICES.values()),
        BOJ_ARCHIVE_URL,
        BOJ_REFERENCE_ARCHIVE_URL,
        BOJ_RESCHEDULE_NOTICE_URL,
    ]
    initial_documents, initial_manifest = fetch_pages(initial_urls)
    boe_urls = parse_boe_archive(initial_documents[BOE_ARCHIVE_URL])
    boj_urls = [url for _, url in parse_boj_archive(initial_documents[BOJ_ARCHIVE_URL])]
    detail_documents, detail_manifest = fetch_pages([*boe_urls, *boj_urls])
    catalogue = build_catalogue(initial_documents, detail_documents)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(catalogue, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOG_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(
        {
            "retrieved_at_utc": g0.utc_now(),
            "sources": initial_manifest + detail_manifest,
        },
        SOURCE_MANIFEST_PATH,
    )
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_scheduled_central_bank_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "event_rows": len(catalogue),
        "scheduled_event_rows": int(catalogue["scheduled"].eq("scheduled").sum()),
        "source_pages": len(initial_manifest) + len(detail_manifest),
        "artifacts": {
            "catalogue": artifact(CATALOG_PATH),
            "counts": artifact(COUNTS_PATH),
            "source_manifest": artifact(SOURCE_MANIFEST_PATH),
            "freeze": artifact(FREEZE_PATH),
            "report": artifact(REPORT_PATH),
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
                    "expected_bank_counts": EXPECTED_BANK_COUNTS,
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
