"""Freeze official Bank of Japan Tankan Summary clocks without reading markets."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
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
    "japan_tankan_catalogue_20260907a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "japan_tankan_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "japan_tankan_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "japan_tankan_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "japan_tankan_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "japan_tankan_catalogue_result.json"

BOJ_ROOT = "https://www.boj.or.jp"
ARCHIVE_URLS = (
    f"{BOJ_ROOT}/en/statistics/tk/gaiyo/2016/index.htm",
    f"{BOJ_ROOT}/en/statistics/tk/gaiyo/2021/index.htm",
)
FAQ_URL = f"{BOJ_ROOT}/en/statistics/outline/exp/tk/faqtk06.htm"
SOURCE_URLS = (*ARCHIVE_URLS, FAQ_URL)
EXPECTED_ROWS = 24
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b BoJ Tankan source audit"
JST = ZoneInfo("Asia/Tokyo")
RELEASE_CLOCK = "08:50"
PARTITIONS = {
    2020: "context_only_2020",
    2021: "development_2021_2023",
    2022: "development_2021_2023",
    2023: "development_2021_2023",
    2024: "internal_validation_2024_2025",
    2025: "internal_validation_2024_2025",
}
EXPECTED_RELEASES = {
    2020: ("2020-04-01", "2020-07-01", "2020-10-01", "2020-12-14"),
    2021: ("2021-04-01", "2021-07-01", "2021-10-01", "2021-12-13"),
    2022: ("2022-04-01", "2022-07-01", "2022-10-03", "2022-12-14"),
    2023: ("2023-04-03", "2023-07-03", "2023-10-02", "2023-12-13"),
    2024: ("2024-04-01", "2024-07-01", "2024-10-01", "2024-12-13"),
    2025: ("2025-04-01", "2025-07-01", "2025-10-01", "2025-12-15"),
}
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "June": 6,
    "Jul": 7,
    "July": 7,
    "Aug": 8,
    "Sep": 9,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
SURVEY_MONTHS = {"March": 3, "June": 6, "September": 9, "December": 12}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def normal_text(document: str | bytes) -> str:
    soup = BeautifulSoup(document, "html.parser")
    return " ".join(soup.get_text(" ", strip=True).replace("\xa0", " ").split())


def official_boj_url(url: str, *, pdf: bool | None = None) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    if not (
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "www.boj.or.jp"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.query == ""
        and parsed.fragment == ""
    ):
        return False
    is_pdf = (
        re.fullmatch(
            r"/en/statistics/tk/gaiyo/(?:2016|2021)/tka\d{4}\.pdf",
            parsed.path,
        )
        is not None
    )
    is_html = parsed.path in {
        "/en/statistics/tk/gaiyo/2016/index.htm",
        "/en/statistics/tk/gaiyo/2021/index.htm",
        "/en/statistics/outline/exp/tk/faqtk06.htm",
    }
    if pdf is True:
        return is_pdf
    if pdf is False:
        return is_html
    return is_html or is_pdf


def fetch_page(
    url: str,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    if not official_boj_url(url, pdf=False):
        raise ValueError(f"Non-official or unapproved BoJ page URL: {url}")
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"BoJ source returned {response.status_code}: {url}")
    if str(response.url) != url or not official_boj_url(str(response.url), pdf=False):
        raise ValueError("BoJ response did not come from the exact approved URL")
    document = content.decode(response.encoding or "utf-8", errors="strict")
    if "Bank of Japan" not in normal_text(document):
        raise ValueError(f"BoJ identity marker missing: {url}")
    return document, {
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "retrieved_at_utc": g0.utc_now(),
    }


def fetch_sources(
    *, requester: Callable[..., Any] = requests.get
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    documents: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for url in SOURCE_URLS:
        document, entry = fetch_page(url, requester=requester)
        documents[url] = document
        entries.append(entry)
    return documents, entries


def parse_archive_date(value: str) -> datetime:
    text = " ".join(value.replace("\xa0", " ").split())
    match = re.fullmatch(
        r"(Jan|Feb|Mar|Apr|May|Jun|June|Jul|July|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
        r"(\d{1,2}),\s+(\d{4})",
        text,
    )
    if match is None:
        raise ValueError(f"Unrecognised BoJ Tankan archive date: {value!r}")
    month, day, year = match.groups()
    return datetime(int(year), MONTHS[month], int(day))


def parse_archive(document: str, source_url: str) -> list[dict[str, str]]:
    if not official_boj_url(source_url, pdf=False) or source_url not in ARCHIVE_URLS:
        raise ValueError(f"Unapproved BoJ Tankan archive: {source_url}")
    records: list[dict[str, str]] = []
    soup = BeautifulSoup(document, "html.parser")
    for row in soup.select("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        date_text = cells[0].get_text(" ", strip=True)
        detail_text = " ".join(cells[1].get_text(" ", strip=True).split())
        detail_match = re.fullmatch(
            r"(March|June|September|December)\s+(\d{4})\s+Survey",
            detail_text,
        )
        if detail_match is None:
            continue
        pdf_links = [
            urljoin(source_url, str(link.get("href")))
            for link in cells[2].find_all("a", href=True)
            if str(link.get("href")).casefold().endswith(".pdf")
        ]
        if not pdf_links:
            continue
        if len(pdf_links) != 1 or not official_boj_url(pdf_links[0], pdf=True):
            raise ValueError(f"BoJ Tankan row lacks one approved Summary PDF: {detail_text}")
        release = parse_archive_date(date_text)
        survey_month_name, survey_year_text = detail_match.groups()
        records.append(
            {
                "release_date_local": release.date().isoformat(),
                "reference_period": (
                    f"{int(survey_year_text):04d}-{SURVEY_MONTHS[survey_month_name]:02d}"
                ),
                "source_archive_url": source_url,
                "source_artifact_url": pdf_links[0],
            }
        )
    return records


def validate_faq(document: str) -> None:
    text = normal_text(document)
    required = (
        "release times are always at 8:50 a.m. Japan Standard Time",
        "The Bank announces the Tankan release dates and times beforehand",
    )
    if not all(value.casefold() in text.casefold() for value in required):
        raise ValueError("BoJ Tankan FAQ no longer proves the frozen 08:50 JST rule")


def build_catalogue(documents: Mapping[str, str]) -> DataFrame:
    if set(documents) != set(SOURCE_URLS):
        raise ValueError("BoJ fetched document set differs from the frozen source list")
    validate_faq(documents[FAQ_URL])
    parsed = [
        record
        for source_url in ARCHIVE_URLS
        for record in parse_archive(documents[source_url], source_url)
    ]
    rows: list[dict[str, Any]] = []
    for record in parsed:
        release_date = datetime.fromisoformat(record["release_date_local"])
        if release_date.year not in PARTITIONS:
            continue
        local = pd.Timestamp(
            release_date.replace(hour=8, minute=50), tz=JST
        )
        rows.append(
            {
                "event_id": f"jp_boj_tankan_{record['reference_period'].replace('-', '')}",
                "event_family": "japan_boj_tankan_summary",
                "reference_period": record["reference_period"],
                "release_date_local": record["release_date_local"],
                "official_release_clock_local": RELEASE_CLOCK,
                "official_release_timestamp_local": local.isoformat(),
                "anchor_utc": local.tz_convert("UTC").isoformat(),
                "source_clock_rule_url": FAQ_URL,
                "source_archive_url": record["source_archive_url"],
                "source_artifact_url": record["source_artifact_url"],
                "timestamp_quality": (
                    "official_always_0850_jst_rule_not_server_posting_timestamp"
                ),
                "intraday_eligible": True,
                "postponement_status": "no_complete_historical_postponement_log",
                "market_expectation_status": "unavailable_not_imputed",
                "whole_event_partition": PARTITIONS[release_date.year],
                "selection_method": "complete_boj_tankan_summary_archive_2020_2025",
            }
        )
    catalogue = DataFrame.from_records(rows).sort_values(
        "anchor_utc", kind="stable"
    ).reset_index(drop=True)
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: DataFrame) -> None:
    required = {
        "event_id",
        "event_family",
        "reference_period",
        "release_date_local",
        "anchor_utc",
        "source_artifact_url",
        "timestamp_quality",
        "whole_event_partition",
        "market_expectation_status",
    }
    missing = sorted(required.difference(catalogue.columns))
    if missing:
        raise ValueError(f"BoJ Tankan catalogue lacks columns: {missing}")
    if len(catalogue) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} Tankan Summary releases, got {len(catalogue)}")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("BoJ Tankan event identifiers are not unique")
    observed = {
        year: tuple(
            catalogue.loc[
                pd.to_datetime(catalogue["release_date_local"]).dt.year.eq(year),
                "release_date_local",
            ]
        )
        for year in PARTITIONS
    }
    if observed != EXPECTED_RELEASES:
        raise ValueError(f"BoJ Tankan fixed release dates changed: {observed}")
    if not catalogue["official_release_clock_local"].eq(RELEASE_CLOCK).all():
        raise ValueError("BoJ Tankan release clock changed")
    if not catalogue["source_artifact_url"].map(
        lambda value: official_boj_url(str(value), pdf=True)
    ).all():
        raise ValueError("BoJ Tankan catalogue contains a non-official PDF")
    anchors = pd.to_datetime(catalogue["anchor_utc"], utc=True, errors="raise")
    if not anchors.is_monotonic_increasing or anchors.duplicated().any():
        raise ValueError("BoJ Tankan anchors are not unique and chronological")
    expected_partitions = pd.to_datetime(catalogue["release_date_local"]).dt.year.map(
        PARTITIONS
    )
    if not catalogue["whole_event_partition"].eq(expected_partitions).all():
        raise ValueError("BoJ Tankan whole-event partition changed")
    forbidden = ("profit", "future_return", "market_volume", "ohlcv")
    if any(
        token in column.casefold()
        for column in catalogue.columns
        for token in forbidden
    ):
        raise ValueError("BoJ Tankan source catalogue contains a market outcome column")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby("whole_event_partition", sort=True)
        .agg(
            releases=("event_id", "nunique"),
            first_release_utc=("anchor_utc", "min"),
            last_release_utc=("anchor_utc", "max"),
        )
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_japan_tankan_catalogue_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "plain_question": (
            "Can the official BoJ Summary archives and its always-08:50-JST rule supply "
            "a complete outcome-blind Tankan clock set for 2020-2025?"
        ),
        "release_rows": len(catalogue),
        "unique_release_clocks": int(catalogue["anchor_utc"].nunique()),
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            (
                "Each quarterly Tankan Summary is one event; later comprehensive data "
                "are not a new event."
            ),
            (
                "08:50 JST is the BoJ's official always-used release rule, not a "
                "measured server-posting time."
            ),
            "The official archive does not prove an exhaustive historical postponement log.",
            (
                "Tankan respondent forecasts are not market consensus and are not used "
                "as surprise labels."
            ),
            "No crypto, market outcome, direction, causation, or profit data was read.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Japan BoJ Tankan Source Catalogue",
            "",
            f"- Official quarterly Summary releases: `{freeze['release_rows']}`",
            f"- Unique nominal release clocks: `{freeze['unique_release_clocks']}`",
            "- Official release convention: `08:50 Asia/Tokyo`",
            "- Historical market expectations: **Unavailable; not guessed**",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "This complete 2020-2025 archive is suitable for an activity-first sensitivity "
            "test at the official BoJ clock. Its time must remain labelled as a nominal "
            "official release convention rather than a measured server-posting timestamp.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_japan_tankan_catalogue":
        raise ValueError("Existing BoJ Tankan catalogue result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing BoJ Tankan artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    documents, source_entries = fetch_sources()
    catalogue = build_catalogue(documents)
    counts = count_catalogue(catalogue)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        "schema_version": 1,
        "status": "frozen_official_boj_tankan_source_pages",
        "retrieved_at_utc": g0.utc_now(),
        "source_root": BOJ_ROOT,
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
        "status": "completed_japan_tankan_catalogue",
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
                    "expected_release_rows": EXPECTED_ROWS,
                    "source_pages": len(SOURCE_URLS),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
