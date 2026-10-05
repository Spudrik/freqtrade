"""Build a small, outcome-blind Federal Reserve liquidity-event catalogue.

The pilot is deliberately limited to three primary-source Federal Reserve press
releases that were selected before reading any crypto market outcomes.  It parses
the official page date and the exact ``For release at ...`` clock, converts that
clock with the named US Eastern timezone, and records only short action metadata.
It does not read OHLCV, news outcomes, model results, or historical expectations.

The parser fails if an exact official release time is absent.  A date-only page is
not silently converted to midnight because that would invent an intraday event
time and could bias later event studies.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

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
USER_DATA_DIR = REPO_ROOT / "user_data"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "fed_liquidity_catalogue_pilot_20260904a"
)
FED_DOMAIN = "federalreserve.gov"
FED_TIMEZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

URL_2020_MEASURES = (
    "https://www.federalreserve.gov/newsevents/pressreleases/"
    "monetary20200323b.htm"
)
URL_2023_BTFP = (
    "https://www.federalreserve.gov/newsevents/pressreleases/"
    "monetary20230312a.htm"
)
URL_2024_BTFP_END = (
    "https://www.federalreserve.gov/newsevents/pressreleases/"
    "monetary20240124a.htm"
)

TIMING_QUALITIES = {"actual_document_time"}
SCHEDULE_VALUES = {"scheduled", "unscheduled"}

REQUIRED_COLUMNS = (
    "event_id",
    "official_release_local_text",
    "available_at_utc",
    "timing_quality",
    "facility_action_type",
    "action_summary",
    "scheduled_or_unscheduled",
    "source_url",
    "source_retrieval_timestamp",
    "expectation_consensus",
)

OUTPUT_COLUMNS = REQUIRED_COLUMNS + (
    "release_date_text",
    "release_time_text",
    "official_title",
    "schedule_basis",
    "source_content_sha256",
)


# These are the three already source-justified pages.  The action descriptions are
# short, trader-readable metadata written before any market outcome is inspected.
# They are not assertions about price direction or trading usefulness.
PILOT_SPECS: tuple[dict[str, str], ...] = (
    {
        "event_id": "fed_2020_03_23_extensive_support_measures",
        "source_url": URL_2020_MEASURES,
        "facility_action_type": "asset_purchases_and_credit_facilities",
        "action_summary": (
            "Expanded Treasury and agency mortgage-backed-security purchases and "
            "announced credit facilities to support household, business, and "
            "municipal credit flow during the pandemic disruption."
        ),
        "scheduled_or_unscheduled": "unscheduled",
        "schedule_basis": (
            "Emergency support announcement; not a regular scheduled policy release."
        ),
    },
    {
        "event_id": "fed_2023_03_12_btfp_creation",
        "source_url": URL_2023_BTFP,
        "facility_action_type": "bank_term_funding_program_creation",
        "action_summary": (
            "Created the Bank Term Funding Program, offering up to one-year loans "
            "against eligible securities valued at par to provide bank liquidity."
        ),
        "scheduled_or_unscheduled": "unscheduled",
        "schedule_basis": (
            "Sunday banking-stress announcement; not a regular scheduled policy release."
        ),
    },
    {
        "event_id": "fed_2024_01_24_btfp_termination_notice",
        "source_url": URL_2024_BTFP_END,
        "facility_action_type": "bank_term_funding_program_termination_notice",
        "action_summary": (
            "Announced that the Bank Term Funding Program would stop making new "
            "loans on March 11 as scheduled and adjusted pricing until expiration."
        ),
        "scheduled_or_unscheduled": "scheduled",
        "schedule_basis": (
            "The release states that the program would cease new lending on its "
            "previously scheduled March 11 end date."
        ),
    },
)

EXPECTED_TITLE_FRAGMENTS = {
    URL_2020_MEASURES: "extensive new measures",
    URL_2023_BTFP: "additional funding",
    URL_2024_BTFP_END: "BTFP",
}


class OfficialReleaseTimeMissing(ValueError):
    """Raised when an official page does not expose an exact release clock."""


class FederalReservePageParser:
    """Extract the small set of fields needed from a Federal Reserve page."""

    _DATE_PATTERN = re.compile(
        r'<p\b[^>]*class=["\'][^"\']*\barticle__time\b[^"\']*["\'][^>]*>'
        r"(?P<value>.*?)</p>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _TIME_PATTERN = re.compile(
        r'<p\b[^>]*class=["\'][^"\']*\breleaseTime\b[^"\']*["\'][^>]*>'
        r"\s*For\s+release\s+at\s+"
        r"(?P<time>\d{1,2}:\d{2}\s*[ap]\.m\.)\s+"
        r"(?P<zone>EST|EDT)\b",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _TITLE_PATTERN = re.compile(
        r'<h3\b[^>]*class=["\'][^"\']*\btitle\b[^"\']*["\'][^>]*>'
        r"(?P<value>.*?)</h3>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def _clean_html_text(value: str) -> str:
        without_tags = re.sub(r"<[^>]+>", " ", value)
        return " ".join(unescape(without_tags).split())

    @classmethod
    def parse(cls, html: str) -> dict[str, str]:
        """Parse date, exact local release clock, and title from official HTML."""

        date_match = cls._DATE_PATTERN.search(html)
        if date_match is None:
            raise ValueError("official release date text is absent from Federal Reserve page")
        release_date_text = cls._clean_html_text(date_match.group("value"))
        try:
            release_date = datetime.strptime(
                release_date_text, "%B %d, %Y"
            ).date()
        except ValueError as exc:
            raise ValueError(
                f"unrecognised official release date text: {release_date_text!r}"
            ) from exc

        time_match = cls._TIME_PATTERN.search(html)
        if time_match is None:
            raise OfficialReleaseTimeMissing(
                "official release time absent; cannot derive available_at UTC "
                "without guessing an intraday timestamp"
            )
        release_time_text = " ".join(time_match.group("time").split())
        zone_text = time_match.group("zone").upper()
        time_for_parse = release_time_text.replace(".", "").upper()
        local_clock = datetime.strptime(time_for_parse, "%I:%M %p").time()
        local_release = datetime.combine(
            release_date, local_clock, tzinfo=FED_TIMEZONE
        )
        actual_zone = local_release.tzname()
        if actual_zone != zone_text:
            raise ValueError(
                "official timezone abbreviation does not match America/New_York "
                f"for {release_date_text}: page={zone_text}, zoneinfo={actual_zone}"
            )

        title_match = cls._TITLE_PATTERN.search(html)
        official_title = (
            cls._clean_html_text(title_match.group("value"))
            if title_match is not None
            else ""
        )
        if not official_title:
            raise ValueError("official release title is absent from Federal Reserve page")
        return {
            "release_date_text": release_date_text,
            "release_time_text": f"{release_time_text} {zone_text}",
            "official_release_local_text": (
                f"{release_date_text}; For release at {release_time_text} {zone_text}"
            ),
            "available_at_utc": _iso_utc(local_release),
            "official_title": official_title,
        }


def _iso_utc(value: datetime | pd.Timestamp | str) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not (
        host == FED_DOMAIN or host.endswith(f".{FED_DOMAIN}")
    ):
        raise ValueError(f"Non-official Federal Reserve source: {url}")
    if not parsed.path.startswith("/newsevents/pressreleases/"):
        raise ValueError(f"Unexpected Federal Reserve source path: {url}")


def _validate_spec(spec: Mapping[str, Any]) -> None:
    required = {
        "event_id",
        "source_url",
        "facility_action_type",
        "action_summary",
        "scheduled_or_unscheduled",
        "schedule_basis",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"Pilot spec missing fields: {missing}")
    _validate_source_url(str(spec["source_url"]))
    if str(spec["scheduled_or_unscheduled"]) not in SCHEDULE_VALUES:
        raise ValueError(f"Invalid schedule value: {spec['scheduled_or_unscheduled']}")
    if not str(spec["action_summary"]).strip():
        raise ValueError(f"Empty action summary: {spec['event_id']}")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        encoding = getattr(response, "encoding", None) or "utf-8"
        return bytes(content).decode(encoding, errors="replace")
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError("Federal Reserve response has neither content nor text")


def fetch_and_parse_spec(
    spec: Mapping[str, Any],
    *,
    retrieved_at_utc: str,
    requester: Callable[..., Any] = requests.get,
    timeout: int = 45,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch and parse one official page, returning a record and source metadata."""

    _validate_spec(spec)
    source_url = str(spec["source_url"])
    response = requester(
        source_url,
        headers={"User-Agent": "Objective02b-fed-liquidity-catalogue/1.0"},
        timeout=timeout,
    )
    status_code = int(response.status_code)
    if status_code != 200:
        raise RuntimeError(
            f"Official Federal Reserve source returned {status_code}: {source_url}"
        )
    raw_content = getattr(response, "content", None)
    if raw_content is None:
        content = _response_text(response).encode("utf-8")
    elif isinstance(raw_content, str):
        content = raw_content.encode("utf-8")
    else:
        content = bytes(raw_content)
    content_sha256 = hashlib.sha256(content).hexdigest()
    parsed = FederalReservePageParser.parse(_response_text(response))
    expected_fragment = EXPECTED_TITLE_FRAGMENTS.get(source_url)
    if expected_fragment and expected_fragment.casefold() not in parsed[
        "official_title"
    ].casefold():
        raise ValueError(
            f"Official title does not match the frozen pilot spec for {source_url}"
        )
    record = {
        "event_id": str(spec["event_id"]),
        **parsed,
        "timing_quality": "actual_document_time",
        "facility_action_type": str(spec["facility_action_type"]),
        "action_summary": str(spec["action_summary"]),
        "scheduled_or_unscheduled": str(spec["scheduled_or_unscheduled"]),
        "source_url": source_url,
        "source_retrieval_timestamp": retrieved_at_utc,
        "expectation_consensus": None,
        "schedule_basis": str(spec["schedule_basis"]),
        "source_content_sha256": content_sha256,
    }
    manifest = {
        "event_id": str(spec["event_id"]),
        "source_url": source_url,
        "http_status": status_code,
        "content_sha256": content_sha256,
        "retrieved_at_utc": retrieved_at_utc,
    }
    return record, manifest


def validate_catalogue(catalogue: DataFrame) -> None:
    """Validate the three-row catalogue without reading market outcomes."""

    missing = [column for column in OUTPUT_COLUMNS if column not in catalogue]
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")
    if len(catalogue) != len(PILOT_SPECS):
        raise ValueError(
            f"Pilot must contain exactly {len(PILOT_SPECS)} records, got {len(catalogue)}"
        )
    if catalogue["event_id"].duplicated().any():
        raise ValueError("Duplicate event identifiers detected.")
    expected_ids = {str(spec["event_id"]) for spec in PILOT_SPECS}
    actual_ids = set(catalogue["event_id"].astype(str))
    if actual_ids != expected_ids:
        raise ValueError(f"Pilot event IDs differ from frozen scope: {actual_ids}")
    if not catalogue["timing_quality"].isin(TIMING_QUALITIES).all():
        raise ValueError("Unexpected timing quality in catalogue.")
    if not catalogue["scheduled_or_unscheduled"].isin(SCHEDULE_VALUES).all():
        raise ValueError("Unexpected schedule value in catalogue.")
    if not catalogue["expectation_consensus"].isna().all():
        raise ValueError("Expectation/consensus must remain explicitly null in pilot.")
    for value in catalogue["available_at_utc"]:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("available_at_utc must be timezone-aware.")
    for value in catalogue["source_url"]:
        _validate_source_url(str(value))


def build_catalogue_from_pages(
    *,
    page_html: Mapping[str, str],
    retrieved_at_utc: str,
    content_hashes: Mapping[str, str] | None = None,
) -> DataFrame:
    """Build records from already-fetched official page text.

    This pure helper keeps parser tests independent from the network.  Production
    execution uses :func:`fetch_and_parse_spec` and records hashes of fetched bytes.
    """

    records: list[dict[str, Any]] = []
    for spec in PILOT_SPECS:
        _validate_spec(spec)
        source_url = str(spec["source_url"])
        if source_url not in page_html:
            raise ValueError(f"Missing official page fixture for {source_url}")
        parsed = FederalReservePageParser.parse(page_html[source_url])
        expected_fragment = EXPECTED_TITLE_FRAGMENTS.get(source_url)
        if expected_fragment and expected_fragment.casefold() not in parsed[
            "official_title"
        ].casefold():
            raise ValueError(
                f"Official title does not match the frozen pilot spec for {source_url}"
            )
        records.append(
            {
                "event_id": str(spec["event_id"]),
                **parsed,
                "timing_quality": "actual_document_time",
                "facility_action_type": str(spec["facility_action_type"]),
                "action_summary": str(spec["action_summary"]),
                "scheduled_or_unscheduled": str(spec["scheduled_or_unscheduled"]),
                "source_url": source_url,
                "source_retrieval_timestamp": retrieved_at_utc,
                "expectation_consensus": None,
                "schedule_basis": str(spec["schedule_basis"]),
                "source_content_sha256": (
                    content_hashes.get(source_url, "")
                    if content_hashes is not None
                    else ""
                ),
            }
        )
    catalogue = DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    validate_catalogue(catalogue)
    return catalogue


def _json_records(catalogue: DataFrame) -> list[dict[str, Any]]:
    records = catalogue.to_dict(orient="records")
    for record in records:
        if pd.isna(record["expectation_consensus"]):
            record["expectation_consensus"] = None
    return records


def render_report(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Federal Reserve liquidity catalogue pilot",
            "",
            f"- Status: `{freeze['status']}`",
            f"- Records: `{freeze['event_count']}`",
            "- Official pages: **three named Federal Reserve press releases only**",
            "- Market outcomes read: **No**",
            "- Profit or trading performance used: **No**",
            "- Historical expectation/consensus: **explicitly null**",
            "",
            "## Timing rule",
            "",
            (
                "The parser requires the page's exact `For release at ... EST/EDT` "
                "clock. It converts that clock using `America/New_York`, checks the "
                "seasonal abbreviation, and fails instead of guessing when the clock "
                "is absent."
            ),
            "",
            "## Scope limitation",
            "",
            (
                "This is a three-event data-foundation pilot, not a complete Federal "
                "Reserve history and not evidence that any event predicts crypto. "
                "Later tests must establish source coverage, controls, and whole-event "
                "confirmation before any market conclusion."
            ),
            "",
        ]
    )


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    """Fetch the frozen three-page scope and write a reproducible metadata pilot."""

    result_path = OUTPUT_ROOT / "fed_liquidity_catalogue_pilot_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_fed_liquidity_catalogue_pilot":
            raise ValueError("Existing Fed liquidity pilot result is not terminal.")
        return result

    run_retrieved_at = g0.utc_now()
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for spec in PILOT_SPECS:
        record, source_meta = fetch_and_parse_spec(
            spec,
            retrieved_at_utc=run_retrieved_at,
        )
        records.append(record)
        manifest.append(source_meta)
    catalogue = DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    validate_catalogue(catalogue)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    catalogue_path = OUTPUT_ROOT / "fed_liquidity_event_catalogue.csv"
    freeze_path = OUTPUT_ROOT / "fed_liquidity_catalogue_freeze.json"
    report_path = OUTPUT_ROOT / "fed_liquidity_catalogue_report.md"
    g0.atomic_write_csv(catalogue, catalogue_path)
    freeze = {
        "schema_version": 1,
        "status": "completed_fed_liquidity_catalogue_pilot",
        "created_at_utc": run_retrieved_at,
        "outcomes_read": False,
        "profit_used": False,
        "event_count": len(catalogue),
        "scope": "Exactly three frozen official Federal Reserve pages.",
        "expectation_consensus_policy": (
            "Null for every record; no historical expectations were reconstructed."
        ),
        "records": _json_records(catalogue),
        "source_manifest": manifest,
        "limitations": [
            "Three pages are a source pilot, not a complete Fed event family.",
            "No OHLCV, crypto prices, market outcomes, or model results were read.",
            "No expectation or consensus field was available or reconstructed.",
            (
                "Later market tests must use the exact available_at_utc timestamp "
                "and whole-event controls."
            ),
        ],
        "analysis_script": {
            "path": str(ANALYSIS_PATH.resolve()),
            "sha256": g0.sha256_file(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_fed_liquidity_catalogue_pilot",
        "created_at_utc": run_retrieved_at,
        "outcomes_read": False,
        "profit_used": False,
        "event_count": len(catalogue),
        "artifacts": {
            "catalogue": {
                "path": str(catalogue_path.resolve()),
                "sha256": g0.sha256_file(catalogue_path),
            },
            "freeze": {
                "path": str(freeze_path.resolve()),
                "sha256": g0.sha256_file(freeze_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": g0.sha256_file(report_path),
            },
        },
    }
    g0.atomic_write_json(result, result_path)
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
                    "output_root": str(OUTPUT_ROOT),
                    "scope_count": len(PILOT_SPECS),
                    "outcomes_will_be_read": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
