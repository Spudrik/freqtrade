"""Freeze a small, outcome-blind official central-bank decision catalogue.

This pilot is a data-foundation task for Objective 02b.  It deliberately does
not read OHLCV, market features, news outcomes, or result JSON files.  The
catalogue is limited to three primary-source decisions for each of the ECB,
Bank of England, and Bank of Japan so later event tests can start from known
publication facts instead of remembered market moves.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
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
    / "central_bank_catalogue_pilot_20260904a"
)
INFORMATION_START = date(2020, 1, 1)
INFORMATION_CUTOFF = date(2026, 9, 4)
UTC = ZoneInfo("UTC")

ALLOWED_BANKS = {"ECB", "BoE", "BoJ"}
OFFICIAL_DOMAINS = {
    "ECB": "ecb.europa.eu",
    "BoE": "bankofengland.co.uk",
    "BoJ": "boj.or.jp",
}
TIMING_QUALITIES = {
    "actual_document_time",
    "official_fixed_schedule_time",
    "exception_notice_time",
    "date_only",
}
SCHEDULE_VALUES = {"scheduled", "unscheduled"}

REQUIRED_COLUMNS = (
    "bank",
    "event_id",
    "meeting_decision_date",
    "available_at_utc",
    "local_release_timestamp_text",
    "timing_quality",
    "scheduled",
    "decision_action",
    "vote_split",
    "guidance_summary",
    "official_source_urls",
    "expected_consensus",
    "source_retrieval_timestamp",
)
OUTPUT_COLUMNS = REQUIRED_COLUMNS + (
    "selection_role",
    "availability_note",
)


def _iso_utc(value: datetime) -> str:
    stamp = value
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _date_only(date_text: str) -> str:
    """Represent a date-only source with an explicit non-intraday sentinel."""

    return f"{date_text}T00:00:00Z"


def local_release_to_utc(date_text: str, time_text: str, timezone: str) -> str:
    """Convert a source-stated local release time and validate its zone."""

    local = datetime.strptime(
        f"{date_text} {time_text}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=ZoneInfo(timezone))
    return _iso_utc(local)


def _urls(*values: str) -> str:
    return json.dumps(list(values), separators=(",", ":"))


# These are deliberately hand-selected representative records, not a claimed
# complete event family.  Their source pages are all official primary sources.
# Date-only pages do not provide a trustworthy intraday publication timestamp;
# their UTC midnight value is a sentinel and must not be used as an exact clock.
PILOT_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "bank": "ECB",
        "event_id": "ecb_2020_03_18_pepp_emergency",
        "meeting_decision_date": "2020-03-18",
        "available_at_utc": _date_only("2020-03-18"),
        "local_release_timestamp_text": "18 March 2020 (page gives date, not time)",
        "timing_quality": "date_only",
        "scheduled": "unscheduled",
        "decision_action": (
            "Launched the temporary EUR 750 billion Pandemic Emergency Purchase "
            "Programme (PEPP) with flexible purchases."
        ),
        "vote_split": None,
        "guidance_summary": (
            "The response was intended to counter pandemic risks to monetary-policy "
            "transmission and the euro-area outlook."
        ),
        "official_source_urls": _urls(
            "https://www.ecb.europa.eu/press/pr/date/2020/html/"
            "ecb.pr200318_1~3949d6f266.en.html",
            "https://www.ecb.europa.eu/press/accounts/2020/html/"
            "ecb.mg200409_1~baf4b2ad06.en.html",
        ),
        "expected_consensus": None,
        "selection_role": "timing_exception_or_emergency",
        "availability_note": (
            "ECB decision and official meeting-account pages expose dates but no exact "
            "document time; do not use for intraday anchoring without a separate official clock."
        ),
    },
    {
        "bank": "ECB",
        "event_id": "ecb_2022_07_21_rate_hike_tpi",
        "meeting_decision_date": "2022-07-21",
        "available_at_utc": _date_only("2022-07-21"),
        "local_release_timestamp_text": "21 July 2022 (page gives date, not time)",
        "timing_quality": "date_only",
        "scheduled": "scheduled",
        "decision_action": (
            "Raised the three key ECB interest rates by 50 basis points and "
            "approved the Transmission Protection Instrument."
        ),
        "vote_split": None,
        "guidance_summary": (
            "The Governing Council cited inflation risks and the need to protect "
            "monetary-policy transmission."
        ),
        "official_source_urls": _urls(
            "https://www.ecb.europa.eu/press/pr/date/2022/html/"
            "ecb.mp220721~53e5bdd317.en.html"
        ),
        "expected_consensus": None,
        "selection_role": "changed_policy_or_instrument",
        "availability_note": (
            "The official page gives the decision date only in its machine-readable "
            "publication metadata."
        ),
    },
    {
        "bank": "ECB",
        "event_id": "ecb_2024_07_18_rates_unchanged",
        "meeting_decision_date": "2024-07-18",
        "available_at_utc": _date_only("2024-07-18"),
        "local_release_timestamp_text": "18 July 2024 (page gives date, not time)",
        "timing_quality": "date_only",
        "scheduled": "scheduled",
        "decision_action": "Kept the three key ECB interest rates unchanged.",
        "vote_split": None,
        "guidance_summary": (
            "The Council retained a data-dependent, meeting-by-meeting approach "
            "while financing conditions remained restrictive."
        ),
        "official_source_urls": _urls(
            "https://www.ecb.europa.eu/press/pr/date/2024/html/"
            "ecb.mp240718~b9e0ddd9d5.en.html"
        ),
        "expected_consensus": None,
        "selection_role": "unchanged_decision",
        "availability_note": (
            "The official page gives the decision date but not a verified intraday "
            "publication time."
        ),
    },
    {
        "bank": "BoE",
        "event_id": "boe_2020_03_19_special_easing",
        "meeting_decision_date": "2020-03-19",
        "available_at_utc": _date_only("2020-03-19"),
        "local_release_timestamp_text": "Published on 19 March 2020; special MPC meeting",
        "timing_quality": "date_only",
        "scheduled": "unscheduled",
        "decision_action": (
            "Cut Bank Rate by 15 basis points to 0.1%, increased government and "
            "corporate bond purchases by GBP 200 billion, and enlarged TFSME."
        ),
        "vote_split": "unanimous for the stated measures",
        "guidance_summary": (
            "The MPC cited rapidly tightening UK and global financial conditions "
            "during the COVID-19 shock."
        ),
        "official_source_urls": _urls(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2020/monetary-policy-summary-for-the-special-monetary-policy-committee-"
            "meeting-on-19-march-2020"
        ),
        "expected_consensus": None,
        "selection_role": "timing_exception_or_emergency",
        "availability_note": (
            "The official page records publication date but no exact publication "
            "time; intraday use requires a separate timestamp source."
        ),
    },
    {
        "bank": "BoE",
        "event_id": "boe_2021_12_15_rate_hike",
        "meeting_decision_date": "2021-12-15",
        "available_at_utc": _date_only("2021-12-16"),
        "local_release_timestamp_text": "Published on 16 December 2021; meeting ended 15 December",
        "timing_quality": "date_only",
        "scheduled": "scheduled",
        "decision_action": (
            "Raised Bank Rate by 15 basis points to 0.25% and maintained the "
            "existing corporate and government bond purchase stocks."
        ),
        "vote_split": "8-1 for the Bank Rate increase; purchase stocks unanimous",
        "guidance_summary": (
            "The MPC said incoming labour-market and cost-pressure information "
            "supported a modest tightening step."
        ),
        "official_source_urls": _urls(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2021/december-2021."
        ),
        "expected_consensus": None,
        "selection_role": "changed_policy_or_instrument",
        "availability_note": (
            "Publication is dated the day after the meeting; exact release time is "
            "not exposed by the official page."
        ),
    },
    {
        "bank": "BoE",
        "event_id": "boe_2023_09_20_rate_unchanged",
        "meeting_decision_date": "2023-09-20",
        "available_at_utc": _date_only("2023-09-21"),
        "local_release_timestamp_text": (
            "Published on 21 September 2023; meeting ended 20 September"
        ),
        "timing_quality": "date_only",
        "scheduled": "scheduled",
        "decision_action": (
            "Maintained Bank Rate at 5.25% and reduced the government-bond purchase "
            "stock by GBP 100 billion over the following twelve months."
        ),
        "vote_split": "5-4 to maintain Bank Rate; gilt-stock reduction unanimous",
        "guidance_summary": (
            "The MPC noted restrictive conditions and retained a data-dependent "
            "assessment of inflation persistence."
        ),
        "official_source_urls": _urls(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2023/september-2023"
        ),
        "expected_consensus": None,
        "selection_role": "unchanged_decision",
        "availability_note": (
            "Publication is dated the day after the meeting; exact release time is "
            "not exposed by the official page."
        ),
    },
    {
        "bank": "BoJ",
        "event_id": "boj_2020_03_16_covid_easing",
        "meeting_decision_date": "2020-03-16",
        "available_at_utc": local_release_to_utc(
            "2020-03-16", "14:06", "Asia/Tokyo"
        ),
        "local_release_timestamp_text": (
            "Monday, March 16 at 14:06 Japan Standard Time"
        ),
        "timing_quality": "actual_document_time",
        "scheduled": "unscheduled",
        "decision_action": (
            "Enhanced easing through a new corporate-financing operation, larger "
            "CP/corporate-bond purchase limits, and active ETF/J-REIT purchases."
        ),
        "vote_split": (
            "7-2 for maintaining the YCC market-operation guideline; other stated "
            "measures unanimous"
        ),
        "guidance_summary": (
            "The Bank responded to unstable financial and capital markets and the "
            "COVID-19 outbreak."
        ),
        "official_source_urls": _urls(
            "https://www.boj.or.jp/en/mopo/mpmdeci/state_2020/k200316b.htm",
            "https://www.boj.or.jp/en/mopo/mpmsche_minu/m_ref/rel200316l.htm",
        ),
        "expected_consensus": None,
        "selection_role": "timing_exception_or_emergency",
        "availability_note": (
            "The decision page states the exact publication time; the second official "
            "page documents the 2020 schedule change."
        ),
    },
    {
        "bank": "BoJ",
        "event_id": "boj_2022_12_20_ycc_change",
        "meeting_decision_date": "2022-12-20",
        "available_at_utc": local_release_to_utc(
            "2022-12-20", "12:01", "Asia/Tokyo"
        ),
        "local_release_timestamp_text": (
            "Tuesday, December 20, 2022 at 12:01 Japan Standard Time"
        ),
        "timing_quality": "actual_document_time",
        "scheduled": "scheduled",
        "decision_action": (
            "Modified the conduct of yield-curve control to improve market functioning "
            "while maintaining accommodative financial conditions."
        ),
        "vote_split": "unanimous for the YCC guideline",
        "guidance_summary": (
            "The Bank cited deteriorating bond-market functioning and the need to "
            "improve transmission of monetary easing."
        ),
        "official_source_urls": _urls(
            "https://www.boj.or.jp/en/mopo/mpmdeci/state_2022/k221220a.htm"
        ),
        "expected_consensus": None,
        "selection_role": "changed_policy_or_instrument",
        "availability_note": "The official decision page states the exact release time.",
    },
    {
        "bank": "BoJ",
        "event_id": "boj_2024_01_23_policy_unchanged",
        "meeting_decision_date": "2024-01-23",
        "available_at_utc": local_release_to_utc(
            "2024-01-23", "12:09", "Asia/Tokyo"
        ),
        "local_release_timestamp_text": (
            "Tuesday, January 23 at 12:09 Japan Standard Time"
        ),
        "timing_quality": "actual_document_time",
        "scheduled": "scheduled",
        "decision_action": (
            "Maintained the yield-curve-control and asset-purchase guidelines, "
            "including the -0.1% policy-rate balance rate."
        ),
        "vote_split": "unanimous for the stated YCC and asset-purchase guidelines",
        "guidance_summary": (
            "The Bank said it would patiently continue easing while responding to "
            "economic, price, and financial developments."
        ),
        "official_source_urls": _urls(
            "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240123a.htm"
        ),
        "expected_consensus": None,
        "selection_role": "unchanged_decision",
        "availability_note": "The official decision page states the exact release time.",
    },
)


def _parse_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError("official_source_urls must be a non-empty URL list.")
    return parsed


def _validate_source_url(bank: str, url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    required_domain = OFFICIAL_DOMAINS[bank]
    if parsed.scheme != "https" or not (
        host == required_domain or host.endswith(f".{required_domain}")
    ):
        raise ValueError(f"Non-official source for {bank}: {url}")


def _validate_record(row: Mapping[str, Any]) -> None:
    bank = str(row["bank"])
    decision_date = date.fromisoformat(str(row["meeting_decision_date"]))
    if not INFORMATION_START <= decision_date < INFORMATION_CUTOFF:
        raise ValueError(f"Decision date outside pilot window: {row}")
    timing_quality = str(row["timing_quality"])
    if timing_quality not in TIMING_QUALITIES:
        raise ValueError(f"Invalid timing quality: {timing_quality}")
    schedule = str(row["scheduled"])
    if schedule not in SCHEDULE_VALUES:
        raise ValueError(f"Invalid schedule value: {schedule}")
    available = datetime.fromisoformat(
        str(row["available_at_utc"]).replace("Z", "+00:00")
    )
    if available.tzinfo is None or available.utcoffset() is None:
        raise ValueError(f"available_at_utc is not timezone-aware: {row}")
    if timing_quality == "date_only" and (
        available.hour,
        available.minute,
        available.second,
    ) != (0, 0, 0):
        raise ValueError("date_only records must use the UTC-midnight sentinel.")
    if row["expected_consensus"] is not None and not pd.isna(
        row["expected_consensus"]
    ):
        raise ValueError("Expected/consensus must remain explicitly null in pilot.")
    for url in _parse_urls(row["official_source_urls"]):
        _validate_source_url(bank, url)


def validate_catalogue(catalogue: DataFrame) -> None:
    """Validate the small catalogue without looking at market outcomes."""

    missing = [column for column in REQUIRED_COLUMNS if column not in catalogue]
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")
    if catalogue.empty:
        raise ValueError("Catalogue cannot be empty.")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("Duplicate event identifiers detected.")
    if set(catalogue["bank"].astype(str)) != ALLOWED_BANKS:
        raise ValueError("Pilot must contain exactly the three requested banks.")
    counts = catalogue.groupby("bank")["event_id"].size().to_dict()
    if counts != {"ECB": 3, "BoE": 3, "BoJ": 3}:
        raise ValueError(f"Pilot requires exactly three records per bank, got {counts}")
    for row in catalogue.to_dict(orient="records"):
        _validate_record(row)


def build_catalogue(*, retrieved_at_utc: str) -> DataFrame:
    """Return the frozen pilot rows with one source-retrieval timestamp."""

    records = copy.deepcopy(list(PILOT_EVENTS))
    for record in records:
        record["source_retrieval_timestamp"] = retrieved_at_utc
    catalogue = DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    validate_catalogue(catalogue)
    return catalogue


def _source_urls(catalogue: DataFrame) -> list[str]:
    urls: list[str] = []
    for value in catalogue["official_source_urls"]:
        for url in _parse_urls(value):
            if url not in urls:
                urls.append(url)
    return urls


def fetch_source_manifest(
    urls: Iterable[str],
    *,
    retrieved_at_utc: str,
    requester: Callable[..., Any] = requests.get,
    timeout: int = 45,
) -> list[dict[str, Any]]:
    """Fetch only official pages and retain metadata, never long source text."""

    manifest: list[dict[str, Any]] = []
    for url in urls:
        bank = next(
            bank
            for bank, domain in OFFICIAL_DOMAINS.items()
            if (urlparse(url).hostname or "").casefold().endswith(domain)
        )
        _validate_source_url(bank, url)
        response = requester(
            url,
            headers={"User-Agent": "Objective02b-central-bank-catalogue/1.0"},
            timeout=timeout,
        )
        content = bytes(response.content)
        if int(response.status_code) != 200:
            raise RuntimeError(f"Official source returned {response.status_code}: {url}")
        manifest.append(
            {
                "bank": bank,
                "url": url,
                "http_status": int(response.status_code),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "retrieved_at_utc": retrieved_at_utc,
            }
        )
    return manifest


def _json_records(catalogue: DataFrame) -> list[dict[str, Any]]:
    records = catalogue.to_dict(orient="records")
    for record in records:
        record["official_source_urls"] = _parse_urls(record["official_source_urls"])
        if pd.isna(record["expected_consensus"]):
            record["expected_consensus"] = None
        if pd.isna(record["vote_split"]):
            record["vote_split"] = None
    return records


def render_report(freeze: Mapping[str, Any]) -> str:
    counts = freeze["counts_by_bank"]
    timing = freeze["timing_quality_counts"]
    return "\n".join(
        [
            "# Central-bank catalogue pilot",
            "",
            f"- Status: `{freeze['status']}`",
            f"- Records: `{freeze['event_count']}` ({counts})",
            f"- Timing qualities: `{timing}`",
            "- Market outcomes read: **No**",
            "- Expected/consensus values: **explicitly null; not reconstructed**",
            "",
            "## Scope",
            "",
            (
                "This is a small official-source catalogue, not a complete central-bank "
                "history. It supplies three representative ECB, BoE, and BoJ decisions "
                "for later event-scoped tests."
            ),
            "",
            "## Clock limitation",
            "",
            (
                "BoJ records have document times from the official pages. ECB and BoE "
                "pages in this pilot expose dates but not verified intraday publication "
                "times, so their UTC-midnight values are date-only sentinels and must "
                "not anchor minute-level market tests."
            ),
            "",
        ]
    )


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "central_bank_catalogue_pilot_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_central_bank_catalogue_pilot":
            raise ValueError("Existing central-bank pilot result is not terminal.")
        return result

    retrieved_at_utc = g0.utc_now()
    catalogue = build_catalogue(retrieved_at_utc=retrieved_at_utc)
    manifest = fetch_source_manifest(
        _source_urls(catalogue), retrieved_at_utc=retrieved_at_utc
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    catalogue_path = OUTPUT_ROOT / "central_bank_event_catalogue.csv"
    freeze_path = OUTPUT_ROOT / "central_bank_catalogue_freeze.json"
    report_path = OUTPUT_ROOT / "central_bank_catalogue_report.md"
    g0.atomic_write_csv(catalogue, catalogue_path)
    freeze = {
        "schema_version": 1,
        "status": "completed_central_bank_catalogue_pilot",
        "created_at_utc": retrieved_at_utc,
        "outcomes_read": False,
        "event_count": len(catalogue),
        "counts_by_bank": {
            bank: int((catalogue["bank"] == bank).sum()) for bank in sorted(ALLOWED_BANKS)
        },
        "timing_quality_counts": {
            quality: int((catalogue["timing_quality"] == quality).sum())
            for quality in sorted(TIMING_QUALITIES)
            if bool((catalogue["timing_quality"] == quality).any())
        },
        "events": _json_records(catalogue),
        "source_manifest": manifest,
        "scope_limit": "Three representative completed decisions per bank; not a complete family.",
        "limitations": [
            "No historical expectation or consensus is attached.",
            "Date-only ECB and BoE pages cannot support exact intraday anchoring.",
            "The catalogue is for later event tests, not a trading rule or market result.",
        ],
    }
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_central_bank_catalogue_pilot",
        "created_at_utc": retrieved_at_utc,
        "outcomes_read": False,
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
