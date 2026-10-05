"""Build an outcome-blind official SEC hyperscaler 8-K catalogue."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
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
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "sec_hyperscaler_8k_catalogue_20260906a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "sec_hyperscaler_8k_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "sec_hyperscaler_8k_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "sec_hyperscaler_8k_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "sec_hyperscaler_8k_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "sec_hyperscaler_8k_catalogue_result.json"

SEC_DATA_ROOT = "https://data.sec.gov"
SEC_ARCHIVES_ROOT = "https://www.sec.gov"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2026-01-01T00:00:00Z")
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b SEC catalogue research@example.com"
FORMS = frozenset({"8-K", "8-K/A"})
EARNINGS_ITEM_CODE = "2.02"
PARTITIONS = {
    2020: "context_only_2020",
    2021: "development_2021_2023",
    2022: "development_2021_2023",
    2023: "development_2021_2023",
    2024: "internal_validation_2024_2025",
    2025: "internal_validation_2024_2025",
}
COMPANIES: dict[str, dict[str, str]] = {
    "0000789019": {
        "company_key": "microsoft",
        "company_name": "MICROSOFT CORP",
        "ticker": "MSFT",
    },
    "0001652044": {
        "company_key": "alphabet",
        "company_name": "Alphabet Inc.",
        "ticker": "GOOGL",
    },
    "0001018724": {
        "company_key": "amazon",
        "company_name": "AMAZON COM INC",
        "ticker": "AMZN",
    },
    "0001326801": {
        "company_key": "meta",
        "company_name": "Meta Platforms, Inc.",
        "ticker": "META",
    },
}
SUBMISSION_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
    "items",
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def official_submission_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "data.sec.gov"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
        and re.fullmatch(
            r"/submissions/(?:CIK\d{10}|CIK\d{10}-submissions-\d{3})\.json",
            parsed.path,
        )
    )


def official_filing_url(url: str, cik: str, accession_number: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    accession_digits = accession_number.replace("-", "")
    expected_prefix = f"/Archives/edgar/data/{int(cik)}/{accession_digits}/"
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "www.sec.gov"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith(expected_prefix)
        and parsed.path != expected_prefix
    )


def main_submission_url(cik: str) -> str:
    if cik not in COMPANIES:
        raise ValueError(f"CIK is outside the fixed SEC catalogue: {cik}")
    return f"{SEC_DATA_ROOT}/submissions/CIK{cik}.json"


def supplemental_submission_url(cik: str, name: str) -> str:
    expected = rf"CIK{cik}-submissions-\d{{3}}\.json"
    if re.fullmatch(expected, name) is None:
        raise ValueError(f"Invalid SEC supplemental filename for CIK {cik}: {name}")
    return f"{SEC_DATA_ROOT}/submissions/{name}"


def fetch_json(
    url: str,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch exactly one official SEC JSON document without retries."""
    if not official_submission_url(url):
        raise ValueError(f"Non-official SEC submission URL: {url}")
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    status = int(response.status_code)
    if status != 200:
        raise RuntimeError(f"SEC source returned {status}: {url}")
    if not official_submission_url(str(response.url)):
        raise ValueError("SEC response did not come from the approved official URL")
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"SEC source is not valid UTF-8 JSON: {url}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"SEC source root is not an object: {url}")
    return document, {
        "url": url,
        "http_status": status,
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def validate_main_identity(main_document: Mapping[str, Any], cik: str) -> None:
    expected = COMPANIES[cik]
    reported_cik = str(main_document.get("cik", "")).zfill(10)
    if reported_cik != cik:
        raise ValueError(f"SEC main submission CIK disagrees with fixed mapping: {cik}")
    if main_document.get("name") != expected["company_name"]:
        raise ValueError(f"SEC main submission name disagrees with fixed mapping: {cik}")


def supplement_references(main_document: Mapping[str, Any], cik: str) -> list[dict[str, Any]]:
    filings = main_document.get("filings")
    if not isinstance(filings, Mapping) or not isinstance(filings.get("files"), list):
        raise ValueError(f"SEC main submission has no historical-file bindings: {cik}")
    selected: list[dict[str, Any]] = []
    for value in filings["files"]:
        if not isinstance(value, Mapping):
            raise ValueError(f"SEC historical-file binding is invalid: {cik}")
        name = value.get("name")
        filing_from = value.get("filingFrom")
        filing_to = value.get("filingTo")
        if not all(isinstance(item, str) for item in (name, filing_from, filing_to)):
            raise ValueError(f"SEC historical-file binding is incomplete: {cik}")
        range_start = pd.Timestamp(f"{filing_from}T00:00:00Z")
        range_end = pd.Timestamp(f"{filing_to}T23:59:59.999999999Z")
        if range_start.tz is None or range_end.tz is None or range_end < range_start:
            raise ValueError(f"SEC historical-file dates are invalid: {cik} {name}")
        if range_end >= START and range_start < END_EXCLUSIVE:
            selected.append(
                {
                    "name": name,
                    "filing_from": filing_from,
                    "filing_to": filing_to,
                    "url": supplemental_submission_url(cik, name),
                }
            )
    return sorted(selected, key=lambda item: str(item["name"]))


def submission_rows(
    document: Mapping[str, Any],
    *,
    cik: str,
    source_url: str,
    source_kind: str,
) -> list[dict[str, Any]]:
    missing = [column for column in SUBMISSION_COLUMNS if column not in document]
    if missing:
        raise ValueError(f"SEC submission rows are missing {missing}: {source_url}")
    non_lists = [
        column for column in SUBMISSION_COLUMNS if not isinstance(document[column], list)
    ]
    if non_lists:
        raise ValueError(f"SEC submission columns are not lists {non_lists}: {source_url}")
    lengths = {len(document[column]) for column in SUBMISSION_COLUMNS}
    if len(lengths) != 1:
        raise ValueError(f"SEC submission columns have inconsistent lengths: {source_url}")
    records = []
    for values in zip(*(document[column] for column in SUBMISSION_COLUMNS), strict=True):
        record = dict(zip(SUBMISSION_COLUMNS, values, strict=True))
        record.update({"cik": cik, "submission_source_url": source_url, "source_kind": source_kind})
        records.append(record)
    return records


def parse_item_codes(items: str) -> list[str]:
    if not isinstance(items, str):
        raise ValueError("SEC 8-K items value must be a string")
    codes = [value.strip() for value in items.split(",") if value.strip()]
    if any(re.fullmatch(r"\d\.\d{2}", value) is None for value in codes):
        raise ValueError(f"SEC 8-K item code is invalid: {items}")
    if len(codes) != len(set(codes)):
        raise ValueError(f"SEC 8-K item codes are duplicated: {items}")
    return codes


def exact_utc_timestamp(value: Any) -> pd.Timestamp:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value
    ) is None:
        raise ValueError(f"SEC acceptanceDateTime is not an exact UTC timestamp: {value}")
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError(f"SEC acceptanceDateTime is not UTC: {value}")
    return timestamp


def filing_source_url(cik: str, accession_number: str, primary_document: str) -> str:
    if re.fullmatch(r"\d{10}", cik) is None:
        raise ValueError(f"Invalid fixed CIK: {cik}")
    if re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession_number) is None:
        raise ValueError(f"Invalid SEC accession number: {accession_number}")
    if not primary_document or primary_document.startswith("/") or ".." in primary_document:
        raise ValueError(f"Invalid SEC primary document: {primary_document}")
    return (
        f"{SEC_ARCHIVES_ROOT}/Archives/edgar/data/{int(cik)}/"
        f"{accession_number.replace('-', '')}/{primary_document}"
    )


def build_catalogue(records: Sequence[Mapping[str, Any]]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record["form"]) not in FORMS:
            continue
        cik = str(record["cik"])
        filing_date = pd.Timestamp(str(record["filingDate"]))
        if filing_date.tz is not None or filing_date != filing_date.normalize():
            raise ValueError("SEC filingDate is not an exact date")
        accepted = exact_utc_timestamp(record["acceptanceDateTime"])
        if not (START <= accepted < END_EXCLUSIVE):
            continue
        if not (START.date() <= filing_date.date() < END_EXCLUSIVE.date()):
            continue
        accession_number = str(record["accessionNumber"])
        primary_document = str(record["primaryDocument"])
        source_url = filing_source_url(cik, accession_number, primary_document)
        if not official_filing_url(source_url, cik, accession_number):
            raise ValueError(f"SEC filing URL is not canonical: {source_url}")
        item_codes = parse_item_codes(record["items"])
        report_date = str(record["reportDate"])
        if report_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", report_date) is None:
            raise ValueError(f"SEC reportDate is invalid: {report_date}")
        company = COMPANIES[cik]
        rows.append(
            {
                "event_id": f"sec_{cik}_{accession_number.replace('-', '')}",
                "cik": cik,
                "company_key": company["company_key"],
                "company_name": company["company_name"],
                "ticker": company["ticker"],
                "form": str(record["form"]),
                "acceptance_datetime_utc": str(record["acceptanceDateTime"]),
                "filing_date": filing_date.date().isoformat(),
                "report_date": report_date,
                "accession_number": accession_number,
                "primary_document": primary_document,
                "items": str(record["items"]),
                "item_codes_json": json.dumps(item_codes, separators=(",", ":")),
                "earnings_like_candidate": EARNINGS_ITEM_CODE in item_codes,
                "source_url": source_url,
                "submission_source_url": str(record["submission_source_url"]),
                "submission_source_kind": str(record["source_kind"]),
                "timestamp_quality": "sec_acceptance_datetime_exact_utc",
                "acceptance_time_boundary": (
                    "SEC_acceptance_time_not_claimed_as_first_public_news_time"
                ),
                "market_expectation_status": "unavailable_not_imputed",
                "whole_event_partition": PARTITIONS[accepted.year],
                "selection_method": "complete_fixed_cik_sec_8k_and_8k_amendment_filings",
            }
        )
    catalogue = DataFrame.from_records(rows)
    if catalogue.empty:
        raise ValueError("SEC catalogue has no 8-K rows in the fixed period")
    catalogue = catalogue.sort_values(
        ["cik", "accession_number", "submission_source_kind"], kind="stable"
    ).reset_index(drop=True)
    duplicated = catalogue.duplicated(["cik", "accession_number"], keep=False)
    if duplicated.any():
        comparison_columns = [
            column
            for column in catalogue.columns
            if column
            not in {
                "cik",
                "accession_number",
                "submission_source_url",
                "submission_source_kind",
            }
        ]
        disagreements = (
            catalogue.loc[duplicated]
            .groupby(["cik", "accession_number"], dropna=False)[comparison_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if disagreements.any():
            raise ValueError("Duplicate SEC CIK/accession records disagree")
        catalogue = catalogue.drop_duplicates(["cik", "accession_number"], keep="first")
    catalogue = (
        catalogue.assign(
            _accepted=pd.to_datetime(
                catalogue["acceptance_datetime_utc"], utc=True, format="ISO8601"
            )
        )
        .sort_values(["_accepted", "cik", "accession_number"], kind="stable")
        .drop(columns="_accepted")
        .reset_index(drop=True)
    )
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue_shape(catalogue: DataFrame) -> None:
    required = {
        "event_id", "cik", "company_key", "company_name", "ticker", "form",
        "acceptance_datetime_utc", "filing_date", "report_date", "accession_number",
        "primary_document", "items", "item_codes_json", "earnings_like_candidate",
        "source_url", "submission_source_url", "submission_source_kind",
        "timestamp_quality", "acceptance_time_boundary", "market_expectation_status",
        "whole_event_partition", "selection_method",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"SEC catalogue is missing: {required - set(catalogue)}")
    forbidden_tokens = ("profit", "future_return", "price", "volume", "direction")
    if any(
        token in column.casefold()
        for column in catalogue.columns
        for token in forbidden_tokens
    ):
        raise ValueError("SEC source catalogue contains a market outcome column")
    if catalogue.duplicated(["cik", "accession_number"]).any():
        raise ValueError("SEC catalogue CIK/accession pairs are not unique")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("SEC catalogue event IDs are not unique")
    if set(catalogue["cik"]) != set(COMPANIES):
        raise ValueError("SEC catalogue does not cover every fixed CIK")
    if not set(catalogue["form"]).issubset(FORMS):
        raise ValueError("SEC catalogue contains a non-8-K form")


def validate_catalogue_constants(catalogue: DataFrame) -> None:
    timestamps = pd.to_datetime(catalogue["acceptance_datetime_utc"], utc=True, format="ISO8601")
    if not (timestamps.ge(START) & timestamps.lt(END_EXCLUSIVE)).all():
        raise ValueError("SEC catalogue timestamp is outside the fixed period")
    if not all(
        official_filing_url(row.source_url, row.cik, row.accession_number)
        for row in catalogue.itertuples(index=False)
    ):
        raise ValueError("SEC catalogue has a non-official filing URL")
    if not catalogue["submission_source_url"].map(official_submission_url).all():
        raise ValueError("SEC catalogue has a non-official submission URL")
    fixed = {
        "timestamp_quality": "sec_acceptance_datetime_exact_utc",
        "acceptance_time_boundary": "SEC_acceptance_time_not_claimed_as_first_public_news_time",
        "market_expectation_status": "unavailable_not_imputed",
        "selection_method": "complete_fixed_cik_sec_8k_and_8k_amendment_filings",
    }
    for column, expected in fixed.items():
        if not catalogue[column].eq(expected).all():
            raise ValueError(f"SEC catalogue {column} changed")
    expected_partitions = timestamps.dt.year.map(PARTITIONS)
    if not catalogue["whole_event_partition"].eq(expected_partitions).all():
        raise ValueError("SEC catalogue whole-event partitions changed")


def validate_catalogue_rows(catalogue: DataFrame) -> None:
    for row in catalogue.itertuples(index=False):
        item_codes = json.loads(row.item_codes_json)
        if item_codes != parse_item_codes(row.items):
            raise ValueError("SEC catalogue item codes disagree with the source items")
        if row.earnings_like_candidate != (EARNINGS_ITEM_CODE in item_codes):
            raise ValueError("SEC catalogue earnings candidate rule changed")


def validate_catalogue(catalogue: DataFrame) -> None:
    validate_catalogue_shape(catalogue)
    validate_catalogue_constants(catalogue)
    validate_catalogue_rows(catalogue)


def validate_source_bindings(
    manifests: Sequence[Mapping[str, Any]],
    required_supplements: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    main_by_cik = {
        str(entry["cik"]): entry
        for entry in manifests
        if entry.get("source_role") == "main_submission"
    }
    if set(main_by_cik) != set(COMPANIES):
        raise ValueError("SEC source manifest is missing a fixed-CIK main submission")
    for cik, expected in required_supplements.items():
        actual = {
            str(entry["supplement_name"])
            for entry in manifests
            if entry.get("source_role") == "supplement_submission"
            and entry.get("cik") == cik
        }
        if actual != {str(value["name"]) for value in expected}:
            raise ValueError(f"SEC supplemental bindings are incomplete for CIK {cik}")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby(
            ["company_key", "cik", "form", "whole_event_partition", "earnings_like_candidate"],
            sort=True,
            dropna=False,
        )
        .agg(
            filings=("event_id", "nunique"),
            first_acceptance_utc=("acceptance_datetime_utc", "min"),
            last_acceptance_utc=("acceptance_datetime_utc", "max"),
        )
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    item_codes = sorted(
        {code for value in catalogue["item_codes_json"] for code in json.loads(value)}
    )
    return {
        "schema_version": 1,
        "status": "frozen_sec_hyperscaler_8k_catalogue_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "period_utc": {
            "start_inclusive": START.isoformat(),
            "end_exclusive": END_EXCLUSIVE.isoformat(),
        },
        "fixed_identity_mapping": COMPANIES,
        "forms": sorted(FORMS),
        "frozen_item_codes": item_codes,
        "earnings_like_candidate_rule": "Item 2.02 present in SEC items",
        "timestamp_rule": "Preserve SEC acceptanceDateTime exact UTC timestamp.",
        "timestamp_boundary": (
            "SEC acceptanceDateTime is a filing-acceptance clock and is not claimed "
            "to equal the first public-news time."
        ),
        "selection_rule": (
            "Complete SEC submissions rows for fixed CIKs in the fixed period, limited "
            "to Forms 8-K and 8-K/A and deduplicated deterministically by CIK/accession."
        ),
        "source_rule": (
            "Use official data.sec.gov main submissions JSON plus only date-overlapping "
            "historical files referenced by that company's main submission JSON."
        ),
        "event_rows": len(catalogue),
        "earnings_like_candidates": int(catalogue["earnings_like_candidate"].sum()),
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            "No market, news, crypto, return, direction, causation, or profit data was read.",
            "An Item 2.02 filing is an earnings-like candidate, not a measured surprise.",
            "Filing acceptance time must not be presented as a first-public-news timestamp.",
            "Later testing must split and hold out whole filings/events by the frozen partition.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# SEC Hyperscaler 8-K Source Catalogue",
            "",
            f"- Fixed-company 8-K / 8-K/A rows: `{freeze['event_rows']}`",
            f"- Item 2.02 earnings-like candidates: `{freeze['earnings_like_candidates']}`",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "SEC acceptanceDateTime is retained as an exact UTC filing-acceptance clock. "
            "It is not claimed to be the first public-news time.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_sec_hyperscaler_8k_catalogue":
        raise ValueError("Existing SEC catalogue result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing SEC catalogue artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    all_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    required_supplements: dict[str, list[dict[str, Any]]] = {}
    for cik in COMPANIES:
        main_url = main_submission_url(cik)
        main_document, main_manifest = fetch_json(main_url)
        validate_main_identity(main_document, cik)
        main_manifest.update({"cik": cik, "source_role": "main_submission"})
        manifests.append(main_manifest)
        recent = main_document.get("filings", {}).get("recent")
        if not isinstance(recent, Mapping):
            raise ValueError(f"SEC main submission has no recent filing rows: {cik}")
        all_records.extend(
            submission_rows(recent, cik=cik, source_url=main_url, source_kind="main")
        )
        supplements = supplement_references(main_document, cik)
        required_supplements[cik] = supplements
        for supplement in supplements:
            supplement_document, supplement_manifest = fetch_json(str(supplement["url"]))
            supplement_manifest.update(
                {
                    "cik": cik,
                    "source_role": "supplement_submission",
                    "supplement_name": supplement["name"],
                    "main_binding": {
                        "filing_from": supplement["filing_from"],
                        "filing_to": supplement["filing_to"],
                    },
                }
            )
            manifests.append(supplement_manifest)
            all_records.extend(
                submission_rows(
                    supplement_document,
                    cik=cik,
                    source_url=str(supplement["url"]),
                    source_kind="supplement",
                )
            )
    validate_source_bindings(manifests, required_supplements)
    catalogue = build_catalogue(all_records)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(catalogue, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(
        {"retrieved_at_utc": g0.utc_now(), "sources": manifests}, SOURCE_MANIFEST_PATH
    )
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_sec_hyperscaler_8k_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "event_rows": len(catalogue),
        "earnings_like_candidates": int(catalogue["earnings_like_candidate"].sum()),
        "source_json_documents": len(manifests),
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
                    "fixed_ciks": list(COMPANIES),
                    "period_utc": {
                        "start_inclusive": START.isoformat(),
                        "end_exclusive": END_EXCLUSIVE.isoformat(),
                    },
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
