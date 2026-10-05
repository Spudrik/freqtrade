"""Build an outcome-blind, date-only FDIC failed-bank catalogue."""

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
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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
    "fdic_failed_bank_catalogue_20260906a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "fdic_failed_bank_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "fdic_failed_bank_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "fdic_failed_bank_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "fdic_failed_bank_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "fdic_failed_bank_catalogue_result.json"

API_URL = "https://api.fdic.gov/banks/failures"
API_DOCS_URL = "https://api.fdic.gov/banks/docs"
FAILED_BANK_LIST_URL = (
    "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/"
)
START_DATE = pd.Timestamp("2020-01-01")
END_DATE_INCLUSIVE = pd.Timestamp("2025-12-31")
EXPECTED_FAILURE_ROWS = 13
EXPECTED_YEAR_COUNTS = {2020: 4, 2021: 0, 2022: 0, 2023: 5, 2024: 2, 2025: 2}
EPISODE_GAP_DAYS = 3
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b FDIC source audit"
FILTER = "FAILDATE:[2020-01-01 TO 2025-12-31]"
FIELDS = (
    "NAME,CERT,FIN,CITYST,FAILDATE,FAILYR,SAVR,RESTYPE,RESTYPE1,"
    "QBFDEP,QBFASSET,COST,CHCLASS1"
)
QUERY_PARAMS: dict[str, str | int] = {
    "filters": FILTER,
    "fields": FIELDS,
    "sort_by": "FAILDATE",
    "sort_order": "ASC",
    "limit": 100,
    "offset": 0,
}
REQUIRED_SOURCE_FIELDS = {
    "ID",
    "NAME",
    "CERT",
    "FIN",
    "CITYST",
    "FAILDATE",
    "FAILYR",
    "SAVR",
    "RESTYPE",
    "RESTYPE1",
    "QBFDEP",
    "QBFASSET",
    "COST",
    "CHCLASS1",
}
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


def official_api_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    allowed_query = {
        "filters",
        "fields",
        "sort_by",
        "sort_order",
        "limit",
        "offset",
    }
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "api.fdic.gov"
        and parsed.path == "/banks/failures"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.fragment
        and set(parse_qs(parsed.query)).issubset(allowed_query)
    )


def fetch_source(
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not official_api_url(API_URL):
        raise ValueError("Configured FDIC API URL is not official")
    response = requester(
        API_URL,
        params=QUERY_PARAMS,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Official FDIC source returned {response.status_code}")
    if not official_api_url(str(response.url)):
        raise ValueError("FDIC response did not come from the approved official URL")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Official FDIC response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Official FDIC response root is not an object")
    return payload, {
        "url": str(response.url),
        "http_status": int(response.status_code),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def flatten_source(payload: Mapping[str, Any]) -> DataFrame:
    meta = payload.get("meta")
    rows = payload.get("data")
    if not isinstance(meta, Mapping) or not isinstance(rows, list):
        raise ValueError("FDIC response lacks meta or data records")
    total = meta.get("total")
    if not isinstance(total, int) or total != EXPECTED_FAILURE_ROWS:
        raise ValueError(
            f"FDIC frozen-period count changed: {total}; expected {EXPECTED_FAILURE_ROWS}"
        )
    parameters = meta.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("FDIC response lacks echoed query parameters")
    if str(parameters.get("filters")) != FILTER:
        raise ValueError("FDIC response filter differs from the frozen date filter")
    if len(rows) != total:
        raise ValueError("FDIC response is incomplete")
    records: list[dict[str, Any]] = []
    for wrapper in rows:
        if not isinstance(wrapper, Mapping) or not isinstance(wrapper.get("data"), Mapping):
            raise ValueError("FDIC failure record wrapper is invalid")
        record = dict(wrapper["data"])
        missing = REQUIRED_SOURCE_FIELDS - set(record)
        if missing:
            raise ValueError(f"FDIC failure record is missing fields: {sorted(missing)}")
        records.append(record)
    frame = DataFrame.from_records(records)
    if frame.empty or frame["ID"].astype("string").duplicated().any():
        raise ValueError("FDIC failure IDs are empty or duplicated")
    return frame


def parse_failure_dates(values: pd.Series) -> pd.Series:
    parsed = []
    for value in values.astype("string"):
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", str(value)) is None:
            raise ValueError(f"FDIC failure date is invalid: {value}")
        parsed.append(pd.Timestamp(datetime.strptime(str(value), "%m/%d/%Y")))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns]")


def add_episode_groups(catalogue: DataFrame) -> DataFrame:
    output = catalogue.sort_values(["failure_date", "event_id"], kind="stable").copy()
    dates = pd.to_datetime(output["failure_date"], errors="raise")
    new_episode = dates.diff().dt.days.gt(EPISODE_GAP_DAYS).fillna(True)
    episode_number = new_episode.cumsum()
    output["episode_id"] = ""
    output["episode_members"] = 0
    for _, indices in output.groupby(episode_number, sort=True).groups.items():
        positions = list(indices)
        member_ids = sorted(output.loc[positions, "event_id"].astype(str))
        start = dates.loc[positions].min().strftime("%Y%m%d")
        digest = hashlib.sha256("|".join(member_ids).encode()).hexdigest()[:10]
        output.loc[positions, "episode_id"] = f"fdic_episode_{start}_{digest}"
        output.loc[positions, "episode_members"] = len(positions)
    return output.reset_index(drop=True)


def build_catalogue(source: DataFrame) -> DataFrame:
    dates = parse_failure_dates(source["FAILDATE"])
    years = source["FAILYR"].astype("string")
    if not years.eq(dates.dt.year.astype("string")).all():
        raise ValueError("FDIC failure year disagrees with failure date")
    if not dates.between(START_DATE, END_DATE_INCLUSIVE, inclusive="both").all():
        raise ValueError("FDIC failure date is outside the frozen period")
    catalogue = DataFrame(
        {
            "event_id": "fdic_failure_" + source["ID"].astype("string"),
            "fdic_failure_id": source["ID"].astype("string"),
            "institution_name": source["NAME"].astype("string"),
            "certificate_number": source["CERT"].astype("Int64"),
            "financial_institution_number": source["FIN"].astype("string"),
            "location": source["CITYST"].astype("string"),
            "failure_date": dates.dt.strftime("%Y-%m-%d"),
            "official_confirmed_date": dates.dt.strftime("%Y-%m-%d"),
            "official_release_time_utc": pd.Series(pd.NA, index=source.index, dtype="string"),
            "time_precision": "official_failure_date_only",
            "intraday_eligible": False,
            "insurance_fund_code": source["SAVR"].astype("string"),
            "resolution_type": source["RESTYPE"].astype("string"),
            "transaction_type_code": source["RESTYPE1"].astype("string"),
            "charter_class_code": source["CHCLASS1"].astype("string"),
            "deposits_thousand_usd": pd.to_numeric(source["QBFDEP"], errors="raise"),
            "assets_thousand_usd": pd.to_numeric(source["QBFASSET"], errors="raise"),
            "estimated_loss_thousand_usd": pd.to_numeric(source["COST"], errors="raise"),
            "source_url": API_URL,
            "failed_bank_list_url": FAILED_BANK_LIST_URL,
            "press_release_url": pd.Series(pd.NA, index=source.index, dtype="string"),
            "press_release_url_status": "not_supplied_by_failure_api",
            "market_expectation_status": "unavailable_not_imputed",
            "whole_event_partition": dates.dt.year.map(PARTITIONS),
            "selection_method": "complete_official_fdic_failures_fixed_2020_2025",
        }
    )
    catalogue = add_episode_groups(catalogue)
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: DataFrame) -> None:
    required = {
        "event_id",
        "fdic_failure_id",
        "institution_name",
        "failure_date",
        "official_release_time_utc",
        "time_precision",
        "intraday_eligible",
        "episode_id",
        "episode_members",
        "source_url",
        "market_expectation_status",
        "whole_event_partition",
        "selection_method",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"FDIC catalogue is missing: {required - set(catalogue)}")
    if len(catalogue) != EXPECTED_FAILURE_ROWS:
        raise ValueError("FDIC catalogue row count changed")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("FDIC catalogue event IDs are duplicated")
    dates = pd.to_datetime(catalogue["failure_date"], errors="raise")
    actual_year_counts = dates.dt.year.value_counts().to_dict()
    expected_nonzero = {
        year: count for year, count in EXPECTED_YEAR_COUNTS.items() if count > 0
    }
    if actual_year_counts != expected_nonzero:
        raise ValueError("FDIC catalogue year counts changed")
    if catalogue["official_release_time_utc"].notna().any():
        raise ValueError("FDIC date-only source unexpectedly contains an intraday clock")
    fixed = {
        "time_precision": "official_failure_date_only",
        "intraday_eligible": False,
        "source_url": API_URL,
        "press_release_url_status": "not_supplied_by_failure_api",
        "market_expectation_status": "unavailable_not_imputed",
        "selection_method": "complete_official_fdic_failures_fixed_2020_2025",
    }
    for column, expected in fixed.items():
        if not catalogue[column].eq(expected).all():
            raise ValueError(f"FDIC catalogue {column} changed")
    forbidden_tokens = ("profit", "future_return", "price", "volume", "direction")
    if any(
        token in column.casefold()
        for column in catalogue.columns
        for token in forbidden_tokens
    ):
        raise ValueError("FDIC source catalogue contains a market outcome column")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby("whole_event_partition", sort=True)
        .agg(
            failures=("event_id", "nunique"),
            independent_date_episodes=("episode_id", "nunique"),
            first_failure_date=("failure_date", "min"),
            last_failure_date=("failure_date", "max"),
        )
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_fdic_failed_bank_catalogue_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "plain_question": (
            "Can the official FDIC failure history supply a complete date-level banking-"
            "stress context catalogue for 2020-2025?"
        ),
        "period": {
            "start_inclusive": START_DATE.date().isoformat(),
            "end_inclusive": END_DATE_INCLUSIVE.date().isoformat(),
        },
        "source_api_url": API_URL,
        "source_docs_url": API_DOCS_URL,
        "source_filter": FILTER,
        "expected_failure_rows": EXPECTED_FAILURE_ROWS,
        "expected_year_counts": EXPECTED_YEAR_COUNTS,
        "episode_rule": (
            "After sorting by official failure date, consecutive failures no more than "
            f"{EPISODE_GAP_DAYS} calendar days apart share one date-level episode."
        ),
        "failure_rows": len(catalogue),
        "independent_date_episodes": int(catalogue["episode_id"].nunique()),
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            "The FDIC API supplies an effective failure date but not a safe public-release clock.",
            "No midnight, market-open, market-close, or press-release time is invented.",
            "This catalogue is ineligible for minute-level lead/lag or first-public attribution.",
            "Later use is limited to conservative daily or multi-day event context.",
            "No crypto, price, volume, direction, causation, or profit outcome was read.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# FDIC Failed-Bank Source Catalogue",
            "",
            f"- Official failure rows: `{freeze['failure_rows']}`",
            f"- Date-level episodes: `{freeze['independent_date_episodes']}`",
            "- Intraday timestamps available: **No**",
            "- Market outcomes read: **No**",
            "- Profit used: **No**",
            "",
            "This is a complete date-level source catalogue for the frozen period. It can "
            "support conservative daily or multi-day context checks, not minute-level "
            "reaction timing, first-public attribution, direction, or a trading rule.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_fdic_failed_bank_catalogue":
        raise ValueError("Existing FDIC catalogue result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing FDIC catalogue artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    payload, manifest = fetch_source()
    source = flatten_source(payload)
    catalogue = build_catalogue(source)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(catalogue, counts)
    meta = payload["meta"]
    manifest.update(
        {
            "retrieved_at_utc": g0.utc_now(),
            "frozen_query": QUERY_PARAMS,
            "reported_total": meta["total"],
            "source_index": meta.get("index"),
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(manifest, SOURCE_MANIFEST_PATH)
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_fdic_failed_bank_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "failure_rows": len(catalogue),
        "independent_date_episodes": int(catalogue["episode_id"].nunique()),
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
                    "time_precision": "date_only",
                    "expected_rows": EXPECTED_FAILURE_ROWS,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
