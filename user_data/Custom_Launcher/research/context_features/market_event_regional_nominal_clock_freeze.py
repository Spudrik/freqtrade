"""Freeze China NBS nominal clocks and recheck Eurostat historical coverage."""

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
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
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
    "regional_nominal_clock_batch_20260912a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "china_nbs_nominal_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "regional_nominal_source_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "regional_nominal_clock_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "regional_nominal_clock_freeze_report.md"
RESULT_PATH = OUTPUT_ROOT / "regional_nominal_clock_freeze_result.json"

NBS_ROOT = "https://www.stats.gov.cn"
NBS_INDEX_URL = f"{NBS_ROOT}/xxgk/sjfb/fbrcb/"
EUROSTAT_ROOT = "https://ec.europa.eu"
EUROSTAT_API = f"{EUROSTAT_ROOT}/eurostat/o/calendars/eventsJson"
START_YEAR = 2021
END_YEAR = 2025
YEARS = tuple(range(START_YEAR, END_YEAR + 1))
WORKERS = 4
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b regional nominal-clock source audit"
SHANGHAI = ZoneInfo("Asia/Shanghai")
PARTITIONS = {
    2021: "development_2021_2023",
    2022: "development_2021_2023",
    2023: "development_2021_2023",
    2024: "internal_validation_2024_2025",
    2025: "internal_validation_2024_2025",
}
FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "china_nbs_price_release": {
        "label_pattern": r"^居民消费价格指数月度报告$",
        "companion_label_patterns": (r"^工业生产者价格指数月度报告$",),
        "plain_name": "China consumer and producer price release",
        "expected_per_year": 12,
        "companion_information": "CPI and PPI are published at the same frozen clock",
    },
    "china_nbs_pmi_release": {
        "label_pattern": r"^(?:中国)?采购经理指数月度报告.*$",
        "companion_label_patterns": (),
        "plain_name": "China official purchasing-manager release",
        "expected_per_year": 12,
        "companion_information": "manufacturing non-manufacturing and composite PMI bundle",
    },
    "china_nbs_national_economy_release": {
        "label_pattern": r"^国民经济运行情况(?:新闻发布会)?$",
        "companion_label_patterns": (),
        "plain_name": "China national-economy release bundle",
        "expected_per_year": 11,
        "companion_information": (
            "GDP industrial retail investment and related releases share this clock"
        ),
    },
}
EUROSTAT_MAJOR_TITLES = {
    "eurostat_flash_inflation": "Flash estimate inflation euro area",
    "eurostat_preliminary_gdp": "Preliminary flash estimate GDP - EU and euro area",
    "eurostat_unemployment": "Unemployment",
}
DATE_TOKEN = re.compile(r"(?P<day>\d{1,2})\s*/\s*(?P<weekday>[一二三四五六日天])")
TIME_TOKEN = re.compile(r"(?P<hour>\d{1,2})\s*[:\uff1a]\s*(?P<minute>\d{2})")
WEEKDAY_NUMBER = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def official_url(url: str, *, host: str, path_prefix: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == host.casefold()
        and parsed.username is None
        and parsed.password is None
        and port is None
        and parsed.path.startswith(path_prefix)
        and parsed.fragment == ""
    )


def fetch_bytes(
    url: str,
    *,
    expected_host: str,
    path_prefix: str,
    requester: Callable[..., Any] = requests.get,
) -> tuple[bytes, dict[str, Any]]:
    if not official_url(url, host=expected_host, path_prefix=path_prefix):
        raise ValueError(f"Unapproved official source URL: {url}")
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"},
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Official source returned {response.status_code}: {url}")
    if str(response.url) != url:
        raise ValueError(f"Official source redirected unexpectedly: {url}")
    return content, {
        "url": url,
        "http_status": int(response.status_code),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def decode_utf8(content: bytes, source: str) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Official source is not valid UTF-8: {source}") from exc


def nbs_year_urls(index_document: str) -> dict[int, str]:
    soup = BeautifulSoup(index_document, "html.parser")
    found: dict[int, str] = {}
    for anchor in soup.find_all("a", href=True):
        label = " ".join(anchor.get_text(" ", strip=True).split())
        match = re.fullmatch(r"(20\d{2})年国家统计局主要统计信息发布日程表", label)
        if match is None:
            continue
        year = int(match.group(1))
        if year not in YEARS:
            continue
        url = urljoin(NBS_INDEX_URL, str(anchor["href"]))
        if not official_url(url, host="www.stats.gov.cn", path_prefix="/xxgk/sjfb/fbrcb/"):
            raise ValueError(f"NBS index linked outside the approved archive: {url}")
        if year in found and found[year] != url:
            raise ValueError(f"NBS index has two different pages for {year}")
        found[year] = url
    if set(found) != set(YEARS):
        raise ValueError(f"NBS annual calendar coverage changed: {sorted(found)}")
    return dict(sorted(found.items()))


def _cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def _month_columns(table: DataFrame) -> dict[int, int]:
    for row_number, row in table.iterrows():
        values = [_cell_text(value) for value in row.tolist()]
        if len(values) < 14 or values[0] != "序号" or values[1] != "内容":
            continue
        columns: dict[int, int] = {}
        for column, value in enumerate(values[2:], start=2):
            match = re.fullmatch(r"(\d{1,2})月", value)
            if match is not None:
                columns[int(match.group(1))] = column
        if set(columns) == set(range(1, 13)):
            return columns
        raise ValueError(f"NBS month header is incomplete on table row {row_number}")
    raise ValueError("NBS annual calendar table header was not found")


def _family_rows(table: DataFrame, pattern: str) -> tuple[pd.Series, pd.Series]:
    labels = table.iloc[:, 1].map(_cell_text)
    selected = table.loc[labels.str.contains(pattern, regex=True, na=False)]
    if len(selected) != 2:
        raise ValueError(f"Expected a date/time row pair for NBS pattern {pattern!r}")
    first, second = selected.iloc[0], selected.iloc[1]
    first_text = " ".join(_cell_text(value) for value in first.iloc[2:])
    second_text = " ".join(_cell_text(value) for value in second.iloc[2:])
    if len(DATE_TOKEN.findall(first_text)) < 10 or len(TIME_TOKEN.findall(second_text)) < 10:
        raise ValueError(f"NBS date/time rows are not in the expected order for {pattern!r}")
    return first, second


def _cell_events(
    date_text: str, time_text: str, *, context: str
) -> list[tuple[int, str, int, int]]:
    date_matches = list(DATE_TOKEN.finditer(date_text))
    time_matches = list(TIME_TOKEN.finditer(time_text))
    if not date_matches and not time_matches:
        return []
    if len(time_matches) == 1 and len(date_matches) > 1:
        time_matches = time_matches * len(date_matches)
    if len(date_matches) != len(time_matches):
        raise ValueError(f"NBS {context} has unequal date/time counts")
    return [
        (
            int(date_match.group("day")),
            date_match.group("weekday"),
            int(time_match.group("hour")),
            int(time_match.group("minute")),
        )
        for date_match, time_match in zip(date_matches, time_matches, strict=True)
    ]


def parse_nbs_calendar(document: str, *, year: int, source_url: str) -> DataFrame:
    tables = pd.read_html(StringIO(document), header=None)
    candidates = [table for table in tables if table.shape[1] >= 14]
    if len(candidates) != 1:
        raise ValueError(f"NBS calendar page {year} has an ambiguous main table")
    table = candidates[0]
    month_columns = _month_columns(table)
    rows: list[dict[str, Any]] = []
    for family, spec in FAMILY_SPECS.items():
        date_row, time_row = _family_rows(table, str(spec["label_pattern"]))
        companion_rows = [
            _family_rows(table, str(pattern))
            for pattern in spec["companion_label_patterns"]
        ]
        for month, column in month_columns.items():
            events = _cell_events(
                _cell_text(date_row.iloc[column]),
                _cell_text(time_row.iloc[column]),
                context=f"{family} {year}-{month:02d}",
            )
            for companion_number, (companion_date, companion_time) in enumerate(
                companion_rows,
                start=1,
            ):
                companion_events = _cell_events(
                    _cell_text(companion_date.iloc[column]),
                    _cell_text(companion_time.iloc[column]),
                    context=f"{family} companion {companion_number} {year}-{month:02d}",
                )
                if companion_events != events:
                    raise ValueError(
                        f"NBS {family} companion clock differs in {year}-{month:02d}"
                    )
            for day, weekday, hour, minute in events:
                local_date = date(year, month, day)
                if local_date.weekday() != WEEKDAY_NUMBER[weekday]:
                    raise ValueError(
                        f"NBS weekday mismatch for {family}: {local_date} {weekday}"
                    )
                local = pd.Timestamp(
                    year=year,
                    month=month,
                    day=day,
                    hour=hour,
                    minute=minute,
                    tz=SHANGHAI,
                )
                anchor = local.tz_convert("UTC")
                rows.append(
                    {
                        "event_id": (
                            f"{family}_{local.strftime('%Y%m%d_%H%M')}_beijing"
                        ),
                        "event_family": family,
                        "event_plain_name": spec["plain_name"],
                        "companion_information": spec["companion_information"],
                        "companion_clock_verified": bool(companion_rows),
                        "release_date_local": local.date().isoformat(),
                        "nominal_release_clock_local": local.strftime("%H:%M"),
                        "nominal_release_timestamp_local": local.isoformat(),
                        "anchor_utc": anchor.isoformat(),
                        "source_url": source_url,
                        "source_year": year,
                        "timestamp_quality": (
                            "official_annual_calendar_nominal_minute_not_actual_page_complete"
                        ),
                        "strict_intraday_attribution_eligible": False,
                        "nominal_clock_sensitivity_eligible": True,
                        "market_expectation_status": "unavailable_not_imputed",
                        "whole_event_partition": PARTITIONS[year],
                        "selection_method": (
                            "complete_three_fixed_nbs_annual_calendar_rows_2021_2025"
                        ),
                    }
                )
    catalogue = DataFrame.from_records(rows).sort_values(
        ["anchor_utc", "event_family"], kind="stable"
    ).reset_index(drop=True)
    expected = sum(int(spec["expected_per_year"]) for spec in FAMILY_SPECS.values())
    if len(catalogue) != expected:
        raise ValueError(f"NBS {year} expected {expected} rows; got {len(catalogue)}")
    for family, spec in FAMILY_SPECS.items():
        count = int(catalogue["event_family"].eq(family).sum())
        if count != int(spec["expected_per_year"]):
            raise ValueError(
                f"NBS {year} {family} expected {spec['expected_per_year']}; got {count}"
            )
    return catalogue


def validate_catalogue(catalogue: DataFrame) -> None:
    required = {
        "event_id",
        "event_family",
        "anchor_utc",
        "source_url",
        "timestamp_quality",
        "strict_intraday_attribution_eligible",
        "nominal_clock_sensitivity_eligible",
        "market_expectation_status",
        "whole_event_partition",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"Regional catalogue is missing: {required - set(catalogue)}")
    if len(catalogue) != 175:
        raise ValueError(f"Expected 175 NBS events; got {len(catalogue)}")
    if catalogue["event_id"].duplicated().any():
        raise ValueError("NBS catalogue identifiers are not unique")
    expected_by_family = {
        family: int(spec["expected_per_year"]) * len(YEARS)
        for family, spec in FAMILY_SPECS.items()
    }
    if catalogue["event_family"].value_counts().to_dict() != expected_by_family:
        raise ValueError("NBS family row counts changed")
    if catalogue["strict_intraday_attribution_eligible"].astype(bool).any():
        raise ValueError("Nominal NBS clocks were incorrectly marked exact")
    if not catalogue["nominal_clock_sensitivity_eligible"].astype(bool).all():
        raise ValueError("An NBS row is not eligible for the frozen sensitivity test")
    if not catalogue["market_expectation_status"].eq("unavailable_not_imputed").all():
        raise ValueError("NBS market expectations were unexpectedly imputed")
    price_rows = catalogue["event_family"].eq("china_nbs_price_release")
    if not catalogue.loc[price_rows, "companion_clock_verified"].astype(bool).all():
        raise ValueError("NBS CPI/PPI shared clocks were not verified")
    if catalogue.loc[~price_rows, "companion_clock_verified"].astype(bool).any():
        raise ValueError("An unrelated NBS family was marked as a shared-clock release")
    anchors = pd.to_datetime(catalogue["anchor_utc"], utc=True, errors="raise")
    if anchors.isna().any():
        raise ValueError("NBS catalogue has an invalid UTC anchor")
    forbidden = ("profit", "return", "direction", "ohlcv", "market_volume")
    if any(token in column.casefold() for column in catalogue for token in forbidden):
        raise ValueError("NBS source catalogue contains a market-outcome column")


def eurostat_url(year: int) -> str:
    query = urlencode(
        {
            "start": f"{year}-01-01T00:00:00Z",
            "end": f"{year + 1}-01-01T00:00:00Z",
            "isEuroindicator": "true",
        }
    )
    return f"{EUROSTAT_API}?{query}"


def parse_eurostat_rows(
    content: bytes, *, year: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        rows = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Eurostat {year} response is not valid UTF-8 JSON") from exc
    if not isinstance(rows, list):
        raise ValueError(f"Eurostat {year} response is not a list")
    counts = {
        family: sum(
            str(row.get("title", "")) == title for row in rows if isinstance(row, dict)
        )
        for family, title in EUROSTAT_MAJOR_TITLES.items()
    }
    return rows, counts


def source_counts(catalogue: DataFrame, eurostat: Mapping[int, Mapping[str, Any]]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (year, family, partition), group in catalogue.groupby(
        ["source_year", "event_family", "whole_event_partition"], sort=True
    ):
        rows.append(
            {
                "region": "china",
                "source_year": int(year),
                "source_family": family,
                "whole_event_partition": partition,
                "events": int(group["event_id"].nunique()),
                "market_test_status": "eligible_nominal_clock_sensitivity",
            }
        )
    for year, record in sorted(eurostat.items()):
        partition = PARTITIONS[year]
        family_counts = {
            "eurostat_all_euroindicator_rows": record["all_euroindicator_rows"],
            **record["major_family_counts"],
        }
        for family, count in family_counts.items():
            rows.append(
                {
                    "region": "euro_area",
                    "source_year": int(year),
                    "source_family": family,
                    "whole_event_partition": partition,
                    "events": int(count),
                    "market_test_status": (
                        "coverage_parked_official_calendar_unavailable_for_year"
                        if int(record["all_euroindicator_rows"]) == 0
                        else "coverage_parked_missing_development_history"
                    ),
                }
            )
    return DataFrame.from_records(rows)


def freeze_document(
    catalogue: DataFrame,
    counts: DataFrame,
    eurostat: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_regional_nominal_clock_batch_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "plain_question": (
            "Do complete official China NBS release-calendar clocks support a bounded "
            "BTC/ETH activity sensitivity test, and does Eurostat expose enough historical "
            "calendar rows for the same two-period design?"
        ),
        "china": {
            "release_rows": len(catalogue),
            "families": list(FAMILY_SPECS),
            "timestamp_offsets_minutes": [-30, 0, 30],
            "activity_horizons_minutes": [60, 240],
            "assets": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "minimum_median_activity_score": 1.20,
            "minimum_above_control_rate": 0.55,
            "minimum_events_by_partition": {
                "development_2021_2023": 12,
                "internal_validation_2024_2025": 8,
            },
            "controls_per_event": 12,
            "control_search_weeks": 60,
            "collision_hours": 4,
            "whole_batch_permutations": 2000,
            "familywide_probability_ceiling": 0.05,
            "retention_rule": (
                "nominal offset must pass both periods; at least one adjacent offset must "
                "reach 1.10 median activity and 0.50 above-control rate in both periods; "
                "the linked whole-event familywide chance probability must be <=0.05"
            ),
        },
        "eurostat": {
            "calendar_counts": {str(year): dict(record) for year, record in eurostat.items()},
            "market_outcome_decision": "coverage_parked_no_two_period_official_calendar_history",
        },
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            "NBS annual calendars explicitly say their planned dates can be adjusted.",
            "The NBS clocks are nominal association clocks, not proof of exact first publication.",
            "CPI and PPI share one price-release event rather than two independent events.",
            "The national-economy row represents one simultaneous information bundle.",
            (
                "Eurostat market outcomes remain closed because its current official API "
                "has no 2021-2024 rows."
            ),
            "No historical consensus exists here; direction is unavailable and not imputed.",
            "No crypto outcome, direction, causation, trading rule, or profit was read.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    eurostat = freeze["eurostat"]["calendar_counts"]
    return "\n".join(
        [
            "# Regional Nominal-Clock Source Freeze",
            "",
            f"- China NBS frozen release clocks: `{freeze['china']['release_rows']}`",
            f"- China families: `{freeze['china']['families']}`",
            f"- Eurostat official calendar rows by year: `{eurostat}`",
            "- China timestamp status: **Nominal sensitivity only**",
            "- Eurostat outcome status: **Coverage-parked**",
            "- Direction tested: **No**",
            "- Profit used: **No**",
            "",
            "The China catalogue can proceed to a bounded activity test with shifted-clock "
            "views. Eurostat cannot support the same two-period historical design from its "
            "currently exposed official calendar, so no Eurostat market outcome will be opened.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_regional_nominal_clock_source_freeze":
        raise ValueError("Existing regional nominal-clock result is invalid")
    required = {
        "catalogue",
        "counts",
        "source_manifest",
        "freeze",
        "report",
        "analysis_script",
    }
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("Existing regional source result has an incomplete artifact set")
    for value in artifacts.values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing regional source artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    expected_outputs = (
        CATALOGUE_PATH,
        COUNTS_PATH,
        SOURCE_MANIFEST_PATH,
        FREEZE_PATH,
        REPORT_PATH,
    )
    if not overwrite:
        partial_outputs = [str(path) for path in expected_outputs if path.exists()]
        if partial_outputs:
            raise FileExistsError(
                "Partial regional source-freeze outputs exist without a result record: "
                + ", ".join(partial_outputs)
            )

    index_content, index_manifest = fetch_bytes(
        NBS_INDEX_URL,
        expected_host="www.stats.gov.cn",
        path_prefix="/xxgk/sjfb/fbrcb/",
    )
    index_document = decode_utf8(index_content, NBS_INDEX_URL)
    annual_urls = nbs_year_urls(index_document)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        fetched = list(
            executor.map(
                lambda item: (
                    item[0],
                    *fetch_bytes(
                        item[1],
                        expected_host="www.stats.gov.cn",
                        path_prefix="/xxgk/sjfb/fbrcb/",
                    ),
                ),
                annual_urls.items(),
            )
        )
    annual_frames: list[DataFrame] = []
    source_documents: list[dict[str, Any]] = [index_manifest]
    for year, content, manifest in fetched:
        document = decode_utf8(content, manifest["url"])
        annual_frames.append(
            parse_nbs_calendar(document, year=int(year), source_url=manifest["url"])
        )
        source_documents.append(manifest)
    catalogue = pd.concat(annual_frames, ignore_index=True).sort_values(
        ["anchor_utc", "event_family"], kind="stable"
    ).reset_index(drop=True)
    validate_catalogue(catalogue)

    eurostat: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        url = eurostat_url(year)
        content, manifest = fetch_bytes(
            url,
            expected_host="ec.europa.eu",
            path_prefix="/eurostat/o/calendars/eventsJson",
        )
        rows, major_counts = parse_eurostat_rows(content, year=year)
        source_documents.append(manifest)
        eurostat[year] = {
            "all_euroindicator_rows": len(rows),
            "major_family_counts": major_counts,
        }
    if any(eurostat[year]["all_euroindicator_rows"] for year in range(2021, 2025)):
        raise ValueError("Eurostat unexpectedly exposed old rows; review before opening outcomes")
    if eurostat[2025]["all_euroindicator_rows"] <= 0:
        raise ValueError("Eurostat current historical sentinel returned no 2025 rows")

    counts = source_counts(catalogue, eurostat)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    source_manifest = {
        "schema_version": 1,
        "status": "frozen_official_regional_source_responses",
        "retrieved_at_utc": g0.utc_now(),
        "documents": source_documents,
    }
    g0.atomic_write_json(source_manifest, SOURCE_MANIFEST_PATH)
    freeze = freeze_document(catalogue, counts, eurostat)
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    temporary_report = REPORT_PATH.with_suffix(f".md.{os.getpid()}.tmp")
    temporary_report.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    temporary_report.replace(REPORT_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_regional_nominal_clock_source_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "china_release_rows": len(catalogue),
        "eurostat_outcome_status": freeze["eurostat"]["market_outcome_decision"],
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
                    "china_expected_release_rows": 175,
                    "eurostat_market_outcomes_will_remain_closed": True,
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
