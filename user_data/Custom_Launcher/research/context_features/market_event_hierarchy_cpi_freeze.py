"""Freeze a complete, timestamp-safe CPI release family before market outcomes.

FRED/ALFRED supplies the initial publication date for each CPI observation and
the values that were known on each publication date.  This module converts that
history into a small set of predeclared inflation-temperature signs.  Those
signs describe hotter/cooler published inflation than the previous release;
they are deliberately not called market surprises because historical consensus
expectations are not available from this source.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
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
    / "cpi_release_family_20260904a"
)
SOURCE_ROOT = OUTPUT_ROOT / "source"
DEFAULT_KEY_FILE = Path(r"C:\Users\engin\OneDrive\Desktop\FREDAPI.json")

FRED_API_BASE = "https://api.stlouisfed.org/fred"
FRED_OBSERVATIONS_DOC = (
    "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"
)
BLS_CPI_ARCHIVE = "https://www.bls.gov/bls/news-release/cpi.htm"
BLS_CPI_SCHEDULE = "https://www.bls.gov/schedule/news_release/cpi.htm"
INFORMATION_CUTOFF_UTC = layer1.INFORMATION_CUTOFF_UTC
OBSERVATION_START = "2020-01-01"
FAMILY_START = "2021-01-01"
FRED_REALTIME_START = "1776-07-04"
FRED_REALTIME_END = "9999-12-31"
RELEASE_TIME_LOCAL = time(8, 30)
EASTERN = ZoneInfo("America/New_York")

SERIES = {
    "headline_mom": {
        "id": "CPIAUCSL",
        "seasonality": "seasonally_adjusted",
        "comparison_months": 1,
    },
    "core_mom": {
        "id": "CPILFESL",
        "seasonality": "seasonally_adjusted",
        "comparison_months": 1,
    },
    "headline_yoy": {
        "id": "CPIAUCNS",
        "seasonality": "not_seasonally_adjusted",
        "comparison_months": 12,
    },
    "core_yoy": {
        "id": "CPILFENS",
        "seasonality": "not_seasonally_adjusted",
        "comparison_months": 12,
    },
}
SIGN_COLUMNS = tuple(f"temperature_{name}" for name in SERIES)
COMPOSITE_SIGN_COLUMNS = (
    "temperature_headline_agreement",
    "temperature_core_agreement",
    "temperature_broad_majority",
)
ALL_SIGN_COLUMNS = SIGN_COLUMNS + COMPOSITE_SIGN_COLUMNS


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_api_key(path: Path) -> str:
    """Load the explicitly configured local key without ever serializing it."""

    if not path.is_file():
        raise FileNotFoundError(f"FRED API key file not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    nested = document.get("fred")
    key = nested.get("api_key") if isinstance(nested, Mapping) else None
    key = key or document.get("api_key") or document.get("fred_api_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("FRED API key file has no supported api_key field.")
    return key.strip()


def _safe_request_contract(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in params.items() if key != "api_key"}


def fetch_json(
    endpoint: str,
    *,
    params: Mapping[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_params = {**params, "api_key": api_key, "file_type": "json"}
    response = requests.get(
        f"{FRED_API_BASE}/{endpoint}", params=request_params, timeout=60
    )
    response.raise_for_status()
    document = response.json()
    if document.get("error_code"):
        raise ValueError(str(document.get("error_message") or document["error_code"]))
    contract = {
        "endpoint": endpoint,
        "params_without_api_key": _safe_request_contract(request_params),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "fetched_at_utc": g0.utc_now(),
    }
    return document, contract


def _initial_params(series_id: str) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "output_type": 4,
        "realtime_start": FRED_REALTIME_START,
        "realtime_end": FRED_REALTIME_END,
        "observation_start": OBSERVATION_START,
    }


def _vintage_params(
    series_id: str, vintage_dates: Sequence[str], observation_end: str
) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "output_type": 2,
        "vintage_dates": ",".join(vintage_dates),
        "observation_start": OBSERVATION_START,
        "observation_end": observation_end,
    }


def fetch_family_sources(
    api_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Fetch four initial-release histories and four release-date vintages."""

    initial: dict[str, dict[str, Any]] = {}
    initial_contracts: dict[str, dict[str, Any]] = {}
    for definition in SERIES.values():
        series_id = str(definition["id"])
        initial[series_id], initial_contracts[series_id] = fetch_json(
            "series/observations",
            params=_initial_params(series_id),
            api_key=api_key,
        )

    headline_rows = initial[str(SERIES["headline_mom"]["id"])]["observations"]
    usable = [
        row
        for row in headline_rows
        if row["date"] >= FAMILY_START
        and row.get("value") not in (None, ".")
        and pd.Timestamp(row["realtime_start"], tz="UTC")
        < pd.Timestamp(INFORMATION_CUTOFF_UTC)
    ]
    if not usable:
        raise ValueError("FRED returned no usable historical CPI initial releases.")
    vintage_dates = sorted({str(row["realtime_start"]) for row in usable})
    observation_end = max(str(row["date"]) for row in usable)

    vintages: dict[str, dict[str, Any]] = {}
    vintage_contracts: dict[str, dict[str, Any]] = {}
    for definition in SERIES.values():
        series_id = str(definition["id"])
        vintages[series_id], vintage_contracts[series_id] = fetch_json(
            "series/observations",
            params=_vintage_params(series_id, vintage_dates, observation_end),
            api_key=api_key,
        )
    contracts = {
        "initial_release_requests": initial_contracts,
        "release_vintage_requests": vintage_contracts,
        "vintage_date_count": len(vintage_dates),
        "observation_end": observation_end,
    }
    return initial, vintages, contracts


def _rows_by_observation(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["date"]): dict(row) for row in document.get("observations", [])}


def _vintage_value(
    document: Mapping[str, Any],
    *,
    series_id: str,
    observation_date: str,
    vintage_date: str,
) -> float:
    rows = _rows_by_observation(document)
    row = rows.get(observation_date)
    column = f"{series_id}_{vintage_date.replace('-', '')}"
    if row is None or row.get(column) in (None, "."):
        raise ValueError(
            f"Missing {series_id} value for observation {observation_date} "
            f"in vintage {vintage_date}."
        )
    return float(row[column])


def _shift_month(observation_date: str, months: int) -> str:
    period = pd.Period(observation_date, freq="M") - months
    return period.start_time.date().isoformat()


def _release_anchor(release_date: str) -> str:
    local = datetime.combine(date.fromisoformat(release_date), RELEASE_TIME_LOCAL, EASTERN)
    return local.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


def _temperature_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _agreement(first: int, second: int) -> int:
    return first if first != 0 and first == second else 0


def _majority(values: Sequence[int]) -> int:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    if positives >= 3:
        return 1
    if negatives >= 3:
        return -1
    return 0


def build_release_catalog(
    initial: Mapping[str, Mapping[str, Any]],
    vintages: Mapping[str, Mapping[str, Any]],
) -> tuple[DataFrame, DataFrame]:
    """Build the complete family using release-date values only."""

    initial_by_series = {
        str(definition["id"]): _rows_by_observation(initial[str(definition["id"])])
        for definition in SERIES.values()
    }
    headline_id = str(SERIES["headline_mom"]["id"])
    missing_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(INFORMATION_CUTOFF_UTC)

    for observation_date, headline_row in sorted(initial_by_series[headline_id].items()):
        if observation_date < FAMILY_START:
            continue
        release_date = str(headline_row["realtime_start"])
        if pd.Timestamp(release_date, tz="UTC") >= cutoff:
            continue
        if headline_row.get("value") in (None, "."):
            missing_records.append(
                {
                    "observation_month": observation_date,
                    "release_date": release_date,
                    "status": "official_initial_observation_missing",
                    "handling": "excluded_not_imputed",
                }
            )
            continue

        for definition in SERIES.values():
            series_id = str(definition["id"])
            row = initial_by_series[series_id].get(observation_date)
            if row is None or row.get("value") in (None, "."):
                raise ValueError(
                    f"CPI family is incomplete for {observation_date}: {series_id} missing."
                )
            if str(row["realtime_start"]) != release_date:
                raise ValueError(
                    f"CPI family release-date mismatch for {observation_date}: {series_id}."
                )

        record: dict[str, Any] = {
            "event_id": f"cpi_{observation_date[:7].replace('-', '_')}",
            "event_family": "complete_monthly_us_cpi_initial_releases",
            "observation_month": observation_date,
            "release_date": release_date,
            "anchor_utc": _release_anchor(release_date),
            "whole_event_partition": layer1.whole_event_partition(
                pd.Timestamp(_release_anchor(release_date))
            ),
            "market_consensus_status": "unavailable_not_imputed",
        }
        metric_error: ValueError | None = None
        for name, definition in SERIES.items():
            series_id = str(definition["id"])
            months = int(definition["comparison_months"])
            try:
                current = _vintage_value(
                    vintages[series_id],
                    series_id=series_id,
                    observation_date=observation_date,
                    vintage_date=release_date,
                )
                comparison_date = _shift_month(observation_date, months)
                previous = _vintage_value(
                    vintages[series_id],
                    series_id=series_id,
                    observation_date=comparison_date,
                    vintage_date=release_date,
                )
            except ValueError as exc:
                metric_error = exc
                break
            record[f"{name}_released_pct"] = round((current / previous - 1.0) * 100.0, 1)
        if metric_error is not None:
            missing_records.append(
                {
                    "observation_month": observation_date,
                    "release_date": release_date,
                    "status": "required_release_vintage_comparison_missing",
                    "handling": "excluded_not_imputed",
                }
            )
            continue
        records.append(record)

    catalog = DataFrame.from_records(records).sort_values("anchor_utc", kind="stable")
    for name in SERIES:
        released = f"{name}_released_pct"
        delta = f"{name}_change_from_previous_release_pp"
        temperature = f"temperature_{name}"
        catalog[delta] = catalog[released].sub(catalog[released].shift(1)).round(1)
        catalog[temperature] = catalog[delta].map(
            lambda value: np.nan if pd.isna(value) else _temperature_sign(float(value))
        )
    catalog["temperature_headline_agreement"] = [
        np.nan
        if pd.isna(mom) or pd.isna(yoy)
        else _agreement(int(mom), int(yoy))
        for mom, yoy in zip(
            catalog["temperature_headline_mom"],
            catalog["temperature_headline_yoy"],
            strict=True,
        )
    ]
    catalog["temperature_core_agreement"] = [
        np.nan
        if pd.isna(mom) or pd.isna(yoy)
        else _agreement(int(mom), int(yoy))
        for mom, yoy in zip(
            catalog["temperature_core_mom"],
            catalog["temperature_core_yoy"],
            strict=True,
        )
    ]
    catalog["temperature_broad_majority"] = [
        np.nan
        if any(pd.isna(value) for value in values)
        else _majority([int(value) for value in values])
        for values in catalog[list(SIGN_COLUMNS)].itertuples(index=False, name=None)
    ]
    catalog["published_information_direction"] = np.select(
        [
            catalog["temperature_broad_majority"].eq(1),
            catalog["temperature_broad_majority"].eq(-1),
        ],
        ["hotter_than_previous_release", "cooler_than_previous_release"],
        default="mixed_or_unchanged",
    )
    coverage = DataFrame.from_records(
        missing_records,
        columns=["observation_month", "release_date", "status", "handling"],
    )
    return catalog.reset_index(drop=True), coverage


def sign_definitions() -> list[dict[str, Any]]:
    definitions = []
    for name in SERIES:
        definitions.append(
            {
                "id": f"temperature_{name}",
                "rule": (
                    f"Sign of the one-decimal released {name.replace('_', ' ')} rate "
                    "minus that rate in the previous monthly release."
                ),
                "hotter": 1,
                "cooler": -1,
                "unchanged_or_abstain": 0,
            }
        )
    definitions.extend(
        [
            {
                "id": "temperature_headline_agreement",
                "rule": "Use a sign only when headline month-on-month and year-on-year agree.",
            },
            {
                "id": "temperature_core_agreement",
                "rule": "Use a sign only when core month-on-month and year-on-year agree.",
            },
            {
                "id": "temperature_broad_majority",
                "rule": "Use a sign only when at least three of the four measures agree.",
            },
        ]
    )
    return definitions


def build_freeze_document(
    catalog: DataFrame,
    coverage: DataFrame,
    source_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    partition_counts = {
        str(key): int(value)
        for key, value in catalog["whole_event_partition"].value_counts().items()
    }
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "status": "frozen_cpi_family_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "information_cutoff_utc": INFORMATION_CUTOFF_UTC,
        "outcomes_read": False,
        "profit_used": False,
        "family": {
            "name": "complete monthly U.S. CPI initial releases",
            "selection": "every usable initial monthly release from 2021 to cutoff",
            "release_time": "08:30 America/New_York converted with historical DST",
            "event_count": len(catalog),
            "partition_counts": partition_counts,
            "missing_release_rows": len(coverage),
            "missing_policy": "exclude and never impute",
        },
        "published_measure_contract": {
            "rates": {
                name: {
                    "series_id": definition["id"],
                    "seasonality": definition["seasonality"],
                    "comparison_months": definition["comparison_months"],
                    "rounding_decimals": 1,
                }
                for name, definition in SERIES.items()
            },
            "release_vintage_rule": (
                "For each release, calculate rates only from the ALFRED values visible on "
                "that release date; use no later revisions."
            ),
            "signs": sign_definitions(),
        },
        "interpretation_boundary": {
            "valid_description": (
                "hotter, cooler, mixed, or unchanged published inflation compared with "
                "the previous release"
            ),
            "not_available": "timestamped historical market-consensus expectations",
            "forbidden_claim": (
                "Do not call these values surprises and do not infer that the information "
                "was unexpected or unpriced."
            ),
        },
        "frozen_direct_batch": {
            "primary_hypothesis": (
                "Cooler published inflation maps to positive crypto direction and hotter "
                "published inflation maps to negative crypto direction."
            ),
            "immediate_assets": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "immediate_timeframe": "1m",
            "immediate_horizons_minutes": [5, 15, 30, 60, 120, 240],
            "later_scopes": [
                "btc",
                "eth",
                "established_alt_median",
                "meme_median",
                "broad_market_median",
            ],
            "later_timeframe": "1h",
            "later_horizons_hours": [1, 2, 4, 8, 24],
            "later_anchor_rule": (
                "Start at the first complete hourly candle after the 08:30 release; do not "
                "mix the pre-release half hour into the outcome."
            ),
            "controls": [
                "always up",
                "always down",
                "same-direction pre-event movement",
                "previous release's available inflation-temperature sign",
                "repeat after excluding releases within 24 hours of FOMC",
            ],
            "lead_rule": (
                "At least 10 usable cases in development and validation; at least 55% "
                "direction accuracy in both; and at least two percentage points better "
                "overall than the best constant, pre-event direction, and "
                "previous-release-sign controls."
            ),
            "strong_target": "65% under the same cross-period and control requirements",
            "selection_rule": "finish every frozen sign, scope, and horizon before branching",
        },
        "later_queue_not_part_of_this_batch": [
            "release sign plus first 5/15-minute BTC/ETH confirmation",
            "leader-to-group transmission",
            "local single-level and level-cluster modification",
            "post-event range and fade behaviour",
        ],
        "source_contracts": {
            "fred_series_observations_documentation": FRED_OBSERVATIONS_DOC,
            "bls_cpi_archive": BLS_CPI_ARCHIVE,
            "bls_cpi_schedule": BLS_CPI_SCHEDULE,
            "data_attribution": "U.S. Bureau of Labor Statistics via FRED/ALFRED",
            "api_key": "local_secret_used_but_not_serialized",
            "requests": source_contracts,
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def render_report(freeze: Mapping[str, Any]) -> str:
    family = freeze["family"]
    return "\n".join(
        [
            "# CPI Release Family Freeze",
            "",
            f"- Status: `{freeze['status']}`",
            "- Market outcomes read: **No**",
            f"- Complete usable releases: `{family['event_count']}`",
            f"- Missing official rows excluded: `{family['missing_release_rows']}`",
            "- Historical market expectations: **not available and not imputed**",
            "",
            "## Plain boundary",
            "",
            (
                "This family says whether published inflation was hotter or cooler than in "
                "the previous release. It does not say whether traders expected it."
            ),
            "",
            "## Frozen next test",
            "",
            (
                "Test every predeclared measure against Bitcoin and Ethereum minute reactions "
                "and against later broad, established-coin, and meme-coin movement. Complete "
                "the full sibling batch before following any apparent winner."
            ),
            "",
        ]
    )


def _write_sources(
    initial: Mapping[str, Mapping[str, Any]],
    vintages: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for kind, documents in (("initial", initial), ("vintages", vintages)):
        for series_id, document in documents.items():
            path = SOURCE_ROOT / f"fred_{series_id}_{kind}.json"
            g0.atomic_write_json(document, path)
            artifacts[f"{series_id}_{kind}"] = artifact(path)
    return artifacts


def execute(*, key_file: Path, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "cpi_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_cpi_family_freeze":
            raise ValueError("Existing CPI freeze result is not terminal.")
        return result

    api_key = load_api_key(key_file)
    initial, vintages, request_contracts = fetch_family_sources(api_key)
    catalog, coverage = build_release_catalog(initial, vintages)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_artifacts = _write_sources(initial, vintages)
    freeze = build_freeze_document(catalog, coverage, request_contracts)
    serialized = json.dumps(freeze, sort_keys=True)
    if api_key in serialized:
        raise ValueError("Secret API key would be serialized; refusing to write freeze.")

    catalog_path = OUTPUT_ROOT / "cpi_release_catalog.csv"
    coverage_path = OUTPUT_ROOT / "cpi_source_coverage.csv"
    freeze_path = OUTPUT_ROOT / "cpi_freeze.json"
    report_path = OUTPUT_ROOT / "cpi_freeze_report.md"
    g0.atomic_write_csv(catalog, catalog_path)
    g0.atomic_write_csv(coverage, coverage_path)
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_cpi_family_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "event_count": len(catalog),
        "missing_release_rows": len(coverage),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "catalog": artifact(catalog_path),
            "coverage": artifact(coverage_path),
            "report": artifact(report_path),
            "sources": source_artifacts,
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
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
                    "key_will_be_serialized": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(
        json.dumps(
            execute(key_file=args.key_file, overwrite=args.overwrite),
            indent=2,
        )
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
