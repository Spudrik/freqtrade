"""Build an outcome-blind ESMA sovereign-rating publication catalogue."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
import time
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
    "esma_sovereign_rating_catalogue_20260906a"
)
RAW_PATH = OUTPUT_ROOT / "esma_sovereign_rating_action_rows.parquet"
CATALOG_PATH = OUTPUT_ROOT / "esma_sovereign_rating_event_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "esma_sovereign_rating_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "esma_sovereign_rating_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "esma_sovereign_rating_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "esma_sovereign_rating_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "esma_sovereign_rating_catalogue_result.json"

SOLR_URL = (
    "https://registers.esma.europa.eu/solr/esma_registers_radar/select"
)
DATASET_URL = "https://data.europa.eu/data/datasets/eu-rp?locale=en"
HELP_URL = "https://registers.esma.europa.eu/publication/helpApp"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2026-01-01T00:00:00Z")
PAGE_SIZE = 5000
ACTION_YEAR_PAGE_SIZE = 12000
REQUEST_TIMEOUT = 90
REQUEST_PAUSE_SECONDS = 0.15
USER_AGENT = "Objective02b-ESMA-sovereign-rating-audit/1.0"

JURISDICTIONS: dict[str, dict[str, Any]] = {
    "US": {
        "label": "United States",
        "group": "major_global",
        "aliases": (
            "United States of America",
            "United States of America, Government of",
        ),
    },
    "GB": {
        "label": "United Kingdom",
        "group": "major_global",
        "aliases": (
            "United Kingdom of Great Britain and Northern Ireland",
            "United Kingdom",
            "United Kingdom, Government of",
            "United Kingdom of Great Britain and Nothern Ireland",
        ),
    },
    "JP": {
        "label": "Japan",
        "group": "major_global",
        "aliases": ("Japan", "Japan, Government of"),
    },
    "CN": {
        "label": "China",
        "group": "major_global",
        "aliases": (
            "People's Republic of China",
            "China",
            "China, People's Republic of",
            "China, Government of",
        ),
    },
    "AT": {
        "label": "Austria",
        "group": "selected_euro_area",
        "aliases": (
            "Austria, Republic of",
            "Republic of Austria",
            "Austria",
            "AUSTRIA",
            "Austria, Government of",
        ),
    },
    "BE": {
        "label": "Belgium",
        "group": "selected_euro_area",
        "aliases": (
            "Belgium, Kingdom of",
            "Kingdom of Belgium",
            "Belgium",
            "Belgium, Government of",
        ),
    },
    "CY": {
        "label": "Cyprus",
        "group": "selected_euro_area",
        "aliases": (
            "Cyprus, Republic of",
            "Cyprus",
            "CYPRUS, REPUBLIC OF (GOVERNMENT)",
            "Republic of Cyprus",
            "Cyprus, Government of",
        ),
    },
    "DE": {
        "label": "Germany",
        "group": "selected_euro_area",
        "aliases": (
            "Germany, Federal Republic of",
            "Federal Republic of Germany",
            "Germany",
            "Germany, Government of",
        ),
    },
    "EE": {
        "label": "Estonia",
        "group": "selected_euro_area",
        "aliases": (
            "Estonia, Republic of",
            "ESTONIA, REPUBLIC OF (GOVERNMENT)",
            "Estonia",
            "Republic of Estonia",
            "Estonia, Government of",
        ),
    },
    "ES": {
        "label": "Spain",
        "group": "selected_euro_area",
        "aliases": (
            "Spain, Kingdom of",
            "Kingdom of Spain",
            "Spain",
            "Spain, Government of",
        ),
    },
    "FI": {
        "label": "Finland",
        "group": "selected_euro_area",
        "aliases": (
            "Finland, Republic of",
            "Republic of Finland",
            "Finland",
            "Finland, Government of",
        ),
    },
    "FR": {
        "label": "France",
        "group": "selected_euro_area",
        "aliases": (
            "France, Republic of",
            "French Republic",
            "Republic of France",
            "French Republic (France)",
            "France",
            "Republique Francaise",
        ),
    },
    "GR": {
        "label": "Greece",
        "group": "selected_euro_area",
        "aliases": (
            "Hellenic Republic",
            "Greece",
            "Hellenic Republic (Greece)",
            "Greece, Government of",
        ),
    },
    "HR": {
        "label": "Croatia",
        "group": "selected_euro_area",
        "aliases": (
            "Croatia, Republic of",
            "Republic of Croatia",
            "Croatia",
            "Croatia, Government of",
        ),
    },
    "IE": {
        "label": "Ireland",
        "group": "selected_euro_area",
        "aliases": (
            "Ireland, Republic of",
            "Ireland",
            "Ireland, Government of",
            "Republic of Ireland",
        ),
    },
    "IT": {
        "label": "Italy",
        "group": "selected_euro_area",
        "aliases": (
            "Italy, Republic of",
            "Republic of Italy",
            "Italian Republic",
            "Italy",
            "Italy, Government of",
        ),
    },
    "LT": {
        "label": "Lithuania",
        "group": "selected_euro_area",
        "aliases": (
            "Lithuania, Republic of",
            "Republic of Lithuania",
            "Lithuania",
            "Lithuania, Government of",
        ),
    },
    "LU": {
        "label": "Luxembourg",
        "group": "selected_euro_area",
        "aliases": (
            "Luxembourg, Grand Duchy of",
            "Grand Duchy of Luxembourg",
            "LUXEMBOURG, GRAND DUCHY OF (GOVERNMENT)",
            "Luxembourg",
            "Luxembourg, Government of",
        ),
    },
    "LV": {
        "label": "Latvia",
        "group": "selected_euro_area",
        "aliases": (
            "Latvia, Republic of",
            "Republic of Latvia",
            "Latvia",
            "Latvia, Government of",
        ),
    },
    "MT": {
        "label": "Malta",
        "group": "selected_euro_area",
        "aliases": (
            "Malta, Republic of",
            "MALTA, REPUBLIC OF (GOVERNMENT)",
            "Republic of Malta",
            "Malta",
            "Malta, Government of",
        ),
    },
    "NL": {
        "label": "Netherlands",
        "group": "selected_euro_area",
        "aliases": (
            "The Netherlands, Kingdom of",
            "Kingdom of the Netherlands",
            "Netherlands",
            "State of The Netherlands",
            "Netherlands, Government of",
        ),
    },
    "PT": {
        "label": "Portugal",
        "group": "selected_euro_area",
        "aliases": (
            "Portugal, Republic of",
            "Portuguese Republic",
            "Republic of Portugal",
            "Portuguese Republic (Portugal)",
            "Portugal, Government of",
            "Portugal",
            "REPUBLICA DE PORTUGAL",
        ),
    },
    "SI": {
        "label": "Slovenia",
        "group": "selected_euro_area",
        "aliases": (
            "Slovenia, Republic of",
            "Republic of Slovenia",
            "Slovenia",
            "The Republic of Slovenia",
            "Slovenia, Government of",
        ),
    },
    "SK": {
        "label": "Slovakia",
        "group": "selected_euro_area",
        "aliases": (
            "Slovak Republic",
            "Slovakia",
            "Slovakia, Government of",
        ),
    },
}

PARENT_FIELDS = (
    "id,craCode,craName,issuerName,countryName,countryCode,ratingTypeCode,"
    "sectorCode,ratedObjectCode,timeHorizonType,timeHorizonDescr,"
    "localForeignCurrencyCode,localForeignCurrencyValue,"
    "ratingIssuanceLocationType,ratingIssuanceLocationDesc,respCraLeiCode"
)
ACTION_FIELDS = (
    "id,actionId,parent_id,actionsActionType,actionsActionTypeLabel,"
    "actionsRatingValueLabel,actionsRacValidityDatetime,actionsDefaultFlag,"
    "actionsRatingIssuanceLocationType,actionsRatingIssuanceLocationDesc"
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def official_solr_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == "registers.esma.europa.eu"
        and parsed.path == "/solr/esma_registers_radar/select"
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.fragment
    )


def jurisdiction_query() -> str:
    codes = " OR ".join(JURISDICTIONS)
    return (
        "sectorCode:SV AND ratedObjectCode:ISR AND "
        f"countryCode:({codes})"
    )


def fetch_solr_documents(
    query: str,
    *,
    filters: Sequence[str],
    fields: str,
    requester: Callable[..., Any] = requests.get,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
    page_size: int = PAGE_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not official_solr_url(SOLR_URL):
        raise ValueError("ESMA Solr source URL is not official")
    documents: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    start = 0
    expected_total: int | None = None
    while expected_total is None or start < expected_total:
        params: list[tuple[str, str | int]] = [
            ("q", query),
            ("start", start),
            ("rows", page_size),
            ("sort", "id asc"),
            ("fl", fields),
            ("wt", "json"),
        ]
        params.extend(("fq", value) for value in filters)
        response = requester(
            SOLR_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        )
        content = bytes(response.content)
        if int(response.status_code) != 200:
            raise RuntimeError(
                f"Official ESMA source returned {response.status_code}: {response.url}"
            )
        if not official_solr_url(str(response.url)):
            raise ValueError("ESMA response did not come from the approved official URL")
        payload = json.loads(content)
        if int(payload.get("responseHeader", {}).get("status", -1)) != 0:
            raise ValueError("Official ESMA Solr response was not successful")
        result = payload.get("response", {})
        total = int(result.get("numFound", -1))
        page = list(result.get("docs", []))
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise ValueError("ESMA result count changed during the paged source audit")
        if total > start and not page:
            raise ValueError("ESMA returned an empty page before the result ended")
        documents.extend(page)
        manifest.append(
            {
                "url": str(response.url),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "start": start,
                "returned": len(page),
                "num_found": total,
            }
        )
        start += len(page)
        if len(page) == 0:
            break
        if pause_seconds:
            time.sleep(pause_seconds)
    if expected_total is None or len(documents) != expected_total:
        raise ValueError("ESMA paged result is incomplete")
    return documents, manifest


def selected_parent_frame(documents: Sequence[Mapping[str, Any]]) -> DataFrame:
    frame = DataFrame.from_records(documents)
    required = {
        "id",
        "craCode",
        "craName",
        "issuerName",
        "countryCode",
        "ratingTypeCode",
        "sectorCode",
        "ratedObjectCode",
    }
    if not required.issubset(frame):
        raise ValueError(f"ESMA parent records are missing: {required - set(frame)}")
    allowed_pairs = pd.MultiIndex.from_tuples(
        [
            (code, alias)
            for code, config in JURISDICTIONS.items()
            for alias in config["aliases"]
        ],
        names=["countryCode", "issuerName"],
    )
    record_pairs = pd.MultiIndex.from_frame(
        frame[["countryCode", "issuerName"]].astype("string")
    )
    selected = frame.loc[record_pairs.isin(allowed_pairs)].copy()
    if selected.empty:
        raise ValueError("No national-sovereign ESMA parent records were selected")
    if selected["id"].duplicated().any():
        raise ValueError("ESMA parent IDs are not unique")
    if set(selected["countryCode"]) != set(JURISDICTIONS):
        missing = set(JURISDICTIONS) - set(selected["countryCode"])
        raise ValueError(f"ESMA has no selected national-sovereign parents for: {missing}")
    if not selected["sectorCode"].eq("SV").all():
        raise ValueError("Selected ESMA parents are not state ratings")
    if not selected["ratedObjectCode"].eq("ISR").all():
        raise ValueError("Selected ESMA parents are not issuer ratings")
    return selected


def source_action_sign(code: str, label: str) -> str:
    code = code.strip().upper()
    text = label.casefold()
    if code == "DG":
        return "negative_rating_action"
    if code == "UP":
        return "positive_rating_action"
    if code == "DF":
        if "removed" in text and "default" in text:
            return "positive_rating_action"
        if "default" in text:
            return "negative_rating_action"
        return "unsigned_or_lifecycle_action"
    if code == "SP":
        if "removed" in text and "suspension" in text:
            return "positive_rating_action"
        if "suspension" in text:
            return "negative_rating_action"
        return "unsigned_or_lifecycle_action"
    if "placed under negative" in text or "removed under positive" in text:
        return "negative_rating_action"
    if "placed under positive" in text or "removed under negative" in text:
        return "positive_rating_action"
    return "unsigned_or_lifecycle_action"


def action_category(code: str, label: str) -> str:
    code = code.strip().upper()
    sign = source_action_sign(code, label)
    if sign != "unsigned_or_lifecycle_action":
        return "signed_change_candidate"
    if code == "AF" or "maintained under" in label.casefold():
        return "ordinary_affirmation"
    if code in {"NW", "WD"}:
        return "database_or_rating_lifecycle"
    return "other_unsigned_action"


def build_action_rows(
    parents: DataFrame, action_documents: Sequence[Mapping[str, Any]]
) -> DataFrame:
    actions = DataFrame.from_records(action_documents)
    required = {
        "id",
        "actionId",
        "parent_id",
        "actionsActionType",
        "actionsActionTypeLabel",
        "actionsRacValidityDatetime",
    }
    if not required.issubset(actions):
        raise ValueError(f"ESMA action records are missing: {required - set(actions)}")
    if actions.duplicated(["id", "parent_id"]).any():
        raise ValueError("ESMA raw action document/parent pairs are not unique")
    actions = actions.loc[actions["parent_id"].isin(set(parents["id"]))].copy()
    parent_columns = [
        "id",
        "craCode",
        "craName",
        "issuerName",
        "countryName",
        "countryCode",
        "timeHorizonType",
        "timeHorizonDescr",
        "localForeignCurrencyCode",
        "localForeignCurrencyValue",
    ]
    optional_parent_columns = set(parent_columns) - {
        "id",
        "craCode",
        "craName",
        "issuerName",
        "countryCode",
    }
    for column in optional_parent_columns:
        if column not in parents:
            parents[column] = pd.NA
    if "actionsRatingValueLabel" not in actions:
        actions["actionsRatingValueLabel"] = pd.NA
    output = actions.merge(
        parents[parent_columns],
        left_on="parent_id",
        right_on="id",
        how="left",
        validate="many_to_one",
        suffixes=("_action", "_parent"),
    )
    output["available_at_utc"] = pd.to_datetime(
        output["actionsRacValidityDatetime"],
        utc=True,
        errors="raise",
        format="ISO8601",
    )
    output = output.loc[
        output["available_at_utc"].ge(START)
        & output["available_at_utc"].lt(END_EXCLUSIVE)
    ].copy()
    output["jurisdiction"] = output["countryCode"].map(
        {code: config["label"] for code, config in JURISDICTIONS.items()}
    )
    output["jurisdiction_group"] = output["countryCode"].map(
        {code: config["group"] for code, config in JURISDICTIONS.items()}
    )
    output["source_action_sign"] = [
        source_action_sign(str(code), str(label))
        for code, label in zip(
            output["actionsActionType"],
            output["actionsActionTypeLabel"],
            strict=True,
        )
    ]
    output["action_category"] = [
        action_category(str(code), str(label))
        for code, label in zip(
            output["actionsActionType"],
            output["actionsActionTypeLabel"],
            strict=True,
        )
    ]
    output["market_expectation_status"] = "unavailable_not_imputed"
    output["timestamp_quality"] = "esma_reported_utc_publication_time"
    output["source_url"] = SOLR_URL
    return output.sort_values(
        ["available_at_utc", "countryCode", "craCode", "id_action"], kind="stable"
    ).reset_index(drop=True)


def json_values(values: Sequence[Any]) -> str:
    cleaned = sorted({str(value) for value in values if pd.notna(value)})
    return json.dumps(cleaned, ensure_ascii=True, separators=(",", ":"))


def partition_for_year(year: int) -> str:
    if year == 2020:
        return "context_only_2020"
    if year <= 2023:
        return "development_2021_2023"
    return "internal_validation_2024_2025"


def build_catalogue(action_rows: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["countryCode", "available_at_utc", "craCode"]
    for (country_code, timestamp, cra_code), group in action_rows.groupby(
        keys, sort=True, dropna=False
    ):
        signs = set(group["source_action_sign"]) - {"unsigned_or_lifecycle_action"}
        if signs == {"positive_rating_action"}:
            source_sign = "positive_rating_action"
        elif signs == {"negative_rating_action"}:
            source_sign = "negative_rating_action"
        elif len(signs) > 1:
            source_sign = "mixed_rating_actions"
        else:
            source_sign = "unsigned_or_lifecycle_action"
        stamp = pd.Timestamp(timestamp)
        safe_cra = re.sub(r"[^A-Za-z0-9]+", "_", str(cra_code)).strip("_")
        source_key = f"{country_code}|{stamp.isoformat()}|{cra_code}"
        source_digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
        clock_digest = hashlib.sha256(
            stamp.isoformat().encode("utf-8")
        ).hexdigest()[:12]
        country_clock_digest = hashlib.sha256(
            f"{country_code}|{stamp.isoformat()}".encode()
        ).hexdigest()[:12]
        records.append(
            {
                "event_id": (
                    f"esma_sovereign_{country_code}_"
                    f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{safe_cra}_{source_digest}"
                ),
                "market_window_group_id": f"esma_clock_{clock_digest}",
                "country_clock_group_id": (
                    f"esma_sovereign_{country_code}_clock_{country_clock_digest}"
                ),
                "available_at_utc": stamp,
                "country_code": country_code,
                "jurisdiction": group["jurisdiction"].iloc[0],
                "jurisdiction_group": group["jurisdiction_group"].iloc[0],
                "cra_code": cra_code,
                "cra_names_json": json_values(group["craName"]),
                "issuer_names_json": json_values(group["issuerName"]),
                "action_codes_json": json_values(group["actionsActionType"]),
                "action_labels_json": json_values(group["actionsActionTypeLabel"]),
                "rating_values_json": json_values(group["actionsRatingValueLabel"]),
                "source_action_sign": source_sign,
                "action_categories_json": json_values(group["action_category"]),
                "raw_action_rows": len(group),
                "parent_rating_records": group["parent_id"].nunique(),
                "action_ids": group["actionId"].nunique(),
                "time_horizons_json": json_values(group["timeHorizonDescr"]),
                "currency_scopes_json": json_values(
                    group["localForeignCurrencyValue"]
                ),
                "timestamp_quality": "esma_reported_utc_publication_time",
                "market_expectation_status": "unavailable_not_imputed",
                "whole_event_partition": partition_for_year(stamp.year),
                "source_url": SOLR_URL,
                "dataset_url": DATASET_URL,
                "selection_method": (
                    "fixed_national_sovereign_aliases_state_issuer_ratings"
                ),
            }
        )
    catalogue = DataFrame.from_records(records)
    validate_catalogue(catalogue, action_rows)
    return catalogue.sort_values("available_at_utc", kind="stable").reset_index(
        drop=True
    )


def validate_catalogue(catalogue: DataFrame, action_rows: DataFrame) -> None:
    required = {
        "event_id",
        "market_window_group_id",
        "country_clock_group_id",
        "available_at_utc",
        "country_code",
        "jurisdiction",
        "jurisdiction_group",
        "cra_code",
        "source_action_sign",
        "raw_action_rows",
        "timestamp_quality",
        "market_expectation_status",
        "whole_event_partition",
        "source_url",
        "dataset_url",
        "selection_method",
    }
    if not required.issubset(catalogue):
        raise ValueError(f"ESMA catalogue is missing: {required - set(catalogue)}")
    if catalogue.empty or catalogue["event_id"].duplicated().any():
        raise ValueError("ESMA publication clusters are empty or not unique")
    if set(catalogue["country_code"]) != set(JURISDICTIONS):
        raise ValueError("ESMA publication clusters do not cover every jurisdiction")
    timestamps = pd.to_datetime(catalogue["available_at_utc"], utc=True, errors="raise")
    if not (timestamps.ge(START) & timestamps.lt(END_EXCLUSIVE)).all():
        raise ValueError("ESMA publication cluster timestamp is outside the frozen period")
    if int(catalogue["raw_action_rows"].sum()) != len(action_rows):
        raise ValueError("ESMA publication clustering lost raw action rows")
    fixed_values = {
        "timestamp_quality": "esma_reported_utc_publication_time",
        "market_expectation_status": "unavailable_not_imputed",
        "source_url": SOLR_URL,
        "dataset_url": DATASET_URL,
        "selection_method": "fixed_national_sovereign_aliases_state_issuer_ratings",
    }
    for column, expected in fixed_values.items():
        if not catalogue[column].eq(expected).all():
            raise ValueError(f"ESMA catalogue {column} changed")
    forbidden = {"profit", "future_return", "price", "volume"}
    if forbidden.intersection(column.casefold() for column in catalogue.columns):
        raise ValueError("ESMA source catalogue contains a market outcome column")


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    return (
        catalogue.groupby(
            ["jurisdiction_group", "country_code", "jurisdiction"], sort=True
        )
        .agg(
            publication_clusters=("event_id", "nunique"),
            signed_change_candidates=(
                "source_action_sign",
                lambda values: int((values != "unsigned_or_lifecycle_action").sum()),
            ),
            first_publication_utc=("available_at_utc", "min"),
            last_publication_utc=("available_at_utc", "max"),
        )
        .reset_index()
    )


def freeze_document(
    parents: DataFrame,
    action_rows: DataFrame,
    catalogue: DataFrame,
    counts: DataFrame,
) -> dict[str, Any]:
    action_vocabulary = (
        action_rows.groupby(
            [
                "actionsActionType",
                "actionsActionTypeLabel",
                "source_action_sign",
                "action_category",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("raw_action_rows")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "schema_version": 1,
        "status": "frozen_esma_sovereign_ratings_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "direction_tested": False,
        "plain_question": (
            "Can ESMA supply enough distinct, exactly timed national-sovereign rating "
            "publications for a later activity-first crypto test?"
        ),
        "scope_name": "ESMA_reporting_universe_selected_national_sovereigns",
        "period_utc": {
            "start_inclusive": START.isoformat(),
            "end_exclusive": END_EXCLUSIVE.isoformat(),
        },
        "jurisdictions": {
            code: {
                "label": config["label"],
                "group": config["group"],
                "issuer_aliases": list(config["aliases"]),
            }
            for code, config in JURISDICTIONS.items()
        },
        "source_query": {
            "solr_url": SOLR_URL,
            "parent_filter": jurisdiction_query(),
            "parent_type": "type_s:parent",
            "action_child_join": "{!child of=type_s:parent}",
            "action_time_field": "actionsRacValidityDatetime",
        },
        "cluster_rule": (
            "One event per country, exact ESMA action-validity/publication timestamp, "
            "and reporting CRA code; rating dimensions and repeated action records at "
            "that clock remain joined."
        ),
        "independence_rule": (
            "Agency-level publication records remain separate, but all records at the "
            "same exact UTC clock share market_window_group_id. Later market tests must "
            "count, split, and hold out by that group so simultaneous publications are "
            "not treated as independent evidence."
        ),
        "source_action_sign_rule": {
            "negative": [
                "downgrade",
                "default",
                "suspension",
                "placed under negative outlook/watch",
                "removed from positive outlook/watch",
            ],
            "positive": [
                "upgrade",
                "placed under positive outlook/watch",
                "removed from negative outlook/watch",
            ],
            "meaning": (
                "This is the direction of the official rating action, not a crypto "
                "price prediction."
            ),
        },
        "observed_action_vocabulary": action_vocabulary,
        "selected_parent_records": len(parents),
        "raw_action_rows": len(action_rows),
        "publication_clusters": len(catalogue),
        "signed_change_candidate_clusters": int(
            catalogue["source_action_sign"]
            .ne("unsigned_or_lifecycle_action")
            .sum()
        ),
        "counts": counts.to_dict(orient="records"),
        "boundaries": [
            "Coverage is the ESMA reporting universe, not all global rating agencies.",
            "The ESMA UTC action-validity field is treated as the reported publication "
            "clock under the applicable reporting standard.",
            "Multiple rows at one agency clock are one publication, not independent "
            "market events.",
            "All countries and agencies sharing an exact clock carry one common market "
            "window group for downstream independence.",
            "Historical market expectations are unavailable and are not imputed.",
            "No crypto outcome, direction, causation, profit, or trading rule is tested.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    rows = [
        "# ESMA Sovereign-Rating Source Audit",
        "",
        f"- Selected national-sovereign parent records: "
        f"`{freeze['selected_parent_records']}`",
        f"- Raw rating-action rows: `{freeze['raw_action_rows']}`",
        f"- Agency/time publication clusters: `{freeze['publication_clusters']}`",
        f"- Signed rating-change candidates: "
        f"`{freeze['signed_change_candidate_clusters']}`",
        "- Market outcomes read: **No**",
        "- Profit used: **No**",
        "",
        "| Group | Country | Publication clusters | Signed candidates |",
        "|---|---|---:|---:|",
    ]
    rows.extend(
        f"| {row['jurisdiction_group']} | {row['jurisdiction']} | "
        f"{row['publication_clusters']} | {row['signed_change_candidates']} |"
        for row in freeze["counts"]
    )
    rows.extend(
        [
            "",
            "This is a source-coverage result. It does not show that ratings move "
            "crypto or predict crypto direction. A market test may be frozen only "
            "after the independent publication counts and chronology are reviewed.",
            "",
        ]
    )
    return "\n".join(rows)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_esma_sovereign_rating_catalogue":
        raise ValueError("Existing ESMA catalogue result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing ESMA catalogue artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    base_query = jurisdiction_query()
    parent_documents: list[dict[str, Any]] = []
    parent_manifest: list[dict[str, Any]] = []
    for country_code in JURISDICTIONS:
        country_query = (
            "type_s:parent AND sectorCode:SV AND ratedObjectCode:ISR AND "
            f"countryCode:{country_code}"
        )
        documents, manifest = fetch_solr_documents(
            country_query,
            filters=(),
            fields=PARENT_FIELDS,
        )
        parent_documents.extend(documents)
        parent_manifest.extend(manifest)
    action_documents: list[dict[str, Any]] = []
    action_manifest: list[dict[str, Any]] = []
    action_query = f"{{!child of=type_s:parent}}({base_query})"
    for year in range(START.year, END_EXCLUSIVE.year):
        documents, manifest = fetch_solr_documents(
            action_query,
            filters=(
                "entity_type:action",
                "actionsRacValidityDatetime:"
                f"[{year}-01-01T00:00:00Z TO {year + 1}-01-01T00:00:00Z}}",
            ),
            fields=ACTION_FIELDS,
            page_size=ACTION_YEAR_PAGE_SIZE,
        )
        action_documents.extend(documents)
        action_manifest.extend(manifest)
    parents = selected_parent_frame(parent_documents)
    action_rows = build_action_rows(parents, action_documents)
    catalogue = build_catalogue(action_rows)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(parents, action_rows, catalogue, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(action_rows, RAW_PATH)
    g0.atomic_write_csv(catalogue, CATALOG_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(
        {
            "retrieved_at_utc": g0.utc_now(),
            "official_dataset_url": DATASET_URL,
            "official_machine_to_machine_help": HELP_URL,
            "requests": [*parent_manifest, *action_manifest],
        },
        SOURCE_MANIFEST_PATH,
    )
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_esma_sovereign_rating_catalogue",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "selected_parent_records": len(parents),
        "raw_action_rows": len(action_rows),
        "publication_clusters": len(catalogue),
        "signed_change_candidate_clusters": int(
            catalogue["source_action_sign"]
            .ne("unsigned_or_lifecycle_action")
            .sum()
        ),
        "artifacts": {
            "raw_actions": artifact(RAW_PATH),
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
                    "profit_will_be_used": False,
                    "jurisdictions": len(JURISDICTIONS),
                    "period": {
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
