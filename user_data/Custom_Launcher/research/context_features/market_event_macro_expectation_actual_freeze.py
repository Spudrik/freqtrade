"""Join frozen market-implied expectations to official first-published macro values."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as fred_helpers,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
EXPECTATION_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "macro_expectation_catalogue_20260910a"
)
EXPECTATION_RESULT_PATH = EXPECTATION_ROOT / "kalshi_macro_expectation_catalogue_result.json"
EXPECTATION_EVENTS_PATH = EXPECTATION_ROOT / "kalshi_macro_expectation_events.csv"
EXPECTATION_SUMMARY_PATH = EXPECTATION_ROOT / "kalshi_macro_expectation_source_summary.csv"

CPI_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "cpi_release_family_20260904a"
)
CPI_RESULT_PATH = CPI_ROOT / "cpi_freeze_result.json"
CPI_CATALOG_PATH = CPI_ROOT / "cpi_release_catalog.csv"

BROAD_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "broad_event_relevance_20260904a"
)
BROAD_RESULT_PATH = BROAD_ROOT / "broad_relevance_freeze_result.json"
BROAD_CATALOG_PATH = BROAD_ROOT / "broad_relevance_event_catalog.csv"

FOMC_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "layer1_honest_history_20260903a"
)
FOMC_RESULT_PATH = FOMC_ROOT / "layer1_result.json"
FOMC_CATALOG_PATH = FOMC_ROOT / "official_fomc_event_catalog.csv"

OUTPUT_ROOT = EXPECTATION_ROOT / "official_actual_freeze_20260911a"
CATALOG_PATH = OUTPUT_ROOT / "macro_expectation_actual_catalog.csv"
SOURCE_PATH = OUTPUT_ROOT / "fred_official_actual_sources.json"
FREEZE_PATH = OUTPUT_ROOT / "macro_expectation_actual_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "macro_expectation_actual_freeze_result.json"

DEFAULT_KEY_FILE = fred_helpers.DEFAULT_KEY_FILE
MAX_EXPECTATION_LEAD = pd.Timedelta(hours=12)
NEAR_RELEASE_LEAD = pd.Timedelta(minutes=10)
SOURCE_READY = "cross_period_ready"
FED_CATEGORY_ORDER = {
    "Cut >25bps": -2,
    "Cut 25bps": -1,
    "Hike 0bps": 0,
    "Hike 25bps": 1,
    "Hike >25bps": 2,
}
MONTH_NUMBER = {
    month: number
    for number, month in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
REFERENCE_PERIOD_RE = re.compile(
    r"-(?P<year>\d{2})(?P<month>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"(?P<known_variant>B|T)?$"
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def verify_parent(
    result_path: Path,
    *,
    expected_status: str,
    artifact_key: str,
    artifact_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Parent result is not terminal: {result_path}")
    expected_hash = result["artifacts"][artifact_key]["sha256"]
    if expected_hash != g0.sha256_file(artifact_path):
        raise ValueError(f"Parent artifact changed: {artifact_path}")
    return result


def load_inputs() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    verify_parent(
        EXPECTATION_RESULT_PATH,
        expected_status="completed_full_macro_expectation_catalogue",
        artifact_key="events",
        artifact_path=EXPECTATION_EVENTS_PATH,
    )
    verify_parent(
        EXPECTATION_RESULT_PATH,
        expected_status="completed_full_macro_expectation_catalogue",
        artifact_key="summary",
        artifact_path=EXPECTATION_SUMMARY_PATH,
    )
    verify_parent(
        CPI_RESULT_PATH,
        expected_status="completed_cpi_family_freeze",
        artifact_key="catalog",
        artifact_path=CPI_CATALOG_PATH,
    )
    verify_parent(
        BROAD_RESULT_PATH,
        expected_status="completed_broad_event_relevance_freeze",
        artifact_key="catalog",
        artifact_path=BROAD_CATALOG_PATH,
    )
    verify_parent(
        FOMC_RESULT_PATH,
        expected_status="completed_event_hierarchy_layer1",
        artifact_key="event_catalog",
        artifact_path=FOMC_CATALOG_PATH,
    )

    expectations = pd.read_csv(EXPECTATION_EVENTS_PATH)
    expectations["market_close_time_utc"] = pd.to_datetime(
        expectations["market_close_time_utc"], utc=True, format="mixed"
    )
    summary = pd.read_csv(EXPECTATION_SUMMARY_PATH)
    cpi = pd.read_csv(CPI_CATALOG_PATH)
    cpi["anchor_utc"] = pd.to_datetime(cpi["anchor_utc"], utc=True)
    broad = pd.read_csv(BROAD_CATALOG_PATH)
    broad["anchor_utc"] = pd.to_datetime(broad["anchor_utc"], utc=True)
    employment = broad[broad["event_family"].eq("employment")].copy()
    fomc = pd.read_csv(FOMC_CATALOG_PATH)
    fomc["anchor_utc"] = pd.to_datetime(fomc["anchor_utc"], utc=True)
    fomc = fomc[fomc["information_status"].eq("historical_official_release")].copy()
    return expectations, summary, cpi, employment, fomc


def match_following_anchor(
    close_time: pd.Timestamp,
    official: DataFrame,
    *,
    reference_period: str | None = None,
    period_column: str | None = None,
) -> Mapping[str, Any] | None:
    same_date = official[official["anchor_utc"].dt.date.eq(close_time.date())].copy()
    if reference_period is not None:
        if period_column is None or period_column not in same_date:
            raise ValueError("A period column is required for reference-period matching")
        same_date = same_date[
            pd.to_datetime(same_date[period_column]).dt.date.astype(str)
            == reference_period
        ].copy()
    same_date["anchor_delay"] = same_date["anchor_utc"] - close_time
    matches = same_date[
        same_date["anchor_delay"].gt(pd.Timedelta(0))
        & same_date["anchor_delay"].le(MAX_EXPECTATION_LEAD)
    ]
    if len(matches) > 1:
        raise ValueError(f"Several official anchors match cutoff {close_time}")
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def reference_period_from_event_ticker(event_ticker: str) -> str:
    match = REFERENCE_PERIOD_RE.search(event_ticker.upper())
    if match is None:
        raise ValueError(f"Cannot read reference month from event ticker {event_ticker!r}")
    year = 2000 + int(match.group("year"))
    month = MONTH_NUMBER[match.group("month")]
    return f"{year:04d}-{month:02d}-01"


def restrict_employment_to_expectations(
    expectations: DataFrame,
    employment: DataFrame,
) -> DataFrame:
    selected = expectations[
        expectations["component"].isin({"nonfarm_payrolls", "unemployment_rate"})
        & expectations["usable_2h"].map(strict_bool)
    ]
    event_ids: set[str] = set()
    for row in selected.to_dict(orient="records"):
        close_time = pd.Timestamp(row["market_close_time_utc"])
        reference_period = reference_period_from_event_ticker(str(row["event_ticker"]))
        matched = match_following_anchor(
            close_time,
            employment,
            reference_period=reference_period,
            period_column="observation_period",
        )
        if matched is not None:
            event_ids.add(str(matched["event_id"]))
    restricted = employment[employment["event_id"].isin(event_ids)].copy()
    if restricted.empty:
        raise ValueError("No official employment periods match usable expectations")
    return restricted


def fetch_fred_json(
    endpoint: str,
    *,
    params: Mapping[str, Any],
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the shared client while preventing API keys appearing in failures."""

    try:
        return fred_helpers.fetch_json(endpoint, params=params, api_key=api_key)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        series = params.get("series_id", "unknown")
        raise RuntimeError(
            f"FRED request failed with status {status} for {endpoint}, series {series}"
        ) from None


def fetch_fred_actual_sources(
    employment: DataFrame,
    fomc: DataFrame,
    *,
    key_file: Path,
) -> dict[str, Any]:
    api_key = fred_helpers.load_api_key(key_file)
    employment = employment.sort_values("anchor_utc", kind="stable")
    release_dates = sorted(set(employment["anchor_utc"].dt.date.astype(str)))
    observation_start = (
        pd.to_datetime(employment["observation_period"]).min()
        - pd.DateOffset(months=1)
    ).date().isoformat()
    observation_end = pd.to_datetime(employment["observation_period"]).max().date().isoformat()

    documents: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    for series_id in ("PAYEMS", "UNRATE"):
        initial, initial_contract = fetch_fred_json(
            "series/observations",
            params={
                "series_id": series_id,
                "output_type": 4,
                "realtime_start": fred_helpers.FRED_REALTIME_START,
                "realtime_end": fred_helpers.FRED_REALTIME_END,
                "observation_start": observation_start,
                "observation_end": observation_end,
            },
            api_key=api_key,
        )
        vintage, vintage_contract = fetch_fred_json(
            "series/observations",
            params={
                "series_id": series_id,
                "output_type": 2,
                "vintage_dates": ",".join(release_dates),
                "observation_start": observation_start,
                "observation_end": observation_end,
            },
            api_key=api_key,
        )
        documents[f"{series_id}_initial"] = initial
        documents[f"{series_id}_release_vintages"] = vintage
        contracts[f"{series_id}_initial"] = initial_contract
        contracts[f"{series_id}_release_vintages"] = vintage_contract

    start = (fomc["anchor_utc"].min() - pd.Timedelta(days=2)).date().isoformat()
    end = (fomc["anchor_utc"].max() + pd.Timedelta(days=2)).date().isoformat()
    target, target_contract = fetch_fred_json(
        "series/observations",
        params={
            "series_id": "DFEDTARU",
            "observation_start": start,
            "observation_end": end,
        },
        api_key=api_key,
    )
    documents["DFEDTARU"] = target
    contracts["DFEDTARU"] = target_contract
    return {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "source": "FRED/ALFRED API using official BLS and Federal Reserve series",
        "api_key_serialized": False,
        "documents": documents,
        "request_contracts_without_api_key": contracts,
    }


def initial_release_map(document: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["date"]): str(row["realtime_start"])
        for row in document.get("observations", [])
        if row.get("value") not in (None, ".")
    }


def build_jobs_actuals(
    employment: DataFrame,
    sources: Mapping[str, Any],
) -> DataFrame:
    documents = sources["documents"]
    pay_initial = initial_release_map(documents["PAYEMS_initial"])
    unrate_initial = initial_release_map(documents["UNRATE_initial"])
    rows: list[dict[str, Any]] = []
    for row in employment.to_dict(orient="records"):
        observation = str(row["observation_period"])
        release_date = pd.Timestamp(row["anchor_utc"]).date().isoformat()
        payroll_available = pay_initial.get(observation) == release_date
        unemployment_available = unrate_initial.get(observation) == release_date
        if not payroll_available and not unemployment_available:
            raise ValueError(f"No official jobs actual is available for {observation}")

        payroll_change = None
        if payroll_available:
            previous = fred_helpers._shift_month(observation, 1)
            current_payroll = fred_helpers._vintage_value(
                documents["PAYEMS_release_vintages"],
                series_id="PAYEMS",
                observation_date=observation,
                vintage_date=release_date,
            )
            previous_payroll = fred_helpers._vintage_value(
                documents["PAYEMS_release_vintages"],
                series_id="PAYEMS",
                observation_date=previous,
                vintage_date=release_date,
            )
            payroll_change = (current_payroll - previous_payroll) * 1_000

        unemployment = None
        if unemployment_available:
            unemployment = fred_helpers._vintage_value(
                documents["UNRATE_release_vintages"],
                series_id="UNRATE",
                observation_date=observation,
                vintage_date=release_date,
            )
        rows.append(
            {
                "official_event_id": row["event_id"],
                "anchor_utc": row["anchor_utc"],
                "observation_period": observation,
                "nonfarm_payrolls_actual": payroll_change,
                "unemployment_rate_actual": unemployment,
                "actual_vintage_date": release_date,
            }
        )
    return DataFrame.from_records(rows)


def fred_values(document: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["date"]): float(row["value"])
        for row in document.get("observations", [])
        if row.get("value") not in (None, ".")
    }


def fed_category(change_bps: float) -> str:
    if change_bps < -25:
        return "Cut >25bps"
    if change_bps < 0:
        return "Cut 25bps"
    if change_bps == 0:
        return "Hike 0bps"
    if change_bps <= 25:
        return "Hike 25bps"
    return "Hike >25bps"


def build_fed_actuals(fomc: DataFrame, sources: Mapping[str, Any]) -> DataFrame:
    values = fred_values(sources["documents"]["DFEDTARU"])
    rows: list[dict[str, Any]] = []
    for row in fomc.to_dict(orient="records"):
        decision_date = pd.Timestamp(row["anchor_utc"]).date()
        current_key = decision_date.isoformat()
        next_key = (decision_date + timedelta(days=1)).isoformat()
        if current_key not in values or next_key not in values:
            raise ValueError(f"Missing DFEDTARU decision values for {current_key}")
        change_bps = (values[next_key] - values[current_key]) * 100
        rows.append(
            {
                "official_event_id": row["event_id"],
                "anchor_utc": row["anchor_utc"],
                "policy_change_bps": change_bps,
                "federal_reserve_decision_actual": fed_category(change_bps),
            }
        )
    return DataFrame.from_records(rows)


def sign(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    if float(numeric) == 0:
        return 0
    return 1 if float(numeric) > 0 else -1


def strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"Expected an explicit boolean, received {value!r}")


def component_semantics(component: str) -> tuple[str, str]:
    if component in {"headline_cpi", "core_cpi"}:
        return (
            "higher_than_expected_is_provisionally_hawkish_but_context_can_override",
            "inverse_of_surprise",
        )
    if component in {"nonfarm_payrolls", "unemployment_rate"}:
        return (
            "dual_growth_and_rate_channels_no_universal_crypto_sign",
            "not_preassigned",
        )
    if component == "federal_reserve_decision":
        return (
            "more_hawkish_than_top_expected_category_is_provisionally_negative_but_guidance_is_missing",
            "inverse_of_ordered_category_surprise",
        )
    return ("source_not_cross_period_ready", "not_preassigned")


def build_joined_catalog(
    expectations: DataFrame,
    summary: DataFrame,
    cpi: DataFrame,
    employment: DataFrame,
    jobs_actuals: DataFrame,
    fomc: DataFrame,
    fed_actuals: DataFrame,
) -> DataFrame:
    readiness = summary.set_index("component")["source_readiness"].to_dict()
    jobs_by_id = jobs_actuals.set_index("official_event_id").to_dict(orient="index")
    fed_by_id = fed_actuals.set_index("official_event_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for source_row in expectations.to_dict(orient="records"):
        component = str(source_row["component"])
        close_time = pd.Timestamp(source_row["market_close_time_utc"])
        reference_period = None
        if component in {
            "headline_cpi",
            "core_cpi",
            "nonfarm_payrolls",
            "unemployment_rate",
        }:
            reference_period = reference_period_from_event_ticker(
                str(source_row["event_ticker"])
            )
        official: Mapping[str, Any] | None = None
        actual_value: float | None = None
        actual_category: str | None = None
        actual_units: str | None = None
        official_id: str | None = None
        policy_change_bps: float | None = None
        actual_status = "official_actual_not_matched"

        if component in {"headline_cpi", "core_cpi"}:
            official = match_following_anchor(
                close_time,
                cpi,
                reference_period=reference_period,
                period_column="observation_month",
            )
            if official is not None:
                official_id = str(official["event_id"])
                column = (
                    "headline_mom_released_pct"
                    if component == "headline_cpi"
                    else "core_mom_released_pct"
                )
                actual_value = float(official[column])
                actual_units = "percentage_points_month_on_month"
                actual_status = "official_initial_release_joined"
        elif component in {"nonfarm_payrolls", "unemployment_rate"}:
            official = match_following_anchor(
                close_time,
                employment,
                reference_period=reference_period,
                period_column="observation_period",
            )
            if official is not None:
                official_id = str(official["event_id"])
                actual = jobs_by_id[official_id]
                column = (
                    "nonfarm_payrolls_actual"
                    if component == "nonfarm_payrolls"
                    else "unemployment_rate_actual"
                )
                if pd.notna(actual[column]):
                    actual_value = float(actual[column])
                    actual_units = (
                        "jobs"
                        if component == "nonfarm_payrolls"
                        else "percentage_points"
                    )
                    actual_status = "official_release_vintage_joined"
                else:
                    actual_status = "official_component_actual_unavailable"
        elif component == "federal_reserve_decision":
            official = match_following_anchor(close_time, fomc)
            if official is not None:
                official_id = str(official["event_id"])
                actual = fed_by_id[official_id]
                actual_category = str(actual["federal_reserve_decision_actual"])
                policy_change_bps = float(actual["policy_change_bps"])
                actual_value = float(FED_CATEGORY_ORDER[actual_category])
                actual_units = "ordered_policy_decision_category"
                actual_status = "official_target_range_change_joined"

        expectation_value = pd.to_numeric(source_row.get("expectation_value"), errors="coerce")
        if component == "federal_reserve_decision":
            expected_category = str(source_row.get("expectation_value") or "")
            expectation_value = FED_CATEGORY_ORDER.get(expected_category)
        surprise = (
            float(actual_value) - float(expectation_value)
            if actual_value is not None and pd.notna(expectation_value)
            else None
        )
        semantics, prediction_rule = component_semantics(component)
        eligible = bool(
            readiness.get(component) == SOURCE_READY
            and strict_bool(source_row["usable_2h"])
            and actual_status.endswith("joined")
        )
        provisional_direction = None
        if eligible and prediction_rule.startswith("inverse"):
            surprise_sign = sign(surprise)
            provisional_direction = (
                -surprise_sign if surprise_sign not in (None, 0) else surprise_sign
            )

        rows.append(
            {
                **source_row,
                "source_readiness": readiness.get(component),
                "reference_period": reference_period,
                "official_event_id": official_id,
                "official_anchor_utc": (
                    pd.Timestamp(official["anchor_utc"]).isoformat()
                    if official is not None
                    else None
                ),
                "official_actual_status": actual_status,
                "official_actual_value": actual_value,
                "official_actual_category": actual_category,
                "official_policy_change_bps": policy_change_bps,
                "actual_units": actual_units,
                "expectation_lead_minutes": (
                    (
                        pd.Timestamp(official["anchor_utc"]) - close_time
                    ).total_seconds()
                    / 60
                    if official is not None
                    else None
                ),
                "near_release_expectation": bool(
                    official is not None
                    and pd.Timestamp(official["anchor_utc"]) - close_time
                    <= NEAR_RELEASE_LEAD
                ),
                "market_implied_expectation_numeric": (
                    float(expectation_value) if pd.notna(expectation_value) else None
                ),
                "actual_minus_market_expectation": surprise,
                "surprise_sign": sign(surprise),
                "direction_semantics": semantics,
                "provisional_crypto_direction": provisional_direction,
                "direction_test_eligible": eligible,
                "signed_surprise_available": bool(eligible and sign(surprise) not in (None, 0)),
            }
        )
    return DataFrame.from_records(rows).sort_values(
        ["market_close_time_utc", "component"], kind="stable"
    )


def freeze_document(catalog: DataFrame, source: Mapping[str, Any]) -> dict[str, Any]:
    eligible = catalog[catalog["direction_test_eligible"]].copy()
    component_counts = (
        catalog.groupby("component", sort=True)
        .agg(
            expectation_rows=("event_ticker", "size"),
            usable_expectations=("usable_2h", "sum"),
            official_actuals=("official_event_id", "count"),
            later_test_eligible=("direction_test_eligible", "sum"),
            nonzero_signed_surprises=("signed_surprise_available", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "schema_version": 1,
        "status": "frozen_macro_actual_minus_expectation_before_crypto_outcomes",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "kalshi_contract_resolution_used": False,
        "plain_question": (
            "Can clean pre-release expectations be joined to the official value first "
            "published after the quote cutoff, without using later revisions?"
        ),
        "anchor_rule": (
            "The official release must be later than the market cutoff on the same date "
            "and no more than twelve hours away. Forecast age in minutes is retained; "
            "ten minutes or less is separately marked as near-release."
        ),
        "actual_value_rules": {
            "cpi": "Existing frozen ALFRED release-vintage month-on-month values.",
            "jobs": (
                "PAYEMS current minus prior month and UNRATE current month, both from the "
                "ALFRED vintage dated to that Employment Situation release. PAYEMS is "
                "converted from thousands into individual jobs before comparison."
            ),
            "fed": (
                "Change in official target-range upper bound from the decision date to "
                "the following effective date using DFEDTARU."
            ),
        },
        "interpretation_guardrails": [
            (
                "A surprise is actual minus the market-implied expectation, not good "
                "news minus bad news."
            ),
            "Headline and core CPI are one release episode, not independent events.",
            "Payroll and unemployment are one jobs-report episode, not independent events.",
            (
                "Jobs surprises retain competing growth and rate channels; no universal "
                "crypto sign is assigned."
            ),
            (
                "An as-expected Fed decision does not mean no news because statement "
                "guidance and the press conference remain unmeasured."
            ),
            (
                "Core PCE and GDP remain visible but are not direction-test eligible "
                "because their expectation sources did not pass both periods."
            ),
            (
                "A forecast captured hours before a release is valid but older; it must "
                "not be treated as equivalent to one captured within ten minutes."
            ),
        ],
        "eligible_rows": len(eligible),
        "eligible_whole_episodes": int(eligible["expectation_episode_id"].nunique()),
        "component_counts": component_counts,
        "source_contracts": source["request_contracts_without_api_key"],
        "parents": {
            "expectation_events": artifact(EXPECTATION_EVENTS_PATH),
            "expectation_summary": artifact(EXPECTATION_SUMMARY_PATH),
            "cpi_catalog": artifact(CPI_CATALOG_PATH),
            "employment_catalog": artifact(BROAD_CATALOG_PATH),
            "fomc_catalog": artifact(FOMC_CATALOG_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def execute(*, key_file: Path, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_macro_expectation_actual_freeze":
            raise ValueError("Existing expectation-actual freeze is invalid")
        for value in result["artifacts"].values():
            path = Path(value["path"])
            if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
                raise ValueError(f"Existing actual-freeze artifact changed: {path}")
        return result

    expectations, summary, cpi, employment, fomc = load_inputs()
    employment = restrict_employment_to_expectations(expectations, employment)
    sources = fetch_fred_actual_sources(employment, fomc, key_file=key_file)
    jobs_actuals = build_jobs_actuals(employment, sources)
    fed_actuals = build_fed_actuals(fomc, sources)
    catalog = build_joined_catalog(
        expectations,
        summary,
        cpi,
        employment,
        jobs_actuals,
        fomc,
        fed_actuals,
    )
    freeze = freeze_document(catalog, sources)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalog, CATALOG_PATH)
    g0.atomic_write_json(sources, SOURCE_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_macro_expectation_actual_freeze",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "eligible_rows": freeze["eligible_rows"],
        "eligible_whole_episodes": freeze["eligible_whole_episodes"],
        "artifacts": {
            "catalog": artifact(CATALOG_PATH),
            "source": artifact(SOURCE_PATH),
            "freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "crypto_outcomes_will_be_read": False,
                    "key_file": str(args.key_file),
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
