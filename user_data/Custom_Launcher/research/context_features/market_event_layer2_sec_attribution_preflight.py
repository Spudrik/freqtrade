"""Audit whether the retained SEC activity association can be causally attributed."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_activity_family_validation as activity_validation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as breadth_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_validation_freeze as validation_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_sec_hyperscaler_8k_catalogue as sec_catalogue,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = validation_freeze.OUTPUT_ROOT / "sec_attribution_preflight_20260908a"
CLOCK_INVENTORY_PATH = OUTPUT_ROOT / "sec_cluster_first_public_clock_inventory.csv"
OFFICIAL_PAGE_PROBE_PATH = OUTPUT_ROOT / "sec_official_release_page_probe.csv"
MARKET_CONTROL_PATH = OUTPUT_ROOT / "sec_intraday_market_control_inventory.csv"
DECISION_PATH = OUTPUT_ROOT / "sec_attribution_decision.csv"
REPORT_PATH = OUTPUT_ROOT / "sec_attribution_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "sec_attribution_preflight_result.json"

FRED_CROSS_ASSET_PATH = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "cross_asset_relevance_20260904a/fred_cross_asset_sources.json"
)
LIVE_CAUSAL_PATH = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/source_preflight/"
    "global_context_source_preflight_20260904a_causal.parquet"
)
HISTORICAL_PERIOD_START = pd.Timestamp("2021-01-01T00:00:00Z")
HISTORICAL_PERIOD_END = pd.Timestamp("2026-01-01T00:00:00Z")

OFFICIAL_RELEASE_PAGE_PROBES: tuple[dict[str, Any], ...] = (
    {
        "company": "Microsoft",
        "release_date": "2025-01-29",
        "official_url": (
            "https://www.microsoft.com/en-us/Investor/earnings/"
            "FY-2025-Q2/press-release-webcast"
        ),
        "page_observation": (
            "Release date and later conference-call clock are visible; an exact "
            "press-release publication minute was not exposed."
        ),
    },
    {
        "company": "Alphabet",
        "release_date": "2025-02-04",
        "official_url": (
            "https://abc.xyz/investor/news/news-details/2025/"
            "Alphabet-Announces-Fourth-Quarter-2024-and-Fiscal-Year-"
            "Results-02-04-2025/default.aspx"
        ),
        "page_observation": (
            "The official release page exposes the date but no exact publication minute."
        ),
    },
    {
        "company": "Meta",
        "release_date": "2025-01-29",
        "official_url": (
            "https://investor.atmeta.com/investor-news/press-release-details/2025/"
            "Meta-Reports-Fourth-Quarter-and-Full-Year-2024-Results/"
        ),
        "page_observation": (
            "The official release page exposes the date but no exact publication minute."
        ),
    },
    {
        "company": "Amazon",
        "release_date": "2025-02-06",
        "official_url": (
            "https://ir.aboutamazon.com/news-release/news-release-details/2025/"
            "Amazon-com-Announces-Fourth-Quarter-Results/default.aspx"
        ),
        "page_observation": (
            "The official release page exposes the date but no exact publication minute."
        ),
    },
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = activity_validation.load_validation_contract()
    routes = {str(row["branch_id"]): row for row in freeze["routes"]}
    route = routes.get("sec_clock_and_equity_alternative_controls")
    if route is None:
        raise ValueError("Frozen SEC attribution route is missing")
    activity = json.loads(
        activity_validation.RESULT_PATH.read_text(encoding="utf-8")
    )
    if activity.get("status") != "completed_layer2_activity_family_validation":
        raise ValueError("SEC activity-family validation is not terminal")
    decisions = {
        str(row["branch_id"]): row for row in activity.get("decisions", [])
    }
    sec_decision = decisions.get("sec_activity_full_family_randomization")
    if sec_decision is None or sec_decision.get("verdict") != (
        "retained_after_familywide_randomization"
    ):
        raise ValueError("SEC activity family is not the retained parent association")
    return route, sec_decision


def build_sec_clock_inventory() -> DataFrame:
    filings = pd.read_csv(sec_catalogue.CATALOGUE_PATH)
    filings = filings.loc[
        filings["whole_event_partition"].isin(breadth_freeze.PARTITIONS)
    ].copy()
    filings = breadth_freeze.fixed_window_clusters(
        filings,
        anchor_column="acceptance_datetime_utc",
        prefix="sec_hyperscaler_window",
    )
    frozen_events = pd.read_csv(breadth_freeze.EVENTS_PATH)
    frozen_earnings = frozen_events.loc[
        frozen_events["event_source"].eq("sec_hyperscaler_earnings")
    ].copy()
    frozen_ids = set(frozen_earnings["base_event_id"].astype(str))

    records: list[dict[str, Any]] = []
    for cluster_id, group in filings.groupby("market_window_group_id", sort=False):
        earnings = group.loc[
            group["earnings_like_candidate"].astype(str).str.lower().eq("true")
        ]
        if earnings.empty:
            continue
        first = group.iloc[0]
        records.append(
            {
                "cluster_id": str(cluster_id),
                "whole_event_partition": str(first["whole_event_partition"]),
                "sec_cluster_anchor_utc": pd.Timestamp(first["cluster_anchor_utc"]),
                "sec_member_filings": len(group),
                "earnings_like_filings": len(earnings),
                "companies": ";".join(sorted(set(group["company_key"].astype(str)))),
                "tickers": ";".join(sorted(set(group["ticker"].astype(str)))),
                "accession_numbers": ";".join(
                    sorted(set(group["accession_number"].astype(str)))
                ),
                "independent_first_public_utc": pd.NaT,
                "first_public_clock_status": (
                    "not_available_in_validated_local_archive"
                ),
                "first_public_clock_source": "",
                "exact_clock_usable": False,
                "boundary": (
                    "SEC acceptance is exact, but is not proof of the first public "
                    "earnings-news time."
                ),
            }
        )
    output = DataFrame.from_records(records)
    if set(output["cluster_id"].astype(str)) != frozen_ids:
        raise ValueError("Reconstructed SEC earnings clusters differ from frozen events")
    if len(output) != len(frozen_earnings):
        raise ValueError("SEC clock inventory does not contain one row per frozen cluster")
    return output.sort_values("sec_cluster_anchor_utc", kind="stable").reset_index(
        drop=True
    )


def official_page_probe_inventory() -> DataFrame:
    records = []
    for probe in OFFICIAL_RELEASE_PAGE_PROBES:
        records.append(
            {
                **probe,
                "official_page_checked": True,
                "release_date_visible": True,
                "exact_publication_minute_visible": False,
                "exact_clock_usable": False,
                "scope_limit": (
                    "Representative source probe only; not a complete historical "
                    "first-public-clock archive."
                ),
            }
        )
    return DataFrame.from_records(records)


def fred_daily_record(series_id: str, document: Mapping[str, Any]) -> dict[str, Any]:
    observations = [
        row
        for row in document.get("observations", [])
        if str(row.get("value", ".")) not in {"", "."}
    ]
    dates = pd.to_datetime(
        [row["date"] for row in observations], errors="coerce", utc=True
    )
    valid_dates = dates[~dates.isna()]
    return {
        "control_source": f"FRED {series_id}",
        "local_artifact": str(FRED_CROSS_ASSET_PATH.resolve()),
        "rows": len(observations),
        "first_source_time": valid_dates.min() if len(valid_dates) else pd.NaT,
        "last_source_time": valid_dates.max() if len(valid_dates) else pd.NaT,
        "timestamp_resolution": "date_only_daily_close",
        "historical_period_overlap": bool(
            len(valid_dates)
            and valid_dates.min() < HISTORICAL_PERIOD_END
            and valid_dates.max() >= HISTORICAL_PERIOD_START
        ),
        "eligible_for_15m_control": False,
        "reason": (
            "Historical coverage exists, but daily observations cannot explain a "
            "15-minute move around an earnings release."
        ),
    }


def market_control_inventory() -> DataFrame:
    fred = json.loads(FRED_CROSS_ASSET_PATH.read_text(encoding="utf-8"))
    records = [
        fred_daily_record("NASDAQCOM", fred["NASDAQCOM"]),
        fred_daily_record("VIXCLS", fred["VIXCLS"]),
    ]

    live = pd.read_parquet(LIVE_CAUSAL_PATH)
    relevant_ids = {
        "fred_us_equity_daily",
        "stooq_us_equity_risk",
        "stooq_global_equity_risk",
    }
    live = live.loc[live["source_id"].astype(str).isin(relevant_ids)].copy()
    live["available_at"] = pd.to_datetime(live["available_at"], errors="coerce", utc=True)
    live["source_ts_parsed"] = pd.to_datetime(
        live["source_ts"], errors="coerce", utc=True
    )
    for (source_id, metric_key), group in live.groupby(
        ["source_id", "metric_key"], sort=True
    ):
        first = group["available_at"].min()
        last = group["available_at"].max()
        overlaps = bool(first < HISTORICAL_PERIOD_END and last >= HISTORICAL_PERIOD_START)
        source_times = group["source_ts_parsed"].dropna()
        records.append(
            {
                "control_source": f"live collector {source_id}/{metric_key}",
                "local_artifact": str(LIVE_CAUSAL_PATH.resolve()),
                "rows": len(group),
                "first_source_time": (
                    source_times.min() if len(source_times) else pd.NaT
                ),
                "last_source_time": (
                    source_times.max() if len(source_times) else pd.NaT
                ),
                "timestamp_resolution": (
                    "collector_history_beginning_in_2026; source cadence varies"
                ),
                "historical_period_overlap": overlaps,
                "eligible_for_15m_control": False,
                "reason": (
                    "Collection began in 2026 and has no overlap with the frozen "
                    "2021-2025 SEC events."
                ),
            }
        )
    return DataFrame.from_records(records)


def attribution_decision(
    clock_inventory: DataFrame,
    controls: DataFrame,
    route: Mapping[str, Any],
    sec_parent: Mapping[str, Any],
) -> dict[str, Any]:
    exact_clock_count = int(clock_inventory["exact_clock_usable"].fillna(False).sum())
    cluster_count = len(clock_inventory)
    clock_coverage = exact_clock_count / cluster_count if cluster_count else 0.0
    eligible_controls = int(
        controls["eligible_for_15m_control"].fillna(False).sum()
    )
    minimum_coverage = float(route["minimum_first_public_clock_coverage"])
    attribution_pass = clock_coverage >= minimum_coverage and eligible_controls > 0
    return {
        "branch_id": "sec_clock_and_equity_alternative_controls",
        "frozen_sec_earnings_clusters": cluster_count,
        "exact_independent_first_public_clocks": exact_clock_count,
        "exact_first_public_clock_coverage": clock_coverage,
        "minimum_required_clock_coverage": minimum_coverage,
        "eligible_intraday_equity_or_volatility_controls": eligible_controls,
        "attribution_gate_pass": attribution_pass,
        "parent_familywise_p_value": float(sec_parent["familywise_p_value"]),
        "association_decision": (
            "retained_event_time_association_after_familywide_randomization"
        ),
        "attribution_decision": (
            "attribution_supported"
            if attribution_pass
            else "deferred_missing_first_public_clocks_and_intraday_market_controls"
        ),
        "interpretation": (
            "Earnings-related SEC acceptance times remain associated with unusual "
            "15-minute BTC/ETH activity. Current evidence cannot tell whether the filing, "
            "an earlier press release, or the simultaneous equity-market move caused it."
        ),
    }


def render_report(decision: Mapping[str, Any]) -> str:
    percent = 100 * float(decision["exact_first_public_clock_coverage"])
    return "\n".join(
        [
            "# SEC Earnings-Time Attribution Preflight",
            "",
            "## Plain result",
            "",
            "The previously retained result still says that BTC and ETH were unusually "
            "busy near earnings-related SEC filing times. It does not yet establish what "
            "caused that activity.",
            "",
            f"- Frozen earnings clusters: {decision['frozen_sec_earnings_clusters']}.",
            f"- Independently verified first-public minutes: "
            f"{decision['exact_independent_first_public_clocks']} ({percent:.1f}%).",
            f"- Required exact-clock coverage: "
            f"{100 * float(decision['minimum_required_clock_coverage']):.0f}%.",
            f"- Eligible historical intraday Nasdaq/VIX controls: "
            f"{decision['eligible_intraday_equity_or_volatility_controls']}.",
            "",
            "Official investor pages were useful for confirming representative release "
            "dates, but did not expose a dependable exact publication minute. Local FRED "
            "Nasdaq and VIX history is daily, while the collected higher-frequency market "
            "context starts in 2026 and therefore does not overlap the 2021-2025 events.",
            "",
            "## Decision",
            "",
            "Retain the statistically unusual event-time association. Defer first-public "
            "and equity-adjusted attribution until a complete timestamped press-release "
            "archive and genuine intraday Nasdaq/VIX history are available. Daily prices "
            "must not be substituted into this 15-minute test.",
            "",
            "No direction, causation, profit, entry, exit, or trading rule was inferred.",
            "",
        ]
    )


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer2_sec_attribution_preflight":
            raise ValueError("Existing SEC attribution result is not terminal")
        return result

    route, sec_parent = load_contracts()
    clocks = build_sec_clock_inventory()
    probes = official_page_probe_inventory()
    controls = market_control_inventory()
    decision = attribution_decision(clocks, controls, route, sec_parent)
    decision_frame = DataFrame.from_records([decision])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(clocks, CLOCK_INVENTORY_PATH)
    g0.atomic_write_csv(probes, OFFICIAL_PAGE_PROBE_PATH)
    g0.atomic_write_csv(controls, MARKET_CONTROL_PATH)
    g0.atomic_write_csv(decision_frame, DECISION_PATH)
    REPORT_PATH.write_text(render_report(decision), encoding="utf-8", newline="\n")

    result = {
        "schema_version": 1,
        "status": "completed_layer2_sec_attribution_preflight",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": False,
        "new_market_outcomes_read": False,
        "branch_completed": "sec_clock_and_equity_alternative_controls",
        "decision": decision,
        "source_limits": [
            "No validated local archive of exact first-public earnings minutes.",
            "Historical local Nasdaq and VIX data are daily, not intraday.",
            "Collected higher-frequency cross-market context begins in 2026.",
        ],
        "potential_future_source": (
            "A complete timestamped press-release distributor archive may supply exact "
            "clocks, but representative secondary timestamps must not be treated as a "
            "validated complete history."
        ),
        "artifacts": {
            "clock_inventory": artifact(CLOCK_INVENTORY_PATH),
            "official_page_probe": artifact(OFFICIAL_PAGE_PROBE_PATH),
            "market_controls": artifact(MARKET_CONTROL_PATH),
            "decision": artifact(DECISION_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=g0.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
