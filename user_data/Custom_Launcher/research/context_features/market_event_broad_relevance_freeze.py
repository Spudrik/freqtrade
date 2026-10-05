"""Freeze a breadth-first market-event relevance screen before reading prices.

The batch deliberately asks only whether several different event families are
followed by unusual BTC/ETH activity.  It does not infer direction, surprise,
causation, or trading value.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as fred_helpers,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
USER_DATA_DIR = REPO_ROOT / "user_data"
CONTEXT_SNAPSHOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "confluence_cache"
    / "trader_confluence_1h_latest.parquet"
)
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "broad_event_relevance_20260904a"
)
DEFAULT_KEY_FILE = Path(r"C:\Users\engin\OneDrive\Desktop\FREDAPI.json")
INFORMATION_CUTOFF_UTC = layer1.INFORMATION_CUTOFF_UTC
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
RELEASE_TIME_LOCAL = time(8, 30)

SCHEDULED_FAMILIES: dict[str, dict[str, str]] = {
    "producer_prices": {
        "series_id": "PPIACO",
        "label": "Producer prices",
        "release": "Producer Price Index",
    },
    "employment": {
        "series_id": "PAYEMS",
        "label": "Employment and jobs",
        "release": "Employment Situation",
    },
    "consumer_spending_prices": {
        "series_id": "PCEPI",
        "label": "Consumer spending and prices",
        "release": "Personal Income and Outlays",
    },
    "retail_sales": {
        "series_id": "RSAFS",
        "label": "Retail sales",
        "release": "Advance Monthly Sales for Retail and Food Services",
    },
    "economic_growth": {
        "series_id": "GDPC1",
        "label": "Economic growth",
        "release": "Gross Domestic Product",
    },
}

TOPIC_FAMILIES: dict[str, dict[str, str]] = {
    "geopolitical_conflict": {
        "column": "ctx_gdelt_conflict_event_count_24h",
        "label": "Geopolitical conflict coverage",
    },
    "trade_and_sanctions": {
        "column": "ctx_gdelt_sanctions_trade_url_count_24h",
        "label": "Trade and sanctions coverage",
    },
    "oil_and_energy": {
        "column": "ctx_gdelt_oil_energy_url_count_24h",
        "label": "Oil and energy coverage",
    },
    "banking_and_credit": {
        "column": "ctx_gdelt_banking_credit_url_count_24h",
        "label": "Banking and credit coverage",
    },
    "broad_economy": {
        "column": "ctx_gdelt_macro_url_count_24h",
        "label": "Broad economic coverage",
    },
    "crypto_specific": {
        "column": "ctx_gdelt_crypto_url_count_24h",
        "label": "Crypto-specific coverage",
    },
}

SCHEDULED_HORIZONS_MINUTES = (5, 15, 30, 60, 120, 240)
TOPIC_HORIZONS_HOURS = (1, 2, 4, 8, 24)
TOPIC_HISTORY_SAME_HOUR_OBSERVATIONS = 365
TOPIC_MINIMUM_SAME_HOUR_OBSERVATIONS = 120
TOPIC_SPIKE_QUANTILE = 0.99
TOPIC_EPISODE_COOLDOWN_HOURS = 168
MINIMUM_PARTITION_EVENTS = 3
MINIMUM_ABOVE_CONTROL_RATE = 0.55
MINIMUM_MEDIAN_ACTIVITY_SCORE = 1.20


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def release_anchor(release_date: str) -> pd.Timestamp:
    local = datetime.combine(
        date.fromisoformat(release_date), RELEASE_TIME_LOCAL, EASTERN
    )
    return pd.Timestamp(local.astimezone(UTC))


def fetch_scheduled_sources(
    api_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    documents: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for definition in SCHEDULED_FAMILIES.values():
        series_id = definition["series_id"]
        document, contract = fred_helpers.fetch_json(
            "series/observations",
            params={
                "series_id": series_id,
                "output_type": 4,
                "realtime_start": fred_helpers.FRED_REALTIME_START,
                "realtime_end": fred_helpers.FRED_REALTIME_END,
                "observation_start": "2020-01-01",
            },
            api_key=api_key,
        )
        documents[series_id] = document
        contracts[series_id] = contract
    return documents, contracts


def build_scheduled_catalog(
    documents: Mapping[str, Mapping[str, Any]],
) -> DataFrame:
    records: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(INFORMATION_CUTOFF_UTC)
    for family, definition in SCHEDULED_FAMILIES.items():
        series_id = definition["series_id"]
        for row in documents[series_id].get("observations", []):
            release_date = str(row["realtime_start"])
            anchor = release_anchor(release_date)
            if anchor < pd.Timestamp("2021-01-01T00:00:00Z") or anchor >= cutoff:
                continue
            if row.get("value") in (None, "."):
                continue
            observation_date = str(row["date"])
            records.append(
                {
                    "event_id": f"{family}_{observation_date[:7].replace('-', '_')}",
                    "event_group": "scheduled_release",
                    "event_family": family,
                    "event_label": definition["label"],
                    "anchor_utc": anchor,
                    "whole_event_partition": layer1.whole_event_partition(anchor),
                    "selection_method": "complete_initial_release_family",
                    "selection_value": np.nan,
                    "selection_threshold": np.nan,
                    "source_column": series_id,
                    "source_semantics": (
                        "Official-series initial release timing only; released value, "
                        "market expectation, surprise, and direction are not used."
                    ),
                    "observation_period": observation_date,
                }
            )
    catalog = DataFrame.from_records(records)
    if catalog.empty:
        raise ValueError("No scheduled release events were reconstructed.")
    if catalog["event_id"].duplicated().any():
        raise ValueError("Scheduled release event identifiers are not unique.")
    return catalog.sort_values(["anchor_utc", "event_family"], kind="stable")


def load_topic_activity() -> DataFrame:
    columns = [
        "date",
        "ctx_max_source_available_at",
        *(definition["column"] for definition in TOPIC_FAMILIES.values()),
    ]
    frame = pd.read_parquet(CONTEXT_SNAPSHOT, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["ctx_max_source_available_at"] = pd.to_datetime(
        frame["ctx_max_source_available_at"], utc=True, errors="coerce"
    )
    return (
        frame.sort_values("date", kind="stable")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def causal_topic_threshold(values: Series, frame: DataFrame) -> Series:
    log_values = np.log1p(pd.to_numeric(values, errors="coerce").clip(lower=0))
    segment = frame["date"].diff().ne(pd.Timedelta(hours=1)).cumsum()
    utc_hour = frame["date"].dt.hour
    return log_values.groupby([segment, utc_hour]).transform(
        lambda group: group.shift(1)
        .rolling(
            TOPIC_HISTORY_SAME_HOUR_OBSERVATIONS,
            min_periods=TOPIC_MINIMUM_SAME_HOUR_OBSERVATIONS,
        )
        .quantile(TOPIC_SPIKE_QUANTILE)
    )


def select_topic_episodes(frame: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(INFORMATION_CUTOFF_UTC)
    source_safe = frame["ctx_max_source_available_at"].le(frame["date"])
    for family, definition in TOPIC_FAMILIES.items():
        column = definition["column"]
        raw = pd.to_numeric(frame[column], errors="coerce")
        log_value = np.log1p(raw.clip(lower=0))
        threshold = causal_topic_threshold(raw, frame)
        candidate_positions = frame.index[
            frame["date"].ge(pd.Timestamp("2021-01-01T00:00:00Z"))
            & frame["date"].lt(cutoff)
            & source_safe
            & raw.gt(0)
            & log_value.gt(threshold)
        ]
        last_anchor: pd.Timestamp | None = None
        for position in candidate_positions:
            anchor = pd.Timestamp(frame.at[position, "date"])
            if last_anchor is not None and anchor - last_anchor < pd.Timedelta(
                hours=TOPIC_EPISODE_COOLDOWN_HOURS
            ):
                continue
            records.append(
                {
                    "event_id": f"{family}_{anchor.strftime('%Y_%m_%d_%H')}",
                    "event_group": "gdelt_topic_activity",
                    "event_family": family,
                    "event_label": definition["label"],
                    "anchor_utc": anchor,
                    "whole_event_partition": layer1.whole_event_partition(anchor),
                    "selection_method": (
                        "causal_same_utc_hour_99th_percentile_with_7_day_cooldown"
                    ),
                    "selection_value": float(raw.at[position]),
                    "selection_threshold": float(np.expm1(threshold.at[position])),
                    "source_column": column,
                    "source_semantics": (
                        "Historical GDELT topic-activity proxy only; it does not identify "
                        "a story, severity, truth, novelty, or bullish/bearish direction."
                    ),
                    "observation_period": "",
                }
            )
            last_anchor = anchor
    catalog = DataFrame.from_records(records)
    if catalog.empty:
        raise ValueError("No GDELT topic activity episodes were selected.")
    if catalog["event_id"].duplicated().any():
        raise ValueError("Topic activity event identifiers are not unique.")
    return catalog.sort_values(["anchor_utc", "event_family"], kind="stable")


def event_count_table(catalog: DataFrame) -> DataFrame:
    return (
        catalog.groupby(
            ["event_group", "event_family", "event_label", "whole_event_partition"],
            dropna=False,
        )
        .size()
        .rename("events")
        .reset_index()
        .sort_values(["event_group", "event_family", "whole_event_partition"])
    )


def build_freeze_document(
    catalog: DataFrame,
    counts: DataFrame,
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_broad_event_relevance_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "plain_question": (
            "Are several different scheduled-release or broad-news families followed "
            "by unusually large Bitcoin and Ethereum activity?"
        ),
        "explicit_limits": [
            "This batch does not predict direction.",
            "It does not isolate a release surprise or prove that news caused a move.",
            "GDELT fields are broad activity proxies, not interpreted stories.",
            "Every family completes before any result can create a follow-up batch.",
            "CPI is parked and is not re-tested in this batch.",
        ],
        "event_families": {
            "scheduled": list(SCHEDULED_FAMILIES),
            "gdelt_topic_activity": list(TOPIC_FAMILIES),
        },
        "outcomes": {
            "assets": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            "scheduled_timeframe": "1m",
            "scheduled_horizons_minutes": list(SCHEDULED_HORIZONS_MINUTES),
            "topic_timeframe": "1h",
            "topic_horizons_hours": list(TOPIC_HORIZONS_HOURS),
            "activity_components": [
                "absolute close movement",
                "high-to-low range",
                "traded volume",
            ],
        },
        "control": (
            "Same clock time in earlier weeks, using only controls available before "
            "each event and excluding nearby selected events."
        ),
        "screening_rule": {
            "minimum_events_per_development_and_validation_partition": (
                MINIMUM_PARTITION_EVENTS
            ),
            "minimum_fraction_above_control": MINIMUM_ABOVE_CONTROL_RATE,
            "minimum_median_activity_multiple": MINIMUM_MEDIAN_ACTIVITY_SCORE,
            "lead_definition": (
                "At least one horizon passes in both development and validation for "
                "both BTC and ETH. A one-market pass is retained only as a narrower lead."
            ),
            "meaning": "Breadth-screening lead only, not proof or trading promotion.",
        },
        "topic_selection": {
            "quantile": TOPIC_SPIKE_QUANTILE,
            "same_hour_history": TOPIC_HISTORY_SAME_HOUR_OBSERVATIONS,
            "minimum_history": TOPIC_MINIMUM_SAME_HOUR_OBSERVATIONS,
            "cooldown_hours": TOPIC_EPISODE_COOLDOWN_HOURS,
            "timestamp_rule": "maximum source availability must be no later than hour",
        },
        "event_counts": counts.to_dict(orient="records"),
        "source_contracts": {
            "fred": contracts,
            "fred_api_key": "local_secret_used_but_not_serialized",
            "context_snapshot": artifact(CONTEXT_SNAPSHOT),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def render_report(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Broad Event Relevance Freeze",
            "",
            f"- Status: `{freeze['status']}`",
            "- Market outcomes read: **No**",
            "- Scheduled families: `5`",
            "- Broad news-activity families: `6`",
            "- CPI: **parked; not re-tested**",
            "",
            "The complete eleven-family batch must finish before an individual lead is followed.",
            "This first screen asks about unusual activity only, not market direction or profit.",
            "",
        ]
    )


def execute(*, key_file: Path, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "broad_relevance_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_broad_event_relevance_freeze":
            raise ValueError("Existing broad relevance freeze is not terminal.")
        return result

    api_key = fred_helpers.load_api_key(key_file)
    scheduled_sources, contracts = fetch_scheduled_sources(api_key)
    scheduled = build_scheduled_catalog(scheduled_sources)
    topics = select_topic_episodes(load_topic_activity())
    catalog = pd.concat([scheduled, topics], ignore_index=True).sort_values(
        ["anchor_utc", "event_group", "event_family"], kind="stable"
    )
    counts = event_count_table(catalog)
    freeze = build_freeze_document(catalog, counts, contracts)
    if api_key in json.dumps(freeze, sort_keys=True):
        raise ValueError("Secret API key would be serialized; refusing to write.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    catalog_path = OUTPUT_ROOT / "broad_relevance_event_catalog.csv"
    counts_path = OUTPUT_ROOT / "broad_relevance_event_counts.csv"
    source_path = OUTPUT_ROOT / "fred_initial_release_sources.json"
    freeze_path = OUTPUT_ROOT / "broad_relevance_freeze.json"
    report_path = OUTPUT_ROOT / "broad_relevance_freeze_report.md"
    g0.atomic_write_csv(catalog, catalog_path)
    g0.atomic_write_csv(counts, counts_path)
    g0.atomic_write_json(scheduled_sources, source_path)
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_broad_event_relevance_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "event_count": len(catalog),
        "family_count": int(catalog["event_family"].nunique()),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "catalog": artifact(catalog_path),
            "counts": artifact(counts_path),
            "fred_sources": artifact(source_path),
            "report": artifact(report_path),
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
                    "outcomes_will_be_read": False,
                    "families": len(SCHEDULED_FAMILIES) + len(TOPIC_FAMILIES),
                    "output_root": str(OUTPUT_ROOT),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(
        json.dumps(
            execute(key_file=args.key_file, overwrite=args.overwrite), indent=2
        )
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
