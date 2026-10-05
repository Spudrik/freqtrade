"""Freeze a broad cross-market shock screen before reading crypto outcomes."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
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
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "cross_asset_relevance_20260904a"
)
DEFAULT_KEY_FILE = Path(r"C:\Users\engin\OneDrive\Desktop\FREDAPI.json")
INFORMATION_CUTOFF_UTC = layer1.INFORMATION_CUTOFF_UTC

MARKET_FAMILIES: dict[str, dict[str, Any]] = {
    "technology_equities": {
        "series_id": "NASDAQCOM",
        "label": "US technology shares",
        "change": "percent",
        "frequency": "daily",
        "expected_crypto_relation": "same_direction_later_test_only",
    },
    "market_fear": {
        "series_id": "VIXCLS",
        "label": "US market fear",
        "change": "percent",
        "frequency": "daily",
        "expected_crypto_relation": "opposite_direction_later_test_only",
    },
    "ten_year_yield": {
        "series_id": "DGS10",
        "label": "US ten-year bond yield",
        "change": "difference",
        "frequency": "daily",
        "expected_crypto_relation": "not_assumed",
    },
    "broad_us_dollar": {
        "series_id": "DTWEXBGS",
        "label": "Broad US dollar",
        "change": "percent",
        "frequency": "daily",
        "expected_crypto_relation": "opposite_direction_later_test_only",
    },
    "crude_oil": {
        "series_id": "DCOILWTICO",
        "label": "Crude oil",
        "change": "percent",
        "frequency": "daily",
        "expected_crypto_relation": "not_assumed",
    },
    "yield_curve": {
        "series_id": "T10Y2Y",
        "label": "US yield curve",
        "change": "difference",
        "frequency": "daily",
        "expected_crypto_relation": "not_assumed",
    },
    "financial_conditions": {
        "series_id": "NFCI",
        "label": "US financial conditions",
        "change": "difference",
        "frequency": "weekly",
        "expected_crypto_relation": "opposite_direction_later_test_only",
    },
}

HORIZONS_HOURS = (4, 8, 24, 72)
SHOCK_QUANTILE = 0.95
DAILY_HISTORY = 252
DAILY_MINIMUM_HISTORY = 126
WEEKLY_HISTORY = 104
WEEKLY_MINIMUM_HISTORY = 52
DAILY_COOLDOWN_DAYS = 7
WEEKLY_COOLDOWN_DAYS = 28
MINIMUM_PARTITION_EVENTS = 10
MINIMUM_ABOVE_CONTROL_RATE = 0.55
MINIMUM_MEDIAN_ACTIVITY_SCORE = 1.20


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def fetch_sources(
    api_key: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    documents: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    cutoff_date = pd.Timestamp(INFORMATION_CUTOFF_UTC).date().isoformat()
    for definition in MARKET_FAMILIES.values():
        series_id = str(definition["series_id"])
        document, contract = fred_helpers.fetch_json(
            "series/observations",
            params={
                "series_id": series_id,
                "output_type": 4,
                "realtime_start": "2020-01-01",
                "realtime_end": cutoff_date,
                "observation_start": "2020-01-01",
            },
            api_key=api_key,
        )
        documents[series_id] = document
        contracts[series_id] = contract
    return documents, contracts


def conservative_anchor(realtime_start: str) -> pd.Timestamp:
    """Use midnight after the reported availability date, never the observation date."""

    return pd.Timestamp(realtime_start, tz="UTC") + pd.Timedelta(days=1)


def series_frame(
    document: Mapping[str, Any], definition: Mapping[str, Any]
) -> DataFrame:
    rows = DataFrame.from_records(document.get("observations", []))
    if rows.empty:
        raise ValueError(f"No rows returned for {definition['series_id']}.")
    rows["observation_date"] = pd.to_datetime(rows["date"], utc=True)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["anchor_utc"] = rows["realtime_start"].map(conservative_anchor)
    rows = rows.dropna(subset=["value"]).sort_values("observation_date", kind="stable")
    if definition["change"] == "percent":
        rows["source_change"] = rows["value"].pct_change()
        change_unit = "fractional_change"
    else:
        rows["source_change"] = rows["value"].diff()
        change_unit = "index_or_percentage_point_change"
    history = DAILY_HISTORY if definition["frequency"] == "daily" else WEEKLY_HISTORY
    minimum = (
        DAILY_MINIMUM_HISTORY
        if definition["frequency"] == "daily"
        else WEEKLY_MINIMUM_HISTORY
    )
    rows["shock_threshold"] = (
        rows["source_change"]
        .abs()
        .shift(1)
        .rolling(history, min_periods=minimum)
        .quantile(SHOCK_QUANTILE)
    )
    rows["change_unit"] = change_unit
    return rows


def build_event_catalog(
    documents: Mapping[str, Mapping[str, Any]],
) -> DataFrame:
    records: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(INFORMATION_CUTOFF_UTC)
    for family, definition in MARKET_FAMILIES.items():
        frame = series_frame(documents[str(definition["series_id"])], definition)
        candidates = frame.loc[
            frame["anchor_utc"].ge(pd.Timestamp("2021-01-01T00:00:00Z"))
            & frame["anchor_utc"].lt(cutoff)
            & frame["source_change"].abs().gt(frame["shock_threshold"])
        ]
        cooldown = pd.Timedelta(
            days=(
                DAILY_COOLDOWN_DAYS
                if definition["frequency"] == "daily"
                else WEEKLY_COOLDOWN_DAYS
            )
        )
        last_anchor: pd.Timestamp | None = None
        for row in candidates.itertuples(index=False):
            anchor = pd.Timestamp(row.anchor_utc)
            if last_anchor is not None and anchor - last_anchor < cooldown:
                continue
            records.append(
                {
                    "event_id": f"{family}_{anchor.strftime('%Y_%m_%d')}",
                    "event_family": family,
                    "event_label": definition["label"],
                    "series_id": definition["series_id"],
                    "anchor_utc": anchor,
                    "observation_date": row.observation_date,
                    "whole_event_partition": layer1.whole_event_partition(anchor),
                    "source_change": float(row.source_change),
                    "absolute_source_change": abs(float(row.source_change)),
                    "shock_threshold": float(row.shock_threshold),
                    "change_unit": row.change_unit,
                    "expected_crypto_relation": definition[
                        "expected_crypto_relation"
                    ],
                    "source_semantics": (
                        "Initial FRED/ALFRED value, selected by an extreme change using "
                        "only earlier source observations. Crypto outcomes were not used."
                    ),
                }
            )
            last_anchor = anchor
    catalog = DataFrame.from_records(records)
    if catalog.empty:
        raise ValueError("No cross-market shock events were selected.")
    if catalog["event_id"].duplicated().any():
        raise ValueError("Cross-market event identifiers are not unique.")
    return catalog.sort_values(["anchor_utc", "event_family"], kind="stable")


def event_counts(catalog: DataFrame) -> DataFrame:
    return (
        catalog.groupby(
            ["event_family", "event_label", "whole_event_partition"], dropna=False
        )
        .size()
        .rename("events")
        .reset_index()
        .sort_values(["event_family", "whole_event_partition"])
    )


def build_freeze(
    catalog: DataFrame,
    counts: DataFrame,
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_cross_asset_relevance_before_crypto_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "plain_question": (
            "Do unusually large changes across several wider markets precede unusually "
            "large Bitcoin and Ethereum activity?"
        ),
        "families": list(MARKET_FAMILIES),
        "horizons_hours": list(HORIZONS_HOURS),
        "selection": {
            "shock_quantile": SHOCK_QUANTILE,
            "daily_history_observations": DAILY_HISTORY,
            "weekly_history_observations": WEEKLY_HISTORY,
            "anchor_rule": (
                "Midnight UTC after the FRED realtime_start date, so the source value "
                "is treated as known conservatively."
            ),
            "cooldown": "7 days for daily series; 28 days for weekly series",
        },
        "screening_rule": {
            "minimum_events_per_development_and_validation_partition": (
                MINIMUM_PARTITION_EVENTS
            ),
            "minimum_fraction_above_control": MINIMUM_ABOVE_CONTROL_RATE,
            "minimum_median_activity_multiple": MINIMUM_MEDIAN_ACTIVITY_SCORE,
            "lead_definition": (
                "At least one horizon passes in both chronological partitions for BTC "
                "and ETH; one-market results remain narrow leads only."
            ),
        },
        "limits": [
            "Activity only; signed direction is stored but not tested in this batch.",
            "Association does not establish that the wider market caused crypto activity.",
            "All seven families finish before any branch is considered.",
        ],
        "event_counts": counts.to_dict(orient="records"),
        "source_contracts": {
            "requests": contracts,
            "api_key": "local_secret_used_but_not_serialized",
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def render_report(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Cross-Asset Relevance Freeze",
            "",
            f"- Status: `{freeze['status']}`",
            "- Crypto outcomes read: **No**",
            "- Wider-market families: `7`",
            "- Direction tested: **No**",
            "",
            "The complete seven-family batch must finish before any individual lead is followed.",
            "",
        ]
    )


def execute(*, key_file: Path, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "cross_asset_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_cross_asset_relevance_freeze":
            raise ValueError("Existing cross-asset freeze is not terminal.")
        return result

    api_key = fred_helpers.load_api_key(key_file)
    sources, contracts = fetch_sources(api_key)
    catalog = build_event_catalog(sources)
    counts = event_counts(catalog)
    freeze = build_freeze(catalog, counts, contracts)
    if api_key in json.dumps(freeze, sort_keys=True):
        raise ValueError("Secret API key would be serialized; refusing to write.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    catalog_path = OUTPUT_ROOT / "cross_asset_event_catalog.csv"
    counts_path = OUTPUT_ROOT / "cross_asset_event_counts.csv"
    sources_path = OUTPUT_ROOT / "fred_cross_asset_sources.json"
    freeze_path = OUTPUT_ROOT / "cross_asset_freeze.json"
    report_path = OUTPUT_ROOT / "cross_asset_freeze_report.md"
    g0.atomic_write_csv(catalog, catalog_path)
    g0.atomic_write_csv(counts, counts_path)
    g0.atomic_write_json(sources, sources_path)
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_cross_asset_relevance_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "event_count": len(catalog),
        "family_count": int(catalog["event_family"].nunique()),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "catalog": artifact(catalog_path),
            "counts": artifact(counts_path),
            "sources": artifact(sources_path),
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
                    "families": len(MARKET_FAMILIES),
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
