"""Freeze five scheduled-macro direction baselines before reading signed returns.

This deliberately simple rate-pressure hypothesis maps an acceleration in the
published measure versus its previous publication to negative short crypto direction,
and a deceleration to positive direction.  It is not a consensus surprise and is not
assumed to be the correct economic channel in every background regime.
"""

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
    market_event_broad_relevance_freeze as broad,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
PARENT_ROOT = broad.OUTPUT_ROOT
PARENT_RESULT_PATH = PARENT_ROOT / "broad_relevance_freeze_result.json"
PARENT_FREEZE_PATH = PARENT_ROOT / "broad_relevance_freeze.json"
PARENT_CATALOG_PATH = PARENT_ROOT / "broad_relevance_event_catalog.csv"
PARENT_SOURCES_PATH = PARENT_ROOT / "fred_initial_release_sources.json"
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "scheduled_macro_direction_20260905a"
)
CATALOG_PATH = OUTPUT_ROOT / "scheduled_macro_direction_catalog.csv"
COUNTS_PATH = OUTPUT_ROOT / "scheduled_macro_direction_counts.csv"
FREEZE_PATH = OUTPUT_ROOT / "scheduled_macro_direction_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "scheduled_macro_direction_freeze_report.md"
RESULT_PATH = OUTPUT_ROOT / "scheduled_macro_direction_freeze_result.json"

EVALUATION_PARTITIONS = (
    "development_2021_2023",
    "internal_validation_2024_2025",
)
HORIZONS_MINUTES = (5, 15, 30, 60)
ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
MINIMUM_EVENTS_PER_PARTITION = 6
MINIMUM_OVERALL_ACCURACY = 0.55
MINIMUM_PARTITION_ACCURACY = 0.50
MINIMUM_CONTROL_LIFT = 0.03
CONTROL_WEEKS = 26
CONTROL_COUNT = 12
MINIMUM_CONTROLS = 10
CONTROL_EVENT_EXCLUSION_HOURS = 24

SERIES_METHODS: dict[str, str] = {
    "PPIACO": "published_index_monthly_rate_acceleration",
    "PAYEMS": "published_payroll_monthly_change_acceleration",
    "PCEPI": "published_index_monthly_rate_acceleration",
    "RSAFS": "published_level_monthly_rate_acceleration",
    "GDPC1": "published_level_quarterly_rate_acceleration",
}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def sign(value: Any) -> int:
    if value is None or pd.isna(value) or float(value) == 0.0:
        return 0
    return 1 if float(value) > 0 else -1


def load_parent_inputs() -> tuple[DataFrame, Mapping[str, Any], dict[str, Any]]:
    result = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(PARENT_FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_broad_event_relevance_freeze":
        raise ValueError("Parent broad event freeze is not terminal")
    if freeze.get("status") != "frozen_broad_event_relevance_before_market_outcomes":
        raise ValueError("Parent broad event definitions are not frozen")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Parent source freeze unexpectedly opened market outcomes")
    for key, path in (
        ("freeze", PARENT_FREEZE_PATH),
        ("catalog", PARENT_CATALOG_PATH),
        ("fred_sources", PARENT_SOURCES_PATH),
    ):
        if result["artifacts"][key]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Parent artifact changed: {path}")
    catalog = pd.read_csv(PARENT_CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    catalog = catalog.loc[
        catalog["event_group"].eq("scheduled_release")
        & catalog["whole_event_partition"].isin(EVALUATION_PARTITIONS)
    ].copy()
    sources = json.loads(PARENT_SOURCES_PATH.read_text(encoding="utf-8"))
    return catalog, sources, result


def transformed_source_rows(
    source_document: Mapping[str, Any], series_id: str
) -> DataFrame:
    frame = DataFrame.from_records(source_document.get("observations", []))
    required = {"date", "realtime_start", "value"}
    if not required.issubset(frame):
        raise ValueError(f"{series_id} source is missing fields: {required - set(frame)}")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"{series_id} has duplicate initial observation periods")
    frame["published_value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["published_value"].isna().any():
        raise ValueError(f"{series_id} contains a missing published value")
    if series_id == "PAYEMS":
        frame["published_change"] = frame["published_value"].diff()
    else:
        frame["published_change"] = frame["published_value"].pct_change(
            fill_method=None
        )
    frame["published_acceleration"] = frame["published_change"].diff()
    frame["acceleration_sign"] = frame["published_acceleration"].map(sign)
    frame["predicted_crypto_direction"] = -frame["acceleration_sign"]
    frame["previous_release_predicted_direction"] = frame[
        "predicted_crypto_direction"
    ].shift(1)
    frame["release_date_from_source"] = frame["realtime_start"].astype(str)
    return frame[
        [
            "date",
            "release_date_from_source",
            "published_value",
            "published_change",
            "published_acceleration",
            "acceleration_sign",
            "predicted_crypto_direction",
            "previous_release_predicted_direction",
        ]
    ].rename(columns={"date": "observation_period"})


def build_direction_catalog(
    scheduled: DataFrame, sources: Mapping[str, Mapping[str, Any]]
) -> DataFrame:
    pieces: list[DataFrame] = []
    for series_id, method in SERIES_METHODS.items():
        events = scheduled.loc[scheduled["source_column"].eq(series_id)].copy()
        if events.empty:
            raise ValueError(f"No scheduled events for frozen series {series_id}")
        transformed = transformed_source_rows(sources[series_id], series_id)
        merged = events.merge(
            transformed,
            on="observation_period",
            how="left",
            validate="one_to_one",
        )
        if merged["published_acceleration"].isna().any():
            raise ValueError(f"Could not reconstruct every {series_id} acceleration")
        if not merged.apply(
            lambda row: str(row.release_date_from_source)
            == pd.Timestamp(row.anchor_utc).tz_convert("America/New_York").date().isoformat(),
            axis=1,
        ).all():
            raise ValueError(f"{series_id} release dates disagree with frozen anchors")
        merged["source_series_id"] = series_id
        merged["published_change_method"] = method
        pieces.append(merged)
    catalog = pd.concat(pieces, ignore_index=True).sort_values(
        ["anchor_utc", "event_family"], kind="stable"
    )
    catalog["release_cluster_id"] = catalog["anchor_utc"].map(
        lambda value: "scheduled_release_" + pd.Timestamp(value).strftime("%Y%m%dT%H%M%SZ")
    )
    catalog["direction_semantics"] = (
        "Provisional inverse rate-pressure baseline from acceleration versus the "
        "previous published measure; not actual minus consensus and not universal."
    )
    if catalog["event_id"].duplicated().any():
        raise ValueError("Scheduled direction event IDs are not unique")
    if not catalog["predicted_crypto_direction"].isin([-1, 1]).all():
        raise ValueError("Every scored release requires a non-zero frozen direction")
    return catalog.reset_index(drop=True)


def counts_table(catalog: DataFrame) -> DataFrame:
    return (
        catalog.groupby(
            ["event_family", "source_series_id", "whole_event_partition"],
            sort=True,
        )
        .agg(
            events=("event_id", "nunique"),
            positive_acceleration=("acceleration_sign", lambda values: int((values > 0).sum())),
            negative_acceleration=("acceleration_sign", lambda values: int((values < 0).sum())),
        )
        .reset_index()
    )


def freeze_document(
    catalog: DataFrame, counts: DataFrame, parent_result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_scheduled_macro_direction_before_signed_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "plain_question": (
            "When a published inflation growth jobs or spending measure accelerates "
            "versus its previous publication does short crypto direction more often "
            "follow the inverse rate-pressure interpretation?"
        ),
        "critical_limit": (
            "This is change versus the previous publication, not actual minus the "
            "market's timestamped forecast. Growth, jobs, and spending may have an "
            "opposite risk-on channel in some backgrounds."
        ),
        "families": list(broad.SCHEDULED_FAMILIES),
        "series_methods": SERIES_METHODS,
        "partitions": list(EVALUATION_PARTITIONS),
        "assets": list(ASSETS),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "direction_rule": "acceleration maps to down; deceleration maps to up",
        "controls": [
            "development-period majority direction",
            "causal pre-event return over the same duration",
            "previous release's frozen predicted direction",
            "story-sign rotation within family and partition",
            "same sign at matched prior same-weekday-and-clock ordinary periods",
        ],
        "retention_rule": {
            "minimum_events_per_partition": MINIMUM_EVENTS_PER_PARTITION,
            "minimum_overall_accuracy": MINIMUM_OVERALL_ACCURACY,
            "minimum_each_partition_accuracy": MINIMUM_PARTITION_ACCURACY,
            "minimum_lift_over_strongest_control": MINIMUM_CONTROL_LIFT,
            "family_result": (
                "Both BTC and ETH at one frozen horizon is a family lead; one asset "
                "is narrow. Every pass remains provisional pending a whole-event "
                "family-wide search check."
            ),
        },
        "control_selection": {
            "search_weeks": CONTROL_WEEKS,
            "controls_per_event": CONTROL_COUNT,
            "minimum_controls": MINIMUM_CONTROLS,
            "exclude_within_hours_of_any_scheduled_event": CONTROL_EVENT_EXCLUSION_HOURS,
        },
        "event_rows": len(catalog),
        "release_clocks": int(catalog["release_cluster_id"].nunique()),
        "counts": counts.to_dict(orient="records"),
        "parent_contracts": {
            "parent_result": parent_result,
            "parent_catalog": artifact(PARENT_CATALOG_PATH),
            "parent_sources": artifact(PARENT_SOURCES_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def report_text(freeze: Mapping[str, Any]) -> str:
    lines = [
        "# Scheduled Macro Direction Freeze",
        "",
        f"- Status: `{freeze['status']}`",
        f"- Frozen releases: `{freeze['event_rows']}`",
        f"- Distinct release clocks: `{freeze['release_clocks']}`",
        "- Families: producer prices, employment, consumer spending/prices, retail sales, GDP",
        "- Market outcomes read: **No**",
        "- Profit used: **No**",
        "",
        "The direction is an intentionally simple inverse rate-pressure baseline. It is "
        "not a market-consensus surprise, and a failure does not invalidate the already "
        "retained event-activity clocks.",
        "",
    ]
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_scheduled_macro_direction_freeze":
            raise ValueError("Existing scheduled macro direction freeze is not terminal")
        return result
    scheduled, sources, parent_result = load_parent_inputs()
    catalog = build_direction_catalog(scheduled, sources)
    counts = counts_table(catalog)
    if (counts["events"] < MINIMUM_EVENTS_PER_PARTITION).any():
        raise ValueError("At least one family lacks the frozen per-partition event minimum")
    freeze = freeze_document(catalog, counts, parent_result)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalog, CATALOG_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_scheduled_macro_direction_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "event_rows": len(catalog),
        "release_clocks": int(catalog["release_cluster_id"].nunique()),
        "artifacts": {
            "catalog": artifact(CATALOG_PATH),
            "counts": artifact(COUNTS_PATH),
            "freeze": artifact(FREEZE_PATH),
            "report": artifact(REPORT_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "families": len(SERIES_METHODS),
                    "outcomes_will_be_read": False,
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
