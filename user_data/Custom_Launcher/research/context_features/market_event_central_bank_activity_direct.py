"""Test frozen central-bank clocks for repeatable BTC and ETH activity."""

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

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as shared,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_catalogue as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
SOURCE_RESULT_PATH = frozen.RESULT_PATH
FREEZE_PATH = frozen.FREEZE_PATH
CATALOG_PATH = frozen.CATALOG_PATH
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260905a"
OUTCOME_PATH = OUTPUT_ROOT / "central_bank_activity_rows.parquet"
COVERAGE_PATH = OUTPUT_ROOT / "central_bank_activity_coverage.csv"
SUMMARY_PATH = OUTPUT_ROOT / "central_bank_activity_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "central_bank_activity_route_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "central_bank_activity_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "central_bank_activity_result.json"

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
VARIANTS = ("all_scheduled_decisions", "exclude_other_major_event_within_4h")
COLLISION_HOURS = 4
EVENT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy"
)
EXTERNAL_EVENT_SOURCES = (
    {
        "catalog": EVENT_ROOT / "cpi_release_family_20260904a/cpi_release_catalog.csv",
        "result": EVENT_ROOT / "cpi_release_family_20260904a/cpi_freeze_result.json",
        "status": "completed_cpi_family_freeze",
        "artifact_key": "catalog",
    },
    {
        "catalog": (
            EVENT_ROOT
            / "layer1_honest_history_20260903a/official_fomc_event_catalog.csv"
        ),
        "result": EVENT_ROOT / "layer1_honest_history_20260903a/layer1_result.json",
        "status": "completed_event_hierarchy_layer1",
        "artifact_key": "event_catalog",
    },
    {
        "catalog": (
            EVENT_ROOT
            / "scheduled_macro_direction_20260905a/scheduled_macro_direction_catalog.csv"
        ),
        "result": (
            EVENT_ROOT
            / "scheduled_macro_direction_20260905a/"
            "scheduled_macro_direction_freeze_result.json"
        ),
        "status": "completed_scheduled_macro_direction_freeze",
        "artifact_key": "catalog",
    },
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_scheduled_central_bank_catalogue":
        raise ValueError("Central-bank source catalogue is not complete")
    if freeze.get("status") != "frozen_scheduled_central_banks_before_market_outcomes":
        raise ValueError("Central-bank market question was not frozen")
    if freeze.get("outcomes_read") or result.get("outcomes_read"):
        raise ValueError("Central-bank freeze unexpectedly read market outcomes")
    if freeze.get("profit_used") or result.get("profit_used"):
        raise ValueError("Central-bank freeze unexpectedly used profit")
    for name, path in (("catalogue", CATALOG_PATH), ("freeze", FREEZE_PATH)):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen central-bank artifact changed: {path}")
    catalogue = pd.read_csv(CATALOG_PATH)
    catalogue["available_at_utc"] = pd.to_datetime(
        catalogue["available_at_utc"], utc=True
    )
    eligible = catalogue.loc[catalogue["scheduled"].eq("scheduled")].copy()
    return freeze, eligible


def load_external_anchors() -> tuple[list[pd.Timestamp], list[dict[str, Any]]]:
    anchors: list[pd.Timestamp] = []
    artifacts: list[dict[str, Any]] = []
    for source in EXTERNAL_EVENT_SOURCES:
        path = Path(source["catalog"])
        result_path = Path(source["result"])
        if not path.is_file() or not result_path.is_file():
            raise FileNotFoundError(
                f"Required scheduled-event source or result missing: {path}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != source["status"]:
            raise ValueError(f"Scheduled-event source is not terminal: {result_path}")
        if result.get("outcomes_read") or result.get("profit_used"):
            raise ValueError(f"Scheduled-event source is not outcome-blind: {result_path}")
        recorded = result["artifacts"][str(source["artifact_key"])]["sha256"]
        if recorded != g0.sha256_file(path):
            raise ValueError(f"Frozen scheduled-event catalogue changed: {path}")
        frame = pd.read_csv(path, usecols=["anchor_utc"])
        anchors.extend(pd.to_datetime(frame["anchor_utc"], utc=True).tolist())
        artifacts.append(
            {
                "catalogue": artifact(path),
                "source_result": artifact(result_path),
                "outcomes_read": result.get("outcomes_read"),
                "profit_used_field_status": (
                    "explicit_false"
                    if result.get("profit_used") is False
                    else "legacy_absent_no_true_flag"
                ),
            }
        )
    return anchors, artifacts


def minute_window_status(frame: DataFrame, position: int, length: int) -> str:
    if position < 0 or position + length > len(frame):
        return "incomplete_future_window"
    dates = frame.iloc[position : position + length]["date"]
    expected = pd.date_range(dates.iloc[0], periods=length, freq="1min")
    if not dates.reset_index(drop=True).equals(pd.Series(expected)):
        return "timestamp_gap"
    return "usable"


def contiguous_window_metrics(
    frame: DataFrame, position: int, length: int
) -> dict[str, float] | None:
    if minute_window_status(frame, position, length) != "usable":
        return None
    return shared.window_metrics(frame, position, length)


def prior_week_controls(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[dict[str, float]]:
    controls: list[dict[str, float]] = []
    for weeks in range(1, shared.CONTROL_SEARCH_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if (
            shared.minimum_event_distance_hours(blocked_anchors, control_anchor)
            <= COLLISION_HOURS
        ):
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        metrics = contiguous_window_metrics(frame, position, horizon)
        if metrics is not None:
            controls.append(metrics)
        if len(controls) >= shared.CONTROL_COUNT:
            break
    return controls


def add_collision_flags(
    catalogue: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> DataFrame:
    output = catalogue.copy()
    central = list(output["available_at_utc"])
    flags: list[bool] = []
    for position, anchor in enumerate(central):
        other_anchors = [*central[:position], *central[position + 1 :], *external_anchors]
        flags.append(
            shared.minimum_event_distance_hours(other_anchors, pd.Timestamp(anchor))
            <= COLLISION_HOURS
        )
    output["other_major_scheduled_event_within_4h"] = flags
    return output


def extract_outcomes(
    catalogue: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    blocked_anchors = [*catalogue["available_at_utc"].tolist(), *external_anchors]
    for pair in ASSETS:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicate_count = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = shared.position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
                "duplicate_timestamps_removed": duplicate_count,
            }
        )
        for event in catalogue.itertuples(index=False):
            anchor = pd.Timestamp(event.available_at_utc)
            event_position = positions.get(anchor)
            for horizon in (5, 15, 30, 60):
                if event_position is None:
                    eligibility_status = "missing_anchor"
                    response = None
                else:
                    eligibility_status = minute_window_status(
                        frame, event_position, horizon
                    )
                    response = contiguous_window_metrics(
                        frame, event_position, horizon
                    )
                controls = (
                    prior_week_controls(
                        frame,
                        positions,
                        anchor=anchor,
                        horizon=horizon,
                        blocked_anchors=blocked_anchors,
                    )
                    if response is not None
                    else []
                )
                score, ratios = shared.activity_score(response, controls)
                if response is not None and not np.isfinite(score):
                    eligibility_status = "too_few_clean_controls"
                records.append(
                    {
                        "event_id": event.event_id,
                        "bank": event.bank,
                        "anchor_utc": anchor,
                        "whole_event_partition": event.whole_event_partition,
                        "pair": pair,
                        "horizon_minutes": horizon,
                        "control_count": len(controls),
                        "eligibility_status": eligibility_status,
                        "activity_score": score,
                        "above_control_median": (
                            bool(score > 1.0) if np.isfinite(score) else pd.NA
                        ),
                        "abs_return_ratio": ratios["abs_return"],
                        "range_ratio": ratios["range"],
                        "volume_ratio": ratios["volume"],
                        "other_major_scheduled_event_within_4h": (
                            event.other_major_scheduled_event_within_4h
                        ),
                        "timing_quality": event.timing_quality,
                    }
                )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def summarize_outcomes(outcomes: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    records: list[dict[str, Any]] = []
    minimum_events = int(freeze["market_test"]["minimum_events_per_partition"])
    minimum_score = float(freeze["market_test"]["minimum_median_activity_score"])
    minimum_rate = float(freeze["market_test"]["minimum_above_control_rate"])
    for variant in VARIANTS:
        selected = outcomes.loc[outcomes["activity_score"].notna()].copy()
        if variant == "exclude_other_major_event_within_4h":
            selected = selected.loc[
                ~selected["other_major_scheduled_event_within_4h"]
            ]
        for keys, group in selected.groupby(
            ["bank", "pair", "horizon_minutes", "whole_event_partition"],
            sort=True,
        ):
            bank, pair, horizon, partition = keys
            event_count = int(group["event_id"].nunique())
            median_score = float(group["activity_score"].median())
            above_rate = float(group["above_control_median"].mean())
            records.append(
                {
                    "variant": variant,
                    "bank": bank,
                    "pair": pair,
                    "horizon_minutes": int(horizon),
                    "whole_event_partition": partition,
                    "events": event_count,
                    "median_activity_score": median_score,
                    "above_control_rate": above_rate,
                    "median_abs_return_ratio": float(group["abs_return_ratio"].median()),
                    "median_range_ratio": float(group["range_ratio"].median()),
                    "median_volume_ratio": float(group["volume_ratio"].median()),
                    "partition_pass": (
                        event_count >= minimum_events
                        and median_score >= minimum_score
                        and above_rate >= minimum_rate
                    ),
                }
            )
    return DataFrame.from_records(records)


def classify_routes(summary: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    records: list[dict[str, Any]] = []
    minimum_events = int(freeze["market_test"]["minimum_events_per_partition"])
    required_cells = {(variant, partition) for variant in VARIANTS for partition in PARTITIONS}
    for bank in ("ECB", "BoE", "BoJ"):
        for pair in ASSETS:
            for horizon in (5, 15, 30, 60):
                route = summary.loc[
                    summary["bank"].eq(bank)
                    & summary["pair"].eq(pair)
                    & summary["horizon_minutes"].eq(horizon)
                ]
                observed = set(
                    zip(
                        route["variant"],
                        route["whole_event_partition"],
                        strict=False,
                    )
                )
                sufficient = (
                    required_cells.issubset(observed)
                    and route["events"].ge(minimum_events).all()
                )
                all_pass = bool(
                    len(route) == len(required_cells) and route["partition_pass"].all()
                )
                all_variant = route.loc[
                    route["variant"].eq("all_scheduled_decisions")
                ]
                all_variant_pass = bool(
                    len(all_variant) == len(PARTITIONS)
                    and all_variant["partition_pass"].all()
                )
                if all_pass:
                    verdict = "retained_repeatable_activity_lead"
                elif not sufficient:
                    verdict = "coverage_limited"
                elif all_variant_pass:
                    verdict = "overlap_dependent_not_retained"
                else:
                    verdict = "weak_or_inconsistent"
                records.append(
                    {
                        "bank": bank,
                        "pair": pair,
                        "horizon_minutes": horizon,
                        "verdict": verdict,
                        "required_partition_variant_cells": len(required_cells),
                        "observed_partition_variant_cells": len(observed),
                        "minimum_events_per_cell": (
                            int(route["events"].min()) if not route.empty else 0
                        ),
                        "boe_clock_limit": (
                            "standard noon time plus known documented exceptions; "
                            "exception inventory is not proven exhaustive"
                            if bank == "BoE"
                            else "not_applicable"
                        ),
                        "interpretation_limit": (
                            "Activity only; no direction, causation, profit, or trading rule."
                        ),
                    }
                )
    return DataFrame.from_records(records)


def render_report(decisions: DataFrame, summary: DataFrame, outcomes: DataFrame) -> str:
    retained = decisions.loc[
        decisions["verdict"].eq("retained_repeatable_activity_lead")
    ]
    rows = [
        "# Central-Bank Decision Activity Review",
        "",
        "Question: do individual ECB, Bank of England, or Bank of Japan decisions "
        "repeatedly make BTC or ETH unusually active during the next 5-60 minutes?",
        "",
        "| Bank | Market | Retained windows |",
        "|---|---|---|",
    ]
    for bank in ("ECB", "BoE", "BoJ"):
        for pair in ASSETS:
            matched = retained.loc[retained["bank"].eq(bank) & retained["pair"].eq(pair)]
            windows = ", ".join(
                f"{int(value)}m" for value in matched["horizon_minutes"].sort_values()
            )
            rows.append(f"| {bank} | {pair.split('/')[0]} | {windows or '-'} |")
    rows.extend(
        [
            "",
            f"- Independent scheduled decisions with usable rows: "
            f"`{outcomes['event_id'].nunique()}`",
            f"- Retained bank/market/window routes: `{len(retained)}` of `{len(decisions)}`",
            "- A route had to pass both time periods and both the all-release and "
            "no-nearby-major-event versions.",
            "- Bank of England noon clocks include known official exceptions, but the "
            "exception inventory is not proven exhaustive.",
            "- This tests unusual movement, range, and volume. It does not test up/down "
            "direction, profit, or a trading rule.",
            "- Event/window eligibility: "
            + ", ".join(
                f"{status}={count}"
                for status, count in outcomes["eligibility_status"]
                .value_counts(dropna=False)
                .items()
            ),
            "",
            "## Strongest descriptive cells",
            "",
        ]
    )
    strongest = summary.sort_values(
        ["partition_pass", "median_activity_score"], ascending=[False, False]
    ).head(12)
    rows.extend(
        "- "
        f"{row.bank} {row.pair.split('/')[0]} {int(row.horizon_minutes)}m, "
        f"{row.whole_event_partition}, {row.variant}: "
        f"{row.events} events, median activity {row.median_activity_score:.2f}x, "
        f"above ordinary weeks {row.above_control_rate:.1%}, "
        f"{'pass' if row.partition_pass else 'fail'}."
        for row in strongest.itertuples(index=False)
    )
    rows.append("")
    return "\n".join(rows)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_central_bank_activity_direct_test":
            raise ValueError("Existing central-bank activity result is invalid")
        return result
    freeze, catalogue = load_frozen_inputs()
    external_anchors, external_artifacts = load_external_anchors()
    catalogue = add_collision_flags(catalogue, external_anchors)
    outcomes, coverage = extract_outcomes(catalogue, external_anchors)
    summary = summarize_outcomes(outcomes, freeze)
    decisions = classify_routes(summary, freeze)
    report = render_report(decisions, summary, outcomes)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(outcomes, OUTCOME_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    retained = decisions.loc[
        decisions["verdict"].eq("retained_repeatable_activity_lead")
    ]
    result = {
        "schema_version": 1,
        "status": "completed_central_bank_activity_direct_test",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": False,
        "eligible_event_rows": len(catalogue),
        "usable_event_rows": int(outcomes["event_id"].nunique()),
        "retained_routes": len(retained),
        "route_count": len(decisions),
        "external_event_catalogues": external_artifacts,
        "source_inputs": {
            "source_result": artifact(SOURCE_RESULT_PATH),
            "catalogue": artifact(CATALOG_PATH),
            "freeze": artifact(FREEZE_PATH),
        },
        "collision_rule": {
            "hours": COLLISION_HOURS,
            "meaning": (
                "Exclude a central-bank release or prior-week control within four "
                "hours of another frozen central-bank, CPI, FOMC, or scheduled-US-"
                "macro release clock."
            ),
        },
        "artifacts": {
            "outcomes": artifact(OUTCOME_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "summary": artifact(SUMMARY_PATH),
            "decisions": artifact(DECISION_PATH),
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
                    "profit_will_be_used": False,
                    "direction_will_be_tested": False,
                    "assets": ASSETS,
                    "variants": VARIANTS,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
