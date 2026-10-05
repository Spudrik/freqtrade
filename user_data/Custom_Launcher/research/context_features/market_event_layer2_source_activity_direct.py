"""Run the frozen multi-source BTC/ETH activity batch without testing profit."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
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
    market_event_central_bank_activity_direct as central_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.RESULT_PATH
FREEZE_PATH = frozen.FREEZE_PATH
EVENTS_PATH = frozen.EVENTS_PATH
CONTROLS_PATH = frozen.CONTROLS_PATH
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "source_activity_direct_20260907a"
OUTCOME_PATH = OUTPUT_ROOT / "source_activity_rows.parquet"
COVERAGE_PATH = OUTPUT_ROOT / "source_activity_coverage.csv"
SUMMARY_PATH = OUTPUT_ROOT / "source_activity_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "source_activity_route_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "source_activity_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "source_activity_result.json"

VARIANTS = ("all_events", "exclude_other_frozen_event_within_4h")
PARTITIONS = frozen.PARTITIONS


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_layer2_breadth_outcome_blind_freeze":
        raise ValueError("Layer 2 breadth freeze result is not terminal")
    if freeze.get("status") != "frozen_layer2_breadth_batch_before_market_outcomes":
        raise ValueError("Layer 2 breadth definitions are not frozen")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Layer 2 breadth freeze unexpectedly opened outcomes")
    if result.get("profit_used") or freeze.get("profit_used"):
        raise ValueError("Layer 2 breadth freeze unexpectedly used profit")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("events", EVENTS_PATH),
        ("controls", CONTROLS_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen Layer 2 breadth artifact changed: {path}")
    events = pd.read_csv(EVENTS_PATH)
    controls = pd.read_csv(CONTROLS_PATH)
    for frame, columns in (
        (events, ("anchor_utc", "decision_utc")),
        (controls, ("event_decision_utc", "control_anchor_utc")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return freeze, events, controls


def response_for_horizon(
    frame: DataFrame, position: int | None, horizon: int
) -> tuple[str, dict[str, float] | None]:
    if position is None:
        return "missing_anchor", None
    status = central_activity.minute_window_status(frame, position, horizon)
    if status != "usable":
        return status, None
    return status, central_activity.contiguous_window_metrics(frame, position, horizon)


def extract_outcomes(
    freeze: Mapping[str, Any], events: DataFrame, control_map: DataFrame
) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    controls_by_event = {
        str(event_id): list(group.sort_values("control_rank")["control_anchor_utc"])
        for event_id, group in control_map.groupby("event_id", sort=False)
    }
    for pair in freeze["activity_test"]["assets"]:
        path = g0.ohlcv_path(str(pair), "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicates = int(frame["date"].duplicated().sum())
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
                "duplicate_timestamps_removed": duplicates,
            }
        )
        for event in events.itertuples(index=False):
            horizons = freeze["activity_routes"][event.event_source][
                "horizons_minutes"
            ]
            event_position = positions.get(pd.Timestamp(event.decision_utc))
            for horizon in horizons:
                status, response = response_for_horizon(
                    frame, event_position, int(horizon)
                )
                control_metrics: list[dict[str, float]] = []
                control_gaps = 0
                for anchor in controls_by_event.get(str(event.event_id), []):
                    control_status, metrics = response_for_horizon(
                        frame, positions.get(pd.Timestamp(anchor)), int(horizon)
                    )
                    if control_status == "usable" and metrics is not None:
                        control_metrics.append(metrics)
                    else:
                        control_gaps += 1
                score, ratios = (
                    shared.activity_score(response, control_metrics)
                    if response is not None
                    else (
                        np.nan,
                        {"abs_return": np.nan, "range": np.nan, "volume": np.nan},
                    )
                )
                if response is not None and not np.isfinite(score):
                    status = "too_few_clean_controls"
                records.append(
                    {
                        "event_id": event.event_id,
                        "base_event_id": event.base_event_id,
                        "event_source": event.event_source,
                        "anchor_utc": event.anchor_utc,
                        "decision_utc": event.decision_utc,
                        "decision_delay_seconds": event.decision_delay_seconds,
                        "whole_event_partition": event.whole_event_partition,
                        "pair": pair,
                        "horizon_minutes": int(horizon),
                        "control_count": len(control_metrics),
                        "control_gap_count": control_gaps,
                        "eligibility_status": status,
                        "activity_score": score,
                        "above_control_median": (
                            bool(score > 1.0) if np.isfinite(score) else pd.NA
                        ),
                        "abs_return_ratio": ratios["abs_return"],
                        "range_ratio": ratios["range"],
                        "volume_ratio": ratios["volume"],
                        "other_frozen_event_within_4h": (
                            event.other_frozen_event_within_4h
                        ),
                        "timestamp_quality": event.timestamp_quality,
                    }
                )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def summarize_outcomes(outcomes: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    minimum_score = float(
        freeze["activity_test"]["minimum_median_activity_score"]
    )
    minimum_rate = float(freeze["activity_test"]["minimum_above_control_rate"])
    records: list[dict[str, Any]] = []
    for event_source, config in freeze["activity_routes"].items():
        for variant in VARIANTS:
            selected = outcomes.loc[
                outcomes["event_source"].eq(event_source)
                & outcomes["activity_score"].notna()
            ].copy()
            if variant == "exclude_other_frozen_event_within_4h":
                selected = selected.loc[~selected["other_frozen_event_within_4h"]]
            grouped = selected.groupby(
                ["pair", "horizon_minutes", "whole_event_partition"], sort=True
            )
            for keys, group in grouped:
                pair, horizon, partition = keys
                required = int(config["minimum_events_by_partition"][partition])
                event_count = int(group["base_event_id"].nunique())
                median_score = float(group["activity_score"].median())
                above_rate = float(group["above_control_median"].mean())
                records.append(
                    {
                        "event_source": event_source,
                        "variant": variant,
                        "pair": pair,
                        "horizon_minutes": int(horizon),
                        "whole_event_partition": partition,
                        "events": event_count,
                        "required_events": required,
                        "median_activity_score": median_score,
                        "above_control_rate": above_rate,
                        "median_abs_return_ratio": float(
                            group["abs_return_ratio"].median()
                        ),
                        "median_range_ratio": float(group["range_ratio"].median()),
                        "median_volume_ratio": float(group["volume_ratio"].median()),
                        "partition_pass": (
                            event_count >= required
                            and median_score >= minimum_score
                            and above_rate >= minimum_rate
                        ),
                    }
                )
    return DataFrame.from_records(records)


def classify_routes(summary: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    records: list[dict[str, Any]] = []
    required_cells = {
        (variant, partition) for variant in VARIANTS for partition in PARTITIONS
    }
    for source, config in freeze["activity_routes"].items():
        for pair in freeze["activity_test"]["assets"]:
            for horizon in config["horizons_minutes"]:
                route = summary.loc[
                    summary["event_source"].eq(source)
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
                sufficient = required_cells.issubset(observed) and all(
                    int(row.events) >= int(row.required_events)
                    for row in route.itertuples(index=False)
                )
                complete_pass = bool(
                    len(route) == len(required_cells) and route["partition_pass"].all()
                )
                raw = route.loc[route["variant"].eq("all_events")]
                raw_pass = bool(
                    len(raw) == len(PARTITIONS) and raw["partition_pass"].all()
                )
                if complete_pass:
                    verdict = "retained_exploratory_activity_lead_pending_family_shuffle"
                elif not sufficient:
                    verdict = "coverage_limited"
                elif raw_pass:
                    verdict = "overlap_dependent_not_retained"
                else:
                    verdict = "weak_or_inconsistent"
                records.append(
                    {
                        "event_source": source,
                        "plain_name": config["plain_name"],
                        "pair": pair,
                        "horizon_minutes": int(horizon),
                        "verdict": verdict,
                        "observed_partition_variant_cells": len(observed),
                        "minimum_events_per_cell": (
                            int(route["events"].min()) if not route.empty else 0
                        ),
                        "timestamp_limit": config["timestamp_limit"],
                        "interpretation_limit": (
                            "Activity only; no direction, causation, profit, or trading rule."
                        ),
                    }
                )
    return DataFrame.from_records(records)


def render_report(decisions: DataFrame, summary: DataFrame) -> str:
    retained = decisions.loc[
        decisions["verdict"].eq(
            "retained_exploratory_activity_lead_pending_family_shuffle"
        )
    ]
    lines = [
        "# Layer 2 Multi-Source Activity Review",
        "",
        "This batch asks whether official event clocks coincide with unusual BTC or ETH "
        "movement, range, and volume. It does not test profit or infer direction.",
        "",
        f"- Source/asset/horizon routes tested: `{len(decisions)}`",
        f"- Exploratory routes surviving both periods and collision control: `{len(retained)}`",
        "",
        "## Route decisions",
        "",
    ]
    for source, group in decisions.groupby("event_source", sort=False):
        counts = group["verdict"].value_counts().to_dict()
        lines.append(f"- `{source}`: `{counts}`")
    lines.extend(
        [
            "",
            "Any survivor still needs one family-wide whole-event shuffle check. Related "
            "horizons and BTC/ETH rows are not independent discoveries.",
            "",
            f"Detailed partition rows: `{len(summary)}`.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_layer2_multi_source_activity_review":
        raise ValueError("Existing multi-source activity result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing multi-source activity artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    freeze, events, controls = load_frozen_inputs()
    outcomes, coverage = extract_outcomes(freeze, events, controls)
    summary = summarize_outcomes(outcomes, freeze)
    decisions = classify_routes(summary, freeze)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(outcomes, OUTCOME_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    REPORT_PATH.write_text(
        render_report(decisions, summary), encoding="utf-8", newline="\n"
    )
    retained = int(
        decisions["verdict"]
        .eq("retained_exploratory_activity_lead_pending_family_shuffle")
        .sum()
    )
    result = {
        "schema_version": 1,
        "status": "completed_layer2_multi_source_activity_review",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": True,
        "profit_used": False,
        "direction_tested": False,
        "event_rows": len(events),
        "outcome_rows": len(outcomes),
        "route_rows": len(decisions),
        "retained_exploratory_routes": retained,
        "artifacts": {
            "outcomes": artifact(OUTCOME_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "summary": artifact(SUMMARY_PATH),
            "decisions": artifact(DECISION_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "freeze": artifact(FREEZE_PATH),
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
                    "outcomes_will_be_read": True,
                    "profit_will_be_used": False,
                    "direction_will_be_tested": False,
                    "source_routes": list(frozen.ACTIVITY_ROUTES),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
