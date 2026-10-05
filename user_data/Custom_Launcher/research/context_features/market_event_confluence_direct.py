"""Test every frozen event-confluence route against ordinary and isolated-event controls."""

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
    market_context_independent_direct as context_direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_confluence_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.OUTPUT_ROOT / "event_confluence_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "event_confluence_freeze.json"
CATALOG_PATH = frozen.OUTPUT_ROOT / "event_confluence_route_catalog.csv"
SOURCE_PATH = frozen.OUTPUT_ROOT / "combined_source_event_catalog.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
REQUIRED_PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
MARKET_SCOPES = {
    "btc": context_direct.MARKET_SCOPES["btc"],
    "eth": context_direct.MARKET_SCOPES["eth"],
    "established_alts": context_direct.MARKET_SCOPES["established_alts"],
}
SIGNED_ROUTES = {"aligned_positive_market_signs", "aligned_negative_market_signs"}
ISOLATED_REFERENCE = "isolated_family_reference"
CONTROL_SEARCH_WEEKS = 52
CONTROL_COUNT = 12
CONTROL_EXCLUSION_HOURS = 72


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def parse_time_column(frame: DataFrame, column: str) -> None:
    frame[column] = pd.to_datetime(
        frame[column], utc=True, errors="raise", format="mixed"
    )


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_confluence_freeze":
        raise ValueError("Event-confluence freeze is not terminal.")
    if freeze.get("status") != "frozen_event_confluence_before_crypto_outcomes":
        raise ValueError("Event-confluence definitions are not frozen.")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Event-confluence freeze unexpectedly read crypto outcomes.")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("catalog", CATALOG_PATH),
        ("sources", SOURCE_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen event-confluence artifact changed: {path}")
    catalog = pd.read_csv(CATALOG_PATH)
    sources = pd.read_csv(SOURCE_PATH)
    parse_time_column(catalog, "anchor_utc")
    parse_time_column(sources, "decision_hour_utc")
    return freeze, catalog, sources


def load_market_frames() -> tuple[dict[str, DataFrame], DataFrame]:
    frames: dict[str, DataFrame] = {}
    coverage: list[dict[str, Any]] = []
    pairs = sorted({pair for members in MARKET_SCOPES.values() for pair in members})
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicates = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
        frame["_position"] = np.arange(len(frame), dtype=int)
        frames[pair] = frame.set_index("date", drop=False)
        coverage.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "rows": len(frame),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "duplicate_timestamps_removed": duplicates,
            }
        )
    return frames, DataFrame.from_records(coverage)


def prior_week_control_anchors(
    anchor: pd.Timestamp, blocked: Sequence[pd.Timestamp]
) -> tuple[pd.Timestamp, ...]:
    controls: list[pd.Timestamp] = []
    exclusion = pd.Timedelta(hours=CONTROL_EXCLUSION_HOURS)
    for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
        candidate = anchor - pd.Timedelta(weeks=weeks)
        if any(abs(candidate - event) <= exclusion for event in blocked):
            continue
        controls.append(candidate)
        if len(controls) >= CONTROL_COUNT:
            break
    return tuple(controls)


def extract_outcomes(catalog: DataFrame, frames: Mapping[str, DataFrame]) -> DataFrame:
    blocked_by_route = {
        route_id: tuple(pd.Timestamp(value) for value in route["anchor_utc"].unique())
        for route_id, route in catalog.groupby("route_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for event in catalog.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_utc)
        controls = prior_week_control_anchors(
            anchor, blocked_by_route[str(event.route_id)]
        )
        for horizon in frozen.HORIZONS_HOURS:
            pair_metrics: dict[str, dict[str, float]] = {}
            for pair, frame in frames.items():
                metrics = context_direct.pair_event_metrics(
                    frame, anchor, controls, horizon
                )
                if metrics is not None:
                    pair_metrics[pair] = metrics
            for scope, members in MARKET_SCOPES.items():
                scoped = context_direct.scope_from_pairs(
                    scope,
                    [pair_metrics[pair] for pair in members if pair in pair_metrics],
                )
                if scoped is None:
                    continue
                records.append(
                    {
                        "route_event_id": event.route_event_id,
                        "route_id": event.route_id,
                        "route_label": event.route_label,
                        "anchor_utc": anchor,
                        "whole_event_partition": event.whole_event_partition,
                        "signed_prediction": int(event.signed_prediction),
                        "component_family_count": int(event.family_count_24h),
                        "component_source_group_count": int(
                            event.source_group_count_24h
                        ),
                        "component_families": event.component_families,
                        "background_state": event.background_state,
                        "local_level_state": event.local_level_state,
                        "horizon_hours": horizon,
                        "market_scope": scope,
                        **scoped,
                    }
                )
    return DataFrame.from_records(records)


def add_result_flags(outcomes: DataFrame) -> DataFrame:
    out = outcomes.copy()
    out["reaction_success"] = out["activity_score"].ge(
        frozen.MINIMUM_MEDIAN_ACTIVITY_SCORE
    )
    out["actual_direction"] = out["signed_return"].map(context_direct.sign)
    out["issued_direction_call"] = out["signed_prediction"].ne(0)
    valid = out["issued_direction_call"] & out["actual_direction"].ne(0)
    out["direction_success"] = out["signed_prediction"].eq(
        out["actual_direction"]
    ).where(valid)
    recent_direction = out["recent_24h_return"].map(context_direct.sign)
    out["recent_trend_success"] = recent_direction.eq(out["actual_direction"]).where(
        recent_direction.ne(0) & out["actual_direction"].ne(0)
    )
    out["rotated_prediction"] = 0
    grouping = ["route_id", "market_scope", "horizon_hours", "whole_event_partition"]
    for _keys, indexes in out.groupby(grouping, sort=False).groups.items():
        ordered = out.loc[indexes].sort_values("anchor_utc", kind="stable")
        predictions = ordered["signed_prediction"].to_numpy(dtype=int)
        if len(predictions):
            out.loc[ordered.index, "rotated_prediction"] = np.roll(predictions, 1)
    out["rotated_direction_success"] = out["rotated_prediction"].eq(
        out["actual_direction"]
    ).where(out["rotated_prediction"].ne(0) & out["actual_direction"].ne(0))
    out["source_batch"] = "historical_event_confluence"
    out = context_direct.add_majority_control(
        out.rename(columns={"route_id": "family", "signed_prediction": "predicted_direction"})
    ).rename(columns={"family": "route_id", "predicted_direction": "signed_prediction"})
    out["joint_success"] = (
        out["reaction_success"].eq(True) & out["direction_success"].eq(True)
    ).where(out["direction_success"].notna())
    return out


def summarize(outcomes: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "route_label",
        "market_scope",
        "horizon_hours",
        "whole_event_partition",
    ]
    records: list[dict[str, Any]] = []
    for values, group in outcomes.groupby(keys, sort=False):
        calls = group.loc[group["issued_direction_call"]]
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "whole_events": int(group["route_event_id"].nunique()),
                "median_activity_multiple": group["activity_score"].median(),
                "reaction_success_rate": group["reaction_success"].mean(),
                "direction_calls": int(calls["route_event_id"].nunique()),
                "direction_success_rate": calls["direction_success"].mean(),
                "recent_trend_success_rate": calls["recent_trend_success"].mean(),
                "majority_direction_success_rate": calls[
                    "majority_direction_success"
                ].mean(),
                "rotated_direction_success_rate": calls[
                    "rotated_direction_success"
                ].mean(),
                "joint_success_rate": calls["joint_success"].mean(),
                "minimum_member_count": int(group["member_count"].min()),
                "minimum_control_count": int(
                    group["minimum_pair_control_count"].min()
                ),
            }
        )
    return DataFrame.from_records(records)


def partition_rows(rows: DataFrame) -> dict[str, pd.Series] | None:
    output: dict[str, pd.Series] = {}
    for partition in REQUIRED_PARTITIONS:
        selected = rows.loc[rows["whole_event_partition"].eq(partition)]
        if selected.empty:
            return None
        output[partition] = selected.iloc[0]
    return output


def isolated_rows(
    summary: DataFrame, *, scope: str, horizon: int
) -> dict[str, pd.Series] | None:
    rows = summary.loc[
        summary["route_id"].eq(ISOLATED_REFERENCE)
        & summary["market_scope"].eq(scope)
        & summary["horizon_hours"].eq(horizon)
    ]
    return partition_rows(rows)


def activity_pass(rows: Mapping[str, pd.Series]) -> bool:
    return all(
        int(row["whole_events"]) >= frozen.MINIMUM_PARTITION_EVENTS
        and float(row["reaction_success_rate"]) >= frozen.MINIMUM_ABOVE_CONTROL_RATE
        and float(row["median_activity_multiple"])
        >= frozen.MINIMUM_MEDIAN_ACTIVITY_SCORE
        for row in rows.values()
    )


def exceeds_isolated(
    rows: Mapping[str, pd.Series], reference: Mapping[str, pd.Series]
) -> bool:
    return all(
        float(rows[partition]["median_activity_multiple"])
        > float(reference[partition]["median_activity_multiple"])
        and float(rows[partition]["reaction_success_rate"])
        > float(reference[partition]["reaction_success_rate"])
        for partition in REQUIRED_PARTITIONS
    )


def direction_pass(rows: Mapping[str, pd.Series]) -> bool:
    for row in rows.values():
        if int(row["direction_calls"]) < frozen.MINIMUM_PARTITION_EVENTS:
            return False
        comparator = context_direct.comparator_max(row)
        if (
            pd.isna(row["direction_success_rate"])
            or float(row["direction_success_rate"])
            < frozen.MINIMUM_DIRECTION_ACCURACY
            or pd.isna(comparator)
            or float(row["direction_success_rate"]) - comparator
            < frozen.MINIMUM_DIRECTION_LIFT_VS_CONTROLS
        ):
            return False
    return True


def route_cell_results(
    route_id: str, route_rows: DataFrame, summary: DataFrame
) -> dict[str, Any]:
    supported_cells = 0
    activity_cells: list[str] = []
    enhanced_cells: list[str] = []
    direction_cells: list[str] = []
    joint_cells: list[str] = []
    for (scope, horizon), cell in route_rows.groupby(
        ["market_scope", "horizon_hours"], sort=False
    ):
        partitions = partition_rows(cell)
        if partitions is None or not all(
            int(row["whole_events"]) >= frozen.MINIMUM_PARTITION_EVENTS
            for row in partitions.values()
        ):
            continue
        supported_cells += 1
        label = f"{scope}:{int(horizon)}h"
        passed_activity = activity_pass(partitions)
        if passed_activity:
            activity_cells.append(label)
        reference = isolated_rows(summary, scope=str(scope), horizon=int(horizon))
        if (
            passed_activity
            and route_id != ISOLATED_REFERENCE
            and reference is not None
            and exceeds_isolated(partitions, reference)
        ):
            enhanced_cells.append(label)
        passed_direction = direction_pass(partitions)
        if passed_direction:
            direction_cells.append(label)
        if passed_activity and passed_direction and all(
            pd.notna(row["joint_success_rate"])
            and float(row["joint_success_rate"]) >= frozen.JOINT_LEAD_FLOOR
            for row in partitions.values()
        ):
            joint_cells.append(label)
    return {
        "supported_cells": supported_cells,
        "activity_cells": activity_cells,
        "enhanced_cells": enhanced_cells,
        "direction_cells": direction_cells,
        "joint_cells": joint_cells,
    }


def route_verdict(route_id: str, result: Mapping[str, Any]) -> str:
    if result["joint_cells"]:
        return "repeatable_joint_activity_and_direction_lead"
    if result["direction_cells"]:
        return "repeatable_direction_lead"
    if result["enhanced_cells"]:
        return "repeatable_overlap_activity_lead"
    if route_id == ISOLATED_REFERENCE and result["activity_cells"]:
        return "isolated_event_activity_reference"
    if result["activity_cells"]:
        return "activity_repeats_but_overlap_adds_no_value"
    if result["supported_cells"] == 0:
        return "insufficient_two_partition_coverage"
    return "weak_or_inconsistent"


def classify(summary: DataFrame, catalog: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    route_definitions = catalog[["route_id", "route_label"]].drop_duplicates()
    for route in route_definitions.itertuples(index=False):
        route_rows = summary.loc[summary["route_id"].eq(route.route_id)]
        result = route_cell_results(str(route.route_id), route_rows, summary)
        records.append(
            {
                "route_id": route.route_id,
                "route_label": route.route_label,
                "verdict": route_verdict(str(route.route_id), result),
                "supported_scope_horizon_cells": result["supported_cells"],
                "activity_cells": "; ".join(result["activity_cells"]),
                "enhanced_over_isolated_cells": "; ".join(
                    result["enhanced_cells"]
                ),
                "direction_cells": "; ".join(result["direction_cells"]),
                "joint_cells": "; ".join(result["joint_cells"]),
            }
        )
    return DataFrame.from_records(records).sort_values("route_id", kind="stable")


def render_report(verdicts: DataFrame) -> str:
    lines = [
        "# Event-Confluence Direct Review",
        "",
        "Every frozen sibling route was completed before any descendant was selected.",
        "",
        "| Question | Result | Extra activity beyond isolated events | Direction |",
        "|---|---|---|---|",
    ]
    for row in verdicts.itertuples(index=False):
        lines.append(
            f"| {row.route_label} | `{row.verdict}` | "
            f"{row.enhanced_over_isolated_cells or '-'} | {row.direction_cells or '-'} |"
        )
    lines.extend(
        [
            "",
            "Activity must repeat in both 2021-2023 and 2024-2025 against ordinary-time "
            "controls. An overlap lead must also beat isolated events in both periods.",
            "",
            "Signed routes require at least ten events in each period. No direction is "
            "inferred from scheduled releases or unsigned news counts.",
            "",
            "This is not a profit, entry, exit, or trading-rule test.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "event_confluence_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_confluence_direct_review":
            raise ValueError("Existing event-confluence direct result is not terminal.")
        return result
    _freeze, catalog, _sources = load_frozen_inputs()
    frames, coverage = load_market_frames()
    outcomes = add_result_flags(extract_outcomes(catalog, frames))
    summary = summarize(outcomes)
    verdicts = classify(summary, catalog)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes_path = OUTPUT_ROOT / "route_scope_outcomes.parquet"
    summary_path = OUTPUT_ROOT / "route_scope_summary.csv"
    verdicts_path = OUTPUT_ROOT / "route_verdicts.csv"
    coverage_path = OUTPUT_ROOT / "market_coverage.csv"
    report_path = OUTPUT_ROOT / "event_confluence_plain_review.md"
    g0.atomic_write_parquet(outcomes, outcomes_path)
    g0.atomic_write_csv(summary, summary_path)
    g0.atomic_write_csv(verdicts, verdicts_path)
    g0.atomic_write_csv(coverage, coverage_path)
    report_path.write_text(render_report(verdicts), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_event_confluence_direct_review",
        "created_at_utc": g0.utc_now(),
        "route_count": int(verdicts["route_id"].nunique()),
        "verdict_counts": verdicts["verdict"].value_counts().to_dict(),
        "outcomes_are_profit_or_trading_targets": False,
        "artifacts": {
            "outcomes": artifact(outcomes_path),
            "summary": artifact(summary_path),
            "verdicts": artifact(verdicts_path),
            "coverage": artifact(coverage_path),
            "report": artifact(report_path),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, result_path)
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
                    "mode": "setup_only",
                    "freeze": str(FREEZE_PATH),
                    "output_root": str(OUTPUT_ROOT),
                    "profit_or_trading_target": False,
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
