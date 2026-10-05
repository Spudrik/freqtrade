"""Test frozen Treasury refunding clocks for repeatable BTC and ETH activity."""

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
    market_event_central_bank_activity_direct as central_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_treasury_refunding_catalogue as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
SOURCE_RESULT_PATH = frozen.RESULT_PATH
FREEZE_PATH = frozen.FREEZE_PATH
CATALOG_PATH = frozen.CATALOG_PATH
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260906a"
OUTCOME_PATH = OUTPUT_ROOT / "treasury_refunding_activity_rows.parquet"
COVERAGE_PATH = OUTPUT_ROOT / "treasury_refunding_activity_coverage.csv"
SUMMARY_PATH = OUTPUT_ROOT / "treasury_refunding_activity_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "treasury_refunding_activity_route_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "treasury_refunding_activity_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "treasury_refunding_activity_result.json"

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
HORIZONS = (5, 15, 30, 60)
PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
VARIANTS = (
    "all_quarterly_statements",
    "exclude_other_major_scheduled_event_within_4h",
)
COLLISION_HOURS = 4


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def validate_recorded_artifact(
    result: Mapping[str, Any], key: str, path: Path
) -> None:
    if result["artifacts"][key]["sha256"] != g0.sha256_file(path):
        raise ValueError(f"Frozen Treasury artifact changed: {path}")


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_treasury_refunding_catalogue":
        raise ValueError("Treasury source catalogue is not complete")
    if freeze.get("status") != "frozen_treasury_refunding_before_market_outcomes":
        raise ValueError("Treasury market question was not frozen")
    if freeze.get("outcomes_read") or result.get("outcomes_read"):
        raise ValueError("Treasury source freeze unexpectedly read market outcomes")
    if freeze.get("profit_used") is not False or result.get("profit_used") is not False:
        raise ValueError("Treasury source freeze lacks an explicit no-profit declaration")
    validate_recorded_artifact(result, "catalogue", CATALOG_PATH)
    validate_recorded_artifact(result, "freeze", FREEZE_PATH)
    catalogue = pd.read_csv(CATALOG_PATH)
    catalogue["anchor_utc"] = pd.to_datetime(catalogue["anchor_utc"], utc=True)
    return freeze, catalogue


def load_external_anchors() -> tuple[list[pd.Timestamp], list[dict[str, Any]]]:
    anchors, artifacts = central_activity.load_external_anchors()
    _, central_catalogue = central_activity.load_frozen_inputs()
    anchors.extend(central_catalogue["available_at_utc"].tolist())
    artifacts.append(
        {
            "catalogue": artifact(central_activity.CATALOG_PATH),
            "source_result": artifact(central_activity.SOURCE_RESULT_PATH),
            "outcomes_read": False,
            "profit_used_field_status": "explicit_false",
        }
    )
    return anchors, artifacts


def add_collision_flags(
    catalogue: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> DataFrame:
    output = catalogue.copy()
    output["other_major_scheduled_event_within_4h"] = [
        shared.minimum_event_distance_hours(external_anchors, pd.Timestamp(anchor))
        <= COLLISION_HOURS
        for anchor in output["anchor_utc"]
    ]
    return output


def extract_outcomes(
    catalogue: DataFrame, external_anchors: Sequence[pd.Timestamp]
) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    blocked_anchors = [*catalogue["anchor_utc"].tolist(), *external_anchors]
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
        records.extend(
            extract_pair_outcomes(
                catalogue,
                pair=pair,
                frame=frame,
                positions=positions,
                blocked_anchors=blocked_anchors,
            )
        )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def extract_pair_outcomes(
    catalogue: DataFrame,
    *,
    pair: str,
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in catalogue.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_utc)
        event_position = positions.get(anchor)
        for horizon in HORIZONS:
            status, response = response_for_horizon(frame, event_position, horizon)
            controls = (
                central_activity.prior_week_controls(
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
                status = "too_few_clean_controls"
            records.append(
                {
                    "event_id": event.event_id,
                    "anchor_utc": anchor,
                    "whole_event_partition": event.whole_event_partition,
                    "pair": pair,
                    "horizon_minutes": horizon,
                    "control_count": len(controls),
                    "eligibility_status": status,
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
                }
            )
    return records


def response_for_horizon(
    frame: DataFrame, position: int | None, horizon: int
) -> tuple[str, dict[str, float] | None]:
    if position is None:
        return "missing_anchor", None
    status = central_activity.minute_window_status(frame, position, horizon)
    return status, central_activity.contiguous_window_metrics(frame, position, horizon)


def summarize_outcomes(outcomes: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    rules = freeze["market_test"]
    minimum_events = rules["minimum_events_by_partition"]
    minimum_score = float(rules["minimum_median_activity_score"])
    minimum_rate = float(rules["minimum_above_control_rate"])
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = outcomes.loc[
            outcomes["activity_score"].notna()
            & outcomes["whole_event_partition"].isin(PARTITIONS)
        ].copy()
        if variant == "exclude_other_major_scheduled_event_within_4h":
            selected = selected.loc[
                ~selected["other_major_scheduled_event_within_4h"]
            ]
        grouped = selected.groupby(
            ["pair", "horizon_minutes", "whole_event_partition"], sort=True
        )
        for keys, group in grouped:
            pair, horizon, partition = keys
            event_count = int(group["event_id"].nunique())
            median_score = float(group["activity_score"].median())
            above_rate = float(group["above_control_median"].mean())
            records.append(
                {
                    "variant": variant,
                    "pair": pair,
                    "horizon_minutes": int(horizon),
                    "whole_event_partition": partition,
                    "events": event_count,
                    "required_events": int(minimum_events[partition]),
                    "median_activity_score": median_score,
                    "above_control_rate": above_rate,
                    "median_abs_return_ratio": float(
                        group["abs_return_ratio"].median()
                    ),
                    "median_range_ratio": float(group["range_ratio"].median()),
                    "median_volume_ratio": float(group["volume_ratio"].median()),
                    "partition_pass": (
                        event_count >= int(minimum_events[partition])
                        and median_score >= minimum_score
                        and above_rate >= minimum_rate
                    ),
                }
            )
    return DataFrame.from_records(records)


def route_has_coverage(route: DataFrame, minimum_events: Mapping[str, int]) -> bool:
    required_cells = {
        (variant, partition) for variant in VARIANTS for partition in PARTITIONS
    }
    observed = set(
        zip(route["variant"], route["whole_event_partition"], strict=False)
    )
    if not required_cells.issubset(observed):
        return False
    return all(
        int(row.events) >= int(minimum_events[row.whole_event_partition])
        for row in route.itertuples(index=False)
    )


def classify_routes(summary: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    rules = freeze["market_test"]
    minimum_events = rules["minimum_events_by_partition"]
    minimum_score = float(rules["minimum_median_activity_score"])
    minimum_rate = float(rules["minimum_above_control_rate"])
    records: list[dict[str, Any]] = []
    for pair in ASSETS:
        for horizon in HORIZONS:
            route = summary.loc[
                summary["pair"].eq(pair)
                & summary["horizon_minutes"].eq(horizon)
            ]
            sufficient = route_has_coverage(route, minimum_events)
            all_pass = bool(len(route) == 4 and route["partition_pass"].all())
            raw = route.loc[route["variant"].eq("all_quarterly_statements")]
            raw_pass = bool(len(raw) == 2 and raw["partition_pass"].all())
            raw_signal_pass = bool(
                len(raw) == 2
                and raw["median_activity_score"].ge(minimum_score).all()
                and raw["above_control_rate"].ge(minimum_rate).all()
            )
            if all_pass:
                verdict = "retained_exploratory_activity_lead"
            elif raw_signal_pass and not sufficient:
                verdict = "coverage_limited"
            elif raw_pass:
                verdict = "overlap_dependent_not_retained"
            else:
                verdict = "weak_or_inconsistent"
            records.append(
                {
                    "pair": pair,
                    "horizon_minutes": horizon,
                    "verdict": verdict,
                    "minimum_events_per_cell": (
                        int(route["events"].min()) if not route.empty else 0
                    ),
                    "interpretation_limit": (
                        "Exploratory activity only; no direction, causation, profit, "
                        "or trading rule."
                    ),
                }
            )
    return DataFrame.from_records(records)


def render_report(decisions: DataFrame, summary: DataFrame, outcomes: DataFrame) -> str:
    retained = decisions.loc[
        decisions["verdict"].eq("retained_exploratory_activity_lead")
    ]
    rows = [
        "# Treasury Refunding Activity Review",
        "",
        "Question: do complete quarterly Treasury refunding statement clocks "
        "repeatedly make Bitcoin or Ethereum unusually active over 5-60 minutes?",
        "",
        "| Market | Retained windows |",
        "|---|---|",
    ]
    for pair in ASSETS:
        matched = retained.loc[retained["pair"].eq(pair)]
        windows = ", ".join(
            f"{int(value)}m" for value in matched["horizon_minutes"].sort_values()
        )
        rows.append(f"| {pair.split('/')[0]} | {windows or '-'} |")
    rows.extend(
        [
            "",
            f"- Catalogue events with any usable row: "
            f"`{outcomes.loc[outcomes['activity_score'].notna(), 'event_id'].nunique()}`",
            f"- Retained market/window routes: `{len(retained)}` of `{len(decisions)}`",
            "- A route had to pass both periods and both the all-release and "
            "no-nearby-major-event versions.",
            "- With only 12 development and 8 later releases, every result remains "
            "exploratory.",
            "- This tests unusual movement, range, and volume. It does not test "
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
        f"{row.pair.split('/')[0]} {int(row.horizon_minutes)}m, "
        f"{row.whole_event_partition}, {row.variant}: "
        f"{row.events} events, median activity {row.median_activity_score:.2f}x, "
        f"above ordinary weeks {row.above_control_rate:.1%}, "
        f"{'pass' if row.partition_pass else 'fail'}."
        for row in strongest.itertuples(index=False)
    )
    rows.append("")
    return "\n".join(rows)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_treasury_refunding_activity_direct_test":
        raise ValueError("Existing Treasury activity result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing Treasury activity artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
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
        decisions["verdict"].eq("retained_exploratory_activity_lead")
    ]
    result = {
        "schema_version": 1,
        "status": "completed_treasury_refunding_activity_direct_test",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": False,
        "catalogue_event_rows": len(catalogue),
        "usable_event_rows": int(
            outcomes.loc[outcomes["activity_score"].notna(), "event_id"].nunique()
        ),
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
                "Exclude a Treasury release or prior-week control within four hours "
                "of a frozen CPI, FOMC, scheduled-US-macro, ECB, BoE, or BoJ clock."
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
