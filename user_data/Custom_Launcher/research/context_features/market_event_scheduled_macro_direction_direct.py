"""Test the frozen five-family scheduled-macro direction baseline.

The test uses BTC and ETH minute candles, whole development/validation release
periods, complete windows, and predeclared simple controls.  It measures direction
only; it does not calculate profit or promote a trading rule.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as cpi_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.RESULT_PATH
FREEZE_PATH = frozen.FREEZE_PATH
CATALOG_PATH = frozen.CATALOG_PATH
PARENT_CATALOG_PATH = frozen.PARENT_CATALOG_PATH
CPI_CATALOG_PATH = cpi_frozen.OUTPUT_ROOT / "cpi_release_catalog.csv"
FOMC_CATALOG_PATH = layer1.OUTPUT_ROOT / "official_fomc_event_catalog.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260905a"
DETAIL_ROOT = OUTPUT_ROOT / "details"
OUTCOME_PATH = DETAIL_ROOT / "scheduled_macro_direction_rows.parquet"
SUMMARY_PATH = OUTPUT_ROOT / "scheduled_macro_direction_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "scheduled_macro_direction_decisions.csv"
FAMILY_PATH = OUTPUT_ROOT / "scheduled_macro_direction_family_decisions.csv"
COVERAGE_PATH = OUTPUT_ROOT / "scheduled_macro_direction_market_coverage.csv"
REPORT_PATH = OUTPUT_ROOT / "scheduled_macro_direction_report.md"
RESULT_PATH = OUTPUT_ROOT / "scheduled_macro_direction_result.json"


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_scheduled_macro_direction_freeze":
        raise ValueError("Scheduled-macro direction freeze is not terminal")
    if freeze.get("status") != "frozen_scheduled_macro_direction_before_signed_outcomes":
        raise ValueError("Scheduled-macro direction definitions were not frozen")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Scheduled-macro freeze unexpectedly contains outcomes")
    for name, path in (("freeze", FREEZE_PATH), ("catalog", CATALOG_PATH)):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen scheduled-macro artifact changed: {path}")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    return result, freeze, catalog


def optional_anchor_catalog(
    path: Path, *, event_group: str | None = None
) -> list[pd.Timestamp]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    if event_group is not None and "event_group" in frame:
        frame = frame.loc[frame["event_group"].eq(event_group)]
    if "anchor_utc" not in frame:
        raise ValueError(f"Anchor catalogue has no anchor_utc column: {path}")
    return list(pd.to_datetime(frame["anchor_utc"], utc=True))


def blocked_scheduled_anchors() -> list[pd.Timestamp]:
    anchors = optional_anchor_catalog(
        PARENT_CATALOG_PATH, event_group="scheduled_release"
    )
    anchors.extend(optional_anchor_catalog(CPI_CATALOG_PATH))
    anchors.extend(optional_anchor_catalog(FOMC_CATALOG_PATH))
    return sorted(set(anchors))


def position_by_date(frame: DataFrame) -> dict[pd.Timestamp, int]:
    return {stamp: int(position) for position, stamp in enumerate(frame["date"])}


def _is_complete_minute_span(dates: Series, start: pd.Timestamp, length: int) -> bool:
    if len(dates) != length or length < 1:
        return False
    expected_end = start + pd.Timedelta(minutes=length - 1)
    return bool(
        dates.iloc[0] == start
        and dates.iloc[-1] == expected_end
        and dates.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
    )


def future_return(
    frame: DataFrame, position: int, length: int
) -> tuple[float, str | None]:
    if position < 0 or position + length > len(frame):
        return np.nan, "incomplete_future_window"
    window = frame.iloc[position : position + length]
    start = pd.Timestamp(frame.iloc[position]["date"])
    if not _is_complete_minute_span(window["date"], start, length):
        return np.nan, "minute_gap_in_future_window"
    opening = float(window.iloc[0]["open"])
    closing = float(window.iloc[-1]["close"])
    if not np.isfinite(opening) or opening <= 0 or not np.isfinite(closing):
        return np.nan, "invalid_future_price"
    return float(closing / opening - 1.0), None


def pre_event_return(frame: DataFrame, position: int, length: int) -> float:
    start_position = position - length
    if start_position < 0 or position >= len(frame):
        return np.nan
    dates = frame.iloc[start_position : position + 1]["date"]
    anchor = pd.Timestamp(frame.iloc[position]["date"])
    expected_start = anchor - pd.Timedelta(minutes=length)
    if len(dates) != length + 1:
        return np.nan
    if dates.iloc[0] != expected_start or dates.iloc[-1] != anchor:
        return np.nan
    if not dates.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        return np.nan
    opening = float(frame.iloc[start_position]["open"])
    anchor_open = float(frame.iloc[position]["open"])
    if not np.isfinite(opening) or opening <= 0 or not np.isfinite(anchor_open):
        return np.nan
    return float(anchor_open / opening - 1.0)


def direction(value: Any) -> int:
    if value is None or pd.isna(value) or float(value) == 0.0:
        return 0
    return 1 if float(value) > 0 else -1


def minimum_event_distance_hours(
    anchors: Iterable[pd.Timestamp], target: pd.Timestamp
) -> float:
    distances = [abs((target - anchor).total_seconds()) / 3600 for anchor in anchors]
    return min(distances) if distances else np.inf


def matched_ordinary_returns(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[float]:
    returns: list[float] = []
    for weeks in range(1, frozen.CONTROL_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if (
            minimum_event_distance_hours(blocked_anchors, control_anchor)
            <= frozen.CONTROL_EVENT_EXCLUSION_HOURS
        ):
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        value, reason = future_return(frame, position, horizon)
        if reason is None and direction(value) != 0:
            returns.append(value)
        if len(returns) >= frozen.CONTROL_COUNT:
            break
    return returns


def _hit(predicted: Any, actual: Any) -> bool | None:
    predicted_direction = direction(predicted)
    actual_direction = direction(actual)
    if predicted_direction == 0 or actual_direction == 0:
        return None
    return predicted_direction == actual_direction


def extract_direction_rows(
    catalog: DataFrame,
    *,
    blocked_anchors: Sequence[pd.Timestamp] | None = None,
) -> tuple[DataFrame, DataFrame]:
    blocked = list(blocked_anchors) if blocked_anchors is not None else blocked_scheduled_anchors()
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in frozen.ASSETS:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
            }
        )
        for event in catalog.itertuples(index=False):
            anchor = pd.Timestamp(event.anchor_utc)
            position = positions.get(anchor)
            for horizon in frozen.HORIZONS_MINUTES:
                if position is None:
                    response_return = np.nan
                    abstention_reason = "missing_anchor_candle"
                    pre_return = np.nan
                    controls: list[float] = []
                else:
                    response_return, abstention_reason = future_return(
                        frame, position, horizon
                    )
                    pre_return = pre_event_return(frame, position, horizon)
                    controls = matched_ordinary_returns(
                        frame,
                        positions,
                        anchor=anchor,
                        horizon=horizon,
                        blocked_anchors=blocked,
                    )
                actual_direction = direction(response_return)
                if abstention_reason is None and actual_direction == 0:
                    abstention_reason = "zero_future_return"
                predicted = int(event.predicted_crypto_direction)
                ordinary_hits = [_hit(predicted, value) for value in controls]
                valid_ordinary_hits = [value for value in ordinary_hits if value is not None]
                records.append(
                    {
                        "event_id": event.event_id,
                        "release_cluster_id": event.release_cluster_id,
                        "event_family": event.event_family,
                        "source_series_id": event.source_series_id,
                        "anchor_utc": anchor,
                        "whole_event_partition": event.whole_event_partition,
                        "pair": pair,
                        "horizon_minutes": int(horizon),
                        "predicted_direction": predicted,
                        "previous_release_predicted_direction": (
                            int(event.previous_release_predicted_direction)
                            if pd.notna(event.previous_release_predicted_direction)
                            else 0
                        ),
                        "response_return": response_return,
                        "actual_direction": actual_direction,
                        "pre_event_return": pre_return,
                        "pretrend_direction": direction(pre_return),
                        "ordinary_control_count": len(valid_ordinary_hits),
                        "ordinary_control_accuracy": (
                            float(np.mean(valid_ordinary_hits))
                            if len(valid_ordinary_hits) >= frozen.MINIMUM_CONTROLS
                            else np.nan
                        ),
                        "abstention_reason": abstention_reason,
                    }
                )
    rows = DataFrame.from_records(records)
    rows["direction_hit"] = [
        _hit(predicted, actual)
        for predicted, actual in zip(
            rows["predicted_direction"], rows["actual_direction"], strict=True
        )
    ]
    rows["pretrend_hit"] = [
        _hit(predicted, actual)
        for predicted, actual in zip(
            rows["pretrend_direction"], rows["actual_direction"], strict=True
        )
    ]
    rows["previous_release_hit"] = [
        _hit(predicted, actual)
        for predicted, actual in zip(
            rows["previous_release_predicted_direction"],
            rows["actual_direction"],
            strict=True,
        )
    ]
    rows = add_rotated_and_majority_controls(rows)
    return rows, DataFrame.from_records(coverage)


def _circular_previous(values: Series) -> Series:
    if values.empty:
        return values
    array = values.to_numpy(copy=True)
    return Series(np.roll(array, 1), index=values.index)


def add_rotated_and_majority_controls(rows: DataFrame) -> DataFrame:
    result = rows.sort_values(
        ["event_family", "pair", "horizon_minutes", "whole_event_partition", "anchor_utc"],
        kind="stable",
    ).copy()
    rotate_keys = [
        "event_family",
        "pair",
        "horizon_minutes",
        "whole_event_partition",
    ]
    result["rotated_predicted_direction"] = result.groupby(
        rotate_keys, sort=False
    )["predicted_direction"].transform(_circular_previous)
    result["rotated_hit"] = [
        _hit(predicted, actual)
        for predicted, actual in zip(
            result["rotated_predicted_direction"],
            result["actual_direction"],
            strict=True,
        )
    ]

    development = result.loc[
        result["whole_event_partition"].eq(frozen.EVALUATION_PARTITIONS[0])
        & result["actual_direction"].isin([-1, 1])
    ]
    majority = (
        development.groupby(
            ["event_family", "pair", "horizon_minutes"], sort=False
        )["actual_direction"]
        .sum()
        .map(direction)
        .rename("development_majority_direction")
        .reset_index()
    )
    result = result.merge(
        majority,
        on=["event_family", "pair", "horizon_minutes"],
        how="left",
        validate="many_to_one",
    )
    result["development_majority_direction"] = result[
        "development_majority_direction"
    ].fillna(0).astype(int)
    result["development_majority_hit"] = [
        _hit(predicted, actual)
        for predicted, actual in zip(
            result["development_majority_direction"],
            result["actual_direction"],
            strict=True,
        )
    ]
    return result.sort_values(
        ["anchor_utc", "event_family", "pair", "horizon_minutes"], kind="stable"
    ).reset_index(drop=True)


def _rate(values: Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if len(valid) else np.nan


def summarize_group(group: DataFrame) -> dict[str, Any]:
    usable = group.loc[group["direction_hit"].notna()]
    ordinary_eligible = usable.loc[usable["ordinary_control_accuracy"].notna()]
    ordinary = ordinary_eligible["ordinary_control_accuracy"]
    ordinary_paired_event_accuracy = _rate(ordinary_eligible["direction_hit"])
    ordinary_period_accuracy = float(ordinary.mean()) if len(ordinary) else np.nan
    return {
        "events_total": int(group["event_id"].nunique()),
        "events_usable": int(usable["event_id"].nunique()),
        "events_abstained": int(group["event_id"].nunique() - usable["event_id"].nunique()),
        "accuracy": _rate(usable["direction_hit"]),
        "development_majority_accuracy": _rate(
            usable["development_majority_hit"]
        ),
        "pretrend_accuracy": _rate(usable["pretrend_hit"]),
        "previous_release_accuracy": _rate(usable["previous_release_hit"]),
        "rotated_sign_accuracy": _rate(usable["rotated_hit"]),
        "ordinary_paired_event_accuracy": ordinary_paired_event_accuracy,
        "ordinary_period_accuracy": ordinary_period_accuracy,
        "ordinary_paired_lift": (
            float(ordinary_paired_event_accuracy - ordinary_period_accuracy)
            if np.isfinite(ordinary_paired_event_accuracy)
            and np.isfinite(ordinary_period_accuracy)
            else np.nan
        ),
        "ordinary_control_eligible_events": int(ordinary.shape[0]),
        "median_response_return": (
            float(usable["response_return"].median()) if len(usable) else np.nan
        ),
    }


def direction_summaries(rows: DataFrame) -> DataFrame:
    keys = ["event_family", "source_series_id", "pair", "horizon_minutes"]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key, strict=True))
        for partition, partition_rows in group.groupby(
            "whole_event_partition", sort=False
        ):
            records.append(
                {
                    **base,
                    "partition": partition,
                    **summarize_group(partition_rows),
                }
            )
        evaluation = group.loc[
            group["whole_event_partition"].isin(frozen.EVALUATION_PARTITIONS)
        ]
        records.append(
            {
                **base,
                "partition": "development_plus_validation",
                **summarize_group(evaluation),
            }
        )
    return DataFrame.from_records(records)


def _finite_max(values: Sequence[Any]) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    return max(finite) if finite else np.nan


def route_decisions(summary: DataFrame) -> DataFrame:
    keys = ["event_family", "source_series_id", "pair", "horizon_minutes"]
    records: list[dict[str, Any]] = []
    for key, group in summary.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key, strict=True))
        by_partition = group.set_index("partition")
        development = by_partition.loc[frozen.EVALUATION_PARTITIONS[0]]
        validation = by_partition.loc[frozen.EVALUATION_PARTITIONS[1]]
        combined = by_partition.loc["development_plus_validation"]
        nonordinary_controls = [
            combined["development_majority_accuracy"],
            combined["pretrend_accuracy"],
            combined["previous_release_accuracy"],
            combined["rotated_sign_accuracy"],
        ]
        best_nonordinary_control = _finite_max(nonordinary_controls)
        coverage_ok = bool(
            int(development["events_usable"]) >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(validation["events_usable"]) >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(development["ordinary_control_eligible_events"])
            >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(validation["ordinary_control_eligible_events"])
            >= frozen.MINIMUM_EVENTS_PER_PARTITION
        )
        threshold_ok = bool(
            float(combined["accuracy"]) >= frozen.MINIMUM_OVERALL_ACCURACY
            and float(development["accuracy"]) >= frozen.MINIMUM_PARTITION_ACCURACY
            and float(validation["accuracy"]) >= frozen.MINIMUM_PARTITION_ACCURACY
        )
        nonordinary_control_ok = bool(
            np.isfinite(best_nonordinary_control)
            and float(combined["accuracy"])
            >= best_nonordinary_control + frozen.MINIMUM_CONTROL_LIFT
        )
        ordinary_control_ok = bool(
            pd.notna(combined["ordinary_paired_lift"])
            and float(combined["ordinary_paired_lift"])
            >= frozen.MINIMUM_CONTROL_LIFT
        )
        candidate = (
            coverage_ok
            and threshold_ok
            and nonordinary_control_ok
            and ordinary_control_ok
        )
        if not coverage_ok:
            decision = "coverage_limited"
        elif candidate:
            decision = "provisional_requires_familywide_validation"
        else:
            decision = "failed_frozen_direction_baseline"
        records.append(
            {
                **base,
                "development_events": int(development["events_usable"]),
                "validation_events": int(validation["events_usable"]),
                "development_accuracy": float(development["accuracy"]),
                "validation_accuracy": float(validation["accuracy"]),
                "overall_accuracy": float(combined["accuracy"]),
                "best_nonordinary_control_accuracy": best_nonordinary_control,
                "lift_over_best_nonordinary_control": (
                    float(combined["accuracy"] - best_nonordinary_control)
                    if np.isfinite(best_nonordinary_control)
                    else np.nan
                ),
                "ordinary_paired_event_accuracy": float(
                    combined["ordinary_paired_event_accuracy"]
                ),
                "ordinary_period_accuracy": float(
                    combined["ordinary_period_accuracy"]
                ),
                "ordinary_paired_lift": float(combined["ordinary_paired_lift"]),
                "candidate": candidate,
                "decision": decision,
            }
        )
    return DataFrame.from_records(records)


def family_decisions(decisions: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for family, group in decisions.groupby("event_family", sort=True):
        candidates = group.loc[group["candidate"]]
        btc_horizons = set(
            candidates.loc[candidates["pair"].eq(frozen.ASSETS[0]), "horizon_minutes"]
        )
        eth_horizons = set(
            candidates.loc[candidates["pair"].eq(frozen.ASSETS[1]), "horizon_minutes"]
        )
        shared = sorted(btc_horizons & eth_horizons)
        if shared:
            decision = "provisional_two_market_lead_requires_familywide_validation"
        elif btc_horizons or eth_horizons:
            decision = "narrow_one_market_candidate_requires_familywide_validation"
        elif group["decision"].eq("coverage_limited").all():
            decision = "coverage_limited"
        else:
            decision = "no_raw_direction_lead"
        records.append(
            {
                "event_family": family,
                "decision": decision,
                "btc_candidate_horizons_minutes": "; ".join(
                    str(int(value)) for value in sorted(btc_horizons)
                ),
                "eth_candidate_horizons_minutes": "; ".join(
                    str(int(value)) for value in sorted(eth_horizons)
                ),
                "shared_candidate_horizons_minutes": "; ".join(
                    str(int(value)) for value in shared
                ),
                "interpretation_limit": (
                    "Direction correlation only; no expectation surprise, causation, "
                    "profit, or trading-rule claim."
                ),
            }
        )
    return DataFrame.from_records(records)


def render_report(
    decisions: DataFrame, families: DataFrame, rows: DataFrame
) -> str:
    candidates = decisions.loc[decisions["candidate"]]
    abstentions = int(rows["direction_hit"].isna().sum())
    lines = [
        "# Scheduled Macro Direction Direct Review",
        "",
        "## Plain question",
        "",
        "Does acceleration versus the previous published figure provide repeatable "
        "5-60 minute Bitcoin or Ethereum direction for producer prices, employment, "
        "consumer spending/prices, retail sales, or GDP?",
        "",
        "## Boundaries",
        "",
        "- This is not actual versus the market forecast.",
        "- Complete one-minute windows only; missing or zero-return outcomes abstain.",
        "- Development and later validation releases remain separate.",
        "- The frozen sign must beat simple directional and matched-week controls.",
        "- No profit, causation, or trading-rule claim is made.",
        "",
        "## Outcome",
        "",
        f"- Scored cells: `{len(decisions)}`",
        f"- Raw candidate cells: `{len(candidates)}`",
        f"- Abstained event/asset/horizon rows: `{abstentions}`",
        "- Any raw candidate still requires a whole-event family-wide search check.",
        "",
        "## Family decisions",
        "",
    ]
    for row in families.itertuples(index=False):
        lines.append(f"- `{row.event_family}`: `{row.decision}`")
    return "\n".join(lines) + "\n"


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") not in {
            "completed_no_raw_direction_lead",
            "completed_provisional_candidates_require_familywide_validation",
        }:
            raise ValueError("Existing scheduled-macro direction result is not terminal")
        return result
    freeze_result, freeze_document, catalog = load_frozen_inputs()
    rows, coverage = extract_direction_rows(catalog)
    expected_rows = len(catalog) * len(frozen.ASSETS) * len(frozen.HORIZONS_MINUTES)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} outcome rows, found {len(rows)}")
    summary = direction_summaries(rows)
    decisions = route_decisions(summary)
    families = family_decisions(decisions)
    candidate_count = int(decisions["candidate"].sum())
    status = (
        "completed_provisional_candidates_require_familywide_validation"
        if candidate_count
        else "completed_no_raw_direction_lead"
    )
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(rows, OUTCOME_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    g0.atomic_write_csv(families, FAMILY_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    REPORT_PATH.write_text(
        render_report(decisions, families, rows), encoding="utf-8", newline="\n"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "created_at_utc": g0.utc_now(),
        "outcomes_read": True,
        "profit_used": False,
        "event_rows": len(catalog),
        "outcome_rows": len(rows),
        "candidate_cells": candidate_count,
        "familywide_validation_required": bool(candidate_count),
        "frozen_input": freeze_result,
        "frozen_question": freeze_document["plain_question"],
        "artifacts": {
            "direction_rows": artifact(OUTCOME_PATH),
            "summary": artifact(SUMMARY_PATH),
            "decisions": artifact(DECISION_PATH),
            "family_decisions": artifact(FAMILY_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
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
                    "outcomes_will_be_read": True,
                    "families": len(frozen.SERIES_METHODS),
                    "assets": list(frozen.ASSETS),
                    "horizons_minutes": list(frozen.HORIZONS_MINUTES),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
