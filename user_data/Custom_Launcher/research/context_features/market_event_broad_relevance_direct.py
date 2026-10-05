"""Run the frozen breadth-first market-event activity screen."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as cpi_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.OUTPUT_ROOT / "broad_relevance_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "broad_relevance_freeze.json"
CATALOG_PATH = frozen.OUTPUT_ROOT / "broad_relevance_event_catalog.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
CPI_CATALOG_PATH = cpi_frozen.OUTPUT_ROOT / "cpi_release_catalog.csv"
FOMC_CATALOG_PATH = layer1.OUTPUT_ROOT / "official_fomc_event_catalog.csv"

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
CONTROL_SEARCH_WEEKS = 26
CONTROL_COUNT = 12
MINIMUM_CONTROLS = 6
SCHEDULED_CONTROL_EXCLUSION_HOURS = 24
TOPIC_CONTROL_EXCLUSION_HOURS = frozen.TOPIC_EPISODE_COOLDOWN_HOURS


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_broad_event_relevance_freeze":
        raise ValueError("Broad relevance freeze result is not terminal.")
    if freeze.get("status") != "frozen_broad_event_relevance_before_market_outcomes":
        raise ValueError("Broad relevance definitions are not frozen.")
    if freeze.get("outcomes_read") or result.get("outcomes_read"):
        raise ValueError("Broad relevance freeze unexpectedly read market outcomes.")
    for name, path in (("freeze", FREEZE_PATH), ("catalog", CATALOG_PATH)):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen broad relevance artifact changed: {path}")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    return freeze, catalog


def position_by_date(frame: DataFrame) -> dict[pd.Timestamp, int]:
    return {stamp: int(position) for position, stamp in enumerate(frame["date"])}


def window_metrics(frame: DataFrame, position: int, length: int) -> dict[str, float] | None:
    if position < 0 or position + length > len(frame):
        return None
    window = frame.iloc[position : position + length]
    if len(window) != length:
        return None
    opening = float(window.iloc[0]["open"])
    if not np.isfinite(opening) or opening <= 0:
        return None
    close_return = float(window.iloc[-1]["close"] / opening - 1.0)
    return {
        "abs_return": abs(close_return),
        "range": float((window["high"].max() - window["low"].min()) / opening),
        "volume": float(window["volume"].sum()),
    }


def minimum_event_distance_hours(
    anchors: Iterable[pd.Timestamp], target: pd.Timestamp
) -> float:
    distances = [abs((target - anchor).total_seconds()) / 3600 for anchor in anchors]
    return min(distances) if distances else np.inf


def prior_week_controls(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
    exclusion_hours: int,
) -> list[dict[str, float]]:
    controls: list[dict[str, float]] = []
    for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if (
            minimum_event_distance_hours(blocked_anchors, control_anchor)
            <= exclusion_hours
        ):
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        metrics = window_metrics(frame, position, horizon)
        if metrics is not None:
            controls.append(metrics)
        if len(controls) >= CONTROL_COUNT:
            break
    return controls


def activity_score(
    response: Mapping[str, float], controls: Sequence[Mapping[str, float]]
) -> tuple[float, dict[str, float]]:
    if len(controls) < MINIMUM_CONTROLS:
        return np.nan, {"abs_return": np.nan, "range": np.nan, "volume": np.nan}
    ratios: dict[str, float] = {}
    for key in ("abs_return", "range", "volume"):
        baseline = float(np.median([float(control[key]) for control in controls]))
        ratios[key] = (
            float(response[key]) / baseline
            if np.isfinite(baseline) and baseline > 0
            else np.nan
        )
    values = list(ratios.values())
    score = float(np.median(values)) if all(np.isfinite(values)) else np.nan
    return score, ratios


def optional_anchor_catalog(path: Path) -> list[pd.Timestamp]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path, usecols=["anchor_utc"])
    return list(pd.to_datetime(frame["anchor_utc"], utc=True))


def group_config(event_group: str) -> tuple[str, tuple[int, ...], int]:
    if event_group == "scheduled_release":
        return "1m", frozen.SCHEDULED_HORIZONS_MINUTES, SCHEDULED_CONTROL_EXCLUSION_HOURS
    if event_group == "gdelt_topic_activity":
        return "1h", frozen.TOPIC_HORIZONS_HOURS, TOPIC_CONTROL_EXCLUSION_HOURS
    raise ValueError(f"Unsupported event group: {event_group}")


def extract_outcomes(catalog: DataFrame) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    external_scheduled = optional_anchor_catalog(CPI_CATALOG_PATH) + optional_anchor_catalog(
        FOMC_CATALOG_PATH
    )

    for event_group, group_events in catalog.groupby("event_group", sort=False):
        timeframe, horizons, exclusion_hours = group_config(str(event_group))
        if event_group == "scheduled_release":
            blocked_common = list(group_events["anchor_utc"]) + external_scheduled
        else:
            blocked_common = []

        for pair in ASSETS:
            path = g0.ohlcv_path(pair, timeframe)
            frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
            frame = frame.drop_duplicates("date").reset_index(drop=True)
            positions = position_by_date(frame)
            coverage.append(
                {
                    "event_group": event_group,
                    "pair": pair,
                    "timeframe": timeframe,
                    "path": str(path.resolve()),
                    "sha256": g0.sha256_file(path),
                    "first_candle_utc": frame["date"].min(),
                    "last_candle_utc": frame["date"].max(),
                    "rows": len(frame),
                    "duplicate_timestamps_removed": 0,
                }
            )
            for event in group_events.itertuples(index=False):
                anchor = pd.Timestamp(event.anchor_utc)
                position = positions.get(anchor)
                if position is None:
                    continue
                blocked = (
                    blocked_common
                    if event_group == "scheduled_release"
                    else list(
                        group_events.loc[
                            group_events["event_family"].eq(event.event_family),
                            "anchor_utc",
                        ]
                    )
                )
                for horizon in horizons:
                    response = window_metrics(frame, position, horizon)
                    if response is None:
                        continue
                    controls = prior_week_controls(
                        frame,
                        positions,
                        anchor=anchor,
                        horizon=horizon,
                        blocked_anchors=blocked,
                        exclusion_hours=exclusion_hours,
                    )
                    score, ratios = activity_score(response, controls)
                    records.append(
                        {
                            "event_id": event.event_id,
                            "event_group": event_group,
                            "event_family": event.event_family,
                            "event_label": event.event_label,
                            "anchor_utc": anchor,
                            "whole_event_partition": event.whole_event_partition,
                            "pair": pair,
                            "timeframe": timeframe,
                            "horizon": int(horizon),
                            "horizon_unit": "minutes" if timeframe == "1m" else "hours",
                            "control_count": len(controls),
                            "activity_score": score,
                            "above_control_median": (
                                bool(score > 1.0) if np.isfinite(score) else pd.NA
                            ),
                            "abs_return_ratio": ratios["abs_return"],
                            "range_ratio": ratios["range"],
                            "volume_ratio": ratios["volume"],
                        }
                    )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def summarize(outcomes: DataFrame) -> DataFrame:
    usable = outcomes.loc[outcomes["activity_score"].notna()].copy()
    if usable.empty:
        return DataFrame()
    summary = (
        usable.groupby(
            [
                "event_group",
                "event_family",
                "event_label",
                "pair",
                "timeframe",
                "horizon",
                "horizon_unit",
                "whole_event_partition",
            ],
            dropna=False,
        )
        .agg(
            events=("event_id", "nunique"),
            median_activity_score=("activity_score", "median"),
            above_control_rate=("above_control_median", "mean"),
            median_abs_return_ratio=("abs_return_ratio", "median"),
            median_range_ratio=("range_ratio", "median"),
            median_volume_ratio=("volume_ratio", "median"),
        )
        .reset_index()
    )
    summary["partition_pass"] = (
        summary["events"].ge(frozen.MINIMUM_PARTITION_EVENTS)
        & summary["above_control_rate"].gt(frozen.MINIMUM_ABOVE_CONTROL_RATE)
        & summary["median_activity_score"].ge(frozen.MINIMUM_MEDIAN_ACTIVITY_SCORE)
    )
    return summary


def classify_families(summary: DataFrame, catalog: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    required_partitions = {
        "development_2021_2023",
        "internal_validation_2024_2025",
    }
    for family_row in (
        catalog[["event_group", "event_family", "event_label"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        family = summary.loc[summary["event_family"].eq(family_row.event_family)]
        qualifying: dict[str, list[str]] = {pair: [] for pair in ASSETS}
        sufficient_partitions = False
        for pair in ASSETS:
            pair_rows = family.loc[family["pair"].eq(pair)]
            for (horizon, unit), horizon_rows in pair_rows.groupby(
                ["horizon", "horizon_unit"], sort=True
            ):
                partitions = set(horizon_rows["whole_event_partition"])
                if not required_partitions.issubset(partitions):
                    continue
                required = horizon_rows.loc[
                    horizon_rows["whole_event_partition"].isin(required_partitions)
                ]
                if required["events"].ge(frozen.MINIMUM_PARTITION_EVENTS).all():
                    sufficient_partitions = True
                if len(required) == 2 and required["partition_pass"].all():
                    qualifying[pair].append(f"{int(horizon)} {unit}")

        markets_passing = sum(bool(values) for values in qualifying.values())
        if not sufficient_partitions:
            verdict = "coverage_limited"
        elif markets_passing == len(ASSETS):
            verdict = "repeatable_activity_lead_both_markets"
        elif markets_passing == 1:
            verdict = "narrow_activity_lead_one_market"
        else:
            verdict = "weak_or_inconsistent"
        records.append(
            {
                "event_group": family_row.event_group,
                "event_family": family_row.event_family,
                "event_label": family_row.event_label,
                "verdict": verdict,
                "btc_qualifying_horizons": "; ".join(qualifying[ASSETS[0]]),
                "eth_qualifying_horizons": "; ".join(qualifying[ASSETS[1]]),
                "interpretation_limit": (
                    "Broad activity screen only; no direction, causation, profit, or "
                    "promotion claim."
                ),
            }
        )
    return DataFrame.from_records(records).sort_values(
        ["event_group", "event_family"], kind="stable"
    )


def render_report(verdicts: DataFrame, outcomes: DataFrame) -> str:
    rows = [
        "# Broad Event Relevance Result",
        "",
        "All eleven frozen families completed before interpretation.",
        "CPI remained parked and was not re-tested.",
        "",
        "| Idea | Result | BTC windows | ETH windows |",
        "|---|---|---|---|",
    ]
    for row in verdicts.itertuples(index=False):
        rows.append(
            f"| {row.event_label} | `{row.verdict}` | "
            f"{row.btc_qualifying_horizons or '-'} | "
            f"{row.eth_qualifying_horizons or '-'} |"
        )
    rows.extend(
        [
            "",
            f"- Whole event/horizon observations: `{len(outcomes)}`",
            "- Meaning: a lead says activity was repeatedly larger than earlier same-clock weeks.",
            "- It does not say which direction price moved or prove the named event caused it.",
            "",
        ]
    )
    return "\n".join(rows)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "broad_relevance_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_broad_event_relevance_screen":
            raise ValueError("Existing broad relevance result is not terminal.")
        return result

    freeze, catalog = load_frozen_inputs()
    outcomes, coverage = extract_outcomes(catalog)
    summary = summarize(outcomes)
    verdicts = classify_families(summary, catalog)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes_path = OUTPUT_ROOT / "broad_relevance_event_outcomes.csv"
    summary_path = OUTPUT_ROOT / "broad_relevance_summary.csv"
    verdicts_path = OUTPUT_ROOT / "broad_relevance_verdicts.csv"
    coverage_path = OUTPUT_ROOT / "broad_relevance_market_coverage.csv"
    report_path = OUTPUT_ROOT / "broad_relevance_report.md"
    g0.atomic_write_csv(outcomes, outcomes_path)
    g0.atomic_write_csv(summary, summary_path)
    g0.atomic_write_csv(verdicts, verdicts_path)
    g0.atomic_write_csv(coverage, coverage_path)
    report_path.write_text(
        render_report(verdicts, outcomes), encoding="utf-8", newline="\n"
    )
    result = {
        "schema_version": 1,
        "status": "completed_broad_event_relevance_screen",
        "created_at_utc": g0.utc_now(),
        "freeze_sha256": g0.sha256_file(FREEZE_PATH),
        "families_completed": int(verdicts["event_family"].nunique()),
        "expected_families": len(frozen.SCHEDULED_FAMILIES) + len(frozen.TOPIC_FAMILIES),
        "verdict_counts": verdicts["verdict"].value_counts().to_dict(),
        "cpi_status": "parked_not_retested",
        "branching_status": "none_until_complete_batch_review",
        "screening_rule": freeze["screening_rule"],
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
                    "status": "ready_not_executed",
                    "families": len(frozen.SCHEDULED_FAMILIES)
                    + len(frozen.TOPIC_FAMILIES),
                    "cpi": "parked_not_retested",
                    "output_root": str(OUTPUT_ROOT),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
