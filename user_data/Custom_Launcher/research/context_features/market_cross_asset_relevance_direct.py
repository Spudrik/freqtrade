"""Run the frozen cross-market activity screen across BTC and ETH."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_cross_asset_relevance_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as shared,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.OUTPUT_ROOT / "cross_asset_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "cross_asset_freeze.json"
CATALOG_PATH = frozen.OUTPUT_ROOT / "cross_asset_event_catalog.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
CONTROL_EXCLUSION_HOURS = 168


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_cross_asset_relevance_freeze":
        raise ValueError("Cross-asset freeze result is not terminal.")
    if freeze.get("status") != "frozen_cross_asset_relevance_before_crypto_outcomes":
        raise ValueError("Cross-asset definitions are not frozen.")
    if freeze.get("outcomes_read") or result.get("outcomes_read"):
        raise ValueError("Cross-asset freeze unexpectedly read crypto outcomes.")
    for name, path in (("freeze", FREEZE_PATH), ("catalog", CATALOG_PATH)):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen cross-asset artifact changed: {path}")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    return freeze, catalog


def extract_outcomes(catalog: DataFrame) -> tuple[DataFrame, DataFrame]:
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in shared.ASSETS:
        path = g0.ohlcv_path(pair, "1h")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicate_count = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = shared.position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1h",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
                "duplicate_timestamps_removed": duplicate_count,
            }
        )
        for family, family_events in catalog.groupby("event_family", sort=False):
            blocked = list(family_events["anchor_utc"])
            for event in family_events.itertuples(index=False):
                anchor = pd.Timestamp(event.anchor_utc)
                position = positions.get(anchor)
                if position is None:
                    continue
                for horizon in frozen.HORIZONS_HOURS:
                    response = shared.window_metrics(frame, position, horizon)
                    if response is None:
                        continue
                    controls = shared.prior_week_controls(
                        frame,
                        positions,
                        anchor=anchor,
                        horizon=horizon,
                        blocked_anchors=blocked,
                        exclusion_hours=CONTROL_EXCLUSION_HOURS,
                    )
                    score, ratios = shared.activity_score(response, controls)
                    records.append(
                        {
                            "event_id": event.event_id,
                            "event_family": family,
                            "event_label": event.event_label,
                            "anchor_utc": anchor,
                            "whole_event_partition": event.whole_event_partition,
                            "source_change": event.source_change,
                            "expected_crypto_relation": event.expected_crypto_relation,
                            "pair": pair,
                            "horizon": horizon,
                            "control_count": len(controls),
                            "activity_score": score,
                            "above_control_median": (
                                bool(score > 1.0) if pd.notna(score) else pd.NA
                            ),
                            "abs_return_ratio": ratios["abs_return"],
                            "range_ratio": ratios["range"],
                            "volume_ratio": ratios["volume"],
                        }
                    )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def summarize(outcomes: DataFrame) -> DataFrame:
    usable = outcomes.loc[outcomes["activity_score"].notna()].copy()
    summary = (
        usable.groupby(
            [
                "event_family",
                "event_label",
                "pair",
                "horizon",
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


def classify(summary: DataFrame, catalog: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    partitions = {"development_2021_2023", "internal_validation_2024_2025"}
    for event in catalog[["event_family", "event_label"]].drop_duplicates().itertuples(
        index=False
    ):
        family = summary.loc[summary["event_family"].eq(event.event_family)]
        qualifying: dict[str, list[str]] = {pair: [] for pair in shared.ASSETS}
        sufficient = False
        for pair in shared.ASSETS:
            for horizon, rows in family.loc[family["pair"].eq(pair)].groupby("horizon"):
                if not partitions.issubset(set(rows["whole_event_partition"])):
                    continue
                required = rows.loc[rows["whole_event_partition"].isin(partitions)]
                if required["events"].ge(frozen.MINIMUM_PARTITION_EVENTS).all():
                    sufficient = True
                if len(required) == 2 and required["partition_pass"].all():
                    qualifying[pair].append(f"{int(horizon)} hours")
        passing_markets = sum(bool(values) for values in qualifying.values())
        if not sufficient:
            verdict = "coverage_limited"
        elif passing_markets == len(shared.ASSETS):
            verdict = "repeatable_activity_lead_both_markets"
        elif passing_markets == 1:
            verdict = "narrow_activity_lead_one_market"
        else:
            verdict = "weak_or_inconsistent"
        records.append(
            {
                "event_family": event.event_family,
                "event_label": event.event_label,
                "verdict": verdict,
                "btc_qualifying_horizons": "; ".join(qualifying[shared.ASSETS[0]]),
                "eth_qualifying_horizons": "; ".join(qualifying[shared.ASSETS[1]]),
                "direction_status": "not_tested_in_this_batch",
            }
        )
    return DataFrame.from_records(records).sort_values("event_family", kind="stable")


def render_report(verdicts: DataFrame) -> str:
    rows = [
        "# Cross-Asset Relevance Result",
        "",
        "All seven frozen wider-market families completed before interpretation.",
        "",
        "| Wider-market idea | Result | BTC windows | ETH windows |",
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
            "This screen tests unusual crypto activity only. Direction remains unopened.",
            "",
        ]
    )
    return "\n".join(rows)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "cross_asset_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_cross_asset_relevance_screen":
            raise ValueError("Existing cross-asset result is not terminal.")
        return result
    freeze, catalog = load_frozen_inputs()
    outcomes, coverage = extract_outcomes(catalog)
    summary = summarize(outcomes)
    verdicts = classify(summary, catalog)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes_path = OUTPUT_ROOT / "cross_asset_outcomes.csv"
    summary_path = OUTPUT_ROOT / "cross_asset_summary.csv"
    verdicts_path = OUTPUT_ROOT / "cross_asset_verdicts.csv"
    coverage_path = OUTPUT_ROOT / "cross_asset_market_coverage.csv"
    report_path = OUTPUT_ROOT / "cross_asset_report.md"
    g0.atomic_write_csv(outcomes, outcomes_path)
    g0.atomic_write_csv(summary, summary_path)
    g0.atomic_write_csv(verdicts, verdicts_path)
    g0.atomic_write_csv(coverage, coverage_path)
    report_path.write_text(render_report(verdicts), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_cross_asset_relevance_screen",
        "created_at_utc": g0.utc_now(),
        "freeze_sha256": g0.sha256_file(FREEZE_PATH),
        "families_completed": int(verdicts["event_family"].nunique()),
        "expected_families": len(frozen.MARKET_FAMILIES),
        "verdict_counts": verdicts["verdict"].value_counts().to_dict(),
        "direction_status": "not_tested_in_this_batch",
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
                    "families": len(frozen.MARKET_FAMILIES),
                    "direction": "not_tested",
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
