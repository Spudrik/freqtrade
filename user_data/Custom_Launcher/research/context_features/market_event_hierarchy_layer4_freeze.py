"""Freeze or coverage-park Layer 4 three-block event-hierarchy chains."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
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
    market_event_hierarchy_layer3_direct as layer3,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
LAYER3_RESULT = layer3.OUTPUT_ROOT / "layer3_direct_result.json"
LAYER4_CANDIDATES = layer3.OUTPUT_ROOT / "layer4_candidate_cells.csv"
OUTPUT_ROOT = layer3.frozen.OUTPUT_ROOT.parent / "layer4_three_block_20260904a"
MINIMUM_EVENTS = 10


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_layer3() -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(LAYER3_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_hierarchy_layer3_pairwise_review":
        raise ValueError("Layer 3 pairwise review is not terminal.")
    if result.get("profit_used") or result.get("event_direction_without_surprise_used"):
        raise ValueError("Layer 3 crossed the frozen research boundary.")
    for group in ("summary_artifacts", "detail_artifacts"):
        for item in result[group].values():
            path = Path(item["path"])
            if item["sha256"] != g0.sha256_file(path):
                raise ValueError(f"Layer 3 artifact changed after review: {path}")
    return result, pd.read_csv(LAYER4_CANDIDATES)


def parent_detail_paths() -> tuple[Path, Path]:
    parent = json.loads(layer3.frozen.LAYER2_RESULT.read_text(encoding="utf-8"))
    return (
        Path(parent["detail_artifacts"]["pair_metrics"]["path"]),
        Path(parent["detail_artifacts"]["background_details"]["path"]),
    )


def load_parent_support() -> tuple[DataFrame, DataFrame]:
    metrics_path, background_path = parent_detail_paths()
    metrics = pd.read_parquet(
        metrics_path,
        columns=[
            "event_id",
            "event_source",
            "whole_event_partition",
            "sample_type",
            "sample_anchor_utc",
            "pair",
            "horizon_hours",
            "signed_return",
            "local_level_state",
            "nearest_level",
            "nearest_level_family",
        ],
    )
    background = pd.read_parquet(
        background_path,
        columns=[
            "event_id",
            "event_source",
            "sample_type",
            "background",
        ],
    )
    metrics["sample_anchor_utc"] = pd.to_datetime(metrics["sample_anchor_utc"], utc=True)
    return metrics, background


def retained_local_groups(candidates: DataFrame) -> DataFrame:
    local = candidates.loc[
        candidates["pair_family"].eq("event_plus_local_technical_state")
    ].copy()
    if local.empty:
        return DataFrame(
            columns=["event_source", "scope", "state", "evidence_horizons"]
        )
    grouped = local.groupby(
        ["event_source", "scope", "state"], dropna=False, sort=False
    )["horizon_hours"].agg(
        lambda values: ",".join(str(int(float(value))) for value in sorted(values))
    )
    return grouped.rename("evidence_horizons").reset_index()


def chain_preflight(
    candidates: DataFrame, metrics: DataFrame, background: DataFrame
) -> tuple[DataFrame, DataFrame]:
    groups = retained_local_groups(candidates)
    background_event = background.loc[
        background["sample_type"].eq("event"),
        ["event_id", "event_source", "background"],
    ].drop_duplicates(["event_id", "event_source"])
    active: list[dict[str, Any]] = []
    parked: list[dict[str, Any]] = []
    scope_pair = {"btc": "BTC/USDT:USDT", "eth": "ETH/USDT:USDT"}
    for row in groups.itertuples(index=False):
        if row.scope not in scope_pair:
            parked.append(
                {
                    "chain_id": f"chain__{row.event_source}__{row.scope}__{row.state}",
                    "event_source": row.event_source,
                    "scope": row.scope,
                    "local_state": row.state,
                    "evidence_horizons": row.evidence_horizons,
                    "status": "coverage_parked",
                    "reason": "The retained cohort cell has no single-asset fade target.",
                }
            )
            continue
        asset = metrics.loc[
            metrics["event_source"].eq(row.event_source)
            & metrics["sample_type"].eq("event")
            & metrics["pair"].eq(scope_pair[row.scope])
            & metrics["horizon_hours"].eq(4)
        ].merge(
            background_event,
            on=["event_id", "event_source"],
            how="left",
            validate="one_to_one",
        )
        asset["positive_confirmation"] = asset["signed_return"].gt(0)
        asset["negative_background"] = asset["background"].eq(
            "negative_30d_background"
        )
        asset["at_local_state"] = asset["local_level_state"].eq(row.state)
        full = asset.loc[
            asset["positive_confirmation"]
            & asset["negative_background"]
            & asset["at_local_state"]
        ]
        counts = {
            "full_chain_event_count": len(full),
            "without_local_event_count": int(
                (asset["positive_confirmation"] & asset["negative_background"]).sum()
            ),
            "without_background_event_count": int(
                (asset["positive_confirmation"] & asset["at_local_state"]).sum()
            ),
            "without_confirmation_event_count": int(
                (asset["negative_background"] & asset["at_local_state"]).sum()
            ),
            "development_full_count": int(
                full["whole_event_partition"].eq("development_2021_2023").sum()
            ),
            "validation_full_count": int(
                full["whole_event_partition"].eq(
                    "internal_validation_2024_2025"
                ).sum()
            ),
        }
        record = {
            "chain_id": f"chain__{row.event_source}__{row.scope}__{row.state}",
            "event_source": row.event_source,
            "scope": row.scope,
            "local_state": row.state,
            "evidence_horizons": row.evidence_horizons,
            "target": (
                "After a positive four-hour move, at least half of that move is lost "
                "by hour 24."
            ),
            **counts,
        }
        enough = (
            counts["full_chain_event_count"] >= MINIMUM_EVENTS
            and counts["without_local_event_count"] >= MINIMUM_EVENTS
            and counts["without_background_event_count"] >= MINIMUM_EVENTS
            and counts["without_confirmation_event_count"] >= MINIMUM_EVENTS
            and counts["development_full_count"] >= 3
            and counts["validation_full_count"] >= 3
        )
        if enough:
            active.append({**record, "status": "frozen_ready"})
        else:
            parked.append(
                {
                    **record,
                    "status": "coverage_parked",
                    "reason": (
                        "Fewer than 10 complete full-chain events or fewer than three "
                        "in each chronological period."
                    ),
                }
            )
    return DataFrame.from_records(active), DataFrame.from_records(parked)


def local_component_attribution(candidates: DataFrame, metrics: DataFrame) -> DataFrame:
    groups = retained_local_groups(candidates)
    scope_pair = {"btc": "BTC/USDT:USDT", "eth": "ETH/USDT:USDT"}
    records: list[dict[str, Any]] = []
    for row in groups.itertuples(index=False):
        pair = scope_pair.get(row.scope)
        if pair is None:
            continue
        events = metrics.loc[
            metrics["event_source"].eq(row.event_source)
            & metrics["sample_type"].eq("event")
            & metrics["pair"].eq(pair)
            & metrics["horizon_hours"].eq(1)
            & metrics["local_level_state"].eq(row.state)
        ].drop_duplicates("event_id")
        total = len(events)
        for values, group in events.groupby(
            ["nearest_level_family", "nearest_level"], dropna=False, sort=False
        ):
            records.append(
                {
                    "event_source": row.event_source,
                    "scope": row.scope,
                    "local_state": row.state,
                    "evidence_horizons": row.evidence_horizons,
                    "nearest_level_family": values[0],
                    "nearest_level": values[1],
                    "event_count": len(group),
                    "share_of_local_events": len(group) / total if total else 0.0,
                }
            )
    return DataFrame.from_records(records)


def freeze_document(
    layer3_result: dict[str, Any], active: DataFrame, parked: DataFrame
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed_layer4_freeze_and_coverage_gate",
        "created_at_utc": g0.utc_now(),
        "new_chain_outcomes_read": False,
        "active_chain_count": len(active),
        "parked_chain_count": len(parked),
        "three_block_definition": {
            "block_1": "named scheduled event or causal aggregate-news activity spike",
            "block_2": "negative causal BTC 30-day background plus observed positive 4h move",
            "block_3": "retained single calculated-level state at event open",
            "outcome": "loss of at least half the positive 4h move by hour 24",
            "maximum_blocks": 3,
        },
        "coverage_gate": {
            "minimum_full_chain_events": MINIMUM_EVENTS,
            "minimum_each_immediate_ablation": MINIMUM_EVENTS,
            "minimum_each_chronological_period": 3,
            "rule": (
                "Do not open the new 24-hour chain outcome unless the full chain and "
                "every immediate ablation have adequate whole-event support."
            ),
        },
        "carried_parked_routes": [
            "first-hour broad direction: isolated Layer 3 cells only",
            "prior-month range position: isolated Layer 3 cells only",
            "named leader chains: no repeatable Layer 2 leader",
            "orderbook chain: only four FOMC and two GDELT overlap events",
            "post-event range chain: no stable Layer 2 range link",
        ],
        "parent": {
            "layer3_result": artifact(LAYER3_RESULT),
            "layer4_candidate_cells": artifact(LAYER4_CANDIDATES),
            "layer3_candidate_count": int(layer3_result["layer4_candidate_cell_count"]),
        },
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer4_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer4_freeze_and_coverage_gate":
            raise ValueError("Existing Layer 4 freeze result is not terminal.")
        return result
    layer3_result, candidates = load_layer3()
    metrics, background = load_parent_support()
    active, parked = chain_preflight(candidates, metrics, background)
    attribution = local_component_attribution(candidates, metrics)
    freeze = freeze_document(layer3_result, active, parked)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "active": OUTPUT_ROOT / "layer4_active_chains.csv",
        "parked": OUTPUT_ROOT / "layer4_parked_chains.csv",
        "attribution": OUTPUT_ROOT / "layer4_local_component_attribution.csv",
        "freeze": OUTPUT_ROOT / "layer4_freeze.json",
    }
    g0.atomic_write_csv(active, paths["active"])
    g0.atomic_write_csv(parked, paths["parked"])
    g0.atomic_write_csv(attribution, paths["attribution"])
    g0.atomic_write_json(freeze, paths["freeze"])
    result = {
        "schema_version": 1,
        "status": "completed_layer4_freeze_and_coverage_gate",
        "created_at_utc": g0.utc_now(),
        "new_chain_outcomes_read": False,
        "active_chain_count": len(active),
        "parked_chain_count": len(parked),
        "layer4_terminal": len(active) == 0,
        "artifacts": {name: artifact(path) for name, path in paths.items()},
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
        print(json.dumps({"status": "ready_not_executed", "outcomes_read": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
