"""Freeze the complete Layer 3 event-hierarchy pairwise batch."""

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
    market_event_hierarchy_layer2_direct as layer2,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_freeze as layer2_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
LAYER2_RESULT = layer2.OUTPUT_ROOT / "layer2_direct_result.json"
LAYER2_DECISIONS = layer2.OUTPUT_ROOT / "route_decisions.csv"
LAYER2_LOCAL = layer2.OUTPUT_ROOT / "local_context_summary.csv"
OUTPUT_ROOT = layer2_freeze.OUTPUT_ROOT.parent / "layer3_pairwise_20260904a"

EVENT_SOURCES = ("official_fomc", "gdelt_activity_spike")
BROAD_SCOPES = ("btc", "eth", "established_alts", "memes")
BROAD_HORIZONS = (2, 4, 8, 24)
RANGE_ASSETS = ("btc", "eth")
RANGE_HORIZONS = (1, 2, 4, 8, 24)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def load_layer2() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(LAYER2_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_hierarchy_layer2_direct_review":
        raise ValueError("Layer 2 individual-link review is not terminal.")
    if result.get("profit_used") or result.get("event_direction_without_surprise_used"):
        raise ValueError("Layer 2 crossed the frozen research boundary.")
    for group in ("summary_artifacts", "detail_artifacts"):
        for item in result[group].values():
            path = Path(item["path"])
            if g0.sha256_file(path) != item["sha256"]:
                raise ValueError(f"Layer 2 artifact changed after review: {path}")
    decisions = pd.read_csv(LAYER2_DECISIONS)
    local = pd.read_csv(LAYER2_LOCAL)
    return result, decisions, local


def active_pair_matrix(local: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for source in EVENT_SOURCES:
        records.append(
            {
                "pair_id": f"background_event_confirmation__{source}",
                "pair_family": "background_plus_event_confirmation",
                "event_source": source,
                "scope": "btc",
                "horizon_hours": 24,
                "state": "negative_30d_background_after_positive_4h_move",
                "parent_evidence": "retained slow-background conditional link",
            }
        )
    for source in EVENT_SOURCES:
        for scope in BROAD_SCOPES:
            for horizon in BROAD_HORIZONS:
                records.append(
                    {
                        "pair_id": (
                            f"event_broad_confirmation__{source}__{scope}__{horizon}h"
                        ),
                        "pair_family": "event_plus_first_broad_response",
                        "event_source": source,
                        "scope": scope,
                        "horizon_hours": horizon,
                        "state": "confirmed_directional_first_hour",
                        "parent_evidence": (
                            "event activity retained; named leader not retained, so use "
                            "simultaneous broad confirmation only"
                        ),
                    }
                )
    passed_local = local.loc[_truthy(local["meets_55_floor_and_beats_far"])].copy()
    for row in passed_local.itertuples(index=False):
        records.append(
            {
                "pair_id": (
                    "event_local_state__"
                    f"{row.event_source}__{row.cohort}__{int(row.horizon_hours)}h__"
                    f"{row.local_level_state}"
                ),
                "pair_family": "event_plus_local_technical_state",
                "event_source": row.event_source,
                "scope": row.cohort,
                "horizon_hours": int(row.horizon_hours),
                "state": row.local_level_state,
                "parent_evidence": "retained Layer 2 local-state cell",
            }
        )
    for source in EVENT_SOURCES:
        for scope in RANGE_ASSETS:
            for horizon in RANGE_HORIZONS:
                records.append(
                    {
                        "pair_id": f"event_range_edge__{source}__{scope}__{horizon}h",
                        "pair_family": "event_plus_prior_30d_range_position",
                        "event_source": source,
                        "scope": scope,
                        "horizon_hours": horizon,
                        "state": "outer_20pct_of_prior_30d_range",
                        "parent_evidence": (
                            "retained local-level family and the predeclared range-position pair"
                        ),
                    }
                )
    output = DataFrame.from_records(records)
    if output["pair_id"].duplicated().any():
        raise ValueError("Layer 3 pair identifiers are not unique.")
    output["status"] = "frozen_ready"
    return output


def parked_pair_matrix() -> DataFrame:
    records = [
        {
            "pair_family": "background_plus_event_surprise",
            "status": "coverage_parked",
            "reason": "No reconstructable historical expectation-and-surprise sign exists.",
        },
        {
            "pair_family": "named_leader_plus_coin_group_breadth",
            "status": "parent_evidence_parked",
            "reason": "Layer 2 found simultaneous or absent movement, not a repeatable leader.",
        },
        {
            "pair_family": "named_leader_plus_coin_local_state",
            "status": "parent_evidence_parked",
            "reason": "A named leader did not survive Layer 2.",
        },
        {
            "pair_family": "named_leader_plus_pressure_attention_or_compression",
            "status": "parent_evidence_parked",
            "reason": "A named leader did not survive Layer 2.",
        },
        {
            "pair_family": "event_or_leader_plus_orderbook_disturbance",
            "status": "coverage_parked",
            "reason": (
                "Validated historical Bybit overlap contains only four FOMC and two "
                "selected GDELT episodes, below the ten-event floor."
            ),
        },
        {
            "pair_family": "post_event_range_plus_smaller_event_or_technical_area",
            "status": "parent_evidence_parked",
            "reason": "No Layer 2 post-event range scope passed the stable-range link.",
        },
    ]
    return DataFrame.from_records(records)


def freeze_document(
    layer2_result: dict[str, Any], active: DataFrame, parked: DataFrame
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_before_layer3_pair_outcomes",
        "created_at_utc": g0.utc_now(),
        "parent_layer": {
            "result": artifact(LAYER2_RESULT),
            "selection_note": (
                "Layer 2 results selected these questions. The 2021-2025 history is "
                "development evidence for Layer 3, not untouched confirmation."
            ),
        },
        "active_pair_count": len(active),
        "parked_pair_count": len(parked),
        "common_contract": {
            "minimum_whole_events": 10,
            "use_whole_event_partitions": True,
            "development_partition": "development_2021_2023",
            "internal_validation_partition": "internal_validation_2024_2025",
            "retention_floor": 0.55,
            "main_target": 0.65,
            "profit_or_trade_return_target": False,
            "signed_event_direction_without_surprise": False,
            "ablation_rule": (
                "The pair must beat each component alone on comparable rows and lose "
                "its claimed advantage when the second component is shuffled."
            ),
            "family_repetition_rule": (
                "A lone passing cell remains provisional. Layer 4 eligibility normally "
                "requires repetition at adjacent horizons or in both event sources."
            ),
        },
        "pair_definitions": {
            "background_plus_event_confirmation": {
                "market_confirmation": "BTC closes positive over the first 4 hours",
                "background": "causal BTC return over the preceding 30 days is negative",
                "outcome": "the positive 4-hour move has lost at least half by hour 24",
                "component_controls": [
                    "all confirmed positive event moves regardless of background",
                    "negative-background confirmed moves at matched ordinary times",
                    "positive-background event moves",
                    "background labels shuffled within chronological partition",
                ],
                "cell_pass": [
                    "at least 10 negative-background positive event moves",
                    "fade rate at least 55%",
                    "at least 10 percentage points above event-only",
                    "at least 5 percentage points above background-only controls",
                    "negative-versus-positive gap is positive in both periods",
                    "actual gap exceeds the 75th percentile of shuffled gaps",
                ],
            },
            "event_plus_first_broad_response": {
                "confirmation_cohort": [
                    "BTC",
                    "ETH",
                    "BNB",
                    "SOL",
                    "XRP",
                    "ADA",
                    "TRX",
                    "AVAX",
                    "LINK",
                ],
                "confirmation": (
                    "at least 6 available coins, at least 60% moving the same way, "
                    "and median first-hour activity at least 1.25 times its causal norm"
                ),
                "outcome": "same direction from the end of hour 1 to the chosen horizon",
                "horizons_hours": list(BROAD_HORIZONS),
                "component_controls": [
                    "all event times using the observed first-hour direction",
                    "the same confirmation at matched ordinary times",
                    "confirmation state and direction shuffled within period",
                ],
                "cell_pass": [
                    "at least 10 confirmed events and 10 confirmed controls",
                    "continuation at least 55%",
                    "at least 5 percentage points above both component controls",
                    "at least 50% in both chronological periods",
                    "actual rate exceeds the 75th percentile of shuffled rates",
                ],
            },
            "event_plus_local_technical_state": {
                "states": "only the 11 Layer 2 cells retained before this freeze",
                "outcome": "direction-neutral activity relative to causal local norms",
                "component_controls": [
                    "same event source away from the frozen levels",
                    "same local state at matched ordinary times",
                    "local-state labels shuffled within coin and period",
                ],
                "cell_pass": [
                    "at least 10 target events plus adequate far/control rows",
                    "event target beats far state with at least 55% paired success",
                    "event target-versus-far gap exceeds the ordinary-time gap",
                    "increment is positive in both periods",
                    "actual increment exceeds the 75th percentile of shuffled increments",
                ],
                "meme_scope_note": (
                    "Uses available members of the frozen top-ten list and cannot be called "
                    "full-cohort confirmation."
                ),
            },
            "event_plus_prior_30d_range_position": {
                "range": "prior 720 completed hourly highs and lows, minimum 360 hours",
                "edge": "event open is in the outer 20% of that causal range",
                "middle": "event open is between 20% and 80% of that range",
                "outcome": "direction-neutral activity relative to causal local norms",
                "component_controls": [
                    "same event source in the middle of its range",
                    "range-edge state at matched ordinary times",
                    "range-position labels shuffled within coin and period",
                ],
                "cell_pass": [
                    "at least 10 edge events, 10 middle events, and adequate controls",
                    "edge events beat their paired controls at least 55% of the time",
                    "event edge-versus-middle gap exceeds the ordinary-time gap",
                    "increment is positive in both periods",
                    "actual increment exceeds the 75th percentile of shuffled increments",
                ],
            },
        },
        "active_pair_families": sorted(active["pair_family"].unique().tolist()),
        "parked_pair_families": parked.to_dict(orient="records"),
        "parent_result_counts": {
            "retained_routes": int(layer2_result["retained_route_count"]),
            "parked_routes": int(layer2_result["parked_route_count"]),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "layer2_decisions": artifact(LAYER2_DECISIONS),
            "layer2_local_summary": artifact(LAYER2_LOCAL),
            "layer2_pair_metrics": layer2_result["detail_artifacts"]["pair_metrics"],
            "layer2_background_rows": layer2_result["detail_artifacts"][
                "background_details"
            ],
        },
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer3_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer3_pairwise_freeze":
            raise ValueError("Existing Layer 3 freeze result is not terminal.")
        return result
    layer2_result, _, local = load_layer2()
    active = active_pair_matrix(local)
    parked = parked_pair_matrix()
    freeze = freeze_document(layer2_result, active, parked)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    active_path = OUTPUT_ROOT / "layer3_active_pair_matrix.csv"
    parked_path = OUTPUT_ROOT / "layer3_parked_pair_matrix.csv"
    freeze_path = OUTPUT_ROOT / "layer3_freeze.json"
    g0.atomic_write_csv(active, active_path)
    g0.atomic_write_csv(parked, parked_path)
    g0.atomic_write_json(freeze, freeze_path)
    result = {
        "schema_version": 1,
        "status": "completed_layer3_pairwise_freeze",
        "created_at_utc": g0.utc_now(),
        "new_pair_outcomes_read": False,
        "active_pair_count": len(active),
        "parked_pair_count": len(parked),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "active_pairs": artifact(active_path),
            "parked_pairs": artifact(parked_path),
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
        print(json.dumps({"status": "ready_not_executed", "outcomes_read": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
