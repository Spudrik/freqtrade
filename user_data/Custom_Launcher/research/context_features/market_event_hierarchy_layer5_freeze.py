"""Freeze untouched future confirmation for the surviving event-hierarchy leads."""

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
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer3_direct as layer3,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer4_freeze as layer4,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
LAYER3_RESULT = layer3.OUTPUT_ROOT / "layer3_direct_result.json"
LAYER4_RESULT = layer4.OUTPUT_ROOT / "layer4_freeze_result.json"
OFFICIAL_EVENTS = layer1.OUTPUT_ROOT / "official_fomc_event_catalog.csv"
OUTPUT_ROOT = layer4.OUTPUT_ROOT.parent / "layer5_untouched_confirmation_20260904a"


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def validate_parents() -> tuple[dict[str, Any], dict[str, Any]]:
    layer3_result = json.loads(LAYER3_RESULT.read_text(encoding="utf-8"))
    layer4_result = json.loads(LAYER4_RESULT.read_text(encoding="utf-8"))
    if layer3_result.get("status") != "completed_event_hierarchy_layer3_pairwise_review":
        raise ValueError("Layer 3 pairwise review is not terminal.")
    if layer4_result.get("status") != "completed_layer4_freeze_and_coverage_gate":
        raise ValueError("Layer 4 coverage gate is not terminal.")
    if not layer4_result.get("layer4_terminal"):
        raise ValueError("Layer 4 still has an active chain requiring outcome testing.")
    for result in (layer3_result, layer4_result):
        for group in ("summary_artifacts", "detail_artifacts", "artifacts"):
            for item in result.get(group, {}).values():
                path = Path(item["path"])
                if item["sha256"] != g0.sha256_file(path):
                    raise ValueError(f"Parent artifact changed before Layer 5: {path}")
    return layer3_result, layer4_result


def prospective_official_events() -> DataFrame:
    events = pd.read_csv(OFFICIAL_EVENTS)
    events["anchor_utc"] = pd.to_datetime(events["anchor_utc"], utc=True)
    prospective = events.loc[
        events["whole_event_partition"].eq("prospective_untouched_after_freeze")
    ].copy()
    prospective["outcomes_opened"] = False
    prospective["confirmation_status"] = "waiting_for_release"
    columns = [
        "event_id",
        "event_family",
        "event_subfamily",
        "anchor_utc",
        "summary_of_economic_projections",
        "whole_event_partition",
        "outcomes_opened",
        "confirmation_status",
    ]
    if len(prospective) != 11:
        raise ValueError("The frozen official prospective event count is no longer 11.")
    return prospective[columns].sort_values("anchor_utc", kind="stable")


def confirmation_routes() -> DataFrame:
    records = [
        {
            "route_id": "future_fomc_activity",
            "parent_lead": "official_event_to_market_activity",
            "source": "official_fomc",
            "eligible_scopes": "btc,eth,established_alts",
            "horizons_hours": "1,2,4,8,24",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": "first 10 frozen future releases; no result-led skipping",
            "success_rule": (
                "event activity beats matched prior-state controls at least 55%, "
                "with components and scopes reported separately"
            ),
        },
        {
            "route_id": "future_gdelt_activity",
            "parent_lead": "gdelt_spike_to_market_activity",
            "source": "gdelt_activity_spike",
            "eligible_scopes": "btc,eth,established_alts",
            "horizons_hours": "1,2,4,8,24",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": (
                "first 10 non-colliding episodes selected by the frozen causal detector"
            ),
            "success_rule": (
                "activity beats matched prior-state and high-news controls at least 55%"
            ),
        },
        {
            "route_id": "future_fomc_negative_background_fade",
            "parent_lead": "background_plus_event_confirmation",
            "source": "official_fomc",
            "eligible_scopes": "btc",
            "horizons_hours": "4,24",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": (
                "first 10 releases with negative prior-30d BTC background and positive 4h move"
            ),
            "success_rule": (
                "at least 55% fade; at least +10 points versus event-only and +5 "
                "versus background-only; positive-background and shuffled controls retained"
            ),
        },
        {
            "route_id": "future_gdelt_negative_background_fade",
            "parent_lead": "background_plus_event_confirmation",
            "source": "gdelt_activity_spike",
            "eligible_scopes": "btc",
            "horizons_hours": "4,24",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": (
                "first 10 frozen-detector episodes with negative prior-30d BTC "
                "background and positive 4h move"
            ),
            "success_rule": (
                "at least 55% fade; same event-only, background-only, positive-background, "
                "and shuffled controls as Layer 3"
            ),
        },
        {
            "route_id": "future_gdelt_btc_single_level_activity",
            "parent_lead": "event_plus_local_technical_state",
            "source": "gdelt_activity_spike",
            "eligible_scopes": "btc",
            "horizons_hours": "1,2",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": "first 10 frozen-detector episodes at a causal single level",
            "success_rule": (
                "both adjacent horizons beat event-only, level-only, far, and shuffled controls"
            ),
        },
        {
            "route_id": "future_fomc_eth_single_level_activity",
            "parent_lead": "event_plus_local_technical_state",
            "source": "official_fomc",
            "eligible_scopes": "eth",
            "horizons_hours": "2,4",
            "minimum_qualifying_events": 10,
            "checkpoint_rule": "first 10 frozen future releases at a causal ETH single level",
            "success_rule": (
                "both adjacent horizons beat event-only, level-only, far, and shuffled controls"
            ),
        },
    ]
    routes = DataFrame.from_records(records)
    routes["qualifying_events_observed"] = 0
    routes["outcomes_opened"] = False
    routes["status"] = "waiting_for_unseen_events"
    return routes


def freeze_document(
    official: DataFrame,
    routes: DataFrame,
    layer3_result: dict[str, Any],
    layer4_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_layer5_untouched_confirmation_waiting",
        "created_at_utc": g0.utc_now(),
        "outcomes_opened": False,
        "official_future_event_count": len(official),
        "confirmation_route_count": len(routes),
        "common_rules": {
            "whole_events_only": True,
            "minimum_success_rate": 0.55,
            "main_target": 0.65,
            "profit_or_trade_return_target": False,
            "unsigned_news_direction": False,
            "abstain_when_source_or_confirmation_is_missing": True,
            "no_optional_stopping": (
                "Open results only at the frozen route checkpoint; retain every qualifying "
                "episode in arrival order."
            ),
            "source_failure": (
                "A missing or late source creates an abstention/coverage row, not guessed data."
            ),
        },
        "direction_boundary": (
            "No route predicts direction from unsigned news. The fade routes begin only "
            "after a positive four-hour market move has been observed."
        ),
        "not_carried_forward": [
            "isolated first-hour continuation cells",
            "isolated prior-month range-edge cells",
            "named-leader routes",
            "full top-ten meme amplification",
            "three-block chains with fewer than ten complete events",
            "post-event stable-range routes",
        ],
        "parents": {
            "layer3_result": artifact(LAYER3_RESULT),
            "layer4_result": artifact(LAYER4_RESULT),
            "layer3_retained_route_count": int(layer3_result["retained_route_count"]),
            "layer4_active_chain_count": int(layer4_result["active_chain_count"]),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "official_event_catalog": artifact(OFFICIAL_EVENTS),
        },
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer5_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "frozen_layer5_untouched_confirmation_waiting":
            raise ValueError("Existing Layer 5 freeze result is not terminal.")
        return result
    layer3_result, layer4_result = validate_parents()
    official = prospective_official_events()
    routes = confirmation_routes()
    freeze = freeze_document(official, routes, layer3_result, layer4_result)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "official_events": OUTPUT_ROOT / "layer5_prospective_official_events.csv",
        "routes": OUTPUT_ROOT / "layer5_confirmation_routes.csv",
        "freeze": OUTPUT_ROOT / "layer5_freeze.json",
    }
    g0.atomic_write_csv(official, paths["official_events"])
    g0.atomic_write_csv(routes, paths["routes"])
    g0.atomic_write_json(freeze, paths["freeze"])
    result = {
        "schema_version": 1,
        "status": "frozen_layer5_untouched_confirmation_waiting",
        "created_at_utc": g0.utc_now(),
        "outcomes_opened": False,
        "official_future_event_count": len(official),
        "confirmation_route_count": len(routes),
        "current_stage_terminal_until_new_events": True,
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
        print(json.dumps({"status": "ready_not_executed", "outcomes_opened": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
