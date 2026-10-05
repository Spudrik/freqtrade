"""Freeze Generation 15's broad direct reaction-path attribution batch.

The freeze reads only terminal summaries and source manifests.  It names the complete
sibling set, controls, targets, support rules, and interpretation before any Generation
15 outcome column is opened.
"""

from __future__ import annotations

# Bind numerical pools before pandas/FreqAI-adjacent imports.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as g11z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)


G14_JOINT_REVIEW = (
    g14z.FREEZE_PATH.parent
    / "g14_broad_combinations"
    / "freqai"
    / "g14_joint_review_20260822a"
    / "g14_joint_review.json"
)
G13_DIRECT_RESULT = (
    g13d.RECORD_ROOT
    / g13d.DEFAULT_RUN_ID
    / "g13_direct_control_result.json"
)
FREEZE_PATH = (
    g14z.FREEZE_PATH.parent
    / "g14_broad_combinations"
    / "g15_broad_direct_freeze_20260822a.json"
)

HORIZONS = tuple(g11z.HORIZONS)
CONTROLS = ("matched_random_time", "near_miss")
LEVEL_FAMILIES = (
    "confirmed_swing",
    "generic_prior_range",
    "generic_round_number",
    "tlv2_forecast_zone",
    "tlv2_ranked",
    "volume_profile_explicit_prior",
    "volume_profile_nodes",
    "volume_profile_settled",
)
SOURCE_TIMEFRAMES = tuple(g11z.SOURCE_TIMEFRAMES)
PATH_METRICS = (
    "absolute_excursion_atr",
    "range_ratio",
    "volume_ratio",
    "absolute_pressure_change",
    "dwell_fraction",
    "crossings",
    "hit_0_5atr",
    "censored_time_to_0_5atr",
)
NORMAL_GROUPS = {
    "btc_separate": ("BTC/USDT:USDT",),
    "smart_contract_platforms": (
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "BNB/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
    ),
    "other_established_alts": (
        "DOGE/USDT:USDT",
        "LINK/USDT:USDT",
        "TRX/USDT:USDT",
        "XRP/USDT:USDT",
    ),
}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_sources() -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not G14_JOINT_REVIEW.is_file():
        raise FileNotFoundError(G14_JOINT_REVIEW)
    g14 = json.loads(G14_JOINT_REVIEW.read_text(encoding="utf-8"))
    if g14.get("status") != "completed_generation14_joint_review":
        raise ValueError("Generation 14 joint review is not terminal.")
    if not g14.get("all_frozen_siblings_completed_or_parked_before_review"):
        raise ValueError("Generation 14 sibling gate is incomplete.")
    if not G13_DIRECT_RESULT.is_file():
        raise FileNotFoundError(G13_DIRECT_RESULT)
    direct = json.loads(G13_DIRECT_RESULT.read_text(encoding="utf-8"))
    if direct.get("status") != "completed_generation13_direct_controls":
        raise ValueError("Generation 13 matched-control source is not terminal.")
    source = Path(direct["source_contracts"]["generation6_event_manifest"]["path"])
    expected = direct["source_contracts"]["generation6_event_manifest"]["sha256"]
    if not source.is_file() or g0.sha256_file(source) != expected:
        raise ValueError("Frozen Generation 6 event manifest changed.")
    return g14, direct, source


def sibling_routes() -> list[dict[str, Any]]:
    return [
        {
            "route_id": "reaction_path_decomposition",
            "plain_question": (
                "When price reaches a real calculated area, which separate behaviours "
                "change versus equally active ordinary times and genuine near misses?"
            ),
            "surface": "all adequately supported calculated contacts",
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
            "controls": list(CONTROLS),
        },
        {
            "route_id": "level_family_attribution",
            "plain_question": (
                "Which actual level families repeatedly locate price, volume, pressure, "
                "volatility, dwell, crossing, or timing reactions?"
            ),
            "families": list(LEVEL_FAMILIES),
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
            "controls": list(CONTROLS),
        },
        {
            "route_id": "source_timeframe_attribution",
            "plain_question": (
                "Does a completed 1h, 4h, 8h, or 1d source level change the kind or "
                "timing of reaction, without granting higher timeframe automatic priority?"
            ),
            "source_timeframes": list(SOURCE_TIMEFRAMES),
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
            "controls": list(CONTROLS),
        },
        {
            "route_id": "contact_activity_combinations",
            "plain_question": (
                "After the contact candle closes, do unusually high volume plus range, "
                "or high volume plus absolute pressure change, add information beyond "
                "either component and quiet-contact controls?"
            ),
            "states": {
                "high_contact_volume": "contact_volume_ratio >= 1.25",
                "high_contact_range": "contact_range_ratio >= 1.25",
                "high_absolute_pressure_change": "abs(contact_pressure_change) >= 0.50",
            },
            "combinations": [
                "high_contact_volume AND high_contact_range",
                "high_contact_volume AND high_absolute_pressure_change",
            ],
            "component_ablations_required": True,
            "targets_begin_after_contact_candle": True,
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
        },
        {
            "route_id": "contact_lifecycle_and_approach",
            "plain_question": (
                "Do first or repeated contacts, time since the prior same-level contact, "
                "or fast versus slow arrival change the direction-neutral reaction path?"
            ),
            "causal_states": {
                "fresh_30d": "no earlier same level_identity contact in the prior 30d",
                "repeat_within_24h": "earlier same level_identity contact within 24h",
                "repeat_1d_to_7d": "earlier same level_identity contact 24h-7d ago",
                "repeat_7d_to_30d": "earlier same level_identity contact 7d-30d ago",
                "fast_arrival": "pre_distance_atr >= 1.0",
                "slow_arrival": "pre_distance_atr <= 0.25",
            },
            "left_censor_warmup_days": 30,
            "controls": ["matched alternative lifecycle state", "within-period shuffled state"],
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
        },
        {
            "route_id": "explicit_cluster_composition",
            "plain_question": (
                "Do independently produced, cross-timeframe, same-mechanism, or opposing "
                "clusters change reaction paths beyond matched isolated calculated levels?"
            ),
            "cluster_states": [
                "same_mechanism_agreement",
                "different_mechanism_agreement",
                "any_cross_timeframe_cluster",
                "opposing_side_overlap",
            ],
            "control": "matched isolated actual level contact",
            "component_independence_preserved": True,
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
        },
        {
            "route_id": "historical_btc_orderbook_conditioning",
            "plain_question": (
                "On periods with genuine timestamp-safe coverage, does active versus quiet "
                "historical BTC order-book pressure alter reactions at calculated areas "
                "after local and wider-crypto state are matched?"
            ),
            "coverage_scope": (
                "older normal validation, first later-normal period, and meme periods only; "
                "newest later-normal confirmation is unavailable"
            ),
            "states": ["regime_active", "regime_quiet"],
            "controls": ["matched quiet/current state", "causal_72h-old", "within-period shuffled"],
            "missing_is_zero": False,
            "classification_if_positive": "exploratory_source_limited_lead",
            "metrics": list(PATH_METRICS),
            "horizons_hours": list(HORIZONS),
        },
    ]


def build_freeze() -> dict[str, Any]:
    g14, direct, source_manifest = validate_sources()
    routes = sibling_routes()
    return {
        "schema_version": 1,
        "generation": 15,
        "status": "frozen_before_generation15_outcomes",
        "plain_objective": (
            "Keep the search broad while separating what physically changes after a "
            "calculated-area contact and which level, timeframe, contact, cluster, coin, "
            "or supported order-book conditions contribute incremental information."
        ),
        "user_authorization": (
            "2026-08-22 explicit approval to continue broad combinations research after "
            "confirming multiple coins, horizons, timeframes, and inputs remain in scope"
        ),
        "parent_interpretation": {
            "short_location_effect": "retained at 1h-4h across adjacent reaction definitions",
            "level_identity_h1": "weak point lead requiring attribution",
            "broad_bundles": "not retained",
            "long_local_plus_8h": "parked after failed confirmation",
            "direction_or_profit": "not tested",
        },
        "source_contracts": {
            "generation14_joint_review": artifact(G14_JOINT_REVIEW),
            "generation13_direct_controls": artifact(G13_DIRECT_RESULT),
            "generation6_event_manifest": artifact(source_manifest),
        },
        "source_summary": {
            "generation14_supported_model_commands": 89,
            "generation13_matched_rows_before_horizon_purges": int(
                direct["summary"]["matched_rows_before_horizon_purges"]
            ),
            "generation14_all_siblings_terminal": bool(
                g14["all_frozen_siblings_completed_or_parked_before_review"]
            ),
            "generation15_outcome_values_read": False,
        },
        "cohorts": ["normal", "meme"],
        "normal_groups_predeclared": {
            key: list(value) for key, value in NORMAL_GROUPS.items()
        },
        "meme_scope": "unchanged frozen top-ten traded meme cohort",
        "horizons_hours": list(HORIZONS),
        "path_metrics": list(PATH_METRICS),
        "sibling_routes": routes,
        "sibling_count": len(routes),
        "matching_contract": {
            "future_outcomes_used_for_matching": False,
            "exact_keys": [
                "pair",
                "analysis period",
                "source timeframe",
                "level family where applicable",
                "approach state",
            ],
            "continuous_state_match": list(g13d.MATCH_FEATURES),
            "pre_distance_atr_caliper": 0.10,
            "maximum_control_reuse": 3,
            "minimum_event_separation_hours": "equal to each tested horizon",
            "control_times_at_any_real_contact_removed": True,
        },
        "support_rules": {
            "minimum_independent_rows_per_pair_cell": 20,
            "minimum_coins_for_broad_or_group_cell": 5,
            "btc_one_coin_results": "asset-specific only",
            "insufficient_cells": "park without pooling or threshold relaxation",
            "orderbook": "at least 50 eligible rows and five coins in each claimed period",
        },
        "decision_rules": {
            "point": (
                "The paired equal-coin effect has one consistent sign in both periods and "
                "against every declared control, with at least five adequately supported "
                "coins; dwell may retain either predeclared sign but may not switch sign."
            ),
            "strict": (
                "The point ladder also keeps weekly-block bootstrap uncertainty entirely "
                "on the same side of zero and is not dominated by one coin."
            ),
            "combination": (
                "A contact or cluster combination must beat its quiet/matched control and "
                "every immediately simpler component; more conditions receive no credit "
                "for merely changing the sample."
            ),
            "portability": (
                "Report normal, later normal, meme, BTC, smart-contract-platform, and other "
                "established-alt cells separately; coherent smaller groups are valid but "
                "must retain their declared scope."
            ),
        },
        "sequencing": {
            "complete_sibling_set_frozen_before_any_generation15_outcome": True,
            "all_siblings_terminal_or_honestly_parked_before_joint_review": True,
            "no_generation16_descendant_before_joint_review": True,
            "automatic_descendant_launch": False,
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_tested": False,
            "joint_55_percent_direction_target_reached": False,
            "trading_promotion": False,
        },
        "runtime": {
            "maximum_workers": 4,
            "worker_threads_each": 1,
            "bulky_artifacts_root": (
                "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
                "generation15_branches"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FREEZE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = build_freeze()
    g0.atomic_write_json(freeze, args.output)
    print(json.dumps({
        "status": freeze["status"],
        "sibling_count": freeze["sibling_count"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
