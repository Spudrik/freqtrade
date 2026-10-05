from __future__ import annotations

# The freeze reads manifests and definitions only; keep imports deterministic.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_joint_review as g10r,
)


OUTPUT_ROOT = g10r.g10z.OUTPUT_ROOT
REVIEW_ROOT = OUTPUT_ROOT / "generation11_review"
FREEZE_PATH = REVIEW_ROOT / "g11_frozen_broad_conditional_batch_20260822a.json"
G10_REVIEW_PATH = (
    OUTPUT_ROOT
    / "generation10_review"
    / g10r.DEFAULT_REVIEW_ID
    / "g10_terminal_review.json"
)
G6_EVENT_MANIFEST = g6d.EVENT_MANIFEST

HORIZONS = (1, 2, 4, 8)
HORIZON_BANDS = {
    "short": (1, 2),
    "medium": (4, 8),
}
SOURCE_TIMEFRAMES = ("1h", "4h", "8h", "1d")
COHORTS = ("normal", "meme")
LEVEL_CONTROLS = ("matched_random_time", "price_shift", "stale_72h", "near_miss")
VALIDATION_PERIODS = {
    "normal": ("validation_early", "validation_late"),
    "meme": ("meme_validation_early", "meme_validation_late"),
}


@dataclass(frozen=True)
class RouteQuestion:
    route_id: str
    family: str
    plain_question: str
    causal_inputs: tuple[str, ...]
    outcomes: tuple[str, ...]
    comparison: str


ROUTES = (
    RouteQuestion(
        route_id="activity_displacement",
        family="local_ohlcv_participation",
        plain_question=(
            "Do calculated areas behave differently when current local activity is "
            "clearly quiet, ordinary, or active, and does that relationship persist "
            "from one to eight hours?"
        ),
        causal_inputs=(
            "relative volume",
            "volume acceleration magnitude",
            "absolute candle pressure",
            "pressure persistence",
        ),
        outcomes=("volume", "range", "absolute movement", "dwell", "crossings"),
        comparison=(
            "Actual calculated-area contacts versus matched ordinary times, shifted "
            "prices, 72-hour-old levels, and near misses inside each frozen activity state."
        ),
    ),
    RouteQuestion(
        route_id="trend_timeframe_interaction",
        family="local_trend_and_multi_timeframe_location",
        plain_question=(
            "Does local trend strength or momentum become useful when it agrees or "
            "conflicts with calculated areas from different timeframes?"
        ),
        causal_inputs=(
            "EMA slope",
            "moving-average separation",
            "return slope and acceleration",
            "ADX, RSI, and MACD",
            "1h/4h/8h/1d level source and cross-timeframe relationship",
        ),
        outcomes=("range", "absolute movement", "away/through path", "dwell"),
        comparison=(
            "Trend-aligned versus trend-opposed contacts, separately by source timeframe "
            "and against the complete location-control ladder."
        ),
    ),
    RouteQuestion(
        route_id="crypto_market_alignment",
        family="cross_market_regime",
        plain_question=(
            "When a coin approaches a calculated area, do local direction and the wider "
            "BTC, ETH, and cohort direction together distinguish continuation through the "
            "area from rejection away from it?"
        ),
        causal_inputs=(
            "local signed trend vote",
            "BTC 1h/4h/24h returns",
            "ETH 1h/4h/24h returns",
            "cohort breadth, dispersion, and absolute activity",
        ),
        outcomes=("reaction", "away/through direction", "joint reaction and direction"),
        comparison=(
            "Frozen local-only, market-only, consensus, majority-path, and always-through "
            "comparators on independent causal contacts."
        ),
    ),
    RouteQuestion(
        route_id="cluster_obstacle_geometry",
        family="level_cluster_and_room_geometry",
        plain_question=(
            "Do independent calculated-level agreements, opposing levels, and available "
            "room combine with the approach path to change reaction size or resolution?"
        ),
        causal_inputs=(
            "same- and different-mechanism cross-timeframe agreement",
            "opposing-side overlap",
            "isolated-versus-cluster state",
            "approach side and distance",
        ),
        outcomes=("absolute movement", "away/through path", "dwell", "crossings"),
        comparison=(
            "Clusters versus isolated contacts and opposing-obstacle versus open-side "
            "contacts, with source timeframe and control density kept explicit."
        ),
    ),
    RouteQuestion(
        route_id="external_calm_and_stress",
        family="news_and_orderbook_context",
        plain_question=(
            "Do source-ready news activity or BTC order-book pressure explain when local "
            "technical relationships work, fail, or become overwhelmed?"
        ),
        causal_inputs=(
            "GDELT quiet/middle/shock state and topic groups",
            "BTC order-book quiet/middle/active state, coverage, and pressure",
            "local activity state",
        ),
        outcomes=("volume", "range", "absolute movement", "away/through path"),
        comparison=(
            "Quiet versus active external states only where timestamp-safe historical "
            "coverage passes; unsupported cohort/period cells are parked, never filled."
        ),
    ),
    RouteQuestion(
        route_id="market_group_portability",
        family="coin_and_cohort_transfer",
        plain_question=(
            "Which reaction relationships repeat across rational coin groups, and which "
            "are only single-coin observations?"
        ),
        causal_inputs=(
            "BTC separate",
            "established altcoins",
            "smart-contract platforms",
            "payments and transfer coins",
            "frozen top-ten traded meme cohort",
        ),
        outcomes=("all direction-neutral outcomes", "joint reaction and direction"),
        comparison=(
            "Equal-coin group summaries plus per-coin evidence. BNB is an audit row and "
            "cannot pass alone; a group lead requires at least three rational peers."
        ),
    ),
    RouteQuestion(
        route_id="timeframe_horizon_persistence",
        family="source_timeframe_and_reaction_horizon",
        plain_question=(
            "Does a calculated area's source timeframe affect how quickly and how long "
            "the market reacts, without assuming that a higher timeframe has precedence?"
        ),
        causal_inputs=("1h/4h/8h/1d source timeframe", "level family and zone width"),
        outcomes=(
            "1h/2h short-band reaction",
            "4h/8h medium-band reaction",
            "time to movement",
            "dwell and crossings",
        ),
        comparison=(
            "Every source timeframe is scored across every horizon; a claim must repeat "
            "through an adjacent horizon band rather than selecting one favourable horizon."
        ),
    ),
)

DIRECTION_METHODS = (
    "development_majority_path",
    "always_through_approach_baseline",
    "local_ema_and_return_slope_vote",
    "local_five_indicator_vote",
    "btc_eth_and_cohort_direction_vote",
    "local_and_crypto_market_consensus",
    "active_market_consensus_only",
    "opposing_obstacle_rejection_else_consensus",
)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_sources() -> dict[str, Any]:
    terminal = json.loads(G10_REVIEW_PATH.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed_generation10_terminal_review":
        raise ValueError("Generation 10 terminal review is not complete.")
    if terminal["completion_gate"]["objective_resolved"] is not False:
        raise ValueError("Generation 10 did not leave an unresolved objective.")
    events = json.loads(G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    if events.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("Generation 6 causal event cache is not terminal.")
    if len(events.get("tasks", [])) != 20:
        raise ValueError("The normal-plus-meme event portfolio is incomplete.")
    if events["summary"].get("causal_timestamp_violations") != 0:
        raise ValueError("The causal event cache has timestamp violations.")
    return {
        "generation10_terminal_review": artifact(G10_REVIEW_PATH),
        "generation6_causal_event_manifest": artifact(G6_EVENT_MANIFEST),
    }


def validate_existing_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation11_broad_screen_outcomes":
        raise ValueError("Generation 11 freeze is not terminal.")
    for item in frozen["source_contracts"].values():
        source = Path(item["path"])
        if not source.is_file() or g0.sha256_file(source) != item["sha256"]:
            raise ValueError(f"Frozen Generation 11 source changed: {source}")
    return frozen


def freeze_generation11() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        return validate_existing_freeze()
    sources = validate_sources()
    frozen = {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "generation": 11,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation11_broad_screen_outcomes",
        "authorization": {
            "authorized_by_user_on": "2026-08-22",
            "scope": (
                "Continue broad investigation of combinations of data after Generation "
                "10, retaining the 55% useful-result floor and avoiding one-result tunnels."
            ),
            "ordinary_timestamp_trade_selection": False,
            "live_or_dry_run_change": False,
        },
        "evidence_label": "outcome_reused_exploratory_relationship_screen",
        "evidence_limit": (
            "The raw periods have been used by earlier questions. New combinations are "
            "frozen before this screen, but survivors require genuinely later confirmation."
        ),
        "cohorts": list(COHORTS),
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "horizons_hours": list(HORIZONS),
        "horizon_bands": {key: list(value) for key, value in HORIZON_BANDS.items()},
        "validation_periods": {
            key: list(value) for key, value in VALIDATION_PERIODS.items()
        },
        "level_controls": list(LEVEL_CONTROLS),
        "routes": [asdict(route) for route in ROUTES],
        "route_count": len(ROUTES),
        "direction_methods": list(DIRECTION_METHODS),
        "reaction_definition": {
            "price": (
                "Absolute path excursion reaches at least the larger of 0.5 prior ATR "
                "or the causal zone half-width."
            ),
            "volume": (
                "Mean future volume through the scored horizon is at least 1.25 times "
                "the causal trailing 24-candle median."
            ),
            "success": "Both price and volume conditions are true.",
        },
        "direction_definition": {
            "away": "The 0.5-ATR away threshold is reached first inside the horizon.",
            "through": "The 0.5-ATR through threshold is reached first inside the horizon.",
            "tie": "Both are reached on the same first candle.",
            "unresolved": "Neither threshold is reached inside the horizon.",
            "joint_success": (
                "An issued call succeeds only when the reaction definition is met and "
                "the away/through call is correct; ties and unresolved paths fail."
            ),
        },
        "direction_pass_rule": {
            "minimum_joint_success": 0.55,
            "main_joint_target": 0.65,
            "minimum_issued_calls": 100,
            "minimum_coins": 5,
            "minimum_coverage": 0.20,
            "required_periods": "both validation periods in the declared scope",
            "comparator_margin": 0.02,
            "strict_uncertainty_rule": (
                "Weekly-block bootstrap lower bound above 0.50 and above the strongest "
                "eligible simple comparator."
            ),
        },
        "direction_neutral_pass_rule": {
            "minimum_events_per_cell": 50,
            "minimum_group_coins": 5,
            "minimum_rational_subgroup_coins": 3,
            "period_rule": "same effect sign in both declared validation periods",
            "horizon_rule": (
                "same effect sign in both adjacent horizons of a frozen short or medium band"
            ),
            "control_rule": (
                "actual area must retain the relationship against matched ordinary time, "
                "shifted price, 72-hour-old level, and near-miss comparisons where applicable"
            ),
        },
        "sequencing": {
            "all_routes_preflight_before_outcomes": True,
            "all_routes_terminal_before_descendants": True,
            "result_inspired_questions_queue_only": True,
            "freqai_descendants_wait_for_joint_direct_review": True,
        },
        "runtime": {
            "maximum_workers": 4,
            "native_threads_per_worker": 1,
            "stop_if_source_integrity_fails": True,
        },
        "research_boundary": {
            "profit_used": False,
            "entries_or_exits_built": False,
            "strategy_promotion": False,
            "direction_only_at_causal_calculated_area_contacts": True,
            "single_coin_can_pass": False,
        },
        "source_contracts": sources,
    }
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the broad post-Generation-10 relationship screen."
    )
    _ = parser.parse_args(argv)
    result = freeze_generation11()
    print(
        json.dumps(
            {
                "status": result["status"],
                "routes": result["route_count"],
                "cohorts": result["cohorts"],
                "source_timeframes": result["source_timeframes"],
                "horizons_hours": result["horizons_hours"],
                "direction_methods": len(result["direction_methods"]),
                "freeze_path": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
