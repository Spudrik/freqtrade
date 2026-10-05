"""Freeze the next breadth-first whole-event conditional investigation batch."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_simple_signal_families as simple,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_conditional_episode_batch_20260909a"
SOURCE_ROOT = simple.SOURCE_ROOT
OUTPUT_ROOT = SOURCE_ROOT / RUN_ID
FREEZE_PATH = OUTPUT_ROOT / "conditional_episode_freeze.json"
REANALYSIS_PATH = (
    SOURCE_ROOT
    / "event_whole_episode_reanalysis_20260909a"
    / "whole_episode_reanalysis_result.json"
)
CACHE_MANIFEST_PATH = simple.SOURCE_CACHE_MANIFEST_PATH
MEME_ROOT = SOURCE_ROOT / "event_meme_transmission_20260909a"
MEME_RESULT_PATH = MEME_ROOT / "meme_transmission_result.json"
MEME_VALIDATION_PATH = MEME_ROOT / "meme_transmission_direction_validation.json"

LEAD_RATE = 0.55
MAIN_TARGET_RATE = 0.65
MINIMUM_CONDITIONAL_EFFECT = 0.10
MINIMUM_EVENTS_PER_STATE = 10
MINIMUM_DIRECTION_EPISODES = 20

EXPECTED_SURPRISE_TOKENS = ("expectation", "consensus", "forecast", "surprise")

SIBLINGS: tuple[dict[str, Any], ...] = (
    {
        "route_id": "event_plus_market_readiness_and_background",
        "role_question": "driver_plus_background_or_modifier",
        "plain_question": (
            "Does the same kind of event produce a stronger, weaker, longer, or more "
            "reversible reaction when the market is already busy, under negative slow "
            "pressure, unusually volatile, or near the edge of its prior range?"
        ),
        "inputs": [
            "event family and event kind",
            "prior-hour relative volume",
            "prior-30-day return",
            "prior-30-day volatility",
            "prior-30-day range position",
            "pre-event broad-crypto alignment",
        ],
        "primary_outcomes": ["future volume reaction at 1h 4h 8h"],
        "secondary_outcomes": ["range and absolute price movement"],
        "controls": [
            "parent-linked no-event clocks in the same market-state category",
            "event episodes with the modifier absent",
            "state-only matched clocks",
        ],
        "availability": "ready_for_direct_test",
    },
    {
        "route_id": "distinct_event_and_narrative_accumulation",
        "role_question": "accumulator_or_override",
        "plain_question": (
            "Do two or more distinct known event families within 24 hours change the "
            "reaction compared with one known family, including aligned and conflicting "
            "signed information where a causal sign exists?"
        ),
        "inputs": [
            "distinct event-family count within 24h",
            "positive and negative signed counts within 24h",
            "signed balance",
            "current event family",
        ],
        "primary_outcomes": ["future volume reaction at 1h 4h 8h"],
        "secondary_outcomes": ["range and absolute price movement"],
        "controls": [
            "single-family event episodes",
            "each episode's parent-linked no-event clocks",
            "mixed-sign versus aligned-sign accumulation",
        ],
        "availability": "ready_for_catalogue_accumulation_test",
        "limitation": (
            "This cache counts distinct frozen event families; it is not a complete "
            "semantic archive of every news story or narrative."
        ),
    },
    {
        "route_id": "true_expectation_surprise",
        "role_question": "signed_event_driver",
        "plain_question": (
            "Does the released value differ from the timestamped market expectation and "
            "does that surprise explain short Bitcoin or Ethereum direction?"
        ),
        "inputs": [
            "official initial release value",
            "timestamped historical consensus expectation",
            "release availability time",
        ],
        "primary_outcomes": ["5m 15m 30m 60m signed BTC and ETH reaction"],
        "controls": [
            "previous-release change proxy",
            "matched non-release weeks",
            "whole-release sign reshuffles",
        ],
        "availability": "coverage_parked_missing_timestamped_expectation",
        "limitation": (
            "Existing signed-source fields are not accepted as substitutes for a real "
            "historical market expectation."
        ),
    },
    {
        "route_id": "initial_btc_eth_to_established_transmission",
        "role_question": "leader_confirmation_and_transmission",
        "plain_question": (
            "After the first event-hour Bitcoin move is known, does Ethereum agreement "
            "or the established coins' own early agreement improve the following one- "
            "or three-hour direction estimate?"
        ),
        "inputs": [
            "confirmed first Bitcoin reaction",
            "first Ethereum reaction",
            "each established coin's first reaction",
            "pre-event market state",
        ],
        "primary_outcomes": ["following 1h and 3h established-cohort direction"],
        "secondary_outcomes": ["following group volume and range"],
        "controls": [
            "Bitcoin first move alone",
            "follower's own first move alone",
            "matched non-event clocks with equally large first moves",
        ],
        "availability": "ready_for_later_2026_direct_test",
        "decision_time": "end_of_first_event_hour_not_original_release_time",
    },
    {
        "route_id": "continuous_coin_local_level_and_cluster_modifier",
        "role_question": "local_modifier",
        "plain_question": (
            "Does actual distance to an isolated level or independent cluster, its "
            "source/timeframe, and cluster composition alter event reaction size, path, "
            "or reversal?"
        ),
        "inputs": [
            "single-level presence and ATR distance",
            "level source role and timeframe",
            "cluster ATR distance",
            "independent-family count",
            "prior crossing count",
        ],
        "primary_outcomes": ["future range and volume reaction at 1h 4h 8h"],
        "secondary_outcomes": ["absolute movement and later reversal candidates"],
        "controls": [
            "same event class without the local state",
            "parent-linked controls with the same local state",
            "isolated levels and clusters kept separate",
        ],
        "availability": "ready_for_direct_test",
        "fixed_geometry": {
            "near_cluster_max_atr": 0.10,
            "richer_cluster_min_independent_families": 3,
        },
    },
    {
        "route_id": "verified_btc_to_meme_transmission",
        "role_question": "group_transmission_and_conditional_amplification",
        "plain_question": (
            "Only after Bitcoin has a verified reaction, do prior market readiness, slow "
            "background, and early meme-cohort agreement help estimate the next one or "
            "three hours?"
        ),
        "inputs": [
            "confirmed Bitcoin reaction",
            "prior Bitcoin activity",
            "prior-30-day Bitcoin background",
            "early meme-cohort agreement",
            "ordinary meme sensitivity and volatility",
        ],
        "primary_outcomes": ["following 1h and 3h meme-cohort direction"],
        "secondary_outcomes": [
            "same-hour breadth",
            "movement beyond ordinary Bitcoin sensitivity",
        ],
        "controls": [
            "established-coin cohort",
            "matched non-event Bitcoin reactions",
            "ordinary beta and volatility adjustment",
        ],
        "availability": "ready_for_later_2026_development_test",
        "limitation": (
            "Meme coins are not tested against event families that have no verified "
            "Bitcoin reaction. Existing 2026 episodes are development evidence only."
        ),
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expectation_columns(feature_columns: Sequence[str]) -> list[str]:
    return sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in EXPECTED_SURPRISE_TOKENS)
    )


def _verify_contract(contract: Mapping[str, Any]) -> None:
    path = Path(str(contract["path"]))
    if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
        raise ValueError(f"Source contract changed: {path}")


def build_freeze() -> dict[str, Any]:
    reanalysis = _load_json(REANALYSIS_PATH)
    if reanalysis.get("status") != "completed_whole_episode_reanalysis":
        raise ValueError("Whole-episode reanalysis is not terminal.")
    meme_result = _load_json(MEME_RESULT_PATH)
    if meme_result.get("status") != "completed_event_meme_transmission_review":
        raise ValueError("Meme source result is not terminal.")
    cache_manifest = _load_json(CACHE_MANIFEST_PATH)
    available_columns = [str(column) for column in cache_manifest["feature_columns"]]
    surprise_columns = expectation_columns(available_columns)
    if surprise_columns:
        raise ValueError(
            "Expectation-like columns appeared after the route was classified unavailable."
        )
    for item in cache_manifest["inventory"]:
        for path_key, hash_key in (
            ("feature_path", "feature_sha256"),
            ("evaluation_path", "evaluation_sha256"),
        ):
            _verify_contract({"path": item[path_key], "sha256": item[hash_key]})

    return {
        "schema_version": 1,
        "status": "frozen_conditional_episode_batch_before_new_outcome_analysis",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "plain_objective": (
            "Complete several different whole-episode questions before allowing any one "
            "result to create a narrower descendant."
        ),
        "evidence_state": (
            "Retrospective development and stability testing only. All current source "
            "periods have been inspected previously, so no result may be called fresh "
            "confirmation; later whole events remain required."
        ),
        "scope": {
            "profit_used": False,
            "trading_rule_tested": False,
            "freqai_required_before_direct_link_exists": False,
            "meme_route_requires_verified_btc_reaction": True,
            "siblings_must_finish_before_branching": True,
        },
        "decision_rules": {
            "minimum_direction_or_joint_rate": LEAD_RATE,
            "main_target_rate": MAIN_TARGET_RATE,
            "minimum_events_per_modifier_state": MINIMUM_EVENTS_PER_STATE,
            "minimum_direction_episodes": MINIMUM_DIRECTION_EPISODES,
            "minimum_conditional_effect": MINIMUM_CONDITIONAL_EFFECT,
            "modifier_rule": (
                "A modifier may be retained when it changes the event relationship by "
                "at least ten percentage points with adequate support and compatible "
                "chronological evidence; it need not work alone."
            ),
            "driver_rule": (
                "Judge a possible driver with family-specific clocks and matched controls. "
                "Do not use a downstream response to disprove the upstream driver."
            ),
            "negative_rule": (
                "A null rejects only the named role representation horizon and scope."
            ),
            "branch_rule": (
                "Finish or coverage-park every sibling then review the batch jointly. "
                "Queue all inspired branches until that review."
            ),
        },
        "event_classes": {
            "scheduled_policy_or_macro": [
                "scheduled_us_policy",
                "scheduled_us_macro",
                "scheduled_uk_macro",
                "scheduled_asia_macro",
            ],
            "discrete_media_finance_or_corporate": [
                "unexpected_media_activity",
                "european_finance_news",
                "corporate_technology_news",
            ],
            "daily_cross_market_state": ["daily_cross_market_state"],
        },
        "siblings": list(SIBLINGS),
        "expectation_source_gate": {
            "expectation_columns_found": surprise_columns,
            "decision": "coverage_parked_missing_timestamped_expectation",
            "plain_reason": (
                "The current cache has signed source proxies but no historical consensus "
                "forecast. Previous release direction must not be relabelled as surprise."
            ),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "whole_episode_reanalysis": artifact(REANALYSIS_PATH),
            "cache_manifest": artifact(CACHE_MANIFEST_PATH),
            "simple_freeze": artifact(simple.FREEZE_PATH),
            "simple_calls": artifact(simple.CALLS_PATH),
            "meme_result": artifact(MEME_RESULT_PATH),
            "meme_validation": artifact(MEME_VALIDATION_PATH),
        },
    }


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        existing = _load_json(FREEZE_PATH)
        if (
            existing.get("status")
            != "frozen_conditional_episode_batch_before_new_outcome_analysis"
        ):
            raise ValueError("Existing conditional-episode freeze is not valid.")
        return existing
    result = build_freeze()
    g0.atomic_write_json(result, FREEZE_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = freeze(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
