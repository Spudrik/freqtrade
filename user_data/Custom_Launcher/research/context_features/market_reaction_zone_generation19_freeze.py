"""Freeze Generation 19's complete breadth-first combination layer."""

from __future__ import annotations

# Repository-local imports follow the root-path bootstrap.
# ruff: noqa: E402
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
    market_reaction_zone_generation18_joint_review as g18j,
)


DEFAULT_BATCH_ID = "g19_broad_combinations_20260823a"
OUTPUT_ROOT = g18j.REVIEW_ROOT.parent / "g19_broad_combinations"
FREEZE_PATH = OUTPUT_ROOT.parent / "g19_broad_combinations_freeze_20260823a.json"
PARENT_REVIEW_PATH = (
    g18j.REVIEW_ROOT / g18j.DEFAULT_REVIEW_ID / "g18_joint_review.json"
)
PARENT_QUEUE_PATH = (
    g18j.REVIEW_ROOT / g18j.DEFAULT_REVIEW_ID / "g19_sibling_branch_queue.json"
)
HORIZONS_HOURS = (1, 2, 4, 8)
RECROSS_PRE_WINDOWS_HOURS = (2, 4)
SOURCE_TIMEFRAMES = ("1h", "4h", "8h")
LOCATION_CONTROLS = (
    "matched_random_time",
    "near_miss",
    "stale_72h",
    "price_shift",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = json.loads(PARENT_REVIEW_PATH.read_text(encoding="utf-8"))
    queue = json.loads(PARENT_QUEUE_PATH.read_text(encoding="utf-8"))
    if parent.get("status") != "completed_generation18_joint_review":
        raise ValueError("Generation 18 joint review is not terminal.")
    if not parent.get("all_siblings_terminal_or_parked_before_review"):
        raise ValueError("Generation 18 siblings were not terminal before review.")
    if queue.get("status") != "queued_after_complete_generation18_joint_review":
        raise ValueError("Generation 19 sibling queue is invalid.")
    if len(queue.get("siblings", [])) != 7:
        raise ValueError("Generation 19 queue does not contain all seven siblings.")
    return parent, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g19a_exact_coordinate_recross_specificity",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Does the exact current calculated coordinate locate recrossing, or "
                "does a broad nearby busy area explain the result just as well?"
            ),
            "surfaces": [
                "adaptive_volume_profile_nodes",
                "donchian_boundaries",
                "rolling_vwap_deviation_bands",
            ],
            "controls": list(LOCATION_CONTROLS),
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "crossing_count")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "The same named family or exact level must beat ordinary time, near "
                "miss, causal stale, and price-shifted coordinates in both later "
                "blocks under an honest broad, frozen-group, or asset-specific label."
            ),
        },
        {
            "branch_id": "g19b_regime_level_incremental_combinations",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Do cross-coin dispersion or completed local activity improve unseen "
                "volume and traffic estimates beyond price state, level information, "
                "and either context alone?"
            ),
            "method": "FreqAI regression with identical rows and a frozen ablation ladder",
            "profiles": [
                "price_and_indicator_baseline",
                "level_only",
                "dispersion_only",
                "local_activity_only",
                "level_plus_dispersion",
                "level_plus_local_activity",
                "stale_level_plus_dispersion",
                "stale_level_plus_local_activity",
            ],
            "maximum_active_blocks_per_interaction": 2,
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("future_volume_ratio", "crossing_count")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A combination must improve unseen error and ranking against the "
                "price/indicator baseline, level-only, context-only, and causal stale-"
                "level counterpart on identical eligible rows in at least two frozen "
                "chronological cells, without one coin supplying the result."
            ),
        },
        {
            "branch_id": "g19c_market_specific_level_mechanisms",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Do the separately labelled meme rolling-boundary/LVN, established-"
                "coin rolling-VWAP, and BTC Volume Profile behaviours repeat?"
            ),
            "predeclared_market_mechanisms": [
                "top_ten_memes__donchian_boundaries",
                "top_ten_memes__vp_lb72_bins96_nearest_lvn_q10",
                "all_normal__rolling_vwap_deviation_bands",
                "btc_separate__adaptive_volume_profile_nodes",
            ],
            "matched_nonmember_check": True,
            "controls": list(LOCATION_CONTROLS),
            "pass_rule": (
                "Retain only the exact predeclared market label when it repeats in both "
                "later blocks, beats every location control, has adequate member "
                "support, and is not renamed as a broad crypto effect."
            ),
        },
        {
            "branch_id": "g19d_recross_timing_and_activity_onset",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "After a level contact, is recrossing newly starting, continuing traffic "
                "that was already underway, or neither?"
            ),
            "pre_windows_hours": list(RECROSS_PRE_WINDOWS_HOURS),
            "post_windows_hours": list(HORIZONS_HOURS),
            "states": [
                "new_onset_pre_zero_post_positive",
                "persistent_pre_positive_post_positive",
                "pre_only_pre_positive_post_zero",
                "quiet_pre_zero_post_zero",
            ],
            "controls": list(LOCATION_CONTROLS),
            "pass_rule": (
                "A new-onset claim requires more pre-zero/post-positive episodes at the "
                "current location than every control in both later blocks. A persistent "
                "traffic result is retained separately and never described as contact-"
                "caused activity."
            ),
        },
        {
            "branch_id": "g19e_rational_acceptance_zone_extension",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Does a repaired causal repeated-close acceptance zone locate repeatable "
                "traffic beyond rolling VWAP and artificial locations?"
            ),
            "construction": {
                "source_timeframe": "1h",
                "history_hours": 168,
                "bin_width_atr": 0.25,
                "source": "completed closes strictly before the decision candle",
                "repair": (
                    "keep contacted width, distinguish fresh arrival from already-inside "
                    "occupancy, and compare directly with rolling VWAP"
                ),
            },
            "controls": [
                "matched_random_time",
                "near_miss",
                "stale_72h",
                "price_shift",
                "rolling_vwap_width_matched",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "dwell_fraction")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "Retain only a fresh-arrival or occupancy-labelled relationship that "
                "repeats in both later blocks, beats every control with comparable zone "
                "width and coverage, and is not merely a wider-zone advantage."
            ),
        },
        {
            "branch_id": "g19f_external_context_accumulation",
            "status_at_freeze": "parked_coverage",
            "plain_question": (
                "Do orderbook, historical news, or non-crypto global sources improve "
                "the level reaction once two complete timestamp-ready blocks exist?"
            ),
            "park_reason": (
                "Generation 18 found no source with timestamp-ready observations in both "
                "required later blocks. Missing data cannot be called quiet or zero."
            ),
            "reopen_rule": "Two complete aligned blocks with availability flags are required.",
        },
        {
            "branch_id": "g19g_direction_after_reaction_gate",
            "status_at_freeze": "parked_current_rules_failed",
            "plain_question": (
                "Can a frozen one-minute sample jointly predict an abnormal reaction and "
                "its immediate direction at least 55% of issued calls?"
            ),
            "park_reason": (
                "The latest 40-episode comparison reached only 35% joint success. The "
                "current batch first tests whether a more exact reaction gate exists."
            ),
            "reopen_rule": (
                "Only a terminal Generation 19 reaction gate with a newly frozen "
                "independent episode sample can reopen signed one-minute work."
            ),
        },
    ]


def freeze_batch(batch_id: str) -> dict[str, Any]:
    parent, queue = load_parent()
    branches = branch_definitions()
    queued_ids = {item["branch_id"] for item in queue["siblings"]}
    branch_ids = {item["branch_id"] for item in branches}
    if queued_ids != branch_ids:
        raise ValueError("Generation 19 freeze drifted from the joint-review queue.")
    freeze = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation19_outcomes",
        "batch_id": batch_id,
        "objective": (
            "Test five materially different direction-neutral explanations and "
            "combinations without letting one result redirect the unfinished batch."
        ),
        "sequencing": {
            "all_seven_siblings_frozen_together": True,
            "five_active_two_honestly_parked": True,
            "all_terminal_or_parked_before_joint_review": True,
            "no_generation20_descendant_before_joint_review": True,
            "automatic_descendant_launch": False,
            "authorized_by_latest_user_message_on_2026_08_23": True,
            "authorized_branch_layer": 3,
        },
        "branches": branches,
        "main_horizons_hours": list(HORIZONS_HOURS),
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "cohorts": {
            "normal": 10,
            "top10_memes": 10,
            "btc_separate": True,
            "predeclared_coin_groups": True,
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "direction_branch_parked": True,
            "trading_promotion": False,
            "activity_and_recrossing_are_not_direction": True,
        },
        "source_contracts": {
            "generation18_joint_review": artifact(PARENT_REVIEW_PATH),
            "generation19_joint_queue": artifact(PARENT_QUEUE_PATH),
        },
        "parent_summary": parent["summary"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in freeze.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 19 batch changed after its freeze.")
        return existing
    g0.atomic_write_json(freeze, FREEZE_PATH)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args(argv)
    freeze = freeze_batch(args.batch_id)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "branches": len(freeze["branches"]),
                "active_branches": sum(
                    item["status_at_freeze"] == "frozen_pending"
                    for item in freeze["branches"]
                ),
                "parked_branches": sum(
                    item["status_at_freeze"].startswith("parked")
                    for item in freeze["branches"]
                ),
                "output": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
