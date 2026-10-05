"""Freeze the complete Generation 18 confirmation layer before its outcomes open."""

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
    market_reaction_zone_generation17_joint_review as g17j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)


DEFAULT_BATCH_ID = "g18_broad_confirmation_20260823a"
OUTPUT_ROOT = g17j.REVIEW_ROOT.parent / "g18_broad_confirmation"
FREEZE_PATH = OUTPUT_ROOT.parent / "g18_broad_confirmation_freeze_20260823a.json"
PARENT_REVIEW_PATH = (
    g17j.REVIEW_ROOT / g17j.DEFAULT_REVIEW_ID / "g17_joint_review.json"
)
PARENT_QUEUE_PATH = (
    g17j.REVIEW_ROOT / g17j.DEFAULT_REVIEW_ID / "g18_sibling_branch_queue.json"
)
HORIZONS_HOURS = (1, 2, 4)
SOURCE_TIMEFRAMES = ("1h", "4h", "8h")
CONTROLS = (
    "matched_random_time",
    "near_miss",
    "stale_72h",
    "price_shift",
)

CONFIRMATION_PERIODS = {
    "normal": (
        {
            "id": "g18_normal_holdout_early",
            "start_utc": "2026-07-20T00:00:00Z",
            "end_utc_exclusive": "2026-08-05T00:00:00Z",
        },
        {
            "id": "g18_normal_holdout_late",
            "start_utc": "2026-08-05T00:00:00Z",
            "end_utc_exclusive": "2026-08-20T00:00:00Z",
        },
    ),
    "meme": (
        {
            "id": "g18_meme_holdout_early",
            "start_utc": "2026-07-14T00:00:00Z",
            "end_utc_exclusive": "2026-07-29T00:00:00Z",
        },
        {
            "id": "g18_meme_holdout_late",
            "start_utc": "2026-07-29T00:00:00Z",
            "end_utc_exclusive": "2026-08-13T00:00:00Z",
        },
    ),
}

VP_CONFIRMATION_SPECS = (
    "vp_lb72_bins48_nearest_hvn_q80",
    "vp_lb72_bins96_nearest_hvn_q80",
    "vp_lb72_bins48_nearest_lvn_q10",
    "vp_lb72_bins96_nearest_lvn_q10",
    "vp_lb72_bins48_nearest_lvn_q20",
    "vp_lb72_bins96_nearest_lvn_q20",
)
DONCHIAN_CONFIRMATION_SPECS = (
    "donchian_lb24_upper",
    "donchian_lb24_lower",
    "donchian_lb72_upper",
    "donchian_lb72_lower",
)
RIVAL_FAMILIES = (
    "rolling_vwap_deviation_bands",
    "weekly_pivot_grid",
    "generic_ma_bollinger_negative_control",
)
REACTION_FORMS = (
    "any_recross",
    "repeated_recross",
    "two_sided_traversal",
    "one_sided_rejection",
    "one_sided_breakthrough",
    "dwell_fraction",
)
REGIME_BLOCKS = (
    "local_volume_and_range_activity",
    "local_volatility_and_bollinger_compression",
    "local_rsi_macd_and_adx_state",
    "btc_eth_and_equal_weight_crypto_activity",
    "crypto_breadth_dispersion_and_correlation",
    "local_and_wider_trend_agreement_or_conflict",
    "source_ready_orderbook_state",
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
    if parent.get("status") != "completed_generation17_joint_review":
        raise ValueError("Generation 17 joint review is not terminal.")
    if not parent.get("all_six_siblings_terminal_before_review"):
        raise ValueError("Generation 17 siblings were not all terminal before review.")
    if queue.get("status") != "queued_after_complete_generation17_joint_review":
        raise ValueError("Generation 18 sibling queue is not valid.")
    if len(queue.get("siblings", [])) != 6:
        raise ValueError("Generation 18 queue does not contain all six siblings.")
    return parent, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g18a_density_geometry_confirmation",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "On untouched later dates, do gradually denser or wider arrangements "
                "of calculated areas still locate more crossing traffic?"
            ),
            "baseline": "contact state plus completed-contact activity",
            "method": (
                "Direct monotonic density bands and one-component-at-a-time ablations; "
                "the Generation 17 FreqAI result remains the parent model evidence"
            ),
            "components": [
                "independent family count: 1, 2, or 3+",
                "convergent variant count: 1-2, 3-5, or 6+",
                "nearest-level distance",
                "zone width",
                "cross-family cluster flag",
            ],
            "targets": [f"crossings_h{horizon}" for horizon in HORIZONS_HOURS],
            "pass_rule": (
                "A component is retained only when its ordered relationship has the "
                "same sign in both frozen holdout halves, repeats in an honest broad or "
                "predeclared coin-group scope, and is not explained by contact activity, "
                "zone width, one coin, or the control locations."
            ),
        },
        {
            "branch_id": "g18b_reaction_form_confirmation",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Which physical paths after contact repeat on untouched dates: recross, "
                "repeated recross, traversal, rejection, breakthrough, or dwell?"
            ),
            "baseline": "matched ordinary times and genuine near misses",
            "method": "Direct equal-coin comparisons on the frozen holdout halves",
            "reaction_forms": list(REACTION_FORMS),
            "horizons_hours": list(HORIZONS_HOURS),
            "pass_rule": (
                "Retain only the same path form and sign that beats ordinary time and "
                "near miss in both holdout halves with adequate coin support and no "
                "single-coin dominance."
            ),
        },
        {
            "branch_id": "g18c_level_source_confirmation",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Do the adaptive Volume Profile and rolling high/low families survive "
                "untouched chronology and neighbouring rational settings?"
            ),
            "baseline": "all four frozen artificial or misplaced location controls",
            "method": "Family and neighbouring-parameter confirmation without refitting",
            "volume_profile_specs": list(VP_CONFIRMATION_SPECS),
            "donchian_specs": list(DONCHIAN_CONFIRMATION_SPECS),
            "rival_families": list(RIVAL_FAMILIES),
            "controls": list(CONTROLS),
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("crossings", "unsigned_reaction")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A family is retained only when a neighbouring parameter plateau, not "
                "one isolated setting, beats every control in both holdout halves and "
                "repeats across normal and meme or a predeclared narrower market scope."
            ),
        },
        {
            "branch_id": "g18d_context_and_market_regimes",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Do local activity, standard indicators, wider crypto state, or "
                "source-ready orderbook conditions change reaction likelihood beyond "
                "the calculated area itself?"
            ),
            "baseline": "level-contact effect and each context block separately",
            "method": (
                "Outcome-independent causal regime bands and two-block interactions; "
                "missing external observations remain unavailable"
            ),
            "regime_blocks": list(REGIME_BLOCKS),
            "maximum_active_blocks_per_interaction": 2,
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("unsigned_reaction", "crossings", "volume_ratio")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A level-plus-context relationship must beat level alone and context "
                "alone on identical eligible rows in both holdout halves. Current "
                "orderbook must beat stale and shuffled orderbook. Unsupported news or "
                "non-crypto global data stays parked."
            ),
        },
        {
            "branch_id": "g18e_coin_group_and_timeframe_portability",
            "status_at_freeze": "frozen_pending",
            "plain_question": (
                "Do the same rational level questions transfer across BTC, similar coin "
                "groups, memes, and 1h/4h/8h source calculations?"
            ),
            "baseline": "same-family lower-timeframe contact plus matched controls",
            "method": (
                "Recalculate a compact predeclared source set on completed 1h, 4h, and "
                "8h candles, then compare isolated and cross-timeframe contacts"
            ),
            "source_timeframes": list(SOURCE_TIMEFRAMES),
            "source_bar_definitions": {
                "adaptive_volume_profile": (
                    "72 completed source bars; 48 bins; nearest HVN q80 and LVN q10"
                ),
                "rolling_high_low": "24 and 72 completed source bars",
                "rolling_vwap": "72 completed source bars with plus/minus two deviations",
            },
            "market_scopes": [
                "all_normal",
                "btc_separate",
                "smart_contract_platforms",
                "other_established_alts",
                "top_ten_memes",
            ],
            "single_and_cluster_surfaces_equal": True,
            "higher_timeframe_has_automatic_precedence": False,
            "pass_rule": (
                "A timeframe or group is retained only under its exact label when the "
                "same level meaning beats controls in both holdout halves, has adequate "
                "member support, and is not just wider-zone coverage. Cross-timeframe "
                "clusters must beat their contacted components."
            ),
        },
        {
            "branch_id": "g18f_direction_after_reaction_gate",
            "status_at_freeze": "parked_current_rules_failed",
            "plain_question": (
                "Can one-minute inputs predict both reaction and direction after a "
                "qualified reaction gate exists?"
            ),
            "park_reason": (
                "Generation 17 tested 34 causal methods and none reached the 55% joint "
                "reaction-plus-direction floor. No new signed-direction run is justified "
                "inside this sibling layer."
            ),
            "reopen_rule": (
                "Only a jointly reviewed, newly confirmed reaction gate and a newly "
                "frozen independent episode sample may reopen this lane."
            ),
        },
    ]


def freeze_batch(batch_id: str) -> dict[str, Any]:
    parent, queue = load_parent()
    branches = branch_definitions()
    queued_ids = {item["branch_id"] for item in queue["siblings"]}
    branch_ids = {item["branch_id"] for item in branches}
    if queued_ids != branch_ids:
        raise ValueError("Generation 18 freeze drifted from the jointly reviewed queue.")

    atlas_result_path = (
        g17l.RECORD_ROOT / g17l.DEFAULT_RUN_ID / "g17_level_source_atlas_result.json"
    )
    if not atlas_result_path.is_file():
        raise FileNotFoundError(atlas_result_path)
    atlas_result = json.loads(atlas_result_path.read_text(encoding="utf-8"))
    if atlas_result.get("status") != "completed_generation17_level_source_atlas":
        raise ValueError("Generation 17 level-source atlas is not terminal.")

    freeze = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation18_outcomes",
        "batch_id": batch_id,
        "objective": (
            "Confirm several independent direction-neutral reaction-location leads on "
            "untouched later time blocks without allowing one early result to dominate."
        ),
        "sequencing": {
            "all_six_siblings_frozen_together": True,
            "five_active_one_honestly_parked": True,
            "all_terminal_or_parked_before_joint_review": True,
            "no_generation19_descendant_before_joint_review": True,
            "automatic_descendant_launch": False,
            "authorized_by_user_on_2026_08_23": True,
            "authorized_branch_layer": 2,
        },
        "branches": branches,
        "cohorts": {
            "normal": 10,
            "top10_memes": 10,
            "btc_separate": True,
            "similar_coin_groups_retained": True,
        },
        "confirmation_periods": CONFIRMATION_PERIODS,
        "main_horizons_hours": list(HORIZONS_HOURS),
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "controls": list(CONTROLS),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "direction_branch_parked": True,
            "trading_promotion": False,
            "reaction_and_traffic_are_not_direction": True,
        },
        "source_contracts": {
            "generation17_joint_review": artifact(PARENT_REVIEW_PATH),
            "generation18_joint_queue": artifact(PARENT_QUEUE_PATH),
            "generation17_level_atlas": artifact(atlas_result_path),
            "generation17_atlas_event_root": str(
                (
                    g17l.ARTIFACT_ROOT
                    / g17l.DEFAULT_RUN_ID
                    / "pair_events"
                ).resolve()
            ),
        },
        "parent_summary": parent["summary"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in freeze.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 18 batch changed after its freeze.")
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
