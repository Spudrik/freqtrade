"""Freeze the first branch batch from the conditional whole-event joint review."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
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
    market_event_conditional_episode_freeze_2026 as parent_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_joint_review_2026 as parent_review,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_interaction_branch_batch_20260909a"
OUTPUT_ROOT = parent_review.OUTPUT_ROOT / RUN_ID
FREEZE_PATH = OUTPUT_ROOT / "event_interaction_branch_freeze.json"
EVENT_CATALOG_PATH = parent_freeze.SOURCE_ROOT / "event_signal_fresh_event_catalog.csv"
SAMPLE_CATALOG_PATH = parent_freeze.SOURCE_ROOT / "event_signal_fresh_sample_catalog.csv"

FAMILY_ROLE_AND_DOMAIN: dict[str, tuple[str, str]] = {
    "fomc_policy_decision": ("upstream_information", "us_monetary_policy"),
    "us_cpi": ("upstream_information", "us_inflation"),
    "uk_cpi": ("upstream_information", "uk_inflation"),
    "japan_tankan": ("upstream_information", "japan_business_conditions"),
    "gdelt_unexpected_activity": ("media_attention_proxy", "aggregate_media_attention"),
    "esma_sovereign_rating": ("upstream_information", "sovereign_credit"),
    "sec_hyperscaler_earnings": ("upstream_information", "technology_earnings"),
    "cross_market_fear": ("market_response", "us_equity_risk"),
    "cross_technology_equities": ("market_response", "us_equity_risk"),
    "cross_broad_us_dollar": ("market_response", "us_dollar"),
    "cross_ten_year_yield": ("market_response", "us_interest_rates"),
    "cross_yield_curve": ("market_response", "us_interest_rates"),
    "cross_crude_oil": ("market_response", "energy_and_geopolitical_risk"),
    "cross_financial_conditions": (
        "composite_market_response",
        "broad_financial_conditions_composite",
    ),
}

BRANCH_SIBLINGS: tuple[dict[str, Any], ...] = (
    {
        "branch_id": "role_aware_cross_market_overlap",
        "questions": [
            (
                "Do two or more distinct cross-market response domains identify "
                "continued crypto activity?"
            ),
            (
                "Does upstream information plus a separate observed cross-market "
                "response add more than either role alone?"
            ),
            (
                "Are two genuinely distinct upstream information domains supported "
                "often enough to test narrative accumulation?"
            ),
        ],
        "outcomes": [
            "above-normal crypto volume at 1h 4h 8h",
            (
                "signed BTC and ETH direction only when at least two available "
                "crypto-response signs agree"
            ),
        ],
        "controls": [
            "single response domain",
            "upstream-only and response-only episodes",
            "parent-linked no-event clocks",
            "rotated response signs for signed direction",
        ],
        "anchor_rule": "latest_known_anchor_in_whole_episode",
    },
    {
        "branch_id": "signed_accumulation_semantics_audit",
        "questions": [
            "Which families created the old raw same-sign count?",
            (
                "Were the signs comparable market meanings or merely positive and "
                "negative source arithmetic?"
            ),
            "How many counts remain after driver-domain deduplication and availability checks?",
        ],
        "outcomes": [],
        "controls": ["source semantics and availability only; no price outcome read"],
        "anchor_rule": "all_frozen_samples_audited_then_whole_episode_collapsed",
    },
    {
        "branch_id": "upper_and_lower_range_edge_discrete_events",
        "questions": [
            (
                "Does a discrete event near the upper fifth of the prior 30-day range "
                "add one-hour activity versus the middle?"
            ),
            "Does the same apply near the lower fifth?",
        ],
        "outcomes": ["above-normal volume at one hour"],
        "controls": [
            "middle 60 percent of the prior range",
            "parent-linked clocks with the same upper lower or middle location",
            "upper and lower edges reported separately",
        ],
        "anchor_rule": "earliest_known_anchor_in_whole_episode",
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_freeze() -> dict[str, Any]:
    parent = _load_json(parent_review.RESULT_PATH)
    if parent.get("status") != "completed_conditional_episode_joint_review":
        raise ValueError("Parent joint review is not terminal.")
    families = set(FAMILY_ROLE_AND_DOMAIN)
    catalog_header = EVENT_CATALOG_PATH.read_text(encoding="utf-8").splitlines()[0]
    required_columns = {
        "event_family",
        "model_anchor_utc",
        "source_sign_primary",
        "crypto_relation_sign",
        "signed_semantics",
        "event_episode_id",
    }
    if not required_columns.issubset(set(catalog_header.split(","))):
        raise ValueError("Event catalogue lacks required role-aware audit columns.")
    return {
        "schema_version": 1,
        "status": "frozen_event_interaction_branches_before_branch_outcomes",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "plain_objective": (
            "Resolve all three joint-review branches together while separating upstream "
            "information from downstream market response and duplicate measurements."
        ),
        "scope": {
            "retrospective_branch_development_only": True,
            "profit_used": False,
            "trading_rule_tested": False,
            "freqai_used_before_direct_controls": False,
            "all_siblings_complete_before_next_descendant": True,
        },
        "role_rules": {
            "upstream_information": (
                "Scheduled releases or discrete reports that can initiate new information."
            ),
            "market_response": (
                "Observed movement in equities fear dollar rates oil or another market; "
                "it can confirm transmission but is not counted as a second news cause."
            ),
            "media_attention_proxy": (
                "An aggregate media-activity burst can mark attention or information "
                "arrival, but without story semantics it is not counted as a distinct "
                "upstream news driver."
            ),
            "composite_market_response": (
                "A broad composite is not an extra independent domain when its detailed "
                "components are already present."
            ),
            "source_sign": (
                "Raw source arithmetic is never called bullish or bearish across unlike "
                "families. Use crypto-relation sign only where explicitly available and "
                "keep previous-release CPI separate from true expectation surprise."
            ),
        },
        "family_role_and_domain": {
            family: {"role": role, "domain": domain}
            for family, (role, domain) in FAMILY_ROLE_AND_DOMAIN.items()
        },
        "known_families": sorted(families),
        "branches": list(BRANCH_SIBLINGS),
        "decision_rules": {
            "minimum_state_episodes": 10,
            "minimum_conditional_effect": 0.10,
            "minimum_signed_success_rate": 0.55,
            "main_signed_target_rate": 0.65,
            "repeat_rule": (
                "Require compatible material evidence in two chronological periods and no "
                "material opposite-period result. Otherwise label unresolved or parked."
            ),
            "current_data_rule": (
                "An older lead without later support remains historical and needs future "
                "whole events; it is not current confirmation."
            ),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "parent_joint_review": artifact(parent_review.RESULT_PATH),
            "event_catalog": artifact(EVENT_CATALOG_PATH),
            "sample_catalog": artifact(SAMPLE_CATALOG_PATH),
            "cache_manifest": artifact(parent_freeze.CACHE_MANIFEST_PATH),
        },
    }


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        existing = _load_json(FREEZE_PATH)
        if existing.get("status") != "frozen_event_interaction_branches_before_branch_outcomes":
            raise ValueError("Existing interaction branch freeze is not valid.")
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
