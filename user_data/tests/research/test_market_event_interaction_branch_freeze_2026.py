# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_interaction_branch_freeze_2026 as frozen,
)


def test_market_measurements_are_not_upstream_news() -> None:
    assert frozen.FAMILY_ROLE_AND_DOMAIN["cross_market_fear"] == (
        "market_response",
        "us_equity_risk",
    )
    assert frozen.FAMILY_ROLE_AND_DOMAIN["us_cpi"][0] == "upstream_information"
    assert frozen.FAMILY_ROLE_AND_DOMAIN["gdelt_unexpected_activity"][0] == (
        "media_attention_proxy"
    )


def test_related_market_measurements_share_a_domain() -> None:
    assert frozen.FAMILY_ROLE_AND_DOMAIN["cross_market_fear"][1] == frozen.FAMILY_ROLE_AND_DOMAIN[
        "cross_technology_equities"
    ][1]
    ten_year_domain = frozen.FAMILY_ROLE_AND_DOMAIN["cross_ten_year_yield"][1]
    curve_domain = frozen.FAMILY_ROLE_AND_DOMAIN["cross_yield_curve"][1]
    assert ten_year_domain == curve_domain


def test_all_three_joint_review_branches_are_frozen_together() -> None:
    assert len(frozen.BRANCH_SIBLINGS) == 3
