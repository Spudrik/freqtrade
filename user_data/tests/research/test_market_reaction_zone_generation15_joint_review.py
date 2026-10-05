# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_joint_review as g15,
)


def test_group_support_matches_predeclared_cohorts() -> None:
    assert g15.GROUP_MIN_COINS == {
        "btc_separate": 1,
        "smart_contract_platforms": 5,
        "other_established_alts": 4,
        "frozen_top_ten_memes": 5,
    }


def test_group_periods_keep_orderbook_latest_period_out() -> None:
    assert g15.group_periods(
        "smart_contract_platforms",
        "recent_normal_chronology",
        orderbook=True,
    ) == ("recent_confirmation_early",)
    assert g15.group_periods(
        "smart_contract_platforms",
        "recent_normal_chronology",
    ) == ("recent_confirmation_early", "recent_confirmation_late")


def test_pair_group_mapping_keeps_btc_separate() -> None:
    assert g15.group_for_pair("normal", "BTC/USDT:USDT") == "btc_separate"
    assert (
        g15.group_for_pair("normal", "SOL/USDT:USDT")
        == "smart_contract_platforms"
    )
    assert g15.group_for_pair("meme", "PENGU/USDT:USDT") == "frozen_top_ten_memes"
