# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_freeze_2026 as frozen,
)


def test_batch_has_five_or_more_materially_distinct_siblings() -> None:
    assert len(frozen.SIBLINGS) >= 5
    assert len({item["route_id"] for item in frozen.SIBLINGS}) == len(frozen.SIBLINGS)


def test_expectation_gate_does_not_accept_signed_proxy_columns() -> None:
    columns = [
        "signed_source__primary_mean",
        "event_identity__family_us_cpi",
        "recent__relative_volume",
    ]
    assert frozen.expectation_columns(columns) == []
    assert frozen.expectation_columns(["cpi_consensus_forecast"]) == [
        "cpi_consensus_forecast"
    ]


def test_meme_route_requires_verified_bitcoin_reaction() -> None:
    meme = next(
        item
        for item in frozen.SIBLINGS
        if item["route_id"] == "verified_btc_to_meme_transmission"
    )
    assert "verified reaction" in meme["plain_question"]
