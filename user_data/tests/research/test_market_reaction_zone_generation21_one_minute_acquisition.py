# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_one_minute_acquisition as g21a,
)


def test_generation21_acquisition_is_bounded_and_isolated() -> None:
    assert g21a.MAXIMUM_WORKERS == 2
    assert "g21_broad_siblings" in str(g21a.RECORD_ROOT)
    assert g21a.ACQUISITION_CSV.name == "g21_one_minute_acquisition_intervals.csv"
