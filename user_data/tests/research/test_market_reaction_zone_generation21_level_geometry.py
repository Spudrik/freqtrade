# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_level_geometry as g21l,
)


def test_volume_profile_roles_are_mutually_named() -> None:
    assert g21l.role_group("vp_lb72_bins48_poc") == "point_of_control"
    assert (
        g21l.role_group("vp_lb168_bins96_nearest_value_boundary")
        == "value_area_boundary"
    )
    assert g21l.role_group("vp_lb72_binsfd_nearest_hvn_q90") == "high_volume_node"
    assert g21l.role_group("vp_lb720_bins24_nearest_lvn_q10") == "low_volume_node"
    assert g21l.role_group("unrelated") is None


def test_geometry_states_keep_singles_and_clusters_separate() -> None:
    assert g21l.geometry_state(1) == "isolated_single"
    assert g21l.geometry_state(2) == "two_family_cluster"
    assert g21l.geometry_state(3) == "three_plus_family_cluster"
    assert g21l.geometry_state(8) == "three_plus_family_cluster"
    assert g21l.geometry_state(None) == "unavailable"
