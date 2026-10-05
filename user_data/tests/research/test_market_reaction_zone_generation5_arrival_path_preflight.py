# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_arrival_path_preflight as g5d,
)


def test_boundary_distance_uses_nearest_zone_edge() -> None:
    assert g5d.boundary_distance(106.0, 100.0, 2.0, 4.0) == 1.0
    assert g5d.boundary_distance(101.0, 100.0, 2.0, 4.0) == 0.0


def test_prior_near_miss_count_counts_episodes_not_candles() -> None:
    high = np.array([96.5, 98.5, 98.8, 95.0, 98.7, 100.0])
    low = high - 0.2
    count = g5d.prior_near_miss_episode_count(
        index=5,
        level=100.0,
        width=1.0,
        high=high,
        low=low,
    )
    assert count == 2


def test_fixed_boundary_state_requires_clean_prior_interval() -> None:
    base = pd.DataFrame(
        {
            "high": np.r_[np.full(12, 95.0), 101.0, np.full(12, 95.0), 101.0],
            "low": np.r_[np.full(12, 94.0), 99.0, np.full(12, 94.0), 99.0],
            "pre_close": np.full(26, 95.0),
        }
    )
    base.loc[13, ["high", "low"]] = [101.0, 99.0]
    events = pd.DataFrame(
        {
            "base_index": [12, 25],
            "level_price": [100.0, 100.0],
            "zone_half_width": [1.0, 1.0],
        }
    )
    states = g5d.classify_fixed_boundary_states(events, base=base)
    assert states.tolist() == [g5d.FRESH_STATE, "repeat_contact"]


def test_global_independence_is_enforced_within_each_surface() -> None:
    frame = pd.DataFrame(
        {
            "cohort": ["normal"] * 3,
            "pair": ["ETH/USDT:USDT"] * 3,
            "scope_id": ["scope"] * 3,
            "boundary_kind": ["actual_density_zone"] * 3,
            "event_state": [g5d.FRESH_STATE] * 3,
            "event_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", "2026-01-01T05:00:00Z"]
            ),
            "pre_distance_atr": [1.0, 0.5, 0.5],
            "base_index": [0, 2, 5],
        }
    )
    selected = g5d.select_independent_events(frame)
    assert selected["base_index"].tolist() == [0, 5]


def test_path_decision_keeps_generic_separate_from_exact_location() -> None:
    scope = "normal_isolated_168h_confirmed_swing_density"
    rows = []
    for period in ("validation_early", "validation_late"):
        for boundary, passed in (
            ("actual_density_zone", True),
            ("same_width_same_approach_no_level", True),
            ("causal_shuffled_location", False),
        ):
            rows.append(
                {
                    "scope_id": scope,
                    "path_feature": "distance_contraction_in_prior_atr_24h",
                    "period": period,
                    "boundary_kind": boundary,
                    "support_passed": passed,
                }
            )
    decision = g5d.path_route_decisions(pd.DataFrame(rows)).iloc[0]
    assert bool(decision["generic_outcome_phase_eligible"])
    assert not bool(decision["exact_location_outcome_phase_eligible"])


def test_selection_projection_rejects_reaction_outcomes() -> None:
    assert g5d.forbidden_selection_columns(
        ["pair", "contact_volume_ratio", "abs_excursion_atr_h1"]
    ) == ["abs_excursion_atr_h1", "contact_volume_ratio"]
