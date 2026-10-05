# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_joint_review_2026 as review,
)


def test_coherent_pattern_requires_three_pairs_or_two_horizons() -> None:
    base = {
        "route_id": "route",
        "role": "modifier",
        "condition": "condition",
        "outcome": "volume_reaction",
        "control_match": "same_state",
        "event_scope": "all_catalogued_events",
        "verdict": "retained_retrospective_modifier_lead_needs_future_events",
        "median_conditional_effect_supported_periods": 0.2,
    }
    rows = []
    for pair in ("BTC", "ETH", "BNB"):
        rows.append({**base, "pair": pair, "horizon_hours": 1})
    rows.append({**base, "pair": "BTC", "horizon_hours": 4})
    patterns = review.coherent_patterns(pd.DataFrame(rows))
    assert set(patterns["pattern_kind"]) == {
        "same_horizon_at_least_three_pairs",
        "same_pair_at_least_two_horizons",
    }


def test_unresolved_cells_are_not_mechanism_candidates() -> None:
    row = {
        "route_id": "route",
        "role": "modifier",
        "condition": "condition",
        "outcome": "volume_reaction",
        "control_match": "same_state",
        "event_scope": "all_catalogued_events",
        "verdict": "unresolved_effect_changes_sign_between_periods",
        "median_conditional_effect_supported_periods": 0.2,
        "pair": "BTC",
        "horizon_hours": 1,
    }
    assert review.coherent_patterns(pd.DataFrame([row])).empty
