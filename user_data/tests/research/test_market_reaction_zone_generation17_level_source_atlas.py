# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)


def test_source_registry_is_broad_and_predeclared() -> None:
    specs = g17l.source_specs()
    families = {spec.family for spec in specs}
    assert len(specs) == 113
    assert families == {
        "adaptive_volume_profile_nodes",
        "rolling_vwap_deviation_bands",
        "donchian_boundaries",
        "weekly_pivot_grid",
        "generic_ma_bollinger_negative_control",
    }
    assert set(g17l.CONTROLS) == {
        "actual",
        "matched_random_time",
        "near_miss",
        "stale_72h",
        "price_shift",
    }


def test_profile_snapshot_returns_rational_nodes() -> None:
    prices = np.linspace(90.0, 110.0, 100)
    weights = np.concatenate([np.ones(50), np.full(50, 3.0)])
    result = g17l.profile_snapshot(
        prices,
        weights,
        prices - 0.5,
        prices + 0.5,
        reference=100.0,
        bins_mode=24,
    )
    assert set(result) == set(g17l.PROFILE_ROLES)
    assert all(np.isfinite(value) for value in result.values())
    assert 89.0 < result["poc"] < 111.0


def test_whole_decision_requires_every_control_in_both_periods() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        for comparison in g17l.CONTROL_COMPARISONS:
            rows.append(
                {
                    "scope_kind": "single_level",
                    "scope_value": "level",
                    "metric": "unsigned_reaction",
                    "horizon_hours": 2,
                    "market_scope": "all_normal",
                    "period": period,
                    "comparison": comparison,
                    "equal_coin_difference": 0.1,
                    "bootstrap_lower_95": 0.01,
                    "point_period_pass": True,
                    "strict_period_pass": True,
                }
            )
    decision = g17l.whole_decisions(pd.DataFrame(rows)).iloc[0]
    assert decision["complete_control_period_ladder"]
    assert decision["strict_repeated"]
