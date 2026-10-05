# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation19 as g19f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freqai_cache as g19c,
)


def test_smoke_registry_uses_one_exact_comparison_and_two_profiles() -> None:
    registry = g19c.build_registry(
        [
            f"{g19c.LOCAL_TREND}__value",
            f"{g19c.LEVEL}__value",
            f"{g19c.STALE_LEVEL}__value",
            f"{g19c.LOCAL_ACTIVITY}__value",
            g19c.DISPERSION_COLUMN,
        ]
    )
    profiles, comparisons = g19f.active_registry(registry, "normal")
    profiles, comparisons = g19f.smoke_registry(profiles, comparisons)

    assert len(profiles) == 2
    assert len(comparisons) == 1
    assert comparisons[0]["route_id"] == "level_plus_dispersion"
    assert comparisons[0]["control_type"] == "level_only"


def test_profile_config_routes_exact_generation19_contract(monkeypatch) -> None:
    profile = {
        "profile_id": "profile",
        "feature_columns": [f"{g19c.LOCAL_TREND}__value"],
        "required_ready_blocks": [g19c.LOCAL_TREND],
        "targets": list(g19c.TARGETS),
        "seed": g19c.SEED,
    }
    monkeypatch.setattr(
        g19f.g13,
        "profile_config",
        lambda *_args, **_kwargs: {
            "market_reaction_zone_g13": {
                "target_columns": list(g19c.TARGETS),
                "maximum_target_horizon_hours": 8,
            }
        },
    )
    config = g19f.profile_config(
        {"freqai": {"feature_parameters": {}}},
        identifier="identifier",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g19f.Path("features"),
        event_dir=g19f.Path("events"),
        profile=profile,
        train_days=100,
        backtest_days=30,
        technical_smoke=True,
    )

    assert "market_reaction_zone_g13" not in config
    assert config["market_reaction_zone_g19"]["target_columns"] == list(g19c.TARGETS)
    assert config["market_reaction_zone_g19"]["maximum_target_horizon_hours"] == 8


def test_joint_decisions_require_two_complete_cells() -> None:
    rows = []
    for window, cohort, retained in (
        (g19f.g13z.STANDARD, "normal", True),
        (g19f.g13z.RECENT, "normal", True),
        (g19f.g13z.STANDARD, "meme", False),
    ):
        rows.append(
            {
                "route_id": "level_plus_local_activity",
                "target": "&-g19_crossing_count_h2",
                "evaluation_window": window,
                "cohort": cohort,
                "all_controls_strict": retained,
                "all_controls_point_positive": retained,
                "minimum_equal_coin_paired_mae_gain": 0.01 if retained else -0.01,
                "minimum_bootstrap_lower": 0.001 if retained else -0.02,
            }
        )

    decision = g19f.joint_decisions(pd.DataFrame.from_records(rows)).iloc[0]
    assert decision["status"] == "strict_at_least_two_cells"
    assert decision["strict_cells"] == 2
    assert decision["normal_standard_and_recent_point"]
