# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation23 as g23f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23c,
)


def test_profile_config_routes_exact_generation23_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        g23f.g22f,
        "profile_config",
        lambda *_args, **_kwargs: {
            "market_reaction_zone_g22": {"target_columns": list(g23c.TARGETS)},
            "freqai": {"model_training_parameters": {"n_jobs": 1}},
        },
    )
    profile = next(iter(g23c.build_registry()["profiles"].values()))

    config = g23f.profile_config(
        {},
        identifier="identifier",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g23f.Path("features"),
        event_dir=g23f.Path("events"),
        profile=profile,
        train_days=365,
        backtest_days=15,
        technical_smoke=False,
    )

    assert "market_reaction_zone_g22" not in config
    assert config["market_reaction_zone_g23"]["target_columns"] == list(g23c.TARGETS)


def test_control_decision_requires_all_five_controls_in_both_periods() -> None:
    periods = ["early", "late"]
    rows = []
    for period in periods:
        for role, mae, spearman, separation in (
            ("full_interaction", 0.5, 0.4, 0.8),
            ("constant_training_median", 0.8, 0.0, 0.0),
            ("level_geometry_only", 0.7, 0.2, 0.3),
            ("market_state_only", 0.65, 0.25, 0.4),
            ("within_pair_time_shuffled_training_labels", 0.75, 0.0, 0.1),
            ("generation22_uncalibrated_parent", 0.6, 0.3, 0.5),
        ):
            rows.append(
                {
                    "cohort": "normal",
                    "period": period,
                    "role": role,
                    "target": g23c.TARGETS[0],
                    "market_scope": "all_normal",
                    "coin_support_pass": True,
                    "median_absolute_error": mae,
                    "spearman_rank_correlation": spearman,
                    "top_minus_bottom_difference": separation,
                    "top_quartile_above_training_median_fraction": 0.60,
                }
            )

    comparisons, decisions = g23f.control_comparisons(
        pd.DataFrame.from_records(rows), periods
    )

    assert len(comparisons) == 10
    assert set(comparisons["control"]) == set(g23f.ALL_CONTROLS)
    assert decisions.iloc[0]["status"] == "exploratory_unsigned_reaction_lead"
    assert decisions.iloc[0]["all_controls_both_periods_pass"]


def test_parent_control_failure_rejects_otherwise_complete_candidate() -> None:
    periods = ["early", "late"]
    rows = []
    for period in periods:
        for role, mae, spearman, separation in (
            ("full_interaction", 0.5, 0.4, 0.8),
            ("constant_training_median", 0.8, 0.0, 0.0),
            ("level_geometry_only", 0.7, 0.2, 0.3),
            ("market_state_only", 0.65, 0.25, 0.4),
            ("within_pair_time_shuffled_training_labels", 0.75, 0.0, 0.1),
            ("generation22_uncalibrated_parent", 0.4, 0.5, 0.9),
        ):
            rows.append(
                {
                    "cohort": "normal",
                    "period": period,
                    "role": role,
                    "target": g23c.TARGETS[0],
                    "market_scope": "all_normal",
                    "coin_support_pass": True,
                    "median_absolute_error": mae,
                    "spearman_rank_correlation": spearman,
                    "top_minus_bottom_difference": separation,
                    "top_quartile_above_training_median_fraction": 0.60,
                }
            )

    comparisons, decisions = g23f.control_comparisons(
        pd.DataFrame.from_records(rows), periods
    )

    parent = comparisons.loc[
        comparisons["control"].eq("generation22_uncalibrated_parent")
    ]
    assert not parent["period_control_pass"].any()
    assert not decisions.iloc[0]["all_controls_both_periods_pass"]
