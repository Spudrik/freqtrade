# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation22 as g22f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22c,
)


def test_profile_config_routes_exact_generation22_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        g22f.g13,
        "profile_config",
        lambda *_args, **_kwargs: {
            "market_reaction_zone_g13": {"target_columns": list(g22c.TARGETS)},
            "freqai": {"model_training_parameters": {}},
        },
    )
    profile = next(iter(g22c.build_registry()["profiles"].values()))
    config = g22f.profile_config(
        {},
        identifier="identifier",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g22f.Path("features"),
        event_dir=g22f.Path("events"),
        profile=profile,
        train_days=365,
        backtest_days=15,
        technical_smoke=False,
    )

    assert "market_reaction_zone_g13" not in config
    assert config["market_reaction_zone_g22"]["target_columns"] == list(g22c.TARGETS)
    assert config["freqai"]["model_training_parameters"]["n_jobs"] == 1


def test_prediction_metrics_report_quartile_separation_without_auc() -> None:
    result = g22f.prediction_metrics(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        threshold=4.0,
    )

    assert result["spearman_rank_correlation"] == 1.0
    assert result["top_minus_bottom_difference"] == 6.0
    assert result["top_quartile_above_training_median_fraction"] == 1.0
    assert not any("auc" in key for key in result)


def test_control_decision_requires_all_four_controls_in_both_periods() -> None:
    periods = ["early", "late"]
    rows = []
    for period in periods:
        for role, mae, spearman, separation in (
            ("full_interaction", 0.5, 0.4, 0.8),
            ("constant_training_median", 0.8, 0.0, 0.0),
            ("level_geometry_only", 0.7, 0.2, 0.3),
            ("market_state_only", 0.65, 0.25, 0.4),
            ("within_pair_time_shuffled_training_labels", 0.75, 0.0, 0.1),
        ):
            rows.append(
                {
                    "cohort": "normal",
                    "period": period,
                    "role": role,
                    "target": g22c.TARGETS[0],
                    "market_scope": "all_normal",
                    "coin_support_pass": True,
                    "median_absolute_error": mae,
                    "spearman_rank_correlation": spearman,
                    "top_minus_bottom_difference": separation,
                    "top_quartile_above_training_median_fraction": 0.60,
                }
            )

    comparisons, decisions = g22f.control_comparisons(
        pd.DataFrame.from_records(rows), periods
    )
    assert len(comparisons) == 8
    assert decisions.iloc[0]["status"] == "exploratory_unsigned_reaction_lead"
    assert decisions.iloc[0]["all_controls_both_periods_pass"]
