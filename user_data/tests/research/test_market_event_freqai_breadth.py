# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth as run,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as frozen,
)


def test_profile_config_routes_the_frozen_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        run.g22f,
        "profile_config",
        lambda *_args, **_kwargs: {
            "market_reaction_zone_g22": {
                "target_columns": list(frozen.TARGETS)
            },
            "freqai": {"model_training_parameters": {"n_jobs": 8}},
        },
    )
    profile = frozen.build_registry()["profiles"]["event_plus_recent_market"]

    config = run.profile_config(
        {},
        identifier="identifier",
        pairs=["BTC/USDT:USDT"],
        feature_dir=run.Path("features"),
        event_dir=run.Path("events"),
        profile=profile,
        train_days=900,
        backtest_days=180,
        technical_smoke=False,
    )

    assert "market_reaction_zone_g22" not in config
    assert config["market_event_freqai_breadth"]["target_columns"] == list(
        frozen.TARGETS
    )
    assert config["freqai"]["model_training_parameters"]["n_jobs"] == 1


def test_prediction_metrics_translate_to_correct_side_rate() -> None:
    metrics = run.prediction_metrics(
        pd.Series([-2.0, -1.0, 1.0, 2.0]),
        pd.Series([-1.0, -0.5, 0.5, 1.0]),
        reference=0.0,
        majority_positive=True,
    )

    assert metrics["correct_side_rate"] == 1.0
    assert metrics["training_majority_correct_rate"] == 0.5
    assert metrics["correct_minus_training_majority"] == 0.5


def test_direct_decision_requires_both_periods() -> None:
    rows = []
    for period in run.FULL_SETTINGS["validation_periods"]:
        rows.append(
            {
                "market_scope": "asset:BTC/USDT:USDT",
                "sample_scope": "actual_events",
                "profile_id": "event_identity_only",
                "role": "event_identity_only",
                "target": frozen.TARGETS[0],
                "target_kind": "activity",
                "period": period,
                "independent_samples": 30,
                "eligible_coins": 1,
                "correct_side_rate": 0.60,
                "correct_minus_training_majority": 0.05,
                "spearman_rank_correlation": 0.10,
            }
        )
    decisions = run.direct_decisions(
        pd.DataFrame.from_records(rows),
        run.FULL_SETTINGS["validation_periods"],
        technical_smoke=False,
    )

    assert decisions.iloc[0]["status"] == "exploratory_signal"
    assert decisions.iloc[0]["repeated_55_percent_pass"]


def test_combination_requires_two_point_uplift_over_both_components() -> None:
    rows = []
    for period in run.FULL_SETTINGS["validation_periods"]:
        for role, accuracy, mae, rank in (
            ("event_plus_recent_market", 0.61, 0.40, 0.30),
            ("event_identity_only", 0.58, 0.50, 0.20),
            ("recent_market_only", 0.57, 0.55, 0.10),
        ):
            rows.append(
                {
                    "market_scope": "all_five_equal_weight",
                    "sample_scope": "actual_events",
                    "profile_id": role,
                    "role": role,
                    "target": frozen.TARGETS[0],
                    "target_kind": "activity",
                    "period": period,
                    "correct_side_rate": accuracy,
                    "correct_minus_training_majority": 0.05,
                    "median_absolute_error": mae,
                    "spearman_rank_correlation": rank,
                    "independent_samples": 30,
                }
            )
    periods, decisions = run.combination_decisions(
        pd.DataFrame.from_records(rows),
        [
            {
                "candidate": "event_plus_recent_market",
                "components": ["event_identity_only", "recent_market_only"],
            }
        ],
        run.FULL_SETTINGS["validation_periods"],
    )

    assert periods["period_pass"].all()
    assert decisions.iloc[0]["status"] == "combination_adds_information"
