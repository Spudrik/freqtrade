# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation11 as g11,
)


def test_active_registry_parks_unsupported_news_without_dropping_other_routes() -> None:
    frozen, cache, _ = g11.load_sources("normal")

    profiles, comparisons, audit = g11.active_registry(frozen, cache, "normal")

    assert len(profiles) == 31
    assert len(comparisons) == 24
    assert len(audit["active_routes"]) == 7
    assert audit["parked_routes"] == ["news_context"]
    assert audit["unreferenced_supported_profiles"] == [
        "g11__normal__news_context__baseline__seed42"
    ]


def test_profile_config_uses_generation11_key_and_eight_hour_horizon() -> None:
    frozen, cache, _ = g11.load_sources("meme")
    profiles, _, _ = g11.active_registry(frozen, cache, "meme")
    profile = next(iter(profiles.values()))
    base = {
        "exchange": {"pair_whitelist": []},
        "freqai": {
            "feature_parameters": {},
            "data_split_parameters": {},
            "model_training_parameters": {"n_estimators": 100},
        },
    }

    output = g11.profile_config(
        base,
        identifier="test",
        pairs=("DOGE/USDT:USDT",),
        feature_dir=Path("unused"),
        event_dir=Path("unused"),
        profile=profile,
        train_days=160,
        backtest_days=30,
        technical_smoke=False,
    )

    assert "market_reaction_zone_g6" not in output
    assert output["market_reaction_zone_g11"]["target_columns"] == list(
        g11.g11z.TARGETS
    )
    assert output["freqai"]["feature_parameters"]["label_period_candles"] == 8


def test_binary_metrics_report_skill_separately_from_base_rate() -> None:
    actual = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    prediction = pd.Series([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65])

    result = g11.binary_prediction_metrics(actual, prediction)

    assert result["base_rate"] == 0.7
    assert result["majority_accuracy"] == 0.7
    assert result["accuracy_at_half"] == 1.0
    assert result["balanced_accuracy_at_half"] == 1.0
    assert result["rank_auc"] == 1.0
    assert result["top_minus_bottom_rate"] > 0.0


def test_route_decision_requires_every_control_in_both_periods() -> None:
    rows = []
    for control in ("simple", "stale", "shuffle"):
        for period in ("validation_early", "validation_late"):
            rows.append(
                {
                    "question_id": "question",
                    "route_id": "route",
                    "route_type": "add_one_information_family",
                    "target": "&-g11_reaction_h1",
                    "seed": 42,
                    "plain_question": "Does it help?",
                    "expected_controls_for_route": 3,
                    "comparison_id": control,
                    "control_type": control,
                    "period": period,
                    "formal_decision_group": True,
                    "strict_period_pass": True,
                    "provisional_period_pass": True,
                    "rows": 100,
                    "positive_coins": 7,
                    "equal_coin_paired_mae_gain": 0.01,
                    "bootstrap_lower": 0.001,
                    "not_dominated_by_one_coin": True,
                }
            )
    manifest = {"validation_periods": ["validation_early", "validation_late"]}

    complete = g11.route_decisions(manifest, pd.DataFrame(rows)).iloc[0]
    incomplete = g11.route_decisions(manifest, pd.DataFrame(rows[:-2])).iloc[0]

    assert complete["status"] == "strict_initial_lead_pending_confirmation"
    assert incomplete["status"] == "not_retained_in_initial_attribution"


def test_plain_route_summary_keeps_all_joint_statuses_visible() -> None:
    joint = pd.DataFrame(
        {
            "route_id": ["route", "route", "other"],
            "target": ["reaction_h1", "reaction_h2", "reaction_h1"],
            "status": [
                "strict_in_both_cohorts_pending_confirmation",
                "not_retained_after_joint_initial_batch",
                "normal_cohort_only_lead_pending_confirmation",
            ],
        }
    )

    summary = g11.plain_route_summary(joint).set_index("route_id")

    assert summary.loc["route", "targets_reviewed"] == 2
    assert (
        summary.loc[
            "route", "count__strict_in_both_cohorts_pending_confirmation"
        ]
        == 1
    )
    assert (
        summary.loc["route", "targets__not_retained_after_joint_initial_batch"]
        == "reaction_h2"
    )
