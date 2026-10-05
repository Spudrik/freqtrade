# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation12 as g12,
)


def test_active_registries_cover_all_three_frozen_runs() -> None:
    frozen, _, _ = g12.load_sources("normal")
    recent_profiles, recent_comparisons = g12.active_registry(
        frozen, stage=g12.g12z.STAGE_RECENT, cohort="normal"
    )
    normal_profiles, normal_comparisons = g12.active_registry(
        frozen, stage=g12.g12z.STAGE_ATTRIBUTION, cohort="normal"
    )
    meme_profiles, meme_comparisons = g12.active_registry(
        frozen, stage=g12.g12z.STAGE_ATTRIBUTION, cohort="meme"
    )

    assert len(recent_profiles) == 16
    assert len(recent_comparisons) == 15
    assert len(normal_profiles) == len(meme_profiles) == 44
    assert len(normal_comparisons) == len(meme_comparisons) == 60


def test_recent_period_assignment_uses_frozen_half_open_windows() -> None:
    manifest = {
        "run_stage": g12.g12z.STAGE_RECENT,
        "recent_periods": list(g12.g12z.RECENT_PERIODS),
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-31T23:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-06-30T23:00:00Z",
                    "2026-07-01T00:00:00Z",
                    "2026-08-20T23:00:00Z",
                    "2026-08-21T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    result = g12.assign_evaluation_periods(frame, manifest)

    assert result["period"].tolist() == [
        "outside_generation12_recent_windows",
        "recent_confirmation_early",
        "recent_confirmation_early",
        "recent_confirmation_late",
        "recent_confirmation_late",
        "outside_generation12_recent_windows",
    ]


def test_profile_config_routes_exact_columns_to_generation12_strategy() -> None:
    frozen, _, _ = g12.load_sources("meme")
    profiles, _ = g12.active_registry(
        frozen, stage=g12.g12z.STAGE_ATTRIBUTION, cohort="meme"
    )
    profile = next(iter(profiles.values()))
    base = {
        "exchange": {"pair_whitelist": []},
        "freqai": {
            "feature_parameters": {},
            "data_split_parameters": {},
            "model_training_parameters": {"n_estimators": 100},
        },
    }

    output = g12.profile_config(
        base,
        identifier="test",
        pairs=("DOGE/USDT:USDT",),
        feature_dir=g12.Path("unused"),
        event_dir=g12.Path("unused"),
        profile=profile,
        train_days=160,
        backtest_days=30,
        technical_smoke=False,
    )

    assert "market_reaction_zone_g11" not in output
    assert output["market_reaction_zone_g12"]["feature_columns"] == profile[
        "feature_columns"
    ]


def test_requested_runs_keep_recent_before_both_attribution_cohorts() -> None:
    assert g12.requested_runs("all", "all") == [
        (g12.g12z.STAGE_RECENT, "normal"),
        (g12.g12z.STAGE_ATTRIBUTION, "normal"),
        (g12.g12z.STAGE_ATTRIBUTION, "meme"),
    ]
