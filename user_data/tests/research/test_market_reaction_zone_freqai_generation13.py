# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation13 as g13,
)


def test_active_registries_cover_both_frozen_stages() -> None:
    frozen, cache, _ = g13.load_sources("normal")

    mtf_profiles, mtf_comparisons = g13.active_registry(
        frozen, cache, stage=g13.g13z.MTF_STAGE, cohort="normal"
    )
    long_profiles, long_comparisons = g13.active_registry(
        frozen, cache, stage=g13.g13z.LONG_STAGE, cohort="normal"
    )

    assert len(mtf_profiles) == 19
    assert len(mtf_comparisons) == 18
    assert len(long_profiles) == 10
    assert len(long_comparisons) == 9


def test_requested_runs_preserve_all_six_sibling_cells() -> None:
    runs = g13.requested_runs("all", "all", "all")

    assert len(runs) == 6
    assert runs[:3] == [
        (g13.g13z.MTF_STAGE, g13.g13z.STANDARD, "normal"),
        (g13.g13z.MTF_STAGE, g13.g13z.RECENT, "normal"),
        (g13.g13z.MTF_STAGE, g13.g13z.STANDARD, "meme"),
    ]


def test_stage_inventory_routes_mtf_and_long_sources() -> None:
    _, cache, _ = g13.load_sources("normal")

    mtf = g13.stage_inventory(cache, stage=g13.g13z.MTF_STAGE)
    long = g13.stage_inventory(cache, stage=g13.g13z.LONG_STAGE)

    assert "mtf_feature_cache" in mtf[0]["feature_path"]
    assert "mtf_event_cache" in mtf[0]["event_path"]
    assert "long_feature_cache" in long[0]["feature_path"]
    assert "long_event_cache" in long[0]["event_path"]


def test_recent_period_assignment_uses_frozen_half_open_windows() -> None:
    manifest = {
        "evaluation_window": g13.g13z.RECENT,
        "recent_periods": list(g13.g12z.RECENT_PERIODS),
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-31T23:00:00Z",
                    "2026-04-01T00:00:00Z",
                    "2026-07-01T00:00:00Z",
                    "2026-08-21T00:00:00Z",
                ]
            ),
            "period": ["old"] * 4,
        }
    )

    result = g13.assign_evaluation_periods(frame, manifest)

    assert result["period"].tolist() == [
        "outside_generation13_recent_windows",
        "recent_confirmation_early",
        "recent_confirmation_late",
        "outside_generation13_recent_windows",
    ]


def test_profile_config_routes_dynamic_horizon_and_exact_columns() -> None:
    base = json.loads(g13.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    frozen, cache, _ = g13.load_sources("normal")
    profiles, _ = g13.active_registry(
        frozen, cache, stage=g13.g13z.LONG_STAGE, cohort="normal"
    )
    profile = next(iter(profiles.values()))

    config = g13.profile_config(
        base,
        identifier="test-g13",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g13.Path("features"),
        event_dir=g13.Path("events"),
        profile=profile,
        train_days=365,
        backtest_days=30,
        technical_smoke=True,
        maximum_target_horizon_hours=48,
    )

    research = config["market_reaction_zone_g13"]
    assert research["feature_columns"] == profile["feature_columns"]
    assert research["target_columns"] == profile["targets"]
    assert research["maximum_target_horizon_hours"] == 48
    assert config["freqai"]["feature_parameters"]["label_period_candles"] == 48
