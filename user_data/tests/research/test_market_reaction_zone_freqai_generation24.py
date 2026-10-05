# ruff: noqa: S101

from __future__ import annotations

import json

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation24 as g24f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24c,
)


def test_full_registry_selects_all_four_model_seed_cells() -> None:
    profiles, comparisons = g24f.active_registry(
        g24c.build_registry(), "normal", technical_smoke=False
    )

    assert len(profiles) == 20
    assert len(comparisons) == 24
    assert {profile["model_class"] for profile in profiles.values()} == {
        "LightGBMRegressorMultiTarget",
        "XGBoostRegressorMultiTarget",
    }


def test_smoke_registry_keeps_both_model_family_control_ladders() -> None:
    profiles, comparisons = g24f.active_registry(
        g24c.build_registry(), "meme", technical_smoke=True
    )

    assert len(profiles) == 10
    assert len(comparisons) == 12
    assert {profile["role"] for profile in profiles.values()} == set(g24c.PROFILE_ROLES)
    assert {profile["model_class"] for profile in profiles.values()} == {
        "LightGBMRegressorMultiTarget",
        "XGBoostRegressorMultiTarget",
    }


def test_xgboost_profile_removes_lightgbm_only_parameters() -> None:
    registry = g24c.build_registry()
    profile = next(
        item
        for item in registry["profiles"].values()
        if item["cohort"] == "normal" and item["model_class"] == "XGBoostRegressorMultiTarget"
    )
    base = json.loads(g24f.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    config = g24f.profile_config(
        base,
        identifier="g24-test-xgb",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g24f.REPO_ROOT,
        event_dir=g24f.REPO_ROOT,
        profile=profile,
        train_days=365,
        backtest_days=30,
        technical_smoke=True,
    )
    parameters = config["freqai"]["model_training_parameters"]

    assert parameters["verbosity"] == 0
    assert parameters["tree_method"] == "hist"
    assert "min_child_samples" not in parameters
    assert "num_leaves" not in parameters
