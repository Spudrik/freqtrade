# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation7_preflight as g7,
)


def test_branch_registry_matches_frozen_batch() -> None:
    batch = json.loads(g7.FROZEN_BATCH.read_text(encoding="utf-8"))
    frozen_ids = {str(branch["id"]) for branch in batch["branches"]}

    assert len(g7.BRANCH_SPECS) == 10
    assert set(g7.BRANCH_SPECS) == frozen_ids
    assert sum(
        branch.get("weak_component_exception") is not None
        for branch in batch["branches"]
    ) == 2


def test_feature_columns_include_current_and_causal_placebo() -> None:
    names = [
        "date",
        "level__distance",
        "level_placebo__distance",
        "ohlcv_volume_pressure__ratio",
        "ohlcv_volume_pressure_placebo__ratio",
        "unrelated__value",
    ]

    selected = g7.feature_columns_for_blocks(
        names, ("level", "ohlcv_volume_pressure")
    )

    assert selected == names[:-1]


def test_volume_profile_variants_count_as_one_dependency_group() -> None:
    dates = pd.to_datetime(
        ["2025-01-01T00:00:00Z"] * 3 + ["2025-01-02T00:00:00Z"], utc=True
    )
    metadata = pd.DataFrame(
        {
            "date": dates,
            "level_family": [
                "volume_profile_explicit_prior",
                "volume_profile_nodes",
                "generic_prior_range",
                "tlv2_forecast_zone",
            ],
            "source_timeframe": ["4h", "4h", "1d", "4h"],
        }
    )

    counts = g7.independent_dependency_group_count(metadata).set_index("date")

    assert counts.loc[dates[0], "independent_dependency_groups"] == 2
    assert counts.loc[dates[-1], "independent_dependency_groups"] == 1


def test_scope_dates_selects_prior_range_without_outcomes(tmp_path: Path) -> None:
    dates = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "level__family_generic_prior_range_count": [1.0, 0.0],
        }
    )

    mask = g7.scope_dates(
        "generic_prior_range",
        features=features,
        evaluation_path=tmp_path / "unused.parquet",
    )

    assert mask.tolist() == [True, False]


def test_scope_dates_requires_different_mechanism_agreement(tmp_path: Path) -> None:
    dates = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "timeframe_4h__different_mechanism_agreement": [0.0, 0.0],
            "timeframe_8h__different_mechanism_agreement": [1.0, 0.0],
            "timeframe_1d__different_mechanism_agreement": [0.0, 0.0],
        }
    )

    mask = g7.scope_dates(
        "different_mechanism_cross_timeframe",
        features=features,
        evaluation_path=tmp_path / "unused.parquet",
    )

    assert mask.tolist() == [True, False]
