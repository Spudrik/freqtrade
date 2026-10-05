# ruff: noqa: S101

"""Synthetic tests for whole-release scheduled-macro validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_validation as validation,
)


def test_observed_statistics_count_cells_families_and_shared_horizons() -> None:
    rows = pd.DataFrame(
        [
            {
                "event_family": "jobs",
                "pair": "BTC/USDT:USDT",
                "horizon_minutes": 5,
                "candidate": True,
            },
            {
                "event_family": "jobs",
                "pair": "ETH/USDT:USDT",
                "horizon_minutes": 5,
                "candidate": True,
            },
            {
                "event_family": "jobs",
                "pair": "BTC/USDT:USDT",
                "horizon_minutes": 15,
                "candidate": True,
            },
            {
                "event_family": "retail",
                "pair": "BTC/USDT:USDT",
                "horizon_minutes": 60,
                "candidate": True,
            },
            {
                "event_family": "retail",
                "pair": "ETH/USDT:USDT",
                "horizon_minutes": 60,
                "candidate": False,
            },
        ]
    )
    assert validation.observed_statistics(rows) == {
        "candidate_cells": 4,
        "max_candidate_cells_one_family": 3,
        "max_shared_horizons_one_family": 1,
    }


def test_permutation_preserves_sign_counts_within_family_and_partition() -> None:
    catalog = pd.DataFrame(
        [
            {
                "event_id": f"{family}-{partition}-{index}",
                "event_family": family,
                "whole_event_partition": partition,
                "anchor_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index),
                "predicted_crypto_direction": sign,
            }
            for family in ("jobs", "retail")
            for partition in validation.frozen.EVALUATION_PARTITIONS
            for index, sign in enumerate((-1, -1, 1, 1))
        ]
    )
    shuffled = validation.permuted_event_signs(catalog, np.random.default_rng(123))
    catalog["shuffled"] = catalog["event_id"].map(shuffled)
    for _, group in catalog.groupby(["event_family", "whole_event_partition"]):
        assert sorted(group["shuffled"]) == sorted(group["predicted_crypto_direction"])


def test_apply_permutation_keeps_one_sign_across_assets_and_flips_week_control() -> None:
    rows = pd.DataFrame(
        [
            {
                "event_id": "event-a",
                "event_family": "jobs",
                "pair": pair,
                "horizon_minutes": horizon,
                "whole_event_partition": validation.frozen.EVALUATION_PARTITIONS[0],
                "anchor_utc": pd.Timestamp("2024-01-01", tz="UTC"),
                "predicted_direction": 1,
                "actual_direction": 1,
                "ordinary_control_accuracy": 0.75,
                "pretrend_hit": True,
                "previous_release_hit": False,
            }
            for pair in validation.frozen.ASSETS
            for horizon in validation.frozen.HORIZONS_MINUTES
        ]
    )
    changed = validation.apply_permuted_signs(rows, {"event-a": -1})
    assert changed["predicted_direction"].eq(-1).all()
    assert changed["direction_hit"].eq(False).all()
    assert changed["ordinary_control_accuracy"].eq(0.25).all()


def test_validation_summary_uses_plus_one_familywise_probability() -> None:
    observed = {
        "candidate_cells": 4,
        "max_candidate_cells_one_family": 3,
        "max_shared_horizons_one_family": 1,
    }
    null = pd.DataFrame(
        {
            "candidate_cells": [0, 4, 5],
            "max_candidate_cells_one_family": [0, 1, 3],
            "max_shared_horizons_one_family": [0, 0, 1],
        }
    )
    summary = validation.validation_summary(observed, null).set_index("metric")
    assert summary.loc["candidate_cells", "familywise_probability"] == 0.75
    assert summary.loc["max_candidate_cells_one_family", "familywise_probability"] == 0.5
    assert summary.loc["max_shared_horizons_one_family", "familywise_probability"] == 0.5


def test_validated_family_requires_two_markets_and_both_adjusted_checks() -> None:
    candidate_rows = pd.DataFrame(
        [
            {
                "event_family": "jobs",
                "pair": pair,
                "horizon_minutes": horizon,
                "candidate": True,
            }
            for pair in validation.frozen.ASSETS
            for horizon in validation.frozen.HORIZONS_MINUTES
        ]
        + [
            {
                "event_family": "retail",
                "pair": "BTC/USDT:USDT",
                "horizon_minutes": 60,
                "candidate": True,
            }
        ]
    )
    null = pd.DataFrame(
        {
            "max_candidate_cells_one_family": [0] * 99 + [8],
            "max_shared_horizons_one_family": [0] * 99 + [4],
        }
    )
    families = validation.validated_families(candidate_rows, null).set_index("event_family")
    assert families.loc["jobs", "decision"].startswith("retained")
    assert families.loc["retail", "decision"].startswith("parked")
    assert not bool(families.loc["jobs", "independent_confirmation"])
    assert not bool(families.loc["jobs", "profit_used"])
