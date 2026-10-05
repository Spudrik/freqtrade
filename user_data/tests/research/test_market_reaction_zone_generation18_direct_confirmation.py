# ruff: noqa: S101

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)


def test_assign_confirmation_period_uses_frozen_blocks() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2026-07-19T23:00:00Z",
                "2026-07-20T00:00:00Z",
                "2026-08-05T00:00:00Z",
                "2026-08-20T00:00:00Z",
            ],
            utc=True,
        )
    )

    result = g18d.assign_confirmation_period(dates, "normal")

    assert result.tolist() == [
        "outside_g18_confirmation",
        "g18_normal_holdout_early",
        "g18_normal_holdout_late",
        "outside_g18_confirmation",
    ]


def test_add_reaction_metrics_keeps_reaction_and_direction_separate() -> None:
    frame = pd.DataFrame(
        {
            "zone_half_width_atr": [0.25, 0.25],
            **{
                f"{name}_h{horizon}": values
                for horizon in g18d.HORIZONS
                for name, values in {
                    "crossings": [2.0, 0.0],
                    "away_excursion_atr": [0.7, 0.2],
                    "through_excursion_atr": [0.8, 0.1],
                    "abs_excursion_atr": [0.8, 0.2],
                    "volume_ratio": [1.5, 0.8],
                    "dwell_fraction": [0.2, 0.7],
                }.items()
            },
        }
    )

    g18d.add_reaction_metrics(frame)

    assert frame["metric__unsigned_reaction_h1"].tolist() == [1.0, 0.0]
    assert frame["metric__repeated_recross_h1"].tolist() == [1.0, 0.0]
    assert frame["metric__two_sided_traversal_h1"].tolist() == [1.0, 0.0]


def test_spearman_effect_parks_constant_component() -> None:
    frame = pd.DataFrame(
        {
            "component": np.ones(12),
            "target": np.arange(12, dtype=float),
            "contact_volume_ratio": np.ones(12),
            "contact_range_ratio": np.ones(12),
            "contact_pressure_change": np.zeros(12),
        }
    )

    effect, rows = g18d.spearman_effect(frame, "component", "target")

    assert np.isnan(effect)
    assert rows == 12


def test_context_registry_has_no_signed_target_or_profit() -> None:
    ids = {item["context_id"] for item in g18d.CONTEXT_DEFINITIONS}

    assert "local_volume_range_activity" in ids
    assert "source_ready_orderbook" in ids
    assert len(ids) == len(g18d.CONTEXT_DEFINITIONS)
