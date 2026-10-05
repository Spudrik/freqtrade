# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_anchor_contexts as g15,
)


def test_lifecycle_bins_only_unambiguous_single_level_contacts() -> None:
    frame = pd.DataFrame(
        {
            "g8_level_identity__contacted_level_count": [1, 1, 1, 1, 1, 2],
            "g8_level_history__prior_contact_available_fraction": [0, 1, 1, 1, 1, 0],
            "g8_level_history__mean_hours_since_prior_contact": [
                float("nan"),
                12,
                48,
                240,
                800,
                float("nan"),
            ],
        }
    )

    result = g15.classify_lifecycle(frame)

    assert result.tolist() == [
        "fresh_30d",
        "repeat_within_24h",
        "repeat_1d_to_7d",
        "repeat_7d_to_30d",
        "fresh_30d",
        "ambiguous",
    ]


def test_orderbook_state_requires_ready_and_excludes_middle() -> None:
    ready = pd.Series([True, True, True, False])
    regime = pd.Series(["active", "quiet", "middle", "active"])

    result = g15.orderbook_state(ready, regime)

    assert result.tolist() == [
        "active",
        "quiet",
        "unavailable_or_middle",
        "unavailable_or_middle",
    ]


def test_activity_contract_requires_both_component_ablations_and_quiet() -> None:
    primary, placebos = g15.comparison_contract(
        "contact_activity_combinations", "high_volume_and_range"
    )

    assert primary == (
        "versus_high_volume_only",
        "versus_high_range_only",
        "versus_quiet_contact",
    )
    assert placebos == ()


def test_orderbook_recent_cell_declares_only_supported_early_period() -> None:
    periods = g15.expected_periods(
        "historical_btc_orderbook_conditioning",
        "normal",
        "recent_normal_chronology",
    )

    assert periods == ("recent_confirmation_early",)
