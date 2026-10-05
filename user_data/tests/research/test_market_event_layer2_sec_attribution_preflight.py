# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_sec_attribution_preflight as module,
)


def test_sec_cluster_inventory_matches_frozen_earnings_events() -> None:
    inventory = module.build_sec_clock_inventory()

    assert len(inventory) > 0
    assert inventory["cluster_id"].is_unique
    assert not inventory["exact_clock_usable"].any()


def test_daily_fred_history_is_never_used_as_fifteen_minute_control() -> None:
    row = module.fred_daily_record(
        "TEST",
        {
            "observations": [
                {"date": "2021-01-01", "value": "100"},
                {"date": "2021-01-04", "value": "101"},
            ]
        },
    )

    assert row["historical_period_overlap"] is True
    assert row["eligible_for_15m_control"] is False
    assert row["timestamp_resolution"] == "date_only_daily_close"


def test_attribution_requires_clocks_and_intraday_controls() -> None:
    clocks = module.DataFrame(
        {
            "exact_clock_usable": [True] * 8 + [False] * 2,
        }
    )
    controls = module.DataFrame(
        {
            "eligible_for_15m_control": [False],
        }
    )
    route = {"minimum_first_public_clock_coverage": 0.8}
    parent = {"familywise_p_value": 0.004}

    decision = module.attribution_decision(clocks, controls, route, parent)

    assert decision["exact_first_public_clock_coverage"] == 0.8
    assert decision["attribution_gate_pass"] is False
    assert decision["association_decision"].startswith("retained")
