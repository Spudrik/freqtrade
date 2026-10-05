# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer5_freeze as freeze,
)


def test_layer5_routes_keep_only_repeated_simple_and_pair_leads() -> None:
    routes = freeze.confirmation_routes()
    assert len(routes) == 6
    assert not routes["outcomes_opened"].any()
    assert (routes["qualifying_events_observed"] == 0).all()
    assert (routes["minimum_qualifying_events"] == 10).all()
    assert set(routes["parent_lead"]) == {
        "official_event_to_market_activity",
        "gdelt_spike_to_market_activity",
        "background_plus_event_confirmation",
        "event_plus_local_technical_state",
    }


def test_official_confirmation_events_remain_unopened_and_prospective() -> None:
    events = freeze.prospective_official_events()
    assert len(events) == 11
    assert not events["outcomes_opened"].any()
    assert events["whole_event_partition"].eq(
        "prospective_untouched_after_freeze"
    ).all()
    assert events["anchor_utc"].is_monotonic_increasing
