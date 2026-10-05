# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.strategies.MarketReactionZoneFreqAIGeneration11Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
    MarketReactionZoneG11ConfigurableFreqAIResearchStrategy,
)


def test_generation11_training_labels_obey_readiness_and_eight_hour_embargo(
    monkeypatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    events = pd.DataFrame(
        {
            "date": dates,
            "ready__g11_minimal_contact": [False, *([True] * 19)],
        }
    )
    for target in TARGET_COLUMNS:
        events[target] = 1.0
    strategy = object.__new__(
        MarketReactionZoneG11ConfigurableFreqAIResearchStrategy
    )
    strategy.config = {
        "market_reaction_zone_g11": {
            "feature_blocks": ["g11_minimal_contact"],
            "required_ready_blocks": ["g11_minimal_contact"],
            "target_columns": list(TARGET_COLUMNS),
        }
    }
    monkeypatch.setattr(strategy, "_event_cache", lambda pair: events)
    frame = pd.DataFrame({"date": dates.astype("datetime64[ms, UTC]")})

    labelled = strategy.set_freqai_targets(
        frame, metadata={"pair": "BTC/USDT:USDT"}
    )

    cutoff = dates.max() - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS)
    assert labelled.loc[0, list(TARGET_COLUMNS)].isna().all()
    eligible = (dates <= cutoff) & (dates > dates[0])
    assert labelled.loc[eligible, list(TARGET_COLUMNS)].notna().all().all()
    assert labelled.loc[dates > cutoff, list(TARGET_COLUMNS)].isna().all().all()


def test_generation11_rejects_unknown_target() -> None:
    strategy = object.__new__(
        MarketReactionZoneG11ConfigurableFreqAIResearchStrategy
    )
    strategy.config = {
        "market_reaction_zone_g11": {
            "target_columns": ["&-g11_not_a_frozen_target"]
        }
    }

    try:
        strategy._configured_target_columns()
    except ValueError as exc:
        assert "Unknown Generation 11" in str(exc)
    else:
        raise AssertionError("Unknown target should fail closed")
