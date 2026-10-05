# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_freeze as freeze,
)


def test_news_thresholds_use_only_earlier_same_hour_rows() -> None:
    dates = pd.date_range("2020-01-01", periods=24 * 125, freq="1h", tz="UTC")
    values = [100.0] * len(dates)
    frame = DataFrame(
        {
            "date": dates,
            "gdelt_1h": values,
            "gdelt_24h": values,
            "ctx_max_source_available_at": dates,
        }
    )
    target = frame.index[-1]
    frame.loc[target, "gdelt_1h"] = 10000.0
    scored = freeze.add_causal_news_thresholds(frame)
    assert scored.loc[target, "news_q999"] == np.log1p(100.0)
    assert bool(scored.loc[target, "news_spike_raw"])


def test_news_episode_selection_uses_first_crossing_and_cooldown() -> None:
    dates = pd.to_datetime(
        ["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z", "2021-01-10T00:00:00Z"]
    )
    frame = DataFrame(
        {
            "date": dates,
            "gdelt_1h": [1000.0, 2000.0, 3000.0],
            "news_q999": [1.0, 1.0, 1.0],
            "news_spike_raw": [True, True, True],
        }
    )
    selected = freeze.select_news_episodes(frame)
    assert selected["anchor_utc"].tolist() == [dates[0], dates[2]]


def test_layer1_freeze_and_control_contracts_are_available() -> None:
    _, layer1_freeze, events = freeze.load_layer1()
    assert layer1_freeze["outcomes_read"] is False
    assert len(events) == 45
    assert events["anchor_utc"].is_monotonic_increasing
