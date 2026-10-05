# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_continuous_context as g23x,
)


def synthetic_context() -> pd.DataFrame:
    rows = 1200
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    data: dict[str, object] = {"date": dates}
    values = np.arange(rows, dtype=float)
    for window in g23x.CONTEXT_WINDOWS:
        for input_name in g23x.CONTEXT_INPUTS:
            data[g23x.context_column(input_name, window)] = values
    return pd.DataFrame(data)


def test_context_registry_creates_broad_single_input_tertiles(monkeypatch) -> None:
    context = synthetic_context()
    monkeypatch.setattr(
        g23x.g22d,
        "CALIBRATION_END_EXCLUSIVE",
        pd.Timestamp("2026-03-01T00:00:00Z"),
    )

    registry = g23x.context_registry(context)
    expanded = g23x.context_with_bands(context, registry)

    assert len(registry) == 10
    assert set(expanded["band__w4__btc_absolute_return_over_atr"].unique()) == {
        "low",
        "middle",
        "high",
    }


def test_attach_context_expands_one_event_to_ten_independent_questions(
    monkeypatch,
) -> None:
    context = synthetic_context()
    monkeypatch.setattr(
        g23x.g22d,
        "CALIBRATION_END_EXCLUSIVE",
        pd.Timestamp("2026-03-01T00:00:00Z"),
    )
    registry = g23x.context_registry(context)
    context = g23x.context_with_bands(context, registry)
    events = pd.DataFrame(
        {
            "event_time": [context["date"].iloc[1100]],
            "control": ["actual"],
            "pre_distance_atr": [0.1],
        }
    )

    expanded = g23x.attach_context_bands(events, context, registry)

    assert len(expanded) == 10
    assert expanded[["context_window_hours", "context_input"]].drop_duplicates().shape[0] == 10


def test_continuous_context_registry_matches_frozen_route() -> None:
    frozen = g23x.load_freeze()

    assert frozen["status"] == "frozen_before_generation23_outcomes"
    assert len(g23x.CONTROLS) == 6
