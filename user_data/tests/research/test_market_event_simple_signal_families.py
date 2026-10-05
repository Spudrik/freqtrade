# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_simple_signal_families as simple,
)


def test_simple_surface_covers_exactly_five_families() -> None:
    assert len({item["family_id"] for item in simple.SIGNALS}) == 5
    assert len(simple.SIGNALS) == 6


def test_plain_signal_operators() -> None:
    values = pd.Series([-1.0, 0.0, 1.0])
    assert simple.issue_signal(
        values, {"operator": "less_than", "threshold": 0.0}
    ).tolist() == [True, False, False]
    assert simple.issue_signal(
        values, {"operator": "greater_than_or_equal", "threshold": 0.0}
    ).tolist() == [False, True, True]
    assert simple.issue_signal(
        pd.Series([0.1, 0.5, 0.9]),
        {"operator": "outside_open_interval", "threshold": [0.2, 0.8]},
    ).tolist() == [True, False, True]


def test_false_text_is_not_coerced_to_true() -> None:
    assert simple._coerce_bool(pd.Series(["true", "false", None])).tolist() == [
        True,
        False,
        False,
    ]
    assert simple._coerce_bool(pd.Series([1.0, 0.0, float("nan")])).tolist() == [
        True,
        False,
        False,
    ]
