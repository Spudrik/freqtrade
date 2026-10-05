# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_cache as g14c,
)


def test_merge_eight_hour_features_is_date_exact_and_outcome_blind() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    long = pd.DataFrame({"date": dates, "existing": [1.0, 2.0]})
    mtf = pd.DataFrame({"date": dates})
    for number, column in enumerate(g14c.EIGHT_HOUR_COLUMNS):
        mtf[column] = [float(number), float(number + 1)]

    output = g14c.merge_eight_hour_features(long, mtf)

    assert len(output) == 2
    assert set(g14c.EIGHT_HOUR_COLUMNS).issubset(output.columns)
    assert not any(column.startswith("&-") for column in output.columns)


def test_merge_readiness_fills_missing_clock_as_unready() -> None:
    dates = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    left = pd.DataFrame({"date": dates, "target": [0.0, 1.0]})
    right = pd.DataFrame({"date": dates[:1]})
    for column in g14c.READINESS_COLUMNS:
        right[column] = True

    output = g14c.merge_readiness(left, right)

    assert output.loc[0, list(g14c.READINESS_COLUMNS)].all()
    assert not output.loc[1, list(g14c.READINESS_COLUMNS)].any()


def test_merge_rejects_duplicate_feature_dates() -> None:
    date = pd.Timestamp("2026-01-01", tz="UTC")
    long = pd.DataFrame({"date": [date]})
    mtf = pd.DataFrame({"date": [date, date]})
    for column in g14c.EIGHT_HOUR_COLUMNS:
        mtf[column] = [1.0, 2.0]

    with pytest.raises(ValueError, match="duplicate"):
        g14c.merge_eight_hour_features(long, mtf)
