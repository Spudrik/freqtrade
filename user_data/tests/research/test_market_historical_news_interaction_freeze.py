# ruff: noqa: S101

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_historical_news_interaction_freeze"
)
freeze = importlib.import_module(MODULE)


def feature_frame(rows: int = 240) -> DataFrame:
    frame = DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "news_any_present": False,
            "gkg_present": False,
            "bad_news_escalating": False,
            "news_activity__pct_rank_720h": 0.0,
            "macro_stress__z_168h": 0.0,
            "macro_stress__delta_6h": 0.0,
            "crypto_stress__z_168h": 0.0,
            "near_range_low": False,
            "near_range_high": False,
            "break_lower_low_24h": False,
            "break_higher_high_24h": False,
            "compression_state": False,
            "ret_6h": 0.0,
            "ret_24h": 0.0,
            "volume_z_24h": 0.0,
            "volatility_24h": 0.01,
            "range_position_30d": 0.5,
        }
    )
    return frame


def theory(theory_id: str) -> dict[str, object]:
    return next(value for value in freeze.THEORIES if value["theory_id"] == theory_id)


def test_macro_relief_mask_and_stale_copy_use_only_past_source_values() -> None:
    frame = feature_frame()
    frame.loc[10, "news_any_present"] = True
    frame.loc[4, "macro_stress__z_168h"] = 2.0
    frame.loc[10, "macro_stress__delta_6h"] = -1.0
    frame.loc[[10, 178], "near_range_low"] = True
    source, market = freeze.build_component_masks(
        frame, theory("macro_relief_near_low")
    )
    stale_source, _ = freeze.build_component_masks(
        frame,
        theory("macro_relief_near_low"),
        source_lag_hours=168,
    )
    assert source.sum() == 1
    assert source.iloc[10] and market.iloc[10]
    assert stale_source.sum() == 1
    assert stale_source.iloc[178]


def test_cluster_mask_uses_gap_from_previous_raw_trigger() -> None:
    frame = feature_frame(rows=60)
    mask = pd.Series(False, index=frame.index)
    mask.iloc[[0, 1, 25, 50]] = True
    clusters = freeze.cluster_mask(frame["date"], mask, gap_hours=24)
    assert clusters["feature_candle_open_utc"].tolist() == [
        frame.loc[0, "date"],
        frame.loc[50, "date"],
    ]
    assert clusters["raw_trigger_rows"].tolist() == [3, 1]


def test_event_exclusion_rejects_controls_at_or_within_24_hours() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC"))
    allowed = freeze.outside_event_exclusion(
        dates, pd.Series([dates.iloc[24]]), hours=24
    )
    assert not allowed.iloc[0]
    assert not allowed.iloc[48]
    assert allowed.iloc[49]


def test_matching_is_same_month_and_without_reuse() -> None:
    frame = feature_frame(rows=24 * 40)
    events = DataFrame(
        {
            "episode_id": ["a", "b"],
            "feature_candle_open_utc": [frame.loc[100, "date"], frame.loc[200, "date"]],
        }
    )
    candidates = DataFrame(
        {
            "feature_candle_open_utc": [
                frame.loc[110, "date"],
                frame.loc[210, "date"],
                frame.loc[800, "date"],
            ]
        }
    )
    frame.loc[100, "ret_6h"] = 1.0
    frame.loc[200, "ret_6h"] = 2.0
    frame.loc[110, "ret_6h"] = 1.1
    frame.loc[210, "ret_6h"] = 2.1
    matches = freeze.match_controls(
        frame,
        events,
        candidates,
        theory_id="test",
        window_id="window",
        control_kind="market_only",
        match_columns=("ret_6h",),
    )
    assert len(matches) == 2
    assert matches["control_feature_candle_open_utc"].nunique() == 2
    assert all(
        event.year == control.year and event.month == control.month
        for event, control in zip(
            matches["event_feature_candle_open_utc"],
            matches["control_feature_candle_open_utc"],
            strict=True,
        )
    )


def test_freeze_feature_contract_contains_no_outcomes() -> None:
    forbidden = ("future", "profit", "target", "label")
    assert not any(
        token in column.lower()
        for column in freeze.FEATURE_COLUMNS
        for token in forbidden
    )
    assert np.isfinite(freeze.MINIMUM_JOINT_RATE)
