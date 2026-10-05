# ruff: noqa: S101

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_direct as direct,
)


def test_window_and_pretrend_do_not_overlap() -> None:
    frame = DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [1.0] * 6,
        }
    )
    response = direct._future_window(frame, 3, 2)
    assert response is not None
    assert np.isclose(response["response_return"], 104.5 / 103.0 - 1.0)
    assert np.isclose(direct._pre_return(frame, 3, 2), 103.0 / 101.0 - 1.0)


def test_hotter_inflation_maps_to_negative_crypto_direction() -> None:
    outcomes = DataFrame.from_records(
        [
            {
                "event_id": "one",
                "response_return": -0.01,
                "pre_return": 0.01,
            },
            {
                "event_id": "two",
                "response_return": 0.02,
                "pre_return": -0.01,
            },
        ]
    )
    catalog = DataFrame(
        {
            "event_id": ["one", "two"],
            **{
                column: [1.0, -1.0]
                for column in direct.frozen.ALL_SIGN_COLUMNS
            },
        }
    )
    rows = direct.expand_direction_rows(outcomes, catalog)
    selected = rows.loc[rows["sign_id"].eq(direct.frozen.ALL_SIGN_COLUMNS[0])]
    assert selected["predicted_direction"].tolist() == [-1.0, 1.0]
    assert selected["direction_hit"].tolist() == [True, True]


def _summary_rows(validation_accuracy: float) -> DataFrame:
    common = {
        "sample_variant": "all_releases",
        "clock": "immediate_1m",
        "scope": "btc",
        "horizon": 30,
        "horizon_unit": "minutes",
        "sign_id": "temperature_core_mom",
        "always_up_accuracy": 0.50,
        "always_down_accuracy": 0.50,
        "best_constant_accuracy": 0.50,
        "pretrend_accuracy": 0.50,
        "previous_release_sign_accuracy": 0.50,
        "median_response_return": 0.0,
    }
    return DataFrame.from_records(
        [
            {
                **common,
                "partition": direct.EVALUATION_PARTITIONS[0],
                "n_events": 20,
                "accuracy": 0.60,
            },
            {
                **common,
                "partition": direct.EVALUATION_PARTITIONS[1],
                "n_events": 20,
                "accuracy": validation_accuracy,
            },
            {
                **common,
                "partition": "development_plus_validation",
                "n_events": 40,
                "accuracy": (0.60 + validation_accuracy) / 2,
            },
        ]
    )


def test_route_requires_cross_period_floor_and_control_uplift() -> None:
    retained = direct.route_decisions(_summary_rows(0.60)).iloc[0]
    assert retained["verdict"] == "retained_cross_period_direction_lead"
    rejected = direct.route_decisions(_summary_rows(0.50)).iloc[0]
    assert rejected["verdict"] == "rejected_below_55_percent_cross_period"
