# ruff: noqa: S101

from __future__ import annotations

import numpy as np

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_cpi_transmission_validation as module,
)


def synthetic_route() -> module.TransmissionRoute:
    return module.TransmissionRoute(
        key=("btc", 15, 60, "group"),
        source_call=np.ones(10, dtype=bool),
        source_sign=np.ones(10),
        later_sign=np.asarray([1, 1, 1, 1, 1, 1, -1, -1, -1, -1]),
        own_sign=np.asarray([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),
        partitions=np.asarray(
            ["development_2021_2023"] * 5
            + ["internal_validation_2024_2025"] * 5
        ),
        control_rate=0.40,
    )


def test_transmission_measurement_uses_every_frozen_gate() -> None:
    rule = {
        "minimum_issued_whole_events": 10,
        "minimum_overall_direction_rate": 0.55,
        "minimum_each_partition_rate": 0.5,
        "minimum_control_uplift": 0.05,
    }

    measured = module.transmission_route_measurement(
        synthetic_route(), np.arange(10), rule
    )

    assert measured["event_direction_rate"] == 0.6
    assert measured["development_direction_rate"] == 1.0
    assert measured["validation_direction_rate"] == 0.2
    assert measured["all_unchanged_gates_pass"] is False


def test_shuffle_never_moves_event_across_partition() -> None:
    partitions = np.asarray(
        ["development_2021_2023"] * 4 + ["internal_validation_2024_2025"] * 3
    )
    shuffled = module.shuffled_source_indexes(partitions, np.random.default_rng(4))

    assert set(shuffled[:4]) == {0, 1, 2, 3}
    assert set(shuffled[4:]) == {4, 5, 6}


def test_candidate_call_uses_follower_specific_own_sign() -> None:
    rows = module.DataFrame(
        {
            "ada_own_sign": [1.0, -1.0],
            "established_group_median_own_sign": [-1.0, 1.0],
        }
    )

    issued, sign = module.candidate_call(
        rows, "follower_own_first_move", "ADA/USDT:USDT"
    )

    assert issued.tolist() == [True, True]
    assert sign.tolist() == [1.0, -1.0]
