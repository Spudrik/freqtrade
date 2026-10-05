# ruff: noqa: S101

from __future__ import annotations

import numpy as np

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_activity_family_validation as module,
)


def test_candidate_activity_scores_rotates_event_inside_matched_set() -> None:
    metrics = np.asarray(
        [
            [2.0, 4.0, 6.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )

    scores = module.candidate_activity_scores(metrics)

    assert scores[0] == 2.0
    assert scores[1] == 1.0


def test_candidate_activity_scores_requires_six_remaining_controls() -> None:
    metrics = np.ones((6, 3), dtype=float)

    scores = module.candidate_activity_scores(metrics)

    assert np.isnan(scores).all()


def test_route_statistic_uses_weakest_required_gate() -> None:
    scores = np.asarray([1.4, 1.3, 1.2, 0.9, 0.8, 1.4, 1.3, 1.2, 1.1, 0.8])
    cells = (
        module.CellSpec(np.arange(0, 5), 5),
        module.CellSpec(np.arange(5, 10), 5),
    )

    statistic = module.route_statistic(
        scores, cells, minimum_score=1.2, minimum_rate=0.55
    )

    expected = min(1.2 / 1.2, 3 / 5 / 0.55, 1.2 / 1.2, 4 / 5 / 0.55)
    assert statistic == expected


def test_selected_scores_keeps_one_choice_per_whole_event() -> None:
    matrix = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )
    event_codes = np.asarray([0, 0, 1])
    choices = np.asarray([2, 1])

    selected = module.selected_scores(matrix, event_codes, choices)

    assert selected.tolist() == [12.0, 22.0, 31.0]


def test_sec_connected_family_uses_weaker_asset() -> None:
    statistics = {
        ("source", "BTC/USDT:USDT", "15"): 1.4,
        ("source", "ETH/USDT:USDT", "15"): 1.1,
    }

    combined = module.sec_connected_family_statistics(statistics)

    assert combined[("source", "15")] == 1.1


def test_parent_comparison_allows_only_new_mechanical_coverage() -> None:
    rows = module.DataFrame(
        {
            "parent_activity_score": [1.2, np.nan],
            module.SCORE_COLUMNS[0]: [1.2, 1.1],
        }
    )

    comparison = module.compare_parent_scores(rows)

    assert comparison["newly_eligible_rows"] == 1
    assert comparison["parent_finite_current_missing_rows"] == 0
    assert comparison["existing_rows_with_changed_score"] == 0
