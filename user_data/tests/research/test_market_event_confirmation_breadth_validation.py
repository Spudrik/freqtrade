from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_confirmation_breadth_validation as validation,
)


def _summary(cells: list[tuple[str, str]], *, passing: bool = False) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "analysis_group": group,
                "route_id": route,
                "leader_type": "btc",
                "response_scope": "btc",
                "total_horizon_hours": 2,
                "meets_conditional_direction_rule": passing,
                "meets_65_target": False,
            }
            for group, route in cells
        ]
    )


def _response(
    cells: list[tuple[str, str]],
    event_count: int = 12,
    *,
    missing: set[tuple[int, int]] | None = None,
    confirmed_by_rank: dict[int, bool] | None = None,
    success_by_rank: dict[int, bool] | None = None,
) -> pd.DataFrame:
    missing = missing or set()
    confirmed_by_rank = confirmed_by_rank or {rank: True for rank in range(13)}
    success_by_rank = success_by_rank or {rank: bool(rank % 2) for rank in range(13)}
    rows: list[dict[str, object]] = []
    for cell_index, (group, route) in enumerate(cells):
        for event_index in range(event_count):
            event_id = f"event_{event_index:02d}"
            partition = "p1" if event_index < event_count // 2 else "p2"
            for rank in range(13):
                if (cell_index, rank) in missing:
                    continue
                rows.append(
                    {
                        "event_id": event_id,
                        "event_family": group,
                        "event_source": route,
                        "leader_type": "btc",
                        "scope": "btc",
                        "horizon_hours": 2,
                        "whole_event_partition": partition,
                        "sample_type": "event" if rank == 0 else "control",
                        "control_type": "event" if rank == 0 else "same_weekday_hour_prior_week",
                        "control_rank": rank,
                        "confirmed": confirmed_by_rank.get(rank, False),
                        "response_usable": True,
                        "direction_usable": True,
                        "continued": success_by_rank.get(rank, False),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _prepared(
    cells: list[tuple[str, str]],
    *,
    event_count: int = 12,
    missing: set[tuple[int, int]] | None = None,
    confirmed_by_rank: dict[int, bool] | None = None,
    success_by_rank: dict[int, bool] | None = None,
) -> validation.PreparedData:
    summary = _summary(cells)
    response = _response(
        cells,
        event_count,
        missing=missing,
        confirmed_by_rank=confirmed_by_rank,
        success_by_rank=success_by_rank,
    )
    return validation.prepare_data(summary, response)


def test_freeze_writes_contract_without_randomizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = [("recent_media", "live_news_activity"), ("recent_media", "live_web_activity")]
    direct_path = tmp_path / "direct.json"
    summary_path = tmp_path / "summary.csv"
    response_path = tmp_path / "response.parquet"
    freeze_path = tmp_path / "freeze.json"
    summary = _summary(cells, passing=True)
    summary.to_csv(summary_path, index=False)
    direct_path.write_text(
        json.dumps(
            {
                "status": "completed_event_confirmation_breadth_direct",
                "outcomes_opened": True,
                "cell_count": 2,
                "passing_cell_count": 2,
            }
        ),
        encoding="utf-8",
    )
    response_path.write_bytes(b"synthetic-response-placeholder")
    monkeypatch.setattr(validation, "DIRECT_RESULT_PATH", direct_path)
    monkeypatch.setattr(validation, "CELL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(validation, "RESPONSE_PARQUET_PATH", response_path)
    monkeypatch.setattr(validation, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(validation, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(validation, "EXPECTED_CELL_COUNT", 2)
    monkeypatch.setattr(validation, "EXPECTED_PASSING_CELL_COUNT", 2)

    def fail_if_randomized(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze must not randomize")

    monkeypatch.setattr(validation, "randomization_maxima", fail_if_randomized)
    plan = validation.freeze_plan(overwrite=True)

    assert plan["status"] == "frozen_post_screen_before_familywise_randomization"
    assert plan["outcomes_already_opened"] is True
    assert plan["cell_universe"]["count"] == 2
    assert plan["randomization"]["seed"] == validation.PERMUTATION_SEED
    assert plan["randomization"]["permutations"] == validation.PERMUTATION_ITERATIONS
    assert plan["groups"]["recent_media"] == list(
        validation.GROUP_ROUTE_IDS["recent_media"]
    )
    assert set(plan["artifacts"]) == {
        "direct_result",
        "cell_summary",
        "response_parquet",
    }


def test_shared_pseudo_event_choice_applies_to_every_cell() -> None:
    prepared = _prepared(
        [
            ("recent_media", "live_news_activity"),
            ("recent_media", "live_web_activity"),
        ],
        event_count=12,
        success_by_rank={rank: rank == 1 for rank in range(13)},
    )
    choices = {event_id: 1 for event_id in prepared.event_ids}
    selected = validation.cell_statistics(prepared, choices)
    assert selected["pair_rate"].tolist() == [1.0, 1.0]
    assert selected["event_only_rate"].tolist() == [1.0, 1.0]

    actual = validation.cell_statistics(prepared, {event_id: 0 for event_id in prepared.event_ids})
    assert actual["pair_rate"].tolist() == [0.0, 0.0]


def test_confirmation_only_rate_collapses_each_event_before_averaging() -> None:
    summary = _summary([("recent_media", "live_news_activity")])
    response = _response(
        [("recent_media", "live_news_activity")],
        event_count=2,
        confirmed_by_rank={
            0: True,
            1: True,
            2: True,
            **{rank: False for rank in range(3, 13)},
        },
        success_by_rank={
            0: False,
            1: True,
            2: False,
            **{rank: False for rank in range(3, 13)},
        },
    )
    response.loc[
        response["event_id"].eq("event_01") & response["control_rank"].eq(1),
        "continued",
    ] = False
    prepared = validation.prepare_data(summary, response)
    stats = validation.cell_statistics(prepared, {event_id: 0 for event_id in prepared.event_ids})

    # Event 0's two controls average to .5; event 1's two controls average to .0.
    # The event-level collapsed control rate is therefore .25, not row-weighted .333.
    assert stats.loc[0, "confirmation_only_rate"] == pytest.approx(0.25)


def test_missing_selected_row_abstains_from_all_selected_set_metrics() -> None:
    prepared = _prepared(
        [("recent_media", "live_news_activity"), ("recent_media", "live_web_activity")],
        event_count=12,
        missing={(0, 1)},
    )
    choices = {event_id: 1 for event_id in prepared.event_ids}
    stats = validation.cell_statistics(prepared, choices)

    assert not validation.metric_arrays(prepared, choices).selected_present[0].all()
    assert stats.loc[0, "selected_usable_event_count"] == 0
    assert stats.loc[0, "confirmed_control_event_count"] == 0
    assert stats.loc[1, "selected_usable_event_count"] == 12


def test_hash_tamper_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cells = [("recent_media", "live_news_activity"), ("recent_media", "live_web_activity")]
    direct_path = tmp_path / "direct.json"
    summary_path = tmp_path / "summary.csv"
    response_path = tmp_path / "response.parquet"
    freeze_path = tmp_path / "freeze.json"
    _summary(cells, passing=True).to_csv(summary_path, index=False)
    direct_path.write_text(
        json.dumps(
            {
                "status": "completed_event_confirmation_breadth_direct",
                "outcomes_opened": True,
                "cell_count": 2,
            }
        ),
        encoding="utf-8",
    )
    response_path.write_bytes(b"synthetic")
    monkeypatch.setattr(validation, "DIRECT_RESULT_PATH", direct_path)
    monkeypatch.setattr(validation, "CELL_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(validation, "RESPONSE_PARQUET_PATH", response_path)
    monkeypatch.setattr(validation, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(validation, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(validation, "EXPECTED_CELL_COUNT", 2)
    monkeypatch.setattr(validation, "EXPECTED_PASSING_CELL_COUNT", 2)
    plan = validation.freeze_plan(overwrite=True)

    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        validation._validate_plan_hashes(plan)


def test_randomization_is_deterministic_for_fixed_seed() -> None:
    prepared = _prepared(
        [
            ("recent_media", "live_news_activity"),
            ("historical_context", "market_fear"),
        ],
        event_count=12,
    )
    partition_config = {
        "recent_media": ("p1", "p2"),
        "historical_cross_asset": ("p1", "p2"),
    }
    first = validation.randomization_maxima(
        prepared,
        iterations=8,
        seed=validation.PERMUTATION_SEED,
        partitions_by_group=partition_config,
    )
    second = validation.randomization_maxima(
        prepared,
        iterations=8,
        seed=validation.PERMUTATION_SEED,
        partitions_by_group=partition_config,
    )
    pd.testing.assert_frame_equal(first, second)


def test_global_maximum_is_maximum_of_family_maxima() -> None:
    prepared = _prepared(
        [
            ("recent_media", "live_news_activity"),
            ("historical_context", "market_fear"),
        ],
        event_count=12,
    )
    maxima = validation.randomization_maxima(
        prepared,
        iterations=20,
        seed=7,
        partitions_by_group={
            "recent_media": ("p1", "p2"),
            "historical_cross_asset": ("p1", "p2"),
        },
    )
    families = maxima[["recent_media", "historical_confluence", "historical_cross_asset"]]
    expected = families.max(axis=1, skipna=True)
    np.testing.assert_allclose(
        maxima["global"].to_numpy(), expected.to_numpy(), equal_nan=True
    )


def test_familywise_p_value_uses_add_one_and_wilson_interval() -> None:
    exceedances, probability, lower, upper = validation._familywise_comparison(
        0.5, np.asarray([0.6, 0.5, 0.4, np.nan])
    )
    assert exceedances == 2
    assert probability == pytest.approx(3 / 5)
    assert 0.0 <= lower < probability < upper <= 1.0


def test_observed_result_rows_preserve_cell_keys_after_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = [("recent_media", "live_news_activity")]
    summary = _summary(cells, passing=True)
    prepared = _prepared(cells)
    observed = validation.cell_statistics(prepared)
    maxima = pd.DataFrame(
        {
            "permutation": [1, 2],
            "recent_media": [0.0, 0.1],
            "historical_confluence": [np.nan, np.nan],
            "historical_cross_asset": [np.nan, np.nan],
            "global": [0.0, 0.1],
        }
    )
    monkeypatch.setattr(validation, "EXPECTED_PASSING_CELL_COUNT", 1)

    result = validation._observed_result_rows(summary, observed, maxima)

    assert result.loc[0, "analysis_group"] == "recent_media"
    assert result.loc[0, "route_id"] == "live_news_activity"
    assert result.loc[0, "leader_type"] == "btc"
