# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_preflight as g8p,
)


def test_every_cell_has_a_distinct_ready_name_and_profile_blocks() -> None:
    names = [g8.cell_ready_name(cell) for cell in g8p.CELLS]
    assert len(names) == len(set(names)) == 19
    assert all(g8.cell_blocks(cell) for cell in g8p.CELLS)
    assert all("g8_level_base" in g8.cell_blocks(cell) for cell in g8p.CELLS)
    participation = {
        "g8_participation_relative_volume",
        "g8_participation_volume_acceleration",
        "g8_participation_absolute_pressure",
        "g8_participation_pressure_persistence",
    }
    assert all(
        participation.issubset(g8.cell_blocks(cell))
        for cell in g8p.CELLS
        if not cell.branch_id.startswith("g8g_")
    )


def test_segment_history_separates_first_occupancy_and_retests() -> None:
    valid = np.ones(16, dtype=bool)
    contact = np.array(
        [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ]
    )
    level = np.full(16, 100.0)
    width = np.full(16, 1.0)
    history = g8.segment_history(
        valid=valid,
        contact=contact,
        level=level,
        width=width,
        explicit_identity=None,
    )
    assert history["first"][0] == 1.0
    assert history["occupancy"][1] == 1.0
    assert history["first_retest"][8] == 1.0
    assert history["later_retest"][15] == 1.0
    assert history["prior_episodes"][15] == 2.0


def test_zone_width_move_starts_a_new_fallback_lineage() -> None:
    valid = np.ones(4, dtype=bool)
    contact = np.ones(4, dtype=bool)
    level = np.array([100.0, 100.2, 102.0, 102.1])
    width = np.full(4, 1.0)
    history = g8.segment_history(
        valid=valid,
        contact=contact,
        level=level,
        width=width,
        explicit_identity=None,
    )
    assert history["first"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert history["occupancy"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert history["persistence"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_stale_and_shuffle_controls_have_no_timestamp_or_self_match() -> None:
    dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "period": "development",
            "demo__value": np.arange(200, dtype=float),
        }
    )
    output, audit = g8.attach_stale_and_shuffled(
        frame, pair="BTC/USDT:USDT", blocks=("demo",)
    )
    assert audit == [
        {
            "block": "demo",
            "stale_hours": 72,
            "stale_timestamp_violations": 0,
            "shuffled_self_matches": 0,
        }
    ]
    assert output["demo_stale__value"].notna().sum() == 128
    assert output["demo_shuffled__value"].notna().all()
    assert not np.array_equal(
        output["demo__value"].to_numpy(), output["demo_shuffled__value"].to_numpy()
    )


def test_run_length_counts_consecutive_hourly_band_state() -> None:
    values = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, np.nan, 1.0])
    assert g8.run_length(values).tolist()[:6] == [1.0, 2.0, 1.0, 2.0, 3.0, 1.0]
    assert np.isnan(g8.run_length(values)[6])
    assert g8.run_length(values)[7] == 1.0


def test_shuffle_uses_only_complete_sparse_source_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    values = np.full(100, np.nan)
    values[20:] = np.arange(80, dtype=float)
    frame = pd.DataFrame(
        {"date": dates, "period": "development", "sparse__value": values}
    )
    output, audit = g8.attach_stale_and_shuffled(
        frame, pair="ETH/USDT:USDT", blocks=("sparse",)
    )
    assert audit[0]["shuffled_self_matches"] == 0
    assert output.loc[:19, "sparse_shuffled__value"].isna().all()
    assert output.loc[20:, "sparse_shuffled__value"].notna().all()


def test_stale_control_preserves_timezone_when_a_period_has_no_source_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "period": ["development"] * 4 + ["validation"] * 4,
            "sparse__value": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
        }
    )

    output, audit = g8.attach_stale_and_shuffled(
        frame, pair="SOL/USDT:USDT", blocks=("sparse",)
    )

    assert audit[0]["stale_timestamp_violations"] == 0
    validation = output.loc[output["period"] == "validation"]
    assert validation["sparse_stale__value"].isna().all()
    assert validation["sparse_shuffled__value"].isna().all()
