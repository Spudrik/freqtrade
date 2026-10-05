# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_preflight as g8,
)


def test_generation8_cells_cover_all_frozen_siblings_without_direction_targets() -> None:
    assert len(g8.CELLS) == 19
    assert len({cell.branch_id for cell in g8.CELLS}) == 8
    assert sum(cell.branch_id.startswith("g8h_") for cell in g8.CELLS) == 6
    assert all(cell.parent_ready.startswith("g7_common_") for cell in g8.CELLS)
    assert all(
        not column.startswith("&-")
        for cell in g8.CELLS
        for column in g8.required_predictor_columns(cell)
    )


def test_orderbook_attribution_is_limited_to_smart_contract_platforms() -> None:
    cell = next(cell for cell in g8.CELLS if cell.branch_id.startswith("g8g_"))
    manifest = {
        "pairs": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
        ]
    }
    assert g8.selected_pairs(cell, manifest) == (
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
    )


def test_predictor_bins_are_frozen_from_development_predictors_only() -> None:
    cell = next(
        cell
        for cell in g8.CELLS
        if cell.branch_id.startswith("g8e_") and cell.cohort == "normal"
    )
    columns = g8.required_predictor_columns(cell)
    rows = 180
    frame = pd.DataFrame(
        {
            "period": ["development"] * 150 + ["validation_early"] * 30,
            **{
                column: np.concatenate(
                    [np.arange(150, dtype=float), np.full(30, 100_000.0)]
                )
                for column in columns
            },
        }
    )
    assert len(frame) == rows
    records = g8.predictor_bin_rows(cell, frame)
    assert records
    assert all(record["reaction_outcomes_opened"] is False for record in records)
    assert all(record["upper_boundary"] < 150 for record in records)


def test_cell_support_requires_both_validations_and_five_coins() -> None:
    cell = next(
        cell
        for cell in g8.CELLS
        if cell.branch_id.startswith("g8a_") and cell.cohort == "normal"
    )
    records = []
    for period, rows_per_pair in (
        ("development", 25),
        ("validation_early", 12),
        ("validation_late", 12),
    ):
        for index in range(5):
            for row in range(rows_per_pair):
                records.append(
                    {"pair": f"PAIR{index}", "period": period, "row": row}
                )
    support = g8.cell_support_record(cell, pd.DataFrame.from_records(records))
    assert support["status"].startswith("supported")
    assert support["reaction_outcomes_opened"] is False


def test_representation_contract_keeps_profit_and_direction_closed() -> None:
    contract = g8.representation_contract()
    assert contract["outcome_boundary"] == {
        "profit": False,
        "future_signed_direction": False,
        "entry_exit_construction": False,
        "reaction_targets_read_by_preflight": False,
    }
    assert contract["prior_reaction_history"]["allowed_only_after_support_freeze"]
    assert "active zone half-width" in contract["lineage_identity"]["deterministic_fallback"]
