# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)


def test_selected_level_specs_excludes_previously_null_and_sparse_ranks() -> None:
    cache = pd.DataFrame(
        {
            "vp_poc": [1.0],
            "vp_native_half_width": [0.1],
            "vp_score_abs": [1.0],
            "vp_value_area_width_pct": [0.1],
            "vp_state": [1.0],
            "generic_ema_20": [1.0],
            "tlv2_resistance_line_rank0": [1.0],
            "tlv2_resistance_slope_rank0": [0.0],
            "tlv2_resistance_line_width_rank0": [0.1],
            "tlv2_resistance_score_rank0": [1.0],
            "tlv2_resistance_line_id_rank0": [1.0],
            "tlv2_resistance_pivot_count_rank0": [2.0],
            "tlv2_resistance_absorbed_pivot_count_rank0": [0.0],
            "tlv2_resistance_line_rank1": [1.0],
            "tlv2_resistance_slope_rank1": [0.0],
            "tlv2_resistance_line_width_rank1": [0.1],
            "tlv2_resistance_score_rank1": [1.0],
            "tlv2_resistance_line_id_rank1": [1.0],
            "tlv2_resistance_pivot_count_rank1": [2.0],
            "tlv2_resistance_absorbed_pivot_count_rank1": [0.0],
        }
    )

    specs = g6.selected_level_specs(cache)
    identities = {(spec.family, spec.name, spec.representation) for spec in specs}

    assert ("volume_profile_settled", "poc", "settled") in identities
    assert not any(family == "generic_moving_average" for family, _, _ in identities)
    assert ("tlv2_ranked", "resistance_rank0", "projected") in identities
    assert not any("rank1" in name for _, name, _ in identities)


def test_causal_state_features_do_not_use_current_candle() -> None:
    rows = 120
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, rows))
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 1400.0, rows),
            "base_atr": 2.0,
            "period": "validation_early",
        }
    )
    original, _ = g6.causal_state_features(base)
    changed = base.copy()
    changed.loc[rows - 1, ["high", "low", "close", "volume"]] = [1000.0, 1.0, 900.0, 1e9]
    revised, _ = g6.causal_state_features(changed)

    pd.testing.assert_series_equal(original.iloc[-1], revised.iloc[-1])


def test_supported_cell_requires_both_periods_and_five_coins() -> None:
    records = []
    for period in ("validation_early", "validation_late"):
        for pair in ("A", "B", "C", "D", "E"):
            records.append(
                {
                    "cohort": "normal",
                    "timeframe": "1h",
                    "level_family": "family",
                    "level_name": "level",
                    "representation": "settled",
                    "period": period,
                    "pair": pair,
                    "event_count": 10,
                }
            )
    detail = pd.DataFrame.from_records(records)

    supported = g6.summarize_supported_cells(
        detail,
        keys=("cohort", "timeframe", "level_family", "level_name", "representation"),
    )
    assert bool(supported.iloc[0]["supported"]) is True

    detail.loc[
        detail["period"].eq("validation_late") & detail["pair"].eq("E"),
        "event_count",
    ] = 0
    parked = g6.summarize_supported_cells(
        detail,
        keys=("cohort", "timeframe", "level_family", "level_name", "representation"),
    )
    assert bool(parked.iloc[0]["supported"]) is False


def test_smaller_predeclared_group_can_pass_without_seven_of_ten() -> None:
    records = []
    for period in ("validation_early", "validation_late"):
        for pair in ("ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT"):
            records.append(
                {
                    "cohort": "normal",
                    "pair": pair,
                    "question": "volume_profile_value_area_reaction",
                    "period": period,
                    "event_count": 10,
                }
            )
    result = g6.summarize_group_transfer(pd.DataFrame.from_records(records), pd.DataFrame())
    row = result.loc[
        result["group_id"].eq("smart_contract_platforms")
        & result["question"].eq("volume_profile_value_area_reaction")
    ].iloc[0]

    assert bool(row["supported"]) is True


def test_timeframe_masks_keep_agreement_and_opposition_distinct() -> None:
    native_upper = np.array([True, False, True, False])
    native_lower = np.array([False, True, False, False])
    higher_upper = np.array([True, False, False, True])
    higher_lower = np.array([False, True, True, False])
    mechanism = {
        ("swing", "1h"): native_upper,
        ("profile", "1h"): native_lower,
        ("swing", "4h"): higher_upper,
        ("profile", "4h"): higher_lower,
    }
    sides = {
        ("upper", "1h"): native_upper,
        ("lower", "1h"): native_lower,
        ("upper", "4h"): higher_upper,
        ("lower", "4h"): higher_lower,
    }

    masks = g6.timeframe_relationship_masks(
        mechanism_contacts=mechanism,
        side_contacts=sides,
        row_count=4,
    )

    assert masks[("4h", "same_mechanism_agreement")].tolist() == [True, True, False, False]
    assert masks[("4h", "opposing_side_overlap")].tolist() == [False, False, True, False]
