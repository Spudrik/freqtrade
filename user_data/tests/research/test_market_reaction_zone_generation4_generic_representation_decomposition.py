# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    CORE_STATE_FEATURES,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_generic_representation_decomposition import (  # noqa: E501
    DIRECT_NO_LEVEL,
    LOCATION_CONTROLS,
    OUTCOMES,
    QUESTION_BOLLINGER,
    QUESTION_MA_BUNDLE,
    QUESTION_SMA50,
    assign_bands,
    attach_question_rows,
    canonical_period_map,
    canonical_periods,
    exact_residual_assessment,
    monotonic_orientation,
    question_mask,
    representation_event_rows,
    supported_question_timeframes,
    validate_frozen_branch,
)


def test_frozen_g4e_contract_is_available() -> None:
    branch = validate_frozen_branch()

    assert branch["id"] == "g4e_generic_level_representation_decomposition"
    assert branch["iteration_cap"] == 4


def test_assign_bands_preserves_boundaries_and_missing() -> None:
    edge = {
        "distinct_three_band_edges": True,
        "low_upper_edge": 0.25,
        "middle_upper_edge": 0.75,
    }

    result = assign_bands(pd.Series([0.0, 0.25, 0.5, 0.75, 1.0, np.nan]), edge)

    assert result.tolist()[:5] == ["low", "low", "middle", "middle", "high"]
    assert pd.isna(result.iloc[5])


def test_meme_period_ids_are_mapped_by_manifest_roles() -> None:
    manifest = {
        "data": {
            "chronological_periods": [
                {"id": "meme_development", "role": "development"},
                {
                    "id": "meme_validation_early",
                    "role": "chronological_internal_validation",
                },
                {
                    "id": "meme_validation_late",
                    "role": "chronological_internal_validation",
                },
            ]
        }
    }
    mapping = canonical_period_map(manifest)

    result = canonical_periods(
        pd.Series(["meme_development", "meme_validation_early", "outside"]),
        mapping,
    )

    assert result.tolist() == ["development", "validation_early", "outside"]


def test_question_masks_keep_the_three_frozen_questions_distinct() -> None:
    frame = pd.DataFrame(
        {
            "source_timeframe": ["1h", "4h", "8h", "1d"],
            "density_family": [
                "bollinger_band",
                "simple_moving_average",
                "simple_moving_average",
                "bollinger_band",
            ],
            "level_name": [
                "bollinger_20_upper",
                "sma_20",
                "sma_50",
                "bollinger_20_lower",
            ],
            "actual_generic_scope": [
                "isolated_generic_level",
                "cross_timeframe_generic_cluster",
                "isolated_generic_level",
                "isolated_generic_level",
            ],
        }
    )

    assert question_mask(frame, QUESTION_BOLLINGER).tolist() == [True, False, False, False]
    assert question_mask(frame, QUESTION_MA_BUNDLE).tolist() == [False, True, True, False]
    assert question_mask(frame, QUESTION_SMA50).tolist() == [False, False, False, False]

    frame.loc[2, "source_timeframe"] = "1h"
    assert question_mask(frame, QUESTION_SMA50).tolist() == [False, False, True, False]


def test_attach_question_rows_requires_same_band_and_numeric_common_support() -> None:
    frame = _match_frame()
    states = pd.DataFrame(
        {
            "base_index": [0, 1, 2, 3],
            "distribution_stretch_percentile__1h": [0.10, 0.20, 0.80, 0.90],
            "ma_bundle_distance_atr__1h": [0.10, 0.20, 0.80, 0.90],
        }
    )
    edges = pd.DataFrame(
        {
            "source_timeframe": ["1h", "1h"],
            "representation": [
                "distribution_stretch_percentile",
                "ma_bundle_distance_atr",
            ],
            "low_upper_edge": [0.33, 0.33],
            "middle_upper_edge": [0.66, 0.66],
            "distinct_three_band_edges": [True, True],
        }
    )

    output = attach_question_rows(frame, states=states, edges=edges)
    bollinger = output.loc[output["question"].eq(QUESTION_BOLLINGER)].iloc[0]
    ma = output.loc[output["question"].eq(QUESTION_MA_BUNDLE)].iloc[0]

    assert bollinger["actual_representation_band"] == "low"
    assert bollinger["control_representation_band"] == "low"
    assert bool(bollinger["representation_common_support"])
    assert ma["actual_representation_band"] == "high"
    assert ma["control_representation_band"] == "high"
    assert bool(ma["representation_common_support"])


def test_missing_representation_is_excluded_instead_of_zero_filled() -> None:
    frame = _match_frame().iloc[[0]].copy()
    states = pd.DataFrame(
        {
            "base_index": [0, 1],
            "distribution_stretch_percentile__1h": [0.10, np.nan],
        }
    )
    edges = pd.DataFrame(
        {
            "source_timeframe": ["1h"],
            "representation": ["distribution_stretch_percentile"],
            "low_upper_edge": [0.33],
            "middle_upper_edge": [0.66],
            "distinct_three_band_edges": [True],
        }
    )

    output = attach_question_rows(frame, states=states, edges=edges)

    assert pd.isna(output.iloc[0]["control_representation_band"])
    assert not bool(output.iloc[0]["representation_common_support"])


def test_outcomes_open_only_after_both_windows_and_periods_have_support() -> None:
    rows = []
    for question in (QUESTION_BOLLINGER, QUESTION_SMA50):
        for period in ("validation_early", "validation_late"):
            for response_window in ("h1", "h4"):
                rows.append(
                    {
                        "analysis_scope": "normal_alts",
                        "question": question,
                        "source_timeframe": "1h",
                        "control": DIRECT_NO_LEVEL,
                        "period": period,
                        "response_window": response_window,
                        "common_support_rows": (100 if question == QUESTION_BOLLINGER else 10),
                        "common_support_coins": (8 if question == QUESTION_BOLLINGER else 3),
                    }
                )

    opened, parked = supported_question_timeframes(pd.DataFrame(rows))

    assert (QUESTION_BOLLINGER, "1h") in opened
    assert (QUESTION_SMA50, "1h") not in opened
    assert (
        parked.loc[
            parked["question"].eq(QUESTION_SMA50) & parked["source_timeframe"].eq("1h"),
            "status",
        ].item()
        == "parked_preflight_common_support"
    )


def test_exact_sma50_gate_requires_every_control_in_both_periods() -> None:
    rows = []
    for control in LOCATION_CONTROLS:
        rows.append(_exact_summary_row(control=control, period="development"))
        rows.append(_exact_summary_row(control=control, period="validation_early"))
        rows.append(_exact_summary_row(control=control, period="validation_late"))
    result = exact_residual_assessment(pd.DataFrame(rows))

    assert len(result) == 1
    assert bool(result.iloc[0]["retained_exact_location_residual"])
    assert result.iloc[0]["expected_effect_sign"] == -1

    missing = pd.DataFrame(rows).loc[
        lambda frame: (
            ~(frame["control"].eq(LOCATION_CONTROLS[-1]) & frame["period"].eq("validation_late"))
        )
    ]
    parked = exact_residual_assessment(missing)
    assert not bool(parked.iloc[0]["retained_exact_location_residual"])


def test_representation_event_rows_builds_both_surfaces_without_direction() -> None:
    frame = _match_frame().iloc[[0]].copy()
    frame["question"] = QUESTION_BOLLINGER
    frame["identity"] = "bollinger_20_upper"
    frame["representation"] = "distribution_stretch_percentile"
    frame["actual_representation_band"] = "low"
    frame["control_representation_band"] = "middle"
    frame["actual_representation_value"] = 0.10
    frame["control_representation_value"] = 0.50
    frame["analysis_scope"] = "normal_alts"
    frame["actual_event_time"] = pd.Timestamp("2025-01-01", tz="UTC")
    frame["control_event_time"] = pd.Timestamp("2025-02-01", tz="UTC")
    frame["actual_base_index"] = 10
    frame["control_base_index"] = 20
    frame["period"] = "validation_early"
    frame["response_window"] = "h4"
    for outcome, spec in OUTCOMES.items():
        frame[f"actual__{outcome}"] = 2.0
        frame[f"control__{outcome}"] = 1.0
        if spec["response_window"] == "h1":
            frame.loc[:, "response_window"] = "h4"
    for feature in CORE_STATE_FEATURES:
        frame[f"actual_state__{feature}"] = 0.1
        frame[f"control_state__{feature}"] = 0.2

    events = representation_event_rows(frame)

    assert set(events["surface"]) == {"exact_contact", "exact_line_removed"}
    assert set(events["outcome"]) == {
        outcome for outcome, spec in OUTCOMES.items() if spec["response_window"] == "h4"
    }
    assert events["direction_prediction"].eq(False).all()
    assert events["profit_optimization"].eq(False).all()


def test_monotonic_orientation_rejects_middle_band_reversal() -> None:
    assert monotonic_orientation({"low": 1.0, "middle": 2.0, "high": 3.0}) == "increasing"
    assert monotonic_orientation({"low": 3.0, "middle": 2.0, "high": 1.0}) == "decreasing"
    assert monotonic_orientation({"low": 1.0, "middle": 3.0, "high": 2.0}) == "not_monotonic"


def _match_frame() -> pd.DataFrame:
    rows = [
        {
            "pair": "ETH/USDT:USDT",
            "density_family": "bollinger_band",
            "level_name": "bollinger_20_upper",
            "source_timeframe": "1h",
            "actual_generic_scope": "isolated_generic_level",
            "actual_base_index": 0,
            "control_base_index": 1,
        },
        {
            "pair": "ETH/USDT:USDT",
            "density_family": "simple_moving_average",
            "level_name": "sma_20",
            "source_timeframe": "1h",
            "actual_generic_scope": "cross_timeframe_generic_cluster",
            "actual_base_index": 2,
            "control_base_index": 3,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["control"] = DIRECT_NO_LEVEL
    frame["actual_state__state_local_bb_position"] = 0.10
    frame["control_state__state_local_bb_position"] = 0.12
    frame["actual_state__state_local_ema50_gap_atr"] = 0.10
    frame["control_state__state_local_ema50_gap_atr"] = 0.12
    return frame


def _exact_summary_row(*, control: str, period: str) -> dict[str, object]:
    return {
        "analysis_scope": "normal_alts",
        "question": QUESTION_SMA50,
        "source_timeframe": "1h",
        "identity": "sma_50",
        "outcome": "volume_ratio_h4",
        "control": control,
        "period": period,
        "conditioning": "representation_common_support",
        "equal_coin_delta_median": -0.2,
        "positive_coins": 1,
        "negative_coins": 7,
        "support_pass": True,
    }
