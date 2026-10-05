# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)


def test_add_metric_columns_makes_pressure_reactions_directionless() -> None:
    frame = pd.DataFrame(
        {
            metric: [1.0]
            for metric in g6d.RAW_METRIC_COLUMNS
            if metric
            not in {
                "contact_pressure_change",
                "pressure_change_h1",
                "pressure_change_h4",
            }
        }
    )
    frame["contact_pressure_change"] = -2.0
    frame["pressure_change_h1"] = -3.0
    frame["pressure_change_h4"] = 4.0

    result = g6d.add_metric_columns(frame)

    assert result.loc[0, "absolute_contact_pressure_change"] == 2.0
    assert result.loc[0, "absolute_pressure_change_h1"] == 3.0
    assert result.loc[0, "absolute_pressure_change_h4"] == 4.0


def test_control_candidate_requires_both_periods_and_all_core_controls() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        for control in g6d.CORE_LOCATION_CONTROLS:
            rows.append(
                {
                    "cohort": "normal",
                    "level_family": "family",
                    "source_timeframe": "1h",
                    "metric": "volume_ratio_h1",
                    "period": period,
                    "comparison_control": control,
                    "median_actual": 2.0,
                    "median_control": 1.0,
                    "median_difference": 1.0,
                    "effect_sign": 1,
                    "rows_actual": 60,
                    "rows_control": 60,
                    "coins_actual": 5,
                    "coins_control": 5,
                }
            )
    effects = pd.DataFrame.from_records(rows)
    result = g6d.repeated_control_candidates(
        effects,
        cell_keys=("cohort", "level_family", "source_timeframe", "metric"),
        required_controls=g6d.CORE_LOCATION_CONTROLS,
    )

    assert bool(result.iloc[0]["candidate"]) is True

    missing = effects.loc[
        ~(
            effects["period"].eq("validation_late")
            & effects["comparison_control"].eq("stale_72h")
        )
    ]
    rejected = g6d.repeated_control_candidates(
        missing,
        cell_keys=("cohort", "level_family", "source_timeframe", "metric"),
        required_controls=g6d.CORE_LOCATION_CONTROLS,
    )

    assert bool(rejected.iloc[0]["candidate"]) is False


def test_scoped_interaction_can_treat_btc_as_a_one_coin_market() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        rows.append(
            {
                "lane": "orderbook_state",
                "scope": "btc_pair_local",
                "cohort": "normal",
                "feature": "btc_absolute_pressure",
                "level_family": "generic_prior_range",
                "metric": "volume_ratio_h1",
                "period": period,
                "effect_sign": 1,
                "low_actual_rows": 60,
                "low_control_rows": 60,
                "high_actual_rows": 60,
                "high_control_rows": 60,
                "low_actual_coins": 1,
                "low_control_coins": 1,
                "high_actual_coins": 1,
                "high_control_coins": 1,
            }
        )
    interactions = pd.DataFrame.from_records(rows)

    btc_result = g6d.repeated_interaction_candidates(
        interactions,
        min_coins=1,
    )
    broad_market_result = g6d.repeated_interaction_candidates(interactions)

    assert bool(btc_result.iloc[0]["candidate"]) is True
    assert bool(broad_market_result.iloc[0]["candidate"]) is False


def test_market_group_screen_obeys_preflight_and_keeps_btc_separate() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        for control, value in (("actual", 2.0), ("matched_random_time", 1.0)):
            for event_number in range(50):
                row = {
                    "cohort": "normal",
                    "pair": "BTC/USDT:USDT",
                    "source_timeframe": "1h",
                    "level_family": "volume_profile_settled",
                    "level_name": "vah",
                    "control": control,
                    "period": period,
                    "market_group": "btc_separate",
                    "smart_contract_platform": False,
                    "event_number": event_number,
                }
                row.update({metric: value for metric in g6d.METRICS})
                row.update({column: False for column in g6d.RELATIONSHIP_COLUMNS})
                rows.append(row)
    support = pd.DataFrame(
        {
            "group_id": ["btc_separate", "payments_and_transfer"],
            "question": [
                "volume_profile_value_area_reaction",
                "volume_profile_value_area_reaction",
            ],
            "supported": [True, False],
        }
    )

    effects, candidates = g6d.market_group_screen(
        pd.DataFrame.from_records(rows),
        support,
    )

    assert set(effects["group_id"]) == {"btc_separate"}
    assert set(candidates["group_id"]) == {"btc_separate"}
    assert candidates["candidate"].all()
