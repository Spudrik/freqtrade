from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_freeze as g9,
)


def test_model_portfolio_is_breadth_first_and_bounded() -> None:
    assert len(g9.MODEL_QUESTIONS) == 7
    assert sum(question.cohort == "normal" for question in g9.MODEL_QUESTIONS) == 5
    assert sum(question.cohort == "meme" for question in g9.MODEL_QUESTIONS) == 2
    assert all(len(question.blocks) == 3 for question in g9.MODEL_QUESTIONS)
    assert {question.target for question in g9.MODEL_QUESTIONS} == {
        g9.VOLUME_TARGET,
        g9.RANGE_TARGET,
    }


def test_registry_has_all_three_seeds_and_nine_controls() -> None:
    profiles, comparisons = g9.build_model_registry()
    assert len(profiles) == 7 * 3 * 10
    assert len(comparisons) == 7 * 3 * 9
    assert {profile["seed"] for profile in profiles.values()} == set(g9.SEEDS)
    assert max(len(profile["blocks"]) for profile in profiles.values()) == 3
    grouped: dict[tuple[str, int], list[dict]] = {}
    for comparison in comparisons:
        grouped.setdefault(
            (comparison["question_id"], comparison["seed"]), []
        ).append(comparison)
    assert all(len(items) == 9 for items in grouped.values())
    for items in grouped.values():
        types = [item["control_type"] for item in items]
        assert types.count("immediately_simpler_leave_one_input_out") == 3
        assert types.count("single_input_causal_stale") == 3
        assert types.count("single_input_within_period_nonself_shuffle") == 3


def test_one_minute_selection_columns_exclude_signed_future_direction() -> None:
    opened = {
        *g9.EPISODE_FEATURE_COLUMNS,
        *g9.RAW_ANCHOR_COLUMNS,
        g9.VOLUME_TARGET,
        "period",
    }
    assert not any("direction" in column.lower() for column in opened)
    assert not any("return" in column.lower() for column in opened)
    assert g9.VOLUME_TARGET in opened


def test_diagnostic_sample_has_every_frozen_stratum_and_distinct_pairs() -> None:
    rows = []
    for cohort in ("normal", "meme"):
        for index, (period, timeframe) in enumerate(
            (period, timeframe)
            for period in g9.PERIODS[cohort]
            for timeframe in g9.ONE_MINUTE_TIMEFRAMES
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "period": period,
                    "source_timeframe": timeframe,
                    "pair": f"{cohort}_{index}/USDT:USDT",
                    "episode_id": f"{cohort}_{index}",
                    "selection_hash": f"{index:02d}",
                }
            )
    sample = g9.select_diagnostic_sample(pd.DataFrame.from_records(rows))
    assert len(sample) == 12
    assert sample["sample_stratum"].nunique() == 12
    assert all(
        group["pair"].nunique() == 6
        for _, group in sample.groupby("cohort", observed=True)
    )
