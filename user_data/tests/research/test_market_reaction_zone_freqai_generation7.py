# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)


def test_every_branch_has_complete_eight_profile_common_row_ladder() -> None:
    for branch_id, spec in g7f.BRANCH_RUNTIME.items():
        profiles = {
            definition["role"]: definition
            for definition in g7f.PROFILES.values()
            if definition["branch_id"] == branch_id
        }

        assert tuple(profiles) == g7f.PROFILE_ROLES
        assert len(profiles) == 8
        assert {
            tuple(definition["required_ready_blocks"])
            for definition in profiles.values()
        } == {(spec.ready_block,)}
        assert profiles[g7f.COMPLETE_ROLE]["blocks"] == (
            *spec.level_blocks,
            *spec.component_a_blocks,
            *spec.component_b_blocks,
        )


def test_every_complete_model_has_seven_frozen_challenges() -> None:
    for branch_id in g7f.BRANCH_RUNTIME:
        comparisons = [
            item for item in g7f.COMPARISONS if item["branch_id"] == branch_id
        ]

        assert len(comparisons) == 7
        assert {item["baseline_role"] for item in comparisons} == set(
            g7f.PROFILE_ROLES
        ).difference({g7f.COMPLETE_ROLE})


def test_sparse_stale_placebo_skips_intervening_missing_rows() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-04T00:00:00Z",
            "2026-01-05T00:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "period": ["a"] * 4,
            "special__value": [1.0, np.nan, 4.0, 5.0],
        }
    )

    result = g7f.attach_sparse_stale_placebo(frame, block="special")

    assert np.isnan(result.loc[0, "special_placebo__value"])
    assert np.isnan(result.loc[1, "special_placebo__value"])
    assert result.loc[2, "special_placebo__value"] == 1.0
    assert result.loc[3, "special_placebo__value"] == 1.0
    valid = result["source_date"].notna()
    assert (
        result.loc[valid, "source_date"]
        <= result.loc[valid, "date"] - pd.Timedelta(hours=72)
    ).all()


def test_independent_cluster_uses_closest_dependency_independent_pair() -> None:
    date = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = pd.DataFrame(
        {
            "date": [date, date, date],
            "level_family": [
                "generic_prior_range",
                "volume_profile_explicit_prior",
                "generic_round_number",
            ],
            "source_timeframe": ["1h", "4h", "1h"],
            "level_price": [100.00, 100.05, 102.00],
            "base_atr": [1.0, 1.0, 1.0],
            "zone_half_width_atr": [0.02, 0.02, 0.02],
            "pre_distance_atr": [0.1, 0.2, 0.3],
            "contact_close_distance_atr": [0.01, 0.02, 0.03],
        }
    )

    first, second = g7f.aggregate_independent_cluster(rows)

    assert len(first) == len(second) == 1
    assert first.loc[0, f"{g7f.SPECIAL_CLUSTER_A}__group_prior_range"] == 1.0
    assert second.loc[0, f"{g7f.SPECIAL_CLUSTER_B}__group_volume_profile"] == 1.0
    assert np.isclose(
        second.loc[0, f"{g7f.SPECIAL_CLUSTER_B}__zone_gap_atr"], 0.01
    )


def test_group_decision_requires_all_seven_controls_in_both_periods() -> None:
    branch_id = "g7a_local_participation_and_btc_activity"
    target = g7f.FROZEN_BRANCHES[branch_id]["targets"][0]
    records = []
    for comparison in range(7):
        for period in ("validation_early", "validation_late"):
            records.append(
                {
                    "branch_id": branch_id,
                    "target": target,
                    "group_id": "established_altcoins",
                    "group_members": "A,B,C,D,E",
                    "comparison_id": f"comparison_{comparison}",
                    "baseline_role": f"baseline_{comparison}",
                    "period": period,
                    "rows": 100,
                    "positive_coins": 5,
                    "equal_coin_paired_mae_gain": 0.1,
                    "bootstrap_lower": 0.01,
                    "not_dominated_by_one_coin": True,
                    "strict_period_pass": True,
                    "provisional_period_pass": True,
                }
            )
    manifest = {"validation_periods": ["validation_early", "validation_late"]}

    retained = g7f.branch_target_group_decisions(
        manifest, pd.DataFrame.from_records(records)
    )
    records[-1]["strict_period_pass"] = False
    rejected = g7f.branch_target_group_decisions(
        manifest, pd.DataFrame.from_records(records)
    )

    assert retained.loc[0, "status"] == "control_resistant_complete_interaction"
    assert rejected.loc[0, "status"] == "provisional_complete_interaction"
