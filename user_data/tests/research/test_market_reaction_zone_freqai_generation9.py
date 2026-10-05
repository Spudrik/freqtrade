from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation9 as g9f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_freeze as g9z,
)


def _score_rows(*, strict: bool, point: bool) -> pd.DataFrame:
    question = g9z.MODEL_QUESTIONS[0]
    rows = []
    for seed in g9z.SEEDS:
        for control_number in range(9):
            for period in ("validation_early", "validation_late"):
                rows.append(
                    {
                        "question_id": question.question_id,
                        "branch_id": "g9a",
                        "surface": "limited_three_source",
                        "route_id": "complete_three_source_model",
                        "route_type": "limited_three_source_attribution",
                        "target": question.target,
                        "group_id": "full_normal_cohort",
                        "group_members": "A,B,C,D,E",
                        "seed": seed,
                        "period": period,
                        "comparison_id": f"c{seed}_{control_number}",
                        "expected_controls_for_route": 9,
                        "strict_period_pass": strict,
                        "provisional_period_pass": point,
                        "equal_coin_paired_mae_gain": 0.1,
                        "bootstrap_lower": 0.01 if strict else -0.01,
                        "plain_name": question.plain_name,
                        "mechanism": question.mechanism,
                        "result_group": "all_declared_groups",
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_joint_seed_decision_requires_every_control_period_and_seed() -> None:
    manifest = {"validation_periods": ["validation_early", "validation_late"]}
    seed_rows = g9f.route_decisions(manifest, _score_rows(strict=True, point=True))
    joint = g9f.joint_seed_decisions(seed_rows)
    assert len(seed_rows) == 3
    assert joint.iloc[0]["status"] == "strict_three_seed_lead"
    assert joint.iloc[0]["strict_seed_count"] == 3


def test_joint_seed_decision_fails_if_one_seed_fails() -> None:
    scores = _score_rows(strict=True, point=True)
    scores.loc[scores["seed"].eq(73), "strict_period_pass"] = False
    scores.loc[scores["seed"].eq(73), "provisional_period_pass"] = False
    manifest = {"validation_periods": ["validation_early", "validation_late"]}
    seed_rows = g9f.route_decisions(manifest, scores)
    joint = g9f.joint_seed_decisions(seed_rows)
    assert joint.iloc[0]["status"] == "failed_three_seed_replication"
    assert joint.iloc[0]["strict_seed_count"] == 2
