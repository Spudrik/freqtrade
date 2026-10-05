from __future__ import annotations

# ruff: noqa: S101
import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_active_guardrail_audit_2026 as audit,
)


def _row(
    *,
    date: str,
    sample_id: str,
    sample_kind: str,
    episode: str,
    period: str = "p1",
    pair: str = "BTC",
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date, tz="UTC"),
        "sample_id": sample_id,
        "sample_kind": sample_kind,
        "parent_episode_ids_json": json.dumps([episode]),
        "period": period,
        "pair": pair,
        "value": sample_id,
    }


def test_collapse_actual_keeps_earliest_episode_anchor_and_all_controls() -> None:
    frame = pd.DataFrame(
        [
            _row(
                date="2025-01-01 02:00",
                sample_id="later",
                sample_kind="actual_event",
                episode="e1",
            ),
            _row(
                date="2025-01-01 01:00",
                sample_id="earlier",
                sample_kind="actual_event",
                episode="e1",
            ),
            _row(
                date="2025-01-08 01:00",
                sample_id="control",
                sample_kind="matched_control",
                episode="e1",
            ),
        ]
    )

    corrected, summary = audit.collapse_actual_to_one_anchor(
        frame, period_column="period"
    )

    actual = corrected.loc[corrected["sample_kind"].eq("actual_event")]
    assert actual["sample_id"].tolist() == ["earlier"]
    assert corrected["sample_kind"].eq("matched_control").sum() == 1
    assert summary["removed_repeated_actual_anchors"] == 1


def test_collapse_actual_preserves_profiles_as_separate_identities() -> None:
    frame = pd.DataFrame(
        [
            {
                **_row(
                    date="2025-01-01 01:00",
                    sample_id="a",
                    sample_kind="actual_event",
                    episode="e1",
                ),
                "profile_id": "p1",
            },
            {
                **_row(
                    date="2025-01-01 02:00",
                    sample_id="b",
                    sample_kind="actual_event",
                    episode="e1",
                ),
                "profile_id": "p1",
            },
            {
                **_row(
                    date="2025-01-01 01:00",
                    sample_id="a",
                    sample_kind="actual_event",
                    episode="e1",
                ),
                "profile_id": "p2",
            },
        ]
    )

    corrected, _ = audit.collapse_actual_to_one_anchor(
        frame,
        period_column="period",
        identity_columns=("profile_id", "pair"),
    )

    assert len(corrected) == 2
    assert set(corrected["profile_id"]) == {"p1", "p2"}


def test_decision_change_summary_labels_lost_and_new_cells() -> None:
    old = pd.DataFrame({"key": ["a", "b"], "passed": [True, False]})
    new = pd.DataFrame({"key": ["a", "b"], "passed": [False, True]})

    result = audit._decision_change_summary(
        old, new, keys=("key",), decision_column="passed"
    )

    assert result["lost_after_correction"] == 1
    assert result["new_after_correction"] == 1
    assert {row["change"] for row in result["changed_cells"]} == {
        "lost_after_correction",
        "new_after_correction",
    }
