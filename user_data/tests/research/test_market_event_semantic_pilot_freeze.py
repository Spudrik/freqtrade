from __future__ import annotations

# ruff: noqa: S101
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_pilot_freeze as pilot,
)


def test_near_duplicate_title_rule_is_conservative() -> None:
    first = "Federal Reserve holds rates steady and signals patience on inflation"
    close = "Federal Reserve holds rates steady, signals patience on inflation"
    different = "Bitcoin exchange launches a new staking product for customers"

    assert pilot.near_duplicate_titles(first, close)
    assert not pilot.near_duplicate_titles(first, different)


def test_exact_title_episodes_split_repeated_titles_after_48_hours() -> None:
    rows = []
    for number, stamp in enumerate(
        (
            "2026-06-01T00:00:00Z",
            "2026-06-01T12:00:00Z",
            "2026-06-05T00:00:00Z",
        )
    ):
        rows.append(
            {
                "article_key": f"news:{number}",
                "source_id": f"source-{number}",
                "source_family": "news",
                "source_group": "macro_economics",
                "title": "Repeated release title",
                "summary": "",
                "collected_ts": pd.Timestamp(stamp),
                "whole_story_partition": pilot.BLOCKS[0][0],
                "eligible_for_pilot": True,
                "priority_band": "medium",
                "final_priority_score": 50.0,
                "exact_title_key": pilot._duplicate_key("Repeated release title"),
            }
        )
    eligible, exact = pilot.assign_exact_episodes(pd.DataFrame(rows))

    assert len(eligible) == 3
    assert len(exact) == 2
    assert sorted(exact["article_count"].tolist()) == [1, 2]


def test_balanced_selection_is_deterministic_and_covers_strata() -> None:
    frame = pd.DataFrame(
        [
            {
                "story_id": f"story-{index}",
                "family": "news" if index % 2 else "web",
                "group": f"group-{index % 3}",
            }
            for index in range(30)
        ]
    )
    first = pilot.balanced_select(
        frame,
        12,
        id_column="story_id",
        stratum_columns=("family", "group"),
        seed="fixed",
    )
    second = pilot.balanced_select(
        frame,
        12,
        id_column="story_id",
        stratum_columns=("family", "group"),
        seed="fixed",
    )

    assert first["story_id"].tolist() == second["story_id"].tolist()
    assert first.groupby(["family", "group"]).size().eq(2).all()


def test_completed_label_schema_requires_traceable_evidence() -> None:
    pending = {
        "story_id": "story-a",
        "article_keys": ["news:a"],
        "label_status": "complete",
        "topics": ["central_bank_policy"],
        "impact_channels": ["interest_rates"],
        "direction": "mixed",
        "severity": "moderate",
        "confidence": 0.8,
        "evidence_article_keys": [],
        "classifier_model": "test-model",
        "classifier_version": "v1",
        "prompt_version": "p1",
    }
    report = pilot.validate_completed_labels([pending], {"story-a"})

    assert not report["semantic_schema_gate_passed"]
    assert "evidence_article_keys_required" in report["errors"][0]["errors"]

    pending["evidence_article_keys"] = ["news:a"]
    report = pilot.validate_completed_labels([pending], {"story-a"})
    assert report["semantic_schema_gate_passed"]


def test_partition_targets_preserve_requested_total() -> None:
    assert sum(pilot.partition_targets(121).values()) == 121
    assert pilot.partition_targets(121)[pilot.BLOCKS[0][0]] == 60
    assert pilot.partition_targets(121)[pilot.BLOCKS[1][0]] == 61
