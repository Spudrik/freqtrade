from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_outcome_freeze as freeze,
)


def _label(
    story_id: str,
    timestamp: str,
    partition: str,
    *,
    direction: str = "unclear",
    severity: str = "minor",
) -> dict[str, object]:
    return {
        "story_id": story_id,
        "first_seen_at": pd.Timestamp(timestamp),
        "whole_story_partition": partition,
        "direction": direction,
        "severity": severity,
        "novelty": "new",
        "confidence": 0.8,
        "article_count": 1,
        "independent_source_count": 1,
        "independent_source_group_count": 1,
        "representative_title": story_id,
    }


def test_common_cooldown_is_inclusive_and_keeps_earliest_story() -> None:
    partition = freeze.PARTITIONS[0]
    labels = pd.DataFrame.from_records(
        [
            _label("b", "2026-06-01T05:59:59Z", partition),
            _label("a", "2026-06-01T00:00:00Z", partition),
            _label("c", "2026-06-01T06:00:00Z", partition),
            _label("d", "2026-06-01T12:00:00Z", partition),
        ]
    )

    cooled = freeze.common_cooldown(labels, hours=6)

    assert cooled["story_id"].tolist() == ["a", "c", "d"]


def test_reviewed_labels_pass_schema_hash_and_exact_frozen_coverage() -> None:
    labels, parents = freeze.load_reviewed_labels()
    routes = freeze.build_routes(labels)
    coverage = freeze.route_coverage(routes)

    assert len(labels) == 120
    assert parents["reviewed_labels"]["sha256"] == (
        freeze.EXPECTED_REVIEWED_LABELS_SHA256
    )
    assert set(routes["route_id"]) == set(freeze.ROUTES)
    assert coverage["coverage_eligible"].all()
    actual = (
        routes.groupby(["route_id", "whole_event_partition"])
        .size()
        .to_dict()
    )
    expected = {
        (route, partition): count
        for route, by_partition in freeze.EXPECTED_COVERAGE.items()
        for partition, count in by_partition.items()
    }
    assert actual == expected


def test_major_and_independent_source_confluence_are_not_routes() -> None:
    labels, _parents = freeze.load_reviewed_labels()
    routes = freeze.build_routes(labels)

    assert "major_only" not in set(routes["route_id"])
    assert "independent_source_confluence" not in set(routes["route_id"])
    assert not (labels["independent_source_group_count"] >= 2).any()


def test_each_route_event_gets_twelve_prior_controls_outside_exclusion() -> None:
    labels, _parents = freeze.load_reviewed_labels()
    routes = freeze.build_routes(labels)
    controls = freeze.prior_week_controls(routes)
    blocked = [pd.Timestamp(value) for value in routes["anchor_utc"].unique()]

    assert controls.groupby("route_event_id").size().eq(12).all()
    assert controls["control_rank"].between(1, 12).all()
    assert (
        pd.to_datetime(controls["control_anchor_utc"], utc=True).dt.weekday
        == pd.to_datetime(controls["event_anchor_utc"], utc=True).dt.weekday
    ).all()
    for timestamp in pd.to_datetime(controls["control_anchor_utc"], utc=True):
        assert all(
            abs(timestamp - other) > pd.Timedelta(hours=24) for other in blocked
        )


def test_reviewed_label_hash_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "reviewed.jsonl"
    changed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "REVIEWED_LABELS", changed)

    with pytest.raises(ValueError, match="Reviewed semantic-label hash changed"):
        freeze.load_reviewed_labels()


def test_parent_manifest_artifact_hash_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot_root = tmp_path / "pilot"
    pilot_root.mkdir()
    template = pilot_root / "story_label_template.jsonl"
    template.write_text("{}\n", encoding="utf-8")
    manifest = {
        "artifacts": {
            "story_label_template.jsonl": {
                "sha256": "0" * 64,
            }
        }
    }
    monkeypatch.setattr(freeze, "PILOT_ROOT", pilot_root)

    with pytest.raises(ValueError, match="artifact hash changed"):
        freeze._verify_manifest_artifacts(manifest)


def test_freeze_document_is_terminal_and_keeps_outcomes_closed() -> None:
    partition = freeze.PARTITIONS[0]
    labels = pd.DataFrame.from_records(
        [
            _label(
                f"story-{index}",
                f"2026-06-{index + 1:02d}T00:00:00Z",
                partition,
                direction="positive",
                severity="moderate",
            )
            for index in range(3)
        ]
    )
    routes = freeze.build_routes(labels)
    controls = freeze.prior_week_controls(routes)
    cohort = {
        "cohorts": {
            "established_alts": ["SOL/USDT:USDT"],
            "top_ten_traded_memes": ["DOGE/USDT:USDT"],
        }
    }

    document = freeze.freeze_document(routes, controls, {}, cohort)

    assert document["status"] == (
        "frozen_semantic_story_questions_before_market_outcomes"
    )
    assert document["outcomes_read"] is False
    assert document["profit_used"] is False
    assert document["direction_target"]["unclear_or_mixed"] == "abstain"
    assert document["excluded_routes"]["major_only"]


def test_existing_terminal_result_must_remain_outcome_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed_semantic_story_outcome_freeze",
                "outcomes_read": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(freeze, "RESULT_PATH", result_path)

    result = freeze.execute()

    assert result["outcomes_read"] is False
