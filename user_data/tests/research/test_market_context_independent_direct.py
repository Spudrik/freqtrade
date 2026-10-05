from __future__ import annotations

# ruff: noqa: S101
import json

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_context_independent_direct as direct,
)


def _summary_row(
    *,
    partition: str,
    events: int = 10,
    activity_rate: float = 0.60,
    activity_multiple: float = 1.30,
    direction_calls: int = 10,
    direction_rate: float = 0.70,
    trend_rate: float = 0.65,
    majority_rate: float = 0.64,
    rotated_rate: float = 0.63,
    joint_rate: float = 0.60,
) -> dict[str, object]:
    return {
        "source_batch": "synthetic",
        "family": "synthetic_family",
        "family_label": "Synthetic family",
        "source_role": "synthetic_role",
        "market_scope": "btc",
        "horizon_hours": 1,
        "whole_event_partition": partition,
        "whole_events": events,
        "median_activity_multiple": activity_multiple,
        "reaction_success_rate": activity_rate,
        "direction_calls": direction_calls,
        "direction_success_rate": direction_rate,
        "recent_trend_success_rate": trend_rate,
        "majority_direction_success_rate": majority_rate,
        "rotated_direction_success_rate": rotated_rate,
        "joint_success_rate": joint_rate,
    }


def test_load_frozen_inputs_parses_mixed_iso_precision_and_filters_string_flags(
    tmp_path, monkeypatch
) -> None:
    freeze_path = tmp_path / "freeze.json"
    events_path = tmp_path / "events.csv"
    controls_path = tmp_path / "controls.csv"
    observations_path = tmp_path / "observations.parquet"
    result_path = tmp_path / "result.json"

    freeze_payload = {
        "status": "frozen_independent_context_sources_before_market_outcomes",
        "outcomes_read": False,
    }
    freeze_path.write_text(json.dumps(freeze_payload), encoding="utf-8")
    events_path.write_text(
        "event_id,family,anchor_utc,available_at,coverage_ready\n"
        "kept,synthetic,2026-06-01T00:00:00Z,2026-06-01T00:00:00.123456Z,TRUE\n"
        "discarded,synthetic,2026-06-02T00:00:00.123456Z,2026-06-02T00:00:00Z,false\n"
        "also_discarded,synthetic,2026-06-03T00:00:00Z,2026-06-03T00:00:00Z,1\n",
        encoding="utf-8",
    )
    controls_path.write_text(
        "event_id,event_anchor_utc,control_anchor_utc\n"
        "kept,2026-06-01T00:00:00Z,2026-05-31T00:00:00.123Z\n",
        encoding="utf-8",
    )
    observations_path.write_bytes(b"synthetic observations")
    artifacts = {
        name: {"sha256": direct.g0.sha256_file(path)}
        for name, path in {
            "freeze": freeze_path,
            "events": events_path,
            "controls": controls_path,
            "observations": observations_path,
        }.items()
    }
    result_path.write_text(
        json.dumps(
            {
                "status": "completed_independent_context_source_freeze",
                "outcomes_read": False,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(direct, "FREEZE_RESULT_PATH", result_path)
    monkeypatch.setattr(direct, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(direct, "EVENTS_PATH", events_path)
    monkeypatch.setattr(direct, "CONTROLS_PATH", controls_path)
    monkeypatch.setattr(direct, "OBSERVATIONS_PATH", observations_path)

    loaded_freeze, events, controls = direct.load_frozen_inputs()

    assert loaded_freeze == freeze_payload
    assert events["event_id"].tolist() == ["kept"]
    assert str(events.loc[0, "anchor_utc"]) == "2026-06-01 00:00:00+00:00"
    assert str(events.loc[0, "available_at"]) == "2026-06-01 00:00:00.123456+00:00"
    assert str(controls.loc[0, "control_anchor_utc"]) == (
        "2026-05-31 00:00:00.123000+00:00"
    )


def test_comparators_ignore_missing_values_and_return_nan_when_empty() -> None:
    assert direct.comparator_max(
        pd.Series(
            {
                "recent_trend_success_rate": np.nan,
                "majority_direction_success_rate": 0.62,
                "rotated_direction_success_rate": None,
            }
        )
    ) == pytest.approx(0.62)
    assert direct.maximum_finite(np.nan, None, pd.NA, 0.41) == pytest.approx(0.41)
    assert np.isnan(
        direct.comparator_max(
            pd.Series(
                {
                    "recent_trend_success_rate": np.nan,
                    "majority_direction_success_rate": None,
                    "rotated_direction_success_rate": pd.NA,
                }
            )
        )
    )
    assert np.isnan(direct.maximum_finite(np.nan, None, pd.NA))


def test_classify_cell_requires_both_partitions_for_activity_and_direction() -> None:
    development = pd.Series(_summary_row(partition="development_test"))
    validation = pd.Series(
        _summary_row(
            partition="validation_test",
            direction_rate=0.68,
            trend_rate=0.66,
            majority_rate=0.64,
            rotated_rate=0.65,
        )
    )

    result = direct.classify_cell(development, validation)

    assert result["supported"] is True
    assert result["activity_pass"] is True
    assert result["direction_pass"] is False
    assert result["joint_pass"] is False


def test_classify_cell_passes_joint_only_when_direction_beats_every_control() -> None:
    development = pd.Series(
        _summary_row(
            partition="development_test",
            direction_rate=0.71,
            trend_rate=0.66,
            majority_rate=0.67,
            rotated_rate=0.65,
        )
    )
    validation = pd.Series(
        _summary_row(
            partition="validation_test",
            direction_rate=0.72,
            trend_rate=0.67,
            majority_rate=0.66,
            rotated_rate=0.65,
        )
    )

    result = direct.classify_cell(development, validation)

    assert result == {
        "supported": True,
        "activity_pass": True,
        "direction_pass": True,
        "joint_pass": True,
        "validation_direction": pytest.approx(0.72),
        "validation_joint": pytest.approx(0.60),
    }


def test_classify_cell_rejects_insufficient_event_count() -> None:
    development = pd.Series(_summary_row(partition="development_test", events=9))
    validation = pd.Series(_summary_row(partition="validation_test"))

    assert direct.classify_cell(development, validation) == {"supported": False}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "joint_cells": ["btc:1h"],
                "activity_cells": ["btc:1h"],
                "direction_cells": ["btc:1h"],
                "supported_cells": 1,
                "best_validation_direction": 0.80,
            },
            "repeatable_joint_activity_and_direction_lead",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": ["btc:1h"],
                "direction_cells": ["btc:4h"],
                "supported_cells": 2,
                "best_validation_direction": 0.80,
            },
            "repeatable_activity_and_separate_direction_lead",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": ["btc:1h"],
                "direction_cells": [],
                "supported_cells": 1,
                "best_validation_direction": np.nan,
            },
            "repeatable_activity_only_lead",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": [],
                "direction_cells": ["btc:1h"],
                "supported_cells": 1,
                "best_validation_direction": 0.80,
            },
            "repeatable_direction_only_lead",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": [],
                "direction_cells": [],
                "supported_cells": 0,
                "best_validation_direction": np.nan,
            },
            "insufficient_two_partition_coverage",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": [],
                "direction_cells": [],
                "supported_cells": 1,
                "best_validation_direction": 0.55,
            },
            "one_period_or_control_limited_direction_watchlist",
        ),
        (
            {
                "joint_cells": [],
                "activity_cells": [],
                "direction_cells": [],
                "supported_cells": 1,
                "best_validation_direction": 0.54,
            },
            "weak_or_inconsistent",
        ),
    ],
)
def test_family_verdict_precedence(kwargs: dict[str, object], expected: str) -> None:
    assert direct.family_verdict(**kwargs) == expected


def test_partition_pair_selects_first_development_and_validation_rows() -> None:
    rows = DataFrame(
        {
            "whole_event_partition": [
                "other",
                "development_test",
                "development_later",
                "validation_test",
                "internal_validation_test",
            ],
            "event_id": ["other", "dev_first", "dev_later", "val_first", "val_later"],
        }
    )

    pair = direct.partition_pair(rows)

    assert pair is not None
    development, validation = pair
    assert development["event_id"].tolist() == ["dev_first"]
    assert validation["event_id"].tolist() == ["val_first"]


def test_partition_pair_returns_none_without_both_partition_types() -> None:
    rows = DataFrame(
        {
            "whole_event_partition": ["development_test", "other"],
            "event_id": ["dev", "other"],
        }
    )

    assert direct.partition_pair(rows) is None
