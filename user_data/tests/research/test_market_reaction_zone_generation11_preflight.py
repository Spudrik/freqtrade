# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_preflight as preflight,
)


def test_preflight_column_contract_rejects_outcomes() -> None:
    with pytest.raises(ValueError, match="Outcome columns"):
        preflight.validate_input_columns(["pair", "abs_excursion_atr_h4"])


def test_thresholds_use_only_each_cohorts_development_period(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "MIN_EVENTS", 2)
    rows = []
    for cohort, development in (("normal", "development"), ("meme", "meme_development")):
        for index, value in enumerate((1.0, 2.0, 3.0), start=1):
            row = {
                "cohort": cohort,
                "pair": f"{cohort}{index}",
                "event_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=index),
                "control": "actual",
                "period": development,
                "contact_close_distance_atr": float(index),
            }
            row.update({feature: value for feature in preflight.THRESHOLD_FEATURES})
            rows.append(row)
        validation = dict(rows[-1])
        validation["pair"] = f"{cohort}_future"
        validation["event_time"] += pd.Timedelta(days=10)
        validation["period"] = "validation_early"
        validation.update({feature: 1000.0 for feature in preflight.THRESHOLD_FEATURES})
        rows.append(validation)

    result = preflight.build_thresholds(pd.DataFrame.from_records(rows))

    assert result["upper_tercile"].max() < 4.0
    assert result["status"].eq("frozen_from_development_predictors").all()


def test_activity_state_uses_frozen_cohort_thresholds() -> None:
    thresholds = pd.DataFrame(
        {
            "cohort": ["normal", "meme"],
            "feature": ["state__relative_volume", "state__relative_volume"],
            "lower_tercile": [1.0, 10.0],
            "upper_tercile": [2.0, 20.0],
            "status": [
                "frozen_from_development_predictors",
                "frozen_from_development_predictors",
            ],
        }
    )
    frame = pd.DataFrame(
        {
            "cohort": ["normal", "normal", "normal", "meme"],
            "state__relative_volume": [0.5, 1.5, 2.5, 15.0],
        }
    )

    result = preflight.assign_activity_state(frame, thresholds)

    assert result.tolist() == ["quiet", "typical", "active", "typical"]
