from __future__ import annotations

# ruff: noqa: S101
import json

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation4 as g4d,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    TARGET_COLUMNS,
    MarketReactionZoneG4DUnconditionalFreqAIResearchStrategy,
)


def test_g4d_deterministic_shift_never_returns_self_shift() -> None:
    first = g4d.deterministic_shift(7, "surface|pair|period")
    second = g4d.deterministic_shift(7, "surface|pair|period")

    assert first == second
    assert 1 <= first < 7


def test_g4d_independence_requires_more_than_target_horizon() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T04:00:00Z",
                    "2026-01-01T05:00:00Z",
                    "2026-01-01T10:00:00Z",
                ]
            ),
            "event_state": ["a", "b", "c", "d"],
            "level_identity": ["x"] * 4,
        }
    )

    result = g4d.independently_spaced_events(frame, hours=4)

    assert result["date"].tolist() == [
        frame.loc[0, "date"],
        frame.loc[2, "date"],
        frame.loc[3, "date"],
    ]
    assert result["date"].diff().dropna().dt.total_seconds().min() / 3600.0 > 4.0


def test_g4d_shuffled_control_preserves_values_without_self_assignment() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="6h", tz="UTC"),
            "period": ["validation_early"] * 5,
            "level__value": np.arange(5, dtype=float),
            "geometry__count": np.arange(10, 15, dtype=float),
        }
    )

    result, audit = g4d.attach_shuffled_control(
        frame,
        pair="BTC/USDT:USDT",
        surface_id="surface",
    )

    assert sorted(result["placebo__level_value"]) == sorted(result["level__value"])
    assert sorted(result["placebo__geometry_count"]) == sorted(result["geometry__count"])
    assert audit[0]["self_assignments"] == 0
    assert not result["placebo__level_value"].eq(result["level__value"]).all()


def test_g4d_missing_feature_row_is_excluded_not_zero_filled() -> None:
    dates = pd.date_range("2025-01-01", periods=3, freq="6h", tz="UTC")
    events = pd.DataFrame(
        {
            "pair": ["BTC/USDT:USDT"] * 3,
            "date": dates,
            "period": ["validation_early"] * 3,
            "event_state": ["state"] * 3,
            "level_identity": ["level"] * 3,
            "source_available_at": dates,
            "level__value": [1.0, np.nan, 3.0],
            "geometry__count": [1.0, 2.0, 3.0],
            "mtf__count": [1.0, 2.0, 3.0],
            "interaction__value": [1.0, 2.0, 3.0],
            **{target: [1.0, 2.0, 3.0] for target in TARGET_COLUMNS},
        }
    )
    events.attrs["surface_id"] = "surface"
    wide = pd.DataFrame(
        {
            "date": dates,
            "wide__state": [1.0, 2.0, 3.0],
        }
    )
    manifest = {
        "data": {
            "chronological_periods": [
                {
                    "id": "validation_early",
                    "start_utc": "2025-01-01T00:00:00Z",
                    "end_utc_exclusive": "2025-01-02T00:00:00Z",
                    "role": "chronological_internal_validation",
                }
            ]
        }
    }

    result, audit = g4d.apply_event_eligibility(
        events,
        manifest=manifest,
        wide=wide,
        pair="BTC/USDT:USDT",
    )

    assert len(result) == 2
    assert audit["incomplete_rows_dropped"] == 1
    assert 0.0 not in result["level__value"].tolist()


def test_g4d_band_thresholds_use_development_only() -> None:
    events = pd.DataFrame(
        {
            "period": ["development"] * 30 + ["validation_early"] * 30,
            TARGET_COLUMNS[0]: list(range(30)) + [10_000.0] * 30,
            TARGET_COLUMNS[1]: list(range(30)) + [20_000.0] * 30,
            TARGET_COLUMNS[2]: list(range(30)) + [30_000.0] * 30,
        }
    )

    thresholds, audit = g4d.target_band_thresholds(
        {"BTC/USDT:USDT": events},
        development="development",
    )

    lower, upper = thresholds[("BTC/USDT:USDT", TARGET_COLUMNS[0])]
    assert lower < 20.0
    assert upper < 30.0
    assert set(audit["status"]) == {"supported"}


def test_g4d_regression_metrics_reward_lower_error_and_ranking() -> None:
    actual = np.linspace(0.0, 1.0, 30)
    frame = pd.DataFrame(
        {
            "prediction": actual + 0.01,
            "actual": actual,
            "prediction_band": np.repeat([0.0, 1.0, 2.0], 10),
            "actual_band": np.repeat([0.0, 1.0, 2.0], 10),
        }
    )

    result = g4d.regression_metrics(frame)

    assert result["status"] == "scored"
    assert result["mean_absolute_error"] < 0.02
    assert result["prediction_actual_spearman"] > 0.99
    assert result["exact_band_accuracy"] == 1.0


def test_g4d_retention_gate_requires_both_periods_and_five_coins() -> None:
    periods = ("validation_early", "validation_late")
    rows: list[dict[str, object]] = []
    for period in periods:
        for comparison_id, mae, rank in (
            ("combined_current_vs_market_state", 0.1, 0.02),
            ("combined_current_vs_shuffled", 0.05, 0.01),
        ):
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "scope": "normal_ex_btc",
                    "scope_type": "cohort",
                    "period": period,
                    "target": TARGET_COLUMNS[0],
                    "status_profile": "scored",
                    "status_control": "scored",
                    "mean_absolute_error_improvement": mae,
                    "prediction_actual_spearman_improvement": rank,
                }
            )
        for index in range(6):
            rows.append(
                {
                    "comparison_id": "combined_current_vs_market_state",
                    "scope": f"pair::COIN{index}/USDT:USDT",
                    "scope_type": "pair",
                    "period": period,
                    "target": TARGET_COLUMNS[0],
                    "status_profile": "scored",
                    "status_control": "scored",
                    "mean_absolute_error_improvement": 0.01,
                    "prediction_actual_spearman_improvement": 0.01,
                }
            )
    manifest = {
        "surface_id": "surface",
        "cohort": "normal",
        "validation_periods": list(periods),
        "technical_smoke_not_evidence": False,
        "profiles": list(g4d.PROFILES),
    }

    decisions = g4d.retention_decisions(manifest, pd.DataFrame(rows))

    selected = decisions.loc[decisions["target"].eq(TARGET_COLUMNS[0])].iloc[0]
    assert selected["status"] == "retained_research_lead"
    checks = json.loads(selected["period_checks"])
    assert all(item["positive_coin_count"] == 6 for item in checks)


def test_g4d_training_labels_embargo_last_four_hours(monkeypatch) -> None:
    dates = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
    events = pd.DataFrame({"date": dates})
    for target in TARGET_COLUMNS:
        events[target] = 1.0
    strategy = object.__new__(MarketReactionZoneG4DUnconditionalFreqAIResearchStrategy)
    monkeypatch.setattr(strategy, "_event_cache", lambda pair: events)
    frame = pd.DataFrame({"date": dates.astype("datetime64[ms, UTC]")})

    labelled = strategy.set_freqai_targets(frame, metadata={"pair": "BTC/USDT:USDT"})

    cutoff = dates.max() - pd.Timedelta(hours=4)
    assert labelled.loc[dates <= cutoff, list(TARGET_COLUMNS)].notna().all().all()
    assert labelled.loc[dates > cutoff, list(TARGET_COLUMNS)].isna().all().all()


def test_g4d_time_only_baseline_has_variance_without_market_inputs() -> None:
    strategy = object.__new__(MarketReactionZoneG4DUnconditionalFreqAIResearchStrategy)
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")}
    )

    result = strategy.feature_engineering_expand_basic(frame, metadata={})

    assert result["%-g4d_causal_calendar_time_years"].nunique() == 4
    assert not any(
        token in column
        for column in result
        for token in ("return", "volume", "level", "pressure")
    )
