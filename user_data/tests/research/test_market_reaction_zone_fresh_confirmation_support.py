# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as freshz,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_support as support,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_anchored_vwap as g24v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24cache,
)


def synthetic_base(start: str, periods: int) -> DataFrame:
    dates = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 98.0,
            "low": 97.0,
            "close": 98.0,
            "volume": np.linspace(10.0, 20.0, periods),
            "base_atr": 4.0,
            "pre_close": 98.0,
        }
    )


def test_fresh_contact_events_use_frozen_periods_and_causal_sources() -> None:
    base = synthetic_base("2026-08-19T20:00:00Z", 30)
    base.loc[[5, 6, 15], "high"] = 101.0
    base.loc[[5, 6, 15], "low"] = 99.5
    level = np.full(len(base), 100.0)
    source = pd.to_datetime(base["date"], utc=True).shift(1)

    events = support.fresh_support_events(
        base,
        pair="TEST/USDT:USDT",
        cohort="normal",
        level_name="test_level",
        level=level,
        control="actual",
        event_kind="contact",
        source_open=source,
    )

    assert events["event_time"].tolist() == [
        pd.Timestamp("2026-08-20T01:00:00Z"),
        pd.Timestamp("2026-08-20T11:00:00Z"),
    ]
    assert set(events["period"]) == {"fresh_early"}
    assert (pd.to_datetime(events["source_open"], utc=True) < events["event_time"]).all()


def test_future_level_source_is_rejected() -> None:
    base = synthetic_base("2026-08-20T00:00:00Z", 24)
    base.loc[2, ["high", "low"]] = [101.0, 99.5]
    source = pd.to_datetime(base["date"], utc=True).shift(-1)
    with pytest.raises(ValueError, match="Future level source"):
        support.fresh_support_events(
            base,
            pair="TEST/USDT:USDT",
            cohort="normal",
            level_name="future_level",
            level=np.full(len(base), 100.0),
            control="actual",
            event_kind="contact",
            source_open=source,
        )


def test_matched_random_stays_in_period_and_away_from_actual_events() -> None:
    base = synthetic_base("2026-08-20T00:00:00Z", 32 * 24)
    base["high"] = 103.0
    base["low"] = 99.0
    actual_indexes = np.asarray([24, 17 * 24], dtype=int)
    periods = freshz.assign_fresh_period(base["date"])
    actual = DataFrame(
        {
            "period": periods.iloc[actual_indexes].to_numpy(),
            "approach_state": "from_below",
            "pre_distance_atr": 1.0,
            "base_index": actual_indexes,
        }
    )

    matched = support.fresh_matched_random_time_support(
        base,
        pair="TEST/USDT:USDT",
        cohort="normal",
        level_name="test_level",
        actual=actual,
    )

    assert set(matched["period"]) == {"fresh_early", "fresh_late"}
    assert len(matched) == 2
    for row in matched.itertuples(index=False):
        same_period_actual = int(
            actual.loc[actual["period"].eq(row.period), "base_index"].iloc[0]
        )
        assert abs(int(row.base_index) - same_period_actual) > support.COOLDOWN_HOURS


def test_selected_vwap_surface_matches_frozen_parent_equation() -> None:
    base = synthetic_base("2026-08-01T00:00:00Z", 10 * 24)
    base["high"] = base["close"] + 2.0
    base["low"] = base["close"] - 2.0
    selected = support._selected_vwap_surface(base, "TEST/USDT:USDT", "actual")
    parent = next(
        item
        for item in g24v.vwap_surfaces(base, "actual", "TEST/USDT:USDT")
        if item["level_name"] == support.VWAP_LEVEL_NAME
    )
    np.testing.assert_allclose(
        selected["level"],
        parent["level"],
        equal_nan=True,
    )
    assert pd.to_datetime(selected["source_open"], utc=True).equals(
        pd.to_datetime(parent["source_open"], utc=True)
    )


def test_convergence_builder_keeps_fresh_periods_and_no_outcome_columns() -> None:
    periods = 2300
    dates = pd.date_range("2026-06-20T00:00:00Z", periods=periods, freq="h", tz="UTC")
    angle = np.arange(periods, dtype=float) / 13.0
    close = 100.0 + 2.0 * np.sin(angle) + 0.4 * np.sin(angle / 7.0)
    base = DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.55,
            "low": close - 0.55,
            "close": close,
            "volume": 1000.0 + 50.0 * np.cos(angle),
            "base_atr": 1.0,
            "pre_close": pd.Series(close).shift(1),
        }
    )

    pooled, audits = support.build_convergence_pair(base, "TEST/USDT:USDT")

    assert audits
    assert set(item["period"] for item in audits) == {"fresh_early", "fresh_late"}
    assert set(pooled["period"]).issubset({"fresh_early", "fresh_late"})
    assert set(pooled["control"]).issubset(set(support.CONVERGENCE_CONTROLS))
    assert not any("future" in column or "profit" in column for column in pooled.columns)


def test_state_surface_matches_existing_outcome_blind_cache() -> None:
    manifest_path = g24cache.cache_manifest_path("normal")
    if not manifest_path.is_file() or not g22cache.CONTEXT_PATH.is_file():
        pytest.skip("Historical outcome-blind feature cache is not available.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(row for row in manifest["inventory"] if row["pair"] == "BTC/USDT:USDT")
    cached_path = Path(item["feature_path"])
    if not cached_path.is_file():
        pytest.skip("Historical BTC feature cache is not available.")
    base, _ = g20s.base_and_state("BTC/USDT:USDT", "normal")
    context = pd.read_parquet(g22cache.CONTEXT_PATH)
    actual, _ = support.state_feature_surface(base.reset_index(drop=True), context, "BTC/USDT:USDT")
    cached = pd.read_parquet(cached_path)
    columns = ["date", *support.STATE_FEATURE_COLUMNS]
    left = actual[columns].copy()
    right = cached[columns].copy()
    left["date"] = pd.to_datetime(left["date"], utc=True)
    right["date"] = pd.to_datetime(right["date"], utc=True)
    joined = left.merge(right, on="date", suffixes=("__new", "__cached"), validate="one_to_one")
    assert len(joined) >= 1000
    for column in support.STATE_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            joined[f"{column}__new"],
            joined[f"{column}__cached"],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_waiting_gate_creates_no_d_drive_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_root = tmp_path / "records"
    artifact_root = tmp_path / "bulky_support"
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text("{}", encoding="utf-8")
    frozen = freshz.freeze()
    waiting_coverage = {
        "batch_id": frozen["batch_id"],
        "status": "waiting_for_complete_fresh_ohlcv",
        "coverage": {"all_pair_coverage_pass": False, "pair_cells_passing": 0},
        "future_reaction_outcomes_read": False,
    }
    monkeypatch.setattr(support, "RECORD_ROOT", record_root)
    monkeypatch.setattr(support, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(
        support,
        "load_inputs",
        lambda *_args, **_kwargs: (frozen, waiting_coverage),
    )
    monkeypatch.setattr(
        support,
        "coverage_result_path",
        lambda _run_id: coverage_path,
    )

    result = support.prepare_support(
        "waiting_test",
        "coverage_test",
        refresh_coverage=False,
    )

    assert result["status"] == "waiting_for_complete_fresh_ohlcv"
    assert result["bulky_support_directory_created"] is False
    assert not artifact_root.exists()
    assert (record_root / "waiting_test" / "fresh_confirmation_support_result.json").is_file()
    assert result["future_reaction_outcomes_read"] is False


def test_current_frozen_contract_matches_reused_builders() -> None:
    frozen = freshz.freeze()
    support.validate_frozen_support_contract(frozen)
    assert frozen["status"] == "frozen_before_fresh_confirmation_outcomes"
    assert support.STATE_MODEL_FEATURE_COLUMNS == tuple(g22cache.STATE_FEATURES)
    assert support.VWAP_SCOPE_VALUE.endswith("centre_0.0")
