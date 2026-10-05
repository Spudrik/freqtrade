"""Integrity checks for the outcome-blind Stage-2 source assembly."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Custom_Launcher/research/context_features/market_signal_portfolio_stage2_sources.py"
)
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("market_signal_portfolio_stage2_sources", SCRIPT)
assert spec and spec.loader
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)


def test_future_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Future source input"):
        stage2._row(
            "recent_volume_persistence",
            decision_utc="2026-01-01T00:00:00Z",
            latest_input_utc="2026-01-01T00:01:00Z",
            call_state="issued",
        )


def test_frozen_rows_are_causal_and_keep_control_clock() -> None:
    result = stage2.run()
    rows = pd.read_parquet(result["rows_path"])
    assert len(rows) == result["decision_row_count"] == 6863
    assert set(rows.call_state) == {"issued", "no_call"}
    assert rows.loc[rows.call_state.eq("no_call"), "no_call_reason"].ne("").all()
    assert rows["signal_direction"].dropna().isin([-1, 0, 1]).all()
    assert (
        pd.to_datetime(
            rows.latest_input_utc.loc[rows.latest_input_utc.ne("")], utc=True, format="mixed"
        )
        .le(
            pd.to_datetime(
                rows.decision_utc.loc[rows.latest_input_utc.ne("")], utc=True, format="mixed"
            )
        )
        .all()
    )
    control = rows.loc[
        rows.source_episode_id.eq("layer2:fomc_2021_01_27|control|fixed_prior_week|1")
    ].iloc[0]
    assert control.anchor_utc != "2021-01-27T19:00:00Z"
    assert pd.Timestamp(control.decision_utc) == pd.Timestamp(control.anchor_utc) + pd.Timedelta(
        hours=4
    )
    assert result["outcomes_read"] is False
    assert result["missing_lanes"] == [
        "calculated_area_contact_traffic",
        "local_multitimeframe_support_resistance_map",
    ]
