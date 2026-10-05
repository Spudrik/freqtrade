# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_anchored_vwap as g24v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_context_coverage as g24b,
)


def synthetic_base(rows: int = 1000) -> pd.DataFrame:
    close = 100.0 + np.linspace(0.0, 20.0, rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(100.0, 200.0, rows),
            "base_atr": np.full(rows, 1.0),
        }
    )


def test_current_session_vwap_excludes_current_candle() -> None:
    base = synthetic_base()
    centre, _, _ = g24v.weighted_session_stats(base, "1d", g24v.MODES[0])
    changed = base.copy()
    changed.loc[110, ["high", "low", "close", "volume"]] = 1e9
    revised, _, _ = g24v.weighted_session_stats(changed, "1d", g24v.MODES[0])

    assert centre.iloc[110] == revised.iloc[110]
    assert centre.iloc[111] != revised.iloc[111]


def test_common_gate_registers_all_five_routes_and_two_freqai_cohorts() -> None:
    contracts = g24c.all_support_contracts()

    assert len(contracts) == 6
    assert len({str(path) for path, _ in contracts}) == 6


def test_context_coverage_registry_matches_frozen_batch() -> None:
    frozen = g24b.load_freeze()

    assert frozen["status"] == "frozen_before_generation24_outcomes"
    assert g24b.CONTROLS[-1] == "same_band_no_level_time"
