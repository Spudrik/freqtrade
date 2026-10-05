# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_convergence_representation as g25r,
)


def _rows(control: str, count: int, start: int = 0) -> pd.DataFrame:
    indexes = np.arange(start, start + count)
    return pd.DataFrame(
        {
            "control": control,
            "period": "g18_normal_early",
            "event_time": pd.date_range("2026-07-20", periods=count, freq="h", tz="UTC"),
            "level_name": [f"surface_{value % 3}" for value in indexes],
            "pre_distance_atr": 0.1 + (indexes % 5) * 0.1,
            "approach_state": np.where(indexes % 2, "from_above", "from_below"),
            "level_price": 100.0 + indexes,
            "zone_half_width": 1.0,
            "pair": "BTC/USDT:USDT",
            "cohort": "normal",
        }
    )


def test_equal_support_surface_balances_every_control() -> None:
    parts = [_rows("actual", 15)]
    parts.extend(_rows(control, 12 + number) for number, control in enumerate(g25r.CONTROLS))
    support, audit = g25r.equal_support_surface(pd.concat(parts, ignore_index=True))
    counts = support["control"].value_counts().to_dict()
    assert set(counts) == {"actual", *g25r.CONTROLS}
    assert set(counts.values()) == {12}
    assert audit[0]["representation_gate_pass"] is True


def test_equal_support_surface_fails_small_representation_without_opening_outcomes() -> None:
    parts = [_rows("actual", 9)]
    parts.extend(_rows(control, 20) for control in g25r.CONTROLS)
    support, audit = g25r.equal_support_surface(pd.concat(parts, ignore_index=True))
    assert not support.empty
    assert audit[0]["common_count"] == 9
    assert audit[0]["representation_gate_pass"] is False
    assert not any(column.startswith("metric__") for column in support.columns)


def test_deduplication_counts_contributing_definitions() -> None:
    duplicate = pd.concat([_rows("actual", 3), _rows("actual", 3)], ignore_index=True)
    duplicate.loc[3:, "level_name"] = "extra_surface"
    pooled = g25r._deduplicate_pool(duplicate)
    assert len(pooled) == 3
    assert pooled["source_definition_count"].eq(2).all()
