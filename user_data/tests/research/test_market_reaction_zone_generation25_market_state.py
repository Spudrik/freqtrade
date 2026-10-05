# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_market_state_activity as g25a,
)


def test_causal_percentile_does_not_change_when_future_changes() -> None:
    values = pd.Series(np.linspace(1.0, 1000.0, 1000))
    original = g25a.causal_rolling_percentile(values)
    changed = values.copy()
    changed.iloc[800:] = 1e9
    perturbed = g25a.causal_rolling_percentile(changed)
    assert np.allclose(original.iloc[:800], perturbed.iloc[:800], equal_nan=True)


def test_frozen_controls_are_plainly_distinct() -> None:
    assert set(g25a.CONTROLS) == {
        "constant_training_median",
        "within_pair_time_shuffled_training_labels",
        "simple_recent_activity",
        "causal_stale_activity_24h",
    }
    assert g25a.CANDIDATE_ROLE not in g25a.CONTROLS


def test_parent_paths_use_full_evidence_run_not_smoke() -> None:
    for cohort in ("normal", "meme"):
        path = g25a.parent_manifest_path(cohort)
        assert g24_smoke_absent(path.name, path.parent.name)


def g24_smoke_absent(*parts: str) -> bool:
    return all("smoke" not in part for part in parts)
