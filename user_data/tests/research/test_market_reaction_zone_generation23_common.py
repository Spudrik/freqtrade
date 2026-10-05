# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_common as g23c,
)


def test_score_control_ladders_uses_shared_generation18_acceptance(monkeypatch) -> None:
    events = pd.DataFrame({"placeholder": [1]})
    summary = pd.DataFrame({"summary": [1]})
    contrasts = pd.DataFrame({"contrast": [1]})
    scores = pd.DataFrame({"score": [1]})
    decisions = pd.DataFrame({"decision": [1]})

    monkeypatch.setattr(g23c.g22a, "metric_summary", lambda value: summary)
    monkeypatch.setattr(
        g23c.g18d, "paired_contrasts", lambda value, controls: contrasts
    )
    monkeypatch.setattr(
        g23c.g18d, "period_scores", lambda value, keys: scores
    )
    monkeypatch.setattr(
        g23c.g18d,
        "whole_decisions",
        lambda value, keys, controls: decisions,
    )

    actual = g23c.score_control_ladders(events, controls=("control_a",))

    assert actual == (contrasts, scores, decisions)
