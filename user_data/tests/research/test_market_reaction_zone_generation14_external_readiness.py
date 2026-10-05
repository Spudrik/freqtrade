# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_external_readiness as g14e,
)


def test_external_summary_requires_rows_and_coin_breadth() -> None:
    frame = pd.DataFrame(
        {
            "window": ["standard_validation"] * 60,
            "analysis_period": ["period_a"] * 60,
            "pair": [f"PAIR-{index % 6}" for index in range(60)],
            "ready__g11_news_context": True,
            "ready__g11_orderbook_context": [index % 6 == 0 for index in range(60)],
        }
    )

    output = g14e.summarize("normal", frame).set_index("source")

    assert bool(output.loc["historical_news", "model_gate_passed"])
    assert not bool(output.loc["historical_orderbook", "model_gate_passed"])


def test_standard_window_uses_frozen_period_names() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"], utc=True),
            "period": ["validation_early"],
            "ready__g11_news_context": [False],
            "ready__g11_orderbook_context": [False],
        }
    )

    output = g14e.analysis_windows(frame, "normal")

    assert output.loc[0, "window"] == "standard_validation"
    assert output.loc[0, "analysis_period"] == "validation_early"
