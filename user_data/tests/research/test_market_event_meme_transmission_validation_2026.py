# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_meme_transmission_validation_2026 as validation,
)


def test_binomial_upper_tail_is_exact_for_small_case() -> None:
    assert validation._binomial_upper_tail(3, 3) == 0.125
    assert validation._binomial_upper_tail(2, 3) == 0.5
