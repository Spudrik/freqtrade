# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_new_period_preflight as g5c,
)


def test_source_manifest_keeps_only_new_slices_as_confirmation() -> None:
    manifest, path, _, artifact_dir = g5c.source_manifest_contract("unit_test_contract")
    periods = {item["id"]: item for item in manifest["data"]["chronological_periods"]}
    assert periods["g5c_exposed_training_bridge"]["role"] == (
        "exposed_training_only_not_confirmation"
    )
    assert periods["g5c_confirmation_early"]["start_utc"] == "2026-07-20T00:00:00Z"
    assert periods["g5c_confirmation_late"]["end_utc_exclusive"] == (
        "2026-08-20T00:00:00Z"
    )
    assert manifest["research_boundary"]["predict_direction"] is False
    assert manifest["research_boundary"]["optimize_trade_profit"] is False
    assert str(path).startswith(str(g5c.RECORD_ROOT))
    assert str(artifact_dir).startswith(str(g5c.LARGE_ROOT))


def test_coverage_gate_uses_rows_and_coins_not_target_values() -> None:
    pairs = (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
    )
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for surface in g5c.SURFACES:
        for pair in pairs:
            periods: list[str] = []
            if not pair.startswith("BTC/"):
                periods.extend(["g5c_confirmation_early"] * 4)
                periods.extend(["g5c_confirmation_late"] * 4)
            frames[(surface, pair)] = pd.DataFrame(
                {
                    "period": periods,
                    g5c.TARGET_COLUMN: [999999.0] * len(periods),
                }
            )
    coverage = g5c.coverage_table(frames, pairs)
    confirmation = coverage.loc[coverage["period"].isin(g5c.CONFIRMATION_PERIODS)]
    assert confirmation["status"].eq("supported").all()
    assert confirmation["primary_coins"].eq(5).all()
    assert confirmation["primary_rows"].eq(20).all()
    assert not confirmation["outcome_values_read_for_coverage_decision"].any()


def test_coverage_gate_rejects_fewer_than_five_non_btc_coins() -> None:
    pairs = (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
    )
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for surface in g5c.SURFACES:
        for pair in pairs:
            periods = ["g5c_confirmation_early"] * 10
            periods += ["g5c_confirmation_late"] * 10
            frames[(surface, pair)] = pd.DataFrame({"period": periods})
    coverage = g5c.coverage_table(frames, pairs)
    confirmation = coverage.loc[coverage["period"].isin(g5c.CONFIRMATION_PERIODS)]
    assert confirmation["primary_rows"].eq(40).all()
    assert confirmation["primary_coins"].eq(4).all()
    assert confirmation["status"].eq("insufficient_common_support").all()
