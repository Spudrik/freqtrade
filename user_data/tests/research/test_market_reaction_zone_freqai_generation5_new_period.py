# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation5_new_period as g5c,
)


def test_controls_are_nonself_and_stale_is_strictly_prior() -> None:
    rows = 4
    events = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-20", periods=rows, freq="4h", tz="UTC"),
            "period": ["g5c_confirmation_early"] * rows,
            **{
                column: np.arange(rows, dtype=float) + offset
                for offset, column in enumerate(g5c.CURRENT_COLUMNS)
            },
        }
    )
    output, audit = g5c.attach_controls(events, pair="ETH/USDT:USDT")
    first = g5c.CURRENT_COLUMNS[0]
    shuffled = g5c.SHUFFLE_COLUMNS[0]
    stale = g5c.STALE_COLUMNS[0]
    assert not np.array_equal(output[first].to_numpy(), output[shuffled].to_numpy())
    assert output.loc[1:, stale].tolist() == output.loc[:2, first].tolist()
    assert pd.isna(output.loc[0, stale])
    assert audit[0]["shuffle_self_assignments"] == 0
    assert audit[0]["stale_future_source_violations"] == 0


def test_post_control_coverage_uses_only_period_counts() -> None:
    results = []
    for index, pair in enumerate(
        (
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "BNB/USDT:USDT",
            "SOL/USDT:USDT",
            "XRP/USDT:USDT",
            "ADA/USDT:USDT",
        )
    ):
        results.append(
            {
                "pair": pair,
                "period_rows": {
                    "g5c_confirmation_early": 4 if index else 7,
                    "g5c_confirmation_late": 4 if index else 7,
                },
                "arbitrary_target_summary": 999999.0,
            }
        )
    coverage = g5c.post_control_coverage(results)
    assert coverage["primary_rows"].eq(20).all()
    assert coverage["primary_coins"].eq(5).all()
    assert coverage["status"].eq("supported").all()
    assert not coverage["outcome_values_used_for_coverage_decision"].any()


def test_comparison_requires_uncertainty_interval_above_zero() -> None:
    row = pd.Series(
        {
            "supported": True,
            "equal_coin_paired_mae_gain": 0.1,
            "block_bootstrap_lower_95": 0.0,
            "positive_coin_count": g5c.MIN_RETAIN_COINS,
            "leave_one_coin_out_positive": True,
            "spearman_change": 0.01,
            "profile_calibration_slope": 0.5,
            "profile_band_rows": g5c.MIN_SCORABLE_ROWS,
        }
    )
    assert not g5c.comparison_passed(row)
    row["block_bootstrap_lower_95"] = 0.001
    assert g5c.comparison_passed(row)


def test_model_cutoff_audit_requires_one_july20_file_per_pair(tmp_path) -> None:
    prediction_dir = tmp_path / "backtesting_predictions"
    prediction_dir.mkdir()
    epoch = int(g5c.CONFIRMATION_START.timestamp())
    for stem in ("cb_eth", "cb_sol"):
        (prediction_dir / f"{stem}_{epoch}_prediction.feather").touch()
    audit = g5c.model_cutoff_audit(
        tmp_path,
        ("ETH/USDT:USDT", "SOL/USDT:USDT"),
    )
    assert audit["passed"]
    (prediction_dir / f"cb_eth_{epoch + 86400}_prediction.feather").touch()
    assert not g5c.model_cutoff_audit(
        tmp_path,
        ("ETH/USDT:USDT", "SOL/USDT:USDT"),
    )["passed"]
