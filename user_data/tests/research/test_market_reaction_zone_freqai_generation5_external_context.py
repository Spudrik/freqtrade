# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation5_external_context as g5e,
)


def test_causal_regime_uses_only_prior_activity_for_thresholds() -> None:
    dates = pd.date_range("2026-01-01", periods=900, freq="1h", tz="UTC")
    activity = pd.Series(1.0, index=dates)
    activity.iloc[400:800] = 2.0
    activity.iloc[800] = 100.0
    ready = pd.Series(True, index=dates)
    regime, thresholds = g5e.causal_regime(activity, ready)
    assert thresholds.loc[dates[800], "upper"] < 100.0
    assert regime.loc[dates[800]] == "shock"
    assert regime.iloc[100] == "unavailable"


def test_daily_carry_expires_instead_of_turning_missing_into_zero() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    context = pd.DataFrame(
        {
            "date": dates,
            "context_source_future_violation": np.zeros(len(dates)),
            "ctx_max_source_available_at": pd.Series(
                pd.NaT,
                index=range(len(dates)),
                dtype="datetime64[ns, UTC]",
            ),
            "ctx_test_metric": [5.0] + [np.nan] * (len(dates) - 1),
        }
    )
    context.loc[0, "ctx_max_source_available_at"] = dates[0]
    carried, ready = g5e.causal_daily_carry(context, ("ctx_test_metric",))
    assert carried.loc[30, "value__test_metric"] == 5.0
    assert pd.isna(carried.loc[31, "value__test_metric"])
    assert ready.iloc[:31].all()
    assert not ready.iloc[31:].any()


def test_post_control_gate_uses_rows_not_reaction_values() -> None:
    pairs = (
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
    )
    rows = []
    for period in g5e.SURFACES["normal_thin_lvn_mixed_cluster"]["validation_periods"]:
        for pair in pairs:
            for index in range(10):
                rows.append(
                    {
                        "pair": pair,
                        "period": period,
                        "score_selected": True,
                        "context_regime": "quiet" if index < 5 else "shock",
                        g5e.TARGET_COLUMN: 999999.0,
                    }
                )
        for index in range(100):
            rows.append(
                {
                    "pair": "BTC/USDT:USDT",
                    "period": period,
                    "score_selected": True,
                    "context_regime": "shock",
                    g5e.TARGET_COLUMN: -999999.0,
                }
            )
    coverage = pd.DataFrame(
        g5e.coverage_rows(
            pd.DataFrame(rows),
            surface_id="normal_thin_lvn_mixed_cluster",
            context_block="gdelt_aggregate",
        )
    )
    assert coverage["primary_rows"].eq(50).all()
    assert coverage["primary_coins"].eq(5).all()
    assert coverage["btc_rows"].eq(100).all()
    assert coverage["status"].eq("supported").all()
    assert not coverage["reaction_outcomes_used_for_gate"].any()


def test_context_comparison_requires_confidence_interval_above_zero() -> None:
    row = pd.Series(
        {
            "supported": True,
            "equal_coin_paired_mae_gain": 0.1,
            "block_bootstrap_lower_95": 0.0,
            "positive_coin_count": g5e.MINIMUM_COINS_PER_PERIOD,
            "leave_one_coin_out_positive": True,
            "spearman_change": 0.01,
            "profile_calibration_slope": 0.5,
            "profile_band_rows": g5e.MIN_SCORABLE_ROWS,
        }
    )
    assert not g5e.comparison_passed(row)
    row["block_bootstrap_lower_95"] = 0.001
    assert g5e.comparison_passed(row)


def test_quiet_shock_contrast_bootstrap_detects_repeated_difference() -> None:
    rows = []
    for coin in range(5):
        pair = f"COIN{coin}/USDT:USDT"
        for week in range(6):
            date = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=7 * week)
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "context_regime": "quiet",
                    "level_error_gain": 0.1 + coin * 0.001,
                }
            )
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "context_regime": "shock",
                    "level_error_gain": 0.4 + coin * 0.001,
                }
            )
    point, lower, upper, coins = g5e.regime_contrast_bootstrap(
        pd.DataFrame(rows),
        seed_key="unit-test",
        samples=128,
    )
    assert coins == 5
    assert point > 0.0
    assert lower > 0.0
    assert upper >= lower
