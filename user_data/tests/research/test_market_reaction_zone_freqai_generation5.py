# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation5 as g5b,
)


def test_contact_clock_keeps_contact_and_next_volume_denominators_causal() -> None:
    volume = np.arange(1.0, 32.0)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(volume), freq="1h", tz="UTC"),
            "open": np.full(len(volume), 100.0),
            "high": np.full(len(volume), 102.0),
            "low": np.full(len(volume), 99.0),
            "close": np.full(len(volume), 101.0),
            "volume": volume,
        }
    )
    clocks = g5b.contact_clock_frame(frame)
    position = 25
    expected_contact = volume[position] / np.median(volume[position - 24 : position])
    expected_next = volume[position + 1] / np.median(volume[position - 23 : position + 1])
    assert np.isclose(clocks.loc[position, g5b.CONTACT_TARGET], expected_contact)
    assert np.isclose(clocks.loc[position, g5b.NEXT_VOLUME_TARGET], expected_next)


def test_source_feature_groups_separate_contact_resolution_from_precontact() -> None:
    frame = pd.DataFrame(
        columns=[
            "date",
            "level__pre_distance_atr",
            "level__zone_support_fraction",
            "level__arrival_repeat_contact",
            "geometry__other_density_zone_count",
            "geometry__peer_contact_count",
            "mtf__reference_level_count",
        ]
    )
    groups = g5b.classify_source_features(frame)
    assert groups["path"] == ("level__pre_distance_atr",)
    assert "level__arrival_repeat_contact" in groups["resolution"]
    assert "geometry__peer_contact_count" in groups["resolution"]
    assert "geometry__other_density_zone_count" in groups["proximity"]
    assert "mtf__reference_level_count" in groups["proximity"]
    assert groups["level"] == ("level__zone_support_fraction",)


def test_structurally_empty_proximity_is_allowed_but_path_is_not() -> None:
    frame = pd.DataFrame(
        {
            "pre_proximity__count": [0.0, 0.0],
            "pre_path__distance": [1.0, 1.0],
        }
    )
    pruned, kept = g5b.prune_feature_block(
        frame,
        prefix="pre_proximity",
        pair="ETH/USDT:USDT",
        surface_id="normal_isolated_confirmed_swing_density",
    )
    assert kept == []
    assert "pre_proximity__count" not in pruned

    with np.testing.assert_raises_regex(ValueError, "pre_path block is empty"):
        g5b.prune_feature_block(
            frame,
            prefix="pre_path",
            pair="ETH/USDT:USDT",
            surface_id="test",
        )


def test_shuffled_and_stale_placebos_preserve_clock_rules() -> None:
    events = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="6h", tz="UTC"),
            "period": ["validation_early"] * 4,
            "pre_path__distance": [1.0, 2.0, 3.0, 4.0],
            "pre_level__width": [10.0, 20.0, 30.0, 40.0],
            "pre_proximity__count": [0.0, 1.0, 2.0, 3.0],
        }
    )
    output, audit = g5b.attach_placebo_groups(
        events,
        candidate_columns=(
            "pre_path__distance",
            "pre_level__width",
            "pre_proximity__count",
        ),
        pair="ETH/USDT:USDT",
        surface_id="test",
    )
    assert output.loc[1:, "stale_path__distance"].tolist() == [1.0, 2.0, 3.0]
    assert output.loc[0, "stale_path__distance"] != output.loc[0, "stale_path__distance"]
    assert not np.array_equal(
        output["shuffle_path__distance"].to_numpy(),
        output["pre_path__distance"].to_numpy(),
    )
    assert audit[0]["shuffle_self_assignments"] == 0
    assert audit[0]["stale_future_source_violations"] == 0


def test_block_bootstrap_reports_positive_equal_coin_gain() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["ETH/USDT:USDT"] * 4 + ["SOL/USDT:USDT"] * 4,
            "date": pd.date_range("2026-01-01", periods=8, freq="7D", tz="UTC"),
            "paired_error_gain": [0.1, 0.2, 0.1, 0.2, 0.2, 0.3, 0.2, 0.3],
        }
    )
    point, lower, upper = g5b.deterministic_block_bootstrap(
        frame,
        seed_key="test",
        samples=128,
    )
    assert point > 0.0
    assert lower > 0.0
    assert upper > lower


def test_block_bootstrap_preserves_unequal_block_row_counts() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["ETH/USDT:USDT"] * 11 + ["SOL/USDT:USDT"] * 4,
            "date": (
                [pd.Timestamp("2026-01-01", tz="UTC")] * 10
                + [pd.Timestamp("2026-01-08", tz="UTC")]
                + list(pd.date_range("2026-01-01", periods=4, freq="7D", tz="UTC"))
            ),
            "paired_error_gain": [1.0] * 10 + [11.0] + [3.0] * 4,
        }
    )
    point, _, _ = g5b.deterministic_block_bootstrap(
        frame,
        seed_key="unequal-block-rows",
        samples=128,
    )
    expected_eth = 21.0 / 11.0
    expected_equal_coin_point = (expected_eth + 3.0) / 2.0
    assert np.isclose(point, expected_equal_coin_point)


def test_comparison_pass_requires_positive_calibration_slope() -> None:
    row = pd.Series(
        {
            "supported": True,
            "equal_coin_paired_mae_gain": 0.1,
            "positive_coin_count": g5b.MIN_RETAIN_COINS,
            "leave_one_coin_out_positive": True,
            "spearman_change": 0.01,
            "profile_calibration_slope": -0.1,
            "profile_band_rows": g5b.MIN_SCORABLE_ROWS,
        }
    )
    assert not g5b.comparison_passed(row)


def test_clock_comparisons_do_not_treat_nowcast_as_forecast() -> None:
    mapping = {item[0]: item[4] for item in g5b.CLOCK_COMPARISONS}
    assert mapping["contact_close_nowcast_increment"] == "same_candle_description_or_nowcast"
    assert (
        mapping["strict_precontact_contact_volume_vs_state"]
        == "strict_pre_contact_forecast"
    )
    assert (
        mapping["contact_close_next_volume_vs_state"]
        == "contact_close_causal_next_candle_forecast"
    )
