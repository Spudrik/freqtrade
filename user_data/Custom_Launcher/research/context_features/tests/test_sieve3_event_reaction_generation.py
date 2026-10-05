from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
CONTEXT_DIR = REPO_ROOT / "user_data" / "Custom_Launcher" / "research" / "context_features"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEXT_DIR))

from sieve3_event_reaction_research import (  # noqa: E402
    build_exit_counterfactuals,
    matched_controls,
    nearest_neighbor_candidates,
    pair_token,
    parse_horizons,
)
from sieve3_event_freqai_round1 import target_metadata  # noqa: E402
from sieve3_event_level_relationships import _level_state, parse_bands  # noqa: E402
from user_data.strategies.Sieve3EventReactionFreqAIResearchStrategy import (  # noqa: E402
    build_sieve3_reaction_targets,
)


class Sieve3EventReactionGenerationTests(unittest.TestCase):
    def test_chunked_nearest_neighbors_match_exact_euclidean_order(self) -> None:
        pool = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]],
            dtype="float64",
        )
        events = np.array([[0.9, 0.1], [2.5, 2.8]], dtype="float64")
        distances, indices = nearest_neighbor_candidates(
            pool,
            events,
            3,
            chunk_size=1,
        )
        expected_distances = np.sqrt(
            np.sum((events[:, np.newaxis, :] - pool[np.newaxis, :, :]) ** 2, axis=2)
        )
        expected_indices = np.argsort(expected_distances, axis=1, kind="stable")[:, :3]
        self.assertTrue(np.array_equal(indices, expected_indices))
        self.assertTrue(
            np.allclose(
                distances,
                np.take_along_axis(expected_distances, expected_indices, axis=1),
            )
        )

    def test_extended_horizons_and_timing_targets_exist(self) -> None:
        rows = 180
        close = pd.Series(100.0 + np.linspace(0.0, 20.0, rows))
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC"),
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 1.5,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(100.0, 200.0, rows),
            }
        )

        labelled = build_sieve3_reaction_targets(frame)

        for horizon in (1, 2, 4, 8, 12, 24, 48):
            self.assertIn(f"&-future_return_{horizon}h", labelled)
            self.assertIn(f"&-future_upside_peak_step_{horizon}h", labelled)
            self.assertIn(f"&-future_downside_peak_step_{horizon}h", labelled)
        for horizon in (4, 8, 12, 24, 48):
            self.assertIn(f"&-up_before_down_1atr_{horizon}h", labelled)
            self.assertIn(f"&-first_1atr_touch_step_{horizon}h", labelled)

    def test_pair_and_horizon_parsing_is_explicit(self) -> None:
        self.assertEqual(pair_token("BNB/USDT:USDT"), "bnb")
        self.assertEqual(parse_horizons("48,1,4,4"), (1, 4, 48))
        with self.assertRaises(ValueError):
            parse_horizons("3")
        self.assertEqual(
            target_metadata("&-future_volume_ratio_12h"),
            ("volume_activity", 12),
        )

    def test_visible_state_matcher_returns_diagnostics(self) -> None:
        dates = pd.date_range("2025-01-01", periods=240, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "pair": "BTC/USDT:USDT",
                "decision_time": dates,
                "summary__onset_count": 0,
                "atr_pct": np.linspace(0.01, 0.03, len(dates)),
                "pre_return_6h": np.sin(np.arange(len(dates)) / 20.0) / 100.0,
                "pre_return_24h": np.sin(np.arange(len(dates)) / 40.0) / 50.0,
                "pre_volume_ratio_24h": 1.0 + np.cos(np.arange(len(dates)) / 15.0) / 10.0,
                "pre_pressure_6h": np.sin(np.arange(len(dates)) / 10.0),
                "pre_pressure_24h": np.sin(np.arange(len(dates)) / 30.0),
                "pre_volatility_24h": np.linspace(0.005, 0.02, len(dates)),
                "pre_range_position_168h": np.linspace(0.0, 1.0, len(dates)),
            }
        )
        event_mask = pd.Series(False, index=frame.index)
        event_mask.iloc[[100, 180]] = True
        frame.loc[event_mask, "summary__onset_count"] = 1

        controls = matched_controls(
            frame,
            event_mask,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(len(controls), 2)
        self.assertIn("control_match_distance", controls)
        self.assertIn("control_match_time_separation_h", controls)
        self.assertTrue(controls["control_match_distance"].notna().all())
        self.assertFalse(controls["control_match_reused"].any())

    def test_exit_counterfactual_labels_remain_distinct(self) -> None:
        decision_time = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")
        dates = pd.date_range(
            decision_time - pd.Timedelta(hours=1), periods=6, freq="1h", tz="UTC"
        )
        actual = pd.DataFrame(
            {
                "pair": ["BTC/USDT:USDT"] * len(dates),
                "date": dates,
                "decision_time": dates,
                "high": [104.0, 101.0, 106.0, 103.0, 102.0, 103.0],
                "low": [102.0, 99.0, 98.0, 92.0, 97.0, 101.0],
                "close": [103.0, 100.0, 102.0, 99.0, 101.0, 102.0],
            }
        )
        exits = pd.DataFrame(
            {
                "entry_id": ["example_long"],
                "exit_family": ["example_exit"],
                "pair": ["BTC/USDT:USDT"],
                "entry_side": ["long"],
                "entry_timeframe": ["1h"],
                "open_date": [decision_time - pd.Timedelta(hours=5)],
                "decision_time": [decision_time],
                "open_rate": [95.0],
                "close_rate": [100.0],
                "profit_ratio": [0.05],
                "exit_reason": ["example"],
                "source_backtest_path": ["example.zip"],
                "source_sparse": [False],
            }
        )

        events, summary = build_exit_counterfactuals(
            exits,
            actual,
            windows={
                "test": (
                    pd.Timestamp("2025-01-01", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                )
            },
            horizons=(4,),
            earlier_offsets=(1,),
        )

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(float(events.iloc[0]["earlier_exit_advantage_1h"]), 0.03)
        self.assertAlmostEqual(float(events.iloc[0]["hold_instead_delta"]), 0.02)
        self.assertAlmostEqual(float(events.iloc[0]["missed_additional_profit"]), 0.06)
        self.assertAlmostEqual(float(events.iloc[0]["avoided_loss_after_exit"]), 0.08)
        self.assertEqual(int(summary.iloc[0]["unique_trade_paths"]), 1)

    def test_exit_counterfactual_uses_unclipped_raw_path_extremes(self) -> None:
        decision_time = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")
        dates = pd.date_range(decision_time, periods=3, freq="1h", tz="UTC")
        actual = pd.DataFrame(
            {
                "pair": ["BTC/USDT:USDT"] * 3,
                "date": dates,
                "decision_time": dates,
                "high": [101.0, 130.0, 141.0],
                "low": [99.0, 98.0, 100.0],
                "close": [100.0, 120.0, 140.0],
            }
        )
        exits = pd.DataFrame(
            {
                "entry_id": ["example_long"],
                "exit_family": ["example_exit"],
                "pair": ["BTC/USDT:USDT"],
                "entry_side": ["long"],
                "entry_timeframe": ["1h"],
                "open_date": [decision_time - pd.Timedelta(hours=1)],
                "decision_time": [decision_time],
                "open_rate": [95.0],
                "close_rate": [100.0],
                "profit_ratio": [0.05],
                "exit_reason": ["example"],
                "source_backtest_path": ["example.zip"],
                "source_sparse": [False],
            }
        )

        events, _ = build_exit_counterfactuals(
            exits,
            actual,
            windows={
                "test": (
                    pd.Timestamp("2025-01-01", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                )
            },
            horizons=(2,),
            earlier_offsets=(),
        )

        self.assertAlmostEqual(float(events.iloc[0]["hold_instead_delta"]), 0.40)
        self.assertAlmostEqual(float(events.iloc[0]["best_post_exit_delta"]), 0.41)
        self.assertLessEqual(
            float(events.iloc[0]["hold_instead_delta"]),
            float(events.iloc[0]["best_post_exit_delta"]),
        )

    def test_level_relationship_does_not_assume_higher_tf_precedence(self) -> None:
        decision_time = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")
        events = pd.DataFrame(
            {
                "pair": ["BTC/USDT:USDT"],
                "decision_time": [decision_time],
                "entry_side": ["long"],
            }
        )
        levels = pd.DataFrame(
            {
                "pair": ["BTC/USDT:USDT"],
                "decision_time": [decision_time],
                "level_state_close": [100.0],
                "support__4h:trendline": [99.5],
                "resistance__1d:va_edge": [100.5],
            }
        )

        result = _level_state(
            events,
            levels,
            bands=parse_bands("0.01"),
            evidence_source="test",
        )

        self.assertEqual(result.iloc[0]["relationship_state"], "mixed")
        self.assertEqual(result.iloc[0]["mtf_precedence_state"], "opposing_higher_tf")


if __name__ == "__main__":
    unittest.main()
