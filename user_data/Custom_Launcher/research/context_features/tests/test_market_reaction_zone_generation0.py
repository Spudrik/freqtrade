from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    ClusterComponent,
    add_nested_relationships,
    classify_cluster,
    connected_level_clusters,
    contacted_component_metadata,
    dedupe_contact_candidates,
    largest_connected_interval_component,
    metric_spec,
    positive_fraction,
    select_representation_specs,
    source_availability_metadata,
    transformed_metric,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    PROFILES,
    TARGETS,
    common_prediction_keys,
    profile_config,
    target_metadata,
    validate_manifest_request,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    MAX_NEW_WORKERS,
    AtlasTask,
    LevelSpec,
    candidate_level_columns,
    episode_start_mask,
    event_frame,
    future_path_matrices,
    generic_level_frame,
    level_specs,
    load_manifest,
    nice_number_step,
    normalize_dates,
    resolved_level_values,
    sha256_file,
    timeframe_delta,
    transform_control,
    validate_cache_metadata,
    validate_worker_count,
    wilder_atr,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0_review import (  # noqa: E501
    KEY_COLUMNS,
    METRIC_COLUMNS,
    PRIMARY_DELTA_COLUMNS,
    add_cohort_lenses,
    aggregate_comparisons,
    attribute_calibration_cells,
    build_control_screen,
    calibratable_attribute_columns,
    comparison_from_summary,
    coverage_row,
    timing_comparison_from_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    LOCAL_MATCH_FEATURES,
    attach_overlap,
    causal_local_state,
    nearest_state_pairs,
)
from user_data.strategies.MarketReactionZoneFreqAIResearchStrategy import (
    SELECTED_MTF_LEVEL_TIMEFRAMES,
    TARGET_HORIZONS,
    MarketReactionZoneMarketStateFreqAIResearchStrategy,
    _future_extreme,
    _future_mean,
    build_market_reaction_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
MANIFEST = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation0_manifest.json"
)


class MarketReactionZoneGeneration0Tests(unittest.TestCase):
    def test_manifest_keeps_direction_and_profit_out_of_generation_zero(self) -> None:
        manifest = load_manifest(MANIFEST)
        self.assertFalse(manifest["research_boundary"]["predict_direction"])
        self.assertFalse(manifest["research_boundary"]["optimize_trade_profit"])
        self.assertNotIn("3d", manifest["data"]["source_timeframes"])

    def test_worker_cap_rejects_oversubscription(self) -> None:
        validate_worker_count(MAX_NEW_WORKERS)
        with self.assertRaises(ValueError):
            validate_worker_count(MAX_NEW_WORKERS + 1)
        with self.assertRaises(ValueError):
            validate_worker_count(84)

    def test_cache_validation_rejects_changed_ohlcv_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            source = root / "source.feather"
            cache = root / "levels.parquet"
            indicator = Path(validate_cache_metadata.__code__.co_filename).resolve()
            manifest.write_text('{"frozen": true}', encoding="utf-8")
            source.write_bytes(b"original ohlcv")
            cache.write_bytes(b"cache placeholder")
            metadata = {
                "manifest_sha256": sha256_file(manifest),
                "source": str(source),
                "source_sha256": sha256_file(source),
                "indicator_sha256": {
                    str(indicator.relative_to(REPO_ROOT)): sha256_file(indicator)
                },
                "direction_prediction": False,
                "profit_optimization": False,
            }
            cache.with_suffix(".meta.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            validate_cache_metadata(cache, manifest)
            source.write_bytes(b"updated ohlcv")
            with self.assertRaisesRegex(ValueError, "OHLCV source changed"):
                validate_cache_metadata(cache, manifest)

    def test_generation1_local_state_does_not_change_when_future_candles_change(self) -> None:
        rows = 260
        close = pd.Series(np.linspace(100.0, 150.0, rows))
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(1000.0, 2000.0, rows),
            }
        )
        frame["base_atr"] = wilder_atr(frame, 14).shift(1)
        changed = frame.copy()
        changed.loc[220:, ["open", "high", "low", "close", "volume"]] *= 5.0
        before = causal_local_state(frame)
        after = causal_local_state(changed)
        pd.testing.assert_frame_equal(before.iloc[:220], after.iloc[:220])

    def test_generation1_matching_uses_continuous_state_with_bounded_reuse(self) -> None:
        columns = list(LOCAL_MATCH_FEATURES)
        actual = pd.DataFrame(
            {column: np.linspace(0.0, 1.0, 12) for column in columns}
        )
        control = pd.DataFrame(
            {column: np.linspace(0.01, 1.01, 12) for column in columns}
        )
        actual["pre_distance_atr"] = 1.0
        control["pre_distance_atr"] = 1.0
        actual["event_time"] = pd.date_range(
            "2024-01-01", periods=12, freq="1h", tz="UTC"
        )
        control["event_time"] = pd.date_range(
            "2023-01-01", periods=12, freq="1h", tz="UTC"
        )
        pairs, audit = nearest_state_pairs(actual, control, state_columns=columns)
        self.assertEqual(audit["eligible_actual"], 12)
        self.assertEqual(len(pairs), 12)
        counts = pd.Series([pair[1] for pair in pairs]).value_counts()
        self.assertLessEqual(int(counts.max()), 3)

    def test_generation1_overlap_separates_source_from_other_real_levels(self) -> None:
        events = pd.DataFrame(
            {
                "base_index": [0, 1],
                "level_price": [100.19, 110.0],
                "zone_half_width": [0.10, 0.10],
            }
        )
        real = np.array([[100.0, 100.38], [100.0, 110.0]])
        result = attach_overlap(
            events,
            real_matrix=real,
            base_atr=np.array([1.0, 1.0]),
            zone="tight_base_atr",
            source_index=0,
        )
        self.assertTrue(bool(result.loc[0, "overlap_source_current_level"]))
        self.assertEqual(int(result.loc[0, "overlap_other_real_level_count"]), 1)
        self.assertFalse(bool(result.loc[1, "overlap_source_current_level"]))
        self.assertEqual(int(result.loc[1, "overlap_other_real_level_count"]), 1)

    def test_source_availability_uses_candle_close(self) -> None:
        self.assertEqual(timeframe_delta("1h"), pd.Timedelta(hours=1))
        self.assertEqual(timeframe_delta("4h"), pd.Timedelta(hours=4))
        self.assertEqual(timeframe_delta("1d"), pd.Timedelta(days=1))

    def test_datetime_sources_normalize_to_one_merge_precision(self) -> None:
        milliseconds = pd.Series(np.array(["2024-01-01T00:00:00"], dtype="datetime64[ms]"))
        microseconds = pd.Series(np.array(["2024-01-01T00:00:00"], dtype="datetime64[us]"))
        left = normalize_dates(milliseconds)
        right = normalize_dates(microseconds)
        self.assertEqual(str(left.dtype), "datetime64[ns, UTC]")
        self.assertEqual(left.dtype, right.dtype)

    def test_candidate_level_inventory_excludes_ids_scores_and_counts(self) -> None:
        frame = pd.DataFrame(
            {
                "tlv2_support_line_rank0": [10.0],
                "tlv2_support_line_id_rank0": [123.0],
                "tlv2_support_pivot_count_rank0": [3.0],
                "tlv2_support_score_rank0": [0.8],
                "pg2_slot_1_upper": [11.0],
                "pg2_slot_1_upper_pivots": [4.0],
                "generic_round_step": [5.0],
                "generic_round_above": [15.0],
                "pc_flag_upper": [12.0],
                "pc_flag_breakout_above_upper": [1.0],
                "pc_flag_confirmation_level": [12.5],
            }
        )
        selected = candidate_level_columns(frame)
        self.assertEqual(
            selected,
            [
                "tlv2_support_line_rank0",
                "pg2_slot_1_upper",
                "generic_round_above",
                "pc_flag_upper",
                "pc_flag_confirmation_level",
            ],
        )

    def test_sparse_level_specs_use_pattern_activity_and_score_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "pc_flag_upper": [12.0],
                "pc_flag_breakout_above_upper": [1.0],
                "pc_flag_pattern_present": [1.0],
                "pc_flag_pattern_confirmed": [0.0],
                "pc_flag_indicator_score": [0.75],
                "pc_flag_direction": [1],
            }
        )
        specs = level_specs(frame, ("g0b3",))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].column, "pc_flag_upper")
        self.assertEqual(specs[0].active_columns, ("pc_flag_pattern_present",))
        self.assertEqual(specs[0].score_column, "pc_flag_indicator_score")
        self.assertEqual(specs[0].identity_column, "pc_flag_direction")
        self.assertIn("pc_flag_pattern_confirmed", specs[0].attribute_columns)

    def test_nice_number_steps_are_scale_adaptive(self) -> None:
        target = pd.Series([0.012, 0.08, 1.2, 38.0, 760.0])
        actual = nice_number_step(target).to_numpy()
        expected = np.array([0.01, 0.1, 1.0, 50.0, 1000.0])
        np.testing.assert_allclose(actual, expected)

    def test_generic_levels_do_not_change_when_future_ohlcv_changes(self) -> None:
        rows = 900
        date = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
        close = pd.Series(np.linspace(100.0, 180.0, rows))
        frame = pd.DataFrame(
            {
                "date": date,
                "open": close - 0.1,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(1000.0, 2000.0, rows),
            }
        )
        changed = frame.copy()
        changed.loc[850:, ["open", "high", "low", "close", "volume"]] *= 10.0
        before = generic_level_frame(frame)
        after = generic_level_frame(changed)
        pd.testing.assert_frame_equal(before.iloc[:850], after.iloc[:850])

    def test_projected_line_advances_only_after_source_candle_is_available(self) -> None:
        merged = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"]),
                "source_open": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]),
                "line": [10.0, 10.0],
                "slope": [1.0, 1.0],
            }
        )
        spec = LevelSpec(
            name="line",
            family="test",
            batch="g0b1",
            column="line",
            representation="projected",
            slope_column="slope",
        )
        np.testing.assert_allclose(resolved_level_values(merged, spec, "1h"), [11.0, 12.0])

    def test_episode_start_requires_six_clear_bars_unless_level_replaced(self) -> None:
        condition = np.array([True, True, False, False, False, False, False, False, True, True])
        level = np.full(10, 100.0)
        width = np.full(10, 1.0)
        starts = episode_start_mask(condition, level, width, cooldown=6)
        self.assertEqual(np.flatnonzero(starts).tolist(), [0, 8])
        level[1] = 103.0
        starts = episode_start_mask(condition, level, width, cooldown=6)
        self.assertEqual(np.flatnonzero(starts).tolist(), [0, 1, 8])

    def test_reaction_paths_describe_away_and_through_without_direction_target(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        merged = pd.DataFrame(
            {
                "date": dates,
                "period": ["development"] * 4,
                "open": [9.0, 10.0, 10.0, 10.0],
                "high": [10.2, 11.0, 12.0, 10.0],
                "low": [9.8, 9.0, 8.0, 10.0],
                "close": [10.0, 10.5, 9.0, 10.0],
                "volume": [100.0] * 4,
                "candle_pressure": [0.0] * 4,
                "base_atr": [1.0] * 4,
                "pre_close": [9.0, 10.0, 10.5, 9.0],
                "pre_range_median_24": [1.0] * 4,
                "pre_volume_median_24": [100.0] * 4,
                "pre_pressure_mean_24": [0.0] * 4,
                "volatility_band": [2] * 4,
                "contact_range_band": [2] * 4,
            }
        )
        paths = future_path_matrices(merged, 2)
        spec = LevelSpec("test", "test", "g0b1", "unused")
        event = event_frame(
            merged=merged,
            paths=paths,
            indexes=np.array([0]),
            levels=np.array([10.0]),
            widths=np.array([0.1]),
            spec=spec,
            pair="BTC/USDT:USDT",
            timeframe="1h",
            control="actual",
            zone_method="tight_base_atr",
            source_available=pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"])),
            source_open=pd.Series(pd.to_datetime(["2023-12-31T23:00:00Z"])),
            horizons=(1, 2),
            match_tier=None,
        )
        self.assertEqual(event.loc[0, "approach_state"], "from_below")
        self.assertAlmostEqual(event.loc[0, "through_excursion_atr_h2"], 2.0)
        self.assertAlmostEqual(event.loc[0, "away_excursion_atr_h2"], 2.0)
        self.assertAlmostEqual(event.loc[0, "abs_excursion_atr_h2"], 2.0)

    def test_price_shift_control_does_not_jump_sides_every_candle(self) -> None:
        rows = 12
        level = np.full(rows, 100.0)
        atr = np.linspace(1.0, 2.0, rows)
        dates = pd.Series(pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"))
        transformed = transform_control(
            control="price_shift",
            level=level,
            valid=np.ones(rows, dtype=bool),
            native_width=np.full(rows, np.nan),
            source_available=dates,
            source_open=dates,
            base_atr=atr,
            stable_key="stable-test-level",
        )
        displacement = transformed["level"] - level
        self.assertTrue(np.all(np.sign(displacement) == np.sign(displacement[0])))
        np.testing.assert_allclose(np.abs(displacement), 2.0 * atr)

    def test_review_comparison_keeps_behaviour_delta_not_profit(self) -> None:
        base = {
            "pair": "BTC/USDT:USDT",
            "source_timeframe": "1h",
            "batch": "g0b1",
            "level_family": "test_level",
            "level_name": "test",
            "representation": "settled",
            "zone_method": "standard_base_atr",
            "period": "development",
            "approach_state": "all_approaches",
            "horizon_hours": 4,
            "event_count": 100,
            "unique_event_days": 80,
            "largest_day_event_fraction": 0.04,
        }
        actual = {**base, "control": "actual"}
        matched = {**base, "control": "matched_random_time"}
        for column in METRIC_COLUMNS:
            actual[column] = 2.0
            matched[column] = 1.5
        summary = pd.DataFrame([actual, matched])
        comparison = comparison_from_summary(summary)
        matched_row = comparison.loc[comparison["control"] == "matched_random_time"].iloc[0]
        self.assertAlmostEqual(matched_row["abs_excursion_mean_delta"], 0.5)
        self.assertNotIn("profit", " ".join(comparison.columns).lower())
        self.assertTrue(set(KEY_COLUMNS).issubset(comparison.columns))

    def test_review_adds_primary_and_broad_cohort_lenses(self) -> None:
        manifest = load_manifest(MANIFEST)
        frame = pd.DataFrame(
            {
                "pair": ["BTC/USDT:USDT", "SOL/USDT:USDT"],
                "value": [1.0, 2.0],
            }
        )
        expanded = add_cohort_lenses(frame, manifest)
        self.assertEqual(len(expanded), 5)
        self.assertEqual(
            set(expanded.loc[expanded["pair"] == "BTC/USDT:USDT", "cohort"]),
            {"btc_distinct", "all_pairs"},
        )
        self.assertEqual(
            set(expanded.loc[expanded["pair"] == "SOL/USDT:USDT", "cohort"]),
            {"large_smart_contract", "all_pairs", "all_non_btc"},
        )

    def test_coverage_separates_level_exposure_from_contact_frequency(self) -> None:
        task = AtlasTask(
            pair="BTC/USDT:USDT",
            timeframe="1h",
            cache_families=("generic",),
            level_batches=("g0b2",),
            zone_methods=("standard_base_atr",),
            controls=("actual",),
            manifest_path=str(MANIFEST),
            overwrite=False,
        )
        row = coverage_row(
            task=task,
            scope="level_spec",
            batch="g0b2",
            level_family="test",
            level_name="test_level",
            representation="settled",
            zone_method="standard_base_atr",
            period="development",
            period_mask=np.array([True, True, True, True]),
            eligible=np.array([True, True, False, True]),
            contact=np.array([True, True, False, False]),
            starts=np.array([True, False, False, False]),
            available_count=np.array([1, 1, 0, 1], dtype=np.int16),
            contact_count=np.array([1, 1, 0, 0], dtype=np.int16),
            member_level_specs=1,
            width_atr=np.array([0.1, 0.2, np.nan, 0.4]),
        )
        self.assertEqual(row["eligible_level_bars"], 3)
        self.assertEqual(row["contact_bars"], 2)
        self.assertEqual(row["episodes"], 1)
        self.assertAlmostEqual(row["eligible_level_fraction"], 0.75)
        self.assertAlmostEqual(row["contact_bar_fraction_of_eligible"], 2.0 / 3.0)
        self.assertAlmostEqual(row["median_zone_half_width_atr"], 0.2)
        self.assertFalse(row["direction_prediction"])
        self.assertFalse(row["profit_optimization"])

    def test_aggregation_does_not_count_missing_control_comparisons(self) -> None:
        rows = pd.DataFrame(
            {
                "cohort": ["all_pairs", "all_pairs"],
                "pair": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "period": ["development", "development"],
                "control": ["matched_random_time", "matched_random_time"],
                "actual_n": [100.0, 100.0],
                "control_n": [95.0, np.nan],
                "actual_largest_day_fraction": [0.02, 0.02],
                **{column: [0.1, np.nan] for column in PRIMARY_DELTA_COLUMNS},
            }
        )
        result = aggregate_comparisons(rows, ["cohort", "period", "control"])
        self.assertEqual(result.loc[0, "cell_count"], 1)
        self.assertEqual(result.loc[0, "pair_count"], 1)

    def test_control_screen_reports_higher_and_lower_behaviour(self) -> None:
        row = {
            "cohort": "all_pairs",
            "source_timeframe": "4h",
            "batch": "g0b2",
            "level_family": "test",
            "level_name": "test_level",
            "representation": "settled",
            "zone_method": "standard_base_atr",
            "approach_state": "all_approaches",
            "horizon_hours": 4,
            "control": "matched_random_time",
            "cell_count": 3,
            "pair_count": 3,
            "period_count": 3,
            "actual_events": 300,
            "largest_cell_event_fraction": 0.4,
        }
        for column in PRIMARY_DELTA_COLUMNS:
            row[f"{column}_median"] = 0.1
            row[f"{column}_positive_fraction"] = 1.0
        row["dwell_fraction_mean_delta_median"] = -0.1
        screen = build_control_screen(pd.DataFrame([row]))
        self.assertEqual(
            screen.loc[0, "control_sign_summary__abs_excursion_mean"],
            "higher_than_all_available",
        )
        self.assertEqual(
            screen.loc[0, "control_sign_summary__dwell_fraction_mean"],
            "lower_than_all_available",
        )

    def test_timing_review_reports_reach_rate_and_conditional_speed(self) -> None:
        base = {
            "pair": "BTC/USDT:USDT",
            "source_timeframe": "4h",
            "batch": "g0b2",
            "level_family": "test",
            "level_name": "test_level",
            "representation": "settled",
            "zone_method": "standard_base_atr",
            "period": "development",
            "approach_state": "from_below",
        }
        rows = []
        for control, values in (
            ("actual", [1.0, np.nan]),
            ("matched_random_time", [2.0, 3.0]),
        ):
            for value in values:
                row = {**base, "control": control}
                for column in (
                    "time_to_abs_0_5atr",
                    "time_to_abs_1_0atr",
                    "time_to_away_0_5atr",
                    "time_to_away_1_0atr",
                    "time_to_through_0_5atr",
                    "time_to_through_1_0atr",
                ):
                    row[column] = value
                rows.append(row)
        comparison = timing_comparison_from_events(pd.DataFrame(rows))
        target = comparison.loc[comparison["reaction_threshold"] == "abs_0_5atr"].iloc[0]
        self.assertAlmostEqual(target["actual_reach_rate"], 0.5)
        self.assertAlmostEqual(target["control_reach_rate"], 1.0)
        self.assertAlmostEqual(target["reach_rate_delta"], -0.5)
        self.assertAlmostEqual(target["conditional_median_time_delta_hours"], -1.5)

    def test_calibration_inventory_excludes_categorical_numeric_codes(self) -> None:
        columns = calibratable_attribute_columns(
            {
                "level_score",
                "attr_vp_state",
                "attr_tlv2_support_pivot_count_rank0",
                "attr_pg2_slot_1_direction",
                "attr_pg2_slot_1_width_atr",
                "attr_pg2_slot_1_confirmation_tier",
                "attr_generic_round_step",
            }
        )
        self.assertIn("level_score", columns)
        self.assertIn("attr_tlv2_support_pivot_count_rank0", columns)
        self.assertIn("attr_pg2_slot_1_width_atr", columns)
        self.assertIn("attr_pg2_slot_1_confirmation_tier", columns)
        self.assertNotIn("attr_vp_state", columns)
        self.assertNotIn("attr_pg2_slot_1_direction", columns)
        self.assertNotIn("attr_generic_round_step", columns)

    def test_attribute_calibration_beats_state_matched_shuffle_for_ordered_score(
        self,
    ) -> None:
        rows = 60
        score = np.linspace(0.01, 1.0, rows)
        events = pd.DataFrame(
            {
                "pair": "BTC/USDT:USDT",
                "source_timeframe": "4h",
                "batch": "g0b1",
                "level_family": "test_family",
                "level_name": "test_level",
                "representation": "settled",
                "control": "actual",
                "zone_method": "standard_base_atr",
                "period": "development",
                "approach_state": "from_below",
                "event_time": pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC"),
                "base_index": np.arange(rows),
                "volatility_band": np.arange(rows) % 3,
                "contact_range_band": (np.arange(rows) // 3) % 2,
                "level_score": score,
                "abs_excursion_atr_h4": 0.5 + 2.0 * score,
            }
        )
        result = attribute_calibration_cells(
            events,
            attribute_columns=("level_score",),
            targets=(("abs_excursion_atr_h4", "absolute_price_excursion_atr", 4),),
        )
        pooled = result.loc[result["approach_state"] == "all_approaches"].iloc[0]
        self.assertAlmostEqual(pooled["observed_spearman"], 1.0)
        self.assertGreater(
            abs(pooled["observed_spearman"]), abs(pooled["shuffled_spearman"])
        )
        self.assertGreater(pooled["top_bottom_difference"], 0.0)
        self.assertNotIn("profit", " ".join(result.columns).lower())
        self.assertNotIn("direction_prediction", result.columns)

    def test_freqai_future_helpers_use_exact_later_window(self) -> None:
        values = pd.Series([10.0, 2.0, 8.0, 4.0, 20.0])
        expected_max = pd.Series([8.0, 8.0, 20.0, np.nan, np.nan])
        expected_min = pd.Series([2.0, 4.0, 4.0, np.nan, np.nan])
        expected_mean = pd.Series([5.0, 6.0, 12.0, np.nan, np.nan])
        pd.testing.assert_series_equal(_future_extreme(values, 2, "max"), expected_max)
        pd.testing.assert_series_equal(_future_extreme(values, 2, "min"), expected_min)
        pd.testing.assert_series_equal(_future_mean(values, 2), expected_mean)

    def test_freqai_targets_are_nondirectional_and_leave_future_tail_unknown(self) -> None:
        rows = 100
        close = pd.Series(np.linspace(100.0, 120.0, rows))
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(1000.0, 1200.0, rows),
            }
        )
        labelled = build_market_reaction_targets(frame)
        target_columns = [column for column in labelled if column.startswith("&-")]
        self.assertEqual(set(target_columns), set(TARGETS))
        self.assertNotIn("direction", " ".join(target_columns).lower())
        for horizon in TARGET_HORIZONS:
            columns = [column for column in target_columns if f"_{horizon}h" in column]
            self.assertTrue(columns)
            self.assertTrue(labelled.loc[rows - horizon :, columns].isna().all().all())

    def test_freqai_profile_controls_change_levels_not_training_settings(self) -> None:
        config_path = REPO_ROOT / "user_data" / "configs" / (
            "config_market_reaction_zone_freqai.example.json"
        )
        base = json.loads(config_path.read_text(encoding="utf-8"))
        native = profile_config(
            base,
            identifier="native",
            pairs=("BTC/USDT:USDT",),
            level_shift_hours=0,
            train_days=365,
            backtest_days=180,
            technical_smoke=False,
        )
        placebo = profile_config(
            base,
            identifier="placebo",
            pairs=("BTC/USDT:USDT",),
            level_shift_hours=168,
            train_days=365,
            backtest_days=180,
            technical_smoke=False,
        )
        self.assertEqual(
            native["freqai"]["model_training_parameters"],
            placebo["freqai"]["model_training_parameters"],
        )
        self.assertEqual(
            native["freqai"]["data_split_parameters"],
            placebo["freqai"]["data_split_parameters"],
        )
        self.assertEqual(
            native["market_reaction_zone_freqai"]["level_feature_shift_hours"], 0
        )
        self.assertEqual(
            placebo["market_reaction_zone_freqai"]["level_feature_shift_hours"], 168
        )
        self.assertFalse(native["freqai"]["save_backtest_models"])
        self.assertEqual(
            set(PROFILES),
            {
                "market_state",
                "level_only_native",
                "combined_native",
                "combined_native_placebo",
                "combined_mtf",
                "combined_mtf_placebo",
            },
        )

    def test_freqai_scoring_uses_identical_prediction_keys(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        left = pd.DataFrame(
            {"pair": "BTC/USDT:USDT", "date": dates, TARGETS[0]: [1.0, 2.0, 3.0]}
        )
        right = pd.DataFrame(
            {
                "pair": "BTC/USDT:USDT",
                "date": dates[1:],
                TARGETS[0]: [2.0, 3.0],
            }
        )
        common, audit = common_prediction_keys(
            {"market_state": left, "combined_native": right},
            ("BTC/USDT:USDT",),
        )
        self.assertEqual(common["date"].tolist(), dates[1:].tolist())
        self.assertTrue(audit["common_rows"].eq(2).all())
        self.assertEqual(audit["rows_excluded_for_fair_comparison"].tolist(), [1, 0])

    def test_freqai_target_metadata_covers_every_frozen_target(self) -> None:
        metadata = [target_metadata(target) for target in TARGETS]
        self.assertEqual(len(metadata), 4 * len(TARGET_HORIZONS))
        self.assertEqual({horizon for _, horizon in metadata}, set(TARGET_HORIZONS))
        self.assertEqual(SELECTED_MTF_LEVEL_TIMEFRAMES, ("1h", "4h", "8h", "1d"))

    def test_freqai_resume_rejects_changed_run_settings(self) -> None:
        worker = REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / (
            "Scripts/python.exe"
        )
        manifest = {
            "run_id": "fixed-run",
            "commands": [
                {
                    "profile_id": "market_state",
                    "command": [str(worker), "-m", "freqtrade"],
                }
            ],
            "pairs": ["BTC/USDT:USDT"],
            "timerange": "20250101-20250701",
            "train_period_days": 90,
            "backtest_period_days": 90,
            "technical_smoke_not_evidence": True,
        }
        validate_manifest_request(
            manifest,
            profiles=("market_state",),
            pairs=("BTC/USDT:USDT",),
            timerange="20250101-20250701",
            train_days=90,
            backtest_days=90,
            technical_smoke=True,
            python_exe=worker,
        )
        with self.assertRaisesRegex(ValueError, "incompatible settings"):
            validate_manifest_request(
                manifest,
                profiles=("market_state",),
                pairs=("BTC/USDT:USDT",),
                timerange="20240101-20250701",
                train_days=90,
                backtest_days=90,
                technical_smoke=True,
                python_exe=worker,
            )

    def test_freqai_market_baseline_excludes_tested_level_math(self) -> None:
        rows = 900
        close = pd.Series(np.linspace(100.0, 180.0, rows))
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC"),
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.linspace(1000.0, 2000.0, rows),
            }
        )
        strategy = MarketReactionZoneMarketStateFreqAIResearchStrategy(config={})
        features = strategy.feature_engineering_expand_basic(frame, {})
        feature_names = {column for column in features if column.startswith("%-")}
        self.assertTrue({"%-rsi_14", "%-macd_12_26_atr", "%-adx_14"}.issubset(feature_names))
        forbidden_fragments = (
            "range_position_",
            "distance_prior_high_",
            "distance_prior_low_",
            "bollinger_",
        )
        self.assertFalse(
            any(fragment in name for fragment in forbidden_fragments for name in feature_names)
        )

    def test_cluster_representation_modes_never_double_count_one_physical_line(self) -> None:
        common = {
            "name": "support_rank0",
            "family": "tlv2_ranked",
            "batch": "g0b1",
            "column": "tlv2_support_line_rank0",
        }
        specs = [
            LevelSpec(**common, representation="projected", slope_column="support_slope"),
            LevelSpec(**common, representation="held"),
            LevelSpec(
                name="poc",
                family="volume_profile_settled",
                batch="g0b1",
                column="vp_poc",
            ),
        ]
        projected = select_representation_specs(specs, representation_mode="projected")
        held = select_representation_specs(specs, representation_mode="held_sensitivity")
        self.assertEqual([spec.representation for spec in projected], ["projected", "settled"])
        self.assertEqual([spec.representation for spec in held], ["held", "settled"])

    def test_connected_cluster_classes_preserve_family_and_timeframe_structure(self) -> None:
        components = [
            ClusterComponent(
                index=0,
                key="1h|generic_prior_range|rolling_high_24|settled",
                source_key="1h|generic_prior_range|generic_rolling_high_24",
                family="generic_prior_range",
                timeframe="1h",
                name="rolling_high_24",
                column="generic_rolling_high_24",
                representation="settled",
                interpretation="activity_transit",
                dynamic_representation=False,
                mechanism="rolling_price_extreme",
            ),
            ClusterComponent(
                index=1,
                key="4h|volume_profile_settled|poc|settled",
                source_key="4h|volume_profile_settled|vp_poc",
                family="volume_profile_settled",
                timeframe="4h",
                name="poc",
                column="vp_poc",
                representation="settled",
                interpretation="acceptance_stickiness",
                dynamic_representation=False,
                mechanism="volume_profile",
            ),
        ]
        clusters = connected_level_clusters(
            values=np.array([100.0, 100.4]),
            atr=1.0,
            components=components,
            scale="standard",
            require_dynamic=False,
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["cluster_class"], "different_family_cross_timeframe")
        self.assertTrue(clusters[0]["interpretation_conflict"])
        self.assertTrue(clusters[0]["independent_mechanism_cluster"])
        self.assertEqual(clusters[0]["mechanism_count"], 2)
        self.assertEqual(classify_cluster(1, 1), "same_family_same_timeframe")
        self.assertEqual(classify_cluster(2, 1), "different_family_same_timeframe")
        self.assertEqual(classify_cluster(1, 2), "same_family_cross_timeframe")

    def test_held_sensitivity_requires_a_dynamic_line_component(self) -> None:
        components = [
            ClusterComponent(
                index=index,
                key=f"1h|family|level_{index}|settled",
                source_key=f"1h|family|column_{index}",
                family="family",
                timeframe="1h",
                name=f"level_{index}",
                column=f"column_{index}",
                representation="settled",
                interpretation="unclassified",
                dynamic_representation=False,
            )
            for index in range(2)
        ]
        self.assertEqual(
            connected_level_clusters(
                values=np.array([100.0, 100.1]),
                atr=1.0,
                components=components,
                scale="tight",
                require_dynamic=True,
            ),
            [],
        )

    def test_nested_cluster_and_episode_deduplication_are_physical_not_name_based(self) -> None:
        tight = {
            "member_index_signature": "0;1",
            "nested_relationship": "none",
            "nested_scale_count": 1,
        }
        wide = {
            "member_index_signature": "0;1;2",
            "nested_relationship": "none",
            "nested_scale_count": 1,
        }
        by_scale = {"tight": [tight], "wide": [wide]}
        add_nested_relationships(by_scale)
        self.assertEqual(tight["nested_relationship"], "narrow_core_of_wider_cluster")
        self.assertEqual(wide["nested_relationship"], "wide_envelope_with_narrow_core")

        rows = [
            {
                "representation_mode": "projected",
                "cluster_scale": "tight",
                "row_index": 10,
                "lower": 99.0,
                "upper": 101.0,
                "center": 100.0,
            },
            {
                "representation_mode": "projected",
                "cluster_scale": "tight",
                "row_index": 12,
                "lower": 100.5,
                "upper": 102.0,
                "center": 101.25,
            },
            {
                "representation_mode": "projected",
                "cluster_scale": "tight",
                "row_index": 19,
                "lower": 100.5,
                "upper": 102.0,
                "center": 101.25,
            },
        ]
        selected = dedupe_contact_candidates(rows)
        self.assertEqual([row["row_index"] for row in selected], [10, 19])

    def test_family_ablation_rechecks_connected_intervals(self) -> None:
        self.assertEqual(
            largest_connected_interval_component(
                np.array([100.0, 101.0, 102.0]),
                np.array([0.6, 0.6, 0.6]),
            ),
            3,
        )
        self.assertEqual(
            largest_connected_interval_component(
                np.array([100.0, 102.0]),
                np.array([0.6, 0.6]),
            ),
            1,
        )

    def test_cluster_contact_separates_nearby_components_from_touched_components(self) -> None:
        components = [
            ClusterComponent(
                index=index,
                key=f"1h|family_{index}|level|settled",
                source_key=f"1h|family_{index}|column",
                family=f"family_{index}",
                timeframe="1h" if index == 0 else "4h",
                name="level",
                column="column",
                representation="settled",
                interpretation=(
                    "activity_transit" if index == 0 else "acceptance_stickiness"
                ),
                dynamic_representation=False,
            )
            for index in range(2)
        ]
        metadata = contacted_component_metadata(
            cluster={"member_indexes_list": (0, 1)},
            row_values=np.array([100.0, 100.4]),
            components=components,
            atr=1.0,
            candle_high=100.15,
            candle_low=99.95,
            scale="tight",
            price_shift=0.0,
        )
        self.assertEqual(metadata["contacted_component_count"], 1)
        self.assertEqual(metadata["contact_structure_class"], "single_component_contact")
        self.assertFalse(metadata["contacted_interpretation_conflict"])
        self.assertEqual(
            metadata["contacted_component_keys"],
            "1h|family_0|level|settled",
        )
        self.assertEqual(metadata["contacted_component_families"], "family_0")
        self.assertEqual(metadata["contacted_component_timeframes"], "1h")
        self.assertEqual(metadata["contacted_mechanism_count"], 1)
        self.assertFalse(metadata["contacted_independent_mechanisms"])

    def test_cluster_review_pressure_and_timing_transforms_are_non_directional(self) -> None:
        pressure = metric_spec(
            "pressure_change_h4", "pressure_change_magnitude", 4, absolute=True
        )
        np.testing.assert_allclose(
            transformed_metric(pd.Series([-2.0, 1.0]), pressure), [2.0, 1.0]
        )
        timing = metric_spec(
            "time_to_abs_1_0atr",
            "reaction_timing",
            48,
            lower_is_stronger=True,
            missing_fill=49.0,
        )
        np.testing.assert_allclose(
            transformed_metric(pd.Series([3.0, np.nan]), timing), [3.0, 49.0]
        )
        self.assertEqual(timing["orientation"], -1.0)

    def test_positive_fraction_ignores_unscored_values(self) -> None:
        self.assertEqual(positive_fraction(pd.Series([1.0, -1.0, np.nan])), 0.5)

    def test_cluster_source_age_uses_causally_available_component_updates(self) -> None:
        availability = {
            "1h": pd.Series(pd.to_datetime(["2024-01-01T09:00:00Z"])),
            "1d": pd.Series(pd.to_datetime(["2023-12-31T00:00:00Z"])),
        }
        result = source_availability_metadata(
            0,
            ("1h", "1d"),
            availability,
            event_time=pd.Timestamp("2024-01-01T10:00:00Z"),
        )
        self.assertEqual(result["source_available_at"], pd.Timestamp("2024-01-01T09:00:00Z"))
        self.assertEqual(result["oldest_source_update_age_hours"], 34.0)
        self.assertEqual(result["oldest_source_update_age_band"], "gt_24h")


if __name__ == "__main__":
    unittest.main()
