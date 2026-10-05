from __future__ import annotations

# Keep numerical libraries single-threaded. This diagnostic is intentionally one-worker.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    timeframe_hours,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_one_minute_replay import (  # noqa: E501
    ARTIFACT_ROOT,
    COHORT_SOURCES,
    FEATURE_WARMUP_HOURS,
    FUTURES_DATA_DIR,
    REPORT_ROOT,
    WINDOW_HOURS,
    artifact_record,
    merge_overlapping_acquisition_intervals,
)


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_FREEZE_RUN_ID = "g3g_thin_lvn_1m_freeze_20260814a"
ORDERBOOK_PATH = (
    REPO_ROOT
    / "user_data"
    / "orderbook_data"
    / "historical_bybit"
    / "features"
    / "orderbook_trader_state_1h_bybit_linear_latest.parquet"
)
TECHNICAL_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "8h")
TIMEFRAME_PREFIX = {
    "1m": "tf_1m",
    "5m": "tf_5m",
    "15m": "tf_15m",
    "1h": "tf_1h",
    "4h": "tf_4h",
    "8h": "tf_8h",
}
PANDAS_TIMEFRAME = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "8h": "8h",
}
CHECKPOINT_MINUTES = {
    "1h": (1, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720),
    "4h": (1, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880),
    "8h": (1, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 5760),
    "1d": (1, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 5760, 10080),
}
REACTION_HORIZON_MINUTES = 60
CONTROL_HORIZON_MINUTES = 120
CONTROL_EVENT_SEPARATION_MINUTES = 240
REACTION_EXCURSION_HALF_WIDTHS = 1.0
REACTION_VOLUME_RATIO_MINIMUM = 1.5
CONTROL_MEAN_DISTANCE_LIMIT = 0.75
CONTROL_MAX_DISTANCE_LIMIT = 2.0
BOUNDARY_ACTIVITY_RATIO_LIMIT = 1.5
BOUNDARY_ZONE_FRACTION_LIMIT = 0.20
MINIMUM_SCREEN_ROWS = 5
MINIMUM_QUEUE_ROWS_PER_COHORT = 12

MATCH_FEATURES = (
    "match_return_15m",
    "match_return_60m",
    "match_return_240m",
    "match_realized_volatility_60m",
    "match_atr14_pct",
    "match_volume_ratio_20m",
    "match_pressure_20m",
    "match_range_position_240m",
)

PRIMARY_OUTCOMES = (
    "excursion_strength_60_half_widths",
    "volume_ratio_post_pre_60m",
    "through_excursion_60_half_widths",
    "away_excursion_60_half_widths",
    "pressure_change_toward_away_60m",
    "realized_volatility_ratio_60m",
    "zone_overlap_fraction_60m",
    "first_direction_numeric",
    "first_path_numeric",
    "first_zone_resolution_numeric",
)


TECHNICAL_DEFINITIONS = {
    "return_1bar": "close change over one completed bar",
    "return_3bar": "close change over three completed bars",
    "return_12bar": "close change over twelve completed bars",
    "realized_volatility_14bar": "standard deviation of the last fourteen completed-bar returns",
    "volume_ratio_20bar": "last completed-bar volume divided by its trailing twenty-bar mean",
    "volume_z_60bar": "last completed-bar volume standardized against the trailing sixty bars",
    "pressure_5bar": "five-bar volume-weighted candle-body pressure from minus one to plus one",
    "pressure_20bar": "twenty-bar volume-weighted candle-body pressure from minus one to plus one",
    "pressure_60bar": "sixty-bar volume-weighted candle-body pressure from minus one to plus one",
    "rsi14": "fourteen-bar relative-strength index",
    "bollinger_position20": "close position between the twenty-bar lower and upper Bollinger bands",
    "bollinger_width20": "twenty-bar Bollinger-band width divided by its centre line",
    "macd_histogram_pct": "MACD(12,26,9) histogram divided by price",
    "sma20_distance_pct": "close distance from the twenty-bar simple moving average",
    "sma50_distance_pct": "close distance from the fifty-bar simple moving average",
    "sma200_distance_pct": "close distance from the two-hundred-bar simple moving average",
    "sma20_minus_sma50_pct": "twenty-bar SMA minus fifty-bar SMA, divided by price",
    "sma50_minus_sma200_pct": "fifty-bar SMA minus two-hundred-bar SMA, divided by price",
    "ema12_distance_pct": "close distance from the twelve-bar exponential moving average",
    "ema26_distance_pct": "close distance from the twenty-six-bar exponential moving average",
    "ema50_distance_pct": "close distance from the fifty-bar exponential moving average",
    "ema12_minus_ema26_pct": "twelve-bar EMA minus twenty-six-bar EMA, divided by price",
    "atr14_pct": "fourteen-bar average true range divided by price",
    "adx14": "fourteen-bar average directional index",
    "di_spread14": "positive directional index minus negative directional index",
    "cmf20": "twenty-bar Chaikin money flow",
    "obv_trend20": "twenty-bar on-balance-volume change divided by trailing volume",
    "range_position20": "close position inside the preceding twenty-bar high-low range",
    "range_position50": "close position inside the preceding fifty-bar high-low range",
    "bars_since_rsi_cross_above_50": "completed bars since RSI crossed upward through 50",
    "bars_since_rsi_cross_below_50": "completed bars since RSI crossed downward through 50",
    "bars_since_rsi_cross_above_30": "completed bars since RSI recovered above 30",
    "bars_since_rsi_cross_below_70": "completed bars since RSI fell below 70",
    "bars_since_close_cross_above_bollinger_upper": "completed bars since close crossed above the upper Bollinger band",
    "bars_since_close_cross_below_bollinger_upper": "completed bars since close crossed back below the upper Bollinger band",
    "bars_since_close_cross_below_bollinger_lower": "completed bars since close crossed below the lower Bollinger band",
    "bars_since_close_cross_above_bollinger_lower": "completed bars since close crossed back above the lower Bollinger band",
    "bars_since_macd_bull_cross": "completed bars since the MACD line crossed above its signal",
    "bars_since_macd_bear_cross": "completed bars since the MACD line crossed below its signal",
    "bars_since_close_cross_above_sma20": "completed bars since close crossed above SMA(20)",
    "bars_since_close_cross_below_sma20": "completed bars since close crossed below SMA(20)",
    "bars_since_close_cross_above_sma50": "completed bars since close crossed above SMA(50)",
    "bars_since_close_cross_below_sma50": "completed bars since close crossed below SMA(50)",
    "bars_since_close_cross_above_sma200": "completed bars since close crossed above SMA(200)",
    "bars_since_close_cross_below_sma200": "completed bars since close crossed below SMA(200)",
    "bars_since_sma20_cross_above_sma50": "completed bars since SMA(20) crossed above SMA(50)",
    "bars_since_sma20_cross_below_sma50": "completed bars since SMA(20) crossed below SMA(50)",
    "bars_since_sma50_cross_above_sma200": "completed bars since SMA(50) crossed above SMA(200)",
    "bars_since_sma50_cross_below_sma200": "completed bars since SMA(50) crossed below SMA(200)",
    "bars_since_ema12_cross_above_ema26": "completed bars since EMA(12) crossed above EMA(26)",
    "bars_since_ema12_cross_below_ema26": "completed bars since EMA(12) crossed below EMA(26)",
    "bars_since_di_cross_bullish": "completed bars since positive DI crossed above negative DI",
    "bars_since_di_cross_bearish": "completed bars since positive DI crossed below negative DI",
}


@dataclass(frozen=True)
class PairSurfaces:
    minute: DataFrame
    match: DataFrame
    technical: dict[str, DataFrame]
    base: DataFrame
    aligned_levels: list[Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-worker, direct one-minute replay for the already-frozen G3G "
            "thin-LVN sample. It records continuous paths, causal technical state, "
            "cross recency, exact cluster composition, and matched no-level controls."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freeze-run-id", default=DEFAULT_FREEZE_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expanded-boundaries",
        action="store_true",
        help="Use the single permitted all-episode doubling of pre/post context windows.",
    )
    args = parser.parse_args(argv)
    return run_analysis(
        run_id=args.run_id,
        freeze_run_id=args.freeze_run_id,
        overwrite=args.overwrite,
        expanded_boundaries=args.expanded_boundaries,
    )


def run_analysis(
    *, run_id: str, freeze_run_id: str, overwrite: bool, expanded_boundaries: bool
) -> int:
    freeze_dir = REPORT_ROOT / freeze_run_id
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g3g_analysis_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"Run already exists: {record_path}")
    freeze_record_path = freeze_dir / "g3g_freeze_record.json"
    coverage_record_path = freeze_dir / "g3g_one_minute_coverage_record.json"
    sample_path = freeze_dir / "g3g_frozen_episode_sample.csv"
    components_path = freeze_dir / "g3g_frozen_sample_components.csv"
    freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    coverage_record = json.loads(coverage_record_path.read_text(encoding="utf-8"))
    if freeze_record.get("status") != "frozen_before_one_minute_paths":
        raise ValueError("The requested G3G parent sample is not frozen.")
    if coverage_record.get("status") != "passed":
        raise ValueError("The requested G3G one-minute coverage did not pass audit.")
    for key, path in (
        ("frozen_sample_csv", sample_path),
        ("frozen_sample_components_csv", components_path),
    ):
        frozen_artifact = freeze_record["artifacts"][key]
        if sha256_file(path) != str(frozen_artifact["sha256"]):
            raise ValueError(f"Frozen input changed: {path}")

    request = analysis_request_contract(
        freeze_run_id=freeze_run_id,
        freeze_record_path=freeze_record_path,
        coverage_record_path=coverage_record_path,
        expanded_boundaries=expanded_boundaries,
    )
    record: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        sample = pd.read_csv(sample_path)
        components = pd.read_csv(components_path)
        for column in ("event_time", "source_available_at", "source_open"):
            if column in sample:
                sample[column] = pd.to_datetime(sample[column], utc=True)
        for column in ("event_time", "component_source_available_at", "component_source_open"):
            if column in components:
                components[column] = pd.to_datetime(components[column], utc=True)
        effective_windows = analysis_window_hours(expanded_boundaries)
        analysis_coverage = audit_analysis_coverage(
            sample=sample,
            window_hours=effective_windows,
        )
        if not bool(analysis_coverage["passed"].all()):
            raise ValueError("The effective analysis windows do not have exact one-minute coverage.")
        orderbook = load_orderbook_surface()
        sources: dict[str, PairSurfaces] = {}
        actual_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        match_rows: list[dict[str, Any]] = []
        for _, episode in sample.sort_values("sample_selection_order").iterrows():
            pair = str(episode["pair"])
            if pair not in sources:
                sources[pair] = load_pair_surfaces(pair, cohort=str(episode["cohort"]))
            pair_surfaces = sources[pair]
            component_slice = components.loc[
                components["episode_id"].astype(str).eq(str(episode["episode_id"]))
            ].copy()
            result = analyze_episode(
                episode=episode,
                components=component_slice,
                surfaces=pair_surfaces,
                orderbook=orderbook,
                window_hours=effective_windows,
            )
            actual_rows.append(result["actual"])
            control_rows.append(result["control"])
            path_rows.extend(result["paths"])
            boundary_rows.append(result["boundary"])
            match_rows.extend(result["match_audit"])

        actual = DataFrame(actual_rows).sort_values("sample_selection_order").reset_index(drop=True)
        controls = DataFrame(control_rows).sort_values("sample_selection_order").reset_index(drop=True)
        combined = pd.concat([actual, controls], ignore_index=True, sort=False)
        paths = DataFrame(path_rows).sort_values(
            ["cohort", "episode_id", "event_kind", "checkpoint_minutes"]
        )
        boundaries = DataFrame(boundary_rows).sort_values("sample_selection_order")
        match_audit = DataFrame(match_rows).sort_values(
            ["sample_selection_order", "match_feature"]
        )
        catalog = feature_catalog(combined)
        screen = relationship_screen(actual, catalog)
        comparisons = control_comparisons(combined)
        summary = summarize_analysis(
            actual=actual,
            controls=controls,
            boundaries=boundaries,
            screen=screen,
        )

        event_path = artifact_dir / "g3g_event_replay.csv"
        checkpoint_path = artifact_dir / "g3g_checkpoint_paths.csv"
        feature_catalog_path = run_dir / "g3g_feature_catalog.csv"
        relationship_path = artifact_dir / "g3g_relationship_screen.csv"
        comparison_path = run_dir / "g3g_control_comparisons.csv"
        boundary_path = run_dir / "g3g_boundary_audit.csv"
        analysis_coverage_path = run_dir / "g3g_analysis_coverage_audit.csv"
        match_path = run_dir / "g3g_no_level_match_audit.csv"
        summary_path = run_dir / "g3g_analysis_summary.json"
        plot_path = run_dir / "g3g_aligned_one_minute_paths.png"
        atomic_write_csv(combined, event_path)
        atomic_write_csv(paths, checkpoint_path)
        atomic_write_csv(catalog, feature_catalog_path)
        atomic_write_csv(screen, relationship_path)
        atomic_write_csv(comparisons, comparison_path)
        atomic_write_csv(boundaries, boundary_path)
        atomic_write_csv(analysis_coverage, analysis_coverage_path)
        atomic_write_csv(match_audit, match_path)
        atomic_write_json(summary, summary_path)
        write_aligned_path_plot(paths, plot_path)

        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "actual_episodes": len(actual),
                "control_episodes": len(controls),
                "control_balance_usable": int(controls["control_balance_usable"].sum()),
                "boundary_extension_indicated": int(
                    boundaries["boundary_extension_indicated"].sum()
                ),
                "external_orderbook_available": int(actual["btc_orderbook_available"].sum()),
                "expanded_boundaries": expanded_boundaries,
                "summary": summary,
                "artifacts": {
                    "event_replay_csv": artifact_record(event_path),
                    "checkpoint_paths_csv": artifact_record(checkpoint_path),
                    "feature_catalog_csv": artifact_record(feature_catalog_path),
                    "relationship_screen_csv": artifact_record(relationship_path),
                    "control_comparisons_csv": artifact_record(comparison_path),
                    "boundary_audit_csv": artifact_record(boundary_path),
                    "analysis_coverage_audit_csv": artifact_record(analysis_coverage_path),
                    "no_level_match_audit_csv": artifact_record(match_path),
                    "summary_json": artifact_record(summary_path),
                    "aligned_path_plot": artifact_record(plot_path),
                },
            }
        )
        atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def analysis_request_contract(
    *,
    freeze_run_id: str,
    freeze_record_path: Path,
    coverage_record_path: Path,
    expanded_boundaries: bool,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_run_id": freeze_run_id,
        "freeze_record": str(freeze_record_path.resolve()),
        "freeze_record_sha256": sha256_file(freeze_record_path),
        "coverage_record": str(coverage_record_path.resolve()),
        "coverage_record_sha256": sha256_file(coverage_record_path),
        "expanded_boundaries": expanded_boundaries,
        "effective_windows_hours": {
            key: {"pre": value[0], "post": value[1]}
            for key, value in analysis_window_hours(expanded_boundaries).items()
        },
        "objective": (
            "Test whether the frozen high-thinness LVN-plus-independent-peer-cluster "
            "episodes differ from same-state no-level minutes, and describe rather than "
            "optimize which causal inputs accompany immediate traversal or rejection."
        ),
        "primary_hypothesis": (
            "Qualified thin-LVN cluster contacts show stronger sixty-minute volume and "
            "price displacement than same-pair, pre-event, same-state minutes that touch "
            "no causally available calculated level."
        ),
        "strongest_alternative": (
            "Ordinary short-term momentum and already-active volume or volatility explain "
            "the apparent response; the exact LVN and peer cluster add nothing."
        ),
        "reaction_definition": {
            "horizon_minutes": REACTION_HORIZON_MINUTES,
            "movement_origin": (
                "contact-minute close; direction and symmetric excursions begin with the "
                "next one-minute candle so the known approach cannot count as a reaction"
            ),
            "minimum_absolute_excursion_parent_zone_half_widths": (
                REACTION_EXCURSION_HALF_WIDTHS
            ),
            "minimum_post_to_pre_volume_ratio": REACTION_VOLUME_RATIO_MINIMUM,
            "note": (
                "This is a declared diagnostic label. Continuous displacement, excursion, "
                "volume, pressure, dwell, crossings, and timing remain primary."
            ),
        },
        "zone_resolution_geometry": {
            "breakout": "first post-contact reach of the far edge of the frozen zone",
            "rejection": (
                "first post-contact reach one additional parent half-width beyond the "
                "near entry edge"
            ),
            "contact_minute_used_for_ordering": False,
        },
        "controls": {
            "same_state_no_level": {
                "same_pair": True,
                "candidate_time": "strictly before the frozen event",
                "future_overlap_with_actual_event": False,
                "causally_available_level_contacts": 0,
                "same_pre_60m_trend_sign": True,
                "match_features": list(MATCH_FEATURES),
                "mean_robust_distance_limit": CONTROL_MEAN_DISTANCE_LIMIT,
                "maximum_robust_distance_limit": CONTROL_MAX_DISTANCE_LIMIT,
            },
            "simple_recent_trend": "sign of the causal pre-contact sixty-minute return",
            "majority_direction": "descriptive retrospective cohort majority only",
            "base_reaction_rate": True,
        },
        "technical_timeframes": list(TECHNICAL_TIMEFRAMES),
        "technical_indicators": list(TECHNICAL_DEFINITIONS),
        "technical_availability": "last fully completed bar before contact or pseudo-contact",
        "checkpoints_by_anchor_minutes": {
            key: list(values) for key, values in CHECKPOINT_MINUTES.items()
        },
        "boundary_audit": {
            "maximum_zone_fraction_first_or_last_hour": BOUNDARY_ZONE_FRACTION_LIMIT,
            "maximum_activity_or_volatility_ratio": BOUNDARY_ACTIVITY_RATIO_LIMIT,
            "interpretation": (
                "Any flag is recorded as an expansion candidate; because this one direct "
                "pass has already retained long anchor-scaled windows, flagged episodes are "
                "not silently dropped or selectively extended."
            ),
        },
        "model_training": False,
        "profit_optimization": False,
        "indicator_parameter_tuning": False,
        "worker_count": 1,
        "promotion_allowed": False,
        "implementation_repair_history": [
            (
                "The first technical execution failed before event outcomes because Pandas "
                "3 rejects the former 5m resampling alias. The root parser now maps human "
                "timeframe labels to explicit 5min/15min offsets and is regression-tested."
            ),
            (
                "The first expanded invocation passed the new window map to the control "
                "path calculator by mistake and stopped before completing an episode. The "
                "argument now routes only to control selection and the boundary audit."
            ),
            (
                "The sequential expansion download wrapper returned a non-zero shell status "
                "after its final command; the independent audit subsequently verified every "
                "one of 318,960 episode-window rows with zero duplicates or gaps."
            ),
            (
                "A pre-final diagnostic counted the near entry edge during the contact "
                "minute as an away threshold. Because the approach had already visited that "
                "edge, the apparent away-first result was tautological. Final direction "
                "starts after the contact bar from its close, and final zone rejection must "
                "travel one extra half-width beyond the entry edge."
            ),
        ],
    }


def analysis_window_hours(expanded_boundaries: bool) -> dict[str, tuple[int, int]]:
    multiplier = 2 if expanded_boundaries else 1
    return {
        timeframe: (pre_hours * multiplier, post_hours * multiplier)
        for timeframe, (pre_hours, post_hours) in WINDOW_HOURS.items()
    }


def analysis_acquisition_intervals(
    sample: DataFrame, *, expanded_boundaries: bool
) -> DataFrame:
    windows = analysis_window_hours(expanded_boundaries)
    rows: list[dict[str, Any]] = []
    for _, episode in sample.iterrows():
        anchor = str(episode["cluster_causal_anchor_timeframe"])
        pre_hours, post_hours = windows[anchor]
        event_time = pd.Timestamp(episode["event_time"])
        visible_start = event_time - pd.Timedelta(hours=pre_hours)
        visible_end = event_time + pd.Timedelta(hours=post_hours + 1)
        rows.append(
            {
                "episode_ids": str(episode["episode_id"]),
                "cohorts": str(episode["cohort"]),
                "pair": str(episode["pair"]),
                "parent_contact_hours_utc": event_time.isoformat(),
                "causal_anchor_timeframes": anchor,
                "maximum_visible_pre_hours": pre_hours,
                "maximum_visible_post_hours": post_hours,
                "feature_warmup_hours": FEATURE_WARMUP_HOURS,
                "visible_start_utc": visible_start,
                "visible_end_exclusive_utc": visible_end,
                "interval_start_utc": visible_start
                - pd.Timedelta(hours=FEATURE_WARMUP_HOURS),
                "interval_end_exclusive_utc": visible_end,
                "exchange": "binance",
                "market_type": "futures",
                "timeframe": "1m",
                "data_format": "feather",
            }
        )
    return merge_overlapping_acquisition_intervals(DataFrame(rows))


def audit_analysis_coverage(
    *, sample: DataFrame, window_hours: dict[str, tuple[int, int]]
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, episode in sample.iterrows():
        anchor = str(episode["cluster_causal_anchor_timeframe"])
        pre_hours, post_hours = window_hours[anchor]
        event_time = pd.Timestamp(episode["event_time"])
        start = event_time - pd.Timedelta(hours=pre_hours + FEATURE_WARMUP_HOURS)
        end = event_time + pd.Timedelta(hours=post_hours + 1)
        path = FUTURES_DATA_DIR / f"{pair_stem(str(episode['pair']))}-1m-futures.feather"
        minute = load_ohlcv(path)
        selected = minute.loc[(minute["date"] >= start) & (minute["date"] < end)].copy()
        expected = int((end - start).total_seconds() // 60)
        differences = selected["date"].diff().dropna()
        passed = bool(
            len(selected) == expected
            and not selected["date"].duplicated().any()
            and (differences.eq(pd.Timedelta(minutes=1)).all() if len(selected) > 1 else False)
            and (selected.iloc[0]["date"] == start if len(selected) else False)
            and (
                selected.iloc[-1]["date"] == end - pd.Timedelta(minutes=1)
                if len(selected)
                else False
            )
        )
        rows.append(
            {
                "episode_id": str(episode["episode_id"]),
                "cohort": str(episode["cohort"]),
                "pair": str(episode["pair"]),
                "anchor_timeframe": anchor,
                "interval_start_utc": start,
                "interval_end_exclusive_utc": end,
                "expected_rows": expected,
                "actual_rows": len(selected),
                "duplicate_minutes": int(selected["date"].duplicated().sum()),
                "non_one_minute_gaps": int(differences.ne(pd.Timedelta(minutes=1)).sum()),
                "source_file": str(path.resolve()),
                "source_file_sha256": sha256_file(path),
                "passed": passed,
            }
        )
    return DataFrame(rows).sort_values(["cohort", "pair", "interval_start_utc"])


def load_pair_surfaces(pair: str, *, cohort: str) -> PairSurfaces:
    minute_path = FUTURES_DATA_DIR / f"{pair_stem(pair)}-1m-futures.feather"
    hour_path = FUTURES_DATA_DIR / f"{pair_stem(pair)}-1h-futures.feather"
    minute = load_ohlcv(minute_path)
    hourly = load_ohlcv(hour_path)
    technical = {
        "1m": technical_frame(minute, timeframe="1m"),
        "5m": technical_frame(resample_ohlcv(minute, "5m"), timeframe="5m"),
        "15m": technical_frame(resample_ohlcv(minute, "15m"), timeframe="15m"),
        "1h": technical_frame(hourly, timeframe="1h"),
        "4h": technical_frame(resample_ohlcv(hourly, "4h"), timeframe="4h"),
        "8h": technical_frame(resample_ohlcv(hourly, "8h"), timeframe="8h"),
    }
    source = next(item for item in COHORT_SOURCES if item.cohort == cohort)
    manifest = load_manifest(source.manifest_path)
    base = prepare_base_market_frame(pair, manifest)
    aligned_levels = aligned_selected_levels_from_verified_prefix(
        pair=pair,
        base=base,
        manifest_path=source.manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    return PairSurfaces(
        minute=minute,
        match=minute_match_frame(minute),
        technical=technical,
        base=base,
        aligned_levels=aligned_levels,
    )


def load_ohlcv(path: Path) -> DataFrame:
    frame = pd.read_feather(path, columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def resample_ohlcv(frame: DataFrame, timeframe: str) -> DataFrame:
    if timeframe == "1m":
        return frame.copy()
    indexed = frame.set_index("date")
    output = indexed.resample(PANDAS_TIMEFRAME[timeframe], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    output = output.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return output


def technical_frame(frame: DataFrame, *, timeframe: str) -> DataFrame:
    """Return causal indicator values stamped when each source bar is fully complete."""
    output = frame.copy()
    close = output["close"].astype(float)
    high = output["high"].astype(float)
    low = output["low"].astype(float)
    open_ = output["open"].astype(float)
    volume = output["volume"].astype(float).clip(lower=0.0)
    price_denominator = close.replace(0.0, np.nan)
    bar_range = (high - low).replace(0.0, np.nan)
    body_pressure = ((close - open_) / bar_range).clip(-1.0, 1.0).fillna(0.0)
    signed_body_volume = body_pressure * volume
    returns = close.pct_change(fill_method=None)

    output["return_1bar"] = close.pct_change(1, fill_method=None)
    output["return_3bar"] = close.pct_change(3, fill_method=None)
    output["return_12bar"] = close.pct_change(12, fill_method=None)
    output["realized_volatility_14bar"] = returns.rolling(14, min_periods=14).std(ddof=0)
    volume_mean20 = volume.rolling(20, min_periods=20).mean()
    volume_mean60 = volume.rolling(60, min_periods=30).mean()
    volume_std60 = volume.rolling(60, min_periods=30).std(ddof=0).replace(0.0, np.nan)
    output["volume_ratio_20bar"] = volume / volume_mean20.replace(0.0, np.nan)
    output["volume_z_60bar"] = (volume - volume_mean60) / volume_std60
    for window in (5, 20, 60):
        output[f"pressure_{window}bar"] = (
            signed_body_volume.rolling(window, min_periods=window).sum()
            / volume.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
        )

    rsi = relative_strength_index(close, 14)
    output["rsi14"] = rsi
    bollinger_mid = close.rolling(20, min_periods=20).mean()
    bollinger_std = close.rolling(20, min_periods=20).std(ddof=0)
    bollinger_upper = bollinger_mid + 2.0 * bollinger_std
    bollinger_lower = bollinger_mid - 2.0 * bollinger_std
    bollinger_span = (bollinger_upper - bollinger_lower).replace(0.0, np.nan)
    output["bollinger_position20"] = (close - bollinger_lower) / bollinger_span
    output["bollinger_width20"] = bollinger_span / bollinger_mid.replace(0.0, np.nan)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    output["macd_histogram_pct"] = (macd_line - macd_signal) / price_denominator

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    for period, average in ((20, sma20), (50, sma50), (200, sma200)):
        output[f"sma{period}_distance_pct"] = (close - average) / price_denominator
    output["sma20_minus_sma50_pct"] = (sma20 - sma50) / price_denominator
    output["sma50_minus_sma200_pct"] = (sma50 - sma200) / price_denominator
    for period, average in ((12, ema12), (26, ema26), (50, ema50)):
        output[f"ema{period}_distance_pct"] = (close - average) / price_denominator
    output["ema12_minus_ema26_pct"] = (ema12 - ema26) / price_denominator

    atr, plus_di, minus_di, adx = directional_movement(high, low, close, 14)
    output["atr14_pct"] = atr / price_denominator
    output["adx14"] = adx
    output["di_spread14"] = plus_di - minus_di
    money_flow_multiplier = (
        ((close - low) - (high - close)) / bar_range
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output["cmf20"] = (
        (money_flow_multiplier * volume).rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    output["obv_trend20"] = (
        obv.diff(20) / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    for window in (20, 50):
        prior_high = high.shift(1).rolling(window, min_periods=window).max()
        prior_low = low.shift(1).rolling(window, min_periods=window).min()
        output[f"range_position{window}"] = (
            (close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
        )

    add_cross_recency(output, "rsi_cross_above_50", crosses_above(rsi, 50.0))
    add_cross_recency(output, "rsi_cross_below_50", crosses_below(rsi, 50.0))
    add_cross_recency(output, "rsi_cross_above_30", crosses_above(rsi, 30.0))
    add_cross_recency(output, "rsi_cross_below_70", crosses_below(rsi, 70.0))
    add_cross_recency(
        output,
        "close_cross_above_bollinger_upper",
        crosses_above(close, bollinger_upper),
    )
    add_cross_recency(
        output,
        "close_cross_below_bollinger_upper",
        crosses_below(close, bollinger_upper),
    )
    add_cross_recency(
        output,
        "close_cross_below_bollinger_lower",
        crosses_below(close, bollinger_lower),
    )
    add_cross_recency(
        output,
        "close_cross_above_bollinger_lower",
        crosses_above(close, bollinger_lower),
    )
    add_cross_recency(output, "macd_bull_cross", crosses_above(macd_line, macd_signal))
    add_cross_recency(output, "macd_bear_cross", crosses_below(macd_line, macd_signal))
    for period, average in ((20, sma20), (50, sma50), (200, sma200)):
        add_cross_recency(
            output,
            f"close_cross_above_sma{period}",
            crosses_above(close, average),
        )
        add_cross_recency(
            output,
            f"close_cross_below_sma{period}",
            crosses_below(close, average),
        )
    add_cross_recency(
        output,
        "sma20_cross_above_sma50",
        crosses_above(sma20, sma50),
    )
    add_cross_recency(
        output,
        "sma20_cross_below_sma50",
        crosses_below(sma20, sma50),
    )
    add_cross_recency(
        output,
        "sma50_cross_above_sma200",
        crosses_above(sma50, sma200),
    )
    add_cross_recency(
        output,
        "sma50_cross_below_sma200",
        crosses_below(sma50, sma200),
    )
    add_cross_recency(
        output,
        "ema12_cross_above_ema26",
        crosses_above(ema12, ema26),
    )
    add_cross_recency(
        output,
        "ema12_cross_below_ema26",
        crosses_below(ema12, ema26),
    )
    add_cross_recency(output, "di_cross_bullish", crosses_above(plus_di, minus_di))
    add_cross_recency(output, "di_cross_bearish", crosses_below(plus_di, minus_di))

    delta = pd.Timedelta(PANDAS_TIMEFRAME[timeframe])
    output["available_at"] = output["date"] + delta
    numeric_columns = output.select_dtypes(include=["number", "bool"]).columns
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return output


def relative_strength_index(close: Series, period: int) -> Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative)
    return rsi.where(average_loss.ne(0.0), 100.0)


def directional_movement(
    high: Series, low: Series, close: Series, period: int
) -> tuple[Series, Series, Series, Series]:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    plus_smoothed = plus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    minus_smoothed = minus_dm.ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    plus_di = 100.0 * plus_smoothed / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_smoothed / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr, plus_di, minus_di, adx


def crosses_above(left: Series, right: Series | float) -> Series:
    right_series = right if isinstance(right, Series) else Series(right, index=left.index)
    return left.gt(right_series) & left.shift(1).le(right_series.shift(1))


def crosses_below(left: Series, right: Series | float) -> Series:
    right_series = right if isinstance(right, Series) else Series(right, index=left.index)
    return left.lt(right_series) & left.shift(1).ge(right_series.shift(1))


def add_cross_recency(frame: DataFrame, name: str, crossed: Series) -> None:
    positions = np.arange(len(frame), dtype=float)
    last_cross = Series(np.where(crossed.fillna(False), positions, np.nan), index=frame.index).ffill()
    frame[f"bars_since_{name}"] = Series(positions, index=frame.index) - last_cross


def minute_match_frame(minute: DataFrame) -> DataFrame:
    output = minute[["date", "open", "high", "low", "close", "volume"]].copy()
    close = output["close"].astype(float)
    high = output["high"].astype(float)
    low = output["low"].astype(float)
    open_ = output["open"].astype(float)
    volume = output["volume"].astype(float).clip(lower=0.0)
    returns = close.pct_change(fill_method=None)
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    pressure = ((close - open_) / (high - low).replace(0.0, np.nan)).clip(-1.0, 1.0)
    signed_volume = pressure.fillna(0.0) * volume
    output["match_return_15m"] = close.pct_change(15, fill_method=None)
    output["match_return_60m"] = close.pct_change(60, fill_method=None)
    output["match_return_240m"] = close.pct_change(240, fill_method=None)
    output["match_realized_volatility_60m"] = returns.rolling(60, min_periods=60).std(ddof=0)
    output["match_atr14_pct"] = atr14 / close.replace(0.0, np.nan)
    output["match_volume_ratio_20m"] = (
        volume / volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    )
    output["match_pressure_20m"] = (
        signed_volume.rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    prior_high = high.shift(1).rolling(240, min_periods=240).max()
    prior_low = low.shift(1).rolling(240, min_periods=240).min()
    output["match_range_position_240m"] = (
        (close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
    )
    # A pseudo-contact beginning at row i may only use values through row i-1.
    output[list(MATCH_FEATURES)] = output[list(MATCH_FEATURES)].shift(1)
    return output


def technical_snapshot(
    technical: dict[str, DataFrame], timestamp: pd.Timestamp
) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for timeframe, frame in technical.items():
        eligible = frame.loc[frame["available_at"] <= timestamp]
        prefix = TIMEFRAME_PREFIX[timeframe]
        if eligible.empty:
            for feature in TECHNICAL_DEFINITIONS:
                snapshot[f"{prefix}__{feature}"] = math.nan
            continue
        row = eligible.iloc[-1]
        for feature in TECHNICAL_DEFINITIONS:
            snapshot[f"{prefix}__{feature}"] = finite_or_nan(row.get(feature))
    snapshot.update(multitimeframe_alignment(snapshot))
    return snapshot


def multitimeframe_alignment(snapshot: dict[str, float]) -> dict[str, float]:
    def values(feature: str) -> list[float]:
        output = []
        for prefix in TIMEFRAME_PREFIX.values():
            value = finite_or_nan(snapshot.get(f"{prefix}__{feature}"))
            if np.isfinite(value):
                output.append(value)
        return output

    return {
        "mtf_rsi_above_50_count": float(sum(value > 50.0 for value in values("rsi14"))),
        "mtf_macd_positive_count": float(
            sum(value > 0.0 for value in values("macd_histogram_pct"))
        ),
        "mtf_close_above_sma20_count": float(
            sum(value > 0.0 for value in values("sma20_distance_pct"))
        ),
        "mtf_ema12_above_ema26_count": float(
            sum(value > 0.0 for value in values("ema12_minus_ema26_pct"))
        ),
        "mtf_positive_pressure_count": float(
            sum(value > 0.0 for value in values("pressure_20bar"))
        ),
        "mtf_elevated_volume_count": float(
            sum(value > 1.0 for value in values("volume_ratio_20bar"))
        ),
        "mtf_adx_above_25_count": float(sum(value >= 25.0 for value in values("adx14"))),
    }


def analyze_episode(
    *,
    episode: Series,
    components: DataFrame,
    surfaces: PairSurfaces,
    orderbook: DataFrame,
    window_hours: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    contact_time = first_parent_zone_contact(episode, surfaces.minute)
    anchor = str(episode["cluster_causal_anchor_timeframe"])
    checkpoints = CHECKPOINT_MINUTES[anchor]
    actual_match = match_snapshot(surfaces.match, contact_time)
    actual_technical = technical_snapshot(surfaces.technical, contact_time)
    actual_cluster = cluster_snapshot(episode, components)
    actual_orderbook = orderbook_snapshot(orderbook, contact_time)
    common = episode_identity(episode)
    actual_paths = checkpoint_paths(
        minute=surfaces.minute,
        timestamp=contact_time,
        reference_price=float(episode["level_price"]),
        zone_half_width=float(episode["zone_half_width"]),
        level_name=str(episode["level_name"]),
        checkpoints=checkpoints,
        episode_id=str(episode["episode_id"]),
        cohort=str(episode["cohort"]),
        event_kind="actual_cluster_contact",
    )
    actual_primary = primary_event_outcomes(actual_paths, actual_match)
    actual_row = {
        **common,
        "event_kind": "actual_cluster_contact",
        "contact_time": contact_time,
        "reference_price": float(episode["level_price"]),
        "zone_half_width": float(episode["zone_half_width"]),
        "base_atr": float(episode["base_atr"]),
        **actual_match,
        **actual_technical,
        **actual_cluster,
        **actual_orderbook,
        **actual_primary,
        "control_state_distance_mean": math.nan,
        "control_state_distance_max": math.nan,
        "control_balance_usable": False,
        "causal_level_contacts_at_event": int(
            causal_level_contact_count(
                timestamp=contact_time,
                candle=minute_row_at(surfaces.minute, contact_time),
                base=surfaces.base,
                aligned_levels=surfaces.aligned_levels,
            )
        ),
    }

    control_selection = select_no_level_control(
        episode=episode,
        actual_contact_time=contact_time,
        actual_match=actual_match,
        surfaces=surfaces,
        window_hours=window_hours,
    )
    control_time = pd.Timestamp(control_selection["timestamp"])
    control_reference = float(
        minute_row_at(surfaces.minute, control_time)["open"]
    )
    control_paths = checkpoint_paths(
        minute=surfaces.minute,
        timestamp=control_time,
        reference_price=control_reference,
        zone_half_width=float(episode["zone_half_width"]),
        level_name=str(episode["level_name"]),
        checkpoints=tuple(value for value in checkpoints if value <= CONTROL_HORIZON_MINUTES),
        episode_id=str(episode["episode_id"]),
        cohort=str(episode["cohort"]),
        event_kind="same_state_no_level",
    )
    control_primary = primary_event_outcomes(
        control_paths, control_selection["match_snapshot"]
    )
    control_row = {
        **common,
        "event_kind": "same_state_no_level",
        "contact_time": control_time,
        "reference_price": control_reference,
        "zone_half_width": float(episode["zone_half_width"]),
        "base_atr": float(episode["base_atr"]),
        **control_selection["match_snapshot"],
        **technical_snapshot(surfaces.technical, control_time),
        **empty_cluster_snapshot(),
        **orderbook_snapshot(orderbook, control_time),
        **control_primary,
        "control_state_distance_mean": float(control_selection["distance_mean"]),
        "control_state_distance_max": float(control_selection["distance_max"]),
        "control_balance_usable": bool(control_selection["balance_usable"]),
        "causal_level_contacts_at_event": 0,
    }
    boundary = boundary_audit(
        episode=episode,
        contact_time=contact_time,
        minute=surfaces.minute,
        window_hours=window_hours,
    )
    return {
        "actual": actual_row,
        "control": control_row,
        "paths": [*actual_paths, *control_paths],
        "boundary": boundary,
        "match_audit": control_selection["match_audit"],
    }


def episode_identity(episode: Series) -> dict[str, Any]:
    copied = {
        "episode_id": str(episode["episode_id"]),
        "sample_selection_order": int(episode["sample_selection_order"]),
        "cohort": str(episode["cohort"]),
        "period": str(episode["period"]),
        "pair": str(episode["pair"]),
        "parent_contact_hour": pd.Timestamp(episode["event_time"]),
        "level_name": str(episode["level_name"]),
        "approach_state": str(episode["approach_state"]),
        "source_timeframe": str(episode["source_timeframe"]),
        "cluster_causal_anchor_timeframe": str(
            episode["cluster_causal_anchor_timeframe"]
        ),
        "parent_high_thinness_score": float(episode["parent_high_thinness_score"]),
        "parent_contact_volume_ratio_1h": float(episode["contact_volume_ratio"]),
        "parent_contact_pressure_change_1h": float(episode["contact_pressure_change"]),
        "parent_contact_range_ratio_1h": float(episode["contact_range_ratio"]),
    }
    for column in episode.index:
        if str(column).startswith("state_"):
            copied[f"parent_{column}"] = finite_or_nan(episode[column])
    return copied


def first_parent_zone_contact(episode: Series, minute: DataFrame) -> pd.Timestamp:
    start = pd.Timestamp(episode["event_time"])
    end = start + pd.Timedelta(hours=1)
    lower = float(episode["level_price"]) - float(episode["zone_half_width"])
    upper = float(episode["level_price"]) + float(episode["zone_half_width"])
    parent_hour = minute.loc[(minute["date"] >= start) & (minute["date"] < end)]
    contacts = parent_hour.loc[
        parent_hour["high"].ge(lower) & parent_hour["low"].le(upper)
    ]
    if contacts.empty:
        raise ValueError(
            f"No one-minute reproduction of frozen parent contact: {episode['episode_id']}"
        )
    return pd.Timestamp(contacts.iloc[0]["date"])


def minute_row_at(minute: DataFrame, timestamp: pd.Timestamp) -> Series:
    selected = minute.loc[minute["date"].eq(timestamp)]
    if len(selected) != 1:
        raise ValueError(f"Expected one minute at {timestamp}, found {len(selected)}.")
    return selected.iloc[0]


def match_snapshot(match: DataFrame, timestamp: pd.Timestamp) -> dict[str, float]:
    row = minute_row_at(match, timestamp)
    return {feature: finite_or_nan(row[feature]) for feature in MATCH_FEATURES}


def cluster_snapshot(episode: Series, components: DataFrame) -> dict[str, Any]:
    primary = components.loc[components["in_primary_connected_contact_cluster"].eq(True)].copy()
    peers = primary.loc[primary["component_kind"].eq("peer_level")].copy()
    if primary.empty or peers.empty:
        raise ValueError(f"Frozen cluster components missing: {episode['episode_id']}")
    prices = pd.to_numeric(primary["component_level_price"], errors="coerce")
    base_atr = float(episode["base_atr"])
    dependency_groups = sorted(set(peers["component_dependency_group"].astype(str)))
    timeframes = sorted(set(peers["component_timeframe"].astype(str)), key=timeframe_hours)
    families = sorted(set(peers["component_family"].astype(str)))
    signature = ";".join(
        sorted(
            f"{row.component_family}:{row.component_name}@{row.component_timeframe}"
            for row in peers.itertuples()
        )
    )
    return {
        "cluster_primary_peer_level_count": int(len(peers)),
        "cluster_primary_dependency_group_count": int(len(dependency_groups)),
        "cluster_primary_timeframe_count": int(len(timeframes)),
        "cluster_primary_span_atr": float((prices.max() - prices.min()) / base_atr),
        "cluster_parent_to_peer_median_distance_atr": float(
            (prices.loc[peers.index] - float(episode["level_price"])).abs().median()
            / base_atr
        ),
        "cluster_dependency_groups": ";".join(dependency_groups),
        "cluster_timeframes": ";".join(timeframes),
        "cluster_families": ";".join(families),
        "cluster_component_signature": signature,
        "cluster_price_average_family_count": int(
            peers["component_dependency_group"].astype(str).eq("price_average_family").sum()
        ),
        "cluster_rolling_price_extreme_count": int(
            peers["component_dependency_group"].astype(str).eq("rolling_price_extreme").sum()
        ),
        "cluster_round_number_count": int(
            peers["component_dependency_group"].astype(str).eq("round_number").sum()
        ),
        "cluster_peer_1h_count": int(peers["component_timeframe"].astype(str).eq("1h").sum()),
        "cluster_peer_4h_count": int(peers["component_timeframe"].astype(str).eq("4h").sum()),
        "cluster_peer_8h_count": int(peers["component_timeframe"].astype(str).eq("8h").sum()),
    }


def empty_cluster_snapshot() -> dict[str, Any]:
    return {
        "cluster_primary_peer_level_count": 0,
        "cluster_primary_dependency_group_count": 0,
        "cluster_primary_timeframe_count": 0,
        "cluster_primary_span_atr": math.nan,
        "cluster_parent_to_peer_median_distance_atr": math.nan,
        "cluster_dependency_groups": "none",
        "cluster_timeframes": "none",
        "cluster_families": "none",
        "cluster_component_signature": "same_state_no_level",
        "cluster_price_average_family_count": 0,
        "cluster_rolling_price_extreme_count": 0,
        "cluster_round_number_count": 0,
        "cluster_peer_1h_count": 0,
        "cluster_peer_4h_count": 0,
        "cluster_peer_8h_count": 0,
    }


def checkpoint_paths(
    *,
    minute: DataFrame,
    timestamp: pd.Timestamp,
    reference_price: float,
    zone_half_width: float,
    level_name: str,
    checkpoints: Iterable[int],
    episode_id: str,
    cohort: str,
    event_kind: str,
) -> list[dict[str, Any]]:
    if not np.isfinite(zone_half_width) or zone_half_width <= 0.0:
        raise ValueError(f"Invalid parent zone half-width for {episode_id}.")
    rows: list[dict[str, Any]] = []
    through_sign = 1.0 if level_name == "lvn_above" else -1.0
    contact_bar = minute_row_at(minute, timestamp)
    movement_origin = float(contact_bar["close"])
    decision_time = timestamp + pd.Timedelta(minutes=1)
    for checkpoint in checkpoints:
        horizon = int(checkpoint)
        direction_post = minute.loc[
            (minute["date"] >= decision_time)
            & (minute["date"] < decision_time + pd.Timedelta(minutes=horizon))
        ].copy()
        activity_post = minute.loc[
            (minute["date"] >= timestamp)
            & (minute["date"] < timestamp + pd.Timedelta(minutes=horizon))
        ].copy()
        pre = minute.loc[
            (minute["date"] >= timestamp - pd.Timedelta(minutes=horizon))
            & (minute["date"] < timestamp)
        ].copy()
        if (
            len(direction_post) != horizon
            or len(activity_post) != horizon
            or len(pre) != horizon
        ):
            raise ValueError(
                f"Incomplete {horizon}m path for {episode_id} {event_kind}: "
                f"pre={len(pre)} activity={len(activity_post)} "
                f"direction={len(direction_post)}"
            )
        max_up = max(
            0.0,
            float((direction_post["high"].max() - movement_origin) / zone_half_width),
        )
        max_down = max(
            0.0,
            float((movement_origin - direction_post["low"].min()) / zone_half_width),
        )
        close_displacement = float(
            (direction_post.iloc[-1]["close"] - movement_origin) / zone_half_width
        )
        volume_ratio = float(
            activity_post["volume"].mean() / pre["volume"].mean()
            if pre["volume"].mean() > 0.0
            else math.nan
        )
        pre_pressure = volume_weighted_pressure(pre)
        post_pressure = volume_weighted_pressure(activity_post)
        pressure_change = post_pressure - pre_pressure
        pre_volatility = realized_volatility(pre)
        post_volatility = realized_volatility(activity_post)
        zone_overlap = direction_post["high"].ge(reference_price - zone_half_width) & direction_post[
            "low"
        ].le(reference_price + zone_half_width)
        close_side = np.sign(
            direction_post["close"].to_numpy(dtype=float) - reference_price
        )
        crossings = int(np.sum(close_side[1:] != close_side[:-1])) if len(close_side) > 1 else 0
        first_up = first_true_position(
            direction_post["high"].ge(movement_origin + zone_half_width)
        )
        first_down = first_true_position(
            direction_post["low"].le(movement_origin - zone_half_width)
        )
        if through_sign > 0.0:
            breakout_condition = direction_post["high"].ge(reference_price + zone_half_width)
            rejection_condition = direction_post["low"].le(
                reference_price - 2.0 * zone_half_width
            )
            breakout_beyond = max(
                0.0,
                float(
                    (
                        direction_post["high"].max()
                        - (reference_price + zone_half_width)
                    )
                    / zone_half_width
                ),
            )
            rejection_beyond = max(
                0.0,
                float(
                    (
                        (reference_price - zone_half_width)
                        - direction_post["low"].min()
                    )
                    / zone_half_width
                ),
            )
        else:
            breakout_condition = direction_post["low"].le(reference_price - zone_half_width)
            rejection_condition = direction_post["high"].ge(
                reference_price + 2.0 * zone_half_width
            )
            breakout_beyond = max(
                0.0,
                float(
                    (
                        (reference_price - zone_half_width)
                        - direction_post["low"].min()
                    )
                    / zone_half_width
                ),
            )
            rejection_beyond = max(
                0.0,
                float(
                    (
                        direction_post["high"].max()
                        - (reference_price + zone_half_width)
                    )
                    / zone_half_width
                ),
            )
        first_breakout = first_true_position(breakout_condition)
        first_rejection = first_true_position(rejection_condition)
        rows.append(
            {
                "episode_id": episode_id,
                "cohort": cohort,
                "event_kind": event_kind,
                "timestamp": timestamp,
                "decision_time_after_contact_bar": decision_time,
                "checkpoint_minutes": horizon,
                "reference_price": reference_price,
                "movement_origin_contact_close": movement_origin,
                "zone_half_width": zone_half_width,
                "close_displacement_half_widths": close_displacement,
                "close_displacement_through_positive_half_widths": (
                    close_displacement * through_sign
                ),
                "maximum_up_excursion_half_widths": max_up,
                "maximum_down_excursion_half_widths": max_down,
                "maximum_absolute_excursion_half_widths": max(max_up, max_down),
                "through_excursion_half_widths": max_up if through_sign > 0.0 else max_down,
                "away_excursion_half_widths": max_down if through_sign > 0.0 else max_up,
                "through_direction": "up" if through_sign > 0.0 else "down",
                "first_up_threshold_minute": first_up,
                "first_down_threshold_minute": first_down,
                "first_zone_breakout_minute": first_breakout,
                "first_zone_rejection_minute": first_rejection,
                "breakout_excursion_beyond_far_edge_half_widths": breakout_beyond,
                "rejection_excursion_beyond_entry_edge_half_widths": rejection_beyond,
                "volume_ratio_post_pre": volume_ratio,
                "pressure_pre": pre_pressure,
                "pressure_post": post_pressure,
                "pressure_change": pressure_change,
                "pressure_change_toward_through": pressure_change * through_sign,
                "pressure_change_toward_away": pressure_change * -through_sign,
                "realized_volatility_pre": pre_volatility,
                "realized_volatility_post": post_volatility,
                "realized_volatility_ratio": (
                    post_volatility / pre_volatility if pre_volatility > 0.0 else math.nan
                ),
                "zone_overlap_fraction": float(zone_overlap.mean()),
                "close_crossings_of_reference": crossings,
            }
        )
    return rows


def primary_event_outcomes(
    paths: list[dict[str, Any]], pre_state: dict[str, float]
) -> dict[str, Any]:
    sixty = next(row for row in paths if row["checkpoint_minutes"] == REACTION_HORIZON_MINUTES)
    timestamp = pd.Timestamp(sixty["timestamp"])
    reference = float(sixty["reference_price"])
    half_width = float(sixty["zone_half_width"])
    # Reconstruct first-passage ordering from the already bounded path inputs.
    event_kind = str(sixty["event_kind"])
    # first_direction is attached by first-passage data stored temporarily by caller below.
    # It is filled in by first_direction_from_paths after the compact path list is made.
    excursion = float(sixty["maximum_absolute_excursion_half_widths"])
    volume_ratio = float(sixty["volume_ratio_post_pre"])
    reaction = bool(
        excursion >= REACTION_EXCURSION_HALF_WIDTHS
        and volume_ratio >= REACTION_VOLUME_RATIO_MINIMUM
    )
    # The exact first direction is recovered from the one-minute checkpoint sequence: the
    # 1m/3m/... rows alone cannot disambiguate two thresholds within a long checkpoint.
    first_direction = first_direction_from_path_rows(paths)
    first_zone_resolution = first_zone_resolution_from_path_rows(paths)
    trend_value = finite_or_nan(pre_state.get("match_return_60m"))
    trend_call = int(np.sign(trend_value)) if np.isfinite(trend_value) else 0
    direction_numeric = {"up": 1, "down": -1, "tie": 0, "unreached": 0}[first_direction]
    direction_callable = first_direction in {"up", "down"}
    trend_correct = bool(direction_callable and trend_call == direction_numeric)
    through_sign = 1 if str(sixty["through_direction"]) == "up" else -1
    first_path = (
        "through"
        if direction_callable and direction_numeric == through_sign
        else "away"
        if direction_callable
        else first_direction
    )
    first_path_numeric = {"through": 1, "away": -1, "tie": 0, "unreached": 0}[
        first_path
    ]
    first_zone_resolution_numeric = {
        "breakout": 1,
        "rejection": -1,
        "tie": 0,
        "unreached": 0,
    }[first_zone_resolution]
    return {
        "reaction_60m": reaction,
        "excursion_strength_60_half_widths": excursion,
        "volume_ratio_post_pre_60m": volume_ratio,
        "pressure_change_60m": float(sixty["pressure_change"]),
        "pressure_change_toward_through_60m": float(
            sixty["pressure_change_toward_through"]
        ),
        "pressure_change_toward_away_60m": float(
            sixty["pressure_change_toward_away"]
        ),
        "realized_volatility_ratio_60m": float(sixty["realized_volatility_ratio"]),
        "zone_overlap_fraction_60m": float(sixty["zone_overlap_fraction"]),
        "reference_crossings_60m": int(sixty["close_crossings_of_reference"]),
        "through_excursion_60_half_widths": float(sixty["through_excursion_half_widths"]),
        "away_excursion_60_half_widths": float(sixty["away_excursion_half_widths"]),
        "breakout_excursion_beyond_far_edge_60_half_widths": float(
            sixty["breakout_excursion_beyond_far_edge_half_widths"]
        ),
        "rejection_excursion_beyond_entry_edge_60_half_widths": float(
            sixty["rejection_excursion_beyond_entry_edge_half_widths"]
        ),
        "first_direction": first_direction,
        "first_direction_numeric": direction_numeric,
        "first_path_relative_to_approach": first_path,
        "first_path_numeric": first_path_numeric,
        "first_path_away": bool(first_path == "away"),
        "first_path_through": bool(first_path == "through"),
        "first_zone_resolution": first_zone_resolution,
        "first_zone_resolution_numeric": first_zone_resolution_numeric,
        "first_zone_rejection": bool(first_zone_resolution == "rejection"),
        "first_zone_breakout": bool(first_zone_resolution == "breakout"),
        "direction_callable": direction_callable,
        "simple_trend_direction_call": trend_call,
        "simple_trend_direction_correct": trend_correct,
        "simple_trend_joint_reaction_and_direction": bool(reaction and trend_correct),
        "primary_outcome_timestamp": timestamp,
        "primary_reference_price": reference,
        "primary_movement_origin_contact_close": float(
            sixty["movement_origin_contact_close"]
        ),
        "primary_zone_half_width": half_width,
        "primary_event_kind": event_kind,
    }


def first_direction_from_path_rows(paths: list[dict[str, Any]]) -> str:
    row = next(item for item in paths if item["checkpoint_minutes"] == REACTION_HORIZON_MINUTES)
    up_at = row["first_up_threshold_minute"]
    down_at = row["first_down_threshold_minute"]
    if up_at is None and down_at is None:
        return "unreached"
    if up_at is None:
        return "down"
    if down_at is None:
        return "up"
    if up_at == down_at:
        return "tie"
    return "up" if up_at < down_at else "down"


def first_zone_resolution_from_path_rows(paths: list[dict[str, Any]]) -> str:
    row = next(item for item in paths if item["checkpoint_minutes"] == REACTION_HORIZON_MINUTES)
    breakout_at = row["first_zone_breakout_minute"]
    rejection_at = row["first_zone_rejection_minute"]
    if breakout_at is None and rejection_at is None:
        return "unreached"
    if breakout_at is None:
        return "rejection"
    if rejection_at is None:
        return "breakout"
    if breakout_at == rejection_at:
        return "tie"
    return "breakout" if breakout_at < rejection_at else "rejection"


def first_true_position(values: Series) -> int | None:
    positions = np.flatnonzero(values.fillna(False).to_numpy(dtype=bool))
    return int(positions[0] + 1) if len(positions) else None


def volume_weighted_pressure(frame: DataFrame) -> float:
    bar_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    pressure = ((frame["close"] - frame["open"]) / bar_range).clip(-1.0, 1.0).fillna(0.0)
    volume = frame["volume"].clip(lower=0.0)
    denominator = float(volume.sum())
    return float((pressure * volume).sum() / denominator) if denominator > 0.0 else math.nan


def realized_volatility(frame: DataFrame) -> float:
    returns = frame["close"].pct_change(fill_method=None).dropna()
    return float(returns.std(ddof=0)) if not returns.empty else math.nan


def select_no_level_control(
    *,
    episode: Series,
    actual_contact_time: pd.Timestamp,
    actual_match: dict[str, float],
    surfaces: PairSurfaces,
    window_hours: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    anchor = str(episode["cluster_causal_anchor_timeframe"])
    pre_hours = int(window_hours[anchor][0])
    visible_start = pd.Timestamp(episode["event_time"]) - pd.Timedelta(hours=pre_hours)
    latest = actual_contact_time - pd.Timedelta(minutes=CONTROL_EVENT_SEPARATION_MINUTES)
    scale_source = surfaces.match.loc[
        (surfaces.match["date"] >= visible_start)
        & (surfaces.match["date"] <= latest)
    ].dropna(subset=list(MATCH_FEATURES))
    scales = robust_scales(scale_source[list(MATCH_FEATURES)].to_numpy(dtype=float))
    candidates = surfaces.match.loc[
        (surfaces.match["date"] >= visible_start)
        & (surfaces.match["date"] <= latest)
        & surfaces.match["date"].dt.minute.mod(5).eq(0)
    ].copy()
    candidates = candidates.dropna(subset=list(MATCH_FEATURES))
    actual_vector = np.asarray([actual_match[feature] for feature in MATCH_FEATURES], dtype=float)
    if not np.isfinite(actual_vector).all():
        raise ValueError(f"Actual match state is incomplete: {episode['episode_id']}")
    actual_trend_sign = int(np.sign(actual_match["match_return_60m"]))
    candidate_trend_sign = np.sign(candidates["match_return_60m"].to_numpy(dtype=float))
    candidates = candidates.loc[candidate_trend_sign == actual_trend_sign].copy()
    if candidates.empty:
        raise ValueError(f"No same-trend control candidates: {episode['episode_id']}")

    candidate_count_before_no_level = len(candidates)
    no_level_rows: list[int] = []
    for index, candidate in candidates.iterrows():
        count = causal_level_contact_count(
            timestamp=pd.Timestamp(candidate["date"]),
            candle=candidate,
            base=surfaces.base,
            aligned_levels=surfaces.aligned_levels,
        )
        if count == 0:
            no_level_rows.append(index)
    candidates = candidates.loc[no_level_rows].copy()
    candidate_count_after_no_level = len(candidates)
    if candidates.empty:
        raise ValueError(f"No causal no-level control candidates: {episode['episode_id']}")

    values = candidates[list(MATCH_FEATURES)].to_numpy(dtype=float)
    standardized = np.abs(values - actual_vector[None, :]) / scales[None, :]
    candidates["distance_mean"] = standardized.mean(axis=1)
    candidates["distance_max"] = standardized.max(axis=1)
    candidates = candidates.sort_values(["distance_mean", "distance_max", "date"])
    selected = candidates.iloc[0]
    # Recalculate by value so the audit remains correct after candidate sorting.
    selected_vector = np.asarray([selected[feature] for feature in MATCH_FEATURES], dtype=float)
    selected_distances = np.abs(selected_vector - actual_vector) / scales
    distance_mean = float(selected_distances.mean())
    distance_max = float(selected_distances.max())
    balance_usable = bool(
        distance_mean <= CONTROL_MEAN_DISTANCE_LIMIT
        and distance_max <= CONTROL_MAX_DISTANCE_LIMIT
    )
    selection_order = int(episode["sample_selection_order"])
    audit = [
        {
            "episode_id": str(episode["episode_id"]),
            "sample_selection_order": selection_order,
            "cohort": str(episode["cohort"]),
            "pair": str(episode["pair"]),
            "actual_contact_time": actual_contact_time,
            "control_time": pd.Timestamp(selected["date"]),
            "match_feature": feature,
            "actual_value": float(actual_vector[position]),
            "control_value": float(selected_vector[position]),
            "robust_scale": float(scales[position]),
            "absolute_robust_distance": float(selected_distances[position]),
            "control_distance_mean": distance_mean,
            "control_distance_max": distance_max,
            "control_balance_usable": balance_usable,
            "candidate_count_before_no_level_filter": int(candidate_count_before_no_level),
            "candidate_count_after_no_level_filter": int(candidate_count_after_no_level),
            "selected_causal_level_contact_count": 0,
        }
        for position, feature in enumerate(MATCH_FEATURES)
    ]
    return {
        "timestamp": pd.Timestamp(selected["date"]),
        "match_snapshot": {
            feature: finite_or_nan(selected[feature]) for feature in MATCH_FEATURES
        },
        "distance_mean": distance_mean,
        "distance_max": distance_max,
        "balance_usable": balance_usable,
        "match_audit": audit,
    }


def robust_scales(values: np.ndarray) -> np.ndarray:
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scales = q75 - q25
    standard = np.nanstd(values, axis=0)
    scales = np.where(scales > 1e-12, scales, standard)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return scales.astype(float)


def causal_level_contact_count(
    *,
    timestamp: pd.Timestamp,
    candle: Series,
    base: DataFrame,
    aligned_levels: list[Any],
) -> int:
    dates = pd.DatetimeIndex(pd.to_datetime(base["date"], utc=True))
    position = int(dates.searchsorted(timestamp.floor("h"), side="right") - 1)
    if position < 0 or position >= len(base):
        return 0
    base_atr = finite_or_nan(base.iloc[position].get("base_atr"))
    if not np.isfinite(base_atr) or base_atr <= 0.0:
        return 0
    high = float(candle["high"])
    low = float(candle["low"])
    contacts = 0
    for item in aligned_levels:
        if not bool(item.valid[position]):
            continue
        available = pd.Timestamp(item.source_available.iloc[position])
        if available > timestamp:
            continue
        level = finite_or_nan(item.level[position])
        if not np.isfinite(level) or level <= 0.0:
            continue
        half_width = max(0.5 * base_atr, abs(level) * 0.0005)
        if high >= level - half_width and low <= level + half_width:
            contacts += 1
    return contacts


def boundary_audit(
    *,
    episode: Series,
    contact_time: pd.Timestamp,
    minute: DataFrame,
    window_hours: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    anchor = str(episode["cluster_causal_anchor_timeframe"])
    pre_hours, post_hours = window_hours[anchor]
    event_time = pd.Timestamp(episode["event_time"])
    visible_start = event_time - pd.Timedelta(hours=pre_hours)
    visible_end = event_time + pd.Timedelta(hours=post_hours + 1)
    selected = minute.loc[
        (minute["date"] >= visible_start) & (minute["date"] < visible_end)
    ].copy()
    expected = int((visible_end - visible_start).total_seconds() // 60)
    if len(selected) != expected:
        raise ValueError(f"Boundary window incomplete for {episode['episode_id']}.")
    reference = float(episode["level_price"])
    half_width = float(episode["zone_half_width"])
    zone = selected["high"].ge(reference - half_width) & selected["low"].le(
        reference + half_width
    )
    first = selected.iloc[:60]
    last = selected.iloc[-60:]
    pre = selected.loc[
        (selected["date"] >= contact_time - pd.Timedelta(minutes=60))
        & (selected["date"] < contact_time)
    ]
    first_zone_fraction = float(zone.iloc[:60].mean())
    last_zone_fraction = float(zone.iloc[-60:].mean())
    completed_pre = selected.loc[selected["date"] < contact_time].copy()
    baseline_hourly_volume = completed_pre.assign(
        hour=completed_pre["date"].dt.floor("h")
    ).groupby("hour")["volume"].mean().median()
    first_volume_ratio = (
        float(first["volume"].mean() / baseline_hourly_volume)
        if baseline_hourly_volume > 0.0
        else math.nan
    )
    pre_volume = float(pre["volume"].mean())
    last_volume_ratio = (
        float(last["volume"].mean() / pre_volume) if pre_volume > 0.0 else math.nan
    )
    pre_volatility = realized_volatility(pre)
    last_volatility = realized_volatility(last)
    last_volatility_ratio = (
        last_volatility / pre_volatility if pre_volatility > 0.0 else math.nan
    )
    first_minimum_distance_atr = float(
        (first["close"] - reference).abs().min() / float(episode["base_atr"])
    )
    left_near_level = bool(first_minimum_distance_atr <= 2.0)
    flags = {
        "left_edge_already_in_zone": first_zone_fraction > BOUNDARY_ZONE_FRACTION_LIMIT,
        "left_edge_activity_already_elevated": (
            left_near_level
            and
            np.isfinite(first_volume_ratio)
            and first_volume_ratio > BOUNDARY_ACTIVITY_RATIO_LIMIT
        ),
        "right_edge_still_in_zone": last_zone_fraction > BOUNDARY_ZONE_FRACTION_LIMIT,
        "right_edge_volume_not_settled": (
            np.isfinite(last_volume_ratio)
            and last_volume_ratio > BOUNDARY_ACTIVITY_RATIO_LIMIT
        ),
        "right_edge_volatility_not_settled": (
            np.isfinite(last_volatility_ratio)
            and last_volatility_ratio > BOUNDARY_ACTIVITY_RATIO_LIMIT
        ),
    }
    return {
        "episode_id": str(episode["episode_id"]),
        "sample_selection_order": int(episode["sample_selection_order"]),
        "cohort": str(episode["cohort"]),
        "pair": str(episode["pair"]),
        "anchor_timeframe": anchor,
        "contact_time": contact_time,
        "visible_start": visible_start,
        "visible_end_exclusive": visible_end,
        "visible_rows": len(selected),
        "first_hour_zone_fraction": first_zone_fraction,
        "last_hour_zone_fraction": last_zone_fraction,
        "first_hour_minimum_distance_atr": first_minimum_distance_atr,
        "first_hour_near_level_within_2atr": left_near_level,
        "first_hour_volume_to_precontact_median": first_volume_ratio,
        "last_hour_volume_to_precontact_hour": last_volume_ratio,
        "last_hour_volatility_to_precontact_hour": last_volatility_ratio,
        **flags,
        "boundary_extension_indicated": bool(any(flags.values())),
    }


def load_orderbook_surface() -> DataFrame:
    if not ORDERBOOK_PATH.is_file():
        return DataFrame()
    columns = [
        "date",
        "source_max_ts",
        "obts_feature_present",
        "obts_coverage_ratio",
        "obts_spread_bps_mean",
        "obts_microprice_offset_bps_last",
        "obts_pressure_25bps_mean",
        "obts_pressure_25bps_last",
        "obts_pressure_25bps_slope",
        "obts_imbalance_25bps_last",
        "obts_resistance_zone_distance_bps",
        "obts_support_zone_distance_bps",
        "obts_volatility_expansion_state",
    ]
    frame = pd.read_parquet(ORDERBOOK_PATH, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["source_max_ts"] = pd.to_datetime(frame["source_max_ts"], utc=True)
    return frame.sort_values("date").reset_index(drop=True)


def orderbook_snapshot(orderbook: DataFrame, timestamp: pd.Timestamp) -> dict[str, Any]:
    empty = {
        "btc_orderbook_available": False,
        "btc_orderbook_usable_coverage": False,
        "btc_orderbook_age_hours": math.nan,
        "btc_orderbook_coverage_ratio": math.nan,
        "btc_orderbook_spread_bps_mean": math.nan,
        "btc_orderbook_microprice_offset_bps_last": math.nan,
        "btc_orderbook_pressure_25bps_mean": math.nan,
        "btc_orderbook_pressure_25bps_last": math.nan,
        "btc_orderbook_pressure_25bps_slope": math.nan,
        "btc_orderbook_imbalance_25bps_last": math.nan,
        "btc_orderbook_resistance_distance_bps": math.nan,
        "btc_orderbook_support_distance_bps": math.nan,
        "btc_orderbook_volatility_expansion_state": math.nan,
    }
    if orderbook.empty:
        return empty
    eligible = orderbook.loc[
        (orderbook["date"] <= timestamp)
        & (orderbook["source_max_ts"] <= timestamp)
        & orderbook["obts_feature_present"].eq(1.0)
    ]
    if eligible.empty:
        return empty
    row = eligible.iloc[-1]
    age_hours = float((timestamp - pd.Timestamp(row["date"])).total_seconds() / 3600.0)
    if age_hours > 2.0:
        return empty
    coverage = finite_or_nan(row["obts_coverage_ratio"])
    return {
        "btc_orderbook_available": True,
        "btc_orderbook_usable_coverage": bool(np.isfinite(coverage) and coverage >= 0.80),
        "btc_orderbook_age_hours": age_hours,
        "btc_orderbook_coverage_ratio": coverage,
        "btc_orderbook_spread_bps_mean": finite_or_nan(row["obts_spread_bps_mean"]),
        "btc_orderbook_microprice_offset_bps_last": finite_or_nan(
            row["obts_microprice_offset_bps_last"]
        ),
        "btc_orderbook_pressure_25bps_mean": finite_or_nan(
            row["obts_pressure_25bps_mean"]
        ),
        "btc_orderbook_pressure_25bps_last": finite_or_nan(
            row["obts_pressure_25bps_last"]
        ),
        "btc_orderbook_pressure_25bps_slope": finite_or_nan(
            row["obts_pressure_25bps_slope"]
        ),
        "btc_orderbook_imbalance_25bps_last": finite_or_nan(
            row["obts_imbalance_25bps_last"]
        ),
        "btc_orderbook_resistance_distance_bps": finite_or_nan(
            row["obts_resistance_zone_distance_bps"]
        ),
        "btc_orderbook_support_distance_bps": finite_or_nan(
            row["obts_support_zone_distance_bps"]
        ),
        "btc_orderbook_volatility_expansion_state": finite_or_nan(
            row["obts_volatility_expansion_state"]
        ),
    }


def feature_catalog(events: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe, prefix in TIMEFRAME_PREFIX.items():
        for feature, definition in TECHNICAL_DEFINITIONS.items():
            rows.append(
                {
                    "feature": f"{prefix}__{feature}",
                    "category": "indicator_cross"
                    if feature.startswith("bars_since_")
                    else indicator_category(feature),
                    "source_timeframe": timeframe,
                    "relationship_or_definition": definition,
                    "availability": "last fully completed source bar before the event",
                    "used_in_no_level_matching": False,
                    "screened_against_outcomes": True,
                }
            )
    alignment_definitions = {
        "mtf_rsi_above_50_count": "number of 1m, 5m, 15m, 1h, 4h, and 8h RSI values above 50",
        "mtf_macd_positive_count": "number of the six source timeframes with a positive MACD histogram",
        "mtf_close_above_sma20_count": "number of the six source timeframes closing above SMA(20)",
        "mtf_ema12_above_ema26_count": "number of the six source timeframes with EMA(12) above EMA(26)",
        "mtf_positive_pressure_count": "number of the six source timeframes with positive twenty-bar body pressure",
        "mtf_elevated_volume_count": "number of the six source timeframes with volume above its twenty-bar mean",
        "mtf_adx_above_25_count": "number of the six source timeframes with ADX at or above 25",
    }
    for feature, definition in alignment_definitions.items():
        rows.append(
            {
                "feature": feature,
                "category": "cross_timeframe_alignment",
                "source_timeframe": "1m+5m+15m+1h+4h+8h",
                "relationship_or_definition": definition,
                "availability": "last fully completed bar on every included timeframe",
                "used_in_no_level_matching": False,
                "screened_against_outcomes": True,
            }
        )
    match_definitions = {
        "match_return_15m": "price change over the causal fifteen minutes before contact",
        "match_return_60m": "price change over the causal sixty minutes before contact",
        "match_return_240m": "price change over the causal four hours before contact",
        "match_realized_volatility_60m": "one-minute return volatility during the causal preceding hour",
        "match_atr14_pct": "one-minute ATR(14) divided by price immediately before contact",
        "match_volume_ratio_20m": "last completed minute volume divided by its trailing twenty-minute mean",
        "match_pressure_20m": "twenty-minute volume-weighted candle-body pressure before contact",
        "match_range_position_240m": "price position inside the causal preceding four-hour range",
    }
    for feature, definition in match_definitions.items():
        rows.append(
            {
                "feature": feature,
                "category": "no_level_matching_state",
                "source_timeframe": "1m",
                "relationship_or_definition": definition,
                "availability": "completed one-minute candles strictly before the event",
                "used_in_no_level_matching": True,
                "screened_against_outcomes": True,
            }
        )
    cluster_definitions = {
        "parent_high_thinness_score": "causal thinness score of the frozen parent one-hour LVN",
        "cluster_primary_peer_level_count": "number of peer lines in the connected contacted cluster",
        "cluster_primary_dependency_group_count": "number of independent peer mechanisms, not line aliases",
        "cluster_primary_timeframe_count": "number of distinct peer source timeframes",
        "cluster_primary_span_atr": "price span of the connected cluster divided by parent-hour ATR",
        "cluster_parent_to_peer_median_distance_atr": "median parent-to-peer distance divided by parent-hour ATR",
        "cluster_price_average_family_count": "peer lines from moving-average or price-average mechanisms",
        "cluster_rolling_price_extreme_count": "peer lines from rolling high/low mechanisms",
        "cluster_round_number_count": "peer lines from round-number mechanisms",
        "cluster_peer_1h_count": "connected peer lines sourced from 1h",
        "cluster_peer_4h_count": "connected peer lines sourced from 4h",
        "cluster_peer_8h_count": "connected peer lines sourced from 8h",
    }
    for feature, definition in cluster_definitions.items():
        rows.append(
            {
                "feature": feature,
                "category": "level_cluster_geometry",
                "source_timeframe": "1h+4h+8h as recorded per episode",
                "relationship_or_definition": definition,
                "availability": "causally frozen no later than the parent contact hour",
                "used_in_no_level_matching": False,
                "screened_against_outcomes": True,
            }
        )
    for column in events.columns:
        if column.startswith("parent_state_"):
            rows.append(
                {
                    "feature": column,
                    "category": "parent_hour_market_state",
                    "source_timeframe": "1h or named wider state",
                    "relationship_or_definition": column.removeprefix("parent_state_").replace(
                        "_", " "
                    ),
                    "availability": "inherited from the corrected causal G3A event state",
                    "used_in_no_level_matching": False,
                    "screened_against_outcomes": True,
                }
            )
    orderbook_definitions = {
        "btc_orderbook_coverage_ratio": "fraction of expected Bybit BTC linear orderbook minutes present in the last completed hour",
        "btc_orderbook_spread_bps_mean": "mean BTC orderbook spread in basis points",
        "btc_orderbook_microprice_offset_bps_last": "latest BTC microprice offset from mid price in basis points",
        "btc_orderbook_pressure_25bps_mean": "mean BTC bid-versus-ask pressure within 25 basis points",
        "btc_orderbook_pressure_25bps_last": "latest BTC bid-versus-ask pressure within 25 basis points",
        "btc_orderbook_pressure_25bps_slope": "within-hour change in BTC 25-basis-point pressure",
        "btc_orderbook_imbalance_25bps_last": "latest BTC depth imbalance within 25 basis points",
        "btc_orderbook_resistance_distance_bps": "BTC distance to its current orderbook resistance zone",
        "btc_orderbook_support_distance_bps": "BTC distance to its current orderbook support zone",
        "btc_orderbook_volatility_expansion_state": "causal BTC orderbook-derived volatility expansion state",
    }
    for feature, definition in orderbook_definitions.items():
        rows.append(
            {
                "feature": feature,
                "category": "external_btc_orderbook_context",
                "source_timeframe": "completed Bybit BTC 1h",
                "relationship_or_definition": definition,
                "availability": "only when source_max_ts is not later than event and age is at most two hours",
                "used_in_no_level_matching": False,
                "screened_against_outcomes": True,
            }
        )
    catalog = DataFrame(rows).drop_duplicates("feature").sort_values(
        ["category", "source_timeframe", "feature"]
    )
    return catalog.reset_index(drop=True)


def indicator_category(feature: str) -> str:
    if "volume" in feature or feature in {"cmf20", "obv_trend20"}:
        return "volume_or_participation"
    if "pressure" in feature or feature == "di_spread14":
        return "pressure"
    if "atr" in feature or "volatility" in feature or "bollinger_width" in feature:
        return "volatility"
    if "bollinger" in feature:
        return "bollinger"
    if "rsi" in feature:
        return "rsi"
    if "macd" in feature:
        return "macd"
    if "sma" in feature or "ema" in feature:
        return "moving_average"
    if "range_position" in feature:
        return "price_location"
    if "return" in feature:
        return "price_direction"
    if "adx" in feature:
        return "trend_strength"
    return "other_indicator"


def relationship_screen(actual: DataFrame, catalog: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    catalog_by_feature = catalog.set_index("feature")
    for cohort, subset in (
        ("all", actual),
        ("normal", actual.loc[actual["cohort"].eq("normal")]),
        ("meme", actual.loc[actual["cohort"].eq("meme")]),
    ):
        for feature in catalog["feature"].astype(str):
            if feature not in subset or not pd.api.types.is_numeric_dtype(subset[feature]):
                continue
            for outcome in PRIMARY_OUTCOMES:
                paired = subset[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(paired) < MINIMUM_SCREEN_ROWS:
                    correlation = math.nan
                    leave_one_out_fraction = math.nan
                elif paired[feature].nunique() < 2 or paired[outcome].nunique() < 2:
                    correlation = math.nan
                    leave_one_out_fraction = math.nan
                else:
                    correlation = float(paired[feature].corr(paired[outcome], method="spearman"))
                    leave_one_out_fraction = leave_one_out_sign_fraction(
                        paired[feature], paired[outcome], correlation
                    )
                meta = catalog_by_feature.loc[feature]
                rows.append(
                    {
                        "cohort": cohort,
                        "feature": feature,
                        "category": str(meta["category"]),
                        "source_timeframe": str(meta["source_timeframe"]),
                        "outcome": outcome,
                        "independent_episode_count": int(len(paired)),
                        "feature_unique_values": int(paired[feature].nunique()),
                        "outcome_unique_values": int(paired[outcome].nunique()),
                        "spearman_correlation": correlation,
                        "leave_one_out_same_sign_fraction": leave_one_out_fraction,
                        "exploratory_cross_cohort_queue_signal": False,
                    }
                )
    screen = DataFrame(rows)
    for (feature, outcome), group in screen.loc[
        screen["cohort"].isin(["normal", "meme"])
    ].groupby(["feature", "outcome"], sort=False):
        if set(group["cohort"]) != {"normal", "meme"}:
            continue
        normal = group.loc[group["cohort"].eq("normal")].iloc[0]
        meme = group.loc[group["cohort"].eq("meme")].iloc[0]
        correlations = [
            finite_or_nan(normal["spearman_correlation"]),
            finite_or_nan(meme["spearman_correlation"]),
        ]
        stability = [
            finite_or_nan(normal["leave_one_out_same_sign_fraction"]),
            finite_or_nan(meme["leave_one_out_same_sign_fraction"]),
        ]
        queue = bool(
            int(normal["independent_episode_count"]) >= MINIMUM_QUEUE_ROWS_PER_COHORT
            and int(meme["independent_episode_count"]) >= MINIMUM_QUEUE_ROWS_PER_COHORT
            and
            np.isfinite(correlations).all()
            and np.sign(correlations[0]) == np.sign(correlations[1])
            and min(abs(value) for value in correlations) >= 0.40
            and np.isfinite(stability).all()
            and min(stability) >= 0.80
        )
        screen.loc[
            screen["feature"].eq(feature) & screen["outcome"].eq(outcome),
            "exploratory_cross_cohort_queue_signal",
        ] = queue
    return screen.sort_values(
        ["exploratory_cross_cohort_queue_signal", "outcome", "cohort", "feature"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def leave_one_out_sign_fraction(left: Series, right: Series, full: float) -> float:
    if not np.isfinite(full) or full == 0.0 or len(left) <= 3:
        return math.nan
    signs: list[bool] = []
    for position in range(len(left)):
        keep = np.arange(len(left)) != position
        candidate_left = left.iloc[keep]
        candidate_right = right.iloc[keep]
        if candidate_left.nunique() < 2 or candidate_right.nunique() < 2:
            continue
        value = candidate_left.corr(candidate_right, method="spearman")
        if np.isfinite(value) and value != 0.0:
            signs.append(bool(np.sign(value) == np.sign(full)))
    return float(np.mean(signs)) if signs else math.nan


def control_comparisons(events: DataFrame) -> DataFrame:
    metrics = (
        "reaction_60m",
        "first_path_away",
        "first_path_through",
        "first_zone_rejection",
        "first_zone_breakout",
        "excursion_strength_60_half_widths",
        "volume_ratio_post_pre_60m",
        "pressure_change_60m",
        "pressure_change_toward_through_60m",
        "pressure_change_toward_away_60m",
        "realized_volatility_ratio_60m",
        "zone_overlap_fraction_60m",
        "reference_crossings_60m",
        "through_excursion_60_half_widths",
        "away_excursion_60_half_widths",
        "breakout_excursion_beyond_far_edge_60_half_widths",
        "rejection_excursion_beyond_entry_edge_60_half_widths",
    )
    rows: list[dict[str, Any]] = []
    for cohort, subset in (
        ("all", events),
        ("normal", events.loc[events["cohort"].eq("normal")]),
        ("meme", events.loc[events["cohort"].eq("meme")]),
    ):
        actual = subset.loc[subset["event_kind"].eq("actual_cluster_contact")]
        control = subset.loc[
            subset["event_kind"].eq("same_state_no_level")
            & subset["control_balance_usable"].eq(True)
        ]
        shared = sorted(set(actual["episode_id"]).intersection(control["episode_id"]))
        actual = actual.loc[actual["episode_id"].isin(shared)].set_index("episode_id")
        control = control.loc[control["episode_id"].isin(shared)].set_index("episode_id")
        for metric in metrics:
            left = pd.to_numeric(actual[metric], errors="coerce").astype(float)
            right = pd.to_numeric(control[metric], errors="coerce").astype(float)
            paired = pd.concat([left.rename("actual"), right.rename("control")], axis=1).dropna()
            if metric in {
                "reaction_60m",
                "first_path_away",
                "first_path_through",
                "first_zone_rejection",
                "first_zone_breakout",
            }:
                actual_value = float(paired["actual"].mean()) if len(paired) else math.nan
                control_value = float(paired["control"].mean()) if len(paired) else math.nan
                statistic = "paired_rate"
            else:
                actual_value = float(paired["actual"].median()) if len(paired) else math.nan
                control_value = float(paired["control"].median()) if len(paired) else math.nan
                statistic = "paired_median"
            differences = paired["actual"] - paired["control"]
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "statistic": statistic,
                    "balanced_pair_count": int(len(paired)),
                    "actual_value": actual_value,
                    "same_state_no_level_value": control_value,
                    "actual_minus_control": actual_value - control_value,
                    "paired_difference_median": float(differences.median())
                    if len(differences)
                    else math.nan,
                    "paired_positive_fraction": float((differences > 0.0).mean())
                    if len(differences)
                    else math.nan,
                }
            )
    return DataFrame(rows)


def summarize_analysis(
    *,
    actual: DataFrame,
    controls: DataFrame,
    boundaries: DataFrame,
    screen: DataFrame,
) -> dict[str, Any]:
    cohort_summaries: dict[str, Any] = {}
    reaction_comparison_supported = True
    for cohort in ("normal", "meme"):
        left = actual.loc[actual["cohort"].eq(cohort)].copy()
        right = controls.loc[
            controls["cohort"].eq(cohort) & controls["control_balance_usable"].eq(True)
        ].copy()
        shared = sorted(set(left["episode_id"]).intersection(right["episode_id"]))
        paired_left = left.loc[left["episode_id"].isin(shared)]
        paired_right = right.loc[right["episode_id"].isin(shared)]
        callable_rows = left.loc[left["direction_callable"].eq(True)]
        direction_counts = callable_rows["first_direction"].value_counts().to_dict()
        majority_accuracy = (
            max(direction_counts.values()) / len(callable_rows)
            if direction_counts and len(callable_rows)
            else math.nan
        )
        actual_reaction_rate = float(paired_left["reaction_60m"].mean()) if shared else math.nan
        control_reaction_rate = (
            float(paired_right["reaction_60m"].mean()) if shared else math.nan
        )
        diagnostic_support = bool(
            len(shared) >= 4
            and int(paired_left["reaction_60m"].sum()) >= 4
            and actual_reaction_rate > control_reaction_rate
        )
        reaction_comparison_supported &= diagnostic_support
        away_callable = left.loc[left["first_path_relative_to_approach"].isin(["away", "through"])]
        paired_control_away_rate = (
            float(paired_right["first_path_away"].mean()) if shared else math.nan
        )
        paired_actual_away_rate = (
            float(paired_left["first_path_away"].mean()) if shared else math.nan
        )
        retrospective_away_joint = left["reaction_60m"] & left["first_path_away"]
        zone_resolution_rows = left.loc[
            left["first_zone_resolution"].isin(["breakout", "rejection"])
        ]
        cohort_summaries[cohort] = {
            "actual_episode_count": int(len(left)),
            "balanced_control_count": int(len(right)),
            "actual_reaction_count": int(left["reaction_60m"].sum()),
            "actual_reaction_rate": float(left["reaction_60m"].mean()),
            "balanced_paired_actual_reaction_rate": actual_reaction_rate,
            "balanced_same_state_no_level_reaction_rate": control_reaction_rate,
            "median_actual_excursion_half_widths": float(
                left["excursion_strength_60_half_widths"].median()
            ),
            "median_actual_volume_ratio": float(left["volume_ratio_post_pre_60m"].median()),
            "first_direction_counts": {str(key): int(value) for key, value in direction_counts.items()},
            "direction_callable_count": int(len(callable_rows)),
            "simple_trend_correct_count": int(
                callable_rows["simple_trend_direction_correct"].sum()
            ),
            "simple_trend_accuracy_when_direction_callable": (
                float(callable_rows["simple_trend_direction_correct"].mean())
                if len(callable_rows)
                else math.nan
            ),
            "simple_trend_joint_reaction_and_direction_count": int(
                left["simple_trend_joint_reaction_and_direction"].sum()
            ),
            "simple_trend_joint_rate_all_events": float(
                left["simple_trend_joint_reaction_and_direction"].mean()
            ),
            "retrospective_majority_direction_accuracy": majority_accuracy,
            "first_path_counts": {
                str(key): int(value)
                for key, value in left["first_path_relative_to_approach"].value_counts().items()
            },
            "retrospective_away_call_issued_count": int(len(away_callable)),
            "retrospective_away_call_abstention_count": int(len(left) - len(away_callable)),
            "retrospective_away_call_conditional_direction_rate": (
                float(away_callable["first_path_away"].mean())
                if len(away_callable)
                else math.nan
            ),
            "retrospective_away_call_joint_reaction_and_direction_count": int(
                retrospective_away_joint.sum()
            ),
            "retrospective_away_call_joint_rate_all_events": float(
                retrospective_away_joint.mean()
            ),
            "balanced_paired_actual_away_rate": paired_actual_away_rate,
            "balanced_same_state_no_level_away_rate": paired_control_away_rate,
            "first_zone_resolution_counts": {
                str(key): int(value)
                for key, value in left["first_zone_resolution"].value_counts().items()
            },
            "zone_resolution_callable_count": int(len(zone_resolution_rows)),
            "zone_rejection_rate_when_resolved": (
                float(zone_resolution_rows["first_zone_rejection"].mean())
                if len(zone_resolution_rows)
                else math.nan
            ),
            "balanced_paired_actual_zone_rejection_rate": (
                float(paired_left["first_zone_rejection"].mean()) if shared else math.nan
            ),
            "balanced_same_state_no_level_zone_rejection_rate": (
                float(paired_right["first_zone_rejection"].mean()) if shared else math.nan
            ),
            "btc_orderbook_available_count": int(left["btc_orderbook_available"].sum()),
            "btc_orderbook_usable_coverage_count": int(
                left["btc_orderbook_usable_coverage"].sum()
            ),
            "diagnostic_reaction_comparison_supported": diagnostic_support,
        }
    queue_rows = screen.loc[
        screen["exploratory_cross_cohort_queue_signal"].eq(True)
        & screen["cohort"].eq("all")
    ].copy()
    queue_signals = [
        {
            "feature": str(row.feature),
            "category": str(row.category),
            "source_timeframe": str(row.source_timeframe),
            "outcome": str(row.outcome),
        }
        for row in queue_rows.itertuples()
    ]
    return {
        "status": "diagnostic_complete",
        "cohorts": cohort_summaries,
        "reaction_comparison_supported_in_both_cohorts": reaction_comparison_supported,
        "boundary_extension_indicated_count": int(
            boundaries["boundary_extension_indicated"].sum()
        ),
        "boundary_flag_counts": {
            column: int(boundaries[column].sum())
            for column in (
                "left_edge_already_in_zone",
                "left_edge_activity_already_elevated",
                "right_edge_still_in_zone",
                "right_edge_volume_not_settled",
                "right_edge_volatility_not_settled",
            )
        },
        "screened_feature_count": int(screen["feature"].nunique()),
        "screened_feature_outcome_pairs": int(
            screen[["feature", "outcome"]].drop_duplicates().shape[0]
        ),
        "exploratory_cross_cohort_queue_signal_count": len(queue_signals),
        "exploratory_cross_cohort_queue_signals": queue_signals,
        "interpretation_limits": [
            "Twelve frozen episodes are a diagnostic sample, not chronological confirmation.",
            "The technical relationship screen is deliberately exhaustive for logging but is not multiple-testing-adjusted evidence.",
            "The retrospective majority direction is an optimistic comparator, not a deployable rule.",
            "The corrected post-contact sample does not show a universal away-first path; any narrower directional pattern must be frozen before a larger confirmation batch.",
            "Bybit orderbook values describe global BTC context and are not local orderbooks for the sampled altcoins.",
            "No FreqAI model, profit target, entry rule, exit rule, or indicator parameter was fitted.",
        ],
    }


def write_aligned_path_plot(paths: DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, cohort in zip(axes, ("normal", "meme"), strict=True):
        cohort_paths = paths.loc[
            paths["cohort"].eq(cohort) & paths["checkpoint_minutes"].le(120)
        ]
        for event_kind, style, alpha in (
            ("same_state_no_level", "--", 0.35),
            ("actual_cluster_contact", "-", 0.70),
        ):
            selected = cohort_paths.loc[cohort_paths["event_kind"].eq(event_kind)]
            for _, group in selected.groupby("episode_id", sort=False):
                axis.plot(
                    group["checkpoint_minutes"],
                    group["close_displacement_through_positive_half_widths"],
                    linestyle=style,
                    linewidth=1.0,
                    alpha=alpha,
                    color="#1f77b4" if event_kind == "actual_cluster_contact" else "#888888",
                )
            medians = selected.groupby("checkpoint_minutes")[
                "close_displacement_through_positive_half_widths"
            ].median()
            axis.plot(
                medians.index,
                medians.values,
                linestyle=style,
                linewidth=2.6,
                color="#d62728" if event_kind == "actual_cluster_contact" else "#222222",
                label="actual median" if event_kind == "actual_cluster_contact" else "no-level median",
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(f"{cohort.capitalize()} cohort (six frozen episodes)")
        axis.set_xlabel("Minutes after actual or pseudo-contact")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Close displacement: positive through, negative away (zone half-widths)")
    figure.suptitle("G3G aligned one-minute replay: actual clusters versus same-state no-level times")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if np.isfinite(numeric) else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
