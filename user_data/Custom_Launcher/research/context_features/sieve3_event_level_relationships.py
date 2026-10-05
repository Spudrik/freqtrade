from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from general_exit_level_reaction_study import level_prices, load_cache
from sieve3_event_reaction_research import (
    DEFAULT_DIRECT_HORIZONS,
    actual_frame,
    atomic_json,
    atomic_parquet,
    family_for_entry,
    oriented_outcomes,
    parse_horizons,
    parse_windows,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_EVENT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation0"
)
DEFAULT_LEVEL_CACHE = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "general_exit_level_reaction_cache"
)
BASELINE_PAIR_TO_CACHE_KEY = {
    "BTC/USDT:USDT": "BTC",
    "ETH/USDT:USDT": "ETH",
    "SOL/USDT:USDT": "SOL",
}
TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "8h": 8, "1d": 24, "3d": 72}


def parse_bands(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(token.strip()) for token in raw.split(",") if token.strip()}))
    if not values or any(value <= 0.0 or value > 0.10 for value in values):
        raise ValueError("Proximity bands must be within (0, 0.10]")
    return values


def parse_pairs(raw: str) -> dict[str, str]:
    pairs = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not pairs:
        raise ValueError("At least one level-reaction pair is required")
    output: dict[str, str] = {}
    for pair in pairs:
        if "/" not in pair:
            raise ValueError(f"Invalid pair: {pair!r}")
        base = pair.split("/", 1)[0].split(":", 1)[0]
        if not base.replace("-", "").isalnum():
            raise ValueError(f"Invalid pair base: {pair!r}")
        output[pair] = base
    return output


def cache_file(cache_key: str) -> str:
    return f"{cache_key.lower()}_general_exit_levels_5tf.parquet"


def _level_state(
    events: DataFrame,
    levels: DataFrame,
    *,
    bands: tuple[float, ...],
    evidence_source: str,
) -> DataFrame:
    merged = events.merge(levels, on=["pair", "decision_time"], how="inner", validate="many_to_one")
    if merged.empty:
        return DataFrame()
    price = pd.to_numeric(merged["level_state_close"], errors="coerce").replace(0.0, np.nan)
    is_long = merged["entry_side"].astype(str).str.lower().eq("long").to_numpy()
    support_columns = [column for column in merged if column.startswith("support__")]
    resistance_columns = [column for column in merged if column.startswith("resistance__")]
    support = merged[support_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    resistance = merged[resistance_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    price_values = price.to_numpy(dtype="float64")[:, None]
    support_distance = np.abs(support / price_values - 1.0)
    resistance_distance = np.abs(resistance / price_values - 1.0)
    support_valid = support <= price_values
    resistance_valid = resistance >= price_values
    support_tf = np.array(
        [TIMEFRAME_HOURS[column.split("__", 1)[1].split(":", 1)[0]] for column in support_columns],
        dtype="float64",
    )
    resistance_tf = np.array(
        [TIMEFRAME_HOURS[column.split("__", 1)[1].split(":", 1)[0]] for column in resistance_columns],
        dtype="float64",
    )
    rows: list[DataFrame] = []
    for band in bands:
        support_near = np.isfinite(support_distance) & (support_distance <= band)
        resistance_near = np.isfinite(resistance_distance) & (resistance_distance <= band)
        support_valid_near = support_near & support_valid
        resistance_valid_near = resistance_near & resistance_valid
        same_near = np.where(is_long[:, None], support_valid_near, resistance_valid_near)
        opposing_near = np.where(is_long[:, None], resistance_valid_near, support_valid_near)
        same_tf_matrix = np.where(
            is_long[:, None],
            np.where(support_valid_near, support_tf, np.nan),
            np.where(resistance_valid_near, resistance_tf, np.nan),
        )
        opposing_tf_matrix = np.where(
            is_long[:, None],
            np.where(resistance_valid_near, resistance_tf, np.nan),
            np.where(support_valid_near, support_tf, np.nan),
        )
        same_count = same_near.sum(axis=1)
        opposing_count = opposing_near.sum(axis=1)
        any_near = support_near.any(axis=1) | resistance_near.any(axis=1)
        same_max_tf = np.where(
            same_count > 0,
            np.nan_to_num(same_tf_matrix, nan=0.0).max(axis=1),
            np.nan,
        )
        opposing_max_tf = np.where(
            opposing_count > 0,
            np.nan_to_num(opposing_tf_matrix, nan=0.0).max(axis=1),
            np.nan,
        )
        state = np.select(
            [
                (same_count > 0) & (opposing_count > 0),
                same_count > 0,
                opposing_count > 0,
                any_near,
            ],
            ["mixed", "same_direction_only", "opposing_only", "crossed_or_displaced_only"],
            default="no_nearby_level",
        )
        precedence = np.select(
            [
                (same_count > 0) & (opposing_count == 0),
                (opposing_count > 0) & (same_count == 0),
                (same_count > 0) & (opposing_count > 0) & (same_max_tf > opposing_max_tf),
                (same_count > 0) & (opposing_count > 0) & (opposing_max_tf > same_max_tf),
                (same_count > 0) & (opposing_count > 0) & (same_max_tf == opposing_max_tf),
            ],
            ["same_only", "opposing_only", "same_direction_higher_tf", "opposing_higher_tf", "equal_highest_tf"],
            default="none_or_displaced",
        )
        out = merged.copy()
        out["evidence_source"] = evidence_source
        out["proximity_band_pct"] = float(band)
        out["relationship_state"] = state
        out["mtf_precedence_state"] = precedence
        out["same_direction_level_count"] = same_count
        out["opposing_level_count"] = opposing_count
        out["same_direction_max_tf_hours"] = same_max_tf
        out["opposing_max_tf_hours"] = opposing_max_tf
        out["support_level_count"] = support_near.sum(axis=1)
        out["resistance_level_count"] = resistance_near.sum(axis=1)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _summary(states: DataFrame, *, horizons: tuple[int, ...]) -> DataFrame:
    grouping = [
        "evidence_source",
        "entry_id",
        "family",
        "entry_side",
        "window",
        "pair_scope",
        "proximity_band_pct",
        "relationship_state",
        "mtf_precedence_state",
    ]
    scopes = [states.assign(pair_scope=states["pair"]), states.assign(pair_scope="pooled")]
    scoped_states = pd.concat(scopes, ignore_index=True)
    summaries: list[DataFrame] = []
    metric_names = (
        "signed_return",
        "best_move_if_held_atr",
        "worst_move_if_held_atr",
        "path_balance_atr",
        "reaction_magnitude_atr",
        "volume_ratio",
        "pressure_alignment",
        "volatility_ratio",
        "target_before_invalidation",
        "first_1atr_touch_step",
    )
    for horizon in horizons:
        enriched = scoped_states.copy()
        for side in ("long", "short"):
            mask = enriched["entry_side"].astype(str).str.lower().eq(side)
            if not mask.any():
                continue
            outcomes = oriented_outcomes(enriched.loc[mask], side, horizon)
            for metric in metric_names:
                enriched.loc[mask, metric] = pd.to_numeric(
                    outcomes[metric], errors="coerce"
                ).to_numpy()
        grouped = enriched.groupby(grouping, dropna=False, observed=True)
        summary = grouped[list(metric_names)].agg(["count", "mean", "median"])
        summary.columns = [
            f"{metric}_{'rows' if statistic == 'count' else statistic}"
            for metric, statistic in summary.columns
        ]
        summary = summary.reset_index()
        summary["event_rows"] = grouped.size().to_numpy()
        summary["horizon_hours"] = int(horizon)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else DataFrame()


def build_level_relationships(
    *,
    event_output_dir: Path,
    level_cache_dir: Path,
    output_dir: Path,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    horizons: tuple[int, ...],
    bands: tuple[float, ...],
    placebo_shift_hours: int,
    pair_to_cache_key: dict[str, str] | None = None,
) -> dict[str, Any]:
    pair_to_cache_key = pair_to_cache_key or BASELINE_PAIR_TO_CACHE_KEY
    entries_path = event_output_dir / "cache" / "entry_events_long.parquet"
    if not entries_path.exists():
        raise FileNotFoundError(entries_path)
    entries = pd.read_parquet(entries_path)
    entries["decision_time"] = pd.to_datetime(entries["decision_time"], utc=True, errors="coerce")
    entries = entries.rename(
        columns={"side": "entry_side", "timeframe": "entry_timeframe"}
    )
    entries = entries[
        entries["is_onset"].eq(1) & entries["pair"].isin(pair_to_cache_key)
    ].copy()
    entries["family"] = entries["entry_id"].map(family_for_entry)
    entries["window"] = "outside"
    for name, (start, end) in windows.items():
        entries.loc[
            entries["decision_time"].ge(start) & entries["decision_time"].lt(end),
            "window",
        ] = name
    entries = entries[~entries["window"].eq("outside")]

    level_frames: list[DataFrame] = []
    stale_frames: list[DataFrame] = []
    reaction_frames: list[DataFrame] = []
    cache_coverage: list[dict[str, Any]] = []
    overall_start = min(start for start, _ in windows.values())
    overall_end = max(end for _, end in windows.values())
    expected_decisions = pd.date_range(
        overall_start,
        overall_end - pd.Timedelta(hours=1),
        freq="1h",
    )
    for pair, cache_key in pair_to_cache_key.items():
        cache_path = level_cache_dir / cache_file(cache_key)
        cache_indexed = load_cache(cache_path)
        decision_index = pd.DatetimeIndex(cache_indexed.index + pd.Timedelta(hours=1))
        missing_window_hours = expected_decisions.difference(decision_index)
        pair_event_times = pd.DatetimeIndex(
            entries.loc[entries["pair"].eq(pair), "decision_time"].dropna().unique()
        )
        missing_event_times = pair_event_times.difference(decision_index)
        if len(missing_window_hours) or len(missing_event_times):
            raise ValueError(
                f"Level cache coverage defect for {pair}: "
                f"missing_window_hours={len(missing_window_hours)} "
                f"missing_event_times={len(missing_event_times)} path={cache_path}"
            )
        cache_coverage.append(
            {
                "pair": pair,
                "path": str(cache_path),
                "rows": int(len(cache_indexed)),
                "decision_start": decision_index.min().isoformat(),
                "decision_end": decision_index.max().isoformat(),
                "missing_window_hours": int(len(missing_window_hours)),
                "eligible_event_times": int(len(pair_event_times)),
                "missing_event_times": int(len(missing_event_times)),
            }
        )
        cache = cache_indexed.reset_index()
        level_columns = [column for column in cache if column.startswith("st_")]
        stale = cache.copy()
        stale[level_columns] = stale[level_columns].shift(placebo_shift_hours)
        for source, frame, collector in (
            ("contemporaneous_levels", cache, level_frames),
            (f"stale_{placebo_shift_hours}h_levels", stale, stale_frames),
        ):
            support = level_prices(frame.set_index("date"), "support").add_prefix("support__")
            resistance = level_prices(frame.set_index("date"), "resistance").add_prefix("resistance__")
            state = pd.concat(
                [frame.set_index("date")[["close"]].rename(columns={"close": "level_state_close"}), support, resistance],
                axis=1,
            ).reset_index()
            state["decision_time"] = pd.to_datetime(state["date"], utc=True) + pd.Timedelta(hours=1)
            state["pair"] = pair
            collector.append(state.drop(columns="date"))
        reaction_frames.append(actual_frame(pair, overall_start, overall_end))
    reactions = pd.concat(reaction_frames, ignore_index=True)
    target_columns = [column for column in reactions if column.startswith("&-")]
    entries = entries.merge(
        reactions[["pair", "decision_time", *target_columns]],
        on=["pair", "decision_time"],
        how="inner",
        validate="many_to_one",
    )
    actual_states = _level_state(
        entries,
        pd.concat(level_frames, ignore_index=True),
        bands=bands,
        evidence_source="contemporaneous_levels",
    )
    stale_states = _level_state(
        entries,
        pd.concat(stale_frames, ignore_index=True),
        bands=bands,
        evidence_source=f"stale_{placebo_shift_hours}h_levels",
    )
    states = pd.concat([actual_states, stale_states], ignore_index=True)
    summary = _summary(states, horizons=horizons)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_columns = [
        "evidence_source",
        "entry_id",
        "family",
        "entry_side",
        "entry_timeframe",
        "pair",
        "decision_time",
        "window",
        "proximity_band_pct",
        "relationship_state",
        "mtf_precedence_state",
        "same_direction_level_count",
        "opposing_level_count",
        "same_direction_max_tf_hours",
        "opposing_max_tf_hours",
        "support_level_count",
        "resistance_level_count",
    ]
    states_path = output_dir / "entry_level_relationship_states.parquet"
    summary_path = output_dir / "entry_level_reaction_summary.csv"
    atomic_parquet(states_path, states[[column for column in state_columns if column in states]])
    summary.to_csv(summary_path, index=False)
    meta = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "objective": "Test low-dimensional entry proximity and higher-timeframe precedence relationships without rerunning the rejected all-feature bundle.",
        "pairs": list(pair_to_cache_key),
        "cache_coverage": cache_coverage,
        "bands": list(bands),
        "horizons_hours": list(horizons),
        "placebo_shift_hours": int(placebo_shift_hours),
        "timestamp_rule": "Entry decision_time joins level state from the just-closed 1h candle; no future level row is used.",
        "role_rule": "For longs, valid support at/below price is same-direction and valid resistance at/above price is opposing; shorts reverse those roles. Crossed/displaced nearby levels remain a separate state.",
        "precedence_rule": "Higher-timeframe precedence is a test label derived from the greatest timeframe among nearby valid same-direction and opposing levels; it is not assumed to be correct.",
        "interpretation_limit": "Association only; compare contemporaneous classification with the 168h stale-level placebo and preserve sparse/contradictory cells.",
        "entry_event_rows": int(len(entries)),
        "relationship_state_rows": int(len(states)),
        "summary_rows": int(len(summary)),
        "states_path": str(states_path),
        "summary_path": str(summary_path),
    }
    atomic_json(output_dir / "entry_level_relationship_meta.json", meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the bounded Sieve3 entry/level proximity relationship map.")
    parser.add_argument("--event-output-dir", type=Path, default=DEFAULT_EVENT_OUTPUT)
    parser.add_argument("--level-cache-dir", type=Path, default=DEFAULT_LEVEL_CACHE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--windows-json", type=Path, default=DEFAULT_EVENT_OUTPUT / "windows.json")
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_DIRECT_HORIZONS))
    parser.add_argument("--bands", default="0.0025,0.005,0.01,0.02")
    parser.add_argument("--placebo-shift-hours", type=int, default=168)
    parser.add_argument(
        "--pairs",
        default=",".join(BASELINE_PAIR_TO_CACHE_KEY),
        help="Comma-separated frozen pair universe; cache filenames use the lower-case base asset.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.event_output_dir / "level_relationships"
    result = build_level_relationships(
        event_output_dir=args.event_output_dir,
        level_cache_dir=args.level_cache_dir,
        output_dir=output_dir,
        windows=parse_windows(args.windows_json),
        horizons=parse_horizons(args.horizons),
        bands=parse_bands(args.bands),
        placebo_shift_hours=args.placebo_shift_hours,
        pair_to_cache_key=parse_pairs(args.pairs),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
