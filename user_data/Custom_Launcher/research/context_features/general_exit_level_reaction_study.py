from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "general_exit_level_reaction_cache"
)
DEFAULT_OUTPUT_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "reports"
    / "general_exit_level_reaction"
)
CACHE_FILES = {
    "BTC": "btc_general_exit_levels_5tf.parquet",
    "ETH": "eth_general_exit_levels_5tf.parquet",
    "SOL": "sol_general_exit_levels_5tf.parquet",
}
TIMEFRAMES = ("1h", "4h", "8h", "1d", "3d")
FAMILIES = ("va_edge", "prior_poc", "hvn", "trendline")
TIMEFRAME_ORDER = {timeframe: index for index, timeframe in enumerate(TIMEFRAMES)}
FAMILY_ORDER = {family: index for index, family in enumerate(FAMILIES)}
WINDOWS = {
    "development": (
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
    ),
    "holdout": (
        pd.Timestamp("2025-04-01", tz="UTC"),
        pd.Timestamp("2026-04-01", tz="UTC"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure explicit named-level and exact-cluster reactions over 1-4 candles."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cluster-band", type=float, default=0.005)
    parser.add_argument("--touch-band", type=float, default=0.001)
    parser.add_argument("--placebo-shifts", default="72,168")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    placebo_shifts = tuple(
        int(value.strip())
        for value in str(args.placebo_shifts).split(",")
        if value.strip()
    )

    plan = {
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "cluster_band": float(args.cluster_band),
        "touch_band": float(args.touch_band),
        "timeframes": list(TIMEFRAMES),
        "families": list(FAMILIES),
        "reaction_horizons_candles": [1, 2, 3, 4],
        "placebo_level_state_shifts_hours": list(placebo_shifts),
        "timestamp_rule": (
            "Each hit candle is tested against levels available at the preceding candle close."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    frames = {
        pair: load_cache(args.cache_dir / filename)
        for pair, filename in CACHE_FILES.items()
    }
    events = build_events(
        frames,
        cluster_band=float(args.cluster_band),
        touch_band=float(args.touch_band),
    )
    events["evidence_source"] = "contemporaneous_levels"
    events["placebo_shift_hours"] = 0
    placebo_frames: list[DataFrame] = []
    for shift_hours in placebo_shifts:
        shifted = {
            pair: shifted_level_state_frame(frame, shift_hours)
            for pair, frame in frames.items()
        }
        placebo = build_events(
            shifted,
            cluster_band=float(args.cluster_band),
            touch_band=float(args.touch_band),
        )
        placebo["evidence_source"] = "stale_level_placebo"
        placebo["placebo_shift_hours"] = shift_hours
        placebo_frames.append(placebo)
    placebo_events = (
        pd.concat(placebo_frames, ignore_index=True)
        if placebo_frames
        else DataFrame(columns=events.columns)
    )
    summaries = summarize_events(events)
    if not placebo_events.empty:
        summaries.update(
            summarize_actual_vs_placebo(
                pd.concat([events, placebo_events], ignore_index=True)
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "level_reaction_events.parquet"
    events.to_parquet(events_path, index=False)
    outputs = {"events": str(events_path)}
    if not placebo_events.empty:
        placebo_path = args.output_dir / "level_reaction_placebo_events.parquet"
        placebo_events.to_parquet(placebo_path, index=False)
        outputs["placebo_events"] = str(placebo_path)
    for name, frame in summaries.items():
        path = args.output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = str(path)
    meta_path = args.output_dir / "level_reaction_meta.json"
    meta = {
        **plan,
        "event_rows": int(len(events)),
        "first_touch_event_rows": int(events["first_touch_after_4h"].sum()),
        "placebo_event_rows": int(len(placebo_events)),
        "placebo_first_touch_event_rows": int(
            placebo_events["first_touch_after_4h"].sum()
        )
        if not placebo_events.empty
        else 0,
        "outputs": outputs,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**meta, "meta": str(meta_path)}, indent=2))
    return 0


def load_cache(path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .set_index("date")
    )


def shifted_level_state_frame(frame: DataFrame, shift_hours: int) -> DataFrame:
    """Keep the market path fixed while making structural levels deliberately stale."""
    shifted = frame.copy()
    level_columns = [column for column in shifted if str(column).startswith("st_")]
    shifted[level_columns] = shifted[level_columns].shift(int(shift_hours))
    return shifted


def build_events(
    frames: dict[str, DataFrame], *, cluster_band: float, touch_band: float
) -> DataFrame:
    btc = frames["BTC"]
    btc_inputs = prepare_btc_inputs(btc, touch_band=touch_band)
    btc_context_cache: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for pair in ("ETH", "SOL"):
        frame = frames[pair]
        for side in ("resistance", "support"):
            records.extend(
                build_side_events(
                    pair,
                    frame,
                    btc_inputs,
                    side=side,
                    cluster_band=cluster_band,
                    touch_band=touch_band,
                    btc_context_cache=btc_context_cache,
                )
            )
    events = DataFrame(records)
    if events.empty:
        raise ValueError("No named-level reaction events were found")
    events = events[~events["window"].eq("outside")].copy()
    events = events.sort_values(
        ["pair", "side", "composition", "date", "event_kind"]
    ).reset_index(drop=True)
    previous = events.groupby(["pair", "side", "composition"])["date"].shift(1)
    events["hours_since_same_composition_touch"] = (
        events["date"] - previous
    ).dt.total_seconds() / 3600.0
    events["first_touch_after_4h"] = previous.isna() | events[
        "hours_since_same_composition_touch"
    ].gt(4.0)
    return events


def build_side_events(
    pair: str,
    frame: DataFrame,
    btc_inputs: dict[str, dict[str, Any]],
    *,
    side: str,
    cluster_band: float,
    touch_band: float,
    btc_context_cache: dict[tuple[pd.Timestamp, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    prior = frame.shift(1)
    prior_close = numeric(prior, "close").replace(0.0, np.nan)
    levels = level_prices(prior, side)
    levels = levels.where(
        levels.gt(prior_close, axis=0)
        if side == "resistance"
        else levels.lt(prior_close, axis=0)
    )
    high = numeric(frame, "high")
    low = numeric(frame, "low")
    close = numeric(frame, "close")
    touched = high.to_numpy()[:, None] >= levels.to_numpy() * (1.0 - touch_band)
    touched &= low.to_numpy()[:, None] <= levels.to_numpy() * (1.0 + touch_band)
    touched &= np.isfinite(levels.to_numpy())
    event_positions = np.flatnonzero(touched.any(axis=1))
    atr = average_true_range(frame)
    pressure = close_location_pressure(frame)
    pre_toward = (pressure if side == "resistance" else -pressure).shift(1).rolling(
        3, min_periods=3
    ).mean()
    volume = numeric(frame, "volume", 0.0).clip(lower=0.0)
    pre_volume = volume.shift(1).rolling(3, min_periods=3).mean().replace(0.0, np.nan)
    poc_age_hours = {
        timeframe: stability_age_hours(
            numeric(prior, f"st_{timeframe}_vp_prior_poc")
        )
        for timeframe in TIMEFRAMES
    }
    records: list[dict[str, Any]] = []

    for position in event_positions:
        date = frame.index[position]
        row_levels = levels.iloc[position]
        hit_names = list(levels.columns[touched[position]])
        clusters = unique_hit_clusters(
            row_levels, hit_names, cluster_band=cluster_band
        )
        btc_key = (date, side)
        if btc_key not in btc_context_cache:
            btc_context_cache[btc_key] = btc_context(
                btc_inputs,
                date,
                side=side,
                cluster_band=cluster_band,
                touch_band=touch_band,
            )
        context = btc_context_cache[btc_key]

        for name in hit_names:
            surrounding = next(
                (members for members in clusters if name in members), (name,)
            )
            records.append(
                event_record(
                    pair,
                    frame,
                    levels,
                    position,
                    side=side,
                    members=(name,),
                    event_kind="single_level",
                    surrounding_cluster_size=len(surrounding),
                    atr=atr,
                    pressure=pressure,
                    pre_toward=pre_toward,
                    pre_volume=pre_volume,
                    poc_age_hours=poc_age_hours,
                    prior=prior,
                    close_series=close,
                    high_series=high,
                    low_series=low,
                    volume_series=volume,
                    btc=context,
                    touch_band=touch_band,
                )
            )
        for members in clusters:
            if len(members) < 2:
                continue
            records.append(
                event_record(
                    pair,
                    frame,
                    levels,
                    position,
                    side=side,
                    members=members,
                    event_kind=cluster_kind(members),
                    surrounding_cluster_size=len(members),
                    atr=atr,
                    pressure=pressure,
                    pre_toward=pre_toward,
                    pre_volume=pre_volume,
                    poc_age_hours=poc_age_hours,
                    prior=prior,
                    close_series=close,
                    high_series=high,
                    low_series=low,
                    volume_series=volume,
                    btc=context,
                    touch_band=touch_band,
                )
            )
    return records


def level_prices(frame: DataFrame, side: str) -> DataFrame:
    columns: dict[str, Series] = {}
    for timeframe in TIMEFRAMES:
        prefix = f"st_{timeframe}"
        source_columns = {
            "va_edge": f"{prefix}_vp_prior_vah"
            if side == "resistance"
            else f"{prefix}_vp_prior_val",
            "prior_poc": f"{prefix}_vp_prior_poc",
            "hvn": f"{prefix}_vp_hvn_above"
            if side == "resistance"
            else f"{prefix}_vp_hvn_below",
            "trendline": f"{prefix}_tlv2_resistance_line_rank0"
            if side == "resistance"
            else f"{prefix}_tlv2_support_line_rank0",
        }
        for family, source in source_columns.items():
            columns[f"{timeframe}:{family}"] = numeric(frame, source)
    return DataFrame(columns, index=frame.index)


def unique_hit_clusters(
    levels: Series, hit_names: list[str], *, cluster_band: float
) -> list[tuple[str, ...]]:
    clusters: set[tuple[str, ...]] = set()
    finite = levels.dropna()
    for hit_name in hit_names:
        anchor = finite.get(hit_name)
        if anchor is None or not np.isfinite(anchor):
            continue
        members = tuple(
            sorted(
                (
                    name
                    for name, value in finite.items()
                    if abs(float(value) / float(anchor) - 1.0) <= cluster_band
                ),
                key=member_sort_key,
            )
        )
        clusters.add(members)
    return sorted(clusters, key=lambda members: (len(members), members))


def cluster_kind(members: tuple[str, ...]) -> str:
    timeframes = {member.split(":", 1)[0] for member in members}
    families = {member.split(":", 1)[1] for member in members}
    if len(timeframes) == 1 and len(families) > 1:
        return "same_timeframe_cross_type_cluster"
    if len(families) == 1 and len(timeframes) > 1:
        return "same_type_cross_timeframe_cluster"
    return "mixed_type_and_timeframe_cluster"


def event_record(
    pair: str,
    frame: DataFrame,
    levels: DataFrame,
    position: int,
    *,
    side: str,
    members: tuple[str, ...],
    event_kind: str,
    surrounding_cluster_size: int,
    atr: Series,
    pressure: Series,
    pre_toward: Series,
    pre_volume: Series,
    poc_age_hours: dict[str, Series],
    prior: DataFrame,
    close_series: Series,
    high_series: Series,
    low_series: Series,
    volume_series: Series,
    btc: dict[str, Any],
    touch_band: float,
) -> dict[str, Any]:
    date = frame.index[position]
    member_values = levels.loc[date, list(members)].astype(float)
    near_edge = float(member_values.min() if side == "resistance" else member_values.max())
    far_edge = float(member_values.max() if side == "resistance" else member_values.min())
    close = float(close_series.iloc[position])
    candle_atr = float(atr.iloc[position])
    hit_away_pressure = float(-pressure.iloc[position] if side == "resistance" else pressure.iloc[position])
    volume_ratio = float(
        volume_series.iloc[position] / pre_volume.iloc[position]
    ) if np.isfinite(pre_volume.iloc[position]) and pre_volume.iloc[position] > 0 else np.nan
    if side == "resistance":
        hit_state = (
            "rejected_before_near_edge"
            if close < near_edge * (1.0 - touch_band)
            else "accepted_through_far_edge"
            if close > far_edge * (1.0 + touch_band)
            else "closed_inside_zone"
        )
    else:
        hit_state = (
            "rejected_before_near_edge"
            if close > near_edge * (1.0 + touch_band)
            else "accepted_through_far_edge"
            if close < far_edge * (1.0 - touch_band)
            else "closed_inside_zone"
        )

    families = sorted({member.split(":", 1)[1] for member in members}, key=FAMILY_ORDER.get)
    timeframes = sorted(
        {member.split(":", 1)[0] for member in members}, key=TIMEFRAME_ORDER.get
    )
    record: dict[str, Any] = {
        "date": date,
        "window": window_name(date),
        "pair": pair,
        "side": side,
        "event_kind": event_kind,
        "composition": "|".join(members),
        "member_count": len(members),
        "distinct_type_count": len(families),
        "distinct_timeframe_count": len(timeframes),
        "families": "|".join(families),
        "timeframes": "|".join(timeframes),
        "single_family": families[0] if len(families) == 1 else "",
        "single_timeframe": timeframes[0] if len(timeframes) == 1 else "",
        "surrounding_cluster_size": surrounding_cluster_size,
        "zone_width_bps": abs(far_edge / near_edge - 1.0) * 10000.0,
        "hit_state": hit_state,
        "pre_toward_pressure_3h": float(pre_toward.iloc[position]),
        "hit_away_pressure": hit_away_pressure,
        "hit_pressure_turn": float(pre_toward.iloc[position] + hit_away_pressure),
        "hit_volume_vs_prior3": volume_ratio,
        "btc_same_side_named_level_hit_count": btc["hit_count"],
        "btc_same_side_cluster_size": btc["cluster_size"],
        "btc_same_side_cluster_composition": btc["composition"],
        "btc_same_side_near_cluster_size": btc["near_cluster_size"],
    }
    append_quality(
        record, prior, position, members, side, poc_age_hours=poc_age_hours
    )
    append_future_reactions(
        record,
        position,
        side=side,
        near_edge=near_edge,
        atr=candle_atr,
        pressure=pressure,
        pre_volume=pre_volume,
        close=close_series,
        high=high_series,
        low=low_series,
        volume=volume_series,
        touch_band=touch_band,
    )
    return record


def append_future_reactions(
    record: dict[str, Any],
    position: int,
    *,
    side: str,
    near_edge: float,
    atr: float,
    pressure: Series,
    pre_volume: Series,
    close: Series,
    high: Series,
    low: Series,
    volume: Series,
    touch_band: float,
) -> None:
    hit_close = float(close.iloc[position])
    for horizon in (1, 2, 3, 4):
        future = slice(position + 1, position + 1 + horizon)
        if position + horizon >= len(close) or not np.isfinite(atr) or atr <= 0.0:
            for name in (
                "away_excursion_atr",
                "through_excursion_atr",
                "away_advantage_atr",
                "close_away_bps",
                "away_pressure_mean",
                "volume_vs_pre3",
                "reversal_probability_flag",
                "continuation_probability_flag",
                "returned_away_through_near_edge",
            ):
                record[f"{name}_{horizon}c"] = np.nan
            continue
        future_high = float(high.iloc[future].max())
        future_low = float(low.iloc[future].min())
        future_close = float(close.iloc[position + horizon])
        if side == "resistance":
            away = (hit_close - future_low) / atr
            through = (future_high - hit_close) / atr
            close_away_bps = (hit_close / future_close - 1.0) * 10000.0
            away_pressure = -pressure.iloc[future]
            returned = close.iloc[future].min() < near_edge * (1.0 - touch_band)
        else:
            away = (future_high - hit_close) / atr
            through = (hit_close - future_low) / atr
            close_away_bps = (future_close / hit_close - 1.0) * 10000.0
            away_pressure = pressure.iloc[future]
            returned = close.iloc[future].max() > near_edge * (1.0 + touch_band)
        advantage = away - through
        future_volume = float(volume.iloc[future].mean())
        baseline_volume = float(pre_volume.iloc[position])
        record[f"away_excursion_atr_{horizon}c"] = away
        record[f"through_excursion_atr_{horizon}c"] = through
        record[f"away_advantage_atr_{horizon}c"] = advantage
        record[f"close_away_bps_{horizon}c"] = close_away_bps
        record[f"away_pressure_mean_{horizon}c"] = float(away_pressure.mean())
        record[f"volume_vs_pre3_{horizon}c"] = (
            future_volume / baseline_volume if baseline_volume > 0.0 else np.nan
        )
        record[f"reversal_probability_flag_{horizon}c"] = float(
            advantage > 0.25 and away_pressure.mean() > 0.0
        )
        record[f"continuation_probability_flag_{horizon}c"] = float(
            advantage < -0.25 and away_pressure.mean() < 0.0
        )
        record[f"returned_away_through_near_edge_{horizon}c"] = float(returned)


def append_quality(
    record: dict[str, Any],
    prior: DataFrame,
    position: int,
    members: tuple[str, ...],
    side: str,
    *,
    poc_age_hours: dict[str, Series],
) -> None:
    hvn_strength: list[float] = []
    trendline_score: list[float] = []
    trendline_pivots: list[float] = []
    value_area_width: list[float] = []
    poc_age: list[float] = []
    for member in members:
        timeframe, family = member.split(":", 1)
        prefix = f"st_{timeframe}"
        if family == "hvn":
            column = f"{prefix}_vp_hvn_above_strength" if side == "resistance" else f"{prefix}_vp_hvn_below_strength"
            hvn_strength.append(value_at(prior, column, position))
        elif family == "trendline":
            score_column = f"{prefix}_tlv2_resistance_score_rank0" if side == "resistance" else f"{prefix}_tlv2_support_score_rank0"
            pivot_column = f"{prefix}_tlv2_resistance_pivot_count_rank0" if side == "resistance" else f"{prefix}_tlv2_support_pivot_count_rank0"
            trendline_score.append(value_at(prior, score_column, position))
            trendline_pivots.append(value_at(prior, pivot_column, position))
        elif family == "va_edge":
            value_area_width.append(value_at(prior, f"{prefix}_vp_value_area_width_pct", position))
        elif family == "prior_poc":
            poc_age.append(float(poc_age_hours[timeframe].iloc[position]))
    record["hvn_strength_max"] = finite_max(hvn_strength)
    record["trendline_score_max"] = finite_max(trendline_score)
    record["trendline_pivots_max"] = finite_max(trendline_pivots)
    record["value_area_width_mean"] = finite_mean(value_area_width)
    record["prior_poc_age_hours_max"] = finite_max(poc_age)


def prepare_btc_inputs(
    btc: DataFrame, *, touch_band: float
) -> dict[str, dict[str, Any]]:
    prior = btc.shift(1)
    prior_close = numeric(prior, "close")
    high = numeric(btc, "high")
    low = numeric(btc, "low")
    prepared: dict[str, dict[str, Any]] = {}
    for side in ("resistance", "support"):
        levels = level_prices(prior, side)
        levels = levels.where(
            levels.gt(prior_close, axis=0)
            if side == "resistance"
            else levels.lt(prior_close, axis=0)
        )
        touched = high.to_numpy()[:, None] >= levels.to_numpy() * (1.0 - touch_band)
        touched &= low.to_numpy()[:, None] <= levels.to_numpy() * (1.0 + touch_band)
        touched &= np.isfinite(levels.to_numpy())
        prepared[side] = {
            "index": btc.index,
            "levels": levels,
            "prior_close": prior_close,
            "touched": touched,
        }
    return prepared


def btc_context(
    btc_inputs: dict[str, dict[str, Any]],
    date: pd.Timestamp,
    *,
    side: str,
    cluster_band: float,
    touch_band: float,
) -> dict[str, Any]:
    prepared = btc_inputs[side]
    index = prepared["index"]
    if date not in index:
        return {"hit_count": 0, "cluster_size": 0, "composition": "", "near_cluster_size": 0}
    position = index.get_loc(date)
    if not isinstance(position, (int, np.integer)) or position < 1:
        return {"hit_count": 0, "cluster_size": 0, "composition": "", "near_cluster_size": 0}
    levels = prepared["levels"].iloc[position]
    prior_close = float(prepared["prior_close"].iloc[position])
    hit_names = list(levels.index[prepared["touched"][position]])
    clusters = unique_hit_clusters(levels, hit_names, cluster_band=cluster_band)
    largest = max(clusters, key=len, default=())
    distances = ((levels / prior_close) - 1.0).abs()
    near_cluster_size = 0
    for name, distance in distances.dropna().items():
        if distance > 0.01:
            continue
        anchor = levels[name]
        size = int(((levels / anchor) - 1.0).abs().le(cluster_band).sum())
        near_cluster_size = max(near_cluster_size, size)
    return {
        "hit_count": len(hit_names),
        "cluster_size": len(largest),
        "composition": "|".join(largest),
        "near_cluster_size": near_cluster_size,
    }


def summarize_events(events: DataFrame) -> dict[str, DataFrame]:
    primary = events[events["first_touch_after_4h"]].copy()
    single = primary[primary["event_kind"].eq("single_level")].copy()
    cluster = primary[~primary["event_kind"].eq("single_level")].copy()
    return {
        "single_level_summary": grouped_summary(
            single,
            ["window", "single_family", "single_timeframe", "surrounding_cluster_size"],
        ),
        "cluster_structure_summary": grouped_summary(
            cluster,
            [
                "window",
                "event_kind",
                "member_count",
                "distinct_type_count",
                "distinct_timeframe_count",
            ],
        ),
        "exact_cluster_composition_summary": grouped_summary(
            cluster,
            ["window", "event_kind", "composition"],
            minimum_rows=20,
        ),
        "btc_context_summary": grouped_summary(
            primary.assign(
                btc_cluster_bucket=pd.cut(
                    primary["btc_same_side_cluster_size"],
                    bins=[-1, 0, 1, 2, np.inf],
                    labels=["none", "single", "two", "three_plus"],
                )
            ),
            ["window", "event_kind", "btc_cluster_bucket"],
        ),
        "single_level_quality_summary": quality_summary(single),
    }


def summarize_actual_vs_placebo(events: DataFrame) -> dict[str, DataFrame]:
    primary = events[events["first_touch_after_4h"]].copy()
    single = primary[primary["event_kind"].eq("single_level")].copy()
    cluster = primary[~primary["event_kind"].eq("single_level")].copy()
    return {
        "actual_vs_placebo_single_summary": grouped_summary(
            single,
            [
                "evidence_source",
                "placebo_shift_hours",
                "window",
                "single_family",
                "single_timeframe",
                "surrounding_cluster_size",
            ],
        ),
        "actual_vs_placebo_cluster_structure_summary": grouped_summary(
            cluster,
            [
                "evidence_source",
                "placebo_shift_hours",
                "window",
                "event_kind",
                "member_count",
                "distinct_type_count",
                "distinct_timeframe_count",
            ],
        ),
        "actual_vs_placebo_exact_cluster_summary": grouped_summary(
            cluster,
            [
                "evidence_source",
                "placebo_shift_hours",
                "window",
                "event_kind",
                "composition",
            ],
            minimum_rows=20,
        ),
    }


def grouped_summary(
    frame: DataFrame, group_columns: list[str], *, minimum_rows: int = 1
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, observed=True):
        if len(group) < minimum_rows:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["events"] = int(len(group))
        row["hit_volume_vs_prior3_median"] = float(group["hit_volume_vs_prior3"].median())
        row["hit_pressure_turn_mean"] = float(group["hit_pressure_turn"].mean())
        row["accepted_through_fraction"] = float(
            group["hit_state"].eq("accepted_through_far_edge").mean()
        )
        for horizon in (1, 2, 3, 4):
            for column in (
                "away_advantage_atr",
                "close_away_bps",
                "away_pressure_mean",
                "volume_vs_pre3",
                "reversal_probability_flag",
                "continuation_probability_flag",
                "returned_away_through_near_edge",
            ):
                values = pd.to_numeric(
                    group[f"{column}_{horizon}c"], errors="coerce"
                )
                reducer = values.mean if column != "volume_vs_pre3" else values.median
                row[f"{column}_{horizon}c"] = float(reducer())
        rows.append(row)
    return DataFrame(rows)


def quality_summary(single: DataFrame) -> DataFrame:
    quality_by_family = {
        "va_edge": "value_area_width_mean",
        "prior_poc": "prior_poc_age_hours_max",
        "hvn": "hvn_strength_max",
        "trendline": "trendline_score_max",
    }
    frames: list[DataFrame] = []
    for (window, family), group in single.groupby(
        ["window", "single_family"], observed=True
    ):
        column = quality_by_family.get(str(family))
        if column is None:
            continue
        scoped = group.dropna(subset=[column]).copy()
        if len(scoped) < 30 or scoped[column].nunique() < 3:
            continue
        bucket_codes = pd.qcut(
            scoped[column], q=3, labels=False, duplicates="drop"
        )
        highest = int(bucket_codes.max())
        scoped["quality_bucket"] = bucket_codes.map(
            lambda code: "low"
            if int(code) == 0
            else "high"
            if int(code) == highest
            else "middle"
        )
        summary = grouped_summary(
            scoped,
            ["window", "single_family", "quality_bucket"],
        )
        summary["quality_metric"] = column
        summary["quality_value_mean"] = scoped.groupby(
            "quality_bucket", observed=True
        )[column].mean().reindex(summary["quality_bucket"]).to_numpy()
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def average_true_range(frame: DataFrame, period: int = 14) -> Series:
    high = numeric(frame, "high")
    low = numeric(frame, "low")
    close = numeric(frame, "close")
    prior_close = close.shift(1)
    true_range = pd.concat(
        ((high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def close_location_pressure(frame: DataFrame) -> Series:
    high = numeric(frame, "high")
    low = numeric(frame, "low")
    close = numeric(frame, "close")
    return (((close - low) / (high - low).replace(0.0, np.nan)) * 2.0 - 1.0).clip(
        -1.0, 1.0
    )


def stability_age_hours(level: Series) -> Series:
    numeric_level = pd.to_numeric(level, errors="coerce")
    changed = (
        numeric_level.isna()
        | numeric_level.shift(1).isna()
        | numeric_level.pct_change(fill_method=None).abs().gt(0.001)
    )
    return numeric_level.groupby(changed.cumsum()).cumcount().astype(float).where(
        numeric_level.notna(), np.nan
    )


def numeric(frame: DataFrame, column: str, default: float = np.nan) -> Series:
    if column not in frame:
        return Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def value_at(frame: DataFrame, column: str, position: int) -> float:
    if column not in frame:
        return np.nan
    value = pd.to_numeric(Series([frame.iloc[position][column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def finite_max(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else np.nan


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def member_sort_key(member: str) -> tuple[int, int]:
    timeframe, family = member.split(":", 1)
    return TIMEFRAME_ORDER[timeframe], FAMILY_ORDER[family]


def window_name(date: pd.Timestamp) -> str:
    for name, (start, end) in WINDOWS.items():
        if start <= date < end:
            return name
    return "outside"


if __name__ == "__main__":
    raise SystemExit(main())
