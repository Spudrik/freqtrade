"""Build compact timestamp-safe current/stale level-state caches for G1-B3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from general_exit_level_reaction_study import level_prices, load_cache
from sieve3_event_level_relationships import _level_state, cache_file, parse_bands


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def pair_token(pair: str) -> str:
    return pair.split("/", 1)[0].split(":", 1)[0].lower()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def level_frame(cache: DataFrame, pair: str) -> DataFrame:
    support = level_prices(cache, "support").add_prefix("support__")
    resistance = level_prices(cache, "resistance").add_prefix("resistance__")
    output = pd.concat(
        [
            pd.to_numeric(cache["close"], errors="coerce").rename("level_state_close"),
            support,
            resistance,
        ],
        axis=1,
    ).reset_index()
    output["decision_time"] = pd.to_datetime(output["date"], utc=True) + pd.Timedelta(
        hours=1
    )
    output["pair"] = pair
    return output.drop(columns="date")


def pseudo_events(levels: DataFrame, pair: str) -> DataFrame:
    times = levels[["decision_time"]].drop_duplicates().copy()
    records: list[DataFrame] = []
    for side in ("long", "short"):
        frame = times.copy()
        frame["pair"] = pair
        frame["entry_id"] = "all_hourly_level_states"
        frame["family"] = "all_entries"
        frame["entry_side"] = side
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def band_token(band: float) -> str:
    return f"{int(round(band * 10000.0))}bp"


def compact_states(
    states: DataFrame,
    *,
    decision_times: pd.Series,
    evidence_prefix: str,
    bands: tuple[float, ...],
) -> DataFrame:
    output = DataFrame(
        {
            "decision_time": pd.to_datetime(
                decision_times.drop_duplicates().sort_values(), utc=True
            )
        }
    ).set_index("decision_time")
    relationship_values = (
        "mixed",
        "same_direction_only",
        "opposing_only",
        "crossed_or_displaced_only",
        "no_nearby_level",
    )
    precedence_values = (
        "same_only",
        "opposing_only",
        "same_direction_higher_tf",
        "opposing_higher_tf",
        "equal_highest_tf",
        "none_or_displaced",
    )
    for side in ("long", "short"):
        for band in bands:
            group = states[
                states["entry_side"].eq(side)
                & np.isclose(states["proximity_band_pct"], band)
            ].copy()
            if group["decision_time"].duplicated().any():
                raise ValueError(f"Duplicate {evidence_prefix} {side} {band} states")
            group = group.set_index("decision_time").reindex(output.index)
            if group["relationship_state"].isna().any():
                raise ValueError(f"Missing {evidence_prefix} {side} {band} states")
            prefix = f"{evidence_prefix}__{side}__{band_token(band)}"
            relationship = group["relationship_state"].astype(str)
            precedence = group["mtf_precedence_state"].astype(str)
            for value in relationship_values:
                output[f"{prefix}__relationship_{value}"] = relationship.eq(value).astype(
                    float
                )
            for value in precedence_values:
                output[f"{prefix}__precedence_{value}"] = precedence.eq(value).astype(
                    float
                )
            same_count = pd.to_numeric(
                group["same_direction_level_count"], errors="coerce"
            )
            opposing_count = pd.to_numeric(
                group["opposing_level_count"], errors="coerce"
            )
            same_tf = pd.to_numeric(
                group["same_direction_max_tf_hours"], errors="coerce"
            )
            opposing_tf = pd.to_numeric(
                group["opposing_max_tf_hours"], errors="coerce"
            )
            output[f"{prefix}__same_count"] = same_count.fillna(0.0).clip(0.0, 40.0)
            output[f"{prefix}__opposing_count"] = opposing_count.fillna(0.0).clip(
                0.0, 40.0
            )
            output[f"{prefix}__same_max_tf_72h"] = same_tf.fillna(0.0) / 72.0
            output[f"{prefix}__opposing_max_tf_72h"] = opposing_tf.fillna(0.0) / 72.0
            output[f"{prefix}__mixed_equal_highest"] = (
                relationship.eq("mixed") & precedence.eq("equal_highest_tf")
            ).astype(float)
            output[f"{prefix}__mixed_same_higher"] = (
                relationship.eq("mixed")
                & precedence.eq("same_direction_higher_tf")
            ).astype(float)
            output[f"{prefix}__mixed_opposing_higher"] = (
                relationship.eq("mixed") & precedence.eq("opposing_higher_tf")
            ).astype(float)
            output[f"{prefix}__one_sided_max_tf_72h"] = np.where(
                relationship.eq("same_direction_only"),
                same_tf.fillna(0.0) / 72.0,
                np.where(
                    relationship.eq("opposing_only"),
                    opposing_tf.fillna(0.0) / 72.0,
                    0.0,
                ),
            )
    return output.reset_index()


def build_pair_cache(
    *,
    pair: str,
    source_path: Path,
    output_path: Path,
    bands: tuple[float, ...],
    placebo_shift_hours: int,
) -> dict[str, Any]:
    cache = load_cache(source_path)
    current_levels = level_frame(cache, pair)
    events = pseudo_events(current_levels, pair)
    current_states = _level_state(
        events,
        current_levels,
        bands=bands,
        evidence_source="contemporaneous_levels",
    )
    current_compact = compact_states(
        current_states,
        decision_times=current_levels["decision_time"],
        evidence_prefix="level",
        bands=bands,
    )
    stale_cache = cache.copy()
    structural_columns = [column for column in stale_cache if column.startswith("st_")]
    stale_cache[structural_columns] = stale_cache[structural_columns].shift(
        placebo_shift_hours
    )
    stale_levels = level_frame(stale_cache, pair)
    stale_states = _level_state(
        events,
        stale_levels,
        bands=bands,
        evidence_source=f"stale_{placebo_shift_hours}h_levels",
    )
    stale_compact = compact_states(
        stale_states,
        decision_times=stale_levels["decision_time"],
        evidence_prefix="stale_level",
        bands=bands,
    )
    output = current_compact.merge(
        stale_compact, on="decision_time", how="inner", validate="one_to_one"
    )
    output["coverage_present"] = 1.0
    if output["decision_time"].duplicated().any():
        raise ValueError(f"Duplicate B3 cache decision times for {pair}")
    numeric = output.select_dtypes(include=[np.number])
    infinite = int(np.isinf(numeric.to_numpy(dtype="float64")).sum())
    missing = int(numeric.isna().sum().sum())
    if infinite or missing:
        raise ValueError(
            f"Invalid B3 cache numerics for {pair}: inf={infinite} missing={missing}"
        )
    atomic_parquet(output, output_path)
    current_features = [column for column in output if column.startswith("level__")]
    stale_features = [column for column in output if column.startswith("stale_level__")]
    return {
        "pair": pair,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": int(len(output)),
        "start": output["decision_time"].min().isoformat(),
        "end": output["decision_time"].max().isoformat(),
        "duplicate_decision_times": 0,
        "current_features": len(current_features),
        "stale_features": len(stale_features),
        "infinite_numeric_cells": infinite,
        "missing_numeric_cells": missing,
    }


def build_scopes(
    *,
    events_path: Path,
    mapping_path: Path,
    pairs: list[str],
    cache_start: pd.Timestamp,
    cache_end: pd.Timestamp,
) -> DataFrame:
    events = pd.read_parquet(events_path)
    events = events[events["pair"].isin(pairs) & events["is_onset"].astype(bool)].copy()
    mapping = pd.read_csv(
        mapping_path, usecols=["entry_id", "family", "side", "entry_timeframe"]
    ).drop_duplicates()
    mapping = mapping.drop_duplicates(["entry_id", "side"], keep="last")
    events = events.merge(mapping, on=["entry_id", "side"], how="left", validate="many_to_one")
    events["decision_time"] = pd.to_datetime(events["decision_time"], utc=True)
    events = events[
        events["decision_time"].ge(cache_start)
        & events["decision_time"].le(cache_end)
    ].copy()
    events = events.rename(columns={"side": "entry_side"})
    return events[
        ["pair", "decision_time", "entry_id", "family", "entry_side", "entry_timeframe"]
    ].sort_values(["pair", "decision_time", "entry_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation1_manifest", type=Path)
    parser.add_argument("generation0_base", type=Path)
    parser.add_argument("level_cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bands", default="0.0025,0.005,0.01,0.02")
    parser.add_argument("--placebo-shift-hours", type=int, default=168)
    args = parser.parse_args()
    manifest = json.loads(args.generation1_manifest.read_text(encoding="utf-8"))
    pairs = list(manifest["universe"]["pairs"])
    bands = parse_bands(args.bands)
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, Any]] = []
    for pair in pairs:
        token = pair_token(pair)
        audits.append(
            build_pair_cache(
                pair=pair,
                source_path=args.level_cache_dir.resolve() / cache_file(token),
                output_path=cache_dir / f"{token}_sieve3_events_1h.parquet",
                bands=bands,
                placebo_shift_hours=args.placebo_shift_hours,
            )
        )
        print(json.dumps({"pair": pair, "status": "completed"}), flush=True)
    cache_start = max(pd.Timestamp(item["start"]) for item in audits)
    cache_end = min(pd.Timestamp(item["end"]) for item in audits)
    generation0_base = args.generation0_base.resolve()
    scopes = build_scopes(
        events_path=generation0_base / "cache" / "entry_events_long.parquet",
        mapping_path=generation0_base / "direct_entry_portability_summary.csv",
        pairs=pairs,
        cache_start=cache_start,
        cache_end=cache_end,
    )
    scope_path = output_dir / "event_scopes.parquet"
    atomic_parquet(scopes, scope_path)
    audit = {
        "generated_at": now_iso(),
        "batch": "G1-B3",
        "bands": list(bands),
        "placebo_shift_hours": args.placebo_shift_hours,
        "pairs": audits,
        "scope_rows": int(len(scopes)),
        "scope_pairs": int(scopes["pair"].nunique()),
        "scope_entries": int(scopes["entry_id"].nunique()),
        "scope_families": int(scopes["family"].nunique()),
        "scope_path": str(scope_path),
        "timestamp_rule": "Structural levels are computed from the just-completed 1h/4h/8h/1d/3d candles and keyed to decision_time; stale controls shift structural columns 168 hours before recomputing proximity against current price.",
        "control_rule": "equal_highest_tf is inherently mixed; valid simpler controls are mixed/same-higher, mixed/opposing-higher, one-sided states matched on max timeframe, no-nearby-level and stale levels.",
        "future_derived_model_feature_count": 0,
        "status": "passed",
    }
    atomic_json(audit, output_dir / "cache_audit.json")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
