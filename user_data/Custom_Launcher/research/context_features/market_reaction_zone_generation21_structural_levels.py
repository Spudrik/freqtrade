"""Test frozen structural prices from completed abnormal hourly candles."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_freeze as g21z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g21_structural_levels_20260827a"
DEFAULT_SUPPORT_ID = "g21_structural_level_support_20260827a"
RECORD_ROOT = g21z.OUTPUT_ROOT / "structural_levels"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation21_branches"
    / "g21_broad_siblings"
    / "structural_level_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g21_structural_level_support_freeze.json"
HORIZONS = g21z.HORIZONS_HOURS
LEVELS = (
    "event_open",
    "event_close",
    "event_body_midpoint",
    "event_high",
    "event_low",
    "event_typical_price",
)
CONTROLS = (
    "matched_random_time",
    "random_recent_high_activity_candle",
    "near_miss",
    "stale_72h",
    "price_shift",
)
MAXIMUM_LEVEL_AGE = 168
ZONE_HALF_WIDTH_ATR = 0.25
RANDOM_ACTIVITY_DEFINITION = {
    "volume_zscore_minimum": 1.0,
    "range_over_atr_minimum": 1.0,
    "absolute_return_over_atr_minimum": 0.25,
    "exclude_abnormal_candle_definition": True,
    "selection": "stable-hash choice among prior 168 completed hourly candles",
}


def artifact(path: Path) -> dict[str, Any]:
    return g21z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g21z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g21d_abnormal_candle_structural_levels"
    )
    expected = {
        "completed_candle_only": True,
        "volume_zscore_minimum": 2.0,
        "range_over_atr_minimum": 1.5,
        "absolute_return_over_atr_minimum": 0.75,
        "maximum_level_age_hours": MAXIMUM_LEVEL_AGE,
    }
    if frozen.get("status") != "frozen_before_generation21_outcomes":
        raise ValueError("Generation 21 batch is not frozen.")
    if branch["event_definition"] != expected:
        raise ValueError("Generation 21 structural event definition drifted.")
    if tuple(branch["levels"]) != LEVELS or tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 21 structural registry drifted.")
    return frozen


def activity_components(base: DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    atr = pd.to_numeric(base["base_atr"], errors="coerce")
    volume_mean = volume.shift(1).rolling(168, min_periods=168).mean()
    volume_std = volume.shift(1).rolling(168, min_periods=168).std(ddof=0)
    volume_z = volume.sub(volume_mean).div(volume_std.replace(0.0, np.nan))
    range_over_atr = high.sub(low).div(atr)
    return_over_atr = close.diff().abs().div(atr)
    return (
        volume_z.to_numpy(dtype=float),
        range_over_atr.to_numpy(dtype=float),
        return_over_atr.to_numpy(dtype=float),
    )


def prior_abnormal_anchors(base: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return completed abnormal-candle anchors and their defining mask."""
    volume_z, range_over_atr, return_over_atr = activity_components(base)
    abnormal = (
        (volume_z >= 2.0)
        & (range_over_atr >= 1.5)
        & (return_over_atr >= 0.75)
    )
    indexes = np.arange(len(base), dtype=np.int64)
    last = Series(np.where(abnormal, indexes, np.nan)).ffill().shift(1)
    anchors = pd.to_numeric(last, errors="coerce").to_numpy(dtype=float)
    safe = np.where(np.isfinite(anchors), anchors, -1).astype(np.int64)
    ages = indexes - safe
    valid = np.isfinite(anchors) & (ages >= 1) & (ages <= MAXIMUM_LEVEL_AGE)
    safe[~valid] = -1
    return safe, abnormal


def random_recent_activity_anchors(
    base: DataFrame, pair: str, abnormal: np.ndarray
) -> np.ndarray:
    """Choose a causal, deterministic recent active candle for every hourly row."""
    volume_z, range_over_atr, return_over_atr = activity_components(base)
    candidate = (
        (volume_z >= 1.0)
        & (range_over_atr >= 1.0)
        & (return_over_atr >= 0.25)
        & ~abnormal
    )
    candidates = np.flatnonzero(candidate)
    output = np.full(len(base), -1, dtype=np.int64)
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    for row in range(len(base)):
        left = int(np.searchsorted(candidates, row - MAXIMUM_LEVEL_AGE, side="left"))
        right = int(np.searchsorted(candidates, row, side="left"))
        if right <= left:
            continue
        choice = g0.stable_hash_int(
            f"g21-random-active|{pair}|{dates.iloc[row].isoformat()}"
        ) % (right - left)
        output[row] = int(candidates[left + choice])
    return output


def values_at_anchor(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    output = np.full(len(anchors), np.nan, dtype=float)
    valid = (anchors >= 0) & (anchors < len(values))
    output[valid] = values[anchors[valid]]
    return output


def coordinate_surfaces(base: DataFrame, pair: str) -> dict[str, Any]:
    actual_anchor, abnormal = prior_abnormal_anchors(base)
    random_anchor = random_recent_activity_anchors(base, pair, abnormal)
    opened = pd.to_numeric(base["open"], errors="coerce").to_numpy(dtype=float)
    closed = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")

    def coordinates(anchors: np.ndarray) -> dict[str, np.ndarray]:
        event_open = values_at_anchor(opened, anchors)
        event_close = values_at_anchor(closed, anchors)
        event_high = values_at_anchor(high, anchors)
        event_low = values_at_anchor(low, anchors)
        return {
            "event_open": event_open,
            "event_close": event_close,
            "event_body_midpoint": (event_open + event_close) / 2.0,
            "event_high": event_high,
            "event_low": event_low,
            "event_typical_price": (event_high + event_low + event_close) / 3.0,
        }

    def source_dates(anchors: np.ndarray) -> Series:
        result = Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
        valid = anchors >= 0
        result.loc[valid] = dates.iloc[anchors[valid]].to_numpy()
        return result

    return {
        "abnormal": abnormal,
        "actual_anchor": actual_anchor,
        "random_anchor": random_anchor,
        "actual": coordinates(actual_anchor),
        "random": coordinates(random_anchor),
        "actual_source_open": source_dates(actual_anchor),
        "random_source_open": source_dates(random_anchor),
    }


def support_events(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    level_name: str,
    level: np.ndarray,
    control: str,
    event_kind: str,
    source_open: Series,
) -> DataFrame:
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    width = ZONE_HALF_WIDTH_ATR * atr
    valid = np.isfinite(level) & (level > 0.0) & np.isfinite(width) & (width > 0.0)
    contact = valid & (high >= level - width) & (low <= level + width)
    if event_kind == "contact":
        condition = contact
    elif event_kind == "near_miss":
        expanded = valid & (high >= level - 2.0 * width) & (low <= level + 2.0 * width)
        condition = expanded & ~contact
    else:
        raise ValueError(event_kind)
    starts = g0.episode_start_mask(condition, level, width, cooldown=6)
    starts[max(len(starts) - max(HORIZONS), 0) :] = False
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    period = g18d.assign_confirmation_period(dates, cohort)
    starts &= period.ne("outside_g18_confirmation").to_numpy(dtype=bool)
    indexes = np.flatnonzero(starts)
    approach = np.select(
        [
            pre_close[indexes] < level[indexes] - width[indexes],
            pre_close[indexes] > level[indexes] + width[indexes],
        ],
        ["from_below", "from_above"],
        default="already_inside_or_unclear",
    )
    return DataFrame(
        {
            "cohort": cohort,
            "pair": pair,
            "period": period.iloc[indexes].to_numpy(),
            "control": control,
            "event_time": dates.iloc[indexes].to_numpy(),
            "base_index": indexes,
            "level_family": "abnormal_candle_structural_levels",
            "level_name": level_name,
            "level_price": level[indexes],
            "zone_half_width": width[indexes],
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
            "base_atr": atr[indexes],
            "pre_distance_atr": np.abs(level[indexes] - pre_close[indexes])
            / atr[indexes],
            "approach_state": approach,
            "source_open": source_open.iloc[indexes].to_numpy(),
            "match_tier": "not_applicable",
        }
    )


def matched_random_time_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    level_name: str,
    actual: DataFrame,
) -> DataFrame:
    """Create outcome-blind pseudo-level contacts matched to actual approach geometry."""
    if actual.empty:
        return DataFrame()
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    period = g18d.assign_confirmation_period(dates, cohort).astype(str).to_numpy()
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    width = ZONE_HALF_WIDTH_ATR * atr
    excluded = np.zeros(len(base), dtype=bool)
    excluded[actual["base_index"].to_numpy(dtype=int)] = True
    excluded = np.convolve(excluded.astype(np.int8), np.ones(13, dtype=np.int8), mode="same") > 0
    eligible = (
        np.isfinite(pre_close)
        & np.isfinite(atr)
        & (atr > 0.0)
        & (np.arange(len(base)) < len(base) - max(HORIZONS))
        & ~excluded
    )
    match = actual[["period", "approach_state", "pre_distance_atr"]].copy()
    match["distance_band"] = np.digitize(
        match["pre_distance_atr"].to_numpy(dtype=float),
        [0.10, 0.25, 0.50, 1.0, 2.0],
    )
    selected: list[tuple[int, float, str]] = []
    for key, group in match.groupby(
        ["period", "approach_state", "distance_band"],
        sort=False,
        dropna=False,
    ):
        wanted_period, approach, _ = key
        distance = float(group["pre_distance_atr"].median())
        if approach == "from_below":
            sign = 1.0
        elif approach == "from_above":
            sign = -1.0
        else:
            sign = 0.0
            distance = 0.0
        if sign:
            distance = max(distance, ZONE_HALF_WIDTH_ATR * 1.01)
        pseudo_level = pre_close + sign * distance * atr
        contact = (high >= pseudo_level - width) & (low <= pseudo_level + width)
        candidate = np.flatnonzero(eligible & contact & (period == str(wanted_period)))
        rng = np.random.default_rng(
            g0.stable_hash_int(
                f"g21-matched-random|{pair}|{level_name}|{wanted_period}|{approach}|{key[2]}"
            )
        )
        blocked = np.zeros(len(base), dtype=bool)
        count = 0
        for position in rng.permutation(candidate):
            position = int(position)
            if blocked[position]:
                continue
            selected.append((position, float(pseudo_level[position]), "exact_geometry"))
            blocked[max(0, position - 6) : min(len(base), position + 7)] = True
            count += 1
            if count >= len(group):
                break
    if not selected:
        return DataFrame()
    selected.sort(key=lambda item: item[0])
    indexes = np.asarray([item[0] for item in selected], dtype=int)
    levels = np.asarray([item[1] for item in selected], dtype=float)
    approach = np.select(
        [
            pre_close[indexes] < levels - width[indexes],
            pre_close[indexes] > levels + width[indexes],
        ],
        ["from_below", "from_above"],
        default="already_inside_or_unclear",
    )
    return DataFrame(
        {
            "cohort": cohort,
            "pair": pair,
            "period": period[indexes],
            "control": "matched_random_time",
            "event_time": dates.iloc[indexes].to_numpy(),
            "base_index": indexes,
            "level_family": "abnormal_candle_structural_levels",
            "level_name": level_name,
            "level_price": levels,
            "zone_half_width": width[indexes],
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
            "base_atr": atr[indexes],
            "pre_distance_atr": np.abs(levels - pre_close[indexes]) / atr[indexes],
            "approach_state": approach,
            "source_open": pd.NaT,
            "match_tier": [item[2] for item in selected],
        }
    )


def pair_support(pair: str, cohort: str, overwrite: bool) -> dict[str, Any]:
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control"])
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "existing",
        }
    base, _ = g20s.base_and_state(pair, cohort)
    surfaces = coordinate_surfaces(base, pair)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    parts: list[DataFrame] = []
    for level_name in LEVELS:
        actual_level = np.asarray(surfaces["actual"][level_name], dtype=float)
        actual = support_events(
            base,
            pair=pair,
            cohort=cohort,
            level_name=level_name,
            level=actual_level,
            control="actual",
            event_kind="contact",
            source_open=surfaces["actual_source_open"],
        )
        parts.append(actual)
        parts.append(
            matched_random_time_support(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                actual=actual,
            )
        )
        parts.append(
            support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=np.asarray(surfaces["random"][level_name], dtype=float),
                control="random_recent_high_activity_candle",
                event_kind="contact",
                source_open=surfaces["random_source_open"],
            )
        )
        parts.append(
            support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=actual_level,
                control="near_miss",
                event_kind="near_miss",
                source_open=surfaces["actual_source_open"],
            )
        )
        parts.append(
            support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=g0.shift_array(actual_level, 72),
                control="stale_72h",
                event_kind="contact",
                source_open=surfaces["actual_source_open"].shift(72),
            )
        )
        direction = 1.0 if g0.stable_hash_int(f"{pair}|{level_name}") % 2 == 0 else -1.0
        parts.append(
            support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=actual_level + direction * 2.0 * atr,
                control="price_shift",
                event_kind="contact",
                source_open=surfaces["actual_source_open"],
            )
        )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "level_name", "event_time"], inplace=True, ignore_index=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "abnormal_candles": int(np.asarray(surfaces["abnormal"]).sum()),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def cohort_pairs() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for cohort, manifest_path in g17l.COHORT_MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend((cohort, str(pair)) for pair in manifest["data"]["pairs"])
    return rows


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation21_structural_outcomes":
            raise ValueError("Invalid Generation 21 structural support freeze.")
        return manifest
    pairs = cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    inventory = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g21_structural_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_structural_outcomes",
        "branch_id": "g21d_abnormal_candle_structural_levels",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "completed_candle_levels_only": True,
        "levels": list(LEVELS),
        "controls": list(CONTROLS),
        "random_recent_high_activity_definition": RANDOM_ACTIVITY_DEFINITION,
        "inventory": inventory,
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def metric_summary(events: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    rows: list[DataFrame] = []
    for metric in (
        "any_recross",
        "repeated_recross",
        "dwell_fraction",
        "future_volume_ratio",
    ):
        for horizon in HORIZONS:
            column = f"metric__{metric}_h{horizon}"
            grouped = (
                events[keys + [column]]
                .dropna(subset=[column])
                .groupby(keys, observed=True, sort=False)[column]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
            )
            grouped["metric"] = metric
            grouped["horizon_hours"] = horizon
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    if g0.sha256_file(ANALYSIS_PATH) != manifest["source_contracts"]["analysis_script"]["sha256"]:
        raise ValueError("Generation 21 structural analysis changed after support freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g21_structural_levels_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen structural support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        events["g18_period"] = events["period"].astype(str)
        events["scope_kind"] = "single_structural_coordinate"
        events["scope_value"] = events["level_name"].astype(str)
        parts.append(events)
        print(
            json.dumps(
                {"phase": "g21_structural_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    events = pd.concat(parts, ignore_index=True, sort=False)
    summary = metric_summary(events)
    contrasts = g18d.paired_contrasts(summary, CONTROLS)
    question_keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g21_structural_pair_contrasts.csv",
        "scores": run_dir / "g21_structural_period_scores.csv",
        "decisions": run_dir / "g21_structural_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation21_structural_levels",
        "branch_completed": "g21d_abnormal_candle_structural_levels",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "indicator_files_modified": False,
        },
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--prepare-support", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_support:
        print(json.dumps(freeze_support(overwrite=args.overwrite), indent=2))
        return 0
    return execute(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
