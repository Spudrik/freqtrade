"""Test Generation 20's frozen causal change-point anchored-VWAP levels."""

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
    market_reaction_zone_generation20_freeze as g20z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)


DEFAULT_RUN_ID = "g20_change_point_avwap_20260827a"
DEFAULT_SUPPORT_ID = "g20_change_point_support_20260827a"
ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = g20z.OUTPUT_ROOT / "change_point_avwap"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation20_branches"
    / "g20_broad_siblings"
    / "change_point_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g20_change_point_support_freeze.json"
HORIZONS = g20z.HORIZONS_HOURS
LEVELS = (
    "anchored_vwap",
    "anchored_vwap_plus_1p5_sigma",
    "anchored_vwap_minus_1p5_sigma",
)
CONTROLS = (
    "rolling_vwap_168h_width_matched",
    "random_recent_anchor",
    "near_miss",
    "stale_72h",
    "price_shift",
)
MAXIMUM_ANCHOR_AGE = 168
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g20z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g20z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g20d_change_point_anchored_vwap_level"
    )
    expected = {
        "volume_zscore_minimum": 2.0,
        "range_over_atr_minimum": 1.5,
        "absolute_return_over_atr_minimum": 0.75,
        "maximum_anchor_age_hours": MAXIMUM_ANCHOR_AGE,
        "completed_candle_only": True,
    }
    if frozen.get("status") != "frozen_before_generation20_outcomes":
        raise ValueError("Generation 20 batch is not frozen.")
    if branch["change_point_definition"] != expected:
        raise ValueError("Generation 20 change-point definition drifted after freeze.")
    return frozen


def interval_sum(cumulative: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    result = np.full(len(starts), np.nan, dtype=float)
    valid = (starts >= 0) & (ends >= starts) & (ends < len(cumulative))
    if not valid.any():
        return result
    selected_starts = starts[valid]
    selected_ends = ends[valid]
    values = cumulative[selected_ends].copy()
    has_prefix = selected_starts > 0
    values[has_prefix] -= cumulative[selected_starts[has_prefix] - 1]
    result[valid] = values
    return result


def anchored_statistics(
    price: np.ndarray,
    volume: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.where(np.isfinite(volume) & (volume > 0.0), volume, 0.0)
    value = np.where(np.isfinite(price), price, 0.0)
    weight_sum = interval_sum(np.cumsum(weight), starts, ends)
    weighted_price = interval_sum(np.cumsum(weight * value), starts, ends)
    weighted_square = interval_sum(np.cumsum(weight * value * value), starts, ends)
    mean = np.divide(
        weighted_price,
        weight_sum,
        out=np.full(len(starts), np.nan),
        where=np.isfinite(weight_sum) & (weight_sum > 0.0),
    )
    variance = np.divide(
        weighted_square,
        weight_sum,
        out=np.full(len(starts), np.nan),
        where=np.isfinite(weight_sum) & (weight_sum > 0.0),
    ) - np.square(mean)
    sigma = np.sqrt(np.maximum(variance, 0.0))
    return mean, sigma


def change_point_surfaces(base: DataFrame, pair: str) -> dict[str, Any]:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    atr = pd.to_numeric(base["base_atr"], errors="coerce")
    volume_mean = volume.shift(1).rolling(168, min_periods=168).mean()
    volume_std = volume.shift(1).rolling(168, min_periods=168).std(ddof=0)
    volume_z = (volume - volume_mean).div(volume_std.replace(0.0, np.nan))
    range_over_atr = (high - low).div(atr)
    return_over_atr = close.diff().abs().div(atr)
    change_point = (
        volume_z.ge(2.0)
        & range_over_atr.ge(1.5)
        & return_over_atr.ge(0.75)
    )
    row_index = np.arange(len(base), dtype=np.int64)
    last_candidate = Series(
        np.where(change_point, row_index, np.nan), index=base.index
    ).ffill()
    # A candidate candle becomes usable only on the following hourly row.
    anchor = pd.to_numeric(last_candidate.shift(1), errors="coerce").to_numpy(dtype=float)
    starts = np.where(np.isfinite(anchor), anchor, -1).astype(np.int64)
    ends = row_index - 1
    ages = row_index - starts
    valid = np.isfinite(anchor) & (ages >= 1) & (ages <= MAXIMUM_ANCHOR_AGE)
    starts[~valid] = -1
    typical = ((high + low + close) / 3.0).to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    avwap, sigma = anchored_statistics(typical, volume_values, starts, ends)
    avwap[~valid] = np.nan
    sigma[~valid] = np.nan

    rolling_starts = row_index - MAXIMUM_ANCHOR_AGE
    rolling_starts[rolling_starts < 0] = -1
    rolling_mean, rolling_sigma = anchored_statistics(
        typical, volume_values, rolling_starts, ends
    )

    random_lookback = np.array(
        [
            24
            + g0.stable_hash_int(
                f"g20-random-anchor|{pair}|{pd.Timestamp(date).isoformat()}"
            )
            % 145
            for date in pd.to_datetime(base["date"], utc=True)
        ],
        dtype=np.int64,
    )
    random_starts = row_index - random_lookback
    random_starts[random_starts < 0] = -1
    random_mean, random_sigma = anchored_statistics(
        typical, volume_values, random_starts, ends
    )
    dates = pd.to_datetime(base["date"], utc=True)
    source_open = Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    source_open.loc[valid] = dates.iloc[starts[valid]].to_numpy()
    return {
        "change_point": change_point.to_numpy(dtype=bool),
        "anchor_index": starts,
        "anchor_age_hours": np.where(valid, ages, np.nan),
        "source_open": source_open,
        "actual": {
            "anchored_vwap": avwap,
            "anchored_vwap_plus_1p5_sigma": avwap + 1.5 * sigma,
            "anchored_vwap_minus_1p5_sigma": avwap - 1.5 * sigma,
        },
        "rolling": {
            "anchored_vwap": rolling_mean,
            "anchored_vwap_plus_1p5_sigma": rolling_mean + 1.5 * rolling_sigma,
            "anchored_vwap_minus_1p5_sigma": rolling_mean - 1.5 * rolling_sigma,
        },
        "random": {
            "anchored_vwap": random_mean,
            "anchored_vwap_plus_1p5_sigma": random_mean + 1.5 * random_sigma,
            "anchored_vwap_minus_1p5_sigma": random_mean - 1.5 * random_sigma,
        },
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
    dates = pd.to_datetime(base["date"], utc=True)
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
            "level_family": "change_point_anchored_vwap",
            "level_name": level_name,
            "level_price": level[indexes],
            "zone_half_width": width[indexes],
            "base_atr": atr[indexes],
            "pre_distance_atr": np.abs(level[indexes] - pre_close[indexes])
            / atr[indexes],
            "approach_state": approach,
            "source_open": source_open.iloc[indexes].to_numpy(),
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
    surfaces = change_point_surfaces(base, pair)
    source_open = surfaces["source_open"]
    parts: list[DataFrame] = []
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    for level_name in LEVELS:
        actual = np.asarray(surfaces["actual"][level_name], dtype=float)
        controls = {
            "actual": actual,
            "rolling_vwap_168h_width_matched": np.asarray(
                surfaces["rolling"][level_name], dtype=float
            ),
            "random_recent_anchor": np.asarray(
                surfaces["random"][level_name], dtype=float
            ),
            "stale_72h": g0.shift_array(actual, 72),
            "price_shift": actual
            + (1.0 if g0.stable_hash_int(f"{pair}|{level_name}") % 2 == 0 else -1.0)
            * 2.0
            * atr,
        }
        for control, level in controls.items():
            parts.append(
                support_events(
                    base,
                    pair=pair,
                    cohort=cohort,
                    level_name=level_name,
                    level=level,
                    control=control,
                    event_kind="contact",
                    source_open=source_open.shift(72)
                    if control == "stale_72h"
                    else source_open,
                )
            )
        parts.append(
            support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=actual,
                control="near_miss",
                event_kind="near_miss",
                source_open=source_open,
            )
        )
    support = pd.concat(parts, ignore_index=True, sort=False)
    support.sort_values(
        ["control", "level_name", "event_time"], inplace=True, ignore_index=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "change_points": int(np.asarray(surfaces["change_point"]).sum()),
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
        if manifest.get("status") != "frozen_before_generation20_change_point_outcomes":
            raise ValueError("Invalid Generation 20 change-point support freeze.")
        return manifest
    inventory = []
    pairs = cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g20_change_point_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation20_change_point_outcomes",
        "branch_id": "g20d_change_point_anchored_vwap_level",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "completed_candle_change_points_only": True,
        "levels": list(LEVELS),
        "controls": list(CONTROLS),
        "inventory": inventory,
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
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
    for metric in ("any_recross", "repeated_recross", "dwell_fraction", "future_volume_ratio"):
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
    if (
        g0.sha256_file(ANALYSIS_PATH)
        != manifest["source_contracts"]["analysis_script"]["sha256"]
    ):
        raise ValueError("Generation 20 AVWAP analysis changed after support freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g20_change_point_avwap_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen change-point support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        events["g18_period"] = events["period"].astype(str)
        events["scope_kind"] = "single_level"
        events["scope_value"] = events["level_name"].astype(str)
        parts.append(events)
        print(
            json.dumps(
                {"phase": "g20_change_point_outcomes", "processed": number, "total": 20}
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
        "contrasts": run_dir / "g20_change_point_pair_contrasts.csv",
        "scores": run_dir / "g20_change_point_period_scores.csv",
        "decisions": run_dir / "g20_change_point_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation20_change_point_avwap",
        "branch_completed": "g20d_change_point_anchored_vwap_level",
        "strict_rows": int(
            decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "point_rows": int(
            decisions["status"].eq("point_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "indicator_files_modified": False,
        },
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
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
