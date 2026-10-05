"""Test frozen causal log-price trend-channel levels."""

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
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g22_trend_channels_20260827a"
DEFAULT_SUPPORT_ID = "g22_trend_channel_support_20260827a"
RECORD_ROOT = g22z.OUTPUT_ROOT / "trend_channels"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "trend_channel_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g22_trend_channel_support_freeze.json"
HORIZONS = g22z.HORIZONS_HOURS
LOOKBACKS = (72, 168, 720)
COORDINATES = ("trend_centre", "upper_1p5_residual_sd", "lower_1p5_residual_sd")
CONTROLS = (
    "matched_random_time",
    "flat_rolling_log_mean",
    "near_miss",
    "stale_72h",
    "price_shift",
)
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g22z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22c_causal_trend_channel_boundaries"
    )
    if frozen.get("status") != "frozen_before_generation22_outcomes":
        raise ValueError("Generation 22 batch is not frozen.")
    if tuple(branch["lookback_hours"]) != LOOKBACKS:
        raise ValueError("Generation 22 trend lookbacks drifted.")
    if tuple(branch["coordinates"]) != COORDINATES:
        raise ValueError("Generation 22 trend coordinates drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 22 trend controls drifted.")
    return frozen


def causal_channel(close: Series, lookback: int) -> dict[str, np.ndarray]:
    """Fit t-lookback..t-1 in log space and extrapolate exactly once to t."""
    numeric = pd.to_numeric(close, errors="coerce")
    log_close = np.log(numeric.where(numeric > 0.0))
    prior = log_close.shift(1)
    global_x = Series(np.arange(len(prior), dtype=float), index=prior.index)
    n = float(lookback)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    sxx = sum_x2 - sum_x * sum_x / n
    sum_y = prior.rolling(lookback, min_periods=lookback).sum()
    sum_y2 = prior.pow(2).rolling(lookback, min_periods=lookback).sum()
    sum_global_xy = (prior * global_x).rolling(lookback, min_periods=lookback).sum()
    window_start = global_x - (lookback - 1.0)
    sum_local_xy = sum_global_xy - window_start * sum_y
    sxy = sum_local_xy - sum_x * sum_y / n
    slope = sxy / sxx
    intercept = sum_y / n - slope * (lookback - 1.0) / 2.0
    predicted_log = intercept + slope * lookback
    sst = sum_y2 - sum_y.pow(2) / n
    sse = (sst - slope * sxy).clip(lower=0.0)
    residual_sd = np.sqrt(sse / (n - 2.0))
    flat_mean = sum_y / n
    flat_sd = np.sqrt((sst / (n - 1.0)).clip(lower=0.0))
    return {
        "trend_centre": np.exp(predicted_log.to_numpy(dtype=float)),
        "upper_1p5_residual_sd": np.exp(
            (predicted_log + 1.5 * residual_sd).to_numpy(dtype=float)
        ),
        "lower_1p5_residual_sd": np.exp(
            (predicted_log - 1.5 * residual_sd).to_numpy(dtype=float)
        ),
        "flat_trend_centre": np.exp(flat_mean.to_numpy(dtype=float)),
        "flat_upper_1p5_residual_sd": np.exp(
            (flat_mean + 1.5 * flat_sd).to_numpy(dtype=float)
        ),
        "flat_lower_1p5_residual_sd": np.exp(
            (flat_mean - 1.5 * flat_sd).to_numpy(dtype=float)
        ),
        "slope": slope.to_numpy(dtype=float),
        "residual_sd": residual_sd.to_numpy(dtype=float),
    }


def channel_surfaces(base: DataFrame, lookback: int) -> dict[str, Any]:
    channel = causal_channel(base["close"], lookback)
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    source = Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    eligible = np.arange(len(base)) >= lookback
    source.loc[eligible] = dates.shift(lookback).loc[eligible].to_numpy()
    flat = {
        coordinate: np.asarray(channel[f"flat_{coordinate}"], dtype=float)
        for coordinate in COORDINATES
    }
    actual = {
        coordinate: np.asarray(channel[coordinate], dtype=float)
        for coordinate in COORDINATES
    }
    return {
        "actual": actual,
        "flat": flat,
        "source_open": source,
        "slope": channel["slope"],
        "residual_sd": channel["residual_sd"],
    }


def relabel(frame: DataFrame, lookback: int, coordinate: str) -> DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["level_family"] = "causal_trend_channel_boundaries"
    frame["lookback_hours"] = lookback
    frame["coordinate"] = coordinate
    frame["level_name"] = f"trend_lb{lookback}__{coordinate}"
    return frame


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
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    parts: list[DataFrame] = []
    valid_counts: dict[str, int] = {}
    for lookback in LOOKBACKS:
        surfaces = channel_surfaces(base, lookback)
        valid_counts[str(lookback)] = int(np.isfinite(surfaces["actual"]["trend_centre"]).sum())
        for coordinate in COORDINATES:
            level_name = f"trend_lb{lookback}__{coordinate}"
            actual_level = np.asarray(surfaces["actual"][coordinate], dtype=float)
            actual = relabel(
                g21d.support_events(
                    base,
                    pair=pair,
                    cohort=cohort,
                    level_name=level_name,
                    level=actual_level,
                    control="actual",
                    event_kind="contact",
                    source_open=surfaces["source_open"],
                ),
                lookback,
                coordinate,
            )
            parts.append(actual)
            parts.append(
                relabel(
                    g21d.matched_random_time_support(
                        base,
                        pair=pair,
                        cohort=cohort,
                        level_name=level_name,
                        actual=actual,
                    ),
                    lookback,
                    coordinate,
                )
            )
            parts.append(
                relabel(
                    g21d.support_events(
                        base,
                        pair=pair,
                        cohort=cohort,
                        level_name=level_name,
                        level=np.asarray(surfaces["flat"][coordinate], dtype=float),
                        control="flat_rolling_log_mean",
                        event_kind="contact",
                        source_open=surfaces["source_open"],
                    ),
                    lookback,
                    coordinate,
                )
            )
            parts.append(
                relabel(
                    g21d.support_events(
                        base,
                        pair=pair,
                        cohort=cohort,
                        level_name=level_name,
                        level=actual_level,
                        control="near_miss",
                        event_kind="near_miss",
                        source_open=surfaces["source_open"],
                    ),
                    lookback,
                    coordinate,
                )
            )
            parts.append(
                relabel(
                    g21d.support_events(
                        base,
                        pair=pair,
                        cohort=cohort,
                        level_name=level_name,
                        level=g0.shift_array(actual_level, 72),
                        control="stale_72h",
                        event_kind="contact",
                        source_open=surfaces["source_open"].shift(72),
                    ),
                    lookback,
                    coordinate,
                )
            )
            direction = (
                1.0
                if g0.stable_hash_int(f"g22-trend-shift|{pair}|{level_name}") % 2 == 0
                else -1.0
            )
            parts.append(
                relabel(
                    g21d.support_events(
                        base,
                        pair=pair,
                        cohort=cohort,
                        level_name=level_name,
                        level=actual_level + direction * 2.0 * atr,
                        control="price_shift",
                        event_kind="contact",
                        source_open=surfaces["source_open"],
                    ),
                    lookback,
                    coordinate,
                )
            )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "lookback_hours", "coordinate", "event_time"],
        inplace=True,
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "valid_surface_rows": valid_counts,
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation22_trend_outcomes":
            raise ValueError("Invalid Generation 22 trend support freeze.")
        return manifest
    pairs = g22a.cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    inventory = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g22_trend_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_trend_outcomes",
        "branch_id": "g22c_causal_trend_channel_boundaries",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "completed_candles_only": True,
        "lookbacks": list(LOOKBACKS),
        "coordinates": list(COORDINATES),
        "controls": list(CONTROLS),
        "inventory": inventory,
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
            "shared_period_helper": artifact(g22a.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    contracts = manifest["source_contracts"]
    for path, key in (
        (ANALYSIS_PATH, "analysis_script"),
        (g21d.ANALYSIS_PATH, "support_event_helper"),
        (g22a.ANALYSIS_PATH, "shared_period_helper"),
    ):
        if g0.sha256_file(path) != contracts[key]["sha256"]:
            raise ValueError(f"Frozen Generation 22 trend dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g22_trend_channels_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen trend support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        events["g18_period"] = events["period"].astype(str)
        events["scope_kind"] = "lookback_and_coordinate"
        events["scope_value"] = (
            "lb"
            + events["lookback_hours"].astype(int).astype(str)
            + "__"
            + events["coordinate"].astype(str)
        )
        parts.append(events)
        print(
            json.dumps(
                {"phase": "g22_trend_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    events = pd.concat(parts, ignore_index=True, sort=False)
    summary = g22a.metric_summary(events)
    contrasts = g18d.paired_contrasts(summary, CONTROLS)
    question_keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g22_trend_pair_contrasts.csv",
        "scores": run_dir / "g22_trend_period_scores.csv",
        "decisions": run_dir / "g22_trend_decisions.csv",
    }
    for name, frame in (("contrasts", contrasts), ("scores", scores), ("decisions", decisions)):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_trend_channels",
        "branch_completed": "g22c_causal_trend_channel_boundaries",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "trend_definition_tuned_after_outcomes": False,
        },
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
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
