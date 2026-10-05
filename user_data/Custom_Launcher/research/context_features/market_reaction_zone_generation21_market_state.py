"""Run Generation 21's level-free multi-timeframe market-state sibling B."""

from __future__ import annotations

# Bind numerical pools before pandas/numpy imports.
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
    market_reaction_zone_generation1_localization as g1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
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
DEFAULT_RUN_ID = "g21_market_state_20260827a"
DEFAULT_SUPPORT_ID = "g21_market_state_support_20260827a"
RECORD_ROOT = g21z.OUTPUT_ROOT / "market_state"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation21_branches"
    / "g21_broad_siblings"
    / "market_state_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g21_market_state_support_freeze.json"
SOURCE_TIMEFRAMES = ("1h", "4h", "8h")
FORWARD_STEPS = (1, 2, 4)
CONTROLS = (
    "matched_moderate_state",
    "causal_72h_stale_state",
    "within_pair_time_shuffle",
    "same_time_other_coin_state",
)
BLOCKS = {
    "volume_pressure": (
        "relative_volume",
        "volume_acceleration",
        "absolute_pressure",
        "pressure_persistence",
    ),
    "volatility_range": (
        "atr_fraction",
        "prior_range_atr",
        "bollinger_width",
        "range_contraction",
    ),
    "trend_momentum": (
        "ema20_slope",
        "ma_separation",
        "return_slope",
        "return_acceleration",
        "adx14",
        "rsi14_centered",
        "macd_histogram",
    ),
}
ABSOLUTE_SCORE_COLUMNS = {
    "volume_acceleration",
    "absolute_pressure",
    "pressure_persistence",
    "ema20_slope",
    "return_slope",
    "return_acceleration",
    "rsi14_centered",
    "macd_histogram",
}
MIN_PAIR_ROWS = 10


def artifact(path: Path) -> dict[str, Any]:
    return g21z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g21z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g21b_multitimeframe_market_state_reaction_atlas"
    )
    if frozen.get("status") != "frozen_before_generation21_outcomes":
        raise ValueError("Generation 21 batch is not frozen.")
    if tuple(branch["source_timeframes"]) != SOURCE_TIMEFRAMES:
        raise ValueError("Generation 21 market-state timeframes drifted after freeze.")
    return frozen


def timeframe_hours(timeframe: str) -> int:
    return int(timeframe.removesuffix("h"))


def causal_features(base: DataFrame) -> DataFrame:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    candle_range = high - low
    pressure = (2.0 * close - high - low).div(candle_range.replace(0.0, np.nan))
    prior_close = close.shift(1)
    prior_volume = volume.shift(1)
    prior_range = candle_range.shift(1)
    atr = g0.wilder_atr(base, 14).shift(1)
    output = DataFrame({"date": pd.to_datetime(base["date"], utc=True)})
    output["relative_volume"] = prior_volume.div(
        prior_volume.rolling(24, min_periods=24).median()
    )
    output["volume_acceleration"] = np.log1p(prior_volume.clip(lower=0.0)).diff(3)
    output["absolute_pressure"] = pressure.shift(1).abs()
    output["pressure_persistence"] = pressure.shift(1).rolling(
        6, min_periods=6
    ).mean().abs()
    output["atr_fraction"] = atr.div(prior_close.abs())
    output["prior_range_atr"] = prior_range.div(atr)
    bb_mid = prior_close.rolling(20, min_periods=20).mean()
    bb_std = prior_close.rolling(20, min_periods=20).std(ddof=0)
    output["bollinger_width"] = (4.0 * bb_std).div(bb_mid.abs())
    output["range_contraction"] = prior_range.div(
        prior_range.rolling(24, min_periods=24).median()
    )
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean().shift(1)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean().shift(1)
    output["ema20_slope"] = ema20.pct_change(4, fill_method=None)
    output["ma_separation"] = (ema20 - ema50).abs().div(prior_close.abs())
    prior_return = close.pct_change(fill_method=None).shift(1)
    output["return_slope"] = prior_return.rolling(6, min_periods=6).mean()
    output["return_acceleration"] = (
        prior_return.rolling(3, min_periods=3).mean()
        - prior_return.rolling(12, min_periods=12).mean()
    )
    output["adx14"] = g18d.adx(base, 14).shift(1)
    output["rsi14_centered"] = g18d.rsi(close, 14).shift(1) - 50.0
    macd = (
        close.ewm(span=12, adjust=False, min_periods=26).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    output["macd_histogram"] = (macd - signal).shift(1)
    output["base_atr"] = atr
    output["pre_volume_median_24"] = volume.rolling(24, min_periods=24).median().shift(1)
    output["pre_range_median_24"] = candle_range.rolling(
        24, min_periods=24
    ).median().shift(1)
    output["pre_close"] = prior_close
    return output


def add_block_states(
    features: DataFrame, cohort: str
) -> tuple[DataFrame, dict[str, Any]]:
    cutoff = min(
        pd.Timestamp(item["start_utc"])
        for item in g18d.g18z.CONFIRMATION_PERIODS[cohort]
    )
    calibration_mask = features["date"] < cutoff
    registry: dict[str, Any] = {}
    output = features.copy()
    for block, columns in BLOCKS.items():
        percentiles = []
        counts: dict[str, int] = {}
        for column in columns:
            calibration = pd.to_numeric(
                output.loc[calibration_mask, column], errors="coerce"
            ).to_numpy(dtype=float)
            values = pd.to_numeric(output[column], errors="coerce")
            if column in ABSOLUTE_SCORE_COLUMNS:
                calibration = np.abs(calibration)
                values = values.abs()
            counts[column] = int(np.isfinite(calibration).sum())
            percentiles.append(g20s.empirical_percentile(values, calibration))
        matrix = np.column_stack(percentiles)
        finite_count = np.isfinite(matrix).sum(axis=1)
        score = np.divide(
            np.nansum(matrix, axis=1),
            finite_count,
            out=np.full(len(output), np.nan),
            where=finite_count > 0,
        )
        output[f"score__{block}"] = score
        calibration_score = pd.Series(score[calibration_mask.to_numpy()]).dropna()
        if len(calibration_score) < 168:
            raise ValueError(f"Insufficient {block} calibration rows.")
        lower, upper = calibration_score.quantile([1 / 3, 2 / 3]).to_numpy()
        output[f"state__{block}"] = np.select(
            [score <= lower, score > upper],
            ["quiet", "extreme"],
            default="moderate",
        )
        output.loc[~np.isfinite(score), f"state__{block}"] = "unavailable"
        registry[block] = {
            "lower_tertile": float(lower),
            "upper_tertile": float(upper),
            "calibration_end_exclusive_utc": cutoff.isoformat(),
            "calibration_rows": len(calibration_score),
            "feature_rows": counts,
        }
    return output, registry


def add_calendar_features(frame: DataFrame) -> None:
    dates = pd.to_datetime(frame["date"], utc=True)
    hour = dates.dt.hour + dates.dt.minute / 60.0
    day = dates.dt.dayofweek
    frame["calendar_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["calendar_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["calendar_day_sin"] = np.sin(2.0 * np.pi * day / 7.0)
    frame["calendar_day_cos"] = np.cos(2.0 * np.pi * day / 7.0)


def add_state_controls(frame: DataFrame, pair: str, timeframe: str) -> None:
    stale_rows = 72 // timeframe_hours(timeframe)
    for block in BLOCKS:
        state = frame[f"state__{block}"].astype(str)
        frame[f"stale_state__{block}"] = state.shift(stale_rows)
        shuffled = Series(index=frame.index, dtype="object")
        for period, indexes in frame.groupby("period", observed=True, sort=False).groups.items():
            positions = list(indexes)
            values = state.loc[positions].to_numpy(dtype=object)
            if not len(values):
                continue
            shift = 1 + g0.stable_hash_int(
                f"g21-state-shuffle|{pair}|{timeframe}|{block}|{period}"
            ) % max(len(values) - 1, 1)
            shuffled.loc[positions] = np.roll(values, shift)
        frame[f"shuffled_state__{block}"] = shuffled


def build_support(pair: str, cohort: str, timeframe: str, overwrite: bool) -> dict[str, Any]:
    output = SUPPORT_ROOT / cohort / timeframe / f"{g0.pair_file_stem(pair)}.parquet"
    source = g0.ohlcv_path(pair, timeframe)
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["date"])
        return {
            "pair": pair,
            "cohort": cohort,
            "source_timeframe": timeframe,
            "rows": len(existing),
            "path": str(output.resolve()),
            "source_path": str(source.resolve()),
            "source_sha256": g0.sha256_file(source),
            "status": "existing",
        }
    base = g0.load_ohlcv(source).sort_values("date").drop_duplicates("date").reset_index(
        drop=True
    )
    features, registry = add_block_states(causal_features(base), cohort)
    features["pair"] = pair
    features["cohort"] = cohort
    features["source_timeframe"] = timeframe
    features["period"] = g18d.assign_confirmation_period(features["date"], cohort)
    features = features.loc[features["period"].ne("outside_g18_confirmation")].copy()
    add_calendar_features(features)
    add_state_controls(features, pair, timeframe)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(features, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "source_timeframe": timeframe,
        "rows": len(features),
        "state_registry": registry,
        "path": str(output.resolve()),
        "source_path": str(source.resolve()),
        "source_sha256": g0.sha256_file(source),
        "status": "built",
    }


def attach_donor_states(inventory: list[dict[str, Any]]) -> None:
    for (cohort, timeframe), items in pd.DataFrame(inventory).groupby(
        ["cohort", "source_timeframe"], observed=True, sort=False
    ):
        records = items.sort_values("pair").to_dict(orient="records")
        if len(records) < 2:
            raise ValueError(f"No donor breadth for {cohort}/{timeframe}.")
        for index, target in enumerate(records):
            donor = records[(index + 1) % len(records)]
            target_path = Path(target["path"])
            frame = pd.read_parquet(target_path)
            donor_frame = pd.read_parquet(
                donor["path"], columns=["date", *(f"state__{block}" for block in BLOCKS)]
            ).rename(
                columns={
                    f"state__{block}": f"donor_state__{block}" for block in BLOCKS
                }
            )
            frame = frame.merge(donor_frame, on="date", how="left", validate="one_to_one")
            frame["donor_pair"] = str(donor["pair"])
            g0.atomic_write_parquet(frame, target_path)


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
        if manifest.get("status") != "frozen_before_generation21_market_state_outcomes":
            raise ValueError("Invalid Generation 21 market-state support freeze.")
        return manifest
    inventory = []
    tasks = [
        (cohort, pair, timeframe)
        for cohort, pair in cohort_pairs()
        for timeframe in SOURCE_TIMEFRAMES
    ]
    for number, (cohort, pair, timeframe) in enumerate(tasks, start=1):
        inventory.append(build_support(pair, cohort, timeframe, overwrite))
        print(
            json.dumps(
                {
                    "phase": "g21_market_state_support",
                    "processed": number,
                    "total": len(tasks),
                }
            ),
            flush=True,
        )
    attach_donor_states(inventory)
    for item in inventory:
        item["sha256"] = g0.sha256_file(Path(item["path"]))
    manifest = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_market_state_outcomes",
        "branch_id": "g21b_multitimeframe_market_state_reaction_atlas",
        "future_outcomes_opened": False,
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "state_blocks": {key: list(value) for key, value in BLOCKS.items()},
        "controls": list(CONTROLS),
        "inventory": inventory,
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def add_outcomes(frame: DataFrame, pair: str, timeframe: str) -> None:
    base = g0.load_ohlcv(g0.ohlcv_path(pair, timeframe)).sort_values("date").drop_duplicates(
        "date"
    ).reset_index(drop=True)
    dates = pd.to_datetime(base["date"], utc=True)
    position_map = Series(np.arange(len(base), dtype=np.int64), index=dates)
    positions = frame["date"].map(position_map).to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError(f"Frozen state rows are absent from {pair} {timeframe} OHLCV.")
    indexes = positions.astype(np.int64)
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(base["volume"], errors="coerce").to_numpy(dtype=float)
    atr = g0.wilder_atr(base, 14).shift(1).to_numpy(dtype=float)[indexes]
    prior_close = close[indexes - 1]
    pre_volume = pd.Series(volume).rolling(24, min_periods=24).median().shift(1).to_numpy()[
        indexes
    ]
    ranges = high - low
    pre_range = pd.Series(ranges).rolling(24, min_periods=24).median().shift(1).to_numpy()[
        indexes
    ]
    forward_offsets = tuple(range(1, max(FORWARD_STEPS) + 1))
    future_high = g20s.indexed_windows(high, indexes, forward_offsets)
    future_low = g20s.indexed_windows(low, indexes, forward_offsets)
    future_close = g20s.indexed_windows(close, indexes, forward_offsets)
    future_volume = g20s.indexed_windows(volume, indexes, forward_offsets)
    for steps in FORWARD_STEPS:
        valid = np.isfinite(future_close[:, :steps]).all(axis=1)
        frame[f"metric__volume_ratio_h{steps}"] = np.divide(
            np.nanmean(future_volume[:, :steps], axis=1),
            pre_volume,
            out=np.full(len(frame), np.nan),
            where=np.isfinite(pre_volume) & (pre_volume > 0.0) & valid,
        )
        frame[f"metric__range_ratio_h{steps}"] = np.divide(
            np.nanmean(future_high[:, :steps] - future_low[:, :steps], axis=1),
            pre_range,
            out=np.full(len(frame), np.nan),
            where=np.isfinite(pre_range) & (pre_range > 0.0) & valid,
        )
        frame[f"metric__absolute_displacement_atr_h{steps}"] = np.divide(
            np.abs(future_close[:, steps - 1] - prior_close),
            atr,
            out=np.full(len(frame), np.nan),
            where=np.isfinite(atr) & (atr > 0.0) & valid,
        )


def matched_contrasts(frame: DataFrame, block: str, timeframe: str) -> list[dict[str, Any]]:
    other_scores = [f"score__{name}" for name in BLOCKS if name != block]
    match_columns = (
        *other_scores,
        "calendar_hour_sin",
        "calendar_hour_cos",
        "calendar_day_sin",
        "calendar_day_cos",
    )
    masks = {
        "actual": frame[f"state__{block}"].eq("extreme"),
        "matched_moderate_state": frame[f"state__{block}"].eq("moderate"),
        "causal_72h_stale_state": frame[f"stale_state__{block}"].eq("extreme"),
        "within_pair_time_shuffle": frame[f"shuffled_state__{block}"].eq("extreme"),
        "same_time_other_coin_state": frame[f"donor_state__{block}"].eq("extreme"),
    }
    rows: list[dict[str, Any]] = []
    gap_hours = max(FORWARD_STEPS) * timeframe_hours(timeframe)
    for (pair, period), cell in frame.groupby(["pair", "period"], observed=True, sort=False):
        local_masks = {key: mask.loc[cell.index] for key, mask in masks.items()}
        actual = cell.loc[local_masks["actual"]].copy().reset_index(drop=True)
        actual["pre_distance_atr"] = 0.0
        actual["event_time"] = actual["date"]
        for comparison in CONTROLS:
            control = cell.loc[local_masks[comparison]].copy().reset_index(drop=True)
            control["pre_distance_atr"] = 0.0
            control["event_time"] = control["date"]
            pairs, audit = g1.nearest_state_pairs(
                actual,
                control,
                state_columns=match_columns,
                pre_distance_atr_caliper=0.10,
                minimum_event_separation_hours=gap_hours,
            )
            if not pairs:
                continue
            left = actual.iloc[[item[0] for item in pairs]].reset_index(drop=True)
            right = control.iloc[[item[1] for item in pairs]].reset_index(drop=True)
            matched = DataFrame(
                {
                    "pair": pair,
                    "period": period,
                    "comparison": comparison,
                    "actual_event_time": left["date"],
                    "control_event_time": right["date"],
                    "level_name": block,
                    "approach_state": "level_free",
                    "match_distance": [item[2] for item in pairs],
                    "pre_distance_atr_abs_difference": 0.0,
                }
            )
            for column in left.columns:
                if column.startswith("metric__"):
                    matched[f"actual__{column}"] = pd.to_numeric(
                        left[column], errors="coerce"
                    ).to_numpy(dtype=float)
                    matched[f"control__{column}"] = pd.to_numeric(
                        right[column], errors="coerce"
                    ).to_numpy(dtype=float)
            matched = g13d.purge_overlapping_hourly_pairs(
                matched,
                separation_hours=gap_hours,
                group_columns=["pair", "period", "comparison"],
            )
            for metric in ("volume_ratio", "range_ratio", "absolute_displacement_atr"):
                for steps in FORWARD_STEPS:
                    column = f"metric__{metric}_h{steps}"
                    left_values = pd.to_numeric(
                        matched[f"actual__{column}"], errors="coerce"
                    )
                    right_values = pd.to_numeric(
                        matched[f"control__{column}"], errors="coerce"
                    )
                    valid = left_values.notna() & right_values.notna()
                    difference = left_values.loc[valid] - right_values.loc[valid]
                    rows.append(
                        {
                            "branch_id": "g21b_multitimeframe_market_state_reaction_atlas",
                            "state_block": block,
                            "source_timeframe": timeframe,
                            "metric": metric,
                            "horizon_steps": steps,
                            "cohort": str(cell["cohort"].iloc[0]),
                            "pair": str(pair),
                            "period": str(period),
                            "comparison": comparison,
                            "event_rows_actual": int(valid.sum()),
                            "event_rows_control": int(valid.sum()),
                            "difference": float(difference.mean())
                            if len(difference)
                            else np.nan,
                            "eligible_pair": bool(valid.sum() >= MIN_PAIR_ROWS),
                            "eligible_actual_before_matching": int(
                                audit["eligible_actual"]
                            ),
                            "state_matchable_actual": int(
                                audit["state_matchable_actual"]
                            ),
                        }
                    )
    return rows


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    if g0.sha256_file(ANALYSIS_PATH) != manifest["source_contracts"]["analysis_script"][
        "sha256"
    ]:
        raise ValueError("Generation 21 market-state analysis changed after support freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g21_market_state_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    contrasts: list[dict[str, Any]] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen market-state support changed: {path}")
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        add_outcomes(frame, str(item["pair"]), str(item["source_timeframe"]))
        for block in BLOCKS:
            contrasts.extend(
                matched_contrasts(frame, block, str(item["source_timeframe"]))
            )
        print(
            json.dumps(
                {"phase": "g21_market_state_outcomes", "processed": number, "total": 60}
            ),
            flush=True,
        )
    contrast_frame = DataFrame.from_records(contrasts)
    question_keys = (
        "branch_id",
        "state_block",
        "source_timeframe",
        "metric",
        "horizon_steps",
    )
    scores = g18d.period_scores(contrast_frame, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g21_market_state_pair_contrasts.csv",
        "scores": run_dir / "g21_market_state_period_scores.csv",
        "decisions": run_dir / "g21_market_state_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrast_frame),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation21_market_state",
        "branch_completed": "g21b_multitimeframe_market_state_reaction_atlas",
        "strict_rows": int(
            decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "point_rows": int(
            decisions["status"].eq("point_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "calculated_levels_used": False,
            "profit_used": False,
            "future_signed_direction_used": False,
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
