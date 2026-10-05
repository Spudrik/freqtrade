"""Test frozen prices from the last completed UTC day, week, and month."""

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
    market_reaction_zone_generation21_structural_levels as g21d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g22_period_landmarks_20260827a"
DEFAULT_SUPPORT_ID = "g22_period_landmark_support_20260827a"
RECORD_ROOT = g22z.OUTPUT_ROOT / "period_landmarks"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "period_landmark_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g22_period_landmark_support_freeze.json"
HORIZONS = g22z.HORIZONS_HOURS
PERIODS = ("previous_utc_day", "previous_iso_week", "previous_utc_month")
COORDINATES = (
    "period_open",
    "period_high",
    "period_low",
    "period_close",
    "period_range_midpoint",
    "period_typical_price",
)
CONTROLS = g22z.LEVEL_CONTROLS
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g22z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22a_completed_period_price_landmarks"
    )
    if frozen.get("status") != "frozen_before_generation22_outcomes":
        raise ValueError("Generation 22 batch is not frozen.")
    if tuple(branch["source_periods"]) != PERIODS:
        raise ValueError("Generation 22 completed-period registry drifted.")
    if tuple(branch["coordinates"]) != COORDINATES:
        raise ValueError("Generation 22 landmark coordinate registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 22 landmark controls drifted.")
    return frozen


def period_keys(dates: Series, period_kind: str) -> Series:
    values = pd.to_datetime(dates, utc=True, errors="raise")
    day = values.dt.floor("1D")
    if period_kind == "previous_utc_day":
        return day
    if period_kind == "previous_iso_week":
        return day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    if period_kind == "previous_utc_month":
        return pd.Series(
            pd.to_datetime(
                {"year": values.dt.year, "month": values.dt.month, "day": 1},
                utc=True,
            ),
            index=dates.index,
        )
    raise ValueError(period_kind)


def expected_hours(keys: pd.DatetimeIndex, period_kind: str) -> np.ndarray:
    if period_kind == "previous_utc_day":
        ends = keys + pd.Timedelta(days=1)
    elif period_kind == "previous_iso_week":
        ends = keys + pd.Timedelta(days=7)
    elif period_kind == "previous_utc_month":
        ends = keys + pd.offsets.MonthBegin(1)
    else:
        raise ValueError(period_kind)
    return ((ends - keys) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def aggregate_periods(base: DataFrame, period_kind: str) -> tuple[Series, DataFrame]:
    keys = period_keys(base["date"], period_kind)
    working = base[["open", "high", "low", "close"]].copy()
    working["period_key"] = keys.to_numpy()
    table = working.groupby("period_key", sort=True).agg(
        period_open=("open", "first"),
        period_high=("high", "max"),
        period_low=("low", "min"),
        period_close=("close", "last"),
        observed_hours=("close", "count"),
    )
    required = expected_hours(pd.DatetimeIndex(table.index), period_kind) * 0.995
    table["complete"] = table["observed_hours"].to_numpy(dtype=float) >= required
    table["period_range_midpoint"] = (
        pd.to_numeric(table["period_high"], errors="coerce")
        + pd.to_numeric(table["period_low"], errors="coerce")
    ) / 2.0
    table["period_typical_price"] = (
        pd.to_numeric(table["period_high"], errors="coerce")
        + pd.to_numeric(table["period_low"], errors="coerce")
        + pd.to_numeric(table["period_close"], errors="coerce")
    ) / 3.0
    table.loc[~table["complete"], list(COORDINATES)] = np.nan
    return keys, table


def mapped_rows(table: DataFrame, keys: Series, shift: int) -> DataFrame:
    shifted = table.shift(shift)
    shifted["source_open"] = table.index.to_series().shift(shift).to_numpy()
    return shifted.reindex(pd.DatetimeIndex(keys)).reset_index(drop=True)


def random_analogue_rows(
    table: DataFrame, keys: Series, pair: str, period_kind: str
) -> DataFrame:
    index = pd.DatetimeIndex(table.index)
    positions = {timestamp: position for position, timestamp in enumerate(index)}
    selected: dict[pd.Timestamp, int] = {}
    for timestamp in pd.DatetimeIndex(keys.unique()):
        position = positions.get(timestamp)
        if position is None:
            continue
        candidates = [
            candidate
            for candidate in range(max(0, position - 9), position - 1)
            if bool(table.iloc[candidate]["complete"])
        ]
        if not candidates:
            continue
        choice = g0.stable_hash_int(
            f"g22-period-analogue|{pair}|{period_kind}|{timestamp.isoformat()}"
        ) % len(candidates)
        selected[timestamp] = candidates[choice]
    rows: list[dict[str, Any]] = []
    for timestamp in pd.DatetimeIndex(keys):
        chosen = selected.get(timestamp)
        if chosen is None:
            rows.append({column: np.nan for column in COORDINATES} | {"source_open": pd.NaT})
            continue
        row = {column: float(table.iloc[chosen][column]) for column in COORDINATES}
        row["source_open"] = index[chosen]
        rows.append(row)
    result = DataFrame.from_records(rows)
    result["source_open"] = pd.to_datetime(result["source_open"], utc=True)
    return result


def completed_period_surfaces(
    base: DataFrame, pair: str, period_kind: str
) -> dict[str, Any]:
    keys, table = aggregate_periods(base, period_kind)
    actual_rows = mapped_rows(table, keys, 1)
    stale_rows = mapped_rows(table, keys, 2)
    random_rows = random_analogue_rows(table, keys, pair, period_kind)
    return {
        "actual": {
            coordinate: pd.to_numeric(actual_rows[coordinate], errors="coerce").to_numpy(
                dtype=float
            )
            for coordinate in COORDINATES
        },
        "stale": {
            coordinate: pd.to_numeric(stale_rows[coordinate], errors="coerce").to_numpy(
                dtype=float
            )
            for coordinate in COORDINATES
        },
        "random": {
            coordinate: pd.to_numeric(random_rows[coordinate], errors="coerce").to_numpy(
                dtype=float
            )
            for coordinate in COORDINATES
        },
        "actual_source_open": pd.to_datetime(actual_rows["source_open"], utc=True),
        "stale_source_open": pd.to_datetime(stale_rows["source_open"], utc=True),
        "random_source_open": pd.to_datetime(random_rows["source_open"], utc=True),
        "period_table": table,
    }


def relabel(frame: DataFrame, period_kind: str, coordinate: str) -> DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["level_family"] = "completed_period_price_landmarks"
    frame["period_kind"] = period_kind
    frame["coordinate"] = coordinate
    frame["level_name"] = f"{period_kind}__{coordinate}"
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
    complete_counts: dict[str, int] = {}
    for period_kind in PERIODS:
        surfaces = completed_period_surfaces(base, pair, period_kind)
        complete_counts[period_kind] = int(surfaces["period_table"]["complete"].sum())
        for coordinate in COORDINATES:
            level_name = f"{period_kind}__{coordinate}"
            actual_level = np.asarray(surfaces["actual"][coordinate], dtype=float)
            actual = g21d.support_events(
                base,
                pair=pair,
                cohort=cohort,
                level_name=level_name,
                level=actual_level,
                control="actual",
                event_kind="contact",
                source_open=surfaces["actual_source_open"],
            )
            actual = relabel(actual, period_kind, coordinate)
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
                    period_kind,
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
                        level=np.asarray(surfaces["random"][coordinate], dtype=float),
                        control="random_recent_analogue",
                        event_kind="contact",
                        source_open=surfaces["random_source_open"],
                    ),
                    period_kind,
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
                        source_open=surfaces["actual_source_open"],
                    ),
                    period_kind,
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
                        level=np.asarray(surfaces["stale"][coordinate], dtype=float),
                        control="stale_definition",
                        event_kind="contact",
                        source_open=surfaces["stale_source_open"],
                    ),
                    period_kind,
                    coordinate,
                )
            )
            direction = (
                1.0
                if g0.stable_hash_int(f"g22-period-shift|{pair}|{level_name}") % 2 == 0
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
                        source_open=surfaces["actual_source_open"],
                    ),
                    period_kind,
                    coordinate,
                )
            )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "period_kind", "coordinate", "event_time"],
        inplace=True,
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "complete_source_periods": complete_counts,
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
        if manifest.get("status") != "frozen_before_generation22_period_outcomes":
            raise ValueError("Invalid Generation 22 period support freeze.")
        return manifest
    pairs = cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    inventory = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g22_period_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_period_outcomes",
        "branch_id": "g22a_completed_period_price_landmarks",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "completed_periods_only": True,
        "periods": list(PERIODS),
        "coordinates": list(COORDINATES),
        "controls": list(CONTROLS),
        "inventory": inventory,
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
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
        "crossing_count",
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
        raise ValueError("Generation 22 landmark analysis changed after support freeze.")
    if g0.sha256_file(g21d.ANALYSIS_PATH) != manifest["source_contracts"][
        "support_event_helper"
    ]["sha256"]:
        raise ValueError("Frozen support-event helper changed.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g22_period_landmarks_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen period support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        events["g18_period"] = events["period"].astype(str)
        events["scope_kind"] = "period_and_coordinate"
        events["scope_value"] = events["period_kind"].astype(str) + "__" + events[
            "coordinate"
        ].astype(str)
        parts.append(events)
        print(
            json.dumps(
                {"phase": "g22_period_outcomes", "processed": number, "total": 20}
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
        "contrasts": run_dir / "g22_period_pair_contrasts.csv",
        "scores": run_dir / "g22_period_period_scores.csv",
        "decisions": run_dir / "g22_period_decisions.csv",
    }
    for name, frame in (("contrasts", contrasts), ("scores", scores), ("decisions", decisions)):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_period_landmarks",
        "branch_completed": "g22a_completed_period_price_landmarks",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "partial_period_values_used": False,
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
