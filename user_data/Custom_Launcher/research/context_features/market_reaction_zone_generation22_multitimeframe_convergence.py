"""Test frozen causal multi-timeframe level convergence and precedence."""

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
DEFAULT_RUN_ID = "g22_multitimeframe_convergence_20260827a"
DEFAULT_SUPPORT_ID = "g22_multitimeframe_support_20260827a"
RECORD_ROOT = g22z.OUTPUT_ROOT / "multitimeframe_convergence"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "multitimeframe_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g22_multitimeframe_support_freeze.json"
HORIZONS = g22z.HORIZONS_HOURS
TIMEFRAMES = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
FAMILIES = (
    "donchian_prior_boundary",
    "rolling_volume_profile_poc",
    "completed_period_high_low",
    "causal_pivot",
)
CONTROLS = g22z.LEVEL_CONTROLS
GEOMETRIES = (
    "isolated_single_timeframe",
    "two_timeframe_cluster",
    "three_plus_timeframe_cluster",
)
CLUSTER_RADIUS_ATR = 0.50
ZONE_HALF_WIDTH_ATR = 0.25
DONCHIAN_BARS = 20
PROFILE_BARS = 48
PROFILE_BINS = 24
PROFILE_CADENCE_BARS = 4


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g22z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22b_multitimeframe_level_convergence_and_precedence"
    )
    if frozen.get("status") != "frozen_before_generation22_outcomes":
        raise ValueError("Generation 22 batch is not frozen.")
    if tuple(branch["source_timeframes"]) != tuple(TIMEFRAMES):
        raise ValueError("Generation 22 multi-timeframe registry drifted.")
    if tuple(branch["causal_level_families"]) != FAMILIES:
        raise ValueError("Generation 22 multi-timeframe family registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 22 multi-timeframe controls drifted.")
    return frozen


def timeframe_keys(dates: Series, hours: int) -> Series:
    values = pd.to_datetime(dates, utc=True, errors="raise")
    if hours == 1:
        return values.dt.floor("1h")
    if hours == 4:
        return values.dt.floor("4h")
    if hours == 24:
        return values.dt.floor("1D")
    if hours == 168:
        day = values.dt.floor("1D")
        return day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    raise ValueError(hours)


def source_bars(base: DataFrame, hours: int) -> tuple[Series, DataFrame]:
    keys = timeframe_keys(base["date"], hours)
    working = base[["open", "high", "low", "close", "volume"]].copy()
    working["source_key"] = keys.to_numpy()
    bars = working.groupby("source_key", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        observed_hours=("close", "count"),
    )
    bars["complete"] = bars["observed_hours"].ge(max(1.0, hours * 0.995))
    bars.loc[~bars["complete"], ["open", "high", "low", "close", "volume"]] = np.nan
    return keys, bars


def rolling_profile_poc(bars: DataFrame) -> Series:
    typical = (
        pd.to_numeric(bars["high"], errors="coerce")
        + pd.to_numeric(bars["low"], errors="coerce")
        + pd.to_numeric(bars["close"], errors="coerce")
    ) / 3.0
    volume = pd.to_numeric(bars["volume"], errors="coerce")
    output = np.full(len(bars), np.nan, dtype=float)
    last = np.nan
    for end in range(PROFILE_BARS - 1, len(bars)):
        if (end - (PROFILE_BARS - 1)) % PROFILE_CADENCE_BARS and np.isfinite(last):
            output[end] = last
            continue
        prices = typical.iloc[end - PROFILE_BARS + 1 : end + 1].to_numpy(dtype=float)
        weights = volume.iloc[end - PROFILE_BARS + 1 : end + 1].to_numpy(dtype=float)
        valid = np.isfinite(prices) & np.isfinite(weights) & (weights > 0.0)
        if valid.sum() < PROFILE_BARS * 0.95:
            continue
        prices = prices[valid]
        weights = weights[valid]
        low = float(prices.min())
        high = float(prices.max())
        if high <= low:
            last = low
        else:
            edges = np.linspace(low, high, PROFILE_BINS + 1)
            histogram, _ = np.histogram(prices, bins=edges, weights=weights)
            maximum = int(np.argmax(histogram))
            last = float((edges[maximum] + edges[maximum + 1]) / 2.0)
        output[end] = last
    return Series(output, index=bars.index)


def bar_level_table(bars: DataFrame) -> DataFrame:
    levels = DataFrame(index=bars.index)
    levels["donchian_upper"] = pd.to_numeric(bars["high"], errors="coerce").rolling(
        DONCHIAN_BARS, min_periods=DONCHIAN_BARS
    ).max()
    levels["donchian_lower"] = pd.to_numeric(bars["low"], errors="coerce").rolling(
        DONCHIAN_BARS, min_periods=DONCHIAN_BARS
    ).min()
    levels["profile_poc"] = rolling_profile_poc(bars)
    levels["previous_bar_high"] = pd.to_numeric(bars["high"], errors="coerce")
    levels["previous_bar_low"] = pd.to_numeric(bars["low"], errors="coerce")
    levels["pivot"] = (
        pd.to_numeric(bars["high"], errors="coerce")
        + pd.to_numeric(bars["low"], errors="coerce")
        + pd.to_numeric(bars["close"], errors="coerce")
    ) / 3.0
    levels["source_open"] = bars.index
    return levels


def mapped_timeframe_levels(
    base: DataFrame, timeframe: str, hours: int
) -> list[dict[str, Any]]:
    keys, bars = source_bars(base, hours)
    # A source bar and all levels calculated from it become available only in
    # the following source period.
    mapped = bar_level_table(bars).shift(1).reindex(pd.DatetimeIndex(keys)).reset_index(
        drop=True
    )
    registry = (
        ("donchian_prior_boundary", "upper", "donchian_upper"),
        ("donchian_prior_boundary", "lower", "donchian_lower"),
        ("rolling_volume_profile_poc", "poc", "profile_poc"),
        ("completed_period_high_low", "high", "previous_bar_high"),
        ("completed_period_high_low", "low", "previous_bar_low"),
        ("causal_pivot", "pivot", "pivot"),
    )
    source = pd.to_datetime(mapped["source_open"], utc=True)
    return [
        {
            "level_name": f"{timeframe}__{family}__{role}",
            "family": family,
            "role": role,
            "source_timeframe": timeframe,
            "timeframe_hours": hours,
            "level": pd.to_numeric(mapped[column], errors="coerce").to_numpy(dtype=float),
            "source_open": source,
        }
        for family, role, column in registry
    ]


def multitimeframe_surfaces(base: DataFrame) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for timeframe, hours in TIMEFRAMES.items():
        surfaces.extend(mapped_timeframe_levels(base, timeframe, hours))
    return surfaces


def shifted_surfaces(
    surfaces: list[dict[str, Any]],
    base: DataFrame,
    *,
    pair: str,
    mode: str,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    output: list[dict[str, Any]] = []
    common_direction = 1.0 if g0.stable_hash_int(f"g22-mtf-shift|{pair}") % 2 == 0 else -1.0
    for surface in surfaces:
        values = np.asarray(surface["level"], dtype=float)
        source = pd.to_datetime(surface["source_open"], utc=True)
        if mode == "stale_definition":
            level = g0.shift_array(values, 72)
            source_open = source.shift(72)
        elif mode == "price_shift":
            level = values + common_direction * 2.0 * atr
            source_open = source
        elif mode == "random_recent_analogue":
            lags = np.asarray(
                [
                    24
                    + g0.stable_hash_int(
                        f"g22-mtf-analogue|{pair}|{surface['level_name']}|{date.isoformat()}"
                    )
                    % 145
                    for date in dates
                ],
                dtype=int,
            )
            indexes = np.arange(len(base), dtype=int) - lags
            valid = indexes >= 0
            level = np.full(len(base), np.nan, dtype=float)
            level[valid] = values[indexes[valid]]
            source_open = Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
            source_values = source.to_numpy()
            source_open.loc[valid] = source_values[indexes[valid]]
        else:
            raise ValueError(mode)
        copy = dict(surface)
        copy["level"] = level
        copy["source_open"] = source_open
        output.append(copy)
    return output


def geometry_name(count: int) -> str:
    if count <= 1:
        return "isolated_single_timeframe"
    if count == 2:
        return "two_timeframe_cluster"
    return "three_plus_timeframe_cluster"


def annotate_clusters(
    events: DataFrame,
    surfaces: list[dict[str, Any]],
    atr: np.ndarray,
) -> DataFrame:
    if events.empty:
        return events
    indexes = events["base_index"].to_numpy(dtype=int)
    anchors = pd.to_numeric(events["level_price"], errors="coerce").to_numpy(dtype=float)
    timeframes: list[str] = []
    families: list[str] = []
    geometries: list[str] = []
    highest: list[str] = []
    order = {hours: name for name, hours in TIMEFRAMES.items()}
    for index, anchor in zip(indexes, anchors, strict=True):
        close_surfaces = [
            surface
            for surface in surfaces
            if np.isfinite(surface["level"][index])
            and abs(float(surface["level"][index]) - anchor)
            <= CLUSTER_RADIUS_ATR * atr[index]
        ]
        tf_hours = sorted({int(surface["timeframe_hours"]) for surface in close_surfaces})
        family_values = sorted({str(surface["family"]) for surface in close_surfaces})
        count = max(1, len(tf_hours))
        timeframes.append("|".join(order[value] for value in tf_hours))
        families.append("|".join(family_values))
        geometries.append(geometry_name(count))
        highest.append(order[max(tf_hours)] if tf_hours else "unavailable")
    output = events.copy()
    output["source_timeframes_present"] = timeframes
    output["independent_families_present"] = families
    output["timeframe_count"] = [len(value.split("|")) if value else 1 for value in timeframes]
    output["geometry_state"] = geometries
    output["highest_timeframe"] = highest
    return output


def relabel(frame: DataFrame, surface: dict[str, Any]) -> DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["level_family"] = "multitimeframe_level_convergence"
    frame["anchor_source_family"] = surface["family"]
    frame["anchor_role"] = surface["role"]
    frame["anchor_source_timeframe"] = surface["source_timeframe"]
    frame["level_name"] = surface["level_name"]
    return frame


def surface_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    surfaces: list[dict[str, Any]],
    control: str,
    event_kind: str,
) -> DataFrame:
    parts: list[DataFrame] = []
    for surface in surfaces:
        parts.append(
            relabel(
                g21d.support_events(
                    base,
                    pair=pair,
                    cohort=cohort,
                    level_name=str(surface["level_name"]),
                    level=np.asarray(surface["level"], dtype=float),
                    control=control,
                    event_kind=event_kind,
                    source_open=pd.to_datetime(surface["source_open"], utc=True),
                ),
                surface,
            )
        )
    events = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    events = annotate_clusters(events, surfaces, atr)
    events.sort_values(
        ["event_time", "geometry_state", "highest_timeframe", "pre_distance_atr", "level_name"],
        inplace=True,
        kind="stable",
    )
    return events.drop_duplicates(
        ["event_time", "geometry_state", "highest_timeframe"], keep="first"
    ).reset_index(drop=True)


def matched_random_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    actual: DataFrame,
) -> DataFrame:
    parts: list[DataFrame] = []
    for (geometry, highest), cell in actual.groupby(
        ["geometry_state", "highest_timeframe"], observed=True, sort=False
    ):
        matched = g21d.matched_random_time_support(
            base,
            pair=pair,
            cohort=cohort,
            level_name=f"random__{geometry}__{highest}",
            actual=cell,
        )
        if matched.empty:
            continue
        matched["level_family"] = "multitimeframe_level_convergence"
        matched["anchor_source_family"] = "artificial"
        matched["anchor_role"] = "pseudo_contact"
        matched["anchor_source_timeframe"] = "artificial"
        matched["geometry_state"] = geometry
        matched["highest_timeframe"] = highest
        matched["source_timeframes_present"] = "artificial"
        matched["independent_families_present"] = "artificial"
        matched["timeframe_count"] = (
            1
            if geometry == "isolated_single_timeframe"
            else 2
            if geometry == "two_timeframe_cluster"
            else 3
        )
        parts.append(matched)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else DataFrame()


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
    actual_surfaces = multitimeframe_surfaces(base)
    actual = surface_support(
        base,
        pair=pair,
        cohort=cohort,
        surfaces=actual_surfaces,
        control="actual",
        event_kind="contact",
    )
    parts = [actual, matched_random_support(base, pair=pair, cohort=cohort, actual=actual)]
    parts.append(
        surface_support(
            base,
            pair=pair,
            cohort=cohort,
            surfaces=shifted_surfaces(
                actual_surfaces, base, pair=pair, mode="random_recent_analogue"
            ),
            control="random_recent_analogue",
            event_kind="contact",
        )
    )
    parts.append(
        surface_support(
            base,
            pair=pair,
            cohort=cohort,
            surfaces=actual_surfaces,
            control="near_miss",
            event_kind="near_miss",
        )
    )
    for control in ("stale_definition", "price_shift"):
        parts.append(
            surface_support(
                base,
                pair=pair,
                cohort=cohort,
                surfaces=shifted_surfaces(actual_surfaces, base, pair=pair, mode=control),
                control=control,
                event_kind="contact",
            )
        )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "geometry_state", "highest_timeframe", "event_time"],
        inplace=True,
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "actual_geometry_counts": actual["geometry_state"].value_counts().to_dict(),
        "actual_highest_timeframe_counts": actual["highest_timeframe"].value_counts().to_dict(),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation22_multitimeframe_outcomes":
            raise ValueError("Invalid Generation 22 multi-timeframe support freeze.")
        return manifest
    pairs = g22a.cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    inventory = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g22_multitimeframe_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_multitimeframe_outcomes",
        "branch_id": "g22b_multitimeframe_level_convergence_and_precedence",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "source_timeframes": list(TIMEFRAMES),
        "families": list(FAMILIES),
        "geometry_states": list(GEOMETRIES),
        "controls": list(CONTROLS),
        "surface_definition": {
            "donchian_source_bars": DONCHIAN_BARS,
            "profile_source_bars": PROFILE_BARS,
            "profile_bins": PROFILE_BINS,
            "profile_cadence_source_bars": PROFILE_CADENCE_BARS,
            "cluster_radius_atr": CLUSTER_RADIUS_ATR,
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
        },
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


def scoped_events(events: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "period", "control", "event_time"]
    geometry = g18d.nearest_anchor(events, [*keys, "geometry_state"])
    geometry["scope_kind"] = "timeframe_geometry"
    geometry["scope_value"] = geometry["geometry_state"].astype(str)
    highest = g18d.nearest_anchor(events, [*keys, "highest_timeframe"])
    highest["scope_kind"] = "highest_timeframe_present"
    highest["scope_value"] = highest["highest_timeframe"].astype(str)
    return pd.concat([geometry, highest], ignore_index=True, sort=False)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    contracts = manifest["source_contracts"]
    for path, key in (
        (ANALYSIS_PATH, "analysis_script"),
        (g21d.ANALYSIS_PATH, "support_event_helper"),
        (g22a.ANALYSIS_PATH, "shared_period_helper"),
    ):
        if g0.sha256_file(path) != contracts[key]["sha256"]:
            raise ValueError(f"Frozen multi-timeframe dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g22_multitimeframe_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen multi-timeframe support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        parts.append(scoped_events(events))
        print(
            json.dumps(
                {"phase": "g22_multitimeframe_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    events = pd.concat(parts, ignore_index=True, sort=False)
    events["g18_period"] = events["period"].astype(str)
    summary = g22a.metric_summary(events)
    contrasts = g18d.paired_contrasts(summary, CONTROLS)
    question_keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g22_multitimeframe_pair_contrasts.csv",
        "scores": run_dir / "g22_multitimeframe_period_scores.csv",
        "decisions": run_dir / "g22_multitimeframe_decisions.csv",
    }
    for name, frame in (("contrasts", contrasts), ("scores", scores), ("decisions", decisions)):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_multitimeframe_convergence",
        "branch_completed": "g22b_multitimeframe_level_convergence_and_precedence",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "higher_timeframe_precedence_assumed": False,
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
