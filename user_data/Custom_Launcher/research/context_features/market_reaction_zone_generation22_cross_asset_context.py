"""Test level reactions conditioned by causal cross-asset market context."""

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
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)


ANALYSIS_PATH = Path(__file__).resolve()
G17_LEVEL_BUILDER_PATH = Path(g17l.__file__).resolve()
DEFAULT_RUN_ID = "g22_cross_asset_context_20260827a"
DEFAULT_SUPPORT_ID = "g22_cross_asset_support_20260827a"
RECORD_ROOT = g22z.OUTPUT_ROOT / "cross_asset_context"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "cross_asset_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g22_cross_asset_support_freeze.json"
CONTEXT_PATH = SUPPORT_ROOT / "g22_causal_cross_asset_context.parquet"
HORIZONS = g22z.HORIZONS_HOURS
CONTEXT_WINDOWS = (4, 24)
CONTEXT_STATES = (
    "quiet_coherent",
    "broad_synchronized_activity",
    "dispersed_idiosyncratic_activity",
    "btc_led_activity",
)
LEVEL_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "donchian_boundaries",
    "rolling_vwap_deviation_bands",
    "weekly_pivot_grid",
)
CONTROL_COMPARISONS = (*g22z.LEVEL_CONTROLS, "same_state_no_level_time")
CALIBRATION_END_EXCLUSIVE = pd.Timestamp("2026-07-14T00:00:00Z")
ZONE_HALF_WIDTH_ATR = 0.25
SELECTED_LEVELS = (
    "vp_lb72_bins48_poc",
    "vp_lb72_bins48_nearest_hvn_q80",
    "vp_lb72_bins48_nearest_lvn_q10",
    "donchian_lb72_upper",
    "donchian_lb72_lower",
    "donchian_lb168_upper",
    "donchian_lb168_lower",
    "rolling_vwap_lb72_vwap",
    "rolling_vwap_lb72_upper_1sd",
    "rolling_vwap_lb72_lower_1sd",
    "rolling_vwap_lb168_vwap",
    "rolling_vwap_lb168_upper_1sd",
    "rolling_vwap_lb168_lower_1sd",
    "weekly_pivot",
    "weekly_r1",
    "weekly_s1",
)


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g22z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22d_cross_asset_context_conditioned_level_reactions"
    )
    if frozen.get("status") != "frozen_before_generation22_outcomes":
        raise ValueError("Generation 22 batch is not frozen.")
    if tuple(branch["context_windows_hours"]) != CONTEXT_WINDOWS:
        raise ValueError("Generation 22 context windows drifted.")
    if tuple(branch["context_states"]) != CONTEXT_STATES:
        raise ValueError("Generation 22 context state registry drifted.")
    if tuple(branch["level_basket"]) != LEVEL_FAMILIES:
        raise ValueError("Generation 22 context level basket drifted.")
    return frozen


def unique_pairs() -> list[str]:
    return list(dict.fromkeys(pair for _, pair in g22a.cohort_pairs()))


def aligned_pair_inputs() -> dict[str, DataFrame]:
    frames: dict[str, DataFrame] = {}
    cohort_by_pair = {pair: cohort for cohort, pair in g22a.cohort_pairs()}
    for pair in unique_pairs():
        base, _ = g20s.base_and_state(pair, cohort_by_pair[pair])
        selected = base[["date", "close", "volume", "base_atr"]].copy()
        selected["date"] = pd.to_datetime(selected["date"], utc=True, errors="raise")
        frames[pair] = selected.set_index("date").sort_index()
    return frames


def causal_cross_asset_context(frames: dict[str, DataFrame]) -> DataFrame:
    dates = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in frames.values()))))
    close = DataFrame(
        {
            pair: pd.to_numeric(frame["close"], errors="coerce").reindex(dates)
            for pair, frame in frames.items()
        },
        index=dates,
    )
    atr = DataFrame(
        {
            pair: pd.to_numeric(frame["base_atr"], errors="coerce").reindex(dates)
            for pair, frame in frames.items()
        },
        index=dates,
    )
    volume = DataFrame(
        {
            pair: pd.to_numeric(frame["volume"], errors="coerce").reindex(dates)
            for pair, frame in frames.items()
        },
        index=dates,
    )
    prior_close = close.shift(1)
    prior_atr = atr.shift(1)
    prior_volume = volume.shift(1)
    relative_volume = prior_volume.div(
        prior_volume.rolling(168, min_periods=72).median().replace(0.0, np.nan)
    )
    hourly_return = np.log(close.where(close > 0.0)).diff().shift(1)
    market_hourly_return = hourly_return.median(axis=1, skipna=True)
    output = DataFrame({"date": dates})
    btc_pair = "BTC/USDT:USDT"
    for window in CONTEXT_WINDOWS:
        signed_move = prior_close.sub(close.shift(window + 1)).div(prior_atr)
        absolute_move = signed_move.abs()
        median_activity = absolute_move.median(axis=1, skipna=True)
        btc_activity = absolute_move[btc_pair]
        dispersion = signed_move.std(axis=1, skipna=True, ddof=0)
        correlations = DataFrame(
            {
                pair: hourly_return[pair].rolling(window, min_periods=window).corr(
                    market_hourly_return
                )
                for pair in hourly_return
            }
        )
        output[f"btc_absolute_return_over_atr_w{window}"] = btc_activity.to_numpy()
        output[f"equal_weight_median_absolute_return_over_atr_w{window}"] = (
            median_activity.to_numpy()
        )
        output[f"equal_weight_relative_volume_w{window}"] = (
            relative_volume.rolling(window, min_periods=window).mean().median(axis=1).to_numpy()
        )
        output[f"cross_coin_return_dispersion_w{window}"] = dispersion.to_numpy()
        output[f"cross_coin_absolute_return_correlation_w{window}"] = (
            correlations.abs().median(axis=1, skipna=True).to_numpy()
        )
        output[f"btc_to_median_activity_ratio_w{window}"] = btc_activity.div(
            median_activity.replace(0.0, np.nan)
        ).to_numpy()
    return output


def add_context_states(context: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    output = context.copy()
    calibration_mask = pd.to_datetime(output["date"], utc=True) < CALIBRATION_END_EXCLUSIVE
    registry: dict[str, Any] = {}
    for window in CONTEXT_WINDOWS:
        columns = {
            "btc": f"btc_absolute_return_over_atr_w{window}",
            "market": f"equal_weight_median_absolute_return_over_atr_w{window}",
            "volume": f"equal_weight_relative_volume_w{window}",
            "dispersion": f"cross_coin_return_dispersion_w{window}",
            "correlation": f"cross_coin_absolute_return_correlation_w{window}",
            "btc_ratio": f"btc_to_median_activity_ratio_w{window}",
        }
        percentiles: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for name, column in columns.items():
            calibration = pd.to_numeric(
                output.loc[calibration_mask, column], errors="coerce"
            ).to_numpy(dtype=float)
            counts[name] = int(np.isfinite(calibration).sum())
            percentiles[name] = g20s.empirical_percentile(output[column], calibration)
        state = np.full(len(output), "moderate_or_unclassified", dtype=object)
        finite = np.column_stack(list(percentiles.values()))
        available = np.isfinite(finite).all(axis=1)
        quiet = (
            (percentiles["market"] <= 1.0 / 3.0)
            & (percentiles["volume"] <= 1.0 / 3.0)
            & (percentiles["dispersion"] <= 1.0 / 3.0)
        )
        synchronized = (
            (percentiles["market"] > 2.0 / 3.0)
            & (percentiles["volume"] > 2.0 / 3.0)
            & (percentiles["correlation"] > 2.0 / 3.0)
            & (percentiles["dispersion"] <= 2.0 / 3.0)
        )
        dispersed = (
            (percentiles["dispersion"] > 2.0 / 3.0)
            & (percentiles["correlation"] <= 1.0 / 3.0)
        )
        btc_led = (
            (percentiles["btc"] > 2.0 / 3.0)
            & (percentiles["btc_ratio"] > 2.0 / 3.0)
            & (percentiles["market"] <= 2.0 / 3.0)
        )
        # Precedence makes the labels mutually exclusive without using outcomes.
        state[quiet & available] = "quiet_coherent"
        state[synchronized & available] = "broad_synchronized_activity"
        state[dispersed & available] = "dispersed_idiosyncratic_activity"
        state[btc_led & available] = "btc_led_activity"
        state[~available] = "unavailable"
        output[f"context_state_w{window}"] = state
        registry[str(window)] = {
            "calibration_end_exclusive_utc": CALIBRATION_END_EXCLUSIVE.isoformat(),
            "calibration_feature_rows": counts,
            "state_counts": Series(state).value_counts().to_dict(),
        }
    return output, registry


def selected_surfaces(frame: DataFrame) -> list[dict[str, Any]]:
    specs = {spec.name: spec for spec in g17l.source_specs()}
    missing = sorted(name for name in SELECTED_LEVELS if name not in specs)
    if missing:
        raise ValueError(f"Missing fixed Generation 22 level specs: {missing}")
    surfaces: list[dict[str, Any]] = []
    source = pd.to_datetime(frame["source_open"], utc=True)
    for name in SELECTED_LEVELS:
        spec = specs[name]
        surfaces.append(
            {
                "level_name": name,
                "family": spec.family,
                "level": pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(
                    dtype=float
                ),
                "source_open": source,
            }
        )
    return surfaces


def transformed_surfaces(
    surfaces: list[dict[str, Any]],
    frame: DataFrame,
    *,
    pair: str,
    mode: str,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    atr = pd.to_numeric(frame["base_atr"], errors="coerce").to_numpy(dtype=float)
    direction = 1.0 if g0.stable_hash_int(f"g22-context-shift|{pair}") % 2 == 0 else -1.0
    output: list[dict[str, Any]] = []
    for surface in surfaces:
        values = np.asarray(surface["level"], dtype=float)
        source = pd.to_datetime(surface["source_open"], utc=True)
        if mode == "stale_definition":
            level = g0.shift_array(values, 72)
            source_open = source.shift(72)
        elif mode == "price_shift":
            level = values + direction * 2.0 * atr
            source_open = source
        elif mode == "random_recent_analogue":
            lags = np.asarray(
                [
                    24
                    + g0.stable_hash_int(
                        f"g22-context-analogue|{pair}|{surface['level_name']}|{date.isoformat()}"
                    )
                    % 145
                    for date in dates
                ],
                dtype=int,
            )
            indexes = np.arange(len(frame), dtype=int) - lags
            valid = indexes >= 0
            level = np.full(len(frame), np.nan, dtype=float)
            level[valid] = values[indexes[valid]]
            source_open = Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            source_values = source.to_numpy()
            source_open.loc[valid] = source_values[indexes[valid]]
        else:
            raise ValueError(mode)
        copy = dict(surface)
        copy["level"] = level
        copy["source_open"] = source_open
        output.append(copy)
    return output


def surface_events(
    frame: DataFrame,
    *,
    pair: str,
    cohort: str,
    surfaces: list[dict[str, Any]],
    control: str,
    event_kind: str,
) -> DataFrame:
    parts: list[DataFrame] = []
    for surface in surfaces:
        events = g21d.support_events(
            frame,
            pair=pair,
            cohort=cohort,
            level_name=str(surface["level_name"]),
            level=np.asarray(surface["level"], dtype=float),
            control=control,
            event_kind=event_kind,
            source_open=pd.to_datetime(surface["source_open"], utc=True),
        )
        if events.empty:
            continue
        events["level_family"] = "cross_asset_context_level_basket"
        events["source_family"] = surface["family"]
        parts.append(events)
    events = pd.concat(parts, ignore_index=True, sort=False)
    events.sort_values(
        ["event_time", "source_family", "pre_distance_atr", "level_name"],
        inplace=True,
        kind="stable",
    )
    return events.drop_duplicates(["event_time", "source_family"], keep="first")


def matched_random_events(
    frame: DataFrame,
    *,
    pair: str,
    cohort: str,
    actual: DataFrame,
) -> DataFrame:
    parts: list[DataFrame] = []
    for family, cell in actual.groupby("source_family", observed=True, sort=False):
        matched = g21d.matched_random_time_support(
            frame,
            pair=pair,
            cohort=cohort,
            level_name=f"random__{family}",
            actual=cell,
        )
        if matched.empty:
            continue
        matched["level_family"] = "cross_asset_context_level_basket"
        matched["source_family"] = family
        parts.append(matched)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else DataFrame()


def attach_context(events: DataFrame, context: DataFrame) -> DataFrame:
    if events.empty:
        return events
    columns = ["date", *(f"context_state_w{window}" for window in CONTEXT_WINDOWS)]
    lookup = context[columns].rename(columns={"date": "event_time"})
    output = events.merge(lookup, on="event_time", how="left", validate="many_to_one")
    frames: list[DataFrame] = []
    for window in CONTEXT_WINDOWS:
        copy = output.copy()
        copy["context_window_hours"] = window
        copy["context_state"] = copy[f"context_state_w{window}"].astype(str)
        frames.append(copy)
    expanded = pd.concat(frames, ignore_index=True, sort=False)
    return expanded.loc[expanded["context_state"].isin(CONTEXT_STATES)].copy()


def actual_contact_mask(
    frame: DataFrame, surfaces: list[dict[str, Any]]
) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(frame["base_atr"], errors="coerce").to_numpy(dtype=float)
    width = ZONE_HALF_WIDTH_ATR * atr
    contact = np.zeros(len(frame), dtype=bool)
    for surface in surfaces:
        level = np.asarray(surface["level"], dtype=float)
        contact |= np.isfinite(level) & (high >= level - width) & (low <= level + width)
    return contact


def same_state_no_level_events(
    frame: DataFrame,
    *,
    pair: str,
    cohort: str,
    actual_scoped: DataFrame,
    context: DataFrame,
    has_level_contact: np.ndarray,
) -> DataFrame:
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    period = g18d.assign_confirmation_period(dates, cohort).astype(str).to_numpy()
    atr = pd.to_numeric(frame["base_atr"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    context_lookup = context.set_index("date")
    rows: list[dict[str, Any]] = []
    for (window, state, period_name), cell in actual_scoped.groupby(
        ["context_window_hours", "context_state", "period"], observed=True, sort=False
    ):
        state_values = context_lookup[f"context_state_w{int(window)}"].reindex(dates).to_numpy()
        eligible = np.flatnonzero(
            (state_values == state)
            & (period == str(period_name))
            & ~has_level_contact
            & np.isfinite(atr)
            & (atr > 0.0)
            & (np.arange(len(frame)) < len(frame) - max(HORIZONS))
        )
        rng = np.random.default_rng(
            g0.stable_hash_int(
                f"g22-same-state-no-level|{pair}|{window}|{state}|{period_name}"
            )
        )
        blocked = np.zeros(len(frame), dtype=bool)
        selected: list[int] = []
        for index in rng.permutation(eligible):
            index = int(index)
            if blocked[index]:
                continue
            selected.append(index)
            blocked[max(0, index - 6) : min(len(frame), index + 7)] = True
            if len(selected) >= len(cell):
                break
        for index in selected:
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "period": str(period_name),
                    "control": "same_state_no_level_time",
                    "event_time": dates.iloc[index],
                    "base_index": index,
                    "level_family": "cross_asset_context_level_basket",
                    "source_family": "no_level",
                    "level_name": "same_state_current_close_anchor",
                    "level_price": close[index],
                    "zone_half_width": ZONE_HALF_WIDTH_ATR * atr[index],
                    "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
                    "base_atr": atr[index],
                    "pre_distance_atr": 0.0,
                    "approach_state": "already_inside_or_unclear",
                    "source_open": pd.NaT,
                    "match_tier": "exact_context_state_and_period",
                    "context_window_hours": int(window),
                    "context_state": str(state),
                }
            )
    return DataFrame.from_records(rows)


def pair_support(
    pair: str,
    cohort: str,
    context: DataFrame,
    overwrite: bool,
) -> dict[str, Any]:
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
    manifest = json.loads(g17l.COHORT_MANIFESTS[cohort].read_text(encoding="utf-8"))
    frame = g17l.level_surface(pair, manifest)
    surfaces = selected_surfaces(frame)
    actual = surface_events(
        frame,
        pair=pair,
        cohort=cohort,
        surfaces=surfaces,
        control="actual",
        event_kind="contact",
    )
    parts = [attach_context(actual, context)]
    parts.append(
        attach_context(
            matched_random_events(frame, pair=pair, cohort=cohort, actual=actual), context
        )
    )
    parts.append(
        attach_context(
            surface_events(
                frame,
                pair=pair,
                cohort=cohort,
                surfaces=transformed_surfaces(
                    surfaces, frame, pair=pair, mode="random_recent_analogue"
                ),
                control="random_recent_analogue",
                event_kind="contact",
            ),
            context,
        )
    )
    parts.append(
        attach_context(
            surface_events(
                frame,
                pair=pair,
                cohort=cohort,
                surfaces=surfaces,
                control="near_miss",
                event_kind="near_miss",
            ),
            context,
        )
    )
    for control in ("stale_definition", "price_shift"):
        parts.append(
            attach_context(
                surface_events(
                    frame,
                    pair=pair,
                    cohort=cohort,
                    surfaces=transformed_surfaces(
                        surfaces, frame, pair=pair, mode=control
                    ),
                    control=control,
                    event_kind="contact",
                ),
                context,
            )
        )
    actual_scoped = parts[0]
    parts.append(
        same_state_no_level_events(
            frame,
            pair=pair,
            cohort=cohort,
            actual_scoped=actual_scoped,
            context=context,
            has_level_contact=actual_contact_mask(frame, surfaces),
        )
    )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    keys = [
        "control",
        "context_window_hours",
        "context_state",
        "period",
        "event_time",
    ]
    support.sort_values([*keys, "pre_distance_atr", "level_name"], inplace=True, kind="stable")
    support = support.drop_duplicates(keys, keep="first").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "actual_state_counts": (
            support.loc[support["control"].eq("actual"), "context_state"]
            .value_counts()
            .to_dict()
        ),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation22_context_outcomes":
            raise ValueError("Invalid Generation 22 context support freeze.")
        return manifest
    context, state_registry = add_context_states(causal_cross_asset_context(aligned_pair_inputs()))
    SUPPORT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(context, CONTEXT_PATH)
    inventory = []
    pairs = g22a.cohort_pairs()
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, context, overwrite))
        print(
            json.dumps(
                {"phase": "g22_context_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_context_outcomes",
        "branch_id": "g22d_cross_asset_context_conditioned_level_reactions",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "context_windows": list(CONTEXT_WINDOWS),
        "context_states": list(CONTEXT_STATES),
        "selected_level_names": list(SELECTED_LEVELS),
        "controls": list(CONTROL_COMPARISONS),
        "context_state_registry": state_registry,
        "context_artifact": artifact(CONTEXT_PATH),
        "inventory": inventory,
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation17_level_builder": artifact(G17_LEVEL_BUILDER_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
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
        (G17_LEVEL_BUILDER_PATH, "generation17_level_builder"),
        (g21d.ANALYSIS_PATH, "support_event_helper"),
    ):
        if g0.sha256_file(path) != contracts[key]["sha256"]:
            raise ValueError(f"Frozen cross-asset dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g22_cross_asset_context_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen context support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        events["g18_period"] = events["period"].astype(str)
        events["scope_kind"] = "context_window_and_state"
        events["scope_value"] = (
            "w"
            + events["context_window_hours"].astype(int).astype(str)
            + "__"
            + events["context_state"].astype(str)
        )
        parts.append(events)
        print(
            json.dumps(
                {"phase": "g22_context_outcomes", "processed": number, "total": 20}
            ),
            flush=True,
        )
    events = pd.concat(parts, ignore_index=True, sort=False)
    summary = g22a.metric_summary(events)
    contrasts = g18d.paired_contrasts(summary, CONTROL_COMPARISONS)
    question_keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROL_COMPARISONS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": run_dir / "g22_context_pair_contrasts.csv",
        "scores": run_dir / "g22_context_period_scores.csv",
        "decisions": run_dir / "g22_context_decisions.csv",
    }
    for name, frame in (("contrasts", contrasts), ("scores", scores), ("decisions", decisions)):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_cross_asset_context",
        "branch_completed": "g22d_cross_asset_context_conditioned_level_reactions",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "same_state_no_level_control_required": True,
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
