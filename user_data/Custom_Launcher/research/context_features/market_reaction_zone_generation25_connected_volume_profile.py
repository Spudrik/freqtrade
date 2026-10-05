"""Compare causal connected Volume Profile zones with point-node controls."""

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
from pandas import DataFrame


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
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_common as g25c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_convergence_representation as g25r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)
from user_data.Indicators import complex_volume_profile as canonical_vp


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g25c_connected_volume_profile_zones"
DEFAULT_SUPPORT_ID = "g25_connected_volume_profile_support_20260828a"
DEFAULT_RUN_ID = "g25_connected_volume_profile_20260828a"
RECORD_ROOT = g25z.OUTPUT_ROOT / "connected_volume_profile"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
        "generation25_branches/g25_broad_siblings/connected_volume_profile"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g25_connected_volume_profile_support_freeze.json"
WINDOW = 96
BINS = 48
SMOOTH_BINS = 3
HVN_THRESHOLD = 0.70
LVN_THRESHOLD = 0.35
STALE_HOURS = 72
PRICE_SHIFT_ATR = 2.0
ZONE_KINDS = ("connected_hvn_area", "connected_lvn_corridor")
CONTROLS = (
    "point_node_equal_density",
    "matched_random_time",
    "near_miss",
    "stale_definition",
    "price_shift",
)
MIN_REPRESENTATION_ROWS = 10


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def _component_bounds(
    mask: np.ndarray,
    bin_low: np.ndarray,
    bin_high: np.ndarray,
    reference: float,
) -> tuple[float, float, int]:
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return np.nan, np.nan, 0
    breaks = np.flatnonzero(np.diff(indexes) > 1) + 1
    components = np.split(indexes, breaks)
    best: tuple[float, float, int] | None = None
    best_distance = np.inf
    for component in components:
        lower = float(bin_low[int(component[0])])
        upper = float(bin_high[int(component[-1])])
        distance = max(lower - reference, reference - upper, 0.0)
        if distance < best_distance:
            best_distance = distance
            best = (lower, upper, len(component))
    if best is None:
        return np.nan, np.nan, 0
    return best


def _nearest_point(
    mask: np.ndarray,
    centers: np.ndarray,
    reference: float,
    bin_width: float,
) -> tuple[float, float, int]:
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return np.nan, np.nan, 0
    selected = int(indexes[np.argmin(np.abs(centers[indexes] - reference))])
    centre = float(centers[selected])
    return centre - 0.5 * bin_width, centre + 0.5 * bin_width, 1


def _profile_block(
    base: DataFrame,
    target_indexes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build past-only profile geometry for selected base rows."""
    result = {
        name: np.full(len(base), np.nan, dtype=float)
        for name in (
            "hvn_low",
            "hvn_high",
            "hvn_bins",
            "hvn_point_low",
            "hvn_point_high",
            "lvn_low",
            "lvn_high",
            "lvn_bins",
            "lvn_point_low",
            "lvn_point_high",
        )
    }
    if not len(target_indexes):
        return result
    high_all = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low_all = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    volume_all = pd.to_numeric(base["volume"], errors="coerce").to_numpy(dtype=float)
    reference_all = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    offsets = np.arange(-WINDOW, 0, dtype=np.int64)
    bin_numbers = np.arange(BINS, dtype=float)
    eps = np.finfo(float).eps
    for chunk_start in range(0, len(target_indexes), 256):
        indexes = target_indexes[chunk_start : chunk_start + 256]
        windows = indexes[:, None] + offsets[None, :]
        highs = high_all[windows]
        lows = low_all[windows]
        volumes = volume_all[windows]
        profile_low = np.nanmin(lows, axis=1)
        profile_high = np.nanmax(highs, axis=1)
        spans = np.maximum(profile_high - profile_low, eps)
        widths = spans / float(BINS)
        bin_low = profile_low[:, None] + bin_numbers[None, :] * widths[:, None]
        bin_high = bin_low + widths[:, None]
        centers = (bin_low + bin_high) / 2.0
        candle_range = highs - lows
        overlap = (
            np.minimum(highs[:, :, None], bin_high[:, None, :])
            - np.maximum(lows[:, :, None], bin_low[:, None, :])
        )
        weights = np.divide(
            np.clip(overlap, 0.0, None),
            candle_range[:, :, None],
            out=np.zeros_like(overlap),
            where=np.isfinite(candle_range[:, :, None]) & (candle_range[:, :, None] > eps),
        )
        profile = np.sum(weights * volumes[:, :, None], axis=1)
        smoothed = canonical_vp._smooth_profile(profile, SMOOTH_BINS)
        total = profile.sum(axis=1)
        poc = profile.max(axis=1)
        mean = total / float(BINS)
        local_hvn = canonical_vp._hvn_mask(smoothed, poc, HVN_THRESHOLD)
        local_lvn = canonical_vp._lvn_mask(smoothed, total, LVN_THRESHOLD)
        connected_hvn = smoothed >= (poc[:, None] * HVN_THRESHOLD)
        connected_lvn = smoothed <= (mean[:, None] * LVN_THRESHOLD)
        occupied = profile > 0.0
        occupied_indexes = np.where(occupied, np.arange(BINS)[None, :], -1)
        last_occupied = occupied_indexes.max(axis=1)
        first_occupied = np.where(occupied, np.arange(BINS)[None, :], BINS).min(axis=1)
        interior = (
            np.arange(BINS)[None, :] >= first_occupied[:, None]
        ) & (np.arange(BINS)[None, :] <= last_occupied[:, None])
        connected_lvn &= interior
        local_lvn &= interior
        for row_number, base_index in enumerate(indexes):
            reference = float(reference_all[base_index])
            hvn = _component_bounds(
                connected_hvn[row_number],
                bin_low[row_number],
                bin_high[row_number],
                reference,
            )
            hvn_point = _nearest_point(
                local_hvn[row_number], centers[row_number], reference, widths[row_number]
            )
            lvn = _component_bounds(
                connected_lvn[row_number],
                bin_low[row_number],
                bin_high[row_number],
                reference,
            )
            lvn_point = _nearest_point(
                local_lvn[row_number], centers[row_number], reference, widths[row_number]
            )
            for prefix, values in (
                ("hvn", hvn),
                ("hvn_point", hvn_point),
                ("lvn", lvn),
                ("lvn_point", lvn_point),
            ):
                result[f"{prefix}_low"][base_index] = values[0]
                result[f"{prefix}_high"][base_index] = values[1]
                if prefix in ("hvn", "lvn"):
                    result[f"{prefix}_bins"][base_index] = values[2]
    return result


def causal_profile_geometry(base: DataFrame) -> dict[str, np.ndarray]:
    period = base["period"].astype(str)
    indexes = np.flatnonzero(
        period.ne("outside_g18_confirmation").to_numpy(dtype=bool)
        & (np.arange(len(base)) >= WINDOW)
        & (np.arange(len(base)) < len(base) - max(g24z.HORIZONS_HOURS))
    )
    return _profile_block(base, indexes)


def zone_events(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    zone_kind: str,
    lower: np.ndarray,
    upper: np.ndarray,
    bin_count: np.ndarray,
    control: str,
    event_kind: str = "contact",
) -> DataFrame:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    centre = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (upper > lower)
        & np.isfinite(atr)
        & (atr > 0.0)
    )
    contact = valid & (high >= lower) & (low <= upper)
    if event_kind == "contact":
        condition = contact
    elif event_kind == "near_miss":
        margin = 0.25 * atr
        condition = valid & (high >= lower - margin) & (low <= upper + margin) & ~contact
    else:
        raise ValueError(event_kind)
    episode_width = np.maximum(half_width, 0.05 * atr)
    starts = g0.episode_start_mask(condition, centre, episode_width, cooldown=6)
    starts[max(len(starts) - max(g24z.HORIZONS_HOURS), 0) :] = False
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    period = g18d.assign_confirmation_period(dates, cohort)
    starts &= period.ne("outside_g18_confirmation").to_numpy(dtype=bool)
    indexes = np.flatnonzero(starts)
    distance = np.maximum.reduce(
        [
            lower[indexes] - pre_close[indexes],
            pre_close[indexes] - upper[indexes],
            np.zeros(len(indexes)),
        ]
    )
    approach = np.select(
        [pre_close[indexes] < lower[indexes], pre_close[indexes] > upper[indexes]],
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
            "level_family": "connected_volume_profile",
            "level_name": zone_kind,
            "level_price": centre[indexes],
            "zone_low": lower[indexes],
            "zone_high": upper[indexes],
            "zone_half_width": half_width[indexes],
            "zone_bin_count": np.asarray(bin_count, dtype=float)[indexes],
            "base_atr": atr[indexes],
            "pre_distance_atr": distance / atr[indexes],
            "approach_state": approach,
            "source_open": dates.shift(1).iloc[indexes].to_numpy(),
            "match_tier": "not_applicable",
            "scope_kind": "volume_profile_zone_representation",
            "scope_value": zone_kind,
        }
    )


def _transformed_geometry(
    base: DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    pair: str,
    zone_kind: str,
    control: str,
) -> tuple[np.ndarray, np.ndarray]:
    if control == "stale_definition":
        return g0.shift_array(lower, STALE_HOURS), g0.shift_array(upper, STALE_HOURS)
    if control == "price_shift":
        atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
        direction = (
            -1.0
            if g0.stable_hash_int(f"g25-vp-shift|{pair}|{zone_kind}") % 2
            else 1.0
        )
        offset = direction * PRICE_SHIFT_ATR * atr
        return lower + offset, upper + offset
    raise ValueError(control)


def _random_time_events(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    zone_kind: str,
    actual: DataFrame,
) -> DataFrame:
    random = g21d.matched_random_time_support(
        base,
        pair=pair,
        cohort=cohort,
        level_name=f"random_{zone_kind}",
        actual=actual,
    )
    if random.empty:
        return random
    random["control"] = "matched_random_time"
    random["level_family"] = "connected_volume_profile"
    random["level_name"] = zone_kind
    random["zone_low"] = random["level_price"] - random["zone_half_width"]
    random["zone_high"] = random["level_price"] + random["zone_half_width"]
    random["zone_bin_count"] = 1.0
    random["scope_kind"] = "volume_profile_zone_representation"
    random["scope_value"] = zone_kind
    return random


def _equal_density_ladder(parts: dict[str, DataFrame]) -> tuple[DataFrame, list[dict[str, Any]]]:
    output: list[DataFrame] = []
    audits: list[dict[str, Any]] = []
    for period in sorted(set(parts["actual"]["period"].astype(str))):
        pools = {
            control: frame.loc[frame["period"].astype(str).eq(period)].copy()
            for control, frame in parts.items()
        }
        counts = {control: len(frame) for control, frame in pools.items()}
        common = min(counts.values()) if counts else 0
        actual_pool = pools["actual"].sort_values("event_time", kind="stable")
        actual = actual_pool.iloc[g25r._spread_positions(len(actual_pool), common)].copy()
        output.append(actual)
        matched_counts = {"actual": len(actual)}
        for control in CONTROLS:
            matched = g25r._geometry_match(actual, pools[control], common)
            matched["control"] = control
            output.append(matched)
            matched_counts[control] = len(matched)
        audits.append(
            {
                "period": period,
                "source_counts": counts,
                "common_count": common,
                "matched_counts": matched_counts,
                "representation_gate_pass": bool(
                    common >= MIN_REPRESENTATION_ROWS
                    and set(matched_counts.values()) == {common}
                ),
            }
        )
    available = [frame for frame in output if not frame.empty]
    combined = pd.concat(available, ignore_index=True, sort=False) if available else DataFrame()
    return combined, audits


def pair_support(pair: str, cohort: str, *, overwrite: bool) -> dict[str, Any]:
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    audit_path = output.with_suffix(".support.json")
    if output.is_file() and audit_path.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control", "scope_value"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "representation_cells_passing": audit["representation_cells_passing"],
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "audit": artifact(audit_path),
            "status": "existing",
        }
    base, _ = g20s.base_and_state(pair, cohort)
    geometry = causal_profile_geometry(base)
    supports: list[DataFrame] = []
    all_audits: list[dict[str, Any]] = []
    for zone_kind, prefix in (("connected_hvn_area", "hvn"), ("connected_lvn_corridor", "lvn")):
        lower = geometry[f"{prefix}_low"]
        upper = geometry[f"{prefix}_high"]
        bins = geometry[f"{prefix}_bins"]
        actual = zone_events(
            base,
            pair=pair,
            cohort=cohort,
            zone_kind=zone_kind,
            lower=lower,
            upper=upper,
            bin_count=bins,
            control="actual",
        )
        point = zone_events(
            base,
            pair=pair,
            cohort=cohort,
            zone_kind=zone_kind,
            lower=geometry[f"{prefix}_point_low"],
            upper=geometry[f"{prefix}_point_high"],
            bin_count=np.ones(len(base), dtype=float),
            control="point_node_equal_density",
        )
        near = zone_events(
            base,
            pair=pair,
            cohort=cohort,
            zone_kind=zone_kind,
            lower=lower,
            upper=upper,
            bin_count=bins,
            control="near_miss",
            event_kind="near_miss",
        )
        controls: dict[str, DataFrame] = {
            "actual": actual,
            "point_node_equal_density": point,
            "matched_random_time": _random_time_events(
                base, pair=pair, cohort=cohort, zone_kind=zone_kind, actual=actual
            ),
            "near_miss": near,
        }
        for control in ("stale_definition", "price_shift"):
            transformed_low, transformed_high = _transformed_geometry(
                base,
                lower,
                upper,
                pair=pair,
                zone_kind=zone_kind,
                control=control,
            )
            controls[control] = zone_events(
                base,
                pair=pair,
                cohort=cohort,
                zone_kind=zone_kind,
                lower=transformed_low,
                upper=transformed_high,
                bin_count=bins,
                control=control,
            )
        support, audits = _equal_density_ladder(controls)
        supports.append(support)
        for row in audits:
            row["zone_kind"] = zone_kind
        all_audits.extend(audits)
    available = [frame for frame in supports if not frame.empty]
    if not available:
        raise ValueError(f"No Generation 25 connected VP support for {pair}.")
    support = pd.concat(available, ignore_index=True, sort=False)
    support.sort_values(["scope_value", "period", "control", "event_time"], inplace=True)
    support.reset_index(drop=True, inplace=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    audit = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "pair": pair,
        "cohort": cohort,
        "period_cells": all_audits,
        "representation_cells_passing": sum(
            bool(row["representation_gate_pass"]) for row in all_audits
        ),
        "connected_multi_bin_fraction": float(
            pd.to_numeric(
                support.loc[support["control"].eq("actual"), "zone_bin_count"],
                errors="coerce",
            )
            .gt(1.0)
            .mean()
        ),
        "future_outcome_values_read": False,
    }
    g0.atomic_write_json(audit, audit_path)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "representation_cells_passing": audit["representation_cells_passing"],
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "audit": artifact(audit_path),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    g25c.load_branch(BRANCH_ID)
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation25_connected_vp_outcomes":
            raise ValueError("Invalid Generation 25 connected VP support.")
        return manifest
    pairs = g22a.cohort_pairs()
    inventory: list[dict[str, Any]] = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite=overwrite))
        print(
            json.dumps({"phase": "g25_connected_vp_support", "processed": number, "total": 20}),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation25_connected_vp_outcomes",
        "branch_id": BRANCH_ID,
        "profile_contract": {
            "window_completed_candles": WINDOW,
            "bins": BINS,
            "smooth_bins": SMOOTH_BINS,
            "hvn_threshold_of_poc": HVN_THRESHOLD,
            "lvn_threshold_of_mean": LVN_THRESHOLD,
            "current_candle_excluded": True,
            "zone_kinds": list(ZONE_KINDS),
        },
        "controls": list(CONTROLS),
        "inventory": inventory,
        "future_outcome_values_read": False,
        "canonical_indicators_edited": False,
        "source_contracts": {
            "generation25_freeze": artifact(g25z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "canonical_volume_profile_read_only": artifact(Path(canonical_vp.__file__)),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    sibling_supports = g25c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g25_connected_volume_profile_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    events = g25c.open_direct_support_outcomes(
        manifest,
        phase="g25_connected_vp_outcomes",
    )
    contrasts, scores, decisions = g25c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g25c.write_scored_tables(
        run_dir,
        prefix="g25_connected_volume_profile",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation25_connected_volume_profile",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "canonical_indicators_edited": False,
            "same_holdout_exploratory": True,
        },
        "source_contracts": {
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation25_sibling_supports": sibling_supports,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
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
