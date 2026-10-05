"""Test causal anchored VWAP centres and dispersion bands as reaction zones."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g24d_causal_anchored_vwap_zones"
DEFAULT_RUN_ID = "g24_anchored_vwap_20260828a"
DEFAULT_SUPPORT_ID = "g24_anchored_vwap_support_20260828a"
RECORD_ROOT = g24z.OUTPUT_ROOT / "anchored_vwap"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/generation24_branches/g24_broad_siblings/anchored_vwap"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g24_anchored_vwap_support_freeze.json"
ANCHORS = ("1d", "1w", "1M")
MODES = ("current_session_through_previous_completed_candle", "previous_completed_session")
MULTIPLIERS = (0.0, 1.0, 2.0)
MINIMUMS = {"1d": 6, "1w": 24, "1M": 72}
CONTROLS = tuple(g24z.LEVEL_CONTROLS)
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g24c.load_branch(BRANCH_ID)
    if tuple(branch["anchor_timeframes"]) != ANCHORS or tuple(branch["anchor_modes"]) != MODES:
        raise ValueError("Generation 24 VWAP anchor registry drifted.")
    if (
        tuple(branch["dispersion_multipliers"]) != MULTIPLIERS
        or tuple(branch["controls"]) != CONTROLS
    ):
        raise ValueError("Generation 24 VWAP band/control registry drifted.")
    return frozen


def session_key(dates: Series, anchor: str) -> Series:
    values = pd.to_datetime(dates, utc=True, errors="raise")
    if anchor == "1d":
        return values.dt.floor("D")
    if anchor == "1w":
        return values.dt.floor("D") - pd.to_timedelta(values.dt.weekday, unit="D")
    if anchor == "1M":
        return values.dt.tz_localize(None).dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    raise ValueError(anchor)


def weighted_session_stats(
    base: DataFrame, anchor: str, mode: str
) -> tuple[Series, Series, Series]:
    key = session_key(base["date"], anchor)
    price = (
        pd.to_numeric(base["high"], errors="coerce")
        + pd.to_numeric(base["low"], errors="coerce")
        + pd.to_numeric(base["close"], errors="coerce")
    ) / 3.0
    volume = pd.to_numeric(base["volume"], errors="coerce").clip(lower=0.0)
    work = DataFrame({"key": key, "v": volume, "pv": price * volume, "p2v": price.pow(2) * volume})
    if mode == MODES[0]:
        cumulative = work.groupby("key", observed=True)[["v", "pv", "p2v"]].cumsum()
        prior = cumulative.groupby(work["key"], observed=True).shift(1)
        sum_v = prior["v"]
        sum_pv = prior["pv"]
        sum_p2v = prior["p2v"]
        count = work.groupby("key", observed=True).cumcount()
        minimum = MINIMUMS[anchor]
        source = key
    else:
        grouped = work.groupby("key", observed=True).agg(
            sum_v=("v", "sum"), sum_pv=("pv", "sum"), sum_p2v=("p2v", "sum"), count=("v", "size")
        )
        prior = grouped.shift(1)
        sum_v = key.map(prior["sum_v"])
        sum_pv = key.map(prior["sum_pv"])
        sum_p2v = key.map(prior["sum_p2v"])
        count = key.map(prior["count"])
        minimum = MINIMUMS[anchor]
        source = key.map(Series(grouped.index, index=grouped.index).shift(1))
    mean = sum_pv / sum_v.replace(0.0, np.nan)
    variance = (sum_p2v / sum_v.replace(0.0, np.nan) - mean.pow(2)).clip(lower=0.0)
    valid = pd.to_numeric(count, errors="coerce").ge(minimum)
    return (
        mean.where(valid),
        np.sqrt(variance).where(valid),
        pd.to_datetime(source.where(valid), utc=True),
    )


def vwap_surfaces(base: DataFrame, control: str, pair: str) -> list[dict[str, Any]]:
    atr = pd.to_numeric(base["base_atr"], errors="coerce")
    surfaces = []
    for anchor in ANCHORS:
        for mode in MODES:
            centre, dispersion, source = weighted_session_stats(base, anchor, mode)
            for multiplier in MULTIPLIERS:
                sides = ("centre",) if multiplier == 0 else ("lower", "upper")
                for side in sides:
                    sign = -1.0 if side == "lower" else 1.0 if side == "upper" else 0.0
                    level = centre + sign * multiplier * dispersion
                    if control == "stale_definition":
                        lag = {"1d": 24, "1w": 168, "1M": 720}[anchor]
                        level, source = level.shift(lag), source.shift(lag)
                    elif control == "price_shift":
                        direction = (
                            -1.0
                            if g0.stable_hash_int(
                                f"g24-vwap-shift|{pair}|{anchor}|{mode}|{multiplier}|{side}"
                            )
                            % 2
                            else 1.0
                        )
                        level = level + direction * atr
                    elif control == "random_recent_analogue":
                        lag = (168, 336, 504, 672)[
                            g0.stable_hash_int(
                                f"g24-vwap-lag|{pair}|{anchor}|{mode}|{multiplier}|{side}"
                            )
                            % 4
                        ]
                        level, source = level.shift(lag), source.shift(lag)
                    surfaces.append(
                        {
                            "level_name": f"{anchor}__{mode}__{side}_{multiplier:g}",
                            "anchor_timeframe": anchor,
                            "anchor_mode": mode,
                            "band_side": side,
                            "band_multiplier": multiplier,
                            "level": level.to_numpy(float),
                            "source_open": source,
                        }
                    )
    return surfaces


def surface_support(
    base: DataFrame,
    pair: str,
    cohort: str,
    surfaces: list[dict[str, Any]],
    control: str,
    event_kind: str,
) -> DataFrame:
    parts = []
    for surface in surfaces:
        events = g21d.support_events(
            base,
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
        for key in ("anchor_timeframe", "anchor_mode", "band_side", "band_multiplier"):
            events[key] = surface[key]
        events["level_family"] = "anchored_vwap"
        parts.append(events)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else DataFrame()


def matched_random(base: DataFrame, pair: str, cohort: str, actual: DataFrame) -> DataFrame:
    parts = []
    keys = ["anchor_timeframe", "anchor_mode", "band_side", "band_multiplier"]
    for identity, cell in actual.groupby(keys, observed=True, sort=False):
        events = g21d.matched_random_time_support(
            base, pair=pair, cohort=cohort, level_name=f"random_vwap_{identity}", actual=cell
        )
        if events.empty:
            continue
        for key, value in zip(keys, identity, strict=True):
            events[key] = value
        events["level_family"] = "anchored_vwap"
        parts.append(events)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else DataFrame()


def pair_support(pair: str, cohort: str, *, overwrite: bool) -> dict[str, Any]:
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
    actual_surfaces = vwap_surfaces(base, "actual", pair)
    actual = surface_support(base, pair, cohort, actual_surfaces, "actual", "contact")
    parts = [
        actual,
        matched_random(base, pair, cohort, actual),
        surface_support(base, pair, cohort, actual_surfaces, "near_miss", "near_miss"),
    ]
    for control in ("random_recent_analogue", "stale_definition", "price_shift"):
        parts.append(
            surface_support(
                base, pair, cohort, vwap_surfaces(base, control, pair), control, "contact"
            )
        )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True, sort=False)
    support.sort_values(
        [
            "control",
            "anchor_timeframe",
            "anchor_mode",
            "band_multiplier",
            "band_side",
            "event_time",
        ],
        inplace=True,
        kind="stable",
    )
    support.drop_duplicates(["control", "level_name", "event_time"], inplace=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation24_vwap_outcomes":
            raise ValueError("Invalid Generation 24 VWAP support.")
        return manifest
    inventory = []
    pairs = g22a.cohort_pairs()
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite=overwrite))
        print(
            json.dumps({"phase": "g24_vwap_support", "processed": number, "total": len(pairs)}),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_vwap_outcomes",
        "branch_id": BRANCH_ID,
        "inventory": inventory,
        "future_outcome_values_read": False,
        "source_contracts": {
            "generation24_freeze": artifact(g24z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation24_common": artifact(g24c.ANALYSIS_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    output = events.copy()
    output["scope_kind"] = "anchored_vwap_band"
    output["scope_value"] = (
        output["anchor_timeframe"].astype(str)
        + "__"
        + output["anchor_mode"].astype(str)
        + "__"
        + output["band_side"].astype(str)
        + "_"
        + output["band_multiplier"].astype(float).astype(str)
    )
    return output


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch = g24c.require_all_frozen_supports()
    run_dir, result_path = (
        RECORD_ROOT / run_id,
        RECORD_ROOT / run_id / "g24_anchored_vwap_result.json",
    )
    if result_path.is_file() and not overwrite:
        return 0
    events = g24c.open_frozen_support_outcomes(
        manifest, phase="g24_vwap_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g24c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g24c.write_scored_tables(
        run_dir, prefix="g24_anchored_vwap", contrasts=contrasts, scores=scores, decisions=decisions
    )
    result = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_anchored_vwap",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {"profit_used": False, "signed_direction_used": False},
        "source_contracts": {
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation24_sibling_supports": batch,
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
