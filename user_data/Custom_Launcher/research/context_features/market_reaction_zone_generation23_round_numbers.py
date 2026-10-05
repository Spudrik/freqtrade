"""Test frozen causal round-number grids across normal and meme cohorts."""

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
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_common as g23c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g23_round_numbers_20260828a"
DEFAULT_SUPPORT_ID = "g23_round_number_support_20260828a"
RECORD_ROOT = g23z.OUTPUT_ROOT / "round_numbers"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "round_number_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g23_round_number_support_freeze.json"
STEP_MULTIPLIERS = (1.0, 0.5, 0.25)
CONTROLS = (
    "matched_random_time",
    "random_mantissa_grid",
    "half_step_phase_shifted_grid",
    "near_miss",
    "stale_72h",
)
HORIZONS = g23z.HORIZONS_HOURS
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g23c.load_branch("g23d_round_number_and_price_grid_zones")
    if tuple(branch["step_multipliers"]) != STEP_MULTIPLIERS:
        raise ValueError("Generation 23 round-number scale registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 23 round-number controls drifted.")
    if float(branch["zone_half_width_atr"]) != ZONE_HALF_WIDTH_ATR:
        raise ValueError("Generation 23 round-number zone width drifted.")
    return frozen


def grid_phase(pair: str, multiplier: float) -> float:
    return 0.137 + (
        g0.stable_hash_int(f"g23-round-phase|{pair}|{multiplier}") % 727
    ) / 1000.0


def grid_level(prior_close: np.ndarray, step: np.ndarray, phase: float) -> np.ndarray:
    scaled = prior_close / step
    return (np.rint(scaled - phase) + phase) * step


def grid_surfaces(
    base: DataFrame, pair: str, mode: str
) -> list[dict[str, Any]]:
    prior = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(prior) & (prior > 0.0)
    base_step = np.full(len(base), np.nan, dtype=float)
    base_step[valid] = 10.0 ** (np.floor(np.log10(prior[valid])) - 1.0)
    source = pd.to_datetime(base["date"], utc=True, errors="raise").shift(1)
    surfaces: list[dict[str, Any]] = []
    for multiplier in STEP_MULTIPLIERS:
        step = base_step * multiplier
        if mode == "actual":
            level = grid_level(prior, step, 0.0)
            source_open = source
        elif mode == "random_mantissa_grid":
            level = grid_level(prior, step, grid_phase(pair, multiplier))
            source_open = source
        elif mode == "half_step_phase_shifted_grid":
            level = grid_level(prior, step, 0.5)
            source_open = source
        elif mode == "stale_72h":
            actual = grid_level(prior, step, 0.0)
            level = g0.shift_array(actual, 72)
            source_open = source.shift(72)
        else:
            raise ValueError(mode)
        level[~valid] = np.nan
        scale_name = str(multiplier).replace(".", "p")
        surfaces.append(
            {
                "level_name": f"round_grid_step_{scale_name}",
                "grid_step_multiplier": multiplier,
                "level": level,
                "source_open": source_open,
            }
        )
    return surfaces


def relabel(events: DataFrame, surface: dict[str, Any]) -> DataFrame:
    if events.empty:
        return events
    output = events.copy()
    output["level_family"] = "price_scale_round_number_grid"
    output["grid_step_multiplier"] = float(surface["grid_step_multiplier"])
    output["level_name"] = str(surface["level_name"])
    return output


def surface_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    surfaces: list[dict[str, Any]],
    control: str,
    event_kind: str,
) -> DataFrame:
    parts = [
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
        for surface in surfaces
    ]
    available = [part for part in parts if not part.empty]
    return pd.concat(available, ignore_index=True, sort=False) if available else DataFrame()


def matched_random_support(
    base: DataFrame, *, pair: str, cohort: str, actual: DataFrame
) -> DataFrame:
    parts: list[DataFrame] = []
    for multiplier, cell in actual.groupby(
        "grid_step_multiplier", observed=True, sort=False
    ):
        matched = g21d.matched_random_time_support(
            base,
            pair=pair,
            cohort=cohort,
            level_name=f"random_time_grid_{multiplier}",
            actual=cell,
        )
        if matched.empty:
            continue
        matched["level_family"] = "price_scale_round_number_grid"
        matched["grid_step_multiplier"] = float(multiplier)
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
    actual_surfaces = grid_surfaces(base, pair, "actual")
    actual = surface_support(
        base,
        pair=pair,
        cohort=cohort,
        surfaces=actual_surfaces,
        control="actual",
        event_kind="contact",
    )
    parts = [actual, matched_random_support(base, pair=pair, cohort=cohort, actual=actual)]
    for control in (
        "random_mantissa_grid",
        "half_step_phase_shifted_grid",
        "stale_72h",
    ):
        parts.append(
            surface_support(
                base,
                pair=pair,
                cohort=cohort,
                surfaces=grid_surfaces(base, pair, control),
                control=control,
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
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "grid_step_multiplier", "event_time", "pre_distance_atr"],
        inplace=True,
        kind="stable",
    )
    support = support.drop_duplicates(
        ["control", "grid_step_multiplier", "event_time"], keep="first"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "actual_counts": actual["grid_step_multiplier"].value_counts().to_dict(),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation23_round_number_outcomes":
            raise ValueError("Invalid Generation 23 round-number support freeze.")
        return manifest
    inventory = []
    pairs = g22a.cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g23_round_number_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_round_number_outcomes",
        "branch_id": "g23d_round_number_and_price_grid_zones",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "step_multipliers": list(STEP_MULTIPLIERS),
        "controls": list(CONTROLS),
        "grid_definition": {
            "base_step": "10 ** (floor(log10(previous_close)) - 1)",
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
            "random_phase_is_fixed_per_pair_and_scale": True,
        },
        "inventory": inventory,
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation23_common": artifact(g23c.ANALYSIS_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    output = events.copy()
    output["scope_kind"] = "round_grid_scale"
    output["scope_value"] = (
        "step_"
        + output["grid_step_multiplier"].astype(float).astype(str).str.replace(".", "p")
    )
    return output


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch_supports = g23c.require_all_frozen_supports()
    for path, key in (
        (ANALYSIS_PATH, "analysis_script"),
        (g23c.ANALYSIS_PATH, "generation23_common"),
        (g21d.ANALYSIS_PATH, "support_event_helper"),
    ):
        if g0.sha256_file(path) != manifest["source_contracts"][key]["sha256"]:
            raise ValueError(f"Frozen round-number dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g23_round_numbers_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    events = g23c.open_frozen_support_outcomes(
        manifest, phase="g23_round_number_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g23c.score_control_ladders(
        events, controls=CONTROLS
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g23c.write_scored_tables(
        run_dir,
        prefix="g23_round_numbers",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_round_numbers",
        "branch_completed": "g23d_round_number_and_price_grid_zones",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "grid_scales_tuned_after_outcomes": False,
        },
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation23_sibling_supports": batch_supports,
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
