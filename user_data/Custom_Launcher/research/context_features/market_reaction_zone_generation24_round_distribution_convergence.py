"""Test round-grid and rolling-price-distribution convergence as reaction zones."""

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
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_price_distribution as g23e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_round_numbers as g23d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g24c_round_distribution_convergence_attribution"
DEFAULT_RUN_ID = "g24_round_distribution_convergence_20260828a"
DEFAULT_SUPPORT_ID = "g24_round_distribution_convergence_support_20260828a"
RECORD_ROOT = g24z.OUTPUT_ROOT / "round_distribution_convergence"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
        "generation24_branches/g24_broad_siblings/round_distribution_convergence"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g24_round_distribution_convergence_support_freeze.json"
STEP_MULTIPLIERS = (1.0, 0.5, 0.25)
LOOKBACKS = (24, 72, 168, 720)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CONVERGENCE_RADIUS_ATR = 0.25
ZONE_HALF_WIDTH_ATR = 0.25
COMPONENT_CONTROLS = (
    "isolated_round_component",
    "isolated_distribution_component",
    "equal_density_pseudo_convergence",
)
CONTROLS = (*COMPONENT_CONTROLS, *g24z.LEVEL_CONTROLS)


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g24c.load_branch(BRANCH_ID)
    if tuple(branch["round_step_multipliers"]) != STEP_MULTIPLIERS:
        raise ValueError("Generation 24 round-grid registry drifted.")
    if tuple(branch["distribution_lookback_hours"]) != LOOKBACKS:
        raise ValueError("Generation 24 distribution-lookback registry drifted.")
    if tuple(branch["distribution_quantiles"]) != QUANTILES:
        raise ValueError("Generation 24 distribution-quantile registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 24 convergence controls drifted.")
    if float(branch["convergence_radius_atr"]) != CONVERGENCE_RADIUS_ATR:
        raise ValueError("Generation 24 convergence radius drifted.")
    return frozen


def combined_source_open(left: Series, right: Series) -> Series:
    """Return the later of both causal component timestamps for each row."""
    values = pd.concat([pd.to_datetime(left, utc=True), pd.to_datetime(right, utc=True)], axis=1)
    return pd.to_datetime(values.max(axis=1), utc=True)


def combination_surfaces(base: DataFrame, pair: str) -> list[dict[str, Any]]:
    """Build every frozen round x distribution combination without outcomes."""
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    rounds = g23d.grid_surfaces(base, pair, "actual")
    distributions = g23e.actual_surfaces(base)
    output: list[dict[str, Any]] = []
    for round_surface in rounds:
        round_level = np.asarray(round_surface["level"], dtype=float)
        for distribution_surface in distributions:
            distribution_level = np.asarray(distribution_surface["level"], dtype=float)
            valid = (
                np.isfinite(round_level)
                & np.isfinite(distribution_level)
                & np.isfinite(atr)
                & (atr > 0.0)
            )
            convergent = valid & (
                np.abs(round_level - distribution_level) <= CONVERGENCE_RADIUS_ATR * atr
            )
            midpoint = np.where(convergent, (round_level + distribution_level) / 2.0, np.nan)
            output.append(
                {
                    "grid_step_multiplier": float(round_surface["grid_step_multiplier"]),
                    "lookback_hours": int(distribution_surface["lookback_hours"]),
                    "quantile": float(distribution_surface["quantile"]),
                    "round_level": round_level,
                    "distribution_level": distribution_level,
                    "convergent": convergent,
                    "level": midpoint,
                    "source_open": combined_source_open(
                        Series(round_surface["source_open"]),
                        Series(distribution_surface["source_open"]),
                    ),
                }
            )
    return output


def identity(surface: dict[str, Any]) -> str:
    step = str(surface["grid_step_multiplier"]).replace(".", "p")
    quantile = round(100 * float(surface["quantile"]))
    return f"step_{step}__lb{surface['lookback_hours']}__q{quantile:02d}"


def transformed_level(
    surface: dict[str, Any], base: DataFrame, pair: str, mode: str
) -> tuple[np.ndarray, Series]:
    level = np.asarray(surface["level"], dtype=float)
    source = pd.to_datetime(surface["source_open"], utc=True)
    if mode == "stale_definition":
        return g0.shift_array(level, 72), source.shift(72)
    if mode == "random_recent_analogue":
        lag = 24 + 24 * (g0.stable_hash_int(f"g24-convergence-lag|{pair}|{identity(surface)}") % 7)
        return g0.shift_array(level, lag), source.shift(lag)
    if mode == "price_shift":
        atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
        direction = (
            -1.0
            if g0.stable_hash_int(f"g24-convergence-price-shift|{pair}|{identity(surface)}") % 2
            else 1.0
        )
        return level + direction * 2.0 * atr, source
    raise ValueError(mode)


def relabel(events: DataFrame, surface: dict[str, Any], control: str) -> DataFrame:
    if events.empty:
        return events
    output = events.copy()
    output["control"] = control
    output["level_family"] = "round_distribution_convergence"
    output["level_name"] = identity(surface)
    output["grid_step_multiplier"] = float(surface["grid_step_multiplier"])
    output["lookback_hours"] = int(surface["lookback_hours"])
    output["quantile"] = float(surface["quantile"])
    return output


def level_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    surface: dict[str, Any],
    level: np.ndarray,
    source_open: Series,
    control: str,
    event_kind: str,
) -> DataFrame:
    events = g21d.support_events(
        base,
        pair=pair,
        cohort=cohort,
        level_name=identity(surface),
        level=level,
        control=control,
        event_kind=event_kind,
        source_open=source_open,
    )
    return relabel(events, surface, control)


def isolated_component_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    surface: dict[str, Any],
    component: str,
) -> DataFrame:
    control = f"isolated_{component}_component"
    level = np.asarray(surface[f"{component}_level"], dtype=float).copy()
    level[np.asarray(surface["convergent"], dtype=bool)] = np.nan
    return level_support(
        base,
        pair=pair,
        cohort=cohort,
        surface=surface,
        level=level,
        source_open=pd.to_datetime(surface["source_open"], utc=True),
        control=control,
        event_kind="contact",
    )


def pseudo_convergence_support(
    actual: DataFrame,
    isolated_round: DataFrame,
    isolated_distribution: DataFrame,
    *,
    pair: str,
    surface: dict[str, Any],
) -> DataFrame:
    """Sample equal-count nonconvergent component contacts in each period."""
    if actual.empty:
        return DataFrame()
    pool = pd.concat([isolated_round, isolated_distribution], ignore_index=True, sort=False)
    if pool.empty:
        return DataFrame()
    pool = pool.sort_values(["period", "base_index", "control"], kind="stable")
    pool = pool.drop_duplicates(["period", "base_index"], keep="first")
    selected: list[DataFrame] = []
    for period_name, cell in actual.groupby("period", observed=True, sort=False):
        candidates = pool.loc[pool["period"].eq(period_name)].copy()
        if candidates.empty:
            continue
        rng = np.random.default_rng(
            g0.stable_hash_int(f"g24-pseudo-convergence|{pair}|{identity(surface)}|{period_name}")
        )
        order = rng.permutation(candidates.index.to_numpy())
        blocked: set[int] = set()
        keep: list[int] = []
        for row_index in order:
            base_index = int(candidates.at[row_index, "base_index"])
            if any(position in blocked for position in range(base_index - 6, base_index + 7)):
                continue
            keep.append(int(row_index))
            blocked.add(base_index)
            if len(keep) >= len(cell):
                break
        selected.append(candidates.loc[keep])
    if not selected:
        return DataFrame()
    output = pd.concat(selected, ignore_index=True, sort=False)
    output["source_family"] = output["control"].astype(str)
    output["match_tier"] = "same_period_equal_density_nonconvergent_component"
    return relabel(output, surface, "equal_density_pseudo_convergence")


def matched_random(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    surface: dict[str, Any],
    actual: DataFrame,
) -> DataFrame:
    events = g21d.matched_random_time_support(
        base,
        pair=pair,
        cohort=cohort,
        level_name=f"random_{identity(surface)}",
        actual=actual,
    )
    return relabel(events, surface, "matched_random_time")


def pair_support(pair: str, cohort: str, *, overwrite: bool) -> dict[str, Any]:
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control"])
        return {
            "pair": pair,
            "cohort": cohort,
            "rows": len(existing),
            "control_counts": existing["control"].value_counts().to_dict(),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "existing",
        }
    base, _ = g20s.base_and_state(pair, cohort)
    parts: list[DataFrame] = []
    for surface in combination_surfaces(base, pair):
        actual = level_support(
            base,
            pair=pair,
            cohort=cohort,
            surface=surface,
            level=np.asarray(surface["level"], dtype=float),
            source_open=pd.to_datetime(surface["source_open"], utc=True),
            control="actual",
            event_kind="contact",
        )
        isolated_round = isolated_component_support(
            base,
            pair=pair,
            cohort=cohort,
            surface=surface,
            component="round",
        )
        isolated_distribution = isolated_component_support(
            base,
            pair=pair,
            cohort=cohort,
            surface=surface,
            component="distribution",
        )
        parts.extend(
            [
                actual,
                isolated_round,
                isolated_distribution,
                pseudo_convergence_support(
                    actual,
                    isolated_round,
                    isolated_distribution,
                    pair=pair,
                    surface=surface,
                ),
                matched_random(
                    base,
                    pair=pair,
                    cohort=cohort,
                    surface=surface,
                    actual=actual,
                ),
                level_support(
                    base,
                    pair=pair,
                    cohort=cohort,
                    surface=surface,
                    level=np.asarray(surface["level"], dtype=float),
                    source_open=pd.to_datetime(surface["source_open"], utc=True),
                    control="near_miss",
                    event_kind="near_miss",
                ),
            ]
        )
        for control in ("random_recent_analogue", "stale_definition", "price_shift"):
            level, source = transformed_level(surface, base, pair, control)
            parts.append(
                level_support(
                    base,
                    pair=pair,
                    cohort=cohort,
                    surface=surface,
                    level=level,
                    source_open=source,
                    control=control,
                    event_kind="contact",
                )
            )
    available = [part for part in parts if not part.empty]
    if not available:
        raise ValueError(f"No Generation 24 convergence support for {pair}.")
    support = pd.concat(available, ignore_index=True, sort=False)
    support.sort_values(
        [
            "control",
            "grid_step_multiplier",
            "lookback_hours",
            "quantile",
            "event_time",
            "pre_distance_atr",
        ],
        inplace=True,
        kind="stable",
    )
    support.drop_duplicates(
        [
            "control",
            "grid_step_multiplier",
            "lookback_hours",
            "quantile",
            "event_time",
        ],
        keep="first",
        inplace=True,
    )
    support.reset_index(drop=True, inplace=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "control_counts": support["control"].value_counts().to_dict(),
        "path": str(output.resolve()),
        "sha256": g0.sha256_file(output),
        "status": "built",
    }


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation24_convergence_outcomes":
            raise ValueError("Invalid Generation 24 convergence support.")
        return manifest
    pairs = g22a.cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    inventory = []
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite=overwrite))
        print(
            json.dumps({"phase": "g24_convergence_support", "processed": number, "total": 20}),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_convergence_outcomes",
        "branch_id": BRANCH_ID,
        "inventory": inventory,
        "future_outcome_values_read": False,
        "winner_filtering_used": False,
        "combination_count_per_pair": len(STEP_MULTIPLIERS) * len(LOOKBACKS) * len(QUANTILES),
        "controls": list(CONTROLS),
        "source_contracts": {
            "generation24_freeze": artifact(g24z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation24_common": artifact(g24c.ANALYSIS_PATH),
            "round_surface_builder": artifact(g23d.ANALYSIS_PATH),
            "distribution_surface_builder": artifact(g23e.ANALYSIS_PATH),
            "support_event_helper": artifact(g21d.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    output = events.copy()
    output["scope_kind"] = "round_distribution_component_combination"
    output["scope_value"] = (
        "step_"
        + output["grid_step_multiplier"].astype(float).astype(str).str.replace(".", "p")
        + "__lb"
        + output["lookback_hours"].astype(int).astype(str)
        + "__q"
        + (100 * output["quantile"].astype(float)).round().astype(int).astype(str)
    )
    return output


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch = g24c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g24_round_distribution_convergence_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    events = g24c.open_frozen_support_outcomes(
        manifest, phase="g24_convergence_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g24c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g24c.write_scored_tables(
        run_dir,
        prefix="g24_round_distribution_convergence",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_round_distribution_convergence",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "winner_filtering_used": False,
        },
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
