"""Repair Generation 23 same-band no-level coverage without filtering winners."""

from __future__ import annotations

# Bound numerical pools before pandas imports.
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
    market_reaction_zone_generation22_cross_asset_context as g22d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_continuous_context as g23b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g24b_cross_asset_context_no_level_coverage_repair"
DEFAULT_RUN_ID = "g24_context_coverage_20260828a"
DEFAULT_SUPPORT_ID = "g24_context_coverage_support_20260828a"
RECORD_ROOT = g24z.OUTPUT_ROOT / "context_coverage"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/generation24_branches/g24_broad_siblings/context_coverage"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g24_context_coverage_support_freeze.json"
CONTROLS = (*g24z.LEVEL_CONTROLS, "same_band_no_level_time")
HORIZONS = tuple(g24z.HORIZONS_HOURS)
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g24c.load_branch(BRANCH_ID)
    if tuple(branch["context_inputs"]) != g23b.CONTEXT_INPUTS:
        raise ValueError("Generation 24 context inputs drifted.")
    if tuple(branch["context_windows_hours"]) != g23b.CONTEXT_WINDOWS:
        raise ValueError("Generation 24 context windows drifted.")
    if tuple(branch["controls"]) != CONTROLS or branch["winner_filtering_allowed"]:
        raise ValueError("Generation 24 context control contract drifted.")
    return frozen


def repaired_no_level_events(
    frame: DataFrame,
    *,
    pair: str,
    cohort: str,
    actual_expanded: DataFrame,
    context: DataFrame,
    has_level_contact: np.ndarray,
) -> DataFrame:
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    period = g18d.assign_confirmation_period(dates, cohort).astype(str).to_numpy()
    atr = pd.to_numeric(frame["base_atr"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(float)
    lookup = context.set_index("date")
    rows: list[dict[str, Any]] = []
    keys = ["context_window_hours", "context_input", "context_band", "period"]
    for identity, cell in actual_expanded.groupby(keys, observed=True, sort=False):
        window, input_name, band, period_name = identity
        key = f"w{int(window)}__{input_name}"
        bands = lookup[f"band__{key}"].reindex(dates).astype(str).to_numpy()
        values = pd.to_numeric(
            lookup[g23b.context_column(str(input_name), int(window))].reindex(dates),
            errors="coerce",
        ).to_numpy(float)
        percentiles = pd.to_numeric(
            lookup[f"percentile__{key}"].reindex(dates), errors="coerce"
        ).to_numpy(float)
        indexes = np.arange(len(frame), dtype=int)
        eligible = np.flatnonzero(
            (bands == str(band))
            & (period == str(period_name))
            & ~has_level_contact
            & np.isfinite(atr)
            & (atr > 0)
            & (indexes < len(frame) - max(HORIZONS))
        )
        rng = np.random.default_rng(g0.stable_hash_int(f"g24-no-level|{pair}|{identity}"))
        blocked = np.zeros(len(frame), dtype=bool)
        selected: list[int] = []
        for candidate in rng.permutation(eligible):
            index = int(candidate)
            if blocked[index]:
                continue
            selected.append(index)
            blocked[max(0, index - 1) : min(len(frame), index + 2)] = True
            if len(selected) >= len(cell):
                break
        for index in selected:
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "period": str(period_name),
                    "control": "same_band_no_level_time",
                    "event_time": dates.iloc[index],
                    "base_index": index,
                    "level_family": "cross_asset_context_level_basket",
                    "source_family": "no_level",
                    "level_name": "same_band_current_close_anchor",
                    "level_price": close[index],
                    "zone_half_width": ZONE_HALF_WIDTH_ATR * atr[index],
                    "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
                    "base_atr": atr[index],
                    "pre_distance_atr": 0.0,
                    "approach_state": "already_inside_or_unclear",
                    "source_open": pd.NaT,
                    "match_tier": "exact_single_input_band_and_period_spacing_2h",
                    "context_window_hours": int(window),
                    "context_input": str(input_name),
                    "context_band": str(band),
                    "context_value": values[index],
                    "context_percentile": percentiles[index],
                }
            )
    return DataFrame.from_records(rows)


def pair_support(item: dict[str, Any], context: DataFrame, *, overwrite: bool) -> dict[str, Any]:
    pair, cohort = str(item["pair"]), str(item["cohort"])
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
    parent_path = Path(item["path"])
    if g0.sha256_file(parent_path) != item["sha256"]:
        raise ValueError(f"Generation 23 context support changed: {parent_path}")
    parent = pd.read_parquet(parent_path)
    retained = parent.loc[parent["control"].ne("same_band_no_level_time")].copy()
    source_manifest = json.loads(g23b.g17l.COHORT_MANIFESTS[cohort].read_text(encoding="utf-8"))
    frame = g23b.g17l.level_surface(pair, source_manifest)
    surfaces = g22d.selected_surfaces(frame)
    no_level = repaired_no_level_events(
        frame,
        pair=pair,
        cohort=cohort,
        actual_expanded=retained.loc[retained["control"].eq("actual")],
        context=context,
        has_level_contact=g22d.actual_contact_mask(frame, surfaces),
    )
    support = pd.concat([retained, no_level], ignore_index=True, sort=False)
    support.sort_values(
        [
            "control",
            "context_window_hours",
            "context_input",
            "context_band",
            "period",
            "event_time",
        ],
        inplace=True,
        kind="stable",
    )
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
    parent = json.loads(g23b.SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    if parent.get("status") != "frozen_before_generation23_continuous_context_outcomes":
        raise ValueError("Generation 23 context support is invalid.")
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation24_context_outcomes":
            raise ValueError("Invalid Generation 24 context support.")
        return manifest
    parent_context = json.loads(g22d.SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    context = pd.read_parquet(parent_context["context_artifact"]["path"])
    context["date"] = pd.to_datetime(context["date"], utc=True, errors="raise")
    context = g23b.context_with_bands(context, g23b.context_registry(context))
    inventory = []
    for number, item in enumerate(parent["inventory"], start=1):
        inventory.append(pair_support(item, context, overwrite=overwrite))
        print(
            json.dumps(
                {
                    "phase": "g24_context_support",
                    "processed": number,
                    "total": len(parent["inventory"]),
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_context_outcomes",
        "branch_id": BRANCH_ID,
        "inventory": inventory,
        "future_outcome_values_read": False,
        "winner_filtering_used": False,
        "source_contracts": {
            "generation24_freeze": artifact(g24z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation24_common": artifact(g24c.ANALYSIS_PATH),
            "generation23_context_support": artifact(g23b.SUPPORT_MANIFEST),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    output = events.copy()
    output["scope_kind"] = "single_context_input_band"
    output["scope_value"] = (
        "w"
        + output["context_window_hours"].astype(int).astype(str)
        + "__"
        + output["context_input"].astype(str)
        + "__"
        + output["context_band"].astype(str)
    )
    return output


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch_supports = g24c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g24_context_coverage_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    events = g24c.open_frozen_support_outcomes(
        manifest, phase="g24_context_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g24c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g24c.write_scored_tables(
        run_dir,
        prefix="g24_context_coverage",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_context_coverage",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "winner_filtering_used": False,
        },
        "source_contracts": {
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation24_sibling_supports": batch_supports,
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
