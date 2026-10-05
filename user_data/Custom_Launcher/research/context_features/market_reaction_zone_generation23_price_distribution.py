"""Test frozen causal completed-price distribution boundaries."""

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
    market_reaction_zone_generation23_common as g23c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g23_price_distribution_20260828a"
DEFAULT_SUPPORT_ID = "g23_price_distribution_support_20260828a"
RECORD_ROOT = g23z.OUTPUT_ROOT / "price_distribution"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "price_distribution_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g23_price_distribution_support_freeze.json"
LOOKBACKS = (24, 72, 168, 720)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CONTROLS = g23z.LEVEL_CONTROLS
HORIZONS = g23z.HORIZONS_HOURS
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g23c.load_branch("g23e_causal_price_distribution_boundaries")
    if tuple(branch["lookback_hours"]) != LOOKBACKS:
        raise ValueError("Generation 23 distribution lookback registry drifted.")
    if tuple(branch["quantiles"]) != QUANTILES:
        raise ValueError("Generation 23 distribution quantile registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 23 distribution controls drifted.")
    return frozen


def actual_surfaces(base: DataFrame) -> list[dict[str, Any]]:
    typical = (
        pd.to_numeric(base["high"], errors="coerce")
        + pd.to_numeric(base["low"], errors="coerce")
        + pd.to_numeric(base["close"], errors="coerce")
    ) / 3.0
    completed = typical.shift(1)
    source = pd.to_datetime(base["date"], utc=True, errors="raise").shift(1)
    surfaces: list[dict[str, Any]] = []
    for lookback in LOOKBACKS:
        rolling = completed.rolling(lookback, min_periods=lookback)
        for quantile in QUANTILES:
            quantile_tag = str(round(100 * quantile)).zfill(2)
            surfaces.append(
                {
                    "level_name": f"typical_q{quantile_tag}_lb{lookback}",
                    "lookback_hours": lookback,
                    "quantile": quantile,
                    "level": rolling.quantile(quantile).to_numpy(dtype=float),
                    "source_open": source,
                }
            )
    return surfaces


def transformed_surfaces(
    surfaces: list[dict[str, Any]],
    base: DataFrame,
    *,
    pair: str,
    mode: str,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    direction = (
        1.0
        if g0.stable_hash_int(f"g23-distribution-shift|{pair}") % 2 == 0
        else -1.0
    )
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
                        "g23-distribution-analogue|"
                        f"{pair}|{surface['level_name']}|{date.isoformat()}"
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


def relabel(events: DataFrame, surface: dict[str, Any]) -> DataFrame:
    if events.empty:
        return events
    output = events.copy()
    output["level_family"] = "rolling_completed_price_distribution"
    output["lookback_hours"] = int(surface["lookback_hours"])
    output["quantile"] = float(surface["quantile"])
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
    for (lookback, quantile), cell in actual.groupby(
        ["lookback_hours", "quantile"], observed=True, sort=False
    ):
        matched = g21d.matched_random_time_support(
            base,
            pair=pair,
            cohort=cohort,
            level_name=f"random_time_lb{lookback}_q{quantile}",
            actual=cell,
        )
        if matched.empty:
            continue
        matched["level_family"] = "rolling_completed_price_distribution"
        matched["lookback_hours"] = int(lookback)
        matched["quantile"] = float(quantile)
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
    surfaces = actual_surfaces(base)
    actual = surface_support(
        base,
        pair=pair,
        cohort=cohort,
        surfaces=surfaces,
        control="actual",
        event_kind="contact",
    )
    parts = [actual, matched_random_support(base, pair=pair, cohort=cohort, actual=actual)]
    parts.append(
        surface_support(
            base,
            pair=pair,
            cohort=cohort,
            surfaces=surfaces,
            control="near_miss",
            event_kind="near_miss",
        )
    )
    for control in ("random_recent_analogue", "stale_definition", "price_shift"):
        parts.append(
            surface_support(
                base,
                pair=pair,
                cohort=cohort,
                surfaces=transformed_surfaces(surfaces, base, pair=pair, mode=control),
                control=control,
                event_kind="contact",
            )
        )
    support = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    support.sort_values(
        ["control", "lookback_hours", "quantile", "event_time", "pre_distance_atr"],
        inplace=True,
        kind="stable",
    )
    support = support.drop_duplicates(
        ["control", "lookback_hours", "quantile", "event_time"], keep="first"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(support),
        "actual_counts": (
            actual.groupby(["lookback_hours", "quantile"], observed=True)
            .size()
            .astype(int)
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
        if manifest.get("status") != "frozen_before_generation23_distribution_outcomes":
            raise ValueError("Invalid Generation 23 distribution support freeze.")
        return manifest
    inventory = []
    pairs = g22a.cohort_pairs()
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 normal/meme pairs, got {len(pairs)}.")
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, overwrite))
        print(
            json.dumps(
                {"phase": "g23_distribution_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_distribution_outcomes",
        "branch_id": "g23e_causal_price_distribution_boundaries",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "lookbacks": list(LOOKBACKS),
        "quantiles": list(QUANTILES),
        "controls": list(CONTROLS),
        "surface_definition": {
            "input": "completed hourly typical price",
            "causal_shift_hours": 1,
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
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
    output["scope_kind"] = "lookback_and_quantile"
    output["scope_value"] = (
        "lb"
        + output["lookback_hours"].astype(int).astype(str)
        + "__q"
        + (100 * output["quantile"].astype(float)).round().astype(int).astype(str)
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
            raise ValueError(f"Frozen distribution dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g23_price_distribution_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    events = g23c.open_frozen_support_outcomes(
        manifest, phase="g23_distribution_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g23c.score_control_ladders(
        events, controls=CONTROLS
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g23c.write_scored_tables(
        run_dir,
        prefix="g23_price_distribution",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_price_distribution",
        "branch_completed": "g23e_causal_price_distribution_boundaries",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "lookbacks_or_quantiles_tuned_after_outcomes": False,
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
