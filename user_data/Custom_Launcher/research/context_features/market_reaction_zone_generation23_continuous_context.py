"""Repair sparse Generation 22 context labels with broad single-input bands."""

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
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
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
G17_LEVEL_BUILDER_PATH = Path(g17l.__file__).resolve()
DEFAULT_RUN_ID = "g23_continuous_context_20260828a"
DEFAULT_SUPPORT_ID = "g23_continuous_context_support_20260828a"
RECORD_ROOT = g23z.OUTPUT_ROOT / "continuous_context"
SUPPORT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "continuous_context_support"
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g23_continuous_context_support_freeze.json"
CONTEXT_WINDOWS = (4, 24)
CONTEXT_INPUTS = (
    "btc_absolute_return_over_atr",
    "equal_weight_median_absolute_return_over_atr",
    "equal_weight_relative_volume",
    "cross_coin_return_dispersion",
    "cross_coin_absolute_return_correlation",
)
BANDS = ("low", "middle", "high")
CONTROLS = (*g23z.LEVEL_CONTROLS, "same_band_no_level_time")
HORIZONS = g23z.HORIZONS_HOURS
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g23c.load_branch(
        "g23b_continuous_cross_asset_context_at_levels"
    )
    if tuple(branch["context_windows_hours"]) != CONTEXT_WINDOWS:
        raise ValueError("Generation 23 context windows drifted.")
    if tuple(branch["context_inputs"]) != CONTEXT_INPUTS:
        raise ValueError("Generation 23 context input registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 23 context controls drifted.")
    return frozen


def context_column(input_name: str, window: int) -> str:
    return f"{input_name}_w{window}"


def context_registry(context: DataFrame) -> dict[str, dict[str, Any]]:
    dates = pd.to_datetime(context["date"], utc=True, errors="raise")
    calibration = dates < g22d.CALIBRATION_END_EXCLUSIVE
    registry: dict[str, dict[str, Any]] = {}
    for window in CONTEXT_WINDOWS:
        for input_name in CONTEXT_INPUTS:
            column = context_column(input_name, window)
            values = pd.to_numeric(context.loc[calibration, column], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) < 1000:
                raise ValueError(f"Insufficient context calibration rows: {column}")
            key = f"w{window}__{input_name}"
            registry[key] = {
                "column": column,
                "window_hours": window,
                "input_name": input_name,
                "calibration_rows": len(finite),
                "lower_tertile": float(finite.quantile(1.0 / 3.0)),
                "upper_tertile": float(finite.quantile(2.0 / 3.0)),
            }
    return registry


def band_for_values(
    values: pd.Series, lower: float, upper: float
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series("unavailable", index=values.index, dtype="string")
    output.loc[numeric <= lower] = "low"
    output.loc[(numeric > lower) & (numeric <= upper)] = "middle"
    output.loc[numeric > upper] = "high"
    return output


def context_with_bands(
    context: DataFrame, registry: dict[str, dict[str, Any]]
) -> DataFrame:
    output = context.copy()
    calibration_mask = (
        pd.to_datetime(output["date"], utc=True) < g22d.CALIBRATION_END_EXCLUSIVE
    )
    for key, spec in registry.items():
        column = str(spec["column"])
        calibration = pd.to_numeric(
            output.loc[calibration_mask, column], errors="coerce"
        ).to_numpy(dtype=float)
        calibration = calibration[np.isfinite(calibration)]
        output[f"band__{key}"] = band_for_values(
            output[column], float(spec["lower_tertile"]), float(spec["upper_tertile"])
        )
        output[f"percentile__{key}"] = g20s.empirical_percentile(
            output[column], calibration
        )
    return output


def attach_context_bands(
    events: DataFrame,
    context: DataFrame,
    registry: dict[str, dict[str, Any]],
) -> DataFrame:
    if events.empty:
        return events
    value_columns = [str(spec["column"]) for spec in registry.values()]
    derived = [
        name
        for key in registry
        for name in (f"band__{key}", f"percentile__{key}")
    ]
    lookup = context[["date", *value_columns, *derived]].rename(
        columns={"date": "event_time"}
    )
    merged = events.merge(lookup, on="event_time", how="left", validate="many_to_one")
    frames: list[DataFrame] = []
    for key, spec in registry.items():
        copy = merged.copy()
        copy["context_window_hours"] = int(spec["window_hours"])
        copy["context_input"] = str(spec["input_name"])
        copy["context_value"] = pd.to_numeric(copy[str(spec["column"])], errors="coerce")
        copy["context_percentile"] = pd.to_numeric(
            copy[f"percentile__{key}"], errors="coerce"
        )
        copy["context_band"] = copy[f"band__{key}"].astype(str)
        frames.append(copy.loc[copy["context_band"].isin(BANDS)])
    expanded = pd.concat(frames, ignore_index=True, sort=False)
    expanded.sort_values(
        [
            "control",
            "context_window_hours",
            "context_input",
            "context_band",
            "event_time",
            "pre_distance_atr",
        ],
        inplace=True,
        kind="stable",
    )
    return expanded.drop_duplicates(
        [
            "control",
            "context_window_hours",
            "context_input",
            "context_band",
            "event_time",
        ],
        keep="first",
    ).reset_index(drop=True)


def same_band_no_level_events(
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
    atr = pd.to_numeric(frame["base_atr"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    lookup = context.set_index("date")
    rows: list[dict[str, Any]] = []
    group_keys = ["context_window_hours", "context_input", "context_band", "period"]
    for identity, cell in actual_expanded.groupby(group_keys, observed=True, sort=False):
        window, input_name, band, period_name = identity
        key = f"w{int(window)}__{input_name}"
        bands = lookup[f"band__{key}"].reindex(dates).astype(str).to_numpy()
        values = pd.to_numeric(
            lookup[context_column(str(input_name), int(window))].reindex(dates),
            errors="coerce",
        ).to_numpy(dtype=float)
        percentiles = pd.to_numeric(
            lookup[f"percentile__{key}"].reindex(dates), errors="coerce"
        ).to_numpy(dtype=float)
        indexes = np.arange(len(frame), dtype=int)
        eligible = np.flatnonzero(
            (bands == str(band))
            & (period == str(period_name))
            & ~has_level_contact
            & np.isfinite(atr)
            & (atr > 0.0)
            & (indexes < len(frame) - max(HORIZONS))
        )
        rng = np.random.default_rng(
            g0.stable_hash_int(
                f"g23-same-band-no-level|{pair}|{window}|{input_name}|{band}|{period_name}"
            )
        )
        blocked = np.zeros(len(frame), dtype=bool)
        selected: list[int] = []
        for candidate in rng.permutation(eligible):
            index = int(candidate)
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
                    "match_tier": "exact_single_input_band_and_period",
                    "context_window_hours": int(window),
                    "context_input": str(input_name),
                    "context_band": str(band),
                    "context_value": values[index],
                    "context_percentile": percentiles[index],
                }
            )
    return DataFrame.from_records(rows)


def raw_level_controls(
    frame: DataFrame, *, pair: str, cohort: str
) -> tuple[DataFrame, list[dict[str, Any]]]:
    surfaces = g22d.selected_surfaces(frame)
    actual = g22d.surface_events(
        frame,
        pair=pair,
        cohort=cohort,
        surfaces=surfaces,
        control="actual",
        event_kind="contact",
    )
    parts = [actual, g22d.matched_random_events(frame, pair=pair, cohort=cohort, actual=actual)]
    parts.append(
        g22d.surface_events(
            frame,
            pair=pair,
            cohort=cohort,
            surfaces=surfaces,
            control="near_miss",
            event_kind="near_miss",
        )
    )
    for control in ("random_recent_analogue", "stale_definition", "price_shift"):
        parts.append(
            g22d.surface_events(
                frame,
                pair=pair,
                cohort=cohort,
                surfaces=g22d.transformed_surfaces(
                    surfaces, frame, pair=pair, mode=control
                ),
                control=control,
                event_kind="contact",
            )
        )
    return pd.concat([part for part in parts if not part.empty], ignore_index=True), surfaces


def pair_support(
    pair: str,
    cohort: str,
    context: DataFrame,
    registry: dict[str, dict[str, Any]],
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
    source_manifest = json.loads(g17l.COHORT_MANIFESTS[cohort].read_text(encoding="utf-8"))
    frame = g17l.level_surface(pair, source_manifest)
    raw, surfaces = raw_level_controls(frame, pair=pair, cohort=cohort)
    expanded = attach_context_bands(raw, context, registry)
    actual_expanded = expanded.loc[expanded["control"].eq("actual")].copy()
    no_level = same_band_no_level_events(
        frame,
        pair=pair,
        cohort=cohort,
        actual_expanded=actual_expanded,
        context=context,
        has_level_contact=g22d.actual_contact_mask(frame, surfaces),
    )
    support = pd.concat([expanded, no_level], ignore_index=True, sort=False)
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
    parent = json.loads(g22d.SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    if parent.get("status") != "frozen_before_generation22_context_outcomes":
        raise ValueError("Generation 22 cross-asset context support is not frozen.")
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation23_continuous_context_outcomes":
            raise ValueError("Invalid Generation 23 continuous context support freeze.")
        return manifest
    context = pd.read_parquet(Path(parent["context_artifact"]["path"]))
    context["date"] = pd.to_datetime(context["date"], utc=True, errors="raise")
    registry = context_registry(context)
    context = context_with_bands(context, registry)
    inventory = []
    pairs = g22a.cohort_pairs()
    for number, (cohort, pair) in enumerate(pairs, start=1):
        inventory.append(pair_support(pair, cohort, context, registry, overwrite))
        print(
            json.dumps(
                {"phase": "g23_continuous_context_support", "processed": number, "total": 20}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_continuous_context_outcomes",
        "branch_id": "g23b_continuous_cross_asset_context_at_levels",
        "future_outcome_columns_read": False,
        "future_ohlcv_paths_opened": False,
        "context_windows": list(CONTEXT_WINDOWS),
        "context_inputs": list(CONTEXT_INPUTS),
        "bands": list(BANDS),
        "controls": list(CONTROLS),
        "context_registry": registry,
        "inventory": inventory,
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation23_common": artifact(g23c.ANALYSIS_PATH),
            "generation22_context_builder": artifact(g22d.ANALYSIS_PATH),
            "generation22_context_support": artifact(g22d.SUPPORT_MANIFEST),
            "generation22_context_artifact": artifact(
                Path(parent["context_artifact"]["path"])
            ),
            "generation17_level_builder": artifact(G17_LEVEL_BUILDER_PATH),
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


def rank_diagnostics(events: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    keys = [
        "cohort",
        "pair",
        "period",
        "control",
        "context_window_hours",
        "context_input",
    ]
    for identity, cell in events.groupby(keys, observed=True, sort=False):
        x = pd.to_numeric(cell["context_percentile"], errors="coerce")
        for metric in (
            "any_recross",
            "repeated_recross",
            "crossing_count",
            "dwell_fraction",
            "future_volume_ratio",
        ):
            for horizon in HORIZONS:
                y = pd.to_numeric(
                    cell[f"metric__{metric}_h{horizon}"], errors="coerce"
                )
                valid = x.notna() & y.notna()
                if int(valid.sum()) < 12:
                    correlation = np.nan
                else:
                    correlation = x.loc[valid].corr(y.loc[valid], method="spearman")
                record = dict(zip(keys, identity, strict=True))
                record.update(
                    {
                        "metric": metric,
                        "horizon_hours": horizon,
                        "rows": int(valid.sum()),
                        "spearman_context_percentile": correlation,
                    }
                )
                records.append(record)
    return DataFrame.from_records(records)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch_supports = g23c.require_all_frozen_supports()
    for path, key in (
        (ANALYSIS_PATH, "analysis_script"),
        (g23c.ANALYSIS_PATH, "generation23_common"),
        (g22d.ANALYSIS_PATH, "generation22_context_builder"),
        (g22d.SUPPORT_MANIFEST, "generation22_context_support"),
        (G17_LEVEL_BUILDER_PATH, "generation17_level_builder"),
    ):
        if g0.sha256_file(path) != manifest["source_contracts"][key]["sha256"]:
            raise ValueError(f"Frozen continuous-context dependency changed: {key}")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g23_continuous_context_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    events = g23c.open_frozen_support_outcomes(
        manifest, phase="g23_continuous_context_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g23c.score_control_ladders(
        events, controls=CONTROLS
    )
    diagnostics = rank_diagnostics(events)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g23c.write_scored_tables(
        run_dir,
        prefix="g23_continuous_context",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    paths["continuous_rank_diagnostics"] = (
        run_dir / "g23_continuous_context_rank_diagnostics.csv"
    )
    g0.atomic_write_csv(diagnostics, paths["continuous_rank_diagnostics"])
    result = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_continuous_context",
        "branch_completed": "g23b_continuous_cross_asset_context_at_levels",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "rare_multivariable_conjunctions_used": False,
            "continuous_rank_association_is_diagnostic_only": True,
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
