"""Test what kind of direction-neutral behaviour follows calculated-area contact."""

from __future__ import annotations

# Bound dataframe pools before numerical imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
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
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_matched_paths as g15m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_direct_attribution as g16d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)


DEFAULT_RUN_ID = "g17_crossing_semantics_20260822a"
RECORD_ROOT = g17z.OUTPUT_ROOT / "crossing_semantics"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation17_branches"
    / "g17_broad_branch_layer"
    / "crossing_semantics"
)
HORIZONS = (1, 2, 4)
CONTROLS = {
    "matched_random_time": "matched_ordinary_time",
    "near_miss": "genuine_near_miss",
}
METRICS = (
    "any_recross",
    "repeated_recross",
    "two_sided_traversal",
    "one_sided_rejection",
    "one_sided_breakthrough",
    "dwell_fraction",
)
EXTRA_OUTCOMES = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"away_excursion_atr_h{horizon}",
        f"through_excursion_atr_h{horizon}",
        f"dwell_fraction_h{horizon}",
    )
)
RAW_COLUMNS = tuple(dict.fromkeys((*g16d.RAW_READ_COLUMNS, *EXTRA_OUTCOMES)))


def artifact(path: Path) -> dict[str, Any]:
    return g16d.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g17z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation17_outcomes":
        raise ValueError("Generation 17 freeze is invalid.")
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g17b_crossing_semantics"
    )
    if tuple(branch["targets"]) != METRICS:
        raise ValueError("Generation 17 crossing semantics drifted after freeze.")
    return frozen


def raw_frame(task: g16d.PairTask) -> DataFrame:
    path = Path(task.source_path)
    if not path.is_file() or g0.sha256_file(path) != task.source_sha256:
        raise ValueError(f"Frozen source changed: {path}")
    frame = pd.read_parquet(
        path,
        columns=list(RAW_COLUMNS),
        filters=[("control", "in", ["actual", *CONTROLS])],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    frame["absolute_contact_pressure_change"] = pd.to_numeric(
        frame["contact_pressure_change"], errors="coerce"
    ).abs()
    actual_clock = set(
        frame.loc[frame["control"].eq("actual"), "event_time"].astype("int64")
    )
    frame = g13d.add_analysis_windows(frame, task.cohort)
    frame = frame.loc[
        frame["source_timeframe"].isin(task.strict_timeframes)
        & frame["level_family"].isin(task.strict_families)
    ].copy()
    frame = g15m.causal_deduplicate_by_family(frame)
    contaminated = (
        ~frame["control"].eq("actual")
        & frame["event_time"].astype("int64").isin(actual_clock)
    )
    frame = frame.loc[~contaminated].copy()
    for horizon in HORIZONS:
        # Reuse the proven matcher by mapping the additional direction-neutral
        # outcomes into its carried numeric slots after all selection is fixed.
        frame[f"abs_excursion_atr_h{horizon}"] = pd.to_numeric(
            frame[f"away_excursion_atr_h{horizon}"], errors="coerce"
        )
        frame[f"range_ratio_h{horizon}"] = pd.to_numeric(
            frame[f"through_excursion_atr_h{horizon}"], errors="coerce"
        )
        frame[f"volume_ratio_h{horizon}"] = pd.to_numeric(
            frame[f"crossings_h{horizon}"], errors="coerce"
        )
        frame[f"pressure_change_h{horizon}"] = pd.to_numeric(
            frame[f"dwell_fraction_h{horizon}"], errors="coerce"
        )
    frame.attrs["removed_control_rows_at_real_contacts"] = int(contaminated.sum())
    return frame


def numeric(frame: DataFrame, prefix: str, column: str) -> Series:
    return pd.to_numeric(frame[f"{prefix}__{column}"], errors="coerce")


def binary(mask: Series, *required: Series) -> Series:
    output = mask.astype(float)
    missing = Series(False, index=output.index)
    for values in required:
        missing |= values.isna()
    output.loc[missing] = np.nan
    return output


def semantic_values(frame: DataFrame, prefix: str, metric: str, horizon: int) -> Series:
    threshold = numeric(frame, prefix, "zone_half_width_atr").clip(lower=0.5)
    away = numeric(frame, prefix, f"abs_excursion_atr_h{horizon}")
    through = numeric(frame, prefix, f"range_ratio_h{horizon}")
    crossings = numeric(frame, prefix, f"volume_ratio_h{horizon}")
    dwell = numeric(frame, prefix, f"pressure_change_h{horizon}")
    if metric == "any_recross":
        return binary(crossings.ge(1), crossings)
    if metric == "repeated_recross":
        return binary(crossings.ge(2), crossings)
    if metric == "two_sided_traversal":
        return binary((away >= threshold) & (through >= threshold), away, through, threshold)
    if metric == "one_sided_rejection":
        return binary((away >= threshold) & (through < threshold), away, through, threshold)
    if metric == "one_sided_breakthrough":
        return binary((through >= threshold) & (away < threshold), away, through, threshold)
    if metric == "dwell_fraction":
        return dwell
    raise KeyError(metric)


def summarize(
    matched: DataFrame, *, horizon: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        scored = matched[
            [
                "cohort",
                "pair",
                "window",
                "analysis_period",
                "comparison",
                "actual_event_time",
            ]
        ].copy()
        scored["actual_value"] = semantic_values(matched, "actual", metric, horizon)
        scored["control_value"] = semantic_values(matched, "control", metric, horizon)
        scored["difference"] = scored["actual_value"] - scored["control_value"]
        scored = scored.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["actual_value", "control_value", "difference"]
        )
        for key, cell in scored.groupby(
            ["cohort", "pair", "window", "analysis_period", "comparison"],
            observed=True,
            sort=False,
        ):
            pair_rows.append(
                {
                    "route_id": "crossing_semantics",
                    "scope_value": "all_retained_levels",
                    "metric": metric,
                    "horizon_hours": horizon,
                    "cohort": key[0],
                    "pair": key[1],
                    "window": key[2],
                    "analysis_period": key[3],
                    "comparison": key[4],
                    "independent_rows": len(cell),
                    "actual_mean": float(cell["actual_value"].mean()),
                    "control_mean": float(cell["control_value"].mean()),
                    "paired_mean_difference": float(cell["difference"].mean()),
                }
            )
        scored["week"] = pd.to_datetime(
            scored["actual_event_time"], utc=True
        ).dt.floor("7D")
        grouped = (
            scored.groupby(
                ["cohort", "pair", "window", "analysis_period", "comparison", "week"],
                observed=True,
                sort=False,
            )["difference"]
            .agg(["sum", "size"])
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            weekly_rows.append(
                {
                    "route_id": "crossing_semantics",
                    "scope_value": "all_retained_levels",
                    "metric": metric,
                    "horizon_hours": horizon,
                    "cohort": row.cohort,
                    "pair": row.pair,
                    "window": row.window,
                    "analysis_period": row.analysis_period,
                    "comparison": row.comparison,
                    "week": row.week,
                    "difference_sum": float(row.sum),
                    "rows": int(row.size),
                }
            )
    return pair_rows, weekly_rows


def build_pair(task: g16d.PairTask) -> dict[str, Any]:
    output_dir = Path(task.output_dir)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    pair_path = output_dir / "pair_summaries" / f"{stem}.parquet"
    weekly_path = output_dir / "weekly_blocks" / f"{stem}.parquet"
    audit_path = output_dir / "match_audits" / f"{stem}.parquet"
    if all(path.is_file() for path in (pair_path, weekly_path, audit_path)) and not task.overwrite:
        return {
            "status": "existing",
            "cohort": task.cohort,
            "pair": task.pair,
            "pair_summary_path": str(pair_path),
            "pair_summary_sha256": g0.sha256_file(pair_path),
            "weekly_path": str(weekly_path),
            "weekly_sha256": g0.sha256_file(weekly_path),
            "audit_path": str(audit_path),
            "audit_sha256": g0.sha256_file(audit_path),
        }
    raw = raw_frame(task)
    actual = raw.loc[raw["control"].eq("actual")].copy()
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for control, comparison in CONTROLS.items():
        for horizon in HORIZONS:
            matched, audit = g16d.matched_frame(
                actual,
                raw.loc[raw["control"].eq(control)].copy(),
                exact_keys=(
                    "window",
                    "analysis_period",
                    "source_timeframe",
                    "level_family",
                    "approach_state",
                ),
                state_columns=g16d.CONTACT_MATCH_FEATURES,
                horizon=horizon,
                comparison=comparison,
            )
            pairs, weeks = summarize(matched, horizon=horizon)
            pair_rows.extend(pairs)
            weekly_rows.extend(weeks)
            audits.append(
                {
                    "route_id": "crossing_semantics",
                    "scope_value": "all_retained_levels",
                    "comparison": comparison,
                    "horizon_hours": horizon,
                    **audit,
                    "matched_independent_rows": len(matched),
                }
            )
    pair_frame = DataFrame.from_records(pair_rows)
    weekly_frame = DataFrame.from_records(weekly_rows)
    audit_frame = DataFrame.from_records(audits)
    if pair_frame.empty or weekly_frame.empty:
        raise ValueError(f"No Generation 17 semantic summaries for {task.pair}.")
    g0.atomic_write_parquet(pair_frame, pair_path)
    g0.atomic_write_parquet(weekly_frame, weekly_path)
    g0.atomic_write_parquet(audit_frame, audit_path)
    return {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "removed_control_rows_at_real_contacts": raw.attrs[
            "removed_control_rows_at_real_contacts"
        ],
        "pair_summary_rows": len(pair_frame),
        "weekly_rows": len(weekly_frame),
        "audit_rows": len(audit_frame),
        "pair_summary_path": str(pair_path),
        "pair_summary_sha256": g0.sha256_file(pair_path),
        "weekly_path": str(weekly_path),
        "weekly_sha256": g0.sha256_file(weekly_path),
        "audit_path": str(audit_path),
        "audit_sha256": g0.sha256_file(audit_path),
    }


def safe_build_pair(task: g16d.PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {
            "status": "failed",
            "cohort": task.cohort,
            "pair": task.pair,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tasks(tasks: Sequence[g16d.PairTask], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        futures = {executor.submit(safe_build_pair, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g17_crossing_semantics_pair",
                        "processed": index,
                        "total": len(tasks),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def joint_decisions(leads: DataFrame) -> DataFrame:
    keys = ["metric", "horizon_hours", "market_scope"]
    rows: list[dict[str, Any]] = []
    for key, cell in leads.groupby(keys, observed=True, sort=False):
        scope = str(key[2])
        expected_windows = (
            {"standard_validation"}
            if scope == "top_ten_memes"
            else {"standard_validation", "recent_normal_chronology"}
        )
        expected_rows = len(expected_windows) * len(CONTROLS)
        selected = cell.loc[cell["window"].isin(expected_windows)].copy()
        complete = bool(
            len(selected) == expected_rows
            and set(selected["window"]) == expected_windows
            and set(selected["comparison"]) == set(CONTROLS.values())
        )
        signs = set(selected.loc[selected["effect_sign"].ne("mixed"), "effect_sign"])
        same_sign = bool(complete and len(signs) == 1)
        point = bool(same_sign and selected["point_pass"].astype(bool).all())
        strict = bool(same_sign and selected["strict_pass"].astype(bool).all())
        rows.append(
            {
                "metric": key[0],
                "horizon_hours": key[1],
                "market_scope": scope,
                "status": (
                    "strict_repeated"
                    if strict
                    else "point_repeated"
                    if point
                    else "not_retained"
                ),
                "complete_control_chronology": complete,
                "same_effect_sign": same_sign,
                "effect_sign": next(iter(signs)) if len(signs) == 1 else "mixed",
                "point_repeated": point,
                "strict_repeated": strict,
                "minimum_effect": float(selected["minimum_effect"].min())
                if len(selected)
                else np.nan,
                "maximum_effect": float(selected["maximum_effect"].max())
                if len(selected)
                else np.nan,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    load_freeze()
    _, event_manifest, g13_manifests = g16d.load_contracts()
    run_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    result_path = run_dir / "g17_crossing_semantics_result.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # The parent direct contract is the exact source of the retained causal surface.
    parent_frozen, _, _ = g16d.load_contracts()
    strict_families = tuple(parent_frozen["parent_surface"]["strict_level_families"])
    strict_timeframes = tuple(parent_frozen["parent_surface"]["strict_source_timeframes"])
    by_cohort = {str(item["cohort"]): item for item in g13_manifests}
    tasks: list[g16d.PairTask] = []
    for item in event_manifest["tasks"]:
        cohort = str(item["cohort"])
        pair = str(item["pair"])
        cached = next(row for row in by_cohort[cohort]["inventory"] if row["pair"] == pair)
        tasks.append(
            g16d.PairTask(
                cohort=cohort,
                pair=pair,
                source_path=str(item["event_path"]),
                source_sha256=str(item["event_sha256"]),
                feature_path=str(cached["mtf_feature_path"]),
                feature_sha256=str(cached["mtf_feature_sha256"]),
                evaluation_path=str(cached["mtf_evaluation_path"]),
                evaluation_sha256=str(cached["mtf_evaluation_sha256"]),
                strict_families=strict_families,
                strict_timeframes=strict_timeframes,
                output_dir=str(artifact_dir),
                overwrite=bool(args.overwrite),
            )
        )
    inventory = run_tasks(tasks, args.workers)
    failures = [item for item in inventory if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 17 crossing-semantics failures: {failures}")
    pair_summary = g16d.expand_market_scopes(
        pd.concat(
            [pd.read_parquet(item["pair_summary_path"]) for item in inventory],
            ignore_index=True,
        )
    )
    weekly = g16d.expand_market_scopes(
        pd.concat(
            [pd.read_parquet(item["weekly_path"]) for item in inventory],
            ignore_index=True,
        )
    )
    audits = pd.concat(
        [pd.read_parquet(item["audit_path"]) for item in inventory],
        ignore_index=True,
    )
    scores = g16d.equal_coin_scores(pair_summary, weekly)
    leads = g16d.score_leads(scores)
    decisions = joint_decisions(leads)
    paths = {
        "inventory": run_dir / "pair_inventory.csv",
        "pair_summary": run_dir / "pair_semantic_summary.csv",
        "weekly_blocks": artifact_dir / "weekly_semantic_blocks.parquet",
        "match_audits": run_dir / "match_audits.csv",
        "equal_coin_scores": run_dir / "equal_coin_scores.csv",
        "period_leads": run_dir / "period_leads.csv",
        "joint_decisions": run_dir / "joint_semantic_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(inventory), paths["inventory"])
    g0.atomic_write_csv(pair_summary, paths["pair_summary"])
    g0.atomic_write_parquet(weekly, paths["weekly_blocks"])
    g0.atomic_write_csv(audits, paths["match_audits"])
    g0.atomic_write_csv(scores, paths["equal_coin_scores"])
    g0.atomic_write_csv(leads, paths["period_leads"])
    g0.atomic_write_csv(decisions, paths["joint_decisions"])
    result = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation17_crossing_semantics",
        "branch_id": "g17b_crossing_semantics",
        "pairs_completed": len(inventory),
        "pair_failures": 0,
        "metrics": list(METRICS),
        "horizons_hours": list(HORIZONS),
        "strict_repeated": int(decisions["strict_repeated"].sum()),
        "point_repeated": int(decisions["point_repeated"].sum()),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "outcomes_used_for_matching": False,
        },
        "source_contracts": {
            "generation17_freeze": artifact(g17z.FREEZE_PATH),
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps({**result, "result_path": str(result_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
