from __future__ import annotations

# Outcome-blind G5E readiness and event-overlap audit. Frozen parquet only: no live
# SQLite collector is read or paused by this phase.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_external_context as g3h,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.source_coverage_audit import (
    build_source_audit,
)
from user_data.Custom_Launcher.research.context_features.trader_confluence_feature_taxonomy import (
    source_detail,
)


SCHEMA_VERSION = 1
BRANCH_ID = "g5e_external_context_common_support_and_conditioning"
G5_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
G4D_ROOT = (
    OUTPUT_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
REPORT_ROOT = OUTPUT_ROOT / "generation5_branches" / BRANCH_ID
CONTEXT_SNAPSHOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "exports"
    / "context_features_1h_20260626_000458.parquet"
)
ORDERBOOK_SNAPSHOT = (
    REPO_ROOT
    / "user_data"
    / "orderbook_data"
    / "historical_bybit"
    / "features"
    / "orderbook_trader_state_1h_bybit_linear.parquet"
)
MINIMUM_EVENTS_PER_PERIOD = 50
MINIMUM_COINS_PER_PERIOD = 5
GLOBAL_INDEPENDENCE_HOURS = 4

SURFACE_RUNS: dict[str, str] = {
    "normal_thin_lvn_mixed_cluster": "g4d_full_g3a_normal_20260820b_time_baseline_repair",
    "normal_isolated_confirmed_swing_density": (
        "g4d_full_g3d_normal_isolated_20260820b_time_baseline_repair"
    ),
    "normal_confirmed_swing_density_cluster": (
        "g4d_full_g3d_normal_cluster_20260820b_time_baseline_repair"
    ),
    "meme_confirmed_swing_density_cluster": (
        "g4d_full_g3d_meme_cluster_20260820b_time_baseline_repair"
    ),
}

CONTEXT_BLOCKS: dict[str, tuple[str, ...]] = {
    "gdelt_aggregate": ("context_gdelt_events",),
    "global_market_macro": ("context_global_market_macro",),
    "live_news_media": (
        "context_article_source_activity",
        "context_topic_severity",
    ),
}
ORDERBOOK_BLOCK = "btc_bybit_orderbook_pressure"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit timestamp-safe external-context overlap for the four retained G4D "
            "volume surfaces without reading reaction outcomes."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return run_preflight(args.run_id, overwrite=bool(args.overwrite))


def run_preflight(run_id: str, *, overwrite: bool) -> int:
    branch = validate_g5_branch()
    run_dir = REPORT_ROOT / run_id
    record_path = run_dir / "g5e_preflight_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"G5E preflight already exists: {record_path}")
    validate_file(CONTEXT_SNAPSHOT)
    validate_file(ORDERBOOK_SNAPSHOT)
    source_manifests = validate_surface_manifests()
    request = request_contract(run_id, branch, source_manifests)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "branch_id": BRANCH_ID,
        "status": "auditing_outcome_blind_external_overlap",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        context, context_audit, context_masks = load_context_masks(CONTEXT_SNAPSHOT)
        orderbook = g3h.load_causal_pressure_surface(ORDERBOOK_SNAPSHOT)
        rows: list[dict[str, Any]] = []
        input_counts: dict[str, Any] = {}
        for surface_id, manifest in source_manifests.items():
            events = load_surface_events(surface_id, manifest)
            independent = select_global_independent(
                events, hours=GLOBAL_INDEPENDENCE_HOURS
            )
            input_counts[surface_id] = {
                "source_events": len(events),
                "globally_independent_events": len(independent),
                "pairs": int(independent["pair"].nunique()),
                "periods": {
                    str(key): int(value)
                    for key, value in independent.groupby("period", observed=True).size().items()
                },
            }
            rows.extend(
                context_overlap_rows(
                    independent,
                    surface_id=surface_id,
                    validation_periods=tuple(manifest["validation_periods"]),
                    context=context,
                    context_masks=context_masks,
                )
            )
            rows.extend(
                orderbook_overlap_rows(
                    independent,
                    surface_id=surface_id,
                    validation_periods=tuple(manifest["validation_periods"]),
                    orderbook=orderbook,
                )
            )

        overlap = DataFrame(rows).sort_values(
            ["surface_id", "context_block", "period"]
        ).reset_index(drop=True)
        decisions = block_decisions(overlap)
        overlap_path = run_dir / "g5e_external_overlap_by_surface.csv"
        decisions_path = run_dir / "g5e_external_block_decisions.csv"
        context_audit_path = run_dir / "g5e_context_snapshot_block_audit.csv"
        atomic_write_csv(overlap, overlap_path)
        atomic_write_csv(decisions, decisions_path)
        atomic_write_csv(
            context_audit.loc[
                context_audit["source_detail_block"].isin(
                    sorted({item for blocks in CONTEXT_BLOCKS.values() for item in blocks})
                )
            ].reset_index(drop=True),
            context_audit_path,
        )
        opened = decisions.loc[decisions["passed_both_periods"].astype(bool)]
        record.update(
            {
                "status": (
                    "completed_with_fair_context_surfaces"
                    if not opened.empty
                    else "parked_no_external_block_with_fair_overlap"
                ),
                "completed_at_utc": utc_now(),
                "reaction_outcomes_opened": False,
                "direction_outcomes_opened": False,
                "profit_used": False,
                "input_event_counts": input_counts,
                "context_snapshot_rows": len(context),
                "orderbook_snapshot_rows": len(orderbook),
                "overlap_cells": len(overlap),
                "fair_surface_context_cells": len(opened),
                "fair_surface_context_combinations": opened[
                    ["surface_id", "context_block"]
                ].to_dict(orient="records"),
                "parked_surface_context_combinations": decisions.loc[
                    ~decisions["passed_both_periods"].astype(bool),
                    ["surface_id", "context_block", "classification"],
                ].to_dict(orient="records"),
                "artifacts": {
                    "external_overlap_csv": artifact_record(overlap_path),
                    "external_block_decisions_csv": artifact_record(decisions_path),
                    "context_snapshot_block_audit_csv": artifact_record(
                        context_audit_path
                    ),
                },
                "plain_language_result": (
                    "This phase only checked whether each frozen external source has enough "
                    "timestamp-safe overlap with independently selected level events. It did "
                    "not inspect volume, price, direction, or profit outcomes."
                ),
            }
        )
        atomic_write_json(record, record_path)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "fair_surface_context_cells": record[
                        "fair_surface_context_combinations"
                    ],
                    "parked_cells": len(record["parked_surface_context_combinations"]),
                },
                indent=2,
            )
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def validate_g5_branch() -> dict[str, Any]:
    validate_file(G5_BATCH)
    batch = json.loads(G5_BATCH.read_text(encoding="utf-8"))
    if int(batch.get("generation", -1)) != 5:
        raise ValueError("G5 batch generation is not 5")
    if batch.get("status") != "frozen_before_generation5_reaction_outcomes":
        raise ValueError("G5 batch is not frozen before outcomes")
    matches = [item for item in batch.get("branches", []) if item.get("id") == BRANCH_ID]
    if len(matches) != 1 or matches[0].get("status") != "frozen_next_batch":
        raise ValueError(f"Frozen branch missing or invalid: {BRANCH_ID}")
    return matches[0]


def validate_surface_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for surface_id, run_id in SURFACE_RUNS.items():
        path = G4D_ROOT / run_id / "manifest.json"
        validate_file(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"G4D source run is not completed: {run_id}")
        if not manifest.get("validation_periods"):
            raise ValueError(f"G4D source has no validation periods: {run_id}")
        manifest["_manifest_path"] = str(path)
        manifest["_manifest_sha256"] = sha256_file(path)
        manifests[surface_id] = manifest
    return manifests


def request_contract(
    run_id: str,
    branch: dict[str, Any],
    source_manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "branch": branch,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "g5_batch": str(G5_BATCH),
        "g5_batch_sha256": sha256_file(G5_BATCH),
        "context_snapshot": artifact_record(CONTEXT_SNAPSHOT),
        "orderbook_snapshot": artifact_record(ORDERBOOK_SNAPSHOT),
        "surface_manifests": {
            key: {
                "path": value["_manifest_path"],
                "sha256": value["_manifest_sha256"],
                "validation_periods": value["validation_periods"],
                "cohort": value["cohort"],
            }
            for key, value in source_manifests.items()
        },
        "selection": {
            "event_fields": [
                "date",
                "period",
                "event_state",
                "level_identity",
                "source_available_at",
                "pair",
            ],
            "global_independence_hours": GLOBAL_INDEPENDENCE_HOURS,
            "minimum_events_per_period": MINIMUM_EVENTS_PER_PERIOD,
            "minimum_coins_per_period": MINIMUM_COINS_PER_PERIOD,
            "context_blocks": {key: list(value) for key, value in CONTEXT_BLOCKS.items()},
            "orderbook_block": ORDERBOOK_BLOCK,
            "reaction_outcomes_used": False,
            "direction_outcomes_used": False,
            "profit_used": False,
        },
        "missing_data_rule": (
            "Missing, stale, low-coverage, or future-timestamped rows remain unavailable "
            "and are never treated as quiet or zero."
        ),
        "outcome_models_open_only_for_cells_passing_both_periods": True,
    }


def load_surface_events(surface_id: str, manifest: dict[str, Any]) -> DataFrame:
    parts: list[DataFrame] = []
    for item in manifest["source_contracts"]["cache_inventory"]:
        path = Path(str(item["event_path"]))
        validate_file(path)
        if sha256_file(path) != str(item["event_sha256"]):
            raise ValueError(f"G4D event cache changed: {path}")
        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "period",
                "event_state",
                "level_identity",
                "source_available_at",
            ],
        )
        frame["pair"] = str(item["pair"])
        parts.append(frame)
    events = pd.concat(parts, ignore_index=True, sort=False)
    events["date"] = pd.to_datetime(events["date"], utc=True)
    events["source_available_at"] = pd.to_datetime(
        events["source_available_at"], utc=True
    )
    events = events.loc[
        events["period"].isin(manifest["validation_periods"])
    ].copy()
    if (events["source_available_at"] > events["date"]).any():
        raise ValueError(f"Future level source in G4D event cache: {surface_id}")
    events["surface_id"] = surface_id
    events["cohort"] = str(manifest["cohort"])
    events["selection_key"] = events.apply(
        lambda row: (
            f"g5e|{surface_id}|{row['period']}|{row['pair']}|"
            f"{pd.Timestamp(row['date']).isoformat()}|{row['event_state']}|"
            f"{row['level_identity']}"
        ),
        axis=1,
    )
    events["selection_hash"] = events["selection_key"].map(sha256_text)
    if events["selection_key"].duplicated().any():
        raise ValueError(f"Duplicate event keys in G4D surface: {surface_id}")
    return events.reset_index(drop=True)


def select_global_independent(events: DataFrame, *, hours: int) -> DataFrame:
    rows: list[Series] = []
    minimum_ns = int(pd.Timedelta(hours=hours).value)
    for _, period_frame in events.groupby("period", sort=True, observed=True):
        kept_ns: list[int] = []
        for _, row in period_frame.sort_values("selection_hash", kind="mergesort").iterrows():
            event_ns = int(pd.Timestamp(row["date"]).value)
            if any(abs(event_ns - previous) < minimum_ns for previous in kept_ns):
                continue
            rows.append(row)
            kept_ns.append(event_ns)
    if not rows:
        return events.iloc[0:0].copy()
    return DataFrame(rows).sort_values(["period", "date", "pair"]).reset_index(drop=True)


def load_context_masks(
    path: Path,
) -> tuple[DataFrame, DataFrame, dict[str, dict[str, Series]]]:
    required_blocks = {item for blocks in CONTEXT_BLOCKS.values() for item in blocks}
    schema_columns = pq.ParquetFile(path).schema.names
    selected = [
        column
        for column in schema_columns
        if column == "date"
        or column
        in {
            "max_source_available_at",
            "min_source_available_at",
            "missing_available_at_rows",
        }
        or source_detail(f"ctx_{column}") in required_blocks
    ]
    raw = pd.read_parquet(path, columns=selected)
    raw = raw.rename(columns={column: f"ctx_{column}" for column in raw if column != "date"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    maximum = pd.to_datetime(raw["ctx_max_source_available_at"], utc=True, errors="coerce")
    raw["context_present"] = maximum.notna().astype(float)
    raw["context_source_future_violation"] = maximum.gt(raw["date"]).astype(float)
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    audit, masks = build_source_audit(raw, min_source_coverage_ratio=0.01)
    return raw, audit, masks


def context_overlap_rows(
    events: DataFrame,
    *,
    surface_id: str,
    validation_periods: tuple[str, ...],
    context: DataFrame,
    context_masks: dict[str, dict[str, Series]],
) -> list[dict[str, Any]]:
    source = DataFrame({"date": context["date"]})
    for block_id, detail_blocks in CONTEXT_BLOCKS.items():
        ready = Series(True, index=context.index)
        for detail in detail_blocks:
            usable = context_masks.get(detail, {}).get("usable")
            ready &= (
                usable.reindex(context.index).fillna(False)
                if usable is not None
                else False
            )
        source[block_id] = ready.to_numpy(dtype=bool)
    joined = events.merge(source, on="date", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for block_id in CONTEXT_BLOCKS:
        usable = joined[block_id].fillna(False).astype(bool)
        rows.extend(
            summarize_overlap(
                joined,
                usable=usable,
                surface_id=surface_id,
                context_block=block_id,
                validation_periods=validation_periods,
                source_note="frozen_context_parquet_exact_hour_and_timestamp_safe_mask",
            )
        )
    return rows


def orderbook_overlap_rows(
    events: DataFrame,
    *,
    surface_id: str,
    validation_periods: tuple[str, ...],
    orderbook: DataFrame,
) -> list[dict[str, Any]]:
    source = events.copy()
    source["high_event_time"] = source["date"]
    attached = g3h.attach_orderbook_state(
        source, orderbook, side="high", label="current"
    )
    return summarize_overlap(
        attached,
        usable=attached["high_current_usable"].fillna(False).astype(bool),
        surface_id=surface_id,
        context_block=ORDERBOOK_BLOCK,
        validation_periods=validation_periods,
        source_note=(
            "frozen_BTC_Bybit_global_context_only_coverage_ge_0.8_age_le_2h_"
            "source_max_ts_not_after_event"
        ),
    )


def summarize_overlap(
    frame: DataFrame,
    *,
    usable: Series,
    surface_id: str,
    context_block: str,
    validation_periods: tuple[str, ...],
    source_note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period in validation_periods:
        period_mask = frame["period"].eq(period)
        selected = frame.loc[period_mask & usable]
        total = int(period_mask.sum())
        events = len(selected)
        coins = int(selected["pair"].nunique()) if events else 0
        passed = events >= MINIMUM_EVENTS_PER_PERIOD and coins >= MINIMUM_COINS_PER_PERIOD
        rows.append(
            {
                "surface_id": surface_id,
                "cohort": str(frame["cohort"].iloc[0]) if len(frame) else "",
                "context_block": context_block,
                "period": period,
                "independent_events": total,
                "timestamp_safe_ready_events": events,
                "ready_event_fraction": events / total if total else 0.0,
                "ready_coins": coins,
                "minimum_events": MINIMUM_EVENTS_PER_PERIOD,
                "minimum_coins": MINIMUM_COINS_PER_PERIOD,
                "period_gate_passed": bool(passed),
                "source_note": source_note,
                "reaction_outcomes_opened": False,
            }
        )
    return rows


def block_decisions(overlap: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (surface_id, context_block), group in overlap.groupby(
        ["surface_id", "context_block"], sort=True, observed=True
    ):
        passed = bool(group["period_gate_passed"].all()) and len(group) == 2
        rows.append(
            {
                "surface_id": surface_id,
                "cohort": str(group["cohort"].iloc[0]),
                "context_block": context_block,
                "periods": len(group),
                "minimum_ready_events_any_period": int(
                    group["timestamp_safe_ready_events"].min()
                ),
                "minimum_ready_coins_any_period": int(group["ready_coins"].min()),
                "passed_both_periods": passed,
                "classification": (
                    "open_low_dimensional_conditioning_outcomes"
                    if passed
                    else "park_without_outcomes_insufficient_common_support"
                ),
            }
        )
    return DataFrame(rows)


def artifact_record(path: Path) -> dict[str, Any]:
    validate_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
