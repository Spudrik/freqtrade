from __future__ import annotations

# Bound numerical libraries before importing pandas/pyarrow.
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
import json
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation6_review" / "g7_frozen_pairwise_batch.json"
G6_RECORD_ROOT = (
    OUTPUT_ROOT / "generation6_branches" / "g6_freqai_source_ladders"
)
G6_RUNS = {
    "normal": "g6_freqai_full_normal_20260821a",
    "meme": "g6_freqai_full_meme_20260821a",
}
RECORD_ROOT = OUTPUT_ROOT / "generation7_branches" / "g7_preflight"
DEFAULT_RUN_ID = "g7_outcome_blind_preflight_20260821a"
MAX_WORKERS = 4
VALIDATION_PERIODS = {
    "normal": ("validation_early", "validation_late"),
    "meme": ("meme_validation_early", "meme_validation_late"),
}
DEVELOPMENT_PERIOD = {"normal": "development", "meme": "meme_development"}

LEVEL_DEPENDENCY_GROUPS = {
    "confirmed_swing": "confirmed_swing",
    "generic_prior_range": "prior_range",
    "generic_round_number": "round_number",
    "tlv2_forecast_zone": "tlv2",
    "tlv2_ranked": "tlv2",
    "volume_profile_explicit_prior": "volume_profile",
    "volume_profile_nodes": "volume_profile",
    "volume_profile_settled": "volume_profile",
}


@dataclass(frozen=True)
class BranchSpec:
    cohorts: tuple[str, ...]
    required_blocks: tuple[str, ...]
    scope: str
    pair_filter: Callable[[str], bool]


def all_pairs(_: str) -> bool:
    return True


def btc_and_established_altcoins(pair: str) -> bool:
    return pair == "BTC/USDT:USDT" or pair in g6.GROUPS["established_altcoins"]


BRANCH_SPECS: dict[str, BranchSpec] = {
    "g7a_local_participation_and_btc_activity": BranchSpec(
        ("normal", "meme"),
        ("level", "ohlcv_volume_pressure", "cross_market_btc"),
        "all",
        all_pairs,
    ),
    "g7b_local_participation_and_volatility": BranchSpec(
        ("normal", "meme"),
        ("level", "ohlcv_volume_pressure", "ohlcv_volatility_range"),
        "all",
        all_pairs,
    ),
    "g7c_local_participation_and_trend_momentum": BranchSpec(
        ("normal", "meme"),
        ("level", "ohlcv_volume_pressure", "ohlcv_trend_momentum"),
        "all",
        all_pairs,
    ),
    "g7d_cross_timeframe_agreement_and_local_participation": BranchSpec(
        ("normal", "meme"),
        (
            "level",
            "timeframe_4h",
            "timeframe_8h",
            "timeframe_1d",
            "ohlcv_volume_pressure",
        ),
        "different_mechanism_cross_timeframe",
        all_pairs,
    ),
    "g7e_prior_range_level_and_local_participation": BranchSpec(
        ("normal", "meme"),
        ("level", "ohlcv_volume_pressure"),
        "generic_prior_range",
        all_pairs,
    ),
    "g7f_prior_volume_profile_area_and_compression": BranchSpec(
        ("normal", "meme"),
        ("level", "ohlcv_volatility_range"),
        "volume_profile_explicit_prior_4h",
        all_pairs,
    ),
    "g7g_independent_level_family_cluster": BranchSpec(
        ("normal", "meme"),
        ("level",),
        "independent_level_family_cluster",
        all_pairs,
    ),
    "g7h_meme_eth_activity_and_local_participation": BranchSpec(
        ("meme",),
        ("level", "ohlcv_volume_pressure", "cross_market_eth"),
        "all",
        all_pairs,
    ),
    "g7i_orderbook_and_btc_activity_at_levels": BranchSpec(
        ("normal",),
        ("level", "orderbook_btc", "cross_market_btc"),
        "all",
        btc_and_established_altcoins,
    ),
    "g7j_gdelt_activity_and_local_participation": BranchSpec(
        ("normal", "meme"),
        ("level", "news_gdelt", "ohlcv_volume_pressure"),
        "all",
        all_pairs,
    ),
}


def g6_record_root(cohort: str) -> Path:
    return G6_RECORD_ROOT / G6_RUNS[cohort]


@cache
def cache_inventory(cohort: str) -> dict[str, dict[str, Any]]:
    cache_path = g6_record_root(cohort) / "g6_freqai_cache_inventory.json"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    records = json.loads(cache_path.read_text(encoding="utf-8"))
    inventory = {str(item["pair"]): item for item in records}
    if len(inventory) != len(records):
        raise ValueError(f"Duplicate pairs in Generation 6 inventory: {cache_path}")
    return inventory


def load_frozen_batch() -> dict[str, Any]:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    batch = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    if batch.get("status") != "frozen_before_generation7_reaction_outcomes":
        raise ValueError("Generation 7 is not frozen before outcomes.")
    branch_ids = {str(item["id"]) for item in batch.get("branches", [])}
    if branch_ids != set(BRANCH_SPECS):
        raise ValueError(
            f"Generation 7 branch/spec drift: missing={branch_ids - set(BRANCH_SPECS)}, "
            f"extra={set(BRANCH_SPECS) - branch_ids}"
        )
    boundary = batch.get("research_boundary", {})
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Profit optimization is not permitted in Generation 7.")
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Direction prediction is not permitted in Generation 7.")
    return batch


def feature_columns_for_blocks(schema_names: Sequence[str], blocks: Sequence[str]) -> list[str]:
    columns = ["date"]
    for block in blocks:
        for variant in (block, f"{block}_placebo"):
            prefix = f"{variant}__"
            selected = [name for name in schema_names if name.startswith(prefix)]
            if not selected:
                raise ValueError(f"Missing feature columns for block {variant!r}")
            columns.extend(selected)
    return list(dict.fromkeys(columns))


def complete_block(frame: DataFrame, block: str) -> Series:
    columns = [column for column in frame if column.startswith(f"{block}__")]
    if not columns:
        raise ValueError(f"No columns for required block {block!r}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def metadata_by_date(evaluation_path: Path) -> DataFrame:
    columns = ["date", "level_family", "source_timeframe"]
    frame = pd.read_parquet(evaluation_path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date", "level_family", "source_timeframe"])
    return frame.drop_duplicates().reset_index(drop=True)


def independent_dependency_group_count(metadata: DataFrame) -> DataFrame:
    frame = metadata.copy()
    frame["dependency_group"] = frame["level_family"].map(LEVEL_DEPENDENCY_GROUPS)
    if frame["dependency_group"].isna().any():
        unknown = sorted(frame.loc[frame["dependency_group"].isna(), "level_family"].unique())
        raise ValueError(f"Unknown level dependency families: {unknown}")
    return (
        frame.groupby("date", as_index=False, sort=True)
        .agg(independent_dependency_groups=("dependency_group", "nunique"))
        .reset_index(drop=True)
    )


def scope_dates(
    scope: str,
    *,
    features: DataFrame,
    evaluation_path: Path,
) -> Series:
    if scope == "all":
        return pd.Series(True, index=features.index)
    if scope == "different_mechanism_cross_timeframe":
        columns = [
            f"timeframe_{timeframe}__different_mechanism_agreement"
            for timeframe in ("4h", "8h", "1d")
        ]
        return features[columns].fillna(0.0).gt(0.0).any(axis=1)
    if scope == "generic_prior_range":
        return features["level__family_generic_prior_range_count"].fillna(0.0).gt(0.0)
    metadata = metadata_by_date(evaluation_path)
    if scope == "volume_profile_explicit_prior_4h":
        dates = metadata.loc[
            metadata["level_family"].eq("volume_profile_explicit_prior")
            & metadata["source_timeframe"].eq("4h"),
            "date",
        ]
        return features["date"].isin(dates)
    if scope == "independent_level_family_cluster":
        counts = independent_dependency_group_count(metadata)
        eligible_dates = counts.loc[
            counts["independent_dependency_groups"].ge(2), "date"
        ]
        return features["date"].isin(eligible_dates)
    raise ValueError(f"Unknown Generation 7 scope {scope!r}")


def pair_inventory(cohort: str) -> tuple[str, ...]:
    return tuple(cache_inventory(cohort))


def pair_support(branch_id: str, cohort: str, pair: str) -> list[dict[str, Any]]:
    spec = BRANCH_SPECS[branch_id]
    inventory = cache_inventory(cohort)[pair]
    feature_path = Path(inventory["feature_path"])
    event_path = Path(inventory["event_path"])
    evaluation_path = Path(inventory["evaluation_path"])
    for path in (feature_path, event_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    schema_names = pq.read_schema(feature_path).names
    feature_columns = feature_columns_for_blocks(schema_names, spec.required_blocks)
    if spec.scope == "different_mechanism_cross_timeframe":
        feature_columns.extend(
            f"timeframe_{timeframe}__different_mechanism_agreement"
            for timeframe in ("4h", "8h", "1d")
        )
    if spec.scope == "generic_prior_range":
        feature_columns.append("level__family_generic_prior_range_count")
    feature_columns = list(dict.fromkeys(feature_columns))
    features = pd.read_parquet(feature_path, columns=feature_columns)
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="coerce")
    event = pd.read_parquet(event_path, columns=["date", "period"])
    event["date"] = pd.to_datetime(event["date"], utc=True, errors="coerce")
    frame = event.merge(features, on="date", how="inner", validate="one_to_one")
    eligible = scope_dates(spec.scope, features=frame, evaluation_path=evaluation_path)
    for block in spec.required_blocks:
        eligible &= complete_block(frame, block)
        eligible &= complete_block(frame, f"{block}_placebo")
    frame = frame.loc[eligible, ["date", "period"]].copy()
    rows: list[dict[str, Any]] = []
    periods = (DEVELOPMENT_PERIOD[cohort], *VALIDATION_PERIODS[cohort])
    for period in periods:
        selected = frame.loc[frame["period"].eq(period)]
        rows.append(
            {
                "branch_id": branch_id,
                "cohort": cohort,
                "pair": pair,
                "period": period,
                "eligible_independent_events": len(selected),
                "first_event_utc": (
                    selected["date"].min().isoformat() if len(selected) else None
                ),
                "last_event_utc": (
                    selected["date"].max().isoformat() if len(selected) else None
                ),
                "required_blocks": ",".join(spec.required_blocks),
                "scope": spec.scope,
                "feature_path": str(feature_path),
                "feature_sha256": g0.sha256_file(feature_path),
                "event_path": str(event_path),
                "event_sha256": g0.sha256_file(event_path),
                "evaluation_path": str(evaluation_path),
                "evaluation_sha256": g0.sha256_file(evaluation_path),
                "reaction_outcome_columns_read": False,
            }
        )
    return rows


def branch_decision(branch_id: str, cohort: str, support: DataFrame) -> dict[str, Any]:
    spec = BRANCH_SPECS[branch_id]
    periods = VALIDATION_PERIODS[cohort]
    checks: list[dict[str, Any]] = []
    for period in periods:
        selected = support.loc[
            support["branch_id"].eq(branch_id)
            & support["cohort"].eq(cohort)
            & support["period"].eq(period)
        ]
        positive = selected.loc[selected["eligible_independent_events"].gt(0)]
        checks.append(
            {
                "period": period,
                "events": int(selected["eligible_independent_events"].sum()),
                "coins": len(positive),
                "minimum_pair_events": (
                    int(positive["eligible_independent_events"].min())
                    if len(positive)
                    else 0
                ),
            }
        )
    if branch_id == "g7i_orderbook_and_btc_activity_at_levels":
        minimum_events = 50
        minimum_coins = 5
    elif cohort == "meme":
        minimum_events = 50
        minimum_coins = 5
    else:
        minimum_events = 50
        minimum_coins = 5
    supported = all(
        item["events"] >= minimum_events and item["coins"] >= minimum_coins
        for item in checks
    )
    development = support.loc[
        support["branch_id"].eq(branch_id)
        & support["cohort"].eq(cohort)
        & support["period"].eq(DEVELOPMENT_PERIOD[cohort])
    ]
    development_events = int(development["eligible_independent_events"].sum())
    development_coins = int(
        development.loc[development["eligible_independent_events"].gt(0), "pair"].nunique()
    )
    supported &= development_events >= 100 and development_coins >= minimum_coins
    return {
        "branch_id": branch_id,
        "cohort": cohort,
        "status": (
            "supported_for_generation7_models"
            if supported
            else "parked_outcome_blind_insufficient_common_support"
        ),
        "development_events": development_events,
        "development_coins": development_coins,
        "validation_checks": checks,
        "minimum_validation_events": minimum_events,
        "minimum_validation_coins": minimum_coins,
        "required_blocks": list(spec.required_blocks),
        "scope": spec.scope,
        "reaction_outcomes_opened": False,
    }


def run_preflight(run_id: str, workers: int) -> dict[str, Any]:
    batch = load_frozen_batch()
    tasks: list[tuple[str, str, str]] = []
    for branch_id, spec in BRANCH_SPECS.items():
        for cohort in spec.cohorts:
            for pair in pair_inventory(cohort):
                if spec.pair_filter(pair):
                    tasks.append((branch_id, cohort, pair))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(pair_support, branch_id, cohort, pair): (
                branch_id,
                cohort,
                pair,
            )
            for branch_id, cohort, pair in tasks
        }
        for future in as_completed(futures):
            branch_id, cohort, pair = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "branch_id": branch_id,
                        "cohort": cohort,
                        "pair": pair,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if failures:
        raise RuntimeError(f"Generation 7 outcome-blind preflight failures: {failures}")
    support = DataFrame.from_records(rows).sort_values(
        ["branch_id", "cohort", "pair", "period"]
    )
    decisions = DataFrame.from_records(
        [
            branch_decision(branch_id, cohort, support)
            for branch_id, spec in BRANCH_SPECS.items()
            for cohort in spec.cohorts
        ]
    )
    record_dir = RECORD_ROOT / run_id
    record_dir.mkdir(parents=True, exist_ok=True)
    support_path = record_dir / "g7_outcome_blind_support.csv"
    decision_path = record_dir / "g7_outcome_blind_decisions.json"
    g0.atomic_write_csv(support, support_path)
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": g0.utc_now(),
            "decisions": decisions.to_dict(orient="records"),
        },
        decision_path,
    )
    supported = decisions["status"].eq("supported_for_generation7_models")
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation7_outcome_blind_preflight",
        "frozen_batch": {
            "path": str(FROZEN_BATCH),
            "sha256": g0.sha256_file(FROZEN_BATCH),
            "branches": len(batch["branches"]),
        },
        "tasks": len(tasks),
        "support_rows": len(support),
        "branch_cohort_cells": len(decisions),
        "supported_branch_cohort_cells": int(supported.sum()),
        "parked_branch_cohort_cells": int((~supported).sum()),
        "decisions": decisions.to_dict(orient="records"),
        "integrity": {
            "reaction_outcome_columns_read": False,
            "profit_used": False,
            "direction_prediction": False,
            "all_inputs_are_terminal_generation6_caches": True,
            "scope_masks_use_only_level_identity_source_timeframe_and_precontact_features": True,
        },
        "artifacts": {
            "support": {
                "path": str(support_path),
                "sha256": g0.sha256_file(support_path),
            },
            "decisions": {
                "path": str(decision_path),
                "sha256": g0.sha256_file(decision_path),
            },
        },
        "next_action": (
            "Build and run only supported branch/cohort cells. Park unsupported cells "
            "without opening reaction outcomes, then complete the full Generation 7 portfolio "
            "before any result-driven descendant."
        ),
    }
    result_path = record_dir / "manifest.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind common-support and timestamp-integrity preflight for the "
            "frozen Generation 7 pairwise interaction portfolio."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.run_id
    ):
        raise ValueError(f"Invalid run id: {args.run_id!r}")
    result = run_preflight(args.run_id, args.workers)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
