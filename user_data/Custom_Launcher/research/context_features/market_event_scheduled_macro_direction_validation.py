"""Search-adjust the scheduled-macro direction candidates by whole releases.

This is a result-spawned robustness branch, not independent confirmation.  Event
signs are reshuffled within each family and whole-period partition while each sign
stays joined across BTC, ETH, and all four horizons.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = direct.OUTPUT_ROOT / "validation_20260905a"
FREEZE_PATH = OUTPUT_ROOT / "scheduled_macro_validation_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "scheduled_macro_validation_result.json"
NULL_PATH = OUTPUT_ROOT / "whole_release_null_statistics.parquet"
SUMMARY_PATH = OUTPUT_ROOT / "whole_release_validation_summary.csv"
FAMILY_PATH = OUTPUT_ROOT / "scheduled_macro_validated_families.csv"
REPORT_PATH = OUTPUT_ROOT / "scheduled_macro_validation_plain_review.md"

ITERATIONS = 2000
SEED = 20260905
MAX_FAMILYWISE_PROBABILITY = 0.05


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_parent_result() -> dict[str, Any]:
    result = json.loads(direct.RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("status") != (
        "completed_provisional_candidates_require_familywide_validation"
    ):
        raise ValueError("Parent direct result has no provisional candidates")
    if result.get("profit_used"):
        raise ValueError("Parent direct result unexpectedly used profit")
    return result


def observed_statistics(decisions: DataFrame) -> dict[str, int]:
    candidates = decisions.loc[decisions["candidate"].astype(bool)]
    family_counts = candidates.groupby("event_family").size()
    shared_counts: list[int] = []
    for _, family in candidates.groupby("event_family", sort=False):
        btc = set(
            family.loc[
                family["pair"].eq(frozen.ASSETS[0]), "horizon_minutes"
            ].astype(int)
        )
        eth = set(
            family.loc[
                family["pair"].eq(frozen.ASSETS[1]), "horizon_minutes"
            ].astype(int)
        )
        shared_counts.append(len(btc & eth))
    return {
        "candidate_cells": len(candidates),
        "max_candidate_cells_one_family": (
            int(family_counts.max()) if len(family_counts) else 0
        ),
        "max_shared_horizons_one_family": max(shared_counts, default=0),
    }


def build_branch_freeze(parent: Mapping[str, Any], decisions: DataFrame) -> dict[str, Any]:
    candidates = decisions.loc[decisions["candidate"].astype(bool)]
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "status": "frozen_scheduled_macro_validation_before_reshuffle_outcomes",
        "created_at_utc": g0.utc_now(),
        "parent_result": artifact(direct.RESULT_PATH),
        "parent_outcomes_already_reviewed": True,
        "independent_confirmation": False,
        "profit_used": False,
        "candidate_cells": int(parent["candidate_cells"]),
        "candidate_families": sorted(candidates["event_family"].unique()),
        "observed_statistics": observed_statistics(decisions),
        "method": (
            "Shuffle complete event signs within each event family and whole-period "
            "partition. Keep each shuffled sign joined across both assets and every "
            "horizon, rerun the exact frozen thresholds and controls, and compare the "
            "largest result selected from the complete five-family batch."
        ),
        "iterations": ITERATIONS,
        "seed": SEED,
        "maximum_familywise_probability": MAX_FAMILYWISE_PROBABILITY,
        "retention_rule": (
            "A two-market family lead survives this branch only if both its candidate-"
            "cell count and shared-horizon count are at least as unusual as the largest "
            "family selected anywhere in the batch, each at probability no more than 5%."
        ),
        "boundary": (
            "Survival keeps one historical direction lead for expectation and future-"
            "release confirmation. It is not profit evidence or a trading rule."
        ),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def freeze_branch(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        document = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if document.get("status") != (
            "frozen_scheduled_macro_validation_before_reshuffle_outcomes"
        ):
            raise ValueError("Existing scheduled-macro validation freeze is invalid")
        return document
    parent = load_parent_result()
    decisions = pd.read_csv(direct.DECISION_PATH)
    document = build_branch_freeze(parent, decisions)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(document, FREEZE_PATH)
    return document


def load_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != (
        "frozen_scheduled_macro_validation_before_reshuffle_outcomes"
    ):
        raise ValueError("Scheduled-macro validation was not frozen")
    parent = load_parent_result()
    if freeze["parent_result"]["sha256"] != g0.sha256_file(direct.RESULT_PATH):
        raise ValueError("Parent direct result changed after validation freeze")
    rows = pd.read_parquet(direct.OUTCOME_PATH)
    decisions = pd.read_csv(direct.DECISION_PATH)
    catalog = pd.read_csv(frozen.CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    if parent["artifacts"]["direction_rows"]["sha256"] != g0.sha256_file(
        direct.OUTCOME_PATH
    ):
        raise ValueError("Parent direction rows changed after direct review")
    return freeze, rows, decisions, catalog


def permuted_event_signs(catalog: DataFrame, rng: np.random.Generator) -> dict[str, int]:
    events = catalog[
        [
            "event_id",
            "event_family",
            "whole_event_partition",
            "anchor_utc",
            "predicted_crypto_direction",
        ]
    ].sort_values(
        ["event_family", "whole_event_partition", "anchor_utc"], kind="stable"
    )
    mapping: dict[str, int] = {}
    for _, group in events.groupby(
        ["event_family", "whole_event_partition"], sort=False
    ):
        shuffled = rng.permutation(group["predicted_crypto_direction"].to_numpy(int))
        mapping.update(
            {
                str(event_id): int(value)
                for event_id, value in zip(group["event_id"], shuffled, strict=True)
            }
        )
    return mapping


def apply_permuted_signs(rows: DataFrame, signs: Mapping[str, int]) -> DataFrame:
    result = rows.copy()
    original_sign = result["predicted_direction"].astype(int)
    result["predicted_direction"] = result["event_id"].map(signs)
    if result["predicted_direction"].isna().any():
        raise ValueError("A reshuffled sign is missing for at least one event")
    result["predicted_direction"] = result["predicted_direction"].astype(int)
    ordinary = pd.to_numeric(result["ordinary_control_accuracy"], errors="coerce")
    flipped = result["predicted_direction"].ne(original_sign)
    result["ordinary_control_accuracy"] = ordinary.where(~flipped, 1.0 - ordinary)
    result["direction_hit"] = [
        direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["predicted_direction"], result["actual_direction"], strict=True
        )
    ]
    result = result.drop(
        columns=[
            "rotated_predicted_direction",
            "rotated_hit",
            "development_majority_direction",
            "development_majority_hit",
        ],
        errors="ignore",
    )
    return direct.add_rotated_and_majority_controls(result)


def null_distribution(
    rows: DataFrame,
    catalog: DataFrame,
    *,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, int]] = []
    for iteration in range(iterations):
        signs = permuted_event_signs(catalog, rng)
        permuted = apply_permuted_signs(rows, signs)
        summary = direct.direction_summaries(permuted)
        decisions = direct.route_decisions(summary)
        records.append({"iteration": iteration, **observed_statistics(decisions)})
    return DataFrame.from_records(records)


def validation_summary(
    observed: Mapping[str, int], null: DataFrame
) -> DataFrame:
    records: list[dict[str, Any]] = []
    for metric, observed_value in observed.items():
        values = pd.to_numeric(null[metric], errors="coerce")
        exceedances = int(values.ge(observed_value).sum())
        records.append(
            {
                "metric": metric,
                "observed": observed_value,
                "null_median": float(values.median()),
                "null_95th_percentile": float(values.quantile(0.95)),
                "familywise_probability": (exceedances + 1) / (len(values) + 1),
            }
        )
    return DataFrame.from_records(records)


def validated_families(
    decisions: DataFrame, null: DataFrame
) -> DataFrame:
    candidates = decisions.loc[decisions["candidate"].astype(bool)]
    records: list[dict[str, Any]] = []
    for family_name, family in candidates.groupby("event_family", sort=True):
        btc = set(
            family.loc[
                family["pair"].eq(frozen.ASSETS[0]), "horizon_minutes"
            ].astype(int)
        )
        eth = set(
            family.loc[
                family["pair"].eq(frozen.ASSETS[1]), "horizon_minutes"
            ].astype(int)
        )
        candidate_cells = len(family)
        shared_horizons = len(btc & eth)
        cell_probability = (
            int(null["max_candidate_cells_one_family"].ge(candidate_cells).sum()) + 1
        ) / (len(null) + 1)
        shared_probability = (
            int(
                null["max_shared_horizons_one_family"].ge(shared_horizons).sum()
            )
            + 1
        ) / (len(null) + 1)
        two_market = shared_horizons > 0
        survives = bool(
            two_market
            and cell_probability <= MAX_FAMILYWISE_PROBABILITY
            and shared_probability <= MAX_FAMILYWISE_PROBABILITY
        )
        records.append(
            {
                "event_family": family_name,
                "candidate_cells": candidate_cells,
                "shared_btc_eth_horizons": shared_horizons,
                "cell_count_familywise_probability": cell_probability,
                "shared_horizon_familywise_probability": shared_probability,
                "decision": (
                    "retained_historical_lead_for_expectation_and_future_confirmation"
                    if survives
                    else "parked_after_whole_release_search_adjustment"
                ),
                "independent_confirmation": False,
                "profit_used": False,
            }
        )
    return DataFrame.from_records(records)


def render_report(summary: DataFrame, families: DataFrame) -> str:
    lines = [
        "# Scheduled Macro Direction Validation",
        "",
        "This branch reshuffles complete releases within each family and historical "
        "period. One shuffled sign remains joined across Bitcoin, Ethereum, and every "
        "time window, so overlapping observations are not counted as independent bets.",
        "",
        "## Search-adjusted results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: observed `{row.observed}`, chance of at least this "
            f"large `{row.familywise_probability:.2%}`"
        )
    lines.extend(["", "## Family decisions", ""])
    for row in families.itertuples(index=False):
        lines.append(
            f"- `{row.event_family}`: `{row.decision}` "
            f"(cells `{row.candidate_cells}`, shared BTC/ETH windows "
            f"`{row.shared_btc_eth_horizons}`)"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A retained family remains one historical correlation lead. It still needs "
            "actual-versus-expected release data and genuinely later releases. This is "
            "not profit evidence or a trading rule.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    *, overwrite: bool = False, iterations: int = ITERATIONS, seed: int = SEED
) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_scheduled_macro_validation":
            raise ValueError("Existing scheduled-macro validation result is invalid")
        return result
    freeze, rows, decisions, catalog = load_inputs()
    if iterations != int(freeze["iterations"]) or seed != int(freeze["seed"]):
        raise ValueError("Execution iterations and seed must match the frozen branch")
    observed = observed_statistics(decisions)
    if observed != freeze["observed_statistics"]:
        raise ValueError("Observed candidate statistics changed after branch freeze")
    null = null_distribution(rows, catalog, iterations=iterations, seed=seed)
    summary = validation_summary(observed, null)
    families = validated_families(decisions, null)
    g0.atomic_write_parquet(null, NULL_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(families, FAMILY_PATH)
    REPORT_PATH.write_text(
        render_report(summary, families), encoding="utf-8", newline="\n"
    )
    result = {
        "schema_version": 1,
        "status": "completed_scheduled_macro_validation",
        "created_at_utc": g0.utc_now(),
        "iterations": iterations,
        "seed": seed,
        "independent_confirmation": False,
        "profit_used": False,
        "retained_families": families.loc[
            families["decision"].str.startswith("retained"), "event_family"
        ].tolist(),
        "freeze_contract": artifact(FREEZE_PATH),
        "parent_result": artifact(direct.RESULT_PATH),
        "artifacts": {
            "null_statistics": artifact(NULL_PATH),
            "validation_summary": artifact(SUMMARY_PATH),
            "family_decisions": artifact(FAMILY_PATH),
            "plain_review": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.freeze:
        print(json.dumps(freeze_branch(overwrite=args.overwrite), indent=2))
        return os.EX_OK
    if args.execute:
        print(
            json.dumps(
                execute(
                    overwrite=args.overwrite,
                    iterations=args.iterations,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return os.EX_OK
    print(
        json.dumps(
            {
                "status": "ready_not_executed",
                "next": "run --freeze before --execute",
                "iterations": ITERATIONS,
                "seed": SEED,
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
        )
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
