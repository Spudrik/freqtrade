"""Freeze simple macro-surprise direction routes before reading crypto outcomes."""

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

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as cpi_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_macro_expectation_actual_freeze as actual_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_freeze as prior_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
PARENT_RESULT_PATH = actual_frozen.RESULT_PATH
PARENT_FREEZE_PATH = actual_frozen.FREEZE_PATH
PARENT_CATALOG_PATH = actual_frozen.CATALOG_PATH
CPI_RESULT_PATH = cpi_frozen.OUTPUT_ROOT / "cpi_freeze_result.json"
CPI_CATALOG_PATH = cpi_frozen.OUTPUT_ROOT / "cpi_release_catalog.csv"
PRIOR_RESULT_PATH = prior_frozen.RESULT_PATH
PRIOR_CATALOG_PATH = prior_frozen.CATALOG_PATH

OUTPUT_ROOT = actual_frozen.OUTPUT_ROOT / "direction_batch_20260911a"
CATALOG_PATH = OUTPUT_ROOT / "macro_expectation_direction_catalog.csv"
COUNTS_PATH = OUTPUT_ROOT / "macro_expectation_direction_counts.csv"
FREEZE_PATH = OUTPUT_ROOT / "macro_expectation_direction_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "macro_expectation_direction_freeze_result.json"

PARTITIONS = (
    "development_source_2022_2024",
    "later_source_2025_2026",
)
ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
HORIZONS_MINUTES = (5, 15, 30, 60)
MINIMUM_EVENTS_PER_PARTITION = 6
MINIMUM_OVERALL_ACCURACY = 0.55
STRONG_ACCURACY_TARGET = 0.65
MINIMUM_VALIDATION_ACCURACY = 0.55
MINIMUM_CONTROL_LIFT = 0.03
CONTROL_WEEKS = 26
CONTROL_COUNT = 12
MINIMUM_CONTROLS = 10
CONTROL_EVENT_EXCLUSION_HOURS = 24
PERMUTATION_ITERATIONS = 2_000
PERMUTATION_SEED = 20260911


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def verify_result_artifact(
    result_path: Path,
    *,
    expected_status: str,
    artifact_name: str,
    artifact_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Parent result is not terminal: {result_path}")
    expected = result["artifacts"][artifact_name]["sha256"]
    if expected != g0.sha256_file(artifact_path):
        raise ValueError(f"Parent artifact changed: {artifact_path}")
    return result


def load_sources() -> tuple[DataFrame, DataFrame, DataFrame, Mapping[str, Any]]:
    parent_result = verify_result_artifact(
        PARENT_RESULT_PATH,
        expected_status="completed_macro_expectation_actual_freeze",
        artifact_name="catalog",
        artifact_path=PARENT_CATALOG_PATH,
    )
    verify_result_artifact(
        PARENT_RESULT_PATH,
        expected_status="completed_macro_expectation_actual_freeze",
        artifact_name="freeze",
        artifact_path=PARENT_FREEZE_PATH,
    )
    parent_freeze = json.loads(PARENT_FREEZE_PATH.read_text(encoding="utf-8"))
    if parent_freeze.get("crypto_outcomes_read") or parent_freeze.get("profit_used"):
        raise ValueError("Expectation-actual parent unexpectedly contains market outcomes")

    verify_result_artifact(
        CPI_RESULT_PATH,
        expected_status="completed_cpi_family_freeze",
        artifact_name="catalog",
        artifact_path=CPI_CATALOG_PATH,
    )
    verify_result_artifact(
        PRIOR_RESULT_PATH,
        expected_status="completed_scheduled_macro_direction_freeze",
        artifact_name="catalog",
        artifact_path=PRIOR_CATALOG_PATH,
    )

    actual = pd.read_csv(PARENT_CATALOG_PATH)
    actual["market_close_time_utc"] = pd.to_datetime(
        actual["market_close_time_utc"], utc=True, format="mixed"
    )
    actual["official_anchor_utc"] = pd.to_datetime(
        actual["official_anchor_utc"], utc=True, format="mixed"
    )
    actual["direction_test_eligible"] = actual["direction_test_eligible"].map(
        actual_frozen.strict_bool
    )
    cpi = pd.read_csv(CPI_CATALOG_PATH)
    prior = pd.read_csv(PRIOR_CATALOG_PATH)
    return actual, cpi, prior, parent_result


def sign(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) == 0:
        return 0
    return 1 if float(numeric) > 0 else -1


def proxy_maps(
    cpi: DataFrame, prior: DataFrame
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    cpi_proxy: dict[str, dict[str, int]] = {}
    for row in cpi.to_dict(orient="records"):
        headline = -sign(row.get("temperature_headline_mom"))
        core = -sign(row.get("temperature_core_mom"))
        cpi_proxy[str(row["event_id"])] = {
            "headline": headline,
            "core": core,
            "agreement": headline if headline != 0 and headline == core else 0,
        }
    jobs_proxy = {
        str(row["event_id"]): int(row["predicted_crypto_direction"])
        for row in prior.loc[prior["event_family"].eq("employment")].to_dict(
            orient="records"
        )
    }
    return cpi_proxy, jobs_proxy


def canonical_family_events(actual: DataFrame, family: str) -> DataFrame:
    selected = actual[
        actual["family"].eq(family) & actual["direction_test_eligible"]
    ].copy()
    duplicate = selected.duplicated(["official_event_id", "component"])
    if duplicate.any():
        raise ValueError(f"Duplicate component rows exist in {family}")
    records: list[dict[str, Any]] = []
    for official_event_id, group in selected.groupby("official_event_id", sort=False):
        if group["source_block"].nunique() != 1 or group["official_anchor_utc"].nunique() != 1:
            raise ValueError(f"Official event metadata conflicts for {official_event_id}")
        record: dict[str, Any] = {
            "expectation_episode_id": f"{family}_{official_event_id}",
            "source_block": str(group["source_block"].iloc[0]),
            "official_event_id": str(official_event_id),
            "official_anchor_utc": group["official_anchor_utc"].iloc[0],
            "maximum_expectation_lead_minutes": float(
                group["expectation_lead_minutes"].max()
            ),
        }
        for row in group.to_dict(orient="records"):
            component = str(row["component"])
            record[component] = sign(row["surprise_sign"])
            record[f"{component}_close_time"] = pd.Timestamp(
                row["market_close_time_utc"]
            )
            record[f"{component}_lead_minutes"] = float(
                row["expectation_lead_minutes"]
            )
            record[f"{component}_source_episode_id"] = str(
                row["expectation_episode_id"]
            )
        records.append(record)
    return DataFrame.from_records(records)


def route_row(
    row: Mapping[str, Any],
    *,
    route_id: str,
    family: str,
    predicted_direction: int,
    orientation_policy: str,
    source_detail: str,
    previous_proxy_direction: int,
    expectation_lead_minutes: float | None = None,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "family": family,
        "expectation_episode_id": row["expectation_episode_id"],
        "source_block": row["source_block"],
        "official_event_id": row["official_event_id"],
        "anchor_utc": row["official_anchor_utc"],
        "predicted_base_direction": int(predicted_direction),
        "orientation_policy": orientation_policy,
        "source_detail": source_detail,
        "previous_release_proxy_direction": int(previous_proxy_direction),
        "maximum_expectation_lead_minutes": float(
            row["maximum_expectation_lead_minutes"]
            if expectation_lead_minutes is None
            else expectation_lead_minutes
        ),
    }


def build_cpi_routes(
    cpi_rows: DataFrame, proxy: Mapping[str, Mapping[str, int]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in cpi_rows.to_dict(orient="records"):
        event_id = str(row["official_event_id"])
        headline = sign(row.get("headline_cpi"))
        core = sign(row.get("core_cpi"))
        if headline:
            records.append(
                route_row(
                    row,
                    route_id="cpi_headline_surprise",
                    family="consumer_inflation",
                    predicted_direction=-headline,
                    orientation_policy="fixed_inverse_rate_pressure",
                    source_detail="headline actual minus market-implied expectation",
                    previous_proxy_direction=proxy[event_id]["headline"],
                    expectation_lead_minutes=float(
                        row["headline_cpi_lead_minutes"]
                    ),
                )
            )
        if core:
            records.append(
                route_row(
                    row,
                    route_id="cpi_core_surprise",
                    family="consumer_inflation",
                    predicted_direction=-core,
                    orientation_policy="fixed_inverse_rate_pressure",
                    source_detail="core actual minus market-implied expectation",
                    previous_proxy_direction=proxy[event_id]["core"],
                    expectation_lead_minutes=float(row["core_cpi_lead_minutes"]),
                )
            )
        common_cutoff = False
        if headline != 0 and core != 0:
            cutoff_span = abs(
                pd.Timestamp(row["headline_cpi_close_time"])
                - pd.Timestamp(row["core_cpi_close_time"])
            )
            common_cutoff = cutoff_span <= pd.Timedelta(minutes=1)
        if headline == core and headline != 0 and common_cutoff:
            records.append(
                route_row(
                    row,
                    route_id="cpi_headline_core_agreement",
                    family="consumer_inflation",
                    predicted_direction=-headline,
                    orientation_policy="fixed_inverse_rate_pressure",
                    source_detail="headline and core surprises share one sign",
                    previous_proxy_direction=proxy[event_id]["agreement"],
                )
            )
    return records


def build_jobs_routes(jobs_rows: DataFrame, proxy: Mapping[str, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in jobs_rows.to_dict(orient="records"):
        payroll_strength = sign(row.get("nonfarm_payrolls"))
        unemployment_strength = -sign(row.get("unemployment_rate"))
        event_id = str(row["official_event_id"])
        old_proxy = int(proxy.get(event_id, 0))
        component_strengths: list[tuple[str, int, pd.Timestamp]] = []
        if payroll_strength:
            component_strengths.append(
                (
                    "payroll",
                    payroll_strength,
                    pd.Timestamp(row["nonfarm_payrolls_close_time"]),
                )
            )
        if unemployment_strength:
            component_strengths.append(
                (
                    "unemployment",
                    unemployment_strength,
                    pd.Timestamp(row["unemployment_rate_close_time"]),
                )
            )

        common_cutoff = False
        if len(component_strengths) == 2:
            common_cutoff = (
                abs(component_strengths[0][2] - component_strengths[1][2])
                <= pd.Timedelta(minutes=1)
            )
        if len(component_strengths) == 2 and not common_cutoff:
            earliest = min(component_strengths, key=lambda value: value[2])
            balance = earliest[1]
            balance_detail = f"earlier {earliest[0]} cutoff only"
        else:
            balance = sign(sum(value[1] for value in component_strengths))
            balance_detail = (
                "common-cutoff component balance"
                if len(component_strengths) == 2
                else "only one component available"
            )
        if balance:
            records.append(
                route_row(
                    row,
                    route_id="jobs_available_component_balance",
                    family="jobs_report",
                    predicted_direction=-balance,
                    orientation_policy="learn_rate_or_growth_on_development_only",
                    source_detail=(
                        "payroll strength and inverse unemployment surprise; use one "
                        "available component or agreement, abstain on conflict; "
                        f"{balance_detail}"
                    ),
                    previous_proxy_direction=old_proxy,
                )
            )
        if (
            common_cutoff
            and payroll_strength != 0
            and payroll_strength == unemployment_strength
        ):
            records.append(
                route_row(
                    row,
                    route_id="jobs_two_component_agreement",
                    family="jobs_report",
                    predicted_direction=-payroll_strength,
                    orientation_policy="learn_rate_or_growth_on_development_only",
                    source_detail=(
                        "payroll and inverse unemployment surprises both imply the "
                        "same jobs-strength direction"
                    ),
                    previous_proxy_direction=old_proxy,
                )
            )
    return records


def build_fed_diagnostic(actual: DataFrame) -> list[dict[str, Any]]:
    selected = actual[
        actual["component"].eq("federal_reserve_decision")
        & actual["direction_test_eligible"]
        & actual["signed_surprise_available"].map(actual_frozen.strict_bool)
    ]
    records: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        records.append(
            {
                "route_id": "fed_decision_surprise",
                "family": "central_bank_decision",
                "expectation_episode_id": row["expectation_episode_id"],
                "source_block": row["source_block"],
                "official_event_id": row["official_event_id"],
                "anchor_utc": row["official_anchor_utc"],
                "predicted_base_direction": int(row["provisional_crypto_direction"]),
                "orientation_policy": "fixed_inverse_ordered_decision_surprise",
                "source_detail": "actual decision category minus top expected category",
                "previous_release_proxy_direction": 0,
                "maximum_expectation_lead_minutes": float(
                    row["expectation_lead_minutes"]
                ),
            }
        )
    return records


def route_counts(catalog: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for route_id, group in catalog.groupby("route_id", sort=True):
        by_partition = group.groupby("source_block")["expectation_episode_id"].nunique()
        signs_by_partition = group.groupby("source_block")[
            "predicted_base_direction"
        ].nunique()
        coverage = all(
            int(by_partition.get(partition, 0)) >= MINIMUM_EVENTS_PER_PARTITION
            and int(signs_by_partition.get(partition, 0)) >= 2
            for partition in PARTITIONS
        )
        rows.append(
            {
                "route_id": route_id,
                "family": str(group["family"].iloc[0]),
                "development_events": int(by_partition.get(PARTITIONS[0], 0)),
                "later_events": int(by_partition.get(PARTITIONS[1], 0)),
                "development_direction_count": int(
                    signs_by_partition.get(PARTITIONS[0], 0)
                ),
                "later_direction_count": int(signs_by_partition.get(PARTITIONS[1], 0)),
                "historical_direction_test_eligible": coverage,
                "source_decision": (
                    "eligible_for_frozen_btc_eth_direction_test"
                    if coverage
                    else "source_coverage_or_variation_limited"
                ),
            }
        )
    return DataFrame.from_records(rows)


def build_catalog(
    actual: DataFrame,
    cpi: DataFrame,
    prior: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    cpi_proxy, jobs_proxy = proxy_maps(cpi, prior)
    cpi_rows = canonical_family_events(actual, "consumer_inflation")
    jobs_rows = canonical_family_events(actual, "jobs_report")
    records = [
        *build_cpi_routes(cpi_rows, cpi_proxy),
        *build_jobs_routes(jobs_rows, jobs_proxy),
        *build_fed_diagnostic(actual),
    ]
    catalog = DataFrame.from_records(records).sort_values(
        ["anchor_utc", "route_id"], kind="stable"
    )
    if catalog.duplicated(["route_id", "expectation_episode_id"]).any():
        raise ValueError("A route contains duplicate whole-event votes")
    if not catalog["predicted_base_direction"].isin([-1, 1]).all():
        raise ValueError("Every route row must have a non-zero base direction")
    counts = route_counts(catalog)
    eligibility = counts.set_index("route_id")[
        "historical_direction_test_eligible"
    ].to_dict()
    catalog["historical_direction_test_eligible"] = catalog["route_id"].map(
        eligibility
    )
    return catalog.reset_index(drop=True), counts


def freeze_document(
    catalog: DataFrame,
    counts: DataFrame,
    parent_result: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = counts[counts["historical_direction_test_eligible"]]
    return {
        "schema_version": 1,
        "status": "frozen_macro_expectation_direction_before_crypto_outcomes",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "independent_confirmation": False,
        "prior_related_market_results_known": True,
        "plain_question": (
            "Does the difference between a released macro figure and the market's "
            "pre-release expectation provide repeatable 5-60 minute BTC/ETH direction?"
        ),
        "route_order": [
            "cpi_headline_surprise",
            "cpi_core_surprise",
            "cpi_headline_core_agreement",
            "jobs_available_component_balance",
            "jobs_two_component_agreement",
            "fed_decision_surprise",
        ],
        "partitions": list(PARTITIONS),
        "assets": list(ASSETS),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "eligible_routes": eligible["route_id"].tolist(),
        "source_limited_routes": counts.loc[
            ~counts["historical_direction_test_eligible"], "route_id"
        ].tolist(),
        "retention_rule": {
            "minimum_events_each_partition": MINIMUM_EVENTS_PER_PARTITION,
            "minimum_overall_accuracy": MINIMUM_OVERALL_ACCURACY,
            "strong_accuracy_target": STRONG_ACCURACY_TARGET,
            "minimum_later_period_accuracy": MINIMUM_VALIDATION_ACCURACY,
            "minimum_lift_over_strongest_readable_control": MINIMUM_CONTROL_LIFT,
            "both_assets_same_horizon_for_broad_lead": True,
            "whole_family_max_statistic_probability": 0.05,
        },
        "jobs_orientation_rule": (
            "Use 2022-2024 only to choose whether rate-pressure or growth/risk appetite "
            "is the better sign. Freeze that orientation, then score 2025-2026. The two "
            "opposite channels are one question, not two independent discoveries."
        ),
        "controls": [
            "development-period majority direction applied unchanged later",
            "causal pre-event return over the same duration",
            "whole-event rotated source sign within family and partition",
            "same sign at matched prior weekday-and-clock non-event periods",
        ],
        "alternate_signal_comparison": (
            "Previous-release proxy direction is compared on exactly overlapping "
            "announcements and its additional or missing coverage is reported. It is "
            "another possible signal, not a null control that can reject a complementary "
            "expectation signal merely by scoring a smaller subset."
        ),
        "method_revision_after_initial_outcome_audit": (
            "The first direct implementation compared some control accuracies from a "
            "smaller subset with signal accuracy from all events. This freeze revision "
            "requires paired event-by-event null-control lifts. The earlier direct output "
            "was overwritten and is not evidence. This repaired historical rerun remains "
            "non-independent because related outcomes were already known."
        ),
        "matched_control_rule": {
            "search_weeks": CONTROL_WEEKS,
            "controls_per_event": CONTROL_COUNT,
            "minimum_controls": MINIMUM_CONTROLS,
            "exclude_within_hours_of_any_known_macro_event": (
                CONTROL_EVENT_EXCLUSION_HOURS
            ),
        },
        "search_adjustment": {
            "method": (
                "Shuffle complete market-response episodes within family and period, "
                "keeping BTC, ETH, and all horizons together; compare the largest "
                "retained pattern anywhere in the frozen batch."
            ),
            "iterations": PERMUTATION_ITERATIONS,
            "seed": PERMUTATION_SEED,
        },
        "interpretation_guardrails": [
            "A scheduled announcement can drive activity even when its direction call fails.",
            "Volume before or after the release is context or confirmation, not a rival cause.",
            "Headline and core CPI remain one announcement episode.",
            "Payroll and unemployment remain one jobs-report episode.",
            (
                "A combined route uses component forecasts only when their cutoffs are "
                "within one minute; otherwise only the earlier component may vote."
            ),
            "Conflicting jobs components cause abstention rather than a forced direction.",
            "A source-limited route is not evidence that its event family has no effect.",
            "No result is assumed universal across market backgrounds.",
            (
                "Every pass/fail control must be compared with the signal on exactly the "
                "same announcements."
            ),
        ],
        "route_counts": counts.to_dict(orient="records"),
        "catalog_rows": len(catalog),
        "whole_episodes": int(catalog["expectation_episode_id"].nunique()),
        "parents": {
            "expectation_actual_result": parent_result,
            "expectation_actual_catalog": artifact(PARENT_CATALOG_PATH),
            "cpi_previous_release_catalog": artifact(CPI_CATALOG_PATH),
            "jobs_previous_release_catalog": artifact(PRIOR_CATALOG_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_macro_expectation_direction_freeze":
            raise ValueError("Existing direction freeze is invalid")
        for value in result["artifacts"].values():
            path = Path(value["path"])
            if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
                raise ValueError(f"Existing direction-freeze artifact changed: {path}")
        return result

    actual, cpi, prior, parent_result = load_sources()
    catalog, counts = build_catalog(actual, cpi, prior)
    freeze = freeze_document(catalog, counts, parent_result)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalog, CATALOG_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_macro_expectation_direction_freeze",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": False,
        "profit_used": False,
        "eligible_routes": freeze["eligible_routes"],
        "source_limited_routes": freeze["source_limited_routes"],
        "artifacts": {
            "catalog": artifact(CATALOG_PATH),
            "counts": artifact(COUNTS_PATH),
            "freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "crypto_outcomes_will_be_read": False,
                    "output_root": str(OUTPUT_ROOT),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
