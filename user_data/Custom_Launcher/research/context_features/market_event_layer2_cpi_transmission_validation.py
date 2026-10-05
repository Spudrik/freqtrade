"""Validate the frozen CPI leader-to-follower direction leads family-wide."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_cpi_links_direct as cpi_links,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_activity_family_validation as activity_validation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_validation_freeze as validation_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = validation_freeze.OUTPUT_ROOT / "cpi_transmission_validation_20260908a"
NULL_PATH = OUTPUT_ROOT / "cpi_transmission_family_null_maxima.parquet"
RANDOMIZATION_DECISION_PATH = OUTPUT_ROOT / "cpi_transmission_randomization_decision.csv"
COMMON_SAMPLE_PATH = OUTPUT_ROOT / "cpi_common_event_samples.parquet"
COMMON_COMPARISON_PATH = OUTPUT_ROOT / "cpi_common_event_candidate_comparison.csv"
COMMON_DECISION_PATH = OUTPUT_ROOT / "cpi_common_event_primary_decision.csv"
REPORT_PATH = OUTPUT_ROOT / "cpi_transmission_validation_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "cpi_transmission_validation_result.json"

ROUTE_COLUMNS = (
    "leader",
    "leader_window_minutes",
    "follower_end_minutes",
    "follower",
)
PRIMARY_ROUTE = (
    "btc_eth_agreement",
    15,
    60,
    "established_group_median",
)
COMMON_CANDIDATES = (
    "btc_first_move",
    "eth_first_move",
    "btc_eth_agreement",
    "follower_own_first_move",
    "five_market_majority",
)
COMMON_FOLLOWERS = (*cpi_links.frozen.ESTABLISHED_FOLLOWERS, "established_group_median")


@dataclass(frozen=True)
class TransmissionRoute:
    """Dense event arrays for one of the 60 originally searched routes."""

    key: tuple[str, int, int, str]
    source_call: np.ndarray
    source_sign: np.ndarray
    later_sign: np.ndarray
    own_sign: np.ndarray
    partitions: np.ndarray
    control_rate: float


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    validation = activity_validation.load_validation_contract()
    required = {
        "cpi_transmission_full_family_randomization",
        "cpi_common_event_and_self_momentum",
    }
    available = {str(route["branch_id"]) for route in validation["routes"]}
    if not required.issubset(available):
        raise ValueError("Frozen CPI validation routes are missing")
    parent_result = json.loads(cpi_links.RESULT_PATH.read_text(encoding="utf-8"))
    if parent_result.get("status") != "completed_cpi_leader_and_meme_link_review":
        raise ValueError("Parent CPI link result is not terminal")
    if parent_result.get("profit_used"):
        raise ValueError("Parent CPI link review unexpectedly used profit")
    rows = pd.read_parquet(cpi_links.TRANSMISSION_ROWS_PATH)
    summary = pd.read_csv(cpi_links.TRANSMISSION_SUMMARY_PATH)
    if len(summary) != 60:
        raise ValueError("The frozen CPI transmission family is no longer 60 routes")
    return validation, rows, summary


def build_routes(rows: DataFrame) -> tuple[list[str], list[TransmissionRoute]]:
    event_rows = rows.loc[rows["sample_type"].eq("event")].copy()
    event_meta = (
        event_rows[["event_id", "whole_event_partition"]]
        .drop_duplicates()
        .sort_values("event_id", kind="stable")
    )
    if event_meta["event_id"].duplicated().any():
        raise ValueError("CPI event partition mapping is not unique")
    event_ids = event_meta["event_id"].astype(str).tolist()
    partition_by_event = event_meta.set_index("event_id")["whole_event_partition"]
    routes: list[TransmissionRoute] = []
    for values, group in rows.groupby(list(ROUTE_COLUMNS), sort=True):
        key = (str(values[0]), int(values[1]), int(values[2]), str(values[3]))
        events = (
            group.loc[group["sample_type"].eq("event")]
            .set_index("event_id")
            .reindex(event_ids)
        )
        if len(events) != len(event_ids) or events.index.duplicated().any():
            raise ValueError(f"CPI route lacks one row per event: {key}")
        controls = group.loc[
            group["sample_type"].eq("control")
            & group["call_issued"]
            & group["direction_success"].notna()
        ]
        control_rate = (
            float(controls["direction_success"].mean()) if len(controls) else np.nan
        )
        routes.append(
            TransmissionRoute(
                key=key,
                source_call=events["call_issued"].fillna(False).to_numpy(dtype=bool),
                source_sign=pd.to_numeric(events["leader_sign"], errors="coerce").to_numpy(),
                later_sign=np.sign(
                    pd.to_numeric(events["follower_later_return"], errors="coerce").to_numpy()
                ),
                own_sign=pd.to_numeric(
                    events["follower_own_initial_sign"], errors="coerce"
                ).to_numpy(),
                partitions=events.index.map(partition_by_event).to_numpy(dtype=str),
                control_rate=control_rate,
            )
        )
    if len(routes) != 60:
        raise ValueError(f"Expected 60 CPI routes, found {len(routes)}")
    return event_ids, routes


def transmission_route_measurement(
    route: TransmissionRoute,
    source_indexes: np.ndarray,
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    call = route.source_call[source_indexes]
    leader_sign = route.source_sign[source_indexes]
    usable = (
        call
        & np.isfinite(leader_sign)
        & (leader_sign != 0)
        & np.isfinite(route.later_sign)
        & (route.later_sign != 0)
    )
    success = route.later_sign[usable] == leader_sign[usable]
    event_rate = float(np.mean(success)) if len(success) else np.nan
    partition_rates: dict[str, float] = {}
    for partition in cpi_links.frozen.PARTITIONS:
        selected = usable & (route.partitions == partition)
        partition_rates[partition] = (
            float(np.mean(route.later_sign[selected] == leader_sign[selected]))
            if selected.any()
            else np.nan
        )
    own_usable = usable & np.isfinite(route.own_sign) & (route.own_sign != 0)
    own_rate = (
        float(np.mean(route.later_sign[own_usable] == route.own_sign[own_usable]))
        if own_usable.any()
        else np.nan
    )
    minimum_count = int(rule["minimum_issued_whole_events"])
    gates = {
        "overall": event_rate - float(rule["minimum_overall_direction_rate"]),
        "development": partition_rates["development_2021_2023"]
        - float(rule["minimum_each_partition_rate"]),
        "validation": partition_rates["internal_validation_2024_2025"]
        - float(rule["minimum_each_partition_rate"]),
        "ordinary_control": event_rate
        - route.control_rate
        - float(rule["minimum_control_uplift"]),
        "follower_own": event_rate - own_rate,
    }
    finite_gates = np.asarray(list(gates.values()), dtype=float)
    statistic = (
        float(np.min(finite_gates))
        if int(usable.sum()) >= minimum_count and np.isfinite(finite_gates).all()
        else np.nan
    )
    return {
        "issued_event_count": int(usable.sum()),
        "event_direction_rate": event_rate,
        "development_direction_rate": partition_rates["development_2021_2023"],
        "validation_direction_rate": partition_rates["internal_validation_2024_2025"],
        "control_direction_rate": route.control_rate,
        "follower_own_direction_rate": own_rate,
        "minimum_gate_margin": statistic,
        "all_unchanged_gates_pass": bool(np.isfinite(statistic) and statistic >= 0),
    }


def shuffled_source_indexes(
    partitions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle whole event blocks only within their frozen historical period."""

    output = np.arange(len(partitions), dtype=int)
    for partition in cpi_links.frozen.PARTITIONS:
        indexes = np.flatnonzero(partitions == partition)
        output[indexes] = rng.permutation(indexes)
    return output


def full_family_randomization(
    routes: Sequence[TransmissionRoute],
    rule: Mapping[str, Any],
    *,
    iterations: int,
    seed: int,
) -> tuple[DataFrame, DataFrame]:
    identity = np.arange(len(routes[0].partitions), dtype=int)
    observed_records: list[dict[str, Any]] = []
    for route in routes:
        measurement = transmission_route_measurement(route, identity, rule)
        observed_records.append(
            {
                **dict(zip(ROUTE_COLUMNS, route.key, strict=True)),
                **measurement,
            }
        )
    observed = DataFrame.from_records(observed_records)
    rng = np.random.default_rng(seed)
    null_records: list[dict[str, Any]] = []
    for iteration in range(iterations):
        source_indexes = shuffled_source_indexes(routes[0].partitions, rng)
        statistics = [
            transmission_route_measurement(route, source_indexes, rule)[
                "minimum_gate_margin"
            ]
            for route in routes
        ]
        finite = [float(value) for value in statistics if np.isfinite(value)]
        null_records.append(
            {
                "permutation": iteration + 1,
                "maximum_minimum_gate_margin_across_60_routes": (
                    max(finite) if finite else np.nan
                ),
            }
        )
    return observed, DataFrame.from_records(null_records)


def randomization_decision(observed: DataFrame, null: DataFrame) -> DataFrame:
    mask = np.ones(len(observed), dtype=bool)
    for column, value in zip(ROUTE_COLUMNS, PRIMARY_ROUTE, strict=True):
        mask &= observed[column].eq(value).to_numpy()
    selected = observed.loc[mask]
    if len(selected) != 1:
        raise ValueError("The frozen primary CPI route is missing or duplicated")
    row = selected.iloc[0]
    statistic = float(row["minimum_gate_margin"])
    exceedances, finite, p_value = activity_validation.empirical_p_value(
        statistic,
        null["maximum_minimum_gate_margin_across_60_routes"],
    )
    gates_pass = bool(row["all_unchanged_gates_pass"])
    passed = bool(gates_pass and np.isfinite(p_value) and p_value <= 0.05)
    return DataFrame.from_records(
        [
            {
                "branch_id": "cpi_transmission_full_family_randomization",
                **dict(zip(ROUTE_COLUMNS, PRIMARY_ROUTE, strict=True)),
                "issued_event_count": int(row["issued_event_count"]),
                "event_direction_rate": float(row["event_direction_rate"]),
                "development_direction_rate": float(
                    row["development_direction_rate"]
                ),
                "validation_direction_rate": float(row["validation_direction_rate"]),
                "control_direction_rate": float(row["control_direction_rate"]),
                "follower_own_direction_rate": float(
                    row["follower_own_direction_rate"]
                ),
                "minimum_gate_margin": statistic,
                "unchanged_original_gates_pass": gates_pass,
                "finite_permutations": finite,
                "null_maxima_at_least_observed": exceedances,
                "familywise_p_value": p_value,
                "familywise_gate_pass": passed,
                "verdict": (
                    "retained_after_60_route_randomization"
                    if passed
                    else "not_retained_after_60_route_randomization"
                ),
            }
        ]
    )


def common_sample_table(rows: DataFrame) -> DataFrame:
    selected = rows.loc[
        rows["leader_window_minutes"].eq(15)
        & rows["follower_end_minutes"].eq(60)
    ].copy()
    base_columns = [
        "sample_id",
        "event_id",
        "sample_type",
        "control_rank",
        "whole_event_partition",
    ]
    base = selected.loc[
        selected["leader"].eq("btc_eth_agreement")
        & selected["follower"].eq("established_group_median"),
        [
            *base_columns,
            "follower_later_return",
            "follower_own_initial_sign",
        ],
    ].rename(
        columns={
            "follower_later_return": "established_group_median_later_return",
            "follower_own_initial_sign": "established_group_median_own_sign",
        }
    )
    for leader in ("btc_first_move", "eth_first_move", "btc_eth_agreement"):
        leader_rows = selected.loc[
            selected["leader"].eq(leader)
            & selected["follower"].eq("established_group_median"),
            ["sample_id", "call_issued", "leader_sign", "leader_strength_ratio"],
        ].rename(
            columns={
                "call_issued": f"{leader}_call",
                "leader_sign": f"{leader}_sign",
                "leader_strength_ratio": f"{leader}_strength",
            }
        )
        base = base.merge(leader_rows, on="sample_id", how="left", validate="one_to_one")
    for follower in cpi_links.frozen.ESTABLISHED_FOLLOWERS:
        name = follower.split("/")[0].lower()
        follower_rows = selected.loc[
            selected["leader"].eq("btc_eth_agreement")
            & selected["follower"].eq(follower),
            ["sample_id", "follower_later_return", "follower_own_initial_sign"],
        ].rename(
            columns={
                "follower_later_return": f"{name}_later_return",
                "follower_own_initial_sign": f"{name}_own_sign",
            }
        )
        base = base.merge(follower_rows, on="sample_id", how="left", validate="one_to_one")

    sign_columns = [
        "btc_first_move_sign",
        "eth_first_move_sign",
        *[
            f"{pair.split('/')[0].lower()}_own_sign"
            for pair in cpi_links.frozen.ESTABLISHED_FOLLOWERS
        ],
    ]
    numeric_columns = [
        *sign_columns,
        "btc_first_move_strength",
        "eth_first_move_strength",
        "established_group_median_later_return",
        "established_group_median_own_sign",
        *[
            value
            for pair in cpi_links.frozen.ESTABLISHED_FOLLOWERS
            for value in (
                f"{pair.split('/')[0].lower()}_later_return",
                f"{pair.split('/')[0].lower()}_own_sign",
            )
        ],
    ]
    for column in numeric_columns:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base["five_market_majority_sign"] = np.sign(base[sign_columns].sum(axis=1))
    base["complete_candidate_data"] = base[numeric_columns].notna().all(axis=1)
    complete_events = set(
        base.loc[
            base["sample_type"].eq("event") & base["complete_candidate_data"],
            "event_id",
        ]
    )
    base["in_common_event_set"] = base["event_id"].isin(complete_events)
    return base.loc[
        base["in_common_event_set"] & base["complete_candidate_data"]
    ].reset_index(drop=True)


def candidate_call(
    rows: DataFrame, candidate: str, follower: str
) -> tuple[np.ndarray, np.ndarray]:
    if candidate in ("btc_first_move", "eth_first_move", "btc_eth_agreement"):
        call = rows[f"{candidate}_call"].fillna(False).to_numpy(dtype=bool)
        sign = pd.to_numeric(rows[f"{candidate}_sign"], errors="coerce").to_numpy()
        return call, sign
    if candidate == "follower_own_first_move":
        prefix = (
            "established_group_median"
            if follower == "established_group_median"
            else follower.split("/")[0].lower()
        )
        sign = pd.to_numeric(rows[f"{prefix}_own_sign"], errors="coerce").to_numpy()
        return np.isfinite(sign) & (sign != 0), sign
    if candidate == "five_market_majority":
        sign = pd.to_numeric(
            rows["five_market_majority_sign"], errors="coerce"
        ).to_numpy()
        return np.isfinite(sign) & (sign != 0), sign
    raise ValueError(f"Unknown common-event candidate: {candidate}")


def follower_later_sign(rows: DataFrame, follower: str) -> np.ndarray:
    prefix = (
        "established_group_median"
        if follower == "established_group_median"
        else follower.split("/")[0].lower()
    )
    return np.sign(pd.to_numeric(rows[f"{prefix}_later_return"], errors="coerce").to_numpy())


def common_event_comparison(common: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for follower in COMMON_FOLLOWERS:
        later_sign = follower_later_sign(common, follower)
        for candidate in COMMON_CANDIDATES:
            call, sign = candidate_call(common, candidate, follower)
            usable = call & np.isfinite(sign) & np.isfinite(later_sign) & (later_sign != 0)
            success = later_sign == sign
            event_mask = common["sample_type"].eq("event").to_numpy() & usable
            control_mask = common["sample_type"].eq("control").to_numpy() & usable
            event_rate = float(np.mean(success[event_mask])) if event_mask.any() else np.nan
            control_rate = (
                float(np.mean(success[control_mask])) if control_mask.any() else np.nan
            )
            partition_rates = {
                partition: (
                    float(
                        np.mean(
                            success[
                                event_mask
                                & common["whole_event_partition"].eq(partition).to_numpy()
                            ]
                        )
                    )
                    if (
                        event_mask
                        & common["whole_event_partition"].eq(partition).to_numpy()
                    ).any()
                    else np.nan
                )
                for partition in cpi_links.frozen.PARTITIONS
            }
            records.append(
                {
                    "follower": follower,
                    "candidate": candidate,
                    "common_complete_event_count": int(
                        common.loc[common["sample_type"].eq("event"), "event_id"].nunique()
                    ),
                    "issued_event_count": int(event_mask.sum()),
                    "event_direction_rate": event_rate,
                    "development_direction_rate": partition_rates[
                        "development_2021_2023"
                    ],
                    "validation_direction_rate": partition_rates[
                        "internal_validation_2024_2025"
                    ],
                    "issued_control_count": int(control_mask.sum()),
                    "control_direction_rate": control_rate,
                    "control_uplift": event_rate - control_rate,
                    "primary_outcome": follower == "established_group_median",
                    "primary_candidate": candidate == "btc_eth_agreement",
                }
            )
    return DataFrame.from_records(records)


def common_event_primary_decision(
    common: DataFrame, comparison: DataFrame, minimum_events: int
) -> DataFrame:
    primary = comparison.loc[
        comparison["follower"].eq("established_group_median")
        & comparison["candidate"].eq("btc_eth_agreement")
    ]
    if len(primary) != 1:
        raise ValueError("Common-event primary CPI comparison is missing")
    row = primary.iloc[0]
    event_rows = common.loc[common["sample_type"].eq("event")]
    call, leader_sign = candidate_call(
        event_rows, "btc_eth_agreement", "established_group_median"
    )
    later_sign = follower_later_sign(event_rows, "established_group_median")
    own_call, own_sign = candidate_call(
        event_rows, "follower_own_first_move", "established_group_median"
    )
    same_call = call & own_call & np.isfinite(leader_sign) & np.isfinite(later_sign)
    own_same_rate = (
        float(np.mean(later_sign[same_call] == own_sign[same_call]))
        if same_call.any()
        else np.nan
    )
    own_uplift = float(row["event_direction_rate"]) - own_same_rate
    passed = bool(
        int(row["issued_event_count"]) >= minimum_events
        and float(row["event_direction_rate"]) >= 0.55
        and float(row["development_direction_rate"]) >= 0.50
        and float(row["validation_direction_rate"]) >= 0.50
        and float(row["control_uplift"]) >= 0.05
        and own_uplift >= 0.05
    )
    return DataFrame.from_records(
        [
            {
                "branch_id": "cpi_common_event_and_self_momentum",
                "follower": "established_group_median",
                "candidate": "btc_eth_agreement",
                "common_complete_event_count": int(row["common_complete_event_count"]),
                "issued_event_count": int(row["issued_event_count"]),
                "event_direction_rate": float(row["event_direction_rate"]),
                "development_direction_rate": float(
                    row["development_direction_rate"]
                ),
                "validation_direction_rate": float(row["validation_direction_rate"]),
                "control_direction_rate": float(row["control_direction_rate"]),
                "control_uplift": float(row["control_uplift"]),
                "follower_own_rate_on_same_calls": own_same_rate,
                "uplift_over_follower_own_on_same_calls": own_uplift,
                "frozen_gate_pass": passed,
                "verdict": (
                    "retained_incremental_leader_information"
                    if passed
                    else "leader_does_not_add_five_points_over_self_momentum"
                ),
            }
        ]
    )


def render_report(
    randomization: DataFrame,
    common_decision: DataFrame,
    comparison: DataFrame,
    null: DataFrame,
) -> str:
    random = randomization.iloc[0]
    common = common_decision.iloc[0]
    diagnostics = comparison.loc[
        comparison["follower"].eq("established_group_median")
    ].sort_values("candidate")
    rows = [
        "# CPI Leader-Transmission Validation",
        "",
        "This asks whether the first 15 minutes of BTC/ETH movement after CPI adds "
        "directional information for the following 45 minutes in BNB, ADA, and TRX.",
        "",
        "## Frozen decisions",
        "",
        f"- Full 60-route shuffled-family probability: `{random.familywise_p_value:.3%}`; "
        f"decision: `{random.verdict}`.",
        f"- Exact-common-event incremental test: `{common.verdict}`.",
        f"- Primary calls: `{common.issued_event_count}` from "
        f"`{common.common_complete_event_count}` complete CPI events.",
        f"- BTC/ETH agreement accuracy: `{common.event_direction_rate:.1%}`; matched "
        f"ordinary-time accuracy: `{common.control_direction_rate:.1%}`.",
        f"- The followers' own first-15-minute continuation on exactly those calls: "
        f"`{common.follower_own_rate_on_same_calls:.1%}`; BTC/ETH incremental difference: "
        f"`{common.uplift_over_follower_own_on_same_calls:.1%}`.",
        "",
        "## Same-event diagnostics",
        "",
        "| Direction source | Calls | Established-group accuracy | Ordinary-time accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in diagnostics.itertuples(index=False):
        rows.append(
            f"| {row.candidate} | {row.issued_event_count} | "
            f"{row.event_direction_rate:.1%} | {row.control_direction_rate:.1%} |"
        )
    rows.extend(
        [
            "",
            f"- Whole-event within-period permutations: `{len(null)}`.",
            "- All leader definitions, horizons, and followers moved together during each shuffle.",
            "- Individual BNB, ADA, and TRX results are diagnostics; the established-group "
            "median was frozen as primary.",
            "- This tests short event-scoped direction, not profit or an entry/exit rule.",
            "",
        ]
    )
    return "\n".join(rows)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_layer2_cpi_transmission_validation":
        raise ValueError("Existing Layer 2 CPI validation result is invalid")
    for recorded in result.get("artifacts", {}).values():
        path = Path(recorded["path"])
        if not path.is_file() or g0.sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"Existing CPI validation artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    validation, rows, parent_summary = load_inputs()
    _, routes = build_routes(rows)
    rule = cpi_links.load_frozen_inputs()[0]["cpi_immediate_leader_transmission"]
    observed, null = full_family_randomization(
        routes,
        rule,
        iterations=int(validation_freeze.PERMUTATION_ITERATIONS),
        seed=int(validation["permutation_seed"]),
    )
    joined = observed.merge(
        parent_summary[list(ROUTE_COLUMNS) + ["event_direction_rate"]],
        on=list(ROUTE_COLUMNS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_parent"),
    )
    reproduction_error = float(
        (
            joined["event_direction_rate"]
            - joined["event_direction_rate_parent"]
        )
        .abs()
        .max()
    )
    if reproduction_error > 1e-12:
        raise ValueError("Observed CPI route rates did not reproduce the parent summary")
    randomization = randomization_decision(observed, null)
    common = common_sample_table(rows)
    comparison = common_event_comparison(common)
    common_decision = common_event_primary_decision(
        common,
        comparison,
        minimum_events=10,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(null, NULL_PATH)
    g0.atomic_write_csv(randomization, RANDOMIZATION_DECISION_PATH)
    g0.atomic_write_parquet(common, COMMON_SAMPLE_PATH)
    g0.atomic_write_csv(comparison, COMMON_COMPARISON_PATH)
    g0.atomic_write_csv(common_decision, COMMON_DECISION_PATH)
    REPORT_PATH.write_text(
        render_report(randomization, common_decision, comparison, null),
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "schema_version": 1,
        "status": "completed_layer2_cpi_transmission_validation",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": True,
        "direction_scope": "CPI events only; first 15 minutes to following 45 minutes",
        "validation_routes_completed": [
            "cpi_transmission_full_family_randomization",
            "cpi_common_event_and_self_momentum",
        ],
        "permutations": int(validation_freeze.PERMUTATION_ITERATIONS),
        "seed": int(validation["permutation_seed"]),
        "parent_rate_reproduction_max_absolute_error": reproduction_error,
        "randomization_decision": randomization.iloc[0].to_dict(),
        "common_event_decision": common_decision.iloc[0].to_dict(),
        "source_contracts": {
            "validation_freeze": artifact(validation_freeze.FREEZE_PATH),
            "cpi_link_result": artifact(cpi_links.RESULT_PATH),
            "cpi_transmission_rows": artifact(cpi_links.TRANSMISSION_ROWS_PATH),
        },
        "artifacts": {
            "null_maxima": artifact(NULL_PATH),
            "randomization_decision": artifact(RANDOMIZATION_DECISION_PATH),
            "common_samples": artifact(COMMON_SAMPLE_PATH),
            "common_comparison": artifact(COMMON_COMPARISON_PATH),
            "common_decision": artifact(COMMON_DECISION_PATH),
            "report": artifact(REPORT_PATH),
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
                    "validation_routes": [
                        "cpi_transmission_full_family_randomization",
                        "cpi_common_event_and_self_momentum",
                    ],
                    "permutations": validation_freeze.PERMUTATION_ITERATIONS,
                    "profit_will_be_used": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
