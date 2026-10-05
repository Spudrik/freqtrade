"""Test frozen source clocks plus first-hour market confirmation for later direction."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_confirmation_breadth_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_direct as layer2,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT = frozen.OUTPUT_ROOT / "event_confirmation_breadth_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "event_confirmation_breadth_freeze.json"
ROUTES_PATH = frozen.OUTPUT_ROOT / "event_confirmation_route_catalog.csv"
CONTROLS_PATH = frozen.OUTPUT_ROOT / "event_confirmation_control_catalog.csv"
COVERAGE_PATH = frozen.OUTPUT_ROOT / "event_confirmation_route_coverage.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260905a"
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / "event_confirmation_breadth_20260905a"
)

LEADER_TYPES = ("btc", "eth", "btc_eth_agreement", "broad_market")
HORIZONS = (2, 4, 8, 24)
SCOPES_BY_GROUP = {
    "recent_live_media": ("btc", "eth", "established_alts", "memes"),
    "historical_context": ("btc", "eth", "established_alts"),
}
MINIMUM_RESPONSE_ASSETS = {
    "btc": 1,
    "eth": 1,
    "established_alts": 4,
    "memes": 3,
}
PRIMARY_CONTROL = "same_weekday_hour_prior_week"
MINIMUM_CONFIRMED_EVENTS = 10
MINIMUM_EVENTS_PER_PARTITION = 3
ROTATION_COUNT = 19


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _truthy(values: Series) -> Series:
    return values.astype(str).str.lower().eq("true")


def _parse_time(frame: DataFrame, column: str) -> None:
    frame[column] = pd.to_datetime(
        frame[column], utc=True, errors="raise", format="mixed"
    )


def _sign_int(values: Series) -> Series:
    return np.sign(pd.to_numeric(values, errors="coerce")).fillna(0).astype(int)


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame]:
    result = json.loads(FREEZE_RESULT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_confirmation_breadth_freeze":
        raise ValueError("Event-confirmation breadth freeze is not terminal.")
    if freeze.get("status") != (
        "frozen_event_confirmation_breadth_before_market_outcomes"
    ):
        raise ValueError("Event-confirmation questions were not frozen.")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("The outcome-blind freeze unexpectedly opened outcomes.")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("routes", ROUTES_PATH),
        ("controls", CONTROLS_PATH),
        ("coverage", COVERAGE_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen breadth artifact changed: {path}")
    for group in ("parent_freezes", "parent_catalogues", "market_input_contracts"):
        for contract in freeze[group].values():
            path = Path(contract["path"])
            if contract["sha256"] != g0.sha256_file(path):
                raise ValueError(f"Source contract changed after freeze: {path}")

    routes = pd.read_csv(ROUTES_PATH)
    controls = pd.read_csv(CONTROLS_PATH)
    coverage = pd.read_csv(COVERAGE_PATH)
    _parse_time(routes, "anchor_utc")
    for column in ("event_anchor_utc", "control_anchor_utc"):
        _parse_time(controls, column)
    active = coverage.loc[_truthy(coverage["coverage_eligible"]), [
        "analysis_group",
        "route_id",
    ]].drop_duplicates()
    routes = routes.merge(
        active.assign(_active=True),
        on=["analysis_group", "route_id"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="_active")
    controls = controls.merge(
        routes[["route_event_id"]].drop_duplicates(),
        on="route_event_id",
        how="inner",
        validate="many_to_one",
    )
    return freeze, routes, controls, coverage


def sample_table(routes: DataFrame, controls: DataFrame) -> DataFrame:
    event_samples = routes[
        [
            "route_event_id",
            "analysis_group",
            "route_id",
            "whole_event_partition",
            "anchor_utc",
        ]
    ].copy()
    event_samples = event_samples.rename(
        columns={
            "route_event_id": "event_id",
            "analysis_group": "event_family",
            "route_id": "event_source",
            "anchor_utc": "sample_anchor_utc",
        }
    )
    event_samples["sample_type"] = "event"
    event_samples["control_type"] = "event"
    event_samples["control_rank"] = 0

    metadata = routes[
        ["route_event_id", "analysis_group", "route_id", "whole_event_partition"]
    ].drop_duplicates("route_event_id")
    control_samples = controls.merge(
        metadata,
        on=[
            "route_event_id",
            "analysis_group",
            "route_id",
            "whole_event_partition",
        ],
        how="left",
        validate="many_to_one",
    )
    if control_samples["route_id"].isna().any():
        raise ValueError("A control row has no matching frozen event.")
    control_samples = control_samples.rename(
        columns={
            "route_event_id": "event_id",
            "analysis_group": "event_family",
            "route_id": "event_source",
            "control_anchor_utc": "sample_anchor_utc",
        }
    )
    control_samples["sample_type"] = "control"
    columns = [
        "event_id",
        "event_source",
        "event_family",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
    ]
    output = pd.concat(
        [event_samples[columns], control_samples[columns]], ignore_index=True
    )
    output["sample_id"] = (
        output["event_id"].astype(str)
        + "|"
        + output["sample_type"].astype(str)
        + "|"
        + output["control_type"].astype(str)
        + "|"
        + output["control_rank"].astype(str)
    )
    if output["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique.")
    return output


def preflight_market_files(freeze: Mapping[str, Any]) -> DataFrame:
    records: list[dict[str, Any]] = []
    for pair in layer2.all_pairs(freeze):
        path = g0.ohlcv_path(pair, "1h")
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen-cohort 1h OHLCV: {path}")
        records.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return DataFrame.from_records(records)


def extract_market_metrics(
    freeze: Mapping[str, Any], samples: DataFrame
) -> DataFrame:
    parts: list[DataFrame] = []
    for pair in layer2.all_pairs(freeze):
        metrics, _unused_ranges = layer2.extract_pair_samples(pair, samples)
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True)


def _base_identifiers() -> list[str]:
    return [
        "sample_id",
        "event_id",
        "event_source",
        "event_family",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
    ]


def _single_leader_rows(first: DataFrame, pair: str, leader_type: str) -> DataFrame:
    output = first.loc[first["pair"].eq(pair), _base_identifiers() + [
        "activity_score",
        "signed_return",
        "pre_return_30d",
    ]].copy()
    output = output.rename(
        columns={
            "activity_score": "median_first_hour_activity",
            "signed_return": "median_first_hour_return",
            "pre_return_30d": "leader_pre_return_30d",
        }
    )
    output["leader_type"] = leader_type
    output["confirmation_asset_count"] = output[
        "median_first_hour_activity"
    ].notna().astype(int)
    output["directional_agreement"] = 1.0
    output["initial_direction"] = _sign_int(output["median_first_hour_return"])
    output["confirmed"] = (
        output["confirmation_asset_count"].eq(1)
        & output["median_first_hour_activity"].ge(1.25)
        & output["initial_direction"].ne(0)
    )
    return output


def _btc_eth_agreement_rows(first: DataFrame) -> DataFrame:
    pairs = first.loc[first["pair"].isin(["BTC/USDT:USDT", "ETH/USDT:USDT"])].copy()
    pairs["positive"] = pairs["signed_return"].gt(0)
    pairs["negative"] = pairs["signed_return"].lt(0)
    output = pairs.groupby(_base_identifiers(), dropna=False, sort=False).agg(
        confirmation_asset_count=("activity_score", "count"),
        median_first_hour_activity=("activity_score", "median"),
        median_first_hour_return=("signed_return", "median"),
        leader_pre_return_30d=("pre_return_30d", "median"),
        positive_fraction=("positive", "mean"),
        negative_fraction=("negative", "mean"),
        minimum_first_hour_activity=("activity_score", "min"),
    ).reset_index()
    output["leader_type"] = "btc_eth_agreement"
    output["directional_agreement"] = output[
        ["positive_fraction", "negative_fraction"]
    ].max(axis=1)
    agrees = output["directional_agreement"].eq(1.0)
    output["initial_direction"] = np.where(
        agrees, np.sign(output["median_first_hour_return"]), 0
    ).astype(int)
    output["confirmed"] = (
        output["confirmation_asset_count"].eq(2)
        & output["minimum_first_hour_activity"].ge(1.25)
        & output["initial_direction"].ne(0)
    )
    return output.drop(
        columns=["positive_fraction", "negative_fraction", "minimum_first_hour_activity"]
    )


def _broad_leader_rows(first: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    confirmation_pairs = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        *freeze["cohorts"]["established_alts"],
    ]
    pairs = first.loc[first["pair"].isin(confirmation_pairs)].copy()
    pairs["positive"] = pairs["signed_return"].gt(0)
    pairs["negative"] = pairs["signed_return"].lt(0)
    output = pairs.groupby(_base_identifiers(), dropna=False, sort=False).agg(
        confirmation_asset_count=("activity_score", "count"),
        median_first_hour_activity=("activity_score", "median"),
        median_first_hour_return=("signed_return", "median"),
        leader_pre_return_30d=("pre_return_30d", "median"),
        positive_fraction=("positive", "mean"),
        negative_fraction=("negative", "mean"),
    ).reset_index()
    output["leader_type"] = "broad_market"
    output["directional_agreement"] = output[
        ["positive_fraction", "negative_fraction"]
    ].max(axis=1)
    output["initial_direction"] = _sign_int(output["median_first_hour_return"])
    output["confirmed"] = (
        output["confirmation_asset_count"].ge(6)
        & output["directional_agreement"].ge(0.60)
        & output["median_first_hour_activity"].ge(1.25)
        & output["initial_direction"].ne(0)
    )
    return output.drop(columns=["positive_fraction", "negative_fraction"])


def build_leader_rows(metrics: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    first = metrics.loc[metrics["horizon_hours"].eq(1)].copy()
    output = pd.concat(
        [
            _single_leader_rows(first, "BTC/USDT:USDT", "btc"),
            _single_leader_rows(first, "ETH/USDT:USDT", "eth"),
            _btc_eth_agreement_rows(first),
            _broad_leader_rows(first, freeze),
        ],
        ignore_index=True,
    )
    if set(output["leader_type"].unique()) != set(LEADER_TYPES):
        raise ValueError("One or more frozen leader definitions are missing.")
    return output


def build_response_rows(metrics: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    one_hour = metrics.loc[metrics["horizon_hours"].eq(1), [
        "sample_id",
        "pair",
        "signed_return",
    ]].rename(columns={"signed_return": "return_1h"})
    later = metrics.loc[metrics["horizon_hours"].isin(HORIZONS)].merge(
        one_hour,
        on=["sample_id", "pair"],
        how="left",
        validate="many_to_one",
    )
    later["post_first_hour_return"] = (
        later["signed_return"].add(1).div(later["return_1h"].add(1)).sub(1)
    )
    later["scope"] = later["pair"].map(layer2.cohort_map(freeze))
    later = later.loc[later["scope"].isin(MINIMUM_RESPONSE_ASSETS)].copy()
    identifiers = _base_identifiers() + ["horizon_hours", "scope"]
    response = later.groupby(identifiers, dropna=False, sort=False).agg(
        response_asset_count=("post_first_hour_return", "count"),
        median_post_first_hour_return=("post_first_hour_return", "median"),
        response_pre_return_30d=("pre_return_30d", "median"),
    ).reset_index()
    response["minimum_response_assets"] = response["scope"].map(
        MINIMUM_RESPONSE_ASSETS
    )
    response["response_usable"] = (
        response["response_asset_count"].ge(response["minimum_response_assets"])
        & response["median_post_first_hour_return"].notna()
    )
    allowed = [
        scope in SCOPES_BY_GROUP[str(group)]
        for group, scope in response[["event_family", "scope"]].itertuples(
            index=False, name=None
        )
    ]
    return response.loc[allowed].reset_index(drop=True)


def combine_leaders_and_responses(
    leaders: DataFrame, responses: DataFrame
) -> DataFrame:
    leader_columns = [
        "sample_id",
        "leader_type",
        "confirmation_asset_count",
        "median_first_hour_activity",
        "median_first_hour_return",
        "leader_pre_return_30d",
        "directional_agreement",
        "initial_direction",
        "confirmed",
    ]
    output = responses.merge(
        leaders[leader_columns],
        on="sample_id",
        how="left",
        validate="many_to_many",
    )
    if output["leader_type"].isna().any():
        raise ValueError("A response row has no first-hour leader state.")
    output["direction_usable"] = output["initial_direction"].ne(0)
    output["continued"] = (
        output["median_post_first_hour_return"] * output["initial_direction"]
    ).gt(0).where(output["response_usable"] & output["direction_usable"])
    trend_direction = _sign_int(output["response_pre_return_30d"])
    output["prior_30d_trend_continued"] = (
        output["median_post_first_hour_return"] * trend_direction
    ).gt(0).where(output["response_usable"] & trend_direction.ne(0))
    return output


def _safe_rate(values: Series) -> float:
    usable = pd.to_numeric(values, errors="coerce").dropna()
    return float(usable.mean()) if len(usable) else math.nan


def _collapsed_control_rate(frame: DataFrame, column: str) -> tuple[int, float]:
    usable = frame.loc[frame[column].notna()]
    collapsed = usable.groupby("event_id", sort=False)[column].mean()
    return int(collapsed.size), _safe_rate(collapsed)


def _quantile(values: Sequence[float], q: float) -> float:
    usable = [value for value in values if math.isfinite(value)]
    return float(np.quantile(usable, q)) if usable else math.nan


def _rotated_rates(event: DataFrame) -> list[float]:
    ordered = event.sort_values(
        ["whole_event_partition", "sample_anchor_utc"], kind="stable"
    ).copy()
    rates: list[float] = []
    for shift in range(1, ROTATION_COUNT + 1):
        pieces: list[DataFrame] = []
        for _partition, part in ordered.groupby(
            "whole_event_partition", sort=False
        ):
            rotated = part.copy()
            rotated["rotated_confirmed"] = np.roll(
                part["confirmed"].to_numpy(dtype=bool), shift
            )
            rotated["rotated_direction"] = np.roll(
                part["initial_direction"].to_numpy(dtype=int), shift
            )
            pieces.append(rotated)
        rotated = pd.concat(pieces, ignore_index=True)
        selected = rotated.loc[
            rotated["rotated_confirmed"]
            & rotated["response_usable"]
            & rotated["rotated_direction"].ne(0)
        ]
        success = (
            selected["median_post_first_hour_return"]
            * selected["rotated_direction"]
        ).gt(0)
        rates.append(_safe_rate(success))
    return rates


def summarize_cells(rows: DataFrame, freeze: Mapping[str, Any]) -> DataFrame:
    keys = ["event_family", "event_source", "leader_type", "scope", "horizon_hours"]
    records: list[dict[str, Any]] = []
    for values, cell in rows.groupby(keys, dropna=False, sort=False):
        group, route_id, leader_type, scope, horizon = values
        event = cell.loc[
            cell["sample_type"].eq("event")
            & cell["response_usable"]
            & cell["direction_usable"]
        ].copy()
        confirmed = event.loc[event["confirmed"]]
        controls = cell.loc[
            cell["sample_type"].eq("control")
            & cell["control_type"].eq(PRIMARY_CONTROL)
            & cell["confirmed"]
            & cell["response_usable"]
            & cell["direction_usable"]
        ]
        control_count, confirmation_only_rate = _collapsed_control_rate(
            controls, "continued"
        )
        pair_rate = _safe_rate(confirmed["continued"])
        event_only_rate = _safe_rate(event["continued"])
        trend_rate = _safe_rate(confirmed["prior_30d_trend_continued"])
        partitions = freeze["analysis_groups"][str(group)]["partitions"]
        partition_counts = {
            partition: int(
                confirmed["whole_event_partition"].eq(partition).sum()
            )
            for partition in partitions
        }
        partition_rates = {
            partition: _safe_rate(
                confirmed.loc[
                    confirmed["whole_event_partition"].eq(partition), "continued"
                ]
            )
            for partition in partitions
        }
        shuffled_q75 = _quantile(_rotated_rates(event), 0.75)
        passes = (
            len(confirmed) >= MINIMUM_CONFIRMED_EVENTS
            and control_count >= MINIMUM_CONFIRMED_EVENTS
            and pair_rate >= 0.55
            and pair_rate - event_only_rate >= 0.05
            and pair_rate - confirmation_only_rate >= 0.05
            and all(
                count >= MINIMUM_EVENTS_PER_PARTITION
                for count in partition_counts.values()
            )
            and all(rate >= 0.50 for rate in partition_rates.values())
            and math.isfinite(shuffled_q75)
            and pair_rate > shuffled_q75
        )
        record: dict[str, Any] = {
            "analysis_group": group,
            "route_id": route_id,
            "leader_type": leader_type,
            "response_scope": scope,
            "total_horizon_hours": int(horizon),
            "post_first_hour_hours": int(horizon) - 1,
            "confirmed_event_count": len(confirmed),
            "confirmed_control_event_count": control_count,
            "pair_continuation_rate": pair_rate,
            "event_only_continuation_rate": event_only_rate,
            "confirmation_only_continuation_rate": confirmation_only_rate,
            "prior_30d_trend_rate": trend_rate,
            "pair_minus_event_only": pair_rate - event_only_rate,
            "pair_minus_confirmation_only": pair_rate - confirmation_only_rate,
            "shuffled_rate_q75": shuffled_q75,
            "beats_shuffled": bool(
                math.isfinite(pair_rate)
                and math.isfinite(shuffled_q75)
                and pair_rate > shuffled_q75
            ),
            "meets_conditional_direction_rule": bool(passes),
            "meets_65_target": bool(passes and pair_rate >= 0.65),
        }
        for partition in partitions:
            safe = partition.replace("internal_", "").replace("_", "-")
            record[f"{safe}_count"] = partition_counts[partition]
            record[f"{safe}_rate"] = partition_rates[partition]
        records.append(record)
    return DataFrame.from_records(records)


def route_decisions(summary: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for (group, route_id), rows in summary.groupby(
        ["analysis_group", "route_id"], sort=False
    ):
        passed = rows.loc[rows["meets_conditional_direction_rule"]]
        if passed.empty:
            verdict = "no_incremental_direction_lead"
            plain = (
                "No first-hour confirmation route reached the frozen 55% floor "
                "while beating both component controls and the rotated-sign check."
            )
        elif len(passed) == 1:
            verdict = "provisional_single_cell_lead"
            plain = (
                "One conditional cell passed; it remains provisional because adjacent "
                "horizon or alternative-leader repetition was absent."
            )
        else:
            verdict = "repeated_conditional_direction_lead"
            plain = (
                "More than one frozen leader, scope, or horizon cell passed the full "
                "component-control and chronological-partition rule."
            )
        best = passed.sort_values(
            ["meets_65_target", "pair_continuation_rate"], ascending=False
        ).head(1)
        records.append(
            {
                "analysis_group": group,
                "route_id": route_id,
                "verdict": verdict,
                "passing_cells": len(passed),
                "leader_types_passing": ";".join(
                    sorted(map(str, passed["leader_type"].unique()))
                ),
                "response_scopes_passing": ";".join(
                    sorted(map(str, passed["response_scope"].unique()))
                ),
                "horizons_passing": ";".join(
                    str(int(value))
                    for value in sorted(passed["total_horizon_hours"].unique())
                ),
                "best_pair_rate": (
                    float(best.iloc[0]["pair_continuation_rate"])
                    if not best.empty
                    else math.nan
                ),
                "plain_result": plain,
            }
        )
    return DataFrame.from_records(records)


def report_text(decisions: DataFrame, summary: DataFrame) -> str:
    passed = summary.loc[summary["meets_conditional_direction_rule"]].sort_values(
        ["meets_65_target", "pair_continuation_rate"], ascending=False
    )
    lines = [
        "# Event clock plus first-hour confirmation review",
        "",
        (
            "This batch asked whether a news, web, wider-market, or multi-event clock "
            "becomes directionally useful only after BTC, ETH, or the broader market "
            "moves unusually strongly during the first hour."
        ),
        "",
        "| Source clock | Decision | Passing cells | Plain result |",
        "|---|---:|---:|---|",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| `{row.route_id}` | `{row.verdict}` | {row.passing_cells} | "
            f"{row.plain_result} |"
        )
    lines.extend(["", "## Passing cells", ""])
    if passed.empty:
        lines.append("No cell passed every frozen component and shuffle control.")
    else:
        lines.extend(
            [
                "| Source | First-hour sign | Later market | Total window | Correct |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in passed.itertuples(index=False):
            lines.append(
                f"| `{row.route_id}` | `{row.leader_type}` | "
                f"`{row.response_scope}` | {row.total_horizon_hours}h | "
                f"{row.pair_continuation_rate:.1%} |"
            )
    lines.extend(
        [
            "",
            (
                "Union, media-overlap, and historical confluence routes overlap with "
                "their component clocks. They are not independent discoveries."
            ),
            (
                "A passing item is a research lead only. Profit, entries, exits, and "
                "trading-rule promotion were not tested."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "event_confirmation_breadth_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_confirmation_breadth_direct":
            raise ValueError("Existing breadth direct result is not terminal.")
        return result

    freeze, routes, controls, _coverage = load_frozen_inputs()
    samples = sample_table(routes, controls)
    market_files = preflight_market_files(freeze)
    metrics = extract_market_metrics(freeze, samples)
    leaders = build_leader_rows(metrics, freeze)
    responses = build_response_rows(metrics, freeze)
    combined = combine_leaders_and_responses(leaders, responses)
    summary = summarize_cells(combined, freeze)
    decisions = route_decisions(summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "market_files": OUTPUT_ROOT / "market_input_coverage.csv",
        "cell_summary": OUTPUT_ROOT / "confirmation_direction_cell_summary.csv",
        "route_decisions": OUTPUT_ROOT / "confirmation_direction_route_decisions.csv",
        "report": OUTPUT_ROOT / "event_confirmation_breadth_plain_review.md",
        "sample_metrics": DETAIL_ROOT / "event_confirmation_sample_metrics.parquet",
        "leader_rows": DETAIL_ROOT / "event_confirmation_leader_rows.parquet",
        "response_rows": DETAIL_ROOT / "event_confirmation_response_rows.parquet",
    }
    g0.atomic_write_csv(market_files, paths["market_files"])
    g0.atomic_write_csv(summary, paths["cell_summary"])
    g0.atomic_write_csv(decisions, paths["route_decisions"])
    paths["report"].write_text(
        report_text(decisions, summary), encoding="utf-8", newline="\n"
    )
    g0.atomic_write_parquet(metrics, paths["sample_metrics"])
    g0.atomic_write_parquet(leaders, paths["leader_rows"])
    g0.atomic_write_parquet(combined, paths["response_rows"])
    result = {
        "schema_version": 1,
        "status": "completed_event_confirmation_breadth_direct",
        "created_at_utc": g0.utc_now(),
        "outcomes_opened": True,
        "profit_used": False,
        "source_direction_assumed": False,
        "route_count": int(decisions["route_id"].nunique()),
        "cell_count": len(summary),
        "passing_cell_count": int(summary["meets_conditional_direction_rule"].sum()),
        "artifacts": {
            name: artifact(path)
            for name, path in paths.items()
            if name not in {"sample_metrics", "leader_rows", "response_rows"}
        },
        "detail_artifacts": {
            name: artifact(paths[name])
            for name in ("sample_metrics", "leader_rows", "response_rows")
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "requires_terminal_freeze": str(FREEZE_RESULT),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
