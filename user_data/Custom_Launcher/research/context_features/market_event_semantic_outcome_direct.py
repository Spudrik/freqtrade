"""Test frozen semantic-story activity and direction questions against OHLCV."""

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
    market_context_independent_direct as independent,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_semantic_outcome_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260905a"
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / "semantic_outcome_20260905a"
)
RESULT_PATH = OUTPUT_ROOT / "semantic_story_direct_result.json"
EVENT_ROWS_PATH = DETAIL_ROOT / "semantic_story_event_rows.parquet"
MARKET_COVERAGE_PATH = OUTPUT_ROOT / "semantic_story_market_coverage.csv"
ACTIVITY_CELLS_PATH = OUTPUT_ROOT / "semantic_story_activity_cells.csv"
DIRECTION_CELLS_PATH = OUTPUT_ROOT / "semantic_story_direction_cells.csv"
ROUTE_DECISIONS_PATH = OUTPUT_ROOT / "semantic_story_route_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "semantic_story_plain_review.md"

MINIMUM_SCOPE_MEMBERS = {
    "btc": 1,
    "eth": 1,
    "established_alts": 4,
    "memes": 3,
}
ACTIVITY_MULTIPLE = 1.20
MINIMUM_ACTIVITY_RATE = 0.55
MINIMUM_EVENTS_PER_PARTITION_ACTIVITY = 10
MINIMUM_MODERATE_LIFT = 0.05
MINIMUM_DIRECTION_EVENTS = 10
MINIMUM_DIRECTION_PER_PARTITION = 3
MINIMUM_DIRECTION_RATE = 0.55
MINIMUM_DIRECTION_PARTITION_RATE = 0.50
MINIMUM_DIRECTION_LIFT = 0.03
PRIOR_TREND_HOURS = 30 * 24


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _truthy(values: Series) -> Series:
    return values.astype(str).str.lower().eq("true")


def sign(value: Any) -> int:
    if value is None or pd.isna(value) or float(value) == 0:
        return 0
    return 1 if float(value) > 0 else -1


def safe_rate(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else math.nan


def aligned_hour(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.ceil("h")


def _has_complete_hourly_window(
    frame: DataFrame, anchor: pd.Timestamp, horizon: int
) -> bool:
    """Require every candle used by the shared metric helper to be present."""
    start = anchor - pd.Timedelta(hours=25)
    end = anchor + pd.Timedelta(hours=horizon - 1)
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    return len(expected) > 0 and expected.isin(frame.index).all()


def causal_prior_return(
    frame: DataFrame, anchor: pd.Timestamp, hours: int = PRIOR_TREND_HOURS
) -> float:
    """Return the completed pre-event trend without using the event candle."""
    anchor = aligned_hour(anchor)
    end = anchor - pd.Timedelta(hours=1)
    start = end - pd.Timedelta(hours=hours)
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    if len(expected) != hours + 1 or not expected.isin(frame.index).all():
        return math.nan
    start_value = frame.at[start, "close"]
    end_value = frame.at[end, "close"]
    if isinstance(start_value, Series) or isinstance(end_value, Series):
        return math.nan
    start_close = float(start_value)
    end_close = float(end_value)
    if not np.isfinite(start_close) or start_close <= 0 or not np.isfinite(end_close):
        return math.nan
    return end_close / start_close - 1.0


def pair_event_metrics(
    frame: DataFrame,
    event_anchor: pd.Timestamp,
    control_anchors: Sequence[pd.Timestamp],
    horizon: int,
) -> dict[str, float] | None:
    """Use the shared metric implementation after excluding gapped windows."""
    if not _has_complete_hourly_window(frame, event_anchor, horizon):
        return None
    complete_controls = [
        anchor
        for anchor in control_anchors
        if _has_complete_hourly_window(frame, anchor, horizon)
    ]
    return independent.pair_event_metrics(
        frame, event_anchor, complete_controls, horizon
    )


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(frozen.RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(frozen.FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_semantic_story_outcome_freeze":
        raise ValueError("Semantic-story outcome freeze result is not terminal.")
    if freeze.get("status") != (
        "frozen_semantic_story_questions_before_market_outcomes"
    ):
        raise ValueError("Semantic-story questions are not frozen.")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Semantic-story freeze unexpectedly opened outcomes.")
    if result.get("profit_used") or freeze.get("profit_used"):
        raise ValueError("Semantic-story freeze unexpectedly used profit.")
    for name, path in (
        ("routes", frozen.ROUTES_PATH),
        ("coverage", frozen.COVERAGE_PATH),
        ("controls", frozen.CONTROLS_PATH),
        ("freeze", frozen.FREEZE_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen semantic-story artifact changed: {path}")
    for contract in freeze["parent_contracts"].values():
        if not isinstance(contract, Mapping) or "path" not in contract:
            continue
        path = Path(str(contract["path"]))
        if g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Semantic parent contract changed: {path}")
    routes = pd.read_csv(frozen.ROUTES_PATH)
    controls = pd.read_csv(frozen.CONTROLS_PATH)
    routes["anchor_utc"] = pd.to_datetime(
        routes["anchor_utc"], utc=True, errors="raise", format="mixed"
    )
    for column in ("event_anchor_utc", "control_anchor_utc"):
        controls[column] = pd.to_datetime(
            controls[column], utc=True, errors="raise", format="mixed"
        )
    if set(routes["route_id"]) != set(frozen.ROUTES):
        raise ValueError("Semantic direct input contains an unfrozen route.")
    return freeze, routes, controls


def market_scopes(freeze: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    cohorts = freeze["cohorts"]
    return {
        "btc": ("BTC/USDT:USDT",),
        "eth": ("ETH/USDT:USDT",),
        "established_alts": tuple(cohorts["established_alts"]),
        "memes": tuple(cohorts["top_ten_traded_memes"]),
    }


def load_market_frames(
    freeze: Mapping[str, Any],
) -> tuple[dict[str, DataFrame], DataFrame]:
    scopes = market_scopes(freeze)
    pairs = sorted({pair for members in scopes.values() for pair in members})
    frames: dict[str, DataFrame] = {}
    records: list[dict[str, Any]] = []
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen-cohort 1h OHLCV: {path}")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicates = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
        frame["_position"] = np.arange(len(frame), dtype=int)
        frames[pair] = frame.set_index("date", drop=False)
        records.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "rows": len(frame),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "duplicate_timestamps_removed": duplicates,
            }
        )
    return frames, DataFrame.from_records(records)


def _scope_metrics(
    scope: str,
    members: Sequence[str],
    pair_rows: Mapping[str, Mapping[str, float]],
) -> dict[str, float] | None:
    available = [pair_rows[pair] for pair in members if pair in pair_rows]
    if len(available) < MINIMUM_SCOPE_MEMBERS[scope]:
        return None
    return {
        "member_count": len(available),
        "signed_return": float(np.mean([row["signed_return"] for row in available])),
        "abs_return_ratio": float(
            np.median([row["abs_return_ratio"] for row in available])
        ),
        "range_ratio": float(np.median([row["range_ratio"] for row in available])),
        "volume_ratio": float(np.median([row["volume_ratio"] for row in available])),
        "activity_score": float(np.median([row["activity_score"] for row in available])),
        "minimum_pair_control_count": min(
            int(row["control_count"]) for row in available
        ),
    }


def _scope_prior_return(
    frames: Mapping[str, DataFrame],
    members: Sequence[str],
    anchor: pd.Timestamp,
    minimum_members: int,
) -> float:
    returns = [causal_prior_return(frames[pair], anchor) for pair in members]
    finite = [value for value in returns if np.isfinite(value)]
    return float(np.mean(finite)) if len(finite) >= minimum_members else math.nan


def _scope_control_direction_rate(
    frames: Mapping[str, DataFrame],
    members: Sequence[str],
    control_anchors: Sequence[pd.Timestamp],
    horizon: int,
    predicted_direction: int,
    minimum_members: int,
) -> tuple[int, float]:
    if predicted_direction == 0:
        return 0, math.nan
    successes: list[bool] = []
    for anchor in control_anchors:
        returns: list[float] = []
        for pair in members:
            if not _has_complete_hourly_window(frames[pair], anchor, horizon):
                continue
            metrics = independent.window_metrics(frames[pair], anchor, horizon)
            if metrics is not None:
                returns.append(float(metrics["signed_return"]))
        if len(returns) < minimum_members:
            continue
        actual = sign(float(np.mean(returns)))
        if actual:
            successes.append(actual == predicted_direction)
    return len(successes), float(np.mean(successes)) if successes else math.nan


def extract_event_rows(
    freeze: Mapping[str, Any],
    routes: DataFrame,
    controls: DataFrame,
    frames: Mapping[str, DataFrame],
) -> DataFrame:
    scopes = market_scopes(freeze)
    control_map = {
        str(event_id): tuple(
            aligned_hour(value) for value in group["control_anchor_utc"]
        )
        for event_id, group in controls.groupby("route_event_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for event in routes.itertuples(index=False):
        route_event_id = str(event.route_event_id)
        event_anchor = aligned_hour(event.anchor_utc)
        event_controls = control_map.get(route_event_id, ())
        prediction = (
            1
            if event.direction == "positive"
            else -1 if event.direction == "negative" else 0
        )
        if event.route_id != "positive_or_negative":
            prediction = 0
        for horizon in frozen.HORIZONS_HOURS:
            pair_rows: dict[str, Mapping[str, float]] = {}
            for pair, frame in frames.items():
                metrics = pair_event_metrics(frame, event_anchor, event_controls, horizon)
                if metrics is not None:
                    pair_rows[pair] = metrics
            for scope, members in scopes.items():
                scoped = _scope_metrics(scope, members, pair_rows)
                if scoped is None:
                    continue
                control_count, control_rate = _scope_control_direction_rate(
                    frames,
                    members,
                    event_controls,
                    horizon,
                    prediction,
                    MINIMUM_SCOPE_MEMBERS[scope],
                )
                actual_direction = sign(scoped["signed_return"])
                issued = prediction != 0 and actual_direction != 0
                prior_return = _scope_prior_return(
                    frames,
                    members,
                    event_anchor,
                    MINIMUM_SCOPE_MEMBERS[scope],
                )
                prior_direction = sign(prior_return)
                records.append(
                    {
                        "route_event_id": route_event_id,
                        "story_id": event.story_id,
                        "route_id": event.route_id,
                        "route_label": event.route_label,
                        "whole_event_partition": event.whole_event_partition,
                        "story_first_seen_at": event.anchor_utc,
                        "market_anchor_utc": event_anchor,
                        "alignment_delay_minutes": (
                            event_anchor - pd.Timestamp(event.anchor_utc)
                        ).total_seconds()
                        / 60,
                        "direction_label": event.direction,
                        "severity": event.severity,
                        "novelty": event.novelty,
                        "label_confidence": event.confidence,
                        "horizon_hours": horizon,
                        "market_scope": scope,
                        **scoped,
                        "reaction_success": scoped["activity_score"]
                        >= ACTIVITY_MULTIPLE,
                        "predicted_direction": prediction,
                        "actual_direction": actual_direction,
                        "issued_direction_call": issued,
                        "direction_success": (
                            prediction == actual_direction if issued else None
                        ),
                        "prior_30d_return": prior_return,
                        "prior_30d_trend_direction": prior_direction,
                        "prior_30d_trend_success": (
                            prior_direction == actual_direction
                            if prior_direction and actual_direction
                            else None
                        ),
                        "matched_control_direction_count": control_count,
                        "matched_control_direction_rate": control_rate,
                    }
                )
    return add_direction_controls(DataFrame.from_records(records))


def add_direction_controls(rows: DataFrame) -> DataFrame:
    output = rows.copy()
    output["rotated_prediction"] = 0
    rotation_keys = [
        "route_id",
        "market_scope",
        "horizon_hours",
        "whole_event_partition",
    ]
    for _keys, indexes in output.groupby(rotation_keys, sort=False).groups.items():
        ordered = output.loc[indexes].sort_values("market_anchor_utc", kind="stable")
        predictions = ordered["predicted_direction"].to_numpy(dtype=int)
        if len(predictions):
            output.loc[ordered.index, "rotated_prediction"] = np.roll(predictions, 1)
    output["rotated_direction_success"] = (
        output["rotated_prediction"].eq(output["actual_direction"])
    ).where(output["rotated_prediction"].ne(0) & output["actual_direction"].ne(0))

    output["development_majority_direction"] = 0
    majority_keys = ["route_id", "market_scope", "horizon_hours"]
    for _keys, indexes in output.groupby(majority_keys, sort=False).groups.items():
        group = output.loc[indexes]
        development = group.loc[
            group["whole_event_partition"].eq(frozen.PARTITIONS[0])
            & group["actual_direction"].ne(0)
        ]
        if development.empty:
            continue
        majority = 1 if float(development["actual_direction"].mean()) >= 0 else -1
        output.loc[indexes, "development_majority_direction"] = majority
    output["majority_direction_success"] = (
        output["development_majority_direction"].eq(output["actual_direction"])
    ).where(
        output["development_majority_direction"].ne(0)
        & output["actual_direction"].ne(0)
    )
    return output


def _activity_reference_rates(rows: DataFrame) -> dict[tuple[str, int, str], float]:
    all_rows = rows.loc[rows["route_id"].eq("all_cooled_stories")]
    records: dict[tuple[str, int, str], float] = {}
    for (scope, horizon), cell in all_rows.groupby(
        ["market_scope", "horizon_hours"], sort=False
    ):
        records[(str(scope), int(horizon), "overall_all")] = safe_rate(
            cell["reaction_success"]
        )
        lower = cell.loc[~cell["severity"].isin(["moderate", "major"])]
        records[(str(scope), int(horizon), "overall_lower")] = safe_rate(
            lower["reaction_success"]
        )
        for partition in frozen.PARTITIONS:
            part = cell.loc[cell["whole_event_partition"].eq(partition)]
            lower_part = lower.loc[lower["whole_event_partition"].eq(partition)]
            records[(str(scope), int(horizon), f"all|{partition}")] = safe_rate(
                part["reaction_success"]
            )
            records[(str(scope), int(horizon), f"lower|{partition}")] = safe_rate(
                lower_part["reaction_success"]
            )
    return records


def summarize_activity(rows: DataFrame) -> DataFrame:
    references = _activity_reference_rates(rows)
    records: list[dict[str, Any]] = []
    activity_routes = rows.loc[
        rows["route_id"].isin(["all_cooled_stories", "moderate_or_major"])
    ]
    for (route, scope, horizon), cell in activity_routes.groupby(
        ["route_id", "market_scope", "horizon_hours"], sort=False
    ):
        scope = str(scope)
        horizon = int(horizon)
        record: dict[str, Any] = {
            "route_id": route,
            "market_scope": scope,
            "horizon_hours": horizon,
            "whole_events": int(cell["route_event_id"].nunique()),
            "reaction_success_rate": safe_rate(cell["reaction_success"]),
            "median_activity_score": float(cell["activity_score"].median()),
            "all_story_reference_rate": references[
                (scope, horizon, "overall_all")
            ],
            "minor_or_unclear_reference_rate": references[
                (scope, horizon, "overall_lower")
            ],
        }
        passes = True
        for partition in frozen.PARTITIONS:
            part = cell.loc[cell["whole_event_partition"].eq(partition)]
            count = int(part["route_event_id"].nunique())
            rate = safe_rate(part["reaction_success"])
            suffix = "development" if partition == frozen.PARTITIONS[0] else "validation"
            all_rate = references[(scope, horizon, f"all|{partition}")]
            lower_rate = references[(scope, horizon, f"lower|{partition}")]
            record[f"{suffix}_events"] = count
            record[f"{suffix}_reaction_rate"] = rate
            record[f"{suffix}_all_story_reference_rate"] = all_rate
            record[f"{suffix}_minor_or_unclear_reference_rate"] = lower_rate
            passes &= count >= MINIMUM_EVENTS_PER_PARTITION_ACTIVITY
            passes &= math.isfinite(rate) and rate >= MINIMUM_ACTIVITY_RATE
            if route == "moderate_or_major":
                passes &= math.isfinite(all_rate) and rate - all_rate >= MINIMUM_MODERATE_LIFT
                passes &= (
                    math.isfinite(lower_rate)
                    and rate - lower_rate >= MINIMUM_MODERATE_LIFT
                )
        if route == "moderate_or_major":
            overall = float(record["reaction_success_rate"])
            all_overall = float(record["all_story_reference_rate"])
            lower_overall = float(record["minor_or_unclear_reference_rate"])
            passes &= overall - all_overall >= MINIMUM_MODERATE_LIFT
            passes &= overall - lower_overall >= MINIMUM_MODERATE_LIFT
        record["meets_activity_rule"] = bool(passes)
        records.append(record)
    return DataFrame.from_records(records)


def summarize_direction(rows: DataFrame) -> DataFrame:
    signed = rows.loc[rows["route_id"].eq("positive_or_negative")].copy()
    records: list[dict[str, Any]] = []
    for (scope, horizon), cell in signed.groupby(
        ["market_scope", "horizon_hours"], sort=False
    ):
        calls = cell.loc[cell["issued_direction_call"]].copy()
        record: dict[str, Any] = {
            "route_id": "positive_or_negative",
            "market_scope": scope,
            "horizon_hours": int(horizon),
            "issued_events": int(calls["route_event_id"].nunique()),
            "abstentions": int((~cell["issued_direction_call"]).sum()),
            "direction_success_rate": safe_rate(calls["direction_success"]),
            "prior_30d_trend_success_rate": safe_rate(
                calls["prior_30d_trend_success"]
            ),
            "development_majority_success_rate": safe_rate(
                calls["majority_direction_success"]
            ),
            "rotated_story_sign_success_rate": safe_rate(
                calls["rotated_direction_success"]
            ),
            "matched_ordinary_direction_rate": safe_rate(
                calls.loc[
                    calls["matched_control_direction_count"].ge(10),
                    "matched_control_direction_rate",
                ]
            ),
        }
        comparator_values = [
            record["prior_30d_trend_success_rate"],
            record["development_majority_success_rate"],
            record["rotated_story_sign_success_rate"],
            record["matched_ordinary_direction_rate"],
        ]
        finite = [float(value) for value in comparator_values if pd.notna(value)]
        strongest = max(finite) if finite else math.nan
        record["strongest_comparator_rate"] = strongest
        passes = int(record["issued_events"]) >= MINIMUM_DIRECTION_EVENTS
        overall = float(record["direction_success_rate"])
        passes &= math.isfinite(overall) and overall >= MINIMUM_DIRECTION_RATE
        passes &= math.isfinite(strongest) and overall - strongest >= MINIMUM_DIRECTION_LIFT
        for partition in frozen.PARTITIONS:
            part = calls.loc[calls["whole_event_partition"].eq(partition)]
            suffix = "development" if partition == frozen.PARTITIONS[0] else "validation"
            count = int(part["route_event_id"].nunique())
            rate = safe_rate(part["direction_success"])
            record[f"{suffix}_issued_events"] = count
            record[f"{suffix}_direction_rate"] = rate
            passes &= count >= MINIMUM_DIRECTION_PER_PARTITION
            passes &= math.isfinite(rate) and rate >= MINIMUM_DIRECTION_PARTITION_RATE
        record["meets_direction_rule"] = bool(passes)
        records.append(record)
    return DataFrame.from_records(records)


def route_decisions(activity: DataFrame, direction: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for route in ("all_cooled_stories", "moderate_or_major"):
        passed = activity.loc[
            activity["route_id"].eq(route) & activity["meets_activity_rule"]
        ]
        if passed.empty:
            verdict = "no_repeated_activity_lead"
        elif len(passed) == 1:
            verdict = "provisional_single_activity_cell"
        else:
            verdict = "repeated_activity_cells_requires_dependency_review"
        records.append(
            {
                "route_id": route,
                "target": "activity",
                "passing_cells": len(passed),
                "verdict": verdict,
                "plain_result": (
                    "No cell passed the frozen activity rule."
                    if passed.empty
                    else "Passing cells reuse story episodes and are not independent proof."
                ),
            }
        )
    passed_direction = direction.loc[direction["meets_direction_rule"]]
    records.append(
        {
            "route_id": "positive_or_negative",
            "target": "direction",
            "passing_cells": len(passed_direction),
            "verdict": (
                "no_semantic_direction_lead"
                if passed_direction.empty
                else "provisional_direction_cells_requires_dependency_review"
            ),
            "plain_result": (
                "Reviewed story sign did not beat the frozen simple controls."
                if passed_direction.empty
                else (
                    "At least one signed cell passed, but related scopes and "
                    "horizons reuse events."
                )
            ),
        }
    )
    return DataFrame.from_records(records)


def report_text(
    decisions: DataFrame, activity: DataFrame, direction: DataFrame
) -> str:
    lines = [
        "# Semantic Story Market Review",
        "",
        "This test used 120 price-blind reviewed story labels and a shared six-hour "
        "cooldown. Market windows began at the first hourly candle at or after each "
        "story became available.",
        "",
        "## Route decisions",
        "",
        "| Story question | Target | Passing cells | Decision |",
        "|---|---|---:|---|",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| `{row.route_id}` | `{row.target}` | {row.passing_cells} | "
            f"`{row.verdict}` |"
        )
    lines.extend(
        [
            "",
            "## Best activity rows",
            "",
            "| Route | Market | Window | Development | Validation | Passed |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in activity.sort_values(
        ["meets_activity_rule", "validation_reaction_rate"], ascending=False
    ).head(8).itertuples(index=False):
        lines.append(
            f"| `{row.route_id}` | `{row.market_scope}` | {row.horizon_hours}h | "
            f"{row.development_reaction_rate:.1%} | {row.validation_reaction_rate:.1%} | "
            f"`{bool(row.meets_activity_rule)}` |"
        )
    lines.extend(
        [
            "",
            "## Signed-story rows",
            "",
            "| Market | Window | Stories | Direction | Strongest simple control | Passed |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in direction.sort_values(
        ["meets_direction_rule", "direction_success_rate"], ascending=False
    ).itertuples(index=False):
        lines.append(
            f"| `{row.market_scope}` | {row.horizon_hours}h | {row.issued_events} | "
            f"{row.direction_success_rate:.1%} | {row.strongest_comparator_rate:.1%} | "
            f"`{bool(row.meets_direction_rule)}` |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This is market behaviour research, not profit or a trading rule.",
            "- Related markets and windows reuse the same stories and are not independent.",
            "- Unclear and mixed stories abstain from direction.",
            "- The sample contains no independent-source story confluence and only "
            "one major story.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_semantic_story_direct":
            raise ValueError("Existing semantic-story result is not terminal.")
        return result
    freeze, routes, controls = load_frozen_inputs()
    frames, market_coverage = load_market_frames(freeze)
    event_rows = extract_event_rows(freeze, routes, controls, frames)
    activity = summarize_activity(event_rows)
    direction = summarize_direction(event_rows)
    decisions = route_decisions(activity, direction)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(event_rows, EVENT_ROWS_PATH)
    g0.atomic_write_csv(market_coverage, MARKET_COVERAGE_PATH)
    g0.atomic_write_csv(activity, ACTIVITY_CELLS_PATH)
    g0.atomic_write_csv(direction, DIRECTION_CELLS_PATH)
    g0.atomic_write_csv(decisions, ROUTE_DECISIONS_PATH)
    REPORT_PATH.write_text(
        report_text(decisions, activity, direction),
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "schema_version": 1,
        "status": "completed_semantic_story_direct",
        "created_at_utc": g0.utc_now(),
        "outcomes_opened": True,
        "profit_used": False,
        "story_direction_guessed": False,
        "route_count": int(decisions["route_id"].nunique()),
        "event_rows": len(event_rows),
        "activity_cells": len(activity),
        "activity_passing_cells": int(activity["meets_activity_rule"].sum()),
        "direction_cells": len(direction),
        "direction_passing_cells": int(direction["meets_direction_rule"].sum()),
        "artifacts": {
            "market_coverage": artifact(MARKET_COVERAGE_PATH),
            "activity_cells": artifact(ACTIVITY_CELLS_PATH),
            "direction_cells": artifact(DIRECTION_CELLS_PATH),
            "route_decisions": artifact(ROUTE_DECISIONS_PATH),
            "report": artifact(REPORT_PATH),
        },
        "detail_artifact": artifact(EVENT_ROWS_PATH),
    }
    g0.atomic_write_json(result, RESULT_PATH)
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
                    "requires_terminal_freeze": str(frozen.RESULT_PATH),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
