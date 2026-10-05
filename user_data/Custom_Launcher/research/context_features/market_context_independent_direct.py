"""Measure market activity and bounded direction for frozen independent sources."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
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
    market_context_independent_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as cohort_source,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.OUTPUT_ROOT / "independent_context_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "independent_context_freeze.json"
EVENTS_PATH = frozen.OUTPUT_ROOT / "source_event_catalog.csv"
CONTROLS_PATH = frozen.OUTPUT_ROOT / "source_control_catalog.csv"
OBSERVATIONS_PATH = frozen.OUTPUT_ROOT / "source_observations.parquet"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"

MARKET_SCOPES: dict[str, tuple[str, ...]] = {
    "btc": ("BTC/USDT:USDT",),
    "eth": ("ETH/USDT:USDT",),
    "established_alts": cohort_source.GROUPS["established_altcoins"],
    "top_ten_memes": cohort_source.GROUPS["frozen_top_ten_memes"],
}
MINIMUM_SCOPE_MEMBERS = {
    "btc": 1,
    "eth": 1,
    "established_alts": 6,
    "top_ten_memes": 7,
}
HISTORICAL_PARTITIONS = {
    "development_2021_2023",
    "internal_validation_2024_2025",
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_independent_context_source_freeze":
        raise ValueError("Independent context-source freeze is not terminal.")
    if freeze.get("status") != "frozen_independent_context_sources_before_market_outcomes":
        raise ValueError("Independent context-source definitions are not frozen.")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Independent context-source freeze unexpectedly read outcomes.")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("events", EVENTS_PATH),
        ("controls", CONTROLS_PATH),
        ("observations", OBSERVATIONS_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen independent-source artifact changed: {path}")
    events = pd.read_csv(EVENTS_PATH)
    controls = pd.read_csv(CONTROLS_PATH)
    for frame, columns in (
        (events, ("anchor_utc", "available_at")),
        (controls, ("event_anchor_utc", "control_anchor_utc")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(
                frame[column], utc=True, errors="raise", format="mixed"
            )
    coverage_ready = events["coverage_ready"]
    if not pd.api.types.is_bool_dtype(coverage_ready.dtype):
        coverage_ready = coverage_ready.astype(str).str.casefold().eq("true")
    events = events.loc[coverage_ready].copy()
    return freeze, events, controls


def load_market_frames() -> tuple[dict[str, DataFrame], DataFrame]:
    frames: dict[str, DataFrame] = {}
    coverage: list[dict[str, Any]] = []
    pairs = sorted({pair for members in MARKET_SCOPES.values() for pair in members})
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicate_count = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
        frame["_position"] = np.arange(len(frame), dtype=int)
        frames[pair] = frame.set_index("date", drop=False)
        coverage.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "rows": len(frame),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "duplicate_timestamps_removed": duplicate_count,
            }
        )
    return frames, DataFrame.from_records(coverage)


def sign(value: Any) -> int:
    if value is None or pd.isna(value) or float(value) == 0.0:
        return 0
    return 1 if float(value) > 0.0 else -1


def window_metrics(
    frame: DataFrame, anchor: pd.Timestamp, horizon: int
) -> dict[str, float] | None:
    if anchor not in frame.index:
        return None
    position_value = frame.at[anchor, "_position"]
    if isinstance(position_value, pd.Series):
        return None
    position = int(position_value)
    if position < 25 or position + horizon > len(frame):
        return None
    window = frame.iloc[position : position + horizon]
    opening = float(window.iloc[0]["open"])
    if not np.isfinite(opening) or opening <= 0:
        return None
    recent_end = float(frame.iloc[position - 1]["close"])
    recent_start = float(frame.iloc[position - 25]["close"])
    if not np.isfinite(recent_start) or recent_start <= 0:
        return None
    close_return = float(window.iloc[-1]["close"] / opening - 1.0)
    return {
        "signed_return": close_return,
        "abs_return": abs(close_return),
        "range": float((window["high"].max() - window["low"].min()) / opening),
        "volume": float(window["volume"].sum()),
        "recent_24h_return": recent_end / recent_start - 1.0,
    }


def ratio(value: float, controls: Sequence[float]) -> float:
    finite = [float(item) for item in controls if np.isfinite(item)]
    if len(finite) < frozen.MINIMUM_CONTROL_COUNT:
        return np.nan
    baseline = float(np.median(finite))
    return value / baseline if np.isfinite(baseline) and baseline > 0 else np.nan


def pair_event_metrics(
    frame: DataFrame,
    event_anchor: pd.Timestamp,
    control_anchors: Sequence[pd.Timestamp],
    horizon: int,
) -> dict[str, float] | None:
    event = window_metrics(frame, event_anchor, horizon)
    if event is None:
        return None
    controls = [
        metrics
        for anchor in control_anchors
        if (metrics := window_metrics(frame, pd.Timestamp(anchor), horizon)) is not None
    ]
    if len(controls) < frozen.MINIMUM_CONTROL_COUNT:
        return None
    ratios = {
        name: ratio(float(event[name]), [float(row[name]) for row in controls])
        for name in ("abs_return", "range", "volume")
    }
    if not all(np.isfinite(value) for value in ratios.values()):
        return None
    return {
        **event,
        "abs_return_ratio": ratios["abs_return"],
        "range_ratio": ratios["range"],
        "volume_ratio": ratios["volume"],
        "activity_score": float(np.median(list(ratios.values()))),
        "control_count": len(controls),
    }


def scope_from_pairs(
    scope: str, pair_rows: Sequence[Mapping[str, float]]
) -> dict[str, float] | None:
    if len(pair_rows) < MINIMUM_SCOPE_MEMBERS[scope]:
        return None
    activity = [float(row["activity_score"]) for row in pair_rows]
    return {
        "member_count": len(pair_rows),
        "signed_return": float(np.mean([row["signed_return"] for row in pair_rows])),
        "recent_24h_return": float(
            np.mean([row["recent_24h_return"] for row in pair_rows])
        ),
        "abs_return_ratio": float(np.median([row["abs_return_ratio"] for row in pair_rows])),
        "range_ratio": float(np.median([row["range_ratio"] for row in pair_rows])),
        "volume_ratio": float(np.median([row["volume_ratio"] for row in pair_rows])),
        "activity_score": float(np.median(activity)),
        "minimum_pair_control_count": min(int(row["control_count"]) for row in pair_rows),
    }


def parse_horizons(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split(";") if str(item).strip())


def extract_live_source_outcomes(
    events: DataFrame, controls: DataFrame, frames: Mapping[str, DataFrame]
) -> DataFrame:
    control_map = {
        event_id: tuple(group["control_anchor_utc"])
        for event_id, group in controls.groupby("event_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_controls = control_map.get(event.event_id, ())
        for horizon in parse_horizons(event.horizons_hours):
            pair_metrics: dict[str, dict[str, float]] = {}
            for pair, frame in frames.items():
                metrics = pair_event_metrics(
                    frame,
                    pd.Timestamp(event.anchor_utc),
                    event_controls,
                    horizon,
                )
                if metrics is not None:
                    pair_metrics[pair] = metrics

            if event.target_kind == "btc_relative_to_groups":
                btc = (
                    scope_from_pairs("btc", [pair_metrics["BTC/USDT:USDT"]])
                    if "BTC/USDT:USDT" in pair_metrics
                    else None
                )
                for group_scope in ("established_alts", "top_ten_memes"):
                    group_rows = [
                        pair_metrics[pair]
                        for pair in MARKET_SCOPES[group_scope]
                        if pair in pair_metrics
                    ]
                    group = scope_from_pairs(
                        group_scope,
                        group_rows,
                    )
                    if btc is None or group is None:
                        continue
                    records.append(
                        base_record(event, horizon)
                        | {
                            "market_scope": f"btc_relative_to_{group_scope}",
                            "member_count": 1 + int(group["member_count"]),
                            "signed_return": btc["signed_return"] - group["signed_return"],
                            "recent_24h_return": (
                                btc["recent_24h_return"] - group["recent_24h_return"]
                            ),
                            "abs_return_ratio": np.nan,
                            "range_ratio": np.nan,
                            "volume_ratio": np.nan,
                            "activity_score": np.nan,
                            "minimum_pair_control_count": min(
                                int(btc["minimum_pair_control_count"]),
                                int(group["minimum_pair_control_count"]),
                            ),
                        }
                    )
                continue

            for scope, members in MARKET_SCOPES.items():
                scoped = scope_from_pairs(
                    scope,
                    [pair_metrics[pair] for pair in members if pair in pair_metrics],
                )
                if scoped is None:
                    continue
                records.append(base_record(event, horizon) | {"market_scope": scope} | scoped)
    return DataFrame.from_records(records)


def base_record(event: Any, horizon: int) -> dict[str, Any]:
    return {
        "source_batch": "live_independent_sources",
        "event_id": event.event_id,
        "family": event.family,
        "family_label": event.family_label,
        "event_anchor_utc": event.anchor_utc,
        "whole_event_partition": event.whole_event_partition,
        "source_role": event.source_role,
        "source_signal": event.source_signal,
        "source_direction": int(event.source_direction),
        "direction_relation": event.direction_relation,
        "predicted_direction": int(event.predicted_direction),
        "target_kind": event.target_kind,
        "horizon_hours": horizon,
    }


def historical_control_anchors(
    anchor: pd.Timestamp, blocked: Sequence[pd.Timestamp]
) -> tuple[pd.Timestamp, ...]:
    output: list[pd.Timestamp] = []
    for weeks in range(1, 27):
        candidate = anchor - pd.Timedelta(weeks=weeks)
        if any(abs(candidate - event) <= pd.Timedelta(hours=168) for event in blocked):
            continue
        output.append(candidate)
        if len(output) >= frozen.CONTROL_COUNT:
            break
    return tuple(output)


def extract_historical_cross_asset_outcomes(
    freeze: Mapping[str, Any], frames: Mapping[str, DataFrame]
) -> DataFrame:
    contract = freeze["historical_cross_asset_contract"]
    catalog_path = Path(contract["catalog"]["path"])
    if contract["catalog"]["sha256"] != g0.sha256_file(catalog_path):
        raise ValueError("Frozen historical cross-asset catalog changed.")
    catalog = pd.read_csv(catalog_path)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    catalog = catalog.loc[catalog["whole_event_partition"].isin(HISTORICAL_PARTITIONS)]
    direction_contract = contract["direction_contract"]
    records: list[dict[str, Any]] = []
    for family, family_events in catalog.groupby("event_family", sort=False):
        blocked = tuple(family_events["anchor_utc"])
        relation = str(direction_contract[family])
        for event in family_events.itertuples(index=False):
            source_direction = sign(event.source_change)
            if relation == "same":
                prediction = source_direction
            elif relation == "opposite":
                prediction = -source_direction
            else:
                prediction = 0
            controls = historical_control_anchors(pd.Timestamp(event.anchor_utc), blocked)
            for horizon in (4, 8, 24, 72):
                for scope in ("btc", "eth"):
                    pair = MARKET_SCOPES[scope][0]
                    metrics = pair_event_metrics(
                        frames[pair], pd.Timestamp(event.anchor_utc), controls, horizon
                    )
                    if metrics is None:
                        continue
                    records.append(
                        {
                            "source_batch": "historical_cross_asset",
                            "event_id": event.event_id,
                            "family": f"historical_{family}",
                            "family_label": event.event_label,
                            "event_anchor_utc": event.anchor_utc,
                            "whole_event_partition": event.whole_event_partition,
                            "source_role": "historical_external_market_shock",
                            "source_signal": event.source_change,
                            "source_direction": source_direction,
                            "direction_relation": relation,
                            "predicted_direction": prediction,
                            "target_kind": "market",
                            "horizon_hours": horizon,
                            "market_scope": scope,
                            "member_count": 1,
                            **metrics,
                            "minimum_pair_control_count": int(metrics["control_count"]),
                        }
                    )
    return DataFrame.from_records(records)


def add_result_flags(outcomes: DataFrame) -> DataFrame:
    out = outcomes.copy()
    out["reaction_success"] = out["activity_score"].ge(frozen.ACTIVITY_MULTIPLE).where(
        out["activity_score"].notna()
    )
    out["actual_direction"] = out["signed_return"].map(sign)
    out["issued_direction_call"] = out["predicted_direction"].ne(0)
    valid_direction = out["issued_direction_call"] & out["actual_direction"].ne(0)
    out["direction_success"] = (
        out["predicted_direction"].eq(out["actual_direction"]).where(valid_direction)
    )
    trend_direction = out["recent_24h_return"].map(sign)
    out["recent_trend_success"] = trend_direction.eq(out["actual_direction"]).where(
        trend_direction.ne(0) & out["actual_direction"].ne(0)
    )
    out["rotated_prediction"] = 0
    grouping = [
        "source_batch",
        "family",
        "market_scope",
        "horizon_hours",
        "whole_event_partition",
    ]
    for _keys, indexes in out.groupby(grouping, sort=False).groups.items():
        ordered = out.loc[indexes].sort_values("event_anchor_utc", kind="stable")
        predictions = ordered["predicted_direction"].to_numpy(dtype=int)
        if len(predictions):
            out.loc[ordered.index, "rotated_prediction"] = np.roll(predictions, 1)
    out["rotated_direction_success"] = (
        out["rotated_prediction"].eq(out["actual_direction"])
        .where(out["rotated_prediction"].ne(0) & out["actual_direction"].ne(0))
    )
    out = add_majority_control(out)
    out["joint_success"] = (
        out["reaction_success"].eq(True) & out["direction_success"].eq(True)
    ).where(out["reaction_success"].notna() & out["direction_success"].notna())
    return out


def add_majority_control(outcomes: DataFrame) -> DataFrame:
    out = outcomes.copy()
    out["development_majority_direction"] = 0
    keys = ["source_batch", "family", "market_scope", "horizon_hours"]
    for _values, indexes in out.groupby(keys, sort=False).groups.items():
        group = out.loc[indexes]
        development = group.loc[
            group["whole_event_partition"].astype(str).str.startswith("development")
        ]
        directions = development.loc[
            development["actual_direction"].ne(0), "actual_direction"
        ]
        if directions.empty:
            continue
        majority = 1 if float(directions.mean()) >= 0 else -1
        out.loc[indexes, "development_majority_direction"] = majority
    out["majority_direction_success"] = (
        out["development_majority_direction"].eq(out["actual_direction"])
        .where(
            out["development_majority_direction"].ne(0)
            & out["actual_direction"].ne(0)
        )
    )
    return out


def summarize(outcomes: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "source_batch",
        "family",
        "family_label",
        "source_role",
        "market_scope",
        "horizon_hours",
        "whole_event_partition",
    ]
    for values, group in outcomes.groupby(keys, dropna=False, sort=False):
        calls = group.loc[group["issued_direction_call"]]
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "whole_events": int(group["event_id"].nunique()),
                "median_activity_multiple": group["activity_score"].median(),
                "reaction_success_rate": group["reaction_success"].mean(),
                "direction_calls": int(calls["event_id"].nunique()),
                "direction_success_rate": calls["direction_success"].mean(),
                "recent_trend_success_rate": calls["recent_trend_success"].mean(),
                "majority_direction_success_rate": calls[
                    "majority_direction_success"
                ].mean(),
                "rotated_direction_success_rate": calls[
                    "rotated_direction_success"
                ].mean(),
                "joint_success_rate": calls["joint_success"].mean(),
                "abstentions": int((~group["issued_direction_call"]).sum()),
                "median_signed_return": group["signed_return"].median(),
                "minimum_member_count": int(group["member_count"].min()),
            }
        )
    return DataFrame.from_records(rows)


def partition_pair(rows: DataFrame) -> tuple[DataFrame, DataFrame] | None:
    development = rows.loc[
        rows["whole_event_partition"].astype(str).str.startswith("development")
    ]
    validation = rows.loc[
        rows["whole_event_partition"].astype(str).str.contains("validation")
    ]
    if development.empty or validation.empty:
        return None
    return development.iloc[[0]], validation.iloc[[0]]


def comparator_max(row: pd.Series) -> float:
    values = [
        row["recent_trend_success_rate"],
        row["majority_direction_success_rate"],
        row["rotated_direction_success_rate"],
    ]
    finite = [float(value) for value in values if pd.notna(value)]
    return max(finite) if finite else np.nan


def maximum_finite(*values: Any) -> float:
    finite = [float(value) for value in values if pd.notna(value)]
    return max(finite) if finite else np.nan


def classify_cell(dev: pd.Series, val: pd.Series) -> dict[str, Any]:
    enough = (
        int(dev["whole_events"]) >= frozen.MINIMUM_EVENT_COUNT
        and int(val["whole_events"]) >= frozen.MINIMUM_EVENT_COUNT
    )
    if not enough:
        return {"supported": False}

    activity_pass = all(
        pd.notna(row["reaction_success_rate"])
        and float(row["reaction_success_rate"]) >= frozen.MINIMUM_ACTIVITY_RATE
        and float(row["median_activity_multiple"]) >= frozen.ACTIVITY_MULTIPLE
        for row in (dev, val)
    )
    enough_direction_calls = (
        int(dev["direction_calls"]) >= frozen.MINIMUM_EVENT_COUNT
        and int(val["direction_calls"]) >= frozen.MINIMUM_EVENT_COUNT
    )
    direction_pass = enough_direction_calls and all(
        pd.notna(row["direction_success_rate"])
        and float(row["direction_success_rate"]) >= frozen.MINIMUM_DIRECTION_RATE
        and pd.notna(comparator_max(row))
        and float(row["direction_success_rate"]) - comparator_max(row)
        >= frozen.MINIMUM_DIRECTION_EDGE
        for row in (dev, val)
    )
    joint_pass = activity_pass and direction_pass and all(
        pd.notna(row["joint_success_rate"])
        and float(row["joint_success_rate"]) >= frozen.MINIMUM_DIRECTION_RATE
        for row in (dev, val)
    )
    return {
        "supported": True,
        "activity_pass": activity_pass,
        "direction_pass": direction_pass,
        "joint_pass": joint_pass,
        "validation_direction": (
            float(val["direction_success_rate"])
            if int(val["direction_calls"]) >= frozen.MINIMUM_EVENT_COUNT
            else np.nan
        ),
        "validation_joint": (
            float(val["joint_success_rate"])
            if int(val["direction_calls"]) >= frozen.MINIMUM_EVENT_COUNT
            else np.nan
        ),
    }


def family_verdict(
    *,
    joint_cells: list[str],
    activity_cells: list[str],
    direction_cells: list[str],
    supported_cells: int,
    best_validation_direction: float,
) -> str:
    if joint_cells:
        return "repeatable_joint_activity_and_direction_lead"
    if activity_cells and direction_cells:
        return "repeatable_activity_and_separate_direction_lead"
    if activity_cells:
        return "repeatable_activity_only_lead"
    if direction_cells:
        return "repeatable_direction_only_lead"
    if supported_cells == 0:
        return "insufficient_two_partition_coverage"
    if (
        pd.notna(best_validation_direction)
        and float(best_validation_direction) >= frozen.MINIMUM_DIRECTION_RATE
    ):
        return "one_period_or_control_limited_direction_watchlist"
    return "weak_or_inconsistent"


def classify(summary: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for (source_batch, family, label, role), family_rows in summary.groupby(
        ["source_batch", "family", "family_label", "source_role"], sort=False
    ):
        activity_cells: list[str] = []
        direction_cells: list[str] = []
        joint_cells: list[str] = []
        supported_cells = 0
        best_validation_direction = np.nan
        best_validation_joint = np.nan
        for (scope, horizon), cell in family_rows.groupby(
            ["market_scope", "horizon_hours"], sort=False
        ):
            pair = partition_pair(cell)
            if pair is None:
                continue
            development, validation = pair
            dev = development.iloc[0]
            val = validation.iloc[0]
            result = classify_cell(dev, val)
            if not result["supported"]:
                continue
            supported_cells += 1
            label_cell = f"{scope}:{int(horizon)}h"
            if result["activity_pass"]:
                activity_cells.append(label_cell)
            best_validation_direction = maximum_finite(
                best_validation_direction, result["validation_direction"]
            )
            if result["direction_pass"]:
                direction_cells.append(label_cell)
            best_validation_joint = maximum_finite(
                best_validation_joint, result["validation_joint"]
            )
            if result["joint_pass"]:
                joint_cells.append(label_cell)

        verdict = family_verdict(
            joint_cells=joint_cells,
            activity_cells=activity_cells,
            direction_cells=direction_cells,
            supported_cells=supported_cells,
            best_validation_direction=best_validation_direction,
        )
        records.append(
            {
                "source_batch": source_batch,
                "family": family,
                "family_label": label,
                "source_role": role,
                "verdict": verdict,
                "supported_scope_horizon_cells": supported_cells,
                "activity_cells": "; ".join(activity_cells),
                "direction_cells": "; ".join(direction_cells),
                "joint_cells": "; ".join(joint_cells),
                "best_validation_direction_rate": best_validation_direction,
                "best_validation_joint_rate": best_validation_joint,
            }
        )
    return DataFrame.from_records(records).sort_values(
        ["source_batch", "family"], kind="stable"
    )


def render_report(verdicts: DataFrame) -> str:
    rows = [
        "# Independent Context-Source Direct Review",
        "",
        "Each source was tested alone before any source combination or level interaction.",
        "",
        "| Source question | Plain result | Activity cells | Direction cells |",
        "|---|---|---|---|",
    ]
    for row in verdicts.itertuples(index=False):
        rows.append(
            f"| {row.family_label} | `{row.verdict}` | "
            f"{row.activity_cells or '-'} | {row.direction_cells or '-'} |"
        )
    rows.extend(
        [
            "",
            "Reaction means unusually large absolute movement, range, and volume versus "
            "the frozen ordinary-time controls. Direction is scored separately.",
            "",
            "News, web, and Google attention abstain from direction because the frozen "
            "source data contains no trustworthy story sign.",
            "",
            "No result here is a profit, entry, exit, or trading rule.",
            "",
        ]
    )
    return "\n".join(rows)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "independent_context_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_independent_context_direct_review":
            raise ValueError("Existing independent-source direct result is not terminal.")
        return result
    freeze, events, controls = load_frozen_inputs()
    frames, coverage = load_market_frames()
    live = extract_live_source_outcomes(events, controls, frames)
    historical = extract_historical_cross_asset_outcomes(freeze, frames)
    outcomes = add_result_flags(pd.concat([live, historical], ignore_index=True))
    summary = summarize(outcomes)
    verdicts = classify(summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes_path = OUTPUT_ROOT / "event_scope_outcomes.parquet"
    summary_path = OUTPUT_ROOT / "source_scope_summary.csv"
    verdicts_path = OUTPUT_ROOT / "source_verdicts.csv"
    coverage_path = OUTPUT_ROOT / "market_coverage.csv"
    report_path = OUTPUT_ROOT / "independent_context_plain_review.md"
    g0.atomic_write_parquet(outcomes, outcomes_path)
    g0.atomic_write_csv(summary, summary_path)
    g0.atomic_write_csv(verdicts, verdicts_path)
    g0.atomic_write_csv(coverage, coverage_path)
    report_path.write_text(render_report(verdicts), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_independent_context_direct_review",
        "created_at_utc": g0.utc_now(),
        "freeze_sha256": g0.sha256_file(FREEZE_PATH),
        "source_families_completed": int(verdicts["family"].nunique()),
        "live_source_families_completed": int(
            verdicts.loc[
                verdicts["source_batch"].eq("live_independent_sources"), "family"
            ].nunique()
        ),
        "historical_source_families_completed": int(
            verdicts.loc[
                verdicts["source_batch"].eq("historical_cross_asset"), "family"
            ].nunique()
        ),
        "verdict_counts": verdicts["verdict"].value_counts().to_dict(),
        "reaction_definition": (
            "Median of absolute-return, full-range, and volume multiples is at least "
            f"{frozen.ACTIVITY_MULTIPLE:.2f} versus frozen ordinary-time controls."
        ),
        "direction_definition": (
            "Predeclared source sign must reach 55% in both chronological partitions "
            "and beat recent-trend, development-majority, and rotated-source controls "
            "by at least three percentage points."
        ),
        "artifacts": {
            "outcomes": artifact(outcomes_path),
            "summary": artifact(summary_path),
            "verdicts": artifact(verdicts_path),
            "coverage": artifact(coverage_path),
            "report": artifact(report_path),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, result_path)
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
                    "mode": "setup_only",
                    "freeze": str(FREEZE_PATH),
                    "output_root": str(OUTPUT_ROOT),
                    "profit_or_trading_target": False,
                },
                indent=2,
            )
        )
        return 0
    result = execute(overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
