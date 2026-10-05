"""Run the frozen CPI-to-crypto direction batch without profit optimization."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT_PATH = frozen.OUTPUT_ROOT / "cpi_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "cpi_freeze.json"
CATALOG_PATH = frozen.OUTPUT_ROOT / "cpi_release_catalog.csv"
FOMC_CATALOG_PATH = layer1.OUTPUT_ROOT / "official_fomc_event_catalog.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
DETAIL_ROOT = OUTPUT_ROOT / "details"

MINUTE_HORIZONS = (5, 15, 30, 60, 120, 240)
HOUR_HORIZONS = (1, 2, 4, 8, 24)
MINUTE_ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
EVALUATION_PARTITIONS = (
    "development_2021_2023",
    "internal_validation_2024_2025",
)
MIN_PARTITION_EVENTS = 10
LEAD_FLOOR = 0.55
STRONG_TARGET = 0.65
CONTROL_MARGIN = 0.02
ACTIVITY_CONTROL_WEEKS = 12
ACTIVITY_MINIMUM_CONTROLS = 6
EVENT_EXCLUSION_HOURS = 24


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    result = json.loads(FREEZE_RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_cpi_family_freeze":
        raise ValueError("CPI family freeze is not terminal.")
    if freeze.get("status") != "frozen_cpi_family_before_market_outcomes":
        raise ValueError("CPI definitions were not frozen before outcomes.")
    if freeze.get("outcomes_read"):
        raise ValueError("CPI freeze unexpectedly contains market outcomes.")
    if result["artifacts"]["freeze"]["sha256"] != g0.sha256_file(FREEZE_PATH):
        raise ValueError("CPI freeze changed after completion.")
    if result["artifacts"]["catalog"]["sha256"] != g0.sha256_file(CATALOG_PATH):
        raise ValueError("CPI catalog changed after completion.")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    for column in frozen.ALL_SIGN_COLUMNS:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    return result, freeze, catalog


def _sign(value: float) -> float:
    if pd.isna(value) or value == 0:
        return np.nan
    return float(1 if value > 0 else -1)


def _future_window(frame: DataFrame, position: int, length: int) -> dict[str, float] | None:
    if position < 0 or position + length > len(frame):
        return None
    window = frame.iloc[position : position + length]
    if len(window) != length:
        return None
    opening = float(window.iloc[0]["open"])
    if not np.isfinite(opening) or opening <= 0:
        return None
    return {
        "response_return": float(window.iloc[-1]["close"] / opening - 1.0),
        "response_abs_return": float(abs(window.iloc[-1]["close"] / opening - 1.0)),
        "response_range": float((window["high"].max() - window["low"].min()) / opening),
        "response_volume": float(window["volume"].sum()),
    }


def _pre_return(frame: DataFrame, position: int, length: int) -> float:
    if position - length < 0:
        return np.nan
    start = float(frame.iloc[position - length]["open"])
    end = float(frame.iloc[position]["open"])
    if not np.isfinite(start) or start <= 0:
        return np.nan
    return end / start - 1.0


def _position_by_date(frame: DataFrame) -> dict[pd.Timestamp, int]:
    return {stamp: int(position) for position, stamp in enumerate(frame["date"])}


def _minimum_event_distance_hours(
    anchors: Iterable[pd.Timestamp], target: pd.Timestamp
) -> float:
    distances = [abs((target - anchor).total_seconds()) / 3600 for anchor in anchors]
    return min(distances) if distances else np.inf


def _activity_control_metrics(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[dict[str, float]]:
    controls: list[dict[str, float]] = []
    for weeks in range(1, ACTIVITY_CONTROL_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if _minimum_event_distance_hours(blocked_anchors, control_anchor) <= EVENT_EXCLUSION_HOURS:
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        metrics = _future_window(frame, position, horizon)
        if metrics is not None:
            controls.append(metrics)
    return controls


def _activity_score(
    response: Mapping[str, float], controls: Sequence[Mapping[str, float]]
) -> float:
    if len(controls) < ACTIVITY_MINIMUM_CONTROLS:
        return np.nan
    ratios: list[float] = []
    for key in ("response_abs_return", "response_range", "response_volume"):
        baseline = float(np.median([float(control[key]) for control in controls]))
        if np.isfinite(baseline) and baseline > 0:
            ratios.append(float(response[key]) / baseline)
    return float(np.median(ratios)) if len(ratios) == 3 else np.nan


def fomc_collision_flags(catalog: DataFrame) -> Series:
    if not FOMC_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"FOMC catalog not found: {FOMC_CATALOG_PATH}")
    fomc = pd.read_csv(FOMC_CATALOG_PATH, usecols=["anchor_utc"])
    anchors = list(pd.to_datetime(fomc["anchor_utc"], utc=True))
    return catalog["anchor_utc"].map(
        lambda target: _minimum_event_distance_hours(anchors, target) <= 24
    )


def extract_minute_outcomes(
    catalog: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    cpi_anchors = list(catalog["anchor_utc"])
    fomc = pd.read_csv(FOMC_CATALOG_PATH, usecols=["anchor_utc"])
    blocked_anchors = cpi_anchors + list(pd.to_datetime(fomc["anchor_utc"], utc=True))
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in MINUTE_ASSETS:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = _position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
            }
        )
        for event in catalog.itertuples(index=False):
            anchor = pd.Timestamp(event.anchor_utc)
            position = positions.get(anchor)
            if position is None:
                continue
            for horizon in MINUTE_HORIZONS:
                response = _future_window(frame, position, horizon)
                if response is None:
                    continue
                controls = _activity_control_metrics(
                    frame,
                    positions,
                    anchor=anchor,
                    horizon=horizon,
                    blocked_anchors=blocked_anchors,
                )
                records.append(
                    {
                        "event_id": event.event_id,
                        "anchor_utc": anchor,
                        "whole_event_partition": event.whole_event_partition,
                        "fomc_within_24h": bool(event.fomc_within_24h),
                        "clock": "immediate_1m",
                        "scope": "btc" if pair.startswith("BTC/") else "eth",
                        "horizon": horizon,
                        "horizon_unit": "minutes",
                        **response,
                        "pre_return": _pre_return(frame, position, horizon),
                        "activity_control_count": len(controls),
                        "activity_score_vs_same_clock_weeks": _activity_score(
                            response, controls
                        ),
                        "available_coin_count": 1,
                        "required_coin_count": 1,
                    }
                )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def all_pairs_and_scopes() -> tuple[list[str], dict[str, list[str]]]:
    cohorts = layer1.frozen_cohorts()
    established = list(cohorts["established_alts"])
    memes = list(cohorts["top_ten_traded_memes"])
    normal = ["BTC/USDT:USDT", "ETH/USDT:USDT", *established, "DOGE/USDT:USDT"]
    normal = list(dict.fromkeys(normal))
    pairs = list(dict.fromkeys([*normal, *memes]))
    scopes = {
        "btc": ["BTC/USDT:USDT"],
        "eth": ["ETH/USDT:USDT"],
        "established_alt_median": established,
        "meme_median": memes,
        "broad_market_median": normal,
    }
    return pairs, scopes


def extract_hour_pair_outcomes(
    catalog: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    pairs, _ = all_pairs_and_scopes()
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in pairs:
        path = g0.ohlcv_path(pair, "1h")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = _position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1h",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
            }
        )
        for event in catalog.itertuples(index=False):
            event_anchor = pd.Timestamp(event.anchor_utc)
            anchor = event_anchor.ceil("h")
            position = positions.get(anchor)
            if position is None:
                continue
            for horizon in HOUR_HORIZONS:
                response = _future_window(frame, position, horizon)
                if response is None:
                    continue
                records.append(
                    {
                        "event_id": event.event_id,
                        "anchor_utc": event_anchor,
                        "outcome_anchor_utc": anchor,
                        "whole_event_partition": event.whole_event_partition,
                        "fomc_within_24h": bool(event.fomc_within_24h),
                        "pair": pair,
                        "horizon": horizon,
                        "response_return": response["response_return"],
                        "pre_return": _pre_return(frame, position, horizon),
                    }
                )
    return DataFrame.from_records(records), DataFrame.from_records(coverage)


def aggregate_hour_scopes(pair_rows: DataFrame) -> DataFrame:
    _, scopes = all_pairs_and_scopes()
    records: list[dict[str, Any]] = []
    keys = [
        "event_id",
        "anchor_utc",
        "outcome_anchor_utc",
        "whole_event_partition",
        "fomc_within_24h",
        "horizon",
    ]
    for scope, members in scopes.items():
        required = len(members)
        for group_key, group in pair_rows.loc[pair_rows["pair"].isin(members)].groupby(
            keys, sort=False, dropna=False
        ):
            available = int(group["pair"].nunique())
            minimum = required if scope != "meme_median" else 5
            if available < minimum:
                continue
            base = dict(zip(keys, group_key, strict=True))
            records.append(
                {
                    **base,
                    "clock": "post_initial_full_hours",
                    "scope": scope,
                    "horizon_unit": "hours",
                    "response_return": float(group["response_return"].median()),
                    "response_abs_return": float(group["response_return"].abs().median()),
                    "response_range": np.nan,
                    "response_volume": np.nan,
                    "pre_return": float(group["pre_return"].median()),
                    "activity_control_count": 0,
                    "activity_score_vs_same_clock_weeks": np.nan,
                    "available_coin_count": available,
                    "required_coin_count": required,
                }
            )
    return DataFrame.from_records(records)


def expand_direction_rows(outcomes: DataFrame, catalog: DataFrame) -> DataFrame:
    sign_values = catalog[["event_id", *frozen.ALL_SIGN_COLUMNS]].copy()
    records: list[DataFrame] = []
    for sign_id in frozen.ALL_SIGN_COLUMNS:
        signs = sign_values[["event_id", sign_id]].copy()
        signs["current_temperature_sign"] = signs[sign_id]
        signs["previous_temperature_sign"] = signs[sign_id].shift(1)
        signs = signs.drop(columns=[sign_id])
        joined = outcomes.merge(signs, on="event_id", how="left", validate="many_to_one")
        joined["sign_id"] = sign_id
        joined["predicted_direction"] = joined["current_temperature_sign"].map(
            lambda value: np.nan if pd.isna(value) or value == 0 else -float(value)
        )
        joined["previous_release_predicted_direction"] = joined[
            "previous_temperature_sign"
        ].map(lambda value: np.nan if pd.isna(value) or value == 0 else -float(value))
        joined["actual_direction"] = joined["response_return"].map(_sign)
        joined["pretrend_direction"] = joined["pre_return"].map(_sign)
        joined["direction_hit"] = joined["predicted_direction"].eq(
            joined["actual_direction"]
        ).where(joined["predicted_direction"].notna() & joined["actual_direction"].notna())
        joined["previous_release_hit"] = joined[
            "previous_release_predicted_direction"
        ].eq(joined["actual_direction"]).where(
            joined["previous_release_predicted_direction"].notna()
            & joined["actual_direction"].notna()
        )
        joined["pretrend_hit"] = joined["pretrend_direction"].eq(
            joined["actual_direction"]
        ).where(joined["pretrend_direction"].notna() & joined["actual_direction"].notna())
        records.append(joined)
    return pd.concat(records, ignore_index=True)


def _rate(values: Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if len(valid) else np.nan


def summarize_partition(group: DataFrame) -> dict[str, Any]:
    usable = group.loc[group["direction_hit"].notna()].copy()
    actual = usable["actual_direction"]
    always_up = float(actual.eq(1).mean()) if len(actual) else np.nan
    always_down = float(actual.eq(-1).mean()) if len(actual) else np.nan
    return {
        "n_events": int(usable["event_id"].nunique()),
        "accuracy": _rate(usable["direction_hit"]),
        "always_up_accuracy": always_up,
        "always_down_accuracy": always_down,
        "best_constant_accuracy": max(always_up, always_down)
        if len(actual)
        else np.nan,
        "pretrend_accuracy": _rate(usable["pretrend_hit"]),
        "previous_release_sign_accuracy": _rate(usable["previous_release_hit"]),
        "median_response_return": float(usable["response_return"].median())
        if len(usable)
        else np.nan,
    }


def direction_summaries(rows: DataFrame) -> DataFrame:
    keys = ["sample_variant", "clock", "scope", "horizon", "horizon_unit", "sign_id"]
    records: list[dict[str, Any]] = []
    variants = {
        "all_releases": rows,
        "exclude_fomc_within_24h": rows.loc[~rows["fomc_within_24h"]],
    }
    for variant, variant_rows in variants.items():
        variant_rows = variant_rows.copy()
        variant_rows["sample_variant"] = variant
        for key, group in variant_rows.groupby(keys[1:], sort=False, dropna=False):
            key_values = dict(zip(keys[1:], key, strict=True))
            for partition, partition_rows in group.groupby(
                "whole_event_partition", sort=False
            ):
                records.append(
                    {
                        "sample_variant": variant,
                        **key_values,
                        "partition": partition,
                        **summarize_partition(partition_rows),
                    }
                )
            evaluation = group.loc[
                group["whole_event_partition"].isin(EVALUATION_PARTITIONS)
            ]
            records.append(
                {
                    "sample_variant": variant,
                    **key_values,
                    "partition": "development_plus_validation",
                    **summarize_partition(evaluation),
                }
            )
    return DataFrame.from_records(records)


def route_decisions(summary: DataFrame) -> DataFrame:
    keys = ["sample_variant", "clock", "scope", "horizon", "horizon_unit", "sign_id"]
    records: list[dict[str, Any]] = []
    for key, group in summary.groupby(keys, sort=False, dropna=False):
        by_partition = group.set_index("partition")
        development = (
            by_partition.loc[EVALUATION_PARTITIONS[0]]
            if EVALUATION_PARTITIONS[0] in by_partition.index
            else None
        )
        validation = (
            by_partition.loc[EVALUATION_PARTITIONS[1]]
            if EVALUATION_PARTITIONS[1] in by_partition.index
            else None
        )
        combined = by_partition.loc["development_plus_validation"]
        n_development = int(development["n_events"]) if development is not None else 0
        n_validation = int(validation["n_events"]) if validation is not None else 0
        accuracy_development = float(development["accuracy"]) if development is not None else np.nan
        accuracy_validation = float(validation["accuracy"]) if validation is not None else np.nan
        accuracy_combined = float(combined["accuracy"])
        control_rates = [
            float(combined["best_constant_accuracy"]),
            float(combined["pretrend_accuracy"]),
            float(combined["previous_release_sign_accuracy"]),
        ]
        valid_controls = [value for value in control_rates if np.isfinite(value)]
        best_control = max(valid_controls) if valid_controls else np.nan
        enough = (
            n_development >= MIN_PARTITION_EVENTS
            and n_validation >= MIN_PARTITION_EVENTS
        )
        cross_period = (
            np.isfinite(accuracy_development)
            and np.isfinite(accuracy_validation)
            and accuracy_development >= LEAD_FLOOR
            and accuracy_validation >= LEAD_FLOOR
        )
        beats_controls = (
            np.isfinite(accuracy_combined)
            and np.isfinite(best_control)
            and accuracy_combined >= best_control + CONTROL_MARGIN
        )
        if not enough:
            verdict = "parked_insufficient_cross_period_events"
        elif not cross_period:
            verdict = "rejected_below_55_percent_cross_period"
        elif not beats_controls:
            verdict = "rejected_not_better_than_simple_controls"
        else:
            verdict = "retained_cross_period_direction_lead"
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_development": n_development,
                "n_validation": n_validation,
                "accuracy_development": accuracy_development,
                "accuracy_validation": accuracy_validation,
                "accuracy_combined": accuracy_combined,
                "best_simple_control_accuracy": best_control,
                "uplift_over_best_control": accuracy_combined - best_control
                if np.isfinite(accuracy_combined) and np.isfinite(best_control)
                else np.nan,
                "reaches_65_percent_both_periods": bool(
                    enough
                    and accuracy_development >= STRONG_TARGET
                    and accuracy_validation >= STRONG_TARGET
                    and beats_controls
                ),
                "verdict": verdict,
            }
        )
    return DataFrame.from_records(records)


def activity_summaries(minute_outcomes: DataFrame) -> DataFrame:
    usable = minute_outcomes.loc[
        minute_outcomes["activity_score_vs_same_clock_weeks"].notna()
    ].copy()
    records: list[dict[str, Any]] = []
    keys = ["scope", "horizon", "whole_event_partition"]
    for key, group in usable.groupby(keys, sort=False, dropna=False):
        scores = group["activity_score_vs_same_clock_weeks"]
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_events": int(group["event_id"].nunique()),
                "median_activity_score": float(scores.median()),
                "fraction_busier_than_same_clock_median": float(scores.gt(1).mean()),
            }
        )
    return DataFrame.from_records(records)


def render_report(decisions: DataFrame, result: Mapping[str, Any]) -> str:
    retained = decisions.loc[
        decisions["verdict"].eq("retained_cross_period_direction_lead")
    ].sort_values("uplift_over_best_control", ascending=False)
    lines = [
        "# CPI To Crypto Direct Review",
        "",
        f"- Frozen CPI events: `{result['event_count']}`",
        f"- Tested direction routes: `{result['route_count']}`",
        f"- Retained cross-period leads: `{result['retained_route_count']}`",
        "- Profit used: **No**",
        "- Historical market consensus used: **No**",
        "",
        "## Plain conclusion",
        "",
    ]
    if retained.empty:
        lines.append(
            "No frozen CPI-temperature rule cleared the 55% cross-period floor while also "
            "beating the simple direction controls. Published inflation change alone is not "
            "a retained direction signal in this batch."
        )
    else:
        lines.extend(
            [
                (
                    "The following leads cleared the frozen cross-period and control rules. "
                    "They remain research leads, not trading rules."
                ),
                "",
                "| Clock | Scope | Horizon | CPI description | Development | Validation | Uplift |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in retained.head(20).itertuples(index=False):
            lines.append(
                f"| {row.clock} | {row.scope} | {row.horizon} {row.horizon_unit} | "
                f"{row.sign_id} | {row.accuracy_development:.1%} | "
                f"{row.accuracy_validation:.1%} | {row.uplift_over_best_control:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This tests whether newly published inflation was hotter or cooler than the "
                "previous release. It does not test a true market surprise because a reliable "
                "historical consensus series was not available."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "cpi_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_cpi_direct_review":
            raise ValueError("Existing CPI direct result is not terminal.")
        return result

    _, freeze, catalog = load_frozen_inputs()
    catalog = catalog.copy()
    catalog["fomc_within_24h"] = fomc_collision_flags(catalog)
    minute_outcomes, minute_coverage = extract_minute_outcomes(catalog)
    hour_pair_outcomes, hour_coverage = extract_hour_pair_outcomes(catalog)
    hour_scopes = aggregate_hour_scopes(hour_pair_outcomes)
    outcomes = pd.concat([minute_outcomes, hour_scopes], ignore_index=True)
    direction_rows = expand_direction_rows(outcomes, catalog)
    summaries = direction_summaries(direction_rows)
    decisions = route_decisions(summaries)
    activity = activity_summaries(minute_outcomes)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    detail_paths = {
        "minute_outcomes": DETAIL_ROOT / "minute_outcomes.parquet",
        "hour_pair_outcomes": DETAIL_ROOT / "hour_pair_outcomes.parquet",
        "hour_scope_outcomes": DETAIL_ROOT / "hour_scope_outcomes.parquet",
        "direction_rows": DETAIL_ROOT / "direction_rows.parquet",
    }
    for frame, path in (
        (minute_outcomes, detail_paths["minute_outcomes"]),
        (hour_pair_outcomes, detail_paths["hour_pair_outcomes"]),
        (hour_scopes, detail_paths["hour_scope_outcomes"]),
        (direction_rows, detail_paths["direction_rows"]),
    ):
        g0.atomic_write_parquet(frame, path)
    summary_paths = {
        "direction_summary": OUTPUT_ROOT / "direction_summary.csv",
        "route_decisions": OUTPUT_ROOT / "route_decisions.csv",
        "activity_summary": OUTPUT_ROOT / "minute_activity_summary.csv",
        "data_coverage": OUTPUT_ROOT / "data_coverage.csv",
    }
    g0.atomic_write_csv(summaries, summary_paths["direction_summary"])
    g0.atomic_write_csv(decisions, summary_paths["route_decisions"])
    g0.atomic_write_csv(activity, summary_paths["activity_summary"])
    g0.atomic_write_csv(
        pd.concat([minute_coverage, hour_coverage], ignore_index=True),
        summary_paths["data_coverage"],
    )
    retained = decisions["verdict"].eq("retained_cross_period_direction_lead")
    result = {
        "schema_version": 1,
        "status": "completed_cpi_direct_review",
        "created_at_utc": g0.utc_now(),
        "event_count": len(catalog),
        "route_count": len(decisions),
        "retained_route_count": int(retained.sum()),
        "strong_route_count": int(decisions["reaches_65_percent_both_periods"].sum()),
        "profit_used": False,
        "freqai_used": False,
        "historical_market_consensus_used": False,
        "freeze_contract": artifact(FREEZE_PATH),
        "freeze_hypothesis": freeze["frozen_direct_batch"]["primary_hypothesis"],
        "summary_artifacts": {
            name: artifact(path) for name, path in summary_paths.items()
        },
        "detail_artifacts": {name: artifact(path) for name, path in detail_paths.items()},
    }
    report_path = OUTPUT_ROOT / "cpi_plain_review.md"
    report_path.write_text(render_report(decisions, result), encoding="utf-8", newline="\n")
    result["summary_artifacts"]["plain_review"] = artifact(report_path)
    result["analysis_script"] = artifact(ANALYSIS_PATH)
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
                    "status": "ready_not_executed",
                    "output_root": str(OUTPUT_ROOT),
                    "market_outcomes_will_be_read": True,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
