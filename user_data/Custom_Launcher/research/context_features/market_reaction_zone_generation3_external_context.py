from __future__ import annotations

# This bounded diagnostic is deliberately single-threaded.  It post-processes the
# frozen G3A matched pairs and must not compete with Freqtrade workers.
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
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_one_minute_replay import (  # noqa: E501
    artifact_record,
)


SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "g3h_btc_orderbook_pressure_20260814a"
DEFAULT_NORMAL_PAIRS = (
    Path(r"D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones")
    / "generation3_branches"
    / "g3a_thin_lvn_attribution"
    / "g3a_lvn_cluster_attribution_normal10_full_20260814b"
    / "g3a_independent_outcome_pairs.parquet"
)
DEFAULT_MEME_PAIRS = (
    Path(r"D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones")
    / "generation3_branches"
    / "g3a_thin_lvn_attribution"
    / "g3a_lvn_cluster_attribution_meme_full_20260814b"
    / "g3a_independent_outcome_pairs.parquet"
)
DEFAULT_ORDERBOOK = (
    REPO_ROOT
    / "user_data"
    / "orderbook_data"
    / "historical_bybit"
    / "features"
    / "orderbook_trader_state_1h_bybit_linear_latest.parquet"
)
DEFAULT_SOURCE_AUDIT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation3_branches"
    / "g3h_external_context"
    / "g3h_source_readiness_20260814a"
    / "source_coverage_audit_g3h_20260814a.csv"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation3_branches"
    / "g3h_external_context"
)
DEFAULT_ARTIFACT_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData\research_outputs\market_reaction_zones")
    / "generation3_branches"
    / "g3h_external_context"
)

QUESTION_ID = "g3a_lvn_thinness_after_surviving_cluster"
OUTCOME_EXPECTED_DIRECTION = {
    "contact_volume_ratio": 1.0,
    "dwell_fraction_h4": -1.0,
    "crossings_h4": -1.0,
    "volume_ratio_h48": 1.0,
}
PRIMARY_OUTCOME = "contact_volume_ratio"
EXCLUDED_PERIODS = {"previously_exposed_diagnostic"}
PRESSURE_LOOKBACK_HOURS = 24 * 30
PRESSURE_MINIMUM_HISTORY_HOURS = 24 * 7
STALE_CONTROL_HOURS = 24 * 7
ORDERBOOK_MINIMUM_COVERAGE = 0.80
ORDERBOOK_MAXIMUM_AGE_HOURS = 2.0
MINIMUM_DIAGNOSTIC_PAIRS = 30
MINIMUM_DIAGNOSTIC_COINS = 5
MINIMUM_PHASE_PAIRS = 10
MAXIMUM_ONE_COIN_SHARE = 0.35
LEAD_EXPECTED_SIGN_FRACTION = 0.55
MAIN_TARGET_EXPECTED_SIGN_FRACTION = 0.65
LEAVE_ONE_COIN_OUT_SIGN_FRACTION = 0.80

PAIR_COLUMNS = (
    "pair",
    "question_id",
    "attribute_assignment",
    "source_timeframe",
    "level_name",
    "zone_method",
    "period",
    "approach_state",
    "high_base_index",
    "low_base_index",
    "high_event_time",
    "low_event_time",
    "high_attribute_value",
    "low_attribute_value",
    "match_distance",
    "outcome",
    "metric_family",
    "horizon_hours",
    "raw_delta",
    "independent_selection_order",
    "independence_hours",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "G3H: condition corrected G3A thin-LVN matched pairs on causal BTC "
            "orderbook pressure, with a seven-day stale-context control."
        )
    )
    parser.add_argument("--normal-pairs", type=Path, default=DEFAULT_NORMAL_PAIRS)
    parser.add_argument("--meme-pairs", type=Path, default=DEFAULT_MEME_PAIRS)
    parser.add_argument("--orderbook", type=Path, default=DEFAULT_ORDERBOOK)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    contract = request_contract(args)
    if not args.execute:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    result = run_analysis(args, contract)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def request_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(args.run_id),
        "question": (
            "Within the corrected G3A pairs that already match local OHLCV, BTC "
            "OHLCV, breadth, level density, approach, period, level side, and cluster "
            "geometry, does the event occurring during active absolute BTC 25-basis-"
            "point orderbook pressure show a stronger reaction than its quiet-context "
            "partner?"
        ),
        "primary_hypothesis": (
            "The active-pressure event has a larger contact-hour volume ratio than "
            "the quiet-pressure event."
        ),
        "secondary_interpretation": {
            "dwell_fraction_h4": "lower means a cleaner departure from the zone",
            "crossings_h4": "lower means fewer revisits through the zone centre",
            "volume_ratio_h48": "higher means activity remains elevated later",
        },
        "not_tested": [
            "profitability",
            "trade entry or exit rules",
            "post-contact price direction",
            "local altcoin orderbooks",
            "causation",
        ],
        "source_definition": {
            "market": "Bybit BTC linear orderbook; global crypto context only",
            "value": "absolute mean bid-versus-ask pressure within 25 basis points",
            "active": "top third relative to the preceding 30 days of usable source hours",
            "quiet": "bottom third relative to the preceding 30 days of usable source hours",
            "middle": "recorded but excluded from the active-versus-quiet contrast",
            "minimum_history_hours": PRESSURE_MINIMUM_HISTORY_HOURS,
            "minimum_coverage": ORDERBOOK_MINIMUM_COVERAGE,
            "maximum_source_age_hours": ORDERBOOK_MAXIMUM_AGE_HOURS,
            "timestamp_rule": "source_max_ts must not be later than the event timestamp",
        },
        "controls": [
            "reuse corrected G3A matched event pairs on identical rows",
            "keep normal, normal excluding BTC, BTC-only, and meme views separate",
            "exclude the previously exposed diagnostic period",
            "do not count a source-missing row as quiet",
            "do not reuse an event inside one result slice",
            "keep every selected global event timestamp at least the outcome horizon apart",
            (
                "compare current pressure orientation with pressure from seven days "
                "earlier on the same rows"
            ),
            "report development and validation separately",
            "report leave-one-coin-out stability and maximum coin share",
        ],
        "diagnostic_lead_rule": {
            "scope": ["normal_ex_btc", "meme"],
            "minimum_pairs_all": MINIMUM_DIAGNOSTIC_PAIRS,
            "minimum_coins": MINIMUM_DIAGNOSTIC_COINS,
            "minimum_pairs_each_development_and_validation": MINIMUM_PHASE_PAIRS,
            "maximum_one_coin_share": MAXIMUM_ONE_COIN_SHARE,
            "minimum_expected_sign_fraction": LEAD_EXPECTED_SIGN_FRACTION,
            "repeat_positive_median": "development and aggregate validation",
            "stale_control": (
                "current expected-sign fraction must exceed the seven-day stale "
                "orientation"
            ),
            "minimum_leave_one_coin_out_positive_median_fraction": LEAVE_ONE_COIN_OUT_SIGN_FRACTION,
            "meaning": "queue-worthy context lead only; never automatic strategy promotion",
        },
        "main_programme_target_reference": MAIN_TARGET_EXPECTED_SIGN_FRACTION,
        "inputs": {
            "normal_pairs": str(args.normal_pairs),
            "meme_pairs": str(args.meme_pairs),
            "orderbook": str(args.orderbook),
            "source_audit": str(args.source_audit),
        },
    }


def run_analysis(
    args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any]:
    inputs = (args.normal_pairs, args.meme_pairs, args.orderbook, args.source_audit)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen G3H inputs: {missing}")

    report_dir = args.report_root / str(args.run_id)
    artifact_dir = args.artifact_root / str(args.run_id)
    record_path = report_dir / "g3h_run_record.json"
    contract_hash = stable_json_sha256(contract)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(args.run_id),
        "status": "running",
        "started_at_utc": utc_now(),
        "request_contract": contract,
        "request_sha256": contract_hash,
        "inputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in inputs
        },
    }
    atomic_write_json(record, record_path)

    try:
        orderbook = load_causal_pressure_surface(args.orderbook)
        normal = load_pair_rows(args.normal_pairs, cohort="normal")
        meme = load_pair_rows(args.meme_pairs, cohort="meme")
        pairs = pd.concat([normal, meme], ignore_index=True, sort=False)
        enriched = attach_all_orderbook_states(pairs, orderbook)
        overlap = source_overlap_inventory(enriched)

        selected_parts: list[DataFrame] = []
        result_parts: list[DataFrame] = []
        leave_one_out_parts: list[DataFrame] = []
        for view, view_frame in analysis_views(enriched).items():
            for outcome in OUTCOME_EXPECTED_DIRECTION:
                source = view_frame.loc[view_frame["outcome"].eq(outcome)].copy()
                candidates = active_quiet_candidates(source)
                selected = select_globally_independent(candidates)
                if not selected.empty:
                    selected["analysis_view"] = view
                    selected_parts.append(selected)
                results = summarize_selected(selected, view=view, outcome=outcome)
                result_parts.append(results)
                leave_one_out_parts.append(
                    leave_one_coin_out(selected, view=view, outcome=outcome)
                )

        selected_pairs = concat_or_empty(selected_parts)
        results = concat_or_empty(result_parts)
        leave_one_out = concat_or_empty(leave_one_out_parts)
        assessments = primary_assessments(results, leave_one_out)
        summary = build_summary(
            enriched=enriched,
            overlap=overlap,
            selected=selected_pairs,
            results=results,
            leave_one_out=leave_one_out,
            assessments=assessments,
        )

        overlap_path = report_dir / "g3h_source_overlap.csv"
        results_path = report_dir / "g3h_orderbook_results.csv"
        leave_one_out_path = report_dir / "g3h_leave_one_coin_out.csv"
        summary_path = report_dir / "g3h_summary.json"
        selected_path = artifact_dir / "g3h_selected_pair_details.csv"
        atomic_write_csv(overlap, overlap_path)
        atomic_write_csv(results, results_path)
        atomic_write_csv(leave_one_out, leave_one_out_path)
        atomic_write_csv(selected_pairs, selected_path)
        atomic_write_json(summary, summary_path)

        integrity = integrity_audit(enriched, selected_pairs)
        if integrity["source_future_violations"]:
            raise ValueError("Timestamp-unsafe orderbook rows reached the G3H analysis")
        if integrity["selected_exact_event_reuses"]:
            raise ValueError("An event was reused inside a G3H result slice")
        if integrity["selected_global_separation_violations"]:
            raise ValueError("A G3H result slice violates global time independence")

        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "integrity": integrity,
                "summary": summary,
                "artifacts": {
                    "source_overlap": artifact_record(overlap_path),
                    "results": artifact_record(results_path),
                    "leave_one_coin_out": artifact_record(leave_one_out_path),
                    "summary": artifact_record(summary_path),
                    "selected_pair_details": artifact_record(selected_path),
                },
            }
        )
        atomic_write_json(record, record_path)
        return {
            "status": "completed",
            "run_id": str(args.run_id),
            "record": str(record_path),
            "summary": summary,
        }
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise


def load_pair_rows(path: Path, *, cohort: str) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(PAIR_COLUMNS))
    frame = frame.loc[
        frame["question_id"].eq(QUESTION_ID)
        & frame["attribute_assignment"].eq("actual")
        & ~frame["period"].isin(EXCLUDED_PERIODS)
        & frame["outcome"].isin(OUTCOME_EXPECTED_DIRECTION)
    ].copy()
    frame["cohort"] = cohort
    for column in ("high_event_time", "low_event_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    numeric = (
        "high_base_index",
        "low_base_index",
        "match_distance",
        "raw_delta",
        "independent_selection_order",
        "independence_hours",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = [
        "pair",
        "period",
        "outcome",
        "high_event_time",
        "low_event_time",
        "raw_delta",
        "independence_hours",
    ]
    return frame.dropna(subset=required).reset_index(drop=True)


def load_causal_pressure_surface(path: Path) -> DataFrame:
    columns = [
        "date",
        "source_max_ts",
        "obts_feature_present",
        "obts_coverage_ratio",
        "obts_pressure_25bps_mean",
    ]
    frame = pd.read_parquet(path, columns=columns).sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["source_max_ts"] = pd.to_datetime(
        frame["source_max_ts"], utc=True, errors="coerce"
    )
    for column in (
        "obts_feature_present",
        "obts_coverage_ratio",
        "obts_pressure_25bps_mean",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")

    source_safe = (
        frame["source_max_ts"].notna()
        & frame["source_max_ts"].le(frame["date"])
        & frame["obts_feature_present"].eq(1.0)
        & frame["obts_coverage_ratio"].ge(ORDERBOOK_MINIMUM_COVERAGE)
    )
    frame["source_row_usable"] = source_safe
    frame["absolute_pressure"] = (
        frame["obts_pressure_25bps_mean"].abs().where(source_safe)
    )
    indexed = frame.set_index("date")["absolute_pressure"]
    rolling = indexed.rolling(
        f"{PRESSURE_LOOKBACK_HOURS}h",
        min_periods=PRESSURE_MINIMUM_HISTORY_HOURS,
        closed="left",
    )
    frame["pressure_lower_tertile"] = rolling.quantile(1.0 / 3.0).to_numpy()
    frame["pressure_upper_tertile"] = rolling.quantile(2.0 / 3.0).to_numpy()
    history_ready = frame[
        ["pressure_lower_tertile", "pressure_upper_tertile"]
    ].notna().all(axis=1)
    frame["pressure_regime"] = "unavailable"
    frame.loc[source_safe & history_ready, "pressure_regime"] = "middle"
    frame.loc[
        source_safe
        & history_ready
        & frame["absolute_pressure"].le(frame["pressure_lower_tertile"]),
        "pressure_regime",
    ] = "quiet"
    frame.loc[
        source_safe
        & history_ready
        & frame["absolute_pressure"].ge(frame["pressure_upper_tertile"]),
        "pressure_regime",
    ] = "active"
    return frame.reset_index(drop=True)


def attach_all_orderbook_states(pairs: DataFrame, orderbook: DataFrame) -> DataFrame:
    output = pairs.copy()
    for side in ("high", "low"):
        output = attach_orderbook_state(output, orderbook, side=side, label="current")
        output = attach_orderbook_state(
            output,
            orderbook,
            side=side,
            label="stale",
            lag_hours=STALE_CONTROL_HOURS,
        )
    return output


def attach_orderbook_state(
    pairs: DataFrame,
    orderbook: DataFrame,
    *,
    side: str,
    label: str,
    lag_hours: int = 0,
) -> DataFrame:
    event_column = f"{side}_event_time"
    prefix = f"{side}_{label}_"
    left = DataFrame(
        {
            "_row_order": np.arange(len(pairs), dtype=int),
            "target_time": pairs[event_column]
            - pd.to_timedelta(int(lag_hours), unit="h"),
        }
    ).sort_values("target_time")
    source_columns = [
        "date",
        "source_max_ts",
        "source_row_usable",
        "obts_coverage_ratio",
        "obts_pressure_25bps_mean",
        "absolute_pressure",
        "pressure_lower_tertile",
        "pressure_upper_tertile",
        "pressure_regime",
    ]
    merged = pd.merge_asof(
        left,
        orderbook[source_columns].sort_values("date"),
        left_on="target_time",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_order")
    merged = merged.reset_index(drop=True)
    age_hours = (merged["target_time"] - merged["date"]).dt.total_seconds() / 3600.0
    future = merged["source_max_ts"].gt(merged["target_time"])
    usable = (
        merged["source_row_usable"].fillna(False).astype(bool)
        & ~future.fillna(False)
        & age_hours.le(ORDERBOOK_MAXIMUM_AGE_HOURS)
        & merged["pressure_regime"].isin(["active", "middle", "quiet"])
    )

    output = pairs.copy()
    output[prefix + "target_time"] = merged["target_time"].to_numpy()
    output[prefix + "source_date"] = merged["date"].to_numpy()
    output[prefix + "source_max_ts"] = merged["source_max_ts"].to_numpy()
    output[prefix + "age_hours"] = age_hours.to_numpy(dtype=float)
    output[prefix + "coverage_ratio"] = pd.to_numeric(
        merged["obts_coverage_ratio"], errors="coerce"
    ).to_numpy(dtype=float)
    output[prefix + "signed_pressure"] = pd.to_numeric(
        merged["obts_pressure_25bps_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    output[prefix + "absolute_pressure"] = pd.to_numeric(
        merged["absolute_pressure"], errors="coerce"
    ).to_numpy(dtype=float)
    output[prefix + "lower_tertile"] = pd.to_numeric(
        merged["pressure_lower_tertile"], errors="coerce"
    ).to_numpy(dtype=float)
    output[prefix + "upper_tertile"] = pd.to_numeric(
        merged["pressure_upper_tertile"], errors="coerce"
    ).to_numpy(dtype=float)
    output[prefix + "regime"] = np.where(
        usable, merged["pressure_regime"].astype(str), "unavailable"
    )
    output[prefix + "usable"] = usable.to_numpy(dtype=bool)
    output[prefix + "future_violation"] = future.fillna(False).to_numpy(dtype=bool)
    return output


def analysis_views(frame: DataFrame) -> dict[str, DataFrame]:
    normal = frame.loc[frame["cohort"].eq("normal")]
    meme = frame.loc[frame["cohort"].eq("meme")]
    return {
        "normal_all": normal.copy(),
        "normal_ex_btc": normal.loc[~normal["pair"].eq("BTC/USDT:USDT")].copy(),
        "btc_only": normal.loc[normal["pair"].eq("BTC/USDT:USDT")].copy(),
        "meme": meme.copy(),
    }


def active_quiet_candidates(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame.copy()
    high = frame["high_current_regime"]
    low = frame["low_current_regime"]
    discordant = (high.eq("active") & low.eq("quiet")) | (
        high.eq("quiet") & low.eq("active")
    )
    output = frame.loc[discordant].copy()
    output["current_orientation"] = np.where(
        output["high_current_regime"].eq("active"), 1.0, -1.0
    )
    stale_delta = (
        pd.to_numeric(output["high_stale_absolute_pressure"], errors="coerce")
        - pd.to_numeric(output["low_stale_absolute_pressure"], errors="coerce")
    )
    stale_usable = output["high_stale_usable"] & output["low_stale_usable"]
    output["stale_orientation"] = np.where(
        stale_usable & stale_delta.gt(0.0),
        1.0,
        np.where(stale_usable & stale_delta.lt(0.0), -1.0, np.nan),
    )
    output["current_oriented_delta"] = (
        output["raw_delta"] * output["current_orientation"]
    )
    output["stale_oriented_delta"] = output["raw_delta"] * output["stale_orientation"]
    expected = float(OUTCOME_EXPECTED_DIRECTION[str(output["outcome"].iloc[0])])
    output["expected_direction"] = expected
    output["current_expected_oriented_delta"] = (
        output["current_oriented_delta"] * expected
    )
    output["stale_expected_oriented_delta"] = output["stale_oriented_delta"] * expected
    output["current_expected_sign"] = output[
        "current_expected_oriented_delta"
    ].gt(0.0)
    output["stale_expected_sign"] = output[
        "stale_expected_oriented_delta"
    ].gt(0.0).where(output["stale_orientation"].notna())
    output["active_absolute_pressure"] = np.where(
        output["current_orientation"].gt(0.0),
        output["high_current_absolute_pressure"],
        output["low_current_absolute_pressure"],
    )
    output["quiet_absolute_pressure"] = np.where(
        output["current_orientation"].gt(0.0),
        output["low_current_absolute_pressure"],
        output["high_current_absolute_pressure"],
    )
    return output


def select_globally_independent(candidates: DataFrame) -> DataFrame:
    if candidates.empty:
        return candidates.copy()
    selected_rows: list[Series] = []
    for period, period_frame in candidates.groupby("period", sort=True, observed=True):
        used_events: set[tuple[str, int]] = set()
        used_times: list[pd.Timestamp] = []
        ordered = period_frame.sort_values(
            [
                "match_distance",
                "independent_selection_order",
                "pair",
                "high_event_time",
                "low_event_time",
            ],
            kind="mergesort",
        )
        for _, row in ordered.iterrows():
            horizon = float(row["independence_hours"])
            event_keys = (
                (str(row["pair"]), int(row["high_base_index"])),
                (str(row["pair"]), int(row["low_base_index"])),
            )
            event_times = (
                pd.Timestamp(row["high_event_time"]),
                pd.Timestamp(row["low_event_time"]),
            )
            if event_keys[0] in used_events or event_keys[1] in used_events:
                continue
            if abs((event_times[0] - event_times[1]).total_seconds()) < horizon * 3600.0:
                continue
            if any(
                abs((event_time - used_time).total_seconds()) < horizon * 3600.0
                for event_time in event_times
                for used_time in used_times
            ):
                continue
            selected_rows.append(row)
            used_events.update(event_keys)
            used_times.extend(event_times)
    if not selected_rows:
        return candidates.iloc[0:0].copy()
    output = DataFrame(selected_rows).reset_index(drop=True)
    output["global_independent_selection_order"] = (
        output.groupby("period", observed=True).cumcount() + 1
    )
    return output


def summarize_selected(selected: DataFrame, *, view: str, outcome: str) -> DataFrame:
    slices: list[tuple[str, DataFrame]] = [("all", selected)]
    slices.append(
        (
            "phase_development",
            selected.loc[selected["period"].astype(str).str.contains("development")]
            if not selected.empty
            else selected,
        )
    )
    slices.append(
        (
            "phase_validation",
            selected.loc[selected["period"].astype(str).str.contains("validation")]
            if not selected.empty
            else selected,
        )
    )
    if not selected.empty:
        for period in sorted(selected["period"].astype(str).unique()):
            slices.append(
                (f"period_{period}", selected.loc[selected["period"].eq(period)])
            )
    rows = [
        summary_row(frame, view=view, outcome=outcome, slice_name=name)
        for name, frame in slices
    ]
    return DataFrame(rows)


def summary_row(
    frame: DataFrame, *, view: str, outcome: str, slice_name: str
) -> dict[str, Any]:
    current = pd.to_numeric(frame.get("current_oriented_delta"), errors="coerce")
    current_expected = pd.to_numeric(
        frame.get("current_expected_oriented_delta"), errors="coerce"
    )
    stale = pd.to_numeric(frame.get("stale_oriented_delta"), errors="coerce")
    stale_expected = pd.to_numeric(
        frame.get("stale_expected_oriented_delta"), errors="coerce"
    )
    pair_counts = frame["pair"].value_counts() if "pair" in frame else Series(dtype=float)
    current_valid = current_expected.dropna()
    stale_valid = stale_expected.dropna()
    return {
        "analysis_view": view,
        "outcome": outcome,
        "analysis_slice": slice_name,
        "selected_pairs": len(frame),
        "coins": int(frame["pair"].nunique()) if "pair" in frame else 0,
        "maximum_coin_share": (
            float(pair_counts.max() / len(frame)) if len(frame) and len(pair_counts) else math.nan
        ),
        "current_active_minus_quiet_median": finite_median(current),
        "current_active_minus_quiet_mean": finite_mean(current),
        "current_expected_oriented_median": finite_median(current_expected),
        "current_expected_sign_fraction": (
            float(current_valid.gt(0.0).mean()) if len(current_valid) else math.nan
        ),
        "stale_oriented_median": finite_median(stale),
        "stale_expected_oriented_median": finite_median(stale_expected),
        "stale_expected_sign_fraction": (
            float(stale_valid.gt(0.0).mean()) if len(stale_valid) else math.nan
        ),
        "current_minus_stale_expected_sign_fraction": safe_difference(
            float(current_valid.gt(0.0).mean()) if len(current_valid) else math.nan,
            float(stale_valid.gt(0.0).mean()) if len(stale_valid) else math.nan,
        ),
        "stale_usable_pairs": int(stale_valid.notna().sum()),
        "active_absolute_pressure_median": finite_median(
            pd.to_numeric(frame.get("active_absolute_pressure"), errors="coerce")
        ),
        "quiet_absolute_pressure_median": finite_median(
            pd.to_numeric(frame.get("quiet_absolute_pressure"), errors="coerce")
        ),
        "expected_direction": float(OUTCOME_EXPECTED_DIRECTION[outcome]),
    }


def leave_one_coin_out(
    selected: DataFrame, *, view: str, outcome: str
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty or selected["pair"].nunique() < 2:
        return DataFrame(rows)
    slices = {
        "all": selected,
        "phase_development": selected.loc[
            selected["period"].astype(str).str.contains("development")
        ],
        "phase_validation": selected.loc[
            selected["period"].astype(str).str.contains("validation")
        ],
    }
    for slice_name, source in slices.items():
        for omitted in sorted(source["pair"].astype(str).unique()):
            kept = source.loc[~source["pair"].eq(omitted)]
            expected = pd.to_numeric(
                kept["current_expected_oriented_delta"], errors="coerce"
            ).dropna()
            rows.append(
                {
                    "analysis_view": view,
                    "outcome": outcome,
                    "analysis_slice": slice_name,
                    "omitted_pair": omitted,
                    "selected_pairs": len(kept),
                    "coins": int(kept["pair"].nunique()),
                    "expected_oriented_median": finite_median(expected),
                    "expected_sign_fraction": (
                        float(expected.gt(0.0).mean()) if len(expected) else math.nan
                    ),
                }
            )
    return DataFrame(rows)


def primary_assessments(results: DataFrame, leave_one_out: DataFrame) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for view in ("normal_ex_btc", "meme"):
        subset = results.loc[
            results["analysis_view"].eq(view) & results["outcome"].eq(PRIMARY_OUTCOME)
        ].set_index("analysis_slice")
        if not {"all", "phase_development", "phase_validation"}.issubset(subset.index):
            assessments.append({"analysis_view": view, "classification": "insufficient"})
            continue
        all_row = subset.loc["all"]
        development = subset.loc["phase_development"]
        validation = subset.loc["phase_validation"]
        loo = leave_one_out.loc[
            leave_one_out["analysis_view"].eq(view)
            & leave_one_out["outcome"].eq(PRIMARY_OUTCOME)
            & leave_one_out["analysis_slice"].eq("all")
        ]
        loo_positive = (
            float(pd.to_numeric(loo["expected_oriented_median"], errors="coerce").gt(0.0).mean())
            if len(loo)
            else math.nan
        )
        sample_ok = bool(
            int(all_row["selected_pairs"]) >= MINIMUM_DIAGNOSTIC_PAIRS
            and int(all_row["coins"]) >= MINIMUM_DIAGNOSTIC_COINS
            and int(development["selected_pairs"]) >= MINIMUM_PHASE_PAIRS
            and int(validation["selected_pairs"]) >= MINIMUM_PHASE_PAIRS
            and float(all_row["maximum_coin_share"]) <= MAXIMUM_ONE_COIN_SHARE
        )
        repeat_sign = bool(
            float(development["current_expected_oriented_median"]) > 0.0
            and float(validation["current_expected_oriented_median"]) > 0.0
        )
        fraction_ok = bool(
            float(all_row["current_expected_sign_fraction"])
            >= LEAD_EXPECTED_SIGN_FRACTION
        )
        stale_ok = bool(
            float(all_row["current_minus_stale_expected_sign_fraction"]) > 0.0
        )
        loo_ok = bool(
            np.isfinite(loo_positive)
            and loo_positive >= LEAVE_ONE_COIN_OUT_SIGN_FRACTION
        )
        if sample_ok and repeat_sign and fraction_ok and stale_ok and loo_ok:
            classification = "retain_context_lead"
        elif sample_ok and (repeat_sign or fraction_ok or stale_ok):
            classification = "mixed_keep_for_joint_review"
        elif sample_ok:
            classification = "not_supported_in_this_bounded_test"
        else:
            classification = "insufficient_independent_pairs"
        assessments.append(
            {
                "analysis_view": view,
                "classification": classification,
                "sample_rule_passed": sample_ok,
                "development_and_validation_median_repeat": repeat_sign,
                "expected_sign_fraction_rule_passed": fraction_ok,
                "current_beats_stale_rule_passed": stale_ok,
                "leave_one_coin_out_rule_passed": loo_ok,
                "selected_pairs": int(all_row["selected_pairs"]),
                "coins": int(all_row["coins"]),
                "maximum_coin_share": finite_or_none(all_row["maximum_coin_share"]),
                "development_expected_oriented_median": finite_or_none(
                    development["current_expected_oriented_median"]
                ),
                "validation_expected_oriented_median": finite_or_none(
                    validation["current_expected_oriented_median"]
                ),
                "current_expected_sign_fraction": finite_or_none(
                    all_row["current_expected_sign_fraction"]
                ),
                "stale_expected_sign_fraction": finite_or_none(
                    all_row["stale_expected_sign_fraction"]
                ),
                "leave_one_coin_out_positive_median_fraction": finite_or_none(
                    loo_positive
                ),
                "descriptive_65_percent_target_met": bool(
                    float(all_row["current_expected_sign_fraction"])
                    >= MAIN_TARGET_EXPECTED_SIGN_FRACTION
                ),
            }
        )
    return assessments


def source_overlap_inventory(frame: DataFrame) -> DataFrame:
    source = frame.loc[frame["outcome"].eq(PRIMARY_OUTCOME)]
    event_parts: list[DataFrame] = []
    for side in ("high", "low"):
        part = DataFrame(
            {
                "cohort": source["cohort"],
                "period": source["period"],
                "pair": source["pair"],
                "base_index": source[f"{side}_base_index"],
                "event_time": source[f"{side}_event_time"],
                "orderbook_usable": source[f"{side}_current_usable"],
                "orderbook_regime": source[f"{side}_current_regime"],
                "source_future_violation": source[
                    f"{side}_current_future_violation"
                ],
            }
        )
        event_parts.append(part)
    events = pd.concat(event_parts, ignore_index=True).drop_duplicates(
        ["cohort", "pair", "base_index", "event_time"]
    )
    event_time = pd.to_datetime(events["event_time"], utc=True)
    gdelt = validated_gdelt_mask(event_time)
    gkg = validated_gkg_mask(event_time)
    events["validated_gdelt_aggregate_window"] = gdelt
    events["validated_gkg_aggregate_window"] = gkg
    events["validated_gdelt_and_bybit_window"] = gdelt & events["orderbook_usable"]

    rows: list[dict[str, Any]] = []
    for (cohort, period), group in events.groupby(
        ["cohort", "period"], sort=True, observed=True
    ):
        rows.append(
            {
                "cohort": cohort,
                "period": period,
                "unique_matched_events": len(group),
                "pairs": int(group["pair"].nunique()),
                "orderbook_usable_events": int(group["orderbook_usable"].sum()),
                "orderbook_active_events": int(group["orderbook_regime"].eq("active").sum()),
                "orderbook_quiet_events": int(group["orderbook_regime"].eq("quiet").sum()),
                "orderbook_middle_events": int(group["orderbook_regime"].eq("middle").sum()),
                "source_future_violations": int(group["source_future_violation"].sum()),
                "validated_gdelt_aggregate_events": int(
                    group["validated_gdelt_aggregate_window"].sum()
                ),
                "validated_gkg_aggregate_events": int(
                    group["validated_gkg_aggregate_window"].sum()
                ),
                "validated_gdelt_and_bybit_events": int(
                    group["validated_gdelt_and_bybit_window"].sum()
                ),
            }
        )
    return DataFrame(rows)


def validated_gdelt_mask(dates: Series) -> Series:
    dates = pd.to_datetime(dates, utc=True, errors="coerce")
    return (
        dates.between("2020-01-01", "2020-10-31 23:59:59", inclusive="both")
        | dates.between("2021-03-01", "2022-12-31 23:59:59", inclusive="both")
        | dates.between("2023-01-18 07:00:00", "2023-06-30 23:59:59", inclusive="both")
    )


def validated_gkg_mask(dates: Series) -> Series:
    dates = pd.to_datetime(dates, utc=True, errors="coerce")
    return dates.between(
        "2020-01-01", "2020-10-31 23:59:59", inclusive="both"
    ) | dates.between("2021-03-01", "2022-07-31 23:59:59", inclusive="both")


def build_summary(
    *,
    enriched: DataFrame,
    overlap: DataFrame,
    selected: DataFrame,
    results: DataFrame,
    leave_one_out: DataFrame,
    assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = results.loc[results["outcome"].eq(PRIMARY_OUTCOME)]
    primary_all = primary.loc[primary["analysis_slice"].eq("all")]
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "question": (
            "Does causal BTC orderbook pressure condition the reaction of corrected "
            "G3A matched thin-LVN cluster contacts?"
        ),
        "pair_rows_after_period_filter": len(enriched),
        "source_overlap_rows": len(overlap),
        "selected_detail_rows_across_views_and_outcomes": len(selected),
        "result_rows": len(results),
        "leave_one_coin_out_rows": len(leave_one_out),
        "primary_all_views": primary_all.to_dict(orient="records"),
        "primary_assessments": assessments,
        "interpretation_guardrails": [
            "The orderbook belongs to BTC, not the sampled altcoin.",
            (
                "The comparison conditions already matched G3A event pairs; it does "
                "not estimate an unconditional reaction rate."
            ),
            "Active means unusually large absolute pressure, not bullish pressure.",
            "This branch studies reaction activity and zone cleanliness, not direction or profit.",
            (
                "A retained result is a lead for a later frozen confirmation batch, "
                "not a trading rule."
            ),
        ],
    }


def integrity_audit(enriched: DataFrame, selected: DataFrame) -> dict[str, Any]:
    future_columns = [
        column for column in enriched if column.endswith("_future_violation")
    ]
    source_future_violations = int(
        sum(enriched[column].fillna(False).astype(bool).sum() for column in future_columns)
    )
    exact_reuses = 0
    separation_violations = 0
    if not selected.empty:
        for _, group in selected.groupby(
            ["analysis_view", "outcome", "period"], observed=True
        ):
            keys: list[tuple[str, int]] = []
            timed: list[tuple[pd.Timestamp, float]] = []
            for _, row in group.iterrows():
                keys.extend(
                    [
                        (str(row["pair"]), int(row["high_base_index"])),
                        (str(row["pair"]), int(row["low_base_index"])),
                    ]
                )
                timed.extend(
                    [
                        (pd.Timestamp(row["high_event_time"]), float(row["independence_hours"])),
                        (pd.Timestamp(row["low_event_time"]), float(row["independence_hours"])),
                    ]
                )
            exact_reuses += len(keys) - len(set(keys))
            for left in range(len(timed)):
                for right in range(left + 1, len(timed)):
                    difference = abs((timed[left][0] - timed[right][0]).total_seconds())
                    required = max(timed[left][1], timed[right][1]) * 3600.0
                    if difference < required:
                        separation_violations += 1
    return {
        "source_future_violations": source_future_violations,
        "selected_exact_event_reuses": int(exact_reuses),
        "selected_global_separation_violations": int(separation_violations),
        "normal_pair_rows": int(enriched["cohort"].eq("normal").sum()),
        "meme_pair_rows": int(enriched["cohort"].eq("meme").sum()),
        "selected_rows": len(selected),
    }


def concat_or_empty(frames: list[DataFrame]) -> DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    return pd.concat(usable, ignore_index=True, sort=False) if usable else DataFrame()


def finite_median(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if len(numeric) else math.nan


def finite_mean(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.mean()) if len(numeric) else math.nan


def safe_difference(left: float, right: float) -> float:
    return float(left - right) if np.isfinite(left) and np.isfinite(right) else math.nan


def finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


if __name__ == "__main__":
    raise SystemExit(main())
