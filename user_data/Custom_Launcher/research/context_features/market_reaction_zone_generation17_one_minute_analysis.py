"""Run Generation 17's frozen 40-episode one-minute direction diagnostic."""

from __future__ import annotations

# The frozen replay is intentionally single-worker.
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay_analysis as g3a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_one_minute_analysis as g9a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_one_minute_analysis as g16a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_one_minute_acquisition as g17q,
)


DEFAULT_RUN_ID = "g17_one_minute_direction_20260822a"
REPORT_ROOT = g17z.OUTPUT_ROOT / "one_minute_direction"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation17_branches"
    / "g17_broad_branch_layer"
    / "one_minute_direction"
)
COVERAGE_RECORD_NAME = "g17_one_minute_coverage_record.json"
WINDOW_HOURS = {"1h": (12, 12), "4h": (48, 48), "8h": (96, 96)}
HORIZONS = (15, 60, 240, 720)
JOINT_FLOOR = 0.55
JOINT_TARGET = 0.65

EXTRA_METHODS = (
    "approach_rejection",
    "tf_1m_di_direction",
    "tf_5m_di_direction",
    "tf_15m_di_direction",
    "tf_1h_di_direction",
    "multi_timeframe_di_vote",
    "multi_timeframe_cmf_vote",
    "multi_timeframe_obv_vote",
    "multi_timeframe_rsi_vote",
    "multi_timeframe_macd_vote",
    "rsi_1m_extreme_mean_reversion",
    "rsi_5m_extreme_mean_reversion",
    "bollinger_1m_extreme_mean_reversion",
    "bollinger_5m_extreme_mean_reversion",
    "range_1m_edge_mean_reversion",
    "range_5m_edge_mean_reversion",
    "density_pressure_agreement",
    "density_approach_rejection",
    "btc_orderbook_pressure_slope_if_usable",
    "btc_orderbook_nearest_wall_if_usable",
    "contact_orderbook_pressure_agreement",
)
BASE_METHODS = tuple(
    method
    for method in g16a.CALL_METHODS
    if method != "leave_one_out_cohort_majority_comparator"
)
CALL_METHODS = (*BASE_METHODS, *EXTRA_METHODS)


def artifact(path: Path) -> dict[str, Any]:
    return g16a.artifact(path)


def validate_inputs() -> tuple[dict[str, Any], Path, Path]:
    freeze = g17q.prepare_freeze_record()
    freeze_path = g17q.FREEZE_ROOT / g17q.FREEZE_RECORD_NAME
    if freeze.get("status") != "frozen_before_one_minute_paths":
        raise ValueError("Generation 17 one-minute sample is not frozen.")
    sample_record = freeze["artifacts"]["sample_csv"]
    sample_path = Path(sample_record["path"])
    if not sample_path.is_file() or g0.sha256_file(sample_path) != sample_record["sha256"]:
        raise ValueError("Generation 17 analysis sample changed after freeze.")
    coverage_path = g17q.FREEZE_ROOT / COVERAGE_RECORD_NAME
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("status") != "passed":
        raise ValueError("Generation 17 one-minute coverage has not passed.")
    if coverage.get("selection_request_sha256") != freeze.get("request_sha256"):
        raise ValueError("Generation 17 coverage does not match the sample freeze.")
    return {**freeze, "_path": str(freeze_path.resolve())}, sample_path, coverage_path


def request_contract(freeze: dict[str, Any], coverage_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "objective": (
            "Test a broad predeclared set of causal one-minute direction rules on 40 "
            "density-balanced calculated-area contacts."
        ),
        "freeze": artifact(Path(freeze["_path"])),
        "coverage": artifact(coverage_path),
        "horizons_minutes": list(HORIZONS),
        "visible_windows_hours": WINDOW_HOURS,
        "call_methods": list(CALL_METHODS),
        "abstentions_count_as_failures": True,
        "acceptable_joint_floor": JOINT_FLOOR,
        "main_joint_target": JOINT_TARGET,
        "selection_used_future_reaction_or_direction": False,
        "freqai_training": False,
        "profit_used": False,
        "promotion_allowed": False,
        "worker_count": 1,
    }


def analyze_episode(
    episode: Series, *, surfaces: g3a.PairSurfaces, orderbook: DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    contact_time = g3a.first_parent_zone_contact(episode, surfaces.minute)
    paths = g16a.path_rows(
        episode=episode, minute=surfaces.minute, contact_time=contact_time
    )
    state = {
        **g3a.match_snapshot(surfaces.match, contact_time),
        **g3a.technical_snapshot(surfaces.technical, contact_time),
        **g16a.contact_state(surfaces.minute, contact_time),
        **g3a.orderbook_snapshot(orderbook, contact_time),
    }
    identity = {
        **g16a.episode_identity(episode),
        "density_regime": str(episode["density_regime"]),
        "density_score": float(episode["density_score"]),
        "contacted_level_count": float(
            episode["g11_minimal_contact__contacted_level_count"]
        ),
        "distinct_family_count": float(
            episode["g11_minimal_contact__distinct_family_count"]
        ),
    }
    actual = {
        **identity,
        "contact_time": contact_time,
        "decision_time": contact_time + pd.Timedelta(minutes=1),
        "reference_price": float(episode["level_price"]),
        **state,
    }
    outcomes = [
        {**identity, "contact_time": contact_time, **outcome}
        for outcome in g16a.horizon_outcomes(paths)
    ]
    boundary = g3a.boundary_audit(
        episode=episode,
        contact_time=contact_time,
        minute=surfaces.minute,
        window_hours=WINDOW_HOURS,
    )
    return actual, outcomes, paths, boundary


def threshold_call(value: object, lower: float, upper: float, *, reverse: bool) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(numeric):
        return 0
    call = -1 if numeric < lower else 1 if numeric > upper else 0
    return -call if reverse else call


def count_vote(value: object, *, total: int = 6) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(numeric):
        return 0
    return 1 if numeric >= math.ceil(total * 2 / 3) else -1 if numeric <= total // 3 else 0


def approach_call(state: object) -> int:
    # A contact approached from below is resistance and therefore rejects down;
    # a contact approached from above is support and therefore rejects up.
    return -1 if state == "from_below" else 1 if state == "from_above" else 0


def wall_call(event: Series) -> int:
    if not bool(event.get("btc_orderbook_usable_coverage", False)):
        return 0
    support = pd.to_numeric(
        Series([event.get("btc_orderbook_support_distance_bps")]), errors="coerce"
    ).iloc[0]
    resistance = pd.to_numeric(
        Series([event.get("btc_orderbook_resistance_distance_bps")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(support) or not np.isfinite(resistance) or support == resistance:
        return 0
    return 1 if support < resistance else -1


def causal_calls(actual: DataFrame) -> DataFrame:
    base = g16a.causal_calls(actual)
    rows: list[dict[str, Any]] = []
    for _, event in actual.iterrows():
        approach = approach_call(event["approach_state"])
        pressure_vote = g9a.multi_timeframe_vote(event, "pressure_20bar")
        orderbook_pressure = 0
        orderbook_slope = 0
        if bool(event.get("btc_orderbook_usable_coverage", False)):
            orderbook_pressure = g9a.finite_sign(
                event.get("btc_orderbook_pressure_25bps_last")
            )
            orderbook_slope = g9a.finite_sign(
                event.get("btc_orderbook_pressure_25bps_slope")
            )
        high_density = str(event["density_regime"]) == "high"
        methods = {
            "approach_rejection": approach,
            "tf_1m_di_direction": g9a.finite_sign(event["tf_1m__di_spread14"]),
            "tf_5m_di_direction": g9a.finite_sign(event["tf_5m__di_spread14"]),
            "tf_15m_di_direction": g9a.finite_sign(event["tf_15m__di_spread14"]),
            "tf_1h_di_direction": g9a.finite_sign(event["tf_1h__di_spread14"]),
            "multi_timeframe_di_vote": g9a.multi_timeframe_vote(event, "di_spread14"),
            "multi_timeframe_cmf_vote": g9a.multi_timeframe_vote(event, "cmf20"),
            "multi_timeframe_obv_vote": g9a.multi_timeframe_vote(event, "obv_trend20"),
            "multi_timeframe_rsi_vote": count_vote(event["mtf_rsi_above_50_count"]),
            "multi_timeframe_macd_vote": count_vote(event["mtf_macd_positive_count"]),
            "rsi_1m_extreme_mean_reversion": threshold_call(
                event["tf_1m__rsi14"], 30.0, 70.0, reverse=True
            ),
            "rsi_5m_extreme_mean_reversion": threshold_call(
                event["tf_5m__rsi14"], 30.0, 70.0, reverse=True
            ),
            "bollinger_1m_extreme_mean_reversion": threshold_call(
                event["tf_1m__bollinger_position20"], 0.0, 1.0, reverse=True
            ),
            "bollinger_5m_extreme_mean_reversion": threshold_call(
                event["tf_5m__bollinger_position20"], 0.0, 1.0, reverse=True
            ),
            "range_1m_edge_mean_reversion": threshold_call(
                event["tf_1m__range_position20"], 0.10, 0.90, reverse=True
            ),
            "range_5m_edge_mean_reversion": threshold_call(
                event["tf_5m__range_position20"], 0.10, 0.90, reverse=True
            ),
            "density_pressure_agreement": (
                pressure_vote if high_density else 0
            ),
            "density_approach_rejection": approach if high_density else 0,
            "btc_orderbook_pressure_slope_if_usable": orderbook_slope,
            "btc_orderbook_nearest_wall_if_usable": wall_call(event),
            "contact_orderbook_pressure_agreement": g16a.agreement_call(
                g9a.finite_sign(event["contact_candle_pressure"]), orderbook_pressure
            ),
        }
        for method, call in methods.items():
            rows.append(
                {
                    "episode_id": event["episode_id"],
                    "cohort": event["cohort"],
                    "period": event["period"],
                    "pair": event["pair"],
                    "method": method,
                    "method_kind": "causal_candidate",
                    "call_numeric": int(call),
                    "call_direction": "up" if call > 0 else "down" if call < 0 else "abstain",
                    "issued": call != 0,
                }
            )
    additions = DataFrame.from_records(rows)
    calls = pd.concat([base, additions], ignore_index=True)
    observed = set(calls["method"].astype(str))
    if observed != set(CALL_METHODS):
        raise ValueError(f"Generation 17 call registry drift: {sorted(observed)}")
    return calls


def decision_table(scores: DataFrame) -> DataFrame:
    causal = scores.loc[scores["method_kind"].eq("causal_candidate")].copy()
    rows: list[dict[str, Any]] = []
    for (horizon, method), frame in causal.groupby(
        ["horizon_minutes", "method"], observed=True, sort=False
    ):
        by_scope = {str(row.scope): row for row in frame.itertuples(index=False)}
        all_row = by_scope["all"]
        normal = by_scope.get("normal")
        meme = by_scope.get("meme")
        all_rate = float(all_row.joint_success_rate_all_episodes)
        normal_rate = (
            float(normal.joint_success_rate_all_episodes) if normal is not None else np.nan
        )
        meme_rate = (
            float(meme.joint_success_rate_all_episodes) if meme is not None else np.nan
        )
        broad = bool(
            all_rate >= JOINT_FLOOR
            and normal_rate >= JOINT_FLOOR
            and meme_rate >= JOINT_FLOOR
        )
        cohort_specific = bool(
            not broad and (normal_rate >= JOINT_FLOOR or meme_rate >= JOINT_FLOOR)
        )
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "method": method,
                "status": (
                    "broad_point_lead_pending_new_confirmation"
                    if broad
                    else "cohort_point_lead_pending_new_confirmation"
                    if cohort_specific
                    else "not_retained"
                ),
                "joint_rate_all": all_rate,
                "joint_rate_normal": normal_rate,
                "joint_rate_meme": meme_rate,
                "issued_call_coverage_all": float(all_row.issued_call_coverage),
                "abstentions_count_as_failures": True,
                "broad_point_lead": broad,
                "cohort_specific_point_lead": cohort_specific,
                "uncertainty_supported_above_chance": bool(
                    float(all_row.joint_wilson_lower) > 0.50
                ),
                "at_or_above_65pct": bool(all_rate >= JOINT_TARGET),
            }
        )
    return DataFrame.from_records(rows).sort_values(
        ["joint_rate_all", "issued_call_coverage_all"], ascending=False
    )


def compact_summary(
    actual: DataFrame,
    outcomes: DataFrame,
    scores: DataFrame,
    decisions: DataFrame,
    boundaries: DataFrame,
) -> dict[str, Any]:
    causal = scores.loc[
        scores["scope"].eq("all") & scores["method_kind"].eq("causal_candidate")
    ].sort_values(
        ["joint_success_rate_all_episodes", "issued_call_coverage"], ascending=False
    )
    return {
        "status": "completed_generation17_one_minute_direction_diagnostic",
        "actual_episodes": len(actual),
        "cohort_pair_units": len(actual[["cohort", "pair"]].drop_duplicates()),
        "distinct_symbols": int(actual["pair"].nunique()),
        "normal_episodes": int(actual["cohort"].eq("normal").sum()),
        "meme_episodes": int(actual["cohort"].eq("meme").sum()),
        "high_density_episodes": int(actual["density_regime"].eq("high").sum()),
        "low_density_episodes": int(actual["density_regime"].eq("low").sum()),
        "horizons_minutes": list(HORIZONS),
        "direction_methods": len(CALL_METHODS),
        "best_observed_causal_method": causal.iloc[0].to_dict() if len(causal) else {},
        "broad_methods_at_or_above_55pct": int(decisions["broad_point_lead"].sum()),
        "cohort_specific_methods_at_or_above_55pct": int(
            decisions["cohort_specific_point_lead"].sum()
        ),
        "methods_at_or_above_65pct": int(decisions["at_or_above_65pct"].sum()),
        "uncertainty_supported_above_chance": int(
            decisions["uncertainty_supported_above_chance"].sum()
        ),
        "boundary_extension_indicated": int(
            boundaries["boundary_extension_indicated"].sum()
        ),
        "interpretation": (
            "Forty frozen episodes can expose leads but cannot establish a trading rule. "
            "Any 55% result must survive a newly frozen confirmation sample; reaction-only "
            "and conditional direction remain separate from joint success."
        ),
        "freqai_trained": False,
        "profit_used": False,
        "promotion_allowed": False,
    }


def run_analysis(run_id: str, *, overwrite: bool = False) -> int:
    freeze, sample_path, coverage_path = validate_inputs()
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g17_one_minute_analysis_record.json"
    if record_path.is_file():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            existing.get("status")
            == "completed_generation17_one_minute_direction_diagnostic"
            and not overwrite
        ):
            print(json.dumps(existing, indent=2))
            return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request = request_contract(freeze, coverage_path)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "request_contract": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest(),
    }
    g0.atomic_write_json(record, record_path)
    try:
        sample = pd.read_csv(sample_path)
        for column in ("event_time", "source_open", "source_available_at"):
            sample[column] = pd.to_datetime(sample[column], utc=True, errors="raise")
        coverage = g3a.audit_analysis_coverage(sample=sample, window_hours=WINDOW_HOURS)
        if not bool(coverage["passed"].all()):
            raise ValueError("Generation 17 effective analysis windows are incomplete.")
        orderbook = g3a.load_orderbook_surface()
        actual_rows: list[dict[str, Any]] = []
        outcome_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        for _, episode in sample.sort_values("sample_selection_order").iterrows():
            surfaces = g3a.load_pair_surfaces(
                str(episode["pair"]), cohort=str(episode["cohort"])
            )
            actual, outcomes, paths, boundary = analyze_episode(
                episode, surfaces=surfaces, orderbook=orderbook
            )
            actual_rows.append(actual)
            outcome_rows.extend(outcomes)
            path_rows.extend(paths)
            boundary_rows.append(boundary)
            del surfaces
        actual = DataFrame.from_records(actual_rows).sort_values("sample_selection_order")
        outcomes = DataFrame.from_records(outcome_rows).sort_values(
            ["sample_selection_order", "horizon_minutes"]
        )
        paths = DataFrame.from_records(path_rows).sort_values(
            ["episode_id", "checkpoint_minutes"]
        )
        boundaries = DataFrame.from_records(boundary_rows).sort_values(
            "sample_selection_order"
        )
        calls = g16a.evaluated_calls(causal_calls(actual), outcomes)
        scores = g16a.call_summary(calls)
        causal_score = scores["method_kind"].eq("causal_candidate")
        enough = scores["independent_episodes"].ge(20)
        lead = scores["joint_success_rate_all_episodes"].ge(JOINT_FLOOR)
        scores.loc[causal_score, "status"] = "not_retained"
        scores.loc[
            causal_score & enough & lead, "status"
        ] = "point_lead_pending_new_confirmation"
        scores.loc[
            ~causal_score, "status"
        ] = "outcome_only_non_deployable_comparator"
        decisions = decision_table(scores)
        periods = g16a.period_summary(outcomes)
        summary = compact_summary(actual, outcomes, scores, decisions, boundaries)
        outputs = {
            "contact_state": artifact_dir / "g17_one_minute_contact_state.csv",
            "horizon_outcomes": artifact_dir / "g17_one_minute_horizon_outcomes.csv",
            "checkpoint_paths": artifact_dir / "g17_one_minute_checkpoint_paths.csv",
            "direction_calls": run_dir / "g17_one_minute_direction_calls.csv",
            "direction_scores": run_dir / "g17_one_minute_direction_scores.csv",
            "method_decisions": run_dir / "g17_one_minute_method_decisions.csv",
            "period_summary": run_dir / "g17_one_minute_period_summary.csv",
            "boundary_audit": run_dir / "g17_one_minute_boundary_audit.csv",
            "analysis_coverage": run_dir / "g17_one_minute_analysis_coverage.csv",
        }
        frames = {
            "contact_state": actual,
            "horizon_outcomes": outcomes,
            "checkpoint_paths": paths,
            "direction_calls": calls,
            "direction_scores": scores,
            "method_decisions": decisions,
            "period_summary": periods,
            "boundary_audit": boundaries,
            "analysis_coverage": coverage,
        }
        for name, frame in frames.items():
            g0.atomic_write_csv(frame, outputs[name])
        summary_path = run_dir / "g17_one_minute_summary.json"
        g0.atomic_write_json(summary, summary_path)
        record.update(
            {
                "status": summary["status"],
                "completed_at_utc": g0.utc_now(),
                "summary": summary,
                "artifacts": {name: artifact(path) for name, path in outputs.items()}
                | {"summary": artifact(summary_path)},
            }
        )
        g0.atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": g0.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        g0.atomic_write_json(record, record_path)
        raise
    print(json.dumps(summary, indent=2, default=g0.json_default))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return run_analysis(str(args.run_id), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    raise SystemExit(main())
