"""Run and score the compact five-family 2026 confirmation through FreqAI."""

from __future__ import annotations

# Bound numerical pools before importing model/dataframe modules.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
    market_event_freqai_breadth as engine,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026_cache as cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_portfolio as portfolio,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = frozen.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/event_hierarchy/"
    "event_signal_fresh_2026_20260909a/freqai"
)
DEFAULT_RUN_ID = frozen.RUN_ID
DEFAULT_SMOKE_ID = "event_signal_fresh_2026_smoke_20260909a"
MAX_WORKERS = 4
TARGET_PURGE_HOURS = 8
FULL_SETTINGS = {
    "timerange": "20251001-20260821",
    "train_days": 900,
    "backtest_days": 120,
    "validation_periods": list(frozen.VALIDATION_PERIODS),
}
SMOKE_SETTINGS = {
    "timerange": "20260701-20260821",
    "train_days": 900,
    "backtest_days": 51,
    "validation_periods": [frozen.VALIDATION_PERIODS[1]],
}
PERIOD_BOUNDS = {
    frozen.VALIDATION_PERIODS[0]: (
        frozen.CURRENT_START_UTC,
        frozen.CURRENT_SPLIT_UTC,
    ),
    frozen.VALIDATION_PERIODS[1]: (
        frozen.CURRENT_SPLIT_UTC,
        frozen.SOURCE_STOP_UTC,
    ),
}

PAIR_SCORES_NAME = "event_signal_fresh_pair_scores.csv"
SCOPE_SCORES_NAME = "event_signal_fresh_scope_scores.csv"
DECISIONS_NAME = "event_signal_fresh_decisions.csv"
AUDIT_NAME = "event_signal_fresh_prediction_audit.csv"
RESULT_NAME = "event_signal_fresh_result.json"
MEMBERSHIP_NAME = "event_signal_fresh_membership.parquet"


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def active_profiles(
    registry: Mapping[str, Any], *, technical_smoke: bool
) -> dict[str, dict[str, Any]]:
    profiles = {
        str(profile_id): dict(profile)
        for profile_id, profile in registry["profiles"].items()
    }
    if set(profiles) != set(frozen.PROFILE_DEFINITIONS) or len(profiles) != 5:
        raise ValueError("The frozen five-profile fresh-confirmation surface is incomplete.")
    if technical_smoke:
        profile_id = "event_plus_recent_market"
        return {profile_id: profiles[profile_id]}
    return profiles


def _build_manifest(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
    manifest, path = _ORIGINAL_BUILD_MANIFEST(*args, **kwargs)
    manifest["run_stage"] = "event_signal_fresh_2026_confirmation"
    manifest["decision_rule"] = {
        "primary_question": "Will next-hour volume be unusually high?",
        "calibration": (
            "Use the prior-only 90-day normalized prediction threshold at the 60th "
            "percentile, unchanged from the frozen 2024-2025 prototype."
        ),
        "confirmation": (
            "At least 20 independent episodes, at least 55% reaction success, and at "
            "least two percentage points over non-calls in both 2026 halves."
        ),
        "strong": "The same rule with at least 65% reaction success in both halves.",
        "matched_controls": "Reported separately as a diagnostic, not a tuned gate.",
    }
    manifest["research_boundary"] = {
        "profit_used": False,
        "direction_tested": False,
        "trading_rule_tested": False,
        "thresholds_retuned_on_2026": False,
    }
    g0.atomic_write_json(manifest, path)
    return manifest, path


def _episode_ids(values: Series) -> set[str]:
    identifiers: set[str] = set()
    for value in values.dropna():
        identifiers.update(str(item) for item in json.loads(str(value)))
    return identifiers


def _rate(values: Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def build_membership(
    *, manifest: Mapping[str, Any], predictions: Mapping[str, DataFrame], actual: DataFrame
) -> DataFrame:
    reference = pd.read_csv(manifest["storage"]["training_reference"])
    median_lookup = reference.set_index(["pair", "period", "target"])[
        "training_median"
    ].to_dict()
    eligible_actual = actual.loc[
        actual["period"].isin(manifest["validation_periods"])
    ].copy()
    outputs: list[DataFrame] = []
    representatives = {
        str(item["profile_id"]): item for item in frozen.REPRESENTATIVES
    }
    for profile_id, prediction in predictions.items():
        calibrated = portfolio.calibrated_prediction(
            prediction,
            frozen.PRIMARY_TARGET,
            direction_confidence=False,
        ).rename(
            columns={
                "prediction": "activity_prediction",
                "training_mean": "activity_training_mean",
                "training_std": "activity_training_std",
                "calibrated_score": "activity_score",
                "causal_threshold": "activity_threshold",
                "signal_issued": "activity_signal",
            }
        )
        merged = eligible_actual.merge(
            calibrated, on=["pair", "date"], how="left", validate="one_to_one"
        )
        keys = zip(merged["pair"], merged["period"], strict=True)
        merged["activity_reference"] = [
            median_lookup.get((str(pair), str(period), frozen.PRIMARY_TARGET), np.nan)
            for pair, period in keys
        ]
        merged["actual_reaction"] = pd.to_numeric(
            merged[frozen.PRIMARY_TARGET], errors="coerce"
        ).gt(pd.to_numeric(merged["activity_reference"], errors="coerce"))
        merged["activity_signal"] = merged["activity_signal"].fillna(False).astype(bool)
        route = representatives[profile_id]
        merged["family_id"] = str(route["family_id"])
        merged["route_id"] = str(route["route_id"])
        merged["profile_id"] = profile_id
        merged["reason"] = str(route["reason"])
        outputs.append(merged)
    return pd.concat(outputs, ignore_index=True)


def score_pairs(membership: DataFrame) -> DataFrame:
    keys = ["family_id", "route_id", "profile_id", "period", "pair", "sample_kind"]
    rows: list[dict[str, Any]] = []
    for key, cell in membership.groupby(keys, observed=True, sort=False):
        eligible = cell.dropna(
            subset=["activity_prediction", "activity_threshold", "activity_reference"]
        )
        issued = eligible["activity_signal"].fillna(False).astype(bool)
        calls = eligible.loc[issued]
        non_calls = eligible.loc[~issued]
        actual = str(key[-1]) == "actual_event"
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_rows": len(eligible),
                "independent_samples": (
                    len(_episode_ids(eligible["parent_episode_ids_json"]))
                    if actual
                    else int(eligible["sample_id"].nunique())
                ),
                "calls": len(calls),
                "call_independent_samples": (
                    len(_episode_ids(calls["parent_episode_ids_json"]))
                    if actual
                    else int(calls["sample_id"].nunique())
                ),
                "call_rate": len(calls) / max(1, len(eligible)),
                "reaction_rate_all": _rate(eligible["actual_reaction"]),
                "reaction_rate_calls": _rate(calls["actual_reaction"]),
                "reaction_rate_non_calls": _rate(non_calls["actual_reaction"]),
                "reaction_enrichment_vs_non_calls": (
                    _rate(calls["actual_reaction"])
                    - _rate(non_calls["actual_reaction"])
                ),
            }
        )
    return DataFrame.from_records(rows)


RATE_COLUMNS = (
    "call_rate",
    "reaction_rate_all",
    "reaction_rate_calls",
    "reaction_rate_non_calls",
    "reaction_enrichment_vs_non_calls",
)


def add_scope_scores(pair_scores: DataFrame) -> DataFrame:
    individual = pair_scores.copy()
    individual["market_scope"] = "asset:" + individual["pair"].astype(str)
    individual["eligible_coins"] = 1
    keys = ["family_id", "route_id", "profile_id", "period", "sample_kind"]
    pooled: list[dict[str, Any]] = []
    for key, cell in pair_scores.groupby(keys, observed=True, sort=False):
        if set(cell["pair"]) != set(frozen.NORMAL_PAIRS):
            continue
        pooled.append(
            {
                **dict(zip(keys, key, strict=True)),
                "pair": "ALL_FIVE_EQUAL_WEIGHT",
                "market_scope": "all_five_equal_weight",
                "eligible_coins": len(frozen.NORMAL_PAIRS),
                "eligible_rows": int(cell["eligible_rows"].sum()),
                "independent_samples": int(cell["independent_samples"].min()),
                "calls": int(cell["calls"].sum()),
                "call_independent_samples": int(
                    cell["call_independent_samples"].min()
                ),
                **{
                    column: float(pd.to_numeric(cell[column], errors="coerce").mean())
                    for column in RATE_COLUMNS
                },
            }
        )
    return pd.concat((individual, DataFrame.from_records(pooled)), ignore_index=True)


def decide(scope_scores: DataFrame, periods: Sequence[str]) -> DataFrame:
    actual = scope_scores.loc[scope_scores["sample_kind"].eq("actual_event")]
    controls = scope_scores.loc[
        scope_scores["sample_kind"].eq("matched_control")
    ].set_index(["market_scope", "profile_id", "period"])
    expected = set(periods)
    keys = ["family_id", "route_id", "profile_id", "market_scope"]
    rows: list[dict[str, Any]] = []
    for key, cell in actual.groupby(keys, observed=True, sort=False):
        complete = set(cell["period"]) == expected
        support = bool(
            complete
            and cell["call_independent_samples"].ge(20).all()
            and cell["eligible_coins"].ge(1).all()
        )
        retained = bool(
            support
            and cell["reaction_rate_calls"].ge(portfolio.LEAD_RATE).all()
            and cell["reaction_enrichment_vs_non_calls"]
            .ge(portfolio.MIN_REACTION_ENRICHMENT)
            .all()
        )
        strong = bool(retained and cell["reaction_rate_calls"].ge(portfolio.STRONG_RATE).all())
        control_rates: list[float] = []
        for period in cell["period"]:
            control_key = (str(key[3]), str(key[2]), str(period))
            if control_key in controls.index:
                value = controls.loc[control_key, "reaction_rate_calls"]
                if isinstance(value, Series):
                    value = value.iloc[0]
                control_rates.append(float(value))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": (
                    "strong_fresh_activity_confirmation"
                    if strong
                    else "fresh_activity_confirmation"
                    if retained
                    else "not_confirmed_on_fresh_2026"
                ),
                "complete_periods": complete,
                "support_pass": support,
                "fresh_activity_pass": retained,
                "fresh_activity_strong": strong,
                "minimum_call_independent_samples": int(
                    cell["call_independent_samples"].min()
                ),
                "minimum_reaction_rate_calls": float(
                    cell["reaction_rate_calls"].min()
                ),
                "minimum_reaction_enrichment_vs_non_calls": float(
                    cell["reaction_enrichment_vs_non_calls"].min()
                ),
                "minimum_matched_control_call_reaction_rate": (
                    min(control_rates) if control_rates else np.nan
                ),
            }
        )
    return DataFrame.from_records(rows)


def score_run(manifest: dict[str, Any], *, record_dir: Path) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = engine.g0f.load_predictions(
            Path(item["model_dir"]), tuple(str(pair) for pair in item["pairs"])
        )
        audit.update({"profile_id": item["profile_id"], "role": item["role"]})
        if frame.empty or int(audit["duplicate_pair_date_rows"]):
            raise ValueError(f"Invalid fresh predictions for {item['profile_id']}: {audit}")
        predictions[str(item["profile_id"])] = frame
        audits.append(audit)

    actual = engine.load_actual(manifest)
    membership = build_membership(
        manifest=manifest, predictions=predictions, actual=actual
    )
    pair_scores = score_pairs(membership)
    scope_scores = add_scope_scores(pair_scores)
    decisions = decide(scope_scores, manifest["validation_periods"])

    audit_path = record_dir / AUDIT_NAME
    pair_path = record_dir / PAIR_SCORES_NAME
    scope_path = record_dir / SCOPE_SCORES_NAME
    decision_path = record_dir / DECISIONS_NAME
    membership_path = Path(manifest["storage"]["bulky_artifact_dir"]) / MEMBERSHIP_NAME
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(DataFrame.from_records(audits), audit_path)
    g0.atomic_write_csv(pair_scores, pair_path)
    g0.atomic_write_csv(scope_scores, scope_path)
    g0.atomic_write_csv(decisions, decision_path)
    g0.atomic_write_parquet(membership, membership_path)

    all_five = decisions.loc[decisions["market_scope"].eq("all_five_equal_weight")]
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_event_signal_fresh_2026_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_event_signal_fresh_2026_confirmation"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "profiles_completed": len(manifest["commands"]),
        "primary_target": frozen.PRIMARY_TARGET,
        "all_five_fresh_confirmations": int(
            all_five.get("fresh_activity_pass", Series(dtype=bool)).fillna(False).sum()
        ),
        "all_five_strong_confirmations": int(
            all_five.get("fresh_activity_strong", Series(dtype=bool)).fillna(False).sum()
        ),
        "integrity": {
            "real_freqai_model_class": manifest["model_class"],
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "thresholds_retuned_on_2026": False,
            "whole_event_periods_preserved": True,
            "profit_used": False,
            "direction_tested": False,
        },
        "artifacts": {
            "prediction_audit": artifact(audit_path),
            "pair_scores": artifact(pair_path),
            "scope_scores": artifact(scope_path),
            "decisions": artifact(decision_path),
            "membership": artifact(membership_path),
        },
        "interpretation": (
            "This is a later-period activity confirmation only. It does not establish "
            "direction, profit, entry, exit, or a trading rule."
        ),
    }
    result_path = record_dir / RESULT_NAME
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


_ORIGINAL_BUILD_MANIFEST = engine.build_manifest


@contextmanager
def _engine_scope() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "frozen": frozen,
        "cache": cache,
        "ANALYSIS_PATH": ANALYSIS_PATH,
        "RECORD_ROOT": RECORD_ROOT,
        "ARTIFACT_ROOT": ARTIFACT_ROOT,
        "DEFAULT_RUN_ID": DEFAULT_RUN_ID,
        "DEFAULT_SMOKE_ID": DEFAULT_SMOKE_ID,
        "MAX_WORKERS": MAX_WORKERS,
        "TARGET_PURGE_HOURS": TARGET_PURGE_HOURS,
        "FULL_SETTINGS": FULL_SETTINGS,
        "SMOKE_SETTINGS": SMOKE_SETTINGS,
        "PERIOD_BOUNDS": PERIOD_BOUNDS,
        "active_profiles": active_profiles,
        "build_manifest": _build_manifest,
        "score_run": score_run,
    }
    previous = {name: getattr(engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def run(
    *,
    run_id: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    with _engine_scope():
        return engine.run(
            run_id=run_id,
            base_config=base_config,
            python_exe=python_exe,
            profile_workers=profile_workers,
            technical_smoke=technical_smoke,
            prepare_only=prepare_only,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--base-config", type=Path, default=engine.DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=engine.DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    run_id = args.run_id or (
        DEFAULT_SMOKE_ID if args.technical_smoke else DEFAULT_RUN_ID
    )
    result = run(
        run_id=run_id,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=max(1, min(args.profile_workers, MAX_WORKERS)),
        technical_smoke=args.technical_smoke,
        prepare_only=args.prepare_only,
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0 if str(result["status"]).startswith(("completed_", "prepared_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
