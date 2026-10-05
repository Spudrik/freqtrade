"""Run and score the frozen G1-B4 exit-identity/trade-state FreqAI ladder."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from sieve3_event_freqai_round1 import (
    REPO_ROOT,
    USER_DATA_DIR,
    append_log,
    atomic_json,
    load_predictions,
    now_iso,
    ohlcv_path,
    parse_pairs,
    parse_windows,
    score_values,
)
from sieve3_event_generation1_b1_freqai import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_WINDOWS,
    INCLUDE_TIMEFRAMES,
    METRICS,
    add_majority_consistency,
    profile_config,
    run_manifest,
)
from sieve3_event_generation1_b4_cache import (
    DEFAULT_PAIRS,
    ELIGIBLE_GROUPS,
    REASON_CATEGORIES,
    group_token,
)
from user_data.strategies.sieve3_event_reaction_targets import (
    EXIT_ROLE_HORIZONS,
    EXIT_ROLE_SEQUENCE_HORIZONS,
    build_sieve3_exit_role_targets,
)


DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation1"
    / "g1-b4"
    / "freqai"
)
DEFAULT_CACHE = DEFAULT_OUTPUT.parent / "cache"
DEFAULT_SCOPES = DEFAULT_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-05" / "Scripts" / "python.exe"
)

IDENTITY_COLUMNS = [
    *[f"exit_event__{group_token(family, side)}" for family, side in ELIGIBLE_GROUPS],
    *[f"exit_event_any__{side}" for side in ("long", "short")],
    *[f"exit_event_partial__{side}" for side in ("long", "short")],
    *[f"exit_event_final__{side}" for side in ("long", "short")],
    *[f"exit_reason__{category}" for category in REASON_CATEGORIES],
    "exit_event_action_count",
]
STATE_METRICS = (
    "open_count",
    "current_profit_mean",
    "current_profit_median",
    "current_profit_min",
    "current_profit_max",
    "holding_log_mean",
    "holding_log_median",
    "holding_log_max",
    "favourable_excursion_mean",
    "favourable_excursion_max",
    "adverse_excursion_mean",
    "adverse_excursion_max",
    "target_stage_mean",
    "target_stage_max",
    "remaining_fraction_mean",
    "remaining_fraction_min",
    "risk_unit_progress_mean",
    "configured_target_progress_mean",
    "configured_target_available_share",
    "trigger_active_share",
    "trigger_onset_share",
)
TRADE_STATE_COLUMNS = [
    f"trade_state__{side}__{metric}"
    for side in ("long", "short")
    for metric in STATE_METRICS
]


def target_names() -> list[str]:
    output: list[str] = []
    for side in ("long", "short"):
        for horizon in EXIT_ROLE_HORIZONS:
            output.extend(
                [
                    f"&-exit_{side}_hold_delta_{horizon}h",
                    f"&-exit_{side}_missed_profit_{horizon}h",
                    f"&-exit_{side}_avoided_loss_{horizon}h",
                    f"&-exit_{side}_net_regret_{horizon}h",
                ]
            )
            if horizon in EXIT_ROLE_SEQUENCE_HORIZONS:
                output.extend(
                    [
                        f"&-exit_{side}_favourable_peak_step_{horizon}h",
                        f"&-exit_{side}_adverse_peak_step_{horizon}h",
                        f"&-exit_{side}_favourable_before_adverse_{horizon}h",
                    ]
                )
    return output


TARGETS = target_names()

PROFILES: dict[str, dict[str, Any]] = {
    "open_trade_price_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [],
        "theory": "Price, structure and fixed indicators without exit identity or reconstructed trade state.",
    },
    "shifted_exit_family_placebo": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 168,
        "feature_columns": IDENTITY_COLUMNS,
        "theory": "The exact exit-action identity shifted 168 hours is the like-for-like event placebo.",
    },
    "exit_family_identity": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": IDENTITY_COLUMNS,
        "theory": "Executed family, partial/final role and exact reason add post-event path information.",
    },
    "exit_identity_plus_trade_state": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [*IDENTITY_COLUMNS, *TRADE_STATE_COLUMNS],
        "theory": "Exact action plus current profit, age, excursion, target stage, remaining fraction and trigger health identifies the horizon transition.",
    },
    "trade_state_with_exit_family_ablation": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": TRADE_STATE_COLUMNS,
        "theory": "Trade state without action identity tests whether the apparent family result is only generic position state.",
    },
}


def target_metadata(target: str) -> tuple[str, str, int]:
    match = re.search(r"&-exit_(long|short)_(.+)_(\d+)h$", target)
    if not match:
        raise ValueError(target)
    side, family, horizon = match.groups()
    if family == "favourable_before_adverse":
        family = "path_order_binary"
    elif family in {"favourable_peak_step", "adverse_peak_step"}:
        family = "path_timing"
    return side, family, int(horizon)


def build_manifest(
    output_dir: Path,
    base_config_path: Path,
    python_exe: Path,
    pairs: tuple[str, ...],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    selected_profiles: tuple[str, ...],
    timerange: str,
    cache_dir: Path,
    scopes_path: Path,
) -> dict[str, Any]:
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    run_id = f"sieve3_event_g1_b4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    commands: list[dict[str, Any]] = []
    for profile_id in selected_profiles:
        definition = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        profile_dir = output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config = profile_config(base, identifier, definition, pairs, cache_dir)
        settings = config["sieve3_event_reaction"]
        settings["target_names"] = TARGETS
        if definition["feature_columns"]:
            settings["event_feature_columns"] = list(definition["feature_columns"])
        config_path = profile_dir / "config.json"
        atomic_json(config_path, config)
        command = [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(USER_DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            definition["strategy"],
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            timerange,
            "--export",
            "signals",
            "--export-directory",
            str(profile_dir),
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "theory": definition["theory"],
                "identifier": identifier,
                "config_path": str(config_path),
                "output_dir": str(profile_dir),
                "event_feature_columns": list(definition["feature_columns"]),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": 1,
        "batch": "G1-B4",
        "run_id": run_id,
        "created_at": now_iso(),
        "pairs": list(pairs),
        "windows": {
            key: [start.isoformat(), end.isoformat()]
            for key, (start, end) in windows.items()
        },
        "timerange": timerange,
        "targets": TARGETS,
        "include_timeframes": INCLUDE_TIMEFRAMES,
        "event_cache_dir": str(cache_dir),
        "event_scopes": str(scopes_path),
        "worker_python": str(python_exe),
        "comparison_contract": {
            "exit_family_identity": [
                "open_trade_price_indicator_control",
                "shifted_exit_family_placebo",
            ],
            "exit_identity_plus_trade_state": [
                "open_trade_price_indicator_control",
                "shifted_exit_family_placebo",
                "exit_family_identity",
                "trade_state_with_exit_family_ablation",
            ],
            "trade_state_with_exit_family_ablation": [
                "open_trade_price_indicator_control"
            ],
        },
        "event_time_boundary": (
            "Exit order in candle T is represented after T completes; outcomes begin "
            "with the next complete candle."
        ),
        "commands": commands,
    }


def actuals_for_pair(pair: str) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    labelled = build_sieve3_exit_role_targets(raw)
    return labelled[["date", *TARGETS]].rename(
        columns={target: f"{target}_actual" for target in TARGETS}
    )


def scope_dates(scopes_path: Path, pair: str) -> dict[str, set[pd.Timestamp]]:
    frame = pd.read_parquet(scopes_path)
    frame = frame[frame["pair"].eq(pair)].copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    output: dict[str, set[pd.Timestamp]] = {}
    for group_name, group in frame.groupby("group", observed=True):
        exact = group[group["scope_kind"].eq("exact_action")]
        matched = group[group["scope_kind"].eq("matched_open_state")]
        dates = set(exact["decision_time"] - pd.Timedelta(hours=1))
        if len(dates) >= 20:
            output[f"group:{group_name}:all_actions"] = dates
        matched_dates = set(matched["decision_time"] - pd.Timedelta(hours=1))
        if len(matched_dates) >= 20:
            output[f"group:{group_name}:matched_open_state"] = matched_dates
        for final, role in ((False, "partial"), (True, "final")):
            role_group = exact[exact["is_final_action"].eq(final)]
            role_dates = set(role_group["decision_time"] - pd.Timedelta(hours=1))
            if len(role_dates) >= 20:
                output[f"group:{group_name}:{role}"] = role_dates
        for reason, reason_group in exact.groupby("reason_category", observed=True):
            reason_dates = set(reason_group["decision_time"] - pd.Timedelta(hours=1))
            if len(reason_dates) >= 20:
                output[f"group:{group_name}:reason:{reason}"] = reason_dates
    return output


def scope_side(scope: str) -> str | None:
    if scope == "all":
        return None
    first = scope.split(":", 2)[1]
    if first.endswith("__long"):
        return "long"
    if first.endswith("__short"):
        return "short"
    return None


def add_comparisons(scores: DataFrame) -> DataFrame:
    key = ["pair", "window", "scope", "target"]
    comparator_ids = {
        "open_trade_price_indicator_control": "control",
        "shifted_exit_family_placebo": "shifted",
        "exit_family_identity": "identity",
        "trade_state_with_exit_family_ablation": "state",
    }
    scores = scores.copy()
    for metric in METRICS:
        if metric not in scores:
            scores[metric] = np.nan
    output = scores.copy()
    for profile_id, prefix in comparator_ids.items():
        comparator = scores[scores["profile_id"].eq(profile_id)][
            [*key, *METRICS]
        ].rename(columns={metric: f"{prefix}_{metric}" for metric in METRICS})
        output = output.merge(comparator, on=key, how="left", validate="many_to_one")
        for metric in (
            "prediction_actual_spearman",
            "top_minus_bottom",
            "r_squared",
            "roc_auc",
            "average_precision",
        ):
            output[f"{metric}_delta_vs_{prefix}"] = pd.to_numeric(
                output[metric], errors="coerce"
            ) - pd.to_numeric(output[f"{prefix}_{metric}"], errors="coerce")
        for metric in (
            "mean_absolute_error",
            "root_mean_squared_error",
            "brier_score_clipped",
        ):
            output[f"{metric}_skill_vs_{prefix}"] = pd.to_numeric(
                output[f"{prefix}_{metric}"], errors="coerce"
            ) - pd.to_numeric(output[metric], errors="coerce")
    return output


def score_manifest(manifest: dict[str, Any], run_dir: Path) -> DataFrame:
    pairs = tuple(manifest["pairs"])
    windows = {
        name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in manifest["windows"].items()
    }
    actuals = {pair: actuals_for_pair(pair) for pair in pairs}
    scopes = {pair: scope_dates(Path(manifest["event_scopes"]), pair) for pair in pairs}
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = load_predictions(item["identifier"], pairs)
        for pair in pairs:
            merged = predictions[predictions["pair"].eq(pair)].merge(
                actuals[pair], on="date", how="left", validate="one_to_one"
            )
            merged["decision_time"] = merged["date"] + pd.Timedelta(hours=1)
            if "do_predict" in merged:
                merged = merged[
                    pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)
                ]
            for window, (start, end) in windows.items():
                window_frame = merged[
                    merged["decision_time"].ge(start)
                    & merged["decision_time"].lt(end)
                ]
                scope_map = {"all": set(window_frame["date"]), **scopes[pair]}
                for scope, dates in scope_map.items():
                    scoped = window_frame[window_frame["date"].isin(dates)]
                    side = scope_side(scope)
                    eligible_targets = (
                        TARGETS
                        if side is None
                        else [target for target in TARGETS if target.startswith(f"&-exit_{side}_")]
                    )
                    for target in eligible_targets:
                        rows.append(
                            {
                                "profile_id": item["profile_id"],
                                "identifier": item["identifier"],
                                "pair": pair,
                                "window": window,
                                "scope": scope,
                                "target": target,
                                **score_values(scoped, target, f"{target}_actual"),
                            }
                        )
    scores = DataFrame(rows)
    metadata = scores["target"].map(target_metadata)
    scores["target_side"] = metadata.map(lambda item: item[0])
    scores["target_family"] = metadata.map(lambda item: item[1])
    scores["horizon_hours"] = metadata.map(lambda item: item[2])
    scores = add_comparisons(scores)
    scores.to_csv(run_dir / "g1_b4_scores.csv", index=False)
    return scores


def portability_summary(
    scores: DataFrame, manifest: dict[str, Any], run_dir: Path
) -> DataFrame:
    prefixes = {
        "open_trade_price_indicator_control": "control",
        "shifted_exit_family_placebo": "shifted",
        "exit_family_identity": "identity",
        "trade_state_with_exit_family_ablation": "state",
    }
    records: list[DataFrame] = []
    for profile_id, comparators in manifest["comparison_contract"].items():
        profile = scores[
            scores["status"].eq("scored")
            & scores["profile_id"].eq(profile_id)
            & ~scores["scope"].eq("all")
        ].copy()
        for comparator_id in comparators:
            prefix = prefixes[comparator_id]
            work = profile[
                [
                    "pair",
                    "window",
                    "scope",
                    "target",
                    "target_side",
                    "target_family",
                    "horizon_hours",
                    "rows",
                    f"prediction_actual_spearman_delta_vs_{prefix}",
                    f"mean_absolute_error_skill_vs_{prefix}",
                    f"roc_auc_delta_vs_{prefix}",
                    f"brier_score_clipped_skill_vs_{prefix}",
                ]
            ].copy()
            work["profile_id"] = profile_id
            work["comparator_id"] = comparator_id
            work["spearman_delta"] = pd.to_numeric(
                work.pop(f"prediction_actual_spearman_delta_vs_{prefix}"),
                errors="coerce",
            )
            work["mae_skill"] = pd.to_numeric(
                work.pop(f"mean_absolute_error_skill_vs_{prefix}"), errors="coerce"
            )
            work["auc_delta"] = pd.to_numeric(
                work.pop(f"roc_auc_delta_vs_{prefix}"), errors="coerce"
            )
            work["brier_skill"] = pd.to_numeric(
                work.pop(f"brier_score_clipped_skill_vs_{prefix}"), errors="coerce"
            )
            work["rank_and_mae_positive"] = (
                work["spearman_delta"].gt(0.0) & work["mae_skill"].gt(0.0)
            )
            work["auc_and_brier_positive"] = (
                work["auc_delta"].gt(0.0) & work["brier_skill"].gt(0.0)
            )
            records.append(work)
    long = pd.concat(records, ignore_index=True) if records else DataFrame()
    if long.empty:
        summary = DataFrame()
    else:
        grouping = [
            "profile_id",
            "comparator_id",
            "scope",
            "target",
            "target_side",
            "target_family",
            "horizon_hours",
        ]
        summary = (
            long.groupby(
                grouping,
                dropna=False,
                observed=True,
            )
            .agg(
                pair_window_tests=("pair", "size"),
                distinct_pairs=("pair", "nunique"),
                distinct_windows=("window", "nunique"),
                sample_rows=("rows", "sum"),
                median_spearman_delta=("spearman_delta", "median"),
                spearman_positive_share=("spearman_delta", lambda values: values.gt(0.0).mean()),
                median_mae_skill=("mae_skill", "median"),
                mae_positive_share=("mae_skill", lambda values: values.gt(0.0).mean()),
                median_auc_delta=("auc_delta", "median"),
                auc_positive_share=("auc_delta", lambda values: values.gt(0.0).mean()),
                median_brier_skill=("brier_skill", "median"),
                brier_positive_share=("brier_skill", lambda values: values.gt(0.0).mean()),
            )
            .reset_index()
        )
        summary = add_majority_consistency(
            summary, long, grouping, "rank_and_mae_positive", "rank_mae"
        )
        summary = add_majority_consistency(
            summary, long, grouping, "auc_and_brier_positive", "auc_brier"
        )
    summary.to_csv(run_dir / "g1_b4_portability_summary.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scopes", type=Path, default=DEFAULT_SCOPES)
    parser.add_argument("--windows-json", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--timerange", default="20240401-20260629")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    if not args.cache_dir.is_dir() or not args.scopes.is_file():
        raise FileNotFoundError(
            f"B4 cache/scopes are incomplete: {args.cache_dir}, {args.scopes}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    selected = tuple(token.strip() for token in args.profiles.split(",") if token.strip())
    unknown = set(selected) - set(PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = tuple(item["profile_id"] for item in manifest["commands"])
        if existing != selected:
            raise ValueError(
                f"Existing profile manifest differs: existing={existing} requested={selected}"
            )
    else:
        manifest = build_manifest(
            output_dir,
            args.base_config.resolve(),
            args.python_exe.resolve(),
            pairs,
            windows,
            selected,
            args.timerange,
            args.cache_dir.resolve(),
            args.scopes.resolve(),
        )
        atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    log_path = output_dir.parent / "research_log.md"
    if not args.score_only:
        returncode = run_manifest(manifest, manifest_path, log_path)
        if returncode:
            return returncode
    scores = score_manifest(manifest, output_dir)
    portability = portability_summary(scores, manifest, output_dir)
    summary = {
        "finished_at": now_iso(),
        "batch": "G1-B4",
        "profiles": len(manifest["commands"]),
        "score_rows": len(scores),
        "portability_rows": len(portability),
        "scores": str(output_dir / "g1_b4_scores.csv"),
        "portability": str(output_dir / "g1_b4_portability_summary.csv"),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    append_log(
        log_path,
        f"G1-B4 ladder completed with {len(scores)} score rows and {len(portability)} portability rows.",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
