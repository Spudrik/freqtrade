"""Run and score the frozen G1-B2 exact-event activity/component FreqAI ladder."""

from __future__ import annotations

import argparse
import json
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
    actuals_for_pair,
    profile_config,
    run_manifest,
    scope_dates,
    target_metadata,
)


ENTRY_ID = "mtf_confluence_d1_vp_bos_4h_retest_short"
DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation1"
    / "g1-b2"
    / "freqai"
)
DEFAULT_CACHE = DEFAULT_OUTPUT.parent / "cache"
DEFAULT_SCOPES = DEFAULT_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-07" / "Scripts" / "python.exe"
)
ONSET = f"entry_onset__{ENTRY_ID}"
ACTIVE = f"entry_active__{ENTRY_ID}"
COMPONENTS = [
    "component__d1_vp_gate",
    "component__d1_structure_bos_gate",
    "component__d1_prior_level_gate",
    "component__h4_retest_gate",
    "component__h4_structure_gate",
    "component__h4_context_gate",
    "component__d1_gate_count",
    "component__h4_gate_count",
    "component__all_gate_count",
    "component__d1_confluence_pass",
    "component__h4_execution_pass",
    "component__vp_bos_joint",
    "component__source_age_hours",
]


def without(prefixes: tuple[str, ...]) -> list[str]:
    return [column for column in COMPONENTS if not column.startswith(prefixes)]


NO_D1 = without(("component__d1_", "component__vp_bos_", "component__all_"))
NO_VP_BOS = [
    column
    for column in COMPONENTS
    if column
    not in {
        "component__d1_vp_gate",
        "component__d1_structure_bos_gate",
        "component__d1_gate_count",
        "component__d1_confluence_pass",
        "component__vp_bos_joint",
        "component__all_gate_count",
    }
]
NO_H4_RETEST = [
    column
    for column in COMPONENTS
    if column
    not in {
        "component__h4_retest_gate",
        "component__h4_gate_count",
        "component__h4_execution_pass",
        "component__all_gate_count",
    }
]

PROFILES: dict[str, dict[str, Any]] = {
    "fixed_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": None,
        "theory": "Fixed multi-timeframe price and conventional-indicator state without the event or its components.",
    },
    "shifted_exact_event": {
        "strategy": "Sieve3EventReactionFixedIndicatorShiftedOnsetFreqAIResearchStrategy",
        "shift_hours": 168,
        "feature_columns": [ONSET],
        "theory": "The exact onset shifted 168 hours is the timestamp-safe like-for-like event placebo.",
    },
    "shifted_exact_active": {
        "strategy": "Sieve3EventReactionFixedIndicatorShiftedIdentityFreqAIResearchStrategy",
        "shift_hours": 168,
        "feature_columns": [ONSET, ACTIVE],
        "theory": "Onset plus active state shifted 168 hours is the like-for-like persistent-state placebo.",
    },
    "exact_event": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [ONSET],
        "theory": "Exact event onset adds local activity information beyond fixed market state.",
    },
    "exact_event_active": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityActiveFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [ONSET, ACTIVE],
        "theory": "Observable event persistence adds information beyond exact onset.",
    },
    "components_all": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": COMPONENTS,
        "theory": "The event's timestamp-safe D1 and H4 gate ingredients explain the activity relationship without exact identity.",
    },
    "components_no_d1": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": NO_D1,
        "theory": "Remove the D1 confluence group to measure its incremental contribution.",
    },
    "components_no_vp_bos": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": NO_VP_BOS,
        "theory": "Remove the named D1 volume-profile/BOS ingredients to measure their contribution.",
    },
    "components_no_h4_retest": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": NO_H4_RETEST,
        "theory": "Remove the 4h retest/location ingredient and dependent summaries to measure its contribution.",
    },
    "exact_plus_components": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [ONSET, *COMPONENTS],
        "theory": "Exact onset is added to all observable ingredients to test information beyond its components.",
    },
}


def targets() -> list[str]:
    output: list[str] = []
    for horizon in (1, 2, 4):
        output.extend(
            [
                f"&-future_return_{horizon}h",
                f"&-future_upside_{horizon}h_atr",
                f"&-future_downside_{horizon}h_atr",
                f"&-future_path_balance_{horizon}h_atr",
                f"&-future_reaction_magnitude_{horizon}h_atr",
                f"&-future_volume_ratio_{horizon}h",
                f"&-future_pressure_{horizon}h",
                f"&-future_volatility_ratio_{horizon}h",
            ]
        )
    return output


TARGETS = targets()


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
    run_id = f"sieve3_event_g1_b2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    commands: list[dict[str, Any]] = []
    for profile_id in selected_profiles:
        definition = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        profile_dir = output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config = profile_config(base, identifier, definition, pairs, cache_dir)
        settings = config["sieve3_event_reaction"]
        settings["target_names"] = TARGETS
        feature_columns = definition.get("feature_columns")
        if feature_columns:
            settings["event_feature_columns"] = list(feature_columns)
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
                "event_feature_columns": feature_columns or [],
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": 1,
        "batch": "G1-B2",
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
            "exact_event": ["fixed_indicator_control", "shifted_exact_event", "components_all"],
            "exact_event_active": ["fixed_indicator_control", "shifted_exact_active"],
            "components_all": ["fixed_indicator_control", "components_no_d1", "components_no_vp_bos", "components_no_h4_retest"],
            "exact_plus_components": ["fixed_indicator_control", "components_all", "exact_event"],
        },
        "commands": commands,
    }


def add_comparisons(scores: DataFrame) -> DataFrame:
    key = ["pair", "window", "scope", "target"]
    comparator_ids = {
        "fixed_indicator_control": "indicator",
        "shifted_exact_event": "shifted_exact",
        "shifted_exact_active": "shifted_active",
        "exact_event": "exact",
        "components_all": "components",
        "components_no_d1": "no_d1",
        "components_no_vp_bos": "no_vp_bos",
        "components_no_h4_retest": "no_h4_retest",
    }
    scores = scores.copy()
    for metric in METRICS:
        if metric not in scores:
            scores[metric] = np.nan
    output = scores.copy()
    for profile_id, prefix in comparator_ids.items():
        control = scores[scores["profile_id"].eq(profile_id)][[*key, *METRICS]].rename(
            columns={metric: f"{prefix}_{metric}" for metric in METRICS}
        )
        output = output.merge(control, on=key, how="left", validate="many_to_one")
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
    scopes = {
        pair: scope_dates(Path(manifest["event_scopes"]), pair) for pair in pairs
    }
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
                    for target in TARGETS:
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
    scores["target_family"] = metadata.map(lambda item: item[0])
    scores["horizon_hours"] = metadata.map(lambda item: item[1])
    scores = add_comparisons(scores)
    scores.to_csv(run_dir / "g1_b2_scores.csv", index=False)
    return scores


def portability_summary(
    scores: DataFrame, manifest: dict[str, Any], run_dir: Path
) -> DataFrame:
    records: list[DataFrame] = []
    for profile_id, comparators in manifest["comparison_contract"].items():
        profile = scores[
            scores["status"].eq("scored")
            & scores["profile_id"].eq(profile_id)
            & ~scores["scope"].eq("all")
        ].copy()
        for comparator_id in comparators:
            prefix = {
                "fixed_indicator_control": "indicator",
                "shifted_exact_event": "shifted_exact",
                "shifted_exact_active": "shifted_active",
                "exact_event": "exact",
                "components_all": "components",
                "components_no_d1": "no_d1",
                "components_no_vp_bos": "no_vp_bos",
                "components_no_h4_retest": "no_h4_retest",
            }[comparator_id]
            work = profile[
                [
                    "pair",
                    "window",
                    "scope",
                    "target",
                    "target_family",
                    "horizon_hours",
                    "rows",
                    f"prediction_actual_spearman_delta_vs_{prefix}",
                    f"mean_absolute_error_skill_vs_{prefix}",
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
            work["rank_positive"] = work["spearman_delta"].gt(0.0)
            work["mae_positive"] = work["mae_skill"].gt(0.0)
            work["rank_and_mae_positive"] = (
                work["rank_positive"] & work["mae_positive"]
            )
            records.append(work)
    long = pd.concat(records, ignore_index=True)
    grouping = [
        "profile_id",
        "comparator_id",
        "scope",
        "target",
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
            rank_positive_share=("rank_positive", "mean"),
            mae_positive_share=("mae_positive", "mean"),
            median_spearman_delta=("spearman_delta", "median"),
            median_mae_skill=("mae_skill", "median"),
        )
        .reset_index()
    )
    summary = add_majority_consistency(
        summary, long, grouping, "rank_and_mae_positive", "rank_mae"
    )
    summary.to_csv(run_dir / "g1_b2_portability_summary.csv", index=False)
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
    parser.add_argument(
        "--pairs",
        default="BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,ADA/USDT:USDT,DOGE/USDT:USDT,TRX/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT",
    )
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
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
        "batch": "G1-B2",
        "profiles": len(manifest["commands"]),
        "score_rows": len(scores),
        "portability_rows": len(portability),
        "scores": str(output_dir / "g1_b2_scores.csv"),
        "portability": str(output_dir / "g1_b2_portability_summary.csv"),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    append_log(
        log_path,
        f"G1-B2 ladder completed with {len(scores)} score rows and {len(portability)} portability rows.",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
