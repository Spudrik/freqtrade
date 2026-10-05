"""Run and score the frozen G1-B1 precursor/onset attribution FreqAI ladder."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from sieve3_event_freqai_round1 import (
    DEFAULT_CONFIG,
    MODELS_DIR,
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
from user_data.strategies.sieve3_event_reaction_targets import (
    TARGET_HORIZONS,
    build_sieve3_reaction_targets,
)


DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation1"
    / "g1-b1"
    / "freqai"
)
DEFAULT_CACHE = DEFAULT_OUTPUT.parent / "cache"
DEFAULT_SCOPES = DEFAULT_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_WINDOWS = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation0"
    / "windows.json"
)
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
DEFAULT_BASE_CONFIG = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation0"
    / "full_v2"
    / "freqai"
    / "price_control"
    / "config.json"
)
INCLUDE_TIMEFRAMES = ["1h", "4h", "8h", "1d"]
PROFILES = {
    "lean_price_control": {
        "strategy": "Sieve3EventReactionPriceControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "theory": "Multi-timeframe OHLCV/pressure/structure without conventional indicators or event identity.",
    },
    "fixed_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "theory": "Frozen RSI/Bollinger/MACD/SMA/EMA/ADX controls added to the same multi-timeframe price state.",
    },
    "event_onset_identity": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "theory": "Exact selected family onset counts add information beyond the fixed indicator control.",
    },
    "event_onset_active_identity": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityActiveFreqAIResearchStrategy",
        "shift_hours": 0,
        "theory": "Exact selected family onset plus currently observable active state add information.",
    },
    "shifted_event_placebo": {
        "strategy": "Sieve3EventReactionFixedIndicatorShiftedIdentityFreqAIResearchStrategy",
        "shift_hours": 168,
        "theory": "The same onset/active features delayed 168 hours form a timestamp-safe unrelated-event placebo.",
    },
    "shifted_onset_placebo": {
        "strategy": "Sieve3EventReactionFixedIndicatorShiftedOnsetFreqAIResearchStrategy",
        "shift_hours": 168,
        "theory": "The onset-only features delayed 168 hours form the like-for-like timestamp-safe placebo for the onset-only exact model.",
    },
}


def targets() -> list[str]:
    output: list[str] = []
    for horizon in TARGET_HORIZONS:
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


def target_metadata(target: str) -> tuple[str, int]:
    match = re.search(r"_(\d+)h(?:_|$)", target)
    if not match:
        raise ValueError(f"Target horizon is not encoded: {target}")
    horizon = int(match.group(1))
    if "future_return_" in target:
        family = "terminal_direction"
    elif "future_upside_" in target or "future_downside_" in target:
        family = "favourable_adverse_excursion"
    elif "path_balance" in target:
        family = "path_direction"
    elif "reaction_magnitude" in target:
        family = "reaction_magnitude"
    elif "volume_ratio" in target:
        family = "volume_activity"
    elif "pressure" in target:
        family = "pressure_direction"
    elif "volatility_ratio" in target:
        family = "volatility_activity"
    else:
        raise ValueError(target)
    return family, horizon


def profile_config(
    base: dict[str, Any],
    identifier: str,
    profile: dict[str, Any],
    pairs: tuple[str, ...],
    cache_dir: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["freqai"]["enabled"] = True
    config["freqai"]["identifier"] = identifier
    config["freqai"]["purge_old_models"] = 1
    config["freqai"]["train_period_days"] = 365
    config["freqai"]["backtest_period_days"] = 90
    config["freqai"]["feature_parameters"]["include_timeframes"] = INCLUDE_TIMEFRAMES
    config["freqai"]["feature_parameters"]["include_shifted_candles"] = 0
    config["freqai"]["feature_parameters"]["plot_feature_importances"] = 0
    config["freqai"]["data_split_parameters"] = {
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": False,
    }
    config["exchange"]["pair_whitelist"] = list(pairs)
    config["sieve3_event_reaction"] = {
        "event_cache_dir": str(cache_dir),
        "event_scope_end": "2026-06-29T00:00:00+00:00",
        "event_feature_shift_hours": int(profile["shift_hours"]),
        "target_names": TARGETS,
        "precursor_feature_policy": "forbidden_future_derived_scoring_scope_only",
    }
    return config


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
    run_id = f"sieve3_event_g1_b1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    commands: list[dict[str, Any]] = []
    for profile_id in selected_profiles:
        definition = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        profile_dir = output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config_path = profile_dir / "config.json"
        atomic_json(
            config_path,
            profile_config(base, identifier, definition, pairs, cache_dir),
        )
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
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": 1,
        "batch": "G1-B1",
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
        "commands": commands,
    }


def append_missing_profiles(
    manifest: dict[str, Any],
    path: Path,
    base_config_path: Path,
    python_exe: Path,
    pairs: tuple[str, ...],
    selected_profiles: tuple[str, ...],
    cache_dir: Path,
) -> dict[str, Any]:
    """Apply an append-only integrity repair without discarding completed folds."""
    existing = tuple(item["profile_id"] for item in manifest["commands"])
    if selected_profiles[: len(existing)] != existing:
        raise ValueError(
            f"Existing profile manifest differs: existing={existing} requested={selected_profiles}"
        )
    if len(existing) == len(selected_profiles):
        return manifest
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_dir = path.parent
    for profile_id in selected_profiles[len(existing) :]:
        definition = PROFILES[profile_id]
        identifier = f"{manifest['run_id']}_{profile_id}"
        profile_dir = output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config_path = profile_dir / "config.json"
        atomic_json(
            config_path,
            profile_config(base, identifier, definition, pairs, cache_dir),
        )
        manifest["commands"].append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "theory": definition["theory"],
                "identifier": identifier,
                "config_path": str(config_path),
                "output_dir": str(profile_dir),
                "command": [
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
                    str(manifest["timerange"]),
                    "--export",
                    "signals",
                    "--export-directory",
                    str(profile_dir),
                ],
                "status": "pending",
                "attempts": 0,
            }
        )
    manifest["integrity_repairs"] = [
        *manifest.get("integrity_repairs", []),
        {
            "recorded_at": now_iso(),
            "defect": "The onset-only exact model lacked a like-for-like shifted onset-only placebo; the existing shifted placebo contains onset plus active-state features.",
            "repair": "Append shifted_onset_placebo and use profile-matched shifted comparisons without discarding reusable folds.",
        },
    ]
    atomic_json(path, manifest)
    return manifest


def run_manifest(manifest: dict[str, Any], path: Path, log_path: Path) -> int:
    total = len(manifest["commands"])
    batch = str(manifest.get("batch") or "G1-B1")
    for index, item in enumerate(manifest["commands"], start=1):
        if item.get("status") == "completed" and int(item.get("returncode", 1)) == 0:
            continue
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["started_at"] = now_iso()
        item["status"] = "running"
        stdout_path = Path(item["output_dir"]) / f"freqai_stdout_attempt_{item['attempts']}.log"
        stderr_path = Path(item["output_dir"]) / f"freqai_stderr_attempt_{item['attempts']}.log"
        item["stdout_log"] = str(stdout_path)
        item["stderr_log"] = str(stderr_path)
        atomic_json(path, manifest)
        append_log(
            log_path,
            f"{batch} FreqAI profile `{item['profile_id']}` started. Theory: {item['theory']}.",
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                item["command"],
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        item["finished_at"] = now_iso()
        item["returncode"] = int(result.returncode)
        item["status"] = "completed" if result.returncode == 0 else "failed"
        atomic_json(path, manifest)
        append_log(
            log_path,
            f"{batch} FreqAI profile `{item['profile_id']}` {item['status']} with return code {result.returncode}.",
        )
        print(
            json.dumps(
                {
                    "phase": "profiles",
                    "processed": index,
                    "total": total,
                    "profile": item["profile_id"],
                    "status": item["status"],
                }
            ),
            flush=True,
        )
        if result.returncode != 0:
            return int(result.returncode)
    return 0


def actuals_for_pair(pair: str) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    labelled = build_sieve3_reaction_targets(raw)
    for horizon in TARGET_HORIZONS:
        name = f"&-future_reaction_magnitude_{horizon}h_atr"
        labelled[name] = pd.concat(
            [
                pd.to_numeric(labelled[f"&-future_upside_{horizon}h_atr"], errors="coerce"),
                pd.to_numeric(labelled[f"&-future_downside_{horizon}h_atr"], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
    return labelled[["date", *TARGETS]].rename(
        columns={target: f"{target}_actual" for target in TARGETS}
    )


def scope_dates(scopes_path: Path, pair: str) -> dict[str, set[pd.Timestamp]]:
    frame = pd.read_parquet(scopes_path)
    frame = frame[frame["pair"].eq(pair)].copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    output: dict[str, set[pd.Timestamp]] = {}
    for keys, group in frame.groupby(
        ["family", "side", "scope_type"], dropna=False, observed=True
    ):
        family, side, scope_type = keys
        dates = set(group["decision_time"] - pd.Timedelta(hours=1))
        if len(dates) >= 20:
            output[f"family:{family}:{side}:{scope_type}"] = dates
    episode_starts = frame[
        frame["scope_type"].eq("exact_onset") & frame["is_episode_start"].astype(bool)
    ]
    for keys, group in episode_starts.groupby(
        ["family", "side"], dropna=False, observed=True
    ):
        family, side = keys
        dates = set(group["decision_time"] - pd.Timedelta(hours=1))
        if len(dates) >= 20:
            output[f"family:{family}:{side}:episode_start"] = dates
    return output


METRICS = (
    "prediction_actual_spearman",
    "top_minus_bottom",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r_squared",
    "prediction_bias",
    "roc_auc",
    "average_precision",
    "brier_score_clipped",
)


def add_comparisons(scores: DataFrame) -> DataFrame:
    key = ["pair", "window", "scope", "target"]
    comparators = {
        "lean_price_control": "lean",
        "fixed_indicator_control": "indicator",
        "shifted_event_placebo": "shifted_active",
        "shifted_onset_placebo": "shifted_onset",
    }
    scores = scores.copy()
    for metric in METRICS:
        if metric not in scores:
            scores[metric] = np.nan
    output = scores.copy()
    for profile_id, prefix in comparators.items():
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
        output[f"absolute_bias_skill_vs_{prefix}"] = pd.to_numeric(
            output[f"{prefix}_prediction_bias"], errors="coerce"
        ).abs() - pd.to_numeric(output["prediction_bias"], errors="coerce").abs()
    return output


def add_majority_consistency(
    summary: DataFrame,
    observations: DataFrame,
    grouping: list[str],
    positive_column: str,
    prefix: str,
) -> DataFrame:
    """Add pair/window majority counts required by the frozen batch rules."""
    pair_rates = (
        observations.groupby([*grouping, "pair"], dropna=False, observed=True)[
            positive_column
        ]
        .mean()
        .reset_index(name="positive_rate")
    )
    pair_counts = (
        pair_rates.assign(majority=pair_rates["positive_rate"].ge(0.5))
        .groupby(grouping, dropna=False, observed=True)
        .agg(
            **{
                f"{prefix}_eligible_pairs": ("pair", "nunique"),
                f"{prefix}_majority_positive_pairs": ("majority", "sum"),
            }
        )
        .reset_index()
    )
    window_rates = (
        observations.groupby([*grouping, "window"], dropna=False, observed=True)[
            positive_column
        ]
        .mean()
        .reset_index(name="positive_rate")
    )
    window_counts = (
        window_rates.assign(majority=window_rates["positive_rate"].ge(0.5))
        .groupby(grouping, dropna=False, observed=True)
        .agg(
            **{
                f"{prefix}_eligible_windows": ("window", "nunique"),
                f"{prefix}_majority_positive_windows": ("majority", "sum"),
            }
        )
        .reset_index()
    )
    return summary.merge(
        pair_counts, on=grouping, how="left", validate="one_to_one"
    ).merge(window_counts, on=grouping, how="left", validate="one_to_one")


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
    scores.to_csv(run_dir / "g1_b1_scores.csv", index=False)
    return scores


def portability_summary(scores: DataFrame, run_dir: Path) -> DataFrame:
    selected = scores[
        scores["status"].eq("scored")
        & scores["profile_id"].isin(["event_onset_identity", "event_onset_active_identity"])
        & ~scores["scope"].eq("all")
    ].copy()
    active_model = selected["profile_id"].eq("event_onset_active_identity")
    for metric in ("prediction_actual_spearman", "top_minus_bottom", "r_squared"):
        selected[f"{metric}_delta_vs_matched_shifted"] = np.where(
            active_model,
            pd.to_numeric(selected[f"{metric}_delta_vs_shifted_active"], errors="coerce"),
            pd.to_numeric(selected[f"{metric}_delta_vs_shifted_onset"], errors="coerce"),
        )
    selected["mean_absolute_error_skill_vs_matched_shifted"] = np.where(
        active_model,
        pd.to_numeric(
            selected["mean_absolute_error_skill_vs_shifted_active"], errors="coerce"
        ),
        pd.to_numeric(
            selected["mean_absolute_error_skill_vs_shifted_onset"], errors="coerce"
        ),
    )
    selected["rank_both_positive"] = (
        pd.to_numeric(
            selected["prediction_actual_spearman_delta_vs_indicator"], errors="coerce"
        ).gt(0)
        & pd.to_numeric(
            selected["prediction_actual_spearman_delta_vs_matched_shifted"], errors="coerce"
        ).gt(0)
    )
    selected["mae_both_positive"] = (
        pd.to_numeric(selected["mean_absolute_error_skill_vs_indicator"], errors="coerce").gt(0)
        & pd.to_numeric(
            selected["mean_absolute_error_skill_vs_matched_shifted"], errors="coerce"
        ).gt(0)
    )
    selected["rank_and_mae_both_positive"] = (
        selected["rank_both_positive"] & selected["mae_both_positive"]
    )
    grouping = ["profile_id", "scope", "target", "target_family", "horizon_hours"]
    summary = (
        selected.groupby(
            grouping,
            dropna=False,
            observed=True,
        )
        .agg(
            pair_window_tests=("pair", "size"),
            distinct_pairs=("pair", "nunique"),
            distinct_windows=("window", "nunique"),
            sample_rows=("rows", "sum"),
            rank_both_positive_share=("rank_both_positive", "mean"),
            mae_both_positive_share=("mae_both_positive", "mean"),
            median_spearman_delta_vs_indicator=(
                "prediction_actual_spearman_delta_vs_indicator",
                "median",
            ),
            median_spearman_delta_vs_matched_shifted=(
                "prediction_actual_spearman_delta_vs_matched_shifted",
                "median",
            ),
            median_mae_skill_vs_indicator=(
                "mean_absolute_error_skill_vs_indicator",
                "median",
            ),
            median_mae_skill_vs_matched_shifted=(
                "mean_absolute_error_skill_vs_matched_shifted",
                "median",
            ),
        )
        .reset_index()
    )
    summary = add_majority_consistency(
        summary,
        selected,
        grouping,
        "rank_and_mae_both_positive",
        "rank_mae_vs_both_controls",
    )
    summary.to_csv(run_dir / "g1_b1_portability_summary.csv", index=False)
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    log_path = output_dir.parent / "research_log.md"
    selected = tuple(token.strip() for token in args.profiles.split(",") if token.strip())
    unknown = set(selected) - set(PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = append_missing_profiles(
            manifest,
            manifest_path,
            args.base_config.resolve(),
            args.python_exe.resolve(),
            pairs,
            selected,
            args.cache_dir.resolve(),
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
    if not args.score_only:
        returncode = run_manifest(manifest, manifest_path, log_path)
        if returncode:
            return returncode
    scores = score_manifest(manifest, output_dir)
    portability = portability_summary(scores, output_dir)
    summary = {
        "finished_at": now_iso(),
        "batch": "G1-B1",
        "profiles": len(manifest["commands"]),
        "score_rows": len(scores),
        "portability_rows": len(portability),
        "scores": str(output_dir / "g1_b1_scores.csv"),
        "portability": str(output_dir / "g1_b1_portability_summary.csv"),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    append_log(
        log_path,
        f"G1-B1 ladder completed with {len(scores)} score rows and {len(portability)} portability rows.",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
