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
from sklearn.metrics import average_precision_score, roc_auc_score

from sieve3_event_reaction_research import (
    BASELINE_PAIRS,
    DEFAULT_WINDOWS,
    append_log,
    atomic_json,
    build_sieve3_reaction_targets,
    ohlcv_path,
    pair_token,
    parse_pairs,
    parse_windows,
    research_log_path,
)
from user_data.strategies.Sieve3EventReactionFreqAIResearchStrategy import (
    TARGET_HORIZONS,
    TOUCH_HORIZONS,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_sieve3_event_reaction_freqai.example.json"
DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "round1"
)
DEFAULT_PYTHON = REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
MODELS_DIR = USER_DATA_DIR / "models"

PROFILES = {
    "price_control": {
        "strategy": "Sieve3EventReactionPriceControlFreqAIResearchStrategy",
        "theory": "OHLCV state, pressure, rolling extremes, and volatility form the immediate non-event baseline.",
    },
    "event_identity_placebo": {
        "strategy": "Sieve3EventReactionShiftedIdentityFreqAIResearchStrategy",
        "theory": "Sieve3 event identities shifted forward by 168 hours should not match the exact event model if exact timing carries information.",
        "event_feature_shift_hours": 168,
    },
    "level_context": {
        "strategy": "Sieve3EventReactionLevelContextFreqAIResearchStrategy",
        "theory": "Named multi-timeframe level proximity should add reaction information beyond OHLCV state alone.",
    },
    "event_identity": {
        "strategy": "Sieve3EventReactionIdentityFreqAIResearchStrategy",
        "theory": "Exact frozen Sieve3 entry onsets and overlaps should add reaction information beyond OHLCV state alone.",
    },
    "event_level_interaction": {
        "strategy": "Sieve3EventReactionEventLevelInteractionFreqAIResearchStrategy",
        "theory": "Entry identity plus named-level proximity, clusters, and hit response should add more than either context family alone.",
    },
}

TARGETS = [
    *[f"&-future_return_{h}h" for h in TARGET_HORIZONS],
    *[f"&-future_upside_{h}h_atr" for h in TARGET_HORIZONS],
    *[f"&-future_downside_{h}h_atr" for h in TARGET_HORIZONS],
    *[f"&-future_upside_peak_step_{h}h" for h in TARGET_HORIZONS],
    *[f"&-future_downside_peak_step_{h}h" for h in TARGET_HORIZONS],
    *[f"&-future_path_balance_{h}h_atr" for h in TARGET_HORIZONS],
    *[f"&-future_volume_ratio_{h}h" for h in TARGET_HORIZONS],
    *[f"&-future_pressure_{h}h" for h in TARGET_HORIZONS],
    *[f"&-future_volatility_ratio_{h}h" for h in TARGET_HORIZONS],
    *[f"&-up_before_down_1atr_{h}h" for h in TOUCH_HORIZONS],
    *[f"&-first_1atr_touch_step_{h}h" for h in TOUCH_HORIZONS],
]


def target_metadata(target: str) -> tuple[str, int | None]:
    if "upside_peak_step" in target or "downside_peak_step" in target:
        family = "peak_timing"
    elif "first_1atr_touch_step" in target:
        family = "threshold_timing"
    elif "up_before_down_1atr" in target:
        family = "threshold_direction"
    elif "future_upside" in target or "future_downside" in target:
        family = "favourable_adverse_excursion"
    elif "future_path_balance" in target:
        family = "path_direction"
    elif "future_return" in target:
        family = "terminal_direction"
    elif "future_volume_ratio" in target:
        family = "volume_activity"
    elif "future_pressure" in target:
        family = "pressure_direction"
    elif "future_volatility_ratio" in target:
        family = "volatility_activity"
    else:
        family = "other"
    match = re.search(r"_(\d+)h(?:_|$)", target)
    return family, int(match.group(1)) if match else None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def profile_config(
    base: dict[str, Any],
    identifier: str,
    *,
    pairs: tuple[str, ...],
    event_cache_dir: Path,
    event_scope_end: pd.Timestamp,
    event_feature_shift_hours: int = 0,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["freqai"]["enabled"] = True
    config["freqai"]["identifier"] = identifier
    config["freqai"]["purge_old_models"] = 1
    config["freqai"]["train_period_days"] = 365
    config["freqai"]["backtest_period_days"] = 90
    config["freqai"]["feature_parameters"]["label_period_candles"] = 24
    # FreqAI writes one HTML importance plot per target, pair, and rolling fold.
    # Models remain saved for later importance inspection; avoid thousands of
    # redundant plots in the first-round four-profile ladder.
    config["freqai"]["feature_parameters"]["plot_feature_importances"] = 0
    config["freqai"]["data_split_parameters"] = {
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": False,
    }
    config["exchange"]["pair_whitelist"] = list(pairs)
    config["sieve3_event_reaction"] = {
        "event_cache_dir": str(event_cache_dir.resolve()),
        "event_scope_end": event_scope_end.isoformat(),
        "event_feature_shift_hours": int(event_feature_shift_hours),
    }
    return config


def build_manifest(
    *,
    run_dir: Path,
    profiles: list[str],
    python_exe: Path,
    timerange: str,
    base_config: Path,
    pairs: tuple[str, ...],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    event_cache_dir: Path,
) -> dict[str, Any]:
    run_id = f"sieve3_event_generation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    base = json.loads(base_config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        profile_dir = run_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config_path = profile_dir / "config.json"
        atomic_json(
            config_path,
            profile_config(
                base,
                identifier,
                pairs=pairs,
                event_cache_dir=event_cache_dir,
                event_scope_end=max(end for _, end in windows.values()),
                event_feature_shift_hours=int(
                    definition.get("event_feature_shift_hours", 0)
                ),
            ),
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
            str(definition["strategy"]),
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
        "run_id": run_id,
        "created_at": now_iso(),
        "objective": "Test whether exact Sieve3 entry identities improve multi-horizon market-reaction predictions beyond an OHLCV control and a 168-hour displaced event-identity placebo.",
        "timerange": timerange,
        "pairs": list(pairs),
        "event_cache_dir": str(event_cache_dir.resolve()),
        "timestamp_rule": "FreqAI prediction `date` is the candle-open timestamp. The observable decision time is `date + 1h`; frozen windows are applied to decision time, and exact event decision_time is joined to `date = decision_time - 1h`.",
        "windows": {key: [str(start), str(end)] for key, (start, end) in windows.items()},
        "targets": TARGETS,
        "controls": ["price_control", "same pair/window target comparison", "direct matched ordinary candles", "168h shifted event placebo"],
        "commands": commands,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path, log_path: Path) -> int:
    total = len(manifest["commands"])
    for index, item in enumerate(manifest["commands"], start=1):
        if item.get("status") == "completed":
            continue
        output_dir = Path(item["output_dir"])
        item["status"] = "running"
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item.pop("finished_at", None)
        item.pop("returncode", None)
        attempt = int(item["attempts"])
        stdout_path = output_dir / f"freqai_stdout_attempt_{attempt}.log"
        stderr_path = output_dir / f"freqai_stderr_attempt_{attempt}.log"
        item["started_at"] = now_iso()
        item["stdout_log"] = str(stdout_path)
        item["stderr_log"] = str(stderr_path)
        atomic_json(manifest_path, manifest)
        append_log(
            log_path,
            f"FreqAI profile `{item['profile_id']}` started. Theory: {item['theory']} Identifier: `{item['identifier']}`.",
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                [str(part) for part in item["command"]],
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        item["finished_at"] = now_iso()
        item["returncode"] = int(result.returncode)
        item["status"] = "completed" if result.returncode == 0 else "failed"
        atomic_json(manifest_path, manifest)
        append_log(
            log_path,
            f"FreqAI profile `{item['profile_id']}` {item['status']} with return code {result.returncode}. Logs: `{stdout_path}`, `{stderr_path}`.",
        )
        print(
            json.dumps(
                {
                    "phase": "freqai_profiles",
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


def prediction_pair(path: Path, pairs: tuple[str, ...]) -> str | None:
    stem = path.stem.lower()
    for pair in pairs:
        token = re.escape(pair_token(pair))
        if re.search(rf"(?:^|_)cb_{token}(?:_|$)", stem) or re.search(
            rf"^{token}(?:_|$)", stem
        ):
            return pair
    return None


def load_predictions(identifier: str, pairs: tuple[str, ...]) -> DataFrame:
    frames: list[DataFrame] = []
    prediction_dir = MODELS_DIR / identifier / "backtesting_predictions"
    for path in sorted(prediction_dir.glob("*_prediction.feather")):
        pair = prediction_pair(path, pairs)
        if pair is None:
            continue
        frame = pd.read_feather(path)
        frame["pair"] = pair
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    return (
        out.dropna(subset=["date"])
        .drop_duplicates(["pair", "date"], keep="last")
        .sort_values(["pair", "date"])
        .reset_index(drop=True)
    )


def actuals_for_pair(pair: str) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    labelled = build_sieve3_reaction_targets(raw)
    keep = ["date", *[target for target in TARGETS if target in labelled]]
    labelled = labelled[keep].rename(
        columns={target: f"{target}_actual" for target in TARGETS if target in labelled}
    )
    return labelled


def event_scope_for_pair(pair: str, event_cache_dir: Path) -> DataFrame:
    cache = event_cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet"
    frame = pd.read_parquet(cache)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["date"] = frame["decision_time"] - pd.Timedelta(hours=1)
    return frame.drop(columns=["decision_time"])


def exit_scopes_for_pair(
    pair: str, event_cache_dir: Path
) -> dict[str, set[pd.Timestamp]]:
    path = event_cache_dir / "exit_events_long.parquet"
    if not path.exists():
        return {}
    exits = pd.read_parquet(
        path, columns=["pair", "decision_time", "exit_family", "entry_side"]
    )
    exits = exits[exits["pair"].eq(pair)].copy()
    exits["decision_time"] = pd.to_datetime(exits["decision_time"], utc=True, errors="coerce")
    exits["date"] = exits["decision_time"] - pd.Timedelta(hours=1)
    scopes: dict[str, set[pd.Timestamp]] = {
        "executed_exit:any": set(exits["date"].dropna())
    }
    for side, group in exits.groupby("entry_side", dropna=False):
        dates = set(group["date"].dropna())
        if len(dates) >= 20:
            scopes[f"executed_exit:any:{side}"] = dates
    for family, group in exits.groupby("exit_family", dropna=False):
        dates = set(group["date"].dropna())
        if len(dates) >= 20:
            scopes[f"executed_exit:{family}"] = dates
        for side, side_group in group.groupby("entry_side", dropna=False):
            side_dates = set(side_group["date"].dropna())
            if len(side_dates) >= 20:
                scopes[f"executed_exit:{family}:{side}"] = side_dates
    return scopes


def score_values(valid: DataFrame, prediction: str, actual: str) -> dict[str, Any]:
    clean = valid[[prediction, actual]].apply(pd.to_numeric, errors="coerce").dropna()
    result: dict[str, Any] = {"rows": int(len(clean))}
    if len(clean) < 20 or clean[prediction].nunique() < 2 or clean[actual].nunique() < 2:
        return {**result, "status": "unscorable"}
    errors = clean[prediction] - clean[actual]
    residual_sum_squares = float(np.square(errors).sum())
    total_sum_squares = float(
        np.square(clean[actual] - clean[actual].mean()).sum()
    )
    ranked = clean.sort_values(prediction)
    bucket = max(1, int(len(ranked) * 0.2))
    result.update(
        {
            "status": "scored",
            "actual_mean": float(clean[actual].mean()),
            "prediction_mean": float(clean[prediction].mean()),
            "prediction_bias": float(errors.mean()),
            "mean_absolute_error": float(errors.abs().mean()),
            "root_mean_squared_error": float(np.sqrt(np.square(errors).mean())),
            "r_squared": float(1.0 - (residual_sum_squares / total_sum_squares)),
            "prediction_actual_spearman": float(clean[prediction].corr(clean[actual], method="spearman")),
            "top_quintile_actual_mean": float(ranked.tail(bucket)[actual].mean()),
            "bottom_quintile_actual_mean": float(ranked.head(bucket)[actual].mean()),
        }
    )
    result["top_minus_bottom"] = (
        result["top_quintile_actual_mean"] - result["bottom_quintile_actual_mean"]
    )
    unique_actual = set(clean[actual].astype(float).unique())
    if unique_actual.issubset({0.0, 1.0}) and len(unique_actual) == 2:
        result["roc_auc"] = float(roc_auc_score(clean[actual].astype(int), clean[prediction]))
        result["average_precision"] = float(
            average_precision_score(clean[actual].astype(int), clean[prediction])
        )
        clipped = clean[prediction].clip(0.0, 1.0)
        result["brier_score_clipped"] = float(
            np.square(clipped - clean[actual]).mean()
        )
    return result


def scope_masks(frame: DataFrame) -> dict[str, pd.Series]:
    index = frame.index
    masks = {"all": pd.Series(True, index=index)}
    onset_count = pd.to_numeric(frame.get("summary__onset_count", 0), errors="coerce").fillna(0)
    masks["any_entry_onset"] = onset_count.gt(0)
    masks["same_side_overlap"] = pd.to_numeric(
        frame.get("summary__same_side_onset_overlap", 0), errors="coerce"
    ).fillna(0).gt(0)
    masks["opposing_entry_conflict"] = pd.to_numeric(
        frame.get("summary__opposing_onset_conflict", 0), errors="coerce"
    ).fillna(0).gt(0)
    return masks


def score_manifest(manifest: dict[str, Any], run_dir: Path) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    windows = {
        str(name): (
            pd.to_datetime(bounds[0], utc=True),
            pd.to_datetime(bounds[1], utc=True),
        )
        for name, bounds in manifest["windows"].items()
    }
    event_cache_dir = Path(manifest["event_cache_dir"])
    actuals = {pair: actuals_for_pair(pair) for pair in pairs}
    event_scopes = {
        pair: event_scope_for_pair(pair, event_cache_dir) for pair in pairs
    }
    exit_scopes = {
        pair: exit_scopes_for_pair(pair, event_cache_dir) for pair in pairs
    }
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = load_predictions(str(item["identifier"]), pairs)
        for pair in pairs:
            pair_predictions = predictions[predictions["pair"].eq(pair)].copy()
            merged = pair_predictions.merge(actuals[pair], on="date", how="left")
            merged = merged.merge(event_scopes[pair], on="date", how="left")
            merged["decision_time"] = merged["date"] + pd.Timedelta(hours=1)
            if "do_predict" in merged:
                merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)]
            for window, (start, end) in windows.items():
                window_frame = merged[
                    merged["decision_time"].ge(start)
                    & merged["decision_time"].lt(end)
                ].copy()
                masks = scope_masks(window_frame)
                for scope, mask in masks.items():
                    scoped = window_frame[mask]
                    for target in TARGETS:
                        actual = f"{target}_actual"
                        base = {
                            "profile_id": item["profile_id"],
                            "identifier": item["identifier"],
                            "pair": pair,
                            "window": window,
                            "scope": scope,
                            "target": target,
                        }
                        if target not in scoped or actual not in scoped:
                            rows.append({**base, "status": "missing_column", "rows": 0})
                        else:
                            rows.append({**base, **score_values(scoped, target, actual)})
                onset_columns = [column for column in window_frame if column.startswith("entry_onset__")]
                for column in onset_columns:
                    selected = pd.to_numeric(window_frame[column], errors="coerce").fillna(0).gt(0)
                    if int(selected.sum()) < 20:
                        continue
                    entry_id = column.removeprefix("entry_onset__")
                    for target in TARGETS:
                        actual = f"{target}_actual"
                        rows.append(
                            {
                                "profile_id": item["profile_id"],
                                "identifier": item["identifier"],
                                "pair": pair,
                                "window": window,
                                "scope": f"entry:{entry_id}",
                                "target": target,
                                **score_values(window_frame[selected], target, actual),
                            }
                        )
                for scope, dates in exit_scopes[pair].items():
                    selected = window_frame["date"].isin(dates)
                    if int(selected.sum()) < 20:
                        continue
                    for target in TARGETS:
                        actual = f"{target}_actual"
                        rows.append(
                            {
                                "profile_id": item["profile_id"],
                                "identifier": item["identifier"],
                                "pair": pair,
                                "window": window,
                                "scope": scope,
                                "target": target,
                                **score_values(window_frame[selected], target, actual),
                            }
                        )
    scores = DataFrame(rows)
    if scores.empty:
        return scores
    metadata = scores["target"].map(target_metadata)
    scores["target_family"] = metadata.map(lambda value: value[0])
    scores["horizon_hours"] = metadata.map(lambda value: value[1])
    controls = scores[scores["profile_id"].eq("price_control")][
        [
            "pair",
            "window",
            "scope",
            "target",
            "prediction_actual_spearman",
            "top_minus_bottom",
            "mean_absolute_error",
            "root_mean_squared_error",
            "r_squared",
            "prediction_bias",
            "roc_auc",
            "average_precision",
            "brier_score_clipped",
        ]
    ].rename(
        columns={
            column: f"control_{column}"
            for column in (
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
        }
    )
    scores = scores.merge(controls, on=["pair", "window", "scope", "target"], how="left")
    event_placebos = scores[scores["profile_id"].eq("event_identity_placebo")][
        [
            "pair",
            "window",
            "scope",
            "target",
            "prediction_actual_spearman",
            "top_minus_bottom",
            "mean_absolute_error",
            "root_mean_squared_error",
            "r_squared",
            "prediction_bias",
            "roc_auc",
            "average_precision",
            "brier_score_clipped",
        ]
    ].rename(
        columns={
            column: f"event_placebo_{column}"
            for column in (
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
        }
    )
    scores = scores.merge(
        event_placebos,
        on=["pair", "window", "scope", "target"],
        how="left",
    )
    for metric in (
        "prediction_actual_spearman",
        "top_minus_bottom",
        "r_squared",
        "roc_auc",
        "average_precision",
    ):
        scores[f"{metric}_delta_vs_price"] = pd.to_numeric(
            scores.get(metric), errors="coerce"
        ) - pd.to_numeric(scores.get(f"control_{metric}"), errors="coerce")
        scores[f"{metric}_delta_vs_event_placebo"] = pd.to_numeric(
            scores.get(metric), errors="coerce"
        ) - pd.to_numeric(scores.get(f"event_placebo_{metric}"), errors="coerce")
    for metric in (
        "mean_absolute_error",
        "root_mean_squared_error",
        "brier_score_clipped",
    ):
        scores[f"{metric}_skill_vs_price"] = pd.to_numeric(
            scores.get(f"control_{metric}"), errors="coerce"
        ) - pd.to_numeric(scores.get(metric), errors="coerce")
        scores[f"{metric}_skill_vs_event_placebo"] = pd.to_numeric(
            scores.get(f"event_placebo_{metric}"), errors="coerce"
        ) - pd.to_numeric(scores.get(metric), errors="coerce")
    scores["absolute_bias_skill_vs_price"] = pd.to_numeric(
        scores.get("control_prediction_bias"), errors="coerce"
    ).abs() - pd.to_numeric(scores.get("prediction_bias"), errors="coerce").abs()
    scores["absolute_bias_skill_vs_event_placebo"] = pd.to_numeric(
        scores.get("event_placebo_prediction_bias"), errors="coerce"
    ).abs() - pd.to_numeric(scores.get("prediction_bias"), errors="coerce").abs()
    scores.to_csv(run_dir / "freqai_reaction_scores.csv", index=False)
    return scores


def portability_summary(scores: DataFrame, run_dir: Path) -> DataFrame:
    if scores.empty:
        summary = DataFrame()
    else:
        scored = scores[
            scores["status"].eq("scored")
            & scores["scope"].isin(["all", "any_entry_onset"])
            & ~scores["profile_id"].isin(["price_control", "event_identity_placebo"])
        ].copy()
        scored["positive_spearman_delta"] = pd.to_numeric(
            scored["prediction_actual_spearman_delta_vs_price"], errors="coerce"
        ).gt(0)
        scored["positive_mae_skill"] = pd.to_numeric(
            scored["mean_absolute_error_skill_vs_price"], errors="coerce"
        ).gt(0)
        summary = (
            scored.groupby(["profile_id", "scope", "target"], dropna=False)
            .agg(
                pair_window_tests=("pair", "size"),
                distinct_pairs=("pair", "nunique"),
                positive_pair_windows=("positive_spearman_delta", "sum"),
                positive_mae_skill_pair_windows=("positive_mae_skill", "sum"),
                median_spearman=("prediction_actual_spearman", "median"),
                median_spearman_delta_vs_price=("prediction_actual_spearman_delta_vs_price", "median"),
                median_mae_skill_vs_price=("mean_absolute_error_skill_vs_price", "median"),
                median_rmse_skill_vs_price=("root_mean_squared_error_skill_vs_price", "median"),
                median_r_squared_delta_vs_price=("r_squared_delta_vs_price", "median"),
                median_absolute_bias_skill_vs_price=("absolute_bias_skill_vs_price", "median"),
                median_top_bottom=("top_minus_bottom", "median"),
                median_top_bottom_delta_vs_price=("top_minus_bottom_delta_vs_price", "median"),
                median_spearman_delta_vs_event_placebo=(
                    "prediction_actual_spearman_delta_vs_event_placebo",
                    "median",
                ),
                median_top_bottom_delta_vs_event_placebo=(
                    "top_minus_bottom_delta_vs_event_placebo",
                    "median",
                ),
                median_mae_skill_vs_event_placebo=(
                    "mean_absolute_error_skill_vs_event_placebo",
                    "median",
                ),
                median_rmse_skill_vs_event_placebo=(
                    "root_mean_squared_error_skill_vs_event_placebo",
                    "median",
                ),
                median_r_squared_delta_vs_event_placebo=(
                    "r_squared_delta_vs_event_placebo",
                    "median",
                ),
            )
            .reset_index()
            .sort_values(
                ["positive_pair_windows", "median_spearman_delta_vs_price"],
                ascending=False,
            )
        )
    summary.to_csv(run_dir / "freqai_portability_summary.csv", index=False)
    if scores.empty:
        family_summary = DataFrame()
    else:
        family_scored = scores[
            scores["status"].eq("scored")
            & scores["scope"].isin(["all", "any_entry_onset"])
            & ~scores["profile_id"].isin(["price_control", "event_identity_placebo"])
        ].copy()
        family_scored["positive_spearman_delta"] = pd.to_numeric(
            family_scored["prediction_actual_spearman_delta_vs_price"], errors="coerce"
        ).gt(0)
        family_scored["positive_mae_skill"] = pd.to_numeric(
            family_scored["mean_absolute_error_skill_vs_price"], errors="coerce"
        ).gt(0)
        family_summary = (
            family_scored.groupby(
                ["profile_id", "scope", "target_family", "horizon_hours"],
                dropna=False,
            )
            .agg(
                target_pair_window_tests=("target", "size"),
                distinct_targets=("target", "nunique"),
                distinct_pairs=("pair", "nunique"),
                positive_delta_share=("positive_spearman_delta", "mean"),
                positive_mae_skill_share=("positive_mae_skill", "mean"),
                median_spearman_delta_vs_price=(
                    "prediction_actual_spearman_delta_vs_price",
                    "median",
                ),
                median_mae_skill_vs_price=(
                    "mean_absolute_error_skill_vs_price",
                    "median",
                ),
                median_rmse_skill_vs_price=(
                    "root_mean_squared_error_skill_vs_price",
                    "median",
                ),
                median_r_squared_delta_vs_price=(
                    "r_squared_delta_vs_price",
                    "median",
                ),
                median_top_bottom_delta_vs_price=(
                    "top_minus_bottom_delta_vs_price",
                    "median",
                ),
                median_spearman_delta_vs_event_placebo=(
                    "prediction_actual_spearman_delta_vs_event_placebo",
                    "median",
                ),
                median_top_bottom_delta_vs_event_placebo=(
                    "top_minus_bottom_delta_vs_event_placebo",
                    "median",
                ),
                median_mae_skill_vs_event_placebo=(
                    "mean_absolute_error_skill_vs_event_placebo",
                    "median",
                ),
                median_rmse_skill_vs_event_placebo=(
                    "root_mean_squared_error_skill_vs_event_placebo",
                    "median",
                ),
                median_r_squared_delta_vs_event_placebo=(
                    "r_squared_delta_vs_event_placebo",
                    "median",
                ),
            )
            .reset_index()
            .sort_values(
                ["positive_delta_share", "median_spearman_delta_vs_price"],
                ascending=False,
            )
        )
    family_summary.to_csv(run_dir / "freqai_target_family_summary.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and score the bounded Sieve3 event-reaction FreqAI ladder.")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timerange", default="20240401-20260401")
    parser.add_argument("--pairs", default=",".join(BASELINE_PAIRS))
    parser.add_argument("--windows-json", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)
    event_cache_dir = args.output_dir / "cache"

    selected = [token.strip() for token in args.profiles.split(",") if token.strip()]
    unknown = [token for token in selected if token not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    for required in (
        args.python_exe,
        args.config,
        event_cache_dir / "entry_events_long.parquet",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    run_dir = args.output_dir / "freqai"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    log_path = research_log_path(args.output_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(
            run_dir=run_dir,
            profiles=selected,
            python_exe=args.python_exe,
            timerange=args.timerange,
            base_config=args.config,
            pairs=pairs,
            windows=windows,
            event_cache_dir=event_cache_dir,
        )
        atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    returncode = run_manifest(manifest, manifest_path, log_path)
    if returncode != 0:
        return returncode
    scores = score_manifest(manifest, run_dir)
    portability = portability_summary(scores, run_dir)
    summary = {
        "finished_at": now_iso(),
        "profiles": len(manifest["commands"]),
        "score_rows": int(len(scores)),
        "portable_summary_rows": int(len(portability)),
        "scores": str(run_dir / "freqai_reaction_scores.csv"),
        "portability": str(run_dir / "freqai_portability_summary.csv"),
        "target_family_summary": str(run_dir / "freqai_target_family_summary.csv"),
    }
    atomic_json(run_dir / "freqai_run_summary.json", summary)
    append_log(
        log_path,
        f"FreqAI generation ladder completed with {len(scores)} scored/screened rows and {len(portability)} cross-pair/window portability summaries. Results: `{summary['scores']}`.",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
