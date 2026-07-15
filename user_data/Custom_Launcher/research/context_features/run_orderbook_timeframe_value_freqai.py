from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[4]
USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_REPORT_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "reports" / "orderbook_timeframe_value_freqai"
DEFAULT_MODELS_DIR = USER_DATA_DIR / "models"
DATASET_PATH = USER_DATA_DIR / "research_news_data" / "context_features" / "orderbook_timeframe_value" / "orderbook_timeframe_value_latest.parquet"
PAIR = "BTC/USDT:USDT"


PROFILES = {
    "price_control": {
        "strategy": "OrderbookTfPriceOnlyFreqAIStrategy",
        "theory": "Price and volume alone set the matched-row baseline.",
    },
    "ob_1s": {
        "strategy": "OrderbookTf1sFreqAIStrategy",
        "theory": "1-second orderbook metrics should add unique fast shock/pressure information.",
    },
    "ob_1m": {
        "strategy": "OrderbookTf1mFreqAIStrategy",
        "theory": "1-minute orderbook summaries should preserve most useful fast pressure/wall information.",
    },
    "ob_5m": {
        "strategy": "OrderbookTf5mFreqAIStrategy",
        "theory": "5-minute orderbook summaries may be enough for higher-timeframe trading decisions.",
    },
    "ob_1h": {
        "strategy": "OrderbookTf1hFreqAIStrategy",
        "theory": "1-hour orderbook summaries may preserve enough signal for 1h+ decisions.",
    },
    "ob_all": {
        "strategy": "OrderbookTfAllFreqAIStrategy",
        "theory": "Combining resolutions should improve if smaller units carry unique non-duplicated information.",
    },
}

TARGETS = [
    *[f"&-future_return_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-future_max_upside_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-future_max_drawdown_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-long_reward_before_danger_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-short_reward_before_danger_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-large_upside_next_{h}h" for h in (1, 2, 4, 6)],
    *[f"&-large_drawdown_next_{h}h" for h in (1, 2, 4, 6)],
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FreqAI profiles comparing orderbook source timeframes.")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--timerange", default="20260508-20260622")
    parser.add_argument("--profiles", default="price_control,ob_1s,ob_1m,ob_5m,ob_1h,ob_all")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--backtest-days", type=int, default=3)
    parser.add_argument("--plot-feature-importances", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.python_exe.exists():
        raise FileNotFoundError(args.python_exe)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    selected = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = [item for item in selected if item not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {unknown}")

    run_id = f"orderbook_tf_value_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = args.report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for profile_id in selected:
        profile = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        config_path = run_dir / f"{profile_id}_config.json"
        config_path.write_text(
            json.dumps(config(identifier, args.train_days, args.backtest_days, args.plot_feature_importances), indent=2) + "\n",
            encoding="utf-8",
        )
        export_dir = run_dir / profile_id
        export_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(USER_DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            profile["strategy"],
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            args.timerange,
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "identifier": identifier,
                "strategy": profile["strategy"],
                "theory": profile["theory"],
                "config_path": str(config_path),
                "export_dir": str(export_dir),
                "command": command,
                "status": "pending",
            }
        )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "objective": "Use FreqAI to compare whether 1s orderbook features add value over 1m, 5m, and 1h summaries for higher-timeframe BTC decisions.",
        "dataset": str(DATASET_PATH),
        "timerange": args.timerange,
        "pair": PAIR,
        "targets": TARGETS,
        "commands": commands,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path), "commands": len(commands)}, indent=2))
        return 0

    for item in commands:
        stdout_path = Path(str(item["export_dir"])) / "freqai_stdout.log"
        stderr_path = Path(str(item["export_dir"])) / "freqai_stderr.log"
        item["started_at"] = datetime.now().astimezone().isoformat()
        item["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run([str(part) for part in item["command"]], cwd=str(REPO_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
        item["finished_at"] = datetime.now().astimezone().isoformat()
        item["returncode"] = int(result.returncode)
        item["stdout_log"] = str(stdout_path)
        item["stderr_log"] = str(stderr_path)
        item["status"] = "completed" if result.returncode == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if result.returncode != 0:
            print(json.dumps({"status": "failed", "profile_id": item["profile_id"], "stderr": str(stderr_path)}, indent=2))
            return int(result.returncode)

    summary = score_manifest(manifest, args.models_dir)
    summary_path = run_dir / "orderbook_timeframe_value_metrics.csv"
    top_path = run_dir / "orderbook_timeframe_value_top.csv"
    summary.to_csv(summary_path, index=False)
    top_candidates(summary).to_csv(top_path, index=False)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "manifest": str(manifest_path), "metrics": str(summary_path), "top": str(top_path)}, indent=2))
    return 0


def config(identifier: str, train_days: int, backtest_days: int, plot_importances: int) -> dict[str, Any]:
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "dry_run": True,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 1,
        "fiat_display_currency": "USD",
        "timeframe": "1h",
        "dry_run_wallet": 1000,
        "cancel_open_orders_on_exit": True,
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "sandbox": False,
            "skip_pair_validation": True,
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [PAIR],
            "pair_blacklist": [],
        },
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 1.0,
            "check_depth_of_market": {"enabled": True, "bids_to_ask_delta": 1.0},
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 1.0,
        },
        "pairlists": [{"method": "StaticPairList"}],
        "freqai": {
            "enabled": True,
            "activate_tensorboard": False,
            "purge_old_models": 2,
            "train_period_days": train_days,
            "backtest_period_days": backtest_days,
            "live_retrain_hours": 0,
            "identifier": identifier,
            "feature_parameters": {
                "include_timeframes": ["1h"],
                "include_corr_pairlist": [],
                "label_period_candles": 6,
                "include_shifted_candles": 0,
                "DI_threshold": 0,
                "weight_factor": 0.9,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "indicator_periods_candles": [6],
                "plot_feature_importances": int(plot_importances),
            },
            "data_split_parameters": {"test_size": 0.25, "random_state": 1, "shuffle": False},
            "model_training_parameters": {
                "n_estimators": 80,
                "learning_rate": 0.04,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 12,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": 1,
                "verbosity": -1,
            },
        },
        "bot_name": "orderbook_timeframe_value_freqai_research",
        "force_entry_enable": True,
        "initial_state": "stopped",
        "internals": {"process_throttle_secs": 5},
    }


def score_manifest(manifest: dict[str, Any], models_dir: Path) -> DataFrame:
    actuals = actual_labels()
    actuals = actuals.rename(columns={target: f"{target}_actual" for target in TARGETS if target in actuals})
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = load_predictions(models_dir / str(item["identifier"]))
        if predictions.empty:
            rows.append({**row_base(item), "target": "", "status": "missing_predictions", "rows": 0})
            continue
        merged = predictions.merge(actuals, on="date", how="left")
        if "do_predict" in merged:
            merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)].copy()
        for target in TARGETS:
            rows.append(score_target(item, merged, target))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    for column in (
        "roc_auc",
        "average_precision",
        "prediction_actual_corr",
        "top_quintile_actual_rate",
        "bottom_quintile_actual_rate",
    ):
        if column not in summary:
            summary[column] = pd.NA
    control = summary[summary["profile_id"].eq("price_control") & summary["status"].eq("scored")][
        ["target", "roc_auc", "average_precision", "prediction_actual_corr", "top_quintile_actual_rate", "bottom_quintile_actual_rate"]
    ].rename(
        columns={
            "roc_auc": "control_roc_auc",
            "average_precision": "control_average_precision",
            "prediction_actual_corr": "control_prediction_actual_corr",
            "top_quintile_actual_rate": "control_top_quintile_actual_rate",
            "bottom_quintile_actual_rate": "control_bottom_quintile_actual_rate",
        }
    )
    summary = summary.merge(control, on="target", how="left")
    for metric in ("roc_auc", "average_precision", "prediction_actual_corr", "top_quintile_actual_rate", "bottom_quintile_actual_rate"):
        summary[f"{metric}_delta_vs_price"] = pd.to_numeric(summary.get(metric), errors="coerce") - pd.to_numeric(summary.get(f"control_{metric}"), errors="coerce")
    return summary


def actual_labels() -> DataFrame:
    frame = pd.read_parquet(DATASET_PATH)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    ready = pd.to_numeric(frame.get("obtf_all_usable"), errors="coerce").fillna(0.0).gt(0.0)
    for horizon in (1, 2, 4, 6):
        frame[f"&-future_return_{horizon}h"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
        upside = future_extreme(frame["high"], horizon, "max") / frame["close"] - 1.0
        drawdown = future_extreme(frame["low"], horizon, "min") / frame["close"] - 1.0
        frame[f"&-future_max_upside_{horizon}h"] = upside
        frame[f"&-future_max_drawdown_{horizon}h"] = drawdown
        reward, danger, large_move = path_thresholds(horizon)
        frame[f"&-long_reward_before_danger_{horizon}h"] = reward_before_danger(frame["close"], horizon=horizon, reward=reward, danger=-danger, long_side=True)
        frame[f"&-short_reward_before_danger_{horizon}h"] = reward_before_danger(frame["close"], horizon=horizon, reward=reward, danger=-danger, long_side=False)
        frame[f"&-large_upside_next_{horizon}h"] = frame[f"&-future_max_upside_{horizon}h"].ge(large_move).astype(float).where(frame[f"&-future_max_upside_{horizon}h"].notna())
        frame[f"&-large_drawdown_next_{horizon}h"] = frame[f"&-future_max_drawdown_{horizon}h"].le(-large_move).astype(float).where(frame[f"&-future_max_drawdown_{horizon}h"].notna())
    for target in TARGETS:
        if target in frame:
            frame[target] = frame[target].where(ready)
    keep = ["date", *[target for target in TARGETS if target in frame]]
    return frame[keep]


def load_predictions(model_dir: Path) -> DataFrame:
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    frames = []
    for path in files:
        frame = pd.read_feather(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    return out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def score_target(item: dict[str, Any], merged: DataFrame, target: str) -> dict[str, Any]:
    row = {**row_base(item), "target": target}
    actual_col = f"{target}_actual"
    if target not in merged or actual_col not in merged:
        row.update({"status": "missing_column", "rows": 0})
        return row
    valid = merged[[target, actual_col]].apply(pd.to_numeric, errors="coerce").dropna()
    row["rows"] = int(len(valid))
    if valid.empty or valid[target].nunique() < 2:
        row.update({"status": "unscorable", "actual_event_rate": None})
        return row
    row["status"] = "scored"
    row["actual_event_rate"] = float(valid[actual_col].mean())
    ranked = valid.sort_values(target)
    bucket_size = max(1, int(len(ranked) * 0.2))
    bottom = ranked.head(bucket_size)
    top = ranked.tail(bucket_size)
    row["top_quintile_actual_rate"] = float(top[actual_col].mean())
    row["bottom_quintile_actual_rate"] = float(bottom[actual_col].mean())
    row["top_lift_vs_bottom"] = row["top_quintile_actual_rate"] - row["bottom_quintile_actual_rate"]
    row["prediction_actual_corr"] = float(valid[target].corr(valid[actual_col]))
    actual_values = set(valid[actual_col].dropna().astype(float).unique())
    if actual_values.issubset({0.0, 1.0}) and len(actual_values) == 2:
        row["roc_auc"] = float(roc_auc_score(valid[actual_col].astype(int), valid[target]))
        row["average_precision"] = float(average_precision_score(valid[actual_col].astype(int), valid[target]))
    else:
        row["roc_auc"] = None
        row["average_precision"] = None
    return row


def top_candidates(summary: DataFrame) -> DataFrame:
    if summary.empty or "status" not in summary:
        return summary
    scored = summary[summary["status"].eq("scored") & ~summary["profile_id"].eq("price_control")].copy()
    if scored.empty:
        return scored
    scored["rank_score"] = (
        pd.to_numeric(scored.get("roc_auc_delta_vs_price"), errors="coerce").fillna(0.0)
        + pd.to_numeric(scored.get("average_precision_delta_vs_price"), errors="coerce").fillna(0.0)
        + pd.to_numeric(scored.get("top_quintile_actual_rate_delta_vs_price"), errors="coerce").fillna(0.0)
    )
    return scored.sort_values("rank_score", ascending=False).groupby("profile_id").head(8).reset_index(drop=True)


def row_base(item: dict[str, Any]) -> dict[str, Any]:
    return {"profile_id": item.get("profile_id"), "identifier": item.get("identifier"), "strategy": item.get("strategy"), "theory": item.get("theory")}


def future_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
    future = series.shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    result = rolling.max() if method == "max" else rolling.min()
    return result.iloc[::-1]


def path_thresholds(horizon: int) -> tuple[float, float, float]:
    if horizon <= 1:
        return 0.006, 0.004, 0.010
    if horizon <= 2:
        return 0.010, 0.006, 0.014
    if horizon <= 4:
        return 0.015, 0.010, 0.020
    return 0.025, 0.015, 0.030


def reward_before_danger(close: pd.Series, *, horizon: int, reward: float, danger: float, long_side: bool) -> pd.Series:
    values = close.to_numpy(dtype=float)
    out = []
    for idx, current in enumerate(values):
        if not pd.notna(current) or idx + horizon >= len(values):
            out.append(float("nan"))
            continue
        future = values[idx + 1 : idx + horizon + 1] / current - 1.0
        if long_side:
            reward_hits = [i for i, value in enumerate(future) if value >= reward]
            danger_hits = [i for i, value in enumerate(future) if value <= danger]
        else:
            reward_hits = [i for i, value in enumerate(future) if value <= -reward]
            danger_hits = [i for i, value in enumerate(future) if value >= -danger]
        if not reward_hits:
            out.append(0.0)
        elif not danger_hits:
            out.append(1.0)
        elif reward_hits[0] == danger_hits[0]:
            out.append(float("nan"))
        else:
            out.append(float(reward_hits[0] < danger_hits[0]))
    return pd.Series(out, index=close.index)


if __name__ == "__main__":
    raise SystemExit(main())
