from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[5]
USER_DATA_DIR = REPO_ROOT / "user_data"
RESEARCH_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "3ki_freqai_2020_2022"
CONFIG_PATH = RESEARCH_ROOT / "configs" / "config_3ki_historical_news_freqai_btc_futures.json"
STRATEGY_DIR = RESEARCH_ROOT / "strategies"
RUNS_ROOT = RESEARCH_ROOT / "runs"
MODELS_DIR = USER_DATA_DIR / "models"
FUTURES_1H = USER_DATA_DIR / "data" / "binance" / "futures" / "BTC_USDT_USDT-1h-futures.feather"
DEFAULT_PYTHON = REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest" / "Scripts" / "python.exe"
DEFAULT_TIMERANGE = "20200401-20220729"
TARGETS = (
    "&-trend_up_24h",
    "&-trend_down_24h",
    "&-breakout_up_24h",
    "&-shock_down_24h",
    "&-long_reward_before_danger_24h",
    "&-long_danger_before_reward_24h",
    "&-short_reward_before_danger_24h",
    "&-short_danger_before_reward_24h",
    "&-trend_up_72h",
    "&-trend_down_72h",
    "&-long_reward_before_danger_72h",
    "&-short_reward_before_danger_72h",
)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    strategy: str
    timeframes: tuple[str, ...]
    description: str
    control_group: str


JOBS: tuple[JobSpec, ...] = (
    JobSpec("price_only_1h", "ThreeKiHistoricalNewsPriceOnlyFreqAIStrategy", ("1h",), "OHLCV price-only control.", "1h"),
    JobSpec("price_tech_1h", "ThreeKiHistoricalPriceTechFreqAIStrategy", ("1h",), "Existing price plus structure/VP-style technical control.", "1h"),
    JobSpec("price_only_mtf", "ThreeKiHistoricalNewsPriceOnlyFreqAIStrategy", ("1h", "4h", "1d"), "Multi-timeframe OHLCV control.", "mtf"),
    JobSpec("ta_macd_bb_volume_1h", "ThreeKiHistoricalTaMacdBbVolumeFreqAIStrategy", ("1h",), "MACD turn plus Bollinger position plus volume expansion.", "1h"),
    JobSpec("ta_macd_ma_trend_1h", "ThreeKiHistoricalTaMacdMaTrendFreqAIStrategy", ("1h",), "MACD momentum with moving-average trend alignment.", "1h"),
    JobSpec("ta_rsi_macd_recovery_1h", "ThreeKiHistoricalTaRsiMacdRecoveryFreqAIStrategy", ("1h",), "RSI recovery/loss confirmed by MACD and volume pressure.", "1h"),
    JobSpec("ta_bollinger_squeeze_break_1h", "ThreeKiHistoricalTaBollingerSqueezeBreakFreqAIStrategy", ("1h",), "Bollinger squeeze followed by range break and pressure.", "1h"),
    JobSpec("ta_ma_stack_trend_1h", "ThreeKiHistoricalTaMaStackTrendFreqAIStrategy", ("1h",), "Moving-average stack and trend-pressure state.", "1h"),
    JobSpec("ta_oscillator_reversion_1h", "ThreeKiHistoricalTaOscillatorReversionFreqAIStrategy", ("1h",), "Oscillator overextended/recovery states around range position.", "1h"),
    JobSpec("ta_candle_momentum_1h", "ThreeKiHistoricalTaCandleMomentumFreqAIStrategy", ("1h",), "Candlestick pattern clusters with higher-high/lower-low momentum.", "1h"),
    JobSpec("ta_candle_oscillator_1h", "ThreeKiHistoricalTaCandleOscillatorFreqAIStrategy", ("1h",), "Candlestick patterns confirmed or challenged by oscillators.", "1h"),
    JobSpec("ta_adx_trend_pressure_1h", "ThreeKiHistoricalTaAdxTrendPressureFreqAIStrategy", ("1h",), "ADX/DI trend strength with MA and pressure confirmation.", "1h"),
    JobSpec("ta_volume_price_pressure_1h", "ThreeKiHistoricalTaVolumePricePressureFreqAIStrategy", ("1h",), "Volume, signed pressure, range travel, and momentum.", "1h"),
    JobSpec("ta_range_compression_expansion_1h", "ThreeKiHistoricalTaRangeCompressionExpansionFreqAIStrategy", ("1h",), "Compression, expansion, breakout, and breakdown states.", "1h"),
    JobSpec("ta_vp_bb_confluence_1h", "ThreeKiHistoricalTaVpBbConfluenceFreqAIStrategy", ("1h",), "Volume-profile/value-area state with Bollinger and pressure.", "1h"),
    JobSpec("ta_vp_ma_confluence_1h", "ThreeKiHistoricalTaVpMaConfluenceFreqAIStrategy", ("1h",), "Volume-profile/value-area state with MA trend and pressure.", "1h"),
    JobSpec("ta_vp_osc_rejection_1h", "ThreeKiHistoricalTaVpOscillatorRejectionFreqAIStrategy", ("1h",), "Value-area rejection/reclaim with oscillator exhaustion/recovery.", "1h"),
    JobSpec("ta_mtf_trend_agreement", "ThreeKiHistoricalTaMtfTrendAgreementFreqAIStrategy", ("1h", "4h", "1d"), "1h/4h/1d trend and range agreement.", "mtf"),
    JobSpec("ta_mtf_momentum_agreement", "ThreeKiHistoricalTaMtfMomentumAgreementFreqAIStrategy", ("1h", "4h", "1d"), "1h/4h/1d MACD/oscillator/volume momentum agreement.", "mtf"),
    JobSpec("ta_mtf_daily_regime", "ThreeKiHistoricalTaMtfDailyRegimeFreqAIStrategy", ("1h", "4h", "1d"), "Daily regime and intraday state alignment.", "mtf"),
    JobSpec("ta_breakout_continuation_mtf", "ThreeKiHistoricalTaBreakoutContinuationFreqAIStrategy", ("1h", "4h", "1d"), "Breakout continuation from compression with MTF context.", "mtf"),
    JobSpec("ta_breakdown_continuation_mtf", "ThreeKiHistoricalTaBreakdownContinuationFreqAIStrategy", ("1h", "4h", "1d"), "Breakdown continuation from compression with MTF context.", "mtf"),
    JobSpec("ta_all_balanced_mtf", "ThreeKiHistoricalTaAllBalancedFreqAIStrategy", ("1h", "4h", "1d"), "Broad balanced TA, candle, volume, trend, and confluence profile.", "mtf"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 20 FreqAI TA/confluence feature combinations.")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timerange", default=DEFAULT_TIMERANGE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--only-jobs", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.python_exe.exists():
        raise FileNotFoundError(args.python_exe)
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    if not FUTURES_1H.exists():
        raise FileNotFoundError(FUTURES_1H)

    selected = select_jobs(args.only_jobs, args.max_jobs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"ta_confluence_freqai_sweep_{args.timerange}_{stamp}"
    run_root = RUNS_ROOT / safe_name(run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "ta_confluence_sweep_manifest.json"
    ledger_path = run_root / "ta_confluence_sweep_ledger.csv"
    scores_path = run_root / "ta_confluence_sweep_score_metrics.csv"
    comparison_path = run_root / "ta_confluence_sweep_comparison.csv"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timerange": args.timerange,
        "python_exe": str(args.python_exe),
        "jobs": [job.__dict__ for job in selected],
        "targets": list(TARGETS),
        "notes": [
            "FreqAI runs do the modelling; this script only prepares configs and scores FreqAI prediction files.",
            "Controls are run in the same sweep for 1h and MTF feature groups.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "run_root": str(run_root), "jobs": len(selected)}, indent=2))
        return 0

    actuals = build_actuals()
    ledger_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for job in selected:
        result = run_job(job, args.python_exe, args.timerange, run_root)
        ledger_rows.append(result)
        pd.DataFrame(ledger_rows).to_csv(ledger_path, index=False)
        if result["status"] == "completed":
            score_rows.extend(score_job(job, result, actuals))
            pd.DataFrame(score_rows).to_csv(scores_path, index=False)

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        comparison = add_baseline_comparisons(scores)
        comparison.to_csv(comparison_path, index=False)
    else:
        pd.DataFrame().to_csv(comparison_path, index=False)
    print(json.dumps({"status": "done", "run_root": str(run_root), "ledger": str(ledger_path), "scores": str(scores_path), "comparison": str(comparison_path)}, indent=2))
    return 0


def select_jobs(only_jobs: str, max_jobs: int) -> list[JobSpec]:
    jobs = list(JOBS)
    if only_jobs:
        wanted = {item.strip() for item in only_jobs.split(",") if item.strip()}
        jobs = [job for job in jobs if job.job_id in wanted]
    if max_jobs > 0:
        control_ids = {"price_only_1h", "price_tech_1h", "price_only_mtf"}
        controls = [job for job in jobs if job.job_id in control_ids]
        non_controls = [job for job in jobs if job.job_id not in control_ids]
        jobs = controls + non_controls[:max_jobs]
    return jobs


def run_job(job: JobSpec, python_exe: Path, timerange: str, run_root: Path) -> dict[str, Any]:
    job_dir = run_root / job.job_id
    backtest_dir = job_dir / "backtests"
    job_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    identifier = f"3ki-ta-conf-{safe_name(run_root.name)}-{job.job_id}-v1"
    config_path = job_dir / "config.json"
    write_config(config_path, identifier, job.timeframes)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    logfile_path = job_dir / "freqtrade.log"
    command = [
        str(python_exe),
        "-m",
        "freqtrade",
        "backtesting",
        "--userdir",
        str(USER_DATA_DIR),
        "--strategy-path",
        str(STRATEGY_DIR),
        "--config",
        str(config_path),
        "--strategy",
        job.strategy,
        "--freqaimodel",
        "LightGBMRegressorMultiTarget",
        "--timerange",
        timerange,
        "--export",
        "signals",
        "--export-directory",
        str(backtest_dir),
        "--logfile",
        str(logfile_path),
    ]
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.run(command, cwd=str(REPO_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    result_files = sorted(str(path) for path in backtest_dir.glob("*"))
    return {
        "job_id": job.job_id,
        "strategy": job.strategy,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": int(process.returncode),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "timeframes": ",".join(job.timeframes),
        "control_group": job.control_group,
        "description": job.description,
        "config_path": str(config_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "logfile": str(logfile_path),
        "backtest_dir": str(backtest_dir),
        "result_files": ";".join(result_files),
        "command": " ".join(command),
    }


def write_config(path: Path, identifier: str, timeframes: tuple[str, ...]) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    freqai = config.setdefault("freqai", {})
    freqai["identifier"] = identifier
    freqai["train_period_days"] = 120
    freqai["backtest_period_days"] = 365
    feature_parameters = freqai.setdefault("feature_parameters", {})
    feature_parameters["include_timeframes"] = list(timeframes)
    feature_parameters["indicator_periods_candles"] = [6, 12, 24]
    feature_parameters["plot_feature_importances"] = 0
    freqai["model_training_parameters"] = {
        "n_estimators": 20,
        "learning_rate": 0.04,
        "num_leaves": 5,
        "max_depth": 2,
        "min_child_samples": 18,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 1,
        "verbosity": -1,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def build_actuals() -> pd.DataFrame:
    frame = pd.read_feather(FUTURES_1H)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    out = pd.DataFrame({"date": frame["date"]})
    for horizon in (1, 3, 6, 24, 72):
        future_return = close.shift(-horizon) / close - 1.0
        out[f"&-future_return_{horizon}h"] = future_return
        out[f"&-future_up_{horizon}h"] = binary_label(future_return, future_return > 0)
    for horizon in (3, 6, 24, 72):
        future_max = future_close_extreme(close, horizon, "max")
        future_min = future_close_extreme(close, horizon, "min")
        out[f"&-future_max_upside_{horizon}h"] = future_max / close - 1.0
        out[f"&-future_max_drawdown_{horizon}h"] = future_min / close - 1.0
    for horizon, threshold in {3: -0.02, 6: -0.025, 24: -0.04, 72: -0.05}.items():
        source = out.get(f"&-future_max_drawdown_{horizon}h")
        if source is not None:
            out[f"&-shock_down_{horizon}h"] = binary_label(source, source <= threshold)
    for horizon, threshold in {3: 0.02, 6: 0.025, 24: 0.04, 72: 0.05}.items():
        source = out.get(f"&-future_max_upside_{horizon}h")
        if source is not None:
            out[f"&-breakout_up_{horizon}h"] = binary_label(source, source >= threshold)
    for horizon, threshold in {24: 0.03, 72: 0.05}.items():
        future_return = out[f"&-future_return_{horizon}h"]
        out[f"&-future_down_{horizon}h"] = binary_label(future_return, future_return < 0)
        out[f"&-trend_up_{horizon}h"] = binary_label(future_return, future_return >= threshold)
        out[f"&-trend_down_{horizon}h"] = binary_label(future_return, future_return <= -threshold)
        ordered = ordered_target_stop_labels(close, high, low, horizon, reward_threshold=0.03)
        for name, values in ordered.items():
            out[f"&-{name}_{horizon}h"] = values
    return out


def score_job(job: JobSpec, result: dict[str, Any], actuals: pd.DataFrame) -> list[dict[str, Any]]:
    predictions = load_predictions(str(result["identifier"]))
    if predictions.empty:
        return [
            {
                "job_id": job.job_id,
                "strategy": job.strategy,
                "target": "",
                "status": "missing_predictions",
                "identifier": result["identifier"],
            }
        ]
    merged = predictions.merge(actuals, on="date", how="left", suffixes=("_pred", "_actual"))
    if "do_predict" in merged:
        merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)].copy()
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        prediction_column = f"{target}_pred" if f"{target}_pred" in merged else target
        actual_column = f"{target}_actual" if f"{target}_actual" in merged else target
        if prediction_column not in merged or actual_column not in merged:
            rows.append(base_score_row(job, result, target, "missing_column"))
            continue
        valid = merged[["date", prediction_column, actual_column]].copy()
        valid[prediction_column] = pd.to_numeric(valid[prediction_column], errors="coerce")
        valid[actual_column] = pd.to_numeric(valid[actual_column], errors="coerce")
        valid = valid.dropna()
        row = base_score_row(job, result, target, "scored")
        row["rows"] = int(len(valid))
        if valid.empty or valid[prediction_column].nunique() < 2 or valid[actual_column].nunique() < 2:
            row["positive_rate"] = float(valid[actual_column].mean()) if not valid.empty else None
            rows.append(row)
            continue
        y = valid[actual_column].astype(int)
        x = valid[prediction_column]
        row.update(
            {
                "positive_rate": float(y.mean()),
                "auc": float(roc_auc_score(y, x)),
                "ap": float(average_precision_score(y, x)),
                "prediction_actual_corr": float(x.corr(y)),
            }
        )
        ranked = valid.sort_values(prediction_column)
        bucket = max(1, int(len(ranked) * 0.10))
        bottom = ranked.head(bucket)
        top = ranked.tail(bucket)
        row["top10_rate"] = float(top[actual_column].mean())
        row["bottom10_rate"] = float(bottom[actual_column].mean())
        row["top_minus_bottom"] = row["top10_rate"] - row["bottom10_rate"]
        monthly = monthly_auc(valid, prediction_column, actual_column)
        row["months"] = int(len(monthly))
        row["months_auc_gt_50"] = int((monthly["auc"] > 0.50).sum()) if not monthly.empty else 0
        row["median_month_auc"] = float(monthly["auc"].median()) if not monthly.empty else None
        rows.append(row)
    return rows


def base_score_row(job: JobSpec, result: dict[str, Any], target: str, status: str) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "strategy": job.strategy,
        "target": target,
        "status": status,
        "identifier": result["identifier"],
        "timeframes": ",".join(job.timeframes),
        "control_group": job.control_group,
        "description": job.description,
        "backtest_dir": result["backtest_dir"],
        "result_files": result["result_files"],
    }


def add_baseline_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["is_control"] = out["job_id"].isin(["price_only_1h", "price_tech_1h", "price_only_mtf"])
    baseline_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for group, candidates in {
        "1h": ["price_only_1h", "price_tech_1h"],
        "mtf": ["price_only_mtf"],
    }.items():
        control_rows = out[out["job_id"].isin(candidates)].copy()
        for target, target_rows in control_rows.groupby("target"):
            ranked = target_rows[pd.to_numeric(target_rows["auc"], errors="coerce").notna()].copy()
            if ranked.empty:
                continue
            ranked["auc"] = pd.to_numeric(ranked["auc"], errors="coerce")
            ranked = ranked.sort_values("auc", ascending=False)
            best = ranked.iloc[0].to_dict()
            baseline_lookup[(group, str(target))] = best
    best_auc = []
    best_lift = []
    baseline_job = []
    for row in out.itertuples(index=False):
        baseline = baseline_lookup.get((str(row.control_group), str(row.target)))
        if not baseline:
            best_auc.append(None)
            best_lift.append(None)
            baseline_job.append("")
            continue
        baseline_job.append(baseline.get("job_id"))
        baseline_auc = pd.to_numeric(pd.Series([baseline.get("auc")]), errors="coerce").iloc[0]
        row_auc = pd.to_numeric(pd.Series([getattr(row, "auc", None)]), errors="coerce").iloc[0]
        baseline_top_bottom = pd.to_numeric(pd.Series([baseline.get("top_minus_bottom")]), errors="coerce").iloc[0]
        row_top_bottom = pd.to_numeric(pd.Series([getattr(row, "top_minus_bottom", None)]), errors="coerce").iloc[0]
        best_auc.append(float(row_auc - baseline_auc) if pd.notna(row_auc) and pd.notna(baseline_auc) else None)
        best_lift.append(float(row_top_bottom - baseline_top_bottom) if pd.notna(row_top_bottom) and pd.notna(baseline_top_bottom) else None)
    out["best_baseline_job"] = baseline_job
    out["auc_vs_best_baseline"] = best_auc
    out["lift_vs_best_baseline"] = best_lift
    out["classification"] = out.apply(classify_row, axis=1)
    return out.sort_values(
        ["classification", "auc_vs_best_baseline", "auc", "top_minus_bottom"],
        ascending=[True, False, False, False],
    )


def classify_row(row: pd.Series) -> str:
    if row.get("is_control"):
        return "0_control"
    auc_delta = number(row.get("auc_vs_best_baseline"))
    auc = number(row.get("auc"))
    lift_delta = number(row.get("lift_vs_best_baseline"))
    months = number(row.get("months"))
    months_good = number(row.get("months_auc_gt_50"))
    if auc >= 0.56 and auc_delta >= 0.015 and lift_delta > 0 and months >= 6 and months_good >= months * 0.50:
        return "1_exploratory_signal"
    if auc >= 0.54 and auc_delta > 0 and lift_delta >= 0:
        return "2_watchlist"
    if auc_delta < -0.01:
        return "4_worse_than_baseline"
    return "3_no_clear_lift"


def load_predictions(identifier: str) -> pd.DataFrame:
    prediction_dir = MODELS_DIR / identifier / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    frames = []
    for path in files:
        frame = pd.read_feather(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True, errors="coerce")
    return predictions.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def monthly_auc(valid: pd.DataFrame, prediction_column: str, actual_column: str) -> pd.DataFrame:
    work = valid.copy()
    work["month"] = work["date"].dt.tz_convert(None).dt.to_period("M").astype(str)
    rows = []
    for month, group in work.groupby("month"):
        if len(group) < 50 or group[actual_column].nunique() < 2 or group[prediction_column].nunique() < 2:
            continue
        rows.append({"month": month, "rows": int(len(group)), "auc": float(roc_auc_score(group[actual_column].astype(int), group[prediction_column]))})
    return pd.DataFrame(rows)


def future_close_extreme(close: pd.Series, horizon: int, method: str) -> pd.Series:
    future = close.shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    extreme = rolling.max() if method == "max" else rolling.min()
    return extreme.iloc[::-1]


def ordered_target_stop_labels(close: pd.Series, high: pd.Series, low: pd.Series, horizon: int, reward_threshold: float) -> dict[str, pd.Series]:
    danger_threshold = -float(reward_threshold)
    valid_source = close.shift(-horizon)
    long_reward_first = pd.Series(False, index=close.index)
    long_danger_first = pd.Series(False, index=close.index)
    short_reward_first = pd.Series(False, index=close.index)
    short_danger_first = pd.Series(False, index=close.index)
    unresolved = pd.Series(True, index=close.index)
    for step in range(1, horizon + 1):
        future_high = high.shift(-step)
        future_low = low.shift(-step)
        up_hit = (future_high / close - 1.0) >= reward_threshold
        down_hit = (future_low / close - 1.0) <= danger_threshold
        long_reward_first = long_reward_first | (unresolved & up_hit & ~down_hit)
        long_danger_first = long_danger_first | (unresolved & down_hit)
        short_reward_first = short_reward_first | (unresolved & down_hit & ~up_hit)
        short_danger_first = short_danger_first | (unresolved & up_hit)
        unresolved = unresolved & ~up_hit & ~down_hit
    return {
        "long_reward_before_danger": binary_label(valid_source, long_reward_first),
        "long_danger_before_reward": binary_label(valid_source, long_danger_first),
        "short_reward_before_danger": binary_label(valid_source, short_reward_first),
        "short_danger_before_reward": binary_label(valid_source, short_danger_first),
    }


def binary_label(source: pd.Series, condition: pd.Series) -> pd.Series:
    return condition.astype(float).where(source.notna())


def number(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float("nan")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
