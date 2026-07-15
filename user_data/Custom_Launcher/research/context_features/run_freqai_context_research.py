from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import subprocess
import sys

import pandas as pd


APP_DIR = Path(__file__).resolve().parents[2]
USER_DATA_DIR = APP_DIR.parent
REPO_ROOT = USER_DATA_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from launcher_v2.services.entry_sieve_lock import live_entry_sieve_runs  # noqa: E402
from research.context_features.builder import build_context_features, default_paths, read_feature_status  # noqa: E402


DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_context_freqai_research.example.json"
DEFAULT_STRATEGY = "ContextFreqAIResearchStrategy"
DEFAULT_FREQAI_MODEL = "SKLearnRidgeRegressorMultiTarget"
RUN_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_runs"
STATE_PATH = RUN_ROOT / "run_state.json"
ARCHETYPES_PATH = APP_DIR / "research" / "config" / "context_event_archetypes.json"
KNOWN_EVENTS_PATH = APP_DIR / "research" / "config" / "known_btc_event_windows.json"
RETURN_LABEL_HORIZONS = (1, 3, 6, 24, 72, 168, 336, 720, 2160)
PATH_LABEL_HORIZONS = (3, 6, 24, 72, 168, 720)
EVENT_LABELS = (
    "shock_down_3h",
    "shock_down_6h",
    "shock_down_24h",
    "shock_down_72h",
    "breakout_up_3h",
    "breakout_up_6h",
    "breakout_up_24h",
    "breakout_up_72h",
    "breakout_up_720h",
    "regime_up_2160h",
)
SHOCK_THRESHOLDS = {3: -0.02, 6: -0.025, 24: -0.04, 72: -0.05}
BREAKOUT_THRESHOLDS = {3: 0.02, 6: 0.025, 24: 0.04, 72: 0.05}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the context FreqAI research job when Entry Sieve is idle.")
    parser.add_argument("--ignore-sieve", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timerange", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--freqaimodel", default=DEFAULT_FREQAI_MODEL)
    parser.add_argument("--identifier", default="", help="Optional run-specific FreqAI identifier override.")
    parser.add_argument("--python-exe", type=Path, default=REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    state = _read_json(STATE_PATH)
    if state.get("status") == "completed" and not args.force:
        return _finish({"status": "skipped", "reason": "context FreqAI research already completed", "state": state})

    busy_runs = [] if args.ignore_sieve else live_entry_sieve_runs(APP_DIR / "launcher_v2" / "runtime" / "entry_sieve")
    if busy_runs:
        payload = {
            "status": "deferred",
            "reason": "Entry Sieve is still running",
            "busy_runs": _summarise_busy_runs(busy_runs),
            "checked_at": datetime.now().astimezone().isoformat(),
        }
        _write_json(STATE_PATH, payload)
        return _finish(payload)

    if args.check_only:
        return _finish({"status": "ready", "reason": "Entry Sieve is idle"})

    paths = default_paths(APP_DIR)
    feature_summary = build_context_features(paths, mode="update", overlap_days=7, export_parquet=True, make_report=True)
    timerange = args.timerange or _default_timerange(paths.feature_db, paths.btc_ohlcv_path)
    run_dir = RUN_ROOT / datetime.now().strftime("freqai_context_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_run_config(args.config, run_dir, args.identifier) if args.identifier else args.config
    freqai_identifier = _read_freqai_identifier(config_path)
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
        args.strategy,
        "--freqaimodel",
        args.freqaimodel,
        "--timerange",
        timerange,
        "--export",
        "signals",
        "--export-directory",
        str(run_dir),
    ]
    payload = {
        "status": "dry-run" if args.dry_run else "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "command": command,
        "feature_summary": feature_summary.to_dict(),
        "timerange": timerange,
        "freqai_identifier": freqai_identifier,
    }
    _write_json(STATE_PATH, payload)
    if args.dry_run:
        return _finish(payload)

    log_path = run_dir / "freqtrade_backtesting.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    completed = {
        **payload,
        "status": "completed" if process.returncode == 0 else "failed",
        "finished_at": datetime.now().astimezone().isoformat(),
        "returncode": int(process.returncode),
        "log_path": str(log_path),
        "feature_status": read_feature_status(paths.feature_db),
    }
    if process.returncode == 0:
        completed["prediction_reports"] = _write_freqai_prediction_reports(run_dir, freqai_identifier)
    completed["outputs"] = _list_outputs(run_dir)
    _write_json(run_dir / "summary.json", completed)
    _write_json(STATE_PATH, completed)
    return _finish(completed, returncode=process.returncode)


def _default_timerange(feature_db: Path, btc_ohlcv_path: Path) -> str:
    status = read_feature_status(feature_db)
    feature_start = pd.to_datetime(status.get("first_feature_hour"), utc=True, errors="coerce")
    feature_end = pd.to_datetime(status.get("last_feature_hour"), utc=True, errors="coerce")
    if not btc_ohlcv_path.exists() or pd.isna(feature_start) or pd.isna(feature_end):
        return "20260508-20260513"
    ohlcv = pd.read_feather(btc_ohlcv_path, columns=["date"])
    price_start = pd.to_datetime(ohlcv["date"].min(), utc=True, errors="coerce")
    price_end = pd.to_datetime(ohlcv["date"].max(), utc=True, errors="coerce")
    start = max(feature_start, price_start).strftime("%Y%m%d")
    end = (min(feature_end, price_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    return f"{start}-{end}"


def _write_run_config(config_path: Path, run_dir: Path, identifier: str) -> Path:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data.setdefault("freqai", {})["identifier"] = identifier
    output = run_dir / f"{config_path.stem}_{identifier}.json"
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return output


def _read_freqai_identifier(config_path: Path) -> str:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return str(data.get("freqai", {}).get("identifier") or "")


def _write_freqai_prediction_reports(run_dir: Path, identifier: str) -> list[str]:
    prediction_dir = USER_DATA_DIR / "models" / identifier / "backtesting_predictions"
    prediction_files = sorted(prediction_dir.glob("*_prediction.feather"))
    if not prediction_files:
        return []
    frames = []
    for path in prediction_files:
        frame = pd.read_feather(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True, errors="coerce")
    predictions = predictions.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    actuals = _load_actual_future_returns()
    merged = predictions.merge(actuals, on="date", how="left")
    if "do_predict" in merged:
        merged = merged[merged["do_predict"] == 1].copy()

    prediction_columns = ["date", "prediction_file", "do_predict"]
    for horizon in RETURN_LABEL_HORIZONS:
        prediction_columns.extend(
            [
                f"&-future_return_{horizon}h",
                f"actual_future_return_{horizon}h",
                f"&-future_up_{horizon}h",
                f"actual_future_up_{horizon}h",
            ]
        )
    for horizon in PATH_LABEL_HORIZONS:
        prediction_columns.extend(
            [
                f"&-future_max_upside_{horizon}h",
                f"actual_future_max_upside_{horizon}h",
                f"&-future_max_drawdown_{horizon}h",
                f"actual_future_max_drawdown_{horizon}h",
            ]
        )
    for label in EVENT_LABELS:
        prediction_columns.extend([f"&-{label}", f"actual_{label}"])
    prediction_path = run_dir / "freqai_predictions_vs_actual.csv"
    merged[[column for column in prediction_columns if column in merged]].to_csv(prediction_path, index=False)

    metrics_path = run_dir / "freqai_prediction_metrics.csv"
    pd.DataFrame(_prediction_metrics(merged)).to_csv(metrics_path, index=False)
    report_paths = [str(prediction_path), str(metrics_path)]
    for horizon in (1, 3, 6, 24, 72, 168, 720):
        bucket_path = run_dir / f"freqai_prediction_bucket_report_{horizon}h.csv"
        _write_prediction_bucket_report(merged, horizon, bucket_path)
        report_paths.append(str(bucket_path))
    archetype_path = run_dir / "freqai_archetype_outcome_report.csv"
    _write_archetype_outcome_report(merged, archetype_path)
    report_paths.append(str(archetype_path))
    known_event_path = run_dir / "freqai_known_event_window_report.csv"
    _write_known_event_report(merged, known_event_path)
    report_paths.append(str(known_event_path))
    return report_paths


def _load_actual_future_returns() -> pd.DataFrame:
    ohlcv_path = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
    prices = pd.read_feather(ohlcv_path, columns=["date", "close"]).sort_values("date")
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    for horizon in RETURN_LABEL_HORIZONS:
        prices[f"actual_future_return_{horizon}h"] = prices["close"].shift(-horizon) / prices["close"] - 1.0
        prices[f"actual_future_up_{horizon}h"] = _binary_label(
            prices[f"actual_future_return_{horizon}h"],
            prices[f"actual_future_return_{horizon}h"] > 0,
        )
    for horizon in PATH_LABEL_HORIZONS:
        future_max = _future_close_extreme(prices["close"], horizon, "max")
        future_min = _future_close_extreme(prices["close"], horizon, "min")
        prices[f"actual_future_max_upside_{horizon}h"] = future_max / prices["close"] - 1.0
        prices[f"actual_future_max_drawdown_{horizon}h"] = future_min / prices["close"] - 1.0
    for horizon, threshold in SHOCK_THRESHOLDS.items():
        drawdown_column = f"actual_future_max_drawdown_{horizon}h"
        if drawdown_column in prices:
            prices[f"actual_shock_down_{horizon}h"] = _binary_label(
                prices[drawdown_column],
                prices[drawdown_column] <= threshold,
            )
    for horizon, threshold in BREAKOUT_THRESHOLDS.items():
        upside_column = f"actual_future_max_upside_{horizon}h"
        if upside_column in prices:
            prices[f"actual_breakout_up_{horizon}h"] = _binary_label(
                prices[upside_column],
                prices[upside_column] >= threshold,
            )
    prices["actual_breakout_up_720h"] = _binary_label(
        prices["actual_future_max_upside_720h"],
        prices["actual_future_max_upside_720h"] >= 0.10,
    )
    prices["actual_regime_up_2160h"] = _binary_label(
        prices["actual_future_return_2160h"],
        prices["actual_future_return_2160h"] >= 0.20,
    )
    return prices.drop(columns=["close"])


def _prediction_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    targets = [
        *[f"future_return_{horizon}h" for horizon in RETURN_LABEL_HORIZONS],
        *[f"future_up_{horizon}h" for horizon in RETURN_LABEL_HORIZONS],
        *[f"future_max_upside_{horizon}h" for horizon in PATH_LABEL_HORIZONS],
        *[f"future_max_drawdown_{horizon}h" for horizon in PATH_LABEL_HORIZONS],
        *EVENT_LABELS,
    ]
    for target in targets:
        prediction_column = f"&-{target}"
        actual_column = f"actual_{target}"
        if prediction_column not in frame or actual_column not in frame:
            continue
        valid = frame[[prediction_column, actual_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if valid.empty:
            continue
        if target.startswith(("future_return", "future_max_upside", "future_max_drawdown")):
            direction_accuracy = ((valid[prediction_column] > 0) == (valid[actual_column] > 0)).mean()
        else:
            direction_accuracy = ((valid[prediction_column] > 0.5) == (valid[actual_column] > 0.5)).mean()
        corr = valid[prediction_column].corr(valid[actual_column]) if valid[prediction_column].nunique() > 1 else float("nan")
        rows.append(
            {
                "target": target,
                "rows": int(len(valid)),
                "prediction_actual_corr": corr,
                "direction_accuracy_from_return_sign": direction_accuracy,
                "predicted_mean": valid[prediction_column].mean(),
                "actual_mean": valid[actual_column].mean(),
            }
        )
    return rows


def _write_prediction_bucket_report(frame: pd.DataFrame, horizon: int, output: Path) -> None:
    prediction_column = f"&-future_return_{horizon}h"
    actual_column = f"actual_future_return_{horizon}h"
    if prediction_column not in frame or actual_column not in frame:
        pd.DataFrame().to_csv(output, index=False)
        return
    valid = frame[[prediction_column, actual_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty or valid[prediction_column].nunique() < 2:
        pd.DataFrame().to_csv(output, index=False)
        return
    valid["prediction_bucket"] = pd.qcut(valid[prediction_column], q=5, duplicates="drop")
    report = (
        valid.groupby("prediction_bucket", observed=True)
        .agg(
            rows=(actual_column, "size"),
            predicted_return_mean=(prediction_column, "mean"),
            actual_return_mean=(actual_column, "mean"),
            actual_up_rate=(actual_column, lambda values: (values > 0).mean()),
        )
        .reset_index()
    )
    report.to_csv(output, index=False)


def _write_archetype_outcome_report(frame: pd.DataFrame, output: Path) -> None:
    archetypes = _read_json(ARCHETYPES_PATH).get("archetypes", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(archetypes, dict):
        pd.DataFrame().to_csv(output, index=False)
        return
    for archetype, config in sorted(archetypes.items()):
        if not isinstance(config, dict):
            continue
        for outcome in config.get("primary_outcomes", []):
            prediction_column = f"&-{outcome}"
            actual_column = f"actual_{outcome}"
            row: dict[str, Any] = {
                "archetype": archetype,
                "outcome": outcome,
                "topic_keys": ",".join(str(item) for item in config.get("topic_keys", [])),
                "examples": " | ".join(str(item) for item in config.get("examples", [])),
                "expected_shape": str(config.get("expected_shape") or ""),
            }
            if prediction_column in frame and actual_column in frame:
                valid = frame[[prediction_column, actual_column]].apply(pd.to_numeric, errors="coerce").dropna()
                row["rows"] = int(len(valid))
                if not valid.empty:
                    row["prediction_actual_corr"] = (
                        valid[prediction_column].corr(valid[actual_column])
                        if valid[prediction_column].nunique() > 1 and valid[actual_column].nunique() > 1
                        else float("nan")
                    )
                    if outcome.startswith(("future_return", "future_max_upside", "future_max_drawdown")):
                        row["direction_accuracy"] = ((valid[prediction_column] > 0) == (valid[actual_column] > 0)).mean()
                    else:
                        row["direction_accuracy"] = ((valid[prediction_column] > 0.5) == (valid[actual_column] > 0.5)).mean()
                    row["predicted_mean"] = valid[prediction_column].mean()
                    row["actual_mean"] = valid[actual_column].mean()
            else:
                row["rows"] = 0
                row["missing_prediction_column"] = prediction_column not in frame
                row["missing_actual_column"] = actual_column not in frame
            rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def _write_known_event_report(frame: pd.DataFrame, output: Path) -> None:
    events = _read_json(KNOWN_EVENTS_PATH).get("events", [])
    rows: list[dict[str, Any]] = []
    if not isinstance(events, list) or frame.empty or "date" not in frame:
        pd.DataFrame().to_csv(output, index=False)
        return
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], utc=True, errors="coerce")
    dated = dated.dropna(subset=["date"]).sort_values("date")
    outcome_keys = [
        "future_return_3h",
        "future_return_6h",
        "future_return_24h",
        "future_return_72h",
        "future_return_168h",
        "future_return_720h",
        "future_return_2160h",
        "future_max_drawdown_3h",
        "future_max_drawdown_6h",
        "future_max_drawdown_24h",
        "future_max_drawdown_72h",
        "future_max_drawdown_168h",
        "future_max_upside_3h",
        "future_max_upside_6h",
        "future_max_upside_24h",
        "future_max_upside_72h",
        "future_max_upside_720h",
        "shock_down_3h",
        "shock_down_6h",
        "shock_down_24h",
        "shock_down_72h",
        "breakout_up_3h",
        "breakout_up_6h",
        "breakout_up_24h",
        "breakout_up_72h",
        "breakout_up_720h",
        "regime_up_2160h",
    ]
    for event in events:
        if not isinstance(event, dict):
            continue
        anchor = pd.to_datetime(event.get("anchor_utc"), utc=True, errors="coerce")
        if pd.isna(anchor):
            continue
        pre_hours = int(event.get("pre_window_hours") or 168)
        post_hours = int(event.get("post_window_hours") or 720)
        window = dated[(dated["date"] >= anchor - pd.Timedelta(hours=pre_hours)) & (dated["date"] <= anchor + pd.Timedelta(hours=post_hours))]
        at_or_before = dated[dated["date"] <= anchor].tail(1)
        anchor_row = at_or_before.iloc[0] if not at_or_before.empty else None
        row: dict[str, Any] = {
            "event_id": event.get("event_id"),
            "name": event.get("name"),
            "archetype": event.get("archetype"),
            "anchor_utc": anchor.isoformat(),
            "pre_window_hours": pre_hours,
            "post_window_hours": post_hours,
            "rows_in_prediction_window": int(len(window)),
            "expected_shape": event.get("expected_shape"),
        }
        for outcome in outcome_keys:
            actual_column = f"actual_{outcome}"
            prediction_column = f"&-{outcome}"
            if anchor_row is not None and actual_column in dated:
                row[f"actual_{outcome}_at_anchor"] = anchor_row.get(actual_column)
            if anchor_row is not None and prediction_column in dated:
                row[f"predicted_{outcome}_at_anchor"] = anchor_row.get(prediction_column)
            if prediction_column in window:
                values = pd.to_numeric(window[prediction_column], errors="coerce").dropna()
                row[f"predicted_{outcome}_window_mean"] = values.mean() if not values.empty else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def _future_close_extreme(close: pd.Series, horizon: int, method: str) -> pd.Series:
    future = close.shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    extreme = rolling.max() if method == "max" else rolling.min()
    return extreme.iloc[::-1]


def _binary_label(source: pd.Series, condition: pd.Series) -> pd.Series:
    return condition.astype(float).where(source.notna())


def _summarise_busy_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": run.get("job_id"),
            "pid": run.get("pid"),
            "phase": run.get("phase"),
            "source": run.get("source"),
            "message": run.get("message"),
        }
        for run in runs
    ]


def _list_outputs(run_dir: Path) -> list[str]:
    return [str(path) for path in sorted(run_dir.glob("*")) if path.is_file()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _finish(payload: dict[str, Any], *, returncode: int = 0) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
