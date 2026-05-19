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
DEFAULT_FREQAI_MODEL = "LightGBMRegressorMultiTarget"
RUN_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_runs"
STATE_PATH = RUN_ROOT / "run_state.json"


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
    command = [
        str(args.python_exe),
        "-m",
        "freqtrade",
        "backtesting",
        "--userdir",
        str(USER_DATA_DIR),
        "--config",
        str(args.config),
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
        "outputs": _list_outputs(run_dir),
    }
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
