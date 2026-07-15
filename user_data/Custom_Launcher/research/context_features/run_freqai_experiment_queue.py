from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = USER_DATA_DIR / "models"
DEFAULT_STRUCTURAL = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache" / "btc_structural_features_1h_latest.parquet"
DEFAULT_ORDERBOOK = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the next pending queued FreqAI experiment.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--orderbook", type=Path, default=DEFAULT_ORDERBOOK)
    parser.add_argument("--allow-concurrent", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock_fd = _acquire_queue_lock(args.queue.with_suffix(args.queue.suffix + ".lock"))
    if lock_fd is None:
        print(json.dumps({"status": "skipped", "reason": "queue_locked"}, indent=2))
        return 0
    try:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        _unstick_stale_running(queue)
        experiment = _next_pending(queue)
        if experiment is None:
            queue["status"] = _queue_status(queue)
            _write_queue(args.queue, queue)
            print(json.dumps({"status": "idle", "reason": "no_pending_experiments"}, indent=2))
            return 0
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "experiment_id": experiment["id"], "command": experiment["command"]}, indent=2))
            return 0
        if not args.allow_concurrent and _freqai_running():
            print(json.dumps({"status": "skipped", "reason": "freqtrade_or_freqai_process_running"}, indent=2))
            return 0

        experiment["status"] = "running"
        experiment["started_at"] = datetime.now().astimezone().isoformat()
        queue["status"] = "running"
        _write_queue(args.queue, queue)

        run_dir = Path(str(experiment["run_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "freqai_stdout.log"
        stderr_path = run_dir / "freqai_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [str(part) for part in experiment["command"]],
                cwd=str(REPO_ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        experiment["returncode"] = int(result.returncode)
        experiment["finished_at"] = datetime.now().astimezone().isoformat()
        experiment["stdout_log"] = str(stdout_path)
        experiment["stderr_log"] = str(stderr_path)
        if result.returncode != 0:
            experiment["status"] = "failed"
            queue["status"] = _queue_status(queue)
            _write_queue(args.queue, queue)
            print(json.dumps({"status": "failed", "experiment_id": experiment["id"], "returncode": result.returncode}, indent=2))
            return result.returncode

        try:
            from user_data.Custom_Launcher.research.context_features.score_freqai_experiment import score_experiment

            score_experiment(
                experiment,
                ledger_path=Path(str(queue["ledger_path"])),
                models_dir=args.models_dir,
                structural=args.structural,
                orderbook=args.orderbook,
            )
            experiment["status"] = "completed"
        except Exception as exc:
            experiment["status"] = "score_failed"
            experiment["score_error"] = str(exc)
        queue["status"] = _queue_status(queue)
        _write_queue(args.queue, queue)
        print(json.dumps({"status": experiment["status"], "experiment_id": experiment["id"]}, indent=2))
        return 0 if experiment["status"] == "completed" else 2
    finally:
        _release_queue_lock(args.queue.with_suffix(args.queue.suffix + ".lock"), lock_fd)


def _acquire_queue_lock(lock_path: Path) -> int | None:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"{os.getpid()} {datetime.now().astimezone().isoformat()}\n".encode("utf-8"))
    return fd


def _release_queue_lock(lock_path: Path, fd: int) -> None:
    os.close(fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _next_pending(queue: dict[str, Any]) -> dict[str, Any] | None:
    for experiment in queue.get("experiments", []):
        if experiment.get("status") == "pending":
            return experiment
    return None


def _unstick_stale_running(queue: dict[str, Any]) -> None:
    if _freqai_running():
        return
    for experiment in queue.get("experiments", []):
        if experiment.get("status") == "running" and not experiment.get("returncode"):
            experiment["status"] = "pending"
            experiment["stale_running_reset_at"] = datetime.now().astimezone().isoformat()


def _write_queue(path: Path, queue: dict[str, Any]) -> None:
    path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def _queue_status(queue: dict[str, Any]) -> str:
    statuses = {str(item.get("status") or "unknown") for item in queue.get("experiments", [])}
    if not statuses:
        return "empty"
    if "running" in statuses:
        return "running"
    if "pending" in statuses:
        return "pending"
    if statuses <= {"completed"}:
        return "completed"
    if statuses & {"failed", "score_failed"}:
        return "failed"
    return "finished"


def _freqai_running() -> bool:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "$self=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -match '--freqaimodel|LightGBMRegressorMultiTarget|SKLearnRidgeRegressorMultiTarget' } | Select-Object -First 1 | ForEach-Object { '1' }",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() == "1"


if __name__ == "__main__":
    raise SystemExit(main())
