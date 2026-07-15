from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = Path(__file__).resolve().parent / "run_freqai_experiment_queue.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queued FreqAI experiments sequentially until no pending work remains.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sleep-seconds", type=int, default=60)
    parser.add_argument("--max-failures", type=int, default=3)
    args = parser.parse_args()

    failures = 0
    while True:
        counts = _status_counts(args.queue)
        if counts.get("pending", 0) <= 0 and counts.get("running", 0) <= 0:
            print(json.dumps({"status": "idle", "counts": dict(counts)}, indent=2), flush=True)
            return 0
        result = subprocess.run(
            [str(args.python_exe), str(RUNNER), "--queue", str(args.queue)],
            cwd=str(REPO_ROOT),
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures += 1
            if failures >= int(args.max_failures):
                print(json.dumps({"status": "stopped", "reason": "max_failures", "failures": failures}, indent=2), flush=True)
                return result.returncode
        else:
            failures = 0
        time.sleep(max(5, int(args.sleep_seconds)))


def _status_counts(queue_path: Path) -> Counter:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    return Counter(str(item.get("status") or "unknown") for item in queue.get("experiments", []))


if __name__ == "__main__":
    raise SystemExit(main())
