from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from launcher_v2.services.data_tools_watchdog_service import DataToolsWatchdogService, SERVICE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep LauncherV2 data collectors running.")
    parser.add_argument("--once", action="store_true", help="Run one health check and exit.")
    parser.add_argument("--loop", action="store_true", help="Run continuously in this process.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Loop sleep interval when --loop is used.")
    parser.add_argument("--preset", default="", help="Path to LauncherV2 presets.json.")
    parser.add_argument("--heartbeat-stale-minutes", default="10", help="Heartbeat age that counts as stale.")
    parser.add_argument("--services", default=",".join(SERVICE_KEYS), help="Comma-separated service keys to monitor.")
    parser.add_argument("--no-restart-dead", action="store_true", help="Warn only; do not restart dead processes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = DataToolsWatchdogService(APP_DIR)
    state = {
        "heartbeat_stale_minutes": args.heartbeat_stale_minutes,
        "services": [part.strip() for part in str(args.services or "").split(",") if part.strip()],
        "restart_dead": not args.no_restart_dead,
    }
    preset = args.preset or None
    if args.loop and not args.once:
        while True:
            payload = service.run_once(state, preset_path=preset)
            print(json.dumps(_summary(payload), sort_keys=False), flush=True)
            time.sleep(max(5, int(args.interval_seconds)))
    payload = service.run_once(state, preset_path=preset)
    print(json.dumps(_summary(payload), sort_keys=False))
    return 0


def _summary(payload: dict) -> dict:
    return {
        "checked_at": payload.get("checked_at"),
        "rows": [
            {
                "label": row.get("label"),
                "running": row.get("running"),
                "pid": row.get("pid"),
                "action": row.get("action"),
                "last_error": row.get("last_error"),
            }
            for row in payload.get("rows", [])
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
