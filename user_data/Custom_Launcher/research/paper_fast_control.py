"""Publish an approved PAPER risk modifier; cannot submit orders or alter strategies."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json

from user_data.strategies.integrated_paper_context import CONTEXT_FILE, _utc, load_luna_context
from user_data.strategies.paper_fast_context import CONTROL_FILE, load_fast_control, publish_fast_control


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "publish"))
    parser.add_argument("--decision-id")
    parser.add_argument("--bias", type=int, choices=range(-2, 3), default=0)
    parser.add_argument("--exposure", choices=("normal", "reduced"), default="normal")
    parser.add_argument("--entry-permission", choices=("normal", "strong_only", "paused"), default="normal")
    parser.add_argument("--reason")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--missing", action="append", default=[])
    parser.add_argument("--blackout", nargs=2, action="append", default=[], metavar=("START_UTC", "END_UTC"))
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.action == "status":
        control = load_fast_control(now)
        print(json.dumps({"status": control.status, "decision_id": control.decision_id,
                          "long_leverage": control.leverage_for("long", now),
                          "short_leverage": control.leverage_for("short", now),
                          "control_file": str(CONTROL_FILE)}))
        return 0
    if not args.decision_id or not args.reason or not args.source:
        parser.error("publish requires decision-id, reason and at least one fresh source")
    luna = load_luna_context(now)
    if luna.status != "observed":
        raise ValueError("A fresh Luna observation is required")
    expires = min(now + timedelta(hours=4), _utc(json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))["valid_until_utc"]))
    row = {"schema_version": 1, "author": "main_agent", "decision_id": args.decision_id,
           "observed_at_utc": now.isoformat(), "valid_until_utc": expires.isoformat(),
           "bias": args.bias, "exposure": args.exposure, "entry_permission": args.entry_permission,
           "reason": args.reason, "sources": args.source, "unavailable_inputs": args.missing,
           "blackouts": [{"start_utc": start, "end_utc": end} for start, end in args.blackout]}
    publish_fast_control(row, now=now)
    print(json.dumps({"published": True, "decision_id": args.decision_id, "valid_until_utc": expires.isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
