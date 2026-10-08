"""Publish an approved PAPER risk modifier; cannot submit orders or alter strategies."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json

from user_data.strategies.integrated_paper_context import CONTEXT_FILE, _utc, load_luna_context
from user_data.strategies.paper_fast_context import CONTROL_FILE, load_fast_control, publish_fast_control
from user_data.strategies.paper_aggressive_context import (
    AGGRESSIVE_ACCOUNT_KEYS, AGGRESSIVE_ACCOUNTS, CONTROL_FILES,
    load_aggressive_control, publish_aggressive_control,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "publish"))
    parser.add_argument("--account", choices=AGGRESSIVE_ACCOUNT_KEYS,
                        help="Use one exact aggressive PAPER account instead of legacy fast_context")
    parser.add_argument("--decision-id")
    parser.add_argument("--bias", type=int, choices=range(-2, 3))
    parser.add_argument("--exposure", choices=("normal", "reduced"), default="normal")
    parser.add_argument("--entry-permission", choices=("normal", "strong_only", "paused"), default="normal")
    parser.add_argument("--side-permission", choices=("both", "long_only", "short_only"))
    parser.add_argument("--long-leverage-cap", type=float)
    parser.add_argument("--short-leverage-cap", type=float)
    parser.add_argument("--reason")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--missing", action="append", default=[])
    parser.add_argument("--blackout", nargs=2, action="append", default=[], metavar=("START_UTC", "END_UTC"))
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.action == "status":
        if args.account:
            control = load_aggressive_control(args.account, now)
            print(json.dumps({"account": AGGRESSIVE_ACCOUNTS[args.account]["bot_name"],
                "status": control.status, "technical_only": control.technical_only,
                "decision_id": control.decision_id, "bias": control.bias,
                "side_permission": control.side_permission,
                "long_leverage_cap": control.long_leverage_cap,
                "short_leverage_cap": control.short_leverage_cap,
                "exposure": control.exposure, "control_file": str(CONTROL_FILES[args.account])}))
            return 0
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
    if args.account:
        if (args.bias is None or args.side_permission is None
                or args.long_leverage_cap is None or args.short_leverage_cap is None):
            parser.error("aggressive publish requires bias, side-permission and both leverage caps")
        row = {"schema_version": 1, "author": "main_agent",
               "decision_id": args.decision_id or "", "account": AGGRESSIVE_ACCOUNTS[args.account]["bot_name"],
               "observed_at_utc": now.isoformat(), "valid_until_utc": expires.isoformat(),
               "bias": args.bias, "side_permission": args.side_permission,
               "long_leverage_cap": args.long_leverage_cap,
               "short_leverage_cap": args.short_leverage_cap,
               "exposure": args.exposure, "reason": args.reason,
               "sources": args.source, "unavailable_inputs": args.missing}
        publish_aggressive_control(row, args.account, now=now)
        print(json.dumps({"published": True, "account": row["account"],
                          "decision_id": row["decision_id"],
                          "valid_until_utc": expires.isoformat()}))
        return 0
    row = {"schema_version": 1, "author": "main_agent", "decision_id": args.decision_id,
           "observed_at_utc": now.isoformat(), "valid_until_utc": expires.isoformat(),
           "bias": 0 if args.bias is None else args.bias, "exposure": args.exposure, "entry_permission": args.entry_permission,
           "reason": args.reason, "sources": args.source, "unavailable_inputs": args.missing,
           "blackouts": [{"start_utc": start, "end_utc": end} for start, end in args.blackout]}
    publish_fast_control(row, now=now)
    print(json.dumps({"published": True, "decision_id": args.decision_id, "valid_until_utc": expires.isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
