"""Audited local control for the integrated paper manual-override account.

This intentionally refuses any non-dry-run/non-local configuration.  Trade
overrides require a URL in a fresh Luna observation; other interventions
require a recorded reason and source.  No news model calls it directly.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from user_data.strategies.integrated_paper_context import load_luna_context
from user_data.strategies.paper_news_manual import MANUAL_ACCOUNTS, TAG_PREFIX, validate_manual_plan
from datetime import timedelta


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "user_data/configs/integrated_paper_20260926_base.json"
OVERLAY_CONFIG = ROOT / "user_data/configs/integrated_paper_20260926_manual.json"
JOURNAL = ROOT / "user_data/research_news_data/context_features/integrated_paper_20260926/manual_interventions.jsonl"
RUN_RECORD = ROOT / "user_data/research_news_data/context_features/integrated_paper_20260926/run_record.json"
SOURCE_IDS = {"breakout_long", "pullback_long", "resistance_short"}
ALLOWED_PAIRS = {
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT",
}


def _configs(account: str = "manual") -> tuple[dict, dict]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config_path = OVERLAY_CONFIG if account == "manual" else ROOT / f"user_data/configs/paper_{account}.json"
    overlay = json.loads(config_path.read_text(encoding="utf-8"))
    api = overlay["api_server"]
    if (base.get("dry_run") is not True or base.get("trading_mode") != "futures"
            or base.get("exchange", {}).get("name") != "binance"
            or api.get("listen_ip_address") != "127.0.0.1"
            or any(base.get("exchange", {}).get(key) or overlay.get("exchange", {}).get(key)
                   for key in ("key", "secret"))
            or overlay.get("bot_name") != ("integrated_paper_manual" if account == "manual" else f"paper_{account}")):
        raise RuntimeError("Refusing control: configuration is not the expected local paper account")
    if set(base["exchange"]["pair_whitelist"]) != ALLOWED_PAIRS:
        raise RuntimeError("Refusing control: paper pair list drifted")
    expected_port = 8096 if account == "manual" else MANUAL_ACCOUNTS[f"paper_{account}"][0]
    expected_strategy = "IntegratedPaper" if account == "manual" else "PaperNewsManual"
    expected_db = "manual_trades.sqlite" if account == "manual" else f"{account}_trades.sqlite"
    if (api.get("enabled") is not True or api.get("listen_port") != expected_port
            or overlay.get("strategy") != expected_strategy
            or overlay.get("db_url") != f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{expected_db}"
            or overlay.get("force_entry_enable") is not True):
        raise RuntimeError("Refusing control: account/API/database identity drifted")
    return base, overlay


def _api(overlay: dict, route: str, payload: dict | None = None):
    api = overlay["api_server"]
    credentials = f'{api["username"]}:{api["password"]}'.encode("utf-8")
    auth = base64.b64encode(credentials).decode("ascii")
    url = f'http://127.0.0.1:{api["listen_port"]}/api/v1/{route}'
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url, data=body,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=12) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Paper API {route} failed or outcome is uncertain: {error}") from error


def _verify_running_account(overlay: dict) -> dict:
    observed = _api(overlay, "show_config")
    expected = {
        "dry_run": True,
        "strategy": overlay["strategy"],
        "bot_name": overlay["bot_name"],
        "trading_mode": "futures",
        "exchange": "binance",
    }
    if any(observed.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Refusing control: API is not the expected running paper-only account")
    return observed


def _assert_manual_lifecycle_allows(account: str, action: str) -> None:
    """Do not reopen the original manual entry lane once it is draining or parked."""
    if account != "manual" or action not in {"enter", "resume"}:
        return
    record = json.loads(RUN_RECORD.read_text(encoding="utf-8"))
    policy = record["process_recovery"]
    if "manual" not in policy.get("allowed_accounts", []):
        raise RuntimeError("Manual account is absent from the reviewed paper-account lifecycle registry")
    lifecycle = policy.get("account_lifecycle")
    if lifecycle is not None:
        if "manual" not in lifecycle:
            raise RuntimeError("Manual account lifecycle is missing; refusing entry-state override")
        state = lifecycle["manual"]
        if state != "ACTIVE":
            raise RuntimeError(f"Manual {action} is disabled while this account is {state}; existing positions remain manageable")


def _journal(account: str) -> Path:
    return JOURNAL if account == "manual" else MANUAL_ACCOUNTS[f"paper_{account}"][1]


def _record(row: dict, account: str = "manual") -> None:
    journal = _journal(account)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _seen_decision(decision_id: str, account: str = "manual") -> bool:
    journal = _journal(account)
    if not journal.exists():
        return False
    with journal.open(encoding="utf-8") as handle:
        return any(json.loads(line).get("decision_id") == decision_id for line in handle if line.strip())


def _trade(trades: list[dict], trade_id: int) -> dict:
    for trade in trades:
        if int(trade["trade_id"]) == trade_id:
            return trade
    raise ValueError(f"Open paper trade {trade_id} does not exist")


def _equity(balance: dict) -> float:
    value = float(balance["total_bot"])
    if not 0 < value < 1_000_000:
        raise RuntimeError(f"Invalid paper equity from API: {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "enter", "reduce", "exit", "protect", "pause", "resume", "no-action"))
    parser.add_argument("--account", choices=("manual", "news_manual", "news_lab"), default="manual")
    parser.add_argument("--decision-id", default=None)
    parser.add_argument("--reason")
    parser.add_argument("--source")
    parser.add_argument("--pair", choices=sorted(ALLOWED_PAIRS))
    parser.add_argument("--side", choices=("long", "short"))
    parser.add_argument("--source-id", choices=sorted(SOURCE_IDS))
    parser.add_argument("--trade-id", type=int)
    parser.add_argument("--stake-pct", type=float, default=0.02)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--reference-rate", type=float)
    parser.add_argument("--stop-price", type=float)
    parser.add_argument("--take-profit-price", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _assert_manual_lifecycle_allows(args.account,args.action)
    _, overlay = _configs(args.account)
    _verify_running_account(overlay)
    if args.action == "status":
        print(json.dumps({"balance": _api(overlay, "balance"), "trades": _api(overlay, "status")}, indent=2))
        return 0
    if not args.reason or not args.source:
        raise ValueError("Every paper decision needs --reason and --source")
    if args.action in {"enter", "reduce", "exit", "protect"}:
        luna = load_luna_context(datetime.now(timezone.utc))
        if luna.status != "observed" or args.source not in luna.sources:
            raise ValueError("Manual trade override requires a fresh Luna snapshot and one of its source URLs")
    decision_id = args.decision_id or uuid4().hex
    if _seen_decision(decision_id, args.account):
        raise ValueError(f"Decision {decision_id} already recorded; inspect its outcome before retrying")
    row = {
        "decision_id": decision_id, "at_utc": datetime.now(timezone.utc).isoformat(),
        "account": overlay["bot_name"], "action": args.action,
        "reason": args.reason, "source": args.source, "status": "proposed",
    }
    if args.action == "no-action":
        row["status"] = "recorded_no_action"
        _record(row, args.account)
        print(json.dumps(row))
        return 0

    trades = _api(overlay, "status")
    equity = _equity(_api(overlay, "balance"))
    payload: dict = {}
    route = ""
    if args.action == "enter":
        if args.account == "manual":
            if not args.pair or not args.side or not args.source_id or not 0 < args.stake_pct <= 0.02:
                raise ValueError("Enter requires pair, side, source-id and 0 < stake-pct <= 0.02")
            if (args.source_id == "resistance_short") != (args.side == "short"):
                raise ValueError("Selected Sieve exit contract does not match manual trade direction")
            if args.leverage != 1.0:
                raise ValueError("The original integrated paper trial uses 1x leverage only")
            entry_tag = f"manual:{args.source_id}"
        else:
            limits = overlay["paper_manual_limits"]
            if (not args.pair or not args.side
                    or not 0 < args.stake_pct <= float(limits["max_stake_pct"])
                    or not 1 <= args.leverage <= float(limits["max_leverage"])):
                raise ValueError("Manual news entry exceeds its explicit paper envelope")
            now = datetime.now(timezone.utc)
            plan = {"pair": args.pair, "side": args.side, "reference_rate": args.reference_rate,
                    "stake_pct": args.stake_pct, "leverage": args.leverage,
                    "stop_price": args.stop_price, "take_profit_price": args.take_profit_price,
                    "valid_until_utc": (now + timedelta(minutes=5)).isoformat(),
                    "review_due_at_utc": (now + timedelta(hours=4)).isoformat()}
            validate_manual_plan(plan, float(args.reference_rate or 0.))
            row["plan"] = plan
            entry_tag = TAG_PREFIX + decision_id
        stake = round(equity * args.stake_pct, 8)
        existing = [trade for trade in trades if trade["pair"] == args.pair]
        if existing or len(trades) >= 3:
            raise ValueError("Manual entry conflicts with an open trade or three-position limit")
        payload = {"pair": args.pair, "side": args.side, "stakeamount": stake,
                   "leverage": args.leverage, "entry_tag": entry_tag}
        route = "forceenter"
    elif args.action == "protect":
        if args.account == "manual" or args.trade_id is None:
            raise ValueError("Protect requires a news-only account and --trade-id")
        trade = _trade(trades, args.trade_id)
        now = datetime.now(timezone.utc)
        plan = {"pair": trade["pair"], "side": "short" if trade["is_short"] else "long",
                "reference_rate": args.reference_rate, "stake_pct": float(trade["stake_amount"]) / equity,
                "leverage": trade["leverage"], "stop_price": args.stop_price,
                "take_profit_price": args.take_profit_price,
                "valid_until_utc": (now + timedelta(minutes=5)).isoformat(),
                "review_due_at_utc": (now + timedelta(hours=4)).isoformat()}
        validate_manual_plan(plan, float(args.reference_rate or 0.))
        existing_stop = float(trade["stop_loss_abs"])
        if ((trade["is_short"] and args.stop_price > existing_stop)
                or (not trade["is_short"] and args.stop_price < existing_stop)):
            raise ValueError("Protection updates cannot silently loosen the existing stop")
        row.update({"trade_id": args.trade_id, "plan": plan, "status": "recorded_protection"})
        _record(row, args.account)
        print(json.dumps(row))
        return 0
    elif args.action in {"reduce", "exit"}:
        if args.trade_id is None:
            raise ValueError("Reduce/exit requires --trade-id")
        trade = _trade(trades, args.trade_id)
        payload = {"tradeid": str(args.trade_id)}
        if args.action == "reduce":
            if args.fraction is None or not 0 < args.fraction < 1:
                raise ValueError("Reduce requires 0 < --fraction < 1")
            payload["amount"] = float(trade["amount"]) * args.fraction
        route = "forceexit"
    elif args.action in {"pause", "resume"}:
        route = "stopentry" if args.action == "pause" else "start"
    else:
        raise ValueError(args.action)

    row.update({"pair": args.pair, "trade_id": args.trade_id, "request": payload,
                "equity_before": equity, "open_positions_before": len(trades)})
    _record(row, args.account)
    try:
        result = _api(overlay, route, payload)
    except Exception as error:
        _record({**row, "status": "uncertain_manual_check_required", "error": str(error)}, args.account)
        raise
    _record({**row, "status": "submitted", "response": result}, args.account)
    print(json.dumps({"decision_id": decision_id, "response": result}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
