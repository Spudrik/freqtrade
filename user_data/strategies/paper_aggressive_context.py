"""Pure account, expiring-control, and protection contracts for four PAPER experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

from user_data.strategies.integrated_paper_context import CONTEXT_FILE, _utc, load_luna_context


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "user_data/research_news_data/context_features/integrated_paper_20260926"
AGGRESSIVE_PAIRS = frozenset({
    "XRP/USDT:USDT", "ZEC/USDT:USDT", "DOGE/USDT:USDT", "NEAR/USDT:USDT",
    "UNI/USDT:USDT", "SUI/USDT:USDT", "AVAX/USDT:USDT", "ENA/USDT:USDT",
})
LEADER_PAIRS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
AGGRESSIVE_ACCOUNTS = {
    "aggressive_vacuum": {
        "bot_name": "paper_aggressive_vacuum", "strategy": "PaperAggressiveVacuum", "port": 8100,
        "family": "vacuum", "config": "user_data/configs/paper_aggressive_vacuum.json",
    },
    "aggressive_reclaim": {
        "bot_name": "paper_aggressive_reclaim", "strategy": "PaperAggressiveReclaim", "port": 8101,
        "family": "reclaim", "config": "user_data/configs/paper_aggressive_reclaim.json",
    },
    "aggressive_auction": {
        "bot_name": "paper_aggressive_auction", "strategy": "PaperAggressiveAuction", "port": 8102,
        "family": "auction", "config": "user_data/configs/paper_aggressive_auction.json",
    },
    "aggressive_rotation": {
        "bot_name": "paper_aggressive_rotation", "strategy": "PaperAggressiveRotation", "port": 8103,
        "family": "rotation", "config": "user_data/configs/paper_aggressive_rotation.json",
    },
}
AGGRESSIVE_ACCOUNT_KEYS = tuple(AGGRESSIVE_ACCOUNTS)
AGGRESSIVE_STRATEGIES = {row["strategy"]: key for key, row in AGGRESSIVE_ACCOUNTS.items()}
AGGRESSIVE_BOT_NAMES = {row["bot_name"]: key for key, row in AGGRESSIVE_ACCOUNTS.items()}
AGGRESSIVE_LIMITS = {
    "max_margin_pct": 0.15,
    "max_position_risk_pct": 0.03,
    "max_combined_risk_pct": 0.08,
    "emergency_margin_loss_pct": 0.25,
    "fee_allowance": 0.0014,
    "default_leverage": 7.0,
}
PLAN_KEY = "paper_aggressive_plan"
MANUAL_TAG_PREFIX = "aggressive_manual:"
CONTROL_MAX_AGE = timedelta(hours=4)
CONTROL_FILES = {key: REPORT / f"{key}_control.json" for key in AGGRESSIVE_ACCOUNT_KEYS}
CONTROL_JOURNALS = {key: REPORT / f"{key}_control_decisions.jsonl" for key in AGGRESSIVE_ACCOUNT_KEYS}
DECISION_JOURNALS = {key: REPORT / f"{key}_decisions.jsonl" for key in AGGRESSIVE_ACCOUNT_KEYS}


def account_key(value: str) -> str:
    if value in AGGRESSIVE_ACCOUNTS:
        return value
    if value in AGGRESSIVE_BOT_NAMES:
        return AGGRESSIVE_BOT_NAMES[value]
    raise ValueError(f"Unknown aggressive PAPER account: {value}")


def _finite(value, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid {label}")
    return result


@dataclass(frozen=True)
class AggressiveControl:
    status: str
    decision_id: str = "unavailable"
    bias: int = 0
    side_permission: str = "both"
    long_leverage_cap: float = 3.0
    short_leverage_cap: float = 3.0
    exposure: str = "normal"
    reason: str = "No current main-agent control; technical-only 3x mode"

    @property
    def technical_only(self) -> bool:
        return self.status != "observed"

    @property
    def size_factor(self) -> float:
        return 0.5 if self.status == "observed" and self.exposure == "reduced" else 1.0

    def permits(self, side: str) -> bool:
        return (side in {"long", "short"}
                and (self.side_permission == "both" or self.side_permission == f"{side}_only"))

    def leverage_cap(self, side: str) -> float:
        if side not in {"long", "short"}:
            raise ValueError("Side must be long or short")
        if self.technical_only:
            return 3.0
        return self.long_leverage_cap if side == "long" else self.short_leverage_cap

    def desired_leverage(self, side: str, mode: str, score: float) -> float:
        if not self.permits(side):
            return 0.0
        if self.technical_only or mode != "primary":
            return 3.0
        aligned = self.bias > 0 if side == "long" else self.bias < 0
        if score >= 0.80 and aligned:
            return min(10.0, self.leverage_cap(side))
        if score >= 0.55 and (self.bias == 0 or aligned):
            return min(7.0, self.leverage_cap(side))
        return min(3.0, self.leverage_cap(side))


def parse_aggressive_control(row: dict, now: datetime, account: str | None = None) -> AggressiveControl:
    if now.tzinfo is None or now.utcoffset() is None or not isinstance(row, dict):
        raise ValueError("Aggressive control requires an object and timezone-aware clock")
    if row.get("schema_version") != 1 or row.get("author") != "main_agent":
        raise ValueError("Only schema-1 main-agent aggressive controls are accepted")
    bound_account = row.get("account")
    if bound_account not in AGGRESSIVE_BOT_NAMES:
        raise ValueError("Aggressive control must identify one exact PAPER account")
    if account is not None and AGGRESSIVE_ACCOUNTS[account_key(account)]["bot_name"] != bound_account:
        raise ValueError("Aggressive control belongs to a different PAPER account")
    if not isinstance(row.get("decision_id"), str) or not row["decision_id"].strip():
        raise ValueError("Aggressive control requires a decision ID")
    observed, expires = _utc(row["observed_at_utc"]), _utc(row["valid_until_utc"])
    if observed > now + timedelta(minutes=2) or not timedelta(0) < expires - observed <= CONTROL_MAX_AGE:
        raise ValueError("Invalid aggressive control observation/expiry window")
    bias = row.get("bias")
    if type(bias) is not int or bias not in range(-2, 3):
        raise ValueError("Aggressive bias must be an integer from -2 through +2")
    permission = row.get("side_permission")
    if permission not in {"both", "long_only", "short_only"}:
        raise ValueError("Invalid aggressive side permission")
    long_cap = _finite(row.get("long_leverage_cap"), "long leverage cap", positive=True)
    short_cap = _finite(row.get("short_leverage_cap"), "short leverage cap", positive=True)
    if not 3.0 <= long_cap <= 10.0 or not 3.0 <= short_cap <= 10.0:
        raise ValueError("Aggressive leverage caps must be from 3x through 10x")
    exposure = row.get("exposure")
    if exposure not in {"normal", "reduced"}:
        raise ValueError("Aggressive exposure must be normal or reduced")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("Aggressive control requires a reason")
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources or any(not isinstance(item, str) or not item.strip() for item in sources):
        raise ValueError("Aggressive control requires fresh Luna source references")
    unavailable = row.get("unavailable_inputs", [])
    if not isinstance(unavailable, list) or any(not isinstance(item, str) for item in unavailable):
        raise ValueError("Aggressive unavailable-inputs must be a list of strings")
    if now >= expires:
        return AggressiveControl("stale", row["decision_id"])
    return AggressiveControl("observed", row["decision_id"], bias, permission,
                             long_cap, short_cap, exposure, row["reason"].strip())


def load_aggressive_control(account: str, now: datetime, path: Path | None = None) -> AggressiveControl:
    key = account_key(account)
    selected = path or CONTROL_FILES[key]
    if not selected.is_file():
        return AggressiveControl("missing")
    try:
        return parse_aggressive_control(json.loads(selected.read_text(encoding="utf-8")), now, key)
    except (OSError, ValueError, KeyError, TypeError):
        return AggressiveControl("malformed")


def publish_aggressive_control(row: dict, account: str, *, now: datetime | None = None,
                               path: Path | None = None, journal: Path | None = None) -> None:
    key = account_key(account)
    clock = now or datetime.now(timezone.utc)
    if row.get("account") != AGGRESSIVE_ACCOUNTS[key]["bot_name"]:
        raise ValueError("Aggressive control must be bound to its exact PAPER account")
    if parse_aggressive_control(row, clock, key).status != "observed":
        raise ValueError("Cannot publish an expired aggressive control")
    luna = load_luna_context(clock)
    if luna.status != "observed" or not set(row["sources"]) <= set(luna.sources):
        raise ValueError("Aggressive control sources must belong to the fresh Luna observation")
    brief = json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
    if _utc(row["valid_until_utc"]) > _utc(brief["valid_until_utc"]):
        raise ValueError("Aggressive control cannot outlive its supporting Luna brief")
    journal_path = journal or CONTROL_JOURNALS[key]
    control_path = path or CONTROL_FILES[key]
    if journal_path.is_file():
        with journal_path.open(encoding="utf-8") as stream:
            if any(json.loads(line).get("decision_id") == row["decision_id"] for line in stream if line.strip()):
                raise ValueError("Decision already journalled; inspect current control rather than duplicate it")
    from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import write_text_atomic
    encoded = json.dumps(row, allow_nan=False, sort_keys=True)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    write_text_atomic(control_path, json.dumps(row, indent=2, allow_nan=False) + "\n")


def validate_aggressive_config(base: dict, overlay: dict, account: str) -> None:
    key = account_key(account)
    spec = AGGRESSIVE_ACCOUNTS[key]
    api = overlay.get("api_server", {})
    expected_db = f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{key}_trades.sqlite"
    exchange = overlay.get("exchange", {})
    expected_credentials = ("username", "password", "jwt_secret_key", "ws_token")
    if (base.get("dry_run") is not True or base.get("trading_mode") != "futures"
            or base.get("margin_mode") != "isolated" or base.get("exchange", {}).get("name") != "binance"
            or any(base.get("exchange", {}).get(name) for name in ("key", "secret", "password", "privateKey", "private_key"))
            or overlay.get("bot_name") != spec["bot_name"] or overlay.get("strategy") != spec["strategy"]
            or overlay.get("timeframe") != "5m" or overlay.get("max_open_trades") != 3
            or overlay.get("initial_state", "running") != "running" or overlay.get("db_url") != expected_db
            or overlay.get("force_entry_enable") is not True
            or set(exchange.get("pair_whitelist", ())) != AGGRESSIVE_PAIRS
            or set(exchange) - {"pair_whitelist"}
            or any(exchange.get(name) for name in ("key", "secret", "password", "privateKey", "private_key"))
            or api.get("enabled") is not True or api.get("listen_ip_address") != "127.0.0.1"
            or api.get("listen_port") != spec["port"]
            or any(not isinstance(api.get(name), str) or not api[name].strip()
                   or api[name].startswith("CHANGE_ME") for name in expected_credentials)
            or overlay.get("paper_aggressive_limits") != AGGRESSIVE_LIMITS):
        raise RuntimeError(f"{key}: configuration is not the exact isolated aggressive PAPER account")
    if any(overlay.get(name) for name in ("exchange_key", "exchange_secret", "private_key", "privateKey")):
        raise RuntimeError(f"{key}: exchange credentials are forbidden")


def validate_aggressive_manual_plan(plan: dict, rate: float, account: str, *,
                                    fresh: bool = True, equity: float | None = None,
                                    actual_leverage: float | None = None,
                                    control: AggressiveControl | None = None,
                                    now: datetime | None = None) -> None:
    account_key(account)
    if not isinstance(plan, dict) or plan.get("side") not in {"long", "short"}:
        raise ValueError("Aggressive entry plan needs an explicit side")
    if plan.get("pair") not in AGGRESSIVE_PAIRS:
        raise ValueError("Aggressive entry plan uses a pair outside the fixed universe")
    rate = _finite(rate, "current decision price", positive=True)
    values = {name: _finite(plan.get(name), name, positive=True)
              for name in ("reference_rate", "stake_pct", "leverage", "stop_price")}
    limits = AGGRESSIVE_LIMITS
    if values["stake_pct"] > limits["max_margin_pct"] or not 3.0 <= values["leverage"] <= 10.0:
        raise ValueError("Aggressive entry exceeds its margin or leverage envelope")
    if actual_leverage is not None:
        actual = _finite(actual_leverage, "actual leverage", positive=True)
        if not 3.0 <= actual <= values["leverage"] + 1e-9:
            raise ValueError("Actual leverage is outside the approved 3x-to-requested range")
    short = plan["side"] == "short"
    stop = values["stop_price"]
    if (short and stop <= rate) or (not short and stop >= rate):
        raise ValueError("Aggressive stop must be beyond current price on the loss side")
    reference = values["reference_rate"]
    target_value = plan.get("take_profit_price")
    target = _finite(target_value, "take-profit price", positive=True)
    if (short and target >= rate) or (not short and target <= rate):
        raise ValueError("Aggressive target must be on the profitable side")
    if (short and not (target < reference < stop)) or (not short and not (stop < reference < target)):
        raise ValueError("Aggressive stop/target must also bracket the approved reference price")
    if abs(target - rate) < 2.5 * limits["fee_allowance"] * rate:
        raise ValueError("Aggressive take-profit must provide meaningful travel beyond the roundtrip cost allowance")
    loss_fraction = values["leverage"] * (abs(rate - stop) / rate + limits["fee_allowance"])
    if loss_fraction > limits["emergency_margin_loss_pct"] + 1e-9:
        raise ValueError("Aggressive stop exceeds the 25% initial-margin emergency floor")
    if equity is not None:
        equity = _finite(equity, "paper equity", positive=True)
        stake = equity * values["stake_pct"]
        risk = stake * loss_fraction
        if risk > equity * limits["max_position_risk_pct"] + 1e-6:
            raise ValueError("Aggressive entry exceeds the 3% per-trade planned-risk budget")
    for clock_name in ("valid_until_utc", "review_due_at_utc"):
        _utc(plan[clock_name])
    if fresh:
        if control is not None:
            # An explicit main-agent force-entry is a per-decision direction
            # override; it does not inherit the automatic side permission.
            if control.technical_only and values["leverage"] > 3.0:
                raise ValueError("Missing/stale controls require explicit technical-only 3x mode")
            if (not control.technical_only
                    and values["leverage"] > control.leverage_cap(plan["side"])):
                raise ValueError("Aggressive entry exceeds the current side-specific leverage cap")
        if (now or datetime.now(timezone.utc)) >= _utc(plan["valid_until_utc"]):
            raise ValueError("Aggressive entry approval has expired")


def validate_aggressive_filled_plan(plan: dict, *, actual_pair: str | None = None,
                                    actual_side: str | None = None, actual_open_rate: float | None = None,
                                    actual_quantity: float | None = None, actual_stake: float | None = None,
                                    actual_leverage: float | None = None) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Aggressive position has no persisted protection plan")
    required = ("family", "mode", "route", "side", "pair", "entry_tag", "provenance", "reason", "features",
                "reference_rate", "open_rate", "entry_atr", "atr15", "structural_anchor", "stop_price",
                "approved_stop_price", "fill_stop_adjustment", "initial_stop_price", "target_price",
                "target_kind", "planned_loss_usdt",
                "risk_reserve_usdt", "entry_equity_usdt", "stake_usdt", "quantity", "leverage",
                "requested_leverage", "contract_size", "filled_at_utc")
    if any(name not in plan for name in required):
        raise ValueError("Aggressive persisted plan is incomplete")
    if plan["family"] not in {spec["family"] for spec in AGGRESSIVE_ACCOUNTS.values()}:
        raise ValueError("Unknown aggressive family in persisted plan")
    if plan["mode"] not in {"primary", "exploratory", "manual"}:
        raise ValueError("Unknown aggressive entry mode")
    if plan["side"] not in {"long", "short"} or plan["pair"] not in AGGRESSIVE_PAIRS:
        raise ValueError("Aggressive persisted plan identity is invalid")
    if not isinstance(plan["entry_tag"], str) or not plan["entry_tag"]:
        raise ValueError("Aggressive persisted plan lacks an entry tag")
    if plan["provenance"] not in {"automatic_signal", "main_agent_force_entry"}:
        raise ValueError("Aggressive entry provenance is invalid")
    if not isinstance(plan["reason"], str) or not plan["reason"].strip() or not isinstance(plan["features"], dict):
        raise ValueError("Aggressive plan reason/features are invalid")
    numbers = {name: _finite(plan[name], name, positive=True) for name in
               ("reference_rate", "open_rate", "entry_atr", "atr15", "structural_anchor", "stop_price",
                "approved_stop_price", "initial_stop_price",
                "target_price", "planned_loss_usdt", "risk_reserve_usdt", "entry_equity_usdt",
                "stake_usdt", "quantity", "leverage", "requested_leverage", "contract_size")}
    if not 3.0 <= numbers["leverage"] <= 10.0 or numbers["leverage"] > numbers["requested_leverage"] + 1e-9:
        raise ValueError("Aggressive persisted leverage is outside its approved range")
    if numbers["risk_reserve_usdt"] > numbers["entry_equity_usdt"] * AGGRESSIVE_LIMITS["max_position_risk_pct"] + 1e-6:
        raise ValueError("Aggressive risk reserve exceeds the 3% per-trade budget")
    side = plan["side"]
    if side == "long":
        if not numbers["initial_stop_price"] < numbers["open_rate"] < numbers["target_price"]:
            raise ValueError("Aggressive long protection is not side-correct")
        if numbers["approved_stop_price"] >= numbers["reference_rate"]:
            raise ValueError("Aggressive approved long stop is not on the loss side of its reference")
        adjustment_is_tighter = numbers["initial_stop_price"] > numbers["approved_stop_price"] + 1e-9
    else:
        if not numbers["target_price"] < numbers["open_rate"] < numbers["initial_stop_price"]:
            raise ValueError("Aggressive short protection is not side-correct")
        if numbers["approved_stop_price"] <= numbers["reference_rate"]:
            raise ValueError("Aggressive approved short stop is not on the loss side of its reference")
        adjustment_is_tighter = numbers["initial_stop_price"] < numbers["approved_stop_price"] - 1e-9
    adjustment = plan["fill_stop_adjustment"]
    if adjustment not in {None, "risk_budget_tightening"}:
        raise ValueError("Aggressive fill stop adjustment marker is invalid")
    unchanged = math.isclose(numbers["initial_stop_price"], numbers["approved_stop_price"],
                             rel_tol=1e-10, abs_tol=1e-8)
    if (adjustment is None and not unchanged) or (adjustment == "risk_budget_tightening" and
                                                   (unchanged or not adjustment_is_tighter)):
        raise ValueError("Aggressive fill stop adjustment is not explicitly and correctly marked")
    if plan["family"] == "reclaim":
        _finite(plan.get("entry_reclaim_boundary"), "entry reclaim boundary", positive=True)
    if side == "long" and numbers["stop_price"] >= numbers["target_price"]:
        raise ValueError("Aggressive long stop must remain below its target")
    if side == "short" and numbers["stop_price"] <= numbers["target_price"]:
        raise ValueError("Aggressive short stop must remain above its target")
    if side == "long" and numbers["stop_price"] < numbers["initial_stop_price"] - 1e-9:
        raise ValueError("Aggressive long stop was loosened")
    if side == "short" and numbers["stop_price"] > numbers["initial_stop_price"] + 1e-9:
        raise ValueError("Aggressive short stop was loosened")
    minimum_target_travel = 2.5 * AGGRESSIVE_LIMITS["fee_allowance"]
    target_travel = (numbers["target_price"] - numbers["open_rate"]) / numbers["open_rate"]
    if (target_travel if side == "long" else -target_travel) < minimum_target_travel - 1e-9:
        raise ValueError("Aggressive filled target is not fee-meaningful from the actual fill")
    _utc(plan["filled_at_utc"])
    if numbers["stake_usdt"] > numbers["entry_equity_usdt"] * AGGRESSIVE_LIMITS["max_margin_pct"] + 1e-5:
        raise ValueError("Aggressive filled margin exceeds the 15% per-position budget")
    initial_distance = abs(numbers["open_rate"] - numbers["initial_stop_price"]) / numbers["open_rate"]
    if numbers["leverage"] * (initial_distance + AGGRESSIVE_LIMITS["fee_allowance"]) > AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] + 1e-9:
        raise ValueError("Aggressive persisted stop exceeds its initial-margin emergency floor")
    notional = numbers["quantity"] * numbers["open_rate"] * numbers["contract_size"]
    initial_risk = notional * (initial_distance + AGGRESSIVE_LIMITS["fee_allowance"])
    adverse_current_distance = (
        max(0.0, numbers["open_rate"] - numbers["stop_price"])
        if side == "long" else max(0.0, numbers["stop_price"] - numbers["open_rate"])
    ) / numbers["open_rate"]
    current_risk = notional * (adverse_current_distance + AGGRESSIVE_LIMITS["fee_allowance"])
    if (initial_risk > numbers["planned_loss_usdt"] + 1e-5
            or current_risk > numbers["risk_reserve_usdt"] + 1e-5
            or numbers["planned_loss_usdt"] > numbers["risk_reserve_usdt"] + 1e-5):
        raise ValueError("Aggressive filled stop exceeds its durable risk reserve")
    checks = (("pair", plan["pair"], actual_pair), ("side", side, actual_side))
    for label, expected, actual in checks:
        if actual is not None and actual != expected:
            raise ValueError(f"Aggressive plan {label} differs from the persisted position")
    for label, expected, actual in (("open rate", numbers["open_rate"], actual_open_rate),
                                   ("quantity", numbers["quantity"], actual_quantity),
                                   ("stake", numbers["stake_usdt"], actual_stake),
                                   ("leverage", numbers["leverage"], actual_leverage)):
        if actual is not None and not math.isclose(expected, _finite(actual, f"actual {label}", positive=True), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Aggressive plan {label} differs from the persisted position")


def aggressive_fill_protection(*, side: str, rate: float, approved_stop: float,
                               approved_target: float, notional: float, leverage: float,
                               risk_reserve: float) -> tuple[float, str | None, float]:
    """Keep the approved absolute stop unless actual-fill budgets require tightening it."""
    if side not in {"long", "short"}:
        raise ValueError("Aggressive fill protection needs an explicit side")
    rate = _finite(rate, "actual fill price", positive=True)
    approved_stop = _finite(approved_stop, "approved absolute stop", positive=True)
    approved_target = _finite(approved_target, "approved absolute target", positive=True)
    notional = _finite(notional, "actual filled notional", positive=True)
    leverage = _finite(leverage, "actual leverage", positive=True)
    risk_reserve = _finite(risk_reserve, "risk reservation", positive=True)
    direction = 1.0 if side == "long" else -1.0
    approved_distance = direction * (rate - approved_stop) / rate
    if not math.isfinite(approved_distance) or approved_distance <= 0:
        raise ValueError("Actual fill crossed the approved absolute stop")
    fee = AGGRESSIVE_LIMITS["fee_allowance"]
    allowed_distance = min(risk_reserve / notional - fee,
                           AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] / leverage - fee)
    if not math.isfinite(allowed_distance) or allowed_distance <= 0:
        raise ValueError("Actual fill cannot support positive side-correct risk-bounded protection")
    active_distance = min(approved_distance, allowed_distance)
    active_stop = rate * (1.0 - direction * active_distance)
    adjusted = active_distance < approved_distance - 1e-12
    adjustment = "risk_budget_tightening" if adjusted else None
    target_travel = direction * (approved_target - rate) / rate
    minimum_target_travel = 2.5 * fee
    if not math.isfinite(target_travel) or target_travel < minimum_target_travel - 1e-12:
        raise ValueError("Actual fill no longer supports the approved fee-meaningful target")
    actual_risk = notional * (active_distance + fee)
    if actual_risk > risk_reserve + 1e-5:
        raise ValueError("Actual fill risk exceeds its durable reservation")
    return active_stop, adjustment, actual_risk


def build_aggressive_filled_plan(pending: dict, *, pair: str, side: str, open_rate: float,
                                 quantity: float, stake: float, leverage: float,
                                 contract_size: float, filled_at: datetime) -> dict:
    """Freeze actual fill facts and keep worst-case stop risk inside its reservation."""
    if not isinstance(pending, dict):
        raise ValueError("Aggressive fill has no staged approval")
    if not isinstance(filled_at, datetime) or filled_at.tzinfo is None or filled_at.utcoffset() is None:
        raise ValueError("Aggressive fill clock must be timezone-aware")
    if pair != pending.get("pair") or side != pending.get("side"):
        raise ValueError("Aggressive fill identity differs from its staged approval")
    rate = _finite(open_rate, "actual fill price", positive=True)
    qty = _finite(quantity, "actual filled quantity", positive=True)
    margin = _finite(stake, "actual filled margin", positive=True)
    actual_lev = _finite(leverage, "actual leverage", positive=True)
    contract = _finite(contract_size, "contract size", positive=True)
    requested = _finite(pending.get("requested_leverage"), "requested leverage", positive=True)
    equity = _finite(pending.get("entry_equity_usdt"), "entry equity", positive=True)
    reserve = _finite(pending.get("risk_reserve_usdt"), "risk reservation", positive=True)
    original_stop = _finite(pending.get("stop_price"), "staged stop", positive=True)
    original_target = _finite(pending.get("target_price"), "staged target", positive=True)
    if not 3.0 <= actual_lev <= min(requested, 10.0) + 1e-9:
        raise ValueError("Actual leverage is below 3x or exceeds the approval")
    if reserve > min(equity * AGGRESSIVE_LIMITS["max_position_risk_pct"],
                     equity * AGGRESSIVE_LIMITS["max_combined_risk_pct"]) + 1e-5:
        raise ValueError("Staged risk reservation exceeds the per-trade/combined ceiling")
    if margin > equity * AGGRESSIVE_LIMITS["max_margin_pct"] + 1e-5:
        raise ValueError("Actual filled margin exceeds 15% of entry equity")
    notional = qty * rate * contract
    if not math.isclose(notional / actual_lev, margin, rel_tol=0.03, abs_tol=1e-4):
        raise ValueError("Filled amount, price, contract size, leverage and margin do not reconcile")
    stop, fill_stop_adjustment, actual_risk = aggressive_fill_protection(
        side=side, rate=rate, approved_stop=original_stop, approved_target=original_target,
        notional=notional, leverage=actual_lev, risk_reserve=reserve)
    plan = {name: value for name, value in pending.items()
            if name not in {"at", "confirmed", "stop_price", "target_price", "risk_reserve_usdt",
                            "stake_usdt", "leverage", "quantity", "contract_size", "open_rate",
                            "approved_stop_price", "fill_stop_adjustment", "initial_stop_price",
                            "planned_loss_usdt", "filled_at_utc"}}
    plan.update({"pair": pair, "side": side, "open_rate": rate, "quantity": qty,
                 "stake_usdt": margin, "leverage": actual_lev, "requested_leverage": requested,
                 "contract_size": contract, "approved_stop_price": original_stop,
                 "fill_stop_adjustment": fill_stop_adjustment,
                 "stop_price": stop, "initial_stop_price": stop,
                 "target_price": original_target, "planned_loss_usdt": actual_risk,
                 "risk_reserve_usdt": reserve, "entry_equity_usdt": equity,
                 "filled_at_utc": filled_at.astimezone(timezone.utc).isoformat()})
    validate_aggressive_filled_plan(plan, actual_pair=pair, actual_side=side,
                                    actual_open_rate=rate, actual_quantity=qty,
                                    actual_stake=margin, actual_leverage=actual_lev)
    return plan


def aggressive_entry_plan(entry_tag: str | None, now: datetime, account: str, *, fresh: bool = True) -> dict:
    if not isinstance(entry_tag, str) or not entry_tag.startswith(MANUAL_TAG_PREFIX):
        raise ValueError("Entry has no main-agent aggressive decision tag")
    key = account_key(account)
    decision_id = entry_tag[len(MANUAL_TAG_PREFIX):]
    from user_data.strategies.paper_news_manual import decision_rows
    rows = [row for row in decision_rows(DECISION_JOURNALS[key])
            if row.get("decision_id") == decision_id and row.get("account") == AGGRESSIVE_ACCOUNTS[key]["bot_name"]
            and row.get("action") == "enter"]
    if not rows or not isinstance(rows[-1].get("plan"), dict):
        raise ValueError("Aggressive entry has no journalled main-agent plan")
    row = rows[-1]
    plan = dict(row["plan"])
    plan.update(decision_id=decision_id, reason=row.get("reason"), source=row.get("source"))
    if fresh:
        # The authenticated API can return after it has queued the force-entry
        # while the bot's strategy callbacks are still being scheduled. The
        # controller journals `submitted` after a successful response, so the
        # callback path must accept that same, single-use decision until its
        # short validity window closes. Uncertain writes have a distinct
        # status and remain non-retriable.
        if row.get("status") not in {"proposed", "submitted"} or now >= _utc(plan["valid_until_utc"]):
            raise ValueError("Aggressive entry decision is expired, uncertain, or unavailable")
        observation = load_luna_context(now)
        if observation.status != "observed" or row.get("source") not in observation.sources or not row.get("reason"):
            raise ValueError("Aggressive entry needs a fresh, valid Luna observation")
    return plan


def apply_aggressive_protection_update(plan: dict, account: str, trade_id: int, *,
                                       actual_pair: str, actual_side: str,
                                       actual_open_rate: float, actual_quantity: float,
                                       actual_stake: float, actual_leverage: float) -> dict:
    """Apply the latest explicitly journalled tightening without restoring a weaker stop."""
    key = account_key(account)
    from user_data.strategies.paper_news_manual import decision_rows
    bot_name = AGGRESSIVE_ACCOUNTS[key]["bot_name"]
    protections = [item for item in decision_rows(DECISION_JOURNALS[key])
                   if item.get("account") == bot_name and item.get("action") == "protect"
                   and item.get("status") == "recorded_protection" and item.get("trade_id") == trade_id
                   and isinstance(item.get("plan"), dict)]
    if protections:
        update = protections[-1]
        approved = dict(update["plan"])
        if (approved.get("pair") != actual_pair or approved.get("side") != actual_side
                or approved.get("protection_decision_id") != update.get("decision_id")):
            raise ValueError("Journalled protection update has an invalid account/position identity")
        if ((actual_side == "long" and float(approved["stop_price"]) < float(plan["stop_price"]))
                or (actual_side == "short" and float(approved["stop_price"]) > float(plan["stop_price"]))):
            # A later strategy profit-lock may already be tighter than the
            # journalled request. Never restore an older, weaker stop.
            approved["stop_price"] = plan["stop_price"]
        plan = approved
        validate_aggressive_filled_plan(plan, actual_pair=actual_pair, actual_side=actual_side,
            actual_open_rate=actual_open_rate, actual_quantity=actual_quantity,
            actual_stake=actual_stake, actual_leverage=actual_leverage)
    return plan


def load_aggressive_trade_plan(account: str, trade_id: int) -> dict:
    """Read one open account-local plan through SQLite's read-only URI."""
    key = account_key(account)
    if type(trade_id) is not int or trade_id <= 0:
        raise ValueError("Trade ID must be a positive integer")
    database = REPORT / f"{key}_trades.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"{key}: registered trade database is missing")
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT t.pair,t.is_short,t.open_rate,t.amount,t.stake_amount,t.leverage,t.contract_size,c.cd_value "
            "FROM trades t LEFT JOIN trade_custom_data c "
            "ON c.ft_trade_id=t.id AND c.cd_key=? WHERE t.id=? AND t.is_open=1",
            (PLAN_KEY, trade_id)).fetchone()
    if row is None or row[-1] is None:
        raise ValueError("Open aggressive position has no durable protection plan")
    pair, is_short, open_rate, quantity, stake, leverage, contract_size, encoded = row
    side = "short" if is_short else "long"
    plan = json.loads(encoded)
    validate_aggressive_filled_plan(plan, actual_pair=pair, actual_side=side,
        actual_open_rate=float(open_rate), actual_quantity=float(quantity),
        actual_stake=float(stake), actual_leverage=float(leverage))
    if not math.isclose(float(plan["contract_size"]), float(contract_size), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Persisted aggressive contract size differs from the open position")
    return apply_aggressive_protection_update(plan, key, trade_id,
        actual_pair=pair, actual_side=side, actual_open_rate=float(open_rate),
        actual_quantity=float(quantity), actual_stake=float(stake), actual_leverage=float(leverage))
