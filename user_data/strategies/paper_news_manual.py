"""Discretionary paper broker: no automatic entries or news-model order authority.

The main agent approves each trade in the existing controller's account journal.
Only the approved stop and optional target run unattended between agent reviews.
Existing 1h/4h volume-profile calculations are exposed for human/agent inspection;
they cannot create a position. No network/news fetching occurs in this strategy.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from math import isfinite, isclose
from pathlib import Path

from freqtrade.enums import RunMode
from freqtrade.strategy import stoploss_from_absolute
from user_data.strategies.integrated_paper_context import load_luna_context, _utc
from user_data.strategies.paper_trial_level import PaperTrialLevel


LOG = logging.getLogger(__name__)
OUTPUT = Path(__file__).resolve().parents[1] / "research_news_data/context_features/integrated_paper_20260926"
NEWS_JOURNAL = OUTPUT / "news_manual_decisions.jsonl"
NEWS_FAST_JOURNAL = OUTPUT / "news_fast_decisions.jsonl"
SIEVE_PAIRS = frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"})
BASE_NEWS_PAIRS = frozenset({*SIEVE_PAIRS, "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT"})
MANUAL_EMERGENCY_MARGIN_LOSS_PCT = 0.25
MANUAL_ACCOUNTS = {
    "paper_news_manual": (8097, NEWS_JOURNAL),
    "paper_news_lab": (8098, OUTPUT / "news_lab_decisions.jsonl"),
    "paper_news_fast": (8099, NEWS_FAST_JOURNAL),
}
MANUAL_ACCOUNT_PAIRS = {
    "paper_news_manual": BASE_NEWS_PAIRS,
    "paper_news_lab": BASE_NEWS_PAIRS,
    "paper_news_fast": SIEVE_PAIRS,
}
MANUAL_ACCOUNT_CAPS = {
    "paper_news_manual": {"max_stake_pct": 0.25, "max_leverage": 5.0},
    "paper_news_lab": {"max_stake_pct": 0.25, "max_leverage": 5.0},
    "paper_news_fast": {"max_stake_pct": 0.15, "max_leverage": 10.0},
}
PLAN_KEY = "paper_news_manual_plan"
TAG_PREFIX = "news_manual:"


def validate_manual_plan(plan: dict, rate: float, account: str = "paper_news_manual", *,
                         actual_leverage: float | None = None) -> None:
    """Reject ill-defined protection instead of inferring missing price levels.

    Stake ceilings are enforced on fresh entries by the controller and entry
    callbacks. They do not apply to existing-position protection plans, whose
    stake percentage can rise after account equity falls.
    """
    if account not in MANUAL_ACCOUNT_CAPS:
        raise ValueError("Unknown manual paper account")
    caps = MANUAL_ACCOUNT_CAPS[account]
    if plan.get("side") not in {"long", "short"} or not isinstance(plan.get("pair"), str):
        raise ValueError("Manual plan needs an explicit pair and side")
    for key in ("stop_price", "reference_rate", "stake_pct", "leverage"):
        if plan.get(key) is None or not isfinite(float(plan[key])) or float(plan[key]) <= 0:
            raise ValueError(f"Invalid manual plan {key}")
    if not isfinite(rate) or rate <= 0:
        raise ValueError("Invalid current decision price")
    if (plan["pair"] not in MANUAL_ACCOUNT_PAIRS[account]
            or float(plan["leverage"]) > caps["max_leverage"]):
        raise ValueError("Manual plan exceeds this account's pair or risk envelope")
    if actual_leverage is not None:
        if (not isfinite(float(actual_leverage)) or float(actual_leverage) <= 0
                or not isclose(float(plan["leverage"]), float(actual_leverage))):
            raise ValueError("Persisted plan leverage does not match actual position leverage")
    stop = float(plan["stop_price"])
    short = plan["side"] == "short"
    if (short and stop <= rate) or (not short and stop >= rate):
        raise ValueError("Stop must be beyond the current price on the loss side")
    if abs(stop / rate - 1.) * float(plan["leverage"]) >= MANUAL_EMERGENCY_MARGIN_LOSS_PCT:
        raise ValueError("Planned stop must be inside the 25% initial-margin emergency floor")
    target = plan.get("take_profit_price")
    if target is not None:
        target = float(target)
        if not isfinite(target) or target <= 0 or (short and target >= rate) or (not short and target <= rate):
            raise ValueError("Target must be on the profitable side of the current price")
    _utc(plan["valid_until_utc"])
    _utc(plan["review_due_at_utc"])


def decision_rows(path: Path = NEWS_JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entry_plan(entry_tag: str | None, now: datetime, *, fresh: bool,
               path: Path = NEWS_JOURNAL, account: str = "paper_news_manual") -> dict:
    if not isinstance(entry_tag, str) or not entry_tag.startswith(TAG_PREFIX):
        raise ValueError("Entry has no main-agent decision tag")
    decision_id = entry_tag[len(TAG_PREFIX):]
    rows = [r for r in decision_rows(path) if r.get("decision_id") == decision_id
            and r.get("account") == account and r.get("action") == "enter"]
    if not rows or not isinstance(rows[-1].get("plan"), dict):
        raise ValueError("Entry has no journalled main-agent plan")
    row = rows[-1]
    plan = dict(row["plan"])
    if fresh:
        if row.get("status") != "proposed" or now >= _utc(plan["valid_until_utc"]):
            raise ValueError("Entry decision is expired or already submitted")
        observation = load_luna_context(now)
        if observation.status != "observed" or row.get("source") not in observation.sources or not row.get("reason"):
            raise ValueError("New entry requires a fresh, valid Luna observation")
    return plan


class PaperNewsManual(PaperTrialLevel):
    """Paper execution and explicit protection only; discretionary entries stay outside."""

    position_adjustment_enable = False

    def _account(self) -> tuple[str, Path]:
        name = self.config["bot_name"]
        return name, MANUAL_ACCOUNTS[name][1]

    def _entry_plan(self, entry_tag, now, *, fresh=True):
        account, journal = self._account()
        return entry_plan(entry_tag, now, fresh=fresh, path=journal, account=account)

    def _validate_plan(self, plan, rate):
        account, _ = self._account()
        validate_manual_plan(plan, rate, account)

    def bot_start(self, **kwargs) -> None:
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperNewsManual is dry-run only")
        api = self.config.get("api_server", {})
        if (self.config.get("bot_name") not in MANUAL_ACCOUNTS
                or self.config.get("trading_mode") != "futures"
                or self.config.get("exchange", {}).get("name") != "binance"
                or any(self.config.get("exchange", {}).get(k) for k in ("key", "secret"))
                or self.config.get("force_entry_enable") is not True
                or api.get("enabled") is not True
                or api.get("listen_ip_address") != "127.0.0.1"
                or api.get("listen_port") != MANUAL_ACCOUNTS[self.config["bot_name"]][0]):
            raise RuntimeError("Invalid manual-only local paper-account configuration")
        limits = self.config["paper_manual_limits"]
        account, _ = self._account()
        caps = MANUAL_ACCOUNT_CAPS[account]
        if (not 0 < float(limits["max_stake_pct"]) <= caps["max_stake_pct"]
                or not 1 <= float(limits["max_leverage"]) <= caps["max_leverage"]
                or float(limits.get("emergency_margin_loss_pct", MANUAL_EMERGENCY_MARGIN_LOSS_PCT))
                != MANUAL_EMERGENCY_MARGIN_LOSS_PCT
                or self.stoploss != -MANUAL_EMERGENCY_MARGIN_LOSS_PCT
                or set(self.config.get("exchange", {}).get("pair_whitelist", [])) != MANUAL_ACCOUNT_PAIRS[account]):
            raise RuntimeError("Invalid discretionary paper risk envelope")
        expected_db = f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{account.removeprefix('paper_')}_trades.sqlite"
        if self.config.get("db_url") != expected_db:
            raise RuntimeError("Manual paper account has an unexpected database")

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        plan = self._entry_plan(entry_tag, current_time)
        return min(float(plan["leverage"]), float(max_leverage))

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        try:
            plan = self._entry_plan(entry_tag, current_time)
            self._validate_plan(plan, current_rate)
            limits = self.config["paper_manual_limits"]
            if (plan["pair"] != pair or plan["side"] != side
                    or not isclose(float(plan["leverage"]), float(leverage))
                    or not 1 <= leverage <= float(limits["max_leverage"])
                    or not 0 < float(plan["stake_pct"]) <= float(limits["max_stake_pct"])):
                return 0.
            stake = float(self.wallets.get_total_stake_amount()) * float(plan["stake_pct"])
            if stake > max_stake or (min_stake is not None and stake < min_stake):
                return 0.
            return stake
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOG.error("Manual paper sizing refused: %s", error)
            return 0.

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        # Freqtrade's callback wrapper defaults to accepting on exceptions, so
        # expected invalid-plan/IO failures must explicitly return False here.
        try:
            plan = self._entry_plan(entry_tag, current_time)
            self._validate_plan(plan, rate)
            limits = self.config["paper_manual_limits"]
            requested = float(self.wallets.get_total_stake_amount()) * float(plan["stake_pct"])
            return (plan["pair"] == pair and plan["side"] == side
                    and amount > 0
                    and 0 < float(plan["stake_pct"]) <= float(limits["max_stake_pct"])
                    and 1 <= float(plan["leverage"]) <= float(limits["max_leverage"])
                    and amount * rate / float(plan["leverage"]) <= requested * (1. + 1e-8))
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOG.error("Manual paper entry refused: %s", error)
            return False

    def order_filled(self, pair, trade, order, current_time, **kwargs) -> None:
        if order.ft_order_side == trade.entry_side and trade.get_custom_data(PLAN_KEY) is None:
            plan = self._entry_plan(trade.enter_tag, current_time, fresh=False)
            trade.set_custom_data(key=PLAN_KEY, value=plan)

    def _trade_plan(self, trade) -> dict:
        plan = trade.get_custom_data(PLAN_KEY)
        if not isinstance(plan, dict):
            raise RuntimeError(f"Manual paper trade {trade.id} has no approved protection plan")
        account, journal = self._account()
        updates = [r for r in decision_rows(journal) if r.get("account") == account
                   and r.get("action") == "protect" and r.get("status") == "recorded_protection"
                   and r.get("trade_id") == trade.id]
        if updates:
            latest = updates[-1]["plan"]
            if latest["pair"] != trade.pair or (latest["side"] == "short") != trade.is_short:
                raise ValueError("Protection update does not match the paper position")
            plan = latest
        validate_manual_plan(plan, float(plan["reference_rate"]), account,
                             actual_leverage=float(trade.leverage))
        return plan

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        price = float(self._trade_plan(trade)["stop_price"])
        if (trade.is_short and current_rate >= price) or (not trade.is_short and current_rate <= price):
            return None  # custom_exit executes the approved invalidation instead.
        return stoploss_from_absolute(price, current_rate, is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        plan = self._trade_plan(trade)
        stop = float(plan["stop_price"])
        if (trade.is_short and current_rate >= stop) or (not trade.is_short and current_rate <= stop):
            return "manual_news_invalidation"
        target = plan.get("take_profit_price")
        if target is not None and ((trade.is_short and current_rate <= float(target))
                                   or (not trade.is_short and current_rate >= float(target))):
            return "manual_news_approved_target"
        return None
