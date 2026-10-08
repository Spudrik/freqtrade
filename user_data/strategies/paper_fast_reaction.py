"""Matched 5m PAPER reaction bots; contextual account changes risk, not signals.

These are new prospective paper rules inspired by retained reaction/activity
findings, not rescaled Sieve winners or a claim of proven direction prediction.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from math import isfinite

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.strategies.paper_trial_common import PaperTrialBase, paper_user_force_close_reason
from user_data.strategies.paper_fast_context import FastControl, load_fast_control

LOG = logging.getLogger(__name__)
LEADERS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
PAIRS = {*LEADERS, "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT"}
PLAN_KEY = "paper_fast_plan"
FEE_AND_SLIPPAGE = .0014  # 0.10% round-trip fee plus 0.04% sizing/cost allowance.
MAX_POSITION_RISK = .02
MAX_COMBINED_RISK = .04  # All positions count together, even nominally opposite sides.
MAX_MARGIN = .25


def validate_fast_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("side") not in {"long", "short"}:
        raise ValueError("Invalid fast protection plan")
    for name in ("open_rate", "entry_atr", "stop_price", "target_price", "planned_loss_usdt", "entry_equity_usdt", "leverage"):
        if not isfinite(float(plan[name])) or float(plan[name]) <= 0:
            raise ValueError(f"Invalid fast protection value: {name}")
    if not 1 <= float(plan["leverage"]) <= 5:
        raise ValueError("Fast leverage exceeds its paper envelope")
    if float(plan["planned_loss_usdt"]) > MAX_POSITION_RISK * float(plan["entry_equity_usdt"]) + 1e-6:
        raise ValueError("Fast planned loss exceeds its per-position budget")
    if plan["side"] == "long":
        valid = plan["stop_price"] < plan["open_rate"] < plan["target_price"]
    else:
        valid = plan["target_price"] < plan["open_rate"] < plan["stop_price"]
    if not valid:
        raise ValueError("Fast price protection is on the wrong side")


class PaperFastAuto(PaperTrialBase):
    timeframe = "5m"
    startup_candle_count = 240
    stoploss = -.20  # Emergency position stop; entry-frozen price protection is tighter.
    paper_hold_hours = 4
    contextual = False
    account_key = "fast_auto"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pending: dict[tuple[str, str, str], dict] = {}
        self._higher_cache: dict[tuple[str, str], tuple[tuple, DataFrame]] = {}
        self._last_control_status: tuple | None = None

    def bot_start(self, **kwargs) -> None:
        exchange = self.config.get("exchange", {})
        if (self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN
                or self.config.get("trading_mode") != "futures"
                or self.config.get("margin_mode") != "isolated" or exchange.get("name") != "binance"
                or any(exchange.get(k) for k in ("key", "secret", "password", "privateKey"))
                or self.config.get("bot_name") != f"paper_{self.account_key}"
                or self.config.get("db_url") != f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{self.account_key}_trades.sqlite"
                or self.config.get("max_open_trades") != 3
                or set(exchange.get("pair_whitelist", [])) != PAIRS
                or self.config.get("force_entry_enable") or self.config.get("api_server", {}).get("enabled")):
            raise RuntimeError("Fast reaction strategy requires its exact isolated PAPER account, no order API")

    def _control(self, now: datetime) -> FastControl:
        return load_fast_control(now) if self.contextual else FastControl("observed", "automatic_baseline", 0, "normal", "normal")

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        control = self._control(current_time)
        state = (control.status, control.decision_id, control.bias, control.exposure,
                 control.entry_permission, control.leverage_for("long", current_time),
                 control.leverage_for("short", current_time))
        if state != self._last_control_status:
            LOG.info("fast_control_state account=%s state=%s", self.account_key, state)
            self._last_control_status = state

    def informative_pairs(self):
        return sorted({(pair, tf) for pair in self.dp.current_whitelist() for tf in ("15m", "1h", "4h", "1d")}
                      | {(pair, "5m") for pair in LEADERS})

    def _higher(self, pair: str, tf: str) -> DataFrame:
        source = self.dp.get_pair_dataframe(pair=pair, timeframe=tf).copy()
        if source.empty:
            raise ValueError(f"Missing {tf} candles for {pair}")
        last = source.iloc[-1]
        signature = (len(source), *(last[k] for k in ("date", "open", "high", "low", "close", "volume")))
        cached = self._higher_cache.get((pair, tf))
        if cached is not None and cached[0] == signature:
            return cached[1]
        if tf == "15m":
            source["confirm_change"] = source["close"].pct_change()
            source = source[["date", "confirm_change"]]
        elif tf == "1h":
            source["hour_high"] = source["high"].rolling(6).max()
            source["hour_low"] = source["low"].rolling(6).min()
            source["hour_change"] = source["close"].pct_change(4)
            source = source[["date", "hour_high", "hour_low", "hour_change"]]
        elif tf == "4h":
            source = add_volume_profile(source)
            source["four_high"] = source["high"].rolling(20).max()
            source["four_low"] = source["low"].rolling(20).min()
            source = source[["date", "four_high", "four_low", "vp_poc", "vp_hvn_above", "vp_hvn_below", "vp_lvn_above", "vp_lvn_below"]]
        else:
            # The merge delays a daily high/low until that entire day has closed.
            source = source[["date", "high", "low"]].rename(columns={"high": "day_high", "low": "day_low"})
        self._higher_cache[(pair, tf)] = (signature, source)
        return source

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["paper_atr"] = ta.ATR(frame, timeperiod=14)
        frame["prior_volume"] = frame["volume"].shift(1).rolling(36).median()
        # Exclude the breakout candle and present retest candle from range construction.
        frame["range_high"] = frame["high"].shift(2).rolling(12).max()
        frame["range_low"] = frame["low"].shift(2).rolling(12).min()
        for tf in ("15m", "1h", "4h", "1d"):
            frame = merge_informative_pair(frame, self._higher(metadata["pair"], tf), "5m", tf, ffill=True)
        for pair, label in zip(LEADERS, ("btc", "eth")):
            leader = self.dp.get_pair_dataframe(pair=pair, timeframe="5m").copy()
            leader[f"{label}_change"] = leader["close"].pct_change(3)
            # Both are completed 5m candles, so a same-clock merge is causal.
            frame = frame.merge(leader[["date", f"{label}_change"]], on="date", how="left", validate="one_to_one")
        return frame

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe
        frame["enter_long"], frame["enter_short"], frame["enter_tag"] = 0, 0, None
        frame["fast_trigger_level"] = np.nan
        frame["fast_strong"] = False
        atr, close = frame["paper_atr"], frame["close"]
        zone = .25 * atr
        previous = close.shift(1)
        # Each branch below already requires its actual touched level or range.
        # Unrelated profile/day levels must not veto an otherwise complete signal.
        required = ["paper_atr", "prior_volume", "confirm_change_15m", "btc_change", "eth_change"]
        ready = frame[required].notna().all(axis=1) & atr.gt(0) & frame["prior_volume"].gt(0)
        # Require target distance >= twice conservative round-trip cost allowance.
        ready &= (3 * atr / close).ge(2 * FEE_AND_SLIPPAGE)
        participation = frame["volume"].ge(frame["prior_volume"])
        long_confirm = frame["confirm_change_15m"].ge(0) & close.gt(frame["open"])
        short_confirm = frame["confirm_change_15m"].le(0) & close.lt(frame["open"])
        # A broad, simultaneous BTC/ETH move against the entry is a contradiction.
        long_confirm &= ~(frame["btc_change"].lt(-.002) & frame["eth_change"].lt(-.002))
        short_confirm &= ~(frame["btc_change"].gt(.002) & frame["eth_change"].gt(.002))
        levels = ["hour_high_1h", "hour_low_1h", "four_high_4h", "four_low_4h", "vp_poc_4h",
                  "vp_hvn_above_4h", "vp_hvn_below_4h", "vp_lvn_above_4h", "vp_lvn_below_4h",
                  "day_high_1d", "day_low_1d"]
        long_level = pd.Series(np.nan, index=frame.index)
        short_level = pd.Series(np.nan, index=frame.index)
        for name in levels:
            level = frame[name]
            touch = frame["low"].le(level + zone) & frame["high"].ge(level - zone)
            long_hit = previous.gt(level + zone) & touch & close.gt(level + zone) & (close-level).le(1.25 * atr)
            short_hit = previous.lt(level - zone) & touch & close.lt(level - zone) & (level-close).le(1.25 * atr)
            long_level = pd.concat([long_level, level.where(long_hit)], axis=1).max(axis=1)
            short_level = pd.concat([short_level, level.where(short_hit)], axis=1).min(axis=1)
        long_bounce = ready & participation & long_confirm & long_level.notna()
        short_bounce = ready & participation & short_confirm & short_level.notna()
        high, low = frame["range_high"], frame["range_low"]
        long_break = (ready & long_confirm & frame["volume"].ge(1.25 * frame["prior_volume"])
                      & previous.gt(high + zone) & close.shift(2).le(high + zone)
                      & frame["low"].le(high + zone) & close.gt(high + zone) & (close-high).le(1.25 * atr))
        short_break = (ready & short_confirm & frame["volume"].ge(1.25 * frame["prior_volume"])
                       & previous.lt(low-zone) & close.shift(2).ge(low-zone)
                       & frame["high"].ge(low-zone) & close.lt(low-zone) & (low-close).le(1.25 * atr))
        long, short = long_bounce | long_break, short_bounce | short_break
        conflict = long & short
        long, short = long & ~conflict, short & ~conflict
        frame.loc[long, "enter_long"] = 1
        frame.loc[short, "enter_short"] = 1
        frame.loc[long, "enter_tag"] = "fast_level_long"
        frame.loc[short, "enter_tag"] = "fast_level_short"
        frame.loc[long & long_break, "enter_tag"] = "fast_break_retest_long"
        frame.loc[short & short_break, "enter_tag"] = "fast_break_retest_short"
        frame.loc[long, "fast_trigger_level"] = long_level[long]
        frame.loc[short, "fast_trigger_level"] = short_level[short]
        frame.loc[long & long_break, "fast_trigger_level"] = high[long & long_break]
        frame.loc[short & short_break, "fast_trigger_level"] = low[short & short_break]
        frame["fast_strong"] = frame["volume"].ge(1.5 * frame["prior_volume"]) & (close-frame["open"]).abs().ge(.5*atr)
        return frame

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        desired = self._control(current_time).leverage_for(side, current_time)
        return max(1., min(desired, float(max_leverage)))

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Price/trigger/time exits are executed by custom_exit, not vectorized signals.
        dataframe["exit_long"], dataframe["exit_short"] = 0, 0
        return dataframe

    def _current_row(self, pair: str, now: datetime):
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise ValueError(f"No analyzed fast candle for {pair}")
        row = frame.iloc[-1]
        closed_at = pd.Timestamp(row["date"]).to_pydatetime() + timedelta(minutes=5)
        if not timedelta(0) <= now-closed_at < timedelta(minutes=10):
            raise ValueError(f"Stale/future fast candle for {pair}")
        return row

    def _committed_risk(self, equity: float) -> float:
        risk = 0.
        for trade in Trade.get_open_trades():
            if trade.strategy != self.__class__.__name__:
                raise RuntimeError("Foreign strategy trade in isolated fast account")
            plan = trade.get_custom_data(PLAN_KEY)
            if not isinstance(plan, dict):
                raise ValueError("Open fast position lacks persisted protection")
            risk += float(plan["planned_loss_usdt"])
        return min(risk, equity)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        key = (pair, side, str(entry_tag))
        self._pending.pop(key, None)
        # Clear abandoned proposals; only same-loop reservations count.
        self._pending = {k: v for k, v in self._pending.items() if timedelta(0) <= current_time-v["at"] <= timedelta(seconds=30)}
        control = self._control(current_time)
        desired = control.leverage_for(side, current_time)
        if desired <= 0:
            LOG.info("fast_entry_blocked account=%s pair=%s side=%s control=%s decision=%s", self.account_key, pair, side, control.status, control.decision_id)
            return 0.
        row = self._current_row(pair, current_time)
        if control.entry_permission == "strong_only" and not bool(row["fast_strong"]):
            LOG.info("fast_entry_blocked account=%s pair=%s reason=strong_only decision=%s", self.account_key, pair, control.decision_id)
            return 0.
        atr, level = float(row["paper_atr"]), float(row["fast_trigger_level"])
        if not isfinite(atr) or not isfinite(level) or atr <= 0 or current_rate <= 0:
            raise ValueError("Invalid fast entry ATR/level")
        if (side == "long" and current_rate <= level) or (side == "short" and current_rate >= level):
            return 0.
        equity = float(self.wallets.get_total_stake_amount())
        # Optional geometry hooks are used only by tightly scoped subclasses
        # whose entry-frozen protection differs from the original fast rule.
        # With no hook, preserve the fast family sizing formula byte-for-byte.
        entry_context_builder = getattr(self, "_entry_plan_context", None)
        entry_context = entry_context_builder(row, side) if callable(entry_context_builder) else {}
        geometry_builder = getattr(self, "_sizing_geometry", None)
        geometry = (geometry_builder(side, float(current_rate), atr, level,
                                     entry_context.get("target_level"), float(leverage))
                    if callable(geometry_builder) else {})
        unit_loss = (float(geometry["unit_loss_fraction"]) if geometry
                     else float(leverage) * (1.5*atr/current_rate + FEE_AND_SLIPPAGE))
        committed = self._committed_risk(equity) + sum(v["planned_loss_usdt"] for v in self._pending.values())
        budget = min(equity * MAX_POSITION_RISK * control.size_factor,
                     max(0., equity * MAX_COMBINED_RISK-committed))
        stake = min(equity * MAX_MARGIN * control.size_factor, budget/unit_loss, float(max_stake))
        if not isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            return 0.
        plan = {"at": current_time, "entry_atr": atr, "trigger_level": level,
                "planned_loss_usdt": stake*unit_loss, "stake_usdt": stake,
                "entry_equity_usdt": equity, "reference_rate": float(current_rate),
                "leverage": float(leverage), "decision_id": control.decision_id,
                "side": side, "entry_tag": str(entry_tag), "strong": bool(row["fast_strong"])}
        plan.update(entry_context)
        plan.update(geometry)
        self._pending[key] = plan
        LOG.info("fast_entry_plan account=%s pair=%s plan=%s", self.account_key, pair,
                 json.dumps({k: str(v) if k == "at" else v for k,v in plan.items()}))
        return stake

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        plan = self._pending.get((pair, side, str(entry_tag)))
        if plan is None or not timedelta(0) <= current_time-plan["at"] <= timedelta(seconds=30):
            return False
        if plan.get("confirmed"):
            return False
        control = self._control(current_time)
        desired = control.leverage_for(side, current_time)
        if control.decision_id != plan["decision_id"] or desired <= 0 or plan["leverage"] > desired:
            return False
        # A changed quote must still be beyond the trigger and near the sizing quote.
        if (side == "long" and rate <= plan["trigger_level"]) or (side == "short" and rate >= plan["trigger_level"]):
            return False
        if abs(rate-plan["reference_rate"]) > .25*plan["entry_atr"]:
            return False
        plan["confirmed"] = True
        return amount > 0

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        if order.ft_order_side != trade.entry_side or trade.get_custom_data(PLAN_KEY):
            return
        pending = self._pending.pop((pair, "short" if trade.is_short else "long", str(trade.enter_tag)), None)
        if pending is None:
            raise RuntimeError("Fast entry fill has no approved sizing plan")
        atr = pending["entry_atr"]
        direction = -1 if trade.is_short else 1
        filled_plan_builder = getattr(self, "_filled_plan_after_entry", None)
        if callable(filled_plan_builder):
            plan = filled_plan_builder(pending, trade, current_time)
            from user_data.strategies.paper_swing_levels import validate_swing_plan
            validate_swing_plan(plan, actual_leverage=float(trade.leverage))
        else:
            plan = {k: v for k, v in pending.items() if k not in {"at", "confirmed"}}
            plan.update(stop_price=trade.open_rate-direction*1.5*atr,
                        target_price=trade.open_rate+direction*3*atr,
                        open_rate=trade.open_rate,
                        filled_at_utc=current_time.isoformat())
            validate_fast_plan(plan)
        trade.set_custom_data(key=PLAN_KEY, value=plan)
        trade.set_custom_data(key="paper_entry_atr", value=atr)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        plan = trade.get_custom_data(PLAN_KEY)
        if not isinstance(plan, dict):
            raise ValueError("Fast position is missing protection")
        return stoploss_from_absolute(plan["stop_price"], current_rate,
                                      is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        force_close_reason = paper_user_force_close_reason(
            self.config, self.__class__.__name__,
        )
        if force_close_reason:
            return force_close_reason
        plan = trade.get_custom_data(PLAN_KEY)
        if not isinstance(plan, dict):
            raise ValueError("Fast position is missing its approved plan")
        target = plan["target_price"]
        if (trade.is_short and current_rate <= target) or (not trade.is_short and current_rate >= target):
            return "fast_atr_target"
        if current_time-trade.open_date_utc >= timedelta(hours=4):
            return "fast_time_limit"
        if current_time-trade.open_date_utc >= timedelta(hours=1) and abs(current_rate-trade.open_rate) < .5*plan["entry_atr"]:
            return "fast_stalled"
        row = self._current_row(pair, current_time)
        level, zone = plan["trigger_level"], .25*plan["entry_atr"]
        if (trade.is_short and row["close"] > level+zone) or (not trade.is_short and row["close"] < level-zone):
            return "fast_level_failed"
        return None


class PaperFastContext(PaperFastAuto):
    contextual = True
    account_key = "fast_context"
