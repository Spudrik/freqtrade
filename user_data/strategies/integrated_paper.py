"""Paper-only multi-entry Sieve portfolio with bounded, observable context roles.

This is an experimental combination, not a promoted FreqAI prediction.  Exact
Sieve V2 classes produce the entries and manage their own selected exits.  Level,
leader, Luna, and order-book evidence can caution or abstain at entry; every
decision records the observed and unavailable components.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from math import isfinite
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy
from user_data.strategies.integrated_paper_context import load_luna_context
from user_data.strategies.paper_trial_level import PaperTrialLevel
from user_data.strategies.paper_trial_level_orderbook import _recent_pressure
from user_data.strategies.paper_trial_common import paper_user_force_close_reason
from user_data.strategies.sieve3_V2_profit_ladder_three_stage_ratchet_from_mtf_std_daily_prior_high_breakout_long_1h import (
    Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyPriorHighBreakoutLong1h,
)
from user_data.strategies.sieve3_V2_profit_ladder_three_stage_ratchet_from_mtf_std_daily_h4_trend_pullback_reclaim_long_1h import (
    Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyH4TrendPullbackReclaimLong1h,
)
from user_data.strategies.sieve3_V2_profit_level_full_or_ladder_from_ladder_short_res_fail import (
    Sieve3V2ProfitLevelFullOrLadderFromLadderShortResFail,
)


LOG = logging.getLogger(__name__)
OUTPUT = Path(__file__).resolve().parents[1] / "research_news_data/context_features/integrated_paper_20260926"
EVENT_CALENDAR = Path(__file__).resolve().parents[1] / "configs/paper_trial_event_calendar_2026.json"
LEADERS = ("BTC/USDT:USDT", "ETH/USDT:USDT")

# Exit values are the exact selected validation values in the 2026-08-07
# sieve3_v2_freqai_resolved_params_handover; entry values live in source classes.
SOURCE_SPECS = (
    (
        "breakout_long",
        Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyPriorHighBreakoutLong1h,
        "long",
        {"target_1_percent": 5, "target_gap_half_percent_units": 2,
         "partial_1_five_percent_units": 1, "partial_2_five_percent_units": 8,
         "hard_stop_percent": 6},
    ),
    (
        "pullback_long",
        Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyH4TrendPullbackReclaimLong1h,
        "long",
        {"target_1_percent": 3, "target_gap_half_percent_units": 6,
         "partial_1_five_percent_units": 1, "partial_2_five_percent_units": 2,
         "hard_stop_percent": 4},
    ),
    (
        "resistance_short",
        Sieve3V2ProfitLevelFullOrLadderFromLadderShortResFail,
        "short",
        {"exit_steps": 2, "target_1_half_percent_units": 10,
         "target_2_gap_half_percent_units": 6, "partial_1_five_percent_units": 5,
         "stop_after_target_1": "unchanged", "hard_stop_percent": 2},
    ),
)
SOURCE_IDS = {spec[0] for spec in SOURCE_SPECS}


def _utc(value: object) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.to_pydatetime().astimezone(timezone.utc)


def _source_id(tag: str | None) -> str:
    parts = str(tag or "").split(":")
    if len(parts) != 2 or parts[0] not in {"sieve", "manual"} or parts[1] not in SOURCE_IDS:
        raise ValueError(f"Unknown integrated paper entry tag: {tag!r}")
    return parts[1]


class IntegratedPaper(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    startup_candle_count = 400
    process_only_new_candles = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06  # Fail-safe backstop at the widest selected Sieve hard stop.
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._sources = {}
        for source_id, cls, _, selected in SOURCE_SPECS:
            source = cls(config)
            if set(selected) != set(source.ACTIVE_SELL_PARAMS):
                raise RuntimeError(f"Selected sell parameters drifted for {source_id}")
            for name, value in selected.items():
                parameter = getattr(source, name)
                permitted = (parameter.low <= value <= parameter.high
                             if hasattr(parameter, "low") else value in parameter.opt_range)
                if not permitted:
                    raise RuntimeError(f"Invalid selected {source_id} parameter {name}={value!r}")
                parameter.value = value
            self._sources[source_id] = source
        self._levels = PaperTrialLevel(config)
        self._pending_decisions: dict[tuple[str, str, str], tuple[datetime, dict]] = {}
        with EVENT_CALENDAR.open(encoding="utf-8") as handle:
            events = json.load(handle)["events"]
        self._events = tuple(
            (_utc(row["release_utc"]), str(row["id"]))
            for row in events if row.get("auto_eligible") is True
        )

    def bot_start(self, **kwargs) -> None:
        _ = kwargs
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("IntegratedPaper is dry-run only")
        if self.config.get("trading_mode") != "futures" or self.config.get("exchange", {}).get("name") != "binance":
            raise RuntimeError("IntegratedPaper requires the approved Binance futures paper configuration")
        if self.config.get("bot_name") not in {"integrated_paper_auto", "integrated_paper_manual"}:
            raise RuntimeError("Unknown integrated paper account")
        if any(self.config.get("exchange", {}).get(key) for key in ("key", "secret")):
            raise RuntimeError("Paper account must not contain exchange API credentials")
        OUTPUT.mkdir(parents=True, exist_ok=True)

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        pairs = set((pair, "1h") for pair in LEADERS)
        self._levels.dp = self.dp
        pairs.update(self._levels.informative_pairs())
        for source in self._sources.values():
            source.dp = self.dp
            pairs.update(source.informative_pairs())
        return sorted(pairs)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for source_id, _, side, _ in SOURCE_SPECS:
            source = self._sources[source_id]
            source.dp = self.dp
            source_frame = source.populate_indicators(dataframe.copy(), metadata)
            source_frame = source.populate_entry_trend(source_frame, metadata)
            candidate_column = source_frame.get(f"enter_{side}", pd.Series(0, index=source_frame.index))
            signal = pd.to_numeric(candidate_column, errors="coerce").fillna(0).eq(1)
            dataframe[f"candidate_{source_id}"] = signal.to_numpy(dtype=bool)

        self._levels.dp = self.dp
        level_frame = self._levels.populate_indicators(dataframe.copy(), metadata)
        level_frame = self._levels.populate_entry_trend(level_frame, metadata)
        for name in (*self._levels.level_columns, "paper_atr"):
            dataframe[name] = level_frame[name].to_numpy()
        for side in ("long", "short"):
            level_column = level_frame.get(f"enter_{side}", pd.Series(0, index=level_frame.index))
            dataframe[f"level_reaction_{side}"] = level_column.eq(1).to_numpy(dtype=bool)
        dataframe["level_reaction_tag"] = level_frame.get("enter_tag", pd.Series("", index=level_frame.index)).fillna("").to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        long_any = dataframe["candidate_breakout_long"] | dataframe["candidate_pullback_long"]
        short_any = dataframe["candidate_resistance_short"]
        long_allowed = long_any & ~short_any
        short_allowed = short_any & ~long_any
        dataframe.loc[long_allowed, "enter_long"] = 1
        dataframe.loc[short_allowed, "enter_short"] = 1
        dataframe.loc[long_allowed & dataframe["candidate_breakout_long"], "enter_tag"] = "sieve:breakout_long"
        dataframe.loc[long_allowed & ~dataframe["candidate_breakout_long"], "enter_tag"] = "sieve:pullback_long"
        dataframe.loc[short_allowed, "enter_tag"] = "sieve:resistance_short"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def _leader_state(self, pair: str, now: datetime, side: str) -> str:
        frame = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
        completed = frame.loc[frame["date"].map(_utc) + pd.Timedelta(hours=1) <= now]
        if len(completed) < 16 or now - _utc(completed.iloc[-1]["date"]) > timedelta(hours=2):
            return "unobserved"
        previous = completed.iloc[-2]
        latest = completed.iloc[-1]
        typical = (completed["high"] - completed["low"]).iloc[-15:-1].median()
        if not isfinite(float(typical)) or float(typical) <= 0:
            return "unobserved"
        move = float(latest["close"] - previous["close"])
        if abs(move) < 0.5 * float(typical):
            return "quiet"
        return "aligned" if (move > 0) == (side == "long") else "opposed"

    def _decision(self, pair: str, side: str, tag: str, now: datetime, rate: float) -> dict:
        source_id = _source_id(tag)
        if tag.startswith("manual:") and self.config.get("bot_name") != "integrated_paper_manual":
            raise RuntimeError("Manual paper entry attempted in automatic account")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise RuntimeError(f"No analyzed paper candle for {pair}")
        row = frame.iloc[-1]
        candle_time = _utc(row["date"])
        if not timedelta(0) <= now - (candle_time + timedelta(hours=1)) < timedelta(hours=2):
            raise RuntimeError(f"Stale or future paper candle for {pair}")

        supports: list[str] = []
        cautions: list[str] = []
        observed: dict[str, object] = {}
        level_side = "long" if side == "long" else "short"
        if bool(row[f"level_reaction_{level_side}"]):
            supports.append("same_side_level_reaction")
        if bool(row[f"level_reaction_{'short' if side == 'long' else 'long'}"]):
            cautions.append("opposing_level_reaction")
        zone = float(row["paper_atr"]) * 0.25
        if not isfinite(zone) or zone <= 0:
            raise RuntimeError(f"Invalid paper level zone for {pair}")
        barriers = [float(row[name]) for name in self._levels.level_columns
                    if pd.notna(row[name]) and isfinite(float(row[name]))
                    and (0 < float(row[name]) - rate <= zone if side == "long"
                         else 0 < rate - float(row[name]) <= zone)]
        if barriers:
            cautions.append("nearby_opposing_level")
        observed["level"] = {"reaction": str(row["level_reaction_tag"]),
                             "nearby_opposing_count": len(barriers), "zone_price": zone}

        leader = {name: self._leader_state(name, now, side) for name in LEADERS}
        observed["leaders"] = leader
        if all(state == "opposed" for state in leader.values()):
            cautions.append("both_leaders_opposed")
        elif leader[LEADERS[0]] == "opposed":
            cautions.append("bitcoin_opposed")
        elif leader[LEADERS[0]] == "aligned":
            supports.append("bitcoin_aligned")

        recent_events = [event_id for event_time, event_id in self._events
                         if timedelta(0) <= now - event_time <= timedelta(hours=1)]
        observed["scheduled_events"] = recent_events
        if recent_events and leader[LEADERS[0]] == "opposed":
            cautions.append("post_event_bitcoin_opposed")

        luna = load_luna_context(now)
        observed["luna"] = {"status": luna.status, "risk_bias": luna.risk_bias,
                            "event_scale": luna.event_scale, "attention": luna.attention,
                            "event_id": luna.event_id, "sources": list(luna.sources)}
        luna_opposed = luna.status == "observed" and (
            (luna.risk_bias == "risk_off" and side == "long") or
            (luna.risk_bias == "risk_on" and side == "short")
        )
        if luna_opposed:
            cautions.append("sourced_luna_risk_opposed")
        elif luna.status == "observed" and luna.risk_bias in {"risk_on", "risk_off"}:
            supports.append("sourced_luna_risk_aligned")

        pressure, book_status = _recent_pressure(pair, now)
        observed["orderbook"] = {"status": book_status, "top20_3m_imbalance": pressure}
        if book_status == "observed" and pressure is not None:
            if (pressure <= -0.15 and side == "long") or (pressure >= 0.15 and side == "short"):
                cautions.append("opposing_orderbook_pressure")

        major_conflict = luna_opposed and luna.event_scale == "major" and leader[LEADERS[0]] == "opposed"
        abstain = major_conflict or (len(cautions) >= 2 and not supports)
        size_factor = 0.0 if abstain else (0.5 if len(cautions) > len(supports) else 1.0)
        return {
            "at_utc": now.astimezone(timezone.utc).isoformat(), "pair": pair,
            "side": side, "entry_tag": tag, "source_id": source_id,
            "manual": tag.startswith("manual:"), "rate": rate,
            "supports": supports, "cautions": cautions, "observed": observed,
            "size_factor": size_factor, "decision": "abstain" if abstain else "enter",
        }

    def _record(self, decision: dict) -> None:
        name = "manual" if self.config["bot_name"] == "integrated_paper_manual" else "auto"
        path = OUTPUT / f"decisions_{name}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")
        LOG.info("integrated_paper_decision %s", json.dumps(decision, sort_keys=True))

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        _ = (leverage, kwargs)
        key = (pair, side, str(entry_tag))
        self._pending_decisions.pop(key, None)
        try:
            decision = self._decision(pair, side, str(entry_tag), current_time, current_rate)
        except (ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as error:
            LOG.error("Integrated paper entry refused because context could not be verified: %s", error)
            return 0.0
        equity = float(self.wallets.get_total_stake_amount())
        stake = min(equity * 0.02 * decision["size_factor"], float(max_stake))
        if str(entry_tag).startswith("manual:"):
            stake = min(stake, float(proposed_stake))
        if not isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            stake = 0.0
        decision["chosen_stake"] = stake
        self._record(decision)
        # Freqtrade's callback wrapper may substitute its proposed stake after an
        # exception.  Only a successfully journaled decision can pass confirmation.
        self._pending_decisions[key] = (current_time, decision)
        return stake

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        _ = (order_type, amount, rate, time_in_force, kwargs)
        key = (pair, side, str(entry_tag))
        pending = self._pending_decisions.get(key)
        if pending is None or current_time - pending[0] > timedelta(seconds=30):
            LOG.error("Integrated paper entry refused: no fresh recorded sizing decision")
            return False
        return pending[1]["decision"] == "enter" and pending[1]["chosen_stake"] > 0

    def _trade_source(self, trade):
        return self._sources[_source_id(trade.enter_tag)]

    def order_filled(self, pair, trade, order, current_time, **kwargs) -> None:
        source = self._trade_source(trade)
        source.dp = self.dp
        source.order_filled(pair, trade, order, current_time, **kwargs)

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        source = self._trade_source(trade)
        source.dp = self.dp
        return source.custom_stoploss(pair, trade, current_time, current_rate,
                                      current_profit, after_fill, **kwargs)

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs):
        force_close_reason = paper_user_force_close_reason(
            self.config, self.__class__.__name__,
        )
        if force_close_reason:
            return force_close_reason
        source = self._trade_source(trade)
        source.dp = self.dp
        source_exit = source.custom_exit(pair, trade, current_time, current_rate,
                                         current_profit, **kwargs)
        if source_exit:
            return source_exit
        side = "short" if trade.is_short else "long"
        luna = load_luna_context(current_time)
        opposed = luna.status == "observed" and luna.event_scale == "major" and (
            (luna.risk_bias == "risk_off" and side == "long") or
            (luna.risk_bias == "risk_on" and side == "short")
        )
        if not opposed:
            return None
        leaders = {leader: self._leader_state(leader, current_time, side) for leader in LEADERS}
        if not all(state == "opposed" for state in leaders.values()):
            return None
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            return None
        candle_age = current_time - (_utc(frame.iloc[-1]["date"]) + timedelta(hours=1))
        if not timedelta(0) <= candle_age < timedelta(hours=2):
            return None
        opposing_side = "long" if side == "short" else "short"
        if not bool(frame.iloc[-1][f"level_reaction_{opposing_side}"]):
            return None
        if not trade.get_custom_data("integrated_context_exit_recorded"):
            self._record({
                "at_utc": current_time.astimezone(timezone.utc).isoformat(),
                "decision": "context_exit", "pair": pair, "side": side,
                "trade_id": trade.id, "source_id": _source_id(trade.enter_tag),
                "reason": "major_sourced_event_both_leaders_opposed_local_level_reversal",
                "luna_event_id": luna.event_id, "luna_sources": list(luna.sources),
                "leaders": leaders,
            })
            trade.set_custom_data("integrated_context_exit_recorded", True)
        return "context_major_reversal"

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        source = self._trade_source(trade)
        source.dp = self.dp
        return source.adjust_trade_position(
            trade, current_time, current_rate, current_profit, min_stake, max_stake,
            current_entry_rate, current_exit_rate, current_entry_profit,
            current_exit_profit, **kwargs,
        )
