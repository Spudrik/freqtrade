"""Ten parked, paper-only Sieve portfolios with distinct context questions.

These are prospective comparisons, not promoted FreqAI models or ten independent
discoveries.  Each uses the exact selected Sieve entry/exit contracts from
IntegratedPaper.  Only the named source families and decision-time modifiers
differ.  Variant descriptions and frozen inputs live beside the executable code.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from math import isfinite

import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from user_data.strategies.integrated_paper import (
    IntegratedPaper, LEADERS, OUTPUT, SOURCE_IDS, SOURCE_SPECS, _source_id, _utc,
)
from user_data.strategies.integrated_paper_context import load_luna_context
from user_data.strategies.paper_trial_level_orderbook import _recent_pressure


# All profiles keep the same six paper pairs, 1x leverage, 2%-of-equity maximum
# stake, three-position cap, and source-specific Sieve exits as the running trial.
# "Level" means a previously calculable single area OR cluster; clustering gets
# no automatic superiority.  Volume is readiness, never a directional vote.
VARIANT_SPECS = {
    "01": {"name": "Sieve-only reference", "mix": "low", "complexity": "low",
           "sources": SOURCE_IDS, "inputs": (), "rule": "sieve_only"},
    "02": {"name": "Sieve with volume readiness", "mix": "low", "complexity": "low",
           "sources": SOURCE_IDS, "inputs": ("volume",), "rule": "volume_size"},
    "03": {"name": "Long Sieve with completed-hour cross-coin confirmation", "mix": "low", "complexity": "low",
           "sources": ("breakout_long", "pullback_long"), "inputs": ("leader",), "rule": "leader_gate"},
    "04": {"name": "Prior-high breakout at a reacting calculated area", "mix": "low", "complexity": "high",
           "sources": ("breakout_long",), "inputs": ("level",), "rule": "level_reaction_gate"},
    "05": {"name": "Resistance failure with opposing-area room check", "mix": "low", "complexity": "high",
           "sources": ("resistance_short",), "inputs": ("level",), "rule": "level_room_size"},
    "06": {"name": "Sieve after an event-window completed-hour market move", "mix": "medium", "complexity": "medium",
           "sources": SOURCE_IDS, "inputs": ("event", "leader"), "rule": "event_leader_gate"},
    "07": {"name": "Sieve with live order-book caution", "mix": "medium", "complexity": "low",
           "sources": SOURCE_IDS, "inputs": ("book",), "rule": "book_size"},
    "08": {"name": "Sieve with sourced market/news risk and leader", "mix": "medium", "complexity": "medium",
           "sources": SOURCE_IDS, "inputs": ("luna", "leader"), "rule": "luna_leader_size"},
    "09": {"name": "High activity with either local-area or leader confirmation", "mix": "high", "complexity": "low",
           "sources": SOURCE_IDS, "inputs": ("level", "volume", "leader"), "rule": "simple_alignment"},
    "10": {"name": "Event-aware multi-source hierarchy and reversal exit", "mix": "high", "complexity": "high",
           "sources": SOURCE_IDS, "inputs": ("level", "volume", "leader", "event", "book", "luna"),
           "rule": "hierarchy"},
}


class _ParkedPaperVariant:
    """Policy mixin only; concrete classes inherit the exact Sieve adapter."""

    variant_id: str = ""

    @property
    def spec(self) -> dict:
        return VARIANT_SPECS[self.variant_id]

    def bot_start(self, **kwargs) -> None:
        _ = kwargs
        expected = f"integrated_paper_v{self.variant_id}"
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("Parked paper variants are dry-run only")
        if (self.config.get("bot_name") != expected or self.config.get("trading_mode") != "futures"
                or self.config.get("exchange", {}).get("name") != "binance"):
            raise RuntimeError("Wrong paper variant account or exchange")
        if any(self.config.get("exchange", {}).get(key) for key in ("key", "secret")):
            raise RuntimeError("Paper variant must not contain exchange credentials")
        if self.config.get("force_entry_enable") or self.config.get("api_server", {}).get("enabled"):
            raise RuntimeError("Parked variant accounts cannot expose manual-order APIs")
        expected_db = (
            "sqlite:///user_data/research_news_data/context_features/"
            f"integrated_paper_20260926/v{self.variant_id}_trades.sqlite"
        )
        if self.config.get("db_url") != expected_db:
            raise RuntimeError("Paper variant trade database is not isolated")
        if (self.config.get("stake_currency") != "USDT"
                or self.config.get("stake_amount") != "unlimited"
                or self.config.get("max_open_trades") != 3
                or self.config.get("available_capital") != 10000.0
                or self.config.get("margin_mode") != "isolated"):
            raise RuntimeError("Paper variant portfolio risk configuration drifted")
        approved_pairs = {
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
            "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT",
        }
        if set(self.config.get("exchange", {}).get("pair_whitelist", ())) != approved_pairs:
            raise RuntimeError("Paper variant pair cohort drifted")
        (OUTPUT / "variants").mkdir(parents=True, exist_ok=True)

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        pairs: set[tuple[str, str]] = set()
        if "leader" in self.spec["inputs"]:
            pairs.update((pair, "1h") for pair in LEADERS)
        if "level" in self.spec["inputs"]:
            self._levels.dp = self.dp
            pairs.update(self._levels.informative_pairs())
        for source_id in self.spec["sources"]:
            source = self._sources[source_id]
            source.dp = self.dp
            pairs.update(source.informative_pairs())
        return sorted(pairs)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        active = set(self.spec["sources"])
        for source_id, _, side, _ in SOURCE_SPECS:
            if source_id not in active:
                dataframe[f"candidate_{source_id}"] = False
                continue
            source = self._sources[source_id]
            source.dp = self.dp
            source_frame = source.populate_indicators(dataframe.copy(), metadata)
            source_frame = source.populate_entry_trend(source_frame, metadata)
            candidate = source_frame.get(f"enter_{side}", pd.Series(0, index=source_frame.index))
            dataframe[f"candidate_{source_id}"] = pd.to_numeric(candidate, errors="coerce").fillna(0).eq(1).to_numpy()

        if "level" in self.spec["inputs"]:
            self._levels.dp = self.dp
            level_frame = self._levels.populate_indicators(dataframe.copy(), metadata)
            level_frame = self._levels.populate_entry_trend(level_frame, metadata)
            for name in (*self._levels.level_columns, "paper_atr"):
                dataframe[name] = level_frame[name].to_numpy()
            for side in ("long", "short"):
                level_signal = level_frame.get(f"enter_{side}", pd.Series(0, index=level_frame.index))
                dataframe[f"level_reaction_{side}"] = level_signal.eq(1).to_numpy()
            dataframe["level_reaction_tag"] = level_frame.get(
                "enter_tag", pd.Series("", index=level_frame.index)
            ).fillna("").to_numpy()
            if "volume" in self.spec["inputs"]:
                dataframe["prior_20_volume_median"] = level_frame["prior_20_volume_median"].to_numpy()
        elif "volume" in self.spec["inputs"]:
            dataframe["prior_20_volume_median"] = dataframe["volume"].shift(1).rolling(20).median()
        return dataframe

    def _record(self, decision: dict) -> None:
        path = OUTPUT / "variants" / f"decisions_v{self.variant_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")

    def _independent_leaders(self, pair: str, now: datetime, side: str) -> dict[str, str]:
        return {
            leader: self._leader_state(leader, now, side)
            for leader in LEADERS if leader != pair
        }

    @staticmethod
    def _leader_alignment(pair: str, leaders: dict[str, str]) -> tuple[bool, bool]:
        # Bitcoin's own move is not counted as independent confirmation of BTC.
        if pair in LEADERS:
            state = next(iter(leaders.values()), "unobserved")
            return state == "aligned", state == "opposed"
        btc = leaders.get(LEADERS[0], "unobserved")
        eth = leaders.get(LEADERS[1], "unobserved")
        return btc == "aligned" and eth != "opposed", btc == "opposed" and eth == "opposed"

    def _decision(self, pair: str, side: str, tag: str, now: datetime, rate: float) -> dict:
        source_id = _source_id(tag)
        if tag.startswith("manual:") or source_id not in self.spec["sources"]:
            raise ValueError("Manual or inactive-source entry is forbidden in a parked paper variant")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise RuntimeError(f"No analyzed candle for {pair}")
        row = frame.iloc[-1]
        candle_age = now - (_utc(row["date"]) + timedelta(hours=1))
        if not timedelta(0) <= candle_age < timedelta(hours=2):
            raise RuntimeError(f"Stale or future candle for {pair}")

        inputs = self.spec["inputs"]
        observed: dict[str, object] = {}
        supports: list[str] = []
        cautions: list[str] = []
        abstain = False
        factor = 1.0
        level_same = level_opposed = barrier = False
        high_volume = False
        aligned = leader_opposed = False
        book_opposed = False
        luna_opposed = False
        recent_events: list[str] = []
        luna = None

        if "level" in inputs:
            level_same = bool(row[f"level_reaction_{side}"])
            opposite = "short" if side == "long" else "long"
            level_opposed = bool(row[f"level_reaction_{opposite}"])
            zone = float(row["paper_atr"]) * 0.25
            if not isfinite(zone) or zone <= 0:
                raise RuntimeError(f"Invalid calculated level zone for {pair}")
            levels = [float(row[name]) for name in self._levels.level_columns
                      if pd.notna(row[name]) and isfinite(float(row[name]))]
            barrier = any(0 < (level - rate if side == "long" else rate - level) <= zone
                          for level in levels)
            observed["level"] = {"same_side_reaction": level_same,
                                 "opposing_reaction": level_opposed,
                                 "nearby_opposing_area": barrier,
                                 "single_or_cluster": str(row["level_reaction_tag"])}
        if "volume" in inputs:
            median = float(row["prior_20_volume_median"])
            if not isfinite(median) or median <= 0:
                raise RuntimeError(f"Missing causal volume baseline for {pair}")
            ratio = float(row["volume"]) / median
            high_volume = isfinite(ratio) and ratio > 1.0
            observed["volume"] = {"ratio_to_previous_20_median": ratio,
                                  "readiness_high": high_volume}
        if "leader" in inputs:
            leaders = self._independent_leaders(pair, now, side)
            aligned, leader_opposed = self._leader_alignment(pair, leaders)
            observed["independent_leaders"] = leaders
        if "event" in inputs:
            recent_events = [event_id for event_time, event_id in self._events
                             if timedelta(0) <= now - event_time <= timedelta(hours=1)]
            observed["recent_scheduled_events"] = recent_events
        if "book" in inputs:
            pressure, status = _recent_pressure(pair, now)
            book_opposed = status == "observed" and pressure is not None and (
                pressure <= -0.15 if side == "long" else pressure >= 0.15
            )
            observed["orderbook"] = {"status": status, "top20_3m_imbalance": pressure,
                                     "opposed": book_opposed}
        if "luna" in inputs:
            luna = load_luna_context(now)
            luna_opposed = luna.status == "observed" and (
                (luna.risk_bias == "risk_off" and side == "long") or
                (luna.risk_bias == "risk_on" and side == "short")
            )
            observed["luna"] = {"status": luna.status, "risk_bias": luna.risk_bias,
                                "event_scale": luna.event_scale,
                                "event_id": luna.event_id, "sources": list(luna.sources)}

        rule = self.spec["rule"]
        if rule == "volume_size" and not high_volume:
            factor = 0.5
            cautions.append("low_recent_volume_readiness")
        elif rule == "leader_gate":
            abstain = not aligned
            (supports if aligned else cautions).append("independent_leader_alignment" if aligned else "leader_not_aligned")
        elif rule == "level_reaction_gate":
            abstain = not level_same
            (supports if level_same else cautions).append("same_side_calculated_area_reaction" if level_same else "no_same_side_reaction")
        elif rule == "level_room_size" and (barrier or level_opposed):
            factor = 0.5
            cautions.append("nearby_or_opposing_calculated_area")
        elif rule == "event_leader_gate":
            abstain = not recent_events or not aligned
            (supports if not abstain else cautions).append("event_then_leader_aligned" if not abstain else "event_or_leader_not_confirmed")
        elif rule == "book_size" and book_opposed:
            factor = 0.5
            cautions.append("observed_opposing_orderbook_pressure")
        elif rule == "luna_leader_size" and luna_opposed:
            factor = 0.0 if luna and luna.event_scale == "major" and leader_opposed else 0.5
            abstain = factor == 0.0
            cautions.append("sourced_news_risk_opposed")
        elif rule == "simple_alignment":
            abstain = not (high_volume and (level_same or aligned) and not leader_opposed)
            (supports if not abstain else cautions).append("activity_plus_location_or_leader" if not abstain else "alignment_not_observed")
        elif rule == "hierarchy":
            if luna_opposed and luna and luna.event_scale == "major" and leader_opposed:
                abstain = True
                cautions.append("major_sourced_event_and_leaders_opposed")
            elif level_opposed and leader_opposed:
                abstain = True
                cautions.append("local_reversal_and_leaders_opposed")
            elif recent_events and not aligned:
                abstain = True
                cautions.append("event_without_observed_leader_confirmation")
            else:
                if level_same:
                    supports.append("same_side_calculated_area_reaction")
                if aligned:
                    supports.append("independent_leader_aligned")
                if high_volume:
                    supports.append("high_volume_readiness")
                if barrier or book_opposed or luna_opposed or not high_volume:
                    factor = 0.5
                    cautions.append("one_or_more_observed_risk_modifiers")
        elif rule not in {"sieve_only", "volume_size", "leader_gate", "level_reaction_gate",
                          "level_room_size", "event_leader_gate", "book_size", "luna_leader_size",
                          "simple_alignment", "hierarchy"}:
            raise RuntimeError(f"Unknown paper variant rule: {rule}")

        if abstain:
            factor = 0.0
        return {"at_utc": now.astimezone(timezone.utc).isoformat(), "pair": pair,
                "side": side, "entry_tag": tag, "source_id": source_id,
                "variant": self.variant_id, "rule": rule, "rate": rate,
                "supports": supports, "cautions": cautions, "observed": observed,
                "size_factor": factor, "decision": "abstain" if abstain else "enter"}

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if self.spec["rule"] == "hierarchy":
            # This is the one paper hypothesis that tests the existing integrated
            # major-event + leader + local-reversal exit on top of the Sieve exit.
            return super().custom_exit(pair, trade, current_time, current_rate,
                                       current_profit, **kwargs)
        source = self._trade_source(trade)
        source.dp = self.dp
        return source.custom_exit(pair, trade, current_time, current_rate,
                                  current_profit, **kwargs)


class PaperVariant01(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "01"


class PaperVariant02(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "02"


class PaperVariant03(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "03"


class PaperVariant04(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "04"


class PaperVariant05(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "05"


class PaperVariant06(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "06"


class PaperVariant07(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "07"


class PaperVariant08(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "08"


class PaperVariant09(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "09"


class PaperVariant10(_ParkedPaperVariant, IntegratedPaper):
    variant_id = "10"
