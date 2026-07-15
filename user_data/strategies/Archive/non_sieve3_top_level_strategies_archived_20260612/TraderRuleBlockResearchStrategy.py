from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class TraderRuleBlockResearchStrategy(IStrategy):
    """
    Research-only strategy for prebuilt trader-rule signals.

    It does not compute slow custom indicators or read live databases inside
    populate_indicators(). Signals are exported by trader_rule_backtests.py from
    the frozen 1h confluence feature snapshot.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 10
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.02
    use_exit_signal = True
    use_custom_stoploss = False
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trader_rule_signals_sieve_exact_min10_20260605.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_pattern_rectangle_breakdown_short": {"take_profit": 0.050, "hold_hours": 10.0},
        "sieve_prior_day_high_break_vp_long": {"take_profit": 0.020, "hold_hours": 48.0},
        "sieve_prior_day_low_break_vp_short": {"take_profit": 0.020, "hold_hours": 48.0},
        "sieve_pattern_rectangle_breakout_long": {"take_profit": 0.050, "hold_hours": 10.0},
    }

    _signal_cache: DataFrame | None = None

    @classmethod
    def _load_signals(cls) -> DataFrame:
        cached = cls._signal_cache
        if cached is not None:
            return cached
        path = Path(cls.signal_file)
        if not path.exists():
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._signal_cache = empty
            return empty
        frame = pd.read_parquet(path)
        if frame.empty:
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._signal_cache = empty
            return empty
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        keep_columns = [
            "date",
            "enter_long",
            "enter_short",
            "enter_tag",
            "rule_id",
            "concept_id",
            "threshold",
            "signal_score",
            "risk_multiplier",
            "risk_overlay_id",
        ]
        cleaned = frame[[column for column in keep_columns if column in frame.columns]].dropna(subset=["date"]).copy()
        for column in ("enter_long", "enter_short"):
            if column in cleaned.columns:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0).astype("int8")
            else:
                cleaned[column] = 0
        for column in ("enter_tag", "rule_id", "concept_id"):
            if column not in cleaned.columns:
                cleaned[column] = ""
            cleaned[column] = cleaned[column].fillna("").astype(str)
        if "risk_overlay_id" not in cleaned.columns:
            cleaned["risk_overlay_id"] = ""
        cleaned["risk_overlay_id"] = cleaned["risk_overlay_id"].fillna("").astype(str)
        for column in ("threshold", "signal_score"):
            if column in cleaned.columns:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            else:
                cleaned[column] = pd.NA
        if "risk_multiplier" in cleaned.columns:
            cleaned["risk_multiplier"] = pd.to_numeric(cleaned["risk_multiplier"], errors="coerce").fillna(1.0).clip(0.25, 1.75)
        else:
            cleaned["risk_multiplier"] = 1.0
        cleaned = cleaned.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
        cls._signal_cache = cleaned
        return cleaned

    @staticmethod
    def _normalise_date(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, utc=True, errors="coerce")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        frame["date"] = self._normalise_date(frame["date"])
        signals = self._load_signals()
        if signals.empty:
            frame["trader_rule_signal_present"] = 0
            frame["trader_rule_signal_score"] = pd.NA
            frame["trader_rule_id"] = ""
            return frame
        merged = frame.merge(
            signals.rename(
                columns={
                    "enter_long": "trader_rule_enter_long",
                    "enter_short": "trader_rule_enter_short",
                    "enter_tag": "trader_rule_enter_tag",
                    "rule_id": "trader_rule_id",
                    "concept_id": "trader_rule_concept_id",
                    "threshold": "trader_rule_threshold",
                    "signal_score": "trader_rule_signal_score",
                    "risk_multiplier": "trader_rule_risk_multiplier",
                    "risk_overlay_id": "trader_rule_risk_overlay_id",
                }
            ),
            on="date",
            how="left",
        )
        merged["trader_rule_enter_long"] = pd.to_numeric(merged["trader_rule_enter_long"], errors="coerce").fillna(0).astype("int8")
        merged["trader_rule_enter_short"] = pd.to_numeric(merged["trader_rule_enter_short"], errors="coerce").fillna(0).astype("int8")
        merged["trader_rule_signal_present"] = (
            merged["trader_rule_enter_long"].gt(0) | merged["trader_rule_enter_short"].gt(0)
        ).astype("int8")
        for column in ("trader_rule_enter_tag", "trader_rule_id", "trader_rule_concept_id"):
            merged[column] = merged[column].fillna("").astype(str)
        merged["trader_rule_risk_multiplier"] = pd.to_numeric(
            merged.get("trader_rule_risk_multiplier", 1.0), errors="coerce"
        ).fillna(1.0).clip(0.25, 1.75)
        merged["trader_rule_risk_overlay_id"] = merged.get("trader_rule_risk_overlay_id", "").fillna("").astype(str)
        return merged

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        long_signal = dataframe.get("trader_rule_enter_long", 0).astype("int8").gt(0)
        short_signal = dataframe.get("trader_rule_enter_short", 0).astype("int8").gt(0)
        valid = dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[long_signal & valid, "enter_long"] = 1
        dataframe.loc[short_signal & valid, "enter_short"] = 1
        dataframe.loc[(long_signal | short_signal) & valid, "enter_tag"] = dataframe.loc[
            (long_signal | short_signal) & valid, "trader_rule_enter_tag"
        ]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @classmethod
    def _risk_multiplier_for_entry_time(cls, current_time: datetime) -> float:
        if current_time.tzinfo is None:
            entry_time = current_time.replace(tzinfo=timezone.utc)
        else:
            entry_time = current_time.astimezone(timezone.utc)
        signal_time = entry_time - timedelta(hours=1)
        signals = cls._load_signals()
        if signals.empty or "risk_multiplier" not in signals.columns:
            return 1.0
        by_date = signals.set_index("date")["risk_multiplier"]
        value = by_date.get(signal_time)
        if value is None:
            value = by_date.get(entry_time)
        try:
            multiplier = float(value)
        except (TypeError, ValueError):
            return 1.0
        if multiplier != multiplier:
            return 1.0
        return max(0.25, min(1.75, multiplier))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_rate, leverage, entry_tag, side, kwargs
        multiplier = self._risk_multiplier_for_entry_time(current_time)
        stake = proposed_stake * multiplier
        if min_stake is not None:
            stake = max(float(min_stake), stake)
        return min(float(max_stake), stake)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = pair, current_rate, kwargs
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        settings = self.rule_exit_settings.get(rule_id)
        if settings is None:
            return None
        take_profit = float(settings.get("take_profit", 0.02))
        if current_profit >= take_profit:
            return f"{rule_id}_take_profit"
        open_date = getattr(trade, "open_date_utc", None)
        if open_date is None:
            return None
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        elapsed_hours = (current_time - open_date).total_seconds() / 3600.0
        if elapsed_hours >= float(settings.get("hold_hours", 48.0)):
            return f"{rule_id}_time_exit"
        return None
