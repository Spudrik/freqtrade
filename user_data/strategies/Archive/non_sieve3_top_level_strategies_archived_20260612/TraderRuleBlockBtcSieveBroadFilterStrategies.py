from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class _BtcSieveSelectedExitCsvBase(TraderRuleBlockResearchStrategy):
    """
    Research-only base for BTC sieve confluence branches.

    Signals are pre-filtered to rule families that have selected exits. Exit
    settings are loaded from the matching exit-research CSV so repeated branch
    validation does not require hand-copying large dictionaries.
    """

    can_short = True
    use_custom_stoploss = True
    stoploss = -0.035
    exit_settings_file = ""
    _exit_settings_cache: dict[str, dict[str, float]] | None = None

    @classmethod
    def _load_exit_settings(cls) -> dict[str, dict[str, float]]:
        cached = cls._exit_settings_cache
        if cached is not None:
            return cached
        path = Path(cls.exit_settings_file)
        if not path.exists():
            cls._exit_settings_cache = {}
            return {}
        frame = pd.read_csv(path)
        required = {"rule_id", "take_profit", "hold_hours", "stop_loss"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Exit settings file {path} missing columns: {sorted(missing)}")
        settings: dict[str, dict[str, float]] = {}
        for row in frame.itertuples(index=False):
            rule_id = str(getattr(row, "rule_id"))
            settings[rule_id] = {
                "take_profit": float(getattr(row, "take_profit")),
                "hold_hours": float(getattr(row, "hold_hours")),
                "stop_loss": float(getattr(row, "stop_loss")),
            }
        cls._exit_settings_cache = settings
        return settings

    @classmethod
    def _settings_for_trade(cls, trade: Any) -> tuple[str, dict[str, float] | None]:
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        return rule_id, cls._load_exit_settings().get(rule_id)

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        _ = pair, current_time, current_rate, after_fill, kwargs
        _, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        return stoploss_from_open(
            -float(settings["stop_loss"]),
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

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
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        open_date = getattr(trade, "open_date_utc", None)
        if open_date is None:
            return None
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        elapsed_hours = (current_time - open_date).total_seconds() / 3600.0
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveRegimeRangeStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    If BTC sieve entries only fire when regime and range-break context agree,
    do tailored exits create a useful multi-year BTC block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_top30_v2_regime_plus_range_break_selected_exit_rules.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_regime_range_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveRangeBreakAgreesStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    If BTC sieve entries only fire when the range-break direction agrees with
    the entry story, do tailored exits produce a better block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_top30_v2_range_break_agrees_selected_exit_rules.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_range_break_agrees_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveVpRangeStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    If BTC sieve entries only fire when VP context and range-break behaviour
    agree, do tailored exits produce the best broad branch?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_top30_v2_vp_plus_range_break_selected_exit_rules.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_vp_range_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveCompressionRangeQualityFamiliesStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    If the compression/range branch only keeps rule families that contributed
    positively in the first Freqtrade validation, does quality improve without
    killing trade count?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveCompletePatternRareEntriesStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Do rare complete-pattern sieve entries with tailored per-family exits add a
    high-win-rate lead family worth later merging into the BTC rule block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trader_sieve_runtime_signal_exports_20260606_complete_pattern_rare_entries_signals.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_complete_pattern_rare_entries_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityStoryRiskStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Keep the same quality-family entries and tailored exits, but size each
    trade according to the specific trader story behind the rule.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_quality_story_risk.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityStoryObGuardRiskStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Keep the same quality-family entries and tailored exits, size by trader
    story, and reduce size when the orderbook strongly argues against the
    original idea.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_quality_story_ob_guard_risk.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityOrderbookCrashExitStrategy(TraderRuleBlockBtcSieveCompressionRangeQualityFamiliesStrategy):
    """
    Trader question:
    Keep the same quality-family entries and tailored exits, but exit long
    trades early when the orderbook/price state looks like downside escalation.
    """

    feature_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache"
        r"\trader_confluence_1h_latest.parquet"
    )
    _feature_cache: DataFrame | None = None
    orderbook_crash_exit_rule_ids: set[str] | None = None

    @classmethod
    def _load_exit_features(cls) -> DataFrame:
        cached = cls._feature_cache
        if cached is not None:
            return cached
        path = Path(cls.feature_file)
        if not path.exists():
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._feature_cache = empty
            return empty
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        keep = [
            "date",
            "conf_ob_bearish_pressure_agreement",
            "conf_ob_support_removed_strength",
            "conf_ob_support_cleared",
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_ob_downside_vacuum",
            "conf_ob_spread_fragility",
            "conf_ob_single_venue_extreme",
            "conf_ob_bid_absorption",
            "conf_ob_ask_absorption",
            "conf_ob_breakout_failure",
            "conf_ob_breakdown_failure",
            "px_close_breakdown_6h",
            "px_close_breakdown_24h",
            "conf_volume_bullish_confirmation",
            "conf_volume_bearish_confirmation",
            "px_volume_pressure_6h",
            "px_volume_pressure_24h",
            "conf_structure_breakout_trigger_score",
            "conf_structure_breakdown_trigger_score",
            "conf_structure_bullish_state_score",
            "conf_structure_bearish_state_score",
            "st_1h_ms_bos_to_bull",
            "st_1h_ms_bos_to_bear",
            "st_4h_ms_bos_to_bull",
            "st_4h_ms_bos_to_bear",
            "st_1h_vp_above_value_area",
            "st_1h_vp_below_value_area",
            "st_4h_vp_above_value_area",
            "st_4h_vp_below_value_area",
            "st_1d_vp_above_value_area",
            "st_1d_vp_below_value_area",
            "st_failed_breakout_structure_risk",
            "st_failed_breakdown_structure_risk",
            "st_breakout_structure_setup",
            "st_breakdown_structure_setup",
            "conf_failed_breakout_exhaustion_score",
            "conf_failed_breakdown_exhaustion_score",
            "px_close_breakout_6h",
            "px_close_breakout_24h",
            "px_return_1h",
            "px_range_1h",
            "px_volume_z_6h",
            "px_volume_z_24h",
        ]
        cleaned = frame[[column for column in keep if column in frame.columns]].dropna(subset=["date"]).copy()
        for column in cleaned.columns:
            if column != "date":
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)
        cleaned = cleaned.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        cls._feature_cache = cleaned
        return cleaned

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        merged = super().populate_indicators(dataframe, metadata)
        features = self._load_exit_features()
        if features.empty:
            return merged
        merged["date"] = pd.to_datetime(merged["date"], utc=True, errors="coerce")
        return merged.merge(features, on="date", how="left")

    @staticmethod
    def _value_from_row(row: Any, columns: tuple[str, ...]) -> float:
        values: list[float] = []
        for column in columns:
            try:
                value = float(row.get(column, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if value == value:
                values.append(value)
        return max(values) if values else 0.0

    def _orderbook_crash_exit_active(self, pair: str, current_time: datetime, trade: Any) -> bool:
        if trade.is_short:
            return False
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        allowed_rules = self.orderbook_crash_exit_rule_ids
        if allowed_rules is not None and rule_id not in allowed_rules:
            return False
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return False
        current = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        rows = dataframe.loc[dates.eq(current)]
        if rows.empty:
            rows = dataframe.loc[dates.le(current)].tail(1)
        if rows.empty:
            return False
        row = rows.iloc[-1]
        crash_score = 0
        if self._value_from_row(row, ("conf_ob_bearish_pressure_agreement",)) >= 0.35:
            crash_score += 1
        if self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared")) >= 0.35:
            crash_score += 1
        if self._value_from_row(row, ("conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum")) >= 0.35:
            crash_score += 1
        if self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme")) >= 0.50:
            crash_score += 1
        if self._value_from_row(row, ("px_close_breakdown_6h", "px_close_breakdown_24h")) >= 0.50:
            crash_score += 1
        return crash_score >= 2

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        if self._orderbook_crash_exit_active(pair, current_time, trade):
            return f"{rule_id}_orderbook_crash_exit"
        open_date = getattr(trade, "open_date_utc", None)
        if open_date is None:
            return None
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        elapsed_hours = (current_time - open_date).total_seconds() / 3600.0
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveQualitySelectiveOrderbookCrashExitStrategy(TraderRuleBlockBtcSieveQualityOrderbookCrashExitStrategy):
    """
    Trader question:
    Apply the orderbook crash exit only to rule families where the first
    Freqtrade family audit showed it helped, instead of using it as a broad
    long-trade exit.
    """

    orderbook_crash_exit_rule_ids = {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve1_tlv2_resistance_breakout_long_4h",
        "sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h",
        "sieve1_vp_lvn_fast_traverse_long_1h",
    }


class TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveOrderbookCrashExitStrategy
):
    """
    Trader question:
    Keep the best selective orderbook crash exit, then add only the volume,
    structure, volatility, and stale-trade exits for rule families where the
    direct trade-management audit suggested they may help.
    """

    volume_failure_exit_rule_ids = {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve1_vp_lvn_fast_traverse_long_1h",
        "sieve2_reframed_vp_poc_reject_short_1h",
        "sieve1_tlv2_resistance_breakout_long_4h",
    }
    structure_failure_exit_rule_ids = {
        "sieve1_vp_lvn_fast_traverse_long_1h",
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h",
        "sieve2_reframed_vp_poc_reject_short_1h",
    }
    volatility_shock_exit_rule_ids = {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve1_vp_lvn_fast_traverse_long_1h",
    }
    time_to_confirm_exit_rule_ids = {
        "sieve2_reframed_vp_poc_reject_short_1h",
        "sieve1_vp_lvn_fast_traverse_long_1h",
        "sieve1_tlv2_resistance_breakout_long_4h",
    }

    @staticmethod
    def _min_elapsed_hours(current_time: datetime, trade: Any) -> float:
        open_date = getattr(trade, "open_date_utc", None)
        if open_date is None:
            return 0.0
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        current = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        return max((current - open_date).total_seconds() / 3600.0, 0.0)

    def _current_feature_row(self, pair: str, current_time: datetime) -> Any | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        current = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        rows = dataframe.loc[dates.eq(current)]
        if rows.empty:
            rows = dataframe.loc[dates.le(current)].tail(1)
        if rows.empty:
            return None
        return rows.iloc[-1]

    def _volume_failure_exit_active(self, row: Any, trade: Any, rule_id: str, elapsed_hours: float, current_profit: float) -> bool:
        if rule_id not in self.volume_failure_exit_rule_ids or elapsed_hours < 3.0 or current_profit > 0.003:
            return False
        bull = self._value_from_row(row, ("conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"))
        bear = self._value_from_row(row, ("conf_volume_bearish_confirmation",))
        pressure_6h = self._value_from_row(row, ("px_volume_pressure_6h",))
        pressure_24h = self._value_from_row(row, ("px_volume_pressure_24h",))
        bearish_pressure = max(bear, -pressure_6h, -pressure_24h)
        if trade.is_short:
            return bearish_pressure < 0.20 or bull >= 0.45
        return bull < 0.20 or bearish_pressure >= 0.45

    def _structure_failure_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        if rule_id not in self.structure_failure_exit_rule_ids:
            return False
        if trade.is_short:
            return (
                self._value_from_row(
                    row,
                    (
                        "conf_structure_breakout_trigger_score",
                        "conf_structure_bullish_state_score",
                        "st_1h_ms_bos_to_bull",
                        "st_4h_ms_bos_to_bull",
                    ),
                )
                >= 0.45
                or self._value_from_row(row, ("px_close_breakout_6h", "st_1h_vp_above_value_area", "st_4h_vp_above_value_area"))
                >= 0.60
            )
        return (
            self._value_from_row(
                row,
                (
                    "conf_structure_breakdown_trigger_score",
                    "conf_structure_bearish_state_score",
                    "st_1h_ms_bos_to_bear",
                    "st_4h_ms_bos_to_bear",
                ),
            )
            >= 0.45
            or self._value_from_row(row, ("px_close_breakdown_6h", "st_1h_vp_below_value_area", "st_4h_vp_below_value_area"))
            >= 0.60
        )

    def _volatility_shock_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        if rule_id not in self.volatility_shock_exit_rule_ids:
            return False
        ret = self._value_from_row(row, ("px_return_1h",))
        range_hot = self._value_from_row(row, ("px_range_1h", "px_volume_z_6h", "px_volume_z_24h")) >= 0.75
        if trade.is_short:
            return range_hot and ret > 0.008
        return range_hot and ret < -0.008

    def _time_to_confirm_exit_active(self, row: Any, trade: Any, rule_id: str, elapsed_hours: float, current_profit: float) -> bool:
        if rule_id not in self.time_to_confirm_exit_rule_ids or elapsed_hours < 6.0 or current_profit > 0.002:
            return False
        if trade.is_short:
            confirm = self._value_from_row(
                row,
                ("conf_volume_bearish_confirmation", "conf_structure_bearish_state_score", "px_close_breakdown_6h"),
            )
            return confirm < 0.45
        confirm = self._value_from_row(
            row,
            ("conf_volume_bullish_confirmation", "conf_structure_bullish_state_score", "px_close_breakout_6h"),
        )
        return confirm < 0.45

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        if self._orderbook_crash_exit_active(pair, current_time, trade):
            return f"{rule_id}_orderbook_crash_exit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if self._volume_failure_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                return f"{rule_id}_volume_failure_exit"
            if self._structure_failure_exit_active(row, trade, rule_id):
                return f"{rule_id}_structure_failure_exit"
            if self._volatility_shock_exit_active(row, trade, rule_id):
                return f"{rule_id}_volatility_shock_exit"
            if self._time_to_confirm_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                return f"{rule_id}_time_to_confirm_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveQualitySelectiveObVolumeExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the current best selective orderbook exit improve further if only
    family-scoped volume-failure exits are added?
    """

    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualitySelectiveObStructureExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the current best selective orderbook exit improve further if only
    family-scoped structure-failure exits are added?
    """

    volume_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualitySelectiveObVolatilityExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the current best selective orderbook exit improve further if only
    family-scoped volatility-shock exits are added?
    """

    volume_failure_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the current best selective orderbook exit improve further if only
    family-scoped time-to-confirm exits are added?
    """

    volume_failure_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualitySelectiveObTimeStructureExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the best time-to-confirm layer combine with family-scoped structure
    failure to reduce drawdown without giving back too much return?
    """

    volume_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualitySelectiveObTimeVolumeExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the best time-to-confirm layer combine with family-scoped volume
    failure, or does volume cut too many useful continuation trades?
    """

    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()


class TraderRuleBlockBtcSieveQualityPlusCompletePatternRareEntriesStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy
):
    """
    Trader question:
    If we add rare complete-pattern entries to the current best BTC quality
    block, do they add useful trades without damaging the main block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv"
    )


class TraderRuleBlockBtcSieveCompletePatternRareCompressionRangeStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Do rare complete-pattern entries still work if we only keep the ones that
    fired during compression plus range-break context?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_complete_pattern_rare_entries_compression_range.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_complete_pattern_rare_entries_exit_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityPlusRareCompressionRangePriorityStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy
):
    """
    Trader question:
    If rare complete-pattern entries only fire during compression plus range
    break, and those rare flags take priority over same-hour quality entries,
    does the merged block keep the return boost with less drawdown?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_compression_range_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy
):
    """
    Trader question:
    If rare complete-pattern entries only need range-break agreement and take
    priority over same-hour quality entries, does the merged block improve?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_range_break_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv"
    )


class TraderRuleBlockBtcSieveQualityPlusRareAllPriorityStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy
):
    """
    Trader question:
    If all rare complete-pattern entries take priority over same-hour quality
    entries, is the merge better than the previous quality-priority merge?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_all_rare_priority_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv"
    )


_COMPLETE_PATTERN_RARE_RULE_IDS = {
    "sieve2_geometry_descending_channel_lower_breakdown_short_1h",
    "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h",
    "sieve2_geometry_wedge_breakout_long_1h",
    "sieve2_geometry_triangle_squeeze_breakout_long_1h",
}


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareTimeExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy
):
    """
    Trader question:
    If range-break rare entries are added to the quality block, does a
    rare-family time-to-confirm exit reduce drawdown without killing the return
    boost?
    """

    time_to_confirm_exit_rule_ids = (
        TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy.time_to_confirm_exit_rule_ids
        | _COMPLETE_PATTERN_RARE_RULE_IDS
    )


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareStructureExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy
):
    """
    Trader question:
    If range-break rare entries are added to the quality block, does a
    rare-family structure-failure exit catch failed rare breakouts/breakdowns?
    """

    structure_failure_exit_rule_ids = (
        TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy.structure_failure_exit_rule_ids
        | _COMPLETE_PATTERN_RARE_RULE_IDS
    )


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareTimeStructureExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy
):
    """
    Trader question:
    Do rare-family time-to-confirm and structure-failure exits work together,
    or does the combined exit stack cut too many valid rare trades?
    """

    time_to_confirm_exit_rule_ids = (
        TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy.time_to_confirm_exit_rule_ids
        | _COMPLETE_PATTERN_RARE_RULE_IDS
    )
    structure_failure_exit_rule_ids = (
        TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy.structure_failure_exit_rule_ids
        | _COMPLETE_PATTERN_RARE_RULE_IDS
    )


class TraderRuleBlockBtcSieveRareCompressionRangeRareTimeExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    For the rare compression/range block by itself, does time-to-confirm help
    remove failed rare entries while keeping the high-win-rate profile?
    """

    volume_failure_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids = _COMPLETE_PATTERN_RARE_RULE_IDS
    orderbook_crash_exit_rule_ids: set[str] = set()
    signal_file = TraderRuleBlockBtcSieveCompletePatternRareCompressionRangeStrategy.signal_file
    exit_settings_file = TraderRuleBlockBtcSieveCompletePatternRareCompressionRangeStrategy.exit_settings_file


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoSundayRareStructureExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareStructureExitStrategy
):
    """
    Trader question:
    The best merged candidate is strongly negative on Sunday in the audit. Does
    skipping Sunday entries reduce drawdown without removing too much edge?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_range_break_rare_priority_no_sunday.parquet"
    )


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoWeakFamilyRareStructureExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareStructureExitStrategy
):
    """
    Trader question:
    Does removing the one weak family from the best merged candidate improve
    the system, or is its occasional upside still worth keeping?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_range_break_rare_priority_no_weak_family.parquet"
    )


class TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoSundayNoWeakFamilyRareStructureExitStrategy(
    TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareStructureExitStrategy
):
    """
    Trader question:
    If both obvious audit weaknesses are removed, does the cleaner block beat
    the best merged candidate or become too selective?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_range_break_rare_priority_no_sunday_no_weak_family.parquet"
    )


class TraderRuleBlockBtcSieveExpansionPack1BestPerRuleStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Can the next batch of independent BTC sieve lead families work after
    per-rule confluence filtering and tailored exit selection?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_expansion_pack1_best_per_rule_selected_confluence.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_expansion_pack1_best_per_rule_selected.csv"
    )


class TraderRuleBlockBtcSieveExpansionPack1NoCatastrophicFamilyStrategy(
    TraderRuleBlockBtcSieveExpansionPack1BestPerRuleStrategy
):
    """
    Trader question:
    Does expansion pack 1 become usable if the single catastrophic family found
    in Freqtrade validation is removed?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_expansion_pack1_best_per_rule_no_catastrophic_family.parquet"
    )


class TraderRuleBlockBtcSieveExpansionPack1PositiveFamiliesStrategy(
    TraderRuleBlockBtcSieveExpansionPack1BestPerRuleStrategy
):
    """
    Trader question:
    Does expansion pack 1 become a clean add-on if only the Freqtrade-positive
    families are retained?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_expansion_pack1_best_per_rule_positive_families.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1CurrentPriorityStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    If the current best BTC block and expansion pack 1 both signal, should the
    current best block keep priority on same-hour conflicts?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_current_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_current_best_plus_expansion_pack1_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1ExpansionPriorityStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1CurrentPriorityStrategy
):
    """
    Trader question:
    If the current best BTC block and expansion pack 1 both signal, should the
    expansion pack take priority on same-hour conflicts?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoWorstFamilyStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1CurrentPriorityStrategy
):
    """
    Trader question:
    Does removing only the worst family from the high-return merged block cut
    drawdown without losing too much edge?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority_no_worst_family.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1CurrentPriorityStrategy
):
    """
    Trader question:
    Do the two weakest families hurt the high-return merged block enough that
    removing both improves the tradeoff?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority_no_two_weak_families.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoThreeWeakFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1CurrentPriorityStrategy
):
    """
    Trader question:
    Does removing all three negative families make the merged block cleaner, or
    does it discard useful occasional winners?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority_no_three_weak_families.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesTradeManagementStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Does the existing selective orderbook/structure/volume/time invalidation
    layer improve the new best merged block, or does it cut good trades?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority_no_two_weak_families.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_current_best_plus_expansion_pack1_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObTimeStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy
):
    """
    Trader question:
    Does the previously best selective orderbook plus time-to-confirm layer
    still help after the larger merged lead set is cleaned?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack1_expansion_priority_no_two_weak_families.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_current_best_plus_expansion_pack1_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObStructureStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveObStructureExitStrategy
):
    """
    Trader question:
    Does selective orderbook plus structure-failure invalidation reduce the
    remaining stopped trades in the cleaned merged block?
    """

    signal_file = TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObTimeStrategy.signal_file
    exit_settings_file = TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObTimeStrategy.exit_settings_file


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesShortStructureExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Do the short families that caused the latest drawdown window need an early
    exit when market structure flips bullish against them?
    """

    signal_file = TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObTimeStrategy.signal_file
    exit_settings_file = TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesObTimeStrategy.exit_settings_file
    orderbook_crash_exit_rule_ids: set[str] = set()
    volume_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids = {
        "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
        "sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h",
        "sieve2_reversal_double_top_present_short_1h",
    }


class TraderRuleBlockBtcSieveExpansionPack2BalancedBestPerRuleStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Does a second balanced pack of unused sieve-derived BTC lead families work
    after per-rule confluence filtering and tailored exit selection?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_expansion_pack2_balanced_best_per_rule_selected_confluence.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_expansion_pack2_balanced_best_per_rule_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy(_BtcSieveSelectedExitCsvBase):
    """
    Trader question:
    Does expansion pack 2 improve the current best block when the current best
    keeps priority on same-hour conflicts?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack2_current_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260606_btc_current_best_plus_expansion_pack2_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2Pack2PriorityStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy
):
    """
    Trader question:
    Does expansion pack 2 improve the current best block when pack 2 takes
    priority on same-hour conflicts?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_expansion_pack2_pack2_priority.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusPack2NoWorstOverallFamilyStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy
):
    """
    Trader question:
    Does removing only the worst net family from the pack2 merged candidate
    improve the risk/reward tradeoff?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_pack2_no_worst_overall_family.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy
):
    """
    Trader question:
    Does removing the three weakest net families clean up pack2 without cutting
    too much useful coverage?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_pack2_no_top3_overall_weak_families.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusPack2NoPack2WeakFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy
):
    """
    Trader question:
    Do the pack2-added families that were weak after merging need to be removed
    before this becomes a production-alpha candidate?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_pack2_no_pack2_weak_families.parquet"
    )


class TraderRuleBlockBtcSieveCurrentBestPlusPack2NoDrawdownLossFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy
):
    """
    Trader question:
    If the late-2025 drawdown families are removed, does drawdown fall enough
    to justify losing otherwise useful entries?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_current_best_plus_pack2_no_drawdown_loss_families.parquet"
    )


_PACK2_CURRENT_BEST_SIGNAL_FILE = (
    r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
    r"\trading_lead_signals_20260606_btc_current_best_plus_pack2_no_top3_overall_weak_families.parquet"
)
_PACK2_CURRENT_BEST_EXIT_FILE = (
    r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
    r"\trading_lead_exit_research_20260606_btc_current_best_plus_expansion_pack2_selected.csv"
)

_PACK2_SHORT_FAILURE_RULE_IDS = {
    "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
    "sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h",
    "sieve2_reversal_double_top_present_short_1h",
}

_PACK2_LONG_FAILURE_RULE_IDS = {
    "sieve1_ladder_long_sup_hold",
    "sieve1_multi2_prior_vp_breakout_long",
    "sieve1_vp_lvn_fast_traverse_long_1h",
    "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h",
    "sieve2_multi2_vp_prior_week_high_break_vp_val_long",
    "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
}

_PACK2_LONG_ACCEPTANCE_FAILURE_RULE_IDS = _PACK2_LONG_FAILURE_RULE_IDS | {
    "sieve1_bos_bull_continuation_long_1d",
    "sieve1_triple_bottom_confirmed_long_1h",
    "sieve2_bos_bull_continuation_long_1h",
    "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_1h",
    "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h",
    "sieve2_multi2_tlv2_vp_res_break_vp_val_long_1d",
    "sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard",
}

_PACK2_LONG_REPLAY_OB_STRESS_RULE_IDS = _PACK2_LONG_FAILURE_RULE_IDS | {
    "sieve1_bos_bull_continuation_long_1d",
    "sieve1_continuation_flag_present_long_4h",
    "sieve1_geometry_triangle_squeeze_breakout_long_1h",
    "sieve2_bos_bull_continuation_long_1h",
    "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h",
    "sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h",
    "sieve2_multi2_vp_prior_month_high_break_vp_node_long",
    "sieve2_overtrade_multi2_tlv2_vp_res_break_vp_val_long_1h_vp_market_guard",
    "sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard",
}


class TraderRuleBlockBtcSieveCurrentBestPack2ShortStructureFailureExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    The current best still has large short losses when support-break/rejection
    ideas fail. Do only those short families need an early exit when structure
    flips bullish or price accepts above value again?
    """

    signal_file = _PACK2_CURRENT_BEST_SIGNAL_FILE
    exit_settings_file = _PACK2_CURRENT_BEST_EXIT_FILE
    orderbook_crash_exit_rule_ids: set[str] = set()
    volume_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS


class TraderRuleBlockBtcSieveCurrentBestPack2LongVolumeFailureExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    The current best still has large long losses after VP/resistance-break and
    support-hold ideas fail. Do only those long families need a volume/time
    confirmation exit when the move does not prove itself?
    """

    signal_file = _PACK2_CURRENT_BEST_SIGNAL_FILE
    exit_settings_file = _PACK2_CURRENT_BEST_EXIT_FILE
    orderbook_crash_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    volume_failure_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS
    time_to_confirm_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS


class TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    For vulnerable long families in the current best, can orderbook downside
    escalation act as a crash/risk exit without filtering entries too much?
    """

    signal_file = _PACK2_CURRENT_BEST_SIGNAL_FILE
    exit_settings_file = _PACK2_CURRENT_BEST_EXIT_FILE
    volume_failure_exit_rule_ids: set[str] = set()
    structure_failure_exit_rule_ids: set[str] = set()
    volatility_shock_exit_rule_ids: set[str] = set()
    time_to_confirm_exit_rule_ids: set[str] = set()
    orderbook_crash_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS


class TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    Keep the current best long orderbook crash exit, but only exit vulnerable
    shorts when the market has actually reclaimed upward, instead of reacting
    to any broad bullish structure hint.
    """

    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS

    def _structure_failure_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        if not trade.is_short or rule_id not in self.structure_failure_exit_rule_ids:
            return False
        bullish_break = self._value_from_row(
            row,
            ("conf_structure_breakout_trigger_score", "px_close_breakout_6h", "px_close_breakout_24h"),
        )
        bullish_state = self._value_from_row(
            row,
            ("conf_structure_bullish_state_score", "st_1h_ms_bos_to_bull", "st_4h_ms_bos_to_bull"),
        )
        value_reclaim = self._value_from_row(row, ("st_1h_vp_above_value_area", "st_4h_vp_above_value_area"))
        volume_agrees = self._value_from_row(
            row,
            ("conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"),
        )
        return (bullish_break >= 0.65 and value_reclaim >= 0.50) or (
            bullish_break >= 0.55 and bullish_state >= 0.55 and volume_agrees >= 0.30
        )


class TraderRuleBlockBtcSieveCurrentBestReplayExpandedOrderbookStressExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    The worst-long replay showed most remaining large long losses had
    orderbook stress after entry. Does expanding the current best orderbook
    exit only to those observed stress-loss families improve the block?
    """

    orderbook_crash_exit_rule_ids = _PACK2_LONG_REPLAY_OB_STRESS_RULE_IDS


class TraderRuleBlockBtcSieveCurrentBestReplayExpandedObStressStrictShortStrategy(
    TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy
):
    """
    Trader question:
    Does the replay-expanded long orderbook stress exit combine with the
    lower-drawdown strict short reclaim branch?
    """

    orderbook_crash_exit_rule_ids = _PACK2_LONG_REPLAY_OB_STRESS_RULE_IDS


class TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    Does orderbook stress become useful for more long families only when the
    trade is already losing and price/structure support is failing?
    """

    orderbook_crash_exit_rule_ids = _PACK2_LONG_REPLAY_OB_STRESS_RULE_IDS

    def _orderbook_crash_exit_active(self, pair: str, current_time: datetime, trade: Any) -> bool:
        return False

    def _long_ob_stress_invalidation_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id not in self.orderbook_crash_exit_rule_ids:
            return False
        if elapsed_hours < 1.0 or current_profit > -0.004:
            return False
        ob_stress = 0
        if self._value_from_row(row, ("conf_ob_bearish_pressure_agreement",)) >= 0.35:
            ob_stress += 1
        if self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared")) >= 0.35:
            ob_stress += 1
        if self._value_from_row(row, ("conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum")) >= 0.35:
            ob_stress += 1
        if self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme")) >= 0.50:
            ob_stress += 1
        if ob_stress < 2:
            return False
        failed_support = self._value_from_row(
            row,
            (
                "conf_structure_breakdown_trigger_score",
                "px_close_breakdown_6h",
                "px_close_breakdown_24h",
                "st_1h_vp_below_value_area",
                "st_4h_vp_below_value_area",
            ),
        )
        bearish_state = self._value_from_row(
            row,
            ("conf_structure_bearish_state_score", "st_1h_ms_bos_to_bear", "st_4h_ms_bos_to_bear"),
        )
        return failed_support >= 0.50 or bearish_state >= 0.60

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None and self._long_ob_stress_invalidation_exit_active(
            row, trade, rule_id, elapsed_hours, current_profit
        ):
            return f"{rule_id}_ob_stress_invalidation_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationStrictShortStrategy(
    TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationExitStrategy
):
    """
    Trader question:
    Does tighter long orderbook-stress invalidation combine with the best
    lower-drawdown strict short reclaim logic?
    """

    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS

    def _structure_failure_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        return TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
            self, row, trade, rule_id
        )

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if self._long_ob_stress_invalidation_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                return f"{rule_id}_ob_stress_invalidation_exit"
            if self._structure_failure_exit_active(row, trade, rule_id):
                return f"{rule_id}_structure_failure_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestObExitLongAcceptanceFailureStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    Keep the current best long orderbook crash exit, but exit vulnerable long
    continuation/reclaim families only when price loses acceptance and downside
    structure confirms the original long story has failed.
    """

    structure_failure_exit_rule_ids = _PACK2_LONG_ACCEPTANCE_FAILURE_RULE_IDS

    def _structure_failure_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        if trade.is_short or rule_id not in self.structure_failure_exit_rule_ids:
            return False
        bearish_break = self._value_from_row(
            row,
            ("conf_structure_breakdown_trigger_score", "px_close_breakdown_6h", "px_close_breakdown_24h"),
        )
        bearish_state = self._value_from_row(
            row,
            ("conf_structure_bearish_state_score", "st_1h_ms_bos_to_bear", "st_4h_ms_bos_to_bear"),
        )
        value_loss = self._value_from_row(row, ("st_1h_vp_below_value_area", "st_4h_vp_below_value_area"))
        bearish_volume = self._value_from_row(row, ("conf_volume_bearish_confirmation",))
        return (bearish_break >= 0.65 and value_loss >= 0.50) or (
            bearish_break >= 0.55 and bearish_state >= 0.55 and bearish_volume >= 0.30
        )


class TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    The family diagnostic showed `sieve2_bos_bull_continuation_long_1h` was
    net negative in the current best. Does removing only that family improve
    the block, or was it still needed for broader return?
    """

    removed_rule_ids = {"sieve2_bos_bull_continuation_long_1h"}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        populated = super().populate_entry_trend(dataframe, metadata)
        tags = populated["enter_tag"].fillna("").astype(str).str.split().str[0]
        remove = tags.isin(self.removed_rule_ids)
        populated.loc[remove, ["enter_long", "enter_short"]] = 0
        populated.loc[remove, "enter_tag"] = None
        return populated


class TraderRuleBlockBtcSieveCurrentBestFamilySpecificLongExitV1Strategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    Do long-family-specific failure clues from the replay diagnostic cut
    losers without damaging the current-best return branch?
    """

    def _family_specific_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or elapsed_hours < 1.0:
            return False
        support_removed = self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared"))
        support_cleared = self._value_from_row(row, ("conf_ob_support_cleared",))
        spread_fragility = self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme"))
        downside_vacuum = self._value_from_row(
            row, ("conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum")
        )
        local_breakdown = self._value_from_row(row, ("px_close_breakdown_6h",))
        failed_breakout = self._value_from_row(
            row, ("st_failed_breakout_structure_risk", "conf_ob_breakout_failure", "conf_failed_breakout_exhaustion_score")
        )
        failed_structure = self._value_from_row(
            row, ("st_failed_breakdown_structure_risk", "st_failed_breakout_structure_risk")
        )
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        if rule_id == "sieve2_bos_bull_continuation_long_1h":
            return current_profit < -0.002 and local_breakdown >= 0.50
        if rule_id == "sieve1_geometry_triangle_squeeze_breakout_long_1h":
            return current_profit < 0.004 and (support_cleared >= 0.35 or failed_breakout >= 0.45)
        if rule_id == "sieve1_ladder_long_sup_hold":
            return current_profit < 0.003 and failed_structure >= 0.45 and support_cleared >= 0.25
        if rule_id == "sieve1_vp_lvn_fast_traverse_long_1h":
            stress_count = int(support_removed >= 0.35) + int(downside_vacuum >= 0.35) + int(spread_fragility >= 0.50)
            return current_profit < 0.005 and stress_count >= 2
        if rule_id == "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard":
            return current_profit < 0.003 and support_removed >= 0.35 and spread_fragility >= 0.50
        if rule_id == "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h":
            return current_profit < 0.003 and ((bid_absorption >= 0.35 and local_breakdown >= 0.50) or support_removed >= 0.70)
        if rule_id == "sieve2_multi2_vp_prior_month_high_break_vp_node_long":
            return current_profit < 0.003 and bid_absorption >= 0.35 and local_breakdown >= 0.50
        return False

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if self._orderbook_crash_exit_active(pair, current_time, trade):
                return f"{rule_id}_orderbook_crash_exit"
            if self._family_specific_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                return f"{rule_id}_family_specific_long_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class _TraderRuleBlockBtcSieveCurrentBestNoBosSingleFamilyLongExitBase(
    TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy
):
    """
    Research helper for one-family-at-a-time long-exit tests.
    """

    def _single_family_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        _ = row, trade, rule_id, elapsed_hours, current_profit
        return False

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if self._orderbook_crash_exit_active(pair, current_time, trade):
                return f"{rule_id}_orderbook_crash_exit"
            if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                return f"{rule_id}_single_family_long_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestNoBosTriangleSupportClearedExitStrategy(
    _TraderRuleBlockBtcSieveCurrentBestNoBosSingleFamilyLongExitBase
):
    """
    Trader question:
    After removing the weak BOS long family, do triangle squeeze breakout
    longs need an early exit when support clears or breakout failure appears?
    """

    target_rule_id = "sieve1_geometry_triangle_squeeze_breakout_long_1h"

    def _single_family_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id or elapsed_hours < 1.0:
            return False
        support_cleared = self._value_from_row(row, ("conf_ob_support_cleared",))
        failed_breakout = self._value_from_row(
            row,
            ("st_failed_breakout_structure_risk", "conf_ob_breakout_failure", "conf_failed_breakout_exhaustion_score"),
        )
        return current_profit < 0.004 and (support_cleared >= 0.35 or failed_breakout >= 0.45)


class TraderRuleBlockBtcSieveCurrentBestNoBosLvnOrderbookFragilityExitStrategy(
    _TraderRuleBlockBtcSieveCurrentBestNoBosSingleFamilyLongExitBase
):
    """
    Trader question:
    After removing the weak BOS long family, do LVN fast-traverse longs need
    an early exit when support is removed, downside is thin, and the book is fragile?
    """

    target_rule_id = "sieve1_vp_lvn_fast_traverse_long_1h"

    def _single_family_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id or elapsed_hours < 1.0:
            return False
        support_removed = self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared"))
        downside_vacuum = self._value_from_row(
            row, ("conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum")
        )
        spread_fragility = self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme"))
        stress_count = int(support_removed >= 0.35) + int(downside_vacuum >= 0.35) + int(spread_fragility >= 0.50)
        return current_profit < 0.005 and stress_count >= 2


class TraderRuleBlockBtcSieveCurrentBestNoBosLadderStructureFailureExitStrategy(
    _TraderRuleBlockBtcSieveCurrentBestNoBosSingleFamilyLongExitBase
):
    """
    Trader question:
    After removing the weak BOS long family, do ladder support-hold longs need
    an early exit when the structure starts failing and support clears?
    """

    target_rule_id = "sieve1_ladder_long_sup_hold"

    def _single_family_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id or elapsed_hours < 1.0:
            return False
        support_cleared = self._value_from_row(row, ("conf_ob_support_cleared",))
        failed_structure = self._value_from_row(
            row, ("st_failed_breakdown_structure_risk", "st_failed_breakout_structure_risk")
        )
        return current_profit < 0.003 and failed_structure >= 0.45 and support_cleared >= 0.25


class TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoBosTriangleSupportClearedExitStrategy
):
    """
    Trader question:
    Does the current max-return no-BOS triangle long exit combine cleanly with
    the stricter short reclaim failure exit that reduced drawdown?
    """

    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
            elif TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                self, row, trade, rule_id
            ):
                return f"{rule_id}_strict_short_reclaim_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakLongStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy
):
    """
    Trader question:
    Does the balanced branch improve if the high-loss, low-win-rate overtrade
    resistance-break long family is removed instead of trying to exit it later?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy.removed_rule_ids | {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
    }


class TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy
):
    """
    Trader question:
    Does the balanced branch improve if the low-efficiency POC rejection short
    family is removed?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy.removed_rule_ids | {
        "sieve2_reframed_vp_poc_reject_short_1h",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCurrentPriorityStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    If rare complete-pattern entries are added to the current strongest branch,
    but existing current-best signals keep same-hour priority, do the rare
    entries add useful trades without damaging the block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_current_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260607_current_strongest_plus_complete_pattern_rare_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRarePriorityStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    If rare complete-pattern entries are added to the current strongest branch
    and rare entries take same-hour priority, does the high-win-rate rare
    family improve the final block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260607_current_strongest_plus_complete_pattern_rare_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCompressionRangeStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    If complete-pattern rare entries only fire during compression/range-break
    context and take same-hour priority, does the cleaner rare subset improve
    the current strongest branch?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_compression_range_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260607_current_strongest_plus_complete_pattern_rare_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRangeBreakStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    If complete-pattern rare entries only need range-break agreement and take
    same-hour priority, does that filtered rare subset improve the current
    strongest branch?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_range_break_rare_priority.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260607_current_strongest_plus_complete_pattern_rare_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestNoPocNoOvertradePlusRareRangeBreakReplacementStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    If the weak overtrade resistance-break long family is removed and replaced
    with range-break-filtered complete-pattern rare entries, does the current
    strongest branch regain return while keeping the cleaner family profile?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260607_current_strongest_no_poc_no_overtrade_plus_rare_range_break_replacement.parquet"
    )
    exit_settings_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_exit_research_20260607_current_strongest_plus_complete_pattern_rare_selected.csv"
    )


class TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakAndPocShortStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy
):
    """
    Trader question:
    Do the two low-efficiency high-loss-burden families improve the balanced
    branch if removed together?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy.removed_rule_ids | {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve2_reframed_vp_poc_reject_short_1h",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocNoTripleBottomLongStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does the current strongest branch improve if the small low-win-rate triple
    bottom long family is removed instead of adding another exit rule?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy.removed_rule_ids | {
        "sieve1_triple_bottom_confirmed_long_1h",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Do the tiny negative-contribution long families help or hurt once the
    current strongest no-POC branch is already established?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy.removed_rule_ids | {
        "sieve1_triple_bottom_confirmed_long_1h",
        "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_1h",
        "sieve2_multi2_tlv2_vp_res_break_vp_val_long_1d",
    }


class TraderRuleBlockBtcSieveCurrentBestNoTinyShortSupportBreakReclaimExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy
):
    """
    Trader question:
    Does the largest short support-break family need an exit only when the
    breakdown story starts failing and pressure turns against the short?
    """

    target_rule_id = "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h"

    def _targeted_short_reclaim_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if not trade.is_short or rule_id != self.target_rule_id:
            return False
        if elapsed_hours < 2.0 or current_profit > -0.004:
            return False
        volume_against_short = (
            self._value_from_row(row, ("px_volume_pressure_6h",)) > 0.0
            and self._value_from_row(row, ("px_volume_pressure_24h",)) > 0.0
        )
        breakdown_failed = self._value_from_row(row, ("conf_ob_breakdown_failure", "st_failed_breakdown_structure_risk"))
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        reclaim_pressure = self._value_from_row(
            row,
            (
                "conf_structure_bullish_state_score",
                "conf_structure_breakout_trigger_score",
                "px_close_breakout_6h",
            ),
        )
        return volume_against_short and (
            breakdown_failed > 0.0 or bid_absorption > 0.0 or reclaim_pressure >= 0.25
        )

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        rule_id, settings = self._settings_for_trade(trade)
        if settings is not None and current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None and self._targeted_short_reclaim_exit_active(
            row, trade, rule_id, elapsed_hours, current_profit
        ):
            return f"{rule_id}_targeted_short_reclaim_exit"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)


class TraderRuleBlockBtcSieveCurrentBestNoTinyOvertradeResBreakSupportLossExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy
):
    """
    Trader question:
    Does the weak overtrade resistance-break long family need an exit only
    when support is removed, the book is fragile, and the trade is not working?
    """

    target_rule_id = "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard"

    def _targeted_long_support_loss_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id:
            return False
        if elapsed_hours < 1.0 or current_profit > 0.003:
            return False
        support_removed = self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared"))
        spread_fragility = self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme"))
        downside_vacuum = self._value_from_row(
            row, ("conf_ob_downside_vacuum_after_support_removed", "conf_ob_downside_vacuum")
        )
        structure_break = self._value_from_row(
            row, ("conf_structure_breakdown_trigger_score", "px_close_breakdown_6h", "st_1h_vp_below_value_area")
        )
        stress_count = int(support_removed >= 0.25) + int(spread_fragility >= 0.35) + int(downside_vacuum >= 0.25)
        return stress_count >= 2 or (support_removed >= 0.25 and structure_break >= 0.35)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        rule_id, settings = self._settings_for_trade(trade)
        if settings is not None and current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None and self._targeted_long_support_loss_exit_active(
            row, trade, rule_id, elapsed_hours, current_profit
        ):
            return f"{rule_id}_targeted_support_loss_exit"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)


class TraderRuleBlockBtcSieveCurrentBestNoTinyVpValLongFailureExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy
):
    """
    Trader question:
    Does the VP resistance-break VAL long family need an early exit when local
    breakdown or bid absorption shows buyers are not following through?
    """

    target_rule_id = "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h"

    def _targeted_vp_val_long_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id:
            return False
        if elapsed_hours < 1.0 or current_profit > 0.003:
            return False
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        local_breakdown = self._value_from_row(row, ("px_close_breakdown_6h", "conf_structure_breakdown_trigger_score"))
        support_removed = self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared"))
        bearish_state = self._value_from_row(row, ("conf_structure_bearish_state_score", "st_1h_vp_below_value_area"))
        return (
            bid_absorption >= 0.25 and (local_breakdown > 0.0 or support_removed >= 0.35)
        ) or (local_breakdown >= 0.35 and bearish_state >= 0.35)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        rule_id, settings = self._settings_for_trade(trade)
        if settings is not None and current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None and self._targeted_vp_val_long_exit_active(
            row, trade, rule_id, elapsed_hours, current_profit
        ):
            return f"{rule_id}_targeted_vp_val_long_exit"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)


class TraderRuleBlockBtcSieveCurrentBestNoTinyTargetedFamilyExitStackStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoTinyShortSupportBreakReclaimExitStrategy,
    TraderRuleBlockBtcSieveCurrentBestNoTinyOvertradeResBreakSupportLossExitStrategy,
    TraderRuleBlockBtcSieveCurrentBestNoTinyVpValLongFailureExitStrategy,
):
    """
    Trader question:
    Do the three highest-priority current-baseline family-specific exits help
    together without becoming a broad generic exit stack?
    """


class _CurrentBestEntryQualityFilterBase(TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy):
    """
    Research helper for family-specific entry-quality filters.

    These filters do not delete a family globally. They keep the family only
    when the same-hour trader state still supports the original entry story.
    """

    quality_filter_rule_ids: set[str] = set()

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        _ = row, rule_id, is_short
        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        populated = super().populate_entry_trend(dataframe, metadata)
        if not self.quality_filter_rule_ids or "enter_tag" not in populated.columns:
            return populated
        tags = populated["enter_tag"].fillna("").astype(str).str.split().str[0]
        active = tags.isin(self.quality_filter_rule_ids)
        if not active.any():
            return populated
        remove_indexes: list[Any] = []
        for index, row in populated.loc[active].iterrows():
            rule_id = str(tags.loc[index])
            is_short = bool(row.get("enter_short", 0))
            if not self._entry_quality_passes(row, rule_id, is_short):
                remove_indexes.append(index)
        if remove_indexes:
            populated.loc[remove_indexes, ["enter_long", "enter_short"]] = 0
            populated.loc[remove_indexes, "enter_tag"] = None
        return populated


class TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy(
    _CurrentBestEntryQualityFilterBase
):
    """
    Trader question:
    Can the weak overtrade resistance-break long family become cleaner if it
    only enters when breakout/volume context is present and support is not
    already being removed?
    """

    quality_filter_rule_ids = {"sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard"}

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if is_short or rule_id not in self.quality_filter_rule_ids:
            return True
        support_stress = self._value_from_row(
            row,
            (
                "conf_ob_support_removed_strength",
                "conf_ob_support_cleared",
                "conf_ob_downside_vacuum_after_support_removed",
                "conf_ob_downside_vacuum",
            ),
        )
        book_fragile = self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme"))
        breakout_context = self._value_from_row(
            row,
            (
                "conf_structure_breakout_trigger_score",
                "conf_structure_bullish_state_score",
                "st_1h_vp_above_value_area",
                "px_close_breakout_6h",
            ),
        )
        volume_support = self._value_from_row(
            row,
            ("conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"),
        )
        if support_stress >= 0.35 or book_fragile >= 0.60:
            return False
        return breakout_context >= 0.35 or volume_support > 0.0


class TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy(
    _CurrentBestEntryQualityFilterBase
):
    """
    Trader question:
    Can the VP resistance-break VAL long family improve if it avoids entries
    where buyers are already being absorbed or local structure is breaking
    down?
    """

    quality_filter_rule_ids = {"sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h"}

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if is_short or rule_id not in self.quality_filter_rule_ids:
            return True
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        support_stress = self._value_from_row(row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared"))
        local_breakdown = self._value_from_row(row, ("px_close_breakdown_6h", "conf_structure_breakdown_trigger_score"))
        bullish_context = self._value_from_row(
            row,
            (
                "conf_structure_bullish_state_score",
                "conf_structure_breakout_trigger_score",
                "st_1h_vp_above_value_area",
                "st_4h_vp_above_value_area",
            ),
        )
        if bid_absorption >= 0.35 or support_stress >= 0.50 or local_breakdown >= 0.35:
            return False
        return bullish_context >= 0.25


class TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy(
    _CurrentBestEntryQualityFilterBase
):
    """
    Trader question:
    Can the largest support-break short family improve if it avoids shorts
    when the book/volume already shows reclaim or failed-breakdown risk?
    """

    quality_filter_rule_ids = {"sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h"}

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if not is_short or rule_id not in self.quality_filter_rule_ids:
            return True
        bullish_reclaim = self._value_from_row(
            row,
            (
                "conf_ob_breakdown_failure",
                "conf_ob_bid_absorption",
                "conf_structure_bullish_state_score",
                "conf_structure_breakout_trigger_score",
                "px_close_breakout_6h",
            ),
        )
        bearish_context = self._value_from_row(
            row,
            (
                "conf_structure_breakdown_trigger_score",
                "conf_structure_bearish_state_score",
                "st_1h_vp_below_value_area",
                "px_close_breakdown_6h",
                "conf_volume_bearish_confirmation",
            ),
        )
        volume_against_short = self._value_from_row(row, ("px_volume_pressure_6h", "px_volume_pressure_24h"))
        if bullish_reclaim >= 0.35 or volume_against_short > 0.0:
            return False
        return bearish_context >= 0.25


class TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy(
    _CurrentBestEntryQualityFilterBase
):
    """
    Trader question:
    Can the support-break short family keep more of its profitable trades if
    we skip only the clearest reclaim/failure-at-entry states?
    """

    quality_filter_rule_ids = {"sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h"}

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if not is_short or rule_id not in self.quality_filter_rule_ids:
            return True
        reclaim_warning = self._value_from_row(
            row,
            (
                "conf_ob_breakdown_failure",
                "conf_ob_bid_absorption",
                "conf_structure_bullish_state_score",
                "px_close_breakout_6h",
            ),
        )
        bearish_context = self._value_from_row(
            row,
            (
                "conf_structure_breakdown_trigger_score",
                "conf_structure_bearish_state_score",
                "st_1h_vp_below_value_area",
                "px_close_breakdown_6h",
                "conf_volume_bearish_confirmation",
            ),
        )
        volume_against_6h = self._value_from_row(row, ("px_volume_pressure_6h",))
        volume_against_24h = self._value_from_row(row, ("px_volume_pressure_24h",))
        if reclaim_warning >= 0.55 and bearish_context < 0.45:
            return False
        if volume_against_6h > 0.0 and volume_against_24h > 0.0 and reclaim_warning >= 0.25:
            return False
        return bearish_context >= 0.20


class TraderRuleBlockBtcSieveCurrentBestOvertradeAndSoftShortCleanEntryStrategy(
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
    TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy,
):
    """
    Trader question:
    Does the useful overtrade long entry filter combine with a softer
    support-break short filter, without the VP-VAL over-filter that damaged
    the previous stack?
    """

    quality_filter_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy.quality_filter_rule_ids
        | TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy.quality_filter_rule_ids
    )

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if rule_id in TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy.quality_filter_rule_ids:
            return TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy._entry_quality_passes(
                self, row, rule_id, is_short
            )
        if rule_id in TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy.quality_filter_rule_ids:
            return TraderRuleBlockBtcSieveCurrentBestShortSupportBreakSoftCleanEntryStrategy._entry_quality_passes(
                self, row, rule_id, is_short
            )
        return True


class TraderRuleBlockBtcSieveCurrentBestEntryQualityFilterStackStrategy(
    TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
    TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy,
):
    """
    Trader question:
    Do the three family-specific entry-quality filters combine, or does
    filtering multiple still-profitable families remove too much edge?
    """

    quality_filter_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy.quality_filter_rule_ids
        | TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy.quality_filter_rule_ids
        | TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy.quality_filter_rule_ids
    )

    def _entry_quality_passes(self, row: Any, rule_id: str, is_short: bool) -> bool:
        if rule_id in TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy.quality_filter_rule_ids:
            return TraderRuleBlockBtcSieveCurrentBestShortSupportBreakCleanEntryStrategy._entry_quality_passes(
                self, row, rule_id, is_short
            )
        if rule_id in TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy.quality_filter_rule_ids:
            return TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy._entry_quality_passes(
                self, row, rule_id, is_short
            )
        if rule_id in TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy.quality_filter_rule_ids:
            return TraderRuleBlockBtcSieveCurrentBestVpValLongCleanEntryStrategy._entry_quality_passes(
                self, row, rule_id, is_short
            )
        return True


class TraderRuleBlockBtcSieveCurrentBestSignalStackAddOnStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy
):
    """
    Trader question:
    If a later signal points the same way while a trade is already working,
    should that overlap be treated as add/hold evidence instead of being
    ignored as a duplicate entry?
    """

    position_adjustment_enable = True
    max_entry_position_adjustment = 1
    same_direction_add_fraction = 0.25
    same_direction_add_min_elapsed_hours = 1.0
    same_direction_add_min_profit = 0.0

    def _same_direction_signal_active(self, row: Any, trade: Any) -> bool:
        if trade.is_short:
            return self._value_from_row(row, ("trader_rule_enter_short",)) > 0.0
        return self._value_from_row(row, ("trader_rule_enter_long",)) > 0.0

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        _ = current_rate, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if getattr(trade, "has_open_orders", False):
            return None
        if getattr(trade, "nr_of_successful_entries", 0) > 1:
            return None
        if current_profit < self.same_direction_add_min_profit:
            return None
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if elapsed_hours < self.same_direction_add_min_elapsed_hours:
            return None
        row = self._current_feature_row(trade.pair, current_time)
        if row is None or not self._same_direction_signal_active(row, trade):
            return None
        base_stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        add_stake = base_stake * float(self.same_direction_add_fraction)
        if add_stake <= 0.0 or add_stake > max_stake:
            return None
        if min_stake is not None and add_stake < float(min_stake):
            return None
        rule_id, _settings = self._settings_for_trade(trade)
        return add_stake, f"{rule_id}_same_direction_add"


class TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy(
    TraderRuleBlockBtcSieveCurrentBestSignalStackAddOnStrategy
):
    """
    Trader question:
    If we reserve part of the original stake, then deploy that reserve only
    when a later same-direction signal appears, does stacking improve quality
    without exceeding the baseline trade exposure?
    """

    same_direction_initial_stake_fraction = 0.80

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
        baseline_stake = super().custom_stake_amount(
            pair=pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        )
        stake = baseline_stake * float(self.same_direction_initial_stake_fraction)
        if min_stake is not None and stake < float(min_stake):
            return baseline_stake
        return min(float(max_stake), stake)


class TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy(
    TraderRuleBlockBtcSieveCurrentBestSignalStackAddOnStrategy
):
    """
    Trader question:
    Can same-direction signal stacking help if stake is reserved only for
    families where the overlap audit showed useful add/hold evidence?
    """

    same_direction_initial_stake_fraction = 0.80
    signal_stack_reserve_rule_ids = {
        "sieve1_bos_bull_continuation_long_1h",
        "sieve1_continuation_flag_present_long_4h",
        "sieve1_geometry_triangle_squeeze_breakout_long_1h",
        "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
        "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h",
        "sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h",
        "sieve2_multi2_vp_prior_month_high_break_vp_node_long",
        "sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard",
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
    }

    @classmethod
    def _entry_rule_id(cls, entry_tag: str | None) -> str:
        return str(entry_tag or "").split()[0]

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
        baseline_stake = super().custom_stake_amount(
            pair=pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        )
        if self._entry_rule_id(entry_tag) not in self.signal_stack_reserve_rule_ids:
            return baseline_stake
        stake = baseline_stake * float(self.same_direction_initial_stake_fraction)
        if min_stake is not None and stake < float(min_stake):
            return baseline_stake
        return min(float(max_stake), stake)

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        rule_id, _settings = self._settings_for_trade(trade)
        if rule_id not in self.signal_stack_reserve_rule_ids:
            return None
        return super().adjust_trade_position(
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            min_stake=min_stake,
            max_stake=max_stake,
            current_entry_rate=current_entry_rate,
            current_exit_rate=current_exit_rate,
            current_entry_profit=current_entry_profit,
            current_exit_profit=current_exit_profit,
            **kwargs,
        )


class TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy(
    TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy
):
    """
    Trader question:
    Does the selective stacking branch improve if the marginal prior-month
    breakout family is removed from the reserve/add list after it became the
    worst family in the first selective Freqtrade run?
    """

    signal_stack_reserve_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy.signal_stack_reserve_rule_ids
        - {"sieve2_multi2_vp_prior_month_high_break_vp_node_long"}
    )


class TraderRuleBlockBtcSieveCurrentBestSignalStackSharperReserveAddOnStrategy(
    TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy
):
    """
    Trader question:
    Does the sharp stacking branch improve if the weak overtrade
    resistance-break family keeps normal entry size but is no longer eligible
    for reserve/add behaviour?
    """

    signal_stack_reserve_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy.signal_stack_reserve_rule_ids
        - {"sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard"}
    )


class TraderRuleBlockBtcSieveCurrentBestSignalStackSharpNoOvertradeFamilyStrategy(
    TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy
):
    """
    Trader question:
    Does the sharp stacking branch improve if the weak overtrade
    resistance-break family is removed entirely instead of only changing its
    add/hold treatment?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy.removed_rule_ids | {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
    }
    signal_stack_reserve_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy.signal_stack_reserve_rule_ids
        - {"sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard"}
    )


class _CurrentBestPositionManagementBase(TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy):
    """
    Position-management research helper.

    One BTC position can still be managed in pieces: reserve stake for adds,
    add when later evidence agrees, take partial profit, and reduce when the
    position thesis starts weakening.
    """

    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    initial_stake_fraction = 0.80
    same_direction_add_fraction = 0.25
    same_direction_add_min_elapsed_hours = 1.0
    same_direction_add_min_profit = 0.004
    first_partial_profit = 0.030
    first_partial_fraction = 0.25
    second_partial_profit = 0.055
    second_partial_fraction = 0.25
    risk_reduce_min_profit = 0.010
    risk_reduce_fraction = 0.35

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
        baseline_stake = super().custom_stake_amount(
            pair=pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        )
        stake = baseline_stake * float(self.initial_stake_fraction)
        if min_stake is not None and stake < float(min_stake):
            return baseline_stake
        return min(float(max_stake), stake)

    def _same_direction_signal_active(self, row: Any, trade: Any) -> bool:
        if trade.is_short:
            return self._value_from_row(row, ("trader_rule_enter_short",)) > 0.0
        return self._value_from_row(row, ("trader_rule_enter_long",)) > 0.0

    def _opposing_signal_active(self, row: Any, trade: Any) -> bool:
        if trade.is_short:
            return self._value_from_row(row, ("trader_rule_enter_long",)) > 0.0
        return self._value_from_row(row, ("trader_rule_enter_short",)) > 0.0

    def _run_weakening_active(self, row: Any, trade: Any) -> bool:
        if trade.is_short:
            bullish_structure = self._value_from_row(
                row,
                (
                    "conf_structure_bullish_state_score",
                    "conf_structure_breakout_trigger_score",
                    "px_close_breakout_6h",
                    "st_1h_vp_above_value_area",
                ),
            )
            bullish_volume = self._value_from_row(
                row,
                ("conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"),
            )
            breakdown_failure = self._value_from_row(
                row, ("conf_ob_breakdown_failure", "st_failed_breakdown_structure_risk")
            )
            return bullish_structure >= 0.45 or (bullish_volume > 0.0 and breakdown_failure >= 0.25)
        bearish_structure = self._value_from_row(
            row,
            (
                "conf_structure_bearish_state_score",
                "conf_structure_breakdown_trigger_score",
                "px_close_breakdown_6h",
                "st_1h_vp_below_value_area",
            ),
        )
        bearish_volume = self._value_from_row(row, ("conf_volume_bearish_confirmation",))
        support_removed = self._value_from_row(
            row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared", "conf_ob_downside_vacuum")
        )
        return bearish_structure >= 0.45 or (bearish_volume >= 0.35 and support_removed >= 0.25)

    @staticmethod
    def _successful_entries(trade: Any) -> int:
        return int(getattr(trade, "nr_of_successful_entries", 1) or 1)

    @staticmethod
    def _successful_exits(trade: Any) -> int:
        return int(getattr(trade, "nr_of_successful_exits", 0) or 0)

    def _partial_exit_amount(self, trade: Any, fraction: float) -> float | None:
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        amount = stake * float(fraction)
        if amount <= 0.0:
            return None
        return -amount

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        _ = current_rate, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if getattr(trade, "has_open_orders", False):
            return None

        exits = self._successful_exits(trade)
        if exits == 0 and current_profit >= float(self.first_partial_profit):
            amount = self._partial_exit_amount(trade, self.first_partial_fraction)
            if amount is not None:
                return amount, "first_partial_profit"
        if exits == 1 and current_profit >= float(self.second_partial_profit):
            amount = self._partial_exit_amount(trade, self.second_partial_fraction)
            if amount is not None:
                return amount, "second_partial_profit"

        row = self._current_feature_row(trade.pair, current_time)
        if row is None:
            return None

        if exits == 0 and current_profit >= float(self.risk_reduce_min_profit):
            if self._opposing_signal_active(row, trade) or self._run_weakening_active(row, trade):
                amount = self._partial_exit_amount(trade, self.risk_reduce_fraction)
                if amount is not None:
                    return amount, "risk_reduce_partial"

        if self._successful_entries(trade) > self.max_entry_position_adjustment:
            return None
        if current_profit < float(self.same_direction_add_min_profit):
            return None
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if elapsed_hours < float(self.same_direction_add_min_elapsed_hours):
            return None
        if not self._same_direction_signal_active(row, trade):
            return None
        base_stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        add_stake = base_stake * float(self.same_direction_add_fraction)
        if add_stake <= 0.0 or add_stake > max_stake:
            return None
        if min_stake is not None and add_stake < float(min_stake):
            return None
        return add_stake, "same_direction_stack"


class TraderRuleBlockBtcSieveCurrentBestPmBalancedStackPartialStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    Does a balanced reserve/add/partial-exit model improve the current best
    without changing entry logic?
    """


class TraderRuleBlockBtcSieveCurrentBestPmAggressiveStackPartialStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    Does keeping more initial exposure while still allowing adds and partial
    exits improve the current best?
    """

    initial_stake_fraction = 0.90
    same_direction_add_fraction = 0.20
    same_direction_add_min_profit = 0.006
    first_partial_profit = 0.040
    second_partial_profit = 0.070


class TraderRuleBlockBtcSieveCurrentBestPmDefensivePartialStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    Does stronger partial profit-taking and earlier risk reduction reduce
    drawdown enough to justify lower exposure?
    """

    initial_stake_fraction = 0.70
    same_direction_add_fraction = 0.20
    same_direction_add_min_profit = 0.008
    first_partial_profit = 0.022
    first_partial_fraction = 0.35
    second_partial_profit = 0.045
    second_partial_fraction = 0.25
    risk_reduce_min_profit = 0.004
    risk_reduce_fraction = 0.50


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmStrategy(
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
    _CurrentBestPositionManagementBase,
):
    """
    Trader question:
    Does the best current entry-quality improvement become stronger when it
    also manages the BTC position with adds and partial exits?
    """


class TraderRuleBlockBtcSieveCurrentBestPmStackOnlyStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    If we reserve stake only for same-direction adds, without partial exits,
    does stacking improve the current best?
    """

    first_partial_profit = 999.0
    second_partial_profit = 999.0
    risk_reduce_min_profit = 999.0


class TraderRuleBlockBtcSieveCurrentBestPmProfitRunnerStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    Does taking only late partial profits let winners run while still reducing
    end-of-run giveback?
    """

    initial_stake_fraction = 1.0
    same_direction_add_min_profit = 999.0
    first_partial_profit = 0.055
    first_partial_fraction = 0.20
    second_partial_profit = 0.090
    second_partial_fraction = 0.20
    risk_reduce_min_profit = 999.0


class TraderRuleBlockBtcSieveCurrentBestPmRiskReduceOnlyStrategy(_CurrentBestPositionManagementBase):
    """
    Trader question:
    Does reducing only when the run starts weakening protect gains without
    cutting normal winners too early?
    """

    initial_stake_fraction = 1.0
    same_direction_add_min_profit = 999.0
    first_partial_profit = 999.0
    second_partial_profit = 999.0
    risk_reduce_min_profit = 0.012
    risk_reduce_fraction = 0.35


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmStackOnlyStrategy(
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
    TraderRuleBlockBtcSieveCurrentBestPmStackOnlyStrategy,
):
    """
    Trader question:
    Does the best clean-entry branch improve when it only reserves/adds on
    same-direction evidence?
    """


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanPmRiskReduceStrategy(
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
    TraderRuleBlockBtcSieveCurrentBestPmRiskReduceOnlyStrategy,
):
    """
    Trader question:
    Does the best clean-entry branch improve when it only trims on run-ending
    or opposing evidence?
    """


class TraderRuleBlockBtcSieveCurrentBestNoTinyNegativeDecShortFailureExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy
):
    """
    Trader question:
    Can the new strongest branch reduce the December-style short loss cluster
    by exiting only targeted short families after the breakdown thesis starts
    failing while the trade is already losing?
    """

    dec_short_failure_exit_rule_ids = {
        "sieve1_choch_bear_reversal_short_4h",
        "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
        "sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h",
        "sieve2_reversal_double_top_present_short_1h",
    }

    def _dec_short_failure_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if not trade.is_short or rule_id not in self.dec_short_failure_exit_rule_ids:
            return False
        if elapsed_hours < 1.0 or current_profit > -0.006:
            return False
        failed_breakdown = self._value_from_row(row, ("st_failed_breakdown_structure_risk",))
        exhaustion = self._value_from_row(row, ("conf_failed_breakdown_exhaustion_score",))
        support_bounce = self._value_from_row(row, ("ob_spot_obts_price_bounced_support_3h",))
        breakdown_failure = self._value_from_row(
            row,
            (
                "conf_ob_breakdown_failure",
                "ob_spot_obts_breakdown_failure_score",
            ),
        )
        bullish_structure = self._value_from_row(row, ("conf_structure_bullish_state_score",))
        reclaim_count = int(exhaustion >= 0.35) + int(support_bounce > 0.0) + int(breakdown_failure > 0.05)
        return failed_breakdown >= 1.0 and reclaim_count >= 1 and bullish_structure >= 0.25

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
            else:
                if TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                    self, row, trade, rule_id
                ):
                    return f"{rule_id}_strict_short_reclaim_exit"
                if self._dec_short_failure_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_dec_short_failure_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does removing every current no-POC family with negative absolute
    contribution improve quality, or does it cut too much useful coverage?
    """

    removed_rule_ids = TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy.removed_rule_ids | {
        "sieve1_triple_bottom_confirmed_long_1h",
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
        "sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_1h",
        "sieve2_multi2_tlv2_vp_res_break_vp_val_long_1d",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does the current strongest branch improve if the largest short loss-burden
    family exits when volume pressure turns against the short?
    """

    short_volume_pressure_exit_rule_ids = {"sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h"}

    def _short_volume_pressure_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if not trade.is_short or rule_id not in self.short_volume_pressure_exit_rule_ids or elapsed_hours < 1.0:
            return False
        volume_6h = self._value_from_row(row, ("px_volume_pressure_6h",))
        volume_24h = self._value_from_row(row, ("px_volume_pressure_24h",))
        return current_profit < 0.006 and volume_6h > 0.0 and volume_24h > 0.0

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
            else:
                if TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                    self, row, trade, rule_id
                ):
                    return f"{rule_id}_strict_short_reclaim_exit"
                if self._short_volume_pressure_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_short_volume_pressure_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitExpandedStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitStrategy
):
    """
    Trader question:
    Does the same short volume-pressure invalidation help a small set of
    short families whose losers showed clear against-short pressure?
    """

    short_volume_pressure_exit_rule_ids = {
        "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
        "sieve1_choch_bear_reversal_short_4h",
        "sieve2_geometry_triangle_squeeze_breakdown_short_4h",
        "sieve2_reversal_double_top_present_short_1h",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does the largest short loss-burden family improve if we exit only when the
    breakdown story itself looks invalidated, rather than using volume pressure
    alone?
    """

    short_failure_context_exit_rule_ids = {"sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h"}

    def _short_failure_context_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if not trade.is_short or rule_id not in self.short_failure_context_exit_rule_ids:
            return False
        if elapsed_hours < 1.0 or current_profit > 0.004:
            return False
        breakdown_failed = self._value_from_row(row, ("conf_ob_breakdown_failure",))
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        bullish_pressure = self._value_from_row(
            row,
            (
                "px_volume_pressure_6h",
                "conf_volume_bullish_confirmation",
                "conf_ob_pressure_divergence",
            ),
        )
        close_reclaim = self._value_from_row(
            row,
            (
                "px_close_breakout_6h",
                "conf_structure_breakout_trigger_score",
                "conf_structure_bullish_state_score",
            ),
        )
        if breakdown_failed > 0.0 and bullish_pressure > 0.0:
            return True
        if bid_absorption > 0.0 and (bullish_pressure > 0.0 or close_reclaim > 0.0):
            return True
        return current_profit < -0.002 and close_reclaim > 0.0 and bullish_pressure > 0.0

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
            else:
                if TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                    self, row, trade, rule_id
                ):
                    return f"{rule_id}_strict_short_reclaim_exit"
                if self._short_failure_context_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_short_failure_context_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitExpandedStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitStrategy
):
    """
    Trader question:
    Does the same contextual short-failure exit help other short families where
    losers showed bid absorption, breakdown failure, or reclaim pressure?
    """

    short_failure_context_exit_rule_ids = {
        "sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h",
        "sieve1_choch_bear_reversal_short_4h",
        "sieve2_geometry_triangle_squeeze_breakdown_short_4h",
        "sieve2_reversal_double_top_present_short_1h",
    }


class TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitStrategy
):
    """
    Trader question:
    If contextual short-failure exits were too broad, does the same idea help
    only after the short is already clearly losing?
    """

    def _short_failure_context_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if not trade.is_short or rule_id not in self.short_failure_context_exit_rule_ids:
            return False
        if elapsed_hours < 2.0 or current_profit > -0.010:
            return False
        breakdown_failed = self._value_from_row(row, ("conf_ob_breakdown_failure",))
        bid_absorption = self._value_from_row(row, ("conf_ob_bid_absorption",))
        bullish_pressure = self._value_from_row(
            row,
            (
                "px_volume_pressure_6h",
                "conf_volume_bullish_confirmation",
                "conf_ob_pressure_divergence",
            ),
        )
        close_reclaim = self._value_from_row(
            row,
            (
                "px_close_breakout_6h",
                "conf_structure_breakout_trigger_score",
                "conf_structure_bullish_state_score",
            ),
        )
        return (breakdown_failed > 0.0 or bid_absorption > 0.0) and (
            bullish_pressure > 0.0 or close_reclaim > 0.0
        )


class TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitExpandedStrategy(
    TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitStrategy
):
    """
    Trader question:
    Does the stricter loss-only short-failure exit help across the related
    short families without cutting normal winners?
    """

    short_failure_context_exit_rule_ids = (
        TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitExpandedStrategy.short_failure_context_exit_rule_ids
    )


class TraderRuleBlockBtcSieveCurrentBestNoPocLadderSupportFailureExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does the current strongest branch improve if ladder support-hold longs exit
    only when the support-hold story breaks down?
    """

    target_rule_id = "sieve1_ladder_long_sup_hold"

    def _ladder_support_failure_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id:
            return False
        if elapsed_hours < 1.0 or current_profit > 0.002:
            return False
        failed_structure = self._value_from_row(
            row,
            (
                "st_failed_breakdown_structure_risk",
                "st_failed_breakout_structure_risk",
                "conf_structure_bearish_state_score",
            ),
        )
        support_cleared = self._value_from_row(row, ("conf_ob_support_cleared",))
        ask_absorption = self._value_from_row(row, ("conf_ob_ask_absorption",))
        return failed_structure >= 0.45 and (support_cleared > 0.0 or ask_absorption > 0.0)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
                if self._ladder_support_failure_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_ladder_support_failure_exit"
            elif TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                self, row, trade, rule_id
            ):
                return f"{rule_id}_strict_short_reclaim_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestNoPocLvnFragilityLossExitStrategy(
    TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy
):
    """
    Trader question:
    Does the current strongest branch improve if LVN fast-traverse longs exit
    when thin-liquidity travel turns into support removal and book fragility?
    """

    target_rule_id = "sieve1_vp_lvn_fast_traverse_long_1h"

    def _lvn_fragility_loss_exit_active(
        self,
        row: Any,
        trade: Any,
        rule_id: str,
        elapsed_hours: float,
        current_profit: float,
    ) -> bool:
        if trade.is_short or rule_id != self.target_rule_id:
            return False
        if elapsed_hours < 1.0 or current_profit > 0.004:
            return False
        support_removed = self._value_from_row(row, ("conf_ob_support_removed_strength",))
        downside_vacuum = self._value_from_row(
            row,
            (
                "conf_ob_downside_vacuum_after_support_removed",
                "conf_ob_downside_vacuum",
            ),
        )
        spread_fragility = self._value_from_row(row, ("conf_ob_spread_fragility", "conf_ob_single_venue_extreme"))
        failed_structure = self._value_from_row(
            row,
            (
                "st_failed_breakdown_structure_risk",
                "conf_structure_bearish_state_score",
                "conf_ob_breakdown_failure",
            ),
        )
        stress_count = int(support_removed > 0.0) + int(downside_vacuum > 0.0) + int(spread_fragility > 0.0)
        return stress_count >= 2 and failed_structure > 0.0

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, kwargs
        rule_id, settings = self._settings_for_trade(trade)
        if settings is None:
            return None
        if current_profit >= float(settings["take_profit"]):
            return f"{rule_id}_take_profit"
        row = self._current_feature_row(pair, current_time)
        elapsed_hours = self._min_elapsed_hours(current_time, trade)
        if row is not None:
            if not trade.is_short:
                if self._orderbook_crash_exit_active(pair, current_time, trade):
                    return f"{rule_id}_orderbook_crash_exit"
                if self._single_family_long_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_single_family_long_exit"
                if self._lvn_fragility_loss_exit_active(row, trade, rule_id, elapsed_hours, current_profit):
                    return f"{rule_id}_lvn_fragility_loss_exit"
            elif TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                self, row, trade, rule_id
            ):
                return f"{rule_id}_strict_short_reclaim_exit"
        if elapsed_hours >= float(settings["hold_hours"]):
            return f"{rule_id}_time_exit"
        return None


class TraderRuleBlockBtcSieveCurrentBestObExitStrictReclaimAndAcceptanceFailureStrategy(
    TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy
):
    """
    Trader question:
    Do the two stricter failure exits combine, or does combining them still
    over-manage the current best system?
    """

    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS | _PACK2_LONG_ACCEPTANCE_FAILURE_RULE_IDS

    def _structure_failure_exit_active(self, row: Any, trade: Any, rule_id: str) -> bool:
        if trade.is_short:
            return TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy._structure_failure_exit_active(
                self, row, trade, rule_id
            )
        return TraderRuleBlockBtcSieveCurrentBestObExitLongAcceptanceFailureStrategy._structure_failure_exit_active(
            self, row, trade, rule_id
        )


class TraderRuleBlockBtcSieveCurrentBestPack2TargetedFailureExitStackStrategy(
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy
):
    """
    Trader question:
    Do the targeted short-structure, long volume/time-confirmation, and long
    orderbook crash exits combine into a better production-alpha research
    candidate, or do they cut too many valid winners?
    """

    signal_file = _PACK2_CURRENT_BEST_SIGNAL_FILE
    exit_settings_file = _PACK2_CURRENT_BEST_EXIT_FILE
    orderbook_crash_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS
    volume_failure_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS
    time_to_confirm_exit_rule_ids = _PACK2_LONG_FAILURE_RULE_IDS
    structure_failure_exit_rule_ids = _PACK2_SHORT_FAILURE_RULE_IDS
    volatility_shock_exit_rule_ids = {
        "sieve1_ladder_long_sup_hold",
        "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h",
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard",
    }


class _ProfitLockPartialExitMixin:
    """
    Research helper for overtrade rescue tests.

    It does not change entries. It asks whether higher-risk lanes become useful
    when winning trades take partial profit and the remaining position is moved
    toward breakeven/profit-lock levels instead of giving the whole move back.
    """

    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    first_partial_profit = 0.045
    first_partial_fraction = 0.25
    second_partial_profit = 0.085
    second_partial_fraction = 0.20
    reversal_partial_profit = 0.018
    reversal_partial_fraction = 0.30
    profit_lock_levels = (
        (0.035, 0.002),
        (0.060, 0.018),
        (0.095, 0.045),
        (0.140, 0.080),
    )
    feature_file = TraderRuleBlockBtcSieveQualityOrderbookCrashExitStrategy.feature_file
    _profit_lock_feature_cache: DataFrame | None = None

    @classmethod
    def _load_profit_lock_features(cls) -> DataFrame:
        cached = cls._profit_lock_feature_cache
        if cached is not None:
            return cached
        path = Path(cls.feature_file)
        if not path.exists():
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._profit_lock_feature_cache = empty
            return empty
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        keep = [
            "date",
            "conf_ob_support_removed_strength",
            "conf_ob_support_cleared",
            "conf_ob_downside_vacuum",
            "conf_ob_bid_absorption",
            "conf_ob_breakdown_failure",
            "conf_volume_bullish_confirmation",
            "conf_volume_bearish_confirmation",
            "px_volume_pressure_6h",
            "px_volume_pressure_24h",
            "conf_structure_breakout_trigger_score",
            "conf_structure_breakdown_trigger_score",
            "conf_structure_bullish_state_score",
            "conf_structure_bearish_state_score",
            "st_1h_vp_above_value_area",
            "st_1h_vp_below_value_area",
            "st_failed_breakdown_structure_risk",
            "px_close_breakout_6h",
            "px_close_breakdown_6h",
        ]
        cleaned = frame[[column for column in keep if column in frame.columns]].dropna(subset=["date"]).copy()
        for column in cleaned.columns:
            if column != "date":
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)
        cls._profit_lock_feature_cache = cleaned.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        return cls._profit_lock_feature_cache

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        merged = super().populate_indicators(dataframe, metadata)
        features = self._load_profit_lock_features()
        if features.empty:
            return merged
        feature_columns = [column for column in features.columns if column != "date"]
        if any(column in merged.columns for column in feature_columns):
            return merged
        merged["date"] = pd.to_datetime(merged["date"], utc=True, errors="coerce")
        return merged.merge(features, on="date", how="left")

    @staticmethod
    def _value_from_row(row: Any, columns: tuple[str, ...]) -> float:
        values: list[float] = []
        for column in columns:
            try:
                value = float(row.get(column, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if value == value:
                values.append(value)
        return max(values) if values else 0.0

    @staticmethod
    def _min_elapsed_hours(current_time: datetime, trade: Any) -> float:
        open_date = getattr(trade, "open_date_utc", None)
        if open_date is None:
            return 0.0
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=timezone.utc)
        current = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        return max((current - open_date).total_seconds() / 3600.0, 0.0)

    def _current_feature_row(self, pair: str, current_time: datetime) -> Any | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        current = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        rows = dataframe.loc[dates.eq(current)]
        if rows.empty:
            rows = dataframe.loc[dates.le(current)].tail(1)
        if rows.empty:
            return None
        return rows.iloc[-1]

    @staticmethod
    def _successful_exits(trade: Any) -> int:
        return int(getattr(trade, "nr_of_successful_exits", 0) or 0)

    @staticmethod
    def _partial_exit_amount(trade: Any, fraction: float) -> float | None:
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        amount = stake * float(fraction)
        if amount <= 0.0:
            return None
        return -amount

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        base_stop = super().custom_stoploss(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            after_fill=after_fill,
            **kwargs,
        )
        lock_stop: float | None = None
        for trigger, locked_profit in self.profit_lock_levels:
            if current_profit >= float(trigger):
                candidate = stoploss_from_open(
                    float(locked_profit),
                    current_profit,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
                if candidate is not None:
                    lock_stop = candidate if lock_stop is None else min(lock_stop, candidate)
        stops = [value for value in (base_stop, lock_stop) if value is not None]
        return min(stops) if stops else None

    def _profit_reversal_active(self, row: Any, trade: Any) -> bool:
        if trade.is_short:
            bullish_structure = self._value_from_row(
                row,
                (
                    "conf_structure_bullish_state_score",
                    "conf_structure_breakout_trigger_score",
                    "px_close_breakout_6h",
                    "st_1h_vp_above_value_area",
                ),
            )
            bullish_volume = self._value_from_row(
                row,
                ("conf_volume_bullish_confirmation", "px_volume_pressure_6h", "px_volume_pressure_24h"),
            )
            breakdown_failure = self._value_from_row(
                row, ("conf_ob_breakdown_failure", "st_failed_breakdown_structure_risk")
            )
            return bullish_structure >= 0.45 or (bullish_volume > 0.0 and breakdown_failure > 0.0)
        bearish_structure = self._value_from_row(
            row,
            (
                "conf_structure_bearish_state_score",
                "conf_structure_breakdown_trigger_score",
                "px_close_breakdown_6h",
                "st_1h_vp_below_value_area",
            ),
        )
        bearish_volume = self._value_from_row(row, ("conf_volume_bearish_confirmation",))
        support_removed = self._value_from_row(
            row, ("conf_ob_support_removed_strength", "conf_ob_support_cleared", "conf_ob_downside_vacuum")
        )
        return bearish_structure >= 0.45 or (bearish_volume >= 0.35 and support_removed > 0.0)

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        _ = current_rate, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if getattr(trade, "has_open_orders", False):
            return None
        exits = self._successful_exits(trade)
        if exits == 0 and current_profit >= float(self.first_partial_profit):
            amount = self._partial_exit_amount(trade, self.first_partial_fraction)
            if amount is not None:
                return amount, "first_profit_lock_partial"
        if exits == 1 and current_profit >= float(self.second_partial_profit):
            amount = self._partial_exit_amount(trade, self.second_partial_fraction)
            if amount is not None:
                return amount, "second_profit_lock_partial"
        if exits == 0 and current_profit >= float(self.reversal_partial_profit):
            row = self._current_feature_row(trade.pair, current_time)
            if row is not None and self._profit_reversal_active(row, trade):
                amount = self._partial_exit_amount(trade, self.reversal_partial_fraction)
                if amount is not None:
                    return amount, "reversal_profit_protection_partial"
        return None


class _ProfitLockPartialExitTightMixin(_ProfitLockPartialExitMixin):
    first_partial_profit = 0.032
    first_partial_fraction = 0.30
    second_partial_profit = 0.065
    second_partial_fraction = 0.25
    reversal_partial_profit = 0.012
    reversal_partial_fraction = 0.35
    profit_lock_levels = (
        (0.024, 0.000),
        (0.045, 0.012),
        (0.075, 0.035),
        (0.115, 0.070),
    )


class _ProfitLockPartialExitRunnerMixin(_ProfitLockPartialExitMixin):
    first_partial_profit = 0.070
    first_partial_fraction = 0.18
    second_partial_profit = 0.120
    second_partial_fraction = 0.18
    reversal_partial_profit = 0.025
    reversal_partial_fraction = 0.25
    profit_lock_levels = (
        (0.045, 0.002),
        (0.080, 0.025),
        (0.130, 0.070),
        (0.200, 0.130),
    )


class TraderRuleBlockBtcSieveOvertradeRescueExpansionPack1NoCatastrophicProfitLockStrategy(
    _ProfitLockPartialExitMixin,
    TraderRuleBlockBtcSieveExpansionPack1NoCatastrophicFamilyStrategy,
):
    """
    Trader question:
    Can expansion pack 1 become usable if its higher-risk entries are protected
    after profit milestones instead of only filtered by family removal?
    """


class TraderRuleBlockBtcSieveOvertradeRescueExpansionPack1PositiveProfitLockStrategy(
    _ProfitLockPartialExitMixin,
    TraderRuleBlockBtcSieveExpansionPack1PositiveFamiliesStrategy,
):
    """
    Trader question:
    Do the already-positive expansion pack 1 families improve further when
    profitable trades are protected with partial exits and profit-lock stops?
    """


class TraderRuleBlockBtcSieveOvertradeRescueExpansionPack1NoTwoWeakTightProfitLockStrategy(
    _ProfitLockPartialExitTightMixin,
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy,
):
    """
    Trader question:
    Does a tighter rescue layer make the previously overtrading merged pack 1
    candidate worth keeping?
    """


class TraderRuleBlockBtcSieveOvertradeRescueExpansionPack1NoTwoWeakRunnerProfitLockStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy,
):
    """
    Trader question:
    Does a later runner-style rescue layer protect the merged pack 1 candidate
    without cutting its winners too early?
    """


class TraderRuleBlockBtcSieveOvertradeRescuePack2NoTop3ProfitLockStrategy(
    _ProfitLockPartialExitMixin,
    TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy,
):
    """
    Trader question:
    Can the cleaned pack 2 merged candidate recover edge if winner protection
    replaces broader entry filtering?
    """


class TraderRuleBlockBtcSieveOvertradeRescuePack2NoPack2WeakRunnerProfitLockStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestPlusPack2NoPack2WeakFamiliesStrategy,
):
    """
    Trader question:
    Does runner-style profit protection help the pack 2 cleaned branch while
    preserving its larger winning trades?
    """


class TraderRuleBlockBtcSieveOvertradeRescueRegimeRangeTightProfitLockStrategy(
    _ProfitLockPartialExitTightMixin,
    TraderRuleBlockBtcSieveRegimeRangeStrategy,
):
    """
    Trader question:
    Can the high-drawdown regime/range lane be rescued by earlier partial
    exits and breakeven/profit-lock stops?
    """


class TraderRuleBlockBtcSieveOvertradeRescueVpRangeTightProfitLockStrategy(
    _ProfitLockPartialExitTightMixin,
    TraderRuleBlockBtcSieveVpRangeStrategy,
):
    """
    Trader question:
    Can the VP range lane become useful if profitable trades are protected
    before they reverse into the old high-drawdown profile?
    """


class TraderRuleBlockBtcSieveOvertradeRescueRangeBreakAgreesTightProfitLockStrategy(
    _ProfitLockPartialExitTightMixin,
    TraderRuleBlockBtcSieveRangeBreakAgreesStrategy,
):
    """
    Trader question:
    Can the range-break-agreement lane be rescued by treating early profit as
    capital to protect, not as a reason to wait for the old exit?
    """


class TraderRuleBlockBtcSieveOvertradeRescueQualitySelectiveTradeManagementRunnerStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveQualitySelectiveTradeManagementStrategy,
):
    """
    Trader question:
    Does the older selective trade-management lane improve if it adds
    profit-locking instead of relying only on invalidation exits?
    """


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanRunnerProfitLockStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
):
    """
    Trader question:
    Does the current best clean-entry lane keep its return while adding
    production-style runner protection?
    """


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanProfitLockStrategy(
    _ProfitLockPartialExitMixin,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
):
    """
    Trader question:
    Does the current best clean-entry lane improve if winners take staged
    partial profit and the remaining position is protected by profit-lock
    stops?
    """


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanTightProfitLockStrategy(
    _ProfitLockPartialExitTightMixin,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
):
    """
    Trader question:
    Does earlier partial profit and breakeven protection reduce risk enough to
    justify cutting winners sooner on the clean-entry lane?
    """


class _ConservativeDryRunLeverageMixin:
    """
    Dry-run safety helper.

    Futures dry-run/live use should start from 1x unless the user explicitly
    approves higher leverage after observing real behaviour.
    """

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, entry_tag, side, kwargs
        return min(1.0, float(max_leverage))


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanDryRunBaseStrategy(
    _ConservativeDryRunLeverageMixin,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
):
    """
    Dry-run base candidate for the selected BTC lane.

    It preserves the best reproduced entry/exit logic and only adds an
    operational 1x leverage cap. Use this when profit-lock variants fail to
    improve the anchor.
    """


class TraderRuleBlockBtcSieveCurrentBestOvertradeCleanDryRunCandidateStrategy(
    _ConservativeDryRunLeverageMixin,
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestOvertradeResBreakCleanEntryStrategy,
):
    """
    Dry-run candidate wrapper for the selected clean-entry BTC lane.

    It keeps the selected entry logic, adds runner-style profit protection,
    and caps leverage at 1x for operational safety.
    """


class TraderRuleBlockBtcSieveCurrentBestNoNegativeAbsRunnerProfitLockStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy,
):
    """
    Trader question:
    Does the higher-win-rate no-negative-ABS branch become a better balanced
    candidate with later partial exits and profit-lock stops?
    """


class TraderRuleBlockBtcSieveCurrentBestBalancedPocShortRunnerProfitLockStrategy(
    _ProfitLockPartialExitRunnerMixin,
    TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakAndPocShortStrategy,
):
    """
    Trader question:
    Does the high-win-rate balanced POC-short branch improve with runner-style
    profit protection?
    """
