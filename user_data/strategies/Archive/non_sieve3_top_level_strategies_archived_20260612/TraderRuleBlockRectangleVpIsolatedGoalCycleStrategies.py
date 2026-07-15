from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class _RectangleVpIsolatedBase(TraderRuleBlockResearchStrategy):
    can_short = True
    use_custom_stoploss = True
    stoploss = -0.025

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
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        settings = self.rule_exit_settings.get(rule_id)
        if settings is None:
            return None
        stop_loss = float(settings.get("stop_loss", abs(self.stoploss)))
        return stoploss_from_open(-stop_loss, current_profit, is_short=trade.is_short, leverage=trade.leverage)


class TraderRuleBlockRectBreakOrderbookAcceptLongStrategy(_RectangleVpIsolatedBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_rect_break_orderbook_accept_long_isolated.parquet"
    )
    rule_exit_settings = {
        "rect_break_orderbook_accept_long": {"take_profit": 0.055, "hold_hours": 24.0, "stop_loss": 0.025}
    }


class TraderRuleBlockRectSqueezeUpperReleaseLongStrategy(_RectangleVpIsolatedBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_rect_squeeze_upper_release_long_isolated.parquet"
    )
    rule_exit_settings = {
        "rect_squeeze_upper_release_long": {"take_profit": 0.055, "hold_hours": 24.0, "stop_loss": 0.025}
    }


class TraderRuleBlockRectBreakAboveVahLongStrategy(_RectangleVpIsolatedBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_rect_break_above_vah_long_isolated.parquet"
    )
    stoploss = -0.018
    rule_exit_settings = {
        "rect_break_above_vah_long": {"take_profit": 0.035, "hold_hours": 12.0, "stop_loss": 0.018}
    }


class TraderRuleBlockRectBreakBelowValShortStrategy(_RectangleVpIsolatedBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_rect_break_below_val_short_isolated.parquet"
    )
    rule_exit_settings = {
        "rect_break_below_val_short": {"take_profit": 0.055, "hold_hours": 24.0, "stop_loss": 0.025}
    }
