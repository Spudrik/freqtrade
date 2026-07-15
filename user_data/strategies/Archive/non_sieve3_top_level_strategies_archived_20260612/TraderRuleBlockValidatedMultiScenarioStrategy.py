from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockValidatedMultiScenarioStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation for the current mixed long/short lead block.

    This combines the validated orderbook short block, the prior-day-low VP
    short, and the sparse rectangle/VP breakout long. Signals are prebuilt from
    frozen 1h research features; no live collectors or slow indicators run here.
    """

    can_short = True
    use_custom_stoploss = True
    stoploss = -0.035

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_validated_multi_scenario_block.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "lead__validated__support_break_support_cleared_short": {
            "take_profit": 0.050,
            "hold_hours": 18.0,
            "stop_loss": 0.025,
        },
        "lead__validated__val_vacuum_no_support_reclaim_warning_short": {
            "take_profit": 0.035,
            "hold_hours": 8.0,
            "stop_loss": 0.010,
        },
        "sieve_prior_day_low_break_vp_short": {
            "take_profit": 0.020,
            "hold_hours": 48.0,
            "stop_loss": 0.020,
        },
        "sieve_pattern_rectangle_breakout_long": {
            "take_profit": 0.075,
            "hold_hours": 48.0,
            "stop_loss": 0.035,
        },
    }

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
