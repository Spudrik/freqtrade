from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockSupportBreakGoalCycleStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation for the 2026-06-06 TLV2/support-break short cycle.
    """

    can_short = True
    use_custom_stoploss = True
    stoploss = -0.025

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_goal_cycle_support_break_selected.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "near_tlv2_support_breakdown_pressure_short": {
            "take_profit": 0.055,
            "hold_hours": 24.0,
            "stop_loss": 0.025,
        },
        "support_break_orderbook_accept_short": {
            "take_profit": 0.055,
            "hold_hours": 24.0,
            "stop_loss": 0.025,
        },
        "tlv2_support_break_vp_opposition_low_short": {
            "take_profit": 0.050,
            "hold_hours": 18.0,
            "stop_loss": 0.025,
        },
        "structure_breakdown_vp_opposition_low_short": {
            "take_profit": 0.035,
            "hold_hours": 8.0,
            "stop_loss": 0.018,
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
