from __future__ import annotations

from TraderRuleBlockOrderbookSupportRemovalShortStrategy import TraderRuleBlockOrderbookSupportRemovalShortStrategy


class TraderRuleBlockSupportRemovalSupportClearedStrategy(TraderRuleBlockOrderbookSupportRemovalShortStrategy):
    """
    Research-only validation for the support-cleared short sub-lead.

    Split from the broader support-removal block because the quick VAL/vacuum
    short wanted a tighter stop than this support-cleared behaviour.
    """

    stoploss = -0.025

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_support_removal_support_cleared_20260605.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "support_break_support_cleared_short": {"take_profit": 0.050, "hold_hours": 18.0},
    }
