from __future__ import annotations

from TraderRuleBlockSupportRemovalSupportClearedStrategy import TraderRuleBlockSupportRemovalSupportClearedStrategy


class TraderRuleBlockSupportRemovalSupportClearedConfluenceStrategy(TraderRuleBlockSupportRemovalSupportClearedStrategy):
    """
    Research-only validation for support-cleared shorts after selected
    structure/VP confluence filters.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_support_removal_selected_support_cleared.parquet"
    )
