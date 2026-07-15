from __future__ import annotations

from TraderRuleBlockRectangleVpExpansionStrategy import TraderRuleBlockRectangleVpExpansionStrategy


class TraderRuleBlockRectangleVpQualityStrategy(TraderRuleBlockRectangleVpExpansionStrategy):
    """
    Research-only validation of the higher-quality rectangle/VP subset.

    This excludes the broad compression-range variant that supplied many trades
    but weak average edge in the first expanded validation.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_rectangle_vp_expansion_20260605_quality.parquet"
    )
