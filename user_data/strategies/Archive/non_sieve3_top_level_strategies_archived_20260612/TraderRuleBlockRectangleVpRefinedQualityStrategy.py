from __future__ import annotations

from TraderRuleBlockRectangleVpRefinementStrategy import TraderRuleBlockRectangleVpRefinementStrategy


class TraderRuleBlockRectangleVpRefinedQualityStrategy(TraderRuleBlockRectangleVpRefinementStrategy):
    """
    Research-only validation for the refined rectangle/VP quality subset.

    Excludes the broad compression-volume long because Freqtrade validation
    showed it added drawdown and little edge.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_rectangle_vp_refinement_20260605_quality.parquet"
    )
