from __future__ import annotations

from TraderRuleBlockValidatedOrderbookCombinedStrategy import TraderRuleBlockValidatedOrderbookCombinedStrategy


class TraderRuleBlockValidatedOrderbookCoreStrategy(TraderRuleBlockValidatedOrderbookCombinedStrategy):
    """
    Research-only validation for the cleaner core orderbook short block.

    This keeps only the genuinely separate validated leads:
    support-cleared breakdown shorts and strict no-reclaim VAL/vacuum shorts.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_validated_orderbook_overlap_core_validated_orderbook.parquet"
    )
