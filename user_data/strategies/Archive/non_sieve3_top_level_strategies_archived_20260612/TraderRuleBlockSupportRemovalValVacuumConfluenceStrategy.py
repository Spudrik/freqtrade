from __future__ import annotations

from TraderRuleBlockSupportRemovalValVacuumTightStrategy import TraderRuleBlockSupportRemovalValVacuumTightStrategy


class TraderRuleBlockSupportRemovalValVacuumConfluenceStrategy(TraderRuleBlockSupportRemovalValVacuumTightStrategy):
    """
    Research-only validation for quick VAL/vacuum shorts after the selected
    regime confluence filter.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_support_removal_selected_val_vacuum.parquet"
    )
