from __future__ import annotations

from TraderRuleBlockSupportRemovalValVacuumTightStrategy import TraderRuleBlockSupportRemovalValVacuumTightStrategy


class TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy(TraderRuleBlockSupportRemovalValVacuumTightStrategy):
    """
    Research-only validation for quick VAL/vacuum shorts when the book does
    not warn that broken support is being reclaimed.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_val_vacuum_no_reclaim.parquet"
    )
