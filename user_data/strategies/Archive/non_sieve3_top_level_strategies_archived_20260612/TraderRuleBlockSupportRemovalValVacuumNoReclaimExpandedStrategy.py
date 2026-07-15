from __future__ import annotations

from TraderRuleBlockSupportRemovalValVacuumTightStrategy import TraderRuleBlockSupportRemovalValVacuumTightStrategy


class TraderRuleBlockSupportRemovalValVacuumNoReclaimExpandedStrategy(TraderRuleBlockSupportRemovalValVacuumTightStrategy):
    """
    Research-only validation for the expanded no-reclaim VAL/vacuum short.

    This uses the looser no-reclaim threshold found by the expansion sweep:
    more trades than the clean 9-trade no-reclaim lead, but still focused on
    broken support not being reclaimed.
    """

    stoploss = -0.010

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_no_reclaim_lte_46.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.035, "hold_hours": 8.0},
    }
