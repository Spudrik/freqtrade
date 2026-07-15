from __future__ import annotations

from TraderRuleBlockSupportRemovalValVacuumTightStrategy import TraderRuleBlockSupportRemovalValVacuumTightStrategy


class TraderRuleBlockSupportRemovalValVacuumPressureExpandedStrategy(TraderRuleBlockSupportRemovalValVacuumTightStrategy):
    """
    Research-only validation for VAL/vacuum shorts where bearish pressure
    remains present after support removal.

    This is a broader adjacent lead to the no-reclaim version, not a direct
    replacement for it.
    """

    stoploss = -0.010

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_pressure_gte_20.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.035, "hold_hours": 8.0},
    }
