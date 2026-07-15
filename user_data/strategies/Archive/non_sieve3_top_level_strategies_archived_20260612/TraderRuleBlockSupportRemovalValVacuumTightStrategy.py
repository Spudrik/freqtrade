from __future__ import annotations

from TraderRuleBlockOrderbookSupportRemovalShortStrategy import TraderRuleBlockOrderbookSupportRemovalShortStrategy


class TraderRuleBlockSupportRemovalValVacuumTightStrategy(TraderRuleBlockOrderbookSupportRemovalShortStrategy):
    """
    Research-only validation for the quick VAL/vacuum short sub-lead.

    Uses the tighter stop suggested by the exit screen because this behaviour
    should either follow through quickly or be invalidated.
    """

    stoploss = -0.014

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_support_removal_val_vacuum_20260605.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.025, "hold_hours": 6.0},
    }
