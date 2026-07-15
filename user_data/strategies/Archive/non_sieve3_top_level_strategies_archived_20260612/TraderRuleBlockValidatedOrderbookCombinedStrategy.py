from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockValidatedOrderbookCombinedStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation for the non-overlapped validated orderbook short block.

    Signals are prebuilt from frozen lead parquets by
    trading_lead_validated_overlap.py. The signal tag is the lead id, which
    lets each validated lead keep its own exit profile.
    """

    can_short = True
    stoploss = -0.025

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_validated_orderbook_overlap_combined_validated_orderbook.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "lead__validated__support_break_support_cleared_short": {"take_profit": 0.050, "hold_hours": 18.0},
        "lead__validated__val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.025, "hold_hours": 6.0},
        "lead__validated__val_vacuum_no_support_reclaim_warning_short": {"take_profit": 0.025, "hold_hours": 6.0},
        "lead__validated__val_vacuum_loose_no_reclaim_short": {"take_profit": 0.035, "hold_hours": 8.0},
        "lead__validated__val_vacuum_bearish_pressure_short": {"take_profit": 0.035, "hold_hours": 8.0},
    }
