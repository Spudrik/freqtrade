from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockOrderbookSupportRemovalShortStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation of orderbook support-removal short refinements.

    Signals are prebuilt from the frozen 1h confluence feature cache by
    trading_lead_orderbook_support_removal_short_refinement.py.
    """

    stoploss = -0.025

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_orderbook_support_removal_short_refinement_20260605_selected.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "support_break_support_cleared_short": {"take_profit": 0.050, "hold_hours": 18.0},
        "val_break_downside_vacuum_after_support_removed_short": {"take_profit": 0.025, "hold_hours": 6.0},
    }
