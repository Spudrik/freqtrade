from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockSupportBreakShortExpansionStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation of adjacent support-break short expansion leads.

    Signals are prebuilt from the frozen 1h confluence feature cache by
    trading_lead_support_break_short_expansion.py.
    """

    stoploss = -0.025

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_support_break_short_expansion_20260605_selected.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "near_tlv2_support_breakdown_pressure_short": {"take_profit": 0.055, "hold_hours": 24.0},
        "support_break_orderbook_accept_short": {"take_profit": 0.055, "hold_hours": 24.0},
        "tlv2_support_break_vp_opposition_low_short": {"take_profit": 0.050, "hold_hours": 18.0},
        "structure_breakdown_vp_opposition_low_short": {"take_profit": 0.035, "hold_hours": 8.0},
    }
