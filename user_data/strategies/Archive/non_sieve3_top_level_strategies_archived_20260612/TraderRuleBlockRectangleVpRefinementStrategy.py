from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockRectangleVpRefinementStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation strategy for refined rectangle/VP signals.

    Signals are prebuilt from the frozen 1h confluence feature cache by
    trading_lead_rectangle_vp_refinement.py. No source collection or slow
    indicator computation happens inside the strategy.
    """

    stoploss = -0.035

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_rectangle_vp_refinement_20260605_selected.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "compression_break_vp_long_volume_persistent": {"take_profit": 0.075, "hold_hours": 48.0},
        "rect_orderbook_accept_long_quality_keep": {"take_profit": 0.055, "hold_hours": 24.0},
        "rect_squeeze_long_quality_keep": {"take_profit": 0.075, "hold_hours": 48.0},
        "rect_val_short_vp_opposition_low": {"take_profit": 0.050, "hold_hours": 18.0},
        "rect_val_short_range_mid_break": {"take_profit": 0.050, "hold_hours": 18.0},
        "rect_val_short_moderate_volume_break": {"take_profit": 0.050, "hold_hours": 10.0},
        "compression_break_vp_long_value_acceptance": {"take_profit": 0.055, "hold_hours": 24.0},
    }
