from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockRectangleVpExpansionStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation strategy for expanded rectangle/compression + VP signals.

    Signals are prebuilt from the frozen confluence parquet by
    trading_lead_rectangle_vp_expansion.py. This strategy only loads those
    signals and applies simple per-rule take-profit/time exits.
    """

    stoploss = -0.035

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_rectangle_vp_expansion_20260605_selected.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "rect_break_orderbook_accept_long": {"take_profit": 0.055, "hold_hours": 24.0},
        "rect_squeeze_upper_release_long": {"take_profit": 0.055, "hold_hours": 24.0},
        "rect_break_above_vah_long": {"take_profit": 0.035, "hold_hours": 12.0},
        "rect_break_below_val_short": {"take_profit": 0.055, "hold_hours": 24.0},
        "rect_break_lvn_thin_air_long": {"take_profit": 0.050, "hold_hours": 10.0},
        "rect_multi_tf_upper_break_long": {"take_profit": 0.055, "hold_hours": 24.0},
        "rect_1d_upper_break_vp_long": {"take_profit": 0.050, "hold_hours": 10.0},
        "compression_range_break_vp_long": {"take_profit": 0.075, "hold_hours": 48.0},
        "rect_squeeze_lower_release_short": {"take_profit": 0.050, "hold_hours": 10.0},
    }
