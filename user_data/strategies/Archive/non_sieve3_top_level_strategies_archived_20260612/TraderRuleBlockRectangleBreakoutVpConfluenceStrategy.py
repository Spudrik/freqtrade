from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockRectangleBreakoutVpConfluenceStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation for the isolated rectangle breakout long lead.

    This uses only the VP-supported rectangle breakout long signals selected by
    the per-rule confluence screen. It deliberately excludes the prior-day-low
    short filter so the validation answers one trader question.
    """

    can_short = False
    stoploss = -0.035

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_tailored_exit_selected_confluence.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_pattern_rectangle_breakout_long": {"take_profit": 0.075, "hold_hours": 48.0}
    }
