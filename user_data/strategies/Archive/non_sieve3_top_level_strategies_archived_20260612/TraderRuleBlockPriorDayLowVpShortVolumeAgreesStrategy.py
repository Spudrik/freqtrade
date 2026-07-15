from __future__ import annotations

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation for prior-day-low VP short with directional volume agreement.
    """

    can_short = True
    stoploss = -0.020

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_prior_day_low_vp_short_volume_agrees.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_prior_day_low_break_vp_short": {"take_profit": 0.020, "hold_hours": 48.0}
    }
