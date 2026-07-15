from __future__ import annotations

try:
    from .TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy
except ImportError:  # pragma: no cover - Freqtrade direct strategy loading fallback
    from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockConfluenceResearchStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only variant using the best per-rule confluence-filtered signal set.

    This keeps the baseline strategy reproducible while allowing filtered entries
    to be tested in the Freqtrade engine.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260605_best_per_rule_selected_confluence.parquet"
    )

    stoploss = -0.035

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_pattern_rectangle_breakout_long": {"take_profit": 0.075, "hold_hours": 48.0, "research_stop_loss": 0.035},
        "sieve_prior_day_low_break_vp_short": {"take_profit": 0.020, "hold_hours": 48.0, "research_stop_loss": 0.020},
    }
