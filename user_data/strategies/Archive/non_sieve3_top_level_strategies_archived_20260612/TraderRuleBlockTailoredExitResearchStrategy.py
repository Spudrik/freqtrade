from __future__ import annotations

try:
    from .TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy
except ImportError:  # pragma: no cover - Freqtrade direct strategy loading fallback
    from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockTailoredExitResearchStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only variant of TraderRuleBlockResearchStrategy with tailored exits
    selected by trading_lead_exit_research.py.

    The baseline TraderRuleBlockResearchStrategy keeps the earlier fixed-exit
    setup so both backtest results remain reproducible.
    """

    stoploss = -0.035

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_pattern_rectangle_breakdown_short": {"take_profit": 0.050, "hold_hours": 10.0, "research_stop_loss": 0.025},
        "sieve_prior_day_high_break_vp_long": {"take_profit": 0.020, "hold_hours": 72.0, "research_stop_loss": 0.035},
        "sieve_prior_day_low_break_vp_short": {"take_profit": 0.050, "hold_hours": 36.0, "research_stop_loss": 0.035},
        "sieve_pattern_rectangle_breakout_long": {"take_profit": 0.075, "hold_hours": 48.0, "research_stop_loss": 0.035},
    }
