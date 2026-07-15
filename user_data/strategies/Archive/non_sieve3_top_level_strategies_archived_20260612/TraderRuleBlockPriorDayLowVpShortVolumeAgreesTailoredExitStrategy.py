from __future__ import annotations

from TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy import TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy


class TraderRuleBlockPriorDayLowVpShortVolumeAgreesTailoredExitStrategy(
    TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy
):
    """
    Research-only validation for prior-day-low VP short with the selected
    downside-continuation exit from the exit sweep.
    """

    stoploss = -0.025

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve_prior_day_low_break_vp_short": {"take_profit": 0.050, "hold_hours": 48.0}
    }
