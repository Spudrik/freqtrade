"""Paper E: untouched level rule plus explicitly journalled manual paper actions."""

from __future__ import annotations

from user_data.strategies.paper_trial_level import PaperTrialLevel


class PaperTrialLevelOverlay(PaperTrialLevel):
    # Allows /forceenter to add to an existing paper trade. No automatic adds.
    position_adjustment_enable = True
    max_entry_position_adjustment = 1

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        return None
