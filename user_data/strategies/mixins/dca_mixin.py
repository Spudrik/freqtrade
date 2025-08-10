from __future__ import annotations

from freqtrade.persistence import Trade


class DCAMixin:
    """Mixin providing scaled DCA entry calculation."""

    # Scaling factors for the position after each entry.
    # The first element (index 0) represents the target total position size after
    # the first filled order (initial entry). Further elements represent the
    # target total size after each subsequent DCA order.
    dca_scaling: list[float] = [1.0, 1.5, 2.25]

    def calculate_scaled_dca_entries(self, trade: Trade) -> float | None:
        """Return stake difference for next DCA order.

        The method uses :attr:`dca_scaling` to determine the desired total stake
        after the next DCA step. ``trade.nr_of_successful_entries`` indicates how
        many entries have already been filled. The returned value is the
        additional stake amount required to scale the position to the desired
        size for the next step. ``None`` is returned when no further scaling is
        configured.
        """
        filled_entries = trade.nr_of_successful_entries
        if filled_entries >= len(self.dca_scaling):
            return None

        # Desired total stake after executing next DCA step
        target_total = trade.stake_amount * self.dca_scaling[filled_entries]
        # Current stake in position
        current_stake = trade.stake_amount * filled_entries
        # Difference is the additional stake needed for next order
        return max(0.0, target_total - current_stake)
