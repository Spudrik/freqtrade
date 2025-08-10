"""Mixin providing default DCA and partial-close logic for strategies.

The mixin implements :meth:`adjust_trade_position` with a simple approach:

* If a trade is more than 5% in profit and no partial exits happened yet,
  sell half of the position to secure profit.
* If a trade falls more than 5% below average entry price, add to the
  position (DCA) with progressively larger orders.

The returned stake amounts are validated against ``min_stake`` and
``max_stake`` ensuring the logic works for both spot and leveraged
trading.
"""
from __future__ import annotations

from datetime import datetime

from freqtrade.persistence import Trade


class PositionAdjustMixin:
    """Default position adjustment helper.

    Inherit this mixin alongside your strategy to enable basic DCA and
    partial close behaviour.
    """

    # Enable position adjustments for strategies using this mixin
    position_adjustment_enable: bool = True
    # Allow up to 3 additional entries (4 total orders)
    max_entry_position_adjustment: int = 3

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> tuple[float | None, str | None] | float | None:
        """Handle DCA and partial exits for an open trade.

        Returns a positive stake to increase the position (DCA), a
        negative stake to reduce the position (partial close) or ``None``
        to keep the position unchanged.  The stake is capped by
        ``max_stake`` and checked against ``min_stake`` so that exchanges
        requirements and available balance are honoured.  No leverage
        changes are performed - existing trade leverage continues to
        apply for any adjustment.
        """

        # Ignore if there are already open orders for this trade
        if trade.has_open_orders:
            return None

        # Example partial close: take half the position once profit > 5%
        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            stake = trade.stake_amount / 2
            if min_stake is not None and stake < min_stake:
                return None
            return -stake, "half_profit_5%"

        # Only consider additional entries when drawdown exceeds 5%
        if current_profit > -0.05:
            return None

        # Basic protection - avoid buying in a falling candle
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        previous_candle = dataframe.iloc[-2]
        if last_candle["close"] < previous_candle["close"]:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        try:
            # Use size of first entry and increase stake progressively
            stake = filled_entries[0].stake_amount_filled
            stake *= 1 + (count_of_entries * 0.25)
            stake = min(stake, max_stake)
            if min_stake is not None and stake < min_stake:
                return None
            return stake, "1/3rd_increase"
        except Exception:
            return None
