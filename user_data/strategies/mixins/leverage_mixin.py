"""Mixin adding leverage rules for futures strategies."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from freqtrade.wallets import Wallets


class LeverageMixin:
    """Provide dynamic leverage selection for strategies.

    The mixin implements the leverage rules from the original strategy.
    It targets a constant position size by adjusting leverage according to the
    stake calculated for the trade.  Pair specific overrides can be provided
    through ``leverage_per_pair``.
    """

    # Desired position value (stake * leverage) for each trade in stake currency.
    target_position_size: float = 100.0

    # Optional per-pair fixed leverage values. Example: {"BTC/USDT": 3.0}
    leverage_per_pair: dict[str, float] = {}

    # Attributes provided by IStrategy at runtime.
    wallets: Wallets
    config: dict[str, Any]

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Return leverage for a new trade.

        Leverage is capped by the exchange provided ``max_leverage`` and is
        calculated so that the resulting position value stays around
        ``target_position_size``.
        """

        # Use the stake amount that will be used for this trade to keep leverage
        # and stake calculation coherent.
        stake_amount = self.wallets.get_trade_stake_amount(pair, self.config["max_open_trades"])

        # Resolve leverage: per-pair override or dynamic calculation.
        lev = self.leverage_per_pair.get(pair)
        if lev is None and stake_amount > 0:
            lev = self.target_position_size / stake_amount
        elif lev is None:
            lev = proposed_leverage

        # Ensure leverage stays within sensible boundaries.
        lev = max(1.0, min(lev, max_leverage))
        return lev
