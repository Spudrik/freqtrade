from __future__ import annotations

from typing import Any


class TakeProfitMixin:
    """Mixin providing per-trade take-profit handling.

    The mixin stores a take-profit ratio for each trade and only updates the
    value when it changes.  Ratios that would violate the current stop-loss are
    ignored.  The method returns the new ratio when it changes, otherwise
    ``None`` which allows the strategy to retain the previous value.
    """

    def __init__(self) -> None:
        # Cache of take-profit ratios per trade
        self._tp_cache: dict[int, float] = {}

    def set_tp(self, trade: Any, ratio: float | None, stop_loss: float) -> float | None:
        """Set a take-profit ratio for ``trade``.

        Args:
            trade: Trade instance or any object with an ``id`` attribute. The
                identifier is used to store the take-profit value.
            ratio: Desired take-profit ratio (e.g. ``0.05`` for 5%). ``None``
                removes a previously stored ratio.
            stop_loss: Current stop-loss expressed as a negative ratio.  The
                take-profit must be greater than the absolute stop-loss to be
                considered valid.

        Returns:
            The ratio when a new valid value is stored, otherwise ``None``.
        """

        trade_id = getattr(trade, "id", trade)

        # Reset stored TP when ``None`` is passed
        if ratio is None:
            self._tp_cache.pop(trade_id, None)
            return None

        # Do not allow take profit below or equal to the absolute stop-loss
        if stop_loss is not None and ratio <= abs(stop_loss):
            return None

        previous = self._tp_cache.get(trade_id)
        if previous == ratio:
            return None

        self._tp_cache[trade_id] = ratio
        return ratio
