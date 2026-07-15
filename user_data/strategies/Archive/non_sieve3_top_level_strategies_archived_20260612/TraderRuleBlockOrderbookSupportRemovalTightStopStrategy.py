from __future__ import annotations

from TraderRuleBlockOrderbookSupportRemovalShortStrategy import TraderRuleBlockOrderbookSupportRemovalShortStrategy


class TraderRuleBlockOrderbookSupportRemovalTightStopStrategy(TraderRuleBlockOrderbookSupportRemovalShortStrategy):
    """
    Research-only validation of the support-removal short block with a tighter
    global stop.

    The direct exit screen preferred a 1.4% stop for the quick VAL/vacuum short.
    This checks whether that practical compromise improves the Freqtrade result
    before adding more complex per-rule stop handling.
    """

    stoploss = -0.014
