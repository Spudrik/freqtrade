"""Common hyperopt parameter spaces for strategies."""

from __future__ import annotations

from typing import Dict, List

from freqtrade.optimize.space import Integer, SKDecimal
from freqtrade.strategy import IntParameter, DecimalParameter


class HyperoptMixin:
    """Mixin supplying reusable hyperopt parameter spaces."""

    # Hyperopt parameter spaces as class attributes
    buy_rsi = IntParameter(10, 40, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(60, 90, default=70, space="sell", optimize=True)
    buy_adx = IntParameter(5, 40, default=20, space="buy", optimize=True)
    sell_adx = IntParameter(5, 40, default=20, space="sell", optimize=True)
    stoploss = DecimalParameter(-0.10, -0.02, default=-0.05, space="sell", optimize=True)

    class HyperOpt:
        """Container for spaces consumed by Freqtrade's hyperopt."""

        @staticmethod
        def stoploss_space() -> List[SKDecimal]:
            return [SKDecimal(-0.10, -0.02, decimals=3, name="stoploss")]

        @staticmethod
        def roi_space() -> List[Integer | SKDecimal]:
            return [
                Integer(10, 120, name="roi_t1"),
                Integer(10, 60, name="roi_t2"),
                Integer(10, 40, name="roi_t3"),
                SKDecimal(0.01, 0.04, decimals=3, name="roi_p1"),
                SKDecimal(0.01, 0.07, decimals=3, name="roi_p2"),
                SKDecimal(0.01, 0.20, decimals=3, name="roi_p3"),
            ]

        @staticmethod
        def generate_roi_table(params: Dict) -> Dict[int, float]:
            roi_table: Dict[int, float] = {}
            roi_table[0] = params["roi_p1"] + params["roi_p2"] + params["roi_p3"]
            roi_table[params["roi_t3"]] = params["roi_p1"] + params["roi_p2"]
            roi_table[params["roi_t3"] + params["roi_t2"]] = params["roi_p1"]
            roi_table[
                params["roi_t3"] + params["roi_t2"] + params["roi_t1"]
            ] = 0
            return roi_table
