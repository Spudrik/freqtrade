from __future__ import annotations

from datetime import datetime

from freqtrade.strategy import Trade, stoploss_from_absolute


class StoplossMixin:
    """Mixin providing custom stoploss logic."""

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """Return stoploss distance relative to current_rate.

        Uses the Parabolic SAR value from the analyzed dataframe as an absolute stop
        price and converts this to a relative stoploss for the current rate.  The
        strategy using this mixin must populate the ``sar`` indicator within its
        ``populate_indicators`` method.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        stoploss_price = last_candle.get("sar")

        if stoploss_price and stoploss_price < current_rate:
            return stoploss_from_absolute(
                stoploss_price,
                current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )
        return None
