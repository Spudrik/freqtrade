"""Shared, paper-only position sizing and exits for the 2026-09 trial.

These classes are not Sieve/Hyperopt candidates.  Keep trading logic in the
specific paper strategy modules and preserve the common risk contract here.
"""

from __future__ import annotations

from datetime import timedelta
from math import isfinite

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class PaperTrialBase(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    startup_candle_count = 400
    minimal_roi = {"0": 100.0}
    stoploss = -0.25  # Emergency backstop; normal exits use entry-frozen ATR.
    use_custom_stoploss = True
    use_exit_signal = True
    trailing_stop = False
    process_only_new_candles = True
    position_adjustment_enable = False
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}
    paper_hold_hours = 24

    def leverage(
        self, pair, current_time, current_rate, proposed_leverage,
        max_leverage, entry_tag, side, **kwargs,
    ) -> float:
        return 1.0

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake,
        min_stake, max_stake, leverage, entry_tag, side, **kwargs,
    ) -> float:
        equity = float(self.wallets.get_total_stake_amount())
        stake = min(equity * 0.02, float(max_stake))
        if not isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            return 0.0
        return stake

    def order_filled(self, pair, trade: Trade, order, current_time, **kwargs) -> None:
        if order.ft_order_side != trade.entry_side or trade.get_custom_data("paper_entry_atr"):
            return
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty or "paper_atr" not in frame:
            raise RuntimeError(f"Missing entry ATR for paper trade {trade.id} on {pair}")
        atr = float(frame.iloc[-1]["paper_atr"])
        if not isfinite(atr) or atr <= 0:
            raise RuntimeError(f"Invalid entry ATR for paper trade {trade.id} on {pair}")
        trade.set_custom_data(key="paper_entry_atr", value=atr)

    @staticmethod
    def _entry_atr(trade: Trade) -> float:
        value = trade.get_custom_data("paper_entry_atr")
        if value is None or not isfinite(float(value)) or float(value) <= 0:
            raise RuntimeError(f"Paper trade {trade.id} has no valid entry ATR")
        return float(value)

    def custom_stoploss(
        self, pair, trade: Trade, current_time, current_rate,
        current_profit, after_fill, **kwargs,
    ) -> float:
        atr = self._entry_atr(trade)
        stop_price = trade.open_rate + (1.5 * atr if trade.is_short else -1.5 * atr)
        return stoploss_from_absolute(
            stop_price, current_rate, is_short=trade.is_short, leverage=trade.leverage,
        )

    def custom_exit(
        self, pair, trade: Trade, current_time, current_rate, current_profit, **kwargs,
    ) -> str | None:
        atr = self._entry_atr(trade)
        target = trade.open_rate + (-3.0 * atr if trade.is_short else 3.0 * atr)
        if (trade.is_short and current_rate <= target) or (not trade.is_short and current_rate >= target):
            return "paper_atr_target"
        if current_time - trade.open_date_utc >= timedelta(hours=self.paper_hold_hours):
            return "paper_time_limit"
        return None
