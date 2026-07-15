from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockValidatedOrderbookCoreStrategy import TraderRuleBlockValidatedOrderbookCoreStrategy


class TraderRuleBlockValidatedOrderbookCoreTailoredExitStrategy(TraderRuleBlockValidatedOrderbookCoreStrategy):
    """
    Research-only validation for the core orderbook block with per-lead exits.

    The base core strategy can keep per-lead targets and hold times, but Freqtrade
    has one static stoploss unless custom stoploss is enabled. This class applies
    the stop selected by the exit sweep for each lead id.
    """

    use_custom_stoploss = True
    stoploss = -0.025

    rule_exit_settings: dict[str, dict[str, float]] = {
        "lead__validated__support_break_support_cleared_short": {
            "take_profit": 0.050,
            "hold_hours": 18.0,
            "stop_loss": 0.025,
        },
        "lead__validated__val_vacuum_no_support_reclaim_warning_short": {
            "take_profit": 0.035,
            "hold_hours": 8.0,
            "stop_loss": 0.010,
        },
    }

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        _ = pair, current_time, current_rate, after_fill, kwargs
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        settings = self.rule_exit_settings.get(rule_id)
        if settings is None:
            return None
        stop_loss = float(settings.get("stop_loss", abs(self.stoploss)))
        return stoploss_from_open(-stop_loss, current_profit, is_short=trade.is_short, leverage=trade.leverage)
