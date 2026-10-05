"""Paper-only, higher-frequency Sieve pivot-rejection candidate.

The entry and exit are the locked Sieve3 V2 source and its selected validation
parameters.  This adapter changes only paper-account safety and position size.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from math import isfinite

from freqtrade.enums import RunMode

from user_data.strategies.sieve3_V2_profit_ladder_three_stage_ratchet_from_pivot_midrange_reject_short_1h import (
    Sieve3V2ProfitLadderThreeStageRatchetFromPivotMidrangeRejectShort1h,
)


LOG = logging.getLogger(__name__)
SELECTED_SELL_PARAMS = {
    "target_1_percent": 4,
    "target_gap_half_percent_units": 2,
    "partial_1_five_percent_units": 8,
    "partial_2_five_percent_units": 8,
    "hard_stop_percent": 2,
}


class PaperFastPivot(Sieve3V2ProfitLadderThreeStageRatchetFromPivotMidrangeRejectShort1h):
    """Test whether a frequent, historically positive pivot short survives paper execution."""

    stoploss = -0.02  # Emergency floor matches the selected Sieve hard stop.
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if set(SELECTED_SELL_PARAMS) != set(self.ACTIVE_SELL_PARAMS):
            raise RuntimeError("Selected Sieve exit parameters no longer match source")
        for name, value in SELECTED_SELL_PARAMS.items():
            parameter = getattr(self, name)
            if not parameter.low <= value <= parameter.high:
                raise RuntimeError(f"Selected Sieve exit parameter is invalid: {name}={value}")
            parameter.value = value
        self._pending_stake: dict[tuple[str, str, str], tuple[datetime, float]] = {}

    def bot_start(self, **kwargs) -> None:
        _ = kwargs
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperFastPivot is dry-run only")
        if self.config.get("bot_name") != "paper_fast_pivot":
            raise RuntimeError("Unknown paper account")
        if self.config.get("trading_mode") != "futures" or self.config.get("exchange", {}).get("name") != "binance":
            raise RuntimeError("PaperFastPivot requires the Binance futures paper configuration")
        if any(self.config.get("exchange", {}).get(key) for key in ("key", "secret")):
            raise RuntimeError("Paper account must not contain exchange credentials")
        if self.config.get("force_entry_enable") or self.config.get("api_server", {}).get("enabled"):
            raise RuntimeError("PaperFastPivot forbids force entries and an order API")
        if {name: getattr(self, name).value for name in SELECTED_SELL_PARAMS} != SELECTED_SELL_PARAMS:
            raise RuntimeError("Selected Sieve exit parameters drifted at startup")

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        _ = (current_rate, proposed_stake, leverage, kwargs)
        key = (pair, side, str(entry_tag))
        self._pending_stake.pop(key, None)
        equity = float(self.wallets.get_total_stake_amount())
        stake = min(equity * 0.02, float(max_stake))
        if not isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            return 0.0
        self._pending_stake[key] = (current_time, stake)
        LOG.info("PaperFastPivot stake pair=%s side=%s amount=%.8f", pair, side, stake)
        return stake

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        _ = (order_type, amount, rate, time_in_force, kwargs)
        pending = self._pending_stake.pop((pair, side, str(entry_tag)), None)
        if pending is None or not timedelta(0) <= current_time - pending[0] <= timedelta(seconds=30):
            LOG.error("PaperFastPivot refused entry without a fresh sizing decision")
            return False
        return pending[1] > 0
