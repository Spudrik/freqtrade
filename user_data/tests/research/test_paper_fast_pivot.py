"""Focused paper safety and selected-Sieve-contract checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from freqtrade.enums import RunMode
from user_data.strategies.paper_fast_pivot import PaperFastPivot, SELECTED_SELL_PARAMS


def _config() -> dict:
    with open("user_data/configs/integrated_paper_20260926_base.json", encoding="utf-8") as handle:
        config = json.load(handle)
    with open("user_data/configs/integrated_paper_fast_pivot.json", encoding="utf-8") as handle:
        config.update(json.load(handle))
    config["runmode"] = RunMode.DRY_RUN
    return config


def test_selected_entry_exit_and_paper_account() -> None:
    strategy = PaperFastPivot(_config())
    assert strategy.SOURCE_ENTRY_STEM == "pivot_midrange_reject_short_1h"
    assert {name: getattr(strategy, name).value for name in SELECTED_SELL_PARAMS} == SELECTED_SELL_PARAMS
    assert strategy._focused_plan()["hard_stop_ratio"] == pytest.approx(0.02)
    assert strategy.stoploss == -0.02
    strategy.bot_start()


def test_live_mode_credentials_and_order_api_are_refused() -> None:
    for field, value in (("dry_run", False), ("runmode", RunMode.LIVE), ("force_entry_enable", True)):
        config = _config()
        config[field] = value
        with pytest.raises(RuntimeError):
            PaperFastPivot(config).bot_start()
    config = _config()
    config["exchange"]["key"] = "unexpected"
    with pytest.raises(RuntimeError, match="credentials"):
        PaperFastPivot(config).bot_start()


def test_stake_is_bounded_and_requires_a_fresh_decision() -> None:
    strategy = PaperFastPivot(_config())
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.0)
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    tag = "pivot_midrange_reject_short_1h"
    args = ("BTC/USDT:USDT", now, 100.0, 3333.0, 10.0, 3333.0, 1.0, tag, "short")
    assert not strategy.confirm_trade_entry("BTC/USDT:USDT", "market", 2.0, 100.0, "gtc", now, tag, "short")
    assert strategy.custom_stake_amount(*args) == pytest.approx(200.0)
    assert strategy.confirm_trade_entry("BTC/USDT:USDT", "market", 2.0, 100.0, "gtc", now, tag, "short")
    assert not strategy.confirm_trade_entry("BTC/USDT:USDT", "market", 2.0, 100.0, "gtc", now, tag, "short")
