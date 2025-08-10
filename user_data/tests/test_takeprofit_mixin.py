# ruff: noqa: S101
from user_data.strategies.mixins.takeprofit_mixin import TakeProfitMixin


class DummyTrade:
    def __init__(self, trade_id: int):
        self.id = trade_id


def test_set_tp_returns_ratio_when_new():
    mixin = TakeProfitMixin()
    trade = DummyTrade(1)
    assert mixin.set_tp(trade, 0.1, -0.05) == 0.1


def test_set_tp_returns_none_when_same():
    mixin = TakeProfitMixin()
    trade = DummyTrade(1)
    mixin.set_tp(trade, 0.1, -0.05)
    assert mixin.set_tp(trade, 0.1, -0.05) is None


def test_set_tp_respects_stoploss():
    mixin = TakeProfitMixin()
    trade = DummyTrade(1)
    # Ratio below absolute stop-loss should be ignored
    assert mixin.set_tp(trade, 0.04, -0.05) is None


def test_set_tp_none_resets_ratio():
    mixin = TakeProfitMixin()
    trade = DummyTrade(1)
    mixin.set_tp(trade, 0.1, -0.05)
    assert mixin.set_tp(trade, None, -0.05) is None
    # After reset a new value is accepted again
    assert mixin.set_tp(trade, 0.1, -0.05) == 0.1
