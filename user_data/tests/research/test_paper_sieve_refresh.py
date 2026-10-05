"""Bounded, DB/network-free PAPER stage regressions with real Order semantics."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ccxt import DECIMAL_PLACES, TICK_SIZE
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from freqtrade.persistence import LocalTrade, Order
from freqtrade.strategy import stoploss_from_absolute
from user_data.strategies import paper_sieve_refresh as paper


NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
PARTIAL_CLASSES = (paper.PaperSievePivotPartial, paper.PaperSieveD1SupportBreakLong,
                   paper.PaperSieveD1VpBosShort)
ALL_CLASSES = (*PARTIAL_CLASSES, paper.PaperSieveH4VpLvnLong)


class MemoryTrade(LocalTrade):
    def get_custom_data(self, key, default=None):
        return deepcopy(self.custom.get(key, default))

    def set_custom_data(self, key, value):
        self.custom[key] = deepcopy(value)


def strategy(cls):
    config = {
        'dry_run': True, 'runmode': RunMode.DRY_RUN, 'trading_mode': 'futures',
        'margin_mode': 'isolated', 'bot_name': cls.PAPER_BOT_NAME,
        'strategy': cls.__name__, 'max_open_trades': 3, 'force_entry_enable': False,
        'exchange': {'name': 'binance', 'key': '', 'secret': '',
                     'pair_whitelist': sorted(cls.PAPER_PAIRS)},
        'api_server': {'enabled': False},
    }
    result = cls(config)
    result.ft_load_hyper_params()
    return result


def frame_for(result, *, trigger=False, invalidation=False):
    row = {column: 100.0 for column in result.FOCUSED_REQUIRED_COLUMNS}
    row.update(date=NOW - timedelta(hours=2 if trigger or invalidation else 8),
               open=100.0, close=100.0, high=101.0, low=99.0,
               pivot_low_last=90.0, pivot_mid=105.0,
               prior_high_1h=110.0, range_mid_1h=95.0)
    if trigger:
        row.update(high=111.0, low=89.0)
    if invalidation:
        row['close'] = 106.0 if isinstance(result, paper.PaperSievePivotPartial) else 94.0
    frame = pd.DataFrame([row])
    result.dp = SimpleNamespace(get_analyzed_dataframe=lambda *args: (frame.copy(), NOW))
    return frame


def setup(cls, *, leverage=1.0, initial=203.4):
    result = strategy(cls)
    trade = MemoryTrade(pair='BTC/USDT:USDT', is_short=cls is not paper.PaperSieveD1SupportBreakLong,
                        open_rate=100.0, amount=initial * leverage / 100.0,
                        stake_amount=initial, leverage=leverage, open_date=NOW - timedelta(hours=3),
                        min_rate=100.0, max_rate=100.0, amount_precision=2,
                        precision_mode=DECIMAL_PLACES, contract_size=1.0, stop_loss=0.0)
    trade.custom = {}
    frame_for(result)
    state = result._focused_new_state(trade.pair, trade, NOW)
    result._focused_save_state(trade, state)
    return result, trade, state


def exit_order(trade, quantity, *, price=97.0, tag='focused_partial_stage_1',
               open_order=False, status='closed', amount=None):
    order = Order(order_id=str(len(trade.orders)), ft_order_side=trade.exit_side,
                  ft_order_tag=tag, ft_is_open=open_order, status=status,
                  amount=quantity if amount is None else amount, filled=quantity,
                  price=price, average=price, cost=quantity * price,
                  ft_amount=quantity if amount is None else amount, ft_price=price,
                  order_filled_date=NOW - timedelta(minutes=1))
    order._trade_bt = trade
    trade.orders.append(order)
    return order


def adjust(result, trade, profit=.03, when=NOW):
    return result.adjust_trade_position(trade, when, 97.0 if trade.is_short else 103.0,
                                      profit, None, 10000.0, 100.0, 100.0, profit, profit)


@pytest.mark.parametrize('cls', ALL_CLASSES)
def test_parameter_locks_and_source_entries_unchanged(cls, monkeypatch):
    result = strategy(cls)
    result._assert_selected_parameters()
    assert result.stoploss == result.EMERGENCY_STOP_FLOOR
    base = cls.__bases__[-1]
    assert cls.populate_entry_trend is base.populate_entry_trend
    name, expected = next(iter(result.LOCKED_SELL_PARAMS.items()))
    parameter = getattr(result, name)
    monkeypatch.setattr(parameter, 'value', 'corrupt' if isinstance(expected, str) else expected + 1)
    with pytest.raises(RuntimeError, match='frozen value'):
        result._assert_selected_parameters()


@pytest.mark.parametrize('cls', PARTIAL_CLASSES)
@pytest.mark.parametrize('leverage', [1.0, 3.0])
@pytest.mark.parametrize('price', [80.0, 120.0])
def test_rounded_completion_uses_entry_cost_not_exit_proceeds(cls, leverage, price):
    result, trade, state = setup(cls, leverage=leverage)
    stage = state['stages']['stage_1']
    stage.update(target_stake=10.17, status='requested', credited_stake=1234.0)
    order = exit_order(trade, .1 * leverage, price=price)
    assert order.stake_amount_filled == pytest.approx(10.0 * price / 100.0)
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['credited_stake'] == pytest.approx(10.0)
    assert stage['status'] == 'filled'
    assert stage['filled_at'] == order.order_filled_utc.isoformat()
    snapshot = deepcopy(state)
    result._focused_sync_partial_stages(trade, state, NOW + timedelta(seconds=5))
    assert state == snapshot


@pytest.mark.parametrize('cls', PARTIAL_CLASSES)
def test_underfill_open_cancelled_and_repeated_tags_reconcile(cls):
    result, trade, state = setup(cls)
    stage = state['stages']['stage_1']
    stage.update(target_stake=10.17, status='filled', credited_stake=1000.0, filled_at=NOW.isoformat())
    first = exit_order(trade, .04, price=150.0, amount=.1, status='canceled')
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'ready' and stage['filled_at'] is None
    assert stage['credited_stake'] == pytest.approx(4.0)
    second = exit_order(trade, .02, amount=.06, open_order=True, status='open')
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'requested' and stage['credited_stake'] == pytest.approx(6.0)
    second.ft_is_open = False
    second.status = 'canceled'
    second.filled = None
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'ready' and stage['credited_stake'] == pytest.approx(4.0)
    exit_order(trade, .06, price=75.0)
    exit_order(trade, 1.0, tag='unrelated')
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'filled' and stage['credited_stake'] == pytest.approx(10.0)
    assert first.safe_filled == .04


def test_contract_tick_rounding_and_zero_target_do_not_fake_completion():
    result, trade, state = setup(paper.PaperSievePivotPartial)
    trade.precision_mode, trade.amount_precision, trade.contract_size = TICK_SIZE, 1.0, .01
    stage = state['stages']['stage_1']
    stage.update(target_stake=10.17, status='requested')
    exit_order(trade, .1)
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'filled'
    trade.orders.clear()
    stage.update(target_stake=.1, status='filled', filled_at=NOW.isoformat())
    result._focused_sync_partial_stages(trade, state, NOW)
    assert stage['status'] == 'ready' and stage['filled_at'] is None
    result._focused_save_state(trade, state)
    frame_for(result, trigger=True)
    trade.min_rate = 89.0
    assert adjust(result, trade) is None
    assert result._focused_state(trade)['stages']['stage_1']['status'] == 'ready'


@pytest.mark.parametrize('cls', [paper.PaperSievePivotPartial, paper.PaperSieveD1SupportBreakLong])
def test_five_percent_partial_retry_completion_invalidation_and_stop(cls):
    result, trade, state = setup(cls)
    frame_for(result, trigger=True)
    trade.min_rate, trade.max_rate = 89.0, 111.0
    request = adjust(result, trade)
    assert request[0] == pytest.approx(-10.17) and request[1] == 'focused_partial_stage_1'
    # Pending callbacks before the order has been inserted must not duplicate it.
    assert adjust(result, trade) is None
    exit_order(trade, .04, status='canceled', amount=.1)
    trade.stake_amount -= 4.0
    retry = adjust(result, trade, when=NOW + timedelta(seconds=6))
    # First retry is requested once; a subsequent same-tick callback holds.
    assert retry is not None and retry[0] == pytest.approx(-6.17)
    exit_order(trade, .06)
    trade.stake_amount -= 6.0
    assert adjust(result, trade, when=NOW + timedelta(seconds=7)) is None
    stop = result.custom_stoploss(trade.pair, trade, NOW + timedelta(seconds=8),
                                 97.0 if trade.is_short else 103.0, .03, True)
    expected_price = 103.0 if trade.is_short else 100.0
    assert stop == pytest.approx(stoploss_from_absolute(expected_price,
                               current_rate=97.0 if trade.is_short else 103.0,
                               is_short=trade.is_short, leverage=1.0))
    assert result._focused_state(trade)['stop_price'] == expected_price
    # Move the candle beyond the previously processed trigger candle.
    invalid_frame = frame_for(result, invalidation=True)
    invalid_frame['date'] = NOW - timedelta(hours=1)
    assert result.custom_exit(trade.pair, trade, NOW, 97.0 if trade.is_short else 103.0, .03) == 'focused_source_invalidation'


def test_vp_named_stages_progress_without_retry_order_count_early_stop():
    result, trade, state = setup(paper.PaperSieveD1VpBosShort, initial=101.7)
    assert adjust(result, trade, profit=.019) is None
    request = adjust(result, trade)
    assert request[0] == pytest.approx(-10.17) and request[1] == 'focused_partial_stage_1'
    exit_order(trade, .03, amount=.1, status='canceled')
    exit_order(trade, .03, amount=.07, status='canceled')
    assert trade.nr_of_successful_exits == 2
    stop = result.custom_stoploss(trade.pair, trade, NOW + timedelta(seconds=1), 97.0, .03, True)
    assert stop == pytest.approx(stoploss_from_absolute(102.0, current_rate=97.0, is_short=True))
    exit_order(trade, .04)
    assert adjust(result, trade, profit=.044, when=NOW + timedelta(seconds=2)) is None
    stop = result.custom_stoploss(trade.pair, trade, NOW, 97.0, .03, True)
    assert stop == pytest.approx(stoploss_from_absolute(100.0, current_rate=97.0, is_short=True))
    request = adjust(result, trade, profit=.045, when=NOW + timedelta(seconds=3))
    assert request[0] == pytest.approx(-10.17) and request[1] == 'focused_partial_stage_2'
    exit_order(trade, .1, price=94.0, tag='focused_partial_stage_2')
    assert adjust(result, trade, profit=.069, when=NOW + timedelta(seconds=4)) is None
    stop = result.custom_stoploss(trade.pair, trade, NOW, 94.0, .06, True)
    assert stop == pytest.approx(stoploss_from_absolute(98.0, current_rate=94.0, is_short=True))
    assert result.custom_exit(trade.pair, trade, NOW, 93.0, .07) == 'focused_profit_target_3_full'


@pytest.mark.parametrize('stored', ['state', 'trade'])
def test_legacy_false_completion_repaired_without_stop_loosening(stored):
    result, trade, state = setup(paper.PaperSieveD1VpBosShort)
    state['stages']['stage_1'].update(target_stake=10.17, status='filled', credited_stake=11.0)
    state['stop_price'] = 98.0 if stored == 'state' else None
    trade.stop_loss = 98.0 if stored == 'trade' else 0.0
    result._focused_save_state(trade, state)
    exit_order(trade, .09, price=125.0)
    stop = result.custom_stoploss(trade.pair, trade, NOW, 97.0, .03, True)
    reconciled = result._focused_state(trade)
    assert reconciled['stages']['stage_1']['status'] == 'ready'
    assert reconciled['stages']['stage_1']['credited_stake'] == 9.0
    assert stop == pytest.approx(stoploss_from_absolute(98.0, current_rate=97.0, is_short=True))
    assert reconciled['version'] == state['version'] and reconciled['contract'] == state['contract']
    assert set(reconciled) == set(state)


def test_support_gaining_underfill_does_not_install_entry_stop():
    result, trade, state = setup(paper.PaperSieveD1SupportBreakLong)
    state['stages']['stage_1'].update(target_stake=10.17, status='filled', credited_stake=10.8)
    result._focused_save_state(trade, state)
    exit_order(trade, .09, price=120.0, amount=.1, status='canceled')
    stop = result.custom_stoploss(trade.pair, trade, NOW, 103.0, .03, True)
    assert result._focused_state(trade)['stages']['stage_1']['status'] == 'ready'
    assert stop == pytest.approx(stoploss_from_absolute(97.0, current_rate=103.0, is_short=False))


@pytest.mark.parametrize('cls', PARTIAL_CLASSES)
def test_matching_open_order_blocks_adjustment_and_stage_completion(cls):
    result, trade, state = setup(cls)
    state['stages']['stage_1'].update(target_stake=10.17, status='filled')
    result._focused_save_state(trade, state)
    exit_order(trade, .1, amount=.1, open_order=True, status='open')
    assert adjust(result, trade, profit=.08) is None
    assert result._focused_state(trade)['stages']['stage_1']['status'] == 'requested'
    assert result.custom_stoploss(trade.pair, trade, NOW, 97.0 if trade.is_short else 103.0,
                                 .03, True) is None


def test_lvn_exit_methods_are_not_overridden():
    cls = paper.PaperSieveH4VpLvnLong
    base = cls.__bases__[-1]
    for method in ('_focused_sync_partial_stages', 'adjust_trade_position', 'custom_stoploss', 'custom_exit'):
        assert getattr(cls, method, None) is getattr(base, method, None)
