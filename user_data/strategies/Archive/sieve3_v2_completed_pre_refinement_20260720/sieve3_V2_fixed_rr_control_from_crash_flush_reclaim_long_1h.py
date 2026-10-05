from __future__ import annotations
import pandas as pd
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Capitulation flushes that wick and reclaim should be cleaner than raw fades.'
import numpy as np
from freqtrade.strategy import IStrategy
SIDE = 'long'
ENTRY_MODE = 'flush_reclaim'
ENTRY_TAG = 'crash_flush_reclaim_long_1h'
ENTRY_SOURCE_STAGE = 'sieve2'


def _safe_div(numer: Series, denom: Series) -> Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom

def _rsi(close: Series, length: int=14) -> Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def _atr(frame: DataFrame, length: int=14) -> Series:
    high = pd.to_numeric(frame['high'], errors='coerce')
    low = pd.to_numeric(frame['low'], errors='coerce')
    close = pd.to_numeric(frame['close'], errors='coerce')
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=max(2, length // 2)).mean()
SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromCrashFlushReclaimLong1H'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_crash_flush_reclaim_long_1h.py:Sieve3ExitFixedTpSlFromCrashFlushReclaimLong1H'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'fixed_rr_control'
EXIT_THEORY = 'fixed_rr_control'
EXIT_HYPOTHESIS = 'A compact named fixed reward/risk control compares exact entry-relative objectives without source-level inference.'
PRIMARY_TRIGGER = 'drop_pct >= 0.13 with lower_wick_ratio >= 0.55 and close > compression_mid'
PRIMARY_GUARD = 'volume_ratio >= 1.02 or price_spike >= 0.042'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'none'
ACTIVE_SELL_PARAMS = ('target_percent', 'stop_percent')


RESEARCH_PATH = 'sieve3_exit_fixed_rr_control'
class Sieve3V2FixedRrControlFromCrashFlushReclaimLong1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_fixed_rr_control'
    EXIT_HYPOTHESIS = 'A compact named fixed reward/risk control compares exact entry-relative objectives without source-level inference.'
    timeframe = '1h'
    can_short = False
    startup_candle_count = 320
    process_only_new_candles = True
    max_entry_position_adjustment = 0
    INTERFACE_VERSION = 3
    crash_lookback = 56
    compression_lookback = 5
    drop_pct_min = 0.13
    # Legacy inert entry parameter: compression_width_max=0.44 (fixed entry mode/gate).
    volume_ratio_min = 1.02
    price_spike_pct_min = 0.042
    # Legacy inert entry parameter: rsi_max=15 (fixed entry mode/gate).
    # Legacy inert entry parameter: reclaim_buffer_pct=0.009 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_gate_mode='soft' (fixed entry mode/gate).

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        high = pd.to_numeric(dataframe['high'], errors='coerce')
        low = pd.to_numeric(dataframe['low'], errors='coerce')
        open_ = pd.to_numeric(dataframe['open'], errors='coerce')
        volume = pd.to_numeric(dataframe['volume'], errors='coerce').fillna(0.0)
        look = int(self.crash_lookback)
        comp = int(self.compression_lookback)
        roll_high = close.rolling(look, min_periods=max(2, look // 2)).max()
        roll_low = close.rolling(look, min_periods=max(2, look // 2)).min()
        comp_high = high.rolling(comp, min_periods=max(2, comp // 2)).max()
        comp_low = low.rolling(comp, min_periods=max(2, comp // 2)).min()
        comp_mean = close.rolling(comp, min_periods=max(2, comp // 2)).mean()
        dataframe['rsi'] = _rsi(close, 14)
        dataframe['atr'] = _atr(dataframe, 14)
        dataframe['drop_pct'] = _safe_div(roll_high - close, roll_high).fillna(0.0)
        dataframe['compression_width'] = _safe_div(comp_high - comp_low, comp_mean).fillna(0.0)
        dataframe['compression_high'] = comp_high
        dataframe['compression_low'] = comp_low
        dataframe['compression_mid'] = (comp_high + comp_low) / 2.0
        dataframe['rolling_low'] = roll_low
        dataframe['volume_ratio'] = _safe_div(volume, volume.rolling(comp, min_periods=max(2, comp // 2)).mean()).fillna(0.0)
        dataframe['price_spike'] = _safe_div((close - close.shift(1)).abs(), close.shift(1).abs()).fillna(0.0)
        dataframe['body_ratio'] = _safe_div((close - open_).abs(), high - low).fillna(0.0)
        dataframe['lower_wick_ratio'] = _safe_div(np.minimum(open_, close) - low, high - low).fillna(0.0)
        dataframe['upper_wick_ratio'] = _safe_div(high - np.maximum(open_, close), high - low).fillna(0.0)
        dataframe['trend_mean'] = close.rolling(max(look * 2, comp * 2), min_periods=max(3, look)).mean()
        return dataframe

    

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        drop_ok = pd.to_numeric(dataframe['drop_pct'], errors='coerce').ge(float(self.drop_pct_min))
        volume_ok = pd.to_numeric(dataframe['volume_ratio'], errors='coerce').ge(float(self.volume_ratio_min))
        spike_ok = volume_ok | pd.to_numeric(dataframe['price_spike'], errors='coerce').ge(float(self.price_spike_pct_min))
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        compression_mid = pd.to_numeric(dataframe['compression_mid'], errors='coerce')
        return drop_ok & pd.to_numeric(dataframe['lower_wick_ratio'], errors='coerce').ge(0.55) & close.gt(compression_mid) & spike_ok

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        dataframe.loc[condition, ['enter_long', 'enter_tag']] = (1, ENTRY_TAG)
        return dataframe
    SOURCE_ENTRY_STEM = 'crash_flush_reclaim_long_1h'
    FOCUSED_EXIT_CONTRACT = 'fixed_rr_control'
    FOCUSED_SOURCE_PROFILE = {'side': 'long'}
    FOCUSED_EXIT_PLANS = {
        'baseline_3_3': {'contract': 'fixed_rr_control', 'name': 'baseline_3_3', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.03, 'fixed_target_ratio': 0.03},
        'reward_4_2': {'contract': 'fixed_rr_control', 'name': 'reward_4_2', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.02, 'fixed_target_ratio': 0.04},
        'defensive_2_3': {'contract': 'fixed_rr_control', 'name': 'defensive_2_3', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.03, 'fixed_target_ratio': 0.02},
    }
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2FixedRrControlFromCrashFlushReclaimLong1H:fixed_rr_control'
    LOCKED_BUY_PARAMS = {'crash_lookback': 56, 'compression_lookback': 5, 'drop_pct_min': 0.13, 'compression_width_max': 0.44, 'volume_ratio_min': 1.02, 'price_spike_pct_min': 0.042, 'rsi_max': 15, 'reclaim_buffer_pct': 0.009, 'rsi_gate_mode': 'soft'}
    ACTIVE_SELL_PARAMS = ('target_percent', 'stop_percent')
    position_adjustment_enable = False
    use_custom_stoploss = True
    use_custom_roi = True
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('baseline_3_3', 'reward_4_2', 'defensive_2_3'), default='baseline_3_3', space='sell', optimize=False, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_percent = IntParameter(2, 12, default=3, space='sell', optimize=True, load=True)
    target_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = None
        return dataframe

    def _focused_side(self, trade: Any) -> str:
        side = 'short' if bool(getattr(trade, 'is_short', False)) else 'long'
        declared = str(self.FOCUSED_SOURCE_PROFILE['side'])
        if declared != 'both' and declared != side:
            raise ValueError(f'trade side {side!r} violates exact profile side {declared!r}')
        return side

    def _focused_state(self, trade: Any) -> dict[str, Any]:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            raise RuntimeError('fixed-RR trade is missing its frozen exit plan state')
        if not isinstance(state, dict):
            raise ValueError('fixed-RR trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('fixed-RR trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('fixed-RR trade state contract mismatch')
        if state.get('side') != self._focused_side(trade):
            raise ValueError('fixed-RR trade state side mismatch')
        if state.get('plan') not in self.FOCUSED_EXIT_PLANS:
            raise ValueError('fixed-RR trade state has an unknown plan')
        return dict(state)

    def _focused_new_state(self, trade: Any) -> dict[str, Any]:
        plan_name = str(self.exit_plan.value)
        if plan_name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown fixed-RR exit plan: {plan_name}')
        plan = self.FOCUSED_EXIT_PLANS[plan_name]
        return {
            'version': self.FOCUSED_STATE_VERSION,
            'contract': self.FOCUSED_EXIT_CONTRACT,
            'plan': plan_name,
            'side': self._focused_side(trade),
            'fixed_target_ratio': float(self.target_percent.value) * 0.01,
            'hard_stop_ratio': float(self.stop_percent.value) * 0.01,
        }

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = pair, current_time, kwargs
        if getattr(order, 'ft_order_side', None) == getattr(trade, 'entry_side', None):
            if trade.get_custom_data(key=self.FOCUSED_STATE_KEY) is None:
                trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=self._focused_new_state(trade))

    def custom_roi(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, trade_duration, entry_tag, kwargs
        state = self._focused_state(trade)
        if side != state['side']:
            raise ValueError('fixed-RR custom ROI side mismatch')
        return float(state['fixed_target_ratio'])

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, current_profit, after_fill, kwargs
        state = self._focused_state(trade)
        is_short = state['side'] == 'short'
        stop_ratio = float(state['hard_stop_ratio'])
        stop_price = float(trade.open_rate) * (1.0 + stop_ratio if is_short else 1.0 - stop_ratio)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=is_short,
            leverage=float(getattr(trade, 'leverage', 1.0) or 1.0),
        )
