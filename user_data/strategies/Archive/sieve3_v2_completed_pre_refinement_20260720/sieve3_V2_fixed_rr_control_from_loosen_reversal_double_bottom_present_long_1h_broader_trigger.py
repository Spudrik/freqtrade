from __future__ import annotations
import pandas as pd
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pattern_reversal import add_pattern_reversal
ENTRY_MODE = 'entry_loosen_reversal_double_bottom_present_long_1h_broader_trigger'
ENTRY_TAG = 'loosen_reversal_double_bottom_present_long_1h_broader_trigger'
ENTRY_SOURCE_STAGE = 'sieve2'
UPDATE_HYPOTHESIS = 'High-win sparse entry may be too restrictive; broaden structural timing/quality thresholds while preserving the original entry idea.'
SIDE = 'long'
TIMEFRAME = '1h'


def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column in frame.columns:
        value = frame[column]
    elif default is not ...:
        value = pd.Series(default, index=frame.index)
    else:
        raise KeyError(column)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_STAGE = 'sieve3'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromLoosenReversalDoubleBottomPresentLong1HBroaderTrigger'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_loosen_reversal_double_bottom_present_long_1h_broader_trigger.py:Sieve3ExitFixedTpSlFromLoosenReversalDoubleBottomPresentLong1HBroaderTrigger'
LINEAGE_STATUS = 'legacy_executable_surface_defaults_frozen; promotion_lock_unverified'
EXIT_FAMILY = 'fixed_rr_control'
EXIT_THEORY = 'fixed_rr_control'
EXIT_HYPOTHESIS = 'Fixed reward/risk plans control the locked pat_double_bottom_pattern_present with score >= 0.72 and close > pat_double_bottom_confirmation_level entry without an indicator target.'
PRIMARY_TRIGGER = 'pat_double_bottom_pattern_present with score >= 0.72 and close > pat_double_bottom_confirmation_level'
PRIMARY_GUARD = 'selected 48-candle bullish pressure >= 0.2; volume and optional Sieve2 guards disabled'
TARGET_PROVIDER = 'exit_plan fixed target: 2%, 3%, or 4% favorable move'
INVALIDATION_PROVIDER = 'exit_plan fixed hard stop: 2% or 3% adverse move'
ACTIVE_SELL_PARAMS = ('target_percent', 'stop_percent')


RESEARCH_PATH = 'sieve3_exit_fixed_rr_control'
class Sieve3V2FixedRrControlFromLoosenReversalDoubleBottomPresentLong1HBroaderTrigger(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_fixed_rr_control'
    EXIT_HYPOTHESIS = 'Fixed reward/risk plans control the locked pat_double_bottom_pattern_present with score >= 0.72 and close > pat_double_bottom_confirmation_level entry without an indicator target.'
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    # use_volume_guard = False (locked off; gated entry branch removed).
    # volume_guard_window = 12 (inactive after its locked entry gate/mode was removed).
    # volume_ratio_min = 1.3 (inactive after its locked entry gate/mode was removed).
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.2
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    score_min = 0.72
    confirmation_buffer_pct = 0.0

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_reversal(dataframe, timeframe=self.timeframe, min_double_quality=float(self.score_min), min_head_shoulders_quality=float(self.score_min))
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, 'open')
            high = _num(dataframe, 'high')
            low = _num(dataframe, 'low')
            volume = _num(dataframe, 'volume').clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == 'short' else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pat_double_bottom_confirmation_level',)).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        level = _num(dataframe, 'pat_double_bottom_confirmation_level', np.nan).mul(1.0 + float(self.confirmation_buffer_pct))
        condition = _bool(dataframe, 'pat_double_bottom_pattern_present') & _num(dataframe, 'pat_double_bottom_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'close').gt(level)
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'loosen_reversal_double_bottom_present_long_1h_broader_trigger'
    FOCUSED_EXIT_CONTRACT = 'fixed_rr_control'
    FOCUSED_SOURCE_PROFILE = {'side': 'long'}
    FOCUSED_EXIT_PLANS = {
        'baseline_3_3': {'contract': 'fixed_rr_control', 'name': 'baseline_3_3', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.03, 'fixed_target_ratio': 0.03},
        'reward_4_2': {'contract': 'fixed_rr_control', 'name': 'reward_4_2', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.02, 'fixed_target_ratio': 0.04},
        'defensive_2_3': {'contract': 'fixed_rr_control', 'name': 'defensive_2_3', 'action_sequence': ('fixed_target_full', 'hard_stop'), 'hard_stop_ratio': 0.03, 'fixed_target_ratio': 0.02},
    }
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2FixedRrControlFromLoosenReversalDoubleBottomPresentLong1HBroaderTrigger:fixed_rr_control'
    LOCKED_BUY_PARAMS = {'use_volume_guard': False, 'volume_guard_window': 12, 'volume_ratio_min': 1.3, 'use_pressure_guard': True, 'pressure_window': 48, 'pressure_min': 0.2, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'score_min': 0.72, 'confirmation_buffer_pct': 0.0}
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
