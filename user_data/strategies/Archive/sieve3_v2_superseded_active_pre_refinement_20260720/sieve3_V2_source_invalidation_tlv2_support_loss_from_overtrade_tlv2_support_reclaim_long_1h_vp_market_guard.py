from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
SECONDARY_GUARD = 'none'
SOURCE_ENTRY_SIGNATURE_SHA256 = 'a276045de6fec9c81d55c6c3a45ed0b58cc0da87741a62c1f681a187cad7c21d'
LOCKED_BUY_SOURCE = 'verified effective lock from user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/Sieve2OvertradeTlv2SupportReclaimLong1hVPMarketGuard__auto_generic_1h_2020_q2_q3.json via user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/results/20260612T143424_entry_sieve2_top_level_remaining120_cores8_20260612.jsonl:116'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.market_state import add_market_state

SIEVE2_VP_GUARD_MODES = [
    "score_or_context",
    "node_confirm",
    "value_area_confirm",
    "breakout_acceptance",
    "rejection_confirm",
    "poc_hvn_reject",
    "prior_level_confirm",
]


def add_sieve2_guard_indicators(strategy: Any, dataframe: DataFrame, metadata: dict | None) -> DataFrame:
    frame = dataframe.copy()
    frame = add_market_state(frame, window=int(_param(strategy, "sieve2_market_window", 24)), prefix="s2m")
    frame = add_volume_profile(
        frame,
        window=int(_param(strategy, "sieve2_vp_window", 96)),
        bins=int(_param(strategy, "sieve2_vp_bins", 36)),
        value_area_pct=0.70,
        price_source="hlc3",
        smooth_bins=3,
        pressure_delta_min=0.05,
        node_near_pct=0.01,
        prefix="s2vp",
    )
    return frame

def apply_sieve2_optional_guards(strategy: Any, dataframe: DataFrame, condition: Series, side: str) -> Series:
    guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
    normalized_side = "short" if str(side or "").lower() == "short" else "long"
    always_on = _enabled(getattr(strategy, "SIEVE2_ALWAYS_ON_GUARDS", False))
    if always_on or _enabled(_param(strategy, "use_sieve2_vp_guard", False)):
        guarded &= _vp_guard(
            dataframe,
            normalized_side,
            str(_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on or _enabled(_param(strategy, "use_sieve2_market_guard", False)):
        market_mode = str(_param(strategy, 'sieve2_market_guard_mode', 'pressure_or_trend'))
        market_pressure_min = float(_param(strategy, 'sieve2_market_pressure_min', 0.07))
        market_trend_min = float(_param(strategy, 'sieve2_market_trend_min', 0.25))
        market_pressure = _num(dataframe, "s2m_pressure_ratio")
        market_trend = _num(dataframe, "s2m_trend_z")
        directional_pressure = (
            market_pressure.ge(market_pressure_min)
            if normalized_side == "long"
            else market_pressure.le(-market_pressure_min)
        )
        trend_state = (
            market_trend.ge(market_trend_min)
            if normalized_side == "long"
            else market_trend.le(-market_trend_min)
        )
        avoids_adverse_pressure = (
            market_pressure.ge(-market_pressure_min)
            if normalized_side == "long"
            else market_pressure.le(market_pressure_min)
        )
        avoids_adverse_trend = (
            market_trend.ge(-market_trend_min)
            if normalized_side == "long"
            else market_trend.le(market_trend_min)
        )
        if market_mode == "pressure_and_trend":
            market_condition = directional_pressure & trend_state
        elif market_mode == "directional_pressure":
            market_condition = directional_pressure
        elif market_mode == "trend_state":
            market_condition = trend_state
        elif market_mode == "avoid_adverse_pressure":
            market_condition = avoids_adverse_pressure & avoids_adverse_trend
        elif market_mode == "avoid_chop":
            market_condition = (
                market_pressure.abs().ge(market_pressure_min)
                | market_trend.abs().ge(market_trend_min)
            )
        else:
            market_condition = directional_pressure | trend_state
        guarded &= market_condition.fillna(False)
    return guarded.fillna(False)



def _vp_guard(frame: DataFrame, side: str, mode: str, score_min: float, context_min: float) -> Series:
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    score = _num(frame, f"s2vp_score_{side}")
    other = _num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _num(frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear")
    in_value = _bool(frame, "s2vp_in_value_area")
    above_value = _bool(frame, "s2vp_above_value_area")
    below_value = _bool(frame, "s2vp_below_value_area")
    node_entry = _bool(frame, f"s2vp_node_entry_{side}")
    node_hold = _bool(frame, f"s2vp_node_hold_{side}")
    prior_vah = _num(frame, "s2vp_prior_vah")
    prior_val = _num(frame, "s2vp_prior_val")
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        value_side = above_value if side == "long" else below_value
        return (value_side | (in_value & base_context)).fillna(False)
    if mode == "breakout_acceptance":
        accepted_beyond_value = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (accepted_beyond_value & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "rejection_confirm":
        rejection = (
            (low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah))
            if side == "long"
            else (high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val))
        )
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        level_confirm = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (level_confirm & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)

def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
ENTRY_MODE = 'entry_overtrade_tlv2_support_reclaim_long_1h_vp_market_guard'
ENTRY_TAG = 'overtrade_tlv2_support_reclaim_long_1h_vp_market_guard'
UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'
SIDE = 'long'
TIMEFRAME = '1h'


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)
SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve2OvertradeTlv2SupportReclaimLong1hVPMarketGuard'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/sieve2_overtrade_tlv2_support_reclaim_long_1h_vp_market_guard__auto_generic_1h_2020_q2_q3/full_cycle_2020_2026/tp_3_sl_3/backtest-result-2026-06-12_19-31-19.zip!backtest-result-2026-06-12_19-31-19_Sieve2OvertradeTlv2SupportReclaimLong1hVPMarketGuard.py:Sieve2OvertradeTlv2SupportReclaimLong1hVPMarketGuard'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'source_invalidation'
EXIT_THEORY = 'source_invalidation_tlv2_support_loss'
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_source_invalidation_tlv2_support_loss'
EXIT_HYPOTHESIS = 'A confirmed close through the entry-frozen entry_causal_tlv2_support_rank0_loss boundary invalidates the source thesis and closes the position.'
PRIMARY_TRIGGER = 'TLV2 rank-0 support score/distance qualification plus bullish close holding the configured reclaim buffer'
PRIMARY_GUARD = 'locked common candle guards plus always-on Sieve2 VP score-or-context and market pressure-or-trend guards'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'provider=entry_causal_tlv2_support_rank0_loss;mode=level;long.level=tlv2_support_line_rank0'
ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')


class Sieve3V2SourceInvalidationTlv2SupportLossFromOvertradeTlv2SupportReclaimLong1hVpMarketGuard(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_tighten_overtrade'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: tighten_overtrade; widen guard thresholds and adjust entry-strength search space.'
    SIEVE2_ALWAYS_ON_GUARDS = True
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    # Legacy inactive entry parameter: use_sieve2_vp_guard=True (proven inactive in the fixed entry path).
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.15
    sieve2_vp_context_min = 0.15
    # Legacy inactive entry parameter: use_sieve2_market_guard=True (proven inactive in the fixed entry path).
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.03
    sieve2_market_trend_min = 0.15
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: sieve2_rs_score_min=0.3 (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_volume_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: volume_guard_window=24 (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: volume_ratio_min=0.8 (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_pressure_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: pressure_window=12 (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: pressure_min=0.35 (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_accumulation_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_body_direction_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_close_direction_guard=False (proven inactive in the fixed entry path).
    pivot_strength = 2
    min_line_score = 0.4
    min_active_bars = 8
    max_distance_atr = 10.0
    proximity_rank_weight = 0.05
    reclaim_buffer_pct = 0.02

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_trendline_projection_v2(dataframe, timeframe=self.timeframe, pivot_strength=int(self.pivot_strength), raw_line_output_count=1, min_output_line_score=float(self.min_line_score), min_output_active_bars=int(self.min_active_bars), max_active_line_distance_atr_mult=float(self.max_distance_atr), proximity_rank_weight=float(self.proximity_rank_weight), output_prefix='tlv2')
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        _num(dataframe, 'close')
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        line = _num(dataframe, 'tlv2_support_line_rank0')
        condition = _num(dataframe, 'tlv2_support_score_rank0').ge(float(self.min_line_score)) & _num(dataframe, 'tlv2_support_distance_atr_rank0').le(float(self.max_distance_atr)) & _num(dataframe, 'close').ge(line.mul(1.0 - float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))
        condition &= self._common_guards(dataframe)
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    FOCUSED_EXIT_CONTRACT = 'source_invalidation'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'invalidation': {'provider': 'entry_causal_tlv2_support_rank0_loss', 'mode': 'level', 'long': {'level': 'tlv2_support_line_rank0', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'source_invalidation_close1': {'contract': 'source_invalidation', 'name': 'source_invalidation_close1', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025}, 'source_invalidation_close2': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025}, 'source_invalidation_close2_band_0_5': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2_band_0_5', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.005}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'tlv2_support_line_rank0')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2SourceInvalidationTlv2SupportLossFromOvertradeTlv2SupportReclaimLong1hVpMarketGuard:source_invalidation'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'use_volume_guard': False, 'volume_guard_window': 24, 'volume_ratio_min': 0.8, 'use_pressure_guard': False, 'pressure_window': 12, 'pressure_min': 0.35, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'pivot_strength': 2, 'min_line_score': 0.4, 'min_active_bars': 8, 'max_distance_atr': 10.0, 'proximity_rank_weight': 0.05, 'reclaim_buffer_pct': 0.02}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('source_invalidation_close1', 'source_invalidation_close2', 'source_invalidation_close2_band_0_5'), default='source_invalidation_close1', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    max_hold_scale = IntParameter(1, 4, default=2, space='sell', optimize=True, load=True)
    max_hold_scale.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = None
        return dataframe

    @staticmethod
    def _focused_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')

    @staticmethod
    def _focused_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _focused_number(frame: DataFrame, column: str) -> Series:
        return pd.to_numeric(frame[column], errors='coerce')

    @staticmethod
    def _focused_boolean(frame: DataFrame, column: str) -> Series:
        return frame[column].fillna(False).astype(bool)

    def _focused_close_times(self, frame: DataFrame) -> Series:
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        return dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit='m')

    def _focused_closed_frame(self, frame: DataFrame, current_time: Any) -> DataFrame:
        if frame is None or frame.empty:
            raise RuntimeError('focused exit runtime requires a non-empty analyzed dataframe')
        if 'date' not in frame.columns:
            raise KeyError('focused exit runtime requires the dataframe date column')
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for closed-candle evaluation')
        close_times = self._focused_close_times(frame)
        closed = frame.loc[close_times.le(now)].copy()
        return closed.sort_values('date').drop_duplicates('date', keep='last')

    def _focused_analyzed_frame(self, pair: str, current_time: Any) -> DataFrame:
        if getattr(self, 'dp', None) is None:
            raise RuntimeError("focused exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return self._focused_closed_frame(frame, current_time)

    def _focused_require_columns(self, frame: DataFrame) -> None:
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f'focused source profile is missing exact columns: {missing}')

    def _focused_side(self, trade: Any) -> str:
        side = 'short' if bool(getattr(trade, 'is_short', False)) else 'long'
        declared = str(self.FOCUSED_SOURCE_PROFILE['side'])
        if declared != 'both' and declared != side:
            raise ValueError(f'trade side {side!r} violates exact profile side {declared!r}')
        return side

    def _focused_plan(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        name = str((state or {}).get('plan') or self.exit_plan.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
        plan['max_hold_candles'] = max(1, int(round(float(plan.get('max_hold_candles', 336)) * float(self.max_hold_scale.value) / 2.0)))
        return plan

    @staticmethod
    def _focused_decision(action: str='hold', tag: str | None=None) -> dict[str, Any]:
        return {'action': action, 'tag': tag}

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('focused exit trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('focused exit trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('focused exit trade state contract mismatch')
        return dict(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=dict(state))

    def _focused_side_binding(self, role: str, side: str, slot: str | None=None) -> Mapping[str, Any] | None:
        if role == 'target':
            targets = self.FOCUSED_SOURCE_PROFILE.get('targets', {})
            provider = targets.get(str(slot))
        else:
            provider = self.FOCUSED_SOURCE_PROFILE.get(role)
        if not isinstance(provider, Mapping):
            return None
        binding = provider.get(side)
        return binding if isinstance(binding, Mapping) else None

    def _focused_frozen_level(self, row: Series, binding: Mapping[str, Any] | None, side: str, entry_rate: float, role: str) -> float | None:
        if binding is None:
            return None
        available_column = binding.get('available')
        if available_column:
            availability = row[str(available_column)]
            if bool(pd.isna(availability)):
                if role == 'invalidation':
                    raise ValueError(f'required invalidation availability is NaN: {available_column}')
                return None
            if not bool(availability):
                if role == 'invalidation':
                    raise ValueError(f'required invalidation level is unavailable: {available_column}')
                return None
        level_column = binding.get('level')
        if not level_column:
            if role == 'invalidation':
                raise ValueError('required invalidation profile has no level column')
            return None
        level = self._focused_float(row[str(level_column)])
        if level is None or level <= 0.0:
            if role == 'invalidation':
                raise ValueError(f'required invalidation level must be positive and finite: {level_column}')
            return None
        if role == 'invalidation':
            valid = level > entry_rate if side == 'short' else level < entry_rate
            if not valid:
                raise ValueError(f'required invalidation level is on the wrong side of entry: {level_column}')
            return level
        if role != 'target':
            raise ValueError(f'unsupported frozen level role: {role}')
        minimum = float(self.FOCUSED_SOURCE_PROFILE['min_level_distance'])
        valid = level < entry_rate * (1.0 - minimum) if side == 'short' else level > entry_rate * (1.0 + minimum)
        return level if valid else None

    def _focused_new_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, 'safe_price', None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit runtime requires a positive entry rate')
        freeze_time = (getattr(order, 'order_date_utc', None) if order is not None else None) or (getattr(order, 'order_date', None) if order is not None else None) or getattr(trade, 'open_date_utc', None) or getattr(trade, 'date_entry_fill_utc', None) or current_time
        frame = self._focused_analyzed_frame(pair, freeze_time)
        if frame.empty:
            return None
        self._focused_require_columns(frame)
        row = frame.iloc[-1]
        levels: dict[str, float | None] = {}
        target_specs = self.FOCUSED_SOURCE_PROFILE.get('targets', {})
        if isinstance(target_specs, Mapping):
            for slot in target_specs:
                levels[str(slot)] = self._focused_frozen_level(row, self._focused_side_binding('target', side, str(slot)), side, entry_rate, 'target')
        invalidation_binding = self._focused_side_binding('invalidation', side)
        levels['invalidation'] = self._focused_frozen_level(row, invalidation_binding, side, entry_rate, 'invalidation')
        invalidation_level = levels.get('invalidation')
        for slot in ('target_1', 'target_2'):
            target_level = levels.get(slot)
            if target_level is not None and invalidation_level is not None and math.isclose(target_level, invalidation_level, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'resolved {slot} equals the source invalidation level')
        first = levels.get('target_1')
        second = levels.get('target_2')
        if first is not None and second is not None:
            ordered = second < first if side == 'short' else second > first
            if not ordered:
                raise ValueError('entry-frozen target_2 must be beyond target_1')
        selected_plan = str(self.exit_plan.value)
        fill_time = getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None)
        if fill_time is None:
            raise ValueError('focused exit runtime requires an entry fill timestamp')
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'stop_price': None}

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any], current_time: Any) -> DataFrame:
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for post-entry evaluation')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        close_times = self._focused_close_times(frame)
        return frame.loc[dates.ge(filled_at) & close_times.le(now)]


    @staticmethod

    @staticmethod
    def _focused_any_n(condition: Series, count: int) -> bool:
        needed = max(1, int(count))
        values = condition.fillna(False).astype(bool)
        if len(values) < needed:
            return False
        confirmed = values.rolling(window=needed, min_periods=needed).sum().ge(needed)
        return bool(confirmed.any())
    @staticmethod
    def _focused_last_n(condition: Series, count: int) -> bool:
        needed = max(1, int(count))
        return len(condition) >= needed and bool(condition.tail(needed).fillna(False).all())

    @staticmethod
    def _focused_distinct_periods(frame: DataFrame, condition: Series, timeframe: str) -> Series:
        minutes = int(timeframe_to_minutes(timeframe))
        if minutes <= 0:
            raise ValueError('invalidation evidence timeframe must be at least one minute')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        period_ids = dates.map(lambda value: timeframe_to_prev_date(timeframe, value.to_pydatetime()))
        return condition.fillna(False).groupby(period_ids, sort=True).any()


    @staticmethod
    def _focused_tighter_stop(side: str, current: float, candidate: float) -> float:
        return min(current, candidate) if side == 'short' else max(current, candidate)

    @staticmethod
    def _focused_hard_stop_price(plan: Mapping[str, Any], state: Mapping[str, Any]) -> float:
        entry_rate = float(state['entry_rate'])
        ratio = float(plan['hard_stop_ratio'])
        return entry_rate * (1.0 + ratio if state['side'] == 'short' else 1.0 - ratio)

    def _focused_hard_stop_breached(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_rate: float) -> bool:
        stop_price = self._focused_hard_stop_price(plan, state)
        return current_rate >= stop_price if state['side'] == 'short' else current_rate <= stop_price



    def _focused_context(self, pair: str, trade: Any, current_time: Any, current_rate: float, current_profit: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return (self._focused_plan(), {}, {}, self._focused_decision())
        before = repr(state)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state, current_time)
        events = self._focused_contract_events(post_entry, state, plan, current_profit)
        events['age_candles'] = len(post_entry)
        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision('full', 'focused_hard_stop')
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision['action'] == 'hold' and events['age_candles'] >= int(plan['max_hold_candles']):
            decision = self._focused_decision('full', 'focused_max_hold')
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return (plan, state, events, decision)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = kwargs
        _, _, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        return str(decision['tag']) if decision['action'] == 'full' else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = (after_fill, kwargs)
        plan, state, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        if not state:
            return None
        if decision['action'] == 'full':
            return None
        desired = self._focused_stop_overlay(plan, state, current_profit)
        persisted = self._focused_float(state.get('stop_price'))
        stop_price = desired if persisted is None else self._focused_tighter_stop(str(state['side']), persisted, desired)
        if state['side'] == 'short' and stop_price <= current_rate or (state['side'] == 'long' and stop_price >= current_rate):
            return None
        if persisted is None or not math.isclose(stop_price, persisted, rel_tol=0.0, abs_tol=1e-12):
            state['stop_price'] = stop_price
            self._focused_save_state(trade, state)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen'):
            return True
        if post_entry.empty:
            return False
        side = str(state['side'])
        provider = self.FOCUSED_SOURCE_PROFILE['invalidation']
        binding = provider[side]
        mode = str(provider['mode'])
        band = float(plan.get('invalidation_band') or 0.0)
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        close = self._focused_number(post_entry, 'close')
        if level is None:
            level_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        elif side == 'short':
            level_condition = close.gt(level * (1.0 + band))
        else:
            level_condition = close.lt(level * (1.0 - band))
        evidence_column = binding.get('evidence')
        evidence_condition = self._focused_boolean(post_entry, str(evidence_column)) if evidence_column else pd.Series(False, index=post_entry.index, dtype='bool')
        confirmations = int(plan.get('invalidation_confirmations') or 1)
        if mode == 'level':
            confirmed = self._focused_any_n(level_condition, confirmations)
        elif mode == 'evidence':
            event_condition = evidence_condition
        elif mode == 'level_and_evidence':
            event_condition = level_condition & evidence_condition
        elif mode == 'level_or_evidence':
            event_condition = evidence_condition
        else:
            raise ValueError(f'unsupported invalidation mode: {mode}')
        if mode != 'level':
            evidence_timeframe = provider.get('timeframe')
            if evidence_timeframe is not None:
                event_condition = self._focused_distinct_periods(post_entry, event_condition, str(evidence_timeframe))
            event_seen = bool(event_condition.fillna(False).any())
            confirmed = self._focused_any_n(level_condition, confirmations) or event_seen if mode == 'level_or_evidence' else event_seen
        if confirmed:
            state['invalidation_seen'] = True
            state['phase'] = 'invalidated'
        return confirmed

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return {'invalidation': self._focused_invalidation_event(post_entry, state, plan)}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (state, plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        return self._focused_decision()

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        stop_price = self._focused_hard_stop_price(plan, state)
        return stop_price

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, 'ft_order_side', None) == getattr(trade, 'entry_side', None) and self._focused_state(trade) is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
            return None
        return None
