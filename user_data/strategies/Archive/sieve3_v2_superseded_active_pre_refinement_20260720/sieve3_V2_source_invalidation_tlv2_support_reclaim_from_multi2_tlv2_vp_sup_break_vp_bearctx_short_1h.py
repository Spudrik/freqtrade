from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute, IntParameter
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_TAG = "multi2_tlv2_vp_sup_break_vp_bearctx_short_1h"
SIDE = "short"
TIMEFRAME = "1h"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = "Sieve3ExitBreakevenFromMulti2Tlv2VpSupBreakVpBearctxShort1H"
SOURCE_STRATEGY = (
    "user_data/strategies/sieve3_exit_breakeven_from_multi2_tlv2_vp_sup_break_vp_bearctx_short_1h.py:"
    f"{SOURCE_ENTRY_CLASS}"
)
LOCKED_BUY_SOURCE = "verified migration-ledger effective buy lock"
SOURCE_ENTRY_SIGNATURE_SHA256 = "45c34a4895c8227a66991b39c91f9e3a2f373638a7a1e555b91c752537579740"
LINEAGE_STATUS = "verified"
SOURCE_RESULT_BATCH = "verified_migration_ledger"
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
EXIT_FAMILY = "source_invalidation"
EXIT_THEORY = "source_invalidation_tlv2_support_reclaim"
EXIT_HYPOTHESIS = (
    "A confirmed close above the entry-frozen broken TLV2 support invalidates the short breakdown thesis."
)
PRIMARY_TRIGGER = "1h close cross below TLV2 support with the locked line buffer"
PRIMARY_GUARD = "VP short score plus bearish context and non-bullish VP market state"
SECONDARY_GUARD = "locked bearish directional-volume pressure guard"
TARGET_PROVIDER = "none"
INVALIDATION_PROVIDER = (
    "provider=entry-frozen broken TLV2 support reclaim;mode=level;short.level=tlv2_support_line_rank0"
)
ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')





def _cross_below(series: Series, level: Series) -> Series:
    return series.lt(level) & series.shift(1).ge(level.shift(1))


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)

def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))

def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

ENTRY_MODE = 'entry_multi2_tlv2_vp_sup_break_vp_bearctx_short_1h'

TLV2_KIND = 'support_breakdown'

TLV2_LINE = 'support'

VP_CONFIRM = 'bear_context'

class Sieve3V2SourceInvalidationTlv2SupportReclaimFromMulti2Tlv2VpSupBreakVpBearctxShort1H(IStrategy):
    """Locked TLV2/VP short entry with frozen broken-support reclaim invalidation."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = True
    use_sieve2_vp_guard = False
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.25
    sieve2_vp_context_min = 0.28
    use_sieve2_market_guard = False
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.07
    sieve2_market_trend_min = 0.25
    sieve2_rs_benchmark_pair = 'BTC/USDT:USDT'
    sieve2_rs_score_min = 0.45
    use_volume_guard = False
    volume_guard_window = 12
    volume_ratio_min = 0.8
    use_pressure_guard = True
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    vp_level_buffer_pct = 0.016
    line_slope_min_pct = 0.0015
    max_entry_position_adjustment = 0

    pressure_window = 12
    pressure_min = 0.35

    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = "hlc3"
    vp_smooth_bins = 3
    vp_hvn_threshold = 0.7
    vp_lvn_threshold = 0.35
    vp_pressure_delta_min = 0.05
    vp_node_near_pct = 0.01
    vp_volume_percentile_min = 0.55
    vp_score_window = 48
    vp_fast_traverse_atr_mult = 1.2
    vp_entry_score_margin = 0.02
    vp_score_min = 0.4
    vp_context_min = 0.2

    pivot_strength = 2
    min_line_score = 0.5
    min_active_bars = 8
    max_distance_atr = 3.0
    proximity_rank_weight = 0.05
    line_buffer_pct = 0.012

    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': False, 'volume_guard_window': 12, 'volume_ratio_min': 0.8, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.35, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'vp_score_min': 0.4, 'vp_context_min': 0.2, 'vp_level_buffer_pct': 0.016, 'pivot_strength': 2, 'min_line_score': 0.5, 'min_active_bars': 8, 'max_distance_atr': 3.0, 'proximity_rank_weight': 0.05, 'line_buffer_pct': 0.012, 'line_slope_min_pct': 0.0015}

    SOURCE_ENTRY_STEM = ENTRY_TAG
    FOCUSED_EXIT_CONTRACT = "source_invalidation"
    FOCUSED_SOURCE_PROFILE = {
        "side": "short",
        "invalidation": {
            "provider": "entry-frozen broken TLV2 support reclaim",
            "level": "tlv2_support_line_rank0",
        },
    }
    FOCUSED_EXIT_PLANS = {
        "source_invalidation_close1": {
            "hard_stop_ratio": 0.03,
            "max_hold_candles": 336,
            "invalidation_confirmations": 1,
            "invalidation_band": 0.0025,
        },
        "source_invalidation_close2": {
            "hard_stop_ratio": 0.03,
            "max_hold_candles": 336,
            "invalidation_confirmations": 2,
            "invalidation_band": 0.0025,
        },
        "source_invalidation_close2_band_0_5": {
            "hard_stop_ratio": 0.03,
            "max_hold_candles": 336,
            "invalidation_confirmations": 2,
            "invalidation_band": 0.005,
        },
    }
    FOCUSED_REQUIRED_COLUMNS = ("date", "close", "tlv2_support_line_rank0")
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = (
        "sieve3_v2_focused:"
        "Sieve3V2SourceInvalidationTlv2SupportReclaimFromMulti2Tlv2VpSupBreakVpBearctxShort1H:"
        "source_invalidation"
    )
    ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')

    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(
        tuple(FOCUSED_EXIT_PLANS),
        default="source_invalidation_close1",
        space="sell",
        optimize=False,
        load=True,
    )
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    invalidation_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_band_quarter_percent = IntParameter(0, 6, default=1, space='sell', optimize=True, load=True)
    invalidation_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    max_hold_scale = IntParameter(1, 4, default=2, space='sell', optimize=True, load=True)
    max_hold_scale.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        if bool(self.use_volume_guard):
            volume = _num(dataframe, 'volume').clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):
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
        if bool(self.use_accumulation_guard):
            window = int(self.pressure_window)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.le(0.0) if SIDE == 'short' else accumulation.ge(0.0)
        if bool(self.use_body_direction_guard):
            open_ = _num(dataframe, 'open')
            guard &= close.lt(open_) if SIDE == 'short' else close.gt(open_)
        if bool(self.use_close_direction_guard):
            guard &= close.lt(close.shift(1)) if SIDE == 'short' else close.gt(close.shift(1))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(dataframe, window=int(self.vp_window), bins=int(self.vp_bins), value_area_pct=float(self.vp_value_area_pct), price_source=str(self.vp_price_source), smooth_bins=int(self.vp_smooth_bins), hvn_threshold=float(self.vp_hvn_threshold), lvn_threshold=float(self.vp_lvn_threshold), pressure_delta_min=float(self.vp_pressure_delta_min), node_near_pct=float(self.vp_node_near_pct), volume_percentile_min=float(self.vp_volume_percentile_min), score_window=int(self.vp_score_window), fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult), entry_score_margin=float(self.vp_entry_score_margin), prefix='vp')

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        score_long = _num(dataframe, 'vp_score_long')
        score_short = _num(dataframe, 'vp_score_short')
        bull_context = _num(dataframe, 'vp_context_score_bull')
        bear_context = _num(dataframe, 'vp_context_score_bear')
        balance = _num(dataframe, 'vp_context_score_balance')
        market = _num(dataframe, 'vp_market_context')
        score_min = float(self.vp_score_min)
        context_min = float(self.vp_context_min)
        near = float(self.vp_level_buffer_pct)
        if VP_CONFIRM == 'bull_context':
            return score_long.ge(score_min) & bull_context.ge(context_min) & bull_context.ge(bear_context) & market.ge(0)
        if VP_CONFIRM == 'bear_context':
            return score_short.ge(score_min) & bear_context.ge(context_min) & bear_context.ge(bull_context) & market.le(0)
        if VP_CONFIRM == 'val_reclaim':
            val = _num(dataframe, 'vp_val')
            return close.ge(val.mul(1.0 - near)) & (score_long.ge(score_min) | _bool(dataframe, 'vp_entry_trigger_long') | balance.ge(context_min))
        if VP_CONFIRM == 'vah_reject':
            vah = _num(dataframe, 'vp_vah')
            return close.le(vah.mul(1.0 + near)) & (score_short.ge(score_min) | _bool(dataframe, 'vp_entry_trigger_short') | balance.ge(context_min))
        if VP_CONFIRM == 'node_entry':
            return _bool(dataframe, 'vp_node_entry_short') | score_short.ge(score_min) if SIDE == 'short' else _bool(dataframe, 'vp_node_entry_long') | score_long.ge(score_min)
        return pd.Series(True, index=dataframe.index, dtype='bool')

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(dataframe, timeframe=self.timeframe, pivot_strength=int(self.pivot_strength), raw_line_output_count=1, min_output_line_score=float(self.min_line_score), min_output_active_bars=int(self.min_active_bars), max_active_line_distance_atr_mult=float(self.max_distance_atr), proximity_rank_weight=float(self.proximity_rank_weight), output_prefix='tlv2')

    def _tlv2_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        buffer = float(self.line_buffer_pct)
        slope_min = float(self.line_slope_min_pct)
        if TLV2_LINE == 'support':
            line = _num(dataframe, 'tlv2_support_line_rank0')
            score = _num(dataframe, 'tlv2_support_score_rank0')
            distance = _num(dataframe, 'tlv2_support_distance_atr_rank0')
        else:
            line = _num(dataframe, 'tlv2_resistance_line_rank0')
            score = _num(dataframe, 'tlv2_resistance_score_rank0')
            distance = _num(dataframe, 'tlv2_resistance_distance_atr_rank0')
        active = score.ge(float(self.min_line_score)) & distance.le(float(self.max_distance_atr))
        if TLV2_KIND == 'support_reclaim':
            return active & low.le(line.mul(1.0 + buffer)) & close.ge(line.mul(1.0 - buffer)) & close.gt(open_)
        if TLV2_KIND == 'support_bounce':
            return active & low.le(line.mul(1.0 + buffer)) & close.gt(line) & close.gt(open_)
        if TLV2_KIND == 'rising_support_ride':
            return active & line.pct_change(fill_method=None).gt(slope_min) & close.gt(line)
        if TLV2_KIND == 'resistance_breakout':
            return active & _cross_above(close, line.mul(1.0 + buffer))
        if TLV2_KIND == 'resistance_reject':
            return active & high.ge(line.mul(1.0 - buffer)) & close.le(line.mul(1.0 + buffer)) & close.lt(open_)
        if TLV2_KIND == 'resistance_proximity_reject':
            return active & high.ge(line.mul(1.0 - buffer)) & close.lt(line) & close.lt(open_)
        if TLV2_KIND == 'falling_resistance_ride':
            return active & line.pct_change(fill_method=None).lt(-slope_min) & close.lt(line)
        if TLV2_KIND == 'support_breakdown':
            return active & _cross_below(close, line.mul(1.0 - buffer))
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_tlv2(dataframe)
        dataframe = self._add_vp(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._tlv2_trigger(dataframe) & self._vp_confirm(dataframe)
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @staticmethod
    def _focused_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    @staticmethod
    def _focused_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _focused_close_times(self, frame: DataFrame) -> Series:
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        return dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")

    def _focused_closed_frame(self, frame: DataFrame, current_time: Any) -> DataFrame:
        if frame is None or frame.empty:
            raise RuntimeError("focused exit runtime requires a non-empty analyzed dataframe")
        if "date" not in frame.columns:
            raise KeyError("focused exit runtime requires the dataframe date column")
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError("current_time is required for closed-candle evaluation")
        closed = frame.loc[self._focused_close_times(frame).le(now)].copy()
        return closed.sort_values("date").drop_duplicates("date", keep="last")

    def _focused_analyzed_frame(self, pair: str, current_time: Any) -> DataFrame:
        if getattr(self, "dp", None) is None:
            raise RuntimeError("focused exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return self._focused_closed_frame(frame, current_time)

    def _focused_require_columns(self, frame: DataFrame) -> None:
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f"focused source invalidation is missing required columns: {missing}")

    def _focused_side(self, trade: Any) -> str:
        side = "short" if bool(getattr(trade, "is_short", False)) else "long"
        declared = str(self.FOCUSED_SOURCE_PROFILE["side"])
        if side != declared:
            raise ValueError(f"trade side {side!r} violates focused profile side {declared!r}")
        return side

    def _focused_plan(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        name = str((state or {}).get('plan') or self.exit_plan.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['invalidation_confirmations'] = int(self.invalidation_confirmations.value)
        plan['invalidation_band'] = float(self.invalidation_band_quarter_percent.value) * 0.0025
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
        plan['max_hold_candles'] = max(1, int(round(float(plan.get('max_hold_candles', 336)) * float(self.max_hold_scale.value) / 2.0)))
        return plan

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, Mapping):
            raise ValueError("focused exit trade state must be a mapping")
        if state.get("version") != self.FOCUSED_STATE_VERSION:
            raise ValueError("focused exit trade state version mismatch")
        if state.get("contract") != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError("focused exit trade state contract mismatch")
        return dict(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=dict(state))

    def _focused_new_state(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        order: Any | None = None,
    ) -> dict[str, Any] | None:
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, "safe_price", None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, "open_rate", None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError("focused exit runtime requires a positive entry rate")
        freeze_time = (
            (getattr(order, "order_date_utc", None) if order else None)
            or (getattr(order, "order_date", None) if order else None)
            or getattr(trade, "open_date_utc", None)
            or getattr(trade, "date_entry_fill_utc", None)
            or current_time
        )
        frame = self._focused_analyzed_frame(pair, freeze_time)
        if frame.empty:
            return None
        self._focused_require_columns(frame)
        row = frame.iloc[-1]
        level_column = str(self.FOCUSED_SOURCE_PROFILE["invalidation"]["level"])
        invalidation_level = self._focused_float(row[level_column])
        if invalidation_level is None or invalidation_level <= 0.0:
            raise ValueError(f"required source invalidation level must be positive and finite: {level_column}")
        valid = invalidation_level > entry_rate if side == "short" else invalidation_level < entry_rate
        if not valid:
            raise ValueError(f"required source invalidation level is on the wrong side of entry: {level_column}")
        fill_time = getattr(trade, "date_entry_fill_utc", None) or getattr(trade, "open_date_utc", None)
        if fill_time is None:
            raise ValueError("focused exit runtime requires an entry fill timestamp")
        fill_timestamp = self._focused_utc(fill_time)
        return {
            "version": self.FOCUSED_STATE_VERSION,
            "contract": self.FOCUSED_EXIT_CONTRACT,
            "plan": str(self.exit_plan.value),
            "side": side,
            "entry_rate": float(entry_rate),
            "entry_filled_at": fill_timestamp.isoformat() if fill_timestamp is not None else None,
            "levels": {"invalidation": invalidation_level},
            "invalidation_seen": False,
        }

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any], current_time: Any) -> DataFrame:
        filled_at = self._focused_utc(state.get("entry_filled_at"))
        if filled_at is None:
            raise ValueError("focused exit state has no entry fill timestamp")
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError("current_time is required for post-entry evaluation")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = self._focused_close_times(frame)
        return frame.loc[dates.ge(filled_at) & close_times.le(now)]

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
    def _focused_hard_stop_price(plan: Mapping[str, Any], state: Mapping[str, Any]) -> float:
        entry_rate = float(state["entry_rate"])
        ratio = float(plan["hard_stop_ratio"])
        return entry_rate * (1.0 + ratio if state["side"] == "short" else 1.0 - ratio)

    def _focused_invalidation_event(
        self,
        post_entry: DataFrame,
        state: dict[str, Any],
        plan: Mapping[str, Any],
    ) -> bool:
        if state.get("invalidation_seen"):
            return True
        if post_entry.empty:
            return False
        level = self._focused_float((state.get("levels") or {}).get("invalidation"))
        if level is None or level <= 0.0:
            level_column = str(self.FOCUSED_SOURCE_PROFILE["invalidation"]["level"])
            raise ValueError(f"active trade state requires a positive finite invalidation level: {level_column}")
        close = pd.to_numeric(post_entry["close"], errors="coerce")
        band = float(plan["invalidation_band"])
        condition = close.gt(level * (1.0 + band)) if state["side"] == "short" else close.lt(level * (1.0 - band))
        confirmed = self._focused_any_n(condition, int(plan["invalidation_confirmations"]))
        if confirmed:
            state["invalidation_seen"] = True
        return confirmed

    def _focused_context(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return self._focused_plan(), {}, None
        before = repr(state)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state, current_time)
        hard_stop = self._focused_hard_stop_price(plan, state)
        hard_stop_breached = current_rate >= hard_stop if state["side"] == "short" else current_rate <= hard_stop
        if hard_stop_breached:
            exit_tag = "focused_hard_stop"
        elif self._focused_invalidation_event(post_entry, state, plan):
            exit_tag = "focused_source_invalidation"
        elif len(post_entry) >= int(plan["max_hold_candles"]):
            exit_tag = "focused_max_hold"
        else:
            exit_tag = None
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return plan, state, exit_tag

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = (current_profit, kwargs)
        _, _, exit_tag = self._focused_context(pair, trade, current_time, current_rate)
        return exit_tag

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
        _ = (current_profit, after_fill, kwargs)
        plan, state, exit_tag = self._focused_context(pair, trade, current_time, current_rate)
        if not state:
            return None
        if exit_tag is not None:
            return None
        stop_price = self._focused_hard_stop_price(plan, state)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=state["side"] == "short",
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None):
            if self._focused_state(trade) is None:
                state = self._focused_new_state(pair, trade, current_time, order)
                if state is None:
                    return None
                self._focused_save_state(trade, state)
