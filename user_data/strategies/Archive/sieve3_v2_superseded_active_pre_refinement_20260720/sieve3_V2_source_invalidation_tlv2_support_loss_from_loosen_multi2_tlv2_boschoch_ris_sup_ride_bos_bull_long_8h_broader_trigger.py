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
from user_data.Indicators.market_state import add_market_state
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_TAG = "loosen_multi2_tlv2_boschoch_ris_sup_ride_bos_bull_long_8h_broader_trigger"
SIDE = "long"
TIMEFRAME = "8h"
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = "Sieve3ExitFromLoosenMulti2Tlv2BoschochRisSupRideBosBullLong8hBroaderTrigger"
SOURCE_STRATEGY = (
    "user_data/strategies/"
    "sieve3_exit_from_loosen_multi2_tlv2_boschoch_ris_sup_ride_bos_bull_long_8h_broader_trigger.py:"
    f"{SOURCE_ENTRY_CLASS}"
)
LINEAGE_STATUS = "historical_promotion_lineage_unknown"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
RESEARCH_PATH = 'sieve3_exit_source_invalidation_tlv2_support_loss'
EXIT_FAMILY = "source_invalidation"
EXIT_THEORY = "source_invalidation_tlv2_support_loss"
EXIT_HYPOTHESIS = (
    "A closed-candle loss of the entry-frozen TLV2 rising support invalidates the long support-ride thesis."
)
PRIMARY_TRIGGER = "8h TLV2 rising-support ride confirmed by bullish BOS and non-bearish BOS/CHoCH state"
PRIMARY_GUARD = "always-on Sieve2 VP score-or-context and market pressure-or-trend guards"
SECONDARY_GUARD = "locked volume-ratio and bullish directional-volume pressure guards"
TARGET_PROVIDER = "none"
INVALIDATION_PROVIDER = (
    "provider=entry-frozen TLV2 rising-support loss;mode=level;long.level=tlv2_support_line_rank0"
)
ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')



def _required_num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required fixed-entry numeric column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _required_bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required fixed-entry boolean column is missing: {column}")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


class Sieve3V2SourceInvalidationTlv2SupportLossFromLoosenMulti2Tlv2BoschochRisSupRideBosBullLong8hBroaderTrigger(
    IStrategy
):
    """Locked TLV2/BOS long entry with frozen TLV2 support-loss invalidation."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0

    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.15
    sieve2_vp_context_min = 0.15
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.03
    sieve2_market_trend_min = 0.15

    volume_guard_window = 24
    volume_ratio_min = 1.3
    pressure_window = 48
    pressure_min = 0.05

    pivot_strength = 2
    min_line_score = 0.6
    min_active_bars = 8
    max_distance_atr = 3.0
    proximity_rank_weight = 0.05
    line_slope_min_pct = 0.0015

    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.0

    LOCKED_BUY_PARAMS = {
        "sieve2_vp_window": 96,
        "sieve2_vp_bins": 36,
        "sieve2_vp_score_min": 0.15,
        "sieve2_vp_context_min": 0.15,
        "sieve2_market_window": 24,
        "sieve2_market_pressure_min": 0.03,
        "sieve2_market_trend_min": 0.15,
        "volume_guard_window": 24,
        "volume_ratio_min": 1.3,
        "pressure_window": 48,
        "pressure_min": 0.05,
        "pivot_strength": 2,
        "min_line_score": 0.6,
        "min_active_bars": 8,
        "max_distance_atr": 3.0,
        "proximity_rank_weight": 0.05,
        "line_slope_min_pct": 0.0015,
        "strength": 3,
        "min_prominence_atr": 0.35,
        "min_pivot_spacing_bars": 2,
        "max_pivot_age_bars": 96,
        "breakout_buffer_atr": 0.0,
    }

    SOURCE_ENTRY_STEM = ENTRY_TAG
    FOCUSED_EXIT_CONTRACT = "source_invalidation"
    FOCUSED_SOURCE_PROFILE = {
        "side": "long",
        "invalidation": {
            "provider": "entry-frozen TLV2 rising-support loss",
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
        "Sieve3V2SourceInvalidationTlv2SupportLossFromLoosenMulti2Tlv2BoschochRisSupRideBosBullLong8hBroaderTrigger:"
        "source_invalidation"
    )
    ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')

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
        optimize=True,
        load=True,
    )
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
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

    def _add_sieve2_guard_indicators(self, dataframe: DataFrame) -> DataFrame:
        frame = add_market_state(dataframe.copy(), window=self.sieve2_market_window, prefix="s2m")
        return add_volume_profile(
            frame,
            window=self.sieve2_vp_window,
            bins=self.sieve2_vp_bins,
            value_area_pct=0.70,
            price_source="hlc3",
            smooth_bins=3,
            pressure_delta_min=0.05,
            node_near_pct=0.01,
            prefix="s2vp",
        )

    def _sieve2_guards(self, dataframe: DataFrame) -> Series:
        vp_score = _required_num(dataframe, "s2vp_score_long")
        vp_other = _required_num(dataframe, "s2vp_score_short")
        vp_context = _required_num(dataframe, "s2vp_context_score_bull")
        vp_guard = vp_score.ge(self.sieve2_vp_score_min) & vp_score.gt(vp_other)
        vp_guard |= vp_context.ge(self.sieve2_vp_context_min)

        market_pressure = _required_num(dataframe, "s2m_pressure_ratio")
        market_trend = _required_num(dataframe, "s2m_trend_z")
        market_guard = market_pressure.ge(self.sieve2_market_pressure_min)
        market_guard |= market_trend.ge(self.sieve2_market_trend_min)
        return (vp_guard & market_guard).fillna(False)

    def _common_guards(self, dataframe: DataFrame) -> Series:
        close = _required_num(dataframe, "close")
        open_ = _required_num(dataframe, "open")
        high = _required_num(dataframe, "high")
        low = _required_num(dataframe, "low")
        volume = _required_num(dataframe, "volume").clip(lower=0.0)

        volume_baseline = (
            volume.shift(1)
            .rolling(self.volume_guard_window, min_periods=max(2, self.volume_guard_window // 3))
            .mean()
            .replace(0.0, np.nan)
        )
        volume_guard = volume.ge(volume_baseline.mul(self.volume_ratio_min))

        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume.fillna(0.0)).fillna(0.0)
        pressure_baseline = (
            volume.fillna(0.0)
            .rolling(self.pressure_window, min_periods=max(2, self.pressure_window // 3))
            .sum()
            .replace(0.0, np.nan)
        )
        pressure_ratio = directional_volume.rolling(
            self.pressure_window,
            min_periods=max(2, self.pressure_window // 3),
        ).sum() / pressure_baseline
        return (volume_guard & pressure_ratio.ge(self.pressure_min)).fillna(False)

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            pivot_strength=self.pivot_strength,
            raw_line_output_count=1,
            min_output_line_score=self.min_line_score,
            min_output_active_bars=self.min_active_bars,
            max_active_line_distance_atr_mult=self.max_distance_atr,
            proximity_rank_weight=self.proximity_rank_weight,
            output_prefix="tlv2",
        )

    def _tlv2_trigger(self, dataframe: DataFrame) -> Series:
        close = _required_num(dataframe, "close")
        line = _required_num(dataframe, "tlv2_support_line_rank0")
        score = _required_num(dataframe, "tlv2_support_score_rank0")
        distance = _required_num(dataframe, "tlv2_support_distance_atr_rank0")
        active = score.ge(self.min_line_score) & distance.le(self.max_distance_atr)
        rising = line.pct_change(fill_method=None).gt(self.line_slope_min_pct)
        return (active & rising & close.gt(line)).fillna(False)

    def _add_bos_choch(self, dataframe: DataFrame) -> DataFrame:
        return add_bos_choch(
            dataframe,
            strength=self.strength,
            min_prominence_atr=self.min_prominence_atr,
            min_pivot_spacing_bars=self.min_pivot_spacing_bars,
            max_pivot_age_bars=self.max_pivot_age_bars,
            breakout_buffer_atr=self.breakout_buffer_atr,
            prefix="ms",
        )

    def _ms_confirm(self, dataframe: DataFrame) -> Series:
        bos_bull = _required_bool(dataframe, "ms_bos_to_bull")
        state = _required_num(dataframe, "ms_state")
        return (bos_bull & state.ge(0)).fillna(False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_tlv2(dataframe)
        dataframe = self._add_bos_choch(dataframe)
        return self._add_sieve2_guard_indicators(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._tlv2_trigger(dataframe) & self._ms_confirm(dataframe)
        condition &= self._common_guards(dataframe)
        condition &= self._sieve2_guards(dataframe)
        valid = condition.fillna(False) & _required_num(dataframe, "volume").gt(0.0)
        valid &= _required_num(dataframe, "close").notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
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
