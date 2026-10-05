"""Sieve3 V2 integrated exits for the promoted TLV2/VP long entry.

Preflight contract
------------------
Canonical Sieve2 source:
    sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h
Promotion evidence:
    20260521T012740_entry_all_resume, selected row 319
Promoted baseline:
    TP/SL 2%/2%, 219 trades, 58.9041% win rate, 6.815958% return,
    1.93344% maximum drawdown, 1.372107 profit factor.
Entry lock:
    The 45 promoted buy defaults below are immutable in Sieve3 V2
    (``optimize=False, load=False``). Disabled Sieve2 optional guards remain
    represented by their locked parameters but their dead indicator work is omitted.
Tested legacy controls:
    sieve3_exit_breakeven_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h.py
    sieve3_exit_level_zone_reversal_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h.py
    jobs 20260709T101515_entry_sieve3_exit_speed_batch_046 and
    20260628T025333_entry_sieve3_exit_speed_batch_040.
Refined pilot controls:
    sieve3_rework_exit_{invalidation,layered,target_zone}_from_multi2_*.py
    job 20260711T230600_entry_sieve3_exit_rework_comparison_001.
Consolidation:
    The V2 plans retain the distinct target, invalidation, layered, runner,
    and time/progress hypotheses. Generic legacy EMA/oscillator exit surfaces
    and duplicate one-file-per-exit branches are intentionally not copied.
Target semantics:
    All TLV2, VP, prior-high, and measured-move targets are frozen from the
    last closed signal candle when the entry fills. Provider chains are explicit
    and may only move to a named, source-coherent fallback at initialization.
    TLV2 rank-0 support is deliberately excluded because prefix testing found
    its historical rank selection unstable; shifted 48-candle prior low is the
    causal structural floor used for measured range and hard invalidation.
Invalidation:
    Broken-resistance loss, VP bull-context failure, their conjunction, and
    frozen structural-support failure. Structural-support failure or a confirmed
    broken-level loss plus VP failure is the universal hard thesis failure.
Hyperopt surface:
    One sell-space ``exit_policy_plan`` parameter. Each category is a complete,
    trader-readable policy; no inactive branch thresholds are optimized.
Split decision:
    One V2 file is sufficient because TLV2 structure and VP context form one
    coherent target/invalidation ecosystem for this entry foundation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.market_state import add_market_state
from user_data.Indicators.complex_trendline_projection_v2 import (
    add_trendline_projection_v2,
)
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_TAG = "multi2_tlv2_vp_res_break_vp_bullctx_long_1h"
STATE_KEY = "sieve3_v2_tlv2_vp_long"
STATE_VERSION = 3
LEVEL_BAND = 0.005
MIN_TARGET_MOVE = 0.002
PARTIAL_TAGS = {"s3v2_partial_target", "s3v2_partial_invalidation"}

SIEVE2_VP_GUARD_MODES = (
    "score_or_context",
    "node_confirm",
    "value_area_confirm",
    "breakout_acceptance",
    "rejection_confirm",
    "poc_hvn_reject",
    "prior_level_confirm",
)
SIEVE2_MARKET_GUARD_MODES = (
    "pressure_or_trend",
    "pressure_and_trend",
    "directional_pressure",
    "trend_state",
    "avoid_adverse_pressure",
    "avoid_chop",
)


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))


def _finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


DEFAULT_FALLBACK_ROI = 0.12
DEFAULT_MAX_HOLD_CANDLES = 336


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_integrated"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h"
EXIT_HYPOTHESIS = "Test the integrated exit family while preserving the multi2 tlv2 vp res break vp bullctx long 1h entry behavior."


class ExitPlan:
    __slots__ = (
        "confirmation",
        "fixed_tp",
        "hard_stop",
        "invalidation",
        "invalidation_action",
        "max_hold_candles",
        "partial_fraction",
        "progress_candles",
        "remainder",
        "role",
        "target_1",
        "target_2",
        "target_action",
        "target_band",
    )

    def __init__(
        self,
        role: str,
        target_1: str | None = None,
        target_2: str | None = None,
        confirmation: str = "touch",
        target_action: str = "hold",
        invalidation: str = "combined1",
        invalidation_action: str = "full",
        partial_fraction: float = 0.0,
        remainder: str = "hold",
        progress_candles: int = 0,
        target_band: float = 0.01,
        hard_stop: float = 0.05,
        fixed_tp: float | None = None,
        max_hold_candles: int | None = None,
    ) -> None:
        self.role = role
        self.target_1 = target_1
        self.target_2 = target_2
        self.confirmation = confirmation
        self.target_action = target_action
        self.invalidation = invalidation
        self.invalidation_action = invalidation_action
        self.partial_fraction = float(partial_fraction)
        self.remainder = remainder
        self.progress_candles = int(progress_candles)
        self.target_band = float(target_band)
        self.hard_stop = float(hard_stop)
        self.fixed_tp = float(
            DEFAULT_FALLBACK_ROI
            if fixed_tp is None and role != "baseline"
            else fixed_tp or 0.0
        )
        self.max_hold_candles = int(
            DEFAULT_MAX_HOLD_CANDLES
            if max_hold_candles is None and role != "baseline"
            else max_hold_candles or 0
        )

    def _values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__slots__)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExitPlan) and self._values() == other._values()

    def __hash__(self) -> int:
        return hash(self._values())


class ExitDecision:
    __slots__ = ("action", "fraction", "tag", "tighten")

    def __init__(
        self,
        action: str,
        tag: str | None = None,
        fraction: float = 0.0,
        tighten: str | None = None,
    ) -> None:
        self.action = action
        self.tag = tag
        self.fraction = float(fraction)
        self.tighten = tighten

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExitDecision) and (
            self.action,
            self.tag,
            self.fraction,
            self.tighten,
        ) == (
            other.action,
            other.tag,
            other.fraction,
            other.tighten,
        )


EXIT_PLANS: dict[str, ExitPlan] = {
    "baseline_3_3": ExitPlan(
        "baseline",
        invalidation="none",
        invalidation_action="hold",
        hard_stop=0.03,
        fixed_tp=0.03,
    ),
    "promotion_2_2": ExitPlan(
        "baseline",
        invalidation="none",
        invalidation_action="hold",
        hard_stop=0.02,
        fixed_tp=0.02,
    ),
    "hvn_touch_full": ExitPlan(
        "target_full",
        "hvn",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "hvn_rev1_full": ExitPlan(
        "target_full",
        "hvn",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "vah_rev1_full": ExitPlan(
        "target_full",
        "vah",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.015,
    ),
    "prior48_rev1_full": ExitPlan(
        "target_full",
        "prior48",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.03,
    ),
    "prior96_rev2of3_full": ExitPlan(
        "target_full",
        "prior96",
        confirmation="reversal2of3",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.03,
    ),
    "measured_half_touch_full": ExitPlan(
        "target_full",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "measured_full_rev1_full": ExitPlan(
        "target_full",
        "measured_full",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "level1_full": ExitPlan("invalidation_full", invalidation="level1", invalidation_action="full"),
    "level2_full": ExitPlan("invalidation_full", invalidation="level2", invalidation_action="full"),
    "context2_full": ExitPlan(
        "invalidation_full", invalidation="context2", invalidation_action="full"
    ),
    "combined1_full": ExitPlan(
        "invalidation_full", invalidation="combined1", invalidation_action="full"
    ),
    "level2_tighten": ExitPlan(
        "defensive_tighten", invalidation="level2", invalidation_action="tighten"
    ),
    "context2_tighten": ExitPlan(
        "defensive_tighten", invalidation="context2", invalidation_action="tighten"
    ),
    "hvn_touch_p33_be_level2": ExitPlan(
        "target_partial",
        "hvn",
        confirmation="touch",
        target_action="partial",
        invalidation="level2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "hvn_rev1_p50_trail_level2": ExitPlan(
        "target_partial",
        "hvn",
        confirmation="reversal1",
        target_action="partial",
        invalidation="level2",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "hvn_rev1_p33_be_combined": ExitPlan(
        "target_partial",
        "hvn",
        confirmation="reversal1",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "vah_rev1_p33_be_context2": ExitPlan(
        "target_partial",
        "vah",
        confirmation="reversal1",
        target_action="partial",
        invalidation="context2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "prior48_touch_p33_trail_level2": ExitPlan(
        "target_partial",
        "prior48",
        confirmation="touch",
        target_action="partial",
        invalidation="level2",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "prior96_rev2of3_p50_trail_combined": ExitPlan(
        "target_partial",
        "prior96",
        confirmation="reversal2of3",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "measured_half_touch_p33_be_level2": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="touch",
        target_action="partial",
        invalidation="level2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "measured_half_rev1_p50_trail_context2": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="reversal1",
        target_action="partial",
        invalidation="context2",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "measured_full_rev1_p33_trail_combined": ExitPlan(
        "target_partial",
        "measured_full",
        confirmation="reversal1",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "nearest_rev1_p50_trail_combined": ExitPlan(
        "target_partial",
        "nearest",
        confirmation="reversal1",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "level1_reduce33_hvn": ExitPlan(
        "invalidation_reduce",
        "hvn",
        confirmation="touch",
        target_action="full",
        invalidation="level1",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "context2_reduce33_measured": ExitPlan(
        "invalidation_reduce",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="context2",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "combined1_reduce50_nearest": ExitPlan(
        "invalidation_reduce",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="combined1",
        invalidation_action="partial",
        partial_fraction=0.50,
        remainder="target",
    ),
    "hvn_touch_tighten_level2": ExitPlan(
        "target_tighten",
        "hvn",
        confirmation="touch",
        target_action="tighten",
        invalidation="level2",
        invalidation_action="full",
        remainder="tighten",
    ),
    "measured_half_touch_tighten_combined": ExitPlan(
        "target_tighten",
        "measured_half",
        confirmation="touch",
        target_action="tighten",
        invalidation="combined1",
        invalidation_action="full",
        remainder="tighten",
    ),
    "hvn_then_measured_runner": ExitPlan(
        "dual_target",
        "hvn",
        "measured_full",
        "touch",
        "partial",
        "combined1",
        "full",
        0.33,
        "trail",
    ),
    "prior48_then_measured_runner": ExitPlan(
        "dual_target",
        "prior48",
        "measured_full",
        "touch",
        "partial",
        "level2",
        "full",
        0.50,
        "trail",
    ),
    "progress48_nearest": ExitPlan(
        "progress_failure",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="combined1",
        invalidation_action="full",
        progress_candles=48,
    ),
    "progress96_measured": ExitPlan(
        "progress_failure",
        "measured_full",
        confirmation="touch",
        target_action="full",
        invalidation="combined1",
        invalidation_action="full",
        progress_candles=96,
    ),
}


TARGET_PROVIDER_CHAINS: dict[str, tuple[str, ...]] = {
    "hvn": ("hvn", "vah", "prior48", "measured_half"),
    "vah": ("vah", "hvn", "prior48", "measured_half"),
    "prior48": ("prior48", "hvn", "measured_half"),
    "prior96": ("prior96", "prior48", "hvn", "measured_half"),
    "measured_half": ("measured_half", "prior48", "hvn"),
    "measured_full": ("measured_full", "prior96", "hvn"),
    "nearest": ("hvn", "vah", "prior48", "prior96", "measured_half", "measured_full"),
}


ACTIVE_SELL_PARAMS = ("exit_policy_plan",)


class Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("exit_policy_plan",)

    minimal_roi = {}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    use_custom_roi = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    trailing_stop = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Legacy fixed-off entry parameter: use_sieve2_vp_guard=False (BooleanParameter, buy space).
    # Legacy inert entry parameter: sieve2_vp_guard_mode='score_or_context' (fixed entry mode/gate).
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    # Legacy inert entry parameter: sieve2_vp_score_min=0.25 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_vp_context_min=0.28 (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: use_sieve2_market_guard=False (BooleanParameter, buy space).
    # Legacy inert entry parameter: sieve2_market_guard_mode='pressure_or_trend' (fixed entry mode/gate).
    sieve2_market_window = 24
    # Legacy inert entry parameter: sieve2_market_pressure_min=0.07 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_market_trend_min=0.25 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_rs_score_min=0.45 (fixed entry mode/gate).

    # Legacy inert entry parameter: use_volume_guard=True (fixed entry mode/gate).
    volume_guard_window = 24
    volume_ratio_min = 1.6
    # Legacy inert entry parameter: use_pressure_guard=True (fixed entry mode/gate).
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).

    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = 'hlc3'
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
    vp_context_min = 0.28
    # Legacy inert entry parameter: vp_level_buffer_pct=0.016 (fixed entry mode/gate).

    pivot_strength = 2
    min_line_score = 0.6
    min_active_bars = 8
    max_distance_atr = 3.0
    proximity_rank_weight = 0.05
    line_buffer_pct = 0.0
    # Legacy inert entry parameter: line_slope_min_pct=0.0005 (fixed entry mode/gate).

    exit_policy_plan = CategoricalParameter(
        list(EXIT_PLANS),
        default="baseline_3_3",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_policy_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        _ = (
            pair,
            current_time,
            current_rate,
            proposed_leverage,
            max_leverage,
            entry_tag,
            side,
            kwargs,
        )
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        volume = _num(dataframe, "volume").clip(lower=0.0)

        volume_window = int(self.volume_guard_window)
        volume_baseline = (
            volume.shift(1)
            .rolling(volume_window, min_periods=max(2, volume_window // 3))
            .mean()
            .replace(0.0, np.nan)
        )
        guard &= volume.ge(volume_baseline.mul(float(self.volume_ratio_min)))

        pressure_ratio = (
            _num(dataframe, 's2m_pressure_ratio')
        )
        guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(
            dataframe,
            window=int(self.vp_window),
            bins=int(self.vp_bins),
            value_area_pct=float(self.vp_value_area_pct),
            price_source=str(self.vp_price_source),
            smooth_bins=int(self.vp_smooth_bins),
            hvn_threshold=float(self.vp_hvn_threshold),
            lvn_threshold=float(self.vp_lvn_threshold),
            pressure_delta_min=float(self.vp_pressure_delta_min),
            node_near_pct=float(self.vp_node_near_pct),
            volume_percentile_min=float(self.vp_volume_percentile_min),
            score_window=int(self.vp_score_window),
            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult),
            entry_score_margin=float(self.vp_entry_score_margin),
            prefix="vp",
        )

    def _add_sieve2_guard_indicators(self, dataframe: DataFrame) -> DataFrame:
        frame = add_market_state(dataframe.copy(), window=int(self.sieve2_market_window), prefix="s2m")
        return add_volume_profile(
            frame,
            window=int(self.sieve2_vp_window),
            bins=int(self.sieve2_vp_bins),
            value_area_pct=0.70,
            price_source="hlc3",
            smooth_bins=3,
            pressure_delta_min=0.05,
            node_near_pct=0.01,
            prefix="s2vp",
        )

    @staticmethod
    def _sieve2_vp_guard(
        frame: DataFrame,
        mode: str,
        score_min: float,
        context_min: float,
    ) -> Series:
        close = _num(frame, "close")
        low = _num(frame, "low")
        score = _num(frame, "s2vp_score_long")
        other = _num(frame, "s2vp_score_short")
        context = _num(frame, "s2vp_context_score_bull")
        in_value = _bool(frame, "s2vp_in_value_area")
        above_value = _bool(frame, "s2vp_above_value_area")
        node_entry = _bool(frame, "s2vp_node_entry_long")
        node_hold = _bool(frame, "s2vp_node_hold_long")
        prior_vah = _num(frame, "s2vp_prior_vah")
        prior_val = _num(frame, "s2vp_prior_val")
        base_score = score.ge(score_min) & score.gt(other)
        base_context = context.ge(context_min)
        if mode == "node_confirm":
            return (node_entry | node_hold | (base_score & base_context)).fillna(False)
        if mode == "value_area_confirm":
            return (above_value | (in_value & base_context)).fillna(False)
        if mode == "breakout_acceptance":
            return (
                close.gt(prior_vah) & (base_score | base_context | node_entry | node_hold)
            ).fillna(False)
        if mode == "rejection_confirm":
            rejection = low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah)
            return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
        if mode == "poc_hvn_reject":
            return (node_entry | (base_score & base_context)).fillna(False)
        if mode == "prior_level_confirm":
            return (close.gt(prior_vah) & (base_score | base_context)).fillna(False)
        return (base_score | base_context).fillna(False)

    @staticmethod
    def _sieve2_market_guard(
        frame: DataFrame,
        mode: str,
        pressure_min: float,
        trend_min: float,
    ) -> Series:
        pressure = _num(frame, "s2m_pressure_ratio")
        trend = _num(frame, "s2m_trend_z")
        directional_pressure = pressure.ge(pressure_min)
        trend_state = trend.ge(trend_min)
        avoids_adverse_pressure = pressure.ge(-pressure_min)
        avoids_adverse_trend = trend.ge(-trend_min)
        if mode == "pressure_and_trend":
            return (directional_pressure & trend_state).fillna(False)
        if mode == "directional_pressure":
            return directional_pressure.fillna(False)
        if mode == "trend_state":
            return trend_state.fillna(False)
        if mode == "avoid_adverse_pressure":
            return (avoids_adverse_pressure & avoids_adverse_trend).fillna(False)
        if mode == "avoid_chop":
            return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
        return (directional_pressure | trend_state).fillna(False)

    def _apply_sieve2_optional_guards(self, dataframe: DataFrame, condition: Series) -> Series:
        guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
        return guarded.fillna(False)

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        score_long = _num(dataframe, "vp_score_long")
        bull_context = _num(dataframe, "vp_context_score_bull")
        bear_context = _num(dataframe, "vp_context_score_bear")
        market = _num(dataframe, "vp_market_context")
        return (
            score_long.ge(float(self.vp_score_min))
            & bull_context.ge(float(self.vp_context_min))
            & bull_context.ge(bear_context)
            & market.ge(0.0)
        )

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            pivot_strength=int(self.pivot_strength),
            raw_line_output_count=1,
            min_output_line_score=float(self.min_line_score),
            min_output_active_bars=int(self.min_active_bars),
            max_active_line_distance_atr_mult=float(self.max_distance_atr),
            proximity_rank_weight=float(self.proximity_rank_weight),
            output_prefix="tlv2",
        )

    def _tlv2_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        line = _num(dataframe, "tlv2_resistance_line_rank0")
        score = _num(dataframe, "tlv2_resistance_score_rank0")
        distance = _num(dataframe, "tlv2_resistance_distance_atr_rank0")
        active = score.ge(float(self.min_line_score)) & distance.le(
            float(self.max_distance_atr)
        )
        return active & _cross_above(close, line.mul(1.0 + float(self.line_buffer_pct)))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = self._add_tlv2(dataframe)
        frame = self._add_vp(frame)
        frame = self._add_sieve2_guard_indicators(frame)
        high = _num(frame, "high")
        low = _num(frame, "low")
        frame["s3v2_prior_high_48"] = high.shift(1).rolling(48, min_periods=12).max()
        frame["s3v2_prior_high_96"] = high.shift(1).rolling(96, min_periods=24).max()
        frame["s3v2_prior_low_48"] = low.shift(1).rolling(48, min_periods=12).min()
        return frame

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._tlv2_trigger(dataframe) & self._vp_confirm(dataframe)
        condition &= self._common_guards(dataframe)
        condition = self._apply_sieve2_optional_guards(dataframe, condition)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
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
    def _utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _closed_frame(self, frame: DataFrame, current_time: Any) -> DataFrame:
        if frame is None or frame.empty:
            return DataFrame()
        if current_time is None or "date" not in frame.columns:
            return frame
        now = self._utc(current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(now)]

    def _analyzed_frame(self, pair: str, current_time: Any) -> DataFrame:
        if not getattr(self, "dp", None):
            return DataFrame()
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return self._closed_frame(frame, current_time)

    @staticmethod
    def _directional_target(value: Any, entry_rate: float) -> float | None:
        target = _finite_float(value)
        if target is None or target <= entry_rate * (1.0 + MIN_TARGET_MOVE):
            return None
        return target

    def _frozen_targets(
        self, row: Series | None, entry_rate: float
    ) -> tuple[dict[str, float | None], dict[str, str | None], float | None, float | None]:
        if row is None:
            raise RuntimeError("integrated TLV2/VP target snapshot requires a closed entry candle")
        required = {
            "tlv2_resistance_line_rank0",
            "vp_hvn_above",
            "vp_vah",
            "s3v2_prior_high_48",
            "s3v2_prior_high_96",
            "s3v2_prior_low_48",
        }
        missing = sorted(required - set(row.index))
        if missing:
            raise KeyError(f"integrated TLV2/VP target snapshot missing required source columns: {missing}")
        broken = _finite_float(row["tlv2_resistance_line_rank0"])
        support = _finite_float(row["s3v2_prior_low_48"])
        if support is not None and support >= entry_rate:
            support = None

        raw: dict[str, float | None] = {
            "hvn": self._directional_target(row["vp_hvn_above"], entry_rate),
            "vah": self._directional_target(row["vp_vah"], entry_rate),
            "prior48": self._directional_target(row["s3v2_prior_high_48"], entry_rate),
            "prior96": self._directional_target(row["s3v2_prior_high_96"], entry_rate),
            "measured_half": None,
            "measured_full": None,
        }
        if broken is not None and support is not None and broken > support:
            measured_range = broken - support
            raw["measured_half"] = self._directional_target(
                broken + measured_range * 0.5, entry_rate
            )
            raw["measured_full"] = self._directional_target(broken + measured_range, entry_rate)

        targets: dict[str, float | None] = {}
        providers: dict[str, str | None] = {}
        for requested, chain in TARGET_PROVIDER_CHAINS.items():
            choices = [(name, raw.get(name)) for name in chain if raw.get(name) is not None]
            if requested == "nearest" and choices:
                provider, target = min(choices, key=lambda item: float(item[1]))
            elif choices:
                provider, target = choices[0]
            else:
                provider, target = None, None
            targets[requested] = float(target) if target is not None else None
            providers[requested] = provider
        return targets, providers, broken, support

    def _new_state(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: Any,
    ) -> dict[str, Any]:
        order_rate = _finite_float(getattr(order, "safe_price", None))
        entry_rate = order_rate or float(getattr(trade, "open_rate", 0.0) or 0.0)
        placement_time = getattr(order, "order_date_utc", None)
        if placement_time is None:
            placement_time = getattr(order, "order_date", None)
        frame = self._analyzed_frame(pair, placement_time or current_time)
        row = frame.iloc[-1] if not frame.empty else None
        targets, providers, broken, support = self._frozen_targets(row, entry_rate)
        plan_name = str(self.exit_policy_plan.value)
        plan = EXIT_PLANS[plan_name]
        required_targets = tuple(
            target_name for target_name in (plan.target_1, plan.target_2) if target_name
        )
        resolved_targets = {
            target_name: _finite_float(targets.get(target_name))
            for target_name in required_targets
        }
        unusable_targets = [
            target_name
            for target_name, target in resolved_targets.items()
            if target is None
            or target <= 0.0
            or target <= entry_rate * (1.0 + MIN_TARGET_MOVE)
        ]
        if unusable_targets:
            raise ValueError(
                f"{plan_name} requires usable entry-frozen targets: {unusable_targets}"
            )
        if plan.target_1 and plan.target_2:
            first = resolved_targets[plan.target_1]
            second = resolved_targets[plan.target_2]
            if second <= first * (1.0 + MIN_TARGET_MOVE):
                raise ValueError(
                    f"{plan_name} requires target_2 beyond target_1"
                )

        candle_date = None
        if row is not None and "date" in row.index:
            timestamp = self._utc(row["date"])
            candle_date = timestamp.isoformat() if timestamp is not None else None
        filled_at = (
            getattr(order, "order_filled_utc", None)
            or getattr(trade, "date_entry_fill_utc", None)
            or getattr(trade, "open_date_utc", None)
            or current_time
        )
        filled_timestamp = self._utc(filled_at)
        return {
            "version": STATE_VERSION,
            "plan": str(self.exit_policy_plan.value),
            "phase": "ENTRY",
            "entry_rate": entry_rate,
            "entry_candle": candle_date,
            "entry_filled_at": (
                filled_timestamp.isoformat() if filled_timestamp is not None else None
            ),
            "broken_resistance": broken,
            "structural_support": support,
            "targets": targets,
            "target_providers": providers,
            "target_1_touched_at": None,
            "target_1_confirmation_latched": False,
            "target_2_touched_at": None,
            "partial_filled": False,
            "partial_filled_at": None,
            "partial_tag": None,
            "partial_target_stake": None,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
            "partial_order_credits": {},
            "partial_resolved_order_ids": [],
            "partial_status": "ready",
            "partial_requested_at": None,
            "partial_requested_stake": None,
            "partial_active_order_id": None,
            "terminal_exit_pending": False,
            "terminal_exit_tag": None,
            "stop_floor": None,
        }

    @staticmethod
    def _state(trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=STATE_KEY)
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            return None
        return dict(state)

    @staticmethod
    def _save_state(trade: Any, state: dict[str, Any]) -> None:
        persistent = dict(state)
        persistent.pop("partial_pending", None)
        trade.set_custom_data(key=STATE_KEY, value=persistent)

    @staticmethod
    def _has_open_order(trade: Any) -> bool:
        return bool(trade.open_orders)

    @staticmethod
    def _has_pending_partial(trade: Any, state: dict[str, Any]) -> bool:
        exit_side = getattr(trade, "exit_side", None)
        has_open_partial = any(
            getattr(order, "ft_order_side", None) == exit_side
            and str(getattr(order, "ft_order_tag", None) or "") in PARTIAL_TAGS
            for order in trade.open_orders
        )
        return (
            has_open_partial
            or state.get("partial_status") == "requested"
            or state.get("partial_active_order_id") is not None
        )

    @staticmethod
    def _partial_order_snapshot(order: Any) -> tuple[bool, float]:
        status = str(getattr(order, "status", None) or "").casefold()
        is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
            "canceled",
            "cancelled",
            "closed",
            "expired",
            "rejected",
        }
        return is_open, float(getattr(order, "safe_filled", 0.0) or 0.0)

    def _credit_partial_order(
        self,
        trade: Any,
        state: dict[str, Any],
        order: Any,
        current_time: Any,
    ) -> None:
        is_open, filled = self._partial_order_snapshot(order)
        order_id = str(getattr(order, "order_id", None) or "")
        credit_key = order_id or "|".join(
            str(value or "")
            for value in (
                getattr(order, "ft_order_tag", None),
                getattr(order, "order_date_utc", None)
                or getattr(order, "order_date", None),
                getattr(order, "safe_amount", None),
                getattr(order, "safe_price", None),
            )
        )
        fill_price = float(getattr(order, "safe_price", 0.0) or 0.0)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        credited_for_order = filled * fill_price / leverage if fill_price > 0.0 else 0.0
        credits = state.setdefault("partial_order_credits", {})
        previous_credit = float(credits.get(credit_key) or 0.0)
        if credited_for_order > previous_credit:
            state["partial_realized_stake"] = (
                float(state.get("partial_realized_stake") or 0.0)
                + credited_for_order
                - previous_credit
            )
            credits[credit_key] = credited_for_order
            processed = state.setdefault("partial_fill_order_ids", [])
            if order_id and order_id not in processed:
                processed.append(order_id)

        target_stake = _finite_float(state.get("partial_target_stake"))
        realized = float(state.get("partial_realized_stake") or 0.0)
        target_filled = target_stake is not None and realized >= target_stake - max(
            1e-8, target_stake * 1e-9
        )
        if target_filled:
            state["partial_status"] = "fully_filled"
            state["partial_filled"] = True
            filled_at = getattr(order, "order_filled_utc", None) or current_time
            timestamp = self._utc(filled_at)
            state["partial_filled_at"] = (
                timestamp.isoformat() if timestamp is not None else None
            )
            state["partial_active_order_id"] = None
            state["partial_requested_stake"] = None
            state["phase"] = "REMAINDER"
            return

        if is_open:
            state["partial_status"] = "partially_filled" if realized > 0.0 else "requested"
            state["partial_active_order_id"] = order_id or None
            state["phase"] = "REALIZATION_PENDING"
            return

        if order_id:
            resolved = state.setdefault("partial_resolved_order_ids", [])
            if order_id not in resolved:
                resolved.append(order_id)
        state["partial_status"] = "partially_filled" if realized > 0.0 else "ready"
        state["partial_active_order_id"] = None
        state["partial_requested_at"] = None
        state["partial_requested_stake"] = None

    def _reconcile_partial_orders(
        self, trade: Any, state: dict[str, Any], current_time: Any
    ) -> None:
        exit_side = getattr(trade, "exit_side", None)
        orders = tuple(getattr(trade, "orders", ()) or ())
        matching = [
            order
            for order in orders
            if getattr(order, "ft_order_side", None) == exit_side
            and str(getattr(order, "ft_order_tag", None) or "") in PARTIAL_TAGS
        ]
        for order in matching:
            self._credit_partial_order(trade, state, order, current_time)
        if state.get("partial_status") == "fully_filled":
            return

        resolved = {str(value) for value in state.setdefault("partial_resolved_order_ids", [])}
        unresolved = [
            order
            for order in matching
            if not getattr(order, "order_id", None)
            or str(getattr(order, "order_id", None)) not in resolved
        ]
        open_order = next(
            (order for order in reversed(unresolved) if self._partial_order_snapshot(order)[0]),
            None,
        )
        if open_order is not None:
            self._credit_partial_order(trade, state, open_order, current_time)
            return

        if state.get("partial_status") == "requested":
            requested_at = self._utc(state.get("partial_requested_at"))
            now = self._utc(current_time)
            if requested_at is not None and now is not None and requested_at == now:
                return
            realized = float(state.get("partial_realized_stake") or 0.0)
            state["partial_status"] = "partially_filled" if realized > 0.0 else "ready"
            state["partial_active_order_id"] = None
            state["partial_requested_at"] = None
            state["partial_requested_stake"] = None

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None)
            and self._state(trade) is None
        ):
            self._save_state(trade, self._new_state(pair, trade, order, current_time))
            return None

        tag = str(getattr(order, "ft_order_tag", None) or "")
        if (
            getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and tag in PARTIAL_TAGS
        ):
            state = self._state(trade)
            if state is not None and not state.get("partial_filled"):
                state["partial_tag"] = tag
                self._credit_partial_order(trade, state, order, current_time)
                self._save_state(trade, state)
        return None

    @staticmethod
    def _last_n(condition: Series, count: int) -> bool:
        needed = max(1, int(count))
        return len(condition) >= needed and bool(condition.tail(needed).fillna(False).all())

    @classmethod
    def _confirmation(
        cls,
        frame: DataFrame,
        target: float | None,
        confirmation: str,
        touched_at: Any,
        band: float,
        entry_rate: float,
    ) -> tuple[str | None, bool]:
        prior_touch = cls._utc(touched_at)
        prior_touch_iso = prior_touch.isoformat() if prior_touch is not None else None
        if target is None or frame.empty or entry_rate <= 0.0 or "date" not in frame.columns:
            return prior_touch_iso, False

        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        touch_threshold = max(
            target * (1.0 - band),
            entry_rate * (1.0 + MIN_TARGET_MOVE),
        )
        if prior_touch is None:
            touched_rows = _num(frame, "high").ge(touch_threshold) & dates.notna()
            positions = np.flatnonzero(touched_rows.to_numpy())
            if len(positions) == 0:
                return None, False
            prior_touch = cls._utc(dates.iloc[int(positions[0])])
            prior_touch_iso = prior_touch.isoformat() if prior_touch is not None else None

        post_touch = frame.loc[dates.ge(prior_touch)]
        if post_touch.empty:
            return prior_touch_iso, False
        if confirmation == "touch":
            return prior_touch_iso, True
        opposite = _num(post_touch, "close").lt(_num(post_touch, "open"))
        if confirmation == "reversal1":
            return prior_touch_iso, bool(opposite.tail(1).all())
        if confirmation == "reversal2of3":
            return prior_touch_iso, len(opposite) >= 3 and int(opposite.tail(3).sum()) >= 2
        return prior_touch_iso, False

    @staticmethod
    def _invalidation_event(
        mode: str,
        level_1: bool,
        level_2: bool,
        context_2: bool,
        combined_1: bool,
    ) -> bool:
        return {
            "none": False,
            "level1": level_1,
            "level2": level_2,
            "context2": context_2,
            "combined1": combined_1,
        }.get(mode, False)

    def _events(
        self,
        trade: Any,
        state: dict[str, Any],
        frame: DataFrame,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        plan = EXIT_PLANS[state["plan"]]
        targets = state.get("targets", {})
        target_frame = frame
        entry_filled_at = self._utc(state.get("entry_filled_at"))
        if entry_filled_at is not None and "date" in frame.columns:
            target_frame = frame.loc[
                pd.to_datetime(frame["date"], utc=True, errors="coerce").ge(entry_filled_at)
            ]
        entry_rate = float(state.get("entry_rate") or 0.0)
        touched_1_at, confirmed_1 = self._confirmation(
            target_frame,
            _finite_float(targets.get(plan.target_1)) if plan.target_1 else None,
            plan.confirmation,
            state.get("target_1_touched_at"),
            plan.target_band,
            entry_rate,
        )
        if plan.target_action == "partial" and plan.confirmation.startswith("reversal"):
            if confirmed_1:
                state["target_1_confirmation_latched"] = True
            confirmed_1 = bool(state.get("target_1_confirmation_latched"))
        target_2_frame = target_frame
        partial_filled_at = self._utc(state.get("partial_filled_at"))
        if partial_filled_at is not None and "date" in target_frame.columns:
            target_2_frame = target_frame.loc[
                pd.to_datetime(target_frame["date"], utc=True, errors="coerce").ge(
                    partial_filled_at
                )
            ]
        if state.get("partial_filled"):
            touched_2_at, confirmed_2 = self._confirmation(
                target_2_frame,
                _finite_float(targets.get(plan.target_2)) if plan.target_2 else None,
                "touch",
                state.get("target_2_touched_at"),
                plan.target_band,
                entry_rate,
            )
        else:
            touched_2_at, confirmed_2 = None, False
        state["target_1_touched_at"] = touched_1_at
        state["target_2_touched_at"] = touched_2_at
        if touched_1_at is not None and state.get("phase") not in {
            "REALIZATION_PENDING",
            "REMAINDER",
        }:
            state["phase"] = "TARGET_ZONE"

        close = _num(frame, "close")
        broken = _finite_float(state.get("broken_resistance"))
        support = _finite_float(state.get("structural_support"))
        if broken is None or frame.empty:
            level_condition = pd.Series(False, index=frame.index, dtype="bool")
        else:
            level_condition = close.lt(broken * (1.0 - LEVEL_BAND))
        context_condition = _num(frame, "vp_context_score_bear").gt(
            _num(frame, "vp_context_score_bull")
        ) & _num(frame, "vp_market_context").lt(0.0)
        level_1 = self._last_n(level_condition, 1)
        level_2 = self._last_n(level_condition, 2)
        context_1 = self._last_n(context_condition, 1)
        context_2 = self._last_n(context_condition, 2)
        combined_1 = level_1 and context_1
        support_failure = bool(
            support is not None
            and not frame.empty
            and float(close.iloc[-1]) < support * (1.0 - LEVEL_BAND)
        )

        age_candles = len(target_frame)
        progress_high = _finite_float(_num(target_frame, "high").max())
        favorable = (
            progress_high / entry_rate - 1.0
            if progress_high is not None and entry_rate > 0.0
            else 0.0
        )
        time_failure = bool(
            plan.progress_candles and age_candles >= plan.progress_candles and favorable < 0.01
        )
        max_hold = bool(
            plan.max_hold_candles and age_candles >= plan.max_hold_candles
        )
        return {
            "hard_invalidation": (
                False if plan.role == "baseline" else support_failure or (level_2 and context_1)
            ),
            "invalidation": self._invalidation_event(
                plan.invalidation, level_1, level_2, context_2, combined_1
            ),
            "target_1": confirmed_1,
            "target_2": confirmed_2,
            "time_failure": time_failure,
            "max_hold": max_hold,
            "profit_bucket": (
                "loss"
                if current_profit < -0.005
                else "flat"
                if current_profit < 0.005
                else "profit"
                if current_profit < 0.02
                else "strong_profit"
            ),
        }, state

    @staticmethod
    def evaluate_policy(
        plan: ExitPlan,
        state: dict[str, Any],
        events: dict[str, Any],
    ) -> ExitDecision:
        if state.get("partial_pending"):
            return ExitDecision("hold")
        if state.get("terminal_exit_pending"):
            return ExitDecision(
                "full",
                str(state.get("terminal_exit_tag") or "s3v2_hard_invalidation"),
            )
        if events.get("hard_invalidation"):
            return ExitDecision("full", "s3v2_hard_invalidation")
        if plan.invalidation_action == "full" and events.get("invalidation"):
            return ExitDecision("full", "s3v2_plan_invalidation")
        if events.get("time_failure"):
            return ExitDecision("full", "s3v2_progress_failure")
        if bool(state.get("partial_filled")) and events.get("target_2"):
            return ExitDecision("full", "s3v2_second_target")
        if plan.target_action == "full" and events.get("target_1"):
            return ExitDecision("full", "s3v2_target_full")
        if (
            plan.target_action == "partial"
            and events.get("target_1")
            and not state.get("partial_filled")
        ):
            if events.get("profit_bucket") == "loss":
                return ExitDecision("full", "s3v2_target_reversal_loss")
            return ExitDecision(
                "partial",
                "s3v2_partial_target",
                plan.partial_fraction,
            )
        if (
            plan.invalidation_action == "partial"
            and events.get("invalidation")
            and not state.get("partial_filled")
        ):
            if events.get("profit_bucket") == "loss":
                return ExitDecision("full", "s3v2_invalidation_loss")
            return ExitDecision(
                "partial",
                "s3v2_partial_invalidation",
                plan.partial_fraction,
            )
        if plan.target_action == "tighten" and events.get("target_1"):
            return ExitDecision("tighten", tighten="target")
        if plan.invalidation_action == "tighten" and events.get("invalidation"):
            return ExitDecision("tighten", tighten="invalidation")
        if events.get("max_hold"):
            return ExitDecision("full", "s3v2_max_hold")
        return ExitDecision("hold")

    def _decision_context(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> tuple[ExitPlan, dict[str, Any], dict[str, Any], ExitDecision] | None:
        state = self._state(trade)
        if state is None or state.get("plan") not in EXIT_PLANS:
            return None
        previous = dict(state)
        self._reconcile_partial_orders(trade, state, current_time)
        frame = self._analyzed_frame(pair, current_time)
        partial_pending = self._has_pending_partial(trade, state)
        events, state = self._events(
            trade, state, frame, current_time, current_rate, current_profit
        )
        plan = EXIT_PLANS[state["plan"]]
        state["partial_pending"] = partial_pending
        if partial_pending:
            state["phase"] = "REALIZATION_PENDING"
            evaluable_state = dict(state)
            evaluable_state["partial_pending"] = False
            deferred = self.evaluate_policy(plan, evaluable_state, events)
            if deferred.action == "full":
                state["terminal_exit_pending"] = True
                state["terminal_exit_tag"] = (
                    deferred.tag or "s3v2_deferred_full_exit"
                )
        if state != previous:
            self._save_state(trade, state)
        return plan, state, events, self.evaluate_policy(plan, state, events)

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = kwargs
        context = self._decision_context(pair, trade, current_time, current_rate, current_profit)
        if context is None:
            return None
        plan, state, _, decision = context
        if state.get("partial_pending"):
            return None
        if decision.action == "full":
            return decision.tag
        desired_floor = self._desired_stop_price(plan, state, decision, current_rate)
        return "s3v2_stop_floor_breached" if desired_floor >= current_rate else None

    def custom_roi(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float | None:
        _ = pair, current_time, trade_duration, entry_tag, side, kwargs
        state = self._state(trade)
        plan_name = (state or {}).get("plan")
        if plan_name not in EXIT_PLANS:
            plan_name = str(self.exit_policy_plan.value)
        plan = EXIT_PLANS[plan_name]
        return plan.fixed_tp if plan.fixed_tp > 0.0 else None

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        _ = (
            current_profit,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            kwargs,
        )
        context = self._decision_context(
            str(getattr(trade, "pair", "")),
            trade,
            current_time,
            current_rate,
            current_profit,
        )
        if context is None:
            return None
        _, state, _, decision = context
        if decision.action != "partial" or decision.fraction <= 0.0:
            return None
        if state.get("partial_status") not in {"ready", "partially_filled"}:
            return None
        tag = str(decision.tag or "s3v2_partial")
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        target_stake = _finite_float(state.get("partial_target_stake"))
        realized = float(state.get("partial_realized_stake") or 0.0)
        new_target = target_stake is None
        if new_target:
            target_stake = stake * decision.fraction
        remaining = max(0.0, target_stake - realized)
        request = min(remaining, stake)
        if request <= 0.0 or request >= stake:
            return None
        if min_stake is not None:
            minimum = float(min_stake)
            if request < minimum:
                return None
            if 0.0 < stake - request < minimum:
                request = stake - minimum
            if request < minimum or request <= 0.0 or request >= stake:
                return None
        if new_target:
            target_stake = request
            state["partial_target_stake"] = target_stake
        state["partial_tag"] = tag
        state["partial_status"] = "requested"
        requested_at = self._utc(current_time)
        state["partial_requested_at"] = (
            requested_at.isoformat() if requested_at is not None else None
        )
        state["partial_requested_stake"] = request
        state["partial_active_order_id"] = None
        state["phase"] = "REALIZATION_PENDING"
        self._save_state(trade, state)
        return -request, tag

    @staticmethod
    def _desired_stop_price(
        plan: ExitPlan,
        state: dict[str, Any],
        decision: ExitDecision,
        current_rate: float,
    ) -> float:
        entry_rate = float(state.get("entry_rate") or current_rate)
        stop_price = entry_rate * (1.0 - plan.hard_stop)
        support = _finite_float(state.get("structural_support"))
        if plan.role != "baseline" and support is not None and support < entry_rate:
            stop_price = max(stop_price, support * (1.0 - LEVEL_BAND))
        if state.get("partial_filled"):
            if plan.remainder == "breakeven":
                stop_price = max(stop_price, entry_rate)
            elif plan.remainder == "trail":
                stop_price = max(stop_price, current_rate * 0.985)
        if decision.action == "tighten" and decision.tighten == "target":
            target = _finite_float(state.get("targets", {}).get(plan.target_1))
            if target is not None:
                stop_price = max(stop_price, target * 0.99)
        if decision.action == "tighten" and decision.tighten == "invalidation":
            broken = _finite_float(state.get("broken_resistance"))
            if broken is not None:
                stop_price = max(stop_price, broken * (1.0 - LEVEL_BAND))
            if current_rate > entry_rate:
                stop_price = max(stop_price, entry_rate * 1.001)
        return stop_price

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
        _ = after_fill, kwargs
        state = self._state(trade)
        plan_name = (state or {}).get("plan")
        if plan_name not in EXIT_PLANS:
            plan_name = str(self.exit_policy_plan.value)
        selected_plan = EXIT_PLANS[plan_name]
        decision = ExitDecision("hold")
        context = self._decision_context(pair, trade, current_time, current_rate, current_profit)
        if context is not None:
            plan, state, _, decision = context
            if state.get("partial_pending") or decision.action == "full":
                return None
            selected_plan = plan
        working_state = state or {
            "entry_rate": float(getattr(trade, "open_rate", current_rate) or current_rate)
        }
        stop_price = self._desired_stop_price(
            selected_plan,
            working_state,
            decision,
            current_rate,
        )

        persisted_floor = _finite_float(working_state.get("stop_floor"))
        if persisted_floor is not None:
            stop_price = max(stop_price, persisted_floor)
        if stop_price >= current_rate:
            return None
        if state is not None and (persisted_floor is None or stop_price > persisted_floor):
            state["stop_floor"] = stop_price
            self._save_state(trade, state)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=False,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
