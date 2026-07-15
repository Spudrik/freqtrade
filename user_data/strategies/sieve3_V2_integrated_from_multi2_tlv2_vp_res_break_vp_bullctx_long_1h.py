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
    BooleanParameter,
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.complex_trendline_projection_v2 import (
    add_trendline_projection_v2,
)
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_TAG = "multi2_tlv2_vp_res_break_vp_bullctx_long_1h"
STATE_KEY = "sieve3_v2_tlv2_vp_long"
STATE_VERSION = 2
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


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))


def _finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


class ExitPlan:
    __slots__ = (
        "confirmation",
        "fixed_tp",
        "hard_stop",
        "invalidation",
        "invalidation_action",
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
        fixed_tp: float = 0.0,
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
        self.fixed_tp = float(fixed_tp)

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
        target_band=0.005,
    ),
    "hvn_rev1_full": ExitPlan("target_full", "hvn", confirmation="reversal1", target_action="full"),
    "vah_rev1_full": ExitPlan(
        "target_full",
        "vah",
        confirmation="reversal1",
        target_action="full",
        target_band=0.015,
    ),
    "prior48_rev1_full": ExitPlan(
        "target_full",
        "prior48",
        confirmation="reversal1",
        target_action="full",
        target_band=0.03,
    ),
    "prior96_rev2of3_full": ExitPlan(
        "target_full",
        "prior96",
        confirmation="reversal2of3",
        target_action="full",
        target_band=0.03,
    ),
    "measured_half_touch_full": ExitPlan(
        "target_full",
        "measured_half",
        confirmation="touch",
        target_action="full",
        target_band=0.005,
    ),
    "measured_full_rev1_full": ExitPlan(
        "target_full", "measured_full", confirmation="reversal1", target_action="full"
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


class Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False

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

    use_sieve2_vp_guard = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    sieve2_vp_guard_mode = CategoricalParameter(
        list(SIEVE2_VP_GUARD_MODES),
        default="score_or_context",
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_window = CategoricalParameter(
        [48, 96, 168], default=96, space="buy", optimize=False, load=False
    )
    sieve2_vp_bins = CategoricalParameter(
        [24, 36, 48], default=36, space="buy", optimize=False, load=False
    )
    sieve2_vp_score_min = CategoricalParameter(
        [0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=False, load=False
    )
    sieve2_vp_context_min = CategoricalParameter(
        [0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=False, load=False
    )
    use_sieve2_market_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    sieve2_market_guard_mode = CategoricalParameter(
        list(SIEVE2_MARKET_GUARD_MODES),
        default="pressure_or_trend",
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_window = CategoricalParameter(
        [12, 24, 48, 96], default=24, space="buy", optimize=False, load=False
    )
    sieve2_market_pressure_min = CategoricalParameter(
        [0.03, 0.07, 0.12, 0.18, 0.25], default=0.07, space="buy", optimize=False, load=False
    )
    sieve2_market_trend_min = CategoricalParameter(
        [0.0, 0.25, 0.50, 0.80], default=0.25, space="buy", optimize=False, load=False
    )
    sieve2_rs_benchmark_pair = CategoricalParameter(
        ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        default="BTC/USDT:USDT",
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_rs_score_min = CategoricalParameter(
        [0.35, 0.45, 0.55, 0.65], default=0.45, space="buy", optimize=False, load=False
    )

    use_volume_guard = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    volume_guard_window = CategoricalParameter(
        [12, 24, 48], default=24, space="buy", optimize=False, load=False
    )
    volume_ratio_min = CategoricalParameter(
        [0.8, 1.0, 1.3, 1.6], default=1.6, space="buy", optimize=False, load=False
    )
    use_pressure_guard = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    pressure_window = CategoricalParameter(
        [12, 24, 48], default=24, space="buy", optimize=False, load=False
    )
    pressure_min = CategoricalParameter(
        [0.05, 0.1, 0.15, 0.2, 0.35], default=0.15, space="buy", optimize=False, load=False
    )
    use_accumulation_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    use_body_direction_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    use_close_direction_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )

    vp_window = CategoricalParameter(
        [48, 96, 144], default=96, space="buy", optimize=False, load=False
    )
    vp_bins = CategoricalParameter(
        [36, 48, 72], default=48, space="buy", optimize=False, load=False
    )
    vp_value_area_pct = CategoricalParameter(
        [0.65, 0.70, 0.75], default=0.70, space="buy", optimize=False, load=False
    )
    vp_price_source = CategoricalParameter(
        ["hlc3", "ohlc4"], default="hlc3", space="buy", optimize=False, load=False
    )
    vp_smooth_bins = CategoricalParameter(
        [2, 3, 4], default=3, space="buy", optimize=False, load=False
    )
    vp_hvn_threshold = CategoricalParameter(
        [0.65, 0.70, 0.78], default=0.70, space="buy", optimize=False, load=False
    )
    vp_lvn_threshold = CategoricalParameter(
        [0.25, 0.35, 0.45], default=0.35, space="buy", optimize=False, load=False
    )
    vp_pressure_delta_min = CategoricalParameter(
        [0.0, 0.05, 0.1], default=0.05, space="buy", optimize=False, load=False
    )
    vp_node_near_pct = CategoricalParameter(
        [0.006, 0.010, 0.016], default=0.010, space="buy", optimize=False, load=False
    )
    vp_volume_percentile_min = CategoricalParameter(
        [0.45, 0.55, 0.65], default=0.55, space="buy", optimize=False, load=False
    )
    vp_score_window = CategoricalParameter(
        [24, 48, 72], default=48, space="buy", optimize=False, load=False
    )
    vp_fast_traverse_atr_mult = CategoricalParameter(
        [0.9, 1.2, 1.6], default=1.2, space="buy", optimize=False, load=False
    )
    vp_entry_score_margin = CategoricalParameter(
        [0.0, 0.02, 0.05], default=0.02, space="buy", optimize=False, load=False
    )
    vp_score_min = CategoricalParameter(
        [0.15, 0.25, 0.4], default=0.4, space="buy", optimize=False, load=False
    )
    vp_context_min = CategoricalParameter(
        [0.2, 0.28, 0.45], default=0.28, space="buy", optimize=False, load=False
    )
    vp_level_buffer_pct = CategoricalParameter(
        [0.003, 0.006, 0.010, 0.016], default=0.016, space="buy", optimize=False, load=False
    )

    pivot_strength = CategoricalParameter(
        [2, 3, 4], default=2, space="buy", optimize=False, load=False
    )
    min_line_score = CategoricalParameter(
        [0.4, 0.5, 0.6], default=0.6, space="buy", optimize=False, load=False
    )
    min_active_bars = CategoricalParameter(
        [4, 8, 16], default=8, space="buy", optimize=False, load=False
    )
    max_distance_atr = CategoricalParameter(
        [3.0, 6.0, 10.0], default=3.0, space="buy", optimize=False, load=False
    )
    proximity_rank_weight = CategoricalParameter(
        [0.0, 0.05, 0.1], default=0.05, space="buy", optimize=False, load=False
    )
    line_buffer_pct = CategoricalParameter(
        [0.0, 0.003, 0.006, 0.012], default=0.0, space="buy", optimize=False, load=False
    )
    line_slope_min_pct = CategoricalParameter(
        [0.0, 0.0005, 0.0015], default=0.0005, space="buy", optimize=False, load=False
    )

    exit_policy_plan = CategoricalParameter(
        list(EXIT_PLANS),
        default="baseline_3_3",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_policy_plan.batch_tags = "family:exits", "mode:sieve3_v2_integrated"

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
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)

        volume_window = int(self.volume_guard_window.value)
        volume_baseline = (
            volume.shift(1)
            .rolling(volume_window, min_periods=max(2, volume_window // 3))
            .mean()
            .replace(0.0, np.nan)
        )
        guard &= volume.ge(volume_baseline.mul(float(self.volume_ratio_min.value)))

        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume.fillna(0.0)).fillna(0.0)
        pressure_window = int(self.pressure_window.value)
        pressure_baseline = (
            volume.rolling(pressure_window, min_periods=max(2, pressure_window // 3))
            .sum()
            .replace(0.0, np.nan)
        )
        pressure_ratio = (
            directional_volume.rolling(
                pressure_window, min_periods=max(2, pressure_window // 3)
            ).sum()
            / pressure_baseline
        )
        guard &= pressure_ratio.ge(float(self.pressure_min.value))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(
            dataframe,
            window=int(self.vp_window.value),
            bins=int(self.vp_bins.value),
            value_area_pct=float(self.vp_value_area_pct.value),
            price_source=str(self.vp_price_source.value),
            smooth_bins=int(self.vp_smooth_bins.value),
            hvn_threshold=float(self.vp_hvn_threshold.value),
            lvn_threshold=float(self.vp_lvn_threshold.value),
            pressure_delta_min=float(self.vp_pressure_delta_min.value),
            node_near_pct=float(self.vp_node_near_pct.value),
            volume_percentile_min=float(self.vp_volume_percentile_min.value),
            score_window=int(self.vp_score_window.value),
            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult.value),
            entry_score_margin=float(self.vp_entry_score_margin.value),
            prefix="vp",
        )

    @staticmethod
    def _add_market_state(frame: DataFrame, window: int) -> DataFrame:
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        baseline = (
            volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        )
        returns = close.pct_change(fill_method=None)
        window_return = close.pct_change(window, fill_method=None)
        volatility = returns.rolling(window, min_periods=max(3, window // 3)).std() * np.sqrt(
            float(window)
        )
        frame["s2m_pressure_ratio"] = (
            directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline
        )
        frame["s2m_trend_z"] = window_return / volatility.replace(0.0, np.nan)
        frame["s2m_close_location"] = close_location
        return frame

    def _add_sieve2_guard_indicators(self, dataframe: DataFrame) -> DataFrame:
        frame = self._add_market_state(dataframe.copy(), int(self.sieve2_market_window.value))
        return add_volume_profile(
            frame,
            window=int(self.sieve2_vp_window.value),
            bins=int(self.sieve2_vp_bins.value),
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
        prior_vah = _num(frame, "s2vp_prior_vah", np.nan)
        prior_val = _num(frame, "s2vp_prior_val", np.nan)
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
        if bool(self.use_sieve2_vp_guard.value):
            guarded &= self._sieve2_vp_guard(
                dataframe,
                str(self.sieve2_vp_guard_mode.value),
                float(self.sieve2_vp_score_min.value),
                float(self.sieve2_vp_context_min.value),
            )
        if bool(self.use_sieve2_market_guard.value):
            guarded &= self._sieve2_market_guard(
                dataframe,
                str(self.sieve2_market_guard_mode.value),
                float(self.sieve2_market_pressure_min.value),
                float(self.sieve2_market_trend_min.value),
            )
        return guarded.fillna(False)

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        score_long = _num(dataframe, "vp_score_long")
        bull_context = _num(dataframe, "vp_context_score_bull")
        bear_context = _num(dataframe, "vp_context_score_bear")
        market = _num(dataframe, "vp_market_context")
        return (
            score_long.ge(float(self.vp_score_min.value))
            & bull_context.ge(float(self.vp_context_min.value))
            & bull_context.ge(bear_context)
            & market.ge(0.0)
        )

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            pivot_strength=int(self.pivot_strength.value),
            raw_line_output_count=1,
            min_output_line_score=float(self.min_line_score.value),
            min_output_active_bars=int(self.min_active_bars.value),
            max_active_line_distance_atr_mult=float(self.max_distance_atr.value),
            proximity_rank_weight=float(self.proximity_rank_weight.value),
            output_prefix="tlv2",
        )

    def _tlv2_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        line = _num(dataframe, "tlv2_resistance_line_rank0", np.nan)
        score = _num(dataframe, "tlv2_resistance_score_rank0")
        distance = _num(dataframe, "tlv2_resistance_distance_atr_rank0", np.nan)
        active = score.ge(float(self.min_line_score.value)) & distance.le(
            float(self.max_distance_atr.value)
        )
        return active & _cross_above(close, line.mul(1.0 + float(self.line_buffer_pct.value)))

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
        broken = _finite_float(row.get("tlv2_resistance_line_rank0")) if row is not None else None
        support = _finite_float(row.get("s3v2_prior_low_48")) if row is not None else None
        if support is not None and support >= entry_rate:
            support = None

        raw: dict[str, float | None] = {
            "hvn": self._directional_target(row.get("vp_hvn_above"), entry_rate)
            if row is not None
            else None,
            "vah": self._directional_target(row.get("vp_vah"), entry_rate)
            if row is not None
            else None,
            "prior48": self._directional_target(row.get("s3v2_prior_high_48"), entry_rate)
            if row is not None
            else None,
            "prior96": self._directional_target(row.get("s3v2_prior_high_96"), entry_rate)
            if row is not None
            else None,
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
        plan = EXIT_PLANS[str(self.exit_policy_plan.value)]
        if plan.target_1 and plan.target_2:
            first = _finite_float(targets.get(plan.target_1))
            second = _finite_float(targets.get(plan.target_2))
            if first is not None and (second is None or second <= first * (1.0 + MIN_TARGET_MOVE)):
                targets[plan.target_2] = None
                providers[plan.target_2] = None
        candle_date = None
        if row is not None and "date" in row.index:
            timestamp = self._utc(row["date"])
            candle_date = timestamp.isoformat() if timestamp is not None else None
        return {
            "version": STATE_VERSION,
            "plan": str(self.exit_policy_plan.value),
            "phase": "ENTRY",
            "entry_rate": entry_rate,
            "entry_candle": candle_date,
            "broken_resistance": broken,
            "structural_support": support,
            "targets": targets,
            "target_providers": providers,
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "partial_filled": False,
            "partial_filled_at": None,
            "partial_tag": None,
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
    def _has_pending_partial(trade: Any) -> bool:
        exit_side = getattr(trade, "exit_side", None)
        return any(
            getattr(order, "ft_order_side", None) == exit_side
            and str(getattr(order, "ft_order_tag", None) or "") in PARTIAL_TAGS
            for order in trade.open_orders
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
            and float(getattr(order, "safe_filled", 0.0) or 0.0) > 0.0
        ):
            state = self._state(trade)
            if state is not None and not state.get("partial_filled"):
                filled_at = getattr(order, "order_filled_utc", None) or current_time
                state["partial_filled"] = True
                state["partial_tag"] = tag
                state["partial_filled_at"] = self._utc(filled_at).isoformat()
                state["phase"] = "REMAINDER"
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
        entry_candle = self._utc(state.get("entry_candle"))
        if entry_candle is not None and "date" in frame.columns:
            target_frame = frame.loc[
                pd.to_datetime(frame["date"], utc=True, errors="coerce").gt(entry_candle)
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

        age_candles = 0
        opened = self._utc(getattr(trade, "open_date_utc", None))
        now = self._utc(current_time)
        if opened is not None and now is not None:
            age_minutes = max(0.0, (now - opened).total_seconds() / 60.0)
            age_candles = int(age_minutes // timeframe_to_minutes(self.timeframe))
        progress_high = _finite_float(_num(target_frame, "high").max())
        favorable = (
            progress_high / entry_rate - 1.0
            if progress_high is not None and entry_rate > 0.0
            else 0.0
        )
        time_failure = bool(
            plan.progress_candles and age_candles >= plan.progress_candles and favorable < 0.01
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
        }, state

    @staticmethod
    def evaluate_policy(
        plan: ExitPlan,
        state: dict[str, Any],
        events: dict[str, bool],
    ) -> ExitDecision:
        if events.get("hard_invalidation"):
            return ExitDecision("full", "s3v2_hard_invalidation")
        if state.get("partial_pending"):
            return ExitDecision("hold")
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
            return ExitDecision(
                "partial",
                "s3v2_partial_invalidation",
                plan.partial_fraction,
            )
        if plan.target_action == "tighten" and events.get("target_1"):
            return ExitDecision("tighten", tighten="target")
        if plan.invalidation_action == "tighten" and events.get("invalidation"):
            return ExitDecision("tighten", tighten="invalidation")
        return ExitDecision("hold")

    def _decision_context(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> tuple[ExitPlan, dict[str, Any], dict[str, bool], ExitDecision] | None:
        state = self._state(trade)
        if state is None or state.get("plan") not in EXIT_PLANS:
            return None
        previous = dict(state)
        frame = self._analyzed_frame(pair, current_time)
        partial_pending = self._has_pending_partial(trade)
        events, state = self._events(
            trade, state, frame, current_time, current_rate, current_profit
        )
        if state != previous:
            self._save_state(trade, state)
        state["partial_pending"] = partial_pending
        if partial_pending:
            state["phase"] = "REALIZATION_PENDING"
        plan = EXIT_PLANS[state["plan"]]
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
        decision = context[3]
        return decision.tag if decision.action == "full" else None

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
        return plan.fixed_tp if plan.role == "baseline" and plan.fixed_tp > 0.0 else None

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
            min_stake,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            kwargs,
        )
        if self._has_open_order(trade):
            return None
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
        tag = str(decision.tag or "s3v2_partial")
        state["partial_tag"] = tag
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        return -stake * decision.fraction, tag

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
        entry_rate = float(
            (state or {}).get("entry_rate")
            or getattr(trade, "open_rate", current_rate)
            or current_rate
        )
        plan_name = (state or {}).get("plan")
        if plan_name not in EXIT_PLANS:
            plan_name = str(self.exit_policy_plan.value)
        selected_plan = EXIT_PLANS[plan_name]
        stop_price = entry_rate * (1.0 - selected_plan.hard_stop)
        support = _finite_float((state or {}).get("structural_support"))
        if selected_plan.role != "baseline" and support is not None and support < entry_rate:
            stop_price = max(stop_price, support * (1.0 - LEVEL_BAND))

        context = self._decision_context(pair, trade, current_time, current_rate, current_profit)
        if context is not None:
            plan, state, _, decision = context
            if state.get("partial_pending") or decision.action == "full":
                return None
            if state.get("partial_filled"):
                if plan.remainder == "breakeven":
                    stop_price = max(stop_price, entry_rate * 1.001)
                elif plan.remainder == "trail":
                    stop_price = max(stop_price, current_rate * 0.985)
            if decision.action == "tighten":
                if decision.tighten == "target":
                    target = _finite_float(state.get("targets", {}).get(plan.target_1))
                    if target is not None:
                        stop_price = max(stop_price, target * 0.99)
                if decision.tighten == "invalidation":
                    broken = _finite_float(state.get("broken_resistance"))
                    if broken is not None:
                        stop_price = max(stop_price, broken * (1.0 - LEVEL_BAND))
                    if current_rate > entry_rate:
                        stop_price = max(stop_price, entry_rate * 1.001)

        persisted_floor = _finite_float((state or {}).get("stop_floor"))
        if persisted_floor is not None:
            stop_price = max(stop_price, persisted_floor)
        if state is not None and (persisted_floor is None or stop_price > persisted_floor):
            state["stop_floor"] = stop_price
            self._save_state(trade, state)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=False,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
