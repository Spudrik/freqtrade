"""Sieve3 V2 integrated exits for the promoted prior-month-high breakout long.

Preflight contract
------------------
Canonical Sieve2 source:
    sieve2_prior_month_high_breakout_long
Promotion evidence:
    Job 20260521T012740_entry_all_resume, result row 668.
    Result-row SHA256:
    ca228736950864656aaab2a13b0d99e21a47ed741aa2d4f0cc2ce7008fcc4dac
Promoted baseline:
    TP/SL 2%/2%, 418 trades, 53.5885% win rate, 4.047944% return,
    4.07119% maximum drawdown, 1.101507 profit factor.
Archived lineage:
    Backtest archive SHA256:
    047b03671a3da2e8b3db83da731eb21034ddf6aaaa6550ba7e9001e34dee413c
    Sieve2 source snapshot SHA256:
    414c6fed7ab6e143fa04fe3d8b944fc9109a388bfe44237f3dfeadc5e6763af7
    Sieve2 params snapshot SHA256:
    bd41e40fe5234ea113dfe1788cfe443545189b16f7b2dc4c87000b29e5651a7e
    Archived buy-default SHA256:
    6c40a762abfe47263cc9f40829257501fc4172c7f79d804ae5cc51f74c5192fd
    Selected sparse buy-param SHA256:
    13bafcd62854d9e39a1cc4681f9a523f63981ff502f5983c53f0ff2136a68e55
Entry lock:
    The 57-value effective promoted buy lock is embedded below with
    optimize=False and load=False. Effective-lock SHA256:
    8aeb180d6a07431b365ad820e9537f4146c4f9c2ca7c8a7f781ee280ddf813d3
    Baseline entry-signature SHA256:
    27a42b4c7c3d98f4655503059914a481072517b3cf8f43fb633fb46fdfce07f3
    Baseline entry-trend SHA256:
    e6547d8fc120d82c172d59929a63f7c953e8d5e1bd08b1e7e1893991caf0e7e5
Tested controls:
    sieve3_exit_breakeven_from_prior_month_high_breakout_long.py
    sieve3_exit_level_zone_reversal_from_prior_month_high_breakout_long.py
    Jobs 20260710T061808_entry_sieve3_exit_speed_batch_048 and
    20260704T092624_entry_sieve3_exit_speed_batch_042.
Refined pilot controls:
    sieve3_rework_exit_{invalidation,layered,target_zone}_from_prior_month_*.py
    Job 20260711T230600_entry_sieve3_exit_rework_comparison_001.
Consolidation:
    The 36 policies retain fixed controls, pure target and invalidation
    ablations, integrated partial/remainder plans, reduce-then-target plans,
    target tightening, dual-target runners, and coherent follow-through tests.
    Generic indicator exits and duplicate legacy numeric sweeps are discarded.
Target provider hierarchy and semantics:
    Every target is frozen_at_entry from the last candle closed when the entry
    order was placed. Requested providers use explicit fallback chains over
    prior-day/week highs, shifted rolling highs, VP HVN/VAH, and measured
    prior-month range extensions. Wrong-side and trivial targets are rejected.
    Targets never adapt farther away after entry.
Breakout invalidation contract:
    The source boundary is the frozen prior-month high. Named invalidations are
    one or two confirmed closes back through it, or one/two closes back through
    it after a prior closed-candle retest held. A deep boundary loss or loss of
    the frozen prior-month low is the universal non-baseline hard failure.
Policy groups:
    baseline, target_full, invalidation_full, defensive_tighten,
    target_partial, invalidation_reduce, target_tighten, dual_target,
    and progress_failure.
Active sell surface:
    exit_policy_plan only. No buy parameter is loadable or optimized.
Split decision:
    One file is coherent because all plans share the same frozen monthly
    breakout boundary, period/structure/VP obstacle hierarchy, and callback
    lifecycle. A second target ecosystem would only duplicate state handling.
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
    DecimalParameter,
    IntParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.complex_volume_profile import add_volume_profile


SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "sieve2_prior_month_high_breakout_long"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = "sieve3_V2_integrated"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = (
    "Manage a prior-month-high breakout against frozen period, structure, and VP "
    "obstacles while invalidating only on closed-candle boundary failure."
)
ENTRY_TAG = "prior_month_high_breakout_long"
STATE_KEY = "sieve3_v2_prior_month_high_long"
STATE_VERSION = 4
LEVEL_BAND = 0.005
RETEST_BAND = 0.004
DEEP_FAILURE_BAND = 0.025
MIN_TARGET_MOVE = 0.002
FOLLOW_THROUGH_MOVE = 0.01
PARTIAL_TAGS = {"s3v2_partial_target", "s3v2_partial_invalidation"}

RESULT_ROW_SHA256 = "ca228736950864656aaab2a13b0d99e21a47ed741aa2d4f0cc2ce7008fcc4dac"
BACKTEST_ARCHIVE_SHA256 = "047b03671a3da2e8b3db83da731eb21034ddf6aaaa6550ba7e9001e34dee413c"
SNAPSHOT_STRATEGY_SHA256 = "414c6fed7ab6e143fa04fe3d8b944fc9109a388bfe44237f3dfeadc5e6763af7"
SNAPSHOT_PARAMS_SHA256 = "bd41e40fe5234ea113dfe1788cfe443545189b16f7b2dc4c87000b29e5651a7e"
LOCKED_BUY_PARAMS_SHA256 = "8aeb180d6a07431b365ad820e9537f4146c4f9c2ca7c8a7f781ee280ddf813d3"

PERIOD_CHOICES = ("day", "week", "month")
PRICE_SOURCE_CHOICES = ("close", "hl2", "hlc3", "ohlc4")
VP_GUARD_MODES = ("direction", "score", "context", "score_or_context", "balance")
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


def _finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


DEFAULT_FALLBACK_ROI = 0.08
DEFAULT_MAX_HOLD_CANDLES = 720


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
        invalidation: str = "boundary2",
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
    "period_nearest_touch_full": ExitPlan(
        "target_full",
        "period_nearest",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "prior_week_rev1_full": ExitPlan(
        "target_full",
        "prior_week",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.015,
    ),
    "swing48_touch_full": ExitPlan(
        "target_full",
        "swing48",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "swing96_rev2of3_full": ExitPlan(
        "target_full",
        "swing96",
        confirmation="reversal2of3",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.02,
    ),
    "vp_hvn_rev1_full": ExitPlan(
        "target_full",
        "vp_hvn",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "measured_half_rev1_full": ExitPlan(
        "target_full",
        "measured_half",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.015,
    ),
    "boundary1_full": ExitPlan(
        "invalidation_full", invalidation="boundary1", invalidation_action="full"
    ),
    "boundary2_full": ExitPlan(
        "invalidation_full", invalidation="boundary2", invalidation_action="full"
    ),
    "retest1_full": ExitPlan(
        "invalidation_full", invalidation="retest1", invalidation_action="full"
    ),
    "retest2_full": ExitPlan(
        "invalidation_full", invalidation="retest2", invalidation_action="full"
    ),
    "boundary1_tighten": ExitPlan(
        "defensive_tighten",
        invalidation="boundary1",
        invalidation_action="tighten",
    ),
    "retest1_tighten": ExitPlan(
        "defensive_tighten",
        invalidation="retest1",
        invalidation_action="tighten",
    ),
    "period_nearest_touch_p33_be_boundary2": ExitPlan(
        "target_partial",
        "period_nearest",
        confirmation="touch",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "period_nearest_rev1_p50_trail_retest1": ExitPlan(
        "target_partial",
        "period_nearest",
        confirmation="reversal1",
        target_action="partial",
        invalidation="retest1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "prior_week_rev1_p33_be_boundary2": ExitPlan(
        "target_partial",
        "prior_week",
        confirmation="reversal1",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "swing48_touch_p33_trail_boundary2": ExitPlan(
        "target_partial",
        "swing48",
        confirmation="touch",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "swing48_rev1_p50_be_retest1": ExitPlan(
        "target_partial",
        "swing48",
        confirmation="reversal1",
        target_action="partial",
        invalidation="retest1",
        partial_fraction=0.50,
        remainder="breakeven",
    ),
    "swing96_rev2of3_p33_trail_boundary2": ExitPlan(
        "target_partial",
        "swing96",
        confirmation="reversal2of3",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "vp_hvn_touch_p33_be_retest1": ExitPlan(
        "target_partial",
        "vp_hvn",
        confirmation="touch",
        target_action="partial",
        invalidation="retest1",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "vp_hvn_rev1_p50_trail_boundary2": ExitPlan(
        "target_partial",
        "vp_hvn",
        confirmation="reversal1",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "vp_vah_rev1_p33_trail_retest2": ExitPlan(
        "target_partial",
        "vp_vah",
        confirmation="reversal1",
        target_action="partial",
        invalidation="retest2",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "measured_half_touch_p33_be_boundary2": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="touch",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "measured_half_rev1_p50_trail_retest1": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="reversal1",
        target_action="partial",
        invalidation="retest1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "nearest_rev1_p33_trail_boundary2": ExitPlan(
        "target_partial",
        "nearest",
        confirmation="reversal1",
        target_action="partial",
        invalidation="boundary2",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "boundary1_reduce33_period_nearest": ExitPlan(
        "invalidation_reduce",
        "period_nearest",
        confirmation="touch",
        target_action="full",
        invalidation="boundary1",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "retest1_reduce50_vp_hvn": ExitPlan(
        "invalidation_reduce",
        "vp_hvn",
        confirmation="touch",
        target_action="full",
        invalidation="retest1",
        invalidation_action="partial",
        partial_fraction=0.50,
        remainder="target",
    ),
    "boundary2_reduce33_measured_half": ExitPlan(
        "invalidation_reduce",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="boundary2",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "period_nearest_touch_tighten_boundary2": ExitPlan(
        "target_tighten",
        "period_nearest",
        confirmation="touch",
        target_action="tighten",
        invalidation="boundary2",
        invalidation_action="full",
        remainder="tighten",
    ),
    "vp_hvn_touch_tighten_retest1": ExitPlan(
        "target_tighten",
        "vp_hvn",
        confirmation="touch",
        target_action="tighten",
        invalidation="retest1",
        invalidation_action="full",
        remainder="tighten",
    ),
    "measured_half_touch_tighten_boundary2": ExitPlan(
        "target_tighten",
        "measured_half",
        confirmation="touch",
        target_action="tighten",
        invalidation="boundary2",
        invalidation_action="full",
        remainder="tighten",
    ),
    "period_nearest_then_swing96_runner": ExitPlan(
        "dual_target",
        "period_nearest",
        "swing96",
        "touch",
        "partial",
        "boundary2",
        "full",
        0.33,
        "trail",
    ),
    "vp_hvn_then_measured_runner": ExitPlan(
        "dual_target",
        "vp_hvn",
        "measured_full",
        "touch",
        "partial",
        "retest2",
        "full",
        0.50,
        "trail",
    ),
    "progress24_nearest": ExitPlan(
        "progress_failure",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="boundary2",
        invalidation_action="full",
        progress_candles=24,
    ),
    "progress48_measured": ExitPlan(
        "progress_failure",
        "measured_full",
        confirmation="touch",
        target_action="full",
        invalidation="retest2",
        invalidation_action="full",
        progress_candles=48,
    ),
}


TARGET_PROVIDER_CHAINS: dict[str, tuple[str, ...]] = {
    "period_nearest": ("prior_day", "prior_week", "swing48", "vp_hvn", "measured_half"),
    "prior_week": ("prior_week", "swing48", "vp_hvn", "measured_half"),
    "swing48": ("swing48", "swing96", "vp_hvn", "measured_half"),
    "swing96": ("swing96", "swing168", "vp_hvn", "measured_full"),
    "vp_hvn": ("vp_hvn", "vp_vah", "swing48", "measured_half"),
    "vp_vah": ("vp_vah", "vp_hvn", "swing48", "measured_half"),
    "measured_half": ("measured_half", "swing48", "vp_hvn"),
    "measured_full": ("measured_full", "swing96", "vp_hvn"),
    "nearest": (
        "prior_day",
        "prior_week",
        "swing48",
        "swing96",
        "swing168",
        "vp_hvn",
        "vp_vah",
        "measured_half",
        "measured_full",
    ),
}


class Sieve3V2IntegratedFromPriorMonthHighBreakoutLong(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 336
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

    use_sieve2_vp_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
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
        [0.15, 0.25, 0.35, 0.50],
        default=0.25,
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_context_min = CategoricalParameter(
        [0.18, 0.28, 0.38, 0.50],
        default=0.28,
        space="buy",
        optimize=False,
        load=False,
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
        [0.03, 0.07, 0.12, 0.18, 0.25],
        default=0.07,
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_trend_min = CategoricalParameter(
        [0.0, 0.25, 0.50, 0.80],
        default=0.25,
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_rs_benchmark_pair = CategoricalParameter(
        ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        default="BTC/USDT:USDT",
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_rs_score_min = CategoricalParameter(
        [0.35, 0.45, 0.55, 0.65],
        default=0.45,
        space="buy",
        optimize=False,
        load=False,
    )

    breakout_buffer_pct = CategoricalParameter(
        [0.0, 0.003, 0.004, 0.006, 0.012],
        default=0.003,
        space="buy",
        optimize=False,
        load=False,
    )
    reclaim_buffer_pct = CategoricalParameter(
        [0.0, 0.006, 0.012, 0.02],
        default=0.006,
        space="buy",
        optimize=False,
        load=False,
    )
    sweep_buffer_pct = CategoricalParameter(
        [0.0, 0.006, 0.012, 0.02],
        default=0.006,
        space="buy",
        optimize=False,
        load=False,
    )
    zone_near_pct = CategoricalParameter(
        [0.005, 0.01, 0.02],
        default=0.01,
        space="buy",
        optimize=False,
        load=False,
    )
    rolling_level_lookback = CategoricalParameter(
        [48, 72, 120, 168],
        default=72,
        space="buy",
        optimize=False,
        load=False,
    )
    equal_level_lookback = CategoricalParameter(
        [48, 72, 120], default=72, space="buy", optimize=False, load=False
    )
    equal_level_tolerance_pct = CategoricalParameter(
        [0.003, 0.006, 0.012],
        default=0.006,
        space="buy",
        optimize=False,
        load=False,
    )
    equal_level_min_touches = CategoricalParameter(
        [2, 3, 4], default=2, space="buy", optimize=False, load=False
    )
    confluence_period = CategoricalParameter(
        list(PERIOD_CHOICES),
        default="day",
        space="buy",
        optimize=False,
        load=False,
    )
    avwap_anchor_lookback = CategoricalParameter(
        [72, 120, 240], default=120, space="buy", optimize=False, load=False
    )
    avwap_band_mult = CategoricalParameter(
        [0.75, 1.25, 2.0],
        default=1.25,
        space="buy",
        optimize=False,
        load=False,
    )
    zone_impulse_window = IntParameter(
        12, 96, default=36, space="buy", optimize=False, load=False
    )
    zone_impulse_atr_min = DecimalParameter(
        0.20,
        3.00,
        decimals=2,
        default=0.80,
        space="buy",
        optimize=False,
        load=False,
    )
    zone_body_fraction_min = DecimalParameter(
        0.30,
        0.90,
        decimals=2,
        default=0.55,
        space="buy",
        optimize=False,
        load=False,
    )
    zone_volume_ratio_min = DecimalParameter(
        0.00,
        3.00,
        decimals=2,
        default=1.10,
        space="buy",
        optimize=False,
        load=False,
    )
    zone_max_age_bars = CategoricalParameter(
        [24, 72, 168], default=72, space="buy", optimize=False, load=False
    )
    use_volume_guard = BooleanParameter(
        default=True, space="buy", optimize=False, load=False
    )
    volume_window = CategoricalParameter(
        [12, 24, 48], default=24, space="buy", optimize=False, load=False
    )
    volume_ratio_min = CategoricalParameter(
        [0.8, 1.0, 1.3, 1.6],
        default=1.3,
        space="buy",
        optimize=False,
        load=False,
    )
    pressure_min = CategoricalParameter(
        [0.05, 0.1, 0.2, 0.35],
        default=0.35,
        space="buy",
        optimize=False,
        load=False,
    )
    use_close_direction_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    vp_window = IntParameter(
        24, 168, default=96, space="buy", optimize=False, load=False
    )
    vp_bins = IntParameter(24, 72, default=48, space="buy", optimize=False, load=False)
    vp_value_area_pct = DecimalParameter(
        0.55,
        0.85,
        decimals=2,
        default=0.70,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_price_source = CategoricalParameter(
        list(PRICE_SOURCE_CHOICES),
        default="hlc3",
        space="buy",
        optimize=False,
        load=False,
    )
    vp_smooth_bins = IntParameter(
        1, 6, default=3, space="buy", optimize=False, load=False
    )
    vp_hvn_threshold = DecimalParameter(
        0.50,
        0.90,
        decimals=2,
        default=0.70,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_lvn_threshold = DecimalParameter(
        0.10,
        0.55,
        decimals=2,
        default=0.35,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_pressure_delta_min = DecimalParameter(
        0.00,
        0.35,
        decimals=2,
        default=0.05,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_node_near_pct = CategoricalParameter(
        [0.005, 0.01, 0.02],
        default=0.01,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_volume_percentile_min = DecimalParameter(
        0.00,
        0.90,
        decimals=2,
        default=0.55,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_score_window = IntParameter(
        12, 120, default=48, space="buy", optimize=False, load=False
    )
    vp_fast_traverse_atr_mult = CategoricalParameter(
        [0.8, 1.2, 1.8],
        default=1.2,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_entry_score_margin = CategoricalParameter(
        [0.0, 0.02, 0.05, 0.1],
        default=0.02,
        space="buy",
        optimize=False,
        load=False,
    )
    use_vp_1h_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    vp_guard_mode = CategoricalParameter(
        list(VP_GUARD_MODES),
        default="score_or_context",
        space="buy",
        optimize=False,
        load=False,
    )
    vp_score_min = DecimalParameter(
        0.00,
        1.00,
        decimals=2,
        default=0.25,
        space="buy",
        optimize=False,
        load=False,
    )
    vp_context_min = DecimalParameter(
        0.00,
        1.00,
        decimals=2,
        default=0.28,
        space="buy",
        optimize=False,
        load=False,
    )
    use_vp_4h_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    vp_4h_window = IntParameter(
        12, 96, default=48, space="buy", optimize=False, load=False
    )
    vp_4h_bins = IntParameter(
        16, 64, default=36, space="buy", optimize=False, load=False
    )
    use_vp_1d_guard = BooleanParameter(
        default=False, space="buy", optimize=False, load=False
    )
    vp_1d_window = IntParameter(
        10, 84, default=30, space="buy", optimize=False, load=False
    )
    vp_1d_bins = IntParameter(
        16, 64, default=36, space="buy", optimize=False, load=False
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

    @staticmethod
    def _period_key(dates: Series, period: str) -> Series:
        if period == "week":
            iso = dates.dt.isocalendar()
            return iso["year"].astype("string").str.cat(
                iso["week"].astype("string").str.zfill(2),
                sep="-",
            )
        if period == "month":
            return dates.dt.strftime("%Y-%m")
        return dates.dt.strftime("%Y-%m-%d")

    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.volume_window.value)
        volume_mean = _num(dataframe, "volume").rolling(
            window,
            min_periods=max(2, window // 3),
        ).mean()
        candle_range = _num(dataframe, "high").sub(_num(dataframe, "low")).replace(
            0.0,
            np.nan,
        )
        close_location = (
            _num(dataframe, "close")
            .sub(_num(dataframe, "low"))
            .div(candle_range)
            .clip(0.0, 1.0)
        )
        dataframe["entry_close_location"] = close_location
        dataframe["entry_volume_ratio"] = _num(dataframe, "volume").div(
            volume_mean.replace(0.0, np.nan)
        )
        dataframe["entry_pressure"] = close_location.sub(0.5).mul(2.0)
        return dataframe

    def _add_prior_period_levels(self, dataframe: DataFrame) -> DataFrame:
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        for period in PERIOD_CHOICES:
            key = self._period_key(dates, period)
            grouped = (
                pd.DataFrame({"period": key, "high": high, "low": low})
                .groupby("period", sort=True)
                .agg(period_high=("high", "max"), period_low=("low", "min"))
            )
            grouped["prior_high"] = grouped["period_high"].shift(1)
            grouped["prior_low"] = grouped["period_low"].shift(1)
            dataframe[f"prior_{period}_high"] = key.map(grouped["prior_high"]).astype(
                "float64"
            )
            dataframe[f"prior_{period}_low"] = key.map(grouped["prior_low"]).astype(
                "float64"
            )
        return dataframe

    @staticmethod
    def _add_structural_targets(dataframe: DataFrame) -> DataFrame:
        high = _num(dataframe, "high").shift(1)
        low = _num(dataframe, "low").shift(1)
        for lookback in (48, 96, 168):
            dataframe[f"s3v2_swing_high_{lookback}"] = high.rolling(
                lookback,
                min_periods=max(4, lookback // 4),
            ).max()
        dataframe["s3v2_swing_low_48"] = low.rolling(48, min_periods=12).min()
        return dataframe

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
        pressure = (
            (body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0
        ).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        baseline = (
            volume.rolling(window, min_periods=max(2, window // 3))
            .sum()
            .replace(0.0, np.nan)
        )
        returns = close.pct_change(fill_method=None)
        window_return = close.pct_change(window, fill_method=None)
        volatility = returns.rolling(
            window,
            min_periods=max(3, window // 3),
        ).std() * np.sqrt(float(window))
        frame["s2m_pressure_ratio"] = (
            directional_volume.rolling(
                window,
                min_periods=max(2, window // 3),
            ).sum()
            / baseline
        )
        frame["s2m_trend_z"] = window_return / volatility.replace(0.0, np.nan)
        frame["s2m_close_location"] = close_location
        return frame

    def _add_sieve2_guard_indicators(self, dataframe: DataFrame) -> DataFrame:
        frame = self._add_market_state(
            dataframe.copy(),
            int(self.sieve2_market_window.value),
        )
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
            return (
                rejection & (base_score | base_context | node_entry | node_hold)
            ).fillna(False)
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
            return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(
                False
            )
        return (directional_pressure | trend_state).fillna(False)

    def _apply_sieve2_optional_guards(
        self,
        dataframe: DataFrame,
        condition: Series,
    ) -> Series:
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

    @staticmethod
    def _score_guard(
        dataframe: DataFrame,
        prefix: str,
        score_min: float,
    ) -> Series:
        score = _num(dataframe, f"{prefix}_score_long")
        opposite = _num(dataframe, f"{prefix}_score_short")
        return score.ge(score_min) & score.ge(opposite)

    @staticmethod
    def _context_guard(
        dataframe: DataFrame,
        prefix: str,
        context_min: float,
    ) -> Series:
        context = _num(dataframe, f"{prefix}_context_score_bull")
        opposite = _num(dataframe, f"{prefix}_context_score_bear")
        market_ok = _num(dataframe, f"{prefix}_market_context").ge(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(
        self,
        dataframe: DataFrame,
        prefix: str,
        mode: str,
        score_min: float,
        context_min: float,
    ) -> Series:
        score_ok = self._score_guard(dataframe, prefix, score_min)
        context_ok = self._context_guard(dataframe, prefix, context_min)
        balance_ok = _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)
        if mode == "score":
            return score_ok
        if mode == "context":
            return context_ok
        if mode == "score_or_context":
            return score_ok | context_ok
        if mode == "balance":
            return balance_ok
        return _num(dataframe, f"{prefix}_market_context").ge(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = self._add_volume_pressure(dataframe)
        frame = self._add_prior_period_levels(frame)
        frame = self._add_structural_targets(frame)
        frame = self._add_vp(frame)
        return self._add_sieve2_guard_indicators(frame)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        close = _num(dataframe, "close")
        level = _num(dataframe, "prior_month_high", np.nan)
        trigger = level.mul(1.0 + float(self.breakout_buffer_pct.value))
        condition = close.ge(trigger) & close.shift(1).lt(trigger.shift(1))
        if bool(self.use_volume_guard.value):
            condition &= _num(dataframe, "entry_volume_ratio").ge(
                float(self.volume_ratio_min.value)
            )
            condition &= _num(dataframe, "entry_pressure").ge(
                float(self.pressure_min.value)
            )
        if bool(self.use_close_direction_guard.value):
            condition &= close.gt(close.shift(1))
        if bool(self.use_vp_1h_guard.value):
            condition &= self._vp_guard(
                dataframe,
                "vp",
                str(self.vp_guard_mode.value),
                float(self.vp_score_min.value),
                float(self.vp_context_min.value),
            )
        condition = self._apply_sieve2_optional_guards(dataframe, condition)

        valid = (
            condition.fillna(False)
            & dataframe["volume"].gt(0.0)
            & dataframe["close"].notna()
        )
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
        close_times = dates + pd.to_timedelta(
            timeframe_to_minutes(self.timeframe),
            unit="m",
        )
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
        self,
        row: Series | None,
        entry_rate: float,
    ) -> tuple[
        dict[str, float | None],
        dict[str, str | None],
        float | None,
        float | None,
    ]:
        boundary = _finite_float(row.get("prior_month_high")) if row is not None else None
        if boundary is not None and boundary <= 0.0:
            boundary = None
        period_floor = _finite_float(row.get("prior_month_low")) if row is not None else None
        if (
            period_floor is not None
            and (
                period_floor <= 0.0
                or period_floor >= entry_rate
                or (boundary is not None and period_floor >= boundary)
            )
        ):
            period_floor = None

        raw: dict[str, float | None] = {
            "prior_day": self._directional_target(
                row.get("prior_day_high"),
                entry_rate,
            )
            if row is not None
            else None,
            "prior_week": self._directional_target(
                row.get("prior_week_high"),
                entry_rate,
            )
            if row is not None
            else None,
            "swing48": self._directional_target(
                row.get("s3v2_swing_high_48"),
                entry_rate,
            )
            if row is not None
            else None,
            "swing96": self._directional_target(
                row.get("s3v2_swing_high_96"),
                entry_rate,
            )
            if row is not None
            else None,
            "swing168": self._directional_target(
                row.get("s3v2_swing_high_168"),
                entry_rate,
            )
            if row is not None
            else None,
            "vp_hvn": self._directional_target(row.get("vp_hvn_above"), entry_rate)
            if row is not None
            else None,
            "vp_vah": self._directional_target(row.get("vp_vah"), entry_rate)
            if row is not None
            else None,
            "measured_half": None,
            "measured_full": None,
        }
        if boundary is not None and period_floor is not None:
            monthly_range = boundary - period_floor
            raw["measured_half"] = self._directional_target(
                boundary + monthly_range * 0.5,
                entry_rate,
            )
            raw["measured_full"] = self._directional_target(
                boundary + monthly_range,
                entry_rate,
            )

        targets: dict[str, float | None] = {}
        providers: dict[str, str | None] = {}
        for requested, chain in TARGET_PROVIDER_CHAINS.items():
            choices = [(name, raw.get(name)) for name in chain if raw.get(name) is not None]
            period_choices = [
                (name, raw.get(name))
                for name in ("prior_day", "prior_week")
                if raw.get(name) is not None
            ]
            if requested == "period_nearest" and period_choices:
                provider, target = min(period_choices, key=lambda item: float(item[1]))
            elif requested == "nearest" and choices:
                provider, target = min(choices, key=lambda item: float(item[1]))
            elif choices:
                provider, target = choices[0]
            else:
                provider, target = None, None
            targets[requested] = float(target) if target is not None else None
            providers[requested] = provider
        return targets, providers, boundary, period_floor

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
        targets, providers, boundary, period_floor = self._frozen_targets(row, entry_rate)
        plan = EXIT_PLANS[str(self.exit_policy_plan.value)]
        if plan.target_1 and plan.target_2:
            first = _finite_float(targets.get(plan.target_1))
            second = _finite_float(targets.get(plan.target_2))
            if first is not None and (
                second is None or second <= first * (1.0 + MIN_TARGET_MOVE)
            ):
                targets[plan.target_2] = None
                providers[plan.target_2] = None

        candle_date = None
        if row is not None and "date" in row.index:
            timestamp = self._utc(row["date"])
            candle_date = timestamp.isoformat() if timestamp is not None else None
        filled_at = (
            getattr(order, "order_filled_utc", None)
            or getattr(trade, "date_entry_fill_utc", None)
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
            "breakout_boundary": boundary,
            "period_floor": period_floor,
            "targets": targets,
            "target_providers": providers,
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "retest_held_at": None,
            "partial_filled": False,
            "partial_filled_at": None,
            "partial_tag": None,
            "partial_target_stake": None,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
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
                order_id = str(getattr(order, "order_id", "") or "")
                processed = list(state.get("partial_fill_order_ids") or [])
                if not order_id or order_id in processed:
                    return None
                plan = EXIT_PLANS[state["plan"]]
                fill_price = float(getattr(order, "safe_price", 0.0) or 0.0)
                fill_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
                leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
                filled_stake = fill_amount * fill_price / leverage
                target_stake = _finite_float(state.get("partial_target_stake"))
                if target_stake is None:
                    target_stake = (
                        float(getattr(trade, "stake_amount", 0.0) or 0.0) + filled_stake
                    ) * plan.partial_fraction
                    state["partial_target_stake"] = target_stake
                realized = float(state.get("partial_realized_stake") or 0.0) + filled_stake
                state["partial_realized_stake"] = realized
                state["partial_fill_order_ids"] = [*processed, order_id]
                state["partial_tag"] = tag
                if target_stake > 0.0 and realized >= target_stake * 0.995:
                    filled_at = getattr(order, "order_filled_utc", None) or current_time
                    state["partial_filled"] = True
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
        boundary_1: bool,
        boundary_2: bool,
        retest_1: bool,
        retest_2: bool,
    ) -> bool:
        return {
            "none": False,
            "boundary1": boundary_1,
            "boundary2": boundary_2,
            "retest1": retest_1,
            "retest2": retest_2,
        }.get(mode, False)

    def _events(
        self,
        trade: Any,
        state: dict[str, Any],
        frame: DataFrame,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = trade, current_time, current_rate
        plan = EXIT_PLANS[state["plan"]]
        targets = state.get("targets", {})
        target_frame = frame
        entry_filled_at = self._utc(state.get("entry_filled_at"))
        if entry_filled_at is not None and "date" in frame.columns:
            target_frame = frame.loc[
                pd.to_datetime(frame["date"], utc=True, errors="coerce").ge(
                    entry_filled_at
                )
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
                pd.to_datetime(
                    target_frame["date"],
                    utc=True,
                    errors="coerce",
                ).ge(partial_filled_at)
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

        close = _num(target_frame, "close")
        low = _num(target_frame, "low")
        boundary = _finite_float(state.get("breakout_boundary"))
        period_floor = _finite_float(state.get("period_floor"))
        if boundary is None or target_frame.empty:
            boundary_condition = pd.Series(False, index=target_frame.index, dtype="bool")
            held_retest = pd.Series(False, index=target_frame.index, dtype="bool")
        else:
            boundary_condition = close.lt(boundary * (1.0 - LEVEL_BAND))
            held_retest = low.le(boundary * (1.0 + RETEST_BAND)) & close.ge(
                boundary * (1.0 - LEVEL_BAND)
            )
        dates = pd.to_datetime(
            target_frame.get("date", pd.Series(index=target_frame.index, dtype="object")),
            utc=True,
            errors="coerce",
        )
        persisted_held_at = self._utc(state.get("retest_held_at"))
        held_positions = np.flatnonzero((held_retest & dates.notna()).to_numpy())
        observed_held_at = (
            self._utc(dates.iloc[int(held_positions[0])])
            if held_positions.size
            else None
        )
        held_timestamps = [
            timestamp
            for timestamp in (persisted_held_at, observed_held_at)
            if timestamp is not None
        ]
        if held_timestamps:
            state["retest_held_at"] = min(held_timestamps).isoformat()
        prior_hold_seen = pd.Series(False, index=target_frame.index, dtype="bool")
        if persisted_held_at is not None:
            prior_hold_seen |= dates.gt(persisted_held_at)
        if observed_held_at is not None:
            prior_hold_seen |= dates.gt(observed_held_at)
        failed_retest = boundary_condition & prior_hold_seen

        boundary_1 = self._last_n(boundary_condition, 1)
        boundary_2 = self._last_n(boundary_condition, 2)
        retest_1 = self._last_n(failed_retest, 1)
        retest_2 = self._last_n(failed_retest, 2)
        last_close = _finite_float(close.iloc[-1]) if not close.empty else None
        floor_failure = bool(
            period_floor is not None
            and last_close is not None
            and last_close < period_floor * (1.0 - LEVEL_BAND)
        )
        deep_boundary_failure = bool(
            boundary is not None
            and last_close is not None
            and last_close < boundary * (1.0 - DEEP_FAILURE_BAND)
        )

        age_candles = len(target_frame)
        progress_high = _finite_float(_num(target_frame, "high").max())
        favorable = (
            progress_high / entry_rate - 1.0
            if progress_high is not None and entry_rate > 0.0
            else 0.0
        )
        stalled_near_boundary = bool(
            boundary is not None
            and last_close is not None
            and last_close <= boundary * (1.0 + FOLLOW_THROUGH_MOVE)
        )
        time_failure = bool(
            plan.progress_candles
            and age_candles >= plan.progress_candles
            and favorable < FOLLOW_THROUGH_MOVE
            and stalled_near_boundary
        )
        max_hold = bool(
            plan.max_hold_candles and age_candles >= plan.max_hold_candles
        )
        return {
            "hard_invalidation": (
                False
                if plan.role == "baseline"
                else floor_failure or deep_boundary_failure
            ),
            "invalidation": self._invalidation_event(
                plan.invalidation,
                boundary_1,
                boundary_2,
                retest_1,
                retest_2,
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
        partial_decision = (
            Sieve3V2IntegratedFromPriorMonthHighBreakoutLong._partial_decision(
                plan,
                state,
                events,
            )
        )
        if partial_decision is not None:
            return partial_decision
        if plan.target_action == "tighten" and events.get("target_1"):
            return ExitDecision("tighten", tighten="target")
        if plan.invalidation_action == "tighten" and events.get("invalidation"):
            return ExitDecision("tighten", tighten="invalidation")
        if events.get("max_hold"):
            return ExitDecision("full", "s3v2_max_hold")
        return ExitDecision("hold")

    @staticmethod
    def _partial_decision(
        plan: ExitPlan,
        state: dict[str, Any],
        events: dict[str, Any],
    ) -> ExitDecision | None:
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
        return None

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
        frame = self._analyzed_frame(pair, current_time)
        partial_pending = self._has_pending_partial(trade)
        events, state = self._events(
            trade,
            state,
            frame,
            current_time,
            current_rate,
            current_profit,
        )
        state["partial_pending"] = partial_pending
        plan = EXIT_PLANS[state["plan"]]
        decision = self.evaluate_policy(plan, state, events)
        if partial_pending:
            state["phase"] = "REALIZATION_PENDING"
            if decision.action == "full" and not state.get("terminal_exit_pending"):
                state["terminal_exit_pending"] = True
                state["terminal_exit_tag"] = decision.tag or "s3v2_deferred_full_exit"
            elif state.get("terminal_exit_pending") and not state.get("terminal_exit_tag"):
                state["terminal_exit_tag"] = decision.tag or "s3v2_hard_invalidation"
        if state != previous:
            self._save_state(trade, state)
        return plan, state, events, decision

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
        context = self._decision_context(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
        )
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
        target_stake = _finite_float(state.get("partial_target_stake"))
        if target_stake is None:
            target_stake = stake * decision.fraction
            state["partial_target_stake"] = target_stake
        realized = float(state.get("partial_realized_stake") or 0.0)
        remaining = max(0.0, target_stake - realized)
        if remaining <= max(1e-8, target_stake * 0.005):
            return None
        request = min(remaining, stake)
        if min_stake is not None and 0.0 < stake - request < float(min_stake):
            request = max(0.0, stake - float(min_stake))
        if request <= 0.0 or request >= stake:
            return None
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
        period_floor = _finite_float(state.get("period_floor"))
        if plan.role != "baseline" and period_floor is not None and period_floor < entry_rate:
            stop_price = max(stop_price, period_floor * (1.0 - LEVEL_BAND))
        if state.get("partial_filled"):
            if plan.remainder == "breakeven":
                stop_price = max(stop_price, entry_rate * 1.001)
            elif plan.remainder == "trail":
                stop_price = max(stop_price, current_rate * 0.985)
        if decision.action == "tighten" and decision.tighten == "target":
            target = _finite_float(state.get("targets", {}).get(plan.target_1))
            if target is not None:
                stop_price = max(stop_price, target * 0.99)
        if decision.action == "tighten" and decision.tighten == "invalidation":
            boundary = _finite_float(state.get("breakout_boundary"))
            if boundary is not None:
                stop_price = max(stop_price, boundary * (1.0 - LEVEL_BAND))
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
        context = self._decision_context(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
        )
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
