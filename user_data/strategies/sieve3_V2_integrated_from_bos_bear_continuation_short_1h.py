"""Sieve3 V2 integrated exits for the promoted BOS bear-continuation short.

Preflight contract
------------------
Canonical Sieve2 source:
    sieve2_bos_bear_continuation_short_1h
Authoritative promotion evidence:
    Approved migration ledger schema v3, generated 2026-07-15T08:36:08.860415+01:00.
    Result job 20260521T012740_entry_all_resume, physical row 35.
    Result row SHA256:
    39ed7ff5fefdfc9b7da8f1b83c5cd6605446e80d9606dbb6cd84c6aa6353c1fa
    Backtest ZIP SHA256:
    f23276a5044d8240d3964066934a7bdf0f98fe6fd14f73d77939a97424467279
    Archived strategy SHA256:
    6bc7daa51224de20f6571caa9aac720ef10a3c7ada785cad62d9d07ac0d36960
    Archived params SHA256:
    d84941160db4e1b18b8df1928bb1f459046e7f2760deac5c3cbefac386b7c70d
    Archived source-default SHA256:
    6f74863399ab0c0e3d1350a8c5adda7eeddfb677ad90097dd2147c04e7c2a44b
    Selected buy-overlay SHA256:
    cf93f3da9737a5dfe19ce4805080d072bcc52ac5280e8206683eff99c11d184f
    Effective buy-lock SHA256:
    e0dafbb57757201a00342a94e1ea63361f2071fd2dedddd520117a851769a4fd
    Baseline entry signature SHA256:
    4cfa1adb917d25bc350f21803d65f3fb9d257fd68651ac17dd30516e82b96df4
Promoted baseline:
    TP/SL 2%/2%, 36 trades, 55.5556% win rate, 0.559391% return,
    1.267514% maximum drawdown, and 1.167154 profit factor.
Entry lock:
    EFFECTIVE_BUY_LOCK is the exact archived source-default map overlaid by the
    eight selected buy parameters. Every buy parameter is immutable in Sieve3
    V2 (``optimize=False, load=False``). Entry logic is otherwise unchanged;
    BOS diagnostics only expose confirmed swing and broken-level columns from
    the same indicator call and do not alter the entry-driving outputs.
Tested controls preserved:
    sieve3_exit_breakeven_from_bos_bear_continuation_short_1h.py
      jobs 20260613T_repaired_sieve3_exit_multimetric_fresh40_10pairs_longcycle_300epoch_2win_2seed_cores20
      and 20260705T180307_entry_sieve3_exit_speed_batch_043.
    sieve3_exit_fixed_tp_sl_from_bos_bear_continuation_short_1h.py
      job 20260710T061808_entry_sieve3_exit_speed_batch_048.
    sieve3_exit_level_zone_reversal_from_bos_bear_continuation_short_1h.py
      jobs 20260621T154147_entry_sieve3_exit_speed_batch_016 through
      20260624T100159_entry_sieve3_exit_speed_seedcheck_sieve3_exit_pattern_autobatch_022_seed314159.
    Refined invalidation/layered/target-zone pilot controls remain preserved in
      job 20260711T230600_entry_sieve3_exit_rework_comparison_001.
Consolidated hypotheses:
    Fixed 3/3 and promoted 2/2 controls; pure BOS/VP/swing targets; pure
    broken-level, opposing-structure, and VP-context invalidations; target
    partials with breakeven or closed-candle structure trails; invalidation
    reductions; target tightening; dual-target runners; and explicit progress
    failure. Generic unrelated TA exits and duplicate legacy branches are not
    copied.
Target semantics:
    ``frozen_at_entry``. The last closed placement/signal candle freezes the
    confirmed BOS swing, shifted prior lows, source-computed VP levels, and BOS
    measured moves. Explicit provider chains may choose a named source-coherent
    fallback only during initialization. Touch, reversal, progress, and trail
    evidence starts at the actual fill and uses closed candles only.
Invalidation contract:
    A short thesis weakens when price closes back above the frozen broken
    support, bullish BOS/CHoCH appears, bullish structure persists, or the
    source-computed VP context turns bullish. The universal hard thesis failure
    is a close above the frozen invalidating swing, or a two-close support
    reclaim confirmed by bullish structure/VP evidence.
Named policy groups:
    32 complete plans: 2 baseline controls, 6 pure targets, 6 pure
    invalidations, 1 defensive tighten plan, 8 target-partial layered plans,
    3 invalidation-reduce plans, 2 target-tighten plans, 2 dual-target runners,
    and 2 progress-failure plans.
Active Hyperopt surface:
    One sell-space ``exit_policy_plan`` parameter. Every category is a complete
    policy; there are no hidden Boolean gates or branch-local sweep parameters.
Split decision:
    One file is sufficient because confirmed BOS swings, the broken support,
    measured moves, and the source's existing VP guard dataframe form one
    coherent structure-management ecosystem.
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
from user_data.Indicators.pattern_bos_choch import add_bos_choch


SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = "sieve3_v2_integrated_bos_structure_exit"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = (
    "Frozen BOS/VP/swing targets interact with broken-level and opposing-structure "
    "invalidation through complete layered short-management policies."
)

SOURCE_RESULT_ROW_SHA256 = (
    "39ed7ff5fefdfc9b7da8f1b83c5cd6605446e80d9606dbb6cd84c6aa6353c1fa"
)
SOURCE_BACKTEST_ARCHIVE_SHA256 = (
    "f23276a5044d8240d3964066934a7bdf0f98fe6fd14f73d77939a97424467279"
)
SOURCE_SNAPSHOT_STRATEGY_SHA256 = (
    "6bc7daa51224de20f6571caa9aac720ef10a3c7ada785cad62d9d07ac0d36960"
)
SOURCE_SNAPSHOT_PARAMS_SHA256 = (
    "d84941160db4e1b18b8df1928bb1f459046e7f2760deac5c3cbefac386b7c70d"
)
SOURCE_EFFECTIVE_BUY_LOCK_SHA256 = (
    "e0dafbb57757201a00342a94e1ea63361f2071fd2dedddd520117a851769a4fd"
)
SOURCE_BASELINE_ENTRY_SIGNATURE_SHA256 = (
    "4cfa1adb917d25bc350f21803d65f3fb9d257fd68651ac17dd30516e82b96df4"
)

ENTRY_TAG = "bos_bear_continuation_short_1h"
STATE_KEY = "sieve3_v2_bos_bear_short"
STATE_VERSION = 1
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

EFFECTIVE_BUY_LOCK: dict[str, Any] = {
    "use_sieve2_vp_guard": False,
    "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96,
    "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25,
    "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT",
    "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True,
    "volume_guard_window": 24,
    "volume_ratio_min": 1.6,
    "use_pressure_guard": True,
    "pressure_window": 48,
    "pressure_min": 0.2,
    "use_accumulation_guard": False,
    "use_body_direction_guard": False,
    "use_close_direction_guard": False,
    "strength": 3,
    "min_prominence_atr": 0.35,
    "min_pivot_spacing_bars": 2,
    "max_pivot_age_bars": 96,
    "breakout_buffer_atr": 0.3,
    "use_state_guard": False,
}


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
DEFAULT_MAX_HOLD_CANDLES = 336


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

    def signature(self) -> str:
        return "|".join(str(value) for value in self._values())

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
    "measured_half_touch_full": ExitPlan(
        "target_full",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "measured_full_reversal1_full": ExitPlan(
        "target_full",
        "measured_full",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "swing_reversal1_full": ExitPlan(
        "target_full",
        "swing",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.015,
    ),
    "prior96_reversal2of3_full": ExitPlan(
        "target_full",
        "prior96",
        confirmation="reversal2of3",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.03,
    ),
    "vp_lvn_touch_full": ExitPlan(
        "target_full",
        "vp_lvn",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "vp_poc_reversal1_full": ExitPlan(
        "target_full",
        "vp_poc",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.015,
    ),
    "reclaim1_full": ExitPlan(
        "invalidation_full", invalidation="reclaim1", invalidation_action="full"
    ),
    "reclaim2_full": ExitPlan(
        "invalidation_full", invalidation="reclaim2", invalidation_action="full"
    ),
    "opposite1_full": ExitPlan(
        "invalidation_full", invalidation="opposite1", invalidation_action="full"
    ),
    "opposite_state2_full": ExitPlan(
        "invalidation_full", invalidation="opposite_state2", invalidation_action="full"
    ),
    "vp2_full": ExitPlan(
        "invalidation_full", invalidation="vp2", invalidation_action="full"
    ),
    "combined1_full": ExitPlan(
        "invalidation_full", invalidation="combined1", invalidation_action="full"
    ),
    "reclaim2_tighten": ExitPlan(
        "defensive_tighten", invalidation="reclaim2", invalidation_action="tighten"
    ),
    "measured_half_touch_p33_be_reclaim2": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="touch",
        target_action="partial",
        invalidation="reclaim2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "measured_half_reversal1_p50_trail_combined1": ExitPlan(
        "target_partial",
        "measured_half",
        confirmation="reversal1",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "measured_full_reversal1_p33_trail_opposite1": ExitPlan(
        "target_partial",
        "measured_full",
        confirmation="reversal1",
        target_action="partial",
        invalidation="opposite1",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "swing_touch_p33_be_reclaim2": ExitPlan(
        "target_partial",
        "swing",
        confirmation="touch",
        target_action="partial",
        invalidation="reclaim2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "prior96_reversal2of3_p50_trail_combined1": ExitPlan(
        "target_partial",
        "prior96",
        confirmation="reversal2of3",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.50,
        remainder="trail",
        target_band=0.03,
    ),
    "vp_lvn_touch_p33_be_reclaim2": ExitPlan(
        "target_partial",
        "vp_lvn",
        confirmation="touch",
        target_action="partial",
        invalidation="reclaim2",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "vp_poc_reversal1_p50_trail_vp2": ExitPlan(
        "target_partial",
        "vp_poc",
        confirmation="reversal1",
        target_action="partial",
        invalidation="vp2",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "nearest_reversal1_p50_trail_combined1": ExitPlan(
        "target_partial",
        "nearest",
        confirmation="reversal1",
        target_action="partial",
        invalidation="combined1",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "reclaim1_reduce33_measured_half": ExitPlan(
        "invalidation_reduce",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="reclaim1",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "opposite1_reduce33_measured_full": ExitPlan(
        "invalidation_reduce",
        "measured_full",
        confirmation="touch",
        target_action="full",
        invalidation="opposite1",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "vp2_reduce50_nearest": ExitPlan(
        "invalidation_reduce",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="vp2",
        invalidation_action="partial",
        partial_fraction=0.50,
        remainder="target",
    ),
    "measured_half_touch_tighten_reclaim2": ExitPlan(
        "target_tighten",
        "measured_half",
        confirmation="touch",
        target_action="tighten",
        invalidation="reclaim2",
        invalidation_action="full",
        remainder="tighten",
    ),
    "swing_touch_tighten_combined1": ExitPlan(
        "target_tighten",
        "swing",
        confirmation="touch",
        target_action="tighten",
        invalidation="combined1",
        invalidation_action="full",
        remainder="tighten",
    ),
    "measured_half_p33_then_measured_full_runner": ExitPlan(
        "dual_target",
        "measured_half",
        "measured_full",
        "touch",
        "partial",
        "combined1",
        "full",
        0.33,
        "trail",
    ),
    "vp_lvn_p50_then_measured_full_runner": ExitPlan(
        "dual_target",
        "vp_lvn",
        "measured_full",
        "touch",
        "partial",
        "reclaim2",
        "full",
        0.50,
        "trail",
    ),
    "progress48_measured_half": ExitPlan(
        "progress_failure",
        "measured_half",
        confirmation="touch",
        target_action="full",
        invalidation="combined1",
        invalidation_action="full",
        progress_candles=48,
    ),
    "progress96_measured_full": ExitPlan(
        "progress_failure",
        "measured_full",
        confirmation="touch",
        target_action="full",
        invalidation="combined1",
        invalidation_action="full",
        progress_candles=96,
    ),
}

EXIT_PLAN_SIGNATURES = {name: plan.signature() for name, plan in EXIT_PLANS.items()}

TARGET_PROVIDER_CHAINS: dict[str, tuple[str, ...]] = {
    "measured_half": ("measured_half", "prior48", "vp_lvn"),
    "measured_full": ("measured_full", "prior96", "vp_lvn"),
    "swing": ("swing", "prior48", "measured_half"),
    "prior48": ("prior48", "prior96", "measured_half"),
    "prior96": ("prior96", "prior48", "measured_half"),
    "vp_hvn": ("vp_hvn", "vp_lvn", "vp_val", "measured_half"),
    "vp_lvn": ("vp_lvn", "vp_hvn", "vp_val", "vp_poc", "measured_half"),
    "vp_poc": ("vp_poc", "vp_val", "vp_lvn", "measured_half"),
    "vp_val": ("vp_val", "vp_poc", "vp_lvn", "measured_half"),
    "nearest": (
        "swing",
        "prior48",
        "prior96",
        "vp_hvn",
        "vp_lvn",
        "vp_poc",
        "vp_val",
        "measured_half",
        "measured_full",
    ),
}


class Sieve3V2IntegratedFromBosBearContinuationShort1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    SIEVE_STAGE = SIEVE_STAGE
    SOURCE_STRATEGY = SOURCE_STRATEGY
    SOURCE_RESULT_BATCH = SOURCE_RESULT_BATCH
    RESEARCH_PATH = RESEARCH_PATH
    ENTRY_SOURCE_STAGE = ENTRY_SOURCE_STAGE
    EXIT_HYPOTHESIS = EXIT_HYPOTHESIS

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

    use_sieve2_vp_guard = CategoricalParameter(
        [False, True],
        default=EFFECTIVE_BUY_LOCK["use_sieve2_vp_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_guard_mode = CategoricalParameter(
        list(SIEVE2_VP_GUARD_MODES),
        default=EFFECTIVE_BUY_LOCK["sieve2_vp_guard_mode"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_window = CategoricalParameter(
        [48, 96, 168],
        default=EFFECTIVE_BUY_LOCK["sieve2_vp_window"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_bins = CategoricalParameter(
        [24, 36, 48],
        default=EFFECTIVE_BUY_LOCK["sieve2_vp_bins"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_score_min = CategoricalParameter(
        [0.15, 0.25, 0.35, 0.50],
        default=EFFECTIVE_BUY_LOCK["sieve2_vp_score_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_vp_context_min = CategoricalParameter(
        [0.18, 0.28, 0.38, 0.50],
        default=EFFECTIVE_BUY_LOCK["sieve2_vp_context_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_sieve2_market_guard = CategoricalParameter(
        [False, True],
        default=EFFECTIVE_BUY_LOCK["use_sieve2_market_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_guard_mode = CategoricalParameter(
        list(SIEVE2_MARKET_GUARD_MODES),
        default=EFFECTIVE_BUY_LOCK["sieve2_market_guard_mode"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_window = CategoricalParameter(
        [12, 24, 48, 96],
        default=EFFECTIVE_BUY_LOCK["sieve2_market_window"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_pressure_min = CategoricalParameter(
        [0.03, 0.07, 0.12, 0.18, 0.25],
        default=EFFECTIVE_BUY_LOCK["sieve2_market_pressure_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_market_trend_min = CategoricalParameter(
        [0.0, 0.25, 0.50, 0.80],
        default=EFFECTIVE_BUY_LOCK["sieve2_market_trend_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_rs_benchmark_pair = CategoricalParameter(
        ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        default=EFFECTIVE_BUY_LOCK["sieve2_rs_benchmark_pair"],
        space="buy",
        optimize=False,
        load=False,
    )
    sieve2_rs_score_min = CategoricalParameter(
        [0.35, 0.45, 0.55, 0.65],
        default=EFFECTIVE_BUY_LOCK["sieve2_rs_score_min"],
        space="buy",
        optimize=False,
        load=False,
    )

    use_volume_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_volume_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    volume_guard_window = CategoricalParameter(
        [12, 24, 48],
        default=EFFECTIVE_BUY_LOCK["volume_guard_window"],
        space="buy",
        optimize=False,
        load=False,
    )
    volume_ratio_min = CategoricalParameter(
        [0.8, 1.0, 1.3, 1.6],
        default=EFFECTIVE_BUY_LOCK["volume_ratio_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_pressure_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_pressure_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    pressure_window = CategoricalParameter(
        [12, 24, 48],
        default=EFFECTIVE_BUY_LOCK["pressure_window"],
        space="buy",
        optimize=False,
        load=False,
    )
    pressure_min = CategoricalParameter(
        [0.05, 0.1, 0.15, 0.2, 0.35],
        default=EFFECTIVE_BUY_LOCK["pressure_min"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_accumulation_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_accumulation_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_body_direction_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_body_direction_guard"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_close_direction_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_close_direction_guard"],
        space="buy",
        optimize=False,
        load=False,
    )

    strength = IntParameter(
        2,
        5,
        default=EFFECTIVE_BUY_LOCK["strength"],
        space="buy",
        optimize=False,
        load=False,
    )
    min_prominence_atr = DecimalParameter(
        0.20,
        1.20,
        decimals=2,
        default=EFFECTIVE_BUY_LOCK["min_prominence_atr"],
        space="buy",
        optimize=False,
        load=False,
    )
    min_pivot_spacing_bars = IntParameter(
        1,
        8,
        default=EFFECTIVE_BUY_LOCK["min_pivot_spacing_bars"],
        space="buy",
        optimize=False,
        load=False,
    )
    max_pivot_age_bars = IntParameter(
        24,
        160,
        default=EFFECTIVE_BUY_LOCK["max_pivot_age_bars"],
        space="buy",
        optimize=False,
        load=False,
    )
    breakout_buffer_atr = CategoricalParameter(
        [0.0, 0.15, 0.3],
        default=EFFECTIVE_BUY_LOCK["breakout_buffer_atr"],
        space="buy",
        optimize=False,
        load=False,
    )
    use_state_guard = BooleanParameter(
        default=EFFECTIVE_BUY_LOCK["use_state_guard"],
        space="buy",
        optimize=False,
        load=False,
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
        if bool(self.use_volume_guard.value):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window.value)
            baseline = (
                volume.shift(1)
                .rolling(window, min_periods=max(2, window // 3))
                .mean()
                .replace(0.0, np.nan)
            )
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min.value)))
        if bool(self.use_pressure_guard.value) or bool(self.use_accumulation_guard.value):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = (
                (body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0
            ).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard.value):
            window = int(self.pressure_window.value)
            pressure_baseline = (
                volume.rolling(window, min_periods=max(2, window // 3))
                .sum()
                .replace(0.0, np.nan)
            )
            pressure_ratio = (
                directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
                / pressure_baseline
            )
            guard &= pressure_ratio.le(-float(self.pressure_min.value))
        if bool(self.use_accumulation_guard.value):
            window = int(self.pressure_window.value)
            accumulation = directional_volume.rolling(
                window, min_periods=max(2, window // 3)
            ).sum()
            guard &= accumulation.le(0.0)
        if bool(self.use_body_direction_guard.value):
            guard &= close.lt(_num(dataframe, "open"))
        if bool(self.use_close_direction_guard.value):
            guard &= close.lt(close.shift(1))
        return guard.fillna(False)

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
            window, min_periods=max(3, window // 3)
        ).std() * np.sqrt(float(window))
        frame["s2m_pressure_ratio"] = (
            directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            / baseline
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
        high = _num(frame, "high")
        score = _num(frame, "s2vp_score_short")
        other = _num(frame, "s2vp_score_long")
        context = _num(frame, "s2vp_context_score_bear")
        in_value = _bool(frame, "s2vp_in_value_area")
        below_value = _bool(frame, "s2vp_below_value_area")
        node_entry = _bool(frame, "s2vp_node_entry_short")
        node_hold = _bool(frame, "s2vp_node_hold_short")
        prior_vah = _num(frame, "s2vp_prior_vah", np.nan)
        prior_val = _num(frame, "s2vp_prior_val", np.nan)
        base_score = score.ge(score_min) & score.gt(other)
        base_context = context.ge(context_min)
        if mode == "node_confirm":
            return (node_entry | node_hold | (base_score & base_context)).fillna(False)
        if mode == "value_area_confirm":
            return (below_value | (in_value & base_context)).fillna(False)
        if mode == "breakout_acceptance":
            return (
                close.lt(prior_val) & (base_score | base_context | node_entry | node_hold)
            ).fillna(False)
        if mode == "rejection_confirm":
            rejection = high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val)
            return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(
                False
            )
        if mode == "poc_hvn_reject":
            return (node_entry | (base_score & base_context)).fillna(False)
        if mode == "prior_level_confirm":
            return (close.lt(prior_val) & (base_score | base_context)).fillna(False)
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
        directional_pressure = pressure.le(-pressure_min)
        trend_state = trend.le(-trend_min)
        avoids_adverse_pressure = pressure.le(pressure_min)
        avoids_adverse_trend = trend.le(trend_min)
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

    def _apply_sieve2_optional_guards(
        self, dataframe: DataFrame, condition: Series
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

    def _add_structure(self, dataframe: DataFrame) -> DataFrame:
        return add_bos_choch(
            dataframe,
            strength=int(self.strength.value),
            min_prominence_atr=float(self.min_prominence_atr.value),
            min_pivot_spacing_bars=int(self.min_pivot_spacing_bars.value),
            max_pivot_age_bars=int(self.max_pivot_age_bars.value),
            breakout_buffer_atr=float(self.breakout_buffer_atr.value),
            include_diagnostics=True,
            prefix="ms",
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = self._add_structure(dataframe)
        frame = self._add_sieve2_guard_indicators(frame)
        low = _num(frame, "low")
        frame["s3v2_prior_low_48"] = low.shift(1).rolling(48, min_periods=12).min()
        frame["s3v2_prior_low_96"] = low.shift(1).rolling(96, min_periods=24).min()
        return frame

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_bos_to_bear")
        if bool(self.use_state_guard.value):
            condition &= _num(dataframe, "ms_state").le(0)
        condition &= self._common_guards(dataframe)
        condition = self._apply_sieve2_optional_guards(dataframe, condition)
        valid = (
            condition.fillna(False)
            & dataframe["volume"].gt(0.0)
            & dataframe["close"].notna()
        )
        dataframe.loc[valid, "enter_short"] = 1
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
        if target is None or target >= entry_rate * (1.0 - MIN_TARGET_MOVE):
            return None
        return target

    @staticmethod
    def _directional_invalidation(value: Any, entry_rate: float) -> float | None:
        invalidation = _finite_float(value)
        if invalidation is None or invalidation <= entry_rate:
            return None
        return invalidation

    def _frozen_targets(
        self, row: Series | None, entry_rate: float
    ) -> tuple[
        dict[str, float | None],
        dict[str, str | None],
        float | None,
        float | None,
    ]:
        broken = _finite_float(row.get("ms_break_level")) if row is not None else None
        invalidation = (
            self._directional_invalidation(row.get("ms_invalidation_level"), entry_rate)
            if row is not None
            else None
        )
        if invalidation is None and row is not None:
            invalidation = self._directional_invalidation(
                row.get("ms_last_swing_high"), entry_rate
            )

        raw: dict[str, float | None] = {
            "swing": self._directional_target(row.get("ms_prev_swing_low"), entry_rate)
            if row is not None
            else None,
            "prior48": self._directional_target(row.get("s3v2_prior_low_48"), entry_rate)
            if row is not None
            else None,
            "prior96": self._directional_target(row.get("s3v2_prior_low_96"), entry_rate)
            if row is not None
            else None,
            "vp_hvn": self._directional_target(row.get("s2vp_hvn_below"), entry_rate)
            if row is not None
            else None,
            "vp_lvn": self._directional_target(row.get("s2vp_lvn_below"), entry_rate)
            if row is not None
            else None,
            "vp_poc": self._directional_target(row.get("s2vp_poc"), entry_rate)
            if row is not None
            else None,
            "vp_val": self._directional_target(row.get("s2vp_val"), entry_rate)
            if row is not None
            else None,
            "measured_half": None,
            "measured_full": None,
        }
        if broken is not None and invalidation is not None and invalidation > broken:
            measured_range = invalidation - broken
            raw["measured_half"] = self._directional_target(
                broken - measured_range * 0.5, entry_rate
            )
            raw["measured_full"] = self._directional_target(
                broken - measured_range, entry_rate
            )

        targets: dict[str, float | None] = {}
        providers: dict[str, str | None] = {}
        for requested, chain in TARGET_PROVIDER_CHAINS.items():
            choices = [(name, raw.get(name)) for name in chain if raw.get(name) is not None]
            if requested == "nearest" and choices:
                provider, target = max(choices, key=lambda item: float(item[1]))
            elif choices:
                provider, target = choices[0]
            else:
                provider, target = None, None
            targets[requested] = float(target) if target is not None else None
            providers[requested] = provider
        return targets, providers, broken, invalidation

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
        targets, providers, broken, invalidation = self._frozen_targets(row, entry_rate)
        plan = EXIT_PLANS[str(self.exit_policy_plan.value)]
        if plan.target_1 and plan.target_2:
            first = _finite_float(targets.get(plan.target_1))
            second = _finite_float(targets.get(plan.target_2))
            if first is not None and (
                second is None or second >= first * (1.0 - MIN_TARGET_MOVE)
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
            "broken_support": broken,
            "structural_invalidation": invalidation,
            "targets": targets,
            "target_providers": providers,
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "closed_favorable_low": None,
            "partial_filled": False,
            "partial_filled_at": None,
            "partial_tag": None,
            "partial_target_stake": None,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
            "terminal_fill": False,
            "terminal_exit_pending": False,
            "terminal_exit_tag": None,
            "stop_ceiling": None,
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

    @staticmethod
    def _is_terminal_trade(trade: Any) -> bool:
        if getattr(trade, "is_open", None) is False:
            return True
        amount = _finite_float(getattr(trade, "amount", None))
        if amount is not None and amount <= 1e-12:
            return True
        stake = _finite_float(getattr(trade, "stake_amount", None))
        return stake is not None and stake <= 1e-8

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
                filled_at = getattr(order, "order_filled_utc", None) or current_time
                if self._is_terminal_trade(trade):
                    state["partial_filled"] = True
                    state["partial_filled_at"] = self._utc(filled_at).isoformat()
                    state["terminal_fill"] = True
                    state["terminal_exit_pending"] = False
                    state["terminal_exit_tag"] = None
                    state["phase"] = "TERMINAL"
                elif target_stake > 0.0 and realized >= target_stake * 0.995:
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
        touch_threshold = min(
            target * (1.0 + band),
            entry_rate * (1.0 - MIN_TARGET_MOVE),
        )
        if prior_touch is None:
            touched_rows = _num(frame, "low").le(touch_threshold) & dates.notna()
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
        opposite = _num(post_touch, "close").gt(_num(post_touch, "open"))
        if confirmation == "reversal1":
            return prior_touch_iso, bool(opposite.tail(1).all())
        if confirmation == "reversal2of3":
            return prior_touch_iso, len(opposite) >= 3 and int(opposite.tail(3).sum()) >= 2
        return prior_touch_iso, False

    @staticmethod
    def _invalidation_event(
        mode: str,
        reclaim_1: bool,
        reclaim_2: bool,
        opposite_1: bool,
        opposite_state_2: bool,
        vp_2: bool,
        combined_1: bool,
    ) -> bool:
        return {
            "none": False,
            "reclaim1": reclaim_1,
            "reclaim2": reclaim_2,
            "opposite1": opposite_1,
            "opposite_state2": opposite_state_2,
            "vp2": vp_2,
            "combined1": combined_1,
        }.get(mode, False)

    def _post_fill_frame(self, frame: DataFrame, filled_at: Any) -> DataFrame:
        timestamp = self._utc(filled_at)
        if timestamp is None or frame.empty or "date" not in frame.columns:
            return frame
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        return frame.loc[dates.ge(timestamp)]

    def _events(
        self,
        state: dict[str, Any],
        frame: DataFrame,
        current_profit: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        plan = EXIT_PLANS[state["plan"]]
        targets = state.get("targets", {})
        post_fill = self._post_fill_frame(frame, state.get("entry_filled_at"))
        entry_rate = float(state.get("entry_rate") or 0.0)
        touched_1_at, confirmed_1 = self._confirmation(
            post_fill,
            _finite_float(targets.get(plan.target_1)) if plan.target_1 else None,
            plan.confirmation,
            state.get("target_1_touched_at"),
            plan.target_band,
            entry_rate,
        )
        target_2_frame = self._post_fill_frame(post_fill, state.get("partial_filled_at"))
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
            "TERMINAL",
        }:
            state["phase"] = "TARGET_ZONE"

        close = _num(post_fill, "close")
        broken = _finite_float(state.get("broken_support"))
        invalidation = _finite_float(state.get("structural_invalidation"))
        if broken is None or post_fill.empty:
            reclaim_condition = pd.Series(False, index=post_fill.index, dtype="bool")
        else:
            reclaim_condition = close.gt(broken * (1.0 + LEVEL_BAND))
        opposite_event = _bool(post_fill, "ms_choch_to_bull") | _bool(
            post_fill, "ms_bos_to_bull"
        )
        opposite_state = _num(post_fill, "ms_state").gt(0.0)
        vp_bull = (
            _num(post_fill, "s2vp_context_score_bull").gt(
                _num(post_fill, "s2vp_context_score_bear")
            )
            & _num(post_fill, "s2vp_score_long").gt(
                _num(post_fill, "s2vp_score_short")
            )
            & _num(post_fill, "s2vp_market_context").gt(0.0)
        )
        reclaim_1 = self._last_n(reclaim_condition, 1)
        reclaim_2 = self._last_n(reclaim_condition, 2)
        opposite_1 = self._last_n(opposite_event, 1)
        opposite_state_2 = self._last_n(opposite_state, 2)
        vp_1 = self._last_n(vp_bull, 1)
        vp_2 = self._last_n(vp_bull, 2)
        combined_1 = reclaim_1 and (opposite_1 or vp_1)
        structural_failure = bool(
            invalidation is not None
            and not post_fill.empty
            and float(close.iloc[-1]) > invalidation * (1.0 + LEVEL_BAND)
        )

        favorable_low = _finite_float(_num(post_fill, "low").min())
        previous_low = _finite_float(state.get("closed_favorable_low"))
        if favorable_low is not None:
            state["closed_favorable_low"] = (
                min(previous_low, favorable_low) if previous_low is not None else favorable_low
            )
        age_candles = len(post_fill)
        favorable = (
            1.0 - favorable_low / entry_rate
            if favorable_low is not None and entry_rate > 0.0
            else 0.0
        )
        time_failure = bool(
            plan.progress_candles
            and age_candles >= plan.progress_candles
            and favorable < 0.01
        )
        max_hold = bool(
            plan.max_hold_candles and age_candles >= plan.max_hold_candles
        )
        return {
            "hard_invalidation": (
                False
                if plan.role == "baseline"
                else structural_failure or (reclaim_2 and (opposite_1 or vp_1))
            ),
            "invalidation": self._invalidation_event(
                plan.invalidation,
                reclaim_1,
                reclaim_2,
                opposite_1,
                opposite_state_2,
                vp_2,
                combined_1,
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
        if state.get("phase") == "TERMINAL" or state.get("partial_pending"):
            return ExitDecision("hold")
        if state.get("terminal_exit_pending"):
            return ExitDecision(
                "full", str(state.get("terminal_exit_tag") or "s3v2_hard_invalidation")
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
            return ExitDecision("partial", "s3v2_partial_target", plan.partial_fraction)
        if (
            plan.invalidation_action == "partial"
            and events.get("invalidation")
            and not state.get("partial_filled")
        ):
            if events.get("profit_bucket") == "loss":
                return ExitDecision("full", "s3v2_invalidation_loss")
            return ExitDecision(
                "partial", "s3v2_partial_invalidation", plan.partial_fraction
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
        current_profit: float,
    ) -> tuple[ExitPlan, dict[str, Any], dict[str, Any], ExitDecision] | None:
        state = self._state(trade)
        if state is None or state.get("plan") not in EXIT_PLANS:
            return None
        legacy_terminal_exit = "terminal_exit_tag" not in state
        if legacy_terminal_exit:
            state["terminal_exit_tag"] = (
                "s3v2_hard_invalidation"
                if state.get("terminal_exit_pending")
                else None
            )
        previous = dict(state)
        frame = self._analyzed_frame(pair, current_time)
        partial_pending = self._has_pending_partial(trade)
        events, state = self._events(state, frame, current_profit)
        state["partial_pending"] = partial_pending
        plan = EXIT_PLANS[state["plan"]]
        if partial_pending:
            state["phase"] = "REALIZATION_PENDING"
            policy_state = dict(state)
            policy_state["partial_pending"] = False
            deferred_decision = self.evaluate_policy(plan, policy_state, events)
            if deferred_decision.action == "full":
                state["terminal_exit_pending"] = True
                if not state.get("terminal_exit_tag"):
                    state["terminal_exit_tag"] = deferred_decision.tag
        if legacy_terminal_exit or state != previous:
            self._save_state(trade, state)
        return plan, state, events, self.evaluate_policy(plan, state, events)

    @staticmethod
    def _desired_stop_price(
        plan: ExitPlan,
        state: dict[str, Any],
        decision: ExitDecision,
        current_rate: float,
    ) -> float:
        entry_rate = float(state.get("entry_rate") or current_rate)
        stop_price = entry_rate * (1.0 + plan.hard_stop)
        invalidation = _finite_float(state.get("structural_invalidation"))
        if plan.role != "baseline" and invalidation is not None and invalidation > entry_rate:
            stop_price = min(stop_price, invalidation * (1.0 + LEVEL_BAND))
        if state.get("partial_filled"):
            if plan.remainder == "breakeven":
                stop_price = min(stop_price, entry_rate * 0.999)
            elif plan.remainder == "trail":
                favorable_low = _finite_float(state.get("closed_favorable_low"))
                if favorable_low is not None:
                    stop_price = min(stop_price, favorable_low * 1.015)
        if decision.action == "tighten" and decision.tighten == "target":
            target = _finite_float(state.get("targets", {}).get(plan.target_1))
            if target is not None:
                stop_price = min(stop_price, target * 1.01)
        if decision.action == "tighten" and decision.tighten == "invalidation":
            broken = _finite_float(state.get("broken_support"))
            if broken is not None:
                stop_price = min(stop_price, broken * (1.0 + LEVEL_BAND))
            favorable_low = _finite_float(state.get("closed_favorable_low"))
            if favorable_low is not None and favorable_low < entry_rate:
                stop_price = min(stop_price, entry_rate * 0.999)
        return stop_price

    @staticmethod
    def _effective_stop_price(
        plan: ExitPlan,
        state: dict[str, Any],
        decision: ExitDecision,
        current_rate: float,
    ) -> float:
        stop_price = Sieve3V2IntegratedFromBosBearContinuationShort1H._desired_stop_price(
            plan, state, decision, current_rate
        )
        persisted = _finite_float(state.get("stop_ceiling"))
        return min(stop_price, persisted) if persisted is not None else stop_price

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
        context = self._decision_context(pair, trade, current_time, current_profit)
        if context is None:
            return None
        plan, state, _, decision = context
        if state.get("partial_pending") or state.get("phase") == "TERMINAL":
            return None
        if decision.action == "full":
            return decision.tag
        desired_stop = self._effective_stop_price(plan, state, decision, current_rate)
        return "s3v2_stop_ceiling_breached" if desired_stop <= current_rate else None

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
            min_stake,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            kwargs,
        )
        context = self._decision_context(
            str(getattr(trade, "pair", "")), trade, current_time, current_profit
        )
        if context is None or self._has_open_order(trade):
            return None
        _, state, _, decision = context
        if (
            state.get("phase") == "TERMINAL"
            or decision.action != "partial"
            or decision.fraction <= 0.0
        ):
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
        request_stake = min(remaining, stake)
        if request_stake <= 0.0:
            return None
        self._save_state(trade, state)
        return -request_stake, tag

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
        context = self._decision_context(pair, trade, current_time, current_profit)
        if context is not None:
            plan, state, _, decision = context
            if (
                state.get("partial_pending")
                or state.get("phase") == "TERMINAL"
                or decision.action == "full"
            ):
                return None
            selected_plan = plan
        working_state = state or {
            "entry_rate": float(getattr(trade, "open_rate", current_rate) or current_rate)
        }
        stop_price = self._effective_stop_price(
            selected_plan, working_state, decision, current_rate
        )
        if stop_price <= current_rate:
            return None
        persisted = _finite_float(working_state.get("stop_ceiling"))
        if state is not None and (persisted is None or stop_price < persisted):
            state["stop_ceiling"] = stop_price
            self._save_state(trade, state)
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=True,
            leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
        )
