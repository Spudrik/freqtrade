"""
STATUS: DRAFT / ITERATION 10

This file is an evolving scaffold for a Freqtrade hybrid ladder strategy.

It is NOT a stable reference.
It is NOT guaranteed to reflect the latest design decisions.
It may intentionally diverge from earlier or later versions.
Deviation is acceptable when justified by better logic, better testing, or cleaner architecture.

Core design of this draft
-------------------------
- One aggregate Freqtrade trade per pair.
- Nine virtual levels managed in persistent trade custom-data.
- Closed-candle logic only.
- One action per candle per open trade.
- L2-L6 are anchor-spaced using ATR percent-of-price.
- L7-L9 are drawdown-band entries measured from current average position price.
- Deep drawdown qualification uses candle low, while the final entry decision still
  happens on the closed candle after recovery / stabilization checks pass.
- Block 1 (L1-L3) uses small seed inventory and does not peel.
- Blocks 2-3 use simple level-local peel / rebuy behaviour.
- Hyperopt surface is intentionally broad and organized into spaces so tuning can be
  done in batches without changing architecture.
- Event and CUSUM-style features are precomputed as normal dataframe columns so they can
  be wired into later guards / triggers without adding callback-specific state machines.
- Regime multipliers can adapt shallow spacing and exit targets without changing the core ladder logic.

Maintenance rule
----------------
If the current design would now be described differently, this file is outdated and should be updated.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

try:
    from user_data.Indicators.market_regime import (
        add_regime_input_columns,
        add_regime_response_columns,
        add_regime_score_columns,
        add_regime_state_columns,
    )
except ImportError:
    from user_data.Indicators.market_regime import (
        add_regime_input_columns,
        add_regime_response_columns,
        add_regime_score_columns,
        add_regime_state_columns,
    )

logger = logging.getLogger(__name__)

HYPEROPT_GROUP_ENV = "HYBRID_RECOVERY_HYPEROPT_GROUPS"
HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"


def _split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def _resolve_hyperopt_group_params(strategy_cls: type, group_tokens: set[str]) -> set[str]:
    """
    Resolve grouped HyperOpt targets directly from strategy `batch_tags`.

    Supported tokens:
    - `all` to enable every optimize-capable parameter.
    - `family_name` shorthand, matched as `family:family_name`.
    - Full tag tokens like `family:capital_shape` or `mode:entry`.
    """
    if not group_tokens:
        return set()

    optimize_params: set[str] = set()
    tag_index: dict[str, set[str]] = {}
    for name, value in vars(strategy_cls).items():
        if not hasattr(value, "optimize"):
            continue
        param_name = str(name)
        optimize_params.add(param_name)
        for tag in getattr(value, "batch_tags", ()) or ():
            tag_key = str(tag).strip()
            if tag_key:
                tag_index.setdefault(tag_key, set()).add(param_name)

    normalized_tokens = {str(token).strip() for token in group_tokens if str(token).strip()}
    if "all" in normalized_tokens:
        return set(optimize_params)

    selected: set[str] = set()
    unknown: set[str] = set()
    for token in sorted(normalized_tokens):
        direct = tag_index.get(token, set())
        family = tag_index.get(f"family:{token}", set()) if ":" not in token else set()
        matched = set(direct) | set(family)
        if matched:
            selected.update(matched)
        else:
            unknown.add(token)

    if unknown:
        logger.warning("Unknown hyperopt group token(s): %s", ", ".join(sorted(unknown)))
    return selected


def _apply_grouped_hyperopt_surface(strategy_cls: type) -> None:
    """
    Enable a narrow hyperopt surface when batch-run environment variables are set.

    Normal GUI/manual usage is unchanged. During batched runs, pass one or both:
    - HYBRID_RECOVERY_HYPEROPT_GROUPS=capital_shape or family:capital_shape
    - HYBRID_RECOVERY_HYPEROPT_PARAMS=gap_mult_l5,exit_target_b3
    """
    group_names = _split_hyperopt_tokens(os.environ.get(HYPEROPT_GROUP_ENV))
    explicit_params = _split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    optimize_params = {str(name) for name, value in vars(strategy_cls).items() if hasattr(value, "optimize")}
    selected_params = _resolve_hyperopt_group_params(strategy_cls, group_names) | explicit_params
    if not selected_params:
        return

    unknown_explicit = sorted(param for param in explicit_params if param not in optimize_params)
    if unknown_explicit:
        logger.warning("Unknown explicit hyperopt parameter(s): %s", ", ".join(unknown_explicit))

    touched = 0
    for name, value in vars(strategy_cls).items():
        if hasattr(value, "optimize"):
            value.optimize = name in selected_params
            touched += 1
    logger.info(
        "Grouped hyperopt surface active: groups=%s params=%s enabled=%s total_parameters=%s",
        ",".join(sorted(group_names)) or "-",
        ",".join(sorted(explicit_params)) or "-",
        ",".join(sorted(selected_params)),
        touched,
    )


def tagged_parameter(param: Any, *tags: str) -> Any:
    """
    Attach stable multi-label metadata directly to a Freqtrade hyperopt parameter.

    Purpose
    -------
    - Keep grouping / batch-selection metadata inside the strategy definition so it
      cannot drift away into a sidecar file.
    - Allow one parameter to participate in several orthogonal concepts at once,
      for example family, role, signal domain, regime relevance and action stage.
    - Leave normal Freqtrade behaviour unchanged because the framework only cares
      about the parameter object itself and its `space` / `optimize` properties.
    """
    setattr(param, "batch_tags", tuple(str(tag) for tag in tags if str(tag)))
    return param


def collect_tagged_parameters(strategy_cls: type) -> dict[str, tuple[str, ...]]:
    """Return a name -> tags index for any parameter that exposes batch_tags."""
    tag_map: dict[str, tuple[str, ...]] = {}
    for name, value in vars(strategy_cls).items():
        tags = getattr(value, "batch_tags", None)
        if tags:
            tag_map[str(name)] = tuple(str(tag) for tag in tags)
    return tag_map

from freqtrade.persistence import Order, Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_absolute,
)


class HybridRecoveryGridStrategy(IStrategy):
    """
    Simplified level-driven hybrid ladder scaffold.

    Mental model
    ------------
    - One real Freqtrade trade contains multiple internal virtual levels.
    - L2-L6 are armed from fixed anchor spacing below the previous filled level.
    - L7-L9 are armed from drawdown bands relative to the current average position
      price, with candle-low qualification and closed-candle recovery checks.
    - Once a level is opened, that level can peel on strength and rebuy on a later
      revisit of its own anchor.
    - Event and CUSUM-style features are generated as standard dataframe columns.
      The base strategy does not force them yet; they are intended for later
      hyperoptable guards / triggers added in the usual Freqtrade style.

    Design intent
    -------------
    Keep the code level-centric and readable.
    Avoid abstract phase machines unless they are proven necessary later.
    Keep state small enough to remain robust in persistent trade custom-data.
    Expose broad parameter surfaces for staged hyperopt instead of hard-coding early
    assumptions into the strategy.
    """

    INTERFACE_VERSION = 3

    # ------------------------------------------------------------------
    # Framework settings
    # ------------------------------------------------------------------

    can_short = False
    timeframe = "1h"
    startup_candle_count = 240
    process_only_new_candles = True

    # custom_exit() is used for full exits, so exit signals must stay enabled.
    use_exit_signal = True
    use_custom_stoploss = True
    use_custom_roi = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.70

    position_adjustment_enable = True
    max_entry_position_adjustment = 8  # 9 virtual levels total = 1 seed + 8 adds.

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    LEVEL_COUNT = 9
    LEVEL_TO_BLOCK = (1, 1, 1, 2, 2, 2, 3, 3, 3)
    MIN_STAKE_BUFFER_MULT = 1.01
    BASE_OPEN_TRADE_CAP = 2
    OPEN_TRADE_CAP_UNLOCK_LEVEL_1 = 7

    ATR_WINDOW_CHOICES = [14, 24, 36, 48, 72, 96]
    ATR_EMA_SPAN_CHOICES = [10, 20, 40]
    VOLUME_EMA_SPAN_CHOICES = [10, 20, 40]
    SUPPORT_LOOKBACK_CHOICES = [2, 3, 5, 8]
    REGIME_CONFIRM_CHOICES = [1, 2, 3, 4, 5]
    RSI_SEED_MAX_CHOICES = [30, 40, 50, 60, 70, 80, 100]
    PEEL_PROFIT_CHOICES = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
    EXIT_TARGET_B1_CHOICES = [0.005, 0.010, 0.015, 0.020]
    EXIT_TARGET_B2_CHOICES = [0.015, 0.020, 0.025, 0.030]
    EXIT_TARGET_B3_CHOICES = [0.020, 0.040, 0.060, 0.080]
    GAP_MULT_L4_CHOICES = [0.4, 0.6, 0.8, 1.0]
    GAP_MULT_L5_CHOICES = [0.4, 0.6, 0.8, 1.0]
    GAP_MULT_L6_CHOICES = [0.5, 0.8, 1.0, 1.3]
    DEEP_DRAWDOWN_L7_CHOICES = [0.01, 0.03, 0.05, 0.08]
    DEEP_DRAWDOWN_STEP_L8_L9_CHOICES = [0.005, 0.01]
    BTC_1D_RSI_BULL_MIN_CHOICES = [45, 50, 55, 60, 65, 70]
    BTC_1D_RSI_BEAR_MAX_CHOICES = [25, 30, 35, 40, 45, 50, 55]
    DEEP_PREV_LEVEL_BUFFER_PCT = 0.01
    MIN_GAP_PCT = 0.005
    FULL_EXIT_MIN_PCT = 0.005
    FULL_EXIT_CAP_PCT = 0.20
    EVENT_CUSUM_LOOKBACK_CHOICES = [2, 3, 5, 8]
    EVENT_CUSUM_THRESHOLD_CHOICES = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

    # ------------------------------------------------------------------
    # Hyperopt: stake schedule / capital shape
    # Compatibility note: map broad custom groups into built-in Freqtrade spaces
    # so older installs can hyperopt them without rejecting custom names.
    # Space mapping used here: buy
    # ------------------------------------------------------------------

    fixed_wallet_pct = tagged_parameter(

        DecimalParameter(0.005, 0.050, decimals=3, default=0.04, space="buy", optimize=True, load=True),

        "family:capital_shape",

        "role:stake_budget",

        "signal:capital",

        "regime:all",

        "scope:global",

        "domain:capital",
        "action:position_sizing",
        "mode:capital_scaling",

    )
    leverage_target = tagged_parameter(
        IntParameter(1, 10, default=1, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:leverage",
        "signal:capital",
        "regime:all",
        "scope:global",
        "domain:capital",
        "action:position_sizing",
        "mode:capital_scaling",
    )
    leverage_mult_bull = tagged_parameter(
        DecimalParameter(0.80, 1.25, decimals=2, default=1.05, space="buy", optimize=True, load=True),
        "family:leverage_regime",
        "role:leverage_multiplier",
        "signal:capital",
        "regime:bull",
        "scope:global",
        "domain:capital",
        "action:position_sizing",
        "mode:regime_leverage_scaling",
    )
    leverage_mult_chop = tagged_parameter(
        DecimalParameter(0.55, 1.00, decimals=2, default=0.80, space="buy", optimize=True, load=True),
        "family:leverage_regime",
        "role:leverage_multiplier",
        "signal:capital",
        "regime:chop",
        "scope:global",
        "domain:capital",
        "action:position_sizing",
        "mode:regime_leverage_scaling",
    )
    leverage_mult_bear = tagged_parameter(
        DecimalParameter(0.35, 0.80, decimals=2, default=0.60, space="buy", optimize=True, load=True),
        "family:leverage_regime",
        "role:leverage_multiplier",
        "signal:capital",
        "regime:bear",
        "scope:global",
        "domain:capital",
        "action:position_sizing",
        "mode:regime_leverage_scaling",
    )
    leverage_seed_mult_breakout = tagged_parameter(
        DecimalParameter(1.00, 1.45, decimals=2, default=1.20, space="buy", optimize=True, load=True),
        "family:leverage_seed_response",
        "role:leverage_multiplier",
        "signal:capital",
        "mode:breakout_seed",
        "scope:global",
        "domain:capital",
        "action:seed_entry",
        "mode:seed_leverage_scaling",
    )
    leverage_seed_mult_support = tagged_parameter(
        DecimalParameter(0.65, 1.00, decimals=2, default=0.85, space="buy", optimize=True, load=True),
        "family:leverage_seed_response",
        "role:leverage_multiplier",
        "signal:capital",
        "mode:support_reversal_seed",
        "mode:support_bounce_seed",
        "scope:global",
        "domain:capital",
        "action:seed_entry",
        "mode:seed_leverage_scaling",
    )
    leverage_seed_mult_crash = tagged_parameter(
        DecimalParameter(0.40, 0.85, decimals=2, default=0.65, space="buy", optimize=True, load=True),
        "family:leverage_seed_response",
        "role:leverage_multiplier",
        "signal:capital",
        "mode:crash_recovery_seed",
        "scope:global",
        "domain:capital",
        "action:seed_entry",
        "mode:seed_leverage_scaling",
    )
    leverage_vol_brake_trigger = tagged_parameter(
        DecimalParameter(1.35, 2.00, decimals=2, default=1.60, space="buy", optimize=True, load=True),
        "family:leverage_risk_overlay",
        "role:volatility_threshold",
        "signal:volatility",
        "regime:all",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:volatility_brake",
    )
    leverage_vol_brake_mult = tagged_parameter(
        DecimalParameter(0.45, 0.90, decimals=2, default=0.70, space="buy", optimize=True, load=True),
        "family:leverage_risk_overlay",
        "role:leverage_multiplier",
        "signal:volatility",
        "regime:all",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:volatility_brake",
    )
    leverage_cap_bull_breakout = tagged_parameter(
        IntParameter(1, 10, default=8, space="buy", optimize=True, load=True),
        "family:leverage_caps",
        "role:leverage_cap",
        "signal:risk",
        "regime:bull",
        "mode:breakout_seed",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:leverage_cap",
    )
    leverage_cap_bull = tagged_parameter(
        IntParameter(1, 10, default=6, space="buy", optimize=True, load=True),
        "family:leverage_caps",
        "role:leverage_cap",
        "signal:risk",
        "regime:bull",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:leverage_cap",
    )
    leverage_cap_chop = tagged_parameter(
        IntParameter(1, 8, default=4, space="buy", optimize=True, load=True),
        "family:leverage_caps",
        "role:leverage_cap",
        "signal:risk",
        "regime:chop",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:leverage_cap",
    )
    leverage_cap_bear = tagged_parameter(
        IntParameter(1, 6, default=3, space="buy", optimize=True, load=True),
        "family:leverage_caps",
        "role:leverage_cap",
        "signal:risk",
        "regime:bear",
        "scope:global",
        "domain:risk",
        "action:position_sizing",
        "mode:leverage_cap",
    )
    l5_mult = tagged_parameter(
        DecimalParameter(1.00, 2.0, decimals=1, default=2.0, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:bear_or_mixed",
        "scope:level_L5",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l6_mult = tagged_parameter(
        DecimalParameter(1.00, 2.0, decimals=1, default=2.2, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:bear_or_mixed",
        "scope:level_L6",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l7_mult = tagged_parameter(
        DecimalParameter(1.00, 2.0, decimals=1, default=2.0, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:bear_or_mixed",
        "scope:level_L7",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l8_mult = tagged_parameter(
        DecimalParameter(1.00, 2.0, decimals=1, default=1.7, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:bear_or_mixed",
        "scope:level_L8",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l9_mult = tagged_parameter(
        DecimalParameter(1.00, 2.0, decimals=1, default=1.7, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:bear_or_mixed",
        "scope:level_L9",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l1_mult = tagged_parameter(
        DecimalParameter(1.0, 3.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:all",
        "scope:level_L1",
        "domain:capital",
        "action:position_sizing",
        "mode:capital_scaling",
    )
    l2_mult = tagged_parameter(
        DecimalParameter(1.0, 3.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:all",
        "scope:level_L2",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l3_mult = tagged_parameter(
        DecimalParameter(1.0, 3.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "role:stake_multiplier",
        "signal:capital",
        "regime:all",
        "scope:level_L3",
        "domain:capital",
        "action:add_entry",
        "mode:capital_scaling",
    )
    l9_stoploss_pct = tagged_parameter(
        DecimalParameter(0.03, 0.30, decimals=2, default=0.10, space="buy", optimize=True, load=True),
        "family:capital_shape",
        "family:deep_recovery",
        "role:catastrophic_stop",
        "signal:risk",
        "regime:bear_or_mixed",
        "scope:level_L9",
        "domain:risk",
        "action:stoploss_exit",
        "mode:deep_drawdown_recovery",
    )
    atr_window = tagged_parameter(
        CategoricalParameter(ATR_WINDOW_CHOICES, default=96, space="buy", optimize=True, load=True),
        "family:core_indicators",
        "role:lookback",
        "signal:volatility",
        "regime:all",
        "scope:global",
        "action:indicator_foundation",
    )
    atr_ema_span = tagged_parameter(
        CategoricalParameter(ATR_EMA_SPAN_CHOICES, default=40, space="buy", optimize=True, load=True),
        "family:core_indicators",
        "role:smoothing",
        "signal:volatility",
        "regime:all",
        "scope:global",
        "action:indicator_foundation",
    )
    volume_ema_span = tagged_parameter(
        CategoricalParameter(VOLUME_EMA_SPAN_CHOICES, default=10, space="buy", optimize=True, load=True),
        "family:core_indicators",
        "role:smoothing",
        "signal:volume",
        "regime:all",
        "scope:global",
        "action:indicator_foundation",
    )
    support_lookback = tagged_parameter(
        CategoricalParameter(SUPPORT_LOOKBACK_CHOICES, default=8, space="buy", optimize=True, load=True),
        "family:core_indicators",
        "role:lookback",
        "signal:price_structure",
        "regime:all",
        "scope:global",
        "action:level_detection",
    )
    # ------------------------------------------------------------------
    # Hyperopt: regime adaptation
    # Compatibility note: grouped into built-in buy/sell spaces for older Freqtrade.
    # Space mapping used here: buy / sell
    # ------------------------------------------------------------------

    regime_adx_trend_min = tagged_parameter(

        DecimalParameter(10.0, 50.0, decimals=1, default=43.6, space="buy", optimize=True, load=True),

        "family:regime_detector",

        "role:trend_threshold",

        "signal:trend_strength",

        "regime:all",

        "scope:global",

        "domain:regime",
        "action:classify_regime",
        "mode:btc_regime_context",

    )
    regime_bear_atr_spike = tagged_parameter(
        DecimalParameter(1.00, 2.50, decimals=1, default=1.2, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "role:volatility_threshold",
        "signal:volatility",
        "regime:bear",
        "scope:global",
        "domain:regime",
        "action:classify_regime",
        "mode:btc_regime_context",
    )
    regime_confirm_bars = tagged_parameter(
        CategoricalParameter(REGIME_CONFIRM_CHOICES, default=2, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "role:hysteresis",
        "signal:regime_persistence",
        "regime:all",
        "scope:global",
        "domain:regime",
        "action:classify_regime",
        "mode:btc_regime_context",
    )
    # gap mult: scales distance between level anchors (smaller=tighter grid, larger=wider grid).
    regime_gap_mult_bull = tagged_parameter(
        DecimalParameter(0.50, 2.50, decimals=1, default=0.5, space="buy", optimize=True, load=True),
        "family:regime_response_bull_chop",
        "role:spacing_multiplier",
        "signal:regime_response",
        "regime:bull",
        "scope:ladder_spacing",
        "domain:add",
        "domain:regime",
        "action:add_entry",
        "mode:regime_spacing_response",
    )
    regime_gap_mult_chop = tagged_parameter(
        DecimalParameter(0.50, 2.50, decimals=1, default=0.5, space="buy", optimize=True, load=True),
        "family:regime_response_bull_chop",
        "role:spacing_multiplier",
        "signal:regime_response",
        "regime:chop",
        "scope:ladder_spacing",
        "domain:add",
        "domain:regime",
        "action:add_entry",
        "mode:regime_spacing_response",
    )
    regime_gap_mult_bear = tagged_parameter(
        DecimalParameter(0.50, 3.50, decimals=1, default=0.8, space="buy", optimize=True, load=True),
        "family:regime_response_bear",
        "role:spacing_multiplier",
        "signal:regime_response",
        "regime:bear",
        "scope:ladder_spacing",
        "domain:add",
        "domain:regime",
        "action:add_entry",
        "mode:regime_spacing_response",
    )
    # exit mult: scales full-trade profit target by regime.
    regime_exit_mult_bull = tagged_parameter(
        DecimalParameter(0.50, 2.00, decimals=1, default=1.5, space="sell", optimize=True, load=True),
        "family:regime_response_bull_chop",
        "family:exit_peel",
        "role:exit_target_multiplier",
        "signal:regime_response",
        "regime:bull",
        "scope:full_exit",
        "domain:exit",
        "domain:regime",
        "action:full_exit",
        "mode:regime_exit_response",
    )
    regime_exit_mult_chop = tagged_parameter(
        DecimalParameter(0.50, 2.00, decimals=1, default=1.6, space="sell", optimize=True, load=True),
        "family:regime_response_bull_chop",
        "family:exit_peel",
        "role:exit_target_multiplier",
        "signal:regime_response",
        "regime:chop",
        "scope:full_exit",
        "domain:exit",
        "domain:regime",
        "action:full_exit",
        "mode:regime_exit_response",
    )
    regime_exit_mult_bear = tagged_parameter(
        DecimalParameter(0.50, 2.00, decimals=1, default=0.9, space="sell", optimize=True, load=True),
        "family:regime_response_bear",
        "family:exit_peel",
        "role:exit_target_multiplier",
        "signal:regime_response",
        "regime:bear",
        "scope:full_exit",
        "domain:exit",
        "domain:regime",
        "action:full_exit",
        "mode:regime_exit_response",
    )
    use_btc_1d_rsi_regime = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "family:feature_enables",
        "switch:enable",
        "switch:regime_enable",
        "controls:btc_1d_rsi_regime",
        "domain:regime",
        "role:informative_toggle",
        "signal:btc_informative",
        "regime:all",
        "scope:global",
        "action:classify_regime",
        "mode:btc_regime_context",
    )
    enable_pair_1d_context = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "family:feature_enables",
        "switch:enable",
        "switch:regime_enable",
        "controls:pair_1d_context",
        "domain:regime",
        "role:informative_toggle",
        "signal:pair_informative",
        "regime:all",
        "scope:global",
        "action:classify_regime",
        "mode:pair_regime_context",
    )
    btc_1d_rsi_bull_min = tagged_parameter(
        CategoricalParameter(BTC_1D_RSI_BULL_MIN_CHOICES, default=55, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "role:threshold",
        "signal:btc_informative",
        "regime:bull",
        "scope:global",
        "domain:regime",
        "action:classify_regime",
        "mode:btc_regime_context",
    )
    btc_1d_rsi_bear_max = tagged_parameter(
        CategoricalParameter(BTC_1D_RSI_BEAR_MAX_CHOICES, default=45, space="buy", optimize=True, load=True),
        "family:regime_detector",
        "role:threshold",
        "signal:btc_informative",
        "regime:bear",
        "scope:global",
        "domain:regime",
        "action:classify_regime",
        "mode:btc_regime_context",
    )
    # ------------------------------------------------------------------
    # Hyperopt: structural spacing / seed / add filters
    # Space: entry
    # ------------------------------------------------------------------

    vol_unit_floor_pct = tagged_parameter(

        DecimalParameter(0.001, 0.030, decimals=2, default=0.02, space="buy", optimize=True, load=True),

        "family:core_indicators",

        "family:shallow_spacing",

        "role:volatility_floor",

        "signal:volatility",

        "regime:all",

        "scope:ladder_spacing",

        "domain:add",

        "action:add_entry",

        "mode:ladder_spacing",

    )
    vol_unit_cap_pct = tagged_parameter(
        DecimalParameter(0.020, 0.300, decimals=2, default=0.02, space="buy", optimize=True, load=True),
        "family:core_indicators",
        "family:shallow_spacing",
        "role:volatility_cap",
        "signal:volatility",
        "regime:all",
        "scope:ladder_spacing",
        "domain:add",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    gap_mult_l2 = tagged_parameter(
        DecimalParameter(0.20, 2.50, decimals=1, default=2.1, space="buy", optimize=True, load=True),
        "family:shallow_spacing",
        "role:spacing_multiplier",
        "signal:price_structure",
        "regime:bull_or_chop",
        "scope:level_L2",
        "domain:add",
        "domain:price_action",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    gap_mult_l3 = tagged_parameter(
        DecimalParameter(0.20, 3.00, decimals=1, default=0.5, space="buy", optimize=True, load=True),
        "family:shallow_spacing",
        "role:spacing_multiplier",
        "signal:price_structure",
        "regime:bull_or_chop",
        "scope:level_L3",
        "domain:add",
        "domain:price_action",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    gap_mult_l4 = tagged_parameter(
        CategoricalParameter(GAP_MULT_L4_CHOICES, default=0.8, space="buy", optimize=True, load=True),
        "family:shallow_spacing",
        "role:spacing_multiplier",
        "signal:price_structure",
        "regime:bull_or_chop",
        "scope:level_L4",
        "domain:add",
        "domain:price_action",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    gap_mult_l5 = tagged_parameter(
        CategoricalParameter(GAP_MULT_L5_CHOICES, default=1.0, space="buy", optimize=True, load=True),
        "family:shallow_spacing",
        "family:capital_shape",
        "role:spacing_multiplier",
        "signal:price_structure",
        "regime:all",
        "scope:level_L5",
        "domain:add",
        "domain:price_action",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    gap_mult_l6 = tagged_parameter(
        CategoricalParameter(GAP_MULT_L6_CHOICES, default=1.3, space="buy", optimize=True, load=True),
        "family:shallow_spacing",
        "family:capital_shape",
        "role:spacing_multiplier",
        "signal:price_structure",
        "regime:all",
        "scope:level_L6",
        "domain:add",
        "domain:price_action",
        "action:add_entry",
        "mode:ladder_spacing",
    )
    deep_drawdown_l7_pct = tagged_parameter(
        CategoricalParameter(DEEP_DRAWDOWN_L7_CHOICES, default=0.12, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:drawdown_threshold",
        "signal:drawdown",
        "regime:bear_or_mixed",
        "scope:level_L7",
        "domain:rebuy",
        "domain:risk",
        "action:rebuy_entry",
        "mode:deep_drawdown_recovery",
    )
    deep_drawdown_step_l8_l9_pct = tagged_parameter(
        CategoricalParameter(DEEP_DRAWDOWN_STEP_L8_L9_CHOICES, default=0.04, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:drawdown_step",
        "signal:drawdown",
        "regime:bear_or_mixed",
        "scope:levels_L8_L9",
        "domain:rebuy",
        "domain:risk",
        "action:rebuy_entry",
        "mode:deep_drawdown_recovery",
    )
    support_reversal_1h_rsi_max = tagged_parameter(
        CategoricalParameter(RSI_SEED_MAX_CHOICES, default=60, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "family:entry_filters", "role:threshold", "signal:momentum", "signal:reversal", "signal:support", "timeframe:1h",
        "domain:entry", "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_rsi_max = tagged_parameter(
        CategoricalParameter(RSI_SEED_MAX_CHOICES, default=70, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "family:entry_filters", "role:threshold", "signal:momentum", "signal:support", "timeframe:1h",
        "domain:entry", "action:seed_entry", "mode:support_bounce_seed",
    )
    crash_recovery_rsi_max = tagged_parameter(
        CategoricalParameter(RSI_SEED_MAX_CHOICES, default=80, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "family:entry_filters", "role:threshold", "signal:momentum", "signal:risk", "signal:crash_guard",
        "domain:entry", "action:seed_entry", "mode:crash_recovery_seed",
    )
    support_reversal_1h_bb_lower_buffer = tagged_parameter(
        DecimalParameter(0.990, 1.030, decimals=3, default=1.018, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "family:entry_filters", "role:buffer_threshold", "signal:support", "signal:reversal", "signal:price_structure", "timeframe:1h",
        "domain:entry", "domain:price_action", "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_bb_lower_buffer = tagged_parameter(
        DecimalParameter(0.990, 1.030, decimals=3, default=1.018, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "family:entry_filters", "role:buffer_threshold", "signal:support", "signal:price_structure", "timeframe:1h",
        "domain:entry", "domain:price_action", "action:seed_entry", "mode:support_bounce_seed",
    )
    support_reversal_1h_use_below_ema_fast = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:price_action_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:trend_location",
        "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_use_below_ema_fast = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:price_action_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:trend_location",
        "action:seed_entry", "mode:support_bounce_seed",
    )
    support_reversal_1h_use_bb_lower = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:price_action_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:support", "signal:reversal", "signal:price_structure",
        "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_use_bb_lower = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:price_action_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:support", "signal:price_structure",
        "action:seed_entry", "mode:support_bounce_seed",
    )
    support_reversal_1h_require_stabilizing = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:price_confirmation", "signal:reversal",
        "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_require_stabilizing = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "timeframe:1h",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:price_confirmation", "signal:support",
        "action:seed_entry", "mode:support_bounce_seed",
    )
    crash_recovery_require_stabilizing = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable",
        "domain:entry", "domain:price_action", "role:gate_toggle", "signal:price_confirmation", "signal:crash_guard",
        "action:seed_entry", "mode:crash_recovery_seed",
    )
    support_reversal_1h_use_volume_confirm = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:volume_enable", "timeframe:1h",
        "domain:entry", "domain:volume", "role:gate_toggle", "signal:volume", "signal:reversal",
        "action:seed_entry", "mode:support_reversal_seed",
    )
    support_bounce_1h_use_volume_confirm = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:volume_enable", "timeframe:1h",
        "domain:entry", "domain:volume", "role:gate_toggle", "signal:volume", "signal:support",
        "action:seed_entry", "mode:support_bounce_seed",
    )
    crash_recovery_use_volume_confirm = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "family:entry_filters", "family:feature_enables", "switch:enable", "switch:entry_enable", "switch:volume_enable",
        "domain:entry", "domain:volume", "role:gate_toggle", "signal:volume", "signal:risk", "signal:crash_guard",
        "action:seed_entry", "mode:crash_recovery_seed",
    )
    breakout_1h_require_stabilizing = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:breakout_seed", "timeframe:1h", "domain:entry", "domain:price_action",
        "action:seed_entry", "role:gate_toggle", "signal:breakout", "signal:price_confirmation",
    )
    breakout_1h_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.5, 2.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1h", "domain:entry", "domain:volume", "action:seed_entry", "role:threshold", "signal:volume",
    )
    breakout_1h_use_macd_confirm = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:breakout_seed", "timeframe:1h", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:breakout", "signal:momentum",
    )
    breakout_1h_rsi_min = tagged_parameter(
        CategoricalParameter([35, 40, 45, 50, 55, 60], default=45, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1h", "domain:entry", "action:seed_entry", "role:threshold", "signal:momentum",
    )
    breakout_1h_confirm_bars = tagged_parameter(
        CategoricalParameter([1, 2, 3], default=2, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1h", "domain:entry", "action:seed_entry", "role:threshold", "signal:breakout",
    )
    breakout_1h_retest_buffer = tagged_parameter(
        DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1h", "domain:entry", "domain:price_action", "action:seed_entry", "role:threshold", "signal:price_structure",
    )
    breakout_1d_require_1h_trigger = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:breakout_seed", "timeframe:1d", "domain:entry", "action:seed_entry", "role:gate_toggle",
    )
    breakout_1d_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.5, 2.0, decimals=1, default=0.8, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1d", "domain:entry", "domain:volume", "action:seed_entry", "role:threshold", "signal:volume",
    )
    breakout_1d_use_btc_context = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:breakout_seed", "timeframe:1d", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    breakout_1d_confirm_bars = tagged_parameter(
        CategoricalParameter([1, 2, 3], default=2, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1d", "domain:entry", "action:seed_entry", "role:threshold", "signal:breakout",
    )
    breakout_1d_retest_buffer = tagged_parameter(
        DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:breakout_seed", "timeframe:1d", "domain:entry", "domain:price_action", "action:seed_entry", "role:threshold", "signal:price_structure",
    )
    support_reversal_1h_use_rsi_rising = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:reversal", "signal:momentum",
    )
    support_reversal_1h_use_macd_hist_rising = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:reversal", "signal:momentum",
    )
    support_reversal_1h_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.5, 2.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "domain:volume", "action:seed_entry", "role:threshold", "signal:volume",
    )
    support_reversal_1h_undercut_buffer = tagged_parameter(
        DecimalParameter(0.000, 0.030, decimals=3, default=0.005, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "domain:price_action", "action:seed_entry", "role:threshold",
        "signal:support", "signal:reversal", "signal:price_structure",
    )
    support_reversal_1h_reclaim_lookback = tagged_parameter(
        CategoricalParameter([2, 3, 4, 5], default=4, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "action:seed_entry", "role:lookback",
        "signal:support", "signal:reversal", "signal:price_structure",
    )
    support_reversal_1h_confirm_bars = tagged_parameter(
        CategoricalParameter([1, 2, 3], default=1, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_reversal_seed", "timeframe:1h", "domain:entry", "action:seed_entry", "role:threshold", "signal:reversal",
    )
    support_reversal_1d_require_1h_trigger = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_reversal_seed", "timeframe:1d", "domain:entry", "action:seed_entry", "role:gate_toggle",
    )
    support_reversal_1d_confirm_bars = tagged_parameter(
        CategoricalParameter([1, 2, 3], default=1, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_reversal_seed", "timeframe:1d", "domain:entry", "action:seed_entry", "role:threshold", "signal:reversal",
    )
    support_reversal_1d_use_btc_context = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:support_reversal_seed", "timeframe:1d", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    support_bounce_1h_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.5, 2.0, decimals=1, default=0.8, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_bounce_seed", "timeframe:1h", "domain:entry", "domain:volume", "action:seed_entry", "role:threshold", "signal:volume",
    )
    support_bounce_1h_support_distance_max = tagged_parameter(
        DecimalParameter(0.005, 0.050, decimals=3, default=0.020, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_bounce_seed", "timeframe:1h", "domain:entry", "domain:price_action", "action:seed_entry", "role:threshold",
        "signal:support", "signal:price_structure",
    )
    support_bounce_1h_no_undercut_lookback = tagged_parameter(
        CategoricalParameter([2, 3, 4, 5], default=4, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:support_bounce_seed", "timeframe:1h", "domain:entry", "action:seed_entry", "role:lookback", "signal:support",
    )
    support_bounce_1h_require_green_candle = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_bounce_seed", "timeframe:1h", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:price_confirmation",
    )
    support_bounce_1h_require_close_gt_prev = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_bounce_seed", "timeframe:1h", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:price_confirmation",
    )
    support_bounce_1d_require_1h_trigger = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:support_bounce_seed", "timeframe:1d", "domain:entry", "action:seed_entry", "role:gate_toggle",
    )
    support_bounce_1d_use_btc_context = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:support_bounce_seed", "timeframe:1d", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    crash_recovery_use_rsi_rising = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:crash_recovery_seed", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:risk", "signal:crash_guard", "signal:momentum",
    )
    crash_recovery_use_macd_hist_rising = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:crash_recovery_seed", "domain:entry", "action:seed_entry",
        "role:gate_toggle", "signal:risk", "signal:crash_guard", "signal:momentum",
    )
    crash_recovery_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.5, 3.0, decimals=1, default=1.0, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:crash_recovery_seed", "domain:entry", "domain:volume", "action:seed_entry", "role:threshold", "signal:volume",
    )
    crash_recovery_support_reclaim_required = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_quality_gates",
        "switch:enable", "switch:entry_enable", "mode:crash_recovery_seed", "domain:entry", "domain:price_action",
        "action:seed_entry", "role:gate_toggle", "signal:support", "signal:reversal", "signal:crash_guard",
    )
    crash_recovery_confirm_bars = tagged_parameter(
        CategoricalParameter([1, 2, 3], default=1, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_tuning_all", "family:entry_seed_thresholds",
        "mode:crash_recovery_seed", "domain:entry", "action:seed_entry", "role:threshold", "signal:crash_guard",
    )
    breakout_block_on_hostile_btc_context = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:breakout_seed", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    support_reversal_block_on_hostile_btc_context = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:support_reversal_seed", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    support_bounce_block_on_hostile_btc_context = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:support_bounce_seed", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    crash_recovery_block_on_hostile_btc_context = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:crash_recovery_seed_tuning", "family:entry_seed_context_filters",
        "switch:enable", "switch:entry_enable", "mode:crash_recovery_seed", "domain:entry", "domain:context",
        "action:seed_entry", "role:gate_toggle", "signal:btc_context",
    )
    breakout_bear_require_retest = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:breakout_seed_tuning", "family:entry_seed_quality_gates", "family:entry_seed_context_filters",
        "switch:enable", "mode:breakout_seed", "regime:bear", "domain:entry", "domain:regime", "action:seed_entry",
        "role:gate_toggle", "signal:breakout", "signal:price_structure",
    )
    support_reversal_bear_require_confirmed_reclaim = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:support_reversal_seed_tuning", "family:entry_seed_quality_gates", "family:entry_seed_context_filters",
        "switch:enable", "mode:support_reversal_seed", "regime:bear", "domain:entry", "domain:regime", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:reversal", "signal:price_structure",
    )
    support_bounce_bear_require_stronger_support_hold = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:support_bounce_seed_tuning", "family:entry_seed_quality_gates", "family:entry_seed_context_filters",
        "switch:enable", "mode:support_bounce_seed", "regime:bear", "domain:entry", "domain:regime", "action:seed_entry",
        "role:gate_toggle", "signal:support", "signal:price_structure",
    )
    use_add_close_gt_prev = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:entry_filters",
        "family:feature_enables",
        "switch:enable",
        "switch:add_enable",
        "switch:price_action_enable",
        "controls:add_close_gt_prev",
        "domain:entry",
        "domain:price_action",
        "family:shallow_spacing",
        "role:gate_toggle",
        "signal:price_confirmation",
        "regime:bull_or_chop",
        "scope:add_entry",
        "action:add_entry",
        "mode:support_reversal_seed",
        "mode:support_bounce_seed",
    )
    enable_breakout_seed_1h = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:breakout_seed", "timeframe:1h",
        "signal:breakout", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_support_reversal_seed_1h = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:support_reversal_seed", "timeframe:1h",
        "signal:support", "signal:reversal", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_support_bounce_seed_1h = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:support_bounce_seed", "timeframe:1h",
        "signal:support", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_breakout_seed_1d = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:breakout_seed", "timeframe:1d",
        "signal:breakout", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_support_reversal_seed_1d = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:support_reversal_seed", "timeframe:1d",
        "signal:support", "signal:reversal", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_support_bounce_seed_1d = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:support_bounce_seed", "timeframe:1d",
        "signal:support", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_combo_1d_1h_breakout = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:breakout_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:combo_1d_1h_breakout", "timeframe:combo_1d_1h",
        "signal:breakout", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_combo_1d_1h_support_reversal = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_reversal_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:combo_1d_1h_support_reversal", "timeframe:combo_1d_1h",
        "signal:support", "signal:reversal", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_combo_1d_1h_support_bounce = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:support_bounce_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:combo_1d_1h_support_bounce", "timeframe:combo_1d_1h",
        "signal:support", "signal:price_structure", "signal:volume", "signal:btc_context", "signal:pair_context",
    )
    enable_crash_recovery_seed = tagged_parameter(
        BooleanParameter(default=True, space="buy", optimize=True, load=True),
        "family:crash_recovery_seed", "family:entry_seed_enables", "family:feature_enables", "family:entry_filters",
        "switch:enable", "switch:entry_enable",
        "domain:entry", "action:seed_entry", "role:gate_toggle",
        "mode:crash_recovery_seed", "timeframe:combo_1d_1h",
        "signal:risk", "signal:crash_guard", "signal:reversal", "signal:price_structure", "signal:volume",
    )
    support_bounce_1h_use_hold_support = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:entry_filters",
        "family:feature_enables",
        "switch:enable",
        "switch:entry_enable",
        "switch:price_action_enable",
        "controls:support_bounce_hold_support",
        "domain:entry",
        "domain:price_action",
        "role:gate_toggle",
        "signal:price_structure",
        "signal:support",
        "regime:bull_or_chop",
        "scope:seed_entry",
        "action:seed_entry",
        "mode:support_bounce_seed",
    )
    support_bounce_1h_use_local_min_bounce = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:entry_filters",
        "family:feature_enables",
        "switch:enable",
        "switch:entry_enable",
        "switch:price_action_enable",
        "controls:support_bounce_local_min_bounce",
        "domain:entry",
        "domain:price_action",
        "role:gate_toggle",
        "signal:price_structure",
        "signal:support",
        "regime:bull_or_chop",
        "scope:seed_entry",
        "action:seed_entry",
        "mode:support_bounce_seed",
    )
    breakout_1h_use_volume_confirm = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:entry_filters",
        "family:feature_enables",
        "switch:enable",
        "switch:entry_enable",
        "switch:volume_enable",
        "switch:price_action_enable",
        "controls:breakout_volume_confirm",
        "domain:entry",
        "domain:price_action",
        "domain:volume",
        "role:gate_toggle",
        "signal:price_structure",
        "signal:volume",
        "signal:breakout",
        "regime:bull_or_chop",
        "scope:seed_entry",
        "action:seed_entry",
        "mode:breakout_seed",
    )
    # ------------------------------------------------------------------
    # Portfolio risk release
    # Notes:
    # - This is intentionally not hyperopted.
    # - Default cap is 2 open trades.
    # - Free one extra slot for each open trade that reaches L7+.
    # - Bot-level max_open_trades must still be >= the highest cap you want.
    # ------------------------------------------------------------------

    _aggregate_regime_stats = {
        "bull": {"trades": 0, "wins": 0, "profit": 0.0, "deepest_level": 0},
        "bear": {"trades": 0, "wins": 0, "profit": 0.0, "deepest_level": 0},
        "chop": {"trades": 0, "wins": 0, "profit": 0.0, "deepest_level": 0},
    }


    # ------------------------------------------------------------------
    # Hyperopt: peel / rebuy / crash guard
    # Compatibility note: grouped into built-in buy space for older Freqtrade.
    # Space: buy
    # ------------------------------------------------------------------

    peel_profit_pct = tagged_parameter(

        CategoricalParameter(PEEL_PROFIT_CHOICES, default=0.020, space="buy", optimize=True, load=True),

        "family:exit_peel",

        "role:peel_target",

        "signal:profit_take",

        "regime:bull_or_chop",

        "scope:peelable_levels",

        "action:peel_exit",

    )
    # Shared peel size ratio for all peelable levels (L4-L9).
    # Actual peel stake is max(level_size * ratio, min_stake * 1.01), capped by live size.
    peel_fraction_shared = tagged_parameter(
        DecimalParameter(0.3, 0.8, decimals=1, default=0.3, space="buy", optimize=True, load=True),
        "family:exit_peel",
        "role:peel_size",
        "signal:position_reduction",
        "regime:bull_or_chop",
        "scope:peelable_levels",
        "action:peel_exit",
    )
    rebuy_crash_min_votes = tagged_parameter(
        IntParameter(1, 6, default=6, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:vote_threshold",
        "signal:crash_guard",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "domain:rebuy",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    rebuy_adx_crash_min = tagged_parameter(
        DecimalParameter(10.0, 50.0, decimals=1, default=15.6, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:threshold",
        "signal:trend_strength",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "domain:rebuy",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    rebuy_atr_spike_limit = tagged_parameter(
        DecimalParameter(0.80, 2.00, decimals=1, default=0.92, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:threshold",
        "signal:volatility",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "domain:rebuy",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    rebuy_volume_ratio_min = tagged_parameter(
        DecimalParameter(0.30, 2.00, decimals=1, default=1.48, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "role:threshold",
        "signal:volume",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "domain:rebuy",
        "domain:volume",
        "action:rebuy_entry",
        "mode:volume_confirmed_rebuy",
    )
    rebuy_use_volume_vote = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:feature_enables",
        "switch:enable",
        "switch:rebuy_enable",
        "switch:volume_enable",
        "controls:rebuy_volume_vote",
        "domain:rebuy",
        "domain:volume",
        "role:gate_toggle",
        "signal:volume",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "action:rebuy_entry",
        "mode:volume_confirmed_rebuy",
    )
    use_rebuy_crash_guard = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:feature_enables",
        "switch:enable",
        "switch:rebuy_enable",
        "switch:guard_enable",
        "switch:risk_enable",
        "controls:rebuy_crash_guard",
        "domain:rebuy",
        "domain:risk",
        "domain:regime",
        "role:gate_toggle",
        "signal:crash_guard",
        "signal:risk",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    enable_support_local_min_rebuy = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:feature_enables",
        "family:rebuy_enables",
        "switch:enable",
        "switch:rebuy_enable",
        "switch:price_action_enable",
        "controls:support_local_min_rebuy",
        "domain:rebuy",
        "domain:price_action",
        "role:gate_toggle",
        "signal:price_structure",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "action:rebuy_entry",
        "mode:support_local_min_rebuy",
    )
    enable_volume_confirmed_rebuy = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:feature_enables",
        "family:rebuy_enables",
        "switch:enable",
        "switch:rebuy_enable",
        "switch:volume_enable",
        "controls:volume_confirmed_rebuy",
        "domain:rebuy",
        "domain:volume",
        "role:gate_toggle",
        "signal:volume",
        "regime:bear_or_mixed",
        "scope:rebuy",
        "action:rebuy_entry",
        "mode:volume_confirmed_rebuy",
    )
    enable_resistance_local_max_peel = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:exit_peel",
        "family:feature_enables",
        "family:peel_enables",
        "switch:enable",
        "switch:peel_enable",
        "switch:price_action_enable",
        "controls:resistance_local_max_peel",
        "domain:peel",
        "domain:price_action",
        "role:gate_toggle",
        "signal:price_structure",
        "signal:profit_take",
        "regime:bull_or_chop",
        "scope:peelable_levels",
        "action:peel_exit",
        "mode:resistance_local_max_peel",
    )
    enable_breakout_hold_no_peel = tagged_parameter(
        BooleanParameter(default=False, space="buy", optimize=True, load=True),
        "family:exit_peel",
        "family:feature_enables",
        "family:peel_enables",
        "switch:enable",
        "switch:peel_enable",
        "switch:volume_enable",
        "switch:price_action_enable",
        "controls:breakout_hold_no_peel",
        "domain:peel",
        "domain:price_action",
        "domain:volume",
        "role:gate_toggle",
        "signal:breakout",
        "signal:volume",
        "signal:price_structure",
        "regime:bull_or_chop",
        "scope:peelable_levels",
        "action:block_peel",
        "mode:breakout_hold_no_peel",
    )
    enable_volume_confirmed_full_exit = tagged_parameter(
        BooleanParameter(default=False, space="sell", optimize=True, load=True),
        "family:exit_peel",
        "family:feature_enables",
        "family:exit_enables",
        "switch:enable",
        "switch:exit_enable",
        "switch:volume_enable",
        "controls:volume_confirmed_full_exit",
        "domain:exit",
        "domain:volume",
        "role:gate_toggle",
        "signal:volume",
        "regime:bull_or_chop_or_bear",
        "scope:full_exit",
        "action:full_exit",
        "mode:volume_confirmed_full_exit",
    )
    # ------------------------------------------------------------------
    # Hyperopt: extreme condition guards
    # Space: buy
    # ------------------------------------------------------------------
    extreme_volume_threshold = tagged_parameter(
        CategoricalParameter([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], default=15.0, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:extreme_guard",
        "role:threshold",
        "signal:volume",
        "regime:bear_or_mixed",
        "scope:global",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    extreme_speed_threshold = tagged_parameter(
        CategoricalParameter([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], default=4.0, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:extreme_guard",
        "role:threshold",
        "signal:price_velocity",
        "regime:bear_or_mixed",
        "scope:global",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    extreme_downtrend_adx = tagged_parameter(
        CategoricalParameter([20, 25, 30, 35, 40, 45, 50, 60], default=35, space="buy", optimize=True, load=True),
        "family:deep_recovery",
        "family:extreme_guard",
        "role:threshold",
        "signal:trend_strength",
        "regime:bear_or_mixed",
        "scope:global",
        "domain:risk",
        "action:crash_guard",
        "mode:rebuy_crash_guard",
    )
    # ------------------------------------------------------------------
    # Hyperopt: full exit ladder
    # Compatibility note: grouped into built-in sell space for older Freqtrade.
    # Space: sell
    # ------------------------------------------------------------------

    exit_target_b1 = tagged_parameter(

        CategoricalParameter(EXIT_TARGET_B1_CHOICES, default=0.010, space="sell", optimize=True, load=True),

        "family:exit_peel",

        "role:block_exit_target",

        "signal:profit_take",

        "regime:bull_or_chop",

        "scope:block_1",

        "action:full_exit",

    )
    exit_target_b2 = tagged_parameter(
        CategoricalParameter(EXIT_TARGET_B2_CHOICES, default=0.020, space="sell", optimize=True, load=True),
        "family:exit_peel",
        "role:block_exit_target",
        "signal:profit_take",
        "regime:bull_or_chop",
        "scope:block_2",
        "action:full_exit",
    )
    exit_target_b3 = tagged_parameter(
        CategoricalParameter(EXIT_TARGET_B3_CHOICES, default=0.040, space="sell", optimize=True, load=True),
        "family:exit_peel",
        "role:block_exit_target",
        "signal:profit_take",
        "regime:bull_or_chop_or_bear",
        "scope:block_3",
        "action:full_exit",
    )
    plot_config = {
        "main_plot": {
            "bb_mid": {},
            "bb_lower": {},
            "ema_fast": {},
            "ema_slow": {},
        },
        "subplots": {
            "Momentum": {
                "rsi": {},
                "mfi": {},
                "macd_hist": {},
            },
            "Trend": {
                "adx": {},
                "plus_di": {},
                "minus_di": {},
            },
            "Regime State": {
                "regime_code": {},
                "regime_confidence": {},
                "regime_changed": {},
            },
            "Regime Scores": {
                "regime_bull_score": {},
                "regime_bear_score": {},
                "regime_chop_score": {},
                "regime_crash_score": {},
                "regime_bull_pressure": {},
                "regime_bear_pressure": {},
                "regime_crash_pressure": {},
                "regime_directional_pressure": {},
            },
            "Regime Inputs": {
                "regime_adx": {},
                "regime_atr_ratio": {},
                "regime_volume_ratio": {},
                "regime_btc_1d_rsi": {},
                "regime_btc_1d_trend_score": {},
                "regime_pair_1d_trend_score": {},
            },
            "Regime Response": {
                "regime_gap_mult_active": {},
                "regime_exit_mult_active": {},
                "regime_stake_mult_active": {},
                "regime_spacing_mult_active": {},
                "regime_peel_fraction_mult_active": {},
            },
            "Structure": {
                "touches_local_min_8": {},
                "touches_local_max_8": {},
                "holds_support_8": {},
                "breaks_local_max_with_volume_8": {},
            },
        },
    }

    # ------------------------------------------------------------------
    # Indicator section
    # ------------------------------------------------------------------

    def _btc_informative_pair(self) -> str:
        cfg = self.config if isinstance(self.config, dict) else {}
        stake = str(cfg.get("stake_currency") or "USDT").upper()
        trading_mode = str(cfg.get("trading_mode") or "spot").lower()
        if trading_mode in ("futures", "margin"):
            return f"BTC/{stake}:{stake}"
        return f"BTC/{stake}"

    @staticmethod
    def _pair_informative_pair(metadata: dict[str, Any]) -> str | None:
        pair = str(metadata.get("pair") or "").strip()
        return pair or None

    def _merge_informative_columns(
        self,
        base_dataframe: DataFrame,
        informative: DataFrame | None,
        prefix: str,
        informative_timeframe: str,
        column_builders: dict[str, Any],
    ) -> dict[str, pd.Series]:
        """
        Build informative-timeframe columns and align them onto the base timeframe.

        The informative data is kept small, derived columns are created once, then
        merged back with merge_informative_pair so the strategy only sees information that would
        have been available at the close of the last completed informative candle.
        """
        aligned: dict[str, pd.Series] = {}
        if informative is None or informative.empty or "date" not in informative.columns:
            for name in column_builders:
                aligned[f"{prefix}_{name}"] = pd.Series(np.nan, index=base_dataframe.index, dtype="float64")
            return aligned

        informative_small = informative.copy()
        informative_small = informative_small.sort_values("date")
        built: dict[str, pd.Series] = {}
        for name, builder in column_builders.items():
            try:
                series = builder(informative_small)
            except Exception:
                series = pd.Series(np.nan, index=informative_small.index, dtype="float64")
            built[name] = pd.Series(series, index=informative_small.index)

        informative_payload = informative_small[["date"]].copy()
        for name, series in built.items():
            informative_payload[name] = series

        merged = merge_informative_pair(
            base_dataframe.copy(),
            informative_payload,
            self.timeframe,
            informative_timeframe,
            ffill=True,
        )

        for name in built:
            merged_name = f"{name}_{informative_timeframe}"
            col = f"{prefix}_{name}"
            aligned[col] = merged[merged_name].ffill() if merged_name in merged.columns else pd.Series(np.nan, index=base_dataframe.index, dtype="float64")
            aligned[col].index = base_dataframe.index
        return aligned

    def informative_pairs(self) -> list[tuple[str, str]]:
        pairs = {(self._btc_informative_pair(), "1d")}
        whitelist: list[str] = []
        if self.dp:
            try:
                current_whitelist = self.dp.current_whitelist()
                if isinstance(current_whitelist, list):
                    whitelist = [str(pair) for pair in current_whitelist if str(pair)]
            except Exception as exc:
                logger.debug("Could not read current whitelist for 1d informative pairs: %s", exc)
        for pair in whitelist:
            pairs.add((pair, "1d"))
        return sorted(pairs)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Keep indicator generation parameter-agnostic.

        Hyperoptable parameters must not be used directly here because hyperopt does
        not recalculate indicators for every epoch unless explicitly configured to do
        so. Instead, precompute a finite set of reusable indicator variants here and
        select among them later inside callbacks / signal methods.

        Performance note:
        Build derived columns in one side dataframe and concatenate once at the end.
        This avoids repeated frame.insert() fragmentation warnings during hyperopt and
        reduces needless dataframe block churn.
        """
        pair = str(metadata.get("pair") or "")

        bb_window = 20
        std_mult = 2.0

        close = dataframe["close"]
        open_ = dataframe["open"]
        high = dataframe["high"]
        low = dataframe["low"]
        volume = dataframe["volume"]
        close_safe = close.replace(0, np.nan)

        new_cols: dict[str, pd.Series] = {}
        informative_struct_cache: dict[int, dict[str, pd.Series]] = {}

        def _structural_columns_for_frame(df: DataFrame) -> dict[str, pd.Series]:
            cache_key = id(df)
            cached = informative_struct_cache.get(cache_key)
            if cached is not None:
                return cached

            df_close = df["close"]
            df_open = df["open"]
            df_high = df["high"]
            df_low = df["low"]
            df_volume = df["volume"]
            resistance = df_high.rolling(20).max().shift(1)
            support = df_low.rolling(20).min().shift(1)
            volume_ratio = df_volume / df_volume.ewm(span=20, adjust=False).mean().replace(0, np.nan)
            macd_hist = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)["macdhist"]
            rsi_series = ta.RSI(df, timeperiod=14)
            is_green = df_close > df_open
            close_gt_prev = df_close > df_close.shift(1)

            breakout_raw = df_close > resistance
            breakout_confirmed = breakout_raw.rolling(2).sum() >= 2
            breakout_retest_hold = (
                breakout_raw.rolling(5).max().shift(1).fillna(False).astype(bool)
                & (df_low <= resistance * 1.01)
                & (df_close >= resistance)
                & close_gt_prev
            )
            support_undercut = (df_low < support * 0.995) | (df_close < support)
            support_reclaim = (
                support_undercut.rolling(4).max().shift(1).fillna(False).astype(bool)
                & (df_close >= support)
                & is_green
            )
            support_reversal_confirmed = (
                support_reclaim
                & close_gt_prev
                & (rsi_series >= rsi_series.shift(1))
                & (macd_hist >= macd_hist.shift(1))
            )
            support_bounce_confirmed = (
                (df_low <= support * 1.01)
                & (df_close >= support)
                & ~support_undercut.rolling(4).max().fillna(False).astype(bool)
                & is_green
                & close_gt_prev
                & (volume_ratio >= 0.8)
            )
            cached = {
                "resistance_ref": resistance,
                "support_ref": support,
                "breakout_raw": breakout_raw.fillna(False),
                "breakout_confirmed": breakout_confirmed.fillna(False),
                "breakout_retest_hold": breakout_retest_hold.fillna(False),
                "support_undercut": support_undercut.fillna(False),
                "support_reclaim": support_reclaim.fillna(False),
                "support_reversal_confirmed": support_reversal_confirmed.fillna(False),
                "support_bounce_confirmed": support_bounce_confirmed.fillna(False),
            }
            informative_struct_cache[cache_key] = cached
            return cached

        def _informative_struct(name: str) -> Any:
            return lambda df: _structural_columns_for_frame(df)[name]

        informative_builders = {
            "close": lambda df: df["close"],
            "open": lambda df: df["open"],
            "high": lambda df: df["high"],
            "low": lambda df: df["low"],
            "volume": lambda df: df["volume"],
            "rsi": lambda df: ta.RSI(df, timeperiod=14),
            "mfi": lambda df: ta.MFI(df, timeperiod=14),
            "adx": lambda df: ta.ADX(df, timeperiod=14),
            "plus_di": lambda df: ta.PLUS_DI(df, timeperiod=14),
            "minus_di": lambda df: ta.MINUS_DI(df, timeperiod=14),
            "ema_fast": lambda df: ta.EMA(df, timeperiod=20),
            "ema_slow": lambda df: ta.EMA(df, timeperiod=50),
            "macd_hist": lambda df: ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)["macdhist"],
            "atr": lambda df: ta.ATR(df, timeperiod=14),
            "volume_ema": lambda df: df["volume"].ewm(span=20, adjust=False).mean(),
            "prev_low_5": lambda df: df["low"].rolling(5).min().shift(1),
            "prev_high_5": lambda df: df["high"].rolling(5).max().shift(1),
            "prev_low_20": lambda df: df["low"].rolling(20).min().shift(1),
            "prev_high_20": lambda df: df["high"].rolling(20).max().shift(1),
            "resistance_ref": _informative_struct("resistance_ref"),
            "support_ref": _informative_struct("support_ref"),
            "breakout_raw": _informative_struct("breakout_raw"),
            "breakout_confirmed": _informative_struct("breakout_confirmed"),
            "breakout_retest_hold": _informative_struct("breakout_retest_hold"),
            "support_undercut": _informative_struct("support_undercut"),
            "support_reclaim": _informative_struct("support_reclaim"),
            "support_reversal_confirmed": _informative_struct("support_reversal_confirmed"),
            "support_bounce_confirmed": _informative_struct("support_bounce_confirmed"),
        }

        if self.dp:
            btc_pair = self._btc_informative_pair()
            try:
                btc_df = self.dp.get_pair_dataframe(pair=btc_pair, timeframe="1d")
                btc_cols = self._merge_informative_columns(dataframe, btc_df, "btc_1d", "1d", informative_builders)
                new_cols.update(btc_cols)
            except Exception as exc:
                logger.debug("BTC informative 1d data unavailable for %s: %s", btc_pair, exc)
                new_cols.update(self._merge_informative_columns(dataframe, None, "btc_1d", "1d", informative_builders))

            pair_informative = self._pair_informative_pair(metadata)
            try:
                pair_df = self.dp.get_pair_dataframe(pair=pair_informative, timeframe="1d") if pair_informative else None
                pair_cols = self._merge_informative_columns(dataframe, pair_df, "pair_1d", "1d", informative_builders)
                new_cols.update(pair_cols)
            except Exception as exc:
                logger.debug("Pair informative 1d data unavailable for %s: %s", pair_informative or pair, exc)
                new_cols.update(self._merge_informative_columns(dataframe, None, "pair_1d", "1d", informative_builders))
        else:
            new_cols.update(self._merge_informative_columns(dataframe, None, "btc_1d", "1d", informative_builders))
            new_cols.update(self._merge_informative_columns(dataframe, None, "pair_1d", "1d", informative_builders))

        for prefix in ("btc_1d", "pair_1d"):
            close_1d = new_cols[f"{prefix}_close"]
            atr_1d = new_cols[f"{prefix}_atr"]
            atr_1d_safe = close_1d.replace(0, np.nan)
            new_cols[f"{prefix}_atr_pct"] = atr_1d / atr_1d_safe
            volume_ema_1d = new_cols[f"{prefix}_volume_ema"].replace(0, np.nan)
            new_cols[f"{prefix}_volume_ratio"] = new_cols[f"{prefix}_volume"] / volume_ema_1d
            new_cols[f"{prefix}_trend_stack_bull"] = (close_1d > new_cols[f"{prefix}_ema_fast"]) & (new_cols[f"{prefix}_ema_fast"] > new_cols[f"{prefix}_ema_slow"])
            new_cols[f"{prefix}_trend_stack_bear"] = (close_1d < new_cols[f"{prefix}_ema_fast"]) & (new_cols[f"{prefix}_ema_fast"] < new_cols[f"{prefix}_ema_slow"])
            new_cols[f"{prefix}_touches_prev_low_20"] = (close_1d <= new_cols[f"{prefix}_prev_low_20"]) | ((new_cols[f"{prefix}_prev_low_20"] - close_1d) / new_cols[f"{prefix}_prev_low_20"].replace(0, np.nan) <= 0.01)
            new_cols[f"{prefix}_touches_prev_high_20"] = (close_1d >= new_cols[f"{prefix}_prev_high_20"]) | ((close_1d - new_cols[f"{prefix}_prev_high_20"]) / new_cols[f"{prefix}_prev_high_20"].replace(0, np.nan) >= -0.01)

        ema_fast = ta.EMA(dataframe, timeperiod=20)
        ema_slow = ta.EMA(dataframe, timeperiod=50)
        rsi = ta.RSI(dataframe, timeperiod=14)
        mfi = ta.MFI(dataframe, timeperiod=14)
        adx = ta.ADX(dataframe, timeperiod=14)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)

        new_cols["ema_fast"] = ema_fast
        new_cols["ema_slow"] = ema_slow
        new_cols["rsi"] = rsi
        new_cols["mfi"] = mfi
        new_cols["adx"] = adx
        new_cols["plus_di"] = plus_di
        new_cols["minus_di"] = minus_di

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        new_cols["macd"] = macd["macd"]
        new_cols["macd_signal"] = macd["macdsignal"]
        new_cols["macd_hist"] = macd["macdhist"]

        bb_mid = close.rolling(bb_window).mean()
        bb_std = close.rolling(bb_window).std(ddof=0)
        bb_lower = bb_mid - std_mult * bb_std
        new_cols["bb_mid"] = bb_mid
        new_cols["bb_lower"] = bb_lower

        atr_pct_map: dict[int, pd.Series] = {}
        atr_pct_ema_map: dict[tuple[int, int], pd.Series] = {}
        volume_ema_map: dict[int, pd.Series] = {}

        for window in self.ATR_WINDOW_CHOICES:
            atr_col = f"atr_{window}"
            atr_pct_col = f"atr_pct_{window}"

            atr_series = ta.ATR(dataframe, timeperiod=window)
            atr_pct_series = atr_series / close_safe

            new_cols[atr_col] = atr_series
            new_cols[atr_pct_col] = atr_pct_series
            atr_pct_map[window] = atr_pct_series

            for span in self.ATR_EMA_SPAN_CHOICES:
                atr_pct_ema_series = atr_pct_series.ewm(span=span, adjust=False).mean()
                new_cols[f"atr_pct_ema_{window}_{span}"] = atr_pct_ema_series
                atr_pct_ema_map[(window, span)] = atr_pct_ema_series

        for span in self.VOLUME_EMA_SPAN_CHOICES:
            volume_ema_series = volume.ewm(span=span, adjust=False).mean()
            new_cols[f"volume_ema_{span}"] = volume_ema_series
            volume_ema_map[span] = volume_ema_series

        for lookback in self.SUPPORT_LOOKBACK_CHOICES:
            prev_low = low.rolling(lookback).min().shift(1)
            prev_high = high.rolling(lookback).max().shift(1)
            prev_low_safe = prev_low.replace(0, np.nan)
            prev_high_safe = prev_high.replace(0, np.nan)

            new_cols[f"rolling_low_prev_{lookback}"] = prev_low
            new_cols[f"rolling_high_prev_{lookback}"] = prev_high
            new_cols[f"support_distance_pct_{lookback}"] = (close - prev_low) / prev_low_safe
            new_cols[f"resistance_distance_pct_{lookback}"] = (prev_high - close) / prev_high_safe
            new_cols[f"touches_local_min_{lookback}"] = low <= prev_low
            new_cols[f"touches_local_max_{lookback}"] = high >= prev_high

        close_pct_change = close.pct_change().fillna(0.0)
        abs_close_pct_change = close_pct_change.abs()
        candle_range_pct = (high - low) / close_safe
        candle_body_pct = (close - open_).abs() / close_safe

        new_cols["close_pct_change"] = close_pct_change
        new_cols["abs_close_pct_change"] = abs_close_pct_change
        new_cols["candle_range_pct"] = candle_range_pct
        new_cols["candle_body_pct"] = candle_body_pct

        for window in self.ATR_WINDOW_CHOICES:
            for span in self.ATR_EMA_SPAN_CHOICES:
                atr_ref = atr_pct_ema_map[(window, span)].replace(0, np.nan)
                move_vs_vol = close_pct_change / atr_ref
                range_vs_vol = candle_range_pct / atr_ref
                body_vs_vol = candle_body_pct / atr_ref
                move_abs_vs_vol = move_vs_vol.abs()
                move_up_vs_vol = move_vs_vol.clip(lower=0.0)
                move_down_vs_vol = (-move_vs_vol).clip(lower=0.0)

                new_cols[f"event_move_vs_vol_{window}_{span}"] = move_vs_vol
                new_cols[f"event_range_vs_vol_{window}_{span}"] = range_vs_vol
                new_cols[f"event_body_vs_vol_{window}_{span}"] = body_vs_vol
                new_cols[f"event_move_abs_vs_vol_{window}_{span}"] = move_abs_vs_vol
                new_cols[f"event_move_up_vs_vol_{window}_{span}"] = move_up_vs_vol
                new_cols[f"event_move_down_vs_vol_{window}_{span}"] = move_down_vs_vol

                for lookback in self.EVENT_CUSUM_LOOKBACK_CHOICES:
                    new_cols[f"event_cusum_style_pos_{window}_{span}_{lookback}"] = move_up_vs_vol.rolling(lookback).sum()
                    new_cols[f"event_cusum_style_neg_{window}_{span}_{lookback}"] = move_down_vs_vol.rolling(lookback).sum()

        for span, volume_ema_series in volume_ema_map.items():
            volume_ratio = volume / volume_ema_series.replace(0, np.nan)
            new_cols[f"volume_ratio_{span}"] = volume_ratio

        is_green = close > open_
        close_gt_prev_close = close > close.shift(1)
        below_ema_fast = close < ema_fast
        below_bb_mid = close < bb_mid

        new_cols["is_green"] = is_green
        new_cols["close_gt_prev_close"] = close_gt_prev_close
        new_cols["below_ema_fast"] = below_ema_fast
        new_cols["below_bb_mid"] = below_bb_mid

        structure_volume_ratio = new_cols[f"volume_ratio_{self.VOLUME_EMA_SPAN_CHOICES[0]}"]
        for lookback in self.SUPPORT_LOOKBACK_CHOICES:
            prev_low = new_cols[f"rolling_low_prev_{lookback}"]
            prev_high = new_cols[f"rolling_high_prev_{lookback}"]
            prev_low_safe = prev_low.replace(0, np.nan)
            prev_high_safe = prev_high.replace(0, np.nan)
            breakout_raw = close > prev_high
            support_undercut = (low < prev_low * 0.995) | (close < prev_low)
            recent_breakout = breakout_raw.rolling(5).max().shift(1).fillna(False).astype(bool)
            recent_undercut = support_undercut.rolling(4).max().shift(1).fillna(False).astype(bool)
            current_or_recent_undercut = support_undercut.rolling(4).max().fillna(False).astype(bool)
            support_reclaim = recent_undercut & (close >= prev_low) & is_green
            support_reversal_confirmed = (
                support_reclaim
                & close_gt_prev_close
                & (rsi >= rsi.shift(1))
                & (macd["macdhist"] >= macd["macdhist"].shift(1))
            )
            support_bounce_confirmed = (
                (low <= prev_low * 1.01)
                & (close >= prev_low)
                & ~current_or_recent_undercut
                & is_green
                & close_gt_prev_close
                & (structure_volume_ratio >= 0.8)
            )

            new_cols[f"holds_support_{lookback}"] = (
                (low <= prev_low)
                & (close >= prev_low)
                & is_green
                & (structure_volume_ratio >= 1.0)
            )
            new_cols[f"holds_local_min_with_volume_{lookback}"] = (
                new_cols[f"touches_local_min_{lookback}"]
                & (close >= prev_low)
                & (structure_volume_ratio >= 1.0)
                & (rsi >= 40)
            )
            new_cols[f"rejects_local_max_{lookback}"] = (
                new_cols[f"touches_local_max_{lookback}"]
                & (close <= prev_high)
                & (structure_volume_ratio >= 1.0)
            )
            new_cols[f"breaks_local_max_with_volume_{lookback}"] = (
                (close > prev_high)
                & close_gt_prev_close
                & (structure_volume_ratio >= 1.0)
                & (macd["macdhist"] > 0)
            )
            new_cols[f"breakout_distance_pct_{lookback}"] = (close - prev_high) / prev_high_safe
            new_cols[f"support_recovery_pct_{lookback}"] = (close - prev_low) / prev_low_safe
            new_cols[f"resistance_ref_{lookback}"] = prev_high
            new_cols[f"support_ref_{lookback}"] = prev_low
            new_cols[f"breakout_raw_{lookback}"] = breakout_raw.fillna(False)
            for confirm in (1, 2, 3):
                new_cols[f"breakout_confirmed_{lookback}_{confirm}"] = (breakout_raw.rolling(confirm).sum() >= confirm).fillna(False)
            new_cols[f"breakout_confirmed_{lookback}"] = new_cols[f"breakout_confirmed_{lookback}_2"]
            new_cols[f"breakout_retest_hold_{lookback}"] = (
                recent_breakout
                & (low <= prev_high * 1.01)
                & (close >= prev_high)
                & close_gt_prev_close
            ).fillna(False)
            new_cols[f"support_undercut_{lookback}"] = support_undercut.fillna(False)
            new_cols[f"support_reclaim_{lookback}"] = support_reclaim.fillna(False)
            new_cols[f"support_reversal_confirmed_{lookback}"] = support_reversal_confirmed.fillna(False)
            new_cols[f"support_bounce_confirmed_{lookback}"] = support_bounce_confirmed.fillna(False)

        trend_stack_bull = (close > ema_fast) & (ema_fast > ema_slow)
        trend_stack_bear = (close < ema_fast) & (ema_fast < ema_slow)
        di_bull = plus_di >= minus_di
        di_bear = minus_di > plus_di

        new_cols["trend_stack_bull"] = trend_stack_bull
        new_cols["trend_stack_bear"] = trend_stack_bear
        new_cols["di_bull"] = di_bull
        new_cols["di_bear"] = di_bear

        features = pd.DataFrame(new_cols, index=dataframe.index)
        enriched = pd.concat([dataframe.copy(), features], axis=1)

        regime_params = {
            "regime_adx_trend_min": float(self.regime_adx_trend_min.value),
            "regime_bear_atr_spike": float(self.regime_bear_atr_spike.value),
            "regime_confirm_bars": int(self.regime_confirm_bars.value),
            "regime_confirm_choices": tuple(int(v) for v in self.REGIME_CONFIRM_CHOICES),
            "regime_gap_mult_bear": float(self.regime_gap_mult_bear.value),
            "regime_exit_mult_bear": float(self.regime_exit_mult_bear.value),
            "regime_gap_mult_bull": float(self.regime_gap_mult_bull.value),
            "regime_gap_mult_chop": float(self.regime_gap_mult_chop.value),
            "regime_exit_mult_bull": float(self.regime_exit_mult_bull.value),
            "regime_exit_mult_chop": float(self.regime_exit_mult_chop.value),
            "use_btc_1d_rsi_regime": bool(self.use_btc_1d_rsi_regime.value),
            "enable_pair_1d_context": bool(self.enable_pair_1d_context.value),
            "btc_1d_rsi_bull_min": float(self.btc_1d_rsi_bull_min.value),
            "btc_1d_rsi_bear_max": float(self.btc_1d_rsi_bear_max.value),
            "atr_pct_col": self._selected_atr_pct_col(),
            "atr_pct_ema_col": self._selected_atr_pct_ema_col(),
            "volume_ratio_col": f"volume_ratio_{int(self.volume_ema_span.value)}",
        }
        enriched = add_regime_input_columns(enriched, regime_params)
        enriched = add_regime_score_columns(enriched, regime_params)
        enriched = add_regime_state_columns(enriched, regime_params)
        enriched = add_regime_response_columns(enriched, regime_params)
        return enriched

    def _false_mask(self, dataframe: DataFrame) -> pd.Series:
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _all_conditions(self, dataframe: DataFrame, conditions: list[pd.Series]) -> pd.Series:
        if not conditions:
            return self._false_mask(dataframe)
        result = pd.Series(True, index=dataframe.index, dtype="bool")
        for condition in conditions:
            result &= pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
        return result

    def _bool_col(self, dataframe: DataFrame, column: str) -> pd.Series:
        if column not in dataframe.columns:
            return self._false_mask(dataframe)
        return dataframe[column].astype("boolean").fillna(False).astype(bool)

    def _num_col(self, dataframe: DataFrame, column: str) -> pd.Series:
        if column not in dataframe.columns:
            return pd.Series(np.nan, index=dataframe.index, dtype="float64")
        return pd.to_numeric(dataframe[column], errors="coerce")

    def _context_state_for_seed(self, data: DataFrame | pd.Series, seed_pattern: str) -> dict[str, Any]:
        def classify(prefix: str) -> Any:
            if isinstance(data, pd.DataFrame):
                if f"{prefix}_close" not in data.columns:
                    return pd.Series("unknown", index=data.index, dtype="object")
                trend_bull = self._bool_col(data, f"{prefix}_trend_stack_bull")
                trend_bear = self._bool_col(data, f"{prefix}_trend_stack_bear")
                breakout = self._bool_col(data, f"{prefix}_breakout_confirmed") | self._bool_col(data, f"{prefix}_breakout_retest_hold")
                support = self._bool_col(data, f"{prefix}_support_reversal_confirmed") | self._bool_col(data, f"{prefix}_support_bounce_confirmed")
                known = self._num_col(data, f"{prefix}_close").notna()
                aligned = breakout | trend_bull if seed_pattern == "breakout" else support
                hostile = trend_bear & ~aligned
                state = pd.Series("neutral", index=data.index, dtype="object")
                state.loc[~known] = "unknown"
                state.loc[aligned & known] = "aligned"
                state.loc[hostile & known] = "hostile"
                return state

            if data.get(f"{prefix}_close") is None or pd.isna(data.get(f"{prefix}_close")):
                return "unknown"
            trend_bull = bool(data.get(f"{prefix}_trend_stack_bull", False))
            trend_bear = bool(data.get(f"{prefix}_trend_stack_bear", False))
            breakout = bool(data.get(f"{prefix}_breakout_confirmed", False) or data.get(f"{prefix}_breakout_retest_hold", False))
            support = bool(data.get(f"{prefix}_support_reversal_confirmed", False) or data.get(f"{prefix}_support_bounce_confirmed", False))
            aligned = (breakout or trend_bull) if seed_pattern == "breakout" else support
            if aligned:
                return "aligned"
            if trend_bear:
                return "hostile"
            return "neutral"

        return {"btc_context": classify("btc_1d"), "pair_context": classify("pair_1d")}

    def _context_allows_seed(self, context_state: dict[str, Any], seed_pattern: str) -> Any:
        block_param = {
            "breakout": bool(self.breakout_block_on_hostile_btc_context.value),
            "support_reversal": bool(self.support_reversal_block_on_hostile_btc_context.value),
            "support_bounce": bool(self.support_bounce_block_on_hostile_btc_context.value),
            "crash_recovery": bool(self.crash_recovery_block_on_hostile_btc_context.value),
        }.get(seed_pattern, False)
        btc_state = context_state.get("btc_context", "unknown")
        if isinstance(btc_state, pd.Series):
            return ~(block_param & (btc_state == "hostile"))
        return not (block_param and btc_state == "hostile")

    def _context_profile(self, ledger: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
        entry_context = ledger.get("entry_context") if isinstance(ledger.get("entry_context"), dict) else {}
        seed_pattern = str(entry_context.get("seed_pattern") or "unknown")
        state = self._context_state_for_seed(view["last"], seed_pattern)
        if state.get("btc_context") == "aligned" and state.get("pair_context") == "aligned":
            return {"stake_mult": 1.1, "target_mult": 1.1, "context_state": state}
        if state.get("btc_context") == "hostile":
            return {"stake_mult": 0.7, "peel_fraction_mult": 1.25, "context_state": state}
        return {"context_state": state}

    def _entry_base_filters(self, dataframe: DataFrame) -> list[pd.Series]:
        return [self._num_col(dataframe, "volume") > 0]

    def _stabilizing_mask(self, dataframe: DataFrame) -> pd.Series:
        return self._bool_col(dataframe, "is_green") & self._bool_col(dataframe, "close_gt_prev_close")

    def _volume_confirm_mask(self, dataframe: DataFrame, threshold: float = 1.0) -> pd.Series:
        return self._num_col(dataframe, f"volume_ratio_{int(self.volume_ema_span.value)}") >= float(threshold)

    def _rising_mask(self, dataframe: DataFrame, column: str) -> pd.Series:
        values = self._num_col(dataframe, column)
        return values >= values.shift(1)

    def _entry_bear_context_mask(self, dataframe: DataFrame) -> pd.Series:
        confirm = int(self.regime_confirm_bars.value)
        return self._bool_col(dataframe, f"regime_bear_confirm_{confirm}")

    def _recent_true_mask(self, series: pd.Series, bars: int) -> pd.Series:
        window = max(1, int(bars))
        return pd.Series(series, index=series.index).fillna(False).astype(bool).rolling(window).max().fillna(False).astype(bool)

    def _confirmed_for_bars_mask(self, series: pd.Series, bars: int) -> pd.Series:
        window = max(1, int(bars))
        return pd.Series(series, index=series.index).fillna(False).astype(bool).rolling(window).sum().fillna(0) >= window

    def _entry_support_reversal_quality_filters(self, dataframe: DataFrame) -> list[pd.Series]:
        conditions: list[pd.Series] = []
        if bool(self.support_reversal_1h_use_below_ema_fast.value):
            conditions.append(self._bool_col(dataframe, "below_ema_fast"))
        conditions.append(self._num_col(dataframe, "rsi") < float(self.support_reversal_1h_rsi_max.value))
        if bool(self.support_reversal_1h_use_rsi_rising.value):
            conditions.append(self._rising_mask(dataframe, "rsi"))
        if bool(self.support_reversal_1h_use_macd_hist_rising.value):
            conditions.append(self._rising_mask(dataframe, "macd_hist"))
        if bool(self.support_reversal_1h_use_bb_lower.value):
            conditions.append(self._num_col(dataframe, "close") <= self._num_col(dataframe, "bb_lower") * float(self.support_reversal_1h_bb_lower_buffer.value))
        if bool(self.support_reversal_1h_require_stabilizing.value):
            conditions.append(self._stabilizing_mask(dataframe))
        if bool(self.support_reversal_1h_use_volume_confirm.value):
            conditions.append(self._volume_confirm_mask(dataframe, float(self.support_reversal_1h_volume_ratio_min.value)))
        return conditions

    def _entry_support_bounce_quality_filters(self, dataframe: DataFrame) -> list[pd.Series]:
        conditions: list[pd.Series] = []
        if bool(self.support_bounce_1h_use_below_ema_fast.value):
            conditions.append(self._bool_col(dataframe, "below_ema_fast"))
        conditions.append(self._num_col(dataframe, "rsi") < float(self.support_bounce_1h_rsi_max.value))
        if bool(self.support_bounce_1h_use_bb_lower.value):
            conditions.append(self._num_col(dataframe, "close") <= self._num_col(dataframe, "bb_lower") * float(self.support_bounce_1h_bb_lower_buffer.value))
        if bool(self.support_bounce_1h_require_stabilizing.value):
            conditions.append(self._stabilizing_mask(dataframe))
        if bool(self.support_bounce_1h_use_volume_confirm.value):
            conditions.append(self._volume_confirm_mask(dataframe, float(self.support_bounce_1h_volume_ratio_min.value)))
        return conditions

    def _entry_crash_recovery_quality_filters(self, dataframe: DataFrame) -> list[pd.Series]:
        conditions: list[pd.Series] = [self._num_col(dataframe, "rsi") < float(self.crash_recovery_rsi_max.value)]
        if bool(self.crash_recovery_require_stabilizing.value):
            conditions.append(self._stabilizing_mask(dataframe))
        if bool(self.crash_recovery_use_volume_confirm.value):
            conditions.append(self._volume_confirm_mask(dataframe, float(self.crash_recovery_volume_ratio_min.value)))
        if bool(self.crash_recovery_use_rsi_rising.value):
            conditions.append(self._rising_mask(dataframe, "rsi"))
        if bool(self.crash_recovery_use_macd_hist_rising.value):
            conditions.append(self._rising_mask(dataframe, "macd_hist"))
        return conditions

    def _entry_mask_1h_trigger_for_daily_seed(self, dataframe: DataFrame, seed_pattern: str) -> pd.Series:
        lookback = int(self.support_lookback.value)
        close_gt_prev = self._bool_col(dataframe, "close_gt_prev_close")
        green = self._bool_col(dataframe, "is_green")
        stabilizing = close_gt_prev & green
        if seed_pattern == "breakout":
            volume_light = self._volume_confirm_mask(dataframe, float(self.breakout_1d_volume_ratio_min.value))
            hist_rising = self._rising_mask(dataframe, "macd_hist")
            trigger = close_gt_prev & (
                green
                | self._bool_col(dataframe, f"breakout_retest_hold_{lookback}")
                | (self._num_col(dataframe, "close") > self._num_col(dataframe, f"resistance_ref_{lookback}"))
            )
            return trigger & volume_light & hist_rising
        if seed_pattern == "support_reversal":
            trigger = close_gt_prev & green & (
                self._bool_col(dataframe, f"support_reclaim_{lookback}")
                | self._bool_col(dataframe, f"support_reversal_confirmed_{lookback}")
                | stabilizing
            )
            return trigger
        if seed_pattern == "support_bounce":
            near_support = self._num_col(dataframe, f"support_distance_pct_{lookback}") <= 0.02
            trigger = close_gt_prev & green & (
                near_support
                | self._bool_col(dataframe, f"holds_support_{lookback}")
                | self._bool_col(dataframe, f"support_bounce_confirmed_{lookback}")
                | stabilizing
            )
            return trigger
        return self._false_mask(dataframe)

    def _entry_mask_breakout_1h(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_breakout_seed_1h.value):
            return self._false_mask(dataframe)
        lookback = int(self.support_lookback.value)
        confirm = int(self.breakout_1h_confirm_bars.value)
        resistance = self._num_col(dataframe, f"resistance_ref_{lookback}")
        retest_buffer = float(self.breakout_1h_retest_buffer.value)
        retest_hold = self._bool_col(dataframe, f"breakout_retest_hold_{lookback}") | (
            (self._num_col(dataframe, "low") <= resistance * (1.0 + retest_buffer))
            & (self._num_col(dataframe, "close") >= resistance)
        )
        structural = self._bool_col(dataframe, f"breakout_confirmed_{lookback}_{confirm}") | retest_hold
        bear = self._entry_bear_context_mask(dataframe)
        if bool(self.breakout_bear_require_retest.value):
            structural = (~bear & structural) | (bear & retest_hold)
        conditions = self._entry_base_filters(dataframe) + [structural]
        if bool(self.breakout_1h_require_stabilizing.value):
            conditions.append(self._stabilizing_mask(dataframe))
        conditions.append(self._num_col(dataframe, "rsi") >= float(self.breakout_1h_rsi_min.value))
        if bool(self.breakout_1h_use_volume_confirm.value):
            conditions.append(self._volume_confirm_mask(dataframe, float(self.breakout_1h_volume_ratio_min.value)))
        if bool(self.breakout_1h_use_macd_confirm.value):
            conditions.append(self._num_col(dataframe, "macd_hist") > 0)
        conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "breakout"), "breakout"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_support_reversal_1h(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_support_reversal_seed_1h.value):
            return self._false_mask(dataframe)
        lookback = int(self.support_lookback.value)
        confirm_bars = int(self.support_reversal_1h_confirm_bars.value)
        confirmed = self._confirmed_for_bars_mask(self._bool_col(dataframe, f"support_reversal_confirmed_{lookback}"), confirm_bars)
        support = self._num_col(dataframe, f"support_ref_{lookback}")
        undercut_buffer = float(self.support_reversal_1h_undercut_buffer.value)
        buffered_undercut = self._bool_col(dataframe, f"support_undercut_{lookback}") | (
            support.notna() & (self._num_col(dataframe, "low") <= support * (1.0 - undercut_buffer))
        )
        reclaim_recent = self._recent_true_mask(buffered_undercut, int(self.support_reversal_1h_reclaim_lookback.value)) & self._bool_col(dataframe, f"support_reclaim_{lookback}")
        structural = confirmed | reclaim_recent
        if bool(self.support_reversal_bear_require_confirmed_reclaim.value):
            bear = self._entry_bear_context_mask(dataframe)
            structural = (~bear & structural) | (bear & confirmed)
        conditions = self._entry_base_filters(dataframe) + self._entry_support_reversal_quality_filters(dataframe) + [structural]
        conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "support_reversal"), "support_reversal"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_support_bounce_1h(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_support_bounce_seed_1h.value):
            return self._false_mask(dataframe)
        lookback = int(self.support_lookback.value)
        structural = self._bool_col(dataframe, f"support_bounce_confirmed_{lookback}")
        if bool(self.support_bounce_1h_use_hold_support.value):
            structural |= self._bool_col(dataframe, f"holds_support_{lookback}")
        if bool(self.support_bounce_1h_use_local_min_bounce.value):
            structural |= self._bool_col(dataframe, f"holds_local_min_with_volume_{lookback}")
        if bool(self.support_bounce_bear_require_stronger_support_hold.value):
            bear = self._entry_bear_context_mask(dataframe)
            structural = (~bear & structural) | (bear & self._bool_col(dataframe, f"support_bounce_confirmed_{lookback}"))
        conditions = self._entry_base_filters(dataframe) + self._entry_support_bounce_quality_filters(dataframe) + [structural]
        near_support = self._num_col(dataframe, f"support_distance_pct_{lookback}") <= float(self.support_bounce_1h_support_distance_max.value)
        conditions.append(near_support)
        no_recent_undercut = ~self._recent_true_mask(self._bool_col(dataframe, f"support_undercut_{lookback}"), int(self.support_bounce_1h_no_undercut_lookback.value))
        conditions.append(no_recent_undercut)
        if bool(self.support_bounce_1h_require_green_candle.value):
            conditions.append(self._bool_col(dataframe, "is_green"))
        if bool(self.support_bounce_1h_require_close_gt_prev.value):
            conditions.append(self._bool_col(dataframe, "close_gt_prev_close"))
        conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "support_bounce"), "support_bounce"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_breakout_1d(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_breakout_seed_1d.value):
            return self._false_mask(dataframe)
        confirm = int(self.breakout_1d_confirm_bars.value)
        resistance = self._num_col(dataframe, "pair_1d_resistance_ref")
        retest_buffer = float(self.breakout_1d_retest_buffer.value)
        retest_hold = self._bool_col(dataframe, "pair_1d_breakout_retest_hold") | (
            (self._num_col(dataframe, "pair_1d_low") <= resistance * (1.0 + retest_buffer))
            & (self._num_col(dataframe, "pair_1d_close") >= resistance)
        )
        confirmed = self._confirmed_for_bars_mask(self._bool_col(dataframe, "pair_1d_breakout_raw"), confirm) | self._bool_col(dataframe, "pair_1d_breakout_confirmed")
        structural = confirmed | retest_hold
        conditions = self._entry_base_filters(dataframe) + [structural]
        if bool(self.breakout_1d_require_1h_trigger.value):
            conditions.append(self._entry_mask_1h_trigger_for_daily_seed(dataframe, "breakout"))
        if bool(self.breakout_1d_use_btc_context.value):
            conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "breakout"), "breakout"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_support_reversal_1d(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_support_reversal_seed_1d.value):
            return self._false_mask(dataframe)
        confirm = int(self.support_reversal_1d_confirm_bars.value)
        confirmed = self._confirmed_for_bars_mask(self._bool_col(dataframe, "pair_1d_support_reversal_confirmed"), confirm)
        structural = confirmed | (
            self._bool_col(dataframe, "pair_1d_support_undercut") & self._bool_col(dataframe, "pair_1d_support_reclaim")
        )
        conditions = self._entry_base_filters(dataframe) + [structural]
        if bool(self.support_reversal_1d_require_1h_trigger.value):
            conditions.append(self._entry_mask_1h_trigger_for_daily_seed(dataframe, "support_reversal"))
        if bool(self.support_reversal_1d_use_btc_context.value):
            conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "support_reversal"), "support_reversal"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_support_bounce_1d(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_support_bounce_seed_1d.value):
            return self._false_mask(dataframe)
        conditions = self._entry_base_filters(dataframe) + [self._bool_col(dataframe, "pair_1d_support_bounce_confirmed")]
        if bool(self.support_bounce_1d_require_1h_trigger.value):
            conditions.append(self._entry_mask_1h_trigger_for_daily_seed(dataframe, "support_bounce"))
        if bool(self.support_bounce_1d_use_btc_context.value):
            conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "support_bounce"), "support_bounce"))
        return self._all_conditions(dataframe, conditions)

    def _entry_mask_crash_recovery(self, dataframe: DataFrame) -> pd.Series:
        if not bool(self.enable_crash_recovery_seed.value):
            return self._false_mask(dataframe)
        volume_extreme = self._num_col(dataframe, f"volume_ratio_{int(self.volume_ema_span.value)}") >= float(self.extreme_volume_threshold.value)
        atr_ref = self._num_col(dataframe, self._selected_atr_pct_ema_col()).replace(0, np.nan)
        downside_speed = (-self._num_col(dataframe, "close_pct_change")) / atr_ref
        speed_extreme = downside_speed.rolling(4).max() >= float(self.extreme_speed_threshold.value)
        downtrend_recent = (
            (self._num_col(dataframe, "adx") >= float(self.extreme_downtrend_adx.value))
            & (self._num_col(dataframe, "minus_di") > self._num_col(dataframe, "plus_di"))
        ).rolling(4).max().fillna(False).astype(bool)
        recovery = (
            self._bool_col(dataframe, "is_green")
            & self._bool_col(dataframe, "close_gt_prev_close")
            & (self._num_col(dataframe, "rsi") >= self._num_col(dataframe, "rsi").shift(1))
            & (self._num_col(dataframe, "macd_hist") >= self._num_col(dataframe, "macd_hist").shift(1))
        )
        lookback = int(self.support_lookback.value)
        if bool(self.crash_recovery_support_reclaim_required.value):
            recovery &= (
                self._bool_col(dataframe, f"support_reclaim_{lookback}")
                | self._bool_col(dataframe, f"support_reversal_confirmed_{lookback}")
            )
        recovery = self._confirmed_for_bars_mask(recovery, int(self.crash_recovery_confirm_bars.value))
        conditions = self._entry_base_filters(dataframe) + self._entry_crash_recovery_quality_filters(dataframe) + [volume_extreme | speed_extreme | downtrend_recent, recovery]
        conditions.append(self._context_allows_seed(self._context_state_for_seed(dataframe, "crash_recovery"), "crash_recovery"))
        return self._all_conditions(dataframe, conditions)

    def _resolve_entry_seed(self, dataframe: DataFrame, masks: dict[str, pd.Series]) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        normalized = {tag: pd.Series(mask, index=dataframe.index).fillna(False).astype(bool) for tag, mask in masks.items()}
        combo_masks = {
            "combo_1d_1h_breakout": (
                normalized.get("breakout_seed_1d", self._false_mask(dataframe))
                & normalized.get("breakout_seed_1h", self._false_mask(dataframe))
                & bool(self.enable_combo_1d_1h_breakout.value)
            ),
            "combo_1d_1h_support_reversal": (
                normalized.get("support_reversal_seed_1d", self._false_mask(dataframe))
                & normalized.get("support_reversal_seed_1h", self._false_mask(dataframe))
                & bool(self.enable_combo_1d_1h_support_reversal.value)
            ),
            "combo_1d_1h_support_bounce": (
                normalized.get("support_bounce_seed_1d", self._false_mask(dataframe))
                & normalized.get("support_bounce_seed_1h", self._false_mask(dataframe))
                & bool(self.enable_combo_1d_1h_support_bounce.value)
            ),
        }
        priority = [
            "combo_1d_1h_breakout",
            "combo_1d_1h_support_reversal",
            "combo_1d_1h_support_bounce",
            "crash_recovery_seed",
            "breakout_seed_1d",
            "support_reversal_seed_1d",
            "support_bounce_seed_1d",
            "breakout_seed_1h",
            "support_reversal_seed_1h",
            "support_bounce_seed_1h",
        ]
        resolved = {**normalized, **combo_masks}
        unresolved = pd.Series(True, index=dataframe.index, dtype="bool")
        for tag in priority:
            mask = resolved.get(tag, self._false_mask(dataframe)) & unresolved
            dataframe.loc[mask, ["enter_long", "enter_tag"]] = (1, tag)
            unresolved &= ~mask
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        masks = {
            "breakout_seed_1h": self._entry_mask_breakout_1h(dataframe),
            "support_reversal_seed_1h": self._entry_mask_support_reversal_1h(dataframe),
            "support_bounce_seed_1h": self._entry_mask_support_bounce_1h(dataframe),
            "breakout_seed_1d": self._entry_mask_breakout_1d(dataframe),
            "support_reversal_seed_1d": self._entry_mask_support_reversal_1d(dataframe),
            "support_bounce_seed_1d": self._entry_mask_support_bounce_1d(dataframe),
            "crash_recovery_seed": self._entry_mask_crash_recovery(dataframe),
        }
        return self._resolve_entry_seed(dataframe, masks)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    # ------------------------------------------------------------------
    # Persistent ledger helpers
    # ------------------------------------------------------------------

    def _blank_level(self) -> dict[str, Any]:
        return {
            "anchor_price": None,
            "live_stake_quote": 0.0,
            "filled_once": False,
            "cycle_count": 0,
            "next_action": "buy",
            "last_buy_size_quote": 0.0,
        }

    def _default_entry_context(self) -> dict[str, Any]:
        return {
            "seed_tag": None,
            "seed_pattern": None,
            "seed_timeframe": None,
            "seed_combo": False,
            "entry_regime": None,
            "entry_candle_id": None,
            "entry_support_ref": None,
            "entry_resistance_ref": None,
            "entry_structure_lookback": None,
            "btc_1d_context": {},
            "pair_1d_context": {},
        }

    def _default_ledger(self) -> dict[str, Any]:
        return {
            "last_action_candle": None,
            "pending_action": None,
            "planned_stakes_quote": [0.0] * self.LEVEL_COUNT,
            "exec_counters": self._default_exec_counters(),
            "levels": [self._blank_level() for _ in range(self.LEVEL_COUNT)],
            "entry_context": self._default_entry_context(),
            "regime_summary": {
                "seed_regime": None,
                "exit_regime": None,
                "deepest_level_reached": 0,
                "regime_candle_counts": {"bull": 0, "bear": 0, "chop": 0},
                "action_regime_counts": {
                    "add": {"bull": 0, "bear": 0, "chop": 0},
                    "rebuy": {"bull": 0, "bear": 0, "chop": 0},
                    "peel": {"bull": 0, "bear": 0, "chop": 0},
                    "full_exit": {"bull": 0, "bear": 0, "chop": 0},
                },
                "deepest_level_regime": None,
            },
            "last_processed_candle": None,
        }

    @staticmethod
    def _default_exec_counters() -> dict[str, int]:
        return {
            "adjust_calls": 0,
            "adjust_no_action": 0,
            "adjust_selected_add": 0,
            "adjust_selected_peel": 0,
            "adjust_selected_rebuy": 0,
            "check_peel_calls": 0,
            "check_peel_hits": 0,
            "check_add_calls": 0,
            "check_add_hits": 0,
            "check_add_reject_next_level": 0,
            "check_add_reject_mean_reversion": 0,
            "check_add_reject_anchor_missing": 0,
            "check_add_reject_close_above_anchor": 0,
            "check_add_reject_deep_drawdown": 0,
            "check_add_reject_deep_recovery": 0,
            "check_add_reject_prev_level_buffer": 0,
            "check_add_reject_stake_nonpositive": 0,
            "check_add_reject_stake_below_min": 0,
            "check_add_reject_stake_above_max": 0,
            "check_rebuy_calls": 0,
            "check_rebuy_hits": 0,
            "fill_applied_add": 0,
            "fill_applied_peel": 0,
            "fill_applied_rebuy": 0,
        }

    def _exec_counters(self, ledger: dict[str, Any]) -> dict[str, int]:
        counters = ledger.get("exec_counters")
        if not isinstance(counters, dict):
            counters = self._default_exec_counters()
            ledger["exec_counters"] = counters
            return counters

        defaults = self._default_exec_counters()
        for key, default_value in defaults.items():
            try:
                counters[key] = int(counters.get(key, default_value))
            except (TypeError, ValueError):
                counters[key] = int(default_value)
        return counters

    def _bump_exec_counter(self, ledger: dict[str, Any], key: str, inc: int = 1) -> int:
        counters = self._exec_counters(ledger)
        counters[key] = int(counters.get(key, 0)) + int(inc)
        return counters[key]

    def _log_exec_counters(self, trade: Trade, ledger: dict[str, Any], context: str) -> None:
        counters = self._exec_counters(ledger)
        logger.debug(
            "exec_counts pair=%s trade_id=%s ctx=%s adjust_calls=%d no_action=%d selected(add=%d peel=%d rebuy=%d) checks(peel=%d/%d add=%d/%d rebuy=%d/%d) add_rejects(next=%d meanrev=%d anchor=%d close=%d deep_dd=%d deep_recovery=%d prev_buf=%d stake<=0=%d min=%d max=%d) fills(add=%d peel=%d rebuy=%d)",
            trade.pair,
            getattr(trade, "id", None),
            context,
            counters["adjust_calls"],
            counters["adjust_no_action"],
            counters["adjust_selected_add"],
            counters["adjust_selected_peel"],
            counters["adjust_selected_rebuy"],
            counters["check_peel_hits"],
            counters["check_peel_calls"],
            counters["check_add_hits"],
            counters["check_add_calls"],
            counters["check_rebuy_hits"],
            counters["check_rebuy_calls"],
            counters["check_add_reject_next_level"],
            counters["check_add_reject_mean_reversion"],
            counters["check_add_reject_anchor_missing"],
            counters["check_add_reject_close_above_anchor"],
            counters["check_add_reject_deep_drawdown"],
            counters["check_add_reject_deep_recovery"],
            counters["check_add_reject_prev_level_buffer"],
            counters["check_add_reject_stake_nonpositive"],
            counters["check_add_reject_stake_below_min"],
            counters["check_add_reject_stake_above_max"],
            counters["fill_applied_add"],
            counters["fill_applied_peel"],
            counters["fill_applied_rebuy"],
        )

    def _get_ledger(self, trade: Trade) -> dict[str, Any]:
        ledger = trade.get_custom_data(key="grid_ledger")
        if not ledger:
            return self._default_ledger()
        self._exec_counters(ledger)

        # Migration: ensure regime_summary exists
        if "regime_summary" not in ledger:
            ledger["regime_summary"] = self._default_ledger()["regime_summary"]
        if "entry_context" not in ledger or not isinstance(ledger.get("entry_context"), dict):
            ledger["entry_context"] = self._default_entry_context()
        else:
            defaults = self._default_entry_context()
            for key, value in defaults.items():
                ledger["entry_context"].setdefault(key, value)
        if "last_processed_candle" not in ledger:
            ledger["last_processed_candle"] = None

        levels = ledger.get("levels")
        planned = ledger.get("planned_stakes_quote")
        if isinstance(levels, list):
            for idx, level in enumerate(levels[: self.LEVEL_COUNT]):
                if not isinstance(level, dict):
                    continue

                filled_once = bool(level.get("filled_once", False))
                live_stake = float(level.get("live_stake_quote") or 0.0)
                target_stake = 0.0
                if isinstance(planned, list) and idx < len(planned):
                    target_stake = float(planned[idx] or 0.0)

                if "last_buy_size_quote" not in level:
                    if filled_once and live_stake > 0.0:
                        level["last_buy_size_quote"] = float(live_stake)
                    elif filled_once and target_stake > 0.0:
                        level["last_buy_size_quote"] = float(target_stake)
                    else:
                        level["last_buy_size_quote"] = 0.0
                else:
                    level["last_buy_size_quote"] = float(level.get("last_buy_size_quote") or 0.0)

                if "next_action" not in level:
                    if not filled_once:
                        level["next_action"] = "buy"
                    else:
                        # Migration fallback for old ledgers:
                        # if old accounting shows this level was peeled (target > live),
                        # the next expected action should be buy; otherwise sell.
                        missing = max(0.0, target_stake - live_stake)
                        level["next_action"] = "buy" if missing > 1e-12 else "sell"
                else:
                    next_action = str(level.get("next_action") or "buy").lower()
                    level["next_action"] = "sell" if next_action == "sell" else "buy"

        return ledger

    def _set_ledger(self, trade: Trade, ledger: dict[str, Any]) -> None:
        trade.set_custom_data(key="grid_ledger", value=ledger)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_order_price(order: Order) -> float:
        for attr in ("safe_price", "average", "price"):
            value = getattr(order, attr, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _safe_order_amount(order: Order) -> float:
        for attr in ("safe_amount", "filled", "amount"):
            value = getattr(order, attr, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _current_candle_id(last_candle: pd.Series) -> str:
        return pd.Timestamp(last_candle["date"]).isoformat()

    @staticmethod
    def _num(series: pd.Series, key: str) -> float | None:
        value = series.get(key)
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _json_safe_bool(value: Any) -> bool:
        if value is None or pd.isna(value):
            return False
        return bool(value)

    @staticmethod
    def _json_safe_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _structural_context_from_candle(self, last_candle: pd.Series) -> dict[str, Any]:
        def build(prefix: str) -> dict[str, Any]:
            return {
                "breakout_confirmed": self._json_safe_bool(
                    last_candle.get(f"{prefix}_breakout_confirmed", False)
                    or last_candle.get(f"{prefix}_breakout_retest_hold", False)
                ),
                "support_reversal_confirmed": self._json_safe_bool(last_candle.get(f"{prefix}_support_reversal_confirmed", False)),
                "support_bounce_confirmed": self._json_safe_bool(last_candle.get(f"{prefix}_support_bounce_confirmed", False)),
                "trend_stack_bull": self._json_safe_bool(last_candle.get(f"{prefix}_trend_stack_bull", False)),
                "trend_stack_bear": self._json_safe_bool(last_candle.get(f"{prefix}_trend_stack_bear", False)),
                "rsi": self._json_safe_float(last_candle.get(f"{prefix}_rsi")),
                "volume_ratio": self._json_safe_float(last_candle.get(f"{prefix}_volume_ratio")),
            }

        return {
            "btc_1d_context": build("btc_1d"),
            "pair_1d_context": build("pair_1d"),
        }

    def _seed_context_from_enter_tag(self, enter_tag: str | None) -> dict[str, Any]:
        tag = str(enter_tag or "").strip()
        mapping = {
            "combo_1d_1h_breakout": ("breakout", "combo_1d_1h", True),
            "combo_1d_1h_support_reversal": ("support_reversal", "combo_1d_1h", True),
            "combo_1d_1h_support_bounce": ("support_bounce", "combo_1d_1h", True),
            "breakout_seed_1d": ("breakout", "1d", False),
            "breakout_seed_1h": ("breakout", "1h", False),
            "support_reversal_seed_1d": ("support_reversal", "1d", False),
            "support_reversal_seed_1h": ("support_reversal", "1h", False),
            "support_bounce_seed_1d": ("support_bounce", "1d", False),
            "support_bounce_seed_1h": ("support_bounce", "1h", False),
            "crash_recovery_seed": ("crash_recovery", "mixed", False),
        }
        pattern, timeframe, combo = mapping.get(tag, ("unknown", "unknown", False))
        return {
            "seed_tag": tag or None,
            "seed_pattern": pattern,
            "seed_timeframe": timeframe,
            "seed_combo": bool(combo),
        }

    def _entry_structure_refs_from_candle(self, seed_context: dict[str, Any], last_candle: pd.Series) -> dict[str, Any]:
        lookback = int(self.support_lookback.value)
        pattern = str(seed_context.get("seed_pattern") or "unknown")
        timeframe = str(seed_context.get("seed_timeframe") or "unknown")
        use_1d_ref = timeframe in ("1d", "combo_1d_1h")
        support_col = "pair_1d_support_ref" if use_1d_ref else f"support_ref_{lookback}"
        resistance_col = "pair_1d_resistance_ref" if use_1d_ref else f"resistance_ref_{lookback}"
        return {
            "entry_support_ref": self._json_safe_float(last_candle.get(support_col)) if pattern in ("support_bounce", "support_reversal", "crash_recovery") else None,
            "entry_resistance_ref": self._json_safe_float(last_candle.get(resistance_col)) if pattern == "breakout" else None,
            "entry_structure_lookback": 20 if use_1d_ref else lookback,
        }

    def _build_entry_context(self, trade: Trade, last_candle: pd.Series | None) -> dict[str, Any]:
        enter_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None)
        context = self._default_entry_context()
        context.update(self._seed_context_from_enter_tag(enter_tag))
        if last_candle is None:
            return context
        context["entry_regime"] = self._regime(last_candle)
        context["entry_candle_id"] = self._current_candle_id(last_candle)
        context.update(self._entry_structure_refs_from_candle(context, last_candle))
        context.update(self._structural_context_from_candle(last_candle))
        context["context_state"] = self._context_state_for_seed(last_candle, str(context.get("seed_pattern") or "unknown"))
        return context


    def _block_for_level(self, level_idx: int) -> int:
        return int(self.LEVEL_TO_BLOCK[level_idx])

    def _effective_min_entry_stake(
        self,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
    ) -> float:
        desired = max(0.0, float(proposed_stake or 0.0))
        max_valid = max_stake is not None and float(max_stake) > 0.0
        cap = float(max_stake) if max_valid else None
        floor = 0.0
        if min_stake is not None and float(min_stake) > 0.0:
            floor = float(min_stake) * self.MIN_STAKE_BUFFER_MULT

        capped_desired = min(desired, cap) if cap is not None else desired
        stake = max(floor, capped_desired)
        if cap is not None:
            stake = min(stake, cap)
        return float(stake)

    def _stake_floor_from_min(self, min_stake: float | None) -> float | None:
        if min_stake is None:
            return None
        min_stake_val = float(min_stake)
        if min_stake_val <= 0.0:
            return None
        return min_stake_val * self.MIN_STAKE_BUFFER_MULT

    def _ensure_min_planned_stakes(
        self,
        ledger: dict[str, Any],
        min_stake: float | None,
        max_stake: float,
    ) -> None:
        floor_stake = self._stake_floor_from_min(min_stake)
        if floor_stake is None:
            return

        planned = ledger.get("planned_stakes_quote")
        if not isinstance(planned, list) or len(planned) != self.LEVEL_COUNT:
            return

        floor_capped = min(float(max_stake), float(floor_stake))
        changed = False
        for idx in range(self.LEVEL_COUNT):
            current = float(planned[idx] or 0.0)
            if current < floor_capped:
                planned[idx] = float(floor_capped)
                changed = True

        if changed:
            ledger["planned_stakes_quote"] = [float(x) for x in planned]

    def _build_planned_stakes(
        self,
        min_entry_stake_quote: float,
        wallet_total_quote: float,
    ) -> list[float]:
        """
        Explicit 9-level quote-stake ladder.

        L1-L3: minimum stake scaled by per-level multipliers.
        L4: max(minimum stake, fixed_wallet_pct * wallet).
        L5-L9: chained multipliers from the previous level.
        """
        min_entry = float(min_entry_stake_quote)
        block_fixed = max(min_entry, float(wallet_total_quote) * float(self.fixed_wallet_pct.value))

        stakes = [0.0] * self.LEVEL_COUNT
        stakes[0] = min_entry * float(self.l1_mult.value)
        stakes[1] = min_entry * float(self.l2_mult.value)
        stakes[2] = min_entry * float(self.l3_mult.value)

        stakes[3] = block_fixed
        stakes[4] = stakes[3] * float(self.l5_mult.value)
        stakes[5] = stakes[4] * float(self.l6_mult.value)
        stakes[6] = stakes[5] * float(self.l7_mult.value)
        stakes[7] = stakes[6] * float(self.l8_mult.value)
        stakes[8] = stakes[7] * float(self.l9_mult.value)

        return stakes

    def _target_stake(self, ledger: dict[str, Any], level_idx: int) -> float:
        return float(ledger["planned_stakes_quote"][level_idx] or 0.0)

    def _live_stake(self, ledger: dict[str, Any], level_idx: int) -> float:
        return float(ledger["levels"][level_idx].get("live_stake_quote") or 0.0)

    def _missing_stake(self, ledger: dict[str, Any], level_idx: int) -> float:
        # Missing stake is the refill amount needed to bring a previously peeled level
        # back to its planned level size.
        return max(0.0, self._target_stake(ledger, level_idx) - self._live_stake(ledger, level_idx))

    def _level_is_open(self, ledger: dict[str, Any], level_idx: int) -> bool:
        return self._live_stake(ledger, level_idx) > 1e-12

    def _selected_atr_pct_col(self) -> str:
        return f"atr_pct_{int(self.atr_window.value)}"

    def _selected_atr_pct_ema_col(self) -> str:
        return f"atr_pct_ema_{int(self.atr_window.value)}_{int(self.atr_ema_span.value)}"

    def _selected_volume_ema_col(self) -> str:
        return f"volume_ema_{int(self.volume_ema_span.value)}"

    def _selected_support_col(self) -> str:
        return f"rolling_low_prev_{int(self.support_lookback.value)}"

    def _open_trades_snapshot(self) -> list[Trade]:
        try:
            trades = Trade.get_trades_proxy(is_open=True)
            return list(trades) if trades else []
        except Exception:
            try:
                # Fallback for older Freqtrade APIs.
                trades = Trade.get_open_trades()
                return list(trades) if trades else []
            except Exception:
                return []

    def _max_filled_level_for_trade(self, trade: Trade) -> int:
        """
        Return deepest filled virtual level as 1..9, or 0 when unknown/uninitialized.
        """
        ledger = trade.get_custom_data(key="grid_ledger")
        if not isinstance(ledger, dict):
            return 0
        levels = ledger.get("levels")
        if not isinstance(levels, list):
            return 0

        deepest = 0
        for idx, level in enumerate(levels[: self.LEVEL_COUNT]):
            if isinstance(level, dict) and bool(level.get("filled_once", False)):
                deepest = idx + 1
        return deepest

    def _dynamic_allowed_open_trades(self, open_trades: list[Trade]) -> int:
        """
        Simple portfolio risk-release cap.

        Objective:
        avoid filling the wallet with many shallow trades before a market-wide selloff,
        while still releasing fresh capacity once existing trades are already trapped in
        deeper recovery states.

        Rule set:
        - start with 2 open trades max
        - add one extra slot for EACH open trade that is at L7+

        This uses reached entry depth as the release signal rather than a floating PnL
        percentage because level depth is already the strategy's structural definition of
        drawdown severity.
        """
        depths = [self._max_filled_level_for_trade(t) for t in open_trades]
        base_cap = int(self.BASE_OPEN_TRADE_CAP)

        l7_plus_count = sum(1 for depth in depths if depth >= self.OPEN_TRADE_CAP_UNLOCK_LEVEL_1)
        allowed = base_cap + l7_plus_count

        return allowed


    def _regime(self, last_candle: pd.Series) -> str:
        """
        Read the already-classified regime from analyzed dataframe columns.

        Strategy action logic remains three-state (`bull` / `bear` / `chop`) for
        behavior stability. Crash/unknown regime codes are mapped safely onto this
        legacy action space.
        """
        regime_code = self._regime_code(last_candle)
        if regime_code == 1:
            return "bull"
        if regime_code in (-2, -1):
            return "bear"
        return "chop"

    def _regime_code(self, last_candle: pd.Series) -> int:
        code = self._num(last_candle, "regime_code")
        if code is None:
            return 9
        return int(round(code))

    def _regime_gap_mult(self, regime: str) -> float:
        if regime == "bull":
            return float(self.regime_gap_mult_bull.value)
        if regime == "bear":
            return float(self.regime_gap_mult_bear.value)
        return float(self.regime_gap_mult_chop.value)

    def _regime_exit_mult(self, regime: str) -> float:
        if regime == "bull":
            return float(self.regime_exit_mult_bull.value)
        if regime == "bear":
            return float(self.regime_exit_mult_bear.value)
        return float(self.regime_exit_mult_chop.value)

    def _regime_gap_mult_active(self, last_candle: pd.Series) -> float:
        mult = self._num(last_candle, "regime_gap_mult_active")
        if mult is not None and np.isfinite(mult) and mult > 0.0:
            return float(mult)
        return self._regime_gap_mult(self._regime(last_candle))

    def _regime_exit_mult_active(self, last_candle: pd.Series) -> float:
        mult = self._num(last_candle, "regime_exit_mult_active")
        if mult is not None and np.isfinite(mult) and mult > 0.0:
            return float(mult)
        return self._regime_exit_mult(self._regime(last_candle))

    def _leverage_regime_mult(self, regime: str) -> float:
        if regime == "bull":
            return float(self.leverage_mult_bull.value)
        if regime == "bear":
            return float(self.leverage_mult_bear.value)
        return float(self.leverage_mult_chop.value)

    def _leverage_seed_mult(self, seed_pattern: str) -> float:
        if seed_pattern == "breakout":
            return float(self.leverage_seed_mult_breakout.value)
        if seed_pattern in ("support_reversal", "support_bounce"):
            return float(self.leverage_seed_mult_support.value)
        if seed_pattern == "crash_recovery":
            return float(self.leverage_seed_mult_crash.value)
        return 1.0

    def _leverage_cap(self, regime: str, seed_pattern: str) -> float:
        if regime == "bull":
            if seed_pattern == "breakout":
                return max(1.0, float(self.leverage_cap_bull_breakout.value))
            return max(1.0, float(self.leverage_cap_bull.value))
        if regime == "bear":
            return max(1.0, float(self.leverage_cap_bear.value))
        return max(1.0, float(self.leverage_cap_chop.value))

    def _leverage_vol_brake_mult(self, last_candle: pd.Series) -> float:
        atr_pct = self._num(last_candle, self._selected_atr_pct_col())
        atr_pct_ema = self._num(last_candle, self._selected_atr_pct_ema_col())
        if atr_pct is None or atr_pct_ema is None or atr_pct_ema <= 0.0:
            return 1.0
        trigger = float(self.leverage_vol_brake_trigger.value)
        if atr_pct >= atr_pct_ema * trigger:
            return float(self.leverage_vol_brake_mult.value)
        return 1.0

    def _latest_analyzed_candle(self, pair: str) -> pd.Series | None:
        data_provider = getattr(self, "dp", None)
        if data_provider is None:
            return None
        try:
            dataframe, _ = data_provider.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        except Exception:
            return None
        if dataframe is None or dataframe.empty:
            return None
        return dataframe.iloc[-1].squeeze()

    def _gap_mult_values(self) -> list[float]:
        # L2-L6 remain anchor-spaced. L7-L9 are drawdown-band driven and therefore
        # do not consume shallow spacing multipliers.
        return [
            0.0,
            float(self.gap_mult_l2.value),
            float(self.gap_mult_l3.value),
            float(self.gap_mult_l4.value),
            float(self.gap_mult_l5.value),
            float(self.gap_mult_l6.value),
            0.0,
            0.0,
            0.0,
        ]

    def _is_deep_drawdown_level(self, level_idx: int) -> bool:
        return level_idx >= 6

    def _deep_drawdown_targets(self) -> list[float]:
        start = float(self.deep_drawdown_l7_pct.value)
        step_3 = float(self.deep_drawdown_step_l8_l9_pct.value)

        l7 = start
        l8 = l7 + step_3
        l9 = l8 + step_3

        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, l7, l8, l9]

    def _deep_drawdown_target(self, level_idx: int) -> float:
        return float(self._deep_drawdown_targets()[level_idx])

    def _min_gap_values(self) -> list[float]:
        return [0.0] + [float(self.MIN_GAP_PCT)] * (self.LEVEL_COUNT - 1)

    def _peel_profit_pct(self, level_idx: int) -> float:
        block = self._block_for_level(level_idx)
        if block in (2, 3):
            return float(self.peel_profit_pct.value)
        return 0.0

    def _peel_fraction(self, level_idx: int) -> float:
        block = self._block_for_level(level_idx)
        if block in (2, 3):
            return float(self.peel_fraction_shared.value)
        return 0.0

    def _level_can_peel(self, ledger: dict[str, Any], level_idx: int) -> bool:
        level = ledger["levels"][level_idx]
        return (
            self._level_is_open(ledger, level_idx)
            and bool(level.get("filled_once", False))
            and self._level_next_action(ledger, level_idx) == "sell"
            and self._peel_fraction(level_idx) > 0.0
        )

    def _level_can_rebuy(self, ledger: dict[str, Any], level_idx: int) -> bool:
        level = ledger["levels"][level_idx]
        if self._block_for_level(level_idx) == 1:
            return False
        return (
            bool(level.get("filled_once", False))
            and self._level_next_action(ledger, level_idx) == "buy"
        )

    def _next_unfilled_level(self, ledger: dict[str, Any]) -> int | None:
        for idx, level in enumerate(ledger["levels"]):
            if not bool(level.get("filled_once", False)):
                return idx
        return None

    def _level_anchor(self, ledger: dict[str, Any], level_idx: int) -> float | None:
        anchor = ledger["levels"][level_idx].get("anchor_price")
        return float(anchor) if anchor is not None else None

    def _level_next_action(self, ledger: dict[str, Any], level_idx: int) -> str:
        level = ledger["levels"][level_idx]
        next_action = str(level.get("next_action") or "buy").lower()
        return "sell" if next_action == "sell" else "buy"

    def _level_last_buy_size(self, ledger: dict[str, Any], level_idx: int) -> float:
        level = ledger["levels"][level_idx]
        last_buy = float(level.get("last_buy_size_quote") or 0.0)
        if last_buy > 0.0:
            return last_buy
        target = self._target_stake(ledger, level_idx)
        if target > 0.0:
            return target
        return self._live_stake(ledger, level_idx)

    def _vol_unit_pct(self, last_candle: pd.Series) -> float:
        atr_pct = self._num(last_candle, self._selected_atr_pct_col())
        if atr_pct is None:
            atr_pct = float(self.vol_unit_floor_pct.value)
        return float(np.clip(
            atr_pct,
            float(self.vol_unit_floor_pct.value),
            float(self.vol_unit_cap_pct.value),
        ))

    def _gap_pct(self, level_idx: int, last_candle: pd.Series) -> float:
        vol_unit = self._vol_unit_pct(last_candle)
        gap_mult = self._gap_mult_values()[level_idx] * self._regime_gap_mult_active(last_candle)
        min_gap = self._min_gap_values()[level_idx]
        return max(min_gap, vol_unit * gap_mult)

    def _ensure_next_anchor(self, ledger: dict[str, Any], last_candle: pd.Series) -> None:
        next_idx = self._next_unfilled_level(ledger)
        if next_idx is None or next_idx == 0:
            return
        if self._is_deep_drawdown_level(next_idx):
            return

        level = ledger["levels"][next_idx]
        if level["anchor_price"] is not None:
            return

        prev_anchor = self._level_anchor(ledger, next_idx - 1)
        if prev_anchor is None:
            return

        gap_pct = self._gap_pct(next_idx, last_candle)
        level["anchor_price"] = prev_anchor * (1.0 - gap_pct)

    def _deepest_filled_level(self, ledger: dict[str, Any]) -> int:
        deepest = 0
        for idx in range(self.LEVEL_COUNT):
            if bool(ledger["levels"][idx].get("filled_once", False)):
                deepest = idx + 1
        return deepest

    def _exit_target_for_level(self, level: int) -> float | None:
        if level <= 0:
            return None
        block = self._block_for_level(min(level, self.LEVEL_COUNT) - 1)
        if block == 1:
            return float(self.exit_target_b1.value)
        if block == 2:
            return float(self.exit_target_b2.value)
        return float(self.exit_target_b3.value)

    def _full_exit_target(self, ledger: dict[str, Any], last_candle: pd.Series, leverage: float) -> float | None:
        reached = self._deepest_filled_level(ledger)
        target_for_block = self._exit_target_for_level(reached)
        if target_for_block is None:
            return None

        lev = max(1.0, float(leverage or 1.0))
        target = target_for_block * lev * self._regime_exit_mult_active(last_candle)
        return float(np.clip(target, self.FULL_EXIT_MIN_PCT, self.FULL_EXIT_CAP_PCT))

    def _base_seed_profile(self, seed_pattern: str) -> dict[str, Any]:
        profiles = {
            "breakout": {
                "max_levels": 3, "stake_mult": 1.4, "spacing_mult": 0.6, "target_mult": 1.4,
                "peel_fraction_mult": 0.8, "peel_priority": "low", "add_allowed": True,
                "rebuy_allowed": False, "peel_allowed": True, "deep_recovery_allowed": False,
                "breakout_hold_required": True, "support_failure_exit_allowed": False,
            },
            "support_bounce": {
                "max_levels": 6, "stake_mult": 1.0, "spacing_mult": 1.0, "target_mult": 0.9,
                "peel_fraction_mult": 1.1, "peel_priority": "normal", "add_allowed": True,
                "rebuy_allowed": True, "peel_allowed": True, "deep_recovery_allowed": False,
                "breakout_hold_required": False, "support_failure_exit_allowed": True,
            },
            "support_reversal": {
                "max_levels": 6, "stake_mult": 0.8, "spacing_mult": 1.2, "target_mult": 0.9,
                "peel_fraction_mult": 1.2, "peel_priority": "high", "add_allowed": True,
                "rebuy_allowed": True, "peel_allowed": True, "deep_recovery_allowed": True,
                "breakout_hold_required": False, "support_failure_exit_allowed": True,
            },
            "crash_recovery": {
                "max_levels": 9, "stake_mult": 0.6, "spacing_mult": 1.8, "target_mult": 0.7,
                "peel_fraction_mult": 1.4, "peel_priority": "high", "add_allowed": True,
                "rebuy_allowed": True, "peel_allowed": True, "deep_recovery_allowed": True,
                "breakout_hold_required": False, "support_failure_exit_allowed": False,
            },
        }
        return dict(profiles.get(str(seed_pattern), {
            "max_levels": 0, "stake_mult": 1.0, "spacing_mult": 1.0, "target_mult": 1.0,
            "peel_fraction_mult": 1.0, "peel_priority": "high", "add_allowed": False,
            "rebuy_allowed": False, "peel_allowed": False, "deep_recovery_allowed": False,
            "breakout_hold_required": False, "support_failure_exit_allowed": False,
        }))

    def _timeframe_profile(self, seed_timeframe: str, seed_combo: bool) -> dict[str, Any]:
        if seed_combo or seed_timeframe == "combo_1d_1h":
            return {"stake_mult": 1.8, "spacing_mult": 1.4, "target_mult": 1.8}
        if seed_timeframe == "1d":
            return {"stake_mult": 1.5, "spacing_mult": 1.8, "target_mult": 1.5}
        if seed_timeframe == "1h":
            return {"stake_mult": 1.0, "spacing_mult": 1.0, "target_mult": 1.0}
        return {}

    def _regime_profile(self, regime: str) -> dict[str, Any]:
        if regime == "bull":
            return {"stake_mult": 1.2, "spacing_mult": 0.8, "peel_fraction_mult": 0.9}
        if regime == "bear":
            return {"stake_mult": 0.75, "spacing_mult": 1.5, "peel_fraction_mult": 1.4}
        return {"stake_mult": 1.0, "spacing_mult": 1.1, "peel_fraction_mult": 1.2}

    def _regime_profile_from_candle(self, last_candle: pd.Series) -> dict[str, Any]:
        stake_mult = self._num(last_candle, "regime_stake_mult_active")
        spacing_mult = self._num(last_candle, "regime_spacing_mult_active")
        peel_fraction_mult = self._num(last_candle, "regime_peel_fraction_mult_active")
        values = (stake_mult, spacing_mult, peel_fraction_mult)
        if all(value is not None and np.isfinite(value) and value > 0.0 for value in values):
            return {
                "stake_mult": float(stake_mult),
                "spacing_mult": float(spacing_mult),
                "peel_fraction_mult": float(peel_fraction_mult),
            }
        return self._regime_profile(self._regime(last_candle))

    def _danger_overlay(self, last_candle: pd.Series | dict[str, Any]) -> dict[str, Any]:
        if not self._is_extreme_condition(last_candle):
            return {}
        return {
            "add_allowed": False,
            "rebuy_allowed": False,
            "peel_allowed": True,
            "target_mult": 0.6,
            "peel_fraction_mult": 1.5,
        }

    def _merge_trade_profiles(self, *profiles: dict[str, Any]) -> dict[str, Any]:
        result = {
            "max_levels": 0, "stake_mult": 1.0, "spacing_mult": 1.0, "target_mult": 1.0,
            "peel_fraction_mult": 1.0, "peel_priority": "normal", "add_allowed": True,
            "rebuy_allowed": True, "peel_allowed": True, "deep_recovery_allowed": True,
            "breakout_hold_required": True, "support_failure_exit_allowed": True,
        }
        for profile in profiles:
            if not profile:
                continue
            for key, value in profile.items():
                if key in ("stake_mult", "spacing_mult", "target_mult", "peel_fraction_mult"):
                    result[key] = float(result.get(key, 1.0)) * float(value)
                elif key == "max_levels":
                    result[key] = int(value) if int(result.get(key, 0)) <= 0 else min(int(result.get(key, 0)), int(value))
                elif key in ("add_allowed", "rebuy_allowed", "peel_allowed", "deep_recovery_allowed", "breakout_hold_required", "support_failure_exit_allowed"):
                    result[key] = bool(result.get(key, True)) and bool(value)
                else:
                    result[key] = value
        return result

    def _effective_trade_profile(self, ledger: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
        entry_context = ledger.get("entry_context") if isinstance(ledger.get("entry_context"), dict) else self._default_entry_context()
        seed_pattern = str(entry_context.get("seed_pattern") or "unknown")
        seed_timeframe = str(entry_context.get("seed_timeframe") or "unknown")
        base = self._base_seed_profile(seed_pattern)
        if seed_pattern == "unknown":
            base["peel_allowed"] = any(self._live_stake(ledger, idx) > 0.0 for idx in range(self.LEVEL_COUNT))
        profile = self._merge_trade_profiles(
            base,
            self._timeframe_profile(seed_timeframe, bool(entry_context.get("seed_combo", False))),
            self._regime_profile_from_candle(view["last"]),
            self._context_profile(ledger, view),
            self._danger_overlay(view["last"]),
        )
        profile["max_levels"] = int(np.clip(int(profile.get("max_levels", 0)), 0, self.LEVEL_COUNT))
        for key in ("stake_mult", "spacing_mult", "target_mult", "peel_fraction_mult"):
            profile[key] = float(profile.get(key, 1.0))
        for key in ("add_allowed", "rebuy_allowed", "peel_allowed", "deep_recovery_allowed", "breakout_hold_required", "support_failure_exit_allowed"):
            profile[key] = bool(profile.get(key, False))
        return profile

    def _mean_reversion_ok(self, last_candle: pd.Series) -> bool:
        if not bool(self.use_add_close_gt_prev.value):
            return True
        return bool(last_candle.get("close_gt_prev_close", False))

    def _strong_downtrend(self, last_candle: pd.Series, adx_threshold: float) -> bool:
        adx = self._num(last_candle, "adx")
        plus_di = self._num(last_candle, "plus_di")
        minus_di = self._num(last_candle, "minus_di")
        if adx is None or plus_di is None or minus_di is None:
            return False
        return adx >= adx_threshold and minus_di > plus_di

    def _is_extreme_condition(self, last_candle: pd.Series) -> bool:
        """
        Detect exceptionally high levels of volume, speed, and downside trend.
        Used to block entries and force exits during potential crashes.
        """
        # 1. Very high volume ratio
        volume = self._num(last_candle, "volume")
        volume_ema = self._num(last_candle, self._selected_volume_ema_col())
        if volume is None or volume_ema is None or volume_ema == 0:
            vol_extreme = False
        else:
            vol_extreme = volume > (volume_ema * float(self.extreme_volume_threshold.value))

        # 2. High speed (movement relative to average volatility)
        # Using absolute pct change relative to EMA of ATR pct
        close_pct_change = abs(float(last_candle.get("close_pct_change", 0.0)))
        atr_pct_ema = self._num(last_candle, self._selected_atr_pct_ema_col())
        if atr_pct_ema is None or atr_pct_ema == 0:
            speed_extreme = False
        else:
            speed_extreme = (close_pct_change / atr_pct_ema) > float(self.extreme_speed_threshold.value)

        # 3. Strong downtrend
        downtrend_extreme = self._strong_downtrend(last_candle, float(self.extreme_downtrend_adx.value))

        # Combine: all three must be true for an "extreme" designation.
        # This catches the specific intersection of high volume + high speed + downtrend.
        return vol_extreme and speed_extreme and downtrend_extreme

    def _crash_recovery_confirmed(self, last_candle: pd.Series) -> bool:
        lookback = int(self.support_lookback.value)
        rsi = self._num(last_candle, "rsi")
        if rsi is not None and rsi >= float(self.crash_recovery_rsi_max.value):
            return False
        stabilizing = bool(last_candle.get("is_green", False)) and bool(last_candle.get("close_gt_prev_close", False))
        structural_recovery = bool(
            last_candle.get(f"support_reclaim_{lookback}", False)
            or last_candle.get(f"support_reversal_confirmed_{lookback}", False)
            or last_candle.get(f"support_bounce_confirmed_{lookback}", False)
            or last_candle.get("pair_1d_support_reversal_confirmed", False)
            or last_candle.get("pair_1d_support_bounce_confirmed", False)
        )
        if bool(self.crash_recovery_require_stabilizing.value) and not stabilizing:
            return False
        if bool(self.crash_recovery_use_volume_confirm.value) and not self._volume_ratio_at_least(last_candle, 1.0):
            return False
        if bool(self.crash_recovery_support_reclaim_required.value) and not structural_recovery:
            return False
        return bool(stabilizing or structural_recovery)

    def _unresolved_extreme_danger(self, last_candle: pd.Series) -> bool:
        return self._is_extreme_condition(last_candle) and not self._crash_recovery_confirmed(last_candle)

    def _crash_recovery_entry_allowed(self, last_candle: pd.Series) -> bool:
        return self._crash_recovery_confirmed(last_candle)

    def _normal_entry_blocked_by_extreme(self, last_candle: pd.Series) -> bool:
        return self._unresolved_extreme_danger(last_candle)

    def _rebuy_crash_guard_score(self, view: dict[str, Any]) -> int:
        """
        Light aggregate filter for rebuys.

        Rebuy is still driven by structural anchor revisit. This score only blocks
        obvious downside acceleration using volume, direction, momentum and volatility.
        """
        last_candle = view["last"]
        prev_candle = view["prev"]
        score = 0

        rsi_last = self._num(last_candle, "rsi")
        rsi_prev = self._num(prev_candle, "rsi")
        if rsi_last is not None and rsi_prev is not None and rsi_last >= rsi_prev:
            score += 1

        mfi_last = self._num(last_candle, "mfi")
        mfi_prev = self._num(prev_candle, "mfi")
        if mfi_last is not None and mfi_prev is not None and mfi_last >= mfi_prev:
            score += 1

        hist_last = self._num(last_candle, "macd_hist")
        hist_prev = self._num(prev_candle, "macd_hist")
        if hist_last is not None and hist_prev is not None and hist_last >= hist_prev:
            score += 1

        if not self._strong_downtrend(last_candle, float(self.rebuy_adx_crash_min.value)):
            score += 1

        atr_pct = self._num(last_candle, self._selected_atr_pct_col())
        atr_pct_ema = self._num(last_candle, self._selected_atr_pct_ema_col())
        if (
            atr_pct is not None
            and atr_pct_ema is not None
            and atr_pct <= atr_pct_ema * float(self.rebuy_atr_spike_limit.value)
        ):
            score += 1

        if bool(self.rebuy_use_volume_vote.value):
            volume = self._num(last_candle, "volume")
            volume_ema = self._num(last_candle, self._selected_volume_ema_col())
            if (
                volume is not None
                and volume_ema is not None
                and volume >= volume_ema * float(self.rebuy_volume_ratio_min.value)
            ):
                score += 1

        return score

    def _rebuy_not_crashing(self, view: dict[str, Any]) -> bool:
        if not bool(self.use_rebuy_crash_guard.value):
            return True
        return self._rebuy_crash_guard_score(view) >= int(self.rebuy_crash_min_votes.value)

    def _volume_ratio_at_least(self, last_candle: pd.Series, ratio: float) -> bool:
        volume = self._num(last_candle, "volume")
        volume_ema = self._num(last_candle, self._selected_volume_ema_col())
        if volume is None or volume_ema is None or volume_ema <= 0.0:
            return False
        return volume >= volume_ema * float(ratio)

    def _support_or_local_min_rebuy_ok(self, view: dict[str, Any]) -> bool:
        last_candle = view["last"]
        lookback = int(self.support_lookback.value)
        return bool(
            last_candle.get(f"holds_support_{lookback}", False)
            or last_candle.get(f"holds_local_min_with_volume_{lookback}", False)
            or (
                last_candle.get(f"touches_local_min_{lookback}", False)
                and last_candle.get("close_gt_prev_close", False)
            )
        )

    def _volume_confirmed_rebuy_ok(self, view: dict[str, Any]) -> bool:
        return self._volume_ratio_at_least(view["last"], float(self.rebuy_volume_ratio_min.value))

    def _resistance_or_local_max_peel_ok(self, view: dict[str, Any]) -> bool:
        last_candle = view["last"]
        lookback = int(self.support_lookback.value)
        resistance_distance = self._num(last_candle, f"resistance_distance_pct_{lookback}")
        return bool(
            last_candle.get(f"touches_local_max_{lookback}", False)
            or last_candle.get(f"rejects_local_max_{lookback}", False)
            or (resistance_distance is not None and resistance_distance <= self.MIN_GAP_PCT)
        )

    def _breakout_hold_no_peel(self, view: dict[str, Any]) -> bool:
        lookback = int(self.support_lookback.value)
        return bool(view["last"].get(f"breaks_local_max_with_volume_{lookback}", False))

    def _volume_confirmed_full_exit_ok(self, view: dict[str, Any]) -> bool:
        return self._volume_ratio_at_least(view["last"], 1.0)

    def _profile_invalidation_exit_tag(self, ledger: dict[str, Any], view: dict[str, Any], profile: dict[str, Any]) -> str | None:
        entry_context = ledger.get("entry_context") if isinstance(ledger.get("entry_context"), dict) else {}
        close = float(view.get("candle_close") or 0.0)
        if close <= 0.0:
            return None

        if bool(profile.get("breakout_hold_required", False)):
            resistance_ref = entry_context.get("entry_resistance_ref")
            if resistance_ref is not None and close < float(resistance_ref) * (1.0 - self.MIN_GAP_PCT):
                return "breakout_failed_exit"

        if bool(profile.get("support_failure_exit_allowed", False)):
            support_ref = entry_context.get("entry_support_ref")
            if support_ref is not None and close < float(support_ref) * (1.0 - self.MIN_GAP_PCT):
                return "support_failed_exit"

        return None

    def _deep_recovery_ready(self, level_idx: int, view: dict[str, Any]) -> bool:
        return self._rebuy_not_crashing(view)

    def _closed_candle_view(self, pair: str, trade: Trade, ledger: dict[str, Any]) -> dict[str, Any] | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or len(dataframe) < 3:
            return None

        last_candle = dataframe.iloc[-1].squeeze()
        prev_candle = dataframe.iloc[-2].squeeze()
        self._ensure_next_anchor(ledger, last_candle)

        regime = self._regime(last_candle)
        candle_id = self._current_candle_id(last_candle)
        candle_close = float(last_candle["close"])
        candle_low = float(last_candle["low"])

        # Update regime diagnostics
        if ledger.get("last_processed_candle") != candle_id:
            summary = ledger["regime_summary"]
            summary["regime_candle_counts"][regime] = summary["regime_candle_counts"].get(regime, 0) + 1
            ledger["last_processed_candle"] = candle_id

            if summary.get("seed_regime") is None:
                summary["seed_regime"] = regime

            deepest = self._max_filled_level_for_trade(trade)
            if deepest > summary.get("deepest_level_reached", 0):
                summary["deepest_level_reached"] = deepest
                summary["deepest_level_regime"] = regime

        return {
            "last": last_candle,
            "prev": prev_candle,
            "candle_id": candle_id,
            "candle_close": candle_close,
            "candle_low": candle_low,
            "candle_profit": float(trade.calc_profit_ratio(candle_close)),
            "candle_low_profit": float(trade.calc_profit_ratio(candle_low)),
            "mean_reversion_ok": self._mean_reversion_ok(last_candle),
            "regime": regime,
        }

    def _log_position_action(
        self,
        trade: Trade,
        action: dict[str, Any],
        view: dict[str, Any],
    ) -> None:
        level_idx = int(action.get("level", -1))
        logger.debug(
            "pair=%s trade_id=%s action=%s level=L%02d stake=%0.8f close=%0.8f candle_profit=%0.6f regime=%s candle_id=%s",
            trade.pair,
            getattr(trade, "id", None),
            action.get("type"),
            level_idx + 1,
            float(action.get("stake") or 0.0),
            float(view.get("candle_close") or 0.0),
            float(view.get("candle_profit") or 0.0),
            view.get("regime"),
            view.get("candle_id"),
        )

    def _log_position_fill(
        self,
        trade: Trade,
        action_type: str,
        level_idx: int,
        filled_quote: float,
        price: float,
        amount: float,
        target_stake: float,
        live_stake: float,
        cycle_count: int,
    ) -> None:
        logger.debug(
            "pair=%s trade_id=%s fill=%s level=L%02d filled_quote=%0.8f live_stake=%0.8f target_stake=%0.8f cycle_count=%d order_price=%0.8f order_amount=%0.8f",
            trade.pair,
            getattr(trade, "id", None),
            action_type,
            level_idx + 1,
            float(filled_quote),
            float(live_stake),
            float(target_stake),
            int(cycle_count),
            float(price),
            float(amount),
        )

    # ------------------------------------------------------------------
    # Rule engines
    # ------------------------------------------------------------------

    def _check_peel(
        self,
        ledger: dict[str, Any],
        view: dict[str, Any],
        min_stake: float | None,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(profile.get("peel_allowed", False)):
            return None
        # TBD: Optional event / CUSUM guards can be added here later.
        # Objective: only peel when the rebound has enough quality / impulse to justify
        # reducing inventory, without forcing unnecessary delay during normal recoveries.
        #
        # Example later guard:
        # require upward impulse or bullish CUSUM accumulation before peeling.

        floor_stake = self._stake_floor_from_min(min_stake)

        for idx in range(self.LEVEL_COUNT - 1, -1, -1):
            if not self._level_can_peel(ledger, idx):
                continue

            anchor = self._level_anchor(ledger, idx)
            if anchor is None:
                continue

            trigger = anchor * (1.0 + self._peel_profit_pct(idx))
            if view["candle_close"] < trigger:
                continue
            if bool(self.enable_resistance_local_max_peel.value) and not self._resistance_or_local_max_peel_ok(view):
                continue
            if bool(self.enable_breakout_hold_no_peel.value) and self._breakout_hold_no_peel(view):
                continue

            level_live = self._live_stake(ledger, idx)
            last_buy_size = self._level_last_buy_size(ledger, idx)
            peel_stake = min(
                level_live,
                last_buy_size * self._peel_fraction(idx) * float(profile.get("peel_fraction_mult", 1.0)),
            )
            if peel_stake <= 0.0:
                continue
            if floor_stake is not None:
                # Keep peel/rebuy cycle exchange-valid: peel actions should not create
                # sub-minimum refill requirements.
                peel_stake = min(level_live, max(peel_stake, floor_stake))
                if peel_stake < floor_stake:
                    continue

            logger.debug("_check_peel")
            return {
                "type": "peel",
                "level": idx,
                "stake": -peel_stake,
                "tag": f"peel_L{idx + 1:02d}",
            }
        return None

    def _check_add(
        self,
        ledger: dict[str, Any],
        view: dict[str, Any],
        min_stake: float | None,
        max_stake: float,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        # TBD: Optional event / CUSUM guards can be added here later.
        # Objective: test whether new level adds should require volatility-normalized
        # impulse / expansion checks, or whether structural price location alone is best.
        #
        # Example later guard:
        # reject adds unless downside move is meaningfully extended relative to ATR%.

        next_idx = self._next_unfilled_level(ledger)
        if next_idx is None or next_idx == 0:
            self._bump_exec_counter(ledger, "check_add_reject_next_level")
            return None
        if next_idx + 1 > int(profile.get("max_levels", 0)):
            self._bump_exec_counter(ledger, "check_add_reject_next_level")
            return None

        if self._is_deep_drawdown_level(next_idx):
            if not bool(profile.get("deep_recovery_allowed", False)):
                self._bump_exec_counter(ledger, "check_add_reject_deep_recovery")
                return None
            target_drawdown = self._deep_drawdown_target(next_idx)
            if view["candle_low_profit"] > -target_drawdown:
                self._bump_exec_counter(ledger, "check_add_reject_deep_drawdown")
                return None

            prev_anchor = self._level_anchor(ledger, next_idx - 1)
            if prev_anchor is None:
                self._bump_exec_counter(ledger, "check_add_reject_anchor_missing")
                return None

            if view["candle_close"] > prev_anchor * (1.0 - self.DEEP_PREV_LEVEL_BUFFER_PCT):
                self._bump_exec_counter(ledger, "check_add_reject_prev_level_buffer")
                return None

            if not self._deep_recovery_ready(next_idx, view):
                self._bump_exec_counter(ledger, "check_add_reject_deep_recovery")
                return None
        else:
            if not view["mean_reversion_ok"]:
                self._bump_exec_counter(ledger, "check_add_reject_mean_reversion")
                return None

            anchor = self._level_anchor(ledger, next_idx)
            if anchor is None:
                self._bump_exec_counter(ledger, "check_add_reject_anchor_missing")
                return None

            adjusted_anchor = anchor
            prev_anchor = self._level_anchor(ledger, next_idx - 1)
            if prev_anchor is not None and prev_anchor > 0.0:
                base_gap = max(0.0, 1.0 - (anchor / prev_anchor))
                adjusted_anchor = prev_anchor * (1.0 - (base_gap * float(profile.get("spacing_mult", 1.0))))

            if view["candle_close"] > adjusted_anchor:
                self._bump_exec_counter(ledger, "check_add_reject_close_above_anchor")
                return None

        stake = self._target_stake(ledger, next_idx)
        if stake <= 0.0:
            self._bump_exec_counter(ledger, "check_add_reject_stake_nonpositive")
            return None

        floor_stake = self._stake_floor_from_min(min_stake)
        if floor_stake is not None:
            if float(max_stake) < floor_stake:
                self._bump_exec_counter(ledger, "check_add_reject_stake_above_max")
                return None
            stake = max(stake, floor_stake)

        # Keep level accounting coherent: never exceed intended target level size.
        if stake > self._target_stake(ledger, next_idx):
            self._bump_exec_counter(ledger, "check_add_reject_stake_below_min")
            return None
        if stake > float(max_stake):
            self._bump_exec_counter(ledger, "check_add_reject_stake_above_max")
            return None

        logger.debug("_check_add")
        return {
            "type": "add",
            "level": next_idx,
            "stake": stake,
            "tag": f"add_L{next_idx + 1:02d}",
        }

    def _check_rebuy(
        self,
        ledger: dict[str, Any],
        view: dict[str, Any],
        min_stake: float | None,
        max_stake: float,
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        # TBD: Optional event / CUSUM guards can be added here later.
        # Objective: test whether rebuys improve when anchor revisits also require a
        # volatility-normalized move / CUSUM accumulation / volume expansion signal.
        #
        # Example later guard:
        # require downside event exhaustion or positive reversal impulse before refilling.
        if not bool(profile.get("rebuy_allowed", False)):
            return None
        if not self._rebuy_not_crashing(view):
            return None
        if bool(self.enable_support_local_min_rebuy.value) and not self._support_or_local_min_rebuy_ok(view):
            return None
        if bool(self.enable_volume_confirmed_rebuy.value) and not self._volume_confirmed_rebuy_ok(view):
            return None

        floor_stake = self._stake_floor_from_min(min_stake)
        if floor_stake is not None and float(max_stake) < floor_stake:
            return None

        for idx in range(self.LEVEL_COUNT - 1, -1, -1):
            if self._block_for_level(idx) == 1:
                continue
            if not self._level_can_rebuy(ledger, idx):
                continue

            anchor = self._level_anchor(ledger, idx)
            if anchor is None:
                continue

            if view["candle_close"] > anchor:
                continue

            # Rebuy reuses the exact last buy/rebuy size for this anchor level.
            rebuy_stake = self._level_last_buy_size(ledger, idx)
            if rebuy_stake <= 0.0:
                continue

            if floor_stake is not None:
                rebuy_stake = max(rebuy_stake, floor_stake)

            if rebuy_stake > float(max_stake):
                continue

            logger.debug("_check_rebuy")
            return {
                "type": "rebuy",
                "level": idx,
                "stake": rebuy_stake,
                "tag": f"rebuy_L{idx + 1:02d}",
            }
        return None

    def _select_action(
        self,
        trade: Trade,
        ledger: dict[str, Any],
        view: dict[str, Any],
        min_stake: float | None,
        max_stake: float,
    ) -> dict[str, Any] | None:
        """
        Priority order for position adjustments only:
        1. Peel deepest eligible open level on strength.
        2. Rebuy a previously peeled level on revisit of its anchor, but only if
           the revisit is not happening during obvious crash continuation.
        3. Add the next new level if structure extends lower.
        """
        if ledger.get("last_action_candle") == view["candle_id"]:
            return None

        pending = ledger.get("pending_action")
        if pending and (not trade.has_open_orders) and pending.get("candle_id") != view["candle_id"]:
            ledger["pending_action"] = None

        profile = self._effective_trade_profile(ledger, view)

        self._bump_exec_counter(ledger, "check_peel_calls")
        action = self._check_peel(ledger, view, min_stake, profile)
        if action:
            self._bump_exec_counter(ledger, "check_peel_hits")
            return action

        if not bool(profile.get("add_allowed", False)) and not bool(profile.get("rebuy_allowed", False)):
            return None

        if bool(profile.get("rebuy_allowed", False)):
            self._bump_exec_counter(ledger, "check_rebuy_calls")
            action = self._check_rebuy(ledger, view, min_stake, max_stake, profile)
            if action:
                self._bump_exec_counter(ledger, "check_rebuy_hits")
                return action

        if bool(profile.get("add_allowed", False)):
            self._bump_exec_counter(ledger, "check_add_calls")
            action = self._check_add(ledger, view, min_stake, max_stake, profile)
            if action:
                self._bump_exec_counter(ledger, "check_add_hits")
                return action

        return None

    # ------------------------------------------------------------------
    # Freqtrade callbacks
    # ------------------------------------------------------------------

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        _ = current_time
        _ = current_rate
        _ = proposed_leverage
        _ = side
        _ = kwargs

        base_target = float(self.leverage_target.value)
        seed_context = self._seed_context_from_enter_tag(entry_tag)
        seed_pattern = str(seed_context.get("seed_pattern") or "unknown")
        last_candle = self._latest_analyzed_candle(pair)

        regime = "chop"
        if last_candle is not None:
            regime = self._regime(last_candle)

        dynamic_leverage = (
            base_target
            * self._leverage_regime_mult(regime)
            * self._leverage_seed_mult(seed_pattern)
        )
        if last_candle is not None:
            dynamic_leverage *= self._leverage_vol_brake_mult(last_candle)
        if not np.isfinite(dynamic_leverage):
            dynamic_leverage = 1.0

        regime_cap = self._leverage_cap(regime, seed_pattern)
        if max_leverage is None or float(max_leverage) <= 0.0:
            max_allowed = regime_cap
        else:
            max_allowed = min(regime_cap, float(max_leverage))

        return float(np.clip(dynamic_leverage, 1.0, max(1.0, max_allowed)))

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str = "long",
        **kwargs,
    ) -> bool:
        _ = order_type
        _ = amount
        _ = rate
        _ = time_in_force
        _ = current_time
        _ = kwargs

        if side != "long":
            return False

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and not dataframe.empty:
            last_candle = dataframe.iloc[-1].squeeze()
            if entry_tag == "crash_recovery_seed":
                if not self._crash_recovery_entry_allowed(last_candle):
                    logger.debug(f"Crash recovery entry rejected: no recovery confirmation for {pair}")
                    return False
            elif self._normal_entry_blocked_by_extreme(last_candle):
                logger.debug(f"Entry blocked: extreme condition detected for {pair}")
                return False

        open_trades = self._open_trades_snapshot()
        allowed = self._dynamic_allowed_open_trades(open_trades)
        open_count = len(open_trades)

        if open_count >= allowed:
            depths = [self._max_filled_level_for_trade(t) for t in open_trades]
            level7_pressure_count = sum(1 for d in depths if d >= self.OPEN_TRADE_CAP_UNLOCK_LEVEL_1)
            logger.debug(
                "entry_cap_block pair=%s open=%d allowed=%d level7_count=%d",
                pair,
                open_count,
                allowed,
                level7_pressure_count,
            )
            return False

        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        _ = pair
        _ = current_time
        _ = current_rate
        _ = leverage
        _ = side
        _ = kwargs

        seed_context = self._seed_context_from_enter_tag(entry_tag)
        profile = self._merge_trade_profiles(
            self._base_seed_profile(str(seed_context.get("seed_pattern") or "unknown")),
            self._timeframe_profile(str(seed_context.get("seed_timeframe") or "unknown"), bool(seed_context.get("seed_combo", False))),
        )
        min_entry_stake = self._effective_min_entry_stake(proposed_stake * float(profile.get("stake_mult", 1.0)), min_stake, max_stake)
        wallet_total_quote = float(self.wallets.get_total_stake_amount())
        planned_stakes = self._build_planned_stakes(min_entry_stake, wallet_total_quote)
        return float(planned_stakes[0])

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        Full-exit decision ladder.

        Use average trade entry (`trade.open_rate`) implicitly through Freqtrade's
        profit calculation, but evaluate against the latest closed candle so the
        strategy stays consistent with closed-candle-only management.
        """
        _ = current_time
        _ = current_rate
        _ = current_profit
        _ = kwargs

        ledger = self._get_ledger(trade)
        if not bool(ledger["levels"][0].get("filled_once", False)):
            return None

        view = self._closed_candle_view(pair, trade, ledger)
        if view is None:
            return None
        profile = self._effective_trade_profile(ledger, view)

        # TBD: Optional event / CUSUM guards can be added here later.
        # Objective: test whether full exits should require rebound-quality confirmation
        # instead of taking every average-entry-plus-target touch immediately.
        #
        # Example later guard:
        # only allow full exit if rebound impulse or positive CUSUM accumulation confirms.

        # Force full exit during extreme crash conditions
        if self._is_extreme_condition(view["last"]):
            reached = self._deepest_filled_level(ledger)
            regime = view["regime"]
            summary = ledger["regime_summary"]
            summary["exit_regime"] = regime
            summary["action_regime_counts"]["full_exit"][regime] = summary["action_regime_counts"]["full_exit"].get(regime, 0) + 1
            self._set_ledger(trade, ledger)
            logger.debug(f"Forced exit: extreme condition detected for {pair}")
            return f"extreme_exit_L{min(reached, self.LEVEL_COUNT):02d}"

        invalidation_tag = self._profile_invalidation_exit_tag(ledger, view, profile)
        if invalidation_tag:
            regime = view["regime"]
            summary = ledger["regime_summary"]
            summary["exit_regime"] = regime
            summary["action_regime_counts"]["full_exit"][regime] = summary["action_regime_counts"]["full_exit"].get(regime, 0) + 1
            self._set_ledger(trade, ledger)
            return invalidation_tag

        target = self._full_exit_target(ledger, view["last"], float(getattr(trade, "leverage", 1.0) or 1.0))
        if target is None:
            return None
        target = float(np.clip(target * float(profile.get("target_mult", 1.0)), self.FULL_EXIT_MIN_PCT, self.FULL_EXIT_CAP_PCT))

        if view["candle_profit"] >= target:
            if bool(self.enable_volume_confirmed_full_exit.value) and not self._volume_confirmed_full_exit_ok(view):
                self._set_ledger(trade, ledger)
                return None

            reached = self._deepest_filled_level(ledger)
            # Update regime summary for full_exit
            regime = view["regime"]
            summary = ledger["regime_summary"]
            summary["exit_regime"] = regime
            summary["action_regime_counts"]["full_exit"][regime] = summary["action_regime_counts"]["full_exit"].get(regime, 0) + 1

            self._set_ledger(trade, ledger)
            logger.debug("_custom_exit-full_exit")
            return f"full_exit_L{min(reached, self.LEVEL_COUNT):02d}"

        self._set_ledger(trade, ledger)
        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        _ = current_time
        _ = current_rate
        _ = current_profit
        _ = current_entry_rate
        _ = current_exit_rate
        _ = current_entry_profit
        _ = current_exit_profit
        _ = kwargs

        ledger = self._get_ledger(trade)
        self._bump_exec_counter(ledger, "adjust_calls")

        if trade.has_open_orders:
            self._set_ledger(trade, ledger)
            return None
        if trade.is_short:
            self._set_ledger(trade, ledger)
            return None

        # Ensure all levels can place exchange-valid orders for this market min-stake.
        self._ensure_min_planned_stakes(ledger, min_stake, max_stake)

        if not bool(ledger["levels"][0].get("filled_once", False)):
            self._set_ledger(trade, ledger)
            return None

        view = self._closed_candle_view(trade.pair, trade, ledger)
        if view is None:
            self._set_ledger(trade, ledger)
            return None

        action = self._select_action(trade, ledger, view, min_stake, max_stake)
        if not action:
            self._bump_exec_counter(ledger, "adjust_no_action")
            self._set_ledger(trade, ledger)
            counters = self._exec_counters(ledger)
            if int(counters["adjust_calls"]) % 25 == 0:
                self._log_exec_counters(trade, ledger, "adjust_no_action_snapshot")
            return None

        action_type = str(action.get("type") or "")
        if action_type in ("add", "peel", "rebuy"):
            self._bump_exec_counter(ledger, f"adjust_selected_{action_type}")
            # Track regime action
            regime = view["regime"]
            summary = ledger["regime_summary"]
            if action_type in summary["action_regime_counts"]:
                summary["action_regime_counts"][action_type][regime] = summary["action_regime_counts"][action_type].get(regime, 0) + 1

        ledger["last_action_candle"] = view["candle_id"]
        ledger["pending_action"] = {
            "type": action["type"],
            "level": action.get("level"),
            "stake": action.get("stake"),
            "candle_id": view["candle_id"],
        }
        self._set_ledger(trade, ledger)
        self._log_position_action(trade, action, view)
        self._log_exec_counters(trade, ledger, f"adjust_action_{action_type or 'unknown'}")
        logger.debug("_adjust_trade_position-action")
        return action["stake"], action["tag"]

    def custom_stoploss(
            self,
            pair: str,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            after_fill: bool,
            **kwargs,
    ) -> float | None:
        """
        Dynamic max-loss stop based on the trade's current averaged entry price.

        Behaviour
        ---------
        - Always active.
        - If no adds occur, this behaves like a normal loss stop from the original entry.
        - If adds occur, trade.open_rate shifts to the new averaged position entry,
          so the stop automatically shifts as well.
        - This is not tied to any ladder level being reached.
        """
        _ = pair
        _ = current_time
        _ = current_profit
        _ = after_fill
        _ = kwargs

        avg_entry = float(trade.open_rate or 0.0)
        if avg_entry <= 0.0:
            return None

        stop_price = avg_entry * (1.0 - float(self.l9_stoploss_pct.value))
        if stop_price <= 0.0:
            return None

        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def _log_trade_summary(self, trade: Trade, ledger: dict[str, Any], view: dict[str, Any]) -> None:
        summary = ledger.get("regime_summary", {})
        counts = summary.get("regime_candle_counts", {})
        actions = summary.get("action_regime_counts", {})

        total_candles = sum(counts.values())
        dominant_regime = "none"
        if total_candles > 0:
            dominant_regime = max(counts, key=counts.get)

        seed = summary.get("seed_regime", "N/A")
        exit_regime = view.get("regime", "N/A")
        deepest = summary.get("deepest_level_reached", 0)
        profit = view.get("candle_profit", 0.0)

        logger.debug(
            f"TRADE_SUMMARY: pair={trade.pair} id={getattr(trade, 'id', 'N/A')} "
            f"seed={seed} exit={exit_regime} dominant={dominant_regime} "
            f"deepest_L={deepest} profit={profit:.4f} "
            f"candles(bull={counts.get('bull',0)} bear={counts.get('bear',0)} chop={counts.get('chop',0)}) "
            f"actions(add={sum(actions.get('add', {}).values())} rebuy={sum(actions.get('rebuy', {}).values())} peel={sum(actions.get('peel', {}).values())})"
        )

    def _log_aggregate_summary(self) -> None:
        logger.debug("AGGREGATE_REGIME_SUMMARY:")
        for regime, stats in self._aggregate_regime_stats.items():
            if stats["trades"] > 0:
                winrate = stats["wins"] / stats["trades"]
                avg_profit = stats["profit"] / stats["trades"]
                avg_deepest = stats["deepest_level"] / stats["trades"]
                logger.debug(
                    f"  {regime.upper()}: trades={stats['trades']} winrate={winrate:.2%} "
                    f"avg_profit={avg_profit:.4f} avg_deepest_L={avg_deepest:.2f}"
                )

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        ledger = self._get_ledger(trade)
        view = self._closed_candle_view(pair, trade, ledger)
        if view:
            self._log_trade_summary(trade, ledger, view)

            # Update aggregate stats
            summary = ledger.get("regime_summary", {})
            seed = summary.get("seed_regime")
            if seed in self._aggregate_regime_stats:
                stats = self._aggregate_regime_stats[seed]
                stats["trades"] += 1
                profit = view["candle_profit"]
                stats["profit"] += profit
                if profit > 0:
                    stats["wins"] += 1
                stats["deepest_level"] += summary.get("deepest_level_reached", 0)

            # Periodically or at end log aggregate
            if self.config.get("runmode") in ("backtest", "hyperopt"):
                self._log_aggregate_summary()

        return True

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """
        Update the ledger from actual fills where practical.

        Notes on accounting:
        - Entry fills use the filled quote amount from price * amount.
        - Partial exits reduce level inventory by the negative stake amount that was
          requested in adjust_trade_position(). This keeps the internal level ledger
          aligned with Freqtrade's partial-exit stake model.
        - The initial ladder rebuild uses an estimate of pre-entry wallet size:
          current wallet + first filled seed stake. This avoids non-persistent caches
          while staying closer to the intended wallet-bucket logic.
        """
        _ = pair
        _ = current_time
        _ = kwargs

        ledger = self._get_ledger(trade)
        price = self._safe_order_price(order)
        amount = self._safe_order_amount(order)
        filled_quote = max(0.0, price * amount)

        if (
            not bool(ledger["levels"][0].get("filled_once", False))
            and order.ft_order_side == trade.entry_side
            and trade.nr_of_successful_entries == 1
        ):
            ledger = self._default_ledger()
            wallet_total_quote = float(self.wallets.get_total_stake_amount()) + filled_quote
            planned_stakes = self._build_planned_stakes(filled_quote, wallet_total_quote)
            ledger["planned_stakes_quote"] = [float(x) for x in planned_stakes]
            last_candle = None
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                if dataframe is not None and not dataframe.empty:
                    last_candle = dataframe.iloc[-1].squeeze()
            except Exception as exc:
                logger.debug("Could not build entry_context for %s: %s", trade.pair, exc)
            ledger["entry_context"] = self._build_entry_context(trade, last_candle)

            level0 = ledger["levels"][0]
            level0["anchor_price"] = price
            level0["live_stake_quote"] = filled_quote
            level0["filled_once"] = True
            level0["last_buy_size_quote"] = filled_quote
            level0["next_action"] = "buy"

            self._set_ledger(trade, ledger)
            return None

        pending = ledger.get("pending_action")
        if not pending:
            return None

        action_type = pending.get("type")
        level_idx = pending.get("level")
        pending_stake = abs(float(pending.get("stake") or 0.0))

        if level_idx is None:
            ledger["pending_action"] = None
            self._set_ledger(trade, ledger)
            return None

        level = ledger["levels"][level_idx]
        target_stake = self._target_stake(ledger, level_idx)

        if action_type == "add":
            actual_quote = filled_quote if filled_quote > 0.0 else pending_stake
            if level.get("anchor_price") is None:
                level["anchor_price"] = float(price) if price > 0.0 else trade.open_rate
            level["live_stake_quote"] = max(0.0, self._live_stake(ledger, level_idx) + actual_quote)
            level["filled_once"] = True
            level["last_buy_size_quote"] = actual_quote
            level["next_action"] = "sell" if self._peel_fraction(level_idx) > 0.0 else "buy"
            self._bump_exec_counter(ledger, "fill_applied_add")

        elif action_type == "peel":
            level["live_stake_quote"] = max(0.0, self._live_stake(ledger, level_idx) - pending_stake)
            level["next_action"] = "buy"
            self._bump_exec_counter(ledger, "fill_applied_peel")

        elif action_type == "rebuy":
            actual_quote = filled_quote if filled_quote > 0.0 else pending_stake
            level["live_stake_quote"] = max(0.0, self._live_stake(ledger, level_idx) + actual_quote)
            level["last_buy_size_quote"] = actual_quote
            level["next_action"] = "sell"
            level["cycle_count"] = int(level.get("cycle_count", 0)) + 1
            self._bump_exec_counter(ledger, "fill_applied_rebuy")

        self._log_position_fill(
            trade=trade,
            action_type=str(action_type),
            level_idx=int(level_idx),
            filled_quote=float(filled_quote if filled_quote > 0.0 else pending_stake),
            price=float(price),
            amount=float(amount),
            target_stake=float(target_stake),
            live_stake=float(self._live_stake(ledger, level_idx)),
            cycle_count=int(level.get("cycle_count", 0)),
        )

        ledger["pending_action"] = None
        self._set_ledger(trade, ledger)
        self._log_exec_counters(trade, ledger, f"order_filled_{action_type}")
        return None


_apply_grouped_hyperopt_surface(HybridRecoveryGridStrategy)
HybridRecoveryGridStrategy.PARAM_TAG_MAP = collect_tagged_parameters(HybridRecoveryGridStrategy)
