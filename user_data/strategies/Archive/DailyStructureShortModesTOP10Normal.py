from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_absolute,
)

from entry_sieve_tools import apply_explicit_hyperopt_surface, entry_sieve_minimal_roi, entry_sieve_stoploss

logger = logging.getLogger(__name__)


def tagged_parameter(param: Any, *tags: str) -> Any:
    family_map = {
        "entries": "entries",
        "exits": "exits",
        "adjust_position": "adjust_position",
        "stake": "stake",
        "risk": "risk",
    }
    normalized: list[str] = []
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag or ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "family":
            mapped = family_map.get(value.lower())
            if mapped:
                normalized.append(f"family:{mapped}")
        elif key == "mode" and value:
            normalized.append(f"mode:{value}")
    deduped: list[str] = []
    for tag in normalized:
        if tag not in deduped:
            deduped.append(tag)
    setattr(param, "batch_tags", tuple(deduped))
    return param


def _is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def _short_modes_parameter_family_mode(name: str) -> tuple[str, str] | None:
    entries_structure = {
        "d1_level_lookback",
        "h1_local_lookback",
        "d1_zone_atr_mult",
        "d1_zone_pct_min",
        "h1_level_near_daily_pct",
        "breakout_buffer_pct",
    }
    entries_volume = {
        "volume_window",
        "volume_gate_mode",
        "volume_ratio_min",
        "pressure_min",
    }
    entries_signal = {
        "enable_long_res_break",
        "enable_short_res_fail",
        "enable_long_sup_hold",
        "enable_short_sup_break",
        "enable_long_sup_reclaim",
        "enable_short_res_reclaim",
        "enable_long_res_retest_hold",
        "enable_short_sup_retest_reject",
        "long_breakout_confirm_mode",
        "short_break_confirm_mode",
        "long_support_mode",
        "long_breakout_retest_bars",
        "short_break_retest_bars",
        "short_res_fail_mode",
        "short_fail_volume_mode",
        "short_fail_wick_min_pct",
        "reclaim_lookback_bars",
        "reclaim_buffer_pct",
        "reclaim_require_volume_pressure",
        "entry_retest_buffer_pct",
        "entry_retest_lookback_bars",
        "retest_require_volume_pressure",
        "short_res_reclaim_mode",
        "short_sup_retest_reject_mode",
        "short_reclaim_volume_mode",
        "short_retest_volume_mode",
        "short_close_location_max",
        "short_break_close_location_max",
        "short_min_body_fraction",
        "short_acceptance_bars",
    }
    stake_params = {"leverage_opt", "stake_fraction"}
    adjust_params = {
        "add1_mult",
        "add2_mult",
        "enable_long_adds",
        "enable_short_adds",
        "add_retest_buffer_pct",
        "enable_volume_size_scaling",
        "volume_size_mult_max",
    }
    exits_core = {
        "peel1_fraction",
        "peel2_fraction",
        "peel_min_profit_pct",
        "peel_at_h1_level",
        "full_exit_at_d1_opposite_level",
        "stop_level_buffer_pct",
        "trail_after_profit_pct",
        "trail_level_buffer_pct",
    }
    exits_composed = {
        "enable_entry_family_exit_profiles",
        "continuation_exit_profile",
        "reversal_exit_profile",
        "enable_targeted_d1_exit",
        "targeted_d1_exit_buffer_pct",
        "targeted_d1_trail_buffer_pct",
    }
    risk_params = {
        "enable_portfolio_risk_overlay",
        "portfolio_max_gross_exposure_pct",
        "portfolio_drawdown_entry_block_pct",
        "portfolio_disable_adds_on_risk",
    }

    if name in entries_structure:
        return ("entries", "entry_structure_core")
    if name in entries_volume:
        return ("entries", "entry_volume_gate")
    if name in entries_signal:
        return ("entries", "entry_short_mode_stack")
    if name in stake_params:
        return ("stake", "stake_base")
    if name in adjust_params:
        return ("adjust_position", "adjust_add_scale")
    if name in exits_core:
        return ("exits", "exit_peel_trail")
    if name in exits_composed:
        return ("exits", "exit_targeted_with_profiles")
    if name in risk_params:
        return ("risk", "risk_portfolio_overlay")
    return None



class DailyStructureShortModesTOP10Normal(IStrategy):
    """
    Short-focused daily-structure / hourly-price-action hyperopt strategy.

    Design intent
    -------------
    - Trade on 1h candles.
    - Use completed same-pair 1d candles to define the major support/resistance zones.
    - Use 1h local highs/lows and directional volume pressure to decide whether the
      daily level was accepted, rejected, held, or broken.
    - Keep the management ledger tiny: one seed, two optional adds, two optional peels.
    - Short entries only for this research file. Long-side indicator and management code remains
      structurally compatible, but populate_entry_trend only emits short signals.
    - Use plain Freqtrade hyperopt parameters. Mode parameters include "none" so
      hyperopt can disable a mode without a separate enable switch.

    Entry reasons
    -------------
    - long_res_break:  resistance breakout acceptance.
    - short_res_fail:  failed resistance breakout / rejection.
    - long_sup_hold:   support hold or undercut/reclaim.
    - short_sup_break: support breakdown acceptance.
    - long_sup_reclaim: support loss that failed to accept lower, then reclaimed.
    - short_res_reclaim: resistance break that failed to accept higher, then reclaimed lower.
    - long_res_retest_hold: old resistance retested as support after a break.
    - short_sup_retest_reject: old support retested as resistance after a break.
    """

    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "1h"
    startup_candle_count = 600
    process_only_new_candles = True

    use_exit_signal = False
    use_custom_stoploss = False
    use_custom_roi = False
    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)

    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    # ------------------------------------------------------------------
    # Finite indicator choices.
    #
    # Lookbacks are categorical instead of free integer ranges so the indicator
    # variants are all precomputed once. Entry/management logic then selects the
    # active column by parameter value without relying on indicators being
    # recalculated for every hyperopt epoch.
    # ------------------------------------------------------------------

    D1_LEVEL_LOOKBACK_CHOICES = [10, 20, 30, 40, 60]
    H1_LOCAL_LOOKBACK_CHOICES = [6, 12, 24, 36, 48]
    VOLUME_WINDOW_CHOICES = [6, 12, 24, 36, 48]

    # ------------------------------------------------------------------
    # Hyperopt: structure detection
    # ------------------------------------------------------------------

    d1_level_lookback = CategoricalParameter(D1_LEVEL_LOOKBACK_CHOICES, default=20, space="buy", optimize=True, load=True)
    h1_local_lookback = CategoricalParameter(H1_LOCAL_LOOKBACK_CHOICES, default=12, space="buy", optimize=True, load=True)
    d1_zone_atr_mult = DecimalParameter(0.05, 0.50, decimals=2, default=0.20, space="buy", optimize=True, load=True)
    d1_zone_pct_min = DecimalParameter(0.001, 0.020, decimals=3, default=0.005, space="buy", optimize=True, load=True)
    h1_level_near_daily_pct = DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True)
    breakout_buffer_pct = DecimalParameter(0.000, 0.020, decimals=3, default=0.003, space="buy", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: volume pressure
    # ------------------------------------------------------------------

    volume_window = CategoricalParameter(VOLUME_WINDOW_CHOICES, default=12, space="buy", optimize=True, load=True)
    volume_gate_mode = CategoricalParameter(["none", "ratio_only", "pressure_only", "ratio_and_pressure"], default="ratio_and_pressure", space="buy", optimize=True, load=True)
    volume_ratio_min = DecimalParameter(0.8, 3.0, decimals=1, default=1.2, space="buy", optimize=True, load=True)
    pressure_min = DecimalParameter(0.50, 0.80, decimals=2, default=0.58, space="buy", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: entry enables and confirmation modes
    # ------------------------------------------------------------------

    enable_long_res_break = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    enable_short_res_fail = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    enable_long_sup_hold = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    enable_short_sup_break = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    enable_long_sup_reclaim = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    enable_short_res_reclaim = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    enable_long_res_retest_hold = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    enable_short_sup_retest_reject = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    long_breakout_confirm_mode = CategoricalParameter(["close_break", "break_retest"], default="close_break", space="buy", optimize=False, load=False)
    short_break_confirm_mode = CategoricalParameter(["none", "close_break", "break_retest"], default="close_break", space="buy", optimize=True, load=True)
    long_support_mode = CategoricalParameter(["touch_hold", "undercut_reclaim"], default="undercut_reclaim", space="buy", optimize=False, load=False)
    long_breakout_retest_bars = IntParameter(2, 12, default=6, space="buy", optimize=False, load=False)
    short_break_retest_bars = IntParameter(2, 12, default=6, space="buy", optimize=True, load=True)
    short_res_fail_mode = CategoricalParameter(["none", "wick_reject", "local_reject", "wick_or_local"], default="wick_or_local", space="buy", optimize=True, load=True)
    short_fail_volume_mode = CategoricalParameter(["none", "pressure"], default="pressure", space="buy", optimize=True, load=True)
    short_fail_wick_min_pct = DecimalParameter(0.001, 0.030, decimals=3, default=0.005, space="buy", optimize=True, load=True)
    reclaim_lookback_bars = IntParameter(2, 24, default=8, space="buy", optimize=True, load=True)
    reclaim_buffer_pct = DecimalParameter(0.001, 0.030, decimals=3, default=0.008, space="buy", optimize=True, load=True)
    reclaim_require_volume_pressure = BooleanParameter(default=True, space="buy", optimize=False, load=False)
    entry_retest_buffer_pct = DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True)
    entry_retest_lookback_bars = IntParameter(2, 24, default=8, space="buy", optimize=True, load=True)
    retest_require_volume_pressure = BooleanParameter(default=True, space="buy", optimize=False, load=False)

    short_res_reclaim_mode = CategoricalParameter(["none", "trap_reclaim"], default="trap_reclaim", space="buy", optimize=True, load=True)
    short_sup_retest_reject_mode = CategoricalParameter(["none", "retest_reject"], default="retest_reject", space="buy", optimize=True, load=True)
    short_reclaim_volume_mode = CategoricalParameter(["none", "pressure"], default="pressure", space="buy", optimize=True, load=True)
    short_retest_volume_mode = CategoricalParameter(["none", "pressure"], default="pressure", space="buy", optimize=True, load=True)
    short_close_location_max = DecimalParameter(0.20, 0.85, decimals=2, default=0.55, space="buy", optimize=True, load=True)
    short_break_close_location_max = DecimalParameter(0.10, 0.75, decimals=2, default=0.45, space="buy", optimize=True, load=True)
    short_min_body_fraction = DecimalParameter(0.00, 0.70, decimals=2, default=0.15, space="buy", optimize=True, load=True)
    short_acceptance_bars = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: stake / leverage / add shape
    # ------------------------------------------------------------------

    leverage_opt = IntParameter(1, 7, default=2, space="buy", optimize=True, load=True)
    stake_fraction = DecimalParameter(0.25, 1.00, decimals=2, default=0.50, space="buy", optimize=True, load=True)
    add1_mult = DecimalParameter(0.5, 1.5, decimals=1, default=1.0, space="buy", optimize=True, load=True)
    add2_mult = DecimalParameter(0.5, 2.0, decimals=1, default=1.0, space="buy", optimize=True, load=True)
    enable_long_adds = BooleanParameter(default=False, space="buy", optimize=False, load=False)
    enable_short_adds = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    add_retest_buffer_pct = DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True)
    enable_volume_size_scaling = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    volume_size_mult_max = DecimalParameter(1.0, 2.0, decimals=1, default=1.3, space="buy", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: peels / exits / stop
    # ------------------------------------------------------------------

    peel1_fraction = DecimalParameter(0.20, 0.50, decimals=2, default=0.30, space="sell", optimize=True, load=True)
    peel2_fraction = DecimalParameter(0.20, 0.50, decimals=2, default=0.30, space="sell", optimize=True, load=True)
    peel_min_profit_pct = DecimalParameter(0.002, 0.080, decimals=3, default=0.010, space="sell", optimize=True, load=True)
    peel_at_h1_level = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    full_exit_at_d1_opposite_level = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    stop_level_buffer_pct = DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="sell", optimize=True, load=True)
    trail_after_profit_pct = DecimalParameter(0.005, 0.080, decimals=3, default=0.025, space="sell", optimize=True, load=True)
    trail_level_buffer_pct = DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="sell", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: entry-aware exit profiles.
    #
    # These stay small on purpose. The neutral baseline remains the current
    # exit engine, and these only add family-aware bias when explicitly enabled.
    # ------------------------------------------------------------------

    enable_entry_family_exit_profiles = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    continuation_exit_profile = CategoricalParameter(["balanced", "runner"], default="balanced", space="sell", optimize=True, load=True)
    reversal_exit_profile = CategoricalParameter(["balanced", "target_clip"], default="balanced", space="sell", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Hyperopt: targeted daily-level exits.
    #
    # Intent:
    # - For reversal-style entries, let Explorer test harvesting into the
    #   opposite daily level slightly before the exact touch, and tighten the
    #   trail when price is already pressing into that target area.
    # ------------------------------------------------------------------

    enable_targeted_d1_exit = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    targeted_d1_exit_buffer_pct = DecimalParameter(0.002, 0.020, decimals=3, default=0.006, space="sell", optimize=True, load=True)
    targeted_d1_trail_buffer_pct = DecimalParameter(0.002, 0.020, decimals=3, default=0.008, space="sell", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Dormant conservative portfolio-risk overlay.
    #
    # Intent:
    # - Keep this deliberately small and conservative so we can test whether a
    #   simple entry governor helps without dragging the whole strategy around.
    # - When enabled, it only blocks new entries and optional adds based on
    #   gross wallet exposure and live drawdown. It does not scale stake,
    #   leverage, or exits.
    # ------------------------------------------------------------------

    enable_portfolio_risk_overlay = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    portfolio_max_gross_exposure_pct = CategoricalParameter([150, 200, 250], default=200, space="buy", optimize=True, load=True)
    portfolio_drawdown_entry_block_pct = CategoricalParameter([12, 16, 20], default=16, space="buy", optimize=True, load=True)
    portfolio_disable_adds_on_risk = BooleanParameter(default=True, space="buy", optimize=True, load=True)

    # ------------------------------------------------------------------
    # Future research scaffolding, intentionally inactive.
    #
    # Volatility leverage adapter:
    # - Intent: higher pair-level volatility should reduce leverage, not widen
    #   entry logic. This keeps risk per coin more comparable.
    # - Suggested batch tag: batch:volatility_leverage.
    # - Suggested hook: multiply leverage_opt in leverage() by a clipped
    #   volatility multiplier derived from d1_atr / close.
    #
    # enable_volatility_leverage_adapter = BooleanParameter(
    #     BooleanParameter(default=False, space="buy", optimize=True, load=True),
    #     "family:capital",
    #     "family:feature_enables",
    #     "batch:enables",
    #     "batch:volatility_leverage",
    #     "switch:enable",
    #     "domain:capital",
    #     "domain:risk",
    #     "role:gate_toggle",
    # )
    # volatility_leverage_floor = DecimalParameter(
    #     DecimalParameter(0.25, 1.00, decimals=2, default=0.60, space="buy", optimize=True, load=True),
    #     "family:capital",
    #     "batch:volatility_leverage",
    #     "domain:capital",
    #     "domain:risk",
    #     "role:leverage_floor",
    # )
    # volatility_leverage_atr_pct_high = DecimalParameter(
    #     DecimalParameter(0.02, 0.16, decimals=2, default=0.08, space="buy", optimize=True, load=True),
    #     "family:capital",
    #     "batch:volatility_leverage",
    #     "domain:risk",
    #     "role:volatility_threshold",
    # )

    plot_config = {
        "main_plot": {
            "d1_resistance": {},
            "d1_support": {},
            "d1_res_zone_upper": {},
            "d1_res_zone_lower": {},
            "d1_sup_zone_upper": {},
            "d1_sup_zone_lower": {},
            "h1_local_high": {},
            "h1_local_low": {},
            "plot_stop_long_res_break": {},
            "plot_stop_long_sup_hold": {},
            "plot_stop_short_res_fail": {},
            "plot_stop_short_sup_break": {},
        },
        "subplots": {
            "Volume Pressure": {
                "volume_ratio": {},
                "up_volume_pressure": {},
                "down_volume_pressure": {},
            },
            "Signal State": {
                "plot_long_res_break": {},
                "plot_short_res_fail": {},
                "plot_long_sup_hold": {},
                "plot_short_sup_break": {},
                "plot_long_sup_reclaim": {},
                "plot_short_res_reclaim": {},
                "plot_long_res_retest_hold": {},
                "plot_short_sup_retest_reject": {},
            },
            "Local Structure": {
                "h1_breaks_local_high": {},
                "h1_rejects_local_high": {},
                "h1_reclaims_local_low": {},
                "h1_breaks_local_low": {},
            },
            "Trade Management": {
                "plot_long_add_zone": {},
                "plot_short_add_zone": {},
                "plot_long_peel_level": {},
                "plot_short_peel_level": {},
            },
        },
    }

    # ------------------------------------------------------------------
    # Informative-data helpers
    # ------------------------------------------------------------------

    def informative_pairs(self) -> list[tuple[str, str]]:
        """Request same-pair 1d candles for every pair currently in the whitelist."""
        if not self.dp:
            return []
        try:
            return [(pair, "1d") for pair in self.dp.current_whitelist()]
        except Exception as exc:
            logger.debug("Could not build informative_pairs(): %s", exc)
            return []

    @staticmethod
    def _atr(informative: DataFrame, period: int = 14) -> Series:
        """Small pandas ATR implementation to avoid adding TA-Lib dependency here."""
        high = informative["high"]
        low = informative["low"]
        close = informative["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _daily_informative_dataframe(self, metadata: dict[str, Any]) -> DataFrame | None:
        if not self.dp:
            return None
        pair = str(metadata.get("pair") or "").strip()
        if not pair:
            return None
        try:
            return self.dp.get_pair_dataframe(pair=pair, timeframe="1d")
        except Exception as exc:
            logger.debug("1d informative dataframe unavailable for %s: %s", pair, exc)
            return None

    def _add_daily_structure(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        """
        Add completed-daily support/resistance columns to the 1h dataframe.

        merge_informative_pair shifts/aligns informative candles so the 1h strategy
        does not see future daily information before that daily candle is complete.
        """
        informative = self._daily_informative_dataframe(metadata)
        daily_cols: list[str] = []

        if informative is None or informative.empty:
            result = dataframe.copy()
            for lookback in self.D1_LEVEL_LOOKBACK_CHOICES:
                for name in (f"d1_resistance_{lookback}", f"d1_support_{lookback}"):
                    result[name] = np.nan
                    daily_cols.append(name)
            for name in ("d1_open", "d1_high", "d1_low", "d1_close", "d1_atr"):
                result[name] = np.nan
            return result

        inf = informative.copy().sort_values("date")
        inf["d1_open"] = inf["open"]
        inf["d1_high"] = inf["high"]
        inf["d1_low"] = inf["low"]
        inf["d1_close"] = inf["close"]
        inf["d1_atr"] = self._atr(inf, 14)

        for lookback in self.D1_LEVEL_LOOKBACK_CHOICES:
            res_col = f"d1_resistance_{lookback}"
            sup_col = f"d1_support_{lookback}"
            inf[res_col] = inf["high"].rolling(lookback).max().shift(1)
            inf[sup_col] = inf["low"].rolling(lookback).min().shift(1)
            daily_cols.extend([res_col, sup_col])

        keep_cols = ["date", "d1_open", "d1_high", "d1_low", "d1_close", "d1_atr"] + daily_cols
        merged = merge_informative_pair(
            dataframe.copy(),
            inf[keep_cols],
            self.timeframe,
            "1d",
            ffill=True,
        )

        # Copy de-suffixed aliases back onto the 1h frame. This keeps the plotting
        # and signal code readable while merge_informative_pair still handles the
        # no-lookahead alignment.
        for col in keep_cols:
            if col == "date":
                continue
            merged_col = f"{col}_1d"
            merged[col] = merged[merged_col] if merged_col in merged.columns else np.nan
        return merged

    # ------------------------------------------------------------------
    # Indicator section
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = self._add_daily_structure(dataframe, metadata)

        close = dataframe["close"].replace(0, np.nan)
        high = dataframe["high"]
        low = dataframe["low"]
        open_ = dataframe["open"]
        volume = dataframe["volume"]

        new_cols: dict[str, Series] = {}

        for lookback in self.H1_LOCAL_LOOKBACK_CHOICES:
            local_high = high.rolling(lookback).max().shift(1)
            local_low = low.rolling(lookback).min().shift(1)
            new_cols[f"h1_local_high_{lookback}"] = local_high
            new_cols[f"h1_local_low_{lookback}"] = local_low
            new_cols[f"h1_breaks_local_high_{lookback}"] = close > local_high
            new_cols[f"h1_breaks_local_low_{lookback}"] = close < local_low
            new_cols[f"h1_rejects_local_high_{lookback}"] = (high > local_high) & (close < local_high)
            new_cols[f"h1_reclaims_local_low_{lookback}"] = (low < local_low) & (close > local_low)
            new_cols[f"h1_holds_local_low_{lookback}"] = (low <= local_low) & (close >= local_low) & (close >= open_)
            new_cols[f"h1_rejects_local_low_from_below_{lookback}"] = (low < local_low) & (close < local_low)

        for window in self.VOLUME_WINDOW_CHOICES:
            volume_ma = volume.rolling(window).mean().replace(0, np.nan)
            up_volume = volume.where(close > open_, 0.0)
            down_volume = volume.where(close < open_, 0.0)
            total_volume = volume.rolling(window).sum().replace(0, np.nan)
            new_cols[f"volume_ma_{window}"] = volume_ma
            new_cols[f"volume_ratio_{window}"] = volume / volume_ma
            new_cols[f"up_volume_pressure_{window}"] = up_volume.rolling(window).sum() / total_volume
            new_cols[f"down_volume_pressure_{window}"] = down_volume.rolling(window).sum() / total_volume

        features = pd.DataFrame(new_cols, index=dataframe.index)
        dataframe = pd.concat([dataframe.copy(), features], axis=1)

        # Human-readable active aliases for FreqUI plots. Hyperopt logic uses the
        # specific variant columns selected by the current parameter values.
        h1_lb = int(self.h1_local_lookback.value)
        vol_window = int(self.volume_window.value)
        d1_lb = int(self.d1_level_lookback.value)

        dataframe["h1_local_high"] = dataframe[f"h1_local_high_{h1_lb}"]
        dataframe["h1_local_low"] = dataframe[f"h1_local_low_{h1_lb}"]
        dataframe["h1_breaks_local_high"] = dataframe[f"h1_breaks_local_high_{h1_lb}"].astype(float)
        dataframe["h1_breaks_local_low"] = dataframe[f"h1_breaks_local_low_{h1_lb}"].astype(float)
        dataframe["h1_rejects_local_high"] = dataframe[f"h1_rejects_local_high_{h1_lb}"].astype(float)
        dataframe["h1_reclaims_local_low"] = dataframe[f"h1_reclaims_local_low_{h1_lb}"].astype(float)
        dataframe["volume_ratio"] = dataframe[f"volume_ratio_{vol_window}"]
        dataframe["up_volume_pressure"] = dataframe[f"up_volume_pressure_{vol_window}"]
        dataframe["down_volume_pressure"] = dataframe[f"down_volume_pressure_{vol_window}"]
        dataframe["d1_resistance"] = dataframe[f"d1_resistance_{d1_lb}"]
        dataframe["d1_support"] = dataframe[f"d1_support_{d1_lb}"]

        zone_width = self._zone_width(dataframe)
        dataframe["d1_res_zone_upper"] = dataframe["d1_resistance"] + zone_width
        dataframe["d1_res_zone_lower"] = dataframe["d1_resistance"] - zone_width
        dataframe["d1_sup_zone_upper"] = dataframe["d1_support"] + zone_width
        dataframe["d1_sup_zone_lower"] = dataframe["d1_support"] - zone_width

        stop_buf = float(self.stop_level_buffer_pct.value)
        dataframe["plot_stop_long_res_break"] = dataframe["d1_resistance"] * (1.0 - stop_buf)
        dataframe["plot_stop_long_sup_hold"] = dataframe["d1_support"] * (1.0 - stop_buf)
        dataframe["plot_stop_short_res_fail"] = dataframe["d1_resistance"] * (1.0 + stop_buf)
        dataframe["plot_stop_short_sup_break"] = dataframe["d1_support"] * (1.0 + stop_buf)

        dataframe["plot_long_add_zone"] = dataframe["d1_resistance"]
        dataframe["plot_short_add_zone"] = dataframe["d1_support"]
        dataframe["plot_long_peel_level"] = dataframe["h1_local_high"]
        dataframe["plot_short_peel_level"] = dataframe["h1_local_low"]

        # Signal plot placeholders are overwritten in populate_entry_trend().
        for col in (
            "plot_long_res_break",
            "plot_short_res_fail",
            "plot_long_sup_hold",
            "plot_short_sup_break",
            "plot_long_sup_reclaim",
            "plot_short_res_reclaim",
            "plot_long_res_retest_hold",
            "plot_short_sup_retest_reject",
        ):
            dataframe[col] = 0.0

        return dataframe

    # ------------------------------------------------------------------
    # Dataframe utility helpers
    # ------------------------------------------------------------------

    def _false_mask(self, dataframe: DataFrame) -> Series:
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _true_mask(self, dataframe: DataFrame) -> Series:
        return pd.Series(True, index=dataframe.index, dtype="bool")

    def _bool_col(self, dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return self._false_mask(dataframe)
        return pd.Series(dataframe[column], index=dataframe.index).astype("boolean").fillna(False).astype(bool)

    def _num_col(self, dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return pd.Series(np.nan, index=dataframe.index, dtype="float64")
        return pd.to_numeric(dataframe[column], errors="coerce")

    def _all_conditions(self, dataframe: DataFrame, conditions: list[Series]) -> Series:
        if not conditions:
            return self._false_mask(dataframe)
        result = self._true_mask(dataframe)
        for condition in conditions:
            result &= pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
        return result

    def _d1_res_col(self) -> str:
        return f"d1_resistance_{int(self.d1_level_lookback.value)}"

    def _d1_sup_col(self) -> str:
        return f"d1_support_{int(self.d1_level_lookback.value)}"

    def _h1_high_col(self) -> str:
        return f"h1_local_high_{int(self.h1_local_lookback.value)}"

    def _h1_low_col(self) -> str:
        return f"h1_local_low_{int(self.h1_local_lookback.value)}"

    def _volume_ratio_col(self) -> str:
        return f"volume_ratio_{int(self.volume_window.value)}"

    def _up_pressure_col(self) -> str:
        return f"up_volume_pressure_{int(self.volume_window.value)}"

    def _down_pressure_col(self) -> str:
        return f"down_volume_pressure_{int(self.volume_window.value)}"

    def _zone_width(self, dataframe: DataFrame) -> Series:
        atr_width = self._num_col(dataframe, "d1_atr") * float(self.d1_zone_atr_mult.value)
        pct_width = self._num_col(dataframe, "close") * float(self.d1_zone_pct_min.value)
        return pd.concat([atr_width, pct_width], axis=1).max(axis=1)

    def _near_level(self, price: Series, level: Series, pct: float) -> Series:
        return ((price - level).abs() / level.replace(0, np.nan)) <= float(pct)

    def _volume_pressure_ok(self, dataframe: DataFrame, side: str) -> Series:
        mode = str(self.volume_gate_mode.value)
        if mode == "none":
            return self._true_mask(dataframe)
        volume_ratio_ok = self._num_col(dataframe, self._volume_ratio_col()) >= float(self.volume_ratio_min.value)
        if side == "long":
            pressure_ok = self._num_col(dataframe, self._up_pressure_col()) >= float(self.pressure_min.value)
        else:
            pressure_ok = self._num_col(dataframe, self._down_pressure_col()) >= float(self.pressure_min.value)
        if mode == "ratio_only":
            return volume_ratio_ok
        if mode == "pressure_only":
            return pressure_ok
        return volume_ratio_ok & pressure_ok

    def _recent_true(self, mask: Series, bars: int) -> Series:
        window = max(1, int(bars))
        series = pd.Series(mask, index=mask.index).fillna(False).astype(bool)
        return series.rolling(window, min_periods=1).max().shift(1).fillna(False).astype(bool)

    def _candle_close_location(self, dataframe: DataFrame) -> Series:
        high = self._num_col(dataframe, "high")
        low = self._num_col(dataframe, "low")
        close = self._num_col(dataframe, "close")
        span = (high - low).replace(0, np.nan)
        return ((close - low) / span).clip(0.0, 1.0)

    def _candle_body_fraction(self, dataframe: DataFrame) -> Series:
        high = self._num_col(dataframe, "high")
        low = self._num_col(dataframe, "low")
        close = self._num_col(dataframe, "close")
        open_ = self._num_col(dataframe, "open")
        span = (high - low).replace(0, np.nan)
        return ((close - open_).abs() / span).clip(0.0, 1.0)

    def _short_acceptance_ok(self, raw_mask: Series, bars: int) -> Series:
        bars = max(1, int(bars))
        mask = pd.Series(raw_mask, index=raw_mask.index).fillna(False).astype(bool)
        if bars <= 1:
            return mask
        return mask.rolling(bars, min_periods=bars).sum().eq(bars).fillna(False).astype(bool)

    def _long_res_clean_break_condition(self, dataframe: DataFrame, require_volume: bool = False) -> Series:
        close = self._num_col(dataframe, "close")
        d1_res = self._num_col(dataframe, self._d1_res_col())
        h1_high = self._num_col(dataframe, self._h1_high_col())
        buffer = float(self.breakout_buffer_pct.value)
        conditions = [
            d1_res.notna(),
            h1_high.notna(),
            self._near_level(h1_high, d1_res, float(self.h1_level_near_daily_pct.value)),
            close > d1_res * (1.0 + buffer),
            close > h1_high,
        ]
        if require_volume:
            conditions.append(self._volume_pressure_ok(dataframe, "long"))
        return self._all_conditions(dataframe, conditions)

    def _short_sup_clean_break_condition(self, dataframe: DataFrame, require_volume: bool = False) -> Series:
        close = self._num_col(dataframe, "close")
        d1_sup = self._num_col(dataframe, self._d1_sup_col())
        h1_low = self._num_col(dataframe, self._h1_low_col())
        buffer = float(self.breakout_buffer_pct.value)
        conditions = [
            d1_sup.notna(),
            h1_low.notna(),
            self._near_level(h1_low, d1_sup, float(self.h1_level_near_daily_pct.value)),
            close < d1_sup * (1.0 - buffer),
            close < h1_low,
        ]
        if require_volume:
            conditions.append(self._volume_pressure_ok(dataframe, "short"))
        return self._all_conditions(dataframe, conditions)

    def _long_breakout_mask(self, dataframe: DataFrame) -> Series:
        if not bool(self.enable_long_res_break.value):
            return self._false_mask(dataframe)

        close = self._num_col(dataframe, "close")
        low = self._num_col(dataframe, "low")
        d1_res = self._num_col(dataframe, self._d1_res_col())
        h1_high = self._num_col(dataframe, self._h1_high_col())
        zone = self._zone_width(dataframe)
        buffer = float(self.breakout_buffer_pct.value)

        clean_break = (close > d1_res * (1.0 + buffer)) & (close > h1_high)
        local_high_near_res = self._near_level(h1_high, d1_res, float(self.h1_level_near_daily_pct.value))

        if str(self.long_breakout_confirm_mode.value) == "break_retest":
            recent_break = clean_break.rolling(int(self.long_breakout_retest_bars.value)).max().shift(1).fillna(False).astype(bool)
            structural = recent_break & (low <= d1_res + zone) & (close >= d1_res)
        else:
            structural = clean_break

        return self._all_conditions(
            dataframe,
            [
                d1_res.notna(),
                h1_high.notna(),
                local_high_near_res,
                structural,
                self._volume_pressure_ok(dataframe, "long"),
            ],
        )

    def _short_res_fail_mask(self, dataframe: DataFrame) -> Series:
        if str(self.short_res_fail_mode.value) == "none":
            return self._false_mask(dataframe)

        high = self._num_col(dataframe, "high")
        close = self._num_col(dataframe, "close")
        d1_res = self._num_col(dataframe, self._d1_res_col())
        h1_high = self._num_col(dataframe, self._h1_high_col())
        wick_above_res = ((high - d1_res) / d1_res.replace(0, np.nan)) >= float(self.short_fail_wick_min_pct.value)
        local_reject = self._bool_col(dataframe, f"h1_rejects_local_high_{int(self.h1_local_lookback.value)}")
        local_high_near_res = self._near_level(h1_high, d1_res, float(self.h1_level_near_daily_pct.value))
        close_location_ok = self._candle_close_location(dataframe) <= float(self.short_close_location_max.value)
        body_ok = self._candle_body_fraction(dataframe) >= float(self.short_min_body_fraction.value)

        mode = str(self.short_res_fail_mode.value)
        if mode == "wick_reject":
            rejection_ok = wick_above_res
        elif mode == "local_reject":
            rejection_ok = local_reject
        else:
            rejection_ok = wick_above_res | local_reject

        conditions = [
            d1_res.notna(),
            h1_high.notna(),
            local_high_near_res,
            high > d1_res,
            close < d1_res,
            close < h1_high,
            rejection_ok,
            close_location_ok,
            body_ok,
        ]
        if str(self.short_fail_volume_mode.value) == "pressure":
            conditions.append(self._volume_pressure_ok(dataframe, "short"))
        return self._all_conditions(dataframe, conditions)


    def _long_support_mask(self, dataframe: DataFrame) -> Series:
        if not bool(self.enable_long_sup_hold.value):
            return self._false_mask(dataframe)

        close = self._num_col(dataframe, "close")
        low = self._num_col(dataframe, "low")
        d1_sup = self._num_col(dataframe, self._d1_sup_col())
        h1_low = self._num_col(dataframe, self._h1_low_col())
        zone = self._zone_width(dataframe)
        local_low_near_sup = self._near_level(h1_low, d1_sup, float(self.h1_level_near_daily_pct.value))
        h1_reclaim = self._bool_col(dataframe, f"h1_reclaims_local_low_{int(self.h1_local_lookback.value)}")
        h1_hold = self._bool_col(dataframe, f"h1_holds_local_low_{int(self.h1_local_lookback.value)}")

        if str(self.long_support_mode.value) == "touch_hold":
            structural = (low <= d1_sup + zone) & (close >= d1_sup) & (h1_hold | h1_reclaim)
        else:
            structural = (low < d1_sup) & (close > d1_sup) & h1_reclaim

        return self._all_conditions(
            dataframe,
            [
                d1_sup.notna(),
                h1_low.notna(),
                local_low_near_sup,
                structural,
                self._volume_pressure_ok(dataframe, "long"),
            ],
        )

    def _short_support_break_mask(self, dataframe: DataFrame) -> Series:
        if str(self.short_break_confirm_mode.value) == "none":
            return self._false_mask(dataframe)

        close = self._num_col(dataframe, "close")
        high = self._num_col(dataframe, "high")
        low = self._num_col(dataframe, "low")
        d1_sup = self._num_col(dataframe, self._d1_sup_col())
        h1_low = self._num_col(dataframe, self._h1_low_col())
        zone = self._zone_width(dataframe)
        buffer = float(self.breakout_buffer_pct.value)
        close_location_ok = self._candle_close_location(dataframe) <= float(self.short_break_close_location_max.value)
        body_ok = self._candle_body_fraction(dataframe) >= float(self.short_min_body_fraction.value)
        raw_clean_break = (
            (low < d1_sup - zone * 0.25)
            & (close < d1_sup * (1.0 - buffer))
            & (close < h1_low)
            & close_location_ok
            & body_ok
        )
        clean_break = self._short_acceptance_ok(raw_clean_break, int(self.short_acceptance_bars.value))
        local_low_near_sup = self._near_level(h1_low, d1_sup, float(self.h1_level_near_daily_pct.value))

        if str(self.short_break_confirm_mode.value) == "break_retest":
            recent_break = clean_break.rolling(int(self.short_break_retest_bars.value)).max().shift(1).fillna(False).astype(bool)
            structural = recent_break & (high >= d1_sup - zone) & (close <= d1_sup) & close_location_ok
        else:
            structural = clean_break

        return self._all_conditions(
            dataframe,
            [
                d1_sup.notna(),
                h1_low.notna(),
                local_low_near_sup,
                structural,
                self._volume_pressure_ok(dataframe, "short"),
            ],
        )


    def _long_support_reclaim_mask(self, dataframe: DataFrame) -> Series:
        if not bool(self.enable_long_sup_reclaim.value):
            return self._false_mask(dataframe)

        close = self._num_col(dataframe, "close")
        low = self._num_col(dataframe, "low")
        d1_sup = self._num_col(dataframe, self._d1_sup_col())
        h1_low = self._num_col(dataframe, self._h1_low_col())
        zone = self._zone_width(dataframe)
        reclaim_buffer = float(self.reclaim_buffer_pct.value)
        lookback = int(self.reclaim_lookback_bars.value)
        local_low_near_sup = self._near_level(h1_low, d1_sup, float(self.h1_level_near_daily_pct.value))
        h1_reclaim = self._bool_col(dataframe, f"h1_reclaims_local_low_{int(self.h1_local_lookback.value)}")
        h1_hold = self._bool_col(dataframe, f"h1_holds_local_low_{int(self.h1_local_lookback.value)}")

        support_lost = (low < d1_sup - zone) | (close < d1_sup * (1.0 - reclaim_buffer))
        recent_support_lost = self._recent_true(support_lost, lookback)
        recent_accepted_short = self._recent_true(self._short_sup_clean_break_condition(dataframe, require_volume=True), lookback)
        reclaim = (low <= d1_sup + zone) & (close >= d1_sup * (1.0 + reclaim_buffer)) & (h1_reclaim | h1_hold)

        conditions = [
            d1_sup.notna(),
            h1_low.notna(),
            local_low_near_sup,
            recent_support_lost,
            ~recent_accepted_short,
            reclaim,
        ]
        if bool(self.reclaim_require_volume_pressure.value):
            conditions.append(self._volume_pressure_ok(dataframe, "long"))
        return self._all_conditions(dataframe, conditions)

    def _short_res_reclaim_mask(self, dataframe: DataFrame) -> Series:
        if str(self.short_res_reclaim_mode.value) == "none":
            return self._false_mask(dataframe)

        high = self._num_col(dataframe, "high")
        close = self._num_col(dataframe, "close")
        d1_res = self._num_col(dataframe, self._d1_res_col())
        h1_high = self._num_col(dataframe, self._h1_high_col())
        zone = self._zone_width(dataframe)
        reclaim_buffer = float(self.reclaim_buffer_pct.value)
        lookback = int(self.reclaim_lookback_bars.value)
        local_high_near_res = self._near_level(h1_high, d1_res, float(self.h1_level_near_daily_pct.value))
        h1_reject = self._bool_col(dataframe, f"h1_rejects_local_high_{int(self.h1_local_lookback.value)}")
        close_location_ok = self._candle_close_location(dataframe) <= float(self.short_close_location_max.value)
        body_ok = self._candle_body_fraction(dataframe) >= float(self.short_min_body_fraction.value)

        resistance_broken = (high > d1_res + zone) | (close > d1_res * (1.0 + reclaim_buffer))
        recent_resistance_broken = self._recent_true(resistance_broken, lookback)
        recent_accepted_long = self._recent_true(self._long_res_clean_break_condition(dataframe, require_volume=True), lookback)
        reclaim = (
            (high >= d1_res - zone)
            & (close <= d1_res * (1.0 - reclaim_buffer))
            & (close <= h1_high)
            & (h1_reject | (high >= h1_high))
            & close_location_ok
            & body_ok
        )

        conditions = [
            d1_res.notna(),
            h1_high.notna(),
            local_high_near_res,
            recent_resistance_broken,
            ~recent_accepted_long,
            reclaim,
        ]
        if str(self.short_reclaim_volume_mode.value) == "pressure":
            conditions.append(self._volume_pressure_ok(dataframe, "short"))
        return self._all_conditions(dataframe, conditions)


    def _long_res_retest_hold_mask(self, dataframe: DataFrame) -> Series:
        if not bool(self.enable_long_res_retest_hold.value):
            return self._false_mask(dataframe)

        close = self._num_col(dataframe, "close")
        low = self._num_col(dataframe, "low")
        d1_res = self._num_col(dataframe, self._d1_res_col())
        h1_low = self._num_col(dataframe, self._h1_low_col())
        lookback = int(self.entry_retest_lookback_bars.value)
        retest_buffer = float(self.entry_retest_buffer_pct.value)
        recent_break = self._recent_true(self._long_res_clean_break_condition(dataframe, require_volume=True), lookback)
        local_low_near_res = self._near_level(h1_low, d1_res, float(self.h1_level_near_daily_pct.value))
        h1_reclaim = self._bool_col(dataframe, f"h1_reclaims_local_low_{int(self.h1_local_lookback.value)}")
        h1_hold = self._bool_col(dataframe, f"h1_holds_local_low_{int(self.h1_local_lookback.value)}")
        retest_holds = (low <= d1_res * (1.0 + retest_buffer)) & (close >= d1_res) & (h1_reclaim | h1_hold | (close >= h1_low))

        conditions = [
            d1_res.notna(),
            h1_low.notna(),
            local_low_near_res,
            recent_break,
            retest_holds,
        ]
        if bool(self.retest_require_volume_pressure.value):
            conditions.append(self._volume_pressure_ok(dataframe, "long"))
        return self._all_conditions(dataframe, conditions)

    def _short_sup_retest_reject_mask(self, dataframe: DataFrame) -> Series:
        if str(self.short_sup_retest_reject_mode.value) == "none":
            return self._false_mask(dataframe)

        high = self._num_col(dataframe, "high")
        close = self._num_col(dataframe, "close")
        d1_sup = self._num_col(dataframe, self._d1_sup_col())
        h1_high = self._num_col(dataframe, self._h1_high_col())
        lookback = int(self.entry_retest_lookback_bars.value)
        retest_buffer = float(self.entry_retest_buffer_pct.value)
        recent_break = self._recent_true(self._short_sup_clean_break_condition(dataframe, require_volume=True), lookback)
        local_high_near_sup = self._near_level(h1_high, d1_sup, float(self.h1_level_near_daily_pct.value))
        h1_reject = self._bool_col(dataframe, f"h1_rejects_local_high_{int(self.h1_local_lookback.value)}")
        close_location_ok = self._candle_close_location(dataframe) <= float(self.short_close_location_max.value)
        body_ok = self._candle_body_fraction(dataframe) >= float(self.short_min_body_fraction.value)
        retest_rejects = (
            (high >= d1_sup * (1.0 - retest_buffer))
            & (close <= d1_sup)
            & (close <= h1_high)
            & (h1_reject | (high >= h1_high))
            & close_location_ok
            & body_ok
        )

        conditions = [
            d1_sup.notna(),
            h1_high.notna(),
            local_high_near_sup,
            recent_break,
            retest_rejects,
        ]
        if str(self.short_retest_volume_mode.value) == "pressure":
            conditions.append(self._volume_pressure_ok(dataframe, "short"))
        return self._all_conditions(dataframe, conditions)


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        short_res_reclaim = self._short_res_reclaim_mask(dataframe)
        short_res_fail = self._short_res_fail_mask(dataframe)
        short_sup_retest_reject = self._short_sup_retest_reject_mask(dataframe)
        short_sup_break = self._short_support_break_mask(dataframe)

        dataframe["plot_long_res_break"] = 0.0
        dataframe["plot_long_sup_hold"] = 0.0
        dataframe["plot_long_sup_reclaim"] = 0.0
        dataframe["plot_long_res_retest_hold"] = 0.0
        dataframe["plot_short_res_fail"] = short_res_fail.astype(float)
        dataframe["plot_short_sup_break"] = short_sup_break.astype(float)
        dataframe["plot_short_res_reclaim"] = short_res_reclaim.astype(float)
        dataframe["plot_short_sup_retest_reject"] = short_sup_retest_reject.astype(float)

        unresolved = pd.Series(True, index=dataframe.index, dtype="bool")

        for tag, mask in (
            ("short_res_reclaim", short_res_reclaim),
            ("short_res_fail", short_res_fail),
            ("short_sup_retest_reject", short_sup_retest_reject),
            ("short_sup_break", short_sup_break),
        ):
            selected = pd.Series(mask, index=dataframe.index).fillna(False).astype(bool) & unresolved
            dataframe.loc[selected, "enter_short"] = 1
            dataframe.loc[selected, "enter_tag"] = tag
            unresolved &= ~selected

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    # ------------------------------------------------------------------
    # Small persistent trade ledger
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_family(tag: str | None) -> str:
        tag = str(tag or "")
        if tag in (
            "long_res_break",
            "short_res_fail",
            "long_sup_hold",
            "short_sup_break",
            "long_sup_reclaim",
            "short_res_reclaim",
            "long_res_retest_hold",
            "short_sup_retest_reject",
        ):
            return tag
        return "unknown"

    def _default_ledger(self, trade: Trade) -> dict[str, Any]:
        entry_tag = self._entry_family(getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None))
        return {
            "entry_tag": entry_tag,
            "is_short": bool(getattr(trade, "is_short", False)),
            "seed_stake": float(getattr(trade, "stake_amount", 0.0) or 0.0),
            "add_count": 0,
            "peel_count": 0,
            "last_action_candle": None,
        }

    def _get_ledger(self, trade: Trade) -> dict[str, Any]:
        ledger = trade.get_custom_data(key="daily_structure_ladder")
        if not isinstance(ledger, dict):
            ledger = self._default_ledger(trade)
        defaults = self._default_ledger(trade)
        for key, value in defaults.items():
            ledger.setdefault(key, value)
        ledger["add_count"] = int(ledger.get("add_count") or 0)
        ledger["peel_count"] = int(ledger.get("peel_count") or 0)
        ledger["seed_stake"] = float(ledger.get("seed_stake") or getattr(trade, "stake_amount", 0.0) or 0.0)
        ledger["is_short"] = bool(ledger.get("is_short", getattr(trade, "is_short", False)))
        return ledger

    @staticmethod
    def _set_ledger(trade: Trade, ledger: dict[str, Any]) -> None:
        trade.set_custom_data(key="daily_structure_ladder", value=ledger)

    def _latest_analyzed_view(self, pair: str) -> dict[str, Any] | None:
        if not self.dp:
            return None
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        except Exception as exc:
            logger.debug("Could not read analyzed dataframe for %s: %s", pair, exc)
            return None
        if dataframe is None or dataframe.empty:
            return None
        last = dataframe.iloc[-1].squeeze()
        return {
            "dataframe": dataframe,
            "last": last,
            "candle_id": pd.Timestamp(last["date"]).isoformat() if "date" in last else None,
        }

    @staticmethod
    def _safe_float(row: Series, key: str) -> float | None:
        value = row.get(key)
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scalar_zone_width(self, last: Series) -> float:
        atr = self._safe_float(last, "d1_atr")
        close = self._safe_float(last, "close")
        atr_width = 0.0 if atr is None else atr * float(self.d1_zone_atr_mult.value)
        pct_width = 0.0 if close is None else close * float(self.d1_zone_pct_min.value)
        return max(atr_width, pct_width)

    def _scalar_volume_ratio(self, last: Series) -> float:
        value = self._safe_float(last, self._volume_ratio_col())
        return 0.0 if value is None else value

    @staticmethod
    def _entry_exit_family_name(entry_tag: str | None) -> str:
        tag = str(entry_tag or "")
        if tag in ("long_res_break", "short_sup_break", "long_res_retest_hold", "short_sup_retest_reject"):
            return "continuation"
        if tag in ("short_res_fail", "long_sup_hold", "long_sup_reclaim", "short_res_reclaim"):
            return "reversal"
        return "other"

    def _exit_profile_config(self, entry_tag: str | None) -> dict[str, float]:
        config = {
            "peel_profit_mult": 1.0,
            "peel1_mult": 1.0,
            "peel2_mult": 1.0,
            "trail_activation_mult": 1.0,
            "trail_buffer_mult": 1.0,
        }
        if not bool(self.enable_entry_family_exit_profiles.value):
            return config

        family = self._entry_exit_family_name(entry_tag)
        if family == "continuation" and str(self.continuation_exit_profile.value) == "runner":
            config.update(
                {
                    "peel_profit_mult": 1.25,
                    "peel1_mult": 0.75,
                    "peel2_mult": 0.90,
                    "trail_activation_mult": 1.35,
                    "trail_buffer_mult": 1.20,
                }
            )
        elif family == "reversal" and str(self.reversal_exit_profile.value) == "target_clip":
            config.update(
                {
                    "peel_profit_mult": 0.80,
                    "peel1_mult": 1.25,
                    "peel2_mult": 1.10,
                    "trail_activation_mult": 0.80,
                    "trail_buffer_mult": 0.85,
                }
            )
        return config

    def _targeted_d1_target_level(self, trade: Trade, entry_tag: str | None, last: Series) -> float | None:
        if not bool(self.enable_targeted_d1_exit.value):
            return None
        if self._entry_exit_family_name(entry_tag) != "reversal":
            return None

        d1_res = self._safe_float(last, self._d1_res_col())
        d1_sup = self._safe_float(last, self._d1_sup_col())
        if trade.is_short:
            return d1_sup
        return d1_res

    def _targeted_d1_target_reached(self, trade: Trade, last: Series, target_level: float) -> bool:
        close = self._safe_float(last, "close")
        high = self._safe_float(last, "high")
        low = self._safe_float(last, "low")
        if close is None:
            return False

        buffer = float(self.targeted_d1_exit_buffer_pct.value)
        if trade.is_short:
            trigger = target_level * (1.0 + buffer)
            return bool((low is not None and low <= trigger) or close <= trigger)
        trigger = target_level * (1.0 - buffer)
        return bool((high is not None and high >= trigger) or close >= trigger)

    def _portfolio_risk_overlay_enabled(self) -> bool:
        return bool(self.enable_portfolio_risk_overlay.value)

    @staticmethod
    def _open_trades_snapshot() -> list[Trade]:
        try:
            trades = Trade.get_trades_proxy(is_open=True)
            return list(trades) if trades else []
        except Exception:
            try:
                trades = Trade.get_open_trades()
                return list(trades) if trades else []
            except Exception:
                return []

    def _wallet_total_quote(self) -> float:
        if self.wallets:
            try:
                total = float(self.wallets.get_total_stake_amount())
                if total > 0.0:
                    return total
            except Exception:
                pass
            stake_currency = str(self.config.get("stake_currency") or "")
            if stake_currency:
                try:
                    total = float(self.wallets.get_total(stake_currency))
                    if total > 0.0:
                        return total
                except Exception:
                    pass
        try:
            return max(0.0, float(self.config.get("available_capital") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _portfolio_risk_state(self) -> dict[str, Any]:
        state = getattr(self, "_daily_structure_portfolio_risk", None)
        if not isinstance(state, dict):
            state = {"equity_peak": None}
            setattr(self, "_daily_structure_portfolio_risk", state)
        return state

    def _portfolio_drawdown_pct(self) -> float:
        wallet_total = self._wallet_total_quote()
        if wallet_total <= 0.0:
            return 0.0

        state = self._portfolio_risk_state()
        peak = state.get("equity_peak")
        peak = wallet_total if peak is None else max(float(peak), wallet_total)
        state["equity_peak"] = peak
        if peak <= 0.0:
            return 0.0
        return float(max(0.0, (peak - wallet_total) / peak))

    @staticmethod
    def _trade_effective_exposure(trade: Trade) -> float:
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        leverage = max(1.0, float(getattr(trade, "leverage", 1.0) or 1.0))
        return max(0.0, stake * leverage)

    def _portfolio_risk_snapshot(self) -> dict[str, Any]:
        open_trades = self._open_trades_snapshot()
        wallet_total = self._wallet_total_quote()

        long_exposure = sum(
            self._trade_effective_exposure(trade)
            for trade in open_trades
            if not bool(getattr(trade, "is_short", False))
        )
        short_exposure = sum(
            self._trade_effective_exposure(trade)
            for trade in open_trades
            if bool(getattr(trade, "is_short", False))
        )
        gross_exposure = long_exposure + short_exposure
        if wallet_total <= 0.0:
            wallet_total = max(1.0, gross_exposure / 2.0) if gross_exposure > 0.0 else 1.0

        return {
            "wallet_total_quote": float(wallet_total),
            "open_trade_count": len(open_trades),
            "long_exposure": float(long_exposure),
            "short_exposure": float(short_exposure),
            "gross_exposure": float(gross_exposure),
            "gross_exposure_ratio": float(gross_exposure / wallet_total),
            "drawdown_pct": self._portfolio_drawdown_pct(),
        }

    def _portfolio_risk_adjustment(self, pair: str, side: str, current_time: datetime) -> dict[str, Any]:
        _ = pair
        _ = side, current_time
        snapshot = self._portfolio_risk_snapshot()

        gross_cap_ratio = float(self.portfolio_max_gross_exposure_pct.value) / 100.0
        drawdown_block_ratio = float(self.portfolio_drawdown_entry_block_pct.value) / 100.0
        exposure_blocked = snapshot["gross_exposure_ratio"] >= gross_cap_ratio
        drawdown_blocked = snapshot["drawdown_pct"] >= drawdown_block_ratio
        entry_blocked = bool(exposure_blocked or drawdown_blocked)
        add_blocked = bool(self.portfolio_disable_adds_on_risk.value) and entry_blocked
        trigger_reason = "drawdown" if drawdown_blocked else "gross_exposure" if exposure_blocked else "clear"

        return {
            "snapshot": snapshot,
            "risk_state": "blocked" if entry_blocked else "green",
            "entry_blocked": entry_blocked,
            "add_blocked": add_blocked,
            "trigger_reason": trigger_reason,
            "gross_cap_ratio": gross_cap_ratio,
            "drawdown_block_ratio": drawdown_block_ratio,
        }

    # ------------------------------------------------------------------
    # Capital callbacks
    # ------------------------------------------------------------------

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
        **kwargs: Any,
    ) -> bool:
        _ = order_type, amount, rate, time_in_force, entry_tag, kwargs
        if not self._portfolio_risk_overlay_enabled():
            return True

        risk = self._portfolio_risk_adjustment(pair, side, current_time)
        if not risk["entry_blocked"]:
            return True

        snapshot = risk["snapshot"]
        logger.debug(
            "portfolio_risk_entry_block pair=%s side=%s state=%s reason=%s dd=%.4f gross=%.3f cap=%.3f",
            pair,
            side,
            risk["risk_state"],
            risk["trigger_reason"],
            snapshot["drawdown_pct"],
            snapshot["gross_exposure_ratio"],
            risk["gross_cap_ratio"],
        )
        return False

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
        _ = current_rate, proposed_leverage, entry_tag, kwargs
        if not self.position_adjustment_enable:
            return 1.0
        max_allowed = float(max_leverage) if max_leverage and max_leverage > 0 else float(self.leverage_opt.value)
        leverage = float(np.clip(float(self.leverage_opt.value), 1.0, min(7.0, max_allowed)))
        return leverage

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
        **kwargs: Any,
    ) -> float:
        _ = current_rate, leverage, entry_tag, kwargs
        if not self.position_adjustment_enable:
            stake = float(proposed_stake or 0.0)
            if min_stake is not None and float(min_stake) > 0:
                stake = max(stake, float(min_stake) * 1.01)
            if max_stake is not None and float(max_stake) > 0:
                stake = min(stake, float(max_stake))
            return float(max(0.0, stake))
        stake = float(proposed_stake or 0.0) * float(self.stake_fraction.value)

        if bool(self.enable_volume_size_scaling.value):
            view = self._latest_analyzed_view(pair)
            if view:
                ratio = self._scalar_volume_ratio(view["last"])
                stake *= min(float(self.volume_size_mult_max.value), 1.0 + max(0.0, ratio - 1.0) * 0.25)

        if min_stake is not None and float(min_stake) > 0:
            stake = max(stake, float(min_stake) * 1.01)
        if max_stake is not None and float(max_stake) > 0:
            stake = min(stake, float(max_stake))
        return float(max(0.0, stake))

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _add_multiplier(self, add_count: int) -> float:
        return float(self.add1_mult.value) if int(add_count) == 0 else float(self.add2_mult.value)

    def _add_allowed_for_trade(self, ledger: dict[str, Any]) -> bool:
        if ledger["add_count"] >= 2:
            return False
        if ledger["is_short"]:
            return bool(self.enable_short_adds.value)
        return bool(self.enable_long_adds.value)

    def _long_add_ok(self, ledger: dict[str, Any], last: Series) -> bool:
        close = self._safe_float(last, "close")
        low = self._safe_float(last, "low")
        d1_res = self._safe_float(last, self._d1_res_col())
        d1_sup = self._safe_float(last, self._d1_sup_col())
        h1_low = self._safe_float(last, self._h1_low_col())
        if close is None or low is None or h1_low is None:
            return False
        zone = self._scalar_zone_width(last)
        add_buffer = float(self.add_retest_buffer_pct.value)
        entry_tag = str(ledger.get("entry_tag") or "")

        if entry_tag in ("long_res_break", "long_res_retest_hold") and d1_res is not None:
            retests_broken_resistance = low <= d1_res * (1.0 + add_buffer) and close >= d1_res
            local_low_holds_near_res = abs(h1_low - d1_res) / max(d1_res, 1e-12) <= float(self.h1_level_near_daily_pct.value) and low <= h1_low and close >= h1_low
            return bool(retests_broken_resistance or local_low_holds_near_res)

        if entry_tag in ("long_sup_hold", "long_sup_reclaim") and d1_sup is not None:
            retests_support = low <= d1_sup + zone and close >= d1_sup
            local_low_reclaim = low <= h1_low * (1.0 + add_buffer) and close >= h1_low
            return bool(retests_support or local_low_reclaim)

        return False

    def _short_add_ok(self, ledger: dict[str, Any], last: Series) -> bool:
        close = self._safe_float(last, "close")
        high = self._safe_float(last, "high")
        d1_res = self._safe_float(last, self._d1_res_col())
        d1_sup = self._safe_float(last, self._d1_sup_col())
        h1_high = self._safe_float(last, self._h1_high_col())
        if close is None or high is None or h1_high is None:
            return False
        add_buffer = float(self.add_retest_buffer_pct.value)
        entry_tag = str(ledger.get("entry_tag") or "")

        if entry_tag in ("short_sup_break", "short_sup_retest_reject") and d1_sup is not None:
            retests_broken_support = high >= d1_sup * (1.0 - add_buffer) and close <= d1_sup
            local_high_rejects = high >= h1_high and close <= h1_high
            return bool(retests_broken_support or local_high_rejects)

        if entry_tag in ("short_res_fail", "short_res_reclaim") and d1_res is not None:
            # Disabled by default through enable_short_adds. If explicitly enabled,
            # only add when resistance rejects again.
            return bool(high >= d1_res and close <= d1_res and high >= h1_high and close <= h1_high)

        return False

    def _peel_ok(self, trade: Trade, ledger: dict[str, Any], last: Series, current_profit: float) -> bool:
        if ledger["peel_count"] >= 2:
            return False
        profile = self._exit_profile_config(ledger.get("entry_tag"))
        min_profit = float(self.peel_min_profit_pct.value) * float(profile["peel_profit_mult"])
        if float(current_profit) < min_profit:
            return False
        if not bool(self.peel_at_h1_level.value):
            return True

        close = self._safe_float(last, "close")
        high = self._safe_float(last, "high")
        low = self._safe_float(last, "low")
        h1_high = self._safe_float(last, self._h1_high_col())
        h1_low = self._safe_float(last, self._h1_low_col())
        if close is None or high is None or low is None:
            return False

        if trade.is_short:
            return bool(h1_low is not None and (low <= h1_low or close <= h1_low))
        return bool(h1_high is not None and (high >= h1_high or close >= h1_high))

    def _peel_fraction(self, ledger: dict[str, Any]) -> float:
        count = int(ledger["peel_count"])
        base_fraction = float(self.peel1_fraction.value) if count == 0 else float(self.peel2_fraction.value)
        profile = self._exit_profile_config(ledger.get("entry_tag"))
        multiplier = float(profile["peel1_mult"]) if count == 0 else float(profile["peel2_mult"])
        return float(np.clip(base_fraction * multiplier, 0.10, 0.75))

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
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        _ = current_time, current_rate, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if not self.position_adjustment_enable:
            return None

        if trade.has_open_orders:
            return None

        ledger = self._get_ledger(trade)
        view = self._latest_analyzed_view(trade.pair)
        if view is None:
            self._set_ledger(trade, ledger)
            return None

        last = view["last"]
        candle_id = view["candle_id"]
        if candle_id is not None and ledger.get("last_action_candle") == candle_id:
            self._set_ledger(trade, ledger)
            return None

        # Peel before add. This favours reducing exposure into local structure rather
        # than adding and peeling on the same candle.
        if self._peel_ok(trade, ledger, last, current_profit):
            fraction = self._peel_fraction(ledger)
            reduce_stake = max(0.0, float(trade.stake_amount or 0.0) * fraction)
            if min_stake is not None and reduce_stake < float(min_stake) * 1.01:
                reduce_stake = float(min_stake) * 1.01
            reduce_stake = min(reduce_stake, float(trade.stake_amount or 0.0))
            if reduce_stake > 0.0:
                ledger["peel_count"] += 1
                ledger["last_action_candle"] = candle_id
                self._set_ledger(trade, ledger)
                tag = f"peel_{ledger['peel_count']}"
                logger.debug("daily_structure_peel pair=%s trade_id=%s stake=%s tag=%s", trade.pair, getattr(trade, "id", None), reduce_stake, tag)
                return -reduce_stake, tag

        add_limit = 2
        if self._portfolio_risk_overlay_enabled():
            side = "short" if trade.is_short else "long"
            risk = self._portfolio_risk_adjustment(trade.pair, side, current_time)
            if bool(risk["add_blocked"]):
                add_limit = 0

        if ledger["add_count"] < add_limit and self._add_allowed_for_trade(ledger):
            add_ok = self._short_add_ok(ledger, last) if trade.is_short else self._long_add_ok(ledger, last)
            if add_ok:
                base = float(ledger.get("seed_stake") or trade.stake_amount or 0.0)
                add_stake = base * self._add_multiplier(int(ledger["add_count"]))
                if min_stake is not None and float(min_stake) > 0:
                    add_stake = max(add_stake, float(min_stake) * 1.01)
                if max_stake is not None and float(max_stake) > 0:
                    add_stake = min(add_stake, float(max_stake))
                if add_stake > 0.0:
                    ledger["add_count"] += 1
                    ledger["last_action_candle"] = candle_id
                    self._set_ledger(trade, ledger)
                    tag = f"add_{ledger['add_count']}"
                    logger.debug("daily_structure_add pair=%s trade_id=%s stake=%s tag=%s", trade.pair, getattr(trade, "id", None), add_stake, tag)
                    return add_stake, tag

        self._set_ledger(trade, ledger)
        return None

    # ------------------------------------------------------------------
    # Full exits and stoploss
    # ------------------------------------------------------------------

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        _ = current_time, current_rate, kwargs
        if not self.use_exit_signal:
            return None
        view = self._latest_analyzed_view(pair)
        if view is None:
            return None
        last = view["last"]
        ledger = self._get_ledger(trade)
        close = self._safe_float(last, "close")
        d1_res = self._safe_float(last, self._d1_res_col())
        d1_sup = self._safe_float(last, self._d1_sup_col())
        if close is None:
            return None

        entry_tag = str(ledger.get("entry_tag") or "")
        target_level = self._targeted_d1_target_level(trade, entry_tag, last)
        if float(current_profit) > 0 and target_level is not None and self._targeted_d1_target_reached(trade, last, target_level):
            if trade.is_short:
                return "short_targeted_d1_exit"
            return "long_targeted_d1_exit"

        if bool(self.full_exit_at_d1_opposite_level.value) and float(current_profit) > 0:
            if not trade.is_short and entry_tag in ("long_sup_hold", "long_sup_reclaim") and d1_res is not None and close >= d1_res:
                return "long_support_to_d1_res"
            if trade.is_short and entry_tag in ("short_res_fail", "short_res_reclaim") and d1_sup is not None and close <= d1_sup:
                return "short_res_fail_to_d1_sup"

        # Fast invalidation exits. These are intentionally structural, not indicator based.
        buffer = float(self.stop_level_buffer_pct.value)
        if not trade.is_short and entry_tag in ("long_res_break", "long_res_retest_hold") and d1_res is not None and close < d1_res * (1.0 - buffer):
            return "long_breakout_failed"
        if not trade.is_short and entry_tag in ("long_sup_hold", "long_sup_reclaim") and d1_sup is not None and close < d1_sup * (1.0 - buffer):
            return "long_support_failed"
        if trade.is_short and entry_tag in ("short_res_fail", "short_res_reclaim") and d1_res is not None and close > d1_res * (1.0 + buffer):
            return "short_resistance_failed"
        if trade.is_short and entry_tag in ("short_sup_break", "short_sup_retest_reject") and d1_sup is not None and close > d1_sup * (1.0 + buffer):
            return "short_breakdown_failed"

        return None

    def _structural_stop_price(self, trade: Trade, ledger: dict[str, Any], last: Series, current_profit: float) -> float | None:
        entry_tag = str(ledger.get("entry_tag") or "")
        close = self._safe_float(last, "close")
        d1_res = self._safe_float(last, self._d1_res_col())
        d1_sup = self._safe_float(last, self._d1_sup_col())
        h1_high = self._safe_float(last, self._h1_high_col())
        h1_low = self._safe_float(last, self._h1_low_col())
        if close is None:
            return None

        profile = self._exit_profile_config(entry_tag)
        level_buffer = float(self.stop_level_buffer_pct.value)
        trail_buffer = float(self.trail_level_buffer_pct.value) * float(profile["trail_buffer_mult"])
        trail_after_profit = float(self.trail_after_profit_pct.value) * float(profile["trail_activation_mult"])
        trailing_active = float(current_profit) >= trail_after_profit
        target_level = self._targeted_d1_target_level(trade, entry_tag, last)
        if target_level is not None and self._targeted_d1_target_reached(trade, last, target_level) and float(current_profit) > 0:
            trailing_active = True
            trail_buffer = min(trail_buffer, float(self.targeted_d1_trail_buffer_pct.value))

        if not trade.is_short:
            if entry_tag in ("long_res_break", "long_res_retest_hold") and d1_res is not None:
                base_stop = d1_res * (1.0 - level_buffer)
            elif d1_sup is not None:
                base_stop = d1_sup * (1.0 - level_buffer)
            else:
                base_stop = float(trade.open_rate or close) * (1.0 - 0.10)

            if trailing_active and h1_low is not None:
                base_stop = max(base_stop, h1_low * (1.0 - trail_buffer))
            if base_stop >= close:
                return None
            return float(base_stop)

        if entry_tag in ("short_res_fail", "short_res_reclaim") and d1_res is not None:
            base_stop = d1_res * (1.0 + level_buffer)
        elif d1_sup is not None:
            base_stop = d1_sup * (1.0 + level_buffer)
        else:
            base_stop = float(trade.open_rate or close) * (1.0 + 0.10)

        if trailing_active and h1_high is not None:
            base_stop = min(base_stop, h1_high * (1.0 + trail_buffer))
        if base_stop <= close:
            return None
        return float(base_stop)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        _ = current_time, after_fill, kwargs
        if not self.use_custom_stoploss:
            return None
        view = self._latest_analyzed_view(pair)
        if view is None:
            return None
        ledger = self._get_ledger(trade)
        stop_price = self._structural_stop_price(trade, ledger, view["last"], current_profit)
        if stop_price is None or stop_price <= 0.0:
            return None
        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


def _apply_short_modes_parameter_tags() -> None:
    for name, value in vars(DailyStructureShortModesTOP10Normal).items():
        if not _is_parameter_object(value):
            continue
        assignment = _short_modes_parameter_family_mode(name)
        if assignment is None:
            continue
        family, mode = assignment
        tagged_parameter(value, f"family:{family}", f"mode:{mode}")


_apply_short_modes_parameter_tags()
apply_explicit_hyperopt_surface(DailyStructureShortModesTOP10Normal)
