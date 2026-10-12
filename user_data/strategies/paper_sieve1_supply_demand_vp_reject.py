"""Paper-only Sieve1 supply-zone/VP rejection and demand-zone/VP reclaim.

The tested Sieve1 short passed independently; the mirrored long and the
combined long/short strategy have not been validated as profitable.

Selected source evidence: ``D:\FreqTradeStuffLargeData\sieve_launcher_runtime\entry_sieve\results\20260518T025922_entry_confluence.jsonl:6``
(``sieve1_multi2_supply_vp_reject_short``, 3,702 trades, 1.689 trades/day,
TP/SL 3/2, validation ``20200101-20260101``). Exact source snapshot and
selected params are in
``D:\FreqTradeStuffLargeData\sieve_launcher_runtime\entry_sieve\backtests\sieve1_multi2_supply_vp_reject_short__auto_generic_1h_2020_q2_q3\full_cycle_2020_2026\tp_3_sl_2\backtest-result-2026-05-18_03-19-47.zip``
(``..._Sieve1Multi2SupplyVpRejectShort.py`` and ``...json``). Parameters:
reclaim buffer 0.02, zone proximity 0.02, max age 72 bars, VP mode
``score_or_context``, score minimum 0.25, context minimum 0.20; optional volume
guard disabled. The experimental long mirrors those entry thresholds.
Volume-profile settings below are the selected source defaults (96 bars,
48 bins, hlc3, 70% value area, and corresponding existing indicator defaults).
The saved entry-only exit contract is TP 3% / SL 2% on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy

from user_data.Indicators.complex_volume_profile import add_volume_profile


class PaperSieve1SupplyDemandVpReject(IStrategy):
    """Frozen 1h supply rejection short plus its explicitly untested mirror."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 0.03}
    stoploss = -0.02
    use_custom_roi = False
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    exit_profit_offset = 0.0

    # Frozen Sieve1 selected parameters. The optional volume guard was false.
    ZONE_IMPULSE_WINDOW = 36
    ZONE_IMPULSE_ATR_MIN = 0.80
    ZONE_BODY_FRACTION_MIN = 0.55
    ZONE_VOLUME_RATIO_MIN = 1.10
    ZONE_MAX_AGE_BARS = 72
    ZONE_NEAR_PCT = 0.02
    RECLAIM_BUFFER_PCT = 0.02
    VP_GUARD_MODE = "score_or_context"
    VP_SCORE_MIN = 0.25
    VP_CONTEXT_MIN = 0.20

    # Same active indicator and parameter values used by the saved source.
    VP_SETTINGS = {
        "window": 96,
        "bins": 48,
        "value_area_pct": 0.70,
        "price_source": "hlc3",
        "smooth_bins": 3,
        "hvn_threshold": 0.70,
        "lvn_threshold": 0.35,
        "pressure_delta_min": 0.05,
        "node_near_pct": 0.010,
        "volume_percentile_min": 0.55,
        "score_window": 48,
        "fast_traverse_atr_mult": 1.20,
        "entry_score_margin": 0.02,
        "prefix": "vp",
    }

    def bot_start(self, **kwargs: object) -> None:
        self._validate_effective_paper_config()
        parent = getattr(super(), "bot_start", None)
        if callable(parent):
            parent(**kwargs)
        self._validate_effective_paper_config()

    def _validate_effective_paper_config(self) -> None:
        config = getattr(self, "config", {})
        if config.get("dry_run") is not True or config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError(f"{self.__class__.__name__} is PAPER-only and rejects live mode")

        roi = self.minimal_roi
        try:
            roi = {int(key): float(value) for key, value in roi.items()}
        except (AttributeError, TypeError, ValueError):
            pass
        expected = {
            "timeframe": "1h",
            "startup_candle_count": 336,
            "process_only_new_candles": True,
            "can_short": True,
            "minimal_roi": {0: 0.03},
            "stoploss": -0.02,
            "use_custom_roi": False,
            "use_exit_signal": False,
            "use_custom_stoploss": False,
            "position_adjustment_enable": False,
            "max_entry_position_adjustment": 0,
            "trailing_stop": False,
            "trailing_stop_positive": None,
            "trailing_stop_positive_offset": 0.0,
            "trailing_only_offset_is_reached": False,
            "exit_profit_only": False,
            "ignore_roi_if_entry_signal": False,
            "exit_profit_offset": 0.0,
            "ZONE_IMPULSE_WINDOW": 36,
            "ZONE_IMPULSE_ATR_MIN": 0.80,
            "ZONE_BODY_FRACTION_MIN": 0.55,
            "ZONE_VOLUME_RATIO_MIN": 1.10,
            "ZONE_MAX_AGE_BARS": 72,
            "ZONE_NEAR_PCT": 0.02,
            "RECLAIM_BUFFER_PCT": 0.02,
            "VP_GUARD_MODE": "score_or_context",
            "VP_SCORE_MIN": 0.25,
            "VP_CONTEXT_MIN": 0.20,
            "VP_SETTINGS": {
                "window": 96,
                "bins": 48,
                "value_area_pct": 0.70,
                "price_source": "hlc3",
                "smooth_bins": 3,
                "hvn_threshold": 0.70,
                "lvn_threshold": 0.35,
                "pressure_delta_min": 0.05,
                "node_near_pct": 0.010,
                "volume_percentile_min": 0.55,
                "score_window": 48,
                "fast_traverse_atr_mult": 1.20,
                "entry_score_margin": 0.02,
                "prefix": "vp",
            },
        }
        actual = {name: getattr(self, name, None) for name in expected}
        actual["minimal_roi"] = roi
        mismatches = [
            f"{name}={actual[name]!r} (expected {value!r})"
            for name, value in expected.items()
            if type(actual[name]) is not type(value) or actual[name] != value
        ]
        if mismatches:
            raise RuntimeError(
                f"{self.__class__.__name__} rejects overrides to frozen PAPER settings: "
                + "; ".join(mismatches)
            )

    @staticmethod
    def _num(frame: DataFrame, name: str) -> Series:
        if name not in frame:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _atr(cls, frame: DataFrame, period: int = 14) -> Series:
        high = cls._num(frame, "high")
        low = cls._num(frame, "low")
        close = cls._num(frame, "close")
        previous_close = close.shift(1)
        true_range = pd.concat(
            [high.sub(low), high.sub(previous_close).abs(), low.sub(previous_close).abs()],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(period, min_periods=max(2, period // 2)).mean()

    @staticmethod
    def _bars_since(signal: Series) -> Series:
        clean = pd.Series(signal).fillna(False).astype(bool)
        positions = pd.Series(np.arange(len(clean), dtype="float64"), index=clean.index)
        last_hit = positions.where(clean).ffill()
        return positions.sub(last_hit).fillna(9999.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        open_ = self._num(frame, "open")
        high = self._num(frame, "high")
        low = self._num(frame, "low")
        close = self._num(frame, "close")
        volume = self._num(frame, "volume").clip(lower=0.0)

        body = close.sub(open_).abs()
        candle_range = high.sub(low).replace(0.0, np.nan)
        body_fraction = body.div(candle_range)
        atr = self._atr(frame)
        volume_mean = volume.rolling(
            self.ZONE_IMPULSE_WINDOW,
            min_periods=max(2, self.ZONE_IMPULSE_WINDOW // 3),
        ).mean().replace(0.0, np.nan)
        volume_ratio = volume.div(volume_mean)
        quality = body_fraction.ge(self.ZONE_BODY_FRACTION_MIN) & volume_ratio.ge(
            self.ZONE_VOLUME_RATIO_MIN
        )
        bullish_impulse = (
            close.gt(open_)
            & close.sub(open_).ge(atr.mul(self.ZONE_IMPULSE_ATR_MIN))
            & quality
        )
        bearish_impulse = (
            close.lt(open_)
            & open_.sub(close).ge(atr.mul(self.ZONE_IMPULSE_ATR_MIN))
            & quality
        )
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        body_high = pd.concat([open_, close], axis=1).max(axis=1)

        # Zones/ages are shifted one completed candle, matching the saved Sieve1 source.
        frame["demand_zone_low"] = low.where(bullish_impulse).ffill().shift(1)
        frame["demand_zone_high"] = body_low.where(bullish_impulse).ffill().shift(1)
        frame["supply_zone_low"] = body_high.where(bearish_impulse).ffill().shift(1)
        frame["supply_zone_high"] = high.where(bearish_impulse).ffill().shift(1)
        frame["demand_zone_age"] = self._bars_since(bullish_impulse).shift(1)
        frame["supply_zone_age"] = self._bars_since(bearish_impulse).shift(1)

        return add_volume_profile(frame, **self.VP_SETTINGS)

    def _demand_reclaim(self, frame: DataFrame) -> Series:
        low = self._num(frame, "low")
        close = self._num(frame, "close")
        open_ = self._num(frame, "open")
        zone_high = self._num(frame, "demand_zone_high")
        age_ok = self._num(frame, "demand_zone_age").le(self.ZONE_MAX_AGE_BARS)
        return (
            age_ok
            & low.le(zone_high.mul(1.0 + self.ZONE_NEAR_PCT))
            & close.ge(zone_high.mul(1.0 + self.RECLAIM_BUFFER_PCT))
            & close.gt(open_)
        )

    def _supply_reject(self, frame: DataFrame) -> Series:
        high = self._num(frame, "high")
        close = self._num(frame, "close")
        open_ = self._num(frame, "open")
        zone_low = self._num(frame, "supply_zone_low")
        age_ok = self._num(frame, "supply_zone_age").le(self.ZONE_MAX_AGE_BARS)
        return (
            age_ok
            & high.ge(zone_low.mul(1.0 - self.ZONE_NEAR_PCT))
            & close.le(zone_low.mul(1.0 - self.RECLAIM_BUFFER_PCT))
            & close.lt(open_)
        )

    def _vp_guard(self, frame: DataFrame, side: str) -> Series:
        score = self._num(frame, f"vp_score_{side}")
        opposite_side = "short" if side == "long" else "long"
        opposite_score = self._num(frame, f"vp_score_{opposite_side}")
        score_ok = score.ge(self.VP_SCORE_MIN) & score.ge(opposite_score)

        context_name = "bull" if side == "long" else "bear"
        opposite_context_name = "bear" if side == "long" else "bull"
        context = self._num(frame, f"vp_context_score_{context_name}")
        opposite_context = self._num(frame, f"vp_context_score_{opposite_context_name}")
        market_context = self._num(frame, "vp_market_context")
        market_ok = market_context.ge(0) if side == "long" else market_context.le(0)
        context_ok = context.ge(self.VP_CONTEXT_MIN) & context.ge(opposite_context) & market_ok

        if self.VP_GUARD_MODE == "score":
            return score_ok
        if self.VP_GUARD_MODE == "context":
            return context_ok
        if self.VP_GUARD_MODE == "score_or_context":
            return score_ok | context_ok
        return pd.Series(False, index=frame.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        frame["enter_long"] = 0
        frame["enter_short"] = 0
        frame["enter_tag"] = None

        valid = self._num(frame, "volume").gt(0.0) & self._num(frame, "close").notna()
        long_condition = self._demand_reclaim(frame) & self._vp_guard(frame, "long") & valid
        short_condition = self._supply_reject(frame) & self._vp_guard(frame, "short") & valid
        long_condition = long_condition.fillna(False)
        short_condition = short_condition.fillna(False)
        conflict = long_condition & short_condition
        long_condition &= ~conflict
        short_condition &= ~conflict

        frame.loc[long_condition, "enter_long"] = 1
        frame.loc[long_condition, "enter_tag"] = "experimental_demand_vp_reclaim_long"
        frame.loc[short_condition, "enter_short"] = 1
        frame.loc[short_condition, "enter_tag"] = "sieve1_supply_vp_reject_short"
        return frame

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe
