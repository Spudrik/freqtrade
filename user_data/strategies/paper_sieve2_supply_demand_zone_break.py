"""Paper-only Sieve2 supply-break long / demand-break short pairing.

The two source entries passed independently; their combination and the
side-by-side paper result have not been validated as profitable.

Selected rows: `20260521T012740_entry_all_resume.jsonl:807` (Sieve2 supply
breakout long, 3335 trades, TP/SL 2/2) and `:96` (Sieve2 demand breakdown
short, 2839 trades, TP/SL 3/2), validation `20200101-20260101`. Source
modules and parameter snapshots are the matching Sieve2 originals under
`Archive/sieve2_originals_failed/` and `D:\FreqTradeStuffLargeData\sieve_launcher_runtime\entry_sieve\params\`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy


class PaperSieve2SupplyDemandZoneBreak(IStrategy):
    """Frozen 1h signals from Sieve2; paired performance remains untested."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = True

    # Sieve2 fixed exits selected from the highest-trades/day qualifying rows.
    # Long TP/SL = 2/2; short TP/SL = 3/2. The static 3% ROI matches the short
    # target; custom_roi supplies the lower 2% long target. The native stop is 2%.
    minimal_roi = {"0": 0.03}
    stoploss = -0.02
    use_exit_signal = True
    use_custom_roi = True
    use_custom_stoploss = False
    position_adjustment_enable = False
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    exit_profit_offset = 0.0

    LONG_TAKE_PROFIT = 0.02
    SHORT_TAKE_PROFIT = 0.03

    # Source settings frozen from the selected Sieve2 parameter JSONs.
    ZONE_IMPULSE_WINDOW = 36
    ZONE_IMPULSE_ATR_MIN = 0.80
    ZONE_BODY_FRACTION_MIN = 0.55
    ZONE_VOLUME_RATIO_MIN = 1.10
    BREAKOUT_BUFFER_PCT = 0.012
    LONG_ZONE_MAX_AGE = 72
    SHORT_ZONE_MAX_AGE = 168
    LONG_VOLUME_WINDOW = 24
    SHORT_VOLUME_WINDOW = 12
    VOLUME_RATIO_MIN = 1.30
    LONG_PRESSURE_MIN = 0.10
    SHORT_PRESSURE_MIN = 0.35

    def bot_start(self, **kwargs: object) -> None:
        config = getattr(self, "config", {})
        if config.get("dry_run") is not True or config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError(f"{self.__class__.__name__} is PAPER-only and rejects live mode")
        self._validate_effective_settings()
        parent = getattr(super(), "bot_start", None)
        if callable(parent):
            parent(**kwargs)
        if config.get("dry_run") is not True or config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError(f"{self.__class__.__name__} is PAPER-only and rejects live mode")
        self._validate_effective_settings()

    def _validate_effective_settings(self) -> None:
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
            "use_custom_roi": True,
            "use_exit_signal": True,
            "use_custom_stoploss": False,
            "position_adjustment_enable": False,
            "trailing_stop": False,
            "trailing_stop_positive": None,
            "trailing_stop_positive_offset": 0.0,
            "trailing_only_offset_is_reached": False,
            "exit_profit_only": False,
            "ignore_roi_if_entry_signal": False,
            "exit_profit_offset": 0.0,
        }
        actual = {name: getattr(self, name, None) for name in expected}
        actual["minimal_roi"] = roi
        mismatches = [
            f"{name}={actual[name]!r} (expected {value!r})"
            for name, value in expected.items()
            if actual[name] != value
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
        impulse_volume_mean = volume.rolling(
            self.ZONE_IMPULSE_WINDOW,
            min_periods=max(2, self.ZONE_IMPULSE_WINDOW // 3),
        ).mean().replace(0.0, np.nan)
        impulse_volume_ratio = volume.div(impulse_volume_mean)
        quality = body_fraction.ge(self.ZONE_BODY_FRACTION_MIN) & impulse_volume_ratio.ge(
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

        # Shift completed impulse zones one bar, matching the saved Sieve2 source.
        frame["demand_zone_low"] = low.where(bullish_impulse).ffill().shift(1)
        frame["supply_zone_high"] = high.where(bearish_impulse).ffill().shift(1)
        frame["demand_zone_age"] = self._bars_since(bullish_impulse).shift(1)
        frame["supply_zone_age"] = self._bars_since(bearish_impulse).shift(1)

        frame["paper_long_volume_ratio"] = volume.div(
            volume.rolling(
                self.LONG_VOLUME_WINDOW,
                min_periods=max(2, self.LONG_VOLUME_WINDOW // 3),
            ).mean().replace(0.0, np.nan)
        )
        frame["paper_short_volume_ratio"] = volume.div(
            volume.rolling(
                self.SHORT_VOLUME_WINDOW,
                min_periods=max(2, self.SHORT_VOLUME_WINDOW // 3),
            ).mean().replace(0.0, np.nan)
        )
        frame["paper_entry_pressure"] = close.sub(low).div(candle_range).clip(0.0, 1.0).sub(0.5).mul(2.0)
        return frame

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        frame["enter_long"] = 0
        frame["enter_short"] = 0
        frame["enter_tag"] = None

        close = self._num(frame, "close")
        volume = self._num(frame, "volume")
        supply_trigger = self._num(frame, "supply_zone_high").mul(1.0 + self.BREAKOUT_BUFFER_PCT)
        demand_trigger = self._num(frame, "demand_zone_low").mul(1.0 - self.BREAKOUT_BUFFER_PCT)
        long_condition = (
            self._num(frame, "supply_zone_age").le(self.LONG_ZONE_MAX_AGE)
            & close.ge(supply_trigger)
            & close.shift(1).lt(supply_trigger.shift(1))
            & self._num(frame, "paper_long_volume_ratio").ge(self.VOLUME_RATIO_MIN)
            & self._num(frame, "paper_entry_pressure").ge(self.LONG_PRESSURE_MIN)
        )
        short_condition = (
            self._num(frame, "demand_zone_age").le(self.SHORT_ZONE_MAX_AGE)
            & close.le(demand_trigger)
            & close.shift(1).gt(demand_trigger.shift(1))
            & self._num(frame, "paper_short_volume_ratio").ge(self.VOLUME_RATIO_MIN)
            & self._num(frame, "paper_entry_pressure").le(-self.SHORT_PRESSURE_MIN)
        )

        valid = volume.gt(0.0) & close.notna()
        long_condition = long_condition.fillna(False) & valid
        short_condition = short_condition.fillna(False) & valid
        conflict = long_condition & short_condition
        long_condition &= ~conflict
        short_condition &= ~conflict

        frame.loc[long_condition, "enter_long"] = 1
        frame.loc[long_condition, "enter_tag"] = "sieve2_supply_zone_breakout_long"
        frame.loc[short_condition, "enter_short"] = 1
        frame.loc[short_condition, "enter_tag"] = "sieve2_demand_zone_breakdown_short"
        return frame

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def custom_roi(
        self,
        pair: str,
        trade: object,
        current_time: object,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs: object,
    ) -> float | None:
        _ = pair, trade, current_time, trade_duration, entry_tag, kwargs
        if side == "long":
            return self.LONG_TAKE_PROFIT
        if side == "short":
            return self.SHORT_TAKE_PROFIT
        return None
