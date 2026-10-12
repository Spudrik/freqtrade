"""Paper-only Sieve2 prior-day/volume-profile breakout pair.

The saved VP-confluent long passed the screen. The mirrored prior-day-low
short is experimental; pairing both sides is not validated as profitable.

Selected source row: `20260521T012740_entry_all_resume.jsonl:770`,
Sieve2Multi2VpPriorDayHighBreakVpValLong, 2931 trades, TP/SL 3/2,
validation `20200101-20260101`; its source and parameter snapshot are the
matching originals under `Archive/sieve2_originals_failed/` and
`D:\FreqTradeStuffLargeData\sieve_launcher_runtime\entry_sieve\params\`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy

from user_data.Indicators.complex_volume_profile import add_volume_profile


class PaperSieve2PriorDayVpBreak(IStrategy):
    """Frozen 1h prior-day breakout plus the exact bearish VP mirror."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = True

    # Saved tested long TP/SL = 3/2; the untested mirrored short uses the same
    # fixed 3/2 thresholds so the pair has one explicit exit contract.
    minimal_roi = {"0": 0.03}
    stoploss = -0.02
    use_exit_signal = False
    use_custom_roi = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    exit_profit_offset = 0.0

    # Exact selected Sieve2 values, with source defaults made explicit.
    PRIOR_LEVEL_BUFFER_PCT = 0.006
    VP_CONFIRM_BUFFER_PCT = 0.010
    VP_SCORE_MIN = 0.15
    VP_CONTEXT_MIN = 0.45
    VOLUME_GUARD_WINDOW = 12
    VOLUME_RATIO_MIN = 1.60
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
            "startup_candle_count": 220,
            "process_only_new_candles": True,
            "can_short": True,
            "minimal_roi": {0: 0.03},
            "stoploss": -0.02,
            "use_custom_roi": False,
            "use_exit_signal": False,
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

    @staticmethod
    def _dates(frame: DataFrame) -> Series:
        if "date" in frame:
            return pd.to_datetime(frame["date"], utc=True, errors="coerce")
        return pd.Series(pd.to_datetime(frame.index, utc=True, errors="coerce"), index=frame.index)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        dates = self._dates(frame)
        period_key = dates.dt.strftime("%Y-%m-%d")
        high = self._num(frame, "high")
        low = self._num(frame, "low")
        prior = pd.DataFrame({"period": period_key, "high": high, "low": low}, index=frame.index)
        levels = prior.groupby("period", sort=False).agg(period_high=("high", "max"), period_low=("low", "min"))
        levels["prior_day_high"] = levels["period_high"].shift(1)
        levels["prior_day_low"] = levels["period_low"].shift(1)
        frame["prior_day_high"] = period_key.map(levels["prior_day_high"])
        frame["prior_day_low"] = period_key.map(levels["prior_day_low"])

        frame = add_volume_profile(frame, **self.VP_SETTINGS)
        volume = self._num(frame, "volume").clip(lower=0.0)
        baseline = volume.shift(1).rolling(
            self.VOLUME_GUARD_WINDOW,
            min_periods=max(2, self.VOLUME_GUARD_WINDOW // 3),
        ).mean().replace(0.0, np.nan)
        frame["paper_volume_guard"] = volume.ge(baseline.mul(self.VOLUME_RATIO_MIN))
        return frame

    def _prior_day_long(self, frame: DataFrame) -> Series:
        close = self._num(frame, "close")
        open_ = self._num(frame, "open")
        trigger = self._num(frame, "prior_day_high").mul(1.0 + self.PRIOR_LEVEL_BUFFER_PCT)
        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1)) & close.gt(open_)

    def _prior_day_short(self, frame: DataFrame) -> Series:
        close = self._num(frame, "close")
        open_ = self._num(frame, "open")
        trigger = self._num(frame, "prior_day_low").mul(1.0 - self.PRIOR_LEVEL_BUFFER_PCT)
        return close.le(trigger) & close.shift(1).gt(trigger.shift(1)) & close.lt(open_)

    def _vp_val_reclaim(self, frame: DataFrame) -> Series:
        close = self._num(frame, "close")
        score_long = self._num(frame, "vp_score_long")
        balance = self._num(frame, "vp_context_score_balance")
        trigger = frame.get("vp_entry_trigger_long", pd.Series(False, index=frame.index))
        return close.ge(self._num(frame, "vp_val").mul(1.0 - self.VP_CONFIRM_BUFFER_PCT)) & (
            score_long.ge(self.VP_SCORE_MIN)
            | pd.Series(trigger, index=frame.index).fillna(False).astype(bool)
            | balance.ge(self.VP_CONTEXT_MIN)
        )

    def _vp_vah_reject(self, frame: DataFrame) -> Series:
        close = self._num(frame, "close")
        score_short = self._num(frame, "vp_score_short")
        balance = self._num(frame, "vp_context_score_balance")
        trigger = frame.get("vp_entry_trigger_short", pd.Series(False, index=frame.index))
        return close.le(self._num(frame, "vp_vah").mul(1.0 + self.VP_CONFIRM_BUFFER_PCT)) & (
            score_short.ge(self.VP_SCORE_MIN)
            | pd.Series(trigger, index=frame.index).fillna(False).astype(bool)
            | balance.ge(self.VP_CONTEXT_MIN)
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        frame = dataframe.copy()
        frame["enter_long"] = 0
        frame["enter_short"] = 0
        frame["enter_tag"] = None
        close = self._num(frame, "close")
        valid = self._num(frame, "volume").gt(0.0) & close.notna() & frame["paper_volume_guard"].fillna(False)
        long_condition = self._prior_day_long(frame) & self._vp_val_reclaim(frame) & valid
        short_condition = self._prior_day_short(frame) & self._vp_vah_reject(frame) & valid
        conflict = long_condition.fillna(False) & short_condition.fillna(False)
        long_condition = long_condition.fillna(False) & ~conflict
        short_condition = short_condition.fillna(False) & ~conflict
        frame.loc[long_condition, "enter_long"] = 1
        frame.loc[long_condition, "enter_tag"] = "sieve2_vp_prior_day_high_break_long"
        frame.loc[short_condition, "enter_short"] = 1
        frame.loc[short_condition, "enter_tag"] = "experimental_prior_day_low_vp_vah_break_short"
        return frame

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe
