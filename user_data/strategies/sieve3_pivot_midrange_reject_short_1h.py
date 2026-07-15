from __future__ import annotations

SIEVE_STAGE = "sieve3"
SOURCE_SIEVE2_STRATEGY = "sieve2_pivot_midrange_reject_short_1h"
SOURCE_RESULT_BATCH = "20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01"
SIEVE3_PROMOTION_NOTE = "Promoted from Sieve2 survivor: 2875 trades, 57.08% winrate, -10.28% profit, 19.82% max DD at TP/SL 3/3; high-winrate overtrader kept for Sieve3 guard tightening"
NOVEL_IDEA = True
SOURCE_STRATEGY = "novel"
SOURCE_RESULT_BATCH = "manual_20260612"
RESEARCH_PATH = "pivot_1h"
UPDATE_HYPOTHESIS = "Failing at the pivot-defined midpoint can mark a useful rejection short."

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi, entry_sieve_stoploss
from user_data.Indicators.pivot_foundation import build_clean_pivot_source

SIDE = "short"
ENTRY_MODE = "midrange_reject"
ENTRY_TAG = "pivot_midrange_reject_short_1h"


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", "mode:sieve2_pivot"))
    return param


def _safe_div(numer: Series, denom: Series) -> Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom


def _atr(frame: DataFrame, length: int = 14) -> Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=max(2, length // 2)).mean()


class Sieve3PivotMidrangeRejectShort1h(IStrategy):
    timeframe = "1h"
    can_short = True
    startup_candle_count = 360
    minimal_roi = entry_sieve_minimal_roi(0.03)
    stoploss = entry_sieve_stoploss(-0.03)
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    pivot_strength = tagged_parameter(IntParameter(2, 4, default=2, space="buy", optimize=True, load=True))
    min_prominence_atr = tagged_parameter(DecimalParameter(0.20, 1.20, default=0.45, decimals=2, space="buy", optimize=True, load=True))
    min_pivot_spacing_bars = tagged_parameter(IntParameter(1, 10, default=3, space="buy", optimize=True, load=True))
    min_pivot_distance_atr = tagged_parameter(DecimalParameter(0.00, 1.50, default=0.20, decimals=2, space="buy", optimize=True, load=True))
    reclaim_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.006, decimals=3, space="buy", optimize=True, load=True))
    score_min = tagged_parameter(DecimalParameter(0.20, 0.80, default=0.35, decimals=2, space="buy", optimize=True, load=True))
    range_atr_mult = tagged_parameter(DecimalParameter(0.50, 3.50, default=1.20, decimals=2, space="buy", optimize=True, load=True))
    freshness_max_bars = tagged_parameter(IntParameter(4, 80, default=24, space="buy", optimize=True, load=True))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        frame = dataframe.copy()
        frame["atr"] = _atr(frame, 14)
        frame["bar_index"] = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
        pivots = build_clean_pivot_source(
            body_high=pd.to_numeric(frame["high"], errors="coerce"),
            body_low=pd.to_numeric(frame["low"], errors="coerce"),
            atr=pd.to_numeric(frame["atr"], errors="coerce"),
            bar_index=frame["bar_index"],
            strength=int(self.pivot_strength.value),
            min_prominence_atr=float(self.min_prominence_atr.value),
            min_prominence_pct=0.0,
            min_pivot_spacing_bars=int(self.min_pivot_spacing_bars.value),
            min_pivot_distance_atr=float(self.min_pivot_distance_atr.value),
            min_pivot_distance_pct=0.0,
        )
        frame = frame.join(pd.DataFrame(pivots))
        frame["pivot_high_last"] = frame["pivot_high"].ffill()
        frame["pivot_low_last"] = frame["pivot_low"].ffill()
        frame["pivot_high_prev"] = frame["pivot_high"].where(frame["pivot_high"].notna()).shift(1).ffill()
        frame["pivot_low_prev"] = frame["pivot_low"].where(frame["pivot_low"].notna()).shift(1).ffill()
        frame["pivot_mid"] = (frame["pivot_high_last"] + frame["pivot_low_last"]) / 2.0
        frame["pivot_range"] = (frame["pivot_high_last"] - frame["pivot_low_last"]).abs()
        frame["pivot_age_high"] = (frame["bar_index"] - frame["pivot_high_available_index"].ffill()).fillna(999.0)
        frame["pivot_age_low"] = (frame["bar_index"] - frame["pivot_low_available_index"].ffill()).fillna(999.0)
        frame["volume_ratio"] = _safe_div(pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0), pd.to_numeric(frame["volume"], errors="coerce").rolling(20, min_periods=5).mean()).fillna(0.0)
        return frame

    def _entry_condition(self, frame: DataFrame) -> Series:
        close = pd.to_numeric(frame["close"], errors="coerce")
        open_ = pd.to_numeric(frame["open"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        atr = pd.to_numeric(frame["atr"], errors="coerce")
        pivot_high = pd.to_numeric(frame["pivot_high_last"], errors="coerce")
        pivot_low = pd.to_numeric(frame["pivot_low_last"], errors="coerce")
        pivot_high_prev = pd.to_numeric(frame["pivot_high_prev"], errors="coerce")
        pivot_low_prev = pd.to_numeric(frame["pivot_low_prev"], errors="coerce")
        pivot_mid = pd.to_numeric(frame["pivot_mid"], errors="coerce")
        pivot_range = pd.to_numeric(frame["pivot_range"], errors="coerce")
        volume_ratio = pd.to_numeric(frame["volume_ratio"], errors="coerce")
        reclaim = float(self.reclaim_buffer_pct.value)
        score_min = float(self.score_min.value)
        range_need = float(self.range_atr_mult.value)
        freshness_max = float(self.freshness_max_bars.value)
        pivot_high_score = pd.to_numeric(frame["pivot_high_score"], errors="coerce").ffill()
        pivot_low_score = pd.to_numeric(frame["pivot_low_score"], errors="coerce").ffill()
        pivot_high_prom = pd.to_numeric(frame["pivot_high_prominence"], errors="coerce").ffill()
        pivot_low_prom = pd.to_numeric(frame["pivot_low_prominence"], errors="coerce").ffill()
        age_high = pd.to_numeric(frame["pivot_age_high"], errors="coerce")
        age_low = pd.to_numeric(frame["pivot_age_low"], errors="coerce")

        if ENTRY_MODE == "reclaim_after_confirmation":
            return low.lt(pivot_low * (1.0 - reclaim)) & close.gt(pivot_low) & age_low.le(freshness_max) & pivot_low_score.ge(score_min)
        if ENTRY_MODE == "reject_after_confirmation":
            return high.gt(pivot_high * (1.0 + reclaim)) & close.lt(pivot_high) & age_high.le(freshness_max) & pivot_high_score.ge(score_min)
        if ENTRY_MODE == "prominence_breakout":
            return pivot_high_prom.ge(float(self.min_prominence_atr.value)) & close.gt(pivot_high * (1.0 + reclaim)) & volume_ratio.ge(1.0)
        if ENTRY_MODE == "prominence_breakdown":
            return pivot_low_prom.ge(float(self.min_prominence_atr.value)) & close.lt(pivot_low * (1.0 - reclaim)) & volume_ratio.ge(1.0)
        if ENTRY_MODE == "higher_low_sequence":
            return pivot_low.gt(pivot_low_prev * (1.0 + reclaim)) & close.gt(pivot_mid) & pivot_low_score.ge(score_min)
        if ENTRY_MODE == "lower_high_sequence":
            return pivot_high.lt(pivot_high_prev * (1.0 - reclaim)) & close.lt(pivot_mid) & pivot_high_score.ge(score_min)
        if ENTRY_MODE == "midrange_reclaim":
            return low.lt(pivot_mid * (1.0 - reclaim)) & close.gt(pivot_mid) & pivot_range.ge(atr * range_need)
        if ENTRY_MODE == "midrange_reject":
            return high.gt(pivot_mid * (1.0 + reclaim)) & close.lt(pivot_mid) & pivot_range.ge(atr * range_need)
        if ENTRY_MODE == "gap_expansion_long":
            return pivot_range.ge(atr * range_need) & close.gt(pivot_high * (1.0 + reclaim)) & age_low.le(freshness_max)
        if ENTRY_MODE == "gap_expansion_short":
            return pivot_range.ge(atr * range_need) & close.lt(pivot_low * (1.0 - reclaim)) & age_high.le(freshness_max)
        if ENTRY_MODE == "score_flip_long":
            return pivot_low_score.gt(pivot_high_score) & pivot_low_score.ge(score_min) & close.gt(pivot_low)
        if ENTRY_MODE == "score_flip_short":
            return pivot_high_score.gt(pivot_low_score) & pivot_high_score.ge(score_min) & close.lt(pivot_high)
        return pd.Series(False, index=frame.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        if SIDE == "long":
            dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)
        else:
            dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, ENTRY_TAG)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        return dataframe
